#include "engine/PersistentCache.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace splash::engine {
namespace {
constexpr uint64_t metadataVersion = 1;
constexpr size_t blockWords = KvCache::pageTokens + 3;
struct Decoded {
  uint64_t stateId;
  std::vector<uint64_t> state;
  std::span<const uint64_t> blocks;
};
Decoded decode(const model::CacheStore::Prefix &prefix, uint64_t kvBytes,
               uint64_t stateBytes) {
  const auto &data = prefix.metadata;
  if (data.size() < 5 || data[0] != metadataVersion || data[1] != prefix.id ||
      data[2] > 64 || data.size() < data[2] + 4)
    throw std::runtime_error("invalid persistent prefix metadata");
  const size_t countAt = 3 + data[2];
  const uint64_t count = data[countAt];
  if (!count || count > UINT32_MAX / KvCache::pageTokens ||
      count != (data.size() - countAt - 1) / blockWords ||
      (data.size() - countAt - 1) % blockWords)
    throw std::runtime_error("invalid persistent prefix length");
  std::map<uint64_t, uint64_t> records;
  for (auto record : prefix.records)
    records.emplace(record.id, record.bytes);
  if (records.size() != count + 1 || records[data[1]] != stateBytes)
    throw std::runtime_error("incomplete persistent state");
  const auto blocks = std::span(data).subspan(countAt + 1);
  std::set<uint64_t> distinct;
  for (size_t offset = 0; offset < blocks.size(); offset += blockWords) {
    const uint64_t id = blocks[offset];
    if (!id || id == data[1] || records[id] != kvBytes ||
        !distinct.insert(id).second)
      throw std::runtime_error("incomplete persistent KV");
    for (size_t token = 3; token < blockWords; ++token)
      if (blocks[offset + token] > UINT32_MAX)
        throw std::runtime_error("invalid persistent token");
  }
  return {data[1], {data.begin() + 3, data.begin() + countAt}, blocks};
}
} // namespace

PersistentCache::PersistentCache(PersistentCacheConfig config, KvCache &kv,
                                 StateCache &states, model::KvTier &tier,
                                 CacheRecency &recency,
                                 std::function<bool()> makeRoom,
                                 const std::function<void()> &completion)
    : config_(std::move(config)), kv_(kv), states_(states), tier_(tier),
      recency_(recency), makeRoom_(std::move(makeRoom)),
      completion_(completion) {
  if (!config_.capacityBytes || !config_.store || !config_.kvFile ||
      !config_.stateFile || !config_.stateStorage)
    throw std::invalid_argument("incomplete persistent cache configuration");
  if (!config_.writeBurstBytes)
    config_.writeBurstBytes = config_.capacityBytes;
  writeCredit_ = static_cast<double>(config_.writeBurstBytes);
  load();
}
PersistentCache::~PersistentCache() {
  finish();
  for (auto &[_, entry] : entries_)
    for (auto &record : entry.records)
      record->releaseDurable();
  // Drain bounded metadata work too, so graceful close retains exactly the
  // committed index. Payloads with no published prefix remain disposable.
  try {
    if (touches_)
      static_cast<void>(touches_->wait());
    static_cast<void>(flushTouches());
    static_cast<void>(config_.stateFile->metadata([] {})->wait());
  } catch (...) {
  }
}

void PersistentCache::load() {
  auto prefixes = config_.store->load();
  // Select newest complete prefixes before opening slots, so a smaller
  // quota never needs to materialize the old working set or allocate GPU RAM.
  std::set<uint64_t> kept;
  uint64_t bytes = 0;
  std::vector<model::CacheStore::Prefix> selected;
  for (auto it = prefixes.rbegin(); it != prefixes.rend(); ++it) {
    bool valid = true;
    try {
      static_cast<void>(decode(*it, config_.kvFile->slotBytes(),
                               config_.stateFile->slotBytes()));
    } catch (const std::exception &) {
      valid = false;
      ++failures_;
    }
    uint64_t extra = 0;
    for (auto record : it->records)
      if (!kept.contains(record.id)) {
        if (record.bytes >
            config_.capacityBytes - std::min(extra, config_.capacityBytes)) {
          valid = false;
          break;
        }
        extra += record.bytes;
      }
    if (!valid || extra > config_.capacityBytes - bytes) {
      config_.store->erase(it->id);
      continue;
    }
    bytes += extra;
    for (auto record : it->records)
      kept.insert(record.id);
    selected.push_back(std::move(*it));
  }
  config_.store->collectUnreferenced();
  for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
    const auto decoded = decode(*it, config_.kvFile->slotBytes(),
                                config_.stateFile->slotBytes());
    auto stateSlot = config_.stateFile->reopen(decoded.stateId);
    const uint64_t boundary =
        decoded.blocks.size() / blockWords * KvCache::pageTokens;
    std::shared_ptr<const CompositeState> state;
    try {
      state = config_.stateStorage->reopenState({stateSlot, decoded.state},
                                                boundary);
    } catch (const std::invalid_argument &) {
      ++failures_;
      config_.store->erase(it->id);
      continue;
    }
    if (!state)
      throw std::runtime_error("model cannot reopen a persistent state");
    Entry entry{it->id, 0, recency_.next(), {std::move(stateSlot)}};
    for (size_t offset = 0; offset < decoded.blocks.size();
         offset += blockWords) {
      auto slot = config_.kvFile->reopen(decoded.blocks[offset]);
      if (!slot)
        throw std::runtime_error("persistent cache exceeds shared disk quota");
      std::array<uint32_t, KvCache::pageTokens> tokens;
      std::copy_n(decoded.blocks.begin() + offset + 3, tokens.size(),
                  tokens.begin());
      entry.block = kv_.importDisk(
          entry.block, tokens,
          {decoded.blocks[offset + 1], decoded.blocks[offset + 2]},
          tier_.reopenSlot(slot));
      entry.records.push_back(std::move(slot));
    }
    states_.importDisk(entry.block, std::move(state));
    remember(std::move(entry));
    ++restored_;
  }
}

void PersistentCache::publish(uint64_t block, bool reused) {
  if (states_.checkpoint(block))
    return;
  if (entries_.contains(block)) {
    touch(block);
    return;
  }
  if (job_ && job_->state.kvBlock() == block)
    return;
  for (auto &candidate : pending_)
    if (candidate.block == block) {
      candidate.reused |= reused;
      return;
    }
  // Candidates contain IDs only; they neither pin memory nor grow with the
  // number of tokens. A burst keeps the most recent reusable boundaries.
  if (pending_.size() == 16)
    pending_.pop_front();
  pending_.push_back({block, reused});
}
void PersistentCache::touch(uint64_t block) {
  auto found = entries_.find(block);
  if (found == entries_.end())
    return;
  found->second.used = recency_.next();
  found->second.touched = true;
}

bool PersistentCache::flushTouches() {
  if (touches_ && !touches_->ready())
    return false;
  const bool finished = touches_ != nullptr;
  if (touches_ && !touches_->wait())
    ++failures_;
  touches_.reset();
  std::vector<std::pair<uint64_t, uint64_t>> touched;
  for (auto &[_, entry] : entries_)
    if (entry.touched) {
      touched.emplace_back(entry.used, entry.id);
      entry.touched = false;
    }
  if (touched.empty())
    return finished;
  std::sort(touched.begin(), touched.end());
  touches_ = config_.stateFile->metadata(
      [store = config_.store, touched] {
        for (auto [_, id] : touched)
          store->touch(id);
      },
      completion_);
  return true;
}

void PersistentCache::invalidate(uint64_t block) {
  if (job_ && std::find(job_->blocks.begin(), job_->blocks.end(), block) !=
                  job_->blocks.end())
    job_->failed = true;
  const auto disk = kv_.contains(block) ? kv_.slot(block) : nullptr;
  const auto record = disk ? disk->record() : nullptr;
  for (auto it = entries_.begin(); it != entries_.end();) {
    auto current = it++;
    if (current->first == block ||
        (record && std::find(current->second.records.begin(),
                             current->second.records.end(),
                             record) != current->second.records.end()))
      forget(current);
  }
}

void PersistentCache::start(uint64_t block, bool reused) {
  if (!kv_.contains(block) || !states_.contains(block))
    return;
  auto chain = kv_.chain(block);
  // Recurrent/draft snapshots have a large fixed cost even for tiny prompts.
  // Keep short prefixes in the existing RAM/temporary tiers; admitting them
  // here would spend writes and displace expensive, reusable continuations.
  if (chain.blocks.size() * uint64_t{KvCache::pageTokens} <
      config_.minimumTokens)
    return;
  // Every candidate must fit in the durable quota by itself. No speculative
  // eviction of useful prefixes for a candidate that can never be committed.
  const uint64_t pageBytes = config_.kvFile->slotBytes();
  if (config_.stateFile->slotBytes() > config_.capacityBytes ||
      chain.blocks.size() >
          (config_.capacityBytes - config_.stateFile->slotBytes()) / pageBytes)
    return;
  uint64_t extra = config_.stateFile->slotBytes();
  uint64_t writes = states_.diskCopy(block) ? 0 : extra;
  for (uint64_t id : chain.blocks) {
    const auto slot = kv_.slot(id);
    const auto record = slot ? slot->record() : nullptr;
    if (!record || !references_.contains(record->recordId()))
      extra += pageBytes;
    if (!slot)
      writes += pageBytes;
  }
  // Under pressure, an unseen tail must prove useful through a real hit.
  // Existing branch/junction states already represent an observed prefix.
  // Free space still admits first-use sessions for immediate crash recovery.
  if (!reused && !kv_.stateBelow(block) &&
      extra > config_.capacityBytes - used_) {
    ++admissionSkips_;
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  const double burst = static_cast<double>(config_.writeBurstBytes);
  writeCredit_ = std::min(
      burst,
      writeCredit_ + std::chrono::duration<double>(now - writeRefill_).count() *
                         config_.writeBytesPerSecond);
  writeRefill_ = now;
  if (writeCredit_ < writes) {
    ++writeThrottles_;
    return;
  }
  auto lease =
      states_.acquireDeepest(std::span(&block, 1), CacheAccess::Maintenance);
  if (!lease)
    return;
  writeCredit_ -= writes;
  auto job = std::make_unique<Job>(
      Job{std::move(*lease), std::move(chain.blocks), {}, {}, {}, false});
  for (uint64_t id : job->blocks)
    kv_.retainActive(id, CacheAccess::Maintenance);
  job_ = std::move(job);
}

bool PersistentCache::poll() {
  bool progress = flushTouches();
  if (!job_ && !pending_.empty()) {
    const auto candidate = pending_.front();
    pending_.pop_front();
    start(candidate.block, candidate.reused);
    progress = true;
  }
  if (!job_)
    return progress;
  Job &job = *job_;
  if (job.commit) {
    if (!job.commit->ready())
      return progress;
    if (job.commit->wait() && !job.failed) {
      remember(std::move(*job.entry));
      ++saved_;
    } else {
      const uint64_t id = job.entry->id;
      static_cast<void>(config_.stateFile->metadata(
          [store = config_.store, id] { store->erase(id); }));
      ++failures_;
    }
    finish();
    return true;
  }
  for (auto it = job.copies.begin(); it != job.copies.end();) {
    if (!it->transfer->ready()) {
      ++it;
      continue;
    }
    const bool written = it->transfer->finish();
    kv_.setTransferring(it->block, false);
    if (!written) {
      kv_.setSlot(it->block, nullptr);
      job.failed = true;
    }
    it = job.copies.erase(it);
    progress = true;
  }
  const uint64_t block = job.state.kvBlock();
  if (!job.failed && !states_.writing(block) && !states_.diskCopy(block)) {
    if (!states_.copyToDisk(block, completion_, makeRoom_)) {
      if (states_.writing())
        return progress;
      job.failed = true;
    } else
      progress = true;
  }
  if (!job.failed) {
    for (uint64_t id : job.blocks) {
      if (kv_.transferring(id) || kv_.slot(id))
        continue;
      if (kv_.page(id) == KvCache::noPage || !tier_.writable()) {
        job.failed = true;
        break;
      }
      if (!tier_.canDemote())
        break;
      auto slot = tier_.acquireSlot();
      while (!slot && makeRoom_())
        slot = tier_.acquireSlot();
      if (!slot) {
        job.failed = true;
        break;
      }
      auto transfer = tier_.demote(kv_.page(id), slot, completion_);
      if (!transfer)
        break;
      kv_.setSlot(id, std::move(slot));
      kv_.setTransferring(id, true);
      job.copies.push_back({id, std::move(transfer)});
      progress = true;
    }
  }
  if (job.failed) {
    if (!job.copies.empty() || states_.writing(block))
      return progress;
    ++failures_;
    finish();
    return true;
  }
  const auto state = states_.diskCopy(block);
  if (!state || !job.copies.empty())
    return progress;
  for (uint64_t id : job.blocks)
    if (!kv_.slot(id) || kv_.transferring(id))
      return progress;
  const auto record = state->diskRecord();
  if (!record.slot) {
    ++failures_;
    finish();
    return true;
  }
  Entry entry{record.slot->recordId(), block, recency_.next(), {record.slot}};
  const auto prefix = describe(entry, record, job.blocks);
  while (additionalBytes(entry) > config_.capacityBytes - used_)
    if (!evictOldest()) {
      ++failures_;
      finish();
      return true;
    }
  job.entry = std::move(entry);
  job.commit = config_.stateFile->metadata(
      [store = config_.store, prefix] { store->save(prefix); }, completion_);
  return true;
}

model::CacheStore::Prefix
PersistentCache::describe(Entry &entry, const DiskStateRecord &state,
                          std::span<const uint64_t> blocks) const {
  model::CacheStore::Prefix prefix{
      entry.id, {metadataVersion, entry.id, state.metadata.size()}, {}};
  prefix.metadata.insert(prefix.metadata.end(), state.metadata.begin(),
                         state.metadata.end());
  prefix.metadata.push_back(blocks.size());
  prefix.records.push_back({entry.id, config_.stateFile->slotBytes()});
  for (uint64_t block : blocks) {
    const auto slot = kv_.slot(block)->record();
    if (!slot || !slot->recordId())
      throw std::logic_error("nonpersistent KV slot in persistent cache");
    const auto key = kv_.key(block);
    prefix.metadata.insert(prefix.metadata.end(),
                           {slot->recordId(), key.images.lo, key.images.hi});
    prefix.metadata.insert(prefix.metadata.end(), key.tokens.begin(),
                           key.tokens.end());
    prefix.records.push_back({slot->recordId(), config_.kvFile->slotBytes()});
    entry.records.push_back(slot);
  }
  return prefix;
}

uint64_t PersistentCache::additionalBytes(const Entry &entry) const {
  uint64_t extra = 0;
  for (size_t i = 0; i < entry.records.size(); ++i)
    if (!references_.contains(entry.records[i]->recordId()))
      extra +=
          i == 0 ? config_.stateFile->slotBytes() : config_.kvFile->slotBytes();
  return extra;
}
void PersistentCache::remember(Entry entry) {
  if (auto old = entries_.find(entry.block); old != entries_.end())
    forget(old);
  for (size_t i = 0; i < entry.records.size(); ++i) {
    entry.records[i]->retainDurable();
    auto &[count, bytes] = references_[entry.records[i]->recordId()];
    if (!count++) {
      bytes =
          i == 0 ? config_.stateFile->slotBytes() : config_.kvFile->slotBytes();
      used_ += bytes;
    }
  }
  entries_.emplace(entry.block, std::move(entry));
}
void PersistentCache::forget(std::map<uint64_t, Entry>::iterator entry) {
  const uint64_t id = entry->second.id;
  static_cast<void>(config_.stateFile->metadata(
      [store = config_.store, id] { store->erase(id); }));
  for (auto record : entry->second.records) {
    record->releaseDurable();
    auto found = references_.find(record->recordId());
    if (!--found->second.count) {
      used_ -= found->second.bytes;
      references_.erase(found);
    }
  }
  entries_.erase(entry);
}
std::optional<CacheEvictionCandidate>
PersistentCache::evictionCandidate() const noexcept {
  if (entries_.empty())
    return std::nullopt;
  auto oldest = std::min_element(entries_.begin(), entries_.end(),
                                 [](const auto &a, const auto &b) {
                                   return a.second.used < b.second.used;
                                 });
  return CacheEvictionCandidate{oldest->first, oldest->second.used};
}

bool PersistentCache::evictOldest() {
  const auto oldest = evictionCandidate();
  if (!oldest)
    return false;
  forget(entries_.find(oldest->id));
  return true;
}
void PersistentCache::finish() {
  if (!job_)
    return;
  // Normal completion reaches here only after copies finish. On shutdown the
  // tier owns and drains staging IO; unpublished slots remain disposable.
  job_->copies.clear();
  if (job_->commit)
    static_cast<void>(job_->commit->wait());
  for (uint64_t block : job_->blocks)
    kv_.releaseActive(block, CacheAccess::Maintenance);
  job_.reset();
}
PersistentCacheSnapshot PersistentCache::snapshot() const {
  return {config_.capacityBytes,
          used_,
          saved_,
          restored_,
          failures_,
          static_cast<uint32_t>(entries_.size()),
          job_ != nullptr || !pending_.empty(),
          admissionSkips_,
          writeThrottles_};
}
} // namespace splash::engine
