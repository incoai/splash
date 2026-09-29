#include "engine/PersistentCache.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace splash::engine {
namespace {
constexpr uint64_t metadataVersion = 3;
constexpr size_t blockWords = KvCache::pageTokens + 3;
struct DecodedState {
  CachedStateBlock block;
  std::vector<uint64_t> metadata;
  std::vector<model::CacheStore::Record> records;
};
struct Decoded {
  std::vector<DecodedState> states;
  std::span<const uint64_t> blocks;
};
Decoded decode(const model::CacheStore::Prefix &prefix, uint64_t kvBytes,
               std::span<const CacheGroupSpec> groups) {
  const auto &data = prefix.metadata;
  size_t cursor = 0;
  const auto next = [&]() -> uint64_t {
    if (cursor == data.size())
      throw std::runtime_error("truncated persistent manifest");
    return data[cursor++];
  };
  if (next() != metadataVersion || next() != prefix.id)
    throw std::runtime_error("invalid persistent manifest version or identity");
  const uint64_t stateCount = next();
  if (stateCount > (data.size() - cursor) / 7)
    throw std::runtime_error("invalid persistent group count");
  std::map<uint64_t, uint64_t> records, used;
  for (auto record : prefix.records)
    if (!record.id || !record.bytes ||
        !records.emplace(record.id, record.bytes).second)
      throw std::runtime_error("invalid persistent record directory");
  Decoded result;
  std::map<std::pair<CacheGroupId, uint32_t>, size_t> stateIndex;
  std::map<uint64_t, CacheGroupId> recordGroups;
  for (uint64_t i = 0; i < stateCount; ++i) {
    const auto group = next(), begin = next(), end = next();
    const auto metadataCount = next(), recordCount = next();
    if (group > UINT32_MAX || begin > end || end > UINT32_MAX ||
        begin % KvCache::pageTokens || end % KvCache::pageTokens ||
        metadataCount > data.size() - cursor || !recordCount ||
        recordCount > (data.size() - cursor - metadataCount) / 2)
      throw std::runtime_error("invalid persistent group block");
    DecodedState state{{static_cast<CacheGroupId>(group),
                        static_cast<uint32_t>(begin),
                        static_cast<uint32_t>(end),
                        {}},
                       {},
                       {}};
    if (!stateIndex
             .emplace(std::pair{state.block.group, state.block.end},
                      result.states.size())
             .second)
      throw std::runtime_error("duplicate persistent group endpoint");
    state.metadata.assign(data.begin() + cursor,
                          data.begin() + cursor + metadataCount);
    cursor += metadataCount;
    std::set<uint64_t> distinct;
    for (uint64_t record = 0; record < recordCount; ++record) {
      const auto id = next(), bytes = next();
      const auto found = records.find(id);
      if (found == records.end() || found->second != bytes ||
          !distinct.insert(id).second)
        throw std::runtime_error("incomplete persistent group payload");
      const auto [owner, added] = recordGroups.emplace(id, state.block.group);
      if (!added && owner->second != state.block.group)
        throw std::runtime_error("persistent payload aliases different groups");
      state.records.push_back({id, bytes});
      used.emplace(id, bytes);
    }
    result.states.push_back(std::move(state));
  }
  const uint64_t count = next();
  if (!count || count > UINT32_MAX / KvCache::pageTokens ||
      count != (data.size() - cursor) / blockWords ||
      (data.size() - cursor) % blockWords)
    throw std::runtime_error("invalid persistent prefix length");
  result.blocks = std::span(data).subspan(cursor);
  for (size_t offset = 0; offset < result.blocks.size(); offset += blockWords) {
    const auto id = result.blocks[offset];
    const auto found = records.find(id);
    if (found == records.end() || found->second != kvBytes ||
        !used.emplace(id, kvBytes).second)
      throw std::runtime_error("incomplete persistent target KV");
    for (size_t token = 3; token < blockWords; ++token)
      if (result.blocks[offset + token] > UINT32_MAX)
        throw std::runtime_error("invalid persistent token");
  }
  if (used != records || result.blocks[(count - 1) * blockWords] != prefix.id)
    throw std::runtime_error("persistent manifest record directory mismatch");
  const uint32_t boundary = count * KvCache::pageTokens;
  CacheGroupCoordinator coordinator(
      std::vector<CacheGroupSpec>(groups.begin(), groups.end()));
  for (const auto &state : result.states) {
    const auto spec =
        std::find_if(groups.begin(), groups.end(),
                     [&](const auto &g) { return g.id == state.block.group; });
    if (spec == groups.end() || state.block.end > boundary ||
        (spec->kind == CacheGroupKind::Checkpoint
             ? state.block.begin != boundary || state.block.end != boundary
             : state.block.begin >= state.block.end))
      throw std::runtime_error(
          "persistent group does not match model declaration");
  }
  if (!coordinator.complete(
          boundary,
          [&](CacheGroupId group,
              uint32_t end) -> std::optional<CachedStateBlock> {
            const auto found = stateIndex.find({group, end});
            return found == stateIndex.end()
                       ? std::nullopt
                       : std::optional(result.states[found->second].block);
          }))
    throw std::runtime_error(
        "persistent manifest has incomplete group coverage");
  return result;
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
      !config_.stateStorage)
    throw std::invalid_argument("incomplete persistent cache configuration");
  if (!config_.writeBurstBytes)
    config_.writeBurstBytes = config_.capacityBytes;
  writeCredit_ = static_cast<double>(config_.writeBurstBytes);
  try {
    load();
  } catch (...) {
    for (auto &[_, entry] : entries_)
      for (auto &record : entry.records)
        record->releaseDurable();
    throw;
  }
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
    static_cast<void>(config_.kvFile->metadata([] {})->wait());
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
                               config_.stateStorage->cacheGroups()));
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
                                config_.stateStorage->cacheGroups());
    const uint32_t boundary =
        decoded.blocks.size() / blockWords * KvCache::pageTokens;
    auto state = std::make_shared<RestoreState>();
    state->boundary = boundary;
    Entry entry{it->id, 0, recency_.next(), {}};
    std::set<uint64_t> retained;
    try {
      for (const auto &part : decoded.states) {
        auto block = part.block;
        block.payload = config_.stateStorage->reopenState(
            block, {part.metadata, part.records});
        if (!block.payload)
          throw std::invalid_argument("model cannot reopen cache group");
        auto record = block.payload->diskRecord();
        record.components.push_back(record.slot);
        for (const auto &slot : record.components) {
          if (!slot)
            throw std::invalid_argument("model reopened incomplete payload");
          if (retained.insert(slot->recordId()).second)
            entry.records.push_back(slot);
        }
        state->blocks.push_back(std::move(block));
      }
    } catch (const std::invalid_argument &) {
      ++failures_;
      config_.store->erase(it->id);
      continue;
    }
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

void PersistentCache::observe(uint64_t block, uint64_t submission) {
  if (uint64_t{kv_.chainLength(block)} * KvCache::pageTokens < config_.minimumTokens)
    return;
  const uint64_t fingerprint = kv_.key(block).indexHash;
  auto found = demands_.find(fingerprint);
  if (found == demands_.end()) {
    if (demands_.size() == demandHistoryCapacity) {
      auto oldest = demands_.find(demandOrder_.oldest()->id);
      RecencyOrder::unlink(oldest->second.recency);
      demands_.erase(oldest);
    }
    found = demands_.emplace(fingerprint,
        Demand{submission, false, RecencyOrder::allocate(fingerprint)}).first;
  } else {
    found->second.repeated |= found->second.firstSubmission != submission;
    RecencyOrder::unlink(found->second.recency);
  }
  demandOrder_.link(found->second.recency, ++demandClock_, fingerprint);
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
  std::vector<uint64_t> ids;
  ids.reserve(touched.size());
  for (auto [_, id] : touched)
    ids.push_back(id);
  touches_ = config_.kvFile->metadata(
      [store = config_.store, ids = std::move(ids)] { store->touch(ids); },
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

std::optional<uint64_t>
PersistentCache::admit(uint64_t block, const RestoreState &state, bool reused) {
  if (!kv_.contains(block)) return std::nullopt;
  auto chain = kv_.chain(block);
  if (state.boundary != chain.blocks.size() * KvCache::pageTokens)
    throw std::invalid_argument("persistent candidate boundary mismatch");
  // Recurrent/draft snapshots have a large fixed cost even for tiny prompts.
  // Keep short prefixes in the existing RAM/temporary tiers; admitting them
  // here would spend writes and displace expensive, reusable continuations.
  if (chain.blocks.size() * uint64_t{KvCache::pageTokens} <
      config_.minimumTokens)
    return std::nullopt;
  // Account shared records and missing writes identically for RAM and lane sources.
  const uint64_t stateBytes = state.bytes();
  const uint64_t pageBytes = config_.kvFile->slotBytes();
  if (stateBytes > config_.capacityBytes ||
      chain.blocks.size() > (config_.capacityBytes - stateBytes) / pageBytes)
    return std::nullopt;
  uint64_t extra = stateBytes;
  uint64_t writes = 0;
  std::set<std::vector<const void *>, std::less<>> sources;
  std::set<uint64_t> knownRecords;
  for (const auto &part : state.blocks) {
    const auto resources = part.payload->resources();
    // Payload wrappers may share backing; account each source write once.
    std::vector<const void *> identity;
    for (const auto &resource : resources)
      identity.push_back(resource.identity);
    if (identity.empty())
      identity.push_back(part.payload.get());
    if (sources.insert(std::move(identity)).second)
      writes += part.payload->offloadBytes();
    auto record = part.payload->diskRecord();
    if (record.slot)
      record.components.push_back(record.slot);
    for (const auto &slot : record.components)
      if (slot && knownRecords.insert(slot->recordId()).second &&
          references_.contains(slot->recordId()))
        extra -= slot->bytes();
  }
  for (uint64_t id : chain.blocks) {
    const auto slot = kv_.slot(id);
    const auto record = slot ? slot->record() : nullptr;
    if (!record || !references_.contains(record->recordId()))
      extra += pageBytes;
    if (!slot)
      writes += pageBytes;
  }
  // Repeated demand need not have produced a cache hit: its earlier state
  // may have been rejected or evicted before the next request arrived.
  const auto demand = demands_.find(kv_.key(block).indexHash);
  reused |= demand != demands_.end() && demand->second.repeated;
  // Under pressure, one-use tails leave room for observed reusable prefixes.
  // Existing branch/junction states already represent an observed prefix.
  // Free space still admits first-use sessions for immediate crash recovery.
  if (!reused && !kv_.stateBelow(block) &&
      extra > config_.capacityBytes - used_) {
    ++admissionSkips_;
    return std::nullopt;
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
    return std::nullopt;
  }
  writeCredit_ -= writes;
  return writes;
}

void PersistentCache::begin(RestoreLease state) {
  auto blocks = kv_.chain(state.kvBlock()).blocks;
  auto job = std::make_unique<Job>(
      Job{std::move(state), std::move(blocks), {}, {}, {}, false});
  for (uint64_t id : job->blocks)
    kv_.retainActive(id, CacheAccess::Maintenance);
  job_ = std::move(job);
}

void PersistentCache::start(uint64_t block, bool reused) {
  if (!kv_.contains(block)) return;
  auto lease = states_.acquireDeepest(std::span(&block, 1), CacheAccess::Maintenance);
  if (!lease || lease->kvBlock() != block || !admit(block, *lease->state(), reused))
    return;
  begin(std::move(*lease));
}

bool PersistentCache::capture(uint64_t block, const SnapshotWritePlan &plan) {
  if (job_ || states_.writing()) return false;
  const auto credit = admit(block, *plan.source, false);
  if (!credit) return false;
  if (!states_.publishToDisk(block, plan.write, completion_, makeRoom_, false)) {
    writeCredit_ += *credit;
    return false;
  }
  auto lease = states_.acquireDeepest(std::span(&block, 1), CacheAccess::Maintenance);
  if (!lease || lease->kvBlock() != block)
    throw std::logic_error("direct snapshot did not publish a complete restore point");
  begin(std::move(*lease));
  std::erase_if(pending_, [block](const auto &candidate) { return candidate.block == block; });
  return true;
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
      static_cast<void>(config_.kvFile->metadata(
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
  const auto anchor = kv_.slot(block)->record();
  Entry entry{anchor->recordId(), block, recency_.next(), {}};
  const auto prefix = describe(entry, *state, job.blocks);
  while (additionalBytes(entry) > config_.capacityBytes - used_)
    if (!evictOldest()) {
      ++failures_;
      finish();
      return true;
    }
  job.entry = std::move(entry);
  job.commit = config_.kvFile->metadata(
      [store = config_.store, prefix] { store->save(prefix); }, completion_);
  return true;
}

model::CacheStore::Prefix
PersistentCache::describe(Entry &entry, const RestoreState &state,
                          std::span<const uint64_t> blocks) const {
  model::CacheStore::Prefix prefix{
      entry.id, {metadataVersion, entry.id, state.blocks.size()}, {}};
  std::set<uint64_t> retained;
  const auto rememberRecord =
      [&](const std::shared_ptr<model::SlotFile::Slot> &slot) {
        if (!slot || !slot->recordId())
          throw std::logic_error("incomplete persistent payload");
        if (retained.insert(slot->recordId()).second) {
          prefix.records.push_back({slot->recordId(), slot->bytes()});
          entry.records.push_back(slot);
        }
      };
  for (const auto &part : state.blocks) {
    const auto record = part.payload->diskRecord();
    prefix.metadata.insert(prefix.metadata.end(),
                           {part.group, part.begin, part.end,
                            record.metadata.size(),
                            1 + record.components.size()});
    prefix.metadata.insert(prefix.metadata.end(), record.metadata.begin(),
                           record.metadata.end());
    rememberRecord(record.slot);
    prefix.metadata.insert(prefix.metadata.end(),
                           {record.slot->recordId(), record.slot->bytes()});
    for (const auto &component : record.components) {
      rememberRecord(component);
      prefix.metadata.insert(prefix.metadata.end(),
                             {component->recordId(), component->bytes()});
    }
  }
  prefix.metadata.push_back(blocks.size());
  for (uint64_t block : blocks) {
    const auto slot = kv_.slot(block)->record();
    rememberRecord(slot);
    const auto key = kv_.key(block);
    prefix.metadata.insert(prefix.metadata.end(),
                           {slot->recordId(), key.images.lo, key.images.hi});
    const auto tokens = key.tokens;
    prefix.metadata.insert(prefix.metadata.end(), tokens.begin(), tokens.end());
  }
  return prefix;
}

uint64_t PersistentCache::additionalBytes(const Entry &entry) const {
  uint64_t extra = 0;
  for (size_t i = 0; i < entry.records.size(); ++i)
    if (!references_.contains(entry.records[i]->recordId()))
      extra += entry.records[i]->bytes();
  return extra;
}
void PersistentCache::remember(Entry entry) {
  if (auto old = entries_.find(entry.block); old != entries_.end())
    forget(old);
  for (size_t i = 0; i < entry.records.size(); ++i) {
    entry.records[i]->retainDurable();
    auto &[count, bytes] = references_[entry.records[i]->recordId()];
    if (!count++) {
      bytes = entry.records[i]->bytes();
      used_ += bytes;
    }
  }
  entries_.emplace(entry.block, std::move(entry));
}
void PersistentCache::forget(std::map<uint64_t, Entry>::iterator entry) {
  const uint64_t id = entry->second.id;
  static_cast<void>(config_.kvFile->metadata(
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
          job_ != nullptr || !pending_.empty() || touches_ != nullptr ||
              std::any_of(entries_.begin(), entries_.end(),
                          [](const auto &item) { return item.second.touched; }),
          admissionSkips_,
          writeThrottles_};
}
} // namespace splash::engine
