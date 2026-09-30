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
             : state.block.end - state.block.begin != KvCache::pageTokens))
      throw std::runtime_error(
          "persistent group does not match model declaration");
  }
  const auto matched = coordinator.match(
      boundary, KvCache::pageTokens,
      [&](CacheGroupId group, uint32_t end) -> std::optional<CachedStateBlock> {
        const auto found = stateIndex.find({group, end});
        return found == stateIndex.end()
                   ? std::nullopt
                   : std::optional(result.states[found->second].block);
      });
  if (!matched || matched->boundary != boundary)
    throw std::runtime_error(
        "persistent manifest has incomplete group coverage");
  return result;
}
} // namespace

PersistentCache::PersistentCache(PersistentCacheConfig config, KvCache &kv,
                                 StateCache &states, model::KvTier &tier,
                                 CacheRecency &recency,
                                 std::function<void(uint64_t)> discardKv,
                                 const std::function<void()> &completion)
    : config_(std::move(config)), kv_(kv), states_(states), tier_(tier),
      recency_(recency), discardKv_(std::move(discardKv)),
      completion_(completion) {
  if (!config_.budget || !config_.budget->capacityBytes() || !config_.store ||
      !config_.kvFile || !config_.stateStorage)
    throw std::invalid_argument("incomplete persistent cache configuration");
  load();
}
PersistentCache::~PersistentCache() {
  const bool committed =
      job_ && job_->commit && job_->commit->wait() && !job_->failed;
  finish(committed);
  try {
    if (touches_)
      static_cast<void>(touches_->wait());
    static_cast<void>(flushTouches());
    static_cast<void>(config_.kvFile->metadata([] {})->wait());
  } catch (...) {
  }
  // Committed manifests survive process-local handle destruction. CacheStore
  // collects only records with no committed manifest reference.
}

bool PersistentCache::stored(const Slot &slot, const Location &location) const {
  if (!kv_.contains(location.block))
    return false;
  if (!location.group) {
    const auto copy = kv_.slot(location.block);
    return copy && copy->record().get() == &slot;
  }
  const auto copy = states_.diskCopy(*location.group, location.block);
  if (!copy)
    return false;
  const auto record = copy->diskRecord();
  return record.slot.get() == &slot ||
         std::any_of(record.components.begin(), record.components.end(),
                     [&](const auto &part) { return part.get() == &slot; });
}
void PersistentCache::hold(Entry &entry, const std::shared_ptr<Slot> &slot,
                           Location location) {
  if (!slot || !slot->recordId())
    throw std::logic_error("missing durable record");
  auto position = references_.find(slot->recordId());
  if (position == references_.end()) {
    // Prepare both allocations before publishing a reference. Failure cannot
    // leave an empty record directory entry that later admission mistakes for
    // an already charged durable slot.
    Reference reference{1, {location}};
    entry.records.reserve(entry.records.size() + 1);
    references_.emplace(slot->recordId(), std::move(reference));
    entry.records.push_back(slot);
    return;
  }
  auto &reference = position->second;
  auto found = std::find_if(reference.locations.begin(),
                            reference.locations.end(), [&](const auto &at) {
                              return at.block == location.block &&
                                     at.group == location.group;
                            });
  if (found == reference.locations.end())
    reference.locations.push_back(location);
  else
    found->temporary |= location.temporary;
  if (std::find(entry.records.begin(), entry.records.end(), slot) !=
      entry.records.end())
    return;
  entry.records.push_back(slot);
  ++reference.count;
}
void PersistentCache::release(Entry &entry) {
  // State references leave before target ancestry. Target leaves leave before
  // their parents, so disk-only records can disappear without orphan branches.
  std::sort(entry.records.begin(), entry.records.end(),
            [&](const auto &a, const auto &b) {
              const auto &left =
                  references_.at(a->recordId()).locations.front();
              const auto &right =
                  references_.at(b->recordId()).locations.front();
              if (left.group.has_value() != right.group.has_value())
                return left.group.has_value();
              return left.block > right.block;
            });
  for (auto &slot : entry.records) {
    auto found = references_.find(slot->recordId());
    if (--found->second.count)
      continue;
    auto locations = std::move(found->second.locations);
    references_.erase(found);
    if (!slot->durable())
      continue; // Reservation failed before ownership moved.
    const bool borrowed =
        std::any_of(locations.begin(), locations.end(), [&](const auto &at) {
          return at.temporary && stored(*slot, at);
        });
    if (borrowed && slot->returnTemporary())
      continue;
    slot->retire();
    for (const auto &at : locations) {
      if (!stored(*slot, at))
        continue;
      if (at.group)
        states_.invalidateDisk(*at.group, at.block, slot.get());
      else
        discardKv_(at.block);
    }
  }
  entry.records.clear();
}
void PersistentCache::forget(std::map<uint64_t, Entry>::iterator found) {
  const auto id = found->second.id;
  static_cast<void>(config_.kvFile->metadata(
      [store = config_.store, id] { store->erase(id); }));
  release(found->second);
  entries_.erase(found);
}
bool PersistentCache::evictOldest() {
  if (entries_.empty())
    return false;
  forget(std::min_element(entries_.begin(), entries_.end(),
                          [](const auto &a, const auto &b) {
                            return a.second.used < b.second.used;
                          }));
  return true;
}

void PersistentCache::publish(uint64_t block) {
  if (states_.checkpoint(block) || entries_.contains(block) ||
      (job_ && job_->entry.block == block) ||
      std::find(pending_.begin(), pending_.end(), block) != pending_.end())
    return;
  if (pending_.size() == 16)
    pending_.pop_front();
  pending_.push_back(block);
}
void PersistentCache::touch(uint64_t block) {
  auto found = entries_.find(block);
  if (found == entries_.end())
    return;
  found->second.used = recency_.next();
  found->second.touched = true;
}
void PersistentCache::start(uint64_t block) {
  if (!kv_.contains(block))
    return;
  auto lease = states_.acquireDeepest(std::span(&block, 1));
  if (!lease || lease->kvBlock() != block)
    return;
  auto source = states_.diskSource(block, *lease->state());
  const auto plan = config_.stateStorage->prepareWrite(std::move(source));
  if (plan.source && plan.write)
    static_cast<void>(begin(block, plan, std::move(lease)));
}
bool PersistentCache::capture(uint64_t block, const RestoreState &source) {
  if (job_ || states_.writing())
    return false;
  // A partial cache entry may already own some groups. Reuse those records
  // before planning IO, just as for RAM publication; replacing only the
  // missing groups keeps the manifest and the live index on the same copies.
  const auto plan =
      config_.stateStorage->prepareWrite(states_.diskSource(block, source));
  return plan.source && plan.write && begin(block, plan, {});
}
bool PersistentCache::begin(uint64_t block, const SnapshotWritePlan &plan,
                            std::optional<RestoreLease> lease) {
  if (plan.source->boundary < 512 || !kv_.contains(block))
    return false;
  auto blocks = kv_.chain(block).blocks;
  if (std::any_of(blocks.begin(), blocks.end(),
                  [&](auto id) { return kv_.transferring(id); }))
    return false;
  if (blocks.size() * KvCache::pageTokens != plan.source->boundary)
    throw std::invalid_argument("persistent candidate boundary mismatch");
  job_ = std::make_unique<Job>();
  auto &job = *job_;
  job.entry = {0, block, recency_.next(), {}};
  job.lease = std::move(lease);
  job.blocks = std::move(blocks);
  for (auto id : job.blocks)
    kv_.retainActive(id);
  try {
    uint64_t total = 0, extra = 0;
    const auto count = [&](uint64_t bytes, bool durable = false) {
      if (bytes > config_.budget->capacityBytes() - total)
        return false;
      total += bytes;
      if (!durable)
        extra += bytes;
      return true;
    };
    const auto existing = [&](const std::shared_ptr<Slot> &slot, Location at) {
      at.temporary = !slot->durable() && stored(*slot, at);
      const bool unique =
          std::find(job.entry.records.begin(), job.entry.records.end(), slot) ==
          job.entry.records.end();
      hold(job.entry, slot,
           at); // Protect shared durable records before LRU eviction.
      return !unique || count(slot->bytes(), slot->durable());
    };
    for (const auto &part : plan.source->blocks) {
      const auto endpoint = job.blocks[part.end / KvCache::pageTokens - 1];
      const auto record = part.payload->diskRecord();
      uint64_t newBytes = part.payload->bytes();
      if (record.slot) {
        if (record.slot->bytes() > newBytes)
          throw std::logic_error("invalid state footprint");
        newBytes -= record.slot->bytes();
        if (!existing(record.slot, {endpoint, part.group})) {
          finish();
          return false;
        }
      }
      for (const auto &component : record.components) {
        if (component->bytes() > newBytes)
          throw std::logic_error("invalid state components");
        newBytes -= component->bytes();
        if (!existing(component, {endpoint, part.group})) {
          finish();
          return false;
        }
      }
      if (!count(newBytes)) {
        finish();
        return false;
      }
    }
    for (auto id : job.blocks) {
      const auto slot = kv_.slot(id);
      if (slot) {
        if (!existing(slot->record(), {id, {}})) {
          finish();
          return false;
        }
      } else if (!count(config_.kvFile->slotBytes())) {
        finish();
        return false;
      }
    }
    do {
      job.reservation = model::DiskReservation::acquire(config_.budget, extra);
    } while (!job.reservation && evictOldest());
    if (!job.reservation) {
      finish();
      return false;
    }
    for (auto &slot : job.entry.records)
      slot->transfer(*job.reservation);
    auto write = plan.write(completion_, &*job.reservation);
    if (!write) {
      finish();
      return false;
    }
    job.disk = write->state();
    for (const auto &part : job.disk->blocks) {
      const auto endpoint = job.blocks[part.end / KvCache::pageTokens - 1];
      auto record = part.payload->diskRecord();
      hold(job.entry, record.slot, {endpoint, part.group});
      for (const auto &component : record.components)
        hold(job.entry, component, {endpoint, part.group});
    }
    states_.publishDiskWrite(block, std::move(write));
    if (!job.lease)
      job.lease = states_.acquireDeepest(std::span(&block, 1));
    if (!job.lease)
      throw std::logic_error("disk snapshot did not publish a restore point");
    std::erase(pending_, block);
    return true;
  } catch (...) {
    finish();
    throw;
  }
}

bool PersistentCache::poll() {
  bool progress = flushTouches();
  if (!job_ && !pending_.empty() && !states_.writing()) {
    const auto candidate = pending_.front();
    if (kv_.contains(candidate)) {
      const auto blocks = kv_.chain(candidate).blocks;
      // Finish temporary IO before borrowing a slot: a failed demotion must
      // resolve back to RAM before the manifest computes its reservation.
      if (std::any_of(blocks.begin(), blocks.end(),
                      [&](auto id) { return kv_.transferring(id); }))
        return progress;
    }
    pending_.pop_front();
    try {
      start(candidate);
    } catch (const std::exception &) {
      ++failures_;
    }
    progress = true;
  }
  if (!job_)
    return progress;
  auto &job = *job_;
  if (job.commit) {
    if (!job.commit->ready())
      return progress;
    const bool saved = job.commit->wait() && !job.failed;
    if (!saved) {
      const auto id = job.entry.id;
      static_cast<void>(config_.kvFile->metadata(
          [store = config_.store, id] { store->erase(id); }));
      ++failures_;
    }
    finish(saved);
    return true;
  }
  for (auto it = job.copies.begin(); it != job.copies.end();) {
    if (!it->transfer->ready()) {
      ++it;
      continue;
    }
    const bool written = it->transfer->finish();
    kv_.setTransferring(it->block, false);
    if (!written)
      job.failed = true;
    it = job.copies.erase(it);
    progress = true;
  }
  if (!job.failed && !states_.writing(job.entry.block)) {
    for (const auto &part : job.disk->blocks) {
      const auto id = job.blocks[part.end / KvCache::pageTokens - 1];
      if (!states_.diskCopy(part.group, id)) {
        job.failed = true;
        break;
      }
    }
  }
  if (!job.failed) {
    for (auto id : job.blocks) {
      if (kv_.slot(id) || kv_.transferring(id))
        continue;
      if (kv_.page(id) == KvCache::noPage || !tier_.writable()) {
        job.failed = true;
        break;
      }
      if (!tier_.canDemote())
        break;
      auto record = config_.kvFile->acquire(*job.reservation);
      auto slot = tier_.reopenSlot(record);
      auto transfer = tier_.demote(kv_.page(id), slot, completion_);
      if (!transfer)
        throw std::logic_error(
            "KV staging admission changed during publication");
      hold(job.entry, record, {id, {}});
      kv_.setSlot(id, std::move(slot));
      kv_.setTransferring(id, true);
      job.copies.push_back({id, std::move(transfer)});
      progress = true;
    }
  }
  if (!job.copies.empty() || states_.writing(job.entry.block))
    return progress;
  if (job.failed) {
    ++failures_;
    finish();
    return true;
  }
  for (auto id : job.blocks)
    if (!kv_.slot(id) || kv_.transferring(id))
      return progress;
  job.entry.id = kv_.slot(job.entry.block)->record()->recordId();
  auto prefix = describe(job);
  job.commit = config_.kvFile->metadata(
      [store = config_.store, prefix = std::move(prefix)] {
        store->save(prefix);
      },
      completion_);
  return true;
}
void PersistentCache::finish(bool committed) {
  if (!job_)
    return;
  // Transfers drain before records and pins leave; committed jobs transfer
  // their existing record references to the manifest without charging again.
  for (auto &copy : job_->copies) {
    copy.transfer.reset();
    kv_.setTransferring(copy.block, false);
  }
  job_->copies.clear();
  if (job_->disk)
    states_.finishDiskWrite();
  if (job_->commit)
    static_cast<void>(job_->commit->wait());
  job_->lease.reset();
  if (committed) {
    entries_.emplace(job_->entry.block, std::move(job_->entry));
    ++saved_;
  } else
    release(job_->entry);
  for (auto id : job_->blocks)
    kv_.releaseActive(id);
  job_.reset();
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
            config_.budget->capacityBytes() -
                std::min(extra, config_.budget->capacityBytes())) {
          valid = false;
          break;
        }
        extra += record.bytes;
      }
    if (!valid || extra > config_.budget->capacityBytes() - bytes) {
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
    uint64_t extra = 0;
    for (const auto &record : it->records)
      if (!references_.contains(record.id))
        extra += record.bytes;
    auto reservation = model::DiskReservation::acquire(config_.budget, extra);
    if (!reservation)
      throw std::logic_error("startup durable reservation failed");
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
            block, {part.metadata, part.records}, *reservation);
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
      auto slot = config_.kvFile->reopen(decoded.blocks[offset], *reservation);
      if (!slot)
        throw std::runtime_error(
            "persistent cache exceeds reserved disk quota");
      std::array<uint32_t, KvCache::pageTokens> tokens;
      std::copy_n(decoded.blocks.begin() + offset + 3, tokens.size(),
                  tokens.begin());
      entry.block = kv_.importDisk(
          entry.block, tokens,
          {decoded.blocks[offset + 1], decoded.blocks[offset + 2]},
          tier_.reopenSlot(slot));
      entry.records.push_back(std::move(slot));
    }
    states_.importDisk(entry.block, state);
    entry.records.clear();
    const auto chain = kv_.chain(entry.block).blocks;
    for (const auto &part : state->blocks) {
      const auto block = chain[part.end / KvCache::pageTokens - 1];
      auto record = part.payload->diskRecord();
      hold(entry, record.slot, {block, part.group});
      for (const auto &component : record.components)
        hold(entry, component, {block, part.group});
    }
    for (auto block : chain)
      hold(entry, kv_.slot(block)->record(), {block, {}});
    entries_.emplace(entry.block, std::move(entry));
    ++restored_;
  }
}

model::CacheStore::Prefix PersistentCache::describe(const Job &job) const {
  const auto &entry = job.entry;
  const auto &state = *job.disk;
  const auto &blocks = job.blocks;
  model::CacheStore::Prefix prefix{
      entry.id, {metadataVersion, entry.id, state.blocks.size()}, {}};
  std::set<uint64_t> retained;
  const auto rememberRecord =
      [&](const std::shared_ptr<model::SlotFile::Slot> &slot) {
        if (!slot || !slot->recordId())
          throw std::logic_error("incomplete persistent payload");
        if (retained.insert(slot->recordId()).second) {
          prefix.records.push_back({slot->recordId(), slot->bytes()});
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
  const auto slot = kv_.contains(block) ? kv_.slot(block) : nullptr;
  const auto record = slot ? slot->record() : nullptr;
  for (auto it = entries_.begin(); it != entries_.end();) {
    auto current = it++;
    if (current->first == block ||
        (record && std::find(current->second.records.begin(),
                             current->second.records.end(),
                             record) != current->second.records.end()))
      forget(current);
  }
}
bool PersistentCache::busy() const noexcept {
  return job_ || !pending_.empty() || touches_ ||
         std::any_of(entries_.begin(), entries_.end(),
                     [](const auto &item) { return item.second.touched; });
}
PersistentCacheSnapshot PersistentCache::snapshot() const {
  return {config_.budget->capacityBytes(),
          config_.budget->usedBytes(),
          config_.budget->readBytes(),
          config_.budget->writtenBytes(),
          saved_,
          restored_,
          failures_,
          static_cast<uint32_t>(entries_.size()),
          busy()};
}
} // namespace splash::engine
