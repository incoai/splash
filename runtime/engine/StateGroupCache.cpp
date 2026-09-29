#include "engine/StateGroupCache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {

StateBlockLease::StateBlockLease(StateGroupCache &owner, uint64_t kvBlock,
                                 uint32_t boundary,
                                 std::shared_ptr<const StatePayload> state,
                                 CacheAccess access) noexcept
    : access_(access), owner_(&owner), kvBlock_(kvBlock), boundary_(boundary),
      state_(std::move(state)) {}

StateBlockLease::StateBlockLease(StateBlockLease &&other) noexcept
    : access_(other.access_), owner_(std::exchange(other.owner_, nullptr)),
      kvBlock_(std::exchange(other.kvBlock_, 0)),
      boundary_(std::exchange(other.boundary_, 0)),
      state_(std::move(other.state_)) {}

StateBlockLease &StateBlockLease::operator=(StateBlockLease &&other) noexcept {
  if (this == &other)
    return *this;
  reset();
  access_ = other.access_;
  owner_ = std::exchange(other.owner_, nullptr);
  kvBlock_ = std::exchange(other.kvBlock_, 0);
  boundary_ = std::exchange(other.boundary_, 0);
  state_ = std::move(other.state_);
  return *this;
}

StateBlockLease::~StateBlockLease() noexcept { reset(); }

void StateBlockLease::reset() noexcept {
  if (owner_)
    owner_->release(kvBlock_, access_);
  owner_ = nullptr;
  kvBlock_ = 0;
  boundary_ = 0;
  state_.reset();
}

std::optional<StateBlockLease>
StateGroupCache::acquireBlock(uint64_t kvBlock, CacheAccess access) {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.invalid)
    return std::nullopt;
  Entry &entry = found->second;
  if (!kv_.contains(kvBlock)) {
    throw std::logic_error("group state outlived its target KV block");
  }
  if (entry.pins == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("group state pin count overflowed");
  }
  if (!entry.pins && pinnedEntries_ == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("group state pinned entry count overflowed");
  }
  kv_.retainActive(kvBlock, access);
  if (!entry.pins)
    ++pinnedEntries_;
  ++entry.pins;
  reindex(kvBlock, entry);
  const uint32_t boundary = kv_.chainLength(kvBlock) * KvCache::pageTokens;
  return StateBlockLease(*this, kvBlock, boundary, copy(entry), access);
}

bool StateGroupCache::touchIfStored(uint64_t kvBlock, bool checkpoint) {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.invalid)
    return false;
  if (!kv_.contains(kvBlock)) {
    throw std::logic_error("group state outlived its target KV block");
  }
  Entry &entry = found->second;
  if (!checkpoint)
    makeOrdinary(kvBlock, entry);
  entry.lastUsed = recency_.next();
  reindex(kvBlock, entry);
  ++deduplicatedPublications_;
  return true;
}

void StateGroupCache::touch(uint64_t kvBlock) noexcept {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end())
    return;
  found->second.lastUsed = recency_.next();
  reindex(kvBlock, found->second);
}

void StateGroupCache::publish(uint64_t kvBlock,
                              std::shared_ptr<const StatePayload> state,
                              bool checkpoint, uint32_t begin) {
  if (!state || !state->bytes()) {
    throw std::invalid_argument("group state payload is empty");
  }
  if (state->residentBytes() != state->bytes())
    throw std::invalid_argument("published state must be in RAM");
  if (!kv_.contains(kvBlock)) {
    throw std::invalid_argument("group state KV block is unknown");
  }
  if (publications_ == std::numeric_limits<uint64_t>::max())
    throw std::overflow_error("group state publication count overflowed");
  const uint64_t stateBytes = state->bytes();
  if (bytes_ > std::numeric_limits<uint64_t>::max() - stateBytes) {
    throw std::overflow_error("group state byte count overflowed");
  }
  if (resident(kvBlock))
    throw std::logic_error("duplicate group state key");

  Entry &entry = publicationEntry(kvBlock, checkpoint);
  retainRam(entry, *state);
  entry.begin = begin == UINT32_MAX
                    ? kv_.chainLength(kvBlock) * KvCache::pageTokens
                    : begin;
  entry.ram = std::move(state);
  entry.lastUsed = recency_.next();
  reindex(kvBlock, entry);
  ++publications_;
}

bool StateGroupCache::copyToDisk(uint64_t block,
                                 const std::function<void()> &completion,
                                 const std::function<bool()> &makeRoom) {
  auto found = entries_.find(block);
  if (found == entries_.end() || found->second.invalid)
    return false;
  Entry &entry = found->second;
  if (entry.disk)
    return true;
  if (!entry.ram || !entry.ram->canOffload())
    return false;
  auto transfer = startWrite(
      [state = entry.ram](std::function<void()> done) {
        return state->offload(std::move(done));
      },
      completion, makeRoom);
  if (!transfer)
    return false;
  beginWrite(block, entry, std::move(transfer));
  reindex(block, entry);
  return true;
}

std::shared_ptr<const StatePayload>
StateGroupCache::diskCopy(uint64_t block) const {
  auto found = entries_.find(block);
  return found != entries_.end() && !found->second.invalid && !writing(block)
             ? found->second.disk
             : nullptr;
}

bool StateGroupCache::importDisk(uint64_t block,
                                 std::shared_ptr<const StatePayload> state,
                                 uint32_t begin, bool checkpoint) {
  if (!state || state->residentBytes() || !kv_.contains(block))
    throw std::invalid_argument("invalid imported state");
  if (begin == UINT32_MAX)
    begin = kv_.chainLength(block) * KvCache::pageTokens;
  if (contains(block) && begin < entries_.at(block).begin) {
    // A wider immutable image can restore older boundaries too. Replace a
    // narrow slice only when no reader or write owns that publication.
    if (writing(block) || !evict(block).evicted)
      return false;
  }
  if (contains(block)) {
    auto &entry = entries_.at(block);
    // A disk image may cover more than the resident descriptor, but never
    // less. Keep an existing disk owner and any in-flight transfer intact.
    if (entry.disk || (entry.ram && (begin > entry.begin ||
                                   state->bytes() != entry.ram->bytes())))
      return false;
    retainDisk(entry, *state);
    if (!entry.ram)
      entry.begin = begin;
    entry.disk = std::move(state);
    reindex(block, entry);
    return true;
  }
  Entry &entry = publicationEntry(block, checkpoint);
  retainDisk(entry, *state);
  entry.begin = begin;
  entry.disk = std::move(state);
  entry.lastUsed = recency_.next();
  reindex(block, entry);
  return true;
}

std::optional<CacheEvictionCandidate>
StateGroupCache::candidate(uint64_t block, bool disk,
                           bool duplicate) const noexcept {
  const auto found = entries_.find(block);
  if (found == entries_.end())
    return std::nullopt;
  const auto &entry = found->second;
  if (entry.invalid || entry.pins)
    return std::nullopt;
  if (disk ? (!entry.disk || writing(block) || entry.disk->durable() ||
              bool(entry.ram) != duplicate)
           : !entry.ram)
    return std::nullopt;
  return CacheEvictionCandidate{block, entry.lastUsed, group_};
}

std::optional<CachedStateBlock> StateGroupCache::peek(uint64_t block) const {
  const auto found = entries_.find(block);
  if (found == entries_.end() || found->second.invalid || !copy(found->second))
    return std::nullopt;
  return CachedStateBlock{group_, found->second.begin,
                          kv_.chainLength(block) * KvCache::pageTokens,
                          copy(found->second)};
}

GroupCheckpoint StateGroupCache::checkpoint(uint64_t kvBlock) const noexcept {
  const auto found = entries_.find(kvBlock);
  if (found == entries_.end() || !found->second.checkpoint)
    return {};
  return {kvBlock, found->second.publication};
}

bool StateGroupCache::retireCheckpoint(GroupCheckpoint checkpoint) noexcept {
  const auto found = entries_.find(checkpoint.kvBlock);
  if (found == entries_.end() || !found->second.checkpoint ||
      found->second.publication != checkpoint.publication)
    return true;
  return erase(checkpoint.kvBlock, true).evicted;
}

void StateGroupCache::makeOrdinary(uint64_t kvBlock, Entry &entry) {
  kv_.noteState(kvBlock);
  if (!entry.checkpoint)
    return;
  --checkpointEntries_;
  checkpointBytes_ -= checkpointResources_.release(entry.ramResources);
  entry.checkpoint = false;
}

bool StateGroupCache::contains(uint64_t kvBlock) const noexcept {
  const auto found = entries_.find(kvBlock);
  return found != entries_.end() && !found->second.invalid;
}

uint64_t StateGroupCache::resumePoint() const noexcept {
  // Checkpoints can survive cancellation but remain disposable. Prefer an
  // ordinary state for speculative protection, regardless of recency.
  return ordinary_.newestId() ? ordinary_.newestId() : checkpoints_.newestId();
}

bool StateGroupCache::resident(uint64_t kvBlock) const noexcept {
  const auto found = entries_.find(kvBlock);
  return found != entries_.end() && found->second.ram != nullptr;
}

std::optional<CacheEvictionCandidate>
StateGroupCache::evictionCandidate(bool checkpoints) const noexcept {
  for (const auto candidate :
       {checkpoints ? checkpoints_.oldest() : std::nullopt, ordinary_.oldest()})
    if (candidate)
      return CacheEvictionCandidate{candidate->id, candidate->lastUsed, group_};
  return std::nullopt;
}

std::optional<CacheEvictionCandidate>
StateGroupCache::diskCandidate(bool duplicate) const noexcept {
  const auto &order = duplicate ? duplicates_ : diskOnly_;
  for (auto candidate = order.oldest(); candidate;
       candidate = order.next(*candidate))
    if (!entries_.find(candidate->id)->second.disk->durable())
      return CacheEvictionCandidate{candidate->id, candidate->lastUsed, group_};
  return std::nullopt;
}

StateEviction StateGroupCache::reclaim(uint64_t kvBlock,
                                       std::function<void()> completion,
                                       const std::function<bool()> &makeRoom) {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.pins || !found->second.ram)
    return {};
  Entry &entry = found->second;
  // One write at a time.
  const bool writable =
      offloadEnabled_ && !entry.disk && entry.ram->canOffload();
  if (writable && pending_)
    return {false, 0, true};
  if (writable) {
    if (auto transfer = startWrite(
            [state = entry.ram](std::function<void()> done) {
              return state->offload(std::move(done));
            },
            completion, makeRoom))
      beginWrite(kvBlock, entry, std::move(transfer));
  }
  if (!entry.disk)
    return erase(kvBlock, false);
  // A write reads its own copy, so the RAM copy is free at once.
  const uint64_t reclaimable = entry.ram->reclaimableBytes();
  const uint64_t reclaimed = std::min(reclaimable, releaseRam(entry));
  entry.ram.reset();
  reindex(kvBlock, entry);
  return {true, reclaimed};
}

StateEviction StateGroupCache::evict(uint64_t kvBlock) noexcept {
  return erase(kvBlock, false);
}

void StateGroupCache::dropDisk(uint64_t kvBlock) {
  Entry &target = entry(kvBlock);
  if (!target.ram || !target.disk)
    throw std::logic_error("only a redundant disk copy is dropped");
  if (writing(kvBlock))
    throw std::logic_error("a disk copy being written cannot be dropped");
  discardDisk(target);
  reindex(kvBlock, target);
}

void StateGroupCache::invalidate(uint64_t kvBlock,
                                 const StatePayload *state) noexcept {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.invalid)
    return;
  Entry &target = found->second;
  if (target.ram && target.disk.get() == state) {
    // The RAM copy stands; only the copy that failed to read leaves.
    discardDisk(target);
    ++invalidations_;
    reindex(kvBlock, target);
    return;
  }
  if (copy(target).get() != state)
    return;
  target.invalid = true;
  ++invalidations_;
  reindex(kvBlock, target);
  static_cast<void>(evict(kvBlock));
}

void StateGroupCache::invalidate(uint64_t kvBlock) noexcept {
  const auto found = entries_.find(kvBlock);
  if (found != entries_.end())
    invalidate(kvBlock, copy(found->second).get());
}

bool StateGroupCache::promotable(uint64_t kvBlock,
                                 const StatePayload *source) const noexcept {
  const auto found = entries_.find(kvBlock);
  return found != entries_.end() && !found->second.invalid &&
         !found->second.ram && found->second.disk.get() == source;
}

void StateGroupCache::promote(uint64_t kvBlock, const StatePayload *source,
                              std::shared_ptr<const StatePayload> state) {
  if (!promotable(kvBlock, source))
    return;
  if (!state || state->bytes() != source->bytes() ||
      state->residentBytes() != state->bytes())
    throw std::invalid_argument("invalid promoted state");
  if (state->bytes() > std::numeric_limits<uint64_t>::max() - bytes_)
    throw std::overflow_error("promoted state byte count overflowed");
  Entry &target = entry(kvBlock);
  retainRam(target, *state);
  target.ram = std::move(state);
  reindex(kvBlock, target);
  ++promotions_;
}

void StateGroupCache::setExternalWrite(uint64_t block,
                                       const StatePayload *source,
                                       bool writing) noexcept {
  const auto found = entries_.find(block);
  if (found == entries_.end() || found->second.disk.get() != source)
    return;
  found->second.externalWrite = writing;
  reindex(block, found->second);
}

bool StateGroupCache::pollOffload() {
  if (!pending_ || !pending_->transfer->ready())
    return false;
  PendingOffload done = std::move(*pending_);
  pending_.reset();
  const bool written = done.transfer->finish();
  // A failure counts even when its entry left or changed meanwhile.
  if (!written)
    ++offloadFailures_;
  auto found = entries_.find(done.kvBlock);
  if (found == entries_.end())
    return true;
  Entry &target = found->second;
  if (target.disk == done.transfer->state()) {
    if (!written)
      discardDisk(target);
    if (!target.ram && !target.disk) {
      // Nothing is left of the state; a pinned reader releases it, and a
      // publication meanwhile takes the entry over.
      target.invalid = true;
      reindex(done.kvBlock, target);
      static_cast<void>(evict(done.kvBlock));
      return true;
    }
  }
  reindex(done.kvBlock, target);
  return true;
}

StateEviction StateGroupCache::erase(uint64_t kvBlock,
                                     bool retirement) noexcept {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.pins)
    return {};
  Entry &target = found->second;
  const uint64_t reclaimable = target.ram ? target.ram->reclaimableBytes() : 0;
  const uint64_t reclaimed = std::min(reclaimable, releaseRam(target));
  diskBytes_ -= diskResources_.release(target.diskResources);
  if (target.checkpoint) {
    --checkpointEntries_;
    if (!retirement)
      ++checkpointEvictions_;
  }
  unlink(target);
  entries_.erase(found);
  kv_.countState(kvBlock, false);
  if (retirement)
    ++checkpointRetirements_;
  else
    ++evictions_;
  return {true, reclaimed};
}

StateCacheSnapshot StateGroupCache::snapshot() const noexcept {
  StateCacheSnapshot result;
  result.entries = static_cast<uint32_t>(std::min<uint64_t>(
      entries_.size(), std::numeric_limits<uint32_t>::max()));
  result.pinned = pinnedEntries_;
  result.bytes = bytes_;
  result.diskBytes = diskBytes_;
  result.offloads = offloads_;
  result.offloadFailures = offloadFailures_;
  result.invalidations = invalidations_;
  result.promotions = promotions_;
  result.publications = publications_;
  result.deduplicatedPublications = deduplicatedPublications_;
  result.evictions = evictions_;
  result.checkpointEntries = static_cast<uint32_t>(std::min<uint64_t>(
      checkpointEntries_, std::numeric_limits<uint32_t>::max()));
  result.checkpointBytes = checkpointBytes_;
  result.checkpointRetirements = checkpointRetirements_;
  result.checkpointEvictions = checkpointEvictions_;
  return result;
}

void StateGroupCache::release(uint64_t kvBlock, CacheAccess access) noexcept {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || !found->second.pins)
    std::terminate();
  Entry &target = found->second;
  --target.pins;
  if (access == CacheAccess::Request)
    target.lastUsed = recency_.next();
  if (!target.pins) {
    if (!pinnedEntries_)
      std::terminate();
    --pinnedEntries_;
    reindex(kvBlock, target);
    if (target.invalid)
      static_cast<void>(evict(kvBlock));
  }
  kv_.releaseActive(kvBlock, access);
}

StateGroupCache::Entry &StateGroupCache::entry(uint64_t kvBlock) {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end())
    throw std::out_of_range("unknown group state");
  return found->second;
}

StateGroupCache::Entry &StateGroupCache::entryFor(uint64_t kvBlock) {
  auto found = entries_.find(kvBlock);
  if (found != entries_.end())
    return found->second;
  Entry fresh;
  fresh.ramNode = RecencyOrder::allocate(kvBlock);
  fresh.diskNode = RecencyOrder::allocate(kvBlock);
  fresh.publication = ++publicationSequence_;
  Entry &placed = entries_.emplace(kvBlock, std::move(fresh)).first->second;
  kv_.countState(kvBlock, true);
  return placed;
}

StateGroupCache::Entry &StateGroupCache::publicationEntry(uint64_t kvBlock,
                                                          bool checkpoint) {
  const bool fresh = !entries_.contains(kvBlock);
  Entry &entry = entryFor(kvBlock);
  // An entry a failed read condemned gives its copy up (one a failed write
  // condemned has none); readers of that copy keep their own handle to it.
  if (entry.invalid) {
    discardDisk(entry);
    entry.invalid = false;
  }
  if (fresh && checkpoint) {
    entry.checkpoint = true;
    ++checkpointEntries_;
  }
  if (!checkpoint)
    makeOrdinary(kvBlock, entry);
  return entry;
}

std::unique_ptr<StateOffload>
StateGroupCache::startWrite(const StateBlockWriter &write,
                            const std::function<void()> &completion,
                            const std::function<bool()> &makeRoom) {
  if (pending_)
    return {};
  std::unique_ptr<StateOffload> transfer = write(completion);
  while (!transfer && makeRoom && makeRoom())
    transfer = write(completion);
  return transfer;
}

void StateGroupCache::beginWrite(uint64_t kvBlock, Entry &target,
                                 std::unique_ptr<StateOffload> transfer) {
  retainDisk(target, *transfer->state());
  target.disk = transfer->state();
  pending_.emplace(PendingOffload{kvBlock, std::move(transfer)});
  ++offloads_;
}

// RAM copies wait for eviction in one order per class; disk copies wait for
// replacement as redundant copies or as the only copy. A pinned or invalid
// entry, or a copy being written, is in no order.
void StateGroupCache::reindex(uint64_t kvBlock, Entry &target) noexcept {
  unlink(target);
  if (target.pins || target.invalid)
    return;
  if (target.ram)
    (target.checkpoint ? checkpoints_ : ordinary_)
        .link(target.ramNode, target.lastUsed, kvBlock);
  if (target.disk && !writing(kvBlock))
    (target.ram ? duplicates_ : diskOnly_)
        .link(target.diskNode, target.lastUsed, kvBlock);
}

void StateGroupCache::unlink(Entry &target) noexcept {
  if (target.ramNode.linked())
    RecencyOrder::unlink(target.ramNode);
  if (target.diskNode.linked())
    RecencyOrder::unlink(target.diskNode);
}

void StateGroupCache::retainRam(Entry &entry, const StatePayload &state) {
  auto resources = state.resources();
  const uint64_t added = ramResources_.retain(resources);
  try {
    if (entry.checkpoint)
      checkpointBytes_ += checkpointResources_.retain(resources);
  } catch (...) {
    ramResources_.release(resources);
    throw;
  }
  bytes_ += added;
  entry.ramResources = std::move(resources);
}

uint64_t StateGroupCache::releaseRam(Entry &entry) noexcept {
  const uint64_t released = ramResources_.release(entry.ramResources);
  bytes_ -= released;
  if (entry.checkpoint)
    checkpointBytes_ -= checkpointResources_.release(entry.ramResources);
  entry.ramResources.clear();
  return released;
}

void StateGroupCache::retainDisk(Entry &entry, const StatePayload &state) {
  auto resources = state.resources();
  diskBytes_ += diskResources_.retain(resources);
  entry.diskResources = std::move(resources);
}

void StateGroupCache::discardDisk(Entry &target) noexcept {
  if (!target.disk)
    return;
  diskBytes_ -= diskResources_.release(target.diskResources);
  target.diskResources.clear();
  target.disk.reset();
}

} // namespace splash::engine
