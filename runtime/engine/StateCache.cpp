#include "engine/StateCache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {

CompositeStateLease::CompositeStateLease(
    StateCache &owner, uint64_t kvBlock, uint32_t boundary,
    std::shared_ptr<const CompositeState> state) noexcept
    : owner_(&owner), kvBlock_(kvBlock), boundary_(boundary),
      state_(std::move(state)) {}

CompositeStateLease::CompositeStateLease(CompositeStateLease &&other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      kvBlock_(std::exchange(other.kvBlock_, 0)),
      boundary_(std::exchange(other.boundary_, 0)),
      state_(std::move(other.state_)) {}

CompositeStateLease &
CompositeStateLease::operator=(CompositeStateLease &&other) noexcept {
  if (this == &other)
    return *this;
  reset();
  owner_ = std::exchange(other.owner_, nullptr);
  kvBlock_ = std::exchange(other.kvBlock_, 0);
  boundary_ = std::exchange(other.boundary_, 0);
  state_ = std::move(other.state_);
  return *this;
}

CompositeStateLease::~CompositeStateLease() noexcept { reset(); }

void CompositeStateLease::reset() noexcept {
  if (owner_)
    owner_->release(kvBlock_);
  owner_ = nullptr;
  kvBlock_ = 0;
  boundary_ = 0;
  state_.reset();
}

std::optional<CompositeStateLease>
StateCache::acquireDeepest(std::span<const uint64_t> kvChain) {
  for (auto block = kvChain.rbegin(); block != kvChain.rend(); ++block) {
    if (auto lease = acquireBlock(*block))
      return lease;
  }
  return std::nullopt;
}

std::optional<CompositeStateLease> StateCache::acquireBlock(uint64_t kvBlock) {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.invalid)
    return std::nullopt;
  Entry &entry = found->second;
  if (!kv_.contains(kvBlock)) {
    throw std::logic_error("composite state outlived its target KV block");
  }
  if (entry.pins == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("composite state pin count overflowed");
  }
  if (!entry.pins && pinnedEntries_ == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("composite state pinned entry count overflowed");
  }
  kv_.retainActive(kvBlock);
  if (!entry.pins)
    ++pinnedEntries_;
  ++entry.pins;
  reindex(kvBlock, entry);
  const uint32_t boundary = kv_.chainLength(kvBlock) * KvCache::pageTokens;
  return CompositeStateLease(*this, kvBlock, boundary, copy(entry));
}

void StateCache::recordLookup(bool hit, bool disk) noexcept {
  hit ? ++hits_ : ++misses_;
  if (disk) ++diskHits_;
}

bool StateCache::touchIfResident(uint64_t kvBlock, bool checkpoint) {
  return resident(kvBlock) && touchIfStored(kvBlock, checkpoint);
}

bool StateCache::touchIfStored(uint64_t kvBlock, bool checkpoint) {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.invalid)
    return false;
  if (!kv_.contains(kvBlock)) {
    throw std::logic_error("composite state outlived its target KV block");
  }
  Entry &entry = found->second;
  if (!checkpoint)
    makeOrdinary(kvBlock, entry);
  if (!entry.pins)
    entry.lastUsed = recency_.next();
  reindex(kvBlock, entry);
  ++deduplicatedPublications_;
  return true;
}

void StateCache::touch(uint64_t kvBlock) noexcept {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.pins)
    return;
  found->second.lastUsed = recency_.next();
  reindex(kvBlock, found->second);
}

void StateCache::publish(uint64_t kvBlock,
                         std::shared_ptr<const CompositeState> state,
                         bool checkpoint) {
  if (!state || !state->bytes()) {
    throw std::invalid_argument("composite state payload is empty");
  }
  if (state->residentBytes() != state->bytes())
    throw std::invalid_argument("published state must be in RAM");
  if (!kv_.contains(kvBlock)) {
    throw std::invalid_argument("composite state KV block is unknown");
  }
  if (publications_ == std::numeric_limits<uint64_t>::max())
    throw std::overflow_error("composite state publication count overflowed");
  const uint64_t stateBytes = state->bytes();
  if (bytes_ > std::numeric_limits<uint64_t>::max() - stateBytes) {
    throw std::overflow_error("composite state byte count overflowed");
  }
  if (resident(kvBlock))
    throw std::logic_error("duplicate composite state key");

  Entry &entry = publicationEntry(kvBlock, checkpoint);
  entry.ram = std::move(state);
  bytes_ += stateBytes;
  if (entry.checkpoint)
    checkpointBytes_ += stateBytes;
  if (!entry.pins)
    entry.lastUsed = recency_.next();
  reindex(kvBlock, entry);
  ++publications_;
}

bool StateCache::publishToDisk(uint64_t kvBlock, const StateWriter &write,
                               const std::function<void()> &completion,
                               const std::function<bool()> &makeRoom, bool checkpoint) {
  if (!kv_.contains(kvBlock)) {
    throw std::invalid_argument("composite state KV block is unknown");
  }
  if (publications_ == std::numeric_limits<uint64_t>::max())
    throw std::overflow_error("composite state publication count overflowed");
  if (resident(kvBlock))
    throw std::logic_error("duplicate composite state key");
  // The state is on disk already; a second copy would add nothing.
  if (touchIfStored(kvBlock, checkpoint))
    return true;
  // A checkpoint replaces older copies like any state: it is the only
  // progress a suspended request keeps once the quota is full.
  std::unique_ptr<StateOffload> transfer = startWrite(write, completion, makeRoom);
  if (!transfer)
    return false;
  Entry &entry = publicationEntry(kvBlock, checkpoint);
  beginWrite(kvBlock, entry, std::move(transfer));
  if (!entry.pins)
    entry.lastUsed = recency_.next();
  reindex(kvBlock, entry);
  ++publications_;
  return true;
}

StateCheckpoint StateCache::checkpoint(uint64_t kvBlock) const noexcept {
  const auto found = entries_.find(kvBlock);
  if (found == entries_.end() || !found->second.checkpoint)
    return {};
  return {kvBlock, found->second.publication};
}

bool StateCache::retireCheckpoint(StateCheckpoint checkpoint) noexcept {
  const auto found = entries_.find(checkpoint.kvBlock);
  if (found == entries_.end() || !found->second.checkpoint ||
      found->second.publication != checkpoint.publication)
    return true;
  return erase(checkpoint.kvBlock, true).evicted;
}

void StateCache::makeOrdinary(uint64_t kvBlock, Entry &entry) {
  kv_.noteState(kvBlock);
  if (!entry.checkpoint)
    return;
  --checkpointEntries_;
  checkpointBytes_ -= entry.ram ? entry.ram->bytes() : 0;
  entry.checkpoint = false;
}

bool StateCache::contains(uint64_t kvBlock) const noexcept {
  const auto found = entries_.find(kvBlock);
  return found != entries_.end() && !found->second.invalid;
}

uint64_t StateCache::resumePoint() const noexcept {
  // Checkpoints can survive cancellation but remain disposable. Prefer an
  // ordinary state for speculative protection, regardless of recency.
  return ordinary_.newestId() ? ordinary_.newestId()
                                  : checkpoints_.newestId();
}

bool StateCache::resident(uint64_t kvBlock) const noexcept {
  const auto found = entries_.find(kvBlock);
  return found != entries_.end() && found->second.ram != nullptr;
}

std::optional<CacheEvictionCandidate>
StateCache::evictionCandidate(bool keepResumePoint, bool checkpoints) const noexcept {
  const uint64_t kept = keepResumePoint ? resumePoint() : 0;
  for (const auto candidate :
       {checkpoints ? checkpoints_.oldest() : std::nullopt, ordinary_.oldest()})
    if (candidate && candidate->id != kept)
      return candidate;
  return std::nullopt;
}

std::optional<CacheEvictionCandidate>
StateCache::diskCandidate(bool duplicate) const noexcept {
  return duplicate ? duplicates_.oldest() : diskOnly_.oldest();
}

StateEviction StateCache::reclaim(uint64_t kvBlock, std::function<void()> completion,
                                  const std::function<bool()> &makeRoom,
                                  bool waitForWrite) {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.pins || !found->second.ram)
    return {};
  Entry &entry = found->second;
  const uint64_t reclaimed = entry.ram->bytes();
  // One write at a time.
  const bool writable = !entry.disk && entry.ram->canOffload();
  if (writable && pending_ && waitForWrite)
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
  bytes_ -= reclaimed;
  if (entry.checkpoint)
    checkpointBytes_ -= reclaimed;
  entry.ram.reset();
  reindex(kvBlock, entry);
  return {true, reclaimed};
}

StateEviction StateCache::evict(uint64_t kvBlock) noexcept {
  return erase(kvBlock, false);
}

void StateCache::dropDisk(uint64_t kvBlock) {
  Entry &target = entry(kvBlock);
  if (!target.ram || !target.disk)
    throw std::logic_error("only a redundant disk copy is dropped");
  if (writing(kvBlock))
    throw std::logic_error("a disk copy being written cannot be dropped");
  discardDisk(target);
  reindex(kvBlock, target);
}

void StateCache::invalidate(uint64_t kvBlock, const CompositeState *state) noexcept {
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

void StateCache::invalidate(uint64_t kvBlock) noexcept {
  const auto found = entries_.find(kvBlock);
  if (found != entries_.end())
    invalidate(kvBlock, copy(found->second).get());
}

bool StateCache::promotable(uint64_t kvBlock, const CompositeState *source) const noexcept {
  const auto found = entries_.find(kvBlock);
  return found != entries_.end() && !found->second.invalid && !found->second.ram &&
         found->second.disk.get() == source;
}

void StateCache::promote(uint64_t kvBlock, const CompositeState *source,
                         std::shared_ptr<const CompositeState> state) {
  if (!promotable(kvBlock, source))
    return;
  if (!state || state->bytes() != source->bytes() ||
      state->residentBytes() != state->bytes())
    throw std::invalid_argument("invalid promoted state");
  if (state->bytes() > std::numeric_limits<uint64_t>::max() - bytes_)
    throw std::overflow_error("promoted state byte count overflowed");
  Entry &target = entry(kvBlock);
  target.ram = std::move(state);
  bytes_ += target.ram->bytes();
  if (target.checkpoint)
    checkpointBytes_ += target.ram->bytes();
  reindex(kvBlock, target);
  ++promotions_;
}

bool StateCache::pollOffload() {
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

StateEviction StateCache::erase(uint64_t kvBlock, bool retirement) noexcept {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.pins)
    return {};
  Entry &target = found->second;
  const uint64_t reclaimed = target.ram ? target.ram->bytes() : 0;
  bytes_ -= reclaimed;
  if (target.disk)
    diskBytes_ -= target.disk->bytes();
  if (target.checkpoint) {
    --checkpointEntries_;
    checkpointBytes_ -= reclaimed;
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

StateCacheSnapshot StateCache::snapshot() const noexcept {
  StateCacheSnapshot result;
  result.entries = static_cast<uint32_t>(std::min<uint64_t>(
      entries_.size(), std::numeric_limits<uint32_t>::max()));
  result.pinned = pinnedEntries_;
  result.bytes = bytes_;
  result.diskBytes = diskBytes_;
  result.offloads = offloads_;
  result.offloadFailures = offloadFailures_;
  result.invalidations = invalidations_;
  result.diskHits = diskHits_;
  result.promotions = promotions_;
  result.promotionsSkipped = promotionsSkipped_;
  result.hits = hits_;
  result.misses = misses_;
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

void StateCache::release(uint64_t kvBlock) noexcept {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || !found->second.pins)
    std::terminate();
  Entry &target = found->second;
  --target.pins;
  if (!target.pins) {
    if (!pinnedEntries_)
      std::terminate();
    --pinnedEntries_;
    target.lastUsed = recency_.next();
    reindex(kvBlock, target);
    if (target.invalid)
      static_cast<void>(evict(kvBlock));
  }
  kv_.releaseActive(kvBlock);
}

StateCache::Entry &StateCache::entry(uint64_t kvBlock) {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end())
    throw std::out_of_range("unknown composite state");
  return found->second;
}

StateCache::Entry &StateCache::entryFor(uint64_t kvBlock) {
  auto found = entries_.find(kvBlock);
  if (found != entries_.end())
    return found->second;
  Entry fresh;
  fresh.ramNode = RecencyOrder::allocate(kvBlock);
  fresh.diskNode = RecencyOrder::allocate(kvBlock);
  fresh.publication = publications_ + 1;
  Entry &placed = entries_.emplace(kvBlock, std::move(fresh)).first->second;
  kv_.countState(kvBlock, true);
  return placed;
}

StateCache::Entry &StateCache::publicationEntry(uint64_t kvBlock, bool checkpoint) {
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

std::unique_ptr<StateOffload> StateCache::startWrite(const StateWriter &write,
                                                     const std::function<void()> &completion,
                                                     const std::function<bool()> &makeRoom) {
  if (pending_)
    return {};
  std::unique_ptr<StateOffload> transfer = write(completion);
  while (!transfer && makeRoom && makeRoom())
    transfer = write(completion);
  return transfer;
}

void StateCache::beginWrite(uint64_t kvBlock, Entry &target,
                            std::unique_ptr<StateOffload> transfer) {
  target.disk = transfer->state();
  diskBytes_ += target.disk->bytes();
  pending_.emplace(PendingOffload{kvBlock, std::move(transfer)});
  ++offloads_;
}

// RAM copies wait for eviction in one order per class; disk copies wait for
// replacement as redundant copies or as the only copy. A pinned or invalid
// entry, or a copy being written, is in no order.
void StateCache::reindex(uint64_t kvBlock, Entry &target) noexcept {
  unlink(target);
  if (target.pins || target.invalid)
    return;
  if (target.ram)
    (target.checkpoint ? checkpoints_ : ordinary_).link(target.ramNode, target.lastUsed, kvBlock);
  if (target.disk && !writing(kvBlock))
    (target.ram ? duplicates_ : diskOnly_).link(target.diskNode, target.lastUsed, kvBlock);
}

void StateCache::unlink(Entry &target) noexcept {
  if (target.ramNode.linked())
    RecencyOrder::unlink(target.ramNode);
  if (target.diskNode.linked())
    RecencyOrder::unlink(target.diskNode);
}

void StateCache::discardDisk(Entry &target) noexcept {
  if (!target.disk)
    return;
  diskBytes_ -= target.disk->bytes();
  target.disk.reset();
}

} // namespace splash::engine
