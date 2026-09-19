#include "engine/Cache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {

Cache::Cache(KvPool &pool, CacheNamespace cacheNamespace, model::KvTier *kvTier)
    : pool_(pool), tier_(kvTier), kv_(pool, cacheNamespace, recency_),
      states_(kv_, recency_), makeRoom_([this] { return freeDiskSpace(); }) {}

void Cache::beginRequest(uint64_t requestId) {
  if (!requestId)
    throw std::invalid_argument("invalid request id");
  auto [_, inserted] = requests_.emplace(requestId, Request{});
  if (!inserted)
    throw std::invalid_argument("duplicate request id");
}

void Cache::endRequest(uint64_t requestId) {
  auto found = requests_.find(requestId);
  if (found == requests_.end())
    return;
  Request &active = found->second;
  for (uint32_t page : active.pages)
    pool_.releasePage(page, false);
  if (!active.cachedBlocks.empty())
    kv_.releaseActive(active.cachedBlocks.back());
  // Refresh used states within their class. Ordinary states remain newer
  // than the finished KV tail; checkpoints retain their lower priority.
  for (uint64_t block : active.cachedBlocks)
    states_.touch(block);
  if (active.pendingRestores) {
    for (auto &[_, restore] : restores_)
      std::erase(restore.waiters, requestId);
  }
  requests_.erase(found);
}

CacheLookup Cache::lookup(std::span<const uint32_t> prompt,
                          std::span<const ImageSpan> images) {
  CacheLookup result;
  if (prompt.empty())
    return result;

  // Leave one real input token to regenerate request-specific anchor logits.
  const size_t maximumBlocks = (prompt.size() - 1) / KvCache::pageTokens;
  std::vector<uint64_t> matchedKvBlocks;
  matchedKvBlocks.reserve(maximumBlocks);
  uint64_t parent = 0;
  for (size_t index = 0; index < maximumBlocks; ++index) {
    const size_t begin = index * KvCache::pageTokens;
    auto match =
        kv_.find(parent, prompt.subspan(begin, KvCache::pageTokens),
                 blockImageIdentity(begin, KvCache::pageTokens, images));
    if (!match)
      break;
    parent = match->id;
    matchedKvBlocks.push_back(parent);
  }
  if (matchedKvBlocks.empty())
    return result;

  kv_.touch(parent);
  result.kvBoundary =
      static_cast<uint32_t>(matchedKvBlocks.size() * KvCache::pageTokens);
  result.state = states_.acquireDeepest(matchedKvBlocks);
  const uint64_t stateBlock = result.state ? result.state->kvBlock() : 0;
  for (auto block = matchedKvBlocks.rbegin(); block != matchedKvBlocks.rend(); ++block) {
    if (*block == stateBlock) break;
    if (kv_.hadState(*block) && !states_.contains(*block)) {
      result.lostState = true;
      break;
    }
  }
  if (result.state) {
    const uint32_t depth = kv_.chainLength(result.state->kvBlock());
    for (uint32_t index = 0; index < depth; ++index)
      result.diskBlocks += kv_.page(matchedKvBlocks[index]) == KvCache::noPage;
  }
  return result;
}

void Cache::recordLookup(const CacheLookup &result) {
  ++lookup_.lookups;
  lookup_.kvHitTokens += result.kvBoundary;
  lookup_.kvDiskHitTokens += uint64_t{result.diskBlocks} * KvCache::pageTokens;
  lookup_.stateHitTokens += result.resumeBoundary();
  states_.recordLookup(result.state.has_value(),
                       result.state && !result.state->state()->residentBytes());
  if (result.junctionBoundary())
    ++lookup_.lazyJunctions;
  if (result.lostState)
    ++lookup_.lostStateMisses;
}

PageTableView Cache::pageTable(uint64_t requestId) const {
  const Request &active = request(requestId);
  return {active.pages, active.pageTableRevision};
}

uint64_t Cache::publishCommittedBlocks(uint64_t requestId,
                                       std::span<const uint32_t> exactTokens,
                                       uint32_t committedTokens,
                                       std::span<const ImageSpan> images) {
  Request &active = request(requestId);
  if (committedTokens > exactTokens.size()) {
    throw std::invalid_argument("committed KV exceeds exact token history");
  }
  const uint32_t fullBlocks = committedTokens / KvCache::pageTokens;
  const uint64_t requiredPages =
      (uint64_t{committedTokens} + KvCache::pageTokens - 1) / KvCache::pageTokens;
  if (requiredPages > active.pages.size()) {
    throw std::logic_error("committed KV has no physical request page");
  }
  if (fullBlocks < active.cachedBlocks.size())
    throw std::logic_error("committed KV history moved backwards");
  while (active.cachedBlocks.size() < fullBlocks) {
    const uint32_t logical = static_cast<uint32_t>(active.cachedBlocks.size());
    const uint64_t parent = logical ? active.cachedBlocks.back() : 0;
    const uint32_t begin = logical * KvCache::pageTokens;
    auto inserted =
        kv_.insert(parent, exactTokens.subspan(begin, KvCache::pageTokens),
                   active.pages[logical],
                   blockImageIdentity(begin, KvCache::pageTokens, images));
    const uint32_t writerPage = active.pages[logical];
    const bool replacePage = inserted.physicalPage != writerPage;
    if (replacePage)
      pool_.retainPage(inserted.physicalPage, false);
    try {
      kv_.retainActive(inserted.id);
      try {
        active.cachedBlocks.push_back(inserted.id);
      } catch (...) {
        kv_.releaseActive(inserted.id);
        throw;
      }
    } catch (...) {
      if (replacePage)
        pool_.releasePage(inserted.physicalPage, false);
      throw;
    }
    if (replacePage) {
      active.pages[logical] = inserted.physicalPage;
      ++active.pageTableRevision;
      pool_.releasePage(writerPage, false);
    }
    if (parent)
      kv_.releaseActive(parent);
  }
  return active.cachedBlocks.empty() ? 0 : active.cachedBlocks.back();
}

uint64_t Cache::blockAt(uint64_t requestId, uint32_t boundary) const {
  if (!boundary || boundary % KvCache::pageTokens) {
    throw std::invalid_argument("state boundary is not a complete KV block");
  }
  const Request &active = request(requestId);
  const size_t index = boundary / KvCache::pageTokens - 1;
  if (index >= active.cachedBlocks.size()) {
    throw std::out_of_range("state boundary KV block is not published");
  }
  return active.cachedBlocks[index];
}

bool Cache::reuseCompositeState(uint64_t kvBlock, bool checkpoint) {
  const bool reused = states_.touchIfResident(kvBlock, checkpoint);
  if (reused && !checkpoint)
    kv_.noteState(kvBlock);
  return reused;
}

void Cache::publishCompositeState(uint64_t kvBlock,
                                  std::shared_ptr<const CompositeState> state,
                                  bool checkpoint) {
  states_.publish(kvBlock, std::move(state), checkpoint);
  if (!checkpoint)
    kv_.noteState(kvBlock);
}

bool Cache::publishStateToDisk(uint64_t kvBlock, const StateWriter &write, bool checkpoint) {
  if (!states_.publishToDisk(kvBlock, write, completionNotifier_, makeRoom_, checkpoint))
    return false;
  if (!checkpoint)
    kv_.noteState(kvBlock);
  return true;
}

StateCheckpoint Cache::checkpointState(uint64_t kvBlock) const noexcept {
  return states_.checkpoint(kvBlock);
}

bool Cache::retireCheckpointState(StateCheckpoint checkpoint) noexcept {
  return states_.retireCheckpoint(checkpoint);
}

// Page admission: pages come from the free list or from evicted resident
// leaves; a page that must be written first comes back when its copy lands.

TokenAdmission Cache::ensureTokens(uint64_t requestId, uint64_t tokenCount) {
  Request &active = request(requestId);
  const uint64_t needed64 =
      (tokenCount + KvCache::pageTokens - 1) / KvCache::pageTokens;
  if (needed64 > std::numeric_limits<uint32_t>::max()) {
    return {KvPageAcquireFailure::LogicalCapacity,
            std::numeric_limits<uint32_t>::max(), pool_.freePageCount()};
  }
  const uint32_t needed = static_cast<uint32_t>(needed64);
  if (needed <= active.pages.size())
    return {};
  std::vector<uint32_t> acquired;
  if (const TokenAdmission admission =
          admitPages(needed - static_cast<uint32_t>(active.pages.size()), acquired);
      !admission.granted())
    return admission;
  try {
    active.pages.insert(active.pages.end(), acquired.begin(), acquired.end());
  } catch (...) {
    for (uint32_t page : acquired)
      pool_.releasePage(page, false);
    throw;
  }
  ++active.pageTableRevision;
  return {};
}

TokenAdmission Cache::admitPages(uint32_t count, std::vector<uint32_t> &pages) {
  const Shortfall shortfall = makeLogicalPages(count);
  if (shortfall != Shortfall::Covered) {
    return {shortfall == Shortfall::Pending ? KvPageAcquireFailure::Pending
                                            : KvPageAcquireFailure::LogicalCapacity,
            count, pool_.freePageCount()};
  }
  KvPageAcquisition acquired = pool_.acquirePages(count, false);
  if (!acquired.granted()) {
    // The pool could not map backing for its free pages. Demoted pages carry
    // their backing back: wait once those on their way cover what the free
    // backed pages do not, and until then reclaim more.
    const uint32_t missing = count - std::min(count, pool_.freeResidentPageCount());
    const bool covered = pendingPages_ > 0 && pendingPages_ >= missing;
    return {covered ? KvPageAcquireFailure::Pending : acquired.failure,
            count, pool_.freePageCount(), acquired.allocationFailure};
  }
  pages = std::move(acquired.pages);
  return {};
}

Cache::Shortfall Cache::makeLogicalPages(uint32_t count) {
  while (pool_.freePageCount() < count) {
    // Pages already on their way back cover the shortfall: wait for them
    // rather than demoting more.
    if (pendingPages_ >= count - pool_.freePageCount())
      return Shortfall::Pending;
    switch (evictOneKvBlock()) {
    case Eviction::Evicted:
      continue;
    case Eviction::Pending:
      return Shortfall::Pending;
    case Eviction::None:
      // Restores in flight hold their blocks out of the orders; when they
      // land those blocks are leaves with disk copies, free to give up.
      return transfersInFlight() ? Shortfall::Pending : Shortfall::Exhausted;
    }
  }
  return Shortfall::Covered;
}

Cache::Eviction Cache::evictOneKvBlock() {
  uint64_t previous = 0;
  while (auto candidate = kv_.evictionCandidate(previous)) {
    previous = candidate->id;
    switch (reclaimKvLeaf(candidate->id)) {
    case LeafReclaim::Started:
      return Eviction::Evicted;
    case LeafReclaim::Pending:
      // The ring, the quota or the state write is busy for every leaf
      // alike; scanning on would only find the same answer.
      return Eviction::Pending;
    case LeafReclaim::Impossible:
      break;
    }
  }
  return Eviction::None;
}

// Reclaim: one victim at a time in the shared recency order, states and
// resident KV leaves alike.

uint64_t Cache::reclaimCache(uint64_t targetBytes, bool evictAll) {
  // Empty backing that is waiting behind an in-flight release will satisfy
  // part of the target by itself; evicting more cache now would only
  // discard reusable prefixes without returning memory any sooner. Pages
  // whose copies are being written count the same way.
  if (releaseDeferred())
    return 0;
  uint64_t released = reclaimEmptyExtents();
  auto needsMore = [&] { return evictAll || released + pendingBytes() < targetBytes; };
  while (needsMore() && !releaseDeferred()) {
    const CacheReclaimResult result = reclaimOne();
    if (!result.madeProgress)
      break;
    released += result.reclaimedBytes;
  }
  return released;
}

CacheReclaimResult Cache::reclaimOne(CacheReclaimMode mode) {
  if (mode == CacheReclaimMode::ReleaseBacking) {
    if (releaseDeferred())
      return {};
    if (const uint64_t bytes = reclaimEmptyExtents())
      return {true, bytes};
  }

  // Oldest first across both kinds. A state whose write must wait for the
  // one in flight stays, as does a KV leaf the ring cannot take now; the
  // other kind may still give, and the next pass takes what waited.
  const std::optional<CacheEvictionCandidate> state =
      states_.evictionCandidate();
  const bool checkpoint = state && states_.checkpoint(state->id);
  bool stateOpen = state.has_value();
  bool kvOpen = !checkpoint;
  bool pending = false;
  std::optional<CacheEvictionCandidate> kv = kvOpen ? oldestKvLeaf(0) : std::nullopt;
  while (stateOpen || (kvOpen && kv)) {
    if (stateOpen && (!kvOpen || !kv || state->lastUsed <= kv->lastUsed)) {
      const StateEviction eviction =
          states_.reclaim(state->id, completionNotifier_, makeRoom_, true);
      if (eviction.evicted)
        return {true, eviction.reclaimedBytes};
      if (!eviction.pending)
        throw std::logic_error("state eviction candidate became pinned");
      stateOpen = false;
      pending = pending || transfersInFlight();
      continue;
    }
    switch (reclaimKvLeaf(kv->id)) {
    case LeafReclaim::Started:
      return {true, mode == CacheReclaimMode::ReleaseBacking ? reclaimEmptyExtents() : 0};
    case LeafReclaim::Pending:
      kvOpen = false;
      pending = pending || transfersInFlight();
      break;
    case LeafReclaim::Impossible:
      kv = oldestKvLeaf(kv->id);
      break;
    }
  }
  // Nothing to reclaim now; whatever is in flight still comes back.
  return {false, 0, pending || transfersInFlight()};
}

bool Cache::reclaimOneState(bool checkpointsOnly) {
  const std::optional<CacheEvictionCandidate> state =
      states_.evictionCandidate();
  return state && (!checkpointsOnly || states_.checkpoint(state->id)) &&
         states_.reclaim(state->id, completionNotifier_, makeRoom_).evicted;
}

std::optional<CacheEvictionCandidate> Cache::oldestKvLeaf(uint64_t after) const {
  while (auto candidate = kv_.evictionCandidate(after)) {
    after = candidate->id;
    if (!states_.resident(candidate->id))
      return candidate;
  }
  return std::nullopt;
}

Cache::LeafReclaim Cache::reclaimKvLeaf(uint64_t block) {
  // A state in RAM goes first: to disk when the tier takes it, away
  // otherwise. A state already on disk costs nothing and stays; one whose
  // write must wait keeps its leaf until then.
  if (states_.resident(block)) {
    const StateEviction eviction =
        states_.reclaim(block, completionNotifier_, makeRoom_, true);
    if (!eviction.evicted)
      return eviction.pending ? LeafReclaim::Pending : LeafReclaim::Impossible;
  }
  if (kv_.slot(block)) {
    kv_.dropPage(block);
    return LeafReclaim::Started;
  }
  const bool keep = states_.contains(block) || kv_.hasDiskChildren(block);
  if (keep) {
    const LeafReclaim demotion = demoteKv(block);
    // A leaf its disk subtree depends on stays in RAM until the tier can
    // write it; dropping it would orphan every copy below it.
    if (demotion != LeafReclaim::Impossible || kv_.hasDiskChildren(block))
      return demotion;
  }
  static_cast<void>(states_.evict(block));
  kv_.erase(block);
  return LeafReclaim::Started;
}

uint64_t Cache::reclaimEmptyExtents() {
  const uint64_t before = pool_.residentBackingBytes();
  static_cast<void>(pool_.reclaimEmptyExtents(false));
  const uint64_t after = pool_.residentBackingBytes();
  return before >= after ? before - after : 0;
}

bool Cache::transfersInFlight() const noexcept {
  return pendingPages_ > 0 || !restores_.empty() || !demotions_.empty() ||
         states_.writing();
}

uint64_t Cache::pendingBytes() const noexcept {
  return uint64_t{pendingPages_} * pool_.bytesPerPage();
}

bool Cache::releaseDeferred() const noexcept {
  return pool_.reclaimableExtentCount() > 0 && !pool_.releaseReady();
}

bool Cache::releasePending() const noexcept {
  return !pool_.releaseReady();
}

uint64_t Cache::releaseGeneration() const noexcept {
  return pool_.releaseGeneration();
}

void Cache::releaseUnusedKvBacking() {
  while (pool_.reclaimEmptyExtents(true))
    pool_.awaitRelease();
}

// Disk tier: restores and demotions in flight, the quota they draw on, and
// states promoted back into RAM.

TokenAdmission Cache::restoreRequest(uint64_t requestId, const CacheLookup &lookup) {
  Request &active = request(requestId);
  if (!active.pages.empty() || !active.cachedBlocks.empty()) {
    throw std::logic_error("restore target already owns KV pages");
  }
  if (!lookup.state)
    return {};
  KvCache::Chain chain = kv_.chain(lookup.state->kvBlock());
  const uint32_t missing = static_cast<uint32_t>(
      std::count(chain.pages.begin(), chain.pages.end(), KvCache::noPage));
  std::vector<uint32_t> fresh;
  if (missing) {
    if (!tier_)
      throw std::logic_error("disk-only KV block without a disk tier");
    if (const TokenAdmission admission = admitPages(missing, fresh); !admission.granted())
      return admission;
  }
  std::vector<uint32_t> retained;
  retained.reserve(chain.pages.size());
  kv_.retainActive(chain.blocks.back());
  try {
    size_t next = 0;
    for (size_t index = 0; index < chain.blocks.size(); ++index) {
      const uint64_t block = chain.blocks[index];
      if (chain.pages[index] == KvCache::noPage) {
        // Root first, so every restored block finds its parent resident.
        chain.pages[index] = fresh[next++];
        kv_.adoptPage(block, chain.pages[index]);
        startRestore(block);
      } else {
        pool_.retainPage(chain.pages[index], false);
        retained.push_back(chain.pages[index]);
      }
      if (auto restore = restores_.find(block); restore != restores_.end()) {
        restore->second.waiters.push_back(requestId);
        ++active.pendingRestores;
      }
    }
  } catch (...) {
    for (uint32_t page : retained)
      pool_.releasePage(page, false);
    for (uint32_t page : fresh)
      pool_.releasePage(page, false);
    kv_.releaseActive(chain.blocks.back());
    active.pendingRestores = 0;
    throw;
  }
  active.pages = std::move(chain.pages);
  active.cachedBlocks = std::move(chain.blocks);
  ++active.pageTableRevision;
  return {};
}

KvRestoreStatus Cache::kvRestoreStatus(uint64_t requestId) const {
  const Request &active = request(requestId);
  if (active.restoreFailed)
    return KvRestoreStatus::Failed;
  return active.pendingRestores ? KvRestoreStatus::Pending : KvRestoreStatus::None;
}

void Cache::startRestore(uint64_t block) {
  kv_.setTransferring(block, true);
  Restore &restore = restores_[block];
  restore.transfer =
      tier_->restore(kv_.slot(block), kv_.page(block), completionNotifier_);
}

void Cache::promoteState(const CacheLookup &lookup, StateRestore &transfer) {
  if (!lookup.state) return;
  const auto block = lookup.state->kvBlock();
  const auto *source = lookup.state->state().get();
  if (!states_.promotable(block, source)) return;
  // Promotion is optional and uses the ordinary snapshot admission/reclaimer.
  // The admitted request can run even when no cache buffer is available.
  try {
    auto state = transfer.snapshot();
    if (!state) {
      // The restored state is the most recently used one; the oldest RAM
      // copy makes room for it unless its write has to wait.
      const auto victim = states_.evictionCandidate();
      if (victim &&
          states_.reclaim(victim->id, completionNotifier_, makeRoom_, true).evicted)
        state = transfer.snapshot();
    }
    if (state) states_.promote(block, source, std::move(state));
    else states_.promotionSkipped();
  } catch (const std::exception &) {
    states_.promotionSkipped();
  }
}

Cache::LeafReclaim Cache::demoteKv(uint64_t block) {
  if (!tier_ || !tier_->writable())
    return LeafReclaim::Impossible;
  // A quota or a ring that only transfers in flight can release is worth
  // waiting for; one that nothing will free is not, and the leaf goes.
  std::shared_ptr<model::KvDiskSlot> slot = acquireDiskSlot();
  if (!slot)
    return transfersInFlight() ? LeafReclaim::Pending : LeafReclaim::Impossible;
  auto transfer = tier_->demote(kv_.page(block), slot, completionNotifier_);
  if (!transfer) {
    ++kvTier_.demotionsRefused;
    return transfersInFlight() ? LeafReclaim::Pending : LeafReclaim::Impossible;
  }
  kv_.setSlot(block, std::move(slot));
  kv_.setTransferring(block, true);
  ++pendingPages_;
  demotions_.push_back({block, std::move(transfer)});
  return LeafReclaim::Started;
}

std::shared_ptr<model::KvDiskSlot> Cache::acquireDiskSlot() {
  for (;;) {
    if (auto slot = tier_->acquireSlot())
      return slot;
    if (!freeDiskSpace())
      return {};
  }
}

bool Cache::freeDiskSpace() {
  const auto older = [](const std::optional<CacheEvictionCandidate> &left,
                        const std::optional<CacheEvictionCandidate> &right) {
    return left && (!right || left->lastUsed < right->lastUsed);
  };
  // A redundant copy loses nothing: its data stays in RAM.
  const auto kvDuplicate = kv_.diskCandidate(true);
  const auto stateDuplicate = states_.diskCandidate(true);
  if (kvDuplicate || stateDuplicate) {
    if (older(stateDuplicate, kvDuplicate))
      states_.dropDisk(stateDuplicate->id);
    else
      kv_.setSlot(kvDuplicate->id, nullptr);
    return true;
  }
  const auto kvLeaf = kv_.diskCandidate(false);
  const auto stateOnly = states_.diskCandidate(false);
  if (!kvLeaf && !stateOnly)
    return false;
  if (older(stateOnly, kvLeaf)) {
    static_cast<void>(states_.evict(stateOnly->id));
    return true;
  }
  static_cast<void>(states_.evict(kvLeaf->id));
  kv_.erase(kvLeaf->id);
  return true;
}

bool Cache::pollTransfers() {
  bool progressed = states_.pollOffload();
  if (!tier_)
    return progressed;
  tier_->poll();
  for (auto entry = restores_.begin(); entry != restores_.end();) {
    auto &[block, restore] = *entry;
    if (!restore.transfer)
      restore.transfer =
          tier_->restore(kv_.slot(block), kv_.page(block), completionNotifier_);
    if (!restore.transfer || !restore.transfer->ready()) {
      ++entry;
      continue;
    }
    const bool restored = restore.transfer->finish();
    kv_.setTransferring(block, false);
    if (restored)
      ++kvTier_.restores;
    else
      ++kvTier_.restoreFailures;
    for (uint64_t requestId : restore.waiters) {
      auto waiter = requests_.find(requestId);
      if (waiter == requests_.end())
        continue;
      --waiter->second.pendingRestores;
      if (!restored)
        waiter->second.restoreFailed = true;
    }
    if (!restored) {
      // The block leaves with its last user; so does any state it held.
      states_.invalidate(block);
      kv_.poison(block);
    }
    entry = restores_.erase(entry);
    progressed = true;
  }
  for (auto demotion = demotions_.begin(); demotion != demotions_.end();) {
    if (!demotion->transfer->ready()) {
      ++demotion;
      continue;
    }
    const uint64_t block = demotion->block;
    const bool written = demotion->transfer->finish();
    demotion = demotions_.erase(demotion);
    --pendingPages_;
    kv_.setTransferring(block, false);
    if (!written) {
      ++kvTier_.demotionFailures;
      kv_.setSlot(block, nullptr);
    } else {
      ++kvTier_.demotions;
      // A request that matched the block meanwhile keeps its page; the copy
      // makes the next reclaim of the leaf free.
      if (kv_.residentLeaf(block))
        kv_.dropPage(block);
    }
    progressed = true;
  }
  return progressed;
}

CacheSnapshot Cache::snapshot() const {
  KvTierSnapshot tier = kvTier_;
  tier.pendingPages = pendingPages_;
  tier.diskBlocks = kv_.snapshot().diskBlocks;
  if (tier_) {
    tier.capacityBytes = tier_->capacityBytes();
    tier.usedBytes = tier_->usedBytes();
    tier.diskBytes = uint64_t{tier.diskBlocks} * tier_->slotBytes();
  }
  return {pool_.snapshot(),
          kv_.snapshot(),
          states_.snapshot(),
          tier,
          lookup_,
          static_cast<uint32_t>(requests_.size())};
}

Cache::Request &Cache::request(uint64_t requestId) {
  auto found = requests_.find(requestId);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

const Cache::Request &Cache::request(uint64_t requestId) const {
  auto found = requests_.find(requestId);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

} // namespace splash::engine
