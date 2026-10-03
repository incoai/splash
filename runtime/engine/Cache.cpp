#include "engine/Cache.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {
namespace {

// The spans that can still overlap the block at blockBegin or a later one:
// sorted and disjoint, a span that ends before a block ends before every
// later block too.
std::span<const ImageSpan> spansFrom(std::span<const ImageSpan> images, uint64_t blockBegin) {
  const auto ended = [&](const ImageSpan &span) { return span.end() <= blockBegin; };
  return {std::partition_point(images.begin(), images.end(), ended), images.end()};
}

} // namespace

Cache::Cache(KvPool &pool, KvTier *kvTier,
             std::shared_ptr<const model::DiskBudget> diskBudget)
    : pool_(pool), tier_(kvTier), diskBudget_(std::move(diskBudget)),
      kv_(pool, recency_),
      states_(kv_, recency_, [this](bool inUse) { return freeDiskSpace(inUse); }) {}

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
  if (active.pendingRestores) {
    for (auto &[_, restore] : restores_)
      std::erase(restore.waiters, requestId);
    // Unsubmitted reads have no IO to drain. Return their pages leaf first;
    // shared restores and already submitted transfers retain their ownership.
    // So does a block another request published a state in RAM on meanwhile:
    // a state in RAM sits on resident KV, so its read goes on.
    for (auto block = active.cachedBlocks.rbegin(); block != active.cachedBlocks.rend(); ++block) {
      const auto restore = restores_.find(*block);
      if (restore != restores_.end() && !restore->second.transfer &&
          restore->second.waiters.empty() && !states_.stateResident(*block) &&
          kv_.abandonRestore(*block))
        restores_.erase(restore);
    }
  }
  requests_.erase(found);
  dropPoisoned();
}

size_t Cache::extendMatch(std::vector<uint64_t> &blocks, std::span<const uint32_t> prompt,
                          std::span<const ImageSpan> images) const {
  if (prompt.empty())
    return 0;
  // Leave one real input token to regenerate request-specific anchor logits.
  const size_t maximumBlocks = (prompt.size() - 1) / KvCache::pageTokens;
  blocks.reserve(maximumBlocks);
  size_t hashed = 0;
  for (size_t index = blocks.size(); index < maximumBlocks; ++index) {
    const size_t begin = index * KvCache::pageTokens;
    const uint64_t parent = blocks.empty() ? 0 : blocks.back();
    images = spansFrom(images, begin);
    ++hashed;
    auto match = kv_.find(parent, prompt.subspan(begin, KvCache::pageTokens),
                          blockImageIdentity(begin, KvCache::pageTokens, images));
    if (!match)
      break;
    blocks.push_back(match->id);
  }
  return hashed;
}

uint32_t Cache::stateTokens(std::span<const uint64_t> blocks) const {
  for (size_t i = blocks.size(); i > 0; --i) {
    if (states_.contains(blocks[i - 1]))
      return static_cast<uint32_t>(i * KvCache::pageTokens);
  }
  return 0;
}

CacheProbe Cache::probe(std::span<const uint32_t> prompt,
                        std::span<const ImageSpan> images) {
  CacheProbe result;
  result.owner_ = this;
  result.kvGeneration_ = kv_.generation();
  lookup_.probeHashedBlocks += extendMatch(result.blocks_, prompt, images);
  result.cachedTokens_ = stateTokens(result.blocks_);
  return result;
}

void Cache::refresh(CacheProbe &probe, std::span<const uint32_t> prompt,
                    std::span<const ImageSpan> images) {
  if (probe.owner_ != this)
    throw std::logic_error("a scheduling probe belongs to another cache");
  if (probe.kvGeneration_ != kv_.generation()) {
    // See KvCache's generation: the chain holds up to its first block that
    // no longer matches.
    probe.blocks_.erase(std::find_if(probe.blocks_.begin(), probe.blocks_.end(),
                                     [&](uint64_t block) { return !kv_.matchable(block); }),
                        probe.blocks_.end());
    lookup_.probeHashedBlocks += extendMatch(probe.blocks_, prompt, images);
    probe.kvGeneration_ = kv_.generation();
  }
  // States come and go without a change of the KV graph.
  probe.cachedTokens_ = stateTokens(probe.blocks_);
}

CacheLookup Cache::lookup(std::span<const uint32_t> prompt,
                          std::span<const ImageSpan> images,
                          const CacheProbe *probe) {
  CacheLookup result;
  std::vector<uint64_t> fallback;
  std::span<const uint64_t> blocks;
  if (probe && probe->owner_ == this && probe->kvGeneration_ == kv_.generation()) {
    blocks = probe->blocks_;
  } else {
    extendMatch(fallback, prompt, images);
    blocks = fallback;
  }
  if (!blocks.empty()) {
    kv_.touch(blocks.back());
    // A chain with a disk suffix keeps its resident boundary as warm as its
    // tip: a restore adopts the disk blocks below it.
    const auto resident =
        std::find_if(blocks.rbegin(), blocks.rend(), [&](uint64_t block) {
          return kv_.page(block) != KvCache::noPage;
        });
    if (resident != blocks.rbegin() && resident != blocks.rend())
      kv_.touch(*resident);
    result.kvBoundary = static_cast<uint32_t>(blocks.size() * KvCache::pageTokens);
    result.state = states_.acquireDeepest(blocks);
    if (result.kvBoundary > result.resumeBoundary() && kv_.stateBelow(blocks.back()))
      result.junctionBoundary = result.kvBoundary;
  }
  const uint64_t stateBlock = result.state ? result.state->kvBlock() : 0;
  for (auto block = blocks.rbegin(); block != blocks.rend(); ++block) {
    if (*block == stateBlock) break;
    if (kv_.hadState(*block) && !states_.contains(*block)) {
      result.lostState = true;
      break;
    }
  }
  return result;
}

void Cache::recordLookup(const CacheLookup &result) {
  lookup_.kvHitTokens += result.kvBoundary;
  if (result.state && !result.state->state()->residentBytes())
    ++lookup_.stateDiskHits;
  if (result.junctionBoundary)
    ++lookup_.lazyJunctions;
  if (result.lostState)
    ++lookup_.lostStateMisses;
}

PageTableView Cache::pageTable(uint64_t requestId) const {
  const Request &active = request(requestId);
  return {active.pages, active.pageTableRevision, active.firstChangedPage};
}

void Cache::pagesChanged(Request &active, uint32_t first) noexcept {
  ++active.pageTableRevision;
  active.firstChangedPage = first;
}

void Cache::publishCommittedBlocks(uint64_t requestId,
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
    images = spansFrom(images, begin);
    auto inserted =
        kv_.insert(parent, exactTokens.subspan(begin, KvCache::pageTokens),
                   active.pages[logical],
                   blockImageIdentity(begin, KvCache::pageTokens, images));
    const uint32_t writerPage = active.pages[logical];
    const bool replacePage = inserted.physicalPage != writerPage;
    if (replacePage)
      pool_.retainPage(inserted.physicalPage, false);
    kv_.retainActive(inserted.id);
    active.cachedBlocks.push_back(inserted.id);
    if (replacePage) {
      active.pages[logical] = inserted.physicalPage;
      pagesChanged(active, logical);
      pool_.releasePage(writerPage, false);
    }
    if (parent)
      kv_.releaseActive(parent);
  }
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
  return states_.reuseCompositeState(kvBlock, checkpoint);
}

bool Cache::reuseStoredState(uint64_t kvBlock, bool checkpoint) {
  return states_.reuseStoredState(kvBlock, checkpoint);
}

void Cache::publishCompositeState(uint64_t kvBlock,
                                  std::shared_ptr<const CompositeState> state,
                                  bool checkpoint) {
  states_.publishCompositeState(kvBlock, std::move(state), checkpoint);
}

bool Cache::publishStateToDisk(uint64_t kvBlock, const StateWriter &write, bool checkpoint) {
  return states_.publishStateToDisk(kvBlock, write, checkpoint);
}

StateCheckpoint Cache::checkpointState(uint64_t kvBlock) const noexcept {
  return states_.checkpointState(kvBlock);
}

bool Cache::stateResident(uint64_t kvBlock) const noexcept {
  return states_.stateResident(kvBlock);
}

bool Cache::retireCheckpointState(StateCheckpoint checkpoint) noexcept {
  return states_.retireCheckpointState(checkpoint);
}

// Page admission: pages come from the pool, which grows by an extent when
// the governor admits one. Making room is the caller's reclaim; a page that
// must be written first comes back when its copy lands.

TokenAdmission Cache::ensureTokens(uint64_t requestId, uint64_t tokenCount) {
  Request &active = request(requestId);
  const uint64_t needed64 =
      (tokenCount + KvCache::pageTokens - 1) / KvCache::pageTokens;
  if (needed64 > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("KV target exceeds the page id range");
  const uint32_t needed = static_cast<uint32_t>(needed64);
  const auto previousSize = static_cast<uint32_t>(active.pages.size());
  if (needed <= previousSize)
    return {};
  std::vector<uint32_t> acquired;
  if (const TokenAdmission admission = admitPages(needed - previousSize, acquired);
      !admission.granted())
    return admission;
  active.pages.insert(active.pages.end(), acquired.begin(), acquired.end());
  pagesChanged(active, previousSize);
  return {};
}

TokenAdmission Cache::admitPages(uint32_t count, std::vector<uint32_t> &pages) {
  KvPageAcquisition acquired = pool_.acquirePages(count);
  if (!acquired.granted()) {
    // Demoted pages free theirs when their copies land: wait once those on
    // their way cover what the free pages do not, and until then reclaim.
    const uint32_t missing = count - std::min(count, pool_.freePageCount());
    const uint32_t pending = pendingPages();
    const bool covered = pending > 0 && pending >= missing;
    return {covered ? TokenAdmissionFailure::Pending : TokenAdmissionFailure::Denied,
            count, pool_.freePageCount(), acquired.failure};
  }
  pages = std::move(acquired.pages);
  return {};
}

// Reclaim: memory returns as whole extents. Free pages go first, an empty
// extent as it is and the pages scattered over the others once they fill
// one; then one victim at a time: KV no state restores through, then the
// shared recency order, states and resident KV leaves alike.

uint64_t Cache::releaseEmptyExtents(bool keepRunway) {
  return pool_.reclaimEmptyExtents(keepRunway, std::numeric_limits<uint32_t>::max());
}

CacheReclaimResult Cache::evictAll() {
  uint64_t released = releaseEmptyExtents(false);
  bool evicted = false;
  // Everything unpinned goes before anything moves.
  while (evictOne(ReclaimClass::InUse, false, false).madeProgress)
    evicted = true;
  do {
    released += releaseEmptyExtents(false);
  } while (compactExtent());
  return {.madeProgress = evicted || released != 0, .reclaimedBytes = released};
}

CacheReclaimResult Cache::reclaimOne(CacheReclaimMode mode, ReclaimClass upTo,
                                     bool keepResumePoint, bool keepRunway) {
  const bool release = mode == CacheReclaimMode::ReleaseExtents;
  if (release) {
    if (const CacheReclaimResult released = releaseExtent(keepRunway);
        released.madeProgress)
      return released;
  }
  CacheReclaimResult evicted =
      evictOne(upTo, keepResumePoint, mode == CacheReclaimMode::ReusePages);
  if (evicted.madeProgress && release)
    evicted.reclaimedBytes += pool_.reclaimEmptyExtents(keepRunway, 1);
  return evicted;
}

CacheReclaimResult Cache::evictOne(ReclaimClass upTo, bool keepResumePoint, bool pagesOnly) {
  // Disposable checkpoints go first, except in a page scan, to which their
  // buffers give no page; one whose write must wait for the one in flight
  // stays and holds back nothing else.
  if (const auto oldest = states_.checkpointCandidate(keepResumePoint); oldest && !pagesOnly) {
    if (states_.reclaim(oldest->id, StateCache::Unwritten::Wait).evicted)
      return {.madeProgress = true, .evictedState = true};
  }

  // Then oldest first across ordinary states and KV, or the leaves alone
  // in a page scan; the next pass takes what waited. States in use and the
  // KV they need follow in a pass of their own, up to InUse and once no
  // transfer in flight can return what is needed first.
  for (const bool inUse : {false, true}) {
    if (inUse && upTo == ReclaimClass::Ordinary)
      break;
    if (inUse && transfersInFlight())
      return {.pending = true};
    if (const auto victim = reclaimOldest({.inUse = inUse,
                                           .keepResumePoint = keepResumePoint,
                                           .unwritten = StateCache::Unwritten::Wait,
                                           .timing = ReclaimTiming::CanWait,
                                           .pagesOnly = pagesOnly}))
      return {.madeProgress = true, .evictedState = !victim->kv};
  }
  // Nothing to reclaim now. Transfers land only in pollTransfers(), so what
  // was in flight during the pass still is, and comes back.
  return {.pending = transfersInFlight()};
}

CacheReclaimResult Cache::reclaimForPages(uint32_t pages, CacheReclaimMode mode,
                                          ReclaimClass upTo) {
  CacheReclaimResult total;
  while (pool_.freePageCount() + pendingPages() < pages) {
    const CacheReclaimResult step = reclaimOne(mode, upTo);
    if (!step.madeProgress) {
      total.pending = step.pending;
      break;
    }
    total.madeProgress = true;
    // A state's buffers, once the model's pool returns them, may let the
    // retry grow the pool. While the host refuses growth they give no page,
    // and the leaf the state sat on goes next.
    if (step.evictedState && mode == CacheReclaimMode::KeepExtents)
      break;
  }
  return total;
}

uint32_t Cache::reusablePages(ReclaimClass upTo) const {
  const uint32_t unheld = pool_.reusablePages();
  if (upTo == ReclaimClass::InUse)
    return unheld;
  return unheld - kv_.idlePagesOnChains(states_.usedStates());
}

std::optional<Cache::Victim> Cache::reclaimOldest(const VictimScan &scan) {
  // A dead leaf never needs a demotion; one whose disk subtree is busy
  // stays, and the next goes.
  if (!scan.inUse) {
    for (auto leaf = oldestDeadKvLeaf(0); leaf; leaf = oldestDeadKvLeaf(leaf->id))
      if (reclaimKvLeaf(leaf->id, scan.timing) == LeafReclaim::Started)
        return Victim{true};
  }
  std::optional<CacheEvictionCandidate> state;
  if (!scan.pagesOnly)
    state = scan.inUse ? states_.inUseCandidate() : states_.ordinaryCandidate(scan.keepResumePoint);
  std::optional<CacheEvictionCandidate> kv = oldestKvLeaf(0, scan.inUse, scan.pagesOnly);
  ReclaimTiming timing = scan.timing;
  while (state || kv) {
    if (state && (!kv || state->lastUsed <= kv->lastUsed)) {
      if (states_.reclaim(state->id, scan.unwritten).evicted)
        return Victim{false};
      state.reset();
      continue;
    }
    // A page scan's leaf gives up its state in RAM first. The resume point
    // the scan keeps stays, and the leaf under it.
    if (states_.stateResident(kv->id)) {
      if ((!scan.keepResumePoint || kv->id != states_.resumePoint()) &&
          states_.reclaim(kv->id, scan.unwritten).evicted)
        return Victim{false};
      kv = oldestKvLeaf(kv->id, scan.inUse, scan.pagesOnly);
      continue;
    }
    switch (reclaimKvLeaf(kv->id, timing)) {
    case LeafReclaim::Started:
      return Victim{true};
    case LeafReclaim::Pending:
      // The tier takes no demotion until a transfer lands: the leaves that
      // need one wait with this one, and those whose page frees now go.
      timing = ReclaimTiming::Immediate;
      [[fallthrough]];
    case LeafReclaim::Impossible:
      kv = oldestKvLeaf(kv->id, scan.inUse, scan.pagesOnly);
      break;
    }
  }
  return std::nullopt;
}

StateRoom Cache::reclaimOneState(bool checkpointsOnly, uint64_t forBlock, bool growth) {
  using Unwritten = StateCache::Unwritten;
  const bool inUse = !checkpointsOnly && states_.inUse(forBlock);
  // Only a publication in use cannot leave a victim to the write in flight:
  // its lane's state moves on with the next command.
  const Unwritten unwritten = inUse ? Unwritten::Drop : Unwritten::Wait;
  const auto recycle = [&](const std::optional<CacheEvictionCandidate> &state,
                           Unwritten mode) {
    return state && states_.reclaim(state->id, mode).evicted;
  };
  // Without growth an extent's bytes are no room for a snapshot, and only
  // states go.
  if (growth && !checkpointsOnly) {
    if (const CacheReclaimResult released = releaseExtent(false); released.reclaimedBytes)
      return {true, released.reclaimedBytes};
  }
  if (recycle(states_.checkpointCandidate(false), unwritten))
    return {true};
  if (checkpointsOnly)
    return {};
  // KV makes room with the extent its free pages fill, released at once,
  // and goes only while one can be emptied.
  if (growth && extentWithinReach()) {
    while (const auto victim = reclaimOldest({.inUse = false,
                                              .keepResumePoint = false,
                                              .unwritten = unwritten,
                                              .timing = ReclaimTiming::Immediate,
                                              .pagesOnly = false})) {
      if (!victim->kv)
        return {true};
      if (const CacheReclaimResult released = releaseExtent(false); released.reclaimedBytes)
        return {true, released.reclaimedBytes};
    }
  } else if (recycle(states_.ordinaryCandidate(false), unwritten)) {
    return {true};
  }
  if (inUse && recycle(states_.inUseCandidate(), Unwritten::Wait))
    return {true};
  return {};
}

CacheReclaimResult Cache::reclaimStateForLane(ReclaimClass upTo) {
  // Running work takes a state in use after every other, once no transfer
  // in flight can return what is needed first.
  std::optional<CacheEvictionCandidate> state = states_.checkpointCandidate(false);
  if (!state)
    state = states_.ordinaryCandidate(false);
  if (!state && upTo == ReclaimClass::InUse && !transfersInFlight())
    state = states_.inUseCandidate();
  if (!state)
    return {.pending = transfersInFlight()};
  const StateEviction eviction = states_.reclaim(state->id, StateCache::Unwritten::Wait);
  return {.madeProgress = eviction.evicted,
          .pending = eviction.pending,
          .evictedState = eviction.evicted};
}

std::optional<CacheEvictionCandidate> Cache::oldestDeadKvLeaf(uint64_t after) const {
  while (auto candidate = kv_.evictionCandidate(after)) {
    after = candidate->id;
    if (!kvNeededByState(candidate->id))
      return candidate;
  }
  return std::nullopt;
}

std::optional<CacheEvictionCandidate> Cache::oldestKvLeaf(uint64_t after, bool inUse,
                                                          bool withRamState) const {
  while (auto candidate = kv_.evictionCandidate(after)) {
    after = candidate->id;
    if ((withRamState || !states_.stateResident(candidate->id)) &&
        kvNeededByStateInUse(candidate->id) == inUse)
      return candidate;
  }
  return std::nullopt;
}

Cache::LeafReclaim Cache::reclaimKvLeaf(uint64_t block, ReclaimTiming timing) {
  // No state in RAM sits on the leaf (oldestKvLeaf): one went first, in the
  // order every victim goes in, or just before its leaf in a page scan. A
  // state already on disk costs nothing and stays while the leaf has a disk
  // copy.
  if (kv_.slot(block)) {
    kv_.dropPage(block);
    return LeafReclaim::Started;
  }
  // The leaf is written for a state on it or below it, and disk copies below
  // that no state needs go with it. A writable tier means a demotion, whose
  // page returns only when the copy lands.
  if (kvNeededByState(block)) {
    if (timing == ReclaimTiming::Immediate && kvTierWritable())
      return LeafReclaim::Impossible;
    const LeafReclaim demotion = demoteKv(block);
    if (demotion != LeafReclaim::Impossible)
      return demotion;
    // A leaf its disk subtree depends on stays in RAM while the tier may
    // still write it: dropping it would orphan every copy below it. Once a
    // failed write has closed the tier, the subtree goes with it, as without
    // a tier, rather than holding the leaf's RAM until the server restarts.
    if (kv_.stateBelow(block) && kvTierWritable())
      return LeafReclaim::Impossible;
  }
  if (kv_.hasDiskChildren(block) && !dropDiskSubtree(block))
    return LeafReclaim::Impossible;
  states_.evict(block);
  kv_.erase(block);
  return LeafReclaim::Started;
}

bool Cache::dropDiskSubtree(uint64_t block) {
  if (states_.writing())
    return false;
  const std::vector<uint64_t> subtree = kv_.subtree(block);
  if (subtree.empty())
    return false;
  for (uint64_t below : subtree) {
    states_.evict(below);
    kv_.erase(below);
  }
  return true;
}

void Cache::dropPoisoned() {
  std::erase_if(poisoned_, [&](uint64_t block) {
    if (!kv_.contains(block))
      return true;
    for (uint64_t below : kv_.subtree(block)) {
      states_.evict(below);
      // A poisoned block below may have left with its last child.
      if (kv_.contains(below))
        kv_.erase(below);
    }
    return !kv_.contains(block);
  });
}

bool Cache::extentWithinReach() const {
  return pendingPages() + pool_.extentPages() <= reusablePages(ReclaimClass::Ordinary);
}

CacheReclaimResult Cache::releaseExtent(bool keepRunway) {
  if (const uint64_t bytes = pool_.reclaimEmptyExtents(keepRunway, 1))
    return {true, bytes};
  if (compactExtent())
    return {true, pool_.reclaimEmptyExtents(keepRunway, 1)};
  return {};
}

bool Cache::compactExtent() {
  const auto start = std::chrono::steady_clock::now();
  std::vector<uint32_t> inTransfer;
  inTransfer.reserve(demotions_.size() + restores_.size());
  for (const Demotion &demotion : demotions_)
    inTransfer.push_back(kv_.page(demotion.block));
  for (const auto &[block, _] : restores_)
    inTransfer.push_back(kv_.page(block));
  const KvPageMoves moves = pool_.compactExtent(inTransfer);
  if (moves.empty())
    return false;
  kv_.followPages(moves);
  for (auto &[_, active] : requests_) {
    const size_t pages = active.pages.size();
    size_t first = pages;
    for (size_t index = 0; index < pages; ++index) {
      uint32_t &page = active.pages[index];
      const uint32_t destination = moves.follow(page);
      if (destination == page)
        continue;
      page = destination;
      first = std::min(first, index);
    }
    if (first < pages)
      pagesChanged(active, static_cast<uint32_t>(first));
  }
  extentCompactMaxMilliseconds_ = std::max(
      extentCompactMaxMilliseconds_,
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count());
  return true;
}

bool Cache::transfersInFlight() const noexcept {
  return !restores_.empty() || !demotions_.empty() || states_.writing();
}

uint64_t Cache::pendingBytes() const noexcept {
  return uint64_t{pendingPages()} * pool_.bytesPerPage();
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
  // Resident blocks are a rooted prefix of the chain.
  const auto firstOnDisk = std::find(chain.pages.begin(), chain.pages.end(), KvCache::noPage);
  const auto missing = static_cast<uint32_t>(chain.pages.end() - firstOnDisk);
  std::vector<uint32_t> fresh;
  if (missing) {
    if (!tier_)
      throw std::logic_error("disk-only KV block without a disk tier");
    const TokenAdmission admission = admitPages(missing, fresh);
    if (!admission.granted())
      return admission;
  }
  kv_.retainActive(chain.blocks.back());
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
    }
    if (auto restore = restores_.find(block); restore != restores_.end()) {
      restore->second.waiters.push_back(requestId);
      ++active.pendingRestores;
    }
  }
  active.pages = std::move(chain.pages);
  active.cachedBlocks = std::move(chain.blocks);
  pagesChanged(active, 0);
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
  if (tier_->canRestore())
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
  auto state = transfer.snapshot();
  if (!state) {
    // The restored state is the most recently used one; the oldest RAM copy
    // makes room for it. Promotion only saves a later read: it takes only
    // RAM whose state keeps a copy, never a state's only copy.
    auto victim = states_.checkpointCandidate(false);
    if (!victim)
      victim = states_.ordinaryCandidate(false);
    if (victim && states_.reclaim(victim->id, StateCache::Unwritten::Keep).evicted)
      state = transfer.snapshot();
  }
  if (state) states_.promote(block, source, std::move(state));
  else states_.promotionSkipped();
}

Cache::LeafReclaim Cache::demoteKv(uint64_t block) {
  if (!kvTierWritable())
    return LeafReclaim::Impossible;
  // Do not discard a disk copy for a transfer that cannot start yet.
  if (!tier_->canDemote()) {
    ++kvTier_.demotionsRefused;
    return transfersInFlight() ? LeafReclaim::Pending : LeafReclaim::Impossible;
  }
  // Room in the quota or the tier that transfers in flight will free is
  // worth waiting for; room that nothing will free is not, and the leaf goes.
  std::shared_ptr<KvDiskSlot> slot = acquireDiskSlot(kvNeededByStateInUse(block));
  if (!slot)
    return transfersInFlight() ? LeafReclaim::Pending : LeafReclaim::Impossible;
  // Making room may have taken the states the leaf was kept for; it then
  // goes like any leaf nothing needs.
  if (!kvNeededByState(block))
    return LeafReclaim::Impossible;
  auto transfer = tier_->demote(kv_.page(block), slot, completionNotifier_);
  if (!transfer) {
    ++kvTier_.demotionsRefused;
    return transfersInFlight() ? LeafReclaim::Pending : LeafReclaim::Impossible;
  }
  kv_.setSlot(block, std::move(slot));
  kv_.setTransferring(block, true);
  demotions_.push_back({block, std::move(transfer)});
  return LeafReclaim::Started;
}

std::shared_ptr<KvDiskSlot> Cache::acquireDiskSlot(bool inUse) {
  for (;;) {
    if (auto slot = tier_->acquireSlot())
      return slot;
    if (!freeDiskSpace(inUse))
      return {};
  }
}

bool Cache::freeDiskSpace(bool inUse) {
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
  // A leaf holding a state in use stays with that state.
  auto kvLeaf = kv_.diskCandidate(false);
  while (kvLeaf && kvNeededByStateInUse(kvLeaf->id))
    kvLeaf = kv_.diskCandidate(false, kvLeaf->id);
  const auto stateOnly = states_.diskCandidate(false);
  if (!kvLeaf && !stateOnly) {
    // Last, and only for a copy in use: the only copy of a state in use. The
    // leaf it held is ordinary then, and goes on a later call.
    const auto used = inUse ? states_.inUseDiskCandidate() : std::nullopt;
    if (!used)
      return false;
    states_.evict(used->id);
    return true;
  }
  if (older(stateOnly, kvLeaf)) {
    states_.evict(stateOnly->id);
    return true;
  }
  states_.evict(kvLeaf->id);
  kv_.erase(kvLeaf->id);
  return true;
}

bool Cache::pollTransfers() {
  bool progressed = states_.pollOffload();
  if (!tier_)
    return progressed;
  tier_->poll();
  // Waiting restores start in block order and stop at the first the tier
  // refuses: started ones stay a prefix of each chain, and a full tier is
  // asked once per poll, not once per waiting block.
  bool refused = false;
  for (auto entry = restores_.begin(); entry != restores_.end();) {
    auto &[block, restore] = *entry;
    if (!restore.transfer && !refused) {
      if (tier_->canRestore())
        restore.transfer =
            tier_->restore(kv_.slot(block), kv_.page(block), completionNotifier_);
      refused = !restore.transfer;
    }
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
      // The block leaves with its last user; so does any state it held, and
      // what lies below it, which no lookup reaches any more.
      states_.invalidate(block);
      kv_.poison(block);
      poisoned_.push_back(block);
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
    kv_.setTransferring(block, false);
    if (!written) {
      ++kvTier_.demotionFailures;
      kv_.setSlot(block, nullptr);
    } else {
      ++kvTier_.demotions;
      // A request that matched the block meanwhile keeps its page, and so
      // does a state published on it in RAM; the copy makes the next reclaim
      // of the leaf free.
      if (kv_.residentLeaf(block) && !states_.stateResident(block))
        kv_.dropPage(block);
    }
    progressed = true;
  }
  if (progressed)
    dropPoisoned();
  return progressed;
}

CacheSnapshot Cache::snapshot() const {
  KvTierSnapshot tier = kvTier_;
  tier.pendingPages = pendingPages();
  tier.diskBlocks = kv_.diskBlocks();
  if (diskBudget_) {
    tier.capacityBytes = diskBudget_->capacityBytes();
    tier.usedBytes = diskBudget_->usedBytes();
    tier.fileBytes = diskBudget_->fileBytes();
    tier.readBytes = diskBudget_->readBytes();
    tier.writtenBytes = diskBudget_->writtenBytes();
  }
  if (tier_)
    tier.diskBytes = uint64_t{tier.diskBlocks} * tier_->slotBytes();
  return {pool_.snapshot(),
          states_.snapshot(),
          tier,
          lookup_,
          static_cast<uint32_t>(requests_.size()),
          extentCompactMaxMilliseconds_};
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
