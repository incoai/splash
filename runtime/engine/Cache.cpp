#include "engine/Cache.hpp"

#include "AwakeClock.hpp"
#include "engine/DiskLabels.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <unordered_set>
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

// The left candidate exists and was used before the right one, if any.
bool older(const std::optional<CacheEvictionCandidate> &left,
           const std::optional<CacheEvictionCandidate> &right) noexcept {
  return left && (!right || left->lastUsed < right->lastUsed);
}

} // namespace

Cache::Cache(KvPool &pool, KvTier *kvTier, std::shared_ptr<model::DiskBudget> diskBudget,
             std::optional<CacheSpillPolicy> spill)
    : pool_(pool), tier_(kvTier), spill_(spill), persistent_(kvTier && kvTier->persistent()),
      diskBudget_(std::move(diskBudget)), kv_(pool, recency_, spill.has_value(), persistent_),
      states_(kv_, recency_, [this](bool inUse) { return freeDiskSpace(inUse); }, persistent_) {
  // Spilled K/V lives in the tier's copies, within its quota.
  if (spill_ && (!tier_ || !diskBudget_))
    throw std::invalid_argument("a cache whose requests spill needs a disk tier and its quota");
}

CacheAdoption Cache::adopt(std::vector<PersistedKv> blocks, std::vector<PersistedState> states) {
  if (!persistent_)
    throw std::logic_error("a cache takes back copies only from a persistent tier");
  if (!requests_.empty() || kv_.diskBlocks() || states_.snapshot().entries)
    throw std::logic_error("a cache takes back copies only before anything else is in it");
  CacheAdoption result;
  // Every block's label by id. A parent's id is lower than its children's,
  // so a chain resolves in id order and a cycle cannot form.
  struct Recorded final {
    KvBlockLabel label;
    size_t index = 0;
    // Tokens up to and including the block; zero while its chain is broken.
    uint32_t tokens = 0;
    bool needed = false;
  };
  std::map<uint64_t, Recorded> recorded;
  // New ids continue after every id a record names, taken back or not: a
  // record whose clearing a crash lost must never name a block of this
  // process.
  uint64_t highestId = 0;
  for (size_t index = 0; index < blocks.size(); ++index) {
    const auto label = decodeLabel<KvBlockLabel>(blocks[index].label);
    if (label)
      highestId = std::max(highestId, label->id);
    if (!label || !label->id || label->parent >= label->id ||
        !recorded.emplace(label->id, Recorded{*label, index}).second)
      ++result.dropped;
  }
  for (auto &[id, block] : recorded) {
    if (!block.label.parent) {
      block.tokens = KvCache::pageTokens;
    } else if (auto parent = recorded.find(block.label.parent);
               parent != recorded.end() && parent->second.tokens) {
      block.tokens = parent->second.tokens + KvCache::pageTokens;
    }
  }
  // A state comes back with its whole chain; a block only for a state.
  std::vector<std::pair<StateLabel, size_t>> kept;
  for (size_t index = 0; index < states.size(); ++index) {
    const auto label = decodeLabel<StateLabel>(states[index].label);
    if (label)
      highestId = std::max(highestId, label->block);
    const auto block = label ? recorded.find(label->block) : recorded.end();
    if (block == recorded.end() || !block->second.tokens ||
        block->second.tokens != label->tokens) {
      ++result.dropped;
      continue;
    }
    kept.emplace_back(*label, index);
    for (auto above = block; above != recorded.end() && !above->second.needed;
         above = recorded.find(above->second.label.parent))
      above->second.needed = true;
  }
  // A record its file refuses, or whose key another block took, stays
  // behind like a damaged one, and so does everything below it.
  uint64_t lastUsed = 0;
  std::unordered_set<uint64_t> adopted;
  for (auto &[id, block] : recorded) {
    std::shared_ptr<KvDiskSlot> slot;
    if (block.needed && (!block.label.parent || adopted.contains(block.label.parent))) {
      try {
        slot = blocks[block.index].adopt();
      } catch (const std::invalid_argument &) {
      }
    }
    if (!slot || !kv_.adoptDiskBlock(id, block.label.parent, block.label.tokens,
                                     {block.label.identityLo, block.label.identityHi},
                                     std::move(slot), block.label.lastUsed)) {
      ++result.dropped;
      continue;
    }
    adopted.insert(id);
    lastUsed = std::max(lastUsed, block.label.lastUsed);
  }
  // A block holds one state; a second record for it is a stale one.
  std::unordered_set<uint64_t> stated;
  for (const auto &[label, index] : kept) {
    std::shared_ptr<const CompositeState> state;
    if (adopted.contains(label.block) && stated.insert(label.block).second) {
      try {
        state = states[index].adopt(label.tokens);
      } catch (const std::invalid_argument &) {
      }
    }
    if (!state) {
      ++result.dropped;
      continue;
    }
    states_.adoptDisk(label.block, std::move(state), label.lastUsed, label.checkpoint != 0);
    lastUsed = std::max(lastUsed, label.lastUsed);
  }
  recency_.continueAfter(lastUsed);
  kv_.continueIdsAfter(highestId);
  // A lower limit than the earlier process had takes the oldest copies.
  result.dropped += fitDiskLimit();
  const StateCacheSnapshot held = states_.snapshot();
  result.states = held.entries;
  result.blocks = kv_.diskBlocks();
  result.bytes = held.diskBytes + uint64_t{result.blocks} * tier_->slotBytes();
  adoption_ = result;
  return result;
}

void Cache::beginRequest(uint64_t requestId, KvMode mode) {
  if (!requestId)
    throw std::invalid_argument("invalid request id");
  if (mode == KvMode::Spill && !spill_)
    throw std::logic_error("only a cache with a spill policy takes requests that spill");
  auto [entry, inserted] = requests_.emplace(requestId, Request{});
  if (!inserted)
    throw std::invalid_argument("duplicate request id");
  entry->second.mode = mode;
  if (mode == KvMode::Spill)
    entry->second.spillOrder = ++spillSequence_;
}

void Cache::endRequest(uint64_t requestId) {
  auto found = requests_.find(requestId);
  if (found == requests_.end())
    return;
  Request &active = found->second;
  // Its last cut's copies go before its pages: a block nothing holds then
  // may leave with its copy.
  active.sources.reset();
  returnCommitment(active, active.committedPages);
  for (uint32_t page : active.pages)
    kv_.unholdPage(page);
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
  const bool keptFull = active.mode == KvMode::Resident;
  requests_.erase(found);
  // The Full pages it kept from spilling may spill now.
  if (spill_ && keptFull)
    rescanSpills();
  dropPoisoned();
}

size_t Cache::extendMatch(std::vector<uint64_t> &blocks, std::span<const uint32_t> prompt,
                          PromptIdentity identity) const {
  if (prompt.empty())
    return 0;
  // Leave one real input token to regenerate request-specific anchor logits.
  const size_t maximumBlocks = (prompt.size() - 1) / KvCache::pageTokens;
  blocks.reserve(maximumBlocks);
  size_t hashed = 0;
  for (size_t index = blocks.size(); index < maximumBlocks; ++index) {
    const size_t begin = index * KvCache::pageTokens;
    const uint64_t parent = blocks.empty() ? 0 : blocks.back();
    identity.images = spansFrom(identity.images, begin);
    ++hashed;
    auto match = kv_.find(parent, prompt.subspan(begin, KvCache::pageTokens),
                          blockIdentity(begin, KvCache::pageTokens, identity.images, identity.salt));
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

CacheProbe Cache::probe(std::span<const uint32_t> prompt, PromptIdentity identity) {
  CacheProbe result;
  result.owner_ = this;
  result.kvGeneration_ = kv_.generation();
  lookup_.probeHashedBlocks += extendMatch(result.blocks_, prompt, identity);
  result.cachedTokens_ = stateTokens(result.blocks_);
  return result;
}

void Cache::refresh(CacheProbe &probe, std::span<const uint32_t> prompt,
                    PromptIdentity identity) {
  if (probe.owner_ != this)
    throw std::logic_error("a scheduling probe belongs to another cache");
  if (probe.kvGeneration_ != kv_.generation()) {
    // See KvCache's generation: the chain holds up to its first block that
    // no longer matches.
    probe.blocks_.erase(std::find_if(probe.blocks_.begin(), probe.blocks_.end(),
                                     [&](uint64_t block) { return !kv_.matchable(block); }),
                        probe.blocks_.end());
    lookup_.probeHashedBlocks += extendMatch(probe.blocks_, prompt, identity);
    probe.kvGeneration_ = kv_.generation();
  }
  // States come and go without a change of the KV graph.
  probe.cachedTokens_ = stateTokens(probe.blocks_);
}

CacheLookup Cache::lookup(std::span<const uint32_t> prompt,
                          PromptIdentity identity,
                          const CacheProbe *probe) {
  CacheLookup result;
  std::vector<uint64_t> fallback;
  std::span<const uint64_t> blocks;
  if (probe && probe->owner_ == this && probe->kvGeneration_ == kv_.generation()) {
    blocks = probe->blocks_;
  } else {
    extendMatch(fallback, prompt, identity);
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

PageTableState Cache::pageTable(uint64_t requestId) const {
  const Request &active = request(requestId);
  if (active.dirtyFrom)
    return {active.pages, active.revision + 1, *active.dirtyFrom};
  return {active.pages, active.revision, active.firstChanged};
}

PageTableView Cache::cutPageTable(uint64_t requestId) {
  Request &active = request(requestId);
  if (active.dirtyFrom) {
    ++active.revision;
    active.firstChanged = *active.dirtyFrom;
    active.dirtyFrom.reset();
    active.sources = cutSources(active);
  }
  return {active.pages, active.sources, active.revision, active.firstChanged};
}

std::shared_ptr<const model::KvPageSources> Cache::cutSources(const Request &active) const {
  if (!active.indexPages)
    return nullptr;
  using Chunk = model::KvPageSources::Chunk;
  constexpr uint32_t chunkPages = model::KvPageSources::kChunkPages;
  const auto pages = static_cast<uint32_t>(active.pages.size());
  const uint32_t chunks = (pages + chunkPages - 1) / chunkPages;
  // The last cut's chunks before the first change hold what they held; a
  // list only grows, so each of them is whole.
  const uint32_t kept = active.sources ? std::min(active.firstChanged / chunkPages,
                                                   static_cast<uint32_t>(active.sources->chunks().size()))
                                       : 0;
  std::vector<std::shared_ptr<const Chunk>> result;
  result.reserve(chunks);
  if (kept)
    result.assign(active.sources->chunks().begin(), active.sources->chunks().begin() + kept);
  for (uint32_t chunk = kept; chunk < chunks; ++chunk) {
    auto cut = std::make_shared<Chunk>();
    const uint32_t end = std::min(pages, (chunk + 1) * chunkPages);
    for (uint32_t page = chunk * chunkPages; page < end; ++page) {
      model::KvPageSource &source = cut->pages[page % chunkPages];
      source = sourceOf(active.pages[page]);
      cut->indexPages += source.indexed();
    }
    result.push_back(std::move(cut));
  }
  return std::make_shared<const model::KvPageSources>(std::move(result), pages);
}

model::KvPageSource Cache::sourceOf(uint32_t page) const {
  if (pool_.form(page) == kv::PageForm::Full)
    return {};
  // A request reaches an Index page only through the spilled block on it,
  // whose copy holds its K/V.
  const uint64_t block = kv_.blockOn(page);
  if (!block || !kv_.slot(block))
    throw std::logic_error("a page of the index form holds no spilled block");
  return {block, tier_->copy(kv_.slot(block))};
}

void Cache::pagesChanged(Request &active, uint32_t first) noexcept {
  active.dirtyFrom = std::min(active.dirtyFrom.value_or(first), first);
}

uint32_t Cache::publishCommittedBlocks(uint64_t requestId,
                                       std::span<const uint32_t> exactTokens,
                                       uint32_t committedTokens,
                                       PromptIdentity identity) {
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
  uint32_t swappedTokens = 0;
  while (active.cachedBlocks.size() < fullBlocks) {
    const uint32_t logical = static_cast<uint32_t>(active.cachedBlocks.size());
    const uint64_t parent = logical ? active.cachedBlocks.back() : 0;
    const uint32_t begin = logical * KvCache::pageTokens;
    identity.images = spansFrom(identity.images, begin);
    auto inserted =
        kv_.insert(parent, exactTokens.subspan(begin, KvCache::pageTokens),
                   active.pages[logical],
                   blockIdentity(begin, KvCache::pageTokens, identity.images, identity.salt));
    const uint32_t writerPage = active.pages[logical];
    const bool replacePage = inserted.physicalPage != writerPage;
    if (replacePage)
      kv_.holdPage(inserted.physicalPage);
    kv_.retainActive(inserted.id);
    active.cachedBlocks.push_back(inserted.id);
    if (replacePage) {
      active.pages[logical] = inserted.physicalPage;
      pagesChanged(active, logical);
      kv_.unholdPage(writerPage);
      active.indexPages += pool_.form(inserted.physicalPage) == kv::PageForm::Index;
      swappedTokens += KvCache::pageTokens;
    }
    if (parent)
      kv_.releaseActive(parent);
    // A Spill request's page draws its commitment when it is written, or
    // returns it when the block has a copy already.
    if (active.mode == KvMode::Spill) {
      if (kv_.kvCopied(inserted.id))
        returnCommitment(active, 1);
      else
        active.owedCopies.push_back(inserted.id);
    }
  }
  return swappedTokens;
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

TokenAdmission Cache::ensureTokens(uint64_t requestId, uint64_t tokenCount,
                                   uint64_t writtenTokens) {
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
  const uint64_t written = std::min(writtenTokens, tokenCount);
  for (uint32_t index = 0; index < acquired.size(); ++index)
    if (uint64_t{previousSize + index + 1} * KvCache::pageTokens > written)
      pool_.clearPage(acquired[index]);
  active.pages.insert(active.pages.end(), acquired.begin(), acquired.end());
  pagesChanged(active, previousSize);
  return {};
}

TokenAdmission Cache::admitPages(uint32_t count, std::vector<uint32_t> &pages,
                                  kv::PageForm form) {
  KvPageAcquisition acquired = pool_.acquirePages(count, form);
  if (!acquired.granted()) {
    // Demoted and spilled pages free theirs when their copies land: wait
    // once those on their way cover what the free pages do not, and until
    // then reclaim.
    const uint32_t missing = count - std::min(count, pool_.freePageCount(form));
    const uint32_t pending = pendingPages(form);
    const bool covered = pending > 0 && pending >= missing;
    return {covered ? TokenAdmissionFailure::Pending : TokenAdmissionFailure::Denied,
            count, pool_.freePageCount(form), acquired.failure, form};
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
  while (evictOne({.mode = CacheReclaimMode::KeepExtents, .upTo = ReclaimClass::InUse}).madeProgress)
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
      evictOne({.mode = mode, .upTo = upTo, .keepResumePoint = keepResumePoint});
  if (evicted.madeProgress && release)
    evicted.reclaimedBytes += pool_.reclaimEmptyExtents(keepRunway, 1);
  return evicted;
}

CacheReclaimResult Cache::evictOne(const EvictionStep &step) {
  const bool pagesOnly = step.mode == CacheReclaimMode::ReusePages;
  // Whether a transfer in flight can return what the step needs before
  // anything in use goes.
  const auto returns = [&] {
    return step.refused ? returnsInFlight(*step.refused) : transfersInFlight();
  };
  // Disposable checkpoints go first, except in a page scan, to which their
  // buffers give no page; one whose write must wait for the one in flight
  // stays and holds back nothing else.
  if (const auto oldest = states_.checkpointCandidate(step.keepResumePoint); oldest && !pagesOnly) {
    if (states_.reclaim(oldest->id, StateCache::Unwritten::Wait).evicted)
      return {.madeProgress = true, .evictedState = true};
  }

  // Then oldest first across ordinary states and KV, or the leaves alone
  // in a page scan; the next pass takes what waited. States in use and the
  // KV they need follow in a pass of their own, up to InUse and once no
  // transfer in flight can return what is needed first.
  for (const bool inUse : {false, true}) {
    if (inUse && step.upTo == ReclaimClass::Ordinary)
      break;
    if (inUse && returns())
      return {.pending = true};
    if (const auto victim = reclaimOldest({.inUse = inUse,
                                           .keepResumePoint = step.keepResumePoint,
                                           .unwritten = StateCache::Unwritten::Wait,
                                           .timing = ReclaimTiming::CanWait,
                                           .pagesOnly = pagesOnly}))
      return {.madeProgress = true, .evictedState = !victim->kv};
  }
  // Nothing to reclaim now. Transfers land only in pollTransfers(), so what
  // was in flight during the pass still is, and comes back.
  return {.pending = returns()};
}

CacheReclaimResult Cache::reclaimForPages(uint32_t pages, CacheReclaimMode mode,
                                          ReclaimClass upTo, kv::PageForm form) {
  CacheReclaimResult total;
  // In a cache that spills, the extents of the other form cannot serve the
  // shortfall: those empty beyond the runway each form keeps go, for the
  // retry to grow by. For Index pages, so does a Full extent their free
  // pages empty: the Full pages spills freed fund the next Index extent.
  const auto releaseOtherForm = [&] {
    if (!spill_)
      return false;
    const kv::PageForm other = form == kv::PageForm::Full ? kv::PageForm::Index : kv::PageForm::Full;
    const uint32_t all = std::numeric_limits<uint32_t>::max();
    uint64_t released = pool_.reclaimEmptyExtents(true, all, other);
    if (!released && form == kv::PageForm::Index && compactExtent(other))
      released = pool_.reclaimEmptyExtents(true, all, other);
    total.reclaimedBytes += released;
    total.madeProgress = total.madeProgress || released;
    return released != 0;
  };
  if (releaseOtherForm())
    return total;
  // Leaves of either form go, oldest first: an idle spilled chain's Index
  // leaves go before the Full pages of its dense history above them.
  while (pool_.freePageCount(form) + pendingPages(form) < pages) {
    const CacheReclaimResult step =
        evictOne({.mode = mode, .upTo = upTo, .refused = Shortfall{pages, form}});
    if (!step.madeProgress) {
      total.pending = step.pending;
      break;
    }
    total.madeProgress = true;
    // A state's buffers, once the model's pool returns them, may let the
    // retry grow the pool. While the host refuses growth they give no page,
    // and the leaf the state sat on goes next. So may an extent of the
    // other form the step emptied.
    if ((step.evictedState && mode == CacheReclaimMode::KeepExtents) || releaseOtherForm())
      break;
  }
  return total;
}

CacheReclaimResult Cache::reclaimForMemory(ReclaimClass upTo, bool keepRunway) {
  const Shortfall memory{pool_.extentPages(), kv::PageForm::Full};
  // Free pages first, as the extent they fill; one that empties the runway
  // kept gives nothing yet, and the next one does.
  for (CacheReclaimResult released; (released = releaseExtent(keepRunway)).madeProgress;)
    if (released.reclaimedBytes)
      return released;
  // Then the victims in reclaimOne's order while their KV can empty an
  // extent, until one is released or a state goes; otherwise only a state.
  // Whatever it returns says whether KV went.
  bool evictedKv = false;
  const auto outcome = [&](CacheReclaimResult result) {
    result.evictedKv = evictedKv;
    return result;
  };
  while (extentWithinReach(upTo)) {
    const CacheReclaimResult evicted =
        evictOne({.mode = CacheReclaimMode::ReleaseExtents, .upTo = upTo, .refused = memory});
    if (!evicted.madeProgress || evicted.evictedState)
      return outcome(evicted);
    evictedKv = true;
    if (const CacheReclaimResult released = releaseExtent(keepRunway); released.reclaimedBytes)
      return outcome(released);
  }
  return outcome(recycleState(upTo, memory));
}

uint32_t Cache::reusablePages(ReclaimClass upTo, kv::PageForm form) const {
  const uint32_t unheld = pool_.reusablePages(form);
  if (upTo == ReclaimClass::InUse)
    return unheld;
  return unheld - kv_.idlePagesOnChains(states_.usedStates(), form);
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

StateRoom Cache::reclaimOneState(StateClass state, bool growth) {
  using Unwritten = StateCache::Unwritten;
  const bool checkpointsOnly = state == StateClass::Checkpoint;
  // Only a state in use cannot leave a victim to the write in flight:
  // refused, it reaches the disk tier only through that write.
  const Unwritten unwritten = state == StateClass::InUse ? Unwritten::Drop : Unwritten::Wait;
  const auto recycle = [&](const std::optional<CacheEvictionCandidate> &candidate,
                           Unwritten mode) {
    return candidate && states_.reclaim(candidate->id, mode).evicted;
  };
  // Without growth an extent's bytes are no room for a state's buffers, and
  // only states go.
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
  if (state == StateClass::InUse && recycle(states_.inUseCandidate(), Unwritten::Wait))
    return {true};
  return {};
}

CacheReclaimResult Cache::reclaimStateForLane(ReclaimClass upTo) {
  // No page covers a state's buffers.
  return recycleState(upTo, Shortfall{});
}

CacheReclaimResult Cache::recycleState(ReclaimClass upTo, const Shortfall &shortfall) {
  // Running work takes a state in use after every other, once no transfer
  // in flight can return what is needed first.
  std::optional<CacheEvictionCandidate> state = states_.checkpointCandidate(false);
  if (!state)
    state = states_.ordinaryCandidate(false);
  if (!state && upTo == ReclaimClass::InUse && !returnsInFlight(shortfall))
    state = states_.inUseCandidate();
  if (!state)
    return {.pending = returnsInFlight(shortfall)};
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
    // file has stopped taking writes, the subtree goes with it, as without a
    // tier, rather than holding the leaf's RAM until the server restarts.
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

bool Cache::extentWithinReach(ReclaimClass upTo) const {
  return writes(WriteClass::Reclaim) + pool_.extentPages() <= reusablePages(upTo);
}

CacheReclaimResult Cache::releaseExtent(bool keepRunway) {
  if (const uint64_t bytes = pool_.reclaimEmptyExtents(keepRunway, 1))
    return {true, bytes};
  if (compactExtent())
    return {true, pool_.reclaimEmptyExtents(keepRunway, 1)};
  return {};
}

bool Cache::compactExtent(std::optional<kv::PageForm> form) {
  const auto start = AwakeClock::now();
  std::vector<uint32_t> inTransfer;
  inTransfer.reserve(writes_.size() + restores_.size());
  for (const Write &write : writes_)
    inTransfer.push_back(kv_.page(write.block));
  for (const auto &[block, _] : restores_)
    inTransfer.push_back(kv_.page(block));
  const KvPageMoves moves = pool_.compactExtent(inTransfer, form);
  if (moves.empty())
    return false;
  followMoves(moves);
  extentCompactMaxMilliseconds_ =
      std::max(extentCompactMaxMilliseconds_, millisecondsSince(start));
  return true;
}

void Cache::followMoves(const KvPageMoves &moves) {
  kv_.followPages(moves);
  for (auto &[_, active] : requests_) {
    const size_t pages = active.pages.size();
    size_t first = pages;
    for (size_t index = 0; index < pages; ++index) {
      uint32_t &page = active.pages[index];
      const uint32_t destination = moves.follow(page);
      if (destination == page)
        continue;
      if (pool_.form(destination) != pool_.form(page))
        pool_.form(destination) == kv::PageForm::Index ? ++active.indexPages : --active.indexPages;
      page = destination;
      first = std::min(first, index);
    }
    if (first < pages)
      pagesChanged(active, static_cast<uint32_t>(first));
  }
}

bool Cache::transfersInFlight() const noexcept {
  // A page written through holds no block a reclaim could take: only one
  // that spills once its copy lands returns something.
  return !restores_.empty() || writes(WriteClass::Reclaim) || writes(WriteClass::Persist) ||
         states_.writing() || spillingWrites();
}

bool Cache::returnsInFlight(const Shortfall &shortfall) const noexcept {
  if (!restores_.empty() || writes(WriteClass::Reclaim) || writes(WriteClass::Persist) ||
      states_.writing())
    return true;
  // With no demotion in flight, the pages of the form that transfers free
  // are those written through that will spill.
  const uint32_t freed = pendingPages(shortfall.form);
  return freed && shortfall.pages &&
         pool_.freePageCount(shortfall.form) + freed >= shortfall.pages;
}

uint64_t Cache::pendingBytes() const noexcept {
  return uint64_t{pendingPages(kv::PageForm::Full)} * pool_.bytesPerPage();
}

uint32_t Cache::pendingPages(kv::PageForm form) const noexcept {
  // Only a Full page waits for its copy: an Index page's block has one
  // already, and its page frees at once.
  return form == kv::PageForm::Full ? writes(WriteClass::Reclaim) + spillingWrites() : 0;
}

uint32_t Cache::writes(WriteClass write) const noexcept {
  return static_cast<uint32_t>(
      std::ranges::count(writes_, write, [](const Write &each) { return each.write; }));
}

uint32_t Cache::spillingWrites() const noexcept {
  // A page written through frees once its copy lands, when it may spill
  // then and an Index page is there to take its index.
  if (!spill_)
    return 0;
  uint32_t spillable = 0;
  for (const Write &write : writes_)
    spillable += write.write == WriteClass::Spill && spillEligibility(write.block) == Eligibility::Eligible;
  return std::min(spillable, pool_.freePageCount(kv::PageForm::Index));
}

// Spilling: commitments of quota, write-through of Spill requests' pages,
// and their moves to the index form once their copies landed.

uint64_t Cache::spillRoom() const noexcept {
  if (!spill_)
    return 0;
  const uint64_t active = heldSlotBytes() + diskBudget_->committedBytes();
  const uint64_t limit = diskBudget_->activeLimitBytes();
  return limit > active ? limit - active : 0;
}

bool Cache::commitSpill(uint64_t requestId, uint32_t pages) {
  Request &active = request(requestId);
  if (active.mode != KvMode::Spill)
    throw std::logic_error("only a Spill request commits quota");
  return kvTierWritable() && commit(active, pages);
}

bool Cache::convertToSpill(uint64_t requestId, uint32_t pages) {
  Request &active = request(requestId);
  if (!spill_ || active.mode == KvMode::Spill || spillGate() != SpillGate::Open ||
      !commit(active, pages))
    return false;
  active.mode = KvMode::Spill;
  active.spillOrder = ++spillSequence_;
  oweCopies(active, active.cachedBlocks);
  // The blocks it shares with Spill requests may spill now, and its own
  // writes start at once: what lands frees pages its retry waits for
  // (pendingPages). Its blocks whose copies landed before, as a chain it
  // restored has them, spill at once into free Index pages, whose Full
  // pages the retry takes.
  rescanSpills();
  static_cast<void>(writeThrough());
  static_cast<void>(moveLandedPages(false));
  return true;
}

uint32_t Cache::copiedBlocks(uint64_t block) const {
  if (!block)
    return 0;
  const KvCache::Chain chain = kv_.chain(block);
  return static_cast<uint32_t>(std::count_if(chain.blocks.begin(), chain.blocks.end(),
                                             [&](uint64_t id) { return kv_.kvCopied(id); }));
}

bool Cache::commit(Request &active, uint32_t pages) {
  if (pages && !diskBudget_->commit(uint64_t{pages} * tier_->slotBytes(), heldSlotBytes()))
    return false;
  active.committedPages += pages;
  return true;
}

void Cache::recommit(Request &active) {
  if (!commit(active, 1))
    ++spilled_.revokedCommitments;
}

void Cache::returnCommitment(Request &active, uint32_t pages) noexcept {
  pages = std::min(pages, active.committedPages);
  if (!pages)
    return;
  active.committedPages -= pages;
  diskBudget_->uncommit(uint64_t{pages} * tier_->slotBytes());
}

void Cache::oweCopies(Request &active, std::span<const uint64_t> blocks) {
  for (uint64_t block : blocks)
    if (!kv_.kvCopied(block))
      active.owedCopies.push_back(block);
}

uint32_t Cache::writeThrough() {
  uint32_t started = 0;
  if (!spill_ || !kvTierWritable())
    return started;
  // After a write found the volume full, until the volume, measured again,
  // leaves the copies of the KV requests hold room for another slot. A write
  // draws its request's commitment, or commits its slot, within the active
  // limit, and gives up cached copies for it (acquireDiskSlot), which a
  // size's quota lets fill all of that room.
  if (writeThroughPaused_) {
    if (heldSlotBytes() + tier_->slotBytes() > diskBudget_->activeLimitBytes())
      return started;
    writeThroughPaused_ = false;
  }
  // The requests that began to spill first write first.
  std::vector<Request *> spilling;
  for (auto &[_, active] : requests_)
    if (active.mode == KvMode::Spill && !active.owedCopies.empty())
      spilling.push_back(&active);
  std::sort(spilling.begin(), spilling.end(),
            [](const Request *left, const Request *right) { return left->spillOrder < right->spillOrder; });
  for (Request *active : spilling)
    if (!writeThrough(*active, started))
      break;
  return started;
}

bool Cache::writeThrough(Request &active, uint32_t &started) {
  while (!active.owedCopies.empty()) {
    const uint64_t block = active.owedCopies.front();
    // Another request's write, or a restore, gave it a copy meanwhile.
    if (kv_.kvCopied(block)) {
      active.owedCopies.pop_front();
      returnCommitment(active, 1);
      continue;
    }
    if (!tier_->canDemote(WriteClass::Spill))
      return false;
    // Past its commitment, as when its output outgrows the projection, a
    // page commits its slot itself: the copy of a block a request holds is
    // active KV, within the active limit beside what other requests
    // committed. Short of that room its page stays Full, and theirs are
    // still written.
    if (!active.committedPages && !commit(active, 1))
      return true;
    // A slot gives up cached copies, never the only copy of a state in use;
    // without one the volume holds no more active KV, for any request.
    std::shared_ptr<KvDiskSlot> slot = acquireDiskSlot(model::DiskBudget::Draw::Commitment, false);
    if (!slot)
      return false;
    --active.committedPages;
    auto transfer = tier_->demote(kv_.page(block), slot, WriteClass::Spill, completionNotifier_);
    if (!transfer) {
      // A file that stopped taking writes took none: the slot goes back,
      // and the commitment it drew is the request's again.
      slot.reset();
      recommit(active);
      return false;
    }
    labelKv(block, slot);
    kv_.setSlot(block, std::move(slot));
    kv_.setTransfer(block, KvCache::Transfer::Draining);
    writes_.push_back({block, WriteClass::Spill, std::move(transfer)});
    active.owedCopies.pop_front();
    ++started;
  }
  return true;
}

Cache::Eligibility Cache::spillEligibility(uint64_t block) const {
  const uint32_t index = kv_.chainLength(block) - 1;
  if (index < spill_->densePages)
    return Eligibility::KeptFull;
  bool held = false;
  for (const auto &[_, holder] : requests_) {
    if (holder.cachedBlocks.size() <= index || holder.cachedBlocks[index] != block)
      continue;
    if (holder.mode != KvMode::Spill)
      return Eligibility::KeptFull;
    if (index + spill_->windowPages >= holder.cachedBlocks.size())
      return Eligibility::Later;
    held = true;
  }
  // Nothing frees a page no request holds: it stays a cached Full page.
  return held ? Eligibility::Eligible : Eligibility::KeptFull;
}

void Cache::rescanSpills() noexcept {
  for (auto &[_, active] : requests_)
    active.spillScan = 0;
}

TokenAdmission Cache::spillWritten(SpilledExtents extents) { return moveLandedPages(true, extents); }

TokenAdmission Cache::moveLandedPages(bool allocate, SpilledExtents extents) {
  if (!spill_)
    return {};
  std::vector<uint32_t> pages;
  for (auto &[_, active] : requests_) {
    if (active.mode != KvMode::Spill)
      continue;
    const auto blocks = static_cast<uint32_t>(active.cachedBlocks.size());
    const uint32_t end = blocks > spill_->windowPages ? blocks - spill_->windowPages : 0;
    // The scan's mark passes what is spilled or kept Full, and stops at the
    // first page that may spill later. Copies land in the order they were
    // written, so the scan ends at the first page whose copy has not.
    bool settled = true;
    for (uint32_t index = std::max(active.spillScan, spill_->densePages); index < end; ++index) {
      const uint32_t page = active.pages[index];
      const uint64_t block = active.cachedBlocks[index];
      // A writer that kept a page of its own beside its block's restore
      // keeps it Full.
      Eligibility eligibility = Eligibility::KeptFull;
      if (pool_.form(page) == kv::PageForm::Full && kv_.page(block) == page) {
        if (!kv_.kvCopied(block) || kv_.transfer(block) != KvCache::Transfer::None)
          break;
        eligibility = spillEligibility(block);
      }
      if (eligibility == Eligibility::Eligible)
        pages.push_back(page);
      settled = settled && eligibility == Eligibility::KeptFull;
      if (settled)
        active.spillScan = index + 1;
    }
  }
  // Requests on one prefix find its pages each.
  std::sort(pages.begin(), pages.end());
  pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
  // Free Index pages first, the runway's among them: a spill never waits
  // for an allocation. The rest, if it may allocate, asks for extents of
  // the index form.
  const auto now = std::min<size_t>(pages.size(), pool_.freePageCount(kv::PageForm::Index));
  if (!allocate)
    pages.resize(now);
  for (const std::span<const uint32_t> part :
       {std::span<const uint32_t>(pages).first(now), std::span<const uint32_t>(pages).subspan(now)}) {
    if (part.empty())
      continue;
    const KvPageMoveResult moved = pool_.movePages(part, kv::PageForm::Index);
    if (!moved.granted())
      return {TokenAdmissionFailure::Denied, static_cast<uint32_t>(part.size()),
              pool_.freePageCount(kv::PageForm::Index), moved.failure, kv::PageForm::Index};
    followMoves(moved.moves);
    spilled_.spillMoves += part.size();
  }
  if (!allocate)
    return {};
  // Then the runway, an extent's free Index pages, is refilled: the next
  // spill frees Full pages without an allocation, at the memory wall too,
  // where a converted request's pages spill through it (pendingPages).
  const uint32_t free = pool_.freePageCount(kv::PageForm::Index);
  if (free < pool_.extentPages()) {
    if (const metal::AllocationResult runway = pool_.fillRunway(kv::PageForm::Index); !runway)
      return {TokenAdmissionFailure::Denied, pool_.extentPages() - free, free, runway.failure,
              kv::PageForm::Index};
  }
  // The memory the moves freed leaves with the Full extents they emptied,
  // once the runway stands: at the wall, which refuses it, they stay, and
  // the requests grow into their pages. Released, the Full pages left first
  // gather in the fewest extents.
  if (!pages.empty() && extents != SpilledExtents::Kept) {
    if (extents == SpilledExtents::Released)
      while (compactExtent(kv::PageForm::Full)) {
      }
    static_cast<void>(pool_.reclaimEmptyExtents(extents == SpilledExtents::KeepRunway,
                                                std::numeric_limits<uint32_t>::max(), kv::PageForm::Full));
  }
  return {};
}

bool Cache::holdsIndexPages(uint64_t requestId) const { return request(requestId).indexPages != 0; }

void Cache::poisonCopy(uint64_t block) {
  // Lanes that read it in one command report it each.
  if (!kv_.matchable(block))
    return;
  // As a failed restore does: nothing below it matches any more.
  states_.invalidate(block);
  kv_.poison(block);
  poisoned_.push_back(block);
  dropPoisoned();
}

KvSpillSnapshot Cache::spillSnapshot() const {
  KvSpillSnapshot result = spilled_;
  result.spilledPages = kv_.indexBlocks();
  result.writeThroughInFlight = writes(WriteClass::Spill);
  result.writeThroughPaused = writeThroughPaused_;
  result.rewrites = kv_.rewrites();
  result.restore = restoreLatency_.snapshot();
  if (spill_) {
    result.spilledBytes = uint64_t{result.spilledPages} * tier_->slotBytes();
    result.activeLimitBytes = diskBudget_->activeLimitBytes();
    result.heldBytes = heldSlotBytes();
    result.committedBytes = diskBudget_->committedBytes();
  }
  return result;
}

void Cache::checkInvariants() const {
  const auto check = [](bool holds, const char *what) {
    if (!holds)
      throw std::logic_error(std::string("cache invariant broken: ") + what);
  };
  kv_.checkInvariants([this](uint64_t block) { return states_.onDisk(block); });
  uint64_t committed = 0;
  for (const auto &[_, active] : requests_) {
    check(active.cachedBlocks.size() <= active.pages.size(),
          "a request has a page for each of its blocks");
    uint32_t indexPages = 0;
    for (uint32_t index = 0; index < active.pages.size(); ++index) {
      const uint32_t page = active.pages[index];
      check(pool_.pageActive(page), "a request holds its pages");
      if (pool_.form(page) != kv::PageForm::Index)
        continue;
      ++indexPages;
      // Rows that read the whole page table read only Full pages.
      check(index < active.cachedBlocks.size() && kv_.blockOn(page) == active.cachedBlocks[index],
            "an Index page holds the request's own block");
      check(!spill_ || index >= spill_->densePages, "no page of the dense history holds only its index");
    }
    check(indexPages == active.indexPages, "a request counts its Index pages");
    if (!active.dirtyFrom && active.revision) {
      // What the last cut said holds still.
      const model::KvPageSources *sources = active.sources.get();
      check((sources != nullptr) == (indexPages != 0),
            "a cut has sources exactly when a page is Index");
      for (uint32_t page = 0; sources && page < active.pages.size(); ++page)
        check((*sources)[page].block == sourceOf(active.pages[page]).block,
              "a cut's source names its page's block");
    }
    check(active.mode == KvMode::Spill || (!active.committedPages && active.owedCopies.empty()),
          "only a Spill request commits quota or owes copies");
    // Its pages spill only up to the first block on its own page whose copy
    // has not landed.
    if (active.mode == KvMode::Spill) {
      const std::unordered_set<uint64_t> owed(active.owedCopies.begin(), active.owedCopies.end());
      for (uint32_t index = 0; index < active.cachedBlocks.size(); ++index) {
        const uint64_t block = active.cachedBlocks[index];
        check(kv_.page(block) != active.pages[index] || kv_.kvCopied(block) || owed.contains(block),
              "a Spill request owes the copy of each block on its pages without one");
      }
    }
    committed += active.committedPages;
  }
  if (spill_)
    check(committed * tier_->slotBytes() == diskBudget_->committedBytes(),
          "the commitments are the requests'");
  for (const Write &write : writes_)
    check(kv_.transfer(write.block) == KvCache::Transfer::Draining, "a page being written is Draining");
  for (const auto &[block, _] : restores_)
    check(kv_.transfer(block) == KvCache::Transfer::Filling, "a page being restored is Filling");
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
  const bool spills = active.mode == KvMode::Spill;
  const uint32_t densePages = spill_ ? spill_->densePages : 0;
  // Resident blocks are a rooted prefix of the chain. Past a Spill request's
  // dense history a disk-only block comes back as its index alone. A
  // spilled block comes back whole for a Resident request that would be
  // its only holder, once no transfer moves it: no other request can read
  // its page while its bytes land.
  uint32_t full = 0, index = 0;
  std::vector<uint32_t> unspilled;
  for (uint32_t position = 0; position < chain.blocks.size(); ++position) {
    const uint32_t page = chain.pages[position];
    const uint64_t block = chain.blocks[position];
    if (page == KvCache::noPage)
      ++(spills && position >= densePages ? index : full);
    else if (!spills && pool_.form(page) == kv::PageForm::Index && !kv_.held(block) &&
             kv_.transfer(block) == KvCache::Transfer::None)
      unspilled.push_back(page);
  }
  if ((full || index) && !tier_)
    throw std::logic_error("disk-only KV block without a disk tier");
  // All or nothing: pages one admission took go back when the next fails.
  std::vector<uint32_t> fullPages, indexPages;
  const auto giveBack = [&] {
    for (const std::vector<uint32_t> *pages : {&fullPages, &indexPages})
      for (uint32_t page : *pages)
        pool_.releasePage(page, false);
  };
  if (full) {
    if (const TokenAdmission admission = admitPages(full, fullPages); !admission.granted())
      return admission;
  }
  if (index) {
    if (const TokenAdmission admission = admitPages(index, indexPages, kv::PageForm::Index);
        !admission.granted()) {
      giveBack();
      return admission;
    }
  }
  if (!unspilled.empty()) {
    const KvPageMoveResult moved = pool_.movePages(unspilled, kv::PageForm::Full);
    if (!moved.granted()) {
      giveBack();
      return {TokenAdmissionFailure::Denied, static_cast<uint32_t>(unspilled.size()),
              pool_.freePageCount(), moved.failure};
    }
    followMoves(moved.moves);
    spilled_.unspills += unspilled.size();
  }
  kv_.retainActive(chain.blocks.back());
  size_t nextFull = 0, nextIndex = 0;
  for (uint32_t position = 0; position < chain.blocks.size(); ++position) {
    const uint64_t block = chain.blocks[position];
    uint32_t &page = chain.pages[position];
    if (page == KvCache::noPage) {
      // Root first, so every restored block finds its parent resident. The
      // admission's reference is the request's.
      page = spills && position >= densePages ? indexPages[nextIndex++] : fullPages[nextFull++];
      kv_.adoptPage(block, page);
      startRestore(block);
    } else {
      // A spilled block moved back to a Full page is restored whole.
      const bool moved = kv_.page(block) != page;
      page = kv_.page(block);
      kv_.holdPage(page);
      if (moved)
        startRestore(block);
    }
    active.indexPages += pool_.form(page) == kv::PageForm::Index;
    if (auto restore = restores_.find(block); restore != restores_.end()) {
      restore->second.waiters.push_back(requestId);
      ++active.pendingRestores;
    }
  }
  active.pages = std::move(chain.pages);
  active.cachedBlocks = std::move(chain.blocks);
  pagesChanged(active, 0);
  // A Spill request's commitment holds a slot for each page without a copy
  // (copiedBlocks).
  if (spills)
    oweCopies(active, active.cachedBlocks);
  if (tier_)
    tier_->submitRestores();
  return {};
}

KvRestoreStatus Cache::kvRestoreStatus(uint64_t requestId) const {
  const Request &active = request(requestId);
  if (active.restoreFailed)
    return KvRestoreStatus::Failed;
  return active.pendingRestores ? KvRestoreStatus::Pending : KvRestoreStatus::None;
}

void Cache::startRestore(uint64_t block) {
  kv_.setTransfer(block, KvCache::Transfer::Filling);
  Restore &restore = restores_[block];
  restore.started = AwakeClock::now();
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
  if (!tier_->canDemote(WriteClass::Reclaim)) {
    ++kvTier_.demotionsRefused;
    return transfersInFlight() ? LeafReclaim::Pending : LeafReclaim::Impossible;
  }
  // Room in the quota or the tier that transfers in flight will free is
  // worth waiting for; room that nothing will free is not, and the leaf goes.
  std::shared_ptr<KvDiskSlot> slot =
      acquireDiskSlot(model::DiskBudget::Draw::Cache, kvNeededByStateInUse(block));
  if (!slot)
    return transfersInFlight() ? LeafReclaim::Pending : LeafReclaim::Impossible;
  // Making room may have taken the states the leaf was kept for; it then
  // goes like any leaf nothing needs.
  if (!kvNeededByState(block))
    return LeafReclaim::Impossible;
  auto transfer = tier_->demote(kv_.page(block), slot, WriteClass::Reclaim, completionNotifier_);
  if (!transfer) {
    ++kvTier_.demotionsRefused;
    return transfersInFlight() ? LeafReclaim::Pending : LeafReclaim::Impossible;
  }
  labelKv(block, slot);
  kv_.setSlot(block, std::move(slot));
  kv_.setTransfer(block, KvCache::Transfer::Draining);
  writes_.push_back({block, WriteClass::Reclaim, std::move(transfer)});
  return LeafReclaim::Started;
}

void Cache::labelKv(uint64_t block, const std::shared_ptr<KvDiskSlot> &slot) {
  if (!persistent_)
    return;
  const KvCache::Key key = kv_.key(block);
  KvBlockLabel label{block, key.parent, kv_.lastUsed(block), key.identity.lo, key.identity.hi, {}};
  std::copy(key.tokens.begin(), key.tokens.end(), label.tokens.begin());
  tier_->label(slot, encodeLabel(label));
}

std::shared_ptr<KvDiskSlot> Cache::acquireDiskSlot(model::DiskBudget::Draw draw, bool inUse) {
  // Making room is no use while the limit cannot hold a slot.
  if (diskBudget_ &&
      (draw == model::DiskBudget::Draw::Commitment ? diskBudget_->activeLimitBytes()
                                                   : diskBudget_->limitBytes()) < tier_->slotBytes())
    return {};
  // A committed page is a request's KV, which comes before any cached copy:
  // once nothing else is left, it takes the spilled chain of a state in RAM
  // too, the state first. Only write-through draws on a commitment, and it
  // holds no state.
  const bool committed = draw == model::DiskBudget::Draw::Commitment;
  for (;;) {
    if (auto slot = tier_->acquireSlot(draw))
      return slot;
    if (!freeDiskSpace(inUse) && !(committed && evictStateOnDiskLeaf()))
      return {};
  }
}

bool Cache::evictStateOnDiskLeaf() {
  for (auto leaf = kv_.diskCandidate(false); leaf; leaf = kv_.diskCandidate(false, leaf->id)) {
    if (states_.stateResident(leaf->id) && !kvNeededByStateInUse(leaf->id)) {
      states_.evict(leaf->id);
      return true;
    }
  }
  return false;
}

bool Cache::freeDiskSpace(bool inUse) {
  if (persistent_)
    return dropUnneededCopy() || dropOldestPoint(inUse);
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
  // A leaf holding a state in use stays with that state, and one holding a
  // state in RAM, as a spilled leaf can, stays until reclaim takes the state
  // or a committed page evicts it (evictStateOnDiskLeaf): a reclaim makes
  // room here for that state's own write, or for a demotion beside the state
  // it picked.
  auto kvLeaf = kv_.diskCandidate(false);
  while (kvLeaf && (kvNeededByStateInUse(kvLeaf->id) || states_.stateResident(kvLeaf->id)))
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

bool Cache::dropOldestPoint(bool inUse) {
  const auto redundant = states_.diskCandidate(true);
  const auto only = states_.diskCandidate(false);
  std::optional<CacheEvictionCandidate> victim = older(only, redundant) ? only : redundant;
  if (!victim && inUse)
    victim = states_.inUseDiskCandidate();
  if (!victim)
    return false;
  if (states_.stateResident(victim->id))
    states_.dropDisk(victim->id);
  else
    states_.evict(victim->id);
  return true;
}

uint32_t Cache::fitDiskLimit() {
  if (!diskBudget_)
    return 0;
  // Active KV first, the copies of the KV requests hold and the commitments:
  // a commitment goes while they exceed their limit, the newest Spill
  // request's first, and the request runs on with what it has.
  while (diskBudget_->committedBytes() &&
         heldSlotBytes() + diskBudget_->committedBytes() > diskBudget_->activeLimitBytes()) {
    Request *newest = nullptr;
    for (auto &[_, active] : requests_)
      if (active.committedPages && (!newest || active.spillOrder > newest->spillOrder))
        newest = &active;
    if (!newest)
      throw std::logic_error("quota is committed to no request");
    returnCommitment(*newest, newest->committedPages);
    ++spilled_.revokedCommitments;
  }
  // Then cached copies, within what active KV leaves of the limit.
  const auto cachedPastLimit = [&] {
    const uint64_t held = heldSlotBytes(), used = diskBudget_->usedBytes();
    const uint64_t active = held + diskBudget_->committedBytes(), limit = diskBudget_->limitBytes();
    return used - std::min(used, held) > (limit > active ? limit - active : 0);
  };
  uint32_t dropped = 0;
  while (cachedPastLimit() && freeDiskSpace(false))
    ++dropped;
  return dropped;
}

void Cache::keepDiskLimit() {
  if (diskMeasure_)
    diskMeasure_();
  static_cast<void>(fitDiskLimit());
}

bool Cache::dropUnneededCopy() {
  const auto unneeded = kv_.unneededCopy();
  if (!unneeded)
    return false;
  // A Full block's copy is redundant; a spilled or disk-only leaf's is its
  // only K/V, and it goes with the block.
  if (const uint32_t page = kv_.page(unneeded->id);
      page != KvCache::noPage && pool_.form(page) == kv::PageForm::Full)
    kv_.setSlot(unneeded->id, nullptr);
  else
    kv_.erase(unneeded->id);
  return true;
}

bool Cache::pollTransfers() {
  bool progressed = states_.pollOffload();
  if (tier_ && pollTierTransfers())
    progressed = true;
  if (tier_ && progressed)
    dropPoisoned();
  // A write that found the volume full has the volume measured at once,
  // once the writes that landed or failed are collected: what the tier holds
  // then counts no failed write.
  if (diskMeasure_ && diskBudget_->fullReported())
    keepDiskLimit();
  return progressed;
}

bool Cache::pollTierTransfers() {
  bool progressed = false;
  tier_->poll();
  // Waiting restores start in block order and stop at the first the tier
  // refuses: started ones stay a prefix of each chain, and a full tier is
  // asked once per poll, not once per waiting block. Those started go out
  // together after the pass.
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
    kv_.setTransfer(block, KvCache::Transfer::None);
    if (restored) {
      ++kvTier_.restores;
      ++(pool_.form(kv_.page(block)) == kv::PageForm::Index ? spilled_.indexRestores
                                                             : spilled_.wholeRestores);
      restoreLatency_.add(static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(AwakeClock::now() - restore.started)
              .count()));
    } else {
      ++kvTier_.restoreFailures;
    }
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
  tier_->submitRestores();
  for (auto entry = writes_.begin(); entry != writes_.end();) {
    if (!entry->transfer->ready()) {
      ++entry;
      continue;
    }
    const uint64_t block = entry->block;
    const WriteClass write = entry->write;
    const bool written = entry->transfer->finish();
    entry = writes_.erase(entry);
    kv_.setTransfer(block, KvCache::Transfer::None);
    progressed = true;
    // A write that failed leaves its page as it was, without a copy. Whatever
    // the write was for, each Spill request that holds the block on its page
    // owes the copy again, first: it took the copy in flight as landed, or
    // drew the slot on its commitment. It commits a slot for it once more,
    // which the copy it writes draws or the copy another writes returns, as
    // when it published the block; one that still owes it keeps the slot it
    // committed. The pages past the block spill once its copy has landed. A
    // writer that kept a page of its own beside the block never spills that
    // page, and owes nothing.
    if (!written) {
      kv_.setSlot(block, nullptr);
      const uint32_t index = kv_.chainLength(block) - 1;
      for (auto &[_, holder] : requests_) {
        if (holder.mode != KvMode::Spill || index >= holder.cachedBlocks.size() ||
            holder.cachedBlocks[index] != block || holder.pages[index] != kv_.page(block) ||
            std::ranges::find(holder.owedCopies, block) != holder.owedCopies.end())
          continue;
        recommit(holder);
        holder.owedCopies.push_front(block);
      }
    }
    switch (write) {
    case WriteClass::Reclaim:
      ++(written ? kvTier_.demotions : kvTier_.demotionFailures);
      // A request that matched the block meanwhile keeps its page, and so
      // does a state published on it in RAM; the copy makes the next reclaim
      // of the leaf free.
      if (written && kv_.residentLeaf(block) && !states_.stateResident(block))
        kv_.dropPage(block);
      break;
    case WriteClass::Persist:
      // The page stays: the copy is for the next process.
      ++(written ? kvTier_.copies : kvTier_.copyFailures);
      break;
    case WriteClass::Spill:
      // The page stays Full until spillWritten moves it.
      if (written) {
        ++spilled_.writeThroughPages;
        spilled_.writeThroughBytes += tier_->slotBytes();
        break;
      }
      ++spilled_.writeThroughFailures;
      // A volume that filled holds write-through back until it has room; a
      // file that stopped taking writes stops it for good.
      writeThroughPaused_ = kvTierWritable();
      break;
    }
  }
  return progressed;
}

PersistStatus Cache::persist(uint64_t block) {
  if (!persistent_)
    throw std::logic_error("only a persistent tier keeps restore points");
  if (!kv_.contains(block) || !states_.ordinary(block) || superseded(block))
    return PersistStatus::Unneeded;
  if (!kvTierWritable())
    return PersistStatus::Refused;
  const KvCache::Chain chain = kv_.chain(block);
  // A point larger than what the tier may hold now is not kept: making room
  // for it would give up every other copy first, and waiting for room would
  // hold back the points after it.
  if (diskBudget_ && uint64_t{chain.blocks.size()} * tier_->slotBytes() + states_.bytes(block) >
                         diskBudget_->limitBytes())
    return PersistStatus::Refused;
  // Nothing starts while a restore waits: the tier serves the request first.
  if (!restores_.empty() && !states_.onDisk(block))
    return PersistStatus::Busy;
  // The state goes first, as the write budget admits it; its chain follows.
  const PersistStatus state = states_.persist(block);
  if (state != PersistStatus::Durable && state != PersistStatus::Started)
    return state;
  bool inFlight = state == PersistStatus::Started;
  for (size_t index = 0; index < chain.blocks.size(); ++index) {
    const uint64_t id = chain.blocks[index];
    if (kv_.slot(id)) {
      inFlight = inFlight || kv_.transfer(id) != KvCache::Transfer::None;
      continue;
    }
    // Only a resident block lacks a disk copy.
    if (!restores_.empty() || !tier_->canDemote(WriteClass::Persist))
      return inFlight ? PersistStatus::Started : PersistStatus::Busy;
    std::shared_ptr<KvDiskSlot> slot = acquireDiskSlot(model::DiskBudget::Draw::Cache, false);
    // Making room gives up the oldest point, which may be this one: it does
    // not fit then. Otherwise its chain stays whole, as a state needs it.
    if (!slot || !states_.onDisk(block))
      return PersistStatus::Refused;
    auto transfer = tier_->demote(chain.pages[index], slot, WriteClass::Persist, completionNotifier_);
    if (!transfer)
      return inFlight ? PersistStatus::Started : PersistStatus::Busy;
    labelKv(id, slot);
    kv_.setSlot(id, std::move(slot));
    kv_.setTransfer(id, KvCache::Transfer::Draining);
    writes_.push_back({id, WriteClass::Persist, std::move(transfer)});
    inFlight = true;
  }
  return inFlight ? PersistStatus::Started : PersistStatus::Durable;
}

void Cache::relabelStates() const { states_.relabel(); }

bool Cache::superseded(uint64_t block) const {
  if (!kv_.stateBelow(block))
    return false;
  uint32_t branches = 0;
  for (const uint64_t child : kv_.children(block)) {
    if (holdsOrdinaryState(child) && ++branches > 1)
      return false;
  }
  return branches == 1;
}

bool Cache::holdsOrdinaryState(uint64_t block) const {
  std::vector<uint64_t> pending{block};
  while (!pending.empty()) {
    const uint64_t id = pending.back();
    pending.pop_back();
    if (states_.ordinary(id))
      return true;
    // Only where a state lies below is there anything to find.
    if (kv_.stateBelow(id)) {
      const std::vector<uint64_t> children = kv_.children(id);
      pending.insert(pending.end(), children.begin(), children.end());
    }
  }
  return false;
}

CacheSnapshot Cache::snapshot() const {
  KvTierSnapshot tier = kvTier_;
  tier.pendingPages = pendingPages(kv::PageForm::Full);
  tier.diskBlocks = kv_.diskBlocks();
  if (diskBudget_) {
    tier.capacityBytes = diskBudget_->capacityBytes();
    tier.limitBytes = diskBudget_->limitBytes();
    tier.usedBytes = diskBudget_->usedBytes();
    tier.fileBytes = diskBudget_->fileBytes();
    tier.volumeAvailableBytes = diskBudget_->volumeAvailableBytes();
    tier.writesGated = diskBudget_->writesStopped()  ? DiskWritesGate::Closed
                       : !diskBudget_->admits(false) ? DiskWritesGate::Budget
                                                     : DiskWritesGate::No;
    tier.writeBudgetBytes =
        static_cast<uint64_t>(std::max<int64_t>(0, diskBudget_->writeBudgetBytes()));
    tier.writesDeferredMicroseconds = diskBudget_->writesDeferredMicroseconds();
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
          extentCompactMaxMilliseconds_,
          persistent_,
          adoption_};
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
