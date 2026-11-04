#include "engine/Cache.hpp"

#include "AwakeClock.hpp"
#include "engine/DiskLabels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <compare>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <tuple>
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

// The whole extents `pages` more free pages of a form complete beside
// `free` ones, which compaction gathers, past the empty one a runway keeps.
uint32_t completedExtents(uint32_t free, uint32_t pages, uint32_t extentPages, bool keepRunway) noexcept {
  const uint32_t kept = keepRunway ? 1 : 0;
  const auto beyond = [&](uint32_t count) { return std::max(count / extentPages, kept) - kept; };
  return beyond(free + pages) - beyond(free);
}

} // namespace

Cache::Cache(KvPool &pool, KvTier *kvTier, std::shared_ptr<model::DiskBudget> diskBudget,
             std::optional<CacheSpillPolicy> spill, std::optional<CacheWindowPolicy> window)
    : pool_(pool), tier_(kvTier), spill_(spill), persistent_(kvTier && kvTier->persistent()),
      window_(window), diskBudget_(std::move(diskBudget)),
      kv_(pool, recency_, spill.has_value(), persistent_, window ? window->windowPages : 0),
      states_(kv_, recency_, [this](bool inUse) { return freeDiskSpace(inUse, false); }, persistent_) {
  // Spilled K/V lives in the tier's copies, within its quota.
  if (spill_ && (!tier_ || !diskBudget_))
    throw std::invalid_argument("a cache whose requests spill needs a disk tier and its quota");
  // A spillable layout's pages keep every layer (kv::Layout::slimmable).
  if (window_ && (!window_->windowPages || spill_))
    throw std::invalid_argument("a windowed cache keeps a window of pages, and its requests do not spill");
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
    kv::PageForm form = kv::PageForm::Full;
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
    const auto label = decodeKvSlotLabel(blocks[index].label, window_.has_value());
    if (label)
      highestId = std::max(highestId, label->block.id);
    // A copy of a page of another form than Full or Slim is a later
    // build's, which this one does not restore.
    if (!label || !label->block.id || label->block.parent >= label->block.id ||
        (label->form != kv::PageForm::Full && label->form != kv::PageForm::Slim) ||
        !recorded.emplace(label->block.id, Recorded{label->block, label->form, index}).second)
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
  // A state comes back with its whole chain, whose tail with a window holds
  // Full copies; a block only for a state.
  const auto tailFull = [&](std::map<uint64_t, Recorded>::const_iterator block) {
    for (uint32_t index = 0; window_ && block != recorded.end() && index < window_->windowPages;
         ++index, block = recorded.find(block->second.label.parent))
      if (block->second.form != kv::PageForm::Full)
        return false;
    return true;
  };
  std::vector<std::pair<StateLabel, size_t>> kept;
  for (size_t index = 0; index < states.size(); ++index) {
    const auto label = decodeLabel<StateLabel>(states[index].label);
    if (label)
      highestId = std::max(highestId, label->block);
    const auto block = label ? recorded.find(label->block) : recorded.end();
    if (block == recorded.end() || !block->second.tokens ||
        block->second.tokens != label->tokens || !tailFull(block)) {
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
                                     std::move(slot), block.label.lastUsed, block.form)) {
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
  // The rows of the request's next command lie at or past its published
  // blocks, and their windows reach back no further than windowPages pages
  // before them: those pages and every one after hold every layer.
  if (window_) {
    for (uint32_t index = windowStart(active); index < active.pages.size(); ++index)
      if (pool_.form(active.pages[index]) != kv::PageForm::Full)
        throw std::logic_error("a page a row's window reads holds only the layers no window bounds");
  }
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
  if (pool_.form(page) != kv::PageForm::Index)
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
    // A writer that keeps its page beside the block, while a restore fills
    // the block's or a write reads it, trims it on its own; so does each
    // other request that holds the page the block left for the writer's.
    if (window_ && !replacePage && kv_.page(inserted.id) != writerPage)
      active.privatePages.push_back(logical);
    if (window_ && inserted.previousPage != KvCache::noPage &&
        pool_.form(inserted.previousPage) == kv::PageForm::Full) {
      for (auto &[_, holder] : requests_)
        if (logical < holder.cachedBlocks.size() && holder.pages[logical] == inserted.previousPage)
          holder.privatePages.push_back(logical);
    }
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
  if (window_ && !kv_.tailFull(kvBlock))
    throw std::logic_error("a state's tail holds only the layers no window bounds");
  states_.publishCompositeState(kvBlock, std::move(state), checkpoint);
}

bool Cache::publishStateToDisk(uint64_t kvBlock, const StateWriter &write, bool checkpoint) {
  if (window_ && !kv_.tailFull(kvBlock))
    throw std::logic_error("a state's tail holds only the layers no window bounds");
  return states_.publishStateToDisk(kvBlock, write, checkpoint);
}

bool Cache::stateTailFull(uint64_t kvBlock) {
  if (!window_ || kv_.tailFull(kvBlock))
    return true;
  ++refusedStates_;
  return false;
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
                                   uint64_t writtenTokens, std::array<uint32_t, 3> held) {
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
  if (const TokenAdmission admission =
          admitPages(needed - previousSize, acquired, kv::PageForm::Full,
                     held[static_cast<size_t>(kv::PageForm::Full)]);
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
                                  kv::PageForm form, uint32_t held) {
  KvPageAcquisition acquired = pool_.acquirePages(count, form, held);
  if (!acquired.granted()) {
    // Demoted and spilled pages free theirs when their copies land: wait
    // once those on their way cover what the free pages beyond those held
    // do not, and until then reclaim.
    const uint32_t free = pool_.freePageCount(form);
    const uint32_t missing = count - std::min(count, free - std::min(free, held));
    const uint32_t pending = pendingPages(form);
    const bool covered = pending > 0 && pending >= missing;
    return {covered ? TokenAdmissionFailure::Pending : TokenAdmissionFailure::Denied,
            count, pool_.freePageCount(form), acquired.failure, form};
  }
  pages = std::move(acquired.pages);
  return {};
}

Returning Cache::returning() const { return returningOf(std::nullopt); }

Returning Cache::returningFor(std::span<const uint64_t> requestIds) const { return returningOf(requestIds); }

Returning Cache::returningOf(std::optional<std::span<const uint64_t>> requestIds) const {
  Returning result;
  std::vector<const Request *> holders;
  if (requestIds)
    for (const uint64_t id : *requestIds)
      if (const auto found = requests_.find(id); found != requests_.end())
        holders.push_back(&found->second);
  // The requests' own pages: the blocks of their page tables on their pages.
  const auto own = [&](uint64_t block) {
    if (!requestIds)
      return true;
    const uint32_t index = kv_.chainLength(block) - 1;
    return std::ranges::any_of(holders, [&](const Request *active) {
      return index < active->cachedBlocks.size() && active->cachedBlocks[index] == block &&
             active->pages[index] == kv_.page(block);
    });
  };
  uint32_t spilling = 0;
  const std::unordered_set<uint64_t> freeing = freeingDemotions();
  for (const Write &write : writes_) {
    if (!own(write.block))
      continue;
    // A demotion lands a page only where its page frees as it lands; every
    // one counts as a transfer in flight.
    if (write.write == WriteClass::Reclaim)
      result.landingPages[static_cast<size_t>(pool_.form(kv_.page(write.block)))] += freeing.contains(write.block);
    else if (spill_ && spillsOnLanding(write))
      ++spilling;
    else
      continue;
    ++result.transfers;
    result.oldestMilliseconds = std::max(result.oldestMilliseconds, millisecondsSince(write.started));
  }
  constexpr auto full = static_cast<size_t>(kv::PageForm::Full);
  result.landingPages[full] += std::min(spilling, pool_.freePageCount(kv::PageForm::Index));
  if (spill_ && writes(WriteClass::Spill)) {
    // Requests on one prefix owe its blocks each; a block another request's
    // write drains already lands among the transfers.
    std::unordered_set<uint64_t> owed;
    for (const auto &[_, holder] : requests_)
      if (holder.mode == KvMode::Spill && (!requestIds || std::ranges::find(holders, &holder) != holders.end()))
        for (const uint64_t block : holder.owedCopies)
          if (kv_.transfer(block) == KvCache::Transfer::None && spillEligibility(block) == Eligibility::Eligible)
            owed.insert(block);
    result.owedPages[full] = static_cast<uint32_t>(owed.size());
  }
  for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index})
    result.landingBytes += extentsCompleted(form, result.landingPages[static_cast<size_t>(form)]);
  // The owed pages complete extents beside every landing, the free pages
  // counted once.
  const uint32_t landing = requestIds ? returningOf(std::nullopt).landingPages[full] : result.landingPages[full];
  result.owedBytes = extentsCompleted(kv::PageForm::Full, result.owedPages[full], landing);
  return result;
}

uint32_t Cache::convertiblePages(uint64_t requestId) const {
  if (!spill_)
    return 0;
  const Request &active = request(requestId);
  const auto blocks = static_cast<uint32_t>(active.cachedBlocks.size());
  const uint32_t end = blocks > spill_->windowPages ? blocks - spill_->windowPages : 0;
  uint32_t pages = 0;
  for (uint32_t index = spill_->densePages; index < end; ++index) {
    const uint64_t block = active.cachedBlocks[index];
    const uint32_t page = active.pages[index];
    if (pool_.form(page) != kv::PageForm::Full || kv_.page(block) != page)
      continue;
    // As spillEligibility: every other holder spills and has passed it.
    const bool kept = std::ranges::any_of(requests_, [&](const auto &entry) {
      const Request &holder = entry.second;
      return &holder != &active && holder.cachedBlocks.size() > index && holder.cachedBlocks[index] == block &&
             (holder.mode != KvMode::Spill || index + spill_->windowPages >= holder.cachedBlocks.size());
    });
    pages += kept ? 0 : 1;
  }
  return pages;
}

uint32_t Cache::residentChainPages(std::span<const uint64_t> chain) const {
  uint32_t pages = 0;
  for (const uint64_t block : chain) {
    if (!kv_.contains(block))
      continue;
    const uint32_t page = kv_.page(block);
    pages += page != KvCache::noPage && pool_.form(page) != kv::PageForm::Index;
  }
  return pages;
}

Need Cache::growthNeed(uint32_t pages) const {
  constexpr kv::PageForm full = kv::PageForm::Full;
  const uint64_t extent = pool_.extentPages();
  const uint32_t free = pool_.freePageCount(full);
  const uint64_t beyond = pages - std::min(pages, free);
  Need need{.bytes = (beyond + extent - 1) / extent * extent * pool_.bytesPerPage(full)};
  need.keptPages[static_cast<size_t>(full)] = std::min(pages, free);
  return need;
}

bool Cache::freesOnLanding(uint64_t block, const std::unordered_set<uint64_t> &freeing) const {
  return kv_.leafOnceChildrenLeave(block) && !states_.stateResident(block) && !pool_.pageActive(kv_.page(block)) &&
         (kv_.residentLeaf(block) || std::ranges::all_of(kv_.children(block), [&](uint64_t child) {
            return kv_.page(child) == KvCache::noPage || freeing.contains(child);
          }));
}

std::unordered_set<uint64_t> Cache::freeingDemotions() const {
  std::unordered_set<uint64_t> freeing;
  for (const Write &write : writes_)
    if (write.write == WriteClass::Reclaim && freesOnLanding(write.block, freeing))
      freeing.insert(write.block);
  return freeing;
}

std::array<uint32_t, 3> Cache::demotedPages() const {
  std::array<uint32_t, 3> pages{};
  const std::unordered_set<uint64_t> freeing = freeingDemotions();
  for (const Write &write : writes_)
    if (freeing.contains(write.block))
      ++pages[static_cast<size_t>(pool_.form(kv_.page(write.block)))];
  return pages;
}

std::array<uint32_t, 3> Cache::demotionPages() const noexcept {
  std::array<uint32_t, 3> pages{};
  for (const Write &write : writes_)
    if (write.write == WriteClass::Reclaim)
      ++pages[static_cast<size_t>(pool_.form(kv_.page(write.block)))];
  return pages;
}

uint64_t Cache::extentsCompleted(kv::PageForm form, uint32_t pages, uint32_t beside, bool keepRunway) const noexcept {
  const uint32_t extent = pool_.extentPages();
  return uint64_t{completedExtents(pool_.freePageCount(form) + beside, pages, extent, keepRunway)} * extent *
         pool_.bytesPerPage(form);
}

// The walk: the cache as a source of the units the memory ledger takes,
// over the orders its reclaim steps take from.

Cache::Walk Cache::walkFor(const VictimQuery &query) const {
  const ClaimSpec claim = query.claim ? *query.claim : ClaimSpec{};
  const Need &need = query.need;
  Walk walk;
  // What the class may take of certain reuse, which a request would rebuild
  // at its whole prefill (MemoryLedger::dryRun has the same table): every
  // queued request's chain and the newest resume point for in-service work
  // and the host at critical pressure; the resume point but never a queued
  // chain for a start, optional work and a weights restore, since a running
  // lane beats a queued one but a new request does not; neither for growth
  // and the host below critical pressure. What is in use only in-service
  // work that can wait and reaches it, and the host at critical pressure. A
  // need the host refused runs as an Immediate claim of ordinary reach. A
  // device need's unit holds many victims, so the walk leaves out what the
  // claim may not take rather than mark it. Every walk keeps the chain a
  // start's placement reuses (VictimQuery::kept).
  const bool hostCritical = claim.cls == ClaimClass::Host && query.critical;
  const bool immediate = claim.timing == NeedTiming::Immediate || need.hostRefused;
  const bool queuedChains = claim.cls == ClaimClass::InService || hostCritical;
  const bool resumePoint = queuedChains || claim.cls == ClaimClass::Start || claim.cls == ClaimClass::Optional ||
                           claim.cls == ClaimClass::Floor;
  const bool inUse =
      (claim.cls == ClaimClass::InService && !immediate && claim.reach == ReclaimClass::InUse) || hostCritical;
  walk.scope.upTo = inUse ? ReclaimClass::InUse : ReclaimClass::Ordinary;
  walk.keepResumePoint = !resumePoint;
  walk.restored = queuedChains ? nullptr : &wanted_;
  walk.reused = query.kept;
  walk.timing = immediate ? ReclaimTiming::Immediate : ReclaimTiming::CanWait;
  // A state that needs a write frees its memory as a unit that lands later,
  // which the reclaims of an Immediate claim pass over.
  walk.skipStateWrites = walk.timing == ReclaimTiming::Immediate;
  // Memory for later, growth and a pass keep the empty extent the next
  // request takes, and every claim keeps the Index runway, where landed
  // copies spill at the memory wall (moveLandedPages) and a conversion
  // lands, unless the host is critical.
  walk.keepRunway = !query.critical && (claim.cls == ClaimClass::Optional || claim.cls == ClaimClass::Growth ||
                                         claim.cls == ClaimClass::Host);
  walk.keepIndexRunway = !query.critical;
  walk.hostRefused = need.hostRefused;
  // A state that goes ends a unit for a need of its buffers, and of bytes,
  // which its buffers meet once they leave the model's pool.
  walk.statesAreUnits = need.stateBuffers != 0 || (need.bytes && !need.hostRefused);
  if (need.bytes && !need.hostRefused) {
    // Memory the engine's limit refused, whatever pages it was for: an
    // extent released covers it.
    walk.device = true;
    walk.refused = Shortfall{pool_.extentPages(), kv::PageForm::Full};
    // A KV need takes the free pages of its form first, and those of the
    // other forms its attempt takes: releasing their extent would give it
    // bytes for an extent it then allocates again.
    if (need.pages || std::ranges::any_of(need.keptPages, [](uint32_t each) { return each != 0; })) {
      walk.form = need.form;
      for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index}) {
        const auto each = static_cast<size_t>(form);
        const uint32_t takes = form == need.form && need.pages ? pool_.freePageCount(form) : need.keptPages[each];
        walk.keptPages[each] = std::min(pool_.freePageCount(form), takes);
      }
    }
  } else if (need.pages) {
    // Pages in place; a host that refuses growth takes them in a page scan.
    walk.pages = need.pages;
    walk.form = need.form;
    walk.scope.pagesOnly = need.hostRefused;
    walk.refused = Shortfall{need.pages, need.form};
  } else if (need.stateBuffers || need.hostRefused) {
    walk.scope.statesOnly = true;
    walk.refused = Shortfall{};
  }
  return walk;
}

std::optional<Cache::WalkUnit> Cache::nextUnit(const WalkScope &scope, WalkCursor &cursor) const {
  using Phase = WalkCursor::Phase;
  const auto give = [&](Unit unit, CacheEvictionCandidate candidate) {
    cursor.last = unit;
    cursor.lastCandidate = candidate;
    return WalkUnit{unit, candidate};
  };
  const auto moveOn = [&](Phase next) {
    cursor = WalkCursor{.phase = next};
  };
  for (;;) {
    // The order of the unit given last moves past it now, as a scan moves on
    // once it has tried a candidate.
    const std::optional<Unit> last = std::exchange(cursor.last, std::nullopt);
    switch (cursor.phase) {
    case Phase::Checkpoints:
      if (scope.pagesOnly) {
        moveOn(scope.statesOnly ? Phase::Ordinary : Phase::DeadLeaves);
        continue;
      }
      if (!cursor.started) {
        cursor.state = states_.checkpointCandidate(false);
        cursor.started = true;
      } else if (last == Unit::Checkpoint) {
        cursor.state = states_.checkpointCandidate(false, {}, cursor.lastCandidate);
      }
      if (cursor.state)
        return give(Unit::Checkpoint, *cursor.state);
      moveOn(scope.statesOnly ? Phase::Ordinary : Phase::DeadLeaves);
      continue;
    case Phase::DeadLeaves:
      if (!cursor.started) {
        cursor.leaf = oldestDeadKvLeaf(0);
        cursor.started = true;
      } else if (last == Unit::DeadLeaf) {
        cursor.leaf = oldestDeadKvLeaf(cursor.lastCandidate.id);
      }
      if (cursor.leaf)
        return give(Unit::DeadLeaf, *cursor.leaf);
      moveOn(Phase::Ordinary);
      continue;
    case Phase::Ordinary:
    case Phase::InUse: {
      const bool inUse = cursor.phase == Phase::InUse;
      if (inUse && scope.upTo == ReclaimClass::Ordinary) {
        moveOn(Phase::Done);
        continue;
      }
      const Unit state = inUse ? Unit::UsedState : Unit::State;
      const Unit leaf = inUse ? Unit::UsedLeaf : Unit::Leaf;
      const Unit leafState = inUse ? Unit::UsedLeafState : Unit::LeafState;
      const auto stateAfter = [&](std::optional<CacheEvictionCandidate> after) {
        return inUse ? states_.inUseCandidate(after) : states_.ordinaryCandidate(false, {}, after);
      };
      if (!cursor.started) {
        // A page scan takes a state only where it sits on the leaf that goes
        // next.
        if (!scope.pagesOnly)
          cursor.state = stateAfter(std::nullopt);
        if (!scope.statesOnly)
          cursor.leaf = oldestKvLeaf(0, inUse, scope.pagesOnly);
        cursor.started = true;
      } else if (last == state) {
        cursor.state = stateAfter(cursor.lastCandidate);
      } else if (last == leaf || last == leafState) {
        cursor.leaf = oldestKvLeaf(cursor.lastCandidate.id, inUse, scope.pagesOnly);
      }
      if (cursor.statesHeld)
        cursor.state.reset();
      if (cursor.state && (!cursor.leaf || cursor.state->lastUsed <= cursor.leaf->lastUsed))
        return give(state, *cursor.state);
      if (cursor.leaf)
        return give(scope.pagesOnly && states_.stateResident(cursor.leaf->id) ? leafState : leaf, *cursor.leaf);
      moveOn(inUse ? Phase::Done : Phase::InUse);
      continue;
    }
    case Phase::Done:
      return std::nullopt;
    }
  }
}

std::optional<Cache::WalkUnit> Cache::nextStepUnit(const Walk &walk, WalkCursor &cursor) const {
  for (std::optional<WalkUnit> unit; (unit = nextUnit(walk.scope, cursor));) {
    if (!kept(*unit, walk))
      return unit;
  }
  return std::nullopt;
}

Cache::WalkCursor Cache::cursorAfter(Unit unit, uint64_t block) const {
  using Phase = WalkCursor::Phase;
  const bool known = unit == Unit::Checkpoint || unit == Unit::State || unit == Unit::UsedState;
  if (!known || !states_.contains(block))
    return WalkCursor{.phase = Phase::Done};
  return WalkCursor{.phase = unit == Unit::Checkpoint ? Phase::Checkpoints
                             : unit == Unit::State    ? Phase::Ordinary
                                                      : Phase::InUse,
                    .started = true,
                    .last = unit,
                    .lastCandidate = {block, states_.lastUsed(block)}};
}

void Cache::holdAfter(const Walk &walk, WalkCursor &cursor) const {
  using Phase = WalkCursor::Phase;
  const auto waits = [&](uint64_t block) {
    return !skipsState(walk, block) && states_.reclaimOutcome(block, walk.unwritten).pending;
  };
  if (cursor.phase == Phase::Checkpoints && cursor.last == Unit::Checkpoint && waits(cursor.lastCandidate.id)) {
    cursor = WalkCursor{.phase = walk.scope.statesOnly ? Phase::Ordinary : Phase::DeadLeaves};
    return;
  }
  if (cursor.phase != Phase::Ordinary && cursor.phase != Phase::InUse)
    return;
  // The states of the phase up to the unit, as a step reached them.
  const bool inUse = cursor.phase == Phase::InUse;
  const CacheEvictionCandidate at = cursor.lastCandidate;
  const bool atState = cursor.last == Unit::State || cursor.last == Unit::UsedState;
  for (std::optional<CacheEvictionCandidate> state = inUse ? states_.inUseCandidate() : states_.ordinaryCandidate(false);
       state && (state->lastUsed < at.lastUsed || (state->lastUsed == at.lastUsed && (!atState || state->id <= at.id)));
       state = inUse ? states_.inUseCandidate(state) : states_.ordinaryCandidate(false, {}, state)) {
    if (!kept({inUse ? Unit::UsedState : Unit::State, *state}, walk) &&
        waits(state->id)) {
      cursor.statesHeld = true;
      cursor.state.reset();
      return;
    }
  }
}

bool Cache::kept(const WalkUnit &unit, const Walk &walk) const {
  const uint64_t block = unit.candidate.id;
  const bool resumePoint = walk.keepResumePoint && block == states_.resumePoint();
  switch (unit.unit) {
  case Unit::Checkpoint:
  case Unit::State:
    return resumePoint || walk.keepsState(block);
  case Unit::LeafState:
  case Unit::UsedLeafState:
    return resumePoint || walk.keepsBlock(block);
  case Unit::Leaf:
  case Unit::UsedLeaf:
    return walk.keepsBlock(block);
  default:
    return false;
  }
}

bool Cache::certainReuse(uint64_t block) const {
  return block == states_.resumePoint() || std::ranges::find(wanted_.states(), block) != wanted_.states().end();
}

bool Cache::skipsState(const Walk &walk, uint64_t block) const {
  return walk.skipStateWrites && states_.reclaimWrites(block);
}

StateEviction Cache::reclaimState(const Walk &walk, uint64_t block) {
  if (skipsState(walk, block))
    return {};
  return states_.reclaim(block, walk.unwritten);
}

Cache::UnitTaken Cache::takeUnit(const Walk &walk, const WalkUnit &unit, ReclaimTiming timing) {
  const uint64_t block = unit.candidate.id;
  switch (unit.unit) {
  case Unit::Checkpoint:
  case Unit::State:
  case Unit::LeafState:
  case Unit::UsedState:
  case Unit::UsedLeafState: {
    const StateEviction eviction = reclaimState(walk, block);
    return {eviction.evicted, eviction.pending};
  }
  case Unit::DeadLeaf:
  case Unit::Leaf:
  case Unit::UsedLeaf:
    switch (reclaimKvLeaf(block, timing)) {
    case LeafReclaim::Started:
      return {true, false};
    case LeafReclaim::Pending:
      return {false, true};
    case LeafReclaim::Impossible:
      return {};
    }
    return {};
  default:
    return {};
  }
}

bool Cache::returns(const Walk &walk) const noexcept {
  return walk.refused ? returnsInFlight(*walk.refused) : transfersInFlight();
}

namespace {

// Victim::key: the unit above the block's id; a device need's unit holds
// its plan's serial above its place in the plan.
constexpr unsigned kUnitShift = 56;
constexpr unsigned kPlanShift = 24;
constexpr uint64_t kPlanMask = (uint64_t{1} << kPlanShift) - 1;

uint64_t unitKey(uint8_t unit, uint64_t block) noexcept { return uint64_t{unit} << kUnitShift | block; }
uint8_t unitOfKey(uint64_t key) noexcept { return static_cast<uint8_t>(key >> kUnitShift); }
uint64_t blockOfKey(uint64_t key) noexcept { return key & ((uint64_t{1} << kUnitShift) - 1); }

} // namespace

std::optional<engine::Victim> Cache::victimOf(const Walk &walk, const WalkUnit &unit) const {
  const uint64_t block = unit.candidate.id;
  const StateEviction outcome = states_.reclaimOutcome(block, walk.unwritten);
  if (!outcome.evicted && !outcome.pending)
    return std::nullopt;
  const bool inUse = unit.unit == Unit::UsedState;
  engine::Victim victim{.key = unitKey(static_cast<uint8_t>(unit.unit), block),
                        .stateBuffers = outcome.evicted ? 1u : 0u,
                        // One that waits for the state write in flight frees
                        // nothing yet, and one that needs a write lands later.
                        .later = outcome.pending || states_.reclaimWrites(block),
                        .certainReuse = inUse || certainReuse(block),
                        .cls = inUse ? ReclaimClass::InUse : ReclaimClass::Ordinary};
  return victim;
}

// The reclaims a need's units make one after another, weighed without
// making them: a device need's memory reclaims (takeReclaim), a page need's
// evictions (evictOne), over counts of the pool's extents and of the blocks
// and states the reclaims before took. Each order of the walk resumes where
// the last reclaim left it, since what a reclaim passed over stays for the
// next: a checkpoint or a state that waits for the write in flight, a leaf
// whose demotion the tier refused, one its disk subtree holds. What joins an
// order is kept apart: a parent whose last resident child left, as recent
// as that child, and a leaf whose state left RAM. A page scan takes a state
// only where it sits on the leaf that goes next, which then goes with the
// next reclaim. A state written as it goes takes the one write slot; one the
// tier takes no write of is dropped, and the KV only it needed goes as a
// dead leaf. Moved pages follow compaction to the extents it fills, fullest
// first.
// A demotion's page frees as its copy lands. One the tier and the quota have
// room for now is sure to start: its page counts as landing once the quota
// holds what its write takes, and with the pages landing before it may
// complete an extent (Cache::extentsCompleted), as a unit's own demotions
// do toward the extent it empties. What goes only once another transfer
// lands (a parent whose resident children's copies are on their way, a leaf
// whose demotion the tier refuses while transfers are in flight or whose
// copy ahead is in flight, a block no request holds whose restore is in
// flight) goes as a later step, after what goes now of its class, by the
// generation of landings it waits for: once the transfers in flight have
// landed (generation 1), then once those started then land (generation 2),
// and so on, as the claim's retries reach them. take() stops at a unit's
// first later step and starts the writes of the later demotions of that
// step's class whose pages free as they land (Cache::freesOnLanding), a
// child's before its parent's, so a run's pages free as those writes land;
// they take the tier's room now. What is in use goes only once the ordinary
// class's later steps have; the claim's retries take the rest as landings
// allow. A restored leaf's page counts toward the extent that leaves, whose
// pages the units after it no longer reach, but frees nothing a unit is
// weighed at. A write the quota holds no room for may give up redundant
// copies (freeDiskSpace), oldest first, as many as the write needs: from
// then on no copy of what RAM holds too is counted on, nor is a demotion
// sure. In a page need's walk of a windowed cache, a state that leaves the
// cache unpins its tail, whose trim into free Slim pages frees Full pages.
// The reclaims of an Immediate claim pass over what only a write or a
// demotion frees; a unit whose reclaims first reach such cache comes after
// the unit a claim that can wait gets from there, which lands later.
struct Cache::ReclaimPlan final {
  ReclaimPlan(const Cache &cache, const Walk &walk, uint32_t serial);
  // The next unit, none once the reclaims change nothing more.
  [[nodiscard]] std::optional<engine::Victim> next();
  [[nodiscard]] bool weighs(const Walk &other) const noexcept;
  // What take() does of a unit next() gave: the victims that go now, before
  // its first later step (all of them when it has none), and the blocks of
  // the later demotions of that step's class, ordinary or in use, whose
  // writes it starts, in the order the plan takes them: a child before its
  // parent.
  struct Takes final {
    std::optional<uint32_t> now;
    std::vector<uint64_t> writes;
    bool laterInUse = false;
  };
  // Those of the unit in the plan the latest peek weighed it in; after
  // another peek or a take, those of the first such unit of the cache as it
  // stands.
  [[nodiscard]] static Takes takesOf(const Cache &cache, const std::shared_ptr<ReclaimPlan> &plan, const Walk &walk,
                                     const engine::Victim &victim);

  const Cache &cache;
  const Walk walk;
  const uint32_t serial;
  uint32_t produced = 0;

private:
  // An order of the walk: the candidates of the cache's own order after the
  // last one visited (head, once looked up), and those that joined it, by
  // recency.
  struct Order final {
    std::optional<CacheEvictionCandidate> at;
    std::optional<CacheEvictionCandidate> head;
    bool looked = false;
    std::set<std::pair<uint64_t, uint64_t>> joined;
  };
  // The allocated extents of a form: the empty ones, those that hold pages
  // and may move (by used pages, then index), and the free pages of each
  // kind.
  struct Form final {
    uint32_t allocated = 0;
    uint32_t freePages = 0;
    uint32_t freeInUse = 0;
    std::set<uint32_t> empty;
    std::set<std::pair<uint32_t, uint32_t>> holders;
  };
  struct Step final {
    bool progress = false;
    uint64_t bytes = 0;
    // Pages of the walk's form a leaf freed, now or by a demotion sure to
    // start.
    uint32_t pages = 0;
    bool state = false;
    bool kv = false;
    // Stopped before the first uncopied cache of the unit (reachesUncopied).
    bool uncopied = false;
  };
  // What the reclaims of the unit being weighed released, freed and gave up
  // so far, and the pages demotions free as they land, by form, before it.
  struct Weighed final {
    std::array<uint32_t, 3> demotions{};
    std::array<uint32_t, 3> landing{};
    uint64_t bytes = 0;
    uint32_t pages = 0;
    uint32_t states = 0;
    bool changed = false;
  };
  enum class LeafGoes : uint8_t { Copy, Erase, Demotion, Pending, Stays };

  [[nodiscard]] std::optional<engine::Victim> weigh(Weighed weighed);
  [[nodiscard]] Step step();
  [[nodiscard]] Step evict();
  [[nodiscard]] Step evictPhase(bool inUse);
  [[nodiscard]] Step recycle();
  // The pool.
  [[nodiscard]] uint32_t extentOf(uint32_t page) const;
  void unlist(uint32_t index);
  void list(uint32_t index);
  void freePage(uint32_t page);
  void fixPage(uint32_t page);
  [[nodiscard]] uint64_t releaseEmpty();
  [[nodiscard]] bool compact();
  [[nodiscard]] std::pair<bool, uint64_t> releaseExtent();
  [[nodiscard]] std::array<uint32_t, 3> completed(const std::array<uint32_t, 3> &before, bool restored) const;
  [[nodiscard]] uint64_t laterBytes(const std::array<uint32_t, 3> &before) const;
  [[nodiscard]] std::array<bool, 3> reach() const;
  [[nodiscard]] bool inReach(uint64_t block) const;
  // The orders.
  [[nodiscard]] std::optional<CacheEvictionCandidate> cacheNext(int order,
                                                                std::optional<CacheEvictionCandidate> after) const;
  [[nodiscard]] std::optional<CacheEvictionCandidate> head(int order);
  void visit(int order);
  void join(uint64_t block);
  void joinLater(uint64_t block);
  void joinParent(uint64_t block);
  // The blocks.
  [[nodiscard]] uint64_t recencyOf(uint64_t block) const;
  [[nodiscard]] uint64_t erasedRecency(uint64_t block) const;
  [[nodiscard]] uint64_t lastToLeave(uint64_t block) const;
  [[nodiscard]] bool hasResidentChild(uint64_t block) const;
  [[nodiscard]] bool needed(uint64_t block) const;
  [[nodiscard]] bool stateBelow(uint64_t block) const;
  [[nodiscard]] bool leaf(uint64_t block, bool onceLanded = false) const;
  [[nodiscard]] bool used(uint64_t block) const;
  [[nodiscard]] uint32_t leafGeneration(uint64_t block) const;
  void joinSelf(uint64_t block);
  [[nodiscard]] bool ramState(uint64_t block) const;
  [[nodiscard]] bool keptPoint(uint64_t block, uint32_t generation) const;
  [[nodiscard]] uint64_t laterPoint() const;
  [[nodiscard]] bool keptState(uint64_t block) const;
  [[nodiscard]] bool keptLeaf(uint64_t block) const;
  [[nodiscard]] bool writes(uint64_t block) const;
  [[nodiscard]] bool transfers() const;
  [[nodiscard]] ReclaimTiming timing() const noexcept;
  [[nodiscard]] bool skipsWrites() const noexcept;
  [[nodiscard]] bool reachesUncopied(uint64_t block, bool state) const;
  [[nodiscard]] StateEviction stateGoes(uint64_t block) const;
  [[nodiscard]] LeafGoes leafGoes(uint64_t block, ReclaimTiming timing, bool landed = false) const;
  [[nodiscard]] LeafGoes leafGoesUncopied(uint64_t block, ReclaimTiming timing, bool landed) const;
  [[nodiscard]] bool slotFits(uint64_t bytes, bool inUse) const;
  void stateOut(uint64_t block, bool inUse);
  void demotionLands(uint64_t block);
  [[nodiscard]] uint32_t takeState(uint64_t block, bool inUse);
  [[nodiscard]] uint32_t trimTail(uint64_t block);
  [[nodiscard]] bool pinnedBesides(uint64_t block) const;
  [[nodiscard]] Step takeLeaf(uint64_t block, LeafGoes goes, bool inUse, bool later = false);
  [[nodiscard]] Step takeLater(bool inUse);
  void took();
  void tookLater(uint64_t block, LeafGoes goes, bool inUse);
  void mark(uint64_t block, bool state, bool inUse);

  // The orders: checkpoints, dead leaves, ordinary states and leaves, those
  // in use; and a state alone's (recycle) checkpoints, ordinary and in use.
  enum : int { Checkpoints, DeadLeaves, States, Leaves, UsedStates, UsedLeaves, Recycled, RecycledStates,
               RecycledUsed, Orders };
  std::array<Order, Orders> orders_;
  bool checkpointsDone_ = false;
  std::array<bool, 2> statesHeld_{};
  // The leaves of each class, ordinary and in use, that go once transfers
  // land: by the generation of landings each waits for, then by recency.
  struct Later final {
    uint32_t generation = 0;
    uint64_t recency = 0;
    uint64_t block = 0;
    friend auto operator<=>(const Later &, const Later &) = default;
  };
  std::array<std::set<Later>, 2> later_;
  // The generation the walk has reached, 0 for what goes now; and when each
  // block whose page frees as a demotion lands, or that a later step took,
  // leaves RAM (one that went now left at 0, before them): the generation
  // from which it has left, at the landings of that generation or after
  // them, as a later step, and in the order the cache does each, landing
  // writes in the order they started (pollTierTransfers) and taking one
  // leaf after another.
  struct Gone final {
    uint32_t generation = 0;
    bool taken = false;
    uint32_t order = 0;
    friend auto operator<=>(const Gone &, const Gone &) = default;
  };
  uint32_t generation_ = 0;
  std::unordered_map<uint64_t, Gone> gone_;
  uint32_t departures_ = 0;

  uint32_t extentPages_;
  std::vector<KvPool::ExtentCounts> extents_;
  std::array<Form, 3> forms_;
  std::unordered_map<uint32_t, uint32_t> movedTo_;
  // By form: the pages the walk may reuse (Cache::reusablePages) less those
  // of the extents that leave, and the pages demotions hold, those in
  // flight, then those the reclaims start.
  std::array<uint32_t, 3> reusable_{};
  std::array<uint32_t, 3> demotions_;
  // The pages demotions free as they land, by form: those in flight, then
  // those sure to start, then those of later steps; and those restored
  // leaves free, which no unit is weighed at.
  std::array<uint32_t, 3> landing_;
  std::array<uint32_t, 3> restored_{};
  std::optional<uint32_t> tierRoom_;
  // What the writes the reclaims start take of the quota.
  uint64_t quotaBytes_ = 0;
  bool copiesMayGo_ = false;
  // How far slotFits followed the quota's replacement order, and the bytes
  // the copies it passed free: a lookahead that changes no weighing.
  mutable DiskCursor diskCursor_;
  mutable uint64_t replacedBytes_ = 0;
  bool writing_;
  std::unordered_set<uint32_t> freedPages_;
  // The pages trims leave Slim: those a page need's walk gives first, then
  // those of the tails states unpin, into the free Slim pages left.
  std::unordered_set<uint32_t> trimmed_;
  uint32_t slimRoom_ = 0;
  std::unordered_set<uint64_t> leftRam_, demoting_, diskParents_, statesOut_, statesDropped_, copyingAhead_,
      restoring_;
  // The demotions in flight whose pages free once what holds them left
  // (demotionLands), by the order they land in (Gone).
  std::unordered_map<uint64_t, uint32_t> heldDemotions_;
  // The states dropPoisoned erases once the transfers in flight land, for a
  // walk that keeps the resume point.
  std::unordered_set<uint64_t> dropped_;
  std::unordered_map<uint64_t, uint64_t> recency_;
  // The unit being weighed: whether it holds a victim of certain reuse or
  // in use, and a state written as it goes; where it started; the victims
  // it took that go now, and what take() does of it.
  bool certain_ = false;
  bool inUse_ = false;
  bool written_ = false;
  bool ended_ = false;
  std::array<uint32_t, 3> unitDemotions_{};
  std::array<uint32_t, 3> unitLanding_{};
  std::array<uint32_t, 3> unitRestored_{};
  uint32_t nowTaken_ = 0;
  Takes unitTakes_;
  std::vector<Takes> takes_;
  // A claim that can wait's plan, weighed from where an Immediate one first
  // reached uncopied cache in a unit; the unit that goes on past it, and
  // comes next; and a step that goes on where it stopped there.
  bool waits_ = false;
  bool passing_ = false;
  std::optional<engine::Victim> passed_;
  Takes passedTakes_;
  bool resumes_ = false;
};

Cache::ReclaimPlan::ReclaimPlan(const Cache &owner, const Walk &plan, uint32_t id)
    : cache(owner), walk(plan), serial(id), checkpointsDone_(plan.scope.pagesOnly),
      extentPages_(owner.pool_.extentPages()), demotions_(owner.demotionPages()), landing_(owner.demotedPages()),
      tierRoom_(owner.tier_ ? owner.tier_->demotionRoom(WriteClass::Reclaim) : std::optional<uint32_t>(0u)),
      writing_(owner.states_.writing()) {
  for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index})
    reusable_[static_cast<size_t>(form)] = owner.reusablePages(plan.scope.upTo, form);
  std::vector<uint32_t> inTransfer;
  for (const Write &write : cache.writes_)
    inTransfer.push_back(cache.kv_.page(write.block));
  for (const auto &[block, _] : cache.restores_)
    inTransfer.push_back(cache.kv_.page(block));
  extents_ = cache.pool_.extentCounts(inTransfer);
  for (uint32_t index = 0; index < extents_.size(); ++index)
    list(index);
  for (const uint32_t page : cache.trimmedBy(walk))
    trimmed_.insert(page);
  if (cache.trims(walk)) {
    const uint32_t room = cache.trimRoom();
    slimRoom_ = room - std::min<uint32_t>(room, static_cast<uint32_t>(trimmed_.size()));
  }
  // What goes once the transfers in flight land (generation 1): the parent
  // of a page a demotion frees (freesOnLanding), a leaf whose copy ahead is
  // in flight, and a leaf no request holds whose restore is in flight. A
  // demotion whose page a child or a state in RAM holds frees it once they
  // leave (demotionLands). The writes land in the order they started.
  const std::unordered_set<uint64_t> freeing = cache.freeingDemotions();
  for (const Write &write : cache.writes_) {
    if (freeing.contains(write.block)) {
      demoting_.insert(write.block);
      gone_[write.block] = {.generation = 1, .order = departures_};
    } else if (write.write == WriteClass::Persist) {
      copyingAhead_.insert(write.block);
    } else if (write.write == WriteClass::Reclaim) {
      heldDemotions_.emplace(write.block, departures_);
    }
    ++departures_;
  }
  for (const auto &[block, _] : cache.restores_)
    restoring_.insert(block);
  for (const Write &write : cache.writes_) {
    if (freeing.contains(write.block))
      joinParent(write.block);
    else if (write.write == WriteClass::Persist && leaf(write.block, true))
      joinLater(write.block);
  }
  for (const auto &[block, _] : cache.restores_)
    if (leaf(block, true))
      joinLater(block);
  // Once the transfers in flight land, dropPoisoned erases the subtrees below
  // poisoned blocks that no request or lease holds or uses, with their
  // states, among which may be the resume point the walk keeps (keptPoint).
  if (walk.keepResumePoint) {
    const KvCache &kv = cache.kv_;
    for (const uint64_t poisoned : cache.poisoned_) {
      if (!kv.contains(poisoned))
        continue;
      std::vector<uint64_t> below = kv.children(poisoned), states;
      bool held = false;
      for (size_t index = 0; index < below.size() && !held; ++index) {
        const uint64_t block = below[index];
        held = kv.held(block) || cache.states_.pinned(block) ||
               (kv.page(block) != KvCache::noPage && cache.pool_.pageActive(kv.page(block)));
        if (cache.states_.contains(block))
          states.push_back(block);
        const std::vector<uint64_t> children = kv.children(block);
        below.insert(below.end(), children.begin(), children.end());
      }
      if (!held)
        dropped_.insert(states.begin(), states.end());
    }
  }
}

bool Cache::ReclaimPlan::weighs(const Walk &other) const noexcept { return other == walk; }

Cache::ReclaimPlan::Takes Cache::ReclaimPlan::takesOf(const Cache &cache, const std::shared_ptr<ReclaimPlan> &plan,
                                                      const Walk &walk, const engine::Victim &victim) {
  const uint64_t key = blockOfKey(victim.key);
  const auto unit = static_cast<uint32_t>(key & kPlanMask);
  if (plan && plan->serial == static_cast<uint32_t>(key >> kPlanShift) && unit < plan->takes_.size() &&
      plan->weighs(walk))
    return plan->takes_[unit];
  // A unit that goes at once comes past those that land later, which an
  // Immediate claim passes over.
  ReclaimPlan fresh(cache, walk, 0);
  for (std::optional<engine::Victim> next; (next = fresh.next());)
    if (victim.later || !next->later)
      return fresh.takes_.back();
  return {};
}

std::optional<engine::Victim> Cache::ReclaimPlan::next() {
  std::optional<engine::Victim> unit = std::exchange(passed_, std::nullopt);
  if (unit) {
    unitTakes_ = passedTakes_;
  } else {
    if (ended_)
      return std::nullopt;
    certain_ = inUse_ = written_ = false;
    unitDemotions_ = demotions_;
    unitLanding_ = landing_;
    unitRestored_ = restored_;
    nowTaken_ = 0;
    unitTakes_ = {};
    unit = weigh({.demotions = demotions_, .landing = landing_});
  }
  if (unit) {
    unit->key = unitKey(static_cast<uint8_t>(Unit::Reclaim), uint64_t{serial} << kPlanShift | produced++);
    takes_.push_back(unitTakes_);
  }
  return unit;
}

// The reclaims of the unit from what it weighed so far, up to the one that
// frees what the need lacks: for a device need, which counts states, one
// step (an extent released now or completed by the pages its demotions free
// as they land, or a state given up), for a page need a page of its form.
// The reclaims that give up states and the leaves of other forms on the way
// are the unit's too, unless the need counts states.
std::optional<engine::Victim> Cache::ReclaimPlan::weigh(Weighed weighed) {
  for (;;) {
    const Step taken = walk.device ? step() : evict();
    weighed.bytes += taken.bytes;
    weighed.pages += taken.pages;
    weighed.states += taken.state;
    weighed.changed = weighed.changed || taken.progress || taken.kv;
    if (taken.uncopied) {
      // The unit a claim that can wait gets from here comes first: it lands
      // later, so free() passes over it for an Immediate claim, which then
      // reached cache a copy ahead would have let go at once. This unit goes
      // on past that cache.
      ReclaimPlan waits = *this;
      waits.waits_ = true;
      const std::optional<engine::Victim> later = waits.weigh(weighed);
      passing_ = true;
      std::optional<engine::Victim> unit = weigh(weighed);
      passing_ = false;
      if (!later || !later->later)
        return unit;
      passed_ = std::move(unit);
      passedTakes_ = std::exchange(unitTakes_, waits.unitTakes_);
      return later;
    }
    if (!taken.progress) {
      ended_ = true;
      break;
    }
    if (taken.pages || walk.statesAreUnits)
      break;
  }
  if (!weighed.changed)
    return std::nullopt;
  if (walk.device) {
    // The extents its landings and its restored leaves complete leave as
    // they land: the units after it reach no further than the pages of each
    // form left.
    const std::array<uint32_t, 3> extents = completed(weighed.landing, true);
    for (size_t form = 0; form < extents.size(); ++form)
      reusable_[form] -= std::min(reusable_[form], extents[form] * extentPages_);
  }
  // Bytes that land free later, also those of demotions in flight that
  // its victims let free their pages (demotionLands).
  const uint64_t later = walk.device ? laterBytes(weighed.landing) : 0;
  return engine::Victim{.bytes = weighed.bytes + later,
                        .pages = weighed.pages,
                        .form = walk.form,
                        .stateBuffers = weighed.states,
                        .later = later || demotions_ != weighed.demotions || written_ || unitTakes_.now.has_value(),
                        .certainReuse = certain_,
                        .cls = inUse_ ? ReclaimClass::InUse : ReclaimClass::Ordinary};
}

// takeReclaim, weighed; one that stopped at uncopied cache goes on there.
// Its victims go while KV of some form can empty an extent (reach), its
// leaves that lead to such an extent alone (inReach). The pages the unit's
// own demotions free as they land count toward the extent its victims
// empty, and it ends once they, or with them the pages of its restored
// leaves, complete one.
Cache::ReclaimPlan::Step Cache::ReclaimPlan::step() {
  if (!std::exchange(resumes_, false)) {
    for (std::pair<bool, uint64_t> released; (released = releaseExtent()).first;)
      if (released.second)
        return {.progress = true, .bytes = released.second};
  }
  bool kv = false;
  while (std::ranges::any_of(reach(), std::identity{})) {
    const Step evicted = evict();
    if (evicted.uncopied) {
      resumes_ = true;
      return {.kv = kv, .uncopied = true};
    }
    if (!evicted.progress || evicted.state)
      return {.progress = evicted.progress, .state = evicted.state, .kv = kv};
    kv = true;
    if (const uint64_t bytes = releaseExtent().second)
      return {.progress = true, .bytes = bytes, .kv = true};
    if (std::ranges::any_of(completed(unitLanding_, true), [](uint32_t extents) { return extents != 0; }))
      return {.progress = true, .kv = true};
  }
  Step recycled = recycle();
  resumes_ = recycled.uncopied;
  recycled.kv = kv;
  return recycled;
}

// evictOne, weighed; a page scan takes no checkpoint (checkpointsDone_).
Cache::ReclaimPlan::Step Cache::ReclaimPlan::evict() {
  while (!checkpointsDone_) {
    const std::optional<CacheEvictionCandidate> checkpoint = head(Checkpoints);
    if (!checkpoint) {
      checkpointsDone_ = true;
      break;
    }
    if (reachesUncopied(checkpoint->id, true))
      return {.uncopied = true};
    visit(Checkpoints);
    if (keptState(checkpoint->id))
      continue;
    // As evictPhase's states: one that waits for its write goes as a unit
    // that lands later and the newer checkpoints follow it; one that waits
    // and is not taken holds them back.
    const StateEviction goes = stateGoes(checkpoint->id);
    const bool later = goes.pending && writes(checkpoint->id) && !keptPoint(checkpoint->id, generation_ + 1);
    if (goes.evicted || later) {
      if (goes.evicted)
        took();
      return {.progress = true, .pages = takeState(checkpoint->id, false), .state = true};
    }
    checkpointsDone_ = goes.pending;
  }
  for (std::optional<CacheEvictionCandidate> dead; (dead = head(DeadLeaves));) {
    visit(DeadLeaves);
    if (!inReach(dead->id))
      continue;
    if (const LeafGoes goes = leafGoes(dead->id, timing()); goes == LeafGoes::Copy || goes == LeafGoes::Erase)
      return takeLeaf(dead->id, goes, false);
  }
  if (const Step taken = evictPhase(false); taken.progress || taken.uncopied)
    return taken;
  if (walk.scope.upTo == ReclaimClass::InUse) {
    if (const Step taken = evictPhase(true); taken.progress || taken.uncopied)
      return taken;
  }
  return {};
}

// takeOldest over the ordinary states and leaves, or those in use; in a page
// scan over the leaves alone, the state in RAM on one going first. A leaf
// whose demotion the tier refuses while transfers are in flight goes once
// one lands, with the class's other later steps after what goes now, in the
// next generation.
Cache::ReclaimPlan::Step Cache::ReclaimPlan::evictPhase(bool inUse) {
  const int states = inUse ? UsedStates : States, leaves = inUse ? UsedLeaves : Leaves;
  bool &held = statesHeld_[inUse];
  for (;;) {
    const std::optional<CacheEvictionCandidate> state =
        held || walk.scope.pagesOnly ? std::nullopt : head(states);
    const std::optional<CacheEvictionCandidate> leaf = head(leaves);
    if (!state && !leaf)
      return timing() == ReclaimTiming::CanWait ? takeLater(inUse) : Step{};
    if (state && (!leaf || state->lastUsed <= leaf->lastUsed)) {
      if (reachesUncopied(state->id, true))
        return {.uncopied = true};
      visit(states);
      if (keptState(state->id))
        continue;
      // One that needs a write and waits for the write slot frees its memory
      // as a unit that lands later, once that write and its own have landed,
      // unless it is the resume point by then; the newer states of its phase
      // follow it, each a unit of its own. One that waits and is not taken
      // holds them back.
      const StateEviction goes = stateGoes(state->id);
      const bool later = goes.pending && writes(state->id) && !keptPoint(state->id, generation_ + 1);
      held = goes.pending && !later;
      if (goes.evicted || later) {
        if (goes.evicted)
          took();
        return {.progress = true, .pages = takeState(state->id, inUse), .state = true};
      }
      continue;
    }
    if (ramState(leaf->id)) {
      // A page scan's state on the leaf (LeafState), kept with the leaf's
      // chain or as the resume point; its leaf joins the leaves once it went.
      if (keptLeaf(leaf->id) || keptPoint(leaf->id, generation_)) {
        visit(leaves);
        continue;
      }
      if (reachesUncopied(leaf->id, true))
        return {.uncopied = true};
      visit(leaves);
      if (!stateGoes(leaf->id).evicted)
        continue;
      took();
      return {.progress = true, .pages = takeState(leaf->id, inUse), .state = true};
    }
    if (!inReach(leaf->id)) {
      visit(leaves);
      continue;
    }
    if (reachesUncopied(leaf->id, false))
      return {.uncopied = true};
    visit(leaves);
    if (keptLeaf(leaf->id))
      continue;
    switch (const LeafGoes goes = leafGoes(leaf->id, timing())) {
    case LeafGoes::Copy:
    case LeafGoes::Erase:
    case LeafGoes::Demotion:
      return takeLeaf(leaf->id, goes, inUse);
    case LeafGoes::Pending:
      later_[inUse].insert({generation_ + 1, recencyOf(leaf->id), leaf->id});
      break;
    case LeafGoes::Stays:
      break;
    }
  }
}

// The leaf of the class that goes once transfers land, of the earliest
// generation, the oldest, and with it in a page scan the state in RAM on
// it: its own copy ahead or restore lands, its children's copies land, or
// the tier, with a transfer landed, takes its demotion. The walk reaches
// that generation.
Cache::ReclaimPlan::Step Cache::ReclaimPlan::takeLater(bool inUse) {
  for (auto &queue = later_[inUse]; !queue.empty();) {
    const Later next = *queue.begin();
    queue.erase(queue.begin());
    const uint64_t block = next.block;
    if (!leaf(block) && !leaf(block, true))
      continue;
    if (keptLeaf(block) || (ramState(block) && keptPoint(block, std::max(generation_, next.generation))) ||
        !inReach(block))
      continue;
    const LeafGoes goes = leafGoes(block, ReclaimTiming::CanWait, true);
    if (goes == LeafGoes::Stays || goes == LeafGoes::Pending || (ramState(block) && !stateGoes(block).evicted))
      continue;
    generation_ = std::max(generation_, next.generation);
    tookLater(block, goes, inUse);
    if (ramState(block))
      stateOut(block, inUse);
    return takeLeaf(block, goes, inUse, true);
  }
  return {};
}

// recycleState, weighed: the first state of states alone.
Cache::ReclaimPlan::Step Cache::ReclaimPlan::recycle() {
  for (const int order : {Recycled, RecycledStates, RecycledUsed}) {
    if (order == RecycledUsed && walk.scope.upTo == ReclaimClass::Ordinary)
      break;
    for (std::optional<CacheEvictionCandidate> state; (state = head(order));) {
      if (reachesUncopied(state->id, true))
        return {.uncopied = true};
      if (keptState(state->id) || (skipsWrites() && cache.states_.reclaimWrites(state->id))) {
        visit(order);
        continue;
      }
      // One that waits for its write goes as a unit that lands later, as in
      // evictPhase; one that waits and is not taken ends the recycling.
      const StateEviction goes = stateGoes(state->id);
      if (!goes.evicted && !(goes.pending && writes(state->id) && !keptPoint(state->id, generation_ + 1)))
        return {};
      visit(order);
      if (goes.evicted)
        took();
      return {.progress = true, .pages = takeState(state->id, order == RecycledUsed), .state = true};
    }
  }
  return {};
}

uint32_t Cache::ReclaimPlan::extentOf(uint32_t page) const {
  const auto moved = movedTo_.find(page);
  return moved != movedTo_.end() ? moved->second : page / extentPages_;
}

void Cache::ReclaimPlan::unlist(uint32_t index) {
  const KvPool::ExtentCounts &extent = extents_[index];
  if (!extent.allocated)
    return;
  Form &form = forms_[static_cast<size_t>(extent.form)];
  --form.allocated;
  form.freePages -= extent.freePages;
  if (!extent.usedPages) {
    form.empty.erase(index);
    return;
  }
  form.freeInUse -= extent.freePages;
  if (!extent.fixed)
    form.holders.erase({extent.usedPages, index});
}

void Cache::ReclaimPlan::list(uint32_t index) {
  const KvPool::ExtentCounts &extent = extents_[index];
  if (!extent.allocated)
    return;
  Form &form = forms_[static_cast<size_t>(extent.form)];
  ++form.allocated;
  form.freePages += extent.freePages;
  if (!extent.usedPages) {
    form.empty.insert(index);
    return;
  }
  form.freeInUse += extent.freePages;
  if (!extent.fixed)
    form.holders.insert({extent.usedPages, index});
}

void Cache::ReclaimPlan::freePage(uint32_t page) {
  const uint32_t index = extentOf(page);
  unlist(index);
  --extents_[index].usedPages;
  ++extents_[index].freePages;
  list(index);
  freedPages_.insert(page);
}

void Cache::ReclaimPlan::fixPage(uint32_t page) {
  const uint32_t index = extentOf(page);
  unlist(index);
  extents_[index].fixed = true;
  list(index);
}

// KvPool::reclaimEmptyExtents(keepRunway, 1).
uint64_t Cache::ReclaimPlan::releaseEmpty() {
  for (const kv::PageForm each : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index}) {
    Form &form = forms_[static_cast<size_t>(each)];
    if (form.empty.size() <= (walk.keeps(each) ? 1u : 0u) || !walk.releases(each, form.freePages, extentPages_))
      continue;
    const uint32_t index = *form.empty.begin();
    unlist(index);
    extents_[index].allocated = false;
    reusable_[static_cast<size_t>(each)] -= extentPages_;
    return uint64_t{extentPages_} * cache.pool_.bytesPerPage(each);
  }
  return 0;
}

// KvPool::compactExtent: the extent of the fewest pages, once the free pages
// of the others that hold pages cover it, its pages, lowest first, to the
// free pages of those, fullest first.
bool Cache::ReclaimPlan::compact() {
  for (const kv::PageForm each : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index}) {
    const Form &form = forms_[static_cast<size_t>(each)];
    if (!form.allocated || form.holders.empty() || form.freeInUse < extentPages_ ||
        !walk.releases(each, form.freePages, extentPages_))
      continue;
    const uint32_t emptied = form.holders.begin()->second;
    std::vector<uint32_t> receivers;
    for (uint32_t index = 0; index < extents_.size(); ++index) {
      const KvPool::ExtentCounts &extent = extents_[index];
      if (index != emptied && extent.allocated && extent.form == each && extent.usedPages && extent.freePages)
        receivers.push_back(index);
    }
    std::stable_sort(receivers.begin(), receivers.end(), [&](uint32_t left, uint32_t right) {
      return extents_[left].usedPages > extents_[right].usedPages;
    });
    // Its own pages, then those an earlier compaction moved in.
    std::vector<uint32_t> moving;
    const uint32_t first = emptied * extentPages_;
    for (uint32_t page = first; page < first + extentPages_; ++page)
      if ((cache.kv_.blockOn(page) || cache.pool_.pageActive(page)) && !freedPages_.contains(page) &&
          !movedTo_.contains(page))
        moving.push_back(page);
    std::vector<uint32_t> movedIn;
    for (const auto &[page, extent] : movedTo_)
      if (extent == emptied && !freedPages_.contains(page))
        movedIn.push_back(page);
    std::ranges::sort(movedIn);
    moving.insert(moving.end(), movedIn.begin(), movedIn.end());
    std::vector<uint32_t> taken(extents_.size());
    auto receiver = receivers.begin();
    for (const uint32_t page : moving) {
      while (receiver != receivers.end() && taken[*receiver] == extents_[*receiver].freePages)
        ++receiver;
      if (receiver == receivers.end())
        break;
      movedTo_[page] = *receiver;
      ++taken[*receiver];
    }
    for (const uint32_t index : receivers) {
      unlist(index);
      extents_[index].usedPages += taken[index];
      extents_[index].freePages -= taken[index];
      list(index);
    }
    unlist(emptied);
    extents_[emptied].freePages = extentPages_;
    extents_[emptied].usedPages = 0;
    list(emptied);
    return true;
  }
  return false;
}

// Cache::releaseExtent.
std::pair<bool, uint64_t> Cache::ReclaimPlan::releaseExtent() {
  if (const uint64_t bytes = releaseEmpty())
    return {true, bytes};
  if (compact())
    return {true, releaseEmpty()};
  return {false, 0};
}

// The extents of each form the pages of the demotions sure to start since
// `before` complete as they land, beside the free pages and those landing
// before, past the runway the walk keeps (Cache::extentsCompleted); with
// `restored`, beside the pages of the restored leaves too, those since the
// unit began among them.
std::array<uint32_t, 3> Cache::ReclaimPlan::completed(const std::array<uint32_t, 3> &before, bool restored) const {
  std::array<uint32_t, 3> extents{};
  for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index}) {
    const auto index = static_cast<size_t>(form);
    const uint32_t from = before[index] + (restored ? unitRestored_[index] : 0);
    const uint32_t to = landing_[index] + (restored ? restored_[index] : 0);
    extents[index] = completedExtents(forms_[index].freePages + from, to - from, extentPages_, walk.keeps(form));
  }
  return extents;
}

// The bytes of the extents the pages of the demotions sure to start since
// `before` complete as they land; restored leaves free none.
uint64_t Cache::ReclaimPlan::laterBytes(const std::array<uint32_t, 3> &before) const {
  const std::array<uint32_t, 3> extents = completed(before, false);
  uint64_t bytes = 0;
  for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index})
    bytes += uint64_t{extents[static_cast<size_t>(form)]} * extentPages_ * cache.pool_.bytesPerPage(form);
  return bytes;
}

// Cache::formsInReach for the unit: of each form, the pages left to reuse,
// less those the demotions before the unit hold.
std::array<bool, 3> Cache::ReclaimPlan::reach() const {
  std::array<bool, 3> forms{};
  for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index}) {
    const auto index = static_cast<size_t>(form);
    forms[index] = walk.reaches(form, reusable_[index], unitDemotions_[index], extentPages_);
  }
  return forms;
}

// A device need's leaf goes only where it leads to an extent in reach
// (Cache::leadsToReach); one that leads to none frees no memory, and stays
// (Cache::takeOldest).
bool Cache::ReclaimPlan::inReach(uint64_t block) const {
  return !walk.device || cache.leadsToReach(block, reach());
}

std::optional<CacheEvictionCandidate> Cache::ReclaimPlan::cacheNext(int order,
                                                                   std::optional<CacheEvictionCandidate> after) const {
  const StateCache &states = cache.states_;
  switch (order) {
  case Checkpoints:
  case Recycled:
    return states.checkpointCandidate(false, {}, after);
  case States:
  case RecycledStates:
    return states.ordinaryCandidate(false, {}, after);
  case UsedStates:
  case RecycledUsed:
    return states.inUseCandidate(after);
  case DeadLeaves:
    return cache.oldestDeadKvLeaf(after ? after->id : 0);
  case Leaves:
  case UsedLeaves:
    return cache.oldestKvLeaf(after ? after->id : 0, order == UsedLeaves, walk.scope.pagesOnly);
  default:
    return std::nullopt;
  }
}

std::optional<CacheEvictionCandidate> Cache::ReclaimPlan::head(int index) {
  Order &order = orders_[index];
  const bool states = index != DeadLeaves && index != Leaves && index != UsedLeaves;
  if (!order.looked) {
    order.head = cacheNext(index, order.at);
    // What a reclaim before took has left the order.
    while (order.head && (states ? statesOut_.contains(order.head->id)
                                 : leftRam_.contains(order.head->id) || demoting_.contains(order.head->id)))
      order.head = cacheNext(index, order.head);
    order.looked = true;
  }
  const std::optional<CacheEvictionCandidate> joined =
      order.joined.empty() ? std::nullopt
                           : std::optional(CacheEvictionCandidate{order.joined.begin()->second,
                                                                  order.joined.begin()->first});
  if (joined && (!order.head || std::pair{joined->lastUsed, joined->id} < std::pair{order.head->lastUsed, order.head->id}))
    return joined;
  return order.head;
}

void Cache::ReclaimPlan::visit(int index) {
  Order &order = orders_[index];
  const std::optional<CacheEvictionCandidate> joined =
      order.joined.empty() ? std::nullopt
                           : std::optional(CacheEvictionCandidate{order.joined.begin()->second,
                                                                  order.joined.begin()->first});
  if (joined && (!order.head || std::pair{joined->lastUsed, joined->id} < std::pair{order.head->lastUsed, order.head->id})) {
    order.joined.erase(order.joined.begin());
    return;
  }
  order.at = order.head;
  order.looked = false;
}

void Cache::ReclaimPlan::join(uint64_t block) {
  // A state in RAM keeps its leaf out of the leaf orders until it goes,
  // except in a page scan, which takes it with its leaf.
  if (!walk.scope.pagesOnly && ramState(block))
    return;
  const bool dead = !needed(block);
  const bool inUse = !dead && cache.kvNeededByStateInUse(block);
  if (inUse && walk.scope.upTo == ReclaimClass::Ordinary)
    return;
  orders_[dead ? DeadLeaves : inUse ? UsedLeaves : Leaves].joined.insert({recencyOf(block), block});
}

// A leaf that goes once a transfer lands joins its class's later steps, a
// dead one the ordinary class's, by the rule join() keeps, at the
// generation it is a leaf from.
void Cache::ReclaimPlan::joinLater(uint64_t block) {
  if (!walk.scope.pagesOnly && ramState(block))
    return;
  const bool inUse = needed(block) && cache.kvNeededByStateInUse(block);
  if (inUse && walk.scope.upTo == ReclaimClass::Ordinary)
    return;
  later_[inUse].insert({leafGeneration(block), recencyOf(block), block});
}

// A block whose state left RAM: a leaf now, or once what is on its way
// lands.
void Cache::ReclaimPlan::joinSelf(uint64_t block) {
  if (leaf(block))
    join(block);
  else if (leaf(block, true))
    joinLater(block);
}

// The parent of a block that leaves RAM, now or as its copy lands, is a
// leaf once its last resident child left, as recent as that child
// (KvCache::inherit): the block, or a sibling that leaves after it
// (lastToLeave).
void Cache::ReclaimPlan::joinParent(uint64_t block) {
  const uint64_t parent = cache.kv_.key(block).parent;
  if (!parent)
    return;
  const bool now = leaf(parent);
  if (!now && !leaf(parent, true))
    return;
  recency_[parent] = std::max(recencyOf(parent), recencyOf(lastToLeave(block)));
  now ? join(parent) : joinLater(parent);
}

uint64_t Cache::ReclaimPlan::recencyOf(uint64_t block) const {
  const auto inherited = recency_.find(block);
  return inherited != recency_.end() ? inherited->second : cache.kv_.lastUsed(block);
}

// The recency an Erase leaves the leaf as it drops the leaf's disk-only
// subtree (dropDiskSubtree): in KvCache::subtree's order each block erased
// passes its recency to its parent (KvCache::inherit), the leaf at every
// child and a disk-only parent at its last. A block the reclaims before
// erased has left with its subtree; one they dropped through its copy is
// disk-only.
uint64_t Cache::ReclaimPlan::erasedRecency(uint64_t block) const {
  const KvCache &kv = cache.kv_;
  const auto erased = [&](uint64_t below) {
    for (; below != block; below = kv.key(below).parent)
      if (leftRam_.contains(below) && !kv.slot(below))
        return true;
    return false;
  };
  uint64_t recency = recencyOf(block);
  // What the last child erased so far gives each disk-only parent, which
  // comes after its children.
  std::unordered_map<uint64_t, uint64_t> passed;
  for (const uint64_t below : kv.subtree(block)) {
    if (erased(below))
      continue;
    const auto child = passed.find(below);
    const uint64_t own = child == passed.end() ? recencyOf(below) : std::max(recencyOf(below), child->second);
    const uint64_t parent = kv.key(below).parent;
    if (parent == block)
      recency = std::max(recency, own);
    else
      passed[parent] = own;
  }
  return recency;
}

// Of a block that leaves RAM and its siblings whose pages free as their
// demotions land or later steps take them, the one that leaves last (Gone),
// a block that goes now before any of them.
uint64_t Cache::ReclaimPlan::lastToLeave(uint64_t block) const {
  uint64_t last = block;
  for (const uint64_t sibling : cache.kv_.children(cache.kv_.key(block).parent))
    if (demoting_.contains(sibling) && (!demoting_.contains(last) || gone_.at(last) < gone_.at(sibling)))
      last = sibling;
  return last;
}

// A child of the block holds a page the reclaims before left (KvCache's
// resident children).
bool Cache::ReclaimPlan::hasResidentChild(uint64_t block) const {
  const KvCache &kv = cache.kv_;
  return std::ranges::any_of(kv.children(block), [&](uint64_t child) {
    return kv.page(child) != KvCache::noPage && !leftRam_.contains(child);
  });
}

bool Cache::ReclaimPlan::needed(uint64_t block) const {
  // Only a state restores a disk-only chain, through its own block and
  // every block above: one the reclaims dropped restores nothing.
  return (cache.states_.contains(block) && !statesDropped_.contains(block)) || stateBelow(block);
}

bool Cache::ReclaimPlan::stateBelow(uint64_t block) const {
  if (!cache.kv_.stateBelow(block))
    return false;
  if (statesDropped_.empty())
    return true;
  // A block below in transfer or active hides the subtree, which then keeps
  // a state below, as the cache's count says.
  const std::vector<uint64_t> below = cache.kv_.subtree(block);
  return below.empty() || std::ranges::any_of(below, [&](uint64_t each) {
           return cache.states_.contains(each) && !statesDropped_.contains(each);
         });
}

// KvCache::residentLeaf once the reclaims before took the children they did;
// onceLanded, once the transfers on their way land too: its own copy ahead
// or restore, its children's demotions.
bool Cache::ReclaimPlan::leaf(uint64_t block, bool onceLanded) const {
  const KvCache &kv = cache.kv_;
  if (!kv.contains(block) || leftRam_.contains(block) || demoting_.contains(block))
    return false;
  if (kv.page(block) == KvCache::noPage || used(block) ||
      (kv.transfer(block) != KvCache::Transfer::None &&
       !(onceLanded && (copyingAhead_.contains(block) || restoring_.contains(block)))))
    return false;
  return std::ranges::none_of(kv.children(block), [&](uint64_t child) {
    return kv.page(child) != KvCache::noPage && !leftRam_.contains(child) &&
           !(onceLanded && demoting_.contains(child));
  });
}

// A request or a lease uses, holds or reads the resident block, or a failed
// read poisoned it: it stays no leaf (KvCache::residentLeaf), and its page
// frees as no demotion lands (Cache::freesOnLanding).
bool Cache::ReclaimPlan::used(uint64_t block) const {
  const KvCache &kv = cache.kv_;
  return !kv.matchable(block) || kv.held(block) || cache.pool_.pageActive(kv.page(block)) ||
         cache.states_.pinned(block) || std::ranges::any_of(cache.requests_, [&](const auto &entry) {
           return !entry.second.cachedBlocks.empty() && entry.second.cachedBlocks.back() == block;
         });
}

// The generation from which a leaf once landed is one: its own copy ahead
// or restore has landed (generation 1), and each resident child has left.
uint32_t Cache::ReclaimPlan::leafGeneration(uint64_t block) const {
  uint32_t generation = copyingAhead_.contains(block) || restoring_.contains(block) ? 1 : 0;
  for (const uint64_t child : cache.kv_.children(block))
    if (const auto gone = gone_.find(child); gone != gone_.end())
      generation = std::max(generation, gone->second.generation);
  return generation;
}

bool Cache::ReclaimPlan::ramState(uint64_t block) const {
  return cache.states_.stateResident(block) && !statesOut_.contains(block);
}

// The resume point a walk that keeps it keeps at a step of the generation:
// the cache's (StateCache::resumePoint), and from the first landing on,
// once dropPoisoned has erased it with the states below poisoned blocks,
// the one the cache has then (laterPoint).
bool Cache::ReclaimPlan::keptPoint(uint64_t block, uint32_t generation) const {
  if (!walk.keepResumePoint)
    return false;
  const uint64_t point = cache.states_.resumePoint();
  return block == point || (generation != 0 && dropped_.contains(point) && block == laterPoint());
}

// StateCache::resumePoint once dropPoisoned has erased what it will
// (dropped_): the newest ordinary state left in RAM, else such a
// checkpoint, those the reclaims took gone.
uint64_t Cache::ReclaimPlan::laterPoint() const {
  const StateCache &states = cache.states_;
  const auto left = [&](const CacheEvictionCandidate &state) {
    return !dropped_.contains(state.id) && !statesOut_.contains(state.id);
  };
  std::optional<CacheEvictionCandidate> newest;
  for (auto state = states.ordinaryCandidate(false); state; state = states.ordinaryCandidate(false, {}, state))
    if (left(*state))
      newest = state;
  if (!newest)
    for (auto state = states.checkpointCandidate(false); state; state = states.checkpointCandidate(false, {}, state))
      if (left(*state))
        newest = state;
  return newest ? newest->id : 0;
}

bool Cache::ReclaimPlan::keptState(uint64_t block) const {
  return keptPoint(block, generation_) || walk.keepsState(block);
}

bool Cache::ReclaimPlan::keptLeaf(uint64_t block) const { return walk.keepsBlock(block); }

// A reclaim of the state writes it: it has no copy, or a redundant one that
// may have gone.
bool Cache::ReclaimPlan::writes(uint64_t block) const {
  const StateCache &states = cache.states_;
  return states.reclaimWrites(block) || (copiesMayGo_ && states.onDisk(block));
}

// transfersInFlight once the reclaims before started what they did.
bool Cache::ReclaimPlan::transfers() const {
  return cache.transfersInFlight() || std::ranges::any_of(demotions_, [](uint32_t pages) { return pages != 0; }) ||
         writing_;
}

// A claim that can wait's plan writes and demotes what an Immediate one
// passes over.
ReclaimTiming Cache::ReclaimPlan::timing() const noexcept {
  return waits_ ? ReclaimTiming::CanWait : walk.timing;
}

bool Cache::ReclaimPlan::skipsWrites() const noexcept {
  return !waits_ && walk.skipStateWrites;
}

// The first cache of the unit that an Immediate claim's reclaims pass over
// as only a write or a demotion frees it: a state that needs a write, a
// leaf a state needs without a copy while the tier takes writes.
bool Cache::ReclaimPlan::reachesUncopied(uint64_t block, bool state) const {
  if (passing_ || timing() != ReclaimTiming::Immediate)
    return false;
  if (state)
    return !keptState(block) && skipsWrites() && writes(block);
  return !keptLeaf(block) && !cache.kv_.slot(block) && needed(block) && cache.kvTierWritable();
}

// StateCache::reclaimOutcome with the write the reclaims before started.
StateEviction Cache::ReclaimPlan::stateGoes(uint64_t block) const {
  const bool written = writes(block);
  if (skipsWrites() && written)
    return {};
  const StateEviction outcome = cache.states_.reclaimOutcome(block, walk.unwritten);
  if (outcome.evicted && writing_ && written && walk.unwritten != StateCache::Unwritten::Drop)
    return {false, true};
  return outcome;
}

// reclaimKvLeaf and demoteKv with what the reclaims before took and started.
// A copy of what the leaf's page holds that may have gone for room counts
// only where the leaf would go at once without it, and then as the copy it
// may still be, which the leaf's parent keeps below it; that of a spilled
// page, its only K/V, stays. Once a transfer has landed (`landed`), the tier
// has room for the demotion.
Cache::ReclaimPlan::LeafGoes Cache::ReclaimPlan::leafGoes(uint64_t block, ReclaimTiming timing, bool landed) const {
  const KvCache &kv = cache.kv_;
  if (!kv.slot(block))
    return leafGoesUncopied(block, timing, landed);
  if (!copiesMayGo_ || cache.pool_.form(kv.page(block)) == kv::PageForm::Index)
    return LeafGoes::Copy;
  const LeafGoes uncopied = leafGoesUncopied(block, timing, landed);
  return uncopied == LeafGoes::Erase ? LeafGoes::Copy : uncopied;
}

Cache::ReclaimPlan::LeafGoes Cache::ReclaimPlan::leafGoesUncopied(uint64_t block, ReclaimTiming timing,
                                                                  bool landed) const {
  const KvCache &kv = cache.kv_;
  if (needed(block)) {
    const bool writable = cache.kvTierWritable();
    if (writable) {
      if (timing == ReclaimTiming::Immediate)
        return LeafGoes::Stays;
      // The tier's write share, and a slot in the quota the way the take
      // finds one: free quota, else the copies making room gives up.
      const bool room = (landed || !tierRoom_ || *tierRoom_) &&
                        slotFits(cache.tier_->slotBytes(), cache.kvNeededByStateInUse(block));
      if (room)
        return LeafGoes::Demotion;
      if (transfers())
        return LeafGoes::Pending;
    }
    if (stateBelow(block) && writable)
      return LeafGoes::Stays;
  }
  if ((kv.hasDiskChildren(block) || diskParents_.contains(block)) && (writing_ || kv.subtree(block).empty()))
    return LeafGoes::Stays;
  return LeafGoes::Erase;
}

// Whether a write of `bytes` finds a slot in the quota after the writes the
// reclaims started: free quota first, then the copies freeDiskSpace gives
// up, in its order (Cache::nextDiskReplacement). Without a quota any slot
// fits; a limit that cannot hold the slot fits none.
bool Cache::ReclaimPlan::slotFits(uint64_t bytes, bool inUse) const {
  const model::DiskBudget *budget = cache.diskBudget_.get();
  if (!budget)
    return true;
  if (budget->limitBytes() < bytes)
    return false;
  while (budget->usedBytes() + quotaBytes_ + bytes > budget->limitBytes() + replacedBytes_) {
    const std::optional<DiskReplacement> next = cache.nextDiskReplacement(diskCursor_, inUse, false);
    if (!next)
      return false;
    replacedBytes_ += cache.replacedBytes(*next);
    diskCursor_.pass(*next);
  }
  return true;
}

// A state's RAM copy goes: its write takes the one write slot and, as a
// demotion's does, a slot of the quota the states' file shares with the KV
// tier (openDiskTier), past which making room gives up redundant copies;
// without a copy and without a write it is dropped.
void Cache::ReclaimPlan::stateOut(uint64_t block, bool inUse) {
  const StateCache &states = cache.states_;
  const bool written = writes(block);
  mark(block, true, inUse);
  statesOut_.insert(block);
  if (written) {
    writing_ = written_ = true;
    const model::DiskBudget *budget = cache.diskBudget_.get();
    quotaBytes_ += model::SlotFile::slotBytesFor(states.bytes(block));
    copiesMayGo_ = copiesMayGo_ || !budget || budget->usedBytes() + quotaBytes_ > budget->limitBytes();
  } else if (!states.onDisk(block)) {
    statesDropped_.insert(block);
  }
  demotionLands(block);
}

// A demotion in flight on a block that was no resident leaf, or kept a
// state in RAM, frees its page as it lands once the block's last resident
// child or its state left RAM, unless a request or a lease still uses it
// (Cache::freesOnLanding): for a claim that can wait, the page lands, with
// the transfers in flight or once what held it left, and the parent goes
// once it has.
void Cache::ReclaimPlan::demotionLands(uint64_t block) {
  const KvCache &kv = cache.kv_;
  const auto held = heldDemotions_.find(block);
  if (timing() == ReclaimTiming::Immediate || held == heldDemotions_.end() || ramState(block) || used(block) ||
      hasResidentChild(block))
    return;
  gone_[block] = {.generation = std::max(generation_, 1u), .order = held->second};
  heldDemotions_.erase(held);
  demoting_.insert(block);
  ++landing_[static_cast<size_t>(cache.pool_.form(kv.page(block)))];
  joinParent(block);
}

// A state goes, and its leaf joins the leaves; one that leaves the cache
// unpins its tail, whose trim frees the pages it returns.
uint32_t Cache::ReclaimPlan::takeState(uint64_t block, bool inUse) {
  stateOut(block, inUse);
  joinSelf(block);
  return statesDropped_.contains(block) ? trimTail(block) : 0;
}

// The Full pages of a dropped state's tail that no other state's tail
// holds and every window has passed, trimmed into the free Slim pages left
// (KvCache::trimCandidate, Cache::trimWindows without an allocation).
uint32_t Cache::ReclaimPlan::trimTail(uint64_t block) {
  const KvCache &kv = cache.kv_;
  uint32_t trimmed = 0;
  for (uint32_t index = 0; block && slimRoom_ && index < cache.window_->windowPages;
       ++index, block = kv.key(block).parent) {
    const uint32_t page = kv.page(block);
    if (page == KvCache::noPage || trimmed_.contains(page) || cache.pool_.form(page) != kv::PageForm::Full ||
        kv.transfer(block) != KvCache::Transfer::None || !kv.matchable(block) || leftRam_.contains(block) ||
        demoting_.contains(block) || pinnedBesides(block) || !cache.windowsPassed(page, kv.chainLength(block) - 1))
      continue;
    trimmed_.insert(page);
    --slimRoom_;
    ++trimmed;
  }
  return trimmed;
}

// A state the reclaims leave, in RAM or on disk, holds the block in its
// tail: it sits on the block or below it, within the window.
bool Cache::ReclaimPlan::pinnedBesides(uint64_t block) const {
  std::vector<std::pair<uint64_t, uint32_t>> pending{{block, 0u}};
  while (!pending.empty()) {
    const auto [id, depth] = pending.back();
    pending.pop_back();
    if (cache.states_.contains(id) && !statesDropped_.contains(id))
      return true;
    if (depth + 1 < cache.window_->windowPages)
      for (const uint64_t child : cache.kv_.children(id))
        pending.emplace_back(child, depth + 1);
  }
  return false;
}

// A victim of the unit goes now, before any later step.
void Cache::ReclaimPlan::took() {
  if (!unitTakes_.now)
    ++nowTaken_;
}

// The unit's first later step ends what take() makes of it now. take()
// starts the writes of the later demotions of that step's class, of blocks
// without a state in RAM, at once, so that they take the tier's room now:
// what is in use goes only once the ordinary class's later steps have.
void Cache::ReclaimPlan::tookLater(uint64_t block, LeafGoes goes, bool inUse) {
  if (!unitTakes_.now) {
    unitTakes_.now = nowTaken_;
    unitTakes_.laterInUse = inUse;
  }
  if (goes != LeafGoes::Demotion || ramState(block) || inUse != unitTakes_.laterInUse)
    return;
  unitTakes_.writes.push_back(block);
  if (tierRoom_ && *tierRoom_)
    --*tierRoom_;
}

// A leaf goes: its page frees now, through its copy or erased, or as its
// demotion's copy lands, a generation after the walk's. A later step's page
// frees once the transfers before it land, in the walk's generation, and
// holds its extent until then. A restored leaf's page counts only toward
// the extents that leave (restored_): restores free nothing.
Cache::ReclaimPlan::Step Cache::ReclaimPlan::takeLeaf(uint64_t block, LeafGoes goes, bool inUse, bool later) {
  if (!later)
    took();
  const KvCache &kv = cache.kv_;
  const uint32_t page = kv.page(block);
  const kv::PageForm form = trimmed_.contains(page) ? kv::PageForm::Slim : cache.pool_.form(page);
  mark(block, false, inUse);
  const uint64_t parent = kv.key(block).parent;
  const bool restored = restoring_.contains(block);
  std::array<uint32_t, 3> &landing = restored ? restored_ : landing_;
  const bool counted = !restored && form == walk.form;
  if (goes == LeafGoes::Demotion) {
    // It is sure to start with the tier's room, or once a transfer landed:
    // leafGoesUncopied granted its slot in the quota (slotFits).
    const model::DiskBudget *budget = cache.diskBudget_.get();
    const uint64_t slot = cache.tier_->slotBytes();
    const bool sure = (tierRoom_ || later) && budget;
    demoting_.insert(block);
    gone_[block] = {.generation = generation_ + 1, .order = departures_++};
    ++demotions_[static_cast<size_t>(form)];
    if (tierRoom_ && !later)
      --*tierRoom_;
    quotaBytes_ += slot;
    copiesMayGo_ = copiesMayGo_ || !budget || budget->usedBytes() + quotaBytes_ > budget->limitBytes();
    fixPage(page);
    landing[static_cast<size_t>(form)] += sure;
    joinParent(block);
    return {.progress = true, .pages = sure && counted ? 1u : 0u};
  }
  if (later) {
    fixPage(page);
    ++landing[static_cast<size_t>(form)];
    demoting_.insert(block);
    gone_[block] = {.generation = generation_, .taken = true, .order = departures_++};
  } else {
    freePage(page);
    leftRam_.insert(block);
    // As KvCache::dropPage and erase do (KvCache::inherit), the parent takes
    // the leaf's recency once it holds no resident child, a leaf the walk
    // takes or not, before a demotion of its own lands; an Erase first gives
    // the leaf what dropping its disk-only subtree passes up.
    if (goes == LeafGoes::Erase && (kv.hasDiskChildren(block) || diskParents_.contains(block)))
      recency_[block] = erasedRecency(block);
    if (parent && !hasResidentChild(parent) && !cache.pool_.pageActive(kv.page(parent)))
      recency_[parent] = std::max(recencyOf(parent), recencyOf(block));
    demotionLands(parent);
  }
  if (goes == LeafGoes::Copy) {
    if (parent)
      diskParents_.insert(parent);
  } else {
    // Its disk-only subtree goes with it, and the states on it.
    if (cache.states_.contains(block))
      statesDropped_.insert(block);
    if (kv.hasDiskChildren(block) || diskParents_.contains(block))
      for (const uint64_t below : kv.subtree(block))
        if (cache.states_.contains(below))
          statesDropped_.insert(below);
  }
  joinParent(block);
  return {.progress = true, .pages = counted ? 1u : 0u};
}

void Cache::ReclaimPlan::mark(uint64_t block, bool state, bool inUse) {
  const bool resumePoint = state && block == cache.states_.resumePoint();
  const bool wanted = state ? std::ranges::find(cache.wanted_.states(), block) != cache.wanted_.states().end()
                            : cache.wanted_.holds(block);
  const bool certain = inUse || resumePoint || wanted;
  certain_ = certain_ || certain;
  inUse_ = inUse_ || inUse;
}

std::optional<engine::Victim> Cache::peekReclaim(const Walk &walk, const engine::Victim *after) const {
  // The units of one walk come one after another: the next peek goes on with
  // the plan the last one left, and any other starts one.
  const uint64_t key = after ? blockOfKey(after->key) : 0;
  const uint32_t serial = static_cast<uint32_t>(key >> kPlanShift);
  const uint64_t wanted = after ? (key & kPlanMask) + 1 : 0;
  if (!after || !plan_ || plan_->serial != serial || plan_->produced != wanted ||
      !plan_->weighs(walk)) {
    plan_ = std::make_shared<ReclaimPlan>(*this, walk, ++planSerial_ & kPlanMask);
    for (uint64_t unit = 0; unit < wanted; ++unit)
      if (!plan_->next())
        return std::nullopt;
  }
  return plan_->next();
}

uint32_t Cache::trimRoom() const {
  uint32_t empty = 0;
  for (const KvPool::ExtentCounts &extent : pool_.extentCounts({}))
    empty += extent.allocated && !extent.usedPages && extent.form == kv::PageForm::Slim;
  return pool_.freePageCount(kv::PageForm::Slim) - (empty > 1 ? (empty - 1) * pool_.extentPages() : 0);
}

bool Cache::trims(const Walk &walk) const noexcept {
  return walk.pages && window_ && window_->trim && walk.form == kv::PageForm::Full;
}

std::vector<uint32_t> Cache::trimmedBy(const Walk &walk) const {
  if (!trims(walk))
    return {};
  std::vector<uint32_t> pages = trimmablePages();
  pages.resize(std::min<size_t>(pages.size(), pool_.freePageCount(kv::PageForm::Slim)));
  return pages;
}

uint64_t Cache::landedSince(const Walk &walk, const std::array<uint32_t, 3> &before) const noexcept {
  if (walk.timing == ReclaimTiming::Immediate)
    return 0;
  const std::array<uint32_t, 3> landing = demotedPages();
  uint64_t bytes = 0;
  for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index}) {
    const auto index = static_cast<size_t>(form);
    bytes += extentsCompleted(form, landing[index] - std::min(landing[index], before[index]), before[index],
                              walk.keeps(form));
  }
  return bytes;
}

std::optional<engine::Victim> Cache::peek(const VictimQuery &query, const engine::Victim *after) const {
  const Walk walk = walkFor(query);
  if (walk.device)
    return peekReclaim(walk, after);
  const Unit afterUnit = after ? static_cast<Unit>(unitOfKey(after->key)) : Unit{};
  // A page need's free pages first: the other forms' empty extents, then a
  // windowed cache's trims for Full pages; then its evictions, which free
  // the pages of its form the cache holds beyond the free ones, and the Full
  // pages a state that goes lets trim: of a form it holds none of, a walk
  // would weigh every unit for none.
  if (walk.pages) {
    const auto evictions = [&](const engine::Victim *previous) -> std::optional<engine::Victim> {
      if (!previous && !trims(walk) && reusablePages(walk.scope.upTo, walk.form) <= pool_.freePageCount(walk.form))
        return std::nullopt;
      return peekReclaim(walk, previous);
    };
    if (after && afterUnit != Unit::Extent)
      return evictions(afterUnit == Unit::Reclaim ? after : nullptr);
    if (!after && (spill_ || window_)) {
      // The empty extents of the other forms beyond the runway each keeps.
      std::array<uint32_t, 3> empty{};
      for (const KvPool::ExtentCounts &extent : pool_.extentCounts({}))
        empty[static_cast<size_t>(extent.form)] += extent.allocated && !extent.usedPages;
      uint64_t bytes = 0;
      for (const kv::PageForm other : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index})
        if (const uint32_t count = empty[static_cast<size_t>(other)]; other != walk.form && count > 1)
          bytes += uint64_t{count - 1} * pool_.extentPages() * pool_.bytesPerPage(other);
      if (bytes)
        return engine::Victim{.key = unitKey(static_cast<uint8_t>(Unit::Extent), 0), .bytes = bytes};
    }
    if (const auto pages = static_cast<uint32_t>(std::min<size_t>(trimmedBy(walk).size(), trimRoom())))
      return engine::Victim{.key = unitKey(static_cast<uint8_t>(Unit::Trim), 0), .pages = pages};
    return evictions(nullptr);
  }
  // A state need's units, a state each.
  WalkCursor cursor;
  if (after) {
    cursor = cursorAfter(afterUnit, blockOfKey(after->key));
    holdAfter(walk, cursor);
  }
  for (std::optional<WalkUnit> unit; (unit = nextStepUnit(walk, cursor));)
    if (std::optional<engine::Victim> victim = victimOf(walk, *unit))
      return victim;
  return std::nullopt;
}

FreeResult Cache::take(const VictimQuery &query, const engine::Victim &victim) {
  const std::shared_ptr<ReclaimPlan> plan = std::exchange(plan_, nullptr);
  const Walk walk = walkFor(query);
  const uint64_t block = blockOfKey(victim.key);
  const auto met = [](uint64_t nowBytes = 0, uint64_t laterBytes = 0) {
    return FreeResult{FreeOutcome::Met, nowBytes, laterBytes};
  };
  switch (static_cast<Unit>(unitOfKey(victim.key))) {
  case Unit::Extent:
    if (const uint64_t released = releaseOtherForms(walk.form))
      return met(released);
    return {};
  case Unit::Trim:
    // Into free Slim pages alone, as peek weighed them: the ledger takes
    // outside any allocation's scope, and the Slim runway refills where the
    // engine trims in one (Engine::prepare).
    return trimWindows(false) ? met() : FreeResult{};
  case Unit::Reclaim: {
    // As peek weighed the unit: a device need's memory reclaims or a page
    // need's evictions until they free what it was weighed at, the bytes of
    // the extents released now and of those its demotions complete as they
    // land, or pages of its form, free now or as demotions land; in a
    // windowed cache, a page need's trims of what a state that left
    // unpinned. A copy given up for room or a write the tier refused may
    // bring in another extent than weighed; a unit weighed at nothing goes
    // on, for a need of pages alone, past states to the end of the reclaims.
    // A unit with later steps takes the victims before the first of them and
    // starts the writes of its later demotions whose pages free as they land,
    // a child's before its parent's; its retries take the rest as landings
    // allow.
    const ReclaimPlan::Takes takes =
        ReclaimPlan::takesOf(*this, plan, walk, victim);
    const std::array<uint32_t, 3> demotions = demotionPages();
    const std::array<uint32_t, 3> landing = demotedPages();
    const auto form = static_cast<size_t>(walk.form);
    const uint32_t pagesBefore = pool_.freePageCount(walk.form) + landing[form];
    uint64_t released = 0, later = 0;
    uint32_t pages = 0;
    bool took = false;
    CacheReclaimResult reclaimed;
    const auto count = [&] {
      later = landedSince(walk, landing);
      const uint32_t pagesNow = pool_.freePageCount(walk.form) + demotedPages()[form];
      pages = pagesNow - std::min(pagesNow, pagesBefore);
    };
    victimsLeft_ = takes.now;
    do {
      reclaimed = walk.device ? takeReclaim(walk, demotions, landing) : evictOne(walk);
      if (!walk.device && reclaimed.evictedState && trims(walk))
        static_cast<void>(trimWindows(false));
      released += reclaimed.reclaimedBytes;
      took = took || reclaimed.madeProgress || reclaimed.evictedKv;
      count();
    } while (reclaimed.madeProgress &&
             (walk.device ? released + later < victim.bytes
                          : pages < victim.pages || (!victim.pages && !walk.statesAreUnits && !pages)));
    victimsLeft_.reset();
    // The unit's later demotions whose pages free as they land, until the
    // tier or the quota has no room for one now; an Immediate claim waits
    // for no write (reclaimKvLeaf).
    if (walk.timing != ReclaimTiming::Immediate) {
      std::unordered_set<uint64_t> freeing = freeingDemotions();
      for (const uint64_t write : takes.writes) {
        if (!kv_.contains(write) || kv_.transfer(write) != KvCache::Transfer::None || kv_.slot(write) ||
            !kvNeededByState(write) || !freesOnLanding(write, freeing))
          continue;
        if (demoteKv(write) != LeafReclaim::Started)
          break;
        freeing.insert(write);
        took = true;
      }
    }
    count();
    if (took)
      return met(released, walk.device ? later : 0);
    return {reclaimed.pending ? FreeOutcome::Pending : FreeOutcome::Short};
  }
  case Unit::Checkpoint:
  case Unit::State:
  case Unit::UsedState:
    break;
  default:
    return {};
  }
  if (!states_.reclaimable(block))
    return {};
  const UnitTaken taken = takeUnit(walk, {static_cast<Unit>(unitOfKey(victim.key)), {block, 0}}, walk.timing);
  if (taken.taken)
    return met();
  return {taken.pending ? FreeOutcome::Pending : FreeOutcome::Short};
}

void Cache::setWantedChains(std::vector<std::span<const uint64_t>> chains) {
  wanted_ = {};
  for (const std::span<const uint64_t> chain : chains)
    wanted_.add(chain);
}

// Reclaim: memory returns as whole extents. Free pages go first, an empty
// extent as it is and the pages scattered over the others once they fill
// one; then one victim at a time: KV no state restores through, then the
// shared recency order, states and resident KV leaves alike. Each entry
// point below takes the walk's units, as peek gives them and take takes
// them.

uint64_t Cache::releaseEmptyExtents(bool keepRunway) {
  return pool_.reclaimEmptyExtents(keepRunway, std::numeric_limits<uint32_t>::max());
}

CacheReclaimResult Cache::evictAll(ReclaimClass upTo) {
  uint64_t released = releaseEmptyExtents(false);
  bool evicted = false;
  // Everything unpinned goes before anything moves.
  while (evictOne({.scope = {.upTo = upTo}}).madeProgress)
    evicted = true;
  do {
    released += releaseEmptyExtents(false);
  } while (upTo == ReclaimClass::InUse && compactExtent());
  return {.madeProgress = evicted || released != 0, .reclaimedBytes = released};
}

CacheReclaimResult Cache::reclaimOne(CacheReclaimMode mode, ReclaimClass upTo,
                                     bool keepResumePoint, bool keepRunway) {
  return reclaimStep({.scope = {.upTo = upTo, .pagesOnly = mode == CacheReclaimMode::ReusePages},
                      .keepResumePoint = keepResumePoint},
                     mode == CacheReclaimMode::ReleaseExtents, keepRunway);
}

CacheReclaimResult Cache::reclaimStep(const Walk &walk, bool release, bool keepRunway) {
  if (release) {
    if (const CacheReclaimResult released = releaseExtent(keepRunway);
        released.madeProgress)
      return released;
  }
  CacheReclaimResult evicted = evictOne(walk);
  if (evicted.madeProgress && release)
    evicted.reclaimedBytes += pool_.reclaimEmptyExtents(keepRunway, 1);
  return evicted;
}

CacheReclaimResult Cache::evictOne(const Walk &walk, std::optional<std::array<bool, 3>> reach) {
  if (const std::optional<Victim> victim = takeOldest(walk, {}, reach))
    return {.madeProgress = true, .evictedState = !victim->kv};
  // Nothing to reclaim now. Transfers land only in pollTransfers(), so what
  // was in flight during the pass still is, and comes back.
  return {.pending = returns(walk)};
}

std::optional<Cache::Victim> Cache::takeOldest(const Walk &walk, WalkCursor cursor,
                                               std::optional<std::array<bool, 3>> reach) {
  using Phase = WalkCursor::Phase;
  ReclaimTiming timing = walk.timing;
  for (Phase phase = cursor.phase; const std::optional<WalkUnit> unit = nextStepUnit(walk, cursor);) {
    // A take ends where its unit's later steps begin (victimsLeft_).
    if (victimsLeft_ && !*victimsLeft_)
      return std::nullopt;
    // The leaves of each class wait for the tier on their own.
    if (cursor.phase != phase) {
      phase = cursor.phase;
      timing = walk.timing;
    }
    const bool leaf = unit->unit == Unit::DeadLeaf || unit->unit == Unit::Leaf || unit->unit == Unit::UsedLeaf;
    if (leaf && reach && !leadsToReach(unit->candidate.id, *reach))
      continue;
    const UnitTaken taken = takeUnit(walk, *unit, unit->unit == Unit::DeadLeaf ? walk.timing : timing);
    if (taken.taken) {
      if (victimsLeft_)
        --*victimsLeft_;
      return Victim{leaf};
    }
    if (!taken.pending)
      continue;
    switch (unit->unit) {
    case Unit::Checkpoint:
      // One whose write must wait for the one in flight ends the
      // checkpoints, holding back nothing else.
      cursor = WalkCursor{.phase = walk.scope.statesOnly ? Phase::Ordinary : Phase::DeadLeaves};
      break;
    case Unit::State:
    case Unit::UsedState:
      // The newer states of its class follow it once that write has landed.
      cursor.statesHeld = true;
      cursor.state.reset();
      break;
    case Unit::Leaf:
    case Unit::UsedLeaf:
      // The tier takes no demotion until a transfer lands: the leaves that
      // need one wait with this one, and those whose page frees now go.
      timing = ReclaimTiming::Immediate;
      break;
    default:
      break;
    }
  }
  return std::nullopt;
}

uint64_t Cache::releaseOtherForms(kv::PageForm form) {
  if (!spill_ && !window_)
    return 0;
  const uint32_t all = std::numeric_limits<uint32_t>::max();
  uint64_t released = 0;
  for (const kv::PageForm other : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index})
    if (other != form)
      released += pool_.reclaimEmptyExtents(true, all, other);
  if (!released && form == kv::PageForm::Index && compactExtent(kv::PageForm::Full))
    released = pool_.reclaimEmptyExtents(true, all, kv::PageForm::Full);
  return released;
}

CacheReclaimResult Cache::takeReclaim(const Walk &walk, const std::array<uint32_t, 3> &demotions,
                                      const std::array<uint32_t, 3> &unit) {
  plan_.reset();
  // Free pages first, as the extent they fill; one that empties the runway
  // kept gives nothing yet, and the next one does.
  for (CacheReclaimResult released; (released = releaseExtent(walk)).madeProgress;)
    if (released.reclaimedBytes)
      return released;
  // Then the victims in reclaimOne's order while KV of some form can empty
  // an extent past the runway the walk keeps, the leaves that lead to such
  // an extent alone, until one is released, the pages the unit's demotions
  // free as they land complete one, or a state goes; otherwise only a
  // state. Those pages count toward the extent; the demotions in flight
  // before the unit hold theirs, of their form. Whatever it returns says
  // whether KV went.
  bool evictedKv = false;
  const auto outcome = [&](CacheReclaimResult result) {
    result.evictedKv = evictedKv;
    return result;
  };
  for (;;) {
    const std::array<bool, 3> reach = formsInReach(walk, demotions);
    if (std::ranges::none_of(reach, std::identity{}))
      break;
    const CacheReclaimResult evicted = evictOne(walk, reach);
    if (!evicted.madeProgress || evicted.evictedState)
      return outcome(evicted);
    evictedKv = true;
    if (const CacheReclaimResult released = releaseExtent(walk); released.reclaimedBytes)
      return outcome(released);
    if (landedSince(walk, unit))
      return outcome({.madeProgress = true});
  }
  return outcome(recycleState(walk));
}

uint32_t Cache::reusablePages(ReclaimClass upTo, kv::PageForm form) const {
  const uint32_t unheld = pool_.reusablePages(form);
  if (upTo == ReclaimClass::InUse)
    return unheld;
  return unheld - kv_.idlePagesOnChains(states_.usedStates(), form);
}

StateRoom Cache::reclaimOneState(StateClass state, bool growth) {
  using Unwritten = StateCache::Unwritten;
  const bool checkpointsOnly = state == StateClass::Checkpoint;
  // Only a state in use cannot leave a victim to the write in flight:
  // refused, it reaches the disk tier only through that write.
  const Walk walk{.timing = ReclaimTiming::Immediate,
                  .unwritten = state == StateClass::InUse ? Unwritten::Drop : Unwritten::Wait};
  // Without growth an extent's bytes are no room for a state's buffers, and
  // only states go.
  if (growth && !checkpointsOnly) {
    if (const CacheReclaimResult released = releaseExtent(false); released.reclaimedBytes)
      return {true, released.reclaimedBytes};
  }
  // The oldest checkpoint, then the oldest ordinary state, as the walk for
  // states alone gives them.
  WalkCursor states;
  const std::optional<WalkUnit> checkpoint = nextUnit({.statesOnly = true}, states);
  if (checkpoint && checkpoint->unit == Unit::Checkpoint && takeUnit(walk, *checkpoint, walk.timing).taken)
    return {true};
  if (checkpointsOnly)
    return {};
  // KV makes room with the extent its free pages fill, released at once,
  // and goes only while one it leads to can be emptied (takeOldest).
  std::array<bool, 3> reach{};
  if (growth)
    reach = formsInReach(walk, demotionPages());
  if (std::ranges::any_of(reach, std::identity{})) {
    while (const std::optional<Victim> victim =
               takeOldest(walk, WalkCursor{.phase = WalkCursor::Phase::DeadLeaves}, reach)) {
      if (!victim->kv)
        return {true};
      if (const CacheReclaimResult released = releaseExtent(false); released.reclaimedBytes)
        return {true, released.reclaimedBytes};
    }
  } else {
    WalkCursor ordinary{.phase = WalkCursor::Phase::Ordinary};
    const std::optional<WalkUnit> oldest = nextUnit({.statesOnly = true}, ordinary);
    if (oldest && oldest->unit == Unit::State && takeUnit(walk, *oldest, walk.timing).taken)
      return {true};
  }
  return {};
}

StateRoom Cache::reclaimInUseState() {
  WalkCursor cursor{.phase = WalkCursor::Phase::InUse};
  const std::optional<WalkUnit> oldest = nextUnit({.upTo = ReclaimClass::InUse, .statesOnly = true}, cursor);
  return {oldest && takeUnit({}, *oldest, ReclaimTiming::CanWait).taken};
}

CacheReclaimResult Cache::reclaimStateForLane(ReclaimClass upTo) {
  // No page covers a state's buffers.
  return recycleState({.scope = {.upTo = upTo}, .refused = Shortfall{}});
}

CacheReclaimResult Cache::recycleState(const Walk &walk) {
  // The oldest checkpoint, else the oldest ordinary state; running work
  // takes a state in use after every other, once no transfer in flight can
  // return what is needed first.
  Walk states = walk;
  states.scope.statesOnly = true;
  states.scope.pagesOnly = false;
  WalkCursor cursor;
  while (const std::optional<WalkUnit> state = nextStepUnit(states, cursor)) {
    if (skipsState(states, state->candidate.id))
      continue;
    if (victimsLeft_ && !*victimsLeft_)
      return {};
    const StateEviction eviction = states_.reclaim(state->candidate.id, states.unwritten);
    if (eviction.evicted && victimsLeft_)
      --*victimsLeft_;
    return {.madeProgress = eviction.evicted, .pending = eviction.pending, .evictedState = eviction.evicted};
  }
  return {.pending = returns(states)};
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
    // A dead leaf goes, or stays, with the dead leaves (oldestDeadKvLeaf).
    if (kvNeededByState(candidate->id) && (withRamState || !states_.stateResident(candidate->id)) &&
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
  // page returns only when the copy lands: the reclaim reached a leaf a copy
  // ahead would have let go at once.
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
    kv_.erasePoisonedLeaf(block);
    return !kv_.contains(block);
  });
}

bool Cache::leadsToReach(uint64_t block, const std::array<bool, 3> &reach) const {
  for (; block; block = kv_.key(block).parent)
    if (reach[static_cast<size_t>(pool_.form(kv_.page(block)))])
      return true;
  return false;
}

std::array<bool, 3> Cache::formsInReach(const Walk &walk, const std::array<uint32_t, 3> &demoting) const {
  std::array<bool, 3> reach{};
  for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index}) {
    const auto index = static_cast<size_t>(form);
    reach[index] = walk.reaches(form, reusablePages(walk.scope.upTo, form), demoting[index], pool_.extentPages());
  }
  return reach;
}

CacheReclaimResult Cache::releaseExtent(const Walk &walk) {
  constexpr std::array kForms{kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index};
  const auto releases = [&](kv::PageForm form) {
    return walk.releases(form, pool_.freePageCount(form), pool_.extentPages());
  };
  for (const kv::PageForm form : kForms)
    if (releases(form))
      if (const uint64_t bytes = pool_.reclaimEmptyExtents(walk.keeps(form), 1, form))
        return {true, bytes};
  for (const kv::PageForm form : kForms)
    if (releases(form) && compactExtent(form))
      return {true, pool_.reclaimEmptyExtents(walk.keeps(form), 1, form)};
  return {};
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
      active.indexPages += pool_.form(destination) == kv::PageForm::Index;
      active.indexPages -= pool_.form(page) == kv::PageForm::Index;
      page = destination;
      first = std::min(first, index);
    }
    if (first < pages)
      pagesChanged(active, static_cast<uint32_t>(first));
  }
}

bool Cache::transfersInFlight() const noexcept {
  // A write counts by what its landing frees: a page written through or
  // copied ahead holds no block a reclaim could take, and returns something
  // only where it spills once its copy lands or is a leaf the copy then lets
  // go.
  return !restores_.empty() || writes(WriteClass::Reclaim) || copyAheadReturns() ||
         states_.writing() || spillingWrites();
}

bool Cache::returnsInFlight(const Shortfall &shortfall) const noexcept {
  if (!restores_.empty() || writes(WriteClass::Reclaim) || copyAheadReturns() || states_.writing())
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

uint64_t Cache::copyingAheadBytes() const noexcept {
  uint64_t bytes = states_.persistingBytes();
  for (const Write &write : writes_)
    if (write.write == WriteClass::Persist)
      bytes += pool_.bytesPerPage(pool_.form(kv_.page(write.block)));
  return bytes;
}

uint32_t Cache::writes(WriteClass write) const noexcept {
  return static_cast<uint32_t>(
      std::ranges::count(writes_, write, [](const Write &each) { return each.write; }));
}

bool Cache::copyAheadReturns() const noexcept {
  return std::ranges::any_of(writes_, [&](const Write &each) {
    return each.write == WriteClass::Persist && kv_.residentLeaf(each.block);
  });
}

uint32_t Cache::spillingWrites() const noexcept {
  // A page a Spill request holds frees once its copy lands, written through
  // or copied ahead, when it may spill then and an Index page is there to
  // take its index.
  if (!spill_)
    return 0;
  const auto spillable = std::ranges::count_if(writes_, [&](const Write &write) { return spillsOnLanding(write); });
  return std::min(static_cast<uint32_t>(spillable), pool_.freePageCount(kv::PageForm::Index));
}

bool Cache::spillsOnLanding(const Write &write) const {
  // pendingPages counts every demotion already (writes(Reclaim)).
  return write.write != WriteClass::Reclaim && spillEligibility(write.block) == Eligibility::Eligible;
}

bool Cache::owedCopiesReturn(uint32_t pages, kv::PageForm form) const {
  if (form != kv::PageForm::Full)
    return false;
  const uint32_t needed = pages ? pages : pool_.extentPages();
  const uint32_t free = std::min(pool_.freePageCount(kv::PageForm::Full), needed);
  const uint32_t copies = copiesToSpill(std::max(needed - free, 1u));
  return copies && free + copies >= needed;
}

uint32_t Cache::copiesToSpill(uint32_t enough) const {
  if (!spill_)
    return 0;
  // Requests on one prefix owe its blocks each, one maybe a block another
  // writes.
  std::unordered_set<uint64_t> blocks;
  for (const Write &write : writes_)
    if (spillsOnLanding(write))
      blocks.insert(write.block);
  for (const auto &[_, active] : requests_)
    for (auto block = active.owedCopies.begin(); writes(WriteClass::Spill) && active.mode == KvMode::Spill &&
                                                 block != active.owedCopies.end() && blocks.size() < enough;
         ++block)
      if (spillEligibility(*block) == Eligibility::Eligible)
        blocks.insert(*block);
  return static_cast<uint32_t>(blocks.size());
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
  return commit(active, pages);
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
  // restored has them, spill into free Index pages at once, whose Full
  // pages the retry takes.
  rescanSpills();
  static_cast<void>(writeThrough());
  static_cast<void>(moveLandedPages(std::nullopt));
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

TokenAdmission Cache::spillWritten(SpilledExtents extents) { return moveLandedPages(extents); }

TokenAdmission Cache::moveLandedPages(std::optional<SpilledExtents> extents) {
  if (!spill_)
    return {};
  std::vector<uint32_t> pages;
  for (auto &[_, active] : requests_) {
    if (active.mode != KvMode::Spill)
      continue;
    const auto blocks = static_cast<uint32_t>(active.cachedBlocks.size());
    const uint32_t end = blocks > spill_->windowPages ? blocks - spill_->windowPages : 0;
    // The scan's mark passes what is spilled or kept Full, and stops at the
    // first page that may spill later. A page waits for its own copy to
    // land: one being written, or whose block has none of its own, as a
    // block recomputed over another conversation's copy (KvCache::insert)
    // until its rewrite lands. The pages past it whose copies landed, as a
    // chain's persisted points have them, spill now.
    bool settled = true;
    for (uint32_t index = std::max(active.spillScan, spill_->densePages); index < end; ++index) {
      const uint32_t page = active.pages[index];
      const uint64_t block = active.cachedBlocks[index];
      // A writer that kept a page of its own beside its block's restore
      // keeps it Full.
      Eligibility eligibility = Eligibility::KeptFull;
      if (pool_.form(page) == kv::PageForm::Full && kv_.page(block) == page)
        eligibility = kv_.kvCopied(block) && kv_.transfer(block) == KvCache::Transfer::None
                          ? spillEligibility(block)
                          : Eligibility::Later;
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
  if (!extents)
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
  if (!extents)
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
  if (!pages.empty()) {
    if (*extents == SpilledExtents::Released)
      while (compactExtent(kv::PageForm::Full)) {
      }
    static_cast<void>(pool_.reclaimEmptyExtents(*extents == SpilledExtents::KeepRunway,
                                                std::numeric_limits<uint32_t>::max(), kv::PageForm::Full));
  }
  return {};
}

uint32_t Cache::indexPages(uint64_t requestId) const { return request(requestId).indexPages; }

bool Cache::holdsIndexPages(uint64_t requestId) const { return indexPages(requestId) != 0; }

// Trimming: the pages every window has passed give up their windowed
// layers' K/V between commands.

uint32_t Cache::windowStart(const Request &active) const noexcept {
  const auto published = static_cast<uint32_t>(active.cachedBlocks.size());
  return published - std::min(published, window_->windowPages);
}

bool Cache::windowsPassed(uint32_t page, uint32_t index) const {
  if (!pool_.pageActive(page))
    return true;
  // A request holds the page at its chain index, past which it has
  // published windowPages blocks once its rows reach none of its keys.
  return std::ranges::none_of(requests_, [&](const auto &entry) {
    const Request &holder = entry.second;
    return index < holder.pages.size() && holder.pages[index] == page && index >= windowStart(holder);
  });
}

std::vector<uint32_t> Cache::trimmablePages() const {
  std::vector<uint32_t> pages;
  for (auto candidate = kv_.trimCandidate(0); candidate; candidate = kv_.trimCandidate(candidate->id)) {
    const uint32_t page = kv_.page(candidate->id);
    if (windowsPassed(page, kv_.chainLength(candidate->id) - 1))
      pages.push_back(page);
  }
  return pages;
}

uint32_t Cache::trimWindows(bool allocate) {
  if (!window_ || !window_->trim)
    return 0;
  std::vector<uint32_t> pages = trimmablePages();
  // So do the pages requests keep beside their blocks, once each: a
  // writer's own, or one a block left for a writer's, which every request
  // that held it keeps.
  std::vector<uint32_t> kept;
  for (auto &[_, active] : requests_) {
    std::erase_if(active.privatePages,
                  [&](uint32_t index) { return pool_.form(active.pages[index]) != kv::PageForm::Full; });
    for (const uint32_t index : active.privatePages)
      if (windowsPassed(active.pages[index], index))
        kept.push_back(active.pages[index]);
  }
  std::ranges::sort(kept);
  const auto repeated = std::ranges::unique(kept);
  pages.insert(pages.end(), kept.begin(), repeated.begin());
  // Free Slim pages first, the runway's among them; with `allocate` the
  // rest asks for extents of the form, an extent's pages at a time, and a
  // refusal leaves the rest Full: a claim the host exempts for one
  // reservation (MemoryGovernor) trims one extent more on each retry.
  const auto now = std::min<size_t>(pages.size(), pool_.freePageCount(kv::PageForm::Slim));
  if (!allocate)
    pages.resize(now);
  uint32_t moved = 0;
  for (size_t at = 0; at < pages.size();) {
    const size_t count = at < now ? now - at : std::min<size_t>(pool_.extentPages(), pages.size() - at);
    const KvPageMoveResult trimmed = pool_.movePages(std::span<const uint32_t>(pages).subspan(at, count),
                                                     kv::PageForm::Slim);
    if (!trimmed.granted())
      break;
    followMoves(trimmed.moves);
    moved += static_cast<uint32_t>(count);
    at += count;
  }
  trimmedPages_ += moved;
  // The next pass, at the memory wall too, trims without an allocation.
  if (allocate && pool_.freePageCount(kv::PageForm::Slim) < pool_.extentPages())
    static_cast<void>(pool_.fillRunway(kv::PageForm::Slim));
  return moved;
}

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
    // What its next command's windows read holds every layer.
    if (window_) {
      for (uint32_t index = windowStart(active); index < active.pages.size(); ++index)
        check(pool_.form(active.pages[index]) == kv::PageForm::Full, "a request's window holds every layer");
      for (const uint32_t index : active.privatePages)
        check(index < active.cachedBlocks.size() && kv_.page(active.cachedBlocks[index]) != active.pages[index],
              "a request keeps a page of its own beside its block");
    }
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
    // Each block on its own page without a copy of its own is owed: its
    // page spills once that copy lands.
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

TokenAdmission Cache::restoreRequest(uint64_t requestId, const CacheLookup &lookup,
                                     std::array<uint32_t, 3> held) {
  Request &active = request(requestId);
  if (!active.pages.empty() || !active.cachedBlocks.empty()) {
    throw std::logic_error("restore target already owns KV pages");
  }
  if (!lookup.state)
    return {};
  KvCache::Chain chain = kv_.chain(lookup.state->kvBlock());
  const bool spills = active.mode == KvMode::Spill;
  const uint32_t densePages = spill_ ? spill_->densePages : 0;
  // Resident blocks are a rooted prefix of the chain. A disk-only block
  // comes back as what its copy holds (KvCache::copyForm), and past a Spill
  // request's dense history as its index alone. A spilled block comes back
  // whole for a Resident request that would be its only holder, once no
  // transfer moves it: no other request can read its page while its bytes
  // land.
  const auto restoredForm = [&](uint32_t position) {
    return spills && position >= densePages ? kv::PageForm::Index : kv_.copyForm(chain.blocks[position]);
  };
  // The disk-only blocks' pages by form (kv::PageForm), and those taken.
  std::array<uint32_t, 3> restored{};
  std::array<std::vector<uint32_t>, 3> taken;
  std::vector<uint32_t> unspilled;
  for (uint32_t position = 0; position < chain.blocks.size(); ++position) {
    const uint32_t page = chain.pages[position];
    const uint64_t block = chain.blocks[position];
    if (page == KvCache::noPage)
      ++restored[static_cast<size_t>(restoredForm(position))];
    else if (!spills && pool_.form(page) == kv::PageForm::Index && !kv_.held(block) &&
             kv_.transfer(block) == KvCache::Transfer::None)
      unspilled.push_back(page);
  }
  if (std::ranges::any_of(restored, [](uint32_t pages) { return pages != 0; }) && !tier_)
    throw std::logic_error("disk-only KV block without a disk tier");
  // All or nothing: pages one admission took go back when the next fails,
  // and the refusal names the free pages of every form the attempt takes
  // (beyond those held), which the room made for it keeps: their extents,
  // emptied, would go, and the retry would allocate them again.
  const auto refused = [&](TokenAdmission admission) {
    for (const std::vector<uint32_t> &pages : taken)
      for (uint32_t page : pages)
        pool_.releasePage(page, false);
    for (size_t each = 0; each < restored.size(); ++each) {
      const uint64_t takes = restored[each] + (each == static_cast<size_t>(kv::PageForm::Full) ? unspilled.size() : 0);
      if (takes)
        admission.keptPages[each] = static_cast<uint32_t>(
            std::min<uint64_t>(takes + held[each], std::numeric_limits<uint32_t>::max()));
    }
    return admission;
  };
  for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index}) {
    const auto each = static_cast<size_t>(form);
    if (!restored[each])
      continue;
    if (const TokenAdmission admission = admitPages(restored[each], taken[each], form, held[each]);
        !admission.granted())
      return refused(admission);
  }
  if (!unspilled.empty()) {
    const KvPageMoveResult moved =
        pool_.movePages(unspilled, kv::PageForm::Full, held[static_cast<size_t>(kv::PageForm::Full)]);
    if (!moved.granted())
      return refused({TokenAdmissionFailure::Denied, static_cast<uint32_t>(unspilled.size()),
                      pool_.freePageCount(), moved.failure});
    followMoves(moved.moves);
    spilled_.unspills += unspilled.size();
  }
  kv_.retainActive(chain.blocks.back());
  std::array<size_t, 3> next{};
  for (uint32_t position = 0; position < chain.blocks.size(); ++position) {
    const uint64_t block = chain.blocks[position];
    uint32_t &page = chain.pages[position];
    if (page == KvCache::noPage) {
      // Root first, so every restored block finds its parent resident. The
      // admission's reference is the request's.
      const auto form = static_cast<size_t>(restoredForm(position));
      page = taken[form][next[form]++];
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

void Cache::promoteState(const CacheLookup &lookup,
                         const std::function<std::shared_ptr<const CompositeState>()> &copy) {
  if (!lookup.state) return;
  const auto block = lookup.state->kvBlock();
  const auto *source = lookup.state->state().get();
  if (!states_.promotable(block, source)) return;
  // The copy only saves a later read, and the admitted request runs without
  // it: its room is what its claim is admitted or freed, nothing the cache
  // takes for it here.
  auto state = copy();
  if (state) states_.promote(block, source, std::move(state));
  else states_.promotionSkipped();
}

Cache::LeafReclaim Cache::demoteKv(uint64_t block) {
  if (!kvTierWritable())
    return LeafReclaim::Impossible;
  // Room in the quota or the tier that transfers in flight will free is
  // worth waiting for; room that nothing will free is not, and the leaf
  // goes.
  const auto room = [&] { return transfersInFlight() ? LeafReclaim::Pending : LeafReclaim::Impossible; };
  // Do not discard a disk copy for a transfer that cannot start yet.
  if (!tier_->canDemote(WriteClass::Reclaim)) {
    ++kvTier_.demotionsRefused;
    return room();
  }
  std::shared_ptr<KvDiskSlot> slot =
      acquireDiskSlot(model::DiskBudget::Draw::Cache, kvNeededByStateInUse(block));
  if (!slot)
    return room();
  // Making room may have taken the states the leaf was kept for; it then
  // goes like any leaf nothing needs.
  if (!kvNeededByState(block))
    return LeafReclaim::Impossible;
  auto transfer = tier_->demote(kv_.page(block), slot, WriteClass::Reclaim, completionNotifier_);
  if (!transfer) {
    ++kvTier_.demotionsRefused;
    return room();
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
  KvSlotLabel label{{block, key.parent, kv_.lastUsed(block), key.identity.lo, key.identity.hi, {}},
                    pool_.form(kv_.page(block))};
  std::copy(key.tokens.begin(), key.tokens.end(), label.block.tokens.begin());
  tier_->label(slot, encodeKvSlotLabel(label, window_.has_value()));
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
    if (!freeDiskSpace(inUse, committed) && !(committed && evictStateOnDiskLeaf()))
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

bool Cache::reusedCopy(uint64_t block, bool state) const {
  // A KV leaf's copy goes with the state it holds.
  return certainReuse(block) || (!state && wanted_.holds(block));
}

bool Cache::freeDiskSpace(bool inUse, bool active) {
  const std::optional<DiskReplacement> next = nextDiskReplacement({}, inUse, active);
  if (!next)
    return false;
  const uint64_t id = next->at.id;
  switch (next->kind) {
  case DiskVictim::KvCopy:
    // The copy of a block whose page holds its K/V is redundant; in a
    // persistent tier a spilled or disk-only leaf's copy no state needs is
    // its only K/V, and it goes with the block.
    if (const uint32_t page = kv_.page(id);
        !persistent_ || (page != KvCache::noPage && pool_.form(page) != kv::PageForm::Index))
      kv_.setSlot(id, nullptr);
    else
      kv_.erase(id);
    break;
  case DiskVictim::KvLeaf:
    states_.evict(id);
    kv_.erase(id);
    break;
  case DiskVictim::StateCopy:
  case DiskVictim::StateOnly:
  case DiskVictim::StateInUse:
    if (states_.stateResident(id))
      states_.dropDisk(id);
    else
      states_.evict(id);
    break;
  }
  return true;
}

void Cache::DiskCursor::pass(const DiskReplacement &replacement) {
  switch (replacement.kind) {
  case DiskVictim::KvCopy: kvCopy = replacement.at; break;
  case DiskVictim::StateCopy: stateCopy = replacement.at; break;
  case DiskVictim::KvLeaf:
    kvLeaf = replacement.at;
    statesGone.insert(replacement.at.id);
    break;
  case DiskVictim::StateOnly: stateOnly = replacement.at; break;
  case DiskVictim::StateInUse: stateInUse = replacement.at; break;
  }
}

std::optional<Cache::DiskReplacement> Cache::nextDiskReplacement(const DiskCursor &cursor, bool inUse,
                                                                 bool active) const {
  const auto pick = [](DiskVictim kind, const std::optional<CacheEvictionCandidate> &at) {
    return at ? std::optional<DiskReplacement>(DiskReplacement{kind, *at}) : std::nullopt;
  };
  // The only copy of what a queued request restores, or of the resume
  // point, stays but for a Spill lane's active KV: a running lane beats a
  // queued one, a copy for the cache does not.
  auto stateOnly = states_.diskCandidate(false, cursor.stateOnly);
  while (stateOnly && (cursor.statesGone.contains(stateOnly->id) || (!active && reusedCopy(stateOnly->id, true))))
    stateOnly = states_.diskCandidate(false, stateOnly);
  const auto stateCopy = states_.diskCandidate(true, cursor.stateCopy);
  // Last, and only for a copy in use: the only copy of a state in use. The
  // leaf it held is ordinary then, and goes on a later call.
  const auto stateInUse = [&] {
    return pick(DiskVictim::StateInUse, inUse ? states_.inUseDiskCandidate(cursor.stateInUse) : std::nullopt);
  };
  if (persistent_) {
    if (const auto unneeded = kv_.unneededCopy(cursor.kvCopy))
      return pick(DiskVictim::KvCopy, unneeded);
    if (stateOnly || stateCopy)
      return older(stateOnly, stateCopy) ? pick(DiskVictim::StateOnly, stateOnly)
                                         : pick(DiskVictim::StateCopy, stateCopy);
    return stateInUse();
  }
  // A redundant copy loses nothing: its data stays in RAM.
  const auto kvCopy = kv_.diskCandidate(true, cursor.kvCopy ? cursor.kvCopy->id : 0);
  if (kvCopy || stateCopy)
    return older(stateCopy, kvCopy) ? pick(DiskVictim::StateCopy, stateCopy) : pick(DiskVictim::KvCopy, kvCopy);
  // A leaf holding a state in use stays with that state, and one holding a
  // state in RAM, as a spilled leaf can, stays until reclaim takes the state
  // or a committed page evicts it (evictStateOnDiskLeaf): a reclaim makes
  // room here for that state's own write, or for a demotion beside the state
  // it picked.
  auto kvLeaf = kv_.diskCandidate(false, cursor.kvLeaf ? cursor.kvLeaf->id : 0);
  while (kvLeaf && (kvNeededByStateInUse(kvLeaf->id) || states_.stateResident(kvLeaf->id) ||
                    (!active && reusedCopy(kvLeaf->id, false))))
    kvLeaf = kv_.diskCandidate(false, kvLeaf->id);
  if (kvLeaf || stateOnly)
    return older(stateOnly, kvLeaf) ? pick(DiskVictim::StateOnly, stateOnly) : pick(DiskVictim::KvLeaf, kvLeaf);
  return stateInUse();
}

uint64_t Cache::replacedBytes(const DiskReplacement &replacement) const {
  const uint64_t id = replacement.at.id;
  const auto stateBytes = [&] {
    return states_.onDisk(id) ? model::SlotFile::slotBytesFor(states_.bytes(id)) : uint64_t{0};
  };
  switch (replacement.kind) {
  case DiskVictim::KvCopy: return tier_->slotBytes();
  case DiskVictim::KvLeaf: return tier_->slotBytes() + stateBytes();
  case DiskVictim::StateCopy:
  case DiskVictim::StateOnly:
  case DiskVictim::StateInUse: return stateBytes();
  }
  return 0;
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
  while (cachedPastLimit() && freeDiskSpace(false, false))
    ++dropped;
  return dropped;
}

void Cache::keepDiskLimit() {
  if (diskMeasure_)
    diskMeasure_();
  static_cast<void>(fitDiskLimit());
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
  // A demotion whose page frees once its children's demotions land waits
  // for them: it lands after them, here or at a later poll, so that it
  // frees its page too. The tier finishes writes in the order they started,
  // but a child's may finish after this loop passed it and before it
  // reaches the parent's.
  const std::unordered_set<uint64_t> freeing = freeingDemotions();
  for (auto entry = writes_.begin(); entry != writes_.end();) {
    if (!entry->transfer->ready() || (freeing.contains(entry->block) && !kv_.residentLeaf(entry->block))) {
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
    // committed. Its page spills once that copy lands. A writer that kept a
    // page of its own beside the block never spills that page, and owes
    // nothing.
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
      // The page stays: a reclaim then frees it through the copy, and a
      // persistent tier's next process takes the copy back.
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
  // A point's copies land over many calls: its chain is taken once, and
  // each call goes on from the first block whose copy has not landed, so
  // that a call costs what is left to copy rather than the chain's depth
  // (4000 blocks at 128K tokens).
  const auto settled = [this](PersistStatus status) {
    if (status != PersistStatus::Started && status != PersistStatus::Busy)
      persisting_ = {};
    return status;
  };
  if (!newestPoint(block))
    return settled(PersistStatus::Unneeded);
  if (!kvTierWritable())
    return settled(PersistStatus::Refused);
  if (persisting_.block != block)
    persisting_ = {block, kv_.chain(block).blocks, 0};
  // Waiting for room would hold back the points after this one.
  if (!pointFits(persisting_))
    return settled(PersistStatus::Refused);
  // Nothing starts while the tier serves what is needed now: a restore that
  // waits, a reclaim's demotion and a Spill request's write through go
  // first.
  const bool yields = !restores_.empty() || writes(WriteClass::Reclaim) || writes(WriteClass::Spill);
  if (yields && !states_.onDisk(block))
    return settled(PersistStatus::Busy);
  // The state goes first, as the write budget admits it; its chain follows.
  // Only a persistent tier makes room for either.
  const PersistStatus state = states_.persist(block, persistent_);
  if (state != PersistStatus::Durable && state != PersistStatus::Started)
    return settled(state);
  bool inFlight = state == PersistStatus::Started;
  const std::vector<uint64_t> &blocks = persisting_.blocks;
  // A temporary tier may give up a copy below the frontier meanwhile, so a
  // call that finds nothing in flight walks the whole chain once more.
  for (size_t from = persisting_.landed;; from = 0) {
    for (size_t index = from; index < blocks.size(); ++index) {
      const uint64_t id = blocks[index];
      if (kv_.slot(id)) {
        if (kv_.transfer(id) != KvCache::Transfer::None)
          inFlight = true;
        else if (index == persisting_.landed)
          ++persisting_.landed;
        continue;
      }
      persisting_.landed = std::min(persisting_.landed, index);
      // Only a resident block lacks a disk copy.
      if (yields || !tier_->canDemote(WriteClass::Persist))
        return settled(inFlight ? PersistStatus::Started : PersistStatus::Busy);
      std::shared_ptr<KvDiskSlot> slot = persistent_
                                             ? acquireDiskSlot(model::DiskBudget::Draw::Cache, false)
                                             : tier_->acquireSlot(model::DiskBudget::Draw::Cache);
      // Making room gives up the oldest point, which may be this one: it
      // does not fit then. Otherwise its chain stays whole, as a state needs
      // it. A temporary tier, which makes none, keeps what it copied.
      if (!slot || !states_.onDisk(block))
        return settled(PersistStatus::Refused);
      auto transfer = tier_->demote(kv_.page(id), slot, WriteClass::Persist, completionNotifier_);
      if (!transfer)
        return settled(inFlight ? PersistStatus::Started : PersistStatus::Busy);
      labelKv(id, slot);
      kv_.setSlot(id, std::move(slot));
      kv_.setTransfer(id, KvCache::Transfer::Draining);
      writes_.push_back({id, WriteClass::Persist, std::move(transfer)});
      inFlight = true;
    }
    if (inFlight || !from)
      break;
  }
  return settled(inFlight ? PersistStatus::Started : PersistStatus::Durable);
}

bool Cache::newestPoint(uint64_t block) const {
  return kv_.contains(block) && states_.ordinary(block) && !superseded(block);
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

bool Cache::pointFits(const PersistCursor &point) const {
  if (!diskBudget_)
    return true;
  const uint64_t limit = diskBudget_->limitBytes();
  if (persistent_)
    return uint64_t{point.blocks.size()} * tier_->slotBytes() + states_.bytes(point.block) <= limit;
  uint64_t lacking = states_.onDisk(point.block) ? 0 : states_.bytes(point.block);
  for (const uint64_t id : point.blocks)
    if (!kv_.slot(id))
      lacking += tier_->slotBytes();
  const uint64_t taken = diskBudget_->usedBytes() + diskBudget_->committedBytes();
  return taken <= limit && lacking <= limit - taken;
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
  std::optional<KvWindowSnapshot> window;
  if (window_)
    window = KvWindowSnapshot{.windowPages = window_->windowPages,
                              .trim = window_->trim,
                              .pinnedBlocks = kv_.pinnedBlocks(),
                              .trimmedPages = trimmedPages_,
                              .upgrades = kv_.upgrades(),
                              .refusedStates = refusedStates_};
  return {pool_.snapshot(),
          states_.snapshot(),
          tier,
          lookup_,
          static_cast<uint32_t>(requests_.size()),
          extentCompactMaxMilliseconds_,
          persistent_,
          adoption_,
          window};
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
