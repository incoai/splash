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
  // A copy that lands, leaves or turns out another command's changes the
  // copies each request holding its block on that page counts
  // (Request::heldCopies). A block sits at its depth in each holder's chain;
  // a request's page and its block's move together (followMoves).
  if (spill_)
    kv_.onCopyChanged([this](uint64_t block, uint32_t depth, uint32_t page, bool copied) {
      for (auto &[_, holder] : requests_)
        if (depth <= holder.cachedBlocks.size() && holder.cachedBlocks[depth - 1] == block &&
            holder.pages[depth - 1] == page)
          copied ? ++holder.heldCopies : --holder.heldCopies;
    });
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

std::span<const uint64_t> CacheProbe::restoredChain() const noexcept {
  return matchableChain(std::span<const uint64_t>(blocks_).first(cachedTokens_ / KvCache::pageTokens));
}

std::span<const uint64_t> CacheProbe::matchableChain(std::span<const uint64_t> chain) const noexcept {
  if (kvGeneration_ == owner_->kv_.generation())
    return chain;
  const auto end = std::ranges::find_if(chain, [&](uint64_t block) { return !owner_->kv_.matchable(block); });
  return chain.first(static_cast<size_t>(end - chain.begin()));
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
    // See KvCache's generation.
    probe.blocks_.resize(probe.matchableChain(probe.blocks_).size());
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
    if (spill_)
      active.heldCopies += holdsCopy(active, logical);
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
  if (!acquired.granted())
    return {TokenAdmissionFailure::Denied, count, pool_.freePageCount(form), acquired.failure, form};
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
  if (spill_ && spill_->always && writes(WriteClass::Spill)) {
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
  Need need{.bytes = (beyond + extent - 1) / extent * extent * pool_.bytesPerPage(full), .kv = true};
  need.keptPages[static_cast<size_t>(full)] = std::min(pages, free);
  return need;
}

uint64_t Cache::growthPages(uint64_t bytes) const {
  constexpr kv::PageForm full = kv::PageForm::Full;
  const uint64_t extent = pool_.extentPages();
  return pool_.freePageCount(full) + bytes / (extent * pool_.bytesPerPage(full)) * extent;
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
  using Reuse = TakeRights::CertainReuse;
  using Reserves = TakeRights::Reserves;
  const TakeRights &rights = query.rights;
  const Need &need = query.need;
  Walk walk;
  // The walk leaves out what the claim may not take (takeRights): a device
  // or a page need's reclaims take many victims, so nothing is marked. Every
  // walk keeps the chain a start's placement reuses (VictimQuery::kept), and
  // the Index runway (Walk::keeps), where landed pages move without an
  // allocation (moveLandedPages), a lane's release among them.
  walk.scope.upTo = rights.inUse ? ReclaimClass::InUse : ReclaimClass::Ordinary;
  walk.keepResumePoint = rights.certainReuse == Reuse::None;
  walk.restored = rights.certainReuse == Reuse::All ? nullptr : &wanted_;
  walk.reused = query.kept;
  walk.timing = rights.waitsForWrites ? ReclaimTiming::CanWait : ReclaimTiming::Immediate;
  walk.keepRunway = rights.reserves == Reserves::None;
  walk.hostRefused = need.hostRefused;
  // A state that goes ends a unit for a need of its buffers, and of bytes,
  // which its buffers meet once they leave the model's pool.
  walk.statesAreUnits = need.lacksStateBuffers() || (need.bytes && !need.hostRefused);
  if (need.bytes && !need.hostRefused) {
    // Memory the engine's limit refused, whatever pages it was for: an
    // extent released covers it.
    walk.device = true;
    walk.inPlace = need.pages != 0;
    walk.refused = Shortfall{pool_.extentPages(), kv::PageForm::Full};
    // A KV need takes the free pages of its form first, and those of the
    // other forms its attempt takes: releasing their extent would give it
    // bytes for an extent it then allocates again. It keeps too those the
    // other sources' units move pages into (VictimQuery::movedInto), which
    // they count on.
    if (need.pages || std::ranges::any_of(need.keptPages, [](uint32_t each) { return each != 0; })) {
      walk.form = need.form;
      for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index}) {
        const auto each = static_cast<size_t>(form);
        const uint64_t takes =
            uint64_t{form == need.form && need.pages ? pool_.freePageCount(form) : need.keptPages[each]} +
            query.movedInto[each];
        walk.keptPages[each] = static_cast<uint32_t>(std::min<uint64_t>(pool_.freePageCount(form), takes));
      }
    }
  } else if (need.pages) {
    // Pages in place; a host that refuses growth takes them in a page scan.
    walk.pages = need.pages;
    walk.form = need.form;
    walk.scope.pagesOnly = need.hostRefused;
    walk.refused = Shortfall{need.pages, need.form};
  } else if (need.lacksStateBuffers() || need.hostRefused) {
    walk.scope.statesOnly = true;
    walk.refused = Shortfall{};
  }
  return walk;
}

std::optional<Cache::WalkUnit> Cache::nextUnit(const Walk &walk, WalkCursor &cursor) const {
  using Phase = WalkCursor::Phase;
  const WalkScope &scope = walk.scope;
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
        cursor.state = states_.checkpointCandidate();
        cursor.started = true;
      } else if (last == Unit::Checkpoint) {
        cursor.state = states_.checkpointCandidate(cursor.lastCandidate);
      }
      if (cursor.state)
        return give(Unit::Checkpoint, *cursor.state);
      moveOn(scope.statesOnly ? Phase::Ordinary : Phase::DeadLeaves);
      continue;
    case Phase::DeadLeaves:
      // The leaves no state restores through: they save no prefill, and
      // free their page with no IO.
      if (!cursor.started) {
        cursor.leaf = kv_.leafCandidate({.dead = true}, 0);
        cursor.started = true;
      } else if (last == Unit::DeadLeaf) {
        cursor.leaf = kv_.leafCandidate({.dead = true}, cursor.lastCandidate.id);
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
        return inUse ? states_.inUseCandidate(after) : states_.ordinaryCandidate(after);
      };
      // The leaves of the class a state restores through, those with the
      // state in RAM in a page scan alone (the others take the state first),
      // and of the rest only those with a copy while the walk cannot wait
      // for a demotion (reclaimKvLeaf). A dead leaf goes, or stays, with the
      // dead leaves.
      const KvCache::LeafWalk leaves{
          .inUse = inUse,
          .withStateInRam = scope.pagesOnly,
          .copiedOnly = kvTierWritable() && (walk.timing == ReclaimTiming::Immediate || cursor.demotionsHeld)};
      if (!cursor.started) {
        // A page scan takes a state only where it sits on the leaf that goes
        // next.
        if (!scope.pagesOnly)
          cursor.state = stateAfter(std::nullopt);
        if (!scope.statesOnly)
          cursor.leaf = kv_.leafCandidate(leaves, 0);
        cursor.started = true;
      } else if (last == state) {
        cursor.state = stateAfter(cursor.lastCandidate);
      } else if (last == leaf || last == leafState) {
        cursor.leaf = kv_.leafCandidate(leaves, cursor.lastCandidate.id);
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
  for (std::optional<WalkUnit> unit; (unit = nextUnit(walk, cursor));) {
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

void Cache::holdAfter(const Walk &walk, WalkCursor &cursor, uint64_t staged) const {
  using Phase = WalkCursor::Phase;
  const auto waits = [&](uint64_t block) {
    return stateReclaim(walk, block, staged).when == StateReclaim::When::Later;
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
  for (std::optional<CacheEvictionCandidate> state = inUse ? states_.inUseCandidate() : states_.ordinaryCandidate();
       state && (state->lastUsed < at.lastUsed || (state->lastUsed == at.lastUsed && (!atState || state->id <= at.id)));
       state = inUse ? states_.inUseCandidate(state) : states_.ordinaryCandidate(state)) {
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

Cache::StateReclaim Cache::stateReclaim(const Walk &walk, uint64_t block, uint64_t staged) const {
  using When = StateReclaim::When;
  const bool write = states_.reclaimWrites(block);
  const bool busy = states_.writing() || (staged && staged != block);
  if (write && (!walk.writes || (walk.timing == ReclaimTiming::Immediate && busy)))
    return {When::PassedOver, true};
  // As reclaim() goes: a write waits for the staging buffer, a copy ahead
  // keeps its state until it lands.
  const bool later = write ? busy : states_.reclaimOutcome(block).pending;
  return {later ? When::Later : When::Now, write || later};
}

std::vector<uint64_t> Cache::keptStates(const Walk &walk) const {
  std::vector<uint64_t> kept;
  if (const uint64_t point = walk.keepResumePoint ? states_.resumePoint() : 0)
    kept.push_back(point);
  for (const RestoredChains *chains : {walk.restored, walk.reused})
    if (chains)
      kept.insert(kept.end(), chains->states().begin(), chains->states().end());
  return kept;
}

KvCache::Keeps Cache::keptChains(const Walk &walk, ReclaimClass upTo) noexcept {
  using Keep = KvCache::Keep;
  KvCache::Keeps chains = 0;
  if (upTo != ReclaimClass::InUse)
    chains |= KvCache::keepBit(Keep::Used);
  if (walk.restored)
    chains |= KvCache::keepBit(Keep::Wanted);
  if (walk.keepResumePoint)
    chains |= KvCache::keepBit(Keep::ResumePoint);
  return chains;
}

uint64_t Cache::stagedState(const Walk &walk) const {
  return states_.reclaims(walk.scope.upTo == ReclaimClass::InUse, keptStates(walk)).staged;
}

StateEviction Cache::reclaimState(const Walk &walk, uint64_t block) {
  if (stateReclaim(walk, block).when == StateReclaim::When::PassedOver)
    return {};
  return states_.reclaim(block);
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
    switch (reclaimKvLeaf(walk, block, timing)) {
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
  if (walk.refused ? returnsInFlight(walk, *walk.refused) : transfersInFlight(walk))
    return true;
  // An extent's worth of free pages that compaction cannot gather while a
  // write in flight reads a page among them returns, for memory, as it
  // lands.
  return walk.device && std::ranges::any_of(writes_, [&](const Write &write) {
           return pool_.freePageCount(pool_.form(kv_.page(write.block))) >= pool_.extentPages();
         });
}

namespace {

// Victim::key: the unit above the block's id, or above the phase of a
// device or a page need's reclaims (ReclaimPhase).
constexpr unsigned kUnitShift = 56;

uint64_t unitKey(uint8_t unit, uint64_t block) noexcept { return uint64_t{unit} << kUnitShift | block; }
uint8_t unitOfKey(uint64_t key) noexcept { return static_cast<uint8_t>(key >> kUnitShift); }
uint64_t blockOfKey(uint64_t key) noexcept { return key & ((uint64_t{1} << kUnitShift) - 1); }

} // namespace

std::optional<engine::Victim> Cache::victimOf(const Walk &walk, const WalkUnit &unit, uint64_t staged) const {
  const uint64_t block = unit.candidate.id;
  const StateReclaim reclaim = stateReclaim(walk, block, staged);
  const bool now = reclaim.when == StateReclaim::When::Now;
  return engine::Victim{.key = unitKey(static_cast<uint8_t>(unit.unit), block),
                        .stateBuffers = now ? 1u : 0u,
                        .write = reclaim.write,
                        .later = !now,
                        .cls = unit.unit == Unit::UsedState ? ReclaimClass::InUse : ReclaimClass::Ordinary};
}

bool Cache::trims(const Walk &walk) const noexcept {
  return walk.pages && window_ && window_->trim && walk.form == kv::PageForm::Full;
}

uint32_t Cache::landingExtents(const Walk &walk, kv::PageForm form, uint32_t landing) const noexcept {
  return completedExtents(pool_.freePageCount(form), landing, pool_.extentPages(), walk.keeps(form));
}

uint64_t Cache::landingFill(const Walk &walk) const noexcept {
  if (walk.timing == ReclaimTiming::Immediate)
    return 0;
  const std::array<uint32_t, 3> landing = demotedPages();
  uint64_t bytes = 0;
  for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index})
    bytes += uint64_t{landingExtents(walk, form, landing[static_cast<size_t>(form)])} * pool_.extentPages() *
             pool_.bytesPerPage(form);
  return bytes;
}

const Cache::ReclaimReach::Amount &Cache::ReclaimReach::of(ReclaimPhase phase) const noexcept {
  switch (phase) {
  case ReclaimPhase::Now:
  case ReclaimPhase::UsedNow:
    return now;
  case ReclaimPhase::Written:
  case ReclaimPhase::UsedWritten:
    return written;
  case ReclaimPhase::Later:
  case ReclaimPhase::UsedLater:
    return later;
  }
  return later;
}

Cache::Walk Cache::phaseWalk(const Walk &walk, ReclaimPhase phase) const {
  using enum ReclaimPhase;
  Walk each = walk;
  if (phase < UsedNow)
    each.scope.upTo = ReclaimClass::Ordinary;
  each.timing = phase == Later || phase == UsedLater ? ReclaimTiming::CanWait : ReclaimTiming::Immediate;
  each.writes = phase != Now && phase != UsedNow;
  return each;
}

Cache::ReclaimReach Cache::reach(const Walk &walk, ReclaimClass upTo) const {
  using Keep = KvCache::Keep;
  const uint32_t extent = pool_.extentPages();
  const bool inUse = upTo == ReclaimClass::InUse;
  // The states the walk keeps beside those in use, which it passes over:
  // the resume point and the states the chains it keeps restore from.
  const std::vector<uint64_t> kept = keptStates(walk);
  // The states the walk takes free now when they need no write; the first
  // that needs one while no write holds the staging buffer frees at once
  // (written); the others wait for that buffer.
  const StateCache::Reclaims states = states_.reclaims(inUse, kept);
  ReclaimReach counted;
  counted.now.states = states.now;
  counted.written.states = states.staged ? 1 : 0;
  counted.later.states = states.later;
  // What holds a page and its chain above it in place, which frees later: a
  // block in use or in transfer, a page a request holds; a state that needs
  // a write, which stays in RAM until its write starts and needs its chain
  // on the SSD after; with a writable tier, a block without a copy that a
  // state that stays needs; and the chains the walk keeps, which the KV
  // cache counts as they change (keptChains), but for the chain a start
  // reuses, which it walks (KvCache::pageReach). Once the staged state's
  // write has started, its chain holds only its blocks without a copy.
  KvCache::Keeps mask = KvCache::keepBit(Keep::Busy) | KvCache::keepBit(Keep::Moving) | keptChains(walk, upTo);
  if (kvTierWritable())
    mask |= KvCache::keepBit(Keep::Uncopied);
  if (states.writes)
    mask |= KvCache::keepBit(Keep::Writer);
  if (states.usedWrites)
    mask |= KvCache::keepBit(Keep::UsedWriter);
  if (walk.restored && walk.restored != &wanted_)
    throw std::logic_error("a walk restores chains the cache does not count");
  const std::span<const uint64_t> reused = walk.reused ? walk.reused->states() : std::span<const uint64_t>{};
  // A write in flight frees what it holds once it lands only where a take
  // waits for it (returns): a demotion, a copy whose block a reclaim of
  // the class then frees through it (copyAheadReturns), under --kv-spill
  // always a write through whose page spills as it lands. Above any other,
  // no reclaim reaches a page until it has landed. The extents that hold a
  // page a transfer reads or fills, which compaction does not move, empty
  // only then too.
  std::vector<uint64_t> unwaited;
  std::array<uint32_t, 3> pinned{};
  {
    std::unordered_set<uint32_t> extentsPinned;
    const auto pin = [&](uint64_t block) {
      if (const uint32_t page = kv_.page(block);
          page != KvCache::noPage && extentsPinned.insert(page / pool_.extentPages()).second)
        ++pinned[static_cast<size_t>(pool_.form(page))];
    };
    for (const Write &write : writes_) {
      pin(write.block);
      if (write.write != WriteClass::Reclaim && !copyAheadReturns(write, walk, upTo) &&
          !(spill_ && spillsOnLanding(write)))
        unwaited.push_back(write.block);
    }
    for (const auto &[block, _] : restores_)
      pin(block);
  }
  const KvCache::PageReach pages = kv_.pageReach(mask, reused, unwaited, states.staged);

  // A page scan takes a state only where it sits on the leaf that goes next
  // (Cache::nextUnit): of those that free now, the RAM copies at a block that
  // frees now.
  if (walk.scope.pagesOnly)
    counted.now.states = std::min(counted.now.states, pages.candidates);
  // The pages landing as their demotions complete (demotedPages). A walk
  // that can wait counts what they free with the rest (Cache::take credits
  // it, landingFill): a demotion frees its page as it lands once nothing in
  // RAM holds the block (freesOnLanding), so a phase that takes the state
  // on such a block and what frees now below it lets it land too, each
  // demotion judged on those started before it, as freeingDemotions judges
  // them. Those of the first phase, then once the staged state's write has
  // started.
  const std::array<uint32_t, 3> landing = demotedPages();
  const auto landingOnce = [&](bool stagedGone) {
    std::array<uint32_t, 3> pages = landing;
    if (walk.timing != ReclaimTiming::CanWait)
      return pages;
    std::unordered_set<uint64_t> freeing = freeingDemotions();
    const Walk first = phaseWalk(walk, inUse ? ReclaimPhase::UsedNow : ReclaimPhase::Now);
    const auto stateGoes = [&](uint64_t block) {
      return !states_.stateResident(block) || (stagedGone && block == states.staged) ||
             (states_.reclaimable(block) && (inUse || !states_.inUse(block)) &&
              std::ranges::find(kept, block) == kept.end() &&
              stateReclaim(first, block).when == StateReclaim::When::Now);
    };
    for (const Write &write : writes_) {
      const uint64_t block = write.block;
      if (write.write != WriteClass::Reclaim || freeing.contains(block) || !kv_.leafOnceChildrenLeave(block) ||
          pool_.pageActive(kv_.page(block)) || !stateGoes(block) ||
          !std::ranges::all_of(kv_.children(block), [&](uint64_t child) {
            return kv_.page(child) == KvCache::noPage || freeing.contains(child) || !(kv_.keeps(child) & mask);
          }))
        continue;
      freeing.insert(block);
      ++pages[static_cast<size_t>(pool_.form(kv_.page(block)))];
    }
    return pages;
  };
  const std::array<uint32_t, 3> landingNow = landingOnce(false);
  const std::array<uint32_t, 3> landingWritten = states.staged ? landingOnce(true) : landingNow;
  // A page walk that trims also frees the Full pages it moves into Slim
  // ones, as many as the Slim pages free or freed allow: it trims after a
  // state goes (Cache::take), but for the staged state's write, the pages
  // the windows passed and the tails of the states gone, each at most
  // windowPages blocks.
  const uint32_t passed = trims(walk) ? passedPages() : 0;
  const auto trimmable = [&](uint32_t gone) -> uint32_t {
    if (!gone || !trims(walk))
      return 0;
    return passed +
           static_cast<uint32_t>(std::min<uint64_t>(kv_.pinnedBlocks(), uint64_t{gone} * window_->windowPages));
  };
  // Of each form, the pages a reclaim of the class reaches (the free ones
  // and those reachable), that free now (the free ones and those nothing
  // holds), once the staged state's write has started (the written ones
  // too) or later (the rest).
  std::array<uint32_t, 3> now{}, written{}, all{};
  for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index}) {
    const auto index = static_cast<size_t>(form);
    const uint32_t allocated = pool_.allocatedExtents(form) * extent;
    all[index] = std::min(allocated, pool_.freePageCount(form) + pages.reachable[index]);
    now[index] = std::min(all[index], pool_.freePageCount(form) + pages.now[index]);
    written[index] = std::min(all[index], pool_.freePageCount(form) + pages.now[index] + pages.written[index]);
    // Compaction gathers the pages that stay into the fewest extents, and
    // the rest empty, but for the free pages a KV need takes and the
    // runway the walk keeps. A need in place empties none of its form: each
    // take ends once it has a page more there (takeReclaim), which the next
    // keeps with the other free pages.
    const uint64_t extentBytes = uint64_t{extent} * pool_.bytesPerPage(form);
    const auto extents = [&](uint32_t pages) {
      if (walk.inPlace && form == walk.form)
        return 0u;
      const uint32_t whole = (pages - std::min(pages, walk.keptPages[index])) / extent;
      return whole - std::min(whole, walk.keeps(form) ? 1u : 0u);
    };
    // Until its transfer lands, an extent that holds a pinned page stays.
    const auto atOnce = [&](uint32_t pages) {
      const uint32_t movable = pool_.allocatedExtents(form) - std::min(pool_.allocatedExtents(form), pinned[index]);
      return std::min(extents(pages), movable - std::min(movable, walk.keeps(form) ? 1u : 0u));
    };
    // The extents the landing pages complete beside the free ones count
    // already (returning, landingExtents). Through a phase, with those: the
    // extents its takes release, or for a walk that can wait, if more, the
    // extents its takes credit, released or completed by landing pages
    // beside the pages then free (Cache::take): those the free, freed and
    // landing pages fill together, but no more than its releases and one
    // per extent's worth of landing pages begun (a take's free pages may
    // complete one beside landing pages that the next releases).
    const uint32_t landed = landingExtents(walk, form, landing[index]);
    const auto through = [&](uint32_t pages, uint32_t lands) -> uint32_t {
      const uint32_t released = atOnce(pages) + landed;
      if (walk.timing != ReclaimTiming::CanWait)
        return released;
      return std::max(released, std::min(completedExtents(0, pages + lands, extent, walk.keeps(form)),
                                         atOnce(pages) + (lands + extent - 1) / extent));
    };
    const uint32_t throughNow = through(now[index], landingNow[index]);
    const uint32_t throughWritten = std::max(throughNow, through(written[index], landingWritten[index]));
    counted.now.bytes += (throughNow - std::min(throughNow, landed)) * extentBytes;
    counted.written.bytes += (throughWritten - throughNow) * extentBytes;
    counted.later.bytes += (extents(all[index]) - std::min(extents(all[index]), throughWritten)) * extentBytes;
  }
  const auto need = static_cast<size_t>(walk.form);
  uint32_t pagesNow = now[need], pagesWritten = written[need], pagesAll = all[need];
  if (trims(walk)) {
    const auto slim = static_cast<size_t>(kv::PageForm::Slim);
    const uint32_t allocated = pool_.allocatedExtents(kv::PageForm::Full) * extent;
    const uint32_t goneNow = counted.now.states,
                   goneAll = goneNow + counted.written.states + counted.later.states;
    pagesNow = std::min(allocated, pagesNow + std::min(trimmable(goneNow), now[slim]));
    pagesWritten = std::min(allocated, pagesWritten + std::min(trimmable(goneNow), written[slim]));
    pagesAll = std::min(allocated, pagesAll + std::min(trimmable(goneAll), all[slim]));
  }
  const uint32_t free = pool_.freePageCount(walk.form);
  counted.now.pages = pagesNow - std::min(pagesNow, free);
  counted.written.pages = pagesWritten - std::min(pagesWritten, std::max(pagesNow, free));
  counted.later.pages = pagesAll - std::min(pagesAll, std::max(pagesWritten, free) + landing[need]);
  return counted;
}

std::optional<engine::Victim> Cache::reclaimUnit(const Walk &walk, ReclaimPhase phase, const ReclaimReach &ordinary,
                                                 const ReclaimReach *inUse) const {
  const bool used = phase >= ReclaimPhase::UsedNow;
  const Walk each = phaseWalk(walk, phase);
  ReclaimReach::Amount unit = ordinary.of(phase);
  if (used) {
    // The staging buffer writes an ordinary state before one in use.
    if (phase == ReclaimPhase::UsedWritten && unit.states)
      return std::nullopt;
    const ReclaimReach::Amount &all = inUse->of(phase);
    unit = {.bytes = all.bytes - std::min(all.bytes, unit.bytes),
            .pages = all.pages - std::min(all.pages, unit.pages),
            .states = all.states - std::min(all.states, unit.states)};
  }
  // A page need counts pages alone; a device need pages only in place,
  // where its take frees them; states only where they are units.
  if (!walk.device)
    unit.bytes = 0;
  if (walk.device && !walk.inPlace)
    unit.pages = 0;
  if (!walk.statesAreUnits)
    unit.states = 0;
  if (!unit.bytes && !unit.pages && !unit.states)
    return std::nullopt;
  return engine::Victim{.key = unitKey(static_cast<uint8_t>(Unit::Reclaim), static_cast<uint64_t>(phase)),
                        .bytes = unit.bytes,
                        .pages = unit.pages,
                        .form = walk.form,
                        .stateBuffers = unit.states,
                        .write = each.writes,
                        .later = each.timing == ReclaimTiming::CanWait,
                        .cls = used ? ReclaimClass::InUse : ReclaimClass::Ordinary};
}

std::optional<engine::Victim> Cache::peek(const VictimQuery &query, const engine::Victim *after) const {
  const Walk walk = walkFor(query);
  const Unit afterUnit = after ? static_cast<Unit>(unitOfKey(after->key)) : Unit{};
  if (walk.device || walk.pages) {
    // A page need's free pages first: the other forms' empty extents, then
    // a windowed cache's trims for Full pages.
    if (walk.pages && !after && (spill_ || window_)) {
      // The empty extents of the other forms beyond the runway each keeps.
      uint64_t bytes = 0;
      for (const kv::PageForm other : {kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index})
        if (const uint32_t count = pool_.emptyExtents(other); other != walk.form && count > 1)
          bytes += uint64_t{count - 1} * pool_.extentPages() * pool_.bytesPerPage(other);
      if (bytes)
        return engine::Victim{.key = unitKey(static_cast<uint8_t>(Unit::Extent), 0), .bytes = bytes};
    }
    if (walk.pages && (!after || afterUnit == Unit::Extent))
      if (const uint32_t pages = trims(walk) ? std::min(passedPages(), pool_.freePageCount(kv::PageForm::Slim)) : 0)
        return engine::Victim{.key = unitKey(static_cast<uint8_t>(Unit::Trim), 0), .pages = pages};
    // Then the reclaims, phase by phase, of what the walk's reclaims reach,
    // counted once: of the ordinary class, and from the first phase of what
    // is in use, for a walk that takes it, of that class too.
    auto phase = static_cast<uint8_t>(ReclaimPhase::Now);
    if (after && afterUnit == Unit::Reclaim)
      phase = static_cast<uint8_t>(blockOfKey(after->key) + 1);
    std::optional<ReclaimReach> ordinary, inUse;
    for (; phase <= static_cast<uint8_t>(ReclaimPhase::UsedLater); ++phase) {
      const auto each = static_cast<ReclaimPhase>(phase);
      const bool used = each >= ReclaimPhase::UsedNow;
      if (used && walk.scope.upTo != ReclaimClass::InUse)
        break;
      if (!ordinary)
        ordinary = reach(walk, ReclaimClass::Ordinary);
      if (used && !inUse)
        inUse = reach(walk, ReclaimClass::InUse);
      if (std::optional<engine::Victim> unit = reclaimUnit(walk, each, *ordinary, used ? &*inUse : nullptr))
        return unit;
    }
    return std::nullopt;
  }
  // A state need's units, a state each, counted in the walk's order: the
  // staging buffer takes the first that needs a write.
  const uint64_t staged = stagedState(walk);
  WalkCursor cursor;
  if (after) {
    cursor = cursorAfter(afterUnit, blockOfKey(after->key));
    holdAfter(walk, cursor, staged);
  }
  for (std::optional<WalkUnit> unit; (unit = nextStepUnit(walk, cursor));)
    if (std::optional<engine::Victim> victim = victimOf(walk, *unit, staged))
      return victim;
  return std::nullopt;
}

FreeResult Cache::take(const VictimQuery &query, const engine::Victim &victim) {
  const Walk walk = walkFor(query);
  const uint64_t block = blockOfKey(victim.key);
  const auto met = [](uint64_t nowBytes = 0) { return FreeResult{FreeOutcome::Met, nowBytes}; };
  switch (static_cast<Unit>(unitOfKey(victim.key))) {
  case Unit::Extent:
    if (const uint64_t released = releaseOtherForms(walk.form))
      return met(released);
    return {};
  case Unit::Trim:
    // Into free Slim pages alone, as peek counted them: the ledger takes
    // outside any allocation's scope, and the Slim runway refills where the
    // engine trims in one (Engine::prepare).
    if (trimWindows(false))
      return {.outcome = FreeOutcome::Met, .pages = victim.pages};
    return {};
  case Unit::Reclaim: {
    // One reclaim of the unit's phase: the state the staging buffer writes
    // (recycleState), else a device need's memory reclaim (takeReclaim), a
    // page need's eviction, and in a windowed cache the trims of what a
    // state that left unpinned. What it freed now, and the extents and pages
    // its demotions free as they land.
    const auto phase = static_cast<ReclaimPhase>(block);
    const bool written = phase == ReclaimPhase::Written || phase == ReclaimPhase::UsedWritten;
    const Walk each = phaseWalk(walk, phase);
    const auto form = static_cast<size_t>(walk.form);
    // The landing pages of the need's form, for a walk that can wait, and
    // the free ones with them.
    const auto landingPages = [&] {
      return each.timing == ReclaimTiming::CanWait ? demotedPages()[form] : 0u;
    };
    const auto pages = [&] { return pool_.freePageCount(walk.form) + landingPages(); };
    const uint32_t landing = landingPages();
    const uint32_t pagesBefore = pages();
    const uint64_t fillBefore = walk.device ? landingFill(walk) : 0;
    const ReclaimResult reclaimed =
        written       ? recycleState(each)
        : walk.device ? takeReclaim(each, demotionPages(), fillBefore)
                      : evictOne(each);
    if (!walk.device && !written && reclaimed.evictedState && trims(walk))
      static_cast<void>(trimWindows(false));
    if (!reclaimed.madeProgress && !reclaimed.evictedKv)
      return {reclaimed.pending ? FreeOutcome::Pending : FreeOutcome::Short};
    const uint32_t pagesAfter = pages();
    // An extent released of free pages that landing pages would complete
    // counts as landing already, for a claim that can wait: in what it
    // waits for, or in a take before. It is not counted again.
    const uint64_t filled = reclaimed.reclaimedBytes + (walk.device ? landingFill(walk) : 0);
    const uint64_t gave = filled - std::min(filled, fillBefore);
    const uint64_t now = std::min(reclaimed.reclaimedBytes, gave);
    return {.outcome = gave > now || landingPages() > landing ? FreeOutcome::Pending : FreeOutcome::Met,
            .nowBytes = now,
            .laterBytes = gave - now,
            .pages = pagesAfter - std::min(pagesAfter, pagesBefore),
            .stateBuffers = reclaimed.evictedState ? 1u : 0u};
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
    return {.outcome = FreeOutcome::Met, .stateBuffers = 1};
  return {taken.pending ? FreeOutcome::Pending : FreeOutcome::Short};
}

void Cache::demoteChainsAbove(const Walk &walk, size_t firstWrite, const std::function<bool()> &enough) {
  if (walk.timing == ReclaimTiming::Immediate)
    return;
  std::unordered_set<uint64_t> freeing = freeingDemotions();
  const size_t started = writes_.size();
  for (size_t index = firstWrite; index < started; ++index) {
    if (writes_[index].write != WriteClass::Reclaim || !freeing.contains(writes_[index].block))
      continue;
    for (uint64_t above = kv_.key(writes_[index].block).parent; above; above = kv_.key(above).parent) {
      if (!kv_.contains(above) || kv_.transfer(above) != KvCache::Transfer::None || kv_.slot(above) ||
          !kvNeededByState(above) || walk.keepsBlock(above) ||
          (walk.scope.upTo == ReclaimClass::Ordinary && kvNeededByStateInUse(above)) ||
          !freesOnLanding(above, freeing))
        break;
      if (enough() || demoteKv(walk, above) != LeafReclaim::Started)
        return;
      freeing.insert(above);
    }
  }
}

void Cache::setWantedChains(std::vector<std::span<const uint64_t>> chains) {
  RestoredChains wanted;
  for (const std::span<const uint64_t> chain : chains)
    wanted.add(chain);
  // The KV cache counts the pages the chains keep: the chains that begin,
  // then those that end, so that a chain that stays changes nothing. A
  // block that left took its count with it.
  for (const uint64_t last : wanted.states())
    if (kv_.contains(last))
      kv_.countWanted(last, true);
  for (const uint64_t last : wanted_.states())
    if (kv_.contains(last))
      kv_.countWanted(last, false);
  wanted_ = std::move(wanted);
}

// Reclaim: memory returns as whole extents. Free pages go first, an empty
// extent as it is and the pages scattered over the others once they fill
// one; then one victim at a time: KV no state restores through, then the
// shared recency order, states and resident KV leaves alike. The steps
// below take the walk's units, as peek gives them and take takes them.

uint64_t Cache::releaseEmptyExtents(bool keepRunway) {
  return pool_.reclaimEmptyExtents(keepRunway, std::numeric_limits<uint32_t>::max());
}

Cache::ReclaimResult Cache::evictOne(const Walk &walk, std::optional<std::array<bool, 3>> reach) {
  if (const std::optional<Victim> victim = takeOldest(walk, reach))
    return {.madeProgress = true, .evictedState = !victim->kv};
  // Nothing to reclaim now. Transfers land only in pollTransfers(), so what
  // was in flight during the pass still is, and comes back.
  return {.pending = returns(walk)};
}

std::optional<Cache::Victim> Cache::takeOldest(const Walk &walk, std::optional<std::array<bool, 3>> reach) {
  using Phase = WalkCursor::Phase;
  WalkCursor cursor;
  while (const std::optional<WalkUnit> unit = nextStepUnit(walk, cursor)) {
    const bool leaf = unit->unit == Unit::DeadLeaf || unit->unit == Unit::Leaf || unit->unit == Unit::UsedLeaf;
    if (leaf && reach && !leadsToReach(unit->candidate.id, *reach))
      continue;
    // The leaves of each class wait for the tier on their own: a new phase
    // starts with a fresh cursor.
    const UnitTaken taken =
        takeUnit(walk, *unit, cursor.demotionsHeld ? ReclaimTiming::Immediate : walk.timing);
    if (taken.taken)
      return Victim{leaf};
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
      cursor.demotionsHeld = true;
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

Cache::ReclaimResult Cache::takeReclaim(const Walk &walk, const std::array<uint32_t, 3> &demotions,
                                      uint64_t fillBefore) {
  // Free pages first, as the extent they fill; one that empties the runway
  // kept gives nothing yet, and the next one does.
  for (ReclaimResult released; (released = releaseExtent(walk)).madeProgress;)
    if (released.reclaimedBytes)
      return released;
  // Then the walk's victims while some form is in reach (formsInReach), the
  // leaves that lead to such a form alone, until one is released, the free
  // and landing pages fill one more extent than when the take began
  // (landingFill), a need of pages (Walk::inPlace) has one more page of its
  // form, free or, for a walk that can wait, landing (as Cache::take counts
  // it), or a state goes; otherwise only a state. A leaf it demotes starts
  // the demotions of its chain above it (demoteChainsAbove) until then.
  // Whatever it returns says whether KV went.
  bool evictedKv = false;
  const auto outcome = [&](ReclaimResult result) {
    result.evictedKv = evictedKv;
    return result;
  };
  const auto index = static_cast<size_t>(walk.form);
  const auto inPlacePages = [&] {
    return pool_.freePageCount(walk.form) + (walk.timing == ReclaimTiming::CanWait ? demotedPages()[index] : 0);
  };
  const uint32_t inPlaceBefore = inPlacePages();
  const auto enough = [&] {
    return landingFill(walk) > fillBefore || (walk.inPlace && inPlacePages() > inPlaceBefore);
  };
  for (;;) {
    const std::array<bool, 3> reach = formsInReach(walk, demotions);
    if (std::ranges::none_of(reach, std::identity{}))
      break;
    const size_t writes = writes_.size();
    const ReclaimResult evicted = evictOne(walk, reach);
    if (!evicted.madeProgress || evicted.evictedState)
      return outcome(evicted);
    evictedKv = true;
    demoteChainsAbove(walk, writes, enough);
    if (const ReclaimResult released = releaseExtent(walk); released.reclaimedBytes)
      return outcome(released);
    if (enough())
      return outcome({.madeProgress = true});
  }
  return outcome(recycleState(walk));
}

uint32_t Cache::reusablePages(ReclaimClass upTo, kv::PageForm form) const {
  const uint32_t unheld = pool_.reusablePages(form);
  if (upTo == ReclaimClass::InUse)
    return unheld;
  return unheld - kv_.keptPages(KvCache::keepBit(KvCache::Keep::Used))[static_cast<size_t>(form)];
}

Cache::ReclaimResult Cache::recycleState(const Walk &walk) {
  // The oldest checkpoint, else the oldest ordinary state; running work
  // takes a state in use after every other, once no transfer in flight can
  // return what is needed first. A Written take writes the first state that
  // needs a write in the one recency order, the one reach staged, unless an
  // older state that needs none goes first; the ledger then peeks again.
  Walk states = walk;
  states.scope.statesOnly = true;
  states.scope.pagesOnly = false;
  WalkCursor cursor;
  while (const std::optional<WalkUnit> state = nextStepUnit(states, cursor)) {
    const StateEviction eviction = reclaimState(states, state->candidate.id);
    if (!eviction.evicted && !eviction.pending)
      continue;
    return {.madeProgress = eviction.evicted, .pending = eviction.pending, .evictedState = eviction.evicted};
  }
  return {.pending = returns(states)};
}

Cache::LeafReclaim Cache::reclaimKvLeaf(const Walk &walk, uint64_t block, ReclaimTiming timing) {
  // No state in RAM sits on the leaf (nextUnit): one went first, in the
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
  // ahead would have let go at once. A demotion that cannot start, with no
  // slot to be had and nothing in flight to wait for (demoteKv), lets the
  // leaf go as without a tier, with the disk subtree that needs it: a take
  // never stops at a leaf it can neither write nor drop.
  if (kvNeededByState(block)) {
    if (timing == ReclaimTiming::Immediate && kvTierWritable())
      return LeafReclaim::Impossible;
    const LeafReclaim demotion = demoteKv(walk, block);
    if (demotion != LeafReclaim::Impossible)
      return demotion;
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
    const uint32_t reusable = reusablePages(walk.scope.upTo, form);
    reach[index] = walk.reaches(form, reusable, demoting[index], pool_.extentPages()) ||
                   (walk.inPlace && form == walk.form &&
                    reusable > pool_.freePageCount(form) + demotionPages()[index]);
  }
  return reach;
}

Cache::ReclaimResult Cache::releaseExtent(const Walk &walk) {
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

bool Cache::compactExtent(std::optional<kv::PageForm> form) {
  const auto start = AwakeClock::now();
  const KvPageMoves moves = pool_.compactExtent(pagesInTransfer(), form);
  if (moves.empty())
    return false;
  followMoves(moves);
  extentCompactMaxMilliseconds_ =
      std::max(extentCompactMaxMilliseconds_, millisecondsSince(start));
  return true;
}

std::vector<uint32_t> Cache::pagesInTransfer() const {
  std::vector<uint32_t> pages;
  pages.reserve(writes_.size() + restores_.size());
  for (const Write &write : writes_)
    pages.push_back(kv_.page(write.block));
  for (const auto &[block, _] : restores_)
    pages.push_back(kv_.page(block));
  return pages;
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
  // The walk of a reclaim that takes all it may: what is in use, no chain
  // kept.
  return transfersInFlight(Walk{.scope = {.upTo = ReclaimClass::InUse}});
}

bool Cache::writesInFlight() const noexcept { return !writes_.empty() || states_.writing(); }

bool Cache::transfersInFlight(const Walk &walk) const noexcept {
  // A write counts by what its landing frees: a page written through or
  // copied ahead returns something only where, under --kv-spill always, it
  // spills once its copy lands, or where the walk then frees its block
  // through the copy.
  return !restores_.empty() || writes(WriteClass::Reclaim) || copyAheadReturns(walk) || states_.writing() ||
         spillingWrites();
}

bool Cache::returnsInFlight(const Walk &walk, const Shortfall &shortfall) const noexcept {
  if (!restores_.empty() || writes(WriteClass::Reclaim) || copyAheadReturns(walk) || states_.writing())
    return true;
  // With no demotion in flight, the pages of the form that transfers free
  // are those written through that will spill.
  const uint32_t freed = pendingPages(shortfall.form);
  return freed && shortfall.pages &&
         pool_.freePageCount(shortfall.form) + freed >= shortfall.pages;
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

bool Cache::copyAheadReturns(const Write &write, const Walk &walk, ReclaimClass upTo) const noexcept {
  // Its block, idle once the copy lands, is in the walk's reach as
  // KvCache::pageReach counts an idle page.
  const KvCache::Keeps stays = KvCache::keepBit(KvCache::Keep::Busy) | keptChains(walk, upTo);
  return write.write != WriteClass::Reclaim && !(kv_.keeps(write.block) & stays) &&
         !(walk.reused && walk.reused->holds(write.block));
}

bool Cache::copyAheadReturns(const Walk &walk) const noexcept {
  return std::ranges::any_of(writes_,
                             [&](const Write &each) { return copyAheadReturns(each, walk, walk.scope.upTo); });
}

uint32_t Cache::spillingWrites() const noexcept {
  // Under --kv-spill always a page a Spill request holds frees once its
  // copy lands, written through or copied ahead, when it may spill then and
  // an Index page is there to take its index.
  if (!spill_)
    return 0;
  const auto spillable = std::ranges::count_if(writes_, [&](const Write &write) { return spillsOnLanding(write); });
  return std::min(static_cast<uint32_t>(spillable), pool_.freePageCount(kv::PageForm::Index));
}

bool Cache::spillsOnLanding(const Write &write) const {
  // pendingPages counts every demotion already (writes(Reclaim)).
  return spill_->always && write.write != WriteClass::Reclaim &&
         spillEligibility(write.block) == Eligibility::Eligible;
}

// Spilling: commitments of quota, write-through of Spill requests' pages,
// and their moves to the index form as a claim releases them, or under
// --kv-spill always once their copies landed.

uint64_t Cache::spillRoom() const noexcept {
  if (!spill_)
    return 0;
  const uint64_t active = heldSlotBytes() + diskBudget_->committedBytes();
  const uint64_t limit = diskBudget_->activeLimitBytes();
  return limit > active ? limit - active : 0;
}

bool Cache::commitSpill(uint64_t requestId, uint32_t pages, const ChainCopies &copies) {
  Request &active = request(requestId);
  if (active.mode != KvMode::Spill)
    throw std::logic_error("only a Spill request commits quota");
  if (!commit(active, pages, copies.unheld))
    return false;
  active.committedCopies = copies.copied;
  return true;
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
  // writes start at once: its retry releases what lands, and its blocks
  // whose copies landed before, as a chain it restored has them, at once.
  rescanSpills();
  static_cast<void>(writeThrough());
  return true;
}

bool Cache::takesQuota(uint64_t block) const { return kv_.kvCopied(block) && !kv_.held(block); }

bool Cache::holdsCopy(const Request &active, uint32_t index) const {
  const uint64_t block = active.cachedBlocks[index];
  return kv_.page(block) == active.pages[index] && kv_.kvCopied(block);
}

bool Cache::restoreTakesQuota(std::span<const uint64_t> chain) const {
  if (!spill_)
    return false;
  for (const uint64_t block : chain)
    if (takesQuota(block))
      return true;
  return false;
}

bool Cache::holdsActiveKvQuota(uint64_t requestId) const {
  const auto found = requests_.find(requestId);
  return found != requests_.end() && (found->second.committedPages || found->second.heldCopies);
}

Cache::ChainCopies Cache::chainCopies(uint64_t block) const {
  ChainCopies copies{};
  if (!block)
    return copies;
  for (const uint64_t id : kv_.chain(block).blocks) {
    copies.copied += kv_.kvCopied(id);
    copies.unheld += takesQuota(id);
  }
  return copies;
}

bool Cache::commit(Request &active, uint32_t pages, uint32_t unheld) {
  if ((pages || unheld) && !diskBudget_->commit(uint64_t{pages} * tier_->slotBytes(),
                                    heldSlotBytes() + uint64_t{unheld} * tier_->slotBytes()))
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
    // Another request's write, or a restore, gave it a copy meanwhile, or
    // its restore failed and its page holds nothing to write.
    if (kv_.kvCopied(block) || !kv_.matchable(block)) {
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

Cache::Eligibility Cache::spillEligibility(uint64_t block, const ReleaseRule &rule) const {
  const uint32_t index = kv_.chainLength(block) - 1;
  if (index < spill_->densePages)
    return Eligibility::KeptFull;
  bool held = false;
  for (const auto &[id, holder] : requests_) {
    if (holder.cachedBlocks.size() <= index || holder.cachedBlocks[index] != block)
      continue;
    if (holder.mode != KvMode::Spill)
      return Eligibility::KeptFull;
    if (index + spill_->windowPages >= holder.cachedBlocks.size() || (rule.takesFrom && !rule.takesFrom(id)))
      return Eligibility::Later;
    held = true;
  }
  // Nothing frees a page no request holds: it stays a cached Full page.
  if (!held)
    return Eligibility::KeptFull;
  return index < rule.kept.size() && rule.kept[index] == block ? Eligibility::Later : Eligibility::Eligible;
}

void Cache::rescanSpills() noexcept {
  for (auto &[_, active] : requests_)
    active.spillScan = 0;
}

TokenAdmission Cache::spillWritten(SpilledExtents extents, const std::function<void(uint64_t)> &funds) {
  if (!spill_)
    return {};
  // Under --kv-spill always the pages whose copies landed move now.
  // Otherwise a lane's pages stay Full until a claim releases them
  // (releasePages), and only the runway refills.
  return moveLandedPages([this] { return spill_->always ? landedPages() : std::vector<uint32_t>{}; }, extents,
                         funds);
}

void Cache::forReleasable(const Request &active, const ReleaseRule &rule, const ReleaseCount &count,
                          const std::function<bool(uint32_t, uint64_t, bool)> &each) const {
  if (!spill_ || active.mode != KvMode::Spill)
    return;
  const auto blocks = static_cast<uint32_t>(active.cachedBlocks.size());
  const uint32_t end = blocks > spill_->windowPages ? blocks - spill_->windowPages : 0;
  // A copy owed lands only while the SSD cache writes Spill requests'
  // copies: with none in flight the tier's share, the quota or the volume
  // holds it back.
  const bool writing = writes(WriteClass::Spill) != 0;
  for (uint32_t index = std::max(active.spillScan, spill_->densePages); index < end; ++index) {
    const uint32_t page = active.pages[index];
    const uint64_t block = active.cachedBlocks[index];
    if (pool_.form(page) != kv::PageForm::Full || kv_.page(block) != page || count.blocks.contains(block) ||
        spillEligibility(block, rule) != Eligibility::Eligible)
      continue;
    const KvCache::Transfer transfer = kv_.transfer(block);
    const bool clean = kv_.kvCopied(block) && transfer == KvCache::Transfer::None;
    if (!clean && transfer != KvCache::Transfer::Draining && !(writing && transfer == KvCache::Transfer::None))
      continue;
    if (!each(index, block, clean))
      return;
  }
}

std::vector<uint64_t> Cache::holdersAt(uint32_t index, uint64_t block) const {
  std::vector<uint64_t> holders;
  for (const auto &[id, holder] : requests_)
    if (holder.cachedBlocks.size() > index && holder.cachedBlocks[index] == block)
      holders.push_back(id);
  return holders;
}

Cache::ReleasablePages Cache::releasablePages(uint64_t requestId, const ReleaseRule &rule,
                                              const ReleaseCount &count) const {
  ReleasablePages result;
  forReleasable(request(requestId), rule, count, [&](uint32_t index, uint64_t block, bool clean) {
    if (!result.clean && !result.landing)
      for (const uint64_t holder : holdersAt(index, block))
        if (!count.holders.contains(holder))
          result.holders.push_back(holder);
    ++(clean ? result.clean : result.landing);
    return true;
  });
  return result;
}

void Cache::countRelease(uint64_t requestId, const ReleaseMoves &moves, bool landing, const ReleaseRule &rule,
                         ReleaseCount &count) const {
  uint32_t counted = 0;
  forReleasable(request(requestId), rule, count, [&](uint32_t index, uint64_t block, bool clean) {
    if (counted == moves.pages)
      return false;
    if (!clean && !landing)
      return true;
    if (!counted++)
      for (const uint64_t holder : holdersAt(index, block))
        count.holders.insert(holder);
    count.blocks.insert(block);
    return true;
  });
  // The pages its moves free lie beside the next release's moves.
  count.beside += moves.pages;
  count.extents += moves.extents;
}

uint32_t Cache::releaseBeside(const VictimQuery &query) const {
  constexpr kv::PageForm full = kv::PageForm::Full;
  if (!spill_ || spill_->always)
    return 0;
  const uint32_t extent = pool_.extentPages();
  const uint32_t empty = pool_.emptyExtents(full) * extent;
  const uint32_t reusable = pool_.reusablePages(full);
  if (reusable - std::min(reusable, empty) >= extent)
    return 0;
  const uint32_t free = pool_.freePageCount(full);
  const uint32_t scattered = free - std::min(free, empty);
  return scattered - std::min(scattered, walkFor(query).keptPages[static_cast<size_t>(full)]);
}

Cache::ReleaseMoves Cache::releaseMoves(uint32_t pages, bool landing, const ReleaseCount &count) const {
  const uint32_t extent = pool_.extentPages();
  // The extents they complete beside the free pages, as reclaims and landings
  // count them; no Full runway stays (SpilledExtents::Released).
  uint32_t extents = completedExtents(count.beside, pages, extent, false);
  if (!landing) {
    const uint32_t unpinned = pool_.unfixedExtents(pagesInTransfer(), kv::PageForm::Full);
    extents = std::min(extents, unpinned - std::min(unpinned, count.extents));
  }
  const uint32_t moves = std::min(pages, extents * extent);
  const uint32_t firstRunway = std::min(moves, pool_.freePageCount(kv::PageForm::Index));
  if (!moves || firstRunway + pool_.freePageCount(kv::PageForm::Full) < extent)
    return {};
  return {.pages = moves, .extents = extents};
}

uint32_t Cache::inPlaceIndexPages(uint32_t pages) const noexcept {
  const uint64_t besideRunway = uint64_t{pages} + pool_.extentPages();
  return pages && pool_.freePageCount(kv::PageForm::Index) >= besideRunway ? static_cast<uint32_t>(besideRunway) : 0;
}

uint32_t Cache::inPlaceMoves(uint32_t pages, uint32_t clean) const {
  const uint32_t extent = pool_.extentPages();
  if (!pages || pages > clean)
    return 0;
  if (inPlaceIndexPages(pages))
    return pages;
  return clean - pages >= extent && releaseMoves(extent, false).pages ? pages + extent : 0;
}

uint32_t Cache::releasePages(uint64_t requestId, uint32_t pages, bool inPlace, const ReleaseRule &rule,
                             const std::function<void(uint64_t)> &funds, uint32_t beside) {
  Request &active = request(requestId);
  if (!spill_ || active.mode != KvMode::Spill || !pages)
    return 0;
  // A need of bytes takes whole extents, which only the Full pages left
  // gathered in the fewest extents leave (SpilledExtents::Released), and
  // only those compaction can empty, whose bytes pay for the next runway:
  // past them the moves would cost the lane the reads of its blocks and
  // free nothing (releaseMoves). In place an extent more leaves so, past
  // the claim's pages (inPlaceMoves); alone they move only into the free
  // Index pages its reclaims kept (walkFor), gathering nothing it takes.
  const auto found =
      static_cast<uint32_t>(landedPages(active, pages + (inPlace ? pool_.extentPages() : 0), rule).size());
  const uint32_t wanted =
      inPlace ? inPlaceMoves(pages, found) : releaseMoves(found, false, {.beside = beside}).pages;
  if (!wanted)
    return 0;
  const uint32_t claimed = inPlace ? pool_.freePageCount(kv::PageForm::Full) + pages : 0;
  const uint64_t moves = spilled_.spillMoves;
  const uint64_t before = pool_.allocatedBytes();
  static_cast<void>(moveLandedPages(
      [&] { return landedPages(active, wanted - static_cast<uint32_t>(spilled_.spillMoves - moves), rule); },
      inPlace && wanted == pages ? std::nullopt : std::optional(SpilledExtents::Released), funds, claimed));
  const auto moved = static_cast<uint32_t>(spilled_.spillMoves - moves);
  // A release counts where it freed what it was for: Full pages in place,
  // or bytes.
  if (moved && (inPlace || pool_.allocatedBytes() < before)) {
    ++spilled_.releases;
    spilled_.releasedPages += moved;
  }
  return moved;
}

std::vector<uint32_t> Cache::landedPages() {
  std::vector<uint32_t> pages;
  for (auto &[_, active] : requests_)
    if (active.mode == KvMode::Spill) {
      const std::vector<uint32_t> landed = landedPages(active, std::numeric_limits<uint32_t>::max(), {});
      pages.insert(pages.end(), landed.begin(), landed.end());
    }
  // Requests on one prefix find its pages each.
  std::sort(pages.begin(), pages.end());
  pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
  return pages;
}

std::vector<uint32_t> Cache::landedPages(Request &active, uint32_t most, const ReleaseRule &rule) {
  std::vector<uint32_t> pages;
  const auto blocks = static_cast<uint32_t>(active.cachedBlocks.size());
  const uint32_t end = blocks > spill_->windowPages ? blocks - spill_->windowPages : 0;
  // The scan's mark passes what is spilled or kept Full, and stops at the
  // first page that may spill later. A page waits for its own copy to
  // land: one being written, or whose block has none of its own, as a
  // block recomputed over another conversation's copy (KvCache::insert)
  // until its rewrite lands. The pages past it whose copies landed, as a
  // chain's persisted points have them, are those a release, or under
  // --kv-spill always the landing, moves.
  bool settled = true;
  for (uint32_t index = std::max(active.spillScan, spill_->densePages); index < end && pages.size() < most;
       ++index) {
    const uint32_t page = active.pages[index];
    const uint64_t block = active.cachedBlocks[index];
    // A writer that kept a page of its own beside its block's restore
    // keeps it Full.
    Eligibility eligibility = Eligibility::KeptFull;
    if (pool_.form(page) == kv::PageForm::Full && kv_.page(block) == page)
      eligibility = kv_.kvCopied(block) && kv_.transfer(block) == KvCache::Transfer::None
                        ? spillEligibility(block, rule)
                        : Eligibility::Later;
    if (eligibility == Eligibility::Eligible)
      pages.push_back(page);
    settled = settled && eligibility == Eligibility::KeptFull;
    if (settled)
      active.spillScan = index + 1;
  }
  return pages;
}

TokenAdmission Cache::moveLandedPages(const std::function<std::vector<uint32_t>()> &landed,
                                      std::optional<SpilledExtents> extents,
                                      const std::function<void(uint64_t)> &funds, uint32_t claimed) {
  // Free Index pages take the landed pages first, the runway's among them,
  // so a spill never waits for an allocation; without `extents` the rest
  // stays Full. With them the pages move a runway at a time: the Full
  // extents each runway's moves empty leave as `extents` says (released,
  // the Full pages left first gather in the fewest extents), past the
  // `claimed` Full pages a claim takes in place, and their bytes pay for the
  // Index extent the runway refills with next (`funds`): where they cover
  // it, it passes the engine's limit and the host, under critical pressure
  // too, since on Qwen3.8-Flash-Next a Full extent returns 17 times an Index
  // one's bytes. The pages that wait are
  // found again for each runway, since gathering moves them. The runway then
  // stands for the next spill, at the memory wall too, where a lane's
  // release moves its pages through it.
  for (std::vector<uint32_t> pages = landed();; pages = landed()) {
    const size_t count = std::min<size_t>(pages.size(), pool_.freePageCount(kv::PageForm::Index));
    if (count) {
      followMoves(pool_.movePages(std::span<const uint32_t>(pages).first(count), kv::PageForm::Index).moves);
      spilled_.spillMoves += count;
    }
    if (!extents)
      return {};
    if (!pages.empty()) {
      if (*extents == SpilledExtents::Released)
        while (compactExtent(kv::PageForm::Full)) {
        }
      // The pages that wait free theirs too.
      const uint64_t freeing = uint64_t{pool_.freePageCount(kv::PageForm::Full)} + pages.size() - count;
      const auto past = static_cast<uint32_t>(freeing > claimed ? (freeing - claimed) / pool_.extentPages() : 0);
      const uint64_t released =
          pool_.reclaimEmptyExtents(*extents == SpilledExtents::KeepRunway, past, kv::PageForm::Full);
      if (released && funds)
        funds(released);
    }
    const bool waiting = count < pages.size();
    const uint32_t free = pool_.freePageCount(kv::PageForm::Index);
    if (!waiting && free >= pool_.extentPages())
      return {};
    if (const metal::AllocationResult runway = pool_.fillRunway(kv::PageForm::Index); !runway)
      return {TokenAdmissionFailure::Denied,
              waiting ? static_cast<uint32_t>(pages.size() - count) : pool_.extentPages() - free, free,
              runway.failure, kv::PageForm::Index};
  }
}

bool Cache::owesCopies(uint64_t requestId) const { return !request(requestId).owedCopies.empty(); }

uint32_t Cache::indexPages(uint64_t requestId) const { return request(requestId).indexPages; }

bool Cache::holdsIndexPages(uint64_t requestId) const { return indexPages(requestId) != 0; }

bool Cache::restoresIndexPages(std::span<const uint64_t> chain) const {
  if (!spill_)
    return false;
  for (uint32_t position = 0; position < chain.size(); ++position) {
    const uint32_t page = kv_.page(chain[position]);
    if (page == KvCache::noPage ? position >= spill_->densePages : pool_.form(page) == kv::PageForm::Index)
      return true;
  }
  return false;
}

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

std::vector<uint32_t> Cache::trimmablePages(size_t most) const {
  std::vector<uint32_t> pages;
  for (auto candidate = kv_.trimCandidate(0); candidate && pages.size() < most;
       candidate = kv_.trimCandidate(candidate->id)) {
    const uint32_t page = kv_.page(candidate->id);
    if (windowsPassed(page, kv_.chainLength(candidate->id) - 1))
      pages.push_back(page);
  }
  return pages;
}

uint32_t Cache::trimmableCount() const {
  // A page a request holds at its chain index, within the windowPages
  // blocks before its last published one or past it, which its window reads.
  std::unordered_set<uint64_t> read;
  for (const auto &[_, holder] : requests_)
    for (uint32_t index = windowStart(holder); index < holder.pages.size(); ++index) {
      ++windowVisits_;
      if (const uint64_t block = kv_.blockOn(holder.pages[index]);
          block && kv_.chainLength(block) == index + 1 && kv_.trimmable(block))
        read.insert(block);
    }
  return kv_.trimCandidates() - static_cast<uint32_t>(read.size());
}

std::vector<uint32_t> Cache::passedKeptPages() const {
  std::vector<uint32_t> kept;
  for (const auto &[_, active] : requests_)
    for (const uint32_t index : active.privatePages) {
      ++windowVisits_;
      if (pool_.form(active.pages[index]) == kv::PageForm::Full && windowsPassed(active.pages[index], index))
        kept.push_back(active.pages[index]);
    }
  std::ranges::sort(kept);
  kept.erase(std::ranges::unique(kept).begin(), kept.end());
  return kept;
}

uint32_t Cache::passedPages() const {
  return trimmableCount() + static_cast<uint32_t>(passedKeptPages().size());
}

uint32_t Cache::trimWindows(bool allocate) {
  if (!window_ || !window_->trim)
    return 0;
  // Without `allocate`, no more than the free Slim pages take.
  std::vector<uint32_t> pages =
      trimmablePages(allocate ? std::numeric_limits<size_t>::max() : pool_.freePageCount(kv::PageForm::Slim));
  // So do the pages requests keep beside their blocks, once each: a
  // writer's own, or one a block left for a writer's, which every request
  // that held it keeps (passedKeptPages). One no longer Full leaves the
  // request's list.
  for (auto &[_, active] : requests_)
    std::erase_if(active.privatePages,
                  [&](uint32_t index) { return pool_.form(active.pages[index]) != kv::PageForm::Full; });
  const std::vector<uint32_t> kept = passedKeptPages();
  pages.insert(pages.end(), kept.begin(), kept.end());
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
  // The KV cache counts what each block's state asks of its KV, the
  // waiting chains that end at it and the resume point, as they are.
  std::unordered_map<uint64_t, uint32_t> wanted;
  for (const uint64_t last : wanted_.states())
    ++wanted[last];
  const uint64_t point = states_.resumePoint();
  kv_.checkInvariants([this](uint64_t block) { return states_.onDisk(block); },
                      [&](uint64_t block) {
                        // A state in RAM sits on resident KV (KvCache::pageReach).
                        check(!states_.stateResident(block) || kv_.page(block) != KvCache::noPage,
                              "a state in RAM sits on a block with a page");
                        const auto chains = wanted.find(block);
                        return KvCache::Marks{.state = states_.marks(block),
                                              .stateInRam = states_.stateResident(block),
                                              .wanted = chains == wanted.end() ? 0 : chains->second,
                                              .resumePoint = block == point};
                      },
                      // A walk takes a resident leaf by what restores
                      // through it, in use or not, the state at it in RAM,
                      // its copy (nextUnit).
                      [this](uint64_t block) {
                        const bool needed = kvNeededByState(block);
                        return KvCache::LeafKind{.dead = !needed,
                                                 .inUse = needed && kvNeededByStateInUse(block),
                                                 .stateInRam = needed && states_.stateResident(block),
                                                 .copied = needed && kv_.slot(block) != nullptr};
                      });
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
    // Each block on its own page without a copy of its own is owed: a
    // release takes its page once that copy lands.
    if (active.mode == KvMode::Spill) {
      const std::unordered_set<uint64_t> owed(active.owedCopies.begin(), active.owedCopies.end());
      for (uint32_t index = 0; index < active.cachedBlocks.size(); ++index) {
        const uint64_t block = active.cachedBlocks[index];
        check(kv_.page(block) != active.pages[index] || kv_.kvCopied(block) || owed.contains(block),
              "a Spill request owes the copy of each block on its pages without one");
        // A release takes the oldest pages from the scan's mark on
        // (landedPages): the mark passes only what is spilled or kept Full.
        check(index >= active.spillScan || index < spill_->densePages ||
                  pool_.form(active.pages[index]) != kv::PageForm::Full || kv_.page(block) != active.pages[index] ||
                  spillEligibility(block) != Eligibility::Eligible,
              "a Spill request's scan mark passed a page a release takes");
      }
    }
    committed += active.committedPages;
    if (spill_) {
      uint32_t heldCopies = 0;
      for (uint32_t index = 0; index < active.cachedBlocks.size(); ++index)
        heldCopies += holdsCopy(active, index);
      check(heldCopies == active.heldCopies, "a request counts the blocks on its pages with a copy");
    }
  }
  if (spill_)
    check(committed * tier_->slotBytes() == diskBudget_->committedBytes(),
          "the commitments are the requests'");
  // The trim count, which the window reads bound, counts the pages the
  // trims take.
  if (window_)
    check(trimmableCount() == trimmablePages(std::numeric_limits<size_t>::max()).size(),
          "the trim count is that of the pages every window has passed");
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
  if (spill_)
    for (uint32_t index = 0; index < active.cachedBlocks.size(); ++index)
      active.heldCopies += holdsCopy(active, index);
  pagesChanged(active, 0);
  // A Spill request's commitment holds a slot for each page without a copy
  // (chainCopies). A reclaim may have written copies for blocks of the
  // chain since it committed, as allocate() runs one around this restore:
  // their pages are written already, and their copies count as held now.
  if (spills) {
    const auto copied = static_cast<uint32_t>(
        std::ranges::count_if(active.cachedBlocks, [&](uint64_t block) { return kv_.kvCopied(block); }));
    if (copied > active.committedCopies)
      returnCommitment(active, copied - active.committedCopies);
    oweCopies(active, active.cachedBlocks);
  }
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

Cache::LeafReclaim Cache::demoteKv(const Walk &walk, uint64_t block) {
  if (!kvTierWritable())
    return LeafReclaim::Impossible;
  // A demotion that cannot start waits while a transfer in flight returns
  // something to the walk (transfersInFlight(walk)): a reclaim write, which
  // frees room in the tier or the quota, or a copy ahead whose block the
  // walk then frees through it, which frees memory though no quota: with
  // the quota full the walk waits for such a copy too. A copy ahead the walk
  // cannot free through is not waited for; the tier's share keeps Reclaim
  // writes in flight whenever its slots are full, so a full tier is always
  // waited out. With nothing in flight for the walk, the leaf goes without
  // a copy.
  const auto room = [&] { return transfersInFlight(walk) ? LeafReclaim::Pending : LeafReclaim::Impossible; };
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
  const std::optional<DiskReplacement> next = nextDiskReplacement(inUse, active);
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

std::optional<Cache::DiskReplacement> Cache::nextDiskReplacement(bool inUse, bool active) const {
  const auto pick = [](DiskVictim kind, const std::optional<CacheEvictionCandidate> &at) {
    return at ? std::optional<DiskReplacement>(DiskReplacement{kind, *at}) : std::nullopt;
  };
  // The only copy of what a queued request restores, or of the resume
  // point, stays but for a Spill lane's active KV: a running lane beats a
  // queued one, a copy for the cache does not.
  auto stateOnly = states_.diskCandidate(false);
  while (stateOnly && !active && reusedCopy(stateOnly->id, true))
    stateOnly = states_.diskCandidate(false, stateOnly);
  const auto stateCopy = states_.diskCandidate(true);
  // Last, and only for a copy in use: the only copy of a state in use. The
  // leaf it held is ordinary then, and goes on a later call.
  const auto stateInUse = [&] {
    return pick(DiskVictim::StateInUse, inUse ? states_.inUseDiskCandidate() : std::nullopt);
  };
  if (persistent_) {
    if (const auto unneeded = kv_.unneededCopy())
      return pick(DiskVictim::KvCopy, unneeded);
    if (stateOnly || stateCopy)
      return older(stateOnly, stateCopy) ? pick(DiskVictim::StateOnly, stateOnly)
                                         : pick(DiskVictim::StateCopy, stateCopy);
    return stateInUse();
  }
  // A redundant copy loses nothing: its data stays in RAM.
  const auto kvCopy = kv_.diskCandidate(true);
  if (kvCopy || stateCopy)
    return older(stateCopy, kvCopy) ? pick(DiskVictim::StateCopy, stateCopy) : pick(DiskVictim::KvCopy, kvCopy);
  // A leaf holding a state in use stays with that state, and one holding a
  // state in RAM, as a spilled leaf can, stays until reclaim takes the state
  // or a committed page evicts it (evictStateOnDiskLeaf): a reclaim makes
  // room here for that state's own write, or for a demotion beside the state
  // it picked.
  auto kvLeaf = kv_.diskCandidate(false);
  while (kvLeaf && (kvNeededByStateInUse(kvLeaf->id) || states_.stateResident(kvLeaf->id) ||
                    (!active && reusedCopy(kvLeaf->id, false))))
    kvLeaf = kv_.diskCandidate(false, kvLeaf->id);
  if (kvLeaf || stateOnly)
    return older(stateOnly, kvLeaf) ? pick(DiskVictim::StateOnly, stateOnly) : pick(DiskVictim::KvLeaf, kvLeaf);
  return stateInUse();
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
    // committed. A release takes its page once that copy lands. A writer
    // that kept a page of its own beside the block never spills that page,
    // and owes nothing.
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
      // Only a resident block lacks a disk copy. One whose restore failed
      // holds nothing to write, and no lookup reaches the point past it.
      if (!kv_.matchable(id))
        return settled(PersistStatus::Unneeded);
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
