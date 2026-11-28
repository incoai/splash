#include "engine/Cache.hpp"

#include "AwakeClock.hpp"
#include "engine/DiskLabels.hpp"
#include "StderrLine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <compare>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <vector>
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
    : pool_(pool), tier_(kvTier), parts_(kvTier ? kvTier->components() : 0), spill_(spill),
      persistent_(kvTier && kvTier->persistent()),
      window_(window), diskBudget_(std::move(diskBudget)),
      kv_(pool, recency_, spill.has_value(), persistent_, window ? window->windowPages : 0,
          window ? window->unitPages : 0),
      states_(kv_, recency_, [this](uint64_t block, bool inUse, bool ahead, uint64_t pointsBefore) {
                // A state's write has its point's worth (writeWorth), as of
                // its last use.
                return freeDiskSpace(inUse, false, ahead, {.rank = writeWorth(block).rank, .lastUse = pointsBefore});
              },
              persistent_) {
  // Spilled K/V lives in the tier's copies, within its quota.
  if (spill_ && (!tier_ || !diskBudget_))
    throw std::invalid_argument("a cache whose requests spill needs a disk tier and its quota");
  // A spillable layout's slots hold block records of every layer, and a
  // spilled page no layer's K/V (kv::SlotFormat).
  if (window_ && (!window_->windowPages || spill_))
    throw std::invalid_argument("a windowed cache keeps a window of pages, and its requests do not spill");
  // A block's copy has a part for its base unit, and one for its window
  // unit where it has one.
  if (tier_ && parts_ != (kv::componentBit(kv::Component::Base) |
                          (window_ && window_->unitPages ? kv::componentBit(kv::Component::Window) : 0)))
    throw std::invalid_argument("a KV tier's components are not those of the cache's blocks");
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
  // Every block's label by id, from its base part's record, with its window
  // part's where copies have two. A parent's id is lower than its
  // children's, so a chain resolves in id order and a cycle cannot form.
  struct Recorded final {
    KvBlockLabel label;
    size_t index = 0;
    std::optional<size_t> window;
    // The copy's serial, which a part added later carries.
    uint64_t serial = 0;
    // Tokens up to and including the block; zero while its chain is broken.
    uint32_t tokens = 0;
    bool needed = false;
  };
  std::map<uint64_t, Recorded> recorded;
  // New ids and copy serials continue after every one a record names,
  // taken back or not: a record whose clearing a crash lost must never name
  // a block or a copy of this process.
  uint64_t highestId = 0;
  const bool paired = parts_ != kv::componentBit(kv::Component::Base);
  // A copy's window part by its block's id, identity and the copy's serial,
  // which pair it with its base part whatever else their labels hold.
  using PartKey = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>;
  const auto partKey = [](const KvSlotLabel &label) {
    return PartKey{label.block.id, label.block.identityLo, label.block.identityHi, label.copy};
  };
  std::map<PartKey, size_t> windows;
  std::vector<std::pair<KvSlotLabel, size_t>> bases;
  for (size_t index = 0; index < blocks.size(); ++index) {
    const auto label = decodeKvSlotLabel(blocks[index].label, paired);
    if (label) {
      highestId = std::max(highestId, label->block.id);
      copySerial_ = std::max(copySerial_, label->copy);
    }
    if (!label || !label->block.id || label->block.parent >= label->block.id ||
        !(parts_ & kv::componentBit(blocks[index].part)))
      ++result.dropped;
    else if (blocks[index].part == kv::Component::Base)
      bases.emplace_back(*label, index);
    else if (!windows.emplace(partKey(*label), index).second)
      ++result.dropped;
  }
  // A copy holds what its block held: its base part, and its window part
  // where it held its unit, paired with the base part. A window part alone
  // is what a crash before its base part landed leaves: no copy.
  for (const auto &[label, index] : bases) {
    std::optional<size_t> window;
    if (paired) {
      const auto found = windows.find(partKey(label));
      if (found != windows.end()) {
        window = found->second;
        windows.erase(found);
      }
    }
    if (!recorded.emplace(label.block.id, Recorded{label.block, index, window, label.copy}).second)
      ++result.dropped;
  }
  result.dropped += static_cast<uint32_t>(windows.size());
  for (auto &[id, block] : recorded) {
    if (!block.label.parent) {
      block.tokens = KvCache::pageTokens;
    } else if (auto parent = recorded.find(block.label.parent);
               parent != recorded.end() && parent->second.tokens) {
      block.tokens = parent->second.tokens + KvCache::pageTokens;
    }
  }
  // A state comes back with its whole chain, and the window parts of its
  // tail, which a resume reads; a block only for a state.
  const auto tailWhole = [&](std::map<uint64_t, Recorded>::const_iterator block) {
    for (uint32_t index = 0; window_ && block != recorded.end() && index < window_->windowPages;
         ++index, block = recorded.find(block->second.label.parent))
      if (paired && !block->second.window)
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
        block->second.tokens != label->tokens || !tailWhole(block)) {
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
        slot = blocks[block.index].adopt(nullptr);
        if (block.window)
          slot = blocks[*block.window].adopt(std::move(slot));
      } catch (const std::invalid_argument &) {
        slot.reset();
      }
    }
    if (!slot || !kv_.adoptDiskBlock(id, block.label.parent, block.label.tokens,
                                     {block.label.identityLo, block.label.identityHi},
                                     std::move(slot), block.label.lastUsed)) {
      ++result.dropped;
      continue;
    }
    kv_.setCopySerial(id, block.serial);
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
  // A window part only a point not taken back needed goes.
  for (std::optional<CacheEvictionCandidate> part; (part = unneededWindow());) {
    tier_->giveUp(kv_.slot(part->id), kv::componentBit(kv::Component::Window));
    kv_.partGivenUp(part->id);
    ++result.dropped;
  }
  recency_.continueAfter(lastUsed);
  kv_.continueIdsAfter(highestId);
  // A lower limit than the earlier process had takes the oldest copies.
  result.dropped += fitDiskLimit();
  const StateCacheSnapshot held = states_.snapshot();
  result.states = held.entries;
  result.blocks = kv_.diskBlocks();
  result.bytes = held.diskBytes + kv_.diskBytes();
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
  entry->second.windows.resize(kv_.blockPages() - 1);
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
  for (const std::vector<uint32_t> &table : active.windows)
    for (const uint32_t page : table)
      if (page != KvCache::noPage)
        pool_.releasePage(page, false);
  if (!active.cachedBlocks.empty()) {
    kv_.releaseActive(active.cachedBlocks.back());
    // Its chain's end ends no search any more (superseded).
    kv_.historyChanged(active.cachedBlocks.back());
  }
  if (active.pendingRestores) {
    for (auto &[_, restore] : restores_)
      std::erase(restore.waiters, requestId);
    // A read no one waits for that has not started holds no IO: its pages go.
    std::erase_if(unitReads_, [&](UnitRead &read) {
      std::erase(read.waiters, requestId);
      if (read.transfer || !read.waiters.empty())
        return false;
      for (const uint32_t page : std::span(read.pages).subspan(1))
        pool_.releasePage(page, false);
      return true;
    });
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
  result.rendersReply = std::ranges::any_of(blocks, [&](uint64_t block) { return kv_.heldReply(block); });
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
    return {active.pages, active.windows, active.revision + 1, *active.dirtyFrom};
  return {active.pages, active.windows, active.revision, active.firstChanged};
}

PageTableView Cache::cutPageTable(uint64_t requestId) {
  Request &active = request(requestId);
  // The units of the blocks the window passed leave the lane's tables, the
  // entry absent before the pages go: the pages are free once no other
  // lane, and no state's tail, holds them.
  if (window_)
    for (const uint32_t start = windowStart(active); active.windowFrom < start; ++active.windowFrom) {
      for (std::vector<uint32_t> &table : active.windows)
        if (const uint32_t page = std::exchange(table[active.windowFrom], KvCache::noPage); page != KvCache::noPage)
          pool_.releasePage(page, false);
      pagesChanged(active, active.windowFrom);
    }
  if (active.dirtyFrom) {
    ++active.revision;
    active.firstChanged = *active.dirtyFrom;
    active.dirtyFrom.reset();
    active.sources = cutSources(active);
  }
  return {active.pages, active.windows, active.sources, active.revision, active.firstChanged};
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
    const std::vector<uint32_t> writerWindow = windowAt(active, logical);
    auto inserted =
        kv_.insert(parent, exactTokens.subspan(begin, KvCache::pageTokens), active.pages[logical],
                   blockIdentity(begin, KvCache::pageTokens, identity.images, identity.salt), writerWindow);
    const uint32_t writerPage = active.pages[logical];
    const bool replacePage = inserted.physicalPage != writerPage;
    if (replacePage)
      holdBlock(active, logical, inserted.id);
    kv_.retainActive(inserted.id);
    if (!active.cachedBlocks.empty())
      kv_.historyChanged(active.cachedBlocks.back());
    active.cachedBlocks.push_back(inserted.id);
    kv_.historyChanged(inserted.id);
    if (replacePage) {
      pagesChanged(active, logical);
      kv_.unholdPage(writerPage);
      for (const uint32_t page : writerWindow)
        if (page != KvCache::noPage)
          pool_.releasePage(page, false);
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

std::optional<KvCache::TailUnits> Cache::tailUnits(uint64_t kvBlock, std::optional<uint64_t> lane) const {
  KvCache::TailUnits units;
  if (!window_ || !kv_.unitPages())
    return units;
  const auto found = lane ? requests_.find(*lane) : requests_.end();
  // The lane that wrote a block's base page holds the unit of that command
  // with it: a base page and its unit are written and taken together.
  // Disk-only, a block holds no unit.
  for (uint64_t block = kvBlock, index = 0; block && index < window_->windowPages;
       ++index, block = kv_.key(block).parent) {
    if (kv_.page(block) == KvCache::noPage || kv_.holdsUnit(block))
      continue;
    const uint32_t at = kv_.chainLength(block) - 1;
    if (found == requests_.end())
      return std::nullopt;
    const Request &holder = found->second;
    if (at >= holder.cachedBlocks.size() || holder.cachedBlocks[at] != block || holder.pages[at] != kv_.page(block) ||
        holder.windows.front()[at] == KvCache::noPage)
      return std::nullopt;
    units.emplace_back(block, windowAt(holder, at));
  }
  return units;
}

bool Cache::stateTailPresent(uint64_t kvBlock, std::optional<uint64_t> lane) {
  if (tailUnits(kvBlock, lane))
    return true;
  ++refusedStates_;
  return false;
}

void Cache::publishCompositeState(uint64_t kvBlock,
                                  std::shared_ptr<const CompositeState> state,
                                  bool checkpoint, std::optional<uint64_t> lane) {
  const std::optional<KvCache::TailUnits> units = tailUnits(kvBlock, lane);
  if (!units)
    throw std::logic_error("a state's tail holds a block without its window unit");
  const KvCache::DeferredReleases scope{kv_};
  states_.publishCompositeState(kvBlock, std::move(state), checkpoint, *units);
}

bool Cache::publishStateToDisk(uint64_t kvBlock, const StateWriter &write, bool checkpoint,
                               std::optional<uint64_t> lane) {
  const std::optional<KvCache::TailUnits> units = tailUnits(kvBlock, lane);
  if (!units)
    throw std::logic_error("a state's tail holds a block without its window unit");
  const KvCache::DeferredReleases scope{kv_};
  return states_.publishStateToDisk(kvBlock, write, checkpoint, *units);
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
                                   uint64_t writtenTokens, std::array<uint32_t, kv::kPageForms> held) {
  Request &active = request(requestId);
  const uint64_t needed64 =
      (tokenCount + KvCache::pageTokens - 1) / KvCache::pageTokens;
  if (needed64 > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("KV target exceeds the page id range");
  const uint32_t needed = static_cast<uint32_t>(needed64);
  const auto previousSize = static_cast<uint32_t>(active.pages.size());
  if (needed <= previousSize)
    return {};
  // Each new block's pages, base page first, all or none.
  std::vector<uint32_t> acquired;
  const uint32_t count = needed - previousSize, kept = held[static_cast<size_t>(kv::PageForm::Full)];
  const uint32_t blockPages = kv_.blockPages();
  const TokenAdmission admission = admitPages(count * blockPages, acquired, kv::PageForm::Full, kept);
  if (!admission.granted())
    return admission;
  const uint64_t written = std::min(writtenTokens, tokenCount);
  for (uint32_t index = 0; index < acquired.size(); ++index)
    if (uint64_t{previousSize + index / blockPages + 1} * KvCache::pageTokens > written)
      pool_.clearPage(acquired[index]);
  for (uint32_t block = 0; block < count; ++block) {
    active.pages.push_back(acquired[block * blockPages]);
    for (uint32_t table = 0; table < active.windows.size(); ++table)
      active.windows[table].push_back(acquired[block * blockPages + 1 + table]);
  }
  pagesChanged(active, previousSize);
  return {};
}

std::vector<uint32_t> Cache::windowAt(const Request &active, uint32_t index) {
  std::vector<uint32_t> window;
  for (const std::vector<uint32_t> &table : active.windows)
    window.push_back(table[index]);
  return window;
}

uint32_t Cache::windowStart(const Request &active) const noexcept {
  const auto published = static_cast<uint32_t>(active.cachedBlocks.size());
  return published - std::min(published, window_->windowPages);
}

void Cache::holdBlock(Request &active, uint32_t index, uint64_t block) {
  active.pages[index] = kv_.page(block);
  kv_.holdPage(active.pages[index]);
  // With a window the writer keeps its page (KvCache::insert), so only a
  // layout without window tables comes here.
  const std::span<const uint32_t> window = kv_.window(block);
  for (uint32_t table = 0; table < active.windows.size(); ++table) {
    pool_.retainPage(window[table], false);
    active.windows[table][index] = window[table];
  }
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
      result.landingPages[static_cast<size_t>(pool_.form(kv_.page(write.block)))] +=
          write.unit || freeing.contains(write.block) ? writePages(write) : 0;
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
  for (const kv::PageForm form : kv::kAllPageForms)
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

Need Cache::growthNeed(uint32_t blocks) const {
  constexpr kv::PageForm full = kv::PageForm::Full;
  const uint32_t pages = blocks * kv_.blockPages();
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
  return (pool_.freePageCount(full) + bytes / (extent * pool_.bytesPerPage(full)) * extent) / kv_.blockPages();
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
    if (write.write == WriteClass::Reclaim && !write.unit && freesOnLanding(write.block, freeing))
      freeing.insert(write.block);
  return freeing;
}

uint32_t Cache::writePages(const Write &write) const {
  // A unit write's unit, while no point in memory has come to hold it.
  if (write.unit)
    return kv_.unitLeaving(write.block) ? kv_.unitPages() : 0;
  return kv_.pagesHeld(write.block);
}

std::array<uint32_t, kv::kPageForms> Cache::demotedPages() const {
  std::array<uint32_t, kv::kPageForms> pages{};
  const std::unordered_set<uint64_t> freeing = freeingDemotions();
  for (const Write &write : writes_)
    if (write.write == WriteClass::Reclaim && (write.unit || freeing.contains(write.block)))
      pages[static_cast<size_t>(pool_.form(kv_.page(write.block)))] += writePages(write);
  return pages;
}

std::array<uint32_t, kv::kPageForms> Cache::demotionPages() const noexcept {
  std::array<uint32_t, kv::kPageForms> pages{};
  for (const Write &write : writes_)
    if (write.write == WriteClass::Reclaim)
      pages[static_cast<size_t>(pool_.form(kv_.page(write.block)))] += write.unit ? kv_.unitPages()
                                                                                  : kv_.pagesHeld(write.block);
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
    walk.bytes = need.bytes;
    walk.inPlacePages = need.pages;
    walk.refused = Shortfall{pool_.extentPages(), kv::PageForm::Full};
    // A KV need takes the free pages of its form first, and those of the
    // other forms its attempt takes: releasing their extent would give it
    // bytes for an extent it then allocates again. Any need keeps too those
    // the other sources' units, or the claimant's own next page, move pages
    // into (VictimQuery::movedInto), which they count on.
    const auto some = [](uint32_t each) { return each != 0; };
    if (need.pages || std::ranges::any_of(need.keptPages, some) || std::ranges::any_of(query.movedInto, some)) {
      walk.form = need.form;
      for (const kv::PageForm form : kv::kAllPageForms) {
        const auto each = static_cast<size_t>(form);
        const uint64_t takes =
            uint64_t{form == need.form && need.pages ? pool_.freePageCount(form) : need.keptPages[each]} +
            query.movedInto[each];
        walk.keptPages[each] = static_cast<uint32_t>(std::min<uint64_t>(pool_.freePageCount(form), takes));
      }
    }
  } else if (need.pages) {
    // Pages in place; a host that refuses growth takes them in a page scan,
    // which takes a state only on the leaf it goes before, where a state's
    // release frees no page: a windowed layout's frees the window units of
    // its tail, and its page need walks as any.
    walk.pages = need.pages;
    walk.form = need.form;
    walk.scope.pagesOnly = need.hostRefused && !(window_ && kv_.unitPages());
    walk.refused = Shortfall{need.pages, need.form};
  } else if (need.lacksStateBuffers() || need.hostRefused) {
    walk.scope.statesOnly = true;
    walk.refused = Shortfall{};
  }
  return walk;
}

std::optional<Cache::WalkUnit> Cache::nextUnit(const Walk &walk, WalkCursor &cursor) const {
  if (checkKept_)
    checkKept();
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
  for (;; ++searchVisits_) {
    // The order of the unit given last moves past it now, as a scan moves on
    // once it has tried a candidate.
    const std::optional<Unit> last = std::exchange(cursor.last, std::nullopt);
    switch (cursor.phase) {
    case Phase::Checkpoints: {
      // The checkpoints no running lane recovers from, oldest first: they
      // save the least. A lane's recovery point leaves by use.
      if (scope.pagesOnly) {
        moveOn(phaseAfter(walk, Phase::Checkpoints));
        continue;
      }
      if (!cursor.started) {
        cursor.state = states_.checkpointCandidate();
        cursor.started = true;
      } else if (last == Unit::Checkpoint) {
        cursor.state = states_.checkpointCandidate(cursor.lastCandidate);
      }
      while (cursor.state && recoveryPoint(cursor.state->id))
        cursor.state = states_.checkpointCandidate(cursor.state);
      if (cursor.state)
        return give(Unit::Checkpoint, *cursor.state);
      moveOn(phaseAfter(walk, Phase::Checkpoints));
      continue;
    }
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
      moveOn(phaseAfter(walk, Phase::DeadLeaves));
      continue;
    case Phase::Tails: {
      // The releases of states for their tails' units, stage by stage, each
      // stage's states oldest first as their releases free units
      // (tailReleases), judged again at each step.
      if (!cursor.started) {
        cursor.started = true;
        cursor.state.reset();
      }
      if (const std::optional<CacheEvictionCandidate> next =
              nextTailRelease(walk, cursor.tails, cursor.state, {}, {}, {})) {
        cursor.state = *next;
        return give(Unit::Tail, *next);
      }
      if (cursor.tails == TailRelease::History) {
        cursor = WalkCursor{.phase = Phase::Tails, .tails = TailRelease::BehindLanes};
        continue;
      }
      moveOn(phaseAfter(walk, Phase::Tails));
      continue;
    }
    case Phase::Unwritten:
      // The units held for points on the SSD alone, oldest first, which
      // leave as their window parts are written: for a walk that writes and
      // waits.
      if (!walk.writes || walk.timing != ReclaimTiming::CanWait || !takesCacheWrites()) {
        moveOn(phaseAfter(walk, Phase::Unwritten));
        continue;
      }
      if (!cursor.started) {
        cursor.leaf = kv_.unwrittenUnitCandidate(0);
        cursor.started = true;
      } else if (last == Unit::Unwritten) {
        cursor.leaf = kv_.contains(cursor.lastCandidate.id) ? kv_.unwrittenUnitCandidate(cursor.lastCandidate.id)
                                                            : kv_.unwrittenUnitCandidate(0);
      }
      // Not one a chain the walk keeps holds, or that a point the walk
      // keeps needs (unitTakes).
      while (cursor.leaf &&
             ((kv_.keeps(cursor.leaf->id) & keptChains(walk, scope.upTo)) || !unitTakes(walk, cursor.leaf->id)))
        cursor.leaf = kv_.unwrittenUnitCandidate(cursor.leaf->id);
      if (cursor.leaf)
        return give(Unit::Unwritten, *cursor.leaf);
      moveOn(phaseAfter(walk, Phase::Unwritten));
      continue;
    case Phase::History:
    case Phase::ByUse:
    case Phase::InUse: {
      const bool inUse = cursor.phase == Phase::InUse, history = cursor.phase == Phase::History;
      if (inUse && scope.upTo == ReclaimClass::Ordinary) {
        moveOn(Phase::Done);
        continue;
      }
      const Unit state = inUse ? Unit::UsedState : Unit::State;
      const Unit leaf = inUse ? Unit::UsedLeaf : Unit::Leaf;
      const Unit leafState = inUse ? Unit::UsedLeafState : Unit::LeafState;
      // History that leaves with no write goes in a phase of its own, before
      // any state nothing superseded. A phase by use goes over its items by
      // what their leaving costs (WalkCursor::Pass), each pass by recency.
      using Pass = WalkCursor::Pass;
      const bool split = !history;
      const Pass pass = cursor.pass;
      const auto stateAfter = [&](std::optional<CacheEvictionCandidate> after) {
        if (pass == Pass::Recovery) {
          auto checkpoint = states_.checkpointCandidate(after);
          while (checkpoint && !recoveryPoint(checkpoint->id))
            checkpoint = states_.checkpointCandidate(checkpoint);
          return checkpoint;
        }
        const auto next = [&](std::optional<CacheEvictionCandidate> each) {
          return inUse ? states_.inUseCandidate(each) : states_.ordinaryCandidate(each);
        };
        auto state = next(after);
        while (state && ((!inUse && !ordinaryTakes(walk, state->id, history)) ||
                         (split && lostOnLeaving(state->id) != (pass == Pass::Lost))))
          state = next(state);
        return state;
      };
      // The leaves of the class a state restores through, those with the
      // state in RAM in a page scan alone (the others take the state first):
      // at no loss those with a copy, and those a demotion writes where the
      // tier takes writes and the walk can wait for one (reclaimKvLeaf); lost
      // the others where the tier takes none. A dead leaf goes, or stays,
      // with the dead leaves.
      const bool demotes = takesCacheWrites() && walk.timing == ReclaimTiming::CanWait && !cursor.demotionsHeld;
      const KvCache::LeafWalk leaves{
          .inUse = inUse, .withStateInRam = scope.pagesOnly, .copiedOnly = pass == Pass::NoLoss && !demotes};
      const auto leafFrom = [&](uint64_t after) -> std::optional<CacheEvictionCandidate> {
        if (pass == Pass::Recovery || (pass == Pass::Lost && takesCacheWrites()))
          return std::nullopt;
        std::optional<CacheEvictionCandidate> each = kv_.leafCandidate(leaves, after);
        while (each && pass == Pass::Lost && kv_.copyHoldsPages(each->id))
          each = kv_.leafCandidate(leaves, each->id);
        return each;
      };
      const Unit stateUnit = pass == Pass::Recovery ? Unit::Checkpoint : state;
      if (!cursor.started) {
        // A page scan takes a state only where it sits on the leaf that goes
        // next.
        if (!scope.pagesOnly)
          cursor.state = stateAfter(std::nullopt);
        if (!scope.statesOnly && !history)
          cursor.leaf = leafFrom(0);
        cursor.started = true;
      } else if (last == stateUnit) {
        cursor.state = stateAfter(cursor.lastCandidate);
      } else if (last == leaf || last == leafState) {
        cursor.leaf = leafFrom(cursor.lastCandidate.id);
      }
      if (cursor.restartLeaves) {
        cursor.restartLeaves = false;
        cursor.leaf = scope.statesOnly || history ? std::nullopt : leafFrom(0);
      }
      if (cursor.statesHeld)
        cursor.state.reset();
      if (cursor.state && (!cursor.leaf || cursor.state->lastUsed <= cursor.leaf->lastUsed))
        return give(stateUnit, *cursor.state);
      if (cursor.leaf)
        return give(scope.pagesOnly && states_.stateResident(cursor.leaf->id) ? leafState : leaf, *cursor.leaf);
      // What a unit of the phase waits for holds back the rest of it.
      if (split && pass != Pass::Lost)
        cursor = WalkCursor{.phase = cursor.phase,
                            .pass = pass == Pass::NoLoss && !inUse ? Pass::Recovery : Pass::Lost,
                            .statesHeld = cursor.statesHeld,
                            .demotionsHeld = cursor.demotionsHeld};
      else
        moveOn(phaseAfter(walk, cursor.phase));
      continue;
    }
    case Phase::Done:
      return std::nullopt;
    }
  }
}

std::optional<Cache::WalkUnit> Cache::nextStepUnit(const Walk &walk, WalkCursor &cursor) const {
  // The cache does not change while the walk looks for its next unit: its
  // questions of history and of the lanes' points are asked once.
  const MemoScope scope{*this};
  for (std::optional<WalkUnit> unit; (unit = nextUnit(walk, cursor));) {
    ++searchVisits_;
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
  const Phase phase = unit == Unit::State       ? (freeHistory(block) ? Phase::History : Phase::ByUse)
                      : unit == Unit::UsedState ? Phase::InUse
                      : recoveryPoint(block)    ? Phase::ByUse
                                                : Phase::Checkpoints;
  using Pass = WalkCursor::Pass;
  return WalkCursor{.phase = phase,
                    .pass = phase != Phase::ByUse && phase != Phase::InUse ? Pass::NoLoss
                            : unit == Unit::Checkpoint                     ? Pass::Recovery
                            : lostOnLeaving(block)                         ? Pass::Lost
                                                                           : Pass::NoLoss,
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
    cursor = WalkCursor{.phase = phaseAfter(walk, cursor.phase)};
    return;
  }
  if (cursor.phase != Phase::ByUse && cursor.phase != Phase::InUse)
    return;
  // The states of the phase up to the unit, as a step reached them: its
  // passes in order (WalkCursor::Pass), each by recency.
  using Pass = WalkCursor::Pass;
  const bool inUse = cursor.phase == Phase::InUse;
  const CacheEvictionCandidate at = cursor.lastCandidate;
  const bool atState = cursor.last == Unit::State || cursor.last == Unit::UsedState;
  for (std::optional<CacheEvictionCandidate> state = inUse ? states_.inUseCandidate() : states_.ordinaryCandidate();
       state; state = inUse ? states_.inUseCandidate(state) : states_.ordinaryCandidate(state)) {
    const Pass pass = lostOnLeaving(state->id) ? Pass::Lost : Pass::NoLoss;
    const bool before =
        pass < cursor.pass ||
        (pass == cursor.pass &&
         (state->lastUsed < at.lastUsed || (state->lastUsed == at.lastUsed && (!atState || state->id <= at.id))));
    if (before && !kept({inUse ? Unit::UsedState : Unit::State, *state}, walk) && (inUse || !freeHistory(state->id)) &&
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
  case Unit::Tail:
    return resumePoint || walk.keepsState(block);
  case Unit::Leaf:
  case Unit::UsedLeaf:
    return walk.keepsBlock(block);
  default:
    return false;
  }
}

bool Cache::lostOnLeaving(uint64_t block) const {
  return !states_.onDisk(block) && !states_.reclaimWrites(block);
}

bool Cache::ordinaryTakes(const Walk &walk, uint64_t block, bool history, const std::unordered_set<uint64_t> &released,
                          const KvCache::Gone &gone, const std::unordered_set<uint64_t> &evicted) const {
  // History, or any state with another in RAM below it on its path, whose
  // chain keeps the blocks, goes only where it frees what the need lacks: a
  // state's buffers, or for a need of pages those its tail holds and, on a
  // leaf, the leaf's page, which the count frees with it (reach) and which
  // no other phase offers while the state sits on it. Elsewhere on a layout
  // without windows it holds none, and a page need that took it would cost
  // a conversation the state it falls back to for no page: history, or the
  // point before a newest one no request went on from.
  // Of a count, its releases taken: the pins they took off (`gone`), the
  // RAM copies they took (`released`).
  const auto left = gone.find(block);
  const bool serves = walk.statesAreUnits ||
                      kv_.windowPins(block) > (left == gone.end() ? 0 : left->second.first) ||
                      (!walk.scope.statesOnly && kv_.residentLeaf(block));
  // A state in RAM below the block, which keeps its chain's pages.
  const auto keptBelow = [&] {
    if (!kv_.stateBelow(block))
      return false;
    for (std::vector<uint64_t> pending = kv_.children(block); !pending.empty();) {
      const uint64_t at = pending.back();
      pending.pop_back();
      ++searchVisits_;
      if (states_.stateResident(at) && !released.contains(at))
        return true;
      if (kv_.stateBelow(at))
        for (const uint64_t child : kv_.children(at))
          pending.push_back(child);
    }
    return false;
  };
  return (serves || (!history && !keptBelow())) && freeHistory(block, evicted) == history;
}

bool Cache::freeHistory(uint64_t block, const std::unordered_set<uint64_t> &gone) const {
  return superseded(block, gone) && !releaseMode(block, gone).writes;
}

bool Cache::recoveryPoint(uint64_t block) const {
  return pool_.pageActive(kv_.page(block)) && !states_.onDisk(block) && !states_.reclaimWrites(block) &&
         !superseded(block);
}

Cache::WalkCursor::Phase Cache::phaseAfter(const Walk &walk, WalkCursor::Phase phase) const {
  using Phase = WalkCursor::Phase;
  // The order of giving up, one table for the walk and its count: the
  // checkpoints no lane recovers from, the dead leaves, the history that
  // leaves with no write, the tails' units, the units of points on the SSD
  // alone, then by use (what leaves at no loss before what is lost), then
  // what is in use.
  constexpr std::array order{Phase::Checkpoints, Phase::DeadLeaves, Phase::History, Phase::Tails,
                             Phase::Unwritten,   Phase::ByUse,      Phase::InUse,   Phase::Done};
  // A walk for states' buffers takes no leaf, no tail and no window unit:
  // a release frees units. Without window units there are none.
  const auto applies = [&](Phase each) {
    const bool units = each == Phase::Tails || each == Phase::Unwritten;
    return (each != Phase::DeadLeaves && !units) ||
           (!walk.scope.statesOnly && (!units || (window_ && kv_.unitPages())));
  };
  auto at = std::ranges::find(order, phase);
  if (at == order.end() || *at == Phase::Done)
    return Phase::Done;
  while (!applies(*++at)) {
  }
  return *at;
}

bool Cache::certainReuse(uint64_t block) const {
  return block == states_.resumePoint() || std::ranges::find(wanted_.states(), block) != wanted_.states().end();
}

Cache::StateReclaim Cache::stateReclaim(const Walk &walk, uint64_t block, uint64_t staged,
                                        const std::unordered_set<uint64_t> &gone) const {
  using When = StateReclaim::When;
  // The staging buffer writes it where it leaves through its write, not
  // unwritten (releaseMode).
  const bool evict = releaseMode(block, gone).evict;
  const bool write = states_.reclaimWrites(block) && !evict;
  const bool busy = states_.writing() || (staged && staged != block);
  if (write && (!walk.writes || (walk.timing == ReclaimTiming::Immediate && busy)))
    return {When::PassedOver, true};
  // As reclaim() goes: a write waits for the staging buffer, a copy ahead
  // keeps its state until it lands; one that leaves unwritten waits for no
  // write but its own copy ahead.
  const bool later = write ? busy : evict ? states_.writing(block) : states_.reclaimOutcome(block).pending;
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
  return releaseState(block, releaseMode(block).evict);
}

KvCache::PinRelease Cache::pinRelease(uint64_t block, bool evict) const {
  const bool ram = states_.stateResident(block) && !countDemoted(block), disk = states_.onDisk(block);
  if (evict)
    return {.ram = ram, .disk = disk};
  return {.ram = ram, .written = ram && !disk && states_.reclaimWrites(block)};
}

Cache::Worth Cache::writeWorth(uint64_t block, const std::unordered_set<uint64_t> &gone) const {
  return {.rank = static_cast<uint8_t>(!states_.inUse(block) && superseded(block, gone) ? 1 : 2),
          .lastUse = kv_.lastUsed(block)};
}

Cache::ReleaseMode Cache::releaseMode(uint64_t block, const std::unordered_set<uint64_t> &gone,
                                      const uint64_t *below) const {
  if (!states_.stateResident(block) || countDemoted(block))
    return {.evict = true};
  KvCache::Gone pins;
  const KvCache::PinRelease release = pinRelease(block, false);
  // Its state's write (the staging buffer's, where the state tier takes it)
  // and its tail's window parts' (the KV tier's).
  const bool writes = release.written || !kv_.freedBy(block, pins, release).written.empty();
  if (!keptLeaving(block, gone, writes, below))
    return {.evict = true, .worth = writeWorth(block, gone)};
  return {.evict = false, .writes = writes, .worth = writeWorth(block, gone)};
}

bool Cache::keptLeaving(uint64_t block, const std::unordered_set<uint64_t> &gone, bool writes,
                        const uint64_t *below) const {
  // A copy whose parts landed stays in every mode (it writes nothing), so a
  // persistent tier's landed point is never lost. Where the KV tier takes no
  // writes no window part is written. History no request uses is written
  // only where sparse history keeps it, its tail's window parts ahead of
  // any walk (writeWindowsAhead): its release in RAM queues them, and on the
  // SSD alone it holds its units only while they are queued or in flight,
  // else it goes for them, its copy with it (the tails), so that no point
  // sparse history keeps holds units past its write's latency.
  // What its copy lacks: its tail's window parts not landed, asked of its
  // own copy alone, in RAM as on the SSD alone, which only a landing changes;
  // asked once a point within a scope.
  std::optional<bool> lacks;
  const auto lacking = [&] {
    if (!lacks) {
      if (!window_ || !kv_.unitPages()) {
        lacks = false;
      } else if (supersededMemo_) {
        const auto [at, added] = supersededMemo_->tailLanded.try_emplace(block);
        if (added)
          at->second = tailLanded(block);
        lacks = !at->second;
      } else {
        lacks = !tailLanded(block);
      }
    }
    return *lacks;
  };
  if (!takesCacheWrites() && lacking())
    return false;
  if (writeWorth(block, gone).rank != 1)
    return true;
  if (!writes && !lacking())
    return true;
  // Queued, its parts are being written: it stays until they land, whatever
  // was written below it since (a write a walk starts on its chain may make
  // a point below it no history on a persistent tier, Cache::superseded).
  if (windowsAhead_.contains(block))
    return true;
  if (!historyWritten(block, gone, below))
    return false;
  // In RAM its release queues them, which a count's demotion of it stands
  // for too.
  return states_.stateResident(block) || !lacking();
}

bool Cache::copyStays(uint64_t block, const std::unordered_set<uint64_t> &gone, const uint64_t *below) const {
  if (states_.stateResident(block) && !countDemoted(block)) {
    const ReleaseMode mode = releaseMode(block, gone, below);
    return !mode.evict && (states_.onDisk(block) || mode.writes);
  }
  // On the SSD alone its state's copy stands: what it lacks is its window
  // parts (keptLeaving).
  return states_.onDisk(block) && keptLeaving(block, gone, false, below);
}

bool Cache::historyWritten(uint64_t block, const std::unordered_set<uint64_t> &gone, const uint64_t *below) const {
  // A chain's first point is written, and then one past the spacing from
  // the nearest copy below it.
  const uint64_t from = below ? *below : copiedBelow(block, gone);
  return !from || uint64_t{kv_.chainLength(block) - kv_.chainLength(from)} * KvCache::pageTokens >
                      sparseHistoryTokens_;
}

uint64_t Cache::copiedBelow(uint64_t block, const std::unordered_set<uint64_t> &gone) const {
  // Asked once a block within a peek or a step's search, where nothing
  // changes (MemoScope), but within a count of releases, against the states
  // it evicted (`gone`).
  const bool memo = supersededMemo_ && (gone.empty() ? !supersededMemo_->gone : supersededMemo_->gone == &gone);
  if (memo)
    if (const auto found = supersededMemo_->copied.find(block); found != supersededMemo_->copied.end())
      return found->second;
  // Up its chain to the nearest point, as far as sparse history's spacing,
  // whose copy on the SSD stays as it leaves memory (copyStays); past one
  // that leaves without a copy, the point it leaves for. How a point leaves
  // asks for the point below it: asked once, and that point's answer is the
  // one past it.
  uint64_t copied = 0;
  uint64_t id = kv_.key(block).parent;
  for (uint32_t blocks = 1; id && blocks * KvCache::pageTokens <= sparseHistoryTokens_;
       ++blocks, id = kv_.key(id).parent, ++searchVisits_) {
    if (!states_.contains(id) || gone.contains(id))
      continue;
    const uint64_t under = copiedBelow(id, gone);
    copied = copyStays(id, gone, &under) ? id : under;
    break;
  }
  if (memo)
    supersededMemo_->copied.emplace(block, copied);
  return copied;
}

StateEviction Cache::releaseState(uint64_t block, bool evict, bool counted) {
  KvCache::Gone gone;
  const KvCache::Freed freed = kv_.freedBy(block, gone, pinRelease(block, evict));
  // A point sparse history keeps queues the window parts it leaves held for
  // a write ahead of any walk (keptLeaving).
  const bool ahead = !evict && writeWorth(block).rank == 1 && window_ && kv_.unitPages() && !tailLanded(block);
  const uint64_t before = kv_.unitsFreed();
  // A write makes room in the quota as the quota gives copies up, whatever
  // that frees (keepQuota's way).
  const bool writes = !evict && states_.reclaimWrites(block);
  StateEviction eviction{.evicted = true};
  if (evict)
    states_.evict(block);
  else
    eviction = states_.reclaim(block);
  // Count and take are one predicate: a release frees what freedBy names
  // (checkReleases, under test).
  const uint64_t took = kv_.unitsFreed() - before;
  if (!writes && counted && !quotaGaveUpState_)
    unitsReleased_ += took;
  if (checkReleases_ && !writes && eviction.evicted && took != freed.now.size())
    throw std::logic_error("a state's release freed " + std::to_string(took) +
                           " window units where its count names " + std::to_string(freed.now.size()));
  if (ahead && eviction.evicted)
    windowsAhead_.insert(block);
  return eviction;
}

void Cache::writeWindowsAhead() {
  // A walk that writes and waits, as far as a unit's write asks.
  const Walk ahead{.scope = {.upTo = ReclaimClass::InUse}};
  for (auto point = windowsAhead_.begin(); point != windowsAhead_.end();) {
    // Gone, back in RAM, or its window parts landed: nothing left to write.
    if (!states_.contains(*point) || states_.stateResident(*point) || tailLanded(*point)) {
      point = windowsAhead_.erase(point);
      continue;
    }
    // Each unit of its tail not yet under way; one the tier cannot take
    // ends the point's wait, and the tails take it for its units.
    bool kept = true;
    uint64_t id = *point;
    for (uint32_t index = 0; kept && id && index < window_->windowPages;
         ++index, id = kv_.key(id).parent, ++searchVisits_)
      if (kv_.contains(id) && kv_.unwrittenUnit(id))
        kept = writeUnit(ahead, id) != LeafReclaim::Impossible;
    point = kept ? std::next(point) : windowsAhead_.erase(point);
  }
}

std::optional<CacheEvictionCandidate> Cache::nextTailRelease(const Walk &walk, TailRelease stage,
                                                             const std::optional<CacheEvictionCandidate> &after,
                                                             const std::unordered_set<uint64_t> &released,
                                                             const std::unordered_set<uint64_t> &evicted,
                                                             const KvCache::Gone &gone) const {
  // A release that writes, the state's or its units' window parts, only for
  // a walk that writes and waits.
  const bool writes = walk.writes && walk.timing == ReclaimTiming::CanWait;
  // Of the releases a count made before (`released`), those that evicted
  // their states (`evicted`): a demoted point stays a state, on the SSD.
  // Behind the lanes, of each running lane's chain its newest state, of
  // those the releases before it leave, which the lane resumes from should
  // it be preempted.
  std::unordered_set<uint64_t> laneKept;
  if (stage == TailRelease::BehindLanes)
    for (const auto &[_, active] : requests_) {
      const std::vector<uint64_t> &chain = active.cachedBlocks;
      const auto holds = [&](uint64_t block) {
        ++searchVisits_;
        return states_.contains(block) && !evicted.contains(block);
      };
      if (const auto newest = std::find_if(chain.rbegin(), chain.rend(), holds); newest != chain.rend())
        laneKept.insert(*newest);
    }
  // The resume point once the releases before have gone: the newest
  // ordinary state in RAM left, else the newest checkpoint
  // (StateCache::resumePoint), found from the newest down.
  uint64_t resumePoint = 0;
  for (const bool checkpoint : {false, true})
    for (auto state = states_.newestCandidate(checkpoint); state && !resumePoint && walk.keepResumePoint;
         state = states_.newestCandidate(checkpoint, state)) {
      ++searchVisits_;
      resumePoint = released.contains(state->id) ? 0 : state->id;
    }
  // A point only on the SSD holds units only while their window parts are
  // not written, which a reclaim writes (Unit::Unwritten): it leaves for
  // them, with its copy, where its copy does not stay (copyStays). The
  // points the count's releases left on the SSD alone are offered as such.
  const bool countDemotes = supersededMemo_ && supersededMemo_->gone == &evicted;
  // Whether a state is a candidate of the stage: of the checkpoints, the
  // ordinary states and the only copies on the SSD, none in use, pinned,
  // kept by the walk (the certain reuse its claim may not take) or
  // `released` already; at History the superseded ones, behind the lanes
  // all but those the stage keeps of each running lane's chain; and how its
  // release takes it, which writes only for a walk that writes and waits.
  // Each judged once a call: nothing changes within it.
  std::unordered_map<uint64_t, std::optional<ReleaseMode>> judged;
  const auto candidate = [&](uint64_t id) -> std::optional<ReleaseMode> {
    const auto [at, added] = judged.try_emplace(id);
    if (!added)
      return at->second;
    ++searchVisits_;
    const StateCache::Order order = states_.orderOf(id);
    bool onlyOnDisk = false;
    if (order == StateCache::Order::None)
      return std::nullopt;
    if (order == StateCache::Order::DiskOnly) {
      if (copyStays(id, evicted))
        return std::nullopt;
    } else if (countDemotes && supersededMemo_->demoted.contains(id) && !evicted.contains(id)) {
      onlyOnDisk = !copyStays(id, evicted);
    }
    if ((released.contains(id) && !onlyOnDisk) || laneKept.contains(id) || walk.keepsState(id) ||
        (walk.keepResumePoint && id == resumePoint) || (stage == TailRelease::History && !superseded(id, evicted)))
      return std::nullopt;
    const ReleaseMode mode = releaseMode(id, evicted);
    if (!writes && mode.writes)
      return std::nullopt;
    return judged[id] = mode;
  };
  const uint32_t windowPages = window_->windowPages;
  // A block whose unit the candidates' releases free together, or leave to
  // a write of its window part: every pin on it, beside those `gone`, a
  // candidate's, the candidates whose tails hold it being the states within
  // a window below it (unitPoints), whose releases take `memory` RAM pins
  // off it and change its disk pins by `disk`.
  const auto heldBlock = [&](uint64_t block) {
    const std::span<const uint32_t> window = kv_.window(block);
    return window.empty() || kv_.transfer(block) != KvCache::Transfer::None ||
           std::ranges::any_of(window, [&](uint32_t page) { return pool_.pageActive(page); }) ||
           (stage != TailRelease::History && !pool_.pageActive(kv_.page(block)));
  };
  const auto pinsOf = [&](uint64_t id) {
    const std::optional<ReleaseMode> mode = candidate(id);
    if (!mode)
      return std::pair<uint32_t, int32_t>{};
    const KvCache::PinRelease release = pinRelease(id, mode->evict);
    return std::pair<uint32_t, int32_t>{release.ram ? 1 : 0, (release.written ? 1 : 0) - (release.disk ? 1 : 0)};
  };
  const auto freesWith = [&](uint64_t block, uint32_t memory, int32_t disk) {
    const auto left = gone.find(block);
    const auto [goneMemory, goneDisk] = left == gone.end() ? std::pair<uint32_t, int32_t>{} : left->second;
    const int64_t disks = static_cast<int64_t>(kv_.copyPins(block)) + goneDisk + disk;
    return kv_.windowPins(block) - goneMemory == memory && (writes || disks <= 0 || kv_.windowCopied(block));
  };
  // Under test, the block judged by its points walked afresh (checkTails).
  const auto freeable = [&](uint64_t block) {
    if (heldBlock(block))
      return false;
    uint32_t memory = 0;
    int32_t disk = 0;
    for (const uint64_t id : unitPoints(block)) {
      const auto [ram, disks] = pinsOf(id);
      memory += ram;
      disk += disks;
    }
    return freesWith(block, memory, disk);
  };
  // Whether the state's tail holds a block it frees, going up from the state
  // a block at a time. The states within a window below each block are those
  // below the block before it, a block nearer, those of its other children's
  // subtrees within the window and the block itself: kept as the walk goes
  // up, each visited once, rather than walked afresh at each block, so a
  // tail costs its window and the branches beside it, not a window of
  // windows. A subtree is gathered, and a point's candidacy judged, only
  // once the walk asks a block of them, within the window of that block, as
  // a walk afresh would.
  struct Point final {
    uint64_t id = 0;
    int64_t depth = 0;
    bool judged = false;
    bool live = true;
    uint32_t memory = 0;
    int32_t disk = 0;
  };
  std::vector<Point> points;
  std::vector<size_t> deepest, unjudged;
  std::vector<std::pair<uint64_t, int64_t>> ungathered;
  const auto tailFrees = [&](uint64_t state) {
    points.clear();
    deepest.clear();
    unjudged.clear();
    ungathered.clear();
    uint32_t memory = 0;
    int32_t disk = 0;
    const auto byDepth = [&](size_t a, size_t b) { return points[a].depth < points[b].depth; };
    // The states of the subtree at `from` (`depth` below the state) within
    // `room` blocks of it.
    const auto gather = [&](uint64_t from, int64_t depth, int64_t room) {
      for (std::vector<std::pair<uint64_t, uint32_t>> pending{{from, 0}}; !pending.empty();) {
        const auto [id, below] = pending.back();
        pending.pop_back();
        ++searchVisits_;
        if (states_.contains(id)) {
          points.push_back({.id = id, .depth = depth + below});
          deepest.push_back(points.size() - 1);
          std::ranges::push_heap(deepest, byDepth);
          unjudged.push_back(points.size() - 1);
        }
        if (below + 1 < room)
          for (const uint64_t child : kv_.children(id))
            pending.emplace_back(child, below + 1);
      }
    };
    uint64_t block = state, came = 0;
    for (uint32_t index = 0; block && index < windowPages; ++index, ++searchVisits_) {
      if (!came) {
        ungathered.emplace_back(block, 0);
      } else {
        if (states_.contains(block)) {
          points.push_back({.id = block, .depth = -int64_t{index}});
          deepest.push_back(points.size() - 1);
          std::ranges::push_heap(deepest, byDepth);
          unjudged.push_back(points.size() - 1);
        }
        for (const uint64_t child : kv_.children(block))
          if (child != came)
            ungathered.emplace_back(child, 1 - int64_t{index});
        // Those a window from this block no longer reaches.
        while (!deepest.empty() && points[deepest.front()].depth + index >= windowPages) {
          Point &out = points[deepest.front()];
          out.live = false;
          memory -= out.memory;
          disk -= out.disk;
          std::ranges::pop_heap(deepest, byDepth);
          deepest.pop_back();
        }
      }
      bool frees = false;
      if (!heldBlock(block)) {
        // Within a window of this block: `depth` + below + index < window.
        for (const auto &[from, depth] : ungathered)
          if (const int64_t room = int64_t{windowPages} - index - depth; room > 0)
            gather(from, depth, room);
        ungathered.clear();
        for (const size_t at : unjudged)
          if (Point &each = points[at]; each.live && !each.judged) {
            each.judged = true;
            std::tie(each.memory, each.disk) = pinsOf(each.id);
            memory += each.memory;
            disk += each.disk;
          }
        unjudged.clear();
        frees = freesWith(block, memory, disk);
      }
      if (checkTails_) {
        const uint64_t visits = searchVisits_;
        const bool afresh = freeable(block);
        searchVisits_ = visits;
        if (afresh != frees)
          throw std::logic_error("a tail release's search judged a block other than its points walked afresh do");
      }
      if (frees)
        return true;
      came = block;
      block = kv_.key(block).parent;
    }
    return false;
  };
  // The candidates oldest first past `after`, the checkpoints, the ordinary
  // states and the only copies on the SSD in one order, to the first whose
  // tail holds a block it frees.
  for (std::optional<CacheEvictionCandidate> next = after;;) {
    std::optional<CacheEvictionCandidate> oldest;
    for (const std::optional<CacheEvictionCandidate> &each :
         {states_.checkpointCandidate(next), states_.ordinaryCandidate(next), states_.diskCandidate(false, next)})
      if (each && (!oldest || std::pair{each->lastUsed, each->id} < std::pair{oldest->lastUsed, oldest->id}))
        oldest = each;
    if (!oldest)
      return std::nullopt;
    next = oldest;
    if (candidate(oldest->id) && tailFrees(oldest->id))
      return oldest;
  }
}

Cache::ReleasedUnits Cache::releasedUnits(const Walk &walk, ReclaimClass upTo, uint64_t staged,
                                          const std::function<bool(const ReleasedUnits &)> &enough) const {
  ReleasedUnits units;
  if (!window_ || !kv_.unitPages())
    return units;
  const Walk now = phaseWalk(walk, upTo == ReclaimClass::InUse ? ReclaimPhase::UsedNow : ReclaimPhase::Now);
  KvCache::Gone gone;
  std::unordered_set<uint64_t> released, evicted;
  // What the releases so far free, each unit once, whichever releases left
  // it to a write (a release with no write and a later one may both), and
  // none a later release frees now.
  const auto counted = [&](ReleasedUnits each) {
    each.states = released;
    std::unordered_set<uint64_t> once(each.now.begin(), each.now.end());
    searchVisits_ += each.now.size() + each.written.size();
    std::erase_if(each.written, [&](uint64_t block) { return !once.insert(block).second; });
    return each;
  };
  // A count stops once what it has counted covers the need (Cache::reach),
  // asked before the first release and after the 1st, 4th, 16th ...: the
  // prefix it counted is exact, and asking costs what was counted so far.
  if (enough && enough(units))
    return units;
  size_t releases = 0, nextAsk = 1;
  bool stop = false;
  // A count's own answers against what it evicts; those against nothing
  // gone, of chains landed and of lanes' deepest points stay with the peek.
  const MemoScope scope{*this};
  SupersededMemo &memo = *supersededMemo_;
  memo.gone = &evicted;
  memo.seen.clear();
  memo.answers.clear();
  memo.walked.clear();
  memo.passed.clear();
  memo.demoted.clear();
  memo.copied.clear();
  struct GoneEnd final {
    SupersededMemo &memo;
    ~GoneEnd() {
      memo.gone = nullptr;
      memo.copied.clear();
    }
  } const goneEnd{memo};
  // Its units with no write free now; those a write in flight moves free as
  // it lands, which a walk reaches whatever keeps their base pages.
  const auto release = [&](uint64_t block, bool evict) {
    released.insert(block);
    // A release with no write keeps the state only through its disk copy,
    // a point only on the SSD from then on; a release of such a point
    // evicts it.
    evict = evict || countDemoted(block);
    const KvCache::Freed freed = kv_.freedBy(block, gone, pinRelease(block, evict));
    // The point each block spaces its writes from (copiedBelow) passes over
    // the points that leave without a copy, as far down its chain as they
    // go: an eviction may change it for a block however far up (the history
    // it leaves below it, Cache::superseded), so none of those found before
    // stands. A demotion changes none: whether a point's copy stays is the
    // same before and after its release (copyStays).
    if (evict || !states_.onDisk(block)) {
      evicted.insert(block);
      memo.copied.clear();
    } else if (states_.stateResident(block)) {
      memo.demoted.insert(block);
    }
    units.now.insert(units.now.end(), freed.now.begin(), freed.now.end());
    units.written.insert(units.written.end(), freed.landing.begin(), freed.landing.end());
    // The units it leaves held for its disk copy alone, which a walk that
    // writes and waits writes (Unit::Tail, Unit::Unwritten).
    if (walk.writes && walk.timing == ReclaimTiming::CanWait && takesCacheWrites())
      units.written.insert(units.written.end(), freed.written.begin(), freed.written.end());
    if (enough && ++releases == nextAsk) {
      nextAsk *= 4;
      stop = enough(counted(units));
    }
  };
  // A state a reclaim of the walk takes with no write, as reclaimState
  // goes: one that leaves now and is history, which goes with its copy
  // where keeping it would wait for a write, or has none, which its reclaim
  // drops (StateCache::reclaim).
  const auto drops = [&](uint64_t block) {
    const StateReclaim reclaim = stateReclaim(now, block, 0, evicted);
    return reclaim.when == StateReclaim::When::Now && !reclaim.write;
  };
  const auto evicts = [&](uint64_t block) {
    return releaseMode(block, evicted).evict || !states_.onDisk(block);
  };
  const auto leftOver = [&](uint64_t block) { return !released.contains(block); };
  // The checkpoints no lane recovers from.
  const auto checkpoints = [&] {
    for (auto state = states_.checkpointCandidate(); state && !stop;
         state = states_.checkpointCandidate(state), ++searchVisits_)
      if (leftOver(state->id) && drops(state->id) && !kept({Unit::Checkpoint, *state}, now) &&
          !recoveryPoint(state->id))
        release(state->id, evicts(state->id));
  };
  // A phase of states by use, as the walk goes over it (ordinaryTakes): the
  // history that serves the need; the states by use, those whose leaving
  // loses nothing first (WalkCursor::lost), the running lanes' recovery
  // points among the lost; and what is in use.
  const auto byUse = [&](bool inUse, bool history) {
    const auto next = [&](std::optional<CacheEvictionCandidate> each) {
      return inUse ? states_.inUseCandidate(each) : states_.ordinaryCandidate(each);
    };
    for (const bool lost : {false, true}) {
      if (lost && history)
        return;
      if (lost && !inUse)
        for (auto state = states_.checkpointCandidate(); state && !stop;
             state = states_.checkpointCandidate(state), ++searchVisits_)
          if (leftOver(state->id) && recoveryPoint(state->id) && drops(state->id) &&
              !kept({Unit::Checkpoint, *state}, now))
            release(state->id, evicts(state->id));
      for (auto state = next(std::nullopt); state && !stop; state = next(state), ++searchVisits_)
        if (leftOver(state->id) && (history || lostOnLeaving(state->id) == lost) && drops(state->id) &&
            !kept({inUse ? Unit::UsedState : Unit::State, *state}, now) &&
            (inUse || ordinaryTakes(now, state->id, history, released, gone, evicted)))
          release(state->id, evicts(state->id));
    }
  };
  // The tails' releases a stage at a time, each past the one before in the
  // stage's order, judged on those before: one can make the next no
  // history, or free what the next was to.
  const auto stageReleases = [&](const Walk &stepWalk, TailRelease stage, const auto &take) {
    for (std::optional<CacheEvictionCandidate> last; !stop;) {
      last = nextTailRelease(stepWalk, stage, last, released, evicted, gone);
      if (!last)
        return;
      take(last->id);
    }
  };
  // The walk's phases in its order (phaseAfter), those of states.
  using Phase = WalkCursor::Phase;
  for (Phase phase = Phase::Checkpoints; phase != Phase::Done && !stop; phase = phaseAfter(now, phase))
    switch (phase) {
    case Phase::Checkpoints:
      checkpoints();
      break;
    case Phase::History:
    case Phase::ByUse:
      byUse(false, phase == Phase::History);
      break;
    case Phase::InUse:
      if (upTo == ReclaimClass::InUse)
        byUse(true, false);
      break;
    case Phase::Tails:
      for (const TailRelease stage : {TailRelease::History, TailRelease::BehindLanes})
        stageReleases(now, stage, [&](uint64_t block) { release(block, releaseMode(block, evicted).evict); });
      break;
    default:
      break;
    }
  // Stopped: the phases that write free past what covers the need.
  if (stop)
    return counted(std::move(units));
  units.states = released;
  // A release that writes: what it frees as it starts, and leaves to the
  // writes of window parts or to the landing of writes in flight.
  const auto writes = [&](uint64_t block) {
    released.insert(block);
    const KvCache::Freed freed = kv_.freedBy(block, gone, pinRelease(block, false));
    units.written.insert(units.written.end(), freed.written.begin(), freed.written.end());
    units.written.insert(units.written.end(), freed.landing.begin(), freed.landing.end());
    return freed;
  };
  // The staged state's write, whose units with their window parts free as
  // its RAM copy leaves; those still in flight land with the rest
  // (landingOnce).
  if (staged && !released.contains(staged))
    units.staged = writes(staged).now;
  // A walk that writes and waits: the tails' demotions, then the states that
  // wait for the staging buffer.
  if (walk.writes && walk.timing == ReclaimTiming::CanWait) {
    const auto later = [&](uint64_t block) {
      const KvCache::Freed freed = writes(block);
      units.written.insert(units.written.end(), freed.now.begin(), freed.now.end());
    };
    const Walk wait = phaseWalk(walk, upTo == ReclaimClass::InUse ? ReclaimPhase::UsedLater : ReclaimPhase::Later);
    stageReleases(wait, TailRelease::BehindLanes, later);
    const auto waits = [&](uint64_t block, Unit unit) {
      return !released.contains(block) && !kept({unit, {block, 0}}, wait) && states_.reclaimWrites(block);
    };
    for (auto state = states_.checkpointCandidate(); state; state = states_.checkpointCandidate(state), ++searchVisits_)
      if (waits(state->id, Unit::Checkpoint))
        later(state->id);
    for (auto state = states_.ordinaryCandidate(); state; state = states_.ordinaryCandidate(state), ++searchVisits_)
      if (waits(state->id, Unit::State))
        later(state->id);
    if (upTo == ReclaimClass::InUse)
      for (auto state = states_.inUseCandidate(); state; state = states_.inUseCandidate(state), ++searchVisits_)
        if (waits(state->id, Unit::UsedState))
          later(state->id);
  }
  // Each unit once, whichever releases left it to a write (a release with
  // no write and a later one may both), and none a later release frees now.
  std::unordered_set<uint64_t> once(units.now.begin(), units.now.end());
  std::erase_if(units.written, [&](uint64_t block) { return !once.insert(block).second; });
  return units;
}

std::vector<uint64_t> Cache::unitPoints(uint64_t block) const {
  std::vector<uint64_t> points;
  for (std::vector<std::pair<uint64_t, uint32_t>> pending{{block, 0}}; !pending.empty();) {
    const auto [id, depth] = pending.back();
    pending.pop_back();
    ++searchVisits_;
    if (states_.contains(id))
      points.push_back(id);
    if (depth + 1 < window_->windowPages)
      for (const uint64_t child : kv_.children(id))
        pending.emplace_back(child, depth + 1);
  }
  return points;
}

bool Cache::unitTakes(const Walk &walk, uint64_t block) const {
  return std::ranges::none_of(unitPoints(block), [&](uint64_t id) {
    return (states_.inUse(id) && walk.scope.upTo != ReclaimClass::InUse) || states_.pinned(id) ||
           kept({Unit::State, {id, 0}}, walk);
  });
}

std::vector<uint64_t> Cache::unitsKept(const Walk &walk) const {
  std::vector<uint64_t> blocks;
  if (!window_ || !kv_.unitPages() || !takesCacheWrites())
    return blocks;
  for (auto unit = kv_.unwrittenUnitCandidate(0); unit; unit = kv_.unwrittenUnitCandidate(unit->id))
    if (!unitTakes(walk, unit->id))
      blocks.push_back(unit->id);
  return blocks;
}

bool Cache::dropUnitPoints(const Walk &walk, uint64_t block) {
  if (walk.timing == ReclaimTiming::Immediate || !kv_.unwrittenUnit(block) || !unitTakes(walk, block))
    return false;
  // The states at the block and below it, within a window of it, whose
  // tails hold it: all on the SSD alone, as its unit is unwritten. They
  // leave through the one exit, as every state a reclaim gives up, given up
  // as the quota or the tier refused the write, which no count foresees.
  for (const uint64_t id : unitPoints(block))
    static_cast<void>(releaseState(id, true, false));
  return !kv_.holdsUnit(block);
}

Cache::LeafReclaim Cache::writeUnit(const Walk &walk, uint64_t block) {
  if (!takesCacheWrites() || walk.timing == ReclaimTiming::Immediate || !walk.writes)
    return LeafReclaim::Impossible;
  const auto room = [&] { return transfersInFlight(walk) ? LeafReclaim::Pending : LeafReclaim::Impossible; };
  if (!tier_->canDemote(WriteClass::Reclaim)) {
    ++kvTier_.demotionsRefused;
    return room();
  }
  // Its window part, with its base part where its copy lacks one: a
  // persistent tier pairs a window part with its base part.
  const auto [parts, into] = copyParts(block);
  const KvCache::PassOver ownCopy{kv_, block};
  // A unit only history needs is written at the worth of history
  // (writeWorth).
  const std::vector<uint64_t> points = unitPoints(block);
  const bool history = std::ranges::all_of(points, [&](uint64_t id) { return writeWorth(id).rank == 1; });
  std::shared_ptr<KvDiskSlot> slot =
      acquireDiskSlot(parts, model::DiskBudget::Draw::Cache, kvNeededByStateInUse(block), false, into,
                      {.rank = static_cast<uint8_t>(history ? 1 : 2), .lastUse = kv_.lastUsed(block)});
  // Making room may have taken the points the unit was kept for, and the
  // unit with them. (A slot acquired into the block's copy is its window
  // part already.)
  if (!kv_.holdsUnit(block) || !kv_.copyPins(block) || kv_.windowPins(block)) {
    giveSlotBack(slot, parts, into != nullptr);
    return kv_.holdsUnit(block) ? LeafReclaim::Impossible : LeafReclaim::Started;
  }
  // A quota that refused the write leaves the unit to go without a copy.
  if (!slot && quotaRefused_) {
    ++kvTier_.demotionsRefused;
    return LeafReclaim::Impossible;
  }
  if (!slot)
    return room();
  if (!startKvWrite(block, std::move(slot), parts, into != nullptr, WriteClass::Reclaim, true)) {
    ++kvTier_.demotionsRefused;
    return room();
  }
  return LeafReclaim::Started;
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
  case Unit::Tail: {
    // A demotion writes the window parts of the units it leaves held for
    // the SSD alone, which the walk needs now.
    const ReleaseMode mode = releaseMode(block);
    KvCache::Gone gone;
    // An eviction leaves what other points still need to the walk's later
    // units (Unit::Unwritten).
    const std::vector<uint64_t> written =
        mode.evict ? std::vector<uint64_t>{} : kv_.freedBy(block, gone, pinRelease(block, false)).written;
    const StateEviction eviction = releaseState(block, mode.evict);
    bool pending = eviction.pending;
    for (const uint64_t each : written)
      if (eviction.evicted && kv_.contains(each) && kv_.unwrittenUnit(each))
        pending = writeUnit(walk, each) == LeafReclaim::Pending || pending;
    return {eviction.evicted, pending && !eviction.evicted};
  }
  case Unit::Unwritten:
    switch (writeUnit(walk, block)) {
    case LeafReclaim::Started:
      return {true, false};
    case LeafReclaim::Pending:
      return {false, true};
    case LeafReclaim::Impossible:
      // A unit the tier cannot take with nothing in flight goes with the
      // points on the SSD that need it, as a leaf that cannot be written
      // goes with what lies below it.
      return {dropUnitPoints(walk, block), false};
    }
    return {};
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

uint32_t Cache::landingExtents(const Walk &walk, kv::PageForm form, uint32_t landing) const noexcept {
  const uint32_t free = pool_.freePageCount(form);
  return completedExtents(free - std::min(free, walk.keptPages[static_cast<size_t>(form)]), landing,
                          pool_.extentPages(), walk.keeps(form));
}

uint64_t Cache::landingFill(const Walk &walk) const noexcept {
  if (walk.timing == ReclaimTiming::Immediate || keepsExtents_)
    return 0;
  const std::array<uint32_t, kv::kPageForms> landing = demotedPages();
  uint64_t bytes = 0;
  // A need in place keeps the pages of its form as they land (reach).
  for (const kv::PageForm form : kv::kAllPageForms)
    bytes += walk.inPlace && form == walk.form
                 ? 0
                 : uint64_t{landingExtents(walk, form, landing[static_cast<size_t>(form)])} * pool_.extentPages() *
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
  ReclaimReach taken;
  taken.now.states = states.now;
  taken.written.states = states.staged ? 1 : 0;
  taken.later.states = states.later;
  // What holds a page and its chain above it in place, which frees later: a
  // block in use or in transfer, a page a request holds; a state that needs
  // a write, which stays in RAM until its write starts and needs its chain
  // on the SSD after; with a writable tier, a block without a copy that a
  // state that stays needs; and the chains the walk keeps, which the KV
  // cache counts as they change (keptChains), but for the chain a start
  // reuses, which it walks (KvCache::pageReach). Once the staged state's
  // write has started, its chain holds only its blocks without a copy.
  KvCache::Keeps mask = KvCache::keepBit(Keep::Busy) | KvCache::keepBit(Keep::Moving) | keptChains(walk, upTo);
  if (takesCacheWrites())
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
  std::array<uint32_t, kv::kPageForms> pinned{};
  {
    std::unordered_set<uint32_t> extentsPinned;
    const auto pin = [&](uint64_t block) {
      if (kv_.page(block) == KvCache::noPage)
        return;
      for (const uint32_t page : kv_.pages(block))
        if (extentsPinned.insert(page / pool_.extentPages()).second)
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
  // The units the walk's releases of states free with no write, and those
  // its demotions free once they write.
  const std::vector<uint64_t> unitsKeptHere = unitsKept(phaseWalk(walk, inUse ? ReclaimPhase::UsedLater : ReclaimPhase::Later));
  // What the releases counted free, priced: the pages and extents of each
  // phase, with the states the walk takes.
  const auto price = [&](const ReleasedUnits &released) {
    ReclaimReach counted = taken;
    const KvCache::PageReach pages = kv_.pageReach(mask, reused, unwaited, states.staged, released.now, released.written,
                                                   takesCacheWrites(), released.staged, unitsKeptHere);
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
    const std::array<uint32_t, kv::kPageForms> landing = demotedPages();
    const auto landingOnce = [&](bool stagedGone) {
      std::array<uint32_t, kv::kPageForms> pages = landing;
      if (walk.timing != ReclaimTiming::CanWait)
        return pages;
      const std::unordered_set<uint64_t> already = freeingDemotions();
      std::unordered_set<uint64_t> freeing;
      const Walk first = phaseWalk(walk, inUse ? ReclaimPhase::UsedNow : ReclaimPhase::Now);
      const auto stateGoes = [&](uint64_t block) {
        return !states_.stateResident(block) || (stagedGone && block == states.staged) ||
               (states_.reclaimable(block) && (inUse || !states_.inUse(block)) &&
                std::ranges::find(kept, block) == kept.end() &&
                stateReclaim(first, block).when == StateReclaim::When::Now);
      };
      // A unit write's unit lands once every state in memory whose tail holds
      // it goes: those at its block and below it, within a window of it.
      const auto unitLands = [&](uint64_t block) {
        if (kv_.unitLeaving(block))
          return false;
        for (std::vector<std::pair<uint64_t, uint32_t>> pending{{block, 0}}; !pending.empty(); ++searchVisits_) {
          const auto [id, depth] = pending.back();
          pending.pop_back();
          if (states_.stateResident(id) && !stateGoes(id) && !released.states.contains(id))
            return false;
          if (depth + 1 < window_->windowPages)
            for (const uint64_t child : kv_.children(id))
              pending.emplace_back(child, depth + 1);
        }
        return true;
      };
      for (const Write &write : writes_)
        if (write.unit && unitLands(write.block))
          pages[static_cast<size_t>(pool_.form(kv_.page(write.block)))] += kv_.unitPages();
      for (const Write &write : writes_) {
        const uint64_t block = write.block;
        // Those freeing already, counted in demotedPages, in their place.
        if (write.write == WriteClass::Reclaim && !write.unit && already.contains(block)) {
          freeing.insert(block);
          continue;
        }
        if (write.write != WriteClass::Reclaim || write.unit ||
            !kv_.leafOnceChildrenLeave(block) ||
            pool_.pageActive(kv_.page(block)) || !stateGoes(block) ||
            !std::ranges::all_of(kv_.children(block), [&](uint64_t child) {
              return kv_.page(child) == KvCache::noPage || freeing.contains(child) || !(kv_.keeps(child) & mask);
            }))
          continue;
        freeing.insert(block);
        pages[static_cast<size_t>(pool_.form(kv_.page(block)))] += kv_.pagesHeld(block);
      }
      return pages;
    };
    const std::array<uint32_t, kv::kPageForms> landingNow = landingOnce(false);
    const std::array<uint32_t, kv::kPageForms> landingWritten = states.staged ? landingOnce(true) : landingNow;
    // Of each form, the pages a reclaim of the class reaches (the free ones
    // and those reachable), that free now (the free ones and those nothing
    // holds), once the staged state's write has started (the written ones
    // too) or later (the rest).
    std::array<uint32_t, kv::kPageForms> now{}, written{}, all{};
    for (const kv::PageForm form : kv::kAllPageForms) {
      const auto index = static_cast<size_t>(form);
      const uint32_t allocated = pool_.allocatedExtents(form) * extent;
      all[index] = std::min(allocated, pool_.freePageCount(form) + pages.reachable[index]);
      now[index] = std::min(all[index], pool_.freePageCount(form) + pages.now[index]);
      written[index] = std::min(all[index], pool_.freePageCount(form) + pages.now[index] + pages.written[index]);
      // Compaction gathers the pages that stay into the fewest extents, and
      // the rest empty, but for the free pages a KV need takes and the
      // runway the walk keeps. A need in place empties none of its form, nor
      // counts landing pages there as extents: it keeps every page of its
      // form it frees, landing or not (takeReclaim, releaseExtent,
      // landingFill), which its pages count.
      if (walk.inPlace && form == walk.form)
        continue;
      const uint64_t extentBytes = uint64_t{extent} * pool_.bytesPerPage(form);
      const auto extents = [&](uint32_t pages) {
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
        return std::max(released, std::min(completedExtents(0, pages - std::min(pages, walk.keptPages[index]) + lands, extent, walk.keeps(form)),
                                           atOnce(pages) + (lands + extent - 1) / extent));
      };
      const uint32_t throughNow = through(now[index], landingNow[index]);
      const uint32_t throughWritten = std::max(throughNow, through(written[index], landingWritten[index]));
      counted.now.bytes += (throughNow - std::min(throughNow, landed)) * extentBytes;
      counted.written.bytes += (throughWritten - throughNow) * extentBytes;
      counted.later.bytes += (extents(all[index]) - std::min(extents(all[index]), throughWritten)) * extentBytes;
    }
    const auto need = static_cast<size_t>(walk.form);
    const uint32_t pagesNow = now[need], pagesWritten = written[need], pagesAll = all[need];
    const uint32_t free = pool_.freePageCount(walk.form);
    counted.now.pages = pagesNow - std::min(pagesNow, free);
    counted.written.pages = pagesWritten - std::min(pagesWritten, std::max(pagesNow, free));
    counted.later.pages = pagesAll - std::min(pagesAll, std::max(pagesWritten, free) + landing[need]);
    return counted;
  };
  // A count stops once its phase that frees now covers the need and the
  // most one step of a take frees past it: a take frees no more than the
  // units through it counted (VictimSource::peek), and takes steps until
  // they free what the claim lacks, no more than the need, the last of
  // them no more than one step's most (stepMost) past it. A step may free
  // the units of a release the count did not simulate, but no more than
  // its most, which the count covers. A count that covers neither runs to
  // the end as before; stopped, the units of the phases that write are
  // those of the releases counted.
  Walk pagesOf = walk;
  pagesOf.device = false;
  const uint64_t stepPages = stepMost(pagesOf), stepBytes = stepMost(walk);
  const auto covers = [&](const ReleasedUnits &units) {
    if (!walk.device && !walk.pages)
      return false;
    const ReclaimReach::Amount now = price(units).now;
    // A device need in place is met by pages of its form, whose extents
    // stay (MemoryLedger: a need met in place lacks no bytes).
    if (walk.device && walk.inPlace)
      return now.pages >= walk.inPlacePages + stepPages;
    if (walk.device)
      return now.bytes >= walk.bytes + stepBytes;
    return now.pages >= walk.pages + stepPages;
  };
  using Enough = std::function<bool(const ReleasedUnits &)>;
  return price(releasedUnits(walk, upTo, states.staged, countsBounded_ ? Enough{covers} : Enough{}));
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
  const MemoScope scope{*this};
  const Walk walk = walkFor(query);
  const Unit afterUnit = after ? static_cast<Unit>(unitOfKey(after->key)) : Unit{};
  if (walk.device || walk.pages) {
    // A page need's free pages first: the other forms' empty extents.
    if (walk.pages && !after && spill_) {
      // The empty extents of the other forms beyond the runway each keeps.
      uint64_t bytes = 0;
      for (const kv::PageForm other : kv::kAllPageForms)
        if (const uint32_t count = pool_.emptyExtents(other); other != walk.form && count > 1)
          bytes += uint64_t{count - 1} * pool_.extentPages() * pool_.bytesPerPage(other);
      if (bytes)
        return engine::Victim{.key = unitKey(static_cast<uint8_t>(Unit::Extent), 0), .bytes = bytes};
    }
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

FreeResult Cache::reclaimStep(const Walk &walk, ReclaimPhase phase) {
  // One reclaim of the unit's phase: the state the staging buffer writes
  // (recycleState), else a device need's memory reclaim (takeReclaim) or a
  // page need's eviction. What it freed now, the extents its demotions
  // complete as they land as peek counts them (landingFill), and the pages
  // of the need's form, free or landing; the take settles what lands over
  // all its steps (Cache::take).
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
  if (!reclaimed.madeProgress && !reclaimed.evictedKv)
    return {reclaimed.pending ? FreeOutcome::Pending : FreeOutcome::Short};
  const uint32_t pagesAfter = pages();
  const uint64_t fillAfter = walk.device ? landingFill(walk) : 0;
  return {.outcome = fillAfter > fillBefore || landingPages() > landing ? FreeOutcome::Pending : FreeOutcome::Met,
          .nowBytes = reclaimed.reclaimedBytes,
          .laterBytes = fillAfter - std::min(fillAfter, fillBefore),
          .pages = pagesAfter - std::min(pagesAfter, pagesBefore),
          .stateBuffers = reclaimed.evictedState ? 1u : 0u};
}

uint64_t Cache::unitMost(const WalkUnit &unit) const {
  // Asked in the first phase only (Cache::take's budget), whose walk is
  // Immediate and writes nothing: a state's take there releases it with no
  // write, dropping its RAM copy or the state (reclaimState, releaseState),
  // which frees exactly the units freedBy names (releaseState checks it) and
  // erases no block: a leaf it leaves dead goes in a later step as a unit of
  // its own; no write starts, so no unit lands and no copy is given up for
  // room. The extents a device need's step releases are other forms' bytes,
  // no page of the need's form. A page scan's state on a leaf goes with the
  // leaf (LeafState): its page and unit.
  const uint64_t block = unit.candidate.id;
  const uint32_t unitPages = window_ ? kv_.unitPages() : 0;
  const uint64_t own = 1 + (unitPages && kv_.holdsUnit(block) ? unitPages : 0);
  if (unit.unit == Unit::DeadLeaf || unit.unit == Unit::Leaf || unit.unit == Unit::UsedLeaf)
    return own;
  uint64_t units = 0;
  if (unitPages)
    for (const bool evict : {false, true}) {
      KvCache::Gone gone;
      const KvCache::Freed freed = kv_.freedBy(block, gone, pinRelease(block, evict));
      units = std::max<uint64_t>(units, freed.now.size() + freed.written.size() + freed.landing.size());
    }
  return units * unitPages + (kv_.residentLeaf(block) ? own : 0);
}

uint64_t Cache::stepMost(const Walk &walk) const {
  // A leaf with its unit, or a state's tail of units (or units left to
  // writes that land), and for a device need the extents those pages fill
  // with the free ones and the one a run of leaves empties (takeReclaim).
  const uint64_t pages =
      window_ ? uint64_t{window_->windowPages + 1} * kv_.unitPages() + 1 : uint64_t{1} + kv_.unitPages();
  if (!walk.device)
    return pages;
  const uint64_t extent = pool_.extentPages();
  uint64_t pageBytes = 0;
  for (const kv::PageForm form : kv::kAllPageForms)
    pageBytes = std::max<uint64_t>(pageBytes, pool_.bytesPerPage(form));
  return ((pages + extent - 1) / extent + 2) * extent * pageBytes;
}

FreeResult Cache::take(const VictimQuery &query, const engine::Victim &victim) {
  quotaGaveUpState_ = false;
  // A walk resumes only within the take that walked before it.
  resume_.reset();
  struct ResumeEnd final {
    std::optional<WalkResume> &resume;
    ~ResumeEnd() { resume.reset(); }
  } const resumeEnd{resume_};
  const Walk walk = walkFor(query);
  const uint64_t block = blockOfKey(victim.key);
  const auto met = [](uint64_t nowBytes = 0) { return FreeResult{FreeOutcome::Met, nowBytes}; };
  switch (static_cast<Unit>(unitOfKey(victim.key))) {
  case Unit::Extent:
    if (const uint64_t released = releaseOtherForms(walk.form))
      return met(released);
    return {};
  case Unit::Reclaim: {
    // Reclaims of the unit's phase, one at a time, until they free what the
    // claim lacks or the next could pass what the unit counted (stepMost):
    // what they free stays within the count, and a claim counts once for
    // its unit rather than once a step. They go on only while the ledger
    // would choose the same unit again after each (MemoryLedger::take): in
    // the first phase, of the ledger's first stage, whatever a step frees
    // but the other forms' empty extents, which peek offers before it; in a
    // later phase while no step frees a state, whose buffers would give the
    // model's pools units of an earlier stage and whose leaves, gone dead,
    // the cache's first phase. A step that gives nothing ends it, as does
    // one that frees a state a need of buffers counts; one that gave and
    // waits for what it started to land goes on, as the ledger goes on. The
    // staging buffer's write (Written) is one state, one step.
    // A device need in place is counted by pages of its form (reach): its
    // steps go on by pages, until they free the pages or the bytes it lacks
    // (the extents the pages it frees beyond those it keeps fill, which the
    // ledger counts as meeting it too); a state's buffers meet none of it.
    const auto phase = static_cast<ReclaimPhase>(block);
    const bool bytes = walk.device && !walk.inPlace;
    const uint64_t lacking = bytes ? query.lackingBytes : query.lackingPages;
    const uint64_t counted = bytes ? victim.bytes : victim.pages;
    Walk pagesOf = walk;
    pagesOf.device = bytes;
    // Even a count that ran to its end bounds one step from the cache it
    // counted, not a run of steps: a step may take pins off a tail whose
    // release the next step frees (take forked seed 12632). So past the
    // first step, which the count covers with its margin (reach), each step
    // of the first phase by pages takes a unit only if what that unit frees
    // at most (unitMost, the take's own predicate: freedBy for a state, the
    // page and its unit for a leaf, asked as it is reached, after the steps
    // before) fits what is left of the count (stepBudget_); the others, and
    // a device need's bytes, keep one step's most as their margin.
    const uint64_t most = stepMost(pagesOf);
    const bool budgeted = phase == ReclaimPhase::Now && !bytes;
    struct BudgetEnd final {
      std::optional<uint64_t> &budget;
      bool &keeps;
      ~BudgetEnd() {
        budget.reset();
        keeps = false;
      }
    } const budgetEnd{stepBudget_, keepsExtents_};
    const auto extentsElsewhere = [&] {
      uint32_t extents = 0;
      for (const kv::PageForm other : kv::kAllPageForms)
        extents += other != walk.form && spill_ && walk.pages ? pool_.emptyExtents(other) : 0;
      return extents;
    };
    const auto goesOn = [&](const FreeResult &step) {
      return step.gave() && countsBounded_ && phase != ReclaimPhase::Written && phase != ReclaimPhase::UsedWritten &&
             (phase == ReclaimPhase::Now || !step.stateBuffers) &&
             !(step.stateBuffers && (query.need.lacksStateBuffers() || bytes));
    };
    const uint32_t extentsBefore = extentsElsewhere();
    // What lands is settled once, over all the steps: the extents the
    // landing pages complete beside the free ones (landingFill) at the
    // take's start and at its end. A step that releases free pages a
    // landing was to complete an extent with cancels that landing only as
    // far as the take as a whole does, a later step's frees completing it
    // again (take plain seed 101189), and the claim's credit drops only as
    // the ledger's count of what lands does over the take (returning).
    const uint64_t fillStart = walk.device ? landingFill(walk) : 0;
    const auto credited = [&] { return walk.device && query.rights.waitsForLandings ? returning().landingBytes : 0; };
    const uint64_t creditedStart = credited();
    FreeResult taken = reclaimStep(walk, phase);
    // Under test, the answers the cache keeps after each step (checkKept).
    if (checkKept_)
      checkKept();
    for (FreeResult step = taken; goesOn(step);) {
      const uint64_t freed = bytes ? taken.nowBytes + taken.laterBytes : taken.pages;
      if (freed >= lacking || (budgeted ? freed >= counted : freed + most > counted) ||
          extentsElsewhere() > extentsBefore ||
          (walk.inPlace && query.lackingBytes && taken.nowBytes + taken.laterBytes >= query.lackingBytes))
        break;
      if (budgeted)
        stepBudget_ = counted - freed;
      // In place, a step past the first releases no extent and credits no
      // landing: what its unit frees is counted in pages (unitMost), and the
      // bytes the extents of other forms would give past the first step no
      // count bounded (take plain seed 644947: a second step's landing
      // completed an extent the count had credited once).
      keepsExtents_ = budgeted && walk.inPlace;
      step = reclaimStep(walk, phase);
      keepsExtents_ = false;
      if (checkKept_)
        checkKept();
      if (!step.gave())
        break;
      taken.outcome = step.outcome == FreeOutcome::Pending ? FreeOutcome::Pending : taken.outcome;
      taken.nowBytes += step.nowBytes;
      taken.laterBytes += step.laterBytes;
      taken.pages += step.pages;
      taken.stateBuffers += step.stateBuffers;
    }
    if (walk.device && taken.gave()) {
      // An extent released of free pages that landing pages would complete
      // cancels that landing as far as the claim was credited it, which is
      // not counted again: by its own takes (VictimQuery::landingCredited)
      // and, waiting for landings, by the ledger as much as its count
      // dropped over the take, the claim's own credit first
      // (FreeResult::nettedCredited). The rest cancels nothing of the
      // claim's, and the release gives its bytes now: a release beside
      // writes through that still complete the extent as they land, or a
      // host's beside a landing it was never credited (plain seed 11843).
      const uint64_t fillEnd = landingFill(walk), creditedEnd = credited();
      const uint64_t cancelled =
          std::min(fillStart - std::min(fillStart, fillEnd),
                   query.landingCredited + (creditedStart - std::min(creditedStart, creditedEnd)));
      const uint64_t filled = taken.nowBytes + (fillEnd - std::min(fillEnd, fillStart));
      const uint64_t gave = filled - std::min(filled, cancelled);
      const uint64_t now = std::min(taken.nowBytes, gave);
      taken.outcome = gave > now ? FreeOutcome::Pending : taken.outcome;
      taken.nowBytes = now;
      taken.laterBytes = gave - now;
      taken.nettedBytes = filled - gave;
      taken.nettedCredited = std::min(filled - gave, query.landingCredited);
    }
    return taken;
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
    for (uint64_t above = kv_.key(writes_[index].block).parent; above; above = kv_.key(above).parent, ++searchVisits_) {
      if (!kv_.contains(above) || kv_.transfer(above) != KvCache::Transfer::None || kv_.copyHoldsPages(above) ||
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

Cache::ReclaimResult Cache::evictOne(const Walk &walk, std::optional<std::array<bool, kv::kPageForms>> reach) {
  bool waited = false;
  if (const std::optional<Victim> victim = takeOldest(walk, reach, waited))
    return {.madeProgress = true, .evictedState = !victim->kv};
  // Nothing to reclaim now. Transfers land only in pollTransfers(), so what
  // was in flight during the pass still is, and comes back: what a unit
  // waited for, a demotion the tier takes once a write lands among them.
  return {.pending = waited || returns(walk)};
}

std::optional<Cache::Victim> Cache::takeOldest(const Walk &walk,
                                               std::optional<std::array<bool, kv::kPageForms>> reach, bool &waited) {
  using Phase = WalkCursor::Phase;
  WalkCursor cursor;
  // Within one take, a walk after one whose first unit was a leaf it took in
  // a phase past the tails starts where that walk was, its leaves over: taking
  // a leaf with no state at or below it, and freeing no state, adds no unit
  // before it. Not a checkpoint or a recovery point (no state changed, and
  // a search of history visits only blocks with a state below); not a tail
  // release (a leaf's unit is pinned only by a state at it); not a unit held
  // for a point on the SSD (a leaf's going only removes one). A parent left
  // a leaf no state needs is a dead leaf, and one that holds a state may
  // make it serve history: then, or with a state gone, it walks from the
  // head. So it does after anything made room on the SSD or started a write
  // (diskCalls_): a copy or a part given up, or a block in transfer, changes
  // which tails a release frees (nextTailRelease's freeable).
  // No transfer lands within a take (pollTransfers). The phase's states are
  // where they were (a leaf's going changes no state, nor which a phase
  // passes over), its leaves start over: a parent left a leaf may come
  // before the leaf taken.
  // A state the walk passes over because it needs a write and the walk
  // writes none stays passed over while the walk takes leaves: no unit
  // given before the leaf.
  const auto passedOver = [&](const WalkUnit &each) {
    const bool state = each.unit == Unit::Checkpoint || each.unit == Unit::State || each.unit == Unit::LeafState ||
                       each.unit == Unit::UsedState || each.unit == Unit::UsedLeafState;
    return state && !walk.writes && states_.reclaimWrites(each.candidate.id);
  };
  const auto parentChanged = [&](uint64_t parent) {
    return parent && kv_.contains(parent) &&
           (states_.contains(parent) || (kv_.residentLeaf(parent) && !kvNeededByState(parent)));
  };
  if (resume_ && resume_->walk == walk && resume_->reach == reach && resume_->states == states_.entryCount() &&
      resume_->disk == diskCalls_ && !parentChanged(resume_->parent)) {
    cursor = resume_->cursor;
    cursor.last.reset();
    cursor.restartLeaves = true;
    if (checkResumes_) {
      WalkCursor head, resumed = cursor;
      const auto firstTaken = [&](WalkCursor &from) {
        std::optional<WalkUnit> each = nextStepUnit(walk, from);
        while (each && passedOver(*each))
          each = nextStepUnit(walk, from);
        return each;
      };
      const std::optional<WalkUnit> first = firstTaken(head), next = firstTaken(resumed);
      if (first.has_value() != next.has_value() ||
          (first && (first->unit != next->unit || first->candidate.id != next->candidate.id)))
        throw std::logic_error("a take's walk resumed at a unit other than a walk from the head gives");
    }
  }
  resume_.reset();
  uint32_t given = 0;
  while (const std::optional<WalkUnit> unit = nextStepUnit(walk, cursor)) {
    ++searchVisits_;
    const Phase phase = cursor.phase;
    given += passedOver(*unit) ? 0 : 1;
    const bool leaf = unit->unit == Unit::DeadLeaf || unit->unit == Unit::Leaf || unit->unit == Unit::UsedLeaf;
    if (leaf && reach && !leadsToReach(unit->candidate.id, *reach))
      continue;
    // The leaves of each class wait for the tier on their own: a new phase
    // starts with a fresh cursor.
    const uint64_t parent = kv_.key(unit->candidate.id).parent;
    const size_t states = states_.entryCount();
    const bool held = cursor.demotionsHeld || cursor.statesHeld;
    const WalkCursor at = cursor;
    const uint64_t disk = diskCalls_;
    // Within a take's budget (stepBudget_), a unit that could free more than
    // is left of the count ends the step: the ledger counts again.
    const uint64_t most = stepBudget_ ? unitMost(*unit) : 0;
    if (stepBudget_ && most > *stepBudget_)
      return std::nullopt;
    // A release of a point on the SSD alone frees KV, no buffer.
    const bool buffers = !leaf && states_.stateResident(unit->candidate.id);
    const UnitTaken taken =
        takeUnit(walk, *unit, cursor.demotionsHeld ? ReclaimTiming::Immediate : walk.timing);
    if (taken.taken) {
      if (stepBudget_)
        *stepBudget_ -= most;
      if (leaf && !held && given == 1 && states == states_.entryCount() && disk == diskCalls_ &&
          (phase == Phase::ByUse || phase == Phase::InUse))
        resume_ = WalkResume{walk, reach, at, parent, states, disk};
      return Victim{!buffers};
    }
    if (!taken.pending)
      continue;
    waited = true;
    switch (unit->unit) {
    case Unit::Checkpoint:
      // One whose write must wait for the one in flight ends the
      // checkpoints, holding back nothing else; a recovery point, a state
      // by use, holds back the newer states of its phase.
      if (phase == Phase::Checkpoints) {
        cursor = WalkCursor{.phase = phaseAfter(walk, cursor.phase)};
        break;
      }
      [[fallthrough]];
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
  if (!spill_)
    return 0;
  const uint32_t all = std::numeric_limits<uint32_t>::max();
  uint64_t released = 0;
  for (const kv::PageForm other : kv::kAllPageForms)
    if (other != form)
      released += pool_.reclaimEmptyExtents(true, all, other);
  if (!released && form == kv::PageForm::Index && compactExtent(kv::PageForm::Full))
    released = pool_.reclaimEmptyExtents(true, all, kv::PageForm::Full);
  return released;
}

Cache::ReclaimResult Cache::takeReclaim(const Walk &walk, const std::array<uint32_t, kv::kPageForms> &demotions,
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
  // A state that went frees the window units only it held, whose pages may
  // fill extents, which go with it.
  const auto outcome = [&](ReclaimResult result) {
    result.evictedKv = evictedKv;
    if (result.evictedState)
      for (ReclaimResult released; (released = releaseExtent(walk)).madeProgress;)
        result.reclaimedBytes += released.reclaimedBytes;
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
  for (;; ++searchVisits_) {
    const std::array<bool, kv::kPageForms> reach = formsInReach(walk, demotions);
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
  // No form in reach: the walk's states alone, every leaf passed over, a
  // tail's release among them.
  return outcome(evictOne(walk, std::array<bool, kv::kPageForms>{}));
}

uint32_t Cache::reusablePages(ReclaimClass upTo, kv::PageForm form) const {
  const uint32_t unheld = pool_.reusablePages(form);
  if (upTo == ReclaimClass::InUse)
    return unheld;
  return unheld - kv_.keptPages(KvCache::keepBit(KvCache::Keep::Used))[static_cast<size_t>(form)];
}

Cache::ReclaimResult Cache::recycleState(const Walk &walk) {
  // A Written take writes the state its count staged (stagedState), the
  // first the staging buffer takes, which frees at once through the idle
  // buffer: the count and the take name one state. The ledger then peeks
  // again.
  if (const uint64_t staged = stagedState(walk); staged && states_.reclaimable(staged)) {
    const StateEviction eviction = reclaimState(walk, staged);
    if (eviction.evicted || eviction.pending)
      return {.madeProgress = eviction.evicted, .pending = eviction.pending, .evictedState = eviction.evicted};
  }
  return {.pending = returns(walk)};
}

Cache::LeafReclaim Cache::reclaimKvLeaf(const Walk &walk, uint64_t block, ReclaimTiming timing) {
  // No state in RAM sits on the leaf (nextUnit): one went first, in the
  // order every victim goes in, or just before its leaf in a page scan. A
  // state already on disk costs nothing and stays while the leaf has a disk
  // copy of what its pages hold.
  if (kv_.copyHoldsPages(block)) {
    kv_.dropPage(block);
    return LeafReclaim::Started;
  }
  // The leaf is written for a state on it or below it, and disk copies below
  // that no state needs go with it. A tier taking cache writes means a
  // demotion, whose page returns only when the copy lands: the reclaim
  // reached a leaf a copy ahead would have let go at once. A demotion that
  // cannot start (demoteKv), with no slot and nothing in flight, or a spent
  // write budget, lets the leaf go as without a tier, with the disk subtree
  // that needs it: a take never stops at a leaf it can neither write nor drop.
  if (kvNeededByState(block)) {
    if (timing == ReclaimTiming::Immediate && takesCacheWrites())
      return LeafReclaim::Impossible;
    const LeafReclaim demotion = demoteKv(walk, block);
    if (demotion != LeafReclaim::Impossible)
      return demotion;
  }
  if (kv_.hasDiskChildren(block) && !dropDiskSubtree(block))
    return LeafReclaim::Impossible;
  if (states_.contains(block))
    static_cast<void>(releaseState(block, true));
  kv_.erase(block);
  return LeafReclaim::Started;
}

bool Cache::dropDiskSubtree(uint64_t block) {
  // A state's write in flight keeps its own block, not the others: the count
  // reaches a dead leaf's page whatever else is written (Cache::reach).
  const std::vector<uint64_t> subtree = kv_.subtree(block);
  if (subtree.empty() || std::ranges::any_of(subtree, [&](uint64_t below) { return states_.writing(below); }))
    return false;
  for (uint64_t below : subtree) {
    if (states_.contains(below))
      static_cast<void>(releaseState(below, true));
    kv_.erase(below);
  }
  return true;
}

void Cache::dropPoisoned() {
  std::erase_if(poisoned_, [&](uint64_t block) {
    if (!kv_.contains(block))
      return true;
    for (uint64_t below : kv_.subtree(block)) {
      if (states_.contains(below))
        static_cast<void>(releaseState(below, true, false));
      // A poisoned block below may have left with its last child.
      if (kv_.contains(below))
        kv_.erase(below);
    }
    kv_.erasePoisonedLeaf(block);
    return !kv_.contains(block);
  });
}

bool Cache::leadsToReach(uint64_t block, const std::array<bool, kv::kPageForms> &reach) const {
  for (; block; block = kv_.key(block).parent, ++searchVisits_)
    if (reach[static_cast<size_t>(pool_.form(kv_.page(block)))])
      return true;
  return false;
}

std::array<bool, kv::kPageForms> Cache::formsInReach(const Walk &walk,
                                                    const std::array<uint32_t, kv::kPageForms> &demoting) const {
  std::array<bool, kv::kPageForms> reach{};
  for (const kv::PageForm form : kv::kAllPageForms) {
    const auto index = static_cast<size_t>(form);
    const uint32_t reusable = reusablePages(walk.scope.upTo, form);
    reach[index] = walk.reaches(form, reusable, demoting[index], pool_.extentPages()) ||
                   (walk.inPlace && form == walk.form &&
                    reusable > pool_.freePageCount(form) + demotionPages()[index]);
  }
  return reach;
}

Cache::ReclaimResult Cache::releaseExtent(const Walk &walk) {
  if (keepsExtents_)
    return {};
  constexpr std::array kForms = kv::kAllPageForms;
  // A need in place keeps the pages of its form it frees (reach): its
  // attempt takes them free, where an extent released would be allocated
  // again (MemoryLedger's Need).
  const auto releases = [&](kv::PageForm form) {
    return !(walk.inPlace && form == walk.form) && walk.releases(form, pool_.freePageCount(form), pool_.extentPages());
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
  pages.reserve((writes_.size() + restores_.size()) * kv_.blockPages());
  for (const Write &write : writes_)
    for (const uint32_t page : kv_.pages(write.block))
      pages.push_back(page);
  for (const auto &[block, _] : restores_)
    for (const uint32_t page : kv_.pages(block))
      pages.push_back(page);
  for (const UnitRead &read : unitReads_)
    pages.insert(pages.end(), read.pages.begin(), read.pages.end());
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
    // A window page moves on its own, and its table with it.
    for (std::vector<uint32_t> &table : active.windows)
      for (size_t index = 0; index < table.size(); ++index)
        if (const uint32_t destination = moves.follow(table[index]); destination != table[index]) {
          table[index] = destination;
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

bool Cache::unitsLanding() const noexcept {
  return std::ranges::any_of(writes_, [&](const Write &write) { return kv_.unitLeaving(write.block); });
}

bool Cache::writesInFlight() const noexcept { return !writes_.empty() || states_.writing(); }

bool Cache::transfersInFlight(const Walk &walk) const noexcept {
  // A write counts by what its landing frees: a page written through or
  // copied ahead returns something only where, under --kv-spill always, it
  // spills once its copy lands, or where the walk then frees its block
  // through the copy.
  return !restores_.empty() || !unitReads_.empty() || writes(WriteClass::Reclaim) || copyAheadReturns(walk) ||
         states_.writing() || spillingWrites() || unitsLanding();
}

bool Cache::returnsInFlight(const Walk &walk, const Shortfall &shortfall) const noexcept {
  if (!restores_.empty() || !unitReads_.empty() || writes(WriteClass::Reclaim) || copyAheadReturns(walk) ||
      states_.writing() || unitsLanding())
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
  if (form != kv::PageForm::Full)
    return 0;
  uint32_t pages = spillingWrites();
  for (const Write &write : writes_)
    pages += write.write == WriteClass::Reclaim ? (write.unit ? kv_.unitPages() : kv_.pagesHeld(write.block)) : 0;
  return pages;
}

uint64_t Cache::copyingAheadBytes() const noexcept {
  uint64_t bytes = states_.persistingBytes();
  for (const Write &write : writes_)
    if (write.write == WriteClass::Persist)
      bytes += pool_.bytesPerPage(pool_.form(kv_.page(write.block))) * kv_.pagesHeld(write.block);
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
  // writes start at once: its claim releases its blocks whose copies landed
  // before, as a chain it restored or the copies written ahead have them,
  // at once, and its retries what lands.
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

uint32_t Cache::diskOnlyTokens(uint64_t block) const {
  if (!block)
    return 0;
  const KvCache::Chain chain = kv_.chain(block);
  return static_cast<uint32_t>(std::ranges::count_if(chain.pages, [&](uint32_t page) {
           return page == KvCache::noPage || pool_.form(page) == kv::PageForm::Index;
         })) * KvCache::pageTokens;
}

bool Cache::commit(Request &active, uint32_t pages, uint32_t unheld) {
  if ((pages || unheld) && !diskBudget_->commit(uint64_t{pages} * spillSlotBytes(),
                                    heldSlotBytes() + uint64_t{unheld} * spillSlotBytes()))
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
  diskBudget_->uncommit(uint64_t{pages} * spillSlotBytes());
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
    if (heldSlotBytes() + spillSlotBytes() > diskBudget_->activeLimitBytes())
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
    std::shared_ptr<KvDiskSlot> slot =
        acquireDiskSlot(parts_, model::DiskBudget::Draw::Commitment, false);
    if (!slot)
      return false;
    --active.committedPages;
    auto transfer = tier_->demote(kv_.pages(block), slot, parts_, WriteClass::Spill, completionNotifier_);
    if (!transfer) {
      // A file that stopped taking writes took none: the slot goes back,
      // and the commitment it drew is the request's again.
      slot.reset();
      recommit(active);
      return false;
    }
    labelKv(block, slot, parts_);
    kv_.setSlot(block, std::move(slot));
    kv_.setTransfer(block, KvCache::Transfer::Draining);
    writes_.push_back({block, WriteClass::Spill, std::move(transfer), parts_});
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
    result.spilledBytes = uint64_t{result.spilledPages} * spillSlotBytes();
    result.activeLimitBytes = diskBudget_->activeLimitBytes();
    result.heldBytes = heldSlotBytes();
    result.committedBytes = diskBudget_->committedBytes();
  }
  return result;
}

void Cache::checkKept() const {
  const auto require = [](bool holds, const char *what) {
    if (!holds)
      throw std::logic_error(std::string("cache invariant broken: ") + what);
  };
  // Each answer of history the cache keeps is the one a search finds now: a
  // stale one would release a state a conversation still resumes from, as
  // count and take alike.
  dropKept();
  static const std::unordered_set<uint64_t> none;
  for (const auto &[block, answer] : history_)
    if (supersededNow(block, none) != answer)
      throw std::logic_error("cache invariant broken: the kept answer of history at block " +
                             std::to_string(block) + " is " + (answer ? "superseded" : "not superseded") +
                             " where a search finds it otherwise");
  for (const auto &[block, holds] : tailUnits_) {
    bool unit = false;
    for (uint64_t id = block, index = 0; id && index < window_->windowPages; ++index, id = kv_.key(id).parent)
      unit = unit || kv_.holdsUnit(id);
    require(unit == holds, "the kept answer of a state's tail holding a unit is the tail's");
  }
}

void Cache::checkInvariants() const {
  const auto check = [](bool holds, const char *what) {
    if (!holds)
      throw std::logic_error(std::string("cache invariant broken: ") + what);
  };
  checkKept();
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
                                                 .copied = needed && kv_.copyHoldsPages(block)};
                      });
  uint64_t committed = 0;
  for (const auto &[_, active] : requests_) {
    check(active.cachedBlocks.size() <= active.pages.size(),
          "a request has a page for each of its blocks");
    // It holds every page of each block of its: on a block's pages, its
    // window unit's too; on its own, a window unit of its own.
    check(active.windows.size() + 1 == kv_.blockPages() &&
              std::ranges::all_of(active.windows,
                                  [&](const std::vector<uint32_t> &table) { return table.size() == active.pages.size(); }),
          "a request has a window table per page of a block's window unit");
    // It holds the units its window reads, from windowFrom on, and no
    // other: of a block on its pages the block's unit, or its own into which
    // it read the block's window part where the block held none
    // (startUnitRead), which the block may take since.
    check(active.windowFrom <= active.pages.size(), "a request's window starts within its pages");
    for (uint32_t index = 0; index < active.pages.size(); ++index)
      for (uint32_t table = 0; table < active.windows.size(); ++table) {
        const uint32_t page = active.windows[table][index];
        if (index < active.windowFrom) {
          check(page == KvCache::noPage, "a request holds no unit of a block its window passed");
          continue;
        }
        check(page != KvCache::noPage && pool_.pageActive(page),
              "a request holds the window unit of each block its window reads");
      }
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
    check(committed * spillSlotBytes() == diskBudget_->committedBytes(),
          "the commitments are the requests'");
  for (const Write &write : writes_)
    check(kv_.transfer(write.block) == KvCache::Transfer::Draining, "a page being written is Draining");
  for (const auto &[block, _] : restores_)
    check(kv_.transfer(block) == KvCache::Transfer::Filling, "a page being restored is Filling");
}


// Disk tier: restores and demotions in flight, the quota they draw on, and
// states promoted back into RAM.

TokenAdmission Cache::restoreRequest(uint64_t requestId, const CacheLookup &lookup,
                                     std::array<uint32_t, kv::kPageForms> held) {
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
  // comes back whole, and past a Spill request's dense history as its index
  // alone. A spilled block comes back whole for a Resident request that
  // would be its only holder, once no transfer moves it: no other request
  // can read its page while its bytes land.
  const auto restoredForm = [&](uint32_t position) {
    return spills && position >= densePages ? kv::PageForm::Index : kv::PageForm::Full;
  };
  // The lane's window reads the units of the resumed state's tail: the
  // first windowPages blocks before its end, and no unit before them.
  const auto restoredBlocks = static_cast<uint32_t>(chain.blocks.size());
  const uint32_t windowFrom = window_ ? restoredBlocks - std::min(restoredBlocks, window_->windowPages) : 0;
  // The pages taken by form (kv::PageForm): each disk-only block's base
  // page, and in the window each block's unit where it holds none: a
  // disk-only block's, which it holds while its restore fills it, or the
  // lane's own, into which a resident block's window part is read.
  const auto restoredPages = [&](uint32_t position) {
    const uint64_t block = chain.blocks[position];
    return (chain.pages[position] == KvCache::noPage ? 1 : 0) +
           (position >= windowFrom && !kv_.holdsUnit(block) ? kv_.unitPages() : 0);
  };
  std::array<uint32_t, kv::kPageForms> restored{};
  std::array<std::vector<uint32_t>, kv::kPageForms> taken;
  std::vector<uint32_t> unspilled;
  for (uint32_t position = 0; position < chain.blocks.size(); ++position) {
    const uint32_t page = chain.pages[position];
    const uint64_t block = chain.blocks[position];
    restored[static_cast<size_t>(restoredForm(position))] += restoredPages(position);
    if (page != KvCache::noPage && !spills && pool_.form(page) == kv::PageForm::Index && !kv_.held(block) &&
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
  for (const kv::PageForm form : kv::kAllPageForms) {
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
  kv_.refreshCopies(chain.blocks);
  // Its state copies too: one order of copies by the conversation's use.
  for (const uint64_t block : chain.blocks)
    states_.refreshCopy(block);
  kv_.retainActive(chain.blocks.back());
  active.windowFrom = windowFrom;
  std::array<size_t, kv::kPageForms> next{};
  for (uint32_t position = 0; position < chain.blocks.size(); ++position) {
    const uint64_t block = chain.blocks[position];
    uint32_t &page = chain.pages[position];
    const auto form = static_cast<size_t>(restoredForm(position));
    if (page == KvCache::noPage) {
      // Root first, so every restored block finds its parent resident. The
      // admission's reference is the request's.
      // In the window it takes its unit too while the restore fills it,
      // which reads its window part with its base part.
      page = taken[form][next[form]++];
      const size_t unit = position >= windowFrom ? kv_.unitPages() : 0;
      kv_.adoptPage(block, page, std::span(taken[form]).subspan(next[form], unit));
      next[form] += unit;
      startRestore(block);
    } else {
      // A spilled block moved back to a Full page is restored whole.
      const bool moved = kv_.page(block) != page;
      page = kv_.page(block);
      kv_.holdPage(page);
      if (position >= active.windowFrom)
        for (const uint32_t each : kv_.window(block))
          pool_.retainPage(each, false);
      if (moved)
        startRestore(block);
    }
    const bool reads = position >= active.windowFrom;
    // A tail block without its unit in memory has its window part on the
    // SSD (KvCache::needsUnit): it is read into the lane's own unit pages.
    std::vector<uint32_t> window(kv_.window(block).begin(), kv_.window(block).end());
    if (reads && kv_.unitPages() && window.empty()) {
      window.assign(taken[form].begin() + static_cast<std::ptrdiff_t>(next[form]),
                    taken[form].begin() + static_cast<std::ptrdiff_t>(next[form] + kv_.unitPages()));
      next[form] += kv_.unitPages();
      startUnitRead(block, window, requestId);
      ++active.pendingRestores;
    }
    for (uint32_t table = 0; table < active.windows.size(); ++table)
      active.windows[table].push_back(reads ? window[table] : KvCache::noPage);
    active.indexPages += pool_.form(page) == kv::PageForm::Index;
    if (auto restore = restores_.find(block); restore != restores_.end()) {
      restore->second.waiters.push_back(requestId);
      ++active.pendingRestores;
    }
  }
  active.pages = std::move(chain.pages);
  if (!active.cachedBlocks.empty())
    kv_.historyChanged(active.cachedBlocks.back());
  active.cachedBlocks = std::move(chain.blocks);
  if (!active.cachedBlocks.empty())
    kv_.historyChanged(active.cachedBlocks.back());
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

void Cache::startUnitRead(uint64_t block, std::span<const uint32_t> pages, uint64_t requestId) {
  if (!kv_.windowCopied(block))
    throw std::logic_error("a tail block holds its window K/V neither in memory nor on the SSD");
  UnitRead read{block, {kv_.page(block)}, nullptr, {requestId}};
  read.pages.insert(read.pages.end(), pages.begin(), pages.end());
  // The read holds the unit pages it fills until it lands, whoever lets
  // them go first. The base page, whose extent the tier reads too, stays
  // the block's, which the request it reads for holds; no compaction moves
  // either (pagesInTransfer). Holding the base page itself would make it
  // active with no request on it, which the count and the take judge
  // apart.
  for (const uint32_t page : pages)
    pool_.retainPage(page, false);
  if (tier_->canRestore())
    read.transfer = tier_->restore(kv_.slot(block), kv::componentBit(kv::Component::Window), read.pages,
                                   completionNotifier_);
  unitReads_.push_back(std::move(read));
}

void Cache::startRestore(uint64_t block) {
  kv_.setTransfer(block, KvCache::Transfer::Filling);
  Restore &restore = restores_[block];
  restore.started = AwakeClock::now();
  if (tier_->canRestore())
    restore.transfer =
        tier_->restore(kv_.slot(block), kv_.heldParts(block) & parts_, kv_.pages(block), completionNotifier_);
}

void Cache::promoteState(const CacheLookup &lookup,
                         const std::function<std::shared_ptr<const CompositeState>()> &copy,
                         std::optional<uint64_t> lane) {
  if (!lookup.state) return;
  const auto block = lookup.state->kvBlock();
  const auto *source = lookup.state->state().get();
  if (!states_.promotable(block, source)) return;
  // A RAM copy's tail holds its units in memory: the lane's, which read its
  // window parts back, taken once the copy is made. The copy only saves a
  // later read, its room what its claim frees (Engine::promotedCopy): that
  // claim may take a state whose tail shared blocks, and their units.
  auto state = tailUnits(block, lane) ? copy() : nullptr;
  const std::optional<KvCache::TailUnits> units =
      state && states_.promotable(block, source) ? tailUnits(block, lane) : std::nullopt;
  if (!units) {
    states_.promotionSkipped();
    return;
  }
  const KvCache::DeferredReleases scope{kv_};
  states_.promote(block, source, std::move(state), *units);
}

Cache::LeafReclaim Cache::demoteKv(const Walk &walk, uint64_t block) {
  if (!takesCacheWrites())
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
  const auto [parts, into] = copyParts(block);
  const KvCache::PassOver ownCopy{kv_, block};
  std::shared_ptr<KvDiskSlot> slot =
      acquireDiskSlot(parts, model::DiskBudget::Draw::Cache, kvNeededByStateInUse(block), false, into,
                      {.lastUse = kv_.lastUsed(block)});
  // Making room may have taken the states the leaf was kept for, slot or
  // not; it then goes like any leaf nothing needs.
  if (!kvNeededByState(block)) {
    giveSlotBack(slot, parts, into != nullptr);
    return LeafReclaim::Impossible;
  }
  // A quota that refused the write leaves the leaf to go without a copy.
  if (!slot && quotaRefused_) {
    ++kvTier_.demotionsRefused;
    return LeafReclaim::Impossible;
  }
  if (!slot)
    return room();
  if (!startKvWrite(block, std::move(slot), parts, into != nullptr, WriteClass::Reclaim)) {
    ++kvTier_.demotionsRefused;
    return room();
  }
  return LeafReclaim::Started;
}

void Cache::giveSlotBack(const std::shared_ptr<KvDiskSlot> &slot, kv::Components parts, bool into) {
  // A new copy's slot goes back as its last holder lets it go.
  if (slot && into)
    tier_->giveUp(slot, parts);
}

bool Cache::startKvWrite(uint64_t block, std::shared_ptr<KvDiskSlot> slot, kv::Components parts, bool into,
                         WriteClass write, bool unit) {
  ++diskCalls_;
  auto transfer = tier_->demote(kv_.pages(block), slot, parts, write, completionNotifier_);
  if (!transfer) {
    giveSlotBack(slot, parts, into);
    return false;
  }
  labelKv(block, slot, parts);
  kv_.setSlot(block, std::move(slot));
  kv_.setTransfer(block, KvCache::Transfer::Draining);
  writes_.push_back({block, write, std::move(transfer), parts, AwakeClock::now(), unit});
  return true;
}

Cache::CopyParts Cache::copyParts(uint64_t block, kv::Components want) const {
  const kv::Components held = kv_.heldParts(block) & parts_ & want;
  const std::shared_ptr<KvDiskSlot> slot = kv_.slot(block);
  // Beside its own copy, the parts it lacks; beside one of another
  // command's K/V, which no part of its own joins, all of them anew.
  if (slot && kv_.ownCopy(block))
    return {static_cast<kv::Components>(held & ~slot->parts()), slot};
  return {held, nullptr};
}

void Cache::labelKv(uint64_t block, const std::shared_ptr<KvDiskSlot> &slot, kv::Components parts) {
  if (!persistent_)
    return;
  // A copy's parts carry one serial, a part added later its copy's.
  if (parts & kv::componentBit(kv::Component::Base))
    kv_.setCopySerial(block, ++copySerial_);
  const KvCache::Key key = kv_.key(block);
  KvSlotLabel label{{block, key.parent, kv_.lastUsed(block), key.identity.lo, key.identity.hi, {}},
                    kv_.copySerial(block)};
  std::copy(key.tokens.begin(), key.tokens.end(), label.block.tokens.begin());
  tier_->label(slot, parts, encodeKvSlotLabel(label, parts_ != kv::componentBit(kv::Component::Base)));
}

std::shared_ptr<KvDiskSlot> Cache::acquireDiskSlot(kv::Components parts, model::DiskBudget::Draw draw, bool inUse,
                                                   bool ahead, const std::shared_ptr<KvDiskSlot> &into, Worth worth) {
  quotaRefused_ = false;
  // Making room is no use while the limit cannot hold the slot.
  if (diskBudget_ &&
      (draw == model::DiskBudget::Draw::Commitment ? diskBudget_->activeLimitBytes()
                                                   : diskBudget_->limitBytes()) < tier_->slotBytes(parts))
    return {};
  // A committed page is a request's KV, which comes before any cached copy:
  // once nothing else is left, it takes the spilled chain of a state in RAM
  // too, the state first. Only write-through draws on a commitment, and it
  // holds no state.
  const bool committed = draw == model::DiskBudget::Draw::Commitment;
  for (;;) {
    if (auto slot = tier_->acquireSlot(parts, draw, into)) {
      quotaFullAhead_ = quotaFullAhead_ && !ahead;
      return slot;
    }
    if (!freeDiskSpace(inUse, committed, ahead, committed ? Worth{} : worth) &&
        !(committed && evictStateOnDiskLeaf()))
      return {};
  }
}

bool Cache::evictStateOnDiskLeaf() {
  for (auto leaf = kv_.diskCandidate(false); leaf; leaf = kv_.diskCandidate(false, leaf->id)) {
    if (states_.stateResident(leaf->id) && !kvNeededByStateInUse(leaf->id)) {
      static_cast<void>(releaseState(leaf->id, true, false));
      return true;
    }
  }
  return false;
}

bool Cache::freeDiskSpace(bool inUse, bool active, bool ahead, Worth worth) {
  ++diskCalls_;
  const auto [next, refused] = nextDiskReplacement(inUse, active, ahead, worth);
  quotaRefused_ = refused;
  // Once a copy ahead first finds a full quota holding nothing it may take,
  // until one finds room again: the conversations in memory keep older
  // points a restart takes back, or none.
  if (!next && ahead && !std::exchange(quotaFullAhead_, true))
    logLine("Persistent cache: the SSD cache is full; new restore points of the conversations in memory are not "
            "written until it has room, and a crash loses what lies past each one's last point on the SSD (a larger "
            "--max-cache-disk gives room).");
  if (!next)
    return false;
  const uint64_t id = next->at.id;
  ++kvTier_.quotaGivenUp[static_cast<size_t>(next->kind)];
  switch (next->kind) {
  case DiskVictim::WindowPart:
    // A copy keeps its base part; one of a window part alone goes whole.
    if (kv_.slot(id)->parts() == kv::componentBit(kv::Component::Window)) {
      kv_.setSlot(id, nullptr);
    } else {
      tier_->giveUp(kv_.slot(id), kv::componentBit(kv::Component::Window));
      kv_.partGivenUp(id);
    }
    break;
  case DiskVictim::KvCopy:
    // The copy of a block whose page holds its K/V is redundant; a spilled
    // or disk-only leaf's copy no state needs is its only K/V, and it goes
    // with the block.
    if (const uint32_t page = kv_.page(id); page != KvCache::noPage && pool_.form(page) != kv::PageForm::Index) {
      // In a persistent tier a point supersedes the state above it only while
      // its chain has landed (superseded): a copy under a state may leave
      // history as a state given up does, and a restart takes back no point
      // below it (stillTakenBack).
      const bool underState = copyUnderState(id);
      quotaGaveUpState_ = quotaGaveUpState_ || underState;
      if (underState)
        brokenCopies_.push_back(id);
      kv_.setSlot(id, nullptr);
      if (copiesGivenUp_.insert(id).second && copiesGivenUp_.size() > 2 * uint64_t{pool_.pageCount()})
        std::erase_if(copiesGivenUp_, [&](uint64_t block) { return !kv_.contains(block); });
    } else
      kv_.erase(id);
    break;
  case DiskVictim::StateCopy:
  case DiskVictim::StateOnly:
  case DiskVictim::StateInUse:
    quotaGaveUpState_ = true;
    if (states_.stateResident(id))
      states_.dropDisk(id);
    else
      static_cast<void>(releaseState(id, true, false));
    break;
  }
  return true;
}

Cache::DiskChoice Cache::nextDiskReplacement(bool inUse, bool active, bool ahead, Worth worth) const {
  // The table is freeDiskSpace's (Cache.hpp).
  const auto pick = [](DiskVictim kind, const std::optional<CacheEvictionCandidate> &at) {
    return at ? std::optional<DiskReplacement>(DiskReplacement{kind, *at}) : std::nullopt;
  };
  // Rank 0.
  if (const std::optional<CacheEvictionCandidate> part = unneededWindow())
    return {pick(DiskVictim::WindowPart, part)};
  if (const auto unneeded = kv_.unneededCopy())
    return {pick(DiskVictim::KvCopy, unneeded)};
  // The oldest only copy of a state that is history (rank 1), and the
  // oldest of a point (rank 2), past what a request restores from: that
  // stays but for a Spill lane's active KV, as a running lane beats a
  // queued one. A point's K/V on the SSD goes with its state, as copies no
  // state needs.
  std::optional<CacheEvictionCandidate> history, point;
  for (auto each = states_.diskCandidate(false); each && !history; each = states_.diskCandidate(false, each)) {
    ++searchVisits_;
    if (!active && certainReuse(each->id))
      continue;
    if (diskHistory(each->id))
      history = each;
    else if (!point)
      point = each;
  }
  // Rank 1, one order, the oldest first: history and the copies of what
  // memory holds, of which a copy ahead, itself one, takes none, and a
  // write of rank 1 only those used before it.
  const bool forHistory = worth.rank < 2;
  std::optional<DiskReplacement> next;
  const auto consider = [&](DiskVictim kind, const std::optional<CacheEvictionCandidate> &at) {
    if (at && (!forHistory || at->lastUsed < worth.lastUse) && (!next || at->lastUsed < next->at.lastUsed))
      next = DiskReplacement{kind, *at};
  };
  consider(DiskVictim::StateOnly, history);
  if (!ahead) {
    consider(DiskVictim::StateCopy, states_.diskCandidate(true));
    consider(DiskVictim::KvCopy, kv_.diskCandidate(true));
  }
  if (next || ahead)
    return {next};
  if (forHistory)
    return {.refused = true};
  // Rank 2: the point used longest ago, only for a write of one used later.
  if (point)
    return point->lastUsed < worth.lastUse ? DiskChoice{pick(DiskVictim::StateOnly, point)}
                                           : DiskChoice{.refused = true};
  // Rank 3, only for a copy in use.
  return {pick(DiskVictim::StateInUse, inUse ? states_.inUseDiskCandidate() : std::nullopt)};
}

bool Cache::diskHistory(uint64_t block) const {
  if (!superseded(block))
    return false;
  // The nearest states below it on each branch: one with an SSD copy
  // supersedes it there; one without passes the search on below it.
  for (std::vector<uint64_t> pending = kv_.children(block); !pending.empty();) {
    const uint64_t at = pending.back();
    pending.pop_back();
    ++searchVisits_;
    if (states_.onDisk(at))
      return true;
    if (kv_.stateBelow(at))
      for (const uint64_t child : kv_.children(at))
        pending.push_back(child);
  }
  return false;
}

std::optional<CacheEvictionCandidate> Cache::unneededWindow() const {
  std::optional<CacheEvictionCandidate> part = kv_.unneededWindowCandidate(0);
  while (part && std::ranges::any_of(unitReads_, [&](const UnitRead &read) { return read.block == part->id; }))
    part = kv_.unneededWindowCandidate(part->id);
  return part;
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
  if (window_ && !windowsAhead_.empty())
    writeWindowsAhead();
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
            tier_->restore(kv_.slot(block), kv_.heldParts(block) & parts_, kv_.pages(block), completionNotifier_);
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
  // The window parts read into lanes' units: started in turn while the tier
  // has room, each holding its pages until it lands.
  for (auto read = unitReads_.begin(); read != unitReads_.end();) {
    if (!read->transfer && !refused) {
      if (tier_->canRestore())
        read->transfer = tier_->restore(kv_.slot(read->block), kv::componentBit(kv::Component::Window), read->pages,
                                        completionNotifier_);
      refused = !read->transfer;
    }
    if (!read->transfer || !read->transfer->ready()) {
      ++read;
      continue;
    }
    const bool restored = read->transfer->finish();
    ++(restored ? kvTier_.restores : kvTier_.restoreFailures);
    for (const uint32_t page : std::span(read->pages).subspan(1))
      pool_.releasePage(page, false);
    for (uint64_t requestId : read->waiters) {
      auto waiter = requests_.find(requestId);
      if (waiter == requests_.end())
        continue;
      --waiter->second.pendingRestores;
      if (!restored)
        waiter->second.restoreFailed = true;
    }
    // A copy that cannot be read leaves with its block, as a restore's.
    if (!restored && kv_.contains(read->block)) {
      states_.invalidate(read->block);
      kv_.poison(read->block);
      poisoned_.push_back(read->block);
    }
    read = unitReads_.erase(read);
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
    const bool unit = entry->unit;
    const kv::Components parts = entry->parts;
    const bool written = entry->transfer->finish();
    entry = writes_.erase(entry);
    // A failed write's parts go before the transfer ends, so that a block
    // whose unit waited for its window part keeps the unit; the parts of its
    // copy that landed before stay, a point on the SSD may need them. A
    // point below that counted the base part in flight as its chain's is
    // taken back no more (stillTakenBack).
    if (const std::shared_ptr<KvDiskSlot> slot = kv_.slot(block); !written && slot) {
      if ((parts & kv::componentBit(kv::Component::Base)) && copyUnderState(block))
        brokenCopies_.push_back(block);
      if (slot->parts() & ~parts) {
        tier_->giveUp(slot, parts);
        kv_.partGivenUp(block);
      } else {
        kv_.setSlot(block, nullptr);
      }
    }
    kv_.setTransfer(block, KvCache::Transfer::None);
    progressed = true;
    if (const std::shared_ptr<KvDiskSlot> slot = kv_.slot(block); written && slot && write != WriteClass::Spill) {
      // The parts it wrote, of a copy that may have held others before.
      uint64_t bytes = 0;
      for (const kv::Component part : kv::kAllComponents)
        bytes += parts & kv::componentBit(part) ? slot->bytes(part) : 0;
      (write == WriteClass::Reclaim ? kvTier_.demotedBytes : kvTier_.copiedBytes) += bytes;
      kvTier_.windowPartBytes += parts & kv::componentBit(kv::Component::Window) ? slot->bytes(kv::Component::Window) : 0;
      if (copiesGivenUp_.erase(block)) {
        ++kvTier_.rewrites;
        kvTier_.rewrittenBytes += bytes;
      }
    }
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
      if (written && !unit && kv_.residentLeaf(block) && !states_.stateResident(block))
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
        spilled_.writeThroughBytes += spillSlotBytes();
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
    if (status != PersistStatus::Started && status != PersistStatus::Busy && persisting_.block) {
      // A restart takes the point back no more (stillTakenBack).
      kv_.historyChanged(persisting_.block);
      persisting_ = {};
    }
    return status;
  };
  if (persistent_ ? !kv_.contains(block) || !states_.ordinary(block) : !newestPoint(block))
    return settled(PersistStatus::Unneeded);
  if (!kvTierWritable())
    return settled(PersistStatus::Refused);
  // A spent write budget holds the whole point back until it refills.
  if (!takesCacheWrites())
    return settled(PersistStatus::Busy);
  if (persisting_.block != block) {
    if (persisting_.block && kv_.contains(persisting_.block))
      kv_.historyChanged(persisting_.block);
    kv_.historyChanged(block);
    persisting_ = {block, kv_.chain(block).blocks, 0, 0};
    for (size_t index = 0; index < persisting_.blocks.size(); ++index)
      persisting_.bytes += copyBytes(persisting_.blocks[index], pointParts(index, persisting_.blocks.size()));
  }
  // Waiting for room would hold back the points after this one.
  if (!pointFits(persisting_))
    return settled(PersistStatus::Refused);
  // Nothing starts while the tier serves what is needed now: a restore that
  // waits, a reclaim's demotion and a Spill request's write through go
  // first.
  const bool yields =
      !restores_.empty() || !unitReads_.empty() || writes(WriteClass::Reclaim) || writes(WriteClass::Spill);
  if (yields && !states_.onDisk(block))
    return settled(PersistStatus::Busy);
  // The chain goes first, the state once every copy of it is under way, so
  // that a point refused for room leaves no state copy behind, only copies
  // of what memory holds that its next stride goes on from. Only a
  // persistent tier makes room for either.
  bool inFlight = false;
  const std::vector<uint64_t> &blocks = persisting_.blocks;
  // A temporary tier may give up a copy below the frontier meanwhile, so a
  // call that finds nothing in flight walks the whole chain once more.
  for (size_t from = persisting_.landed;; from = 0) {
    for (size_t index = from; index < blocks.size(); ++index) {
      const uint64_t id = blocks[index];
      // A copy, of whatever K/V, that holds the parts the point needs of
      // what the block's pages hold.
      const kv::Components want = pointParts(index, blocks.size()), needed = kv_.heldParts(id) & parts_ & want;
      if (kv_.slot(id) && (kv_.slot(id)->parts() & needed) == needed) {
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
      if (yields || !tier_->canDemote(WriteClass::Persist) || !takesCacheWrites())
        return settled(inFlight ? PersistStatus::Started : PersistStatus::Busy);
      // A copy holds what its block holds: the parts its pages hold that
      // it lacks.
      if (kv_.transfer(id) != KvCache::Transfer::None) {
        inFlight = true;
        continue;
      }
      const auto [parts, into] = copyParts(id, want);
      std::shared_ptr<KvDiskSlot> slot =
          persistent_ ? acquireDiskSlot(parts, model::DiskBudget::Draw::Cache, false, true, into)
                      : tier_->acquireSlot(parts, model::DiskBudget::Draw::Cache, into);
      if (!slot)
        return settled(PersistStatus::Refused);
      if (!startKvWrite(id, std::move(slot), parts, into != nullptr, WriteClass::Persist))
        return settled(inFlight ? PersistStatus::Started : PersistStatus::Busy);
      inFlight = true;
    }
    if (inFlight || !from)
      break;
  }
  const PersistStatus state = states_.persist(block, persistent_);
  if (state != PersistStatus::Durable && state != PersistStatus::Started)
    return settled(state);
  return settled(inFlight || state == PersistStatus::Started ? PersistStatus::Started : PersistStatus::Durable);
}

uint64_t Cache::deepestTakenBack(uint64_t block, bool landed) const {
  if (!kv_.contains(block))
    return 0;
  const std::vector<uint64_t> blocks = kv_.chain(block).blocks;
  uint64_t deepest = 0;
  bool whole = true;
  searchVisits_ += blocks.size();
  for (size_t index = 0; index + 1 < blocks.size(); ++index) {
    const uint64_t id = blocks[index];
    whole = whole && (landed ? copyLanded(id, false) : kv_.slot(id) != nullptr);
    // With every base part above it landed (`whole`), a point has landed
    // once its state's copy and its tail's window parts have
    // (pointLanded), which asks no walk of the chain again at each block.
    if (landed ? whole && copyLanded(id, true) && tailLanded(id)
               : (whole && states_.onDisk(id)) || id == persisting_.block)
      deepest = id;
  }
  return deepest;
}

bool Cache::copyLanded(uint64_t block, bool state) const {
  // The base part a chain needs of each block.
  const std::shared_ptr<KvDiskSlot> slot = kv_.slot(block);
  return state ? states_.onDisk(block) && !states_.writing(block)
               : slot && slot->presence(kv::Component::Base) == KvDiskSlot::Presence::Landed &&
                     kv_.transfer(block) == KvCache::Transfer::None;
}

bool Cache::pointLanded(uint64_t block) const {
  return copyLanded(block, true) && tailLanded(block) && chainLanded(block);
}

bool Cache::chainLanded(uint64_t block) const {
  if (!supersededMemo_) {
    for (uint64_t id = block; id; id = kv_.key(id).parent, ++searchVisits_)
      if (!copyLanded(id, false))
        return false;
    return true;
  }
  // Up to the nearest block whose answer the count has, then down again.
  std::unordered_map<uint64_t, bool> &known = supersededMemo_->chainLanded;
  std::vector<uint64_t> path;
  bool landed = true;
  for (uint64_t id = block; id; id = kv_.key(id).parent, ++searchVisits_) {
    if (const auto found = known.find(id); found != known.end()) {
      landed = found->second;
      break;
    }
    path.push_back(id);
  }
  for (auto id = path.rbegin(); id != path.rend(); ++id)
    known[*id] = landed = landed && copyLanded(*id, false);
  return landed;
}

bool Cache::tailLanded(uint64_t block) const {
  if (!window_ || !kv_.unitPages())
    return true;
  uint64_t id = block;
  for (uint32_t index = 0; id && index < window_->windowPages; ++index, id = kv_.key(id).parent, ++searchVisits_) {
    const std::shared_ptr<KvDiskSlot> slot = kv_.slot(id);
    if (!slot || slot->presence(kv::Component::Window) != KvDiskSlot::Presence::Landed)
      return false;
  }
  return true;
}

bool Cache::newestPoint(uint64_t block) const {
  return kv_.contains(block) && states_.ordinary(block) && !superseded(block);
}

void Cache::followGone(SupersededMemo &memo, const std::unordered_set<uint64_t> &gone) const {
  if (memo.seen.size() == gone.size())
    return;
  const auto takeBack = [&](uint64_t id) {
    if (!states_.contains(id))
      return;
    memo.passed.insert(id);
    memo.answers.erase(id);
  };
  for (const uint64_t each : gone) {
    ++searchVisits_;
    if (!memo.seen.insert(each).second || !kv_.contains(each))
      continue;
    takeBack(each);
    SupersededMemo::Walked walk;
    // To the second state that stops searches: whether a request went on
    // from the first (supersededNow's wentOn) reads the states below it.
    uint32_t stops = 0;
    for (uint64_t above = kv_.key(each).parent; above; ++searchVisits_) {
      // Through a state evicted before, from where its walk ended: past the
      // first state that stopped it, to the second. Of every state it
      // passed, gone or not: a count asks of a state it evicted too, and
      // that state's answer reads the states below it.
      if (const auto before = memo.walked.find(above); before != memo.walked.end()) {
        takeBack(above);
        walk.passed.push_back(above);
        for (const uint64_t state : before->second.passed) {
          takeBack(state);
          walk.passed.push_back(state);
        }
        above = before->second.stop;
        ++stops;
        continue;
      }
      takeBack(above);
      if (states_.contains(above))
        walk.passed.push_back(above);
      if (stopsSearch(above, gone) && ++stops >= 2) {
        walk.stop = above;
        break;
      }
      above = kv_.key(above).parent;
    }
    memo.walked.emplace(each, std::move(walk));
  }
}

bool Cache::superseded(uint64_t block, const std::unordered_set<uint64_t> &gone) const {
  if (unrenderedReply(block))
    return supersededNow(block, gone);
  if (gone.empty())
    return keptHistory(block);
  SupersededMemo *const memo = supersededMemo_;
  if (!memo || memo->gone != &gone)
    return supersededNow(block, gone);
  // A state gone changes the answers of those its walk up passed
  // (followGone), and of no other: a search from above them ends at the
  // state that ended the walk.
  followGone(*memo, gone);
  bool answer = false;
  if (!memo->passed.contains(block))
    answer = keptHistory(block);
  else if (const auto found = memo->answers.find(block); found != memo->answers.end())
    answer = found->second;
  else
    answer = memo->answers[block] = supersededNow(block, gone);
  // Under test, the overlay's answer is the one asked afresh of the cache as
  // the count's evictions leave it.
  if (checkKept_ && answer != supersededNow(block, gone))
    throw std::logic_error("cache invariant broken: a count's answer of history at block " + std::to_string(block) +
                           " is not the one a search finds against its evictions");
  return answer;
}

bool Cache::unrenderedReply(uint64_t block) const {
  return kv_.heldReply(block) && kv_.children(block).empty() &&
         std::ranges::none_of(requests_, [&](const auto &entry) {
           return !entry.second.cachedBlocks.empty() && entry.second.cachedBlocks.back() == block;
         });
}

bool Cache::keptHistory(uint64_t block) const {
  dropKept();
  ++searchVisits_;
  if (const auto found = history_.find(block); found != history_.end())
    return found->second;
  static const std::unordered_set<uint64_t> none;
  const bool answer = supersededNow(block, none);
  history_.emplace(block, answer);
  return answer;
}

bool Cache::tailHoldsUnit(uint64_t block) const {
  dropKept();
  ++searchVisits_;
  if (const auto found = tailUnits_.find(block); found != tailUnits_.end())
    return found->second;
  bool holds = false;
  for (uint64_t id = block, index = 0; !holds && id && index < window_->windowPages;
       ++index, id = kv_.key(id).parent, ++searchVisits_)
    holds = kv_.holdsUnit(id);
  tailUnits_.emplace(block, holds);
  return holds;
}

void Cache::dropKept() const {
  using Change = KvCache::Change;
  if (kv_.historyUnchanged())
    return;
  bool overflow = false;
  const std::vector<KvCache::HistoryChange> changes = kv_.takeHistoryChanges(overflow);
  if (overflow) {
    history_.clear();
    tailUnits_.clear();
    return;
  }
  if (history_.empty() && tailUnits_.empty())
    return;
  searchVisits_ += 2 * changes.size();
  // The answers of a block erased go with it.
  for (const auto &[block, change] : changes)
    if (!kv_.contains(block)) {
      history_.erase(block);
      tailUnits_.erase(block);
    }
  // A unit taken or given up changes the tails that hold its block: the
  // states within a window below it.
  if (window_ && !tailUnits_.empty())
    for (const auto &[block, change] : changes) {
      if (change != Change::Unit)
        continue;
      if (!kv_.contains(block))
        continue;
      for (std::vector<std::pair<uint64_t, uint32_t>> pending{{block, 0}}; !pending.empty();) {
        const auto [id, depth] = pending.back();
        pending.pop_back();
        ++searchVisits_;
        tailUnits_.erase(id);
        if (depth + 1 < window_->windowPages)
          for (const uint64_t child : kv_.children(id))
            pending.emplace_back(child, depth + 1);
      }
    }
  if (history_.empty())
    return;
  static const std::unordered_set<uint64_t> none;
  // The answer of a state reads, of the cache, its own block and the blocks
  // its search reaches below it down to the states that end it (their
  // classes, copies, children, pages, the chain ends at them, the states
  // below them), and the blocks beside the paths of a reply's end it meets
  // (leftBehind), whose states are below a block it reaches. A change at a
  // block so changes the answers of the states up from it to the first
  // state above that ends every search through it (stopsSearch: an ordinary
  // state, with a persistent tier one whose point landed whole), that one
  // included, and no other. By event:
  // - a state added, erased, or its class, copies or write changing
  //   (StateCache::reindex, erase): its block;
  // - a block added or erased: its parent, whose children changed, and the
  //   erased block's own answer;
  // - a request's chain end moving (ends): the old end and the new;
  // - a block's page changing: its block. A page turning active or idle
  //   records nothing of its own: a search reads it only at a block with no
  //   state below (the last state's blocks down to a chain's end), and a
  //   request takes or leaves such a block's page only as it takes or
  //   leaves the chain through it (restoreRequest, publishCommittedBlocks,
  //   endRequest), which moves its chain's end, recorded at or below it;
  // - a head or a reply noted at a block (head, heldReply, replyStart): it;
  // - the point persist() copies changing (stillTakenBack): both blocks;
  // - a copy, or a part of it, landing or given up, a transfer ending, a
  //   copy turning foreign (copyLanded, stillTakenBack): its block, and in a
  //   persistent tier with a window each state within a window below it,
  //   whose tail's window parts (tailLanded) a search from above reads even
  //   where the block lies above the state that asks.
  std::vector<uint64_t> starts;
  for (const auto &[block, change] : changes) {
    if (!kv_.contains(block))
      continue;
    if (change == Change::Unit)
      continue;
    starts.push_back(block);
    if (change != Change::Copy || !persistent_ || !window_)
      continue;
    for (std::vector<std::pair<uint64_t, uint32_t>> pending{{block, 0}}; !pending.empty();) {
      const auto [id, depth] = pending.back();
      pending.pop_back();
      ++searchVisits_;
      if (id != block && states_.contains(id))
        starts.push_back(id);
      if (depth + 1 < window_->windowPages)
        for (const uint64_t child : kv_.children(id))
          pending.emplace_back(child, depth + 1);
    }
  }
  // The chains' landed base parts, asked once a block (chainLanded).
  const MemoScope scope{*this};
  // The blocks a walk went on up from: the next walk to meet one is done.
  std::unordered_set<uint64_t> through;
  // The first state that stops the walk may be the one a search from above
  // ends at, whether a request went on from it (supersededNow's wentOn)
  // reads the states below it: the walk goes on to the second, included.
  // Past a block a walk went on from, the walk before it went as far.
  for (const uint64_t start : starts) {
    uint32_t stops = 0;
    for (uint64_t id = start; id; id = kv_.key(id).parent) {
      ++searchVisits_;
      if (!stops && through.contains(id))
        break;
      history_.erase(id);
      if (id != start && stopsSearch(id, none) && ++stops == 2)
        break;
      if (!stops)
        through.insert(id);
    }
  }
}

bool Cache::stateBelow(uint64_t block, const std::unordered_set<uint64_t> &gone) const {
  const uint32_t all = kv_.statesBelow(block);
  // More states below than are gone: some stay.
  if (all > gone.size())
    return true;
  if (!all)
    return false;
  // Else down the blocks that hold or lead to a state, to the first that
  // stays: at most the blocks between the block and the states below it, all
  // of them where every one is gone. Asked only of a reply's sibling
  // (supersededNow's leftBehind); a search marks what stays below each block
  // it visits as it goes.
  // Else down the blocks that hold or lead to a state, to the first that
  // stays.
  for (std::vector<uint64_t> pending = kv_.children(block); !pending.empty();) {
    const uint64_t id = pending.back();
    pending.pop_back();
    ++searchVisits_;
    if (states_.contains(id) && !gone.contains(id))
      return true;
    if (kv_.stateBelow(id)) {
      const std::vector<uint64_t> children = kv_.children(id);
      pending.insert(pending.end(), children.begin(), children.end());
    }
  }
  return false;
}

bool Cache::stopsSearch(uint64_t block, const std::unordered_set<uint64_t> &gone) const {
  return states_.ordinary(block) && !gone.contains(block) && (!persistent_ || pointLanded(block));
}

bool Cache::supersededNow(uint64_t block, const std::unordered_set<uint64_t> &gone) const {
  const auto ends = [&](uint64_t id) {
    searchVisits_ += requests_.size();
    return std::ranges::any_of(requests_, [&](const auto &entry) {
      return !entry.second.cachedBlocks.empty() && entry.second.cachedBlocks.back() == id;
    });
  };
  // A reply's end state no prompt has rendered through, a leaf no request's
  // chain ends at, is history once a prompt left its reply another way: by
  // a block a prefill published, with a state at or below it, beside one of
  // the reply's own up to the nearest state above or the block the reply
  // shares with its prompt (replyStart), where only a block that holds the
  // prompt's tokens too left the reply and not the prompt. A sibling reply
  // shows nothing: two conversations from one prompt each go on from their
  // own.
  const auto unrendered = [&](uint64_t id) { return unrenderedReply(id); };
  const auto leftBehind = [&](uint64_t id) {
    for (uint64_t child = id, at = kv_.key(id).parent; at; at = kv_.key(child = at).parent) {
      const uint32_t shared = kv_.replyStart(child);
      if (!shared && !kv_.heldReply(child))
        break;
      const std::span<const uint32_t> prompt = kv_.key(child).tokens.first(shared);
      for (const uint64_t other : kv_.children(at))
        if (other != child && !kv_.heldReply(other) &&
            ((states_.contains(other) && !gone.contains(other)) || stateBelow(other, gone)) &&
            std::ranges::equal(kv_.key(other).tokens.first(shared), prompt))
          return true;
      if (shared || (states_.contains(at) && !gone.contains(at)))
        break;
    }
    return false;
  };
  if (unrendered(block))
    return leftBehind(block);
  // A request went on from the state at the block: an ordinary state lies
  // below it, not gone, that is no reply's end no prompt rendered through;
  // or a running request's chain reaches it, the request's own point. A
  // finished request's last point, with nothing but its unrendered reply's
  // end below it, supersedes nothing: it may be a side question, a retry or
  // an edit, whatever the length of its reply. A head keeps its own rule.
  const auto wentOn = [&](uint64_t id) {
    if (kv_.head(id))
      return true;
    const uint32_t depth = kv_.chainLength(id);
    if (std::ranges::any_of(requests_, [&](const auto &entry) {
          const std::vector<uint64_t> &chain = entry.second.cachedBlocks;
          return chain.size() >= depth && chain[depth - 1] == id;
        }))
      return true;
    for (std::vector<uint64_t> pending = kv_.children(id); !pending.empty();) {
      const uint64_t at = pending.back();
      pending.pop_back();
      ++searchVisits_;
      if (states_.ordinary(at) && !gone.contains(at) && !unrendered(at))
        return true;
      if (kv_.stateBelow(at))
        for (const uint64_t child : kv_.children(at))
          pending.push_back(child);
    }
    return false;
  };
  // What supersedes it: an ordinary state below it. In a persistent tier,
  // where a restart takes the block's state back, only one a restart takes
  // back too, its state and every block down to it copied and landed: a
  // crash then loses a stride, not the conversation.
  const bool takenBack = persistent_ && stillTakenBack(block);
  // A point still taken back stands for its chain (stillTakenBack): one
  // below it has landed once its own parts and the base parts down from it
  // have, which asks no walk of the whole chain at each state.
  const auto supersedes = [&](uint64_t id) {
    if (!takenBack)
      return true;
    if (!copyLanded(id, true) || !tailLanded(id))
      return false;
    for (uint64_t at = id; at && at != block; at = kv_.key(at).parent, ++searchVisits_)
      if (!copyLanded(at, false))
        return false;
    return true;
  };
  // The nearest such states below it, each down its own path: one alone
  // superseded it, and two are conversations that parted below it, however
  // many pages they share first. A reply's end no prompt has rendered
  // through supersedes nothing, and while it may still go on keeps the
  // block: a turn repeated or edited resumes at its prompt's replay point.
  // A running request that has reached none of them, its chain ending
  // before one or its pages on a path that holds none, still resumes from
  // the block. A prompt's head (noteHead) is superseded only by a deeper
  // head, a system message that grew: the next conversation on it resumes
  // there, whoever went on from it.
  //
  // The blocks below it down to the states that end the search, in one walk:
  // each in order after its parent, marked with whether a state stays below
  // it, one not among the states the releases counted before it (`gone`),
  // which hold nothing below a block once they go. The search reaches a
  // block whose parent it reached, holds a state that stays below, and does
  // not end it; one where none stays below goes no further. One walk of the
  // blocks that lead to a state, each once, whatever is gone.
  struct Node final {
    uint64_t id;
    int32_t parent;
    bool ends;
    bool stays;
  };
  std::vector<Node> nodes{{block, -1, false, false}};
  for (size_t at = 0; at < nodes.size(); ++at) {
    ++searchVisits_;
    const uint64_t id = nodes[at].id;
    if (id != block && states_.ordinary(id) && !gone.contains(id) && (unrendered(id) || supersedes(id))) {
      nodes[at].ends = true;
      continue;
    }
    if (kv_.stateBelow(id))
      for (const uint64_t child : kv_.children(id))
        nodes.push_back({child, static_cast<int32_t>(at), false, false});
  }
  // Children come after their parents: backwards, each tells its parent;
  // then forwards, from the block.
  searchVisits_ += 2 * nodes.size();
  for (size_t at = nodes.size(); at-- > 1;) {
    const Node &node = nodes[at];
    if (node.stays || (states_.contains(node.id) && !gone.contains(node.id)))
      nodes[static_cast<size_t>(node.parent)].stays = true;
  }
  if (!nodes.front().stays)
    return false;
  uint32_t nearest = 0;
  bool head = false;
  std::vector<bool> reached(nodes.size(), false);
  reached.front() = true;
  for (size_t at = 0; at < nodes.size(); ++at) {
    const Node &node = nodes[at];
    if (at && !reached[static_cast<size_t>(node.parent)])
      continue;
    const Node *parent = at ? &nodes[static_cast<size_t>(node.parent)] : nullptr;
    if (parent && (parent->ends || !parent->stays))
      continue;
    reached[at] = true;
    if (node.ends) {
      if (unrendered(node.id)) {
        if (!leftBehind(node.id))
          return false;
        continue;
      }
      // A prompt's replay point no later request went on from supersedes
      // nothing yet: its request may be a side question, a retry or an
      // edit, and the conversation's next request resumes above it. Its
      // path stops there. A head serves every conversation on it, each of
      // which counts from its first request.
      if (!kv_.head(block) && !wentOn(node.id))
        continue;
      if (++nearest > 1)
        return false;
      head = kv_.head(node.id);
      continue;
    }
    const uint32_t page = kv_.page(node.id);
    if (ends(node.id) || (!node.stays && page != KvCache::noPage && pool_.pageActive(page)))
      return false;
  }
  return nearest == 1 && (head || !kv_.head(block));
}

uint64_t Cache::copyBytes(uint64_t block, kv::Components want) const {
  const std::shared_ptr<KvDiskSlot> slot = kv_.slot(block);
  if (kv_.page(block) == KvCache::noPage)
    return slot ? slot->bytes() : 0;
  // What it has and the parts a write gives it.
  const auto [parts, into] = copyParts(block, want);
  return (into ? slot->bytes() : 0) + (parts ? tier_->slotBytes(parts) : 0);
}

kv::Components Cache::pointParts(size_t index, size_t blocks) const noexcept {
  const bool tail = window_ && index + window_->windowPages >= blocks;
  return static_cast<kv::Components>(kv::componentBit(kv::Component::Base) |
                                     (persistent_ && tail ? kv::componentBit(kv::Component::Window) : 0));
}

bool Cache::pointFits(const PersistCursor &point) const {
  if (!diskBudget_)
    return true;
  const uint64_t limit = diskBudget_->limitBytes();
  if (persistent_)
    return point.bytes + states_.bytes(point.block) <= limit;
  uint64_t lacking = states_.onDisk(point.block) ? 0 : states_.bytes(point.block);
  // The parts each block's copy lacks.
  for (size_t index = 0; index < point.blocks.size(); ++index) {
    const uint64_t id = point.blocks[index];
    const kv::Components needed = kv_.heldParts(id) & parts_ & pointParts(index, point.blocks.size());
    const CopyParts lacks = copyParts(id, pointParts(index, point.blocks.size()));
    if (!kv_.slot(id) || (kv_.slot(id)->parts() & needed) != needed)
      lacking += copyBytes(id, pointParts(index, point.blocks.size())) - (lacks.into ? kv_.slot(id)->bytes() : 0);
  }
  const uint64_t taken = diskBudget_->usedBytes() + diskBudget_->committedBytes();
  return taken <= limit && lacking <= limit - taken;
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
    tier.writesGated = diskBudget_->writesStopped() ? DiskWritesGate::Closed
                       : !diskBudget_->admits()     ? DiskWritesGate::Budget
                                                    : DiskWritesGate::No;
    tier.writeBudgetBytes =
        static_cast<uint64_t>(std::max<int64_t>(0, diskBudget_->writeBudgetBytes()));
    tier.writesDeferredMicroseconds = diskBudget_->writesDeferredMicroseconds();
    tier.readBytes = diskBudget_->readBytes();
    tier.writtenBytes = diskBudget_->writtenBytes();
    tier.writeRate = diskBudget_->writeRate();
  }
  tier.diskBytes = kv_.diskBytes();
  std::optional<KvWindowSnapshot> window;
  if (window_)
    window = KvWindowSnapshot{.windowPages = window_->windowPages,
                              .pinnedBlocks = kv_.pinnedBlocks(),
                              .windowUnits = kv_.windowUnits(),
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
