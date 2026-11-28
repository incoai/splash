#include "engine/KvCache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace splash::engine {
namespace {

uint64_t mix(uint64_t hash, uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  return hash;
}

uint64_t indexHash(uint64_t parentHash, std::span<const uint32_t> tokens,
                   BlockIdentity identity) noexcept {
  uint64_t hash = mix(0x6a09e667f3bcc909ULL, parentHash);
  for (uint32_t token : tokens)
    hash = mix(hash, token);
  hash = mix(hash, identity.lo);
  hash = mix(hash, identity.hi);
  return hash;
}

} // namespace

BlockIdentity blockImageIdentity(uint64_t blockBegin, uint32_t blockTokens,
                                 std::span<const ImageSpan> spans) noexcept {
  BlockIdentity identity;
  const uint64_t blockEnd = blockBegin + blockTokens;
  for (const ImageSpan &span : spans) {
    if (span.offset >= blockEnd)
      break;
    if (span.end() <= blockBegin)
      continue;
    // Two independently seeded chains fold the content digest, the grid, the
    // block's alignment inside the span and any own size into 128 bits.
    uint64_t lo = identity.lo ? identity.lo : 0x243f6a8885a308d3ULL;
    uint64_t hi = identity.hi ? identity.hi : 0x13198a2e03707344ULL;
    for (uint64_t value :
         {span.digestLo, span.digestHi,
          (uint64_t{span.gridHeight} << 32) | span.gridWidth,
          (uint64_t{span.offset} << 32) | span.tokens, blockBegin}) {
      lo = mix(lo, value);
      hi = mix(hi, ~value);
    }
    // The image's own size, where its grid pads past it, as the model's image cache keys it.
    if (const uint64_t content = (uint64_t{span.contentHeight} << 32) | span.contentWidth)
      lo = mix(lo, content), hi = mix(hi, ~content);
    identity = {lo, hi};
  }
  return identity;
}

BlockIdentity blockIdentity(uint64_t blockBegin, uint32_t blockTokens,
                            std::span<const ImageSpan> spans, CacheSalt salt) noexcept {
  BlockIdentity identity = blockImageIdentity(blockBegin, blockTokens, spans);
  if (blockBegin || salt == CacheSalt{})
    return identity;
  // The image chains' seeds again, with the salt in both.
  uint64_t lo = identity.lo ? identity.lo : 0x243f6a8885a308d3ULL;
  uint64_t hi = identity.hi ? identity.hi : 0x13198a2e03707344ULL;
  for (uint64_t value : {salt.lo, salt.hi}) {
    lo = mix(lo, value);
    hi = mix(hi, ~value);
  }
  return {lo, hi};
}

KvCache::~KvCache() noexcept {
  pool_.observeActivity({});
  for (const auto &[_, entry] : blocks_) {
    if (entry.page == noPage)
      continue;
    try {
      disown(entry.page, entry.window);
    } catch (...) {
      std::terminate();
    }
  }
}

void KvCache::own(uint32_t page, std::span<const uint32_t> window) {
  if (!window.empty() && window.size() != unitPages_)
    throw std::invalid_argument("a KV cache block's window unit has " + std::to_string(unitPages_) + " pages");
  for (const uint32_t each : window)
    if (each >= pool_.pageCount() || blockOnPage_[each])
      throw std::logic_error("KV cache window page is out of range or holds a block");
  pool_.retainPage(page, true);
  for (const uint32_t each : window)
    pool_.retainPage(each, true);
}

void KvCache::disown(uint32_t page, std::span<const uint32_t> window) {
  pool_.releasePage(page, true);
  for (const uint32_t each : window)
    pool_.releasePage(each, true);
}

std::vector<uint32_t> KvCache::pages(uint64_t blockId) const {
  const Block &entry = block(blockId);
  std::vector<uint32_t> result{entry.page};
  result.insert(result.end(), entry.window.begin(), entry.window.end());
  return result;
}

std::optional<KvCache::BlockMatch>
KvCache::find(uint64_t parentBlock, std::span<const uint32_t> tokens,
              BlockIdentity identity) const {
  if (tokens.size() != pageTokens) {
    return std::nullopt;
  }
  if (parentBlock && !blocks_.contains(parentBlock))
    return std::nullopt;
  const uint64_t hash = indexHash(parentBlock ? block(parentBlock).indexHash : 0, tokens, identity);
  const auto [first, last] = index_.equal_range(hash);
  for (auto candidate = first; candidate != last; ++candidate) {
    const Block &entry = block(candidate->second);
    // Hashes filter candidates; equality requires the complete key.
    if (!entry.poisoned && entry.parent == parentBlock && entry.identity == identity &&
        std::equal(entry.tokens.begin(), entry.tokens.end(), tokens.begin())) {
      return BlockMatch{entry.id, entry.page};
    }
  }
  return std::nullopt;
}

KvCache::BlockMatch KvCache::insert(uint64_t parentBlock,
                                    std::span<const uint32_t> tokens,
                                    uint32_t physicalPage,
                                    BlockIdentity identity,
                                    std::span<const uint32_t> window) {
  if (tokens.size() != pageTokens) {
    throw std::invalid_argument("KV cache block must contain one full page");
  }
  if (physicalPage >= pool_.pageCount()) {
    throw std::out_of_range("KV cache physical page is out of range");
  }
  if (auto existing = find(parentBlock, tokens, identity)) {
    BlockMatch result{existing->id, existing->physicalPage};
    Block &found = block(existing->id);
    if (existing->physicalPage == noPage) {
      adoptPage(existing->id, physicalPage, found.windowPins ? window : std::span<const uint32_t>{});
      // The copy, which holds what another command computed, stays: a
      // restore point's chain on disk may need it. The block spills only
      // once a copy of the writer's page replaced it.
      Block &adopted = block(existing->id);
      setForeignCopy(adopted, true);
      reindex(adopted);
      result.physicalPage = physicalPage;
    } else if (found.transfer == Transfer::Filling) {
      // Its page is still being filled; the writer keeps its own copy. A
      // page being written holds its final bytes, and the writer moves
      // onto it.
      result.physicalPage = physicalPage;
    } else if (windowPages_) {
      // With a window the writer keeps its page, which its window reads
      // whole, and the block takes it, once no write reads its own: K/V
      // another command computed can differ in its last bits, and a row's
      // window would read them beside its own command's.
      if (found.transfer == Transfer::None) {
        result.previousPage = found.page;
        upgrade(found, physicalPage, window);
      }
      result.physicalPage = physicalPage;
    }
    return result;
  }
  if (parentBlock && !blocks_.contains(parentBlock)) {
    throw std::invalid_argument("KV cache parent block is unknown");
  }
  if (parentBlock && block(parentBlock).page == noPage) {
    throw std::logic_error("KV cache parent block has no page");
  }
  if (parentBlock &&
      block(parentBlock).children == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("KV cache child count overflowed");
  }
  if (!nextBlockId_ || nextBlockId_ == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("KV cache block ids exhausted");
  }
  if (blockOnPage_[physicalPage])
    throw std::logic_error("KV cache page already holds a block");

  Block entry;
  entry.id = nextBlockId_;
  entry.parent = parentBlock;
  entry.indexHash = indexHash(parentBlock ? block(parentBlock).indexHash : 0, tokens, identity);
  std::copy(tokens.begin(), tokens.end(), entry.tokens.begin());
  entry.identity = identity;
  entry.depth = parentBlock ? block(parentBlock).depth + 1 : 1;
  entry.ramNode = RecencyOrder::allocate();
  entry.diskNode = RecencyOrder::allocate();
  entry.unneededNode = RecencyOrder::allocate();
  entry.unwrittenNode = RecencyOrder::allocate();
  entry.windowNode = RecencyOrder::allocate();
  // No state's tail holds a new block yet: the writer keeps its unit.
  own(physicalPage, {});
  auto [position, unique] = blocks_.emplace(entry.id, std::move(entry));
  if (!unique)
    throw std::logic_error("duplicate KV cache block id");
  index_.emplace(position->second.indexHash, position->first);
  const uint64_t id = nextBlockId_++;
  Block &placed = block(id);
  placeOnPage(placed, physicalPage);
  if (parentBlock) {
    Block &parent = block(parentBlock);
    ++parent.children;
    ++parent.residentChildren;
    if (parent.firstChild)
      block(parent.firstChild).previousSibling = id;
    placed.nextSibling = parent.firstChild;
    parent.firstChild = id;
    reindex(parent);
    historyChanged(parentBlock);
  }
  placed.lastUsed = recency_.next();
  reindex(placed);
  ++generation_;
  return {id, physicalPage};
}

bool KvCache::adoptDiskBlock(uint64_t blockId, uint64_t parentBlock,
                             std::span<const uint32_t> tokens, BlockIdentity identity,
                             std::shared_ptr<KvDiskSlot> slot, uint64_t lastUsed) {
  if (!blockId || blockId == std::numeric_limits<uint64_t>::max() || blocks_.contains(blockId) ||
      tokens.size() != pageTokens || !slot)
    throw std::invalid_argument("invalid adopted KV cache block");
  if (parentBlock && !blocks_.contains(parentBlock))
    throw std::invalid_argument("adopted KV cache block's parent is unknown");
  if (find(parentBlock, tokens, identity))
    return false;
  if (parentBlock && block(parentBlock).children == std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("KV cache child count overflowed");
  Block entry;
  entry.id = blockId;
  entry.parent = parentBlock;
  entry.indexHash = indexHash(parentBlock ? block(parentBlock).indexHash : 0, tokens, identity);
  std::copy(tokens.begin(), tokens.end(), entry.tokens.begin());
  entry.identity = identity;
  entry.depth = parentBlock ? block(parentBlock).depth + 1 : 1;
  entry.lastUsed = lastUsed;
  entry.ramNode = RecencyOrder::allocate();
  entry.diskNode = RecencyOrder::allocate();
  entry.unneededNode = RecencyOrder::allocate();
  entry.unwrittenNode = RecencyOrder::allocate();
  entry.windowNode = RecencyOrder::allocate();
  Block &placed = blocks_.emplace(blockId, std::move(entry)).first->second;
  index_.emplace(placed.indexHash, blockId);
  giveDiskCopy(placed, std::move(slot));
  if (parentBlock) {
    Block &parent = block(parentBlock);
    ++parent.children;
    if (parent.firstChild)
      block(parent.firstChild).previousSibling = blockId;
    placed.nextSibling = parent.firstChild;
    parent.firstChild = blockId;
    reindex(parent);
  }
  nextBlockId_ = std::max(nextBlockId_, blockId + 1);
  reindex(placed);
  ++generation_;
  return true;
}

void KvCache::continueIdsAfter(uint64_t blockId) noexcept {
  if (blockId != std::numeric_limits<uint64_t>::max())
    nextBlockId_ = std::max(nextBlockId_, blockId + 1);
}

KvCache::Key KvCache::key(uint64_t blockId) const {
  const Block &entry = block(blockId);
  return {entry.parent, entry.tokens, entry.identity};
}

uint64_t KvCache::lastUsed(uint64_t blockId) const { return block(blockId).lastUsed; }

KvCache::Transfer KvCache::transfer(uint64_t blockId) const { return block(blockId).transfer; }

std::vector<uint64_t> KvCache::children(uint64_t blockId) const {
  std::vector<uint64_t> result;
  for (uint64_t child = block(blockId).firstChild; child; child = block(child).nextSibling)
    result.push_back(child);
  return result;
}

void KvCache::retainActive(uint64_t blockId) {
  Block &entry = block(blockId);
  if (entry.activeUsers == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("KV cache active user count overflowed");
  }
  ++entry.activeUsers;
  entry.lastUsed = recency_.next();
  reindex(entry);
}

void KvCache::releaseActive(uint64_t blockId) noexcept {
  auto found = blocks_.find(blockId);
  if (found == blocks_.end() || !found->second.activeUsers)
    std::terminate();
  Block &entry = found->second;
  --entry.activeUsers;
  // A released leaf is the newest; a parent keeps the recency its children
  // pass on.
  if (!entry.activeUsers && !entry.children)
    entry.lastUsed = recency_.next();
  reindex(entry);
  erasePoisonedLeaf(blockId);
}

void KvCache::refreshCopies(std::span<const uint64_t> chain) noexcept {
  for (const uint64_t id : chain) {
    const auto found = blocks_.find(id);
    if (found == blocks_.end())
      std::terminate();
    if (!found->second.slot)
      continue;
    found->second.copyUsed = recency_.next();
    reindex(found->second);
  }
}

void KvCache::touch(uint64_t blockId) noexcept {
  auto found = blocks_.find(blockId);
  if (found == blocks_.end())
    std::terminate();
  found->second.lastUsed = recency_.next();
  reindex(found->second);
}

KvCache::Chain KvCache::chain(uint64_t blockId) const {
  const uint32_t depth = chainLength(blockId);
  Chain result;
  result.blocks.resize(depth);
  result.pages.resize(depth);
  for (uint32_t index = depth; index > 0; --index) {
    const Block &entry = block(blockId);
    result.blocks[index - 1] = entry.id;
    result.pages[index - 1] = entry.page;
    blockId = entry.parent;
  }
  if (blockId)
    throw std::logic_error("KV cache chain exceeds its depth");
  return result;
}

bool KvCache::contains(uint64_t blockId) const noexcept {
  return blocks_.contains(blockId);
}

bool KvCache::matchable(uint64_t blockId) const noexcept {
  const auto found = blocks_.find(blockId);
  return found != blocks_.end() && !found->second.poisoned;
}

uint32_t KvCache::chainLength(uint64_t blockId) const {
  return block(blockId).depth;
}

uint32_t KvCache::page(uint64_t blockId) const { return block(blockId).page; }

std::shared_ptr<KvDiskSlot> KvCache::slot(uint64_t blockId) const {
  return block(blockId).slot;
}

bool KvCache::kvCopied(uint64_t blockId) const {
  const Block &entry = block(blockId);
  return copyHoldsPages(entry) && !entry.foreignCopy;
}

bool KvCache::hasDiskChildren(uint64_t blockId) const {
  const Block &entry = block(blockId);
  return entry.children > entry.residentChildren;
}

template <typename Count>
void KvCache::countAbove(uint64_t blockId, const Count &count) noexcept {
  const auto found = blocks_.find(blockId);
  if (found == blocks_.end())
    std::terminate();
  for (uint64_t above = found->second.parent; above;) {
    const auto parent = blocks_.find(above);
    if (parent == blocks_.end())
      std::terminate();
    count(parent->second);
    above = parent->second.parent;
  }
}

void KvCache::countState(uint64_t blockId, bool added, bool inUse) noexcept {
  const auto found = blocks_.find(blockId);
  if (found == blocks_.end())
    std::terminate();
  added ? ++found->second.statesHere : --found->second.statesHere;
  reindex(found->second);
  countAbove(blockId, [&](Block &entry) {
    added ? ++entry.statesBelow : --entry.statesBelow;
    // The first state below a leaf, or in use below it, or the last, changes
    // its kind.
    bool kindChanged = entry.statesBelow == (added ? 1u : 0u);
    if (inUse) {
      added ? ++entry.statesInUseBelow : --entry.statesInUseBelow;
      kindChanged = kindChanged || entry.statesInUseBelow == (added ? 1u : 0u);
    }
    // And whether its copy is needed (unneededCopy).
    if (kindChanged)
      reindex(entry);
  });
}

void KvCache::countRamState(uint64_t blockId, bool added, const TailUnits &units) noexcept {
  // The state's tail, a bounded walk.
  uint64_t id = blockId;
  for (uint32_t pinned = 0; id && pinned < windowPages_; ++pinned) {
    const auto tail = blocks_.find(id);
    if (tail == blocks_.end())
      std::terminate();
    Block &entry = tail->second;
    const bool before = entry.windowPins || entry.copyPins;
    added ? ++entry.windowPins : --entry.windowPins;
    pinnedBlocks_ += (entry.windowPins || entry.copyPins) - before;
    reunit(entry, units);
    reindex(entry);
    id = entry.parent;
  }
}

void KvCache::reunit(Block &entry, const TailUnits &units) noexcept {
  if (!unitPages_)
    return;
  const bool needs = needsUnit(entry);
  if (needs && entry.window.empty()) {
    // A block read failed (poisoned) leaves with what lies below it.
    const auto unit = std::ranges::find(units, entry.id, &TailUnits::value_type::first);
    if (unit == units.end()) {
      if (entry.poisoned)
        return;
      std::terminate();
    }
    try {
      takeUnit(entry, unit->second);
    } catch (...) {
      std::terminate();
    }
  } else if (!needs && !entry.window.empty() && entry.transfer == Transfer::None) {
    if (deferring_)
      unpinned_.push_back(entry.id);
    else
      giveUnitUp(entry);
  }
}

KvCache::DeferredReleases::~DeferredReleases() {
  kv_.deferring_ = false;
  for (const uint64_t id : std::exchange(kv_.unpinned_, {}))
    if (const auto found = kv_.blocks_.find(id); found != kv_.blocks_.end()) {
      Block &entry = found->second;
      if (!needsUnit(entry) && !entry.window.empty() && entry.transfer == Transfer::None)
        kv_.giveUnitUp(entry);
    }
}

void KvCache::takeUnit(Block &entry, std::span<const uint32_t> window) {
  if (entry.page == noPage || !entry.window.empty() || window.size() != unitPages_)
    throw std::logic_error("a block takes a unit only on its page and while it holds none");
  for (const uint32_t each : window) {
    if (each >= pool_.pageCount() || blockOnPage_[each])
      throw std::logic_error("KV cache window page is out of range or holds a block");
    pool_.retainPage(each, true);
    blockOnPage_[each] = entry.id;
  }
  entry.window.assign(window.begin(), window.end());
  ++windowUnits_;
  historyChanged(entry.id, Change::Unit);
  // Whether its copy holds its pages changes with what it holds.
  reindex(entry);
}

void KvCache::giveUnitUp(Block &entry) noexcept {
  const std::vector<uint32_t> window = std::move(entry.window);
  entry.window.clear();
  --windowUnits_;
  historyChanged(entry.id, Change::Unit);
  // Its pages go free unless a lane holds them, which lanes hold together.
  unitsFreed_ += !pool_.pageActive(window.front());
  for (const uint32_t each : window) {
    blockOnPage_[each] = 0;
    try {
      pool_.releasePage(each, true);
    } catch (...) {
      std::terminate();
    }
  }
  reindex(entry);
}

KvCache::Freed KvCache::freedBy(uint64_t stateBlock, Gone &gone, PinRelease release) const {
  Freed freed;
  uint64_t id = stateBlock;
  for (uint32_t index = 0; id && index < windowPages_; ++index) {
    const Block &entry = block(id);
    auto &[memory, disk] = gone[id];
    const auto needs = [&](uint32_t memoryGone, int32_t diskAdded) {
      return entry.windowPins > memoryGone ||
             (static_cast<int64_t>(entry.copyPins) + diskAdded > 0 && !windowCopied(entry));
    };
    const bool before = needs(memory, disk), inMemory = entry.windowPins > memory;
    memory += release.ram ? 1 : 0;
    disk += (release.written ? 1 : 0) - (release.disk ? 1 : 0);
    const bool idle = !entry.window.empty() && entry.transfer == Transfer::None &&
                      std::ranges::none_of(entry.window, [&](uint32_t page) { return pool_.pageActive(page); });
    if (idle && before && !needs(memory, disk))
      freed.now.push_back(id);
    else if (idle && inMemory && entry.windowPins <= memory && !pool_.pageActive(entry.page) && !entry.activeUsers &&
             !entry.poisoned)
      freed.written.push_back(id);
    else if (!entry.window.empty() && entry.transfer == Transfer::Draining && inMemory &&
             entry.windowPins <= memory && windowCopied(entry))
      freed.landing.push_back(id);
    id = entry.parent;
  }
  return freed;
}

void KvCache::countStateInUse(uint64_t blockId, bool added) noexcept {
  countAbove(blockId, [&](Block &entry) {
    added ? ++entry.statesInUseBelow : --entry.statesInUseBelow;
    if (entry.statesInUseBelow == (added ? 1u : 0u))
      placeLeaf(entry);
  });
}

void KvCache::countDiskState(uint64_t blockId, bool added, const TailUnits &units) noexcept {
  // The tail's disk pins: its blocks' window parts are needed, or their
  // units until those are written.
  uint64_t id = blockId;
  for (uint32_t pinned = 0; id && pinned < windowPages_; ++pinned) {
    const auto tail = blocks_.find(id);
    if (tail == blocks_.end())
      std::terminate();
    Block &entry = tail->second;
    const bool before = entry.windowPins || entry.copyPins;
    added ? ++entry.copyPins : --entry.copyPins;
    pinnedBlocks_ += (entry.windowPins || entry.copyPins) - before;
    reunit(entry, units);
    reindex(entry);
    id = entry.parent;
  }
}

bool KvCache::unwrittenUnit(const Block &entry) const noexcept {
  return unitLeaving(entry) && entry.transfer == Transfer::None;
}

bool KvCache::unitLeaving(const Block &entry) const noexcept {
  if (entry.window.empty() || entry.windowPins || entry.poisoned)
    return false;
  // A write lands whoever holds the base page; a write starts on an idle one.
  if (entry.transfer != Transfer::None)
    return entry.transfer == Transfer::Draining && !needsUnit(entry);
  return entry.copyPins && !windowCopied(entry) && !entry.activeUsers && !pool_.pageActive(entry.page);
}

std::optional<CacheEvictionCandidate> KvCache::unwrittenUnitCandidate(uint64_t after) const {
  return after ? unwritten_.next({after, block(after).lastUsed}) : unwritten_.oldest();
}

std::optional<CacheEvictionCandidate> KvCache::unneededWindowCandidate(uint64_t after) const {
  return passing(unneededWindows_, after);
}

void KvCache::partGivenUp(uint64_t blockId) {
  Block &entry = block(blockId);
  if (!entry.slot || !entry.slot->parts())
    throw std::logic_error("a block's copy gave up a part it lacks, or its last");
  const uint64_t after = entry.slot->bytes();
  diskBytes_ = diskBytes_ - entry.copyBytes + after;
  if (entry.held && !entry.foreignCopy)
    heldCopyBytes_ = heldCopyBytes_ - entry.copyBytes + after;
  entry.copyBytes = after;
  reunit(entry);
  reindex(entry);
  historyChanged(blockId, Change::Copy);
}

bool KvCache::stateBelow(uint64_t blockId) const { return block(blockId).statesBelow > 0; }

uint32_t KvCache::statesBelow(uint64_t blockId) const { return block(blockId).statesBelow; }

bool KvCache::stateInUseBelow(uint64_t blockId) const {
  return block(blockId).statesInUseBelow > 0;
}

void KvCache::noteState(uint64_t blockId) { block(blockId).hadState = true; }

bool KvCache::hadState(uint64_t blockId) const { return block(blockId).hadState; }

void KvCache::noteReply(uint64_t blockId) {
  block(blockId).heldReply = true;
  historyChanged(blockId);
}

bool KvCache::heldReply(uint64_t blockId) const { return block(blockId).heldReply; }

KvCache::Keeps KvCache::keeps(uint64_t blockId) const { return block(blockId).keeps; }

void KvCache::setStateMarks(uint64_t blockId, StateMarks state, bool inRam) noexcept {
  const auto found = blocks_.find(blockId);
  if (found == blocks_.end())
    std::terminate();
  Block &entry = found->second;
  if (entry.state == state && entry.stateInRam == inRam)
    return;
  entry.state = state;
  entry.stateInRam = inRam;
  // Whether the state is in use, or in RAM, places the leaf too.
  placeLeaf(entry);
  rekeep(entry);
}

void KvCache::countWanted(uint64_t blockId, bool added) noexcept {
  const auto found = blocks_.find(blockId);
  if (found == blocks_.end() || (!added && !found->second.wanted))
    std::terminate();
  Block &entry = found->second;
  added ? ++entry.wanted : --entry.wanted;
  if (entry.wanted == (added ? 1u : 0u))
    rekeep(entry);
}

void KvCache::moveResumePoint(uint64_t from, uint64_t to) noexcept {
  if (const auto found = blocks_.find(from); from && found != blocks_.end()) {
    found->second.resumePoint = false;
    rekeep(found->second);
  }
  if (!to)
    return;
  const auto found = blocks_.find(to);
  if (found == blocks_.end())
    std::terminate();
  found->second.resumePoint = true;
  rekeep(found->second);
}

KvCache::Reasons KvCache::reasons(const Block &entry, const std::array<uint32_t, kKeeps> &children,
                                  uint32_t neededChildren) const noexcept {
  const bool resident = entry.page != noPage;
  const bool active = resident && pool_.pageActive(entry.page);
  Reasons result{.here = entry.state.keeps, .needed = entry.state.needed || neededChildren};
  if (resident && (entry.transfer == Transfer::Filling || entry.activeUsers || active ||
                   (entry.poisoned && !entry.residentChildren)))
    result.here |= keepBit(Keep::Busy);
  if (resident && entry.transfer == Transfer::Draining)
    result.here |= keepBit(Keep::Moving);
  if (resident && !copyHoldsPages(entry) && result.needed)
    result.here |= keepBit(Keep::Uncopied);
  if (entry.wanted)
    result.here |= keepBit(Keep::Wanted);
  if (entry.resumePoint)
    result.here |= keepBit(Keep::ResumePoint);
  result.keeps = result.here;
  for (size_t each = 0; each < kKeeps; ++each)
    if (children[each])
      result.keeps |= keepBit(static_cast<Keep>(each));
  // The chains a reclaim keeps count no page from one a request holds up.
  if (active)
    result.keeps &= static_cast<Keeps>(~kChainKeeps);
  if (resident) {
    result.counted =
        keepCount(entry.form, !entry.window.empty(), result.keeps, !active && entry.transfer == Transfer::None);
    if (entry.state.candidate)
      result.candidate = result.keeps;
  }
  return result;
}

void KvCache::rekeep(Block &start) noexcept {
  for (Block *entry = &start;;) {
    const auto [here, keeps, needed, counted, candidate] = reasons(*entry, entry->keepChildren, entry->neededChildren);
    entry->keepsHere = here;
    // Under its reasons and whether it is idle (no transfer moves it).
    if (const uint16_t unwritten =
            unitLeaving(*entry) ? static_cast<uint16_t>(keeps | (entry->transfer == Transfer::None ? 1u << kKeeps : 0u))
                                : noCount;
        unwritten != entry->unwrittenCounted) {
      if (entry->unwrittenCounted != noCount)
        --unwrittenCounts_[entry->unwrittenCounted];
      if (unwritten != noCount)
        ++unwrittenCounts_[unwritten];
      entry->unwrittenCounted = unwritten;
    }
    if (candidate != entry->candidateCounted) {
      if (entry->candidateCounted != noCount)
        --candidateCounts_[entry->candidateCounted];
      if (candidate != noCount)
        ++candidateCounts_[candidate];
      entry->candidateCounted = candidate;
    }
    if (counted != entry->counted) {
      if (entry->counted != noCount)
        --keepCounts_[entry->counted];
      if (counted != noCount)
        ++keepCounts_[counted];
      entry->counted = counted;
    }
    const Keeps changed = keeps ^ entry->keeps;
    const bool neededChanged = needed != entry->needed;
    entry->keeps = keeps;
    entry->needed = needed;
    if ((!changed && !neededChanged) || !entry->parent)
      return;
    const auto parent = blocks_.find(entry->parent);
    if (parent == blocks_.end())
      std::terminate();
    for (size_t each = 0; each < kKeeps; ++each)
      if (const Keeps bit = keepBit(static_cast<Keep>(each)); changed & bit)
        keeps & bit ? ++parent->second.keepChildren[each] : --parent->second.keepChildren[each];
    if (neededChanged)
      needed ? ++parent->second.neededChildren : --parent->second.neededChildren;
    entry = &parent->second;
  }
}

void KvCache::pageActivity(uint32_t page) noexcept {
  if (const uint64_t id = page < blockOnPage_.size() ? blockOnPage_[page] : 0) {
    const auto found = blocks_.find(id);
    if (found == blocks_.end())
      std::terminate();
    // Requests hold a block's window pages with its base page.
    if (found->second.page == page)
      reindex(found->second);
  }
}

void KvCache::forget(Block &entry) noexcept {
  if (entry.counted != noCount)
    --keepCounts_[entry.counted];
  entry.counted = noCount;
  if (entry.candidateCounted != noCount)
    --candidateCounts_[entry.candidateCounted];
  entry.candidateCounted = noCount;
  if (entry.unwrittenCounted != noCount)
    --unwrittenCounts_[entry.unwrittenCounted];
  entry.unwrittenCounted = noCount;
  if (entry.parent) {
    const auto parent = blocks_.find(entry.parent);
    if (parent == blocks_.end())
      std::terminate();
    for (size_t each = 0; each < kKeeps; ++each)
      if (entry.keeps & keepBit(static_cast<Keep>(each)))
        --parent->second.keepChildren[each];
    if (entry.needed)
      --parent->second.neededChildren;
  }
  entry.keeps = 0;
  entry.needed = false;
}

KvCache::PageReach KvCache::pageReach(Keeps mask, std::span<const uint64_t> kept, std::span<const uint64_t> unwaited,
                                      uint64_t written, std::span<const uint64_t> released,
                                      std::span<const uint64_t> toWrite, bool writable,
                                      std::span<const uint64_t> staged, std::span<const uint64_t> unitsKept) const {
  PageReach reach = countReach(mask, kept, unwaited, written, released, toWrite, writable, staged, unitsKept);
  if (checkReaches_) {
    const PageReach rule = reachByRule(mask, kept, unwaited, written, released, toWrite, writable, staged, unitsKept);
    if (reach != rule) {
      const auto forms = [](const std::array<uint32_t, kv::kPageForms> &pages) {
        return std::to_string(pages[0]) + "/" + std::to_string(pages[1]);
      };
      throw std::logic_error("a reach's counts differ from its rule: now " + forms(reach.now) + " vs " +
                             forms(rule.now) + ", written " + forms(reach.written) + " vs " + forms(rule.written) +
                             ", reachable " + forms(reach.reachable) + " vs " + forms(rule.reachable) +
                             ", candidates " + std::to_string(reach.candidates) + " vs " +
                             std::to_string(rule.candidates));
    }
  }
  return reach;
}

KvCache::PageReach KvCache::reachByRule(Keeps mask, std::span<const uint64_t> kept, std::span<const uint64_t> unwaited,
                                        uint64_t written, std::span<const uint64_t> released,
                                        std::span<const uint64_t> toWrite, bool writable,
                                        std::span<const uint64_t> staged, std::span<const uint64_t> unitsKept) const {
  const std::unordered_set<uint64_t> keptUnits(unitsKept.begin(), unitsKept.end());
  constexpr Keeps busy = keepBit(Keep::Busy);
  const Keeps stays = static_cast<Keeps>(busy | (mask & kChainKeeps));
  // Whether the block is one of `chains`' blocks or above one.
  const auto on = [&](std::span<const uint64_t> chains, uint64_t id) {
    for (const uint64_t last : chains)
      for (uint64_t at = last; at; at = block(at).parent)
        if (at == id)
          return true;
    return false;
  };
  // Once the written state has left RAM through its write: the reasons at
  // each block and below it, its own write gone. Up its chain while each
  // block frees, it needing every block as a state on disk does: one
  // without a copy stays where the walk keeps those.
  std::unordered_set<uint64_t> writtenChain;
  if (written) {
    const Keeps kind = block(written).state.keeps & (keepBit(Keep::Writer) | keepBit(Keep::UsedWriter));
    const auto after = [&](const auto &self, uint64_t id) -> Keeps {
      const Block &entry = block(id);
      Keeps keeps = id == written ? static_cast<Keeps>(entry.keepsHere & ~kind) : entry.keepsHere;
      for (const uint64_t child : children(id))
        keeps |= self(self, child);
      if (entry.page != noPage && pool_.pageActive(entry.page))
        keeps &= static_cast<Keeps>(~kChainKeeps);
      return keeps;
    };
    bool uncopied = false;
    for (uint64_t id = written; id; id = block(id).parent) {
      const Block &entry = block(id);
      uncopied = uncopied || !copyHoldsPages(entry);
      const Keeps keeps = static_cast<Keeps>(after(after, id) | (uncopied ? keepBit(Keep::Uncopied) : 0));
      if (entry.page == noPage || (keeps & mask) || on(kept, id) || on(unwaited, id))
        break;
      writtenChain.insert(id);
    }
  }
  const std::unordered_set<uint64_t> releasing(released.begin(), released.end()),
      writing(toWrite.begin(), toWrite.end()), staging(staged.begin(), staged.end());
  PageReach reach;
  for (const auto &[id, entry] : blocks_) {
    if (entry.page == noPage)
      continue;
    const auto form = static_cast<size_t>(entry.form);
    const bool idle = !pool_.pageActive(entry.page) && entry.transfer == Transfer::None;
    const bool keptChain = on(kept, id), aboveWrite = on(unwaited, id);
    const bool now = !(entry.keeps & mask) && !keptChain && !aboveWrite;
    const bool reachable = !(entry.keeps & (idle ? stays : busy)) && !(keptChain && idle) && !aboveWrite;
    const bool inWritten = writtenChain.contains(id);
    // Its unit frees with its page, now where the walk's releases take its
    // last pin, or once a reclaim writes its window part.
    const uint32_t unit = entry.window.empty() ? 0 : unitPages_;
    const bool unitNow = releasing.contains(id);
    const bool unitWritten = (unitLeaving(entry) && (entry.transfer != Transfer::None ||
                                                     (writable && !(entry.keeps & stays & ~busy) && !keptUnits.contains(id)))) ||
                             writing.contains(id);
    const bool unitStaged = staging.contains(id);
    reach.now[form] += (now ? 1 : 0) + (now || unitNow ? unit : 0);
    reach.written[form] += (inWritten ? 1 : 0) + ((inWritten || unitStaged) && !now && !unitNow ? unit : 0);
    reach.reachable[form] += (reachable ? 1 : 0) + (reachable || unitNow || unitWritten || unitStaged ? unit : 0);
    reach.candidates += now && entry.state.candidate ? 1 : 0;
  }
  return reach;
}

KvCache::PageReach KvCache::countReach(Keeps mask, std::span<const uint64_t> kept, std::span<const uint64_t> unwaited,
                                       uint64_t written, std::span<const uint64_t> released,
                                       std::span<const uint64_t> toWrite, bool writable,
                                       std::span<const uint64_t> staged, std::span<const uint64_t> unitsKept) const {
  const std::unordered_set<uint64_t> keptUnits(unitsKept.begin(), unitsKept.end());
  // A unit that leaves by its window part's write, unless a chain the walk
  // keeps holds it or the walk keeps a point it serves, or as that write
  // lands.
  const auto leaving = [&](const Block &entry) {
    return unitLeaving(entry) && (entry.transfer != Transfer::None ||
                                  (writable && !(entry.keeps & mask & kChainKeeps) && !keptUnits.contains(entry.id)));
  };
  constexpr Keeps busy = keepBit(Keep::Busy);
  const Keeps chains = mask & kChainKeeps;
  // What no reclaim of the walk reaches: a page nothing reaches below which
  // something is busy, and an idle one on a chain the walk keeps. A page a
  // write drains frees as the write lands, kept or not.
  const Keeps stays = static_cast<Keeps>(busy | chains);
  const auto reaches = [&](Keeps keeps, bool idle) { return !(keeps & (idle ? stays : busy)); };
  // A block's pages: its base page and its unit, if it holds one.
  const auto pagesOf = [&](const Block &entry) { return 1 + (entry.window.empty() ? 0 : unitPages_); };
  PageReach reach;
  for (unsigned keeps = 0; keeps < (1u << kKeeps); ++keeps) {
    for (const kv::PageForm form : kv::kAllPageForms)
      for (const bool unit : {false, true})
        for (const bool idle : {false, true}) {
          const uint32_t pages = keepCounts_[keepCount(form, unit, static_cast<Keeps>(keeps), idle)] *
                                 (1 + (unit ? unitPages_ : 0));
          if (!(keeps & mask))
            reach.now[static_cast<size_t>(form)] += pages;
          if (reaches(static_cast<Keeps>(keeps), idle))
            reach.reachable[static_cast<size_t>(form)] += pages;
        }
    if (!(keeps & mask))
      reach.candidates += candidateCounts_[keeps];
    // A unit that leaves by its window part's write, or as that write lands,
    // whose base page no reclaim reaches.
    if (writable && (keeps & busy) && !(keeps & chains))
      reach.reachable[static_cast<size_t>(kv::PageForm::Full)] += unwrittenCounts_[keeps | 1u << kKeeps] * unitPages_;
    if (!reaches(static_cast<Keeps>(keeps), false))
      reach.reachable[static_cast<size_t>(kv::PageForm::Full)] += unwrittenCounts_[keeps] * unitPages_;
  }
  // Up the chains of `kept`, which the walk keeps: an idle page stays, and
  // none frees now; past a page a request holds, or one a chain `mask`
  // names keeps, every block is so already. Then above each write no
  // reclaim waits for, which holds every page above it until it lands, up
  // to one something busy holds already. Each block counts out once.
  std::unordered_set<uint64_t> passed, outOfReach, outOfNow;
  const auto cut = [&](uint64_t id, bool keptChain) {
    while (id) {
      ++reachVisits_;
      passed.insert(id);
      const Block &entry = block(id);
      if (entry.page != noPage) {
        const bool idle = !pool_.pageActive(entry.page) && entry.transfer == Transfer::None;
        if (entry.keeps & (keptChain ? stays : busy))
          break;
        const auto form = static_cast<size_t>(entry.form);
        if (reaches(entry.keeps, idle) && (!keptChain || idle) && outOfReach.insert(id).second)
          reach.reachable[form] -= leaving(entry) ? 1 : pagesOf(entry);
        if (!(entry.keeps & mask) && outOfNow.insert(id).second) {
          reach.now[form] -= pagesOf(entry);
          reach.candidates -= entry.state.candidate ? 1 : 0;
        }
      }
      id = entry.parent;
    }
  };
  for (const uint64_t state : kept)
    cut(state, true);
  for (const uint64_t moving : unwaited)
    cut(moving, false);
  // Up the written state's chain while its blocks free once it has left RAM:
  // its write no longer keeps them, and it needs a block without a copy as
  // a state on disk does.
  std::unordered_set<uint64_t> writtenChain;
  if (written) {
    const Keeps kind = block(written).state.keeps & (keepBit(Keep::Writer) | keepBit(Keep::UsedWriter));
    const auto kindIndex = static_cast<size_t>(kind == keepBit(Keep::Writer) ? Keep::Writer : Keep::UsedWriter);
    if (!kind || !(kind & mask))
      throw std::logic_error("the state a reach writes is no state its walk writes");
    bool below = false, uncopied = false;
    for (uint64_t id = written; id;) {
      ++reachVisits_;
      const Block &entry = block(id);
      const bool writes =
          (id != written && (entry.keepsHere & kind)) || entry.keepChildren[kindIndex] > (below ? 1u : 0u);
      uncopied = uncopied || !copyHoldsPages(entry);
      Keeps keeps = writes ? entry.keeps : static_cast<Keeps>(entry.keeps & ~kind);
      if (uncopied)
        keeps |= keepBit(Keep::Uncopied);
      if (entry.page == noPage || (keeps & mask) || passed.contains(id))
        break;
      reach.written[static_cast<size_t>(entry.form)] += pagesOf(entry);
      writtenChain.insert(id);
      below = true;
      id = entry.parent;
    }
  }
  // The units the walk's releases free now, whose blocks the counts above
  // free later or not at all: each moves to now, and out of written.
  for (const uint64_t id : released) {
    ++reachVisits_;
    const Block &entry = block(id);
    if (entry.page == noPage || entry.window.empty())
      throw std::logic_error("a released block holds no unit");
    const auto form = static_cast<size_t>(entry.form);
    const bool idle = !pool_.pageActive(entry.page) && entry.transfer == Transfer::None;
    if ((entry.keeps & mask) || outOfNow.contains(id))
      reach.now[form] += unitPages_;
    if (writtenChain.contains(id))
      reach.written[form] -= unitPages_;
    // An unwritten unit's counts already whatever keeps its page.
    if ((!reaches(entry.keeps, idle) || outOfReach.contains(id)) && !leaving(entry))
      reach.reachable[form] += unitPages_;
  }
  // The units the staged state frees as it leaves RAM through its write,
  // whose blocks free neither now nor then.
  for (const uint64_t id : staged) {
    ++reachVisits_;
    const Block &entry = block(id);
    if (entry.page == noPage || entry.window.empty())
      throw std::logic_error("a block the staged state frees holds no unit");
    const auto form = static_cast<size_t>(entry.form);
    const bool idle = !pool_.pageActive(entry.page) && entry.transfer == Transfer::None;
    if (((entry.keeps & mask) || outOfNow.contains(id)) && !writtenChain.contains(id))
      reach.written[form] += unitPages_;
    if (!reaches(entry.keeps, idle) || outOfReach.contains(id))
      reach.reachable[form] += unitPages_;
  }
  // The unwritten units counted above whose points the walk keeps: out.
  for (const uint64_t id : keptUnits) {
    ++reachVisits_;
    const Block &entry = block(id);
    if (!writable || !unwrittenUnit(entry) || (entry.keeps & mask & kChainKeeps))
      continue;
    if (entry.keeps & busy)
      reach.reachable[static_cast<size_t>(entry.form)] -= unitPages_;
  }
  // The units the walk's releases leave to a reclaim that writes their
  // window parts, whose blocks the counts above reach later or not at all.
  for (const uint64_t id : toWrite) {
    ++reachVisits_;
    const Block &entry = block(id);
    if (entry.page == noPage || entry.window.empty() || unitLeaving(entry))
      throw std::logic_error("a block a release leaves to a write holds no unit, or one counted already");
    const bool idle = !pool_.pageActive(entry.page) && entry.transfer == Transfer::None;
    if (!reaches(entry.keeps, idle) || outOfReach.contains(id))
      reach.reachable[static_cast<size_t>(entry.form)] += unitPages_;
  }
  return reach;
}

std::array<uint32_t, kv::kPageForms> KvCache::keptPages(Keeps chains) const {
  std::array<uint32_t, kv::kPageForms> pages{};
  for (const kv::PageForm form : kv::kAllPageForms)
    for (unsigned keeps = 0; keeps < (1u << kKeeps); ++keeps)
      if (keeps & chains)
        for (const bool unit : {false, true})
          pages[static_cast<size_t>(form)] +=
              keepCounts_[keepCount(form, unit, static_cast<Keeps>(keeps), true)] * (1 + (unit ? unitPages_ : 0));
  return pages;
}

bool KvCache::residentLeaf(uint64_t blockId) const {
  const Block &entry = block(blockId);
  return entry.page != noPage && !entry.residentChildren && !entry.activeUsers &&
         !entry.held && !entry.poisoned;
}

bool KvCache::leafOnceChildrenLeave(uint64_t blockId) const {
  const Block &entry = block(blockId);
  return entry.page != noPage && !entry.activeUsers && !entry.held && !entry.poisoned;
}

void KvCache::setTransfer(uint64_t blockId, Transfer transfer) {
  Block &entry = block(blockId);
  if ((entry.transfer == Transfer::None) == (transfer == Transfer::None))
    throw std::logic_error("KV cache block transfer state did not change");
  entry.transfer = transfer;
  // A copy lands (copyLanded) as its transfer ends.
  historyChanged(blockId, Change::Copy);
  // A unit a transfer moved goes with it once the block needs it no longer.
  if (transfer == Transfer::None)
    reunit(entry);
  reindex(entry);
}

void KvCache::holdPage(uint32_t page) {
  const bool first = !pool_.pageActive(page);
  pool_.retainPage(page, false);
  if (const uint64_t id = blockOnPage_[page]; spill_ && first && id)
    setHeld(block(id), true);
}

void KvCache::unholdPage(uint32_t page) {
  pool_.releasePage(page, false);
  const uint64_t id = blockOnPage_[page];
  if (!spill_ || !id || pool_.pageActive(page))
    return;
  setHeld(block(id), false);
  erasePoisonedLeaf(id);
}

bool KvCache::held(uint64_t blockId) const { return block(blockId).held; }

void KvCache::setHeld(Block &entry, bool held) noexcept {
  if (entry.held == held)
    return;
  entry.held = held;
  if (entry.slot && !entry.foreignCopy)
    held ? heldCopyBytes_ += entry.slot->bytes() : heldCopyBytes_ -= entry.slot->bytes();
  reindex(entry);
}

void KvCache::setSlot(uint64_t blockId, std::shared_ptr<KvDiskSlot> slot) {
  Block &entry = block(blockId);
  if (!slot && entry.page == noPage)
    throw std::logic_error("a disk-only KV cache block is erased, not stripped");
  if (!slot && entry.form == kv::PageForm::Index)
    throw std::logic_error("a spilled KV cache block's copy holds its only K/V");
  giveDiskCopy(entry, std::move(slot));
  reindex(entry);
}

void KvCache::giveDiskCopy(Block &entry, std::shared_ptr<KvDiskSlot> slot) {
  setForeignCopy(entry, false);
  const bool changed = static_cast<bool>(entry.slot) != static_cast<bool>(slot);
  if (changed)
    slot ? ++diskBlocks_ : --diskBlocks_;
  // The block's copy takes what the new slot does, in place of the old, or
  // what parts it gained add.
  const uint64_t before = entry.copyBytes, after = slot ? slot->bytes() : 0;
  diskBytes_ = diskBytes_ - before + after;
  if (entry.held)
    heldCopyBytes_ = heldCopyBytes_ - before + after;
  entry.copyBytes = after;
  if (slot && slot != entry.slot)
    rewrites_ += std::exchange(entry.copied, true);
  entry.slot = std::move(slot);
  historyChanged(entry.id, Change::Copy);
  if (changed && copyChanged_)
    copyChanged_(entry.id, entry.depth, entry.page, static_cast<bool>(entry.slot));
}

void KvCache::setForeignCopy(Block &entry, bool foreign) noexcept {
  if (entry.foreignCopy == foreign)
    return;
  entry.foreignCopy = foreign;
  historyChanged(entry.id, Change::Copy);
  if (!entry.slot)
    return;
  // Only a copy of what the page holds is active KV.
  if (entry.held)
    foreign ? heldCopyBytes_ -= entry.slot->bytes() : heldCopyBytes_ += entry.slot->bytes();
  if (copyChanged_)
    copyChanged_(entry.id, entry.depth, entry.page, !foreign);
}

void KvCache::placeOnPage(Block &entry, uint32_t page, std::span<const uint32_t> window) {
  if (entry.page != noPage) {
    blockOnPage_[entry.page] = 0;
    for (const uint32_t each : entry.window)
      blockOnPage_[each] = 0;
    if (entry.form == kv::PageForm::Index)
      --indexBlocks_;
    setHeld(entry, false);
  }
  windowUnits_ -= entry.window.empty() ? 0 : 1;
  windowUnits_ += window.empty() ? 0 : 1;
  if (entry.page != page)
    historyChanged(entry.id);
  if (entry.window.empty() != window.empty())
    historyChanged(entry.id, Change::Unit);
  entry.page = page;
  entry.window.assign(window.begin(), window.end());
  entry.form = kv::PageForm::Full;
  if (page == noPage)
    return;
  blockOnPage_[page] = entry.id;
  for (const uint32_t each : window)
    blockOnPage_[each] = entry.id;
  entry.form = pool_.form(page);
  if (entry.form == kv::PageForm::Index)
    ++indexBlocks_;
  setHeld(entry, spill_ && pool_.pageActive(page));
}

void KvCache::upgrade(Block &entry, uint32_t page, std::span<const uint32_t> window) {
  if (blockOnPage_[page])
    throw std::logic_error("KV cache page already holds a block");
  const uint32_t previous = entry.page;
  const std::vector<uint32_t> previousWindow = entry.window;
  // It takes the writer's unit only in place of its own, which it holds
  // while a state's tail holds it; else the writer keeps its unit.
  const std::span<const uint32_t> unit = previousWindow.empty() ? std::span<const uint32_t>{} : window;
  own(page, unit);
  // Its copy holds what another command computed.
  if (entry.slot)
    setForeignCopy(entry, true);
  placeOnPage(entry, page, unit);
  reindex(entry);
  disown(previous, previousWindow);
  ++upgrades_;
}

void KvCache::followPages(const KvPageMoves &moves) noexcept {
  // Destinations lie in other extents and were free, so no block is on one.
  // References moved with the pages, and so do the holders' marks.
  for (const kv::PageCopy &move : moves.moved) {
    const uint64_t id = std::exchange(blockOnPage_[move.from], 0);
    if (!id)
      continue;
    const auto found = blocks_.find(id);
    if (found == blocks_.end())
      std::terminate();
    Block &entry = found->second;
    blockOnPage_[move.to] = id;
    // A window page moves on its own, the block's base page staying.
    if (const auto at = std::ranges::find(entry.window, move.from); at != entry.window.end()) {
      *at = move.to;
      continue;
    }
    entry.page = move.to;
    historyChanged(id);
    const kv::PageForm form = pool_.form(move.to);
    indexBlocks_ += form == kv::PageForm::Index;
    indexBlocks_ -= entry.form == kv::PageForm::Index;
    entry.form = form;
    reindex(entry);
  }
}

bool KvCache::abandonRestore(uint64_t blockId) {
  Block &entry = block(blockId);
  if (entry.transfer != Transfer::Filling || entry.residentChildren || entry.activeUsers ||
      entry.held)
    return false;
  setTransfer(blockId, Transfer::None);
  dropPage(blockId);
  return true;
}

void KvCache::dropPage(uint64_t blockId) {
  Block &entry = block(blockId);
  if (entry.page == noPage || !copyHoldsPages(entry))
    throw std::logic_error("KV cache block has no page to drop or no disk copy of what it holds");
  if (entry.residentChildren || entry.activeUsers || entry.held)
    throw std::logic_error("KV cache block is still in use");
  const uint32_t page = entry.page;
  const std::vector<uint32_t> window = entry.window;
  placeOnPage(entry, noPage);
  // Its copy holds its only K/V now, whoever computed it.
  setForeignCopy(entry, false);
  reindex(entry);
  if (entry.parent) {
    Block &parent = block(entry.parent);
    --parent.residentChildren;
    inherit(parent, entry.lastUsed);
  }
  disown(page, window);
}

// A parent that becomes a leaf when a child leaves its tier inherits the
// child's recency: it was used no later than the child and must not jump
// ahead of colder chains.
void KvCache::inherit(Block &parent, uint64_t lastUsed) noexcept {
  const bool leaf = parent.page != noPage ? !parent.residentChildren : !parent.children;
  if (leaf && !parent.activeUsers)
    parent.lastUsed = std::max(parent.lastUsed, lastUsed);
  reindex(parent);
}

void KvCache::adoptPage(uint64_t blockId, uint32_t page, std::span<const uint32_t> window) {
  Block &entry = block(blockId);
  if (entry.page != noPage)
    throw std::logic_error("KV cache block already has a page");
  if (page >= pool_.pageCount())
    throw std::out_of_range("KV cache physical page is out of range");
  if (entry.parent && block(entry.parent).page == noPage)
    throw std::logic_error("KV cache parent block has no page");
  if (blockOnPage_[page])
    throw std::logic_error("KV cache page already holds a block");
  own(page, window);
  placeOnPage(entry, page, window);
  reindex(entry);
  if (entry.parent) {
    Block &parent = block(entry.parent);
    ++parent.residentChildren;
    reindex(parent);
  }
}

void KvCache::poison(uint64_t blockId) {
  Block &entry = block(blockId);
  entry.poisoned = true;
  // It matches no lookup from now on. A request that holds it may still
  // read the copy, and a spilled block's copy holds its only K/V: either
  // copy leaves with the block.
  ++generation_;
  if (!entry.held && (entry.page == noPage || entry.form != kv::PageForm::Index))
    giveDiskCopy(entry, nullptr);
  reindex(entry);
  erasePoisonedLeaf(blockId);
}

std::optional<CacheEvictionCandidate> KvCache::oldestLeaf(uint32_t orders, uint64_t after) const {
  const std::optional<CacheEvictionCandidate> last =
      after ? std::optional(CacheEvictionCandidate{after, block(after).lastUsed}) : std::nullopt;
  std::optional<CacheEvictionCandidate> oldest;
  for (size_t order = 0; order < kLeafOrders; ++order) {
    if (!(orders & 1u << order))
      continue;
    const RecencyOrder &leaves = leaves_[order];
    const std::optional<CacheEvictionCandidate> candidate = last ? leaves.next(*last) : leaves.oldest();
    if (candidate &&
        (!oldest || std::pair{candidate->lastUsed, candidate->id} < std::pair{oldest->lastUsed, oldest->id}))
      oldest = candidate;
  }
  return oldest;
}

std::optional<CacheEvictionCandidate>
KvCache::evictionCandidate(uint64_t after) const {
  return oldestLeaf((1u << kLeafOrders) - 1, after);
}

std::optional<CacheEvictionCandidate> KvCache::leafCandidate(const LeafWalk &walk, uint64_t after) const {
  uint32_t orders = walk.dead ? 1u << leafOrder({.dead = true}) : 0;
  for (const bool inRam : {false, true})
    for (const bool copied : {false, true})
      if (!walk.dead && (!inRam || walk.withStateInRam) && (inRam || copied || !walk.copiedOnly))
        orders |= 1u << leafOrder({.inUse = walk.inUse, .stateInRam = inRam, .copied = copied});
  const std::optional<CacheEvictionCandidate> oldest = oldestLeaf(orders, after);
  leafVisits_ += oldest ? 1 : 0;
  return oldest;
}

std::optional<CacheEvictionCandidate>
KvCache::diskCandidate(bool duplicate, uint64_t after) const {
  return passing(duplicate ? duplicates_ : diskLeaves_, after, duplicate);
}

std::optional<CacheEvictionCandidate> KvCache::passing(const RecencyOrder &order, uint64_t after, bool copies) const {
  std::optional<CacheEvictionCandidate> next = order.oldest();
  if (after) {
    // An order of copies reads the later of a block's use and its chain's refresh (reindex).
    const Block &entry = block(after);
    next = order.next({after, copies ? std::max(entry.lastUsed, entry.copyUsed) : entry.lastUsed});
  }
  return next && next->id == passOver_ ? order.next(*next) : next;
}

std::vector<uint64_t> KvCache::subtree(uint64_t blockId) const {
  std::vector<uint64_t> order;
  std::vector<std::pair<uint64_t, bool>> pending{{blockId, false}};
  while (!pending.empty()) {
    const auto [id, expanded] = pending.back();
    pending.pop_back();
    if (expanded) {
      if (id != blockId)
        order.push_back(id);
      continue;
    }
    const Block &entry = block(id);
    if (id != blockId && (entry.transfer != Transfer::None || entry.activeUsers || entry.held))
      return {};
    pending.push_back({id, true});
    for (uint64_t child = entry.firstChild; child; child = block(child).nextSibling)
      pending.push_back({child, false});
  }
  return order;
}

void KvCache::erase(uint64_t blockId) {
  Block &candidate = block(blockId);
  if (candidate.children || candidate.activeUsers || candidate.held ||
      candidate.transfer != Transfer::None) {
    throw std::logic_error("cannot evict a referenced KV cache block");
  }
  const uint64_t parentId = candidate.parent;
  const uint64_t previous = candidate.previousSibling;
  const uint64_t next = candidate.nextSibling;
  const uint64_t hash = candidate.indexHash;
  const uint32_t page = candidate.page;
  const std::vector<uint32_t> window = candidate.window;
  const uint64_t lastUsed = candidate.lastUsed;
  const bool disk = candidate.slot != nullptr;
  const bool spilled = page != noPage && candidate.form == kv::PageForm::Index;
  const auto [first, last] = index_.equal_range(hash);
  auto indexed = std::find_if(
      first, last, [&](const auto &value) { return value.second == blockId; });
  if (indexed == last)
    throw std::logic_error("KV cache index is incomplete");
  RecencyOrder::unlink(candidate.ramNode);
  RecencyOrder::unlink(candidate.diskNode);
  RecencyOrder::unlink(candidate.unneededNode);
  RecencyOrder::unlink(candidate.unwrittenNode);
  RecencyOrder::unlink(candidate.windowNode);
  forget(candidate);
  if (disk)
    diskBytes_ -= candidate.copyBytes;
  index_.erase(indexed);
  blocks_.erase(blockId);
  ++generation_;
  // Its own answer goes, and its parent has a child less.
  historyChanged(blockId);
  if (parentId)
    historyChanged(parentId);
  if (page != noPage)
    blockOnPage_[page] = 0;
  for (const uint32_t each : window)
    blockOnPage_[each] = 0;
  if (disk)
    --diskBlocks_;
  if (spilled)
    --indexBlocks_;
  if (parentId) {
    Block &parent = block(parentId);
    if (!parent.children)
      throw std::logic_error("KV child count underflowed");
    --parent.children;
    (previous ? block(previous).nextSibling : parent.firstChild) = next;
    if (next)
      block(next).previousSibling = previous;
    if (page != noPage)
      --parent.residentChildren;
    inherit(parent, lastUsed);
    erasePoisonedLeaf(parentId);
  }
  if (page != noPage)
    disown(page, window);
}

KvCache::Block &KvCache::block(uint64_t blockId) {
  auto found = blocks_.find(blockId);
  if (found == blocks_.end())
    throw std::out_of_range("unknown KV cache block");
  return found->second;
}

const KvCache::Block &KvCache::block(uint64_t blockId) const {
  auto found = blocks_.find(blockId);
  if (found == blocks_.end())
    throw std::out_of_range("unknown KV cache block");
  return found->second;
}

// Resident leaves wait in the order of their kind, disk copies in another:
// redundant copies of blocks whose page holds their K/V, or
// the only copies, of a spilled or disk-only block without children. The
// copies no state needs wait in a third as well: the only one of a leaf
// without a state, and a redundant one no state, in memory or on disk,
// restores through; with no state at or below a block, no point's tail needs
// its window part either. In either tier a copy under a state in memory is
// needed: it saves its page's write when memory lets the page go, or when
// that state's conversation copies its next point. A block a request uses or
// holds, one in transfer, or a poisoned one is in no order: quota never
// takes a copy active KV reads, nor RAM reclaim a page a request holds.
// Active KV reads no copy of another command's K/V, which waits as a
// redundant copy beside a page requests use or hold.
KvCache::Orders KvCache::orders(const Block &entry) const noexcept {
  if (entry.transfer != Transfer::None || entry.poisoned)
    return {};
  const bool used = entry.activeUsers || entry.held;
  const bool paged = entry.page != noPage;
  const bool holdsKv = paged && entry.form != kv::PageForm::Index;
  Orders result;
  if (paged && !entry.residentChildren && !used)
    result.leaf = leafOrder(leafKind(entry));
  result.unwritten = unwrittenUnit(entry);
  result.unneededWindow = windowCopied(entry) && !entry.copyPins && !used &&
                          entry.slot->presence(kv::Component::Window) == KvDiskSlot::Presence::Landed;
  if (!entry.slot || (used && !entry.foreignCopy))
    return result;
  // A copy holds a block's only window K/V where a point on the SSD needs it
  // and its unit is not in memory: it is not redundant.
  result.duplicate = holdsKv && !(entry.copyPins && entry.window.empty() && windowCopied(entry));
  result.diskLeaf = !holdsKv && !entry.children;
  result.unneeded = holdsKv ? !entry.statesHere && !entry.statesBelow : !entry.statesHere && !entry.children;
  return result;
}

// A state at the leaf counts once StateCache counts its entry in
// (statesHere): an entry that no longer holds a valid state stays only while
// a lease pins it, which keeps the block in use, out of the leaves.
KvCache::LeafKind KvCache::leafKind(const Block &entry) const noexcept {
  if (!entry.statesHere && !entry.statesBelow)
    return {.dead = true};
  return {.inUse = (entry.state.keeps & keepBit(Keep::Used)) || entry.statesInUseBelow,
          .stateInRam = entry.stateInRam,
          .copied = copyHoldsPages(entry)};
}

void KvCache::placeLeaf(Block &entry) noexcept {
  RecencyOrder::unlink(entry.ramNode);
  if (const std::optional<size_t> leaf = orders(entry).leaf)
    leaves_[*leaf].link(entry.ramNode, entry.lastUsed, entry.id);
}

void KvCache::reindex(Block &entry) noexcept {
  placeLeaf(entry);
  RecencyOrder::unlink(entry.diskNode);
  RecencyOrder::unlink(entry.unneededNode);
  const Orders placed = orders(entry);
  const uint64_t copyUsed = std::max(entry.lastUsed, entry.copyUsed);
  if (placed.duplicate)
    duplicates_.link(entry.diskNode, copyUsed, entry.id);
  else if (placed.diskLeaf)
    diskLeaves_.link(entry.diskNode, entry.lastUsed, entry.id);
  if (placed.unneeded)
    unneeded_.link(entry.unneededNode, copyUsed, entry.id);
  RecencyOrder::unlink(entry.unwrittenNode);
  RecencyOrder::unlink(entry.windowNode);
  if (placed.unwritten)
    unwritten_.link(entry.unwrittenNode, entry.lastUsed, entry.id);
  if (placed.unneededWindow)
    unneededWindows_.link(entry.windowNode, entry.lastUsed, entry.id);
  rekeep(entry);
}

void KvCache::checkInvariants(const std::function<bool(uint64_t blockId)> &stateOnDisk,
                              const std::function<Marks(uint64_t blockId)> &marks,
                              const std::function<LeafKind(uint64_t blockId)> &walkKind) const {
  const auto check = [](bool holds, const char *what) {
    if (!holds)
      throw std::logic_error(std::string("KV cache invariant broken: ") + what);
  };
  uint32_t disk = 0, spilled = 0;
  uint64_t diskBytes = 0, heldCopyBytes = 0;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> children;
  for (const auto &[id, entry] : blocks_) {
    check(entry.id == id, "a block's id");
    disk += entry.slot != nullptr;
    diskBytes += entry.slot ? entry.slot->bytes() : 0;
    check(entry.copyBytes == (entry.slot ? entry.slot->bytes() : 0), "a block counts what its copy takes");
    heldCopyBytes += entry.held && entry.slot && !entry.foreignCopy ? entry.slot->bytes() : 0;
    // Reads of spilled K/V hold a copy only while a request holds its block
    // or a transfer moves it (Cache::cutPageTable), so a copy quota gives up
    // returns its slot at once. A command's cut names blocks its lanes held,
    // and outlives them only while the engine applies its results.
    check(!entry.slot || entry.held || entry.transfer != Transfer::None || entry.slot.use_count() == 1,
          "a copy no request holds and no transfer moves has no reader");
    if (entry.parent) {
      const auto parent = blocks_.find(entry.parent);
      check(parent != blocks_.end(), "a block's parent is in the graph");
      check(parent->second.depth + 1 == entry.depth, "a block's depth");
      // Page-holding blocks form a rooted subtree.
      check(entry.page == noPage || parent->second.page != noPage,
            "a block with a page has a parent with one");
      auto &[all, resident] = children[entry.parent];
      ++all;
      resident += entry.page != noPage;
    }
    check(!entry.foreignCopy || (entry.slot && entry.page != noPage && entry.form != kv::PageForm::Index),
          "a copy of another command's K/V sits beside a page that holds its K/V");
    if (entry.page == noPage) {
      check(entry.window.empty(), "a disk-only block holds no window unit");
      // A point on the SSD whose tail holds it reads its window part; no
      // point in memory sits on a chain that holds a disk-only block.
      check(!unitPages_ || !entry.windowPins, "no RAM copy's tail holds a disk-only block");
      check(!unitPages_ || !entry.copyPins || !entry.slot ||
                entry.slot->presence(kv::Component::Window) == KvDiskSlot::Presence::Landed,
            "a disk-only block a disk copy's tail holds has its window part");
      check(entry.slot != nullptr || entry.poisoned, "a disk-only block has a copy");
      check(!entry.held && entry.transfer == Transfer::None,
            "a disk-only block is neither held nor moving");
      continue;
    }
    // It holds its unit exactly while it needs it (needsUnit), or a transfer
    // moves the unit: a point's window K/V is in memory or on the SSD.
    check(!unitPages_ || (entry.window.empty() ? !needsUnit(entry) || entry.poisoned
                                               : needsUnit(entry) || entry.transfer != Transfer::None),
          "a resident block holds its window unit exactly while it needs it or a transfer moves it");
    check(blockOnPage_[entry.page] == id && (entry.window.empty() || entry.window.size() == unitPages_) &&
              std::ranges::all_of(entry.window, [&](uint32_t each) { return blockOnPage_[each] == id; }),
          "the block's pages name it");
    check(entry.form == pool_.form(entry.page), "a block keeps its page's form");
    check(entry.held == (spill_ && pool_.pageActive(entry.page)),
          "a block is held exactly while a request holds its page");
    if (entry.form == kv::PageForm::Index) {
      ++spilled;
      // A spilled block's copy holds its only K/V, landed unless a restore
      // fills its index.
      check(entry.slot != nullptr && entry.transfer != Transfer::Draining,
            "a spilled block has its landed copy");
    }
    check(entry.transfer != Transfer::Draining || entry.slot, "a page being written has its copy");
  }
  for (const auto &[id, entry] : blocks_) {
    const auto counted = children.find(id);
    const auto [all, resident] =
        counted == children.end() ? std::pair<uint32_t, uint32_t>{} : counted->second;
    check(entry.children == all && entry.residentChildren == resident, "a block counts its children");
  }
  for (uint32_t page = 0; page < blockOnPage_.size(); ++page) {
    const uint64_t id = blockOnPage_[page];
    check(!id || (blocks_.contains(id) &&
                  (blocks_.at(id).page == page || std::ranges::find(blocks_.at(id).window, page) !=
                                                       blocks_.at(id).window.end())),
          "a page names the block on it");
  }
  check(disk == diskBlocks_ && spilled == indexBlocks_ && diskBytes == diskBytes_ &&
            heldCopyBytes == heldCopyBytes_,
        "the counts of copies and spilled blocks, and what copies and held copies take of the quota");
  check(!persistent_ || static_cast<bool>(stateOnDisk),
        "a persistent cache's check names the states on disk");
  // With a window each block counts the tails of the states with a RAM copy
  // and of those with a disk copy that hold it.
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> pins;
  for (const auto &[id, entry] : blocks_) {
    if (!entry.statesHere || !windowPages_)
      continue;
    const bool onDisk = stateOnDisk && stateOnDisk(id);
    uint64_t above = id;
    for (uint32_t index = 0; above && index < windowPages_; ++index, above = blocks_.at(above).parent) {
      pins[above].first += entry.stateInRam ? 1 : 0;
      pins[above].second += onDisk ? 1 : 0;
    }
  }
  std::erase_if(pins, [](const auto &each) { return !each.second.first && !each.second.second; });
  for (const auto &[id, entry] : blocks_) {
    const auto counted = pins.find(id);
    const auto [memory, disk] = counted == pins.end() ? std::pair<uint32_t, uint32_t>{} : counted->second;
    check(entry.windowPins == memory && (!stateOnDisk || entry.copyPins == disk),
          "a block counts the RAM and disk copies' tails that hold it");
  }
  check(!stateOnDisk || pins.size() == pinnedBlocks_, "the count of blocks a state's tail holds");
  check(static_cast<size_t>(std::ranges::count_if(blocks_, [](const auto &each) { return !each.second.window.empty(); })) ==
            windowUnits_ && !deferring_ && unpinned_.empty(),
        "the count of blocks that hold a unit, and no publication's scope outlives it");
  // Each order holds the blocks reindex places in it, under their recency.
  const auto checkOrder = [&](const RecencyOrder &order, const auto &placed, const char *what, bool copies = false) {
    size_t members = 0;
    for (auto member = order.oldest(); member; member = order.next(*member), ++members) {
      const auto found = blocks_.find(member->id);
      check(found != blocks_.end() &&
                (copies ? std::max(found->second.lastUsed, found->second.copyUsed) : found->second.lastUsed) ==
                    member->lastUsed &&
                placed(orders(found->second)),
            what);
    }
    check(members == static_cast<size_t>(std::ranges::count_if(
                         blocks_, [&](const auto &each) { return placed(orders(each.second)); })),
          what);
  };
  // Each resident leaf waits in the order of its kind, the one a walk takes
  // it as, and no other block in any.
  size_t leaves = 0;
  for (size_t order = 0; order < kLeafOrders; ++order)
    for (auto member = leaves_[order].oldest(); member; member = leaves_[order].next(*member), ++leaves) {
      const auto found = blocks_.find(member->id);
      check(found != blocks_.end() && found->second.lastUsed == member->lastUsed &&
                orders(found->second).leaf == order,
            "the order of the resident leaves of a kind");
      check(!walkKind || walkKind(member->id) == leafKind(found->second),
            "a resident leaf's kind is the one a walk takes it as");
    }
  check(leaves == static_cast<size_t>(std::ranges::count_if(
                      blocks_, [&](const auto &each) { return orders(each.second).leaf.has_value(); })),
        "the orders of the resident leaves hold every resident leaf");
  checkOrder(duplicates_, [](const Orders &in) { return in.duplicate; }, "the order of redundant copies", true);
  checkOrder(diskLeaves_, [](const Orders &in) { return in.diskLeaf; }, "the order of only copies");
  checkOrder(unneeded_, [](const Orders &in) { return in.unneeded; }, "the order of copies no state needs", true);
  checkOrder(unwritten_, [](const Orders &in) { return in.unwritten; }, "the order of units held for unwritten parts");
  checkOrder(unneededWindows_, [](const Orders &in) { return in.unneededWindow; },
             "the order of window parts no disk copy needs");
  // Each block's reasons to stay (Keep), recounted children before parents
  // by the rule rekeep keeps them by (reasons): the children with each
  // reason and those below which a state needs its KV, the block's reasons
  // from those, and the resident blocks by form, reasons and idleness.
  std::vector<const Block *> deepestFirst;
  deepestFirst.reserve(blocks_.size());
  for (const auto &[_, entry] : blocks_)
    deepestFirst.push_back(&entry);
  std::ranges::sort(deepestFirst, [](const Block *left, const Block *right) { return left->depth > right->depth; });
  // Of each block, its children with each reason and those below which a
  // state needs its KV.
  struct Children final {
    std::array<uint32_t, kKeeps> keeps{};
    uint32_t needed = 0;
  };
  std::unordered_map<uint64_t, Children> withReason;
  std::array<uint32_t, kv::kPageForms << (kKeeps + 2)> counts{};
  std::array<uint32_t, size_t{1} << kKeeps> candidates{};
  std::array<uint32_t, size_t{2} << kKeeps> unwritten{};
  for (const Block *entry : deepestFirst) {
    check(!marks || marks(entry->id) == Marks{.state = entry->state,
                                              .stateInRam = entry->stateInRam,
                                              .wanted = entry->wanted,
                                              .resumePoint = entry->resumePoint},
          "a block's marks: what its state asks of its KV, the waiting chains that end at it, the resume point");
    const Children below = withReason[entry->id];
    check(entry->keepChildren == below.keeps && entry->neededChildren == below.needed,
          "a block counts its children with each reason");
    const auto [here, keeps, needed, counted, candidate] = reasons(*entry, below.keeps, below.needed);
    check(entry->keepsHere == here && entry->keeps == keeps && entry->needed == needed,
          "a block's reasons to stay, at it and below it, and whether a state there needs its KV");
    check(entry->counted == counted, "a block counts under its form, reasons and idleness");
    if (counted != noCount)
      ++counts[counted];
    check(entry->candidateCounted == candidate, "a block with a RAM copy a reclaim may take counts under its reasons");
    if (candidate != noCount)
      ++candidates[candidate];
    const uint16_t leaving = static_cast<uint16_t>(keeps | (entry->transfer == Transfer::None ? 1u << kKeeps : 0u));
    check(entry->unwrittenCounted == (unitLeaving(*entry) ? leaving : noCount),
          "a block whose unit waits for its window part's write counts under its reasons");
    if (unitLeaving(*entry))
      ++unwritten[leaving];
    if (entry->parent) {
      Children &parent = withReason[entry->parent];
      for (size_t each = 0; each < kKeeps; ++each)
        parent.keeps[each] += (keeps & keepBit(static_cast<Keep>(each))) ? 1 : 0;
      parent.needed += needed ? 1 : 0;
    }
  }
  check(counts == keepCounts_, "the counts of resident blocks by form, reasons and idleness");
  check(candidates == candidateCounts_, "the counts of the RAM copies a reclaim may take by their blocks' reasons");
  check(unwritten == unwrittenCounts_, "the counts of the units that wait for their window parts' writes");
}

void KvCache::erasePoisonedLeaf(uint64_t blockId) noexcept {
  const auto found = blocks_.find(blockId);
  if (found == blocks_.end() || !found->second.poisoned ||
      found->second.children || found->second.activeUsers || found->second.held ||
      found->second.transfer != Transfer::None)
    return;
  try {
    erase(blockId);
  } catch (...) {
    std::terminate();
  }
}

} // namespace splash::engine
