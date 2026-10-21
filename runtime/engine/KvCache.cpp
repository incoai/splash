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
    // Two independently seeded chains fold the content digest, the grid, and
    // the block's alignment inside the span into 128 bits.
    uint64_t lo = identity.lo ? identity.lo : 0x243f6a8885a308d3ULL;
    uint64_t hi = identity.hi ? identity.hi : 0x13198a2e03707344ULL;
    for (uint64_t value :
         {span.digestLo, span.digestHi,
          (uint64_t{span.gridHeight} << 32) | span.gridWidth,
          (uint64_t{span.offset} << 32) | span.tokens, blockBegin}) {
      lo = mix(lo, value);
      hi = mix(hi, ~value);
    }
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
  for (const auto &[_, entry] : blocks_) {
    if (entry.page == noPage)
      continue;
    try {
      pool_.releasePage(entry.page, true);
    } catch (...) {
      std::terminate();
    }
  }
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
                                    BlockIdentity identity) {
  if (tokens.size() != pageTokens) {
    throw std::invalid_argument("KV cache block must contain one full page");
  }
  if (physicalPage >= pool_.pageCount()) {
    throw std::out_of_range("KV cache physical page is out of range");
  }
  if (auto existing = find(parentBlock, tokens, identity)) {
    BlockMatch result{existing->id, existing->physicalPage};
    if (existing->physicalPage == noPage) {
      adoptPage(existing->id, physicalPage);
      // The copy, which holds what another command computed, stays: a
      // restore point's chain on disk may need it. The block spills only
      // once a copy of the writer's page replaced it.
      Block &adopted = block(existing->id);
      setForeignCopy(adopted, true);
      reindex(adopted);
      result.physicalPage = physicalPage;
    } else if (block(existing->id).transfer == Transfer::Filling) {
      // Its page is still being filled; the writer keeps its own copy. A
      // page being written holds its final bytes, and the writer moves
      // onto it.
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

  pool_.retainPage(physicalPage, true);
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
  return entry.slot && !entry.foreignCopy;
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
    if (inUse)
      added ? ++entry.statesInUseBelow : --entry.statesInUseBelow;
  });
}

void KvCache::countStateInUse(uint64_t blockId, bool added) noexcept {
  countAbove(blockId, [&](Block &entry) {
    added ? ++entry.statesInUseBelow : --entry.statesInUseBelow;
  });
}

void KvCache::countDiskState(uint64_t blockId, bool added) noexcept {
  if (!persistent_)
    return;
  for (uint64_t id = blockId; id;) {
    const auto found = blocks_.find(id);
    if (found == blocks_.end())
      std::terminate();
    Block &entry = found->second;
    added ? ++entry.diskStates : --entry.diskStates;
    // Its redundant copy is needed again, or no longer.
    if (entry.diskStates == (added ? 1u : 0u))
      reindex(entry);
    id = entry.parent;
  }
}

bool KvCache::stateBelow(uint64_t blockId) const { return block(blockId).statesBelow > 0; }

bool KvCache::stateInUseBelow(uint64_t blockId) const {
  return block(blockId).statesInUseBelow > 0;
}

void KvCache::noteState(uint64_t blockId) { block(blockId).hadState = true; }

bool KvCache::hadState(uint64_t blockId) const { return block(blockId).hadState; }

uint32_t KvCache::idlePagesOnChains(std::span<const uint64_t> blocks, kv::PageForm form) const {
  std::unordered_set<uint64_t> visited;
  uint32_t pages = 0;
  for (uint64_t blockId : blocks) {
    while (blockId && visited.insert(blockId).second) {
      const Block &entry = block(blockId);
      if (entry.page != noPage) {
        // The request holding this page holds every page above it too.
        if (pool_.pageActive(entry.page))
          break;
        if (entry.transfer == Transfer::None && entry.form == form)
          ++pages;
      }
      // A disk-only block holds no page; the walk goes on above it.
      blockId = entry.parent;
    }
  }
  return pages;
}

bool KvCache::residentLeaf(uint64_t blockId) const {
  const Block &entry = block(blockId);
  return entry.page != noPage && !entry.residentChildren && !entry.activeUsers &&
         !entry.held && entry.transfer == Transfer::None && !entry.poisoned;
}

void KvCache::setTransfer(uint64_t blockId, Transfer transfer) {
  Block &entry = block(blockId);
  if ((entry.transfer == Transfer::None) == (transfer == Transfer::None))
    throw std::logic_error("KV cache block transfer state did not change");
  entry.transfer = transfer;
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
    held ? ++heldCopies_ : --heldCopies_;
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
  if (static_cast<bool>(entry.slot) != static_cast<bool>(slot)) {
    slot ? ++diskBlocks_ : --diskBlocks_;
    if (entry.held)
      slot ? ++heldCopies_ : --heldCopies_;
  }
  if (slot)
    rewrites_ += std::exchange(entry.copied, true);
  entry.slot = std::move(slot);
}

void KvCache::setForeignCopy(Block &entry, bool foreign) noexcept {
  if (entry.foreignCopy == foreign)
    return;
  // Only a copy of what the page holds is active KV.
  if (entry.held && entry.slot)
    foreign ? --heldCopies_ : ++heldCopies_;
  entry.foreignCopy = foreign;
}

void KvCache::placeOnPage(Block &entry, uint32_t page) {
  if (entry.page != noPage) {
    blockOnPage_[entry.page] = 0;
    if (entry.form == kv::PageForm::Index)
      --indexBlocks_;
    setHeld(entry, false);
  }
  entry.page = page;
  entry.form = kv::PageForm::Full;
  if (page == noPage)
    return;
  blockOnPage_[page] = entry.id;
  entry.form = pool_.form(page);
  if (entry.form == kv::PageForm::Index)
    ++indexBlocks_;
  setHeld(entry, spill_ && pool_.pageActive(page));
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
    entry.page = move.to;
    blockOnPage_[move.to] = id;
    const kv::PageForm form = pool_.form(move.to);
    if (form == entry.form)
      continue;
    form == kv::PageForm::Index ? ++indexBlocks_ : --indexBlocks_;
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
  if (entry.page == noPage || !entry.slot)
    throw std::logic_error("KV cache block has no page to drop or no disk copy");
  if (entry.residentChildren || entry.activeUsers || entry.held)
    throw std::logic_error("KV cache block is still in use");
  const uint32_t page = entry.page;
  placeOnPage(entry, noPage);
  // Its copy holds its only K/V now, whoever computed it.
  setForeignCopy(entry, false);
  reindex(entry);
  if (entry.parent) {
    Block &parent = block(entry.parent);
    --parent.residentChildren;
    inherit(parent, entry.lastUsed);
  }
  pool_.releasePage(page, true);
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

void KvCache::adoptPage(uint64_t blockId, uint32_t page) {
  Block &entry = block(blockId);
  if (entry.page != noPage)
    throw std::logic_error("KV cache block already has a page");
  if (page >= pool_.pageCount())
    throw std::out_of_range("KV cache physical page is out of range");
  if (entry.parent && block(entry.parent).page == noPage)
    throw std::logic_error("KV cache parent block has no page");
  if (blockOnPage_[page])
    throw std::logic_error("KV cache page already holds a block");
  pool_.retainPage(page, true);
  placeOnPage(entry, page);
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
  if (!entry.held && (entry.page == noPage || entry.form == kv::PageForm::Full))
    giveDiskCopy(entry, nullptr);
  reindex(entry);
  erasePoisonedLeaf(blockId);
}

std::optional<CacheEvictionCandidate>
KvCache::evictionCandidate(uint64_t after) const {
  if (!after)
    return ramLeaves_.oldest();
  return ramLeaves_.next({after, block(after).lastUsed});
}

std::optional<CacheEvictionCandidate>
KvCache::diskCandidate(bool duplicate, uint64_t after) const {
  const RecencyOrder &order = duplicate ? duplicates_ : diskLeaves_;
  if (!after)
    return order.oldest();
  return order.next({after, block(after).lastUsed});
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
  index_.erase(indexed);
  blocks_.erase(blockId);
  ++generation_;
  if (page != noPage)
    blockOnPage_[page] = 0;
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
    pool_.releasePage(page, true);
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

// Resident leaves wait in one order, disk copies in another: redundant
// copies of blocks whose page holds their K/V, or the only copies, of a
// spilled or disk-only block without children. In a persistent cache the
// disk copies no state needs wait in a third as well: a redundant one that
// no state on disk restores through, and the only one of a leaf without a
// state. A block a request uses or holds, one in transfer, or a poisoned one
// is in no order: quota never takes a copy active KV reads, nor RAM reclaim
// a page a request holds. Active KV reads no copy of another command's K/V,
// which waits as a redundant copy beside a page requests use or hold.
KvCache::Orders KvCache::orders(const Block &entry) const noexcept {
  if (entry.transfer != Transfer::None || entry.poisoned)
    return {};
  const bool used = entry.activeUsers || entry.held;
  const bool paged = entry.page != noPage;
  const bool full = paged && entry.form == kv::PageForm::Full;
  Orders result;
  result.ramLeaf = paged && !entry.residentChildren && !used;
  if (!entry.slot || (used && !entry.foreignCopy))
    return result;
  result.duplicate = full;
  result.diskLeaf = !full && !entry.children;
  result.unneeded = persistent_ && (full ? !entry.diskStates : !entry.statesHere && !entry.children);
  return result;
}

void KvCache::reindex(Block &entry) noexcept {
  RecencyOrder::unlink(entry.ramNode);
  RecencyOrder::unlink(entry.diskNode);
  RecencyOrder::unlink(entry.unneededNode);
  const Orders placed = orders(entry);
  if (placed.ramLeaf)
    ramLeaves_.link(entry.ramNode, entry.lastUsed, entry.id);
  if (placed.duplicate)
    duplicates_.link(entry.diskNode, entry.lastUsed, entry.id);
  else if (placed.diskLeaf)
    diskLeaves_.link(entry.diskNode, entry.lastUsed, entry.id);
  if (placed.unneeded)
    unneeded_.link(entry.unneededNode, entry.lastUsed, entry.id);
}

void KvCache::checkInvariants(const std::function<bool(uint64_t blockId)> &stateOnDisk) const {
  const auto check = [](bool holds, const char *what) {
    if (!holds)
      throw std::logic_error(std::string("KV cache invariant broken: ") + what);
  };
  uint32_t disk = 0, spilled = 0, heldCopies = 0;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> children;
  for (const auto &[id, entry] : blocks_) {
    check(entry.id == id, "a block's id");
    disk += entry.slot != nullptr;
    heldCopies += entry.held && entry.slot && !entry.foreignCopy;
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
    check(!entry.foreignCopy || (entry.slot && entry.page != noPage && entry.form == kv::PageForm::Full),
          "a copy of another command's K/V sits beside a Full page");
    if (entry.page == noPage) {
      check(entry.slot != nullptr || entry.poisoned, "a disk-only block has a copy");
      check(!entry.held && entry.transfer == Transfer::None,
            "a disk-only block is neither held nor moving");
      continue;
    }
    check(blockOnPage_[entry.page] == id, "the page names its block");
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
    check(!id || (blocks_.contains(id) && blocks_.at(id).page == page), "a page names the block on it");
  }
  check(disk == diskBlocks_ && spilled == indexBlocks_ && heldCopies == heldCopies_,
        "the counts of copies, spilled blocks and held copies");
  // In a persistent cache each block counts the states on disk at it and
  // below it.
  std::unordered_map<uint64_t, uint32_t> diskStates;
  check(!persistent_ || static_cast<bool>(stateOnDisk),
        "a persistent cache's check names the states on disk");
  if (persistent_)
    for (const auto &[id, _] : blocks_)
      if (stateOnDisk(id))
        for (uint64_t above = id; above; above = blocks_.at(above).parent)
          ++diskStates[above];
  for (const auto &[id, entry] : blocks_) {
    const auto counted = diskStates.find(id);
    check(entry.diskStates == (counted == diskStates.end() ? 0 : counted->second),
          "a block counts the states on disk at it and below it");
  }
  // Each order holds the blocks reindex places in it, under their recency.
  const auto checkOrder = [&](const RecencyOrder &order, const auto &placed, const char *what) {
    size_t members = 0;
    for (auto member = order.oldest(); member; member = order.next(*member), ++members) {
      const auto found = blocks_.find(member->id);
      check(found != blocks_.end() && found->second.lastUsed == member->lastUsed &&
                placed(orders(found->second)),
            what);
    }
    check(members == static_cast<size_t>(std::ranges::count_if(
                         blocks_, [&](const auto &each) { return placed(orders(each.second)); })),
          what);
  };
  checkOrder(ramLeaves_, [](const Orders &in) { return in.ramLeaf; }, "the order of resident leaves");
  checkOrder(duplicates_, [](const Orders &in) { return in.duplicate; }, "the order of redundant copies");
  checkOrder(diskLeaves_, [](const Orders &in) { return in.diskLeaf; }, "the order of only copies");
  checkOrder(unneeded_, [](const Orders &in) { return in.unneeded; }, "the order of copies no state needs");
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
