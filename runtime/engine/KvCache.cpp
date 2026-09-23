#include "engine/KvCache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {
namespace {

uint64_t mix(uint64_t hash, uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  return hash;
}

} // namespace

ImageIdentity blockImageIdentity(uint64_t blockBegin, uint32_t blockTokens,
                                 std::span<const ImageSpan> spans) noexcept {
  ImageIdentity identity;
  const uint64_t blockEnd = blockBegin + blockTokens;
  for (const ImageSpan &span : spans) {
    if (span.end() <= blockBegin || span.offset >= blockEnd)
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

bool exactKvBlockKeyMatch(const KvBlockKeyView &stored,
                          const KvBlockKeyView &query) noexcept {
  return stored.indexHash == query.indexHash &&
         stored.parentBlock == query.parentBlock &&
         stored.images == query.images &&
         stored.tokens.size() == query.tokens.size() &&
         std::equal(stored.tokens.begin(), stored.tokens.end(),
                    query.tokens.begin());
}

KvCache::~KvCache() noexcept {
  for (const auto &[_, entry] : blocks_) {
    try {
      pool_.releasePage(entry.physicalPage, true);
    } catch (...) {
      std::terminate();
    }
  }
}

uint64_t KvCache::indexHash(uint64_t parentBlock,
                            std::span<const uint32_t> tokens,
                            ImageIdentity images) const noexcept {
  uint64_t hash = mix(0x6a09e667f3bcc909ULL, parentBlock);
  for (uint8_t byte : cacheNamespace_.digest)
    hash = mix(hash, byte);
  for (uint32_t token : tokens)
    hash = mix(hash, token);
  hash = mix(hash, images.lo);
  hash = mix(hash, images.hi);
  return hash;
}

std::optional<KvCache::BlockMatch>
KvCache::find(uint64_t parentBlock, std::span<const uint32_t> tokens,
              ImageIdentity images) const {
  if (tokens.size() != pageTokens) {
    return std::nullopt;
  }
  if (parentBlock && !blocks_.contains(parentBlock))
    return std::nullopt;
  const uint64_t hash = indexHash(parentBlock, tokens, images);
  const KvBlockKeyView query{parentBlock, hash, tokens, images};
  const auto [first, last] = index_.equal_range(hash);
  for (auto candidate = first; candidate != last; ++candidate) {
    const Block &entry = block(candidate->second);
    const KvBlockKeyView stored{entry.parent, entry.indexHash, entry.tokens,
                                entry.images};
    if (exactKvBlockKeyMatch(stored, query)) {
      return BlockMatch{entry.id, entry.physicalPage};
    }
  }
  return std::nullopt;
}

KvCache::InsertResult KvCache::insert(uint64_t parentBlock,
                                      std::span<const uint32_t> tokens,
                                      uint32_t physicalPage,
                                      ImageIdentity images) {
  if (tokens.size() != pageTokens) {
    throw std::invalid_argument("KV cache block must contain one full page");
  }
  if (physicalPage >= pool_.pageCount()) {
    throw std::out_of_range("KV cache physical page is out of range");
  }
  if (auto existing = find(parentBlock, tokens, images)) {
    InsertResult result;
    result.id = existing->id;
    result.physicalPage = existing->physicalPage;
    return result;
  }
  if (parentBlock && !blocks_.contains(parentBlock)) {
    throw std::invalid_argument("KV cache parent block is not resident");
  }
  if (parentBlock &&
      block(parentBlock).children == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("KV cache child count overflowed");
  }
  if (!nextBlockId_ || nextBlockId_ == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("KV cache block ids exhausted");
  }

  Block entry;
  entry.id = nextBlockId_;
  entry.parent = parentBlock;
  entry.indexHash = indexHash(parentBlock, tokens, images);
  std::copy(tokens.begin(), tokens.end(), entry.tokens.begin());
  entry.images = images;
  entry.physicalPage = physicalPage;
  entry.depth = parentBlock ? block(parentBlock).depth + 1 : 1;
  entry.evictionNode = evictionOrder_.extract(
      evictionOrder_.emplace(0, entry.id).first);

  pool_.retainPage(physicalPage, true);
  bool blockInserted = false;
  try {
    auto [position, unique] = blocks_.emplace(entry.id, std::move(entry));
    if (!unique)
      throw std::logic_error("duplicate KV cache block id");
    blockInserted = true;
    index_.emplace(position->second.indexHash, position->first);
  } catch (...) {
    if (blockInserted)
      blocks_.erase(nextBlockId_);
    pool_.releasePage(physicalPage, true);
    throw;
  }
  if (parentBlock) {
    Block &parent = block(parentBlock);
    if (!parent.children && !parent.activeUsers)
      removeEvictable(parentBlock);
    ++parent.children;
  }
  const uint64_t id = nextBlockId_++;
  insertEvictable(id, recency_.next());
  ++generation_;
  InsertResult result;
  result.id = id;
  result.physicalPage = physicalPage;
  result.inserted = true;
  return result;
}

void KvCache::retainActive(uint64_t blockId) {
  Block &entry = block(blockId);
  if (entry.activeUsers == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("KV cache active user count overflowed");
  }
  if (!entry.activeUsers && !entry.children)
    removeEvictable(blockId);
  ++entry.activeUsers;
  entry.lastUsed = recency_.next();
}

void KvCache::releaseActive(uint64_t blockId) noexcept {
  auto found = blocks_.find(blockId);
  if (found == blocks_.end() || !found->second.activeUsers)
    std::terminate();
  --found->second.activeUsers;
  if (!found->second.activeUsers && !found->second.children)
    insertEvictable(blockId, recency_.next());
}

void KvCache::touch(uint64_t blockId) noexcept {
  auto found = blocks_.find(blockId);
  if (found == blocks_.end())
    std::terminate();
  if (found->second.evictable) {
    touchEvictable(blockId);
  } else {
    found->second.lastUsed = recency_.next();
  }
}

KvCache::Chain KvCache::chain(uint64_t blockId) const {
  const uint32_t depth = chainLength(blockId);
  Chain result;
  result.blocks.resize(depth);
  result.pages.resize(depth);
  for (uint32_t index = depth; index > 0; --index) {
    const Block &entry = block(blockId);
    result.blocks[index - 1] = entry.id;
    result.pages[index - 1] = entry.physicalPage;
    blockId = entry.parent;
  }
  if (blockId)
    throw std::logic_error("KV cache chain exceeds its depth");
  return result;
}

bool KvCache::contains(uint64_t blockId) const noexcept {
  return blocks_.contains(blockId);
}

uint32_t KvCache::chainLength(uint64_t blockId) const {
  return block(blockId).depth;
}

std::optional<CacheEvictionCandidate>
KvCache::evictionCandidate(uint64_t after) const {
  auto candidate = after
                       ? evictionOrder_.upper_bound({block(after).lastUsed, after})
                       : evictionOrder_.begin();
  if (candidate == evictionOrder_.end())
    return std::nullopt;
  return CacheEvictionCandidate{candidate->second, candidate->first};
}

void KvCache::erase(uint64_t blockId) {
  const Block &candidate = block(blockId);
  if (candidate.children || candidate.activeUsers) {
    throw std::logic_error("cannot evict a referenced KV cache block");
  }
  const uint64_t parentId = candidate.parent;
  const uint64_t hash = candidate.indexHash;
  const uint32_t page = candidate.physicalPage;
  const uint64_t lastUsed = candidate.lastUsed;
  const auto [first, last] = index_.equal_range(hash);
  auto indexed = std::find_if(
      first, last, [&](const auto &value) { return value.second == blockId; });
  if (indexed == last)
    throw std::logic_error("KV cache index is incomplete");
  removeEvictable(blockId);
  index_.erase(indexed);
  blocks_.erase(blockId);
  ++generation_;
  if (parentId) {
    Block &parent = block(parentId);
    if (!parent.children)
      throw std::logic_error("KV child count underflowed");
    --parent.children;
    if (!parent.children && !parent.activeUsers)
      insertEvictable(parentId, std::max(parent.lastUsed, lastUsed));
  }
  pool_.releasePage(page, true);
}

KvCache::Snapshot KvCache::snapshot() const noexcept {
  const uint64_t count = blocks_.size();
  const uint64_t bytes =
      count > std::numeric_limits<uint64_t>::max() / pool_.bytesPerPage()
          ? std::numeric_limits<uint64_t>::max()
          : count * pool_.bytesPerPage();
  return {static_cast<uint32_t>(
              std::min<uint64_t>(count, std::numeric_limits<uint32_t>::max())),
          bytes};
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

void KvCache::touchEvictable(uint64_t blockId) noexcept {
  removeEvictable(blockId);
  insertEvictable(blockId, recency_.next());
}

// Evictable leaves stay in last-used order. A block that was just inserted,
// released, or touched is the newest. A parent that becomes a leaf when its
// child is erased inherits that child's recency instead: it was used no later
// than the child and must not jump ahead of colder chains.
void KvCache::insertEvictable(uint64_t blockId, uint64_t lastUsed) noexcept {
  Block &entry = blocks_.at(blockId);
  if (entry.evictable || entry.children || entry.activeUsers ||
      entry.evictionNode.empty())
    std::terminate();
  entry.lastUsed = lastUsed;
  entry.evictionNode.value() = {lastUsed, blockId};
  if (!evictionOrder_.insert(std::move(entry.evictionNode)).inserted)
    std::terminate();
  entry.evictable = true;
}

void KvCache::removeEvictable(uint64_t blockId) noexcept {
  Block &entry = blocks_.at(blockId);
  if (!entry.evictable)
    std::terminate();
  entry.evictionNode = evictionOrder_.extract({entry.lastUsed, blockId});
  if (entry.evictionNode.empty())
    std::terminate();
  entry.evictable = false;
}

} // namespace splash::engine
