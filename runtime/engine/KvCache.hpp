#pragma once

#include "engine/CacheRecency.hpp"
#include "engine/KvPool.hpp"
#include "model/Model.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splash::engine {

// One process owns one loaded target+draft model, executable build and Q8
// layout. Their canonical SHA-256 is stored once on the cache instance; KV
// blocks never duplicate strings or layout metadata.
struct CacheNamespace final {
  std::array<uint8_t, 32> digest{};

  bool operator==(const CacheNamespace &) const = default;
};

struct ImageIdentity final {
  uint64_t lo = 0;
  uint64_t hi = 0;

  bool operator==(const ImageIdentity &) const = default;
};

[[nodiscard]] ImageIdentity
blockImageIdentity(uint64_t blockBegin, uint32_t blockTokens,
                   std::span<const ImageSpan> spans) noexcept;

struct KvBlockKeyView final {
  uint64_t parentBlock = 0;
  uint64_t indexHash = 0;
  std::span<const uint32_t> tokens;
  ImageIdentity images;
};

// Hashes filter candidates; equality still requires the complete key.
[[nodiscard]] bool exactKvBlockKeyMatch(const KvBlockKeyView &stored,
                                        const KvBlockKeyView &query) noexcept;

// Content-addressed target-KV blocks. Matching walks the chained full-page
// hashes from the root; composite recurrent states are a separate sparse layer.
// This class owns exactly one prefix reference for every resident block.
class KvCache final {
public:
  static constexpr uint32_t pageTokens = kv::kPageTokens;

  struct BlockMatch {
    uint64_t id = 0;
    uint32_t physicalPage = 0;
  };

  struct InsertResult : BlockMatch {
    bool inserted = false;
  };

  struct Chain final {
    std::vector<uint64_t> blocks;
    std::vector<uint32_t> pages;
  };

  struct Snapshot {
    uint32_t blocks = 0;
    uint64_t bytes = 0;
  };

  KvCache(KvPool &pool, CacheNamespace cacheNamespace, CacheRecency &recency)
      : pool_(pool), cacheNamespace_(cacheNamespace), recency_(recency) {}
  KvCache(const KvCache &) = delete;
  KvCache &operator=(const KvCache &) = delete;
  ~KvCache() noexcept;

  // Keys are the parent block, the exact tokens, and the identity of any
  // image content the rows depend on (zero for text-only blocks).
  [[nodiscard]] std::optional<BlockMatch> find(uint64_t parentBlock,
                                               std::span<const uint32_t> tokens,
                                               ImageIdentity images = {}) const;
  [[nodiscard]] InsertResult insert(uint64_t parentBlock,
                                    std::span<const uint32_t> tokens,
                                    uint32_t physicalPage,
                                    ImageIdentity images = {});

  void retainActive(uint64_t blockId);
  void releaseActive(uint64_t blockId) noexcept;
  void touch(uint64_t blockId) noexcept;

  [[nodiscard]] Chain chain(uint64_t blockId) const;
  [[nodiscard]] bool contains(uint64_t blockId) const noexcept;
  [[nodiscard]] uint32_t chainLength(uint64_t blockId) const;
  [[nodiscard]] uint64_t generation() const noexcept { return generation_; }

  // Only a leaf unused by active requests can be removed. The caller handles
  // any composite state attached to the returned block before erase().
  // Ordered leaf candidates. Pass the previously returned id to continue
  // without allocating or scanning the block table.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  evictionCandidate(uint64_t after = 0) const;
  void erase(uint64_t blockId);

  [[nodiscard]] Snapshot snapshot() const noexcept;

private:
  using EvictionOrder = std::set<std::pair<uint64_t, uint64_t>>;

  struct Block {
    uint64_t id = 0;
    uint64_t parent = 0;
    uint64_t indexHash = 0;
    std::array<uint32_t, pageTokens> tokens{};
    ImageIdentity images;
    uint32_t physicalPage = 0;
    uint32_t children = 0;
    uint32_t activeUsers = 0;
    uint32_t depth = 0;
    uint64_t lastUsed = 0;
    // Allocate the index node once at publication. Pinned/non-leaf blocks
    // retain it here, so unpinning and eviction never allocate under pressure.
    EvictionOrder::node_type evictionNode;
    bool evictable = false;
  };

  [[nodiscard]] uint64_t indexHash(uint64_t parentBlock,
                                   std::span<const uint32_t> tokens,
                                   ImageIdentity images) const noexcept;
  [[nodiscard]] Block &block(uint64_t blockId);
  [[nodiscard]] const Block &block(uint64_t blockId) const;
  void touchEvictable(uint64_t blockId) noexcept;
  void insertEvictable(uint64_t blockId, uint64_t lastUsed) noexcept;
  void removeEvictable(uint64_t blockId) noexcept;

  KvPool &pool_;
  CacheNamespace cacheNamespace_;
  CacheRecency &recency_;
  std::unordered_map<uint64_t, Block> blocks_;
  std::unordered_multimap<uint64_t, uint64_t> index_;
  uint64_t nextBlockId_ = 1;
  // Any graph change invalidates previews, including a newly matched block.
  uint64_t generation_ = 1;
  EvictionOrder evictionOrder_;
};

} // namespace splash::engine
