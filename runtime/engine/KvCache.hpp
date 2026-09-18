#pragma once

#include "engine/CacheRecency.hpp"
#include "engine/KvPool.hpp"
#include "engine/RecencyOrder.hpp"
#include "model/Model.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
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

// Content-addressed target-KV blocks across two tiers. Matching walks the
// chained full-page hashes from the root. A block holds a pool page, a disk
// slot, or both; resident blocks form a subtree at the root, so a matched
// chain is a resident prefix followed by disk-only blocks. Composite
// recurrent states are a separate sparse layer. This class owns exactly one
// prefix reference for every resident block.
class KvCache final {
public:
  static constexpr uint32_t pageTokens = kv::kPageTokens;
  static constexpr uint32_t noPage = std::numeric_limits<uint32_t>::max();

  struct BlockMatch {
    uint64_t id = 0;
    uint32_t physicalPage = noPage;
  };

  struct InsertResult : BlockMatch {
    bool inserted = false;
  };

  // Root first; noPage marks a disk-only block.
  struct Chain final {
    std::vector<uint64_t> blocks;
    std::vector<uint32_t> pages;
  };

  struct Snapshot {
    uint32_t blocks = 0;
    uint64_t bytes = 0;
    uint32_t diskBlocks = 0;
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
  // Existing content is returned as is; a disk-only block adopts the
  // writer's page, and a block in transfer keeps its own.
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

  // Tiers. A block in transfer is moving between them and is neither
  // evicted nor replaced until the transfer is cleared.
  [[nodiscard]] uint32_t page(uint64_t blockId) const;
  [[nodiscard]] std::shared_ptr<model::KvDiskSlot> slot(uint64_t blockId) const;
  [[nodiscard]] bool hasDiskChildren(uint64_t blockId) const;
  [[nodiscard]] uint32_t activeUsers(uint64_t blockId) const;
  // A reusable state was published at this block at some point; lookups that
  // find the block without one report a lost state.
  void noteState(uint64_t blockId);
  [[nodiscard]] bool hadState(uint64_t blockId) const;
  // Resident, without resident children or users: its page can go.
  [[nodiscard]] bool residentLeaf(uint64_t blockId) const;
  [[nodiscard]] bool transferring(uint64_t blockId) const;
  void setTransferring(uint64_t blockId, bool transferring);
  // Publishes the block's disk copy; a resident block may drop it with null.
  void setSlot(uint64_t blockId, std::shared_ptr<model::KvDiskSlot> slot);
  // Returns the page of a resident leaf that has a disk copy to the pool.
  void dropPage(uint64_t blockId);
  // Gives a disk-only block a page whose content follows, by restore or from
  // the request that recomputed it. The parent must be resident.
  void adoptPage(uint64_t blockId, uint32_t page);
  // A restore that failed: the block matches nothing any more and leaves
  // once its subtree and users are gone.
  void poison(uint64_t blockId);

  // Oldest resident leaf (no resident children) that no request uses. Pass
  // the previous candidate to continue the scan without a lookup.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  evictionCandidate(uint64_t after = 0) const;
  // Oldest unused holder of a disk copy to replace: with duplicate, a
  // resident block whose copy is redundant; otherwise a disk-only block
  // without children.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  diskCandidate(bool duplicate) const noexcept;
  // Only a block without children, users or transfer can be removed. The
  // caller handles any composite state attached to it first.
  void erase(uint64_t blockId);

  [[nodiscard]] Snapshot snapshot() const noexcept;

private:
  struct Block {
    uint64_t id = 0;
    uint64_t parent = 0;
    uint64_t indexHash = 0;
    std::array<uint32_t, pageTokens> tokens{};
    ImageIdentity images;
    uint32_t page = noPage;
    std::shared_ptr<model::KvDiskSlot> slot;
    uint32_t children = 0;
    uint32_t residentChildren = 0;
    uint32_t activeUsers = 0;
    uint32_t depth = 0;
    uint64_t lastUsed = 0;
    bool transferring = false;
    bool poisoned = false;
    bool hadState = false;
    RecencyOrder::Node ramNode;
    RecencyOrder::Node diskNode;
  };

  [[nodiscard]] uint64_t indexHash(uint64_t parentHash,
                                   std::span<const uint32_t> tokens,
                                   ImageIdentity images) const noexcept;
  [[nodiscard]] Block &block(uint64_t blockId);
  [[nodiscard]] const Block &block(uint64_t blockId) const;
  // Places the block in the orders its state calls for.
  void reindex(Block &entry) noexcept;
  void unlink(Block &entry) noexcept;
  void giveDiskCopy(Block &entry, std::shared_ptr<model::KvDiskSlot> slot) noexcept;
  void inherit(Block &parent, uint64_t lastUsed) noexcept;
  // A poisoned block leaves as soon as nothing refers to it.
  void erasePoisonedLeaf(uint64_t blockId) noexcept;

  KvPool &pool_;
  CacheNamespace cacheNamespace_;
  CacheRecency &recency_;
  std::unordered_map<uint64_t, Block> blocks_;
  std::unordered_multimap<uint64_t, uint64_t> index_;
  uint64_t nextBlockId_ = 1;
  RecencyOrder ramLeaves_;
  RecencyOrder duplicates_;
  RecencyOrder diskLeaves_;
  uint32_t residentBlocks_ = 0;
  uint32_t diskBlocks_ = 0;
};

} // namespace splash::engine
