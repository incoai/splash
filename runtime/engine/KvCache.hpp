#pragma once

#include "engine/CacheRecency.hpp"
#include "engine/KvPool.hpp"
#include "engine/KvTier.hpp"
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

struct ImageIdentity final {
  uint64_t lo = 0;
  uint64_t hi = 0;

  bool operator==(const ImageIdentity &) const = default;
};

// The spans are sorted by offset and disjoint, as Engine::submit requires.
[[nodiscard]] ImageIdentity
blockImageIdentity(uint64_t blockBegin, uint32_t blockTokens,
                   std::span<const ImageSpan> spans) noexcept;

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

  // Root first; noPage marks a disk-only block.
  struct Chain final {
    std::vector<uint64_t> blocks;
    std::vector<uint32_t> pages;
  };

  KvCache(KvPool &pool, CacheRecency &recency)
      : pool_(pool), recency_(recency), blockOnPage_(pool.pageCount()) {}
  KvCache(const KvCache &) = delete;
  KvCache &operator=(const KvCache &) = delete;
  ~KvCache() noexcept;

  // Keys are the parent block, the exact tokens, and the identity of any
  // image content the rows depend on (zero for text-only blocks).
  [[nodiscard]] std::optional<BlockMatch> find(uint64_t parentBlock,
                                               std::span<const uint32_t> tokens,
                                               ImageIdentity images) const;
  // Existing content is returned as is; a disk-only block adopts the
  // writer's page, and a block in transfer keeps its own.
  [[nodiscard]] BlockMatch insert(uint64_t parentBlock,
                                  std::span<const uint32_t> tokens,
                                  uint32_t physicalPage,
                                  ImageIdentity images);

  void retainActive(uint64_t blockId);
  void releaseActive(uint64_t blockId) noexcept;
  void touch(uint64_t blockId) noexcept;

  [[nodiscard]] Chain chain(uint64_t blockId) const;
  [[nodiscard]] bool contains(uint64_t blockId) const noexcept;
  // The block is in the graph and not poisoned: find() can return it.
  [[nodiscard]] bool matchable(uint64_t blockId) const noexcept;
  [[nodiscard]] uint32_t chainLength(uint64_t blockId) const;
  [[nodiscard]] uint64_t generation() const noexcept { return generation_; }

  // Tiers. A block in transfer is moving between them and is neither
  // evicted nor replaced until the transfer is cleared.
  [[nodiscard]] uint32_t page(uint64_t blockId) const;
  [[nodiscard]] std::shared_ptr<KvDiskSlot> slot(uint64_t blockId) const;
  [[nodiscard]] bool hasDiskChildren(uint64_t blockId) const;
  // StateCache counts each of its entries in and out, and each entry while
  // an unfinished request uses it: a state restores through the KV of every
  // block above its own.
  void countState(uint64_t blockId, bool added, bool inUse) noexcept;
  void countStateInUse(uint64_t blockId, bool added) noexcept;
  // A state sits below the block. Without one, the disk-only blocks below
  // it are never read again.
  [[nodiscard]] bool stateBelow(uint64_t blockId) const;
  // A state an unfinished request uses sits below the block.
  [[nodiscard]] bool stateInUseBelow(uint64_t blockId) const;
  // StateCache notes each ordinary publication or reuse of a state at this
  // block; lookups that find the block without one report a lost state.
  void noteState(uint64_t blockId);
  [[nodiscard]] bool hadState(uint64_t blockId) const;
  // The resident pages, on the chains of these blocks (each block and every
  // block above it), that no request holds and no transfer moves; each
  // counts once. A request holds every page of its chain, so a chain's
  // count stops at the first page one holds.
  [[nodiscard]] uint32_t idlePagesOnChains(std::span<const uint64_t> blocks) const;
  // Resident, without resident children or users: its page can go.
  [[nodiscard]] bool residentLeaf(uint64_t blockId) const;
  void setTransferring(uint64_t blockId, bool transferring);
  // Publishes the block's disk copy; a resident block may drop it with null.
  void setSlot(uint64_t blockId, std::shared_ptr<KvDiskSlot> slot);
  // Returns the page of a resident leaf that has a disk copy to the pool.
  void dropPage(uint64_t blockId);
  // Gives a disk-only block a page whose content follows, by restore or from
  // the request that recomputed it. The parent must be resident.
  void adoptPage(uint64_t blockId, uint32_t page);
  // The pool moved pages: the block on each of them, if any, names the page
  // it moved to. It visits only the moved pages.
  void followPages(const KvPageMoves &moves) noexcept;
  // Abandons an unsubmitted restore at an unused leaf, keeping its disk copy.
  [[nodiscard]] bool abandonRestore(uint64_t blockId);
  // A restore that failed: the block matches nothing any more and leaves
  // once its subtree and users are gone.
  void poison(uint64_t blockId);

  // Oldest resident leaf (no resident children) that no request uses. Pass
  // the previous candidate to continue the scan without a lookup.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  evictionCandidate(uint64_t after) const;
  // Oldest unused holder of a disk copy to replace: with duplicate, a
  // resident block whose copy is redundant; otherwise a disk-only block
  // without children. Pass the previous candidate to continue the scan.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  diskCandidate(bool duplicate, uint64_t after = 0) const;
  // The blocks below a block, each after its children, visiting only that
  // subtree; empty when one of them is in transfer or active. Below a
  // resident leaf they are disk-only; below a poisoned block some may be
  // resident.
  [[nodiscard]] std::vector<uint64_t> subtree(uint64_t blockId) const;
  // Only a block without children, users or transfer can be removed. The
  // caller handles any composite state attached to it first.
  void erase(uint64_t blockId);

  // Blocks with a disk copy, resident or not; the pool counts the resident
  // blocks (KvPoolSnapshot::pagesPrefix).
  [[nodiscard]] uint32_t diskBlocks() const noexcept { return diskBlocks_; }

private:
  struct Block {
    uint64_t id = 0;
    uint64_t parent = 0;
    // Blocks come and go only as leaves, so insert and erase keep these
    // links in O(1) and subtree() never scans the whole cache.
    uint64_t firstChild = 0;
    uint64_t previousSibling = 0;
    uint64_t nextSibling = 0;
    uint64_t indexHash = 0;
    std::array<uint32_t, pageTokens> tokens{};
    ImageIdentity images;
    uint32_t page = noPage;
    std::shared_ptr<KvDiskSlot> slot;
    uint32_t children = 0;
    uint32_t residentChildren = 0;
    uint32_t statesBelow = 0;
    uint32_t statesInUseBelow = 0;
    uint32_t activeUsers = 0;
    uint32_t depth = 0;
    uint64_t lastUsed = 0;
    bool transferring = false;
    bool poisoned = false;
    bool hadState = false;
    RecencyOrder::Node ramNode;
    RecencyOrder::Node diskNode;
  };

  [[nodiscard]] Block &block(uint64_t blockId);
  [[nodiscard]] const Block &block(uint64_t blockId) const;
  // Places the block in the orders its state calls for.
  void reindex(Block &entry) noexcept;
  void giveDiskCopy(Block &entry, std::shared_ptr<KvDiskSlot> slot) noexcept;
  void inherit(Block &parent, uint64_t lastUsed) noexcept;
  // Applies count to every block above this one.
  template <typename Count>
  void countAbove(uint64_t blockId, const Count &count) noexcept;
  // A poisoned block leaves as soon as nothing refers to it.
  void erasePoisonedLeaf(uint64_t blockId) noexcept;

  KvPool &pool_;
  CacheRecency &recency_;
  std::unordered_map<uint64_t, Block> blocks_;
  // The resident block on each page, 0 for none. A resident block owns its
  // page: a writer whose block already exists moves onto that block's page,
  // or keeps its own while the block is in transfer.
  std::vector<uint64_t> blockOnPage_;
  std::unordered_multimap<uint64_t, uint64_t> index_;
  uint64_t nextBlockId_ = 1;
  // Bumped by every insert, erase and poison. While it is unchanged every
  // block a probe matched is still in the graph and still matches, and no
  // new block extends the match. Blocks leave leaf first and a poisoned
  // block hides those below it, so after a change a probe's chain stays
  // valid up to its first block that is no longer matchable.
  uint64_t generation_ = 1;
  RecencyOrder ramLeaves_;
  RecencyOrder duplicates_;
  RecencyOrder diskLeaves_;
  uint32_t diskBlocks_ = 0;
};

} // namespace splash::engine
