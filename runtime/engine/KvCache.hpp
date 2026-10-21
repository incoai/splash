#pragma once

#include "engine/CacheRecency.hpp"
#include "engine/KvPool.hpp"
#include "engine/KvTier.hpp"
#include "engine/RecencyOrder.hpp"
#include "engine/Types.hpp"
#include "model/Model.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splash::engine {

struct BlockIdentity final {
  uint64_t lo = 0;
  uint64_t hi = 0;

  bool operator==(const BlockIdentity &) const = default;
};

// The spans are sorted by offset and disjoint, as Engine::submit requires.
[[nodiscard]] BlockIdentity
blockImageIdentity(uint64_t blockBegin, uint32_t blockTokens,
                   std::span<const ImageSpan> spans) noexcept;
// A block's key beside its parent and tokens: its images' identity, which
// the first block of a salted request's prompt folds its cache salt into,
// so that the chains of different salts part at the root. A persistent tier
// records it with each block, so a change to its values bumps
// kPersistentCacheFormat; dev/tests/engine/kv_page_cache_test.cpp pins them.
[[nodiscard]] BlockIdentity blockIdentity(uint64_t blockBegin, uint32_t blockTokens,
                                          std::span<const ImageSpan> spans,
                                          CacheSalt salt) noexcept;

// Content-addressed target-KV blocks across two tiers. Matching walks the
// chained full-page hashes from the root. A block holds a pool page, a disk
// slot, or both; resident blocks form a subtree at the root, so a matched
// chain is a resident prefix followed by disk-only blocks. A resident block's
// page holds its K/V (kv::PageForm::Full) or, once the block spilled, only
// its index, its K/V living in its copy alone (kv::PageForm::Index).
// Composite recurrent states are a separate sparse layer. This class owns
// exactly one prefix reference for every resident block.
class KvCache final {
public:
  static constexpr uint32_t pageTokens = kv::kPageTokens;
  static constexpr uint32_t noPage = std::numeric_limits<uint32_t>::max();

  struct BlockMatch {
    uint64_t id = 0;
    uint32_t physicalPage = noPage;
  };

  // What moves a block's page from or to the tier: a restore, whose bytes
  // are not valid until it lands (Filling), or a write, whose bytes are
  // final (Draining). Either keeps the page where it is.
  enum class Transfer : uint8_t { None, Filling, Draining };

  // Root first; noPage marks a disk-only block.
  struct Chain final {
    std::vector<uint64_t> blocks;
    std::vector<uint32_t> pages;
  };

  // With spill, requests may spill the blocks (Cache's spill policy): a
  // block on a page a request holds is held (holdPage), and a block spills
  // only to a copy of what its page holds (kvCopied), so that it has one
  // content, whichever form its page has. A persistent cache keeps the
  // copies no state needs in an order of their own (unneededCopy).
  KvCache(KvPool &pool, CacheRecency &recency, bool spill = false, bool persistent = false)
      : pool_(pool), recency_(recency), spill_(spill), persistent_(persistent),
        blockOnPage_(pool.pageCount()) {}
  KvCache(const KvCache &) = delete;
  KvCache &operator=(const KvCache &) = delete;
  ~KvCache() noexcept;

  // Keys are the parent block, the exact tokens, and the block's identity
  // (blockIdentity), which is zero for a text-only block of a prompt without
  // a cache salt.
  [[nodiscard]] std::optional<BlockMatch> find(uint64_t parentBlock,
                                               std::span<const uint32_t> tokens,
                                               BlockIdentity identity) const;
  // Existing content is returned as is: the writer moves onto its page,
  // either form, unless a restore is filling it, and then keeps its own. A
  // disk-only block adopts the writer's page instead and keeps its copy,
  // which holds what another command computed (kvCopied).
  [[nodiscard]] BlockMatch insert(uint64_t parentBlock,
                                  std::span<const uint32_t> tokens,
                                  uint32_t physicalPage,
                                  BlockIdentity identity);
  // Takes back a disk-only block an earlier process left in a persistent
  // tier, under its id, key and recency; a parent comes back before its
  // children. New ids continue after it. False, taking nothing, when
  // another block holds the key.
  [[nodiscard]] bool adoptDiskBlock(uint64_t blockId, uint64_t parentBlock,
                                    std::span<const uint32_t> tokens, BlockIdentity identity,
                                    std::shared_ptr<KvDiskSlot> slot, uint64_t lastUsed);
  // New ids continue after this one, which an earlier process recorded.
  void continueIdsAfter(uint64_t blockId) noexcept;
  // A block's key and recency, which a persistent tier records with its copy.
  struct Key final {
    uint64_t parent = 0;
    std::span<const uint32_t> tokens;
    BlockIdentity identity;
  };
  [[nodiscard]] Key key(uint64_t blockId) const;
  [[nodiscard]] uint64_t lastUsed(uint64_t blockId) const;
  [[nodiscard]] Transfer transfer(uint64_t blockId) const;
  // The blocks right below this one.
  [[nodiscard]] std::vector<uint64_t> children(uint64_t blockId) const;

  void retainActive(uint64_t blockId);
  void releaseActive(uint64_t blockId) noexcept;
  // A request's reference to a page (KvPool::retainPage and releasePage,
  // not as its owner), which marks the block on it held while any request
  // holds the page: neither quota nor RAM reclaim takes a held block or its
  // copy, unless the copy holds what another command computed (kvCopied),
  // which no request reads. Only a cache whose requests may spill
  // (CacheSpillPolicy) holds blocks; on a layout that cannot, quota strips
  // and evicts copies as it always has. A poisoned block leaves once nothing
  // holds it or keeps it otherwise.
  void holdPage(uint32_t page);
  void unholdPage(uint32_t page);
  [[nodiscard]] bool held(uint64_t blockId) const;
  // Held blocks whose K/V has a copy (kvCopied): the copies of active KV
  // (model::DiskBudget::commit). A copy of another command's K/V beside a
  // held page is a cached copy.
  [[nodiscard]] uint32_t heldCopies() const noexcept { return heldCopies_; }
  void touch(uint64_t blockId) noexcept;

  [[nodiscard]] Chain chain(uint64_t blockId) const;
  [[nodiscard]] bool contains(uint64_t blockId) const noexcept;
  // The block is in the graph and not poisoned: find() can return it.
  [[nodiscard]] bool matchable(uint64_t blockId) const noexcept;
  [[nodiscard]] uint32_t chainLength(uint64_t blockId) const;
  [[nodiscard]] uint64_t generation() const noexcept { return generation_; }

  // Tiers. A block in transfer is moving between them and is neither
  // evicted nor dropped until the transfer is cleared.
  [[nodiscard]] uint32_t page(uint64_t blockId) const;
  // The block on a page; 0 for none.
  [[nodiscard]] uint64_t blockOn(uint32_t page) const { return blockOnPage_.at(page); }
  [[nodiscard]] std::shared_ptr<KvDiskSlot> slot(uint64_t blockId) const;
  // The block's K/V has a copy: a slot that holds what its page holds, or
  // its only K/V. The copy a disk-only block keeps beside the page a writer
  // filled it with (insert) is none: a restore point's chain keeps it, until
  // a copy of the page replaces it or the page goes.
  [[nodiscard]] bool kvCopied(uint64_t blockId) const;
  [[nodiscard]] bool hasDiskChildren(uint64_t blockId) const;
  // StateCache counts each of its entries in and out, and each entry while
  // an unfinished request uses it: a state restores through the KV of its
  // own block and every block above.
  void countState(uint64_t blockId, bool added, bool inUse) noexcept;
  void countStateInUse(uint64_t blockId, bool added) noexcept;
  // In a persistent cache, StateCache counts each state's disk copy in from
  // the first attempt of its write and out when it goes: the redundant copies
  // of a chain are what the next process takes back for the states on disk
  // at their blocks or below.
  void countDiskState(uint64_t blockId, bool added) noexcept;
  // A state sits below the block. Without one, the disk-only blocks below
  // it are never read again.
  [[nodiscard]] bool stateBelow(uint64_t blockId) const;
  // A state an unfinished request uses sits below the block.
  [[nodiscard]] bool stateInUseBelow(uint64_t blockId) const;
  // StateCache notes each ordinary publication or reuse of a state at this
  // block; lookups that find the block without one report a lost state.
  void noteState(uint64_t blockId);
  [[nodiscard]] bool hadState(uint64_t blockId) const;
  // The resident pages of the form, on the chains of these blocks (each
  // block and every block above it), that no request holds and no transfer
  // moves; each counts once. A request holds every page of its chain, so a
  // chain's count stops at the first page one holds.
  [[nodiscard]] uint32_t idlePagesOnChains(std::span<const uint64_t> blocks, kv::PageForm form) const;
  // Resident, without resident children, users or holders: its page can go.
  [[nodiscard]] bool residentLeaf(uint64_t blockId) const;
  // Starts a transfer of the block's page, or ends it with None; one at a
  // time.
  void setTransfer(uint64_t blockId, Transfer transfer);
  // Publishes the block's disk copy. A block whose page holds its K/V may
  // drop it with null; one whose page holds only its index cannot, and a
  // disk-only block is erased instead.
  void setSlot(uint64_t blockId, std::shared_ptr<KvDiskSlot> slot);
  // Returns the page of a resident leaf that has a disk copy to the pool.
  void dropPage(uint64_t blockId);
  // Gives a disk-only block a page whose content follows, by restore or from
  // the request that recomputed it. The parent must be resident.
  void adoptPage(uint64_t blockId, uint32_t page);
  // The pool moved pages, within their form (KvPool::compactExtent) or to
  // the other one (KvPool::movePages): the block on each of them, if any,
  // names the page it moved to and takes its form. It visits only the moved
  // pages.
  void followPages(const KvPageMoves &moves) noexcept;
  // Abandons an unsubmitted restore at an unused leaf, keeping its disk copy.
  [[nodiscard]] bool abandonRestore(uint64_t blockId);
  // A restore or a read of its copy that failed: the block matches nothing
  // any more and leaves once its subtree, users and holders are gone. Its
  // copy goes at once, unless a request holds the block or its page holds
  // only its index: then it leaves with the block.
  void poison(uint64_t blockId);

  // Oldest resident leaf (no resident children) that no request uses. Pass
  // the previous candidate to continue the scan without a lookup.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  evictionCandidate(uint64_t after) const;
  // Oldest holder of a disk copy to replace that no request reads (reindex):
  // with duplicate, a resident block whose copy is redundant; otherwise a
  // disk-only block without children. Pass the previous candidate to
  // continue the scan.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  diskCandidate(bool duplicate, uint64_t after = 0) const;
  // In a persistent cache, the oldest holder of a disk copy that no request
  // reads and no state needs: a redundant copy once no state with a disk
  // copy sits at its block or below, as a state in RAM restores through the
  // page, and the only copy of a spilled or disk-only leaf without a state.
  [[nodiscard]] std::optional<CacheEvictionCandidate> unneededCopy() const noexcept {
    return unneeded_.oldest();
  }
  // The blocks below a block, each after its children, visiting only that
  // subtree; empty when one of them is in transfer or active. Below a
  // resident leaf they are disk-only; below a poisoned block some may be
  // resident.
  [[nodiscard]] std::vector<uint64_t> subtree(uint64_t blockId) const;
  // Only a block without children, users, holders or transfer can be
  // removed. Its copy goes with it, and the copy's slot once a reader that
  // still holds it (KvTier::copy) lets go. The caller handles any composite
  // state attached to it first.
  void erase(uint64_t blockId);

  // Blocks with a disk copy, resident or not; the pool counts the resident
  // blocks (KvPoolSnapshot::pagesPrefix).
  [[nodiscard]] uint32_t diskBlocks() const noexcept { return diskBlocks_; }
  // Blocks whose page holds only their index: spilled blocks.
  [[nodiscard]] uint32_t indexBlocks() const noexcept { return indexBlocks_; }
  // Copies given to a block that had one before: writes of content that
  // was on the disk once already.
  [[nodiscard]] uint64_t rewrites() const noexcept { return rewrites_; }
  // Tests: std::logic_error unless every block's links, counts, form and
  // held mark agree with the graph and the pool, every order holds exactly
  // the blocks reindex places in it, and, between commands, no copy of a
  // block that nothing holds or moves has a reader. A persistent cache's
  // caller names the blocks where a state with a disk copy sits
  // (StateCache::onDisk), which the blocks' counts of them must match.
  void checkInvariants(const std::function<bool(uint64_t blockId)> &stateOnDisk = {}) const;

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
    BlockIdentity identity;
    uint32_t page = noPage;
    std::shared_ptr<KvDiskSlot> slot;
    uint32_t children = 0;
    uint32_t residentChildren = 0;
    uint32_t statesHere = 0;
    uint32_t statesBelow = 0;
    uint32_t statesInUseBelow = 0;
    // States with a disk copy at the block or below it, in a persistent
    // cache.
    uint32_t diskStates = 0;
    uint32_t activeUsers = 0;
    uint32_t depth = 0;
    uint64_t lastUsed = 0;
    Transfer transfer = Transfer::None;
    // Its page's form, kept here so that reindex reads no pool state.
    kv::PageForm form = kv::PageForm::Full;
    bool held = false;
    // It had a disk copy once (rewrites).
    bool copied = false;
    // Its copy holds what another command computed than its page, which a
    // writer filled (insert).
    bool foreignCopy = false;
    bool poisoned = false;
    bool hadState = false;
    RecencyOrder::Node ramNode;
    RecencyOrder::Node diskNode;
    RecencyOrder::Node unneededNode;
  };

  // The orders a block waits in, as its fields stand.
  struct Orders final {
    bool ramLeaf = false;
    bool duplicate = false;
    bool diskLeaf = false;
    bool unneeded = false;
  };

  [[nodiscard]] Block &block(uint64_t blockId);
  [[nodiscard]] const Block &block(uint64_t blockId) const;
  [[nodiscard]] Orders orders(const Block &entry) const noexcept;
  // Places the block in its orders.
  void reindex(Block &entry) noexcept;
  void giveDiskCopy(Block &entry, std::shared_ptr<KvDiskSlot> slot);
  // Marks the block's copy as one of another command's K/V, or of its own;
  // the caller reindexes it.
  void setForeignCopy(Block &entry, bool foreign) noexcept;
  // The block takes the page, whose form it keeps and whose holders, if
  // any, hold it; or leaves its page with noPage.
  void placeOnPage(Block &entry, uint32_t page);
  void setHeld(Block &entry, bool held) noexcept;
  void inherit(Block &parent, uint64_t lastUsed) noexcept;
  // Applies count to every block above this one.
  template <typename Count>
  void countAbove(uint64_t blockId, const Count &count) noexcept;
  // A poisoned block leaves as soon as nothing refers to it.
  void erasePoisonedLeaf(uint64_t blockId) noexcept;

  KvPool &pool_;
  CacheRecency &recency_;
  const bool spill_;
  const bool persistent_;
  std::unordered_map<uint64_t, Block> blocks_;
  // The resident block on each page, 0 for none. A resident block owns its
  // page: a writer whose block already exists moves onto that block's page,
  // or keeps its own while a restore fills it.
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
  RecencyOrder unneeded_;
  uint32_t diskBlocks_ = 0;
  uint32_t indexBlocks_ = 0;
  uint32_t heldCopies_ = 0;
  uint64_t rewrites_ = 0;
};

} // namespace splash::engine
