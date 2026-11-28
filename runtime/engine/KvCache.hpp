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
// chained full-page hashes from the root. A block holds its pool pages, a
// disk slot, or both: its base unit, one page (`page`), and with a window
// its window unit (kv::Layout::windowUnitPages pages, `window`), which it
// takes, holds and gives up together. A point's window K/V exists in memory
// or on the SSD: a resident block holds its unit exactly while the tail of a
// state with a RAM copy holds it (countRamState), or that of a state with a
// disk copy does while its copy has no window part (countDiskState, the
// window K/V not written yet), or a transfer moves the unit: lanes hold the
// pages of the units their windows read as they hold any page, and a unit
// nobody holds is free. Resident blocks form a subtree at the root, so a matched
// chain is a resident prefix followed by disk-only blocks. A resident block's
// page holds its K/V (kv::PageForm::Full); once the block spilled, only its
// index, its K/V living in its copy alone (kv::PageForm::Index). A copy holds
// the block's K/V. Composite recurrent states are a separate sparse layer.
// This class owns exactly one prefix reference for every resident block.
class KvCache final {
public:
  static constexpr uint32_t pageTokens = kv::kPageTokens;
  static constexpr uint32_t noPage = kv::kAbsentPage;

  struct BlockMatch {
    uint64_t id = 0;
    uint32_t physicalPage = noPage;
    // From insert(): the page the block left for the writer's (upgrades),
    // which whoever held it keeps; noPage otherwise.
    uint32_t previousPage = noPage;
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
  // content, whichever form its page has. The copies no state needs wait in
  // an order of their own (unneededCopy). With a
  // window of windowPages pages (kv::Layout::windowPages), each block counts
  // the states whose tail, the windowPages blocks that end at the state's
  // block, holds it (countState): what a restore reads whole.
  KvCache(KvPool &pool, CacheRecency &recency, bool spill = false, bool persistent = false,
          uint32_t windowPages = 0, uint32_t unitPages = 0)
      : pool_(pool), recency_(recency), spill_(spill), persistent_(persistent), windowPages_(windowPages),
        unitPages_(unitPages), blockOnPage_(pool.pageCount()) {
    pool_.observeActivity([this](uint32_t page) { pageActivity(page); });
  }
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
  // Full or Index, unless a restore is filling it, and then keeps its own. A
  // disk-only block adopts the writer's page instead and keeps its copy,
  // which holds what another command computed (kvCopied). With a window the
  // writer keeps its page in any case, as its window reads it whole, and a
  // block on a page takes the writer's, once no transfer moves its own
  // (upgrades): the old page stays with whoever holds it (previousPage), and
  // the block's copy stays as one of another command's K/V.
  // A writer's window unit (`window`, unitPages pages) comes with its page:
  // a block that takes the page takes the unit too where a state's tail
  // holds the block, and the writer keeps it otherwise.
  [[nodiscard]] BlockMatch insert(uint64_t parentBlock,
                                  std::span<const uint32_t> tokens,
                                  uint32_t physicalPage,
                                  BlockIdentity identity,
                                  std::span<const uint32_t> window = {});
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
  // (CacheSpillPolicy) holds blocks. A poisoned block leaves once nothing
  // holds it or keeps it otherwise.
  void holdPage(uint32_t page);
  void unholdPage(uint32_t page);
  [[nodiscard]] bool held(uint64_t blockId) const;
  // What the copies of held blocks' K/V (kvCopied) take of the quota
  // (KvDiskSlot::bytes): the copies of active KV (model::DiskBudget::commit),
  // each once however many requests hold its block. A copy of another
  // command's K/V beside a held page is a cached copy.
  [[nodiscard]] uint64_t heldCopyBytes() const noexcept { return heldCopyBytes_; }
  // Called as a block gains or loses a copy of its own K/V (kvCopied), with
  // the block's depth and page, so that each request holding it on that
  // page counts it (Cache::holdsActiveKvQuota). Erasing a block calls none:
  // a block that can be erased is on no request's chain.
  using CopyChanged = std::function<void(uint64_t blockId, uint32_t depth, uint32_t page, bool copied)>;
  void onCopyChanged(CopyChanged changed) { copyChanged_ = std::move(changed); }
  void touch(uint64_t blockId) noexcept;
  // A request's use of a chain, root first: each block with a disk copy
  // takes the newest copy recency (copyUsed), so that the orders of copies
  // (duplicates, unneeded) take the copies of the conversation used longest
  // ago before those of one in use, and a head conversations share stays
  // newest. A block without a copy is in neither order and is not touched;
  // memory's orders keep reading lastUsed, unchanged.
  void refreshCopies(std::span<const uint64_t> chain) noexcept;
  // Each change of what the answers Cache keeps read at a block (Cache::
  // dropKept): what a search of history (superseded) reads there (Search);
  // the state at it, its class, copies or write (State); a copy or a part
  // of it landing or going, which the tails of the states below it read too
  // (Copy); a unit taken or given up, which only those tails read (Unit). Past kHistoryChanges unread, the cache drops every
  // answer.
  static constexpr size_t kHistoryChanges = 65536;
  enum class Change : uint8_t { Search, State, Copy, Unit };
  struct HistoryChange final {
    uint64_t block;
    Change change;
  };
  void historyChanged(uint64_t blockId, Change change = Change::Search) const noexcept {
    if (historyChanges_.size() < kHistoryChanges)
      historyChanges_.push_back({blockId, change});
    else
      historyOverflow_ = true;
  }
  [[nodiscard]] bool historyUnchanged() const noexcept { return historyChanges_.empty() && !historyOverflow_; }
  // The changes since the last call, and whether more went uncounted.
  [[nodiscard]] std::vector<HistoryChange> takeHistoryChanges(bool &overflow) const noexcept {
    overflow = std::exchange(historyOverflow_, false);
    return std::exchange(historyChanges_, {});
  }

  [[nodiscard]] Chain chain(uint64_t blockId) const;
  [[nodiscard]] bool contains(uint64_t blockId) const noexcept;
  // The block is in the graph and not poisoned: find() can return it.
  [[nodiscard]] bool matchable(uint64_t blockId) const noexcept;
  [[nodiscard]] uint32_t chainLength(uint64_t blockId) const;
  [[nodiscard]] uint64_t generation() const noexcept { return generation_; }

  // Tiers. A block in transfer is moving between them and is neither
  // evicted nor dropped until the transfer is cleared.
  [[nodiscard]] uint32_t page(uint64_t blockId) const;
  // The block's window unit's pages, none while it holds no unit, and all
  // its pages, base page first, what the tier reads and writes.
  [[nodiscard]] std::span<const uint32_t> window(uint64_t blockId) const { return block(blockId).window; }
  [[nodiscard]] std::vector<uint32_t> pages(uint64_t blockId) const;
  // The pages of a block that holds its unit: its base page and its window
  // unit.
  [[nodiscard]] uint32_t blockPages() const noexcept { return 1 + unitPages_; }
  [[nodiscard]] uint32_t unitPages() const noexcept { return unitPages_; }
  // The pages a block holds: its base page and its unit, none disk-only.
  [[nodiscard]] uint32_t pagesHeld(uint64_t blockId) const {
    const Block &entry = block(blockId);
    return entry.page == noPage ? 0 : 1 + static_cast<uint32_t>(entry.window.size());
  }
  [[nodiscard]] bool holdsUnit(uint64_t blockId) const { return !block(blockId).window.empty(); }
  // The block on a page, a base page or a window page; 0 for none.
  [[nodiscard]] uint64_t blockOn(uint32_t page) const { return blockOnPage_.at(page); }
  [[nodiscard]] std::shared_ptr<KvDiskSlot> slot(uint64_t blockId) const;
  // The block's K/V has a copy: a slot that holds what its page holds, or
  // its only K/V, a part of each unit it holds. The copy a disk-only block
  // keeps beside the page a writer filled it with (insert) is none: a
  // restore point's chain keeps it, until a copy of the page replaces it or
  // the page goes.
  [[nodiscard]] bool kvCopied(uint64_t blockId) const;
  // The parts of a copy the block's pages hold: its base unit's, and its
  // window unit's while it holds that; and whether its copy, of whatever
  // K/V, holds each of them (its page then frees with no write).
  [[nodiscard]] kv::Components heldParts(uint64_t blockId) const { return heldParts(block(blockId)); }
  [[nodiscard]] bool copyHoldsPages(uint64_t blockId) const { return copyHoldsPages(block(blockId)); }
  // Its copy holds its own K/V, not another command's (insert).
  [[nodiscard]] bool ownCopy(uint64_t blockId) const { return block(blockId).slot && !block(blockId).foreignCopy; }
  // The serial a persistent tier labels the parts of the block's copy with,
  // which a part added later keeps.
  [[nodiscard]] uint64_t copySerial(uint64_t blockId) const { return block(blockId).copySerial; }
  void setCopySerial(uint64_t blockId, uint64_t serial) { block(blockId).copySerial = serial; }
  [[nodiscard]] bool hasDiskChildren(uint64_t blockId) const;
  // StateCache counts each of its entries in and out, and each entry while
  // an unfinished request uses it: a state restores through the KV of its
  // own block and every block above.
  void countState(uint64_t blockId, bool added, bool inUse) noexcept;
  // With a window a state holds its tail, the windowPages blocks that end at
  // its block, which a row past the state's boundary reads whole: StateCache
  // counts each RAM copy in and out (windowPins), whose tail holds the
  // blocks' units, and each disk copy (copyPins, countDiskState), whose tail
  // holds their window parts, or their units until those are written. A
  // resident block that comes to need its unit takes it from `units`, the
  // lane's that publishes or promotes the state (Cache::tailUnits); one that
  // needs it no longer gives it up, once no transfer moves it.
  using TailUnits = std::vector<std::pair<uint64_t, std::vector<uint32_t>>>;
  void countRamState(uint64_t blockId, bool added, const TailUnits &units = {}) noexcept;
  // A publication's scope (Cache::publishCompositeState): a unit whose
  // block's last pin leaves, or whose transfer ends unpinned, is given up as
  // the scope ends, not at once. A publication to the SSD starts its write
  // (StateCache::startWrite) before it pins its tail, and the room the write
  // makes can drop the last pin of a block the new state pins a moment
  // later, whose unit no lane may hold any more.
  class DeferredReleases final {
  public:
    explicit DeferredReleases(KvCache &kv) noexcept : kv_(kv) { kv_.deferring_ = true; }
    ~DeferredReleases();
    DeferredReleases(const DeferredReleases &) = delete;
    DeferredReleases &operator=(const DeferredReleases &) = delete;

  private:
    KvCache &kv_;
  };
  // A write's room (Cache::demoteKv, Cache::writeUnit): the disk orders'
  // candidates pass over the block whose copy the write adds parts to, which
  // it holds, so that room from that copy, freeing nothing, is not offered.
  class PassOver final {
  public:
    PassOver(KvCache &kv, uint64_t block) noexcept : kv_(kv) { kv_.passOver_ = block; }
    ~PassOver() { kv_.passOver_ = 0; }
    PassOver(const PassOver &) = delete;
    PassOver &operator=(const PassOver &) = delete;

  private:
    KvCache &kv_;
  };
  // What a release of a state does to its tail's pins: its RAM copy's leaves
  // (ram), its disk copy's leaves (disk), or one comes as its RAM copy
  // leaves through a write (written). Of each block, the pins in memory that
  // releases before it in the same sequence took off, and the disk pins they
  // added less those they took (Gone).
  struct PinRelease final {
    bool ram = false;
    bool disk = false;
    bool written = false;
  };
  using Gone = std::unordered_map<uint64_t, std::pair<uint32_t, int32_t>>;
  // Of the resident blocks of the state's tail that hold a unit no transfer
  // moves and no lane holds a page of, with `gone` (which it adds its own
  // to): those whose units its release frees now, needed before and not
  // after (now); and those whose last pin in memory it takes while a disk
  // copy's still needs their unwritten window K/V, which free once their
  // window parts are written (written). What a release counts and what it
  // takes (Cache::take).
  // And those a write moves whose last pin in memory it takes, whose units
  // free as their writes land (landing).
  struct Freed final {
    std::vector<uint64_t> now;
    std::vector<uint64_t> written;
    std::vector<uint64_t> landing;
  };
  [[nodiscard]] Freed freedBy(uint64_t stateBlock, Gone &gone, PinRelease release) const;
  // Units given up at a last pin's leaving whose pages no lane held, so that
  // they went free (freedBy's take).
  [[nodiscard]] uint64_t unitsFreed() const noexcept { return unitsFreed_; }
  void countStateInUse(uint64_t blockId, bool added) noexcept;
  // StateCache counts each state's disk copy in from the first attempt of its
  // write and out when it goes: with a window, its tail's copyPins
  // (countRamState).
  void countDiskState(uint64_t blockId, bool added, const TailUnits &units = {}) noexcept;
  // A state sits below the block. Without one, the disk-only blocks below
  // it are never read again.
  [[nodiscard]] bool stateBelow(uint64_t blockId) const;
  // The states below the block, counted as stateBelow counts them.
  [[nodiscard]] uint32_t statesBelow(uint64_t blockId) const;
  // A state an unfinished request uses sits below the block.
  [[nodiscard]] bool stateInUseBelow(uint64_t blockId) const;
  // StateCache notes each ordinary publication or reuse of a state at this
  // block; lookups that find the block without one report a lost state.
  void noteState(uint64_t blockId);
  [[nodiscard]] bool hadState(uint64_t blockId) const;
  // The engine notes each block a reply's decode published (Cache::noteReply):
  // a prompt that matches it renders that reply back as far as the block,
  // whether a state was captured there or not.
  void noteReply(uint64_t blockId);
  [[nodiscard]] bool heldReply(uint64_t blockId) const;
  // The engine notes the block a prompt's head ends at, where it publishes
  // the state the next conversation on that head resumes from.
  void noteHead(uint64_t blockId) {
    block(blockId).head = true;
    historyChanged(blockId);
  }
  // The engine notes the block a reply shares with its prompt's tail and
  // how many of its tokens are the prompt's. It shows no rendering back: it
  // is where a prompt that renders the reply otherwise parts from it.
  void noteReplyStart(uint64_t blockId, uint32_t promptTokens) {
    block(blockId).replyStart = static_cast<uint8_t>(promptTokens);
    historyChanged(blockId);
  }
  [[nodiscard]] uint32_t replyStart(uint64_t blockId) const { return block(blockId).replyStart; }
  [[nodiscard]] bool head(uint64_t blockId) const { return block(blockId).head; }
  // The tail of a state with a RAM copy holds the block; the tails of the
  // states with disk copies that do (countRamState).
  [[nodiscard]] bool tailHeld(uint64_t blockId) const { return block(blockId).windowPins != 0; }
  [[nodiscard]] uint32_t windowPins(uint64_t blockId) const { return block(blockId).windowPins; }
  [[nodiscard]] uint32_t copyPins(uint64_t blockId) const { return block(blockId).copyPins; }
  // Its copy has a window part, landed or being written.
  [[nodiscard]] bool windowCopied(uint64_t blockId) const { return windowCopied(block(blockId)); }
  // Its unit is held for points on the SSD alone, whose window part is not
  // written, and nothing else keeps it: an idle base page, no user, no
  // transfer. A reclaim frees it by writing its window part (Cache::nextUnit).
  [[nodiscard]] bool unwrittenUnit(uint64_t blockId) const { return unwrittenUnit(block(blockId)); }
  // Its unit leaves as a write lands or once one is written (unitLeaving).
  [[nodiscard]] bool unitLeaving(uint64_t blockId) const { return unitLeaving(block(blockId)); }
  // The oldest such block after `after` (0 for the first).
  [[nodiscard]] std::optional<CacheEvictionCandidate> unwrittenUnitCandidate(uint64_t after) const;
  // The oldest block after `after` (0 for the first) whose copy's window
  // part has landed and no state's disk copy needs: a full quota gives it up
  // first (Cache::nextDiskReplacement).
  [[nodiscard]] std::optional<CacheEvictionCandidate> unneededWindowCandidate(uint64_t after) const;
  // A part of the block's copy was given up at the tier: the block counts
  // what its copy takes now.
  void partGivenUp(uint64_t blockId);
  // What leaves a resident page in place when a reclaim takes all it may
  // (Cache::reach): its block, or a block below it, has a reason the
  // reclaim's walk names, and a page stays while one below it does. Busy:
  // the block is used, a request holds its page, a restore fills it, or it
  // is a poisoned leaf, which no reclaim reaches. Moving: a write drains its
  // page, which a reclaim reaches once the write lands. Uncopied: it has no copy and a
  // state that needs its KV sits at it or below (setStateMarks), which a
  // writable tier writes before the page goes. Writer, UsedWriter: the
  // state at it is a RAM copy a reclaim may take that has no disk copy,
  // not in use or in use, which leaves RAM only through a write while the
  // tier takes one and needs its chain from then on. Used: the state at it
  // is in use, which a reclaim of the ordinary class keeps. Wanted: a chain
  // a waiting request restores ends at it (countWanted). ResumePoint: the
  // resume point (moveResumePoint). A request that holds a page holds every
  // page above it, so the chains of Used, Wanted and ResumePoint count no
  // page from one a request holds up.
  enum class Keep : uint8_t { Busy, Moving, Uncopied, Writer, UsedWriter, Used, Wanted, ResumePoint };
  static constexpr size_t kKeeps = 8;
  // A set of Keep reasons.
  using Keeps = uint8_t;
  [[nodiscard]] static constexpr Keeps keepBit(Keep keep) noexcept {
    return static_cast<Keeps>(1u << static_cast<unsigned>(keep));
  }
  // The reasons a reclaim keeps a chain for.
  static constexpr Keeps kChainKeeps =
      static_cast<Keeps>(1u << static_cast<unsigned>(Keep::Used) | 1u << static_cast<unsigned>(Keep::Wanted) |
                         1u << static_cast<unsigned>(Keep::ResumePoint));
  // What the state at a block asks of the KV it restores through: its
  // reasons, Writer or UsedWriter, Used, or none; whether it needs the KV
  // as it is: every state but a RAM copy a reclaim may take without a disk
  // copy, which leaves through a write (that makes its chain needed) or
  // with no copy; and whether it is a RAM copy a reclaim may take
  // (StateCache::candidate). Without a state, none.
  struct StateMarks final {
    Keeps keeps = 0;
    bool needed = false;
    bool candidate = false;
    friend bool operator==(const StateMarks &, const StateMarks &) = default;
  };
  // StateCache names, as they change, the marks of the state at the block
  // and whether it has a RAM copy (inRam), which a reclaim's walk takes
  // before the leaf (LeafKind). A RAM copy sits on a block with a page
  // (Cache::checkInvariants), so a Writer or UsedWriter names a resident block.
  void setStateMarks(uint64_t blockId, StateMarks state, bool inRam) noexcept;
  // The reasons at the block or below it: its page frees now for a walk
  // none of whose reasons they hold.
  [[nodiscard]] Keeps keeps(uint64_t blockId) const;
  // Cache names, as they change, the chains requests waiting to start
  // restore, each by its last block: one more or one less ends at the block.
  void countWanted(uint64_t blockId, bool added) noexcept;
  // StateCache names the resume point as it moves (0 for none).
  void moveResumePoint(uint64_t from, uint64_t to) noexcept;
  // Of each form, in pages (its base page and its unit, if it holds one,
  // each block), what a reclaim's walk that leaves the pages `mask` names
  // in place reaches; the units of `released` blocks, whose last pins the
  // walk's releases of states take (freedBy), free now whether their block
  // does or not: the resident pages that free now, none of `mask`'s
  // reasons at their block or below it (now); those that free too once the
  // state at `written` (a Writer or UsedWriter, on a block with a page as
  // every RAM copy is; 0 for none) has left RAM through its write and needs
  // its chain as a state on disk does (written), up from it to the first
  // block without a page. The chains of the states `kept`, each block and
  // the blocks above it, free neither now nor once written. Every page a
  // reclaim of the walk reaches at all, now or once the writes land
  // (reachable): no Busy at it or below it, nor for an idle page (no write
  // drains it) a chain reason `mask` names or a chain of `kept`, since a
  // page a write drains frees as it lands, kept or not; and none above a
  // write in `unwaited`, which no reclaim waits for. Reachable too, whatever
  // keeps their base pages, the units a reclaim frees by writing their
  // window parts: those of unwrittenUnit blocks, and those of `toWrite`,
  // which the walk's releases of states leave so (freedBy's written); an
  // unwritten one only while the tier takes writes (`writable`) and not one
  // of `unitsKept`. The units of `staged` free once the written state has
  // left RAM (written).
  // The RAM copies a reclaim may take at the blocks that free now
  // (candidates). The cache keeps the counts by reason as blocks, states
  // and transfers change, so a reach costs the chains of `kept`, `unwaited`
  // and `written` alone, not the cache.
  struct PageReach final {
    std::array<uint32_t, kv::kPageForms> now{};
    std::array<uint32_t, kv::kPageForms> written{};
    std::array<uint32_t, kv::kPageForms> reachable{};
    uint32_t candidates = 0;
    friend bool operator==(const PageReach &, const PageReach &) = default;
  };
  [[nodiscard]] PageReach pageReach(Keeps mask, std::span<const uint64_t> kept, std::span<const uint64_t> unwaited,
                                    uint64_t written, std::span<const uint64_t> released = {},
                                    std::span<const uint64_t> toWrite = {}, bool writable = true,
                                    std::span<const uint64_t> staged = {},
                                    std::span<const uint64_t> unitsKept = {}) const;
  // Tests: with `check`, every pageReach also follows its rule block by
  // block over the whole cache, each block's reasons as they stand and the
  // written state's chain recounted from what its blocks hold, and throws
  // std::logic_error where the counts differ.
  void checkReaches(bool check) noexcept { checkReaches_ = check; }
  // The pages of each form no request holds and no transfer moves on the
  // chains of the reasons `chains` names (kChainKeeps), each block's base
  // page and unit.
  [[nodiscard]] std::array<uint32_t, kv::kPageForms> keptPages(Keeps chains) const;
  // Blocks the counts above visited, beside the counts the cache keeps:
  // their work, which grows with the chains they walk and not the cache.
  [[nodiscard]] uint64_t reachVisits() const noexcept { return reachVisits_; }
  // Resident, without resident children, users or holders: its page can go
  // once no transfer moves it.
  [[nodiscard]] bool residentLeaf(uint64_t blockId) const;
  // Resident, without users or holders: a resident leaf once its resident
  // children leave.
  [[nodiscard]] bool leafOnceChildrenLeave(uint64_t blockId) const;
  // Starts a transfer of the block's page, or ends it with None; one at a
  // time.
  void setTransfer(uint64_t blockId, Transfer transfer);
  // Publishes the block's disk copy, of what its page holds now, or the
  // parts a copy it has gained. A block whose page holds its K/V may drop it
  // with null; one whose page holds only its index cannot, and a disk-only
  // block is erased instead.
  void setSlot(uint64_t blockId, std::shared_ptr<KvDiskSlot> slot);
  // Returns the page of a resident leaf that has a disk copy to the pool.
  void dropPage(uint64_t blockId);
  // Gives a disk-only block a page whose content follows, by restore or from
  // the request that recomputed it. The parent must be resident.
  void adoptPage(uint64_t blockId, uint32_t page, std::span<const uint32_t> window = {});
  // The pool moved pages, within their form (KvPool::compactExtent) or to
  // another one (KvPool::movePages): the block on each of them, if any,
  // names the page it moved to and takes its form. It visits only the moved
  // pages.
  void followPages(const KvPageMoves &moves) noexcept;
  // Abandons an unsubmitted restore at an unused leaf, keeping its disk copy.
  [[nodiscard]] bool abandonRestore(uint64_t blockId);
  // A restore or a read of its copy that failed: the block matches nothing
  // any more and leaves once its subtree, users, holders and transfer are
  // gone. Its copy goes at once, unless a request holds the block or its
  // page holds only its index: then it leaves with the block.
  void poison(uint64_t blockId);
  // Erases the block if it is poisoned and nothing refers to it any more: no
  // child, user, holder or transfer.
  void erasePoisonedLeaf(uint64_t blockId) noexcept;

  // Oldest resident leaf (no resident children) that no request uses, of
  // any kind (LeafKind). Pass the previous candidate to continue the scan.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  evictionCandidate(uint64_t after) const;
  // What restores through a resident leaf, by which a reclaim's walk takes
  // it (Cache::nextUnit): nothing, no state at it or below it (dead); else
  // whether a state in use sits at it or below it, whether the state at the
  // leaf has a RAM copy, which the walk takes before the leaf, and whether
  // the leaf has a copy, so that its page frees with no write. A dead leaf's
  // kind names nothing more. The resident leaves of each kind wait in an
  // order of their own, which follows the states and copies as they change,
  // so that a walk goes over the leaves it may take and no other.
  struct LeafKind final {
    bool dead = false;
    bool inUse = false;
    bool stateInRam = false;
    bool copied = false;
    friend bool operator==(const LeafKind &, const LeafKind &) = default;
  };
  // The leaves a walk goes over: the dead ones; or those a state in use
  // restores through (inUse), or those another state does, with the state
  // at the leaf in RAM too (withStateInRam), and of those whose state is
  // not in RAM all, or only the copied ones (copiedOnly).
  struct LeafWalk final {
    bool dead = false;
    bool inUse = false;
    bool withStateInRam = false;
    bool copiedOnly = false;
  };
  // The oldest resident leaf of the walk's kinds after `after` (0 for the
  // first), from the orders of those kinds alone.
  [[nodiscard]] std::optional<CacheEvictionCandidate> leafCandidate(const LeafWalk &walk, uint64_t after) const;
  // The leaves leafCandidate() gave: a walk's work, which grows with the
  // leaves it tries and not the cache.
  [[nodiscard]] uint64_t leafVisits() const noexcept { return leafVisits_; }
  // Oldest holder of a disk copy to replace that no request reads (reindex):
  // with duplicate, a resident block whose copy is redundant; otherwise a
  // disk-only block without children. Pass the previous candidate to
  // continue the scan.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  diskCandidate(bool duplicate, uint64_t after = 0) const;
  // The oldest holder of a disk copy that no request reads and no state
  // needs: the only copy of a spilled or disk-only leaf without a state, and
  // a redundant copy once no state, in memory or on disk, sits at its block
  // or below.
  [[nodiscard]] std::optional<CacheEvictionCandidate> unneededCopy() const noexcept {
    return passing(unneeded_, 0, true);
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
  // What their copies take of the quota (KvDiskSlot::bytes).
  [[nodiscard]] uint64_t diskBytes() const noexcept { return diskBytes_; }
  // Blocks whose page holds only their index: spilled blocks.
  [[nodiscard]] uint32_t indexBlocks() const noexcept { return indexBlocks_; }
  // Copies given to a block that had one before: writes of content that
  // was on the disk once already.
  [[nodiscard]] uint64_t rewrites() const noexcept { return rewrites_; }
  // Blocks that left their page for that of a writer that computed them
  // again (insert).
  [[nodiscard]] uint64_t upgrades() const noexcept { return upgrades_; }
  // Blocks a state's tail holds (countRamState), and those that hold a unit.
  [[nodiscard]] uint32_t pinnedBlocks() const noexcept { return pinnedBlocks_; }
  [[nodiscard]] uint32_t windowUnits() const noexcept { return windowUnits_; }
  // What the cache was told of a block (setStateMarks, countWanted,
  // moveResumePoint).
  struct Marks final {
    StateMarks state;
    bool stateInRam = false;
    uint32_t wanted = 0;
    bool resumePoint = false;
    friend bool operator==(const Marks &, const Marks &) = default;
  };
  // Tests: std::logic_error unless every block's links, counts, form, unit
  // and held mark agree with the graph and the pool, every order holds exactly
  // the blocks reindex places in it, and, between commands, no copy of a
  // block that nothing holds or moves has a reader; with a window, the pins
  // are those of the states' tails. A persistent cache's caller names the
  // blocks where a state with a disk copy sits (StateCache::onDisk), which
  // the blocks' counts of them must match. Every block's reasons to stay
  // (Keep) and the counts of them follow its fields and its marks; a caller
  // that names each block's marks as they should be (`marks`) has them
  // checked too, and one that names the kind a walk takes each resident
  // leaf as (`walkKind`) has the leaves' orders checked against it.
  void checkInvariants(const std::function<bool(uint64_t blockId)> &stateOnDisk = {},
                       const std::function<Marks(uint64_t blockId)> &marks = {},
                       const std::function<LeafKind(uint64_t blockId)> &walkKind = {}) const;

private:
  static constexpr uint16_t noCount = std::numeric_limits<uint16_t>::max();
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
    std::vector<uint32_t> window;
    std::shared_ptr<KvDiskSlot> slot;
    // What the copy took of the quota when last given (KvDiskSlot::bytes),
    // and the serial its parts are labelled with.
    uint64_t copyBytes = 0;
    uint64_t copySerial = 0;
    uint32_t children = 0;
    uint32_t residentChildren = 0;
    uint32_t statesHere = 0;
    uint32_t statesBelow = 0;
    uint32_t statesInUseBelow = 0;
    uint32_t activeUsers = 0;
    uint32_t depth = 0;
    uint64_t lastUsed = 0;
    // When a request last resumed through the block (refreshCopies): the
    // orders of copies read the later of the two, memory's orders lastUsed.
    uint64_t copyUsed = 0;
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
    bool heldReply = false;
    bool head = false;
    uint8_t replyStart = 0;
    // The tails of states with a RAM copy, and of those with a disk copy,
    // each counted once, that hold the block (countRamState).
    uint32_t windowPins = 0;
    uint32_t copyPins = 0;
    // While a resident leaf, in the order of its kind (leaves_).
    RecencyOrder::Node ramNode;
    RecencyOrder::Node diskNode;
    RecencyOrder::Node unneededNode;
    RecencyOrder::Node unwrittenNode;
    RecencyOrder::Node windowNode;
    // What keeps its page, and those above it, in place (Keep): its state's
    // reasons and whether that state needs its KV (setStateMarks), the
    // waiting chains that end at it and whether it is the resume point;
    // from those and its own fields the reasons at it, and those at it or
    // below it with the children that have each, and whether a state that
    // needs its KV sits at it or below, with the children below which one
    // does. A resident block counts in keepCounts_ under its form, its
    // reasons and whether it is idle (no request holds it, no transfer
    // moves it): `counted` names where, noCount for none.
    StateMarks state;
    // No reason: whether the state has a RAM copy, which places a resident
    // leaf (LeafKind).
    bool stateInRam = false;
    uint32_t wanted = 0;
    bool resumePoint = false;
    Keeps keepsHere = 0;
    Keeps keeps = 0;
    bool needed = false;
    uint32_t neededChildren = 0;
    std::array<uint32_t, kKeeps> keepChildren{};
    uint16_t counted = noCount;
    // With a RAM copy a reclaim may take, the reasons it counts under in
    // candidateCounts_; noCount for none.
    uint16_t candidateCounted = noCount;
    // An unwrittenUnit block: the reasons it counts under in
    // unwrittenCounts_; noCount for none.
    uint16_t unwrittenCounted = noCount;
  };
  // keepCounts_'s index of a resident block's form, unit, reasons and
  // idleness.
  [[nodiscard]] static constexpr uint16_t keepCount(kv::PageForm form, bool unit, Keeps keeps, bool idle) noexcept {
    return static_cast<uint16_t>((static_cast<unsigned>(form) << (kKeeps + 2)) | (unit ? 1u << (kKeeps + 1) : 0u) |
                                 (unsigned{keeps} << 1) | (idle ? 1u : 0u));
  }
  // The one rule of a block's reasons to stay (Keep), from its fields, the
  // counts of its children with each reason and of those below which a
  // state needs its KV: the reasons at it (keepsHere); those at it or below
  // it, the chain reasons only up to a page a request holds (keeps);
  // whether a state that needs its KV sits at it or below (needed); and
  // where a resident block counts in keepCounts_ (counted) and, with a RAM
  // copy a reclaim may take, in candidateCounts_ (candidate), noCount for
  // none. rekeep keeps the block by it, and checkInvariants recounts by it.
  struct Reasons final {
    Keeps here = 0;
    Keeps keeps = 0;
    bool needed = false;
    uint16_t counted = noCount;
    uint16_t candidate = noCount;
  };
  [[nodiscard]] Reasons reasons(const Block &entry, const std::array<uint32_t, kKeeps> &children,
                                uint32_t neededChildren) const noexcept;

  // The orders a block waits in, as its fields stand: of the resident
  // leaves, that of its kind (leafOrder).
  struct Orders final {
    std::optional<size_t> leaf;
    bool duplicate = false;
    bool diskLeaf = false;
    bool unneeded = false;
    bool unwritten = false;
    bool unneededWindow = false;
  };

  [[nodiscard]] Block &block(uint64_t blockId);
  [[nodiscard]] const Block &block(uint64_t blockId) const;
  // The order of the resident leaves of a kind: the dead ones', or one per
  // combination of the others.
  static constexpr size_t kLeafOrders = 9;
  [[nodiscard]] static constexpr size_t leafOrder(const LeafKind &kind) noexcept {
    return kind.dead ? 0
                     : 1 + (size_t{kind.inUse} << 2 | size_t{kind.stateInRam} << 1 | size_t{kind.copied});
  }
  [[nodiscard]] LeafKind leafKind(const Block &entry) const noexcept;
  [[nodiscard]] kv::Components heldParts(const Block &entry) const noexcept {
    return static_cast<kv::Components>(kv::componentBit(kv::Component::Base) |
                                       (entry.window.empty() ? 0 : kv::componentBit(kv::Component::Window)));
  }
  [[nodiscard]] bool copyHoldsPages(const Block &entry) const noexcept {
    return entry.slot && (entry.slot->parts() & heldParts(entry)) == heldParts(entry);
  }
  [[nodiscard]] static bool windowCopied(const Block &entry) noexcept {
    return entry.slot && (entry.slot->parts() & kv::componentBit(kv::Component::Window));
  }
  // It needs its unit in memory: resident, and a RAM copy's tail holds it, or
  // a disk copy's while its copy has no window part.
  [[nodiscard]] static bool needsUnit(const Block &entry) noexcept {
    return entry.page != noPage && (entry.windowPins || (entry.copyPins && !windowCopied(entry)));
  }
  [[nodiscard]] bool unwrittenUnit(const Block &entry) const noexcept;
  // An unwrittenUnit, or a unit a write moves that frees as it lands, the
  // block needing it no longer: what a reclaim's reach counts whatever keeps
  // the block's base page (unwrittenCounts_).
  [[nodiscard]] bool unitLeaving(const Block &entry) const noexcept;
  // The block takes its unit from `units` where it comes to need one, or
  // gives it up where it no longer does and no transfer moves it.
  void reunit(Block &entry, const TailUnits &units = {}) noexcept;
  [[nodiscard]] Orders orders(const Block &entry) const noexcept;
  // Places the block in its orders.
  void reindex(Block &entry) noexcept;
  // Moves a resident leaf to the order of its kind, as what restores through
  // it changes; reindex() places it too.
  void placeLeaf(Block &entry) noexcept;
  // The oldest resident leaf after `after` (0 for the first) of the leaf
  // orders whose bits `orders` sets.
  [[nodiscard]] std::optional<CacheEvictionCandidate> oldestLeaf(uint32_t orders, uint64_t after) const;
  // Gives the block a copy, or none.
  void giveDiskCopy(Block &entry, std::shared_ptr<KvDiskSlot> slot);
  // Marks the block's copy as one of another command's K/V, or of its own;
  // the caller reindexes it.
  void setForeignCopy(Block &entry, bool foreign) noexcept;
  // The block takes the page and the window unit, whose form it keeps and
  // whose holders, if any, hold it; or leaves its pages with noPage.
  void placeOnPage(Block &entry, uint32_t page, std::span<const uint32_t> window = {});
  // The block takes a unit, or gives its own up (its last pin left, or the
  // transfer that moved it ended).
  void takeUnit(Block &entry, std::span<const uint32_t> window);
  void giveUnitUp(Block &entry) noexcept;
  // A block takes the pages of a writer that computed it again (insert).
  void upgrade(Block &entry, uint32_t page, std::span<const uint32_t> window);
  // The cache's one reference to each of a block's pages, taken and given
  // back: its base page, and its unit when it holds one.
  void own(uint32_t page, std::span<const uint32_t> window);
  void disown(uint32_t page, std::span<const uint32_t> window);
  void setHeld(Block &entry, bool held) noexcept;
  void inherit(Block &parent, uint64_t lastUsed) noexcept;
  // Applies count to every block above this one.
  template <typename Count>
  void countAbove(uint64_t blockId, const Count &count) noexcept;
  // Brings the block's reasons (Keep) and its count up to date with its
  // fields, and the blocks above it as far as theirs change; reindex()
  // calls it, and so does a change of a page's requests (KvPool's
  // observer). forget() takes a block that leaves the graph out of its
  // parent's counts and keepCounts_ first.
  void rekeep(Block &entry) noexcept;
  void forget(Block &entry) noexcept;
  void pageActivity(uint32_t page) noexcept;

  KvPool &pool_;
  CacheRecency &recency_;
  const bool spill_;
  const bool persistent_;
  const uint32_t windowPages_;
  const uint32_t unitPages_;
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
  // The resident leaves by kind (leafOrder).
  std::array<RecencyOrder, kLeafOrders> leaves_;
  RecencyOrder duplicates_;
  RecencyOrder diskLeaves_;
  RecencyOrder unneeded_;
  uint32_t diskBlocks_ = 0;
  uint64_t diskBytes_ = 0;
  uint32_t indexBlocks_ = 0;
  uint64_t heldCopyBytes_ = 0;
  CopyChanged copyChanged_;
  uint64_t rewrites_ = 0;
  uint64_t upgrades_ = 0;
  uint32_t pinnedBlocks_ = 0;
  uint32_t windowUnits_ = 0;
  uint64_t unitsFreed_ = 0;
  // A publication's scope (DeferredReleases), and the blocks whose last pin
  // left or whose transfer ended in it.
  bool deferring_ = false;
  std::vector<uint64_t> unpinned_;
  // The block a write's room passes over (PassOver), and an order's
  // candidate past it.
  uint64_t passOver_ = 0;
  [[nodiscard]] std::optional<CacheEvictionCandidate> passing(const RecencyOrder &order, uint64_t after = 0,
                                                              bool copies = false) const;
  // The resident blocks by form, unit, reasons (Keep) and idleness
  // (keepCount).
  std::array<uint32_t, kv::kPageForms << (kKeeps + 2)> keepCounts_{};
  // The blocks with a RAM copy a reclaim may take, by their reasons.
  std::array<uint32_t, size_t{1} << kKeeps> candidateCounts_{};
  // The unitLeaving blocks by their reasons, the unwrittenUnit ones in the
  // order of their recency, and the blocks whose window part no disk copy
  // needs.
  std::array<uint32_t, size_t{2} << kKeeps> unwrittenCounts_{};
  RecencyOrder unwritten_;
  RecencyOrder unneededWindows_;
  mutable uint64_t reachVisits_ = 0;
  bool checkReaches_ = false;
  mutable std::vector<HistoryChange> historyChanges_;
  mutable bool historyOverflow_ = false;
  // pageReach from the counts and the chains it walks, and by its rule.
  [[nodiscard]] PageReach countReach(Keeps mask, std::span<const uint64_t> kept, std::span<const uint64_t> unwaited,
                                     uint64_t written, std::span<const uint64_t> released,
                                     std::span<const uint64_t> toWrite, bool writable,
                                     std::span<const uint64_t> staged, std::span<const uint64_t> unitsKept) const;
  [[nodiscard]] PageReach reachByRule(Keeps mask, std::span<const uint64_t> kept, std::span<const uint64_t> unwaited,
                                      uint64_t written, std::span<const uint64_t> released,
                                      std::span<const uint64_t> toWrite, bool writable,
                                      std::span<const uint64_t> staged, std::span<const uint64_t> unitsKept) const;
  mutable uint64_t leafVisits_ = 0;
};

} // namespace splash::engine
