#pragma once

#include "engine/KvCache.hpp"
#include "engine/KvPool.hpp"
#include "engine/KvTier.hpp"
#include "engine/StateCache.hpp"
#include "model/Model.hpp"
#include "model/SlotFile.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace splash::engine {

struct CacheLookup final {
  uint32_t kvBoundary = 0;
  std::optional<CompositeStateLease> state;
  // The matched KV ends at a branch point past the state: another branch
  // goes on below the deepest matched block and holds a state there. Zero
  // at a chain end or inside a dead tail, where this request's own deeper
  // states shadow any junction.
  uint32_t junctionBoundary = 0;
  // A matched block deeper than the state once held a reusable state that
  // is gone from both tiers.
  bool lostState = false;

  [[nodiscard]] uint32_t resumeBoundary() const noexcept {
    return state ? state->boundary() : 0;
  }
};

class Cache;

// A scheduling probe of one prompt and its images: refresh() and lookup()
// take the same spans it was made from.
class CacheProbe final {
public:
  [[nodiscard]] uint32_t cachedTokens() const noexcept { return cachedTokens_; }

private:
  friend class Cache;

  std::vector<uint64_t> blocks_;
  const Cache *owner_ = nullptr;
  uint64_t kvGeneration_ = 0;
  uint32_t cachedTokens_ = 0;
};

struct CacheLookupSnapshot final {
  // KV blocks scheduling probes hashed: admission work between commands.
  uint64_t probeHashedBlocks = 0;
  uint64_t kvHitTokens = 0;
  // Lookups whose state came from disk.
  uint64_t stateDiskHits = 0;
  // Lookups that matched KV past their state at a branch point.
  uint64_t lazyJunctions = 0;
  // Lookups that matched KV where a reusable state used to be.
  uint64_t lostStateMisses = 0;
};

struct KvTierSnapshot final {
  // The disk quota shared by KV pages and states, its current use, and the
  // bytes its files occupy on disk.
  uint64_t capacityBytes = 0;
  uint64_t usedBytes = 0;
  uint64_t fileBytes = 0;
  uint64_t readBytes = 0;
  uint64_t writtenBytes = 0;
  uint64_t demotions = 0;
  uint64_t demotionFailures = 0;
  // Demotions the tier had no room for; the leaf stayed and the requester
  // waited.
  uint64_t demotionsRefused = 0;
  uint64_t restores = 0;
  uint64_t restoreFailures = 0;
  // Pages whose demotion is in flight.
  uint32_t pendingPages = 0;
  uint32_t diskBlocks = 0;
  uint64_t diskBytes = 0;
};

struct CacheSnapshot final {
  KvPoolSnapshot pool;
  StateCacheSnapshot stateCache;
  KvTierSnapshot kvTier;
  CacheLookupSnapshot lookup;
  uint32_t activeRequests = 0;
  // The longest emptying of one extent by moving its pages, re-pointing the
  // cached blocks and requests on them included.
  double extentCompactMaxMilliseconds = 0.0;
};

enum class TokenAdmissionFailure : uint8_t {
  None,
  // The pool could not grow by an extent; allocationFailure says why.
  Denied,
  // A transfer in flight (a KV demotion, a KV restore or the one state write)
  // holds what the request needs; retry when it lands.
  Pending,
};

struct TokenAdmission final {
  TokenAdmissionFailure failure = TokenAdmissionFailure::None;
  uint32_t additionalPages = 0;
  uint32_t availablePages = 0;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;

  [[nodiscard]] bool granted() const noexcept {
    return failure == TokenAdmissionFailure::None;
  }
};

struct PageTableView final {
  std::span<const uint32_t> pages;
  uint64_t revision = 0;
  uint32_t firstChanged = 0;
};

struct CacheReclaimResult final {
  bool madeProgress = false;
  // Bytes of the KV extents the step released.
  uint64_t reclaimedBytes = 0;
  // Nothing was reclaimed, but a transfer in flight (a KV demotion, a KV
  // restore or the one state write) holds what the next reclaim needs.
  // Retry when it lands rather than treating the cache as empty.
  bool pending = false;
  // The step evicted a state from RAM. That frees nothing by itself: the
  // state's buffers return to the model's pool, which reclaimIdleState gives
  // back to the host.
  bool evictedState = false;
};

// What a reclaim step does with memory. KeepExtents: extents stay
// allocated, and the pages a victim frees are reused. ReleaseExtents: free
// pages return as extents first, and an extent a victim empties is
// released. ReusePages, while the host refuses growth: extents stay, and
// only what gives pages goes, KV leaves, and a state only where it sits on
// the leaf that goes next; its buffers would give no page otherwise.
enum class CacheReclaimMode { KeepExtents, ReleaseExtents, ReusePages };

// The highest class a reclaim step may take (the class rule). Ordinary:
// checkpoints, ordinary states and KV, for growth that can wait or yield
// instead: a start a resident lane holds back, and the lane that would
// yield first (Engine::laneToYield) while another lane is resident. InUse:
// after all of those, the states unfinished requests use and the KV they
// restore through, for other running work (any other resident lane, or a
// start no resident lane precedes) and for pressure passes.
enum class ReclaimClass : uint8_t { Ordinary, InUse };

// Whether the caller can wait for KV memory a step leaves to come back.
// CanWait (growth and pressure passes): a KV leaf a state needs is written
// first, and its page returns when the copy lands. Immediate (a snapshot
// between commands): only leaves whose page frees now go, through their disk
// copy or erased; a leaf that needs a demotion is passed over. What happens
// to a state is the caller's StateCache::Unwritten.
enum class ReclaimTiming : uint8_t { CanWait, Immediate };

// What one step of room for a state's snapshot gave: a recycled state's
// buffers, which the snapshot takes as they are, or the bytes of a released
// extent, which the snapshot has to allocate and which may be fewer than a
// state needs.
struct StateRoom final {
  bool made = false;
  uint64_t extentBytes = 0;
  explicit operator bool() const noexcept { return made; }
};

// KV restores a request waits for before it can run.
enum class KvRestoreStatus : uint8_t { None, Pending, Failed };

// A slot an earlier process left in a persistent tier's file: its label
// (KvBlockLabel or StateLabel), and how to take it back, which only what
// the cache keeps is.
struct PersistedKv final {
  std::vector<std::byte> label;
  std::function<std::shared_ptr<KvDiskSlot>()> adopt;
};
struct PersistedState final {
  std::vector<std::byte> label;
  // The disk copy of a state at that many tokens.
  std::function<std::shared_ptr<const CompositeState>(uint32_t tokens)> adopt;
};

// What a start took back from a persistent tier: the states and KV blocks
// the cache holds afterwards, and their bytes.
struct CacheAdoption final {
  uint32_t states = 0;
  uint32_t blocks = 0;
  uint64_t bytes = 0;
  // Records left behind, damaged, incomplete or of no state, and copies
  // given up for the quota.
  uint32_t dropped = 0;
};

// Owns active KV page leases, the content-addressed KV graph and cached
// composite states. Physical recurrent-state cells remain model-owned.
// A state in RAM always sits on a resident KV block: reclaim takes such a
// block's state before its page (oldestKvLeaf skips it, and a page scan
// takes the state a step before the leaf), and endRequest and pollTransfers
// leave such a block its page.
class Cache final {
public:
  // The disk budget is the quota the states' file shares with the KV tier;
  // the states' file can run on it without the tier. A persistent KV tier
  // makes the cache persistent: its copies are labelled for the next
  // process, and the quota keeps whole restore points by the recency of
  // their states (dropOldestPoint).
  Cache(KvPool &pool, KvTier *kvTier,
        std::shared_ptr<const model::DiskBudget> diskBudget);
  Cache(const Cache &) = delete;
  Cache &operator=(const Cache &) = delete;

  [[nodiscard]] bool persistent() const noexcept { return persistent_; }
  // Before the first request, takes back the restore points an earlier
  // process left in a persistent tier: each state whose block's chain is
  // whole, and that chain. Everything else stays behind, and once the files
  // finish adoption its slots are free. Then the oldest copies go while the
  // quota is exceeded. Ids and recency continue after those taken back.
  [[nodiscard]] CacheAdoption adopt(std::vector<PersistedKv> blocks,
                                    std::vector<PersistedState> states);

  void setCompletionNotifier(std::function<void()> notifier) {
    states_.setCompletionNotifier(notifier);
    completionNotifier_ = std::move(notifier);
  }
  // Consumes finished transfers: a written state or KV page frees its RAM, a
  // restored block becomes usable, and restores the tier had no room for
  // start.
  [[nodiscard]] bool pollTransfers();
  void discardState(uint64_t block, const CompositeState *state) {
    states_.discardState(block, state);
  }
  void beginRequest(uint64_t requestId);
  void endRequest(uint64_t requestId);

  // Scheduling probe only: does not pin, touch recency, or count a hit.
  [[nodiscard]] CacheProbe
  probe(std::span<const uint32_t> prompt,
        std::span<const ImageSpan> images);
  // Brings a probe this cache made up to date: after a change of the KV
  // graph its chain is cut at the first block that no longer matches and
  // matched on from there; its cached tokens follow the states either way.
  void refresh(CacheProbe &probe, std::span<const uint32_t> prompt,
               std::span<const ImageSpan> images);

  // Pin the usable prefix before potentially evicting for active allocations.
  // Accounting is separate: failed admission retries are not extra samples.
  // A probe of the prompt that has seen the KV graph as it is supplies the
  // matched chain; otherwise the prompt is matched again.
  [[nodiscard]] CacheLookup lookup(
      std::span<const uint32_t> prompt,
      std::span<const ImageSpan> images,
      const CacheProbe *probe = nullptr);
  void recordLookup(const CacheLookup &lookup);
  void promoteState(const CacheLookup &lookup, StateRestore &transfer);
  // Gives the request the matched chain up to its state. Disk-only blocks
  // get fresh pages and start their restores; the request waits on
  // kvRestoreStatus() before it runs. Pages are admitted like ensureTokens().
  [[nodiscard]] TokenAdmission restoreRequest(uint64_t requestId,
                                              const CacheLookup &lookup);
  [[nodiscard]] KvRestoreStatus kvRestoreStatus(uint64_t requestId) const;

  [[nodiscard]] TokenAdmission ensureTokens(uint64_t requestId,
                                            uint64_t tokenCount);
  // The request's page list. Its revision moves on every change of the
  // list, and the list differs from the one at revision - 1 from
  // firstChanged on.
  [[nodiscard]] PageTableView pageTable(uint64_t requestId) const;

  // Canonicalizes every newly complete Page32 block. Duplicate content swaps
  // the request to the existing immutable page after the writer command has
  // completed; no active command ever aliases a writable page.
  void publishCommittedBlocks(uint64_t requestId,
                              std::span<const uint32_t> exactTokens,
                              uint32_t committedTokens,
                              std::span<const ImageSpan> images);
  [[nodiscard]] uint64_t blockAt(uint64_t requestId, uint32_t boundary) const;
  [[nodiscard]] bool reuseCompositeState(uint64_t kvBlock, bool checkpoint);
  // Reuses the state at this block in either tier, as reuseCompositeState()
  // does a RAM copy.
  [[nodiscard]] bool reuseStoredState(uint64_t kvBlock, bool checkpoint = false);
  void publishCompositeState(uint64_t kvBlock,
                             std::shared_ptr<const CompositeState> state,
                             bool checkpoint);
  // Publishes the state of the lane at this block straight to disk, for a
  // state no cache slot can hold: `write` starts the write from the lane.
  // False when the tier cannot take the state now; nothing is published then.
  // The caller reuses a stored state first (reuseStoredState); publishing
  // over one is a logic error.
  [[nodiscard]] bool publishStateToDisk(uint64_t kvBlock, const StateWriter &write,
                                        bool checkpoint);
  // The request holding the handle is unfinished and its conversation
  // resumes from the state at this block: see StateCache::useState.
  [[nodiscard]] StateUse useState(uint64_t kvBlock) { return states_.useState(kvBlock); }
  [[nodiscard]] StateCheckpoint checkpointState(uint64_t kvBlock) const noexcept;
  // The state at this block has a RAM copy.
  [[nodiscard]] bool stateResident(uint64_t kvBlock) const noexcept;
  // False only while this exact disposable publication is pinned.
  bool retireCheckpointState(StateCheckpoint checkpoint) noexcept;

  // Releases every empty extent, but one with keepRunway, and returns the
  // bytes released.
  [[nodiscard]] uint64_t releaseEmptyExtents(bool keepRunway);
  // Evicts every unpinned entry, in reclaimOne()'s order, and releases every
  // empty extent, the runway too. Only then does it move pages, those that
  // requests and pins still hold, so that nothing is copied and then evicted.
  // Makes progress when anything goes, and reports the bytes of the extents
  // released.
  [[nodiscard]] CacheReclaimResult evictAll();
  // Bytes of pages whose demotion is in flight; they return when the copies
  // land.
  [[nodiscard]] uint64_t pendingBytes() const noexcept;
  // A KV demotion, a KV restore or the one state write is in flight, so
  // memory or quota returns by itself and its completion wakes the engine.
  [[nodiscard]] bool transfersInFlight() const noexcept;
  // One bounded reclaim step, for an allocation retry or a pressure pass: one
  // empty extent, one extent emptied of its pages, one state or one KV leaf,
  // so a denial or a pass frees only what it needs. With ReleaseExtents free
  // pages return first: an empty extent, then the free pages scattered over
  // the others once they cover the extent that holds the fewest pages, whose
  // pages move to them (compactExtent), and an extent a victim empties is
  // released. Only then is anything evicted: disposable checkpoints first,
  // then KV leaves no state restores through, then ordinary states and
  // resident KV leaves, which share one oldest-first access order. States
  // that unfinished requests use and the KV they restore through follow in
  // their own such order, once no transfer in flight can return what is
  // needed first. A chosen state keeps its disk copy when it has one, is
  // written when the tier admits it and dropped otherwise; its buffers return
  // to the model's pool, which reclaimIdleState gives back to the host. A
  // chosen KV leaf frees its page at once when a disk copy exists, is
  // dropped when nothing depends on it, and is otherwise written first: its
  // page returns when the copy has landed, which ensureTokens() reports as
  // Pending so callers wait instead of evicting more. While the tier can
  // start no demotion, the leaves that need one stay and the others still
  // go. A full disk quota replaces the oldest redundant copy of either kind,
  // then the oldest copy that is the only one. The only copies of states in
  // use, and the KV they restore through, make room only for a copy that is
  // itself in use. Active requests and pinned restores are never selected.
  // Progress is distinct from released bytes because evicting a KV
  // reference can make a page reusable without emptying its extent, and the
  // extent a step empties may be the runway it keeps. The step takes nothing
  // of a class above upTo: with Ordinary, once only what is in use is left,
  // it makes no progress, and reports pending while a transfer is in flight.
  // With ReusePages no checkpoint goes, and of each class the leaves go in
  // their own oldest-first order. keepResumePoint stops short of the resume
  // point, the newest ordinary publication (else checkpoint), and a page
  // scan of the leaf under it; states in use are protected by their class.
  // A shrink that no request is waiting for gains the one cell that
  // publication holds and costs the next request a replay of its whole
  // prompt, because a hybrid model cannot resume from cached KV without the
  // recurrent state. keepRunway leaves one empty extent allocated, for the
  // next request.
  [[nodiscard]] CacheReclaimResult reclaimOne(CacheReclaimMode mode, ReclaimClass upTo,
                                              bool keepResumePoint = false,
                                              bool keepRunway = false);
  // The reclaim step for a KV admission the pool denied: evicts in
  // reclaimOne(mode, upTo)'s order, mode being KeepExtents, or ReusePages
  // while the host refuses growth, until the free pages and the pages whose
  // demotion is in flight cover `pages` (the admission's additionalPages),
  // until, with KeepExtents, an evicted state has returned memory (the retry
  // may then grow the pool), or until nothing more of the class can go; one
  // step instead of a retry per page.
  [[nodiscard]] CacheReclaimResult reclaimForPages(uint32_t pages, CacheReclaimMode mode,
                                                   ReclaimClass upTo);
  // The most pages reclaimForPages(pages, mode, upTo) can leave free: every
  // allocated page no request holds, but with Ordinary not the idle KV that
  // states in use restore through.
  [[nodiscard]] uint32_t reusablePages(ReclaimClass upTo) const;
  // One step of room for the snapshot of a state to publish at forBlock,
  // taking nothing of a class above the publication's; the disk tier keeps a
  // state it admits. An optional publication (checkpointsOnly) takes one
  // checkpoint. An ordinary one takes what reclaimOne() takes of the
  // ordinary class, in its order, but only what frees memory now
  // (ReclaimTiming::Immediate), since the snapshot follows at once: an extent
  // of free pages, a checkpoint, KV no state restores through, then the
  // oldest of ordinary states and KV leaves. One in use goes on to the oldest
  // state in use, last. KV frees memory only as an extent it empties,
  // released before the call returns, so KV goes only while evicting it can
  // empty one (extentWithinReach); otherwise the oldest ordinary state goes.
  // Only a publication in use, whose lane's state moves on with its next
  // command, drops a victim whose write must wait for the write in flight;
  // other publications leave it and move on, and a state in use is never
  // dropped so: while the write in flight holds it, nothing goes. Like every
  // release, that needs no command in flight; the engine publishes a lane's
  // states between commands. After a step that gave extent bytes the caller
  // steps again while its snapshot does not fit and the bytes given are less
  // than one snapshot; after a step that gave a state it does not, so a
  // snapshot denied for another reason costs one state or one snapshot's
  // worth of extents at most. While the engine may not grow (`growth` false)
  // an extent's bytes are of no use to a snapshot: extents and KV stay, and
  // every publication takes states alone.
  [[nodiscard]] StateRoom reclaimOneState(bool checkpointsOnly, uint64_t forBlock, bool growth);
  // Recycles one unpinned state, preferring checkpoints, for a lane that
  // takes the state's buffers, taking nothing of a class above upTo. For
  // running work (InUse) a state in use goes once no other state is left and
  // no transfer in flight can return what is needed first; a start that a
  // resident lane holds back (Ordinary) takes nothing in use. A state the
  // tier could take once the write in flight has finished stays and is
  // reported pending, as in reclaimOne. evictableStates(upTo) are those it
  // can take.
  [[nodiscard]] CacheReclaimResult reclaimStateForLane(ReclaimClass upTo);
  [[nodiscard]] uint32_t evictableStates(ReclaimClass upTo) const noexcept {
    return states_.evictableStates(upTo == ReclaimClass::InUse);
  }
  [[nodiscard]] CacheSnapshot snapshot() const;

private:
  // Matches the prompt's pages on from blocks.back() (from the root when
  // empty) and appends each block found; returns the pages it hashed.
  size_t extendMatch(std::vector<uint64_t> &blocks, std::span<const uint32_t> prompt,
                     std::span<const ImageSpan> images) const;
  // The tokens up to the deepest of these blocks that holds a state.
  [[nodiscard]] uint32_t stateTokens(std::span<const uint64_t> blocks) const;

  struct Request final {
    std::vector<uint32_t> pages;
    std::vector<uint64_t> cachedBlocks;
    uint64_t pageTableRevision = 0;
    uint32_t firstChangedPage = 0;
    uint32_t pendingRestores = 0;
    bool restoreFailed = false;
  };
  // Records a change of the request's page list from index first on.
  static void pagesChanged(Request &active, uint32_t first) noexcept;
  struct Demotion final {
    uint64_t block = 0;
    std::unique_ptr<KvTransfer> transfer;
  };
  struct Restore final {
    // Null until the tier has room for it.
    std::unique_ptr<KvTransfer> transfer;
    std::vector<uint64_t> waiters;
  };

  [[nodiscard]] Request &request(uint64_t requestId);
  [[nodiscard]] const Request &request(uint64_t requestId) const;
  // Pages whose demotion is in flight; each keeps its page until the copy
  // lands.
  [[nodiscard]] uint32_t pendingPages() const noexcept {
    return static_cast<uint32_t>(demotions_.size());
  }
  // Pending, in every verdict below and in the results above, means the
  // same thing: a transfer in flight holds what this needs, and it comes
  // back when the transfer lands. Only transfersInFlight() may report it:
  // a caller told to wait for nothing would wait for ever.
  enum class LeafReclaim : uint8_t {
    Started,
    // The tier's room, the quota or the state write's staging buffer is held
    // by transfers in flight.
    Pending,
    // reclaimKvLeaf: the leaf stays for now and scans move on to the next
    // one, because a state on it is pinned, a disk subtree depends on it
    // and the tier has no room that a transfer in flight will free, a disk
    // subtree below it cannot drop yet (active, in transfer, or a state
    // write in flight), or the leaf needs a demotion an Immediate caller
    // cannot wait for. demoteKv: the leaf cannot be written, and may
    // go without a copy, because the tier takes no writes, has no such
    // room, or making room took the states the leaf was kept for.
    Impossible,
  };

  // What one reclaim step gave up: a state, whose buffers return to the
  // model's pool, or a KV leaf, whose memory returns with the extent it
  // leaves empty.
  struct Victim final {
    bool kv = false;
  };

  [[nodiscard]] TokenAdmission admitPages(uint32_t count,
                                          std::vector<uint32_t> &pages);
  // One eviction: checkpoints first, then the ordinary class's
  // reclaimOldest, then, up to InUse, that of what is in use. pagesOnly
  // (CacheReclaimMode::ReusePages) skips the checkpoints and scans for pages.
  [[nodiscard]] CacheReclaimResult evictOne(ReclaimClass upTo, bool keepResumePoint,
                                            bool pagesOnly);
  // One reclaimOldest scan: the class it takes (what is in use, or the
  // ordinary class), whether it keeps the resume point, what becomes of a
  // state whose write cannot start now, how long the caller can wait for KV
  // memory, and whether only what gives pages goes.
  struct VictimScan final {
    bool inUse;
    bool keepResumePoint;
    StateCache::Unwritten unwritten;
    ReclaimTiming timing;
    bool pagesOnly;
  };
  // Gives up one victim of the scan's class. The ordinary class first gives
  // the oldest KV leaf no state restores through (oldestDeadKvLeaf). Then
  // the oldest of the class's states and resident KV leaves goes: a state as
  // `unwritten` says (StateCache::reclaim), a KV leaf as the timing allows
  // (reclaimKvLeaf). One that stays leaves the other kind to give, and once
  // a demotion has to wait for the tier, no other is started: leaves that
  // need one stay. A page scan (pagesOnly) takes the class's leaves alone,
  // in their own order, and a state only where it sits in RAM on the leaf
  // that goes next: that state goes, the leaf on a later step; a resume
  // point it keeps stays with its leaf. Nothing once all stay.
  [[nodiscard]] std::optional<Victim> reclaimOldest(const VictimScan &scan);
  // Oldest resident KV leaf after `after` no state restores through: nothing
  // sits on it or below it, so it saves no prefill, and it frees its page
  // with no IO.
  [[nodiscard]] std::optional<CacheEvictionCandidate> oldestDeadKvLeaf(uint64_t after) const;
  // Oldest resident KV leaf after `after` whose KV a state in use needs
  // exactly when inUse, and whose state, if any, is not in RAM unless
  // withRamState.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  oldestKvLeaf(uint64_t after, bool inUse, bool withRamState) const;
  // Frees the RAM of one resident KV leaf: through its disk copy when it has
  // one, by demotion when a state on it or below it depends on it, by
  // erasure otherwise, with any disk copies below it. Pending when the tier
  // cannot take it right now. A leaf that a disk subtree depends on is
  // dropped only once a failed write has closed the tier, together with
  // that subtree. Immediate passes over a leaf that needs a demotion.
  [[nodiscard]] LeafReclaim reclaimKvLeaf(uint64_t block, ReclaimTiming timing);
  [[nodiscard]] LeafReclaim demoteKv(uint64_t block);
  // A failed write closes the tier; existing copies stay readable.
  [[nodiscard]] bool kvTierWritable() const noexcept { return tier_ && tier_->writable(); }
  // Only a state restores a disk-only chain, through its own block and every
  // block above.
  [[nodiscard]] bool kvNeededByState(uint64_t block) const {
    return states_.contains(block) || kv_.stateBelow(block);
  }
  // The same for a state in use: such KV is in use too.
  [[nodiscard]] bool kvNeededByStateInUse(uint64_t block) const {
    return (states_.inUse(block) && states_.contains(block)) || kv_.stateInUseBelow(block);
  }
  // Erases the disk-only subtree below a resident leaf and the states on it,
  // in use or not; false, erasing nothing, while a block of it is in transfer
  // or active (a lookup holding a state keeps its block active) or a state
  // write is in flight.
  [[nodiscard]] bool dropDiskSubtree(uint64_t block);
  // Nothing below a block whose read failed matches any more. Once none of
  // it is in transfer or active, it is erased with the states on it, in use
  // or not, and the poisoned block leaves with its last user.
  void dropPoisoned();
  // A slot for a new KV copy, replacing older copies while the quota is full;
  // inUse when a state in use needs the copy.
  [[nodiscard]] std::shared_ptr<KvDiskSlot> acquireDiskSlot(bool inUse);
  // Gives up one disk copy for a new copy, in use or not: the oldest
  // redundant one, KV or state, else the oldest that is the only copy, never
  // the KV of a state in RAM. A persistent tier gives up restore points
  // instead (dropOldestPoint). The only copy of a state in use, and the KV
  // it restores through, go only for a copy in use, and after every other.
  // False when the disk holds nothing the new copy may displace.
  [[nodiscard]] bool freeDiskSpace(bool inUse);
  // In a persistent tier a redundant copy is what the next process takes
  // back, and the unit worth keeping is a restore point, a state and its
  // chain. A copy no state needs goes first (dropUnneededCopy), then the
  // oldest state copy, redundant or not: the copies of its chain that no
  // other state needs become unneeded and go on later calls, so a chain
  // stays whole while a state needs it. The only copy of a state in use goes
  // only for a copy in use, and last. False when no state copy can go.
  [[nodiscard]] bool dropOldestPoint(bool inUse);
  [[nodiscard]] bool dropUnneededCopy();
  // A persistent tier's label for the block's disk copy.
  void labelKv(uint64_t block, const std::shared_ptr<KvDiskSlot> &slot);
  void startRestore(uint64_t block);
  // Evicting ordinary KV may empty an extent: the pages Ordinary may reuse
  // (reusablePages), less those whose demotion is in flight, cover one. That
  // counts the leaves that need a demotion too, which an Immediate step
  // passes over while the tier takes writes.
  [[nodiscard]] bool extentWithinReach() const;
  // Returns one extent of free pages: an empty one, or the one compactExtent
  // empties; progress with zero bytes when the extent emptied is the runway
  // kept. No progress when the free pages fill none.
  [[nodiscard]] CacheReclaimResult releaseExtent(bool keepRunway);
  // Empties one extent that still holds pages (KvPool::compactExtent); the
  // blocks and requests on its pages follow them. A page a transfer reads or
  // writes stays where it is until the transfer has landed. False when the
  // free pages cover no extent.
  [[nodiscard]] bool compactExtent();

  KvPool &pool_;
  KvTier *tier_;
  bool persistent_;
  std::shared_ptr<const model::DiskBudget> diskBudget_;
  CacheRecency recency_;
  KvCache kv_;
  StateCache states_;
  std::unordered_map<uint64_t, Request> requests_;
  std::vector<Demotion> demotions_;
  // Block IDs increase from parent to child. Restores start in that order so
  // cancellation can discard an unread suffix without stranding its parents.
  std::map<uint64_t, Restore> restores_;
  // Blocks whose read failed, until they have left.
  std::vector<uint64_t> poisoned_;
  KvTierSnapshot kvTier_;
  CacheLookupSnapshot lookup_;
  double extentCompactMaxMilliseconds_ = 0.0;
  std::function<void()> completionNotifier_;
};

} // namespace splash::engine
