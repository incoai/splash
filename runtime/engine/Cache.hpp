#pragma once

#include "AwakeClock.hpp"
#include "engine/KvCache.hpp"
#include "engine/KvPool.hpp"
#include "engine/KvTier.hpp"
#include "engine/MemoryLedger.hpp"
#include "engine/ReclaimClass.hpp"
#include "engine/StateCache.hpp"
#include "engine/Types.hpp"
#include "model/LatencyHistogram.hpp"
#include "model/Model.hpp"
#include "model/SlotFile.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace splash::engine {

// What a prompt's KV blocks are cached under beside its tokens: the images
// its rows depend on and its request's cache salt (blockIdentity).
struct PromptIdentity final {
  std::span<const ImageSpan> images;
  CacheSalt salt;
};

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
  // A matched block is one a reply's decode published (KvCache::noteReply):
  // the prompt renders an earlier reply back, as far as that block at least.
  bool rendersReply = false;

  [[nodiscard]] uint32_t resumeBoundary() const noexcept {
    return state ? state->boundary() : 0;
  }
};

class Cache;

// A scheduling probe of one prompt: refresh() and lookup() take the same
// identity it was made from.
class CacheProbe final {
public:
  [[nodiscard]] uint32_t cachedTokens() const noexcept { return cachedTokens_; }
  // The chain a start from it restores, as far as its cache still has it
  // (matchableChain): its blocks up to the one whose state it resumes from,
  // none without a state. Every reader reads a probe's chain here, so none
  // meets a block a reclaim took ("unknown KV cache block").
  [[nodiscard]] std::span<const uint64_t> restoredChain() const noexcept;

private:
  friend class Cache;

  // The part of `chain` its cache still has: all of it while the KV graph is
  // as the probe last matched it, else up to its first block no longer
  // matchable (KvCache's generation). A start backing off from a refusal
  // keeps its probe while a reclaim may take its chain, leaf first.
  [[nodiscard]] std::span<const uint64_t> matchableChain(std::span<const uint64_t> chain) const noexcept;
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

// What holds the disk tier's writes back: nothing, the write budget (while
// it is spent the cache writes nothing; only when one is set), or a file
// that stopped taking writes.
enum class DiskWritesGate : uint8_t { No, Budget, Closed };

struct KvTierSnapshot final {
  // The disk quota shared by KV pages and states, the limit its volume leaves
  // it now, its current use, and the bytes its files occupy on disk.
  uint64_t capacityBytes = 0;
  uint64_t limitBytes = 0;
  uint64_t usedBytes = 0;
  uint64_t fileBytes = 0;
  // What the volume had available when the limit was last set.
  uint64_t volumeAvailableBytes = 0;
  DiskWritesGate writesGated = DiskWritesGate::No;
  // What the cache's writes may take now; zero once overdrawn, and zero
  // without a write budget, when nothing is held back.
  uint64_t writeBudgetBytes = 0;
  // How long the files' writes held back for reads a step waited for
  // (model::DiskBudget::writesDeferredMicroseconds).
  uint64_t writesDeferredMicroseconds = 0;
  uint64_t readBytes = 0;
  uint64_t writtenBytes = 0;
  // What the tier's writes took over the last hour and the last day
  // (model::DiskBudget::writeRate).
  model::DiskBudget::WriteRate writeRate;
  uint64_t demotions = 0;
  uint64_t demotionFailures = 0;
  // Demotions the tier had no room for; the leaf stayed and the requester
  // waited.
  uint64_t demotionsRefused = 0;
  uint64_t restores = 0;
  uint64_t restoreFailures = 0;
  // KV pages written while they stayed in RAM, ahead of need (persist).
  uint64_t copies = 0;
  uint64_t copyFailures = 0;
  // Full pages transfers in flight will free (Cache::pendingPages).
  uint32_t pendingPages = 0;
  uint32_t diskBlocks = 0;
  uint64_t diskBytes = 0;
  // Why a tier the start asked for is off (its "SSD cache off" line); empty otherwise.
  std::string offReason;
  // What demotions and copies ahead wrote, the parts each wrote, of them the
  // window parts, and the pages written again after their copy was given up
  // while they stayed in RAM (copiesGivenUp_).
  uint64_t demotedBytes = 0;
  uint64_t copiedBytes = 0;
  uint64_t windowPartBytes = 0;
  uint64_t rewrites = 0;
  uint64_t rewrittenBytes = 0;
  // What the quota gave up for room, by the kind of copy (freeDiskSpace, in
  // Cache::DiskVictim's order): window parts, KV copies, state copies, only
  // copies of states, and of states in use.
  std::array<uint64_t, 5> quotaGivenUp{};
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

// What spilling holds and did (/status kv_spill). Spilled blocks, whose
// pages hold only their index, and their copies' bytes; write-through of
// Spill requests' pages: pages and bytes written, writes in flight and
// failed, and a pause after the volume filled; pages that moved to the index
// form and back, the releases that gave a lane's pages' memory back for a
// claim and their pages, and blocks written again once their copy had gone;
// restores of a page's index and of a whole page, and how long they took;
// the quota of active KV, the copies of the KV requests hold and the
// commitments of Spill requests within it, and commitments revoked.
struct KvSpillSnapshot final {
  uint32_t spilledPages = 0;
  uint64_t spilledBytes = 0;
  uint64_t writeThroughPages = 0;
  uint64_t writeThroughBytes = 0;
  uint32_t writeThroughInFlight = 0;
  uint64_t writeThroughFailures = 0;
  bool writeThroughPaused = false;
  uint64_t spillMoves = 0;
  uint64_t unspills = 0;
  uint64_t releases = 0;
  uint64_t releasedPages = 0;
  uint64_t rewrites = 0;
  uint64_t indexRestores = 0;
  uint64_t wholeRestores = 0;
  model::LatencyStats restore;
  uint64_t activeLimitBytes = 0;
  uint64_t heldBytes = 0;
  uint64_t committedBytes = 0;
  uint64_t revokedCommitments = 0;
};

// What a cache of a windowed layout holds and did (/status kv.window): the
// pages a row's window and a state's tail read, the blocks a state's tail
// holds and those that hold a window unit, the blocks that took the page of
// a writer that computed them again (KvCache::upgrades), and the states
// refused because a block of their tail held no unit (stateTailPresent).
struct KvWindowSnapshot final {
  uint32_t windowPages = 0;
  uint32_t pinnedBlocks = 0;
  uint32_t windowUnits = 0;
  uint64_t upgrades = 0;
  uint64_t refusedStates = 0;
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
  // A persistent tier, and what the start took back from it.
  bool persistent = false;
  CacheAdoption adoption;
  // A windowed layout's; none for another.
  std::optional<KvWindowSnapshot> window;
};

// Whether an admission of pages was refused. A refusal says no more than
// that: what the pages it lacks wait for, or take, the memory ledger
// decides (MemoryLedger::free), from what lands and what the walk reaches.
enum class TokenAdmissionFailure : uint8_t {
  None,
  // The pool could not grow by an extent; allocationFailure says why.
  Denied,
};

struct TokenAdmission final {
  TokenAdmissionFailure failure = TokenAdmissionFailure::None;
  uint32_t additionalPages = 0;
  uint32_t availablePages = 0;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;
  // The form of the pages refused, which a page need makes room for.
  kv::PageForm form = kv::PageForm::Full;
  // The free pages of each form the attempt takes, which the room made for
  // it keeps (Need::keptPages): a restore's (restoreRequest).
  std::array<uint32_t, kv::kPageForms> keptPages{};

  [[nodiscard]] bool granted() const noexcept {
    return failure == TokenAdmissionFailure::None;
  }
};

// The page list a command of a request reads (Cache::cutPageTable): its
// blocks' base pages and, with a window, a table per page of their window
// units, table j holding by logical page the j-th page of each block's, and
// kv::kAbsentPage for a block its window no longer reaches.
struct PageTableView final {
  std::span<const uint32_t> pages;
  std::span<const std::vector<uint32_t>> windows;
  // Where each page reads its K/V (ModelBatchItem::pageSources); null while
  // none holds only its index.
  std::shared_ptr<const model::KvPageSources> sources;
  uint64_t revision = 0;
  uint32_t firstChanged = 0;
};

// A request's page list as it is now, with the revision and first change its
// next cut gives it (Cache::pageTable): what the engine and tests inspect,
// never what a command reads.
struct PageTableState final {
  std::span<const uint32_t> pages;
  std::span<const std::vector<uint32_t>> windows;
  uint64_t revision = 0;
  uint32_t firstChanged = 0;
};

// What a spill-capable target's cache keeps Full for each request that
// spills: the pages its rows read through page tables from the start (its
// dense history, SparseKvShape), and the last pages it committed before its
// open page (the tail window, at least its local rows'). A Spill request's
// other pages stay Full once their copies land, until a claim releases them
// (Cache::releasePages): a lane decodes on its pages in memory while memory
// lasts, and reads back only the blocks a release took. With `always`
// (--kv-spill always, for validation) they move to the index form as their
// copies land (Cache::spillWritten).
struct CacheSpillPolicy final {
  uint32_t densePages = 0;
  uint32_t windowPages = 0;
  bool always = false;
};

// What the cache of a layout with a sliding window (kv::Layout::windowTokens)
// counts: the tail of each state, the windowPages blocks (kv::Layout::
// windowPages) that end at its block, which a request that resumes there
// reads (KvCache::countState); and what a block holds beside its base page,
// its window unit of unitPages pages (kv::Layout::windowUnitPages), while a
// state's tail holds it, and a lane while its window reaches the block.
struct CacheWindowPolicy final {
  uint32_t windowPages = 0;
  uint32_t unitPages = 0;
};

// Whether a request may start to spill now, or what holds it back: the KV
// tier takes no writes (it is off, or a file stopped taking them), or the
// write budget does not cover the writes Spill requests have committed.
enum class SpillGate : uint8_t { Open, TierClosed, WriteBudget };

// What becomes of the Full extents a spill's moves leave empty
// (Cache::spillWritten): they leave memory but one, the runway the next
// Full pages come from without an allocation; or the Full pages left
// gather in the fewest extents (Cache::compactExtent) and every empty one
// leaves.
enum class SpilledExtents : uint8_t { KeepRunway, Released };

// Whether a walk can wait for memory its steps' writes return (Walk::timing):
// a claim that waits for writes (TakeRights::waitsForWrites) walks CanWait,
// any other Immediate, and so do the phases of a device or a page need's
// reclaims but those that free as writes land (Cache::phaseWalk). CanWait: a
// KV leaf a state needs is written first, and its page returns when the
// copy lands; a state that needs a write waits for the staging buffer while
// the one write in flight holds it. Immediate: only what frees now goes. A
// leaf goes through its disk copy or erased; one that needs a demotion is
// passed over. A state that needs a write goes only through the idle
// staging buffer, whose copy frees its RAM at once; while a write holds the
// buffer it is passed over.
enum class ReclaimTiming : uint8_t { CanWait, Immediate };

// The class of a state a command captures, which bounds what is taken for
// its buffers: an optional checkpoint, an ordinary state, or the prompt's
// replay point, which its conversation's next turn resumes from and is in
// use.
enum class StateClass : uint8_t { Checkpoint, Ordinary, InUse };

// The chains requests that hold no lane would restore once they start
// (Cache::setWantedChains): of each, the KV of every block and the state at
// its last, which the request resumes from (CacheProbe::restoredChain).
class RestoredChains final {
public:
  void add(std::span<const uint64_t> chain) {
    if (chain.empty())
      return;
    blocks_.insert(chain.begin(), chain.end());
    states_.push_back(chain.back());
  }
  [[nodiscard]] bool holds(uint64_t block) const { return blocks_.contains(block); }
  // The blocks of the states the chains resume from.
  [[nodiscard]] std::span<const uint64_t> states() const noexcept { return states_; }

private:
  std::unordered_set<uint64_t> blocks_;
  std::vector<uint64_t> states_;
};

// The pages a release may move (Cache::releasePages): requests on one prefix
// share its pages, so only those whose every holder is a request `takesFrom`
// takes pages from, and none on the chain a Resident start restores (`kept`,
// by position), which its count holds as its own in place.
struct ReleaseRule final {
  std::function<bool(uint64_t holder)> takesFrom;
  std::span<const uint64_t> kept;
};

// A count of releases of requests in turn (Engine::EngineVictims::releases),
// each moving its request's oldest pages as its take would: the pages those
// it counted move and the requests they give Index pages, so that a page
// requests on one prefix share, and a holder's first reads, count once.
// Counted for each alone, two lanes on one conversation gave 307 MB more
// than their takes free. Also the Full extents they empty, and the free Full
// pages beside them (Cache::releaseBeside), with the pages their moves free.
struct ReleaseCount final {
  std::unordered_set<uint64_t> blocks;
  std::unordered_set<uint64_t> holders;
  uint32_t extents = 0;
  uint32_t beside = 0;
};

// KV restores a request waits for before it can run.
enum class KvRestoreStatus : uint8_t { None, Pending, Failed };

// A slot an earlier process left in a persistent tier's file: its label
// (KvSlotLabel or StateLabel), and how to take it back, which only what
// the cache keeps is. A KV slot is a part of a block's copy (kv::Component),
// taken back into the copy that holds the block's other parts, or a new one.
struct PersistedKv final {
  std::vector<std::byte> label;
  std::function<std::shared_ptr<KvDiskSlot>(std::shared_ptr<KvDiskSlot> into)> adopt;
  kv::Component part = kv::Component::Base;
};
struct PersistedState final {
  std::vector<std::byte> label;
  // The disk copy of a state at that many tokens.
  std::function<std::shared_ptr<const CompositeState>(uint32_t tokens)> adopt;
};

// What a write of a point's copy is worth in the SSD order
// (Cache::freeDiskSpace's ranks): history no request uses rank 1, any other
// point rank 2 (Cache::writeWorth); and its block's last use.
struct DiskWorth final {
  uint8_t rank = 2;
  uint64_t lastUse = std::numeric_limits<uint64_t>::max();
};

// Owns active KV page leases, the content-addressed KV graph and cached
// composite states. Physical recurrent-state cells remain model-owned.
// A state in RAM always sits on a resident KV block: reclaim takes such a
// block's state before its page (a walk's leaves leave it out, and a page
// scan takes the state a step before the leaf: nextUnit), and endRequest and
// pollTransfers leave such a block its page.
class Cache final : public VictimSource {
public:
  // The disk budget is the quota the states' file shares with the KV tier;
  // the states' file can run on it without the tier. A persistent KV tier
  // makes the cache persistent: its copies are labelled for the next
  // process, and the quota keeps whole restore points by the recency of
  // their states (freeDiskSpace). A windowed layout's cache counts its
  // states' tails (CacheWindowPolicy). A copy has a part for each of the
  // tier's components, and all of them are made, written, counted, given up,
  // read and taken back together: a block's copy is whole or none.
  // With a spill policy, on a layout that can spill (kv::Layout::spillable)
  // and with a tier, requests may spill (KvMode::Spill), and the blocks
  // requests hold keep their copies (KvCache::holdPage).
  Cache(KvPool &pool, KvTier *kvTier, std::shared_ptr<model::DiskBudget> diskBudget,
        std::optional<CacheSpillPolicy> spill = std::nullopt,
        std::optional<CacheWindowPolicy> window = std::nullopt);
  Cache(const Cache &) = delete;
  Cache &operator=(const Cache &) = delete;

  [[nodiscard]] bool persistent() const noexcept { return persistent_; }
  // A file that stopped taking writes closes the KV tier; existing copies
  // stay readable.
  [[nodiscard]] bool kvTierWritable() const noexcept { return tier_ && tier_->writable(); }
  // The tier takes any cache write now: writable, its write budget not spent
  // (model::DiskBudget::admits). Else the cache keeps to memory as without a
  // tier, reading its copies; only a Spill request's writes pass.
  [[nodiscard]] bool takesCacheWrites() const noexcept {
    return kvTierWritable() && (!diskBudget_ || diskBudget_->admits());
  }
  // Before the first request, takes back the restore points an earlier
  // process left in a persistent tier: each state whose block's chain is
  // whole, and that chain. Everything else stays behind, and once the files
  // finish adoption its slots are free. A block comes back only with a
  // part of each component, paired by its id, identity and copy serial
  // (KvSlotLabel); a part alone stays behind with what lies below it.
  // Then copies go while
  // the disk limit is exceeded, as keepDiskLimit() gives them up. Ids
  // and recency continue after those taken back.
  [[nodiscard]] CacheAdoption adopt(std::vector<PersistedKv> blocks,
                                    std::vector<PersistedState> states);
  // One step of giving the restore point at this block its copies on the
  // disk tier while it stays in RAM, ahead of need: a reclaim then frees its
  // memory at once, and a persistent tier's next process takes it back. The
  // tier is busy while it takes no cache writes (takesCacheWrites); then the
  // pages of its chain that have no disk copy are written, root first,
  // beside the commands that may read them, and the state last, from its own
  // buffers, once every copy of the chain is under way: a point refused for
  // room leaves no state copy. Nothing starts while the tier serves
  // what is needed now, a restore that waits, a reclaim's demotion or a
  // Spill request's write through, and the pages take a small share of the
  // tier's writes. A temporary tier's point is unneeded unless still the
  // newest of its conversation (newestPoint), a persistent tier's once no
  // ordinary state, its copies serving the points after it. Every
  // point is refused once a file stopped taking writes, and one larger than
  // the room it may take (pointFits) or given up to make room for itself.
  [[nodiscard]] PersistStatus persist(uint64_t block);
  // The deepest point above this block on its chain that a restart takes
  // back (adopt) once the writes in flight land: a state on the SSD each
  // block of whose chain has a copy, landed or in flight, or the point
  // persist() copies; 0 for none. Walks the chain. It reads no window part:
  // a copy whose window part failed keeps its base part, and a state a
  // reclaim wrote may hold its tail's window K/V as units until they are
  // written, and either counts. `landed` counts only points that have
  // landed whole, window parts too (pointLanded): what a crash now would
  // leave.
  [[nodiscard]] uint64_t deepestTakenBack(uint64_t block, bool landed = false) const;
  // The block's copy, or with `state` its state's, has landed: a copy of
  // what its pages hold, or a disk-only block's of what a tail of it reads.
  [[nodiscard]] bool copyLanded(uint64_t block, bool state) const;
  // The point at the block has landed whole, its three parts: its state's
  // copy, the base part of each block of its chain, the window part of each
  // block of its tail. The one predicate of a persistent tier's durability.
  [[nodiscard]] bool pointLanded(uint64_t block) const;
  // The window part of each block of the state's tail has landed.
  [[nodiscard]] bool tailLanded(uint64_t block) const;
  // Whether a point deepestTakenBack() gave still is, as far as its own
  // copies tell: the point persist() copies, or a block whose own copy and
  // state's copy stand; its tail's window parts stay while its state's copy
  // needs them (KvCache::copyPins). A copy above it may go while those stay,
  // given up as a copy of what memory holds (freeDiskSpace) or lost with a
  // write that failed: takeBrokenCopies() names those blocks, and a point
  // whose chain holds one is found again by a walk.
  [[nodiscard]] bool stillTakenBack(uint64_t block) const {
    return block == persisting_.block || (kv_.contains(block) && kv_.ownCopy(block) && states_.onDisk(block));
  }
  // The blocks whose copy a persistent tier gave up or lost while a state
  // lay at or below them, since the last call, oldest first
  // (stillTakenBack).
  [[nodiscard]] std::vector<uint64_t> takeBrokenCopies() noexcept { return std::exchange(brokenCopies_, {}); }
  // The blocks of the chain ending at this block, root first; none once it
  // is gone.
  [[nodiscard]] std::vector<uint64_t> chainBlocks(uint64_t block) const {
    return kv_.contains(block) ? kv_.chain(block).blocks : std::vector<uint64_t>{};
  }
  // Pages of the chain ending at this block; 0 once it is gone.
  [[nodiscard]] uint32_t chainLength(uint64_t block) const {
    return kv_.contains(block) ? kv_.chainLength(block) : 0;
  }
  // The block holds a restore point a conversation still resumes from: an
  // ordinary state nothing superseded. persist() copies only such a point.
  [[nodiscard]] bool newestPoint(uint64_t block) const;
  // What the copies ahead of need in flight (persist) let a reclaim free at
  // once after they land: the bytes of the pages they copy and of the RAM
  // copy of the state they write.
  [[nodiscard]] uint64_t copyingAheadBytes() const noexcept;
  // At a clean stop: the KV copies no state needs go, as a full quota gives
  // them up first (freeDiskSpace), since the next process takes back only
  // restore points (adopt), each at once, its punch queued, and none past
  // `end`; each state copy's label takes its state's current recency, which
  // the next process orders them by.
  void closePersistentTier(AwakeClock::time_point end = AwakeClock::time_point::max()) {
    while (AwakeClock::now() < end && (kv_.unneededCopy() || unneededWindow()))
      static_cast<void>(freeDiskSpace(false, false));
    states_.relabel();
  }
  // Measures the volume the disk tier lives on and sets the budget's limit
  // (DiskSpace::measure), for keepDiskLimit(); none without such a tier.
  void setDiskMeasure(std::function<void()> measure) { diskMeasure_ = std::move(measure); }
  void setDiskOff(std::string reason) { kvTier_.offReason = std::move(reason); }
  // At a control pass, and when a write found the volume full: measures it
  // again, then gives up copies while the tier holds more than the limit,
  // in the order a full quota replaces them (freeDiskSpace), never a copy in
  // transfer or the only copy of a state in use.
  void keepDiskLimit();
  // When the write budget admits cache writes again, on its clock; empty
  // while it does now or without a budget.
  [[nodiscard]] std::optional<double> diskWritesAdmittedAt() const noexcept {
    return diskBudget_ ? diskBudget_->admitsAgainAt() : std::nullopt;
  }
  // When the write budget covers what Spill requests committed again, so
  // that a request may start to spill (spillGate), on its clock; empty
  // while it does now or without a budget.
  [[nodiscard]] std::optional<double> spillWritesAdmittedAt() const noexcept {
    return diskBudget_ ? diskBudget_->admitsSpillAt(diskBudget_->committedBytes()) : std::nullopt;
  }
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
  // The state at this block, in either tier, matches nothing any more and
  // leaves once nothing uses it.
  void invalidateState(uint64_t block) noexcept { states_.invalidate(block); }
  // A request begins in its mode; endRequest returns what remains of its
  // commitment, as suspension does by ending it.
  void beginRequest(uint64_t requestId, KvMode mode = KvMode::Resident);
  void endRequest(uint64_t requestId);

  // Spilling. Whether a new Spill request may start, or a Resident one
  // convert: a writable tier, and a write budget that covers what Spill
  // requests committed (DiskBudget::admitsSpill).
  [[nodiscard]] SpillGate spillGate() const noexcept {
    if (!kvTierWritable())
      return SpillGate::TierClosed;
    if (diskBudget_ && !diskBudget_->admitsSpill(diskBudget_->committedBytes()))
      return SpillGate::WriteBudget;
    return SpillGate::Open;
  }
  // The quota of active KV, the copies of the KV requests hold and what
  // Spill requests committed (DiskBudget::activeLimitBytes); none without
  // spilling. What it leaves for new commitments, without giving any copy
  // up: activeLimit - held copies - committed.
  [[nodiscard]] uint64_t spillLimit() const noexcept {
    return spill_ ? diskBudget_->activeLimitBytes() : 0;
  }
  [[nodiscard]] uint64_t spillRoom() const noexcept;
  // The blocks on the chain that ends at this block (none for 0) whose K/V
  // has a copy, landed or being written (copied): pages of a Spill request
  // that holds them need no commitment. Of those, the ones no request holds
  // (unheld, takesQuota): a start that restores the chain holds them, and
  // the quota of active KV counts them then (heldSlotBytes); the others it
  // counts already.
  struct ChainCopies final {
    uint32_t copied;
    uint32_t unheld;
  };
  [[nodiscard]] ChainCopies chainCopies(uint64_t block) const;
  // The tokens of the chain that ends at this block (none for 0) whose K/V
  // no page in RAM holds: what a restore of it reads from the SSD cache.
  [[nodiscard]] uint32_t diskOnlyTokens(uint64_t block) const;
  // Commits quota for `pages` pages of a Spill request, which each page it
  // publishes then draws or returns (DiskBudget::commit); false, committing
  // nothing, when spillRoom() cannot hold them beside the unheld copies of
  // the chain it restores, which it holds once it does (`copies`). A copy
  // written for a block of that chain before the restore holds it returns
  // that block's page (restoreRequest). A tier that takes no writes is the
  // spill gate's to refuse: a resume commits there too, and since no write
  // draws its commitment, all of it goes back as the request ends.
  [[nodiscard]] bool commitSpill(uint64_t requestId, uint32_t pages, const ChainCopies &copies = {});
  // Turns a Resident request whose growth was denied into a Spill request:
  // checks the tier, the write budget and the room for `pages` pages,
  // commits them and starts write-through of its published blocks; its
  // pages whose copies landed, as a chain it restored has them, are its
  // claim's to release (releasePages). It allocates nothing; false, changing
  // nothing, when spilling cannot relieve it.
  [[nodiscard]] bool convertToSpill(uint64_t requestId, uint32_t pages);
  // Whether a start that restores `chain` (CacheProbe::restoredChain) would
  // take some of the quota of active KV (spillRoom): a copy on it no
  // request holds (takesQuota).
  [[nodiscard]] bool restoreTakesQuota(std::span<const uint64_t> chain) const;
  // Whether the request holds some of the quota of active KV by its own
  // count: a Spill request's commitment, or a block of its chain it holds
  // with a copy (Request::heldCopies), even one another request holds too,
  // whose copy the quota counts once.
  [[nodiscard]] bool holdsActiveKvQuota(uint64_t requestId) const;
  // Starts writing Spill requests' published pages that have no copy of
  // what they hold (KvCache::kvCopied), as the tier's share for them allows;
  // returns the writes started.
  uint32_t writeThrough();
  // Between commands: refills the Index runway, an extent's worth of free
  // Index pages, which the next spill takes without an allocation. Under
  // --kv-spill always (CacheSpillPolicy::always) it first moves the eligible
  // pages whose copies landed to the index form, a runway at a time: into
  // free Index pages, the runway's among them; then the Full extents those
  // moves left empty go as `extents` says, their bytes told to `funds`
  // (MemoryGovernor::fund), and the runway refills from them. Denied, of the
  // index form, when an allocation was refused: the pages it leaves Full, or
  // the runway's; the caller makes room for them or does without.
  [[nodiscard]] TokenAdmission spillWritten(SpilledExtents extents,
                                            const std::function<void(uint64_t bytes)> &funds = {});
  // The free Full pages compaction gathers with a release's moves for the
  // claim, which complete an extent that the moves alone leave short: those
  // of the extents that hold pages (an empty extent leaves whole), less those
  // its walk keeps for it (Walk::keptPages). None where the pages only the
  // cache holds would complete an extent with them, as its reclaims (reach)
  // and its demotions as they land (returning) count them then: not the
  // walk's reach of the Full form (Walk::reaches), which an empty extent
  // alone gives a claim that keeps no Full runway. None under --kv-spill
  // always, whose landings count them. Counted in whole extents of its own
  // pages, a lane on a finished lane's prefix gave one extent of its 511
  // releasable pages beside 3 free ones, and its hot block floor failed it
  // at a limit where it completes alone, on Qwen3.8-Flash-Next at
  // 9.38G and 9.43G.
  [[nodiscard]] uint32_t releaseBeside(const VictimQuery &query) const;
  // A Spill request's pages a release moves to the index form, oldest
  // first: Full pages of its own blocks past its dense history and short of
  // its tail window, every holder of which spills and has passed them, that
  // the rule lets a release take (ReleaseRule). Clean: their copies landed.
  // Landing: their copies are being written, or owed while the SSD cache
  // writes Spill requests' copies; a release takes them once they land.
  // Holders: those of the oldest, among whom is every holder of a later one,
  // as a block's holders hold its chain: the requests a release gives Index
  // pages. The count starts at the request's scan mark, past what is spilled
  // or kept Full. Past the releases `count` counted, it leaves out the pages
  // they move and, of the holders, the requests they give Index pages.
  struct ReleasablePages final {
    uint32_t clean = 0;
    uint32_t landing = 0;
    std::vector<uint64_t> holders;
  };
  [[nodiscard]] ReleasablePages releasablePages(uint64_t requestId, const ReleaseRule &rule,
                                                const ReleaseCount &count = {}) const;
  // What a release for bytes moves of `pages` pages: whole extents of them,
  // and where they run short the rest, should they complete one beside the
  // free pages `count` holds (completedExtents); and the Full extents those
  // empty, each of which leaves and pays for the next runway, every page of
  // them freeing its Full bytes less an Index page's. None unless its first
  // runway's moves into free Index pages, with the Full pages free already,
  // fill one. Compaction empties only extents that hold pages and none in
  // transfer, so as many as those beyond the extents `count` counted: moves
  // past them would empty none, yet refill the runway and give the lanes
  // Index pages and reads. With `landing`, as the extents stand once the
  // transfers land, which a release of pages whose copies land waits for.
  struct ReleaseMoves final {
    uint32_t pages = 0;
    uint32_t extents = 0;
  };
  [[nodiscard]] ReleaseMoves releaseMoves(uint32_t pages, bool landing, const ReleaseCount &count = {}) const;
  // Counts in `count` a release's `moves` of the request's oldest pages of
  // those releasablePages finds with it, clean or, with `landing`, landing
  // too: the pages, the holders of the oldest, the extents they empty, and
  // the pages they free beside the free pages it holds.
  void countRelease(uint64_t requestId, const ReleaseMoves &moves, bool landing, const ReleaseRule &rule,
                    ReleaseCount &count) const;
  // The free Index pages a release of `pages` Full pages in place moves them
  // into, and the runway past them, where free Index pages hold them all
  // (inPlaceMoves); else none. A claim that counts such a release keeps them
  // from its reclaims (EngineVictims::movesInto, walkFor).
  [[nodiscard]] uint32_t inPlaceIndexPages(uint32_t pages) const noexcept;
  // The pages a release freeing `pages` Full pages in place moves, of a
  // request's `clean` ones, so that an extent of free Index pages stays, the
  // runway the next release moves into: those pages where free Index pages
  // hold them beside it (inPlaceIndexPages); else an extent more, as for
  // bytes, whose emptied Full extent refills it, where releaseMoves moves
  // one; else none. Moved alone into the runway they could leave it short
  // where the room holds no Index extent (6.3 MB on Qwen3.8-Flash-Next), and
  // no release for bytes could start again.
  [[nodiscard]] uint32_t inPlaceMoves(uint32_t pages, uint32_t clean) const;
  // Between commands: moves the request's `pages` oldest clean pages
  // (releasablePages, by the same rule) to the index form, as spillWritten
  // moves landed pages: into free Index pages; for a need of bytes its
  // releaseMoves beside the `beside` free pages (releaseBeside), the Full
  // extents they empty then leave, the Full pages left first gathered in
  // the fewest extents, their bytes told to `funds`, and the runway refills
  // from them, a runway at a time; for a need of pages in place (`inPlace`)
  // its inPlaceMoves, an extent past `pages` moving as for bytes, the
  // `pages` Full pages staying free for the claim. A refused refill ends it
  // with what it moved. Returns the pages moved; a release counts
  // (KvSpillSnapshot::releases) where it freed what it was for.
  uint32_t releasePages(uint64_t requestId, uint32_t pages, bool inPlace, const ReleaseRule &rule,
                        const std::function<void(uint64_t bytes)> &funds = {}, uint32_t beside = 0);
  // Whether a Spill request owes copies no write has started: the SSD
  // cache takes no more of its KV, for the quota, the write budget or a
  // closed tier, or writes other requests' first.
  [[nodiscard]] bool owesCopies(uint64_t requestId) const;
  // Pages of the form that transfers in flight will free: demotions, and
  // for Full under --kv-spill always the pages Spill requests hold that
  // spill once their copies in flight land, written through or copied
  // ahead, as far as free Index pages take them (spillingWrites).
  [[nodiscard]] uint32_t pendingPages(kv::PageForm form) const noexcept;
  // The bytes the KV pool's extents take, of every form.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return pool_.allocatedBytes(); }
  // The bytes one page of the form takes, and the pages of an extent.
  [[nodiscard]] uint64_t bytesPerPage(kv::PageForm form) const noexcept { return pool_.bytesPerPage(form); }
  [[nodiscard]] uint32_t extentPages() const noexcept { return pool_.extentPages(); }
  // The bytes a block holding its window unit takes in memory, its base page
  // and its unit's (KvCache::blockPages); one without a unit takes a page's.
  [[nodiscard]] uint64_t bytesPerBlock() const noexcept {
    return pool_.bytesPerPage(kv::PageForm::Full) * kv_.blockPages();
  }
  // The pages a Resident request's conversion to Spill would move to the
  // Index form: its Full pages that spillEligibility would find Eligible
  // were it Spill, past its dense history and short of its tail window, no
  // Resident request holding them and every Spill holder past them.
  [[nodiscard]] uint32_t convertiblePages(uint64_t requestId) const;
  // The pool's free pages of `form`.
  [[nodiscard]] uint32_t freePages(kv::PageForm form) const noexcept { return pool_.freePageCount(form); }
  // The blocks of `chain`, a start's (CacheProbe::restoredChain), that are in
  // memory in a form its start takes in place (restoreRequest): any but the
  // Index form, which a Resident start moves to Full pages.
  [[nodiscard]] uint32_t residentChainPages(std::span<const uint64_t> chain) const;
  // The need `blocks` more blocks of the lanes make, their pages as
  // ensureTokens takes them (KvPool::takeFreePages): the free pages first,
  // which a walk for it keeps (Need::keptPages), then whole extents.
  [[nodiscard]] Need growthNeed(uint32_t blocks) const;
  // The most blocks whose growthNeed `bytes` meets: of the free pages, then
  // those of the whole extents the bytes allocate.
  [[nodiscard]] uint64_t growthPages(uint64_t bytes) const;
  // The request's pages of the index form, whose K/V only the SSD cache
  // holds.
  [[nodiscard]] uint32_t indexPages(uint64_t requestId) const;
  [[nodiscard]] bool holdsIndexPages(uint64_t requestId) const;
  // Whether a Spill start that restores `chain` (CacheProbe::restoredChain)
  // holds Index pages once it has (restoreRequest): a block of it past the
  // dense history that only the SSD cache holds, or one on an Index page.
  [[nodiscard]] bool restoresIndexPages(std::span<const uint64_t> chain) const;
  // The block's copy could not be read (ModelStepResult::unreadableKvBlocks):
  // it matches nothing any more, and its copy goes with the block.
  void poisonCopy(uint64_t block);

  // Scheduling probe only: does not pin, touch recency, or count a hit.
  [[nodiscard]] CacheProbe
  probe(std::span<const uint32_t> prompt, PromptIdentity identity);
  // Brings a probe this cache made up to date: after a change of the KV
  // graph its chain is cut at the first block that no longer matches and
  // matched on from there; its cached tokens follow the states either way.
  void refresh(CacheProbe &probe, std::span<const uint32_t> prompt,
               PromptIdentity identity);

  // Pin the usable prefix before potentially evicting for active allocations.
  // Accounting is separate: failed admission retries are not extra samples.
  // A probe of the prompt that has seen the KV graph as it is supplies the
  // matched chain; otherwise the prompt is matched again.
  [[nodiscard]] CacheLookup lookup(
      std::span<const uint32_t> prompt,
      PromptIdentity identity,
      const CacheProbe *probe = nullptr);
  void recordLookup(const CacheLookup &lookup);
  // Gives a state the lookup restored from disk the RAM copy `copy` takes
  // (StateRestore::snapshot, in a claim of the request), asked only while
  // the state has none; a copy refused leaves it on disk alone. A windowed
  // state's tail takes the units it lacks from `lane`, the request that
  // restored it (stateTailPresent); without them it stays on disk alone.
  void promoteState(const CacheLookup &lookup,
                    const std::function<std::shared_ptr<const CompositeState>()> &copy,
                    std::optional<uint64_t> lane = std::nullopt);
  // Gives the request the matched chain up to its state. Disk-only blocks
  // get fresh pages and start their restores; the request waits on
  // kvRestoreStatus() before it runs. Pages are admitted like ensureTokens(),
  // those its moved blocks take too: from the free ones beyond `held`.
  // By the request's mode: a Spill request takes a disk-only block past the
  // dense history as an Index page and restores its index alone, and holds
  // spilled blocks as they are. A Resident request restores disk-only blocks
  // whole, and moves a spilled block back to a Full page and restores it
  // whole when no other request holds it and no transfer moves it; else it
  // holds it spilled too.
  [[nodiscard]] TokenAdmission restoreRequest(uint64_t requestId,
                                              const CacheLookup &lookup,
                                              std::array<uint32_t, kv::kPageForms> held = {});
  [[nodiscard]] KvRestoreStatus kvRestoreStatus(uint64_t requestId) const;

  // Gives the request pages for its first tokenCount tokens, which the work
  // they are for writes below writtenTokens, all of them unless said
  // otherwise. A new page with a row the work does not write comes cleared:
  // attention weighs the rows past the request's end by zero, and a BF16
  // value is read as stored, so a non-finite one the page's previous holder
  // left there would reach the request as 0 x NaN. `held[form]`, the free
  // pages of the form waiting claims ranked above the request hold, stay
  // free (KvPool::acquirePages): a request refused the pages waits
  // (Pending) only while those on their way back cover what the free pages
  // beyond them do not.
  [[nodiscard]] TokenAdmission
  ensureTokens(uint64_t requestId, uint64_t tokenCount,
               uint64_t writtenTokens = std::numeric_limits<uint64_t>::max(),
               std::array<uint32_t, kv::kPageForms> held = {});
  // The request's page list as it is now, at the revision and first change
  // its next cut gives it.
  [[nodiscard]] PageTableState pageTable(uint64_t requestId) const;
  // The page list a command of the request reads, once per lane and
  // command: every change since the last cut makes one new revision, which
  // differs from the last from the first page any of them changed, with its
  // sources cut from there. A command reads only a cut list: another change
  // before the next cut stays in the same revision. With a window, the
  // lane first lets go of the units of the blocks its window has passed
  // since the last cut, one step a block (windowStart), each table entry
  // made absent before its pages go.
  [[nodiscard]] PageTableView cutPageTable(uint64_t requestId);

  // Canonicalizes every newly complete Page32 block. A block whose parent,
  // tokens and identity match a cached one swaps the request to that block's
  // immutable page after the writer command has completed; no active command
  // ever aliases a writable page. The cached page holds KV another command
  // computed, which can differ from the request's own in the last bits, and
  // the request's later commands read it. Returns the tokens of the swapped
  // blocks.
  uint32_t publishCommittedBlocks(uint64_t requestId,
                                  std::span<const uint32_t> exactTokens,
                                  uint32_t committedTokens,
                                  PromptIdentity identity);
  [[nodiscard]] uint64_t blockAt(uint64_t requestId, uint32_t boundary) const;
  [[nodiscard]] bool reuseCompositeState(uint64_t kvBlock, bool checkpoint);
  // Reuses the state at this block in either tier, as reuseCompositeState()
  // does a RAM copy.
  [[nodiscard]] bool reuseStoredState(uint64_t kvBlock, bool checkpoint = false);
  // A windowed layout's state is published only where every block of its
  // tail holds its window unit or takes one, in the same step, from `lane`,
  // the publishing request, which holds the block's base page with the
  // unit's pages in its tables; stateTailPresent says whether it is,
  // counting each refusal, and a publication where it is not is a logic
  // error. A layout without a window needs no lane.
  [[nodiscard]] bool stateTailPresent(uint64_t kvBlock, std::optional<uint64_t> lane = std::nullopt);
  void publishCompositeState(uint64_t kvBlock,
                             std::shared_ptr<const CompositeState> state,
                             bool checkpoint, std::optional<uint64_t> lane = std::nullopt);
  // Publishes the state of the lane at this block straight to disk, for a
  // state no cache slot can hold: `write` starts the write from the lane.
  // False when the tier cannot take the state now; nothing is published then.
  // The caller reuses a stored state first (reuseStoredState); publishing
  // over one is a logic error.
  [[nodiscard]] bool publishStateToDisk(uint64_t kvBlock, const StateWriter &write,
                                        bool checkpoint, std::optional<uint64_t> lane = std::nullopt);
  // The request holding the handle is unfinished and its conversation
  // resumes from the state at this block: see StateCache::useState.
  [[nodiscard]] StateUse useState(uint64_t kvBlock) { return states_.useState(kvBlock); }
  // A state at this block is in use, published or yet to be: its writes
  // pass the disk tier's write budget.
  [[nodiscard]] bool stateInUse(uint64_t kvBlock) const noexcept { return states_.inUse(kvBlock); }
  // The block is where a prompt's head ends (KvCache::noteHead).
  void noteHead(uint64_t kvBlock) { kv_.noteHead(kvBlock); }
  // The tokens of the state at this block: its chain's.
  [[nodiscard]] uint32_t stateDepth(uint64_t kvBlock) const { return kv_.chainLength(kvBlock) * KvCache::pageTokens; }
  // Pins the state at this block, out of every victim choice, while the lease lives.
  [[nodiscard]] std::optional<CompositeStateLease> pinState(uint64_t block) { return states_.acquireDeepest({&block, 1}); }
  [[nodiscard]] StateCheckpoint checkpointState(uint64_t kvBlock) const noexcept;
  // The state at this block has a RAM copy.
  [[nodiscard]] bool stateResident(uint64_t kvBlock) const noexcept;
  // False only while this exact disposable publication is pinned.
  bool retireCheckpointState(StateCheckpoint checkpoint) noexcept;
  // Notes the request's published blocks from its `first` on as its reply's (KvCache::noteReply).
  void noteReply(uint64_t requestId, uint32_t first) {
    for (const std::vector<uint64_t> &chain = request(requestId).cachedBlocks; first < chain.size(); ++first)
      kv_.noteReply(chain[first]);
  }
  // Notes the block the request's reply shares with its prompt's last
  // `promptTokens % pageTokens` tokens, once published (KvCache::noteReplyStart).
  void noteReplyStart(uint64_t requestId, uint32_t promptTokens) {
    const std::vector<uint64_t> &chain = request(requestId).cachedBlocks;
    if (const uint32_t at = promptTokens / KvCache::pageTokens; promptTokens % KvCache::pageTokens && at < chain.size())
      kv_.noteReplyStart(chain[at], promptTokens % KvCache::pageTokens);
  }
  // A reply's state, opportunistic or kept (StateCache::markReply).
  void markReply(StateCheckpoint checkpoint) noexcept { states_.markReply(checkpoint); }
  void keepReply(StateCheckpoint checkpoint) noexcept { states_.keepReply(checkpoint); }
  // This exact disposable publication is pinned: retireCheckpointState
  // refuses it.
  [[nodiscard]] bool checkpointPinned(StateCheckpoint checkpoint) const noexcept {
    return states_.checkpointPinned(checkpoint);
  }

  // Sparse history's spacing (historyWritten), by default: a go-back
  // recomputes at most this many tokens; about one point in every 8 turns of
  // a conversation is written.
  static constexpr uint32_t kSparseHistoryTokens = 8192;
  // Tests: sparse history's spacing in tokens, so that a test's chains of a
  // few dozen pages reach it (cache-property runs it at 4 to 8 pages).
  void sparseHistory(uint32_t tokens) noexcept { sparseHistoryTokens_ = tokens; }
  [[nodiscard]] uint32_t sparseHistoryTokens() const noexcept { return sparseHistoryTokens_; }

  // The cache as a source of the units the memory ledger takes, in the order
  // of its walk: free pages first (an empty extent, then compaction toward
  // one; for a page need other forms' empty extents), checkpoints, KV no
  // state restores through,
  // ordinary states and resident leaves by one recency order, then, for a
  // claim that reaches InUse, states in use and the KV they restore through.
  // The units of certain reuse, the resume point and the wanted chains, and
  // what is in use come in their place only to a claim whose rights take
  // them (walkFor, TakeRights).
  //
  // A device need (the engine's limit refused it, whatever pages it was for)
  // and a page need (pages in place; a page scan while the host refuses
  // growth) take the walk's reclaims, one at a time, in six phases, each a
  // unit: what frees with no write (copied, dead, a state with a disk copy
  // or none the tier takes), then the state the idle staging buffer writes,
  // whose RAM frees at once, then what frees only as the writes the
  // reclaims start land, of the ordinary class, then the same of what is in
  // use (ReclaimPhase). So every claim takes every unit that frees with no
  // write before one that needs a write, within the one recency order, and
  // an Immediate claim takes none whose write it would wait for. A page need
  // takes the other forms' empty extents first. A state need's units are
  // its states, one each.
  //
  // peek() changes nothing and counts each unit as VictimSource's contract
  // asks, the units through it an upper bound of what its take frees (a
  // take frees the free pages, and the extents they fill, first, which the
  // units before it may count): a phase's unit what its reclaims can free
  // (reach), from the counts the pool and the KV cache keep, at a cost that
  // does not grow with the cache, and none is given once that is nothing.
  // take() makes one reclaim of the unit's phase, of the state alone in a
  // phase of the staging buffer's write (the copied part of its chain,
  // which its unit counts too, then frees with no write, in the first
  // phase, which the ledger peeks again after a take), and reports what it
  // freed: the bytes of the extents released now, and those it adds to the
  // extents the landing pages complete beside the free ones (laterBytes,
  // landingFill), less any of them a release took; the pages of the need's
  // form free now, or for a claim that can wait landing; and the state
  // buffers it gave up.
  // A device need's reclaim that demotes a leaf starts the demotions of its
  // chain above it, child before parent, while the tier and the quota have
  // room (a waiting claim's chain lands in about ceil(pages/64) ticks rather
  // than one tick a page). Pending when it took nothing and a transfer in
  // flight holds what the walk needs, Short when nothing goes.
  [[nodiscard]] std::string_view name() const noexcept override { return "cache"; }
  [[nodiscard]] std::optional<engine::Victim> peek(const VictimQuery &query,
                                                   const engine::Victim *after) const override;
  FreeResult take(const VictimQuery &query, const engine::Victim &victim) override;
  // What the transfers in flight and the copies Spill requests owe free as
  // they land. Landing: demotions whose pages free as they land
  // (freesOnLanding), by the form of their pages, and under --kv-spill
  // always the pages Spill requests hold, written through or copied ahead,
  // that spill once their copies land, as far as free Index pages take
  // them. Owed: under --kv-spill always, the copies Spill requests owe of
  // the pages that spill once copied, while the SSD cache writes their
  // copies (a Spill write in flight), all of them: the Full extents they
  // empty fund their Index pages. Otherwise a Spill request's pages free
  // only as a claim releases them (releasePages), which counts those whose
  // copies land as a release of their own (releasablePages). The bytes are
  // the whole extents those pages add to the free pages of their form,
  // which compaction returns. A restore frees nothing, nor does the state
  // write, whose RAM copy left as it started, nor a copy ahead of a page
  // that stays.
  [[nodiscard]] Returning returning() const;
  // What returning() counts of the requests' own pages: those of their page
  // tables, written through or copied ahead, that spill once their copies
  // land, as far as free Index pages take them, and the copies they owe,
  // each block once; their owed bytes are the extents the owed pages
  // complete beside every landing, which returning()'s landing bytes count.
  // None but under --kv-spill always.
  [[nodiscard]] Returning returningFor(std::span<const uint64_t> requestIds) const;
  // The pages the demotions in flight free as they land, by form: returning()'s
  // landing pages less the Full pages that under --kv-spill always spill as
  // their copies land, which free only as spillWritten() moves them.
  [[nodiscard]] std::array<uint32_t, kv::kPageForms> demotedPages() const;
  // The chains the requests waiting to start restore (their admission
  // probes): the KV of every block of each, and the state at its last, which
  // the request resumes from (CacheProbe::restoredChain), are of certain
  // reuse until the next call.
  void setWantedChains(std::vector<std::span<const uint64_t>> chains);

  // Releases every empty extent, but one with keepRunway, and returns the
  // bytes released.
  [[nodiscard]] uint64_t releaseEmptyExtents(bool keepRunway);
  // A KV demotion, a KV restore or the one state write is in flight, so
  // memory or quota returns by itself and its completion wakes the engine;
  // so is a copy ahead whose block a reclaim that takes all it may then
  // frees through it (copyAheadReturns), or under --kv-spill always a page
  // written through or copied ahead that spills once its copy lands. A
  // walk's takes wait for those that return to it (transfersInFlight(walk)).
  [[nodiscard]] bool transfersInFlight() const noexcept;
  // A write of any class is in flight, a page's written through among them,
  // whose landing lets a release free the page, or the one state write.
  [[nodiscard]] bool writesInFlight() const noexcept;
  // The most pages of the form a page need's evictions can leave free: every
  // allocated page of the form no request holds, but with Ordinary not the
  // idle KV that states in use restore through.
  [[nodiscard]] uint32_t reusablePages(ReclaimClass upTo, kv::PageForm form = kv::PageForm::Full) const;
  [[nodiscard]] CacheSnapshot snapshot() const;
  // What the reclaim counts looked at beside the counts the caches keep:
  // the blocks of the chain a start reuses and the staged state's
  // (KvCache::reachVisits), and the states the walk keeps and those of its
  // order up to the staged one (StateCache::reclaimVisits); never the cache.
  // and the blocks and states the walk's searches visited: the states a
  // release of tails considers, the history search below each
  // (superseded), the points below an unwritten unit (unitPoints).
  [[nodiscard]] uint64_t reachVisits() const noexcept {
    return kv_.reachVisits() + states_.reclaimVisits() + searchVisits_;
  }
  // Whether a count stops once it covers its walk's need (reach) and a take
  // of its first phase takes steps up to the claim's lack, as they do; off,
  // a count counts all its walk reaches and a take takes one step (tests
  // compare the two).
  void boundCounts(bool bounded) noexcept { countsBounded_ = bounded; }
  // Tests: a walk that resumes within a take (takeOldest) also walks from
  // the head and throws std::logic_error where the two give different units.
  void checkResumes(bool check) noexcept { checkResumes_ = check; }
  // Tests: each block a tail release's search asks (nextTailRelease) is also
  // judged by the states within a window below it walked afresh
  // (unitPoints), std::logic_error where the two differ.
  void checkTails(bool check) noexcept { checkTails_ = check; }
  // The resident leaves the takes' walks were given (KvCache::leafVisits):
  // the leaves they tried, never the cache.
  [[nodiscard]] uint64_t leafVisits() const noexcept { return kv_.leafVisits(); }
  [[nodiscard]] KvSpillSnapshot spillSnapshot() const;
  // Tests, between commands: std::logic_error unless the blocks, pages,
  // references and quota keep every invariant spilling relies on.
  void checkInvariants() const;
  // Window units that went free other than by the releases the walk's
  // count foresees (releaseState with `counted`): those of a state the quota
  // gave up for a write's room, of the points a poisoned block or an
  // unwritable unit takes with it, of a leaf a lost copy takes, and of the
  // releases with no write in a take after its quota gave up a state or a
  // copy under one (quotaGaveUpState_).
  [[nodiscard]] uint64_t unitsFreedElsewhere() const noexcept { return kv_.unitsFreed() - unitsReleased_; }
  // Tests: every reach the reclaim counts take also follows its rule over
  // the whole cache (KvCache::checkReaches).
  void checkReaches(bool check) noexcept { kv_.checkReaches(check); }
  // Tests: every release of a state frees exactly the window units its
  // count names (releaseState), std::logic_error otherwise.
  void checkReleases(bool check) noexcept { checkReleases_ = check; }
  // Tests: each step of a take's walk checks the answers the cache keeps
  // (checkKept) before it reads one, as checkInvariants() does after it.
  void checkKeptAnswers(bool check) noexcept { checkKept_ = check; }

private:
  // A probe reads the KV graph to cut its chain (CacheProbe::matchableChain).
  friend class CacheProbe;
  // Matches the prompt's pages on from blocks.back() (from the root when
  // empty) and appends each block found; returns the pages it hashed.
  size_t extendMatch(std::vector<uint64_t> &blocks, std::span<const uint32_t> prompt,
                     PromptIdentity identity) const;
  // The tokens up to the deepest of these blocks that holds a state.
  [[nodiscard]] uint32_t stateTokens(std::span<const uint64_t> blocks) const;

  struct Request final {
    // Its blocks' base pages by logical page, and its window tables
    // (PageTableView::windows): every page of each block it holds.
    std::vector<uint32_t> pages;
    std::vector<std::vector<uint32_t>> windows;
    // The first logical page whose window entries it may hold: those before
    // are absent (cutPageTable).
    uint32_t windowFrom = 0;
    std::vector<uint64_t> cachedBlocks;
    KvMode mode = KvMode::Resident;
    // The last cut (cutPageTable) and the first page changed since, if any.
    uint64_t revision = 0;
    uint32_t firstChanged = 0;
    std::optional<uint32_t> dirtyFrom;
    std::shared_ptr<const model::KvPageSources> sources;
    // Its pages of the index form.
    uint32_t indexPages = 0;
    // A Spill request: the slots its commitment holds, when it began to
    // spill (revocation goes newest first), the blocks it holds whose copy
    // it still owes, in publication order after those whose write failed,
    // and how far its blocks are known spilled or kept Full by another
    // holder (landedPages).
    uint32_t committedPages = 0;
    // The copies on the chain it restores that its commitment counted
    // (commitSpill).
    uint32_t committedCopies = 0;
    // In a cache whose requests spill, the blocks of its chain it holds on
    // their own pages with a copy of their K/V (holdsCopy): the copies of
    // active KV it holds (holdsActiveKvQuota). Every request holding a block
    // counts its copy; the quota counts it once (KvCache::heldCopyBytes). A
    // block it keeps a page of its own beside, while a restore fills the
    // block's (KvCache::insert), it does not hold.
    uint32_t heldCopies = 0;
    uint64_t spillOrder = 0;
    std::deque<uint64_t> owedCopies;
    uint32_t spillScan = 0;
    uint32_t pendingRestores = 0;
    bool restoreFailed = false;
  };
  // Records a change of the request's page list from index first on.
  static void pagesChanged(Request &active, uint32_t first) noexcept;
  // The window unit of the request's logical page `index`.
  [[nodiscard]] static std::vector<uint32_t> windowAt(const Request &active, uint32_t index);
  // The first logical page a command of the request reads its window from:
  // windowPages pages before its published blocks end, as its rows lie at
  // or past them.
  [[nodiscard]] uint32_t windowStart(const Request &active) const noexcept;
  // The units the blocks of a state's tail that hold none take from the
  // lane that publishes it (stateTailPresent); none when the lane does not
  // hold one of them.
  [[nodiscard]] std::optional<KvCache::TailUnits> tailUnits(uint64_t kvBlock, std::optional<uint64_t> lane) const;
  // The request holds the block's pages at logical page `index` in place of
  // its own, which it still holds.
  void holdBlock(Request &active, uint32_t index, uint64_t block);
  // A page being written to the tier, and what for (WriteClass): a reclaim
  // demotion, whose page frees once the copy lands; a copy persist() writes
  // ahead of need of a page that stays; or a Spill request's page written
  // through, which stays Full until a claim releases it (releasePages), or
  // under --kv-spill always spillWritten moves it.
  // The parts it writes. A reclaim's write of a unit held for points on the
  // SSD alone (`unit`) frees that unit as it lands and leaves the block's
  // base page.
  struct Write final {
    uint64_t block = 0;
    WriteClass write = WriteClass::Reclaim;
    std::unique_ptr<KvTransfer> transfer;
    kv::Components parts = 0;
    AwakeClock::time_point started = AwakeClock::now();
    bool unit = false;
  };
  // A read of a block's window part into a lane's own unit pages, for the
  // tail of the point it resumes, apart from the block's transfer: the pages
  // it keeps where they are until it lands (pagesInTransfer), base page
  // first, whose extent the tier reads for the block's spans, and the
  // requests waiting. It holds the unit pages; the base page is the
  // block's.
  struct UnitRead final {
    uint64_t block = 0;
    std::vector<uint32_t> pages;
    std::unique_ptr<KvTransfer> transfer;
    std::vector<uint64_t> waiters;
  };
  struct Restore final {
    // Null until the tier has room for it.
    std::unique_ptr<KvTransfer> transfer;
    std::vector<uint64_t> waiters;
    AwakeClock::time_point started;
  };

  [[nodiscard]] Request &request(uint64_t requestId);
  [[nodiscard]] const Request &request(uint64_t requestId) const;
  // Pending, in every verdict below and in the results above, means the
  // same thing: a transfer in flight holds what this needs, and it comes
  // back when the transfer lands. Only transfersInFlight() may report it:
  // a caller told to wait for nothing would wait for ever.
  enum class LeafReclaim : uint8_t {
    Started,
    // The tier's room, the quota or the state write's staging buffer is held
    // by transfers in flight, or memory the walk takes returns as one lands
    // (transfersInFlight(walk)).
    Pending,
    // reclaimKvLeaf: the leaf stays for now and scans move on to the next
    // one, because a state on it is pinned, a disk subtree depends on it
    // and the tier has no room that a transfer in flight will free, a disk
    // subtree below it cannot drop yet (active, in transfer, or a state
    // write in flight), or the leaf needs a demotion an Immediate caller
    // cannot wait for. demoteKv: the leaf cannot be written, and may
    // go without a copy, because the tier takes no writes, has no room
    // while nothing in flight returns anything to the walk, or making room
    // took the states the leaf was kept for.
    Impossible,
  };

  // What one reclaim step did.
  struct ReclaimResult final {
    bool madeProgress = false;
    // Bytes of the KV extents the step released.
    uint64_t reclaimedBytes = 0;
    // Nothing was reclaimed, but a transfer in flight (a KV demotion, a KV
    // restore or the one state write) holds what the next reclaim needs.
    // Retry when it lands rather than treating the cache as empty.
    bool pending = false;
    // The step evicted a state from RAM. That frees nothing by itself: the
    // state's buffers return to the model's pool, which releaseIdleBuffer
    // gives back to the host.
    bool evictedState = false;
    // The step evicted KV, whose pages are free for other KV, or will be
    // once their demotion lands. Only a memory reclaim (takeReclaim) reports
    // it: its progress is memory that came back, which KV gives only as an
    // extent it empties.
    bool evictedKv = false;
  };
  // What one reclaim step gave up: a state's RAM copy, whose buffers return
  // to the model's pool, or KV, a leaf's or a unit a point on the SSD alone
  // held, whose memory returns with the extent it leaves empty.
  struct Victim final {
    bool kv = false;
  };

  // The units of the walk peek() and take() go over, and the reclaim steps with
  // them (Victim::key holds the unit above the block's id): the empty extents
  // of the other forms released, for a page need; a phase of the reclaims for a device or a page need (takeReclaim, evictOne),
  // whose id holds the phase; a checkpoint in RAM; a KV leaf no state restores
  // through; an ordinary state in RAM or a resident KV leaf, or in a page scan
  // the state in RAM on the leaf that goes next; and the same of what is in
  // use; a state whose release frees the window units only it holds
  // (tailRelease). peek() gives a state need's states as units of their
  // own, and the reclaim steps take the leaves.
  enum class Unit : uint8_t {
    Extent = 1,
    Reclaim,
    Checkpoint,
    DeadLeaf,
    State,
    Leaf,
    LeafState,
    UsedState,
    UsedLeaf,
    UsedLeafState,
    Tail,
    Unwritten,
  };
  // The releases of states for their tails' window units, in the order
  // every claim's walk takes them (the Tails phase): superseded history,
  // then the points behind the running lanes but each lane's newest, by
  // recency. A lane's plan prices no tails, so a lane at the wall reaches
  // its plan's length only by the tails behind it (1.7.3's #824/#842 rule).
  enum class TailRelease : uint8_t { History, BehindLanes };
  // What a walk goes over: what is in use too with InUse, only the leaves
  // and the states in RAM on them in a page scan, and only states for a
  // state's buffers.
  struct WalkScope final {
    ReclaimClass upTo = ReclaimClass::Ordinary;
    bool pagesOnly = false;
    bool statesOnly = false;
    friend bool operator==(const WalkScope &, const WalkScope &) = default;
  };
  // Where a walk stands: its phase, the candidates of the phase's orders,
  // the oldest first, and the unit it gave last, whose order moves past it
  // at the next step, once whatever was done with it is done, as a reclaim
  // step's scan does. A state that waits for the write in flight holds back
  // the newer states of its phase (statesHeld), and a leaf whose demotion
  // waits for the tier the newer leaves of its phase that need one
  // (demotionsHeld).
  struct WalkCursor final {
    enum class Phase : uint8_t { Checkpoints, DeadLeaves, History, Tails, Unwritten, ByUse, InUse, Done };
    Phase phase = Phase::Checkpoints;
    // A phase by use goes over its items by what their leaving costs, each
    // pass by recency: what leaves at no loss (lostOnLeaving: a state the SSD
    // keeps a copy of or its release writes, a leaf with a copy or that a
    // demotion writes); the running lanes' recovery points, rolling
    // checkpoints each lane's next one supersedes (recoveryPoint); what is
    // lost.
    enum class Pass : uint8_t { NoLoss, Recovery, Lost };
    Pass pass = Pass::NoLoss;
    TailRelease tails = TailRelease::History;
    bool started = false;
    bool statesHeld = false;
    bool demotionsHeld = false;
    // A resumed walk (takeOldest) starts its phase's leaves over.
    bool restartLeaves = false;
    std::optional<CacheEvictionCandidate> state;
    std::optional<CacheEvictionCandidate> leaf;
    std::optional<Unit> last;
    CacheEvictionCandidate lastCandidate{};
  };
  struct WalkUnit final {
    Unit unit = Unit::Checkpoint;
    CacheEvictionCandidate candidate{};
  };
  // What an allocation that was refused lacks, as the free pages that cover
  // it: the KV pages it lacks, of their form; for memory the governor funds,
  // an extent's worth of Full pages, whose extent then leaves memory; none
  // for a state's buffers while the host refuses growth, which only a state
  // gives.
  struct Shortfall final {
    uint32_t pages = 0;
    kv::PageForm form = kv::PageForm::Full;
    friend bool operator==(const Shortfall &, const Shortfall &) = default;
  };
  // What a walk takes, and how: a query's (walkFor) or a reclaim step's.
  // Its scope; how long it can wait for memory (ReclaimTiming); whether it
  // starts writes at all, which the reclaims of what frees with no write
  // (ReclaimPhase::Now) do not (writes); what it passes over then
  // (stateReclaim);
  // what it keeps beside what is in use, the resume point and the restored
  // chains; the shortfall a transfer in flight must cover for a step that
  // finds nothing to report pending (returnsInFlight), any transfer without
  // one; whether the runway stays; and a query's need: a device need's bytes
  // (its unit the memory reclaims up to an extent released or a state given
  // up) or a page need's pages of a form (its unit the evictions up to a
  // page freed), the states on the way among a unit's victims unless the
  // need counts state buffers too (statesAreUnits). A device need that also
  // lacks pages of a form keeps the free pages of that form it takes first,
  // those its takes free there among them, those of the other forms its
  // attempt takes, and those other sources' units move pages into
  // (keptPages, VictimQuery::movedInto): its reclaims release an extent of a
  // form only past them, and none of its own form.
  struct Walk final {
    WalkScope scope;
    ReclaimTiming timing = ReclaimTiming::CanWait;
    bool writes = true;
    bool keepResumePoint = false;
    // The chains it keeps: those queued requests restore (wanted_), or none,
    // and the chain a start's placement reuses (VictimQuery::kept).
    const RestoredChains *restored = nullptr;
    const RestoredChains *reused = nullptr;
    std::optional<Shortfall> refused;
    bool keepRunway = false;
    bool device = false;
    // A device need that lacks pages too: its reclaims take KV for those
    // pages in place as well as for whole extents (Cache::formsInReach,
    // takeReclaim).
    bool inPlace = false;
    bool statesAreUnits = false;
    uint32_t pages = 0;
    // A device need's bytes, and its pages in place: what a count covers
    // before it stops (Cache::reach).
    uint64_t bytes = 0;
    uint32_t inPlacePages = 0;
    kv::PageForm form = kv::PageForm::Full;
    std::array<uint32_t, kv::kPageForms> keptPages{};
    bool hostRefused = false;
    friend bool operator==(const Walk &, const Walk &) = default;
    // Whether it keeps the state at `block`, or the KV block, of a chain.
    [[nodiscard]] bool keepsState(uint64_t block) const {
      const auto in = [&](const RestoredChains *chains) {
        return chains && std::ranges::find(chains->states(), block) != chains->states().end();
      };
      return in(restored) || in(reused);
    }
    [[nodiscard]] bool keepsBlock(uint64_t block) const {
      return (restored && restored->holds(block)) || (reused && reused->holds(block));
    }
    // Whether its releases leave the form one empty extent: the runway its
    // reserves keep, and for every walk the Index runway
    // (TakeRights::Reserves).
    [[nodiscard]] bool keeps(kv::PageForm form) const noexcept {
      return keepRunway || form == kv::PageForm::Index;
    }
    // Whether its releases may take an extent of the form, which has
    // `freePages` free: only past the free pages it keeps (keptPages).
    [[nodiscard]] bool releases(kv::PageForm each, uint32_t freePages, uint32_t extentPages) const noexcept {
      return freePages >= keptPages[static_cast<size_t>(each)] + extentPages;
    }
    // Whether evicting KV of the form may still empty an extent: of the
    // form's `reusable` pages, those the demotions in flight hold
    // (`demoting`) and the empty extent its releases keep (keeps) leave one,
    // past which extentsCompleted counts the extents landings complete too.
    // The free pages a KV need takes first (keptPages) stay in: KV of its
    // form that leaves gives it pages in place.
    [[nodiscard]] bool reaches(kv::PageForm each, uint32_t reusable, uint32_t demoting,
                               uint32_t extentPages) const noexcept {
      return uint64_t{demoting} + uint64_t{extentPages} * (keeps(each) ? 2 : 1) <= reusable;
    }
  };
  [[nodiscard]] Walk walkFor(const VictimQuery &query) const;
  // The walk's next unit, moving the cursor on: the checkpoints, the dead
  // leaves (not for states alone), the ordinary class's states and leaves by
  // recency, then, up to InUse, what is in use. A walk that cannot wait for
  // a demotion, from the start or once one waits for the tier
  // (demotionsHeld), goes over only the leaves whose page frees with no
  // write (reclaimKvLeaf): those with a copy, and in a page scan those under
  // a state in RAM, which goes first. Each leaf comes from the orders of the
  // leaves the walk goes over (KvCache::leafCandidate), so that a walk costs
  // the units it gives, not the cache. Nothing changes.
  [[nodiscard]] std::optional<WalkUnit> nextUnit(const Walk &walk, WalkCursor &cursor) const;
  // The next unit a step of the walk tries: past the units it keeps.
  [[nodiscard]] std::optional<WalkUnit> nextStepUnit(const Walk &walk, WalkCursor &cursor) const;
  // A cursor whose next unit is the one after this state unit of the walk
  // for states alone; at the walk's end when the unit is gone.
  [[nodiscard]] WalkCursor cursorAfter(Unit unit, uint64_t block) const;
  // What a walk that counts its units, and takes none, knows a step would
  // have learnt by the units before the cursor: a checkpoint that waits for
  // the write in flight ends the checkpoints, and an older state of the
  // phase that does holds back the newer ones.
  void holdAfter(const Walk &walk, WalkCursor &cursor, uint64_t staged) const;
  // A step that keeps the resume point, or the restored chains, passes over
  // the unit.
  [[nodiscard]] bool kept(const WalkUnit &unit, const Walk &walk) const;
  // The state is of certain reuse beside what is in use: the resume point,
  // a wanted chain's state.
  [[nodiscard]] bool certainReuse(uint64_t block) const;
  // The ordinary state in RAM is history whose leaving writes nothing
  // (releaseMode): a deeper state superseded it, and it has a complete copy,
  // the tier takes no writes, or sparse history leaves it unwritten. A walk
  // reaches it before any state nothing superseded (the History phase);
  // history that sparse history writes leaves by use, as a point does.
  [[nodiscard]] bool freeHistory(uint64_t block, const std::unordered_set<uint64_t> &gone = {}) const;
  // Sparse history: a superseded point no request uses, leaving memory
  // without a complete SSD copy of its own (its state's and its tail's
  // window parts landed), is written where no point of its chain below it
  // within the spacing (sparseHistoryTokens) has a copy that stays
  // (copiedBelow): a chain's first point, and then one point past the
  // spacing from the last, so a go-back recomputes at most about the
  // spacing. Its writes go at history's worth (writeWorth), its tail's
  // window parts written as transfers are polled (writeWindowsAhead). Else it
  // leaves unwritten, early, as its leaving frees memory now and writes
  // nothing. A
  // superseded point with a complete copy leaves memory with no write, its
  // copy staying. `below`: the point copiedBelow gives, where the caller
  // has it.
  [[nodiscard]] bool historyWritten(uint64_t block, const std::unordered_set<uint64_t> &gone = {},
                                    const uint64_t *below = nullptr) const;
  // The one predicate on a point's SSD copy, which the release mode, the
  // tails and sparse history read: whether its copy stays as it leaves
  // memory. A state in RAM keeps or writes its copy where keptLeaving says;
  // a point on the SSD alone whose window parts have not landed is kept on
  // the same terms, else the tails give it up for its units with its copy.
  // It answers alike before and after the point's own release.
  [[nodiscard]] bool copyStays(uint64_t block, const std::unordered_set<uint64_t> &gone = {},
                               const uint64_t *below = nullptr) const;
  // Whether a point whose leaving writes (`writes`, its tail's window parts
  // among them: `window`) keeps its copy: a copy whose parts landed stays in
  // every mode (it writes nothing), so a persistent tier's landed point is
  // never lost; where the KV tier takes no writes no window part is
  // written; history no request uses is written only where sparse history
  // writes it (historyWritten), and lacking window parts it is kept in RAM,
  // whose release queues them, and on the SSD alone while they are queued
  // (windowsAhead_), else the tails take it.
  [[nodiscard]] bool keptLeaving(uint64_t block, const std::unordered_set<uint64_t> &gone, bool writes,
                                 const uint64_t *below) const;

  // The state's leaving loses it: the SSD holds no copy, and its release
  // would not write one.
  [[nodiscard]] bool lostOnLeaving(uint64_t block) const;
  // The walk's History phase (`history`) or its Ordinary phase takes the
  // ordinary state in RAM: lost history in the one, none in the other, and
  // only where it serves the need or, ordinary, no state in RAM below it
  // keeps its chain. Of a count, against the pins its releases took off
  // (`gone`) and the RAM copies they took (`released`).
  [[nodiscard]] bool ordinaryTakes(const Walk &walk, uint64_t block, bool history,
                                   const std::unordered_set<uint64_t> &released = {},
                                   const KvCache::Gone &gone = {},
                                   const std::unordered_set<uint64_t> &evicted = {}) const;
  // The checkpoint is a running lane's recovery point: on a page a request
  // holds, no deeper state superseded it, and a reclaim would drop it, not
  // write it. It is no disposable checkpoint (Checkpoints) but goes by use,
  // after what leaves at no loss and before what else is lost
  // (WalkCursor::Pass): where the tier writes the ordinary class, a lane at
  // the wall no longer evicts each reply checkpoint it just captured (62
  // published and 61 evicted over a 2,000-token reply, #818); where nothing
  // is written, a lane's rolling checkpoint, which its next supersedes, goes
  // before an idle conversation's newest state, as 1.7.3 took checkpoints.
  [[nodiscard]] bool recoveryPoint(uint64_t block) const;
  // The walk's phase after `phase`.
  [[nodiscard]] WalkCursor::Phase phaseAfter(const Walk &walk, WalkCursor::Phase phase) const;
  // The one worth of a point's write the release mode, a state's write and
  // a unit's window parts' write read (DiskWorth).
  using Worth = DiskWorth;
  [[nodiscard]] Worth writeWorth(uint64_t block, const std::unordered_set<uint64_t> &gone = {}) const;
  // How every release takes a state (reclaimState, the tails, the count):
  // demoted, its RAM copy leaving through its SSD copy or the staging
  // buffer's write and its tail's units as their window parts are written,
  // where its copy stays (keptLeaving); or evicted with its copy. Whether
  // it writes anything (then only a walk that writes and waits takes it),
  // and at what worth. With the states a count of releases took before
  // (`gone`).
  struct ReleaseMode final {
    bool evict = false;
    bool writes = false;
    Worth worth;
  };
  [[nodiscard]] ReleaseMode releaseMode(uint64_t block, const std::unordered_set<uint64_t> &gone = {},
                                        const uint64_t *below = nullptr) const;
  [[nodiscard]] KvCache::PinRelease pinRelease(uint64_t block, bool evict) const;
  // The next state past `after`, oldest first, that a release of tails
  // takes at the stage: of the states it may take (the checkpoints, the
  // ordinary states and the only copies on the SSD, none in use, pinned,
  // kept by the walk or `released` already; at History the superseded ones,
  // behind the lanes all but each running lane's newest), one whose tail
  // holds a block every pin in memory left on which,
  // beside those `gone`, is such a state's, whose unit then frees or, for a
  // walk that writes and waits, is left to the write of its window part,
  // the block's unit idle and, behind the lanes, its base page a lane's.
  // What the walk gives (nextUnit) and what its count releases
  // (releasedUnits) alike; it judges the states it reaches and those whose
  // tails share a block with theirs, not the whole cache.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  nextTailRelease(const Walk &walk, TailRelease stage, const std::optional<CacheEvictionCandidate> &after,
                  const std::unordered_set<uint64_t> &released, const std::unordered_set<uint64_t> &evicted,
                  const KvCache::Gone &gone) const;
  // The blocks whose units the releases of states a walk of the class
  // makes with no write free, each at the release that takes its last pin
  // (KvCache::freedBy), in the walk's order (phaseAfter) and as its phases
  // take them: checkpoints, history, tails, ordinary states and recovery
  // points (these first where the tier takes no writes), and what is in use
  // that leave with no copy (now), and the states those releases take; those the `staged`
  // state's write frees as it starts (staged); and for a walk that writes and
  // waits, those the tails' demotions and the states that wait for the
  // staging buffer free as their writes start or land (written). None
  // without a window unit, and so in no page scan (walkFor).
  struct ReleasedUnits final {
    std::vector<uint64_t> now;
    std::unordered_set<uint64_t> states;
    std::vector<uint64_t> staged;
    std::vector<uint64_t> written;
  };
  // The releases a count simulates, in the walk's order, until `enough`
  // says the units they free so far cover the walk's need (asked before the
  // first and after the 1st, 4th, 16th ... release), or to the end.
  [[nodiscard]] ReleasedUnits releasedUnits(const Walk &walk, ReclaimClass upTo, uint64_t staged,
                                            const std::function<bool(const ReleasedUnits &)> &enough = {}) const;
  // A release of the state at the block, by eviction or reclaim, which
  // frees exactly the units KvCache::freedBy names now (checkReleases):
  // every state the cache gives up leaves through it, the walk's, those a
  // lost leaf takes with it (reclaimKvLeaf, dropDiskSubtree), and with
  // `counted` false those no count foresees: the quota's give-ups for a
  // write's room (freeDiskSpace), the points a poisoned block or an
  // unwritable unit takes with it (dropPoisoned, dropUnitPoints) and a lost
  // copy's leaf (evictStateOnDiskLeaf), whose units are freed elsewhere.
  [[nodiscard]] StateEviction releaseState(uint64_t block, bool evict, bool counted = true);
  // Starts the writes of the window parts of units held for points on the
  // SSD alone, those a release left so or unwrittenUnit ones: Started, or
  // as demoteKv says what holds it back.
  [[nodiscard]] LeafReclaim writeUnit(const Walk &walk, uint64_t block);
  // Gives up the points on the SSD whose tails hold an unwritten unit, and
  // with them the unit, where the walk takes the unit (unitTakes); whether
  // the unit went.
  [[nodiscard]] bool dropUnitPoints(const Walk &walk, uint64_t block);
  // The states at the block and below it within a window, whose tails hold
  // it; whether the walk takes the block's unwritten unit, every one of them
  // a point it may give up should the write find no room (none in use for a
  // class short of InUse, pinned or kept); the unwritten units it does not
  // take, which its reach leaves out.
  [[nodiscard]] std::vector<uint64_t> unitPoints(uint64_t block) const;
  [[nodiscard]] bool unitTakes(const Walk &walk, uint64_t block) const;
  [[nodiscard]] std::vector<uint64_t> unitsKept(const Walk &walk) const;
  // What a walk does with the RAM copy of a state it reaches, the one rule
  // every step and count of the walk follows (holdAfter, reclaimState,
  // victimOf, reach through StateCache::reclaims, recycleState): passes it
  // over when it needs a write (StateCache::reclaimWrites) the walk does not
  // start, as the walk starts none or cannot wait while the staging buffer
  // is busy; else frees it now, with no write or through the idle staging
  // buffer's (`write`); else frees it once the staging buffer is free
  // (Later), or once its own copy ahead lands. The staging buffer writes
  // one state at a time: it is busy while a write is in flight, and for a
  // count of the walk's states taken in order (peek) for every state but
  // the first that needs a write (`staged`, StateCache::Reclaims::staged),
  // which takes it first; a take names none, as it finds the buffer.
  struct StateReclaim final {
    enum class When : uint8_t { PassedOver, Now, Later };
    When when = When::Now;
    bool write = false;
  };
  [[nodiscard]] StateReclaim stateReclaim(const Walk &walk, uint64_t block, uint64_t staged = 0,
                                          const std::unordered_set<uint64_t> &gone = {}) const;
  // The states a walk keeps beside those in use, which it passes over: the
  // resume point while it keeps it and the states the chains it keeps
  // restore from.
  [[nodiscard]] std::vector<uint64_t> keptStates(const Walk &walk) const;
  // The reasons whose chains a walk's reclaims of a class keep, as the KV
  // cache counts them (KvCache::kChainKeeps): below InUse those of the
  // states in use (Used), the waiting requests' unless the walk takes them
  // (Wanted), the resume point's while it keeps it (ResumePoint). The chain
  // a start reuses it walks instead (Walk::reused).
  [[nodiscard]] static KvCache::Keeps keptChains(const Walk &walk, ReclaimClass upTo) noexcept;
  // The first state the walk writes through the idle staging buffer, in its
  // order; 0 for none, or while a write holds the buffer.
  [[nodiscard]] uint64_t stagedState(const Walk &walk) const;
  // What peek() reports of a state unit the walk reaches (stateReclaim, the
  // states counted in the walk's order): what frees now gives its buffers;
  // what the walk passes over or what waits for the staging buffer gives
  // nothing yet, later, so that a claim that does not wait for writes
  // passes over it.
  [[nodiscard]] std::optional<engine::Victim> victimOf(const Walk &walk, const WalkUnit &unit,
                                                       uint64_t staged) const;
  // The phases of a device or a page need's reclaims (Unit::Reclaim): of
  // the ordinary class, then of what is in use, each what frees with no
  // write, then the state whose write the idle staging buffer takes, which
  // frees at once, then what frees only as the writes the reclaims start
  // land.
  enum class ReclaimPhase : uint8_t { Now, Written, Later, UsedNow, UsedWritten, UsedLater };
  // At most what a walk's reclaims of a class free, from counts: of each
  // form, the whole extents of the pages within the class's reach, but the
  // idle pages of the chains it keeps, the free pages a KV need takes and
  // the runway it keeps, as compaction gathers what stays; now, the free
  // pages and those nothing keeps or lets go only later (KvCache::Keep: a
  // state that needs a write, which stays in RAM until its write starts and
  // needs its chain on the SSD after, or that waits for the write in
  // flight; a block a writable tier writes before its page frees, as a
  // state that stays needs it; a block in use or in transfer, a page a
  // request holds; and their chains); written, those the state the idle
  // staging buffer writes lets go once it has left RAM, the copied part of
  // its chain; later the rest, but the extents the landing pages complete
  // beside the free ones (landingExtents), which the ledger counts as
  // landing. For a walk that can wait, now and written count too the
  // extents landing pages complete beside the pages then free, with the
  // landings that taking a state in RAM lets free, as its takes credit them
  // (Cache::take). States: the RAM copies the walk takes, now those
  // that need no write; written the first that needs one, while no write
  // holds the staging buffer, which frees at once; later the others, which
  // wait for that buffer (StateCache::reclaims). The counts cost the chains
  // the walk keeps that the KV cache does not count by itself (the one a
  // start reuses), the staged state's chain and the writes in flight, not
  // the cache.
  struct ReclaimReach final {
    struct Amount final {
      uint64_t bytes = 0;
      uint32_t pages = 0;
      uint32_t states = 0;
    };
    Amount now;
    Amount written;
    Amount later;
    [[nodiscard]] const Amount &of(ReclaimPhase phase) const noexcept;
  };
  [[nodiscard]] ReclaimReach reach(const Walk &walk, ReclaimClass upTo) const;
  // One reclaim of the phase for a take of its unit (Cache::take), which
  // settles what lands over all its steps.
  [[nodiscard]] FreeResult reclaimStep(const Walk &walk, ReclaimPhase phase);
  // The most one reclaim frees in the walk's measure: pages, or a device
  // need's bytes.
  [[nodiscard]] uint64_t stepMost(const Walk &walk) const;
  // The most pages taking the unit frees now: a leaf's page and unit; a
  // state's release the units it frees (KvCache::freedBy, released or
  // demoted) and, on a resident leaf, the leaf's.
  [[nodiscard]] uint64_t unitMost(const WalkUnit &unit) const;
  // The walk of a phase: of its class; Immediate but for what frees as
  // writes land; no write for what frees with none.
  [[nodiscard]] Walk phaseWalk(const Walk &walk, ReclaimPhase phase) const;
  // peek()'s unit of a phase: its amount of what the walk's reclaims reach
  // of the ordinary class (`ordinary`), or for a phase of what is in use, of
  // what they reach of that class (`inUse`) beyond it; none when nothing.
  [[nodiscard]] std::optional<engine::Victim> reclaimUnit(const Walk &walk, ReclaimPhase phase,
                                                          const ReclaimReach &ordinary,
                                                          const ReclaimReach *inUse) const;
  // Starts the demotions of the blocks above each demotion from writes_
  // [firstWrite] on whose page frees as it lands, child before parent,
  // those that then free their pages as they land too (freesOnLanding), of
  // the walk's class and not of a chain it keeps, until they are `enough`
  // or the tier or the quota has no room for one.
  void demoteChainsAbove(const Walk &walk, size_t firstWrite, const std::function<bool()> &enough);
  // Frees the RAM copy of a state the walk reached, as the walk says; one it
  // passes over stays.
  [[nodiscard]] StateEviction reclaimState(const Walk &walk, uint64_t block);
  // What taking one unit of the walk did: it went (a state's RAM copy, a
  // leaf's page now or through its demotion), it waits for a transfer in
  // flight, or it stays.
  struct UnitTaken final {
    bool taken = false;
    bool pending = false;
  };
  [[nodiscard]] UnitTaken takeUnit(const Walk &walk, const WalkUnit &unit, ReclaimTiming timing);
  // Whether a transfer in flight can return what the walk needs first:
  // cover its shortfall (returnsInFlight), or with none, any
  // (transfersInFlight).
  [[nodiscard]] bool returns(const Walk &walk) const noexcept;

  // Takes `count` free pages of the form, past the `held` ones, into
  // `pages`; refused (Denied) when the pool has not that many.
  [[nodiscard]] TokenAdmission admitPages(uint32_t count, std::vector<uint32_t> &pages, kv::PageForm form,
                                          uint32_t held);
  // Whether a transfer in flight can still cover the walk's shortfall once
  // it lands. A restore, a demotion and the state write each hold back what
  // a reclaim would take then: the block a restore fills, the page a
  // demotion frees and the leaf above it, the states that wait for the
  // write; so does a copy persist() writes that returns to the walk
  // (copyAheadReturns). A page a Spill request holds, written through or
  // copied ahead, holds back no cached block: it counts only by the Full
  // page it frees once it spills (pendingPages), where those pages and the
  // free ones cover the shortfall. Without spilling this is
  // transfersInFlight(walk).
  [[nodiscard]] bool returnsInFlight(const Walk &walk, const Shortfall &shortfall) const noexcept;
  // transfersInFlight() as the walk's takes wait for it: of the copies
  // ahead, those that return to the walk.
  [[nodiscard]] bool transfersInFlight(const Walk &walk) const noexcept;
  // A write in flight frees a unit as it lands, whatever its class
  // (KvCache::unitLeaving).
  [[nodiscard]] bool unitsLanding() const noexcept;
  // A copy persist() writes frees nothing as it lands: its page stays, as
  // does a Spill request's once no request holds its block. Until then it
  // holds its block out of every order, and every block above it.
  // Once it lands the block is idle, and a reclaim of the class frees it
  // through the copy after the resident blocks below it, and those above it
  // after it, where the walk's reach counts an idle page (KvCache::pageReach):
  // nothing at the block or below it keeps it for a reason no reclaim takes
  // (Busy: a block a request or a state's lease uses, a page a request
  // holds) or for a chain the walk keeps (keptChains), and no chain a start
  // reuses holds it (Walk::reused). The copy then returns memory to the
  // walk: its reach counts the block and those above it (Cache::reach), and
  // its takes wait for the copy (transfersInFlight(walk)). A state in RAM on
  // the block goes first, the leaf after it, in a page scan too. A copy of a
  // page a Spill request holds frees its page as it spills (spillingWrites).
  [[nodiscard]] bool copyAheadReturns(const Write &write, const Walk &walk, ReclaimClass upTo) const noexcept;
  // Some copy ahead in flight returns to the walk, of its own class.
  [[nodiscard]] bool copyAheadReturns(const Walk &walk) const noexcept;
  // One eviction: the walk's first unit that goes. Disposable checkpoints,
  // KV leaves no state restores through, then ordinary states and resident
  // KV leaves in one oldest-first order; states in use and the KV they
  // restore through follow in their own such order. A chosen state keeps
  // its disk copy when it has one, is written when the tier admits it and
  // dropped otherwise; its buffers return to the model's pool
  // (releaseIdleBuffer). A chosen KV leaf frees its page at once when it
  // has a disk copy, is dropped when nothing depends on it, and is
  // otherwise written first: its page returns when the copy has landed.
  // While the tier can start no demotion, only the leaves that need one
  // stay. Active requests and pinned restores are never selected. A page
  // scan has no checkpoints. A step that finds nothing reports pending while
  // a transfer in flight can return what is needed (returns). With `reach`,
  // a leaf goes only where it leads to an extent in reach (takeOldest).
  [[nodiscard]] ReclaimResult evictOne(const Walk &walk,
                                       std::optional<std::array<bool, kv::kPageForms>> reach = std::nullopt);
  // The units of the walk, each taken in turn until one goes, which it returns.
  // One that waits for a transfer in flight leaves the others to go: a
  // checkpoint ends the checkpoints, a state holds back the newer states of its
  // class, and once a demotion has to wait for the tier, the leaves of its
  // class that need one stay. Nothing once all stay. With `reach`, the forms
  // whose extents evicting KV may still empty (formsInReach), a leaf that leads
  // to none of them (leadsToReach) frees no memory, and the walk passes over
  // it. `waited`: a unit waited for a transfer in flight.
  [[nodiscard]] std::optional<Victim> takeOldest(const Walk &walk,
                                                 std::optional<std::array<bool, kv::kPageForms>> reach, bool &waited);
  // The state the walk's count staged (stagedState) goes through the idle
  // staging buffer's write, as the walk says (reclaimState): the count and
  // the take of the Written phase name one state.
  [[nodiscard]] ReclaimResult recycleState(const Walk &walk);
  // One memory reclaim of a unit of the walk, which began with `demotions`
  // pages held by the demotions in flight (demotionPages) and the landing
  // pages filling `fillBefore` bytes of extents (landingFill): free pages
  // first, as the extent they fill; then, while some form is in reach
  // (formsInReach), the walk's victims, its leaves that lead to such a form
  // alone (leadsToReach), until one is released, the landing pages fill one
  // more, a need of pages (Walk::inPlace) has one more page of its form, or
  // a state goes; otherwise a state of the walk, its leaves passed over.
  [[nodiscard]] ReclaimResult takeReclaim(const Walk &walk, const std::array<uint32_t, kv::kPageForms> &demotions,
                                          uint64_t fillBefore);
  // Frees the RAM of one resident KV leaf: through its disk copy when it has
  // one, by demotion when a state on it or below it depends on it, by
  // erasure otherwise, with any disk copies below it. Pending when the tier
  // cannot take it right now while a transfer in flight returns something
  // to the walk (demoteKv). A leaf whose demotion cannot start while nothing
  // does is erased, together with the disk subtree that depends on it.
  // Immediate passes over a leaf that needs a demotion.
  [[nodiscard]] LeafReclaim reclaimKvLeaf(const Walk &walk, uint64_t block, ReclaimTiming timing);
  [[nodiscard]] LeafReclaim demoteKv(const Walk &walk, uint64_t block);
  // Starts the write of the block's parts into `slot`, acquired for them,
  // into the block's copy where `into`; false, the parts given back
  // (giveSlotBack), when the tier refuses it.
  [[nodiscard]] bool startKvWrite(uint64_t block, std::shared_ptr<KvDiskSlot> slot, kv::Components parts, bool into,
                                  WriteClass write, bool unit = false);
  // The parts a write acquired into the block's copy (`into`) and did not
  // start go back, else the copy counts them taken and never written.
  void giveSlotBack(const std::shared_ptr<KvDiskSlot> &slot, kv::Components parts, bool into);
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
  // or active (a lookup holding a state keeps its block active) or the state
  // write in flight is of a block in it.
  [[nodiscard]] bool dropDiskSubtree(uint64_t block);
  // Nothing below a block whose read failed matches any more. Once none of
  // it is in transfer or active, it is erased with the states on it, in use
  // or not, and the poisoned block leaves once nothing refers to it: with its
  // last user, or here once a transfer that held it has landed.
  void dropPoisoned();
  // Starts the writes of the window parts of the points sparse history keeps
  // (windowsAhead_), each unit of their tails not yet under way, and lets go
  // of a point whose parts landed, that left, or one of whose units the tier
  // cannot take: the tails then take it for its units.
  void writeWindowsAhead();
  // The KV tier's part of pollTransfers().
  [[nodiscard]] bool pollTierTransfers();
  // A new KV copy of a block, a slot for each of its parts, replacing older
  // copies while the tier's limit for the draw has no room for them (the SSD
  // order, freeDiskSpace), and none while that limit cannot hold them;
  // inUse when a state in use needs the copy. A committed page's draw
  // replaces the spilled chain of a state in RAM too, once nothing else is
  // left. worth: what the copy is worth (freeDiskSpace).
  [[nodiscard]] std::shared_ptr<KvDiskSlot> acquireDiskSlot(kv::Components parts, model::DiskBudget::Draw draw,
                                                            bool inUse, bool ahead = false,
                                                            const std::shared_ptr<KvDiskSlot> &into = {},
                                                            Worth worth = {});
  // The SSD order, one for both tiers. A give-up's rank is what it loses;
  // within a rank the oldest goes first, a copy's age its conversation's
  // last use: the later of its block's or state's use and the last restore
  // through it (KvCache::refreshCopies, StateCache::refreshCopy).
  //   0  nothing: a window part or a K/V copy no point on the SSD needs
  //   1  a rewrite or a far go-back: history, and the copies of what memory
  //      holds, state or K/V, in one order
  //   2  a hit: a point's only copy, its state's (its chain's copies then
  //      need nothing: rank 0)
  //   3  a request: the only copy of a state in use
  // A room request takes only what is worth less than what it writes (its
  // Worth: the rank of what it writes and its last use, writeWorth):
  //   a copy ahead (persistent), of what  rank 0 and history
  //   memory holds
  //   a point's only copy (a demotion, a   ranks 0-1, and the points used
  //   state's write; rank 2)               before it (Worth::lastUse)
  //   history's copy (rank 1: what sparse  rank 0, and the rank-1 copies
  //   history writes)                      used before it
  //   a Spill lane's active KV (`active`), every point
  //   the limit as it shrinks (anyPoint)
  //   a copy in use (`inUse`)              rank 3 too
  // With nothing worth less left the write is refused, and its item leaves
  // without a copy (quotaRefused_). Never given up for a copy: the state the
  // resume point or a queued request restores (certainReuse), and so the K/V
  // copies it restores through, nor what a running request uses; only a
  // running lane's active KV takes them. A state on the SSD is history
  // (diskHistory) when a deeper state of its chain supersedes it and holds an
  // SSD copy itself. False when nothing the new copy may displace is left.
  // No dry sum checks ahead that the victims below the bound cover a write:
  // a conversation's chain on the SSD frees only after its tip has gone, so
  // a sum over the orders would see one copy where giving way frees the
  // chain.
  static constexpr uint64_t anyPoint = std::numeric_limits<uint64_t>::max();
  [[nodiscard]] bool freeDiskSpace(bool inUse, bool active, bool ahead = false, Worth worth = {});
  // freeDiskSpace's choice: the copy it gives up next.
  enum class DiskVictim : uint8_t { WindowPart, KvCopy, StateCopy, StateOnly, StateInUse };
  static_assert(static_cast<size_t>(DiskVictim::StateInUse) + 1 == std::tuple_size_v<decltype(KvTierSnapshot::quotaGivenUp)>);
  struct DiskReplacement final {
    DiskVictim kind;
    CacheEvictionCandidate at;
  };
  // Its answer: the next victim, or refused when a point is left and none
  // was used before what the write holds.
  struct DiskChoice final {
    std::optional<DiskReplacement> next;
    bool refused = false;
  };
  [[nodiscard]] DiskChoice nextDiskReplacement(bool inUse, bool active, bool ahead = false, Worth worth = {}) const;
  // History on the SSD: superseded, by a deeper state of its chain that
  // holds an SSD copy (freeDiskSpace's table).
  [[nodiscard]] bool diskHistory(uint64_t block) const;
  // A persistent tier's copy of a block a state lies at or below: a point
  // a restart takes back may restore through it.
  [[nodiscard]] bool copyUnderState(uint64_t block) const {
    return persistent_ && (states_.contains(block) || kv_.stateBelow(block));
  }
  // The oldest window part no state's disk copy needs (KvCache::
  // unneededWindowCandidate) that no lane's read holds (unitReads_).
  [[nodiscard]] std::optional<CacheEvictionCandidate> unneededWindow() const;
  // Evicts the state in RAM, not in use, of the oldest disk leaf holding
  // one, as a spilled leaf can: the leaf and the chain above it become
  // copies freeDiskSpace gives up. False when no disk leaf holds one.
  [[nodiscard]] bool evictStateOnDiskLeaf();
  // Revokes commitments, newest Spill request first, while active KV
  // exceeds its limit, then gives up copies not in use while cached copies
  // exceed what active KV leaves of the limit; returns how many copies went.
  uint32_t fitDiskLimit();
  // A persistent tier's label for the block's disk copy.
  void labelKv(uint64_t block, const std::shared_ptr<KvDiskSlot> &slot, kv::Components parts);
  // The parts a write gives the block's copy, of those `want` names (all by
  // default), a copy holding what its block holds: those its pages hold that
  // its own copy lacks, into it (`into`); beside a copy of another command's
  // K/V, all of them in a new copy.
  static constexpr kv::Components kAnyParts = 0xff;
  struct CopyParts final {
    kv::Components parts = 0;
    std::shared_ptr<KvDiskSlot> into;
  };
  [[nodiscard]] CopyParts copyParts(uint64_t block, kv::Components want = kAnyParts) const;
  // A state at the block is history: of the states below it that supersede
  // it, the nearest down each path, there is exactly one, and no running
  // request still resumes from the block (its chain ends before that state,
  // or its pages are on a path that holds none). With none below, it is a
  // conversation's newest; with two, conversations parted below it, and it
  // serves each of them. A prompt's head (noteHead) is history only behind
  // a deeper head alone, and a reply's end no prompt rendered through once
  // a prompt left its reply another way.
  [[nodiscard]] bool superseded(uint64_t block, const std::unordered_set<uint64_t> &gone = {}) const;
  // What a copy of the block takes of the quota once it has the parts of
  // `want` its pages hold: its slot's bytes and those parts' it lacks;
  // nothing for a disk-only block without a copy, which holds nothing to
  // write.
  [[nodiscard]] uint64_t copyBytes(uint64_t block, kv::Components want = kAnyParts) const;
  // The parts a point's copy gives the block at `index` of its chain of
  // `blocks`: the base part, and on a persistent tier, whose restart needs
  // it, the window part of a block of its tail.
  [[nodiscard]] kv::Components pointParts(size_t index, size_t blocks) const noexcept;
  // The point persist() copies: its chain, root first, how many of its
  // blocks from the root have a copy that landed, and what the chain's
  // copies take of the quota, those still to write by their pages' forms as
  // the cursor took the chain (a 245K-token chain is 7,656 blocks, walked
  // once and not at every step).
  struct PersistCursor final {
    uint64_t block = 0;
    std::vector<uint64_t> blocks;
    size_t landed = 0;
    uint64_t bytes = 0;
  };
  // Whether the restore point persist() copies fits the room it may take.
  // A persistent tier gives up older points for it as a full quota does
  // (freeDiskSpace), so the whole point must fit the limit: more would give
  // up every other copy first. A temporary tier's copies serve reclaims
  // alone, so it gives up no copy for a point, which may never leave RAM:
  // the redundant copies a full quota gives up first hold what reclaims take
  // before this point, and the only copies what RAM no longer holds. Its
  // room is what copies and Spill requests' commitments leave of the limit,
  // which every copy the point lacks must fit, below its frontier too, as a
  // full quota gives those up as well. Without a disk budget the tier's
  // slots bound it.
  [[nodiscard]] bool pointFits(const PersistCursor &point) const;
  void startRestore(uint64_t block);
  // Starts a read of the block's window part into `pages`, the unit of the
  // request that waits for it (UnitRead).
  void startUnitRead(uint64_t block, std::span<const uint32_t> pages, uint64_t requestId);
  // By form, whether evicting the walk's KV may empty an extent
  // (Walk::reaches): of the pages its class may reuse (reusablePages), the
  // `demoting` pages a unit found in flight and the runway the walk keeps
  // leave one. That counts the leaves that need a demotion too, which an
  // Immediate step passes over while the tier takes writes. A page written
  // through is none of them: its Spill request holds it, though under
  // --kv-spill always it frees once its copy lands (pendingPages). For a need
  // of pages (Walk::inPlace) the walk's form is in reach too while KV of it
  // can leave past its free pages and the pages all demotions in flight
  // bring, this take's included: each page that leaves meets the need in
  // place, as reach() counts it, with no extent emptied.
  [[nodiscard]] std::array<bool, kv::kPageForms> formsInReach(
      const Walk &walk, const std::array<uint32_t, kv::kPageForms> &demoting) const;
  // Evicting the resident leaf may lead to an extent of a form in `reach`
  // emptying: its page is of such a form, or the page of a block above it,
  // which may become a leaf once the leaf has gone, as a spilled chain's
  // Index pages above its Full tail do.
  [[nodiscard]] bool leadsToReach(uint64_t block, const std::array<bool, kv::kPageForms> &reach) const;
  // The extents `landing` pages of the form complete beside its free pages
  // past those the walk keeps (Walk::keptPages, which its claim takes in
  // place), past the runway it keeps.
  [[nodiscard]] uint32_t landingExtents(const Walk &walk, kv::PageForm form, uint32_t landing) const noexcept;
  // The bytes of the extents the pages of the demotions landing complete
  // (landingExtents); none for a walk that cannot wait. A reclaim gives
  // what it releases now and what it adds to this.
  [[nodiscard]] uint64_t landingFill(const Walk &walk) const noexcept;
  // For a shortfall of pages of the form in a cache that spills:
  // the empty extents of the other forms beyond the runway each keeps, and
  // for Index pages, when none is, a Full extent compaction empties (the Full
  // pages spills freed fund the next Index extent). Returns the bytes
  // released.
  uint64_t releaseOtherForms(kv::PageForm form);
  // Returns one extent of free pages: an empty one, or the one compactExtent
  // empties; progress with zero bytes when the extent emptied is the runway
  // kept. It leaves each form the runway the walk keeps, and an extent of
  // the form whose free pages it keeps goes only past them (Walk::releases).
  // No progress when the free pages fill none.
  [[nodiscard]] ReclaimResult releaseExtent(const Walk &walk);
  // Empties one extent that still holds pages, of the form or of any
  // (KvPool::compactExtent); the blocks and requests on its pages follow
  // them. A page a transfer reads or writes stays where it is until the
  // transfer has landed. False when the free pages cover no extent.
  [[nodiscard]] bool compactExtent(std::optional<kv::PageForm> form = std::nullopt);
  // The pages transfers in flight read or write.
  [[nodiscard]] std::vector<uint32_t> pagesInTransfer() const;
  // The blocks and requests on moved pages follow them, and the requests'
  // page lists change from the first one moved.
  void followMoves(const KvPageMoves &moves);
  // The sources of the request's pages from its last cut's first change
  // on, beside the chunks before it; null without an Index page.
  [[nodiscard]] std::shared_ptr<const model::KvPageSources>
  cutSources(const Request &active) const;

  // Spilling (a cache with a spill policy). A page a request commits quota
  // for is written whole and takes a slot, as each copy its chain already
  // has does.
  [[nodiscard]] uint64_t spillSlotBytes() const { return tier_->slotBytes(parts_); }
  [[nodiscard]] uint64_t heldSlotBytes() const noexcept { return kv_.heldCopyBytes(); }
  // A Spill request's commitment: locks quota for `pages` slots; false,
  // committing nothing, when active KV's limit cannot hold them beside
  // `unheld` copies it is about to hold.
  [[nodiscard]] bool commit(Request &active, uint32_t pages, uint32_t unheld = 0);
  // A copy no request holds yet (ChainCopies::unheld).
  [[nodiscard]] bool takesQuota(uint64_t block) const;
  // Whether the request holds the block at this index of its chain on the
  // block's own page, with a copy of its K/V (Request::heldCopies).
  [[nodiscard]] bool holdsCopy(const Request &active, uint32_t index) const;
  // A slot drawn for the request came back unwritten: it commits the slot
  // again, or, while active KV's limit cannot hold it, counts the
  // commitment revoked.
  void recommit(Request &active);
  // Returns `pages` slots of the request's commitment.
  void returnCommitment(Request &active, uint32_t pages) noexcept;
  // The request owes a copy of each of these blocks that has none.
  void oweCopies(Request &active, std::span<const uint64_t> blocks);
  // Starts writing the copies the request owes, each on its commitment or,
  // past it, on a slot it commits then. False once the tier's share, the
  // volume or the file stops every request's writes; true when only the
  // active limit's room for this request's next commitment ran out.
  [[nodiscard]] bool writeThrough(Request &active, uint32_t &started);
  // Whether a Full block whose copy has landed may move to the index form:
  // it lies past the dense history, and requests hold it, all of them
  // spilling and outside their tail windows (Eligible). Later: one holds it
  // in its window, which moves on as it commits, or the rule does not let a
  // release take it. KeptFull while a holder reads it Full, a Resident one
  // or its dense history, and without holders.
  enum class Eligibility : uint8_t { Eligible, Later, KeptFull };
  [[nodiscard]] Eligibility spillEligibility(uint64_t block, const ReleaseRule &rule = {}) const;
  // Writes of the class in flight.
  [[nodiscard]] uint32_t writes(WriteClass write) const noexcept;
  // The pages a reclaim write frees as it lands, at most: a unit write's
  // unit, another's block.
  [[nodiscard]] uint32_t writePages(const Write &write) const;
  // Pages Spill requests hold, written through or copied ahead, whose Full
  // pages free as their copies land under --kv-spill always (spillsOnLanding).
  [[nodiscard]] uint32_t spillingWrites() const noexcept;
  // Whether a write's page spills once it lands: under --kv-spill always, a
  // page a Spill request holds past its dense history and tail window,
  // whatever wrote it but a demotion. Otherwise none does: a lane's pages
  // stay Full until a claim releases them.
  [[nodiscard]] bool spillsOnLanding(const Write &write) const;
  // Whether a demotion of the block frees its page as its copy lands: no
  // state in RAM keeps the page (pollTierTransfers), no request holds it,
  // and each resident child leaves by a demotion of `freeing`, started
  // before it. Those children's copies land first (pollTierTransfers), so a
  // run of demotions up a chain frees every page of the run.
  [[nodiscard]] bool freesOnLanding(uint64_t block, const std::unordered_set<uint64_t> &freeing) const;
  // The blocks of the demotions in flight that free their pages as their
  // copies land (freesOnLanding), each judged on those started before it.
  [[nodiscard]] std::unordered_set<uint64_t> freeingDemotions() const;
  // The pages the demotions in flight hold, by form.
  [[nodiscard]] std::array<uint32_t, kv::kPageForms> demotionPages() const noexcept;
  // The bytes of the whole extents `pages` pages add to the free pages of
  // their form and `beside` more pages that free before them (Returning),
  // past the empty one a runway keeps with keepRunway.
  [[nodiscard]] uint64_t extentsCompleted(kv::PageForm form, uint32_t pages, uint32_t beside = 0,
                                          bool keepRunway = false) const noexcept;
  // Returning's count of the requests' writes and owed copies, or of every
  // request's without them.
  [[nodiscard]] Returning returningOf(std::optional<std::span<const uint64_t>> requestIds) const;
  // Rescans every Spill request's blocks from its dense history: a holder
  // that kept some of them Full ended or began to spill.
  void rescanSpills() noexcept;
  // The Spill requests' pages past their dense history and window whose
  // copies landed (spillEligibility), each once and in page order.
  [[nodiscard]] std::vector<uint32_t> landedPages();
  // The request's `most` oldest of them the rule lets a release take
  // (spillEligibility); its scan mark (Request::spillScan) passes the pages
  // that settle.
  [[nodiscard]] std::vector<uint32_t> landedPages(Request &active, uint32_t most, const ReleaseRule &rule);
  // Calls `each` with the index and block of each page releasablePages
  // finds of the request with `count`, oldest first, and whether its copy
  // landed, while `each` returns true.
  void forReleasable(const Request &active, const ReleaseRule &rule, const ReleaseCount &count,
                     const std::function<bool(uint32_t index, uint64_t block, bool clean)> &each) const;
  // The requests whose chains hold `block` at `index`: a release that moves
  // its page gives each of them Index pages.
  [[nodiscard]] std::vector<uint64_t> holdersAt(uint32_t index, uint64_t block) const;
  // Moves the pages `landed` finds to the index form as spillWritten says,
  // the Full extents they empty leaving past the `claimed` free Full pages a
  // claim takes in place, or without `extents` only into free Index pages,
  // which allocate nothing: those past them stay Full, and so do the Full
  // extents they empty.
  [[nodiscard]] TokenAdmission moveLandedPages(const std::function<std::vector<uint32_t>()> &landed,
                                               std::optional<SpilledExtents> extents,
                                               const std::function<void(uint64_t bytes)> &funds,
                                               uint32_t claimed = 0);
  // The copy of the K/V on an Index page.
  [[nodiscard]] model::KvPageSource sourceOf(uint32_t page) const;
  KvPool &pool_;
  KvTier *tier_;
  // The parts of every copy: each of the tier's components.
  kv::Components parts_ = 0;
  // The serial of the last copy labelled, which every later one exceeds,
  // continued after those an earlier process recorded (KvSlotLabel).
  uint64_t copySerial_ = 0;
  std::optional<CacheSpillPolicy> spill_;
  bool persistent_;
  std::optional<CacheWindowPolicy> window_;
  uint64_t refusedStates_ = 0;
  mutable uint64_t searchVisits_ = 0;
  // The answers of superseded() within one count of releases
  // (releasedUnits), against the set of states its releases evicted
  // (`gone`): the count asks of each candidate at each of its steps, which
  // asked afresh is quadratic in what the cache holds. A state the count
  // evicts changes the answer of the states above it up to the nearest
  // that ends their searches (stopsSearch); of each, where that walk ended
  // and the states it passed, so that the next walk through it goes on from
  // there.
  struct SupersededMemo final {
    struct Walked final {
      uint64_t stop = 0;
      std::vector<uint64_t> passed;
    };
    const std::unordered_set<uint64_t> *gone = nullptr;
    std::unordered_set<uint64_t> seen;
    std::unordered_map<uint64_t, bool> answers;
    std::unordered_map<uint64_t, Walked> walked;
    // The states gone and those their walks passed: any other's answer is
    // the one the cache keeps against no state gone (history_).
    std::unordered_set<uint64_t> passed;
    // The states whose RAM copy a release of the count took while their disk
    // copies stay: from then on points only on the SSD, as the walk sees
    // them once that release is taken (tailCandidates offers them so).
    std::unordered_set<uint64_t> demoted;
    // Of each block asked, whether the base part of every block of its chain
    // has landed (pointLanded), which does not change within a count.
    std::unordered_map<uint64_t, bool> chainLanded;
    // Of each point asked, whether its tail's window parts have landed
    // (tailLanded), which does not change within a count either.
    std::unordered_map<uint64_t, bool> tailLanded;
    // Of each block asked, the point sparse history spaces its writes from
    // (copiedBelow).
    std::unordered_map<uint64_t, uint64_t> copied;
  };
  // A release of the count in hand took the state's RAM copy and kept its
  // disk copy (SupersededMemo::demoted).
  [[nodiscard]] bool countDemoted(uint64_t block) const {
    return supersededMemo_ && supersededMemo_->demoted.contains(block);
  }
  // Takes the memo's answers back for the states `gone` gained since.
  void followGone(SupersededMemo &memo, const std::unordered_set<uint64_t> &gone) const;
  // The base part of every block of the chain ending at the block has
  // landed.
  [[nodiscard]] bool chainLanded(uint64_t block) const;
  // A state below the block that is not of `gone`.
  [[nodiscard]] bool stateBelow(uint64_t block, const std::unordered_set<uint64_t> &gone) const;
  // An ordinary state that ends every search of superseded() that meets it,
  // whatever it starts from.
  [[nodiscard]] bool stopsSearch(uint64_t block, const std::unordered_set<uint64_t> &gone) const;
  mutable SupersededMemo *supersededMemo_ = nullptr;
  bool countsBounded_ = true;
  bool checkKept_ = false;
  // Each answer the cache keeps is the one asked afresh now
  // (std::logic_error otherwise).
  void checkKept() const;
  bool checkResumes_ = false;
  bool checkTails_ = false;
  bool checkReleases_ = false;
  // Where the next walk of the take in hand may start (takeOldest): the
  // cursor of the leaf the last walk took, for the same walk and forms in
  // reach, its parent and the states then held.
  struct WalkResume final {
    Walk walk;
    std::optional<std::array<bool, kv::kPageForms>> reach;
    WalkCursor cursor;
    uint64_t parent = 0;
    size_t states = 0;
    uint64_t disk = 0;
  };
  std::optional<WalkResume> resume_;
  // The pages a take's next step may still free within its unit's count:
  // a walk takes no unit that could free more (takeOldest).
  std::optional<uint64_t> stepBudget_;
  // A device need in place's steps past its first keep every page they free
  // (releaseExtent releases none, landingFill credits none): its budget is
  // in pages, and bytes past its first step no count bounded.
  bool keepsExtents_ = false;
  // Calls that may change what the SSD holds or start a write: room made
  // for a copy (freeDiskSpace), a write started (startKvWrite).
  uint64_t diskCalls_ = 0;
  // Holds a memo for what it spans, a peek or a count, where none is held:
  // the cache does not change meanwhile.
  struct MemoScope final {
    const Cache &cache;
    SupersededMemo memo;
    bool owns;
    explicit MemoScope(const Cache &owner) : cache(owner), owns(!owner.supersededMemo_) {
      if (owns)
        cache.supersededMemo_ = &memo;
    }
    ~MemoScope() {
      if (owns)
        cache.supersededMemo_ = nullptr;
    }
    MemoScope(const MemoScope &) = delete;
    MemoScope &operator=(const MemoScope &) = delete;
  };
  [[nodiscard]] bool supersededNow(uint64_t block, const std::unordered_set<uint64_t> &gone) const;
  // The answers of superseded() against no state gone, kept until a change
  // the cache records at a block (KvCache::historyChanged) can make one
  // differ: before an answer is read, dropKept() walks up from each block
  // changed and drops those of the states whose searches reach it. A
  // reply's end no prompt rendered through is not kept: its answer reads
  // the blocks beside its own (leftBehind), not those below it.
  mutable std::unordered_map<uint64_t, bool> history_;
  // Kept the same way: whether a state's tail holds a unit (tailHoldsUnit).
  mutable std::unordered_map<uint64_t, bool> tailUnits_;
  // The nearest point of the block's chain below it, within sparse
  // history's spacing of it, whose copy stays (copyStays), through those
  // that leave without one (0 for none): what sparse history spaces its
  // writes from.
  [[nodiscard]] uint64_t copiedBelow(uint64_t block, const std::unordered_set<uint64_t> &gone = {}) const;
  void dropKept() const;
  [[nodiscard]] bool keptHistory(uint64_t block) const;
  // A block of the state's tail holds a unit.
  [[nodiscard]] bool tailHoldsUnit(uint64_t block) const;
  // A reply's end no prompt has rendered through: a held reply, a leaf, no
  // request's chain ends there.
  [[nodiscard]] bool unrenderedReply(uint64_t block) const;
  uint64_t unitsReleased_ = 0;
  // In this take the quota gave up a state for a write's room, or in a
  // persistent tier a copy under a state, which a landed point needs: a
  // point the give-up leaves history may go next with no write, which no
  // count foresees (what a full quota gives up is known only as the write
  // makes room), and the releases with no write after it in the take count
  // with the quota's (unitsFreedElsewhere; take windowed seed 144893).
  // Nothing is lost: the point is history by the rules, and the next count
  // offers it.
  bool quotaGaveUpState_ = false;
  // The points sparse history keeps whose tail's window parts their release
  // left to be written (keptLeaving): written ahead of any walk as transfers
  // are polled (writeWindowsAhead), each kept while they are queued or in
  // flight, until they land or the tier cannot take one.
  std::unordered_set<uint64_t> windowsAhead_;
  uint32_t sparseHistoryTokens_ = kSparseHistoryTokens;
  CacheAdoption adoption_;
  std::shared_ptr<model::DiskBudget> diskBudget_;
  // A copy ahead found the quota full of only copies, and none has found
  // room since (freeDiskSpace's console line).
  bool quotaFullAhead_ = false;
  // The last room request was refused: what was left was worth no less than
  // what it would write (freeDiskSpace). Its item leaves without a copy.
  bool quotaRefused_ = false;
  // Blocks whose copy went while their page stayed, until written again.
  std::unordered_set<uint64_t> copiesGivenUp_;
  // takeBrokenCopies().
  std::vector<uint64_t> brokenCopies_;
  std::function<void()> diskMeasure_;
  CacheRecency recency_;
  KvCache kv_;
  StateCache states_;
  std::unordered_map<uint64_t, Request> requests_;
  // The chains requests waiting to start restore (setWantedChains).
  RestoredChains wanted_;
  // Window parts being read into lanes' unit pages (UnitRead).
  std::vector<UnitRead> unitReads_;
  // Writes in flight, in the order they started. Only reclaim demotions,
  // and under --kv-spill always writes of pages that spill once they land,
  // free memory a reclaim waits for (pendingPages); the other copies
  // persist() keeps hold only their blocks.
  std::vector<Write> writes_;
  PersistCursor persisting_;
  // Write-through waits after a write found the volume full, until the
  // active limit holds another slot beside the copies of the KV requests
  // hold.
  bool writeThroughPaused_ = false;
  // The order requests began to spill in (Request::spillOrder).
  uint64_t spillSequence_ = 0;
  // Block IDs increase from parent to child. Restores start in that order so
  // cancellation can discard an unread suffix without stranding its parents.
  std::map<uint64_t, Restore> restores_;
  // Blocks whose read failed, until they have left.
  std::vector<uint64_t> poisoned_;
  KvTierSnapshot kvTier_;
  // The spill counters of spillSnapshot(), and the restores' latencies.
  KvSpillSnapshot spilled_;
  model::LatencyHistogram restoreLatency_;
  CacheLookupSnapshot lookup_;
  double extentCompactMaxMilliseconds_ = 0.0;
  std::function<void()> completionNotifier_;
};

} // namespace splash::engine
