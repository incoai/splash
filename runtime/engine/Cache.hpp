#pragma once

#include "AwakeClock.hpp"
#include "engine/KvCache.hpp"
#include "engine/KvPool.hpp"
#include "engine/KvTier.hpp"
#include "engine/ReclaimClass.hpp"
#include "engine/StateCache.hpp"
#include "engine/Types.hpp"
#include "model/LatencyHistogram.hpp"
#include "model/Model.hpp"
#include "model/SlotFile.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
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

// What holds the disk tier's writes back: nothing, the write budget (states
// not in use wait for it; only when one is set), or a file that stopped
// taking writes.
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
  // What writes not in use may take now; zero once overdrawn, and zero
  // without a write budget, when nothing is held back.
  uint64_t writeBudgetBytes = 0;
  // How long the files' writes held back for reads a step waited for
  // (model::DiskBudget::writesDeferredMicroseconds).
  uint64_t writesDeferredMicroseconds = 0;
  uint64_t readBytes = 0;
  uint64_t writtenBytes = 0;
  uint64_t demotions = 0;
  uint64_t demotionFailures = 0;
  // Demotions the tier had no room for; the leaf stayed and the requester
  // waited.
  uint64_t demotionsRefused = 0;
  uint64_t restores = 0;
  uint64_t restoreFailures = 0;
  // KV pages a persistent tier wrote while they stayed in RAM (persist).
  uint64_t copies = 0;
  uint64_t copyFailures = 0;
  // Pages whose demotion is in flight.
  uint32_t pendingPages = 0;
  uint32_t diskBlocks = 0;
  uint64_t diskBytes = 0;
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
// form and back, and blocks written again once their copy had gone;
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
  uint64_t rewrites = 0;
  uint64_t indexRestores = 0;
  uint64_t wholeRestores = 0;
  model::LatencyStats restore;
  uint64_t activeLimitBytes = 0;
  uint64_t heldBytes = 0;
  uint64_t committedBytes = 0;
  uint64_t revokedCommitments = 0;
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
  // The form of the pages refused, which reclaimForPages makes room for.
  kv::PageForm form = kv::PageForm::Full;

  [[nodiscard]] bool granted() const noexcept {
    return failure == TokenAdmissionFailure::None;
  }
};

// The page list a command of a request reads (Cache::cutPageTable).
struct PageTableView final {
  std::span<const uint32_t> pages;
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
  uint64_t revision = 0;
  uint32_t firstChanged = 0;
};

// What a spill-capable target's cache keeps Full for each request that
// spills: the pages its rows read through page tables from the start (its
// dense history, SparseKvShape), and the last pages it committed before its
// open page (the tail window, at least its local rows').
struct CacheSpillPolicy final {
  uint32_t densePages = 0;
  uint32_t windowPages = 0;
};

// Whether a request may start to spill now, or what holds it back: the KV
// tier takes no writes (it is off, or a file stopped taking them), or the
// write budget does not cover the writes Spill requests have committed.
enum class SpillGate : uint8_t { Open, TierClosed, WriteBudget };

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
  // The step evicted KV, whose pages are free for other KV, or will be once
  // their demotion lands. Only reclaimForMemory reports it: its progress is
  // memory that came back, which KV gives only as an extent it empties.
  bool evictedKv = false;
};

// What a reclaim step does with memory. KeepExtents: extents stay
// allocated, and the pages a victim frees are reused. ReleaseExtents: free
// pages return as extents first, and an extent a victim empties is
// released. ReusePages, while the host refuses growth: extents stay, and
// only what gives pages goes, KV leaves, and a state only where it sits on
// the leaf that goes next; its buffers would give no page otherwise.
enum class CacheReclaimMode { KeepExtents, ReleaseExtents, ReusePages };

// What becomes of the Full extents a spill's moves leave empty
// (Cache::spillWritten): they stay allocated for a reclaim to take; they
// leave memory but one, the runway the next Full pages come from without
// an allocation; or the Full pages left gather in the fewest extents
// (Cache::compactExtent) and every empty one leaves.
enum class SpilledExtents : uint8_t { Kept, KeepRunway, Released };

// Whether the caller can wait for KV memory a step leaves to come back.
// CanWait (growth and pressure passes): a KV leaf a state needs is written
// first, and its page returns when the copy lands. Immediate (a state's
// buffers, taken between commands): only leaves whose page frees now go,
// through their disk copy or erased; a leaf that needs a demotion is passed
// over. What happens to a state is the caller's StateCache::Unwritten.
enum class ReclaimTiming : uint8_t { CanWait, Immediate };

// The class of a state a command captures, which bounds what is taken for
// its buffers (Cache::reclaimOneState): an optional checkpoint, an ordinary
// state, or the prompt's replay point, which its conversation's next turn
// resumes from and is in use.
enum class StateClass : uint8_t { Checkpoint, Ordinary, InUse };

// What one step of room for a state's buffers gave: a recycled state's
// buffers, which the reservation takes as they are, or the bytes of a
// released extent, which the reservation has to allocate and which may be
// fewer than a state needs.
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
  // With a spill policy, on a layout that can spill (kv::Layout::spillable)
  // and with a tier, requests may spill (KvMode::Spill), and the blocks
  // requests hold keep their copies (KvCache::holdPage); without one, quota
  // treats copies as it always has.
  Cache(KvPool &pool, KvTier *kvTier, std::shared_ptr<model::DiskBudget> diskBudget,
        std::optional<CacheSpillPolicy> spill = std::nullopt);
  Cache(const Cache &) = delete;
  Cache &operator=(const Cache &) = delete;

  [[nodiscard]] bool persistent() const noexcept { return persistent_; }
  // Before the first request, takes back the restore points an earlier
  // process left in a persistent tier: each state whose block's chain is
  // whole, and that chain. Everything else stays behind, and once the files
  // finish adoption its slots are free. Then copies go while the disk limit
  // is exceeded, as keepDiskLimit() gives them up. Ids and recency continue
  // after those taken back.
  [[nodiscard]] CacheAdoption adopt(std::vector<PersistedKv> blocks,
                                    std::vector<PersistedState> states);
  // One step of making the restore point at this block durable in a
  // persistent tier, while it stays in RAM: the state is written from its
  // own buffers once the write budget admits it, the tier busy until then,
  // and then the pages of its chain that have no disk copy, root first,
  // beside the commands that may read them; they pass the budget, as every
  // KV write does. Nothing starts while a restore waits, and the pages take
  // a small share of the tier's writes. A point is unneeded once it is gone
  // or an ordinary state on its only branch below supersedes it: its
  // conversation went on from there. One larger than what the tier may hold
  // now (its limit), or given up to make room for itself, is refused, and so
  // is every point once a file stopped taking writes.
  [[nodiscard]] PersistStatus persist(uint64_t block);
  // At a clean stop: the label of each state copy a persistent tier keeps
  // takes its state's current recency, which the next process orders them
  // by; a copy takes its state's class as soon as a checkpoint becomes
  // ordinary, which a crash keeps too. KV copies keep the labels of their
  // writes: a chain goes with the states that need it (dropOldestPoint).
  void relabelStates() const;
  // Measures the volume the disk tier lives on and sets the budget's limit
  // (DiskSpace::measure), for keepDiskLimit(); none without such a tier.
  void setDiskMeasure(std::function<void()> measure) { diskMeasure_ = std::move(measure); }
  // At a control pass, and when a write found the volume full: measures it
  // again, then gives up copies while the tier holds more than the limit,
  // in the order a full quota replaces them (freeDiskSpace), never a copy in
  // transfer or the only copy of a state in use.
  void keepDiskLimit();
  // When the write budget admits a state's write not in use again, on its
  // clock; empty while it does now or without a budget.
  [[nodiscard]] std::optional<double> diskWritesAdmittedAt() const noexcept {
    return diskBudget_ ? diskBudget_->admitsAgainAt() : std::nullopt;
  }
  // When the write budget covers what Spill requests committed again, so
  // that a request may start to spill (spillGate), on its clock; empty
  // while it does now or without a budget.
  [[nodiscard]] std::optional<double> spillWritesAdmittedAt() const noexcept {
    return diskBudget_ ? diskBudget_->admitsSpillAt(diskBudget_->committedBytes()) : std::nullopt;
  }
  // At a clean stop: the write budget holds nothing back any more.
  void liftDiskWriteBudget() noexcept {
    if (diskBudget_)
      diskBudget_->lift();
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
  // Commits quota for `pages` pages of a Spill request, which each page it
  // publishes then draws or returns (DiskBudget::commit); false, committing
  // nothing, when spillRoom() cannot hold them.
  [[nodiscard]] bool commitSpill(uint64_t requestId, uint32_t pages);
  // Turns a Resident request whose growth was denied into a Spill request:
  // checks the tier, the write budget and the room for `pages` pages,
  // commits them, starts write-through of its published blocks and moves
  // those whose copies landed already into free Index pages. It allocates
  // nothing; false, changing nothing, when spilling cannot relieve it.
  [[nodiscard]] bool convertToSpill(uint64_t requestId, uint32_t pages);
  // The blocks on the chain that ends at this block (none for 0) whose K/V
  // has a copy, landed or being written: pages of a Spill request that
  // holds them need no commitment.
  [[nodiscard]] uint32_t copiedBlocks(uint64_t block) const;
  // Starts writing Spill requests' published pages that have no copy of
  // what they hold (KvCache::kvCopied), as the tier's share for them allows;
  // returns the writes started.
  uint32_t writeThrough();
  // Between commands: moves the eligible pages whose copies landed to the
  // index form, into free Index pages, the runway's among them, and the
  // rest into extents of the index form it allocates; then refills the
  // runway, an extent's worth of free Index pages, which the next spill
  // takes without an allocation; and once it stands, the Full extents the
  // moves left empty go as `extents` says. Denied, of the index form, when
  // an allocation was refused: the pages it leaves Full, or the runway's;
  // the caller makes room for them or does without.
  [[nodiscard]] TokenAdmission spillWritten(SpilledExtents extents = SpilledExtents::Kept);
  // Pages of the form that transfers in flight will free: demotions, and
  // for Full the eligible write-throughs that an Index page will take.
  [[nodiscard]] uint32_t pendingPages(kv::PageForm form) const noexcept;
  [[nodiscard]] bool holdsIndexPages(uint64_t requestId) const;
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
  void promoteState(const CacheLookup &lookup, StateRestore &transfer);
  // Gives the request the matched chain up to its state. Disk-only blocks
  // get fresh pages and start their restores; the request waits on
  // kvRestoreStatus() before it runs. Pages are admitted like ensureTokens().
  // By the request's mode: a Spill request takes a disk-only block past the
  // dense history as an Index page and restores its index alone, and holds
  // spilled blocks as they are. A Resident request restores disk-only blocks
  // whole, and moves a spilled block back to a Full page and restores it
  // whole when no other request holds it and no transfer moves it; else it
  // holds it spilled too.
  [[nodiscard]] TokenAdmission restoreRequest(uint64_t requestId,
                                              const CacheLookup &lookup);
  [[nodiscard]] KvRestoreStatus kvRestoreStatus(uint64_t requestId) const;

  // Gives the request pages for its first tokenCount tokens, which the work
  // they are for writes below writtenTokens, all of them unless said
  // otherwise. A new page with a row the work does not write comes cleared:
  // attention weighs the rows past the request's end by zero, and a BF16
  // value is read as stored, so a non-finite one the page's previous holder
  // left there would reach the request as 0 x NaN.
  [[nodiscard]] TokenAdmission
  ensureTokens(uint64_t requestId, uint64_t tokenCount,
               uint64_t writtenTokens = std::numeric_limits<uint64_t>::max());
  // The request's page list as it is now, at the revision and first change
  // its next cut gives it.
  [[nodiscard]] PageTableState pageTable(uint64_t requestId) const;
  // The page list a command of the request reads, once per lane and
  // command: every change since the last cut makes one new revision, which
  // differs from the last from the first page any of them changed, with its
  // sources cut from there. A command reads only a cut list: another change
  // before the next cut stays in the same revision.
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
  // A state at this block is in use, published or yet to be: its writes
  // pass the disk tier's write budget.
  [[nodiscard]] bool stateInUse(uint64_t kvBlock) const noexcept { return states_.inUse(kvBlock); }
  [[nodiscard]] StateCheckpoint checkpointState(uint64_t kvBlock) const noexcept;
  // The state at this block has a RAM copy.
  [[nodiscard]] bool stateResident(uint64_t kvBlock) const noexcept;
  // False only while this exact disposable publication is pinned.
  bool retireCheckpointState(StateCheckpoint checkpoint) noexcept;
  // This exact disposable publication is pinned: retireCheckpointState
  // refuses it.
  [[nodiscard]] bool checkpointPinned(StateCheckpoint checkpoint) const noexcept {
    return states_.checkpointPinned(checkpoint);
  }

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
  // step instead of a retry per page. The pages are of the form, whose free
  // and pending pages count, and empty extents of the other form go first.
  // What is in use waits, and the step reports pending, only while a
  // transfer in flight can still cover the pages (returnsInFlight).
  [[nodiscard]] CacheReclaimResult reclaimForPages(uint32_t pages, CacheReclaimMode mode,
                                                   ReclaimClass upTo,
                                                   kv::PageForm form = kv::PageForm::Full);
  // The reclaim step for memory the governor funds that an allocation was
  // refused: a lane's state, or the model's memory for its reads of spilled
  // KV. It takes what reclaimOne(ReleaseExtents, upTo, false, keepRunway)
  // takes, in its order, until memory comes back: an extent's bytes
  // (reclaimedBytes), or a state's buffers, which the model's pool returns
  // (evictedState). A KV leaf gives back memory only as the extent its page
  // empties, so leaves go until one is released, and only while one can be
  // (extentWithinReach(upTo)); otherwise only a state goes, as
  // reclaimStateForLane takes it. It makes progress only once memory has
  // come back, since the KV it takes before then frees pages of extents that
  // stay, which it reports apart (evictedKv); it is pending as
  // reclaimForPages is, while a transfer in flight can still return that
  // memory.
  [[nodiscard]] CacheReclaimResult reclaimForMemory(ReclaimClass upTo, bool keepRunway);
  // The most pages of the form reclaimForPages(pages, mode, upTo, form) can
  // leave free: every allocated page of the form no request holds, but with
  // Ordinary not the idle KV that states in use restore through.
  [[nodiscard]] uint32_t reusablePages(ReclaimClass upTo, kv::PageForm form = kv::PageForm::Full) const;
  // One step of room for the buffers of a state, which a command will
  // capture (Model::reserveState) or which is taken from its lane after the
  // command that ended at it (Model::snapshot), taking nothing of a class
  // above the state's; the disk tier keeps a state it admits. A checkpoint
  // takes one checkpoint. An ordinary state takes what reclaimOne() takes of
  // the ordinary class, in its order, but only what frees memory now
  // (ReclaimTiming::Immediate), since the reservation follows at once: an
  // extent of free pages, a checkpoint, KV no state restores through, then
  // the oldest of ordinary states and KV leaves. One in use goes on to the
  // oldest state in use, last. KV frees memory only as an extent it empties,
  // released before the call returns, so KV goes only while evicting it can
  // empty one (extentWithinReach); otherwise the oldest ordinary state goes.
  // Only a state in use drops a victim whose write must wait for the write
  // in flight: refused, it could reach the disk tier only through that write
  // slot, once its command ends. Other states leave such a victim and move
  // on, and a state in use is never dropped so: while the write in flight
  // holds it, nothing goes. Like every release, that needs no command in
  // flight; the engine makes room for a state before it submits the command
  // that captures it, or once the command that ended at it has completed.
  // After a step that gave extent bytes the caller steps again while its
  // reservation does not fit and the bytes given are less than one state's;
  // after a step that gave a state it does not, so a reservation refused for
  // another reason costs one state or one state's worth of extents at most.
  // While the engine may not grow (`growth` false) an extent's bytes are of
  // no use to a reservation: extents and KV stay, and room comes from states
  // alone.
  [[nodiscard]] StateRoom reclaimOneState(StateClass state, bool growth);
  // Recycles one unpinned state, preferring checkpoints, for a lane that
  // takes the state's buffers, as a lane's state does while the host
  // refuses growth, taking nothing of a class above upTo. For running work
  // (InUse) a state in use goes once no other state is left and no transfer
  // in flight can return what is needed first (returnsInFlight: no page
  // covers a state's buffers); a start that a resident lane holds back
  // (Ordinary) takes nothing in use. A state the tier could take once the
  // write in flight has finished stays and is reported pending, as in
  // reclaimOne. evictableStates(upTo) are those it can take.
  [[nodiscard]] CacheReclaimResult reclaimStateForLane(ReclaimClass upTo);
  [[nodiscard]] uint32_t evictableStates(ReclaimClass upTo) const noexcept {
    return states_.evictableStates(upTo == ReclaimClass::InUse);
  }
  [[nodiscard]] CacheSnapshot snapshot() const;
  [[nodiscard]] KvSpillSnapshot spillSnapshot() const;
  // Tests, between commands: std::logic_error unless the blocks, pages,
  // references and quota keep every invariant spilling relies on.
  void checkInvariants() const;

private:
  // Matches the prompt's pages on from blocks.back() (from the root when
  // empty) and appends each block found; returns the pages it hashed.
  size_t extendMatch(std::vector<uint64_t> &blocks, std::span<const uint32_t> prompt,
                     PromptIdentity identity) const;
  // The tokens up to the deepest of these blocks that holds a state.
  [[nodiscard]] uint32_t stateTokens(std::span<const uint64_t> blocks) const;

  struct Request final {
    std::vector<uint32_t> pages;
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
    // holder (spillWritten).
    uint32_t committedPages = 0;
    uint64_t spillOrder = 0;
    std::deque<uint64_t> owedCopies;
    uint32_t spillScan = 0;
    uint32_t pendingRestores = 0;
    bool restoreFailed = false;
  };
  // Records a change of the request's page list from index first on.
  static void pagesChanged(Request &active, uint32_t first) noexcept;
  // A page being written to the tier, and what for (WriteClass): a reclaim
  // demotion, whose page frees once the copy lands; a copy persist() keeps
  // of a page that stays; or a Spill request's page written through, which
  // stays Full until spillWritten moves it.
  struct Write final {
    uint64_t block = 0;
    WriteClass write = WriteClass::Reclaim;
    std::unique_ptr<KvTransfer> transfer;
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

  [[nodiscard]] TokenAdmission admitPages(uint32_t count, std::vector<uint32_t> &pages,
                                          kv::PageForm form = kv::PageForm::Full);
  // What an allocation that was refused lacks, as the free pages that cover
  // it: the KV pages it lacks, of their form; for memory the governor funds,
  // an extent's worth of Full pages, whose extent then leaves memory; none
  // for a state's buffers while the host refuses growth, which only a state
  // gives.
  struct Shortfall final {
    uint32_t pages = 0;
    kv::PageForm form = kv::PageForm::Full;
  };
  // Whether a transfer in flight can still cover a shortfall once it lands.
  // A restore, a demotion, a persistence write and the state write each
  // hold back what a reclaim would take then: the block a restore fills or a
  // write copies, the page a demotion frees and the leaf above it, the
  // states that wait for the write. A Spill request's page written through
  // holds back no cached block: it counts only by the Full page it frees
  // once it spills (pendingPages), where those pages and the free ones cover
  // the shortfall. Without spilling this is transfersInFlight().
  [[nodiscard]] bool returnsInFlight(const Shortfall &shortfall) const noexcept;
  // One eviction step: what it takes, as reclaimOne's arguments say, and
  // the allocation that was refused, if it is taken for one rather than for
  // a pass.
  struct EvictionStep final {
    CacheReclaimMode mode;
    ReclaimClass upTo;
    bool keepResumePoint = false;
    std::optional<Shortfall> refused;
  };
  // One eviction: checkpoints first, then the ordinary class's
  // reclaimOldest, then, up to InUse, that of what is in use, as reclaimOne
  // describes. ReusePages skips the checkpoints and scans for pages. What is
  // in use waits, and a step that finds nothing reports pending, while a
  // transfer in flight can return what is needed first: one that can still
  // cover the refused allocation's shortfall (returnsInFlight), or for a
  // pass any (transfersInFlight).
  [[nodiscard]] CacheReclaimResult evictOne(const EvictionStep &step);
  // reclaimStateForLane's step for a shortfall: what is in use waits, and
  // finding nothing is pending, while a transfer in flight can still cover
  // it.
  [[nodiscard]] CacheReclaimResult recycleState(ReclaimClass upTo, const Shortfall &shortfall);
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
  // dropped only once a file stopped taking writes, together with that
  // subtree. Immediate passes over a leaf that needs a demotion.
  [[nodiscard]] LeafReclaim reclaimKvLeaf(uint64_t block, ReclaimTiming timing);
  [[nodiscard]] LeafReclaim demoteKv(uint64_t block);
  // A file that stopped taking writes closes the KV tier; existing copies
  // stay readable. KV writes pass the write budget (model::DiskBudget).
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
  // The KV tier's part of pollTransfers().
  [[nodiscard]] bool pollTierTransfers();
  // A slot for a new KV copy, replacing older copies while the tier is at its
  // limit for the draw, and none while that limit cannot hold one; inUse
  // when a state in use needs the copy. A committed page's draw replaces the
  // spilled chain of a state in RAM too, once nothing else is left.
  [[nodiscard]] std::shared_ptr<KvDiskSlot> acquireDiskSlot(model::DiskBudget::Draw draw,
                                                            bool inUse);
  // Gives up one disk copy for a new copy, in use or not: the oldest
  // redundant one, KV or state, else the oldest that is the only copy, never
  // the KV of a state in RAM. A persistent tier gives up restore points
  // instead (dropOldestPoint). The only copy of a state in use, and the KV
  // it restores through, go only for a copy in use, and after every other.
  // False when the disk holds nothing the new copy may displace.
  [[nodiscard]] bool freeDiskSpace(bool inUse);
  // Evicts the state in RAM, not in use, of the oldest disk leaf holding
  // one, as a spilled leaf can: the leaf and the chain above it become
  // copies freeDiskSpace gives up. False when no disk leaf holds one.
  [[nodiscard]] bool evictStateOnDiskLeaf();
  // Revokes commitments, newest Spill request first, while active KV
  // exceeds its limit, then gives up copies not in use while cached copies
  // exceed what active KV leaves of the limit; returns how many copies went.
  uint32_t fitDiskLimit();
  // In a persistent tier a redundant copy is what the next process takes
  // back, and the unit worth keeping is a restore point, a state on disk and
  // its chain. A copy no state needs goes first (dropUnneededCopy), then the
  // oldest state copy, redundant or not: the copies of its chain that no
  // other state needs become unneeded and go on later calls, so a chain
  // stays whole while a state on disk needs it. A state in RAM needs only
  // the copies that hold the only K/V of its chain's blocks: a redundant
  // copy goes once no state on disk restores through it. The only copy of a
  // state in use goes only for a copy in use, and last. False when no state
  // copy can go.
  [[nodiscard]] bool dropOldestPoint(bool inUse);
  [[nodiscard]] bool dropUnneededCopy();
  // A persistent tier's label for the block's disk copy.
  void labelKv(uint64_t block, const std::shared_ptr<KvDiskSlot> &slot);
  // Ordinary states below the block sit on exactly one of its branches.
  [[nodiscard]] bool superseded(uint64_t block) const;
  // An ordinary state at the block or anywhere below it.
  [[nodiscard]] bool holdsOrdinaryState(uint64_t block) const;
  void startRestore(uint64_t block);
  // Evicting KV of the class may empty an extent: the pages upTo may reuse
  // (reusablePages), less those whose demotion is in flight, cover one. That
  // counts the leaves that need a demotion too, which an Immediate step
  // passes over while the tier takes writes. A page written through is none
  // of them: its Spill request holds it, though it frees once its copy lands
  // (pendingPages).
  [[nodiscard]] bool extentWithinReach(ReclaimClass upTo = ReclaimClass::Ordinary) const;
  // Returns one extent of free pages: an empty one, or the one compactExtent
  // empties; progress with zero bytes when the extent emptied is the runway
  // kept. No progress when the free pages fill none.
  [[nodiscard]] CacheReclaimResult releaseExtent(bool keepRunway);
  // Empties one extent that still holds pages, of the form or of either
  // (KvPool::compactExtent); the blocks and requests on its pages follow
  // them. A page a transfer reads or writes stays where it is until the
  // transfer has landed. False when the free pages cover no extent.
  [[nodiscard]] bool compactExtent(std::optional<kv::PageForm> form = std::nullopt);
  // The blocks and requests on moved pages follow them, and the requests'
  // page lists change from the first one moved.
  void followMoves(const KvPageMoves &moves);
  // The sources of the request's pages from its last cut's first change
  // on, beside the chunks before it; null without an Index page.
  [[nodiscard]] std::shared_ptr<const model::KvPageSources>
  cutSources(const Request &active) const;

  // Spilling (a cache with a spill policy).
  [[nodiscard]] uint64_t heldSlotBytes() const noexcept {
    return tier_ ? uint64_t{kv_.heldCopies()} * tier_->slotBytes() : 0;
  }
  // A Spill request's commitment: locks quota for `pages` slots; false,
  // committing nothing, when active KV's limit cannot hold them.
  [[nodiscard]] bool commit(Request &active, uint32_t pages);
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
  // in its window, which moves on as it commits. KeptFull while a holder
  // reads it Full, a Resident one or its dense history, and without
  // holders.
  enum class Eligibility : uint8_t { Eligible, Later, KeptFull };
  [[nodiscard]] Eligibility spillEligibility(uint64_t block) const;
  // Writes of the class in flight.
  [[nodiscard]] uint32_t writes(WriteClass write) const noexcept;
  // Pages written through whose Full pages free once their copies land.
  [[nodiscard]] uint32_t spillingWrites() const noexcept;
  // Rescans every Spill request's blocks from its dense history: a holder
  // that kept some of them Full ended or began to spill.
  void rescanSpills() noexcept;
  // spillWritten(extents), or without `allocate` only its moves into free
  // Index pages, which allocate nothing: those past them stay Full, and so
  // do the Full extents they empty.
  [[nodiscard]] TokenAdmission moveLandedPages(bool allocate, SpilledExtents extents = SpilledExtents::Kept);
  // The copy of the K/V on an Index page.
  [[nodiscard]] model::KvPageSource sourceOf(uint32_t page) const;

  KvPool &pool_;
  KvTier *tier_;
  std::optional<CacheSpillPolicy> spill_;
  bool persistent_;
  CacheAdoption adoption_;
  std::shared_ptr<model::DiskBudget> diskBudget_;
  std::function<void()> diskMeasure_;
  CacheRecency recency_;
  KvCache kv_;
  StateCache states_;
  std::unordered_map<uint64_t, Request> requests_;
  // Writes in flight, in the order they started. Only reclaim demotions,
  // and writes through whose pages will spill, free memory a reclaim waits
  // for (pendingPages); copies persist() keeps hold only their blocks.
  std::vector<Write> writes_;
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
