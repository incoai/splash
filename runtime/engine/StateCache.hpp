#pragma once

#include "engine/CacheRecency.hpp"
#include "engine/KvCache.hpp"
#include "engine/RecencyOrder.hpp"
#include "model/Model.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splash::engine {

class StateCache;

struct StateCheckpoint final {
  uint64_t kvBlock = 0;
  uint64_t publication = 0;
  [[nodiscard]] explicit operator bool() const noexcept { return kvBlock != 0; }
};

class CompositeStateLease final {
public:
  CompositeStateLease(const CompositeStateLease &) = delete;
  CompositeStateLease &operator=(const CompositeStateLease &) = delete;
  CompositeStateLease(CompositeStateLease &&other) noexcept;
  CompositeStateLease &operator=(CompositeStateLease &&other) noexcept;
  ~CompositeStateLease() noexcept;

  [[nodiscard]] uint64_t kvBlock() const noexcept { return kvBlock_; }
  [[nodiscard]] uint32_t boundary() const noexcept { return boundary_; }
  [[nodiscard]] const std::shared_ptr<const CompositeState> &
  state() const noexcept {
    return state_;
  }

private:
  friend class StateCache;
  CompositeStateLease(StateCache &owner, uint64_t kvBlock, uint32_t boundary,
                      std::shared_ptr<const CompositeState> state) noexcept;
  void reset() noexcept;

  StateCache *owner_ = nullptr;
  uint64_t kvBlock_ = 0;
  uint32_t boundary_ = 0;
  std::shared_ptr<const CompositeState> state_;
};

// Marks the block whose state an unfinished request's conversation resumes
// from; see StateCache::useState.
class StateUse final {
public:
  StateUse() = default;
  StateUse(const StateUse &) = delete;
  StateUse &operator=(const StateUse &) = delete;
  StateUse(StateUse &&other) noexcept;
  StateUse &operator=(StateUse &&other) noexcept;
  ~StateUse() noexcept;

  void reset() noexcept;

private:
  friend class StateCache;
  StateUse(StateCache &owner, uint64_t kvBlock) noexcept : owner_(&owner), kvBlock_(kvBlock) {}

  StateCache *owner_ = nullptr;
  uint64_t kvBlock_ = 0;
};

struct StateCacheSnapshot {
  uint32_t entries = 0;
  uint32_t pinned = 0;
  uint64_t bytes = 0;
  uint64_t diskBytes = 0;
  uint64_t offloads = 0;
  uint64_t offloadFailures = 0;
  uint64_t invalidations = 0;
  uint64_t promotions = 0;
  // Restores that left no RAM copy behind: the request runs from the
  // disk copy either way.
  uint64_t promotionsSkipped = 0;
  uint64_t publications = 0;
  uint64_t evictions = 0;
  // Blocks unfinished requests resume from, whether a state is still
  // cached there or not, and valid states at such a block that left all
  // the same.
  uint32_t inUse = 0;
  uint64_t inUseEvictions = 0;
  uint32_t checkpointEntries = 0;
  uint64_t checkpointBytes = 0;
  uint64_t checkpointRetirements = 0;
  // Evictions by a reclaim; rolling retirements are counted separately.
  uint64_t checkpointEvictions = 0;
};

struct StateEviction final {
  bool evicted = false;
  // Not evicted because the one write in flight holds the staging buffer;
  // the copy is written on a later call.
  bool pending = false;
};

// What one step of giving a restore point its copies ahead of need did
// (Cache::persist, StateCache::persist).
enum class PersistStatus : uint8_t {
  // Every copy it needs is complete on disk.
  Durable,
  // Writes it needs are in flight; it is durable once they land.
  Started,
  // The tier takes no more of these writes for now (transfers in flight, or
  // the write budget holds the state back); a later step may.
  Busy,
  // Nothing to keep: the point is gone, or its conversation went on from it.
  Unneeded,
  // The tier cannot keep it: the point is larger than what it may hold now,
  // making room gave the point itself up, or a file stopped taking writes.
  Refused,
};

// Starts the write of a state from the lane that holds it and returns the
// ticket carrying its disk copy, null when the quota cannot admit one; the
// argument is the write's completion hook.
using StateWriter =
    std::function<std::unique_ptr<StateOffload>(std::function<void()>)>;
// Gives up one disk copy for a new copy of a state in use or not; false once
// nothing that copy may displace is left.
using DiskRoom = std::function<bool(bool inUse)>;

// Attaches one immutable target-recurrent + draft-context state to a complete
// target-KV block, in RAM, on disk, or in both. The pair is restored
// atomically. A state that comes back from disk keeps its disk copy, so its
// next eviction from RAM costs no write.
class StateCache final {
public:
  // makeRoom gives up disk copies for a state's write the quota refuses. In
  // a persistent tier every write labels its disk copy (CompositeState::label).
  StateCache(KvCache &kv, CacheRecency &recency, DiskRoom makeRoom, bool persistent)
      : kv_(kv), recency_(recency), makeRoom_(std::move(makeRoom)), persistent_(persistent) {}
  StateCache(const StateCache &) = delete;
  StateCache &operator=(const StateCache &) = delete;

  // Wakes the engine when a state write started here lands.
  void setCompletionNotifier(std::function<void()> notifier) {
    completion_ = std::move(notifier);
  }

  [[nodiscard]] std::optional<CompositeStateLease>
  acquireDeepest(std::span<const uint64_t> kvChain);

  // Reuses a RAM copy without a restore pin. A normal boundary upgrades a
  // checkpoint; a checkpoint cannot downgrade an ordinary state. False for a
  // state that is absent or only on disk: the caller publishes the copy it
  // holds, which is promotion without a read.
  [[nodiscard]] bool reuseCompositeState(uint64_t kvBlock, bool checkpoint);
  // The same for a copy in either tier.
  [[nodiscard]] bool reuseStoredState(uint64_t kvBlock, bool checkpoint);

  // Publishes a RAM copy; a disk copy of the block stays beside it.
  void publishCompositeState(uint64_t kvBlock, std::shared_ptr<const CompositeState> state,
                             bool checkpoint);
  // Publishes a state that has no RAM copy by writing it from its lane: the
  // entry is the disk copy the ticket carries, with the write in flight.
  // False when the one write in flight holds the staging buffer, or when the
  // quota cannot admit the state after makeRoom gave up what it could;
  // nothing is published then. The caller reuses a stored state first
  // (reuseStoredState); publishing over one is a logic error.
  [[nodiscard]] bool publishStateToDisk(uint64_t kvBlock, const StateWriter &write,
                                        bool checkpoint);
  // Publication identity protects replacement states from stale handles.
  [[nodiscard]] StateCheckpoint checkpointState(uint64_t kvBlock) const noexcept;
  // Ensures this publication is no longer a disposable checkpoint. Returns
  // false only when the matching checkpoint is pinned; absent, replaced and
  // upgraded publications already satisfy the postcondition.
  bool retireCheckpointState(StateCheckpoint checkpoint) noexcept;
  // The matching checkpoint is pinned: retireCheckpointState refuses it.
  [[nodiscard]] bool checkpointPinned(StateCheckpoint checkpoint) const noexcept;
  // An unfinished request's conversation resumes from the state at this
  // block, published or yet to be. Until the handle is released that state
  // is in use, and so is the KV it restores through: the candidates below
  // offer states in use apart, and the cache gives them up after everything
  // else and only to running work or to a copy that is itself in use.
  // Requests sharing the block each hold a handle. A state in use is
  // reusable, so a checkpoint there becomes ordinary. Recency is set when
  // the last use ends (unuse): the conversation resumes there next.
  [[nodiscard]] StateUse useState(uint64_t kvBlock);
  [[nodiscard]] bool inUse(uint64_t kvBlock) const noexcept {
    return uses_.contains(kvBlock);
  }

  [[nodiscard]] bool contains(uint64_t kvBlock) const noexcept;
  // An ordinary state, in either tier.
  [[nodiscard]] bool ordinary(uint64_t kvBlock) const noexcept;
  // The size of the state at this block; zero without one.
  [[nodiscard]] uint64_t bytes(uint64_t kvBlock) const noexcept;
  // A RAM copy exists.
  [[nodiscard]] bool stateResident(uint64_t kvBlock) const noexcept;
  // A disk copy exists, written or being written.
  [[nodiscard]] bool onDisk(uint64_t kvBlock) const noexcept;
  // Writes a disk copy of a RAM copy from the RAM copy's own buffers, which
  // stay, ahead of need (Cache::persist): a reclaim then frees the RAM copy
  // without a write, and a persistent tier's next process takes the disk
  // copy back. Until the write lands the RAM copy is no eviction candidate, and
  // the write is the one in flight. Only with makeRoom does the write give
  // up disk copies for its quota. Durable once a disk copy has landed; busy
  // while the write budget holds the state back, and refused once its file
  // takes no write of a copy in use either, or the quota has no room for it.
  [[nodiscard]] PersistStatus persist(uint64_t kvBlock, bool makeRoom);
  // In a persistent tier, labels every disk copy again, with its state's
  // block, class and current recency.
  void relabel() const;
  // The resume point a reclaim may keep (Cache::Walk::keepResumePoint):
  // the newest unpinned ordinary state in RAM, else the newest such
  // checkpoint; 0 without either.
  [[nodiscard]] uint64_t resumePoint() const noexcept;
  // Oldest unpinned checkpoint in RAM, the first class a reclaim frees, or
  // the next after `after`, a candidate that stayed.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  checkpointCandidate(const std::optional<CacheEvictionCandidate> &after = {}) const noexcept;
  // Oldest unpinned ordinary state in RAM, or the next after `after`.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  ordinaryCandidate(const std::optional<CacheEvictionCandidate> &after = {}) const noexcept;
  // Oldest RAM copy of a state in use, which only work of class InUse takes
  // (ReclaimClass), or the next after `after`. The resume point is never in
  // use.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  inUseCandidate(const std::optional<CacheEvictionCandidate> &after = {}) const noexcept;
  // Frees the RAM copy of a candidate above (an unpinned RAM copy; anything
  // else throws std::logic_error): for nothing when a disk copy exists, by
  // writing one when the tier takes it (makeRoom frees quota on its behalf),
  // otherwise by dropping the state. The write copies the state into the
  // staging buffer, so the RAM copy is free at once; while the one write in
  // flight holds that buffer, a state to write stays, pending. A write the
  // quota refuses drops the state. Its buffers return to the model's pool,
  // which releaseIdleBuffer gives back to the host. A RAM copy a write in
  // flight reads (persist) stays until it lands: pending.
  [[nodiscard]] StateEviction reclaim(uint64_t kvBlock);
  // The state at this block may be given to reclaim(): an unpinned, valid
  // RAM copy.
  [[nodiscard]] bool reclaimable(uint64_t kvBlock) const noexcept;
  // A lease holds the state at this block (acquireDeepest), and with it the
  // block.
  [[nodiscard]] bool pinned(uint64_t kvBlock) const noexcept;
  // When the state at this block was last used; 0 without one.
  [[nodiscard]] uint64_t lastUsed(uint64_t kvBlock) const noexcept;
  // A reclaim of the state at this block writes it: it has no disk copy and
  // the tier takes its write.
  [[nodiscard]] bool reclaimWrites(uint64_t kvBlock) const noexcept;
  // Whether the tier takes the write of a writable RAM copy not in use, or
  // in use, now: CompositeState::canOffload, which answers alike for every
  // writable copy of the class, asked of the oldest a reclaim may take;
  // false without one.
  [[nodiscard]] bool takesWrite(bool inUse) const noexcept;
  // What a reclaim's walk over the RAM copies takes (Cache::reach): of the
  // checkpoints and the ordinary states but those `kept` names (the walk
  // keeps none in use), and with inUse the states in use, those that leave
  // with no write (now); the first that leaves through a write, in the
  // walk's order, while no write holds the staging buffer, which frees at
  // once (staged, its block; 0 for none); and the others that leave through
  // one, which wait for that buffer (later). Whether the tier takes the
  // writes of states not in use (writes) and, with inUse, in use
  // (usedWrites). From the counts the orders keep: it costs `kept`, not the
  // states (reclaimVisits).
  struct Reclaims final {
    uint32_t now = 0;
    uint64_t staged = 0;
    uint32_t later = 0;
    bool writes = false;
    bool usedWrites = false;
  };
  [[nodiscard]] Reclaims reclaims(bool inUse, std::span<const uint64_t> kept) const;
  // The states reclaims() looked at: those `kept` names and those of the
  // walk's order up to the staged one, never every state.
  [[nodiscard]] uint64_t reclaimVisits() const noexcept { return reclaimVisits_; }
  // What the state at the block asks of its KV, and whether it is a RAM copy
  // a reclaim may take (KvCache::setStateMarks), as the cache's counts
  // should name it: none without a state.
  [[nodiscard]] KvCache::StateMarks marks(uint64_t kvBlock) const noexcept;
  // What reclaim(kvBlock) of a reclaimable state does, doing nothing:
  // evicted or pending. A write the quota refuses drops the state, which
  // counts as evicted.
  [[nodiscard]] StateEviction reclaimOutcome(uint64_t kvBlock) const noexcept;
  // Removes an unpinned state from both tiers.
  void evict(uint64_t kvBlock) noexcept;
  // Disk replacement: the oldest unpinned disk copy that is redundant (a RAM
  // copy exists) or, without duplicate, one that is the only copy of a state
  // not in use; the next such copy after `after`.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  diskCandidate(bool duplicate, const std::optional<CacheEvictionCandidate> &after = {}) const noexcept;
  // The oldest only copy of a state in use.
  [[nodiscard]] std::optional<CacheEvictionCandidate> inUseDiskCandidate() const noexcept;
  // Drops a redundant disk copy.
  void dropDisk(uint64_t kvBlock);
  // A read of this copy failed: it leaves once unpinned, a RAM copy stays.
  void discardState(uint64_t kvBlock, const CompositeState *state) noexcept;
  // Whatever the block holds leaves once unpinned.
  void invalidate(uint64_t kvBlock) noexcept;
  // Takes back the disk copy of a state an earlier process left at this
  // block in a persistent tier, with the recency and class it had.
  void adoptDisk(uint64_t kvBlock, std::shared_ptr<const CompositeState> state, uint64_t lastUsed,
                 bool checkpoint);
  // A restored disk copy without a RAM copy takes one.
  [[nodiscard]] bool promotable(uint64_t kvBlock, const CompositeState *source) const noexcept;
  void promote(uint64_t kvBlock, const CompositeState *source,
               std::shared_ptr<const CompositeState> state);
  void promotionSkipped() noexcept { ++promotionsSkipped_; }
  // The one state write is in flight; its RAM or quota returns when it lands.
  [[nodiscard]] bool writing() const noexcept { return pending_.has_value(); }
  // The bytes of the RAM copy a write ahead of need reads (persist), which a
  // reclaim frees at once after it lands.
  [[nodiscard]] uint64_t persistingBytes() const noexcept {
    return pending_ && pending_->source ? pending_->source->bytes() : 0;
  }
  [[nodiscard]] bool pollOffload();
  [[nodiscard]] StateCacheSnapshot snapshot() const noexcept;

private:
  friend class CompositeStateLease;
  friend class StateUse;

  struct Entry {
    std::shared_ptr<const CompositeState> ram;
    std::shared_ptr<const CompositeState> disk;
    uint32_t pins = 0;
    uint64_t lastUsed = 0;
    bool checkpoint = false;
    bool invalid = false;
    uint64_t publication = 0;
    RecencyOrder::Node ramNode;
    RecencyOrder::Node diskNode;
    // A writable RAM copy a reclaim may take without a disk copy waits in
    // the order of its class's writes too.
    RecencyOrder::Node writeNode;
  };
  struct PendingOffload {
    uint64_t kvBlock;
    // A RAM copy the write reads in place (persist), held until the write
    // has drained, which the transfer, going first, does.
    std::shared_ptr<const CompositeState> source;
    std::unique_ptr<StateOffload> transfer;
  };

  // The order's oldest entry, or the next after `after`.
  [[nodiscard]] static std::optional<CacheEvictionCandidate>
  oldestOf(const RecencyOrder &order, const std::optional<CacheEvictionCandidate> &after) noexcept;
  [[nodiscard]] static const std::shared_ptr<const CompositeState> &
  copy(const Entry &entry) noexcept {
    return entry.ram ? entry.ram : entry.disk;
  }
  [[nodiscard]] Entry &entry(uint64_t kvBlock);
  // The block's entry, made when it has none.
  [[nodiscard]] Entry &entryFor(uint64_t kvBlock);
  // The entry a new copy takes over, made when the block has none; a
  // condemned one gives up the copies it still holds. A repeated checkpoint
  // keeps its lifetime; an ordinary publication upgrades a checkpoint in
  // either tier so rolling retirement cannot erase it.
  [[nodiscard]] Entry &publicationEntry(uint64_t kvBlock, bool checkpoint);
  // Starts the write of the block's state, giving up quota through
  // makeRoom_ while the tier refuses one, unless told not to make room; null
  // while the one write in flight holds the staging buffer. makeRoom_ leaves
  // states in RAM alone: reclaim holds the entry it writes. The disk copy
  // counts for the KV from the first attempt (KvCache::countDiskState), and
  // no longer once none starts.
  [[nodiscard]] std::unique_ptr<StateOffload> startWrite(uint64_t kvBlock, const StateWriter &write,
                                                         bool makeRoom = true);
  // The disk copy this write carries, counted since startWrite, becomes the
  // entry's; the write is the one in flight, reading `source` in place when
  // given.
  void beginWrite(uint64_t kvBlock, Entry &entry, std::unique_ptr<StateOffload> transfer,
                  std::shared_ptr<const CompositeState> source);
  // In a persistent tier, labels the entry's disk copy with its block,
  // class and recency.
  void labelDisk(uint64_t kvBlock, const Entry &entry) const;
  [[nodiscard]] StateEviction erase(uint64_t kvBlock, bool retirement) noexcept;
  void release(uint64_t kvBlock) noexcept;
  void unuse(uint64_t kvBlock) noexcept;
  [[nodiscard]] std::optional<CompositeStateLease>
  acquireBlock(uint64_t kvBlock);
  // Places the entry in the orders its copies call for, and names what it
  // asks of its KV (marks), whether it has a RAM copy and the resume point to
  // the KV cache.
  void reindex(uint64_t kvBlock, Entry &entry) noexcept;
  // A RAM copy a reclaim may take: in the order of its class.
  [[nodiscard]] bool candidate(uint64_t kvBlock, const Entry &entry) const noexcept;
  [[nodiscard]] KvCache::StateMarks marks(uint64_t kvBlock, const Entry &entry) const noexcept;
  // Names the resume point to the KV cache once it moved.
  void markResumePoint() noexcept;
  void discardRam(Entry &entry) noexcept;
  void discardDisk(uint64_t kvBlock, Entry &entry) noexcept;
  // An ordinary publication or reuse: the block has held a reusable state,
  // and a checkpoint is upgraded, its disk copy labelled with its new class
  // at once.
  void makeOrdinary(uint64_t kvBlock, Entry &entry);
  [[nodiscard]] bool writing(uint64_t kvBlock) const noexcept {
    return pending_ && pending_->kvBlock == kvBlock;
  }
  // The write in flight reads this block's RAM copy (persist).
  [[nodiscard]] bool persisting(uint64_t kvBlock) const noexcept {
    return writing(kvBlock) && pending_->source;
  }

  KvCache &kv_;
  CacheRecency &recency_;
  DiskRoom makeRoom_;
  bool persistent_;
  std::function<void()> completion_;
  std::unordered_map<uint64_t, Entry> entries_;
  RecencyOrder ordinary_;
  RecencyOrder checkpoints_;
  RecencyOrder inUse_;
  // Of each, the writable RAM copies without a disk copy
  // (CompositeState::writable), which leave through a write while the tier
  // takes one.
  RecencyOrder ordinaryWrites_;
  RecencyOrder checkpointWrites_;
  RecencyOrder inUseWrites_;
  // The resume point the KV cache was last told of.
  uint64_t resumeMarked_ = 0;
  RecencyOrder duplicates_;
  RecencyOrder diskOnly_;
  RecencyOrder inUseOnDisk_;
  // Unfinished requests per block they resume from. A block may leave
  // before its last user; its ID is never reused.
  std::unordered_map<uint64_t, uint32_t> uses_;
  uint64_t inUseEvictions_ = 0;
  uint64_t promotions_ = 0;
  uint64_t promotionsSkipped_ = 0;
  uint64_t bytes_ = 0;
  uint64_t diskBytes_ = 0;
  uint32_t pinnedEntries_ = 0;
  uint64_t publications_ = 0;
  mutable uint64_t reclaimVisits_ = 0;
  uint64_t evictions_ = 0;
  uint64_t checkpointEntries_ = 0;
  uint64_t checkpointBytes_ = 0;
  uint64_t checkpointRetirements_ = 0;
  uint64_t checkpointEvictions_ = 0;
  uint64_t offloads_ = 0;
  uint64_t offloadFailures_ = 0;
  uint64_t invalidations_ = 0;
  // Destroyed before entries_: the ticket's disk copy may still be an entry.
  std::optional<PendingOffload> pending_;
};

} // namespace splash::engine
