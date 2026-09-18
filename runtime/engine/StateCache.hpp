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

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != nullptr;
  }
  [[nodiscard]] uint64_t kvBlock() const noexcept { return kvBlock_; }
  [[nodiscard]] uint32_t boundary() const noexcept { return boundary_; }
  [[nodiscard]] const std::shared_ptr<const CompositeState> &
  state() const noexcept {
    return state_;
  }
  void reset() noexcept;

private:
  friend class StateCache;
  CompositeStateLease(StateCache &owner, uint64_t kvBlock, uint32_t boundary,
                      std::shared_ptr<const CompositeState> state) noexcept;

  StateCache *owner_ = nullptr;
  uint64_t kvBlock_ = 0;
  uint32_t boundary_ = 0;
  std::shared_ptr<const CompositeState> state_;
};

struct StateCacheSnapshot {
  uint32_t entries = 0;
  uint32_t pinned = 0;
  uint64_t bytes = 0;
  uint64_t diskBytes = 0;
  uint64_t offloads = 0;
  uint64_t offloadFailures = 0;
  uint64_t invalidations = 0;
  uint64_t diskHits = 0;
  uint64_t promotions = 0;
  // Restores that left no RAM copy behind: the request runs from the
  // disk copy either way.
  uint64_t promotionsSkipped = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t publications = 0;
  uint64_t deduplicatedPublications = 0;
  uint64_t evictions = 0;
  uint32_t checkpointEntries = 0;
  uint64_t checkpointBytes = 0;
  uint64_t checkpointRetirements = 0;
  // Pressure and logical eviction; rolling retirements are counted separately.
  uint64_t checkpointEvictions = 0;
};

struct StateEviction final {
  bool evicted = false;
  uint64_t reclaimedBytes = 0;
  // Not evicted because the one write in flight holds the staging buffer;
  // the copy is written on a later call.
  bool pending = false;
};

// Starts the write of a state from the lane that holds it and returns the
// ticket carrying its disk copy, null when the quota cannot admit one; the
// argument is the write's completion hook.
using StateWriter =
    std::function<std::unique_ptr<StateOffload>(std::function<void()>)>;

// Attaches one immutable target-recurrent + draft-context state to a complete
// target-KV block, in RAM, on disk, or in both. The pair is restored
// atomically. A state that comes back from disk keeps its disk copy, so its
// next eviction from RAM costs no write.
class StateCache final {
public:
  StateCache(KvCache &kv, CacheRecency &recency) : kv_(kv), recency_(recency) {}
  StateCache(const StateCache &) = delete;
  StateCache &operator=(const StateCache &) = delete;

  [[nodiscard]] std::optional<CompositeStateLease>
  acquireDeepest(std::span<const uint64_t> kvChain);
  // Acquisition pins backing; accounting occurs only when admission succeeds.
  void recordLookup(bool hit, bool disk = false) noexcept;

  // Reuses a RAM copy without a restore pin or lookup accounting. A normal
  // boundary upgrades a checkpoint; a checkpoint cannot downgrade an
  // ordinary state. False for a state that is absent or only on disk: the
  // caller publishes the copy it holds, which is promotion without a read.
  [[nodiscard]] bool touchIfResident(uint64_t kvBlock, bool checkpoint = false);

  // Publishes a RAM copy; a disk copy of the block stays beside it.
  void publish(uint64_t kvBlock, std::shared_ptr<const CompositeState> state,
               bool checkpoint = false);
  // Publishes a state that has no RAM copy by writing it from its lane: the
  // entry is the disk copy the ticket carries, with the write in flight. A
  // block whose state is on disk already is published as it is. False when
  // the one write in flight holds the staging buffer, or when the quota
  // cannot admit the state after makeRoom gave up what it could; nothing is
  // published then. A checkpoint takes free quota only.
  [[nodiscard]] bool publishToDisk(uint64_t kvBlock, const StateWriter &write,
                                   const std::function<void()> &completion,
                                   const std::function<bool()> &makeRoom,
                                   bool checkpoint = false);
  // Publication identity protects replacement states from stale handles.
  [[nodiscard]] StateCheckpoint checkpoint(uint64_t kvBlock) const noexcept;
  // Ensures this publication is no longer a disposable checkpoint. Returns
  // false only when the matching checkpoint is pinned; absent, replaced and
  // upgraded publications already satisfy the postcondition.
  bool retireCheckpoint(StateCheckpoint checkpoint) noexcept;
  // Refreshes recency. No-op when absent or pinned.
  void touch(uint64_t kvBlock) noexcept;

  [[nodiscard]] bool contains(uint64_t kvBlock) const noexcept;
  // A RAM copy exists.
  [[nodiscard]] bool resident(uint64_t kvBlock) const noexcept;
  // Oldest RAM copy to free; unpinned checkpoints precede ordinary states
  // regardless of recency.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  evictionCandidate(bool keepResumePoint = false) const noexcept;
  // Frees an unpinned RAM copy: for nothing when a disk copy exists, by
  // writing one when the tier takes it (makeRoom frees quota on its behalf;
  // a disposable checkpoint takes free quota only), by dropping the state
  // otherwise. The RAM is free when the call returns.
  // With waitForWrite, a state that could be written once the write in
  // flight has finished is kept and reported pending instead of dropped.
  [[nodiscard]] StateEviction reclaim(uint64_t kvBlock,
                                      std::function<void()> completion,
                                      const std::function<bool()> &makeRoom = {},
                                      bool waitForWrite = false);
  // Removes an unpinned state from both tiers.
  [[nodiscard]] StateEviction evict(uint64_t kvBlock) noexcept;
  // Disk replacement: the oldest unpinned disk copy that is redundant (a RAM
  // copy exists) or, without duplicate, one that is the only copy.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  diskCandidate(bool duplicate) const noexcept;
  // Drops a redundant disk copy.
  void dropDisk(uint64_t kvBlock);
  // A read of this copy failed: it leaves once unpinned, a RAM copy stays.
  void invalidate(uint64_t kvBlock, const CompositeState *state) noexcept;
  // Whatever the block holds leaves once unpinned.
  void invalidate(uint64_t kvBlock) noexcept;
  // A restored disk copy without a RAM copy takes one.
  [[nodiscard]] bool promotable(uint64_t kvBlock, const CompositeState *source) const noexcept;
  void promote(uint64_t kvBlock, const CompositeState *source,
               std::shared_ptr<const CompositeState> state);
  void promotionSkipped() noexcept { ++promotionsSkipped_; }
  // The one state write is in flight; its RAM or quota returns when it lands.
  [[nodiscard]] bool writing() const noexcept { return pending_.has_value(); }
  [[nodiscard]] bool pollOffload();
  [[nodiscard]] StateCacheSnapshot snapshot() const noexcept;

private:
  friend class CompositeStateLease;

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
  };
  struct PendingOffload {
    uint64_t kvBlock;
    std::unique_ptr<StateOffload> transfer;
  };

  [[nodiscard]] uint64_t resumePoint() const noexcept;
  [[nodiscard]] static const std::shared_ptr<const CompositeState> &
  copy(const Entry &entry) noexcept {
    return entry.ram ? entry.ram : entry.disk;
  }
  [[nodiscard]] Entry &entry(uint64_t kvBlock);
  // The block's entry, made when it has none.
  [[nodiscard]] Entry &entryFor(uint64_t kvBlock);
  // Starts a write, giving up quota through makeRoom while the tier refuses
  // one; null while the one write in flight holds the staging buffer.
  [[nodiscard]] std::unique_ptr<StateOffload>
  startWrite(const StateWriter &write, const std::function<void()> &completion,
             const std::function<bool()> &makeRoom);
  // The disk copy this write carries becomes the entry's; the write is the
  // one in flight.
  void beginWrite(uint64_t kvBlock, Entry &entry, std::unique_ptr<StateOffload> transfer);
  [[nodiscard]] StateEviction erase(uint64_t kvBlock, bool retirement) noexcept;
  void release(uint64_t kvBlock) noexcept;
  [[nodiscard]] std::optional<CompositeStateLease>
  acquireBlock(uint64_t kvBlock);
  // Places the entry in the orders its copies call for.
  void reindex(uint64_t kvBlock, Entry &entry) noexcept;
  static void unlink(Entry &entry) noexcept;
  void discardDisk(Entry &entry) noexcept;
  void makeOrdinary(Entry &entry) noexcept;
  [[nodiscard]] bool writing(uint64_t kvBlock) const noexcept {
    return pending_ && pending_->kvBlock == kvBlock;
  }

  KvCache &kv_;
  CacheRecency &recency_;
  std::unordered_map<uint64_t, Entry> entries_;
  RecencyOrder ordinary_;
  RecencyOrder checkpoints_;
  RecencyOrder duplicates_;
  RecencyOrder diskOnly_;
  uint64_t promotions_ = 0;
  uint64_t promotionsSkipped_ = 0;
  uint64_t bytes_ = 0;
  uint64_t diskBytes_ = 0;
  uint32_t pinnedEntries_ = 0;
  uint64_t diskHits_ = 0;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
  uint64_t publications_ = 0;
  uint64_t deduplicatedPublications_ = 0;
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
