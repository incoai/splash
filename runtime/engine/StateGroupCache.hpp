#pragma once

#include "engine/CacheRecency.hpp"
#include "engine/KvCache.hpp"
#include "engine/RecencyOrder.hpp"
#include "engine/StateResources.hpp"
#include "model/CacheGroups.hpp"
#include "model/Model.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>

namespace splash::engine {

class StateGroupCache;

struct GroupCheckpoint final {
  uint64_t kvBlock = 0;
  uint64_t publication = 0;
  bool operator==(const GroupCheckpoint &) const = default;
  [[nodiscard]] explicit operator bool() const noexcept { return kvBlock != 0; }
};

class StateBlockLease final {
public:
  StateBlockLease(const StateBlockLease &) = delete;
  StateBlockLease &operator=(const StateBlockLease &) = delete;
  StateBlockLease(StateBlockLease &&other) noexcept;
  StateBlockLease &operator=(StateBlockLease &&other) noexcept;
  ~StateBlockLease() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != nullptr;
  }
  [[nodiscard]] uint64_t kvBlock() const noexcept { return kvBlock_; }
  [[nodiscard]] uint32_t boundary() const noexcept { return boundary_; }
  [[nodiscard]] const std::shared_ptr<const StatePayload> &
  state() const noexcept {
    return state_;
  }
  void reset() noexcept;

private:
  friend class StateGroupCache;
  StateBlockLease(StateGroupCache &owner, uint64_t kvBlock, uint32_t boundary,
                  std::shared_ptr<const StatePayload> state,
                  CacheAccess access) noexcept;

  CacheAccess access_ = CacheAccess::Request;
  StateGroupCache *owner_ = nullptr;
  uint64_t kvBlock_ = 0;
  uint32_t boundary_ = 0;
  std::shared_ptr<const StatePayload> state_;
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
using StateBlockWriter =
    std::function<std::unique_ptr<StateOffload>(std::function<void()>)>;

// Owns one model-declared group's blocks in RAM, on disk, or in both. Each
// block has independent recency and pins. A restored RAM copy retains its disk
// record, so subsequent eviction needs no write.
class StateGroupCache final {
public:
  StateGroupCache(KvCache &kv, CacheRecency &recency, CacheGroupId group = 0)
      : kv_(kv), recency_(recency), group_(group) {}
  StateGroupCache(const StateGroupCache &) = delete;
  StateGroupCache &operator=(const StateGroupCache &) = delete;

  // Pins one block selected by the coordinator; maintenance preserves recency.
  [[nodiscard]] std::optional<StateBlockLease>
  acquireBlock(uint64_t kvBlock, CacheAccess access);
  // Reuse upgrades checkpoints but never downgrades ordinary state.
  [[nodiscard]] bool touchIfStored(uint64_t kvBlock, bool checkpoint = false);

  // Write-through uses the same staging and ticket as pressure offload,
  // keeping the resident copy. A disk copy is exported only after completion.
  [[nodiscard]] bool writing(uint64_t kvBlock) const noexcept {
    const auto found = entries_.find(kvBlock);
    return (pending_ && pending_->kvBlock == kvBlock) ||
           (found != entries_.end() && found->second.externalWrite);
  }
  bool copyToDisk(uint64_t block, const std::function<void()> &completion,
                  const std::function<bool()> &makeRoom);
  std::shared_ptr<const StatePayload> diskCopy(uint64_t block) const;
  bool importDisk(uint64_t block, std::shared_ptr<const StatePayload> state,
                  uint32_t begin = UINT32_MAX, bool checkpoint = false);
  [[nodiscard]] std::optional<CachedStateBlock> peek(uint64_t block) const;
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  candidate(uint64_t block, bool disk = false,
            bool duplicate = false) const noexcept;
  void setOffloadEnabled(bool enabled) noexcept { offloadEnabled_ = enabled; }

  // Publishes a RAM copy; a disk copy of the block stays beside it.
  void publish(uint64_t kvBlock, std::shared_ptr<const StatePayload> state,
               bool checkpoint = false, uint32_t begin = UINT32_MAX);
  // Publication identity protects replacement states from stale handles.
  [[nodiscard]] GroupCheckpoint checkpoint(uint64_t kvBlock) const noexcept;
  // Ensures this publication is no longer a disposable checkpoint. Returns
  // false only when the matching checkpoint is pinned; absent, replaced and
  // upgraded publications already satisfy the postcondition.
  bool retireCheckpoint(GroupCheckpoint checkpoint) noexcept;
  // Refreshes actual use even while a background copy holds the state.
  void touch(uint64_t kvBlock) noexcept;

  [[nodiscard]] uint64_t resumePoint() const noexcept;
  [[nodiscard]] bool contains(uint64_t kvBlock) const noexcept;
  // A RAM copy exists.
  [[nodiscard]] bool resident(uint64_t kvBlock) const noexcept;
  // Oldest RAM copy to free; unpinned checkpoints precede ordinary states
  // regardless of recency. Without checkpoints, the oldest ordinary state.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  evictionCandidate(bool checkpoints = true) const noexcept;
  // Frees an unpinned RAM copy: for nothing when a disk copy exists, by
  // writing one when the tier takes it (makeRoom frees quota on its behalf),
  // by dropping the state otherwise. A successful reclaim frees the RAM copy.
  // A state waiting for occupied staging stays cached and reports pending.
  [[nodiscard]] StateEviction
  reclaim(uint64_t kvBlock, std::function<void()> completion,
          const std::function<bool()> &makeRoom = {});
  // Removes an unpinned state from both tiers.
  [[nodiscard]] StateEviction evict(uint64_t kvBlock) noexcept;
  // Disk replacement: the oldest unpinned disk copy that is redundant (a RAM
  // copy exists) or, without duplicate, one that is the only copy.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  diskCandidate(bool duplicate) const noexcept;
  // Drops a redundant disk copy.
  void dropDisk(uint64_t kvBlock);
  // A read of this copy failed: it leaves once unpinned, a RAM copy stays.
  void invalidate(uint64_t kvBlock, const StatePayload *state) noexcept;
  // Whatever the block holds leaves once unpinned.
  void invalidate(uint64_t kvBlock) noexcept;
  // A restored disk copy without a RAM copy takes one.
  [[nodiscard]] bool promotable(uint64_t kvBlock,
                                const StatePayload *source) const noexcept;
  void promote(uint64_t kvBlock, const StatePayload *source,
               std::shared_ptr<const StatePayload> state);
  // The one state write is in flight; its RAM or quota returns when it lands.
  [[nodiscard]] bool writing() const noexcept { return pending_.has_value(); }
  [[nodiscard]] bool pollOffload();
  void setExternalWrite(uint64_t block, const StatePayload *source,
                        bool writing) noexcept;
  [[nodiscard]] StateCacheSnapshot snapshot() const noexcept;

private:
  friend class StateBlockLease;

  struct Entry {
    uint32_t begin = 0;
    bool externalWrite = false;
    std::shared_ptr<const StatePayload> ram;
    std::shared_ptr<const StatePayload> disk;
    std::vector<StateResource> ramResources, diskResources;
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

  [[nodiscard]] static const std::shared_ptr<const StatePayload> &
  copy(const Entry &entry) noexcept {
    return entry.ram ? entry.ram : entry.disk;
  }
  [[nodiscard]] Entry &entry(uint64_t kvBlock);
  // The block's entry, made when it has none.
  [[nodiscard]] Entry &entryFor(uint64_t kvBlock);
  // The entry a new copy takes over, made when the block has none. A
  // repeated checkpoint keeps its lifetime; an ordinary publication upgrades
  // a checkpoint in either tier so rolling retirement cannot erase it.
  [[nodiscard]] Entry &publicationEntry(uint64_t kvBlock, bool checkpoint);
  // Starts a write, giving up quota through makeRoom while the tier refuses
  // one; null while the one write in flight holds the staging buffer.
  // makeRoom leaves states in RAM alone: reclaim holds the entry it writes.
  [[nodiscard]] std::unique_ptr<StateOffload>
  startWrite(const StateBlockWriter &write,
             const std::function<void()> &completion,
             const std::function<bool()> &makeRoom);
  // The disk copy this write carries becomes the entry's; the write is the
  // one in flight.
  void beginWrite(uint64_t kvBlock, Entry &entry,
                  std::unique_ptr<StateOffload> transfer);
  [[nodiscard]] StateEviction erase(uint64_t kvBlock, bool retirement) noexcept;
  void release(uint64_t kvBlock, CacheAccess access) noexcept;
  // Places the entry in the orders its copies call for.
  void reindex(uint64_t kvBlock, Entry &entry) noexcept;
  static void unlink(Entry &entry) noexcept;
  void discardDisk(Entry &entry) noexcept;
  void retainRam(Entry &entry, const StatePayload &state);
  uint64_t releaseRam(Entry &entry) noexcept;
  void retainDisk(Entry &entry, const StatePayload &state);
  // An ordinary publication or reuse: the block has held a reusable state,
  // and a checkpoint is upgraded.
  void makeOrdinary(uint64_t kvBlock, Entry &entry);

  bool offloadEnabled_ = true;
  KvCache &kv_;
  CacheRecency &recency_;
  CacheGroupId group_;
  std::unordered_map<uint64_t, Entry> entries_;
  RecencyOrder ordinary_;
  RecencyOrder checkpoints_;
  RecencyOrder duplicates_;
  RecencyOrder diskOnly_;
  uint64_t promotions_ = 0;
  StateResources ramResources_, diskResources_, checkpointResources_;
  uint64_t bytes_ = 0;
  uint64_t diskBytes_ = 0;
  uint32_t pinnedEntries_ = 0;
  uint64_t publications_ = 0;
  uint64_t publicationSequence_ = 0;
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
