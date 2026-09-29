#pragma once

#include "engine/CacheGroupCoordinator.hpp"
#include "engine/StateGroupCache.hpp"

#include <map>

namespace splash::engine {

using StateWriter =
    std::function<std::unique_ptr<SnapshotOffload>(std::function<void()>)>;

struct StateCheckpoint final {
  uint64_t kvBlock = 0;
  std::vector<std::pair<CacheGroupId, GroupCheckpoint>> groups;
  [[nodiscard]] explicit operator bool() const noexcept { return kvBlock != 0; }
};

// A restore pins all participating groups and the target endpoint before the
// engine admits execution memory. Empty auxiliary bundles still pin target KV.
class RestoreLease final {
public:
  RestoreLease(const RestoreLease &) = delete;
  RestoreLease &operator=(const RestoreLease &) = delete;
  RestoreLease(RestoreLease &&other) noexcept;
  RestoreLease &operator=(RestoreLease &&other) noexcept;
  ~RestoreLease() noexcept;
  [[nodiscard]] explicit operator bool() const noexcept {
    return kv_ != nullptr;
  }
  [[nodiscard]] uint64_t kvBlock() const noexcept { return block_; }
  [[nodiscard]] uint32_t boundary() const noexcept { return state_->boundary; }
  [[nodiscard]] const std::shared_ptr<const RestoreState> &
  state() const noexcept {
    return state_;
  }
  void reset() noexcept;

private:
  friend class StateCache;
  RestoreLease(KvCache &kv, uint64_t block, CacheAccess access,
               std::shared_ptr<const RestoreState> state,
               std::vector<StateBlockLease> leases);
  KvCache *kv_ = nullptr;
  uint64_t block_ = 0;
  CacheAccess access_ = CacheAccess::Maintenance;
  std::shared_ptr<const RestoreState> state_;
  std::vector<StateBlockLease> leases_;
};

// Group stores own residency, IO and LRU. The coordinator owns only coverage:
// no target architecture, distinguished recurrent record or draft special case.
class StateCache final {
public:
  StateCache(KvCache &kv, CacheRecency &recency);
  void configure(std::vector<CacheGroupSpec> groups);
  [[nodiscard]] std::optional<RestoreLease>
  acquireDeepest(std::span<const uint64_t> chain,
                 CacheAccess access = CacheAccess::Request);
  [[nodiscard]] std::optional<RestoreLease> acquireResumePoint();
  [[nodiscard]] bool
  isCheckpoint(const CacheEvictionCandidate &candidate) const noexcept;
  [[nodiscard]] uint32_t matchedBoundary(std::span<const uint64_t> chain) const;
  void recordLookup(bool hit, bool disk) noexcept;
  [[nodiscard]] bool touchIfResident(uint64_t block, bool checkpoint = false);
  [[nodiscard]] bool touchIfStored(uint64_t block, bool checkpoint = false);
  void touch(std::span<const uint64_t> chain) noexcept;
  void publish(uint64_t block, std::shared_ptr<const RestoreState> state,
               bool checkpoint = false);
  void importDisk(uint64_t block, std::shared_ptr<const RestoreState> state);
  [[nodiscard]] bool publishToDisk(uint64_t block, const StateWriter &write,
                                   const std::function<void()> &completion,
                                   const std::function<bool()> &makeRoom,
                                   bool checkpoint = false);
  [[nodiscard]] std::shared_ptr<const RestoreState>
  diskCopy(uint64_t block) const;
  bool copyToDisk(uint64_t block, const std::function<void()> &completion,
                  const std::function<bool()> &makeRoom);
  void setOffloadEnabled(bool enabled) noexcept;
  [[nodiscard]] bool contains(uint64_t block) const noexcept;
  [[nodiscard]] bool resident(uint64_t block) const noexcept;
  [[nodiscard]] StateCheckpoint checkpoint(uint64_t block) const;
  [[nodiscard]] bool
  retireCheckpoint(const StateCheckpoint &checkpoint) noexcept;
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  evictionCandidate(bool checkpoints = true) const noexcept;
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  diskCandidate(bool duplicate) const noexcept;
  [[nodiscard]] StateEviction
  reclaim(uint64_t block, std::function<void()> completion,
          const std::function<bool()> &makeRoom = {},
          std::optional<CacheGroupId> selected = {});
  [[nodiscard]] StateEviction evict(uint64_t block) noexcept;
  void dropDisk(uint64_t block, std::optional<CacheGroupId> selected = {});
  void evictDiskOnly(uint64_t block, std::optional<CacheGroupId> selected = {});
  void invalidate(uint64_t block, const RestoreState *state) noexcept;
  void invalidate(uint64_t block) noexcept;
  [[nodiscard]] bool promotable(uint64_t block,
                                const RestoreState *source) const noexcept;
  void promote(uint64_t block, const RestoreState *source,
               std::shared_ptr<const RestoreState> state);
  void promotionSkipped() noexcept { ++promotionsSkipped_; }
  [[nodiscard]] bool writing(uint64_t block) const noexcept;
  [[nodiscard]] bool writing() const noexcept;
  [[nodiscard]] bool pollOffload();
  [[nodiscard]] StateCacheSnapshot snapshot() const noexcept;

private:
  void validate(uint64_t leaf, const RestoreState &state) const;
  [[nodiscard]] StateGroupCache &group(CacheGroupId id);
  [[nodiscard]] const StateGroupCache &group(CacheGroupId id) const;
  [[nodiscard]] std::optional<RestoreState>
  match(std::span<const uint64_t> chain, bool residentOnly = false) const;
  [[nodiscard]] uint64_t endpoint(uint64_t leaf, uint32_t boundary) const;
  [[nodiscard]] std::optional<CacheGroupId>
  oldestGroup(uint64_t block, bool disk = false,
              bool duplicate = false) const noexcept;
  KvCache &kv_;
  CacheRecency &recency_;
  CacheGroupCoordinator coordinator_;
  std::map<CacheGroupId, std::unique_ptr<StateGroupCache>> groups_;
  uint64_t hits_ = 0, misses_ = 0, diskHits_ = 0, promotionsSkipped_ = 0;
  bool offloadEnabled_ = true;
  struct PendingSnapshot {
    uint64_t leaf;
    std::unique_ptr<SnapshotOffload> transfer;
    struct Block {
      CacheGroupId group;
      uint64_t endpoint;
      std::shared_ptr<const StatePayload> payload;
    };
    std::vector<Block> blocks;
  };
  std::optional<PendingSnapshot> pending_;
  uint64_t directOffloads_ = 0, directFailures_ = 0, directPublications_ = 0;
};

} // namespace splash::engine
