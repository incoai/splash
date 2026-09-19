#pragma once

#include "engine/KvCache.hpp"
#include "engine/KvPool.hpp"
#include "engine/StateCache.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace splash::engine {

struct CacheLookup final {
  uint32_t kvBoundary = 0;
  std::optional<CompositeStateLease> state;

  [[nodiscard]] uint32_t resumeBoundary() const noexcept {
    return state ? state->boundary() : 0;
  }
  [[nodiscard]] uint32_t junctionBoundary() const noexcept {
    return kvBoundary > resumeBoundary() ? kvBoundary : 0;
  }
};

struct CacheLookupSnapshot final {
  uint64_t lookups = 0;
  uint64_t kvHitTokens = 0;
  uint64_t stateHitTokens = 0;
  uint64_t lazyJunctions = 0;
};

struct CacheSnapshot final {
  KvPoolSnapshot pool;
  KvCache::Snapshot kvCache;
  StateCacheSnapshot stateCache;
  CacheLookupSnapshot lookup;
  uint32_t activeRequests = 0;
};

struct TokenAdmission final {
  KvPageAcquireFailure failure = KvPageAcquireFailure::None;
  uint32_t additionalPages = 0;
  uint32_t availablePages = 0;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;

  [[nodiscard]] bool granted() const noexcept {
    return failure == KvPageAcquireFailure::None;
  }
};

struct PageTableView final {
  std::span<const uint32_t> pages;
  uint64_t revision = 0;
};

struct CacheReclaimResult final {
  bool madeProgress = false;
  uint64_t reclaimedBytes = 0;
};

enum class CacheReclaimMode { ReuseBacking, ReleaseBacking };

// Owns active Q8 page leases, the content-addressed KV graph and cached
// composite states. Physical recurrent-state cells remain model-owned.
class Cache final {
public:
  Cache(KvPool &pool, CacheNamespace cacheNamespace);
  Cache(const Cache &) = delete;
  Cache &operator=(const Cache &) = delete;

  void beginRequest(uint64_t requestId);
  void endRequest(uint64_t requestId);

  // Pin the usable prefix before potentially evicting for active allocations.
  // Accounting is separate: failed admission retries are not extra samples.
  [[nodiscard]] CacheLookup lookup(std::span<const uint32_t> prompt,
                                   std::span<const ImageSpan> images = {});
  void recordLookup(const CacheLookup &lookup);
  void restoreRequest(uint64_t requestId, const CacheLookup &lookup);

  [[nodiscard]] TokenAdmission ensureTokens(uint64_t requestId,
                                            uint64_t tokenCount);
  [[nodiscard]] PageTableView pageTable(uint64_t requestId) const;

  // Canonicalizes every newly complete Page32 block. Duplicate content swaps
  // the request to the existing immutable page after the writer command has
  // completed; no active command ever aliases a writable page.
  [[nodiscard]] uint64_t publishCommittedBlocks(
      uint64_t requestId, std::span<const uint32_t> exactTokens,
      uint32_t committedTokens, std::span<const ImageSpan> images = {});
  [[nodiscard]] uint64_t blockAt(uint64_t requestId, uint32_t boundary) const;
  [[nodiscard]] bool reuseCompositeState(uint64_t kvBlock, bool checkpoint = false);
  void publishCompositeState(uint64_t kvBlock,
                             std::shared_ptr<const CompositeState> state,
                             bool checkpoint = false);
  [[nodiscard]] StateCheckpoint checkpointState(uint64_t kvBlock) const noexcept;
  // False only while this exact disposable publication is pinned.
  bool retireCheckpointState(StateCheckpoint checkpoint) noexcept;

  // One cache reclaimer for memory growth and pressure warnings. After empty
  // backing, disposable checkpoints are reclaimed first. Ordinary states and
  // state-free KV leaves retain their shared oldest-first access order.
  // Active requests and pinned restores are never selected. Physical release
  // is paced by the backing: while an earlier release is still being torn
  // down, this pass stops instead of evicting cache whose extents could not
  // be released yet; the caller retries once releaseDeferred() clears.
  // keepResumePoint stops short of the newest state publication. A shrink
  // that no request is waiting for gains the one cell that publication holds
  // and costs the next request a replay of its whole prompt, because a
  // hybrid model cannot resume from cached KV without the recurrent state.
  // Empty backing, older publications and state-free KV are still reclaimed.
  [[nodiscard]] uint64_t reclaimCache(uint64_t targetBytes, bool evictAll,
                                      bool keepResumePoint = false);
  // One bounded reclaim step for an allocation retry. Progress is distinct
  // from physical bytes because evicting a KV reference can make a resident
  // page reusable without immediately emptying its extent.
  [[nodiscard]] CacheReclaimResult reclaimOne(
      CacheReclaimMode mode = CacheReclaimMode::ReleaseBacking,
      bool keepResumePoint = false);
  // Recycles an unpinned state, preferring checkpoints, for a required state
  // publication. KV bytes only return once an extent unmaps.
  [[nodiscard]] bool reclaimOneState(bool checkpointsOnly = false);
  // Empty resident backing exists but the previous release is still in
  // flight; more reclaim work becomes possible without evicting anything.
  [[nodiscard]] bool releaseDeferred() const noexcept;
  // Includes the last unmap, even when no empty extent remains to reclaim.
  [[nodiscard]] bool releasePending() const noexcept;
  [[nodiscard]] uint64_t releaseGeneration() const noexcept;
  // Startup cleanup only: unmap unused backing without evicting cache data,
  // keeping one runway extent. Waits for each paced release.
  void releaseUnusedKvBacking();
  [[nodiscard]] CacheSnapshot snapshot() const;

private:
  struct Request final {
    std::vector<uint32_t> pages;
    std::vector<uint64_t> cachedBlocks;
    uint64_t pageTableRevision = 0;
  };

  [[nodiscard]] Request &request(uint64_t requestId);
  [[nodiscard]] const Request &request(uint64_t requestId) const;
  [[nodiscard]] bool makeLogicalPages(uint32_t count);
  [[nodiscard]] bool evictOneKvBlock();
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  oldestStateFreeKvBlock() const;
  [[nodiscard]] uint64_t reclaimEmptyExtents();

  KvPool &pool_;
  CacheRecency recency_;
  KvCache kv_;
  StateCache states_;
  std::unordered_map<uint64_t, Request> requests_;
  CacheLookupSnapshot lookup_;
};

} // namespace splash::engine
