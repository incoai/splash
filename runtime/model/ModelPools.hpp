#pragma once

// The model's memories that follow the engine's (model::ElasticPool), over
// the mechanisms that hold them: the store of streamed experts (ExpertStore)
// and the hot block cache of spilled K/V (KvBlockCache). The runtime's own,
// the history's staging, the vision arena and the image embeddings, are
// Runtime::Impl's (Runtime::pools). An adapter only calls its mechanism's
// functions: none changes how the mechanism holds or gives back its memory.

#include "model/ElasticPool.hpp"
#include "model/ExpertStore.hpp"
#include "model/KvBlockCache.hpp"

#include <algorithm>
#include <cstdint>

namespace splash::model {

// Gives units back with `releaseOne`, which returns a unit's bytes or zero
// once it has none to give, until their bytes reach `bytes`; returns them.
template <class ReleaseOne> uint64_t releaseUnits(uint64_t bytes, ReleaseOne releaseOne) {
  uint64_t released = 0;
  while (released < bytes) {
    const uint64_t unit = releaseOne();
    if (!unit) break;
    released += unit;
  }
  return released;
}

// The store of streamed experts. Its floor chunks, staging and tables are
// its floor; a unit is an elastic chunk, the last live one of the size with
// the most above its floor going first (ExpertStore::releaseAboveFloor), all
// of them in one compaction per size when the bytes asked reach them
// (releaseToFloor), and the next one of the size with the least above its
// floor coming back (growTowardCap). Released while idle, it holds nothing.
class ExpertStorePool final : public ElasticPool {
public:
  explicit ExpertStorePool(ExpertStore &store) noexcept : store_(store) {}

  [[nodiscard]] PoolState state() const override {
    const ExpertStoreStats stats = store_.stats();
    return {.name = "experts",
            .floorBytes = stats.liveBytes ? stats.floorBytes : 0,
            .currentBytes = stats.liveBytes,
            .capBytes = stats.capBytes,
            .releaseUnitBytes = stats.releaseUnitBytes,
            .releaseUnits = stats.releaseUnits,
            .growUnitBytes = stats.growUnitBytes};
  }
  uint64_t release(uint64_t bytes) override {
    const ExpertStoreStats stats = store_.stats();
    if (stats.liveBytes && bytes >= stats.liveBytes - stats.floorBytes) return store_.releaseToFloor();
    return releaseUnits(bytes, [this] { return store_.releaseAboveFloor(); });
  }
  metal::AllocationResult grow(const metal::AllocationAdmission &admit) override {
    return store_.growTowardCap(admit);
  }

private:
  ExpertStore &store_;
};

// The hot block cache of spilled K/V. The floors of the lanes that decode
// over Index pages are its floor; a unit is an extent, given back above the
// floors (KvBlockCache::releaseAboveFloor) and taken toward the planned
// target while a lane holds a floor (growTowardTarget), which its floors may
// take it past. With no lane reading above the floors (none holds one, or
// the target is no more than the floor) its extents cost only their rebuild:
// after the last Spill lane ends they are up to 975 MiB nothing reads.
class HotBlocksPool final : public ElasticPool {
public:
  explicit HotBlocksPool(KvBlockCache &cache) noexcept : cache_(cache) {}

  [[nodiscard]] PoolState state() const override {
    const KvBlockCacheStats stats = cache_.stats();
    const uint64_t extent = cache_.extentBytes();
    return {.name = "hot_kv",
            .floorBytes = stats.floorBytes,
            .currentBytes = stats.bytes,
            .capBytes = std::max(stats.targetBytes, stats.bytes),
            .releaseUnitBytes = stats.bytes > stats.floorBytes ? extent : 0,
            .growUnitBytes = stats.lanes && stats.bytes < stats.targetBytes ? extent : 0,
            .rebuildOnly = !cache_.readsAboveFloor()};
  }
  uint64_t release(uint64_t bytes) override {
    return releaseUnits(bytes, [this] { return cache_.releaseAboveFloor(); });
  }
  metal::AllocationResult grow(const metal::AllocationAdmission &admit) override {
    return cache_.growTowardTarget(admit);
  }

private:
  KvBlockCache &cache_;
};

} // namespace splash::model
