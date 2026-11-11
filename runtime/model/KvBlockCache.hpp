#pragma once

// The hot block cache of spilled K/V: one cache, which every lane of a
// block-sparse target shares, of the blocks its verify rows select on pages
// that hold only their index (kv::PageForm::Index). After a sparse layer's
// block lists, one host step resolves each entry of every lane's list: a
// block of a Full page is read through the page table, as without spilling;
// any other from its slot of the cache, which a missed block is read into
// first, once for every lane that lists it, with direct reads of its tier
// copy (KvCopy::readAt) on the SSD readers, ahead of everything else they
// read: one read of a page's records from its first missed block to its
// last. The entries' locations tell the attention where
// (ops::AttentionBlocks::locations).
//
// The cache is pages of the pool's layout without an index
// (kv::PageStorage): slot s of layer l is place s & 7 of cache page s >> 3,
// and a location names that page of locationPages and the place. A slot is
// keyed by the copy's id and the block's place, and recycled by adaptive
// replacement per layer (SlotRecycler); a resolve never recycles a slot it
// touched itself, and holds at least as many slots per layer as the lanes
// that hold a floor select blocks (laneSlots each, reserveLane), so a resolve
// always places every block it reads. Its memory is whole extents of the
// pool's pages: the floors of the lanes that hold one, required(n) = (n + 1)
// laneSlots slots for n of them, then, while any does, one extent at a time
// toward the planned target; an idle reclaim gives back what is above the
// floors.
//
// A block that cannot be read (twice) leaves zeros in its slot, which is
// freed, and fails every lane that listed it (takeFailures), not the
// command.
//
// Threads: the host steps of the command in flight own the directory, and
// the engine thread owns it between commands (resizing, merging statistics);
// a step owns what it reads, so that one its command gave up may still run.

#include "Checked.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "metal/abi/PagedAttention.h"
#include "model/KvReadStats.hpp"
#include "model/Model.hpp"
#include "model/SsdReaders.hpp"
#include "ops/KvSlotFormat.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace splash::model {

struct KvBlockCacheOptions final {
  // The slots per layer one decoding lane's rows select at most
  // (SparseKvShape::verifyRows x selectedBlocks).
  uint32_t laneSlots = 0;
  // The size the memory plan gives the cache, which it grows toward when
  // lanes hold floors.
  uint64_t targetBytes = 0;
  std::shared_ptr<SsdReaders> readers;
  // The pool layout's (kv::SlotFormat(layout)).
  kv::SlotFormat format;
};

class KvBlockCache final {
public:
  // `layout` and `extentPages` are the pool's; the cache's pages come from
  // `admission`, as KV memory. Throws std::invalid_argument for a layout
  // that cannot spill (kv::Layout::spillable), a format of another layout,
  // no readers or no lane slots.
  KvBlockCache(metal::MetalBackend &backend, metal::AllocationAdmission admission, const kv::Layout &layout,
               uint32_t extentPages, KvBlockCacheOptions options);
  ~KvBlockCache();
  KvBlockCache(const KvBlockCache &) = delete;
  KvBlockCache &operator=(const KvBlockCache &) = delete;
  // The page table a cache of a pool's `layout`, `laneSlots` and
  // `targetBytes` (KvBlockCacheOptions) maps its pages through, which it
  // allocates as it is built and holds until it goes: an entry for each page
  // of the most extents it holds, its target's or the floors of every lane at
  // once, in whole host pages, as Metal allocates them, in whichever extent
  // size the pool takes (kv::Layout::mostOverExtents), so that the memory
  // plan, made before the pool's, counts it (KvSpillPlan::tableBytes);
  // 32,768 bytes for Qwen3.8-Flash-Next's ten 256-page extents of int8 KV.
  [[nodiscard]] static constexpr uint64_t tableBytes(const kv::Layout &layout, uint32_t laneSlots,
                                                     uint64_t targetBytes) {
    return layout.mostOverExtents([&](uint32_t extentPages) {
      return alignUp(mostPages(layout, extentPages, laneSlots, targetBytes) * sizeof(SplashKvPage));
    });
  }

  // Between commands. The floor of the lane, whose request decodes over
  // Index pages: the cache grows to required(n), (n + 1) laneSlots per
  // layer for the n lanes that hold one, or is refused with nothing changed.
  // releaseLane() gives it back when the request ends or is suspended.
  [[nodiscard]] ModelMemoryAdmission reserveLane(uint32_t stateLane);
  void releaseLane(uint32_t stateLane) noexcept;
  // Whether the lane holds its floor: reserveLane() granted it, and
  // releaseLane() has not given it back.
  [[nodiscard]] bool holdsLane(uint32_t stateLane) const;
  // Whether a lane reads cache above the floors: a lane holds a floor and
  // the target is past the floors. Else the extents above the floors cost
  // only their rebuild.
  [[nodiscard]] bool readsAboveFloor() const noexcept;
  // The bytes the floors of `lanes` more lanes need together, in whole
  // extents once for them all, past the extents that stay for them: all
  // while a lane reads above the floors, else the floors held.
  [[nodiscard]] uint64_t lanesShortfallBytes(uint32_t lanes) const;
  // Between commands, optionally: at most one extent more toward the target,
  // under `admit`.
  [[nodiscard]] metal::AllocationResult growTowardTarget(const metal::AllocationAdmission &admit);
  // An idle reclaim: the cache's last extent above its floors, in bytes;
  // zero at its floors. A reclaim that needs more calls again.
  [[nodiscard]] uint64_t releaseAboveFloor();
  // The bytes of one extent, what the cache grows and gives back at a time.
  [[nodiscard]] uint64_t extentBytes() const noexcept;
  // The floors of `lanes` lanes in whole extents, whatever the cache holds:
  // zero for none.
  [[nodiscard]] uint64_t floorBytes(uint32_t lanes) const noexcept;

  // After `layer`'s block lists: a host step that resolves the entries of
  // lane l's list, whose pages' sources are lanes[l] (null for a lane none
  // of whose pages holds only its index), writes their locations to
  // `locations` (an array of the lists' shape) and counts into `stats`.
  // Returns the lists with their locations, for the located split.
  [[nodiscard]] ops::AttentionBlocks addResolve(metal::CommandGraph &graph, uint32_t layer,
                                                std::span<const std::shared_ptr<const KvPageSources>> lanes,
                                                const ops::AttentionBlocks &blocks,
                                                const metal::MetalBuffer &locations,
                                                std::shared_ptr<KvReadStats> stats);
  // Once the command completed: the blocks its resolves could not read, each
  // for every lane that listed it. Their slots were freed.
  [[nodiscard]] std::vector<KvReadFailure> takeFailures();

  // Engine thread, once a command completed: its resolves' counts, and the
  // totals of every command's. Returns the bytes its resolves read.
  uint64_t mergeStats(const KvReadStats &stats);
  [[nodiscard]] KvBlockCacheStats stats() const;

private:
  // A cache's extents: those of the floors of `lanes` lanes, (lanes + 1)
  // laneSlots slots a layer at 2^SPLASH_ATTENTION_LOCATION_BLOCK_BITS a page;
  // those of a target of `targetBytes`; and the pages of the most it holds,
  // its target's or the floors of every lane at once.
  [[nodiscard]] static constexpr uint32_t floorExtents(uint32_t lanes, uint32_t laneSlots,
                                                       uint32_t extentPages) noexcept {
    const uint64_t extentSlots = uint64_t{extentPages} << SPLASH_ATTENTION_LOCATION_BLOCK_BITS;
    return lanes ? static_cast<uint32_t>(((uint64_t{lanes} + 1) * laneSlots + extentSlots - 1) / extentSlots) : 0;
  }
  [[nodiscard]] static constexpr uint32_t targetExtents(const kv::Layout &layout, uint32_t extentPages,
                                                        uint64_t targetBytes) noexcept {
    return static_cast<uint32_t>(targetBytes / (uint64_t{extentPages} * layout.kvBytesPerModelPage()));
  }
  [[nodiscard]] static constexpr uint64_t mostPages(const kv::Layout &layout, uint32_t extentPages,
                                                    uint32_t laneSlots, uint64_t targetBytes) noexcept {
    return uint64_t{std::max(targetExtents(layout, extentPages, targetBytes),
                             floorExtents(ExecutionLimits::maximumBatchWidth, laneSlots, extentPages))} *
           extentPages;
  }

  struct Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace splash::model
