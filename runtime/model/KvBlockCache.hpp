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

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/KvReadStats.hpp"
#include "model/Model.hpp"
#include "model/SsdReaders.hpp"
#include "ops/KvSlotFormat.hpp"
#include "ops/PagedAttention.hpp"

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

  // Between commands. The floor of the lane, whose request decodes over
  // Index pages: the cache grows to required(n), (n + 1) laneSlots per
  // layer for the n lanes that hold one, or is refused with nothing changed.
  // releaseLane() gives it back when the request ends or is suspended.
  [[nodiscard]] ModelMemoryAdmission reserveLane(uint32_t stateLane);
  void releaseLane(uint32_t stateLane) noexcept;
  // Between commands, optionally: at most one extent more toward the target.
  [[nodiscard]] ModelMemoryAdmission growTowardTarget();
  // An idle reclaim: the cache's last extent above its floors, in bytes;
  // zero at its floors. A reclaim that needs more calls again.
  [[nodiscard]] uint64_t releaseAboveFloor();

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
  // totals of every command's.
  void mergeStats(const KvReadStats &stats);
  [[nodiscard]] KvBlockCacheStats stats() const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace splash::model
