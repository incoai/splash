#pragma once

#include "metal/MetalBackend.hpp"
#include "model/DFlashDraft.hpp"
#include "model/Model.hpp"
#include "model/StateLayout.hpp"
#include "model/SlotFile.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace splash::model {

struct GdnParityBuffers final {
  metal::MetalBuffer stateBase;
  std::vector<metal::MetalBuffer> convolutionLayers;
  std::vector<metal::MetalBuffer> recurrentLayers;
};

class QwenGdnCell final {
public:
  ~QwenGdnCell();
  QwenGdnCell(const QwenGdnCell &) = delete;
  QwenGdnCell &operator=(const QwenGdnCell &) = delete;

  [[nodiscard]] const GdnParityBuffers &buffers() const noexcept {
    return buffers_;
  }

private:
  QwenGdnCell(metal::MetalBackend &backend,
              std::shared_ptr<StateAllocationTracker> tracker,
              GdnStateLayout layout,
              std::string_view label);

  std::shared_ptr<StateAllocationTracker> tracker_;
  GdnParityBuffers buffers_;
  uint64_t actualAllocatedBytes_ = 0;

  friend class QwenStateStorage;
};

struct QwenLogicalLengths final {
  uint64_t targetTokens = 0;
  uint64_t draftBase = 0;
  uint32_t draftLength = 0;

  [[nodiscard]] uint64_t draftEnd() const noexcept {
    return draftBase + draftLength;
  }
  [[nodiscard]] bool
  hasCompleteDraftWindow(uint32_t draftCapacity) const noexcept {
    return draftCapacity && draftEnd() == targetTokens &&
           draftLength == std::min<uint64_t>(targetTokens, draftCapacity);
  }

  bool operator==(const QwenLogicalLengths &) const = default;
};

struct QwenLaneMetadata final {
  uint64_t requestId = 0;
  uint32_t activeParity = 0;
  QwenLogicalLengths lengths;

  [[nodiscard]] bool assigned() const noexcept { return requestId != 0; }
};

class QwenStateStorage;

// One GDN cell plus one draft ring: the buffers a cached state occupies.
struct QwenCachedBuffers final {
  std::shared_ptr<QwenGdnCell> gdn;
  std::shared_ptr<DFlashDraftRing> draft;
};

// Free buffers available for reuse: a lane takes two cells and a ring, a
// cached state one of each. Both return them here; idle buffers are released
// only by reclaim.
struct QwenBufferPool final {
  std::vector<std::shared_ptr<QwenGdnCell>> cells;
  std::vector<std::shared_ptr<DFlashDraftRing>> rings;
  // Cleared when the storage goes away; late returns then just free.
  bool open = true;
};

// One copy of a state on its way to disk, in a buffer of the backend's like
// every other: resident, and set aside by the memory plan. The write reads
// this copy, so the state's own buffers return to the pool as soon as it is
// taken.
struct StateStaging final {
  metal::MetalBuffer buffer;

  [[nodiscard]] std::span<std::byte> bytes() const {
    return {static_cast<std::byte *>(buffer.contents()), buffer.sizeBytes()};
  }
};

// A copy of one lane's committed state, either in RAM (a pooled GDN cell and
// draft ring, returned to the pool when the last reference drops) or on disk
// (one slot of the state file). It cannot be rebuilt from KV pages of either
// format; nothing mutable is exposed.
class QwenCompositeState final : public CompositeState {
public:
  ~QwenCompositeState() override;
  QwenCompositeState(const QwenCompositeState &) = delete;
  QwenCompositeState &operator=(const QwenCompositeState &) = delete;

  [[nodiscard]] uint64_t bytes() const noexcept override {
    return layout_.cachedBytes();
  }
  [[nodiscard]] uint64_t residentBytes() const noexcept override {
    return disk_ ? 0 : bytes();
  }
  [[nodiscard]] bool canOffload() const noexcept override {
    return !disk_ && file_ && file_->writable();
  }
  [[nodiscard]] std::unique_ptr<StateOffload>
  offload(std::function<void()> completion) const override;
  void label(std::vector<std::byte> label) const override;

private:
  QwenCompositeState(std::shared_ptr<QwenBufferPool> pool,
                     QwenCachedBuffers buffers, CompositeStateLayout layout,
                     QwenLogicalLengths lengths,
                     std::shared_ptr<SlotFile> file,
                     std::shared_ptr<StateStaging> staging);
  QwenCompositeState(CompositeStateLayout layout, QwenLogicalLengths lengths,
                     std::shared_ptr<SlotFile> file,
                     std::shared_ptr<SlotFile::Slot> disk);
  // Copies the spans of one state into staging and starts the write that
  // carries them to disk; the ticket's state() is the disk copy. Null when
  // the tier cannot admit another state. The engine starts one state write
  // at a time (StateCache::startWrite), so the staging copy is free here.
  [[nodiscard]] static std::unique_ptr<StateOffload>
  write(const std::shared_ptr<SlotFile> &file,
        const std::shared_ptr<StateStaging> &staging,
        const std::vector<std::span<std::byte>> &spans,
        CompositeStateLayout layout, QwenLogicalLengths lengths,
        std::function<void()> completion);

  std::shared_ptr<QwenBufferPool> pool_;
  QwenCachedBuffers buffers_;
  CompositeStateLayout layout_;
  QwenLogicalLengths lengths_;
  std::shared_ptr<SlotFile> file_;
  std::shared_ptr<StateStaging> staging_;
  std::shared_ptr<SlotFile::Slot> disk_;

  friend class QwenStateStorage;
};

// Live cells keep their buffers; only idle buffers may be reclaimed.
class QwenStateStorage final {
public:
  // The file, when given, holds one state per slot and shares the cache's
  // disk budget.
  QwenStateStorage(metal::MetalBackend &backend,
                   metal::AllocationAdmission admitAllocation,
                   CompositeStateLayout layout,
                   std::shared_ptr<SlotFile> file);

  ~QwenStateStorage();
  QwenStateStorage(const QwenStateStorage &) = delete;
  QwenStateStorage &operator=(const QwenStateStorage &) = delete;

  [[nodiscard]] const QwenLaneMetadata &metadata(uint32_t lane) const;
  // An assigned lane's buffers, read through the cells and the ring it
  // holds: current() is the GDN cell its next transition reads, next() the
  // one that transition writes, draft() its draft ring. swapParity()
  // exchanges current and next.
  [[nodiscard]] const GdnParityBuffers &current(uint32_t lane) const;
  [[nodiscard]] const GdnParityBuffers &next(uint32_t lane) const;
  [[nodiscard]] const std::vector<DFlashDraftRingLayer> &draft(uint32_t lane) const;

  // Activation takes pooled buffers and asks the governor once for all the
  // pool lacks, together with `extraBytes` for what else the request's start
  // allocates (`allocateExtra`, run first in the same admission). A start is
  // only useful whole, so a refusal allocates nothing, leaves the pool as it
  // was and retains its cause. Activation resets the logical lengths; a
  // lane that starts from length zero clears its current cell with
  // clearForColdStart before its first transition. Release returns the
  // lane's buffers to the pool.
  [[nodiscard]] metal::AllocationResult
  tryActivateLane(uint32_t lane, uint64_t requestId, uint64_t extraBytes = 0,
                  const std::function<void()> &allocateExtra = {});
  void releaseLane(uint32_t lane, uint64_t requestId);
  // A lane that starts at length zero without a restore reads zero recurrent
  // state; its first transition overwrites the other parity.
  void clearForColdStart(uint32_t lane);

  // Returns one pooled buffer to macOS, a cell before a ring, keeping one
  // lane's cells and ring when keepLane: what a reclaim step for a denied
  // allocation releases. Zero when none is left to give. Active lanes and
  // cached states are never moved or reclaimed.
  [[nodiscard]] uint64_t releaseOneIdle(bool keepLane) noexcept;
  [[nodiscard]] uint32_t idleCells() const noexcept;
  [[nodiscard]] uint32_t idleRings() const noexcept;
  // What activating a lane lacks in the idle pool, in cached states: each
  // holds one GDN cell and one draft ring, a lane two cells and a ring.
  [[nodiscard]] uint32_t statesToActivate() const noexcept;

  // Hot-path metadata operations; neither performs a buffer copy.
  void updateLengths(uint32_t lane, QwenLogicalLengths lengths);
  void swapParity(uint32_t lane);

  // Copies committed state into a pooled or newly admitted cache slot while
  // the lane retains its own cells. Returns nullptr on capacity pressure,
  // with nothing allocated; dropping a cached state makes its slot available
  // for retry.
  [[nodiscard]] std::shared_ptr<const QwenCompositeState>
  snapshot(uint32_t lane);
  [[nodiscard]] bool canSnapshotToDisk() const noexcept {
    return file_ && file_->writable();
  }
  // Writes the lane's committed state to the disk tier from its own cells,
  // taking no cache slot; the ticket carries the disk copy. Null without a
  // tier that accepts writes, or when the quota cannot admit another state.
  [[nodiscard]] std::unique_ptr<StateOffload>
  snapshotToDisk(uint32_t lane, std::function<void()> completion);
  // Restores `state` into the lane's current cell, and into its draft ring
  // when restoreDraftState. A RAM state is copied now, runs `committed` and
  // returns null; a disk state returns the read, whose finish() runs it.
  [[nodiscard]] std::unique_ptr<StateRestore> beginRestore(
      uint32_t lane, const CompositeState &state, bool restoreDraftState,
      std::function<void()> completion, std::function<void()> committed);
  // Takes back the disk copy of a state at `tokens` that an earlier process
  // recorded in a persistent tier's file. A cached state holds a complete
  // draft window, so its lengths follow from its boundary.
  [[nodiscard]] std::shared_ptr<const CompositeState> adopt(const SlotRecord &record,
                                                            uint32_t tokens);

  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept {
    return allocations_->bytes.load(std::memory_order_relaxed);
  }
  // The buffer a state's write to the disk tier stages through; zero without
  // a tier.
  [[nodiscard]] uint64_t stagingBytes() const noexcept {
    return staging_ ? staging_->buffer.sizeBytes() : 0;
  }
  [[nodiscard]] CompositeStateLayout layout() const noexcept { return layout_; }

private:
  struct Buffers final {
    std::array<std::shared_ptr<QwenGdnCell>,
               CompositeStateLayout::kLaneGdnCells>
        gdn;
    std::shared_ptr<DFlashDraftRing> draft;
  };
  struct Lane final {
    QwenLaneMetadata metadata;
    Buffers cells;
  };

  [[nodiscard]] Lane &lane(uint32_t index);
  [[nodiscard]] const Lane &lane(uint32_t index) const;
  void validateLengths(const QwenLogicalLengths &lengths,
                       bool cacheSnapshot) const;
  static void requireAssigned(const Lane &lane);
  // `cells` GDN cells and a draft ring: the pool's buffers, and one
  // admission for everything the pool lacks and for the caller's extra. A
  // refusal allocates nothing and takes nothing from the pool.
  [[nodiscard]] metal::AllocationResult
  acquire(uint32_t cells, std::string_view label, Buffers &buffers,
          uint64_t extraBytes = 0, const std::function<void()> &allocateExtra = {});
  // The bytes of the cells and the ring the pool lacks of that.
  [[nodiscard]] uint64_t missingBytes(uint32_t cells) const noexcept;
  void restore(uint32_t lane, const QwenCompositeState &state,
               bool restoreDraftState);
  void restoreLengths(uint32_t lane, QwenLogicalLengths lengths, bool restoreDraft);
  [[nodiscard]] std::shared_ptr<const QwenCompositeState>
  snapshot(uint32_t lane, QwenLogicalLengths lengths);

  metal::MetalBackend &backend_;
  metal::AllocationAdmission admitAllocation_;
  CompositeStateLayout layout_;
  std::shared_ptr<StateAllocationTracker> allocations_;
  std::shared_ptr<QwenBufferPool> pool_;
  std::array<Lane, ExecutionLimits::maximumBatchWidth> lanes_;
  std::shared_ptr<SlotFile> file_;
  std::shared_ptr<StateStaging> staging_;
};

} // namespace splash::model
