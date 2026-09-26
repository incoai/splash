#pragma once

#include "metal/MetalBackend.hpp"
#include "model/DFlashDraft.hpp"
#include "model/Model.hpp"
#include "model/StateLayout.hpp"
#include "model/SlotFile.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace splash::model {

struct GdnParityBuffers final {
  metal::MetalBuffer stateBase;
  metal::MetalBuffer convolutionBase;
  metal::MetalBuffer recurrentBase;
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
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept {
    return actualAllocatedBytes_;
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

struct QwenSlotBuffers final {
  // Views retain the slot's GDN and draft allocations without copying data.
  std::array<GdnParityBuffers, 2> gdn;
  std::vector<DFlashDraftRingLayer> draft;
};

struct QwenLogicalLengths final {
  uint64_t targetTokens = 0;
  uint64_t draftBase = 0;
  uint32_t draftLength = 0;
  uint32_t draftCommitCursor = 0;

  [[nodiscard]] uint64_t draftEnd() const noexcept {
    return draftBase + draftLength;
  }
  [[nodiscard]] bool
  hasCompleteDraftWindow(uint32_t draftCapacity) const noexcept {
    return draftCapacity && draftEnd() == targetTokens &&
           draftLength == std::min<uint64_t>(targetTokens, draftCapacity) &&
           draftCommitCursor == targetTokens % draftCapacity;
  }

  bool operator==(const QwenLogicalLengths &) const = default;
};

struct QwenSlotMetadata final {
  bool assigned = false;
  uint64_t requestId = 0;
  uint32_t activeParity = 0;
  QwenLogicalLengths lengths;
};

class QwenStateStorage;

// One GDN cell plus one draft ring: the buffers a cached state occupies.
struct QwenCacheSlot final {
  std::shared_ptr<QwenGdnCell> gdn;
  std::shared_ptr<DFlashDraftRing> draft;
};

// Free buffers available for reuse: a lane takes two cells and a ring, a
// cached state one of each. Both return them here; idle buffers are released
// only by explicit reclaim.
struct QwenBufferPool final {
  std::vector<std::shared_ptr<QwenGdnCell>> cells;
  std::vector<std::shared_ptr<DFlashDraftRing>> rings;
  // Cleared when the storage goes away; late returns then just free.
  bool open = true;
};

// One host copy of a state on its way to disk. The write reads this copy,
// so the state's own buffers return to the pool as soon as it is taken.
struct StateStaging final {
  struct Free {
    void operator()(void *memory) const noexcept { std::free(memory); }
  };
  std::unique_ptr<std::byte, Free> bytes;
  uint64_t size = 0;
  bool busy = false;
};

// An immutable composite snapshot owns a private copy of the state that
// cannot be recovered from sealed Q8 pages. No mutating buffers are exposed
// after construction; its buffers return to the pool when it is dropped.
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

private:
  QwenCompositeState(std::shared_ptr<QwenBufferPool> pool, QwenCacheSlot slot,
                     CompositeStateLayout layout, QwenLogicalLengths lengths,
                     std::shared_ptr<SlotFile> file,
                     std::shared_ptr<StateStaging> staging);
  QwenCompositeState(CompositeStateLayout layout, QwenLogicalLengths lengths,
                     std::shared_ptr<SlotFile> file,
                     std::shared_ptr<SlotFile::Slot> disk);
  // Copies the spans of one state into staging and starts the write that
  // carries them to disk; the ticket's state() is the disk copy. Null when
  // the tier cannot admit another state.
  [[nodiscard]] static std::unique_ptr<StateOffload>
  write(const std::shared_ptr<SlotFile> &file,
        const std::shared_ptr<StateStaging> &staging,
        const std::vector<std::span<std::byte>> &spans,
        CompositeStateLayout layout, QwenLogicalLengths lengths,
        std::function<void()> completion);

  std::shared_ptr<QwenBufferPool> pool_;
  QwenCacheSlot slot_;
  CompositeStateLayout layout_;
  QwenLogicalLengths lengths_;
  std::shared_ptr<SlotFile> file_;
  std::shared_ptr<StateStaging> staging_;
  std::shared_ptr<SlotFile::Slot> disk_;

  friend class QwenStateStorage;
};

// Live cells retain stable backing; only idle buffers may be reclaimed.
class QwenStateStorage final : public model::StateStorage {
public:
  QwenStateStorage(metal::MetalBackend &backend,
                   metal::AllocationAdmission admitAllocation,
                   CompositeStateLayout layout,
                   std::shared_ptr<SlotFile> file = nullptr);

  ~QwenStateStorage() override;
  QwenStateStorage(const QwenStateStorage &) = delete;
  QwenStateStorage &operator=(const QwenStateStorage &) = delete;

  [[nodiscard]] const QwenSlotBuffers &buffers(uint32_t slot) const;
  [[nodiscard]] const QwenSlotMetadata &metadata(uint32_t slot) const;

  // Activation reuses pooled buffers and admits any missing allocations.
  // It clears parity-zero GDN state and resets draft logical lengths; later
  // transitions overwrite the remaining data. Refusals retain their cause.
  // Release returns the lane's buffers to the pool.
  [[nodiscard]] metal::AllocationResult tryActivateSlot(uint32_t slot, uint64_t requestId);
  void releaseSlot(uint32_t slot, uint64_t requestId);

  // Returns pooled buffers beyond the kept counts to macOS. Active lanes and
  // cached states are never moved or reclaimed.
  [[nodiscard]] uint64_t releaseIdle(uint32_t keepCells,
                                     uint32_t keepRings) noexcept override;
  [[nodiscard]] uint32_t idleCells() const noexcept;
  [[nodiscard]] uint32_t idleRings() const noexcept;
  // What activating a slot allocates: the GDN cells and the draft ring the
  // idle pool lacks, since a slot takes pooled buffers first.
  [[nodiscard]] uint64_t activationBytes() const noexcept;

  // Hot-path metadata operations; neither performs a buffer copy.
  void updateLengths(uint32_t slot, QwenLogicalLengths lengths);
  void swapParity(uint32_t slot);

  // Copies committed state into a pooled or newly admitted cache slot while
  // the lane retains its own cells. Returns nullptr on capacity pressure;
  // dropping a cached state makes its slot available for retry.
  [[nodiscard]] std::shared_ptr<const QwenCompositeState>
  snapshot(uint32_t slot);
  [[nodiscard]] bool canSnapshotToDisk() const noexcept {
    return file_ && file_->writable();
  }
  // Writes the lane's committed state to the disk tier from its own cells,
  // taking no cache slot; the ticket carries the disk copy. Null without a
  // tier that accepts writes, or when the quota cannot admit another state.
  [[nodiscard]] std::unique_ptr<StateOffload>
  snapshotToDisk(uint32_t slot, std::function<void()> completion);
  void restore(uint32_t slot, const CompositeState &state,
               bool restoreDraftState);

  [[nodiscard]] std::unique_ptr<StateRestore> beginRestore(
      uint32_t slot, const CompositeState &state, bool restoreDraftState,
      std::function<void()> completion, std::function<void()> committed);

  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept override {
    return allocations_->bytes.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t actualSlotBytes(uint32_t slot) const;
  [[nodiscard]] CompositeStateLayout layout() const noexcept { return layout_; }

private:
  struct Slot final {
    QwenSlotBuffers buffers;
    QwenSlotMetadata metadata;
    std::array<std::shared_ptr<QwenGdnCell>, 2> gdn;
    std::shared_ptr<DFlashDraftRing> draft;
  };

  [[nodiscard]] Slot &slot(uint32_t index);
  [[nodiscard]] const Slot &slot(uint32_t index) const;
  void validateLengths(const QwenLogicalLengths &lengths,
                       bool cacheSnapshot) const;
  static void requireAssigned(const Slot &slot);
  [[nodiscard]] metal::AllocationResult allocateSlot(uint32_t index);
  [[nodiscard]] std::shared_ptr<QwenGdnCell> acquireCell(std::string_view label);
  [[nodiscard]] std::shared_ptr<DFlashDraftRing>
  acquireRing(std::string_view label);
  [[nodiscard]] std::shared_ptr<QwenGdnCell>
  allocateGdnCell(std::string_view label,
                    metal::AllocationFailure *failure = nullptr);
  [[nodiscard]] std::shared_ptr<DFlashDraftRing>
  allocateDraftRing(std::string_view label,
                    metal::AllocationFailure *failure = nullptr);
  static void refreshViews(Slot &slot);
  void restoreLengths(uint32_t slot, QwenLogicalLengths lengths, bool restoreDraft);
  [[nodiscard]] std::shared_ptr<const QwenCompositeState>
  snapshot(uint32_t slot, QwenLogicalLengths lengths);

  metal::MetalBackend &backend_;
  metal::AllocationAdmission admitAllocation_;
  CompositeStateLayout layout_;
  std::shared_ptr<StateAllocationTracker> allocations_;
  std::shared_ptr<QwenBufferPool> pool_;
  std::array<Slot, ExecutionLimits::maximumBatchWidth> slots_;
  std::shared_ptr<SlotFile> file_;
  std::shared_ptr<StateStaging> staging_;
};

} // namespace splash::model
