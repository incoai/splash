#pragma once

#include "metal/CommandGraph.hpp"
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
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace splash::model {

struct GdnParityBuffers final {
  metal::MetalBuffer stateBase;
  std::vector<metal::MetalBuffer> convolutionLayers;
  std::vector<metal::MetalBuffer> recurrentLayers;
  // GdnStateLayout::auxiliaryBytes after the GDN layers' state; empty when 0.
  metal::MetalBuffer auxiliary;
};

// A cell of `layout` in `base` (layout.cellBytes()): every layer's
// convolution state, then every layer's recurrent state, then the auxiliary.
[[nodiscard]] GdnParityBuffers gdnCellBuffers(metal::MetalBackend &backend, metal::MetalBuffer base,
                                              const GdnStateLayout &layout);
// A cached state's cell of `layout` in `base` (layout.cachedCellBytes()):
// every layer's convolution state, then every layer's recurrent codes
// (ops::StateCodec), then the auxiliary, each recurrentLayers view a layer's
// codes and scales.
[[nodiscard]] GdnParityBuffers cachedCellBuffers(metal::MetalBackend &backend, metal::MetalBuffer base,
                                                 const GdnStateLayout &layout);

// A lane's cell holds the running state; a cached state's its codes.
enum class GdnCellKind : uint8_t { Running, Cached };

// One GDN cell: one buffer of layout.cellBytes(), or cachedCellBytes() for a
// cached state's, and its views. A cell of no bytes (a target without GDN
// layers or other state) has no buffer, views or memory.
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
              GdnStateLayout layout, GdnCellKind kind,
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
  // Where the rows the context window holds for the rings end: prefill
  // writes both, and so does decode where it captures states
  // (Model::capturesInDecode); other decode writes the rings alone, which
  // leaves the window behind.
  uint64_t windowEnd = 0;

  [[nodiscard]] uint64_t draftEnd() const noexcept {
    return draftBase + draftLength;
  }
  [[nodiscard]] bool
  hasCompleteDraftWindow(uint32_t draftCapacity) const noexcept {
    return draftCapacity && draftEnd() == targetTokens &&
           draftLength == std::min<uint64_t>(targetTokens, draftCapacity);
  }
  // The context window holds every row of the rings.
  [[nodiscard]] bool hasCurrentContextWindow() const noexcept {
    return windowEnd == draftEnd();
  }

  bool operator==(const QwenLogicalLengths &) const = default;
};

struct QwenLaneMetadata final {
  uint64_t requestId = 0;
  uint32_t activeParity = 0;
  QwenLogicalLengths lengths;
  // The path of rows of the lane's last decode step it accepted, which its
  // current cell does not hold: they wait on the lane's tape for its next
  // decode step (ops::GdnTapeLane::pendingPath). Zero after a prefill step or
  // a restore.
  uint32_t pendingPath = 0;

  [[nodiscard]] bool assigned() const noexcept { return requestId != 0; }
};

class QwenStateStorage;

// One cached cell plus one context window: the buffers a cached state
// occupies.
struct QwenCachedBuffers final {
  std::shared_ptr<QwenGdnCell> gdn;
  std::shared_ptr<DFlashContextWindow> window;
};

// Free buffers available for reuse: a lane takes two cells, the draft rings
// and a context window, a cached state a cached cell and a window. Both
// return them here; idle buffers are released only by reclaim. A cached cell
// is no lane's cell, nor the other way round. The empty cell of cells of no
// bytes (QwenStateStorage) is never pooled.
struct QwenBufferPool final {
  std::vector<std::shared_ptr<QwenGdnCell>> cells;
  std::vector<std::shared_ptr<QwenGdnCell>> cachedCells;
  std::vector<std::shared_ptr<DFlashDraftRing>> rings;
  std::vector<std::shared_ptr<DFlashContextWindow>> windows;
  // Cleared when the storage goes away; late returns then just free.
  bool open = true;

  // Takes back a cell to `into`, unless it is the empty cell.
  static void giveCell(std::vector<std::shared_ptr<QwenGdnCell>> &into, std::shared_ptr<QwenGdnCell> cell) {
    if (cell->buffers().stateBase)
      into.push_back(std::move(cell));
  }
  // Takes back the buffers of a cached state.
  void give(QwenCachedBuffers buffers) {
    if (!open)
      return;
    giveCell(cachedCells, std::move(buffers.gdn));
    windows.push_back(std::move(buffers.window));
  }
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

// The buffers of one cached state, taken before a command writes a state
// into them (QwenStateStorage::reserve): the command writes the codes of its
// recurrent layers and, for a state inside a prefill's rows or a decode
// cycle's, its convolution rows, auxiliary state and window. Sealing moves
// them into that state; a reservation dropped unsealed returns them to the
// pool, which the command's ticket holds off until the GPU is done with them.
class QwenStateReservation final : public StateReservation {
public:
  ~QwenStateReservation() override;
  QwenStateReservation(const QwenStateReservation &) = delete;
  QwenStateReservation &operator=(const QwenStateReservation &) = delete;

  // Its buffers until it is sealed.
  [[nodiscard]] const QwenCachedBuffers &buffers() const;

private:
  QwenStateReservation(std::shared_ptr<QwenBufferPool> pool, QwenCachedBuffers buffers);

  std::shared_ptr<QwenBufferPool> pool_;
  QwenCachedBuffers buffers_;

  friend class QwenStateStorage;
};

// A request's state at a page-aligned boundary, either in RAM (a pooled
// cached cell and context window, returned to the pool when the last
// reference drops) or on disk (one slot of the state file), its recurrent
// layers as codes either way (ops::StateCodec). It cannot be rebuilt from KV
// pages of either format; nothing mutable is exposed.
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
  [[nodiscard]] bool writable() const noexcept override { return !disk_ && file_; }
  [[nodiscard]] bool canOffload(bool inUse) const noexcept override { return writable() && file_->takesWrite(inUse); }
  [[nodiscard]] std::unique_ptr<StateOffload>
  offload(std::function<void()> completion) const override;
  [[nodiscard]] std::unique_ptr<StateOffload>
  persist(std::function<void()> completion) const override;
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
  // Fills staging with one state (`stage`, which writes a state's slot
  // bytes there) and starts the write that carries it to disk; the ticket's
  // state() is the disk copy. Null when the quota cannot admit another
  // state, with nothing staged. The engine starts one state write at a time
  // (StateCache::startWrite), so the staging copy is free here.
  [[nodiscard]] static std::unique_ptr<StateOffload>
  write(const std::shared_ptr<SlotFile> &file,
        const std::shared_ptr<StateStaging> &staging,
        const std::function<void(std::span<std::byte>)> &stage,
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

// Live cells keep their buffers; only idle buffers may be reclaimed. Cells of
// no bytes are all one empty cell (QwenGdnCell), which every lane and state
// shares: it holds nothing to read or write, and nothing to pool or reclaim.
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
  // An assigned lane's buffers, read through the cells, the rings and the
  // window it holds: current() is the GDN cell its next transition reads,
  // next() the one that transition writes, draft() its draft rings and
  // window() their context window. swapParity() and swapDecodeParity()
  // exchange current and next.
  [[nodiscard]] const GdnParityBuffers &current(uint32_t lane) const;
  [[nodiscard]] const GdnParityBuffers &next(uint32_t lane) const;
  [[nodiscard]] const std::vector<DFlashDraftRingLayer> &draft(uint32_t lane) const;
  [[nodiscard]] const metal::MetalBuffer &window(uint32_t lane) const;

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
  // A RAM-only continuation, not a prefix-cache snapshot. The caller drains
  // commands first and stores every lane-indexed scratch value in scratch().
  // Holding it retains both parities, draft rings/window and pending-row
  // metadata under the same allocation ledger. No buffers enter the idle pool.
  class ParkedLane final {
  public:
    ~ParkedLane();
    ParkedLane(const ParkedLane &) = delete;
    ParkedLane &operator=(const ParkedLane &) = delete;
    [[nodiscard]] const QwenLaneMetadata &metadata() const noexcept;
    [[nodiscard]] const metal::MetalBuffer &scratch() const noexcept;
  private:
    struct Impl;
    explicit ParkedLane(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class QwenStateStorage;
  };
  // All allocation and saveScratch complete before the source is detached.
  // Refusal (null) or a throwing save leaves the assigned lane unchanged.
  [[nodiscard]] std::unique_ptr<ParkedLane> parkLane(
      uint32_t lane, uint64_t requestId, uint64_t scratchBytes,
      const std::function<void(const metal::MetalBuffer &)> &saveScratch,
      metal::AllocationFailure &refusal);
  // Caller restores scratch into the free destination first. Success consumes
  // the handle; wrong owner/storage or an occupied destination consumes nothing.
  void restoreParkedLane(uint32_t lane, uint64_t requestId,
                         std::unique_ptr<ParkedLane> &parked);
  // Swaps an assigned lane with a parked owner's buffers, without allocation.
  // The caller has already exchanged lane-indexed scratch. Neither handle is
  // consumed; the parked handle now owns the outgoing resident's state.
  void exchangeParkedLane(uint32_t lane, uint64_t residentId, uint64_t parkedId, ParkedLane &parked);
  // A lane that starts at length zero without a restore reads zero recurrent
  // state; its first transition overwrites the other parity.
  void clearForColdStart(uint32_t lane);
  // A lane whose request found a row that is not finite may hold such rows in
  // its draft rings. Draft attention weighs the ring positions a request has
  // not written by zero, so one would reach the lane's next request as
  // 0 x NaN: the rings are cleared before the lane is released. Its GDN cells
  // and context window need no such care: a start clears or overwrites the
  // cell it reads, a transition writes the other cell before reading it, and
  // rings are computed only from window rows their request wrote.
  void clearDraftRings(uint32_t lane);

  // Returns one pooled buffer to macOS, a lane's cell before a cached one,
  // cells before draft rings and rings before a window, keeping with
  // keepLane the buffers the next lane starts from (keptForStart): what a
  // reclaim step for a denied allocation releases. Zero when none is left to
  // give. Active lanes and cached states are never moved or reclaimed.
  [[nodiscard]] uint64_t releaseOneIdle(bool keepLane) noexcept;
  [[nodiscard]] uint32_t idleCells() const noexcept;
  [[nodiscard]] uint32_t idleCachedCells() const noexcept;
  [[nodiscard]] uint32_t idleRings() const noexcept;
  [[nodiscard]] uint32_t idleWindows() const noexcept;
  // What activating a lane lacks in the idle pool, in cached states: each
  // holds one cached cell and one context window, a lane two cells (none of
  // the pool's when cells have no bytes), a window and the draft rings. No
  // cached state holds a lane's cell or the rings: while the pool lacks
  // either, no number of states makes up the lack, and it gives no value.
  [[nodiscard]] std::optional<uint32_t> statesToActivate() const noexcept;
  // What releaseOneIdle(keepLane), called until it gives none, releases
  // once `evictedStates` more cached states have returned their cached cell
  // and context window to the pool (QwenBufferPool::give), at the layout's
  // sizes, which each allocation covers: with keepLane less what the next
  // lane starts from (keptForStart), and cells of no bytes count none. With
  // `afterStart`, of the pool a lane's activation leaves once it took its
  // buffers from it first (startBytes).
  [[nodiscard]] uint64_t idleBytes(uint32_t evictedStates, bool afterStart, bool keepLane) const noexcept;
  // What a lane's activation allocates: its buffers that the idle pool
  // lacks.
  [[nodiscard]] uint64_t startBytes() const noexcept;

  // Hot-path metadata operations; none performs a buffer copy. A step makes
  // next() current. swapParity: a prefill step's, which read the lane's
  // whole state (no pending rows) and holds its every row. swapDecodeParity:
  // a decode step's, which folded the pending path in and holds the state
  // before its own rows, whose accepted `path` (a row mask of its block, its
  // anchor's row first, metal/abi/VerifyTree.h) waits on the tape.
  void updateLengths(uint32_t lane, QwenLogicalLengths lengths);
  void swapParity(uint32_t lane);
  void swapDecodeParity(uint32_t lane, uint32_t path);

  // Stores committed state into a pooled or newly admitted cache slot while
  // the lane retains its own cells; it throws for a lane with pending rows,
  // whose cell does not hold its state (a decode cycle folds those rows into
  // the state it captures instead: ops::GDN::addFold).
  // Returns nullptr on capacity pressure, with nothing allocated; dropping a
  // cached state makes its slot available for retry.
  [[nodiscard]] std::shared_ptr<const QwenCompositeState>
  snapshot(uint32_t lane);
  // The steps of a snapshot, for a state a command captures
  // (Model::reserveState): reserve() takes a cache slot, or returns nullptr
  // on capacity pressure with nothing allocated and the admission's cause
  // in `refusal`; storeState() stores the lane's committed state into it,
  // running a command of the codec for its recurrent layers, which no other
  // command may be in flight beside; copyBeside() copies the rest alone,
  // the convolution rows, the auxiliary state and the context window, of a
  // state whose codes the command that captures it wrote; copyWindow() the
  // lane's context window alone; seal() moves the slot's buffers into the
  // cached state of `lengths`, which leaves the reservation empty.
  [[nodiscard]] std::shared_ptr<QwenStateReservation> reserve(metal::AllocationFailure &refusal);
  // The codec's quantize and decode of the lane's cell, its other cell
  // standing in for a cached one: what preparing their pipelines reads, which
  // nothing runs.
  void addCodecPreparation(metal::CommandGraph &graph, uint32_t lane) const;
  void storeState(uint32_t lane, QwenStateReservation &reservation);
  void copyBeside(uint32_t lane, QwenStateReservation &reservation) const;
  void copyWindow(uint32_t lane, QwenStateReservation &reservation) const;
  [[nodiscard]] std::shared_ptr<const QwenCompositeState>
  seal(QwenStateReservation &reservation, QwenLogicalLengths lengths) const;
  // Throws unless `lengths` are a lane's, and with cacheSnapshot a cached
  // state's: page-aligned, with a whole draft window and its context window.
  void validateLengths(const QwenLogicalLengths &lengths,
                       bool cacheSnapshot) const;
  [[nodiscard]] bool canSnapshotToDisk(bool inUse) const noexcept {
    return file_ && file_->takesWrite(inUse);
  }
  // Writes the lane's committed state to the disk tier from its own cells,
  // taking no cache slot, its codes made in the staging buffer by a command
  // of the codec; the ticket carries the disk copy. Null without a tier that
  // takes it (canSnapshotToDisk(true)), or when the quota cannot admit
  // another state. The caller asked the write budget first.
  [[nodiscard]] std::unique_ptr<StateOffload>
  snapshotToDisk(uint32_t lane, std::function<void()> completion);
  // Restores `state` into the lane's current cell, its recurrent layers
  // decoded by a command of the codec, which no other command may be in
  // flight beside, and into its context window when restoreDraftState: the
  // draft rings are computed from it again
  // (DFlashDraft::addContextRebuild). A RAM state is restored now, runs
  // `committed` and returns null; a disk state returns the read, which lands
  // in the lane's other cell, free until its next transition writes it, and
  // whose finish() decodes it from there and runs `committed`. The copy of
  // a state that read promotes is the read's bytes, codes as they came.
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
    std::shared_ptr<QwenGdnCell> cached;
    std::shared_ptr<DFlashDraftRing> draft;
    std::shared_ptr<DFlashContextWindow> window;
  };
  struct Lane final {
    QwenLaneMetadata metadata;
    Buffers cells;
  };
  // Counts of the pool's buffers: a lane takes two GDN cells, the draft
  // rings and a context window, and a cached state a cached cell and a
  // window.
  struct BufferCounts final {
    uint32_t cells = 0, cachedCells = 0, rings = 0, windows = 0;
  };
  static constexpr BufferCounts kLaneBuffers{CompositeStateLayout::kLaneGdnCells, 0, 1, 1};
  static constexpr BufferCounts kStateBuffers{0, 1, 0, 1};

  [[nodiscard]] Lane &lane(uint32_t index);
  [[nodiscard]] const Lane &lane(uint32_t index) const;
  static void requireAssigned(const Lane &lane);
  // Assigned, with no decoded rows pending: its current cell holds its state.
  static void requireCommitted(const Lane &lane);
  // A lane's or a cached state's buffers (`counts`): the pool's, and one
  // admission for everything the pool lacks and for the caller's extra. A
  // refusal allocates nothing and takes nothing from the pool.
  [[nodiscard]] metal::AllocationResult
  acquire(BufferCounts counts, std::string_view label, Buffers &buffers, uint64_t extraBytes = 0,
          const std::function<void()> &allocateExtra = {});
  [[nodiscard]] BufferCounts idle() const noexcept;
  // The pooled buffers keepLane keeps for the next lane's start: a lane's
  // two cells, draft rings and window while no lane holds its own, and none
  // beside a lane that does (or one `afterStart` makes), whose own the next
  // start takes once it ends. A lane's cells serve no cached state, which
  // takes codes, so another lane's kept beside one in service stayed past
  // every cached unit a KV admission evicted: after two concurrent chats at
  // a 22 GB limit, 22 cached 4K-token chats where 24 fit before them.
  [[nodiscard]] BufferCounts keptForStart(bool keepLane, bool afterStart) const noexcept;
  // The buffers of `count` beyond those of `taken`, each kind apart.
  [[nodiscard]] static BufferCounts beyond(BufferCounts count, BufferCounts taken) noexcept;
  // Their bytes at the layout's sizes.
  [[nodiscard]] uint64_t bytes(BufferCounts buffers) const noexcept;
  void restore(uint32_t lane, const QwenCompositeState &state,
               bool restoreDraftState);
  void restoreLengths(uint32_t lane, QwenLogicalLengths lengths, bool restoreDraft);
  [[nodiscard]] std::shared_ptr<const QwenCompositeState>
  snapshot(uint32_t lane, QwenLogicalLengths lengths);
  // A cached state of the lane's committed state at `lengths`: a reserved
  // cache slot that `store` fills, sealed; nullptr on capacity pressure.
  [[nodiscard]] std::shared_ptr<const QwenCompositeState>
  keep(uint32_t lane, QwenLogicalLengths lengths, const std::function<void(QwenStateReservation &)> &store);
  // The cached cell a disk state's read lands in: the start of the lane's
  // other cell.
  [[nodiscard]] GdnParityBuffers landing(uint32_t lane) const;
  // A cached state's copy of a disk state read into the lane (landing),
  // codes as they came, and the lane's window.
  [[nodiscard]] std::shared_ptr<const QwenCompositeState>
  promoteRead(uint32_t lane, QwenLogicalLengths lengths);
  // The codec's dispatch between a lane's cell and a cached one.
  void addCode(metal::CommandGraph &graph, bool quantize, const GdnParityBuffers &cell,
               const GdnParityBuffers &cached) const;
  // Runs one command of the codec between a lane's cell and a cached cell:
  // `quantize` codes the lane's recurrent layers into the cached cell's,
  // otherwise the cached cell's codes are decoded into the lane's. The
  // convolution rows and the auxiliary state go along on the CPU.
  void code(bool quantize, const GdnParityBuffers &cell, const GdnParityBuffers &cached);

  metal::MetalBackend &backend_;
  metal::AllocationAdmission admitAllocation_;
  CompositeStateLayout layout_;
  std::shared_ptr<StateAllocationTracker> allocations_;
  std::shared_ptr<QwenBufferPool> pool_;
  std::array<Lane, ExecutionLimits::maximumBatchWidth> lanes_;
  std::shared_ptr<SlotFile> file_;
  std::shared_ptr<StateStaging> staging_;
  // Every lane's and state's cell when cells have no bytes; null otherwise.
  std::shared_ptr<QwenGdnCell> emptyCell_;
};

} // namespace splash::model
