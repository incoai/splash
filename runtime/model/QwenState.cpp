#include "model/QwenState.hpp"

#include <cstring>
#include <algorithm>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

using metal::MetalBuffer;

void *writableContents(const MetalBuffer &buffer, const char *name) {
  void *contents = buffer.contents();
  if (!contents) {
    throw std::logic_error(std::string(name) + " is not CPU-visible");
  }
  return contents;
}

void copyExact(const MetalBuffer &destination, const MetalBuffer &source,
               const char *name) {
  if (destination.sizeBytes() != source.sizeBytes()) {
    throw std::logic_error(std::string(name) + " shape mismatch");
  }
  std::memcpy(writableContents(destination, name),
              writableContents(source, name), destination.sizeBytes());
}

void clear(const MetalBuffer &buffer, const char *name) {
  std::memset(writableContents(buffer, name), 0, buffer.sizeBytes());
}

std::vector<std::span<std::byte>> stateSpans(
    const GdnParityBuffers &gdn, const std::vector<DFlashDraftRingLayer> &draft) {
  std::vector<std::span<std::byte>> spans;
  const auto append = [&](const MetalBuffer &buffer) {
    spans.emplace_back(static_cast<std::byte *>(writableContents(buffer, "state IO")),
                       buffer.sizeBytes());
  };
  append(gdn.stateBase);
  for (const auto &layer : draft) {
    append(layer.keys);
    append(layer.values);
  }
  return spans;
}

class FileOffload final : public StateOffload {
public:
  FileOffload(std::shared_ptr<SlotFile::Operation> operation,
              std::shared_ptr<const CompositeState> state,
              std::shared_ptr<StateStaging> staging)
      : operation_(std::move(operation)), state_(std::move(state)),
        staging_(std::move(staging)) {}
  // The staging copy is the write's source until the worker has stopped.
  ~FileOffload() override {
    operation_->drain();
    staging_->busy = false;
  }
  bool ready() const noexcept override { return operation_->ready(); }
  bool finish() override { return operation_->wait(); }
  const std::shared_ptr<const CompositeState> &state() const noexcept override {
    return state_;
  }
private:
  std::shared_ptr<SlotFile::Operation> operation_;
  std::shared_ptr<const CompositeState> state_;
  std::shared_ptr<StateStaging> staging_;
};

class FileRestore final : public StateRestore {
public:
  FileRestore(std::shared_ptr<SlotFile::Operation> operation,
              std::function<void()> committed,
              std::function<std::shared_ptr<const CompositeState>()> snapshot)
      : operation_(std::move(operation)), committed_(std::move(committed)),
        snapshot_(std::move(snapshot)) {}
  ~FileRestore() override { operation_->drain(); }
  bool ready() const noexcept override { return operation_->ready(); }
  void cancel() noexcept override { operation_->cancel(); }
  bool finish() override {
    if (!operation_->wait()) return false;
    committed_();
    finished_ = true;
    return true;
  }
  std::shared_ptr<const CompositeState> snapshot() override {
    return finished_ ? snapshot_() : nullptr;
  }
private:
  std::shared_ptr<SlotFile::Operation> operation_;
  std::function<void()> committed_;
  std::function<std::shared_ptr<const CompositeState>()> snapshot_;
  bool finished_ = false;
};

} // namespace

QwenGdnCell::QwenGdnCell(metal::MetalBackend &backend,
                         std::shared_ptr<StateAllocationTracker> tracker,
                         GdnStateLayout layout, std::string_view label)
    : tracker_(std::move(tracker)) {
  if (!tracker_)
    throw std::invalid_argument("Qwen state allocation tracker is empty");
  if (!layout.valid())
    throw std::invalid_argument("Qwen GDN state layout is invalid");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  buffers_.stateBase = backend.allocateBuffer(
      layout.cellBytes(), metal::BufferStorage::Shared, label);
  buffers_.convolutionBase = backend.view(
      buffers_.stateBase, 0, layout.convolutionBytes());
  buffers_.convolutionLayers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    buffers_.convolutionLayers[layer] =
        backend.view(buffers_.convolutionBase,
                     uint64_t{layer} * layout.convolutionLayerBytes(),
                     layout.convolutionLayerBytes());
  }
  buffers_.recurrentBase =
      backend.view(buffers_.stateBase, layout.convolutionBytes(),
                   layout.recurrentBytes());
  buffers_.recurrentLayers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    buffers_.recurrentLayers[layer] = backend.view(
        buffers_.recurrentBase,
        uint64_t{layer} * layout.recurrentLayerBytes(),
        layout.recurrentLayerBytes());
  }
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (actualAllocatedBytes_ < layout.cellBytes()) {
    throw std::logic_error("Qwen GDN allocation is below declared bytes");
  }
  tracker_->bytes.fetch_add(actualAllocatedBytes_, std::memory_order_relaxed);
}

QwenGdnCell::~QwenGdnCell() {
  tracker_->bytes.fetch_sub(actualAllocatedBytes_, std::memory_order_relaxed);
}

QwenCompositeState::QwenCompositeState(std::shared_ptr<QwenBufferPool> pool,
                                       QwenCacheSlot slot,
                                       CompositeStateLayout layout,
                                       QwenLogicalLengths lengths,
                                       std::shared_ptr<SlotFile> file,
                                       std::shared_ptr<StateStaging> staging)
    : pool_(std::move(pool)), slot_(std::move(slot)), layout_(layout),
      lengths_(lengths), file_(std::move(file)), staging_(std::move(staging)) {
  if (!pool_ || !slot_.gdn || !slot_.draft) {
    throw std::invalid_argument("composite state buffers are empty");
  }
}

QwenCompositeState::QwenCompositeState(CompositeStateLayout layout,
    QwenLogicalLengths lengths, std::shared_ptr<SlotFile> file,
    std::shared_ptr<SlotFile::Slot> disk)
    : layout_(layout), lengths_(lengths), file_(std::move(file)), disk_(std::move(disk)) {}

std::unique_ptr<StateOffload>
QwenCompositeState::offload(std::function<void()> completion) const {
  if (!canOffload()) return {};
  return write(file_, staging_, stateSpans(slot_.gdn->buffers(), slot_.draft->layers()),
               layout_, lengths_, std::move(completion));
}

std::unique_ptr<StateOffload> QwenCompositeState::write(
    const std::shared_ptr<SlotFile> &file, const std::shared_ptr<StateStaging> &staging,
    const std::vector<std::span<std::byte>> &spans, CompositeStateLayout layout,
    QwenLogicalLengths lengths, std::function<void()> completion) {
  if (staging->busy)
    throw std::logic_error("a composite state write is already in flight");
  auto disk = file->acquire();
  if (!disk) return {};
  auto result = std::shared_ptr<const QwenCompositeState>(
      new QwenCompositeState(layout, lengths, file, disk));
  std::byte *staged = staging->bytes.get();
  for (auto span : spans)
    staged = std::copy(span.begin(), span.end(), staged);
  staging->busy = true;
  std::shared_ptr<SlotFile::Operation> operation;
  try {
    operation = file->write(
        std::move(disk), {std::span<const std::byte>(staging->bytes.get(), staging->size)},
        std::move(completion));
    return std::make_unique<FileOffload>(operation, std::move(result), staging);
  } catch (...) {
    if (operation)
      operation->drain();
    staging->busy = false;
    throw;
  }
}

QwenCompositeState::~QwenCompositeState() {
  if (!pool_ || !pool_->open)
    return;
  pool_->cells.push_back(std::move(slot_.gdn));
  pool_->rings.push_back(std::move(slot_.draft));
}

QwenStateStorage::QwenStateStorage(metal::MetalBackend &backend,
                                   metal::AllocationAdmission admitAllocation,
                                   CompositeStateLayout layout,
                                   std::shared_ptr<SlotFile> file)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout),
      allocations_(std::make_shared<StateAllocationTracker>()),
      pool_(std::make_shared<QwenBufferPool>()) {
  if (!admitAllocation_)
    throw std::invalid_argument("Qwen state allocation admission is required");
  if (!layout_.valid() ||
      layout_.draft.tokens != ExecutionLimits::draftContextTokens) {
    throw std::invalid_argument("Qwen composite state layout is invalid");
  }
  if (file && file->slotBytes() != layout_.cachedBytes())
    throw std::invalid_argument("state file slots do not hold one state");
  if (file) {
    file_ = std::move(file);
    staging_ = std::make_shared<StateStaging>();
    void *memory = nullptr;
    if (::posix_memalign(&memory, SlotFile::kAlignmentBytes, layout_.cachedBytes()) != 0)
      throw std::bad_alloc();
    staging_->bytes.reset(static_cast<std::byte *>(memory));
    staging_->size = layout_.cachedBytes();
    // Touch the pages now rather than on the engine thread at the first write.
    std::memset(memory, 0, layout_.cachedBytes());
  }
}

QwenStateStorage::~QwenStateStorage() {
  pool_->open = false;
  pool_->cells.clear();
  pool_->rings.clear();
}

const QwenSlotBuffers &QwenStateStorage::buffers(uint32_t index) const {
  return slot(index).buffers;
}

const QwenSlotMetadata &QwenStateStorage::metadata(uint32_t index) const {
  return slot(index).metadata;
}

metal::AllocationResult QwenStateStorage::tryActivateSlot(uint32_t index, uint64_t requestId) {
  if (!requestId)
    throw std::invalid_argument("request id must be non-zero");
  Slot &current = slot(index);
  if (current.metadata.assigned) {
    throw std::logic_error("Qwen state slot is already assigned");
  }
  if (auto admission = allocateSlot(index); !admission)
    return admission;

  // A fresh recurrent sequence reads parity zero immediately. Parity one is
  // fully overwritten by the first transition. Draft validity is controlled
  // by the zero logical lengths below.
  clear(current.buffers.gdn[0].convolutionBase, "slot convolution state");
  clear(current.buffers.gdn[0].recurrentBase, "slot recurrent state");
  current.metadata = {true, requestId, 0, {}};
  return true;
}

void QwenStateStorage::releaseSlot(uint32_t index, uint64_t requestId) {
  Slot &current = slot(index);
  requireAssigned(current);
  if (!requestId || current.metadata.requestId != requestId) {
    throw std::logic_error("Qwen state slot owner mismatch");
  }
  // Parity one first, so the next activation pops parity zero first and a
  // reactivated lane gets its previous buffers back in the same order.
  for (uint32_t parity = current.gdn.size(); parity > 0;) {
    --parity;
    if (current.gdn[parity])
      pool_->cells.push_back(std::move(current.gdn[parity]));
  }
  if (current.draft)
    pool_->rings.push_back(std::move(current.draft));
  refreshViews(current);
  current.metadata = {};
}

uint64_t QwenStateStorage::releaseIdle(uint32_t keepCells,
                                       uint32_t keepRings) noexcept {
  const uint64_t before = backend_.memoryStats().allocatedBytes;
  while (pool_->cells.size() > keepCells)
    pool_->cells.pop_back();
  while (pool_->rings.size() > keepRings)
    pool_->rings.pop_back();
  const uint64_t after = backend_.memoryStats().allocatedBytes;
  return before >= after ? before - after : 0;
}

uint32_t QwenStateStorage::idleCells() const noexcept {
  return static_cast<uint32_t>(pool_->cells.size());
}

uint32_t QwenStateStorage::idleRings() const noexcept {
  return static_cast<uint32_t>(pool_->rings.size());
}

uint64_t QwenStateStorage::activationBytes() const noexcept {
  const uint64_t cells = std::tuple_size_v<decltype(Slot::gdn)>;
  const uint64_t missing = cells - std::min<uint64_t>(pool_->cells.size(), cells);
  return missing * layout_.target.cellBytes() +
         (pool_->rings.empty() ? layout_.draft.ringBytes() : 0);
}

void QwenStateStorage::updateLengths(uint32_t index,
                                     QwenLogicalLengths lengths) {
  validateLengths(lengths, false);
  Slot &current = slot(index);
  requireAssigned(current);
  current.metadata.lengths = lengths;
}

void QwenStateStorage::swapParity(uint32_t index) {
  Slot &current = slot(index);
  requireAssigned(current);
  current.metadata.activeParity ^= 1;
}

std::shared_ptr<const QwenCompositeState>
QwenStateStorage::snapshot(uint32_t index) {
  return snapshot(index, slot(index).metadata.lengths);
}

std::shared_ptr<const QwenCompositeState>
QwenStateStorage::snapshot(uint32_t index, QwenLogicalLengths lengths) {
  Slot &source = slot(index);
  requireAssigned(source);
  validateLengths(lengths, true);
  // Pooled buffers first; a denied admission puts a pooled cell back and
  // drops a fresh one, leaving no trace.
  QwenCacheSlot cacheSlot;
  const bool pooledCell = !pool_->cells.empty();
  cacheSlot.gdn = acquireCell("qwen-state-cache-gdn");
  if (!cacheSlot.gdn)
    return nullptr;
  cacheSlot.draft = acquireRing("qwen-state-cache-draft");
  if (!cacheSlot.draft) {
    if (pooledCell)
      pool_->cells.push_back(std::move(cacheSlot.gdn));
    return nullptr;
  }
  const uint32_t active = source.metadata.activeParity;
  copyExact(cacheSlot.gdn->buffers().stateBase,
            source.gdn[active]->buffers().stateBase, "cached GDN state");
  for (uint32_t layer = 0; layer < source.buffers.draft.size(); ++layer) {
    copyExact(cacheSlot.draft->layers()[layer].keys,
              source.buffers.draft[layer].keys, "cached draft keys");
    copyExact(cacheSlot.draft->layers()[layer].values,
              source.buffers.draft[layer].values, "cached draft values");
  }
  return std::shared_ptr<const QwenCompositeState>(new QwenCompositeState(
      pool_, std::move(cacheSlot), layout_, lengths, file_, staging_));
}

std::unique_ptr<StateOffload>
QwenStateStorage::snapshotToDisk(uint32_t index, std::function<void()> completion) {
  Slot &source = slot(index);
  requireAssigned(source);
  validateLengths(source.metadata.lengths, true);
  if (!canSnapshotToDisk())
    return {};
  return QwenCompositeState::write(
      file_, staging_,
      stateSpans(source.buffers.gdn[source.metadata.activeParity], source.buffers.draft),
      layout_, source.metadata.lengths, std::move(completion));
}

std::shared_ptr<QwenGdnCell>
QwenStateStorage::acquireCell(std::string_view label) {
  if (pool_->cells.empty())
    return allocateGdnCell(label);
  std::shared_ptr<QwenGdnCell> cell = std::move(pool_->cells.back());
  pool_->cells.pop_back();
  return cell;
}

std::shared_ptr<DFlashDraftRing>
QwenStateStorage::acquireRing(std::string_view label) {
  if (pool_->rings.empty())
    return allocateDraftRing(label);
  std::shared_ptr<DFlashDraftRing> ring = std::move(pool_->rings.back());
  pool_->rings.pop_back();
  return ring;
}

void QwenStateStorage::restore(uint32_t index, const CompositeState &state,
                               bool restoreDraftState) {
  const auto *typed = dynamic_cast<const QwenCompositeState *>(&state);
  if (!typed) {
    throw std::invalid_argument("composite state is not Qwen state");
  }
  if (typed->layout_ != layout_) {
    throw std::invalid_argument("composite state layout does not match model");
  }
  validateLengths(typed->lengths_, true);
  Slot &destination = slot(index);
  requireAssigned(destination);
  if (!typed->slot_.gdn || !typed->slot_.draft) {
    throw std::invalid_argument("incompatible Qwen composite state");
  }

  const uint32_t active = destination.metadata.activeParity;
  copyExact(destination.gdn[active]->buffers().stateBase,
            typed->slot_.gdn->buffers().stateBase, "restored GDN state");
  if (restoreDraftState) {
    for (uint32_t layer = 0; layer < destination.buffers.draft.size();
         ++layer) {
      copyExact(destination.buffers.draft[layer].keys,
                typed->slot_.draft->layers()[layer].keys,
                "restored draft keys");
      copyExact(destination.buffers.draft[layer].values,
                typed->slot_.draft->layers()[layer].values,
                "restored draft values");
    }
  }
  restoreLengths(index, typed->lengths_, restoreDraftState);
}

void QwenStateStorage::restoreLengths(uint32_t index, QwenLogicalLengths lengths,
                                     bool restoreDraftState) {
  Slot &destination = slot(index);
  if (!restoreDraftState) {
    lengths.draftBase = lengths.targetTokens;
    lengths.draftLength = 0;
    lengths.draftCommitCursor = lengths.targetTokens % layout_.draft.tokens;
  }
  destination.metadata.lengths = lengths;
}

std::unique_ptr<StateRestore> QwenStateStorage::beginRestore(
    uint32_t index, const CompositeState &state, bool restoreDraftState,
    std::function<void()> completion, std::function<void()> committed) {
  const auto *typed = dynamic_cast<const QwenCompositeState *>(&state);
  if (!typed || typed->layout_ != layout_)
    throw std::invalid_argument("incompatible Qwen composite state");
  if (!typed->disk_) {
    restore(index, state, restoreDraftState);
    committed();
    return {};
  }
  validateLengths(typed->lengths_, true);
  Slot &destination = slot(index);
  requireAssigned(destination);
  auto spans = stateSpans(destination.buffers.gdn[destination.metadata.activeParity],
                          destination.buffers.draft);
  auto commit = [this, index, lengths = typed->lengths_, restoreDraftState,
                 committed = std::move(committed)] {
    restoreLengths(index, lengths, restoreDraftState);
    committed();
  };
  auto operation = typed->file_->read(typed->disk_, std::move(spans), std::move(completion));
  try {
    return std::make_unique<FileRestore>(operation, std::move(commit),
        [this, index, lengths = typed->lengths_] { return snapshot(index, lengths); });
  } catch (...) {
    operation->drain();
    throw;
  }
}

uint64_t QwenStateStorage::actualSlotBytes(uint32_t index) const {
  const Slot &current = slot(index);
  uint64_t result = 0;
  if (current.gdn[0])
    result += current.gdn[0]->actualAllocatedBytes();
  if (current.gdn[1])
    result += current.gdn[1]->actualAllocatedBytes();
  if (current.draft)
    result += current.draft->actualAllocatedBytes();
  return result;
}

metal::AllocationResult QwenStateStorage::allocateSlot(uint32_t index) {
  Slot &destination = slot(index);
  if (destination.gdn[0] || destination.gdn[1] || destination.draft) {
    throw std::logic_error("idle Qwen state slot still owns buffers");
  }
  // Pooled buffers first, then the governor for what the pool lacks. A denied
  // admission leaves no trace: pooled buffers go back, fresh ones are dropped.
  std::array<std::shared_ptr<QwenGdnCell>, 2> gdn;
  std::shared_ptr<DFlashDraftRing> draft;
  metal::AllocationFailure failure = metal::AllocationFailure::None;
  uint32_t pooledCells = 0;
  while (pooledCells < gdn.size() && !pool_->cells.empty()) {
    gdn[pooledCells++] = std::move(pool_->cells.back());
    pool_->cells.pop_back();
  }
  const bool pooledRing = !pool_->rings.empty();
  if (pooledRing) {
    draft = std::move(pool_->rings.back());
    pool_->rings.pop_back();
  }
  const auto giveBack = [&] {
    for (uint32_t parity = pooledCells; parity > 0;)
      pool_->cells.push_back(std::move(gdn[--parity]));
    if (pooledRing)
      pool_->rings.push_back(std::move(draft));
  };
  for (uint32_t parity = pooledCells; parity < gdn.size(); ++parity) {
    gdn[parity] = allocateGdnCell("qwen-state-cell-" + std::to_string(index) +
                                  "-gdn-" + std::to_string(parity), &failure);
    if (!gdn[parity]) {
      giveBack();
      return failure;
    }
  }
  if (!pooledRing) {
    draft = allocateDraftRing("qwen-state-cell-" + std::to_string(index) +
                              "-draft", &failure);
    if (!draft) {
      giveBack();
      return failure;
    }
  }
  destination.gdn = std::move(gdn);
  destination.draft = std::move(draft);
  refreshViews(destination);
  return true;
}

std::shared_ptr<QwenGdnCell>
QwenStateStorage::allocateGdnCell(std::string_view label,
                                     metal::AllocationFailure *failure) {
  std::shared_ptr<QwenGdnCell> result;
  const auto admission = admitAllocation_(layout_.target.cellBytes(), [&] {
        result = std::shared_ptr<QwenGdnCell>(
            new QwenGdnCell(backend_, allocations_, layout_.target, label));
      });
  if (!admission) {
    if (failure)
      *failure = admission.failure;
    return {};
  }
  if (!result)
    throw std::logic_error("state admission skipped GDN allocation");
  return result;
}

std::shared_ptr<DFlashDraftRing>
QwenStateStorage::allocateDraftRing(std::string_view label,
                                     metal::AllocationFailure *failure) {
  std::shared_ptr<DFlashDraftRing> result;
  const auto admission = admitAllocation_(layout_.draft.ringBytes(), [&] {
        result = std::shared_ptr<DFlashDraftRing>(
            new DFlashDraftRing(backend_, allocations_, layout_.draft,
                                label));
      });
  if (!admission) {
    if (failure)
      *failure = admission.failure;
    return {};
  }
  if (!result)
    throw std::logic_error("state admission skipped draft allocation");
  return result;
}

void QwenStateStorage::refreshViews(Slot &current) {
  for (uint32_t parity = 0; parity < current.gdn.size(); ++parity) {
    current.buffers.gdn[parity] = current.gdn[parity]
                                      ? current.gdn[parity]->buffers()
                                      : GdnParityBuffers{};
  }
  current.buffers.draft =
      current.draft ? current.draft->layers()
                    : std::vector<DFlashDraftRingLayer>{};
}

QwenStateStorage::Slot &QwenStateStorage::slot(uint32_t index) {
  if (index >= slots_.size()) {
    throw std::out_of_range("invalid Qwen state slot");
  }
  return slots_[index];
}

const QwenStateStorage::Slot &QwenStateStorage::slot(uint32_t index) const {
  if (index >= slots_.size()) {
    throw std::out_of_range("invalid Qwen state slot");
  }
  return slots_[index];
}

void QwenStateStorage::validateLengths(const QwenLogicalLengths &lengths,
                                       bool cacheSnapshot) const {
  if (lengths.draftLength > layout_.draft.tokens ||
      lengths.draftCommitCursor >= layout_.draft.tokens ||
      lengths.draftEnd() > lengths.targetTokens) {
    throw std::invalid_argument("invalid draft ring metadata");
  }
  if (cacheSnapshot &&
      (!lengths.targetTokens ||
       !lengths.hasCompleteDraftWindow(layout_.draft.tokens) ||
       lengths.targetTokens % kv::kPageTokens)) {
    throw std::invalid_argument(
        "composite snapshot requires equal page-aligned committed lengths");
  }
}

void QwenStateStorage::requireAssigned(const Slot &current) {
  if (!current.metadata.assigned || !current.metadata.requestId) {
    throw std::logic_error("Qwen state slot is not assigned");
  }
}

} // namespace splash::model
