#include "model/QwenState.hpp"

#include <algorithm>
#include <cstring>
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

// A grouped transfer drains every submitted operation, including exception
// paths. Different payload geometries have workers sharing the same budget.
struct FileOperations final {
  std::vector<std::shared_ptr<SlotFile::Operation>> items;
  ~FileOperations() {
    for (auto &item : items)
      item->drain();
  }
  bool ready() const noexcept {
    return std::all_of(items.begin(), items.end(),
                       [](const auto &item) { return item->ready(); });
  }
  bool finish() {
    bool success = true;
    for (auto &item : items)
      success = item->wait() && success;
    return success;
  }
  void cancel() noexcept {
    for (auto &item : items)
      item->cancel();
  }
};

class FileOffload final : public StateOffload {
public:
  FileOffload(std::shared_ptr<FileOperations> operation,
              std::shared_ptr<const CompositeState> state,
              std::shared_ptr<StateStaging> staging)
      : operation_(std::move(operation)), state_(std::move(state)),
        staging_(std::move(staging)) {}
  // The staging copy is the write's source until the worker has stopped.
  ~FileOffload() override {
    operation_.reset();
    staging_->busy = false;
  }
  bool ready() const noexcept override { return operation_->ready(); }
  bool finish() override { return operation_->finish(); }
  const std::shared_ptr<const CompositeState> &state() const noexcept override {
    return state_;
  }

private:
  std::shared_ptr<FileOperations> operation_;
  std::shared_ptr<const CompositeState> state_;
  std::shared_ptr<StateStaging> staging_;
};

class FileRestore final : public StateRestore {
public:
  FileRestore(std::shared_ptr<FileOperations> operation,
              std::function<void()> committed,
              std::function<std::shared_ptr<const CompositeState>()> snapshot)
      : operation_(std::move(operation)), committed_(std::move(committed)),
        snapshot_(std::move(snapshot)) {}
  ~FileRestore() override { operation_.reset(); }
  bool ready() const noexcept override { return operation_->ready(); }
  void cancel() noexcept override { operation_->cancel(); }
  bool finish() override {
    if (!operation_->finish())
      return false;
    committed_();
    finished_ = true;
    return true;
  }
  std::shared_ptr<const CompositeState> snapshot() override {
    return finished_ ? snapshot_() : nullptr;
  }

private:
  std::shared_ptr<FileOperations> operation_;
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
  buffers_.convolutionBase =
      backend.view(buffers_.stateBase, 0, layout.convolutionBytes());
  buffers_.convolutionLayers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    buffers_.convolutionLayers[layer] =
        backend.view(buffers_.convolutionBase,
                     uint64_t{layer} * layout.convolutionLayerBytes(),
                     layout.convolutionLayerBytes());
  }
  buffers_.recurrentBase = backend.view(
      buffers_.stateBase, layout.convolutionBytes(), layout.recurrentBytes());
  buffers_.recurrentLayers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    buffers_.recurrentLayers[layer] = backend.view(
        buffers_.recurrentBase, uint64_t{layer} * layout.recurrentLayerBytes(),
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
                                       std::shared_ptr<SlotFile> draftFile,
                                       std::shared_ptr<StateStaging> staging)
    : pool_(std::move(pool)), slot_(std::move(slot)), layout_(layout),
      lengths_(lengths), file_(std::move(file)), staging_(std::move(staging)),
      draftFile_(std::move(draftFile)) {
  if (!pool_ || !slot_.gdn || slot_.draft.empty()) {
    throw std::invalid_argument("composite state buffers are empty");
  }
}

QwenCompositeState::QwenCompositeState(
    CompositeStateLayout layout, QwenLogicalLengths lengths,
    std::shared_ptr<SlotFile> file, std::shared_ptr<SlotFile::Slot> disk,
    std::shared_ptr<SlotFile> draftFile,
    std::vector<std::shared_ptr<SlotFile::Slot>> draftDisk)
    : layout_(layout), lengths_(lengths), file_(std::move(file)),
      disk_(std::move(disk)), draftFile_(std::move(draftFile)),
      draftDisk_(std::move(draftDisk)) {}

DiskStateRecord QwenCompositeState::diskRecord() const {
  DiskStateRecord result{disk_,
                         {lengths_.targetTokens, lengths_.draftBase,
                          lengths_.draftLength, lengths_.draftCommitCursor},
                         draftDisk_};
  if (!disk_)
    for (const auto &block : slot_.draft)
      result.components.push_back(block->disk.lock());
  return result;
}

uint64_t QwenCompositeState::reclaimableBytes() const noexcept {
  if (disk_)
    return 0;
  uint64_t bytes = layout_.target.cellBytes();
  for (const auto &block : slot_.draft)
    if (block.use_count() == 1)
      bytes += layout_.draft.blockBytes();
  return bytes;
}

uint64_t QwenCompositeState::offloadBytes() const noexcept {
  if (disk_)
    return 0;
  uint64_t bytes = layout_.target.cellBytes();
  for (const auto &block : slot_.draft)
    if (block->disk.expired())
      bytes += layout_.draft.blockBytes();
  return bytes;
}

std::vector<StateResource> QwenCompositeState::resources() const {
  std::vector<StateResource> result;
  if (disk_) {
    result.push_back({disk_.get(), disk_->bytes()});
    for (const auto &block : draftDisk_)
      result.push_back({block.get(), block->bytes()});
  } else {
    result.push_back({slot_.gdn.get(), layout_.target.cellBytes()});
    for (const auto &block : slot_.draft)
      result.push_back({block.get(), layout_.draft.blockBytes()});
  }
  return result;
}

std::shared_ptr<SlotFile::Slot>
QwenStateStorage::reopenStateComponent(CacheStore::Record record) {
  if (!draftFile_ || record.bytes != layout_.draft.blockBytes())
    throw std::invalid_argument("invalid persistent draft block size");
  return draftFile_->reopen(record.id);
}

std::shared_ptr<const CompositeState>
QwenStateStorage::reopenState(DiskStateRecord record, uint64_t boundary) {
  if (!file_ || !record.slot || record.metadata.size() != 4 ||
      record.metadata[0] != boundary || record.metadata[1] > boundary ||
      record.metadata[2] > boundary - record.metadata[1] ||
      record.metadata[2] > UINT32_MAX || record.metadata[3] > UINT32_MAX)
    throw std::invalid_argument("invalid persistent state metadata");
  QwenLogicalLengths lengths{record.metadata[0], record.metadata[1],
                             static_cast<uint32_t>(record.metadata[2]),
                             static_cast<uint32_t>(record.metadata[3])};
  validateLengths(lengths, true);
  if (record.slot->bytes() != layout_.target.cellBytes() ||
      record.components.size() !=
          (lengths.draftLength + DraftStateLayout::blockTokens - 1) /
              DraftStateLayout::blockTokens)
    throw std::invalid_argument("incomplete persistent draft window");
  for (const auto &block : record.components)
    if (!block || block->bytes() != layout_.draft.blockBytes())
      throw std::invalid_argument("invalid persistent draft block");
  return std::shared_ptr<const CompositeState>(
      new QwenCompositeState(layout_, lengths, file_, std::move(record.slot),
                             draftFile_, std::move(record.components)));
}

std::unique_ptr<StateOffload>
QwenCompositeState::offload(std::function<void()> completion) const {
  if (!canOffload())
    return {};
  return write(file_, draftFile_, staging_, slot_.gdn->buffers(), slot_.draft,
               layout_, lengths_, std::move(completion));
}

std::unique_ptr<StateOffload> QwenCompositeState::write(
    const std::shared_ptr<SlotFile> &file,
    const std::shared_ptr<SlotFile> &draftFile,
    const std::shared_ptr<StateStaging> &staging, const GdnParityBuffers &gdn,
    const DraftKvCache::Window &blocks, CompositeStateLayout layout,
    QwenLogicalLengths lengths, std::function<void()> completion) {
  if (staging->busy)
    throw std::logic_error("a composite state write is already in flight");
  auto disk = file->acquire();
  if (!disk)
    return {};
  const size_t count =
      (lengths.draftLength + DraftStateLayout::blockTokens - 1) /
      DraftStateLayout::blockTokens;
  std::vector<std::shared_ptr<SlotFile::Slot>> draftDisk;
  std::vector<bool> writeBlock;
  draftDisk.reserve(count);
  writeBlock.reserve(count);
  // Reserve the whole transfer before submitting IO. Failure leaves no
  // partially admitted state and returns every newly reserved slot.
  for (size_t i = 0; i < count; ++i) {
    auto slot = blocks[i]->disk.lock();
    writeBlock.push_back(!slot);
    if (!slot)
      slot = draftFile->acquire();
    if (!slot)
      return {};
    draftDisk.push_back(std::move(slot));
  }
  auto result =
      std::shared_ptr<const QwenCompositeState>(new QwenCompositeState(
          layout, lengths, file, disk, draftFile, draftDisk));
  auto operations = std::make_shared<FileOperations>();
  operations->items.reserve(count + 1);
  std::memcpy(staging->bytes.get(), gdn.stateBase.contents(),
              layout.target.cellBytes());
  auto *draft = staging->bytes.get() + layout.target.cellBytes();
  for (size_t i = 0; i < count; ++i) {
    if (!writeBlock[i])
      continue;
    auto bytes = std::span(draft + i * layout.draft.blockBytes(),
                           layout.draft.blockBytes());
    std::memcpy(bytes.data(), blocks[i]->page->buffer.contents(), bytes.size());
  }
  staging->busy = true;
  try {
    operations->items.push_back(file->write(
        disk, {{staging->bytes.get(), layout.target.cellBytes()}}, completion));
    for (size_t i = 0; i < count; ++i) {
      if (writeBlock[i])
        operations->items.push_back(
            draftFile->write(draftDisk[i],
                             {{draft + i * layout.draft.blockBytes(),
                               layout.draft.blockBytes()}},
                             completion));
    }
    auto ticket =
        std::make_unique<FileOffload>(operations, std::move(result), staging);
    for (size_t i = 0; i < count; ++i)
      blocks[i]->disk = draftDisk[i];
    return ticket;
  } catch (...) {
    operations.reset();
    staging->busy = false;
    throw;
  }
}

QwenCompositeState::~QwenCompositeState() {
  if (!pool_ || !pool_->open)
    return;
  pool_->cells.push_back(std::move(slot_.gdn));
}

QwenStateStorage::QwenStateStorage(metal::MetalBackend &backend,
                                   metal::AllocationAdmission admitAllocation,
                                   CompositeStateLayout layout,
                                   std::shared_ptr<SlotFile> file)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout), allocations_(std::make_shared<StateAllocationTracker>()),
      draftCache_(backend, admitAllocation_, allocations_, layout.draft),
      pool_(std::make_shared<QwenBufferPool>()) {
  if (!admitAllocation_)
    throw std::invalid_argument("Qwen state allocation admission is required");
  if (!layout_.valid() ||
      layout_.draft.tokens != ExecutionLimits::draftContextTokens) {
    throw std::invalid_argument("Qwen composite state layout is invalid");
  }
  if (file && file->slotBytes() != layout_.target.cellBytes())
    throw std::invalid_argument("state file slots do not hold one state");
  if (file) {
    file_ = std::move(file);
    draftFile_ = file_->sibling(layout_.draft.blockBytes());
    staging_ = std::make_shared<StateStaging>();
    void *memory = nullptr;
    if (::posix_memalign(&memory, SlotFile::kAlignmentBytes,
                         layout_.cachedBytes()) != 0)
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
}

const QwenSlotBuffers &QwenStateStorage::buffers(uint32_t index) const {
  return slot(index).buffers;
}

const QwenSlotMetadata &QwenStateStorage::metadata(uint32_t index) const {
  return slot(index).metadata;
}

metal::AllocationResult QwenStateStorage::tryActivateSlot(uint32_t index,
                                                          uint64_t requestId) {
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
  current.draft.reset();
  current.buffers = {};
  current.metadata = {};
}

uint64_t QwenStateStorage::releaseIdle(uint32_t keepCells,
                                       uint32_t keepRings) noexcept {
  const uint64_t before = backend_.memoryStats().allocatedBytes;
  while (pool_->cells.size() > keepCells)
    pool_->cells.pop_back();
  draftCache_.releaseIdle(
      keepRings * (layout_.draft.tokens / DraftStateLayout::blockTokens));
  const uint64_t after = backend_.memoryStats().allocatedBytes;
  return before >= after ? before - after : 0;
}

uint64_t QwenStateStorage::reclaimIdle() noexcept {
  const uint64_t before = backend_.memoryStats().allocatedBytes;
  if (!pool_->cells.empty())
    pool_->cells.pop_back();
  else if (draftCache_.idlePages())
    draftCache_.releaseIdle(draftCache_.idlePages() - 1);
  const uint64_t after = backend_.memoryStats().allocatedBytes;
  return before >= after ? before - after : 0;
}

uint32_t QwenStateStorage::idleCells() const noexcept {
  return static_cast<uint32_t>(pool_->cells.size());
}

uint32_t QwenStateStorage::idleRings() const noexcept {
  return draftCache_.idlePages() /
         (layout_.draft.tokens / DraftStateLayout::blockTokens);
}

uint64_t QwenStateStorage::activationBytes() const noexcept {
  const uint64_t cells = std::tuple_size_v<decltype(Slot::gdn)>;
  const uint64_t missing =
      cells - std::min<uint64_t>(pool_->cells.size(), cells);
  return missing * layout_.target.cellBytes() +
         (layout_.draft.tokens / DraftStateLayout::blockTokens -
          std::min<uint32_t>(draftCache_.idlePages(),
                             layout_.draft.tokens /
                                 DraftStateLayout::blockTokens)) *
             layout_.draft.blockBytes();
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
  cacheSlot.gdn = acquireCell("qwen-state-cache-gdn");
  if (!cacheSlot.gdn)
    return nullptr;
  const size_t count =
      (lengths.draftLength + DraftStateLayout::blockTokens - 1) /
      DraftStateLayout::blockTokens;
  cacheSlot.draft.assign(source.draft->pages.begin(),
                         source.draft->pages.begin() + count);
  const uint32_t active = source.metadata.activeParity;
  copyExact(cacheSlot.gdn->buffers().stateBase,
            source.gdn[active]->buffers().stateBase, "cached GDN state");
  return std::shared_ptr<const QwenCompositeState>(
      new QwenCompositeState(pool_, std::move(cacheSlot), layout_, lengths,
                             file_, draftFile_, staging_));
}

std::unique_ptr<StateOffload>
QwenStateStorage::snapshotToDisk(uint32_t index,
                                 std::function<void()> completion) {
  Slot &source = slot(index);
  requireAssigned(source);
  validateLengths(source.metadata.lengths, true);
  if (!canSnapshotToDisk())
    return {};
  return QwenCompositeState::write(
      file_, draftFile_, staging_,
      source.buffers.gdn[source.metadata.activeParity], source.draft->pages,
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
  if (!typed->slot_.gdn || typed->slot_.draft.empty()) {
    throw std::invalid_argument("incompatible Qwen composite state");
  }

  DraftKvCache::Window pages;
  std::vector<DFlashDraftRingLayer> views;
  if (restoreDraftState) {
    pages = destination.draft->pages;
    std::copy(typed->slot_.draft.begin(), typed->slot_.draft.end(),
              pages.begin());
    views = draftCache_.views(pages);
  }
  const uint32_t active = destination.metadata.activeParity;
  copyExact(destination.gdn[active]->buffers().stateBase,
            typed->slot_.gdn->buffers().stateBase, "restored GDN state");
  if (restoreDraftState) {
    destination.draft->pages = std::move(pages);
    destination.buffers.draft = std::move(views);
  }
  restoreLengths(index, typed->lengths_, restoreDraftState);
}

void QwenStateStorage::restoreLengths(uint32_t index,
                                      QwenLogicalLengths lengths,
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
  // Read only the components execution needs. A skipped draft window cannot
  // be promoted as a complete RAM state; the durable copy remains reusable.
  if (restoreDraftState) {
    if (auto admitted = prepareDraftWrite(index, 0, layout_.draft.tokens);
        !admitted)
      throw metal::MetalAllocationError("cannot admit draft restore pages",
                                        admitted.failure);
  }
  auto operations = std::make_shared<FileOperations>();
  operations->items.reserve(typed->draftDisk_.size() + 1);
  auto commit = [this, index, lengths = typed->lengths_, restoreDraftState,
                 blocks = typed->draftDisk_, committed = std::move(committed)] {
    if (restoreDraftState)
      for (size_t i = 0; i < blocks.size(); ++i)
        slot(index).draft->pages[i]->disk = blocks[i];
    restoreLengths(index, lengths, restoreDraftState);
    committed();
  };
  auto snapshot = [this, index, lengths = typed->lengths_, restoreDraftState] {
    return restoreDraftState ? this->snapshot(index, lengths) : nullptr;
  };
  auto ticket = std::make_unique<FileRestore>(operations, std::move(commit),
                                              std::move(snapshot));
  const auto &gdn =
      destination.buffers.gdn[destination.metadata.activeParity].stateBase;
  operations->items.push_back(typed->file_->read(
      typed->disk_,
      {{static_cast<std::byte *>(gdn.contents()), gdn.sizeBytes()}},
      completion));
  if (restoreDraftState)
    for (size_t i = 0; i < typed->draftDisk_.size(); ++i) {
      const auto &buffer = destination.draft->pages[i]->page->buffer;
      operations->items.push_back(typed->draftFile_->read(
          typed->draftDisk_[i],
          {{static_cast<std::byte *>(buffer.contents()), buffer.sizeBytes()}},
          completion));
    }
  return ticket;
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
  DraftKvCache::AllocationScope allocation(draftCache_);
  std::array<std::shared_ptr<QwenGdnCell>, 2> gdn;
  std::shared_ptr<DFlashDraftRing> draft;
  metal::AllocationFailure failure = metal::AllocationFailure::None;
  uint32_t pooledCells = 0;
  while (pooledCells < gdn.size() && !pool_->cells.empty()) {
    gdn[pooledCells++] = std::move(pool_->cells.back());
    pool_->cells.pop_back();
  }
  const auto giveBack = [&] {
    for (uint32_t parity = pooledCells; parity > 0;)
      pool_->cells.push_back(std::move(gdn[--parity]));
  };
  for (uint32_t parity = pooledCells; parity < gdn.size(); ++parity) {
    gdn[parity] = allocateGdnCell("qwen-state-cell-" + std::to_string(index) +
                                      "-gdn-" + std::to_string(parity),
                                  &failure);
    if (!gdn[parity]) {
      giveBack();
      return failure;
    }
  }
  {
    draft = allocateDraftRing(
        "qwen-state-cell-" + std::to_string(index) + "-draft", &failure);
    if (!draft) {
      giveBack();
      return failure;
    }
  }
  QwenSlotBuffers buffers;
  for (uint32_t parity = 0; parity < gdn.size(); ++parity)
    buffers.gdn[parity] = gdn[parity]->buffers();
  buffers.draft = draftCache_.views(draft->pages);
  destination.gdn = std::move(gdn);
  destination.draft = std::move(draft);
  destination.buffers = std::move(buffers);
  allocation.commit();
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
  static_cast<void>(label);
  auto result = std::make_shared<DFlashDraftRing>();
  result->pages.reserve(layout_.draft.tokens / DraftStateLayout::blockTokens);
  for (uint32_t position = 0; position < layout_.draft.tokens;
       position += DraftStateLayout::blockTokens) {
    auto block = draftCache_.acquire(failure);
    if (!block)
      return {};
    std::memset(block->page->buffer.contents(), 0, layout_.draft.blockBytes());
    result->pages.push_back(std::move(block));
  }
  return result;
}

metal::AllocationResult QwenStateStorage::prepareDraftWrite(uint32_t index,
                                                            uint64_t begin,
                                                            uint64_t end) {
  Slot &current = slot(index);
  requireAssigned(current);
  if (begin > end)
    throw std::invalid_argument("reversed draft write range");
  if (begin == end)
    return true;
  const uint64_t first = begin / DraftStateLayout::blockTokens;
  const uint64_t last =
      (end + DraftStateLayout::blockTokens - 1) / DraftStateLayout::blockTokens;
  const uint64_t count =
      std::min<uint64_t>(last - first, current.draft->pages.size());
  DraftKvCache::AllocationScope allocation(draftCache_);
  std::array<std::shared_ptr<DraftKvCache::Block>, SPLASH_DRAFT_PAGE_COUNT>
      replacements{};
  bool changed = false;
  for (uint64_t offset = 0; offset < count; ++offset) {
    const auto &page =
        current.draft->pages[(first + offset) % current.draft->pages.size()];
    if (page.use_count() <= 1)
      continue;
    metal::AllocationFailure failure = metal::AllocationFailure::None;
    auto replacement = draftCache_.acquire(&failure);
    if (!replacement)
      return failure;
    std::memcpy(replacement->page->buffer.contents(),
                page->page->buffer.contents(), layout_.draft.blockBytes());
    replacements[offset] = std::move(replacement);
    changed = true;
  }
  if (changed) {
    auto pages = current.draft->pages;
    for (uint64_t offset = 0; offset < count; ++offset)
      if (replacements[offset])
        pages[(first + offset) % pages.size()] =
            std::move(replacements[offset]);
    // Build every view before replacing either table. Even a host allocation
    // failure leaves execution and snapshots pointing at the original pages.
    auto views = draftCache_.views(pages);
    current.draft->pages = std::move(pages);
    current.buffers.draft = std::move(views);
  }
  for (uint64_t offset = 0; offset < count; ++offset)
    current.draft->pages[(first + offset) % current.draft->pages.size()]
        ->disk.reset();
  allocation.commit();
  return true;
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
  if (cacheSnapshot && (!lengths.targetTokens ||
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
