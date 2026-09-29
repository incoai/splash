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

// Each payload belongs to one cache group block. Sharing a page never retains
// a recurrent checkpoint, and a checkpoint never owns a draft window.
class QwenStatePayload final : public StatePayload {
public:
  CacheGroupId group = 0;
  uint64_t size = 0;
  std::shared_ptr<QwenGdnCell> cell;
  std::shared_ptr<QwenBufferPool> pool;
  std::shared_ptr<DraftKvCache::Block> page;
  std::shared_ptr<SlotFile> file;
  std::shared_ptr<SlotFile::Slot> disk;
  std::shared_ptr<StateStaging> staging;

  ~QwenStatePayload() override {
    if (cell && pool && pool->open)
      pool->cells.push_back(std::move(cell));
  }
  uint64_t bytes() const noexcept override { return size; }
  uint64_t residentBytes() const noexcept override { return disk ? 0 : size; }
  uint64_t reclaimableBytes() const noexcept override {
    return disk || (page && page.use_count() != 1) ? 0 : size;
  }
  uint64_t offloadBytes() const noexcept override {
    return disk || (page && !page->disk.expired()) ? 0 : size;
  }
  bool durable() const noexcept override { return disk && disk->durable(); }
  bool canOffload() const noexcept override {
    return !disk && file && file->writable();
  }
  DiskStateRecord diskRecord() const override {
    return {disk ? disk : page ? page->disk.lock() : nullptr, {}, {}};
  }
  std::vector<StateResource> resources() const override {
    return {{disk   ? static_cast<const void *>(disk.get())
             : page ? static_cast<const void *>(page.get())
                    : cell.get(),
             size}};
  }
  const MetalBuffer &buffer() const {
    return page ? page->page->buffer : cell->buffers().stateBase;
  }
  std::unique_ptr<StateOffload>
  offload(std::function<void()> completion) const override;
};

struct PayloadWrite final {
  std::shared_ptr<StateStaging> staging;
  std::shared_ptr<FileOperations> operations =
      std::make_shared<FileOperations>();
  std::vector<std::shared_ptr<const StatePayload>> payloads;
  bool ownsStaging = false;
  ~PayloadWrite() {
    operations.reset();
    if (ownsStaging)
      staging->busy = false;
  }
};

std::shared_ptr<PayloadWrite>
writePayloads(std::span<const QwenStatePayload *const> sources,
              const std::shared_ptr<StateStaging> &staging,
              const std::function<void()> &completion) {
  if (!staging)
    return {};
  if (staging->busy)
    throw std::logic_error("state staging write is already in flight");
  auto result = std::make_shared<PayloadWrite>();
  result->staging = staging;
  result->payloads.reserve(sources.size());
  result->operations->items.reserve(sources.size());
  std::vector<uint64_t> offsets;
  offsets.reserve(sources.size());
  uint64_t bytes = 0;
  for (const auto *source : sources) {
    if (!source->canOffload())
      return {};
    auto disk = source->page ? source->page->disk.lock() : nullptr;
    const bool missing = !disk;
    if (!disk)
      disk = source->file->acquire();
    if (!disk)
      return {};
    auto payload = std::make_shared<QwenStatePayload>();
    payload->group = source->group;
    payload->size = source->size;
    payload->file = source->file;
    payload->disk = std::move(disk);
    result->payloads.push_back(std::move(payload));
    offsets.push_back(missing ? bytes : UINT64_MAX);
    if (missing) {
      if (source->size > staging->size - bytes)
        throw std::logic_error("state snapshot exceeds bounded staging");
      std::memcpy(staging->bytes.get() + bytes, source->buffer().contents(),
                  source->size);
      bytes += source->size;
    }
  }
  staging->busy = true;
  result->ownsStaging = true;
  for (size_t i = 0; i < sources.size(); ++i) {
    const auto *source = sources[i];
    const auto record = result->payloads[i]->diskRecord();
    if (offsets[i] != UINT64_MAX)
      result->operations->items.push_back(source->file->write(
          record.slot, {{staging->bytes.get() + offsets[i], source->size}},
          completion));
  }
  for (size_t i = 0; i < sources.size(); ++i)
    if (sources[i]->page)
      sources[i]->page->disk = result->payloads[i]->diskRecord().slot;
  return result;
}

class PayloadOffload final : public StateOffload {
public:
  explicit PayloadOffload(std::shared_ptr<PayloadWrite> write)
      : write_(std::move(write)) {}
  bool ready() const noexcept override { return write_->operations->ready(); }
  bool finish() override { return write_->operations->finish(); }
  const std::shared_ptr<const StatePayload> &state() const noexcept override {
    return write_->payloads.front();
  }

private:
  std::shared_ptr<PayloadWrite> write_;
};

std::unique_ptr<StateOffload>
QwenStatePayload::offload(std::function<void()> completion) const {
  const QwenStatePayload *source = this;
  auto write = writePayloads(std::span(&source, 1), staging, completion);
  return write ? std::make_unique<PayloadOffload>(std::move(write)) : nullptr;
}

class FileSnapshotOffload final : public SnapshotOffload {
public:
  FileSnapshotOffload(std::shared_ptr<PayloadWrite> write,
                      std::shared_ptr<const RestoreState> state)
      : write_(std::move(write)), state_(std::move(state)) {}
  bool ready() const noexcept override { return write_->operations->ready(); }
  bool finish() override { return write_->operations->finish(); }
  const std::shared_ptr<const RestoreState> &state() const noexcept override {
    return state_;
  }

private:
  std::shared_ptr<PayloadWrite> write_;
  std::shared_ptr<const RestoreState> state_;
};

class FileRestore final : public StateRestore {
public:
  FileRestore(std::shared_ptr<FileOperations> operation,
              std::function<void()> committed,
              std::function<std::shared_ptr<const RestoreState>()> snapshot)
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
  std::shared_ptr<const RestoreState> snapshot() override {
    return finished_ ? snapshot_() : nullptr;
  }

private:
  std::shared_ptr<FileOperations> operation_;
  std::function<void()> committed_;
  std::function<std::shared_ptr<const RestoreState>()> snapshot_;
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

std::shared_ptr<const StatePayload>
QwenStateStorage::reopenState(const CachedStateBlock &block,
                              StoredStateRecord record) {
  const auto group = block.group;
  if (group == kDraftWindowGroup &&
      (block.begin >= block.end ||
       block.begin / DraftStateLayout::blockTokens !=
           (block.end - 1) / DraftStateLayout::blockTokens))
    throw std::invalid_argument(
        "persistent draft fragment crosses a physical page");
  const auto file = group == kQwenRecurrentGroup ? file_
                    : group == kDraftWindowGroup ? draftFile_
                                                 : nullptr;
  if (!file || record.records.size() != 1 ||
      record.records.front().bytes != file->slotBytes() ||
      !record.metadata.empty())
    throw std::invalid_argument("invalid persistent cache group payload");
  auto payload = std::make_shared<QwenStatePayload>();
  payload->group = group;
  payload->size = file->slotBytes();
  payload->file = file;
  payload->disk = file->reopen(record.records.front().id);
  if (!payload->disk)
    throw std::invalid_argument("cannot reopen cache group record");
  return payload;
}

std::shared_ptr<const RestoreState> QwenStateStorage::snapshot(uint32_t index) {
  return snapshot(index, slot(index).metadata.lengths);
}

std::shared_ptr<const RestoreState>
QwenStateStorage::snapshot(uint32_t index, QwenLogicalLengths lengths,
                           bool copyGdn) {
  Slot &source = slot(index);
  requireAssigned(source);
  validateLengths(lengths, true);
  auto result = std::make_shared<RestoreState>();
  result->boundary = lengths.targetTokens;
  result->blocks.reserve(2 +
                         layout_.draft.tokens / DraftStateLayout::blockTokens);
  auto recurrent = std::make_shared<QwenStatePayload>();
  recurrent->group = kQwenRecurrentGroup;
  recurrent->size = layout_.target.cellBytes();
  recurrent->file = file_;
  recurrent->staging = staging_;
  if (copyGdn) {
    recurrent->cell = acquireCell("qwen-state-cache-gdn");
    if (!recurrent->cell)
      return {};
    recurrent->pool = pool_;
    copyExact(recurrent->cell->buffers().stateBase,
              source.gdn[source.metadata.activeParity]->buffers().stateBase,
              "cached recurrent state");
  } else {
    recurrent->cell = source.gdn[source.metadata.activeParity];
  }
  result->blocks.push_back({kQwenRecurrentGroup, result->boundary,
                            result->boundary, std::move(recurrent)});
  const uint32_t pageTokens = DraftStateLayout::blockTokens;
  const uint32_t pages = layout_.draft.tokens / pageTokens;
  std::vector<std::shared_ptr<const StatePayload>> payloads(pages);
  for (uint32_t begin = lengths.draftBase; begin < result->boundary;) {
    const uint32_t end = std::min<uint64_t>(
        result->boundary, (uint64_t{begin} / pageTokens + 1) * pageTokens);
    const uint32_t physical = (begin / pageTokens) % pages;
    auto &payload = payloads[physical];
    if (!payload) {
      auto draft = std::make_shared<QwenStatePayload>();
      draft->group = kDraftWindowGroup;
      draft->size = layout_.draft.blockBytes();
      draft->file = draftFile_;
      draft->staging = staging_;
      draft->page = source.draft->pages[physical];
      payload = std::move(draft);
    }
    result->blocks.push_back({kDraftWindowGroup, begin, end, payload});
    begin = end;
  }
  return result;
}

std::unique_ptr<SnapshotOffload>
QwenStateStorage::snapshotToDisk(uint32_t index,
                                 std::function<void()> completion) {
  if (!canSnapshotToDisk())
    return {};
  auto source = snapshot(index, slot(index).metadata.lengths, false);
  std::vector<const QwenStatePayload *> unique;
  std::vector<size_t> indices;
  unique.reserve(source->blocks.size());
  indices.reserve(source->blocks.size());
  for (const auto &block : source->blocks) {
    const auto *payload =
        static_cast<const QwenStatePayload *>(block.payload.get());
    const auto found = std::find(unique.begin(), unique.end(), payload);
    indices.push_back(found - unique.begin());
    if (found == unique.end())
      unique.push_back(payload);
  }
  auto write = writePayloads(unique, staging_, completion);
  if (!write)
    return {};
  auto state = std::make_shared<RestoreState>(*source);
  for (size_t i = 0; i < state->blocks.size(); ++i)
    state->blocks[i].payload = write->payloads[indices[i]];
  return std::make_unique<FileSnapshotOffload>(std::move(write),
                                               std::move(state));
}

std::shared_ptr<QwenGdnCell>
QwenStateStorage::acquireCell(std::string_view label) {
  if (pool_->cells.empty())
    return allocateGdnCell(label);
  std::shared_ptr<QwenGdnCell> cell = std::move(pool_->cells.back());
  pool_->cells.pop_back();
  return cell;
}

namespace {
struct DraftFragment final {
  uint32_t begin, end;
  std::shared_ptr<DraftKvCache::Block> page;
};

void copyDraftSlice(const MetalBuffer &destination, const MetalBuffer &source,
                    DraftStateLayout layout, uint32_t begin, uint32_t end) {
  if (destination.contents() == source.contents())
    return;
  const uint32_t first = begin % layout.blockTokens;
  const uint32_t count = end - begin;
  if (!count || count > layout.blockTokens - first)
    throw std::logic_error("draft fragment crosses a page");
  auto *dst = static_cast<std::byte *>(destination.contents());
  const auto *src = static_cast<const std::byte *>(source.contents());
  const uint64_t tensor =
      uint64_t{layout.kvHeads} * layout.blockTokens * layout.headDimension * 2;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    for (uint32_t head = 0; head < layout.kvHeads; ++head) {
      const uint64_t offset =
          layer * 2 * tensor + (uint64_t{head} * layout.blockTokens + first) *
                                   layout.headDimension * 2;
      std::memcpy(dst + offset, src + offset,
                  uint64_t{count} * layout.headDimension * 2);
    }
    for (uint32_t row = 0; row < layout.kvHeads * layout.headDimension; ++row) {
      const uint64_t offset = (layer * 2 + 1) * tensor +
                              (uint64_t{row} * layout.blockTokens + first) * 2;
      std::memcpy(dst + offset, src + offset, uint64_t{count} * 2);
    }
  }
}
} // namespace

void QwenStateStorage::restore(uint32_t index, const RestoreState &state,
                               bool restoreDraftState) {
  if (!state.resident())
    throw std::invalid_argument("synchronous restore needs resident state");
  auto transfer = beginRestore(index, state, restoreDraftState, {}, [] {});
  if (transfer && !transfer->finish())
    throw std::runtime_error("resident state restore failed");
}

void QwenStateStorage::restoreLengths(uint32_t index,
                                      QwenLogicalLengths lengths,
                                      bool restoreDraftState) {
  if (!restoreDraftState) {
    lengths.draftBase = lengths.targetTokens;
    lengths.draftLength = 0;
    lengths.draftCommitCursor = lengths.targetTokens % layout_.draft.tokens;
  }
  slot(index).metadata.lengths = lengths;
}

std::unique_ptr<StateRestore> QwenStateStorage::beginRestore(
    uint32_t index, const RestoreState &state, bool restoreDraftState,
    std::function<void()> completion, std::function<void()> committed) {
  Slot &destination = slot(index);
  requireAssigned(destination);
  const uint32_t boundary = state.boundary;
  const uint32_t begin = boundary - std::min(boundary, layout_.draft.tokens);
  QwenLogicalLengths lengths{boundary, begin, boundary - begin,
                             boundary % layout_.draft.tokens};
  validateLengths(lengths, true);
  std::shared_ptr<const QwenStatePayload> recurrent;
  std::vector<CachedStateBlock> draft;
  for (const auto &part : state.blocks) {
    auto payload =
        std::dynamic_pointer_cast<const QwenStatePayload>(part.payload);
    if (!payload || payload->group != part.group)
      throw std::invalid_argument("incompatible model cache group payload");
    if (part.group == kQwenRecurrentGroup) {
      if (recurrent || part.begin != boundary || part.end != boundary ||
          payload->size != layout_.target.cellBytes())
        throw std::invalid_argument("invalid recurrent checkpoint");
      recurrent = std::move(payload);
    } else if (part.group == kDraftWindowGroup) {
      if (part.begin >= part.end || part.end > boundary ||
          part.begin / DraftStateLayout::blockTokens !=
              (part.end - 1) / DraftStateLayout::blockTokens ||
          payload->size != layout_.draft.blockBytes())
        throw std::invalid_argument("invalid draft cache fragment");
      draft.push_back(part);
    } else {
      throw std::invalid_argument("unknown model cache group");
    }
  }
  if (!recurrent)
    throw std::invalid_argument("missing recurrent checkpoint");
  std::sort(draft.begin(), draft.end(),
            [](const auto &a, const auto &b) { return a.begin < b.begin; });
  uint32_t covered = begin;
  for (const auto &part : draft) {
    if (part.begin > covered || part.end <= covered)
      throw std::invalid_argument("draft window has a hole or overlap");
    covered = part.end;
  }
  if (covered != boundary)
    throw std::invalid_argument("incomplete draft window");

  const uint32_t pageTokens = DraftStateLayout::blockTokens;
  const uint32_t pageCount = layout_.draft.tokens / pageTokens;
  std::vector<std::vector<CachedStateBlock>> slices(pageCount);
  if (restoreDraftState)
    for (const auto &part : draft)
      slices[(part.begin / pageTokens) % pageCount].push_back(part);
  // Only disk destinations and boundary merges write execution backing. Full
  // resident pages stay shared without a copy or a fresh allocation.
  for (uint32_t page = 0; page < pageCount; ++page) {
    const auto &parts = slices[page];
    if (parts.empty())
      continue;
    const auto &first =
        static_cast<const QwenStatePayload &>(*parts.front().payload);
    const bool writes =
        first.disk ||
        std::any_of(parts.begin(), parts.end(), [&](const auto &part) {
          const auto &payload =
              static_cast<const QwenStatePayload &>(*part.payload);
          return payload.disk || payload.page != first.page;
        });
    if (writes) {
      const auto admission =
          prepareDraftWrite(index, page * pageTokens, (page + 1) * pageTokens);
      if (!admission)
        throw metal::MetalAllocationError("cannot admit draft restore page",
                                          admission.failure);
    }
  }
  DraftKvCache::AllocationScope allocation(draftCache_);
  auto operations = std::make_shared<FileOperations>();
  operations->items.reserve(1 + draft.size());
  auto pages = destination.draft->pages;
  std::vector<DraftFragment> fragments;
  struct Read {
    const QwenStatePayload *payload;
    MetalBuffer buffer;
  };
  std::vector<Read> reads;
  reads.reserve(1 + draft.size());
  const MetalBuffer gdn =
      destination.buffers.gdn[destination.metadata.activeParity].stateBase;
  if (recurrent->disk)
    reads.push_back({recurrent.get(), gdn});
  for (uint32_t physical = 0; physical < pageCount; ++physical) {
    std::vector<std::pair<const void *, std::shared_ptr<DraftKvCache::Block>>>
        sources;
    for (const auto &part : slices[physical]) {
      const auto &payload =
          static_cast<const QwenStatePayload &>(*part.payload);
      const void *identity = payload.disk
                                 ? static_cast<const void *>(payload.disk.get())
                                 : payload.page.get();
      const auto found =
          std::find_if(sources.begin(), sources.end(), [&](const auto &source) {
            return source.first == identity;
          });
      std::shared_ptr<DraftKvCache::Block> page;
      if (found != sources.end()) {
        page = found->second;
      } else if (!payload.disk) {
        page = payload.page;
      } else {
        if (sources.empty())
          page = destination.draft->pages[physical];
        else {
          metal::AllocationFailure failure = metal::AllocationFailure::None;
          page = draftCache_.acquire(&failure);
          if (!page)
            throw metal::MetalAllocationError("cannot admit boundary fragment",
                                              failure);
        }
        reads.push_back({&payload, page->page->buffer});
        page->disk = payload.disk;
      }
      if (found == sources.end())
        sources.emplace_back(identity, page);
      fragments.push_back(
          {std::max(begin, part.begin), part.end, std::move(page)});
    }
    if (sources.size() == 1)
      pages[physical] = sources.front().second;
    // Different fragments sharing a circular slot merge into its private
    // execution page, never into an immutable cached source.
    if (sources.size() > 1)
      pages[physical]->disk.reset();
  }
  auto views = draftCache_.views(pages);
  auto commit = [this, index, lengths, restoreDraftState, recurrent, gdn,
                 pages = std::move(pages), views = std::move(views),
                 fragments = std::move(fragments),
                 committed = std::move(committed)]() mutable {
    if (!recurrent->disk)
      copyExact(gdn, recurrent->buffer(), "restored recurrent checkpoint");
    if (restoreDraftState) {
      for (const auto &fragment : fragments) {
        const auto physical =
            (fragment.begin / DraftStateLayout::blockTokens) % pages.size();
        copyDraftSlice(pages[physical]->page->buffer,
                       fragment.page->page->buffer, layout_.draft,
                       fragment.begin, fragment.end);
      }
      slot(index).draft->pages = std::move(pages);
      slot(index).buffers.draft = std::move(views);
    }
    restoreLengths(index, lengths, restoreDraftState);
    committed();
  };
  auto snapshot = [this, index, lengths, restoreDraftState] {
    return restoreDraftState ? this->snapshot(index, lengths) : nullptr;
  };
  auto ticket = std::make_unique<FileRestore>(operations, std::move(commit),
                                              std::move(snapshot));
  for (const auto &read : reads)
    operations->items.push_back(read.payload->file->read(
        read.payload->disk,
        {{static_cast<std::byte *>(read.buffer.contents()),
          read.buffer.sizeBytes()}},
        completion));
  allocation.commit();
  if (reads.empty()) {
    static_cast<void>(ticket->finish());
    return {};
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
