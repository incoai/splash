#include "model/Runtime.hpp"
#include "model/QwenState.hpp"
#include "model/QwenTarget.hpp"
#include "model/RuntimeArenas.hpp"

#include "metal/CommandGraph.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"
#include "ops/RoPE.hpp"
#include "ops/Sampling.hpp"
#include "ops/Vision.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

// Fixed reserves the memory plan carries beside the planned arenas: Metal
// pipeline objects and encoder scratch, and the process's own runtime overhead.
constexpr uint64_t kPipelineReserveBytes = 256ULL << 20;
constexpr uint64_t kRuntimeOverheadReserveBytes = 512ULL << 20;

using metal::BufferStorage;
using metal::CommandGraph;
using metal::CommandTicket;
using metal::CommandTiming;
using metal::MetalBackend;
using metal::MetalBuffer;

class DeferredMetalTicket final : public ModelBatchTicket {
public:
  using Completion = std::function<std::vector<ModelStepResult>(CommandTiming)>;

  DeferredMetalTicket(CommandTicket ticket, Completion completion,
                      double priorWallMilliseconds = 0.0,
                      bool representativePrefillTiming = true)
      : ticket_(std::move(ticket)), completion_(std::move(completion)),
        wallMilliseconds_(priorWallMilliseconds),
        representativePrefillTiming_(representativePrefillTiming) {}

  bool ready() const noexcept override { return ticket_.ready(); }

  std::vector<ModelStepResult> wait() override {
    if (!completion_) {
      throw std::logic_error("Metal ticket was already consumed");
    }
    CommandTiming timing = ticket_.wait();
    wallMilliseconds_ += timing.wallSeconds * 1000.0;
    Completion completion = std::move(completion_);
    return completion(timing);
  }

  double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }
  bool prefillTimingIsRepresentative() const noexcept override {
    return representativePrefillTiming_;
  }

private:
  CommandTicket ticket_;
  Completion completion_;
  double wallMilliseconds_ = 0.0;
  bool representativePrefillTiming_;
};

class ReadyModelTicket final : public ModelBatchTicket {
public:
  ReadyModelTicket(std::vector<ModelStepResult> results,
                   double wallMilliseconds)
      : results_(std::move(results)), wallMilliseconds_(wallMilliseconds) {}

  bool ready() const noexcept override { return true; }

  std::vector<ModelStepResult> wait() override {
    if (!results_) {
      throw std::logic_error("ready model ticket was already consumed");
    }
    std::vector<ModelStepResult> results = std::move(*results_);
    results_.reset();
    return results;
  }

  double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }

private:
  std::optional<std::vector<ModelStepResult>> results_;
  double wallMilliseconds_ = 0.0;
};

using kv::Q8ChunkedPrefillParams;

bool isStopToken(const RuntimeGeometry &geometry, uint32_t token) noexcept {
  return token == geometry.target.stopTokens[0] ||
         token == geometry.target.stopTokens[1];
}

void requireShared(const MetalBuffer &buffer, std::string_view label) {
  if (!buffer || buffer.storage() != BufferStorage::Shared ||
      !buffer.contents()) {
    throw std::logic_error(std::string(label) + " is not CPU-visible");
  }
}

template <class T>
T *contents(const MetalBuffer &buffer, std::string_view label) {
  requireShared(buffer, label);
  return static_cast<T *>(buffer.contents());
}

void validatePlan(const BatchPlan &plan, std::span<const ModelBatchItem> items,
                  WorkKind expected) {
  if (plan.kind != expected || plan.empty() || plan.width() > kLaneCount ||
      items.size() != plan.items.size()) {
    throw std::invalid_argument("model runtime received an invalid batch plan");
  }
  for (size_t index = 0; index < items.size(); ++index) {
    if (items[index].requestId != plan.items[index].requestId ||
        items[index].stateSlot >= kLaneCount ||
        (expected == WorkKind::Prefill &&
         (!plan.items[index].tokenCount ||
          plan.items[index].tokenCount != items[index].tokenCount ||
          items[index].inputTokens.size() != items[index].tokenCount)) ||
        (expected == WorkKind::Decode &&
         (plan.items[index].tokenCount || items[index].tokenCount ||
          !items[index].inputTokens.empty()))) {
      throw std::invalid_argument("batch items do not match explicit plan");
    }
  }
}

QwenStateStorage &requireQwenStateStorage(StateStorage &storage) {
  auto *qwen = dynamic_cast<QwenStateStorage *>(&storage);
  if (!qwen) {
    throw std::invalid_argument(
        "Qwen runtime requires Qwen composite state storage");
  }
  return *qwen;
}

// Any unassigned slot works: its buffers come from the storage's pool, and
// the governor is asked only for what the pool lacks.
template <class Activate>
StateAdmission admitIdleSlot(const QwenStateStorage &states,
                             Activate activate) {
  for (uint32_t slot = 0; slot < kLaneCount; ++slot) {
    if (states.metadata(slot).assigned)
      continue;
    const metal::AllocationResult admission = activate(slot);
    if (admission)
      return {slot, StateFailure::None};
    return {{}, StateFailure::MemoryPressure, admission.failure};
  }
  return {{}, StateFailure::ConcurrencyLimit};
}

} // namespace

struct Runtime::Impl {
  // Repeated placements share one encode and its buffers. Pixels are released
  // on completion; embeddings remain until every placement has finished.
  struct ImageData final {
    MetalBuffer pixels;
    MetalBuffer embeddings;
    bool encoding = false;
    bool encoded = false;
  };

  struct ImageState final {
    ImageSpan span;
    std::shared_ptr<ImageData> data;
  };

  struct Request final {
    uint64_t id = 0;
    uint32_t slot = 0;
    bool resident = false;
    bool promptComplete = false;
    // Rebuild state from already-emitted tokens without sampling an initial
    // anchor, consuming RNG, or replaying output to the caller.
    bool replayingGeneration = false;
    uint32_t promptTokens = 0;
    uint32_t maxNewTokens = 0;
    uint32_t generatedTokens = 0;
    BatchCohort cohort = BatchCohort::Greedy;
    SamplingParameters sampling;
    ConstraintMode constraint = ConstraintMode::None;
    std::optional<uint32_t> pendingToken;
    // Transient active-request hidden used only while a constrained request
    // waits for its first token mask. Composite cache state never stores it;
    // every cache hit replays one input token and regenerates this value.
    std::vector<uint16_t> finalTargetHidden;
    std::array<float, kSamplingUniformCount> cycleUniforms{};
    // Nonempty selects score-only mode: the final prefill chunk computes raw
    // logits at these token ids instead of selecting an anchor.
    std::vector<uint32_t> scoreTokens;
    std::vector<uint32_t> maskWords;
    // Set only while the current scheduler-owned ticket overlaps grammar-mask
    // computation with target verification. This is model runtime state, not a
    // scheduler decode stage.
    bool verifyMaskInFlight = false;
    uint64_t rngCounter = 0;
    DecodeStage decodeStage = DecodeStage::Regular;
    bool draftContextValid = false;
    uint64_t draftContextThrough = 0;
    std::optional<DraftContextPlan> draftContextPlan;
    std::vector<ImageState> images;
  };

  struct DecodeLaneResult final {
    Request *request = nullptr;
    uint32_t retained = 0;
    uint32_t accepted = 0;
    uint32_t nextAnchor = 0;
    uint32_t currentAnchor = 0;
    uint32_t maximumRetained = 0;
    bool verify = false;
    bool draftForMask = false;
    bool draftComputed = false;
  };

  struct PageTableBinding final {
    uint64_t requestId = 0;
    uint64_t revision = 0;
    uint32_t entries = 0;
  };

  MetalBackend &backend;
  metal::AllocationAdmission admitAllocation;
  const ModelPackage &package;
  const RuntimeGeometry geometry;
  const ops::ExecutionPlans &operators;
  kv::Q8PageStorage &kvPages;
  QwenStateStorage &states;
  std::unique_ptr<PrefillArena> prefillArena;
  std::unique_ptr<DecodeArena> decodeArena;
  std::unordered_map<uint64_t, Request> requests;
  // Allocated for image cache misses and reclaimable once pending encodes
  // finish. Injecting already encoded rows needs no vision arena.
  std::unique_ptr<ops::Vision> vision;
  // Image buffers owned by the current admission attempt until a state cell
  // is activated. Failed attempts leave no image allocations behind.
  std::unordered_map<uint64_t, std::vector<ImageState>> stagedImages;
  // Encoded rows retained for reuse, including prefix hits that land inside
  // an image and still need its remaining rows. Byte-bounded LRU; the memory
  // reclaimer drops it entirely.
  struct CachedEmbeddings final {
    ImageSpan key;
    MetalBuffer embeddings;
  };
  static constexpr uint64_t kEmbeddingCacheBytes = 512ULL * 1024 * 1024;
  std::list<CachedEmbeddings> embeddingCache;
  uint64_t embeddingCacheBytes = 0;
  uint32_t maximumImagePatches = 0;
  uint64_t pipelineReserveBytes = 0;
  uint64_t runtimeOverheadReserveBytes = 0;
  std::array<PageTableBinding, kLaneCount> pageTableBindings{};
  ModelTelemetry counters;
  ops::Sampling sampling;
  QwenTarget targetModel;
  DFlashDraft draftModel;
  explicit Impl(RuntimeContext value)
      : backend(value.backend),
        admitAllocation(std::move(value.admitAllocation)),
        package(value.package), geometry(RuntimeGeometry::from(value.package)),
        operators(value.operators),
        kvPages(value.kvPages),
        states(requireQwenStateStorage(value.stateStorage)),
        maximumImagePatches(value.maximumImagePatches),
        pipelineReserveBytes(value.pipelineReserveBytes),
        runtimeOverheadReserveBytes(value.runtimeOverheadReserveBytes),
        sampling(value.backend, geometry.target.vocabularySize, kDecodeRows),
        targetModel(std::visit(
                        [&](const auto &weights) {
                          return QwenTarget(weights, value.backend, operators);
                        },
                        value.package.target)),
        draftModel(value.package.draft, value.backend, operators) {
    if (!admitAllocation)
      throw std::invalid_argument(
          "model runtime requires allocation admission");
    if (states.layout() != package.stateLayout() ||
        kvPages.layout() != package.targetKvLayout()) {
      throw std::invalid_argument(
          "model runtime resources do not match the loaded package");
    }
    prefillArena = std::make_unique<PrefillArena>(backend, geometry, operators);
    decodeArena = std::make_unique<DecodeArena>(backend, geometry, operators);
  }

  Request &request(uint64_t id) {
    auto found = requests.find(id);
    if (found == requests.end())
      throw std::out_of_range("unknown request");
    return found->second;
  }

  static bool samplingEnabled(const Request &entry) noexcept {
    return entry.sampling.temperature > 0.0F;
  }

  // Qwen3.5 M-RoPE: text rows advance one counter shared by all three axes;
  // an image's rows spread over (t, h, w) from the counter at the image start
  // and the counter then advances by max(merged height, merged width).
  static std::array<uint32_t, 3> ropePosition(const Request &entry,
                                              uint64_t logical) {
    int64_t delta = 0;
    for (const ImageState &image : entry.images) {
      const ImageSpan &span = image.span;
      if (logical < span.offset)
        break;
      const uint32_t mergedHeight = span.gridHeight / 2;
      const uint32_t mergedWidth = span.gridWidth / 2;
      const uint32_t start =
          static_cast<uint32_t>(static_cast<int64_t>(span.offset) + delta);
      if (logical < span.end()) {
        const uint32_t local = static_cast<uint32_t>(logical - span.offset);
        return {start, start + local / mergedWidth,
                start + local % mergedWidth};
      }
      delta += static_cast<int64_t>(std::max(mergedHeight, mergedWidth)) -
               static_cast<int64_t>(span.tokens);
    }
    const uint32_t position =
        static_cast<uint32_t>(static_cast<int64_t>(logical) + delta);
    return {position, position, position};
  }

  uint64_t embeddingBytes(const ImageSpan &span) const {
    return uint64_t{
               ops::Vision::embeddingRows({span.gridHeight, span.gridWidth})} *
           geometry.target.hiddenSize * sizeof(uint16_t);
  }

  static bool sameImage(const ImageSpan &left,
                        const ImageSpan &right) noexcept {
    return left.digestLo == right.digestLo && left.digestHi == right.digestHi &&
           left.gridHeight == right.gridHeight &&
           left.gridWidth == right.gridWidth;
  }

  // Encoded rows for an identical image, moved to the front of the LRU.
  MetalBuffer cachedEmbeddings(const ImageSpan &span) {
    for (auto entry = embeddingCache.begin(); entry != embeddingCache.end();
         ++entry) {
      if (!sameImage(entry->key, span))
        continue;
      embeddingCache.splice(embeddingCache.begin(), embeddingCache, entry);
      return entry->embeddings;
    }
    return {};
  }

  void retainEmbeddings(const ImageState &image) {
    if (!image.data || !image.data->encoded || !image.data->embeddings ||
        embeddingBytes(image.span) > kEmbeddingCacheBytes ||
        cachedEmbeddings(image.span)) {
      return;
    }
    embeddingCache.push_front({image.span, image.data->embeddings});
    embeddingCacheBytes += embeddingBytes(image.span);
    while (embeddingCacheBytes > kEmbeddingCacheBytes) {
      embeddingCacheBytes -= embeddingBytes(embeddingCache.back().key);
      embeddingCache.pop_back();
    }
  }

  // Rows served from the cache stay held by the request using them, so
  // dropping their entry frees nothing until that request ends.
  [[nodiscard]] bool
  embeddingsHeld(const MetalBuffer &embeddings) const noexcept {
    auto holds = [&](const std::vector<ImageState> &images) {
      for (const ImageState &image : images) {
        if (image.data && image.data->embeddings.sameView(embeddings))
          return true;
      }
      return false;
    };
    for (const auto &[_, images] : stagedImages) {
      if (holds(images))
        return true;
    }
    for (const auto &[_, entry] : requests) {
      if (holds(entry.images))
        return true;
    }
    return false;
  }

  uint64_t dropEmbeddingCache() noexcept {
    uint64_t released = 0;
    for (const CachedEmbeddings &entry : embeddingCache) {
      if (!embeddingsHeld(entry.embeddings))
        released += embeddingBytes(entry.key);
    }
    embeddingCache.clear();
    embeddingCacheBytes = 0;
    return released;
  }

  struct ImageAdmission final {
    Impl &runtime;
    uint64_t requestId;
    bool hadVision;
    bool committed = false;

    ImageAdmission(Impl &owner, uint64_t id)
        : runtime(owner), requestId(id), hadVision(bool(owner.vision)) {}
    ~ImageAdmission() {
      if (!committed) {
        runtime.stagedImages.erase(requestId);
        if (!hadVision)
          runtime.vision.reset();
      }
    }
  };

  // Admits the memory an image request needs before its state cell: the
  // shared vision scratch and per-image pixel and embedding buffers, all
  // through the governor, preserving the allocation refusal reason.
  metal::AllocationResult stageImages(const ModelRequest &request) {
    if (request.images.empty() || stagedImages.contains(request.id))
      return true;
    std::vector<ImageState> staged;
    staged.reserve(request.images.size());
    uint64_t bytes = 0;
    for (const ImageSpan &span : request.images) {
      ImageState image{span, {}};
      const auto duplicate = std::find_if(
          staged.begin(), staged.end(), [&](const ImageState &previous) {
            return sameImage(previous.span, span);
          });
      if (duplicate != staged.end()) {
        image.data = duplicate->data;
      } else {
        image.data = std::make_shared<ImageData>();
        image.data->embeddings = cachedEmbeddings(span);
        image.data->encoded = static_cast<bool>(image.data->embeddings);
        if (image.data->encoded)
          ++counters.imageEmbeddingReuses;
        else
          bytes += span.pixelBytes() + embeddingBytes(span);
      }
      staged.push_back(std::move(image));
    }
    // Keep cache references alive during admission. Only misses need the
    // encoder; cached rows can be injected after its arena has been reclaimed.
    if (bytes && !vision) {
      std::unique_ptr<ops::Vision> candidate;
      const auto admission = admitAllocation(
              ops::Vision::scratchBytes(package.vision.tensors.layout,
                                        maximumImagePatches),
              [&] {
                candidate = std::make_unique<ops::Vision>(
                    backend, package.vision.tensors, maximumImagePatches);
              });
      if (!admission)
        return admission;
      vision = std::move(candidate);
    }
    const uint8_t *pixels = request.imagePixels.data();
    const auto allocateImages = [&] {
      for (ImageState &image : staged) {
        const ImageSpan &span = image.span;
        if (!image.data->embeddings) {
          image.data->pixels = backend.allocateBuffer(
              span.pixelBytes(), BufferStorage::Shared, "image pixels");
          std::memcpy(contents<uint8_t>(image.data->pixels, "image pixels"), pixels,
                      static_cast<size_t>(span.pixelBytes()));
          image.data->embeddings = backend.allocateBuffer(
              embeddingBytes(span), BufferStorage::Private, "image embeddings");
        }
        pixels += span.pixelBytes();
      }
    };
    if (bytes) {
      if (auto admission = admitAllocation(bytes, allocateImages); !admission)
        return admission;
    }
    stagedImages.emplace(request.id, std::move(staged));
    return true;
  }

  [[nodiscard]] bool visionIdle() const noexcept {
    if (!stagedImages.empty())
      return false;
    for (const auto &[_, entry] : requests) {
      for (const ImageState &image : entry.images) {
        if (image.data && !image.data->encoded && image.data->embeddings)
          return false;
      }
    }
    return true;
  }

  // Encodes every image whose rows first appear in this chunk and overwrites
  // the chunk's placeholder embedding rows with the image rows. Text-only
  // requests add no dispatches.
  void addImageRows(CommandGraph &graph, Request &entry,
                    const ModelBatchItem &item, uint32_t rowBegin) {
    const uint64_t chunkBegin = item.promptOffset;
    const uint64_t chunkEnd = chunkBegin + item.tokenCount;
    for (ImageState &image : entry.images) {
      const uint64_t begin = std::max<uint64_t>(chunkBegin, image.span.offset);
      const uint64_t end = std::min<uint64_t>(chunkEnd, image.span.end());
      if (begin >= end || !image.data || !image.data->embeddings)
        continue;
      ImageData &data = *image.data;
      if (!data.encoded && !data.encoding) {
        if (!vision)
          throw std::logic_error("image request has no vision encoder");
        vision->encode(graph, {image.span.gridHeight, image.span.gridWidth},
                       data.pixels, data.embeddings);
        data.encoding = true;
        ++counters.imageEncodes;
      }
      const uint32_t rows = static_cast<uint32_t>(end - begin);
      ops::Vision::inject(
          graph, data.embeddings, prefillArena->get(PrefillTensor::Hidden0),
          package.vision.tensors.layout.outputHiddenSize,
          static_cast<uint32_t>(begin - image.span.offset),
          rowBegin + static_cast<uint32_t>(begin - chunkBegin), rows);
    }
  }

  static float nextUniform(Request &entry) noexcept {
    uint64_t value =
        entry.sampling.seed + (++entry.rngCounter) * 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return float(value >> 40) * 0x1p-24F;
  }

  static void stageSamplingCycle(Request &entry) noexcept {
    entry.cycleUniforms.fill(0.0F);
    for (uint32_t index = 1; index < entry.cycleUniforms.size(); ++index) {
      entry.cycleUniforms[index] = nextUniform(entry);
    }
  }

  [[nodiscard]] uint64_t estimatedWarmupPeak() const {
    uint64_t result = 0;
    auto add = [&](uint64_t bytes, std::string_view label) {
      result = checkedAdd(result, bytes, label);
    };
    add(package.targetActualAllocatedBytes(), "warmup target weights");
    add(package.draft.actualAllocatedBytes, "warmup draft weights");
    add(package.vision.actualAllocatedBytes, "warmup vision weights");
    add(states.actualAllocatedBytes(), "warmup state slots");
    add(prefillArena->bytes(), "warmup prefill arena");
    add(decodeArena->bytes(), "warmup decode arena");
    add(kvPages.actualAllocatedBytes(), "warmup Q8 pool");
    add(pipelineReserveBytes, "warmup pipeline reserve");
    add(runtimeOverheadReserveBytes, "warmup runtime reserve");
    return result;
  }

  void copyPageTable(const MetalBuffer &destination,
                     std::span<const uint32_t> pages) const {
    if (pages.empty() || pages.size() > kMaximumPageTableEntries) {
      throw std::invalid_argument("request page table has invalid length");
    }
    auto *target = contents<uint32_t>(destination, "request page table");
    std::copy(pages.begin(), pages.end(), target);
  }

  [[nodiscard]] MetalBuffer synchronizedPageTable(Request &entry,
                                                  const ModelBatchItem &item) {
    if (entry.slot >= pageTableBindings.size())
      throw std::out_of_range("request state slot is outside page tables");
    PageTableBinding &binding = pageTableBindings[entry.slot];
    MetalBuffer destination =
        decodeArena->get(entry.slot, DecodeTensor::PageTable);
    const bool unversioned = item.pageTableRevision == 0;
    if (unversioned || binding.requestId != entry.id ||
        binding.revision != item.pageTableRevision ||
        binding.entries != item.pageTable.size()) {
      copyPageTable(destination, item.pageTable);
      binding = {entry.id, item.pageTableRevision,
                 static_cast<uint32_t>(item.pageTable.size())};
    }
    return destination;
  }

  void addRopeTables(CommandGraph &graph, MetalBuffer targetPositions,
                     uint32_t targetRows, MetalBuffer draftPositions,
                     uint32_t draftRows, MetalBuffer targetCos,
                     MetalBuffer targetSin, MetalBuffer draftCos,
                     MetalBuffer draftSin) const {
    ops::RoPE::addTables(
        graph, std::move(targetPositions), std::move(draftPositions),
        prefillArena->get(PrefillTensor::TargetInverseFrequencies),
        prefillArena->get(PrefillTensor::DraftInverseFrequencies),
        std::move(targetCos), std::move(targetSin), std::move(draftCos),
        std::move(draftSin), {targetRows, draftRows}, kPrefillRows);
  }

  void captureFinalHidden(Request &entry, const MetalBuffer &rows,
                          uint32_t row) const {
    if (row >= kDecodeRows) {
      throw std::out_of_range("final hidden row is out of range");
    }
    const uint16_t *source =
        contents<uint16_t>(rows, "target final hidden source");
    entry.finalTargetHidden.assign(
        source + uint64_t{row} * geometry.target.hiddenSize,
        source + uint64_t{row + 1} * geometry.target.hiddenSize);
  }

  static DispatchDraftCapturePlan
  activeDraftCaptures(const Request &entry, const ModelBatchItem &item) {
    if (!entry.draftContextPlan) {
      throw std::logic_error("prefill request has no draft context plan");
    }
    const uint64_t next = item.logicalPosition + item.tokenCount;
    return draftCaptureSpansForDispatch(
        *entry.draftContextPlan, static_cast<uint32_t>(item.logicalPosition),
        static_cast<uint32_t>(next));
  }

  static uint32_t captureRows(const DispatchDraftCapturePlan &captures) {
    uint64_t rows = 0;
    for (const auto &capture : captures) {
      rows += capture.absoluteEnd - capture.absoluteBegin;
    }
    if (rows > kPrefillRows) {
      throw std::logic_error("draft capture exceeds packed prefill capacity");
    }
    return static_cast<uint32_t>(rows);
  }

  static QwenLogicalLengths
  advanceDraftContext(const QwenLogicalLengths &previous, uint64_t targetTokens,
                      const DispatchDraftCaptureSpan &capture) {
    QwenLogicalLengths next = previous;
    next.targetTokens = targetTokens;
    const uint32_t rows = capture.absoluteEnd - capture.absoluteBegin;
    if (!rows)
      return next;
    const bool continues =
        !capture.resetDraftState &&
        previous.draftEnd() == capture.absoluteBegin &&
        previous.draftCommitCursor == capture.absoluteBegin % kDraftCacheStride;
    const uint64_t combined =
        continues ? uint64_t{previous.draftLength} + rows : rows;
    next.draftLength =
        static_cast<uint32_t>(std::min<uint64_t>(combined, kDraftCacheStride));
    next.draftBase = capture.absoluteEnd - next.draftLength;
    next.draftCommitCursor =
        static_cast<uint32_t>(capture.absoluteEnd % kDraftCacheStride);
    return next;
  }

  void loadPolicyBuffers(Request &entry, uint32_t lane,
                         std::span<const uint32_t> masks) const {
    auto uniforms = decodeArena->get(lane, DecodeTensor::SamplingUniforms);
    auto *uniformData = contents<float>(uniforms, "sampling uniforms");
    std::copy(entry.cycleUniforms.begin(), entry.cycleUniforms.end(),
              uniformData);

    auto constraint = decodeArena->get(lane, DecodeTensor::ConstraintMasks);
    auto *maskData = contents<uint32_t>(constraint, "constraint masks");
    const uint64_t capacity =
        uint64_t{ExecutionLimits::maximumStepTokens} * geometry.maskWords();
    std::fill(maskData, maskData + capacity,
              std::numeric_limits<uint32_t>::max());
    if (!masks.empty()) {
      if (masks.size() > capacity) {
        throw std::invalid_argument("constraint mask exceeds decode arena");
      }
      std::copy(masks.begin(), masks.end(), maskData);
    }
  }

  static ops::SamplingPolicy samplingPolicy(const Request &entry) noexcept {
    const bool enabled = samplingEnabled(entry);
    return {enabled ? entry.sampling.topK : 1,
            enabled ? entry.sampling.temperature : 0.0F,
            enabled ? entry.sampling.topP : 1.0F,
            entry.constraint == ConstraintMode::TokenMask};
  }

  template <class Get>
  static ops::SamplingBuffers samplingBuffersWith(Get d) {
    return {d(DecodeTensor::Logits),
            d(DecodeTensor::TargetTopPartialIds),
            d(DecodeTensor::TargetTopPartialValues),
            d(DecodeTensor::TargetTopIds),
            d(DecodeTensor::TargetTopProbs),
            d(DecodeTensor::SamplingUniforms),
            d(DecodeTensor::ConstraintMasks),
            d(DecodeTensor::OutputTokens),
            d(DecodeTensor::ArgmaxValues),
            d(DecodeTensor::ArgmaxIndices)};
  }

  ops::SamplingBuffers samplingBuffers(uint32_t lanes) const {
    return samplingBuffersWith(
        [&](DecodeTensor t) { return decodeArena->packed(t, lanes); });
  }

  ops::SamplingBuffers samplingBuffersForLane(uint32_t lane) const {
    return samplingBuffersWith(
        [&](DecodeTensor t) { return decodeArena->get(lane, t); });
  }

  void addInitialPolicySelection(CommandGraph &graph, Request &entry,
                                 uint32_t lane, uint32_t rowOffset) const {
    sampling.addInitial(graph, samplingPolicy(entry),
                        samplingBuffersForLane(lane), rowOffset);
  }

  CommandTiming selectPendingFromFinalHidden(Request &entry, uint32_t lane,
                                             std::span<const uint32_t> masks) {
    if (entry.finalTargetHidden.size() != geometry.target.hiddenSize) {
      throw std::logic_error("request has no policy-neutral final hidden");
    }
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->get(lane, tensor);
    };
    auto *hidden =
        contents<uint16_t>(d(DecodeTensor::Hidden0), "pending final hidden");
    for (uint32_t row = 0; row < kDecodeRows; ++row) {
      std::copy(entry.finalTargetHidden.begin(), entry.finalTargetHidden.end(),
                hidden + uint64_t{row} * geometry.target.hiddenSize);
    }
    entry.cycleUniforms.fill(0.0F);
    if (samplingEnabled(entry)) {
      entry.cycleUniforms[0] = nextUniform(entry);
    }
    loadPolicyBuffers(entry, lane, masks);

    CommandGraph graph;
    targetModel.addHead(graph, d(DecodeTensor::Hidden0),
                        d(DecodeTensor::FinalHidden), d(DecodeTensor::Logits),
                        kDecodeRows);
    addInitialPolicySelection(graph, entry, lane, 0);
    CommandTiming timing = backend.submitCommand(graph.dispatches());
    entry.pendingToken = *contents<uint32_t>(d(DecodeTensor::OutputTokens),
                                             "restored prefix next token");
    if (*entry.pendingToken >= geometry.target.vocabularySize) {
      throw std::runtime_error("target policy selected an invalid token");
    }
    return timing;
  }

  Q8ChunkedPrefillParams q8Params(uint64_t logicalPosition,
                                  uint32_t chunkTokens, uint32_t chunkStride,
                                  std::span<const uint32_t> pages) const {
    return ops::PagedAttention::prefillParams(
        logicalPosition, chunkTokens, chunkStride, pages, kvPages.pageCount());
  }

  struct PackedPrefillSequence final {
    Request *entry = nullptr;
    const ModelBatchItem *item = nullptr;
    uint32_t lane = 0;
    uint32_t rowBegin = 0;
    uint32_t attentionStride = 0;
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    uint32_t captureBegin = 0;
    Q8ChunkedPrefillParams q8;
    MetalBuffer pageTable;
    DispatchDraftCapturePlan captures;
  };

  struct PackedPrefillBatch final {
    std::vector<PackedPrefillSequence> sequences;
    uint32_t rows = 0;
    uint32_t capturedRows = 0;
  };

  MetalBuffer prefillU16(PrefillTensor tensor, uint32_t begin, uint32_t rows,
                         uint32_t width) const {
    return backend.view(prefillArena->get(tensor),
                        bytesFor<uint16_t>(uint64_t{begin} * width),
                        bytesFor<uint16_t>(uint64_t{rows} * width));
  }

  PackedPrefillBatch
  preparePackedPrefill(std::span<const ModelBatchItem> items,
                       std::array<Request *, kLaneCount> &entries) {
    PackedPrefillBatch batch;
    batch.sequences.reserve(items.size());
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      const ModelBatchItem &item = items[lane];
      Request &entry = request(item.requestId);
      if (item.tokenCount > kPrefillRows ||
          item.promptOffset > entry.promptTokens ||
          item.tokenCount > entry.promptTokens - item.promptOffset ||
          item.logicalPosition != item.promptOffset || !entry.resident ||
          entry.slot != item.stateSlot) {
        throw std::invalid_argument("invalid packed Qwen prefill item");
      }
      const QwenSlotMetadata &metadata = states.metadata(entry.slot);
      if (!metadata.assigned || metadata.requestId != entry.id ||
          metadata.lengths.targetTokens != item.logicalPosition) {
        throw std::logic_error("packed prefill state length is not exact");
      }
      if (item.tokenCount > kPrefillRows - batch.rows) {
        throw std::invalid_argument("packed prefill exceeds actual-row budget");
      }
      auto captures = activeDraftCaptures(entry, item);
      const uint32_t capturedRows = captureRows(captures);
      if (capturedRows > kPrefillRows - batch.capturedRows) {
        throw std::invalid_argument("packed draft capture exceeds row budget");
      }
      const uint32_t attentionStride =
          ((item.tokenCount + kTileRows - 1) / kTileRows) * kTileRows;
      const Q8ChunkedPrefillParams q8 =
          q8Params(item.logicalPosition, item.tokenCount, attentionStride,
                   item.pageTable);
      MetalBuffer pageTable = synchronizedPageTable(entry, item);
      batch.sequences.push_back({&entry, &item, lane, batch.rows,
                                 attentionStride, queryOffset, kvOffset,
                                 batch.capturedRows, q8, std::move(pageTable),
                                 std::move(captures)});
      entries[lane] = &entry;
      batch.rows += item.tokenCount;
      batch.capturedRows += capturedRows;
      queryOffset += bytesFor<uint16_t>(
          uint64_t{geometry.target.attentionQueryHeads} * attentionStride *
          geometry.target.attentionHeadDimension);
      kvOffset += bytesFor<uint16_t>(
          uint64_t{geometry.target.attentionKvHeads} * attentionStride *
          geometry.target.attentionHeadDimension);
    }
    if (!batch.rows ||
        queryOffset >
            prefillArena->get(PrefillTensor::FullQueries).sizeBytes() ||
        kvOffset > prefillArena->get(PrefillTensor::ChunkKeys).sizeBytes()) {
      throw std::logic_error("packed prefill scratch geometry overflowed");
    }

    auto *input =
        contents<uint32_t>(prefillArena->get(PrefillTensor::InputTokens),
                           "packed prefill input tokens");
    auto *targetPositions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::TargetPositions),
                           "target RoPE positions");
    auto *draftPositions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::DraftPositions),
                           "draft RoPE positions");
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      const ModelBatchItem &item = *sequence.item;
      std::copy(item.inputTokens.begin(), item.inputTokens.end(),
                input + sequence.rowBegin);
      for (uint32_t localRow = 0; localRow < item.tokenCount; ++localRow) {
        const uint32_t row = sequence.rowBegin + localRow;
        if (input[row] >= geometry.target.vocabularySize) {
          throw std::invalid_argument("prompt token is out of vocabulary");
        }
        const std::array<uint32_t, 3> rotary =
            ropePosition(*sequence.entry, item.logicalPosition + localRow);
        std::copy(rotary.begin(), rotary.end(), targetPositions + row * 3);
      }
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        for (uint32_t row = capture.absoluteBegin; row < capture.absoluteEnd;
             ++row) {
          const uint32_t compactRow = sequence.captureBegin +
                                      capture.compactDestinationRow + row -
                                      capture.absoluteBegin;
          draftPositions[compactRow] = row;
        }
      }
    }
    return batch;
  }

  void addPackedDraftContext(CommandGraph &graph,
                             const PackedPrefillBatch &batch) {
    if (!batch.capturedRows)
      return;
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor); };
    std::array<DFlashPrefillSpan, kLaneCount * 2> spans{};
    uint32_t spanCount = 0;
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      const QwenSlotBuffers &slot = states.buffers(sequence.entry->slot);
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        DFlashPrefillSpan &span = spans.at(spanCount++);
        span.compactRow = sequence.captureBegin + capture.compactDestinationRow;
        span.rows = capture.absoluteEnd - capture.absoluteBegin;
        span.startPosition = capture.absoluteBegin;
        span.ring = slot.draft;
      }
    }
    draftModel.addContextPrefill(
        graph,
        {p(PrefillTensor::Captured), p(PrefillTensor::ProjectionSums),
         p(PrefillTensor::ContextProjected), p(PrefillTensor::ContextHidden),
         p(PrefillTensor::ContextQkv), p(PrefillTensor::DraftRopeCos),
         p(PrefillTensor::DraftRopeSin)},
        batch.capturedRows, std::span(spans).first(spanCount));
  }

  void encodePackedPrefillGraph(CommandGraph &graph,
                                std::span<const ModelBatchItem> items,
                                std::array<Request *, kLaneCount> &entries) {
    PackedPrefillBatch batch = preparePackedPrefill(items, entries);
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor); };

    addRopeTables(graph, p(PrefillTensor::TargetPositions), batch.rows,
                  p(PrefillTensor::DraftPositions), batch.capturedRows,
                  p(PrefillTensor::RopeCos), p(PrefillTensor::RopeSin),
                  p(PrefillTensor::DraftRopeCos),
                  p(PrefillTensor::DraftRopeSin));

    targetModel.addEmbedding(graph, p(PrefillTensor::InputTokens),
                             p(PrefillTensor::Hidden0), batch.rows);
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      addImageRows(graph, *sequence.entry, *sequence.item, sequence.rowBegin);
    }

    std::array<QwenTargetPrefillSequence, kLaneCount> modelSequences{};
    const uint32_t modelSequenceCount =
        static_cast<uint32_t>(batch.sequences.size());
    const uint64_t stateBindingCount = uint64_t{modelSequenceCount} *
                                       geometry.target.stateLayout.layers;
    std::vector<MetalBuffer> convolutionIn(stateBindingCount);
    std::vector<MetalBuffer> convolutionOut(stateBindingCount);
    std::vector<MetalBuffer> recurrentIn(stateBindingCount);
    std::vector<MetalBuffer> recurrentOut(stateBindingCount);
    for (uint32_t lane = 0; lane < batch.sequences.size(); ++lane) {
      const PackedPrefillSequence &sequence = batch.sequences[lane];
      QwenTargetPrefillSequence &destination = modelSequences[lane];
      destination.rowBegin = sequence.rowBegin;
      destination.rows = sequence.item->tokenCount;
      destination.attentionStride = sequence.attentionStride;
      destination.queryOffset = sequence.queryOffset;
      destination.kvOffset = sequence.kvOffset;
      destination.q8 = sequence.q8;
      destination.pageTable = sequence.pageTable;
      const uint32_t gdnLayers = geometry.target.stateLayout.layers;
      const uint64_t stateBegin = uint64_t{lane} * gdnLayers;
      destination.convolutionIn =
          std::span(convolutionIn).subspan(stateBegin, gdnLayers);
      destination.convolutionOut =
          std::span(convolutionOut).subspan(stateBegin, gdnLayers);
      destination.recurrentIn =
          std::span(recurrentIn).subspan(stateBegin, gdnLayers);
      destination.recurrentOut =
          std::span(recurrentOut).subspan(stateBegin, gdnLayers);
      const QwenSlotMetadata &metadata = states.metadata(sequence.entry->slot);
      const QwenSlotBuffers &slot = states.buffers(sequence.entry->slot);
      for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
        convolutionIn[stateBegin + layer] =
            slot.gdn[metadata.activeParity].convolutionLayers[layer];
        convolutionOut[stateBegin + layer] =
            slot.gdn[metadata.activeParity ^ 1].convolutionLayers[layer];
        recurrentIn[stateBegin + layer] =
            slot.gdn[metadata.activeParity].recurrentLayers[layer];
        recurrentOut[stateBegin + layer] =
            slot.gdn[metadata.activeParity ^ 1].recurrentLayers[layer];
      }
      destination.captureCount = sequence.captures.size();
      for (uint32_t index = 0; index < sequence.captures.size(); ++index) {
        const DispatchDraftCaptureSpan &capture = sequence.captures[index];
        destination.captures[index] = {
            sequence.rowBegin +
                static_cast<uint32_t>(capture.absoluteBegin -
                                      sequence.item->logicalPosition),
            sequence.captureBegin + capture.compactDestinationRow,
            capture.absoluteEnd - capture.absoluteBegin};
      }
    }
    QwenTargetPrefillBuffers buffers;
    buffers.hidden = {p(PrefillTensor::Hidden0), p(PrefillTensor::Hidden1)};
    buffers.normalized = p(PrefillTensor::Normalized);
    buffers.captured = p(PrefillTensor::Captured);
    buffers.gdnPacked = p(PrefillTensor::GdnPacked);
    buffers.gdnQueries = p(PrefillTensor::GdnQueries);
    buffers.gdnKeys = p(PrefillTensor::GdnKeys);
    buffers.gdnValues = p(PrefillTensor::GdnValues);
    buffers.gdnDecay = p(PrefillTensor::GdnDecay);
    buffers.gdnBeta = p(PrefillTensor::GdnBeta);
    buffers.recurrent = p(PrefillTensor::Recurrent);
    buffers.gdnHidden = p(PrefillTensor::GdnHidden);
    buffers.gdnOutput = p(PrefillTensor::GdnOutput);
    buffers.denseGateScratch = p(PrefillTensor::GateIntermediate);
    buffers.denseIntermediate = p(PrefillTensor::Intermediate);
    buffers.fullPacked = p(PrefillTensor::FullPacked);
    buffers.fullQueries = p(PrefillTensor::FullQueries);
    buffers.fullAttention = p(PrefillTensor::FullAttention);
    buffers.attentionPartials = p(PrefillTensor::AttentionPartials);
    buffers.attentionStatistics = p(PrefillTensor::AttentionStatistics);
    buffers.attentionHidden = p(PrefillTensor::AttentionHidden);
    buffers.attentionOutput = p(PrefillTensor::AttentionOutput);
    buffers.projectionSums = p(PrefillTensor::ProjectionSums);
    buffers.downProjectionSums = p(PrefillTensor::DownProjectionSums);
    buffers.ropeCos = p(PrefillTensor::RopeCos);
    buffers.ropeSin = p(PrefillTensor::RopeSin);
    buffers.chunkKeys = p(PrefillTensor::ChunkKeys);
    buffers.chunkValues = p(PrefillTensor::ChunkValues);
    buffers.selectedExperts = p(PrefillTensor::MoeSelectedExperts);
    buffers.routingWeights = p(PrefillTensor::MoeRoutingWeights);
    buffers.tileDescriptors = p(PrefillTensor::MoeTileDescriptors);
    buffers.tileCount = p(PrefillTensor::MoeTileCount);
    buffers.groupedRoutes = p(PrefillTensor::MoeGroupedRoutes);
    buffers.routeRows = p(PrefillTensor::MoeRouteRows);
    buffers.groupedInput = p(PrefillTensor::MoeGroupedInput);
    buffers.expertIntermediate = p(PrefillTensor::MoeExpertIntermediate);
    buffers.expertOutput = p(PrefillTensor::MoeExpertOutput);
    std::vector<kv::Q8LayerStorage> kvLayers(
        geometry.target.kvLayout.attentionLayers);
    for (uint32_t layer = 0; layer < kvLayers.size(); ++layer)
      kvLayers[layer] = kvPages.layer(layer);
    targetModel.addPrefill(
        graph, std::move(buffers),
        std::span(modelSequences).first(batch.sequences.size()), batch.rows,
        kvLayers);
    addPackedDraftContext(graph, batch);

    for (const PackedPrefillSequence &sequence : batch.sequences) {
      Request &entry = *sequence.entry;
      const ModelBatchItem &item = *sequence.item;
      if (entry.replayingGeneration ||
          item.logicalPosition + item.tokenCount != entry.promptTokens)
        continue;
      const bool scoring = !entry.scoreTokens.empty();
      auto d = [&](DecodeTensor tensor) {
        return decodeArena->get(sequence.lane, tensor);
      };
      const uint32_t lastRows = std::min(item.tokenCount, kDecodeRows);
      ops::DraftAttention::gatherLastRows(
          graph,
          prefillU16(PrefillTensor::Hidden0, sequence.rowBegin,
                     item.tokenCount, geometry.target.hiddenSize),
          d(DecodeTensor::Hidden0), item.tokenCount,
          geometry.target.hiddenSize);
      if (scoring) {
        // Score-only: compute raw logits at the final prompt position; no
        // policy selection, sampling, or anchor is produced.
        targetModel.addHead(graph, d(DecodeTensor::Hidden0),
                            d(DecodeTensor::FinalHidden),
                            d(DecodeTensor::Logits), lastRows);
      } else if (entry.constraint == ConstraintMode::None) {
        if (samplingEnabled(entry)) {
          entry.cycleUniforms.fill(0.0F);
          entry.cycleUniforms[0] = nextUniform(entry);
          loadPolicyBuffers(entry, sequence.lane, {});
        }
        addPrefillPolicy(graph, entry, sequence.lane, lastRows - 1);
      }
    }
  }

  void prepareDecodeLane(Request &entry, const ModelBatchItem &item,
                         uint32_t lane) {
    if (!entry.resident || entry.slot != item.stateSlot ||
        !entry.promptComplete || !entry.pendingToken) {
      throw std::logic_error("decode request is not ready");
    }
    const QwenSlotMetadata &metadata = states.metadata(entry.slot);
    if (metadata.lengths.targetTokens != item.logicalPosition ||
        !metadata.lengths.hasCompleteDraftWindow(kDraftCacheStride)) {
      throw std::logic_error("decode state length is not exact");
    }
    static_cast<void>(synchronizedPageTable(entry, item));
    auto *draftInput = contents<uint32_t>(
        decodeArena->get(lane, DecodeTensor::DraftInputTokens),
        "draft input tokens");
    draftInput[0] = *entry.pendingToken;
    std::fill(draftInput + 1, draftInput + kDecodeRows,
              geometry.target.maskToken);

    auto *positions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::Positions),
                           "decode RoPE positions");
    auto *draftPositions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::DraftPositions),
                           "decode draft RoPE positions");
    for (uint32_t row = 0; row < kDecodeRows; ++row) {
      const std::array<uint32_t, 3> rotary =
          ropePosition(entry, item.logicalPosition + row);
      std::copy(rotary.begin(), rotary.end(), positions + row * 3);
      // The draft is a text model over logical positions.
      draftPositions[row] = static_cast<uint32_t>(item.logicalPosition + row);
    }
    *contents<uint32_t>(decodeArena->get(lane, DecodeTensor::Arrived),
                        "decode arrived") = 0;
    *contents<uint32_t>(decodeArena->get(lane, DecodeTensor::Generation),
                        "decode generation") = 0;
  }

  // Batch lanes beyond the active width replay the last active request so
  // every padded M32 lane binds valid state.
  static Request &laneEntry(std::span<Request *const> entries, uint32_t lane) {
    Request *entry = entries[std::min<size_t>(lane, entries.size() - 1)];
    if (!entry)
      throw std::invalid_argument("empty decode batch lane");
    return *entry;
  }

  void bindDraftRings(
      std::span<Request *const> entries,
      std::vector<std::array<MetalBuffer, kLaneCount>> &keys,
      std::vector<std::array<MetalBuffer, kLaneCount>> &values) const {
    keys.resize(geometry.draft.layers);
    values.resize(geometry.draft.layers);
    for (uint32_t layer = 0; layer < geometry.draft.layers; ++layer) {
      for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
        const auto &ring =
            states.buffers(laneEntry(entries, lane).slot).draft[layer];
        keys[layer][lane] = ring.keys;
        values[layer][lane] = ring.values;
      }
    }
  }

  void encodeDraftBatchGraph(CommandGraph &graph,
                             std::span<Request *const> entries,
                             std::span<const uint64_t> logicalPositions,
                             ops::Q4DispatchStats &stats) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != logicalPositions.size()) {
      throw std::invalid_argument("invalid draft decode batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, lanes);
    };
    std::array<uint32_t, kLaneCount> cacheLengths{};
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      cacheLengths[lane] =
          static_cast<uint32_t>(logicalPositions[std::min(lane, lanes - 1)]);
    }

    DFlashDecodeBuffers buffers;
    for (uint32_t hidden = 0; hidden < buffers.hidden.size(); ++hidden) {
      buffers.hidden[hidden] = d(static_cast<DecodeTensor>(
          static_cast<uint32_t>(DecodeTensor::DraftHidden0) + hidden));
    }
    buffers.normalized = d(DecodeTensor::DraftNormalized);
    buffers.dynamic = d(DecodeTensor::DraftDynamic);
    buffers.convolved = d(DecodeTensor::DraftConvolved);
    buffers.proposalQkv = d(DecodeTensor::DraftProposalQkv);
    buffers.attention = d(DecodeTensor::DraftAttention);
    buffers.projected = d(DecodeTensor::DraftProjected);
    buffers.residual = d(DecodeTensor::DraftResidual);
    buffers.intermediate = d(DecodeTensor::DraftIntermediate);
    buffers.finalHidden = d(DecodeTensor::DraftFinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    buffers.selectorHidden = d(DecodeTensor::SelectorHidden);
    buffers.queryKeys = d(DecodeTensor::DraftQueryKeys);
    buffers.queryValues = d(DecodeTensor::DraftQueryValues);
    buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
    buffers.gateScratch = decodeArena->gateScratch();
    bindDraftRings(entries, buffers.persistentKeys, buffers.persistentValues);
    draftModel.addDecode(graph, std::move(buffers),
                         targetModel.vocabularyProjection(), cacheLengths,
                         lanes, stats);
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      Request &entry = laneEntry(entries, lane);
      if (!entry.pendingToken)
        throw std::invalid_argument("draft batch lane has no anchor");
      anchors[lane] = *entry.pendingToken;
      policies[lane] = samplingPolicy(entry);
    }
    draftModel.addSelection(
        graph,
        {d(DecodeTensor::Logits), d(DecodeTensor::TopPartialIds),
         d(DecodeTensor::TopPartialValues), d(DecodeTensor::Candidates),
         d(DecodeTensor::Unary), d(DecodeTensor::SelectorHidden),
         d(DecodeTensor::SamplingUniforms), d(DecodeTensor::ProposedTokens),
         d(DecodeTensor::ProposalProbs)},
        std::span(anchors).first(lanes), std::span(policies).first(lanes),
        kDraftProposalTokens);
  }

  void encodeTargetVerifyBatchForward(CommandGraph &graph,
                                      std::span<Request *const> entries,
                                      std::span<const ModelBatchItem> items,
                                      ops::Q4DispatchStats &stats) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size()) {
      throw std::invalid_argument("invalid target verify batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, lanes);
    };
    auto paddedItem = [&](uint32_t lane) -> const ModelBatchItem & {
      return items[std::min(lane, lanes - 1)];
    };

    std::array<Q8ChunkedPrefillParams, kLaneCount> q8{};
    std::array<kv::Q8VerifyAttentionParams, kLaneCount> verify{};
    const uint32_t gdnLayers = geometry.target.stateLayout.layers;
    const uint32_t attentionLayers =
        geometry.target.kvLayout.attentionLayers;
    std::vector<MetalBuffer> gdnPacked(gdnLayers);
    std::vector<MetalBuffer> gdnMixed(gdnLayers);
    std::vector<MetalBuffer> gdnDecay(gdnLayers);
    std::vector<MetalBuffer> gdnBeta(gdnLayers);
    std::vector<MetalBuffer> chunkKeys(attentionLayers);
    std::vector<MetalBuffer> chunkValues(attentionLayers);
    QwenTargetVerifyBuffers buffers;
    buffers.hidden = {d(DecodeTensor::Hidden0), d(DecodeTensor::Hidden1)};
    buffers.normalized = d(DecodeTensor::Normalized);
    buffers.recurrent = d(DecodeTensor::Recurrent);
    buffers.gdnHidden = d(DecodeTensor::GdnHidden);
    buffers.gdnOutput = d(DecodeTensor::GdnOutput);
    buffers.denseIntermediate = d(DecodeTensor::Intermediate);
    buffers.fullPacked = d(DecodeTensor::FullPacked);
    buffers.fullQueries = d(DecodeTensor::FullQueries);
    buffers.attentionPartials = d(DecodeTensor::AttentionPartials);
    buffers.attentionStatistics = d(DecodeTensor::AttentionStatistics);
    buffers.fullAttention = d(DecodeTensor::FullAttention);
    buffers.attentionHidden = d(DecodeTensor::AttentionHidden);
    buffers.attentionOutput = d(DecodeTensor::AttentionOutput);
    buffers.ropeCos = d(DecodeTensor::RopeCos);
    buffers.ropeSin = d(DecodeTensor::RopeSin);
    buffers.arrived = d(DecodeTensor::Arrived);
    buffers.generation = d(DecodeTensor::Generation);
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.finalHidden = d(DecodeTensor::FinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    buffers.denseGateScratch = decodeArena->gateScratch();
    buffers.gdnPacked = gdnPacked;
    buffers.gdnMixed = gdnMixed;
    buffers.gdnDecay = gdnDecay;
    buffers.gdnBeta = gdnBeta;
    buffers.chunkKeys = chunkKeys;
    buffers.chunkValues = chunkValues;
    buffers.selectedExperts = d(DecodeTensor::MoeSelectedExperts);
    buffers.routingWeights = d(DecodeTensor::MoeRoutingWeights);
    buffers.tileDescriptors = d(DecodeTensor::MoeTileDescriptors);
    buffers.tileCount = d(DecodeTensor::MoeTileCount);
    buffers.groupedRoutes = d(DecodeTensor::MoeGroupedRoutes);
    buffers.routeRows = d(DecodeTensor::MoeRouteRows);
    buffers.groupedInput = d(DecodeTensor::MoeGroupedInput);
    buffers.expertIntermediate = d(DecodeTensor::MoeExpertIntermediate);
    buffers.expertOutput = d(DecodeTensor::MoeExpertOutput);
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      const ModelBatchItem &item = paddedItem(lane);
      q8[lane] = q8Params(item.logicalPosition, kDecodeRows, kTileRows,
                          item.pageTable);
      verify[lane] = kv::q8VerifyAttentionParams(
          q8[lane].committed_tokens, q8[lane].chunk_tokens,
          q8[lane].chunk_stride, q8[lane].page_table_entries,
          q8[lane].physical_page_count);
      if (!kv::q8VerifyAttentionValidationError(verify[lane]).empty())
        throw std::invalid_argument("invalid batched Q8 verify geometry");
      Request &entry = laneEntry(entries, lane);
      buffers.pageTables[lane] =
          decodeArena->get(entry.slot, DecodeTensor::PageTable);
      const uint32_t active = states.metadata(entry.slot).activeParity;
      buffers.currentGdnStates[lane] =
          states.buffers(entry.slot).gdn[active].stateBase;
      buffers.nextGdnStates[lane] =
          states.buffers(entry.slot).gdn[active ^ 1].stateBase;
    }
    for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
      gdnPacked[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyPackedBase, layer, lanes);
      gdnMixed[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyMixedBase, layer, lanes);
      gdnDecay[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyDecayBase, layer, lanes);
      gdnBeta[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyBetaBase, layer, lanes);
    }
    std::vector<kv::Q8LayerStorage> kvLayers(attentionLayers);
    for (uint32_t layer = 0; layer < attentionLayers; ++layer) {
      chunkKeys[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkKeysBase, layer, lanes);
      chunkValues[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkValuesBase, layer, lanes);
      kvLayers[layer] = kvPages.layer(layer);
    }
    targetModel.addVerify(graph, std::move(buffers), kvLayers, q8, verify,
                          lanes, stats);
  }

  void encodeTargetVerifyBatchPolicy(CommandGraph &graph,
                                     std::span<Request *const> entries) {
    if (entries.empty() || entries.size() > kLaneCount)
      throw std::invalid_argument("invalid target policy batch");
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (!entries[lane])
        throw std::invalid_argument("empty target policy lane");
      policies[lane] = samplingPolicy(*entries[lane]);
    }
    sampling.addVerify(graph, std::span(policies).first(lanes),
                       samplingBuffers(lanes));
  }

  void addPrefillPolicy(CommandGraph &graph, Request &entry, uint32_t lane,
                        uint32_t finalRow) const {
    if (finalRow >= kDecodeRows || entry.constraint != ConstraintMode::None) {
      throw std::invalid_argument("invalid prefill policy boundary");
    }
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->get(lane, tensor);
    };
    targetModel.addHead(graph, d(DecodeTensor::Hidden0),
                        d(DecodeTensor::FinalHidden), d(DecodeTensor::Logits),
                        finalRow + 1);
    addInitialPolicySelection(graph, entry, lane, finalRow);
  }

  void encodeDraftStateCommitBatch(CommandGraph &graph,
                                   std::span<Request *const> entries,
                                   std::span<const ModelBatchItem> items,
                                   ops::Q4DispatchStats &stats) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size()) {
      throw std::invalid_argument("invalid draft state commit batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, lanes);
    };

    std::array<uint32_t, kLaneCount> startPositions{};
    for (uint32_t lane = 0; lane < kLaneCount; ++lane)
      startPositions[lane] = static_cast<uint32_t>(
          items[std::min(lane, lanes - 1)].logicalPosition);
    DFlashContextBuffers buffers;
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.projected = d(DecodeTensor::ContextProjected);
    buffers.hidden = d(DecodeTensor::ContextHidden);
    buffers.qkv = d(DecodeTensor::ContextQkv);
    buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
    buffers.retainedCounts = d(DecodeTensor::RetainedCount);
    bindDraftRings(entries, buffers.persistentKeys, buffers.persistentValues);
    draftModel.addContextCommit(graph, std::move(buffers), startPositions,
                                lanes, stats);
  }

  void encodeBatchAcceptance(CommandGraph &graph,
                             std::span<Request *const> lanes,
                             std::span<const uint32_t> maximumRetained) {
    if (lanes.empty() || lanes.size() > kLaneCount ||
        lanes.size() != maximumRetained.size()) {
      throw std::invalid_argument("invalid DFlash acceptance batch");
    }
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
      if (!lanes[lane] || !maximumRetained[lane] ||
          maximumRetained[lane] > kDecodeRows) {
        throw std::invalid_argument("invalid DFlash acceptance lane");
      }
      policies[lane] = samplingPolicy(*lanes[lane]);
    }
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    sampling.addAcceptance(
        graph,
        {decodeArena->packed(DecodeTensor::ProposedTokens, width),
         decodeArena->packed(DecodeTensor::Candidates, width),
         decodeArena->packed(DecodeTensor::ProposalProbs, width),
         decodeArena->packed(DecodeTensor::TargetTopIds, width),
         decodeArena->packed(DecodeTensor::TargetTopProbs, width),
         decodeArena->packed(DecodeTensor::SamplingUniforms, width),
         decodeArena->packed(DecodeTensor::OutputTokens, width),
         decodeArena->packed(DecodeTensor::RetainedCount, width),
         decodeArena->packed(DecodeTensor::NextAnchor, width),
         decodeArena->packed(DecodeTensor::AcceptedCount, width)},
        maximumRetained, std::span(policies).first(width),
        geometry.target.stopTokens[0], geometry.target.stopTokens[1]);
  }

  void encodeBatchEmbedding(CommandGraph &graph, DecodeTensor tokens,
                            DecodeTensor output, uint32_t lanes) {
    if (!lanes || lanes > kLaneCount)
      throw std::invalid_argument("invalid embedding batch width");
    const uint32_t rows = lanes * kDecodeRows;
    targetModel.addEmbedding(graph, decodeArena->packed(tokens, lanes),
                             decodeArena->packed(output, lanes), rows);
  }

  void encodeBatchVerifyInput(CommandGraph &graph, uint32_t lanes) {
    if (!lanes || lanes > kLaneCount)
      throw std::invalid_argument("invalid verify-input batch width");
    sampling.addVerifyInput(
        graph, decodeArena->packed(DecodeTensor::DraftInputTokens, lanes),
        decodeArena->packed(DecodeTensor::ProposedTokens, lanes),
        decodeArena->packed(DecodeTensor::InputTokens, lanes), lanes);
  }

  void encodeBatchGdnCommit(CommandGraph &graph,
                            std::span<Request *const> lanes) {
    if (lanes.empty() || lanes.size() > kLaneCount)
      throw std::invalid_argument("invalid GDN commit batch");
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    std::array<MetalBuffer, kLaneCount> currentStates;
    std::array<MetalBuffer, kLaneCount> nextStates;
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      Request *entry = lanes[std::min(lane, width - 1)];
      if (!entry)
        throw std::invalid_argument("empty GDN commit lane");
      const uint32_t active = states.metadata(entry->slot).activeParity;
      const auto &gdn = states.buffers(entry->slot).gdn;
      currentStates[lane] = gdn[active].stateBase;
      nextStates[lane] = gdn[active ^ 1].stateBase;
    }
    targetModel.addStateCommit(
        graph,
        {decodeArena->gdnStorage(DecodeTensor::VerifyPackedBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyMixedBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyDecayBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyBetaBase), currentStates,
         nextStates, decodeArena->packed(DecodeTensor::RetainedCount, width)},
        width);
  }

  // A stop token or the last budgeted token needs no target work of its own:
  // the next cycle would only echo it as output. Emitting it as soon as it is
  // selected saves that cycle; the engine is told it has no KV row.
  bool emitTerminalAnchor(Request &entry, ModelStepResult &result) const {
    const bool stop = isStopToken(geometry, *entry.pendingToken);
    if (!stop && entry.maxNewTokens - entry.generatedTokens != 1)
      return false;
    result.outputTokens.push_back(*entry.pendingToken);
    result.outputTokensWithoutKv = 1;
    result.finished = stop;
    ++entry.generatedTokens;
    return true;
  }

  std::vector<ModelStepResult> finalizeDecode(
      std::span<DecodeLaneResult> lanes, std::vector<ModelStepResult> results,
      std::span<const ModelBatchItem> items, const ops::Q4DispatchStats &stats,
      uint32_t planWidth, CommandTiming timing) {
    if (lanes.size() != items.size() || results.size() != items.size())
      throw std::logic_error("decode completion shape changed");

    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      if (!laneResult.verify)
        continue;
      auto d = [&](DecodeTensor tensor) {
        return decodeArena->get(lane, tensor);
      };
      const uint32_t generation =
          *contents<uint32_t>(d(DecodeTensor::Generation), "target generation");
      if (generation != geometry.target.stateLayout.layers)
        throw std::runtime_error("target verify resident grids did not finish");

      laneResult.retained = *contents<uint32_t>(d(DecodeTensor::RetainedCount),
                                                "GPU retained token count");
      laneResult.accepted = *contents<uint32_t>(d(DecodeTensor::AcceptedCount),
                                                "GPU accepted draft count");
      laneResult.nextAnchor =
          *contents<uint32_t>(d(DecodeTensor::NextAnchor), "GPU next anchor");
      if (!laneResult.retained || laneResult.retained > kDecodeRows)
        throw std::runtime_error("target policy produced invalid retention");
      if (laneResult.accepted > kDraftProposalTokens ||
          laneResult.nextAnchor >= geometry.target.vocabularySize) {
        throw std::runtime_error(
            "target policy selected an invalid next anchor");
      }
    }

    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      if (!laneResult.verify)
        continue;
      Request &entry = *laneResult.request;
      const uint32_t *targetTokens =
          contents<uint32_t>(decodeArena->get(lane, DecodeTensor::OutputTokens),
                             "target output tokens");
      std::vector<uint32_t> output;
      output.reserve(laneResult.retained);
      output.push_back(laneResult.currentAnchor);
      output.insert(output.end(), targetTokens,
                    targetTokens + (laneResult.retained - 1));

      states.swapParity(entry.slot);
      const uint64_t nextLength =
          items[lane].logicalPosition + laneResult.retained;
      const QwenLogicalLengths previous = states.metadata(entry.slot).lengths;
      states.updateLengths(
          entry.slot, advanceDraftContext(
                          previous, nextLength,
                          {static_cast<uint32_t>(items[lane].logicalPosition),
                           static_cast<uint32_t>(nextLength), 0, false}));
      entry.generatedTokens += laneResult.retained;
      entry.pendingToken = laneResult.nextAnchor;
      entry.maskWords.clear();
      entry.verifyMaskInFlight = false;
      entry.decodeStage = DecodeStage::Regular;
      ModelStepResult &result = results[lane];
      result = {entry.id,
                0,
                std::move(output),
                false,
                DecodeStage::Regular,
                kDraftProposalTokens,
                std::min(laneResult.accepted, laneResult.retained - 1)};
      if (entry.generatedTokens < entry.maxNewTokens)
        emitTerminalAnchor(entry, result);
    }

    counters.lastDecodeWidth = planWidth;
    counters.lastDecodeFusedOperations = stats.fusedSourceOperations;
    counters.lastDecodeM16Dispatches = stats.m16Dispatches;
    counters.lastDecodeM24Dispatches = stats.m24Dispatches;
    counters.lastDecodeM32Dispatches = stats.m32Dispatches;
    counters.lastDecodeGpuSeconds = timing.gpuSeconds;
    counters.totalDecodeGpuSeconds += timing.gpuSeconds;
    counters.lastDecodeWallSeconds = timing.wallSeconds;
    counters.totalDecodeWallSeconds += timing.wallSeconds;
    return results;
  }

  // A constrained DFlash cycle has one host dependency between three Metal
  // commands: draft proposals define the grammar simulation, while the target
  // forward is independent of the resulting mask.  This ticket keeps the
  // scheduler batch (and therefore its DecodeArena lanes) owned across that
  // dependency.  All state transitions run on the engine thread; completion
  // handlers only wake it, so they capture the wake hook and never the ticket.
  class ConstrainedDecodeTicket final : public ModelBatchTicket {
  public:
    ConstrainedDecodeTicket(Impl &impl, std::vector<DecodeLaneResult> lanes,
                            std::vector<ModelStepResult> results,
                            std::span<const ModelBatchItem> items,
                            const ops::Q4DispatchStats &stats,
                            uint32_t planWidth, CommandTiming priorTiming,
                            const CommandGraph &draft,
                            std::function<void()> completion)
        : impl_(impl), lanes_(std::move(lanes)), results_(std::move(results)),
          items_(items.begin(), items.end()), stats_(stats),
          planWidth_(planWidth), timing_(priorTiming),
          wake_(std::make_shared<std::function<void()>>(
              std::move(completion))) {
      submit(draft);
    }

    std::vector<ModelMaskRequest> takeMaskRequests() override {
      std::vector<ModelMaskRequest> requests;
      if (stage_ == Stage::Draft && command_.ready()) {
        addTiming(command_.wait());
        std::array<Request *, kLaneCount> entries{};
        for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
          DecodeLaneResult &laneResult = lanes_[lane];
          Request &entry = *laneResult.request;
          const uint32_t *proposed = contents<uint32_t>(
              impl_.decodeArena->get(lane, DecodeTensor::ProposedTokens),
              "constrained draft proposals");
          entry.maskWords.clear();
          entry.verifyMaskInFlight = true;

          const uint32_t remaining = entry.maxNewTokens - entry.generatedTokens;
          laneResult.currentAnchor = *entry.pendingToken;
          laneResult.maximumRetained = std::min(remaining, kDecodeRows);
          laneResult.verify = true;
          entries[lane] = &entry;

          if (!abandoned_[lane]) {
            ModelMaskRequest request;
            request.requestId = entry.id;
            request.simulationTokens.reserve(kDecodeRows);
            request.simulationTokens.push_back(*entry.pendingToken);
            request.simulationTokens.insert(request.simulationTokens.end(),
                                            proposed,
                                            proposed + kDraftProposalTokens);
            requests.push_back(std::move(request));
          }
        }

        CommandGraph target;
        const uint32_t width = static_cast<uint32_t>(lanes_.size());
        impl_.encodeBatchVerifyInput(target, width);
        impl_.encodeBatchEmbedding(target, DecodeTensor::InputTokens,
                                   DecodeTensor::Hidden0, width);
        impl_.encodeTargetVerifyBatchForward(
            target, {entries.data(), lanes_.size()}, items_, stats_);
        submit(target);
        stage_ = Stage::TargetForward;
      }

      if (stage_ == Stage::TargetForward && command_.ready()) {
        const CommandTiming forward = command_.wait();
        addTiming(forward);
        targetForwardGpuSeconds_ += forward.gpuSeconds;
        maskWaitStarted_ = std::chrono::steady_clock::now();
        stage_ = Stage::WaitingMask;
      }

      if (stage_ == Stage::WaitingMask) {
        bool masksReady = true;
        for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
          masksReady = masksReady && (abandoned_[lane] ||
                                      !lanes_[lane].request->maskWords.empty());
        }
        if (masksReady) {
          maskWaitSeconds_ +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            *maskWaitStarted_)
                  .count();
          maskWaitStarted_.reset();
          std::array<Request *, kLaneCount> entries{};
          std::array<uint32_t, kLaneCount> maximumRetained{};
          for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
            DecodeLaneResult &laneResult = lanes_[lane];
            Request &entry = *laneResult.request;
            entries[lane] = &entry;
            maximumRetained[lane] = laneResult.maximumRetained;
            impl_.loadPolicyBuffers(
                entry, lane,
                abandoned_[lane] ? std::span<const uint32_t>{}
                                 : std::span<const uint32_t>{entry.maskWords});
          }

          CommandGraph commit;
          impl_.encodeTargetVerifyBatchPolicy(commit,
                                              {entries.data(), lanes_.size()});
          impl_.encodeBatchAcceptance(commit, {entries.data(), lanes_.size()},
                                      {maximumRetained.data(), lanes_.size()});
          impl_.encodeBatchGdnCommit(commit, {entries.data(), lanes_.size()});
          impl_.encodeDraftStateCommitBatch(
              commit, {entries.data(), lanes_.size()}, items_, stats_);
          submit(commit);
          stage_ = Stage::Commit;
        }
      }
      return requests;
    }

    bool ownsMaskWait(uint64_t requestId) const noexcept override {
      if (stage_ == Stage::Draft || stage_ == Stage::Done)
        return false;
      return std::any_of(lanes_.begin(), lanes_.end(),
                         [requestId](const DecodeLaneResult &lane) {
                           return lane.request->id == requestId;
                         });
    }

    void abandonMask(uint64_t requestId) noexcept override {
      if (stage_ == Stage::Done)
        return;
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        if (lanes_[lane].request->id == requestId) {
          abandoned_[lane] = true;
          lanes_[lane].request->maskWords.clear();
          return;
        }
      }
    }

    bool ready() const noexcept override {
      return stage_ == Stage::Commit && command_.ready();
    }

    std::vector<ModelStepResult> wait() override {
      if (!ready())
        throw std::logic_error("constrained decode ticket is not complete");
      addTiming(command_.wait());
      stage_ = Stage::Done;
      ModelTelemetry &counters = impl_.counters;
      ++counters.constrainedMaskOverlapBatches;
      counters.constrainedMaskOverlapRequests += lanes_.size();
      counters.lastConstrainedTargetForwardGpuSeconds =
          targetForwardGpuSeconds_;
      counters.totalConstrainedTargetForwardGpuSeconds +=
          targetForwardGpuSeconds_;
      counters.lastConstrainedMaskWaitSeconds = maskWaitSeconds_;
      counters.totalConstrainedMaskWaitSeconds += maskWaitSeconds_;
      return impl_.finalizeDecode(lanes_, std::move(results_), items_, stats_,
                                  planWidth_, timing_);
    }

    double wallMilliseconds() const noexcept override {
      return timing_.wallSeconds * 1000.0;
    }

  private:
    enum class Stage : uint8_t {
      Draft,
      TargetForward,
      WaitingMask,
      Commit,
      Done
    };

    void submit(const CommandGraph &graph) {
      command_ = impl_.backend.submitCommandAsync(
          graph.dispatches(), [wake = wake_](uint64_t) {
            if (*wake)
              (*wake)();
          });
    }

    void addTiming(CommandTiming value) noexcept {
      timing_.gpuSeconds += value.gpuSeconds;
      timing_.wallSeconds += value.wallSeconds;
    }

    Impl &impl_;
    std::vector<DecodeLaneResult> lanes_;
    std::vector<ModelStepResult> results_;
    std::vector<ModelBatchItem> items_;
    ops::Q4DispatchStats stats_;
    uint32_t planWidth_ = 0;
    Stage stage_ = Stage::Draft;
    CommandTicket command_;
    CommandTiming timing_;
    std::array<bool, kLaneCount> abandoned_{};
    double targetForwardGpuSeconds_ = 0.0;
    double maskWaitSeconds_ = 0.0;
    std::optional<std::chrono::steady_clock::time_point> maskWaitStarted_;
    std::shared_ptr<std::function<void()>> wake_;
  };
};

Runtime::Runtime(RuntimeContext context)
    : impl_(std::make_unique<Impl>(context)) {}

Runtime::~Runtime() = default;

void Runtime::checkHealth() { impl_->backend.checkHealth(); }

bool Runtime::needsHealthCheck() const noexcept {
  return impl_->backend.needsHealthCheck();
}

void Runtime::beginColdRequest(const ModelRequest &request,
                               uint32_t stateSlot) {
  if (auto admission = beginAt(request, stateSlot); !admission) {
    throw metal::MetalAllocationError(
        std::string("unable to allocate sequence state cell: ") +
            metal::allocationFailureName(admission.failure), admission.failure);
  }
  try {
    setDraftContextPlan(
        request.id,
        planDraftContext(0, static_cast<uint32_t>(request.prompt.size()),
                         std::nullopt, {}));
  } catch (...) {
    end(request.id);
    throw;
  }
}

StateAdmission Runtime::begin(const ModelRequest &request) {
  Impl::ImageAdmission images(*impl_, request.id);
  StateAdmission admission = admitIdleSlot(impl_->states, [&](uint32_t slot) {
    if (auto imageAdmission = impl_->stageImages(request); !imageAdmission)
      return imageAdmission;
    return beginAt(request, slot);
  });
  images.committed = admission.granted();
  return admission;
}

void Runtime::suspend(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || entry.verifyMaskInFlight) {
    throw std::logic_error("Qwen request cannot be suspended");
  }
  impl_->states.releaseSlot(entry.slot, requestId);
  static_cast<void>(impl_->states.releaseIdle(0, 0));
  impl_->pageTableBindings[entry.slot] = {};
  entry.images.clear();
  entry.draftContextPlan.reset();
  entry.draftContextValid = false;
  entry.draftContextThrough = 0;
  entry.replayingGeneration |= entry.promptComplete;
  entry.promptComplete = false;
  entry.resident = false;
}

StateAdmission Runtime::resume(const ModelRequest &request) {
  Impl::Request &entry = impl_->request(request.id);
  if (entry.resident) {
    throw std::logic_error("Qwen request is not suspended");
  }
  if (request.prompt.size() < entry.promptTokens) {
    throw std::invalid_argument("recomputed history cannot shorten the prompt");
  }
  Impl::ImageAdmission images(*impl_, request.id);
  StateAdmission admission =
      admitIdleSlot(impl_->states, [&](uint32_t slot) {
        if (auto imageAdmission = impl_->stageImages(request); !imageAdmission)
          return imageAdmission;
        return impl_->states.tryActivateSlot(slot, request.id);
      });
  if (admission.granted()) {
    entry.slot = *admission.cell;
    entry.resident = true;
    entry.promptTokens = static_cast<uint32_t>(request.prompt.size());
    if (auto staged = impl_->stagedImages.find(request.id);
        staged != impl_->stagedImages.end()) {
      entry.images = std::move(staged->second);
      impl_->stagedImages.erase(staged);
    }
  }
  images.committed = admission.granted();
  return admission;
}

metal::AllocationResult Runtime::beginAt(const ModelRequest &request, uint32_t stateSlot) {
  if (!request.id || stateSlot >= kLaneCount || request.prompt.empty()) {
    throw std::invalid_argument("invalid executor request activation");
  }
  if (impl_->requests.contains(request.id)) {
    throw std::logic_error("request is already active");
  }
  Impl::Request entry;
  entry.id = request.id;
  entry.promptTokens = static_cast<uint32_t>(request.prompt.size());
  entry.maxNewTokens = request.maxNewTokens;
  entry.cohort = request.cohort;
  entry.sampling = request.sampling;
  entry.constraint = request.constraint;
  const BatchCohort expected =
      entry.constraint == ConstraintMode::TokenMask
          ? BatchCohort::Constrained
          : (Impl::samplingEnabled(entry) ? BatchCohort::Sampling
                                          : BatchCohort::Greedy);
  if (entry.cohort != expected || !std::isfinite(entry.sampling.temperature) ||
      entry.sampling.temperature < 0.0F ||
      !std::isfinite(entry.sampling.topP) || entry.sampling.topP <= 0.0F ||
      entry.sampling.topP > 1.0F ||
      entry.sampling.topK > ops::kTargetSamplingCandidates ||
      (Impl::samplingEnabled(entry) && !entry.sampling.topK)) {
    throw std::invalid_argument("request sampling/cohort contract is invalid");
  }
  if (!request.scoreTokens.empty()) {
    if (request.maxNewTokens != 0 ||
        request.constraint != ConstraintMode::None ||
        request.cohort != BatchCohort::Greedy || !request.images.empty() ||
        !request.imagePixels.empty() ||
        request.scoreTokens.size() < ExecutionLimits::minimumScoreOptions ||
        request.scoreTokens.size() > ExecutionLimits::maximumScoreOptions) {
      throw std::invalid_argument("invalid score request");
    }
    std::vector<uint32_t> distinct(request.scoreTokens.begin(),
                                   request.scoreTokens.end());
    std::sort(distinct.begin(), distinct.end());
    if (std::adjacent_find(distinct.begin(), distinct.end()) !=
            distinct.end() ||
        std::any_of(distinct.begin(), distinct.end(), [&](uint32_t token) {
          return token >= impl_->geometry.target.vocabularySize;
        })) {
      throw std::invalid_argument("score token is out of vocabulary");
    }
    entry.scoreTokens.assign(request.scoreTokens.begin(),
                             request.scoreTokens.end());
  }
  entry.decodeStage = entry.cohort == BatchCohort::Constrained
                          ? DecodeStage::RequestInitialMask
                          : DecodeStage::Regular;
  if (auto admission = impl_->states.tryActivateSlot(stateSlot, request.id);
      !admission)
    return admission;
  entry.slot = stateSlot;
  entry.resident = true;
  if (auto staged = impl_->stagedImages.find(request.id);
      staged != impl_->stagedImages.end()) {
    entry.images = std::move(staged->second);
    impl_->stagedImages.erase(staged);
  }
  try {
    auto [_, inserted] = impl_->requests.emplace(request.id, std::move(entry));
    if (!inserted) {
      throw std::logic_error("request insertion lost uniqueness");
    }
  } catch (...) {
    impl_->states.releaseSlot(stateSlot, request.id);
    throw;
  }
  return true;
}

void Runtime::restore(uint64_t requestId, uint32_t restoredPrefixLength,
                      std::shared_ptr<const CompositeState> restoredState,
                      bool restoreDraftState) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || !restoredState) {
    throw std::invalid_argument("cannot restore a nonresident request");
  }
  if (restoredPrefixLength >= entry.promptTokens) {
    throw std::invalid_argument(
        "reusable Qwen prefix must leave an input token to replay");
  }
  impl_->states.restore(entry.slot, *restoredState, restoreDraftState);
  if (!restoreDraftState)
    ++impl_->counters.draftStateRestoreSkipped;
  const QwenLogicalLengths &lengths =
      impl_->states.metadata(entry.slot).lengths;
  if (lengths.targetTokens != restoredPrefixLength ||
      (restoreDraftState &&
       !lengths.hasCompleteDraftWindow(kDraftCacheStride)) ||
      (!restoreDraftState && lengths.draftLength != 0)) {
    throw std::invalid_argument("prefix logical length does not match state");
  }
  // Images fully inside the restored prefix are never encoded; their spans
  // stay because rotary positions after them depend on their grids.
  for (Impl::ImageState &image : entry.images) {
    if (image.span.end() <= restoredPrefixLength) {
      image.data.reset();
    }
  }
  entry.promptComplete = false;
  if (!entry.replayingGeneration) {
    entry.finalTargetHidden.clear();
    entry.pendingToken.reset();
  }
  entry.draftContextValid = restoreDraftState;
  entry.draftContextThrough = restoreDraftState ? restoredPrefixLength : 0;
  entry.draftContextPlan.reset();
}

void Runtime::setDraftContextPlan(uint64_t requestId, DraftContextPlan plan) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || plan.replayEnd != entry.promptTokens) {
    throw std::invalid_argument("draft context plan does not match request");
  }
  const uint64_t current =
      impl_->states.metadata(entry.slot).lengths.targetTokens;
  if (plan.replayBegin != current ||
      plan.restoredDraftBoundary !=
          (current ? std::optional<uint32_t>(static_cast<uint32_t>(current))
                   : std::nullopt)) {
    throw std::invalid_argument("draft context plan restore boundary is stale");
  }
  entry.draftContextPlan = std::move(plan);
}

std::vector<ModelStepResult>
Runtime::prefill(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
  return prefillAsync(plan, items, {})->wait();
}

std::unique_ptr<ModelBatchTicket>
Runtime::submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
                std::function<void()> completion) {
  switch (plan.kind) {
  case WorkKind::Prefill:
    return prefillAsync(plan, items, std::move(completion));
  case WorkKind::Decode:
    return decodeAsync(plan, items, std::move(completion));
  }
  throw std::logic_error("unknown model work kind");
}

std::unique_ptr<ModelBatchTicket>
Runtime::prefillAsync(const BatchPlan &plan,
                      std::span<const ModelBatchItem> items,
                      std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Prefill);
  if (plan.decodeStage != DecodeStage::Regular) {
    throw std::invalid_argument("Qwen prefill cannot resume a mask plan");
  }

  std::array<Impl::Request *, kLaneCount> entries{};
  CommandGraph graph;
  impl_->encodePackedPrefillGraph(graph, items, entries);
  const bool encodesImages = std::any_of(
      entries.begin(), entries.begin() + items.size(), [](const auto *entry) {
        return std::any_of(entry->images.begin(), entry->images.end(),
                           [](const auto &image) {
                             return image.data && image.data->encoding;
                           });
      });
  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  auto notify = [completion = std::move(completion)](uint64_t) {
    if (completion)
      completion();
  };
  CommandTicket command =
      impl_->backend.submitCommandAsync(graph.dispatches(), std::move(notify));
  Impl *impl = impl_.get();
  auto finish = [impl, entries,
                 items = std::move(copiedItems)](CommandTiming timing) mutable {
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      for (Impl::ImageState &image : entries[lane]->images) {
        if (!image.data || !image.data->encoding)
          continue;
        image.data->encoding = false;
        image.data->encoded = true;
        image.data->pixels = MetalBuffer{};
      }
    }

    std::vector<ModelStepResult> results;
    results.reserve(items.size());
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      Impl::Request &entry = *entries[lane];
      const ModelBatchItem &item = items[lane];
      const auto captures = Impl::activeDraftCaptures(entry, item);
      const uint32_t capturedRows = Impl::captureRows(captures);
      uint32_t activeRows = 0;
      uint32_t materializationRows = 0;
      for (const auto &capture : captures) {
        activeRows += capture.activeRows;
        materializationRows += capture.materializationRows;
      }
      if (activeRows + materializationRows != capturedRows) {
        throw std::logic_error("draft capture telemetry is inconsistent");
      }
      impl->counters.targetPrefillRows += item.tokenCount;
      impl->counters.draftContextRowsActive += activeRows;
      impl->counters.draftContextRowsMaterialization += materializationRows;
      impl->counters.draftContextRowsAvoided += item.tokenCount - capturedRows;
      for (const auto &capture : captures) {
        const bool continues =
            !capture.resetDraftState && entry.draftContextValid &&
            entry.draftContextThrough == capture.absoluteBegin;
        if (!continues) {
          ++impl->counters.draftStateResets;
        }
        entry.draftContextValid = true;
        entry.draftContextThrough = capture.absoluteEnd;
      }
      impl->states.swapParity(entry.slot);
      uint64_t nextLength = item.logicalPosition + item.tokenCount;
      QwenLogicalLengths lengths = impl->states.metadata(entry.slot).lengths;
      lengths.targetTokens = nextLength;
      for (const auto &capture : captures) {
        lengths = Impl::advanceDraftContext(lengths, nextLength, capture);
      }
      impl->states.updateLengths(entry.slot, lengths);
      entry.promptComplete = nextLength == entry.promptTokens;
      ModelStepResult result{entry.id, item.tokenCount, {}, false,
                             entry.decodeStage, 0, 0};
      if (entry.promptComplete && !entry.replayingGeneration) {
        entry.pendingToken.reset();
        if (!entry.scoreTokens.empty()) {
          // Score-only: read the raw bf16 logits at the final prompt position
          // (row lastRows-1 of the gathered head input) in requested order.
          const uint32_t lastRows = std::min(item.tokenCount, kDecodeRows);
          const uint16_t *logits = contents<uint16_t>(
              impl->decodeArena->get(lane, DecodeTensor::Logits),
              "score logits");
          const uint16_t *row =
              logits + uint64_t{lastRows - 1} *
                           impl->geometry.target.vocabularySize;
          result.scoreLogits.reserve(entry.scoreTokens.size());
          for (uint32_t token : entry.scoreTokens) {
            const float logit =
                std::bit_cast<float>(uint32_t{row[token]} << 16);
            if (!std::isfinite(logit)) {
              // A numerical outcome for this request, not a broken invariant:
              // report it as a lane failure so the engine drops this request
              // before cache publication or output and the batch survives.
              result.scoreLogits.clear();
              result.failure = "score logit is not finite";
              break;
            }
            result.scoreLogits.push_back(logit);
          }
          result.finished = true;
        } else if (entry.constraint == ConstraintMode::None) {
          entry.pendingToken = *contents<uint32_t>(
              impl->decodeArena->get(lane, DecodeTensor::OutputTokens),
              "prefill next token");
          if (!entry.pendingToken ||
              *entry.pendingToken >=
                  impl->geometry.target.vocabularySize) {
            throw std::runtime_error(
                "prefill policy selected an invalid token");
          }
          impl->emitTerminalAnchor(entry, result);
        } else {
          const uint32_t lastRows = std::min(item.tokenCount, kDecodeRows);
          impl->captureFinalHidden(
              entry, impl->decodeArena->get(lane, DecodeTensor::Hidden0),
              lastRows - 1);
        }
      }
      if (entry.promptComplete)
        entry.replayingGeneration = false;
      results.push_back(std::move(result));
    }
    impl->counters.lastPrefillWallSeconds = timing.wallSeconds;
    impl->counters.totalPrefillWallSeconds += timing.wallSeconds;
    impl->counters.lastPrefillGpuSeconds = timing.gpuSeconds;
    impl->counters.totalPrefillGpuSeconds += timing.gpuSeconds;
    return results;
  };
  return std::make_unique<DeferredMetalTicket>(std::move(command),
                                               std::move(finish), 0.0,
                                               !encodesImages);
}

std::vector<ModelStepResult>
Runtime::decode(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
  return decodeAsync(plan, items, {})->wait();
}

std::unique_ptr<ModelBatchTicket>
Runtime::decodeAsync(const BatchPlan &plan,
                     std::span<const ModelBatchItem> items,
                     std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Decode);
  const bool constrained = plan.cohort == BatchCohort::Constrained;
  if (plan.decodeStage != DecodeStage::Regular && !constrained) {
    throw std::invalid_argument(
        "only constrained decode uses a specialized decode stage");
  }

  std::vector<Impl::DecodeLaneResult> lanes(items.size());
  std::vector<ModelStepResult> results(items.size());
  ops::Q4DispatchStats batchStats;
  CommandTiming priorTiming;
  for (uint32_t lane = 0; lane < items.size(); ++lane) {
    const ModelBatchItem &item = items[lane];
    Impl::Request &entry = impl_->request(item.requestId);
    if (entry.cohort != plan.cohort) {
      throw std::invalid_argument("request does not belong to batch cohort");
    }
    if (entry.decodeStage != plan.decodeStage) {
      throw std::logic_error("request decode stage does not match decode plan");
    }
    Impl::DecodeLaneResult &laneResult = lanes[lane];
    laneResult.request = &entry;
    results[lane].requestId = entry.id;

    if (constrained && plan.decodeStage == DecodeStage::RequestInitialMask) {
      if (entry.pendingToken || !entry.maskWords.empty()) {
        throw std::logic_error("initial mask request has stale decode state");
      }
      entry.decodeStage = DecodeStage::ApplyInitialMask;
      results[lane].nextDecodeStage = DecodeStage::ApplyInitialMask;
      continue;
    }

    if (constrained && plan.decodeStage == DecodeStage::ApplyInitialMask) {
      if (entry.maskWords.size() != impl_->geometry.maskWords() ||
          entry.pendingToken) {
        throw std::logic_error("initial anchor mask state is invalid");
      }
      const CommandTiming selection =
          impl_->selectPendingFromFinalHidden(entry, lane, entry.maskWords);
      priorTiming.gpuSeconds += selection.gpuSeconds;
      priorTiming.wallSeconds += selection.wallSeconds;
      entry.maskWords.clear();
      entry.decodeStage = DecodeStage::Regular;
      results[lane].nextDecodeStage = DecodeStage::Regular;
      if (impl_->emitTerminalAnchor(entry, results[lane]))
        continue;
    }

    if (!entry.pendingToken)
      throw std::logic_error("decode request has no current anchor");
    const uint32_t remaining = entry.maxNewTokens - entry.generatedTokens;
    if (!remaining)
      throw std::logic_error("completed request was decoded");
    if (isStopToken(impl_->geometry, *entry.pendingToken) || remaining == 1) {
      throw std::logic_error("terminal anchor was not emitted on selection");
    }

    if (constrained) {
      if (!entry.maskWords.empty()) {
        throw std::logic_error("constrained request has stale mask state");
      }
      laneResult.draftForMask = true;
      impl_->prepareDecodeLane(entry, item, lane);
      if (Impl::samplingEnabled(entry))
        Impl::stageSamplingCycle(entry);
      impl_->loadPolicyBuffers(entry, lane, {});
      laneResult.draftComputed = true;
      continue;
    }

    if (Impl::samplingEnabled(entry))
      Impl::stageSamplingCycle(entry);

    // DFlash has one physical graph: anchor + seven proposal rows. A shorter
    // output budget only lowers the token-exact commit count; it never
    // changes the Metal graph shape.
    laneResult.currentAnchor = *entry.pendingToken;
    laneResult.maximumRetained = std::min(remaining, kDecodeRows);

    impl_->prepareDecodeLane(entry, item, lane);
    impl_->loadPolicyBuffers(entry, lane, {});
    laneResult.draftComputed = true;
    laneResult.verify = true;
  }

  CommandGraph commandGraph;
  uint32_t verified = 0;
  uint32_t draftComputed = 0;
  for (const Impl::DecodeLaneResult &lane : lanes)
    verified += lane.verify ? 1U : 0U;
  for (const Impl::DecodeLaneResult &lane : lanes)
    draftComputed += lane.draftComputed ? 1U : 0U;
  if (verified && verified != lanes.size()) {
    throw std::logic_error("decode batch mixed mask and verify phases");
  }
  const uint32_t width = static_cast<uint32_t>(lanes.size());
  if (draftComputed && draftComputed != lanes.size()) {
    throw std::logic_error("decode batch mixed draft execution phases");
  }
  if (draftComputed || verified) {
    const uint32_t ropeRows = width * kDecodeRows;
    impl_->addRopeTables(
        commandGraph,
        impl_->decodeArena->packed(DecodeTensor::Positions, width), ropeRows,
        impl_->decodeArena->packed(DecodeTensor::DraftPositions, width),
        ropeRows, impl_->decodeArena->packed(DecodeTensor::RopeCos, width),
        impl_->decodeArena->packed(DecodeTensor::RopeSin, width),
        impl_->decodeArena->packed(DecodeTensor::DraftRopeCos, width),
        impl_->decodeArena->packed(DecodeTensor::DraftRopeSin, width));
  }
  if (draftComputed) {
    std::array<Impl::Request *, kLaneCount> requests{};
    std::array<uint64_t, kLaneCount> logicalPositions{};
    for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
      requests[lane] = lanes[lane].request;
      logicalPositions[lane] = items[lane].logicalPosition;
    }
    impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::DraftInputTokens,
                                DecodeTensor::DraftHidden0, width);
    impl_->encodeDraftBatchGraph(commandGraph, {requests.data(), lanes.size()},
                                 {logicalPositions.data(), lanes.size()},
                                 batchStats);
  }
  if (verified) {
    impl_->encodeBatchVerifyInput(commandGraph, width);
    impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::InputTokens,
                                DecodeTensor::Hidden0, width);
    std::array<Impl::Request *, kLaneCount> requests{};
    std::array<uint32_t, kLaneCount> maximumRetained{};
    for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
      requests[lane] = lanes[lane].request;
      maximumRetained[lane] = lanes[lane].maximumRetained;
    }
    impl_->encodeTargetVerifyBatchForward(
        commandGraph, {requests.data(), lanes.size()}, items, batchStats);
    impl_->encodeTargetVerifyBatchPolicy(commandGraph,
                                         {requests.data(), lanes.size()});
    impl_->encodeBatchAcceptance(commandGraph, {requests.data(), lanes.size()},
                                 {maximumRetained.data(), lanes.size()});
    impl_->encodeBatchGdnCommit(commandGraph, {requests.data(), lanes.size()});
    impl_->encodeDraftStateCommitBatch(
        commandGraph, {requests.data(), lanes.size()}, items, batchStats);
  }

  const bool overlapConstraintMask =
      constrained && !lanes.empty() &&
      std::all_of(lanes.begin(), lanes.end(),
                  [](const auto &lane) { return lane.draftForMask; });
  if (overlapConstraintMask) {
    return std::make_unique<Impl::ConstrainedDecodeTicket>(
        *impl_, std::move(lanes), std::move(results), items, batchStats,
        plan.width(), priorTiming, commandGraph, std::move(completion));
  }

  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  const uint32_t planWidth = plan.width();
  Impl *impl = impl_.get();
  auto finish = [impl, lanes = std::move(lanes), results = std::move(results),
                 items = std::move(copiedItems), batchStats, planWidth,
                 priorTiming](CommandTiming timing) mutable {
    timing.gpuSeconds += priorTiming.gpuSeconds;
    timing.wallSeconds += priorTiming.wallSeconds;
    return impl->finalizeDecode(lanes, std::move(results), items, batchStats,
                                planWidth, timing);
  };

  if (commandGraph.empty()) {
    std::vector<ModelStepResult> ready = finish(CommandTiming{});
    return std::make_unique<ReadyModelTicket>(std::move(ready),
                                              priorTiming.wallSeconds * 1000.0);
  }

  auto notify = [completion = std::move(completion)](uint64_t) {
    if (completion)
      completion();
  };
  CommandTicket command = impl_->backend.submitCommandAsync(
      commandGraph.dispatches(), std::move(notify));
  return std::make_unique<DeferredMetalTicket>(
      std::move(command), std::move(finish), priorTiming.wallSeconds * 1000.0);
}

std::shared_ptr<const CompositeState> Runtime::snapshot(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident)
    throw std::logic_error("request is not resident");
  const QwenSlotMetadata &metadata = impl_->states.metadata(entry.slot);
  if (!metadata.lengths.hasCompleteDraftWindow(kDraftCacheStride) ||
      metadata.lengths.targetTokens % kv::kPageTokens) {
    throw std::logic_error("cannot snapshot uncommitted draft state");
  }
  return impl_->states.snapshot(entry.slot);
}

uint64_t Runtime::reclaimIdleState() noexcept {
  // One idle buffer per call, so a denied allocation frees only what it
  // needs; rebuildable caches go once the pool is empty.
  const uint32_t cells = impl_->states.idleCells();
  const uint32_t rings = impl_->states.idleRings();
  if (cells)
    return impl_->states.releaseIdle(cells - 1, rings);
  if (rings)
    return impl_->states.releaseIdle(0, rings - 1);
  uint64_t released = 0;
  released += impl_->dropEmbeddingCache();
  if (impl_->vision && impl_->visionIdle()) {
    released += impl_->vision->arenaBytes();
    impl_->vision.reset();
  }
  return released;
}

void Runtime::provideMask(uint64_t requestId, std::span<const uint32_t> words) {
  Impl::Request &entry = impl_->request(requestId);
  const bool acceptsMask =
      waitsForMask(entry.decodeStage) || entry.verifyMaskInFlight;
  // Initial-mask replies can race resource preemption. They belong to the
  // host continuation, not the released device state.
  if (entry.constraint != ConstraintMode::TokenMask || !acceptsMask ||
      !entry.maskWords.empty()) {
    throw std::logic_error("request is not waiting for a token mask");
  }
  const uint32_t maskWords = impl_->geometry.maskWords();
  uint64_t expected = entry.verifyMaskInFlight
                          ? uint64_t{kDecodeRows + 1} * maskWords
                          : maskWords;
  if (words.size() != expected) {
    throw std::invalid_argument("token mask has the wrong word count");
  }
  const uint32_t rows = static_cast<uint32_t>(words.size() / maskWords);
  for (uint32_t row = 0; row < rows; ++row) {
    auto begin = words.begin() + uint64_t{row} * maskWords;
    if (std::none_of(begin, begin + maskWords,
                     [](uint32_t word) { return word != 0; })) {
      throw std::invalid_argument("token mask row permits no vocabulary token");
    }
  }
  if (entry.verifyMaskInFlight) {
    if (!entry.pendingToken || (words[*entry.pendingToken / 32] &
                                (1U << (*entry.pendingToken % 32))) == 0) {
      throw std::invalid_argument(
          "verify mask is not synchronized to the pending anchor");
    }
  }
  entry.maskWords.assign(words.begin(), words.end());
}

void Runtime::end(uint64_t requestId) {
  impl_->stagedImages.erase(requestId);
  auto found = impl_->requests.find(requestId);
  if (found == impl_->requests.end())
    return;
  for (const Impl::ImageState &image : found->second.images)
    impl_->retainEmbeddings(image);
  if (found->second.resident) {
    impl_->states.releaseSlot(found->second.slot, requestId);
  }
  impl_->requests.erase(found);
}

namespace {

WarmupStepResult warmupResult(uint64_t estimatedPeakBytes, double wallSeconds,
                              std::string detail) {
  if (!estimatedPeakBytes) {
    throw std::logic_error("warmup peak estimate must be nonzero");
  }
  if (!(wallSeconds > 0.0) || !std::isfinite(wallSeconds)) {
    throw std::logic_error("warmup wall time must be finite and positive");
  }
  return {true, estimatedPeakBytes, std::move(detail), wallSeconds, {}};
}

std::vector<uint32_t> warmupPages(kv::Q8PageStorage &storage, uint32_t first,
                                  uint32_t count) {
  if (!count || uint64_t{first} + count > storage.pageCount()) {
    throw std::invalid_argument("warmup Q8 page range is unavailable");
  }
  std::vector<uint32_t> result(count);
  for (uint32_t index = 0; index < count; ++index) {
    const uint32_t page = first + index;
    if (auto admission = storage.ensureResident(page); !admission) {
      throw metal::MetalAllocationError(
          std::string("warmup could not reserve Q8 page backing: ") +
              metal::allocationFailureName(admission.failure), admission.failure);
    }
    result[index] = page;
  }
  return result;
}

} // namespace

void Runtime::prepareWarmupDecode(uint64_t requestId, uint32_t anchor) {
  // Teacher-force a valid input so EOS selected by synthetic prefill cannot
  // prevent the warmup from exercising the real draft/verify/commit graph.
  while (anchor < impl_->geometry.target.vocabularySize &&
         isStopToken(impl_->geometry, anchor))
    ++anchor;
  if (anchor >= impl_->geometry.target.vocabularySize)
    throw std::logic_error("decode warmup has no non-terminal input token");
  auto &entry = impl_->request(requestId);
  entry.pendingToken = anchor;
  entry.generatedTokens = 0;
}

WarmupStepResult Runtime::warmupPrefill(uint32_t rows) {
  using Clock = std::chrono::steady_clock;
  if (!rows || rows > kPrefillRows)
    throw std::invalid_argument("invalid prefill warmup row count");
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 100;
  double wallSeconds = 0.0;
  std::vector<WarmupLaneResult> lanes;
  std::vector<uint32_t> warmupPrompt(rows, 0);
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 16;
  beginColdRequest(request, 0);
  try {
    std::vector<uint32_t> pages = warmupPages(
        impl_->kvPages, 0, (rows + kv::kPageTokens - 1) / kv::kPageTokens);
    BatchPlan plan{WorkKind::Prefill,
                   BatchCohort::Greedy,
                   {{id, rows}},
                   DecodeStage::Regular};
    ModelBatchItem item{id, 0, 0, 0, rows, pages};
    item.inputTokens = request.prompt;
    const auto phaseStart = Clock::now();
    auto result = prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    wallSeconds = std::chrono::duration<double>(Clock::now() - phaseStart).count();
    if (result.size() != 1 || result[0].consumedPromptTokens != rows) {
      throw std::runtime_error("prefill warmup result mismatch");
    }
    lanes.push_back({std::move(result[0]), impl_->request(id).pendingToken,
                     impl_->states.metadata(0).lengths.targetTokens});
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  auto result = warmupResult(impl_->estimatedWarmupPeak(), wallSeconds,
                            "real " + std::to_string(rows) +
                                "-row packed Q8 target+draft prefill [M32]");
  result.lanes = std::move(lanes);
  return result;
}

WarmupStepResult Runtime::warmupDecodeBatch(uint32_t width) {
  using Clock = std::chrono::steady_clock;
  if (!width || width > kLaneCount) {
    throw std::invalid_argument("invalid decode warmup width");
  }
  constexpr uint64_t firstId = std::numeric_limits<uint64_t>::max() - 110;
  // Plan order is deliberately unrelated to physical slot order. DecodeArena
  // lanes belong to the explicit BatchPlan, while recurrent/KV state remains
  // addressed by each item.stateSlot; batching must never assume slot 0..3.
  constexpr std::array<uint32_t, kLaneCount> slotOrder{2, 0, 3, 1};
  double wallSeconds = 0.0;
  std::vector<WarmupLaneResult> lanes;
  std::array<std::vector<uint32_t>, kLaneCount> pages;
  try {
    for (uint32_t lane = 0; lane < width; ++lane) {
      std::vector<uint32_t> warmupPrompt{lane};
      ModelRequest request;
      request.id = firstId + lane;
      request.prompt = warmupPrompt;
      request.maxNewTokens = 16;
      beginColdRequest(request, slotOrder[lane]);
      pages[lane] = warmupPages(impl_->kvPages, 5 + lane, 1);
      BatchPlan prefillPlan{WorkKind::Prefill,
                            BatchCohort::Greedy,
                            {{request.id, 1}},
                            DecodeStage::Regular};
      ModelBatchItem item{request.id, slotOrder[lane], 0, 0, 1, pages[lane]};
      item.inputTokens = request.prompt;
      static_cast<void>(
          prefill(prefillPlan, std::span<const ModelBatchItem>(&item, 1)));
      prepareWarmupDecode(request.id, warmupPrompt.back());
    }
    BatchPlan plan;
    plan.kind = WorkKind::Decode;
    plan.cohort = BatchCohort::Greedy;
    std::vector<ModelBatchItem> items;
    for (uint32_t lane = 0; lane < width; ++lane) {
      plan.items.push_back({firstId + lane, 0});
      items.push_back({firstId + lane, slotOrder[lane], 1, 0, 0, pages[lane]});
    }
    const auto phaseStart = Clock::now();
    auto decoded = decode(plan, items);
    wallSeconds = std::chrono::duration<double>(Clock::now() - phaseStart).count();
    bool committedEveryLane = decoded.size() == width;
    for (uint32_t lane = 0; committedEveryLane && lane < width; ++lane) {
      const auto &lengths = impl_->states.metadata(slotOrder[lane]).lengths;
      committedEveryLane = !decoded[lane].outputTokens.empty() &&
                           lengths.targetTokens > 1 &&
                           lengths.targetTokens ==
                               1 + decoded[lane].outputTokens.size() -
                                   decoded[lane].outputTokensWithoutKv &&
                           lengths.hasCompleteDraftWindow(kDraftCacheStride);
    }
    const bool fusedWidth =
        width == 1 ||
        (width == 2 && impl_->counters.lastDecodeFusedOperations &&
         impl_->counters.lastDecodeM16Dispatches) ||
        (width == 3 && impl_->counters.lastDecodeFusedOperations &&
         impl_->counters.lastDecodeM24Dispatches) ||
        (width == 4 && impl_->counters.lastDecodeFusedOperations &&
         impl_->counters.lastDecodeM32Dispatches);
    const bool fusedMaximum =
        width != kLaneCount || (impl_->counters.lastDecodeM32Dispatches > 0 &&
                                impl_->counters.lastDecodeM16Dispatches == 0);
    if (!committedEveryLane || !fusedWidth || !fusedMaximum ||
        impl_->counters.lastDecodeWidth != width) {
      throw std::runtime_error(
          "decode warmup B" + std::to_string(width) +
          " mismatch [committed=" + std::to_string(committedEveryLane) +
          ",fused=" + std::to_string(fusedWidth) +
          ",maximum=" + std::to_string(fusedMaximum) +
          ",m16=" + std::to_string(impl_->counters.lastDecodeM16Dispatches) +
          ",m24=" + std::to_string(impl_->counters.lastDecodeM24Dispatches) +
          ",m32=" + std::to_string(impl_->counters.lastDecodeM32Dispatches) +
          "]");
    }
    for (uint32_t lane = 0; lane < width; ++lane) {
      lanes.push_back({std::move(decoded[lane]),
                       impl_->request(firstId + lane).pendingToken,
                       impl_->states.metadata(slotOrder[lane]).lengths.targetTokens});
      end(firstId + lane);
    }
  } catch (...) {
    for (uint32_t lane = 0; lane < width; ++lane)
      end(firstId + lane);
    throw;
  }
  auto result = warmupResult(impl_->estimatedWarmupPeak(), wallSeconds,
                            "real B" + std::to_string(width) +
                                " draft/verify/commit decode");
  result.lanes = std::move(lanes);
  return result;
}

WarmupStepResult Runtime::warmupDraftVerifyCommit() {
  // Verify that a further commit preserves equal target and draft lengths.
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 120;
  double wallSeconds = 0.0;
  std::vector<uint32_t> warmupPrompt{1};
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 16;
  beginColdRequest(request, 0);
  try {
    std::vector<uint32_t> pages = warmupPages(impl_->kvPages, 9, 1);
    BatchPlan prefillPlan{WorkKind::Prefill,
                          BatchCohort::Greedy,
                          {{id, 1}},
                          DecodeStage::Regular};
    ModelBatchItem prefillItem{id, 0, 0, 0, 1, pages};
    prefillItem.inputTokens = request.prompt;
    static_cast<void>(
        prefill(prefillPlan, std::span<const ModelBatchItem>(&prefillItem, 1)));
    prepareWarmupDecode(id, warmupPrompt.back());
    BatchPlan decodePlan{
        WorkKind::Decode, BatchCohort::Greedy, {{id, 0}}, DecodeStage::Regular};
    ModelBatchItem decodeItem{id, 0, 1, 0, 0, pages};
    auto result =
        decode(decodePlan, std::span<const ModelBatchItem>(&decodeItem, 1));
    const auto &lengths = impl_->states.metadata(0).lengths;
    if (result.size() != 1 || result[0].outputTokens.empty() ||
        !lengths.hasCompleteDraftWindow(kDraftCacheStride) ||
        lengths.targetTokens <= 1 ||
        lengths.targetTokens != 1 + result[0].outputTokens.size() -
                                    result[0].outputTokensWithoutKv) {
      throw std::runtime_error("draft/target commit length mismatch");
    }
    wallSeconds = impl_->counters.lastDecodeWallSeconds;
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  return warmupResult(impl_->estimatedWarmupPeak(), wallSeconds,
                      "real draft verify acceptance and exact commit");
}

WarmupStepResult Runtime::warmupCompositeStateRestore() {
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 121;
  constexpr uint32_t prefixTokens = 2 * kv::kPageTokens;
  constexpr uint32_t suffixTokens = kDecodeRows;
  constexpr uint32_t promptTokens = prefixTokens + suffixTokens;
  std::vector<uint32_t> warmupPrompt(promptTokens, 2);
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 8;
  std::shared_ptr<const CompositeState> cachedState;
  uint64_t estimatedPeakBytes = impl_->estimatedWarmupPeak();
  double wallSeconds = 0.0;
  beginColdRequest(request, 0);
  try {
    if (impl_->kvPages.pageCount() <= 12) {
      throw std::runtime_error(
          "historical prefix warmup requires at least 13 Q8 pages");
    }
    // Deliberately non-contiguous physical ids exercise page-table lookup.
    const std::vector<uint32_t> pages{12, 10, 11};
    BatchPlan plan{WorkKind::Prefill,
                   BatchCohort::Greedy,
                   {{id, prefixTokens}},
                   DecodeStage::Regular};
    ModelBatchItem item{id, 0, 0, 0, prefixTokens, pages};
    item.inputTokens =
        std::span<const uint32_t>(request.prompt).first(prefixTokens);
    static_cast<void>(prefill(plan, std::span<const ModelBatchItem>(&item, 1)));
    wallSeconds = impl_->counters.lastPrefillWallSeconds;
    cachedState = snapshot(id);
    if (!cachedState)
      throw metal::MetalAllocationError("prefix warmup state allocation failed");
    // The snapshot remains live across restore; its cache slot is allocated
    // through the state storage, so the state term of the estimate already
    // covers it.
    estimatedPeakBytes = impl_->estimatedWarmupPeak();
    end(id);
    beginColdRequest(request, 1);
    restore(id, prefixTokens, cachedState, true);
    setDraftContextPlan(
        id, planDraftContext(prefixTokens, promptTokens, prefixTokens, {}));
    const auto &restored = impl_->states.metadata(1).lengths;
    if (restored.targetTokens != prefixTokens ||
        !restored.hasCompleteDraftWindow(kDraftCacheStride)) {
      throw std::runtime_error("prefix restore length mismatch");
    }

    // Continue from committed Q8 history. This M8 command teacher-forces a
    // new chunk, then the real speculative cycle overwrites its speculative
    // page suffix and advances only the accepted commit length.
    BatchPlan suffixPlan{WorkKind::Prefill,
                         BatchCohort::Greedy,
                         {{id, suffixTokens}},
                         DecodeStage::Regular};
    ModelBatchItem suffix{id,           1,    prefixTokens, prefixTokens,
                          suffixTokens, pages};
    suffix.inputTokens = std::span<const uint32_t>(request.prompt)
                             .subspan(prefixTokens, suffixTokens);
    static_cast<void>(
        prefill(suffixPlan, std::span<const ModelBatchItem>(&suffix, 1)));
    prepareWarmupDecode(id, warmupPrompt.back());
    const double continuationWallSeconds =
        impl_->counters.lastPrefillWallSeconds;
    wallSeconds += continuationWallSeconds;
    BatchPlan decodePlan{
        WorkKind::Decode, BatchCohort::Greedy, {{id, 0}}, DecodeStage::Regular};
    ModelBatchItem decodeItem{id, 1, promptTokens, 0, 0, pages};
    auto decoded =
        decode(decodePlan, std::span<const ModelBatchItem>(&decodeItem, 1));
    const double historicalDecodeWallSeconds =
        impl_->counters.lastDecodeWallSeconds;
    wallSeconds += historicalDecodeWallSeconds;
    const auto &continued = impl_->states.metadata(1).lengths;
    if (decoded.size() != 1 || decoded[0].outputTokens.empty() ||
        !continued.hasCompleteDraftWindow(kDraftCacheStride) ||
        continued.targetTokens <= promptTokens ||
        continued.targetTokens !=
            promptTokens + decoded[0].outputTokens.size() -
                decoded[0].outputTokensWithoutKv) {
      throw std::runtime_error(
          "restored historical prefix did not continue exactly");
    }
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  return warmupResult(
      estimatedPeakBytes, wallSeconds,
      "real direct-Q8 state restore, arbitrary page table, slot move, "
      "bounded restore continuation, and decode");
}

ModelMemoryActual Runtime::actualRuntimeMemory() const {
  return {impl_->states.actualAllocatedBytes(), impl_->prefillArena->bytes(),
          impl_->decodeArena->bytes()};
}

ModelTelemetry Runtime::telemetry() const noexcept {
  ModelTelemetry result = impl_->counters;
  result.stateResidentBytes = impl_->states.actualAllocatedBytes();
  result.warmIdleStateCells = impl_->states.idleCells();
  return result;
}

ModelMemoryPlan plannedRuntimeMemory(const DeviceCapabilities &device,
                                     const ModelPackage &package,
                                     const ops::ExecutionPlans &operators) {
  requireCompatibleModelPackage(package);
  if (device.appleGpuFamily < DeviceCapabilities::kMinimumAppleGpuFamily) {
    throw std::invalid_argument("model runtime requires Apple tensor BF16");
  }
  const RuntimeGeometry geometry = RuntimeGeometry::from(package);
  return {package.stateLayout().activeCellBytes(),
          plannedPrefillBytes(geometry, operators),
          plannedDecodeBytes(geometry, operators), kPipelineReserveBytes,
          kRuntimeOverheadReserveBytes};
}

std::unique_ptr<StateStorage>
createStateStorage(metal::MetalBackend &backend,
                   metal::AllocationAdmission admitAllocation,
                   const ModelPackage &package) {
  requireCompatibleModelPackage(package);
  return std::make_unique<QwenStateStorage>(
      backend, std::move(admitAllocation), package.stateLayout());
}

std::unique_ptr<RuntimeModel> createRuntime(RuntimeContext context) {
  return std::make_unique<Runtime>(std::move(context));
}

} // namespace splash::model
