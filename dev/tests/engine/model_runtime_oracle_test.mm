#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/Types.hpp"
#include "model/Runtime.hpp"
#include "model/QwenState.hpp"
#include "ops/Q8PageStorage.hpp"
#include "ops/Vision.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

constexpr uint32_t kVocabulary = 248320;
constexpr uint32_t kMaskWords = (kVocabulary + 31) / 32;

struct Similarity final {
  double cosine = 0.0;
  double maximumAbsolute = 0.0;
  double leftNorm = 0.0;
  double rightNorm = 0.0;
};

class SimilarityAccumulator final {
public:
  void add(float left, float right) {
    const double a = left;
    const double b = right;
    dot_ += a * b;
    leftSquare_ += a * a;
    rightSquare_ += b * b;
    maximumAbsolute_ = std::max(maximumAbsolute_, std::abs(a - b));
  }

  [[nodiscard]] Similarity result() const {
    const double denominator = std::sqrt(leftSquare_ * rightSquare_);
    const double cosine = denominator > 0.0
                              ? dot_ / denominator
                              : (leftSquare_ == rightSquare_ ? 1.0 : 0.0);
    return {cosine, maximumAbsolute_, std::sqrt(leftSquare_),
            std::sqrt(rightSquare_)};
  }

private:
  double dot_ = 0.0;
  double leftSquare_ = 0.0;
  double rightSquare_ = 0.0;
  double maximumAbsolute_ = 0.0;
};

float bfloatToFloat(uint16_t value) {
  return std::bit_cast<float>(uint32_t{value} << 16);
}

const uint16_t *bfloatContents(const metal::MetalBuffer &buffer,
                               const std::string &label) {
  if (!buffer.contents() || buffer.sizeBytes() % sizeof(uint16_t)) {
    fail(label + " is not CPU-visible BF16 storage");
  }
  return static_cast<const uint16_t *>(buffer.contents());
}

Similarity compareBfloat(const metal::MetalBuffer &left,
                         const metal::MetalBuffer &right,
                         uint64_t maximumSamples = 262144) {
  require(left.sizeBytes() == right.sizeBytes(),
          "BF16 comparison shape mismatch");
  const uint64_t elements = left.sizeBytes() / sizeof(uint16_t);
  const uint64_t stride = std::max<uint64_t>(1, elements / maximumSamples);
  const uint16_t *a = bfloatContents(left, "left BF16 buffer");
  const uint16_t *b = bfloatContents(right, "right BF16 buffer");
  SimilarityAccumulator accumulator;
  for (uint64_t index = 0; index < elements; index += stride) {
    accumulator.add(bfloatToFloat(a[index]), bfloatToFloat(b[index]));
  }
  return accumulator.result();
}

Similarity compareFloat(const metal::MetalBuffer &left,
                        const metal::MetalBuffer &right,
                        uint64_t maximumSamples = 262144) {
  require(left.sizeBytes() == right.sizeBytes(),
          "FP32 comparison shape mismatch");
  require(left.contents() && right.contents() &&
              left.sizeBytes() % sizeof(float) == 0,
          "FP32 comparison buffer is not CPU-visible");
  const uint64_t elements = left.sizeBytes() / sizeof(float);
  const uint64_t stride = std::max<uint64_t>(1, elements / maximumSamples);
  const auto *a = static_cast<const float *>(left.contents());
  const auto *b = static_cast<const float *>(right.contents());
  SimilarityAccumulator accumulator;
  for (uint64_t index = 0; index < elements; index += stride) {
    accumulator.add(a[index], b[index]);
  }
  return accumulator.result();
}

void requireCommittedStateEquivalent(const model::QwenStateStorage &states,
                                     uint32_t speculativeSlot,
                                     uint32_t replaySlot,
                                     const std::string &label) {
  const model::QwenSlotMetadata &speculative = states.metadata(speculativeSlot);
  const model::QwenSlotMetadata &replayed = states.metadata(replaySlot);
  require(speculative.lengths == replayed.lengths,
          label + " logical state differs from replay");

  const model::QwenSlotBuffers &speculativeBuffers =
      states.buffers(speculativeSlot);
  const model::QwenSlotBuffers &replayedBuffers = states.buffers(replaySlot);
  const Similarity convolution = compareBfloat(
      speculativeBuffers.gdn[speculative.activeParity].convolutionBase,
      replayedBuffers.gdn[replayed.activeParity].convolutionBase);
  const Similarity recurrent = compareFloat(
      speculativeBuffers.gdn[speculative.activeParity].recurrentBase,
      replayedBuffers.gdn[replayed.activeParity].recurrentBase);
  const Similarity firstConvolution = compareBfloat(
      speculativeBuffers.gdn[speculative.activeParity].convolutionLayers.front(),
      replayedBuffers.gdn[replayed.activeParity].convolutionLayers.front());
  const Similarity firstRecurrent = compareFloat(
      speculativeBuffers.gdn[speculative.activeParity].recurrentLayers.front(),
      replayedBuffers.gdn[replayed.activeParity].recurrentLayers.front());
  constexpr double aggregateStateCosine = 0.99;
  constexpr double firstLayerStateCosine = 0.9999;
  const bool stateEquivalent =
      convolution.cosine > aggregateStateCosine &&
      recurrent.cosine > aggregateStateCosine &&
      firstConvolution.cosine > firstLayerStateCosine &&
      firstRecurrent.cosine > firstLayerStateCosine;
  if (!stateEquivalent) {
    std::cerr << label << " gdn conv_cos=" << convolution.cosine
              << " conv_max_abs=" << convolution.maximumAbsolute
              << " conv_norms=" << convolution.leftNorm << '/'
              << convolution.rightNorm << " recurrent_cos=" << recurrent.cosine
              << " recurrent_max_abs=" << recurrent.maximumAbsolute
              << " recurrent_norms=" << recurrent.leftNorm << '/'
              << recurrent.rightNorm << '\n';
    for (uint32_t layer = 0; layer < states.layout().target.layers; ++layer) {
      const Similarity layerConvolution = compareBfloat(
          speculativeBuffers.gdn[speculative.activeParity]
              .convolutionLayers[layer],
          replayedBuffers.gdn[replayed.activeParity].convolutionLayers[layer]);
      const Similarity layerRecurrent = compareFloat(
          speculativeBuffers.gdn[speculative.activeParity]
              .recurrentLayers[layer],
          replayedBuffers.gdn[replayed.activeParity].recurrentLayers[layer]);
      std::cerr << "  layer=" << layer
                << " conv_cos=" << layerConvolution.cosine
                << " recurrent_cos=" << layerRecurrent.cosine << '\n';
    }
  }
  // The committed state consumes the optimized M8/M16/M24/M32 target path;
  // teacher-forced replay consumes the ragged-prefill path. Sparse-MoE routing
  // can amplify their BF16/Q4 rounding differences in later layers. Requiring
  // the first GDN layer to remain nearly identical still catches commit-count,
  // stride and state-addressing errors before that amplification is possible.
  require(stateEquivalent, label + " committed GDN state differs from replay");

  for (uint32_t layer = 0; layer < states.layout().draft.layers; ++layer) {
    const Similarity keys = compareBfloat(speculativeBuffers.draft[layer].keys,
                                          replayedBuffers.draft[layer].keys);
    const Similarity values =
        compareBfloat(speculativeBuffers.draft[layer].values,
                      replayedBuffers.draft[layer].values);
    if (!(keys.cosine > 0.99 && values.cosine > 0.99)) {
      std::cerr << label << " draft layer=" << layer
                << " key_cos=" << keys.cosine
                << " value_cos=" << values.cosine << '\n';
    }
    require(keys.cosine > 0.99 && values.cosine > 0.99,
            label + " committed draft state differs from replay");
  }
}

EngineRequest makeRequest(uint64_t id, std::vector<uint32_t> prompt,
                          uint32_t maximumNewTokens,
                          BatchCohort cohort = BatchCohort::Greedy) {
  EngineRequest result;
  result.id = id;
  result.prompt = std::move(prompt);
  result.maxNewTokens = maximumNewTokens;
  result.cohort = cohort;
  return result;
}

void beginCold(model::Runtime &runtime, const EngineRequest &request,
               uint32_t slot) {
  runtime.beginColdRequest(request.modelView(), slot);
}

void restoreActivePrefix(model::Runtime &executor, uint64_t requestId,
                         uint32_t promptTokens, uint32_t boundary,
                         const std::shared_ptr<const CompositeState> &state) {
  executor.restore(requestId, boundary, state, true);
  executor.setDraftContextPlan(
      requestId, planDraftContext(boundary, promptTokens, boundary, {}));
}

ModelStepResult prefillChunk(model::Runtime &executor, uint64_t requestId,
                             uint32_t slot, uint32_t logicalPosition,
                             uint32_t promptOffset,
                             std::span<const uint32_t> inputTokens,
                             const std::vector<uint32_t> &pageTable,
                             BatchCohort cohort = BatchCohort::Greedy,
                             std::optional<bool> representativeTiming = {}) {
  const uint32_t tokenCount = static_cast<uint32_t>(inputTokens.size());
  BatchPlan plan{WorkKind::Prefill,
                 cohort,
                 {{requestId, tokenCount}},
                 DecodeStage::Regular};
  ModelBatchItem item{requestId,    slot,       logicalPosition,
                         promptOffset, tokenCount, pageTable};
  item.inputTokens = inputTokens;
  auto ticket = executor.submit(
      plan, std::span<const ModelBatchItem>(&item, 1), {});
  if (representativeTiming) {
    require(ticket->prefillTimingIsRepresentative() == *representativeTiming,
            "image encoding timing classification changed");
  }
  auto result = ticket->wait();
  require(result.size() == 1 && result[0].consumedPromptTokens == tokenCount,
          "prefill result mismatch");
  return std::move(result[0]);
}

// Sections that compare decode cycles need requests whose prompt did not end
// the sequence already; the runtime emits a stop token from prefill itself.
void requireOpen(const ModelStepResult &prefilled, const char *section) {
  require(prefilled.outputTokens.empty(),
          (std::string(section) + ": prompt ended right after prefill").c_str());
}

void prefillToken(model::Runtime &executor, uint64_t requestId,
                  uint32_t slot, uint32_t logicalPosition,
                  uint32_t promptOffset, uint32_t token,
                  const std::vector<uint32_t> &pageTable,
                  BatchCohort cohort = BatchCohort::Greedy) {
  const std::array<uint32_t, 1> input{token};
  requireOpen(prefillChunk(executor, requestId, slot, logicalPosition,
                           promptOffset, input, pageTable, cohort),
              "single-token prefill");
}

ModelStepResult
decodeOne(model::Runtime &executor, uint64_t requestId, uint32_t slot,
          uint64_t logicalPosition, const std::vector<uint32_t> &pageTable,
          BatchCohort cohort, DecodeStage decodeStage = DecodeStage::Regular) {
  BatchPlan plan{WorkKind::Decode, cohort, {{requestId, 0}}, decodeStage};
  ModelBatchItem item{requestId, slot, logicalPosition, 0, 0, pageTable};
  auto result =
      executor.decode(plan, std::span<const ModelBatchItem>(&item, 1));
  require(result.size() == 1 && result[0].requestId == requestId,
          "decode result mismatch");
  return std::move(result[0]);
}

// A stop token or a one-token budget is emitted by prefill itself; otherwise
// the first output tokens come from one decode cycle.
ModelStepResult firstStep(model::Runtime &executor, ModelStepResult prefilled,
                          uint64_t requestId, uint32_t slot,
                          uint64_t logicalPosition,
                          const std::vector<uint32_t> &pageTable,
                          BatchCohort cohort) {
  if (!prefilled.outputTokens.empty())
    return prefilled;
  return decodeOne(executor, requestId, slot, logicalPosition, pageTable,
                   cohort);
}

struct PendingMaskedDecode final {
  std::unique_ptr<ModelBatchTicket> ticket;
  std::vector<ModelMaskRequest> maskRequests;
};

PendingMaskedDecode
beginMaskedDecode(model::Runtime &executor, const BatchPlan &plan,
                  std::span<const ModelBatchItem> items) {
  PendingMaskedDecode pending;
  pending.ticket = executor.submit(plan, items, [] {});
  while (pending.maskRequests.empty()) {
    pending.maskRequests = pending.ticket->takeMaskRequests();
    require(!pending.ticket->ready(),
            "constrained decode completed before receiving its mask");
    if (pending.maskRequests.empty())
      std::this_thread::yield();
  }
  return pending;
}

std::vector<ModelStepResult>
finishMaskedDecode(PendingMaskedDecode pending) {
  while (!pending.ticket->ready()) {
    require(pending.ticket->takeMaskRequests().empty(),
            "constrained decode requested its mask twice");
    if (!pending.ticket->ready())
      std::this_thread::yield();
  }
  return pending.ticket->wait();
}

PendingMaskedDecode
beginMaskedDecodeOne(model::Runtime &executor, uint64_t requestId,
                     uint32_t slot, uint64_t logicalPosition,
                     const std::vector<uint32_t> &pageTable,
                     DecodeStage decodeStage) {
  BatchPlan plan{WorkKind::Decode, BatchCohort::Constrained,
                 {{requestId, 0}}, decodeStage};
  const std::array items{ModelBatchItem{
      requestId, slot, logicalPosition, 0, 0, pageTable}};
  return beginMaskedDecode(executor, plan, items);
}

std::vector<uint32_t> pageRange(uint32_t first, uint32_t count) {
  std::vector<uint32_t> result;
  result.reserve(count);
  for (uint32_t page = 0; page < count; ++page) {
    result.push_back(first + page);
  }
  return result;
}

std::vector<uint32_t> singletonMasks(std::span<const uint32_t> tokens) {
  std::vector<uint32_t> result(uint64_t{tokens.size()} * kMaskWords, 0);
  for (uint32_t row = 0; row < tokens.size(); ++row) {
    require(tokens[row] < kVocabulary, "singleton mask token is invalid");
    result[uint64_t{row} * kMaskWords + tokens[row] / 32] |=
        1U << (tokens[row] % 32);
  }
  return result;
}

using StateSamples = std::vector<std::pair<std::string, std::vector<float>>>;

StateSamples sampleCommittedState(const model::QwenStateStorage &states,
                                   uint32_t slot) {
  StateSamples result;
  const auto add = [&](std::string name, const metal::MetalBuffer &buffer,
                       bool bfloat) {
    std::vector<float> values;
    const uint64_t count = buffer.sizeBytes() / (bfloat ? 2 : 4);
    const uint64_t stride = std::max<uint64_t>(1, count / 65536);
    for (uint64_t index = 0; index < count; index += stride) {
      values.push_back(bfloat ? bfloatToFloat(static_cast<const uint16_t *>(
                                                 buffer.contents())[index])
                             : static_cast<const float *>(buffer.contents())[index]);
    }
    result.emplace_back(std::move(name), std::move(values));
  };
  const auto &buffers = states.buffers(slot);
  const auto &gdn = buffers.gdn[states.metadata(slot).activeParity];
  add("convolution", gdn.convolutionBase, true);
  add("recurrent", gdn.recurrentBase, false);
  add("first_convolution", gdn.convolutionLayers.front(), true);
  add("first_recurrent", gdn.recurrentLayers.front(), false);
  const auto &lengths = states.metadata(slot).lengths;
  const auto layout = states.layout().draft;
  const uint64_t elements = uint64_t{layout.kvHeads} * lengths.draftLength *
                            layout.headDimension;
  const uint64_t stride = std::max<uint64_t>(1, elements / 65536);
  for (uint32_t layer = 0; layer < buffers.draft.size(); ++layer) {
    const auto *keys = bfloatContents(buffers.draft[layer].keys, "draft keys");
    const auto *values = bfloatContents(buffers.draft[layer].values, "draft values");
    std::vector<float> keySamples, valueSamples;
    for (uint64_t index = 0; index < elements; index += stride) {
      const uint32_t dimension = index % layout.headDimension;
      const uint32_t position = (index / layout.headDimension) % lengths.draftLength;
      const uint32_t head = index / (uint64_t{layout.headDimension} * lengths.draftLength);
      const uint32_t ring = (lengths.draftBase + position) % layout.tokens;
      keySamples.push_back(bfloatToFloat(
          keys[(uint64_t{head} * layout.tokens + ring) * layout.headDimension + dimension]));
      valueSamples.push_back(bfloatToFloat(
          values[(uint64_t{head} * layout.headDimension + dimension) * layout.tokens + ring]));
    }
    result.emplace_back("draft_key_" + std::to_string(layer), std::move(keySamples));
    result.emplace_back("draft_value_" + std::to_string(layer), std::move(valueSamples));
  }
  return result;
}

void compareCommittedSamples(const StateSamples &before,
                              const StateSamples &after, bool exact) {
  require(before.size() == after.size(), "preemption state sample shape changed");
  for (size_t tensor = 0; tensor < before.size(); ++tensor) {
    const auto &[name, values] = before[tensor];
    require(values.size() == after[tensor].second.size(),
            "preemption tensor sample shape changed");
    if (exact) {
      require(values == after[tensor].second,
              "regenerated state differs from independent teacher forcing: " + name);
      continue;
    }
    SimilarityAccumulator comparison;
    for (size_t index = 0; index < values.size(); ++index)
      comparison.add(values[index], after[tensor].second[index]);
    const Similarity result = comparison.result();
    std::cout << "preemption_state " << name << " cosine=" << result.cosine
              << " maximum_absolute=" << result.maximumAbsolute << '\n';
  }
}

struct AllocationFault final {
  int remaining = -1;
  bool throwAfterAllocation = false;
  uint64_t remainingBytes = std::numeric_limits<uint64_t>::max();
};

void requireAtomicImageAdmission(model::Runtime &executor,
                                 metal::MetalBackend &backend,
                                 AllocationFault &fault) {
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  const uint64_t originalSubmissions = backend.submissionCount();
  EngineRequest image = makeRequest(93, {1, 2}, 1);
  image.images = {{0, 1, 2, 2, 139, 431}};
  image.imagePixels.resize(image.images.front().pixelBytes());
  for (bool resume : {false, true}) {
    EngineRequest text = image;
    text.images.clear();
    text.imagePixels.clear();
    if (resume) {
      require(executor.begin(text.modelView()).granted(),
              "full-lane resume setup failed");
      executor.suspend(text.id);
    }
    for (uint32_t lane = 0; lane < model::ExecutionLimits::maximumBatchWidth;
         ++lane) {
      text.id = 100 + lane;
      require(executor.begin(text.modelView()).granted(),
              "full-lane image setup failed");
    }
    const uint64_t before = backend.memoryStats().allocatedBytes;
    fault = {0};
    const StateAdmission denied = resume ? executor.resume(image.modelView())
                                        : executor.begin(image.modelView());
    require(!denied.granted() &&
                denied.failure == StateFailure::ConcurrencyLimit &&
                fault.remaining == 0 &&
                backend.memoryStats().allocatedBytes == before,
            "full execution lanes attempted image memory admission");
    fault = {};
    for (uint32_t lane = 0; lane < model::ExecutionLimits::maximumBatchWidth;
         ++lane)
      executor.end(100 + lane);
    require((resume ? executor.resume(image.modelView())
                    : executor.begin(image.modelView())).granted(),
            "image could not retry after an execution lane became free");
    executor.end(image.id);
    while (executor.reclaimIdleState()) {
    }
    require(backend.memoryStats().allocatedBytes == originalBytes,
            "full-lane image test leaked resources");
  }
  for (bool resume : {false, true}) {
    if (resume) {
      EngineRequest text = image;
      text.images.clear();
      text.imagePixels.clear();
      require(executor.begin(text.modelView()).granted(),
              "image resume setup failed");
      executor.suspend(text.id);
    }
    for (bool sharedVision : {false, true}) {
      EngineRequest keeper = image;
      keeper.id = 94;
      if (sharedVision)
        require(executor.begin(keeper.modelView()).granted(),
                "shared vision setup failed");
      const uint64_t before = backend.memoryStats().allocatedBytes;
      // Fresh vision, image pixels/embeddings, two GDN cells, draft ring.
      // With an existing encoder, only the last four allocations remain.
      for (int boundary = 0; boundary < (sharedVision ? 4 : 5); ++boundary) {
        for (bool throwing : {false, true}) {
          fault = {boundary, throwing};
          bool threw = false;
          try {
            const StateAdmission admission = resume
                ? executor.resume(image.modelView())
                : executor.begin(image.modelView());
            require(!admission.granted() &&
                        admission.failure == StateFailure::MemoryPressure,
                    "image allocation denial was not retryable");
          } catch (const std::runtime_error &error) {
            require(std::string(error.what()) == "injected image allocation",
                    "unexpected image admission exception");
            threw = true;
          }
          fault = {};
          require(threw == throwing, "image admission exception was lost");
          require(backend.memoryStats().allocatedBytes == before,
                  "failed image admission retained or removed shared buffers");
          require(executor.reclaimIdleState() == 0,
                  "failed image admission created false reclamation progress");
        }
      }
      if (sharedVision) {
        executor.end(keeper.id);
        while (executor.reclaimIdleState()) {
        }
      }
    }
    // The same request can still be admitted after every failure mode.
    require((resume ? executor.resume(image.modelView())
                    : executor.begin(image.modelView())).granted(),
            "image request could not retry after allocation failure");
    executor.end(image.id);
    while (executor.reclaimIdleState()) {
        }
    require(backend.memoryStats().allocatedBytes == originalBytes,
            "image admission test leaked resources");
  }
  require(backend.submissionCount() == originalSubmissions,
          "image allocation regression unexpectedly submitted GPU work");
  std::cout << "image_admission_atomic_rollback=PASS\n";
}

// An image whose rows straddle a chunk boundary is encoded by the first
// chunk; the reclaimer may drop the idle encoder before the second chunk
// injects the remaining rows from the retained embeddings. Rows served from
// the embedding cache stay held by their request when the cache is dropped.
void requireImageRowsAfterReclaim(model::Runtime &executor,
                                  metal::MetalBackend &backend,
                                  model::QwenStateStorage &states,
                                  const model::ModelPackage &model,
                                  AllocationFault &fault) {
  while (executor.reclaimIdleState()) {
  }
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  const uint64_t encodesBefore = executor.telemetry().imageEncodes;
  const uint64_t reusesBefore = executor.telemetry().imageEmbeddingReuses;
  std::vector<uint32_t> prompt(128);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = 1 + index;
  EngineRequest request = makeRequest(95, prompt, 1);
  request.images = {{56, 16, 8, 8, 151, 433}};
  request.imagePixels.resize(request.images.front().pixelBytes());
  for (size_t index = 0; index < request.imagePixels.size(); ++index)
    request.imagePixels[index] = static_cast<uint8_t>(index * 7 + 3);
  const std::vector<uint32_t> pages = pageRange(120, 4);
  const StateAdmission admission = executor.begin(request.modelView());
  require(admission.granted(), "straddling image request was not admitted");
  const uint32_t slot = *admission.cell;
  executor.setDraftContextPlan(
      request.id, planDraftContext(0, static_cast<uint32_t>(prompt.size()),
                                   std::nullopt, {}));
  prefillChunk(executor, request.id, slot, 0, 0,
               std::span<const uint32_t>(prompt).first(64), pages,
               BatchCohort::Greedy, false);
  // Nothing else is idle, so the pass releases exactly the encoder arena.
  uint64_t reclaimed = 0;
  while (const uint64_t bytes = executor.reclaimIdleState())
    reclaimed += bytes;
  const uint64_t encoderBytes = ops::Vision::scratchBytes(
      model.vision.tensors.layout, ops::kMaximumImagePatches);
  require(reclaimed == encoderBytes,
          "encoder whose only image is encoded survived reclaim");
  prefillChunk(executor, request.id, slot, 64, 64,
               std::span<const uint32_t>(prompt).subspan(64), pages,
               BatchCohort::Greedy, true);
  require(executor.telemetry().imageEncodes == encodesBefore + 1,
          "straddling image was not encoded exactly once");
  executor.end(request.id);

  // Free only pooled state, keeping the image cache. A cache-only request
  // must fit a fresh state cell without recreating the reclaimed encoder.
  static_cast<void>(states.releaseIdle(0, 0));
  const uint64_t stateBytes = model.stateLayout().activeCellBytes();
  require(stateBytes < encoderBytes, "image budget fixture cannot deny the encoder");
  const uint64_t beforeReuse = backend.memoryStats().allocatedBytes;
  request.id = 96;
  fault.remainingBytes = stateBytes;
  const StateAdmission repeated = executor.begin(request.modelView());
  const uint64_t remainingBytes = fault.remainingBytes;
  fault = {};
  require(repeated.granted() && remainingBytes == 0 &&
              backend.memoryStats().allocatedBytes == beforeReuse + stateBytes &&
              executor.telemetry().imageEmbeddingReuses == reusesBefore + 1,
          "cached image required more than its fresh request state");

  // A mixed hit/miss must keep the cached rows while admitting new resources.
  // Fail at the encoder, image buffers and first state cell, including an
  // exception after allocation, and leave both the cache and live request intact.
  EngineRequest mixed = request;
  mixed.id = 97;
  mixed.images.push_back({80, 16, 8, 8, 157, 439});
  mixed.imagePixels.resize(2 * request.imagePixels.size());
  const uint64_t beforeMixed = backend.memoryStats().allocatedBytes;
  for (int boundary : {0, 1, 2}) {
    for (bool throwing : {false, true}) {
      fault = {boundary, throwing};
      bool threw = false;
      try {
        const StateAdmission denied = executor.begin(mixed.modelView());
        require(!denied.granted() && denied.failure == StateFailure::MemoryPressure,
                "mixed image allocation denial was not retryable");
      } catch (const std::runtime_error &error) {
        require(std::string(error.what()) == "injected image allocation",
                "unexpected mixed image admission exception");
        threw = true;
      }
      fault = {};
      require(threw == throwing &&
                  backend.memoryStats().allocatedBytes == beforeMixed,
              "mixed image admission changed preexisting buffers on failure");
    }
  }
  const uint64_t reusedBeforeMixed = executor.telemetry().imageEmbeddingReuses;
  const ImageSpan &miss = mixed.images.back();
  const uint64_t missingImageBytes = miss.pixelBytes() +
      uint64_t{ops::Vision::embeddingRows({miss.gridHeight, miss.gridWidth})} *
          model.vision.tensors.layout.outputHiddenSize * sizeof(uint16_t);
  require(executor.begin(mixed.modelView()).granted() &&
              executor.telemetry().imageEmbeddingReuses == reusedBeforeMixed + 1 &&
              backend.memoryStats().allocatedBytes ==
                  beforeMixed + encoderBytes + missingImageBytes + stateBytes,
          "mixed image retry did not reuse the cache and allocate only its miss");
  executor.end(mixed.id);

  // The cache-only request still holds its rows: dropping their cache entry
  // frees nothing, so the reclaimer must not credit those bytes.
  const uint64_t heldBytes = backend.memoryStats().allocatedBytes;
  uint64_t released = 0;
  while (const uint64_t bytes = executor.reclaimIdleState())
    released += bytes;
  require(released == heldBytes - backend.memoryStats().allocatedBytes,
          "reclaim credited cached rows a live request still holds");
  executor.end(request.id);

  while (executor.reclaimIdleState()) {
  }
  require(backend.memoryStats().allocatedBytes == originalBytes,
          "image requests leaked resources");
  std::cout << "image_rows_after_reclaim=PASS\n";
}

void requireRepeatedImagePlacements(model::Runtime &executor,
                                     metal::MetalBackend &backend,
                                     model::QwenStateStorage &states,
                                     AllocationFault &fault) {
  while (executor.reclaimIdleState()) {
  }
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  std::vector<uint32_t> prompt(128);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = 1 + index;
  EngineRequest request = makeRequest(98, prompt, 1);
  request.images = {{16, 16, 8, 8, 163, 443}};
  request.imagePixels.resize(request.images.front().pixelBytes());
  for (size_t index = 0; index < request.imagePixels.size(); ++index)
    request.imagePixels[index] = static_cast<uint8_t>(index * 11 + 5);
  require(executor.begin(request.modelView()).granted(),
          "single image allocation fixture was not admitted");
  const uint64_t singleImageBytes =
      backend.memoryStats().allocatedBytes - originalBytes;
  executor.end(request.id);
  while (executor.reclaimIdleState()) {
  }
  require(backend.memoryStats().allocatedBytes == originalBytes,
          "single image allocation fixture retained memory");

  const auto pixels = request.imagePixels;
  for (uint32_t offset : {80U, 104U}) {
    ImageSpan repeated = request.images.front();
    repeated.offset = offset;
    request.images.push_back(repeated);
    request.imagePixels.insert(request.imagePixels.end(), pixels.begin(),
                               pixels.end());
  }
  fault.remainingBytes = singleImageBytes;
  const StateAdmission admitted = executor.begin(request.modelView());
  const uint64_t remaining = fault.remainingBytes;
  fault = {};
  require(admitted.granted() && remaining == 0 &&
              backend.memoryStats().allocatedBytes ==
                  originalBytes + singleImageBytes,
          "repeated placements allocated multiple image buffers");
  const uint32_t slot = *admitted.cell;
  const std::vector<uint32_t> pages = pageRange(120, 4);
  const std::array<uint32_t, 1> checkpoints{64};
  executor.setDraftContextPlan(
      request.id, planDraftContext(0, prompt.size(), std::nullopt, checkpoints));
  const uint64_t encodes = executor.telemetry().imageEncodes;
  prefillChunk(executor, request.id, slot, 0, 0,
               std::span<const uint32_t>(prompt).first(64), pages,
               BatchCohort::Greedy, false);
  auto checkpoint = executor.snapshot(request.id);
  require(checkpoint != nullptr, "image prefix checkpoint allocation failed");
  prefillChunk(executor, request.id, slot, 64, 64,
               std::span<const uint32_t>(prompt).subspan(64), pages,
               BatchCohort::Greedy, true);
  require(executor.telemetry().imageEncodes == encodes + 1,
          "repeated image placements encoded more than once");
  const auto expected = sampleCommittedState(states, slot);
  executor.end(request.id);
  while (executor.reclaimIdleState()) {
  }

  // The first placement is covered by the prefix, but its later duplicates
  // still need their shared image data after the embedding cache is reclaimed.
  request.id = 99;
  const StateAdmission restored = executor.begin(request.modelView());
  require(restored.granted(), "repeated image restore was not admitted");
  restoreActivePrefix(executor, request.id, prompt.size(), 64, checkpoint);
  prefillChunk(executor, request.id, *restored.cell, 64, 64,
               std::span<const uint32_t>(prompt).subspan(64), pages,
               BatchCohort::Greedy, false);
  require(executor.telemetry().imageEncodes == encodes + 2,
          "prefix restore discarded data for a later image placement");
  compareCommittedSamples(expected, sampleCommittedState(states, *restored.cell),
                          true);
  executor.end(request.id);
  checkpoint.reset();
  while (executor.reclaimIdleState()) {
  }
  require(backend.memoryStats().allocatedBytes == originalBytes,
          "repeated image placements retained resources");
  std::cout << "repeated_image_placements=PASS\n";
}

void warmupEos(model::RuntimeContext context, model::ModelPackage &package) {
  uint32_t prefillStop = 0;
  uint32_t decodeStop = 0;
  {
    model::Runtime baseline(context);
    const auto prefill = baseline.warmupPrefill(1);
    const auto decoded = baseline.warmupDecodeBatch(1);
    require(prefill.lanes[0].pendingToken.has_value() &&
                decoded.lanes[0].pendingToken.has_value(),
            "warmup EOS fixture has no deterministic token");
    prefillStop = *prefill.lanes[0].pendingToken;
    decodeStop = *decoded.lanes[0].pendingToken;
    require(decodeStop != 0 && !decoded.lanes[0].step.finished,
            "warmup EOS fixture needs a non-terminal baseline continuation");
  }
  const auto originalTarget = package.descriptor.target;
  const auto setStops = [&](uint32_t stop) {
    std::visit([&](auto &weights) {
      weights.layout.stopTokens = {stop, stop};
      package.descriptor.target = weights.layout;
    }, package.target);
  };
  setStops(prefillStop);
  {
    model::Runtime executor(context);
    const auto prefill = executor.warmupPrefill(1);
    require(prefill.lanes[0].step.finished &&
                prefill.lanes[0].step.outputTokensWithoutKv == 1,
            "EOS fixture did not terminate synthetic prefill");
    for (uint32_t width = 1; width <= 4; ++width) {
      const auto result = executor.warmupDecodeBatch(width);
      require(result.completed && result.lanes.size() == width &&
                  executor.telemetry().lastDecodeWidth == width,
              "prefill EOS skipped the actual decode warmup");
    }
    require(executor.warmupDraftVerifyCommit().completed &&
                executor.warmupCompositeStateRestore().completed,
            "prefill EOS broke commit/restore warmup");
  }
  setStops(decodeStop);
  {
    model::Runtime executor(context);
    const auto result = executor.warmupDecodeBatch(1);
    const auto &lane = result.lanes[0];
    require(result.completed && lane.step.finished &&
                lane.step.outputTokensWithoutKv == 1 &&
                lane.pendingToken == decodeStop && lane.committedTokens > 1 &&
                lane.committedTokens == lane.step.outputTokens.size(),
            "decode EOS was not distinguished from committed KV rows");
  }
  std::visit([&](auto &weights) {
    using Layout = std::decay_t<decltype(weights.layout)>;
    weights.layout = std::get<Layout>(originalTarget);
  }, package.target);
  package.descriptor.target = originalTarget;
  auto &states = static_cast<model::QwenStateStorage &>(context.stateStorage);
  for (uint32_t slot = 0; slot < 4; ++slot)
    require(!states.metadata(slot).assigned,
            "EOS warmup left an active state slot");
  std::cout << "warmup_eos=PASS prefill_stop=" << prefillStop
            << " decode_stop=" << decodeStop << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const bool imagesOnly = argc == 4 && std::string_view(argv[3]) == "--images-only";
    const bool warmupEosOnly = argc == 4 &&
        std::string_view(argv[3]) == "--warmup-eos-only";
    if (argc != 3 && !imagesOnly && !warmupEosOnly) {
      std::cerr << "usage: model-runtime-oracle METALLIB MODEL_ROOT "
                   "[--images-only|--warmup-eos-only]\n";
      return 2;
    }
    metal::MetalBackend backend(argv[1]);
    const auto &device = backend.capabilities();
    if (const auto error = device.validationError())
      fail(*error);
    const uint64_t hostReserveBytes =
        EngineMemoryPolicy::hostAvailableReserveBytes(device.physicalMemoryBytes);
    const auto hostAvailableBytes = queryHostAvailableMemory();
    require(hostAvailableBytes.has_value(),
            "cannot measure available host memory before loading the oracle model");
    require(*hostAvailableBytes > hostReserveBytes,
            "available host memory does not cover the protected macOS reserve");
    uint64_t remainingModelBytes = *hostAvailableBytes - hostReserveBytes;
    const std::filesystem::path modelRoot(argv[2]);
    bool hasModelBytes = false;
    // Match production's packed-file preflight before any model mappings.
    // Subtract from available capacity so package sizes cannot overflow a sum.
    for (const char *directory : {"target", "draft", "vision"}) {
      for (const auto &entry :
           std::filesystem::recursive_directory_iterator(modelRoot / directory)) {
        if (!entry.is_regular_file())
          continue;
        const auto bytes = entry.file_size();
        require(bytes <= remainingModelBytes,
                "oracle model loading exceeds available host memory after protecting " +
                    std::to_string(hostReserveBytes) + " bytes for macOS");
        remainingModelBytes -= bytes;
        hasModelBytes |= bytes != 0;
      }
    }
    require(hasModelBytes, "oracle model package contains no nonempty regular files");
    model::ModelPackage model =
        model::loadModelPackage(backend, modelRoot);
    ops::ExecutionPlans operators(backend.capabilities());
    model::ModelMemoryPlan executorPlan =
        model::plannedRuntimeMemory(backend.capabilities(), model, operators);
    ModelMemoryFootprint footprint{
        model.targetActualAllocatedBytes(),
        model.draft.actualAllocatedBytes,
        model.vision.actualAllocatedBytes,
        model.stateLayout().activeCellBytes(),
        executorPlan.sharedPrefillPlannedAllocatedBytes,
        executorPlan.sharedDecodePlannedAllocatedBytes,
        executorPlan.pipelineReserveBytes,
        executorPlan.runtimeOverheadReserveBytes};
    ModelMemoryProfile profile{
        model.name(), model.maximumContextTokens(),
        model.targetKvLayout(), footprint};
    EngineMemoryPlan memoryPlan =
        requireEngineMemoryPlan(backend.capabilities(), profile);

    // The pool is a whole number of sparse-mapping batches: the 4-head layout
    // maps 128 pages at a time, the 2-head layout 256 (64 KiB tiles).
    const uint32_t pageCount =
        std::max(128U, model.targetKvLayout().sparseMappingBatchPages());
    const EngineMemoryBreakdown &budget = memoryPlan.breakdown();
    require(budget.pipelineReserveBytes <= budget.hardBudgetBytes &&
                budget.runtimeOverheadReserveBytes <
                    budget.hardBudgetBytes - budget.pipelineReserveBytes,
            "runtime reserves consume the oracle Metal budget");
    const uint64_t elasticGrowthCeiling = budget.hardBudgetBytes -
        budget.pipelineReserveBytes - budget.runtimeOverheadReserveBytes;
    MemoryGovernor governor(backend, elasticGrowthCeiling, hostReserveBytes);
    AllocationFault allocationFault;
    const metal::AllocationAdmission admission =
        [admit = governor.allocationAdmission(), &allocationFault](
            uint64_t bytes, const std::function<void()> &allocate) {
          if (bytes > allocationFault.remainingBytes)
            return false;
          if (allocationFault.remaining == 0) {
            if (allocationFault.throwAfterAllocation) {
              require(static_cast<bool>(admit(bytes, allocate)),
                      "test allocation unexpectedly exceeded real budget");
              throw std::runtime_error("injected image allocation");
            }
            return false;
          }
          if (allocationFault.remaining > 0)
            --allocationFault.remaining;
          if (!admit(bytes, allocate))
            return false;
          allocationFault.remainingBytes -= bytes;
          return true;
        };
    metal::AllocationFailure kvAdmissionFailure = metal::AllocationFailure::None;
    kv::Q8PageStorage pages(
        backend,
        [governed = governor.allocationAdmission(), &kvAdmissionFailure](
            uint64_t bytes, const std::function<void()> &allocate)
            -> metal::AllocationResult {
          if (kvAdmissionFailure != metal::AllocationFailure::None)
            return kvAdmissionFailure;
          return governed(bytes, allocate);
        },
        model.targetKvLayout(), pageCount);
    model::QwenStateStorage states(backend,
                                    admission,
                                    model.stateLayout());
    model::RuntimeContext context{
        backend, admission, model, pages, states, operators,
        ops::kMaximumImagePatches, budget.pipelineReserveBytes,
        budget.runtimeOverheadReserveBytes};
    require(executorPlan.sharedDecodePlannedAllocatedBytes <=
                std::numeric_limits<uint64_t>::max() -
                    executorPlan.sharedPrefillPlannedAllocatedBytes,
            "oracle runtime arena reservation overflows");
    auto arenaReservation = governor.tryReserve(
        executorPlan.sharedPrefillPlannedAllocatedBytes +
        executorPlan.sharedDecodePlannedAllocatedBytes);
    require(arenaReservation.has_value(),
            "oracle runtime arenas would consume the protected macOS memory reserve");
    if (warmupEosOnly) {
      warmupEos(context, model);
      return 0;
    }
    model::Runtime executor(context);
    arenaReservation->commit();
    // Fault only physical KV admission, after actual state activation. This
    // exercises Runtime::warmupPrefill's failure propagation and cleanup.
    require(pages.releaseBackingForPage(0), "warmup refusal fixture was not resident");
    pages.awaitRelease();
    const uint64_t beforeWarmupRows = executor.telemetry().targetPrefillRows;
    for (auto failure : {metal::AllocationFailure::HostPressure,
                         metal::AllocationFailure::EngineBudget,
                         metal::AllocationFailure::DriverRejected}) {
      kvAdmissionFailure = failure;
      const uint64_t beforeCommands = backend.submissionCount();
      bool rejected = false;
      try {
        static_cast<void>(executor.warmupPrefill(1));
      } catch (const metal::MetalAllocationError &error) {
        rejected = error.failure() == failure &&
            std::string(error.what()).find("Q8 page backing") != std::string::npos;
      }
      require(rejected && !states.metadata(0).assigned &&
                  executor.telemetry().targetPrefillRows == beforeWarmupRows &&
                  backend.submissionCount() == beforeCommands &&
                  pages.residentPages() == 0,
              "real warmup lost its KV refusal cause or executed/leaked work");
    }
    kvAdmissionFailure = metal::AllocationFailure::None;
    require(static_cast<bool>(pages.ensureResident(0)),
            "warmup refusal fixture failed to recover KV admission");
    static_cast<void>(states.releaseIdle(0, 0));
    requireAtomicImageAdmission(executor, backend, allocationFault);
    for (uint32_t page : pageRange(120, 4))
      require(static_cast<bool>(pages.ensureResident(page)), "image oracle KV backing is unavailable");
    requireImageRowsAfterReclaim(executor, backend, states, model, allocationFault);
    requireRepeatedImagePlacements(executor, backend, states, allocationFault);
    if (imagesOnly) {
      std::cout << "PASS model-runtime-oracle scope=images-only model=" << model.name()
                << " (admission rollback, chunk reclaim, cache-only budget, mixed/repeated images)\n";
      return 0;
    }

    const std::vector<uint32_t> samplingSeedTokens{
        248045, 8678,   198,   24342,  286,    4879,  369,    716,   310,
        830,    11553,  13,    5044,   1683,   15060, 1472,   279,   3274,
        11,     9307,   1328,  30800,  11,     2814,  47675,  25605, 11,
        321,    60445,  55404, 11,     27224,  11,    321,    30246, 303,
        279,    1534,   4087,  13,     248046, 198,   248045, 846,   198,
        9419,   248046, 198,   248045, 74455,  198,   248068, 198};
    // A natural token pattern: a run of zero tokens can make the target select
    // a stop token right after prefill, which leaves no decode cycle to test.
    std::vector<uint32_t> prompt128(128);
    for (uint32_t index = 0; index < prompt128.size(); ++index)
      prompt128[index] = samplingSeedTokens[index % samplingSeedTokens.size()];
    std::vector<uint32_t> prompt129 = prompt128;
    prompt129.push_back(samplingSeedTokens[128 % samplingSeedTokens.size()]);
    const std::vector<uint32_t> pageTable = pageRange(0, 8);
    EngineRequest request = makeRequest(1, prompt128, 16);
    beginCold(executor, request, 0);
    prefillChunk(executor, 1, 0, 0, 0, prompt128, pageTable);
    require(states.metadata(0).lengths.targetTokens == 128 &&
                states.metadata(0).lengths.hasCompleteDraftWindow(
                    states.layout().draft.tokens),
            "prefill state length mismatch");

    const uint64_t predictedPromptSnapshotBytes =
        model.stateLayout().cachedBytes();
    std::shared_ptr<const CompositeState> promptSnapshot =
        executor.snapshot(1);
    require(promptSnapshot != nullptr,
            "prompt snapshot allocation failed");
    require(promptSnapshot->bytes() <= predictedPromptSnapshotBytes,
            "prompt snapshot exceeded preflight prediction");

    const uint64_t beforeFirstDecode = backend.submissionCount();
    ModelStepResult decoded =
        decodeOne(executor, 1, 0, 128, pageTable, BatchCohort::Greedy);
    require(backend.submissionCount() == beforeFirstDecode + 1,
            "speculative verify and commit were not one Metal command");
    require(!decoded.outputTokens.empty(), "decode produced no tokens");
    require(states.metadata(0).lengths.targetTokens ==
                    128 + decoded.outputTokens.size() &&
                states.metadata(0).lengths.hasCompleteDraftWindow(
                    states.layout().draft.tokens),
            "decode committed length mismatch");

    std::cout << "TOKENS";
    for (uint32_t token : decoded.outputTokens) {
      std::cout << ' ' << token;
    }
    std::cout << "\naccepted=" << decoded.acceptedDraftTokens
              << " drafted=" << decoded.draftedTokens
              << " target_length=" << states.metadata(0).lengths.targetTokens
              << '\n';
    const model::ModelTelemetry telemetry = executor.telemetry();
    std::cout << "prefill_wall_seconds=" << telemetry.lastPrefillWallSeconds
              << " decode_cycle_wall_seconds="
              << telemetry.lastDecodeWallSeconds << '\n';

    executor.end(1);
    EngineRequest reusedId = makeRequest(1, {1}, 1);
    beginCold(executor, reusedId, 0);
    executor.end(1);

    // Direct score-only prefill: raw final-position logits, no sampling or
    // decode. The greedy first generated token must be the max among options.
    {
      const uint32_t greedy = decoded.outputTokens.front();
      const uint32_t otherA = greedy == 1 ? 2u : 1u;
      const uint32_t otherB = greedy == 7 ? 8u : 7u;
      EngineRequest scored = makeRequest(99, prompt128, 0);
      scored.scoreTokens = {greedy, otherA, otherB};
      scored.imagePixels = {0};
      bool pixelsRejected = false;
      try {
        beginCold(executor, scored, 0);
      } catch (const std::invalid_argument &) {
        pixelsRejected = true;
      }
      require(pixelsRejected, "score request accepted image pixels without spans");
      scored.imagePixels.clear();
      beginCold(executor, scored, 0);
      ModelStepResult scoredResult =
          prefillChunk(executor, 99, 0, 0, 0, prompt128, pageTable);
      require(scoredResult.finished && scoredResult.outputTokens.empty() &&
                  scoredResult.scoreLogits.size() == 3,
              "score prefill did not return ordered logits without tokens");
      for (float logit : scoredResult.scoreLogits)
        require(std::isfinite(logit), "score logit is not finite");
      require(scoredResult.scoreLogits[0] >= scoredResult.scoreLogits[1] &&
                  scoredResult.scoreLogits[0] >= scoredResult.scoreLogits[2],
              "greedy decode token is not the maximum scored logit");
      bool decodeRejected = false;
      try {
        decodeOne(executor, 99, 0, 128, pageTable, BatchCohort::Greedy);
      } catch (const std::exception &) {
        decodeRejected = true;
      }
      require(decodeRejected, "score request allowed a decode step");
      executor.end(99);
    }


    // Compare the active GDN state from one 16-row chunk and two M8 commits
    // within the numerical tolerance below. Their next-token decisions are
    // diagnostic because the command partitions round differently.
    const std::vector<uint32_t> prompt16{
        248045, 8678, 198,   24342, 286,  4879, 369,   716,
        310,    830,  11553, 13,    5044, 1683, 15060, 1472};
    const std::vector<uint32_t> baselinePages{8};
    const std::vector<uint32_t> partitionedPages{9};
    EngineRequest chunkBaseline = makeRequest(50, prompt16, 2);
    EngineRequest partitioned = makeRequest(51, prompt16, 2);
    beginCold(executor, chunkBaseline, 2);
    ModelStepResult baselinePrefill =
        prefillChunk(executor, 50, 2, 0, 0, prompt16, baselinePages);
    beginCold(executor, partitioned, 3);
    prefillChunk(executor, 51, 3, 0, 0,
                 std::span<const uint32_t>(prompt16).first(8),
                 partitionedPages);
    ModelStepResult partitionedPrefill =
        prefillChunk(executor, 51, 3, 8, 8,
                     std::span<const uint32_t>(prompt16).subspan(8, 8),
                     partitionedPages);

    const model::QwenSlotMetadata &baselineMetadata = states.metadata(2);
    const model::QwenSlotMetadata &partitionedMetadata = states.metadata(3);
    require(baselineMetadata.lengths == partitionedMetadata.lengths &&
                baselineMetadata.lengths.targetTokens == prompt16.size(),
            "partitioned prefill logical state diverged");
    const model::QwenSlotBuffers &baselineState = states.buffers(2);
    const model::QwenSlotBuffers &partitionedState = states.buffers(3);
    const Similarity convolution = compareBfloat(
        baselineState.gdn[baselineMetadata.activeParity].convolutionBase,
        partitionedState.gdn[partitionedMetadata.activeParity].convolutionBase);
    const Similarity recurrent = compareFloat(
        baselineState.gdn[baselineMetadata.activeParity].recurrentBase,
        partitionedState.gdn[partitionedMetadata.activeParity].recurrentBase);
    std::cout
        << "partition_equivalence conv_cos=" << convolution.cosine
        << " parity=" << baselineMetadata.activeParity << '/'
        << partitionedMetadata.activeParity
        << " conv_norms=" << convolution.leftNorm << '/'
        << convolution.rightNorm << " partition_other_conv_norm="
        << compareBfloat(
               baselineState.gdn[baselineMetadata.activeParity].convolutionBase,
               partitionedState.gdn[partitionedMetadata.activeParity ^ 1]
                   .convolutionBase)
               .rightNorm
        << " recurrent_cos=" << recurrent.cosine
        << " recurrent_norms=" << recurrent.leftNorm << '/'
        << recurrent.rightNorm << '\n';
    require(convolution.cosine > 0.999 && recurrent.cosine > 0.999,
            "partitioned prefill numerical state diverged");

    ModelStepResult baselinePending =
        firstStep(executor, std::move(baselinePrefill), 50, 2,
                  prompt16.size(), baselinePages, BatchCohort::Greedy);
    ModelStepResult partitionedPending =
        firstStep(executor, std::move(partitionedPrefill), 51, 3,
                  prompt16.size(), partitionedPages, BatchCohort::Greedy);
    std::cout << "partition_decisions tokens_equal="
              << (baselinePending.outputTokens == partitionedPending.outputTokens)
              << " baseline_rows=" << baselinePending.outputTokens.size()
              << " partitioned_rows=" << partitionedPending.outputTokens.size()
              << " accepted=" << baselinePending.acceptedDraftTokens << '/'
              << partitionedPending.acceptedDraftTokens << '\n';
    executor.end(50);
    executor.end(51);

    // A reusable composite state contains no final hidden. A fresh
    // consumer must replay at least one complete input token; that replay
    // regenerates target hidden, performs the matching draft injection, and
    // selects the consumer's policy-specific anchor.
    std::vector<uint32_t> promptAligned(128);
    for (uint32_t index = 0; index < promptAligned.size(); ++index)
      promptAligned[index] = prompt16[index % prompt16.size()];
    std::vector<uint32_t> policyReplayPrompt = promptAligned;
    policyReplayPrompt.push_back(prompt16.front());
    const std::vector<uint32_t> policyPages = pageRange(10, 5);
    EngineRequest partitionedSampling =
        makeRequest(54, promptAligned, 1, BatchCohort::Sampling);
    partitionedSampling.sampling = {4.0F, 1.0F, 32, 8128};
    beginCold(executor, partitionedSampling, 0);
    prefillChunk(executor, 54, 0, 0, 0,
                 std::span<const uint32_t>(promptAligned).first(120),
                 policyPages, BatchCohort::Sampling);
    prefillChunk(executor, 54, 0, 120, 120,
                 std::span<const uint32_t>(promptAligned).subspan(120, 8),
                 policyPages, BatchCohort::Sampling);
    std::shared_ptr<const CompositeState> partitionedSamplingSnapshot =
        executor.snapshot(54);
    require(partitionedSamplingSnapshot != nullptr,
            "partitioned sampling snapshot allocation failed");
    executor.end(54);
    EngineRequest restoredMicroSampling =
        makeRequest(55, policyReplayPrompt, 2, BatchCohort::Sampling);
    restoredMicroSampling.sampling = partitionedSampling.sampling;
    beginCold(executor, restoredMicroSampling, 0);
    restoreActivePrefix(executor, 55, policyReplayPrompt.size(),
                        promptAligned.size(), partitionedSamplingSnapshot);
    ModelStepResult restoredMicroSample = firstStep(
        executor,
        prefillChunk(executor, 55, 0, promptAligned.size(),
                     promptAligned.size(),
                     std::span<const uint32_t>(policyReplayPrompt).subspan(128,
                                                                          1),
                     policyPages, BatchCohort::Sampling),
        55, 0, policyReplayPrompt.size(), policyPages, BatchCohort::Sampling);
    executor.end(55);
    const std::vector<uint32_t> coldPolicyPages = pageRange(15, 5);
    EngineRequest coldMicroSampling = restoredMicroSampling;
    coldMicroSampling.id = 57;
    beginCold(executor, coldMicroSampling, 0);
    prefillChunk(executor, 57, 0, 0, 0,
                 std::span<const uint32_t>(policyReplayPrompt).first(120),
                 coldPolicyPages, BatchCohort::Sampling);
    prefillChunk(executor, 57, 0, 120, 120,
                 std::span<const uint32_t>(policyReplayPrompt).subspan(120, 8),
                 coldPolicyPages, BatchCohort::Sampling);
    ModelStepResult coldMicroSample = firstStep(
        executor,
        prefillChunk(executor, 57, 0, 128, 128,
                     std::span<const uint32_t>(policyReplayPrompt).subspan(128,
                                                                          1),
                     coldPolicyPages, BatchCohort::Sampling),
        57, 0, policyReplayPrompt.size(), coldPolicyPages,
        BatchCohort::Sampling);
    require(restoredMicroSample.outputTokens == coldMicroSample.outputTokens,
            "one-token recurrent restore replay diverged from cold prefill");
    executor.end(57);
    partitionedSamplingSnapshot.reset();

    const std::vector<uint32_t> prompt8(prompt16.begin(), prompt16.begin() + 8);
    const std::vector<uint32_t> constrainedPages{20};
    EngineRequest constrainedShort =
        makeRequest(56, prompt8, 1, BatchCohort::Constrained);
    constrainedShort.constraint = ConstraintMode::TokenMask;
    beginCold(executor, constrainedShort, 0);
    prefillChunk(executor, 56, 0, 0, 0, prompt8, constrainedPages,
                 BatchCohort::Constrained);
    ModelStepResult microInitialMask =
        decodeOne(executor, 56, 0, prompt8.size(), constrainedPages,
                  BatchCohort::Constrained, DecodeStage::RequestInitialMask);
    require(microInitialMask.nextDecodeStage == DecodeStage::ApplyInitialMask,
            "constrained short prefill did not preserve mask handshake");
    const std::array<uint32_t, 1> forcedMicroToken{106};
    executor.provideMask(56, singletonMasks(forcedMicroToken));
    ModelStepResult forcedMicro =
        decodeOne(executor, 56, 0, prompt8.size(), constrainedPages,
                  BatchCohort::Constrained, DecodeStage::ApplyInitialMask);
    require(forcedMicro.outputTokens ==
                std::vector<uint32_t>{forcedMicroToken.front()},
            "constrained short prefill final hidden is unusable");
    executor.end(56);

    // Output limits only change the token-exact commit count. Every executable
    // decode still runs one anchor plus seven proposal rows.  Compare each
    // retained prefix against a teacher-forced replay of precisely the tokens
    // exposed by the commit cursor; rejected speculative rows must be absent
    // from both the active GDN state and the persistent draft ring. A final
    // budgeted token is emitted without a KV row and is not part of the
    // committed prefix (a one-token budget is emitted by prefill itself).
    for (uint32_t outputLimit = 2; outputLimit <= 8; ++outputLimit) {
      uint64_t id = 10 + outputLimit;
      EngineRequest variant = makeRequest(id, prompt129, outputLimit);
      beginCold(executor, variant, 0);
      restoreActivePrefix(executor, id, prompt129.size(), 128, promptSnapshot);
      ModelStepResult result = firstStep(
          executor,
          prefillChunk(executor, id, 0, 128, 128,
                       std::span<const uint32_t>(prompt129).subspan(128, 1),
                       pageTable),
          id, 0, 129, pageTable, BatchCohort::Greedy);
      require(result.draftedTokens == 7,
              "oracle prompt ended right after prefill; no cycle to test");
      require(!result.outputTokens.empty() &&
                  result.outputTokens.size() <= outputLimit,
              "fixed DFlash-8 decode exceeded its output limit");
      require(result.acceptedDraftTokens <= 7,
              "fixed DFlash-8 acceptance accounting mismatch");
      const size_t stored =
          result.outputTokens.size() - result.outputTokensWithoutKv;
      require(stored >= 1 && states.metadata(0).lengths.targetTokens ==
                                 129 + stored,
              "fixed DFlash-8 state length mismatch");

      std::vector<uint32_t> replayPrompt = prompt129;
      replayPrompt.insert(replayPrompt.end(), result.outputTokens.begin(),
                          result.outputTokens.begin() + stored);
      const uint64_t replayId = 100 + outputLimit;
      // The composite state deliberately excludes target KV.  A valid replay
      // therefore shares the four sealed Page32 history blocks and owns only
      // its writable block; using an unrelated empty page table would test a
      // corrupt hybrid restore rather than GDN commit equivalence.
      std::vector<uint32_t> replayPages = pageTable;
      replayPages[4] = 80;
      beginCold(executor, makeRequest(replayId, replayPrompt, 1), 1);
      restoreActivePrefix(executor, replayId, replayPrompt.size(), 128,
                          promptSnapshot);
      prefillChunk(executor, replayId, 1, 128, 128,
                   std::span<const uint32_t>(replayPrompt).subspan(128, 1),
                   replayPages);
      prefillChunk(executor, replayId, 1, 129, 129,
                   std::span<const uint32_t>(replayPrompt).subspan(129, stored),
                   replayPages);
      requireCommittedStateEquivalent(
          states, 0, 1, "fixed DFlash-8 retained=" + std::to_string(stored));
      executor.end(replayId);
      executor.end(id);
    }

    // A Page32 composite state followed by one replayed token must equal a cold
    // run with the same 128+1 command partition.
    EngineRequest extended = makeRequest(20, prompt129, 2);
    beginCold(executor, extended, 0);
    restoreActivePrefix(executor, 20, prompt129.size(), 128, promptSnapshot);
    ModelStepResult extendedResult = firstStep(
        executor,
        prefillChunk(executor, 20, 0, 128, 128,
                     std::span<const uint32_t>(prompt129).subspan(128, 1),
                     pageTable),
        20, 0, 129, pageTable, BatchCohort::Greedy);
    require(!extendedResult.outputTokens.empty() &&
                states.metadata(0).lengths.targetTokens ==
                    129 + extendedResult.outputTokens.size() -
                        extendedResult.outputTokensWithoutKv,
            "one-token replay boundary failed");
    executor.end(20);

    const std::vector<uint32_t> coldReplayPages = pageRange(21, 5);
    EngineRequest coldExtended = makeRequest(21, prompt129, 2);
    beginCold(executor, coldExtended, 0);
    prefillChunk(executor, 21, 0, 0, 0,
                 std::span<const uint32_t>(prompt129).first(128),
                 coldReplayPages);
    ModelStepResult coldExtendedResult = firstStep(
        executor,
        prefillChunk(executor, 21, 0, 128, 128,
                     std::span<const uint32_t>(prompt129).subspan(128, 1),
                     coldReplayPages),
        21, 0, 129, coldReplayPages, BatchCohort::Greedy);
    require(coldExtendedResult.outputTokens == extendedResult.outputTokens,
            "teacher-forced cached suffix diverged from cold prompt");
    executor.end(21);

    // Recurrent cache entries are policy-neutral. The consumer replays one
    // teacher-forced token, then selects from the regenerated final hidden
    // with its own sampling stream.
    std::vector<uint32_t> samplingPrefix(128);
    for (uint32_t index = 0; index < samplingPrefix.size(); ++index) {
      samplingPrefix[index] =
          samplingSeedTokens[index % samplingSeedTokens.size()];
    }
    std::vector<uint32_t> samplingPrompt = samplingPrefix;
    samplingPrompt.push_back(samplingSeedTokens.front());
    const std::vector<uint32_t> samplingPages = pageRange(26, 5);
    EngineRequest samplingSource =
        makeRequest(30, samplingPrefix, 1, BatchCohort::Sampling);
    samplingSource.sampling = {4.0F, 1.0F, 32, 40106};
    beginCold(executor, samplingSource, 0);
    prefillChunk(executor, 30, 0, 0, 0,
                 std::span<const uint32_t>(samplingPrefix).first(120),
                 samplingPages, BatchCohort::Sampling);
    prefillChunk(executor, 30, 0, 120, 120,
                 std::span<const uint32_t>(samplingPrefix).subspan(120, 8),
                 samplingPages, BatchCohort::Sampling);
    std::shared_ptr<const CompositeState> samplingPromptSnapshot =
        executor.snapshot(30);
    require(samplingPromptSnapshot != nullptr,
            "sampling snapshot allocation failed");
    executor.end(30);

    EngineRequest replayedSampling =
        makeRequest(31, samplingPrompt, 2, BatchCohort::Sampling);
    replayedSampling.sampling = {4.0F, 1.0F, 32, 91199};
    beginCold(executor, replayedSampling, 0);
    restoreActivePrefix(executor, 31, samplingPrompt.size(), 128,
                        samplingPromptSnapshot);
    ModelStepResult replayedSample = firstStep(
        executor,
        prefillChunk(executor, 31, 0, 128, 128,
                     std::span<const uint32_t>(samplingPrompt).subspan(128, 1),
                     samplingPages, BatchCohort::Sampling),
        31, 0, samplingPrompt.size(), samplingPages, BatchCohort::Sampling);
    require(!replayedSample.outputTokens.empty(),
            "replayed sampling hit did not emit an anchor");
    executor.end(31);

    const std::vector<uint32_t> coldSamplingPages = pageRange(31, 5);
    EngineRequest coldSampling = replayedSampling;
    coldSampling.id = 32;
    beginCold(executor, coldSampling, 0);
    prefillChunk(executor, 32, 0, 0, 0,
                 std::span<const uint32_t>(samplingPrompt).first(120),
                 coldSamplingPages, BatchCohort::Sampling);
    prefillChunk(executor, 32, 0, 120, 120,
                 std::span<const uint32_t>(samplingPrompt).subspan(120, 8),
                 coldSamplingPages, BatchCohort::Sampling);
    ModelStepResult coldSample = firstStep(
        executor,
        prefillChunk(executor, 32, 0, 128, 128,
                     std::span<const uint32_t>(samplingPrompt).subspan(128, 1),
                     coldSamplingPages, BatchCohort::Sampling),
        32, 0, samplingPrompt.size(), coldSamplingPages,
        BatchCohort::Sampling);
    require(coldSample.outputTokens == replayedSample.outputTokens,
            "sampling restore replay reused producer policy state");
    executor.end(32);
    samplingPromptSnapshot.reset();

    // The constraint handshake exposes the pending anchor plus all seven
    // DFlash proposals. The next mask therefore has nine rows: the current
    // anchor, seven proposal positions, and one target successor.
    EngineRequest constrained =
        makeRequest(40, prompt129, 2, BatchCohort::Constrained);
    constrained.constraint = ConstraintMode::TokenMask;
    beginCold(executor, constrained, 0);
    restoreActivePrefix(executor, 40, prompt129.size(), 128, promptSnapshot);
    prefillChunk(executor, 40, 0, 128, 128,
                 std::span<const uint32_t>(prompt129).subspan(128, 1),
                 pageTable, BatchCohort::Constrained);
    ModelStepResult initialMask =
        decodeOne(executor, 40, 0, 129, pageTable, BatchCohort::Constrained,
                  DecodeStage::RequestInitialMask);
    require(initialMask.nextDecodeStage == DecodeStage::ApplyInitialMask,
            "initial constrained anchor did not request empty simulation");
    const uint32_t anchorA = 100;
    std::array<uint32_t, 1> initialTokens{anchorA};
    std::vector<uint32_t> initialWords = singletonMasks(initialTokens);
    require(initialWords.size() == kMaskWords,
            "empty simulation did not produce exactly one mask row");
    executor.provideMask(40, initialWords);

    PendingMaskedDecode verify = beginMaskedDecodeOne(
        executor, 40, 0, 129, pageTable, DecodeStage::ApplyInitialMask);
    require(verify.maskRequests.size() == 1 &&
                verify.maskRequests[0].requestId == 40 &&
                verify.maskRequests[0].simulationTokens.size() == 8 &&
                verify.maskRequests[0].simulationTokens.front() == anchorA &&
                verify.ticket->ownsMaskWait(40),
            "constraint proposals were not exposed during target forward");
    uint32_t nextB =
        verify.maskRequests[0].simulationTokens[1] == 101 ? 102 : 101;
    std::array<uint32_t, 9> verifyTokens{anchorA, nextB, 103, 104, 105,
                                         106,     107,   108, 109};
    std::vector<uint32_t> verifyWords = singletonMasks(verifyTokens);
    require(verifyWords.size() == uint64_t{9} * kMaskWords,
            "DFlash-8 verify did not produce nine mask rows");
    executor.provideMask(40, verifyWords);
    auto constrainedResults = finishMaskedDecode(std::move(verify));
    require(constrainedResults.size() == 1,
            "constrained overlap returned the wrong batch width");
    // The rejected proposal makes the masked successor the next anchor; as
    // the last budgeted token it is emitted at once, without a KV row.
    ModelStepResult constrainedFirst = std::move(constrainedResults[0]);
    require(constrainedFirst.nextDecodeStage == DecodeStage::Regular &&
                constrainedFirst.outputTokens ==
                    std::vector<uint32_t>{anchorA, nextB} &&
                constrainedFirst.outputTokensWithoutKv == 1 &&
                !constrainedFirst.finished &&
                constrainedFirst.draftedTokens == 7 &&
                constrainedFirst.acceptedDraftTokens == 0,
            "constrained verify mask offset is incorrect");
    executor.end(40);

    // Accept two of seven proposals and reject the third. The draft work is
    // always seven rows and is counted only after target verification.
    EngineRequest perfectConstraint =
        makeRequest(44, prompt129, 4, BatchCohort::Constrained);
    perfectConstraint.constraint = ConstraintMode::TokenMask;
    beginCold(executor, perfectConstraint, 0);
    restoreActivePrefix(executor, 44, prompt129.size(), 128, promptSnapshot);
    prefillChunk(executor, 44, 0, 128, 128,
                 std::span<const uint32_t>(prompt129).subspan(128, 1),
                 pageTable, BatchCohort::Constrained);
    ModelStepResult perfectInitial =
        decodeOne(executor, 44, 0, 129, pageTable, BatchCohort::Constrained,
                  DecodeStage::RequestInitialMask);
    require(perfectInitial.nextDecodeStage == DecodeStage::ApplyInitialMask,
            "perfect constrained accounting skipped its initial mask");
    const uint32_t perfectAnchor = 120;
    std::array<uint32_t, 1> perfectInitialTokens{perfectAnchor};
    executor.provideMask(44, singletonMasks(perfectInitialTokens));
    PendingMaskedDecode perfectPending = beginMaskedDecodeOne(
        executor, 44, 0, 129, pageTable, DecodeStage::ApplyInitialMask);
    require(perfectPending.maskRequests.size() == 1 &&
                perfectPending.maskRequests[0].simulationTokens.size() == 8,
            "perfect constraint did not overlap its mask request");
    uint32_t rejected =
        perfectPending.maskRequests[0].simulationTokens[3] == 121 ? 122 : 121;
    std::array<uint32_t, 9> perfectVerify{perfectAnchor,
                                          perfectPending.maskRequests[0]
                                              .simulationTokens[1],
                                          perfectPending.maskRequests[0]
                                              .simulationTokens[2],
                                          rejected,
                                          123,
                                          124,
                                          125,
                                          126,
                                          127};
    executor.provideMask(44, singletonMasks(perfectVerify));
    auto perfectResults = finishMaskedDecode(std::move(perfectPending));
    require(perfectResults.size() == 1,
            "perfect constrained overlap returned the wrong width");
    ModelStepResult perfectResult = std::move(perfectResults[0]);
    require(perfectResult.nextDecodeStage == DecodeStage::Regular &&
                perfectResult.outputTokens.size() == 4 &&
                perfectResult.outputTokensWithoutKv == 1 &&
                perfectResult.draftedTokens == 7 &&
                perfectResult.acceptedDraftTokens == 2,
            "constrained acceptance did not report two of seven");
    executor.end(44);

    EngineRequest alternateConstraint =
        makeRequest(41, prompt129, 1, BatchCohort::Constrained);
    alternateConstraint.constraint = ConstraintMode::TokenMask;
    beginCold(executor, alternateConstraint, 0);
    restoreActivePrefix(executor, 41, prompt129.size(), 128, promptSnapshot);
    prefillChunk(executor, 41, 0, 128, 128,
                 std::span<const uint32_t>(prompt129).subspan(128, 1),
                 pageTable, BatchCohort::Constrained);
    ModelStepResult alternateInitial =
        decodeOne(executor, 41, 0, 129, pageTable, BatchCohort::Constrained,
                  DecodeStage::RequestInitialMask);
    require(alternateInitial.nextDecodeStage == DecodeStage::ApplyInitialMask,
            "second exact constrained hit skipped initial mask");
    const uint32_t anchorC = 105;
    std::array<uint32_t, 1> alternateTokens{anchorC};
    executor.provideMask(41, singletonMasks(alternateTokens));
    ModelStepResult alternateOutput =
        decodeOne(executor, 41, 0, 129, pageTable, BatchCohort::Constrained,
                  DecodeStage::ApplyInitialMask);
    require(!alternateOutput.finished &&
                alternateOutput.outputTokensWithoutKv == 1 &&
                alternateOutput.outputTokens ==
                    std::vector<uint32_t>{anchorC} &&
                anchorC != anchorA,
            "exact prefix reused the producer constraint decision");
    executor.end(41);

    // A B2 constrained cycle keeps both lanes reserved while host grammar
    // work overlaps the target forward. No proposal/logit state is copied to
    // a different arena lane between draft and commit.
    EngineRequest crossLane0 =
        makeRequest(42, prompt129, 2, BatchCohort::Constrained);
    crossLane0.constraint = ConstraintMode::TokenMask;
    crossLane0.sampling = {4.0F, 1.0F, 32, 7001};
    EngineRequest crossLane1 = crossLane0;
    crossLane1.id = 43;
    crossLane1.sampling.seed = 7002;
    beginCold(executor, crossLane0, 0);
    restoreActivePrefix(executor, 42, prompt129.size(), 128, promptSnapshot);
    beginCold(executor, crossLane1, 1);
    restoreActivePrefix(executor, 43, prompt129.size(), 128, promptSnapshot);
    const std::vector<uint32_t> crossPages0{0, 1, 2, 3, 36};
    const std::vector<uint32_t> crossPages1{0, 1, 2, 3, 37};
    BatchPlan crossReplayPlan{WorkKind::Prefill,
                              BatchCohort::Constrained,
                              {{42, 1}, {43, 1}},
                              DecodeStage::Regular};
    std::array<ModelBatchItem, 2> crossReplayItems{
        ModelBatchItem{42, 0, 128, 128, 1, crossPages0},
        ModelBatchItem{43, 1, 128, 128, 1, crossPages1}};
    const auto crossReplayToken =
        std::span<const uint32_t>(prompt129).subspan(128, 1);
    crossReplayItems[0].inputTokens = crossReplayToken;
    crossReplayItems[1].inputTokens = crossReplayToken;
    auto crossReplay = executor.prefill(crossReplayPlan, crossReplayItems);
    require(crossReplay.size() == 2 &&
                crossReplay[0].consumedPromptTokens == 1 &&
                crossReplay[1].consumedPromptTokens == 1,
            "B2 recurrent restore did not replay one complete token");
    BatchPlan crossInitialPlan{WorkKind::Decode,
                               BatchCohort::Constrained,
                               {{42, 0}, {43, 0}},
                               DecodeStage::RequestInitialMask};
    std::vector<ModelBatchItem> crossItems{{42, 0, 129, 0, 0, crossPages0},
                                              {43, 1, 129, 0, 0, crossPages1}};
    auto crossInitial = executor.decode(crossInitialPlan, crossItems);
    require(
        crossInitial.size() == 2 &&
            crossInitial[0].nextDecodeStage == DecodeStage::ApplyInitialMask &&
            crossInitial[1].nextDecodeStage == DecodeStage::ApplyInitialMask,
        "B2 constrained initial masks are not empty simulations");
    std::array<uint32_t, 1> crossAnchor0{110};
    std::array<uint32_t, 1> crossAnchor1{111};
    executor.provideMask(42, singletonMasks(crossAnchor0));
    executor.provideMask(43, singletonMasks(crossAnchor1));
    crossInitialPlan.decodeStage = DecodeStage::ApplyInitialMask;
    PendingMaskedDecode crossPending =
        beginMaskedDecode(executor, crossInitialPlan, crossItems);
    require(crossPending.maskRequests.size() == 2 &&
                crossPending.ticket->ownsMaskWait(42) &&
                crossPending.ticket->ownsMaskWait(43),
            "B2 target forward did not own both constraint masks");
    std::array<std::array<uint32_t, 9>, 2> crossVerify{};
    for (uint32_t lane = 0; lane < 2; ++lane) {
      const auto &simulation =
          crossPending.maskRequests[lane].simulationTokens;
      const uint32_t anchor = lane ? crossAnchor1[0] : crossAnchor0[0];
      const uint32_t rejected = simulation[1] == 112 ? 113 : 112;
      crossVerify[lane] =
          {anchor, rejected, 114, 115, 116, 117, 118, 119, 120};
      executor.provideMask(crossPending.maskRequests[lane].requestId,
                           singletonMasks(crossVerify[lane]));
    }
    auto crossResults = finishMaskedDecode(std::move(crossPending));
    require(crossResults.size() == 2 &&
                crossResults[0].outputTokens ==
                    std::vector<uint32_t>{crossAnchor0[0], crossVerify[0][1]} &&
                crossResults[1].outputTokens ==
                    std::vector<uint32_t>{crossAnchor1[0], crossVerify[1][1]} &&
                crossResults[0].outputTokensWithoutKv == 1 &&
                crossResults[1].outputTokensWithoutKv == 1 &&
                crossResults[0].draftedTokens == 7 &&
                crossResults[1].draftedTokens == 7 &&
                crossResults[0].acceptedDraftTokens == 0 &&
                crossResults[1].acceptedDraftTokens == 0,
            "B2 constrained overlap corrupted a lane");
    const model::ModelTelemetry constrainedTelemetry =
        executor.telemetry();
    require(constrainedTelemetry.constrainedMaskOverlapBatches == 3 &&
                constrainedTelemetry.constrainedMaskOverlapRequests == 4 &&
                constrainedTelemetry.totalConstrainedTargetForwardGpuSeconds >
                    0.0,
            "constrained overlap telemetry does not match B1/B2 execution");
    executor.end(42);
    executor.end(43);

    // Width three is a real M24 graph, never a B2+B1 decomposition or a
    // rendezvous for a fourth request. Physical state slots are intentionally
    // permuted to prove that batch lanes belong to plan order.
    constexpr std::array<uint64_t, 3> b3Ids{60, 61, 62};
    constexpr std::array<uint32_t, 3> b3Slots{2, 0, 3};
    std::array<std::vector<uint32_t>, 3> b3Pages{std::vector<uint32_t>{40},
                                                 std::vector<uint32_t>{41},
                                                 std::vector<uint32_t>{42}};
    constexpr std::array<uint32_t, 3> b3Words{279, 314, 264};
    for (uint32_t lane = 0; lane < b3Ids.size(); ++lane) {
      beginCold(executor, makeRequest(b3Ids[lane], {b3Words[lane]}, 16),
                b3Slots[lane]);
      prefillToken(executor, b3Ids[lane], b3Slots[lane], 0, 0, b3Words[lane],
                   b3Pages[lane]);
    }
    BatchPlan b3Plan{WorkKind::Decode,
                     BatchCohort::Greedy,
                     {{b3Ids[0], 0}, {b3Ids[1], 0}, {b3Ids[2], 0}},
                     DecodeStage::Regular};
    std::array<ModelBatchItem, 3> b3Items{
        ModelBatchItem{b3Ids[0], b3Slots[0], 1, 0, 0, b3Pages[0]},
        ModelBatchItem{b3Ids[1], b3Slots[1], 1, 0, 0, b3Pages[1]},
        ModelBatchItem{b3Ids[2], b3Slots[2], 1, 0, 0, b3Pages[2]}};
    auto b3Decoded = executor.decode(b3Plan, b3Items);
    const model::ModelTelemetry b3Telemetry = executor.telemetry();
    require(b3Decoded.size() == 3 && !b3Decoded[0].outputTokens.empty() &&
                !b3Decoded[1].outputTokens.empty() &&
                !b3Decoded[2].outputTokens.empty() &&
                b3Telemetry.lastDecodeWidth == 3 &&
                b3Telemetry.lastDecodeM16Dispatches == 0 &&
                b3Telemetry.lastDecodeM32Dispatches == 0 &&
                b3Telemetry.lastDecodeM24Dispatches > 0,
            "B3 projection graph was decomposed instead of using M24");
    for (uint64_t id : b3Ids)
      executor.end(id);

    // Identical inputs within one B4 graph must make identical decisions.
    // B1 uses a different numerical graph, so its decisions are diagnostic.
    constexpr std::array<uint64_t, 4> equivalentIds{64, 65, 66, 67};
    std::array<std::vector<uint32_t>, 4> equivalentPages{
        std::vector<uint32_t>{43}, std::vector<uint32_t>{44},
        std::vector<uint32_t>{45}, std::vector<uint32_t>{46}};
    for (uint32_t lane = 0; lane < equivalentIds.size(); ++lane) {
      beginCold(executor, makeRequest(equivalentIds[lane], {279}, 16), lane);
      prefillToken(executor, equivalentIds[lane], lane, 0, 0, 279,
                   equivalentPages[lane]);
    }
    BatchPlan equivalentPlan;
    equivalentPlan.kind = WorkKind::Decode;
    equivalentPlan.cohort = BatchCohort::Greedy;
    std::array<ModelBatchItem, 4> equivalentItems;
    for (uint32_t lane = 0; lane < equivalentIds.size(); ++lane) {
      equivalentPlan.items.push_back({equivalentIds[lane], 0});
      equivalentItems[lane] = {equivalentIds[lane],  lane, 1, 0, 0,
                               equivalentPages[lane]};
    }
    auto equivalentB4 = executor.decode(equivalentPlan, equivalentItems);
    for (uint64_t id : equivalentIds)
      executor.end(id);
    beginCold(executor, makeRequest(68, {279}, 16), 0);
    prefillToken(executor, 68, 0, 0, 0, 279, {47});
    ModelStepResult equivalentB1 =
        decodeOne(executor, 68, 0, 1, {47}, BatchCohort::Greedy);
    require(equivalentB4.size() == 4, "B4 equivalence width mismatch");
    for (const ModelStepResult &lane : equivalentB4) {
      require(!lane.outputTokens.empty() && lane.outputTokens.size() <= 16 &&
                  lane.acceptedDraftTokens <= lane.draftedTokens &&
                  lane.outputTokensWithoutKv <= lane.outputTokens.size(),
              "B4 speculative output accounting is invalid");
      const auto &first = equivalentB4.front();
      require(lane.outputTokens == first.outputTokens &&
                  lane.acceptedDraftTokens == first.acceptedDraftTokens &&
                  lane.draftedTokens == first.draftedTokens &&
                  lane.outputTokensWithoutKv == first.outputTokensWithoutKv &&
                  lane.finished == first.finished &&
                  lane.nextDecodeStage == first.nextDecodeStage,
              "identical B4 lanes made different speculative decisions");
    }
    require(!equivalentB1.outputTokens.empty() &&
                equivalentB1.outputTokens.size() <= 16 &&
                equivalentB1.acceptedDraftTokens <= equivalentB1.draftedTokens &&
                equivalentB1.outputTokensWithoutKv <= equivalentB1.outputTokens.size(),
            "B1 speculative output accounting is invalid");
    std::cout << "batch_decisions b4_tokens_equal_b1="
              << (equivalentB4.front().outputTokens == equivalentB1.outputTokens)
              << " accepted=" << equivalentB4.front().acceptedDraftTokens << '/'
              << equivalentB1.acceptedDraftTokens
              << " rows=" << equivalentB4.front().outputTokens.size() << '/'
              << equivalentB1.outputTokens.size() << '\n';
    executor.end(68);

    const std::vector<uint32_t> productionSeedTokens{
        248045, 846,   198,    2427,  38453, 494,    220, 16,     11,  4237,
        1754,   1324,  321,    1141,  6163,  803,    383, 264,    491, 1500,
        13,     14569, 2980,   488,   5372,  220,    17,  15,     15,  13,
        248046, 198,   248045, 74455, 198,   248068, 271, 248069, 271};
    std::vector<uint32_t> productionPrefix(128);
    for (uint32_t index = 0; index < productionPrefix.size(); ++index) {
      productionPrefix[index] =
          productionSeedTokens[index % productionSeedTokens.size()];
    }
    std::vector<uint32_t> productionPrompt = productionPrefix;
    productionPrompt.push_back(
        productionSeedTokens[productionPrefix.size() %
                             productionSeedTokens.size()]);
    beginCold(executor, makeRequest(70, productionPrefix, 16), 0);
    const std::vector<uint32_t> productionPages{48, 49, 50, 51};
    prefillChunk(executor, 70, 0, 0, 0, productionPrefix, productionPages);
    std::shared_ptr<const CompositeState> productionSnapshot =
        executor.snapshot(70);
    require(productionSnapshot != nullptr,
            "production snapshot allocation failed");
    executor.end(70);

    // One packed command consumes exactly 2048 real, unequal rows. Repeating
    // its M32 decode with permuted lanes proves ragged addressing and state
    // isolation without requiring another batch width's numerical decisions.
    constexpr std::array<uint64_t, 4> raggedIds{100, 101, 102, 103};
    constexpr std::array<uint32_t, 4> raggedSlots{3, 1, 0, 2};
    constexpr std::array<uint32_t, 4> raggedRows{1, 31, 257, 1759};
    std::array<std::vector<uint32_t>, 4> raggedPrompts;
    for (uint32_t lane = 0; lane < raggedIds.size(); ++lane) {
      raggedPrompts[lane].reserve(raggedRows[lane]);
      for (uint32_t row = 0; row < raggedRows[lane]; ++row) {
        raggedPrompts[lane].push_back(
            productionSeedTokens[(row + lane) % productionSeedTokens.size()]);
      }
    }
    std::array<std::vector<uint32_t>, 4> raggedPages{
        pageRange(52, 1), pageRange(53, 2), pageRange(55, 9),
        pageRange(64, 56)};
    const auto raggedRequest = [&](uint64_t id, uint32_t lane) {
      const bool sampled = lane % 2;
      auto value = makeRequest(
          id, raggedPrompts[lane], 16,
          sampled ? BatchCohort::Sampling : BatchCohort::Greedy);
      value.sampling = {sampled ? 0.8F : 0.0F, 0.95F, 20, 731 + lane};
      return value;
    };
    BatchPlan raggedPrefillPlan;
    raggedPrefillPlan.kind = WorkKind::Prefill;
    raggedPrefillPlan.cohort = BatchCohort::Greedy;
    std::array<ModelBatchItem, 4> raggedPrefillItems;
    for (uint32_t lane = 0; lane < raggedIds.size(); ++lane) {
      beginCold(executor,
          raggedRequest(raggedIds[lane], lane),
          raggedSlots[lane]);
      raggedPrefillPlan.items.push_back({raggedIds[lane], raggedRows[lane]});
      raggedPrefillItems[lane] = {raggedIds[lane],  raggedSlots[lane], 0, 0,
                                  raggedRows[lane], raggedPages[lane]};
      raggedPrefillItems[lane].inputTokens = raggedPrompts[lane];
    }
    const uint64_t beforeRaggedPrefill = backend.submissionCount();
    auto raggedPrefill =
        executor.prefill(raggedPrefillPlan, raggedPrefillItems);
    require(raggedPrefill.size() == raggedIds.size() &&
                backend.submissionCount() == beforeRaggedPrefill + 1,
            "ragged 2048-row prefill was not one Metal command");
    for (uint32_t lane = 0; lane < raggedIds.size(); ++lane) {
      require(raggedPrefill[lane].consumedPromptTokens == raggedRows[lane] &&
                  states.metadata(raggedSlots[lane]).lengths.targetTokens ==
                      raggedRows[lane],
              "ragged prefill consumed or addressed the wrong rows");
      requireOpen(raggedPrefill[lane], "ragged prefill");
    }

    BatchPlan raggedDecodePlan;
    raggedDecodePlan.kind = WorkKind::Decode;
    raggedDecodePlan.cohort = BatchCohort::Sampling;
    std::array<ModelBatchItem, 4> raggedDecodeItems;
    for (uint32_t lane = 0; lane < raggedIds.size(); ++lane) {
      raggedDecodePlan.items.push_back({raggedIds[lane], 0});
      raggedDecodeItems[lane] = {
          raggedIds[lane],  raggedSlots[lane], raggedRows[lane], 0, 0,
          raggedPages[lane]};
    }
    auto raggedDecoded = executor.decode(raggedDecodePlan, raggedDecodeItems);
    const model::ModelTelemetry raggedDecodeTelemetry =
        executor.telemetry();
    require(raggedDecoded.size() == raggedIds.size() &&
                raggedDecodeTelemetry.lastDecodeWidth == 4 &&
                raggedDecodeTelemetry.lastDecodeM16Dispatches == 0 &&
                raggedDecodeTelemetry.lastDecodeM24Dispatches == 0 &&
                raggedDecodeTelemetry.lastDecodeM32Dispatches > 0,
            "permuted ragged B4 was decomposed instead of using M32");
    for (uint64_t id : raggedIds)
      executor.end(id);

    // Re-run the same real M32 workload with request order and state slots
    // permuted. This isolates cross-lane addressing without conflating M32
    // with the independently optimized M8 numerical path.
    constexpr std::array<uint32_t, 4> raggedPermutation{2, 0, 3, 1};
    constexpr std::array<uint32_t, 4> referenceSlots{1, 3, 0, 2};
    BatchPlan raggedReferencePrefillPlan;
    raggedReferencePrefillPlan.kind = WorkKind::Prefill;
    raggedReferencePrefillPlan.cohort = BatchCohort::Greedy;
    std::array<ModelBatchItem, 4> raggedReferencePrefillItems;
    for (uint32_t order = 0; order < raggedPermutation.size(); ++order) {
      const uint32_t lane = raggedPermutation[order];
      const uint64_t referenceId = 104 + lane;
      beginCold(executor,
          raggedRequest(referenceId, lane),
          referenceSlots[order]);
      raggedReferencePrefillPlan.items.push_back(
          {referenceId, raggedRows[lane]});
      raggedReferencePrefillItems[order] = {
          referenceId, referenceSlots[order], 0,
          0,           raggedRows[lane],      raggedPages[lane]};
      raggedReferencePrefillItems[order].inputTokens = raggedPrompts[lane];
    }
    auto raggedReferencePrefill = executor.prefill(raggedReferencePrefillPlan,
                                                   raggedReferencePrefillItems);
    require(raggedReferencePrefill.size() == raggedPermutation.size(),
            "permuted ragged reference prefill width mismatch");

    BatchPlan raggedReferenceDecodePlan;
    raggedReferenceDecodePlan.kind = WorkKind::Decode;
    raggedReferenceDecodePlan.cohort = BatchCohort::Sampling;
    std::array<ModelBatchItem, 4> raggedReferenceDecodeItems;
    for (uint32_t order = 0; order < raggedPermutation.size(); ++order) {
      const uint32_t lane = raggedPermutation[order];
      const uint64_t referenceId = 104 + lane;
      raggedReferenceDecodePlan.items.push_back({referenceId, 0});
      raggedReferenceDecodeItems[order] = {
          referenceId, referenceSlots[order], raggedRows[lane], 0,
          0,           raggedPages[lane]};
    }
    auto raggedReferenceDecoded =
        executor.decode(raggedReferenceDecodePlan, raggedReferenceDecodeItems);
    require(raggedReferenceDecoded.size() == raggedPermutation.size() &&
                executor.telemetry().lastDecodeWidth == 4 &&
                executor.telemetry().lastDecodeM32Dispatches > 0,
            "permuted ragged reference was not one M32 graph");
    for (uint32_t order = 0; order < raggedPermutation.size(); ++order) {
      const uint32_t lane = raggedPermutation[order];
      const ModelStepResult &reference = raggedReferenceDecoded[order];
      require(reference.outputTokens == raggedDecoded[lane].outputTokens &&
                  reference.acceptedDraftTokens ==
                      raggedDecoded[lane].acceptedDraftTokens,
              "ragged M32 lane changed after order/slot permutation");
      executor.end(104 + lane);
    }

    constexpr std::array<uint64_t, 4> productionB4Ids{71, 72, 73, 74};
    std::array<std::vector<uint32_t>, 4> productionB4Pages{
        std::vector<uint32_t>{48, 49, 50, 51, 72},
        std::vector<uint32_t>{48, 49, 50, 51, 73},
        std::vector<uint32_t>{48, 49, 50, 51, 74},
        std::vector<uint32_t>{48, 49, 50, 51, 75}};
    for (uint32_t lane = 0; lane < productionB4Ids.size(); ++lane) {
      beginCold(executor,
          makeRequest(productionB4Ids[lane], productionPrompt, 16), lane);
      restoreActivePrefix(executor, productionB4Ids[lane],
                          productionPrompt.size(), productionPrefix.size(),
                          productionSnapshot);
    }
    BatchPlan productionB4ReplayPlan{WorkKind::Prefill,
                                     BatchCohort::Greedy,
                                     {{productionB4Ids[0], 1},
                                      {productionB4Ids[1], 1},
                                      {productionB4Ids[2], 1},
                                      {productionB4Ids[3], 1}},
                                     DecodeStage::Regular};
    std::array<ModelBatchItem, 4> productionB4ReplayItems;
    for (uint32_t lane = 0; lane < productionB4Ids.size(); ++lane) {
      productionB4ReplayItems[lane] = {
          productionB4Ids[lane],
          lane,
          productionPrefix.size(),
          static_cast<uint32_t>(productionPrefix.size()),
          1,
          productionB4Pages[lane]};
      productionB4ReplayItems[lane].inputTokens =
          std::span<const uint32_t>(productionPrompt)
              .subspan(productionPrefix.size(), 1);
    }
    auto productionB4Replay =
        executor.prefill(productionB4ReplayPlan, productionB4ReplayItems);
    require(productionB4Replay.size() == 4,
            "B4 recurrent restore did not replay one input token");
    for (const ModelStepResult &replay : productionB4Replay)
      requireOpen(replay, "production B4 replay");
    std::array<std::vector<uint32_t>, 4> productionB4Tokens;
    std::array<std::vector<uint32_t>, 4> productionB4Accepted;
    std::array<uint64_t, 4> productionB4Lengths{
        productionPrompt.size(), productionPrompt.size(),
        productionPrompt.size(), productionPrompt.size()};
    bool productionFinished = false;
    for (uint32_t cycle = 0; cycle < 16 && !productionFinished; ++cycle) {
      BatchPlan cyclePlan;
      cyclePlan.kind = WorkKind::Decode;
      cyclePlan.cohort = BatchCohort::Greedy;
      std::array<ModelBatchItem, 4> cycleItems;
      for (uint32_t lane = 0; lane < productionB4Ids.size(); ++lane) {
        cyclePlan.items.push_back({productionB4Ids[lane], 0});
        cycleItems[lane] = {
            productionB4Ids[lane],  lane, productionB4Lengths[lane], 0, 0,
            productionB4Pages[lane]};
      }
      auto cycleResults = executor.decode(cyclePlan, cycleItems);
      require(cycleResults.size() == 4, "production B4 width mismatch");
      for (uint32_t lane = 0; lane < cycleResults.size(); ++lane) {
        const auto &result = cycleResults[lane];
        const auto &first = cycleResults.front();
        require(!result.outputTokens.empty() &&
                    result.outputTokens.size() <= 16 - productionB4Tokens[lane].size() &&
                    result.acceptedDraftTokens <= result.draftedTokens &&
                    result.outputTokensWithoutKv <= result.outputTokens.size(),
                "production B4 output accounting is invalid");
        require(result.outputTokens == first.outputTokens &&
                    result.acceptedDraftTokens == first.acceptedDraftTokens &&
                    result.draftedTokens == first.draftedTokens &&
                    result.outputTokensWithoutKv == first.outputTokensWithoutKv &&
                    result.finished == first.finished &&
                    result.nextDecodeStage == first.nextDecodeStage,
                "identical B4 lanes made different cycle decisions");
        productionB4Tokens[lane].insert(productionB4Tokens[lane].end(),
                                        cycleResults[lane].outputTokens.begin(),
                                        cycleResults[lane].outputTokens.end());
        productionB4Accepted[lane].push_back(
            cycleResults[lane].acceptedDraftTokens);
        productionB4Lengths[lane] += cycleResults[lane].outputTokens.size() -
                                     cycleResults[lane].outputTokensWithoutKv;
        require(states.metadata(lane).lengths.targetTokens == productionB4Lengths[lane],
                "production B4 committed length differs from its output accounting");
      }
      // Budget exhaustion is the engine's decision; the oracle mirrors it.
      productionFinished =
          cycleResults[0].finished || productionB4Tokens[0].size() >= 16;
    }
    require(productionFinished, "production B4 oracle did not terminate");
    for (uint64_t id : productionB4Ids)
      executor.end(id);

    constexpr std::array<uint64_t, 2> productionB2Ids{76, 77};
    constexpr std::array<uint32_t, 2> productionB2Slots{3, 1};
    std::array<std::vector<uint32_t>, 2> productionB2Pages{
        std::vector<uint32_t>{48, 49, 50, 51, 76},
        std::vector<uint32_t>{48, 49, 50, 51, 77}};
    for (uint32_t lane = 0; lane < productionB2Ids.size(); ++lane) {
      beginCold(executor,
          makeRequest(productionB2Ids[lane], productionPrompt, 16),
          productionB2Slots[lane]);
      restoreActivePrefix(executor, productionB2Ids[lane],
                          productionPrompt.size(), productionPrefix.size(),
                          productionSnapshot);
    }
    BatchPlan productionB2ReplayPlan{
        WorkKind::Prefill,
        BatchCohort::Greedy,
        {{productionB2Ids[0], 1}, {productionB2Ids[1], 1}},
        DecodeStage::Regular};
    std::array<ModelBatchItem, 2> productionB2ReplayItems;
    for (uint32_t lane = 0; lane < productionB2Ids.size(); ++lane) {
      productionB2ReplayItems[lane] = {
          productionB2Ids[lane],
          productionB2Slots[lane],
          productionPrefix.size(),
          static_cast<uint32_t>(productionPrefix.size()),
          1,
          productionB2Pages[lane]};
      productionB2ReplayItems[lane].inputTokens =
          std::span<const uint32_t>(productionPrompt)
              .subspan(productionPrefix.size(), 1);
    }
    auto productionB2Replay =
        executor.prefill(productionB2ReplayPlan, productionB2ReplayItems);
    require(productionB2Replay.size() == 2,
            "B2 recurrent restore did not replay one input token");
    for (const ModelStepResult &replay : productionB2Replay)
      requireOpen(replay, "production B2 replay");
    std::array<std::vector<uint32_t>, 2> productionB2Tokens;
    std::array<std::vector<uint32_t>, 2> productionB2Accepted;
    std::array<uint64_t, 2> productionB2Lengths{productionPrompt.size(),
                                                productionPrompt.size()};
    bool productionB2Finished = false;
    for (uint32_t cycle = 0; cycle < 16 && !productionB2Finished; ++cycle) {
      BatchPlan cyclePlan;
      cyclePlan.kind = WorkKind::Decode;
      cyclePlan.cohort = BatchCohort::Greedy;
      std::array<ModelBatchItem, 2> cycleItems;
      for (uint32_t lane = 0; lane < productionB2Ids.size(); ++lane) {
        cyclePlan.items.push_back({productionB2Ids[lane], 0});
        cycleItems[lane] = {productionB2Ids[lane],
                            productionB2Slots[lane],
                            productionB2Lengths[lane],
                            0,
                            0,
                            productionB2Pages[lane]};
      }
      auto cycleResults = executor.decode(cyclePlan, cycleItems);
      require(cycleResults.size() == 2, "production B2 width mismatch");
      for (uint32_t lane = 0; lane < cycleResults.size(); ++lane) {
        const auto &result = cycleResults[lane];
        const auto &first = cycleResults.front();
        require(!result.outputTokens.empty() &&
                    result.outputTokens.size() <= 16 - productionB2Tokens[lane].size() &&
                    result.acceptedDraftTokens <= result.draftedTokens &&
                    result.outputTokensWithoutKv <= result.outputTokens.size(),
                "production B2 output accounting is invalid");
        require(result.outputTokens == first.outputTokens &&
                    result.acceptedDraftTokens == first.acceptedDraftTokens &&
                    result.draftedTokens == first.draftedTokens &&
                    result.outputTokensWithoutKv == first.outputTokensWithoutKv &&
                    result.finished == first.finished &&
                    result.nextDecodeStage == first.nextDecodeStage,
                "identical B2 lanes made different cycle decisions");
        productionB2Tokens[lane].insert(productionB2Tokens[lane].end(),
                                        cycleResults[lane].outputTokens.begin(),
                                        cycleResults[lane].outputTokens.end());
        productionB2Accepted[lane].push_back(
            cycleResults[lane].acceptedDraftTokens);
        productionB2Lengths[lane] += cycleResults[lane].outputTokens.size() -
                                     cycleResults[lane].outputTokensWithoutKv;
        require(states.metadata(productionB2Slots[lane]).lengths.targetTokens ==
                    productionB2Lengths[lane],
                "production B2 committed length differs from its output accounting");
      }
      productionB2Finished =
          cycleResults[0].finished || productionB2Tokens[0].size() >= 16;
    }
    require(productionB2Finished, "production B2 oracle did not terminate");
    for (uint64_t id : productionB2Ids)
      executor.end(id);

    beginCold(executor, makeRequest(75, productionPrompt, 16), 0);
    const std::vector<uint32_t> productionB1Pages{48, 49, 50, 51, 78};
    restoreActivePrefix(executor, 75, productionPrompt.size(),
                        productionPrefix.size(), productionSnapshot);
    requireOpen(prefillChunk(executor, 75, 0, productionPrefix.size(),
                             productionPrefix.size(),
                             std::span<const uint32_t>(productionPrompt)
                                 .subspan(productionPrefix.size(), 1),
                             productionB1Pages),
                "production B1 replay");
    std::vector<uint32_t> productionB1Tokens;
    std::vector<uint32_t> productionB1Accepted;
    uint64_t productionB1Length = productionPrompt.size();
    bool productionB1Finished = false;
    for (uint32_t cycle = 0; cycle < 16 && !productionB1Finished; ++cycle) {
      ModelStepResult result =
          decodeOne(executor, 75, 0, productionB1Length, productionB1Pages,
                    BatchCohort::Greedy);
      require(!result.outputTokens.empty() &&
                  result.outputTokens.size() <= 16 - productionB1Tokens.size() &&
                  result.acceptedDraftTokens <= result.draftedTokens &&
                  result.outputTokensWithoutKv <= result.outputTokens.size(),
              "production B1 output accounting is invalid");
      productionB1Tokens.insert(productionB1Tokens.end(),
                                result.outputTokens.begin(),
                                result.outputTokens.end());
      productionB1Accepted.push_back(result.acceptedDraftTokens);
      productionB1Length +=
          result.outputTokens.size() - result.outputTokensWithoutKv;
      require(states.metadata(0).lengths.targetTokens == productionB1Length,
              "production B1 committed length differs from its output accounting");
      productionB1Finished =
          result.finished || productionB1Tokens.size() >= 16;
    }
    require(productionB1Finished, "production B1 oracle did not terminate");
    // Per-cycle lane identity and commit accounting above are hard checks.
    // Batch-width rounding can change acceptance and later autoregressive
    // inputs; record those cross-path outcomes once per width.
    std::cout << "multi_cycle_decisions b4_tokens_equal_b1="
              << (productionB4Tokens.front() == productionB1Tokens)
              << " b4_acceptance_equal_b1="
              << (productionB4Accepted.front() == productionB1Accepted)
              << " b2_tokens_equal_b1="
              << (productionB2Tokens.front() == productionB1Tokens)
              << " b2_acceptance_equal_b1="
              << (productionB2Accepted.front() == productionB1Accepted)
              << " b1_rows=" << productionB1Tokens.size()
              << " b2_rows=" << productionB2Tokens.front().size()
              << " b4_rows=" << productionB4Tokens.front().size()
              << " b1_cycles=" << productionB1Accepted.size()
              << " b2_cycles=" << productionB2Accepted.front().size()
              << " b4_cycles=" << productionB4Accepted.front().size() << '\n';
    executor.end(75);
    productionSnapshot.reset();

    promptSnapshot.reset();

    // Recompute preemption keeps policy/grammar continuation on the host but
    // releases all request-owned state. Replay must not duplicate output,
    // sample another initial anchor, or repeat the initial mask handshake.
    struct PreemptionRun final {
      std::vector<uint32_t> transcript;
      uint32_t pendingAnchorIndex = 0;
    };
    const auto runPreemption = [&](BatchCohort cohort, uint32_t preemptionMode) {
      const bool preempt = preemptionMode != 0;
      EngineRequest sequence = makeRequest(80, prompt129, 24, cohort);
      if (cohort == BatchCohort::Sampling)
        sequence.sampling = {0.8F, 0.95F, 20, 91199};
      if (cohort == BatchCohort::Constrained)
        sequence.constraint = ConstraintMode::TokenMask;
      beginCold(executor, sequence, 0);
      uint32_t slot = 0;
      const auto rebuild = [&](bool repeatDuringReplay,
                                bool deliverInitialMask = false) {
        const StateSamples before = repeatDuringReplay
                                        ? sampleCommittedState(states, slot)
                                        : StateSamples{};
        executor.suspend(sequence.id);
        require(states.actualAllocatedBytes() == 0,
                "preempted request retained GDN/draft backing");
        if (deliverInitialMask) {
          const std::array<uint32_t, 1> anchor{100};
          executor.provideMask(sequence.id, singletonMasks(anchor));
        }
        StateAdmission admission = executor.resume(sequence.modelView());
        require(admission.granted(), "recompute admission failed");
        slot = *admission.cell;
        const uint32_t length = static_cast<uint32_t>(sequence.prompt.size());
        executor.setDraftContextPlan(
            sequence.id, planDraftContext(0, length, std::nullopt, {}));
        if (repeatDuringReplay) {
          requireOpen(prefillChunk(executor, sequence.id, slot, 0, 0,
                                   std::span(sequence.prompt).first(32),
                                   pageTable, cohort),
                      "interrupted state replay");
          executor.suspend(sequence.id);
          require(states.actualAllocatedBytes() == 0,
                  "repeated preemption retained state backing");
          admission = executor.resume(sequence.modelView());
          require(admission.granted(), "repeated recompute admission failed");
          slot = *admission.cell;
          executor.setDraftContextPlan(
              sequence.id, planDraftContext(0, length, std::nullopt, {}));
        }
        ModelStepResult replay = prefillChunk(
            executor, sequence.id, slot, 0, 0, sequence.prompt, pageTable, cohort);
        requireOpen(replay, "regeneration replay emitted historical tokens");
        if (repeatDuringReplay) {
          const StateSamples rebuilt = sampleCommittedState(states, slot);
          // Decode and prefill use different floating-point graphs. Record
          // that drift, but compare recovery itself to an independent cold
          // teacher-forced execution with the identical history and geometry.
          compareCommittedSamples(before, rebuilt, false);
          EngineRequest teacher = sequence;
          teacher.id = 82;
          const uint32_t teacherSlot = slot == 0 ? 1 : 0;
          const auto teacherPages = pageRange(80, 8);
          beginCold(executor, teacher, teacherSlot);
          static_cast<void>(prefillChunk(executor, teacher.id, teacherSlot, 0, 0,
                                        teacher.prompt, teacherPages, cohort));
          require(states.metadata(slot).lengths == states.metadata(teacherSlot).lengths,
                  "recomputed logical lengths differ from teacher forcing");
          compareCommittedSamples(rebuilt, sampleCommittedState(states, teacherSlot), true);
          executor.end(teacher.id);
        }
        return replay.nextDecodeStage;
      };
      if (preempt) {
        requireOpen(prefillChunk(executor, sequence.id, slot, 0, 0,
                                 std::span(sequence.prompt).first(32), pageTable,
                                 cohort),
                    "unfinished prefill before preemption");
        static_cast<void>(rebuild(false));
      } else {
        requireOpen(prefillChunk(executor, sequence.id, slot, 0, 0,
                                 sequence.prompt, pageTable, cohort),
                    "preemption prompt");
      }
      DecodeStage stage = DecodeStage::Regular;
      if (cohort == BatchCohort::Constrained) {
        stage = decodeOne(executor, sequence.id, slot, sequence.prompt.size(),
                          pageTable, cohort,
                          DecodeStage::RequestInitialMask).nextDecodeStage;
        require(stage == DecodeStage::ApplyInitialMask,
                "preemption fixture did not request initial mask");
        if (!preempt) {
          const std::array<uint32_t, 1> anchor{100};
          executor.provideMask(sequence.id, singletonMasks(anchor));
        }
      }
      if (preempt)
        require(rebuild(false, cohort == BatchCohort::Constrained) == stage,
                "prompt-end preemption changed decode/mask stage");
      PreemptionRun run;
      auto &transcript = run.transcript;
      bool generationPreempted = false;
      for (uint32_t cycle = 0; cycle < 24 && transcript.size() < 24; ++cycle) {
        ModelStepResult result;
        if (cohort == BatchCohort::Constrained) {
          auto pending = beginMaskedDecodeOne(
              executor, sequence.id, slot, sequence.prompt.size(), pageTable, stage);
          std::array<uint32_t, 9> forced;
          for (uint32_t row = 0; row < forced.size(); ++row)
            forced[row] = 100 + static_cast<uint32_t>(transcript.size()) + row;
          executor.provideMask(sequence.id, singletonMasks(forced));
          result = finishMaskedDecode(std::move(pending)).front();
          for (size_t row = 0; row < result.outputTokens.size(); ++row)
            require(row < forced.size() && result.outputTokens[row] == forced[row],
                    "preempted constrained decode violated its provided mask");
        } else {
          result = decodeOne(executor, sequence.id, slot, sequence.prompt.size(),
                             pageTable, cohort, stage);
        }
        require(!result.outputTokens.empty(), "preempted decode made no progress");
        require(result.outputTokens.size() <= sequence.maxNewTokens - transcript.size() &&
                    result.acceptedDraftTokens <= result.draftedTokens &&
                    result.outputTokensWithoutKv <= result.outputTokens.size(),
                "preempted decode output accounting is invalid");
        if (cohort == BatchCohort::Sampling)
          std::cout << "preemption_sampling mode=" << preemptionMode
                    << " cycle=" << cycle << " rows=" << result.outputTokens.size()
                    << " accepted=" << result.acceptedDraftTokens << '\n';
        transcript.insert(transcript.end(), result.outputTokens.begin(),
                          result.outputTokens.end());
        stage = result.nextDecodeStage;
        if (result.finished || transcript.size() == sequence.maxNewTokens)
          break;
        require(result.outputTokensWithoutKv == 0,
                "nonterminal preemption history contains a token without KV");
        sequence.prompt.insert(sequence.prompt.end(), result.outputTokens.begin(),
                               result.outputTokens.end());
        if (preemptionMode == 2 && !generationPreempted) {
          run.pendingAnchorIndex = static_cast<uint32_t>(transcript.size());
          require(rebuild(true) == stage,
                  "generation replay reset the decode stage");
          generationPreempted = true;
        }
      }
      require(preemptionMode != 2 || generationPreempted,
              "preemption fixture stopped before its generation replay");
      executor.end(sequence.id);
      return run;
    };
    for (BatchCohort cohort : {BatchCohort::Greedy, BatchCohort::Sampling,
                               BatchCohort::Constrained}) {
      const auto reference = runPreemption(cohort, 0);
      const auto promptResumed = runPreemption(cohort, 1);
      // The interrupted 32-row prefix is discarded. Both prompt-only paths
      // finish with the same full 129-row prefill and policy continuation.
      require(promptResumed.transcript == reference.transcript,
              "prompt recomputation changed the request policy/initial anchor");
      const auto resumed = runPreemption(cohort, 2);
      require(resumed.transcript.size() > resumed.pendingAnchorIndex &&
                  reference.transcript.size() > resumed.pendingAnchorIndex &&
                  std::equal(resumed.transcript.begin(),
                             resumed.transcript.begin() + resumed.pendingAnchorIndex + 1,
                             reference.transcript.begin()),
              "preemption changed already-emitted history or its pending anchor");
      if (cohort == BatchCohort::Sampling) {
        const auto repeated = runPreemption(cohort, 2);
        require(resumed.transcript == repeated.transcript,
                "fixed-seed recomputation is not deterministic for the same schedule");
      }
      // Generated history is rebuilt through ragged prefill, whose numerical
      // path differs from decode. Already-emitted tokens and the pending
      // anchor remain exact above; subsequent decisions are diagnostic.
      std::cout << "preemption_decisions cohort=" << static_cast<uint32_t>(cohort)
                << " transcript_equal=" << (resumed.transcript == reference.transcript)
                << " reference_rows=" << reference.transcript.size()
                << " resumed_rows=" << resumed.transcript.size()
                << " preserved_prefix_rows=" << resumed.pendingAnchorIndex + 1 << '\n';
    }

    const auto rowsBeforeInvalidWarmup = executor.telemetry().targetPrefillRows;
    for (uint32_t rows : {0U, model::ExecutionLimits::prefillTokenBudget + 1,
                          std::numeric_limits<uint32_t>::max()}) {
      bool rejected = false;
      try {
        static_cast<void>(executor.warmupPrefill(rows));
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected && executor.telemetry().targetPrefillRows == rowsBeforeInvalidWarmup,
              "invalid warmup rows reached the production prefill phase");
    }
    for (uint32_t rows : {32U, 128U, 512U}) {
      const auto smallerWarmup = executor.warmupPrefill(rows);
      require(smallerWarmup.completed && smallerWarmup.estimatedPeakBytes > 0 &&
                  smallerWarmup.wallSeconds >= executor.telemetry().lastPrefillWallSeconds &&
                  smallerWarmup.wallSeconds > 0 && smallerWarmup.lanes.size() == 1 &&
                  smallerWarmup.lanes[0].step.consumedPromptTokens == rows &&
                  smallerWarmup.lanes[0].committedTokens == rows,
              "parameterized warmup changed actual rows or omitted its measured result");
      if (rows == 32) {
        const auto repeatedWarmup = executor.warmupPrefill(rows);
        require(repeatedWarmup.lanes == smallerWarmup.lanes,
                "adjacent baseline prefill warmups changed their deterministic result");
      }
    }
    model::WarmupStepResult prefillWarmup =
        executor.warmupPrefill(model::ExecutionLimits::prefillTokenBudget);
    require(prefillWarmup.completed && prefillWarmup.estimatedPeakBytes > 0 &&
                prefillWarmup.wallSeconds > 0.0,
            "real prefill warmup did not report timing");
    require(prefillWarmup.wallSeconds >= executor.telemetry().lastPrefillWallSeconds &&
                prefillWarmup.lanes.size() == 1 &&
                prefillWarmup.lanes[0].step.consumedPromptTokens ==
                    model::ExecutionLimits::prefillTokenBudget &&
                prefillWarmup.lanes[0].committedTokens ==
                    model::ExecutionLimits::prefillTokenBudget,
            "prefill warmup omitted production wall time or its deterministic result");
    model::WarmupStepResult batch1 = executor.warmupDecodeBatch(1);
    model::WarmupStepResult batch2 = executor.warmupDecodeBatch(2);
    model::WarmupStepResult batch3 = executor.warmupDecodeBatch(3);
    model::WarmupStepResult batch4 = executor.warmupDecodeBatch(4);
    require(batch2.completed && batch2.wallSeconds > 0.0,
            "real B2 decode warmup did not report timing");
    require(batch3.completed && batch3.wallSeconds > 0.0,
            "real B3 decode warmup did not report timing");
    require(batch4.completed && batch4.wallSeconds > 0.0,
            "real B4 decode warmup did not report timing");
    const std::array batches{&batch1, &batch2, &batch3, &batch4};
    for (size_t laneCount = 1; laneCount <= batches.size(); ++laneCount) {
      const auto &batch = *batches[laneCount - 1];
      require(batch.lanes.size() == laneCount,
              "decode warmup omitted a batch-plan lane result");
      for (const auto &lane : batch.lanes)
        require(!lane.step.outputTokens.empty() && lane.committedTokens > 1,
                "decode warmup omitted its committed deterministic result");
    }
    const model::ModelTelemetry fusedTelemetry =
        executor.telemetry();
    require(batch4.wallSeconds >= fusedTelemetry.lastDecodeWallSeconds,
            "decode warmup excluded production work from phase wall time");
    require(fusedTelemetry.lastDecodeWidth == 4 &&
                fusedTelemetry.lastDecodeFusedOperations > 0 &&
                fusedTelemetry.lastDecodeM16Dispatches == 0 &&
                fusedTelemetry.lastDecodeM24Dispatches == 0 &&
                fusedTelemetry.lastDecodeM32Dispatches > 0,
            "B4 decode did not execute the fused M32 production graph");
    const auto repeatedBatch4 = executor.warmupDecodeBatch(4);
    require(repeatedBatch4.lanes == batch4.lanes,
            "repeated baseline B4 decode changed its deterministic result");
    std::cout << "prefill_2048_wall_seconds=" << prefillWarmup.wallSeconds
              << " b1_cycle_wall_seconds=" << batch1.wallSeconds
              << " b2_cycle_wall_seconds=" << batch2.wallSeconds
              << " b3_cycle_wall_seconds=" << batch3.wallSeconds
              << " b4_cycle_wall_seconds=" << batch4.wallSeconds
              << " b4_fused_source_ops="
              << fusedTelemetry.lastDecodeFusedOperations
              << " b4_m16=" << fusedTelemetry.lastDecodeM16Dispatches
              << " b4_m32=" << fusedTelemetry.lastDecodeM32Dispatches << '\n';
    model::WarmupStepResult historical =
        executor.warmupCompositeStateRestore();
    require(historical.completed && historical.estimatedPeakBytes > 0 &&
                historical.wallSeconds > 0.0,
            "historical restore-continuation warmup did not complete");
    const model::ModelTelemetry historicalTelemetry =
        executor.telemetry();
    std::cout << "cache_restore_prefill_wall_seconds="
              << historicalTelemetry.lastPrefillWallSeconds
              << " cache_restore_b1_cycle_wall_seconds="
              << historicalTelemetry.lastDecodeWallSeconds << '\n';
    std::cout << "model_runtime_oracle_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "model_runtime_oracle_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
