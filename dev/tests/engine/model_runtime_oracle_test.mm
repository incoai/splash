#include "TestChecks.hpp"
#include "TestModel.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/Types.hpp"
#include "metal/BackendInstrumentation.hpp"
#include "model/Runtime.hpp"
#include "model/QwenState.hpp"
#include "ops/PageStorage.hpp"
#include "ops/Sampling.hpp"
#include "ops/Vision.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace splash;
using namespace splash::engine;
using metal::BackendInstrumentation;
using splash::model::IdleMemory;

namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

using splash::test::rejects;
using splash::test::require;

std::string mebibytes(uint64_t bytes) {
  return std::to_string(bytes >> 20) + " MiB";
}

// The oracle loads and allocates without production's memory guard, and it
// injects every refusal its checks expect. When other programs hold memory it
// needs, the run stops here with the numbers: a check would otherwise fail
// for the wrong reason, or catch the refusal as its own.
[[noreturn]] void stopForHostMemory(const std::string &refusal,
                                    uint64_t availableBytes,
                                    uint64_t reserveBytes) {
  std::cerr << "model_runtime_oracle_test: FAIL: " << refusal
            << ", and macOS has " << mebibytes(availableBytes)
            << " available and keeps " << mebibytes(reserveBytes)
            << "; close other programs and retry\n";
  std::exit(1);
}

// A refusal of the memory governor for host memory, with what it measured.
[[noreturn]] void stopForHostMemory(const MemoryGovernor &governor,
                                    const std::string &need, uint64_t bytes) {
  const MemoryGovernorSnapshot host = governor.snapshot();
  stopForHostMemory("the memory governor refused " + mebibytes(bytes) +
                        " for " + need,
                    host.hostAvailableBytes, host.hostReserveBytes);
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
    accumulator.add(ops::tuning::bf16ToFloat(a[index]), ops::tuning::bf16ToFloat(b[index]));
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

// Budgeted greedy decoding and masked verification of the same prefix must
// commit identical state. Both use the same target arithmetic; compare bytes.
// The GDN kernel tests independently check each retained count against FP64.
void requireCommittedStateIdentical(const model::QwenStateStorage &states,
                                    uint32_t budgetLane, uint32_t maskedLane,
                                    const std::string &label) {
  const auto &budget = states.metadata(budgetLane);
  const auto &masked = states.metadata(maskedLane);
  require(budget.lengths == masked.lengths,
          label + " logical state differs between budget and mask commits");
  auto identical = [&](const metal::MetalBuffer &a, const metal::MetalBuffer &b,
                       const std::string &part) {
    require(a.sizeBytes() == b.sizeBytes() && a.contents() && b.contents() &&
                std::memcmp(a.contents(), b.contents(), a.sizeBytes()) == 0,
            label + " " + part + " differs between budget and mask commits");
  };
  identical(states.current(budgetLane).stateBase,
            states.current(maskedLane).stateBase, "GDN state");
  const auto &left = states.draft(budgetLane);
  const auto &right = states.draft(maskedLane);
  for (uint32_t layer = 0; layer < states.layout().draft.layers; ++layer) {
    identical(left[layer].keys, right[layer].keys,
              "draft keys layer=" + std::to_string(layer));
    identical(left[layer].values, right[layer].values,
              "draft values layer=" + std::to_string(layer));
  }
}

EngineRequest makeRequest(uint64_t id, std::vector<uint32_t> prompt,
                          uint32_t maximumNewTokens) {
  EngineRequest result;
  result.id = id;
  result.prompt = std::move(prompt);
  result.maxNewTokens = maximumNewTokens;
  return result;
}

void beginCold(model::Runtime &runtime, const EngineRequest &request,
               uint32_t lane) {
  runtime.beginColdRequest(request.modelView(), lane);
}

// Gives an item the revision of its page list the way the engine's cache
// does: a request's revision moves whenever its list differs from the last
// one it named, which it keeps below the first page that differs. Growing
// tables then take the runtime's incremental page-table writes.
ModelBatchItem withRevision(ModelBatchItem item) {
  struct Named final {
    uint64_t revision = 0;
    uint32_t firstChanged = 0;
    std::vector<uint32_t> pages;
  };
  static std::unordered_map<uint64_t, Named> named;
  Named &last = named[item.requestId];
  if (!last.revision || !std::ranges::equal(last.pages, item.pageTable)) {
    last.firstChanged = static_cast<uint32_t>(
        std::ranges::mismatch(last.pages, item.pageTable).in1 - last.pages.begin());
    ++last.revision;
    last.pages.assign(item.pageTable.begin(), item.pageTable.end());
  }
  item.pageTableRevision = last.revision;
  item.pageTableFirstChanged = last.firstChanged;
  return item;
}

void restoreActivePrefix(model::Runtime &executor, uint64_t requestId,
                         uint32_t promptTokens, uint32_t boundary,
                         const std::shared_ptr<const CompositeState> &state) {
  require(!executor.beginRestore(requestId, boundary, state, true, {}),
          "resident restore returned a read");
  executor.setDraftContextPlan(requestId, planDraftContext(boundary, promptTokens, {}));
}

ModelStepResult prefillChunk(model::Runtime &executor, uint64_t requestId,
                             uint32_t logicalPosition,
                             std::span<const uint32_t> inputTokens,
                             const std::vector<uint32_t> &pageTable,
                             std::optional<bool> representativeTiming = {}) {
  const uint32_t tokenCount = static_cast<uint32_t>(inputTokens.size());
  BatchPlan plan{.kind = WorkKind::Prefill,
                 .items = {{requestId, tokenCount}},
                 .decodeStage = DecodeStage::Regular};
  ModelBatchItem item =
      withRevision({requestId, logicalPosition, tokenCount, pageTable});
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
                  uint32_t logicalPosition, uint32_t token,
                  const std::vector<uint32_t> &pageTable) {
  const std::array<uint32_t, 1> input{token};
  requireOpen(
      prefillChunk(executor, requestId, logicalPosition, input, pageTable),
      "single-token prefill");
}

ModelStepResult decodeOne(model::Runtime &executor, uint64_t requestId,
                          uint64_t logicalPosition,
                          const std::vector<uint32_t> &pageTable,
                          bool constrained = false,
                          DecodeStage decodeStage = DecodeStage::Regular) {
  BatchPlan plan{.kind = WorkKind::Decode,
                 .constrained = constrained,
                 .items = {{requestId, 0}},
                 .decodeStage = decodeStage};
  ModelBatchItem item = withRevision({requestId, logicalPosition, 0, pageTable});
  auto result =
      executor.decode(plan, std::span<const ModelBatchItem>(&item, 1));
  require(result.size() == 1 && result[0].requestId == requestId,
          "decode result mismatch");
  return std::move(result[0]);
}

// A stop token or a one-token budget is emitted by prefill itself; otherwise
// the first output tokens come from one decode cycle.
ModelStepResult firstStep(model::Runtime &executor, ModelStepResult prefilled,
                          uint64_t requestId, uint64_t logicalPosition,
                          const std::vector<uint32_t> &pageTable) {
  if (!prefilled.outputTokens.empty())
    return prefilled;
  return decodeOne(executor, requestId, logicalPosition, pageTable);
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
                     uint64_t logicalPosition,
                     const std::vector<uint32_t> &pageTable) {
  BatchPlan plan{.kind = WorkKind::Decode,
                 .constrained = true,
                 .items = {{requestId, 0}},
                 .decodeStage = DecodeStage::Regular};
  const std::array items{withRevision({requestId, logicalPosition, 0, pageTable})};
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

// Gives the request a mask the runtime must take.
void provideMask(model::Runtime &executor, uint64_t requestId,
                 std::span<const uint32_t> words) {
  const std::optional<std::string> rejected =
      executor.provideMask(requestId, words);
  require(!rejected, "the runtime rejected a usable token mask: " +
                         rejected.value_or(""));
}

// The convolution and recurrent halves of a GDN cell, which hold BF16 and
// FP32 values.
metal::MetalBuffer convolutionHalf(const metal::MetalBackend &backend,
                                   const model::GdnParityBuffers &gdn,
                                   const model::GdnStateLayout &layout) {
  return backend.view(gdn.stateBase, 0, layout.convolutionBytes());
}

metal::MetalBuffer recurrentHalf(const metal::MetalBackend &backend,
                                 const model::GdnParityBuffers &gdn,
                                 const model::GdnStateLayout &layout) {
  return backend.view(gdn.stateBase, layout.convolutionBytes(),
                      layout.recurrentBytes());
}

using StateSamples = std::vector<std::pair<std::string, std::vector<float>>>;

StateSamples sampleCommittedState(const metal::MetalBackend &backend,
                                  const model::QwenStateStorage &states,
                                  uint32_t lane) {
  StateSamples result;
  const auto add = [&](std::string name, const metal::MetalBuffer &buffer,
                       bool bfloat) {
    std::vector<float> values;
    const uint64_t count = buffer.sizeBytes() / (bfloat ? 2 : 4);
    const uint64_t stride = std::max<uint64_t>(1, count / 65536);
    for (uint64_t index = 0; index < count; index += stride) {
      values.push_back(bfloat ? ops::tuning::bf16ToFloat(static_cast<const uint16_t *>(
                                                 buffer.contents())[index])
                             : static_cast<const float *>(buffer.contents())[index]);
    }
    result.emplace_back(std::move(name), std::move(values));
  };
  const auto &gdn = states.current(lane);
  const auto &target = states.layout().target;
  add("convolution", convolutionHalf(backend, gdn, target), true);
  add("recurrent", recurrentHalf(backend, gdn, target), false);
  add("first_convolution", gdn.convolutionLayers.front(), true);
  add("first_recurrent", gdn.recurrentLayers.front(), false);
  const auto &lengths = states.metadata(lane).lengths;
  const auto layout = states.layout().draft;
  constexpr uint32_t window = model::ExecutionLimits::draftContextTokens;
  const uint64_t elements = uint64_t{layout.kvHeads} * lengths.draftLength *
                            layout.headDimension;
  const uint64_t stride = std::max<uint64_t>(1, elements / 65536);
  const auto &ring = states.draft(lane);
  for (uint32_t layer = 0; layer < ring.size(); ++layer) {
    const auto *keys = bfloatContents(ring[layer].keys, "draft keys");
    const auto *values = bfloatContents(ring[layer].values, "draft values");
    std::vector<float> keySamples, valueSamples;
    for (uint64_t index = 0; index < elements; index += stride) {
      const uint32_t dimension = index % layout.headDimension;
      const uint32_t position = (index / layout.headDimension) % lengths.draftLength;
      const uint32_t head = index / (uint64_t{layout.headDimension} * lengths.draftLength);
      const uint32_t ring = (lengths.draftBase + position) % window;
      keySamples.push_back(ops::tuning::bf16ToFloat(
          keys[(uint64_t{head} * window + ring) * layout.headDimension + dimension]));
      valueSamples.push_back(ops::tuning::bf16ToFloat(
          values[(uint64_t{head} * layout.headDimension + dimension) * window + ring]));
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
                                 const model::LoadedModel &model,
                                 AllocationFault &fault) {
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  const uint64_t originalSubmissions =
      BackendInstrumentation::submittedCommands(backend);
  EngineRequest image = makeRequest(93, {1, 2}, 1);
  image.images = {{0, 1, 2, 2, 139, 431}};
  image.imagePixels.resize(image.images.front().pixelBytes());
  // At the budget the engine retries a denied start after each reclaim
  // step. A start is one admission, so a request whose lane does not fit is
  // refused before its encoder arena, sized for its image, or its image
  // buffers are built.
  {
    const ImageSpan &span = image.images.front();
    const uint64_t attemptBytes =
        ops::Vision::scratchBytes(model.vision.tensors.layout,
                                  span.gridHeight * span.gridWidth) +
        span.pixelBytes() +
        uint64_t{ops::Vision::embeddingRows({span.gridHeight, span.gridWidth})} *
            model.vision.tensors.layout.outputHiddenSize * sizeof(uint16_t) +
        model.stateLayout().laneBytes();
    fault.remainingBytes = attemptBytes - 1;
    const StateAdmission denied = executor.begin(image.modelView());
    const uint64_t unspent = fault.remainingBytes;
    fault = {};
    require(!denied.granted() &&
                denied.failure == StateFailure::MemoryPressure &&
                unspent == attemptBytes - 1 &&
                backend.memoryStats().allocatedBytes == originalBytes,
            "an image request whose lane did not fit built its encoder or "
            "image buffers");
  }
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
    while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
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
      // The suspended lane's buffers go back: what a reclaim finds after a
      // failed start below is then that start's own.
      while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
      }
    }
    for (bool sharedVision : {false, true}) {
      EngineRequest keeper = image;
      keeper.id = 94;
      if (sharedVision)
        require(executor.begin(keeper.modelView()).granted(),
                "shared vision setup failed");
      const uint64_t before = backend.memoryStats().allocatedBytes;
      // The start is one admission, of the encoder when none exists, the
      // image pixels and embeddings and the lane's cells and ring: it is
      // refused, or fails after it allocated them all.
      for (bool throwing : {false, true}) {
        fault = {0, throwing};
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
        require(executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches) == 0,
                "failed image admission created false reclamation progress");
      }
      if (sharedVision) {
        executor.end(keeper.id);
        while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
        }
      }
    }
    // The same request can still be admitted after every failure mode.
    require((resume ? executor.resume(image.modelView())
                    : executor.begin(image.modelView())).granted(),
            "image request could not retry after allocation failure");
    executor.end(image.id);
    while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
        }
    require(backend.memoryStats().allocatedBytes == originalBytes,
            "image admission test leaked resources");
  }
  require(BackendInstrumentation::submittedCommands(backend) ==
              originalSubmissions,
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
                                  const model::LoadedModel &model,
                                  AllocationFault &fault) {
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
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
  executor.setDraftContextPlan(
      request.id, planDraftContext(0, static_cast<uint32_t>(prompt.size()), {}));
  prefillChunk(executor, request.id, 0,
               std::span<const uint32_t>(prompt).first(64), pages, false);
  // Nothing else is idle, so the pass releases exactly the encoder arena.
  uint64_t reclaimed = 0;
  while (const uint64_t bytes = executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches))
    reclaimed += bytes;
  const ImageSpan &span = request.images.front();
  const uint64_t encoderBytes = ops::Vision::scratchBytes(
      model.vision.tensors.layout, span.gridHeight * span.gridWidth);
  require(reclaimed == encoderBytes,
          "encoder whose only image is encoded survived reclaim");
  prefillChunk(executor, request.id, 64,
               std::span<const uint32_t>(prompt).subspan(64), pages, true);
  require(executor.telemetry().imageEncodes == encodesBefore + 1,
          "straddling image was not encoded exactly once");
  executor.end(request.id);

  // Free only pooled state, keeping the image cache. A cache-only request
  // must fit a fresh lane's state without recreating the reclaimed encoder.
  while (states.releaseOneIdle(false)) {
  }
  const uint64_t stateBytes = model.stateLayout().laneBytes();
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
  // Fail the start's admission, by refusal and by an exception after it
  // allocated, and leave both the cache and live request intact. Only the
  // admitted attempt counts its cache hit as a reuse.
  EngineRequest mixed = request;
  mixed.id = 97;
  mixed.images.push_back({80, 16, 8, 8, 157, 439});
  mixed.imagePixels.resize(2 * request.imagePixels.size());
  const uint64_t beforeMixed = backend.memoryStats().allocatedBytes;
  const uint64_t reusedBeforeMixed = executor.telemetry().imageEmbeddingReuses;
  for (bool throwing : {false, true}) {
    fault = {0, throwing};
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
  while (const uint64_t bytes = executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches))
    released += bytes;
  require(released == heldBytes - backend.memoryStats().allocatedBytes,
          "reclaim credited cached rows a live request still holds");
  executor.end(request.id);

  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  require(backend.memoryStats().allocatedBytes == originalBytes,
          "image requests leaked resources");
  std::cout << "image_rows_after_reclaim=PASS\n";
}

void requireRepeatedImagePlacements(model::Runtime &executor,
                                     metal::MetalBackend &backend,
                                     model::QwenStateStorage &states,
                                     AllocationFault &fault) {
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
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
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
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
  const uint32_t lane = *admitted.lane;
  const std::vector<uint32_t> pages = pageRange(120, 4);
  const std::array<uint32_t, 1> checkpoints{64};
  executor.setDraftContextPlan(request.id, planDraftContext(0, prompt.size(), checkpoints));
  const uint64_t encodes = executor.telemetry().imageEncodes;
  prefillChunk(executor, request.id, 0,
               std::span<const uint32_t>(prompt).first(64), pages, false);
  auto checkpoint = executor.snapshot(request.id);
  require(checkpoint != nullptr, "image prefix checkpoint allocation failed");
  prefillChunk(executor, request.id, 64,
               std::span<const uint32_t>(prompt).subspan(64), pages, true);
  require(executor.telemetry().imageEncodes == encodes + 1,
          "repeated image placements encoded more than once");
  const auto expected = sampleCommittedState(backend, states, lane);
  executor.end(request.id);
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }

  // The first placement is covered by the prefix, but its later duplicates
  // still need their shared image data after the embedding cache is reclaimed.
  request.id = 99;
  const StateAdmission restored = executor.begin(request.modelView());
  require(restored.granted(), "repeated image restore was not admitted");
  restoreActivePrefix(executor, request.id, prompt.size(), 64, checkpoint);
  prefillChunk(executor, request.id, 64,
               std::span<const uint32_t>(prompt).subspan(64), pages, false);
  require(executor.telemetry().imageEncodes == encodes + 2,
          "prefix restore discarded data for a later image placement");
  compareCommittedSamples(
      expected, sampleCommittedState(backend, states, *restored.lane), true);
  executor.end(request.id);
  checkpoint.reset();
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  require(backend.memoryStats().allocatedBytes == originalBytes,
          "repeated image placements retained resources");
  std::cout << "repeated_image_placements=PASS\n";
}

// Fills every byte of a lane's GDN recurrent state, the FP32 half of the cell
// its next transition reads, with 0xFF: NaN, which the recurrence carries into
// every row the lane computes. Non-finite KV would not do: the paged-attention
// tile of some GPU families gives non-finite keys and values zero weight.
void poisonRecurrentState(const metal::MetalBackend &backend,
                          const model::QwenStateStorage &states, uint32_t lane) {
  const metal::MetalBuffer recurrent =
      recurrentHalf(backend, states.current(lane), states.layout().target);
  std::memset(recurrent.contents(), 0xFF, recurrent.sizeBytes());
}

// Zeroes the pages: a lane computing from a NaN state writes NaN keys and
// values there, which the checks that follow must not find.
void clearPages(const kv::PageStorage &pages, std::span<const uint32_t> ids) {
  for (const uint32_t page : ids)
    for (const std::span<std::byte> bytes : pages.spans(page))
      std::ranges::fill(bytes, std::byte{0});
}

// A request whose own state is non-finite selects outside the vocabulary,
// which the runtime reports as that lane's failure instead of throwing: in a
// decode beside a healthy lane, which commits and keeps decoding, at the end
// of a prefill, and in a constrained request's first selection from its final
// hidden. The failed lanes emit nothing and the backend stays healthy. The
// GDN cells they return to the pool are cleared or overwritten before a lane
// reads them again.
void requireNonFiniteRowFailsOnlyItsLane(model::Runtime &executor,
                                         const kv::PageStorage &pages,
                                         const model::QwenStateStorage &states,
                                         const metal::MetalBackend &backend,
                                         std::span<const uint32_t> chat,
                                         std::span<const uint32_t> prompt) {
  // The chat ends where the assistant starts reasoning, so a greedy lane does
  // not select a stop token there and both decode lanes stay open after
  // prefill. The prefill and constrained lanes fail whatever their tokens.
  const std::vector<uint32_t> prompt40(chat.end() - 40, chat.end());
  const std::vector<uint32_t> prompt80(prompt.begin(), prompt.begin() + 80);
  const std::vector<uint32_t> pagesA{36, 37};
  const std::vector<uint32_t> pagesB{38, 39};
  const std::vector<uint32_t> pagesC{124, 125, 126};
  const auto inVocabulary = [](const ModelStepResult &result) {
    return std::all_of(result.outputTokens.begin(), result.outputTokens.end(),
                       [](uint32_t token) { return token < kVocabulary; });
  };

  beginCold(executor, makeRequest(201, prompt40, 16), 0);
  beginCold(executor, makeRequest(202, prompt40, 16), 1);
  requireOpen(prefillChunk(executor, 201, 0, prompt40, pagesA),
              "non-finite decode fixture");
  requireOpen(prefillChunk(executor, 202, 0, prompt40, pagesB),
              "non-finite decode fixture");
  poisonRecurrentState(backend, states, 0);
  const BatchPlan plan{.kind = WorkKind::Decode,
                       .items = {{201, 0}, {202, 0}},
                       .decodeStage = DecodeStage::Regular};
  const std::array items{
      withRevision({.requestId = 201, .logicalPosition = 40, .pageTable = pagesA}),
      withRevision({.requestId = 202, .logicalPosition = 40, .pageTable = pagesB})};
  const std::vector<ModelStepResult> decoded = executor.decode(plan, items);
  require(decoded.size() == 2 && !decoded[0].failure.empty() &&
              decoded[0].outputTokens.empty(),
          "a decode from a non-finite state did not fail its lane alone");
  require(decoded[1].failure.empty() && !decoded[1].outputTokens.empty() &&
              !decoded[1].finished && inVocabulary(decoded[1]),
          "a non-finite lane disturbed its healthy neighbour");
  executor.end(201);
  const ModelStepResult continued =
      decodeOne(executor, 202, 40 + decoded[1].outputTokens.size(), pagesB);
  require(continued.failure.empty() && !continued.outputTokens.empty() &&
              inVocabulary(continued),
          "the healthy lane did not keep decoding after its neighbour failed");
  executor.end(202);
  clearPages(pages, pagesA);

  beginCold(executor, makeRequest(203, prompt80, 16), 0);
  prefillChunk(executor, 203, 0, std::span(prompt80).first(64), pagesC);
  poisonRecurrentState(backend, states, 0);
  const ModelStepResult prefilled = prefillChunk(
      executor, 203, 64, std::span(prompt80).subspan(64), pagesC);
  require(!prefilled.failure.empty() && prefilled.outputTokens.empty(),
          "a prefill from a non-finite state did not fail its lane");
  executor.end(203);
  clearPages(pages, pagesC);

  EngineRequest constrained = makeRequest(204, prompt80, 16);
  constrained.constraint = ConstraintMode::TokenMask;
  beginCold(executor, constrained, 0);
  prefillChunk(executor, 204, 0, std::span(prompt80).first(64), pagesC);
  poisonRecurrentState(backend, states, 0);
  require(prefillChunk(executor, 204, 64, std::span(prompt80).subspan(64),
                       pagesC)
                  .nextDecodeStage == DecodeStage::ApplyInitialMask,
          "the non-finite constrained fixture did not ask for its first mask");
  provideMask(executor, 204, std::vector<uint32_t>(kMaskWords, 0xFFFFFFFFU));
  const ModelStepResult selected =
      decodeOne(executor, 204, 80, pagesC, true,
                DecodeStage::ApplyInitialMask);
  require(!selected.failure.empty() && selected.outputTokens.empty(),
          "a constrained selection from a non-finite hidden did not fail its lane");
  executor.end(204);
  clearPages(pages, pagesC);
  require(backend.healthy(), "a non-finite lane made the backend unhealthy");
  std::cout << "non_finite_rows=PASS\n";
}

// A 64-token request with one image of the given span, its pixels a pattern.
EngineRequest imageRequest(uint64_t id, ImageSpan span) {
  std::vector<uint32_t> prompt(64);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = 1 + index;
  EngineRequest request = makeRequest(id, prompt, 1);
  request.images = {span};
  request.imagePixels.resize(span.pixelBytes());
  for (size_t index = 0; index < request.imagePixels.size(); ++index)
    request.imagePixels[index] = static_cast<uint8_t>(index * 13 + id);
  return request;
}

// Two requests with the same image share its rows from their start: both
// admitted before either prefills, one ragged command encodes the image
// once and both lanes inject it.
void requireConcurrentRequestsShareOneEncode(model::Runtime &executor,
                                             metal::MetalBackend &backend) {
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  const EngineRequest first = imageRequest(110, {16, 16, 8, 8, 167, 449});
  EngineRequest second = first;
  second.id = 111;
  const uint64_t reuses = executor.telemetry().imageEmbeddingReuses;
  const StateAdmission firstAdmission = executor.begin(first.modelView());
  const uint64_t oneImage = executor.telemetry().imageRowsBytes;
  const StateAdmission secondAdmission = executor.begin(second.modelView());
  require(firstAdmission.granted() && secondAdmission.granted() && oneImage &&
              executor.telemetry().imageRowsBytes == oneImage &&
              executor.telemetry().imageEmbeddingReuses == reuses + 1,
          "a second request with the same image did not share its rows");
  BatchPlan plan{.kind = WorkKind::Prefill,
                 .decodeStage = DecodeStage::Regular};
  std::array<ModelBatchItem, 2> items;
  const std::array<std::vector<uint32_t>, 2> pages{pageRange(116, 4), pageRange(120, 4)};
  for (uint32_t lane = 0; lane < 2; ++lane) {
    const EngineRequest &request = lane ? second : first;
    executor.setDraftContextPlan(
        request.id, planDraftContext(0, static_cast<uint32_t>(request.prompt.size()), {}));
    const auto tokens = static_cast<uint32_t>(request.prompt.size());
    plan.items.push_back({request.id, tokens});
    items[lane] =
        withRevision({.requestId = request.id, .tokenCount = tokens, .pageTable = pages[lane]});
    items[lane].inputTokens = request.prompt;
  }
  const uint64_t encodes = executor.telemetry().imageEncodes;
  static_cast<void>(executor.prefill(plan, items));
  require(executor.telemetry().imageEncodes == encodes + 1,
          "two requests with the same image encoded it twice");
  executor.end(first.id);
  executor.end(second.id);
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  require(backend.memoryStats().allocatedBytes == originalBytes,
          "shared image rows leaked resources");
  std::cout << "concurrent_image_requests_share_one_encode=PASS\n";
}

// A lane suspended inside its image keeps the encoded rows for its resume:
// the cache takes them, and the replay injects them without an encode.
void requireSuspendedLaneKeepsItsRows(model::Runtime &executor,
                                      metal::MetalBackend &backend) {
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  const EngineRequest request = imageRequest(112, {16, 32, 8, 16, 173, 457});
  const std::span<const uint32_t> prompt(request.prompt);
  const std::vector<uint32_t> pages = pageRange(120, 4);
  const StateAdmission admission = executor.begin(request.modelView());
  require(admission.granted(), "suspended image request was not admitted");
  executor.setDraftContextPlan(
      request.id, planDraftContext(0, static_cast<uint32_t>(prompt.size()), {}));
  const uint64_t encodes = executor.telemetry().imageEncodes;
  prefillChunk(executor, request.id, 0, prompt.first(32), pages, false);
  executor.suspend(request.id);
  const StateAdmission resumed = executor.resume(request.modelView());
  require(resumed.granted(), "suspended image request did not resume");
  executor.setDraftContextPlan(
      request.id, planDraftContext(0, static_cast<uint32_t>(prompt.size()), {}));
  prefillChunk(executor, request.id, 0, prompt, pages, true);
  require(executor.telemetry().imageEncodes == encodes + 1,
          "a resumed lane encoded its image again");
  executor.end(request.id);
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  require(backend.memoryStats().allocatedBytes == originalBytes,
          "a suspended image request leaked resources");
  std::cout << "suspended_lane_keeps_its_rows=PASS\n";
}

// Once a request has injected its image's last row the cache owns the rows,
// so reclaim frees them, and the idle encoder, while the request still holds
// its lane.
void requireInjectedRowsBecomeReclaimable(model::Runtime &executor,
                                          metal::MetalBackend &backend) {
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  const EngineRequest request = imageRequest(113, {16, 16, 8, 8, 179, 461});
  const StateAdmission admission = executor.begin(request.modelView());
  require(admission.granted(), "reclaimable image request was not admitted");
  executor.setDraftContextPlan(
      request.id, planDraftContext(0, static_cast<uint32_t>(request.prompt.size()), {}));
  prefillChunk(executor, request.id, 0, request.prompt, pageRange(120, 4),
               false);
  const model::ModelTelemetry injected = executor.telemetry();
  require(injected.embeddingCacheBytes && injected.visionArenaBytes &&
              injected.imageRowsBytes == injected.embeddingCacheBytes,
          "injected rows did not pass to the embedding cache");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  uint64_t released = 0;
  while (const uint64_t bytes = executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches))
    released += bytes;
  require(released == injected.embeddingCacheBytes + injected.visionArenaBytes &&
              before - backend.memoryStats().allocatedBytes == released &&
              executor.telemetry().imageRowsBytes == 0,
          "reclaim did not free the rows a running request has injected");
  executor.end(request.id);
  std::cout << "injected_rows_become_reclaimable=PASS\n";
}

// A start whose restored prefix covers its only image leaves the image out:
// no pixels, embeddings or encoder, only the lane. A restore that stops
// before that prefix would replay rows of an image never staged and is
// refused; one at it continues past the image without encoding it.
void requireCoveredImagesAreNotStaged(model::Runtime &executor,
                                      metal::MetalBackend &backend,
                                      const model::LoadedModel &model) {
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  std::vector<uint32_t> prompt(128);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = 1 + index;
  EngineRequest request = makeRequest(114, prompt, 1);
  request.images = {{40, 16, 8, 8, 181, 463}};
  request.imagePixels.resize(request.images.front().pixelBytes());
  for (size_t index = 0; index < request.imagePixels.size(); ++index)
    request.imagePixels[index] = static_cast<uint8_t>(index * 5 + 1);
  const std::vector<uint32_t> pages = pageRange(120, 4);
  // An earlier turn leaves states before the image and past it.
  const StateAdmission producer = executor.begin(request.modelView());
  require(producer.granted(), "covered image producer was not admitted");
  const std::array<uint32_t, 2> checkpoints{32, 64};
  executor.setDraftContextPlan(request.id, planDraftContext(0, prompt.size(), checkpoints));
  prefillChunk(executor, request.id, 0, std::span<const uint32_t>(prompt).first(32), pages);
  std::shared_ptr<const CompositeState> beforeImage = executor.snapshot(request.id);
  prefillChunk(executor, request.id, 32,
               std::span<const uint32_t>(prompt).subspan(32, 32), pages, false);
  std::shared_ptr<const CompositeState> pastImage = executor.snapshot(request.id);
  require(beforeImage && pastImage, "covered image states were not snapshotted");
  executor.end(request.id);
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }

  const uint64_t encodes = executor.telemetry().imageEncodes;
  const uint64_t reuses = executor.telemetry().imageEmbeddingReuses;
  for (const uint64_t id : {115U, 116U}) {
    request.id = id;
    ModelRequest covered = request.modelView();
    covered.restoredTokens = 64;
    const uint64_t before = backend.memoryStats().allocatedBytes;
    const StateAdmission admission = executor.begin(covered);
    const model::ModelTelemetry started = executor.telemetry();
    require(admission.granted() && !started.imageRowsBytes && !started.visionArenaBytes &&
                started.imageEmbeddingReuses == reuses &&
                backend.memoryStats().allocatedBytes ==
                    before + model.stateLayout().laneBytes(),
            "a start staged an image its restored prefix covers");
    if (id == 115) {
      rejects([&] { static_cast<void>(executor.beginRestore(id, 32, beforeImage, true, {})); },
              "restore stops before images its activation left out",
              "a restore before a left-out image was accepted");
    } else {
      restoreActivePrefix(executor, id, prompt.size(), 64, pastImage);
      prefillChunk(executor, id, 64,
                   std::span<const uint32_t>(prompt).subspan(64), pages, true);
      require(executor.telemetry().imageEncodes == encodes,
              "a restore past an image encoded it");
    }
    executor.end(id);
    while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
    }
  }
  beforeImage.reset();
  pastImage.reset();
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  require(backend.memoryStats().allocatedBytes == originalBytes,
          "covered image starts leaked resources");
  std::cout << "covered_images_are_not_staged=PASS\n";
}

// A start refused memory holds what it matched while the engine reclaims
// before its retry: the encoder its new image would use, and cached rows it
// would share. Reclaim spares both, so the retry needs no more memory than
// the refused attempt did.
void requireRefusedStartKeepsItsRows(model::Runtime &executor,
                                     metal::MetalBackend &backend,
                                     const model::LoadedModel &model,
                                     AllocationFault &fault) {
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  const uint64_t laneBytes = model.stateLayout().laneBytes();
  const auto imageBytes = [&](const ImageSpan &span) {
    return span.pixelBytes() +
           uint64_t{ops::Vision::embeddingRows({span.gridHeight, span.gridWidth})} *
               model.vision.tensors.layout.outputHiddenSize * sizeof(uint16_t);
  };
  std::vector<uint32_t> prompt(128);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = 1 + index;
  // The image straddles the first chunk, which encodes it: the encoder is
  // idle and the rows stay with the request.
  EngineRequest cached = makeRequest(117, prompt, 1);
  cached.images = {{56, 16, 8, 8, 191, 467}};
  cached.imagePixels.resize(cached.images.front().pixelBytes());
  for (size_t index = 0; index < cached.imagePixels.size(); ++index)
    cached.imagePixels[index] = static_cast<uint8_t>(index * 3 + 2);
  const std::vector<uint32_t> pages = pageRange(120, 4);
  const StateAdmission admitted = executor.begin(cached.modelView());
  require(admitted.granted(), "refused-start fixture was not admitted");
  executor.setDraftContextPlan(cached.id, planDraftContext(0, prompt.size(), {}));
  prefillChunk(executor, cached.id, 0,
               std::span<const uint32_t>(prompt).first(64), pages, false);

  // A new image's start, refused one byte short: it holds the idle encoder.
  EngineRequest fresh = imageRequest(118, {16, 16, 8, 8, 193, 479});
  const uint64_t arenaBytes = executor.telemetry().visionArenaBytes;
  {
    fault.remainingBytes = laneBytes + imageBytes(fresh.images.front()) - 1;
    const StateAdmission refused = executor.begin(fresh.modelView());
    fault = {};
    require(!refused.granted() && refused.held &&
                executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches) == 0 &&
                executor.telemetry().visionArenaBytes == arenaBytes,
            "reclaim took the encoder a refused start holds");
  }
  const uint64_t beforeRetry = backend.memoryStats().allocatedBytes;
  require(executor.begin(fresh.modelView()).granted() &&
              backend.memoryStats().allocatedBytes ==
                  beforeRetry + laneBytes + imageBytes(fresh.images.front()),
          "the retry of a refused start built another encoder");
  executor.end(fresh.id);
  while (executor.reclaimIdleState(false, IdleMemory::Buffers)) {
  }
  require(executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches) == arenaBytes,
          "the idle encoder was not the only cache left");
  prefillChunk(executor, cached.id, 64,
               std::span<const uint32_t>(prompt).subspan(64), pages, true);
  executor.end(cached.id);
  while (executor.reclaimIdleState(false, IdleMemory::Buffers)) {
  }

  // The same image's start, refused one byte short of its lane: it holds the
  // cached rows, the only cache, and reclaim frees nothing of them.
  EngineRequest repeat = cached;
  repeat.id = 119;
  const uint64_t encodes = executor.telemetry().imageEncodes;
  {
    fault.remainingBytes = laneBytes - 1;
    const StateAdmission refused = executor.begin(repeat.modelView());
    fault = {};
    require(!refused.granted() && refused.held &&
                executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches) == 0 &&
                executor.telemetry().embeddingCacheBytes != 0,
            "reclaim took the cached rows a refused start holds");
  }
  const uint64_t beforeRepeat = backend.memoryStats().allocatedBytes;
  require(executor.begin(repeat.modelView()).granted() &&
              backend.memoryStats().allocatedBytes == beforeRepeat + laneBytes &&
              executor.telemetry().imageEncodes == encodes,
          "the retry of a refused start did not find its cached rows");
  executor.end(repeat.id);
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  require(backend.memoryStats().allocatedBytes == originalBytes,
          "refused image starts leaked resources");
  std::cout << "refused_start_keeps_its_rows=PASS\n";
}

// A reclaim step frees one unit: with two cached images and an idle encoder
// three steps free the arena, which neither image needs again, then the
// older entry and the newer, and a step limited to buffers frees none of
// them.
void requireReclaimTakesOneCacheUnit(model::Runtime &executor,
                                     metal::MetalBackend &backend,
                                     const model::LoadedModel &model) {
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  const auto rowBytes = [&](const ImageSpan &span) {
    return uint64_t{ops::Vision::embeddingRows({span.gridHeight, span.gridWidth})} *
           model.vision.tensors.layout.outputHiddenSize * sizeof(uint16_t);
  };
  std::vector<uint32_t> prompt(128);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = 1 + index;
  EngineRequest request = makeRequest(120, prompt, 1);
  request.images = {{8, 16, 8, 8, 197, 487}, {32, 64, 16, 16, 199, 491}};
  request.imagePixels.resize(request.images[0].pixelBytes() +
                             request.images[1].pixelBytes());
  for (size_t index = 0; index < request.imagePixels.size(); ++index)
    request.imagePixels[index] = static_cast<uint8_t>(index * 9 + 4);
  const StateAdmission admission = executor.begin(request.modelView());
  require(admission.granted(), "two-image request was not admitted");
  executor.setDraftContextPlan(request.id, planDraftContext(0, prompt.size(), {}));
  prefillChunk(executor, request.id, 0, prompt, pageRange(120, 4), false);
  executor.end(request.id);
  while (executor.reclaimIdleState(false, IdleMemory::Buffers)) {
  }
  const model::ModelTelemetry cached = executor.telemetry();
  require(cached.embeddingCacheBytes ==
                  rowBytes(request.images[0]) + rowBytes(request.images[1]) &&
              cached.visionArenaBytes != 0,
          "a step limited to buffers freed a cache");
  for (const uint64_t expected :
       {cached.visionArenaBytes, rowBytes(request.images[0]), rowBytes(request.images[1])}) {
    require(executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches) == expected,
            "a reclaim step did not free the next cache unit");
  }
  require(executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches) == 0 &&
              backend.memoryStats().allocatedBytes == originalBytes,
          "cache units leaked resources");
  std::cout << "reclaim_takes_one_cache_unit=PASS\n";
}

// A start's encoder covers the largest image it encodes, not the server's
// cap. A start whose image the live encoder covers builds none; one with a
// larger image builds a larger encoder, which replaces the smaller one and
// encodes the images still waiting on it. A refused start holds the live
// encoder only when it covers the start's image.
void requireEncoderFitsItsImages(model::Runtime &executor,
                                 metal::MetalBackend &backend,
                                 const model::LoadedModel &model,
                                 AllocationFault &fault) {
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  const auto scratchBytes = [&](const EngineRequest &request) {
    const ImageSpan &span = request.images.front();
    return ops::Vision::scratchBytes(model.vision.tensors.layout,
                                     span.gridHeight * span.gridWidth);
  };
  const EngineRequest small = imageRequest(123, {16, 16, 8, 8, 223, 503});
  const EngineRequest smaller = imageRequest(124, {16, 4, 4, 4, 227, 509});
  const EngineRequest large = imageRequest(125, {16, 32, 8, 16, 229, 521});
  require(executor.begin(small.modelView()).granted() &&
              executor.telemetry().visionArenaBytes == scratchBytes(small) &&
              executor.begin(smaller.modelView()).granted() &&
              executor.telemetry().visionArenaBytes == scratchBytes(small),
          "an encoder was not sized for the largest image its start encodes");
  require(executor.begin(large.modelView()).granted() &&
              executor.telemetry().visionArenaBytes == scratchBytes(large),
          "a larger image did not replace the smaller encoder");
  const uint64_t encodes = executor.telemetry().imageEncodes;
  for (const EngineRequest *request : {&small, &smaller, &large}) {
    executor.setDraftContextPlan(
        request->id, planDraftContext(0, static_cast<uint32_t>(request->prompt.size()), {}));
    prefillChunk(executor, request->id, 0, request->prompt, pageRange(120, 4));
    executor.end(request->id);
  }
  require(executor.telemetry().imageEncodes == encodes + 3,
          "the larger encoder did not encode the images waiting on the smaller");

  // The idle encoder does not cover a larger image, so that image's refused
  // start holds none of it and reclaim frees the arena before the retry.
  while (executor.reclaimIdleState(false, IdleMemory::Buffers)) {
  }
  const EngineRequest largest = imageRequest(126, {0, 64, 16, 16, 233, 523});
  fault.remainingBytes = 0;
  const StateAdmission refused = executor.begin(largest.modelView());
  fault = {};
  require(!refused.granted() && refused.held &&
              executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches) ==
                  scratchBytes(large),
          "a refused start held an encoder too small for its image");
  require(executor.begin(largest.modelView()).granted() &&
              executor.telemetry().visionArenaBytes == scratchBytes(largest),
          "the retry did not build an encoder for its image");
  executor.end(largest.id);
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  require(backend.memoryStats().allocatedBytes == originalBytes,
          "encoders sized for their images leaked resources");
  std::cout << "encoder_fits_its_images=PASS\n";
}

// A turn that ends in an image leaves its replay point inside it. The state
// in RAM holds the image's rows: reclaim spares them while the state lives,
// the next turn restores there and injects the last rows without the
// encoder, and the hold ends with the state, leaving the rows an ordinary
// cache entry.
void requireReplayPointKeepsItsImageRows(model::Runtime &executor,
                                         metal::MetalBackend &backend) {
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  const uint64_t originalBytes = backend.memoryStats().allocatedBytes;
  std::vector<uint32_t> prompt(128);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = 1 + index;
  // The image ends 8 rows past the page boundary at 64.
  EngineRequest request = makeRequest(121, prompt, 1);
  request.images = {{56, 16, 8, 8, 211, 499}};
  request.imagePixels.resize(request.images.front().pixelBytes());
  for (size_t index = 0; index < request.imagePixels.size(); ++index)
    request.imagePixels[index] = static_cast<uint8_t>(index * 17 + 6);
  const std::vector<uint32_t> pages = pageRange(120, 4);
  const StateAdmission first = executor.begin(request.modelView());
  require(first.granted(), "replay point image request was not admitted");
  const std::array<uint32_t, 1> boundary{64};
  executor.setDraftContextPlan(request.id, planDraftContext(0, prompt.size(), boundary));
  prefillChunk(executor, request.id, 0,
               std::span<const uint32_t>(prompt).first(64), pages, false);
  std::shared_ptr<const CompositeState> replayPoint = executor.snapshot(request.id);
  require(replayPoint != nullptr, "replay point snapshot failed");
  executor.end(request.id);
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  const model::ModelTelemetry held = executor.telemetry();
  require(held.stateHeldImageBytes && held.imageRowsBytes == held.stateHeldImageBytes &&
              !held.visionArenaBytes,
          "reclaim took the image rows a replay point in RAM holds");

  const uint64_t encodes = held.imageEncodes;
  request.id = 122;
  ModelRequest resumed = request.modelView();
  resumed.restoredTokens = 64;
  const StateAdmission next = executor.begin(resumed);
  require(next.granted(), "the next turn was not admitted");
  restoreActivePrefix(executor, request.id, prompt.size(), 64, replayPoint);
  prefillChunk(executor, request.id, 64,
               std::span<const uint32_t>(prompt).subspan(64), pages, true);
  require(executor.telemetry().imageEncodes == encodes &&
              !executor.telemetry().visionArenaBytes,
          "resuming inside an image encoded it again");
  executor.end(request.id);
  replayPoint.reset();
  require(!executor.telemetry().stateHeldImageBytes,
          "the hold outlived the replay point");
  while (executor.reclaimIdleState(false, IdleMemory::BuffersThenCaches)) {
  }
  require(!executor.telemetry().imageRowsBytes &&
              backend.memoryStats().allocatedBytes == originalBytes,
          "the rows outlived the replay point that held them");
  std::cout << "replay_point_keeps_its_image_rows=PASS\n";
}

void warmupEos(model::RuntimeContext context, model::LoadedModel &model) {
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
  const auto originalTarget = model.descriptor.target;
  const auto setStops = [&](uint32_t stop) {
    std::visit([&](auto &weights) {
      weights.layout.stopTokens = {stop, stop};
      model.descriptor.target = weights.layout;
    }, model.target);
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
      require(result.lanes.size() == width,
              "prefill EOS skipped the actual decode warmup");
    }
    require(executor.warmupCompositeStateRestore().wallSeconds > 0.0,
            "prefill EOS broke the restore warmup");
  }
  setStops(decodeStop);
  {
    model::Runtime executor(context);
    const auto result = executor.warmupDecodeBatch(1);
    const auto &lane = result.lanes[0];
    require(lane.step.finished &&
                lane.step.outputTokensWithoutKv == 1 &&
                lane.pendingToken == decodeStop && lane.committedTokens > 1 &&
                lane.committedTokens == lane.step.outputTokens.size(),
            "decode EOS was not distinguished from committed KV rows");
  }
  std::visit([&](auto &weights) {
    using Layout = std::decay_t<decltype(weights.layout)>;
    weights.layout = std::get<Layout>(originalTarget);
  }, model.target);
  model.descriptor.target = originalTarget;
  const model::QwenStateStorage &states = context.stateStorage;
  for (uint32_t lane = 0; lane < 4; ++lane)
    require(!states.metadata(lane).assigned(),
            "EOS warmup left an active state lane");
  std::cout << "warmup_eos=PASS prefill_stop=" << prefillStop
            << " decode_stop=" << decodeStop << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    bool imagesOnly = false, warmupEosOnly = false;
    kv::Format format = kv::Format::Int8;
    if (argc < 3) fail("usage: model-runtime-oracle METALLIB MODEL_ROOT [--kv-format int8|bf16]");
    for (int i = 3; i < argc; ++i) {
      const std::string_view option(argv[i]);
      if (option == "--images-only") imagesOnly = true;
      else if (option == "--warmup-eos-only") warmupEosOnly = true;
      else if (option == "--kv-format" && i + 1 < argc) {
        const std::string_view value(argv[++i]);
        if (value != "int8" && value != "bf16") fail("invalid KV format");
        format = value == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;
      } else fail("unknown model-runtime-oracle option");
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
    const std::filesystem::path modelRoot(argv[2]);
    const auto descriptor = model::inspectModelRoot(modelRoot);
    // Production's weight byte count with a different bound. Production checks
    // it only against the Metal hard budget, then guards host headroom at every
    // Metal operation while loading. This oracle has no such guard, so the
    // weights must fit in reclaimable memory above the macOS reserve before
    // any is loaded; it can refuse a model production starts.
    const uint64_t weightBytes = model::modelWeightBytes(modelRoot, descriptor);
    if (*hostAvailableBytes <= hostReserveBytes ||
        weightBytes > *hostAvailableBytes - hostReserveBytes)
      stopForHostMemory("the weights need " + mebibytes(weightBytes),
                        *hostAvailableBytes, hostReserveBytes);
    model::LoadedModel model =
        model::loadModel(backend, modelRoot, descriptor);
    ops::ExecutionPlans operators(backend.capabilities());
    model::ModelMemoryPlan executorPlan =
        model::plannedRuntimeMemory(model, operators, format);
    ModelMemoryFootprint footprint{
        model.targetActualAllocatedBytes(),
        model.draft.actualAllocatedBytes,
        model.vision.actualAllocatedBytes,
        executorPlan, 0};
    ModelMemoryProfile profile{
        model.name(), model.maximumContextTokens(),
        model.targetKvLayout(format), footprint};
    EngineMemoryPlan memoryPlan =
        test::requireMemoryPlan(backend.capabilities(), profile);

    // A pool of 128 pages, or the smallest extent if larger, in whole extents
    // of the size the memory plan would pick for it.
    const kv::Layout kvLayout = model.targetKvLayout(format);
    const uint32_t budgetPages = std::max(128U, kvLayout.minimumExtentPages());
    const uint32_t extentPages = kvLayout.extentPagesFor(budgetPages);
    const uint32_t pageCount = budgetPages - budgetPages % extentPages;
    const EngineMemoryBreakdown &budget = memoryPlan.breakdown();
    require(budget.pipelineReserveBytes <= budget.hardBudgetBytes &&
                budget.runtimeOverheadReserveBytes <
                    budget.hardBudgetBytes - budget.pipelineReserveBytes,
            "runtime reserves consume the oracle Metal budget");
    const uint64_t elasticGrowthCeiling = budget.hardBudgetBytes -
        budget.pipelineReserveBytes - budget.runtimeOverheadReserveBytes;
    MemoryGovernor governor(backend, elasticGrowthCeiling, hostReserveBytes,
                            queryHostAvailableMemory, 0);
    const metal::AllocationAdmission governed =
        [admit = governor.allocationAdmission(), &governor](
            uint64_t bytes, const std::function<void()> &allocate) {
          const metal::AllocationResult result = admit(bytes, allocate);
          if (result.failure == metal::AllocationFailure::HostPressure)
            stopForHostMemory(governor, "an allocation", bytes);
          return result;
        };
    AllocationFault allocationFault;
    const metal::AllocationAdmission admission =
        [admit = governed, &allocationFault, &backend](
            uint64_t bytes, const std::function<void()> &allocate)
            -> metal::AllocationResult {
          if (bytes > allocationFault.remainingBytes)
            return metal::AllocationFailure::EngineBudget;
          if (allocationFault.remaining == 0) {
            if (allocationFault.throwAfterAllocation) {
              require(static_cast<bool>(admit(bytes, allocate)),
                      "test allocation unexpectedly exceeded real budget");
              throw std::runtime_error("injected image allocation");
            }
            return metal::AllocationFailure::EngineBudget;
          }
          if (allocationFault.remaining > 0)
            --allocationFault.remaining;
          // An admission spends what it allocates.
          const uint64_t before = backend.memoryStats().allocatedBytes;
          if (const metal::AllocationResult result = admit(bytes, allocate); !result)
            return result;
          allocationFault.remainingBytes -=
              backend.memoryStats().allocatedBytes - before;
          return {};
        };
    kv::PageStorage pages(backend, governed, kvLayout, pageCount, extentPages);
    // The oracle's requests address pages directly, without a pool, so every
    // extent is allocated up front.
    for (uint32_t extent = 0; extent < pageCount / extentPages; ++extent)
      require(static_cast<bool>(pages.allocateExtent(extent)),
              "oracle KV extent is unavailable");
    model::QwenStateStorage states(backend,
                                    admission,
                                    model.stateLayout(), nullptr);
    model::RuntimeContext context{backend, model, pages, states, operators};
    require(executorPlan.sharedDecodePlannedAllocatedBytes <=
                std::numeric_limits<uint64_t>::max() -
                    executorPlan.sharedPrefillPlannedAllocatedBytes,
            "oracle runtime arena reservation overflows");
    const uint64_t arenaBytes = executorPlan.sharedPrefillPlannedAllocatedBytes +
                                executorPlan.sharedDecodePlannedAllocatedBytes;
    std::optional<model::Runtime> runtime;
    const metal::AllocationResult arenas = governor.allocationAdmission()(
        arenaBytes, [&] { runtime.emplace(context); });
    if (arenas.failure == metal::AllocationFailure::HostPressure)
      stopForHostMemory(governor, "the runtime arenas", arenaBytes);
    require(static_cast<bool>(arenas),
            "oracle runtime arenas exceed the oracle Metal budget");
    if (warmupEosOnly) {
      // The fixture builds its own runtimes one at a time in the admitted
      // runtime's place, outside the admission, so its failures are its own.
      runtime.reset();
      warmupEos(context, model);
      return 0;
    }
    model::Runtime &executor = *runtime;
    // Warmup runs on the KV runway and never allocates: without it, after
    // actual state activation, warmupPrefill fails and cleans up.
    pages.releaseExtent(0);
    {
      const uint64_t beforeWarmupRows = executor.telemetry().targetPrefillRows;
      const uint64_t beforeCommands =
          BackendInstrumentation::submittedCommands(backend);
      rejects([&] { static_cast<void>(executor.warmupPrefill(1)); },
              "is outside the startup runway", "real warmup ran without its KV runway");
      require(!states.metadata(0).assigned() &&
                  executor.telemetry().targetPrefillRows == beforeWarmupRows &&
                  BackendInstrumentation::submittedCommands(backend) ==
                      beforeCommands,
              "a refused warmup executed or leaked work");
    }
    require(static_cast<bool>(pages.allocateExtent(0)),
            "warmup runway fixture failed to recover its KV extent");
    while (states.releaseOneIdle(false)) {
    }
    // The engine refuses image requests to a model without vision before they
    // reach the runtime, which treats one as a broken invariant.
    if (model.descriptor.hasVision()) {
      requireAtomicImageAdmission(executor, backend, model, allocationFault);
      requireImageRowsAfterReclaim(executor, backend, states, model, allocationFault);
      requireRepeatedImagePlacements(executor, backend, states, allocationFault);
      requireConcurrentRequestsShareOneEncode(executor, backend);
      requireSuspendedLaneKeepsItsRows(executor, backend);
      requireInjectedRowsBecomeReclaimable(executor, backend);
      requireCoveredImagesAreNotStaged(executor, backend, model);
      requireRefusedStartKeepsItsRows(executor, backend, model, allocationFault);
      requireReclaimTakesOneCacheUnit(executor, backend, model);
      requireEncoderFitsItsImages(executor, backend, model, allocationFault);
      requireReplayPointKeepsItsImageRows(executor, backend);
    } else {
      require(!imagesOnly, "--images-only needs a model that serves vision");
      std::cout << "image scenarios: skipped, the model serves text only\n";
    }
    if (imagesOnly) {
      std::cout << "PASS model-runtime-oracle scope=images-only model=" << model.name()
                << " (admission rollback, chunk reclaim, cache-only budget, mixed/repeated images,"
                   " shared, suspended and injected rows, covered images, refused starts,"
                   " one cache unit per reclaim, replay point rows)\n";
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
    prefillChunk(executor, 1, 0, prompt128, pageTable);
    require(states.metadata(0).lengths.targetTokens == 128 &&
                states.metadata(0).lengths.hasCompleteDraftWindow(
                    model::ExecutionLimits::draftContextTokens),
            "prefill state length mismatch");

    const uint64_t predictedPromptSnapshotBytes =
        model.stateLayout().cachedBytes();
    std::shared_ptr<const CompositeState> promptSnapshot =
        executor.snapshot(1);
    require(promptSnapshot != nullptr,
            "prompt snapshot allocation failed");
    require(promptSnapshot->bytes() <= predictedPromptSnapshotBytes,
            "prompt snapshot exceeded preflight prediction");

    const uint64_t beforeFirstDecode =
        BackendInstrumentation::submittedCommands(backend);
    ModelStepResult decoded = decodeOne(executor, 1, 128, pageTable);
    require(BackendInstrumentation::submittedCommands(backend) ==
                beforeFirstDecode + 1,
            "speculative verify and commit were not one Metal command");
    require(!decoded.outputTokens.empty(), "decode produced no tokens");
    require(states.metadata(0).lengths.targetTokens ==
                    128 + decoded.outputTokens.size() &&
                states.metadata(0).lengths.hasCompleteDraftWindow(
                    model::ExecutionLimits::draftContextTokens),
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

    // The runtime builds every policy kernel when it is constructed: after
    // a greedy first token and verify, a sampled, penalized request's
    // compile nothing more.
    {
      const size_t pipelines = BackendInstrumentation::cachedPipelines(backend);
      EngineRequest sampled = makeRequest(2, prompt128, 16);
      sampled.sampling = {
          .temperature = 1.0F, .topP = 0.95F, .topK = 0, .seed = 4242};
      sampled.sampling.minP = 0.05F;
      sampled.sampling.repetitionPenalty = 1.1F;
      sampled.sampling.presencePenalty = 0.5F;
      sampled.sampling.frequencyPenalty = 0.5F;
      beginCold(executor, sampled, 1);
      const std::vector<uint32_t> sampledPages = pageRange(8, 8);
      static_cast<void>(firstStep(
          executor, prefillChunk(executor, 2, 0, prompt128, sampledPages), 2,
          128, sampledPages));
      require(BackendInstrumentation::cachedPipelines(backend) == pipelines,
              "a sampled, penalized request compiled a pipeline the runtime "
              "had not built at construction");
      executor.end(2);
    }

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
      beginCold(executor, scored, 0);
      ModelStepResult scoredResult =
          prefillChunk(executor, 99, 0, prompt128, pageTable);
      require(scoredResult.finished && scoredResult.outputTokens.empty() &&
                  scoredResult.scoreLogits.size() == 3,
              "score prefill did not return ordered logits without tokens");
      for (float logit : scoredResult.scoreLogits)
        require(std::isfinite(logit), "score logit is not finite");
      require(scoredResult.scoreLogits[0] >= scoredResult.scoreLogits[1] &&
                  scoredResult.scoreLogits[0] >= scoredResult.scoreLogits[2],
              "greedy decode token is not the maximum scored logit");
      rejects([&] { decodeOne(executor, 99, 128, pageTable); },
              "decode request has no current anchor", "score request allowed a decode step");
      executor.end(99);
    }
    requireNonFiniteRowFailsOnlyItsLane(executor, pages, states, backend,
                                        samplingSeedTokens, prompt128);


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
        prefillChunk(executor, 50, 0, prompt16, baselinePages);
    beginCold(executor, partitioned, 3);
    prefillChunk(executor, 51, 0,
                 std::span<const uint32_t>(prompt16).first(8),
                 partitionedPages);
    ModelStepResult partitionedPrefill =
        prefillChunk(executor, 51, 8,
                     std::span<const uint32_t>(prompt16).subspan(8, 8),
                     partitionedPages);

    const model::QwenLaneMetadata &baselineMetadata = states.metadata(2);
    const model::QwenLaneMetadata &partitionedMetadata = states.metadata(3);
    require(baselineMetadata.lengths == partitionedMetadata.lengths &&
                baselineMetadata.lengths.targetTokens == prompt16.size(),
            "partitioned prefill logical state diverged");
    const model::GdnStateLayout &gdnLayout = states.layout().target;
    const Similarity convolution =
        compareBfloat(convolutionHalf(backend, states.current(2), gdnLayout),
                      convolutionHalf(backend, states.current(3), gdnLayout));
    const Similarity recurrent =
        compareFloat(recurrentHalf(backend, states.current(2), gdnLayout),
                     recurrentHalf(backend, states.current(3), gdnLayout));
    std::cout
        << "partition_equivalence conv_cos=" << convolution.cosine
        << " parity=" << baselineMetadata.activeParity << '/'
        << partitionedMetadata.activeParity
        << " conv_norms=" << convolution.leftNorm << '/'
        << convolution.rightNorm << " partition_other_conv_norm="
        << compareBfloat(convolutionHalf(backend, states.current(2), gdnLayout),
                         convolutionHalf(backend, states.next(3), gdnLayout))
               .rightNorm
        << " recurrent_cos=" << recurrent.cosine
        << " recurrent_norms=" << recurrent.leftNorm << '/'
        << recurrent.rightNorm << '\n';
    require(convolution.cosine > 0.999 && recurrent.cosine > 0.999,
            "partitioned prefill numerical state diverged");

    ModelStepResult baselinePending =
        firstStep(executor, std::move(baselinePrefill), 50,
                  prompt16.size(), baselinePages);
    ModelStepResult partitionedPending =
        firstStep(executor, std::move(partitionedPrefill), 51,
                  prompt16.size(), partitionedPages);
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
    EngineRequest partitionedSampling = makeRequest(54, promptAligned, 1);
    partitionedSampling.sampling = {
        .temperature = 4.0F, .topP = 1.0F, .topK = 32, .seed = 8128};
    beginCold(executor, partitionedSampling, 0);
    prefillChunk(executor, 54, 0,
                 std::span<const uint32_t>(promptAligned).first(120),
                 policyPages);
    prefillChunk(executor, 54, 120,
                 std::span<const uint32_t>(promptAligned).subspan(120, 8),
                 policyPages);
    std::shared_ptr<const CompositeState> partitionedSamplingSnapshot =
        executor.snapshot(54);
    require(partitionedSamplingSnapshot != nullptr,
            "partitioned sampling snapshot allocation failed");
    executor.end(54);
    EngineRequest restoredMicroSampling =
        makeRequest(55, policyReplayPrompt, 2);
    restoredMicroSampling.sampling = partitionedSampling.sampling;
    beginCold(executor, restoredMicroSampling, 0);
    restoreActivePrefix(executor, 55, policyReplayPrompt.size(),
                        promptAligned.size(), partitionedSamplingSnapshot);
    ModelStepResult restoredMicroSample = firstStep(
        executor,
        prefillChunk(executor, 55, promptAligned.size(),
                     std::span<const uint32_t>(policyReplayPrompt).subspan(128,
                                                                          1),
                     policyPages),
        55, policyReplayPrompt.size(), policyPages);
    executor.end(55);
    const std::vector<uint32_t> coldPolicyPages = pageRange(15, 5);
    EngineRequest coldMicroSampling = restoredMicroSampling;
    coldMicroSampling.id = 57;
    beginCold(executor, coldMicroSampling, 0);
    prefillChunk(executor, 57, 0,
                 std::span<const uint32_t>(policyReplayPrompt).first(120),
                 coldPolicyPages);
    prefillChunk(executor, 57, 120,
                 std::span<const uint32_t>(policyReplayPrompt).subspan(120, 8),
                 coldPolicyPages);
    ModelStepResult coldMicroSample = firstStep(
        executor,
        prefillChunk(executor, 57, 128,
                     std::span<const uint32_t>(policyReplayPrompt).subspan(128,
                                                                          1),
                     coldPolicyPages),
        57, policyReplayPrompt.size(), coldPolicyPages);
    require(restoredMicroSample.outputTokens == coldMicroSample.outputTokens,
            "one-token recurrent restore replay diverged from cold prefill");
    executor.end(57);
    partitionedSamplingSnapshot.reset();

    const std::vector<uint32_t> prompt8(prompt16.begin(), prompt16.begin() + 8);
    const std::vector<uint32_t> constrainedPages{20};
    EngineRequest constrainedShort = makeRequest(56, prompt8, 1);
    constrainedShort.constraint = ConstraintMode::TokenMask;
    beginCold(executor, constrainedShort, 0);
    require(prefillChunk(executor, 56, 0, prompt8, constrainedPages)
                    .nextDecodeStage == DecodeStage::ApplyInitialMask,
            "constrained short prefill did not ask for its first mask");
    const std::array<uint32_t, 1> forcedMicroToken{106};
    provideMask(executor, 56, singletonMasks(forcedMicroToken));
    ModelStepResult forcedMicro =
        decodeOne(executor, 56, prompt8.size(), constrainedPages, true,
                  DecodeStage::ApplyInitialMask);
    require(forcedMicro.outputTokens ==
                std::vector<uint32_t>{forcedMicroToken.front()},
            "constrained short prefill final hidden is unusable");
    executor.end(56);

    // Every decode executes an anchor plus seven proposal rows. Compare each
    // budgeted commit with a constrained cycle that retains the same prefix,
    // rejecting the next proposal unless all eight rows are retained. The
    // constrained request has a larger budget, so both acceptance paths agree on
    // the exact GDN state and draft ring. Kernel tests cover the recurrence's
    // FP64 accuracy; this check does not mix prefill and decode summation
    // orders, whose tiny differences can amplify through the full model.
    for (uint32_t outputLimit = 2; outputLimit <= 8; ++outputLimit) {
      uint64_t id = 10 + outputLimit;
      EngineRequest variant = makeRequest(id, prompt129, outputLimit);
      beginCold(executor, variant, 0);
      restoreActivePrefix(executor, id, prompt129.size(), 128, promptSnapshot);
      ModelStepResult result = firstStep(
          executor,
          prefillChunk(executor, id, 128,
                       std::span<const uint32_t>(prompt129).subspan(128, 1),
                       pageTable),
          id, 129, pageTable);
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

      const uint64_t replayId = 100 + outputLimit;
      std::vector<uint32_t> replayPages = pageTable;
      // Composite snapshots exclude KV: share sealed history and give the
      // comparison its own writable page for the speculative suffix.
      replayPages[4] = 80;
      EngineRequest replayRequest =
          makeRequest(replayId, prompt129, static_cast<uint32_t>(stored + 2));
      replayRequest.constraint = ConstraintMode::TokenMask;
      beginCold(executor, replayRequest, 1);
      restoreActivePrefix(executor, replayId, prompt129.size(), 128,
                          promptSnapshot);
      require(prefillChunk(executor, replayId, 128,
                           std::span<const uint32_t>(prompt129).subspan(128, 1),
                           replayPages)
                      .nextDecodeStage == DecodeStage::ApplyInitialMask,
              "replay did not request its initial mask");
      const std::array<uint32_t, 1> firstAnchor{result.outputTokens.front()};
      provideMask(executor, replayId, singletonMasks(firstAnchor));
      static_cast<void>(decodeOne(executor, replayId, 129, replayPages, true,
                                  DecodeStage::ApplyInitialMask));
      auto pending =
          beginMaskedDecodeOne(executor, replayId, 129, replayPages);
      require(pending.maskRequests.size() == 1 &&
                  pending.maskRequests[0].simulationTokens.size() == 8,
              "replay proposals were not exposed");
      const auto &proposed = pending.maskRequests[0].simulationTokens;
      std::array<uint32_t, 9> maskTokens{};
      maskTokens.fill(100);
      for (size_t row = 0; row < stored; ++row) {
        require(proposed[row] == result.outputTokens[row],
                "replay proposal differs from committed token");
        maskTokens[row] = result.outputTokens[row];
      }
      if (stored < proposed.size())
        maskTokens[stored] = proposed[stored] == 101 ? 102 : 101;
      provideMask(executor, replayId, singletonMasks(maskTokens));
      const auto replayed = finishMaskedDecode(std::move(pending));
      require(replayed.size() == 1 &&
                  replayed[0].acceptedDraftTokens == stored - 1 &&
                  replayed[0].outputTokens.size() -
                          replayed[0].outputTokensWithoutKv == stored &&
                  states.metadata(1).lengths.targetTokens == 129 + stored,
              "replay did not commit exactly the supplied prefix");
      requireCommittedStateIdentical(
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
        prefillChunk(executor, 20, 128,
                     std::span<const uint32_t>(prompt129).subspan(128, 1),
                     pageTable),
        20, 129, pageTable);
    require(!extendedResult.outputTokens.empty() &&
                states.metadata(0).lengths.targetTokens ==
                    129 + extendedResult.outputTokens.size() -
                        extendedResult.outputTokensWithoutKv,
            "one-token replay boundary failed");
    executor.end(20);

    const std::vector<uint32_t> coldReplayPages = pageRange(21, 5);
    EngineRequest coldExtended = makeRequest(21, prompt129, 2);
    beginCold(executor, coldExtended, 0);
    prefillChunk(executor, 21, 0,
                 std::span<const uint32_t>(prompt129).first(128),
                 coldReplayPages);
    ModelStepResult coldExtendedResult = firstStep(
        executor,
        prefillChunk(executor, 21, 128,
                     std::span<const uint32_t>(prompt129).subspan(128, 1),
                     coldReplayPages),
        21, 129, coldReplayPages);
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
    EngineRequest samplingSource = makeRequest(30, samplingPrefix, 1);
    samplingSource.sampling = {
        .temperature = 4.0F, .topP = 1.0F, .topK = 32, .seed = 40106};
    beginCold(executor, samplingSource, 0);
    prefillChunk(executor, 30, 0,
                 std::span<const uint32_t>(samplingPrefix).first(120),
                 samplingPages);
    prefillChunk(executor, 30, 120,
                 std::span<const uint32_t>(samplingPrefix).subspan(120, 8),
                 samplingPages);
    std::shared_ptr<const CompositeState> samplingPromptSnapshot =
        executor.snapshot(30);
    require(samplingPromptSnapshot != nullptr,
            "sampling snapshot allocation failed");
    executor.end(30);

    EngineRequest replayedSampling = makeRequest(31, samplingPrompt, 2);
    replayedSampling.sampling = {
        .temperature = 4.0F, .topP = 1.0F, .topK = 32, .seed = 91199};
    beginCold(executor, replayedSampling, 0);
    restoreActivePrefix(executor, 31, samplingPrompt.size(), 128,
                        samplingPromptSnapshot);
    ModelStepResult replayedSample = firstStep(
        executor,
        prefillChunk(executor, 31, 128,
                     std::span<const uint32_t>(samplingPrompt).subspan(128, 1),
                     samplingPages),
        31, samplingPrompt.size(), samplingPages);
    require(!replayedSample.outputTokens.empty(),
            "replayed sampling hit did not emit an anchor");
    executor.end(31);

    const std::vector<uint32_t> coldSamplingPages = pageRange(31, 5);
    EngineRequest coldSampling = replayedSampling;
    coldSampling.id = 32;
    beginCold(executor, coldSampling, 0);
    prefillChunk(executor, 32, 0,
                 std::span<const uint32_t>(samplingPrompt).first(120),
                 coldSamplingPages);
    prefillChunk(executor, 32, 120,
                 std::span<const uint32_t>(samplingPrompt).subspan(120, 8),
                 coldSamplingPages);
    ModelStepResult coldSample = firstStep(
        executor,
        prefillChunk(executor, 32, 128,
                     std::span<const uint32_t>(samplingPrompt).subspan(128, 1),
                     coldSamplingPages),
        32, samplingPrompt.size(), coldSamplingPages);
    require(coldSample.outputTokens == replayedSample.outputTokens,
            "sampling restore replay reused producer policy state");
    executor.end(32);
    samplingPromptSnapshot.reset();

    const auto beforeConstrained = executor.telemetry();

    // The constraint handshake exposes the pending anchor plus all seven
    // DFlash proposals. The next mask therefore has nine rows: the current
    // anchor, seven proposal positions, and one target successor.
    EngineRequest constrained = makeRequest(40, prompt129, 2);
    constrained.constraint = ConstraintMode::TokenMask;
    beginCold(executor, constrained, 0);
    restoreActivePrefix(executor, 40, prompt129.size(), 128, promptSnapshot);
    require(prefillChunk(executor, 40, 128,
                         std::span<const uint32_t>(prompt129).subspan(128, 1),
                         pageTable)
                    .nextDecodeStage == DecodeStage::ApplyInitialMask,
            "initial constrained anchor did not request empty simulation");
    const uint32_t anchorA = 100;
    std::array<uint32_t, 1> initialTokens{anchorA};
    std::vector<uint32_t> initialWords = singletonMasks(initialTokens);
    require(initialWords.size() == kMaskWords,
            "empty simulation did not produce exactly one mask row");
    provideMask(executor, 40, initialWords);
    // The first token's selection drafts nothing and, with budget left,
    // emits nothing: the first cycle emits it.
    const ModelStepResult anchorSelection =
        decodeOne(executor, 40, 129, pageTable, true,
                  DecodeStage::ApplyInitialMask);
    require(anchorSelection.outputTokens.empty() &&
                anchorSelection.draftedTokens == 0 &&
                anchorSelection.nextDecodeStage == DecodeStage::Regular,
            "the first token's selection drafted or emitted a token");

    PendingMaskedDecode verify =
        beginMaskedDecodeOne(executor, 40, 129, pageTable);
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
    provideMask(executor, 40, verifyWords);
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
    EngineRequest perfectConstraint = makeRequest(44, prompt129, 4);
    perfectConstraint.constraint = ConstraintMode::TokenMask;
    beginCold(executor, perfectConstraint, 0);
    restoreActivePrefix(executor, 44, prompt129.size(), 128, promptSnapshot);
    require(prefillChunk(executor, 44, 128,
                         std::span<const uint32_t>(prompt129).subspan(128, 1),
                         pageTable)
                    .nextDecodeStage == DecodeStage::ApplyInitialMask,
            "perfect constrained accounting skipped its initial mask");
    const uint32_t perfectAnchor = 120;
    std::array<uint32_t, 1> perfectInitialTokens{perfectAnchor};
    provideMask(executor, 44, singletonMasks(perfectInitialTokens));
    static_cast<void>(decodeOne(executor, 44, 129, pageTable, true,
                                DecodeStage::ApplyInitialMask));
    PendingMaskedDecode perfectPending =
        beginMaskedDecodeOne(executor, 44, 129, pageTable);
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
    provideMask(executor, 44, singletonMasks(perfectVerify));
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

    EngineRequest alternateConstraint = makeRequest(41, prompt129, 1);
    alternateConstraint.constraint = ConstraintMode::TokenMask;
    beginCold(executor, alternateConstraint, 0);
    restoreActivePrefix(executor, 41, prompt129.size(), 128, promptSnapshot);
    require(prefillChunk(executor, 41, 128,
                         std::span<const uint32_t>(prompt129).subspan(128, 1),
                         pageTable)
                    .nextDecodeStage == DecodeStage::ApplyInitialMask,
            "second exact constrained hit skipped initial mask");
    const uint32_t anchorC = 105;
    std::array<uint32_t, 1> alternateTokens{anchorC};
    provideMask(executor, 41, singletonMasks(alternateTokens));
    ModelStepResult alternateOutput =
        decodeOne(executor, 41, 129, pageTable, true,
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
    // a different arena lane between draft and commit. One B2 plan selects
    // both lanes' first tokens under their masks; each lane's first cycle
    // runs alone and the B2 cycle continues both.
    EngineRequest crossLane0 = makeRequest(42, prompt129, 3);
    crossLane0.constraint = ConstraintMode::TokenMask;
    crossLane0.sampling = {
        .temperature = 4.0F, .topP = 1.0F, .topK = 32, .seed = 7001};
    EngineRequest crossLane1 = crossLane0;
    crossLane1.id = 43;
    crossLane1.sampling.seed = 7002;
    beginCold(executor, crossLane0, 0);
    restoreActivePrefix(executor, 42, prompt129.size(), 128, promptSnapshot);
    beginCold(executor, crossLane1, 1);
    restoreActivePrefix(executor, 43, prompt129.size(), 128, promptSnapshot);
    const std::vector<uint32_t> crossPages0{0, 1, 2, 3, 36};
    const std::vector<uint32_t> crossPages1{0, 1, 2, 3, 37};
    BatchPlan crossReplayPlan{.kind = WorkKind::Prefill,
                              .items = {{42, 1}, {43, 1}},
                              .decodeStage = DecodeStage::Regular};
    std::array<ModelBatchItem, 2> crossReplayItems{
        withRevision({.requestId = 42, .logicalPosition = 128, .tokenCount = 1,
                      .pageTable = crossPages0}),
        withRevision({.requestId = 43, .logicalPosition = 128, .tokenCount = 1,
                      .pageTable = crossPages1})};
    const auto crossReplayToken =
        std::span<const uint32_t>(prompt129).subspan(128, 1);
    crossReplayItems[0].inputTokens = crossReplayToken;
    crossReplayItems[1].inputTokens = crossReplayToken;
    auto crossReplay = executor.prefill(crossReplayPlan, crossReplayItems);
    require(crossReplay.size() == 2 &&
                crossReplay[0].consumedPromptTokens == 1 &&
                crossReplay[1].consumedPromptTokens == 1 &&
                crossReplay[0].nextDecodeStage ==
                    DecodeStage::ApplyInitialMask &&
                crossReplay[1].nextDecodeStage ==
                    DecodeStage::ApplyInitialMask,
            "B2 recurrent restore did not replay one complete token");
    const std::array<uint32_t, 2> crossAnchors{110, 111};
    provideMask(executor, 42, singletonMasks(std::span(crossAnchors).first(1)));
    provideMask(executor, 43, singletonMasks(std::span(crossAnchors).last(1)));
    const BatchPlan crossInitialPlan{.kind = WorkKind::Decode,
                                     .constrained = true,
                                     .items = {{42, 0}, {43, 0}},
                                     .decodeStage = DecodeStage::ApplyInitialMask};
    const std::vector<ModelBatchItem> crossItems{
        withRevision({.requestId = 42, .logicalPosition = 129, .pageTable = crossPages0}),
        withRevision({.requestId = 43, .logicalPosition = 129, .pageTable = crossPages1})};
    const auto crossInitial = executor.decode(crossInitialPlan, crossItems);
    require(crossInitial.size() == 2 && crossInitial[0].outputTokens.empty() &&
                crossInitial[1].outputTokens.empty() &&
                crossInitial[0].nextDecodeStage == DecodeStage::Regular &&
                crossInitial[1].nextDecodeStage == DecodeStage::Regular,
            "a B2 plan did not select both lanes' first tokens");
    // Each lane's first cycle keeps its anchor; the masked successor that
    // rejects the second proposal becomes its next anchor.
    std::array<uint32_t, 2> crossNext{};
    for (uint32_t lane = 0; lane < 2; ++lane) {
      const uint64_t id = 42 + lane;
      PendingMaskedDecode initial = beginMaskedDecodeOne(
          executor, id, 129, lane ? crossPages1 : crossPages0);
      require(initial.maskRequests.size() == 1 &&
                  initial.maskRequests[0].simulationTokens.front() ==
                      crossAnchors[lane],
              "B1 initial constrained cycle did not draft from its anchor");
      crossNext[lane] =
          initial.maskRequests[0].simulationTokens[1] == 112 ? 113 : 112;
      const std::array<uint32_t, 9> verify{
          crossAnchors[lane], crossNext[lane], 114, 115, 116, 117, 118, 119, 120};
      provideMask(executor, id, singletonMasks(verify));
      const auto initialResults = finishMaskedDecode(std::move(initial));
      require(initialResults.size() == 1 &&
                  initialResults[0].outputTokens ==
                      std::vector<uint32_t>{crossAnchors[lane]} &&
                  initialResults[0].outputTokensWithoutKv == 0,
              "B1 initial constrained cycle did not keep only its anchor");
    }
    BatchPlan crossPlan{.kind = WorkKind::Decode,
                        .constrained = true,
                        .items = {{42, 0}, {43, 0}},
                        .decodeStage = DecodeStage::Regular};
    const std::vector<ModelBatchItem> crossCycleItems{
        withRevision({.requestId = 42, .logicalPosition = 130, .pageTable = crossPages0}),
        withRevision({.requestId = 43, .logicalPosition = 130, .pageTable = crossPages1})};
    PendingMaskedDecode crossPending =
        beginMaskedDecode(executor, crossPlan, crossCycleItems);
    require(crossPending.maskRequests.size() == 2 &&
                crossPending.ticket->ownsMaskWait(42) &&
                crossPending.ticket->ownsMaskWait(43),
            "B2 target forward did not own both constraint masks");
    std::array<std::array<uint32_t, 9>, 2> crossVerify{};
    for (uint32_t lane = 0; lane < 2; ++lane) {
      const auto &simulation =
          crossPending.maskRequests[lane].simulationTokens;
      const uint32_t rejected = simulation[1] == 121 ? 122 : 121;
      crossVerify[lane] =
          {crossNext[lane], rejected, 123, 124, 125, 126, 127, 128, 129};
      provideMask(executor, crossPending.maskRequests[lane].requestId,
                  singletonMasks(crossVerify[lane]));
    }
    auto crossResults = finishMaskedDecode(std::move(crossPending));
    require(crossResults.size() == 2 &&
                crossResults[0].outputTokens ==
                    std::vector<uint32_t>{crossNext[0], crossVerify[0][1]} &&
                crossResults[1].outputTokens ==
                    std::vector<uint32_t>{crossNext[1], crossVerify[1][1]} &&
                crossResults[0].outputTokensWithoutKv == 1 &&
                crossResults[1].outputTokensWithoutKv == 1 &&
                crossResults[0].draftedTokens == 7 &&
                crossResults[1].draftedTokens == 7 &&
                crossResults[0].acceptedDraftTokens == 0 &&
                crossResults[1].acceptedDraftTokens == 0,
            "B2 constrained overlap corrupted a lane");
    const model::ModelTelemetry constrainedTelemetry =
        executor.telemetry();
    require(constrainedTelemetry.constrainedMaskOverlapBatches -
                    beforeConstrained.constrainedMaskOverlapBatches == 5 &&
                constrainedTelemetry.constrainedMaskOverlapRequests -
                    beforeConstrained.constrainedMaskOverlapRequests == 6 &&
                constrainedTelemetry.totalConstrainedTargetForwardGpuSeconds >
                    beforeConstrained.totalConstrainedTargetForwardGpuSeconds,
            "constrained overlap telemetry does not match B1/B2 execution");
    executor.end(42);
    executor.end(43);

    // Width three is a real M24 graph, never a B2+B1 decomposition or a
    // rendezvous for a fourth request. State lanes are intentionally permuted
    // to prove that batch lanes belong to plan order.
    constexpr std::array<uint64_t, 3> b3Ids{60, 61, 62};
    constexpr std::array<uint32_t, 3> b3StateLanes{2, 0, 3};
    std::array<std::vector<uint32_t>, 3> b3Pages{std::vector<uint32_t>{40},
                                                 std::vector<uint32_t>{41},
                                                 std::vector<uint32_t>{42}};
    constexpr std::array<uint32_t, 3> b3Words{279, 314, 264};
    for (uint32_t lane = 0; lane < b3Ids.size(); ++lane) {
      beginCold(executor, makeRequest(b3Ids[lane], {b3Words[lane]}, 16),
                b3StateLanes[lane]);
      prefillToken(executor, b3Ids[lane], 0, b3Words[lane],
                   b3Pages[lane]);
    }
    BatchPlan b3Plan{.kind = WorkKind::Decode,
                     .items = {{b3Ids[0], 0}, {b3Ids[1], 0}, {b3Ids[2], 0}},
                     .decodeStage = DecodeStage::Regular};
    std::array<ModelBatchItem, 3> b3Items{
        withRevision({.requestId = b3Ids[0], .logicalPosition = 1, .pageTable = b3Pages[0]}),
        withRevision({.requestId = b3Ids[1], .logicalPosition = 1, .pageTable = b3Pages[1]}),
        withRevision({.requestId = b3Ids[2], .logicalPosition = 1, .pageTable = b3Pages[2]})};
    auto b3Decoded = executor.decode(b3Plan, b3Items);
    require(b3Decoded.size() == 3 && !b3Decoded[0].outputTokens.empty() &&
                !b3Decoded[1].outputTokens.empty() &&
                !b3Decoded[2].outputTokens.empty(),
            "B3 decode did not run one three-lane graph");
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
      prefillToken(executor, equivalentIds[lane], 0, 279,
                   equivalentPages[lane]);
    }
    BatchPlan equivalentPlan;
    equivalentPlan.kind = WorkKind::Decode;
    std::array<ModelBatchItem, 4> equivalentItems;
    for (uint32_t lane = 0; lane < equivalentIds.size(); ++lane) {
      equivalentPlan.items.push_back({equivalentIds[lane], 0});
      equivalentItems[lane] =
          withRevision({.requestId = equivalentIds[lane], .logicalPosition = 1,
                        .pageTable = equivalentPages[lane]});
    }
    auto equivalentB4 = executor.decode(equivalentPlan, equivalentItems);
    for (uint64_t id : equivalentIds)
      executor.end(id);
    beginCold(executor, makeRequest(68, {279}, 16), 0);
    prefillToken(executor, 68, 0, 279, {47});
    ModelStepResult equivalentB1 = decodeOne(executor, 68, 1, {47});
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
    prefillChunk(executor, 70, 0, productionPrefix, productionPages);
    std::shared_ptr<const CompositeState> productionSnapshot =
        executor.snapshot(70);
    require(productionSnapshot != nullptr,
            "production snapshot allocation failed");
    executor.end(70);

    // One ragged command consumes exactly 2048 real, unequal rows. Repeating
    // its M32 decode with permuted lanes proves ragged addressing and state
    // isolation without requiring another batch width's numerical decisions.
    constexpr std::array<uint64_t, 4> raggedIds{100, 101, 102, 103};
    constexpr std::array<uint32_t, 4> raggedStateLanes{3, 1, 0, 2};
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
      auto value = makeRequest(id, raggedPrompts[lane], 16);
      value.sampling = {.temperature = sampled ? 0.8F : 0.0F,
                        .topP = 0.95F,
                        .topK = 20,
                        .seed = 731 + lane};
      return value;
    };
    BatchPlan raggedPrefillPlan;
    raggedPrefillPlan.kind = WorkKind::Prefill;
    std::array<ModelBatchItem, 4> raggedPrefillItems;
    for (uint32_t lane = 0; lane < raggedIds.size(); ++lane) {
      beginCold(executor,
          raggedRequest(raggedIds[lane], lane),
          raggedStateLanes[lane]);
      raggedPrefillPlan.items.push_back({raggedIds[lane], raggedRows[lane]});
      raggedPrefillItems[lane] = withRevision({.requestId = raggedIds[lane],
                                               .tokenCount = raggedRows[lane],
                                               .pageTable = raggedPages[lane]});
      raggedPrefillItems[lane].inputTokens = raggedPrompts[lane];
    }
    const uint64_t beforeRaggedPrefill =
        BackendInstrumentation::submittedCommands(backend);
    auto raggedPrefill =
        executor.prefill(raggedPrefillPlan, raggedPrefillItems);
    require(raggedPrefill.size() == raggedIds.size() &&
                BackendInstrumentation::submittedCommands(backend) ==
                    beforeRaggedPrefill + 1,
            "ragged 2048-row prefill was not one Metal command");
    for (uint32_t lane = 0; lane < raggedIds.size(); ++lane) {
      require(raggedPrefill[lane].consumedPromptTokens == raggedRows[lane] &&
                  states.metadata(raggedStateLanes[lane]).lengths.targetTokens ==
                      raggedRows[lane],
              "ragged prefill consumed or addressed the wrong rows");
      requireOpen(raggedPrefill[lane], "ragged prefill");
    }

    BatchPlan raggedDecodePlan;
    raggedDecodePlan.kind = WorkKind::Decode;
    std::array<ModelBatchItem, 4> raggedDecodeItems;
    for (uint32_t lane = 0; lane < raggedIds.size(); ++lane) {
      raggedDecodePlan.items.push_back({raggedIds[lane], 0});
      raggedDecodeItems[lane] = withRevision({.requestId = raggedIds[lane],
                                              .logicalPosition = raggedRows[lane],
                                              .pageTable = raggedPages[lane]});
    }
    auto raggedDecoded = executor.decode(raggedDecodePlan, raggedDecodeItems);
    require(raggedDecoded.size() == raggedIds.size(),
            "permuted ragged B4 did not run one four-lane graph");
    for (uint64_t id : raggedIds)
      executor.end(id);

    // Re-run the same real M32 workload with request order and state lanes
    // permuted. This isolates cross-lane addressing without conflating M32
    // with the independently optimized M8 numerical path.
    constexpr std::array<uint32_t, 4> raggedPermutation{2, 0, 3, 1};
    constexpr std::array<uint32_t, 4> referenceStateLanes{1, 3, 0, 2};
    BatchPlan raggedReferencePrefillPlan;
    raggedReferencePrefillPlan.kind = WorkKind::Prefill;
    std::array<ModelBatchItem, 4> raggedReferencePrefillItems;
    for (uint32_t order = 0; order < raggedPermutation.size(); ++order) {
      const uint32_t lane = raggedPermutation[order];
      const uint64_t referenceId = 104 + lane;
      beginCold(executor,
          raggedRequest(referenceId, lane),
          referenceStateLanes[order]);
      raggedReferencePrefillPlan.items.push_back(
          {referenceId, raggedRows[lane]});
      raggedReferencePrefillItems[order] = withRevision(
          {.requestId = referenceId, .tokenCount = raggedRows[lane], .pageTable = raggedPages[lane]});
      raggedReferencePrefillItems[order].inputTokens = raggedPrompts[lane];
    }
    auto raggedReferencePrefill = executor.prefill(raggedReferencePrefillPlan,
                                                   raggedReferencePrefillItems);
    require(raggedReferencePrefill.size() == raggedPermutation.size(),
            "permuted ragged reference prefill width mismatch");

    BatchPlan raggedReferenceDecodePlan;
    raggedReferenceDecodePlan.kind = WorkKind::Decode;
    std::array<ModelBatchItem, 4> raggedReferenceDecodeItems;
    for (uint32_t order = 0; order < raggedPermutation.size(); ++order) {
      const uint32_t lane = raggedPermutation[order];
      const uint64_t referenceId = 104 + lane;
      raggedReferenceDecodePlan.items.push_back({referenceId, 0});
      raggedReferenceDecodeItems[order] = withRevision(
          {.requestId = referenceId, .logicalPosition = raggedRows[lane],
           .pageTable = raggedPages[lane]});
    }
    auto raggedReferenceDecoded =
        executor.decode(raggedReferenceDecodePlan, raggedReferenceDecodeItems);
    require(raggedReferenceDecoded.size() == raggedPermutation.size(),
            "permuted ragged reference was not one four-lane graph");
    for (uint32_t order = 0; order < raggedPermutation.size(); ++order) {
      const uint32_t lane = raggedPermutation[order];
      const ModelStepResult &reference = raggedReferenceDecoded[order];
      require(reference.outputTokens == raggedDecoded[lane].outputTokens &&
                  reference.acceptedDraftTokens ==
                      raggedDecoded[lane].acceptedDraftTokens,
              "ragged M32 lane changed after order/state-lane permutation");
      executor.end(104 + lane);
    }

    // Lanes that finish their prompts in one ragged prefill share one LM
    // head and one selection. Kernels choose their tiles by a command's
    // rows, so the reference for a lane is a command of the same rows, not
    // the lane alone. With the lanes in the reverse order, each finishing
    // lane must select what it selected and a score lane must read the same
    // logits, and a lane whose prompt the command does not finish must stay
    // out of the head and end its prompt as it did. With every other lane's
    // prompt changed, a score lane must still read the same logits.
    {
      constexpr uint32_t kScoredRow = 0, kGreedy = 1, kOpen = 2, kSampled = 3;
      constexpr std::array<uint32_t, 4> rows{33, 40, 64, 72};
      constexpr uint32_t openPromptTokens = 200;
      // A lane's prompt, or another one of the same length.
      const auto prompt = [&](uint32_t lane, bool other) {
        std::vector<uint32_t> tokens(lane == kOpen ? openPromptTokens
                                                   : rows[lane]);
        for (uint32_t row = 0; row < tokens.size(); ++row)
          tokens[row] = productionSeedTokens[(row + 3 * lane + (other ? 17 : 0)) %
                                             productionSeedTokens.size()];
        return tokens;
      };
      const auto requestFor = [&](uint64_t id, uint32_t lane, bool other) {
        EngineRequest value = makeRequest(id, prompt(lane, other), 1);
        if (lane == kScoredRow) {
          value.maxNewTokens = 0;
          value.scoreTokens = {11, 220, 1683};
        } else if (lane == kSampled) {
          value.sampling = {
              .temperature = 0.8F, .topP = 0.95F, .topK = 20, .seed = 4099};
        }
        return value;
      };
      const std::array<std::vector<uint32_t>, 4> pages{
          pageRange(52, 2), pageRange(54, 2), pageRange(56, 7),
          pageRange(63, 3)};
      // One ragged prefill of the four lanes, each on its own state lane, in
      // `order`; `others` gives every lane but the score lane another
      // prompt. The open lane's prompt then ends in a command of its own.
      // The results are by lane.
      uint64_t nextId = 140;
      const auto prefillInOrder = [&](const std::array<uint32_t, 4> &order,
                                      bool others) {
        BatchPlan plan{.kind = WorkKind::Prefill,
                       .decodeStage = DecodeStage::Regular};
        std::array<EngineRequest, 4> requests;
        std::array<ModelBatchItem, 4> items;
        for (uint32_t position = 0; position < order.size(); ++position) {
          const uint32_t lane = order[position];
          requests[lane] =
              requestFor(nextId++, lane, others && lane != kScoredRow);
          beginCold(executor, requests[lane], lane);
          plan.items.push_back({requests[lane].id, rows[lane]});
          items[position] = withRevision({.requestId = requests[lane].id,
                                          .tokenCount = rows[lane],
                                          .pageTable = pages[lane]});
          items[position].inputTokens =
              std::span(requests[lane].prompt).first(rows[lane]);
        }
        std::vector<ModelStepResult> byPosition = executor.prefill(plan, items);
        require(byPosition.size() == items.size(), "ragged prefill width mismatch");
        std::array<ModelStepResult, 4> results;
        for (uint32_t position = 0; position < order.size(); ++position)
          results[order[position]] = std::move(byPosition[position]);
        require(results[kOpen].outputTokens.empty() && !results[kOpen].finished &&
                    states.metadata(kOpen).lengths.targetTokens == rows[kOpen],
                "a lane that did not finish its prompt took part in the head");
        const EngineRequest &open = requests[kOpen];
        results[kOpen] = prefillChunk(executor, open.id, rows[kOpen],
                                      std::span(open.prompt).subspan(rows[kOpen]),
                                      pages[kOpen]);
        for (const EngineRequest &request : requests)
          executor.end(request.id);
        return results;
      };
      const auto sameScores = [](const ModelStepResult &left,
                                 const ModelStepResult &right,
                                 const char *message) {
        require(left.scoreLogits.size() == 3 && right.scoreLogits.size() == 3,
                "the score lane of a shared head returned no logits");
        for (uint32_t option = 0; option < 3; ++option)
          require(std::fabs(left.scoreLogits[option] - right.scoreLogits[option]) <=
                      1e-3F * std::max(1.0F, std::fabs(right.scoreLogits[option])),
                  message);
      };
      const auto inOrder =
          prefillInOrder({kScoredRow, kGreedy, kOpen, kSampled}, false);
      const auto reversed =
          prefillInOrder({kSampled, kOpen, kGreedy, kScoredRow}, false);
      for (const uint32_t lane : {kGreedy, kOpen, kSampled}) {
        require(reversed[lane].outputTokens == inOrder[lane].outputTokens &&
                    inOrder[lane].outputTokens.size() == 1,
                "a lane selected another first token beside the lanes in "
                "another order");
      }
      sameScores(reversed[kScoredRow], inOrder[kScoredRow],
                 "a score lane read other logits beside the lanes in another "
                 "order");
      const auto others =
          prefillInOrder({kScoredRow, kGreedy, kOpen, kSampled}, true);
      sameScores(others[kScoredRow], inOrder[kScoredRow],
                 "a score lane's logits changed with the other lanes' prompts");
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
    BatchPlan productionB4ReplayPlan{.kind = WorkKind::Prefill,
                                     .items = {{productionB4Ids[0], 1},
                                               {productionB4Ids[1], 1},
                                               {productionB4Ids[2], 1},
                                               {productionB4Ids[3], 1}},
                                     .decodeStage = DecodeStage::Regular};
    std::array<ModelBatchItem, 4> productionB4ReplayItems;
    for (uint32_t lane = 0; lane < productionB4Ids.size(); ++lane) {
      productionB4ReplayItems[lane] = withRevision(
          {.requestId = productionB4Ids[lane], .logicalPosition = productionPrefix.size(),
           .tokenCount = 1, .pageTable = productionB4Pages[lane]});
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
      std::array<ModelBatchItem, 4> cycleItems;
      for (uint32_t lane = 0; lane < productionB4Ids.size(); ++lane) {
        cyclePlan.items.push_back({productionB4Ids[lane], 0});
        cycleItems[lane] = withRevision({.requestId = productionB4Ids[lane],
                                         .logicalPosition = productionB4Lengths[lane],
                                         .pageTable = productionB4Pages[lane]});
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
    constexpr std::array<uint32_t, 2> productionB2StateLanes{3, 1};
    std::array<std::vector<uint32_t>, 2> productionB2Pages{
        std::vector<uint32_t>{48, 49, 50, 51, 76},
        std::vector<uint32_t>{48, 49, 50, 51, 77}};
    for (uint32_t lane = 0; lane < productionB2Ids.size(); ++lane) {
      beginCold(executor,
          makeRequest(productionB2Ids[lane], productionPrompt, 16),
          productionB2StateLanes[lane]);
      restoreActivePrefix(executor, productionB2Ids[lane],
                          productionPrompt.size(), productionPrefix.size(),
                          productionSnapshot);
    }
    BatchPlan productionB2ReplayPlan{
        .kind = WorkKind::Prefill,
        .items = {{productionB2Ids[0], 1}, {productionB2Ids[1], 1}},
        .decodeStage = DecodeStage::Regular};
    std::array<ModelBatchItem, 2> productionB2ReplayItems;
    for (uint32_t lane = 0; lane < productionB2Ids.size(); ++lane) {
      productionB2ReplayItems[lane] = withRevision(
          {.requestId = productionB2Ids[lane], .logicalPosition = productionPrefix.size(),
           .tokenCount = 1, .pageTable = productionB2Pages[lane]});
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
      std::array<ModelBatchItem, 2> cycleItems;
      for (uint32_t lane = 0; lane < productionB2Ids.size(); ++lane) {
        cyclePlan.items.push_back({productionB2Ids[lane], 0});
        cycleItems[lane] = withRevision({.requestId = productionB2Ids[lane],
                                         .logicalPosition = productionB2Lengths[lane],
                                         .pageTable = productionB2Pages[lane]});
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
        require(states.metadata(productionB2StateLanes[lane]).lengths.targetTokens ==
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
    requireOpen(prefillChunk(executor, 75, productionPrefix.size(),
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
          decodeOne(executor, 75, productionB1Length, productionB1Pages);
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
    enum class RequestKind : uint32_t { Greedy, Sampling, Constrained };
    // A sampling request keeps the tokens minP leaves it, the topK of those,
    // all of them for 0, then its topP nucleus.
    const auto runPreemption = [&](RequestKind kind, uint32_t preemptionMode,
                                   ops::SamplingPenalties penalties = {},
                                   uint32_t topK = 20, float topP = 0.95F,
                                   uint32_t flags = 0, float minP = 0.0F) {
      const bool preempt = preemptionMode != 0;
      const bool constrained = kind == RequestKind::Constrained;
      EngineRequest sequence = makeRequest(80, prompt129, 24);
      sequence.flags = flags;
      if (kind == RequestKind::Sampling) {
        sequence.sampling = {.temperature = 0.8F,
                             .topP = topP,
                             .topK = topK,
                             .minP = minP,
                             .seed = 91199};
      }
      if (constrained)
        sequence.constraint = ConstraintMode::TokenMask;
      sequence.sampling.presencePenalty = penalties.presence;
      sequence.sampling.frequencyPenalty = penalties.frequency;
      sequence.sampling.repetitionPenalty = penalties.repetition;
      beginCold(executor, sequence, 0);
      uint32_t stateLane = 0;
      // A penalized request resumes in a state lane other than 0 and decodes
      // in batch lane 0: its penalty words follow the state lane, not the
      // batch lane.
      const auto resume = [&] {
        std::optional<EngineRequest> holder;
        if (penalties.active()) {
          holder = makeRequest(81, {1}, 1);
          beginCold(executor, *holder, 0);
        }
        StateAdmission admission = executor.resume(sequence.modelView());
        if (holder)
          executor.end(holder->id);
        require(admission.granted() && (!holder || *admission.lane != 0),
                "recompute admission failed");
        return *admission.lane;
      };
      const auto rebuild = [&](bool repeatDuringReplay,
                                bool deliverInitialMask = false) {
        const StateSamples before =
            repeatDuringReplay ? sampleCommittedState(backend, states, stateLane)
                               : StateSamples{};
        executor.suspend(sequence.id);
        require(!states.metadata(stateLane).assigned(),
                "preempted request retained its GDN/draft buffers");
        if (deliverInitialMask) {
          const std::array<uint32_t, 1> anchor{100};
          provideMask(executor, sequence.id, singletonMasks(anchor));
        }
        stateLane = resume();
        const uint32_t length = static_cast<uint32_t>(sequence.prompt.size());
        executor.setDraftContextPlan(sequence.id, planDraftContext(0, length, {}));
        if (repeatDuringReplay) {
          requireOpen(prefillChunk(executor, sequence.id, 0,
                                   std::span(sequence.prompt).first(32),
                                   pageTable),
                      "interrupted state replay");
          executor.suspend(sequence.id);
          require(!states.metadata(stateLane).assigned(),
                  "repeated preemption retained its state buffers");
          stateLane = resume();
          executor.setDraftContextPlan(sequence.id, planDraftContext(0, length, {}));
        }
        ModelStepResult replay = prefillChunk(
            executor, sequence.id, 0, sequence.prompt, pageTable);
        requireOpen(replay, "regeneration replay emitted historical tokens");
        if (repeatDuringReplay) {
          const StateSamples rebuilt = sampleCommittedState(backend, states, stateLane);
          // Decode and prefill use different floating-point graphs. Record
          // that drift, but compare recovery itself to an independent cold
          // teacher-forced execution with the identical history and geometry.
          compareCommittedSamples(before, rebuilt, false);
          EngineRequest teacher = sequence;
          teacher.id = 82;
          const uint32_t teacherLane = stateLane == 0 ? 1 : 0;
          const auto teacherPages = pageRange(80, 8);
          beginCold(executor, teacher, teacherLane);
          static_cast<void>(prefillChunk(executor, teacher.id, 0,
                                        teacher.prompt, teacherPages));
          require(states.metadata(stateLane).lengths == states.metadata(teacherLane).lengths,
                  "recomputed logical lengths differ from teacher forcing");
          compareCommittedSamples(
              rebuilt, sampleCommittedState(backend, states, teacherLane), true);
          executor.end(teacher.id);
        }
        return replay.nextDecodeStage;
      };
      DecodeStage stage = DecodeStage::Regular;
      if (preempt) {
        requireOpen(prefillChunk(executor, sequence.id, 0,
                                 std::span(sequence.prompt).first(32),
                                 pageTable),
                    "unfinished prefill before preemption");
        stage = rebuild(false);
      } else {
        const ModelStepResult promptEnd = prefillChunk(
            executor, sequence.id, 0, sequence.prompt, pageTable);
        requireOpen(promptEnd, "preemption prompt");
        stage = promptEnd.nextDecodeStage;
      }
      require(stage == (constrained ? DecodeStage::ApplyInitialMask
                                    : DecodeStage::Regular),
              "the prompt's end misreported its first mask");
      // Preempted at its prompt's end, a request replays without asking for
      // its first mask again; a constrained one is given it meanwhile.
      if (preempt)
        require(rebuild(false, constrained) == DecodeStage::Regular,
                "prompt-end preemption asked for the first mask again");
      if (constrained) {
        if (!preempt) {
          const std::array<uint32_t, 1> anchor{100};
          provideMask(executor, sequence.id, singletonMasks(anchor));
        }
        const ModelStepResult selected =
            decodeOne(executor, sequence.id, sequence.prompt.size(), pageTable,
                      true, DecodeStage::ApplyInitialMask);
        require(selected.outputTokens.empty(),
                "the first token's selection emitted a token with budget left");
        stage = selected.nextDecodeStage;
      }
      PreemptionRun run;
      auto &transcript = run.transcript;
      bool generationPreempted = false;
      for (uint32_t cycle = 0; cycle < 24 && transcript.size() < 24; ++cycle) {
        ModelStepResult result;
        if (constrained) {
          auto pending = beginMaskedDecodeOne(
              executor, sequence.id, sequence.prompt.size(), pageTable);
          std::array<uint32_t, 9> forced;
          for (uint32_t row = 0; row < forced.size(); ++row)
            forced[row] = 100 + static_cast<uint32_t>(transcript.size()) + row;
          provideMask(executor, sequence.id, singletonMasks(forced));
          result = finishMaskedDecode(std::move(pending)).front();
          for (size_t row = 0; row < result.outputTokens.size(); ++row)
            require(row < forced.size() && result.outputTokens[row] == forced[row],
                    "preempted constrained decode violated its provided mask");
        } else {
          result = decodeOne(executor, sequence.id, sequence.prompt.size(),
                             pageTable, false, stage);
        }
        require(!result.outputTokens.empty(), "preempted decode made no progress");
        require(result.outputTokens.size() <= sequence.maxNewTokens - transcript.size() &&
                    result.acceptedDraftTokens <= result.draftedTokens &&
                    result.outputTokensWithoutKv <= result.outputTokens.size(),
                "preempted decode output accounting is invalid");
        if (kind == RequestKind::Sampling)
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
    for (RequestKind kind : {RequestKind::Greedy, RequestKind::Sampling,
                             RequestKind::Constrained}) {
      const auto reference = runPreemption(kind, 0);
      const auto promptResumed = runPreemption(kind, 1);
      // The interrupted 32-row prefix is discarded. Both prompt-only paths
      // finish with the same full 129-row prefill and policy continuation.
      require(promptResumed.transcript == reference.transcript,
              "prompt recomputation changed the request policy/initial anchor");
      const auto resumed = runPreemption(kind, 2);
      require(resumed.transcript.size() > resumed.pendingAnchorIndex &&
                  reference.transcript.size() > resumed.pendingAnchorIndex &&
                  std::equal(resumed.transcript.begin(),
                             resumed.transcript.begin() + resumed.pendingAnchorIndex + 1,
                             reference.transcript.begin()),
              "preemption changed already-emitted history or its pending anchor");
      if (kind == RequestKind::Sampling) {
        const auto repeated = runPreemption(kind, 2);
        require(resumed.transcript == repeated.transcript,
                "fixed-seed recomputation is not deterministic for the same schedule");
      }
      // Generated history is rebuilt through ragged prefill, whose numerical
      // path differs from decode. Already-emitted tokens and the pending
      // anchor remain exact above; subsequent decisions are diagnostic.
      std::cout << "preemption_decisions kind=" << static_cast<uint32_t>(kind)
                << " transcript_equal=" << (resumed.transcript == reference.transcript)
                << " reference_rows=" << reference.transcript.size()
                << " resumed_rows=" << resumed.transcript.size()
                << " preserved_prefix_rows=" << resumed.pendingAnchorIndex + 1 << '\n';
    }

    // Score probe of a greedy transcript: for every emitted token, a score
    // request on the context before it returns the raw logits of every
    // prompt and earlier output token and of the emitted one, the host
    // applies the penalties with the counts of the tokens emitted before it,
    // and the emitted token must be the best of them. The penalties only
    // move tokens the context holds, so a count the runtime got wrong (in
    // the prefill's first token, a verify row's draft prefix, or the words a
    // resume rebuilds) shows up as an option that beats the emitted token by
    // about the penalty. The score logits come from the prefill graph and
    // most decisions from the verify graph, whose logits differ by a share of
    // their size: on the 35B UD-Q4_K_M on Apple9, a prompt token's logit
    // divided by the repetition ties a new token's at a repetition 1-2% lower
    // in verify than in prefill (1.29-1.30 against 1.31-1.32). So the emitted
    // token may trail by graphDrift of the values compared, well inside what
    // a wrong count moves (presence here at least 1.5, repetition 1.3 at
    // least 23% of the logit). Returns the largest amount it trails by beyond
    // that.
    constexpr float graphDrift = 0.03F;
    const auto probeLoss = [&](const std::vector<uint32_t> &transcript,
                               const ops::SamplingPenalties &penalties) {
      float worst = 0.0F;
      for (size_t position = 0; position < transcript.size(); ++position) {
        std::vector<uint32_t> context = prompt129;
        context.insert(context.end(), transcript.begin(),
                       transcript.begin() + position);
        std::vector<uint32_t> options = context;
        options.push_back(transcript[position]);
        std::sort(options.begin(), options.end());
        options.erase(std::unique(options.begin(), options.end()), options.end());
        require(options.size() >= model::ExecutionLimits::minimumScoreOptions &&
                    options.size() <= model::ExecutionLimits::maximumScoreOptions,
                "score probe options do not fit one score request");
        EngineRequest score = makeRequest(90, context, 0);
        score.scoreTokens = options;
        beginCold(executor, score, 0);
        const ModelStepResult scored =
            prefillChunk(executor, score.id, 0, context, pageTable);
        executor.end(score.id);
        require(scored.scoreLogits.size() == options.size(),
                "score probe returned the wrong logit count");
        float best = -std::numeric_limits<float>::infinity();
        float emitted = best;
        for (size_t index = 0; index < options.size(); ++index) {
          const uint32_t token = options[index];
          const auto count = static_cast<uint32_t>(
              std::count(transcript.begin(), transcript.begin() + position, token));
          float value = scored.scoreLogits[index];
          if (count || std::find(prompt129.begin(), prompt129.end(), token) !=
                           prompt129.end())
            value = value > 0.0F ? value / penalties.repetition
                                 : value * penalties.repetition;
          if (count)
            value -= penalties.frequency * float(count) + penalties.presence;
          best = std::max(best, value);
          if (token == transcript[position])
            emitted = value;
        }
        const float allowance =
            graphDrift * std::max(std::abs(best), std::abs(emitted));
        worst = std::max(worst, best - emitted - allowance);
      }
      return worst;
    };

    // Penalized requests keep every preemption guarantee above, and the
    // score probe checks their decisions within the drift between score
    // (prefill) and verify logits, and any further drift the unpenalized
    // transcript shows.
    // Negative presence and frequency favour the output's tokens by their
    // counts, so the decisions they change follow the counts, a verify row's
    // draft prefix included; presence 1.5 is Qwen's recommendation, and
    // repetition also reads the prompt's tokens.
    const PreemptionRun control = runPreemption(RequestKind::Greedy, 0);
    const float drift = probeLoss(control.transcript, {});
    const float tolerance = std::max(2.0F * drift, 0.1F);
    std::cout << "penalty_probe control_drift=" << drift
              << " tolerance=" << tolerance << '\n';
    bool penaltiesDecided = false;
    for (const ops::SamplingPenalties penalties :
         {ops::SamplingPenalties{1.0F, -2.0F, -2.0F},
          ops::SamplingPenalties{1.0F, 1.5F, 0.0F},
          ops::SamplingPenalties{1.3F, 1.5F, 0.0F}}) {
      const auto reference = runPreemption(RequestKind::Greedy, 0, penalties);
      const auto promptResumed =
          runPreemption(RequestKind::Greedy, 1, penalties);
      require(promptResumed.transcript == reference.transcript,
              "prompt recomputation changed a penalized transcript");
      const auto resumed = runPreemption(RequestKind::Greedy, 2, penalties);
      require(resumed.transcript.size() > resumed.pendingAnchorIndex &&
                  reference.transcript.size() > resumed.pendingAnchorIndex &&
                  std::equal(resumed.transcript.begin(),
                             resumed.transcript.begin() +
                                 resumed.pendingAnchorIndex + 1,
                             reference.transcript.begin()),
              "preemption changed penalized history or its pending anchor");
      const float referenceLoss = probeLoss(reference.transcript, penalties);
      const float resumedLoss = probeLoss(resumed.transcript, penalties);
      penaltiesDecided |= reference.transcript != control.transcript;
      std::cout << "penalty_probe presence=" << penalties.presence
                << " frequency=" << penalties.frequency
                << " repetition=" << penalties.repetition
                << " changed=" << (reference.transcript != control.transcript)
                << " reference_loss=" << referenceLoss
                << " resumed_loss=" << resumedLoss << '\n';
      require(referenceLoss <= tolerance && resumedLoss <= tolerance,
              "a penalized token is not the best of its penalized logits");
    }
    require(penaltiesDecided, "no penalty changed a decision to probe");
    // Sampled lanes with presence and frequency repeat under a fixed seed;
    // a constrained lane with repetition keeps its masked continuation.
    {
      const ops::SamplingPenalties sampled{1.0F, 1.5F, 0.5F};
      const auto reference = runPreemption(RequestKind::Sampling, 0, sampled);
      require(runPreemption(RequestKind::Sampling, 1, sampled).transcript ==
                  reference.transcript,
              "prompt recomputation changed a penalized sampled transcript");
      const auto resumed = runPreemption(RequestKind::Sampling, 2, sampled);
      require(std::equal(resumed.transcript.begin(),
                         resumed.transcript.begin() +
                             resumed.pendingAnchorIndex + 1,
                         reference.transcript.begin()) &&
                  runPreemption(RequestKind::Sampling, 2, sampled).transcript ==
                      resumed.transcript,
              "penalized sampled recomputation is not deterministic");
      const ops::SamplingPenalties repetition{1.3F, 0.0F, 0.0F};
      const auto masked =
          runPreemption(RequestKind::Constrained, 0, repetition);
      require(
          runPreemption(RequestKind::Constrained, 1, repetition).transcript ==
              masked.transcript,
          "prompt recomputation changed a penalized constrained transcript");
      const auto maskedResumed =
          runPreemption(RequestKind::Constrained, 2, repetition);
      require(std::equal(maskedResumed.transcript.begin(),
                         maskedResumed.transcript.begin() +
                             maskedResumed.pendingAnchorIndex + 1,
                         masked.transcript.begin()),
              "preemption changed penalized constrained history");
    }
    // Sampled requests whose top_k keeps every token keep the preemption
    // guarantees and repeat under a fixed seed, at top_p 0.95 and 1. They
    // ignore the stop tokens: a row that keeps every token can draw one, the
    // first row too (the 35B UD-Q4_K_M at top_p 1 does under this seed),
    // which would end the request at its prompt.
    for (const float topP : {0.95F, 1.0F}) {
      const auto run = [&](uint32_t preemptionMode) {
        return runPreemption(RequestKind::Sampling, preemptionMode, {}, 0, topP,
                             RequestIgnoreEndOfSequence);
      };
      const auto reference = run(0);
      require(run(1).transcript == reference.transcript,
              "prompt recomputation changed a top_k -1 transcript");
      const auto resumed = run(2);
      require(std::equal(resumed.transcript.begin(),
                         resumed.transcript.begin() +
                             resumed.pendingAnchorIndex + 1,
                         reference.transcript.begin()) &&
                  run(2).transcript == resumed.transcript,
              "top_k -1 recomputation is not deterministic");
      std::cout << "top_k -1 top_p=" << topP
                << " transcript_equal=" << (resumed.transcript == reference.transcript)
                << " rows=" << reference.transcript.size() << '\n';
    }
    // Sampled requests that min_p alone cuts (top_k and top_p keep every
    // token) keep the preemption guarantees and repeat under a fixed seed. At
    // min_p 1 each row keeps its most likely token only, so the request
    // selects what a greedy one does, draft tokens and corrections included.
    {
      const auto run = [&](uint32_t preemptionMode, float minP) {
        return runPreemption(RequestKind::Sampling, preemptionMode, {}, 0, 1.0F,
                             RequestIgnoreEndOfSequence, minP);
      };
      const auto reference = run(0, 0.1F);
      require(run(1, 0.1F).transcript == reference.transcript,
              "prompt recomputation changed a min_p transcript");
      const auto resumed = run(2, 0.1F);
      require(std::equal(resumed.transcript.begin(),
                         resumed.transcript.begin() +
                             resumed.pendingAnchorIndex + 1,
                         reference.transcript.begin()) &&
                  run(2, 0.1F).transcript == resumed.transcript,
              "min_p recomputation is not deterministic");
      const auto greedy = runPreemption(RequestKind::Greedy, 0, {}, 20, 0.95F,
                                        RequestIgnoreEndOfSequence);
      const auto heaviest = run(0, 1.0F);
      require(heaviest.transcript == greedy.transcript,
              "min_p 1 did not select each row's most likely token");
      std::cout << "min_p 0.1 transcript_equal="
                << (resumed.transcript == reference.transcript)
                << " rows=" << reference.transcript.size()
                << " min_p 1 greedy_rows=" << greedy.transcript.size() << '\n';
    }

    // Prefill telemetry counts the captures each dispatch makes: a cold
    // prompt with a checkpoint, prefilled in budget-sized chunks, adds what
    // its plan's spans and boundaries hold, a skipped gap and two windows
    // included.
    {
      std::vector<uint32_t> prompt(3800);
      for (uint32_t index = 0; index < prompt.size(); ++index)
        prompt[index] = samplingSeedTokens[index % samplingSeedTokens.size()];
      const std::array<uint32_t, 1> checkpoints{1024};
      const DraftContextPlan plan = planDraftContext(
          0, static_cast<uint32_t>(prompt.size()), checkpoints);
      uint64_t capturedRows = 0;
      uint64_t resets = 0;
      for (const DraftCaptureSpan &span : plan.captureSpans) {
        capturedRows += span.end - span.begin;
        if (span.resetDraftState)
          ++resets;
      }
      uint64_t activeRows = 0;
      uint64_t materializationRows = 0;
      for (const DraftBoundaryPlan &boundary : plan.boundaries) {
        const uint64_t rows = boundary.boundary - boundary.captureBegin;
        if (boundary.purpose == DraftBoundaryPurpose::Active)
          activeRows += rows;
        else
          materializationRows += rows;
      }
      require(resets == 2 && capturedRows < prompt.size(),
              "draft telemetry fixture neither resets twice nor skips rows");
      const model::ModelTelemetry before = executor.telemetry();
      beginCold(executor, makeRequest(91, prompt, 1), 0);
      executor.setDraftContextPlan(91, plan);
      const std::vector<uint32_t> pages = pageRange(0, 119);
      for (uint32_t begin = 0; begin < prompt.size();) {
        const uint32_t rows =
            std::min<uint32_t>(model::ExecutionLimits::prefillTokenBudget,
                               static_cast<uint32_t>(prompt.size()) - begin);
        prefillChunk(executor, 91, begin,
                     std::span<const uint32_t>(prompt).subspan(begin, rows),
                     pages);
        begin += rows;
      }
      executor.end(91);
      const model::ModelTelemetry after = executor.telemetry();
      require(after.targetPrefillRows - before.targetPrefillRows ==
                      prompt.size() &&
                  after.draftContextRowsActive -
                          before.draftContextRowsActive ==
                      activeRows &&
                  after.draftContextRowsMaterialization -
                          before.draftContextRowsMaterialization ==
                      materializationRows &&
                  after.draftContextRowsAvoided -
                          before.draftContextRowsAvoided ==
                      prompt.size() - capturedRows &&
                  after.draftStateResets - before.draftStateResets == resets,
              "prefill draft telemetry does not follow the plan");
      std::cout << "draft_telemetry_follows_plan=PASS\n";
    }

    // A plan whose first capture continues a draft ring the restore skipped
    // fails the prefill that would continue the empty ring, rather than
    // building a shorter window that decode then finds incomplete.
    {
      const std::vector<uint32_t> pages = pageRange(0, 8);
      beginCold(executor, makeRequest(92, prompt128, 1), 0);
      prefillChunk(executor, 92, 0, prompt128, pages);
      std::shared_ptr<const CompositeState> state = executor.snapshot(92);
      require(state != nullptr, "discontinuity fixture snapshot allocation failed");
      executor.end(92);
      std::vector<uint32_t> extended = prompt128;
      extended.insert(extended.end(), prompt128.begin(), prompt128.begin() + 100);
      beginCold(executor, makeRequest(93, extended, 1), 0);
      require(!executor.beginRestore(93, 128, state, false, {}),
              "resident restore returned a read");
      executor.setDraftContextPlan(
          93, planDraftContext(128, static_cast<uint32_t>(extended.size()), {}));
      BatchPlan plan{.kind = WorkKind::Prefill,
                     .items = {{93, 100}},
                     .decodeStage = DecodeStage::Regular};
      ModelBatchItem item = withRevision({.requestId = 93,
                                          .logicalPosition = 128,
                                          .tokenCount = 100,
                                          .pageTable = pages});
      item.inputTokens = std::span<const uint32_t>(extended).subspan(128, 100);
      auto ticket =
          executor.submit(plan, std::span<const ModelBatchItem>(&item, 1), {});
      rejects([&] { static_cast<void>(ticket->wait()); },
              "draft capture does not continue the draft ring",
              "a capture continued a draft ring its restore skipped");
      executor.end(93);
      std::cout << "discontinuous_capture_fails=PASS\n";
    }

    const auto rowsBeforeInvalidWarmup = executor.telemetry().targetPrefillRows;
    for (uint32_t rows : {0U, model::ExecutionLimits::prefillTokenBudget + 1,
                          std::numeric_limits<uint32_t>::max()}) {
      rejects([&] { static_cast<void>(executor.warmupPrefill(rows)); },
              "invalid prefill warmup row count", "invalid warmup rows were accepted");
      require(executor.telemetry().targetPrefillRows == rowsBeforeInvalidWarmup,
              "invalid warmup rows reached the production prefill phase");
    }
    for (uint32_t rows : {32U, 128U, 512U}) {
      const auto smallerWarmup = executor.warmupPrefill(rows);
      require(smallerWarmup.wallSeconds >= executor.telemetry().lastPrefillWallSeconds &&
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
    require(prefillWarmup.wallSeconds > 0.0,
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
    require(batch2.wallSeconds > 0.0,
            "real B2 decode warmup did not report timing");
    require(batch3.wallSeconds > 0.0,
            "real B3 decode warmup did not report timing");
    require(batch4.wallSeconds > 0.0,
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
    const model::ModelTelemetry b4Telemetry =
        executor.telemetry();
    require(batch4.wallSeconds >= b4Telemetry.lastDecodeWallSeconds,
            "decode warmup excluded production work from phase wall time");
    const auto repeatedBatch4 = executor.warmupDecodeBatch(4);
    require(repeatedBatch4.lanes == batch4.lanes,
            "repeated baseline B4 decode changed its deterministic result");
    std::cout << "prefill_2048_wall_seconds=" << prefillWarmup.wallSeconds
              << " b1_cycle_wall_seconds=" << batch1.wallSeconds
              << " b2_cycle_wall_seconds=" << batch2.wallSeconds
              << " b3_cycle_wall_seconds=" << batch3.wallSeconds
              << " b4_cycle_wall_seconds=" << batch4.wallSeconds << '\n';
    model::WarmupStepResult historical =
        executor.warmupCompositeStateRestore();
    require(historical.wallSeconds > 0.0,
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
