#include "tuning/DraftAttentionTuning.hpp"

#include "metal/abi/DraftAttention.h"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace splash::ops::tuning {
namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kWindow = SPLASH_DRAFT_SLIDING_WINDOW;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint64_t kAlignment = 16 * 1024;
constexpr std::array required{WorkloadId{0}, WorkloadId{1}};
constexpr std::array<std::array<uint32_t, kLanes>, 2> histories{{
    {0, 31, 128, 511}, {2100, 4094, 6143, 8193}}};

enum class Tensor : size_t {
  Input0, Input1, Dynamic0, Dynamic1, Weights0, Weights1, Residual0,
  Residual1, Convolution0, Convolution1, Convolution2, Convolution3,
  Qkv, QkvOriginal, Grouped, QueryKeys, QueryValues, Packed, QueryNorm,
  KeyNorm, RopeCos, RopeSin, RingKeys, RingValues, Reference, Count
};
constexpr size_t index(Tensor tensor) { return static_cast<size_t>(tensor); }
constexpr std::array outputs{
    Tensor::Convolution0, Tensor::Convolution1, Tensor::Convolution2,
    Tensor::Convolution3, Tensor::Qkv, Tensor::Grouped, Tensor::QueryKeys,
    Tensor::QueryValues, Tensor::Packed};

uint64_t aligned(uint64_t bytes) {
  if (bytes > std::numeric_limits<uint64_t>::max() - kAlignment + 1)
    throw std::invalid_argument("draft tuning fixture size overflow");
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}

struct FixturePlan final {
  DraftAttentionWorkload workload;
  std::array<uint64_t, index(Tensor::Count)> sizes{};
  uint64_t queryRowBytes = 0;
  uint64_t ringBytes = 0;
  uint64_t referenceBytes = 0;
  uint64_t bytes = 0;

  // Bytes of an output tensor that the candidates must reproduce.
  uint64_t qualifiedBytes(Tensor tensor) const {
    return tensor == Tensor::Grouped ? queryRowBytes : sizes[index(tensor)];
  }

  explicit FixturePlan(DraftAttentionWorkload w) : workload(w) {
    const auto workspace = DraftAttention::plan(w.shape, w.lanes).workspace();
    const uint64_t rows = uint64_t{w.lanes} * kRows;
    for (Tensor tensor : {Tensor::Input0, Tensor::Input1, Tensor::Residual0,
                           Tensor::Residual1, Tensor::Convolution0,
                           Tensor::Convolution1, Tensor::Convolution2,
                           Tensor::Convolution3})
      sizes[index(tensor)] = workspace.convolutionBytes;
    sizes[index(Tensor::Dynamic0)] = sizes[index(Tensor::Dynamic1)] =
        rows * w.shape.dynamicSize * 2;
    sizes[index(Tensor::Weights0)] = sizes[index(Tensor::Weights1)] =
        uint64_t{4} * w.shape.hiddenSize * 2;
    sizes[index(Tensor::Qkv)] = sizes[index(Tensor::QkvOriginal)] = workspace.qkvBytes;
    // The grouped tensor also holds the attention core's fp32 split partials
    // behind the query rows; only the rows are outputs.
    sizes[index(Tensor::Grouped)] = workspace.groupedQueriesBytes;
    sizes[index(Tensor::Packed)] = queryRowBytes =
        rows * w.shape.attentionSize * 2;
    sizes[index(Tensor::QueryKeys)] = workspace.queryKeysBytes;
    sizes[index(Tensor::QueryValues)] = workspace.queryValuesBytes;
    sizes[index(Tensor::QueryNorm)] = sizes[index(Tensor::KeyNorm)] =
        uint64_t{w.shape.headDimension} * 2;
    sizes[index(Tensor::RopeCos)] = sizes[index(Tensor::RopeSin)] =
        rows * (w.shape.headDimension / 2) * sizeof(float);
    ringBytes = uint64_t{w.shape.kvHeads} * kWindow * w.shape.headDimension * 2;
    sizes[index(Tensor::RingKeys)] = sizes[index(Tensor::RingValues)] = ringBytes * w.lanes;
    for (auto tensor : outputs) referenceBytes += qualifiedBytes(tensor);
    sizes[index(Tensor::Reference)] = referenceBytes * required.size();
    for (uint64_t size : sizes) {
      const uint64_t allocation = aligned(size);
      if (allocation > std::numeric_limits<uint64_t>::max() - bytes)
        throw std::invalid_argument("draft tuning fixture size overflow");
      bytes += allocation;
    }
  }
};

struct NumericalMismatch final : std::runtime_error {
  NumericalMismatch() : std::runtime_error("draft tuning output is nonfinite or differs from baseline") {}
};
struct Interrupted final { MeasurementStatus status; };

class Fixture final {
public:
  Fixture(metal::MetalBackend &backend, FixturePlan plan)
      : plan_(std::move(plan)), base_(backend.allocateBuffer(
            plan_.bytes, metal::BufferStorage::Shared, "draft-attention-tuning-fixture")) {
    uint64_t offset = 0;
    for (size_t i = 0; i < plan_.sizes.size(); ++i) {
      buffers_[i] = backend.view(base_, offset, plan_.sizes[i]);
      offset += aligned(plan_.sizes[i]);
    }
    const uint64_t pageBytes = plan_.ringBytes / SPLASH_DRAFT_PAGE_COUNT;
    for (uint32_t lane = 0; lane < plan_.workload.lanes; ++lane) {
      for (uint32_t page = 0; page < SPLASH_DRAFT_PAGE_COUNT; ++page) {
        const uint64_t offset = lane * plan_.ringBytes + page * pageBytes;
        context_[lane].keyPages.push_back(
            backend.view(get(Tensor::RingKeys), offset, pageBytes));
        context_[lane].valuePages.push_back(
            backend.view(get(Tensor::RingValues), offset, pageBytes));
      }
    }
    for (uint32_t lane = plan_.workload.lanes; lane < kLanes; ++lane)
      context_[lane] = context_[0];
  }

  bool initialize(const MeasurementStop &stop) {
    std::memset(base_.contents(), 0, plan_.bytes);
    for (Tensor tensor : {Tensor::Input0, Tensor::Input1, Tensor::Dynamic0,
                           Tensor::Dynamic1, Tensor::Weights0, Tensor::Weights1,
                           Tensor::Residual0, Tensor::Residual1, Tensor::QkvOriginal,
                           Tensor::RingKeys, Tensor::RingValues}) {
      const auto buffer = get(tensor);
      auto *data = static_cast<uint16_t *>(buffer.contents());
      const uint64_t elements = buffer.sizeBytes() / 2;
      uint64_t state = 0x5eed1234ULL + index(tensor) * 137 + plan_.workload.shape.hiddenSize;
      for (uint64_t element = 0; element < elements; ++element) {
        if (element % 65536 == 0 && stop && stop()) return false;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const float value = float((state >> 40) & 0xffffff) / 8388608.0f - 1;
        data[element] = floatToBf16(value * 0.5f);
      }
    }
    for (Tensor tensor : {Tensor::QueryNorm, Tensor::KeyNorm})
      std::fill_n(static_cast<uint16_t *>(get(tensor).contents()),
                   get(tensor).sizeBytes() / 2, floatToBf16(1));
    auto *cosine = static_cast<float *>(get(Tensor::RopeCos).contents());
    auto *sine = static_cast<float *>(get(Tensor::RopeSin).contents());
    for (uint64_t element = 0; element < get(Tensor::RopeCos).sizeBytes() / 4; ++element) {
      cosine[element] = std::cos(float(element) * 0.01f);
      sine[element] = std::sin(float(element) * 0.01f);
    }
    return true;
  }

  void reset() {
    for (Tensor tensor : outputs)
      std::memset(get(tensor).contents(), 0, get(tensor).sizeBytes());
    // QKV preparation normalizes its query portion in place. Every trial gets
    // the original projection output, never a second normalization of it.
    std::memcpy(get(Tensor::Qkv).contents(), get(Tensor::QkvOriginal).contents(),
                 get(Tensor::Qkv).sizeBytes());
  }

  metal::CommandGraph graph(DraftAttentionConfiguration configuration,
                             uint32_t history) const {
    const auto plan = DraftAttention::plan(plan_.workload.shape, plan_.workload.lanes,
                                           configuration);
    metal::CommandGraph graph;
    const auto convolution = [&](Tensor input, Tensor dynamic, Tensor weights,
                                  Tensor residual, Tensor output,
                                  DraftConvolutionStage stage) {
      DraftAttention::addConvolution(graph,
          {get(input), get(dynamic), get(weights), get(residual), get(output)}, plan, stage);
    };
    convolution(Tensor::Input0, Tensor::Dynamic0, Tensor::Weights0, Tensor::Residual0,
                  Tensor::Convolution0, DraftConvolutionStage::Prepare);
    DraftAttention::addPrepare(graph,
        {get(Tensor::Qkv), get(Tensor::Grouped), get(Tensor::QueryNorm),
         get(Tensor::KeyNorm), get(Tensor::RopeCos), get(Tensor::RopeSin),
         get(Tensor::QueryKeys), get(Tensor::QueryValues)}, plan);
    DraftAttention::addDecode(graph,
                              {get(Tensor::Grouped), context_,
                               get(Tensor::QueryKeys),
                               get(Tensor::QueryValues)},
                              histories[history], kWindow, plan);
    DraftAttention::addReorder(graph, get(Tensor::Grouped), get(Tensor::Packed), plan);
    convolution(Tensor::Input1, Tensor::Dynamic0, Tensor::Weights0, Tensor::Residual0,
                  Tensor::Convolution1, DraftConvolutionStage::Residual);
    convolution(Tensor::Input0, Tensor::Dynamic1, Tensor::Weights1, Tensor::Residual1,
                  Tensor::Convolution2, DraftConvolutionStage::Prepare);
    convolution(Tensor::Input1, Tensor::Dynamic1, Tensor::Weights1, Tensor::Residual1,
                  Tensor::Convolution3, DraftConvolutionStage::Residual);
    return graph;
  }

  void qualify(uint32_t history, bool baseline) {
    auto *reference = static_cast<uint8_t *>(get(Tensor::Reference).contents()) +
                        history * plan_.referenceBytes;
    for (Tensor tensor : outputs) {
      const uint64_t bytes = plan_.qualifiedBytes(tensor);
      const auto *values =
          static_cast<const uint16_t *>(get(tensor).contents());
      for (uint64_t i = 0; i < bytes / 2; ++i)
        if (!std::isfinite(bf16ToFloat(values[i]))) throw NumericalMismatch();
      if (haveReference_[history]) {
        if (std::memcmp(reference, values, bytes) != 0)
          throw NumericalMismatch();
      } else {
        if (!baseline) throw std::logic_error("draft baseline was not measured first");
        std::memcpy(reference, values, bytes);
      }
      reference += bytes;
    }
    haveReference_[history] = true;
  }

private:
  metal::MetalBuffer get(Tensor tensor) const { return buffers_[index(tensor)]; }
  FixturePlan plan_;
  metal::MetalBuffer base_;
  std::array<metal::MetalBuffer, index(Tensor::Count)> buffers_{};
  std::array<DraftKvBuffers, kLanes> context_{};
  std::array<bool, required.size()> haveReference_{};
};
} // namespace

uint64_t draftAttentionTuningFixtureBytes(DraftAttentionWorkload workload) {
  return FixturePlan(workload).bytes;
}

DraftAttentionTuningResult tuneDraftAttention(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    DraftAttentionWorkload workload, const MeasurementOptions &options,
    const MeasurementStop &underPressure, const MeasurementStop &shouldStop) {
  DraftAttentionTuningResult result{{workload, {}}, {}, false, {}};
  const auto started = Clock::now();
  const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - started).count(); };
  const auto interrupted = [&] {
    return (shouldStop && shouldStop()) || (underPressure && underPressure()) ||
           elapsed() >= options.maximumWallSeconds;
  };
  try {
    const FixturePlan plan(workload);
    if (!admit || !validMeasurementOptions(options) || interrupted() ||
        plan.bytes > backend.capabilities().maxBufferLengthBytes)
      return result;
    std::unique_ptr<Fixture> fixture;
    bool invoked = false;
    const auto admitted = admit(plan.bytes, [&] {
      if (invoked) throw std::logic_error("draft fixture admission invoked twice");
      invoked = true;
      fixture = std::make_unique<Fixture>(backend, plan);
    });
    if (!admitted) {
      if (fixture) throw std::logic_error("draft admission denied after retaining allocation");
      return result;
    }
    if (!invoked || !fixture)
      throw std::logic_error("draft admission succeeded without allocation");
    if (interrupted() || !fixture->initialize(interrupted))
      return result;
    const auto candidates = DraftAttention::candidates(workload.shape);
    for (size_t candidate = 1; candidate < candidates.size(); ++candidate) {
      for (uint32_t history = 0; history < required.size(); ++history) {
        if (interrupted()) return result;
        auto remaining = options;
        remaining.maximumWallSeconds = options.maximumWallSeconds - elapsed();
        std::exception_ptr numericalFailure;
        auto measured = measureWorkload(
            {static_cast<uint32_t>(candidate)}, required[history], [&](CandidateId selected) {
              if (underPressure && underPressure()) return RunTiming{0, 0, true};
              fixture->reset();
              if (underPressure && underPressure()) return RunTiming{0, 0, true};
              if (shouldStop && shouldStop()) throw Interrupted{MeasurementStatus::Cancelled};
              if (elapsed() >= options.maximumWallSeconds)
                throw Interrupted{MeasurementStatus::BudgetExceeded};
              const auto wallStart = Clock::now();
              const auto graph = fixture->graph(candidates[selected.value], history);
              const auto timing = backend.submitCommand(graph.dispatches());
              const double wall = std::chrono::duration<double>(Clock::now() - wallStart).count();
              try { fixture->qualify(history, selected == kBaseline); }
              catch (const NumericalMismatch &) { numericalFailure = std::current_exception(); }
              return RunTiming{timing.gpuSeconds, wall, underPressure && underPressure()};
            }, remaining, [&] { return numericalFailure || (shouldStop && shouldStop()); });
        if (numericalFailure) {
          measured.status = MeasurementStatus::RunFailed;
          measured.failure = numericalFailure;
        }
        if (measured.status == MeasurementStatus::RunFailed && measured.failure) {
          try { std::rethrow_exception(measured.failure); }
          catch (const Interrupted &interruption) {
            measured.status = interruption.status;
            measured.failure = {};
          }
          catch (...) {}
        }
        const auto status = measured.status;
        result.measurements.push_back(std::move(measured));
        if (status != MeasurementStatus::Completed && status != MeasurementStatus::Rejected) {
          result.failure = result.measurements.back().failure;
          return result;
        }
      }
    }
    if (interrupted()) return result;
    std::vector<WorkloadMeasurements> gpu, wall;
    gpu.reserve(result.measurements.size());
    wall.reserve(result.measurements.size());
    for (const auto &measured : result.measurements) {
      gpu.push_back({measured.workload, measured.rawGpuSamples()});
      wall.push_back({measured.workload, measured.rawWallSamples()});
    }
    std::vector<CandidateMeasurements> gpuCandidates, wallCandidates;
    for (size_t candidate = 1; candidate < candidates.size(); ++candidate) {
      const size_t offset = (candidate - 1) * required.size();
      gpuCandidates.push_back({{static_cast<uint32_t>(candidate)}, {gpu.data() + offset, required.size()}});
      wallCandidates.push_back({{static_cast<uint32_t>(candidate)}, {wall.data() + offset, required.size()}});
    }
    const auto gpuWinner = selectCandidate(gpuCandidates, required, options.policy);
    const auto wallWinner = selectCandidate(wallCandidates, required, options.policy);
    if (gpuWinner.verdict == SelectionVerdict::Selected &&
        wallWinner.verdict == SelectionVerdict::Selected && gpuWinner.candidate == wallWinner.candidate)
      result.choice.configuration = candidates[gpuWinner.candidate.value];
    result.complete = true;
  } catch (...) {
    result.failure = std::current_exception();
  }
  return result;
}
} // namespace splash::ops::tuning
