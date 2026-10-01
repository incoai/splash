#include "tuning/LinearTuning.hpp"

#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace splash::ops::tuning {
namespace {

using Clock = std::chrono::steady_clock;
// ReferenceGate/ReferenceUp hold the exact gate and up projections a split-K
// gate/up plan is held to; they exist only when a gate/up workload has a
// candidate that is not bitwise comparable with its baseline.
enum Field : size_t {
  Input, Output, Sums, Residual, GateScratch, DownSums,
  ReferenceOutput, ReferenceDownSums, ReferenceGate, ReferenceUp,
  PreparedInput, PreparedSums, Partials, Counters, FieldCount
};
struct Region final { uint64_t offset = 0, bytes = 0; };
struct Layout final {
  std::array<Region, FieldCount> fields{};
  uint64_t bytes = 0;
};

// Sequential plans store bitwise-identical outputs for a workload
// (LinearPlan::partialSums). Any other pair may round differently even with
// as many partial sums on both sides (Split128's four K splits and the
// one-lane Split32 tile's four partitions), so it is held to the derived bound.
bool sequential(const LinearPlan &plan) {
  return plan.partialSums() == 1 && !plan.usesSimdgroup();
}
bool bitwiseComparable(const LinearPlan &baseline, const LinearPlan &candidate) {
  return sequential(baseline) && sequential(candidate);
}

uint64_t align(uint64_t value, uint64_t alignment) {
  if (value > std::numeric_limits<uint64_t>::max() - (alignment - 1))
    throw std::overflow_error("Linear tuning fixture size overflow");
  return (value + alignment - 1) & ~(alignment - 1);
}

Layout layout(const DeviceCapabilities &device,
              std::span<const LinearPlan> plans) {
  if (plans.empty()) throw std::logic_error("Linear tuning has no baseline");
  const auto workload = plans.front().workload();
  const uint64_t rows = plans.front().storageRows();
  Layout result;
  result.fields[Input].bytes = rows * workload.matrix.inputSize * 2;
  result.fields[Output].bytes = rows * workload.matrix.outputSize * 2;
  result.fields[ReferenceOutput].bytes = result.fields[Output].bytes;
  if (workload.epilogue == LinearEpilogue::Residual)
    result.fields[Residual].bytes = result.fields[Output].bytes;
  for (const auto &plan : plans) {
    const auto scratch = plan.scratchSize();
    for (auto [field, bytes] : {std::pair{PreparedInput, scratch.input},
                               {PreparedSums, scratch.sums}, {Partials, scratch.partials},
                               {Counters, scratch.counters}})
      result.fields[field].bytes = std::max(result.fields[field].bytes, bytes);
    result.fields[Sums].bytes = std::max(result.fields[Sums].bytes, plan.sumsBytes());
    result.fields[GateScratch].bytes =
        std::max(result.fields[GateScratch].bytes, plan.gateScratchBytes());
    result.fields[DownSums].bytes =
        std::max(result.fields[DownSums].bytes, plan.downSumsBytes());
  }
  result.fields[ReferenceDownSums].bytes = result.fields[DownSums].bytes;
  const bool mixed = std::any_of(plans.begin() + 1, plans.end(), [&](const LinearPlan &plan) {
    return !bitwiseComparable(plans.front(), plan);
  });
  if (mixed && workload.epilogue == LinearEpilogue::GateUp)
    result.fields[ReferenceGate].bytes = result.fields[ReferenceUp].bytes =
        result.fields[Output].bytes;
  for (auto &field : result.fields) {
    field.offset = align(result.bytes, 256);
    if (field.bytes > std::numeric_limits<uint64_t>::max() - field.offset)
      throw std::overflow_error("Linear tuning fixture size overflow");
    result.bytes = field.offset + field.bytes;
  }
  // Metal's physical shared allocation is page-granular. Admit the rounded
  // backing once; its views and reference output do not allocate GPU storage.
  result.bytes = align(result.bytes, 16384);
  if (device.maxBufferLengthBytes && result.bytes > device.maxBufferLengthBytes)
    throw std::invalid_argument("Linear tuning fixture exceeds device buffer limit");
  return result;
}

uint32_t mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352d;
  value ^= value >> 15;
  value *= 0x846ca68b;
  return value ^ (value >> 16);
}
void initialize(metal::MetalBuffer buffer, uint64_t active, uint32_t seed) {
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i)
    values[i] = i < active
        ? floatToBf16(float(int(mix(uint32_t(i) + seed) % 257) - 128) / 257.0f) : 0;
}
void poisonBf16(metal::MetalBuffer buffer) {
  if (!buffer) return;
  auto *values = static_cast<uint16_t *>(buffer.contents());
  std::fill_n(values, buffer.sizeBytes() / 2, uint16_t{0x7fc1});
}
void poisonFloat(metal::MetalBuffer buffer) {
  if (!buffer) return;
  auto *values = static_cast<float *>(buffer.contents());
  std::fill_n(values, buffer.sizeBytes() / 4, std::numeric_limits<float>::quiet_NaN());
}
// Two plans that are not bitwise comparable hold every output element to the
// derived bound instead (LinearNumerics.hpp), with the sequential plan's
// output as the reference whichever side it is. Two split plans each
// reassociate the sum: `associations` counts the sides whose fp32 slack the
// bound covers.
void requireWithinSplitTolerance(LinearWorkload workload, const metal::MetalBuffer &exact,
                                 const metal::MetalBuffer &split,
                                 const metal::MetalBuffer &residual,
                                 const metal::MetalBuffer &gate,
                                 const metal::MetalBuffer &up, uint32_t associations,
                                 float operandSlack) {
  const auto values = [](const metal::MetalBuffer &buffer) {
    return buffer ? static_cast<const uint16_t *>(buffer.contents()) : nullptr;
  };
  const auto *exactValues = values(exact);
  const auto *splitValues = values(split);
  const auto *residualValues = values(residual);
  const auto *gateValues = values(gate);
  const auto *upValues = values(up);
  const uint64_t elements = uint64_t{workload.rows} * workload.matrix.outputSize;
  float maxAbs = 0;
  for (uint64_t i = 0; i < elements; ++i)
    maxAbs = std::max(maxAbs, std::fabs(bf16ToFloat(exactValues[i])));
  const float slack = float(associations) * reassociationSlack(workload.matrix.inputSize, maxAbs) + operandSlack;
  for (uint64_t i = 0; i < elements; ++i) {
    SplitReference reference{bf16ToFloat(exactValues[i])};
    if (residualValues) reference.residual = bf16ToFloat(residualValues[i]);
    if (gateValues) reference.gate = bf16ToFloat(gateValues[i]);
    if (upValues) reference.up = bf16ToFloat(upValues[i]);
    if (!withinSplitTolerance(bf16ToFloat(splitValues[i]), workload.epilogue, reference, slack))
      throw std::runtime_error("Linear tuning split-K candidate output exceeds its bf16 tolerance");
  }
}

void requireFinite(metal::MetalBuffer buffer, bool floats) {
  if (!buffer) return;
  if (floats) {
    const auto *values = static_cast<const float *>(buffer.contents());
    for (uint64_t i = 0; i < buffer.sizeBytes() / 4; ++i)
      if (!std::isfinite(values[i]))
        throw std::runtime_error("Linear tuning produced nonfinite output sums");
  } else {
    const auto *values = static_cast<const uint16_t *>(buffer.contents());
    for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i)
      if ((values[i] & 0x7f80) == 0x7f80)
        throw std::runtime_error("Linear tuning produced nonfinite BF16 output");
  }
}

} // namespace

uint64_t linearTuningFixtureBytes(const DeviceCapabilities &device,
                                  LinearWorkload workload) {
  return layout(device, Linear(device).candidates(workload)).bytes;
}

LinearTuningResult tuneLinear(metal::MetalBackend &backend,
                              const metal::AllocationAdmission &admit,
                              const LinearTuningInput &input,
                              const MeasurementOptions &options,
                              const MeasurementStop &underPressure,
                              const MeasurementStop &shouldStop) {
  const auto start = Clock::now();
  LinearTuningResult result;
  result.choice.workload = input.workload;
  auto elapsed = [&] {
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  try {
    Linear linear(backend.capabilities());
    const auto plans = linear.candidates(input.workload);
    result.choice.configuration = plans.front().configuration();
    if (!validMeasurementOptions(options) || !admit)
      throw std::invalid_argument("invalid Linear tuning measurement options or admission");
    if (input.weights.empty() || input.weights.size() > kMaximumLinearTuningRepresentatives)
      throw std::invalid_argument("Linear tuning requires 1..8 representative weight views");
    // Block-quantized plans are not tuned: their only candidate is the baseline.
    if (input.workload.weightLayout != WeightLayout::Affine64)
      throw std::invalid_argument("Linear tuning takes affine workloads");
    result.representativeCount = static_cast<uint32_t>(input.weights.size());
    for (const auto &weights : input.weights) {
      requireAffineProjection(weights.projection, input.workload.matrix);
      if ((input.workload.epilogue == LinearEpilogue::GateUp) != weights.gate.has_value())
        throw std::invalid_argument("Linear tuning gate projection is required only for GateUp");
      if (weights.gate) requireAffineProjection(*weights.gate, input.workload.matrix);
    }
    const auto fixture = layout(backend.capabilities(), plans);
    auto control = [&] {
      if (!backend.healthy()) throw metal::MetalBackendError(backend.unhealthyReason());
      return !(shouldStop && shouldStop()) &&
          !(underPressure && underPressure()) && elapsed() < options.maximumWallSeconds;
    };
    if (!control()) return result;

    metal::MetalBuffer backing;
    bool invoked = false;
    const auto admitted = admit(fixture.bytes, [&] {
      if (invoked) throw std::logic_error("Linear tuning admission invoked allocation twice");
      invoked = true;
      backing = backend.allocateBuffer(fixture.bytes, metal::BufferStorage::Shared,
                                       "linear-tuning-fixture");
    });
    // The production governor may invoke allocation, catch a normal Metal
    // capacity failure, and return false with no backing to retain.
    if ((admitted && (!invoked || !backing)) || (!admitted && backing))
      throw std::logic_error("Linear tuning admission violated allocation contract");
    if (!admitted || !control()) return result;
    std::array<metal::MetalBuffer, FieldCount> fields;
    for (size_t i = 0; i < fields.size(); ++i)
      if (fixture.fields[i].bytes)
        fields[i] = backend.view(backing, fixture.fields[i].offset, fixture.fields[i].bytes);
    LinearBuffers buffers{fields[Input], fields[Output], fields[Sums], fields[Residual],
                          fields[GateScratch], fields[DownSums],
                          {fields[PreparedInput], fields[PreparedSums], fields[Partials], fields[Counters]}};
    if (fields[Counters]) std::memset(fields[Counters].contents(), 0, fields[Counters].sizeBytes());
    const auto workload = input.workload;
    initialize(buffers.input, uint64_t{workload.rows} * workload.matrix.inputSize, 1949);
    if (buffers.residual)
      initialize(buffers.residual, uint64_t{workload.rows} * workload.matrix.outputSize, 7919);
    auto restore = [&] {
      poisonBf16(buffers.output);
      poisonFloat(buffers.sums);
      poisonFloat(buffers.downSums);
      if (workload.epilogue == LinearEpilogue::UpWithGate)
        initialize(buffers.gateScratch, uint64_t{workload.rows} * workload.matrix.outputSize, 104729);
      else poisonBf16(buffers.gateScratch);
    };
    auto run = [&](CandidateId candidate, uint32_t first, uint32_t repetitions) -> RunTiming {
      if (!backend.healthy()) throw metal::MetalBackendError(backend.unhealthyReason());
      if (underPressure && underPressure()) return {0, 0, true};
      restore();
      // Graph construction is included in wall time, but state restoration is
      // not. Every repetition includes sums and every epilogue phase. Inputs,
      // residuals and UpWithGate's gate are read-only; the production encoder
      // overwrites output/scratch before reuse (including GateUp's first phase).
      const auto wallStart = Clock::now();
      metal::CommandGraph graph;
      for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
        const auto &weights = input.weights[(first + repetition) % input.weights.size()];
        if (workload.phase == LinearPhase::Prefill)
          linear.addPrefillSums(graph, buffers.input, buffers.sums, weights.projection, workload.rows);
        linear.add(graph, buffers, weights.projection, plans.at(candidate.value),
                   weights.gate ? &*weights.gate : nullptr);
      }
      const auto timing = backend.submitCommand(graph.dispatches());
      const double wall = std::chrono::duration<double>(Clock::now() - wallStart).count();
      return {timing.gpuSeconds, wall, underPressure && underPressure()};
    };

    auto requireTiming = [](RunTiming timing) {
      if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds <= 0 ||
          !std::isfinite(timing.wallSeconds) || timing.wallSeconds <= 0)
        throw std::runtime_error("Linear tuning qualification returned invalid timing");
    };
    // Gate/up mixes of split-K and sequential plans are held to the exact
    // gate and up projections of the representative being qualified: one
    // untimed submission per representative, before its candidates run.
    const std::optional<LinearPlan> exactPlain = fields[ReferenceGate]
        ? std::optional{Linear::plan(
              {workload.matrix, workload.rows, LinearPhase::Decode, LinearEpilogue::None},
              {LinearTile::N128, workload.matrix.outputSize / 128})}
        : std::nullopt;
    float operandSlack = 0;
    auto referenceGateUp = [&](uint32_t representative) {
      if (fields[PreparedInput]) {
        operandSlack = simdgroupSlack(workload, buffers.input, input.weights[representative].projection);
        if (input.weights[representative].gate)
          operandSlack = std::max(operandSlack, simdgroupSlack(workload, buffers.input, *input.weights[representative].gate));
      }
      if (!exactPlain) return;
      const auto &weights = input.weights[representative];
      metal::CommandGraph graph;
      linear.add(graph, {buffers.input, fields[ReferenceGate], {}, {}, {}, {}},
                 *weights.gate, *exactPlain);
      linear.add(graph, {buffers.input, fields[ReferenceUp], {}, {}, {}, {}},
                 weights.projection, *exactPlain);
      (void)backend.submitCommand(graph.dispatches());
    };
    auto qualify = [&](size_t candidate, bool baseline) {
      requireFinite(buffers.output, false);
      requireFinite(buffers.downSums, true);
      const bool bitwise = bitwiseComparable(plans[0], plans[candidate]);
      for (const auto pair : {std::pair{Output, ReferenceOutput},
                              std::pair{DownSums, ReferenceDownSums}}) {
        const auto &actual = fields[pair.first];
        const auto &reference = fields[pair.second];
        if (!actual) continue;
        if (baseline) std::memcpy(reference.contents(), actual.contents(), actual.sizeBytes());
        else if (!bitwise && pair.first == Output) {
          const bool baselineExact = sequential(plans[0]);
          requireWithinSplitTolerance(workload, baselineExact ? reference : actual,
                                      baselineExact ? actual : reference, buffers.residual,
                                      fields[ReferenceGate], fields[ReferenceUp],
                                      baselineExact || sequential(plans[candidate]) ? 1 : 2, operandSlack);
        } else if (std::memcmp(reference.contents(), actual.contents(), actual.sizeBytes()))
          throw std::runtime_error("Linear tuning candidate output differs from baseline");
      }
    };
    // Compare every representative against its own baseline, not another
    // layer's output. Reuse these existing baseline timings as the pilot.
    // This qualifies kernels; it does not prove the warmed cache distribution
    // matches a full model graph, which remains a caller-owned acceptance gate.
    double baselineGpuSeconds = 0;
    for (uint32_t representative = 0; representative < result.representativeCount; ++representative) {
      if (!control()) return result;
      referenceGateUp(representative);
      for (size_t i = 0; i < plans.size(); ++i) {
        if (!control()) return result;
        const auto timing = run(CandidateId{uint32_t(i)}, representative, 1);
        if (timing.underPressure || !control()) return result;
        requireTiming(timing);
        qualify(i, i == 0);
        if (!i) baselineGpuSeconds += timing.gpuSeconds;
      }
    }
    const uint32_t requested = measurementBatchRepetitions(
        baselineGpuSeconds / result.representativeCount);
    constexpr uint32_t maximumRepetitions = 16;
    const uint32_t rings = std::min(maximumRepetitions / result.representativeCount,
        (requested + result.representativeCount - 1) / result.representativeCount);
    result.repetitions = rings * result.representativeCount;

    // A complete ring ends at the last representative, whose single-op
    // baseline is still in the reference views. Qualify scratch reuse in the
    // actual repeated graph, for every candidate, before any paired samples.
    for (size_t i = 0; i < plans.size(); ++i) {
      if (!control()) return result;
      const auto timing = run(CandidateId{uint32_t(i)}, 0, result.repetitions);
      if (timing.underPressure || !control()) return result;
      requireTiming(timing);
      qualify(i, false);
    }

    result.measurements.reserve(plans.size() - 1);
    constexpr WorkloadId workloadId{0};
    for (size_t i = 1; i < plans.size(); ++i) {
      if (!control()) return result;
      auto remaining = options;
      remaining.maximumWallSeconds = options.maximumWallSeconds - elapsed();
      result.measurements.push_back(measureWorkload(
          CandidateId{uint32_t(i)}, workloadId,
          [&](CandidateId candidate) { return run(candidate, 0, result.repetitions); },
          remaining, shouldStop));
      const auto &measurement = result.measurements.back();
      if (measurement.status != MeasurementStatus::Completed &&
          measurement.status != MeasurementStatus::Rejected) {
        result.failure = measurement.failure;
        return result;
      }
    }
    if (!control()) return result;
    std::vector<WorkloadMeasurements> gpuWorkloads, wallWorkloads;
    std::vector<CandidateMeasurements> gpuCandidates, wallCandidates;
    gpuWorkloads.reserve(result.measurements.size());
    wallWorkloads.reserve(result.measurements.size());
    gpuCandidates.reserve(result.measurements.size());
    wallCandidates.reserve(result.measurements.size());
    for (const auto &measurement : result.measurements) {
      // Every record here finished the full sample count. Evaluate the metrics
      // independently, including a candidate rejected by the other metric;
      // otherwise filtering could disguise disagreement between their winners.
      gpuWorkloads.push_back({workloadId, measurement.rawGpuSamples()});
      wallWorkloads.push_back({workloadId, measurement.rawWallSamples()});
      gpuCandidates.push_back({measurement.candidate, {&gpuWorkloads.back(), 1}});
      wallCandidates.push_back({measurement.candidate, {&wallWorkloads.back(), 1}});
    }
    const auto gpu = selectCandidate(gpuCandidates, {&workloadId, 1}, options.policy);
    const auto wall = selectCandidate(wallCandidates, {&workloadId, 1}, options.policy);
    if (gpu.verdict == SelectionVerdict::Selected &&
        wall.verdict == SelectionVerdict::Selected && gpu.candidate == wall.candidate)
      result.choice.configuration = plans.at(gpu.candidate.value).configuration();
    result.complete = true;
  } catch (...) {
    result.failure = std::current_exception();
  }
  return result;
}

} // namespace splash::ops::tuning
