#include "tuning/MoeTuning.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace splash::ops::tuning {
namespace {

constexpr CandidateId kAlternative{1};
constexpr std::array<WorkloadId, 2> kDistributions{{{0}, {1}}};
constexpr std::array kScratchFields{
    &MoeBuffers::selectedExperts, &MoeBuffers::routingWeights,
    &MoeBuffers::tileDescriptors, &MoeBuffers::tileCount,
    &MoeBuffers::groupedRoutes, &MoeBuffers::routeRows,
    &MoeBuffers::groupedInput, &MoeBuffers::expertIntermediate,
    &MoeBuffers::expertOutput, &MoeBuffers::groupedSums};
constexpr std::array kWorkspaceFields{
    &MoeWorkspace::selectedExpertsBytes, &MoeWorkspace::routingWeightsBytes,
    &MoeWorkspace::tileDescriptorsBytes, &MoeWorkspace::tileCountBytes,
    &MoeWorkspace::groupedRoutesBytes, &MoeWorkspace::routeRowsBytes,
    &MoeWorkspace::groupedInputBytes, &MoeWorkspace::expertIntermediateBytes,
    &MoeWorkspace::expertOutputBytes, &MoeWorkspace::groupedSumsBytes};

uint64_t aligned(uint64_t value, uint64_t alignment) {
  if (value > std::numeric_limits<uint64_t>::max() - alignment + 1)
    throw std::overflow_error("MoE tuning fixture is too large");
  return (value + alignment - 1) / alignment * alignment;
}

struct Fixture final {
  metal::MetalBuffer arena;
  MoeBuffers buffers;
  std::array<metal::MetalBuffer, kDistributions.size()> references;
};

struct FixtureLayout final {
  std::array<uint64_t, 4 + kWorkspaceFields.size()> sizes;
  uint64_t bytes = 0;
};

FixtureLayout fixtureLayout(const std::array<MoePlan, 2> &plans) {
  const uint64_t rowBytes =
      uint64_t{plans[0].rows()} * plans[0].shape().hiddenSize * sizeof(uint16_t);
  // Input, residual, output and all baseline output mirrors are admitted
  // in the same backing. Scratch fields use their own maximum, not the total
  // of whichever candidate happens to have the larger workspace.
  FixtureLayout layout{{rowBytes, rowBytes, rowBytes,
                         rowBytes * kDistributions.size()}};
  for (size_t field = 0; field < kWorkspaceFields.size(); ++field)
    layout.sizes[field + 4] =
        std::max(plans[0].workspace().*kWorkspaceFields[field],
                 plans[1].workspace().*kWorkspaceFields[field]);
  for (const uint64_t size : layout.sizes) {
    const uint64_t padded = aligned(size, 256);
    if (layout.bytes > std::numeric_limits<uint64_t>::max() - padded)
      throw std::overflow_error("MoE tuning fixture is too large");
    layout.bytes += padded;
  }
  layout.bytes = aligned(layout.bytes, 16 * 1024);
  return layout;
}

Fixture allocateFixture(metal::MetalBackend &backend,
                        const metal::AllocationAdmission &admit,
                        const std::array<MoePlan, 2> &plans) {
  const auto layout = fixtureLayout(plans);
  Fixture fixture;
  if (!admit || layout.bytes > backend.capabilities().maxBufferLengthBytes)
    return fixture;
  bool invoked = false;
  if (!admit(layout.bytes, [&] {
        if (invoked) throw std::logic_error("MoE tuning admission invoked twice");
        invoked = true;
        fixture.arena = backend.allocateBuffer(
            layout.bytes, metal::BufferStorage::Shared, "moe-tuning");
      })) {
    if (fixture.arena)
      throw std::logic_error("MoE tuning admission retained a denied fixture");
    return fixture;
  }
  if (!fixture.arena)
    throw std::logic_error("MoE tuning admission did not allocate its fixture");
  uint64_t offset = 0;
  // A field no candidate uses (the grouped sums of non-register plans) stays
  // empty, as in the runtime arenas.
  auto view = [&](size_t index) {
    if (!layout.sizes[index]) return metal::MetalBuffer{};
    auto result = backend.view(fixture.arena, offset, layout.sizes[index]);
    offset += aligned(layout.sizes[index], 256);
    return result;
  };
  fixture.buffers.input = view(0);
  fixture.buffers.residual = view(1);
  fixture.buffers.output = view(2);
  const auto references = view(3);
  for (size_t i = 0; i < fixture.references.size(); ++i)
    fixture.references[i] = backend.view(references, i * layout.sizes[0], layout.sizes[0]);
  for (size_t field = 0; field < kScratchFields.size(); ++field)
    fixture.buffers.*kScratchFields[field] = view(field + 4);
  return fixture;
}

uint16_t bf16(float value) noexcept {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

// Approximately unit-variance normalized inputs, reproducible without libc's
// random state. Distinct rows leave routing to the learned weights; repeated
// rows exercise a concentrated route pattern without reading routes back.
void reset(Fixture &fixture, const MoeWorkload &workload,
           WorkloadId distribution) {
  auto *input = static_cast<uint16_t *>(fixture.buffers.input.contents());
  auto *residual = static_cast<uint16_t *>(fixture.buffers.residual.contents());
  for (uint32_t row = 0; row < workload.rows; ++row) {
    uint32_t state = 0x9e3779b9U ^
        (distribution.value == 0 ? (row + 1) * 0x85ebca6bU : 0x85ebca6bU);
    auto random = [&] {
      state = state * 1664525U + 1013904223U;
      return (static_cast<float>(state >> 8) / 8388608.0F - 1.0F);
    };
    for (uint32_t column = 0; column < workload.shape.hiddenSize; ++column) {
      const uint64_t index = uint64_t{row} * workload.shape.hiddenSize + column;
      input[index] = bf16(1.7320508F * random());
      residual[index] = bf16(0.125F * random());
    }
  }
  auto *output = static_cast<uint16_t *>(fixture.buffers.output.contents());
  std::fill_n(output, fixture.buffers.output.sizeBytes() / sizeof(uint16_t),
              uint16_t{0x7fc0});
  for (auto field : kScratchFields) {
    const auto &buffer = fixture.buffers.*field;
    std::memset(buffer.contents(), 0, static_cast<size_t>(buffer.sizeBytes()));
  }
}

bool finiteOutput(const metal::MetalBuffer &output) noexcept {
  const auto *values = static_cast<const uint16_t *>(output.contents());
  for (uint64_t index = 0; index < output.sizeBytes() / sizeof(uint16_t); ++index)
    if ((values[index] & 0x7f80) == 0x7f80)
      return false;
  return true;
}

struct Interrupted final { MeasurementStatus status; };

} // namespace

uint64_t moeTuningFixtureBytes(const MoeWorkload &workload) {
  // Router tile selection changes dispatch only; candidate scratch bounds are
  // device-independent and can be reserved before a backend is available.
  return fixtureLayout(ExecutionPlans({}).moeCandidates(workload)).bytes;
}

MoeTuningResult tuneMoe(metal::MetalBackend &backend,
                       const metal::AllocationAdmission &admitAllocation,
                       const MoeTuningInput &input,
                       const MeasurementOptions &options,
                       const MeasurementStop &underPressure,
                       const MeasurementStop &shouldStop) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const auto plans = ExecutionPlans(backend.capabilities()).moeCandidates(input.workload);
  MoeTuningResult result{{input.workload, plans[0].config()}, {}, false, {}};
  WorkloadId current = kDistributions[0];
  auto recordFailure = [&](MeasurementStatus status,
                           std::exception_ptr failure = {}) {
    MeasurementResult measurement;
    measurement.candidate = kAlternative;
    measurement.workload = current;
    measurement.status = status;
    measurement.failure = failure;
    result.measurements.push_back(std::move(measurement));
    result.failure = failure;
  };
  if (!validMeasurementOptions(options)) {
    recordFailure(MeasurementStatus::InvalidInput);
    return result;
  }
  try {
    if (input.weights.empty() || input.weights.size() > kMaximumMoeTuningRepresentatives)
      throw std::invalid_argument("MoE tuning requires 1..8 representatives");
    result.representativeCount = static_cast<uint32_t>(input.weights.size());
    auto remaining = [&] {
      return options.maximumWallSeconds -
          std::chrono::duration<double>(Clock::now() - start).count();
    };
    auto ready = [&] {
      if (underPressure && underPressure()) {
        recordFailure(MeasurementStatus::UnderPressure);
        return false;
      }
      if (shouldStop && shouldStop()) {
        recordFailure(MeasurementStatus::Cancelled);
        return false;
      }
      if (remaining() <= 0) {
        recordFailure(MeasurementStatus::BudgetExceeded);
        return false;
      }
      return true;
    };
    if (!ready())
      return result;
    Fixture fixture;
    try {
      fixture = allocateFixture(backend, admitAllocation, plans);
    } catch (const metal::MetalAllocationError &) {
      return result;
    } catch (const std::bad_alloc &) {
      return result;
    }
    if (!fixture.arena)
      return result;
    // Validate EVERY real view with the production encoder before submitting
    // any probe. A malformed later representative cannot be silently omitted.
    for (const auto &weights : input.weights)
      for (const auto &plan : plans) {
        metal::CommandGraph graph;
        MoE::add(graph, fixture.buffers, weights, plan);
      }
    const auto run = [&](CandidateId candidate, WorkloadId distribution,
                          uint32_t repetitions, uint32_t firstRepresentative) -> RunTiming {
      if (underPressure && underPressure()) return {0, 0, true};
      reset(fixture, input.workload, distribution);
      if (underPressure && underPressure()) return {0, 0, true};
      if (shouldStop && shouldStop()) throw Interrupted{MeasurementStatus::Cancelled};
      if (remaining() <= 0) throw Interrupted{MeasurementStatus::BudgetExceeded};
      const auto begin = Clock::now();
      metal::CommandGraph graph;
      for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
        const uint32_t representative =
            (firstRepresentative + repetition) % result.representativeCount;
        // Every complete operator overwrites route/group/count/output state.
        // Interleaving learned routers is qualified below before this same
        // shared-scratch sequence is used in the timed commands.
        MoE::add(graph, fixture.buffers, input.weights[representative], plans[candidate.value]);
      }
      const auto timing = backend.submitCommand(graph.dispatches());
      const double wall = std::chrono::duration<double>(Clock::now() - begin).count();
      return {timing.gpuSeconds, wall, underPressure && underPressure()};
    };
    const auto probe = [&](CandidateId candidate, WorkloadId distribution,
                            uint32_t repetitions, uint32_t firstRepresentative,
                            RunTiming &timing) {
      current = distribution;
      if (!ready()) return false;
      timing = run(candidate, distribution, repetitions, firstRepresentative);
      if (timing.underPressure) {
        recordFailure(MeasurementStatus::UnderPressure);
        return false;
      }
      if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds <= 0 ||
          !std::isfinite(timing.wallSeconds) || timing.wallSeconds <= 0) {
        recordFailure(MeasurementStatus::InvalidTiming);
        return false;
      }
      return ready();
    };
    const auto qualify = [&](uint32_t representative, WorkloadId distribution,
                              bool remember) {
      const auto &reference = fixture.references[distribution.value];
      if (!finiteOutput(fixture.buffers.output) ||
          (!remember && std::memcmp(reference.contents(), fixture.buffers.output.contents(),
                                     static_cast<size_t>(reference.sizeBytes())))) {
        recordFailure(MeasurementStatus::Rejected, std::make_exception_ptr(
            std::runtime_error("MoE tuning representative " + std::to_string(representative) +
                               " distribution " + std::to_string(distribution.value) +
                               " output is nonfinite or differs from its single baseline")));
        return false;
      }
      if (remember)
        std::memcpy(reference.contents(), fixture.buffers.output.contents(),
                      static_cast<size_t>(reference.sizeBytes()));
      return true;
    };
    // Existing M8/M32 tests require bit-exact candidate equivalence. Preserve
    // that gate for every real representative and both learned-route inputs.
    double baselineGpuSum = 0;
    for (const WorkloadId distribution : kDistributions)
      for (uint32_t representative = 0; representative < result.representativeCount;
           ++representative) {
        RunTiming timing;
        if (!probe(kBaseline, distribution, 1, representative, timing) ||
            !qualify(representative, distribution, true)) return result;
        baselineGpuSum += timing.gpuSeconds;
        if (!probe(kAlternative, distribution, 1, representative, timing) ||
            !qualify(representative, distribution, false)) return result;
      }
    const double baselineGpuMean =
        baselineGpuSum / (result.representativeCount * kDistributions.size());
    if (!std::isfinite(baselineGpuMean) || baselineGpuMean <= 0) {
      recordFailure(MeasurementStatus::InvalidTiming);
      return result;
    }
    const uint32_t requested = measurementBatchRepetitions(baselineGpuMean);
    const uint32_t rings = std::min(16U / result.representativeCount,
        (requested + result.representativeCount - 1) / result.representativeCount);
    result.repetitions = rings * result.representativeCount;
    // Complete rings end at the last representative. Its single-run baseline
    // is retained for each distribution above. Qualify precisely this shared
    // scratch reuse and weight order for both tiles before paired sampling.
    for (const WorkloadId distribution : kDistributions)
      for (const CandidateId candidate : {kBaseline, kAlternative}) {
        RunTiming timing;
        if (!probe(candidate, distribution, result.repetitions, 0, timing) ||
            !qualify(result.representativeCount - 1, distribution, false)) return result;
      }
    for (const WorkloadId distribution : kDistributions) {
      current = distribution;
      if (!ready())
        return result;
      auto measurementOptions = options;
      measurementOptions.maximumWallSeconds = remaining();
      auto measurement = measureWorkload(
          kAlternative, distribution, [&](CandidateId candidate) {
            return run(candidate, distribution, result.repetitions, 0);
          }, measurementOptions, shouldStop);
      if (measurement.status == MeasurementStatus::RunFailed && measurement.failure) {
        try { std::rethrow_exception(measurement.failure); }
        catch (const Interrupted &interruption) {
          measurement.status = interruption.status;
          measurement.failure = {};
        }
        catch (...) {}
      }
      const auto status = measurement.status;
      result.measurements.push_back(std::move(measurement));
      if (status != MeasurementStatus::Completed && status != MeasurementStatus::Rejected) {
        result.failure = result.measurements.back().failure;
        return result;
      }
    }
    if (!ready()) return result;
    std::array<WorkloadMeasurements, 2> gpu, wall;
    for (size_t index = 0; index < result.measurements.size(); ++index) {
      const auto &measurement = result.measurements[index];
      gpu[index] = {measurement.workload, measurement.rawGpuSamples()};
      wall[index] = {measurement.workload, measurement.rawWallSamples()};
    }
    const std::array gpuCandidate{CandidateMeasurements{kAlternative, gpu}};
    const std::array wallCandidate{CandidateMeasurements{kAlternative, wall}};
    const auto gpuSelection = selectCandidate(gpuCandidate, kDistributions, options.policy);
    const auto wallSelection = selectCandidate(wallCandidate, kDistributions, options.policy);
    if (gpuSelection.verdict == SelectionVerdict::Selected &&
        wallSelection.verdict == SelectionVerdict::Selected &&
        gpuSelection.candidate == wallSelection.candidate)
      result.choice.configuration = plans[1].config();
    result.complete = true;
  } catch (const Interrupted &interruption) {
    recordFailure(interruption.status);
  } catch (...) {
    recordFailure(MeasurementStatus::RunFailed, std::current_exception());
  }
  return result;
}

} // namespace splash::ops::tuning
