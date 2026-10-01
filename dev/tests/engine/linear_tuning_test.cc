#include "AffineQ4Fixture.hpp"
#include "tuning/LinearTuning.hpp"

#include "metal/abi/QuantFormat.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {
using namespace splash;
using namespace splash::ops;
using namespace splash::ops::tuning;
using splash::test::deterministicQ4Projection;
using splash::test::mix;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template <class Function> void rejects(Function function) {
  try { function(); }
  catch (const std::invalid_argument &) { return; }
  throw std::runtime_error("invalid Linear fixture accepted");
}

void cpuContracts() {
  DeviceCapabilities device;
  device.appleGpuFamily = 9;
  device.maxBufferLengthBytes = uint64_t{1} << 40;
  for (const uint32_t hidden : {2048U, 5120U}) {
    const LinearMatrix matrix{hidden, hidden};
    for (const auto phase : {LinearPhase::Prefill, LinearPhase::Decode}) {
      for (const uint32_t rows : {8U, 16U, 24U, 32U}) {
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
             phase == LinearPhase::Prefill ? LinearEpilogue::UpWithGate : LinearEpilogue::GateUp}) {
          const LinearWorkload workload{matrix, rows, phase, epilogue};
          const auto plans = Linear(device).candidates(workload);
          const uint64_t bytes = linearTuningFixtureBytes(device, workload);
          const uint64_t base = uint64_t{plans.front().storageRows()} *
              (matrix.inputSize + 2 * matrix.outputSize) * 2;
          require(bytes >= base && bytes % 16384 == 0,
                  "fixture does not cover input/output/reference or physical alignment");
          for (const auto &plan : plans)
            require(bytes >= base + plan.sumsBytes() + plan.gateScratchBytes() +
                2 * plan.downSumsBytes() + plan.scratchSize().bytes(),
                "fixture misses a candidate workspace");
          auto denied = device;
          denied.maxBufferLengthBytes = bytes - 1;
          rejects([&] { (void)linearTuningFixtureBytes(denied, workload); });
          denied.maxBufferLengthBytes = bytes;
          require(linearTuningFixtureBytes(denied, workload) == bytes,
                  "exact admitted capacity rejected");
        }
      }
    }
  }
  for (const LinearWorkload workload : {
       LinearWorkload{{0, 256}, 8}, LinearWorkload{{128, 256}, 8},
       LinearWorkload{{512, 64}, 8}, LinearWorkload{{512, 256}, 7},
       LinearWorkload{{512, 256}, 40},
       LinearWorkload{{512, 256}, 8, static_cast<LinearPhase>(255)},
       LinearWorkload{{512, 256}, 8, LinearPhase::Decode, static_cast<LinearEpilogue>(255)},
       LinearWorkload{{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::UpWithGate},
       LinearWorkload{{512, 256}, 8, LinearPhase::Prefill, LinearEpilogue::GateUp},
       LinearWorkload{{512, 256}, 2049, LinearPhase::Prefill}})
    rejects([&] { (void)linearTuningFixtureBytes(device, workload); });
  MeasurementOptions options;
  require(validMeasurementOptions(options), "default measurement options rejected");
  options.samplePairs = 11;
  require(!validMeasurementOptions(options), "insufficient paired samples accepted");
}

// The tuner must use the supplied packed bytes without copying or modifying
// them; it owns only activation and comparison scratch.
uint64_t fingerprint(const Projection &projection) {
  uint64_t hash = 14695981039346656037ULL;
  for (const auto &buffer : {projection.affine().weights, projection.affine().scales, projection.affine().biases}) {
    const auto *bytes = static_cast<const uint8_t *>(buffer.contents());
    for (uint64_t i = 0; i < buffer.sizeBytes(); ++i)
      hash = (hash ^ bytes[i]) * 1099511628211ULL;
  }
  return hash;
}

// Block-quantized workloads are not tuned, and a block projection never
// reaches the tuner's reads of affine planes (which would throw
// std::bad_variant_access instead).
void blockInputs(metal::MetalBackend &backend, const Projection &projection) {
  const Projection block(projection.outputSize, projection.inputSize,
                         BlockWeights{{QuantizedSegment::planes(GGUF_FMT_Q4K, projection.outputSize,
                                                                projection.inputSize, {}, {}, {})}});
  const LinearWorkload affine{{projection.outputSize, projection.inputSize}, 8};
  LinearWorkload blocks = affine;
  blocks.weightLayout = WeightLayout::Block32;
  const uint64_t before = backend.submissionCount();
  const auto admit = [](uint64_t, const std::function<void()> &) -> metal::AllocationResult {
    throw std::logic_error("invalid tuning input reached admission");
  };
  for (const LinearTuningInput &input : {LinearTuningInput{blocks, {{block, std::nullopt}}},
                                         LinearTuningInput{affine, {{block, std::nullopt}}}}) {
    const auto result = tuneLinear(backend, admit, input);
    require(!result.complete && result.failure && result.measurements.empty(),
            "a block tuning input was accepted");
    rejects([&] { std::rethrow_exception(result.failure); });
  }
  require(backend.submissionCount() == before, "a block tuning input submitted GPU work");
}

void gpuControls(metal::MetalBackend &backend, const Projection &projection) {
  const LinearWorkload workload{{projection.outputSize, projection.inputSize}, 8};
  const LinearTuningInput input{workload, {{projection, std::nullopt}}};
  const auto baseline = Linear(backend.capabilities()).plan(workload).configuration();
  const uint64_t before = backend.submissionCount();
  size_t calls = 0;
  auto admit = [&](uint64_t bytes, const std::function<void()> &allocate) {
    ++calls;
    require(bytes == linearTuningFixtureBytes(backend.capabilities(), workload),
            "admission size differs from planning helper");
    allocate();
    return true;
  };
  auto assertStopped = [&](const LinearTuningResult &result, bool failure) {
    require(!result.complete && result.choice.configuration == baseline &&
        bool(result.failure) == failure && result.measurements.empty(),
        "early exit lost baseline or failure");
    require(backend.submissionCount() == before, "early exit submitted GPU work");
  };
  MeasurementOptions options;
  options.warmupPairs = 0;
  assertStopped(tuneLinear(backend, admit, input, options), true);
  assertStopped(tuneLinear(backend, {}, input), true);
  assertStopped(tuneLinear(backend, admit, {workload, {}}), true);
  assertStopped(tuneLinear(backend, admit, {workload, {{{}, {}}}}), true);
  assertStopped(tuneLinear(backend, admit, {workload, {{projection, projection}}}), true);
  auto tooMany = input;
  tooMany.weights.resize(kMaximumLinearTuningRepresentatives + 1, input.weights[0]);
  assertStopped(tuneLinear(backend, admit, tooMany), true);
  auto invalidSecond = input;
  invalidSecond.weights.push_back({projection, projection});
  assertStopped(tuneLinear(backend, admit, invalidSecond), true);
  assertStopped(tuneLinear(backend, admit, input, {}, {}, [] { return true; }), false);
  assertStopped(tuneLinear(backend, admit, input, {}, [] { return true; }), false);
  options = {};
  options.maximumWallSeconds = 0;
  assertStopped(tuneLinear(backend, admit, input, options), false);
  require(calls == 0, "invalid or cancelled sweep reached admission");
  bool denied = false;
  assertStopped(tuneLinear(backend, [&](uint64_t bytes, const auto &) {
    denied = true;
    require(bytes == linearTuningFixtureBytes(backend.capabilities(), workload),
            "denied admission requested wrong bytes");
    return false;
  }, input), false);
  require(denied, "allocation denial not exercised");
  assertStopped(tuneLinear(backend, [](uint64_t, const auto &) { return true; }, input), true);
  assertStopped(tuneLinear(backend, [](uint64_t, const auto &) -> bool {
    throw metal::MetalAllocationError("test admission capacity failure");
  }, input), true);
  // Denial must not leave physical backing behind. This deliberate contract
  // violation is distinct from a caught allocator failure with empty backing.
  const auto allocated = backend.memoryStats().allocatedBytes;
  assertStopped(tuneLinear(backend, [](uint64_t, const auto &allocate) {
    allocate();
    return false;
  }, input), true);
  require(backend.memoryStats().allocatedBytes == allocated,
          "invalid admission contract leaked the temporary fixture");
  require(backend.healthy(), "admission failure damaged Metal health");
}

void gpuSweep(metal::MetalBackend &backend, std::span<const Projection> projections,
              uint32_t rows, LinearPhase phase, LinearEpilogue epilogue) {
  const auto &projection = projections.front();
  const LinearWorkload workload{{projection.outputSize, projection.inputSize}, rows, phase, epilogue};
  LinearTuningInput input{workload, {}};
  std::vector<uint64_t> weightsBefore;
  for (const auto &weights : projections) {
    input.weights.push_back({weights, epilogue == LinearEpilogue::GateUp
        ? std::optional{weights} : std::nullopt});
    weightsBefore.push_back(fingerprint(weights));
  }
  const auto plans = Linear(backend.capabilities()).candidates(workload);
  // A gate/up sweep mixing split-K and sequential plans computes the exact
  // gate and up projections once per representative, outside the timing.
  bool mixed = false;
  for (const auto &plan : plans) mixed |= plan.partialSums() != plans.front().partialSums() ||
      plan.usesSimdgroup() || plans.front().usesSimdgroup();
  const uint64_t referenceSubmissions =
      mixed && epilogue == LinearEpilogue::GateUp ? projections.size() : 0;
  const uint64_t before = backend.submissionCount();
  const uint64_t allocated = backend.memoryStats().allocatedBytes;
  size_t admissions = 0;
  MeasurementOptions options;
  options.maximumWallSeconds = 30;
  const auto result = tuneLinear(backend, [&](uint64_t bytes, const auto &allocate) {
    ++admissions;
    require(bytes == linearTuningFixtureBytes(backend.capabilities(), workload),
            "sweep admission size mismatch");
    allocate();
    require(backend.memoryStats().allocatedBytes - allocated == bytes,
            "fixture physical bytes differ from admission");
    return true;
  }, input, options);
  if (result.failure) std::rethrow_exception(result.failure);
  require(result.complete && admissions == 1, "bounded sweep incomplete or allocated twice");
  require(result.representativeCount == projections.size() && result.repetitions >= projections.size() &&
      result.repetitions <= 16 && result.repetitions % projections.size() == 0,
      "timed batch does not cover a bounded complete representative ring");
  require(result.measurements.size() + 1 == plans.size(), "candidate measurement missing");
  require(backend.submissionCount() - before == plans.size() * (projections.size() + 1) +
      (plans.size() - 1) * 2 * (options.warmupPairs + options.samplePairs) + referenceSubmissions,
      "sweep did not time one full production command per invocation");
  require(backend.memoryStats().allocatedBytes == allocated, "fixture allocation leaked");
  for (size_t i = 0; i < projections.size(); ++i)
    require(fingerprint(projections[i]) == weightsBefore[i], "tuning modified supplied projection");
  std::vector<WorkloadMeasurements> gpu, wall;
  gpu.reserve(result.measurements.size());
  wall.reserve(result.measurements.size());
  std::vector<CandidateMeasurements> gpuCandidates, wallCandidates;
  constexpr WorkloadId id{0};
  for (size_t i = 0; i < result.measurements.size(); ++i) {
    const auto &measurement = result.measurements[i];
    require(measurement.candidate == CandidateId{uint32_t(i + 1)} &&
        measurement.pairCount == options.samplePairs && !measurement.failure &&
        (measurement.status == MeasurementStatus::Completed ||
         measurement.status == MeasurementStatus::Rejected),
        "candidate raw timings missing or incorrectly identified");
    for (size_t pair = 0; pair < measurement.pairCount; ++pair)
      require(measurement.gpuPairs[pair].first == measurementOrder(pair) &&
          measurement.wallPairs[pair].first == measurementOrder(pair),
          "baseline/candidate order did not alternate");
    gpu.push_back({id, measurement.rawGpuSamples()});
    wall.push_back({id, measurement.rawWallSamples()});
    gpuCandidates.push_back({measurement.candidate, {&gpu.back(), 1}});
    wallCandidates.push_back({measurement.candidate, {&wall.back(), 1}});
  }
  const auto gpuWinner = selectCandidate(gpuCandidates, {&id, 1}, options.policy);
  const auto wallWinner = selectCandidate(wallCandidates, {&id, 1}, options.policy);
  const bool agreed = gpuWinner.verdict == SelectionVerdict::Selected &&
      wallWinner.verdict == SelectionVerdict::Selected && gpuWinner.candidate == wallWinner.candidate;
  require(result.choice.workload == workload && result.choice.configuration ==
      plans[agreed ? gpuWinner.candidate.value : 0].configuration(),
      "tuning accepted a winner without independent GPU/wall agreement");
}

void gpuInterruptions(metal::MetalBackend &backend, Projection &projection) {
  const LinearWorkload workload{{projection.outputSize, projection.inputSize}, 8};
  const LinearTuningInput input{workload, {{projection, std::nullopt}}};
  const auto plans = Linear(backend.capabilities()).candidates(workload);
  require(plans.size() > 1, "interruption fixture has no alternative");
  const auto admit = [](uint64_t, const auto &allocate) { allocate(); return true; };
  MeasurementOptions options;
  options.maximumWallSeconds = 30;
  for (bool pressure : {false, true}) {
    const uint64_t before = backend.submissionCount();
    const uint64_t stopAt = before + 2 * plans.size() + 2 * options.warmupPairs + 3;
    const MeasurementStop stop = [&] { return backend.submissionCount() >= stopAt; };
    const auto result = tuneLinear(backend, admit, input, options,
        pressure ? stop : MeasurementStop{}, pressure ? MeasurementStop{} : stop);
    require(!result.complete && !result.failure && result.measurements.size() == 1 &&
        result.choice.configuration == plans.front().configuration() &&
        result.measurements[0].status == (pressure ? MeasurementStatus::UnderPressure :
            MeasurementStatus::Cancelled) && result.measurements[0].pairCount == 1 &&
        result.measurements[0].measurement.returnedCalls == 3 &&
        backend.submissionCount() == stopAt, "interruption fabricated pairs or continued submitting");
  }
  const uint64_t before = backend.submissionCount();
  const uint64_t failAt = before + 2 * plans.size() + 1;
  const auto result = tuneLinear(backend, admit, input, options, [&] {
    if (backend.submissionCount() >= failAt) throw std::runtime_error("test pressure callback failure");
    return false;
  });
  require(!result.complete && result.failure && backend.submissionCount() == failAt &&
      result.measurements.size() == 1 && result.measurements[0].failure &&
      result.measurements[0].status == MeasurementStatus::RunFailed,
      "run callback failure was lost or retried");
  auto *scales = static_cast<uint16_t *>(projection.affine().scales.contents());
  const uint16_t saved = scales[0];
  scales[0] = 0x7fc1;
  const uint64_t beforeInvalid = backend.submissionCount();
  const auto invalid = tuneLinear(backend, admit, input, options);
  scales[0] = saved;
  require(!invalid.complete && invalid.failure && invalid.measurements.empty() &&
      backend.submissionCount() == beforeInvalid + 1 && backend.healthy(),
      "nonfinite baseline was timed or correctness failure retried");
}

void gpuEveryRepresentative(metal::MetalBackend &backend,
                            const Projection &first, Projection &second) {
  const LinearWorkload workload{{first.outputSize, first.inputSize}, 8};
  const LinearTuningInput input{workload, {{first, {}}, {second, {}}}};
  const auto plans = Linear(backend.capabilities()).candidates(workload);
  auto *scales = static_cast<uint16_t *>(second.affine().scales.contents());
  const auto saved = scales[0];
  scales[0] = 0x7fc1;
  const auto before = backend.submissionCount();
  const auto result = tuneLinear(backend, [](uint64_t, const auto &allocate) {
    allocate(); return true;
  }, input);
  scales[0] = saved;
  require(!result.complete && result.failure && result.measurements.empty() &&
      result.representativeCount == 2 && result.repetitions == 0 &&
      backend.submissionCount() - before == plans.size() + 1,
      "a later representative was not qualified before sampling");
}

// Independent batching check uses the public production encoder, including
// every prefill sum and GateUp phase. Repetition cannot silently depend on
// the preceding representative's scratch, output, or residual values.
void gpuBatchEquivalence(metal::MetalBackend &backend,
                         std::span<const Projection> projections,
                         uint32_t rows, LinearPhase phase,
                         LinearEpilogue epilogue, bool selfComparison = false) {
  const LinearWorkload workload{{projections.front().outputSize,
      projections.front().inputSize}, rows, phase, epilogue};
  Linear linear(backend.capabilities());
  const auto plans = linear.candidates(workload);
  const auto &baseline = plans.front();
  uint64_t gateBytes = 0;
  for (const auto &plan : plans) gateBytes = std::max(gateBytes, plan.gateScratchBytes());
  const auto allocate = [&](uint64_t bytes) {
    return bytes ? backend.allocateBuffer(bytes) : metal::MetalBuffer{};
  };
  LinearBuffers buffers{
      allocate(uint64_t{baseline.storageRows()} * workload.matrix.inputSize * 2),
      allocate(uint64_t{baseline.storageRows()} * workload.matrix.outputSize * 2),
      allocate(baseline.sumsBytes()),
      allocate(epilogue == LinearEpilogue::Residual ?
          uint64_t{baseline.storageRows()} * workload.matrix.outputSize * 2 : 0),
      allocate(gateBytes), allocate(baseline.downSumsBytes())};
  LinearScratchSize scratch;
  // This fixture runs every candidate, whose K split count can require more
  // partials than the default. Match the production tuner's field maxima.
  for (const auto &plan : plans) {
    const auto needed = plan.scratchSize();
    scratch.input = std::max(scratch.input, needed.input);
    scratch.sums = std::max(scratch.sums, needed.sums);
    scratch.partials = std::max(scratch.partials, needed.partials);
    scratch.counters = std::max(scratch.counters, needed.counters);
  }
  buffers.scratch = {allocate(scratch.input), allocate(scratch.sums),
                     allocate(scratch.partials), allocate(scratch.counters)};
  if (buffers.scratch.counters)
    std::memset(buffers.scratch.counters.contents(), 0, scratch.counters);
  const auto fill = [&](const metal::MetalBuffer &buffer, uint32_t width, uint32_t seed) {
    auto *values = static_cast<uint16_t *>(buffer.contents());
    for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i)
      values[i] = i < uint64_t{rows} * width
          ? floatToBf16(float(int(mix(uint32_t(i) + seed) % 257) - 128) / 257.0f) : 0;
  };
  fill(buffers.input, workload.matrix.inputSize, 1949);
  if (buffers.residual) fill(buffers.residual, workload.matrix.outputSize, 7919);
  const auto reset = [&] {
    std::fill_n(static_cast<uint16_t *>(buffers.output.contents()),
                buffers.output.sizeBytes() / 2, uint16_t{0x7fc1});
    for (const auto &scratch : {buffers.sums, buffers.gateScratch, buffers.downSums})
      if (scratch) std::memset(scratch.contents(), 0, scratch.sizeBytes());
    if (epilogue == LinearEpilogue::UpWithGate)
      fill(buffers.gateScratch, workload.matrix.outputSize, 104729);
  };
  const auto invoke = [&](const LinearPlan &plan, uint32_t first, uint32_t repetitions) {
    reset();
    const auto start = std::chrono::steady_clock::now();
    metal::CommandGraph graph;
    for (uint32_t i = 0; i < repetitions; ++i) {
      const auto &projection = projections[(first + i) % projections.size()];
      if (phase == LinearPhase::Prefill)
        linear.addPrefillSums(graph, buffers.input, buffers.sums, projection, rows);
      linear.add(graph, buffers, projection, plan,
          epilogue == LinearEpilogue::GateUp ? &projection : nullptr);
    }
    const auto timing = backend.submitCommand(graph.dispatches());
    return RunTiming{timing.gpuSeconds,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), false};
  };
  const auto snapshot = [](const metal::MetalBuffer &buffer) {
    std::vector<uint8_t> bytes(buffer.sizeBytes());
    if (buffer) std::memcpy(bytes.data(), buffer.contents(), bytes.size());
    return bytes;
  };
  const auto same = [](const metal::MetalBuffer &buffer, const auto &reference) {
    return !buffer || std::memcmp(buffer.contents(), reference.data(), reference.size()) == 0;
  };
  const auto immutableInput = snapshot(buffers.input);
  const auto immutableResidual = snapshot(buffers.residual);
  for (uint32_t repetitions : {1U, uint32_t(projections.size()), 16U}) {
    const uint32_t last = (repetitions - 1) % projections.size();
    for (const auto &plan : plans) {
      (void)invoke(plan, last, 1);
      const auto expected = snapshot(buffers.output);
      const auto expectedSums = snapshot(buffers.downSums);
      const auto expectedGate = snapshot(buffers.gateScratch);
      (void)invoke(plan, 0, repetitions);
      require(same(buffers.output, expected) && same(buffers.downSums, expectedSums),
              "repeated whole operator differs from isolated last representative");
      require(same(buffers.input, immutableInput) && same(buffers.residual, immutableResidual),
              "repeated graph mutated immutable input or residual");
      if (epilogue == LinearEpilogue::UpWithGate)
        require(same(buffers.gateScratch, expectedGate), "UpWithGate mutated its semantic gate input");
      const auto *output = static_cast<const uint16_t *>(buffers.output.contents());
      for (uint64_t i = 0; i < buffers.output.sizeBytes() / 2; ++i)
        require((output[i] & 0x7f80) != 0x7f80, "repeated graph produced nonfinite output");
    }
  }
  if (!selfComparison) return;
  MeasurementOptions options;
  options.maximumWallSeconds = 30;
  // Candidate ID changes order/accounting only: both calls encode exactly the
  // same 16 complete baseline operators. This is a diagnostic, not a synthetic
  // performance win or a replacement for production-graph confirmation.
  const auto measurement = measureWorkload({1}, {0}, [&](CandidateId) {
    return invoke(baseline, 0, 16);
  }, options);
  if (measurement.failure) std::rethrow_exception(measurement.failure);
  require(measurement.pairCount == options.samplePairs &&
      (measurement.status == MeasurementStatus::Completed ||
       measurement.status == MeasurementStatus::Rejected), "baseline self-comparison was interrupted");
  const WorkloadMeasurements gpu{{0}, measurement.rawGpuSamples()};
  const WorkloadMeasurements wall{{0}, measurement.rawWallSamples()};
  const CandidateMeasurements gpuCandidate{{1}, {&gpu, 1}}, wallCandidate{{1}, {&wall, 1}};
  constexpr WorkloadId required{0};
  const auto gpuSelection = selectCandidate({&gpuCandidate, 1}, {&required, 1}, options.policy);
  const auto wallSelection = selectCandidate({&wallCandidate, 1}, {&required, 1}, options.policy);
  const bool jointSelection = gpuSelection.verdict == SelectionVerdict::Selected &&
      wallSelection.verdict == SelectionVerdict::Selected;
  std::cout << "Linear self-comparison repetitions=16 representatives=" << projections.size()
            << " gpu_gain=" << measurement.gpuAssessment.medianPairedGain
            << " wall_gain=" << measurement.wallAssessment.medianPairedGain
            << " joint_selection=" << jointSelection << '\n';
  require(!jointSelection, "identical baseline graphs qualified a false improvement");
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: linear-tuning <production.metallib|--cpu>");
    cpuContracts();
    if (std::string_view(argv[1]) == "--cpu") {
      std::cout << "Linear tuning CPU fixture contracts: PASS\n";
      return 0;
    }
    metal::MetalBackend backend(argv[1]);
    std::array small{deterministicQ4Projection(backend, {512, 256}, 29),
                     deterministicQ4Projection(backend, {512, 256}, 131),
                     deterministicQ4Projection(backend, {512, 256}, 233)};
    gpuControls(backend, small[0]);
    blockInputs(backend, small[0]);
    for (uint32_t rows : {8U, 16U, 24U, 32U}) {
      gpuSweep(backend, small, rows, LinearPhase::Decode, LinearEpilogue::None);
      gpuSweep(backend, small, rows, LinearPhase::Decode, LinearEpilogue::Residual);
    }
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                               LinearEpilogue::UpWithGate})
      gpuSweep(backend, small, 33, LinearPhase::Prefill, epilogue);
    std::array gate{deterministicQ4Projection(backend, {10240, 256}, 29),
                    deterministicQ4Projection(backend, {10240, 256}, 131)};
    for (uint32_t rows : {8U, 16U, 24U, 32U})
      gpuSweep(backend, gate, rows, LinearPhase::Decode, LinearEpilogue::GateUp);
    // K % 1024 == 0 lists the split-K tiles beside the sequential ones, and on
    // a GPU with two or more cores the baseline splits K too (Apple9's
    // simdgroup tile, Apple10's Split128), so every qualification runs the
    // derived bound, including a one-lane split tile's against a Split128
    // baseline with as many partial sums.
    std::array split{deterministicQ4Projection(backend, {512, 1024}, 29),
                     deterministicQ4Projection(backend, {512, 1024}, 131)};
    bool mixedClasses = false;
    for (const auto &plan : Linear(backend.capabilities()).candidates({{512, 1024}, 8}))
      mixedClasses |= plan.partialSums() > 1;
    require(mixedClasses, "split-K candidates are missing for a K % 1024 == 0 workload");
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                               LinearEpilogue::GateUp})
      gpuSweep(backend, split, 8, LinearPhase::Decode, epilogue);
    std::vector<Projection> maximum;
    for (uint32_t i = 0; i < kMaximumLinearTuningRepresentatives; ++i)
      maximum.push_back(deterministicQ4Projection(backend, {512, 256}, 1009 + i));
    gpuSweep(backend, maximum, 8, LinearPhase::Decode, LinearEpilogue::None);
    gpuInterruptions(backend, small[0]);
    gpuEveryRepresentative(backend, small[0], small[1]);
    gpuBatchEquivalence(backend, small, 8, LinearPhase::Decode, LinearEpilogue::None, true);
    gpuBatchEquivalence(backend, small, 8, LinearPhase::Decode, LinearEpilogue::Residual);
    gpuBatchEquivalence(backend, small, 33, LinearPhase::Prefill, LinearEpilogue::UpWithGate);
    gpuBatchEquivalence(backend, gate, 24, LinearPhase::Decode, LinearEpilogue::GateUp);
    std::cout << "Linear tuning production sweeps and controls: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "Linear tuning: " << error.what() << '\n';
    return 1;
  }
}
