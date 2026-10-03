#pragma once

#include "ops/Linear.hpp"
#include "tuning/Measurement.hpp"

#include <optional>
#include <vector>

namespace splash::ops::tuning {

inline constexpr size_t kMaximumLinearTuningRepresentatives = 8;

// The plans tuneLinear measures for an affine workload, the device policy's
// first. The others are the configurations that beat the policy on some key
// when tune-kernels ran the 27B and 35B-A3B models on an M5 Max, an M5 Pro
// and an M3 Max: prefill N128 at eight and at four simdgroups and N256;
// decode N128 and N256 on their full grids, Split128 at each K split up to
// eight on Apple10 and later, and the paired N256 tile on its full grid for
// one-lane plain projections, each once and only where a kernel runs it for
// the workload. A block-quantized workload throws: block plans are not
// tuned.
[[nodiscard]] std::vector<LinearPlan> linearCandidates(
    const DeviceCapabilities &device, LinearWorkload workload);

struct LinearTuningWeights final {
  Projection projection;
  // Required only for GateUp. UpWithGate consumes deterministic gate scratch
  // as its semantic input; it does not include an external gate projection.
  std::optional<Projection> gate;
};

struct LinearTuningInput final {
  LinearWorkload workload;
  // 1..8 immutable representative views supplied by the model. Each timed
  // batch visits them in this fixed order before repeating the ring. The
  // operator never copies weights or invents a cache-flushing workload.
  std::vector<LinearTuningWeights> weights;
};

struct LinearTuningResult final {
  // The selected configuration: the baseline's unless a candidate won.
  LinearConfig configuration;
  std::vector<MeasurementResult> measurements;
  bool complete = false;
  std::exception_ptr failure;
  // Actual complete operators per timed command and supplied weight views.
  // Paired seconds cover the whole batch, not a divided per-operator estimate.
  // repetitions remains zero if qualification/pilot did not finish.
  uint32_t repetitions = 0;
  uint32_t representativeCount = 0;
};

// Exact admitted shared-fixture bytes, including the reference outputs and
// maximum workspace of every linearCandidates plan. CPU-only; invalid shapes
// and a fixture exceeding a supplied nonzero device buffer limit throw.
[[nodiscard]] uint64_t linearTuningFixtureBytes(
    const DeviceCapabilities &device, LinearWorkload workload);

// Offline only: real supplied weights, deterministic BF16 inputs and the
// production Linear graph. Candidate IDs index linearCandidates.
// Every representative is qualified against its own baseline before timing.
// Existing baseline qualification timings select a fixed batch of 1..16 whole
// operators, rounded to complete representative rings, targeting about 5 ms.
// A final batched qualification verifies that scratch reuse is equivalent.
// Restoring scratch/output and exact finite output checks are outside timing;
// wall time covers graph encoding and synchronous submission. The wall budget
// covers this entire sweep, including warmup and output qualification.
// Both timing metrics must independently select the same non-baseline winner.
// complete includes timing rejections, but excludes interrupted/failed sweeps.
// Metal failures are preserved and never retried. No result is persisted here;
// a winner still needs a whole-model A/B before any policy change.
[[nodiscard]] LinearTuningResult tuneLinear(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    const LinearTuningInput &input,
    const MeasurementOptions &options = {},
    const MeasurementStop &underPressure = {},
    const MeasurementStop &shouldStop = {});

} // namespace splash::ops::tuning
