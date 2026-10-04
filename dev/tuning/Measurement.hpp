#pragma once

#include "tuning/Tuning.hpp"

#include <array>
#include <exception>
#include <functional>

namespace splash::ops::tuning {

struct RunTiming final {
  double gpuSeconds = 0;
  double wallSeconds = 0;
  bool underPressure = false;
};

// The caller executes the same warmed workload through its production encoder
// for either ID. It owns input/state restoration and command completion. Each
// return must cover the entire workload, not a selected dispatch within it.
using MeasurementRun = std::function<RunTiming(CandidateId)>;
using MeasurementStop = std::function<bool()>;

inline constexpr size_t kMaximumWarmupPairs = 4;

struct MeasurementOptions final {
  size_t warmupPairs = 2;
  size_t samplePairs = kMinPairedSamples;
  // Zero gives deterministic expiry; a tiny positive budget does not, since
  // consecutive steady-clock reads can be equal.
  double maximumWallSeconds = 5;
  Policy policy;
  // Non-regression gates accept an Uncertain verdict: the median paired gain
  // is nonnegative and both timing spreads are within policy, but the
  // conservative gain straddles zero. Selection keeps requiring Improved.
  bool acceptUncertain = false;
};

[[nodiscard]] bool
validMeasurementOptions(const MeasurementOptions &options) noexcept;

// Amortize submission noise with complete operator repetitions, never partial
// dispatch timing. The operator must prove repeated execution is equivalent
// and use the same count for every candidate. Invalid pilots do not qualify.
[[nodiscard]] uint32_t measurementBatchRepetitions(double baselineGpuSeconds) noexcept;

enum class MeasurementStatus : uint8_t {
  Completed,
  InvalidInput,
  Cancelled,
  BudgetExceeded,
  UnderPressure,
  InvalidTiming,
  Rejected,
  RunFailed,
};

[[nodiscard]] constexpr std::string_view measurementStatusName(MeasurementStatus status) noexcept {
  switch (status) {
  case MeasurementStatus::Completed: return "completed";
  case MeasurementStatus::InvalidInput: return "invalid_input";
  case MeasurementStatus::Cancelled: return "cancelled";
  case MeasurementStatus::BudgetExceeded: return "budget_exceeded";
  case MeasurementStatus::UnderPressure: return "under_pressure";
  case MeasurementStatus::InvalidTiming: return "invalid_timing";
  case MeasurementStatus::Rejected: return "rejected";
  case MeasurementStatus::RunFailed: return "run_failed";
  }
  return "unknown";
}

struct MeasurementAccounting final {
  size_t attemptedCalls = 0;
  size_t returnedCalls = 0;
  // Sums of finite positive reported fields, including a returned unmatched
  // run. Invalid fields do not poison the totals; status records the failure.
  double gpuSeconds = 0;
  double wallSeconds = 0;
};

struct MeasurementResult final {
  CandidateId candidate;
  MeasurementStatus status = MeasurementStatus::InvalidInput;
  MeasurementAccounting warmup;
  MeasurementAccounting measurement;
  double elapsedWallSeconds = 0;
  size_t pairCount = 0;
  std::array<PairedTiming, kMaxPairedSamples> gpuPairs{};
  std::array<PairedTiming, kMaxPairedSamples> wallPairs{};
  TimingAssessment gpuAssessment;
  TimingAssessment wallAssessment;
  // The caller decides how an execution/control callback failure affects its
  // backend. This utility never retries, rolls back or suppresses that detail.
  std::exception_ptr failure;

  // Raw pairs remain available for diagnostics on rejected/interrupted runs.
  // Callers check completion status before selecting each metric independently.
  [[nodiscard]] std::span<const PairedTiming> rawGpuSamples() const noexcept {
    return {gpuPairs.data(), pairCount};
  }
  [[nodiscard]] std::span<const PairedTiming> rawWallSamples() const noexcept {
    return {wallPairs.data(), pairCount};
  }
};

// Warmups and measured pairs each reverse baseline/candidate execution order.
// Requires 1-4 warmup pairs and policy.minimumPairs..64 measured pairs. Bounds
// and cancellation are checked between synchronous calls, never by interrupting
// an active command. A callback must return before its deadline can be observed.
// Both timing metrics must pass the existing noise/non-regression policy;
// meaningful improvement remains selectCandidate's responsibility.
// This function does not encode work, change execution policy or persist data.
[[nodiscard]] MeasurementResult
measureWorkload(CandidateId candidate, const MeasurementRun &run,
                const MeasurementOptions &options = {},
                const MeasurementStop &shouldStop = {});

} // namespace splash::ops::tuning
