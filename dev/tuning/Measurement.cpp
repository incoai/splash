#include "tuning/Measurement.hpp"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <optional>

namespace splash::ops::tuning {

uint32_t measurementBatchRepetitions(double baselineGpuSeconds) noexcept {
  constexpr double targetSeconds = 0.005;
  if (!std::isfinite(baselineGpuSeconds) || baselineGpuSeconds <= 0 ||
      baselineGpuSeconds >= targetSeconds)
    return 1;
  return static_cast<uint32_t>(std::ceil(
      std::min(16.0, targetSeconds / baselineGpuSeconds)));
}

bool validMeasurementOptions(const MeasurementOptions &options) noexcept {
  return options.warmupPairs > 0 &&
         options.warmupPairs <= kMaximumWarmupPairs &&
         options.samplePairs >= options.policy.minimumPairs &&
         options.samplePairs <= kMaxPairedSamples &&
         std::isfinite(options.maximumWallSeconds) &&
         options.maximumWallSeconds >= 0 &&
         evaluate({}, options.policy).verdict != TimingVerdict::InvalidPolicy;
}

MeasurementResult measureWorkload(CandidateId candidate, const MeasurementRun &run,
                                  const MeasurementOptions &options,
                                  const MeasurementStop &shouldStop) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  MeasurementResult result;
  result.candidate = candidate;
  auto elapsed = [&] {
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  auto accepted = [&](const TimingAssessment &assessment) {
    return assessment.qualified() ||
           (options.acceptUncertain &&
            assessment.verdict == TimingVerdict::Uncertain);
  };
  auto finish = [&] {
    result.gpuAssessment = evaluate(result.rawGpuSamples(), options.policy);
    result.wallAssessment = evaluate(result.rawWallSamples(), options.policy);
    if (result.status == MeasurementStatus::Completed &&
        (!accepted(result.gpuAssessment) || !accepted(result.wallAssessment)))
      result.status = MeasurementStatus::Rejected;
    result.elapsedWallSeconds = elapsed();
    return result;
  };
  if (candidate == kBaseline || !run || !validMeasurementOptions(options))
    return finish();

  result.status = MeasurementStatus::Completed;
  auto checkControl = [&] {
    if (shouldStop && shouldStop()) {
      result.status = MeasurementStatus::Cancelled;
      return false;
    }
    if (elapsed() >= options.maximumWallSeconds) {
      result.status = MeasurementStatus::BudgetExceeded;
      return false;
    }
    return true;
  };
  auto invoke = [&](CandidateId id, MeasurementAccounting &accounting)
      -> std::optional<RunTiming> {
    std::optional<RunTiming> timing;
    try {
      if (!checkControl())
        return std::nullopt;
      ++accounting.attemptedCalls;
      timing = run(id);
      ++accounting.returnedCalls;
      const bool validGpu =
          std::isfinite(timing->gpuSeconds) && timing->gpuSeconds > 0;
      const bool validWall =
          std::isfinite(timing->wallSeconds) && timing->wallSeconds > 0;
      if (validGpu)
        accounting.gpuSeconds += timing->gpuSeconds;
      if (validWall)
        accounting.wallSeconds += timing->wallSeconds;
      if (timing->underPressure)
        result.status = MeasurementStatus::UnderPressure;
      else if (!validGpu || !validWall || !std::isfinite(accounting.gpuSeconds) ||
               !std::isfinite(accounting.wallSeconds))
        result.status = MeasurementStatus::InvalidTiming;
      else
        (void)checkControl();
    } catch (...) {
      result.status = MeasurementStatus::RunFailed;
      result.failure = std::current_exception();
    }
    return timing;
  };

  for (size_t pair = 0; pair < options.warmupPairs; ++pair) {
    const bool baselineFirst =
        measurementOrder(pair) == MeasurementOrder::BaselineFirst;
    (void)invoke(baselineFirst ? kBaseline : candidate, result.warmup);
    if (result.status != MeasurementStatus::Completed)
      return finish();
    (void)invoke(baselineFirst ? candidate : kBaseline, result.warmup);
    if (result.status != MeasurementStatus::Completed)
      return finish();
  }
  for (size_t pair = 0; pair < options.samplePairs; ++pair) {
    const auto order = measurementOrder(pair);
    const bool baselineFirst = order == MeasurementOrder::BaselineFirst;
    const auto first = invoke(baselineFirst ? kBaseline : candidate,
                              result.measurement);
    if (result.status != MeasurementStatus::Completed)
      return finish();
    const auto second = invoke(baselineFirst ? candidate : kBaseline,
                               result.measurement);
    // Retain completed pairs even if the second return was invalid, pressured,
    // cancelled or over budget. Never invent the missing half of an aborted pair.
    if (first && second) {
      const auto &baseline = baselineFirst ? *first : *second;
      const auto &tested = baselineFirst ? *second : *first;
      const bool pressure = baseline.underPressure || tested.underPressure;
      result.gpuPairs[result.pairCount] =
          {baseline.gpuSeconds, tested.gpuSeconds, order, pressure};
      result.wallPairs[result.pairCount] =
          {baseline.wallSeconds, tested.wallSeconds, order, pressure};
      ++result.pairCount;
    }
    if (result.status != MeasurementStatus::Completed)
      return finish();
  }
  return finish();
}

} // namespace splash::ops::tuning
