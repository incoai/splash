#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace splash::ops::tuning {

// IDs refer to operator-owned typed configurations. This helper never
// interprets a kernel configuration or encodes GPU work.
struct CandidateId final {
  uint32_t value = 0;
  bool operator==(const CandidateId &) const = default;
};

inline constexpr CandidateId kBaseline{};
inline constexpr size_t kMinPairedSamples = 12;
inline constexpr size_t kMaxPairedSamples = 64;

enum class MeasurementOrder : uint8_t { BaselineFirst, CandidateFirst };

[[nodiscard]] constexpr MeasurementOrder measurementOrder(size_t pair) noexcept {
  return pair % 2 == 0 ? MeasurementOrder::BaselineFirst
                       : MeasurementOrder::CandidateFirst;
}

// Each pair must measure the same warmed workload through the production
// entry point. Record execution order rather than sorting samples by time.
// Either order may start a run, but consecutive pairs must reverse it.
struct PairedTiming final {
  double baselineSeconds = 0;
  double candidateSeconds = 0;
  MeasurementOrder first = MeasurementOrder::BaselineFirst;
  bool underPressure = false;
};

// These are conservative engineering thresholds, not a statistical confidence
// interval or a guarantee about unmeasured workloads. Timing spread is the
// central 80% range / median; paired gain spread is the central 80% gain range.
// The conservative gain subtracts that entire paired range from median gain.
struct Policy final {
  // A policy may require more samples, but cannot weaken the 12-pair floor.
  size_t minimumPairs = kMinPairedSamples;
  double maximumRelativeTimingSpread = 0.10;
  double maximumPairedGainSpread = 0.05;
  double minimumImprovement = 0.03;
};

enum class TimingVerdict : uint8_t {
  Improved,
  Stable,
  InvalidPolicy,
  InsufficientSamples,
  TooManySamples,
  InvalidTiming,
  InvalidOrder,
  UnderPressure,
  Noisy,
  Regressed,
  Uncertain,
};

[[nodiscard]] constexpr std::string_view timingVerdictName(TimingVerdict verdict) noexcept {
  switch (verdict) {
  case TimingVerdict::Improved: return "improved";
  case TimingVerdict::Stable: return "stable";
  case TimingVerdict::InvalidPolicy: return "invalid_policy";
  case TimingVerdict::InsufficientSamples: return "insufficient_samples";
  case TimingVerdict::TooManySamples: return "too_many_samples";
  case TimingVerdict::InvalidTiming: return "invalid_timing";
  case TimingVerdict::InvalidOrder: return "invalid_order";
  case TimingVerdict::UnderPressure: return "under_pressure";
  case TimingVerdict::Noisy: return "noisy";
  case TimingVerdict::Regressed: return "regressed";
  case TimingVerdict::Uncertain: return "uncertain";
  }
  return "unknown";
}

struct TimingAssessment final {
  TimingVerdict verdict = TimingVerdict::InsufficientSamples;
  double baselineMedianSeconds = 0;
  double candidateMedianSeconds = 0;
  double medianPairedGain = 0;
  double pairedGainSpread = 0;
  double baselineRelativeSpread = 0;
  double candidateRelativeSpread = 0;
  double conservativeGain = 0;

  [[nodiscard]] bool qualified() const noexcept {
    return verdict == TimingVerdict::Improved || verdict == TimingVerdict::Stable;
  }
};

[[nodiscard]] TimingAssessment evaluate(std::span<const PairedTiming> samples,
                                        const Policy &policy = {}) noexcept;

// One candidate's paired samples of the measured workload.
struct CandidateMeasurements final {
  CandidateId id;
  std::span<const PairedTiming> samples;
};

enum class SelectionVerdict : uint8_t { Baseline, Selected, InvalidInput };

struct Selection final {
  SelectionVerdict verdict = SelectionVerdict::Baseline;
  CandidateId candidate = kBaseline;
  double conservativeGain = 0;
};

// Duplicate candidate IDs, the baseline's ID or an invalid policy make the
// whole input invalid, which retains the shipped baseline. Only a candidate
// whose samples are Improved can win: its conservative gain meets
// minimumImprovement. The greatest conservative gain wins; ties prefer the
// smaller candidate ID, independently of input order.
// A winner still needs a whole-model A/B before any policy change.
[[nodiscard]] Selection
selectCandidate(std::span<const CandidateMeasurements> candidates,
                const Policy &policy = {}) noexcept;

} // namespace splash::ops::tuning
