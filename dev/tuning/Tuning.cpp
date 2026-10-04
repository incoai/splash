#include "tuning/Tuning.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace splash::ops::tuning {
namespace {

bool validPolicy(const Policy &policy) noexcept {
  return policy.minimumPairs >= kMinPairedSamples &&
         policy.minimumPairs <= kMaxPairedSamples &&
         std::isfinite(policy.maximumRelativeTimingSpread) &&
         policy.maximumRelativeTimingSpread >= 0 &&
         policy.maximumRelativeTimingSpread < 1 &&
         std::isfinite(policy.maximumPairedGainSpread) &&
         policy.maximumPairedGainSpread >= 0 &&
         policy.maximumPairedGainSpread < 1 &&
         std::isfinite(policy.minimumImprovement) &&
         policy.minimumImprovement > 0 && policy.minimumImprovement < 1;
}

struct Distribution final {
  double median;
  double spread;
};

Distribution summarize(std::array<double, kMaxPairedSamples> &values,
                       size_t count) noexcept {
  std::sort(values.begin(), values.begin() + count);
  const double lowMiddle = values[(count - 1) / 2];
  const double highMiddle = values[count / 2];
  // Round the endpoints outward so small runs retain at least the central 80%.
  const size_t low = (count - 1) / 10;
  const size_t high = (9 * (count - 1) + 9) / 10;
  return {lowMiddle + (highMiddle - lowMiddle) * 0.5,
          values[high] - values[low]};
}

} // namespace

TimingAssessment evaluate(std::span<const PairedTiming> samples,
                          const Policy &policy) noexcept {
  TimingAssessment result;
  if (!validPolicy(policy)) {
    result.verdict = TimingVerdict::InvalidPolicy;
    return result;
  }
  if (samples.size() < policy.minimumPairs)
    return result;
  if (samples.size() > kMaxPairedSamples) {
    result.verdict = TimingVerdict::TooManySamples;
    return result;
  }

  std::array<double, kMaxPairedSamples> baseline{}, candidate{}, gains{};
  for (size_t i = 0; i < samples.size(); ++i) {
    const auto &sample = samples[i];
    if (sample.underPressure) {
      result.verdict = TimingVerdict::UnderPressure;
      return result;
    }
    if ((sample.first != MeasurementOrder::BaselineFirst &&
         sample.first != MeasurementOrder::CandidateFirst) ||
        (i != 0 && sample.first == samples[i - 1].first)) {
      result.verdict = TimingVerdict::InvalidOrder;
      return result;
    }
    if (!std::isfinite(sample.baselineSeconds) || sample.baselineSeconds <= 0 ||
        !std::isfinite(sample.candidateSeconds) || sample.candidateSeconds <= 0) {
      result.verdict = TimingVerdict::InvalidTiming;
      return result;
    }
    baseline[i] = sample.baselineSeconds;
    candidate[i] = sample.candidateSeconds;
    gains[i] = 1 - sample.candidateSeconds / sample.baselineSeconds;
    if (!std::isfinite(gains[i])) {
      result.verdict = TimingVerdict::InvalidTiming;
      return result;
    }
  }

  const auto baselineDistribution = summarize(baseline, samples.size());
  const auto candidateDistribution = summarize(candidate, samples.size());
  const auto gainDistribution = summarize(gains, samples.size());
  result.baselineMedianSeconds = baselineDistribution.median;
  result.candidateMedianSeconds = candidateDistribution.median;
  result.medianPairedGain = gainDistribution.median;
  result.pairedGainSpread = gainDistribution.spread;
  result.baselineRelativeSpread =
      baselineDistribution.spread / baselineDistribution.median;
  result.candidateRelativeSpread =
      candidateDistribution.spread / candidateDistribution.median;
  result.conservativeGain = gainDistribution.median - gainDistribution.spread;

  if (!std::isfinite(result.baselineRelativeSpread) ||
      !std::isfinite(result.candidateRelativeSpread) ||
      result.baselineRelativeSpread > policy.maximumRelativeTimingSpread ||
      result.candidateRelativeSpread > policy.maximumRelativeTimingSpread ||
      result.pairedGainSpread > policy.maximumPairedGainSpread) {
    result.verdict = TimingVerdict::Noisy;
  } else if (result.medianPairedGain < 0) {
    result.verdict = TimingVerdict::Regressed;
  } else if (result.conservativeGain < 0) {
    result.verdict = TimingVerdict::Uncertain;
  } else {
    result.verdict = result.conservativeGain >= policy.minimumImprovement
                         ? TimingVerdict::Improved
                         : TimingVerdict::Stable;
  }
  return result;
}

Selection selectCandidate(std::span<const CandidateMeasurements> candidates,
                          const Policy &policy) noexcept {
  Selection result;
  if (!validPolicy(policy)) {
    result.verdict = SelectionVerdict::InvalidInput;
    return result;
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].id == kBaseline) {
      result.verdict = SelectionVerdict::InvalidInput;
      return result;
    }
    for (size_t j = 0; j < i; ++j) {
      if (candidates[i].id == candidates[j].id) {
        result.verdict = SelectionVerdict::InvalidInput;
        return result;
      }
    }
  }

  for (const auto &candidate : candidates) {
    const auto assessment = evaluate(candidate.samples, policy);
    if (assessment.verdict != TimingVerdict::Improved)
      continue;
    const double gain = assessment.conservativeGain;
    if (result.verdict == SelectionVerdict::Baseline ||
        gain > result.conservativeGain ||
        (gain == result.conservativeGain &&
         candidate.id.value < result.candidate.value))
      result = {SelectionVerdict::Selected, candidate.id, gain};
  }
  return result;
}

} // namespace splash::ops::tuning
