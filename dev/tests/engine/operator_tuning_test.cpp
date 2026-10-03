#include "TestChecks.hpp"
#include "tuning/Tuning.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace splash::ops::tuning;

using splash::test::require;

std::vector<PairedTiming> timings(double gain, size_t count = 12) {
  std::vector<PairedTiming> result;
  for (size_t i = 0; i < count; ++i)
    result.push_back({1, 1 - gain, measurementOrder(i), false});
  return result;
}

void validSamples() {
  auto samples = timings(0.125);
  auto assessment = evaluate(samples);
  require(assessment.verdict == TimingVerdict::Improved &&
              assessment.baselineMedianSeconds == 1 &&
              assessment.candidateMedianSeconds == 0.875 &&
              assessment.medianPairedGain == 0.125 &&
              assessment.pairedGainSpread == 0 &&
              assessment.conservativeGain == 0.125,
          "stable improvement was not measured from matched pairs");
  require(evaluate(timings(0)).verdict == TimingVerdict::Stable,
          "unchanged workload did not qualify");
  require(evaluate(timings(0.01)).verdict == TimingVerdict::Stable,
          "tiny improvement was treated as meaningful");
  require(evaluate(timings(-0.01)).verdict == TimingVerdict::Regressed,
          "stable regression was accepted");
  require(evaluate(timings(0.125, 64)).qualified(),
          "maximum bounded sample count was rejected");
  require(evaluate(timings(0.125, 13)).qualified(),
          "odd sample count was rejected");
  Policy exactThreshold;
  exactThreshold.minimumImprovement = 0.125;
  require(evaluate(samples, exactThreshold).verdict == TimingVerdict::Improved,
          "gain equal to the improvement margin was rejected");
  exactThreshold.minimumPairs = 24;
  require(evaluate(samples, exactThreshold).verdict ==
              TimingVerdict::InsufficientSamples,
          "increased minimum pair count was ignored");
  for (auto &sample : samples)
    sample.first = sample.first == MeasurementOrder::BaselineFirst
                       ? MeasurementOrder::CandidateFirst
                       : MeasurementOrder::BaselineFirst;
  require(evaluate(samples).qualified(), "candidate-first run was rejected");

  // A single isolated interruption is excluded by the central spread, but
  // remains in the input for explicit validity/pressure/order checks.
  samples[0].baselineSeconds = 100;
  samples[0].candidateSeconds = 87.5;
  require(evaluate(samples).qualified(), "isolated timing outlier was not robust");
  samples[0].underPressure = true;
  require(evaluate(samples).verdict == TimingVerdict::UnderPressure,
          "outlier filtering concealed pressure");
}

void invalidSamplesAndPolicies() {
  require(evaluate({}).verdict == TimingVerdict::InsufficientSamples,
          "empty samples were accepted");
  require(evaluate(timings(0.125, 11)).verdict ==
              TimingVerdict::InsufficientSamples,
          "insufficient paired samples were accepted");
  require(evaluate(timings(0.125, 65)).verdict == TimingVerdict::TooManySamples,
          "unbounded sample count was accepted");
  for (const double invalid : {0.0, -1.0,
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
    auto samples = timings(0.125);
    samples[0].baselineSeconds = invalid;
    require(evaluate(samples).verdict == TimingVerdict::InvalidTiming,
            "invalid baseline duration was accepted");
    samples = timings(0.125);
    samples[0].candidateSeconds = invalid;
    require(evaluate(samples).verdict == TimingVerdict::InvalidTiming,
            "invalid candidate duration was accepted");
  }
  auto samples = timings(0.125);
  samples[0].baselineSeconds = std::numeric_limits<double>::denorm_min();
  samples[0].candidateSeconds = std::numeric_limits<double>::max();
  require(evaluate(samples).verdict == TimingVerdict::InvalidTiming,
          "overflowed paired ratio was accepted");
  samples = timings(0.125);
  samples[1].first = samples[0].first;
  require(evaluate(samples).verdict == TimingVerdict::InvalidOrder,
          "unbalanced measurement order was accepted");
  samples[0].first = static_cast<MeasurementOrder>(2);
  require(evaluate(samples).verdict == TimingVerdict::InvalidOrder,
          "invalid measurement order was accepted");

  samples = timings(0.125);
  std::array<Policy, 10> policies{};
  policies[0].minimumPairs = 11;
  policies[1].minimumPairs = 65;
  policies[2].maximumRelativeTimingSpread = -1;
  policies[3].maximumRelativeTimingSpread = 1;
  policies[4].maximumRelativeTimingSpread =
      std::numeric_limits<double>::quiet_NaN();
  policies[5].maximumPairedGainSpread = -1;
  policies[6].maximumPairedGainSpread = std::numeric_limits<double>::infinity();
  policies[7].minimumImprovement = 0;
  policies[8].minimumImprovement = 1;
  policies[9].minimumImprovement = std::numeric_limits<double>::quiet_NaN();
  for (const auto &policy : policies)
    require(evaluate(samples, policy).verdict == TimingVerdict::InvalidPolicy,
            "invalid policy was accepted");
}

void matchedNoise() {
  auto samples = timings(0.05);
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i].baselineSeconds = i % 2 == 0 ? 100 : 110;
    samples[i].candidateSeconds = samples[i].baselineSeconds * 0.95;
  }
  require(evaluate(samples).verdict == TimingVerdict::Improved,
          "matched common timing movement hid a stable improvement");
  // Identical independent medians, but matching the wrong samples introduces
  // alternating regressions. Comparing separate medians would miss this.
  for (size_t i = 0; i < samples.size(); i += 2)
    std::swap(samples[i].candidateSeconds, samples[i + 1].candidateSeconds);
  const auto shuffled = evaluate(samples);
  require(shuffled.verdict == TimingVerdict::Noisy &&
              std::abs(shuffled.candidateMedianSeconds /
                           shuffled.baselineMedianSeconds -
                       0.95) < 1e-12,
          "unpaired medians concealed noisy paired gains");

  samples = timings(0.125);
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i].baselineSeconds = i % 2 == 0 ? 1 : 2;
    samples[i].candidateSeconds = samples[i].baselineSeconds * 0.875;
  }
  require(evaluate(samples).verdict == TimingVerdict::Noisy,
          "large common timing movement was accepted");
  samples = timings(0.02);
  for (size_t i = 0; i < 4; ++i)
    samples[i].candidateSeconds = 1.01;
  require(evaluate(samples).verdict == TimingVerdict::Uncertain,
          "positive median concealed a conservative regression");

  // Arithmetic on extreme but finite timings must fail closed.
  samples = timings(0.125);
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i].baselineSeconds = i < 9 ? 1e-300 : 1e300;
    samples[i].candidateSeconds = samples[i].baselineSeconds * 0.875;
  }
  require(evaluate(samples).verdict == TimingVerdict::Noisy,
          "overflowed relative spread was accepted");
}

void selection() {
  const auto fast = timings(0.125);
  const auto medium = timings(0.0625);
  const auto small = timings(0.01);
  const auto slow = timings(-0.01);
  const auto shortRun = timings(0.125, 11);
  std::array candidates{CandidateMeasurements{{1}, medium},
                        CandidateMeasurements{{3}, fast},
                        CandidateMeasurements{{2}, fast},
                        CandidateMeasurements{{4}, slow},
                        CandidateMeasurements{{5}, small},
                        CandidateMeasurements{{6}, shortRun}};
  auto chosen = selectCandidate(candidates);
  require(chosen.verdict == SelectionVerdict::Selected &&
              chosen.candidate == CandidateId{2} &&
              chosen.conservativeGain == 0.125,
          "selection did not rank conservative gain, then ID");
  std::reverse(candidates.begin(), candidates.end());
  require(selectCandidate(candidates).candidate == chosen.candidate,
          "selection changed with candidate order");
  require(selectCandidate(std::span(candidates).last(1)).candidate ==
              CandidateId{1},
          "a smaller meaningful gain did not replace baseline");
  require(selectCandidate(std::span(candidates).first(3)).candidate ==
              kBaseline,
          "regression, small gain or missing samples replaced baseline");
  require(selectCandidate({}).candidate == kBaseline,
          "empty candidate list did not retain baseline");
  auto pressured = fast;
  pressured.back().underPressure = true;
  auto noisy = fast;
  for (size_t i = 0; i < noisy.size(); i += 2)
    noisy[i].candidateSeconds = 1.125;
  const auto neutral = timings(0);
  const std::array unsafe{CandidateMeasurements{{1}, pressured},
                          CandidateMeasurements{{2}, noisy},
                          CandidateMeasurements{{3}, neutral}};
  require(selectCandidate(unsafe).candidate == kBaseline,
          "pressured, noisy or neutral samples replaced baseline");
  Policy margin;
  margin.minimumImprovement = 0.07;
  require(selectCandidate(std::span(candidates).last(1), margin).candidate ==
              kBaseline,
          "a gain below the improvement margin replaced baseline");

  const std::array ambiguous{CandidateMeasurements{{1}, fast},
                             CandidateMeasurements{{1}, medium}};
  const std::array baselineId{CandidateMeasurements{kBaseline, fast}};
  require(selectCandidate(ambiguous).verdict == SelectionVerdict::InvalidInput &&
              selectCandidate(baselineId).verdict ==
                  SelectionVerdict::InvalidInput,
          "ambiguous selection IDs were accepted");
  Policy invalidPolicy;
  invalidPolicy.minimumPairs = 1;
  require(selectCandidate(candidates, invalidPolicy).verdict ==
              SelectionVerdict::InvalidInput,
          "selection ignored an invalid policy");
}

} // namespace

int main() {
  try {
    validSamples();
    invalidSamplesAndPolicies();
    matchedNoise();
    selection();
    std::cout << "operator tuning: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "operator tuning: FAIL: " << error.what() << '\n';
    return 1;
  }
}
