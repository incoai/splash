// SPDX-License-Identifier: Apache-2.0
// Expected-output scoring, conditional acceptance and bidirectional probing
// follow oMLX's _DepthController. This native implementation has fixed-size
// CPU storage, explicit censoring, repeated-rejection backoff and clipped
// one-off timing spikes. It does not import MLX or issue GPU work.
#include "flash/FlashMTPDepth.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace splash::flash {
namespace {
bool positive(double value) { return std::isfinite(value) && value > 0.0; }
} // namespace

FlashMTPDepthController::FlashMTPDepthController(FlashMTPDepthConfig config)
    : config_(config), depth_(config.maximumDepth) {
  if (config.maximumDepth > 3 || !positive(config.acceptanceAlpha) ||
      config.acceptanceAlpha > 1.0 || !positive(config.costHorizonMilliseconds) ||
      !positive(config.repeatedRejectionAlpha) || config.repeatedRejectionAlpha > 1.0 ||
      !std::isfinite(config.initialAcceptance) || config.initialAcceptance < 0 ||
      config.initialAcceptance > 1 || !positive(config.unmeasuredMarginalMilliseconds) ||
      !positive(config.initialCostConfirmationRatio) || config.initialCostConfirmationRatio <= 1.0 ||
      !positive(config.switchGain) || config.switchGain < 1.0 ||
      !positive(config.probePeriodMilliseconds) || !positive(config.stalePeriodMilliseconds) ||
      !positive(config.probeDutyFraction) || config.probeDutyFraction > 1.0 ||
      !positive(config.probeScoreMargin) || config.probeScoreMargin < 1.0 ||
      !config.probeCycles || !config.firstRejectionShallowStreak ||
      config.firstRejectionBaselineStreak < config.firstRejectionShallowStreak)
    throw std::invalid_argument("invalid native MTP depth controller configuration");
  acceptance_.fill(config.initialAcceptance);
  for (uint32_t depth = config.maximumDepth; depth > 0; --depth) {
    initialCostEligible_[depth] = true;
    warmup_[warmupCount_++] = depth;
  }
  // The fastest of three real plain-cycle samples prevents a single warmed
  // shape/driver outlier from making the baseline look permanently expensive.
  for (uint32_t repetition = 0; repetition < 3; ++repetition)
    warmup_[warmupCount_++] = 0;
}

uint32_t FlashMTPDepthController::select(uint32_t maximumAllowedDepth) const noexcept {
  return std::min(depth_, std::min(config_.maximumDepth, maximumAllowedDepth));
}

void FlashMTPDepthController::seedCosts(
    const std::array<std::optional<double>, 4> &costs) {
  for (uint32_t depth = 0; depth <= config_.maximumDepth; ++depth)
    if (costs[depth] && !positive(*costs[depth]))
      throw std::invalid_argument("native MTP seeded costs must be positive finite milliseconds");
  for (uint32_t depth = 0; depth <= config_.maximumDepth; ++depth) {
    costs_[depth] = costs[depth];
    ages_[depth] = 0.0;
  }
  warmupIndex_ = 0;
  warmupCount_ = 0;
  initialCostConfirmation_.reset();
  initialCostEligible_.fill(false);
  for (uint32_t depth = config_.maximumDepth; depth > 0; --depth) {
    initialCostEligible_[depth] = !costs_[depth];
    if (!costs_[depth]) warmup_[warmupCount_++] = depth;
  }
  if (!costs_[0])
    for (uint32_t repetition = 0; repetition < 3; ++repetition)
      warmup_[warmupCount_++] = 0;
  // Full seeding eliminates the per-request calibration tax, while the
  // ordinary staleness/probe machinery still remeasures native route costs.
  bool complete = true;
  for (uint32_t depth = 0; depth <= config_.maximumDepth; ++depth)
    complete &= costs_[depth].has_value();
  if (complete) {
    warmupIndex_ = warmupCount_;
    depth_ = bestDepth();
  } else {
    nextWarmup();
  }
}

void FlashMTPDepthController::updateCost(uint32_t depth, double milliseconds) {
  auto &previous = costs_[depth];
  if (!previous) { previous = milliseconds; return; }
  if (warmupIndex_ < warmupCount_) {
    previous = std::min(*previous, milliseconds);
    return;
  }
  double alpha = -std::expm1(-milliseconds / config_.costHorizonMilliseconds);
  // Damp AND winsorize one-off contention. Damping alone still lets a 10x
  // pause inflate a dormant depth cost for many subsequent probe bursts.
  if (milliseconds > *previous * 2.0) {
    alpha *= 0.25;
    milliseconds = *previous * 2.0;
  }
  previous = *previous * (1.0 - alpha) + milliseconds * alpha;
}

double FlashMTPDepthController::marginalCost() const noexcept {
  uint32_t low = 4, high = 0;
  for (uint32_t depth = 0; depth <= config_.maximumDepth; ++depth) {
    if (!costs_[depth]) continue;
    if (low == 4) low = depth;
    high = depth;
  }
  if (low < high) {
    const double slope = (*costs_[high] - *costs_[low]) / (high - low);
    if (positive(slope)) return slope;
  }
  return config_.unmeasuredMarginalMilliseconds;
}

double FlashMTPDepthController::estimatedCost(uint32_t depth) const noexcept {
  if (costs_[depth]) return *costs_[depth];
  uint32_t nearest = 4;
  double minimum = std::numeric_limits<double>::infinity();
  for (uint32_t other = 0; other <= config_.maximumDepth; ++other) {
    if (!costs_[other]) continue;
    minimum = std::min(minimum, *costs_[other]);
    if (nearest == 4 ||
        std::abs(int(other) - int(depth)) < std::abs(int(nearest) - int(depth)))
      nearest = other;
  }
  if (nearest == 4) return 30.0 + config_.unmeasuredMarginalMilliseconds * depth;
  // A plain row lies below the multi-row verifier's kernel/rollback jump;
  // extrapolating the row slope can fabricate an impossibly cheap baseline.
  if (!depth) return minimum;
  return std::max(0.001, *costs_[nearest] +
      marginalCost() * (double(depth) - double(nearest)));
}

double FlashMTPDepthController::expectedOutputs(uint32_t depth) const {
  if (depth > config_.maximumDepth)
    throw std::invalid_argument("native MTP scoring depth exceeds configured maximum");
  double output = 1.0, prefix = 1.0;
  for (uint32_t position = 0; position < depth; ++position) {
    prefix *= acceptance_[position];
    output += prefix;
  }
  return output;
}

double FlashMTPDepthController::score(uint32_t depth) const {
  return expectedOutputs(depth) / estimatedCost(depth);
}

uint32_t FlashMTPDepthController::bestDepth() const noexcept {
  uint32_t best = depth_;
  double bestScore = -1.0;
  double currentScore = 0.0;
  for (uint32_t depth = 0; depth <= config_.maximumDepth; ++depth) {
    if (!depth && !costs_[0]) continue; // Never park on an invented baseline.
    double expected = 1.0, prefix = 1.0;
    for (uint32_t position = 0; position < depth; ++position) {
      prefix *= acceptance_[position]; expected += prefix;
    }
    const double value = expected / estimatedCost(depth);
    if (depth == depth_) currentScore = value;
    if (value > bestScore) { best = depth; bestScore = value; }
  }
  if (best != depth_ && bestScore < currentScore * config_.switchGain)
    return depth_;
  return best;
}

std::optional<uint32_t> FlashMTPDepthController::bestRival() const noexcept {
  std::optional<uint32_t> best;
  double bestScore = 0.0;
  for (uint32_t depth = 0; depth <= config_.maximumDepth; ++depth) {
    if (depth == depth_) continue;
    const double value = score(depth);
    if (value > bestScore) { best = depth; bestScore = value; }
  }
  if (best && bestScore >= score(depth_) / config_.probeScoreMargin) return best;
  return {};
}

std::optional<uint32_t> FlashMTPDepthController::mostStale() const noexcept {
  std::optional<uint32_t> best;
  double oldest = -1.0;
  // Prefer speculative depths on an unmeasured tie, then discover baseline.
  for (uint32_t offset = 0; offset <= config_.maximumDepth; ++offset) {
    const uint32_t depth = offset == config_.maximumDepth ? 0 : offset + 1;
    if (depth == depth_) continue;
    const double age = costs_[depth] ? ages_[depth] : std::numeric_limits<double>::infinity();
    if (age > oldest) { best = depth; oldest = age; }
  }
  return best;
}

void FlashMTPDepthController::nextWarmup() noexcept {
  while (warmupIndex_ < warmupCount_) {
    const uint32_t candidate = warmup_[warmupIndex_];
    // Seeded positive-depth costs need no startup exploration. Keep all three
    // baseline repetitions if baseline itself has not already been seeded.
    if (candidate && costs_[candidate]) { ++warmupIndex_; continue; }
    depth_ = candidate;
    return;
  }
  depth_ = bestDepth();
  confirmInitialCostIfNeeded();
}

void FlashMTPDepthController::confirmInitialCostIfNeeded() noexcept {
  double largestRatio = config_.initialCostConfirmationRatio;
  std::optional<uint32_t> candidate;
  for (uint32_t depth = 1; depth <= config_.maximumDepth; ++depth) {
    if (!initialCostEligible_[depth] || !costs_[depth]) continue;
    double comparison = std::numeric_limits<double>::infinity();
    for (uint32_t other = 1; other <= config_.maximumDepth; ++other)
      if (other != depth && costs_[other]) comparison = std::min(comparison, *costs_[other]);
    const double ratio = *costs_[depth] / comparison;
    if (ratio > largestRatio) { largestRatio = ratio; candidate = depth; }
  }
  if (!candidate) return;
  initialCostEligible_[*candidate] = false; // Exactly one startup confirmation.
  initialCostConfirmation_ = candidate;
  depth_ = *candidate;
}

void FlashMTPDepthController::observe(uint32_t actualDepth, uint32_t matchedDrafts,
    double milliseconds, bool timeSample) {
  if (actualDepth > config_.maximumDepth || matchedDrafts > actualDepth ||
      !std::isfinite(milliseconds) || milliseconds < 0.0 ||
      (timeSample && !positive(milliseconds)))
    throw std::invalid_argument("invalid native MTP depth observation");
  ++cycles_;
  if (actualDepth) {
    if (!matchedDrafts) {
      if (firstRejections_ < std::numeric_limits<uint32_t>::max()) ++firstRejections_;
    } else firstRejections_ = 0;
  } else firstRejections_ = 0;
  for (uint32_t position = 0; position < actualDepth; ++position) {
    const double hit = position < matchedDrafts ? 1.0 : 0.0;
    const double alpha = position == 0 && !matchedDrafts &&
        firstRejections_ >= config_.firstRejectionShallowStreak
        ? std::max(config_.acceptanceAlpha, config_.repeatedRejectionAlpha)
        : config_.acceptanceAlpha;
    acceptance_[position] = (1.0 - alpha) * acceptance_[position] + alpha * hit;
    if (position >= matchedDrafts) break; // Censored suffix is not rejection.
  }
  const bool confirmedInitialCost = timeSample && initialCostConfirmation_ &&
      actualDepth == *initialCostConfirmation_;
  if (timeSample) {
    if (confirmedInitialCost) {
      costs_[actualDepth] = std::min(*costs_[actualDepth], milliseconds);
      initialCostConfirmation_.reset();
    } else updateCost(actualDepth, milliseconds);
    recentCycleCost_ = milliseconds;
  }
  for (uint32_t depth = 0; depth <= config_.maximumDepth; ++depth)
    if (costs_[depth]) ages_[depth] += milliseconds;
  if (timeSample) ages_[actualDepth] = 0.0;
  sinceProbe_ += milliseconds;
  sinceExploration_ += milliseconds;
  if (!timeSample) return;

  if (firstRejections_ >= config_.firstRejectionBaselineStreak) {
    depth_ = 0; probeLeft_ = 0; sinceProbe_ = 0.0;
    return; // A real depth0 cycle will measure the escape hatch.
  }
  if (firstRejections_ >= config_.firstRejectionShallowStreak && actualDepth) {
    depth_ = 1; probeLeft_ = 0; sinceProbe_ = 0.0;
    return;
  }
  if (warmupIndex_ < warmupCount_) {
    if (actualDepth == warmup_[warmupIndex_]) ++warmupIndex_;
    nextWarmup();
    if (warmupIndex_ == warmupCount_) sinceProbe_ = 0.0;
    return;
  }
  if (confirmedInitialCost) {
    depth_ = bestDepth();
    confirmInitialCostIfNeeded();
    return;
  }
  if (initialCostConfirmation_) {
    depth_ = *initialCostConfirmation_;
    return;
  }
  if (probeLeft_) {
    if (--probeLeft_ == 0) { depth_ = bestDepth(); sinceProbe_ = 0.0; }
    return;
  }
  depth_ = bestDepth();
  if (!config_.maximumDepth) return;
  const double period = std::max(config_.probePeriodMilliseconds,
      config_.probeCycles * recentCycleCost_ / config_.probeDutyFraction);
  if (sinceProbe_ < period) return;
  const bool exploring = sinceExploration_ >= std::max(config_.stalePeriodMilliseconds, period * 2.0);
  const auto candidate = exploring ? mostStale() : bestRival();
  if (candidate) {
    depth_ = *candidate;
    probeLeft_ = config_.probeCycles;
    sinceProbe_ = 0.0;
    if (exploring) sinceExploration_ = 0.0;
  }
}

FlashMTPDepthSnapshot FlashMTPDepthController::snapshot() const noexcept {
  return {depth_, cycles_, warmupIndex_ == warmupCount_ && !initialCostConfirmation_, probeLeft_ != 0,
      probeLeft_, firstRejections_, acceptance_, costs_, ages_};
}

} // namespace splash::flash
