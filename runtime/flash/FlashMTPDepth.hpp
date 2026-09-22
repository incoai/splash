#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace splash::flash {

inline constexpr const char *kFlashMTPDepthSemantics =
    "native-mtp-conditional-acceptance-measured-wall-cost-initial-confirm-bidirectional-probe-v2";

struct FlashMTPDepthConfig final {
  uint32_t maximumDepth = 3;
  double acceptanceAlpha = 0.08;
  // Two consecutive first-proposal failures indicate a possible content
  // change; decay that stale first-position estimate faster until a match.
  double repeatedRejectionAlpha = 0.5;
  double costHorizonMilliseconds = 400.0;
  double initialAcceptance = 0.6;
  // Used only before two different cycle depths have measured costs.
  double unmeasuredMarginalMilliseconds = 7.0;
  double switchGain = 1.03;
  double probePeriodMilliseconds = 1000.0;
  double stalePeriodMilliseconds = 5000.0;
  double probeDutyFraction = 0.15;
  double probeScoreMargin = 1.15;
  uint32_t probeCycles = 4;
  uint32_t firstRejectionShallowStreak = 2;
  uint32_t firstRejectionBaselineStreak = 4;
  // An unseeded speculative startup cost this much larger than another
  // measured speculative shape receives one immediate confirmation. Plain
  // depth0 is excluded because its verifier has a different shape boundary.
  double initialCostConfirmationRatio = 3.0;
};

struct FlashMTPDepthSnapshot final {
  uint32_t depth = 0;
  uint64_t cycles = 0;
  bool calibrated = false;
  bool probing = false;
  uint32_t probeCyclesRemaining = 0;
  uint32_t consecutiveFirstRejections = 0;
  std::array<double, 3> conditionalAcceptance{};
  std::array<std::optional<double>, 4> cycleMilliseconds{};
  std::array<double, 4> costAgeMilliseconds{};
};

// Pure CPU policy; it neither samples tokens nor changes acceptance/rollback.
// Score(d) = expected verified outputs / measured full-cycle wall time.
// Conditional acceptance must use the MATCHED prefix before EOS/quota clamps.
// Feed actual proposed depth after draft EOS, not the requested maximum.
class FlashMTPDepthController final {
public:
  explicit FlashMTPDepthController(FlashMTPDepthConfig config = {});
  [[nodiscard]] uint32_t select(uint32_t maximumAllowedDepth = 3) const noexcept;
  // Wall time includes committed head fold, chains, target verification,
  // partial rollback and feature copies. Set timeSample=false for terminal/
  // quota-limited or one-off maintenance cycles; acceptance still updates,
  // while the calibration/probe timing sweep does not advance. Cancelled or
  // unverified cycles have no acceptance observation and should be skipped.
  void observe(uint32_t actualDepth, uint32_t matchedDrafts,
               double cycleMilliseconds, bool timeSample = true);
  // Optional native cycle measurements from the SAME worker/route/context
  // regime. No acceptance is carried across requests. Positive finite slots
  // seed that depth; null slots remain unmeasured and are probed normally.
  void seedCosts(const std::array<std::optional<double>, 4> &costs);
  [[nodiscard]] FlashMTPDepthSnapshot snapshot() const noexcept;
  [[nodiscard]] double expectedOutputs(uint32_t depth) const;
  [[nodiscard]] double score(uint32_t depth) const;

private:
  [[nodiscard]] double estimatedCost(uint32_t depth) const noexcept;
  [[nodiscard]] double marginalCost() const noexcept;
  [[nodiscard]] uint32_t bestDepth() const noexcept;
  [[nodiscard]] std::optional<uint32_t> bestRival() const noexcept;
  [[nodiscard]] std::optional<uint32_t> mostStale() const noexcept;
  void updateCost(uint32_t depth, double milliseconds);
  void nextWarmup() noexcept;
  void confirmInitialCostIfNeeded() noexcept;

  FlashMTPDepthConfig config_;
  std::array<double, 3> acceptance_{};
  std::array<std::optional<double>, 4> costs_{};
  std::array<double, 4> ages_{};
  std::array<bool, 4> initialCostEligible_{};
  std::optional<uint32_t> initialCostConfirmation_;
  std::array<uint32_t, 6> warmup_{};
  uint32_t warmupCount_ = 0, warmupIndex_ = 0, depth_ = 0, probeLeft_ = 0;
  uint32_t firstRejections_ = 0;
  uint64_t cycles_ = 0;
  double sinceProbe_ = 0.0, sinceExploration_ = 0.0, recentCycleCost_ = 0.0;
};

} // namespace splash::flash
