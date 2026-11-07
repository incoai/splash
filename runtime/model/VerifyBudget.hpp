#pragma once

// The verify rows a MoE target's decode cycles choose
// (ops::DraftSelector::addVerifyRows): the most expected accepted tokens per
// cost, learned online. Acceptance is logit(p) = slope * logit(confidence) +
// intercepts[position] over the last kLabels labels; a cycle of R rows costs
// a + b * R ms, fit per lane count, constrained or not, to its latest cycles.
// A cycle counts at most twice the larger of its fitted price and the mean:
// capped at the price alone, one 2x stall over clustered rows had 45 of the
// next 64 cycles verify their anchors alone.
#include "metal/abi/ExecutionGeometry.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace splash::model {

class VerifyBudget final {
public:
  static constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  static constexpr uint32_t kBlockRows = SPLASH_DRAFT_QUERY_ROWS;
  static constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  static constexpr uint32_t kRefitCycles = 64;
  static constexpr uint32_t kLabels = 4096;
  // Every kExploreCycles-th cycle of a lane count verifies every row, which
  // keeps the rows of the cost fit spread.
  static constexpr uint32_t kExploreCycles = 32;
  // Each cycle weighs kDecay times the next (about the last 33 cycles), as
  // what a row costs follows the requests served.
  static constexpr double kDecay = 0.97;
  // The slope shrinks toward the seed's as much as kPriorCycles spread
  // cycles would pull it.
  static constexpr double kPriorCycles = 5.0;
  // The prior's precision on the slope and on each intercept: about twenty
  // labels' worth each (a label tells the slope about its logit squared, some
  // 2.5, times what it tells its intercept).
  static constexpr double kSlopePrecision = 50.0;
  static constexpr double kInterceptPrecision = 4.0;
  [[nodiscard]] static constexpr double priorPrecision(uint32_t parameter) noexcept {
    return parameter ? kInterceptPrecision : kSlopePrecision;
  }

  struct Acceptance final {
    float slope = 0.0F;
    std::array<float, kPositions> intercepts{};
    bool operator==(const Acceptance &) const = default;
  };
  // The fit of the measured targets' logged cycles (Qwen3.8-Flash-Next,
  // Qwen3.6-35B-A3B: slopes 0.65-0.71).
  static constexpr Acceptance kPrior{0.7F, {0.0F, -0.35F, -0.8F, -0.75F, -0.75F, -0.85F, -0.45F}};

  // A label: a proposal at `position`, whose confidence had logit `feature`,
  // that the target accepted or rejected.
  struct Label final {
    uint32_t position = 0;
    float feature = 0.0F;
    bool accepted = false;
  };

  // A lane of a completed cycle: its confidences, the rows it verified, the
  // proposals the target accepted before its first rejection (over all
  // seven: those past its verify rows are not labels), and whether its
  // proposals were the draft's (a thinking end replaces some).
  struct Lane final {
    std::array<float, kPositions> confidence{};
    uint32_t verifyRows = 0;
    uint32_t accepted = 0;
    bool labels = true;
  };

  [[nodiscard]] const Acceptance &acceptance() const noexcept { return acceptance_; }
  // costs[R] for `lanes` lanes that verify R rows in all, R from `lanes` to
  // eight rows a lane, nondecreasing in R; zero elsewhere.
  [[nodiscard]] std::array<float, kBlockRows * kLanes + 1> costs(uint32_t lanes, bool constrained) const;
  // Whether the next cycle of `lanes` lanes verifies every row.
  [[nodiscard]] bool explores(uint32_t lanes, bool constrained) const;
  // A completed cycle and its commands' wall time.
  void observe(std::span<const Lane> lanes, bool constrained, double milliseconds);

  // The labels kept, oldest first.
  [[nodiscard]] std::vector<Label> labels() const;
  // The maximum a posteriori model of `labels` under the prior, by damped
  // Newton steps from `start`.
  [[nodiscard]] static Acceptance fit(std::span<const Label> labels, const Acceptance &start);
  // The shape of a cycle's cost by its rows (a cycle of every row costs 1).
  [[nodiscard]] static double seed(uint32_t lanes, uint32_t rows);

private:
  // The weighted sums of the latest cycles' rows R and times t.
  struct Cost final {
    double weight = 0.0;
    double rows = 0.0;
    double rowsSquared = 0.0;
    double time = 0.0;
    double rowsTime = 0.0;
  };

  [[nodiscard]] const Cost &table(uint32_t lanes, bool constrained) const;
  // The fit's intercept and slope; none before any cycle.
  [[nodiscard]] std::optional<std::pair<double, double>> line(uint32_t lanes, bool constrained) const;

  Acceptance acceptance_ = kPrior;
  std::vector<Label> labels_;
  size_t next_ = 0;
  uint64_t cycles_ = 0;
  std::array<std::array<Cost, kLanes>, 2> tables_{};
  std::array<std::array<uint64_t, kLanes>, 2> tableCycles_{};
  // The cycles counted at under half their time since the last notice of
  // them, and the awake time (ms) before which no other notice is given.
  uint64_t cuts_ = 0;
  double nextNotice_ = 0.0;
};

} // namespace splash::model
