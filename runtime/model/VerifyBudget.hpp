#pragma once

// What a MoE target's decode cycles choose their verify rows by
// (ops::DraftSelector::addVerifyRows), learned online on the engine thread:
// an acceptance model of the draft's confidences and a table of what a cycle
// costs by its rows.
//
// The acceptance model: once the proposals before it are accepted, a
// proposal is accepted with probability p, where logit(p) = slope *
// logit(confidence) + intercepts[position], the confidence being the largest
// candidate probability the selector saw at its position. Each completed
// cycle labels the proposals it verified, up to the first the target
// rejected, but for those whose confidence is not a probability (a draft row
// that is not finite); the model is refit every kRefitCycles cycles over the
// last kLabels labels by damped Newton steps (IRLS) on their log-likelihood
// with a Gaussian prior around kPrior, which also stands before any label.
//
// The costs: a cycle of L lanes that verify R rows in all takes about
// a + b * R milliseconds, fit to the wall times of the latest cycles of L
// lanes (which for streamed experts include the store's reads of its
// misses) by least squares that weigh each cycle kDecay times the next. The
// latest cycles stand for the requests the batch serves now: what a row costs
// varies with them (a Qwen3.6-35B-A3B cycle with 4 GB of experts adds about
// 3 ms a row on prose and 8 ms on reasoning), so the cycles of other requests
// must not price it. The slope shrinks toward the seed's, the shape the
// measured MoE targets share, at the cycles' level, as much as
// kPriorCycles cycles over a lane count's spread of rows would; before any
// cycle, the seed itself is the cost. A cycle counts at most twice and at
// least half the time the fit gives it, so that a stall does not price the
// rows of the next thirty cycles. Constrained cycles, whose verify waits
// for the host's masks, are fit apart. Every kExploreCycles-th cycle of a
// lane count verifies every row, which keeps the rows of the fit spread.
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
  static constexpr uint32_t kExploreCycles = 32;
  static constexpr double kDecay = 0.97;
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
};

} // namespace splash::model
