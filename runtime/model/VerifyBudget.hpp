#pragma once

// The verify rows a MoE target's decode cycles choose
// (ops::DraftSelector::addVerifyRows): the most expected accepted tokens per
// cost, learned online. A draft row's own chance, that the target selects its
// token where it accepted the row's parent, is logit(p) = slope * logit(q) +
// intercepts[position] for its draft probability q at its proposal position,
// and a row's chance its own times its parent's (DraftTreeParams): greedy
// lanes fit theirs to the last kLabels labels of their trees' rows, sampled
// lanes, whose drawn children their verify rows' rejection sampling accepts,
// apart. A cycle of R rows
// costs a + b * R ms, fit per lane count, constrained or not, to its latest
// cycles.
// A cycle counts at most twice the larger of its fitted price and the mean:
// capped at the price alone, one 2x stall over clustered rows had 45 of the
// next 64 cycles verify their anchors alone.
#include "metal/abi/ExecutionGeometry.h"
#include "ops/LaneRows.hpp"

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
  // Qwen3.6-35B-A3B: slopes 0.65-0.71). It holds for trees' rows: on greedy
  // traced cycles of Qwen3.8-27B and Qwen3.6-35B-A3B, a tree of 16 rows has
  // its rows' own chances within 0.011 and 0.016 of the rate the target
  // accepts them at (over tenths of the chance), where a fit of those rows
  // pooled over both targets gives 0.016 and 0.008.
  static constexpr Acceptance kPrior{0.7F, {0.0F, -0.35F, -0.8F, -0.75F, -0.75F, -0.85F, -0.45F}};

  // A label: a draft row at proposal position `position` (its depth less
  // one), whose draft probability had logit `feature`, under a parent its
  // lane's walk accepted: whether the target selected the row's token there.
  struct Label final {
    uint32_t position = 0;
    float feature = 0.0F;
    bool accepted = false;
    bool operator==(const Label &) const = default;
  };

  // A lane of a completed cycle: the rows it verified, whether it samples,
  // and its tree's labels (treeLabels).
  struct Lane final {
    uint32_t verifyRows = 0;
    bool samples = false;
    std::vector<Label> labels;
  };

  // The labels of a lane's tree, its selector's rows in the order it took
  // them (ops::DraftSelectorBuffers: each row's token, its tree row and its
  // scores, pairs of its draft probability and its estimate), after a walk
  // whose `selections`, the target's, are those of its path's rows in order
  // from its anchor's: each row whose parent is one of those rows, labeled
  // with whether the parent's selection is its token, a row past the lane's
  // verify rows too, but none whose draft probability is not one (a host
  // chain's row no draft step joined, a forced chain's row, or a draft row
  // that is not finite).
  [[nodiscard]] static std::vector<Label> treeLabels(std::span<const uint32_t> tokens,
                                                     std::span<const ops::TreeRow> rows,
                                                     std::span<const float> scores,
                                                     std::span<const uint32_t> selections);

  // The acceptance model of the lanes that sample, or of the greedy ones.
  [[nodiscard]] const Acceptance &acceptance(bool samples) const noexcept { return acceptance_[samples]; }
  // costs[R] for `lanes` lanes that verify R rows in all, R from `lanes` to
  // eight rows a lane, nondecreasing in R; zero elsewhere.
  [[nodiscard]] std::array<float, kBlockRows * kLanes + 1> costs(uint32_t lanes, bool constrained) const;
  // Whether the next cycle of `lanes` lanes verifies every row.
  [[nodiscard]] bool explores(uint32_t lanes, bool constrained) const;
  // A completed cycle and its commands' wall time.
  void observe(std::span<const Lane> lanes, bool constrained, double milliseconds);

  // The labels kept of the lanes that sample, or of the greedy ones, oldest
  // first.
  [[nodiscard]] std::vector<Label> labels(bool samples) const;
  // The maximum a posteriori model of `labels` under the prior, by damped
  // Newton steps from `start`.
  [[nodiscard]] static Acceptance fit(std::span<const Label> labels, const Acceptance &start);
  // The shape of a cycle's cost by its rows (a cycle of every row costs 1).
  [[nodiscard]] static double seed(uint32_t lanes, uint32_t rows);

  // A lane's chances for a cycle's choice of rows (stepRows): the estimates
  // that the target accepts each row its tree would add past its anchor,
  // best first, at most a verify block's rows past it. They are those its
  // last cycle's tree took (ops::DraftSelectorBuffers::treeScores), every
  // row's 1 for a lane without one, merged with this cycle's lookup chain's,
  // each the product of its tokens' rates to its depth (LookupAcceptance,
  // whose agreement with the draft the host does not know).
  [[nodiscard]] static std::vector<float> laneChances(std::span<const float> tree, std::span<const float> lookup);
  // The rows of a cycle of curves.size() lanes (ops::StepRows): the T, from
  // the lane count to a verify block's rows a lane and to the costs' end,
  // with the most expected accepted tokens per costs[T], the fewest rows
  // among equal ones. A cycle of T rows verifies every lane's anchor and the
  // T - lanes largest chances over its lanes' curves, none past a lane's
  // ops::verifyLaneCap(T, lanes) rows, as the GPU's choice of its rows does
  // (ops::DraftSelector::addVerifyRows) with this cycle's chances.
  [[nodiscard]] static ops::StepRows stepRows(std::span<const std::vector<float>> curves,
                                              std::span<const float> costs);

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

  // The greedy lanes', then the sampled ones'.
  std::array<Acceptance, 2> acceptance_{kPrior, kPrior};
  std::array<std::vector<Label>, 2> labels_;
  std::array<size_t, 2> next_{};
  uint64_t cycles_ = 0;
  std::array<std::array<Cost, kLanes>, 2> tables_{};
  std::array<std::array<uint64_t, kLanes>, 2> tableCycles_{};
  // The cycles counted at under half their time since the last notice of
  // them, and the awake time (ms) before which no other notice is given.
  uint64_t cuts_ = 0;
  double nextNotice_ = 0.0;
};

} // namespace splash::model
