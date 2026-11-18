#pragma once

// The online estimates a decode cycle's rows take: each lane's chances that
// the target accepts its tree's rows, which a cycle's choice of its step's
// rows weighs (StepCosts) and its GPU's layout of the step's best rows reads
// (ops::DraftSelector::addVerifyRows).
// A draft row's own chance, that the target selects its token where it
// accepted the row's parent, is logit(p) = slope * logit(q) +
// intercepts[position] for its draft probability q at its proposal position,
// at most its cap, what its parent's host chain child and earlier draws left
// of 1, and a row's chance its own times its parent's (DraftTreeParams):
// greedy lanes fit theirs to the last kLabels labels of their trees' rows,
// sampled lanes, whose drawn children their verify rows' rejection sampling
// accepts, apart, each under a prior (kPrior, or a draft's own).
#include "metal/abi/ExecutionGeometry.h"
#include "ops/LaneRows.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

class VerifyBudget final {
public:
  static constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  // A lane's most rows, a verify block's (ops::verifyLaneCap).
  static constexpr uint32_t kBlockRows = SPLASH_TARGET_VERIFY_ROWS;
  static constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  static constexpr uint32_t kRefitCycles = 64;
  static constexpr uint32_t kLabels = 4096;
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
  // DeepSeek-V4.1-Flash's DSpark, whose confidence is the sigmoid of its
  // confidence head's score: the fit of 1371 labels of its cycles on a 48 GB
  // M4 Pro (prose, chat and arithmetic), its five proposals' positions;
  // the two past its block keep kPrior's, which no label reaches.
  static constexpr Acceptance kDSparkPrior{0.82F, {0.0F, 0.29F, 0.4F, 0.2F, -0.16F, -0.85F, -0.45F}};

  // Cycles whose acceptance model starts from `prior`, and is held to it.
  explicit VerifyBudget(const Acceptance &prior = kPrior) : prior_(prior), acceptance_{prior, prior} {}

  // A label: a draft row at proposal position `position` (its depth less
  // one), whose draft probability had logit `feature`, under a parent its
  // lane's walk accepted: whether the target selected the row's token there.
  // The model's chance of it is at most its `cap`, as its tree's was; where
  // the cap bounds it, the label tells the fit nothing.
  struct Label final {
    uint32_t position = 0;
    float feature = 0.0F;
    bool accepted = false;
    float cap = 1.0F;
    bool operator==(const Label &) const = default;
  };

  // A lane of a completed cycle: whether it samples, and its tree's labels
  // (treeLabels).
  struct Lane final {
    bool samples = false;
    std::vector<Label> labels;
  };

  // The labels of a lane's tree, its selector's rows in the order it took
  // them (ops::DraftSelectorBuffers: each row's token, its tree row, its
  // scores, pairs of its draft probability and its estimate, and its cap),
  // after a walk whose `selections`, the target's, are those of its path's
  // rows in order from its anchor's: each row whose parent is one of those
  // rows, labeled with whether the parent's selection is its token, a row
  // past the lane's verify rows too, but none whose draft probability or cap
  // is not one (a host chain's row no draft step joined, a forced chain's
  // row, or a draft row that is not finite).
  [[nodiscard]] static std::vector<Label> treeLabels(std::span<const uint32_t> tokens,
                                                     std::span<const ops::TreeRow> rows,
                                                     std::span<const float> scores, std::span<const float> caps,
                                                     std::span<const uint32_t> selections);

  // The acceptance model of the lanes that sample, or of the greedy ones.
  [[nodiscard]] const Acceptance &acceptance(bool samples) const noexcept { return acceptance_[samples]; }
  // A completed cycle's lanes, whose labels it keeps.
  void observe(std::span<const Lane> lanes);

  // The labels kept of the lanes that sample, or of the greedy ones, oldest
  // first.
  [[nodiscard]] std::vector<Label> labels(bool samples) const;
  // The maximum a posteriori model of `labels` under `prior`, by damped
  // Newton steps from `start`.
  [[nodiscard]] static Acceptance fit(std::span<const Label> labels, const Acceptance &start,
                                      const Acceptance &prior = kPrior);
  // A lane's chances for a cycle's choice of rows (StepCosts): the estimates
  // that the target accepts each row its tree would add past its anchor,
  // best first, no more of them than its tree's `depth` (its output left
  // past its anchor) nor than a verify block's rows past it. They are the
  // `tree`'s, the draft's rows of the latest tree in which its draft made a
  // child of its anchor (ops::DraftSelectorBuffers::treeScores, a row's
  // drawn children's summing to at most its own), less its lookup chain's
  // path (a lane before its first cycle takes every row's 1 of a draft's
  // block), merged with the `chain`'s: this cycle's host chain's, a lookup
  // chain's each the product of its tokens' rates to its depth
  // (LookupAcceptance, whose agreement with the draft the host does not
  // know), a chain the grammar or the thinking end forces every token's 1;
  // without one, the last chain's path's rows that draft steps joined.
  [[nodiscard]] static std::vector<float> laneChances(std::span<const float> tree, std::span<const float> chain,
                                                      uint32_t depth);

private:
  Acceptance prior_ = kPrior;
  // The greedy lanes', then the sampled ones'.
  std::array<Acceptance, 2> acceptance_{kPrior, kPrior};
  std::array<std::vector<Label>, 2> labels_;
  std::array<size_t, 2> next_{};
  uint64_t cycles_ = 0;
};

} // namespace splash::model
