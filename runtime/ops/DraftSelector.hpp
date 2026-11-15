#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/LaneRows.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace splash::ops {

struct DraftSelectorWorkspace final {
  uint64_t partialIdsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t candidatesBytes = 0;
  uint64_t unaryBytes = 0;
};

struct DraftSelectorBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  // Each lane's host chain (DraftTreeParams in metal/abi/Sampling.h): uint32
  // [lanes][SPLASH_TREE_CHAIN_TOKENS] tokens, and fp32
  // [lanes][SPLASH_TREE_CHAIN_TOKENS][SPLASH_TREE_AGREEMENTS] estimates.
  metal::MetalBuffer chainTokens;
  metal::MetalBuffer chainRates;
  // Each lane's tree (DraftTreeParams), up to SPLASH_TARGET_VERIFY_ROWS rows
  // from row lane x SPLASH_TARGET_VERIFY_ROWS in the order the selector took
  // them, best first: each row's token, its tree row (metal/abi/VerifyTree.h)
  // and its scores, fp32 pairs of its draft probability and its estimate, the
  // chance that the target accepts it, which a later step's choice of rows
  // reads; and uint32 [lanes], the rows each tree holds.
  metal::MetalBuffer treeTokens;
  metal::MetalBuffer trees;
  metal::MetalBuffer treeScores;
  metal::MetalBuffer treeSizes;
  // A decode step's rows (ops::StepRows) as addVerifyRows lays them out from
  // the trees: each row's token, its verify input, and its tree row, packed
  // in lane order; each lane's rows (SplashLaneRows) and its verify rows
  // (uint32), the first of them.
  metal::MetalBuffer inputTokens;
  metal::MetalBuffer treeRows;
  metal::MetalBuffer laneRows;
  metal::MetalBuffer verifyRows;
};

// How one lane's tree grows (DraftTreeParams): from its anchor, over a
// verify block's rows, no row deeper than `depth` (the tokens it may retain
// less one), with the first `chain` tokens of its host chain and up to
// `children` children a draft row (none: its rows are its host chain's).
struct DraftTreeLane final {
  uint32_t anchor = 0;
  uint32_t depth = 0;
  uint32_t chain = 0;
  uint32_t children = 1;
};

// The estimate of a draft step of probability q at proposal position p:
// logit(estimate) = slope * logit(q) + intercepts[p] (DraftTreeParams).
struct DraftCalibration final {
  float slope = 1.0F;
  std::array<float, SPLASH_DRAFT_PROPOSAL_TOKENS> intercepts{};
};

// The draft's selector codebooks, which score the edge from a proposal
// position's predecessor candidate to each of its candidates.
struct DraftCodebooks final {
  metal::MetalBuffer predecessor;
  metal::MetalBuffer successor;
};

// The DFlash draft's proposal policy (draft_select_* in
// metal/kernels/decode/sampling.metal): each lane keeps the
// SPLASH_DRAFT_CANDIDATES most likely draft tokens of every proposal
// position, scores each candidate with its edge from the previous position's
// candidate, and grows its verify block's rows into a tree over them and its
// host chain, best first by the estimate that the target accepts each row's
// path (DraftTreeParams), whatever its policy: a sampled lane's acceptance
// follows the target's draws. A tree of one child a row and no chain is the
// draft's most likely chain.
class DraftSelector final {
public:
  explicit DraftSelector(uint32_t vocabulary);

  // Exact scratch/output bytes for that many proposal positions.
  [[nodiscard]] static DraftSelectorWorkspace workspace(uint32_t positions);

  void add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers, const DraftCodebooks &codebooks,
           std::span<const DraftTreeLane> lanes, const DraftCalibration &calibration) const;
  // After add: the rows of a decode step of `step.lanes` lanes
  // (VerifyRowsParams), each lane's verify rows numbered in preorder: the
  // trees' best rows that the step's rows hold without costs, or as many of
  // them as give the most expected accepted tokens per cost, where costs[R]
  // is the cost of a step that verifies R rows in all, for R from the lane
  // count to the step's rows, in any unit.
  static void addVerifyRows(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers, const StepRows &step,
                            std::span<const float> costs);

private:
  uint32_t vocabulary_ = 0;
};

} // namespace splash::ops
