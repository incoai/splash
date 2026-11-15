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
  // The command's packed rows (ops::LaneRows): each row's token, its verify
  // input; its tree row (metal/abi/VerifyTree.h); and its scores, fp32 pairs
  // of its draft probability and its estimate.
  metal::MetalBuffer inputTokens;
  metal::MetalBuffer treeRows;
  metal::MetalBuffer treeScores;
  // uint32 [lanes]: the rows each lane's tree holds, which addVerifyRows
  // narrows to the rows it verifies.
  metal::MetalBuffer rowCounts;
};

// How one lane's tree grows (DraftTreeParams): from its anchor, over its
// verify block's rows, no row deeper than `depth` (the tokens it may retain
// less one), with the first `chain` tokens of its host chain and up to
// `children` children a draft row (none: its rows are its host chain's).
struct DraftTreeLane final {
  uint32_t anchor = 0;
  LaneRows block{};
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
  // After add: each lane's verify rows (VerifyRowsParams), written over
  // rowCounts, numbered in preorder over the buffers' rows: every row of its
  // tree without costs, or the most expected accepted tokens per cost over
  // the lanes' trees, where costs[R] is the cost of a cycle whose lanes verify
  // R rows in all, for R from the lane count to their blocks' rows, in any
  // unit.
  static void addVerifyRows(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
                            std::span<const float> costs, std::span<const LaneRows> blocks);

private:
  uint32_t vocabulary_ = 0;
};

} // namespace splash::ops
