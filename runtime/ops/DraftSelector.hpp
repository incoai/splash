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
  // and its scores, fp32 pairs of its draft probability (-1 for a row a host
  // chain token made) and its estimate, the chance that the target accepts
  // it, which a later step's choice of rows reads; uint32 [lanes], the rows
  // each tree holds; and of each row the candidate indices of the children
  // it drew, in the order drawn (uint8 [SPLASH_TREE_CHILDREN], 0xff past
  // them), and its next position's candidates' probabilities (fp32
  // [SPLASH_DRAFT_CANDIDATES]), which a sampled verify row's selection tests
  // (metal/abi/Sampling.h); and each row's cap (fp32), at most which the
  // draft step to it estimated it (DraftTreeParams; 1 for a row of no draft
  // probability).
  metal::MetalBuffer treeTokens;
  metal::MetalBuffer trees;
  metal::MetalBuffer treeScores;
  metal::MetalBuffer treeSizes;
  metal::MetalBuffer treeDraws;
  metal::MetalBuffer treeNext;
  metal::MetalBuffer treeCaps;
  // A decode step's rows (ops::StepRows) as addVerifyRows lays them out from
  // the trees: each row's token, its verify input, its tree row, and its
  // source, the row of its lane's tree it is (uint32, SPLASH_NO_TOKEN for a
  // row of no lane), packed in lane order; and each lane's rows
  // (SplashLaneRows).
  metal::MetalBuffer inputTokens;
  metal::MetalBuffer treeRows;
  metal::MetalBuffer laneRows;
  metal::MetalBuffer verifySources;
};

// The estimate of a draft step of probability q at proposal position p:
// logit(estimate) = slope * logit(q) + intercepts[p] (DraftTreeParams).
struct DraftCalibration final {
  float slope = 1.0F;
  std::array<float, SPLASH_DRAFT_PROPOSAL_TOKENS> intercepts{};
};

// How one lane's tree grows (DraftTreeParams): from its anchor, over a
// verify block's rows, no row deeper than `depth` (the tokens it may retain
// less one) nor more rows than depth + 1, with the first `chain` tokens of
// its host chain and up to
// `children` children a draft row (none: its rows are its host chain's), its
// draft steps estimated by `calibration`. A greedy lane (temperature 0) takes
// a row's most likely candidates; a sampled lane draws them without
// replacement at its temperature, with the cycle's `noise`.
struct DraftTreeLane final {
  uint32_t anchor = 0;
  uint32_t depth = 0;
  uint32_t chain = 0;
  uint32_t children = SPLASH_TREE_CHILDREN;
  DraftCalibration calibration{};
  float temperature = 0.0F;
  uint64_t noise = 0;
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
// path (DraftTreeParams), a greedy lane's rows its most likely candidates
// and a sampled lane's its draws, which its verify rows' rejection sampling
// tests. A greedy tree of one child a row and no chain is the draft's most
// likely chain.
class DraftSelector final {
public:
  explicit DraftSelector(uint32_t vocabulary);

  // Exact scratch/output bytes for that many proposal positions.
  [[nodiscard]] static DraftSelectorWorkspace workspace(uint32_t positions);

  void add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers, const DraftCodebooks &codebooks,
           std::span<const DraftTreeLane> lanes) const;
  // After add: the rows of a decode step of `step.lanes` lanes
  // (VerifyRowsParams), the trees' best rows that the step's rows and its
  // lane cap hold, none past an anchor estimated below `floor`, each lane's
  // numbered in preorder, and the step's rows past them of no lane.
  static void addVerifyRows(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers, const StepRows &step,
                            float floor = 0.0F);

private:
  uint32_t vocabulary_ = 0;
};

} // namespace splash::ops
