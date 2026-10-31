#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/Sampling.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace splash::ops {

struct DraftSelectorWorkspace final {
  uint64_t partialIdsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t candidatesBytes = 0;
  uint64_t unaryBytes = 0;
  uint64_t proposalProbabilitiesBytes = 0;
};

struct DraftSelectorBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
  // fp32 [lanes][SPLASH_DRAFT_PROPOSAL_TOKENS]: each lane's confidence at each
  // proposal position, the largest probability among the position's
  // candidates (VerifyRowsParams in metal/abi/Sampling.h).
  metal::MetalBuffer confidence;
};

// What a decode cycle's verify rows follow (DraftSelector::addVerifyRows,
// VerifyRowsParams): the acceptance model logit(p) = slope *
// logit(confidence) + intercepts[position], and costs[R], the cost of a cycle
// whose lanes verify R rows in all, for R from the lane count to eight rows a
// lane, in any unit. With everyRow, each lane verifies its maximum instead.
struct VerifyRowsModel final {
  static constexpr uint32_t kCosts = SPLASH_DRAFT_QUERY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH + 1;
  float slope = 1.0F;
  std::array<float, SPLASH_DRAFT_PROPOSAL_TOKENS> intercepts{};
  std::array<float, kCosts> costs{};
  bool everyRow = true;
};

// A lane's block that a decode cycle fills without the draft (a prompt-lookup
// copy, model::CopySpecIndex): its first `count` proposals and its
// confidence at each of them, which the verify rows read as they read the
// draft's. A count of 0 leaves the lane the draft's block.
struct FixedProposals final {
  uint32_t count = 0;
  std::array<uint32_t, SPLASH_DRAFT_PROPOSAL_TOKENS> tokens{};
  std::array<float, SPLASH_DRAFT_PROPOSAL_TOKENS> confidence{};
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
// choice, and walks the SPLASH_DRAFT_PROPOSAL_TOKENS positions greedily or,
// for a sampling lane, drawing at its temperature; it writes each lane's
// confidence at each position as it walks.
class DraftSelector final {
public:
  explicit DraftSelector(uint32_t vocabulary);

  // Exact scratch/output bytes for that many proposal positions.
  [[nodiscard]] static DraftSelectorWorkspace workspace(uint32_t positions);

  void add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
           const DraftCodebooks &codebooks, std::span<const uint32_t> anchors,
           std::span<const SamplingPolicy> policies) const;
  // After add: each lane's verify rows (a uint32 per lane in `verifyRows`) by
  // `model` from its confidences, at most maximumRows[lane] of them.
  static void addVerifyRows(metal::CommandGraph &graph, const metal::MetalBuffer &confidence,
                            const metal::MetalBuffer &verifyRows, const VerifyRowsModel &model,
                            std::span<const uint32_t> maximumRows);
  // After add, or in its place where every lane's block is fixed, and before
  // addVerifyRows: writes each lane's fixed proposals and their confidence
  // over the draft's, in the cycle's command (the thinking end's proposal
  // writer, decode_draft_thinking_cutoff, writes both rows). Lanes whose
  // count is 0 keep the draft's block.
  static void addFixedProposals(metal::CommandGraph &graph, std::span<const FixedProposals> lanes,
                                const metal::MetalBuffer &proposedTokens,
                                const metal::MetalBuffer &confidence);
  // The confidence `model` reads as acceptance probability `acceptance` at
  // proposal `position`: the inverse of its logit(p) = slope *
  // logit(confidence) + intercepts[position], within the clamp
  // draft_verify_rows reads confidence through. Under everyRow, or a slope
  // that leaves confidence no say, the acceptance itself.
  [[nodiscard]] static float confidenceFor(const VerifyRowsModel &model, uint32_t position,
                                           double acceptance);

private:
  uint32_t vocabulary_ = 0;
};

} // namespace splash::ops
