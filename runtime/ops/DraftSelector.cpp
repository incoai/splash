#include "ops/DraftSelector.hpp"

#include "metal/abi/Sampling.h"
#include "ops/BufferExtent.hpp"

#include <bit>
#include <cmath>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kShards = SPLASH_DRAFT_SAMPLING_SHARDS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group B1 dispatch
// against the 28 groups of B4 (wider groups speed up B1 and slow down B4).
constexpr uint32_t kEdgeThreads = 256;

} // namespace

DraftSelector::DraftSelector(uint32_t vocabulary) : vocabulary_(vocabulary) {
  if (!vocabulary)
    throw std::invalid_argument("invalid draft selector vocabulary");
}

DraftSelectorWorkspace DraftSelector::workspace(uint32_t positions) {
  if (!positions)
    throw std::invalid_argument("invalid draft selector workspace position count");
  const uint64_t candidates = uint64_t{positions} * SPLASH_DRAFT_CANDIDATES;
  // The partial values are followed by each position's 16 x 16 edge table.
  return {candidates * kShards * sizeof(uint32_t),
          candidates * (kShards + SPLASH_DRAFT_CANDIDATES) * sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(float),
          candidates * sizeof(float)};
}

void DraftSelector::add(metal::CommandGraph &graph,
                        const DraftSelectorBuffers &buffers,
                        const DraftCodebooks &codebooks,
                        std::span<const uint32_t> anchors,
                        std::span<const SamplingPolicy> policies) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.vocabulary = vocabulary_;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  // A lane's proposal positions take rows 1-7 of its eight query rows of the
  // logits and the selector hidden rows; a sampling lane draws its proposals
  // with its proposal uniforms and writes their candidates' probabilities.
  const DraftSelectorWorkspace workspace = DraftSelector::workspace(lanes * kPositions);
  const uint64_t rows = uint64_t{lanes} * SPLASH_DRAFT_QUERY_ROWS;
  const uint64_t codebook = uint64_t{vocabulary_} * SPLASH_DRAFT_SELECTOR_RANK * 2;
  requireBytes(buffers.logits, rows * vocabulary_ * sizeof(float), "draft logits");
  requireBytes(buffers.partialIds, workspace.partialIdsBytes, "draft selector partial id");
  requireBytes(buffers.partialValues, workspace.partialValuesBytes, "draft selector partial value");
  requireBytes(buffers.candidates, workspace.candidatesBytes, "draft candidate");
  requireBytes(buffers.unary, workspace.unaryBytes, "draft candidate score");
  requireBytes(buffers.selectorHidden, rows * SPLASH_DRAFT_SELECTOR_RANK * 2, "draft selector hidden");
  requireBytes(codebooks.predecessor, codebook, "draft predecessor codebook");
  requireBytes(codebooks.successor, codebook, "draft successor codebook");
  requireBytes(buffers.proposedTokens, uint64_t{lanes} * kPositions * sizeof(uint32_t), "proposed token");
  requireBytes(buffers.confidence, uint64_t{lanes} * kPositions * sizeof(float), "proposal confidence");
  if (const uint64_t sampledLanes = std::bit_width(params.sampling_mask)) {
    requireBytes(buffers.uniforms,
                 ((sampledLanes - 1) * SPLASH_SAMPLING_UNIFORMS + SPLASH_UNIFORM_PROPOSALS + kPositions) * sizeof(float),
                 "proposal uniform");
    requireBytes(buffers.proposalProbabilities, sampledLanes * kPositions * SPLASH_DRAFT_CANDIDATES * sizeof(float),
                 "proposal probability");
  }
  graph.add("draft_select_top16_sharded",
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            vocabulary_, {uint64_t{lanes} * kPositions * kShards, 1, 1});
  graph.add("draft_select_edges",
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden, codebooks.predecessor,
             codebooks.successor},
            params, {uint64_t{lanes} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add("draft_select_dflash",
            {buffers.candidates, buffers.unary, buffers.partialValues,
             buffers.uniforms, buffers.proposedTokens,
             buffers.proposalProbabilities, buffers.confidence},
            params, {lanes, 1, 1}, {1, 1, 1});
}

void DraftSelector::addVerifyRows(metal::CommandGraph &graph, const metal::MetalBuffer &confidence,
                                  const metal::MetalBuffer &verifyRows, const VerifyRowsModel &model,
                                  std::span<const uint32_t> maximumRows) {
  if (maximumRows.empty() || maximumRows.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft verify rows batch");
  const uint32_t lanes = static_cast<uint32_t>(maximumRows.size());
  VerifyRowsParams params{};
  params.lanes = lanes;
  params.every_row = model.everyRow ? 1 : 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (!maximumRows[lane] || maximumRows[lane] > SPLASH_DRAFT_QUERY_ROWS)
      throw std::invalid_argument("invalid draft verify rows maximum");
    params.maximum[lane] = maximumRows[lane];
  }
  if (!model.everyRow) {
    if (!std::isfinite(model.slope))
      throw std::invalid_argument("invalid draft verify rows model");
    params.slope = model.slope;
    for (uint32_t position = 0; position < kPositions; ++position) {
      if (!std::isfinite(model.intercepts[position]))
        throw std::invalid_argument("invalid draft verify rows model");
      params.intercept[position] = model.intercepts[position];
    }
    // The costs of every total the lanes may verify, from their anchors to
    // eight rows each.
    for (uint32_t rows = lanes; rows <= lanes * SPLASH_DRAFT_QUERY_ROWS; ++rows) {
      if (!(std::isfinite(model.costs[rows]) && model.costs[rows] > 0.0F))
        throw std::invalid_argument("invalid draft verify rows cost");
      params.cost[rows] = model.costs[rows];
    }
  }
  requireBytes(confidence, uint64_t{lanes} * kPositions * sizeof(float), "proposal confidence");
  requireBytes(verifyRows, uint64_t{lanes} * sizeof(uint32_t), "verify rows");
  graph.add("draft_verify_rows", {confidence, verifyRows}, params, {1, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops
