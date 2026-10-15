#include "ops/DraftSelector.hpp"

#include "metal/abi/Sampling.h"
#include "ops/BufferExtent.hpp"

#include <bit>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kShards = SPLASH_DRAFT_SAMPLING_SHARDS;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group dispatch of one
// lane of eight rows against the 28 groups of four (wider groups speed up the
// one lane and slow down the four).
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
                        std::span<const SamplingPolicy> policies,
                        uint32_t rows) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > SPLASH_MAXIMUM_BATCH_WIDTH || rows < 2 ||
      rows > SPLASH_MAXIMUM_VERIFY_ROWS)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  const uint32_t positions = rows - 1;
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.rows = rows;
  params.vocabulary = vocabulary_;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  // A lane's proposal positions take its query rows after the first of the
  // logits and the selector hidden rows; a sampling lane draws its proposals
  // with its proposal uniforms and writes their candidates' probabilities.
  const DraftSelectorWorkspace workspace = DraftSelector::workspace(lanes * positions);
  const uint64_t queryRows = uint64_t{lanes} * rows;
  const uint64_t codebook = uint64_t{vocabulary_} * SPLASH_DRAFT_SELECTOR_RANK * 2;
  requireBytes(buffers.logits, queryRows * vocabulary_ * sizeof(float), "draft logits");
  requireBytes(buffers.partialIds, workspace.partialIdsBytes, "draft selector partial id");
  requireBytes(buffers.partialValues, workspace.partialValuesBytes, "draft selector partial value");
  requireBytes(buffers.candidates, workspace.candidatesBytes, "draft candidate");
  requireBytes(buffers.unary, workspace.unaryBytes, "draft candidate score");
  requireBytes(buffers.selectorHidden, queryRows * SPLASH_DRAFT_SELECTOR_RANK * 2, "draft selector hidden");
  requireBytes(codebooks.predecessor, codebook, "draft predecessor codebook");
  requireBytes(codebooks.successor, codebook, "draft successor codebook");
  requireBytes(buffers.proposedTokens, uint64_t{lanes} * positions * sizeof(uint32_t), "proposed token");
  if (const uint64_t sampledLanes = std::bit_width(params.sampling_mask)) {
    requireBytes(buffers.uniforms,
                 ((sampledLanes - 1) * SPLASH_SAMPLING_UNIFORMS + SPLASH_UNIFORM_PROPOSALS + positions) * sizeof(float),
                 "proposal uniform");
    requireBytes(buffers.proposalProbabilities, sampledLanes * positions * SPLASH_DRAFT_CANDIDATES * sizeof(float),
                 "proposal probability");
  }
  graph.add("draft_select_top16_sharded",
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            params, {uint64_t{lanes} * positions * kShards, 1, 1});
  graph.add("draft_select_edges",
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden, codebooks.predecessor,
             codebooks.successor},
            params, {uint64_t{lanes} * positions, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add("draft_select_dflash",
            {buffers.candidates, buffers.unary, buffers.partialValues,
             buffers.uniforms, buffers.proposedTokens,
             buffers.proposalProbabilities},
            params, {lanes, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops
