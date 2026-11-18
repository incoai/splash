#include "ops/LiLiCorr.hpp"

#include "metal/abi/Gguf.h"
#include "metal/abi/LiLiCorr.h"
#include "metal/abi/Sampling.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kWidth = SPLASH_LILICORR_HIDDEN;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kNodes = SPLASH_LILICORR_NODES;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint64_t kBf16 = sizeof(uint16_t);

// Rows of a projection's input or output buffer for `rows` rows: its tiles'
// rows (LinearPlan::storageRows), at most whole prefill tiles.
constexpr uint64_t storageRows(uint64_t rows) {
  return (rows + GGUF_PREFILL_ROWS - 1) / GGUF_PREFILL_ROWS * GGUF_PREFILL_ROWS;
}

metal::DispatchSize elementwise(uint32_t width, uint32_t rows) { return {width / 256, rows, 1}; }

} // namespace

std::vector<const Projection *> LiLiCorrWeights::projections() const {
  std::vector<const Projection *> result{&tokenProjection, &passProjection, &contextProjection, &featureDown,
                                         &factorProjection, &outHead,       &inHead,            &anchorOutHead};
  for (const LiLiCorrLayerWeights &layer : layers)
    for (const Projection *p : {&layer.inProjection, &layer.outProjection, &layer.upProjection,
                                &layer.downProjection})
      result.push_back(p);
  return result;
}

uint32_t LiLiCorr::maximumRows() noexcept {
  return std::max<uint32_t>(kLanes * kNodes, kLanes * SPLASH_DRAFT_QUERY_ROWS);
}

std::vector<uint64_t> LiLiCorr::scratchBytes(uint32_t hidden) {
  const uint64_t nodes = storageRows(kLanes * kNodes), lanes = storageRows(kLanes);
  return {kLanes * kPositions * sizeof(float),
          nodes * sizeof(uint32_t),
          nodes * kWidth * kBf16,
          nodes * kWidth * kBf16,
          nodes * hidden * kBf16,
          nodes * kWidth * kBf16,
          storageRows(kLanes * SPLASH_DRAFT_QUERY_ROWS) * kWidth * kBf16,
          nodes * kWidth * sizeof(float),
          nodes * kWidth * kBf16,
          nodes * 3 * kWidth * kBf16,
          nodes * kWidth * kBf16,
          nodes * kWidth * kBf16,
          nodes * SPLASH_LILICORR_MLP * kBf16,
          lanes * hidden * kBf16,
          lanes * kWidth * kBf16,
          lanes * kWidth * sizeof(float),
          lanes * kWidth * kBf16,
          lanes * kWidth * kBf16,
          nodes * kWidth * kBf16,
          nodes * kWidth * kBf16};
}

void LiLiCorr::addStoreAnchor(metal::CommandGraph &graph, const metal::MetalBuffer &rows, uint32_t row,
                              const metal::MetalBuffer &anchor, uint32_t hidden, uint32_t anchorRow) {
  if (!hidden || hidden % 256) throw std::invalid_argument("invalid LiLiCorr anchor width");
  requireBytes(anchor, (uint64_t{anchorRow} + 1) * hidden * kBf16, "LiLiCorr anchor");
  requireBytes(rows, (uint64_t{row} + 1) * hidden * kBf16, "LiLiCorr anchor source");
  const uint32_t params[4] = {hidden, row, anchorRow, 0};
  graph.add("lilicorr_store_anchor", {rows, rows, anchor, rows}, params, {hidden / 256, 1, 1});
}

void LiLiCorr::addStoreAcceptedAnchor(metal::CommandGraph &graph, const metal::MetalBuffer &rows,
                                      const metal::MetalBuffer &paths, const metal::MetalBuffer &laneRows,
                                      uint32_t lane, const metal::MetalBuffer &anchor, uint32_t hidden) {
  if (!hidden || hidden % 256) throw std::invalid_argument("invalid LiLiCorr anchor width");
  requireBytes(anchor, uint64_t{hidden} * kBf16, "LiLiCorr anchor");
  const uint32_t params[4] = {hidden, 0xffffffffu, 0, lane};
  graph.add("lilicorr_store_anchor", {rows, paths, anchor, laneRows}, params, {hidden / 256, 1, 1});
}

void LiLiCorr::add(metal::CommandGraph &graph, const Linear &linear, const DraftSelector &selector,
                   const DraftSelectorBuffers &buffers, const LiLiCorrWeights &w, const LiLiCorrScratch &s,
                   const LiLiCorrInputs &inputs, DraftVocabulary vocabulary,
                   std::span<const DraftTreeLane> treeLanes, uint32_t hidden) {
  const uint32_t lanes = static_cast<uint32_t>(treeLanes.size());
  if (!lanes || lanes > kLanes || inputs.anchorRows.size() != lanes || !inputs.finalHidden ||
      w.layers.size() != SPLASH_LILICORR_LAYERS || !hidden || hidden % 256)
    throw std::invalid_argument("invalid LiLiCorr selector batch");
  const uint32_t positions = lanes * kPositions, nodes = lanes * kNodes;
  const DraftSelectorBatch batch = selector.requireBuffers(buffers, vocabulary, treeLanes);
  // Every scratch buffer holds a full batch's rows (scratchBytes).
  const std::vector<uint64_t> bytes = scratchBytes(hidden);
  const metal::MetalBuffer *scratch[] = {&s.lse,        &s.ids,        &s.features,       &s.featureDown, &s.embedded,
                                         &s.token,      &s.pass,       &s.x,              &s.normalized,  &s.wide,
                                         &s.attention,  &s.projected,  &s.mlp,            &s.anchorInput, &s.anchorProjected,
                                         &s.anchorState, &s.anchorRows, &s.anchorOut,     &s.outHead,     &s.inHead};
  for (size_t i = 0; i < bytes.size(); ++i) requireBytes(*scratch[i], bytes[i], "LiLiCorr scratch");
  for (const metal::MetalBuffer &row : inputs.anchorRows)
    requireBytes(row, uint64_t{hidden} * kBf16, "LiLiCorr anchor source");
  const auto project = [&](const metal::MetalBuffer &input, const Projection &p, const metal::MetalBuffer &output,
                           uint32_t rows) {
    linear.add(graph, {.input = input, .output = output, .scratch = s.linear}, p, linear.plan(p, rows));
  };

  // Each position's sixteen best draft tokens, the first eight of them nodes
  // with their features.
  selector.addTopCandidates(graph, buffers, batch);
  graph.add("lilicorr_lse", {buffers.logits, s.lse}, batch.params.columns, {positions, 1, 1}, {1024, 1, 1});
  graph.add("lilicorr_candidates",
            {buffers.partialIds, buffers.partialValues, buffers.candidates, buffers.unary, s.lse, s.ids,
             w.featureNormWeight, w.featureNormBias, w.featureUp, w.featureUpBias, s.features},
            {positions, 1, 1}, {1024, 1, 1});

  // The nodes' inputs.
  Embedding::add(graph, s.ids, inputs.tokenEmbedding, s.embedded, nodes);
  project(s.embedded, w.tokenProjection, s.token, nodes);
  project(inputs.finalHidden, w.passProjection, s.pass, lanes * SPLASH_DRAFT_QUERY_ROWS);
  project(s.features, w.featureDown, s.featureDown, nodes);
  graph.add("lilicorr_embed_sum",
            {s.token, w.tokenBias, s.pass, w.passBias, s.featureDown, w.featureDownBias, w.slotEmbedding,
             w.rankEmbedding, s.x},
            elementwise(kWidth, nodes));

  // The lanes' anchor states.
  for (uint32_t lane = 0; lane < lanes; ++lane)
    addStoreAnchor(graph, inputs.anchorRows[lane], 0, s.anchorInput, hidden, lane);
  project(s.anchorInput, w.contextProjection, s.anchorProjected, lanes);
  graph.add("lilicorr_anchor", {s.anchorProjected, w.contextBias, w.anchorNorm.buffer, s.anchorState, s.anchorRows},
            {lanes, 1, 1});
  project(s.anchorRows, w.anchorOutHead, s.anchorOut, lanes);

  for (const LiLiCorrLayerWeights &layer : w.layers) {
    graph.add("lilicorr_rms", {s.x, layer.attentionNorm.buffer, s.normalized}, {nodes, 1, 1});
    project(s.normalized, layer.inProjection, s.wide, nodes);
    graph.add("lilicorr_attention", {s.wide, layer.inBias, w.attentionBias, s.attention},
              {kNodes, SPLASH_LILICORR_HEADS, lanes}, {32, 1, 1});
    project(s.attention, layer.outProjection, s.projected, nodes);
    graph.add("lilicorr_residual", {s.x, s.projected, layer.outBias}, elementwise(kWidth, nodes));
    graph.add("lilicorr_rms", {s.x, layer.mlpNorm.buffer, s.normalized}, {nodes, 1, 1});
    project(s.normalized, layer.upProjection, s.mlp, nodes);
    graph.add("lilicorr_bias_silu", {s.mlp, layer.upBias}, SPLASH_LILICORR_MLP,
              elementwise(SPLASH_LILICORR_MLP, nodes));
    project(s.mlp, layer.downProjection, s.projected, nodes);
    graph.add("lilicorr_residual", {s.x, s.projected, layer.downBias}, elementwise(kWidth, nodes));
  }

  // The factor heads and the edge tables.
  graph.add("lilicorr_factor_input", {s.x, w.outputNorm.buffer, s.anchorState, s.wide}, {nodes, 1, 1});
  project(s.wide, w.factorProjection, s.normalized, nodes);
  graph.add("lilicorr_bias_silu", {s.normalized, w.factorBias}, kWidth, elementwise(kWidth, nodes));
  project(s.normalized, w.outHead, s.outHead, nodes);
  project(s.normalized, w.inHead, s.inHead, nodes);
  graph.add("lilicorr_scores",
            {s.outHead, w.outHeadBias, s.inHead, w.inHeadBias, s.anchorOut, w.anchorOutHeadBias,
             buffers.partialValues},
            lanes, {kPositions, lanes, 1});

  selector.addTree(graph, buffers, batch);
}

} // namespace splash::ops
