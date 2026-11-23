#include "ops/DSpark.hpp"

#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"
#include "ops/LaneRows.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace splash::ops {
namespace {

constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kRank = SPLASH_DSPARK_RANK;
constexpr uint32_t kJoint = SPLASH_DSPARK_JOINT;
constexpr uint32_t kBlockRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
// The mix keeps 16 columns a thread of its 256 (kernels/shared/dspark_mix.metal).
constexpr uint32_t kMixWidth = 256 * 16;
// The vocabulary pass's groups: each scores its share of the vocabulary for
// every lane, reading each W2 row once for all of them.
constexpr uint32_t kLogitGroups = 512;
// The joint projection's groups: a simdgroup an output, eight a group.
constexpr uint32_t kJointGroups = kJoint / 8;

void requireLanes(uint32_t lanes) {
  if (!lanes || lanes > kLanes) throw std::invalid_argument("invalid DSpark batch width");
}

} // namespace

void DSpark::addMix(metal::CommandGraph &graph, metal::MetalBuffer features, const DSparkMix &mix,
                    metal::MetalBuffer output, uint32_t rows, uint32_t width) {
  if (!rows || !width || width > kMixWidth) throw std::invalid_argument("invalid DSpark mix");
  requireBytes(features, uint64_t{rows} * SPLASH_DSPARK_TAPS * width * 2, "DSpark features");
  requireBytes(mix.gamma, uint64_t{width} * sizeof(float), "DSpark mix gamma");
  requireBytes(output, uint64_t{rows} * width * 2, "DSpark mix output");
  DSparkMixParams params{rows, width, {}};
  std::ranges::copy(mix.alpha, params.alpha);
  graph.add("dspark_mix_rms", {std::move(features), mix.gamma, std::move(output)}, params, {rows, 1, 1});
}

void DSpark::addFrnnCommit(metal::CommandGraph &graph, metal::MetalBuffer rows,
                           std::span<const metal::MetalBuffer> frnn, metal::MetalBuffer retained,
                           metal::MetalBuffer laneRows, uint32_t lanes, uint32_t width) {
  requireLanes(lanes);
  if (frnn.size() != kLanes || !width) throw std::invalid_argument("invalid DSpark f_rnn commit");
  for (uint32_t lane = 0; lane < lanes; ++lane) requireBytes(frnn[lane], uint64_t{width} * 2, "DSpark f_rnn");
  requireBytes(retained, uint64_t{lanes} * sizeof(uint32_t), "DSpark retained rows");
  requireBytes(laneRows, uint64_t{lanes} * sizeof(LaneRows), "DSpark lane rows");
  std::vector<metal::MetalBuffer> bindings{std::move(rows)};
  bindings.insert(bindings.end(), frnn.begin(), frnn.end());
  bindings.push_back(std::move(retained));
  bindings.push_back(std::move(laneRows));
  graph.add("dspark_frnn_commit", std::move(bindings), DSparkCommitParams{lanes, width}, {lanes, 1, 1});
}

DSparkMarkovWorkspace DSpark::markovWorkspace(uint32_t lanes, uint32_t vocabulary) {
  requireLanes(lanes);
  if (!vocabulary) throw std::invalid_argument("invalid DSpark vocabulary");
  const uint64_t partials = uint64_t{lanes} * SPLASH_DSPARK_SHARDS * kCandidates;
  return {uint64_t{lanes} * kJoint * sizeof(float), uint64_t{lanes} * kRank * sizeof(float),
          uint64_t{lanes} * sizeof(uint32_t),       uint64_t{lanes} * vocabulary * sizeof(float),
          partials * sizeof(float),                 partials * sizeof(uint32_t)};
}

void DSpark::addMarkov(metal::CommandGraph &graph, const DSparkMarkovWeights &weights,
                       const DSparkMarkovBuffers &buffers, DSparkMarkovShape shape,
                       std::span<const DraftTreeLane> lanes) {
  const auto count = static_cast<uint32_t>(lanes.size());
  requireLanes(count);
  if (!shape.hidden || !shape.proposed || shape.proposed > shape.vocabulary || !(shape.cap > 0.0F))
    throw std::invalid_argument("invalid DSpark Markov shape");
  const DSparkMarkovWorkspace workspace = markovWorkspace(count, shape.vocabulary);
  const uint64_t rows = uint64_t{count} * kBlockRows;
  requireBytes(weights.joint, uint64_t{kJoint} * (2 * kRank + 2 * shape.hidden) * 2, "DSpark joint projection");
  requireBytes(weights.jointBias, kJoint * sizeof(float), "DSpark joint bias");
  requireBytes(weights.embedding, uint64_t{shape.vocabulary} * kRank * 2, "DSpark Markov embedding");
  requireBytes(weights.foldWeights, uint64_t{shape.vocabulary} * kRank * 2, "DSpark folded weights");
  requireBytes(weights.foldBias, uint64_t{shape.vocabulary} * sizeof(float), "DSpark folded bias");
  requireBytes(buffers.hidden, rows * shape.hidden * 2, "DSpark draft hidden");
  requireBytes(buffers.base, rows * shape.vocabulary * sizeof(float), "DSpark base logits");
  requireBytes(buffers.frnn, uint64_t{count} * shape.hidden * 2, "DSpark f_rnn rows");
  requireBytes(buffers.joint, workspace.jointBytes, "DSpark joint");
  requireBytes(buffers.state, workspace.stateBytes, "DSpark state");
  requireBytes(buffers.previous, workspace.previousBytes, "DSpark previous tokens");
  requireBytes(buffers.logits, workspace.logitsBytes, "DSpark step logits");
  requireBytes(buffers.partialValues, workspace.partialValuesBytes, "DSpark partial values");
  requireBytes(buffers.partialIds, workspace.partialIdsBytes, "DSpark partial ids");
  const uint64_t lattice = uint64_t{count} * kPositions * kCandidates;
  requireBytes(buffers.candidates, lattice * sizeof(uint32_t), "DSpark candidates");
  requireBytes(buffers.unary, lattice * sizeof(float), "DSpark candidate logits");
  DSparkMarkovParams params{count, 0, shape.hidden, shape.vocabulary, shape.proposed, shape.cap, {}, {}, {}};
  for (uint32_t lane = 0; lane < count; ++lane) {
    const DraftTreeLane &l = lanes[lane];
    if (!(l.temperature >= 0.0F && std::isfinite(l.temperature)))
      throw std::invalid_argument("invalid DSpark lane temperature");
    params.anchor[lane] = l.anchor;
    params.temperature[lane] = l.temperature;
    params.noise[lane][0] = static_cast<uint32_t>(l.noise);
    params.noise[lane][1] = static_cast<uint32_t>(l.noise >> 32);
  }
  // Each step reads the state and token the one before it left.
  for (uint32_t step = 0; step < kPositions; ++step) {
    params.step = step;
    graph.add("dspark_markov_joint",
              {weights.joint, weights.jointBias, buffers.state, weights.embedding, buffers.previous, buffers.hidden,
               buffers.frnn, buffers.joint},
              params, {kJointGroups, 1, 1});
    graph.add("dspark_markov_logits",
              {buffers.joint, weights.foldWeights, weights.foldBias, buffers.base, buffers.logits}, params,
              {kLogitGroups, 1, 1});
    graph.add("dspark_markov_top16", {buffers.logits, buffers.partialValues, buffers.partialIds}, params,
              {uint64_t{count} * SPLASH_DSPARK_SHARDS, 1, 1});
    graph.add("dspark_markov_select",
              {buffers.partialValues, buffers.partialIds, buffers.joint, buffers.state, buffers.previous,
               buffers.candidates, buffers.unary},
              params, {count, 1, 1});
  }
}

void DSpark::addFold(metal::CommandGraph &graph, metal::MetalBuffer projected, const DSparkMarkovWeights &weights,
                     uint32_t first, uint32_t rows, uint32_t vocabulary) {
  if (!rows || !vocabulary || first + rows > kRank + 2) throw std::invalid_argument("invalid DSpark fold");
  requireBytes(projected, uint64_t{rows} * vocabulary * sizeof(float), "DSpark fold rows");
  requireBytes(weights.foldWeights, uint64_t{vocabulary} * kRank * 2, "DSpark folded weights");
  requireBytes(weights.foldBias, uint64_t{vocabulary} * sizeof(float), "DSpark folded bias");
  graph.add("dspark_fold_rows", {std::move(projected), weights.foldWeights, weights.foldBias},
            DSparkFoldParams{first, rows, kRank, vocabulary}, {(vocabulary + 255) / 256, 1, 1});
}

} // namespace splash::ops
