#include "ops/Sampling.hpp"

#include "metal/abi/Sampling.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kTargetShards = SPLASH_TARGET_SAMPLING_SHARDS;

// A lane that does not sample takes the argmax path: one candidate, unit
// temperature and top-p, whatever the request carried.
struct EffectivePolicy final {
  uint32_t topK;
  float temperature;
  float topP;
};
EffectivePolicy effectivePolicy(const SamplingPolicy &policy) noexcept {
  if (policy.samples()) return {policy.topK, policy.temperature, policy.topP};
  return {1, 1.0F, 1.0F};
}
constexpr uint32_t kTargetCandidates = kTargetSamplingCandidates;
constexpr uint32_t kDraftShards = SPLASH_DRAFT_SAMPLING_SHARDS;
constexpr uint32_t kDraftCandidates = 16;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group B1 dispatch
// against the 28 groups of B4 (wider groups speed up B1 and slow down B4).
constexpr uint32_t kEdgeThreads = 256;

} // namespace

SamplingWorkspace Sampling::workspace(uint32_t rows) {
  if (!rows)
    throw std::invalid_argument("invalid sampling workspace row count");
  const uint64_t shards = uint64_t{rows} * kTargetShards;
  const uint64_t candidates = uint64_t{rows} * kTargetCandidates;
  return {shards * sizeof(float), shards * sizeof(uint32_t),
          shards * kTargetCandidates * sizeof(uint32_t),
          shards * kTargetCandidates * sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(float)};
}

DraftSelectorWorkspace Sampling::draftWorkspace(uint32_t positions) {
  if (!positions)
    throw std::invalid_argument("invalid draft selector workspace position count");
  const uint64_t candidates = uint64_t{positions} * kDraftCandidates;
  // The partial values are followed by each position's 16 x 16 edge table.
  return {candidates * kDraftShards * sizeof(uint32_t),
          candidates * (kDraftShards + kDraftCandidates) * sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(uint16_t),
          candidates * sizeof(float)};
}

Sampling::Sampling(metal::MetalBackend &backend, uint32_t vocabulary,
                   uint32_t rowsPerLane)
    : backend_(backend), vocabulary_(vocabulary), rowsPerLane_(rowsPerLane),
      maskWords_((vocabulary + 31) / 32) {
  if (!vocabulary || !rowsPerLane)
    throw std::invalid_argument("invalid sampling geometry");
}

void Sampling::addInitial(metal::CommandGraph &graph,
                          const SamplingPolicy &policy,
                          SamplingBuffers buffers,
                          uint32_t rowOffset) const {
  if (rowOffset >= rowsPerLane_)
    throw std::invalid_argument("invalid initial sampling row");
  if (policy.samples() || policy.constrained) {
    const EffectivePolicy effective = effectivePolicy(policy);
    const TargetSamplingParams params{
        vocabulary_, rowOffset, effective.topK, effective.temperature,
        effective.topP, maskWords_, 0, policy.constrained ? 1U : 0U};
    graph.add("decode_sample_top32_sharded",
              {buffers.logits, buffers.partialIds, buffers.partialValues,
               buffers.constraintMasks},
              params, {kTargetShards, 1, 1});
    graph.add("decode_sample_top32_probs",
              {buffers.partialIds, buffers.partialValues, buffers.topIds,
               buffers.topProbabilities},
              params, {1, 1, 1}, {1, 1, 1});
    if (policy.samples()) {
      graph.add("decode_sample_sparse_draw",
                {buffers.topIds, buffers.topProbabilities, buffers.uniforms,
                 buffers.outputTokens},
                {1, 1, 1}, {1, 1, 1});
    } else {
      graph.add("decode_sample_sparse_top1",
                {buffers.topIds, buffers.topProbabilities,
                 buffers.outputTokens},
                {1, 1, 1}, {1, 1, 1});
    }
    return;
  }

  metal::MetalBuffer logits = buffers.logits;
  if (rowOffset) {
    const uint64_t rowBytes = uint64_t{vocabulary_} * sizeof(uint16_t);
    logits = backend_.view(logits, uint64_t{rowOffset} * rowBytes, rowBytes);
  }
  graph.add("decode_sample_argmax_sharded",
            {std::move(logits), buffers.argmaxValues, buffers.argmaxIndices},
            vocabulary_, {kTargetShards, 1, 1});
  graph.add("decode_sample_argmax_reduce",
            {buffers.argmaxValues, buffers.argmaxIndices,
             buffers.outputTokens},
            {1, 1, 1}, {32, 1, 1});
}

void Sampling::addVerify(metal::CommandGraph &graph,
                         std::span<const SamplingPolicy> policies,
                         SamplingBuffers buffers) const {
  if (policies.empty() || policies.size() > kMaximumLanes)
    throw std::invalid_argument("invalid sampling batch width");
  const uint32_t lanes = static_cast<uint32_t>(policies.size());
  const bool constrained = std::any_of(
      policies.begin(), policies.end(),
      [](const SamplingPolicy &policy) { return policy.constrained; });
  const bool sampling = std::any_of(
      policies.begin(), policies.end(),
      [](const SamplingPolicy &policy) { return policy.samples(); });
  const bool greedy = std::any_of(
      policies.begin(), policies.end(),
      [](const SamplingPolicy &policy) { return !policy.samples(); });
  const bool distributed = constrained || sampling;
  const uint32_t rows = lanes * rowsPerLane_;

  if (!distributed) {
    graph.add("decode_sample_argmax_sharded",
              {buffers.logits, buffers.argmaxValues, buffers.argmaxIndices},
              vocabulary_, {uint64_t{rows} * kTargetShards, 1, 1});
    graph.add("decode_sample_argmax_reduce",
              {buffers.argmaxValues, buffers.argmaxIndices,
               buffers.outputTokens},
              {rows, 1, 1}, {32, 1, 1});
    return;
  }

  TargetSamplingBatchParams params{};
  params.vocabulary = vocabulary_;
  params.rows_per_lane = rowsPerLane_;
  params.lanes = lanes;
  params.mask_words = maskWords_;
  for (uint32_t lane = 0; lane < kMaximumLanes; ++lane) {
    const SamplingPolicy &policy = policies[std::min(lane, lanes - 1)];
    const EffectivePolicy effective = effectivePolicy(policy);
    params.top_k[lane] = effective.topK;
    params.temperature[lane] = effective.temperature;
    params.top_p[lane] = effective.topP;
    if (lane < lanes && policy.constrained)
      params.constrained_mask |= uint32_t{1} << lane;
  }
  graph.add("decode_sample_top32_sharded_batch",
            {buffers.logits, buffers.partialIds, buffers.partialValues,
             buffers.constraintMasks},
            params, {uint64_t{rows} * kTargetShards, 1, 1});
  graph.add("decode_sample_top32_probs_batch",
            {buffers.partialIds, buffers.partialValues, buffers.topIds,
             buffers.topProbabilities},
            params, {rows, 1, 1}, {1, 1, 1});
  // Acceptance consumes argmax tokens for greedy lanes and distributions
  // for sampling lanes, including when both share the same target forward.
  if (constrained || greedy) {
    graph.add("decode_sample_sparse_top1",
              {buffers.topIds, buffers.topProbabilities,
               buffers.outputTokens},
              {rows, 1, 1}, {1, 1, 1});
  }
}

void Sampling::addDraftSelector(
    metal::CommandGraph &graph, DraftSelectorBuffers buffers,
    std::span<const uint32_t> anchors,
    std::span<const SamplingPolicy> policies, uint32_t proposalTokens) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > kMaximumLanes || !proposalTokens)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.vocabulary = vocabulary_;
  for (uint32_t lane = 0; lane < kMaximumLanes; ++lane) {
    const uint32_t source = std::min(lane, lanes - 1);
    params.anchor[lane] = anchors[source];
    params.temperature[lane] = policies[source].temperature;
    if (lane < lanes && policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.add("draft_select_top16_sharded",
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            vocabulary_,
            {uint64_t{lanes} * proposalTokens * kDraftShards, 1, 1});
  graph.add("draft_select_edges",
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden,
             buffers.predecessorCodebook, buffers.successorCodebook},
            params, {uint64_t{lanes} * proposalTokens, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add("draft_select_dflash",
            {buffers.candidates, buffers.unary, buffers.partialValues,
             buffers.uniforms, buffers.proposedTokens,
             buffers.proposalProbabilities},
            params, {lanes, 1, 1}, {1, 1, 1});
}

void Sampling::addAcceptance(
    metal::CommandGraph &graph, AcceptanceBuffers buffers,
    std::span<const uint32_t> maximumRetained,
    std::span<const SamplingPolicy> policies, uint32_t stopToken0,
    uint32_t stopToken1) const {
  if (maximumRetained.empty() || maximumRetained.size() != policies.size() ||
      maximumRetained.size() > kMaximumLanes)
    throw std::invalid_argument("invalid DFlash acceptance batch");
  AcceptBatchParams params{};
  params.stop_token_0 = stopToken0;
  params.stop_token_1 = stopToken1;
  params.lanes = static_cast<uint32_t>(maximumRetained.size());
  for (uint32_t lane = 0; lane < params.lanes; ++lane) {
    if (!maximumRetained[lane] || maximumRetained[lane] > rowsPerLane_)
      throw std::invalid_argument("invalid DFlash retention limit");
    params.remaining[lane] = maximumRetained[lane];
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.add("decode_accept_dflash",
            {buffers.proposedTokens, buffers.candidates,
             buffers.proposalProbabilities, buffers.targetTopIds,
             buffers.targetTopProbabilities, buffers.uniforms,
             buffers.outputTokens, buffers.retainedCounts, buffers.nextAnchors,
             buffers.acceptedCounts},
            params, {params.lanes, 1, 1}, {1, 1, 1});
}

void Sampling::addVerifyInput(metal::CommandGraph &graph,
                              metal::MetalBuffer draftInputTokens,
                              metal::MetalBuffer proposedTokens,
                              metal::MetalBuffer verifyInputTokens,
                              uint32_t lanes) const {
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid verify input batch");
  const VerifyInputBatchParams params{lanes, vocabulary_};
  graph.add("verify_input_tokens",
            {std::move(draftInputTokens), std::move(proposedTokens),
             std::move(verifyInputTokens)},
            params, {uint64_t{lanes} * rowsPerLane_, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops
