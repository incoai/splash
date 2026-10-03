#include "ops/Sampling.hpp"

#include "metal/abi/Sampling.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kTargetShards = SPLASH_TARGET_SAMPLING_SHARDS;
constexpr uint32_t kVocabularyThreads = SPLASH_TARGET_VOCABULARY_THREADS;
constexpr uint32_t kVocabularyGroups = SPLASH_TARGET_VOCABULARY_GROUPS;

// A sampled lane keeps its topK most likely tokens, and every token for 0 or
// a topK past the vocabulary (top-k disabled).
uint32_t effectiveTopK(const SamplingPolicy &policy,
                       uint32_t vocabulary) noexcept {
  return policy.topK && policy.topK < vocabulary ? policy.topK : vocabulary;
}
constexpr uint32_t kPenaltyThreads = 256;

void requireVocabulary(std::span<const uint32_t> tokens, size_t vocabulary) {
  if (std::any_of(tokens.begin(), tokens.end(),
                  [&](uint32_t token) { return token >= vocabulary; }))
    throw std::invalid_argument("penalty token is outside the vocabulary");
}

} // namespace

SamplingWorkspace Sampling::workspace(uint32_t rows) {
  if (!rows)
    throw std::invalid_argument("invalid sampling workspace row count");
  const uint64_t shards = uint64_t{rows} * kTargetShards;
  return {shards * sizeof(float),
          shards * sizeof(uint32_t),
          shards * sizeof(TargetShardMass),
          uint64_t{rows} * sizeof(TargetVocabularyRow),
          uint64_t{rows} * SPLASH_TARGET_VOCABULARY_RANGES *
              sizeof(TargetVocabularyRange),
          uint64_t{rows} * sizeof(uint32_t)};
}

void Sampling::rebuildPenaltyWords(std::span<uint32_t> words,
                                   std::span<const uint32_t> history,
                                   uint64_t generatedTokens,
                                   std::optional<uint32_t> pendingToken,
                                   bool markPrompt) {
  if (history.size() <= generatedTokens)
    throw std::logic_error("request history holds no prompt");
  const std::span<const uint32_t> prompt =
      history.first(history.size() - generatedTokens);
  requireVocabulary(history, words.size());
  if (pendingToken)
    requireVocabulary({&*pendingToken, 1}, words.size());
  std::fill(words.begin(), words.end(), 0U);
  if (markPrompt) {
    for (const uint32_t token : prompt)
      words[token] |= SPLASH_PENALTY_PROMPT_BIT;
  }
  for (const uint32_t token : history.subspan(prompt.size()))
    ++words[token];
  if (pendingToken)
    ++words[*pendingToken];
}

// Counts stay far below the prompt bit: a request selects at most one token
// per position of its context.
void Sampling::countPenaltyTokens(std::span<uint32_t> words,
                                  std::span<const uint32_t> selected) {
  requireVocabulary(selected, words.size());
  for (const uint32_t token : selected)
    ++words[token];
}

Sampling::Sampling(uint32_t vocabulary)
    : vocabulary_(vocabulary), maskWords_((vocabulary + 31) / 32) {
  if (!vocabulary)
    throw std::invalid_argument("invalid sampling geometry");
}

void Sampling::addPenalties(metal::CommandGraph &graph,
                            std::span<const SamplingPolicy> policies,
                            const SamplingBuffers &buffers,
                            const PenaltyTable &table, uint32_t rowOffset,
                            bool verify) const {
  SamplingPenaltyParams params{};
  params.vocabulary = vocabulary_;
  params.rows = verify ? SPLASH_TARGET_VERIFY_ROWS : 1;
  params.row_offset = rowOffset;
  const uint64_t rowBytes = uint64_t{vocabulary_} * sizeof(uint32_t);
  for (uint32_t lane = 0; lane < policies.size(); ++lane) {
    const SamplingPenalties &penalties = policies[lane].penalties;
    if (!penalties.active())
      continue;
    // The kernel indexes the whole table by this row.
    if (lane >= table.rows.size() ||
        (uint64_t{table.rows[lane]} + 1) * rowBytes > table.words.sizeBytes())
      throw std::invalid_argument("penalized lane has no penalty table row");
    const uint32_t entry = params.entries++;
    params.logits_lane[entry] = lane;
    params.table_row[entry] = table.rows[lane];
    params.repetition[entry] = penalties.repetition;
    // 1 / 2^-149 overflows; the saturated inverse keeps the product finite.
    params.repetition_inverse[entry] = std::min(
        1.0F / penalties.repetition, std::numeric_limits<float>::max());
    params.presence[entry] = penalties.presence;
    params.frequency[entry] = penalties.frequency;
  }
  if (!params.entries)
    return;
  const metal::DispatchSize groups{
      (vocabulary_ + kPenaltyThreads - 1) / kPenaltyThreads, params.entries, 1};
  if (!verify) {
    graph.add("decode_sample_penalize", {buffers.logits, table.words}, params,
              groups, {kPenaltyThreads, 1, 1});
    return;
  }
  graph.add("decode_sample_penalize_verify",
            {buffers.logits, table.words, buffers.inputTokens}, params, groups,
            {kPenaltyThreads, 1, 1});
}

void Sampling::addInitial(metal::CommandGraph &graph,
                          std::span<const SamplingPolicy> policies,
                          SamplingBuffers buffers, uint32_t rowOffset,
                          uint32_t stopToken0, uint32_t stopToken1,
                          const PenaltyTable &penalties) const {
  if (policies.empty() || policies.size() > kMaximumLanes)
    throw std::invalid_argument("invalid sampling batch width");
  if (rowOffset >= SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("invalid initial sampling row");
  addPenalties(graph, policies, buffers, penalties, rowOffset, false);
  addSelection(graph, policies, buffers,
               {1, rowOffset, 0, SPLASH_UNIFORM_INITIAL, 0}, stopToken0,
               stopToken1);
}

void Sampling::addVerify(metal::CommandGraph &graph,
                         std::span<const SamplingPolicy> policies,
                         SamplingBuffers buffers, uint32_t stopToken0,
                         uint32_t stopToken1,
                         const PenaltyTable &penalties) const {
  if (policies.empty() || policies.size() > kMaximumLanes)
    throw std::invalid_argument("invalid sampling batch width");
  addPenalties(graph, policies, buffers, penalties, 0, true);
  // Verify row r of a lane reads mask row r + 1 and, below the last row,
  // follows draft token r; every row draws with the lane's correction
  // uniform.
  addSelection(graph, policies, buffers,
               {SPLASH_TARGET_VERIFY_ROWS, 0, 1, SPLASH_UNIFORM_CORRECTION,
                SPLASH_DRAFT_PROPOSAL_TOKENS},
               stopToken0, stopToken1);
}

void Sampling::addSelection(metal::CommandGraph &graph,
                            std::span<const SamplingPolicy> policies,
                            const SamplingBuffers &buffers,
                            const TargetRows &rows, uint32_t stopToken0,
                            uint32_t stopToken1) const {
  TargetSamplingParams params{};
  params.vocabulary = vocabulary_;
  params.mask_words = maskWords_;
  params.rows = rows.rows;
  params.logits_row = rows.logitsRow;
  params.mask_row = rows.maskRow;
  params.uniform = rows.uniform;
  params.drafted_rows = rows.draftedRows;
  params.stop_token_0 = stopToken0;
  params.stop_token_1 = stopToken1;
  bool greedy = false;
  for (uint32_t lane = 0; lane < policies.size(); ++lane) {
    const SamplingPolicy &policy = policies[lane];
    if (policy.samples()) {
      params.top_k[lane] = effectiveTopK(policy, vocabulary_);
      params.temperature[lane] = policy.temperature;
      params.top_p[lane] = policy.topP;
      params.min_p[lane] = policy.minP;
      params.sampling_mask |= uint32_t{1} << lane;
    } else {
      greedy = true;
    }
    if (policy.constrained)
      params.constrained_mask |= uint32_t{1} << lane;
    if (policy.excludesStopTokens)
      params.exclude_stop_mask |= uint32_t{1} << lane;
  }
  // Greedy and sampled lanes run their own kernels, each over the selected
  // rows of every lane; the groups of the other kind's lanes return at once.
  const uint64_t selected = uint64_t{policies.size()} * rows.rows;
  if (greedy) {
    graph.add("decode_sample_argmax_sharded",
              {buffers.logits, buffers.constraintMasks, buffers.argmaxValues,
               buffers.argmaxIndices},
              params, {selected * kTargetShards, 1, 1});
    graph.add("decode_sample_argmax_reduce",
              {buffers.argmaxValues, buffers.argmaxIndices,
               buffers.outputTokens},
              params, {selected, 1, 1}, {32, 1, 1});
  }
  if (params.sampling_mask) {
    graph.add("decode_sample_mass_sharded",
              {buffers.logits, buffers.constraintMasks, buffers.partialMasses},
              params, {selected * kTargetShards, 1, 1});
    graph.add("decode_sample_vocabulary_search",
              {buffers.logits, buffers.constraintMasks, buffers.partialMasses,
               buffers.vocabularyRows},
              params, {selected, 1, 1}, {kVocabularyThreads, 1, 1});
    graph.add("decode_sample_vocabulary_draw",
              {buffers.logits, buffers.constraintMasks, buffers.vocabularyRows,
               buffers.inputTokens, buffers.draftCandidates,
               buffers.draftProbabilities, buffers.uniforms,
               buffers.outputTokens, buffers.vocabularyRanges,
               buffers.vocabularyArrivals},
              params, {selected * kVocabularyGroups, 1, 1},
              {kVocabularyThreads, 1, 1});
  }
}

void Sampling::addAcceptance(
    metal::CommandGraph &graph, AcceptanceBuffers buffers,
    std::span<const uint32_t> maximumRetained,
    std::span<const SamplingPolicy> policies, uint32_t stopToken0,
    uint32_t stopToken1) const {
  if (maximumRetained.empty() || maximumRetained.size() != policies.size() ||
      maximumRetained.size() > kMaximumLanes)
    throw std::invalid_argument("invalid DFlash acceptance batch");
  const uint32_t lanes = static_cast<uint32_t>(maximumRetained.size());
  AcceptBatchParams params{};
  params.stop_token_0 = stopToken0;
  params.stop_token_1 = stopToken1;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (!maximumRetained[lane] ||
        maximumRetained[lane] > SPLASH_TARGET_VERIFY_ROWS)
      throw std::invalid_argument("invalid DFlash retention limit");
    params.remaining[lane] = maximumRetained[lane];
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.add("decode_accept_dflash",
            {buffers.proposedTokens, buffers.candidates,
             buffers.proposalProbabilities, buffers.targetVocabularyRows,
             buffers.uniforms, buffers.outputTokens, buffers.retainedCounts,
             buffers.acceptedCounts},
            params, {lanes, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops
