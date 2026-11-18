#include "ops/Sampling.hpp"

#include "metal/abi/Sampling.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <limits>
#include <optional>
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

// The thinking cutoffs of a selection of up to `laneCap` rows a lane, their
// masks from mask row `maskRow` on, and the lanes up to the last one it cuts,
// 0 where it cuts none.
struct ThinkingCutoffs final {
  ThinkingCutoffParams params{};
  uint32_t cutLanes = 0;
};

ThinkingCutoffs thinkingCutoffs(std::span<const SamplingPolicy> policies, uint32_t vocabulary, uint32_t laneCap,
                                uint32_t maskRow) {
  ThinkingCutoffs cutoffs;
  ThinkingCutoffParams &params = cutoffs.params;
  params.vocabulary = vocabulary;
  params.mask_words = (vocabulary + 31) / 32;
  params.mask_row = maskRow;
  for (uint32_t lane = 0; lane < policies.size(); ++lane) {
    const SamplingPolicy &policy = policies[lane];
    const std::optional<ThinkingCutoff> &cutoff = policy.thinkingCutoff;
    params.row[lane] = cutoff ? cutoff->row : laneCap;
    if (params.row[lane] >= laneCap)
      continue;
    if (cutoff->end.empty())
      throw std::invalid_argument("a thinking cutoff has no end");
    if (cutoff->waitsForMask) {
      // Only a constrained lane has mask rows to read, which the kernel
      // reads at the end's first token.
      if (!policy.constrained)
        throw std::invalid_argument("an unconstrained lane's thinking end waits for a mask");
      if (cutoff->end.front() >= vocabulary)
        throw std::invalid_argument("a thinking end that waits for a mask starts outside the vocabulary");
      params.waiting_mask |= uint32_t{1} << lane;
    }
    // The rows from the cutoff's on take as many of its end as a lane's rows
    // hold.
    params.count[lane] = static_cast<uint32_t>(std::min<size_t>(cutoff->end.size(), laneCap - cutoff->row));
    std::copy_n(cutoff->end.begin(), params.count[lane], params.tokens[lane]);
    params.close[lane] = cutoff->end.back();
    params.also_ends[lane] = cutoff->alsoEnds;
    cutoffs.cutLanes = lane + 1;
  }
  return cutoffs;
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

SamplingPenaltyParams Sampling::penaltyParams(std::span<const SamplingPolicy> policies,
                                              const SamplingBuffers &buffers, const PenaltyTable &table,
                                              const TargetRows &rows) const {
  SamplingPenaltyParams params{};
  params.vocabulary = vocabulary_;
  const uint64_t rowBytes = uint64_t{vocabulary_} * sizeof(uint32_t);
  for (uint32_t lane = 0; lane < policies.size(); ++lane) {
    const SamplingPenalties &penalties = policies[lane].penalties;
    if (!penalties.active())
      continue;
    // The kernel indexes the whole table by this row.
    if (lane >= table.rows.size())
      throw std::invalid_argument("penalized lane has no penalty table row");
    requireBytes(table.words, (uint64_t{table.rows[lane]} + 1) * rowBytes, "penalty table");
    const uint32_t entry = params.entries++;
    params.lane[entry] = lane;
    params.table_row[entry] = table.rows[lane];
    params.repetition[entry] = penalties.repetition;
    // 1 / 2^-149 overflows; the saturated inverse keeps the product finite.
    params.repetition_inverse[entry] = std::min(
        1.0F / penalties.repetition, std::numeric_limits<float>::max());
    params.presence[entry] = penalties.presence;
    params.frequency[entry] = penalties.frequency;
  }
  // A lane's rows read their tree rows, and its rows after its first their
  // draft tokens.
  if (params.entries) {
    requireBytes(buffers.treeRows, rows.end * sizeof(TreeRow), "verify tree row");
    if (rows.laneCap > 1)
      requireBytes(buffers.inputTokens, rows.end * sizeof(uint32_t), "verify input token");
  }
  return params;
}

void Sampling::addPenalties(metal::CommandGraph &graph, const SamplingBuffers &buffers, const PenaltyTable &table,
                            const SamplingPenaltyParams &params) const {
  if (!params.entries)
    return;
  graph.add("decode_sample_penalize",
            {buffers.logits, table.words, buffers.inputTokens, buffers.treeRows, buffers.laneRows},
            params, {(vocabulary_ + kPenaltyThreads - 1) / kPenaltyThreads, params.entries, 1},
            {kPenaltyThreads, 1, 1});
}

void Sampling::addInitial(metal::CommandGraph &graph,
                          std::span<const SamplingPolicy> policies,
                          SamplingBuffers buffers, std::span<const uint32_t> rows,
                          const StopTokens &stopTokens,
                          const PenaltyTable &penalties) const {
  if (policies.empty() || policies.size() > kMaximumLanes || rows.size() != policies.size())
    throw std::invalid_argument("invalid sampling batch width");
  const uint32_t lanes = static_cast<uint32_t>(rows.size());
  SampleFirstRowsParams first{};
  uint64_t end = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    first.row[lane] = rows[lane];
    end = std::max(end, uint64_t{rows[lane]} + 1);
  }
  // A lane's one row reads its mask row 0.
  const TargetRows selected{end, 1, 0};
  const TargetSamplingParams params = selection(policies, buffers, selected, stopTokens);
  const SamplingPenaltyParams penalized = penaltyParams(policies, buffers, penalties, selected);
  // Each lane's one row is its tree's root.
  requireBytes(buffers.treeRows, end * sizeof(TreeRow), "verify tree row");
  graph.add("decode_sample_first_rows", {buffers.laneRows, buffers.treeRows}, first, {lanes, 1, 1}, {1, 1, 1});
  addPenalties(graph, buffers, penalties, penalized);
  addSelection(graph, params, policies, buffers, selected);
}

void Sampling::addVerify(metal::CommandGraph &graph,
                         std::span<const SamplingPolicy> policies,
                         SamplingBuffers buffers, const StepRows &step,
                         const StopTokens &stopTokens,
                         const PenaltyTable &penalties) const {
  requireStepRows(step);
  if (policies.size() != step.lanes)
    throw std::invalid_argument("invalid sampling batch width");
  // Verify row r of a lane reads mask row r + 1, and a sampled one tests its
  // draft's children.
  const TargetRows selected{step.rows, step.laneCap, 1, true};
  const TargetSamplingParams params = selection(policies, buffers, selected, stopTokens);
  addPenalties(graph, buffers, penalties, penaltyParams(policies, buffers, penalties, selected));
  addSelection(graph, params, policies, buffers, selected);
}

TargetSamplingParams Sampling::selection(std::span<const SamplingPolicy> policies,
                                         const SamplingBuffers &buffers, const TargetRows &rows,
                                         const StopTokens &stopTokens) const {
  // A lane's rows are those its mask rows, its uniforms and the cutoff's end
  // tokens hold.
  if (!rows.laneCap || rows.laneCap > SPLASH_TARGET_VERIFY_ROWS || rows.end > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("a lane's rows are out of range");
  TargetSamplingParams params{};
  params.vocabulary = vocabulary_;
  params.mask_words = maskWords_;
  params.mask_row = rows.maskRow;
  params.trees = rows.trees ? 1 : 0;
  std::ranges::copy(stopTokens.ids, params.stop_tokens);
  // The rows each policy's kernels reach, up to the last that reads each
  // buffer: the logits and output tokens of every lane, a workspace the
  // workspace() of the selection's rows where a lane of its kind selects, a
  // constrained lane's mask rows, and a sampled lane's uniform of each row.
  const uint32_t lanes = static_cast<uint32_t>(policies.size());
  uint64_t maskEnd = 0, uniformEnd = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const SamplingPolicy &policy = policies[lane];
    if (policy.samples()) {
      params.top_k[lane] = effectiveTopK(policy, vocabulary_);
      params.temperature[lane] = policy.temperature;
      params.top_p[lane] = policy.topP;
      params.min_p[lane] = policy.minP;
      params.sampling_mask |= uint32_t{1} << lane;
      // A verify row's child tests follow the lane's draw uniforms.
      uniformEnd = uint64_t{lane} * SPLASH_SAMPLING_UNIFORMS +
                   (rows.trees ? SPLASH_TARGET_VERIFY_ROWS + uint64_t{rows.laneCap} * SPLASH_TREE_CHILDREN
                               : rows.laneCap);
    }
    if (policy.constrained) {
      params.constrained_mask |= uint32_t{1} << lane;
      const uint64_t maskRows = uint64_t{lane} * (SPLASH_TARGET_VERIFY_ROWS + 1) + rows.maskRow + rows.laneCap;
      maskEnd = maskRows * maskWords_;
    }
    if (policy.excludesStopTokens)
      params.exclude_stop_mask |= uint32_t{1} << lane;
  }
  const uint64_t end = rows.end;
  const uint32_t all = (uint32_t{1} << lanes) - 1;
  const SamplingWorkspace none{}, every = workspace(static_cast<uint32_t>(end));
  const SamplingWorkspace &greedy = params.sampling_mask == all ? none : every;
  const SamplingWorkspace &sampled = params.sampling_mask ? every : none;
  requireBytes(buffers.logits, end * vocabulary_ * sizeof(float), "sampling logits");
  requireBytes(buffers.constraintMasks, maskEnd * sizeof(uint32_t), "constraint mask");
  requireBytes(buffers.outputTokens, end * sizeof(uint32_t), "sampled token");
  requireBytes(buffers.laneRows, uint64_t{lanes} * sizeof(LaneRows), "sampling lane rows");
  requireBytes(buffers.argmaxValues, greedy.argmaxValuesBytes, "argmax value");
  requireBytes(buffers.argmaxIndices, greedy.argmaxIndicesBytes, "argmax index");
  requireBytes(buffers.partialMasses, sampled.partialMassesBytes, "sampling mass");
  requireBytes(buffers.vocabularyRows, sampled.vocabularyRowsBytes, "sampling vocabulary row");
  requireBytes(buffers.vocabularyRanges, sampled.vocabularyRangesBytes, "sampling vocabulary range");
  requireBytes(buffers.vocabularyArrivals, sampled.vocabularyArrivalsBytes, "sampling arrival");
  requireBytes(buffers.uniforms, uniformEnd * sizeof(float), "sampling uniform");
  if (params.trees && params.sampling_mask) {
    const uint64_t treeRows = uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS;
    requireBytes(buffers.treeRows, end * sizeof(TreeRow), "verify tree row");
    requireBytes(buffers.verifySources, end * sizeof(uint32_t), "verify source");
    requireBytes(buffers.candidates,
                 uint64_t{lanes} * SPLASH_DRAFT_PROPOSAL_TOKENS * SPLASH_DRAFT_CANDIDATES * sizeof(uint32_t),
                 "draft candidate");
    requireBytes(buffers.treeDraws, treeRows * SPLASH_TREE_CHILDREN, "draft tree draw");
    requireBytes(buffers.treeNext, treeRows * SPLASH_DRAFT_CANDIDATES * sizeof(float), "draft tree next");
  }
  return params;
}

void Sampling::addSelection(metal::CommandGraph &graph, const TargetSamplingParams &params,
                            std::span<const SamplingPolicy> policies, const SamplingBuffers &buffers,
                            const TargetRows &rows) const {
  // Greedy and sampled lanes run their own kernels, each over a grid of a
  // lane's rows by lanes; the groups of the other kind's lanes, and those
  // past a lane's rows, return at once.
  const uint32_t lanes = static_cast<uint32_t>(policies.size());
  const uint32_t most = rows.laneCap;
  if (params.sampling_mask != (uint32_t{1} << lanes) - 1) {
    graph.add("decode_sample_argmax_sharded",
              {buffers.logits, buffers.constraintMasks, buffers.argmaxValues, buffers.argmaxIndices,
               buffers.laneRows},
              params, {most * kTargetShards, lanes, 1});
    graph.add("decode_sample_argmax_reduce",
              {buffers.argmaxValues, buffers.argmaxIndices, buffers.outputTokens, buffers.laneRows}, params,
              {most, lanes, 1}, {32, 1, 1});
  }
  if (params.sampling_mask) {
    graph.add("decode_sample_mass_sharded",
              {buffers.logits, buffers.constraintMasks, buffers.partialMasses, buffers.laneRows}, params,
              {most * kTargetShards, lanes, 1});
    graph.add("decode_sample_vocabulary_search",
              {buffers.logits, buffers.constraintMasks, buffers.partialMasses, buffers.vocabularyRows,
               buffers.laneRows},
              params, {most, lanes, 1}, {kVocabularyThreads, 1, 1});
    graph.add("decode_sample_vocabulary_draw",
              {buffers.logits, buffers.constraintMasks, buffers.vocabularyRows, buffers.uniforms,
               buffers.outputTokens, buffers.vocabularyRanges, buffers.vocabularyArrivals, buffers.laneRows,
               buffers.treeRows, buffers.verifySources, buffers.candidates, buffers.treeDraws, buffers.treeNext},
              params, {most * kVocabularyGroups, lanes, 1}, {kVocabularyThreads, 1, 1});
  }
  // Only a selection that reaches a lane's thinking cutoff dispatches it.
  const ThinkingCutoffs cutoffs = thinkingCutoffs(policies, vocabulary_, rows.laneCap, params.mask_row);
  if (!cutoffs.cutLanes)
    return;
  // A cut lane's rows after its first read their tokens and tree rows; a lane
  // whose end waits for its mask, a constrained one, reads its mask rows,
  // which selection() checked.
  if (rows.laneCap > 1) {
    requireBytes(buffers.inputTokens, rows.end * sizeof(uint32_t), "verify input token");
    requireBytes(buffers.treeRows, rows.end * sizeof(TreeRow), "verify tree row");
  }
  graph.add("decode_sample_thinking_cutoff",
            {buffers.inputTokens, buffers.constraintMasks, buffers.treeRows, buffers.outputTokens, buffers.laneRows},
            cutoffs.params, {lanes, 1, 1}, {1, 1, 1});
}

void Sampling::addAcceptance(metal::CommandGraph &graph, AcceptanceBuffers buffers, const StepRows &step,
                             std::span<const uint32_t> maximumRetained, const StopTokens &stopTokens) const {
  requireStepRows(step);
  if (maximumRetained.size() != step.lanes)
    throw std::invalid_argument("invalid DFlash acceptance batch");
  const uint32_t lanes = step.lanes;
  AcceptBatchParams params{};
  std::ranges::copy(stopTokens.ids, params.stop_tokens);
  // A lane retains at most its rows.
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (!maximumRetained[lane] || maximumRetained[lane] > step.laneCap)
      throw std::invalid_argument("invalid DFlash retention limit");
    params.remaining[lane] = maximumRetained[lane];
  }
  // A lane reads its rows' tokens, tree rows and selections.
  const uint64_t end = step.rows;
  requireBytes(buffers.inputTokens, end * sizeof(uint32_t), "verify input token");
  requireBytes(buffers.treeRows, end * sizeof(TreeRow), "verify tree row");
  requireBytes(buffers.outputTokens, end * sizeof(uint32_t), "target token");
  requireBytes(buffers.retainedCounts, lanes * sizeof(uint32_t), "retained count");
  requireBytes(buffers.acceptedCounts, lanes * sizeof(uint32_t), "accepted count");
  requireBytes(buffers.retainedRows, lanes * sizeof(uint32_t), "retained rows");
  requireBytes(buffers.laneRows, lanes * sizeof(LaneRows), "lane rows");
  graph.add("decode_accept_tree",
            {buffers.inputTokens, buffers.treeRows, buffers.outputTokens, buffers.retainedCounts,
             buffers.acceptedCounts, buffers.retainedRows, buffers.laneRows},
            params, {lanes, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops
