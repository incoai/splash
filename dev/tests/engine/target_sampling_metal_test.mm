// Target sampling policy against a direct CPU reference.
//
// The penalty kernels must rewrite exactly the logits of the penalized lanes'
// prompt and output tokens, within float rounding of the host arithmetic,
// from each lane's own penalty table row, counting at verify row r the draft
// tokens of rows 1..r; every other logit stays bitwise. The policy then
// selects from them, among the tokens each lane admits: a greedy row takes
// its argmax, and a sampled row draws from its distribution over the whole
// vocabulary (the tokens min_p leaves, the top-k of those, then top-p of
// their renormalized mass). Its largest logit, its draft token's
// probability and its draw must match an evaluation of the same rules in
// double: the draw must fall where the reference distribution's cumulative
// sum in token order places the uniform. DFlash acceptance must accept and
// correct as a sequential decode would. Data within float rounding of a
// min_p or top_p cut or of a draw's boundary fail as ambiguous, or are
// allowed either way, instead of passing by chance. A lane selects the same,
// bit for bit, whatever lanes share its batch; a distribution of one token
// selects the masked argmax; and a lane that ignores end-of-sequence never
// selects a stop token.
// Extreme repetition penalties saturate to exact, finite outcomes, and rows
// of non-finite logits select the sentinel 0xFFFFFFFF. The host
// word helpers and the lifecycle that rebuilds a resumed request's words are
// checked bitwise.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include "metal/abi/Sampling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kUniforms = SPLASH_SAMPLING_UNIFORMS;
constexpr uint32_t kDraftCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr float kFloatMax = std::numeric_limits<float>::max();
// Both stop tokens sit below every spike of fillRow.
constexpr std::array<uint32_t, 2> kStopTokens{1, 2};

using splash::test::rejects;
using splash::test::require;

// Stop tokens in the first shard of the vocabulary and in the last, so a
// lane that excludes them skips them in both.
std::array<uint32_t, 2> shardEdgeStopTokens(uint32_t vocabulary) {
  return {1, vocabulary - 2};
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  // Uniform in [-1, 1).
  float unit() {
    return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F;
  }
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }

private:
  uint64_t state_;
};

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, "target sampling");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

struct Batch final {
  SamplingBuffers buffers;
  uint32_t vocabulary = 0;
  uint32_t rows = 0;

  [[nodiscard]] float *logits() const {
    return static_cast<float *>(buffers.logits.contents());
  }
  [[nodiscard]] float *row(uint32_t index) const {
    return logits() + uint64_t{index} * vocabulary;
  }
  [[nodiscard]] const uint32_t *outputTokens() const {
    return static_cast<const uint32_t *>(buffers.outputTokens.contents());
  }
  [[nodiscard]] uint32_t *inputTokens() const {
    return static_cast<uint32_t *>(buffers.inputTokens.contents());
  }
  [[nodiscard]] uint32_t *masks() const {
    return static_cast<uint32_t *>(buffers.constraintMasks.contents());
  }
  [[nodiscard]] float *uniforms() const {
    return static_cast<float *>(buffers.uniforms.contents());
  }
  [[nodiscard]] const TargetVocabularyRow &record(uint32_t index) const {
    return static_cast<const TargetVocabularyRow *>(
        buffers.vocabularyRows.contents())[index];
  }
  [[nodiscard]] uint32_t maskWords() const { return (vocabulary + 31) / 32; }
  // Outputs a missing dispatch would leave behind cannot pass as results.
  // The arrival counts stay at zero, where every draw leaves them.
  void poison() const {
    for (const auto &buffer :
         {buffers.partialMasses, buffers.vocabularyRows,
          buffers.vocabularyRanges, buffers.argmaxValues,
          buffers.argmaxIndices, buffers.outputTokens})
      std::memset(buffer.contents(), 0xA5, buffer.sizeBytes());
  }
};

Batch makeBatch(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes) {
  const uint32_t rows = lanes * kRows;
  const auto space = Sampling::workspace(rows);
  const auto proposals = DraftSelector::workspace(lanes * kPositions);
  return {SamplingBuffers{
              allocate(backend, uint64_t{rows} * vocabulary * sizeof(float)),
              allocate(backend, space.partialMassesBytes),
              allocate(backend, space.vocabularyRowsBytes),
              allocate(backend, uint64_t{lanes} * kUniforms * sizeof(float)),
              allocate(backend,
                       uint64_t{lanes} * (kRows + 1) * ((vocabulary + 31) / 32) * 4),
              allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
              allocate(backend, space.argmaxValuesBytes),
              allocate(backend, space.argmaxIndicesBytes),
              allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
              allocate(backend, proposals.candidatesBytes),
              allocate(backend, proposals.proposalProbabilitiesBytes),
              allocate(backend, space.vocabularyRangesBytes),
              allocate(backend, space.vocabularyArrivalsBytes)},
          vocabulary, rows};
}

// Low noise with forty distinct spikes: the production shape, whose best
// tokens hold no ties. The stop tokens stay below every spike.
void fillRow(float *row, uint32_t vocabulary, Random &random) {
  for (uint32_t token = 0; token < vocabulary; ++token)
    row[token] = -8.0F + 3.0F * random.unit() * random.unit();
  for (uint32_t spike = 0; spike < 40; ++spike) {
    const uint32_t token = 3 + random.next() % (vocabulary - 3);
    row[token] = 1.0F + 0.25F * float(spike) + 0.1F * random.unit();
  }
}

// The tokens a row may select: those its constraint mask row allows, if it
// has one, less the stop tokens when its lane ignores end-of-sequence.
struct Admission final {
  const uint32_t *mask = nullptr;
  bool excludesStop = false;

  [[nodiscard]] bool operator()(uint32_t token) const {
    if (mask && !(mask[token / 32] & (1U << (token % 32))))
      return false;
    return !excludesStop ||
           (token != kStopTokens[0] && token != kStopTokens[1]);
  }
};

// The row's best count admitted tokens: value descending, id ascending.
std::vector<uint32_t> referenceBest(const float *row, uint32_t vocabulary,
                                    uint32_t count, Admission admits = {}) {
  std::vector<uint32_t> order;
  for (uint32_t token = 0; token < vocabulary; ++token)
    if (admits(token))
      order.push_back(token);
  const auto beats = [&](uint32_t a, uint32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
  };
  const size_t keep = std::min<size_t>(count, order.size());
  std::partial_sort(order.begin(), order.begin() + keep, order.end(), beats);
  order.resize(keep);
  return order;
}

uint32_t referenceArgmax(const float *row, uint32_t vocabulary,
                         Admission admits = {}) {
  return referenceBest(row, vocabulary, 1, admits).front();
}

// How close a top_p cut may lie to a token's preceding mass, relative to the
// top-k mass, before float sums over the whole vocabulary could keep or cut
// that token either way.
constexpr double kTopPMargin = 1e-5;
// How close a logit may lie to the min_p cut, relative to the larger of the
// cut and the row's maximum, before the float cut could keep or drop it.
constexpr double kMinPMargin = 1e-5;

// A row's distribution: its admitted tokens in the order of the logits
// (value descending, id ascending); those that weigh at least min_p of the
// heaviest, which are the ones whose logits are at least max + T * log(min_p);
// the first top_k of them (all of them for 0 or a top_k past the
// vocabulary); then those whose preceding mass is at most top_p of the mass
// kept so far, each weighing exp((logit - max) / T). margin is the distance
// of the top_p target from the nearest preceding mass, relative to that
// mass; the tokens within kTopPMargin of it, and those within kMinPMargin of
// the min_p cut that top_k would keep, are ambiguous.
struct Distribution final {
  std::vector<uint32_t> order;
  float maximum = 0.0F;
  double mass = 0.0;
  double margin = 1.0;
  double ambiguousMass = 0.0;
  std::vector<double> probabilities;
  std::vector<bool> ambiguous;

  [[nodiscard]] double probability(uint32_t token) const {
    return token < probabilities.size() ? probabilities[token] : 0.0;
  }
};

Distribution referenceDistribution(const float *row, uint32_t vocabulary,
                                   const SamplingPolicy &policy,
                                   Admission admits = {}) {
  Distribution result;
  std::vector<uint32_t> &order = result.order;
  for (uint32_t token = 0; token < vocabulary; ++token)
    if (admits(token))
      order.push_back(token);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
  });
  result.maximum = row[order.front()];
  const double maximum = result.maximum;
  const auto weight = [&](uint32_t token) {
    return std::exp((double(row[token]) - maximum) / policy.temperature);
  };
  result.ambiguous.assign(vocabulary, false);
  if (policy.minP > 0.0F) {
    const double cut =
        maximum + double(policy.temperature) * std::log(double(policy.minP));
    const double tolerance =
        kMinPMargin * std::max({1.0, std::fabs(cut), std::fabs(maximum)});
    size_t kept = 0;
    for (size_t rank = 0;
         rank < order.size() && row[order[rank]] >= cut - tolerance; ++rank) {
      if (row[order[rank]] >= cut)
        kept = rank + 1;
      // The heaviest token and its ties always stay.
      if (std::fabs(row[order[rank]] - cut) <= tolerance &&
          row[order[rank]] != result.maximum &&
          (!policy.topK || rank < policy.topK)) {
        result.ambiguous[order[rank]] = true;
        result.ambiguousMass += weight(order[rank]);
      }
    }
    order.resize(kept);
  }
  if (policy.topK && policy.topK < order.size())
    order.resize(policy.topK);
  double topMass = 0.0;
  for (const uint32_t token : order)
    topMass += weight(token);
  if (policy.topP < 1.0F) {
    // The tokens min_p may keep or drop move the mass top_p measures against.
    const double slack = kTopPMargin + result.ambiguousMass / topMass;
    const double target = double(policy.topP) * topMass;
    size_t kept = order.size();
    double before = 0.0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
      const double distance = std::fabs(before - target) / topMass;
      result.margin = std::min(result.margin, distance);
      if (distance <= slack) {
        result.ambiguous[order[rank]] = true;
        result.ambiguousMass += weight(order[rank]);
      }
      if (before > target && kept == order.size())
        kept = rank;
      if (before > target * (1.0 + slack) + slack * topMass)
        break;
      before += weight(order[rank]);
    }
    order.resize(kept);
  }
  result.probabilities.assign(vocabulary, 0.0);
  for (const uint32_t token : order)
    result.mass += weight(token);
  for (const uint32_t token : order)
    result.probabilities[token] = weight(token) / result.mass;
  result.ambiguousMass /= result.mass;
  return result;
}

// The draw a sampled row must make: the token whose interval of the weights'
// cumulative sum, in token order, holds uniform times their total, within
// the float rounding of the GPU's sums and the mass of the tokens the
// distribution may keep or cut (slack, relative to the total).
void requireDraw(uint32_t token, const std::vector<double> &weights,
                 double uniform, double slack, const std::string &label) {
  const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
  const double target = uniform * total;
  const double tolerance = (1e-5 + slack) * total;
  require(token < weights.size() && (weights[token] > 0.0 || slack > 0.0),
          label + ": drew token " + std::to_string(token) +
              ", which has no weight");
  double before = 0.0;
  for (uint32_t other = 0; other < token; ++other)
    before += weights[other];
  require(before - tolerance <= target &&
              target <= before + weights[token] + tolerance,
          label + ": drew token " + std::to_string(token) + " whose interval [" +
              std::to_string(before / total) + ", " +
              std::to_string((before + weights[token]) / total) +
              ") misses " + std::to_string(uniform));
}

// A rejected draft token's residual: each token's probability less its
// draft probability (the first entry for a token listed twice), never below
// zero, or the distribution itself when nothing remains.
std::vector<double> residualWeights(const Distribution &target,
                                    const uint32_t *candidates,
                                    const float *proposal) {
  std::vector<double> weights = target.probabilities;
  for (uint32_t index = 0; index < kDraftCandidates; ++index) {
    const uint32_t token = candidates[index];
    if (token >= weights.size() ||
        std::find(candidates, candidates + index, token) != candidates + index)
      continue;
    weights[token] = std::max(weights[token] - double(proposal[index]), 0.0);
  }
  if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0)
    return target.probabilities;
  return weights;
}

// A row's draw returns its arrival count to zero, where the next one starts.
void requireArrivalsReturned(const Batch &batch, uint32_t row,
                             const std::string &label) {
  require(static_cast<const uint32_t *>(
              batch.buffers.vocabularyArrivals.contents())[row] == 0,
          label + ": the draw left its arrival count");
}

// What the search records of a sampled row besides its end: its largest
// admitted logit.
void requireMaximum(const TargetVocabularyRow &record,
                    const Distribution &target, const std::string &label) {
  require(record.maximum == target.maximum,
          label + ": the scan's largest logit differs");
}

// One verify row of a sampled lane: its largest logit, the target
// probability of its draft token and the correction a rejection takes, or,
// for the last row, the bonus token.
void requireSampledRow(const Batch &batch, uint32_t index,
                       const Distribution &target, const std::string &label) {
  const uint32_t row = index % kRows;
  const TargetVocabularyRow &record = batch.record(index);
  requireArrivalsReturned(batch, index, label);
  requireMaximum(record, target, label);
  const uint32_t lane = index / kRows;
  const float uniform =
      batch.uniforms()[lane * kUniforms + SPLASH_UNIFORM_CORRECTION];
  const uint32_t token = batch.outputTokens()[index];
  if (row == kPositions) {
    requireDraw(token, target.probabilities, uniform, target.ambiguousMass,
                label + " bonus");
    return;
  }
  const uint32_t draft = batch.inputTokens()[index + 1];
  const double expected = target.probability(draft);
  require(target.ambiguous[draft] ||
              (expected == 0.0
                   ? record.draft_probability == 0.0F
                   : std::fabs(record.draft_probability - expected) <=
                         (1e-4 + target.ambiguousMass) * expected),
          label + ": draft token probability " +
              std::to_string(record.draft_probability) + " for " +
              std::to_string(expected));
  const uint64_t position = uint64_t{lane} * kPositions + row;
  requireDraw(token,
              residualWeights(
                  target,
                  static_cast<const uint32_t *>(
                      batch.buffers.draftCandidates.contents()) +
                      position * kDraftCandidates,
                  static_cast<const float *>(
                      batch.buffers.draftProbabilities.contents()) +
                      position * kDraftCandidates),
              uniform, target.ambiguousMass, label + " correction");
}

// The first token after a prompt of a sampled lane, drawn from its row with
// the lane's first uniform; a selection writes one per lane.
void requireInitialDraw(const Batch &batch, uint32_t lane,
                        const Distribution &target, const std::string &label) {
  requireArrivalsReturned(batch, lane, label);
  requireMaximum(batch.record(lane), target, label);
  requireDraw(batch.outputTokens()[lane], target.probabilities,
              batch.uniforms()[lane * kUniforms + SPLASH_UNIFORM_INITIAL],
              target.ambiguousMass, label);
}

// The penalty kernel's arithmetic on the host: the logit of a token the
// output holds count times, or the prompt holds.
float referencePenalty(float value, uint32_t count, bool prompt,
                       const SamplingPenalties &penalties) {
  if (!count && !prompt)
    return value;
  value *= value > 0.0F
               ? std::min(1.0F / penalties.repetition, kFloatMax)
               : penalties.repetition;
  if (count)
    value -= penalties.frequency * float(count) + penalties.presence;
  return std::clamp(value, -kFloatMax, kFloatMax);
}

// A penalized logit as the kernel must write it. The scaling is one IEEE
// product on both sides, so a prompt-only token matches bitwise; the GPU may
// fuse frequency * count + presence, which leaves the difference within an
// ulp of the larger operand of the final subtraction.
bool penaltyMatches(float actual, float value, uint32_t count, bool prompt,
                    const SamplingPenalties &penalties) {
  const float expected = referencePenalty(value, count, prompt, penalties);
  if (!count || std::fabs(expected) == kFloatMax)
    return std::memcmp(&actual, &expected, sizeof(float)) == 0;
  const float subtrahend =
      penalties.frequency * float(count) + penalties.presence;
  const float magnitude = std::max(
      {std::fabs(expected + subtrahend), std::fabs(subtrahend),
       std::fabs(expected)});
  const float ulp = std::nextafter(magnitude, INFINITY) - magnitude;
  return std::fabs(actual - expected) <= 2.0F * ulp;
}

std::span<uint32_t> tableRow(const MetalBuffer &table, uint32_t vocabulary,
                             uint32_t row) {
  return {static_cast<uint32_t *>(table.contents()) + uint64_t{row} * vocabulary,
          vocabulary};
}

// A table of kLanes state lanes, each row with its own words: prompt bits at
// a density of its own, small counts, and a few counts at the 2^20-token
// output limit.
MetalBuffer penaltyTable(MetalBackend &backend, uint32_t vocabulary) {
  MetalBuffer table = allocate(backend, uint64_t{kLanes} * vocabulary * 4);
  for (uint32_t slot = 0; slot < kLanes; ++slot) {
    Random random(0x7461626c65 + slot);
    std::span<uint32_t> words = tableRow(table, vocabulary, slot);
    for (uint32_t token = 0; token < vocabulary; ++token) {
      const uint32_t draw = random.next() % 64;
      uint32_t word = draw < 8 + 6 * slot ? SPLASH_PENALTY_PROMPT_BIT : 0U;
      if (draw % 3 == 0)
        word += 1 + draw % (slot + 3);
      if (draw == 63 && token % 7 == 0)
        word = (word & SPLASH_PENALTY_PROMPT_BIT) + (1U << 20);
      words[token] = word;
    }
  }
  return table;
}

// The draft tokens verify row r's context adds: the token count among the
// lane's verify input rows 1..r.
uint32_t draftedCount(const uint32_t *inputs, uint32_t row, uint32_t token) {
  uint32_t count = 0;
  for (uint32_t input = 1; input <= row; ++input)
    count += inputs[input] == token ? 1U : 0U;
  return count;
}

// Checks every logit of a lane's rows [first, first + rows) against the
// originals: penalized from the lane's words, or bitwise unchanged.
void requirePenalizedRows(const Batch &batch, const std::vector<float> &original,
                          uint32_t lane, uint32_t first, uint32_t rows,
                          std::span<const uint32_t> words,
                          const SamplingPenalties &penalties, bool verify,
                          const std::string &label) {
  const uint32_t vocabulary = batch.vocabulary;
  const uint32_t *inputs = batch.inputTokens() + uint64_t{lane} * kRows;
  for (uint32_t row = first; row < first + rows; ++row) {
    const uint64_t origin = (uint64_t{lane} * kRows + row) * vocabulary;
    for (uint32_t token = 0; token < vocabulary; ++token) {
      const float before = original[origin + token];
      const float after = batch.logits()[origin + token];
      const uint32_t word = words.empty() ? 0U : words[token];
      const uint32_t count = (word & SPLASH_PENALTY_COUNT_MASK) +
                             (verify ? draftedCount(inputs, row, token) : 0U);
      const bool prompt = (word & SPLASH_PENALTY_PROMPT_BIT) != 0;
      require(penalties.active()
                  ? penaltyMatches(after, before, count, prompt, penalties)
                  : std::memcmp(&after, &before, sizeof(float)) == 0,
              label + ": row " + std::to_string(row) + " token " +
                  std::to_string(token) + " wrote " + std::to_string(after) +
                  " for " + std::to_string(before));
    }
  }
}

// Penalized lanes read table rows that are neither their lane indices nor
// within the batch width. Plan lanes do not follow state lanes.
constexpr std::array<std::array<uint32_t, kLanes>, kLanes> kTableRows{
    {{3}, {2, 0}, {1, 3, 0}, {2, 0, 3, 1}}};

// A sampled lane with every penalty, a constrained greedy lane with
// presence alone, an unpenalized sampled lane that ignores end-of-sequence,
// and a greedy lane whose repetition below 1 raises repeated tokens.
const std::array<SamplingPolicy, kLanes> kMixedPolicies{
    SamplingPolicy{32, 0.8F, 0.95F, false, false, {1.3F, 0.5F, 0.25F}},
    SamplingPolicy{1, 0.0F, 1.0F, true, false, {1.0F, 1.5F, 0.0F}},
    SamplingPolicy{20, 1.2F, 0.9F, false, true, {}},
    SamplingPolicy{1, 0.0F, 1.0F, false, false, {0.8F, -2.0F, 2.0F}}};

// Penalized logits of every verify row and of the first token at an offset,
// and the selections the policy makes from them, at B1-B4 with penalized and
// unpenalized lanes side by side. With greedy lanes only, the batch runs the
// argmax kernels alone.
void penalties(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes,
               bool argmaxPath, uint32_t &changedSelections) {
  Random random(0x70656e + uint64_t{vocabulary} * 8 + lanes * 2 + argmaxPath);
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, lanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  for (uint32_t row = 0; row < batch.rows; ++row) {
    fillRow(batch.row(row), vocabulary, random);
    // A stop token leads every other row, so a lane that ignores
    // end-of-sequence has something to skip.
    if (row % 2)
      batch.row(row)[kStopTokens[0]] = 13.0F;
  }
  for (uint32_t index = 0; index < lanes * (kRows + 1) * batch.maskWords();
       ++index)
    batch.masks()[index] = 0xB6DB6DB6U ^ index;
  for (uint32_t uniform = 0; uniform < lanes * kUniforms; ++uniform)
    batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
  std::vector<SamplingPolicy> policies;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    SamplingPolicy policy = kMixedPolicies[lane];
    if (argmaxPath)
      policy = {1, 0.0F, 1.0F, false, false,
                kMixedPolicies[lane == 2 ? 0 : lane].penalties};
    policies.push_back(policy);
    // The draft repeats a token, and its tokens are among the row's best, so
    // the counts they add move selections.
    const std::vector<uint32_t> best =
        referenceBest(batch.row(lane * kRows), vocabulary, 10);
    const std::array<uint32_t, kRows> inputs{
        best[9], best[0], best[1], best[0], best[2], best[3], best[1], best[4]};
    std::copy(inputs.begin(), inputs.end(),
              batch.inputTokens() + uint64_t{lane} * kRows);
  }
  const std::span<const uint32_t> rows =
      std::span(kTableRows[lanes - 1]).first(lanes);
  const std::vector<float> original(batch.logits(),
                                    batch.logits() + uint64_t{batch.rows} * vocabulary);
  const std::string label = std::string(argmaxPath ? "argmax" : "mixed") +
                            " penalties B" + std::to_string(lanes) +
                            " vocabulary " + std::to_string(vocabulary);
  const auto words = [&](uint32_t lane) {
    return tableRow(table, vocabulary, rows[lane]);
  };

  batch.poison();
  CommandGraph verify;
  sampling.addVerify(verify, policies, batch.buffers, kStopTokens[0],
                     kStopTokens[1], {table, rows});
  static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requirePenalizedRows(batch, original, lane, 0, kRows, words(lane),
                         policies[lane].penalties, true, label + " verify");
    for (uint32_t row = lane * kRows; row < (lane + 1) * kRows; ++row) {
      const SamplingPolicy &policy = policies[lane];
      const Admission admits{
          policy.constrained
              ? batch.masks() + (uint64_t{lane} * (kRows + 1) + row % kRows + 1) *
                                    batch.maskWords()
              : nullptr,
          policy.excludesStopTokens};
      if (policy.samples()) {
        requireSampledRow(batch, row,
                          referenceDistribution(batch.row(row), vocabulary,
                                                policy, admits),
                          label + " verify row " + std::to_string(row));
        continue;
      }
      const uint32_t token = batch.outputTokens()[row];
      require(token == referenceArgmax(batch.row(row), vocabulary, admits),
              label + ": a greedy verify row lost its penalized argmax");
      const float *unpenalized = original.data() + uint64_t{row} * vocabulary;
      changedSelections +=
          token != referenceArgmax(unpenalized, vocabulary, admits) ? 1U : 0U;
    }
  }

  // The first token after each lane's prompt: one row per lane at an
  // offset, no draft tokens; every other row keeps its logits.
  constexpr uint32_t kOffset = 5;
  std::copy(original.begin(), original.end(), batch.logits());
  batch.poison();
  CommandGraph initial;
  sampling.addInitial(initial, policies, batch.buffers, kOffset,
                      kStopTokens[0], kStopTokens[1], {table, rows});
  static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const SamplingPolicy &policy = policies[lane];
    const std::string lanePrefix = label + " initial lane " + std::to_string(lane);
    requirePenalizedRows(batch, original, lane, kOffset, 1, words(lane),
                         policy.penalties, false, lanePrefix);
    for (uint32_t row = 0; row < kRows; ++row)
      if (row != kOffset)
        requirePenalizedRows(batch, original, lane, row, 1, {}, {}, false,
                             lanePrefix + ", unselected row");
    const float *selected = batch.row(lane * kRows + kOffset);
    const Admission admits{
        policy.constrained
            ? batch.masks() + uint64_t{lane} * (kRows + 1) * batch.maskWords()
            : nullptr,
        policy.excludesStopTokens};
    if (policy.samples())
      requireInitialDraw(
          batch, lane,
          referenceDistribution(selected, vocabulary, policy, admits),
          lanePrefix);
    else
      require(batch.outputTokens()[lane] ==
                  referenceArgmax(selected, vocabulary, admits),
              lanePrefix + ": the initial token lost its penalized argmax");
  }
}

// A request's DFlash cycle against a sequential decode. Every verify row of
// a lane has nearly the same logits, as a model that would repeat itself,
// so only the draft tokens its context adds keep row r from selecting what
// row 0 did. Row r must select from the distribution a non-speculative decode
// would, given the context through draft token r - 1; the drafts are what
// that decode selects up to a lane's first wrong one, and acceptance must
// stop there with the same correction and next anchor. Sampled lanes accept
// a one-hot draft token while its uniform is below the token's penalized
// probability.
void speculativeExactness(MetalBackend &backend, uint32_t samplingMask) {
  constexpr uint32_t vocabulary = 1003;
  constexpr uint32_t lanes = 2;
  constexpr std::array<uint32_t, lanes> kSlots{3, 1};
  // Lane 0 drafts its first wrong token at position 3; lane 1 drafts every
  // token right and takes the bonus row.
  constexpr std::array<uint32_t, lanes> kWrongAt{3, kPositions};
  const std::array<SamplingPenalties, lanes> penalties{
      SamplingPenalties{1.2F, 2.0F, 2.0F}, SamplingPenalties{1.0F, 1.5F, 0.0F}};
  Random random(0x73706563 + samplingMask);
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, lanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  AcceptanceBuffers acceptance{
      allocate(backend, uint64_t{lanes} * kPositions * sizeof(uint32_t)),
      batch.buffers.draftCandidates,
      batch.buffers.draftProbabilities,
      batch.buffers.vocabularyRows,
      batch.buffers.uniforms,
      batch.buffers.outputTokens,
      allocate(backend, lanes * sizeof(uint32_t)),
      allocate(backend, lanes * sizeof(uint32_t))};
  auto *proposed = static_cast<uint32_t *>(acceptance.proposedTokens.contents());
  auto *candidates = static_cast<uint32_t *>(acceptance.candidates.contents());
  auto *proposal =
      static_cast<float *>(acceptance.proposalProbabilities.contents());
  std::fill(candidates, candidates + lanes * kPositions * kDraftCandidates,
            0xFFFFFFFFU);

  std::vector<SamplingPolicy> policies;
  std::array<std::array<uint32_t, kRows>, lanes> expectedOutput{};
  std::array<uint32_t, lanes> expectedAccepted{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const bool sampled = (samplingMask >> lane) & 1U;
    policies.push_back({sampled ? 32U : 1U, sampled ? 0.9F : 0.0F, 0.9F, false,
                        false, penalties[lane]});
    std::span<const uint32_t> words = tableRow(table, vocabulary, kSlots[lane]);
    std::vector<uint32_t> counts(vocabulary);
    for (uint32_t token = 0; token < vocabulary; ++token)
      counts[token] = words[token] & SPLASH_PENALTY_COUNT_MASK;
    uint32_t *inputs = batch.inputTokens() + lane * kRows;
    float *uniforms = batch.uniforms() + lane * kUniforms;
    inputs[0] = 5 + lane;
    std::vector<float> base(vocabulary);
    fillRow(base.data(), vocabulary, random);
    std::vector<Distribution> targets;
    std::vector<uint32_t> selections;
    for (uint32_t row = 0; row < kRows; ++row) {
      float *logits = batch.row(lane * kRows + row);
      for (uint32_t token = 0; token < vocabulary; ++token)
        logits[token] = base[token] + 0.01F * random.unit();
      // What a sequential decode selects from at this position.
      std::vector<float> penalized(logits, logits + vocabulary);
      for (uint32_t token = 0; token < vocabulary; ++token)
        penalized[token] = referencePenalty(
            logits[token], counts[token],
            (words[token] & SPLASH_PENALTY_PROMPT_BIT) != 0, penalties[lane]);
      const std::vector<uint32_t> order =
          referenceBest(penalized.data(), vocabulary, 8);
      require(penalized[order[0]] - penalized[order[1]] > 1e-4F,
              "test data is ambiguous at a greedy selection");
      targets.push_back(sampled ? referenceDistribution(penalized.data(),
                                                        vocabulary,
                                                        policies[lane])
                                : Distribution{});
      selections.push_back(order[0]);
      if (row == kPositions)
        break;
      // The draft: the best penalized candidate until the wrong position,
      // where it takes the eighth best, whose probability is small.
      const uint32_t draft = row < kWrongAt[lane] ? order[0] : order[7];
      inputs[row + 1] = draft;
      proposed[lane * kPositions + row] = draft;
      candidates[(lane * kPositions + row) * kDraftCandidates] = draft;
      proposal[(lane * kPositions + row) * kDraftCandidates] = 1.0F;
      ++counts[draft];
      const double p = targets.back().probability(draft);
      // Accept a right token surely and refuse the wrong one.
      uniforms[SPLASH_UNIFORM_ACCEPTANCE + row] =
          row < kWrongAt[lane] ? float(p * 0.5) : float(std::min(1.0, p * 2.0 + 0.01));
      require(!sampled || row >= kWrongAt[lane] || p > 1e-4,
              "a sampled lane drafted a token its target cannot accept");
    }
    uniforms[SPLASH_UNIFORM_CORRECTION] = 0.37F;

    // Acceptance as accept_greedy_lane and accept_sampled_lane define it.
    const uint32_t accepted = kWrongAt[lane];
    expectedAccepted[lane] = accepted;
    for (uint32_t position = 0; position < accepted; ++position)
      expectedOutput[lane][position] = inputs[position + 1];
    if (!sampled) {
      expectedOutput[lane][accepted] = selections[accepted];
      continue;
    }
    // The bonus row samples the target; a rejected row samples its residual,
    // the target without the one-hot draft token.
    std::vector<double> weights = targets[accepted].probabilities;
    if (accepted < kPositions)
      weights[inputs[accepted + 1]] = 0.0;
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    const double threshold =
        double(uniforms[SPLASH_UNIFORM_CORRECTION]) * total;
    double cumulative = 0.0;
    uint32_t token = 0;
    while (cumulative + weights[token] <= threshold)
      cumulative += weights[token++];
    require(threshold - cumulative > 1e-5 * total &&
                cumulative + weights[token] - threshold > 1e-5 * total,
            "test data is ambiguous at the correction draw");
    expectedOutput[lane][accepted] = token;
  }

  batch.poison();
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch.buffers, kStopTokens[0],
                     kStopTokens[1], {table, kSlots});
  const std::array<uint32_t, lanes> maximumRetained{kRows, kRows};
  sampling.addAcceptance(graph, acceptance, maximumRetained, policies,
                         kStopTokens[0], kStopTokens[1]);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *retained =
      static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
  const auto *acceptedCounts =
      static_cast<const uint32_t *>(acceptance.acceptedCounts.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const std::string label = "speculative lane " + std::to_string(lane) +
                              " sampling mask " + std::to_string(samplingMask);
    const uint32_t accepted = expectedAccepted[lane];
    require(acceptedCounts[lane] == accepted && retained[lane] == accepted + 1,
            label + ": accepted " + std::to_string(acceptedCounts[lane]) +
                " draft tokens, not " + std::to_string(accepted));
    for (uint32_t position = 0; position <= accepted; ++position)
      require(batch.outputTokens()[lane * kRows + position] ==
                  expectedOutput[lane][position],
              label + ": output differs from a sequential decode");
  }
}

// Extreme repetition penalties saturate instead of overflowing: every
// candidate the penalty drives to the same float limit ties, greedy takes the
// lowest id among them, and sampling draws evenly among the lowest ids top-k
// keeps of them, as the uniform places it.
void extremes(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  constexpr uint32_t kSaturated = 50;
  constexpr uint32_t kTopK = 20;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, 1);
  const MetalBuffer table = allocate(backend, uint64_t{kLanes} * vocabulary * 4);
  const std::span<uint32_t> words = tableRow(table, vocabulary, 2);
  const std::array<uint32_t, 1> rows{2};
  // Tokens 100, 107, ... hold the prompt bit and positive logits large enough
  // to overflow, or, for the mask case, negative ones.
  std::vector<uint32_t> saturating;
  for (uint32_t index = 0; index < kSaturated; ++index)
    saturating.push_back(100 + 7 * (kSaturated - 1 - index));
  std::sort(saturating.begin(), saturating.end());
  struct Case final {
    float repetition;
    float saturatingLogit;
    bool masked;
  };
  for (const Case c : {Case{1e-37F, 45.0F, false},
                       Case{std::ldexp(1.0F, -149), 45.0F, false},
                       Case{1e38F, -6.0F, true}}) {
    float *row = batch.logits();
    for (uint32_t token = 0; token < vocabulary; ++token) {
      row[token] = -3.0F + float(token % 17) * 0.1F;
      words[token] = token % 5 == 0 ? SPLASH_PENALTY_PROMPT_BIT : 0U;
    }
    for (const uint32_t token : saturating) {
      row[token] = c.saturatingLogit + float(token % 3);
      words[token] = SPLASH_PENALTY_PROMPT_BIT;
    }
    // The mask leaves only the saturating tokens.
    std::fill(batch.masks(), batch.masks() + batch.maskWords(), 0U);
    for (const uint32_t token : saturating)
      batch.masks()[token / 32] |= 1U << (token % 32);
    const std::vector<float> original(row, row + kRows * vocabulary);
    // The saturated tokens top-k keeps weigh the same; nothing else weighs.
    std::vector<double> kept(vocabulary, 0.0);
    for (uint32_t index = 0; index < kTopK; ++index)
      kept[saturating[index]] = 1.0;
    for (const float temperature : {0.0F, 1.0F}) {
      for (const float uniform : {0.0F, 0.51F, 0.999F}) {
        std::copy(original.begin(), original.end(), row);
        batch.poison();
        batch.uniforms()[SPLASH_UNIFORM_INITIAL] = uniform;
        const SamplingPolicy policy{temperature > 0.0F ? kTopK : 1U,
                                    temperature,
                                    1.0F,
                                    c.masked,
                                    false,
                                    {c.repetition, 0.0F, 0.0F}};
        CommandGraph graph;
        sampling.addInitial(graph, {&policy, 1}, batch.buffers, 0,
                            kStopTokens[0], kStopTokens[1], {table, rows});
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        const std::string label =
            "repetition " + std::to_string(c.repetition) + " temperature " +
            std::to_string(temperature) + " uniform " + std::to_string(uniform);
        const float limit = c.saturatingLogit > 0.0F ? kFloatMax : -kFloatMax;
        for (const uint32_t token : saturating)
          require(row[token] == limit,
                  label + ": a penalized logit did not saturate");
        for (uint32_t token = 0; token < vocabulary; ++token)
          require(std::isfinite(row[token]),
                  label + ": a penalized logit is not finite");
        const uint32_t token = batch.outputTokens()[0];
        if (temperature == 0.0F) {
          require(token == saturating.front(),
                  label + ": greedy did not take the lowest saturated id");
          continue;
        }
        require(batch.record(0).maximum == limit,
                label + ": the saturated candidates' maximum differs");
        requireDraw(token, kept, uniform, 0.0, label);
      }
    }
  }
}

// Each buffer the selections, their penalties and acceptance reach, at its
// extent and one element short. A buffer holds the rows of every lane up to
// the last lane that reads it (metal/abi/Sampling.h). The first token at row
// 3 of three lanes: a sampled lane, a constrained greedy lane penalized from
// table row 2, and a greedy lane. The verify rows of three lanes: a greedy
// lane, a constrained sampled lane, and a greedy lane penalized from table row
// 1, whose penalty reads the verify input rows past the sampled lane's draft;
// and of two, a greedy and a sampled lane, whose draft reads them. The
// acceptance of the three verify lanes, the last retaining five rows or all
// eight.
void bufferExtents(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003, lanes = 3;
  const Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, lanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const uint64_t logitsRow = uint64_t{vocabulary} * 4, maskRow = uint64_t{batch.maskWords()} * 4;
  const SamplingPenalties penalized{1.0F, 1.5F, 0.0F};
  const SamplingPolicy greedy{}, sampled{32, 0.8F, 0.95F};
  SamplingPolicy constrainedSampled = sampled, constrainedGreedy = greedy, penalizedGreedy = greedy;
  constrainedSampled.constrained = constrainedGreedy.constrained = true;
  constrainedGreedy.penalties = penalizedGreedy.penalties = penalized;
  struct Extent final {
    MetalBuffer SamplingBuffers::*member;
    uint64_t bytes;
    uint64_t element;
    const char *what;
  };
  const auto requireExtents = [&](std::initializer_list<Extent> extents, const auto &encode) {
    for (const Extent &extent : extents)
      splash::test::requireExtent(backend, batch.buffers.*extent.member, extent.bytes, extent.element, extent.what,
                                  [&](CommandGraph &graph, const MetalBuffer &buffer) {
                                    SamplingBuffers changed = batch.buffers;
                                    changed.*extent.member = buffer;
                                    encode(graph, changed);
                                  });
  };
  // The workspaces of `rows` selected rows.
  const auto workspace = [](uint64_t rows) {
    return std::array<uint64_t, 6>{rows * SPLASH_TARGET_SAMPLING_SHARDS * 4,
                                   rows * SPLASH_TARGET_SAMPLING_SHARDS * 4,
                                   rows * SPLASH_TARGET_SAMPLING_SHARDS * sizeof(TargetShardMass),
                                   rows * sizeof(TargetVocabularyRow),
                                   rows * SPLASH_TARGET_VOCABULARY_RANGES * sizeof(TargetVocabularyRange),
                                   rows * 4};
  };

  {
    const std::array policies{sampled, constrainedGreedy, greedy};
    const std::array<uint32_t, lanes> tableRows{0, 2, 0};
    const auto encode = [&](CommandGraph &graph, const SamplingBuffers &buffers, const MetalBuffer &words) {
      sampling.addInitial(graph, policies, buffers, 3, kStopTokens[0], kStopTokens[1], {words, tableRows});
    };
    // One row of every lane up to the last greedy lane, three, and up to the
    // last sampled lane, one.
    const auto greedyRows = workspace(3), sampledRows = workspace(1);
    requireExtents({{&SamplingBuffers::logits, (2 * kRows + 3 + 1) * logitsRow, 4, "sampling logits"},
                    {&SamplingBuffers::constraintMasks, (kRows + 1 + 1) * maskRow, 4, "constraint mask"},
                    {&SamplingBuffers::outputTokens, lanes * 4, 4, "sampled token"},
                    {&SamplingBuffers::argmaxValues, greedyRows[0], 4, "argmax value"},
                    {&SamplingBuffers::argmaxIndices, greedyRows[1], 4, "argmax index"},
                    {&SamplingBuffers::partialMasses, sampledRows[2], sizeof(TargetShardMass), "sampling mass"},
                    {&SamplingBuffers::vocabularyRows, sampledRows[3], sizeof(TargetVocabularyRow),
                     "sampling vocabulary row"},
                    {&SamplingBuffers::vocabularyRanges, sampledRows[4], sizeof(TargetVocabularyRange),
                     "sampling vocabulary range"},
                    {&SamplingBuffers::vocabularyArrivals, sampledRows[5], 4, "sampling arrival"},
                    {&SamplingBuffers::uniforms, (SPLASH_UNIFORM_INITIAL + 1) * 4, 4, "sampling uniform"}},
                   [&](CommandGraph &graph, const SamplingBuffers &buffers) { encode(graph, buffers, table); });
    splash::test::requireExtent(backend, table, 3 * logitsRow, 4, "penalty table",
                                [&](CommandGraph &graph, const MetalBuffer &words) {
                                  encode(graph, batch.buffers, words);
                                });
  }
  {
    const std::array policies{greedy, constrainedSampled, penalizedGreedy};
    const std::array<uint32_t, lanes> tableRows{0, 0, 1};
    const auto encode = [&](CommandGraph &graph, const SamplingBuffers &buffers, const MetalBuffer &words) {
      sampling.addVerify(graph, policies, buffers, kStopTokens[0], kStopTokens[1], {words, tableRows});
    };
    // Every row of every lane up to the last greedy lane, three, and up to
    // the last sampled lane, two, whose draft reads its candidates at all
    // seven positions.
    const auto greedyRows = workspace(3 * kRows), sampledRows = workspace(2 * kRows);
    const uint64_t candidates = 2 * kPositions * kDraftCandidates * 4;
    requireExtents(
        {{&SamplingBuffers::logits, 3 * kRows * logitsRow, 4, "sampling logits"},
         {&SamplingBuffers::constraintMasks, (kRows + 1 + 1 + kRows) * maskRow, 4, "constraint mask"},
         {&SamplingBuffers::outputTokens, lanes * kRows * 4, 4, "sampled token"},
         {&SamplingBuffers::argmaxValues, greedyRows[0], 4, "argmax value"},
         {&SamplingBuffers::argmaxIndices, greedyRows[1], 4, "argmax index"},
         {&SamplingBuffers::partialMasses, sampledRows[2], sizeof(TargetShardMass), "sampling mass"},
         {&SamplingBuffers::vocabularyRows, sampledRows[3], sizeof(TargetVocabularyRow), "sampling vocabulary row"},
         {&SamplingBuffers::vocabularyRanges, sampledRows[4], sizeof(TargetVocabularyRange),
          "sampling vocabulary range"},
         {&SamplingBuffers::vocabularyArrivals, sampledRows[5], 4, "sampling arrival"},
         {&SamplingBuffers::uniforms, (kUniforms + SPLASH_UNIFORM_CORRECTION + 1) * 4, 4, "sampling uniform"},
         {&SamplingBuffers::inputTokens, lanes * kRows * 4, 4, "verify input token"},
         {&SamplingBuffers::draftCandidates, candidates, 4, "draft candidate"},
         {&SamplingBuffers::draftProbabilities, candidates, 4, "draft probability"}},
        [&](CommandGraph &graph, const SamplingBuffers &buffers) { encode(graph, buffers, table); });
    splash::test::requireExtent(backend, table, 2 * logitsRow, 4, "penalty table",
                                [&](CommandGraph &graph, const MetalBuffer &words) {
                                  encode(graph, batch.buffers, words);
                                });
  }
  {
    const std::array policies{greedy, sampled};
    requireExtents({{&SamplingBuffers::inputTokens, 2 * kRows * 4, 4, "verify input token"}},
                   [&](CommandGraph &graph, const SamplingBuffers &buffers) {
                     sampling.addVerify(graph, policies, buffers, kStopTokens[0], kStopTokens[1], {});
                   });
  }

  const std::array policies{greedy, sampled, greedy};
  const AcceptanceBuffers acceptance{allocate(backend, lanes * kPositions * 4),
                                     batch.buffers.draftCandidates,
                                     batch.buffers.draftProbabilities,
                                     batch.buffers.vocabularyRows,
                                     batch.buffers.uniforms,
                                     batch.buffers.outputTokens,
                                     allocate(backend, lanes * 4),
                                     allocate(backend, lanes * 4)};
  const uint64_t candidates = 2 * kPositions * kDraftCandidates * 4;
  for (const uint32_t lastRetained : {5U, kRows}) {
    const std::array<uint32_t, lanes> retained{kRows, kRows, lastRetained};
    for (const auto &[member, bytes, element, what] :
         std::initializer_list<std::tuple<MetalBuffer AcceptanceBuffers::*, uint64_t, uint64_t, const char *>>{
             {&AcceptanceBuffers::proposedTokens, lanes * kPositions * 4, 4, "proposed token"},
             {&AcceptanceBuffers::candidates, candidates, 4, "draft candidate"},
             {&AcceptanceBuffers::proposalProbabilities, candidates, 4, "draft probability"},
             {&AcceptanceBuffers::targetVocabularyRows, (kRows + kPositions) * sizeof(TargetVocabularyRow),
              sizeof(TargetVocabularyRow), "target vocabulary row"},
             {&AcceptanceBuffers::uniforms, (kUniforms + SPLASH_UNIFORM_ACCEPTANCE + kPositions) * 4, 4,
              "acceptance uniform"},
             {&AcceptanceBuffers::outputTokens, (2 * kRows + std::max(kPositions, lastRetained)) * 4, 4,
              "target token"},
             {&AcceptanceBuffers::retainedCounts, lanes * 4, 4, "retained count"},
             {&AcceptanceBuffers::acceptedCounts, lanes * 4, 4, "accepted count"}})
      splash::test::requireExtent(backend, acceptance.*member, bytes, element, what,
                                  [&](CommandGraph &graph, const MetalBuffer &buffer) {
                                    AcceptanceBuffers changed = acceptance;
                                    changed.*member = buffer;
                                    sampling.addAcceptance(graph, changed, retained, policies, kStopTokens[0],
                                                           kStopTokens[1]);
                                  });
  }
}

// A penalized lane needs a table row inside the table. A refused request
// encodes nothing.
void invalidPenalties(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, 1);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const SamplingPolicy penalized{1, 0.0F, 1.0F, false, false,
                                 {1.0F, 1.0F, 0.0F}};
  const std::array<uint32_t, 1> outside{kLanes};
  CommandGraph graph;
  rejects([&] {
    sampling.addVerify(graph, {&penalized, 1}, batch.buffers, 1, 2, {});
  }, "penalized lane has no penalty table row",
          "penalties without a table were accepted");
  rejects([&] {
    sampling.addVerify(graph, {&penalized, 1}, batch.buffers, 1, 2,
                       {table, outside});
  }, "penalty table buffer holds",
          "a table row outside the table was accepted");
  rejects([&] {
    sampling.addInitial(graph, {&penalized, 1}, batch.buffers, 0, 1, 2,
                        {table, outside});
  }, "penalty table buffer holds",
          "an initial table row outside the table was accepted");
  require(graph.empty(), "a refused penalty request encoded a dispatch");
  // Unpenalized lanes need no table row.
  const SamplingPolicy greedy{1, 0.0F, 1.0F, false};
  sampling.addVerify(graph, {&greedy, 1}, batch.buffers, 1, 2, {});
  require(!graph.empty(), "an unpenalized batch encoded nothing");
}

// The draft of one verify position: sixteen candidates with random
// probabilities, and the draft token among them, which verify input row
// position + 1 holds.
void setDraft(const Batch &batch, uint32_t lane, uint32_t position,
              std::vector<uint32_t> candidates, uint32_t draft,
              Random &random) {
  const uint64_t origin =
      (uint64_t{lane} * kPositions + position) * kDraftCandidates;
  auto *ids = static_cast<uint32_t *>(batch.buffers.draftCandidates.contents()) +
              origin;
  auto *proposal =
      static_cast<float *>(batch.buffers.draftProbabilities.contents()) + origin;
  while (candidates.size() < kDraftCandidates)
    candidates.push_back(3 + random.next() % (batch.vocabulary - 3));
  float total = 0.0F;
  for (uint32_t index = 0; index < kDraftCandidates; ++index) {
    ids[index] = candidates[index];
    proposal[index] = 0.05F + 0.5F * (random.unit() + 1.0F);
    total += proposal[index];
  }
  for (uint32_t index = 0; index < kDraftCandidates; ++index)
    proposal[index] /= total;
  batch.inputTokens()[lane * kRows + position + 1] = draft;
}

enum class Shape { Peaked, Flat, Graded, Ties };

// Peaked rows are fillRow's forty spikes, of which a 0.95 nucleus keeps a
// few; flat ones spread their mass over the whole vocabulary; graded ones
// descend over 300 tokens, so a nucleus holds tens to hundreds; tied ones
// take seven values, with -inf at random tokens.
void fillShaped(float *row, uint32_t vocabulary, Shape shape, Random &random) {
  if (shape == Shape::Peaked)
    return fillRow(row, vocabulary, random);
  if (shape == Shape::Ties) {
    for (uint32_t token = 0; token < vocabulary; ++token)
      row[token] = random.next() % 11 == 0
                       ? -INFINITY
                       : float(int(random.next() % 7) - 3);
    return;
  }
  for (uint32_t token = 0; token < vocabulary; ++token)
    row[token] =
        shape == Shape::Flat ? random.unit() : -6.0F + 2.0F * random.unit();
  if (shape == Shape::Graded)
    for (uint32_t rank = 0; rank < 300; ++rank)
      row[3 + (rank * 7919 + 11) % (vocabulary - 3)] =
          4.0F - 0.02F * float(rank) + 0.005F * random.unit();
}

// Sampled lanes with the default top-k of 20, larger ones (one past what a
// group ranks, whose nucleus is searched for rather than walked), a
// disabled one and one past the vocabulary, constrained or ignoring
// end-of-sequence, in batches beside greedy lanes that are constrained or
// ignore end-of-sequence too, over every row shape: every sampled row as
// requireSampledRow checks it, and every greedy row's argmax among the
// tokens its lane admits. The first token after a prompt follows the same
// rules. With minP the sampled lanes cut by min_p first: alone, before a
// top-k and a nucleus, at 1, which leaves the most likely token and its
// ties, and so low that it drops nothing; a greedy lane ignores it.
void sampledRows(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes,
                 bool minP) {
  Random random(0x77696465 + uint64_t{vocabulary} * 8 + lanes +
                (minP ? 64 : 0));
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, lanes);
  const SamplingPolicy narrow{20, 0.9F, 0.95F, false};
  const SamplingPolicy disabled{0, 1.0F, 0.95F, false};
  const SamplingPolicy fifty{50, 0.8F, 0.9F, true};
  const SamplingPolicy thousand{1000, 1.3F, 1.0F, false, true};
  const SamplingPolicy wide{2000, 1.0F, 0.95F, false};
  const SamplingPolicy beyond{vocabulary + 5, 0.6F, 0.7F, false};
  const SamplingPolicy greedy{1, 0.0F, 1.0F, false};
  const SamplingPolicy maskedGreedy{1, 0.0F, 1.0F, true};
  const SamplingPolicy stoplessGreedy{1, 0.0F, 1.0F, false, true};
  const SamplingPolicy floor{0, 1.0F, 1.0F, false, false, {}, 0.05F};
  const SamplingPolicy floorNucleus{50, 0.8F, 0.9F, true, false, {}, 0.2F};
  const SamplingPolicy heaviest{0, 0.9F, 1.0F, false, false, {}, 1.0F};
  const SamplingPolicy floorless{20, 0.9F, 0.95F, false, true, {}, 1e-30F};
  const SamplingPolicy floorGreedy{1, 0.0F, 1.0F, false, false, {}, 0.5F};
  const std::array<std::vector<SamplingPolicy>, kLanes> batches =
      minP ? std::array<std::vector<SamplingPolicy>, kLanes>{
                 std::vector{floor}, std::vector{floorNucleus, floorGreedy},
                 std::vector{heaviest, floorless, stoplessGreedy},
                 std::vector{floor, floorNucleus, greedy, heaviest}}
           : std::array<std::vector<SamplingPolicy>, kLanes>{
                 std::vector{disabled}, std::vector{fifty, maskedGreedy},
                 std::vector{beyond, narrow, stoplessGreedy},
                 std::vector{wide, thousand, greedy, fifty}};
  const std::vector<SamplingPolicy> &policies = batches[lanes - 1];
  for (uint32_t index = 0; index < lanes * (kRows + 1) * batch.maskWords();
       ++index)
    batch.masks()[index] = 0xB6DB6DB6U ^ index;
  const auto admission = [&](uint32_t lane, uint32_t row) {
    return Admission{policies[lane].constrained
                         ? batch.masks() +
                               (uint64_t{lane} * (kRows + 1) + row + 1) *
                                   batch.maskWords()
                         : nullptr,
                     policies[lane].excludesStopTokens};
  };
  std::vector<Distribution> references(batch.rows);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    for (uint32_t row = 0; row < kRows; ++row) {
      const uint32_t index = lane * kRows + row;
      fillShaped(batch.row(index), vocabulary, static_cast<Shape>(row % 3),
                 random);
      // A stop token leads every other row.
      if (row % 2)
        batch.row(index)[kStopTokens[0]] = 13.0F;
      if (policies[lane].samples())
        references[index] =
            referenceDistribution(batch.row(index), vocabulary,
                                  policies[lane], admission(lane, row));
    }
    // Draft tokens from the start and the middle of the distribution, and a
    // random one, which it may not keep.
    for (uint32_t position = 0; position < kPositions; ++position) {
      const std::vector<uint32_t> &kept = references[lane * kRows + position].order;
      std::vector<uint32_t> candidates;
      if (!kept.empty())
        candidates = {kept.front(), kept[kept.size() / 2]};
      candidates.push_back(3 + random.next() % (vocabulary - 3));
      setDraft(batch, lane, position, candidates,
               candidates[candidates.size() == 3 ? position % 3 : 0], random);
    }
    batch.inputTokens()[lane * kRows] = 7;
    for (uint32_t uniform = 0; uniform < kUniforms; ++uniform)
      batch.uniforms()[lane * kUniforms + uniform] =
          0.5F * (random.unit() + 1.0F);
  }
  const std::string label = std::string(minP ? "min_p" : "sampled") +
                            " rows B" + std::to_string(lanes) + " vocabulary " +
                            std::to_string(vocabulary);

  batch.poison();
  CommandGraph verify;
  sampling.addVerify(verify, policies, batch.buffers, kStopTokens[0],
                     kStopTokens[1], {});
  static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
  for (uint32_t index = 0; index < batch.rows; ++index) {
    const uint32_t lane = index / kRows;
    const Admission admits = admission(lane, index % kRows);
    const std::string rowLabel = label + " row " + std::to_string(index);
    if (policies[lane].samples())
      requireSampledRow(batch, index, references[index], rowLabel);
    else
      require(batch.outputTokens()[index] ==
                  referenceArgmax(batch.row(index), vocabulary, admits),
              rowLabel + ": a greedy row lost its argmax");
  }

  // The first token after a prompt, from a row at an offset; a constrained
  // one reads mask row 0, as a request's first token does. One whose
  // distribution keeps every token searches nothing.
  std::vector<SamplingPolicy> initialPolicies = policies;
  initialPolicies.push_back({0, 1.0F, 1.0F, false});
  for (const SamplingPolicy &policy : initialPolicies) {
    for (const uint32_t offset : {1U, 4U}) {
      batch.poison();
      batch.uniforms()[SPLASH_UNIFORM_INITIAL] =
          0.5F * (random.unit() + 1.0F);
      CommandGraph initial;
      sampling.addInitial(initial, {&policy, 1}, batch.buffers, offset,
                          kStopTokens[0], kStopTokens[1], {});
      static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
      const Admission admits{policy.constrained ? batch.masks() : nullptr,
                             policy.excludesStopTokens};
      const std::string rowLabel = label + " initial top_k " +
                                   std::to_string(policy.topK) + " min_p " +
                                   std::to_string(policy.minP) + " offset " +
                                   std::to_string(offset);
      if (policy.samples())
        requireInitialDraw(batch, 0,
                           referenceDistribution(batch.row(offset), vocabulary,
                                                 policy, admits),
                           rowLabel);
      else
        require(batch.outputTokens()[0] ==
                    referenceArgmax(batch.row(offset), vocabulary, admits),
                rowLabel + ": a greedy first token lost its argmax");
    }
  }
}

// Every mixed policy mask against isolated lane execution: a lane's rows
// select the same, bit for bit, whatever lanes share its batch. The outputs
// are poisoned, so a missing argmax or draw cannot pass by reading old data.
void mixedVerify(MetalBackend &backend, uint32_t lanes, uint32_t samplingMask) {
  constexpr uint32_t vocabulary = 1003;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, lanes);
  batch.poison();
  Random random(9831 + lanes);
  std::vector<SamplingPolicy> policies;
  auto *candidates =
      static_cast<uint32_t *>(batch.buffers.draftCandidates.contents());
  auto *proposal =
      static_cast<float *>(batch.buffers.draftProbabilities.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    policies.push_back(
        {8 + lane, (samplingMask & (1U << lane)) ? 0.7F : 0.0F, 0.9F, false});
    for (uint32_t row = 0; row < kRows; ++row)
      fillShaped(batch.row(lane * kRows + row), vocabulary,
                 row % 2 ? Shape::Ties : Shape::Peaked, random);
  }
  for (uint32_t row = 0; row < batch.rows; ++row)
    batch.inputTokens()[row] = random.next() % vocabulary;
  for (uint32_t entry = 0; entry < lanes * kPositions * kDraftCandidates;
       ++entry) {
    candidates[entry] = random.next() % vocabulary;
    proposal[entry] = 0.5F * (random.unit() + 1.0F) / kDraftCandidates;
  }
  for (uint32_t uniform = 0; uniform < lanes * kUniforms; ++uniform)
    batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
  const auto stops = shardEdgeStopTokens(vocabulary);
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch.buffers, stops[0], stops[1], {});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const Batch single = makeBatch(backend, vocabulary, 1);
    single.poison();
    const auto copy = [&](const MetalBuffer &from, const MetalBuffer &to) {
      std::memcpy(to.contents(),
                  static_cast<const uint8_t *>(from.contents()) +
                      lane * to.sizeBytes(),
                  to.sizeBytes());
    };
    copy(batch.buffers.logits, single.buffers.logits);
    copy(batch.buffers.inputTokens, single.buffers.inputTokens);
    copy(batch.buffers.draftCandidates, single.buffers.draftCandidates);
    copy(batch.buffers.draftProbabilities, single.buffers.draftProbabilities);
    copy(batch.buffers.uniforms, single.buffers.uniforms);
    CommandGraph reference;
    sampling.addVerify(reference, std::span(policies).subspan(lane, 1),
                       single.buffers, stops[0], stops[1], {});
    static_cast<void>(backend.submitCommandAsync(reference.dispatches()).wait());
    if (policies[lane].samples())
      require(std::memcmp(&batch.record(lane * kRows), &single.record(0),
                          single.buffers.vocabularyRows.sizeBytes()) == 0,
              "mixed verification changed sampled rows");
    require(std::memcmp(batch.outputTokens() + lane * kRows,
                        single.outputTokens(),
                        single.buffers.outputTokens.sizeBytes()) == 0,
            "mixed verification changed a lane's tokens");
  }
}

// The runtime uploads masks for constrained lanes alone, so an unconstrained
// lane must select the same first token and verify rows over a mask buffer
// of all zeros as over one of all ones.
void unconstrainedRowsIgnoreMasks(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  constexpr uint32_t lanes = 2;
  Sampling sampling(vocabulary);
  const std::array<SamplingPolicy, lanes> policies{
      SamplingPolicy{},
      SamplingPolicy{
          .topK = 0, .temperature = 0.8F, .topP = 0.9F, .minP = 0.05F}};
  const auto stops = shardEdgeStopTokens(vocabulary);
  const auto select = [&](uint32_t maskWord) {
    const Batch batch = makeBatch(backend, vocabulary, lanes);
    batch.poison();
    Random random(4127);
    for (uint32_t row = 0; row < batch.rows; ++row) {
      fillRow(batch.row(row), vocabulary, random);
      batch.inputTokens()[row] = random.next() % vocabulary;
    }
    auto *candidates =
        static_cast<uint32_t *>(batch.buffers.draftCandidates.contents());
    auto *proposal =
        static_cast<float *>(batch.buffers.draftProbabilities.contents());
    for (uint32_t entry = 0; entry < lanes * kPositions * kDraftCandidates;
         ++entry) {
      candidates[entry] = random.next() % vocabulary;
      proposal[entry] = 0.5F * (random.unit() + 1.0F) / kDraftCandidates;
    }
    for (uint32_t uniform = 0; uniform < lanes * kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    std::fill_n(batch.masks(), lanes * (kRows + 1) * batch.maskWords(),
                maskWord);
    CommandGraph verify;
    sampling.addVerify(verify, policies, batch.buffers, stops[0], stops[1], {});
    static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
    std::vector<uint32_t> tokens(batch.outputTokens(),
                                 batch.outputTokens() + batch.rows);
    CommandGraph initial;
    sampling.addInitial(initial, policies, batch.buffers, 3, stops[0],
                        stops[1], {});
    static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
    tokens.insert(tokens.end(), batch.outputTokens(),
                  batch.outputTokens() + lanes);
    return tokens;
  };
  require(select(0) == select(std::numeric_limits<uint32_t>::max()),
          "an unconstrained lane read its constraint mask");
}

// Constrained greedy lanes (the argmax kernels) and constrained sampled
// lanes with top-k 1 (a distribution of one token) must agree with a
// full-vocabulary CPU argmax, including ties, row offsets and masks: the
// first token, every greedy verify row's token, and every sampled verify
// row's draw and its draft token's probability, 1 for the argmax and 0
// otherwise. Every scratch and output buffer is poisoned, so nothing passes
// on stale data. The fp32 logits carry offsets below the bf16 spacing of
// their values, which decide the argmax among equal integer parts: reading
// them rounded would pick the lowest id instead.
void targetTop1(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes) {
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, lanes);
  const uint32_t words = batch.maskWords();
  for (uint32_t row = 0; row < batch.rows; ++row)
    for (uint32_t token = 0; token < vocabulary; ++token)
      batch.row(row)[token] = float(int((token * 7 + row * 13) % 23) - 11) +
                              float(token % 3) * 0x1p-12F;
  for (uint32_t row = 0; row < lanes * (kRows + 1); ++row)
    for (uint32_t token = 0; token < vocabulary; ++token)
      if ((token + row) % 17 == 0)
        batch.masks()[uint64_t{row} * words + token / 32] |= 1U << (token % 32);
  constexpr uint32_t kUnmasked = ~0U;
  const auto expected = [&](uint32_t row, uint32_t maskRow) {
    float best = -INFINITY;
    uint32_t id = ~0U;
    for (uint32_t token = 0; token < vocabulary; ++token) {
      if (maskRow != kUnmasked &&
          !(batch.masks()[uint64_t{maskRow} * words + token / 32] &
            (1U << (token % 32))))
        continue;
      const float value = batch.row(row)[token];
      if (value > best) {
        best = value;
        id = token;
      }
    }
    return id;
  };
  const uint32_t *tokens = batch.outputTokens();
  const auto stops = shardEdgeStopTokens(vocabulary);
  const SamplingPolicy greedy{1, 0.0F, 1.0F, false};
  for (const uint32_t offset : {0U, 3U, kRows - 1}) {
    for (const float temperature : {0.0F, 0.8F}) {
      batch.poison();
      CommandGraph initial;
      const SamplingPolicy single{1, temperature, 0.5F, true};
      sampling.addInitial(initial, {&single, 1}, batch.buffers, offset,
                          stops[0], stops[1], {});
      static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
      require(tokens[0] == expected(offset, 0),
              "initial target differs from masked CPU argmax");
    }
    batch.poison();
    CommandGraph initialArgmax;
    sampling.addInitial(initialArgmax, {&greedy, 1}, batch.buffers, offset,
                        stops[0], stops[1], {});
    static_cast<void>(backend.submitCommandAsync(initialArgmax.dispatches()).wait());
    require(tokens[0] == expected(offset, kUnmasked),
            "initial argmax differs from CPU argmax");
  }
  // Odd verify rows draft their argmax, even ones another token.
  for (uint32_t row = 0; row < batch.rows; ++row) {
    if (row % kRows == kRows - 1)
      continue;
    const uint32_t maskRow = row / kRows * (kRows + 1) + row % kRows + 1;
    const uint32_t id = expected(row, maskRow);
    batch.inputTokens()[row + 1] = row % 2 ? id : (id + 1) % vocabulary;
  }
  batch.poison();
  std::vector<SamplingPolicy> policies(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    policies[lane] = {1, lane % 2 ? 0.8F : 0.0F, 0.5F, true};
  CommandGraph verify;
  sampling.addVerify(verify, policies, batch.buffers, stops[0], stops[1], {});
  static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
  for (uint32_t row = 0; row < batch.rows; ++row) {
    const uint32_t maskRow = row / kRows * (kRows + 1) + row % kRows + 1;
    const uint32_t id = expected(row, maskRow);
    if (!policies[row / kRows].samples()) {
      require(tokens[row] == id,
              "batched target differs from masked CPU argmax");
      continue;
    }
    require(tokens[row] == id,
            "a sampled top-1 row drew other than its masked CPU argmax");
    if (row % kRows != kRows - 1)
      require(batch.record(row).draft_probability ==
                  (batch.inputTokens()[row + 1] == id ? 1.0F : 0.0F),
              "a sampled top-1 row's draft probability is not one-hot");
  }
  batch.poison();
  CommandGraph verifyArgmax;
  sampling.addVerify(verifyArgmax, std::vector<SamplingPolicy>(lanes, greedy),
                     batch.buffers, stops[0], stops[1], {});
  static_cast<void>(backend.submitCommandAsync(verifyArgmax.dispatches()).wait());
  for (uint32_t row = 0; row < batch.rows; ++row)
    require(tokens[row] == expected(row, kUnmasked),
            "batched argmax differs from CPU argmax");
}

// A lane that ignores end-of-sequence never selects a stop token: not in the
// sample after a prefill chunk, not in any verify row, greedy or sampled,
// alone or batched beside lanes that keep them, and a stop token its draft
// proposes is rejected. The stop tokens lead every row, so a lane that keeps
// them selects one; the other lanes take the row's best remaining token.
void excludedStopTokens(MetalBackend &backend, uint32_t vocabulary) {
  const auto stops = shardEdgeStopTokens(vocabulary);
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, kLanes);
  const auto best = [&](uint32_t row) {
    return 5 + row * 7919 % (vocabulary - 8);
  };
  for (uint32_t row = 0; row < batch.rows; ++row) {
    float *values = batch.row(row);
    for (uint32_t token = 0; token < vocabulary; ++token)
      values[token] = -4.0F + float(token % 7) * 0.01F;
    values[best(row)] = 8.0F;
    values[stops[0]] = 12.0F;
    values[stops[1]] = 11.0F;
  }
  float *uniforms = batch.uniforms();
  const uint32_t *tokens = batch.outputTokens();
  const auto stop = [&](uint32_t token) {
    return token == stops[0] || token == stops[1];
  };

  for (const bool excludes : {false, true}) {
    for (const float temperature : {0.0F, 0.8F}) {
      for (const float uniform : {0.0F, 0.5F, 0.999F}) {
        batch.poison();
        uniforms[SPLASH_UNIFORM_INITIAL] = uniform;
        CommandGraph initial;
        const SamplingPolicy policy{32, temperature, 1.0F, false, excludes};
        sampling.addInitial(initial, {&policy, 1}, batch.buffers, kRows - 1,
                            stops[0], stops[1], {});
        static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
        if (excludes)
          require(tokens[0] == best(kRows - 1) ||
                      (temperature > 0.0F && tokens[0] < vocabulary &&
                       !stop(tokens[0])),
                  "an excluding lane selected a stop token after prefill");
        else
          require(stop(tokens[0]), "the initial sample skipped the stop tokens");
      }
    }
  }

  // Drafts propose the first stop token, with all of their mass on it; it
  // is verify input row 1 of each lane.
  AcceptanceBuffers acceptance{
      allocate(backend, uint64_t{kLanes} * kPositions * sizeof(uint32_t)),
      batch.buffers.draftCandidates,
      batch.buffers.draftProbabilities,
      batch.buffers.vocabularyRows,
      batch.buffers.uniforms,
      batch.buffers.outputTokens,
      allocate(backend, kLanes * sizeof(uint32_t)),
      allocate(backend, kLanes * sizeof(uint32_t))};
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    static_cast<uint32_t *>(
        acceptance.proposedTokens.contents())[lane * kPositions] = stops[0];
    batch.inputTokens()[lane * kRows + 1] = stops[0];
    static_cast<uint32_t *>(acceptance.candidates.contents())
        [uint64_t{lane} * kPositions * kDraftCandidates] = stops[0];
    static_cast<float *>(acceptance.proposalProbabilities.contents())
        [uint64_t{lane} * kPositions * kDraftCandidates] = 1.0F;
  }
  std::fill(uniforms, uniforms + kLanes * kUniforms, 0.5F);
  const std::array<uint32_t, kLanes> maximumRetained{kRows, kRows, kRows, kRows};
  for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
    for (uint32_t excludeMask = 0; excludeMask < (1U << lanes); ++excludeMask) {
      for (const uint32_t samplingMask : {0U, 0b0101U, 0b1111U}) {
        std::vector<SamplingPolicy> policies;
        for (uint32_t lane = 0; lane < lanes; ++lane)
          policies.push_back({32, (samplingMask >> lane & 1U) ? 0.8F : 0.0F,
                              1.0F, false, (excludeMask >> lane & 1U) != 0});
        batch.poison();
        CommandGraph verify;
        sampling.addVerify(verify, policies, batch.buffers, stops[0],
                           stops[1], {});
        static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
        // Only an excluding lane's distribution leaves out the stop token
        // drafted at row 0, and none of its rows draws a stop token.
        for (uint32_t row = 0; row < lanes * kRows; ++row) {
          const SamplingPolicy &policy = policies[row / kRows];
          if (!policy.samples()) {
            require(tokens[row] ==
                        (policy.excludesStopTokens ? best(row) : stops[0]),
                    "a greedy verify row mishandled the stop tokens");
            continue;
          }
          require(row % kRows != 0 ||
                      (batch.record(row).draft_probability > 0.0F) !=
                          policy.excludesStopTokens,
                  "a verify distribution mishandled the stop tokens");
          require(!policy.excludesStopTokens ||
                      (tokens[row] < vocabulary && !stop(tokens[row])),
                  "an excluding verify row drew a stop token");
        }
        CommandGraph accept;
        sampling.addAcceptance(accept, acceptance,
                               std::span(maximumRetained).first(lanes),
                               policies, stops[0], stops[1]);
        static_cast<void>(backend.submitCommandAsync(accept.dispatches()).wait());
        const auto *retained =
            static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          const uint32_t *output = tokens + lane * kRows;
          if (!policies[lane].excludesStopTokens) {
            require(retained[lane] == 1 && output[0] == stops[0],
                    "the target did not accept the drafted stop token");
            continue;
          }
          require(retained[lane] >= 1 && retained[lane] <= kRows,
                  "an excluding lane retained no token");
          for (uint32_t index = 0; index < retained[lane]; ++index)
            require(output[index] < vocabulary && !stop(output[index]),
                    "an excluding lane accepted a stop token");
        }
      }
    }
  }
}

// Ties where a distribution ends: more tokens tie at the logit where top_k
// or top_p cuts than a group orders in threadgroup memory, and fewer; the
// distribution keeps the lowest ids among them, and saturated penalized
// logits tie the same way. A top_p cut within the ties a top_k cut kept
// measures against the mass above the ties as well. Draft tokens at both
// sides of the cut test it exactly: the last tie kept has its probability,
// the first one cut has none.
void ties(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 248320;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, 1);
  const MetalBuffer table = allocate(backend, uint64_t{kLanes} * vocabulary * 4);
  const std::array<uint32_t, 1> tableRows{1};
  struct Case final {
    const char *name;
    uint32_t tied;
    uint32_t above;
    SamplingPolicy policy;
  };
  // Every tie holds a positive logit whose repetition 1e-37 saturates.
  for (const Case c :
       {Case{"top_k within 3000 ties", 3000, 10, {1510, 1.0F, 1.0F, false}},
        Case{"top_k within 600 ties", 600, 10, {310, 0.7F, 1.0F, false}},
        Case{"top_k then top_p within 3000 ties",
             3000,
             10,
             {1510, 2.0F, 0.9F, false}},
        Case{"top_p within 3000 ties", 3000, 0, {0, 1.0F, 0.4999F, false}},
        Case{"top_p within 3000 saturated ties",
             3000,
             0,
             {0, 1.0F, 0.9001F, false, false, {1e-37F, 0.0F, 0.0F}}}}) {
    Random random(0x74696573 + c.tied + c.policy.topK);
    const bool saturated = c.policy.penalties.active();
    std::vector<uint32_t> tied;
    for (uint32_t index = 0; index < c.tied; ++index)
      tied.push_back(40 + index * 79);
    std::span<uint32_t> words = tableRow(table, vocabulary, tableRows[0]);
    std::fill(words.begin(), words.end(), 0U);
    for (uint32_t row = 0; row < kRows; ++row) {
      float *logits = batch.row(row);
      for (uint32_t token = 0; token < vocabulary; ++token)
        logits[token] = -12.0F + random.unit();
      for (const uint32_t token : tied) {
        logits[token] = saturated ? 45.0F + float(token % 5) : 2.0F;
        words[token] = saturated ? SPLASH_PENALTY_PROMPT_BIT : 0U;
      }
      for (uint32_t index = 0; index < c.above; ++index)
        logits[5 + index * 13] = 5.0F + float(index);
    }
    // The distribution as the kernels see it: the saturated logits.
    std::vector<float> expected(batch.row(0), batch.row(0) + vocabulary);
    if (saturated)
      for (const uint32_t token : tied)
        expected[token] = kFloatMax;
    const Distribution target =
        referenceDistribution(expected.data(), vocabulary, c.policy);
    require(target.margin > kTopPMargin, std::string(c.name) +
                                             ": test data is ambiguous");
    const uint32_t kept = static_cast<uint32_t>(target.order.size()) - c.above;
    require(kept > 0 && kept < c.tied,
            std::string(c.name) + ": the cut is not within the ties");
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t draft = position % 2 ? tied[kept] : tied[kept - 1];
      setDraft(batch, 0, position, {draft}, draft, random);
    }
    for (uint32_t uniform = 0; uniform < kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    batch.poison();
    CommandGraph verify;
    sampling.addVerify(verify, {&c.policy, 1}, batch.buffers, kStopTokens[0],
                       kStopTokens[1], {table, tableRows});
    static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
    for (uint32_t row = 0; row < kRows; ++row)
      requireSampledRow(batch, row, target,
                        std::string(c.name) + " row " + std::to_string(row));
  }
}

// Where min_p cuts, on rows whose weights are known: the heaviest token
// weighs 1, ten weigh 0.3, twenty 0.1 and a hundred 0.01 at temperature 1.
// min_p keeps the classes that weigh at least it, each tie whole; top_k then
// counts and top_p measures within what it kept, not within the row; and the
// cut follows the temperature, which squares the weights at 0.5. Draft
// tokens of the last class kept and of the first one dropped test the cut
// from both sides: one has its probability, the other none.
void minPCuts(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 4096;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, 1);
  struct Class final {
    uint32_t first;
    uint32_t count;
    float weight;
  };
  constexpr std::array<Class, 4> classes{
      Class{5, 1, 1.0F}, Class{100, 10, 0.3F}, Class{200, 20, 0.1F},
      Class{300, 100, 0.01F}};
  Random random(0x6d696e70);
  for (uint32_t row = 0; row < kRows; ++row) {
    float *logits = batch.row(row);
    std::fill(logits, logits + vocabulary, -1000.0F);
    for (const Class &c : classes)
      for (uint32_t index = 0; index < c.count; ++index)
        logits[c.first + index] = std::log(c.weight);
  }
  batch.inputTokens()[0] = 7;
  struct Case final {
    const char *name;
    SamplingPolicy policy;
    // The tokens the distribution keeps, a token of the last class it keeps
    // and one of the first class it drops.
    uint32_t kept;
    uint32_t last;
    uint32_t dropped;
  };
  for (const Case &c :
       {Case{"min_p 0.2", {0, 1.0F, 1.0F, false, false, {}, 0.2F}, 11, 109,
             200},
        Case{"min_p 0.05", {0, 1.0F, 1.0F, false, false, {}, 0.05F}, 31, 219,
             300},
        Case{"min_p 0.31 drops a whole tie",
             {0, 1.0F, 1.0F, false, false, {}, 0.31F}, 1, 5, 109},
        Case{"min_p 0.05 at temperature 0.5",
             {0, 0.5F, 1.0F, false, false, {}, 0.05F}, 11, 109, 200},
        Case{"min_p 0.2 then top_k 5",
             {5, 1.0F, 1.0F, false, false, {}, 0.2F}, 5, 103, 104},
        Case{"min_p 0.05 then top_p 0.5 of what it kept",
             {0, 1.0F, 0.5F, false, false, {}, 0.05F}, 8, 106, 107},
        Case{"min_p 1", {0, 1.0F, 1.0F, false, false, {}, 1.0F}, 1, 5, 100}}) {
    const Distribution target =
        referenceDistribution(batch.row(0), vocabulary, c.policy);
    require(target.order.size() == c.kept && target.ambiguousMass == 0.0 &&
                target.probability(c.last) > 0.0 &&
                target.probability(c.dropped) == 0.0,
            std::string(c.name) + ": the reference keeps " +
                std::to_string(target.order.size()) + " tokens");
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t draft = position % 2 ? c.dropped : c.last;
      setDraft(batch, 0, position, {draft}, draft, random);
    }
    for (uint32_t uniform = 0; uniform < kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    batch.poison();
    CommandGraph verify;
    sampling.addVerify(verify, {&c.policy, 1}, batch.buffers, kStopTokens[0],
                       kStopTokens[1], {});
    static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
    for (uint32_t row = 0; row < kRows; ++row)
      requireSampledRow(batch, row, target,
                        std::string(c.name) + " row " + std::to_string(row));
  }
}

// DFlash acceptance over broad distributions against a sequential decode,
// beside a lane with the default top-k of 20 and one that min_p cuts. The
// draft proposes each position's most likely token with all of its mass,
// until a lane's wrong position, where it proposes a token the target keeps
// with little probability. Acceptance must keep the right tokens, reject the
// wrong one and correct it from the target less the draft token, or take the
// bonus token from the last row.
void speculativeWholeVocabulary(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 5003;
  constexpr uint32_t lanes = 4;
  constexpr std::array<uint32_t, lanes> kWrongAt{2, kPositions, 4, 6};
  Random random(0x77686f6c);
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, lanes);
  const std::vector<SamplingPolicy> policies{
      {0, 1.0F, 1.0F, false},
      {2000, 0.8F, 0.97F, false},
      {20, 0.9F, 0.9F, false},
      {0, 1.0F, 1.0F, false, false, {}, 0.02F}};
  AcceptanceBuffers acceptance{
      allocate(backend, uint64_t{lanes} * kPositions * sizeof(uint32_t)),
      batch.buffers.draftCandidates,
      batch.buffers.draftProbabilities,
      batch.buffers.vocabularyRows,
      batch.buffers.uniforms,
      batch.buffers.outputTokens,
      allocate(backend, lanes * sizeof(uint32_t)),
      allocate(backend, lanes * sizeof(uint32_t))};
  auto *proposed = static_cast<uint32_t *>(acceptance.proposedTokens.contents());
  auto *candidates =
      static_cast<uint32_t *>(batch.buffers.draftCandidates.contents());
  auto *proposal =
      static_cast<float *>(batch.buffers.draftProbabilities.contents());
  std::fill(candidates, candidates + lanes * kPositions * kDraftCandidates,
            0xFFFFFFFFU);
  std::fill(proposal, proposal + lanes * kPositions * kDraftCandidates, 0.0F);
  std::array<std::array<uint32_t, kRows>, lanes> expectedOutput{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    float *uniforms = batch.uniforms() + lane * kUniforms;
    uniforms[SPLASH_UNIFORM_CORRECTION] = 0.5F * (random.unit() + 1.0F);
    uint32_t *inputs = batch.inputTokens() + lane * kRows;
    inputs[0] = 9;
    for (uint32_t row = 0; row < kRows; ++row) {
      fillShaped(batch.row(lane * kRows + row), vocabulary,
                 row % 2 ? Shape::Flat : Shape::Graded, random);
      const Distribution target = referenceDistribution(
          batch.row(lane * kRows + row), vocabulary, policies[lane]);
      if (row < kPositions) {
        // The best token, or from the wrong position on the sixth best,
        // which the uniform rejects.
        const bool right = row < kWrongAt[lane];
        const uint32_t draft =
            right ? target.order.front()
                  : target.order[std::min<size_t>(5, target.order.size() - 1)];
        const double p = target.probability(draft);
        require(!right || p > 1e-6, "a draft token its target cannot accept");
        inputs[row + 1] = draft;
        proposed[lane * kPositions + row] = draft;
        candidates[(lane * kPositions + row) * kDraftCandidates] = draft;
        proposal[(lane * kPositions + row) * kDraftCandidates] = 1.0F;
        uniforms[SPLASH_UNIFORM_ACCEPTANCE + row] =
            right ? float(p * 0.5) : float(std::min(1.0, p * 2.0 + 0.01));
        if (right)
          expectedOutput[lane][row] = draft;
      }
      if (row != kWrongAt[lane])
        continue;
      // The correction, from the target less the one-hot draft token, or
      // the bonus token. The tokens a top_p cut may keep or cut shift the
      // cumulative sum by at most their mass.
      std::vector<double> weights = target.probabilities;
      if (row < kPositions)
        weights[inputs[row + 1]] = 0.0;
      const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
      const double threshold = uniforms[SPLASH_UNIFORM_CORRECTION] * total;
      const double slack = (1e-5 + target.ambiguousMass) * total;
      double cumulative = 0.0;
      uint32_t token = 0;
      while (cumulative + weights[token] <= threshold)
        cumulative += weights[token++];
      require(threshold - cumulative > slack &&
                  cumulative + weights[token] - threshold > slack &&
                  !target.ambiguous[token],
              "test data is ambiguous at the correction draw");
      expectedOutput[lane][row] = token;
    }
  }
  batch.poison();
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch.buffers, kStopTokens[0],
                     kStopTokens[1], {});
  const std::array<uint32_t, lanes> maximumRetained{kRows, kRows, kRows, kRows};
  sampling.addAcceptance(graph, acceptance, maximumRetained, policies,
                         kStopTokens[0], kStopTokens[1]);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *acceptedCounts =
      static_cast<const uint32_t *>(acceptance.acceptedCounts.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const std::string label = "acceptance lane " + std::to_string(lane);
    require(acceptedCounts[lane] == kWrongAt[lane],
            label + ": accepted " + std::to_string(acceptedCounts[lane]) +
                " draft tokens, not " + std::to_string(kWrongAt[lane]));
    for (uint32_t position = 0; position <= kWrongAt[lane]; ++position)
      require(batch.outputTokens()[lane * kRows + position] ==
                  expectedOutput[lane][position],
              label + ": output " + std::to_string(position) +
                  " differs from a sequential decode");
  }
}

// A rejected draft token's correction when the only weighted tokens of one
// simdgroup's share of the vocabulary are draft candidates the draft proposes
// at least as likely as the target keeps them: nothing of that share remains
// in the residual, and the correction comes from the others, at every
// correction uniform down to 0. (Float sums once left a residue there that a
// uniform of 0 drew, with no token to take.) Acceptance takes the correction
// as the next anchor.
void overProposedResidual(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 4096;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, 1);
  const SamplingPolicy policy{0, 1.0F, 1.0F, false};
  AcceptanceBuffers acceptance{
      allocate(backend, kPositions * sizeof(uint32_t)),
      batch.buffers.draftCandidates,
      batch.buffers.draftProbabilities,
      batch.buffers.vocabularyRows,
      batch.buffers.uniforms,
      batch.buffers.outputTokens,
      allocate(backend, sizeof(uint32_t)),
      allocate(backend, sizeof(uint32_t))};
  // Token 5 leads with weight 1 and token 6 weighs less than an ulp of it;
  // token 1000 weighs 0.5 and forty tokens from 1200 a hundredth each. The
  // others weigh nothing.
  for (uint32_t row = 0; row < kRows; ++row) {
    float *logits = batch.row(row);
    std::fill(logits, logits + vocabulary, -1000.0F);
    logits[5] = 0.0F;
    logits[6] = std::log(0.75F * 0x1p-23F);
    logits[1000] = std::log(0.5F);
    for (uint32_t index = 0; index < 40; ++index)
      logits[1200 + 50 * index] = std::log(0.01F) - 1e-4F * float(index);
  }
  const Distribution target =
      referenceDistribution(batch.row(0), vocabulary, policy);
  // Every position drafts token 5 with 0.9 and token 6 with 0.1, beside
  // fourteen weightless candidates.
  auto *ids =
      static_cast<uint32_t *>(batch.buffers.draftCandidates.contents());
  auto *proposal =
      static_cast<float *>(batch.buffers.draftProbabilities.contents());
  auto *proposed =
      static_cast<uint32_t *>(acceptance.proposedTokens.contents());
  batch.inputTokens()[0] = 5;
  for (uint32_t position = 0; position < kPositions; ++position) {
    for (uint32_t index = 0; index < kDraftCandidates; ++index) {
      const uint64_t entry = position * kDraftCandidates + index;
      ids[entry] = index < 2 ? 5 + index : 6 + index;
      proposal[entry] = index == 0 ? 0.9F : index == 1 ? 0.1F : 0.0F;
    }
    proposed[position] = 5;
    batch.inputTokens()[position + 1] = 5;
  }
  for (const float correction : {0.0F, 0x1p-24F, 0.5F, 1.0F - 0x1p-24F}) {
    const std::string label = "over-proposed residual, correction uniform " +
                              std::to_string(correction);
    // The first draft token is rejected: 0.9 * 0.9 exceeds its probability.
    for (uint32_t uniform = 0; uniform < kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.9F;
    batch.uniforms()[SPLASH_UNIFORM_CORRECTION] = correction;
    batch.poison();
    CommandGraph graph;
    sampling.addVerify(graph, {&policy, 1}, batch.buffers, kStopTokens[0],
                       kStopTokens[1], {});
    const std::array<uint32_t, 1> maximumRetained{kRows};
    sampling.addAcceptance(graph, acceptance, maximumRetained, {&policy, 1},
                           kStopTokens[0], kStopTokens[1]);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    for (uint32_t row = 0; row < kRows; ++row)
      requireSampledRow(batch, row, target,
                        label + " row " + std::to_string(row));
    const auto value = [](const MetalBuffer &buffer) {
      return *static_cast<const uint32_t *>(buffer.contents());
    };
    require(value(acceptance.acceptedCounts) == 0 &&
                value(acceptance.retainedCounts) == 1,
            label + ": acceptance did not take the correction as its anchor");
  }
}

// Bracket searches whose distribution ends far below the row's maximum in
// units of the temperature, or inside a tail of tokens that repetition
// saturates at the lowest float or spreads over millions, select as the
// reference does and take about as long as an ordinary row's search: even
// key splits bound every search to a few passes, where pivots at distances
// of the temperature alone once took thousands and stalled the batch.
void extremeSearches(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 248320;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, 1);
  const MetalBuffer table = allocate(backend, uint64_t{kLanes} * vocabulary * 4);
  const std::array<uint32_t, 1> tableRows{2};
  // A quarter of the vocabulary is in the prompt: fillRow's negative bulk,
  // which repetition scales into the tail, and some of its spikes.
  std::span<uint32_t> words = tableRow(table, vocabulary, tableRows[0]);
  for (uint32_t token = 0; token < vocabulary; token += 4)
    words[token] = SPLASH_PENALTY_PROMPT_BIT;
  Random random(0x65787472);
  for (uint32_t row = 0; row < kRows; ++row)
    fillRow(batch.row(row), vocabulary, random);
  for (uint32_t position = 0; position < kPositions; ++position) {
    const uint32_t draft = 3 + random.next() % (vocabulary - 3);
    setDraft(batch, 0, position, {draft}, draft, random);
  }
  batch.inputTokens()[0] = 7;
  for (uint32_t uniform = 0; uniform < kUniforms; ++uniform)
    batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
  const std::vector<float> original(batch.logits(),
                                    batch.logits() + batch.rows * vocabulary);
  // The fastest of a few runs, with the logits the penalties rewrite restored.
  const auto run = [&](const SamplingPolicy &policy) {
    double fastest = std::numeric_limits<double>::infinity();
    for (uint32_t attempt = 0; attempt < 3; ++attempt) {
      std::copy(original.begin(), original.end(), batch.logits());
      batch.poison();
      CommandGraph verify;
      sampling.addVerify(verify, {&policy, 1}, batch.buffers, kStopTokens[0],
                         kStopTokens[1], {table, tableRows});
      fastest = std::min(
          fastest, backend.submitCommandAsync(verify.dispatches()).wait().gpuSeconds);
    }
    return fastest;
  };
  const double ordinary = run({1000, 1.0F, 1.0F, false});
  // The default request and Qwen's recommended non-thinking sampling, timed
  // beside it.
  for (const SamplingPolicy &policy : {SamplingPolicy{20, 1.0F, 0.95F, false},
                                       SamplingPolicy{20, 0.7F, 0.8F, false}})
    std::cout << "sampling search top_k 20 temperature " << policy.temperature
              << " top_p " << policy.topP << ": " << run(policy) * 1e3
              << " ms, ordinary " << ordinary * 1e3 << " ms\n";
  constexpr uint32_t kTail = vocabulary - 2000;
  struct Case final {
    const char *name;
    SamplingPolicy policy;
  };
  for (const Case c :
       {Case{"top_k 1000 at temperature 1e-6", {1000, 1e-6F, 1.0F, false}},
        Case{"top_k in a tail saturated by repetition 1e38",
             {kTail, 1.0F, 1.0F, false, false, {1e38F, 0.0F, 0.0F}}},
        Case{"top_k in a tail spread by repetition 1e6",
             {kTail, 1.0F, 1.0F, false, false, {1e6F, 0.0F, 0.0F}}},
        Case{"top_k in a saturated tail at temperature 1e-5",
             {kTail, 1e-5F, 1.0F, false, false, {1e38F, 0.0F, 0.0F}}}}) {
    const double seconds = run(c.policy);
    std::cout << "extreme search " << c.name << ": " << seconds * 1e3
              << " ms, ordinary " << ordinary * 1e3 << " ms\n";
    for (uint32_t row = 0; row < kRows; ++row)
      requireSampledRow(batch, row,
                        referenceDistribution(batch.row(row), vocabulary,
                                              c.policy),
                        std::string(c.name) + " row " + std::to_string(row));
    require(seconds <= 8.0 * ordinary,
            std::string(c.name) + ": the search took " +
                std::to_string(seconds / ordinary) + " times an ordinary one");
  }
}

// A row of non-finite logits, which a non-finite hidden row leaves, has no
// token to select: greedy and sampled policies select 0xFFFFFFFF for a first
// token and at every verify row, and acceptance retains it as the next
// anchor. The model runtime reports that sentinel as the request's own
// failure, which keeps the rest of the batch serving.
void nonFiniteRowsSelectTheSentinel(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  constexpr uint32_t kSentinel = 0xFFFFFFFFU;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, 1);
  const AcceptanceBuffers acceptance{
      allocate(backend, kPositions * sizeof(uint32_t)),
      batch.buffers.draftCandidates,
      batch.buffers.draftProbabilities,
      batch.buffers.vocabularyRows,
      batch.buffers.uniforms,
      batch.buffers.outputTokens,
      allocate(backend, sizeof(uint32_t)),
      allocate(backend, sizeof(uint32_t))};
  Random random(0x6e616e);
  for (uint32_t position = 0; position < kPositions; ++position) {
    const uint32_t draft = 3 + random.next() % (vocabulary - 3);
    setDraft(batch, 0, position, {draft}, draft, random);
    static_cast<uint32_t *>(acceptance.proposedTokens.contents())[position] = draft;
  }
  batch.inputTokens()[0] = 7;
  for (uint32_t uniform = 0; uniform < 2 * kRows; ++uniform)
    batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
  for (const SamplingPolicy &policy : {SamplingPolicy{1, 0.0F, 1.0F, false},
                                       SamplingPolicy{20, 1.0F, 0.9F, false}}) {
    const std::string label = policy.samples() ? "sampled" : "greedy";
    std::fill(batch.logits(), batch.logits() + batch.rows * vocabulary,
              std::numeric_limits<float>::quiet_NaN());
    batch.poison();
    CommandGraph initial;
    sampling.addInitial(initial, {&policy, 1}, batch.buffers, 0, kStopTokens[0],
                        kStopTokens[1], {});
    static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
    require(batch.outputTokens()[0] == kSentinel,
            label + ": a first token from a non-finite row is " +
                std::to_string(batch.outputTokens()[0]));
    batch.poison();
    CommandGraph verify;
    sampling.addVerify(verify, {&policy, 1}, batch.buffers, kStopTokens[0],
                       kStopTokens[1], {});
    static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
    for (uint32_t row = 0; row < kRows; ++row)
      require(batch.outputTokens()[row] == kSentinel,
              label + ": verify row " + std::to_string(row) +
                  " of non-finite logits selected " +
                  std::to_string(batch.outputTokens()[row]));
    CommandGraph accept;
    const std::array<uint32_t, 1> maximumRetained{kRows};
    sampling.addAcceptance(accept, acceptance, maximumRetained, {&policy, 1},
                           kStopTokens[0], kStopTokens[1]);
    static_cast<void>(backend.submitCommandAsync(accept.dispatches()).wait());
    const uint32_t retained =
        *static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
    require(retained >= 1 && retained <= kRows &&
                batch.outputTokens()[retained - 1] == kSentinel,
            label + ": acceptance did not retain the sentinel as its anchor");
  }
}

// The host words: prompt bits only when repetition reads them, counts of
// every selected token (the generated history and the pending anchor), and
// no token outside the vocabulary or history without a prompt.
void penaltyWords() {
  std::vector<uint32_t> words(8, 0xDEADBEEFU);
  // A prompt of 1, 3, 3, 7, then the generated 3, 5, 5, 0.
  const std::vector<uint32_t> history{1, 3, 3, 7, 3, 5, 5, 0};
  Sampling::rebuildPenaltyWords(words, history, 4, std::nullopt, true);
  constexpr uint32_t kPrompt = SPLASH_PENALTY_PROMPT_BIT;
  require(words == std::vector<uint32_t>{1, kPrompt, 0, kPrompt + 1, 0, 2, 0,
                                         kPrompt},
          "penalty words with prompt bits differ");
  Sampling::rebuildPenaltyWords(words, history, 4, 6, true);
  require(words == std::vector<uint32_t>{1, kPrompt, 0, kPrompt + 1, 0, 2, 1,
                                         kPrompt},
          "penalty words with a pending token differ");
  Sampling::rebuildPenaltyWords(words, history, 4, std::nullopt, false);
  require(words == std::vector<uint32_t>{1, 0, 0, 1, 0, 2, 0, 0},
          "penalty words without prompt bits differ");
  const std::vector<uint32_t> step{5, 6};
  Sampling::countPenaltyTokens(words, step);
  require(words == std::vector<uint32_t>{1, 0, 0, 1, 0, 3, 1, 0},
          "counted penalty words differ");
  const std::vector<uint32_t> before = words;
  const std::vector<uint32_t> outside{2, 8};
  constexpr const char *kOutside = "penalty token is outside the vocabulary";
  rejects([&] { Sampling::countPenaltyTokens(words, outside); }, kOutside,
          "a counted token outside the vocabulary was accepted");
  rejects([&] {
    Sampling::rebuildPenaltyWords(words, outside, 0, std::nullopt, true);
  }, kOutside, "a prompt token outside the vocabulary was accepted");
  rejects([&] {
    Sampling::rebuildPenaltyWords(words, outside, 1, std::nullopt, false);
  }, kOutside, "a generated token outside the vocabulary was accepted");
  rejects([&] { Sampling::rebuildPenaltyWords(words, step, 0, 8, false); },
          kOutside, "a pending token outside the vocabulary was accepted");
  rejects([&] {
    Sampling::rebuildPenaltyWords(words, step, 2, std::nullopt, false);
  }, "request history holds no prompt",
          "a history without a prompt was accepted");
  require(words == before, "a refused token changed the penalty words");
}

// A request's lifetime as the model runtime keeps its words: activation loads
// the prompt, every selection counts the step's tokens (the first token after
// the prompt, a constrained request's first token, verify commits whose last
// token is the new anchor, cut at a stop token), and a terminal anchor is
// emitted without a step of its own. A resume rebuilds the words from the
// history the engine passes (the prompt, then the emitted outputs) and the
// pending anchor, and continues from them. Wherever a request can be
// suspended (during the prompt, at its end with and without a pending
// anchor, mid-generation, and again during the replay) the rebuild must
// equal the words counted step by step.
void penaltyLifecycle() {
  constexpr uint32_t vocabulary = 97;
  constexpr uint32_t kStop = 96;
  for (const bool markPrompt : {false, true}) {
    for (const bool constrained : {false, true}) {
      for (uint32_t seed = 0; seed < 8; ++seed) {
        Random random(0x6c696665 + seed * 4 + constrained * 2 + markPrompt);
        std::vector<uint32_t> prompt(40);
        for (uint32_t &token : prompt)
          token = random.next() % (vocabulary - 1);
        // The runtime's state: its slot's words, emitted count and anchor;
        // the engine's history, which resume passes as the new prompt.
        std::vector<uint32_t> words(vocabulary);
        std::vector<uint32_t> history = prompt;
        uint32_t generated = 0;
        std::optional<uint32_t> pending;
        Sampling::rebuildPenaltyWords(words, prompt, 0, std::nullopt,
                                      markPrompt);
        const auto commit = [&](std::span<const uint32_t> tokens) {
          Sampling::countPenaltyTokens(words, tokens);
          pending = tokens.back();
        };
        const auto emit = [&](std::span<const uint32_t> tokens) {
          history.insert(history.end(), tokens.begin(), tokens.end());
          generated += static_cast<uint32_t>(tokens.size());
        };
        const auto resume = [&](const char *stage) {
          std::vector<uint32_t> rebuilt(vocabulary, 0xA5A5A5A5U);
          Sampling::rebuildPenaltyWords(rebuilt, history, generated, pending,
                                        markPrompt);
          require(rebuilt == words,
                  std::string("resumed penalty words differ ") + stage);
          words = std::move(rebuilt);
        };
        resume("during the prompt");
        if (constrained)
          resume("while the first token waits for its mask");
        const uint32_t first = random.next() % (vocabulary - 1);
        commit({&first, 1});
        resume("with the first token pending");
        for (uint32_t cycle = 0; cycle < 9 && *pending != kStop; ++cycle) {
          const uint32_t retained = 1 + random.next() % kRows;
          std::vector<uint32_t> tokens(retained);
          for (uint32_t &token : tokens)
            token = random.next() % vocabulary;
          // Acceptance cuts the retained tokens after the first stop token.
          const auto stop = std::find(tokens.begin(), tokens.end(), kStop);
          if (stop != tokens.end())
            tokens.erase(stop + 1, tokens.end());
          std::vector<uint32_t> output{*pending};
          output.insert(output.end(), tokens.begin(), tokens.end() - 1);
          emit(output);
          commit(tokens);
          resume("mid-generation");
          resume("during the replay");
        }
        // The terminal anchor was counted when selected.
        emit({&*pending, 1});
        pending.reset();
        resume("after the terminal anchor");
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string stage = "setup";
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: target-sampling METALLIB");
    stage = "penalty words";
    penaltyWords();
    stage = "penalty lifecycle";
    penaltyLifecycle();
    MetalBackend backend(argv[1]);
    stage = "invalid penalties";
    invalidPenalties(backend);
    stage = "buffer extents";
    bufferExtents(backend);
    stage = "extreme penalties";
    extremes(backend);
    for (const uint32_t samplingMask : {0U, 1U, 2U, 3U}) {
      stage = "speculative exactness, sampling mask " +
              std::to_string(samplingMask);
      speculativeExactness(backend, samplingMask);
    }
    stage = "non-finite rows";
    nonFiniteRowsSelectTheSentinel(backend);
    stage = "ties";
    ties(backend);
    stage = "min_p cuts";
    minPCuts(backend);
    stage = "speculative acceptance over broad distributions";
    speculativeWholeVocabulary(backend);
    stage = "over-proposed residual";
    overProposedResidual(backend);
    stage = "extreme searches";
    extremeSearches(backend);
    stage = "unconstrained rows ignore masks";
    unconstrainedRowsIgnoreMasks(backend);
    for (const uint32_t vocabulary : {1003U, 248320U}) {
      uint32_t changedSelections = 0;
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        const std::string batch = " B" + std::to_string(lanes) +
                                  ", vocabulary " + std::to_string(vocabulary);
        stage = "mixed penalties" + batch;
        penalties(backend, vocabulary, lanes, false, changedSelections);
        stage = "argmax penalties" + batch;
        penalties(backend, vocabulary, lanes, true, changedSelections);
        stage = "sampled rows" + batch;
        sampledRows(backend, vocabulary, lanes, false);
        stage = "min_p rows" + batch;
        sampledRows(backend, vocabulary, lanes, true);
        stage = "top-1 rows" + batch;
        targetTop1(backend, vocabulary, lanes);
      }
      require(changedSelections > 0,
              "the penalties changed no greedy selection");
      stage = "excluded stop tokens, vocabulary " + std::to_string(vocabulary);
      excludedStopTokens(backend, vocabulary);
    }
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
      for (uint32_t mask = 0; mask < (1U << lanes); ++mask) {
        stage = "mixed verify B" + std::to_string(lanes) + ", sampling mask " +
                std::to_string(mask);
        mixedVerify(backend, lanes, mask);
      }
    }
    std::cout << "target_sampling_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "target_sampling_metal_test: FAIL (" << stage
              << "): " << error.what() << '\n';
    return 1;
  }
}
