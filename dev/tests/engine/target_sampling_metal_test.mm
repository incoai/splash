// Target sampling policy against a direct CPU reference.
//
// The penalty kernels must rewrite exactly the logits of the penalized lanes'
// prompt and output tokens, within float rounding of the host arithmetic,
// from each lane's own penalty table row, counting at a row its path's draft
// tokens but the anchor's; every other logit stays bitwise. The policy then
// selects from them, among the tokens each lane admits: a greedy row takes
// its argmax, and a sampled row draws from its distribution over the whole
// vocabulary (the tokens min_p leaves, the top-k of those, then top-p of
// their renormalized mass), row r with its lane's uniform r. Its largest
// logit and its draw must match an evaluation of the same rules in double:
// the draw must fall where the reference distribution's cumulative sum in
// token order places the uniform. Acceptance, which follows a verify block's
// rows while a row's selection is its child's token, must keep what a
// sequential decode would. Data within float rounding of a
// min_p or top_p cut or of a draw's boundary fail as ambiguous, or are
// allowed either way, instead of passing by chance. A lane selects the same,
// bit for bit, whatever lanes share its batch; a distribution of one token
// selects the masked argmax; and a lane that ignores end-of-sequence never
// selects a stop token. A sampled verify row that tests the children its
// draft drew selects as the recursion does in double, row by row; the
// recursion's selections, enumerated, are the target's distribution, and so
// are those of a tree's anchor and verified children, Monte Carlo, also
// where a floor on the rows' estimates leaves children unverified (a floor
// on estimates that read a row's own draws must fail that check); and a row
// whose target keeps only candidates selects one of them.
// Extreme repetition penalties saturate to exact, finite outcomes, and rows
// of non-finite logits select the sentinel 0xFFFFFFFF. The host
// word helpers and the lifecycle that rebuilds a resumed request's words are
// checked bitwise.
#include "ChainRows.hpp"
#include "PenaltyReference.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/Embedding.hpp"
#include "ops/LaneRows.hpp"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include "metal/abi/Sampling.h"

#include <algorithm>
#include <bit>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
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

// A decode lane's rows: the narrow cap's, or a verify block's where the build
// defines TARGET_SAMPLING_LANE_ROWS (target-sampling-16), whose rows past the
// eighth read uniforms, mask rows and paths no lane of the narrow cap reaches;
// and a lane's constraint-mask rows (metal/abi/Sampling.h), one more than a
// verify block's most rows.
#ifndef TARGET_SAMPLING_LANE_ROWS
#define TARGET_SAMPLING_LANE_ROWS splash::ops::kNarrowLaneRows
#endif
constexpr uint32_t kRows = TARGET_SAMPLING_LANE_ROWS;
constexpr uint32_t kMaskRows = SPLASH_TARGET_VERIFY_ROWS + 1;
// A lane's tree rows (the selector's draws and next probabilities of each),
// and the uniforms before its child tests: a verify block's.
constexpr uint32_t kBlock = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kUniforms = SPLASH_SAMPLING_UNIFORMS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kChildren = SPLASH_TREE_CHILDREN;
constexpr float kFloatMax = std::numeric_limits<float>::max();
// Both stop tokens sit below every spike of fillRow; their other places are
// SPLASH_NO_TOKEN.
constexpr StopTokens kStopTokens{1, 2};

using splash::test::chainRow;
using splash::test::penaltyMatches;
using splash::test::referencePenalty;
using splash::test::rejects;
using splash::test::rowsEnd;
using splash::test::require;

// As many stop tokens as the kernels compare, two in the first shard of the
// vocabulary and two in the last, so a lane that excludes them skips them in
// both.
StopTokens shardEdgeStopTokens(uint32_t vocabulary) {
  return {1, 2, vocabulary - 2, vocabulary - 1};
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

MetalBuffer filled(MetalBackend &backend, uint64_t bytes, uint8_t value) {
  MetalBuffer buffer = allocate(backend, bytes);
  std::memset(buffer.contents(), value, bytes);
  return buffer;
}

// Each of `lanes` lanes' verify block: its kRows rows at lane x kRows.
std::vector<LaneRows> blocks(size_t lanes) {
  std::vector<LaneRows> result;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    result.push_back({lane * kRows, kRows});
  return result;
}

// The step of `lanes` lanes' blocks().
StepRows blockStep(size_t lanes) {
  const auto count = static_cast<uint32_t>(lanes);
  return {count, count * kRows, kRows};
}

// Lanes' rows `rows` of a step, as its proposer lays them out on the device
// in `buffers` (their lane rows), each lane verifying every one of its rows;
// the step of them: its rows to the last lane's end, those in no lane's of no
// lane, at the lane cap a step of so many rows takes (verifyLaneCap), or a
// verify block's where a lane holds more rows than a draft's block.
StepRows layOut(const SamplingBuffers &buffers, std::span<const LaneRows> rows) {
  auto *laneRows = static_cast<LaneRows *>(buffers.laneRows.contents());
  uint32_t most = 0;
  for (uint32_t lane = 0; lane < rows.size(); ++lane) {
    laneRows[lane] = rows[lane];
    most = std::max(most, rows[lane].rows);
  }
  const auto lanes = static_cast<uint32_t>(rows.size()), end = static_cast<uint32_t>(rowsEnd(rows));
  return {lanes, end, most > kNarrowLaneRows ? SPLASH_TARGET_VERIFY_ROWS : verifyLaneCap(end, lanes)};
}

// The row each of `lanes` lanes' first token after a prompt is selected
// from, and where the selection writes it: lane x stride + offset, by
// default row `offset` of each lane's block of kRows.
std::vector<uint32_t> initialRows(size_t lanes, uint32_t offset, uint32_t stride = kRows) {
  std::vector<uint32_t> result;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    result.push_back(lane * stride + offset);
  return result;
}

// Each lane's verify block of `lanes` as a chain, in `tree`.
void writeChains(const MetalBuffer &tree, std::span<const LaneRows> lanes) {
  auto *rows = static_cast<TreeRow *>(tree.contents());
  for (const LaneRows &lane : lanes)
    for (uint32_t row = 0; row < lane.rows; ++row)
      rows[lane.first + row] = chainRow(row);
}

// Each lane's verify block of `lanes` as a chain.
MetalBuffer chainTree(MetalBackend &backend, std::span<const LaneRows> lanes) {
  MetalBuffer buffer = allocate(backend, rowsEnd(lanes) * sizeof(TreeRow));
  writeChains(buffer, lanes);
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
  return {SamplingBuffers{
              allocate(backend, uint64_t{rows} * vocabulary * sizeof(float)),
              allocate(backend, space.partialMassesBytes),
              allocate(backend, space.vocabularyRowsBytes),
              allocate(backend, uint64_t{lanes} * kUniforms * sizeof(float)),
              allocate(backend,
                       uint64_t{lanes} * kMaskRows * ((vocabulary + 31) / 32) * 4),
              allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
              allocate(backend, space.argmaxValuesBytes),
              allocate(backend, space.argmaxIndicesBytes),
              allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
              chainTree(backend, blocks(lanes)),
              allocate(backend, space.vocabularyRangesBytes),
              allocate(backend, space.vocabularyArrivalsBytes),
              allocate(backend, uint64_t{lanes} * sizeof(LaneRows)),
              // No row tests children unless a test gives it some.
              filled(backend, uint64_t{rows} * sizeof(uint32_t), 0xff),
              allocate(backend, uint64_t{lanes} * kPositions * kCandidates * sizeof(uint32_t)),
              filled(backend, uint64_t{lanes} * kBlock * kChildren, 0xff),
              allocate(backend, uint64_t{lanes} * kBlock * kCandidates * sizeof(float))},
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
    return !excludesStop || !kStopTokens.contains(token);
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

// One verify row of a sampled lane: its largest logit and its draw from its
// whole distribution with its lane's uniform of the row.
void requireSampledRow(const Batch &batch, uint32_t index,
                       const Distribution &target, const std::string &label) {
  const TargetVocabularyRow &record = batch.record(index);
  requireArrivalsReturned(batch, index, label);
  requireMaximum(record, target, label);
  requireDraw(batch.outputTokens()[index], target.probabilities,
              batch.uniforms()[index / kRows * kUniforms + index % kRows], target.ambiguousMass, label);
}

// The first token after a prompt of a sampled lane, drawn from its row with
// the lane's first uniform; a selection writes it at its row.
void requireInitialDraw(const Batch &batch, uint32_t lane, uint32_t row,
                        const Distribution &target, const std::string &label) {
  requireArrivalsReturned(batch, row, label);
  requireMaximum(batch.record(row), target, label);
  requireDraw(batch.outputTokens()[row], target.probabilities,
              batch.uniforms()[lane * kUniforms], target.ambiguousMass, label);
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
// originals: penalized from the lane's words, or bitwise unchanged. Lane 0's
// rows are the batch's rows, whatever lane they hold.
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
      batch.row(row)[kStopTokens.ids[0]] = 13.0F;
  }
  for (uint32_t index = 0; index < lanes * kMaskRows * batch.maskWords();
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
  sampling.addVerify(verify, policies, batch.buffers, layOut(batch.buffers, blocks(policies.size())),
                     kStopTokens, {table, rows});
  static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requirePenalizedRows(batch, original, lane, 0, kRows, words(lane),
                         policies[lane].penalties, true, label + " verify");
    for (uint32_t row = lane * kRows; row < (lane + 1) * kRows; ++row) {
      const SamplingPolicy &policy = policies[lane];
      const Admission admits{
          policy.constrained
              ? batch.masks() + (uint64_t{lane} * kMaskRows + row % kRows + 1) *
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

  // The first token after each lane's prompt: one row per lane, no draft
  // tokens; every other row keeps its logits. Row = lane, as a command's
  // heads write their lanes' final hidden (Runtime::submitInitialSelection),
  // and a row at an offset into blocks of kRows, which the kernels take
  // alike, reading a lane's state by its lane and a row's by its row.
  for (const auto &[stride, offset] : {std::pair{1U, 0U}, std::pair{kRows, 5U}}) {
    const std::vector<uint32_t> selectedRows = initialRows(policies.size(), offset, stride);
    std::copy(original.begin(), original.end(), batch.logits());
    batch.poison();
    CommandGraph initial;
    sampling.addInitial(initial, policies, batch.buffers, selectedRows, kStopTokens, {table, rows});
    static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
    const std::string layout = " at lane x " + std::to_string(stride) + " + " + std::to_string(offset);
    for (uint32_t row = 0; row < lanes * kRows; ++row)
      if (std::ranges::find(selectedRows, row) == selectedRows.end())
        requirePenalizedRows(batch, original, 0, row, 1, {}, {}, false, label + " initial" + layout + ", unselected row");
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const SamplingPolicy &policy = policies[lane];
      const std::string lanePrefix = label + " initial lane " + std::to_string(lane) + layout;
      const uint32_t row = selectedRows[lane];
      requirePenalizedRows(batch, original, 0, row, 1, words(lane), policy.penalties, false, lanePrefix);
      const float *selected = batch.row(row);
      const Admission admits{
          policy.constrained
              ? batch.masks() + uint64_t{lane} * kMaskRows * batch.maskWords()
              : nullptr,
          policy.excludesStopTokens};
      if (policy.samples())
        requireInitialDraw(batch, lane, row, referenceDistribution(selected, vocabulary, policy, admits), lanePrefix);
      else
        require(batch.outputTokens()[row] == referenceArgmax(selected, vocabulary, admits),
                lanePrefix + ": the initial token lost its penalized argmax");
    }
  }
}

// Acceptance of `lanes` lanes' chains over a batch's verify input,
// selections and lane rows (layOut).
AcceptanceBuffers acceptanceBuffers(MetalBackend &backend, const Batch &batch, uint32_t lanes) {
  return {batch.buffers.inputTokens, batch.buffers.treeRows, batch.buffers.outputTokens,
          allocate(backend, lanes * sizeof(uint32_t)), allocate(backend, lanes * sizeof(uint32_t)),
          allocate(backend, lanes * sizeof(uint32_t)), batch.buffers.laneRows};
}

// The token a uniform draws from weights in token order, failing on data
// within float rounding of an interval's end.
uint32_t referenceDraw(const std::vector<double> &weights, double uniform) {
  const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
  const double threshold = uniform * total;
  double cumulative = 0.0;
  uint32_t token = 0;
  while (cumulative + weights[token] <= threshold)
    cumulative += weights[token++];
  require(threshold - cumulative > 1e-5 * total && cumulative + weights[token] - threshold > 1e-5 * total,
          "test data is ambiguous at a draw");
  return token;
}

// The uniform at the middle of a token's interval of weights in token order.
float intervalMiddle(const std::vector<double> &weights, uint32_t token) {
  const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
  const double before = std::accumulate(weights.begin(), weights.begin() + token, 0.0);
  return float((before + 0.5 * weights[token]) / total);
}

// A request's DFlash cycle against a sequential decode. Every verify row of
// a lane has nearly the same logits, as a model that would repeat itself,
// so only the draft tokens its context adds keep row r from selecting what
// row 0 did. Row r must select from the distribution a non-speculative decode
// would, given the context through draft token r - 1; the drafts are what
// that decode selects up to a lane's first wrong one, and acceptance must
// stop there with the same next token. A sampled lane's row r draws with its
// uniform r, which the test places on its right draft token's interval of the
// distribution, so its draws follow the drafts until the wrong one, which
// its draw there, at 0.37 of the mass, does not take.
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
  const AcceptanceBuffers acceptance = acceptanceBuffers(backend, batch, lanes);

  std::vector<SamplingPolicy> policies;
  std::array<std::array<uint32_t, kRows>, lanes> expectedOutput{};
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
      // The draft: the best penalized candidate until the wrong position,
      // where it takes the eighth best, whose probability is small.
      const uint32_t draft = row < kWrongAt[lane] ? order[0] : order[7];
      uint32_t selection = order[0];
      if (sampled) {
        const std::vector<double> weights =
            referenceDistribution(penalized.data(), vocabulary, policies[lane]).probabilities;
        require(row >= kWrongAt[lane] || weights[draft] > 1e-3,
                "a sampled lane drafted a token its target barely draws");
        uniforms[row] = row < kWrongAt[lane] ? intervalMiddle(weights, draft) : 0.37F;
        selection = referenceDraw(weights, uniforms[row]);
      }
      if (row <= kWrongAt[lane])
        expectedOutput[lane][row] = selection;
      if (row == kPositions)
        break;
      require(row != kWrongAt[lane] || selection != draft, "the wrong draft is the target's selection");
      inputs[row + 1] = draft;
      ++counts[draft];
    }
  }

  batch.poison();
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch.buffers, layOut(batch.buffers, blocks(policies.size())),
                     kStopTokens, {table, kSlots});
  const std::array<uint32_t, lanes> maximumRetained{kRows, kRows};
  sampling.addAcceptance(graph, acceptance, blockStep(policies.size()), maximumRetained, kStopTokens);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *retained =
      static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
  const auto *acceptedCounts =
      static_cast<const uint32_t *>(acceptance.acceptedCounts.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const std::string label = "speculative lane " + std::to_string(lane) +
                              " sampling mask " + std::to_string(samplingMask);
    const uint32_t accepted = kWrongAt[lane];
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
        batch.uniforms()[0] = uniform;
        const SamplingPolicy policy{temperature > 0.0F ? kTopK : 1U,
                                    temperature,
                                    1.0F,
                                    c.masked,
                                    false,
                                    {c.repetition, 0.0F, 0.0F}};
        CommandGraph graph;
        sampling.addInitial(graph, {&policy, 1}, batch.buffers, initialRows(1, 0),
                            kStopTokens, {table, rows});
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
// lane, a constrained sampled lane, whose rows draw with its uniforms, and a
// greedy lane penalized from table row 1, whose penalty reads its verify
// input rows. The acceptance of the three verify lanes, which reads every
// row's token, tree row and selection.
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
      sampling.addInitial(graph, policies, buffers, initialRows(policies.size(), 3), kStopTokens, {words, tableRows});
    };
    // The rows up to row 3 of the last lane's block, the third, and their
    // tree rows, which each lane's root takes; the constrained lane's mask row
    // and the sampled lane's uniform.
    const auto rows = workspace(2 * kRows + 3 + 1);
    requireExtents({{&SamplingBuffers::logits, (2 * kRows + 3 + 1) * logitsRow, 4, "sampling logits"},
                    {&SamplingBuffers::constraintMasks, (kMaskRows + 1) * maskRow, 4, "constraint mask"},
                    {&SamplingBuffers::outputTokens, (2 * kRows + 3 + 1) * 4, 4, "sampled token"},
                    {&SamplingBuffers::argmaxValues, rows[0], 4, "argmax value"},
                    {&SamplingBuffers::argmaxIndices, rows[1], 4, "argmax index"},
                    {&SamplingBuffers::partialMasses, rows[2], sizeof(TargetShardMass), "sampling mass"},
                    {&SamplingBuffers::vocabularyRows, rows[3], sizeof(TargetVocabularyRow),
                     "sampling vocabulary row"},
                    {&SamplingBuffers::vocabularyRanges, rows[4], sizeof(TargetVocabularyRange),
                     "sampling vocabulary range"},
                    {&SamplingBuffers::vocabularyArrivals, rows[5], 4, "sampling arrival"},
                    {&SamplingBuffers::uniforms, 4, 4, "sampling uniform"},
                    {&SamplingBuffers::laneRows, lanes * sizeof(LaneRows), sizeof(LaneRows), "sampling lane rows"},
                    {&SamplingBuffers::treeRows, (2 * kRows + 3 + 1) * sizeof(TreeRow), sizeof(TreeRow),
                     "verify tree row"}},
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
      sampling.addVerify(graph, policies, buffers, blockStep(lanes), kStopTokens, {words, tableRows});
    };
    // Every row of the step, each lane's up to a block's: its mask rows and
    // uniforms (a sampled lane's child tests' too), its verify input and tree
    // rows, which its penalties read, and what its sampled rows test: their
    // sources and the lanes' draft candidates, draws and next probabilities.
    const auto rows = workspace(3 * kRows);
    requireExtents(
        {{&SamplingBuffers::logits, 3 * kRows * logitsRow, 4, "sampling logits"},
         {&SamplingBuffers::constraintMasks, (kMaskRows + 1 + kRows) * maskRow, 4, "constraint mask"},
         {&SamplingBuffers::outputTokens, lanes * kRows * 4, 4, "sampled token"},
         {&SamplingBuffers::argmaxValues, rows[0], 4, "argmax value"},
         {&SamplingBuffers::argmaxIndices, rows[1], 4, "argmax index"},
         {&SamplingBuffers::partialMasses, rows[2], sizeof(TargetShardMass), "sampling mass"},
         {&SamplingBuffers::vocabularyRows, rows[3], sizeof(TargetVocabularyRow), "sampling vocabulary row"},
         {&SamplingBuffers::vocabularyRanges, rows[4], sizeof(TargetVocabularyRange), "sampling vocabulary range"},
         {&SamplingBuffers::vocabularyArrivals, rows[5], 4, "sampling arrival"},
         {&SamplingBuffers::uniforms, (kUniforms + kBlock + kRows * kChildren) * 4, 4, "sampling uniform"},
         {&SamplingBuffers::inputTokens, lanes * kRows * 4, 4, "verify input token"},
         {&SamplingBuffers::treeRows, lanes * kRows * sizeof(TreeRow), sizeof(TreeRow), "verify tree row"},
         {&SamplingBuffers::laneRows, lanes * sizeof(LaneRows), sizeof(LaneRows), "sampling lane rows"},
         {&SamplingBuffers::verifySources, lanes * kRows * 4, 4, "verify source"},
         {&SamplingBuffers::candidates, lanes * kPositions * kCandidates * 4, 4, "draft candidate"},
         {&SamplingBuffers::treeDraws, lanes * kBlock * kChildren, 1, "draft tree draw"},
         {&SamplingBuffers::treeNext, lanes * kBlock * kCandidates * 4, 4, "draft tree next"}},
        [&](CommandGraph &graph, const SamplingBuffers &buffers) { encode(graph, buffers, table); });
    splash::test::requireExtent(backend, table, 2 * logitsRow, 4, "penalty table",
                                [&](CommandGraph &graph, const MetalBuffer &words) {
                                  encode(graph, batch.buffers, words);
                                });
  }

  const AcceptanceBuffers acceptance = acceptanceBuffers(backend, batch, lanes);
  const std::array<uint32_t, lanes> retained{kRows, kRows, 5};
  for (const auto &[member, bytes, element, what] :
       std::initializer_list<std::tuple<MetalBuffer AcceptanceBuffers::*, uint64_t, uint64_t, const char *>>{
           {&AcceptanceBuffers::inputTokens, lanes * kRows * 4, 4, "verify input token"},
           {&AcceptanceBuffers::treeRows, lanes * kRows * sizeof(TreeRow), sizeof(TreeRow), "verify tree row"},
           {&AcceptanceBuffers::outputTokens, lanes * kRows * 4, 4, "target token"},
           {&AcceptanceBuffers::retainedCounts, lanes * 4, 4, "retained count"},
           {&AcceptanceBuffers::acceptedCounts, lanes * 4, 4, "accepted count"},
           {&AcceptanceBuffers::retainedRows, lanes * 4, 4, "retained rows"},
           {&AcceptanceBuffers::laneRows, lanes * sizeof(LaneRows), sizeof(LaneRows), "lane rows"}})
    splash::test::requireExtent(backend, acceptance.*member, bytes, element, what,
                                [&](CommandGraph &graph, const MetalBuffer &buffer) {
                                  AcceptanceBuffers changed = acceptance;
                                  changed.*member = buffer;
                                  sampling.addAcceptance(graph, changed, blockStep(lanes), retained, kStopTokens);
                                });
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
    sampling.addVerify(graph, {&penalized, 1}, batch.buffers, layOut(batch.buffers, blocks(1)), kStopTokens, {});
  }, "penalized lane has no penalty table row",
          "penalties without a table were accepted");
  rejects([&] {
    sampling.addVerify(graph, {&penalized, 1}, batch.buffers, layOut(batch.buffers, blocks(1)), kStopTokens,
                       {table, outside});
  }, "penalty table buffer holds",
          "a table row outside the table was accepted");
  rejects([&] {
    sampling.addInitial(graph, {&penalized, 1}, batch.buffers, initialRows(1, 0), kStopTokens,
                        {table, outside});
  }, "penalty table buffer holds",
          "an initial table row outside the table was accepted");
  require(graph.empty(), "a refused penalty request encoded a dispatch");
  // Unpenalized lanes need no table row.
  const SamplingPolicy greedy{1, 0.0F, 1.0F, false};
  sampling.addVerify(graph, {&greedy, 1}, batch.buffers, layOut(batch.buffers, blocks(1)), kStopTokens, {});
  require(!graph.empty(), "an unpenalized batch encoded nothing");
}

// The draft token of one verify position, which verify input row
// position + 1 holds.
void setDraft(const Batch &batch, uint32_t lane, uint32_t position, uint32_t draft) {
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
  for (uint32_t index = 0; index < lanes * kMaskRows * batch.maskWords();
       ++index)
    batch.masks()[index] = 0xB6DB6DB6U ^ index;
  const auto admission = [&](uint32_t lane, uint32_t row) {
    return Admission{policies[lane].constrained
                         ? batch.masks() +
                               (uint64_t{lane} * kMaskRows + row + 1) *
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
        batch.row(index)[kStopTokens.ids[0]] = 13.0F;
      if (policies[lane].samples())
        references[index] =
            referenceDistribution(batch.row(index), vocabulary,
                                  policies[lane], admission(lane, row));
    }
    // Draft tokens from the start and the middle of the distribution, and a
    // random one, which it may not keep: none changes a draw.
    for (uint32_t position = 0; position < kPositions; ++position) {
      const std::vector<uint32_t> &kept = references[lane * kRows + position].order;
      std::vector<uint32_t> drafts;
      if (!kept.empty())
        drafts = {kept.front(), kept[kept.size() / 2]};
      drafts.push_back(3 + random.next() % (vocabulary - 3));
      setDraft(batch, lane, position, drafts[drafts.size() == 3 ? position % 3 : 0]);
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
  sampling.addVerify(verify, policies, batch.buffers, layOut(batch.buffers, blocks(policies.size())), kStopTokens, {});
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
      batch.uniforms()[0] = 0.5F * (random.unit() + 1.0F);
      CommandGraph initial;
      sampling.addInitial(initial, {&policy, 1}, batch.buffers, initialRows(1, offset),
                          kStopTokens, {});
      static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
      const Admission admits{policy.constrained ? batch.masks() : nullptr,
                             policy.excludesStopTokens};
      const std::string rowLabel = label + " initial top_k " +
                                   std::to_string(policy.topK) + " min_p " +
                                   std::to_string(policy.minP) + " offset " +
                                   std::to_string(offset);
      if (policy.samples())
        requireInitialDraw(batch, 0, offset,
                           referenceDistribution(batch.row(offset), vocabulary,
                                                 policy, admits),
                           rowLabel);
      else
        require(batch.outputTokens()[offset] ==
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
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    policies.push_back(
        {8 + lane, (samplingMask & (1U << lane)) ? 0.7F : 0.0F, 0.9F, false});
    for (uint32_t row = 0; row < kRows; ++row)
      fillShaped(batch.row(lane * kRows + row), vocabulary,
                 row % 2 ? Shape::Ties : Shape::Peaked, random);
  }
  for (uint32_t row = 0; row < batch.rows; ++row)
    batch.inputTokens()[row] = random.next() % vocabulary;
  for (uint32_t uniform = 0; uniform < lanes * kUniforms; ++uniform)
    batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
  const auto stops = shardEdgeStopTokens(vocabulary);
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch.buffers, layOut(batch.buffers, blocks(policies.size())), stops, {});
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
    copy(batch.buffers.uniforms, single.buffers.uniforms);
    CommandGraph reference;
    sampling.addVerify(reference, std::span(policies).subspan(lane, 1),
                       single.buffers, layOut(single.buffers, blocks(1)), stops, {});
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

// Lanes of different row counts, packed one after another (5, 6, 1 and 6
// rows from rows 0, 5, 13 and 14, rows 11 and 12 of no lane), select and
// accept as each lane does alone in a block from row 0, bit for bit: its
// penalized logits, tokens and sampled records, and its retained and
// accepted counts and rows. A lane whose thinking cutoff is at its first row
// proposes its end there, as its selector's host chain does. The lanes mix
// greedy and sampled policies, penalties, a mask, stop exclusion, a
// retention limit below a lane's rows, and thinking cutoffs at a lane's first
// row and inside one; a lane of one row checks no proposal. No kernel
// selects, penalizes or accepts a row of no lane.
void packedLanes(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003, lanes = kLanes;
  const std::array<LaneRows, lanes> packed{{{0, 5}, {5, 6}, {13, 1}, {14, 6}}};
  const std::array<uint32_t, lanes> tableRows{2, 0, 3, 1}, retainedLimits{5, 8, 1, 4};
  const std::array<uint32_t, 3> end{vocabulary - 6, vocabulary - 7, vocabulary - 3};
  std::array<SamplingPolicy, lanes> policies{
      SamplingPolicy{}, SamplingPolicy{32, 0.8F, 0.95F, true, false, {1.3F, 0.5F, 0.25F}},
      SamplingPolicy{1, 0.0F, 1.0F, false, true},
      SamplingPolicy{0, 1.0F, 0.9F, false, false, {0.8F, 0.0F, 0.0F}, 0.05F}};
  policies[0].thinkingCutoff = ThinkingCutoff{0, end, vocabulary - 5};
  policies[2].thinkingCutoff = ThinkingCutoff{0, end, vocabulary - 5};
  policies[3].thinkingCutoff = ThinkingCutoff{3, end, vocabulary - 5};
  const auto stops = shardEdgeStopTokens(vocabulary);
  Sampling sampling(vocabulary);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  Random random(0x7061636b);

  // A batch of `count` lanes and its acceptance buffers.
  struct Run final {
    Batch batch;
    AcceptanceBuffers acceptance;
  };
  const auto makeRun = [&](uint32_t count, std::span<const LaneRows> rows) {
    Batch batch = makeBatch(backend, vocabulary, count);
    batch.buffers.treeRows = chainTree(backend, rows);
    AcceptanceBuffers acceptance = acceptanceBuffers(backend, batch, count);
    return Run{batch, acceptance};
  };
  const auto words = [](const MetalBuffer &buffer) { return static_cast<uint32_t *>(buffer.contents()); };
  const auto run = [&](const Run &r, std::span<const SamplingPolicy> lanePolicies, std::span<const LaneRows> rows,
                       std::span<const uint32_t> laneTableRows, std::span<const uint32_t> limits) {
    r.batch.poison();
    const StepRows step = layOut(r.batch.buffers, rows);
    CommandGraph graph;
    sampling.addVerify(graph, lanePolicies, r.batch.buffers, step, stops, {table, laneTableRows});
    sampling.addAcceptance(graph, r.acceptance, step, limits, stops);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  };

  Run all = makeRun(lanes, packed);
  for (uint32_t row = 0; row < all.batch.rows; ++row)
    fillShaped(all.batch.row(row), vocabulary, row % 2 ? Shape::Ties : Shape::Peaked, random);
  for (uint32_t index = 0; index < lanes * kMaskRows * all.batch.maskWords(); ++index)
    all.batch.masks()[index] = 0xB6DB6DB6U ^ (index * 2654435761U);
  for (uint32_t uniform = 0; uniform < lanes * kUniforms; ++uniform)
    all.batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const LaneRows block = packed[lane];
    all.batch.inputTokens()[block.first] = 7 + lane;
    // Each lane's draft proposes its rows' best tokens, but lane 1's third;
    // a thinking end at a lane's first row takes the end's.
    const std::optional<ThinkingCutoff> &cutoff = policies[lane].thinkingCutoff;
    for (uint32_t row = 1; row < block.rows; ++row) {
      const uint32_t best = referenceArgmax(all.batch.row(block.first + row - 1), vocabulary);
      all.batch.inputTokens()[block.first + row] =
          cutoff && row <= cutoff->replacedProposals() ? cutoff->end[row - 1]
          : lane == 1 && row == 3                     ? (best + 1) % vocabulary
                                                      : best;
    }
  }
  const std::vector<float> logits(all.batch.logits(), all.batch.logits() + uint64_t{all.batch.rows} * vocabulary);
  const std::vector<uint32_t> inputs(all.batch.inputTokens(), all.batch.inputTokens() + all.batch.rows);
  run(all, policies, packed, tableRows, retainedLimits);

  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const LaneRows block = packed[lane];
    const std::array<LaneRows, 1> own{{{0, block.rows}}};
    const Run alone = makeRun(1, own);
    std::copy_n(logits.begin() + uint64_t{block.first} * vocabulary, uint64_t{block.rows} * vocabulary,
                alone.batch.logits());
    std::copy_n(all.batch.masks() + uint64_t{lane} * kMaskRows * all.batch.maskWords(),
                kMaskRows * all.batch.maskWords(), alone.batch.masks());
    std::copy_n(all.batch.uniforms() + lane * kUniforms, kUniforms, alone.batch.uniforms());
    std::copy_n(inputs.begin() + block.first, block.rows, alone.batch.inputTokens());
    run(alone, std::span(policies).subspan(lane, 1), own, std::span(tableRows).subspan(lane, 1),
        std::span(retainedLimits).subspan(lane, 1));

    const std::string label = "packed lane " + std::to_string(lane);
    const auto same = [](const auto *packedValues, const auto *aloneValues, uint64_t count) {
      return std::memcmp(packedValues, aloneValues, count * sizeof(*packedValues)) == 0;
    };
    require(same(all.batch.row(block.first), alone.batch.logits(), uint64_t{block.rows} * vocabulary),
            label + ": its penalized logits differ from alone");
    require(same(all.batch.outputTokens() + block.first, alone.batch.outputTokens(), block.rows),
            label + ": its tokens differ from alone");
    if (policies[lane].samples())
      require(same(&all.batch.record(block.first), &alone.batch.record(0), block.rows),
              label + ": its sampled records differ from alone");
    require(words(all.acceptance.retainedCounts)[lane] == words(alone.acceptance.retainedCounts)[0] &&
                words(all.acceptance.acceptedCounts)[lane] == words(alone.acceptance.acceptedCounts)[0] &&
                words(all.acceptance.retainedRows)[lane] == words(alone.acceptance.retainedRows)[0],
            label + ": its acceptance differs from alone");
  }
  // The greedy lane accepted the three end tokens it proposed and its next
  // best token, and the one-row lane, whose row took the end's first token,
  // none.
  require(words(all.acceptance.acceptedCounts)[0] == 4 && words(all.acceptance.acceptedCounts)[2] == 0 &&
              words(all.acceptance.retainedCounts)[2] == 1 && all.batch.outputTokens()[13] == end[0],
          "packed lanes accepted other than their drafts and ends");
  // No kernel selects a row of no lane (rows 11 and 12, after lane 1's
  // penalized and sampled rows): they keep their logits and their tokens'
  // poison.
  for (const uint32_t at : {11U, 12U})
    require(all.batch.outputTokens()[at] == 0xA5A5A5A5U &&
                std::memcmp(all.batch.row(at), logits.data() + uint64_t{at} * vocabulary, vocabulary * 4) == 0,
            "row " + std::to_string(at) + " of no lane was selected or penalized");
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
    for (uint32_t uniform = 0; uniform < lanes * kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    std::fill_n(batch.masks(), lanes * kMaskRows * batch.maskWords(),
                maskWord);
    CommandGraph verify;
    sampling.addVerify(verify, policies, batch.buffers, layOut(batch.buffers, blocks(policies.size())), stops, {});
    static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
    std::vector<uint32_t> tokens(batch.outputTokens(),
                                 batch.outputTokens() + batch.rows);
    CommandGraph initial;
    sampling.addInitial(initial, policies, batch.buffers, initialRows(policies.size(), 3), stops, {});
    static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
    for (const uint32_t row : initialRows(lanes, 3))
      tokens.push_back(batch.outputTokens()[row]);
    return tokens;
  };
  require(select(0) == select(std::numeric_limits<uint32_t>::max()),
          "an unconstrained lane read its constraint mask");
}

// Constrained greedy lanes (the argmax kernels) and constrained sampled
// lanes with top-k 1 (a distribution of one token) must agree with a
// full-vocabulary CPU argmax, including ties, row offsets and masks: the
// first token, every greedy verify row's token, and every sampled verify
// row's draw. Every scratch and output buffer is poisoned, so nothing passes
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
  for (uint32_t row = 0; row < lanes * kMaskRows; ++row)
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
      sampling.addInitial(initial, {&single, 1}, batch.buffers, initialRows(1, offset),
                          stops, {});
      static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
      require(tokens[offset] == expected(offset, 0),
              "initial target differs from masked CPU argmax");
    }
    batch.poison();
    CommandGraph initialArgmax;
    sampling.addInitial(initialArgmax, {&greedy, 1}, batch.buffers, initialRows(1, offset),
                        stops, {});
    static_cast<void>(backend.submitCommandAsync(initialArgmax.dispatches()).wait());
    require(tokens[offset] == expected(offset, kUnmasked),
            "initial argmax differs from CPU argmax");
  }
  // Odd verify rows draft their argmax, even ones another token.
  for (uint32_t row = 0; row < batch.rows; ++row) {
    if (row % kRows == kRows - 1)
      continue;
    const uint32_t maskRow = row / kRows * kMaskRows + row % kRows + 1;
    const uint32_t id = expected(row, maskRow);
    batch.inputTokens()[row + 1] = row % 2 ? id : (id + 1) % vocabulary;
  }
  batch.poison();
  std::vector<SamplingPolicy> policies(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    policies[lane] = {1, lane % 2 ? 0.8F : 0.0F, 0.5F, true};
  CommandGraph verify;
  sampling.addVerify(verify, policies, batch.buffers, layOut(batch.buffers, blocks(policies.size())), stops, {});
  static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
  for (uint32_t row = 0; row < batch.rows; ++row) {
    const uint32_t maskRow = row / kRows * kMaskRows + row % kRows + 1;
    const uint32_t id = expected(row, maskRow);
    if (!policies[row / kRows].samples()) {
      require(tokens[row] == id,
              "batched target differs from masked CPU argmax");
      continue;
    }
    require(tokens[row] == id,
            "a sampled top-1 row drew other than its masked CPU argmax");
  }
  batch.poison();
  CommandGraph verifyArgmax;
  sampling.addVerify(verifyArgmax, std::vector<SamplingPolicy>(lanes, greedy),
                     batch.buffers, layOut(batch.buffers, blocks(lanes)), stops, {});
  static_cast<void>(backend.submitCommandAsync(verifyArgmax.dispatches()).wait());
  for (uint32_t row = 0; row < batch.rows; ++row)
    require(tokens[row] == expected(row, kUnmasked),
            "batched argmax differs from CPU argmax");
}

// A lane that ignores end-of-sequence never selects a stop token: not in the
// sample after a prefill chunk, not in any verify row, greedy or sampled,
// alone or batched beside lanes that keep them, and a stop token its draft
// proposes is rejected. The four stop tokens lead every row, the one in the
// last place far ahead, so a lane that keeps them selects it, and accepts it
// where its draft proposed it, and acceptance cuts its tokens after it; the
// other lanes take the row's best remaining token. A list of two stop tokens
// pads its other places with SPLASH_NO_TOKEN, which excludes no token.
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
    for (uint32_t place = 0; place < SPLASH_STOP_TOKENS; ++place)
      values[stops.ids[place]] = place + 1 == SPLASH_STOP_TOKENS ? 12.0F : 10.5F;
  }
  float *uniforms = batch.uniforms();
  const uint32_t *tokens = batch.outputTokens();
  const uint32_t leading = stops.ids[SPLASH_STOP_TOKENS - 1];
  const auto stop = [&](uint32_t token) { return stops.contains(token); };

  for (const bool excludes : {false, true}) {
    for (const float temperature : {0.0F, 0.8F}) {
      for (const float uniform : {0.0F, 0.5F, 0.999F}) {
        batch.poison();
        uniforms[0] = uniform;
        CommandGraph initial;
        const SamplingPolicy policy{32, temperature, 1.0F, false, excludes};
        sampling.addInitial(initial, {&policy, 1}, batch.buffers, initialRows(1, kRows - 1),
                            stops, {});
        static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
        const uint32_t token = tokens[kRows - 1];
        if (excludes)
          require(token == best(kRows - 1) ||
                      (temperature > 0.0F && token < vocabulary && !stop(token)),
                  "an excluding lane selected a stop token after prefill");
        else
          require(stop(token), "the initial sample skipped the stop tokens");
      }
    }
  }

  // Drafts propose the leading stop token, verify input row 1 of each lane,
  // which a lane that keeps the stop tokens selects or, sampled, draws at
  // row 0 with the uniforms at 0.5.
  const AcceptanceBuffers acceptance = acceptanceBuffers(backend, batch, kLanes);
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    batch.inputTokens()[lane * kRows + 1] = leading;
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
        sampling.addVerify(verify, policies, batch.buffers, layOut(batch.buffers, blocks(policies.size())), stops, {});
        static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
        // Only an excluding lane's distribution leaves out the stop token
        // drafted at row 0, and none of its rows draws a stop token.
        for (uint32_t row = 0; row < lanes * kRows; ++row) {
          const SamplingPolicy &policy = policies[row / kRows];
          if (!policy.samples()) {
            require(tokens[row] ==
                        (policy.excludesStopTokens ? best(row) : leading),
                    "a greedy verify row mishandled the stop tokens");
            continue;
          }
          require(row % kRows != 0 || (tokens[row] == leading) != policy.excludesStopTokens,
                  "a verify distribution mishandled the stop tokens");
          require(!policy.excludesStopTokens ||
                      (tokens[row] < vocabulary && !stop(tokens[row])),
                  "an excluding verify row drew a stop token");
        }
        CommandGraph accept;
        sampling.addAcceptance(accept, acceptance, blockStep(policies.size()),
                               std::span(maximumRetained).first(lanes), stops);
        static_cast<void>(backend.submitCommandAsync(accept.dispatches()).wait());
        const auto *retained =
            static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          const uint32_t *output = tokens + lane * kRows;
          if (!policies[lane].excludesStopTokens) {
            require(retained[lane] == 1 && output[0] == leading,
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

  // Two stop tokens lead every row and token 0 follows them: a greedy lane
  // that ignores end-of-sequence skips the two and selects token 0, which a
  // padding of zeros would skip too.
  const StopTokens two{leading, stops.ids[SPLASH_STOP_TOKENS - 2]};
  for (uint32_t row = 0; row < batch.rows; ++row) {
    float *values = batch.row(row);
    for (uint32_t place = 0; place + 2 < SPLASH_STOP_TOKENS; ++place) values[stops.ids[place]] = -4.0F;
    values[0] = 10.0F;
  }
  const std::vector<SamplingPolicy> greedy(kLanes, {32, 0.0F, 1.0F, false, true});
  batch.poison();
  CommandGraph padded;
  sampling.addVerify(padded, greedy, batch.buffers, layOut(batch.buffers, blocks(greedy.size())), two, {});
  static_cast<void>(backend.submitCommandAsync(padded.dispatches()).wait());
  for (uint32_t row = 0; row < batch.rows; ++row)
    require(tokens[row] == 0, "the padding of two stop tokens excluded another token");
}

// A lane's thinking cutoff (SamplingPolicy::thinkingCutoff) takes the end of
// thinking from its row on whatever the rows' logits, as many of the end's
// tokens as the selection's rows hold: the first token after a prompt at row
// 0, or verify rows, whose draft acceptance keeps a drafted row only where it
// drafted that row's end token. A draft token before the cutoff that ended
// thinking first, the end's last token, which closes thinking, or the other
// ending token, leaves the rows their own selections, and a cutoff past the
// selection's rows dispatches nothing. At a lane's row 0 the end tokens are
// its proposals (the selector's host chain, ThinkingCutoff::replacedProposals),
// so acceptance keeps all its rows the end covers, and a row with no
// selectable logit keeps its sentinel in place of its end token, so the lane
// still fails. The end of a constrained lane that waits for its mask starts
// at the first row from its cutoff's on whose mask admits the end's first
// token, the rows before keeping their own selections, and proposes nothing.
void thinkingCutoffs(MetalBackend &backend, uint32_t vocabulary) {
  const auto stops = shardEdgeStopTokens(vocabulary);
  const uint32_t close = vocabulary - 3, alsoEnds = vocabulary - 5;
  // Ends of one token, of fewer than a verify's rows and of more than a
  // verify block's; their other tokens repeat five that no row prefers.
  std::vector<std::vector<uint32_t>> ends;
  for (const uint32_t length : {1U, 3U, SPLASH_TARGET_VERIFY_ROWS + 3}) {
    std::vector<uint32_t> &end = ends.emplace_back();
    for (uint32_t index = 0; index + 1 < length; ++index)
      end.push_back(vocabulary - 6 - index % 5);
    end.push_back(close);
  }
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, kLanes);
  // Each row's best token, far above the rest, unless the row prefers top.
  const auto best = [&](uint32_t row) { return 5 + row * 7919 % (vocabulary - 16); };
  const auto fill = [&](uint32_t row, uint32_t top) {
    float *values = batch.row(row);
    for (uint32_t token = 0; token < vocabulary; ++token)
      values[token] = -4.0F + float(token % 7) * 0.01F;
    values[top] = 8.0F;
  };
  const uint32_t *tokens = batch.outputTokens();
  float *uniforms = batch.uniforms();
  const auto dispatches = [](const CommandGraph &graph, const char *pipeline) {
    return std::ranges::count(graph.dispatches(), std::string(pipeline),
                              &splash::metal::ComputeDispatch::pipelineName);
  };
  const auto cutoffDispatches = [&](const CommandGraph &graph) {
    return dispatches(graph, "decode_sample_thinking_cutoff");
  };

  for (const std::vector<uint32_t> &end : ends) {
    for (const float temperature : {0.0F, 0.8F}) {
      for (const uint32_t row : {0U, 1U}) {
        fill(kRows - 1, best(kRows - 1));
        batch.poison();
        uniforms[0] = 0.5F;
        SamplingPolicy policy{32, temperature};
        policy.thinkingCutoff = ThinkingCutoff{row, end, alsoEnds};
        CommandGraph initial;
        sampling.addInitial(initial, {&policy, 1}, batch.buffers, initialRows(1, kRows - 1), stops, {});
        require(cutoffDispatches(initial) == (row == 0 ? 1 : 0), "the first token's cutoff dispatch is wrong");
        static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
        require(tokens[kRows - 1] == (row == 0 ? end[0] : best(kRows - 1)), "the first token missed its cutoff");
      }
    }
  }
  // A constrained lane whose end waits for its mask selects the end's first
  // token as its first token only where its first mask row admits it, and
  // its own selection elsewhere.
  uint32_t *masks = batch.masks();
  const uint32_t words = batch.maskWords();
  // Lane l's mask row `row` refuses `token`.
  const auto refuse = [&](uint32_t lane, uint32_t row, uint32_t token) {
    masks[(uint64_t{lane} * kMaskRows + row) * words + token / 32] &= ~(1U << token % 32);
  };
  for (const std::vector<uint32_t> &end : ends) {
    for (const float temperature : {0.0F, 0.8F}) {
      for (const bool admitted : {false, true}) {
        fill(kRows - 1, best(kRows - 1));
        std::fill_n(masks, words, ~0U);
        if (!admitted)
          refuse(0, 0, end[0]);
        batch.poison();
        uniforms[0] = 0.5F;
        SamplingPolicy policy{32, temperature, 1.0F, true};
        policy.thinkingCutoff = ThinkingCutoff{0, end, alsoEnds, true};
        CommandGraph initial;
        sampling.addInitial(initial, {&policy, 1}, batch.buffers, initialRows(1, kRows - 1), stops, {});
        static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
        require(tokens[kRows - 1] == (admitted ? end[0] : best(kRows - 1)),
                "a first token's end did not wait for its mask");
      }
    }
  }

  // The first tokens left their rows their trees' roots; the verifies below
  // take each lane's block as a chain, as its selector lays it out.
  writeChains(batch.buffers.treeRows, blocks(kLanes));
  const AcceptanceBuffers acceptance = acceptanceBuffers(backend, batch, kLanes);
  // A verify's chain drafts a token at each row after its anchor: at the
  // draft's positions in a lane of eight rows, and past them in a lane of
  // sixteen as the selector's chains do.
  constexpr uint32_t kDrafts = kRows - 1;
  // A draft token, at verify input row position + 1.
  const auto draft = [&](uint32_t lane, uint32_t position, uint32_t token) {
    batch.inputTokens()[lane * kRows + position + 1] = token;
  };
  // A lane whose end starts at its row 0 proposes it: its first
  // replacedProposals() rows after the anchor hold the end's tokens.
  const auto proposeEnds = [&](std::span<const SamplingPolicy> policies) {
    for (uint32_t lane = 0; lane < policies.size(); ++lane)
      if (const auto &cutoff = policies[lane].thinkingCutoff)
        for (uint32_t row = 1; row < kRows && row <= cutoff->replacedProposals(); ++row)
          batch.inputTokens()[lane * kRows + row] = cutoff->end[row - 1];
  };
  const std::array<uint32_t, kLanes> maximumRetained{kRows, kRows, kRows, kRows};
  // Acceptance keeps the drafts up to the first one that is not its row's
  // token, and that row's token.
  const auto retainedRows = [](std::span<const uint32_t> drafts, std::span<const uint32_t> rows) {
    const auto mismatch = std::ranges::mismatch(drafts, rows.first(kDrafts));
    return mismatch.in1 == drafts.end() ? kRows : uint32_t(mismatch.in1 - drafts.begin()) + 1;
  };
  const auto verifyLanes = [&](CommandGraph &verify, std::span<const SamplingPolicy> policies,
                               const std::array<std::array<uint32_t, kRows>, kLanes> &expected,
                               const std::array<std::array<uint32_t, kDrafts>, kLanes> &drafts) {
    std::fill(uniforms, uniforms + kLanes * kUniforms, 0.5F);
    sampling.addVerify(verify, policies, batch.buffers, layOut(batch.buffers, blocks(policies.size())), stops, {});
    require(cutoffDispatches(verify) == 1, "a verify with cutoffs dispatched them other than once");
    sampling.addAcceptance(verify, acceptance, blockStep(policies.size()), maximumRetained, stops);
    static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
    const auto *retained = static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      require(retained[lane] == retainedRows(drafts[lane], expected[lane]),
              "a lane retained the wrong tokens at its cutoff");
      for (uint32_t row = 0; row < retained[lane]; ++row)
        require(tokens[lane * kRows + row] == expected[lane][row], "a lane's tokens missed its thinking cutoff");
    }
  };

  enum class Draft { Best, EndsBefore, AlsoEndsBefore, DraftsFirst, DraftsAll };
  for (const std::vector<uint32_t> &end : ends) {
    for (const uint32_t samplingMask : {0U, 0b0101U, 0b1111U}) {
      for (const Draft kind :
           {Draft::Best, Draft::EndsBefore, Draft::AlsoEndsBefore, Draft::DraftsFirst, Draft::DraftsAll}) {
        // Lane l's cutoff is at row cuts[l]; kRows cuts nothing.
        for (const std::array<uint32_t, kLanes> cuts :
             {std::array<uint32_t, kLanes>{0, 3, 6, 7}, std::array<uint32_t, kLanes>{kRows, 2, 5, 1}}) {
          std::vector<SamplingPolicy> policies;
          // The token each lane's row selects, and the lane's draft tokens.
          std::array<std::array<uint32_t, kRows>, kLanes> expected{};
          std::array<std::array<uint32_t, kDrafts>, kLanes> drafts{};
          for (uint32_t lane = 0; lane < kLanes; ++lane) {
            SamplingPolicy policy{32, (samplingMask >> lane & 1U) ? 0.8F : 0.0F};
            const uint32_t cut = cuts[lane];
            policy.thinkingCutoff = ThinkingCutoff{cut, end, alsoEnds};
            policies.push_back(policy);
            for (uint32_t row = 0; row < kRows; ++row) {
              fill(lane * kRows + row, best(lane * kRows + row));
              expected[lane][row] = best(lane * kRows + row);
              if (row < kDrafts)
                drafts[lane][row] = best(lane * kRows + row);
            }
            // Where thinking ends at a draft token the target agrees with
            // before the cutoff, the rows from the cutoff's on keep their own
            // selections.
            if ((kind == Draft::EndsBefore || kind == Draft::AlsoEndsBefore) && cut >= 2 && cut < kRows) {
              const uint32_t at = cut - 2, ending = kind == Draft::EndsBefore ? close : alsoEnds;
              fill(lane * kRows + at, ending);
              expected[lane][at] = drafts[lane][at] = ending;
            } else {
              for (uint32_t index = 0; index < end.size() && cut + index < kRows; ++index) {
                expected[lane][cut + index] = end[index];
                const bool drafted = kind == Draft::DraftsAll || (kind == Draft::DraftsFirst && index == 0);
                if (drafted && cut + index < kDrafts)
                  drafts[lane][cut + index] = end[index];
              }
            }
            for (uint32_t position = 0; position < kDrafts; ++position)
              draft(lane, position, drafts[lane][position]);
          }
          batch.poison();
          CommandGraph verify;
          verifyLanes(verify, policies, expected, drafts);
        }
      }
    }
  }

  // A constrained lane's end that waits for its mask starts at the first row
  // from its cutoff's on whose mask admits the end's first token, as where a
  // grammar admits the end only between whole characters: lane l's cutoff is
  // at row cuts[l], and the masks of that row and the next waits[l] - 1
  // refuse that token. Those rows keep their own selections, as every row
  // does where no row of the selection admits it, and a draft token among
  // them that ends thinking, which their masks admit, leaves every row its
  // own. Such an end replaces no proposals: only the verify reads its masks.
  enum class Waiting { Best, DraftsFirst, EndsWhileWaiting };
  using PerLane = std::array<uint32_t, kLanes>;
  for (const std::vector<uint32_t> &end : ends) {
    for (const uint32_t samplingMask : {0U, 0b0101U, 0b1111U}) {
      for (const Waiting kind : {Waiting::Best, Waiting::DraftsFirst, Waiting::EndsWhileWaiting}) {
        for (const auto &[cuts, waits] : {std::pair{PerLane{0, 3, 6, 2}, PerLane{1, 2, 1, 3}},
                                          std::pair{PerLane{0, 1, 5, 7}, PerLane{kRows, 0, 3, 1}}}) {
          std::vector<SamplingPolicy> policies;
          std::array<std::array<uint32_t, kRows>, kLanes> expected{};
          std::array<std::array<uint32_t, kDrafts>, kLanes> drafts{};
          std::fill_n(masks, uint64_t{kLanes} * kMaskRows * words, ~0U);
          for (uint32_t lane = 0; lane < kLanes; ++lane) {
            SamplingPolicy policy{32, (samplingMask >> lane & 1U) ? 0.8F : 0.0F, 1.0F, true};
            policy.thinkingCutoff = ThinkingCutoff{cuts[lane], end, alsoEnds, true};
            policies.push_back(policy);
            for (uint32_t row = 0; row < kRows; ++row) {
              fill(lane * kRows + row, best(lane * kRows + row));
              expected[lane][row] = best(lane * kRows + row);
              if (row < kDrafts)
                drafts[lane][row] = best(lane * kRows + row);
            }
            // Verify row r reads mask row r + 1.
            const uint32_t start = std::min(cuts[lane] + waits[lane], kRows);
            for (uint32_t row = cuts[lane]; row < start; ++row)
              refuse(lane, row + 1, end[0]);
            if (kind == Waiting::EndsWhileWaiting && end.size() > 1 && cuts[lane] < start && start <= kDrafts) {
              const uint32_t at = start - 1;
              fill(lane * kRows + at, close);
              expected[lane][at] = drafts[lane][at] = close;
            } else {
              for (uint32_t index = 0; index < end.size() && start + index < kRows; ++index)
                expected[lane][start + index] = end[index];
              if (kind == Waiting::DraftsFirst && start < kDrafts)
                drafts[lane][start] = end[0];
            }
            for (uint32_t position = 0; position < kDrafts; ++position)
              draft(lane, position, drafts[lane][position]);
          }
          for (const SamplingPolicy &policy : policies)
            require(!policy.thinkingCutoff->replacedProposals(), "an end that waits for its mask proposed tokens");
          batch.poison();
          CommandGraph verify;
          verifyLanes(verify, policies, expected, drafts);
        }
      }
    }
  }

  // At row 0, lane 0 has more end tokens left than verify rows, lane 1 three
  // and lane 3 the close alone; each one proposes its end in place of its
  // draft's best tokens. Lane 2's end starts at its row 2, which its draft
  // keeps.
  const std::vector<uint32_t> &end = ends.back();
  const std::span<const uint32_t> whole(end);
  const std::array<ThinkingCutoff, kLanes> started{
      ThinkingCutoff{0, whole, alsoEnds}, ThinkingCutoff{0, whole.last(3), alsoEnds},
      ThinkingCutoff{2, whole, alsoEnds}, ThinkingCutoff{0, whole.last(1), alsoEnds}};
  for (const uint32_t samplingMask : {0U, 0b0101U, 0b1111U}) {
    std::vector<SamplingPolicy> policies;
    std::array<std::array<uint32_t, kRows>, kLanes> expected{};
    std::array<std::array<uint32_t, kDrafts>, kLanes> drafts{};
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      SamplingPolicy policy{32, (samplingMask >> lane & 1U) ? 0.8F : 0.0F};
      policy.thinkingCutoff = started[lane];
      policies.push_back(policy);
      batch.inputTokens()[lane * kRows] = 3 + lane;
      for (uint32_t row = 0; row < kRows; ++row) {
        fill(lane * kRows + row, best(lane * kRows + row));
        expected[lane][row] = best(lane * kRows + row);
        if (row < kDrafts) {
          draft(lane, row, best(lane * kRows + row));
          drafts[lane][row] = best(lane * kRows + row);
        }
      }
      const ThinkingCutoff &cutoff = started[lane];
      for (uint32_t index = 0; index < cutoff.end.size() && cutoff.row + index < kRows; ++index) {
        expected[lane][cutoff.row + index] = cutoff.end[index];
        if (index < cutoff.replacedProposals())
          drafts[lane][index] = cutoff.end[index];
      }
    }
    proposeEnds(policies);
    batch.poison();
    CommandGraph verify;
    verifyLanes(verify, policies, expected, drafts);
  }

  // Of constrained lanes at row 0, only those whose ends do not wait propose
  // them, lane 1 three and lane 2 all a verify block's rows after its anchor
  // hold.
  const std::array<ThinkingCutoff, kLanes> drafting{
      ThinkingCutoff{0, whole, alsoEnds, true}, ThinkingCutoff{0, whole.last(3), alsoEnds},
      ThinkingCutoff{0, whole, alsoEnds}, ThinkingCutoff{0, whole.last(1), alsoEnds, true}};
  const std::array<uint32_t, kLanes> replaced{0, 3, SPLASH_TREE_CHAIN_TOKENS, 0};
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    require(drafting[lane].replacedProposals() == replaced[lane], "an end counted the wrong replaced proposals");

  // Lane l's row nonFinite[l] has no selectable logit: it keeps its sentinel,
  // which acceptance retains after the end tokens before it.
  constexpr uint32_t kSentinel = 0xFFFFFFFFU;
  const std::array<uint32_t, kLanes> nonFinite{0, 2, 5, kRows - 1};
  for (const uint32_t samplingMask : {0U, 0b0101U, 0b1111U}) {
    std::vector<SamplingPolicy> policies;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      SamplingPolicy policy{32, (samplingMask >> lane & 1U) ? 0.8F : 0.0F};
      policy.thinkingCutoff = ThinkingCutoff{0, whole, alsoEnds};
      policies.push_back(policy);
      for (uint32_t row = 0; row < kRows; ++row) {
        fill(lane * kRows + row, best(lane * kRows + row));
        if (row < kDrafts)
          draft(lane, row, best(lane * kRows + row));
      }
      float *values = batch.row(lane * kRows + nonFinite[lane]);
      std::fill(values, values + vocabulary, std::numeric_limits<float>::quiet_NaN());
    }
    std::fill(uniforms, uniforms + kLanes * kUniforms, 0.5F);
    proposeEnds(policies);
    batch.poison();
    CommandGraph verify;
    sampling.addVerify(verify, policies, batch.buffers, layOut(batch.buffers, blocks(policies.size())), stops, {});
    sampling.addAcceptance(verify, acceptance, blockStep(policies.size()), maximumRetained, stops);
    static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
    const auto *retained = static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      const uint32_t bad = nonFinite[lane];
      require(retained[lane] == bad + 1 && tokens[lane * kRows + bad] == kSentinel &&
                  std::ranges::equal(std::span(tokens + lane * kRows, bad), whole.first(bad)),
              "a row with no selectable logit took its end token");
    }
  }

  SamplingPolicy none{32, 0.0F};
  none.thinkingCutoff = ThinkingCutoff{kRows, end, alsoEnds};
  const std::array<SamplingPolicy, 2> uncut{none, SamplingPolicy{}};
  CommandGraph graph;
  sampling.addVerify(graph, uncut, batch.buffers, layOut(batch.buffers, blocks(uncut.size())), stops, {});
  require(cutoffDispatches(graph) == 0, "a verify without a cutoff in its rows dispatched one");

  // Only a constrained lane has masks for its end to wait for, and the
  // kernel reads them at the end's first token, which must be inside the
  // vocabulary.
  SamplingPolicy unconstrained{32, 0.0F};
  unconstrained.thinkingCutoff = ThinkingCutoff{0, end, alsoEnds, true};
  const std::array<uint32_t, 2> outside{vocabulary, close};
  SamplingPolicy outsideStart{32, 0.0F, 1.0F, true};
  outsideStart.thinkingCutoff = ThinkingCutoff{0, outside, alsoEnds, true};
  for (const auto &[policy, refusal] :
       {std::pair{unconstrained, "unconstrained lane's thinking end waits"},
        std::pair{outsideStart, "waits for a mask starts outside the vocabulary"}}) {
    rejects(
        [&] {
          CommandGraph refused;
          sampling.addVerify(refused, {&policy, 1}, batch.buffers, layOut(batch.buffers, blocks(1)), stops, {});
        },
        refusal, "a thinking end waited for a mask it cannot read");
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
  for (const Case &c :
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
      setDraft(batch, 0, position, draft);
    }
    for (uint32_t uniform = 0; uniform < kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    batch.poison();
    CommandGraph verify;
    sampling.addVerify(verify, {&c.policy, 1}, batch.buffers, layOut(batch.buffers, blocks(1)),
                       kStopTokens, {table, tableRows});
    static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
    for (uint32_t row = 0; row < kRows; ++row)
      requireSampledRow(batch, row, target,
                        std::string(c.name) + " row " + std::to_string(row));
  }
}

// A policy that keeps only its most likely token selects greedily.
void onlyTheMostLikelyIsGreedy() {
  require(SamplingPolicy{0, 0.8F}.samples() && !SamplingPolicy{}.samples() &&
              !SamplingPolicy{1, 0.8F}.samples() &&
              !SamplingPolicy{.temperature = 0.8F, .minP = 1.0F}.samples(),
          "a policy that keeps one token sampled");
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
             {0, 1.0F, 0.5F, false, false, {}, 0.05F}, 8, 106, 107}}) {
    const Distribution target =
        referenceDistribution(batch.row(0), vocabulary, c.policy);
    require(target.order.size() == c.kept && target.ambiguousMass == 0.0 &&
                target.probability(c.last) > 0.0 &&
                target.probability(c.dropped) == 0.0,
            std::string(c.name) + ": the reference keeps " +
                std::to_string(target.order.size()) + " tokens");
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t draft = position % 2 ? c.dropped : c.last;
      setDraft(batch, 0, position, draft);
    }
    for (uint32_t uniform = 0; uniform < kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    batch.poison();
    CommandGraph verify;
    sampling.addVerify(verify, {&c.policy, 1}, batch.buffers, layOut(batch.buffers, blocks(1)), kStopTokens, {});
    static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
    for (uint32_t row = 0; row < kRows; ++row)
      requireSampledRow(batch, row, target,
                        std::string(c.name) + " row " + std::to_string(row));
  }
}

// DFlash acceptance over broad distributions against a sequential decode,
// beside a lane with the default top-k of 20 and one that min_p cuts. The
// draft proposes each position's most likely token until a lane's wrong
// position, where it proposes a token the target keeps with little
// probability. Each row draws with its own uniform, which the test places on
// the right draft token's interval, so acceptance keeps the right tokens,
// then the wrong position's draw, which is not its draft, or the bonus
// token from the last row.
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
  const AcceptanceBuffers acceptance = acceptanceBuffers(backend, batch, lanes);
  std::array<std::array<uint32_t, kRows>, lanes> expectedOutput{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    float *uniforms = batch.uniforms() + lane * kUniforms;
    uint32_t *inputs = batch.inputTokens() + lane * kRows;
    inputs[0] = 9;
    for (uint32_t row = 0; row < kRows; ++row) {
      const bool right = row < kWrongAt[lane];
      // A right draft's row descends over 300 tokens, so its draft's
      // interval is wide enough to place a draw in; the others are flat or
      // graded.
      fillShaped(batch.row(lane * kRows + row), vocabulary,
                 !right && row % 2 ? Shape::Flat : Shape::Graded, random);
      const Distribution target = referenceDistribution(
          batch.row(lane * kRows + row), vocabulary, policies[lane]);
      // The best token, or from the wrong position on the sixth best.
      const uint32_t draft =
          right ? target.order.front() : target.order[std::min<size_t>(5, target.order.size() - 1)];
      require(!right || (target.probability(draft) > 0.0 && !target.ambiguous[draft]),
              "a right draft token its target does not keep");
      // The tokens a top_p cut may keep or cut shift the cumulative sum by at
      // most their mass: a draw within that of an interval's end, or of an
      // ambiguous token, is redrawn.
      const double total = std::accumulate(target.probabilities.begin(), target.probabilities.end(), 0.0);
      const double slack = (1e-5 + target.ambiguousMass) * total;
      uint32_t token = 0;
      for (uint32_t attempt = 0;; ++attempt) {
        require(attempt < 100, "test data is ambiguous at every draw");
        uniforms[row] = right ? intervalMiddle(target.probabilities, draft) : 0.5F * (random.unit() + 1.0F);
        const double threshold = uniforms[row] * total;
        double cumulative = 0.0;
        token = 0;
        while (cumulative + target.probabilities[token] <= threshold)
          cumulative += target.probabilities[token++];
        if (threshold - cumulative > slack && cumulative + target.probabilities[token] - threshold > slack &&
            !target.ambiguous[token])
          break;
        require(!right, "test data is ambiguous at a right draft's draw");
      }
      if (row <= kWrongAt[lane])
        expectedOutput[lane][row] = token;
      require(row != kWrongAt[lane] || row == kPositions || token != draft,
              "the wrong draft is the target's draw");
      if (row < kPositions)
        inputs[row + 1] = draft;
    }
  }
  batch.poison();
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch.buffers, layOut(batch.buffers, blocks(policies.size())), kStopTokens, {});
  const std::array<uint32_t, lanes> maximumRetained{kRows, kRows, kRows, kRows};
  sampling.addAcceptance(graph, acceptance, blockStep(policies.size()), maximumRetained, kStopTokens);
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
    setDraft(batch, 0, position, draft);
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
      sampling.addVerify(verify, {&policy, 1}, batch.buffers, layOut(batch.buffers, blocks(1)),
                         kStopTokens, {table, tableRows});
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
  for (const Case &c :
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

// A row with no selectable logit, every one NaN (which a non-finite hidden
// row leaves) or -inf, has no token to select: greedy and sampled policies
// select 0xFFFFFFFF for a first token and at every verify row, and
// acceptance retains it as the next anchor. A penalty on one of its tokens
// leaves that logit, and so the row, as it is (referencePenalty). The model
// runtime reports that sentinel as the request's own failure, which keeps
// the rest of the batch serving.
void nonFiniteRowsSelectTheSentinel(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  constexpr uint32_t kSentinel = 0xFFFFFFFFU;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, 1);
  const AcceptanceBuffers acceptance = acceptanceBuffers(backend, batch, 1);
  Random random(0x6e616e);
  for (uint32_t position = 0; position < kPositions; ++position)
    setDraft(batch, 0, position, 3 + random.next() % (vocabulary - 3));
  batch.inputTokens()[0] = 7;
  for (uint32_t uniform = 0; uniform < 2 * kRows; ++uniform)
    batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
  // Token 29 is in the prompt, so repetition rewrites its logit.
  const std::array<uint32_t, 1> penaltyRows{0};
  const PenaltyTable penalties{
      allocate(backend, uint64_t{vocabulary} * sizeof(uint32_t)), penaltyRows};
  std::memset(penalties.words.contents(), 0, penalties.words.sizeBytes());
  static_cast<uint32_t *>(penalties.words.contents())[29] = SPLASH_PENALTY_PROMPT_BIT;
  for (SamplingPolicy policy : {SamplingPolicy{1, 0.0F, 1.0F, false},
                                SamplingPolicy{20, 1.0F, 0.9F, false}}) {
    for (const float logit : {std::numeric_limits<float>::quiet_NaN(),
                              -std::numeric_limits<float>::infinity()}) {
      for (const float repetition : {1.0F, 1.1F}) {
        policy.penalties.repetition = repetition;
        const std::string label =
            std::string(policy.samples() ? "sampled" : "greedy") +
            (std::isnan(logit) ? ", NaN" : ", -inf") +
            (policy.penalties.active() ? ", penalized" : "");
        std::fill(batch.logits(), batch.logits() + batch.rows * vocabulary, logit);
        batch.poison();
        CommandGraph initial;
        sampling.addInitial(initial, {&policy, 1}, batch.buffers, initialRows(1, 0),
                            kStopTokens, penalties);
        static_cast<void>(backend.submitCommandAsync(initial.dispatches()).wait());
        require(batch.outputTokens()[0] == kSentinel,
                label + ": a first token from a row with no selectable logit is " +
                    std::to_string(batch.outputTokens()[0]));
        batch.poison();
        CommandGraph verify;
        sampling.addVerify(verify, {&policy, 1}, batch.buffers, layOut(batch.buffers, blocks(1)),
                           kStopTokens, penalties);
        static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
        for (uint32_t row = 0; row < kRows; ++row) {
          require(batch.outputTokens()[row] == kSentinel,
                  label + ": verify row " + std::to_string(row) +
                      " with no selectable logit selected " +
                      std::to_string(batch.outputTokens()[row]));
          require(penaltyMatches(batch.row(row)[29], logit, 0, true,
                                 policy.penalties),
                  label + ": a penalty rewrote row " + std::to_string(row) +
                      "'s unselectable logit");
        }
        CommandGraph accept;
        const std::array<uint32_t, 1> maximumRetained{kRows};
        sampling.addAcceptance(accept, acceptance, blockStep(1), maximumRetained, kStopTokens);
        static_cast<void>(backend.submitCommandAsync(accept.dispatches()).wait());
        const uint32_t retained =
            *static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
        require(retained >= 1 && retained <= kRows &&
                    batch.outputTokens()[retained - 1] == kSentinel,
                label + ": acceptance did not retain the sentinel as its anchor");
      }
    }
  }
}

// A verify block as a tree: each row's penalties count the draft tokens of its
// path, its ancestors' but its anchor's, and not those of the rows before it on
// other branches; a thinking cutoff gives each row of its depth on its end
// tokens by depth, but where a token on the row's path up to the cut ended
// thinking, which leaves that row's subtree its own selections, and an end
// that waits for its mask starts on each path at the path's own first row
// whose mask admits it. Lane 0's tree:
// rows 1 and 2 follow the anchor, rows 3 and 4 row 1, row 5 row 2, rows 6
// and 7 row 5; its rows repeat tokens across branches, and row 2 holds the
// close of thinking. Lane 1 is the same tree, sampled.
void treeSelections(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  const uint32_t close = vocabulary - 3, alsoEnds = vocabulary - 5;
  const std::array<uint32_t, 2> end{vocabulary - 6, close};
  constexpr std::array<uint32_t, kRows> parents{0, 0, 0, 1, 1, 2, 5, 5};
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, 2);
  auto *tree = static_cast<TreeRow *>(batch.buffers.treeRows.contents());
  Random random(0x70617468);
  std::array<uint32_t, kRows> depth{}, ancestors{};
  for (uint32_t lane = 0; lane < 2; ++lane)
    for (uint32_t row = 0; row < kRows; ++row) {
      depth[row] = row ? depth[parents[row]] + 1 : 0;
      ancestors[row] = (row ? ancestors[parents[row]] : 0U) | (1U << row);
      tree[lane * kRows + row] = {ancestors[row], static_cast<uint16_t>(row ? parents[row] : SPLASH_TREE_ROOT),
                                  static_cast<uint16_t>(depth[row])};
    }
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const std::array<uint32_t, 2> tableRows{1, 2};
  const SamplingPenalties penalties{1.3F, 0.7F, 0.4F};
  std::vector<SamplingPolicy> policies{SamplingPolicy{1, 0.0F, 1.0F, false, false, penalties},
                                       SamplingPolicy{32, 0.8F, 0.95F, false, false, penalties}};
  for (uint32_t row = 0; row < batch.rows; ++row)
    fillRow(batch.row(row), vocabulary, random);
  // Rows 3 and 5 hold the token of row 1, on two branches; row 6 that of
  // row 5 below it.
  for (uint32_t lane = 0; lane < 2; ++lane) {
    uint32_t *inputs = batch.inputTokens() + lane * kRows;
    const std::vector<uint32_t> best = referenceBest(batch.row(lane * kRows), vocabulary, 8);
    const std::array<uint32_t, kRows> tokens{7, best[0], close, best[1], best[2], best[0], best[0], best[3]};
    std::copy(tokens.begin(), tokens.end(), inputs);
  }
  for (uint32_t uniform = 0; uniform < 2 * kUniforms; ++uniform)
    batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
  const std::vector<float> original(batch.logits(), batch.logits() + uint64_t{batch.rows} * vocabulary);
  batch.poison();
  CommandGraph verify;
  sampling.addVerify(verify, policies, batch.buffers, layOut(batch.buffers, blocks(2)),
                     kStopTokens, {table, tableRows});
  static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
  for (uint32_t lane = 0; lane < 2; ++lane) {
    const uint32_t *inputs = batch.inputTokens() + lane * kRows;
    const std::span<uint32_t> words = tableRow(table, vocabulary, tableRows[lane]);
    for (uint32_t row = 0; row < kRows; ++row)
      for (uint32_t token = 0; token < vocabulary; ++token) {
        uint32_t count = words[token] & SPLASH_PENALTY_COUNT_MASK;
        for (uint32_t path = ancestors[row] & ~1U; path; path &= path - 1)
          count += inputs[std::countr_zero(path)] == token ? 1U : 0U;
        const uint64_t at = (uint64_t{lane} * kRows + row) * vocabulary + token;
        require(penaltyMatches(batch.logits()[at], original[at], count, (words[token] & SPLASH_PENALTY_PROMPT_BIT) != 0,
                               penalties),
                "tree row " + std::to_string(row) + " counted other draft tokens than its path's");
      }
  }

  // The cut at depth 2: rows 3, 4 (below row 1) and 5 (below row 2, whose
  // token closes thinking) take end[0]; rows 6 and 7 at depth 3 end[1] but for
  // the closed branch, whose rows 5-7 keep their own selections.
  for (SamplingPolicy &policy : policies)
    policy.thinkingCutoff = ThinkingCutoff{2, end, alsoEnds};
  std::copy(original.begin(), original.end(), batch.logits());
  batch.poison();
  CommandGraph cut;
  sampling.addVerify(cut, policies, batch.buffers, layOut(batch.buffers, blocks(2)), kStopTokens, {table, tableRows});
  static_cast<void>(backend.submitCommandAsync(cut.dispatches()).wait());
  const std::vector<uint32_t> selected(batch.outputTokens(), batch.outputTokens() + batch.rows);
  for (uint32_t lane = 0; lane < 2; ++lane)
    for (const uint32_t row : {3U, 4U})
      require(selected[lane * kRows + row] == end[0], "a row of the cut's depth did not take the end");
  std::copy(original.begin(), original.end(), batch.logits());
  batch.poison();
  CommandGraph uncut;
  for (SamplingPolicy &policy : policies)
    policy.thinkingCutoff.reset();
  sampling.addVerify(uncut, policies, batch.buffers, layOut(batch.buffers, blocks(2)), kStopTokens, {table, tableRows});
  static_cast<void>(backend.submitCommandAsync(uncut.dispatches()).wait());
  for (uint32_t lane = 0; lane < 2; ++lane)
    for (const uint32_t row : {0U, 1U, 2U, 5U, 6U, 7U})
      require(selected[lane * kRows + row] == batch.outputTokens()[lane * kRows + row],
              "a row above the cut's depth or below a closed path took the end");

  // Where a constrained lane's end waits for its mask, each path cuts at its
  // first row from the cut's depth whose mask admits the end's first token:
  // with the cut at depth 1 and row 1's mask refusing that token, the path
  // through row 1 cuts at rows 3 and 4, and the one through row 2, which here
  // holds no close, at row 2 itself, whose child row 5 takes end[1] and whose
  // grandchildren rows 6 and 7, past the end, their own selections.
  uint32_t *masks = batch.masks();
  const uint32_t words = batch.maskWords();
  std::fill_n(masks, uint64_t{2} * kMaskRows * words, ~0U);
  for (uint32_t lane = 0; lane < 2; ++lane) {
    // Verify row r reads mask row r + 1.
    masks[(uint64_t{lane} * kMaskRows + 2) * words + end[0] / 32] &= ~(1U << end[0] % 32);
    batch.inputTokens()[lane * kRows + 2] = 11;
  }
  const auto select = [&] {
    std::copy(original.begin(), original.end(), batch.logits());
    batch.poison();
    CommandGraph graph;
    sampling.addVerify(graph, policies, batch.buffers, layOut(batch.buffers, blocks(2)), kStopTokens,
                       {table, tableRows});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    return std::vector<uint32_t>(batch.outputTokens(), batch.outputTokens() + batch.rows);
  };
  for (SamplingPolicy &policy : policies)
    policy.constrained = true;
  const std::vector<uint32_t> own = select();
  for (SamplingPolicy &policy : policies)
    policy.thinkingCutoff = ThinkingCutoff{1, end, alsoEnds, true};
  const std::vector<uint32_t> waited = select();
  // The end token each of the eight rows takes; -1 for its own selection.
  constexpr std::array<int, 8> takes{-1, -1, 0, 0, 0, 1, -1, -1};
  for (uint32_t lane = 0; lane < 2; ++lane)
    for (uint32_t row = 0; row < takes.size(); ++row)
      require(waited[lane * kRows + row] == (takes[row] < 0 ? own[lane * kRows + row] : end[takes[row]]),
              "tree row " + std::to_string(row) + "'s waiting end did not start on its own path");
}

// A sampled verify row's draft children (metal/abi/Sampling.h): its next
// position's candidates, the draft's probabilities of them, and the
// candidate indices it drew, in order.
struct Children final {
  std::array<uint32_t, kCandidates> ids{};
  std::array<double, kCandidates> q{};
  std::vector<uint32_t> draws;
};

// The recursion after the children before: the target's probabilities of
// the candidates and its mass outside them, and the draft's probabilities.
struct Residual final {
  std::array<double, kCandidates> p{};
  double outside = 0.0;
  std::array<double, kCandidates> q{};
};

Residual startOf(const std::vector<double> &p, const Children &children) {
  Residual r;
  double inside = 0.0;
  for (uint32_t i = 0; i < kCandidates; ++i) {
    r.p[i] = children.ids[i] < p.size() ? p[children.ids[i]] : 0.0;
    inside += r.p[i];
  }
  r.outside = std::max(1.0 - inside, 0.0);
  r.q = children.q;
  return r;
}

// After child `index` is rejected: the target's residual max(p - q, 0),
// renormalized, and the draft without the child, renormalized by the sum of
// what is left of it. False where nothing is
// left of the residual, which an exact recursion never reaches.
bool reject(Residual &r, uint32_t index) {
  double residual = r.outside;
  for (uint32_t i = 0; i < kCandidates; ++i)
    residual += std::max(r.p[i] - r.q[i], 0.0);
  if (!(residual > 0.0))
    return false;
  for (uint32_t i = 0; i < kCandidates; ++i)
    r.p[i] = std::max(r.p[i] - r.q[i], 0.0) / residual;
  r.outside /= residual;
  r.q[index] = 0.0;
  const double rest = std::accumulate(r.q.begin(), r.q.end(), 0.0);
  for (uint32_t i = 0; rest > 0.0 && i < kCandidates; ++i)
    r.q[i] /= rest;
  return true;
}

// The distribution a row with `children` draws selects from, by enumeration
// of the draft's draws without replacement and of each test's outcome, the
// last residual where every test rejects: the candidates' residual, and
// past them the target's own probabilities outside them.
std::vector<double> selectionDistribution(const std::vector<double> &p, const Children &children,
                                          uint32_t draws) {
  std::vector<double> out(p.size(), 0.0);
  double outsideMass = 0.0;
  std::vector<bool> candidate(p.size(), false);
  for (const uint32_t id : children.ids)
    if (id < p.size())
      candidate[id] = true;
  for (uint32_t token = 0; token < p.size(); ++token)
    outsideMass += candidate[token] ? 0.0 : p[token];
  std::function<void(const Residual &, uint32_t, double)> step = [&](const Residual &r, uint32_t left,
                                                                     double weight) {
    double draft = 0.0;
    for (const double value : r.q)
      draft += value;
    if (!left || !(draft > 0.0)) {
      for (uint32_t i = 0; i < kCandidates; ++i)
        if (children.ids[i] < p.size())
          out[children.ids[i]] += weight * r.p[i];
      for (uint32_t token = 0; token < p.size() && outsideMass > 0.0; ++token)
        if (!candidate[token])
          out[token] += weight * r.outside * p[token] / outsideMass;
      return;
    }
    for (uint32_t i = 0; i < kCandidates; ++i) {
      if (!(r.q[i] > 0.0))
        continue;
      const double accept = std::min(1.0, r.p[i] / r.q[i]);
      out[children.ids[i]] += weight * r.q[i] * accept;
      Residual next = r;
      if (accept < 1.0 && reject(next, i))
        step(next, left - 1, weight * r.q[i] * (1.0 - accept));
    }
  };
  step(startOf(p, children), draws, 1.0);
  return out;
}

// The recursion selects from the target's distribution whatever the draft:
// over 300 random targets and drafts, of one to four children, some
// candidates outside the target's support and some of its mass outside the
// candidates, the enumerated selection is the target's to 1e-12.
void recursiveRejectionSampling() {
  Random random(0x72727321);
  const auto dirichlet = [&](std::span<double> out, uint32_t zeros) {
    double total = 0.0;
    for (double &value : out) {
      value = -std::log(0.5 * (random.unit() + 1.0) + 1e-9);
      total += value;
    }
    for (uint32_t i = 0; i < zeros; ++i)
      out[random.next() % out.size()] = 0.0;
    total = std::accumulate(out.begin(), out.end(), 0.0);
    for (double &value : out)
      value /= total;
  };
  double worst = 0.0;
  for (uint32_t trial = 0; trial < 300; ++trial) {
    std::vector<double> p(24);
    dirichlet(p, trial % 5);
    Children children;
    std::vector<uint32_t> tokens(p.size());
    std::iota(tokens.begin(), tokens.end(), 0U);
    std::shuffle(tokens.begin(), tokens.end(), std::mt19937(trial));
    std::copy_n(tokens.begin(), kCandidates, children.ids.begin());
    dirichlet(children.q, trial % 7);
    const std::vector<double> selected = selectionDistribution(p, children, 1 + trial % 4);
    for (uint32_t token = 0; token < p.size(); ++token)
      worst = std::max(worst, std::fabs(selected[token] - p[token]));
  }
  require(worst < 1e-12, "the recursion's selections are off the target's by " + std::to_string(worst));
}

// The selection of a row that tests `children` with uniforms `tests`, then
// draws the last residual with `uniform`: a candidate by the residual's
// weights in index order, or past them a token outside them by the target's
// probabilities in token order; a row without children draws from its whole
// distribution in token order. Close where a comparison lay within 1e-5 of
// its threshold, which float sums may decide either way.
struct Selection final {
  uint32_t token = 0;
  bool close = false;
};

Selection referenceSelection(const std::vector<double> &p, const Children &children, std::span<const float> tests,
                             double uniform) {
  Selection result;
  // A row without children draws from its whole distribution in token order.
  if (children.draws.empty()) {
    double cumulative = 0.0;
    for (uint32_t token = 0; token < p.size(); ++token) {
      if (!(p[token] > 0.0))
        continue;
      result.close = result.close || std::fabs(uniform - cumulative) < 1e-5 ||
                     std::fabs(uniform - cumulative - p[token]) < 1e-5;
      cumulative += p[token];
      result.token = token;
      if (uniform < cumulative)
        break;
    }
    return result;
  }
  Residual r = startOf(p, children);
  for (uint32_t child = 0; child < children.draws.size(); ++child) {
    const uint32_t index = children.draws[child];
    const double left = tests[child] * r.q[index];
    result.close = result.close || std::fabs(left - r.p[index]) < 1e-5;
    if (left < r.p[index]) {
      result.token = children.ids[index];
      return result;
    }
    if (!reject(r, index))
      break;
  }
  const double inside = std::accumulate(r.p.begin(), r.p.end(), 0.0);
  const double point = uniform * (inside + r.outside);
  double cumulative = 0.0;
  for (uint32_t i = 0; i < kCandidates; ++i) {
    if (!(r.p[i] > 0.0))
      continue;
    result.close = result.close || std::fabs(point - cumulative) < 1e-5 ||
                   std::fabs(point - cumulative - r.p[i]) < 1e-5;
    cumulative += r.p[i];
    if (point < cumulative) {
      result.token = children.ids[i];
      return result;
    }
  }
  std::vector<double> outside = p;
  for (const uint32_t id : children.ids)
    if (id < outside.size())
      outside[id] = 0.0;
  const double total = std::accumulate(outside.begin(), outside.end(), 0.0);
  const double target = (point - inside) / r.outside * total;
  cumulative = 0.0;
  for (uint32_t token = 0; token < outside.size(); ++token) {
    if (!(outside[token] > 0.0))
      continue;
    result.close = result.close || std::fabs(target - cumulative) < 1e-5 * total ||
                   std::fabs(target - cumulative - outside[token]) < 1e-5 * total;
    cumulative += outside[token];
    if (target < cumulative || token + 1 == outside.size()) {
      result.token = token;
      return result;
    }
  }
  return result;
}

// The draft's draws without replacement from q, Gumbel-top-k of its log q.
std::vector<uint32_t> drawChildren(const std::array<double, kCandidates> &q, uint32_t count, std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<std::pair<double, uint32_t>> keys;
  for (uint32_t i = 0; i < kCandidates; ++i)
    if (q[i] > 0.0)
      keys.emplace_back(std::log(q[i]) - std::log(-std::log(unit(rng) + 1e-300)), i);
  std::sort(keys.begin(), keys.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
  std::vector<uint32_t> draws;
  for (uint32_t i = 0; i < count && i < keys.size(); ++i)
    draws.push_back(keys[i].second);
  return draws;
}

// Writes a lane's tree row `row`'s children for its selection: its source
// (the row itself), its draws and its next position's probabilities, and
// the lane's candidates at the position after the row's depth.
void setChildren(const Batch &batch, uint32_t lane, uint32_t row, uint32_t depth, const Children &children) {
  static_cast<uint32_t *>(batch.buffers.verifySources.contents())[lane * kRows + row] = row;
  std::copy(children.ids.begin(), children.ids.end(),
            static_cast<uint32_t *>(batch.buffers.candidates.contents()) + (lane * kPositions + depth) * kCandidates);
  auto *draws = static_cast<uint8_t *>(batch.buffers.treeDraws.contents()) + (lane * kBlock + row) * kChildren;
  std::fill_n(draws, kChildren, uint8_t{0xff});
  std::copy(children.draws.begin(), children.draws.end(), draws);
  auto *next = static_cast<float *>(batch.buffers.treeNext.contents()) + (lane * kBlock + row) * kCandidates;
  for (uint32_t i = 0; i < kCandidates; ++i)
    next[i] = static_cast<float>(children.q[i]);
}

// A sampled verify row that tests the children its draft drew selects as the
// recursion does with its uniforms: for each lane's chain of rows, with
// production and other policies, of zero to four children each, their
// candidates a row's likeliest tokens among others, one of a lane's first row
// each round past the vocabulary (0xffffffff, as the draft's top 16 lists for
// a row of fewer comparable logits, never drawn), row by row against the
// reference, but where a comparison lies within float rounding of its
// threshold or the policy's cut is ambiguous.
void childTests(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, kLanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const std::array<uint32_t, kLanes> tableRows{};
  const std::vector<SamplingPolicy> policies{SamplingPolicy{20, 1.0F, 0.95F}, SamplingPolicy{20, 0.7F, 0.8F},
                                             SamplingPolicy{0, 1.0F, 1.0F}, SamplingPolicy{40, 1.3F, 0.9F}};
  Random random(0x63686c64);
  std::mt19937_64 rng(0x63686c64);
  // A lane of more than eight rows branches at its anchor: its rows from the
  // ninth hang off row 0 beside row 1, then as a chain, so that rows past the
  // eighth at depths the draft proposes draw children too (each with its
  // depth's candidates, which a lane holds once a depth).
  std::array<uint32_t, kRows> depths{};
  for (uint32_t row = 0; row < kRows; ++row) depths[row] = row < 8 ? row : row - 7;
  for (uint32_t lane = 0; lane < kLanes && kRows > 8; ++lane) {
    auto *tree = static_cast<TreeRow *>(batch.buffers.treeRows.contents()) + lane * kRows;
    for (uint32_t row = 8; row < kRows; ++row) {
      const uint32_t parent = row == 8 ? 0 : row - 1;
      tree[row] = {tree[parent].ancestors | 1U << row, static_cast<uint16_t>(parent),
                   static_cast<uint16_t>(depths[row])};
    }
  }
  const auto drawsChildren = [&](uint32_t row) { return row + 1 < kRows && depths[row] < kPositions; };
  uint32_t checked = 0, drawn = 0;
  for (uint32_t round = 0; round < 24; ++round) {
    for (uint32_t row = 0; row < batch.rows; ++row)
      fillRow(batch.row(row), vocabulary, random);
    for (uint32_t uniform = 0; uniform < kLanes * kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    std::vector<std::vector<Children>> rows(kLanes, std::vector<Children>(kRows));
    for (uint32_t lane = 0; lane < kLanes; ++lane)
      for (uint32_t row = 0; row < kRows; ++row) {
        if (!drawsChildren(row))
          continue;
        Children &children = rows[lane][row];
        if (depths[row] != row) {
          // Its depth's row's logits too, whose likeliest tokens are among
          // the candidates, so that its tests' outcomes follow their
          // uniforms (of other tokens' it would reject every child).
          std::copy_n(batch.row(lane * kRows + depths[row]), vocabulary, batch.row(lane * kRows + row));
          children.ids = rows[lane][depths[row]].ids;
        } else {
          // The row's likeliest tokens and others, all distinct.
          std::vector<uint32_t> ids = referenceBest(batch.row(lane * kRows + row), vocabulary, 6);
          while (ids.size() < kCandidates) {
            const uint32_t id = random.next() % vocabulary;
            if (std::find(ids.begin(), ids.end(), id) == ids.end())
              ids.push_back(id);
          }
          std::shuffle(ids.begin(), ids.end(), rng);
          std::copy(ids.begin(), ids.end(), children.ids.begin());
        }
        const bool outside = row == 0 && lane == round % kLanes;
        if (outside)
          children.ids[kCandidates - 1] = 0xffffffffU;
        double total = 0.0;
        for (uint32_t i = 0; i < kCandidates; ++i) {
          double &value = children.q[i];
          value = random.next() % 4 && !(outside && i == kCandidates - 1) ? std::pow(0.5 * (random.unit() + 1.0), 3.0)
                                                                          : 0.0;
          total += value;
        }
        for (double &value : children.q)
          value /= total;
        children.draws = drawChildren(children.q, (round + row + lane) % (kChildren + 1), rng);
        setChildren(batch, lane, row, depths[row], children);
      }
    batch.poison();
    CommandGraph graph;
    sampling.addVerify(graph, policies, batch.buffers, layOut(batch.buffers, blocks(kLanes)), kStopTokens,
                       {table, tableRows});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    for (uint32_t lane = 0; lane < kLanes; ++lane)
      for (uint32_t row = 0; row < kRows; ++row) {
        if (!drawsChildren(row))
          continue;
        const Distribution target = referenceDistribution(batch.row(lane * kRows + row), vocabulary, policies[lane]);
        ++drawn;
        if (target.ambiguousMass > 0.0)
          continue;
        const float *tests = batch.uniforms() + lane * kUniforms + kBlock + row * kChildren;
        const Selection reference =
            referenceSelection(target.probabilities, rows[lane][row], std::span(tests, kChildren),
                               batch.uniforms()[lane * kUniforms + row]);
        if (reference.close)
          continue;
        ++checked;
        require(batch.outputTokens()[lane * kRows + row] == reference.token,
                "round " + std::to_string(round) + " lane " + std::to_string(lane) + " row " + std::to_string(row) +
                    ": selected " + std::to_string(batch.outputTokens()[lane * kRows + row]) + ", not " +
                    std::to_string(reference.token));
      }
  }
  require(checked * 10 >= drawn * 9, "most rows were too close to check");
}

// A sampled verify row whose draft holds all but 1.5e-8 of its q_T on its
// first child, which fp32 holds as 1, tests its second child, once it rejects
// the first, against the rest of the draft renormalized, a fifteenth of which
// the second holds. The target holds 0.5 on the first child, 0.01 on the
// second and 0.49 outside the candidates. Over random uniforms, row by row
// against the recursion in double, but where a comparison lies within float
// rounding of its threshold.
void dominantChild(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, kLanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const std::array<uint32_t, kLanes> tableRows{};
  const SamplingPolicy policy{0, 1.0F, 1.0F};
  const std::vector<SamplingPolicy> policies(kLanes, policy);
  std::vector<float> logits(vocabulary, -1.0e4F);
  logits[500] = std::log(0.5F);
  logits[503] = std::log(0.01F);
  for (uint32_t i = 0; i < 49; ++i)
    logits[600 + i] = std::log(0.01F);
  const Distribution target = referenceDistribution(logits.data(), vocabulary, policy);
  Children children;
  for (uint32_t i = 0; i < kCandidates; ++i) {
    children.ids[i] = i == 0 ? 500U : i == 1 ? 503U : 100U + 3U * i;
    children.q[i] = i == 0 ? 1.0 : 1.0e-9;
  }
  children.draws = {0, 1};
  Random random(0x646f6d69);
  uint32_t checked = 0, seconds = 0;
  for (uint32_t round = 0; round < 8; ++round) {
    for (uint32_t uniform = 0; uniform < kLanes * kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
      for (uint32_t row = 0; row < kPositions; ++row) {
        std::copy(logits.begin(), logits.end(), batch.row(lane * kRows + row));
        setChildren(batch, lane, row, row, children);
      }
    batch.poison();
    CommandGraph graph;
    sampling.addVerify(graph, policies, batch.buffers, layOut(batch.buffers, blocks(kLanes)), kStopTokens,
                       {table, tableRows});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    for (uint32_t lane = 0; lane < kLanes; ++lane)
      for (uint32_t row = 0; row < kPositions; ++row) {
        const float *tests = batch.uniforms() + lane * kUniforms + kBlock + row * kChildren;
        const Selection reference = referenceSelection(target.probabilities, children, std::span(tests, kChildren),
                                                       batch.uniforms()[lane * kUniforms + row]);
        if (reference.close)
          continue;
        ++checked;
        seconds += reference.token == 503 ? 1 : 0;
        require(batch.outputTokens()[lane * kRows + row] == reference.token,
                "round " + std::to_string(round) + " lane " + std::to_string(lane) + " row " + std::to_string(row) +
                    ": selected " + std::to_string(batch.outputTokens()[lane * kRows + row]) + ", not " +
                    std::to_string(reference.token));
      }
  }
  require(checked >= 200 && seconds > 0, "too few rows checked the second child's test");
}

// A sampled verify row whose target keeps only candidates, all as likely,
// rejects a drawn child the target does not keep, its draft weighing the kept
// candidates as the target does. Nothing of the target lies outside the
// candidates, so the row selects a kept one, whatever float rounding leaves
// of one less their probabilities: a draw past them would find no token, the
// vocabulary's first range holding none of them. For two to fifteen kept
// candidates, one count a row.
void keptCandidatesAlone(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, kLanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const std::array<uint32_t, kLanes> tableRows{};
  const std::vector<SamplingPolicy> policies(kLanes, SamplingPolicy{0, 1.0F, 1.0F});
  Random random(0x6b657074);
  for (uint32_t uniform = 0; uniform < kLanes * kUniforms; ++uniform)
    batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
  std::vector<uint32_t> counts(batch.rows, 0);
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    for (uint32_t row = 0; row < kPositions; ++row) {
      const uint32_t count = 2 + (lane * kPositions + row) % (kCandidates - 2);
      counts[lane * kRows + row] = count;
      float *logits = batch.row(lane * kRows + row);
      std::fill(logits, logits + vocabulary, -1.0e4F);
      Children children;
      for (uint32_t i = 0; i < kCandidates; ++i) {
        children.ids[i] = i < count ? 500 + 7 * i : 100 + 3 * i;
        children.q[i] = i < count ? 1.0 / count : 1.0e-12;
        if (i < count)
          logits[children.ids[i]] = 0.0F;
      }
      children.draws = {kCandidates - 1};
      setChildren(batch, lane, row, row, children);
    }
  batch.poison();
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch.buffers, layOut(batch.buffers, blocks(kLanes)), kStopTokens,
                     {table, tableRows});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    for (uint32_t row = 0; row < kPositions; ++row) {
      const uint32_t at = lane * kRows + row, token = batch.outputTokens()[at];
      require(token >= 500 && token < 500 + 7 * counts[at] && (token - 500) % 7 == 0,
              std::to_string(counts[at]) + " kept candidates: row " + std::to_string(at) + " selected " +
                  std::to_string(token));
    }
}

// The targets and drafts of the trees the Monte Carlo checks draw: an
// anchor's logits and its distribution under a policy, and those of each of
// its candidates as a child; the rows below a child, alike. The anchor's
// candidates are six of its likeliest tokens and ten others. Its draft favors
// the less likely of the six, so that a draw is often rejected and the row's
// later tests and residuals decide; or, peaked, holds nearly all on the
// second likeliest, the rest a fifteenth each once the target rejects it. The
// next position's candidates, which every child shares, each child's draft
// weighs otherwise.
struct TreeTargets final {
  std::vector<float> anchor, leaf;
  Distribution root;
  Children first;
  std::vector<std::vector<float>> childLogits;
  std::vector<Distribution> childTargets;
  std::vector<Children> second;
};

TreeTargets treeTargets(uint32_t vocabulary, const SamplingPolicy &policy, bool peaked, Random &random) {
  TreeTargets t;
  t.anchor.resize(vocabulary);
  t.leaf.resize(vocabulary);
  fillRow(t.anchor.data(), vocabulary, random);
  fillRow(t.leaf.data(), vocabulary, random);
  t.root = referenceDistribution(t.anchor.data(), vocabulary, policy);
  require(t.root.ambiguousMass == 0.0, "the test's anchor is ambiguous");
  std::vector<uint32_t> ids = referenceBest(t.anchor.data(), vocabulary, 6);
  while (ids.size() < kCandidates) {
    const uint32_t id = random.next() % vocabulary;
    if (std::find(ids.begin(), ids.end(), id) == ids.end())
      ids.push_back(id);
  }
  std::copy(ids.begin(), ids.end(), t.first.ids.begin());
  for (uint32_t i = 0; i < kCandidates; ++i)
    t.first.q[i] = peaked ? (i == 1 ? 1.0 - 1.5e-8 : 1.0e-9) : i < 6 ? 1.0 + i : 0.2;
  const double firstTotal = std::accumulate(t.first.q.begin(), t.first.q.end(), 0.0);
  for (double &value : t.first.q)
    value /= firstTotal;
  t.childLogits.assign(kCandidates, std::vector<float>(vocabulary));
  for (uint32_t i = 0; i < kCandidates; ++i) {
    fillRow(t.childLogits[i].data(), vocabulary, random);
    t.childTargets.push_back(referenceDistribution(t.childLogits[i].data(), vocabulary, policy));
    require(t.childTargets.back().ambiguousMass == 0.0, "the test's child is ambiguous");
  }
  std::vector<uint32_t> next = referenceBest(t.childLogits[0].data(), vocabulary, 5);
  while (next.size() < kCandidates) {
    const uint32_t id = random.next() % vocabulary;
    if (std::find(next.begin(), next.end(), id) == next.end())
      next.push_back(id);
  }
  t.second.resize(kCandidates);
  for (uint32_t i = 0; i < kCandidates; ++i) {
    std::copy(next.begin(), next.end(), t.second[i].ids.begin());
    double total = 0.0;
    for (uint32_t j = 0; j < kCandidates; ++j)
      total += t.second[i].q[j] = peaked ? (j == 1 ? 1.0 - 1.5e-8 : 1.0e-9) : j < 5 ? 1.0 + (j + i) % 5 : 0.2;
    for (double &value : t.second[i].q)
      value /= total;
  }
  return t;
}

// Whether `counts` follow `probabilities`: their chi-square over the outcomes
// expected 5 times or more (the others pooled) lies within five standard
// deviations of its degrees of freedom, which `text` gives. No outcome the
// distribution never gives may be counted.
bool followsDistribution(const std::vector<uint32_t> &counts, const std::vector<double> &probabilities,
                         const std::string &label, std::string &text) {
  const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
  double chi = 0.0, pooledSeen = 0.0, pooledExpected = 0.0;
  uint32_t bins = 0;
  for (uint32_t token = 0; token < counts.size(); ++token) {
    const double want = probabilities[token] * total;
    require(probabilities[token] > 0.0 || counts[token] == 0,
            label + ": selected token " + std::to_string(token) + ", which the target never selects");
    if (want < 5.0) {
      pooledSeen += counts[token];
      pooledExpected += want;
      continue;
    }
    chi += (counts[token] - want) * (counts[token] - want) / want;
    ++bins;
  }
  if (pooledExpected >= 5.0) {
    chi += (pooledSeen - pooledExpected) * (pooledSeen - pooledExpected) / pooledExpected;
    ++bins;
  }
  const double freedom = bins - 1.0;
  text = "chi-square " + std::to_string(chi) + " over " + std::to_string(freedom) + " degrees of freedom";
  return bins >= 2 && chi < freedom + 5.0 * std::sqrt(2.0 * freedom);
}

// The recursion over trees selects from the target, Monte Carlo: a lane's
// anchor tests three children its draft drew without replacement, the first
// two of them verify rows that test two each of their own. Over 6000 such
// trees, their draws and uniforms fresh, the anchor's selection follows the
// target's distribution at the anchor, and a verified child's selection,
// where acceptance walked to it, the target's at that child: each a
// chi-square over the outcomes expected 5 times or more (the others pooled)
// within five standard deviations of its degrees of freedom, for at least
// `walked` children walked 300 times or more. Under `policy`; the drafts
// favor less likely tokens, or, `peaked`, hold all but 1.5e-8 of their mass
// on one candidate, which fp32 holds as 1.
void childTestsOnTrees(MetalBackend &backend, const SamplingPolicy &policy, bool peaked, uint32_t walked) {
  constexpr uint32_t vocabulary = 1003, kRuns = 1500, kTreeRows = 7;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, kLanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const std::array<uint32_t, kLanes> tableRows{};
  const std::vector<SamplingPolicy> policies(kLanes, policy);
  const std::string variant = "temperature " + std::to_string(policy.temperature) + (peaked ? ", peaked: " : ": ");
  Random random(0x74726565);
  std::mt19937_64 rng(0x74726565);
  const TreeTargets t = treeTargets(vocabulary, policy, peaked, random);
  const std::vector<float> &anchor = t.anchor, &leaf = t.leaf;
  const Distribution &root = t.root;
  const Children &first = t.first;
  const std::vector<std::vector<float>> &childLogits = t.childLogits;
  const std::vector<Distribution> &childTargets = t.childTargets;
  const std::vector<Children> &second = t.second;
  std::vector<uint32_t> anchorCounts(vocabulary, 0);
  std::vector<std::vector<uint32_t>> childCounts(kCandidates, std::vector<uint32_t>(vocabulary, 0));
  // Each lane's rows: its tree's, from its block's first row, whose rows past
  // them are of no lane.
  std::vector<LaneRows> laneRows = blocks(kLanes);
  for (LaneRows &rows : laneRows) rows.rows = kTreeRows;
  const std::array<uint32_t, kLanes> limits{kRows, kRows, kRows, kRows};
  const AcceptanceBuffers acceptance = acceptanceBuffers(backend, batch, kLanes);
  // Preorder: the anchor, the first child and its two, the second and its two.
  constexpr std::array<uint32_t, kTreeRows> parents{0, 0, 1, 1, 0, 4, 4};
  for (uint32_t run = 0; run < kRuns; ++run) {
    std::vector<std::array<uint32_t, 2>> verified(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      Children top = first;
      top.draws = drawChildren(first.q, 3, rng);
      setChildren(batch, lane, 0, 0, top);
      std::copy(anchor.begin(), anchor.end(), batch.row(lane * kRows));
      auto *tree = static_cast<TreeRow *>(batch.buffers.treeRows.contents()) + lane * kRows;
      uint32_t *inputs = batch.inputTokens() + lane * kRows;
      std::array<uint32_t, kRows> ancestors{}, depths{};
      for (uint32_t row = 0; row < kRows; ++row) {
        const uint32_t parent = row < kTreeRows ? parents[row] : row - 1;
        ancestors[row] = (row ? ancestors[parent] : 0U) | (1U << row);
        depths[row] = row ? depths[parent] + 1 : 0;
        tree[row] = {ancestors[row], static_cast<uint16_t>(row ? parent : SPLASH_TREE_ROOT),
                     static_cast<uint16_t>(depths[row])};
      }
      inputs[0] = 1;
      for (uint32_t branch = 0; branch < 2; ++branch) {
        const uint32_t index = top.draws[branch], at = branch ? 4 : 1;
        verified[lane][branch] = index;
        inputs[at] = first.ids[index];
        std::copy(childLogits[index].begin(), childLogits[index].end(), batch.row(lane * kRows + at));
        Children grand = second[index];
        grand.draws = drawChildren(grand.q, 2, rng);
        setChildren(batch, lane, at, 1, grand);
        for (uint32_t grandchild = 0; grandchild < 2; ++grandchild) {
          inputs[at + 1 + grandchild] = grand.ids[grand.draws[grandchild]];
          std::copy(leaf.begin(), leaf.end(), batch.row(lane * kRows + at + 1 + grandchild));
        }
      }
    }
    for (uint32_t uniform = 0; uniform < kLanes * kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    batch.poison();
    CommandGraph graph;
    const StepRows step = layOut(batch.buffers, laneRows);
    sampling.addVerify(graph, policies, batch.buffers, step, kStopTokens, {table, tableRows});
    sampling.addAcceptance(graph, acceptance, step, limits, kStopTokens);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *retained = static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      const uint32_t *kept = static_cast<const uint32_t *>(acceptance.outputTokens.contents()) + lane * kRows;
      ++anchorCounts[kept[0]];
      for (uint32_t branch = 0; branch < 2 && retained[lane] >= 2; ++branch)
        if (kept[0] == first.ids[verified[lane][branch]])
          ++childCounts[verified[lane][branch]][kept[1]];
    }
  }
  const auto chiSquare = [](const std::vector<uint32_t> &counts, const std::vector<double> &probabilities,
                            const std::string &label) {
    std::string text;
    require(followsDistribution(counts, probabilities, label, text), label + ": " + text);
  };
  chiSquare(anchorCounts, root.probabilities, variant + "the anchor's selections");
  uint32_t children = 0;
  for (uint32_t i = 0; i < kCandidates; ++i) {
    if (std::accumulate(childCounts[i].begin(), childCounts[i].end(), 0U) < 300)
      continue;
    chiSquare(childCounts[i], childTargets[i].probabilities, variant + "child " + std::to_string(i) + "'s selections");
    ++children;
  }
  require(children >= walked, variant + "too few children were walked to check their selections");
}

// Selections under a floor on the verify rows (VerifyRowsParams) are the
// target's, Monte Carlo. A lane's anchor draws three children, the first two
// of them rows of its tree, each of which draws two of its own that have no
// row, and draft_verify_rows lays each tree out in a step of three rows a
// lane at a floor between the children's estimates: a child is verified in
// some trees and in others left out, its token the lane's last. With each
// child estimated as the selector estimates it, by the draft's probability
// of the draw that made it and no more than the child drawn before it, the
// anchor's selections follow the target's distribution at the anchor and a
// verified child's, where acceptance walked to it, the target's at that
// child (chi-squares as childTestsOnTrees', for two children or more walked
// 300 times). With each child estimated by the probability of its own first
// draw instead (`ownDraws`), which no selector does, the same floor verifies
// a row for its own draws, and the children's selections must not follow
// their targets: the check tells such a rule. The anchor's draft here favors
// the likelier of its six likeliest candidates, so that the children over
// the floor are those the target selects most.
void childTestsUnderAFloor(MetalBackend &backend, const SamplingPolicy &policy, bool ownDraws) {
  constexpr uint32_t vocabulary = 1003, kRuns = 1500, kTreeRows = 3;
  constexpr float kFloor = 0.15F;
  Sampling sampling(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, kLanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const std::array<uint32_t, kLanes> tableRows{};
  const std::vector<SamplingPolicy> policies(kLanes, policy);
  const std::string variant = ownDraws ? "a floor on estimates of a row's own draws: " : "a floor: ";
  Random random(0x666c6f72);
  std::mt19937_64 rng(0x666c6f72);
  TreeTargets t = treeTargets(vocabulary, policy, false, random);
  std::reverse(t.first.q.begin(), t.first.q.begin() + 6);
  // The lanes' trees, which draft_verify_rows lays out as the batch's rows.
  DraftSelectorBuffers trees{};
  trees.treeTokens = allocate(backend, uint64_t{kLanes} * kBlock * sizeof(uint32_t));
  trees.trees = allocate(backend, uint64_t{kLanes} * kBlock * sizeof(TreeRow));
  trees.treeScores = allocate(backend, uint64_t{kLanes} * kBlock * 2 * sizeof(float));
  trees.treeSizes = allocate(backend, uint64_t{kLanes} * sizeof(uint32_t));
  trees.inputTokens = batch.buffers.inputTokens;
  trees.treeRows = batch.buffers.treeRows;
  trees.laneRows = batch.buffers.laneRows;
  trees.verifySources = batch.buffers.verifySources;
  auto *tokens = static_cast<uint32_t *>(trees.treeTokens.contents());
  auto *rows = static_cast<TreeRow *>(trees.trees.contents());
  auto *scores = static_cast<float *>(trees.treeScores.contents());
  const StepRows step{kLanes, kLanes * kTreeRows, kNarrowLaneRows};
  std::array<uint32_t, kLanes> limits{};
  limits.fill(step.laneCap);
  const AcceptanceBuffers acceptance = acceptanceBuffers(backend, batch, kLanes);
  const auto *laneRows = static_cast<const LaneRows *>(batch.buffers.laneRows.contents());
  const auto *sources = static_cast<const uint32_t *>(batch.buffers.verifySources.contents());
  std::vector<uint32_t> anchorCounts(vocabulary, 0);
  std::vector<std::vector<uint32_t>> childCounts(kCandidates, std::vector<uint32_t>(vocabulary, 0));
  uint32_t verified = 0;
  for (uint32_t run = 0; run < kRuns; ++run) {
    std::vector<std::array<uint32_t, 2>> drawn(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      Children top = t.first;
      top.draws = drawChildren(t.first.q, 3, rng);
      setChildren(batch, lane, 0, 0, top);
      static_cast<uint32_t *>(trees.treeSizes.contents())[lane] = kTreeRows;
      tokens[lane * kBlock] = 1;
      rows[lane * kBlock] = {1, SPLASH_TREE_ROOT, 0};
      float estimate = 1.0F;
      for (uint32_t branch = 0; branch < 2; ++branch) {
        const uint32_t index = top.draws[branch], row = 1 + branch;
        drawn[lane][branch] = index;
        Children grand = t.second[index];
        grand.draws = drawChildren(grand.q, 2, rng);
        setChildren(batch, lane, row, 1, grand);
        tokens[lane * kBlock + row] = t.first.ids[index];
        rows[lane * kBlock + row] = {1U | 1U << row, 0, 1};
        estimate = std::min(estimate, static_cast<float>(ownDraws ? grand.q[grand.draws[0]] : t.first.q[index]));
        scores[2 * (lane * kBlock + row) + 1] = estimate;
      }
    }
    CommandGraph layout;
    DraftSelector::addVerifyRows(layout, trees, step, kFloor);
    static_cast<void>(backend.submitCommandAsync(layout.dispatches()).wait());
    // Each row's logits: its tree row's, the anchor's or a drawn child's.
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      verified += laneRows[lane].rows - 1;
      for (uint32_t at = 0; at < laneRows[lane].rows; ++at) {
        const uint32_t source = sources[laneRows[lane].first + at];
        const std::vector<float> &logits = source ? t.childLogits[drawn[lane][source - 1]] : t.anchor;
        std::copy(logits.begin(), logits.end(), batch.row(laneRows[lane].first + at));
      }
    }
    for (uint32_t uniform = 0; uniform < kLanes * kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    batch.poison();
    CommandGraph graph;
    sampling.addVerify(graph, policies, batch.buffers, step, kStopTokens, {table, tableRows});
    sampling.addAcceptance(graph, acceptance, step, limits, kStopTokens);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *retained = static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      const uint32_t *kept = static_cast<const uint32_t *>(acceptance.outputTokens.contents()) + laneRows[lane].first;
      ++anchorCounts[kept[0]];
      for (uint32_t branch = 0; branch + 1 < laneRows[lane].rows && retained[lane] >= 2; ++branch)
        if (kept[0] == t.first.ids[drawn[lane][branch]])
          ++childCounts[drawn[lane][branch]][kept[1]];
    }
  }
  // The floor left a child out of some trees and verified it in others.
  const uint32_t drawnChildren = 2 * kRuns * kLanes;
  require(verified > drawnChildren / 5 && verified < drawnChildren * 4 / 5,
          variant + std::to_string(verified) + " of " + std::to_string(drawnChildren) +
              " children were verified, too many or too few for the floor to matter");
  std::string text;
  require(followsDistribution(anchorCounts, t.root.probabilities, variant + "the anchor's selections", text),
          variant + "the anchor's selections: " + text);
  uint32_t children = 0;
  std::string failed;
  for (uint32_t i = 0; i < kCandidates; ++i) {
    if (std::accumulate(childCounts[i].begin(), childCounts[i].end(), 0U) < 300)
      continue;
    ++children;
    if (!followsDistribution(childCounts[i], t.childTargets[i].probabilities, variant + "child " + std::to_string(i),
                             text))
      failed += " child " + std::to_string(i) + "'s selections: " + text + ";";
  }
  require(children >= 2, variant + "too few children were walked to check their selections");
  if (ownDraws)
    require(!failed.empty(), variant + "every child's selections followed its target, so the check cannot tell "
                                       "a floor that verifies a row for its own draws");
  else
    require(failed.empty(), variant + failed);
}

// The selector's own trees select from the target, Monte Carlo, end to end
// on the device, where childTestsOnTrees' and childTestsUnderAFloor's draws
// are the host's: each run, draft_select_tree grows every lane's tree from
// the draft's candidates with fresh noise (DraftTreeLane::noise) and records
// of each row the children it drew, in the order drawn, and the q_T its keys
// drew them from (draws, next), draft_verify_rows lays the trees out, the
// verify rows test those records and acceptance walks the rows. A lane's
// anchor draws three children at the policy's temperature from the
// candidates of treeTargets' anchor, its draft favoring the less likely of
// its six likeliest; each child row it takes draws three of its own a
// position deeper, over edges of its own predecessor; and the tree takes
// three rows (depth 2): the first child drawn, then the second or a
// grandchild. A selector that records its draws in another order than it
// drew them, or keeps in `next` another q than its keys drew from, moves the
// anchor's selections off the target's distribution. Over 1500 runs of four
// lanes (6000 trees), the anchor's selections follow the target's
// distribution at the anchor, and a child's, where acceptance walked to it
// 300 times or more (two children at least), the target's at that child, by
// followsDistribution's chi-square: the policy's top 20 at temperature 0.7
// and top_p 0.95 leave each check here 8 degrees of freedom (the anchor and
// four children walked 443 to 902 times), at which a correct sampler fails
// one with probability 4.7e-4 (the chi-square tail past nu + 5 sqrt(2 nu)),
// all five below 2.4e-3; the seeds are fixed, so a passing run repeats.
void childTestsOfTheSelectorsDraws(MetalBackend &backend, const SamplingPolicy &policy) {
  constexpr uint32_t vocabulary = 1003, kRuns = 1500, kTreeRows = 3, kDraftRows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr uint32_t kRank = SPLASH_DRAFT_SELECTOR_RANK;
  Sampling sampling(vocabulary);
  const DraftSelector selector(vocabulary);
  const Batch batch = makeBatch(backend, vocabulary, kLanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const std::array<uint32_t, kLanes> tableRows{};
  const std::vector<SamplingPolicy> policies(kLanes, policy);
  Random random(0x73656c64);
  const TreeTargets t = treeTargets(vocabulary, policy, false, random);
  // The draft's rows: at position 0 the anchor's candidates at their draft
  // probabilities, at position 1 the children's shared ones at the first
  // child's, every other token far below. bf16 edge factors of 0.1 at most,
  // so that a child's next probabilities follow its own predecessor.
  const auto workspace = DraftSelector::workspace(kLanes * kPositions);
  const auto bfloat = [&](uint64_t count) {
    MetalBuffer buffer = allocate(backend, count * sizeof(uint16_t));
    auto *values = static_cast<uint16_t *>(buffer.contents());
    for (uint64_t i = 0; i < count; ++i) {
      const float value = 0.1F * random.unit();
      uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      values[i] = static_cast<uint16_t>(bits >> 16);
    }
    return buffer;
  };
  DraftSelectorBuffers trees{};
  trees.logits = allocate(backend, uint64_t{kLanes} * kDraftRows * vocabulary * sizeof(float));
  trees.partialIds = allocate(backend, workspace.partialIdsBytes);
  trees.partialValues = allocate(backend, workspace.partialValuesBytes);
  trees.candidates = batch.buffers.candidates;
  trees.unary = allocate(backend, workspace.unaryBytes);
  trees.selectorHidden = bfloat(uint64_t{kLanes} * kDraftRows * kRank);
  trees.chainTokens = allocate(backend, uint64_t{kLanes} * SPLASH_TREE_CHAIN_TOKENS * sizeof(uint32_t));
  trees.chainRates =
      allocate(backend, uint64_t{kLanes} * SPLASH_TREE_CHAIN_TOKENS * SPLASH_TREE_AGREEMENTS * sizeof(float));
  trees.treeTokens = allocate(backend, uint64_t{kLanes} * kBlock * sizeof(uint32_t));
  trees.trees = allocate(backend, uint64_t{kLanes} * kBlock * sizeof(TreeRow));
  trees.treeScores = allocate(backend, uint64_t{kLanes} * kBlock * 2 * sizeof(float));
  trees.treeSizes = allocate(backend, uint64_t{kLanes} * sizeof(uint32_t));
  trees.treeDraws = batch.buffers.treeDraws;
  trees.treeNext = batch.buffers.treeNext;
  trees.treeCaps = allocate(backend, uint64_t{kLanes} * kBlock * sizeof(float));
  trees.inputTokens = batch.buffers.inputTokens;
  trees.treeRows = batch.buffers.treeRows;
  trees.laneRows = batch.buffers.laneRows;
  trees.verifySources = batch.buffers.verifySources;
  const DraftCodebooks codebooks{bfloat(uint64_t{vocabulary} * kRank), bfloat(uint64_t{vocabulary} * kRank)};
  auto *draft = static_cast<float *>(trees.logits.contents());
  std::fill_n(draft, uint64_t{kLanes} * kDraftRows * vocabulary, -30.0F);
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    for (uint32_t i = 0; i < kCandidates; ++i) {
      draft[(uint64_t{lane} * kDraftRows + 1) * vocabulary + t.first.ids[i]] = std::log(float(t.first.q[i]));
      draft[(uint64_t{lane} * kDraftRows + 2) * vocabulary + t.second[0].ids[i]] = std::log(float(t.second[0].q[i]));
    }
  const StepRows step{kLanes, kLanes * kTreeRows, kNarrowLaneRows};
  std::array<uint32_t, kLanes> limits{};
  limits.fill(step.laneCap);
  const AcceptanceBuffers acceptance = acceptanceBuffers(backend, batch, kLanes);
  const auto *laneRows = static_cast<const LaneRows *>(batch.buffers.laneRows.contents());
  const auto *sources = static_cast<const uint32_t *>(batch.buffers.verifySources.contents());
  const auto *tokens = static_cast<const uint32_t *>(trees.treeTokens.contents());
  const auto *rows = static_cast<const TreeRow *>(trees.trees.contents());
  // A candidate's index in treeTargets' order, whose child targets it takes.
  const auto indexOf = [&](uint32_t token) {
    return static_cast<uint32_t>(std::find(t.first.ids.begin(), t.first.ids.end(), token) - t.first.ids.begin());
  };
  std::vector<uint32_t> anchorCounts(vocabulary, 0);
  std::vector<std::vector<uint32_t>> childCounts(kCandidates, std::vector<uint32_t>(vocabulary, 0));
  for (uint32_t run = 0; run < kRuns; ++run) {
    std::vector<DraftTreeLane> lanes;
    for (uint32_t lane = 0; lane < kLanes; ++lane)
      lanes.push_back({7, 2, 0, 3, DraftCalibration{}, policy.temperature,
                       uint64_t{random.next()} << 32 | random.next()});
    CommandGraph layout;
    selector.add(layout, trees, codebooks, DraftVocabulary{}, lanes);
    DraftSelector::addVerifyRows(layout, trees, step);
    static_cast<void>(backend.submitCommandAsync(layout.dispatches()).wait());
    // Each row's logits: its tree row's, the anchor's, a child's or a
    // grandchild's.
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      require(laneRows[lane].rows == kTreeRows, "a lane's tree did not take three rows");
      for (uint32_t at = 0; at < laneRows[lane].rows; ++at) {
        const uint32_t row = lane * kBlock + sources[laneRows[lane].first + at];
        const std::vector<float> &logits = rows[row].depth == 0   ? t.anchor
                                           : rows[row].depth == 1 ? t.childLogits[indexOf(tokens[row])]
                                                                  : t.leaf;
        std::copy(logits.begin(), logits.end(), batch.row(laneRows[lane].first + at));
      }
    }
    for (uint32_t uniform = 0; uniform < kLanes * kUniforms; ++uniform)
      batch.uniforms()[uniform] = 0.5F * (random.unit() + 1.0F);
    batch.poison();
    CommandGraph graph;
    sampling.addVerify(graph, policies, batch.buffers, step, kStopTokens, {table, tableRows});
    sampling.addAcceptance(graph, acceptance, step, limits, kStopTokens);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *retained = static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      const uint32_t *kept = static_cast<const uint32_t *>(acceptance.outputTokens.contents()) + laneRows[lane].first;
      ++anchorCounts[kept[0]];
      if (retained[lane] >= 2)
        ++childCounts[indexOf(kept[0])][kept[1]];
    }
  }
  const std::string variant = "the selector's draws at temperature " + std::to_string(policy.temperature) + ": ";
  std::string text;
  require(followsDistribution(anchorCounts, t.root.probabilities, variant + "the anchor's selections", text),
          variant + "the anchor's selections: " + text);
  uint32_t children = 0;
  for (uint32_t i = 0; i < kCandidates; ++i) {
    if (std::accumulate(childCounts[i].begin(), childCounts[i].end(), 0U) < 300)
      continue;
    const std::string label = variant + "child " + std::to_string(i) + "'s selections";
    require(followsDistribution(childCounts[i], t.childTargets[i].probabilities, label, text), label + ": " + text);
    ++children;
  }
  require(children >= 2, variant + "too few children were walked to check their selections");
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
    stage = "policies that keep one token";
    onlyTheMostLikelyIsGreedy();
    stage = "recursive rejection sampling";
    recursiveRejectionSampling();
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
    stage = "extreme searches";
    extremeSearches(backend);
    stage = "unconstrained rows ignore masks";
    unconstrainedRowsIgnoreMasks(backend);
    stage = "packed lanes of mixed row counts";
    packedLanes(backend);
    stage = "tree selections";
    treeSelections(backend);
    stage = "child tests";
    childTests(backend);
    stage = "kept candidates alone";
    keptCandidatesAlone(backend);
    stage = "a dominant child";
    dominantChild(backend);
    stage = "child tests on trees";
    childTestsOnTrees(backend, SamplingPolicy{20, 1.0F, 0.95F}, false, 2);
    stage = "child tests on trees of peaked drafts at temperature 0.3";
    childTestsOnTrees(backend, SamplingPolicy{20, 0.3F, 0.95F}, true, 1);
    stage = "child tests under a floor";
    childTestsUnderAFloor(backend, SamplingPolicy{20, 1.0F, 0.95F}, false);
    stage = "child tests under a floor on estimates of a row's own draws";
    childTestsUnderAFloor(backend, SamplingPolicy{20, 1.0F, 0.95F}, true);
    stage = "child tests of the selector's own draws";
    childTestsOfTheSelectorsDraws(backend, SamplingPolicy{20, 0.7F, 0.95F});
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
      stage = "thinking cutoffs, vocabulary " + std::to_string(vocabulary);
      thinkingCutoffs(backend, vocabulary);
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
