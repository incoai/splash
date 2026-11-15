// RFlash's relaxation of a verify (decode_relax_dflash) against a
// double-precision solver of the block program (RFlashReference.hpp), and
// the draw from the relaxed rows.
//
// The kernel's relaxed probabilities, rest ratios and status must match the
// reference's on random verify batches of one to four lanes, at several
// prices and allowances, whatever rows each lane verifies, on whole rows
// and on rows a top-k, a nucleus or a min-p cuts (candidates outside the
// kept tokens, and a tail at or below the floor). A lane whose constraint
// no price up to the cap
// satisfies keeps the cap's solution. The rows past a lane's verify rows,
// the rows from its thinking cutoff on, a row without a finite kept mass
// and every row of a greedy lane stay untouched (rest 0) and select bitwise
// what a verify without the relaxation selects. The draw of a relaxed row
// must record the relaxed probability of its draft token and draw its
// correction where the residual of the relaxed row, in token order, places
// the uniform. At an enormous price the relaxation vanishes. The search
// records each row's kept mass, which the relaxation reads the candidates'
// probabilities with. Every buffer the relaxation reaches is checked at its
// extent.
#include "RFlashReference.hpp"
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
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;
using splash::test::kRelaxCandidates;
using splash::test::rejects;
using splash::test::relaxReference;
using splash::test::RelaxReferenceResult;
using splash::test::RelaxReferenceRow;
using splash::test::require;
using splash::test::requireExtent;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kUniforms = SPLASH_SAMPLING_UNIFORMS;
constexpr StopTokens kStopTokens{1, 2};
constexpr uint32_t kVocabulary = 4099;

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  // Uniform in [-1, 1).
  float unit() { return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F; }
  // Uniform in [0, 1).
  double fraction() { return static_cast<double>(next() & 0xFFFFFF) / 16777216.0; }
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }

private:
  uint64_t state_;
};

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, "rflash");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

struct Batch final {
  SamplingBuffers buffers;
  uint32_t lanes = 0;

  [[nodiscard]] float *row(uint32_t lane, uint32_t index) const {
    return static_cast<float *>(buffers.logits.contents()) + (uint64_t{lane} * kRows + index) * kVocabulary;
  }
  [[nodiscard]] uint32_t *inputTokens() const { return static_cast<uint32_t *>(buffers.inputTokens.contents()); }
  [[nodiscard]] const uint32_t *outputTokens() const {
    return static_cast<const uint32_t *>(buffers.outputTokens.contents());
  }
  [[nodiscard]] float *uniforms() const { return static_cast<float *>(buffers.uniforms.contents()); }
  [[nodiscard]] uint32_t *candidates(uint32_t lane, uint32_t position) const {
    return static_cast<uint32_t *>(buffers.draftCandidates.contents()) +
           (uint64_t{lane} * kPositions + position) * kRelaxCandidates;
  }
  [[nodiscard]] float *probabilities(uint32_t lane, uint32_t position) const {
    return static_cast<float *>(buffers.draftProbabilities.contents()) +
           (uint64_t{lane} * kPositions + position) * kRelaxCandidates;
  }
  [[nodiscard]] uint32_t *verifyRows() const { return static_cast<uint32_t *>(buffers.verifyRows.contents()); }
  [[nodiscard]] const float *relaxed(uint32_t lane, uint32_t position) const {
    return static_cast<const float *>(buffers.relaxedProbabilities.contents()) +
           (uint64_t{lane} * kPositions + position) * kRelaxCandidates;
  }
  [[nodiscard]] float rest(uint32_t lane, uint32_t position) const {
    return static_cast<const float *>(buffers.relaxedRest.contents())[lane * kPositions + position];
  }
  [[nodiscard]] uint32_t status(uint32_t lane) const {
    return static_cast<const uint32_t *>(buffers.relaxedStatus.contents())[lane];
  }
  [[nodiscard]] const TargetVocabularyRow &record(uint32_t lane, uint32_t index) const {
    return static_cast<const TargetVocabularyRow *>(buffers.vocabularyRows.contents())[lane * kRows + index];
  }
  // Outputs a missing dispatch would leave behind cannot pass as results.
  void poison() const {
    for (const auto &buffer : {buffers.partialMasses, buffers.vocabularyRows, buffers.vocabularyRanges,
                               buffers.argmaxValues, buffers.argmaxIndices, buffers.outputTokens,
                               buffers.relaxedProbabilities, buffers.relaxedRest, buffers.relaxedStatus})
      std::memset(buffer.contents(), 0xA5, buffer.sizeBytes());
  }
};

Batch makeBatch(MetalBackend &backend, uint32_t lanes) {
  const uint32_t rows = lanes * kRows;
  const auto space = Sampling::workspace(rows);
  const auto proposals = DraftSelector::workspace(lanes * kPositions);
  Batch batch;
  batch.lanes = lanes;
  batch.buffers = SamplingBuffers{allocate(backend, uint64_t{rows} * kVocabulary * sizeof(float)),
                                  allocate(backend, space.partialMassesBytes),
                                  allocate(backend, space.vocabularyRowsBytes),
                                  allocate(backend, uint64_t{lanes} * kUniforms * sizeof(float)),
                                  allocate(backend, uint64_t{lanes} * (kRows + 1) * ((kVocabulary + 31) / 32) * 4),
                                  allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
                                  allocate(backend, space.argmaxValuesBytes),
                                  allocate(backend, space.argmaxIndicesBytes),
                                  allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
                                  allocate(backend, proposals.candidatesBytes),
                                  allocate(backend, proposals.proposalProbabilitiesBytes),
                                  allocate(backend, space.vocabularyRangesBytes),
                                  allocate(backend, space.vocabularyArrivalsBytes),
                                  allocate(backend, uint64_t{lanes} * sizeof(uint32_t)),
                                  allocate(backend, uint64_t{lanes} * kPositions * kRelaxCandidates * sizeof(float)),
                                  allocate(backend, uint64_t{lanes} * kPositions * sizeof(float)),
                                  allocate(backend, uint64_t{lanes} * sizeof(uint32_t))};
  std::fill_n(batch.verifyRows(), lanes, kRows);
  return batch;
}

// Low noise with forty distinct spikes: the production shape. The stop
// tokens stay below every spike.
void fillRow(float *row, Random &random) {
  for (uint32_t token = 0; token < kVocabulary; ++token)
    row[token] = -8.0F + 3.0F * random.unit() * random.unit();
  for (uint32_t spike = 0; spike < 40; ++spike) {
    const uint32_t token = 3 + random.next() % (kVocabulary - 3);
    row[token] = 1.0F + 0.25F * float(spike) + 0.1F * random.unit();
  }
}

// The row's softmax weights at the policy's temperature and the sum of
// those the policy keeps: what the kernels read the candidates'
// probabilities with. The tokens that weigh at least minP of the heaviest
// stay, then the topK heaviest of those, then of those the nucleus: the
// heaviest up to the first whose running mass passes topP of theirs
// (distribution_end). The others weigh 0. The rows' weights are distinct.
struct RowWeights final {
  std::vector<double> weights;
  double mass = 0.0;
  [[nodiscard]] double probability(uint32_t token) const { return weights[token] / mass; }
};

RowWeights rowWeights(const float *row, const SamplingPolicy &policy) {
  const float maximum = *std::max_element(row, row + kVocabulary);
  std::vector<double> all(kVocabulary);
  std::vector<uint32_t> kept;
  for (uint32_t token = 0; token < kVocabulary; ++token) {
    all[token] = std::exp((double(row[token]) - double(maximum)) / double(policy.temperature));
    if (all[token] >= double(policy.minP)) kept.push_back(token);
  }
  std::sort(kept.begin(), kept.end(), [&](uint32_t a, uint32_t b) { return all[a] > all[b]; });
  if (policy.topK && policy.topK < kept.size()) kept.resize(policy.topK);
  if (policy.topP < 1.0F) {
    double mass = 0.0;
    for (const uint32_t token : kept) mass += all[token];
    double walked = 0.0;
    for (size_t index = 0; index < kept.size(); ++index) {
      walked += all[kept[index]];
      if (walked > double(policy.topP) * mass) {
        kept.resize(index + 1);
        break;
      }
    }
  }
  RowWeights result;
  result.weights.assign(kVocabulary, 0.0);
  for (const uint32_t token : kept) {
    result.weights[token] = all[token];
    result.mass += all[token];
  }
  return result;
}

// The draft at a position: sixteen distinct candidates, the row's twelve
// best tokens and four others, with q the tempered softmax of the row's
// logits plus noise over them; the draft token is one of them.
void setDraft(const Batch &batch, uint32_t lane, uint32_t position, float temperature, Random &random) {
  const float *row = batch.row(lane, position);
  std::vector<uint32_t> order(kVocabulary);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + 12, order.end(),
                    [&](uint32_t a, uint32_t b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
  std::vector<uint32_t> ids(order.begin(), order.begin() + 12);
  while (ids.size() < kRelaxCandidates) {
    const uint32_t token = random.next() % kVocabulary;
    if (std::find(ids.begin(), ids.end(), token) == ids.end()) ids.push_back(token);
  }
  std::array<double, kRelaxCandidates> scores{};
  double maximum = -std::numeric_limits<double>::infinity();
  for (uint32_t k = 0; k < kRelaxCandidates; ++k) {
    scores[k] = (double(row[ids[k]]) + 1.5 * double(random.unit())) / double(temperature);
    maximum = std::max(maximum, scores[k]);
  }
  double sum = 0.0;
  for (double &score : scores) {
    score = std::exp(score - maximum);
    sum += score;
  }
  uint32_t *candidates = batch.candidates(lane, position);
  float *probabilities = batch.probabilities(lane, position);
  for (uint32_t k = 0; k < kRelaxCandidates; ++k) {
    candidates[k] = ids[k];
    probabilities[k] = static_cast<float>(scores[k] / sum);
  }
  // Draft token r is verify input row r + 1 of the lane.
  batch.inputTokens()[lane * kRows + position + 1] = ids[random.next() % 12];
}

// A random verify batch: every lane's rows filled, drafts at every position,
// anchors and uniforms set.
void fillBatch(const Batch &batch, std::span<const SamplingPolicy> policies, Random &random) {
  for (uint32_t lane = 0; lane < batch.lanes; ++lane) {
    for (uint32_t index = 0; index < kRows; ++index) fillRow(batch.row(lane, index), random);
    batch.inputTokens()[lane * kRows] = 7;
    const float temperature = policies[lane].samples() ? policies[lane].temperature : 1.0F;
    for (uint32_t position = 0; position < kPositions; ++position) setDraft(batch, lane, position, temperature, random);
    for (uint32_t uniform = 0; uniform < kUniforms; ++uniform)
      batch.uniforms()[lane * kUniforms + uniform] = static_cast<float>(random.fraction());
  }
}

// The reference's inputs of a lane's drafted rows: each candidate's target
// probability as the kernel reads it, and its draft probability.
std::vector<RelaxReferenceRow> referenceRows(const Batch &batch, uint32_t lane, uint32_t rows,
                                             const SamplingPolicy &policy) {
  std::vector<RelaxReferenceRow> result(rows);
  for (uint32_t position = 0; position < rows; ++position) {
    const RowWeights weights = rowWeights(batch.row(lane, position), policy);
    const uint32_t *candidates = batch.candidates(lane, position);
    const float *probabilities = batch.probabilities(lane, position);
    for (uint32_t k = 0; k < kRelaxCandidates; ++k) {
      result[position].p[k] = weights.probability(candidates[k]);
      result[position].q[k] = probabilities[k];
    }
  }
  return result;
}

void runVerify(MetalBackend &backend, const Sampling &sampling, const Batch &batch,
               std::span<const SamplingPolicy> policies) {
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch.buffers, kStopTokens, PenaltyTable{});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

SamplingPolicy sampledPolicy(float temperature = 0.9F, uint32_t topK = 0) {
  SamplingPolicy policy;
  policy.topK = topK;
  policy.temperature = temperature;
  policy.topP = 1.0F;
  return policy;
}

// A sampled policy that cuts its rows.
SamplingPolicy cutPolicy(uint32_t topK, float topP, float minP) {
  SamplingPolicy policy = sampledPolicy(0.9F, topK);
  policy.topP = topP;
  policy.minP = minP;
  return policy;
}

struct Outcome final {
  uint32_t slack = 0;
  uint32_t bound = 0;
  uint32_t capped = 0;
};

// A lane's relaxation against the reference of its first `rows` drafted
// rows, the lane's block: the status, each row's candidates and rest, and
// that each relaxed row is a distribution. Returns the status.
uint32_t laneMatchesReference(const Batch &batch, uint32_t lane, uint32_t rows, const SamplingPolicy &policy,
                              const RelaxationSetting &setting) {
  const std::vector<RelaxReferenceRow> inputs = referenceRows(batch, lane, rows, policy);
  const RelaxReferenceResult reference = relaxReference(inputs, setting);
  const uint32_t status = batch.status(lane);
  require(status == reference.status, "the relaxation's status differs from the reference's (lane " +
                                          std::to_string(lane) + ": " + std::to_string(status) + " vs " +
                                          std::to_string(reference.status) + ")");
  for (uint32_t position = 0; position < rows; ++position) {
    const float rest = batch.rest(lane, position);
    require(rest > 0.0F, "a relaxed row has no rest ratio");
    const float *relaxed = batch.relaxed(lane, position);
    double mass = 0.0, tail = 1.0;
    // The kernel solves in float with a secant search, the reference in
    // double by bisection: on rows of this shape the reference
    // implementation's kernel stays within 2e-3 of this reference at
    // every setting but the lowest price, where it reaches 4e-3.
    constexpr double kTolerance = 4e-3;
    for (uint32_t k = 0; k < kRelaxCandidates; ++k) {
      require(std::isfinite(relaxed[k]) && relaxed[k] >= 0.0F, "a relaxed probability is not a probability");
      require(std::fabs(double(relaxed[k]) - reference.relaxed[position][k]) <= kTolerance,
              "a relaxed probability differs from the reference (lane " + std::to_string(lane) + " row " +
                  std::to_string(position) + " slot " + std::to_string(k) + ": " + std::to_string(relaxed[k]) +
                  " vs " + std::to_string(reference.relaxed[position][k]) + ")");
      mass += relaxed[k];
      tail -= inputs[position].p[k];
    }
    // The rest's mass, not its ratio, which is sensitive on a tiny tail.
    tail = std::max(tail, 0.0);
    require(std::fabs(double(rest) - reference.rest[position]) * tail <= kTolerance,
            "a rest ratio differs from the reference (lane " + std::to_string(lane) + " row " +
                std::to_string(position) + ": " + std::to_string(rest) + " vs " +
                std::to_string(reference.rest[position]) + ")");
    // The relaxed row is a distribution.
    require(std::fabs(mass + tail * double(rest) - 1.0) <= kTolerance, "a relaxed row does not sum to 1");
  }
  return status;
}

// The kernel against the reference on `lanes` lanes that sample under
// `policy` and verify `verifyRows` rows each.
void solveMatchesReference(MetalBackend &backend, const RelaxationSetting &setting, uint32_t lanes,
                           std::span<const uint32_t> verifyRows, uint64_t seed, Outcome &outcome,
                           const SamplingPolicy &policy = sampledPolicy()) {
  Random random(seed);
  const Sampling sampling(kVocabulary, setting);
  Batch batch = makeBatch(backend, lanes);
  std::vector<SamplingPolicy> policies(lanes, policy);
  fillBatch(batch, policies, random);
  for (uint32_t lane = 0; lane < lanes; ++lane) batch.verifyRows()[lane] = verifyRows[lane];
  batch.poison();
  runVerify(backend, sampling, batch, policies);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t rows = std::min(verifyRows[lane], kRows) - 1;
    if (!rows) {
      require(batch.status(lane) == SPLASH_RELAX_STATUS_SLACK,
              "a lane that verifies only its anchor was not left slack");
      for (uint32_t position = 0; position < kPositions; ++position)
        require(batch.rest(lane, position) == 0.0F, "a lane that verifies only its anchor relaxed a row");
      continue;
    }
    const uint32_t status = laneMatchesReference(batch, lane, rows, policies[lane], setting);
    if (status == SPLASH_RELAX_STATUS_SLACK) ++outcome.slack;
    if (status == SPLASH_RELAX_STATUS_BOUND) ++outcome.bound;
    if (status == SPLASH_RELAX_STATUS_CAPPED) ++outcome.capped;
    for (uint32_t position = rows; position < kPositions; ++position)
      require(batch.rest(lane, position) == 0.0F, "an untouched row has a rest ratio");
  }
}

void solves(MetalBackend &backend) {
  Outcome outcome;
  uint64_t seed = 11;
  for (const auto &[lambda, alpha] :
       {std::pair{1.5F, 0.5F}, std::pair{1.5F, 0.0F}, std::pair{0.2F, 0.0F}, std::pair{0.6F, 1.5F},
        std::pair{3.0F, 2.5F},
        // An allowance so small that the concavity limit lies past the
        // search's reach, which then ends the search as at alpha 0.
        std::pair{1.5F, 1e-3F},
        // A drift allowance above the unconstrained solution's drift per nat
        // of divergence: the constraint is slack at price 0.
        std::pair{0.2F, 50.0F}}) {
    const RelaxationSetting setting = *RelaxationSetting::validated(lambda, alpha);
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
      const std::array<uint32_t, kLanes> whole{kRows, kRows, kRows, kRows};
      const std::array<uint32_t, kLanes> partial{3, kRows, 1, 5};
      solveMatchesReference(backend, setting, lanes, std::span(whole).first(lanes), seed++, outcome);
      solveMatchesReference(backend, setting, lanes, std::span(partial).first(lanes), seed++, outcome);
    }
  }
  std::cout << "relaxations: " << outcome.slack << " slack, " << outcome.bound << " bound, " << outcome.capped
            << " capped\n";
  require(outcome.bound > 0, "no block bound its price: the search was not exercised");
  require(outcome.slack > 0, "no block was slack");
}

// Rows a policy cuts, as production requests' are. A top-k of 20 keeps the
// draft's twelve best candidates and eight tokens it does not list, so the
// other candidates are dead slots (target probability 0, a draft
// probability above it) beside a live tail. A top-k of 8 keeps only
// candidates: the tail is fp32 rounding, at or below the floor, and the
// row's rest stays 1. A nucleus, searched for or walked within a top-k,
// and a min-p cut the same way, each along its own path of the search
// (distribution_end), whose kept mass the candidates' probabilities are
// read with.
void solvesCutRows(MetalBackend &backend) {
  Outcome outcome;
  uint64_t seed = 211;
  for (const auto &[lambda, alpha] : {std::pair{1.5F, 0.5F}, std::pair{1.5F, 0.0F}, std::pair{0.6F, 1.5F}}) {
    const RelaxationSetting setting = *RelaxationSetting::validated(lambda, alpha);
    for (const SamplingPolicy &policy :
         {cutPolicy(20, 1.0F, 0.0F), cutPolicy(8, 1.0F, 0.0F), cutPolicy(0, 0.999F, 0.0F), cutPolicy(0, 0.95F, 0.0F),
          cutPolicy(20, 0.9F, 0.0F), cutPolicy(0, 1.0F, 1e-3F)}) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        const std::array<uint32_t, kLanes> whole{kRows, kRows, kRows, kRows};
        const std::array<uint32_t, kLanes> partial{3, kRows, 1, 5};
        solveMatchesReference(backend, setting, lanes, std::span(whole).first(lanes), seed++, outcome, policy);
        solveMatchesReference(backend, setting, lanes, std::span(partial).first(lanes), seed++, outcome, policy);
      }
    }
  }
  std::cout << "relaxations of cut rows: " << outcome.slack << " slack, " << outcome.bound << " bound, "
            << outcome.capped << " capped\n";
  require(outcome.bound > 0, "no block of cut rows bound its price");
}

// A lane whose constraint no price up to the cap satisfies keeps the cap's
// solution, its residual accepted. Every row keeps two tokens, both
// candidates, on which the draft's whole mass lies and is a little more
// confident than the target (0.66 against 0.6). The candidates' log-ratios
// lie so close that no price up to the cap tilts one off its draft
// probability, and a row without a tail has no other mass to move: at every
// price the relaxed row is the draft's, whose drift is about twice its
// divergence, above the allowance of half of it.
void cappedLaneKeepsTheCapSolution(MetalBackend &backend) {
  Random random(67);
  const RelaxationSetting setting = *RelaxationSetting::validated(1.5F, 0.5F);
  const Sampling sampling(kVocabulary, setting);
  Batch batch = makeBatch(backend, 1);
  const std::array<SamplingPolicy, 1> policies{sampledPolicy(0.9F, 2)};
  constexpr uint32_t kFirst = 10, kSecond = 11;
  for (uint32_t index = 0; index < kRows; ++index) {
    float *row = batch.row(0, index);
    std::fill_n(row, kVocabulary, -30.0F);
    // The two kept tokens at 0.6 and 0.4.
    row[kFirst] = 0.0F;
    row[kSecond] = -policies[0].temperature * std::log(1.5F);
  }
  batch.inputTokens()[0] = 7;
  for (uint32_t position = 0; position < kPositions; ++position) {
    uint32_t *candidates = batch.candidates(0, position);
    float *probabilities = batch.probabilities(0, position);
    for (uint32_t k = 0; k < kRelaxCandidates; ++k) {
      candidates[k] = k == 0 ? kFirst : k == 1 ? kSecond : 100 + k;
      probabilities[k] = k == 0 ? 0.66F : k == 1 ? 0.34F : 0.0F;
    }
    batch.inputTokens()[position + 1] = kFirst;
  }
  for (uint32_t uniform = 0; uniform < kUniforms; ++uniform)
    batch.uniforms()[uniform] = static_cast<float>(random.fraction());
  batch.poison();
  runVerify(backend, sampling, batch, policies);
  require(batch.status(0) == SPLASH_RELAX_STATUS_CAPPED, "a lane without a feasible price up to the cap is not capped");
  require(laneMatchesReference(batch, 0, kPositions, policies[0], setting) == SPLASH_RELAX_STATUS_CAPPED,
          "the reference does not cap the lane");
  for (uint32_t position = 0; position < kPositions; ++position) {
    require(batch.rest(0, position) == 1.0F, "a row without a tail has a rest ratio other than 1");
    require(std::fabs(batch.relaxed(0, position)[0] - 0.66F) <= 4e-3F &&
                std::fabs(batch.relaxed(0, position)[1] - 0.34F) <= 4e-3F,
            "a capped lane's row is not the cap's solution");
  }
}

// A lane's block stops before its thinking cutoff: the rows from the cutoff
// on take the end of thinking whatever their distributions, so they stay
// untouched and the rows before them are relaxed as a block of their own.
void thinkingCutoffEndsTheBlock(MetalBackend &backend) {
  Random random(71);
  const RelaxationSetting setting = *RelaxationSetting::validated(1.5F, 0.5F);
  const Sampling sampling(kVocabulary, setting);
  Batch batch = makeBatch(backend, 2);
  // No spike and no draft token is token 0.
  static constexpr std::array<uint32_t, 1> kEnd{0};
  constexpr uint32_t kCutoffRow = 3;
  std::array<SamplingPolicy, 2> policies{sampledPolicy(), sampledPolicy()};
  policies[0].thinkingCutoff = ThinkingCutoff{.row = kCutoffRow, .end = kEnd};
  fillBatch(batch, policies, random);
  batch.poison();
  runVerify(backend, sampling, batch, policies);
  static_cast<void>(laneMatchesReference(batch, 0, kCutoffRow, policies[0], setting));
  static_cast<void>(laneMatchesReference(batch, 1, kPositions, policies[1], setting));
  for (uint32_t position = kCutoffRow; position < kPositions; ++position)
    require(batch.rest(0, position) == 0.0F, "a row from the thinking cutoff on was relaxed");
  require(batch.outputTokens()[kCutoffRow] == kEnd[0], "the thinking cutoff's row did not take the end");
}

// A row without a finite kept mass (every logit NaN) has no distribution to
// relax: it stays untouched, and its acceptance of 0 ends the block, so the
// rows before it are relaxed as a block of their own and every other row's
// relaxation stays finite.
void nonFiniteRowEndsTheBlock(MetalBackend &backend) {
  Random random(73);
  const RelaxationSetting setting = *RelaxationSetting::validated(1.5F, 0.5F);
  const Sampling sampling(kVocabulary, setting);
  Batch batch = makeBatch(backend, 1);
  const std::array<SamplingPolicy, 1> policies{sampledPolicy()};
  fillBatch(batch, policies, random);
  constexpr uint32_t kBad = 2;
  std::fill_n(batch.row(0, kBad), kVocabulary, std::numeric_limits<float>::quiet_NaN());
  batch.poison();
  runVerify(backend, sampling, batch, policies);
  static_cast<void>(laneMatchesReference(batch, 0, kBad, policies[0], setting));
  require(batch.rest(0, kBad) == 0.0F, "a row without a finite kept mass was relaxed");
  for (uint32_t position = 0; position < kPositions; ++position) {
    require(std::isfinite(batch.rest(0, position)), "a non-finite row left another row's rest non-finite");
    for (uint32_t k = 0; k < kRelaxCandidates; ++k)
      require(std::isfinite(batch.relaxed(0, position)[k]) &&
                  (position != kBad || batch.relaxed(0, position)[k] == 0.0F),
              "a non-finite row left a relaxed probability non-finite");
  }
}

// The draw of a relaxed row: its draft token's recorded probability is the
// relaxed one, and its correction falls where the residual of the relaxed
// row, in token order, places the uniform. The relaxation is deterministic,
// so a second run with uniforms placed away from the draws' boundaries
// relaxes the same rows.
void drawsFollowRelaxedRows(MetalBackend &backend) {
  Random random(23);
  const RelaxationSetting setting = *RelaxationSetting::validated(1.5F, 0.5F);
  const Sampling sampling(kVocabulary, setting);
  Batch batch = makeBatch(backend, 1);
  const std::array<SamplingPolicy, 1> policies{sampledPolicy()};
  fillBatch(batch, policies, random);
  runVerify(backend, sampling, batch, policies);
  require(batch.status(0) != SPLASH_RELAX_STATUS_NONE,
          "the draw's batch was not relaxed");
  // The residual draw weights of each relaxed row, in token order, and a
  // uniform inside a token's span with a margin on both sides.
  std::vector<std::vector<double>> weights(kPositions);
  std::vector<uint32_t> expected(kPositions, 0xffffffffu);
  for (uint32_t position = 0; position < kPositions; ++position) {
    const float rest = batch.rest(0, position);
    if (!(rest > 0.0F)) continue;
    const RowWeights row = rowWeights(batch.row(0, position), policies[0]);
    const uint32_t *candidates = batch.candidates(0, position);
    const float *q = batch.probabilities(0, position);
    const float *relaxed = batch.relaxed(0, position);
    std::vector<double> &draw = weights[position];
    draw.resize(kVocabulary);
    for (uint32_t token = 0; token < kVocabulary; ++token) draw[token] = row.weights[token] * double(rest);
    for (uint32_t k = 0; k < kRelaxCandidates; ++k)
      draw[candidates[k]] = std::max(double(relaxed[k]) * row.mass - double(q[k]) * row.mass, 0.0);
    const double total = std::accumulate(draw.begin(), draw.end(), 0.0);
    // A token whose span is wide enough to place the uniform inside with a
    // margin float rounding cannot cross.
    std::vector<uint32_t> wide;
    for (uint32_t token = 0; token < kVocabulary; ++token)
      if (draw[token] > 1e-3 * total) wide.push_back(token);
    require(!wide.empty(), "no token of a relaxed row holds a drawable span");
    const uint32_t token = wide[random.next() % wide.size()];
    double before = 0.0;
    for (uint32_t other = 0; other < token; ++other) before += draw[other];
    batch.uniforms()[SPLASH_UNIFORM_CORRECTION] = static_cast<float>((before + 0.5 * draw[token]) / total);
    expected[position] = token;
    // Every row of the lane draws with its correction uniform: one row's
    // choice at a time.
    batch.poison();
    runVerify(backend, sampling, batch, policies);
    require(batch.rest(0, position) == rest, "the relaxation changed between runs");
    require(batch.outputTokens()[position] == expected[position],
            "a relaxed row's correction was not drawn from its residual (row " + std::to_string(position) +
                ": " + std::to_string(batch.outputTokens()[position]) + " vs " + std::to_string(expected[position]) +
                ")");
    const uint32_t draft = batch.inputTokens()[position + 1];
    double relaxedDraft = double(rest) * row.probability(draft);
    for (uint32_t k = 0; k < kRelaxCandidates; ++k)
      if (candidates[k] == draft) relaxedDraft = relaxed[k];
    require(std::fabs(double(batch.record(0, position).draft_probability) - relaxedDraft) <= 1e-5,
            "a relaxed row did not record its draft token's relaxed probability");
  }
}

// Rows past a lane's verify rows, every row of a greedy lane and the bonus
// row stay untouched: their rest is 0 (the greedy lane's status none) and
// their selections are bitwise those of a verify without the relaxation.
void untouchedRows(MetalBackend &backend) {
  Random random(37);
  const RelaxationSetting setting = *RelaxationSetting::validated(1.5F, 0.5F);
  const Sampling relaxing(kVocabulary, setting), plain(kVocabulary);
  Batch batch = makeBatch(backend, 3);
  const std::array<SamplingPolicy, 3> policies{sampledPolicy(), SamplingPolicy{}, sampledPolicy(1.2F)};
  fillBatch(batch, policies, random);
  batch.verifyRows()[0] = 3;
  batch.verifyRows()[2] = kRows;
  batch.poison();
  runVerify(backend, plain, batch, policies);
  std::vector<uint32_t> plainTokens(batch.outputTokens(), batch.outputTokens() + 3 * kRows);
  std::vector<TargetVocabularyRow> plainRecords(3 * kRows);
  for (uint32_t index = 0; index < 3 * kRows; ++index) plainRecords[index] = batch.record(index / kRows, index % kRows);
  batch.poison();
  runVerify(backend, relaxing, batch, policies);
  require(batch.status(1) == SPLASH_RELAX_STATUS_NONE, "a greedy lane has a relaxation status");
  require(batch.status(0) != SPLASH_RELAX_STATUS_NONE && batch.status(2) != SPLASH_RELAX_STATUS_NONE,
          "a sampled lane has no relaxation status");
  for (uint32_t position = 0; position < kPositions; ++position) {
    require(batch.rest(1, position) == 0.0F, "a greedy lane's row was relaxed");
    require((batch.rest(0, position) > 0.0F) == (position < 2),
            "the rows a lane relaxes are not its verify rows less the anchor");
  }
  for (uint32_t lane = 0; lane < 3; ++lane) {
    for (uint32_t index = 0; index < kRows; ++index) {
      // The bonus row, a greedy lane's rows and every drafted row without
      // a rest ratio.
      const bool untouched = lane == 1 || index == kPositions || !(batch.rest(lane, index) > 0.0F);
      if (!untouched) continue;
      const uint32_t s = lane * kRows + index;
      require(batch.outputTokens()[s] == plainTokens[s], "an untouched row selected another token than without the relaxation");
      if (policies[lane].samples())
        require(batch.record(lane, index).draft_probability == plainRecords[s].draft_probability,
                "an untouched row recorded another draft probability than without the relaxation");
    }
  }
}

// At an enormous price every row stays within rounding of the target.
void vanishesAtLargePrice(MetalBackend &backend) {
  Random random(41);
  const RelaxationSetting setting = *RelaxationSetting::validated(1e6F, 0.5F);
  const Sampling sampling(kVocabulary, setting);
  Batch batch = makeBatch(backend, 2);
  const std::array<SamplingPolicy, 2> policies{sampledPolicy(), sampledPolicy(0.7F)};
  fillBatch(batch, policies, random);
  runVerify(backend, sampling, batch, policies);
  for (uint32_t lane = 0; lane < 2; ++lane) {
    require(batch.status(lane) != SPLASH_RELAX_STATUS_NONE,
            "an enormous price left a lane unrelaxed");
    const std::vector<RelaxReferenceRow> inputs = referenceRows(batch, lane, kPositions, policies[lane]);
    for (uint32_t position = 0; position < kPositions; ++position) {
      require(std::fabs(double(batch.rest(lane, position)) - 1.0) <= 1e-3, "an enormous price moved a row's rest");
      for (uint32_t k = 0; k < kRelaxCandidates; ++k)
        require(std::fabs(double(batch.relaxed(lane, position)[k]) - inputs[position].p[k]) <= 1e-3,
                "an enormous price moved a candidate's probability");
    }
  }
}

// The search records the mass each row's policy keeps: the whole row's
// without a cut, a top-k's, a nucleus's (searched for, and walked within a
// top-k) and a min-p's.
void keptMassRecorded(MetalBackend &backend) {
  Random random(53);
  const Sampling sampling(kVocabulary);
  uint32_t policyIndex = 0;
  for (const SamplingPolicy &policy : {sampledPolicy(), cutPolicy(20, 1.0F, 0.0F), cutPolicy(0, 0.95F, 0.0F),
                                       cutPolicy(20, 0.9F, 0.0F), cutPolicy(0, 1.0F, 0.05F)}) {
    Batch batch = makeBatch(backend, 1);
    const std::array<SamplingPolicy, 1> policies{policy};
    fillBatch(batch, policies, random);
    runVerify(backend, sampling, batch, policies);
    for (uint32_t index = 0; index < kRows; ++index)
      require(std::fabs(double(batch.record(0, index).kept_mass) / rowWeights(batch.row(0, index), policy).mass -
                        1.0) <= 1e-5,
              "the search did not record the mass a policy keeps (policy " + std::to_string(policyIndex) + ")");
    ++policyIndex;
  }
}

// Each buffer the relaxation reaches, at its extent and one element short,
// and the settings the operator refuses.
void extentsAndSettings(MetalBackend &backend) {
  const RelaxationSetting setting = *RelaxationSetting::validated(1.5F, 0.5F);
  const Sampling sampling(kVocabulary, setting);
  const Batch batch = makeBatch(backend, 2);
  const std::array<SamplingPolicy, 2> policies{sampledPolicy(), sampledPolicy()};
  const auto encode = [&](const SamplingBuffers &buffers) {
    CommandGraph graph;
    sampling.addVerify(graph, policies, buffers, kStopTokens, PenaltyTable{});
    return graph;
  };
  const auto check = [&](MetalBuffer SamplingBuffers::*member, uint64_t bytes, const char *what) {
    requireExtent(backend, batch.buffers.*member, bytes, 4, what,
                  [&](CommandGraph &graph, const MetalBuffer &view) {
                    SamplingBuffers changed = batch.buffers;
                    changed.*member = view;
                    graph = encode(changed);
                  });
  };
  check(&SamplingBuffers::verifyRows, 2 * sizeof(uint32_t), "verify rows");
  check(&SamplingBuffers::relaxedProbabilities, uint64_t{2} * kPositions * kRelaxCandidates * sizeof(float),
        "relaxed probability");
  check(&SamplingBuffers::relaxedRest, uint64_t{2} * kPositions * sizeof(float), "relaxed rest");
  check(&SamplingBuffers::relaxedStatus, 2 * sizeof(uint32_t), "relaxation status");
  // Without a relaxation none is read.
  const Sampling plain(kVocabulary);
  SamplingBuffers none = batch.buffers;
  none.verifyRows = none.relaxedProbabilities = none.relaxedRest = none.relaxedStatus = MetalBuffer{};
  CommandGraph graph;
  plain.addVerify(graph, policies, none, kStopTokens, PenaltyTable{});
  require(!graph.empty(), "a verify without a relaxation encoded nothing");
  require(!RelaxationSetting::validated(0.0F, 0.5F) && !RelaxationSetting::validated(-1.0F, 0.0F) &&
              !RelaxationSetting::validated(1.5F, -0.5F) && !RelaxationSetting::validated(NAN, 0.0F) &&
              !RelaxationSetting::validated(1.5F, INFINITY) && RelaxationSetting::validated(0.2F, 0.0F),
          "the settings' validation is wrong");
  RelaxationSetting invalid;
  invalid.lambda = 0.0F;
  rejects([&] { Sampling other(kVocabulary, invalid); }, "invalid relaxation setting",
          "a relaxation of price 0 was accepted");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: rflash METALLIB");
    MetalBackend backend(argv[1]);
    solves(backend);
    solvesCutRows(backend);
    cappedLaneKeepsTheCapSolution(backend);
    thinkingCutoffEndsTheBlock(backend);
    nonFiniteRowEndsTheBlock(backend);
    drawsFollowRelaxedRows(backend);
    untouchedRows(backend);
    vanishesAtLargePrice(backend);
    keptMassRecorded(backend);
    extentsAndSettings(backend);
    std::cout << "rflash: ok\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "rflash: " << error.what() << "\n";
    return 1;
  }
}
