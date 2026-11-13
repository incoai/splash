// The prompt-lookup (copyspec) cycle end to end through the decode pipeline's
// own GPU ops: the draft's selection (ops::DraftSelector), the copied block a
// lookup hit writes over the draft's in the cycle's command
// (DraftSelector::addFixedProposals), the verify rows those confidences choose,
// the target's policy and DFlash acceptance (ops::Sampling), and the
// finalizeDecode invariants on the way out. The two neural forwards are
// scripted as logits, exactly as the acceptance path's test scripts them; the
// copied block itself is a REAL lookup hit (model::CopySpecIndex::draftAfter)
// over a request's own earlier tokens.
//
// A mixed batch of a greedy lane whose lookup hits, a sampled lane
// (temperature > 0) whose lookup would also hit, and a constrained greedy lane
// (token-mask constraint mode) whose lookup hits: the sampled lane gets no
// copied proposals (planCopyDraft's rule, whose gate is the production
// SamplingPolicy predicate), and its proposals and confidence stay exactly
// what the draft's selector wrote; the greedy lane's copied block goes through
// verify and, where the target agrees, is accepted token for token; the
// constrained lane's copied proposals still go through grammar masking: its
// target row selects among the tokens its mask admits, so a copied proposal
// its grammar refuses is never accepted and every retained token is admitted.
// The lookup lane's confidence at its copied rows is the one the verify rows'
// model reads as the request's measured copy acceptance
// (DraftSelector::confidenceFor, CopySpecAcceptance), and past them the
// draft's stays: under a cost model that pays for every row, a copy its
// request keeps accepting verifies its copied rows plus their correction,
// and one it keeps refusing verifies only its anchor. A lookup miss keeps the draft's
// block for free, and a block capped below the block width keeps its copied
// prefix with the correction's row at the end.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "model/CopySpec.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
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
using splash::model::CopySpecAcceptance;
using splash::model::CopySpecIndex;
using namespace splash::ops;
using splash::test::require;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kRank = SPLASH_DRAFT_SELECTOR_RANK;
constexpr uint32_t kVocabulary = 1003;
constexpr uint32_t kMaskWords = (kVocabulary + 31) / 32;
// Both stop tokens sit below every token the scenarios use.
constexpr StopTokens kStopTokens{1, 2};

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer = backend.allocateBuffer(
      bytes, BufferStorage::Shared, "copyspec integration");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
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

MetalBuffer randomBfloat(MetalBackend &backend, uint64_t count, Random &random,
                         float scale) {
  MetalBuffer buffer = allocate(backend, count * sizeof(uint16_t));
  auto *words = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index) {
    const float value = scale * random.unit();
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits >> 31;
    const uint32_t exponent = (bits >> 23) & 0xFF;
    const uint32_t mantissa = bits & 0x7FFFFF;
    if (exponent == 0xFF || exponent > 127 + 15) {
      words[index] = static_cast<uint16_t>((sign << 15) | 0x7C00);
    } else if (exponent < 127 - 14) {
      words[index] = static_cast<uint16_t>(sign << 15);
    } else {
      words[index] =
          static_cast<uint16_t>((sign << 15) | ((exponent - 127 + 15) << 10) |
                               (mantissa >> 13));
    }
  }
  return buffer;
}

// One decode cycle's buffers: the draft selector's inputs and outputs, the
// target's verify rows and policy workspace, and the acceptance counts, all
// over the shared tensors the arena holds one of (ProposedTokens, Candidates,
// ProposalProbs, TargetVocabularyRows, SamplingUniforms, OutputTokens).
struct Batch final {
  DraftSelectorBuffers selector;
  SamplingBuffers sampling;
  AcceptanceBuffers acceptance;
  DraftCodebooks codebooks;
  // The selector's confidence output (DraftSelectorBuffers::confidence): a
  // copied block's host step overwrites it beside the proposals.
  MetalBuffer confidence;

  [[nodiscard]] uint32_t *proposed() const {
    return static_cast<uint32_t *>(selector.proposedTokens.contents());
  }
  [[nodiscard]] float *confidences() const {
    return static_cast<float *>(confidence.contents());
  }
  [[nodiscard]] uint32_t *inputs() const {
    return static_cast<uint32_t *>(sampling.inputTokens.contents());
  }
  [[nodiscard]] float *verifyLogits(uint32_t row) const {
    return static_cast<float *>(sampling.logits.contents()) +
           uint64_t{row} * kVocabulary;
  }
  [[nodiscard]] float *draftLogits(uint32_t row) const {
    return static_cast<float *>(selector.logits.contents()) +
           uint64_t{row} * kVocabulary;
  }
  [[nodiscard]] uint32_t *masks() const {
    return static_cast<uint32_t *>(sampling.constraintMasks.contents());
  }
  [[nodiscard]] const uint32_t *outputs() const {
    return static_cast<const uint32_t *>(sampling.outputTokens.contents());
  }
  [[nodiscard]] const uint32_t *retainedCounts() const {
    return static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
  }
  [[nodiscard]] const uint32_t *acceptedCounts() const {
    return static_cast<const uint32_t *>(acceptance.acceptedCounts.contents());
  }
  [[nodiscard]] const uint32_t *verifyRows() const {
    return static_cast<const uint32_t *>(acceptance.verifyRows.contents());
  }
};

Batch makeBatch(MetalBackend &backend, uint32_t lanes, Random &random) {
  const uint32_t rows = lanes * kRows;
  const uint32_t positions = lanes * kPositions;
  const auto sampling = Sampling::workspace(rows);
  const auto proposals = DraftSelector::workspace(positions);
  Batch batch;
  batch.confidence = allocate(backend, uint64_t{positions} * sizeof(float));
  batch.selector = DraftSelectorBuffers{
      allocate(backend, uint64_t{rows} * kVocabulary * sizeof(float)),
      allocate(backend, proposals.partialIdsBytes),
      allocate(backend, proposals.partialValuesBytes),
      allocate(backend, proposals.candidatesBytes),
      allocate(backend, proposals.unaryBytes),
      randomBfloat(backend, uint64_t{rows} * kRank, random, 0.1F),
      allocate(backend,
               uint64_t{lanes} * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, uint64_t{positions} * sizeof(uint32_t)),
      allocate(backend, proposals.proposalProbabilitiesBytes),
      batch.confidence};
  batch.sampling = SamplingBuffers{
      allocate(backend, uint64_t{rows} * kVocabulary * sizeof(float)),
      allocate(backend, sampling.partialMassesBytes),
      allocate(backend, sampling.vocabularyRowsBytes),
      batch.selector.uniforms,
      allocate(backend,
               uint64_t{lanes} * (kRows + 1) * kMaskWords * sizeof(uint32_t)),
      allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
      allocate(backend, sampling.argmaxValuesBytes),
      allocate(backend, sampling.argmaxIndicesBytes),
      allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
      batch.selector.candidates,
      batch.selector.proposalProbabilities,
      allocate(backend, sampling.vocabularyRangesBytes),
      allocate(backend, sampling.vocabularyArrivalsBytes)};
  batch.acceptance = AcceptanceBuffers{
      batch.selector.proposedTokens,
      batch.selector.candidates,
      batch.selector.proposalProbabilities,
      batch.sampling.vocabularyRows,
      batch.selector.uniforms,
      batch.sampling.outputTokens,
      allocate(backend, uint64_t{lanes} * sizeof(uint32_t)),
      allocate(backend, uint64_t{lanes} * sizeof(uint32_t)),
      allocate(backend, uint64_t{lanes} * sizeof(uint32_t))};
  batch.codebooks = DraftCodebooks{
      randomBfloat(backend, uint64_t{kVocabulary} * kRank, random, 0.1F),
      randomBfloat(backend, uint64_t{kVocabulary} * kRank, random, 0.1F)};
  // The uniforms are interior, away from a draw or an acceptance boundary.
  auto *uniforms = static_cast<float *>(batch.selector.uniforms.contents());
  for (uint32_t index = 0; index < lanes * SPLASH_SAMPLING_UNIFORMS; ++index)
    uniforms[index] = 0.25F + 0.01F * float(index % 50);
  // An unconstrained lane never reads its masks, so poison them: the copied
  // lanes' exact outputs below would change if one did.
  std::memset(batch.masks(), 0xA5,
              uint64_t{lanes} * (kRows + 1) * kMaskWords * sizeof(uint32_t));
  return batch;
}

// Low noise with a few spikes above every other token: the stop tokens stay
// below every spike, so no lane's draw or argmax can reach one.
void fillRow(float *row, Random &random, const std::vector<uint32_t> &spikes) {
  for (uint32_t token = 0; token < kVocabulary; ++token)
    row[token] = -8.0F + 1.5F * random.unit() * random.unit();
  for (uint32_t index = 0; index < spikes.size(); ++index)
    row[spikes[index]] = 4.0F - 0.25F * float(index);
}

// One lane's request history and its pending anchor, its policy and its
// retention limit, and the copied block its lookup finds (planCopyDraft).
struct Lane final {
  std::vector<uint32_t> history;
  uint32_t anchor = 0;
  SamplingPolicy policy;
  uint32_t maximumRetained = kRows;
  std::optional<CopySpecIndex> index;
  std::array<uint32_t, kPositions> copied{};
  uint32_t count = 0;
};

// The copied block a greedy lane's cycle drafts where the request copies and
// its lookup hits (Runtime.cpp planCopyDraft): at most min(7,
// maximumRetained - 1) proposals, full length or none. A sampled lane draws
// its proposals and takes no copies whatever its lookup finds.
uint32_t planCopyDraft(const Lane &lane,
                       std::array<uint32_t, kPositions> &copy) {
  copy = {};
  if (lane.policy.samples() || lane.policy.thinkingCutoff ||
      lane.maximumRetained <= 1)
    return 0;
  return lane.index->draftAfter(
      lane.anchor, std::min(kPositions, lane.maximumRetained - 1),
      kStopTokens.tokens(), copy);
}

// One decode cycle of the batch through the pipeline's own ops: the draft's
// selection for every lane, the fixed-proposal writer that puts the copied
// block of the lanes whose lookup hit over the draft's (Runtime.cpp
// copiedProposals: the copied rows' proposals, at the confidence `model`
// reads as a fresh request's copy acceptance), the verify rows those
// confidences choose, and the target's policy and acceptance. The
// verify input rows are the anchor and the proposals, as the verify-input
// dispatch writes them. Returns the lanes with their copied counts.
std::vector<Lane> runCycle(MetalBackend &backend, Batch &batch,
                           std::vector<Lane> lanes,
                           VerifyRowsModel model = {}) {
  const uint32_t width = static_cast<uint32_t>(lanes.size());
  const Sampling sampling(kVocabulary);
  std::vector<SamplingPolicy> policies;
  std::vector<uint32_t> anchors;
  for (const Lane &lane : lanes) {
    policies.push_back(lane.policy);
    anchors.push_back(lane.anchor);
  }

  CommandGraph draft;
  const DraftSelector selector(kVocabulary);
  selector.add(draft, batch.selector, batch.codebooks, anchors, policies);
  static_cast<void>(backend.submitCommandAsync(draft.dispatches()).wait());

  // The selector's own block, which lanes without a copied block, and the
  // rows past a copied block, must keep exactly.
  const std::vector<uint32_t> selectorProposals(
      batch.proposed(), batch.proposed() + width * kPositions);
  const std::vector<float> selectorConfidence(
      batch.confidences(), batch.confidences() + width * kPositions);
  std::vector<FixedProposals> fixed(width);
  for (uint32_t lane = 0; lane < width; ++lane) {
    lanes[lane].count = planCopyDraft(lanes[lane], lanes[lane].copied);
    fixed[lane].count = lanes[lane].count;
    for (uint32_t position = 0; position < lanes[lane].count; ++position) {
      fixed[lane].tokens[position] = lanes[lane].copied[position];
      fixed[lane].confidence[position] =
          DraftSelector::confidenceFor(model, position, CopySpecAcceptance{}.rate());
    }
  }
  CommandGraph copy;
  DraftSelector::addFixedProposals(copy, fixed, batch.selector.proposedTokens, batch.confidence);
  static_cast<void>(backend.submitCommandAsync(copy.dispatches()).wait());
  for (uint32_t lane = 0; lane < width; ++lane)
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t index = lane * kPositions + position;
      if (position < lanes[lane].count) {
        require(batch.proposed()[index] == lanes[lane].copied[position],
                "a copied proposal did not overwrite the draft's");
        require(batch.confidences()[index] == fixed[lane].confidence[position],
                "a copied confidence did not overwrite the draft's");
      } else {
        require(batch.proposed()[index] == selectorProposals[index],
                lanes[lane].count ? "a copy changed a proposal past its copied rows"
                                  : "a lane without a copied block lost the draft's proposals");
        require(batch.confidences()[index] == selectorConfidence[index],
                lanes[lane].count ? "a copy changed a confidence past its copied rows"
                                  : "a lane without a copied block lost the draft's confidence");
      }
    }

  for (uint32_t lane = 0; lane < width; ++lane) {
    uint32_t *inputs = batch.inputs() + uint64_t{lane} * kRows;
    inputs[0] = lanes[lane].anchor;
    std::copy_n(batch.proposed() + uint64_t{lane} * kPositions, kPositions,
                inputs + 1);
  }
  std::vector<uint32_t> maximumRetained;
  for (const Lane &lane : lanes)
    maximumRetained.push_back(lane.maximumRetained);

  CommandGraph verify;
  DraftSelector::addVerifyRows(verify, batch.confidence,
                               batch.acceptance.verifyRows, model,
                               maximumRetained);
  sampling.addVerify(verify, policies, batch.sampling, kStopTokens,
                     PenaltyTable{});
  sampling.addAcceptance(verify, batch.acceptance, maximumRetained, policies,
                         kStopTokens);
  static_cast<void>(backend.submitCommandAsync(verify.dispatches()).wait());
  return lanes;
}

// The invariants finalizeDecode hard-errors on (Runtime.cpp), and the exact
// retention of a block without stop tokens: accepted proposals plus the
// correction, bounded by what the lane verified.
void requireInvariants(const Batch &batch, const std::vector<Lane> &lanes,
                       const std::string &label) {
  for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
    const uint32_t maximum = lanes[lane].maximumRetained;
    const uint32_t verified = batch.verifyRows()[lane];
    const uint32_t retained = batch.retainedCounts()[lane];
    const uint32_t accepted = batch.acceptedCounts()[lane];
    const std::string at = label + " lane " + std::to_string(lane);
    require(verified >= 1 && verified <= maximum,
            at + ": verify rows " + std::to_string(verified) +
                " are outside [1, " + std::to_string(maximum) + "]");
    require(retained >= 1 && retained <= verified,
            at + ": retained " + std::to_string(retained) +
                " are outside [1, " + std::to_string(verified) + "]");
    require(accepted <= kPositions,
            at + ": accepted " + std::to_string(accepted) +
                " of more than the draft proposed");
    const uint32_t expected = std::min(accepted + 1, std::min(maximum, verified));
    require(retained == expected,
            at + ": retained " + std::to_string(retained) + ", not " +
                std::to_string(expected));
  }
}

void requireToken(const Batch &batch, uint32_t lane, uint32_t position,
                  uint32_t expected, const std::string &label) {
  require(batch.outputs()[uint64_t{lane} * kRows + position] == expected,
          label + ": output row " + std::to_string(position) + " of lane " +
              std::to_string(lane) + " is not " + std::to_string(expected));
}

// The request whose earlier tokens repeat: `prefix` appears twice, and what
// followed its first appearance is the block a lookup at the second copies.
std::vector<uint32_t> repeatedHistory(std::span<const uint32_t> prefix,
                                      std::span<const uint32_t> continuation) {
  std::vector<uint32_t> history(prefix.begin(), prefix.end());
  history.insert(history.end(), continuation.begin(), continuation.end());
  history.insert(history.end(), prefix.begin(), prefix.end() - 1);
  return history;
}

// The draft's query rows of one lane: low noise with sixteen candidates a
// proposal position may keep, all away from every other scenario's tokens.
void fillDraftRows(Batch &batch, uint32_t lane, Random &random) {
  for (uint32_t row = 0; row < kRows; ++row) {
    std::vector<uint32_t> spikes;
    for (uint32_t token = 0; token < 16; ++token)
      spikes.push_back(600 + 16 * row + token);
    fillRow(batch.draftLogits(lane * kRows + row), random, spikes);
  }
}

// The mixed batch: a greedy lane whose lookup hits beside a sampled lane whose
// lookup would also hit and a constrained greedy lane whose lookup hits. The
// sampled lane takes no copied proposals and keeps the draft's own block; the
// greedy lane's copied block verifies and, where the target agrees, is
// accepted token for token with the correction's row at the end; the
// constrained lane's copied proposals go through its grammar mask, so the one
// its grammar refuses is not accepted and every retained token is admitted.
void mixedBatch(MetalBackend &backend, Random &random) {
  Batch batch = makeBatch(backend, 3, random);
  std::vector<Lane> lanes(3);

  // Lane 0: a greedy lane whose lookup copies four tokens (its retention
  // leaves min(7, 5 - 1) proposals), and whose target agrees with the whole
  // copied block.
  const std::array<uint32_t, 6> prefix0{100, 101, 102, 103, 104, 105};
  const std::array<uint32_t, 7> continuation0{200, 201, 202, 203,
                                              204, 205, 206};
  lanes[0].history = repeatedHistory(prefix0, continuation0);
  lanes[0].anchor = 105;
  lanes[0].maximumRetained = 5;

  // Lane 1: a sampled lane whose lookup would also hit; planCopyDraft's rule
  // gives it no copied block (SamplingPolicy::samples is its gate).
  const std::array<uint32_t, 6> prefix1{300, 301, 302, 303, 304, 305};
  const std::array<uint32_t, 7> continuation1{310, 311, 312, 313,
                                              314, 315, 316};
  lanes[1].history = repeatedHistory(prefix1, continuation1);
  lanes[1].anchor = 305;
  lanes[1].policy = SamplingPolicy{16, 0.8F, 1.0F, false, true};

  // Lane 2: a constrained greedy lane whose lookup copies all seven tokens,
  // one of which its grammar refuses at row 3.
  const std::array<uint32_t, 6> prefix2{500, 501, 502, 503, 504, 505};
  const std::array<uint32_t, 7> continuation2{510, 511, 512, 513,
                                              514, 515, 516};
  lanes[2].history = repeatedHistory(prefix2, continuation2);
  lanes[2].anchor = 505;
  lanes[2].policy = SamplingPolicy{1, 0.0F, 1.0F, true};

  for (Lane &lane : lanes) {
    lane.index.emplace(lane.history);
    require(lane.index->enabled(), "a scenario history did not index");
  }
  // The sampled lane's lookup would hit, and it still takes no copies.
  std::array<uint32_t, kPositions> wouldCopy{};
  require(lanes[1].index->draftAfter(lanes[1].anchor, kPositions,
                                    kStopTokens.tokens(), wouldCopy) ==
              kPositions,
          "the sampled lane's lookup did not find its block");
  require(lanes[1].policy.samples(), "the sampled lane is not a sampling lane");

  for (uint32_t lane = 0; lane < 3; ++lane)
    fillDraftRows(batch, lane, random);

  // The target: lane 0's rows select its copied block, then the correction.
  fillRow(batch.verifyLogits(0 * kRows + 0), random, {200});
  fillRow(batch.verifyLogits(0 * kRows + 1), random, {201});
  fillRow(batch.verifyLogits(0 * kRows + 2), random, {202});
  fillRow(batch.verifyLogits(0 * kRows + 3), random, {203});
  fillRow(batch.verifyLogits(0 * kRows + 4), random, {21});
  for (uint32_t row = 5; row < kRows; ++row)
    fillRow(batch.verifyLogits(0 * kRows + row), random, {800 + row});

  // Lane 1's rows are its own draws' terrain.
  for (uint32_t row = 0; row < kRows; ++row) {
    std::vector<uint32_t> spikes;
    for (uint32_t token = 0; token < 16; ++token)
      spikes.push_back(940 + 16 * row + token);
    fillRow(batch.verifyLogits(1 * kRows + row), random, spikes);
  }

  // Lane 2's rows select its copied block up to the row its grammar refuses:
  // there the copied token is the row's own argmax but is not admitted, so the
  // selection takes the correction the mask admits.
  fillRow(batch.verifyLogits(2 * kRows + 0), random, {510});
  fillRow(batch.verifyLogits(2 * kRows + 1), random, {511});
  fillRow(batch.verifyLogits(2 * kRows + 2), random, {512});
  fillRow(batch.verifyLogits(2 * kRows + 3), random, {513, 42});
  for (uint32_t row = 4; row < kRows; ++row)
    fillRow(batch.verifyLogits(2 * kRows + row), random, {820 + row});

  // The grammar of the constrained lane: every row admits every token but row
  // 3, whose copied proposal it refuses. Verify row r reads mask row r + 1.
  uint32_t *laneMasks = batch.masks() + uint64_t{2} * (kRows + 1) * kMaskWords;
  for (uint32_t word = 0; word < (kRows + 1) * kMaskWords; ++word)
    laneMasks[word] = 0xFFFFFFFFU;
  laneMasks[4 * kMaskWords + 513 / 32] &= ~(1U << (513 % 32));

  std::vector<Lane> cycle = runCycle(backend, batch, std::move(lanes));
  requireInvariants(batch, cycle, "mixed batch");

  // The greedy lane's copied block was drafted, verified and accepted whole.
  const Lane &greedy = cycle[0];
  require(greedy.count == 4,
          "the greedy lane's lookup copied " + std::to_string(greedy.count) +
              " tokens, not its full block");
  for (uint32_t position = 0; position < 4; ++position) {
    require(batch.proposed()[position] == 200 + position,
            "the copied proposal row " + std::to_string(position) +
                " is not the request's own earlier token");
    require(batch.confidences()[position] == 0.75F,
            "the copied confidence row " + std::to_string(position) +
                " is not a fresh request's copy acceptance");
    requireToken(batch, 0, position, 200 + position, "the copied prefix");
  }
  require(batch.verifyRows()[0] == 5,
          "the greedy lane verified " + std::to_string(batch.verifyRows()[0]) +
              " rows, not its whole block");
  require(batch.acceptedCounts()[0] == 4,
          "the greedy lane accepted " +
              std::to_string(batch.acceptedCounts()[0]) + " proposals");
  require(batch.retainedCounts()[0] == 5,
          "the greedy lane did not retain its block and the correction");
  requireToken(batch, 0, 4, 21, "the bonus row");

  // The sampled lane took no copied proposals: its block and confidence are
  // the draft selector's own, and it never selects a stop token.
  require(cycle[1].count == 0, "a sampled lane was given a copied block");
  for (uint32_t row = 0; row < batch.retainedCounts()[1]; ++row) {
    const uint32_t token = batch.outputs()[1 * kRows + row];
    require(!kStopTokens.contains(token) && token < kVocabulary,
            "the sampled lane selected a stop token");
  }

  // The constrained lane's copied block went through its grammar: the
  // proposal its mask refuses is not accepted and no output row is a token
  // its row's mask refuses.
  const Lane &constrained = cycle[2];
  require(constrained.count == kPositions,
          "the constrained lane's lookup copied " +
              std::to_string(constrained.count) + " tokens");
  require(batch.acceptedCounts()[2] == 3,
          "the constrained lane accepted " +
              std::to_string(batch.acceptedCounts()[2]) + " proposals");
  require(batch.retainedCounts()[2] == 4,
          "the constrained lane did not retain its prefix and correction");
  for (uint32_t position = 0; position < 3; ++position)
    requireToken(batch, 2, position, 510 + position, "the copied prefix");
  requireToken(batch, 2, 3, 42, "the masked correction");
  for (uint32_t row = 0; row < batch.retainedCounts()[2]; ++row) {
    const uint32_t token = batch.outputs()[2 * kRows + row];
    const uint32_t *mask = laneMasks + (row + 1) * kMaskWords;
    require(mask[token / 32] & (1U << (token % 32)),
            "a retained token is one its grammar refuses");
  }
}

// The confidence of a copied block reads back through the verify-rows rule
// (draft_verify_rows) as its request's copy acceptance: under a cost model
// that pays a little more for every row a cycle verifies, a copy its request
// keeps accepting is worth its copied rows plus their correction, and one
// its request keeps refusing only its anchor, where a confidence of 1 would
// claim the whole block for both. The lane's limit is its copied rows plus
// one, as Runtime.cpp's min(7, maximumRetained - 1) leaves it.
void verifyRowsOfACopiedBlock(MetalBackend &backend) {
  VerifyRowsModel model;
  model.everyRow = false;
  for (uint32_t rows = 1; rows <= kRows; ++rows)
    model.costs[rows] = 1.0F + 0.01F * float(rows);
  const auto chosenRows = [&](uint32_t count, double acceptance, bool claimsAll) {
    MetalBuffer confidence = allocate(backend, kPositions * sizeof(float));
    auto *q = static_cast<float *>(confidence.contents());
    for (uint32_t position = 0; position < kPositions; ++position)
      q[position] = claimsAll ? 1.0F : DraftSelector::confidenceFor(model, position, acceptance);
    MetalBuffer verifyRows = allocate(backend, sizeof(uint32_t));
    const std::array<uint32_t, 1> maximum{count + 1};
    CommandGraph graph;
    DraftSelector::addVerifyRows(graph, confidence, verifyRows, model, maximum);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    return *static_cast<const uint32_t *>(verifyRows.contents());
  };
  for (const uint32_t count : {2U, 4U, kPositions}) {
    const uint32_t accepted = chosenRows(count, 0.95, false);
    require(accepted == count + 1,
            "a well accepted copy verified " + std::to_string(accepted) + " rows, not its " +
                std::to_string(count) + " copied rows and their correction");
    const uint32_t refused = chosenRows(count, 0.005, false);
    require(refused == 1,
            "a refused copy verified " + std::to_string(refused) + " rows, not its anchor");
    const uint32_t claimed = chosenRows(count, 0.005, true);
    require(claimed == count + 1,
            "a confidence of 1 claimed " + std::to_string(claimed) + " rows, not the whole block");
  }
}

// A lookup miss keeps the draft's own block and costs nothing, and a block
// capped below the block width keeps its copied prefix with the correction's
// row at the end (proposals = min(7, maximumRetained - 1)).
void missAndRetentionClamp(MetalBackend &backend, Random &random) {
  Batch batch = makeBatch(backend, 2, random);
  std::vector<Lane> lanes(2);

  // Lane 0: no repeated window, so no copied block: the draft's proposals and
  // confidence stand.
  for (uint32_t token = 0; token < 32; ++token)
    lanes[0].history.push_back(700 + token);
  lanes[0].anchor = 799;

  // Lane 1: a lookup hit whose retention leaves room for two proposals.
  const std::array<uint32_t, 6> prefix{800, 801, 802, 803, 804, 805};
  const std::array<uint32_t, 7> continuation{810, 811, 812, 813,
                                             814, 815, 816};
  lanes[1].history = repeatedHistory(prefix, continuation);
  lanes[1].anchor = 805;
  lanes[1].maximumRetained = 3;

  for (Lane &lane : lanes) {
    lane.index.emplace(lane.history);
    require(lane.index->enabled(), "a scenario history did not index");
  }
  require(lanes[0].index->draftAfter(lanes[0].anchor, kPositions,
                                    kStopTokens.tokens(), lanes[0].copied) == 0,
          "the miss lane's lookup found a block");

  for (uint32_t lane = 0; lane < 2; ++lane)
    fillDraftRows(batch, lane, random);
  for (uint32_t row = 0; row < kRows; ++row)
    fillRow(batch.verifyLogits(0 * kRows + row), random, {830 + row});
  fillRow(batch.verifyLogits(1 * kRows + 0), random, {810});
  fillRow(batch.verifyLogits(1 * kRows + 1), random, {811});
  fillRow(batch.verifyLogits(1 * kRows + 2), random, {43});

  std::vector<Lane> cycle = runCycle(backend, batch, std::move(lanes));
  requireInvariants(batch, cycle, "miss and clamp");

  require(cycle[0].count == 0, "a lookup miss drafted a copied block");
  const Lane &clamped = cycle[1];
  require(clamped.count == 2,
          "the capped lookup copied " + std::to_string(clamped.count) +
              " tokens, not its full block");
  require(batch.verifyRows()[1] == 3,
          "the capped lane verified " + std::to_string(batch.verifyRows()[1]) +
              " rows");
  require(batch.acceptedCounts()[1] == 2,
          "the capped lane accepted " +
              std::to_string(batch.acceptedCounts()[1]) + " proposals");
  require(batch.retainedCounts()[1] == 3,
          "the capped lane did not retain its prefix and correction");
  requireToken(batch, 1, 0, 810, "the copied prefix");
  requireToken(batch, 1, 1, 811, "the copied prefix");
  requireToken(batch, 1, 2, 43, "the correction");
}

} // namespace

int main(int argc, char **argv) {
  std::string stage = "setup";
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: copyspec-integration METALLIB");
    MetalBackend backend(argv[1]);
    Random random(0xc0f75e9);
    stage = "mixed batch";
    mixedBatch(backend, random);
    stage = "verify rows of a copied block";
    verifyRowsOfACopiedBlock(backend);
    stage = "lookup miss and retention clamp";
    missAndRetentionClamp(backend, random);
    std::cout << "copyspec_integration_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "copyspec_integration_test: FAIL (" << stage
              << "): " << error.what() << '\n';
    return 1;
  }
}
