// The acceptance path under a prompt-lookup (copyspec) block: a lane whose
// ProposedTokens were host-filled with the block the lookup copied must keep
// the copied prefix exactly as a DFlash draft's, and the finalizeDecode-style
// invariants must hold whatever the block met (Runtime.cpp hard-errors on
// verify rows outside [1, maximumRetained], on retention outside [1, verify
// rows] and on more accepted than the draft proposed). The lookup lane's
// VerifyConfidence is 1 at every position, so addVerifyRows asks for the
// whole block, clamped to the lane's retention limit: with --verify-rows all
// (the every_row model) and with flat costs it takes exactly that limit, and
// with a budget-shaped model it stays inside [1, the limit]. A target that
// agrees with the block accepts all seven proposals and retains the block
// plus the bonus row; one that diverges at a proposal keeps the accepted
// prefix and the correction, and a block capped below the retention limit
// (proposals = min(7, maximumRetained - 1)) keeps its copied prefix with the
// correction's row at the end. A block over-proposed past the retention
// limit keeps at most the verified rows.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/DraftSelector.hpp"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
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
using splash::test::require;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kVocabulary = 1003;
// Both stop tokens sit below every token the scenarios use.
constexpr StopTokens kStopTokens{1, 2};

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, "copyspec acceptance");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

struct Batch final {
  SamplingBuffers sampling;
  AcceptanceBuffers acceptance;
  MetalBuffer confidence;

  [[nodiscard]] uint32_t *proposed() const {
    return static_cast<uint32_t *>(acceptance.proposedTokens.contents());
  }
  [[nodiscard]] uint32_t *inputs() const {
    return static_cast<uint32_t *>(sampling.inputTokens.contents());
  }
  [[nodiscard]] float *logits(uint32_t row) const {
    return static_cast<float *>(sampling.logits.contents()) +
           uint64_t{row} * kVocabulary;
  }
  [[nodiscard]] float *confidences() const {
    return static_cast<float *>(confidence.contents());
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

Batch makeBatch(MetalBackend &backend, uint32_t lanes) {
  const uint32_t rows = lanes * kRows;
  const auto sampling = Sampling::workspace(rows);
  const auto proposals = DraftSelector::workspace(lanes * kPositions);
  Batch batch;
  batch.sampling = SamplingBuffers{
      allocate(backend, uint64_t{rows} * kVocabulary * sizeof(float)),
      allocate(backend, sampling.partialMassesBytes),
      allocate(backend, sampling.vocabularyRowsBytes),
      allocate(backend, uint64_t{lanes} * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, uint64_t{lanes} * (kRows + 1) * ((kVocabulary + 31) / 32) * 4),
      allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
      allocate(backend, sampling.argmaxValuesBytes),
      allocate(backend, sampling.argmaxIndicesBytes),
      allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
      allocate(backend, proposals.candidatesBytes),
      allocate(backend, proposals.proposalProbabilitiesBytes),
      allocate(backend, sampling.vocabularyRangesBytes),
      allocate(backend, sampling.vocabularyArrivalsBytes)};
  batch.acceptance = AcceptanceBuffers{
      allocate(backend, uint64_t{lanes} * kPositions * sizeof(uint32_t)),
      allocate(backend, proposals.candidatesBytes),
      allocate(backend, proposals.proposalProbabilitiesBytes),
      allocate(backend, sampling.vocabularyRowsBytes),
      allocate(backend, uint64_t{lanes} * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      batch.sampling.outputTokens,
      allocate(backend, uint64_t{lanes} * sizeof(uint32_t)),
      allocate(backend, uint64_t{lanes} * sizeof(uint32_t)),
      allocate(backend, uint64_t{lanes} * sizeof(uint32_t))};
  batch.confidence =
      allocate(backend, uint64_t{lanes} * kPositions * sizeof(float));
  return batch;
}

// One lane's lookup block and the target that verifies it: the block's
// proposals fill the lane's ProposedTokens and its verify inputs, and the
// target's rows select them up to divergesAt, `correction` there (the bonus
// token when the whole block agrees) and unrelated tokens after it.
struct Lane final {
  uint32_t anchor = 0;
  std::array<uint32_t, kPositions> proposals{};
  uint32_t divergesAt = kPositions;
  uint32_t correction = 0;
  uint32_t maximumRetained = kRows;
};

void fillLane(Batch &batch, uint32_t lane, const Lane &spec) {
  uint32_t *proposed = batch.proposed() + uint64_t{lane} * kPositions;
  uint32_t *inputs = batch.inputs() + uint64_t{lane} * kRows;
  inputs[0] = spec.anchor;
  for (uint32_t position = 0; position < kPositions; ++position) {
    proposed[position] = spec.proposals[position];
    inputs[position + 1] = spec.proposals[position];
  }
  float *confidence = batch.confidences() + uint64_t{lane} * kPositions;
  for (uint32_t position = 0; position < kPositions; ++position)
    confidence[position] = 1.0F;
  for (uint32_t row = 0; row < kRows; ++row) {
    float *logits = batch.logits(lane * kRows + row);
    for (uint32_t token = 0; token < kVocabulary; ++token)
      logits[token] = -8.0F + 0.25F * float((token + row) % 5);
    const uint32_t selected =
        row < spec.divergesAt
            ? spec.proposals[row]
            : row == spec.divergesAt ? spec.correction : 800 + row;
    logits[selected] = 4.0F;
  }
}

// One verify and acceptance of the batch's host-filled lanes, with the
// verify rows draft_verify_rows picks from the lookup's confidences.
void runAcceptance(MetalBackend &backend, Batch &batch,
                   std::span<const uint32_t> maximumRetained,
                   const VerifyRowsModel &model) {
  const uint32_t lanes =
      static_cast<uint32_t>(maximumRetained.size());
  const Sampling sampling(kVocabulary);
  const std::vector<SamplingPolicy> policies(lanes, SamplingPolicy{});
  CommandGraph graph;
  DraftSelector::addVerifyRows(graph, batch.confidence, batch.acceptance.verifyRows,
                               model, maximumRetained);
  sampling.addVerify(graph, policies, batch.sampling, kStopTokens, PenaltyTable{});
  sampling.addAcceptance(graph, batch.acceptance, maximumRetained, policies,
                         kStopTokens);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

// The invariants finalizeDecode hard-errors on (Runtime.cpp), and the exact
// retention of a block without stop tokens: accepted proposals plus the
// correction, bounded by what the lane verified.
void requireInvariants(const Batch &batch,
                       std::span<const uint32_t> maximumRetained,
                       const std::string &label) {
  const uint32_t lanes = static_cast<uint32_t>(maximumRetained.size());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t verified = batch.verifyRows()[lane];
    const uint32_t retained = batch.retainedCounts()[lane];
    const uint32_t accepted = batch.acceptedCounts()[lane];
    const std::string at = label + " lane " + std::to_string(lane);
    require(verified >= 1 && verified <= maximumRetained[lane],
            at + ": verify rows " + std::to_string(verified) +
                " are outside [1, " + std::to_string(maximumRetained[lane]) + "]");
    require(retained >= 1 && retained <= verified,
            at + ": retained " + std::to_string(retained) +
                " are outside [1, " + std::to_string(verified) + "]");
    require(accepted <= kPositions,
            at + ": accepted " + std::to_string(accepted) +
                " of more than the draft proposed");
    const uint32_t expected =
        std::min(accepted + 1, std::min(maximumRetained[lane], verified));
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

// A lookup block the target agrees with keeps every copied token: seven
// proposals accepted, the block's rows retained plus the bonus row. A lane
// beside it whose target diverges at a proposal keeps its accepted prefix
// and the correction, exactly as a DFlash draft's would.
void lookupBlockAcceptance(MetalBackend &backend) {
  Batch batch = makeBatch(backend, 2);
  Lane agreeing;
  agreeing.anchor = 501;
  agreeing.proposals = {11, 12, 13, 14, 15, 16, 17};
  agreeing.divergesAt = kPositions;
  agreeing.correction = 21;
  Lane diverging;
  diverging.anchor = 502;
  diverging.proposals = {31, 32, 33, 34, 35, 36, 37};
  diverging.divergesAt = 2;
  diverging.correction = 42;
  fillLane(batch, 0, agreeing);
  fillLane(batch, 1, diverging);
  const std::array<uint32_t, 2> maximumRetained{kRows, kRows};
  VerifyRowsModel model; // every row: --verify-rows all
  runAcceptance(backend, batch, maximumRetained, model);
  requireInvariants(batch, maximumRetained, "lookup block");
  require(batch.acceptedCounts()[0] == kPositions,
          "an agreeing lookup block did not accept all seven proposals");
  require(batch.retainedCounts()[0] == kRows,
          "an agreeing lookup block did not retain its rows");
  for (uint32_t position = 0; position < kPositions; ++position)
    requireToken(batch, 0, position, agreeing.proposals[position],
                 "the copied prefix");
  requireToken(batch, 0, kPositions, agreeing.correction, "the bonus row");
  require(batch.acceptedCounts()[1] == 2,
          "the diverging lane accepted " +
              std::to_string(batch.acceptedCounts()[1]) + " proposals");
  require(batch.retainedCounts()[1] == 3,
          "the diverging lane did not retain its prefix and correction");
  requireToken(batch, 1, 0, diverging.proposals[0], "the copied prefix");
  requireToken(batch, 1, 1, diverging.proposals[1], "the copied prefix");
  requireToken(batch, 1, 2, diverging.correction, "the correction");
}

// A block whose first proposal the target refuses: not one drafted token
// matched, which conservative mode counts as its one strike, and the lane
// still retains the correction's row.
void zeroAcceptanceKeepsTheCorrection(MetalBackend &backend) {
  Batch batch = makeBatch(backend, 1);
  Lane lane;
  lane.anchor = 503;
  lane.proposals = {11, 12, 13, 14, 15, 16, 17};
  lane.divergesAt = 0;
  lane.correction = 42;
  fillLane(batch, 0, lane);
  const std::array<uint32_t, 1> maximumRetained{kRows};
  VerifyRowsModel model;
  runAcceptance(backend, batch, maximumRetained, model);
  requireInvariants(batch, maximumRetained, "zero acceptance");
  require(batch.acceptedCounts()[0] == 0,
          "a refused block accepted a proposal");
  require(batch.retainedCounts()[0] == 1,
          "a refused block did not retain exactly the correction");
  requireToken(batch, 0, 0, lane.correction, "the correction");
}

// The match clamp (PLAN.md): a lookup keeps min(7, maximumRetained - 1)
// proposals, so the lane's copied prefix and its correction exactly fill
// what it may retain. Whatever the block, the retained rows are within the
// verify rows and the lane's retention limit, and at least the next anchor
// survives.
void retentionClamp(MetalBackend &backend) {
  Batch batch = makeBatch(backend, 3);
  std::array<Lane, 3> lanes;
  lanes[0].anchor = 501;
  lanes[0].proposals = {11, 12, 13, 14, 15, 16, 17};
  lanes[0].correction = 21;
  lanes[0].maximumRetained = kRows;
  lanes[1].anchor = 502;
  lanes[1].proposals = {21, 22, 23, 24, 25, 26, 27};
  lanes[1].correction = 43;
  lanes[1].maximumRetained = 3;
  lanes[2].anchor = 503;
  lanes[2].proposals = {31, 32, 33, 34, 35, 36, 37};
  lanes[2].correction = 44;
  lanes[2].maximumRetained = 2;
  for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
    // Proposals past min(7, maximumRetained - 1) hold what the block does
    // not: the target refuses them all.
    lanes[lane].divergesAt =
        std::min(kPositions, lanes[lane].maximumRetained - 1);
    fillLane(batch, lane, lanes[lane]);
  }
  const std::array<uint32_t, 3> maximumRetained{kRows, 3, 2};
  VerifyRowsModel model;
  runAcceptance(backend, batch, maximumRetained, model);
  requireInvariants(batch, maximumRetained, "retention clamp");
  for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
    const uint32_t copied = std::min(kPositions, maximumRetained[lane] - 1);
    require(batch.acceptedCounts()[lane] == copied,
            "lane " + std::to_string(lane) + " accepted " +
                std::to_string(batch.acceptedCounts()[lane]) + ", not " +
                std::to_string(copied));
    require(batch.retainedCounts()[lane] == copied + 1,
            "lane " + std::to_string(lane) + " retained " +
                std::to_string(batch.retainedCounts()[lane]) + ", not " +
                std::to_string(copied + 1));
    for (uint32_t position = 0; position < copied; ++position)
      requireToken(batch, lane, position, lanes[lane].proposals[position],
                   "the copied prefix");
    requireToken(batch, lane, copied, lanes[lane].correction, "the correction");
  }
}

// A block over-proposed past the retention limit (the proposal tensor is
// always seven wide): the lane keeps at most the rows it verified, and those
// are copied tokens.
void overProposedBlockClamp(MetalBackend &backend) {
  Batch batch = makeBatch(backend, 1);
  Lane lane;
  lane.anchor = 504;
  lane.proposals = {11, 12, 13, 14, 15, 16, 17};
  lane.divergesAt = kPositions;
  lane.correction = 21;
  lane.maximumRetained = 3;
  fillLane(batch, 0, lane);
  const std::array<uint32_t, 1> maximumRetained{3};
  VerifyRowsModel model;
  runAcceptance(backend, batch, maximumRetained, model);
  requireInvariants(batch, maximumRetained, "over-proposed block");
  require(batch.acceptedCounts()[0] == kPositions,
          "an agreeing over-proposed block did not accept every proposal");
  require(batch.retainedCounts()[0] == 3,
          "an over-proposed block retained past its verify rows");
  for (uint32_t position = 0; position < 3; ++position)
    requireToken(batch, 0, position, lane.proposals[position],
                 "the retained copied prefix");
}

// The lookup lane's confidence is 1 at every proposal position
// (PLAN.md: VerifyConfidence = 1 so addVerifyRows asks for the full block,
// clamped to maximumRetained): with every_row and with flat costs the lane
// takes exactly its limit, and with a budget-shaped model every lane stays
// within [1, its limit].
void verifyRowsOfALookupLane(MetalBackend &backend) {
  constexpr uint32_t lanes = 4;
  Batch batch = makeBatch(backend, lanes);
  const std::array<uint32_t, lanes> maximumRetained{kRows, 3, 1, 2};
  // The first three lanes are lookup lanes (confidence 1); the fourth is a
  // DFlash lane whose confidences spread.
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t position = 0; position < kPositions; ++position)
      batch.confidences()[lane * kPositions + position] =
          lane < lanes - 1 ? 1.0F
                           : 0.1F + 0.2F * float((position * 3) % 5);
  VerifyRowsModel everyRow;
  VerifyRowsModel flat = everyRow;
  flat.everyRow = false;
  flat.slope = 1.0F;
  for (uint32_t rows = lanes; rows <= lanes * kRows; ++rows)
    flat.costs[rows] = 1.0F;
  VerifyRowsModel budget = flat;
  for (uint32_t rows = lanes; rows <= lanes * kRows; ++rows)
    budget.costs[rows] = float(rows) * float(rows);
  for (const VerifyRowsModel &model : {everyRow, flat, budget}) {
    CommandGraph graph;
    DraftSelector::addVerifyRows(graph, batch.confidence,
                                 batch.acceptance.verifyRows, model,
                                 maximumRetained);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t rows = batch.verifyRows()[lane];
      require(rows >= 1 && rows <= maximumRetained[lane],
              "verify rows " + std::to_string(rows) + " of lane " +
                  std::to_string(lane) + " are outside [1, " +
                  std::to_string(maximumRetained[lane]) + "]");
    }
    // Every row and flat costs take the whole block a lookup asks for.
    if (model.everyRow || model.costs[lanes] == 1.0F)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        require(batch.verifyRows()[lane] == maximumRetained[lane],
                "a lookup lane verified " +
                    std::to_string(batch.verifyRows()[lane]) + " rows, not " +
                    std::to_string(maximumRetained[lane]));
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string stage = "setup";
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: copyspec-acceptance METALLIB");
    MetalBackend backend(argv[1]);
    stage = "lookup block acceptance";
    lookupBlockAcceptance(backend);
    stage = "zero acceptance";
    zeroAcceptanceKeepsTheCorrection(backend);
    stage = "retention clamp";
    retentionClamp(backend);
    stage = "over-proposed block";
    overProposedBlockClamp(backend);
    stage = "lookup verify rows";
    verifyRowsOfALookupLane(backend);
    std::cout << "copyspec_acceptance_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "copyspec_acceptance_test: FAIL (" << stage
              << "): " << error.what() << '\n';
    return 1;
  }
}
