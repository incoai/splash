// DFlash draft selector against a direct CPU reference: the sharded top-16
// scan and its reduce must return exactly the sixteen largest logits of every
// proposal row in (value desc, id asc) order, and the codebook walk must pick
// the same tokens as a double-precision evaluation of the same scores. The
// vocabularies cover the production size, an odd size that misaligns the
// 16-byte vectors and leaves shards with only a few tokens, and one wider
// than a single register chunk per thread. The logits are fp32, and their
// order is decided below the bf16 spacing. Each lane's confidence at a
// position must be the largest softmax probability at temperature 1 of its
// candidates' scores, which a sampled lane's draw there does not change; and
// draft_verify_rows must choose the verify rows whose expected tokens per
// cost are the best of every assignment.
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/DraftSelector.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;

constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kRank = SPLASH_DRAFT_SELECTOR_RANK;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;

using splash::test::rejects;
using splash::test::require;
using splash::test::requireExtent;

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

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, "draft selector");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

MetalBuffer randomBfloat(MetalBackend &backend, uint64_t count, Random &random,
                         float scale) {
  MetalBuffer buffer = allocate(backend, count * sizeof(uint16_t));
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    values[index] = floatToBf16(random.unit() * scale);
  return buffer;
}

// Row patterns: peaked logits with a few spikes (the production shape),
// uniform noise, heavy ties from a seven-value alphabet with -inf runs, and
// a row with ten finite tokens so -inf tokens fill the tail by id.
enum class Pattern : uint8_t { Peaked, Uniform, Ties, Sparse };

void fillRow(float *row, uint32_t vocabulary, Pattern pattern,
             Random &random) {
  for (uint32_t token = 0; token < vocabulary; ++token) {
    float value = 0.0F;
    switch (pattern) {
    case Pattern::Peaked:
      value = -8.0F + 3.0F * random.unit() * random.unit();
      break;
    case Pattern::Uniform:
      value = 4.0F * random.unit();
      break;
    case Pattern::Ties:
      value = random.next() % 11 == 0
                  ? -INFINITY
                  : static_cast<float>(int(random.next() % 7)) - 3.0F;
      break;
    case Pattern::Sparse:
      value = -INFINITY;
      break;
    }
    row[token] = value;
  }
  if (pattern == Pattern::Peaked) {
    for (uint32_t spike = 0; spike < 40; ++spike)
      row[random.next() % vocabulary] = 4.0F + 6.0F * random.unit();
  }
  if (pattern == Pattern::Sparse) {
    for (uint32_t finite = 0; finite < 10; ++finite)
      row[random.next() % vocabulary] = random.unit();
  }
}

// The row's sixteen largest tokens: value descending, id ascending on ties.
std::vector<uint32_t> referenceTop16(const float *row, uint32_t vocabulary) {
  std::vector<uint32_t> order(vocabulary);
  std::iota(order.begin(), order.end(), 0U);
  const auto beats = [&](uint32_t a, uint32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
  };
  const size_t keep = std::min<size_t>(kCandidates, vocabulary);
  std::partial_sort(order.begin(), order.begin() + keep, order.end(), beats);
  order.resize(keep);
  return order;
}

struct Case final {
  uint32_t vocabulary;
  uint32_t lanes;
  bool sampling;
};

void runCase(MetalBackend &backend, const Case &c) {
  Random random(0x5e1ec7 + uint64_t{c.vocabulary} * 8 + c.lanes * 2 + c.sampling);
  const uint32_t rows = c.lanes * kRows;
  const uint32_t positions = c.lanes * kPositions;
  const auto workspace = DraftSelector::workspace(positions);
  const DraftSelector selector(c.vocabulary);

  MetalBuffer logits = allocate(backend, uint64_t{rows} * c.vocabulary * sizeof(float));
  auto *logitRows = static_cast<float *>(logits.contents());
  const std::array patterns{Pattern::Peaked, Pattern::Uniform, Pattern::Ties,
                            Pattern::Sparse};
  for (uint32_t row = 0; row < rows; ++row)
    fillRow(logitRows + uint64_t{row} * c.vocabulary, c.vocabulary,
            patterns[(row / kRows + row % kRows) % patterns.size()], random);
  const MetalBuffer selectorHidden =
      randomBfloat(backend, uint64_t{rows} * kRank, random, 0.1F);
  const DraftCodebooks codebooks{
      randomBfloat(backend, uint64_t{c.vocabulary} * kRank, random, 0.1F),
      randomBfloat(backend, uint64_t{c.vocabulary} * kRank, random, 0.1F)};
  DraftSelectorBuffers buffers{
      logits,
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      selectorHidden,
      allocate(backend,
               uint64_t{c.lanes} * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, uint64_t{positions} * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes),
      allocate(backend, uint64_t{positions} * sizeof(float))};
  auto *uniforms = static_cast<float *>(buffers.uniforms.contents());
  for (uint32_t index = 0; index < c.lanes * SPLASH_SAMPLING_UNIFORMS; ++index)
    uniforms[index] = (random.unit() + 1.0F) * 0.5F;
  std::vector<uint32_t> anchors(c.lanes);
  std::vector<SamplingPolicy> policies(c.lanes);
  for (uint32_t lane = 0; lane < c.lanes; ++lane) {
    anchors[lane] = random.next() % c.vocabulary;
    policies[lane] = SamplingPolicy{16, c.sampling ? 0.8F : 0.0F, 1.0F, false};
  }

  CommandGraph graph;
  selector.add(graph, buffers, codebooks, anchors, policies);
  require(graph.dispatches().size() == 3,
          "draft selector dispatch count changed");
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  const auto *candidates =
      static_cast<const uint32_t *>(buffers.candidates.contents());
  const auto *unary = static_cast<const float *>(buffers.unary.contents());
  const auto *tokens =
      static_cast<const uint32_t *>(buffers.proposedTokens.contents());
  const auto *probabilities =
      static_cast<const float *>(buffers.proposalProbabilities.contents());
  const auto *hidden =
      static_cast<const uint16_t *>(selectorHidden.contents());
  const auto *predecessors =
      static_cast<const uint16_t *>(codebooks.predecessor.contents());
  const auto *successors =
      static_cast<const uint16_t *>(codebooks.successor.contents());
  const auto *confidence =
      static_cast<const float *>(buffers.confidence.contents());

  for (uint32_t lane = 0; lane < c.lanes; ++lane) {
    uint32_t predecessor = anchors[lane];
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t global = lane * kPositions + position;
      const float *row =
          logitRows + (uint64_t{lane} * kRows + position + 1) * c.vocabulary;
      const auto expected = referenceTop16(row, c.vocabulary);
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t id = candidates[global * kCandidates + rank];
        const float value = unary[global * kCandidates + rank];
        if (rank < expected.size()) {
          require(id == expected[rank] && value == row[expected[rank]],
                  "draft top-16 candidates differ from the exact sorted order");
        } else {
          require(id == 0xFFFFFFFFU && value == -INFINITY,
                  "draft top-16 padding lost the empty sentinel");
        }
      }

      // Scores as the kernel defines them, in double.
      std::array<double, kCandidates> scores{};
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t candidate =
            std::min(candidates[global * kCandidates + rank], c.vocabulary - 1);
        double edge = 0.0;
        for (uint32_t dim = 0; dim < kRank; ++dim) {
          edge += double(bf16ToFloat(predecessors[uint64_t{predecessor} * kRank + dim])) *
                  bf16ToFloat(hidden[(uint64_t{lane} * kRows + position + 1) * kRank + dim]) *
                  bf16ToFloat(successors[uint64_t{candidate} * kRank + dim]);
        }
        scores[rank] = double(unary[global * kCandidates + rank]) + edge;
      }
      const uint32_t token = tokens[global];
      uint32_t selected = kCandidates;
      for (uint32_t rank = 0; rank < kCandidates; ++rank)
        if (candidates[global * kCandidates + rank] == token)
          selected = selected == kCandidates ? rank : selected;
      require(selected < kCandidates,
              "draft selector proposed a token outside its candidates");
      if (c.sampling) {
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double sum = 0.0;
        std::array<double, kCandidates> reference{};
        for (uint32_t rank = 0; rank < kCandidates; ++rank) {
          reference[rank] = std::exp((scores[rank] - maximum) / 0.8);
          sum += reference[rank];
        }
        const float uniform = uniforms[lane * SPLASH_SAMPLING_UNIFORMS +
                                       SPLASH_UNIFORM_PROPOSALS + position];
        std::array<double, kCandidates> cumulative{};
        uint32_t expectedSelection = kCandidates - 1;
        for (uint32_t rank = 0; rank < kCandidates; ++rank) {
          const float probability = probabilities[global * kCandidates + rank];
          require(std::fabs(probability - reference[rank] / sum) < 1e-4,
                  "draft selector probabilities diverged from the softmax");
          cumulative[rank] = (rank ? cumulative[rank - 1] : 0.0) + reference[rank] / sum;
          if (expectedSelection == kCandidates - 1 && cumulative[rank] > uniform)
            expectedSelection = rank;
        }
        // A draw within fp32 rounding of the crossed boundary may go either way.
        const double boundary =
            std::fabs(cumulative[std::min(selected, expectedSelection)] - uniform);
        require(selected == expectedSelection || boundary < 1e-5,
                "draft selector drew a different candidate than the reference");
      } else {
        uint32_t expectedSelection = 0;
        for (uint32_t rank = 1; rank < kCandidates; ++rank)
          if (scores[rank] > scores[expectedSelection])
            expectedSelection = rank;
        // fp32 accumulation over 256 products of magnitude 1e-3 stays far
        // below this margin; only an exact tie could legitimately differ.
        require(selected == expectedSelection ||
                    std::fabs(scores[selected] - scores[expectedSelection]) < 1e-5,
                "draft selector picked a different greedy candidate");
      }
      const double maximum = *std::max_element(scores.begin(), scores.end());
      double sum = 0.0;
      for (const double score : scores)
        sum += std::exp(score - maximum);
      require(std::fabs(confidence[global] - 1.0 / sum) < 1e-4,
              "draft selector confidence is not the largest candidate probability");
      predecessor = token;
    }
  }
}

// A sampled lane's confidence at a position does not depend on the candidate
// it draws there: runs that differ only in one position's proposal uniform
// write the same confidences up to that position and the same proposals
// before it, while (over flat candidate scores) drawing different proposals
// there. So whether a row is verified never depends on its own draw.
void confidenceIgnoresDraws(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  Random random(0xd2a35);
  const auto workspace = DraftSelector::workspace(kPositions);
  const DraftSelector selector(vocabulary);
  MetalBuffer logits = allocate(backend, uint64_t{kRows} * vocabulary * sizeof(float));
  for (uint32_t row = 0; row < kRows; ++row)
    fillRow(static_cast<float *>(logits.contents()) + uint64_t{row} * vocabulary, vocabulary, Pattern::Uniform,
            random);
  const DraftSelectorBuffers buffers{logits,
                                     allocate(backend, workspace.partialIdsBytes),
                                     allocate(backend, workspace.partialValuesBytes),
                                     allocate(backend, workspace.candidatesBytes),
                                     allocate(backend, workspace.unaryBytes),
                                     randomBfloat(backend, uint64_t{kRows} * kRank, random, 0.1F),
                                     allocate(backend, SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
                                     allocate(backend, kPositions * sizeof(uint32_t)),
                                     allocate(backend, workspace.proposalProbabilitiesBytes),
                                     allocate(backend, kPositions * sizeof(float))};
  const DraftCodebooks codebooks{randomBfloat(backend, uint64_t{vocabulary} * kRank, random, 0.1F),
                                 randomBfloat(backend, uint64_t{vocabulary} * kRank, random, 0.1F)};
  const std::array<uint32_t, 1> anchors{7};
  const std::array policies{SamplingPolicy{16, 1.0F, 1.0F, false}};
  const auto run = [&](uint32_t position, float uniform) {
    auto *uniforms = static_cast<float *>(buffers.uniforms.contents());
    std::fill_n(uniforms, SPLASH_SAMPLING_UNIFORMS, 0.5F);
    uniforms[SPLASH_UNIFORM_PROPOSALS + position] = uniform;
    CommandGraph graph;
    selector.add(graph, buffers, codebooks, anchors, policies);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *tokens = static_cast<const uint32_t *>(buffers.proposedTokens.contents());
    const auto *confidence = static_cast<const float *>(buffers.confidence.contents());
    return std::pair{std::vector<uint32_t>(tokens, tokens + kPositions),
                     std::vector<float>(confidence, confidence + kPositions)};
  };
  uint32_t redrawn = 0;
  for (uint32_t position = 0; position < kPositions; ++position) {
    const auto [lowTokens, lowConfidence] = run(position, 0.05F);
    const auto [highTokens, highConfidence] = run(position, 0.95F);
    require(std::equal(lowConfidence.begin(), lowConfidence.begin() + position + 1, highConfidence.begin()),
            "a draw changed the confidence at or before its position");
    require(std::equal(lowTokens.begin(), lowTokens.begin() + position, highTokens.begin()),
            "a draw changed the proposals before its position");
    redrawn += lowTokens[position] != highTokens[position] ? 1 : 0;
  }
  require(redrawn >= 5, "the uniforms redrew only " + std::to_string(redrawn) + " proposals");
}

// draft_verify_rows against every assignment of 1 to maximum rows per lane:
// for random confidences (with exact 0 and 1), models, costs (concave, flat,
// convex and unordered) and lane maxima, its choice's expected tokens per
// cost, evaluated in double, is the best; with every_row, each lane verifies
// its maximum.
void verifyRowsChoice(MetalBackend &backend) {
  Random random(0xb0d9e7);
  MetalBuffer confidence = allocate(backend, kLanes * kPositions * sizeof(float));
  MetalBuffer verifyRows = allocate(backend, kLanes * sizeof(uint32_t));
  for (uint32_t trial = 0; trial < 400; ++trial) {
    const uint32_t lanes = 1 + trial % kLanes;
    auto *q = static_cast<float *>(confidence.contents());
    for (uint32_t index = 0; index < lanes * kPositions; ++index) {
      const uint32_t kind = random.next() % 8;
      q[index] = kind == 0 ? 0.0F : kind == 1 ? 1.0F : (random.unit() + 1.0F) * 0.5F;
    }
    VerifyRowsModel model;
    model.everyRow = trial % 50 == 49;
    model.slope = 0.3F + 0.5F * (random.unit() + 1.0F);
    for (float &intercept : model.intercepts)
      intercept = 1.5F * random.unit();
    for (uint32_t rows = lanes; rows <= lanes * kRows; ++rows) {
      const double r = rows;
      switch (trial % 4) {
      case 0: model.costs[rows] = static_cast<float>(10.0 + 5.0 * std::sqrt(r)); break;
      case 1: model.costs[rows] = 1.0F; break;
      case 2: model.costs[rows] = static_cast<float>(1.0 + 0.05 * r * r); break;
      default: model.costs[rows] = 0.5F + (random.unit() + 1.0F); break;
      }
    }
    std::vector<uint32_t> maximumRows(lanes);
    for (uint32_t &maximum : maximumRows)
      maximum = 1 + random.next() % kRows;
    CommandGraph graph;
    DraftSelector::addVerifyRows(graph, confidence, verifyRows, model, maximumRows);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *chosen = static_cast<const uint32_t *>(verifyRows.contents());
    if (model.everyRow) {
      require(std::equal(maximumRows.begin(), maximumRows.end(), chosen), "every_row did not verify the maxima");
      continue;
    }
    std::array<std::array<double, kPositions>, kLanes> survival{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      double s = 1.0;
      for (uint32_t position = 0; position < kPositions; ++position) {
        const double c = std::clamp(q[lane * kPositions + position], 1.0e-6F, 1.0F - 1.0e-6F);
        s /= 1.0 + std::exp(-(model.slope * std::log(c / (1.0 - c)) + model.intercepts[position]));
        survival[lane][position] = s;
      }
    }
    const auto value = [&](const std::array<uint32_t, kLanes> &rows) {
      double expected = 0.0;
      uint32_t total = 0;
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        expected += 1.0;
        for (uint32_t position = 0; position + 1 < rows[lane]; ++position)
          expected += survival[lane][position];
        total += rows[lane];
      }
      return expected / model.costs[total];
    };
    double best = 0.0;
    std::array<uint32_t, kLanes> rows{1, 1, 1, 1};
    for (bool more = true; more;) {
      best = std::max(best, value(rows));
      more = false;
      for (uint32_t lane = 0; lane < lanes && !more; ++lane) {
        if (rows[lane] < maximumRows[lane]) {
          ++rows[lane];
          more = true;
        } else {
          rows[lane] = 1;
        }
      }
    }
    std::array<uint32_t, kLanes> picked{1, 1, 1, 1};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      require(chosen[lane] >= 1 && chosen[lane] <= maximumRows[lane], "draft_verify_rows left a lane's range");
      picked[lane] = chosen[lane];
    }
    require(value(picked) >= best * (1.0 - 1e-5),
            "draft_verify_rows chose " + std::to_string(value(picked)) + " tokens per cost where " +
                std::to_string(best) + " was possible");
  }
}

// Each buffer the selector reaches, at its extent and one element short, for
// three lanes of which the second samples: every lane's eight query rows of
// logits and selector hidden rows, the workspaces of its seven positions,
// each codebook's row of every token, and the sampled lanes' proposal
// uniforms and probabilities up to the second lane's.
void bufferExtents(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003, lanes = 3, sampledLanes = 2;
  const DraftSelector selector(vocabulary);
  const auto workspace = DraftSelector::workspace(lanes * kPositions);
  const DraftSelectorBuffers buffers{
      allocate(backend, uint64_t{lanes} * kRows * vocabulary * sizeof(float)),
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      allocate(backend, uint64_t{lanes} * kRows * kRank * 2),
      allocate(backend, uint64_t{lanes} * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, uint64_t{lanes} * kPositions * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes),
      allocate(backend, uint64_t{lanes} * kPositions * sizeof(float))};
  const DraftCodebooks codebooks{allocate(backend, uint64_t{vocabulary} * kRank * 2),
                                 allocate(backend, uint64_t{vocabulary} * kRank * 2)};
  const std::array<uint32_t, lanes> anchors{1, 2, 3};
  const std::array policies{SamplingPolicy{}, SamplingPolicy{16, 0.8F, 1.0F}, SamplingPolicy{}};
  const uint64_t candidates = uint64_t{lanes} * kPositions * kCandidates;
  for (const auto &[member, bytes, element, what] :
       std::initializer_list<std::tuple<MetalBuffer DraftSelectorBuffers::*, uint64_t, uint64_t, const char *>>{
           {&DraftSelectorBuffers::logits, uint64_t{lanes} * kRows * vocabulary * 4, 4, "draft logits"},
           {&DraftSelectorBuffers::partialIds, candidates * SPLASH_DRAFT_SAMPLING_SHARDS * 4, 4,
            "draft selector partial id"},
           {&DraftSelectorBuffers::partialValues, candidates * (SPLASH_DRAFT_SAMPLING_SHARDS + kCandidates) * 4, 4,
            "draft selector partial value"},
           {&DraftSelectorBuffers::candidates, candidates * 4, 4, "draft candidate"},
           {&DraftSelectorBuffers::unary, candidates * 4, 4, "draft candidate score"},
           {&DraftSelectorBuffers::selectorHidden, uint64_t{lanes} * kRows * kRank * 2, 2, "draft selector hidden"},
           {&DraftSelectorBuffers::proposedTokens, uint64_t{lanes} * kPositions * 4, 4, "proposed token"},
           {&DraftSelectorBuffers::uniforms,
            ((sampledLanes - 1) * SPLASH_SAMPLING_UNIFORMS + SPLASH_UNIFORM_PROPOSALS + kPositions) * 4, 4,
            "proposal uniform"},
           {&DraftSelectorBuffers::proposalProbabilities, uint64_t{sampledLanes} * kPositions * kCandidates * 4, 4,
            "proposal probability"},
           {&DraftSelectorBuffers::confidence, uint64_t{lanes} * kPositions * 4, 4, "proposal confidence"}})
    requireExtent(backend, buffers.*member, bytes, element, what, [&](CommandGraph &graph, const MetalBuffer &buffer) {
      DraftSelectorBuffers changed = buffers;
      changed.*member = buffer;
      selector.add(graph, changed, codebooks, anchors, policies);
    });
  for (const auto &[member, what] :
       {std::pair{&DraftCodebooks::predecessor, "draft predecessor codebook"},
        std::pair{&DraftCodebooks::successor, "draft successor codebook"}})
    requireExtent(backend, codebooks.*member, uint64_t{vocabulary} * kRank * 2, 2, what,
                  [&](CommandGraph &graph, const MetalBuffer &buffer) {
                    DraftCodebooks changed = codebooks;
                    changed.*member = buffer;
                    selector.add(graph, buffers, changed, anchors, policies);
                  });
  // draft_verify_rows reads every lane's confidences and writes its rows.
  const std::array<uint32_t, lanes> maximumRows{kRows, kRows, kRows};
  const MetalBuffer verifyRows = allocate(backend, uint64_t{lanes} * sizeof(uint32_t));
  requireExtent(backend, buffers.confidence, uint64_t{lanes} * kPositions * 4, 4, "proposal confidence",
                [&](CommandGraph &graph, const MetalBuffer &buffer) {
                  DraftSelector::addVerifyRows(graph, buffer, verifyRows, VerifyRowsModel{}, maximumRows);
                });
  requireExtent(backend, verifyRows, uint64_t{lanes} * 4, 4, "verify rows",
                [&](CommandGraph &graph, const MetalBuffer &buffer) {
                  DraftSelector::addVerifyRows(graph, buffers.confidence, buffer, VerifyRowsModel{}, maximumRows);
                });
}

void invalidRequests(MetalBackend &backend) {
  rejects([] { DraftSelector(0); }, "invalid draft selector vocabulary",
          "a selector over no vocabulary was accepted");
  const DraftSelector selector(1024);
  const auto workspace = DraftSelector::workspace(kPositions);
  const DraftSelectorBuffers buffers{
      allocate(backend, uint64_t{kRows} * 1024 * sizeof(float)),
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      allocate(backend, uint64_t{kRows} * kRank * 2),
      allocate(backend, SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, kPositions * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes),
      allocate(backend, kPositions * sizeof(float))};
  const DraftCodebooks codebooks{allocate(backend, uint64_t{1024} * kRank * 2),
                                 allocate(backend, uint64_t{1024} * kRank * 2)};
  const std::array<uint32_t, 2> anchors{1, 2};
  const std::array<SamplingPolicy, 1> policies{SamplingPolicy{}};
  CommandGraph graph;
  rejects([&] { selector.add(graph, buffers, codebooks, anchors, policies); },
          "invalid draft selector batch", "anchors without a policy each were accepted");
  const MetalBuffer verifyRows = allocate(backend, kLanes * sizeof(uint32_t));
  const MetalBuffer confidence = allocate(backend, kLanes * kPositions * sizeof(float));
  rejects([&] { DraftSelector::addVerifyRows(graph, confidence, verifyRows, {}, {}); },
          "invalid draft verify rows batch", "verify rows of no lane were encoded");
  rejects([&] { DraftSelector::addVerifyRows(graph, confidence, verifyRows, {}, std::array<uint32_t, 5>{8, 8, 8, 8, 8}); },
          "invalid draft verify rows batch", "verify rows of five lanes were encoded");
  for (const uint32_t maximum : {0U, kRows + 1})
    rejects([&] { DraftSelector::addVerifyRows(graph, confidence, verifyRows, {}, std::array{maximum}); },
            "invalid draft verify rows maximum", "a maximum of verify rows outside a block was accepted");
  VerifyRowsModel valid;
  valid.everyRow = false;
  valid.slope = 0.7F;
  for (uint32_t rows = 2; rows <= 2 * kRows; ++rows)
    valid.costs[rows] = float(rows);
  const std::array<uint32_t, 2> two{kRows, kRows};
  VerifyRowsModel slope = valid, intercept = valid, zero = valid, negative = valid;
  slope.slope = NAN;
  intercept.intercepts[3] = INFINITY;
  zero.costs[2] = 0.0F;
  negative.costs[2 * kRows] = -1.0F;
  for (const auto &[model, message] : {std::pair{slope, "invalid draft verify rows model"},
                                       std::pair{intercept, "invalid draft verify rows model"},
                                       std::pair{zero, "invalid draft verify rows cost"},
                                       std::pair{negative, "invalid draft verify rows cost"}})
    rejects([&] { DraftSelector::addVerifyRows(graph, confidence, verifyRows, model, two); }, message,
            "an invalid verify rows model was accepted");
  require(graph.empty(), "invalid draft selector request encoded a graph");
  // Costs outside the totals the lanes may verify are not read.
  CommandGraph encoded;
  DraftSelector::addVerifyRows(encoded, confidence, verifyRows, valid, two);
  require(!encoded.empty(), "a valid verify rows model was not encoded");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: draft-selector METALLIB");
    MetalBackend backend(argv[1]);
    invalidRequests(backend);
    bufferExtents(backend);
    for (const uint32_t vocabulary : {248320U, 1003U, 270005U}) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        runCase(backend, {vocabulary, lanes, false});
        runCase(backend, {vocabulary, lanes, true});
      }
    }
    confidenceIgnoresDraws(backend);
    verifyRowsChoice(backend);
    std::cout << "draft_selector_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_selector_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
