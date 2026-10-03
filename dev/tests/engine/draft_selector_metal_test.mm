// DFlash draft selector against a direct CPU reference: the sharded top-16
// scan and its reduce must return exactly the sixteen largest logits of every
// proposal row in (value desc, id asc) order, and the codebook walk must pick
// the same tokens as a double-precision evaluation of the same scores. The
// vocabularies cover the production size, an odd size that misaligns the
// 16-byte vectors and leaves shards with only a few tokens, and one wider
// than a single register chunk per thread. The logits are fp32, and their
// order is decided below the bf16 spacing.
#include "TestChecks.hpp"
#include "draft_selector_reference.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/DraftSelector.hpp"
#include "tuning/LinearNumerics.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kRank = SPLASH_DRAFT_SELECTOR_RANK;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;

using splash::test::require;

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid draft selector request was accepted");
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
    values[index] = tuning::floatToBf16(random.unit() * scale);
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
      allocate(backend, workspace.proposalProbabilitiesBytes)};
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
  static_cast<void>(backend.submitCommand(graph.dispatches()));

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
          edge += double(tuning::bf16ToFloat(predecessors[uint64_t{predecessor} * kRank + dim])) *
                  tuning::bf16ToFloat(hidden[(uint64_t{lane} * kRows + position + 1) * kRank + dim]) *
                  tuning::bf16ToFloat(successors[uint64_t{candidate} * kRank + dim]);
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
      predecessor = token;
    }
  }
}

using splash::test::draft_selector::conditional;
using splash::test::draft_selector::expectedLength;
using splash::test::draft_selector::referenceLookahead;
using Score = splash::test::draft_selector::Score<double>;

// Direct randomized fp32 tables isolate the DP from the top-k and edge
// kernels. Poison the unused position-0 rows; they must never affect a path.
// Exact ties and sparse/all-excluded rows keep the lowest candidate index.
void lookaheadTables(MetalBackend &backend, uint32_t lanes) {
  const uint32_t positions = lanes * kPositions;
  const auto workspace = DraftSelector::workspace(positions);
  MetalBuffer candidates = allocate(backend, workspace.candidatesBytes);
  MetalBuffer unary = allocate(backend, workspace.unaryBytes);
  MetalBuffer partials = allocate(backend, workspace.partialValuesBytes);
  MetalBuffer uniforms = allocate(backend, lanes * SPLASH_SAMPLING_UNIFORMS * sizeof(float));
  MetalBuffer tokens = allocate(backend, positions * sizeof(uint32_t));
  MetalBuffer probabilities = allocate(backend, workspace.proposalProbabilitiesBytes);
  auto *ids = static_cast<uint32_t *>(candidates.contents());
  auto *scores = static_cast<float *>(unary.contents());
  float *allTables = static_cast<float *>(partials.contents()) +
                    positions * SPLASH_DRAFT_SAMPLING_SHARDS * kCandidates;
  for (uint32_t trial = 0; trial < 32; ++trial) {
    Random random(0xabc + trial * kLanes + lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t first = lane * kPositions;
      float *tables = allTables + first * kCandidates * kCandidates;
      for (uint32_t p = 0; p < kPositions; ++p) {
        for (uint32_t c = 0; c < kCandidates; ++c) {
          ids[(first + p) * kCandidates + c] =
              1000 + (first + p) * kCandidates + c;
          scores[(first + p) * kCandidates + c] =
              trial == 0 ? 0.0F : trial == 1 && c > 2 ? -INFINITY
                                                        : random.unit() * 3.0F;
        }
        for (uint32_t s = 0; s < kCandidates; ++s)
          for (uint32_t c = 0; c < kCandidates; ++c)
            tables[(p * kCandidates + s) * kCandidates + c] =
                p == 0 && s > 0 ? NAN
                : trial == 0 ? 0.0F
                : trial == 2 && p == kPositions - 1 ? -INFINITY
                                                  : random.unit() * 3.0F;
      }
    }
    SelectorBatchParams params{};
    params.lanes = lanes;
    params.vocabulary = 1 << 20;
    std::memset(probabilities.contents(), 0xA5, probabilities.sizeBytes());
    CommandGraph graph;
    graph.add("draft_select_dflash_lookahead",
              {candidates, unary, partials, uniforms, tokens, probabilities},
              params, {lanes, 1, 1}, {1, 1, 1});
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    const auto *proposed = static_cast<const uint32_t *>(tokens.contents());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t first = lane * kPositions;
      const float *tables = allTables + first * kCandidates * kCandidates;
      const splash::test::draft_selector::Score<float> score =
          [&](uint32_t p, uint32_t s, uint32_t c) {
            require(p != 0 || s == 0, "reference read an unused anchor row");
            return scores[(first + p) * kCandidates + c] +
                   tables[(p * kCandidates + s) * kCandidates + c];
          };
      const auto [best, path] = referenceLookahead(score, kPositions, kCandidates);
      static_cast<void>(best);
      for (uint32_t p = 0; p < kPositions; ++p)
        require(proposed[first + p] == ids[(first + p) * kCandidates + path[p]],
                "look-ahead tokens differ from the fp32 reference DP");
      if (trial == 0)
        require(std::all_of(path.begin(), path.end(), [](uint32_t c) { return c == 0; }),
                "look-ahead tie did not keep the lowest candidate");
    }
    const auto *bytes = static_cast<const unsigned char *>(probabilities.contents());
    require(std::all_of(bytes, bytes + probabilities.sizeBytes(),
                        [](unsigned char byte) { return byte == 0xA5; }),
            "look-ahead greedy lanes wrote sampling probabilities");
  }
}

// One thread per lane: the selection kernel alone over a crafted table where
// the best first candidate leads nowhere and the second starts a chain the
// selector is sure of. Greedy follows the first; look-ahead takes the chain.
void lookaheadCraftedTable(MetalBackend &backend) {
  constexpr uint32_t shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  const auto workspace = DraftSelector::workspace(kPositions);
  MetalBuffer candidates = allocate(backend, workspace.candidatesBytes);
  MetalBuffer unary = allocate(backend, workspace.unaryBytes);
  MetalBuffer partials = allocate(backend, workspace.partialValuesBytes);
  MetalBuffer uniforms = allocate(backend, SPLASH_SAMPLING_UNIFORMS * sizeof(float));
  MetalBuffer tokens = allocate(backend, kPositions * sizeof(uint32_t));
  MetalBuffer probabilities = allocate(backend, workspace.proposalProbabilitiesBytes);
  auto *ids = static_cast<uint32_t *>(candidates.contents());
  auto *scores = static_cast<float *>(unary.contents());
  float *tables = static_cast<float *>(partials.contents()) +
                  kPositions * shards * kCandidates;
  for (uint32_t position = 0; position < kPositions; ++position) {
    for (uint32_t c = 0; c < kCandidates; ++c) {
      ids[position * kCandidates + c] = 1000 + position * kCandidates + c;
      scores[position * kCandidates + c] = 0.0F;
    }
    // From predecessor 1, candidate 1 is all but certain; every other
    // predecessor spreads uniformly.
    if (position > 0)
      tables[(position * kCandidates + 1) * kCandidates + 1] = 10.0F;
  }
  scores[0] = 1.0F;
  scores[1] = 0.5F;
  for (uint32_t c = 2; c < kCandidates; ++c)
    scores[c] = -10.0F;

  SelectorBatchParams params{};
  params.lanes = 1;
  params.vocabulary = 1 << 20;
  for (const bool lookahead : {false, true}) {
    std::memset(tokens.contents(), 0xFF, tokens.sizeBytes());
    CommandGraph graph;
    graph.add(lookahead ? "draft_select_dflash_lookahead" : "draft_select_dflash",
              {candidates, unary, partials, uniforms, tokens, probabilities},
              params, {1, 1, 1}, {1, 1, 1});
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    const auto *proposed = static_cast<const uint32_t *>(tokens.contents());
    for (uint32_t position = 0; position < kPositions; ++position)
      require(proposed[position] ==
                  1000 + position * kCandidates + (lookahead ? 1 : 0),
              lookahead ? "look-ahead selection missed the confident chain"
                        : "greedy selection changed on the crafted table");
  }
}

// The production pipeline in look-ahead mode. Greedy lanes must reach the
// reference optimum over the tables the edge kernel actually wrote; sampling
// lanes must match greedy mode bit for bit, tokens and probabilities.
// Returns how many greedy lanes look-ahead moved off the greedy path.
uint32_t lookaheadPipeline(MetalBackend &backend, uint32_t vocabulary,
                           uint32_t lanes, uint32_t samplingMask) {
  uint32_t changed = 0;
  Random random(0x10c4 + vocabulary + lanes * 16 + samplingMask);
  const uint32_t rows = lanes * kRows;
  const uint32_t positions = lanes * kPositions;
  const auto workspace = DraftSelector::workspace(positions);
  const DraftSelector defaultSelector(vocabulary);
  require(defaultSelector.draftSelection() == DraftSelection::Greedy,
          "draft selection does not default to greedy");

  MetalBuffer logits = allocate(backend, uint64_t{rows} * vocabulary * sizeof(float));
  auto *logitRows = static_cast<float *>(logits.contents());
  const std::array patterns{Pattern::Peaked, Pattern::Uniform, Pattern::Ties,
                            Pattern::Sparse};
  for (uint32_t row = 0; row < rows; ++row)
    fillRow(logitRows + uint64_t{row} * vocabulary, vocabulary,
            patterns[(row / kRows + row % kRows) % patterns.size()], random);
  DraftSelectorBuffers buffers{
      logits,
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      // Larger codebook values than runCase: edges that move the choice.
      randomBfloat(backend, uint64_t{rows} * kRank, random, 0.5F),
      allocate(backend, uint64_t{lanes} * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, uint64_t{positions} * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  const DraftCodebooks codebooks{
      randomBfloat(backend, uint64_t{vocabulary} * kRank, random, 0.5F),
      randomBfloat(backend, uint64_t{vocabulary} * kRank, random, 0.5F)};
  auto *uniforms = static_cast<float *>(buffers.uniforms.contents());
  for (uint32_t index = 0; index < lanes * SPLASH_SAMPLING_UNIFORMS; ++index)
    uniforms[index] = (random.unit() + 1.0F) * 0.5F;
  std::vector<uint32_t> anchors(lanes);
  std::vector<SamplingPolicy> policies(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    anchors[lane] = random.next() % vocabulary;
    policies[lane] =
        SamplingPolicy{16, (samplingMask >> lane & 1U) ? 0.8F : 0.0F, 1.0F, false};
  }

  auto run = [&](DraftSelection selection) {
    const DraftSelector selector(vocabulary, selection);
    std::memset(buffers.proposedTokens.contents(), 0xFF,
                buffers.proposedTokens.sizeBytes());
    std::memset(buffers.proposalProbabilities.contents(), 0xFF,
                buffers.proposalProbabilities.sizeBytes());
    CommandGraph graph;
    selector.add(graph, buffers, codebooks, anchors, policies);
    require(graph.dispatches().size() == 3,
            "draft selector dispatch count changed");
    require(graph.dispatches()[2].pipelineName ==
                (selection == DraftSelection::Lookahead
                     ? "draft_select_dflash_lookahead"
                     : "draft_select_dflash"),
            "draft selection chose the wrong pipeline");
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    const auto *tokens =
        static_cast<const uint32_t *>(buffers.proposedTokens.contents());
    const auto *probabilities =
        static_cast<const float *>(buffers.proposalProbabilities.contents());
    return std::pair{
        std::vector<uint32_t>(tokens, tokens + positions),
        std::vector<float>(probabilities,
                           probabilities + uint64_t{positions} * kCandidates)};
  };
  const auto greedy = run(DraftSelection::Greedy);
  const auto lookahead = run(DraftSelection::Lookahead);
  const auto restored = run(DraftSelection::Greedy);
  require(restored.first == greedy.first &&
              std::memcmp(restored.second.data(), greedy.second.data(),
                          greedy.second.size() * sizeof(float)) == 0,
          "restoring greedy mode changed tokens or probabilities");

  constexpr uint32_t shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  const auto *candidates =
      static_cast<const uint32_t *>(buffers.candidates.contents());
  const auto *unary = static_cast<const float *>(buffers.unary.contents());
  const float *allTables =
      static_cast<const float *>(buffers.partialValues.contents()) +
      uint64_t{lanes} * kPositions * shards * kCandidates;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint64_t first = uint64_t{lane} * kPositions;
    if (policies[lane].samples()) {
      require(std::equal(greedy.first.begin() + first,
                         greedy.first.begin() + first + kPositions,
                         lookahead.first.begin() + first),
              "look-ahead mode changed a sampling lane's draw");
      require(std::memcmp(greedy.second.data() + first * kCandidates,
                          lookahead.second.data() + first * kCandidates,
                          kPositions * kCandidates * sizeof(float)) == 0,
              "look-ahead mode changed a sampling lane's probabilities");
      continue;
    }
    // The kernel adds fp32 unary and edge scores; evaluate the same sums.
    const float *tables = allTables + uint64_t{lane} * kPositions * kCandidates * kCandidates;
    const Score score = [&](uint32_t p, uint32_t s, uint32_t c) {
      return double(unary[(first + p) * kCandidates + c] +
                    tables[(p * kCandidates + s) * kCandidates + c]);
    };
    std::vector<uint32_t> path(kPositions);
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t token = lookahead.first[first + position];
      uint32_t selected = kCandidates;
      for (uint32_t c = 0; c < kCandidates && selected == kCandidates; ++c)
        if (candidates[(first + position) * kCandidates + c] == token)
          selected = c;
      require(selected < kCandidates,
              "look-ahead proposed a token outside its candidates");
      path[position] = selected;
    }
    const auto [best, reference] = referenceLookahead(score, kPositions, kCandidates);
    const double actual = expectedLength(score, path, kCandidates);
    // fp32 softmax and products differ from double only near exact ties.
    require(actual >= best - 1e-4 * (1.0 + best),
            "look-ahead path is worse than the reference optimum");
    // Greedy lanes in greedy mode keep the greedy walk.
    std::vector<uint32_t> greedyPath(kPositions);
    uint32_t predecessor = 0;
    for (uint32_t position = 0; position < kPositions; ++position) {
      uint32_t selected = 0;
      for (uint32_t c = 1; c < kCandidates; ++c)
        if (score(position, predecessor, c) > score(position, predecessor, selected))
          selected = c;
      greedyPath[position] = selected;
      predecessor = selected;
      require(greedy.first[first + position] ==
                  candidates[(first + position) * kCandidates + selected],
              "greedy mode no longer walks greedily");
    }
    require(actual >= expectedLength(score, greedyPath, kCandidates) - 1e-4,
            "look-ahead path is worse than the greedy walk");
    changed += path != greedyPath ? 1U : 0U;
  }
  return changed;
}

void invalidRequests(MetalBackend &backend) {
  rejects([] { DraftSelector(0); });
  rejects([] { DraftSelector(1024, DraftSelection(255)); });
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
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  const DraftCodebooks codebooks{allocate(backend, uint64_t{1024} * kRank * 2),
                                 allocate(backend, uint64_t{1024} * kRank * 2)};
  const std::array<uint32_t, 2> anchors{1, 2};
  const std::array<SamplingPolicy, 1> policies{SamplingPolicy{}};
  CommandGraph graph;
  rejects([&] { selector.add(graph, buffers, codebooks, anchors, policies); });
  require(graph.empty(), "invalid draft selector request encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: draft-selector METALLIB");
    MetalBackend backend(argv[1]);
    invalidRequests(backend);
    for (const uint32_t vocabulary : {248320U, 1003U, 270005U}) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        runCase(backend, {vocabulary, lanes, false});
        runCase(backend, {vocabulary, lanes, true});
      }
    }
    lookaheadCraftedTable(backend);
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
      lookaheadTables(backend, lanes);
    uint32_t changed = 0;
    for (const uint32_t vocabulary : {248320U, 1003U})
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
        for (uint32_t mask = 0; mask < (1U << lanes); ++mask)
          changed += lookaheadPipeline(backend, vocabulary, lanes, mask);
    // The random tables are not built to favour look-ahead; it must still
    // leave the greedy path somewhere, or the pipeline cases test nothing.
    require(changed > 0, "look-ahead never left the greedy path");
    std::cout << "draft_selector_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_selector_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
