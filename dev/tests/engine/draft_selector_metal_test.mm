// DFlash draft selector against direct CPU references. The sharded top-16
// scan and its reduce must return exactly the sixteen largest logits of every
// proposal row in (value desc, id asc) order, and each edge of the codebook
// table its double-precision product. The vocabularies cover the production
// size, an odd size that misaligns the 16-byte vectors and leaves shards
// with only a few tokens, and one wider than a single register chunk per
// thread; the logits are fp32, and their order is decided below the bf16
// spacing.
//
// The tree each lane grows from the candidates (draft_select_tree) must be a
// brute-force reference's, row for row: every path that takes one of a
// row's `children` likeliest candidates a step and every prefix of the lane's
// host chain, scored by the product of its steps' estimates (a draft step's
// by its lane's own calibration), ranked, one row for each token path, as
// many as a verify block holds and none deeper than its depth: its tokens,
// parents, depths, ancestors, draft probabilities (none for a row a host
// chain token made) and estimates. A tree of one child a row and no chain is
// the chain the selector walked before it built trees, token for token, from
// the same scores. A host chain whose tokens the target selects whatever its
// logits, with no draft children, takes the first rows. draft_verify_rows
// must choose the verify rows of every lane whose expected tokens (per cost)
// are the best of every assignment the step's rows and the lanes' cap allow,
// pack the lanes' rows in lane order with the step's other rows on the first
// lanes under the cap, and number a lane's verify rows in preorder, the
// earlier-taken child first.
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
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
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
constexpr uint32_t kBlock = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kChain = SPLASH_TREE_CHAIN_TOKENS;
constexpr uint32_t kAgreements = SPLASH_TREE_AGREEMENTS;

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

// Every lane's draft rows of `lanes` lanes, a selector's buffers over them
// and the codebooks, with each lane's tree of up to kBlock rows at lane x
// kBlock.
struct Fixture final {
  uint32_t vocabulary = 0;
  uint32_t lanes = 0;
  MetalBuffer logits;
  DraftSelectorBuffers buffers;
  DraftCodebooks codebooks;

  [[nodiscard]] const float *logitRow(uint32_t lane, uint32_t position) const {
    return static_cast<const float *>(logits.contents()) +
           (uint64_t{lane} * kRows + position + 1) * vocabulary;
  }
  [[nodiscard]] const uint32_t *candidates(uint32_t lane) const {
    return static_cast<const uint32_t *>(buffers.candidates.contents()) +
           uint64_t{lane} * kPositions * kCandidates;
  }
  [[nodiscard]] const float *unary(uint32_t lane) const {
    return static_cast<const float *>(buffers.unary.contents()) + uint64_t{lane} * kPositions * kCandidates;
  }
  // The edge table draft_select_edges leaves behind the shard partials.
  [[nodiscard]] const float *edges(uint32_t lane, uint32_t position, uint32_t predecessor) const {
    return static_cast<const float *>(buffers.partialValues.contents()) +
           uint64_t{lanes} * kPositions * SPLASH_DRAFT_SAMPLING_SHARDS * kCandidates +
           ((uint64_t{lane} * kPositions + position) * kCandidates + predecessor) * kCandidates;
  }
  [[nodiscard]] const uint32_t *tokens() const { return static_cast<const uint32_t *>(buffers.treeTokens.contents()); }
  [[nodiscard]] const TreeRow *tree() const { return static_cast<const TreeRow *>(buffers.trees.contents()); }
  [[nodiscard]] const float *scores() const { return static_cast<const float *>(buffers.treeScores.contents()); }
  [[nodiscard]] uint32_t rowCount(uint32_t lane) const {
    return static_cast<const uint32_t *>(buffers.treeSizes.contents())[lane];
  }
  [[nodiscard]] uint32_t *chainTokens(uint32_t lane) const {
    return static_cast<uint32_t *>(buffers.chainTokens.contents()) + uint64_t{lane} * kChain;
  }
  [[nodiscard]] float *chainRates(uint32_t lane) const {
    return static_cast<float *>(buffers.chainRates.contents()) + uint64_t{lane} * kChain * kAgreements;
  }
};

// Each lane's draft rows of `patterns` (by position) and its selector hidden
// rows and the codebooks of scale `edgeScale`: the larger, the more an edge
// moves a candidate's probability.
Fixture makeFixture(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes, Random &random,
                    std::span<const Pattern> patterns, float edgeScale) {
  Fixture f;
  f.vocabulary = vocabulary;
  f.lanes = lanes;
  const uint32_t rows = lanes * kRows;
  const auto workspace = DraftSelector::workspace(lanes * kPositions);
  f.logits = allocate(backend, uint64_t{rows} * vocabulary * sizeof(float));
  auto *logitRows = static_cast<float *>(f.logits.contents());
  for (uint32_t row = 0; row < rows; ++row)
    fillRow(logitRows + uint64_t{row} * vocabulary, vocabulary,
            patterns[(row / kRows + row % kRows) % patterns.size()], random);
  f.buffers = {f.logits,
               allocate(backend, workspace.partialIdsBytes),
               allocate(backend, workspace.partialValuesBytes),
               allocate(backend, workspace.candidatesBytes),
               allocate(backend, workspace.unaryBytes),
               randomBfloat(backend, uint64_t{rows} * kRank, random, edgeScale),
               allocate(backend, uint64_t{lanes} * kChain * sizeof(uint32_t)),
               allocate(backend, uint64_t{lanes} * kChain * kAgreements * sizeof(float)),
               allocate(backend, uint64_t{lanes} * kBlock * sizeof(uint32_t)),
               allocate(backend, uint64_t{lanes} * kBlock * sizeof(TreeRow)),
               allocate(backend, uint64_t{lanes} * kBlock * 2 * sizeof(float)),
               allocate(backend, uint64_t{lanes} * sizeof(uint32_t)),
               allocate(backend, uint64_t{lanes} * kBlock * sizeof(uint32_t)),
               allocate(backend, uint64_t{lanes} * kBlock * sizeof(TreeRow)),
               allocate(backend, uint64_t{lanes} * sizeof(LaneRows)),
               allocate(backend, uint64_t{lanes} * sizeof(uint32_t))};
  f.codebooks = {randomBfloat(backend, uint64_t{vocabulary} * kRank, random, edgeScale),
                 randomBfloat(backend, uint64_t{vocabulary} * kRank, random, edgeScale)};
  return f;
}


void run(MetalBackend &backend, const DraftSelector &selector, const Fixture &f, std::span<const DraftTreeLane> lanes) {
  CommandGraph graph;
  selector.add(graph, f.buffers, f.codebooks, lanes);
  require(graph.dispatches().size() == 3, "draft selector dispatch count changed");
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

// A lane's candidates as the selector scored them: their ids, each one's
// fp32 score after each predecessor (unary plus edge, as the kernel adds
// them), its probability in double, and each predecessor's candidates in the
// order of their scores, the lower index first among equal ones.
struct Lattice final {
  std::array<std::array<uint32_t, kCandidates>, kPositions> ids{};
  std::array<std::array<std::array<float, kCandidates>, kCandidates>, kPositions> score{};
  std::array<std::array<std::array<double, kCandidates>, kCandidates>, kPositions> q{};
  std::array<std::array<std::array<uint32_t, kCandidates>, kCandidates>, kPositions> ranked{};
};

Lattice latticeOf(const Fixture &f, uint32_t lane) {
  Lattice l;
  for (uint32_t position = 0; position < kPositions; ++position) {
    std::copy_n(f.candidates(lane) + position * kCandidates, kCandidates, l.ids[position].begin());
    for (uint32_t predecessor = 0; predecessor < (position ? kCandidates : 1); ++predecessor) {
      auto &score = l.score[position][predecessor];
      for (uint32_t i = 0; i < kCandidates; ++i)
        score[i] = f.unary(lane)[position * kCandidates + i] + f.edges(lane, position, predecessor)[i];
      const float best = *std::max_element(score.begin(), score.end());
      double mass = 0.0;
      for (uint32_t i = 0; i < kCandidates; ++i)
        mass += std::exp(double(score[i]) - best);
      for (uint32_t i = 0; i < kCandidates; ++i)
        l.q[position][predecessor][i] = std::exp(double(score[i]) - best) / mass;
      auto &ranked = l.ranked[position][predecessor];
      std::iota(ranked.begin(), ranked.end(), 0U);
      std::stable_sort(ranked.begin(), ranked.end(), [&](uint32_t a, uint32_t b) { return score[a] > score[b]; });
    }
  }
  return l;
}

double estimate(const DraftCalibration &calibration, double q, uint32_t position) {
  q = std::clamp(q, 1.0e-6, 1.0 - 1.0e-6);
  return 1.0 / (1.0 + std::exp(-(calibration.slope * std::log(q / (1.0 - q)) + calibration.intercepts[position])));
}

// A lane's host chain: its tokens and their rates by agreement.
struct Chain final {
  std::vector<uint32_t> tokens;
  std::vector<std::array<float, kAgreements>> rates;
};

uint32_t agreement(const Lattice &l, uint32_t token, uint32_t depth) {
  if (depth >= kPositions)
    return 3;
  if (token == l.ids[depth][0])
    return 0;
  return std::find(l.ids[depth].begin(), l.ids[depth].end(), token) != l.ids[depth].end() ? 1 : 2;
}

// A reference tree's row: its token path from the anchor (the anchor's is
// empty), parent row, draft probability (-1 for a row a host chain token
// made) and estimate.
struct Node final {
  std::vector<uint32_t> path;
  uint32_t parent = 0;
  double q = 1.0;
  double estimate = 1.0;
};

// The brute-force tree of a lane, rows in the order the selector takes them;
// none when two paths' scores lie too close for float sums to order (the
// caller draws other data).
std::optional<std::vector<Node>> referenceTree(const Lattice &l, const DraftTreeLane &lane, const Chain &chain) {
  struct Entry final {
    std::vector<uint32_t> path;
    double score;
    bool chain;
  };
  std::vector<Entry> entries;
  // Draft paths: a row's `children` likeliest candidates after its own.
  std::function<void(std::vector<uint32_t> &, uint32_t, uint32_t, double)> expand =
      [&](std::vector<uint32_t> &path, uint32_t position, uint32_t predecessor, double score) {
        for (uint32_t child = 0; child < lane.children; ++child) {
          const uint32_t index = l.ranked[position][predecessor][child];
          const double q = l.q[position][predecessor][index];
          if (!(q > 0.0))
            continue;
          path.push_back(l.ids[position][index]);
          const double next = score * estimate(lane.calibration, q, position);
          entries.push_back({path, next, false});
          if (position + 1 < lane.depth && position + 1 < kPositions)
            expand(path, position + 1, index, next);
          path.pop_back();
        }
      };
  std::vector<uint32_t> path;
  if (lane.depth > 0)
    expand(path, 0, 0, 1.0);
  double score = 1.0;
  for (uint32_t depth = 0; depth < lane.chain && depth < lane.depth; ++depth) {
    score *= chain.rates[depth][agreement(l, chain.tokens[depth], depth)];
    entries.push_back({std::vector<uint32_t>(chain.tokens.begin(), chain.tokens.begin() + depth + 1), score, true});
  }
  std::stable_sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.score > b.score; });

  std::vector<Node> rows{Node{}};
  std::map<std::vector<uint32_t>, uint32_t> made{{{}, 0}};
  for (size_t index = 0; index < entries.size() && rows.size() < kBlock; ++index) {
    const Entry &entry = entries[index];
    // Float sums decide between paths within this of each other.
    if (index + 1 < entries.size() && entries[index + 1].path != entry.path &&
        entries[index + 1].score > entry.score * (1.0 - 1e-4))
      return std::nullopt;
    if (made.contains(entry.path))
      continue;
    Node node;
    node.path = entry.path;
    node.parent = made.at(std::vector<uint32_t>(entry.path.begin(), entry.path.end() - 1));
    node.estimate = entry.score;
    const uint32_t position = static_cast<uint32_t>(entry.path.size()) - 1;
    // A token's candidate index at its position, or none.
    const auto candidate = [&](uint32_t at, uint32_t token) -> int {
      if (at >= kPositions)
        return -1;
      const auto found = std::find(l.ids[at].begin(), l.ids[at].end(), token);
      return found == l.ids[at].end() ? -1 : int(found - l.ids[at].begin());
    };
    const int own = candidate(position, entry.path.back());
    const int predecessor = position ? candidate(position - 1, entry.path[position - 1]) : 0;
    node.q = entry.chain ? -1.0 : l.q[position][predecessor][own];
    made.emplace(entry.path, static_cast<uint32_t>(rows.size()));
    rows.push_back(node);
  }
  return rows;
}

// A lane's tree from the selector as `rows`, in its order: tokens, parents,
// depths, ancestors, draft probabilities and estimates, and its size.
void requireTree(const Fixture &f, uint32_t lane, uint32_t anchor, const std::vector<Node> &rows,
                 const std::string &label) {
  const uint32_t first = lane * kBlock;
  require(f.rowCount(lane) == rows.size(), label + ": the tree holds " + std::to_string(f.rowCount(lane)) +
                                               " rows, not " + std::to_string(rows.size()));
  std::vector<uint32_t> ancestors(rows.size());
  for (uint32_t row = 0; row < rows.size(); ++row) {
    const Node &node = rows[row];
    const TreeRow &got = f.tree()[first + row];
    ancestors[row] = (row ? ancestors[node.parent] : 0U) | (1U << row);
    const uint32_t token = row ? node.path.back() : anchor;
    require(f.tokens()[first + row] == token && got.parent == (row ? node.parent : SPLASH_TREE_ROOT) &&
                got.depth == node.path.size() && got.ancestors == ancestors[row],
            label + ": row " + std::to_string(row) + " differs from the reference's");
    const float q = f.scores()[2 * (first + row)], e = f.scores()[2 * (first + row) + 1];
    require(row == 0 || (node.q < 0.0 ? q == -1.0F : std::fabs(q - node.q) <= 1e-4 * node.q + 1e-7),
            label + ": row " + std::to_string(row) + "'s draft probability " + std::to_string(q) + " is not " +
                std::to_string(node.q));
    require(std::fabs(e - node.estimate) <= 1e-4 * node.estimate,
            label + ": row " + std::to_string(row) + "'s estimate " + std::to_string(e) + " is not " +
                std::to_string(node.estimate));
  }
}

// The selector's top-16 candidates and their edge table over production-like
// and adversarial rows, and a tree of one child a row and no chain: the walk
// the chain selector took, its proposal at each position the candidate of
// the largest score (unary plus edge, in fp32 as the kernel adds them) after
// the previous one's, the lowest index among equal ones, and each row's draft
// probability the largest of its candidates', the chain's confidence.
void chainCase(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes) {
  Random random(0x5e1ec7 + uint64_t{vocabulary} * 8 + lanes * 2);
  const std::array patterns{Pattern::Peaked, Pattern::Uniform, Pattern::Ties, Pattern::Sparse};
  const Fixture f = makeFixture(backend, vocabulary, lanes, random, patterns, 0.1F);
  const DraftSelector selector(vocabulary);
  std::vector<DraftTreeLane> trees;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    trees.push_back({random.next() % vocabulary, kBlock - 1, 0, 1});
  run(backend, selector, f, trees);

  const auto *hidden = static_cast<const uint16_t *>(f.buffers.selectorHidden.contents());
  const auto *predecessors = static_cast<const uint16_t *>(f.codebooks.predecessor.contents());
  const auto *successors = static_cast<const uint16_t *>(f.codebooks.successor.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    uint32_t predecessorToken = trees[lane].anchor, predecessor = 0;
    const uint32_t first = lane * kBlock;
    require(f.rowCount(lane) == std::min(kBlock, kPositions + 1), "a chain does not hold every position");
    for (uint32_t position = 0; position < kPositions; ++position) {
      const float *row = f.logitRow(lane, position);
      const auto expected = referenceTop16(row, vocabulary);
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t id = f.candidates(lane)[position * kCandidates + rank];
        const float value = f.unary(lane)[position * kCandidates + rank];
        if (rank < expected.size())
          require(id == expected[rank] && value == row[expected[rank]],
                  "draft top-16 candidates differ from the exact sorted order");
        else
          require(id == 0xFFFFFFFFU && value == -INFINITY, "draft top-16 padding lost the empty sentinel");
      }
      // The edges from the previous proposal, in double, against its row of
      // the table.
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t candidate = std::min(f.candidates(lane)[position * kCandidates + rank], vocabulary - 1);
        double edge = 0.0;
        for (uint32_t dim = 0; dim < kRank; ++dim)
          edge += double(bf16ToFloat(predecessors[uint64_t{predecessorToken} * kRank + dim])) *
                  bf16ToFloat(hidden[(uint64_t{lane} * kRows + position + 1) * kRank + dim]) *
                  bf16ToFloat(successors[uint64_t{candidate} * kRank + dim]);
        require(std::fabs(f.edges(lane, position, predecessor)[rank] - edge) < 1e-5,
                "a draft edge differs from its codebook product");
      }
      // The chain walk's choice.
      float scores[kCandidates];
      for (uint32_t i = 0; i < kCandidates; ++i)
        scores[i] = f.unary(lane)[position * kCandidates + i] + f.edges(lane, position, predecessor)[i];
      uint32_t selected = 0;
      for (uint32_t i = 1; i < kCandidates; ++i)
        if (scores[i] > scores[selected])
          selected = i;
      const uint32_t row1 = position + 1;
      if (row1 >= kBlock)
        break;
      const uint32_t token = f.candidates(lane)[position * kCandidates + selected];
      require(f.tokens()[first + row1] == token && f.tree()[first + row1].parent == row1 - 1 &&
                  f.tree()[first + row1].depth == row1,
              "a tree of one child a row is not the chain walk's");
      const float best = *std::max_element(scores, scores + kCandidates);
      double mass = 0.0;
      for (const float score : scores)
        mass += std::exp(double(score) - best);
      require(std::fabs(f.scores()[2 * (first + row1)] - 1.0 / mass) < 1e-4,
              "a chain row's draft probability is not its position's confidence");
      predecessorToken = token;
      predecessor = selected;
    }
  }
}

// Trees of 2 to 4 children a row, with and without host chains, against the
// brute-force reference at every lane count: the chains copy the draft's
// likeliest path for a while (whose rows they share), turn to other
// candidates and to tokens the draft does not hold, and run past its
// positions; their rates and each lane's calibration, the identity or another
// beside lanes of the other, rank them against the draft's rows. Lanes cut
// their trees at a depth or take all of their rows.
void treesMatchTheReference(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  const std::array patterns{Pattern::Peaked};
  uint32_t checked = 0, drawn = 0;
  for (uint32_t trial = 0; checked < 48; ++trial) {
    require(trial < 400, "the reference found too few unambiguous trees");
    Random random(0x74726565 + trial);
    const uint32_t lanes = 1 + trial % kLanes;
    const Fixture f = makeFixture(backend, vocabulary, lanes, random, patterns, 0.4F);
    const DraftSelector selector(vocabulary);
    DraftCalibration calibrated;
    calibrated.slope = 0.7F;
    for (uint32_t position = 0; position < kPositions; ++position)
      calibrated.intercepts[position] = -0.15F * float(position);
    std::vector<DraftTreeLane> trees;
    std::vector<Chain> chains(lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t children = 2 + (trial + lane) % 3;
      const uint32_t depth = (trial + lane) % 4 == 3 ? 2 + (trial + lane) % 3 : kBlock - 1;
      const uint32_t chain = (trial + lane) % 3 == 0 ? 0 : 1 + random.next() % kChain;
      trees.push_back({random.next() % vocabulary, depth, chain, children,
                       (trial + lane) % 2 ? calibrated : DraftCalibration{}});
      chains[lane].tokens.resize(kChain);
      chains[lane].rates.resize(kChain);
      for (uint32_t d = 0; d < kChain; ++d) {
        chains[lane].tokens[d] = random.next() % vocabulary;
        for (float &rate : chains[lane].rates[d])
          rate = 0.05F + 0.95F * 0.5F * (random.unit() + 1.0F);
      }
    }
    // A first run gives the lattices the chains copy from.
    run(backend, selector, f, trees);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const Lattice l = latticeOf(f, lane);
      // The draft's likeliest path for its first `copied` tokens, then its
      // second candidate once, then the random tokens.
      const uint32_t copied = random.next() % 4;
      uint32_t predecessor = 0;
      for (uint32_t d = 0; d < copied && d < kPositions; ++d) {
        const uint32_t index = l.ranked[d][predecessor][0];
        chains[lane].tokens[d] = l.ids[d][index];
        predecessor = index;
      }
      if (copied < kPositions)
        chains[lane].tokens[copied] = l.ids[copied][l.ranked[copied][predecessor][1]];
      std::copy(chains[lane].tokens.begin(), chains[lane].tokens.end(), f.chainTokens(lane));
      for (uint32_t d = 0; d < kChain; ++d)
        std::copy(chains[lane].rates[d].begin(), chains[lane].rates[d].end(), f.chainRates(lane) + d * kAgreements);
    }
    run(backend, selector, f, trees);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      ++drawn;
      const auto reference = referenceTree(latticeOf(f, lane), trees[lane], chains[lane]);
      if (!reference)
        continue;
      ++checked;
      requireTree(f, lane, trees[lane].anchor, *reference,
                  "tree trial " + std::to_string(trial) + " lane " + std::to_string(lane));
    }
  }
  require(checked * 4 >= drawn * 3, "most trees were too ambiguous to check");
}

// A host chain the target selects whatever its logits (a thinking end), at
// rates of 1 and with no draft children, takes the first rows in order, none
// with a draft probability; its depth cuts it.
void forcedChain(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  Random random(0x666f7263);
  const std::array patterns{Pattern::Peaked};
  const Fixture f = makeFixture(backend, vocabulary, 2, random, patterns, 0.1F);
  const DraftSelector selector(vocabulary);
  const std::array<uint32_t, 3> end{vocabulary - 6, vocabulary - 7, vocabulary - 3};
  for (uint32_t lane = 0; lane < 2; ++lane) {
    std::copy(end.begin(), end.end(), f.chainTokens(lane));
    std::fill_n(f.chainRates(lane), end.size() * kAgreements, 1.0F);
  }
  const std::array trees{DraftTreeLane{5, kBlock - 1, 3, 0}, DraftTreeLane{9, 2, 3, 0}};
  run(backend, selector, f, trees);
  for (uint32_t lane = 0; lane < 2; ++lane) {
    const uint32_t made = std::min(3U, trees[lane].depth) + 1;
    std::vector<Node> rows{Node{}};
    for (uint32_t d = 0; d + 1 < made; ++d)
      rows.push_back({std::vector<uint32_t>(end.begin(), end.begin() + d + 1), d, -1.0, 1.0});
    requireTree(f, lane, trees[lane].anchor, rows, "forced chain lane " + std::to_string(lane));
  }
}

// A lane's rows as a tree in the selector's order and their scores, drawn at
// random: each row's parent an earlier row, its estimate at most its parent's.
struct DrawnTree final {
  std::vector<uint32_t> tokens;
  std::vector<uint32_t> parents;
  std::vector<float> estimates;
};

DrawnTree drawTree(Random &random, uint32_t rows, uint32_t anchor) {
  DrawnTree t;
  t.tokens.push_back(anchor);
  t.parents.push_back(SPLASH_TREE_ROOT);
  t.estimates.push_back(1.0F);
  for (uint32_t row = 1; row < rows; ++row) {
    t.tokens.push_back(100 + random.next() % 900);
    t.parents.push_back(random.next() % row);
    // Taken in the order of their estimates; an exact 1 or 0 now and then.
    const uint32_t kind = random.next() % 10;
    const float below = t.estimates.back();
    const float estimate = kind == 0 ? below : kind == 1 ? 0.0F : below * 0.5F * (random.unit() + 1.0F);
    t.estimates.push_back(std::min(estimate, t.estimates[t.parents.back()]));
  }
  return t;
}

// The best assignment of verify rows to lanes by brute force: each lane a
// prefix of its tree's rows, its anchor to at most `cap` of them, the lanes'
// together at most `rows`; with costs the most expected tokens per cost,
// without the most expected tokens. Expected tokens count a lane's anchor 1
// and each other row its estimate, in double.
double bestAssignment(const std::vector<DrawnTree> &trees, uint32_t rows, uint32_t cap,
                      std::span<const float> costs) {
  const uint32_t lanes = static_cast<uint32_t>(trees.size());
  std::vector<uint32_t> kept(lanes, 1);
  double best = 0.0;
  for (bool more = true; more;) {
    double expected = 0.0;
    uint32_t total = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      for (uint32_t row = 0; row < kept[lane]; ++row)
        expected += row ? trees[lane].estimates[row] : 1.0;
      total += kept[lane];
    }
    if (total <= rows)
      best = std::max(best, costs.empty() ? expected : expected / costs[total]);
    more = false;
    for (uint32_t lane = 0; lane < lanes && !more; ++lane) {
      if (kept[lane] < std::min<uint32_t>(cap, static_cast<uint32_t>(trees[lane].tokens.size()))) {
        ++kept[lane];
        more = true;
      } else {
        kept[lane] = 1;
      }
    }
  }
  return best;
}

// draft_verify_rows against every assignment of verify rows: for random
// trees (estimates with exact 0 and repeats, of 1 to kBlock rows), steps of
// 1 to 4 lanes, every count of rows and cap they allow, and costs (concave,
// flat, convex and unordered) or none, its choice's expected tokens (per
// cost) are the best, a lane verifies the first rows its tree took, no lane
// more than the cap and the lanes no more than the step's rows. The lanes'
// rows fill the step's in lane order, the rows past the verify rows on the
// first lanes under the cap. A lane's verify rows are numbered in preorder,
// the earlier-taken child first, with their tokens, depths and ancestors, and
// each row past them holds the anchor's token and extends the path of the
// row before it.
void verifyRowsLayout(MetalBackend &backend) {
  Random random(0xb0d9e7);
  for (uint32_t trial = 0; trial < 600; ++trial) {
    const uint32_t lanes = 1 + trial % kLanes;
    const bool everyRow = trial % 5 == 4;
    const uint32_t cap = 1 + random.next() % kBlock;
    const uint32_t rows = lanes + random.next() % (lanes * cap - lanes + 1);
    std::vector<DrawnTree> trees;
    DraftSelectorBuffers b{};
    b.treeTokens = allocate(backend, uint64_t{lanes} * kBlock * 4);
    b.trees = allocate(backend, uint64_t{lanes} * kBlock * sizeof(TreeRow));
    b.treeScores = allocate(backend, uint64_t{lanes} * kBlock * 8);
    b.treeSizes = allocate(backend, uint64_t{lanes} * 4);
    b.inputTokens = allocate(backend, uint64_t{rows} * 4);
    b.treeRows = allocate(backend, uint64_t{rows} * sizeof(TreeRow));
    b.laneRows = allocate(backend, uint64_t{lanes} * sizeof(LaneRows));
    b.verifyRows = allocate(backend, uint64_t{lanes} * 4);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t size = 1 + random.next() % kBlock;
      trees.push_back(drawTree(random, size, 7 + lane));
      static_cast<uint32_t *>(b.treeSizes.contents())[lane] = size;
      // The selector's rows: depths and ancestors from the parents.
      std::vector<uint32_t> depth(size), ancestors(size);
      for (uint32_t row = 0; row < size; ++row) {
        const uint32_t parent = trees[lane].parents[row];
        depth[row] = row ? depth[parent] + 1 : 0;
        ancestors[row] = (row ? ancestors[parent] : 0U) | (1U << row);
        const uint32_t at = lane * kBlock + row;
        static_cast<uint32_t *>(b.treeTokens.contents())[at] = trees[lane].tokens[row];
        static_cast<TreeRow *>(b.trees.contents())[at] = {ancestors[row], static_cast<uint16_t>(parent),
                                                         static_cast<uint16_t>(depth[row])};
        static_cast<float *>(b.treeScores.contents())[2 * at] = float(row) * 0.01F;
        static_cast<float *>(b.treeScores.contents())[2 * at + 1] = trees[lane].estimates[row];
      }
    }
    std::vector<float> costs(kLanes * kBlock + 1, 0.0F);
    for (uint32_t total = lanes; total <= rows; ++total) {
      const double r = total;
      switch (trial % 4) {
      case 0: costs[total] = static_cast<float>(10.0 + 5.0 * std::sqrt(r)); break;
      case 1: costs[total] = 1.0F; break;
      case 2: costs[total] = static_cast<float>(1.0 + 0.05 * r * r); break;
      default: costs[total] = 0.5F + (random.unit() + 1.0F); break;
      }
    }
    const std::span<const float> used = everyRow ? std::span<const float>{} : std::span<const float>(costs);
    CommandGraph graph;
    DraftSelector::addVerifyRows(graph, b, {lanes, rows, cap}, used);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *verified = static_cast<const uint32_t *>(b.verifyRows.contents());
    const auto *laneRows = static_cast<const LaneRows *>(b.laneRows.contents());
    const std::string label = "verify rows trial " + std::to_string(trial);

    // The choice: prefixes within the cap and the step's rows, the best.
    double expected = 0.0;
    uint32_t total = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      require(verified[lane] >= 1 && verified[lane] <= std::min<uint32_t>(cap, trees[lane].tokens.size()),
              label + ": a lane's verify rows left its tree or its cap");
      for (uint32_t row = 0; row < verified[lane]; ++row)
        expected += row ? trees[lane].estimates[row] : 1.0;
      total += verified[lane];
    }
    require(total <= rows, label + ": the lanes verify more rows than the step holds");
    const double value = everyRow ? expected : expected / costs[total];
    const double best = bestAssignment(trees, rows, cap, used);
    require(value >= best * (1.0 - 1e-5), label + ": draft_verify_rows chose " + std::to_string(value) +
                                              " where " + std::to_string(best) + " was possible");

    // The lanes' rows fill the step's in lane order.
    uint32_t first = 0, left = rows - total;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t more = std::min(cap - verified[lane], left);
      left -= more;
      require(laneRows[lane].first == first && laneRows[lane].rows == verified[lane] + more,
              label + ": lane " + std::to_string(lane) + "'s rows are not the step's next");
      first += laneRows[lane].rows;
    }
    require(first == rows, label + ": the lanes' rows do not fill the step's");

    // Each lane's verify rows in preorder, then its rows past them.
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const DrawnTree &t = trees[lane];
      const uint32_t kept = verified[lane];
      std::vector<uint32_t> order;
      std::function<void(uint32_t)> visit = [&](uint32_t row) {
        order.push_back(row);
        for (uint32_t child = row + 1; child < kept; ++child)
          if (t.parents[child] == row)
            visit(child);
      };
      visit(0);
      std::vector<uint32_t> placed(kept), depth(laneRows[lane].rows), ancestors(laneRows[lane].rows);
      for (uint32_t at = 0; at < kept; ++at)
        placed[order[at]] = at;
      for (uint32_t at = 0; at < laneRows[lane].rows; ++at) {
        const uint32_t row = at < kept ? order[at] : 0;
        const uint32_t parent = !at ? SPLASH_TREE_ROOT : at < kept ? placed[t.parents[row]] : at - 1;
        depth[at] = at ? depth[parent] + 1 : 0;
        ancestors[at] = (at ? ancestors[parent] : 0U) | (1U << at);
        const uint32_t slot = laneRows[lane].first + at;
        const TreeRow &got = static_cast<const TreeRow *>(b.treeRows.contents())[slot];
        require(static_cast<const uint32_t *>(b.inputTokens.contents())[slot] == t.tokens[row] &&
                    got.parent == parent && got.depth == depth[at] && got.ancestors == ancestors[at],
                label + ": lane " + std::to_string(lane) + "'s row " + std::to_string(at) +
                    (at < kept ? " is not its tree's in preorder" : " does not extend the row before it"));
      }
    }
  }
}

// A step's rows go to the likeliest rows of every lane's tree, whichever
// lane holds them: lanes that take more rows than others where their trees'
// estimates are larger, up to the cap.
void verifyRowsAcrossLanes(MetalBackend &backend) {
  constexpr uint32_t lanes = 2, rows = 12, cap = 8;
  DraftSelectorBuffers b{};
  b.treeTokens = allocate(backend, uint64_t{lanes} * kBlock * 4);
  b.trees = allocate(backend, uint64_t{lanes} * kBlock * sizeof(TreeRow));
  b.treeScores = allocate(backend, uint64_t{lanes} * kBlock * 8);
  b.treeSizes = allocate(backend, uint64_t{lanes} * 4);
  b.inputTokens = allocate(backend, uint64_t{rows} * 4);
  b.treeRows = allocate(backend, uint64_t{rows} * sizeof(TreeRow));
  b.laneRows = allocate(backend, uint64_t{lanes} * sizeof(LaneRows));
  b.verifyRows = allocate(backend, uint64_t{lanes} * 4);
  // Lane 0's chain stays likely (0.95 a step), lane 1's falls off at once
  // (0.3 a step).
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t size = std::min(kBlock, 8U);
    static_cast<uint32_t *>(b.treeSizes.contents())[lane] = size;
    float estimate = 1.0F;
    for (uint32_t row = 0; row < size; ++row) {
      const uint32_t at = lane * kBlock + row;
      static_cast<uint32_t *>(b.treeTokens.contents())[at] = 100 * (lane + 1) + row;
      static_cast<TreeRow *>(b.trees.contents())[at] = chainRow(row);
      static_cast<float *>(b.treeScores.contents())[2 * at + 1] = estimate;
      estimate *= lane ? 0.3F : 0.95F;
    }
  }
  CommandGraph graph;
  DraftSelector::addVerifyRows(graph, b, {lanes, rows, cap}, {});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *verified = static_cast<const uint32_t *>(b.verifyRows.contents());
  const auto *laneRows = static_cast<const LaneRows *>(b.laneRows.contents());
  // Lane 0's 8 rows (to the cap), lane 1's anchor and its 3 likeliest.
  require(verified[0] == std::min(cap, kBlock) && verified[1] == rows - verified[0],
          "the step's rows did not go to the likeliest rows across lanes");
  require(laneRows[0].first == 0 && laneRows[0].rows == verified[0] && laneRows[1].first == verified[0] &&
              laneRows[1].rows == verified[1],
          "the lanes' rows are not packed in lane order");
}

// Each buffer the selector and the verify rows reach, at its extent and one
// element short, for three lanes: every lane's eight query rows of logits and
// selector hidden rows, the workspaces of its seven positions, each
// codebook's row of every token, every lane's host chain and tree; and the
// step's rows, lane rows and verify rows.
void bufferExtents(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003, lanes = 3;
  Random random(0x657874);
  const std::array patterns{Pattern::Peaked};
  const Fixture f = makeFixture(backend, vocabulary, lanes, random, patterns, 0.1F);
  const DraftSelector selector(vocabulary);
  const std::array trees{DraftTreeLane{1, 3, 0, 2}, DraftTreeLane{2, kBlock - 1, 2, 1},
                         DraftTreeLane{3, kBlock - 1, 0, 4}};
  const uint64_t candidates = uint64_t{lanes} * kPositions * kCandidates;
  const uint64_t rows = uint64_t{lanes} * kBlock;
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
           {&DraftSelectorBuffers::chainTokens, uint64_t{lanes} * kChain * 4, 4, "draft tree chain token"},
           {&DraftSelectorBuffers::chainRates, uint64_t{lanes} * kChain * kAgreements * 4, 4, "draft tree chain rate"},
           {&DraftSelectorBuffers::treeTokens, rows * 4, 4, "draft tree token"},
           {&DraftSelectorBuffers::trees, rows * sizeof(TreeRow), sizeof(TreeRow), "draft tree row"},
           {&DraftSelectorBuffers::treeScores, rows * 8, 4, "draft tree score"},
           {&DraftSelectorBuffers::treeSizes, uint64_t{lanes} * 4, 4, "draft tree size"}})
    requireExtent(backend, f.buffers.*member, bytes, element, what, [&](CommandGraph &graph, const MetalBuffer &buffer) {
      DraftSelectorBuffers changed = f.buffers;
      changed.*member = buffer;
      selector.add(graph, changed, f.codebooks, trees);
    });
  for (const auto &[member, what] :
       {std::pair{&DraftCodebooks::predecessor, "draft predecessor codebook"},
        std::pair{&DraftCodebooks::successor, "draft successor codebook"}})
    requireExtent(backend, f.codebooks.*member, uint64_t{vocabulary} * kRank * 2, 2, what,
                  [&](CommandGraph &graph, const MetalBuffer &buffer) {
                    DraftCodebooks changed = f.codebooks;
                    changed.*member = buffer;
                    selector.add(graph, f.buffers, changed, trees);
                  });
  // draft_verify_rows reads every lane's tree and writes the step's rows.
  const StepRows step{lanes, 2 * lanes + 1, 3};
  for (const auto &[member, bytes, element, what] :
       std::initializer_list<std::tuple<MetalBuffer DraftSelectorBuffers::*, uint64_t, uint64_t, const char *>>{
           {&DraftSelectorBuffers::treeTokens, rows * 4, 4, "draft tree token"},
           {&DraftSelectorBuffers::trees, rows * sizeof(TreeRow), sizeof(TreeRow), "draft tree row"},
           {&DraftSelectorBuffers::treeScores, rows * 8, 4, "draft tree score"},
           {&DraftSelectorBuffers::treeSizes, uint64_t{lanes} * 4, 4, "draft tree size"},
           {&DraftSelectorBuffers::inputTokens, uint64_t{step.rows} * 4, 4, "verify input token"},
           {&DraftSelectorBuffers::treeRows, uint64_t{step.rows} * sizeof(TreeRow), sizeof(TreeRow),
            "verify tree row"},
           {&DraftSelectorBuffers::laneRows, uint64_t{lanes} * sizeof(LaneRows), sizeof(LaneRows), "verify lane rows"},
           {&DraftSelectorBuffers::verifyRows, uint64_t{lanes} * 4, 4, "verify rows"}})
    requireExtent(backend, f.buffers.*member, bytes, element, what, [&](CommandGraph &graph, const MetalBuffer &buffer) {
      DraftSelectorBuffers changed = f.buffers;
      changed.*member = buffer;
      DraftSelector::addVerifyRows(graph, changed, step, {});
    });
}

void invalidRequests(MetalBackend &backend) {
  rejects([] { DraftSelector(0); }, "invalid draft selector vocabulary",
          "a selector over no vocabulary was accepted");
  constexpr uint32_t vocabulary = 1024;
  Random random(0x696e76);
  const std::array patterns{Pattern::Peaked};
  const Fixture f = makeFixture(backend, vocabulary, 2, random, patterns, 0.1F);
  const DraftSelector selector(vocabulary);
  CommandGraph graph;
  const auto add = [&](std::span<const DraftTreeLane> lanes) { selector.add(graph, f.buffers, f.codebooks, lanes); };
  rejects([&] { add({}); }, "invalid draft selector batch", "a selection of no lane was accepted");
  const std::array<DraftTreeLane, 5> five{};
  rejects([&] { add(five); }, "invalid draft selector batch", "a selection of five lanes was accepted");
  // A tree deeper than a block's rows, a chain past a block's rows after its
  // anchor and more children than the kernel ranks.
  for (const DraftTreeLane &bad : {DraftTreeLane{1, kBlock, 0, 1}, DraftTreeLane{1, 3, kChain + 1, 1},
                                   DraftTreeLane{1, 3, 0, SPLASH_TREE_CHILDREN + 1}})
    rejects([&] { add(std::array{bad}); }, "invalid draft tree lane", "an impossible tree lane was accepted");
  DraftCalibration slope, intercept;
  slope.slope = NAN;
  intercept.intercepts[3] = INFINITY;
  for (const DraftCalibration &bad : {slope, intercept})
    rejects([&] { add(std::array{DraftTreeLane{1, 3, 0, 1}, DraftTreeLane{1, 3, 0, 1, bad}}); },
            "invalid draft tree calibration", "an invalid calibration was accepted");
  // Steps of no lane or five, of no cap or one past a block's rows, and of
  // fewer rows than their anchors or more than their caps hold.
  for (const StepRows &bad : {StepRows{0, 1, 1}, StepRows{5, 5, 1}, StepRows{2, 4, 0}, StepRows{2, 4, kBlock + 1},
                              StepRows{2, 1, 2}, StepRows{2, 5, 2}})
    rejects([&] { DraftSelector::addVerifyRows(graph, f.buffers, bad, {}); }, "invalid decode step rows",
            "impossible step rows were encoded");
  const StepRows two{2, 2 * kBlock, kBlock};
  std::vector<float> costs(2 * kBlock + 1, 1.0F);
  rejects([&] { DraftSelector::addVerifyRows(graph, f.buffers, two, std::span(costs).first(2 * kBlock)); },
          "invalid draft verify rows cost", "costs short of every total were accepted");
  for (const float bad : {0.0F, -1.0F, NAN}) {
    std::vector<float> wrong = costs;
    wrong[2 * kBlock] = bad;
    rejects([&] { DraftSelector::addVerifyRows(graph, f.buffers, two, wrong); }, "invalid draft verify rows cost",
            "a cost that is not positive was accepted");
  }
  require(graph.empty(), "an invalid draft selector request encoded a graph");
  // Costs below the lanes' anchors are not read.
  costs[0] = costs[1] = 0.0F;
  CommandGraph encoded;
  DraftSelector::addVerifyRows(encoded, f.buffers, two, costs);
  require(!encoded.empty(), "valid verify rows were not encoded");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: draft-selector METALLIB");
    MetalBackend backend(argv[1]);
    invalidRequests(backend);
    bufferExtents(backend);
    for (const uint32_t vocabulary : {248320U, 1003U, 270005U})
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
        chainCase(backend, vocabulary, lanes);
    treesMatchTheReference(backend);
    forcedChain(backend);
    verifyRowsLayout(backend);
    verifyRowsAcrossLanes(backend);
    std::cout << "draft_selector_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_selector_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
