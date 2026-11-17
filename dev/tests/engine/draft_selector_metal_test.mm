// DFlash draft selector against direct CPU references. The sharded top-16
// scan and its reduce must return exactly the sixteen largest logits of every
// proposal row in (value desc, id asc) order, and each edge of the codebook
// table its double-precision product. The vocabularies cover the production
// size, an odd size that misaligns the 16-byte vectors and leaves shards
// with only a few tokens, and one wider than a single register chunk per
// thread; over a subset of the vocabulary (DraftVocabulary), its logits rows
// hold the subset's tokens in order and the candidates are those tokens,
// whose codebook rows the edges read, the anchor's whether or not the subset
// holds it. The logits are fp32, and their order is decided below the bf16
// spacing.
//
// The tree each lane grows from the candidates (draft_select_tree) must be a
// brute-force reference's, row for row: every path that takes one of a
// row's `children` likeliest candidates a step and every prefix of the lane's
// host chain, scored by the product of its steps' estimates (a draft step's
// by its lane's own calibration, at most what its parent's chain child and
// earlier draws left of 1; a sampled lane's row takes its candidates of the
// largest log q_T + G, G the Gumbel of the row's key), ranked, one row for
// each token path, as many as a verify block holds and its depth plus one
// (the tokens a walk of it may retain), and none deeper than its depth: its
// tokens, parents, depths, ancestors, draft probabilities (a host chain's row
// takes that of a draft step that joins it, none otherwise), estimates and
// caps, a row's children within its estimate. A tree of one child a row and
// no chain is the chain the selector walked before it built trees, token for
// token, from the same scores. A host chain whose tokens the target selects
// whatever its logits, with no draft children, takes the first rows, and a
// chain's child leaves its drawn siblings what its rate leaves of 1.
// draft_verify_rows
// must lay out the rows of every lane whose expected tokens are the best of
// every assignment the step's rows, the lanes' cap and its floor allow (no
// row past an anchor estimated below the floor: one at it is taken, one just
// under it is not, and an anchor whatever its estimate), each lane exactly
// the rows it took, packed in lane order, and the step's rows past them of
// no lane, each lane 0's anchor and no source; and number a lane's rows in
// preorder, the earlier-taken child first.
#include "ChainRows.hpp"
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
#include <limits>
#include <map>
#include <set>
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
using splash::test::chainRow;
using splash::test::floatToBf16;

constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kRank = SPLASH_DRAFT_SELECTOR_RANK;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kBlock = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kChain = SPLASH_TREE_CHAIN_TOKENS;
constexpr uint32_t kAgreements = SPLASH_TREE_AGREEMENTS;
constexpr uint32_t kChildren = SPLASH_TREE_CHILDREN;

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

// Every lane's draft rows of `lanes` lanes over `subset` of the vocabulary, a
// selector's buffers over them and the codebooks, with each lane's tree of up
// to kBlock rows at lane x kBlock.
struct Fixture final {
  uint32_t vocabulary = 0;
  uint32_t lanes = 0;
  DraftVocabulary subset{};
  MetalBuffer logits;
  DraftSelectorBuffers buffers;
  DraftCodebooks codebooks;

  [[nodiscard]] const float *logitRow(uint32_t lane, uint32_t position) const {
    return static_cast<const float *>(logits.contents()) +
           (uint64_t{lane} * kRows + position + 1) * subset.columns(vocabulary);
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
  [[nodiscard]] const uint8_t *draws() const { return static_cast<const uint8_t *>(buffers.treeDraws.contents()); }
  [[nodiscard]] const float *next() const { return static_cast<const float *>(buffers.treeNext.contents()); }
  [[nodiscard]] const float *caps() const { return static_cast<const float *>(buffers.treeCaps.contents()); }
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

// Each lane's draft rows of `patterns` (by position) over `subset` and its
// selector hidden rows and the codebooks of scale `edgeScale`: the larger,
// the more an edge moves a candidate's probability.
Fixture makeFixture(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes, Random &random,
                    std::span<const Pattern> patterns, float edgeScale, DraftVocabulary subset = {}) {
  Fixture f;
  f.vocabulary = vocabulary;
  f.lanes = lanes;
  f.subset = subset;
  const uint32_t rows = lanes * kRows, columns = subset.columns(vocabulary);
  const auto workspace = DraftSelector::workspace(lanes * kPositions);
  f.logits = allocate(backend, uint64_t{rows} * columns * sizeof(float));
  auto *logitRows = static_cast<float *>(f.logits.contents());
  for (uint32_t row = 0; row < rows; ++row)
    fillRow(logitRows + uint64_t{row} * columns, columns,
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
               allocate(backend, uint64_t{lanes} * kBlock * kChildren),
               allocate(backend, uint64_t{lanes} * kBlock * kCandidates * sizeof(float)),
               allocate(backend, uint64_t{lanes} * kBlock * sizeof(float)),
               allocate(backend, uint64_t{lanes} * kBlock * sizeof(uint32_t)),
               allocate(backend, uint64_t{lanes} * kBlock * sizeof(TreeRow)),
               allocate(backend, uint64_t{lanes} * sizeof(LaneRows)),
               allocate(backend, uint64_t{lanes} * kBlock * sizeof(uint32_t))};
  f.codebooks = {randomBfloat(backend, uint64_t{vocabulary} * kRank, random, edgeScale),
                 randomBfloat(backend, uint64_t{vocabulary} * kRank, random, edgeScale)};
  return f;
}


void run(MetalBackend &backend, const DraftSelector &selector, const Fixture &f, std::span<const DraftTreeLane> lanes) {
  CommandGraph graph;
  selector.add(graph, f.buffers, f.codebooks, f.subset, lanes);
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

// The selector's noise (DraftTreeParams): splitmix64, a row's key from its
// parent's and its token, and the Gumbel of a candidate's token under a row's
// key, from the hash's top 23 bits as the kernel takes them.
uint64_t treeHash(uint64_t value) {
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

uint64_t treeKey(uint64_t parent, uint32_t token) {
  return treeHash(parent + 0x9e3779b97f4a7c15ULL * (uint64_t{token} + 1));
}

double treeGumbel(uint64_t key, uint32_t token) {
  const uint64_t value = treeHash(key ^ 0xd1b54a32d192ed03ULL * (uint64_t{token} + 1));
  const float uniform = (float(value >> 41) + 0.5F) * 0x1p-23F;
  return -std::log(-std::log(double(uniform)));
}

// A lane's draft probability of a candidate after a predecessor: q for a
// greedy lane, q_T for a sampled one, each candidate weighed against the most
// likely, exp((log q - log q_max) / T), none where that weight lies below
// fp32's normal range, which the kernel flushes.
double draftProbability(const std::array<double, kCandidates> &q, const DraftTreeLane &lane, uint32_t index) {
  if (lane.temperature == 0.0F)
    return q[index];
  const double top = *std::max_element(q.begin(), q.end());
  const auto weight = [&](double value) {
    const double w = value > 0.0 ? std::exp((std::log(value) - std::log(top)) / lane.temperature) : 0.0;
    return w < std::numeric_limits<float>::min() ? 0.0 : w;
  };
  double mass = 0.0;
  for (const double value : q)
    mass += weight(value);
  return weight(q[index]) / mass;
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
// made; a sampled lane's q_T) and estimate.
struct Node final {
  std::vector<uint32_t> path;
  uint32_t parent = 0;
  double q = 1.0;
  double estimate = 1.0;
  // The row's draws, in order, and its next position's probabilities; none
  // for a row the selector did not expand.
  std::vector<uint32_t> draws;
  std::vector<double> next;
  // The cap of the draft step to it: what its parent's host chain child and
  // earlier draws left of 1; 1 for a row of no draft probability.
  double cap = 1.0;
};

// The brute-force tree of a lane, rows in the order the selector takes them;
// none when two paths' scores, or two of a sampled row's candidates' keys,
// lie too close for float sums to order (the caller draws other data). A
// sampled lane's row takes its candidates of the largest log q_T + G, G the
// Gumbel of its key, which its path's tokens fold from the lane's noise.
std::optional<std::vector<Node>> referenceTree(const Lattice &l, const DraftTreeLane &lane, const Chain &chain) {
  struct Entry final {
    std::vector<uint32_t> path;
    double score;
    bool chain;
    double cap = 1.0;
  };
  std::vector<Entry> entries;
  // The paths whose sampled expansion drew among keys too close to order.
  std::set<std::vector<uint32_t>> ambiguous;
  // Each expanded path's draws, in order, and next position's probabilities.
  std::map<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, std::vector<double>>> expansions;
  // Draft paths: a row's `children` likeliest candidates after its own, or a
  // sampled lane's draws.
  std::function<void(std::vector<uint32_t> &, uint64_t, uint32_t, uint32_t, double)> expand =
      [&](std::vector<uint32_t> &path, uint64_t key, uint32_t position, uint32_t predecessor, double score) {
        const auto &q = l.q[position][predecessor];
        std::vector<uint32_t> taken;
        if (lane.temperature == 0.0F) {
          taken.assign(l.ranked[position][predecessor].begin(), l.ranked[position][predecessor].begin() + lane.children);
        } else {
          std::vector<std::pair<double, uint32_t>> keys;
          for (uint32_t i = 0; i < kCandidates; ++i)
            if (const double qt = draftProbability(q, lane, i); qt > 0.0)
              keys.emplace_back(std::log(qt) + treeGumbel(key, l.ids[position][i]), i);
          std::sort(keys.begin(), keys.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
          for (size_t i = 0; i < keys.size() && i < lane.children; ++i) {
            taken.push_back(keys[i].second);
            if (i + 1 < keys.size() && keys[i].first - keys[i + 1].first < 1e-4)
              ambiguous.insert(path);
          }
        }
        auto &[draws, next] = expansions[path];
        for (uint32_t i = 0; i < kCandidates; ++i)
          next.push_back(draftProbability(q, lane, i));
        // A child's own estimate is at most what the row's host chain child
        // and the children drawn before it left of 1, and the child is
        // estimated at most as the one drawn before it. A row on the chain's
        // path has the chain's next token for a child, whose rate counts
        // first; a drawn child of that token is it, capped by what the others
        // left, and takes the more of its own estimate and the rate.
        const bool chained = position < lane.chain && std::equal(path.begin(), path.end(), chain.tokens.begin());
        const double rate = chained ? chain.rates[position][agreement(l, chain.tokens[position], position)] : 0.0;
        double bound = score, left = 1.0 - rate, reserved = rate;
        for (const uint32_t index : taken) {
          if (!(q[index] > 0.0))
            break;
          draws.push_back(index);
          const uint32_t token = l.ids[position][index];
          const bool joinsChain = chained && token == chain.tokens[position];
          const double cap = joinsChain ? left + reserved : left;
          path.push_back(token);
          const double own = std::min(estimate(lane.calibration, draftProbability(q, lane, index), position), cap);
          bound = std::min(bound, score * own);
          entries.push_back({path, bound, false, cap});
          left = joinsChain ? cap - std::max(own, reserved) : left - own;
          reserved = joinsChain ? 0.0 : reserved;
          if (position + 1 < lane.depth && position + 1 < kPositions)
            expand(path, treeKey(key, token), position + 1, index, bound);
          path.pop_back();
        }
      };
  std::vector<uint32_t> path;
  if (lane.depth > 0)
    expand(path, lane.noise, 0, 0, 1.0);
  double score = 1.0;
  for (uint32_t depth = 0; depth < lane.chain && depth < lane.depth; ++depth) {
    score *= chain.rates[depth][agreement(l, chain.tokens[depth], depth)];
    entries.push_back({std::vector<uint32_t>(chain.tokens.begin(), chain.tokens.begin() + depth + 1), score, true});
  }
  std::stable_sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.score > b.score; });

  // A row's expansion: a draft entry's, when the selector takes it.
  const auto expanded = [&](Node &node) {
    if (const auto found = expansions.find(node.path); found != expansions.end())
      std::tie(node.draws, node.next) = found->second;
  };
  std::vector<Node> rows{Node{}};
  if (lane.depth > 0)
    expanded(rows[0]);
  std::map<std::vector<uint32_t>, uint32_t> made{{{}, 0}};
  for (size_t index = 0; index < entries.size() && rows.size() < std::min(kBlock, lane.depth + 1); ++index) {
    const Entry &entry = entries[index];
    // Float sums decide between paths within this of each other, and the
    // selector's heap orders paths of one estimate (0 among them, as a chain
    // child's siblings may be) as it pushed them, but for a row's children
    // of one estimate, the bound of the one drawn before, which the order
    // drawn ranks.
    if (index + 1 < entries.size()) {
      const Entry &after = entries[index + 1];
      const bool siblings = after.path.size() == entry.path.size() &&
                            std::equal(entry.path.begin(), entry.path.end() - 1, after.path.begin());
      if (after.path != entry.path && after.score >= entry.score * (1.0 - 1e-4) &&
          !(siblings && after.score == entry.score))
        return std::nullopt;
    }
    const uint32_t position = static_cast<uint32_t>(entry.path.size()) - 1;
    // A token's candidate index at its position, or none.
    const auto candidate = [&](uint32_t at, uint32_t token) -> int {
      if (at >= kPositions)
        return -1;
      const auto found = std::find(l.ids[at].begin(), l.ids[at].end(), token);
      return found == l.ids[at].end() ? -1 : int(found - l.ids[at].begin());
    };
    // A draft entry's probability: its token's after its parent's.
    const auto drafted = [&] {
      const int own = candidate(position, entry.path.back());
      const int predecessor = position ? candidate(position - 1, entry.path[position - 1]) : 0;
      return draftProbability(l.q[position][predecessor], lane, own);
    };
    if (const auto found = made.find(entry.path); found != made.end()) {
      Node &row = rows[found->second];
      // A draft step that joins a host chain's row gives it its probability.
      if (!entry.chain && row.q < 0.0) {
        row.q = drafted();
        row.cap = entry.cap;
      }
      if (!entry.chain && row.draws.empty())
        expanded(row);
      continue;
    }
    Node node;
    node.path = entry.path;
    node.parent = made.at(std::vector<uint32_t>(entry.path.begin(), entry.path.end() - 1));
    node.estimate = entry.score;
    node.q = entry.chain ? -1.0 : drafted();
    node.cap = entry.chain ? 1.0 : entry.cap;
    if (!entry.chain)
      expanded(node);
    made.emplace(entry.path, static_cast<uint32_t>(rows.size()));
    rows.push_back(node);
  }
  // A row's children follow from its expansion.
  for (const Node &node : rows)
    if (ambiguous.contains(node.path))
      return std::nullopt;
  return rows;
}

// The target selects one token after a row, so that the estimates of a
// lane's rows' children, its host chain's and its draft's, sum to at most
// their parent's.
void requireChildrenWithinParents(const Fixture &f, uint32_t lane, const std::string &label) {
  const uint32_t first = lane * kBlock, rows = f.rowCount(lane);
  std::vector<double> children(rows, 0.0);
  for (uint32_t row = 1; row < rows; ++row)
    children[f.tree()[first + row].parent] += f.scores()[2 * (first + row) + 1];
  for (uint32_t row = 0; row < rows; ++row) {
    const double own = f.scores()[2 * (first + row) + 1];
    require(children[row] <= own * (1.0 + 1e-5),
            label + ": row " + std::to_string(row) + "'s children are estimated " + std::to_string(children[row]) +
                " together, past its " + std::to_string(own));
  }
}

// A lane's tree from the selector as `rows`, in its order: tokens, parents,
// depths, ancestors, draft probabilities and estimates, and its size; its
// rows' children within their parents.
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
    require(std::fabs(f.caps()[first + row] - node.cap) <= 1e-5,
            label + ": row " + std::to_string(row) + "'s cap " + std::to_string(f.caps()[first + row]) + " is not " +
                std::to_string(node.cap));
    // Its draws, in order, and its next position's probabilities.
    for (uint32_t child = 0; child < kChildren; ++child)
      require(f.draws()[(first + row) * kChildren + child] ==
                  (child < node.draws.size() ? node.draws[child] : 0xffU),
              label + ": row " + std::to_string(row) + "'s draw " + std::to_string(child) + " differs");
    for (uint32_t i = 0; i < kCandidates; ++i) {
      const double want = node.next.empty() ? 0.0 : node.next[i];
      require(std::fabs(f.next()[(first + row) * kCandidates + i] - want) <= 1e-4 * want + 1e-7,
              label + ": row " + std::to_string(row) + "'s next probability " + std::to_string(i) + " differs");
    }
  }
  requireChildrenWithinParents(f, lane, label);
}

// The selector's top-16 candidates and their edge table over production-like
// and adversarial rows, and a tree of one child a row and no chain: the walk
// the chain selector took, its proposal at each position the candidate of
// the largest score (unary plus edge, in fp32 as the kernel adds them) after
// the previous one's, the lowest index among equal ones, and each row's draft
// probability the largest of its candidates', the chain's confidence. Over
// `tokens` of the vocabulary the candidates are the tokens of the rows'
// largest columns.
void chainCase(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes, DraftVocabulary tokens = {}) {
  Random random(0x5e1ec7 + uint64_t{vocabulary} * 8 + lanes * 2 + uint64_t{tokens.leading} * 16);
  const uint32_t columns = tokens.columns(vocabulary);
  const std::array patterns{Pattern::Peaked, Pattern::Uniform, Pattern::Ties, Pattern::Sparse};
  const Fixture f = makeFixture(backend, vocabulary, lanes, random, patterns, 0.1F, tokens);
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
      const auto expected = referenceTop16(row, columns);
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t id = f.candidates(lane)[position * kCandidates + rank];
        const float value = f.unary(lane)[position * kCandidates + rank];
        if (rank < expected.size())
          require(id == tokens.token(expected[rank]) && value == row[expected[rank]],
                  "draft top-16 candidates differ from the exact sorted order of their tokens");
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
// their trees at a depth, and at its rows plus one, or take all of their
// rows.
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
      // Greedy lanes, and lanes sampling at temperatures 0.7 and 1.
      const float temperature = std::array{0.0F, 0.7F, 1.0F}[(trial / 2 + lane) % 3];
      trees.push_back({random.next() % vocabulary, depth, chain, children,
                       (trial + lane) % 2 ? calibrated : DraftCalibration{}, temperature,
                       uint64_t{random.next()} << 32 | random.next()});
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

// A sampled lane's proposal row that is not finite (its logits and its
// selector hidden row NaN at one position) draws nothing there: every row of
// its tree at that depth holds no draws, so that no verify row tests a child
// of it and none of its candidates, 0xffffffff where the row has fewer
// comparable logits than candidates, is drawn; and no draft row lies past
// that depth.
void nonFiniteRowDrawsNothing(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003, kNonFinite = 1;
  Random random(0x6e616e72);
  const std::array patterns{Pattern::Peaked};
  const Fixture f = makeFixture(backend, vocabulary, 1, random, patterns, 0.4F);
  std::fill_n(static_cast<float *>(f.logits.contents()) + uint64_t{kNonFinite + 1} * vocabulary, vocabulary,
              std::numeric_limits<float>::quiet_NaN());
  std::fill_n(static_cast<uint16_t *>(f.buffers.selectorHidden.contents()) + uint64_t{kNonFinite + 1} * kRank, kRank,
              uint16_t{0x7fc0});
  const DraftSelector selector(vocabulary);
  run(backend, selector, f, std::array{DraftTreeLane{7, kBlock - 1, 0, kChildren, DraftCalibration{}, 0.7F, 0x5eed}});
  bool none = f.rowCount(0) > 1;
  for (uint32_t row = 0; row < f.rowCount(0); ++row) {
    none = none && f.tree()[row].depth <= kNonFinite;
    for (uint32_t child = 0; f.tree()[row].depth == kNonFinite && child < kChildren; ++child)
      none = none && f.draws()[row * kChildren + child] == 0xff;
  }
  require(none, "a sampled lane drew a child of a proposal row that is not finite");
}

// A sampled lane's draws, over many cycles' noise, follow its draft at its
// temperature, without replacement: of one child a row, the anchor's child is
// candidate i as often as q_T(i); of two, the pair {i, j} as often as
// q_T(i) q_T(j) (1 / (1 - q_T(i)) + 1 / (1 - q_T(j))). A chi-square over the
// outcomes expected at least 5 times (the others pooled) stays within five
// standard deviations of its degrees of freedom.
void sampledChildrenDistribution(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003, kRuns = 1500;
  Random random(0x6e6f6973);
  // Flat rows, whose candidates share their mass.
  const std::array patterns{Pattern::Uniform};
  const Fixture f = makeFixture(backend, vocabulary, kLanes, random, patterns, 0.4F);
  const DraftSelector selector(vocabulary);
  for (const auto &[children, temperature] : {std::pair{1U, 0.7F}, std::pair{2U, 1.0F}}) {
    std::vector<std::map<std::vector<uint32_t>, uint32_t>> counts(kLanes);
    for (uint32_t runs = 0; runs < kRuns; ++runs) {
      // A tree as deep as the anchor's children are many holds them and its
      // anchor (its depth's rows and one), each likelier than their own
      // children over flat rows.
      std::vector<DraftTreeLane> trees;
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        trees.push_back({7, children, 0, children, DraftCalibration{}, temperature,
                         treeHash(uint64_t{runs} * kLanes + lane + 1)});
      run(backend, selector, f, trees);
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        require(f.rowCount(lane) == children + 1, "a sampled anchor did not take its children");
        std::vector<uint32_t> drawn(f.tokens() + lane * kBlock + 1, f.tokens() + lane * kBlock + 1 + children);
        std::ranges::sort(drawn);
        ++counts[lane][drawn];
      }
    }
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      const Lattice l = latticeOf(f, lane);
      DraftTreeLane sampled;
      sampled.temperature = temperature;
      std::array<double, kCandidates> qt{};
      for (uint32_t i = 0; i < kCandidates; ++i)
        qt[i] = draftProbability(l.q[0][0], sampled, i);
      std::map<std::vector<uint32_t>, double> expected;
      for (uint32_t i = 0; i < kCandidates; ++i) {
        if (children == 1) {
          expected[{l.ids[0][i]}] += qt[i];
          continue;
        }
        for (uint32_t j = i + 1; j < kCandidates; ++j) {
          std::vector<uint32_t> pair{l.ids[0][i], l.ids[0][j]};
          std::ranges::sort(pair);
          expected[pair] += qt[i] * qt[j] * (1.0 / (1.0 - qt[i]) + 1.0 / (1.0 - qt[j]));
        }
      }
      double chi = 0.0, pooledSeen = 0.0, pooledExpected = 0.0;
      uint32_t bins = 0;
      for (const auto &[outcome, probability] : expected) {
        const double want = probability * kRuns;
        const auto found = counts[lane].find(outcome);
        const double seen = found == counts[lane].end() ? 0.0 : found->second;
        if (want < 5.0) {
          pooledSeen += seen;
          pooledExpected += want;
          continue;
        }
        chi += (seen - want) * (seen - want) / want;
        ++bins;
      }
      if (pooledExpected > 0.0) {
        chi += (pooledSeen - pooledExpected) * (pooledSeen - pooledExpected) / pooledExpected;
        ++bins;
      }
      const double freedom = bins - 1.0;
      require(bins > 2 && chi < freedom + 5.0 * std::sqrt(2.0 * freedom),
              "sampled children of " + std::to_string(children) + " at temperature " + std::to_string(temperature) +
                  ": chi-square " + std::to_string(chi) + " over " + std::to_string(freedom) + " degrees of freedom");
    }
  }
}

// At temperature 0.01, where q^(1/T) underflows fp32 for every candidate of a
// row whose most likely holds less than 0.4, a sampled lane's q_T stays
// finite and sums to 1, and every child it draws has q_T above 0: over rows
// whose sixteen candidates score alike (lanes 0 and 2) and rows whose first
// candidate scores two above the other fifteen (lanes 1 and 3).
void lowTemperatureDraws(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  Random random(0x636f6c64);
  const std::array patterns{Pattern::Peaked};
  const Fixture f = makeFixture(backend, vocabulary, kLanes, random, patterns, 0.0F);
  auto *logits = static_cast<float *>(f.logits.contents());
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    for (uint32_t row = 0; row < kRows; ++row) {
      float *values = logits + (uint64_t{lane} * kRows + row) * vocabulary;
      std::fill(values, values + vocabulary, -8.0F);
      for (uint32_t i = 0; i < kCandidates; ++i)
        values[(37 * (lane + 1) + 61 * i) % vocabulary] = i == 0 && lane % 2 ? 7.0F : 5.0F;
    }
  const DraftSelector selector(vocabulary);
  std::vector<DraftTreeLane> trees;
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    trees.push_back({7, kBlock - 1, 0, kChildren, DraftCalibration{}, 0.01F, treeHash(lane + 1)});
  run(backend, selector, f, trees);
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    require(f.rowCount(lane) > 1, "a sampled lane at temperature 0.01 drew no child");
    for (uint32_t row = 0; row < f.rowCount(lane); ++row) {
      const uint32_t at = lane * kBlock + row;
      const uint8_t *draws = f.draws() + uint64_t{at} * kChildren;
      if (draws[0] == 0xff)
        continue;
      const float *next = f.next() + uint64_t{at} * kCandidates;
      const std::string label = "lane " + std::to_string(lane) + " row " + std::to_string(row);
      double mass = 0.0;
      for (uint32_t i = 0; i < kCandidates; ++i) {
        require(std::isfinite(next[i]) && next[i] >= 0.0F,
                label + ": q_T of candidate " + std::to_string(i) + " is " + std::to_string(next[i]));
        mass += next[i];
      }
      require(std::fabs(mass - 1.0) <= 1e-5, label + ": q_T sums to " + std::to_string(mass));
      for (uint32_t child = 0; child < kChildren && draws[child] != 0xff; ++child)
        require(next[draws[child]] > 0.0F, label + ": drew a child of q_T 0");
    }
  }
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
    // The anchor draws none of its candidates, whose probabilities it holds.
    std::vector<Node> rows{Node{}};
    const Lattice l = latticeOf(f, lane);
    rows[0].next.assign(l.q[0][0].begin(), l.q[0][0].end());
    for (uint32_t d = 0; d + 1 < made; ++d)
      rows.push_back({std::vector<uint32_t>(end.begin(), end.begin() + d + 1), d, -1.0, 1.0, {}, {}});
    requireTree(f, lane, trees[lane].anchor, rows, "forced chain lane " + std::to_string(lane));
  }
}

// A host chain that copies the draft's likeliest path, at rates above the
// draft's first step's estimate, makes its first row before the draft's
// likeliest first step joins it, and the draft's later steps join its other
// rows: a chain's row takes the probability of the draft step that joins it,
// as the reference has it, in either lane.
void chainRowsTheDraftJoins(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  constexpr float kRate = 0.995F;
  const std::array patterns{Pattern::Peaked};
  uint32_t checked = 0;
  for (uint32_t trial = 0; checked < 8; ++trial) {
    require(trial < 64, "the reference found too few unambiguous joined chains");
    Random random(0x6a6f696e + trial);
    const Fixture f = makeFixture(backend, vocabulary, 2, random, patterns, 0.4F);
    const DraftSelector selector(vocabulary);
    std::array trees{DraftTreeLane{random.next() % vocabulary, kBlock - 1, kPositions, 2},
                     DraftTreeLane{random.next() % vocabulary, kBlock - 1, kPositions, 2}};
    run(backend, selector, f, trees);
    std::array<Chain, 2> chains;
    bool chainFirst = true;
    for (uint32_t lane = 0; lane < 2; ++lane) {
      const Lattice l = latticeOf(f, lane);
      // The first draft step's estimate is its q, the calibration the
      // identity: the chain's first row comes first below its rate.
      chainFirst = chainFirst && l.q[0][0][l.ranked[0][0][0]] < kRate - 0.002;
      uint32_t predecessor = 0;
      for (uint32_t d = 0; d < kPositions; ++d) {
        const uint32_t index = l.ranked[d][predecessor][0];
        chains[lane].tokens.push_back(l.ids[d][index]);
        chains[lane].rates.push_back({kRate, kRate, kRate, kRate});
        predecessor = index;
      }
      std::copy(chains[lane].tokens.begin(), chains[lane].tokens.end(), f.chainTokens(lane));
      std::fill_n(f.chainRates(lane), kPositions * kAgreements, kRate);
    }
    if (!chainFirst)
      continue;
    run(backend, selector, f, trees);
    std::array<std::optional<std::vector<Node>>, 2> references;
    for (uint32_t lane = 0; lane < 2; ++lane)
      references[lane] = referenceTree(latticeOf(f, lane), trees[lane], chains[lane]);
    if (!references[0] || !references[1])
      continue;
    ++checked;
    for (uint32_t lane = 0; lane < 2; ++lane)
      requireTree(f, lane, trees[lane].anchor, *references[lane],
                  "joined chain trial " + std::to_string(trial) + " lane " + std::to_string(lane));
    // Each chain made its lane's row 1, and the draft's likeliest first step
    // joined it.
    for (uint32_t lane = 0; lane < 2; ++lane) {
      const float joined = f.scores()[2 * (lane * kBlock + 1)];
      require(f.tokens()[lane * kBlock + 1] == chains[lane].tokens[0] && joined > 0.0F && joined <= 1.0F,
              "a chain's row the draft joined did not take its draft probability");
    }
  }
}

// The estimates of the children a row drew sum to at most its own: of a
// greedy anchor whose candidates its calibration estimates alike (slope 0),
// at 0.3 each, each child is estimated at what the children before it left
// of 1, its cap: 0.3, 0.3, 0.3 and 0.1 of caps 1, 0.7, 0.4 and 0.1, each
// likelier than any of their own children (0.09), so that a tree as deep as
// the children are many holds them all (its depth's rows and one).
void siblingsShareTheirParent(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  Random random(0x73686172);
  const std::array patterns{Pattern::Uniform};
  const Fixture f = makeFixture(backend, vocabulary, 1, random, patterns, 0.1F);
  const DraftSelector selector(vocabulary);
  DraftCalibration flat;
  flat.slope = 0.0F;
  flat.intercepts.fill(std::log(0.3F / 0.7F));
  run(backend, selector, f, std::array{DraftTreeLane{7, kChildren, 0, kChildren, flat}});
  require(f.rowCount(0) == kChildren + 1, "the anchor did not take its children");
  double left = 1.0;
  for (uint32_t row = 1; row <= kChildren; ++row) {
    const double want = std::min(0.3, left);
    require(std::fabs(f.scores()[2 * row + 1] - want) <= 1e-5 && std::fabs(f.caps()[row] - left) <= 1e-5,
            "child " + std::to_string(row) + " is estimated at " + std::to_string(f.scores()[2 * row + 1]) +
                " of cap " + std::to_string(f.caps()[row]) + ", not " + std::to_string(want) + " of " +
                std::to_string(left));
    left -= want;
  }
}

// A row's host chain child counts against the caps of the children its draft
// drew: of an anchor whose candidates a calibration estimates alike (slope
// 0) at 0.3, beside a one-token chain at rate 0.97, lane 0's the draft's
// likeliest token and lane 1's a token none of its candidates hold, the
// first draft row beside the chain's row is capped at and estimated what the
// chain left of 1, 0.03, and the draft's others at 0, so that the anchor's
// children sum to at most its 1; lane 0's likeliest step, which joins the
// chain's row, is that child, capped at 1. A step deeper the draft estimates
// its rows at 0.01, so that a tree of its depth's rows and one, 4, holds the
// chain's row and the first draft row beside it.
void chainCountsAgainstItsSiblings(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  constexpr float kRate = 0.97F;
  Random random(0x63686169);
  const std::array patterns{Pattern::Uniform};
  const Fixture f = makeFixture(backend, vocabulary, 2, random, patterns, 0.1F);
  const DraftSelector selector(vocabulary);
  DraftCalibration flat;
  flat.slope = 0.0F;
  flat.intercepts.fill(std::log(0.01F / 0.99F));
  flat.intercepts[0] = std::log(0.3F / 0.7F);
  const std::array trees{DraftTreeLane{7, 3, 1, kChildren, flat}, DraftTreeLane{9, 3, 1, kChildren, flat}};
  run(backend, selector, f, trees);
  std::array<uint32_t, 2> chain{};
  std::array<uint32_t, 2> beside{};
  for (uint32_t lane = 0; lane < 2; ++lane) {
    const Lattice l = latticeOf(f, lane);
    const auto &ranked = l.ranked[0][0];
    if (lane == 0) {
      chain[lane] = l.ids[0][ranked[0]];
      beside[lane] = l.ids[0][ranked[1]];
    } else {
      while (std::ranges::find(l.ids[0], chain[lane]) != l.ids[0].end())
        ++chain[lane];
      beside[lane] = l.ids[0][ranked[0]];
    }
    *f.chainTokens(lane) = chain[lane];
    std::fill_n(f.chainRates(lane), kAgreements, kRate);
  }
  run(backend, selector, f, trees);
  const double left = 1.0 - double(kRate);
  for (uint32_t lane = 0; lane < 2; ++lane) {
    const std::string label = "chain beside drawn siblings, lane " + std::to_string(lane);
    const uint32_t first = lane * kBlock;
    const auto estimate = [&](uint32_t row) { return double(f.scores()[2 * (first + row) + 1]); };
    require(f.rowCount(lane) == 4, label + ": the tree holds " + std::to_string(f.rowCount(lane)) + " rows, not 4");
    require(f.tokens()[first + 1] == chain[lane] && std::fabs(estimate(1) - kRate) <= 1e-6 &&
                f.caps()[first + 1] == 1.0F,
            label + ": the chain's row is not row 1 at its rate, capped at 1");
    require(lane == 0 ? f.scores()[2 * (first + 1)] >= 0.0F : f.scores()[2 * (first + 1)] == -1.0F,
            label + ": the chain's row is joined by the draft's likeliest step only where it is that token");
    require(f.tokens()[first + 2] == beside[lane] && f.tree()[first + 2].parent == 0 &&
                std::fabs(estimate(2) - left) <= 1e-6 && std::fabs(f.caps()[first + 2] - left) <= 1e-6,
            label + ": the draft row beside the chain is estimated " + std::to_string(estimate(2)) + " of cap " +
                std::to_string(f.caps()[first + 2]) + ", not what the chain left, " + std::to_string(left));
    require(f.tree()[first + 3].parent != 0, label + ": a third child of the anchor outranked the draft's next step");
    requireChildrenWithinParents(f, lane, label);
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

// The best assignment of rows to lanes by brute force: each lane a prefix of
// its tree's rows, its anchor to at most `cap` of them, none past its anchor
// estimated below `floor`, the lanes' together at most `rows`, of the most
// expected tokens. Expected tokens count a lane's anchor 1 and each other row
// its estimate, in double.
double bestAssignment(const std::vector<DrawnTree> &trees, uint32_t rows, uint32_t cap, float floor) {
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
      best = std::max(best, expected);
    more = false;
    for (uint32_t lane = 0; lane < lanes && !more; ++lane) {
      if (kept[lane] < std::min<uint32_t>(cap, static_cast<uint32_t>(trees[lane].tokens.size())) &&
          trees[lane].estimates[kept[lane]] >= floor) {
        ++kept[lane];
        more = true;
      } else {
        kept[lane] = 1;
      }
    }
  }
  return best;
}

// The rows draft_verify_rows takes of each lane: every lane's rows past its
// anchor by estimate (the earlier lane's, then its earlier row, first among
// equal ones), each the step's rows and the lane's cap leave room for, to the
// first estimated below `floor`.
std::vector<uint32_t> ruleAssignment(const std::vector<DrawnTree> &trees, uint32_t rows, uint32_t cap, float floor) {
  const uint32_t lanes = static_cast<uint32_t>(trees.size());
  std::vector<std::pair<float, uint32_t>> order;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 1; row < trees[lane].estimates.size(); ++row)
      order.emplace_back(trees[lane].estimates[row], lane);
  std::stable_sort(order.begin(), order.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
  std::vector<uint32_t> taken(lanes, 1U);
  uint32_t total = lanes;
  for (const auto &entry : order) {
    if (total == rows || entry.first < floor)
      break;
    if (taken[entry.second] == cap)
      continue;
    ++taken[entry.second];
    ++total;
  }
  return taken;
}

// draft_verify_rows against its rule and every assignment of rows: for
// random trees (estimates with exact 0 and repeats, of 1 to kBlock rows),
// steps of 1 to 4 lanes, every count of rows and cap they allow and floors
// of none, at a row's estimate, just over one, anywhere and over them all,
// each lane holds exactly the rows the rule takes, the first rows its tree
// took, no lane more than the cap and the lanes no more than the step's
// rows, and their expected tokens are the best of every assignment that
// takes no row below the floor. An anchor's estimate, 0 here, is not read.
// The lanes' rows are the step's first, in lane order, and its rows past
// them, where the trees hold fewer or the floor left rows out, are of no
// lane: each holds lane 0's anchor and no source. A lane's rows are numbered
// in preorder, the earlier-taken child first, with their tokens, depths,
// ancestors and sources.
void verifyRowsLayout(MetalBackend &backend) {
  Random random(0xb0d9e7);
  uint32_t short_ = 0, floored = 0;
  for (uint32_t trial = 0; trial < 600; ++trial) {
    const uint32_t lanes = 1 + trial % kLanes;
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
    b.verifySources = allocate(backend, uint64_t{rows} * 4);
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
        static_cast<float *>(b.treeScores.contents())[2 * at + 1] = row ? trees[lane].estimates[row] : 0.0F;
      }
    }
    // The floor: none, a row's estimate, the next float over one (over the
    // least normal float where the estimate is 0: a GPU flushes a smaller
    // floor to none, and a host's floor, its line's slope times a rate, is no
    // such float), any, or one over every estimate.
    const DrawnTree &some = trees[random.next() % lanes];
    const float estimate = some.estimates[random.next() % some.estimates.size()];
    const uint32_t kind = random.next() % 6;
    const float floor = kind < 2    ? 0.0F
                        : kind == 2 ? estimate
                        : kind == 3 ? std::nextafter(std::max(estimate, std::numeric_limits<float>::min()), 2.0F)
                        : kind == 4 ? 0.5F * (random.unit() + 1.0F)
                                    : 1.5F;
    // A row holds 0xBAD, no tree's token nor source, until the layout writes
    // it.
    std::fill_n(static_cast<uint32_t *>(b.inputTokens.contents()), rows, 0xBADu);
    std::fill_n(static_cast<uint32_t *>(b.verifySources.contents()), rows, 0xBADu);
    CommandGraph graph;
    DraftSelector::addVerifyRows(graph, b, {lanes, rows, cap}, floor);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *laneRows = static_cast<const LaneRows *>(b.laneRows.contents());
    const std::string label = "verify rows trial " + std::to_string(trial) + ", floor " + std::to_string(floor);

    // The choice: each lane exactly the rule's rows, the best assignment.
    const std::vector<uint32_t> rule = ruleAssignment(trees, rows, cap, floor);
    floored += rule != ruleAssignment(trees, rows, cap, 0.0F);
    double expected = 0.0;
    uint32_t first = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      require(laneRows[lane].first == first && laneRows[lane].rows == rule[lane],
              label + ": lane " + std::to_string(lane) + " holds rows " + std::to_string(laneRows[lane].first) +
                  " to " + std::to_string(laneRows[lane].first + laneRows[lane].rows) + ", not the " +
                  std::to_string(rule[lane]) + " it took from row " + std::to_string(first));
      require(rule[lane] <= std::min<uint32_t>(cap, trees[lane].tokens.size()),
              label + ": a lane's rows left its tree or its cap");
      for (uint32_t row = 0; row < rule[lane]; ++row)
        expected += row ? trees[lane].estimates[row] : 1.0;
      first += laneRows[lane].rows;
    }
    require(first <= rows, label + ": the lanes hold more rows than the step");
    const double best = bestAssignment(trees, rows, cap, floor);
    require(expected >= best * (1.0 - 1e-6), label + ": draft_verify_rows chose " + std::to_string(expected) +
                                                 " expected tokens where " + std::to_string(best) + " were possible");
    // The step's rows past the lanes' are of no lane.
    short_ += first < rows;
    for (uint32_t slot = first; slot < rows; ++slot)
      require(static_cast<const uint32_t *>(b.inputTokens.contents())[slot] == trees[0].tokens[0] &&
                  static_cast<const uint32_t *>(b.verifySources.contents())[slot] == SPLASH_NO_TOKEN,
              label + ": row " + std::to_string(slot) + " of no lane is not lane 0's anchor without a source");

    // Each lane's rows in preorder.
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const DrawnTree &t = trees[lane];
      const uint32_t kept = rule[lane];
      std::vector<uint32_t> order;
      std::function<void(uint32_t)> visit = [&](uint32_t row) {
        order.push_back(row);
        for (uint32_t child = row + 1; child < kept; ++child)
          if (t.parents[child] == row)
            visit(child);
      };
      visit(0);
      std::vector<uint32_t> placed(kept), depth(kept), ancestors(kept);
      for (uint32_t at = 0; at < kept; ++at)
        placed[order[at]] = at;
      for (uint32_t at = 0; at < kept; ++at) {
        const uint32_t row = order[at];
        const uint32_t parent = at ? placed[t.parents[row]] : SPLASH_TREE_ROOT;
        depth[at] = at ? depth[parent] + 1 : 0;
        ancestors[at] = (at ? ancestors[parent] : 0U) | (1U << at);
        const uint32_t slot = laneRows[lane].first + at;
        const TreeRow &got = static_cast<const TreeRow *>(b.treeRows.contents())[slot];
        require(static_cast<const uint32_t *>(b.inputTokens.contents())[slot] == t.tokens[row] &&
                    got.parent == parent && got.depth == depth[at] && got.ancestors == ancestors[at] &&
                    static_cast<const uint32_t *>(b.verifySources.contents())[slot] == row,
                label + ": lane " + std::to_string(lane) + "'s row " + std::to_string(at) +
                    " is not its tree's in preorder");
      }
    }
  }
  // A quarter of the trials' trees hold fewer rows than their steps, and the
  // floor left rows out of a sixth of them.
  require(short_ >= 100, "only " + std::to_string(short_) + " of 600 trials left rows of no lane");
  require(floored >= 100, "the floor left rows out of only " + std::to_string(floored) + " of 600 trials");
}

// The floor on a row's own estimate: of two lanes' chains estimated 0.9, 0.5,
// 0.5, 0.25 and 0.1, and 0.5, 0.3 and 0, past their anchors, in a step of a
// draft's block a lane, a floor of none takes every row; one of 0.5 the rows
// at it and above, 4 and 2 with the anchors; the next float over 0.5 those
// above it, 2 and 1; and one over every estimate the anchors alone, whose
// own estimates, 0 here, the layout does not read. The lanes' rows are
// packed in lane order, and the step's other rows are of no lane, on no path
// and of no parent whatever their tree rows held before.
void verifyRowsFloor(MetalBackend &backend) {
  constexpr uint32_t lanes = 2, cap = 8, rows = lanes * cap;
  const std::array<std::vector<float>, lanes> estimates{
      {{0.0F, 0.9F, 0.5F, 0.5F, 0.25F, 0.1F}, {0.0F, 0.5F, 0.3F, 0.0F}}};
  DraftSelectorBuffers b{};
  b.treeTokens = allocate(backend, uint64_t{lanes} * kBlock * 4);
  b.trees = allocate(backend, uint64_t{lanes} * kBlock * sizeof(TreeRow));
  b.treeScores = allocate(backend, uint64_t{lanes} * kBlock * 8);
  b.treeSizes = allocate(backend, uint64_t{lanes} * 4);
  b.inputTokens = allocate(backend, uint64_t{rows} * 4);
  b.treeRows = allocate(backend, uint64_t{rows} * sizeof(TreeRow));
  b.laneRows = allocate(backend, uint64_t{lanes} * sizeof(LaneRows));
  b.verifySources = allocate(backend, uint64_t{rows} * 4);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    static_cast<uint32_t *>(b.treeSizes.contents())[lane] = static_cast<uint32_t>(estimates[lane].size());
    for (uint32_t row = 0; row < estimates[lane].size(); ++row) {
      const uint32_t at = lane * kBlock + row;
      static_cast<uint32_t *>(b.treeTokens.contents())[at] = 100 * (lane + 1) + row;
      static_cast<TreeRow *>(b.trees.contents())[at] = chainRow(row);
      static_cast<float *>(b.treeScores.contents())[2 * at + 1] = estimates[lane][row];
    }
  }
  struct Case final {
    float floor;
    std::array<uint32_t, lanes> held;
  };
  for (const Case &c : {Case{0.0F, {6, 4}}, Case{0.5F, {4, 2}}, Case{std::nextafter(0.5F, 1.0F), {2, 1}},
                        Case{1.5F, {1, 1}}}) {
    std::fill_n(static_cast<uint32_t *>(b.inputTokens.contents()), rows, 0xBADu);
    std::fill_n(static_cast<uint32_t *>(b.verifySources.contents()), rows, 0xBADu);
    std::fill_n(static_cast<TreeRow *>(b.treeRows.contents()), rows, chainRow(5));
    CommandGraph graph;
    DraftSelector::addVerifyRows(graph, b, {lanes, rows, cap}, c.floor);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *laneRows = static_cast<const LaneRows *>(b.laneRows.contents());
    const auto *tree = static_cast<const TreeRow *>(b.treeRows.contents());
    const auto *tokens = static_cast<const uint32_t *>(b.inputTokens.contents());
    const auto *sources = static_cast<const uint32_t *>(b.verifySources.contents());
    const std::string label = "verify rows at a floor of " + std::to_string(c.floor) + ": ";
    uint32_t first = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t held = laneRows[lane].rows;
      require(held >= 1 && laneRows[lane].first == first && tokens[first] == 100 * (lane + 1) && sources[first] == 0,
              label + "lane " + std::to_string(lane) + " does not hold its anchor after the lanes before it");
      require(held == c.held[lane],
              label + "lane " + std::to_string(lane) + " took " + std::to_string(held) + " rows, not " +
                  std::to_string(c.held[lane]) +
                  (held > c.held[lane] ? ": rows under the floor" : ": it left out rows at the floor or above it"));
      for (uint32_t row = 0; row < held; ++row)
        require(tokens[first + row] == 100 * (lane + 1) + row && sources[first + row] == row,
                label + "a lane's rows are not the first of its chain");
      first += held;
    }
    for (uint32_t slot = first; slot < rows; ++slot)
      require(tokens[slot] == 100 && sources[slot] == SPLASH_NO_TOKEN && !tree[slot].ancestors &&
                  tree[slot].parent == SPLASH_TREE_ROOT && !tree[slot].depth,
              label + "row " + std::to_string(slot) +
                  " of no lane is not lane 0's anchor without a source, a path or a parent");
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
  b.verifySources = allocate(backend, uint64_t{rows} * 4);
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
  DraftSelector::addVerifyRows(graph, b, {lanes, rows, cap});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *laneRows = static_cast<const LaneRows *>(b.laneRows.contents());
  // Lane 0's 8 rows (to the cap), lane 1's anchor and its 3 likeliest,
  // packed in lane order.
  require(laneRows[0].first == 0 && laneRows[0].rows == cap && laneRows[1].first == cap &&
              laneRows[1].rows == rows - cap,
          "the step's rows did not go to the likeliest rows across lanes, packed in lane order");
}

// Each buffer the selector and the verify rows reach, at its extent and one
// element short, for three lanes: every lane's eight query rows of logits and
// selector hidden rows, the workspaces of its seven positions, each
// codebook's row of every token, every lane's host chain and tree; and the
// step's rows and lane rows.
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
           {&DraftSelectorBuffers::treeSizes, uint64_t{lanes} * 4, 4, "draft tree size"},
           {&DraftSelectorBuffers::treeDraws, rows * kChildren, 1, "draft tree draw"},
           {&DraftSelectorBuffers::treeNext, rows * kCandidates * 4, 4, "draft tree next"},
           {&DraftSelectorBuffers::treeCaps, rows * 4, 4, "draft tree cap"}})
    requireExtent(backend, f.buffers.*member, bytes, element, what, [&](CommandGraph &graph, const MetalBuffer &buffer) {
      DraftSelectorBuffers changed = f.buffers;
      changed.*member = buffer;
      selector.add(graph, changed, f.codebooks, {}, trees);
    });
  // Over a subset, a logits row holds its columns.
  constexpr DraftVocabulary subset{512, 960};
  requireExtent(backend, f.buffers.logits, uint64_t{lanes} * kRows * subset.columns(vocabulary) * 4, 4, "draft logits",
                [&](CommandGraph &graph, const MetalBuffer &buffer) {
                  DraftSelectorBuffers changed = f.buffers;
                  changed.logits = buffer;
                  selector.add(graph, changed, f.codebooks, subset, trees);
                });
  for (const auto &[member, what] :
       {std::pair{&DraftCodebooks::predecessor, "draft predecessor codebook"},
        std::pair{&DraftCodebooks::successor, "draft successor codebook"}})
    requireExtent(backend, f.codebooks.*member, uint64_t{vocabulary} * kRank * 2, 2, what,
                  [&](CommandGraph &graph, const MetalBuffer &buffer) {
                    DraftCodebooks changed = f.codebooks;
                    changed.*member = buffer;
                    selector.add(graph, f.buffers, changed, {}, trees);
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
           {&DraftSelectorBuffers::verifySources, uint64_t{step.rows} * 4, 4, "verify source"}})
    requireExtent(backend, f.buffers.*member, bytes, element, what, [&](CommandGraph &graph, const MetalBuffer &buffer) {
      DraftSelectorBuffers changed = f.buffers;
      changed.*member = buffer;
      DraftSelector::addVerifyRows(graph, changed, step);
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
  const auto add = [&](std::span<const DraftTreeLane> lanes) {
    selector.add(graph, f.buffers, f.codebooks, {}, lanes);
  };
  rejects([&] { add({}); }, "invalid draft selector batch", "a selection of no lane was accepted");
  const std::array<DraftTreeLane, 5> five{};
  rejects([&] { add(five); }, "invalid draft selector batch", "a selection of five lanes was accepted");
  // Tokens past the vocabulary, a trailing part before the leading one's end, and no tokens.
  for (const DraftVocabulary tokens : {DraftVocabulary{0, 1025}, DraftVocabulary{600, 500}, DraftVocabulary{0, 1024}})
    rejects([&] { selector.add(graph, f.buffers, f.codebooks, tokens, std::array{DraftTreeLane{1, 3, 0, 1}}); },
            "a draft's tokens lie in its vocabulary", "a draft vocabulary outside the selector's was accepted");
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
    rejects([&] { DraftSelector::addVerifyRows(graph, f.buffers, bad); }, "invalid decode step rows",
            "impossible step rows were encoded");
  require(graph.empty(), "an invalid draft selector request encoded a graph");
  // A floor that is not a number, below none, or without end.
  for (const float bad : {NAN, -0.25F, INFINITY})
    rejects([&] { DraftSelector::addVerifyRows(graph, f.buffers, StepRows{2, 2 * kBlock, kBlock}, bad); },
            "invalid verify rows floor", "an impossible floor was encoded");
  require(graph.empty(), "an invalid floor encoded a graph");
  CommandGraph encoded;
  DraftSelector::addVerifyRows(encoded, f.buffers, StepRows{2, 2 * kBlock, kBlock}, 0.25F);
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
    // Subsets: the Qwen families', one of 555 columns that misalign the
    // vectors, a leading part alone and a trailing part alone.
    for (const auto &[vocabulary, tokens] :
         {std::pair{248320U, DraftVocabulary{148288, 248000}}, std::pair{1003U, DraftVocabulary{512, 960}},
          std::pair{1003U, DraftVocabulary{517, 1003}}, std::pair{1003U, DraftVocabulary{0, 389}}})
      for (const uint32_t lanes : {1U, kLanes})
        chainCase(backend, vocabulary, lanes, tokens);
    treesMatchTheReference(backend);
    sampledChildrenDistribution(backend);
    nonFiniteRowDrawsNothing(backend);
    lowTemperatureDraws(backend);
    forcedChain(backend);
    chainRowsTheDraftJoins(backend);
    siblingsShareTheirParent(backend);
    chainCountsAgainstItsSiblings(backend);
    verifyRowsLayout(backend);
    verifyRowsFloor(backend);
    verifyRowsAcrossLanes(backend);
    std::cout << "draft_selector_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_selector_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
