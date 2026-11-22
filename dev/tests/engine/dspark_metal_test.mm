// A DSpark draft's kernels (ops::DSpark, kernels/decode/dspark.metal,
// kernels/shared/dspark_mix.metal) and its shape of the draft attention and
// selector against host references on random data:
//  - the context mix: softmax-weighted taps, RMS norm, F32 gamma;
//  - the q/k/v prepare of 24 query heads into the core's grouped rows, the
//    padding heads zero, and the core's row-uniform windows over the ring
//    (a windowed layer's and the global layer's) with its rows of 24 heads;
//  - the Markov chain of one to four lanes over a small vocabulary, greedy
//    (the most likely token a step, lowest on ties, its candidates the top
//    16) and sampled (a draw of the host's Gumbel keys, first);
//  - the f_rnn commit of each lane's last retained row;
//  - the selector's tree of a recurrent chain (DraftSelector::addChain):
//    a greedy lane's chain of candidate 0 with leaves beside it, a sampled
//    lane's first child its chain's token.
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/DSpark.h"
#include "metal/abi/DraftAttention.h"
#include "ops/DSpark.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/LaneRows.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace splash;
using metal::CommandGraph;
using metal::MetalBackend;
using metal::MetalBuffer;
using test::bf16ToFloat;
using test::floatToBf16;
using test::require;

constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kSteps = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kRank = SPLASH_DSPARK_RANK;
constexpr uint32_t kJoint = SPLASH_DSPARK_JOINT;
constexpr uint32_t kTaps = SPLASH_DSPARK_TAPS;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kKvHeads = SPLASH_DRAFT_KV_HEADS;
constexpr uint32_t kHeadDim = SPLASH_DRAFT_HEAD_DIMENSION;
constexpr uint32_t kHeads = 24;
constexpr uint32_t kOwn = kHeads / kKvHeads;
constexpr uint32_t kWindow = SPLASH_DRAFT_SLIDING_WINDOW;
// A DSpark draft's attention: 24 query heads, its q/k/v rows of its own.
constexpr ops::DraftAttentionShape kShape{2304, 0, kHeads * kHeadDim + 2 * kKvHeads * kHeadDim, kHeads * kHeadDim,
                                          kHeads, kKvHeads, kHeadDim};

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }

private:
  uint64_t state_;
};

MetalBuffer shared(MetalBackend &backend, uint64_t bytes) { return test::sharedBuffer(backend, std::max<uint64_t>(bytes, 16)); }

template <class T> T *at(const MetalBuffer &buffer) { return static_cast<T *>(buffer.contents()); }

MetalBuffer randomBf16(MetalBackend &backend, uint64_t count, Random &random, float scale = 1.0F) {
  MetalBuffer buffer = shared(backend, count * 2);
  for (uint64_t i = 0; i < count; ++i) at<uint16_t>(buffer)[i] = floatToBf16(scale * random.unit());
  return buffer;
}

void run(MetalBackend &backend, const CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

void mix(MetalBackend &backend) {
  Random random(11);
  constexpr uint32_t rows = 5, width = 2304;
  const MetalBuffer features = randomBf16(backend, uint64_t{rows} * kTaps * width, random, 40.0F);
  ops::DSparkMix weights{{0.1F, 0.2F, 0.6F, 0.1F}, shared(backend, width * 4)};
  for (uint32_t i = 0; i < width; ++i) at<float>(weights.gamma)[i] = 0.5F + random.unit();
  const MetalBuffer output = shared(backend, uint64_t{rows} * width * 2);
  CommandGraph graph;
  ops::DSpark::addMix(graph, features, weights, output, rows, width);
  run(backend, graph);
  for (uint32_t row = 0; row < rows; ++row) {
    std::vector<double> mixed(width);
    double squares = 0.0;
    for (uint32_t c = 0; c < width; ++c) {
      for (uint32_t tap = 0; tap < kTaps; ++tap)
        mixed[c] += weights.alpha[tap] * bf16ToFloat(at<uint16_t>(features)[(uint64_t{row} * kTaps + tap) * width + c]);
      squares += mixed[c] * mixed[c];
    }
    const double inverse = 1.0 / std::sqrt(squares / width + 1e-6);
    for (uint32_t c = 0; c < width; ++c) {
      const double want = mixed[c] * inverse * at<float>(weights.gamma)[c];
      const float got = bf16ToFloat(at<uint16_t>(output)[uint64_t{row} * width + c]);
      require(std::fabs(got - want) <= 2 * test::ulpBf16(static_cast<float>(want)) + 1e-6,
              "the DSpark mix differs from the host's");
    }
  }
  std::cout << "  mix: ok\n";
}

// The prepare of 24 query heads and the core over the rings in row-uniform
// windows, against the host.
void attention(MetalBackend &backend, uint32_t lanes, uint32_t windowKeys, std::array<uint32_t, kLanes> lengths) {
  Random random(0xa77e + lanes * 131 + windowKeys + lengths[0]);
  const auto plan = ops::DraftAttention::plan(kShape, lanes);
  const ops::DraftAttentionWorkspace workspace = plan.workspace();
  const uint64_t rows = uint64_t{lanes} * kRows;
  const MetalBuffer qkv = randomBf16(backend, rows * kShape.qkvSize, random, 3.0F);
  const MetalBuffer queries = shared(backend, workspace.groupedQueriesBytes);
  std::memset(queries.contents(), 0x7f, queries.sizeBytes());
  const MetalBuffer queryKeys = shared(backend, workspace.queryKeysBytes);
  const MetalBuffer queryValues = shared(backend, workspace.queryValuesBytes);
  const MetalBuffer queryNorm = randomBf16(backend, kHeadDim, random), keyNorm = randomBf16(backend, kHeadDim, random);
  const uint64_t ropeRows = uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS;
  const MetalBuffer ropeCos = shared(backend, ropeRows * kHeadDim / 2 * 4), ropeSin = shared(backend, ropeRows * kHeadDim / 2 * 4);
  for (uint64_t i = 0; i < ropeRows * kHeadDim / 2; ++i) {
    at<float>(ropeCos)[i] = std::cos(float(i) * 0.013F);
    at<float>(ropeSin)[i] = std::sin(float(i) * 0.013F);
  }
  std::vector<MetalBuffer> keys, values;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    keys.push_back(randomBf16(backend, uint64_t{kKvHeads} * kWindow * kHeadDim, random));
    values.push_back(randomBf16(backend, uint64_t{kKvHeads} * kWindow * kHeadDim, random));
  }
  CommandGraph prepare;
  ops::DraftAttention::addPrepare(prepare, {qkv, queries, queryNorm, keyNorm, ropeCos, ropeSin, queryKeys, queryValues},
                                  plan);
  run(backend, prepare);
  // The grouped queries: KV head g's four heads, its three own normed and
  // rotated, the fourth zero.
  const auto *packed = at<uint16_t>(qkv);
  const auto *grouped = at<uint16_t>(queries);
  // A row's normed and rotated head, and each value's bound: the rounding
  // of the normed pair it rotates, which the kernel's own inverse may move
  // by an ulp each, through the rotation, and the output's rounding.
  const auto normed = [&](const uint16_t *source, const uint16_t *weight, const float *cosines, const float *sines,
                          std::array<float, kHeadDim> &out, std::array<float, kHeadDim> &bound) {
    double squares = 0.0;
    for (uint32_t d = 0; d < kHeadDim; ++d) squares += double(bf16ToFloat(source[d])) * bf16ToFloat(source[d]);
    const double inverse = 1.0 / std::sqrt(squares / kHeadDim + 1e-6);
    std::array<float, kHeadDim> rounded;
    for (uint32_t d = 0; d < kHeadDim; ++d)
      rounded[d] = bf16ToFloat(floatToBf16(float(bf16ToFloat(source[d]) * inverse * bf16ToFloat(weight[d]))));
    for (uint32_t d = 0; d < kHeadDim / 2; ++d) {
      const float first = rounded[d], second = rounded[d + kHeadDim / 2];
      out[d] = first * cosines[d] - second * sines[d];
      out[d + kHeadDim / 2] = second * cosines[d] + first * sines[d];
      const float pair = test::ulpBf16(first) * std::fabs(cosines[d]) + test::ulpBf16(second) * std::fabs(sines[d]);
      const float swapped =
          test::ulpBf16(second) * std::fabs(cosines[d]) + test::ulpBf16(first) * std::fabs(sines[d]);
      bound[d] = pair + 2 * test::ulpBf16(out[d]);
      bound[d + kHeadDim / 2] = swapped + 2 * test::ulpBf16(out[d + kHeadDim / 2]);
    }
  };
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < kRows; ++row) {
      const uint64_t laneRow = uint64_t{lane} * kRows + row;
      const float *cosines = at<float>(ropeCos) + (uint64_t{lane} * SPLASH_TARGET_VERIFY_ROWS + row) * kHeadDim / 2;
      const float *sines = at<float>(ropeSin) + (uint64_t{lane} * SPLASH_TARGET_VERIFY_ROWS + row) * kHeadDim / 2;
      for (uint32_t padded = 0; padded < SPLASH_DRAFT_QUERY_HEADS; ++padded) {
        const uint32_t group = padded / 4, own = padded % 4;
        const uint16_t *got = grouped + uint64_t{lane} * kRows * SPLASH_DRAFT_ATTENTION_WIDTH +
                              (uint64_t{padded} * kRows + row) * kHeadDim;
        std::array<float, kHeadDim> want{}, bound{};
        if (own < kOwn)
          normed(packed + laneRow * kShape.qkvSize + (group * kOwn + own) * kHeadDim, at<uint16_t>(queryNorm), cosines,
                 sines, want, bound);
        for (uint32_t d = 0; d < kHeadDim; ++d)
          require(std::fabs(bf16ToFloat(got[d]) - want[d]) <= bound[d] + 1e-3F,
                  "the DSpark prepare's grouped queries differ from the host's");
      }
    }
  CommandGraph core;
  ops::DraftAttention::addDecode(core, {queries, keys, values, queryKeys, queryValues}, std::span(lengths).first(lanes),
                                 plan, {windowKeys, true});
  // The core's input, kept for the host: its rows are overwritten.
  std::vector<uint16_t> input(grouped, grouped + rows * SPLASH_DRAFT_ATTENTION_WIDTH);
  run(backend, core);
  const auto *output = at<uint16_t>(queries);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t length = lengths[lane];
    const uint32_t first = length >= windowKeys ? length - windowKeys : 0;
    const auto *ringKeys = at<uint16_t>(keys[lane]), *ringValues = at<uint16_t>(values[lane]);
    const auto *blockKeys = at<uint16_t>(queryKeys) + uint64_t{lane} * kKvHeads * kRows * kHeadDim;
    const auto *blockValues = at<uint16_t>(queryValues) + uint64_t{lane} * kKvHeads * kHeadDim * kRows;
    for (uint32_t head = 0; head < kHeads; ++head) {
      const uint32_t group = head / kOwn, padded = group * 4 + head % kOwn;
      for (uint32_t row = 0; row < kRows; ++row) {
        const uint16_t *query = input.data() + uint64_t{lane} * kRows * SPLASH_DRAFT_ATTENTION_WIDTH +
                                (uint64_t{padded} * kRows + row) * kHeadDim;
        std::vector<double> scores;
        std::vector<const uint16_t *> rowsOfKeys;
        const auto dot = [&](const uint16_t *key) {
          double sum = 0.0;
          for (uint32_t d = 0; d < kHeadDim; ++d) sum += double(bf16ToFloat(query[d])) * bf16ToFloat(key[d]);
          return sum / std::sqrt(double(kHeadDim));
        };
        for (uint32_t position = first; position < length; ++position)
          scores.push_back(dot(ringKeys + (uint64_t{group} * kWindow + position % kWindow) * kHeadDim));
        for (uint32_t key = 0; key < kRows; ++key) scores.push_back(dot(blockKeys + (uint64_t{group} * kRows + key) * kHeadDim));
        const double best = *std::ranges::max_element(scores);
        double sum = 0.0;
        for (double &score : scores) sum += score = std::exp(score - best);
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          double value = 0.0;
          uint32_t index = 0;
          for (uint32_t position = first; position < length; ++position, ++index)
            value += scores[index] * bf16ToFloat(ringValues[(uint64_t{group} * kHeadDim + d) * kWindow + position % kWindow]);
          for (uint32_t key = 0; key < kRows; ++key, ++index)
            value += scores[index] * bf16ToFloat(blockValues[(uint64_t{group} * kHeadDim + d) * kRows + key]);
          value /= sum;
          const float got = bf16ToFloat(output[(uint64_t{lane} * kRows + row) * kHeads * kHeadDim + head * kHeadDim + d]);
          require(std::fabs(got - value) <= 4e-3 + 2 * test::ulpBf16(float(value)),
                  "the DSpark attention differs from the host's at lane " + std::to_string(lane) + " length " +
                      std::to_string(length));
        }
      }
    }
  }
}

// The host's Gumbel of a draw (kernels/decode/dspark.metal dspark_gumbel).
uint64_t hash(uint64_t value) {
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
float gumbel(uint64_t seed, uint32_t step, uint32_t token) {
  const uint64_t value = hash(hash(seed + 0x9e3779b97f4a7c15ULL * (uint64_t{step} + 1)) ^
                              0xd1b54a32d192ed03ULL * (uint64_t{token} + 1));
  const float uniform = (float(value >> 41) + 0.5F) * 0x1p-23F;
  return -std::log(-std::log(uniform));
}

void markov(MetalBackend &backend, uint32_t lanes, bool sampled) {
  Random random(0x3a7c + lanes * 7 + sampled);
  constexpr uint32_t hidden = 256, vocabulary = 3008, proposed = 2999;
  constexpr float cap = 30.0F;
  const uint32_t width = 2 * kRank + 2 * hidden;
  ops::DSparkMarkovWeights weights{randomBf16(backend, uint64_t{kJoint} * width, random, 0.08F),
                                   shared(backend, kJoint * 4), randomBf16(backend, uint64_t{vocabulary} * kRank, random),
                                   randomBf16(backend, uint64_t{vocabulary} * kRank, random, 0.5F),
                                   shared(backend, vocabulary * 4)};
  for (uint32_t i = 0; i < kJoint; ++i) at<float>(weights.jointBias)[i] = 0.3F * random.unit();
  for (uint32_t i = 0; i < vocabulary; ++i) at<float>(weights.foldBias)[i] = random.unit();
  const ops::DSparkMarkovWorkspace workspace = ops::DSpark::markovWorkspace(lanes, vocabulary);
  const uint64_t lattice = uint64_t{lanes} * kSteps * kCandidates;
  ops::DSparkMarkovBuffers buffers{randomBf16(backend, uint64_t{lanes} * kRows * hidden, random),
                                   shared(backend, uint64_t{lanes} * kRows * vocabulary * 4),
                                   randomBf16(backend, uint64_t{lanes} * hidden, random),
                                   shared(backend, workspace.jointBytes),
                                   shared(backend, workspace.stateBytes),
                                   shared(backend, workspace.previousBytes),
                                   shared(backend, workspace.logitsBytes),
                                   shared(backend, workspace.partialValuesBytes),
                                   shared(backend, workspace.partialIdsBytes),
                                   shared(backend, lattice * 4),
                                   shared(backend, lattice * 4)};
  for (uint64_t i = 0; i < uint64_t{lanes} * kRows * vocabulary; ++i) at<float>(buffers.base)[i] = 4.0F * random.unit();
  std::array<ops::DraftTreeLane, kLanes> treeLanes{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    treeLanes[lane].anchor = 17 + 101 * lane;
    treeLanes[lane].temperature = sampled ? 0.7F + 0.2F * lane : 0.0F;
    treeLanes[lane].noise = 0x1234567890abcdefULL * (lane + 1);
  }
  CommandGraph graph;
  ops::DSpark::addMarkov(graph, weights, buffers, {hidden, vocabulary, proposed, cap},
                         std::span(treeLanes).first(lanes));
  run(backend, graph);
  // The host's chain, step by step from the device's own choices (a near
  // tie may choose another token than the host's exact logits would).
  const auto *candidates = at<uint32_t>(buffers.candidates);
  const auto *unary = at<float>(buffers.unary);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    std::vector<double> state(kRank, 0.0);
    uint32_t previous = treeLanes[lane].anchor;
    for (uint32_t step = 0; step < kSteps; ++step) {
      std::vector<double> input(width);
      for (uint32_t k = 0; k < kRank; ++k) {
        input[k] = step ? double(float(state[k])) : 0.0;
        input[kRank + k] = bf16ToFloat(at<uint16_t>(weights.embedding)[uint64_t{previous} * kRank + k]);
      }
      for (uint32_t k = 0; k < hidden; ++k) {
        input[2 * kRank + k] = bf16ToFloat(at<uint16_t>(buffers.hidden)[(uint64_t{lane} * kRows + step + 1) * hidden + k]);
        input[2 * kRank + hidden + k] = bf16ToFloat(at<uint16_t>(buffers.frnn)[uint64_t{lane} * hidden + k]);
      }
      std::vector<double> z(kJoint);
      for (uint32_t j = 0; j < kJoint; ++j) {
        double sum = at<float>(weights.jointBias)[j];
        for (uint32_t k = 0; k < width; ++k) sum += bf16ToFloat(at<uint16_t>(weights.joint)[uint64_t{j} * width + k]) * input[k];
        z[j] = sum;
      }
      std::vector<std::pair<double, uint32_t>> logits;
      for (uint32_t token = 0; token < proposed; ++token) {
        double correction = at<float>(weights.foldBias)[token];
        for (uint32_t k = 0; k < kRank; ++k)
          correction += bf16ToFloat(at<uint16_t>(weights.foldWeights)[uint64_t{token} * kRank + k]) * std::tanh(z[2 * kRank + k]);
        const double base = at<float>(buffers.base)[(uint64_t{lane} * kRows + step + 1) * vocabulary + token];
        logits.emplace_back(cap * std::tanh((base + correction) / cap), token);
      }
      std::ranges::sort(logits, [](const auto &a, const auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
      const uint32_t *got = candidates + (uint64_t{lane} * kSteps + step) * kCandidates;
      const float *gotLogits = unary + (uint64_t{lane} * kSteps + step) * kCandidates;
      // The candidates: the host's top 16, the chosen first; a greedy lane's
      // chosen the most likely.
      std::vector<uint32_t> gotSet(got, got + kCandidates), wantSet;
      for (uint32_t i = 0; i < kCandidates; ++i) wantSet.push_back(logits[i].second);
      std::ranges::sort(gotSet);
      std::ranges::sort(wantSet);
      uint32_t common = 0;
      for (uint32_t token : gotSet) common += std::ranges::binary_search(wantSet, token);
      require(common >= kCandidates - 1, "the Markov candidates are not the host's top 16");
      for (uint32_t i = 0; i < kCandidates; ++i) {
        const auto found = std::ranges::find_if(logits, [&](const auto &entry) { return entry.second == got[i]; });
        require(std::fabs(found->first - gotLogits[i]) < 2e-3, "a Markov candidate's logit differs from the host's");
      }
      for (uint32_t i = 2; i < kCandidates; ++i)
        require(gotLogits[i] <= gotLogits[i - 1] || (i == 1), "the Markov candidates after the chosen are not in order");
      if (!sampled) {
        require(std::fabs(gotLogits[0] - logits[0].first) < 2e-3, "a greedy Markov step chose another token");
      } else {
        // The host's draw over the device's candidates.
        std::vector<std::pair<float, uint32_t>> sorted;
        for (uint32_t i = 0; i < kCandidates; ++i) sorted.emplace_back(gotLogits[i], got[i]);
        std::ranges::sort(sorted, [](const auto &a, const auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        const uint64_t seed = treeLanes[lane].noise;
        float best = -INFINITY;
        uint32_t pick = 0;
        for (uint32_t i = 0; i < kCandidates; ++i) {
          const float key = (sorted[i].first - sorted[0].first) / treeLanes[lane].temperature + gumbel(seed, step, sorted[i].second);
          if (key > best) {
            best = key;
            pick = sorted[i].second;
          }
        }
        require(pick == got[0], "a sampled Markov step chose another draw than the host's");
      }
      // The next step follows the chosen token and the updated state.
      for (uint32_t k = 0; k < kRank; ++k) {
        const double gate = 1.0 / (1.0 + std::exp(-z[k])), candidate = std::tanh(z[kRank + k]);
        state[k] = gate * (step ? double(float(state[k])) : 0.0) + (1.0 - gate) * candidate;
      }
      previous = got[0];
    }
  }
  std::cout << "  markov " << lanes << (sampled ? " sampled" : " greedy") << ": ok\n";
}

// Each lane's last retained row into its f_rnn row, a lane in each binding.
void frnnCommit(MetalBackend &backend) {
  Random random(5);
  constexpr uint32_t width = 2304, lanes = kLanes;
  static_assert(lanes == 8, "a lane in each binding");
  const std::array<ops::LaneRows, lanes> layout{
      {{0, 4}, {4, 1}, {5, 6}, {11, 3}, {14, 2}, {16, 5}, {21, 1}, {22, 4}}};
  const std::array<uint32_t, lanes> retained{0b1011u, 0b1u, 0b110011u, 0b101u, 0b11u, 0b10001u, 0b1u, 0b1001u};
  const std::array<uint32_t, lanes> last{3, 0, 5, 2, 1, 4, 0, 3};
  const MetalBuffer rows = randomBf16(backend, uint64_t{26} * width, random);
  std::array<MetalBuffer, kLanes> frnn;
  for (MetalBuffer &buffer : frnn) buffer = shared(backend, width * 2);
  const MetalBuffer mask = shared(backend, lanes * 4), laneRows = shared(backend, lanes * sizeof(ops::LaneRows));
  std::ranges::copy(retained, at<uint32_t>(mask));
  std::ranges::copy(layout, at<ops::LaneRows>(laneRows));
  CommandGraph graph;
  ops::DSpark::addFrnnCommit(graph, rows, frnn, mask, laneRows, lanes, width);
  run(backend, graph);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    require(std::equal(at<uint16_t>(frnn[lane]), at<uint16_t>(frnn[lane]) + width,
                       at<uint16_t>(rows) + uint64_t{layout[lane].first + last[lane]} * width),
            "the f_rnn commit took another row than the lane's last retained");
  std::cout << "  f_rnn commit: ok\n";
}

void chainTree(MetalBackend &backend, bool sampled) {
  constexpr uint32_t lanes = 2;
  const uint64_t lattice = uint64_t{lanes} * kSteps * kCandidates, treeRows = uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS;
  ops::DraftSelectorBuffers buffers;
  buffers.candidates = shared(backend, lattice * 4);
  buffers.unary = shared(backend, lattice * 4);
  buffers.partialValues = shared(backend, 16);
  buffers.chainTokens = shared(backend, uint64_t{lanes} * SPLASH_TREE_CHAIN_TOKENS * 4);
  buffers.chainRates = shared(backend, uint64_t{lanes} * SPLASH_TREE_CHAIN_TOKENS * SPLASH_TREE_AGREEMENTS * 4);
  buffers.treeTokens = shared(backend, treeRows * 4);
  buffers.trees = shared(backend, treeRows * sizeof(ops::TreeRow));
  buffers.treeScores = shared(backend, treeRows * 8);
  buffers.treeSizes = shared(backend, lanes * 4);
  buffers.treeDraws = shared(backend, treeRows * SPLASH_TREE_CHILDREN);
  buffers.treeNext = shared(backend, treeRows * kCandidates * 4);
  buffers.treeCaps = shared(backend, treeRows * 4);
  // Each position's candidates, the chain's token first; logits decreasing
  // after it, the chain's sometimes lower (a sampled draw).
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t step = 0; step < kSteps; ++step)
      for (uint32_t i = 0; i < kCandidates; ++i) {
        const uint64_t index = (uint64_t{lane} * kSteps + step) * kCandidates + i;
        at<uint32_t>(buffers.candidates)[index] = 1000 * (lane + 1) + 100 * step + i;
        at<float>(buffers.unary)[index] = (sampled && i == 0 ? 2.0F : 5.0F) - 0.25F * i;
      }
  std::array<ops::DraftTreeLane, lanes> trees{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    trees[lane] = {.anchor = 7, .depth = kSteps, .chain = 0, .children = 3, .calibration = {1.0F, {3, 3, 3, 3, 3, 3, 3}}};
    if (sampled) {
      trees[lane].temperature = 1.0F;
      trees[lane].noise = 99 + lane;
    }
  }
  ops::DraftSelector selector(4096);
  CommandGraph graph;
  selector.addChain(graph, buffers, trees);
  run(backend, graph);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t size = at<uint32_t>(buffers.treeSizes)[lane];
    require(size >= 2 && size <= kSteps + 1, "a chain's tree holds its anchor and up to its depth's rows");
    const uint32_t *tokens = at<uint32_t>(buffers.treeTokens) + uint64_t{lane} * SPLASH_TARGET_VERIFY_ROWS;
    const ops::TreeRow *rows = at<ops::TreeRow>(buffers.trees) + uint64_t{lane} * SPLASH_TARGET_VERIFY_ROWS;
    for (uint32_t row = 1; row < size; ++row) {
      const uint32_t depth = rows[row].depth, parent = rows[row].parent;
      const uint32_t token = tokens[row];
      require(token / 1000 == lane + 1 && token % 1000 / 100 == depth - 1, "a tree row's token is not its depth's");
      // A row deeper than one hangs from the chain: its parent is the
      // chain's token of its depth less one.
      if (depth > 1)
        require(tokens[parent] % 100 == 0, "a row's parent is not its chain's token");
    }
    // The chain's first row is its first token.
    require(tokens[1] == 1000 * (lane + 1), "the tree's first row is not its chain's token");
  }
  std::cout << "  chain tree" << (sampled ? " sampled" : " greedy") << ": ok\n";
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 2) throw std::invalid_argument("usage: dspark METALLIB");
      MetalBackend backend(argv[1]);
      mix(backend);
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        attention(backend, lanes, 1024, {0, 1030, 2047, 5000});
        attention(backend, lanes, kWindow - 1, {9, 2048, 2047, 4100});
      }
      std::cout << "  attention: ok\n";
      for (uint32_t lanes : {1u, 4u})
        for (const bool sampled : {false, true}) markov(backend, lanes, sampled);
      frnnCommit(backend);
      chainTree(backend, false);
      chainTree(backend, true);
      std::cout << "dspark_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "dspark_metal_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
