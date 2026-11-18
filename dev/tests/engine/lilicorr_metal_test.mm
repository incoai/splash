// LiLiCorr selector (ops::LiLiCorr) against a double-precision evaluation of
// the same network: synthetic bf16 weights at the compiled widths, a draft of
// 512 hidden values, and each value the kernels store as bf16 (every
// projection's input and output, the features, the factor input) rounded to
// bf16 where they round it. Each position's candidates must be the exact top
// sixteen and its nodes their first eight; the edge tables the walk reads in
// the partial-values scratch must hold the reference's scaled cosines, up to
// the noise of fp32 accumulation under those roundings, and zero elsewhere;
// and each greedy lane must take, at every position, its table row's best
// candidate, which is the reference's wherever the reference's margin is
// above that noise. Requests the selector cannot serve, and scratch one
// element short, are refused before anything is encoded.
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/LiLiCorr.h"
#include "metal/abi/QuantFormat.h"
#include "metal/abi/Sampling.h"
#include "metal/abi/VerifyTree.h"
#include "ops/LiLiCorr.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using splash::test::rejects;
using splash::test::require;
using splash::test::sharedBuffer;

constexpr uint32_t kWidth = SPLASH_LILICORR_HIDDEN;
constexpr uint32_t kNodes = SPLASH_LILICORR_NODES;
constexpr uint32_t kRanks = SPLASH_LILICORR_CANDIDATES;
constexpr uint32_t kHeads = SPLASH_LILICORR_HEADS;
constexpr uint32_t kHeadDimension = SPLASH_LILICORR_HEAD_DIMENSION;
constexpr uint32_t kMlp = SPLASH_LILICORR_MLP;
constexpr uint32_t kFeatures = SPLASH_LILICORR_FEATURES;
constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kShards = SPLASH_DRAFT_SAMPLING_SHARDS;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kBlock = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kHidden = 512;
constexpr uint32_t kVocabulary = 1003;
constexpr double kRmsEpsilon = 1e-6;
// The largest difference of a table entry (8 * cos) from the reference: a
// few bf16 rounding steps that fp32 accumulation takes the other way, each
// moving one value by 2^-8 of itself, over 1024-value cosines.
constexpr double kTableTolerance = 0.02;

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() { return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F; }
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }

private:
  uint64_t state_;
};

double roundBf16(double value) { return bf16ToFloat(floatToBf16(static_cast<float>(value))); }
double silu(double value) { return value / (1.0 + std::exp(-value)); }
const uint16_t *bf16(const MetalBuffer &buffer) { return static_cast<const uint16_t *>(buffer.contents()); }
double at(const MetalBuffer &buffer, uint64_t index) { return bf16ToFloat(bf16(buffer)[index]); }

MetalBuffer zeroed(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer = sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// Synthetic weights: tables of offset + scale * U(-1, 1), and projections
// whose outputs keep their inputs' scale.
class Weights final {
public:
  Weights(MetalBackend &backend, uint64_t seed) : backend_(backend), random_(seed) {}

  MetalBuffer table(uint64_t count, float scale, float offset = 0.0F) {
    MetalBuffer buffer = sharedBuffer(backend_, count * sizeof(uint16_t));
    auto *values = static_cast<uint16_t *>(buffer.contents());
    for (uint64_t i = 0; i < count; ++i) values[i] = floatToBf16(offset + scale * random_.unit());
    return buffer;
  }
  NormWeights norm(uint32_t width) { return {table(width, 0.2F, 1.0F), false}; }
  Projection projection(uint32_t n, uint32_t k) {
    BlockWeights blocks;
    blocks.segments.push_back(
        QuantizedSegment::rows(GGUF_FMT_BF16, n, k, table(uint64_t{n} * k, std::sqrt(3.0F / float(k)))));
    return Projection(n, k, std::move(blocks));
  }

  LiLiCorrWeights lilicorr() {
    LiLiCorrWeights w;
    w.tokenProjection = projection(kWidth, kHidden);
    w.tokenBias = table(kWidth, 0.1F);
    w.passProjection = projection(kWidth, kHidden);
    w.passBias = table(kWidth, 0.1F);
    w.contextProjection = projection(kWidth, kHidden);
    w.contextBias = table(kWidth, 0.1F);
    w.featureNormWeight = table(kFeatures, 0.2F, 1.0F);
    w.featureNormBias = table(kFeatures, 0.1F);
    w.featureUp = table(uint64_t{kWidth} * kFeatures, 0.8F);
    w.featureUpBias = table(kWidth, 0.1F);
    w.featureDown = projection(kWidth, kWidth);
    w.featureDownBias = table(kWidth, 0.1F);
    w.slotEmbedding = table(uint64_t{kPositions} * kWidth, 0.3F);
    w.rankEmbedding = table(uint64_t{kRanks} * kWidth, 0.3F);
    w.attentionBias = table(uint64_t{kHeads} * kNodes * kNodes, 0.5F);
    for (uint32_t layer = 0; layer < SPLASH_LILICORR_LAYERS; ++layer) {
      LiLiCorrLayerWeights l;
      l.attentionNorm = norm(kWidth);
      l.inProjection = projection(3 * kWidth, kWidth);
      l.inBias = table(3 * kWidth, 0.1F);
      l.outProjection = projection(kWidth, kWidth);
      l.outBias = table(kWidth, 0.1F);
      l.mlpNorm = norm(kWidth);
      l.upProjection = projection(kMlp, kWidth);
      l.upBias = table(kMlp, 0.1F);
      l.downProjection = projection(kWidth, kMlp);
      l.downBias = table(kWidth, 0.1F);
      w.layers.push_back(std::move(l));
    }
    w.outputNorm = norm(kWidth);
    w.anchorNorm = norm(kWidth);
    w.factorProjection = projection(kWidth, 3 * kWidth);
    w.factorBias = table(kWidth, 0.1F);
    w.outHead = projection(kWidth, kWidth);
    w.outHeadBias = table(kWidth, 0.1F);
    w.inHead = projection(kWidth, kWidth);
    w.inHeadBias = table(kWidth, 0.1F);
    w.anchorOutHead = projection(kWidth, kWidth);
    w.anchorOutHeadBias = table(kWidth, 0.1F);
    return w;
  }

private:
  MetalBackend &backend_;
  Random random_;
};

// The reference's rows: `rows` rows of a width, row-major.
using Rows = std::vector<double>;

// p's outputs for each of `rows` rows of bf16 values, rounded to bf16.
Rows project(const Projection &p, const Rows &x, uint32_t rows) {
  const uint32_t n = p.outputSize, k = p.inputSize;
  const uint16_t *raw = bf16(p.blocks().segments[0].plane0);
  std::vector<float> w(uint64_t{n} * k);
  for (uint64_t i = 0; i < w.size(); ++i) w[i] = bf16ToFloat(raw[i]);
  Rows y(uint64_t{rows} * n);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t o = 0; o < n; ++o) {
      double sum = 0.0;
      const float *weights = w.data() + uint64_t{o} * k;
      const double *input = x.data() + uint64_t{row} * k;
      for (uint32_t i = 0; i < k; ++i) sum += weights[i] * input[i];
      y[uint64_t{row} * n + o] = roundBf16(sum);
    }
  return y;
}

// The RMS norm of a row of kWidth values, by bf16 weights.
std::vector<double> rms(const double *row, const MetalBuffer &weight) {
  double squares = 0.0;
  for (uint32_t d = 0; d < kWidth; ++d) squares += row[d] * row[d];
  const double inverse = 1.0 / std::sqrt(squares / kWidth + kRmsEpsilon);
  std::vector<double> out(kWidth);
  for (uint32_t d = 0; d < kWidth; ++d) out[d] = row[d] * inverse * at(weight, d);
  return out;
}

// The row's sixteen largest tokens: value descending, id ascending on ties.
std::vector<uint32_t> top16(const float *row) {
  std::vector<uint32_t> order(kVocabulary);
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(order.begin(), order.begin() + kCandidates, order.end(), [&](uint32_t a, uint32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
  });
  order.resize(kCandidates);
  return order;
}

// One lane's edge tables, [position][predecessor][candidate], by the
// reference; also its nodes' token ids.
struct LaneReference final {
  std::vector<double> tables;
  std::vector<uint32_t> ids;
};

LaneReference reference(const LiLiCorrWeights &w, const EmbeddingWeights &embedding, const float *logits,
                        const MetalBuffer &finalHidden, const MetalBuffer &anchorRow, uint32_t lane) {
  LaneReference result{std::vector<double>(uint64_t{kPositions} * kCandidates * kCandidates, 0.0),
                       std::vector<uint32_t>(kNodes)};

  // The nodes' features and token embeddings.
  Rows features(uint64_t{kNodes} * kWidth), embedded(uint64_t{kNodes} * kHidden);
  for (uint32_t position = 0; position < kPositions; ++position) {
    const float *row = logits + (uint64_t{lane} * kRows + position + 1) * kVocabulary;
    const std::vector<uint32_t> top = top16(row);
    double maximum = row[top[0]], sum = 0.0;
    for (uint32_t token = 0; token < kVocabulary; ++token) sum += std::exp(row[token] - maximum);
    const double lse = maximum + std::log(sum);
    for (uint32_t rank = 0; rank < kRanks; ++rank) {
      const uint32_t node = position * kRanks + rank;
      result.ids[node] = top[rank];
      const double logP = row[top[rank]] - lse;
      const double f[kFeatures] = {logP, std::exp(logP), double(row[top[rank]]) - maximum,
                                   double(rank) / double(kRanks - 1), rank == 0 ? 1.0 : 0.0};
      double mean = 0.0, variance = 0.0;
      for (const double v : f) mean += v / kFeatures;
      for (const double v : f) variance += (v - mean) * (v - mean) / kFeatures;
      const double inverse = 1.0 / std::sqrt(variance + SPLASH_LILICORR_FEATURE_EPSILON);
      double normalized[kFeatures];
      for (uint32_t k = 0; k < kFeatures; ++k)
        normalized[k] = (f[k] - mean) * inverse * at(w.featureNormWeight, k) + at(w.featureNormBias, k);
      for (uint32_t j = 0; j < kWidth; ++j) {
        double v = at(w.featureUpBias, j);
        for (uint32_t k = 0; k < kFeatures; ++k) v += at(w.featureUp, uint64_t{j} * kFeatures + k) * normalized[k];
        features[uint64_t{node} * kWidth + j] = roundBf16(silu(v));
      }
      for (uint32_t d = 0; d < kHidden; ++d)
        embedded[uint64_t{node} * kHidden + d] = at(embedding.rows().rows, uint64_t{top[rank]} * kHidden + d);
    }
  }

  // x: the nodes' inputs.
  Rows pass(uint64_t{kRows} * kHidden);
  for (uint64_t i = 0; i < pass.size(); ++i) pass[i] = at(finalHidden, uint64_t{lane} * kRows * kHidden + i);
  const Rows token = project(w.tokenProjection, embedded, kNodes);
  const Rows passed = project(w.passProjection, pass, kRows);
  const Rows featureDown = project(w.featureDown, features, kNodes);
  Rows x(uint64_t{kNodes} * kWidth);
  for (uint32_t node = 0; node < kNodes; ++node) {
    const uint32_t position = node / kRanks, rank = node % kRanks;
    for (uint32_t d = 0; d < kWidth; ++d)
      x[uint64_t{node} * kWidth + d] = token[uint64_t{node} * kWidth + d] + at(w.tokenBias, d) +
                                       passed[uint64_t{position + 1} * kWidth + d] + at(w.passBias, d) +
                                       featureDown[uint64_t{node} * kWidth + d] + at(w.featureDownBias, d) +
                                       at(w.slotEmbedding, uint64_t{position} * kWidth + d) +
                                       at(w.rankEmbedding, uint64_t{rank} * kWidth + d);
  }

  // The anchor state.
  Rows anchor(kHidden);
  for (uint32_t d = 0; d < kHidden; ++d) anchor[d] = at(anchorRow, d);
  Rows context = project(w.contextProjection, anchor, 1);
  for (uint32_t d = 0; d < kWidth; ++d) context[d] += at(w.contextBias, d);
  const std::vector<double> state = rms(context.data(), w.anchorNorm.buffer);
  Rows stateRow(kWidth);
  for (uint32_t d = 0; d < kWidth; ++d) stateRow[d] = roundBf16(state[d]);
  const Rows anchorOut = project(w.anchorOutHead, stateRow, 1);

  for (const LiLiCorrLayerWeights &layer : w.layers) {
    Rows normalized(x.size());
    for (uint32_t node = 0; node < kNodes; ++node) {
      const std::vector<double> n = rms(x.data() + uint64_t{node} * kWidth, layer.attentionNorm.buffer);
      for (uint32_t d = 0; d < kWidth; ++d) normalized[uint64_t{node} * kWidth + d] = roundBf16(n[d]);
    }
    const Rows wide = project(layer.inProjection, normalized, kNodes);
    const auto qkv = [&](uint32_t node, uint32_t column) {
      return wide[uint64_t{node} * 3 * kWidth + column] + at(layer.inBias, column);
    };
    Rows attention(x.size());
    for (uint32_t head = 0; head < kHeads; ++head)
      for (uint32_t query = 0; query < kNodes; ++query) {
        std::array<double, kNodes> scores{};
        for (uint32_t key = 0; key < kNodes; ++key) {
          double dot = 0.0;
          for (uint32_t d = 0; d < kHeadDimension; ++d) {
            const uint32_t column = head * kHeadDimension + d;
            dot += qkv(query, column) * qkv(key, kWidth + column);
          }
          scores[key] = dot / std::sqrt(double(kHeadDimension)) +
                        at(w.attentionBias, (uint64_t{head} * kNodes + query) * kNodes + key);
        }
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double sum = 0.0;
        for (double &score : scores) sum += score = std::exp(score - maximum);
        for (uint32_t d = 0; d < kHeadDimension; ++d) {
          const uint32_t column = head * kHeadDimension + d;
          double value = 0.0;
          for (uint32_t key = 0; key < kNodes; ++key) value += scores[key] * qkv(key, 2 * kWidth + column);
          attention[uint64_t{query} * kWidth + column] = roundBf16(value / sum);
        }
      }
    const Rows out = project(layer.outProjection, attention, kNodes);
    for (uint64_t i = 0; i < x.size(); ++i) x[i] += out[i] + at(layer.outBias, i % kWidth);
    for (uint32_t node = 0; node < kNodes; ++node) {
      const std::vector<double> n = rms(x.data() + uint64_t{node} * kWidth, layer.mlpNorm.buffer);
      for (uint32_t d = 0; d < kWidth; ++d) normalized[uint64_t{node} * kWidth + d] = roundBf16(n[d]);
    }
    Rows mlp = project(layer.upProjection, normalized, kNodes);
    for (uint64_t i = 0; i < mlp.size(); ++i) mlp[i] = roundBf16(silu(mlp[i] + at(layer.upBias, i % kMlp)));
    const Rows down = project(layer.downProjection, mlp, kNodes);
    for (uint64_t i = 0; i < x.size(); ++i) x[i] += down[i] + at(layer.downBias, i % kWidth);
  }

  // The factor heads.
  Rows factorInput(uint64_t{kNodes} * 3 * kWidth);
  for (uint32_t node = 0; node < kNodes; ++node) {
    const std::vector<double> v = rms(x.data() + uint64_t{node} * kWidth, w.outputNorm.buffer);
    double *row = factorInput.data() + uint64_t{node} * 3 * kWidth;
    for (uint32_t d = 0; d < kWidth; ++d) {
      row[d] = roundBf16(v[d]);
      row[kWidth + d] = roundBf16(state[d]);
      row[2 * kWidth + d] = roundBf16(v[d] * state[d]);
    }
  }
  Rows factor = project(w.factorProjection, factorInput, kNodes);
  for (uint64_t i = 0; i < factor.size(); ++i) factor[i] = roundBf16(silu(factor[i] + at(w.factorBias, i % kWidth)));
  Rows outHead = project(w.outHead, factor, kNodes), inHead = project(w.inHead, factor, kNodes);
  Rows anchorHead = anchorOut;
  for (uint64_t i = 0; i < outHead.size(); ++i) {
    outHead[i] += at(w.outHeadBias, i % kWidth);
    inHead[i] += at(w.inHeadBias, i % kWidth);
  }
  for (uint32_t d = 0; d < kWidth; ++d) anchorHead[d] += at(w.anchorOutHeadBias, d);
  const auto cosine = [](const double *a, const double *b) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (uint32_t d = 0; d < kWidth; ++d) {
      dot += a[d] * b[d];
      aa += a[d] * a[d];
      bb += b[d] * b[d];
    }
    return SPLASH_LILICORR_LOGIT_SCALE * dot /
           (std::max(std::sqrt(aa), double(SPLASH_LILICORR_VECTOR_EPSILON)) *
            std::max(std::sqrt(bb), double(SPLASH_LILICORR_VECTOR_EPSILON)));
  };
  for (uint32_t position = 0; position < kPositions; ++position)
    for (uint32_t candidate = 0; candidate < kRanks; ++candidate) {
      const double *in = inHead.data() + uint64_t{position * kRanks + candidate} * kWidth;
      double *table = result.tables.data() + uint64_t{position} * kCandidates * kCandidates;
      if (position == 0) {
        table[candidate] = cosine(in, anchorHead.data());
        continue;
      }
      for (uint32_t predecessor = 0; predecessor < kRanks; ++predecessor)
        table[predecessor * kCandidates + candidate] =
            cosine(outHead.data() + uint64_t{(position - 1) * kRanks + predecessor} * kWidth, in);
    }
  return result;
}

// Everything one selection binds.
struct Fixture final {
  Fixture(MetalBackend &backend, uint32_t lanes, uint64_t seed)
      : lanes(lanes), random(seed), linear(backend.capabilities(), Int8Activations::Off), selector(kVocabulary) {
    Weights weights(backend, seed * 31 + 7);
    w = weights.lilicorr();
    embedding = EmbeddingWeights(kVocabulary, kHidden,
                                 StoredRows(weights.table(uint64_t{kVocabulary} * kHidden, 1.0F), GGUF_FMT_BF16));
    const uint32_t rows = lanes * kRows, positions = lanes * kPositions;
    const DraftSelectorWorkspace workspace = DraftSelector::workspace(positions);
    MetalBuffer logits = zeroed(backend, uint64_t{rows} * kVocabulary * sizeof(float));
    auto *values = static_cast<float *>(logits.contents());
    for (uint64_t i = 0; i < uint64_t{rows} * kVocabulary; ++i) values[i] = 4.0F * random.unit();
    buffers = {logits,
               zeroed(backend, workspace.partialIdsBytes),
               zeroed(backend, workspace.partialValuesBytes),
               zeroed(backend, workspace.candidatesBytes),
               zeroed(backend, workspace.unaryBytes),
               {},
               zeroed(backend, uint64_t{lanes} * SPLASH_TREE_CHAIN_TOKENS * sizeof(uint32_t)),
               zeroed(backend, uint64_t{lanes} * SPLASH_TREE_CHAIN_TOKENS * SPLASH_TREE_AGREEMENTS * sizeof(float)),
               zeroed(backend, uint64_t{lanes} * kBlock * sizeof(uint32_t)),
               zeroed(backend, uint64_t{lanes} * kBlock * sizeof(SplashTreeRow)),
               zeroed(backend, uint64_t{lanes} * kBlock * 2 * sizeof(float)),
               zeroed(backend, uint64_t{lanes} * sizeof(uint32_t)),
               zeroed(backend, uint64_t{lanes} * kBlock * SPLASH_TREE_CHILDREN),
               zeroed(backend, uint64_t{lanes} * kBlock * kCandidates * sizeof(float)),
               zeroed(backend, uint64_t{lanes} * kBlock * sizeof(float)),
               zeroed(backend, uint64_t{lanes} * kBlock * sizeof(uint32_t)),
               zeroed(backend, uint64_t{lanes} * kBlock * sizeof(SplashTreeRow)),
               zeroed(backend, uint64_t{lanes} * sizeof(SplashLaneRows)),
               zeroed(backend, uint64_t{lanes} * kBlock * sizeof(uint32_t))};
    // The draft's final hidden rows, whole prefill tiles of them.
    const uint64_t hiddenRows = (rows + GGUF_PREFILL_ROWS - 1) / GGUF_PREFILL_ROWS * GGUF_PREFILL_ROWS;
    finalHidden = weights.table(hiddenRows * kHidden, 1.0F);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      anchorRows.push_back(weights.table(kHidden, 1.0F));
      // A greedy chain: a row's likeliest candidate a position.
      trees.push_back({random.next() % kVocabulary, kBlock - 1, 0, 1});
    }
    std::vector<MetalBuffer> views;
    for (const uint64_t bytes : LiLiCorr::scratchBytes(kHidden)) views.push_back(zeroed(backend, bytes));
    LinearScratchSize size;
    for (const Projection *p : w.projections()) size.include(linear.scratchSize(p->shape(), LiLiCorr::maximumRows()));
    const auto optional = [&](uint64_t bytes) { return bytes ? zeroed(backend, bytes) : MetalBuffer{}; };
    scratch = LiLiCorrScratch{views[0],  views[1],  views[2],  views[3],  views[4],  views[5],  views[6],
                              views[7],  views[8],  views[9],  views[10], views[11], views[12], views[13],
                              views[14], views[15], views[16], views[17], views[18], views[19],
                              {optional(size.input), optional(size.sums), optional(size.partials),
                               optional(size.counters), optional(size.rotated)}};
  }

  void add(CommandGraph &graph, uint32_t hidden = kHidden) const {
    LiLiCorr::add(graph, linear, selector, buffers, w, scratch, {finalHidden, embedding, anchorRows}, {}, trees,
                  hidden);
  }

  uint32_t lanes;
  Random random;
  Linear linear;
  DraftSelector selector;
  LiLiCorrWeights w;
  EmbeddingWeights embedding;
  DraftSelectorBuffers buffers;
  MetalBuffer finalHidden;
  std::vector<MetalBuffer> anchorRows;
  std::vector<DraftTreeLane> trees;
  LiLiCorrScratch scratch;
};

void runCase(MetalBackend &backend, uint32_t lanes) {
  const Fixture f(backend, lanes, 0x11c0 + lanes);
  CommandGraph graph;
  f.add(graph);
  std::set<std::string> ran;
  for (const auto &dispatch : graph.dispatches()) ran.insert(dispatch.pipelineName);
  for (const char *kernel :
       {"draft_select_top16_sharded", "lilicorr_lse", "lilicorr_candidates", "lilicorr_embed_sum",
        "lilicorr_store_anchor", "lilicorr_anchor", "lilicorr_rms", "lilicorr_attention", "lilicorr_residual",
        "lilicorr_bias_silu", "lilicorr_factor_input", "lilicorr_scores", "draft_select_tree"})
    require(ran.count(kernel), std::string("the selection did not run ") + kernel);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  const auto *logits = static_cast<const float *>(f.buffers.logits.contents());
  const auto *candidates = static_cast<const uint32_t *>(f.buffers.candidates.contents());
  const auto *unary = static_cast<const float *>(f.buffers.unary.contents());
  const auto *ids = static_cast<const uint32_t *>(f.scratch.ids.contents());
  // A chain's tree: the anchor, then position p's token at row p + 1.
  const auto *tokens = static_cast<const uint32_t *>(f.buffers.treeTokens.contents());
  const auto *partials = static_cast<const float *>(f.buffers.partialValues.contents());
  double worst = 0.0;
  uint32_t decided = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const LaneReference ref = reference(f.w, f.embedding, logits, f.finalHidden, f.anchorRows[lane], lane);
    uint32_t predecessor = 0;
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t row = lane * kPositions + position;
      const std::vector<uint32_t> top =
          top16(logits + (uint64_t{lane} * kRows + position + 1) * kVocabulary);
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        require(candidates[row * kCandidates + rank] == top[rank], "a candidate differs from the exact top sixteen");
        require(unary[row * kCandidates + rank] == (rank < kRanks ? 0.0F : -INFINITY),
                "a candidate's unary score is not 0 for a node and -inf past them");
        if (rank < kRanks)
          require(ids[row * kRanks + rank] == top[rank] && ref.ids[position * kRanks + rank] == top[rank],
                  "a node's token is not its candidate's");
      }
      const float *table =
          partials + uint64_t{lanes} * kPositions * kShards * kCandidates + uint64_t{row} * kCandidates * kCandidates;
      const double *expected = ref.tables.data() + uint64_t{position} * kCandidates * kCandidates;
      for (uint32_t i = 0; i < kCandidates * kCandidates; ++i) {
        const bool edge = i / kCandidates < kRanks && i % kCandidates < kRanks && (position || i < kCandidates);
        if (!edge) {
          require(table[i] == 0.0F, "an edge table entry past the nodes' edges is not 0");
          continue;
        }
        worst = std::max(worst, std::fabs(table[i] - expected[i]));
        require(std::fabs(table[i] - expected[i]) < kTableTolerance,
                "lane " + std::to_string(lane) + " position " + std::to_string(position) + " edge " +
                    std::to_string(i) + ": " + std::to_string(table[i]) + " against " + std::to_string(expected[i]));
      }
      // The walk: the best candidate of the predecessor's row, on its own
      // table and on the reference's.
      const float *edges = table + predecessor * kCandidates;
      const double *reference = expected + predecessor * kCandidates;
      uint32_t best = 0, referenceBest = 0;
      for (uint32_t c = 1; c < kRanks; ++c) {
        if (edges[c] > edges[best]) best = c;
        if (reference[c] > reference[referenceBest]) referenceBest = c;
      }
      require(tokens[lane * kBlock + position + 1] == top[best], "the walk did not take its table row's best candidate");
      const double margin = reference[referenceBest] - reference[best];
      require(best == referenceBest || margin < 2 * kTableTolerance,
              "the walk's choice is not the reference's best beyond the noise");
      decided += best == referenceBest ? 1 : 0;
      predecessor = best;
    }
  }
  // Ties within the noise must be rare, or the walk check proves little.
  require(decided * 10 >= lanes * kPositions * 9, "the walk matched the reference at too few positions");
  std::cout << "lilicorr lanes " << lanes << ": largest table difference " << worst << ", " << decided << "/"
            << lanes * kPositions << " walk choices the reference's\n";
}

// Requests the selector cannot serve, refused before encoding anything.
void invalidRequests(MetalBackend &backend) {
  Fixture f(backend, 2, 0xbad);
  const auto refused = [&](auto change, const char *expected, const std::string &message) {
    Fixture copy = f;
    change(copy);
    CommandGraph graph;
    rejects([&] { copy.add(graph); }, expected, message);
    require(graph.empty(), message + " encoded a dispatch");
  };
  refused([](Fixture &c) { c.anchorRows.pop_back(); }, "invalid LiLiCorr selector batch",
          "anchor rows for other than every lane were accepted");
  refused([](Fixture &c) { c.w.layers.pop_back(); }, "invalid LiLiCorr selector batch",
          "a selector of one layer was accepted");
  refused([](Fixture &c) { c.trees.clear(), c.anchorRows.clear(); },
          "invalid LiLiCorr selector batch", "an empty batch was accepted");
  refused([&](Fixture &c) { c.scratch.x = backend.view(f.scratch.x, 0, f.scratch.x.sizeBytes() - 4); },
          "LiLiCorr scratch", "a residual scratch one element short was accepted");
  refused([&](Fixture &c) { c.anchorRows[1] = backend.view(f.anchorRows[1], 0, kHidden * 2 - 2); },
          "LiLiCorr anchor source", "an anchor row one value short was accepted");
  refused([&](Fixture &c) { c.buffers.candidates = backend.view(f.buffers.candidates, 0, 4); }, "draft candidate",
          "a candidate buffer too short was accepted");
  CommandGraph graph;
  rejects([&] { f.add(graph, 300); }, "invalid LiLiCorr selector batch", "a hidden width of 300 was accepted");
  require(graph.empty(), "a hidden width of 300 encoded a dispatch");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: lilicorr METALLIB");
    MetalBackend backend(argv[1]);
    invalidRequests(backend);
    runCase(backend, 1);
    runCase(backend, kLanes);
    std::cout << "lilicorr_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "lilicorr_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
