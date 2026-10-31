// DeepSeek-V4.1's kernels (kernels/shared/manifold_hyper_connection.metal,
// latent_attention.metal, latent_rotation.metal, latent_index.metal, the
// sqrt-softplus router of kernels/shared/moe.metal) through
// ops::ManifoldHyperConnection, ops::LatentAttention, ops::LatentRotation,
// ops::LatentIndex and ops::MoE, against CPU references of
// DeepSeek's model.py computation as dev/tools/deepseek41_reference.py
// carries it out (which matches model.py to 1.4e-6):
// - the manifold-constrained hyper-connections at the model's shape, four
//   streams of 5120 values: a mix's pre and post weights and its comb matrix
//   after 20 Sinkhorn iterations (hc_mixes), a sublayer's input summed by
//   the pre weights and RMS normalized (hc_pre, then the norm), and the
//   streams after a sublayer (hc_post), for rows a mix's threadgroup weighs
//   alone and together, with the model's norm epsilon of 1e-20;
// - the MoE router at the model's 384 experts, top 6: each row's experts by
//   sqrt(softplus(score)) plus the selection bias, the lower id among equal
//   scores, weighted by their unbiased weights over the sum of theirs, times
//   1.5, and the ungated shared expert after them;
// - the latent attention at the model's 64 heads over latents of 512: each
//   row's keys from the window's pool and the compressed one in any order,
//   one to 640 of them, each head's softmax beside its sink;
// - the latent rotations of the window's latents, the compressed latents, the
//   indexer's keys and queries, the queries and the attention's output: the
//   RMS norm of epsilon 1e-20, the interleaved rotation of the last 32 pairs
//   by YaRN's frequencies (and back) to past a million positions, and the FP8
//   and FP4 grids of model.py's act_quant and fp4_act_quant, and the query
//   latent's norm alone, in fp32 with the
//   reference's separate roundings: a few values may round the other way by
//   one step of their grid;
// - the compressor's groups of two rows weighted by the softmax of their gate
//   scores, and the indexer's score of every group a row reaches at
//   compression ratios 2 and 1, the groups it does not reach -inf;
// - the selection of each row's top groups (512 of up to 20,000 scores, many
//   of them equal, the lower group among equal scores, ascending, fewer where
//   fewer are above -inf) and the candidate blocks: each block's highest
//   score, the last reached block forced, the top blocks, and a later layer's
//   scores kept to them, also at the model's shape past 65,536 groups in the
//   block scratch of a piece of the indexer's rows;
// - DSpark's Markov step (kernels/decode/dspark.metal): each lane's token,
//   the argmax of its logits plus its Markov bias, the lower id among ties,
//   its confidence and its draft distribution;
// - the Engram: the host's hash of each token's table rows against the
//   reference's engram_hashes (products past int64 that wrap, negative ones
//   that Python's modulo takes), and the gate of the projected value into
//   each stream;
// - a sequence's pages and state (model/DeepSeek41Target.hpp): a KV source's
//   groups stored at the places of the positions that complete them, and
//   nothing elsewhere; the indexer's scores and the attention through the
//   page table bitwise those of pools of the same keys; a step's key lists
//   of its ring and its own latents; the ring after a step's rows, all of a
//   prefill's or a verify lane's retained ones; and a paired source's open
//   group after a verify, where its retained rows leave one.
// With `bench`, it times the kernels at the model's shapes instead: a decode
// row and a prefill chunk of 256 rows, the indexer and the selection at
// contexts of 8K to 128K tokens.
#include "HostKvExtents.hpp"

#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/model/EngramHash.hpp"
#include "../../../runtime/ops/Engram.hpp"
#include "../../../runtime/ops/LatentAttention.hpp"
#include "../../../runtime/ops/LatentIndex.hpp"
#include "../../../runtime/ops/LatentRotation.hpp"
#include "../../../runtime/ops/Linear.hpp"
#include "../../../runtime/ops/ManifoldHyperConnection.hpp"
#include "../../../runtime/ops/MoE.hpp"
#include "../../../runtime/ops/Weights.hpp"
#include "metal/abi/DSpark.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/LatentAttention.h"
#include "metal/abi/LatentIndex.h"
#include "metal/abi/LatentRotation.h"
#include "metal/abi/ManifoldHyperConnection.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
namespace kv = splash::kv;
namespace ops = splash::ops;
namespace test = splash::test;

int failures = 0;
void check(bool ok, const std::string &what) {
  std::cout << what << (ok ? " ok" : " FAIL") << '\n';
  if (!ok) ++failures;
}

uint16_t bf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  bits += 0x7FFF + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}
float f32(uint16_t v) {
  const uint32_t bits = uint32_t(v) << 16;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

template <class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = backend.allocateBuffer(std::max<uint64_t>(values.size() * sizeof(T), 16), BufferStorage::Shared,
                                              "deepseek41-test");
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}
template <class T> std::vector<T> download(const MetalBuffer &buffer, size_t count) {
  std::vector<T> out(count);
  std::memcpy(out.data(), buffer.contents(), count * sizeof(T));
  return out;
}
std::vector<float> randoms(std::mt19937 &rng, size_t n, float scale) {
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (float &x : v) x = d(rng);
  return v;
}
MetalBuffer scratch(MetalBackend &backend, uint64_t bytes) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, "deepseek41-test");
}
void run(MetalBackend &backend, CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}
std::string scientific(double value) {
  std::ostringstream out;
  out << std::scientific << std::setprecision(1) << value;
  return out.str();
}
// |got - want| within a bf16 step of want plus a small absolute floor.
bool closeBf16(double got, double want, double floor = 1e-3) {
  return std::fabs(got - want) <= std::fabs(want) * (1.0 / 128) + floor;
}

// DeepSeek-V4.1-Flash's residual: four streams of 5120 values, mixes balanced
// by 20 Sinkhorn iterations, hc_eps 1e-6 and RMS norms of epsilon 1e-20.
constexpr uint32_t kHidden = 5120, kStreams = MHC_STREAMS, kMixes = MHC_MIXES, kWidth = kStreams * kHidden;
constexpr ops::ManifoldHyperShape kShape{kHidden, 20, 1e-6f, 1e-20f};

// A row's mix weights as hc_mixes computes them, in fp64: pre, post and the
// comb matrix (row i, column j at 2 * streams + i * streams + j).
std::array<double, kMixes> mixReference(const float *row, const std::vector<float> &fn,
                                        const std::vector<float> &scale, const std::vector<float> &base) {
  double squares = 0.0;
  std::array<double, kMixes> m{};
  for (uint32_t c = 0; c < kWidth; ++c) {
    squares += double(row[c]) * row[c];
    for (uint32_t k = 0; k < kMixes; ++k) m[k] += double(row[c]) * fn[size_t(k) * kWidth + c];
  }
  const double inverse = 1.0 / std::sqrt(squares / kWidth + kShape.normEpsilon);
  for (double &v : m) v *= inverse;
  const double eps = kShape.hcEpsilon;
  std::array<double, kMixes> out{};
  for (uint32_t s = 0; s < kStreams; ++s) {
    out[s] = sigmoid(m[s] * scale[0] + base[s]) + eps;
    out[kStreams + s] = 2.0 * sigmoid(m[kStreams + s] * scale[1] + base[kStreams + s]);
  }
  double comb[kStreams][kStreams];
  for (uint32_t i = 0; i < kStreams; ++i) {
    double peak = -INFINITY, sum = 0.0;
    for (uint32_t j = 0; j < kStreams; ++j) {
      const uint32_t k = 2 * kStreams + i * kStreams + j;
      comb[i][j] = m[k] * scale[2] + base[k];
      peak = std::max(peak, comb[i][j]);
    }
    for (uint32_t j = 0; j < kStreams; ++j) sum += comb[i][j] = std::exp(comb[i][j] - peak);
    for (uint32_t j = 0; j < kStreams; ++j) comb[i][j] = comb[i][j] / sum + eps;
  }
  const auto columns = [&] {
    for (uint32_t j = 0; j < kStreams; ++j) {
      double sum = 0.0;
      for (uint32_t i = 0; i < kStreams; ++i) sum += comb[i][j];
      for (uint32_t i = 0; i < kStreams; ++i) comb[i][j] /= sum + eps;
    }
  };
  columns();
  for (uint32_t iteration = 1; iteration < kShape.iterations; ++iteration) {
    for (uint32_t i = 0; i < kStreams; ++i) {
      double sum = 0.0;
      for (uint32_t j = 0; j < kStreams; ++j) sum += comb[i][j];
      for (uint32_t j = 0; j < kStreams; ++j) comb[i][j] /= sum + eps;
    }
    columns();
  }
  for (uint32_t i = 0; i < kStreams; ++i)
    for (uint32_t j = 0; j < kStreams; ++j) out[2 * kStreams + i * kStreams + j] = comb[i][j];
  return out;
}

void manifoldHyperTest(MetalBackend &backend, uint32_t rows) {
  std::mt19937 rng(41 + rows);
  // Each stream of its own magnitude: the norm is over all of them.
  std::vector<float> streams(size_t(rows) * kWidth);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t s = 0; s < kStreams; ++s) {
      const std::vector<float> values = randoms(rng, kHidden, 0.5f * float(1u << s));
      std::copy(values.begin(), values.end(), streams.begin() + size_t(r) * kWidth + size_t(s) * kHidden);
    }
  const std::vector<float> fn = randoms(rng, size_t(kMixes) * kWidth, 1.0f / std::sqrt(float(kWidth)));
  const std::vector<float> scale{0.8f, 1.3f, 0.6f};
  const std::vector<float> base = randoms(rng, kMixes, 0.5f);
  const std::vector<float> norm = randoms(rng, kHidden, 1.0f);
  std::vector<uint16_t> branch(size_t(rows) * kHidden);
  for (uint16_t &v : branch) v = bf16(std::normal_distribution<float>(0.0f, 2.0f)(rng));

  MetalBuffer streamBuffer = upload(backend, streams), fnBuffer = upload(backend, fn),
              scaleBuffer = upload(backend, scale), baseBuffer = upload(backend, base),
              normBuffer = upload(backend, norm), branchBuffer = upload(backend, branch);
  MetalBuffer mixes = scratch(backend, uint64_t{rows} * kMixes * sizeof(float));
  MetalBuffer chunks = scratch(backend, ops::ManifoldHyperConnection::mixScratchBytes(kShape, rows));
  MetalBuffer input = scratch(backend, uint64_t{rows} * kHidden * sizeof(uint16_t));
  CommandGraph graph;
  ops::ManifoldHyperConnection::addMix(graph, streamBuffer, fnBuffer, scaleBuffer, baseBuffer, chunks, mixes, kShape,
                                       rows);
  ops::ManifoldHyperConnection::addPreNorm(graph, streamBuffer, mixes, normBuffer, input, kShape, rows);
  ops::ManifoldHyperConnection::addPost(graph, streamBuffer, branchBuffer, mixes, kShape, rows);
  run(backend, graph);

  const std::vector<float> gotMixes = download<float>(mixes, size_t(rows) * kMixes);
  const std::vector<uint16_t> gotInput = download<uint16_t>(input, size_t(rows) * kHidden);
  const std::vector<float> gotStreams = download<float>(streamBuffer, size_t(rows) * kWidth);
  double mixError = 0.0, inputMisses = 0.0, streamError = 0.0;
  bool balanced = true;
  for (uint32_t r = 0; r < rows; ++r) {
    const float *row = streams.data() + size_t(r) * kWidth;
    const std::array<double, kMixes> want = mixReference(row, fn, scale, base);
    const float *got = gotMixes.data() + size_t(r) * kMixes;
    for (uint32_t k = 0; k < kMixes; ++k) mixError = std::max(mixError, std::fabs(got[k] - want[k]));
    // Sinkhorn leaves each column of the comb matrix summing to 1, less eps.
    for (uint32_t j = 0; j < kStreams; ++j) {
      double sum = 0.0;
      for (uint32_t i = 0; i < kStreams; ++i) sum += got[2 * kStreams + i * kStreams + j];
      balanced = balanced && std::fabs(sum - 1.0) < 1e-4;
    }
    // The input and the streams after, from the mix the GPU wrote.
    double squares = 0.0;
    std::vector<double> x(kHidden);
    for (uint32_t d = 0; d < kHidden; ++d) {
      for (uint32_t s = 0; s < kStreams; ++s) x[d] += double(got[s]) * row[s * kHidden + d];
      squares += x[d] * x[d];
    }
    const double inverse = 1.0 / std::sqrt(squares / kHidden + kShape.normEpsilon);
    for (uint32_t d = 0; d < kHidden; ++d)
      if (!closeBf16(f32(gotInput[size_t(r) * kHidden + d]), x[d] * inverse * norm[d])) ++inputMisses;
    for (uint32_t j = 0; j < kStreams; ++j)
      for (uint32_t d = 0; d < kHidden; ++d) {
        double v = double(got[kStreams + j]) * f32(branch[size_t(r) * kHidden + d]), magnitude = std::fabs(v);
        for (uint32_t i = 0; i < kStreams; ++i) {
          const double term = double(got[2 * kStreams + i * kStreams + j]) * row[i * kHidden + d];
          v += term;
          magnitude += std::fabs(term);
        }
        const double error = std::fabs(gotStreams[size_t(r) * kWidth + j * kHidden + d] - v) / (magnitude + 1e-30);
        streamError = std::max(streamError, error);
      }
  }
  const std::string at = "manifold hyper-connection, " + std::to_string(rows) + " rows: ";
  check(mixError < 2e-5, at + "pre, post and Sinkhorn comb within 2e-5 of fp64 (" + scientific(mixError) + ")");
  check(balanced, at + "every comb column sums to 1");
  check(inputMisses == 0, at + "pre sum, RMS norm and weights within a bf16 step");
  check(streamError < 1e-6,
        at + "streams after the sublayer within 1e-6 of their terms (" + scientific(streamError) + ")");
}

// DeepSeek-V4.1-Flash's router: 384 experts, the top 6 by sqrt(softplus)
// plus the selection bias, weights scaled by 1.5, and one ungated shared
// expert slice; a narrower input than the model's 5120 keeps the router's
// float projection short.
constexpr ops::MoeShape kRouterShape{.hiddenSize = 512,
                                     .experts = 384,
                                     .expertsPerToken = 6,
                                     .expertIntermediateSize = 2304,
                                     .router = ops::MoeRouter::SqrtSoftplusNormalized,
                                     .routeScale = 1.5f,
                                     .sharedSlices = 1,
                                     .sharedGate = false};

double softplus(double x) { return std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x))); }

void routerTest(MetalBackend &backend, uint32_t rows) {
  const ops::MoeShape shape = kRouterShape;
  check(shape.valid(), "router shape compiles");
  const uint32_t hidden = shape.hiddenSize, experts = shape.experts, topK = shape.expertsPerToken;
  const uint32_t routes = shape.routesPerToken();
  std::mt19937 rng(7 + rows);
  std::vector<uint16_t> input(size_t(rows) * hidden);
  for (uint16_t &v : input) v = bf16(std::normal_distribution<float>(0.0f, 1.0f)(rng));
  const std::vector<float> router = randoms(rng, size_t(experts) * hidden, 0.05f);
  const std::vector<float> bias = randoms(rng, experts, 0.2f);
  MetalBuffer inputBuffer = upload(backend, input), routerBuffer = upload(backend, router),
              biasBuffer = upload(backend, bias);
  MetalBuffer scores = scratch(backend, uint64_t{rows} * shape.routerWidth() * sizeof(float));
  MetalBuffer selected = scratch(backend, uint64_t{rows} * routes * sizeof(uint32_t));
  MetalBuffer weights = scratch(backend, uint64_t{rows} * routes * sizeof(float));
  CommandGraph graph;
  ops::MoE::addRoutes(graph, inputBuffer, ops::QuantizedSegment::floats(experts, hidden, routerBuffer), {}, scores,
                      selected, weights, rows, shape, ops::FloatTile::Simdgroup, biasBuffer);
  run(backend, graph);
  const std::vector<uint32_t> gotSelected = download<uint32_t>(selected, size_t(rows) * routes);
  const std::vector<float> gotWeights = download<float>(weights, size_t(rows) * routes);
  uint32_t mismatches = 0, nearTies = 0;
  double weightError = 0.0;
  for (uint32_t r = 0; r < rows; ++r) {
    std::vector<double> weight(experts), key(experts);
    for (uint32_t e = 0; e < experts; ++e) {
      double score = 0.0;
      for (uint32_t d = 0; d < hidden; ++d)
        score += double(f32(input[size_t(r) * hidden + d])) * router[size_t(e) * hidden + d];
      weight[e] = std::sqrt(softplus(score));
      key[e] = weight[e] + bias[e];
    }
    std::vector<uint32_t> order(experts);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key[a] > key[b]; });
    // fp32 scores may order a near tie at the cut otherwise.
    if (key[order[topK - 1]] - key[order[topK]] < 1e-5) {
      ++nearTies;
      continue;
    }
    double sum = 0.0;
    for (uint32_t k = 0; k < topK; ++k) sum += weight[order[k]];
    const uint32_t *got = gotSelected.data() + size_t(r) * routes;
    const float *gotWeight = gotWeights.data() + size_t(r) * routes;
    for (uint32_t k = 0; k < topK; ++k) {
      mismatches += got[k] != order[k];
      const double want = weight[order[k]] / (sum + 1e-20) * shape.routeScale;
      weightError = std::max(weightError, std::fabs(gotWeight[k] - want) / want);
    }
    mismatches += got[topK] != experts || gotWeight[topK] != 1.0f;
  }
  const std::string at = "sqrt-softplus router, " + std::to_string(rows) + " rows: ";
  check(mismatches == 0, at + "the top 6 by biased weight, in order, then the shared expert (" +
                             std::to_string(nearTies) + " near ties skipped)");
  check(weightError < 1e-5, at + "normalized weights times 1.5 within 1e-5 (" + scientific(weightError) + ")");
}

// DeepSeek-V4.1-Flash's attention: 64 heads over latents of 512 values, a
// window of 128 latents and up to 512 compressed ones per row.
constexpr uint32_t kHeads = 64, kLatent = LATENT_ATTENTION_DIM, kWindow = 128, kSelected = 512;
constexpr uint32_t kKeyStride = kWindow + kSelected, kCompressedPool = 2000;

void latentAttentionTest(MetalBackend &backend, const std::vector<uint32_t> &keyCounts) {
  const uint32_t rows = uint32_t(keyCounts.size());
  std::mt19937 rng(17 + rows);
  const auto bf16Randoms = [&](size_t n, float scale) {
    std::vector<uint16_t> out(n);
    for (uint16_t &v : out) v = bf16(std::normal_distribution<float>(0.0f, scale)(rng));
    return out;
  };
  // Heads of four magnitudes: flat softmaxes and peaked ones.
  std::vector<uint16_t> q(size_t(rows) * kHeads * kLatent);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t h = 0; h < kHeads; ++h)
      for (uint32_t d = 0; d < kLatent; ++d)
        q[(size_t(r) * kHeads + h) * kLatent + d] =
            bf16(std::normal_distribution<float>(0.0f, 0.5f * float(1u << (h % 4)))(rng));
  const std::vector<uint16_t> window = bf16Randoms(size_t(kWindow) * kLatent, 1.0f),
                              compressed = bf16Randoms(size_t(kCompressedPool) * kLatent, 1.0f);
  const std::vector<float> sinks = randoms(rng, kHeads, 2.0f);
  // Each row's keys: window latents from a ring position on, then distinct
  // compressed ones, shuffled together.
  std::vector<uint32_t> indices(size_t(rows) * kKeyStride, 0xFFFFFFFFu), counts(rows);
  for (uint32_t r = 0; r < rows; ++r) {
    const uint32_t count = keyCounts[r], windowKeys = std::min(count, kWindow);
    std::vector<uint32_t> keys;
    const uint32_t start = rng() % kWindow;
    for (uint32_t i = 0; i < windowKeys; ++i) keys.push_back(((start + i) % kWindow) | LATENT_ATTENTION_WINDOW_KEY);
    std::vector<uint32_t> pool(kCompressedPool);
    std::iota(pool.begin(), pool.end(), 0u);
    std::shuffle(pool.begin(), pool.end(), rng);
    keys.insert(keys.end(), pool.begin(), pool.begin() + (count - windowKeys));
    std::shuffle(keys.begin(), keys.end(), rng);
    std::copy(keys.begin(), keys.end(), indices.begin() + size_t(r) * kKeyStride);
    counts[r] = count;
  }
  const float scale = 1.0f / std::sqrt(float(kLatent));
  MetalBuffer qBuffer = upload(backend, q), windowBuffer = upload(backend, window),
              compressedBuffer = upload(backend, compressed), indexBuffer = upload(backend, indices),
              countBuffer = upload(backend, counts), sinkBuffer = upload(backend, sinks);
  MetalBuffer scores = scratch(backend, uint64_t{rows} * kHeads * kKeyStride * sizeof(float));
  MetalBuffer out = scratch(backend, uint64_t{rows} * kHeads * kLatent * sizeof(float));
  const uint64_t partBytes = ops::LatentAttention::partBytes(kHeads, rows);
  MetalBuffer parts = partBytes ? scratch(backend, partBytes) : MetalBuffer{};
  CommandGraph graph;
  ops::LatentAttention::add(graph, qBuffer, windowBuffer, compressedBuffer, indexBuffer, countBuffer, sinkBuffer,
                            scores, parts, out, kHeads, kKeyStride, scale, rows);
  run(backend, graph);
  const std::vector<float> got = download<float>(out, size_t(rows) * kHeads * kLatent);
  double worst = 0.0;
  for (uint32_t r = 0; r < rows; ++r) {
    std::vector<const uint16_t *> keys;
    for (uint32_t k = 0; k < counts[r]; ++k) {
      const uint32_t index = indices[size_t(r) * kKeyStride + k];
      keys.push_back(index & LATENT_ATTENTION_WINDOW_KEY
                         ? window.data() + size_t(index & ~LATENT_ATTENTION_WINDOW_KEY) * kLatent
                         : compressed.data() + size_t(index) * kLatent);
    }
    for (uint32_t h = 0; h < kHeads; ++h) {
      const uint16_t *head = q.data() + (size_t(r) * kHeads + h) * kLatent;
      std::vector<double> s(keys.size());
      double peak = sinks[h];
      for (size_t k = 0; k < keys.size(); ++k) {
        for (uint32_t d = 0; d < kLatent; ++d) s[k] += double(f32(head[d])) * f32(keys[k][d]);
        s[k] *= scale;
        peak = std::max(peak, s[k]);
      }
      double sum = std::exp(sinks[h] - peak);
      for (double &v : s) sum += v = std::exp(v - peak);
      for (uint32_t d = 0; d < kLatent; ++d) {
        double want = 0.0, magnitude = 0.0;
        for (size_t k = 0; k < keys.size(); ++k) {
          want += s[k] / sum * f32(keys[k][d]);
          magnitude += s[k] / sum * std::fabs(f32(keys[k][d]));
        }
        const double error = std::fabs(got[(size_t(r) * kHeads + h) * kLatent + d] - want) / (magnitude + 1e-30);
        worst = std::max(worst, error);
      }
    }
  }
  std::string shape;
  for (uint32_t count : keyCounts) shape += (shape.empty() ? "" : ", ") + std::to_string(count);
  check(worst < 1e-4, "latent attention, keys per row " + shape + ": each head within 1e-4 of fp64 (" +
                          scientific(worst) + ")");
}

// DeepSeek-V4.1-Flash's rotary frequencies of 32 pairs: those of theta 10000
// on the layers without compression, of theta 160000 with YaRN (factor 16
// over 65536 positions, beta 32 and 1) on the others, in fp32 as model.py's
// precompute_freqs_cis.
std::vector<float> ropeFrequencies(bool compressed) {
  constexpr uint32_t dim = 64;
  const float base = compressed ? 160000.0f : 10000.0f;
  std::vector<float> freqs(dim / 2);
  for (uint32_t i = 0; i < dim / 2; ++i) freqs[i] = 1.0f / std::pow(base, float(2 * i) / float(dim));
  if (!compressed) return freqs;
  const double original = 65536, factor = 16;
  const auto corrected = [&](double rotations) {
    return dim * std::log(original / (rotations * 2 * M_PI)) / (2 * std::log(double(base)));
  };
  const double low = std::max(std::floor(corrected(32)), 0.0);
  const double high = std::min(std::ceil(corrected(1)), double(dim - 1));
  for (uint32_t i = 0; i < dim / 2; ++i) {
    const float ramp = float(std::clamp((double(i) - low) / std::max(high - low, 1e-3), 0.0, 1.0));
    const float smooth = 1.0f - ramp;
    freqs[i] = freqs[i] / float(factor) * (1.0f - smooth) + freqs[i] * smooth;
  }
  return freqs;
}

// The reference's grids, in fp32 (dev/tools/deepseek41_reference.py).
float ceilPow2(float x) {
  int exponent;
  const float mantissa = std::frexp(x, &exponent);
  return std::ldexp(1.0f, mantissa == 0.5f ? exponent - 1 : exponent);
}
float roundE4m3(float x) {
  int exponent;
  std::frexp(std::max(std::fabs(x), 0.015625f), &exponent);
  const float step = std::ldexp(1.0f, exponent - 4);
  return std::copysign(std::rint(std::fabs(x) / step) * step, x);
}
float roundE2m1(float x) {
  const float magnitude = std::fabs(x);
  const float step = magnitude < 2.0f ? 0.5f : (magnitude < 4.0f ? 1.0f : 2.0f);
  return std::copysign(std::rint(magnitude / step) * step, x);
}

// A vector as the reference's rms_norm, rotate and fp8_round or fp4_round
// leave it, in fp32 with separate roundings.
std::vector<float> rotationReference(std::vector<float> v, const std::vector<float> &norm,
                                     const std::vector<float> &freqs, uint32_t position,
                                     const ops::LatentRotationShape &shape) {
#pragma clang fp contract(off)
  const uint32_t width = shape.width;
  if (shape.normalize) {
    float squares = 0.0f;
    for (float x : v) squares += x * x;
    const float inverse = 1.0f / std::sqrt(squares / float(width) + shape.normEpsilon);
    for (uint32_t i = 0; i < width; ++i) v[i] = v[i] * inverse * norm[i];
  }
  const uint32_t tail = width - 2 * shape.ropePairs;
  for (uint32_t j = 0; j < shape.ropePairs; ++j) {
    const float angle = float(position) * freqs[j];
    const float cosine = std::cos(angle), sine = shape.inverse ? -std::sin(angle) : std::sin(angle);
    const float even = v[tail + 2 * j], odd = v[tail + 2 * j + 1];
    v[tail + 2 * j] = even * cosine - odd * sine;
    v[tail + 2 * j + 1] = even * sine + odd * cosine;
  }
  if (shape.rounding == ops::LatentRounding::None) return v;
  const uint32_t block = shape.rounding == ops::LatentRounding::Fp4E4m3Scale ? 16 : 32;
  for (uint32_t at = 0; at < width; at += block) {
    float amax = 0.0f;
    for (uint32_t i = at; i < at + block; ++i) amax = std::max(amax, std::fabs(v[i]));
    if (shape.rounding == ops::LatentRounding::Fp8) {
      const float scale = ceilPow2(std::max(amax, 1e-4f) / 448.0f);
      for (uint32_t i = at; i < at + block; ++i) v[i] = roundE4m3(std::clamp(v[i] / scale, -448.0f, 448.0f)) * scale;
    } else {
      const bool e4m3 = shape.rounding == ops::LatentRounding::Fp4E4m3Scale;
      const float ratio = std::max(amax, e4m3 ? 6.0f * 0x1p-9f : 6.0f * 0x1p-126f) / 6.0f;
      const float scale = e4m3 ? roundE4m3(ratio) : ceilPow2(ratio);
      for (uint32_t i = at; i < at + block; ++i) v[i] = roundE2m1(std::clamp(v[i] / scale, -6.0f, 6.0f)) * scale;
    }
  }
  return v;
}

// The key lists of rows at several positions, some before the window fills
// and some whose window wraps the ring, each with a selection of its own
// length: each row's window slots in position order, then its selected
// groups up to the first LATENT_INDEX_NONE; without a selection the window
// alone.
void keyListTest(MetalBackend &backend) {
  constexpr uint32_t window = 128, ring = 128, stride = 16, keyStride = window + stride;
  const std::vector<uint32_t> positions{0, 5, 127, 128, 300, 1000};
  const uint32_t rows = uint32_t(positions.size());
  std::vector<uint32_t> selected(size_t(rows) * stride, LATENT_INDEX_NONE);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t i = 0; i < (r * 3) % (stride + 1); ++i) selected[size_t(r) * stride + i] = 7 * i + r;
  MetalBuffer positionBuffer = upload(backend, positions), selectedBuffer = upload(backend, selected);
  for (const uint32_t selections : {stride, 0u}) {
    MetalBuffer indices = scratch(backend, uint64_t{rows} * keyStride * 4), counts = scratch(backend, rows * 4);
    CommandGraph graph;
    ops::LatentAttention::addKeys(graph, positionBuffer, selectedBuffer, indices, counts, window, ring, selections,
                                  keyStride, rows);
    run(backend, graph);
    const std::vector<uint32_t> got = download<uint32_t>(indices, size_t(rows) * keyStride),
                                gotCounts = download<uint32_t>(counts, rows);
    bool same = true;
    for (uint32_t r = 0; r < rows; ++r) {
      std::vector<uint32_t> want;
      const uint32_t p = positions[r], keys = std::min(window, p + 1);
      for (uint32_t q = p + 1 - keys; q <= p; ++q) want.push_back((q % ring) | LATENT_ATTENTION_WINDOW_KEY);
      for (uint32_t i = 0; i < selections && selected[size_t(r) * stride + i] != LATENT_INDEX_NONE; ++i)
        want.push_back(selected[size_t(r) * stride + i]);
      same = same && gotCounts[r] == want.size() &&
             std::equal(want.begin(), want.end(), got.begin() + size_t(r) * keyStride);
    }
    check(same, std::string("key lists, ") + (selections ? "window and selection" : "window alone") +
                    ": each row's window slots in order, then its groups");
  }
}

// The streams of each row, four copies of its embedding.
void streamInitTest(MetalBackend &backend) {
  constexpr uint32_t rows = 3;
  std::mt19937 rng(5);
  std::vector<uint16_t> embedding(size_t(rows) * kHidden);
  for (uint16_t &v : embedding) v = bf16(std::normal_distribution<float>(0.0f, 1.0f)(rng));
  MetalBuffer embeddingBuffer = upload(backend, embedding), streams = scratch(backend, uint64_t{rows} * kWidth * 4);
  CommandGraph graph;
  ops::ManifoldHyperConnection::addInit(graph, embeddingBuffer, streams, kShape, rows);
  run(backend, graph);
  const std::vector<float> got = download<float>(streams, size_t(rows) * kWidth);
  bool same = true;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t s = 0; s < kStreams; ++s)
      for (uint32_t d = 0; d < kHidden; ++d)
        same = same && got[(size_t(r) * kStreams + s) * kHidden + d] == f32(embedding[size_t(r) * kHidden + d]);
  check(same, "manifold hyper-connection streams: four copies of each row's embedding");
}

// Each row's streams averaged into its slot of a DSpark draft's hidden view
// (ops::ManifoldHyperConnection::addStreamMean): three target layers' slots
// of rows three slots wide, the mean bf16's rounding of the fp32 mean, and
// the other slots untouched.
void streamMeanTest(MetalBackend &backend) {
  constexpr uint32_t rows = 5, slots = 3, stride = slots * kHidden;
  std::mt19937 rng(9);
  const std::vector<float> streams = randoms(rng, size_t(rows) * kWidth, 3.0f);
  MetalBuffer streamBuffer = upload(backend, streams);
  MetalBuffer out = upload(backend, std::vector<uint16_t>(size_t(rows) * stride, 0x7fc0));
  CommandGraph graph;
  ops::ManifoldHyperConnection::addStreamMean(graph, streamBuffer, out, stride, kHidden, kShape, rows);
  run(backend, graph);
  const std::vector<uint16_t> got = download<uint16_t>(out, size_t(rows) * stride);
  bool same = true;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < stride; ++c) {
      const uint16_t value = got[size_t(r) * stride + c];
      if (c < kHidden || c >= 2 * kHidden) {
        same = same && value == 0x7fc0;
        continue;
      }
      float sum = 0.0f;
      for (uint32_t s = 0; s < kStreams; ++s) sum += streams[(size_t(r) * kStreams + s) * kHidden + c - kHidden];
      same = same && value == bf16(sum / float(kStreams));
    }
  check(same, "manifold hyper-connection stream mean: each row's streams averaged into its slot alone");
}

// Step `position` of DSpark's Markov chain (kernels/decode/dspark.metal) of
// three lanes over the model's vocabulary: each lane's token, the argmax of
// its row's logits plus its bias, the lower id of two tied tokens; its
// confidence, the sigmoid of the head's dot with the row's pre-sum of the
// streams rounded to bf16 and the Markov embedding; its draft distribution,
// the token alone; and at the block's last step the positions past it, its
// token at confidence zero. Nothing else is written.
void markovStepTest(MetalBackend &backend, uint32_t position) {
  constexpr uint32_t lanes = 3, laneRows = 8, vocabulary = 129280, rank = 256, block = 5;
  constexpr uint32_t proposals = SPLASH_DRAFT_PROPOSAL_TOKENS, candidates = SPLASH_DRAFT_CANDIDATES;
  constexpr uint32_t rows = lanes * laneRows, sentinel = 0xAAAAAAAAu;
  std::mt19937 rng(41 + position);
  std::vector<float> logits = randoms(rng, size_t(rows) * vocabulary, 4.0f);
  std::vector<float> bias = randoms(rng, size_t(lanes) * vocabulary, 1.0f);
  // Lane 1's row ties two tokens at its largest value.
  for (const uint32_t token : {64000u, 1500u}) {
    logits[(size_t(laneRows) + position) * vocabulary + token] = 90.0f;
    bias[size_t(vocabulary) + token] = 10.0f;
  }
  const std::vector<float> streams = randoms(rng, size_t(rows) * kWidth, 1.0f);
  const std::vector<float> mixes = randoms(rng, size_t(rows) * kMixes, 1.0f);
  const std::vector<float> weights = randoms(rng, kHidden + rank, 0.01f);
  std::vector<uint16_t> markov(size_t(lanes) * rank);
  for (uint16_t &value : markov) value = bf16(std::normal_distribution<float>(0.0f, 1.0f)(rng));
  MetalBuffer logitBuffer = upload(backend, logits), biasBuffer = upload(backend, bias);
  MetalBuffer streamBuffer = upload(backend, streams), mixBuffer = upload(backend, mixes);
  MetalBuffer markovBuffer = upload(backend, markov), weightBuffer = upload(backend, weights);
  MetalBuffer tokens = upload(backend, std::vector<uint32_t>(lanes * proposals, sentinel));
  MetalBuffer confidence = upload(backend, std::vector<float>(lanes * proposals, -1.0f));
  MetalBuffer ids = upload(backend, std::vector<uint32_t>(lanes * proposals * candidates, sentinel));
  MetalBuffer probabilities = upload(backend, std::vector<float>(lanes * proposals * candidates, -1.0f));
  MetalBuffer previous = upload(backend, std::vector<uint32_t>(lanes, sentinel));
  const DSparkStepParams params{lanes, vocabulary, kHidden, rank, block, position, laneRows};
  CommandGraph graph;
  graph.add("dspark_markov_step",
            {logitBuffer, biasBuffer, streamBuffer, mixBuffer, markovBuffer, weightBuffer, tokens, confidence, ids,
             probabilities, previous},
            params, {lanes, 1, 1}, {DSPARK_STEP_THREADS, 1, 1});
  run(backend, graph);
  const std::vector<uint32_t> gotTokens = download<uint32_t>(tokens, lanes * proposals);
  const std::vector<float> gotConfidence = download<float>(confidence, lanes * proposals);
  const std::vector<uint32_t> gotIds = download<uint32_t>(ids, lanes * proposals * candidates);
  const std::vector<float> gotProbabilities = download<float>(probabilities, lanes * proposals * candidates);
  const std::vector<uint32_t> gotPrevious = download<uint32_t>(previous, lanes);
  bool tokensOk = true, confidenceOk = true, distributionOk = true, untouched = true;
  double worst = 0.0;
  const uint32_t end = position + 1 == block ? proposals : position + 1;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const size_t row = size_t(lane) * laneRows + position;
    uint32_t want = 0;
    float best = -INFINITY;
    for (uint32_t token = 0; token < vocabulary; ++token) {
      const float value = logits[row * vocabulary + token] + bias[size_t(lane) * vocabulary + token];
      if (value > best) best = value, want = token;
    }
    if (lane == 1) tokensOk = tokensOk && want == 1500;
    double score = 0.0;
    for (uint32_t c = 0; c < kHidden; ++c) {
      float value = 0.0f;
      for (uint32_t k = 0; k < kStreams; ++k)
        value = std::fma(mixes[row * kMixes + k], streams[(row * kStreams + k) * kHidden + c], value);
      score += double(weights[c]) * f32(bf16(value));
    }
    for (uint32_t c = 0; c < rank; ++c) score += double(weights[kHidden + c]) * f32(markov[size_t(lane) * rank + c]);
    const double sigmoidScore = 1.0 / (1.0 + std::exp(-score));
    tokensOk = tokensOk && gotPrevious[lane] == want;
    for (uint32_t at = 0; at < proposals; ++at) {
      const size_t index = size_t(lane) * proposals + at;
      if (at < position || at >= end) {
        untouched = untouched && gotTokens[index] == sentinel && gotConfidence[index] == -1.0f;
        continue;
      }
      tokensOk = tokensOk && gotTokens[index] == want;
      const double wantConfidence = at == position ? sigmoidScore : 0.0;
      worst = std::max(worst, std::fabs(gotConfidence[index] - wantConfidence));
      for (uint32_t c = 0; c < candidates; ++c)
        distributionOk = distributionOk && gotIds[index * candidates + c] == want &&
                         gotProbabilities[index * candidates + c] == (c ? 0.0f : 1.0f);
    }
  }
  confidenceOk = worst <= 1e-5;
  const std::string step = "DSpark Markov step " + std::to_string(position) + " of " + std::to_string(block);
  check(tokensOk, step + ": each lane's token the argmax of its logits plus its bias, the lower of tied ones");
  check(confidenceOk, step + ": the confidence head's sigmoid within 1e-5 (" + scientific(worst) + ")");
  check(distributionOk, step + ": the draft distribution the token alone");
  check(untouched, step + ": no other proposal position written");
}

void latentRotationTest(MetalBackend &backend, const std::string &name, const ops::LatentRotationShape &shape,
                        bool floatInput, bool compressed, const std::vector<uint32_t> &positions) {
  const uint32_t rows = uint32_t(positions.size()), width = shape.width;
  const size_t values = size_t(rows) * shape.vectors * width;
  std::mt19937 rng(23 + width + shape.vectors);
  // Vectors of several magnitudes, the input as the GPU reads it.
  std::vector<float> input(values);
  for (size_t i = 0; i < values; ++i) {
    const float scale = std::ldexp(1.0f, int(i / width % 7) - 3);
    input[i] = std::normal_distribution<float>(0.0f, scale)(rng);
    if (!floatInput) input[i] = f32(bf16(input[i]));
  }
  const std::vector<float> norm = randoms(rng, width, 1.0f), freqs = ropeFrequencies(compressed);
  MetalBuffer inputBuffer = floatInput ? upload(backend, input) : upload(backend, [&] {
    std::vector<uint16_t> bits(values);
    for (size_t i = 0; i < values; ++i) bits[i] = bf16(input[i]);
    return bits;
  }());
  MetalBuffer normBuffer = upload(backend, norm), freqBuffer = upload(backend, freqs),
              positionBuffer = upload(backend, positions);
  MetalBuffer output = scratch(backend, values * sizeof(uint16_t));
  CommandGraph graph;
  ops::LatentRotation::add(graph, inputBuffer, floatInput, normBuffer, freqBuffer, positionBuffer, output, shape,
                           rows);
  run(backend, graph);
  const std::vector<uint16_t> got = download<uint16_t>(output, values);
  size_t exact = 0, beyondStep = 0;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t vector = 0; vector < shape.vectors; ++vector) {
      const size_t base = (size_t(r) * shape.vectors + vector) * width;
      const std::vector<float> want = rotationReference(
          std::vector<float>(input.begin() + base, input.begin() + base + width), norm, freqs, positions[r], shape);
      for (uint32_t i = 0; i < width; ++i) {
        const float g = f32(got[base + i]), w = shape.rounding == ops::LatentRounding::None ? f32(bf16(want[i]))
                                                                                            : want[i];
        exact += g == w;
        // One step of a grid of 3 mantissa bits (E4M3), or of E2M1 (a third of
        // 1.5), or a scale's doubling.
        beyondStep += std::fabs(g - w) > std::max(std::fabs(w), std::fabs(g)) * 0.5f + 1e-30f;
      }
    }
  const double share = double(exact) / double(values);
  check(share > 0.999 && beyondStep == 0, "latent rotation, " + name + ": " + scientific(1.0 - share) +
                                              " of the values round the other way, none by more than a step");
}

void compressTest(MetalBackend &backend, uint32_t groups, uint32_t ratio, bool floatInput) {
  constexpr uint32_t width = 512;
  std::mt19937 rng(31 + groups);
  const size_t rows = size_t(groups) * ratio;
  // The projections as the GPU reads them: bf16, or fp32 of BF16 weights.
  std::vector<float> kv = randoms(rng, rows * width, 1.0f), gate = randoms(rng, rows * width, 3.0f);
  if (!floatInput)
    for (std::vector<float> *values : {&kv, &gate})
      for (float &v : *values) v = f32(bf16(v));
  const auto buffer = [&](const std::vector<float> &values) {
    if (floatInput) return upload(backend, values);
    std::vector<uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(), bf16);
    return upload(backend, bits);
  };
  MetalBuffer kvBuffer = buffer(kv), gateBuffer = buffer(gate);
  MetalBuffer out = scratch(backend, uint64_t{groups} * width * sizeof(float));
  CommandGraph graph;
  ops::LatentIndex::addCompress(graph, kvBuffer, gateBuffer, out, groups, width, ratio, floatInput);
  run(backend, graph);
  const std::vector<float> got = download<float>(out, size_t(groups) * width);
  double worst = 0.0;
  for (uint32_t g = 0; g < groups; ++g)
    for (uint32_t d = 0; d < width; ++d) {
      std::vector<double> w(ratio);
      double peak = -INFINITY, sum = 0.0, want = 0.0, magnitude = 0.0;
      for (uint32_t i = 0; i < ratio; ++i) peak = std::max(peak, double(gate[(size_t(g) * ratio + i) * width + d]));
      for (uint32_t i = 0; i < ratio; ++i) sum += w[i] = std::exp(gate[(size_t(g) * ratio + i) * width + d] - peak);
      for (uint32_t i = 0; i < ratio; ++i) {
        const double term = kv[(size_t(g) * ratio + i) * width + d] * w[i] / sum;
        want += term;
        magnitude += std::fabs(term);
      }
      worst = std::max(worst, std::fabs(got[size_t(g) * width + d] - want) / (magnitude + 1e-30));
    }
  check(worst < 1e-6, std::string("compressor, ") + (floatInput ? "fp32" : "bf16") + " projections, " +
                          std::to_string(groups) + " groups of " + std::to_string(ratio) +
                          " rows: the gate's softmax weights within 1e-6 (" + scientific(worst) + ")");
}

void indexScoresTest(MetalBackend &backend, uint32_t ratio) {
  constexpr uint32_t heads = 32, dim = LATENT_INDEX_DIM, groups = 600, stride = 640;
  constexpr float weightScale = 1.0f / 64;   // 128^-0.5 * 32^-0.5
  const std::vector<uint32_t> positions{0, 1, 99, 1001, 1199};
  const uint32_t rows = uint32_t(positions.size());
  std::mt19937 rng(37 + ratio);
  std::vector<uint16_t> q(size_t(rows) * heads * dim), keys(size_t(groups) * dim);
  for (uint16_t &v : q) v = bf16(std::normal_distribution<float>(0.0f, 1.0f)(rng));
  for (uint16_t &v : keys) v = bf16(std::normal_distribution<float>(0.0f, 1.0f)(rng));
  const std::vector<float> weights = randoms(rng, size_t(rows) * heads, 1.0f);
  MetalBuffer qBuffer = upload(backend, q), weightBuffer = upload(backend, weights), keyBuffer = upload(backend, keys),
              positionBuffer = upload(backend, positions);
  MetalBuffer scores = scratch(backend, uint64_t{rows} * stride * sizeof(float));
  CommandGraph graph;
  ops::LatentIndex::addScores(graph, qBuffer, weightBuffer, keyBuffer, positionBuffer, scores, heads, groups, stride,
                              ratio, weightScale, rows);
  run(backend, graph);
  const std::vector<float> got = download<float>(scores, size_t(rows) * stride);
  double worst = 0.0;
  bool masked = true;
  for (uint32_t r = 0; r < rows; ++r) {
    const uint32_t reachable = std::min(groups, (positions[r] + 1) / ratio);
    for (uint32_t g = 0; g < stride; ++g) {
      const float value = got[size_t(r) * stride + g];
      if (g >= reachable) {
        masked = masked && value == -INFINITY;
        continue;
      }
      double want = 0.0, magnitude = 0.0;
      for (uint32_t h = 0; h < heads; ++h) {
        double dot = 0.0, dotMagnitude = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
          const double term = double(f32(q[(size_t(r) * heads + h) * dim + d])) * f32(keys[size_t(g) * dim + d]);
          dot += term;
          dotMagnitude += std::fabs(term);
        }
        const double w = double(weights[size_t(r) * heads + h]) * weightScale;
        want += std::max(dot, 0.0) * w;
        magnitude += dotMagnitude * std::fabs(w);
      }
      worst = std::max(worst, std::fabs(value - want) / (magnitude + 1e-30));
    }
  }
  const std::string at = "indexer scores, ratio " + std::to_string(ratio) + ": ";
  check(worst < 1e-6, at + "each reached group within 1e-6 of its terms (" + scientific(worst) + ")");
  check(masked, at + "-inf past the groups each row reaches");
}

// A KV source's groups in a sequence's pages, as DeepSeek41Target keeps them:
// a step of `rows` rows after 301 tokens of a paired source (ratio 2), a
// verify's few rows or a prefill's many, whose attention multiplies in
// simdgroup matrices; its pages spread over extents in a pool of two layers,
// the second the source's.
void pagedLatentTest(MetalBackend &backend, uint32_t rows) {
  constexpr uint32_t committed = 301, ratio = 2;
  const uint32_t end = committed + rows, groups = end / ratio;
  constexpr uint32_t pageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
  const uint32_t pages = (end + pageTokens - 1) / pageTokens;
  constexpr uint32_t indexHeads = 32, indexDim = LATENT_INDEX_DIM, topK = 100, keyStride = kWindow + topK;
  const kv::Layout layout{.attentionLayers = 2, .kvHeads = 1, .headDimension = kLatent,
                          .format = kv::Format::BFloat16, .valuesTokenMajor = true};
  const auto spread = test::HostKvExtents::spread(pages);
  const test::HostKvExtents extents(backend, layout, spread.extentPages, spread.extents);
  const std::vector<uint32_t> ids = test::HostKvExtents::mixedPages(spread, pages, 3);
  MetalBuffer table = scratch(backend, uint64_t{pages} * sizeof(SplashKvPage));
  extents.writeTable(ids, table.contents());
  const ops::LatentIndex::PagedKeys paged{table, extents.layer(1)};
  std::mt19937 rng(61);
  const auto bf16Randoms = [&](size_t n, float scale) {
    std::vector<uint16_t> out(n);
    for (uint16_t &v : out) v = bf16(std::normal_distribution<float>(0.0f, scale)(rng));
    return out;
  };
  const std::vector<uint16_t> latents = bf16Randoms(size_t(groups) * kLatent, 1.0f),
                              indexKeys = bf16Randoms(size_t(groups) * indexDim, 1.0f);
  MetalBuffer latentBuffer = upload(backend, latents), indexKeyBuffer = upload(backend, indexKeys);
  const std::string at = "paged, " + std::to_string(rows) + " rows: ";
  // The groups in two stores: those of the tokens before the step, then the
  // step's.
  {
    constexpr uint32_t before = committed / ratio;
    CommandGraph graph;
    ops::LatentIndex::addStore(graph, latentBuffer, indexKeyBuffer, paged, 0, before, ratio);
    ops::LatentIndex::addStore(graph, backend.view(latentBuffer, uint64_t{before} * kLatent * 2,
                                                   uint64_t{groups - before} * kLatent * 2),
                               backend.view(indexKeyBuffer, uint64_t{before} * indexDim * 2,
                                            uint64_t{groups - before} * indexDim * 2),
                               paged, before, groups - before, ratio);
    run(backend, graph);
  }
  bool stored = true;
  for (uint32_t position = 0; position < pages * pageTokens; ++position) {
    const uint32_t page = ids[position / pageTokens], token = position % pageTokens;
    const uint16_t *key = extents.slab<uint16_t>(1, SPLASH_KV_KEYS, page) + splash_kv_key_element(0, token, 0, kLatent);
    const uint16_t *value = extents.slab<uint16_t>(1, SPLASH_KV_VALUES, page) + extents.valueElement(0, token, 0);
    const bool completes = (position + 1) % ratio == 0 && (position + 1) / ratio <= groups;
    const uint32_t g = (position + 1) / ratio - 1;
    for (uint32_t d = 0; d < kLatent; ++d) {
      stored = stored && key[d] == (completes ? latents[size_t(g) * kLatent + d] : 0);
      stored = stored && value[d] == (completes && d < indexDim ? indexKeys[size_t(g) * indexDim + d] : 0);
    }
    // The first layer's region is no one's.
    for (uint32_t d = 0; d < kLatent; ++d)
      stored = stored && extents.slab<uint16_t>(0, SPLASH_KV_KEYS, page)[splash_kv_key_element(0, token, d, kLatent)] == 0;
  }
  check(stored, at + "each group's latent and index key at the place of the position that completes it, nothing "
                     "elsewhere");

  // The indexer's scores through the pages, bitwise a pool's of the keys.
  std::vector<uint32_t> positions(rows);
  std::iota(positions.begin(), positions.end(), committed);
  MetalBuffer positionBuffer = upload(backend, positions);
  {
    const std::vector<uint16_t> q = bf16Randoms(size_t(rows) * indexHeads * indexDim, 1.0f);
    const std::vector<float> weights = randoms(rng, size_t(rows) * indexHeads, 1.0f);
    MetalBuffer qBuffer = upload(backend, q), weightBuffer = upload(backend, weights);
    MetalBuffer fromPages = scratch(backend, uint64_t{rows} * groups * 4), fromPool = scratch(backend, uint64_t{rows} * groups * 4);
    CommandGraph graph;
    ops::LatentIndex::addScores(graph, qBuffer, weightBuffer, paged, positionBuffer, fromPages, indexHeads, groups,
                                groups, ratio, 1.0f / 64, rows);
    ops::LatentIndex::addScores(graph, qBuffer, weightBuffer, indexKeyBuffer, positionBuffer, fromPool, indexHeads,
                                groups, groups, ratio, 1.0f / 64, rows);
    run(backend, graph);
    check(std::memcmp(fromPages.contents(), fromPool.contents(), uint64_t{rows} * groups * 4) == 0,
          at + "indexer scores bitwise a pool's of the same keys");
  }

  // The step's key lists: its window, of the ring before it and its own
  // latents, then its selected groups as the positions of their places.
  std::vector<uint32_t> selected(size_t(rows) * topK, LATENT_INDEX_NONE);
  for (uint32_t r = 0; r < rows; ++r) {
    const uint32_t reachable = (committed + r + 1) / ratio, count = std::min(reachable, topK - (7 * r) % 64);
    std::vector<uint32_t> all(reachable);
    std::iota(all.begin(), all.end(), 0u);
    std::shuffle(all.begin(), all.end(), rng);
    std::sort(all.begin(), all.begin() + count);
    std::copy(all.begin(), all.begin() + count, selected.begin() + size_t(r) * topK);
  }
  MetalBuffer selectedBuffer = upload(backend, selected), indices = scratch(backend, uint64_t{rows} * keyStride * 4),
              counts = scratch(backend, rows * 4);
  {
    CommandGraph graph;
    ops::LatentAttention::addKeys(graph, positionBuffer, selectedBuffer, indices, counts,
                                  {kWindow, kWindow, committed, ratio}, topK, keyStride, rows);
    run(backend, graph);
  }
  const std::vector<uint32_t> gotIndices = download<uint32_t>(indices, size_t(rows) * keyStride),
                              gotCounts = download<uint32_t>(counts, rows);
  bool lists = true;
  for (uint32_t r = 0; r < rows; ++r) {
    std::vector<uint32_t> want;
    const uint32_t p = committed + r;
    for (uint32_t q = p + 1 - kWindow; q <= p; ++q)
      want.push_back((q < committed ? q % kWindow : kWindow + q - committed) | LATENT_ATTENTION_WINDOW_KEY);
    for (uint32_t i = 0; i < topK && selected[size_t(r) * topK + i] != LATENT_INDEX_NONE; ++i)
      want.push_back((selected[size_t(r) * topK + i] + 1) * ratio - 1);
    lists = lists && gotCounts[r] == want.size() &&
            std::equal(want.begin(), want.end(), gotIndices.begin() + size_t(r) * keyStride);
  }
  check(lists, at + "split key lists, the ring's slots before the step, its own latents after them, then the "
                    "places of its groups");
  // The same lists limited to lane 1's verify rows: those rows' lists as
  // before, the rows past them none.
  {
    const uint32_t verified = rows / 2 + 1;
    MetalBuffer limits = upload(backend, std::vector<uint32_t>{rows, verified, rows}),
                limitedCounts = upload(backend, std::vector<uint32_t>(rows, 0xAAAAAAAAu));
    CommandGraph graph;
    ops::LatentAttention::addKeys(graph, positionBuffer, selectedBuffer, indices, limitedCounts,
                                  {kWindow, kWindow, committed, ratio, limits, 1}, topK, keyStride, rows);
    run(backend, graph);
    const std::vector<uint32_t> limited = download<uint32_t>(limitedCounts, rows);
    bool same = true;
    for (uint32_t r = 0; r < rows; ++r) same = same && limited[r] == (r < verified ? gotCounts[r] : 0u);
    check(same, at + "key lists limited to a lane's verify rows: none past them");
  }

  // The attention over the ring, the step's latents and the pages, bitwise
  // the pooled attention's over the ring and the step's latents in one
  // window pool and the groups at their positions in a compressed one.
  const std::vector<uint16_t> ring = bf16Randoms(size_t(kWindow) * kLatent, 1.0f),
                              step = bf16Randoms(size_t(rows) * kLatent, 1.0f),
                              q = bf16Randoms(size_t(rows) * kHeads * kLatent, 1.0f);
  std::vector<uint16_t> window(ring), pool(size_t(end) * kLatent, 0);
  window.insert(window.end(), step.begin(), step.end());
  for (uint32_t g = 0; g < groups; ++g)
    std::copy_n(latents.begin() + size_t(g) * kLatent, kLatent, pool.begin() + size_t((g + 1) * ratio - 1) * kLatent);
  const std::vector<float> sinks = randoms(rng, kHeads, 2.0f);
  MetalBuffer ringBuffer = upload(backend, ring), stepBuffer = upload(backend, step), qBuffer = upload(backend, q),
              windowBuffer = upload(backend, window), poolBuffer = upload(backend, pool),
              sinkBuffer = upload(backend, sinks);
  const uint64_t outBytes = uint64_t{rows} * kHeads * kLatent * 4;
  MetalBuffer fromPages = scratch(backend, outBytes), fromPools = scratch(backend, outBytes);
  {
    const float scale = 1.0f / std::sqrt(float(kLatent));
    const uint64_t partBytes = ops::LatentAttention::partBytes(kHeads, rows);
    const auto parts = [&] { return partBytes ? scratch(backend, partBytes) : MetalBuffer{}; };
    CommandGraph graph;
    ops::LatentAttention::add(graph, qBuffer, {stepBuffer, ringBuffer, kWindow}, {{}, table, extents.layer(1)},
                              indices, counts, sinkBuffer, scratch(backend, uint64_t{rows} * kHeads * keyStride * 4),
                              parts(), fromPages, kHeads, keyStride, scale, rows);
    ops::LatentAttention::add(graph, qBuffer, windowBuffer, poolBuffer, indices, counts, sinkBuffer,
                              scratch(backend, uint64_t{rows} * kHeads * keyStride * 4), parts(), fromPools, kHeads,
                              keyStride, scale, rows);
    run(backend, graph);
  }
  check(std::memcmp(fromPages.contents(), fromPools.contents(), outBytes) == 0,
        at + "latent attention bitwise the pooled attention over the same keys");

  // The ring after the step: all its rows (a prefill's), a verify lane's
  // three retained ones, and a prefill's 300 rows from the start.
  std::vector<uint32_t> retained{rows, 3, 0, 0};
  MetalBuffer retainedBuffer = upload(backend, retained);
  const auto ringAfter = [&](const std::vector<uint16_t> &before, const std::vector<uint16_t> &latents,
                             uint32_t from, uint32_t kept) {
    std::vector<uint16_t> want(before);
    for (uint32_t p = from; p < from + kept; ++p)
      if (p + kWindow >= from + kept)
        std::copy_n(latents.begin() + size_t(p - from) * kLatent, kLatent, want.begin() + size_t(p % kWindow) * kLatent);
    return want;
  };
  const std::vector<uint16_t> longStep = bf16Randoms(size_t(300) * kLatent, 1.0f);
  MetalBuffer longStepBuffer = upload(backend, longStep);
  const uint64_t ringBytes = uint64_t{kWindow} * kLatent * 2;
  MetalBuffer whole = scratch(backend, ringBytes), lane = scratch(backend, ringBytes),
              fresh = scratch(backend, ringBytes);
  {
    CommandGraph graph;
    ops::LatentAttention::addRingUpdate(graph, ringBuffer, stepBuffer, whole, {}, committed, rows, kWindow);
    ops::LatentAttention::addRingUpdate(graph, ringBuffer, stepBuffer, lane, retainedBuffer, committed, rows, kWindow, 1);
    ops::LatentAttention::addRingUpdate(graph, ringBuffer, longStepBuffer, fresh, {}, 0, 300, kWindow);
    run(backend, graph);
  }
  check(download<uint16_t>(whole, size_t(kWindow) * kLatent) == ringAfter(ring, step, committed, rows) &&
            download<uint16_t>(lane, size_t(kWindow) * kLatent) == ringAfter(ring, step, committed, 3) &&
            download<uint16_t>(fresh, size_t(kWindow) * kLatent) == ringAfter(ring, longStep, 0, 300),
        at + "ring after a step, each slot's last latent of the retained rows, the ring's before them");

  // The open group after a verify: lane 1's two retained rows leave position
  // 302 open, the row of its projections after the open one before the
  // step; three retained rows leave none, and the group stays as it was.
  const std::vector<float> kv = randoms(rng, size_t(rows + 1) * kLatent, 1.0f),
                           gate = randoms(rng, size_t(rows + 1) * kLatent, 1.0f);
  MetalBuffer kvBuffer = upload(backend, kv), gateBuffer = upload(backend, gate);
  const std::vector<float> stale(size_t(2) * kLatent, 7.0f);
  MetalBuffer open = upload(backend, stale), closed = upload(backend, stale);
  retained = {0, 2, 3, 0};
  MetalBuffer commitRetained = upload(backend, retained);
  {
    CommandGraph graph;
    ops::LatentAttention::addOpenCommit(graph, kvBuffer, gateBuffer, open, commitRetained, committed, kLatent, 1);
    ops::LatentAttention::addOpenCommit(graph, kvBuffer, gateBuffer, closed, commitRetained, committed, kLatent, 2);
    run(backend, graph);
  }
  std::vector<float> wantOpen(kv.begin() + 2 * kLatent, kv.begin() + 3 * kLatent);
  wantOpen.insert(wantOpen.end(), gate.begin() + 2 * kLatent, gate.begin() + 3 * kLatent);
  check(download<float>(open, size_t(2) * kLatent) == wantOpen && download<float>(closed, size_t(2) * kLatent) == stale,
        at + "open group after a verify, the last retained row's projections where they leave a group open, else "
                 "as it was");
}

// Each row's top k of `scores` (rows of `stride`, the first `count` read) as
// the reference takes them: by descending score, the lower index first among
// equal ones (a stable argsort), above -inf, ascending, then the none mark.
std::vector<uint32_t> topReference(const std::vector<float> &scores, uint32_t stride, uint32_t count, uint32_t k,
                                   uint32_t row) {
  std::vector<uint32_t> order;
  for (uint32_t i = 0; i < count; ++i)
    if (scores[size_t(row) * stride + i] > -INFINITY) order.push_back(i);
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return scores[size_t(row) * stride + a] > scores[size_t(row) * stride + b];
  });
  if (order.size() > k) order.resize(k);
  std::sort(order.begin(), order.end());
  order.resize(k, LATENT_INDEX_NONE);
  return order;
}

void selectTest(MetalBackend &backend) {
  // Scores of few values, so the k-th often has equals; some rows mostly
  // -inf, one with fewer live scores than k; negative scores too.
  constexpr uint32_t count = 20000, stride = 20480, k = 512;
  const std::vector<uint32_t> live{count, 300, count, 7000};
  const uint32_t rows = uint32_t(live.size());
  std::mt19937 rng(43);
  std::vector<float> scores(size_t(rows) * stride, -INFINITY);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t i = 0; i < count; ++i)
      if (rng() % count < live[r]) scores[size_t(r) * stride + i] = float(int(rng() % 41) - 20) * 0.25f;
  MetalBuffer scoreBuffer = upload(backend, scores);
  MetalBuffer selected = scratch(backend, uint64_t{rows} * k * sizeof(uint32_t));
  CommandGraph graph;
  ops::LatentIndex::addSelect(graph, scoreBuffer, selected, count, k, stride, k, rows);
  run(backend, graph);
  const std::vector<uint32_t> got = download<uint32_t>(selected, size_t(rows) * k);
  bool same = true;
  for (uint32_t r = 0; r < rows; ++r) {
    const std::vector<uint32_t> want = topReference(scores, stride, count, k, r);
    same = same && std::equal(want.begin(), want.end(), got.begin() + size_t(r) * k);
  }
  check(same, "top-512 selection of 20000 scores with many equal ones: the reference's groups, ascending");
}

// The candidate blocks of rows at `positions` of `groups` groups of `ratio`
// tokens: each row's highest score of each block of `block` groups, its top
// `keep` blocks and its scores kept to them, the block scores in the scratch
// of pieces of up to `tile` rows (ops::LatentIndex::blockBytes).
void candidateTest(MetalBackend &backend, uint32_t groups, uint32_t stride, uint32_t block, uint32_t keep,
                   uint32_t ratio, const std::vector<uint32_t> &positions, uint32_t tile) {
  const uint32_t blocks = (groups + block - 1) / block;
  const uint32_t rows = uint32_t(positions.size());
  std::mt19937 rng(47);
  std::vector<float> scores(size_t(rows) * stride);
  for (float &s : scores) s = float(int(rng() % 200)) * 0.125f;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t g = std::min(groups, (positions[r] + 1) / ratio); g < stride; ++g)
      scores[size_t(r) * stride + g] = -INFINITY;
  MetalBuffer scoreBuffer = upload(backend, scores), positionBuffer = upload(backend, positions);
  MetalBuffer blockScores = scratch(backend, ops::LatentIndex::blockBytes(block, tile));
  MetalBuffer kept = scratch(backend, uint64_t{rows} * keep * sizeof(uint32_t));
  CommandGraph graph;
  ops::LatentIndex::addBlocks(graph, scoreBuffer, positionBuffer, blockScores, groups, stride, block, ratio, rows);
  ops::LatentIndex::addSelect(graph, blockScores, kept, blocks, keep, blocks, keep, rows);
  ops::LatentIndex::addMask(graph, scoreBuffer, kept, groups, stride, block, keep, rows);
  run(backend, graph);
  const std::vector<float> gotBlocks = download<float>(blockScores, size_t(rows) * blocks);
  const std::vector<uint32_t> gotKept = download<uint32_t>(kept, size_t(rows) * keep);
  const std::vector<float> gotScores = download<float>(scoreBuffer, size_t(rows) * stride);
  bool blocksRight = true, keptRight = true, maskRight = true;
  std::vector<float> wantBlocks(size_t(rows) * blocks, -INFINITY);
  for (uint32_t r = 0; r < rows; ++r) {
    const uint32_t reachable = std::min(groups, (positions[r] + 1) / ratio);
    for (uint32_t g = 0; g < reachable; ++g)
      wantBlocks[size_t(r) * blocks + g / block] =
          std::max(wantBlocks[size_t(r) * blocks + g / block], scores[size_t(r) * stride + g]);
    if (reachable) wantBlocks[size_t(r) * blocks + (reachable - 1) / block] = INFINITY;
    for (uint32_t b = 0; b < blocks; ++b)
      blocksRight = blocksRight && gotBlocks[size_t(r) * blocks + b] == wantBlocks[size_t(r) * blocks + b];
    const std::vector<uint32_t> want = topReference(wantBlocks, blocks, blocks, keep, r);
    keptRight = keptRight && std::equal(want.begin(), want.end(), gotKept.begin() + size_t(r) * keep);
    std::vector<bool> inKept(blocks);
    for (const uint32_t b : want)
      if (b != LATENT_INDEX_NONE) inKept[b] = true;
    for (uint32_t g = 0; g < groups; ++g) {
      const float expected = inKept[g / block] ? scores[size_t(r) * stride + g] : -INFINITY;
      maskRight = maskRight && gotScores[size_t(r) * stride + g] == expected;
    }
  }
  const std::string at = "candidate blocks of " + std::to_string(groups) + " groups: ";
  check(blocksRight, at + "each block's highest score, the last reached block +inf");
  check(keptRight, at + "the top " + std::to_string(keep) + " blocks, the lower among equal scores");
  check(maskRight, at + "a later layer's scores kept to the kept blocks");
}

// The reference's engram_hashes of a sequence, with tables of DeepSeek's
// shape (2 layers of 4-grams, 8 heads) whose multipliers overflow int64
// (dev/tools/deepseek41_reference.py on random tables; the token map holds
// the sequence's ids alone).
constexpr std::array<int64_t, 8> kEngramMultipliers{
    1769655988976616416, 3516042887798735572, 3247077371564470436,
    937296870971958950, 2828805267852691130, 3872124906513254192,
    4596580183478664977, 3579363423271359924};
constexpr std::array<int64_t, 48> kEngramPrimes{
    384006161, 384005161, 384004161, 384003161, 384002161, 384001161,
    384000161, 383999161, 383998161, 383997161, 383996161, 383995161,
    383994161, 383993161, 383992161, 383991161, 383990161, 383989161,
    383988161, 383987161, 383986161, 383985161, 383984161, 383983161,
    384016663, 384015666, 384014669, 384013672, 384012675, 384011678,
    384010681, 384009684, 384008687, 384007690, 384006693, 384005696,
    384004699, 384003702, 384002705, 384001708, 384000711, 383999714,
    383998717, 383997720, 383996723, 383995726, 383994729, 383993732};
constexpr std::array<int64_t, 48> kEngramOffsets{
    0, 1000003, 2000006, 3000009, 4000012, 5000015,
    6000018, 7000021, 8000024, 9000027, 10000030, 11000033,
    12000036, 13000039, 14000042, 15000045, 16000048, 17000051,
    18000054, 19000057, 20000060, 21000063, 22000066, 23000069,
    0, 999983, 1999966, 2999949, 3999932, 4999915,
    5999898, 6999881, 7999864, 8999847, 9999830, 10999813,
    11999796, 12999779, 13999762, 14999745, 15999728, 16999711,
    17999694, 18999677, 19999660, 20999643, 21999626, 22999609};
constexpr std::array<uint32_t, 8> kEngramTokens{
    0, 2, 128799, 5, 129279, 77,
    77, 1};
constexpr std::array<int32_t, 8> kEngramMapped{
    66469, 2244, 37655, 51064, 89054, 14768,
    14768, 79769};
constexpr std::array<uint32_t, 384> kEngramRows{
    12791501, 305834875, 247652088, 222249301, 229626514, 269783727,
    342720940, 64438992, 217177000, 95052986, 323919133, 135782119,
    298633266, 44483252, 141319399, 205147546, 79354980, 139539822,
    163726664, 151915506, 104106348, 404284351, 284478193, 128674035,
    146828700, 34089793, 315507039, 223050103, 140734651, 68560683,
    6528199, 338646883, 369791549, 203260405, 71594351, 358799083,
    296862212, 269790431, 277583740, 320242139, 278377932, 75096893,
    326047239, 263230539, 270646507, 348295143, 112181718, 330296687,
    135461734, 203145159, 288624584, 7896848, 128969273, 267837698,
    40501962, 214963387, 183860228, 22080061, 226236055, 28334888,
    196367882, 346338876, 94255709, 208103703, 248930704, 291650352,
    311480000, 308419648, 282469296, 233628944, 161898592, 67278240,
    255515097, 12442451, 270633668, 262058413, 370732352, 212643807,
    171806450, 248220281, 19307690, 370790975, 33071730, 158173025,
    362087170, 260810463, 238348600, 294701581, 311312826, 179651907,
    104855880, 86924745, 125858502, 221657151, 374320692, 199855393,
    207170009, 140364927, 52529845, 327667924, 197771842, 46845760,
    258889839, 65902757, 74992642, 28651223, 304509965, 134575546,
    286839288, 377305030, 21980611, 372842514, 392788576, 93584326,
    121514237, 92588148, 390792220, 248153131, 48658042, 176290114,
    196900138, 148479261, 204607792, 365285731, 246500403, 232266477,
    322583953, 133443147, 153243297, 163492738, 105846479, 364310216,
    170871560, 293540906, 348312558, 335186516, 349113357, 165139962,
    320184632, 46248936, 111329311, 131427040, 106542123, 36674560,
    96211027, 38868027, 248175188, 340126349, 314721510, 171960671,
    295843993, 302369315, 264339253, 389386041, 248066668, 224379295,
    318323922, 145907388, 91124854, 153976320, 158572183, 233163374,
    341938565, 100910595, 278054786, 105397816, 350911007, 246625037,
    61651757, 31913367, 309410676, 126113349, 250049727, 297205141,
    267579591, 161173077, 349754094, 41487905, 17558319, 277962345,
    54687594, 115744461, 77127250, 322837669, 104236443, 130078993,
    129640623, 102921333, 49921123, 354635719, 249072672, 117228705,
    12934616, 218920258, 91303739, 14091220, 371284862, 10878182,
    84877663, 209281144, 382115037, 35606774, 73706672, 112416570,
    151736468, 191666366, 232206264, 273356162, 212371172, 173887060,
    140832948, 113208836, 91014724, 74250612, 62916500, 57012388,
    99356398, 1764927, 203315509, 319991481, 351792843, 298719595,
    160771737, 321958953, 115325877, 162765122, 274853835, 67586320,
    308974966, 231009378, 217695252, 269032588, 177645030, 226300406,
    314929500, 59534592, 228112119, 52667638, 301193598, 205699544,
    60785795, 32484862, 287740090, 58542157, 112898385, 66804613,
    304261002, 57266230, 68873125, 296027970, 166542654, 64415338,
    373640183, 326227867, 306173551, 313477235, 225661578, 285541680,
    42115621, 263360723, 181301664, 179926605, 259235546, 35245326,
    206667410, 321169251, 318240444, 197880989, 344103561, 372893491,
    284250779, 78175425, 219718613, 238727095, 172454191, 20899901,
    168068924, 229954567, 206556830, 97875713, 244034368, 243850783,
    201964682, 118376065, 377081655, 210087009, 385384576, 134983901,
    8279358, 366159968, 56658256, 231789705, 123546993, 115934281,
    208951569, 18599696, 339697641, 233735369, 76323097, 251455986,
    375136875, 63372603, 84150492, 53476381, 66925189, 219642236,
    73563122, 396665169, 36986894, 146489780, 341186666, 237094391,
    258269342, 251396935, 313161996, 59550853, 258591847, 142258631,
    94564877, 115510585, 251011600, 29662671, 151238053, 231729059,
    271135689, 269457943, 226695821, 142849323, 263303613, 169882673,
    338680888, 385697547, 310932650, 114386197, 180052917, 123936087,
    259973417, 17919215, 70679174, 34247133, 292625253, 77808212,
    157799332, 148596452, 339633148, 201862635, 99204122, 31657609,
    383217257, 385893744, 39690070, 112591557, 48780735, 28721364,
    338518154, 210193783, 27737412, 175134202, 268396992, 307525782,
    210060949, 9012763, 313605270, 355808135, 135624349, 37068581,
    60140831, 204841099, 363778366, 199135961, 314732489, 326559263,
    234616283, 38903549, 123423766, 104172235, 328849571, 277402003,
    289886063, 366301751, 122652344, 326932285, 211149125, 159299587};

void engramHashTest() {
  std::vector<int32_t> tokenMap(129280, 0);
  for (size_t i = 0; i < kEngramTokens.size(); ++i) tokenMap[kEngramTokens[i]] = kEngramMapped[i];
  const splash::model::EngramHashTables tables{tokenMap, kEngramMultipliers, kEngramPrimes, kEngramOffsets, 2, 4, 8,
                                               2};
  bool same = true;
  for (uint64_t position = 0; position < kEngramTokens.size(); ++position) {
    std::array<uint32_t, 48> rows{};
    splash::model::engramRows(tables, kEngramTokens, position, rows.data());
    same = same && std::equal(rows.begin(), rows.end(), kEngramRows.begin() + position * 48);
  }
  check(same, "Engram hash: each token's 48 rows the reference's, from the sequence's start on");
}

void engramGateTest(MetalBackend &backend, uint32_t rows) {
  std::mt19937 rng(53 + rows);
  std::vector<float> streams(size_t(rows) * kWidth);
  for (float &v : streams) v = std::normal_distribution<float>(0.0f, 1.5f)(rng);
  std::vector<uint16_t> kv(size_t(rows) * (kStreams + 1) * kHidden);
  for (uint16_t &v : kv) v = bf16(std::normal_distribution<float>(0.0f, 0.7f)(rng));
  std::vector<float> weights(size_t(kStreams) * kHidden);
  for (float &w : weights) w = f32(bf16(std::normal_distribution<float>(0.0f, 1.0f)(rng))) *
                               f32(bf16(std::normal_distribution<float>(0.0f, 1.0f)(rng)));
  MetalBuffer streamBuffer = upload(backend, streams), kvBuffer = upload(backend, kv),
              weightBuffer = upload(backend, weights);
  CommandGraph graph;
  ops::Engram::addGate(graph, streamBuffer, kvBuffer, weightBuffer, kHidden, kShape.normEpsilon, rows);
  run(backend, graph);
  const std::vector<float> got = download<float>(streamBuffer, size_t(rows) * kWidth);
  double worst = 0.0;
  for (uint32_t r = 0; r < rows; ++r) {
    const uint16_t *rowKv = kv.data() + size_t(r) * (kStreams + 1) * kHidden;
    for (uint32_t s = 0; s < kStreams; ++s) {
      const float *h = streams.data() + size_t(r) * kWidth + size_t(s) * kHidden;
      double hh = 0.0, kk = 0.0, dot = 0.0;
      for (uint32_t d = 0; d < kHidden; ++d) {
        const double key = f32(rowKv[size_t(s) * kHidden + d]);
        hh += double(h[d]) * h[d];
        kk += key * key;
        dot += double(h[d]) * weights[size_t(s) * kHidden + d] * key;
      }
      dot = dot / std::sqrt(hh / kHidden + kShape.normEpsilon) / std::sqrt(kk / kHidden + kShape.normEpsilon) /
            std::sqrt(double(kHidden));
      const double sign = (dot > 0) - (dot < 0);
      const double gate = sigmoid(sign * std::sqrt(std::max(std::fabs(dot), 1e-6)));
      for (uint32_t d = 0; d < kHidden; ++d) {
        const double value = f32(rowKv[size_t(kStreams) * kHidden + d]);
        const double want = h[d] + gate * value;
        const double error = std::fabs(got[size_t(r) * kWidth + size_t(s) * kHidden + d] - want) /
                             (std::fabs(h[d]) + std::fabs(gate * value) + 1e-30);
        worst = std::max(worst, error);
      }
    }
  }
  check(worst < 1e-5, "Engram gate, " + std::to_string(rows) + " rows: each stream plus its gated value within 1e-5 (" +
                          scientific(worst) + ")");
}

// The GPU time of one run of the dispatches `add` adds to a graph: the wall
// time of a command of `repeats` runs, less that of a command of one, over
// the runs between them.
template <class Add> double microseconds(MetalBackend &backend, Add add, uint32_t repeats = 21) {
  const auto time = [&](uint32_t runs) {
    CommandGraph graph;
    for (uint32_t i = 0; i < runs; ++i) add(graph);
    run(backend, graph);  // warm
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i) run(backend, graph);
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / 3;
  };
  return (time(repeats) - time(1)) / (repeats - 1);
}

void bench(MetalBackend &backend) {
  std::mt19937 rng(1);
  const auto bf16Buffer = [&](size_t n) {
    std::vector<uint16_t> v(n);
    for (uint16_t &x : v) x = bf16(std::normal_distribution<float>(0.0f, 1.0f)(rng));
    return upload(backend, v);
  };
  const auto floatBuffer = [&](size_t n, float scale) { return upload(backend, randoms(rng, n, scale)); };
  for (uint32_t rows : {1u, 256u}) {
    const std::string at = std::to_string(rows) + (rows == 1 ? " row" : " rows");
    MetalBuffer streams = floatBuffer(size_t(rows) * kWidth, 1.0f), fn = floatBuffer(size_t(kMixes) * kWidth, 0.007f);
    MetalBuffer scale = upload(backend, std::vector<float>{0.8f, 1.3f, 0.6f}), base = floatBuffer(kMixes, 0.5f),
                norm = floatBuffer(kHidden, 1.0f), mixes = scratch(backend, uint64_t{rows} * kMixes * 4),
                input = scratch(backend, uint64_t{rows} * kHidden * 2), branch = bf16Buffer(size_t(rows) * kHidden),
                chunks = scratch(backend, ops::ManifoldHyperConnection::mixScratchBytes(kShape, rows));
    std::cout << "mhc mix, " << at << ": " << microseconds(backend, [&](CommandGraph &g) {
      ops::ManifoldHyperConnection::addMix(g, streams, fn, scale, base, chunks, mixes, kShape, rows);
    }) << " us\n";
    // The mix's passes alone.
    const ManifoldHyperParams mixParams{rows, kHidden, kShape.iterations, kShape.hcEpsilon, kShape.normEpsilon};
    std::cout << "  partial pass: " << microseconds(backend, [&](CommandGraph &g) {
      g.add("mhc_mix_partial", {streams, fn, chunks}, mixParams,
            {(rows + MHC_MIX_ROWS - 1) / MHC_MIX_ROWS, (kWidth + MHC_MIX_CHUNK - 1) / MHC_MIX_CHUNK, 1},
            {MHC_THREADS, 1, 1});
    }) << " us, finish pass: " << microseconds(backend, [&](CommandGraph &g) {
      g.add("mhc_mix_finish", {chunks, scale, base, mixes}, mixParams, {rows, 1, 1}, {32, 1, 1});
    }) << " us\n";
    std::cout << "mhc pre norm + post, " << at << ": " << microseconds(backend, [&](CommandGraph &g) {
      ops::ManifoldHyperConnection::addPreNorm(g, streams, mixes, norm, input, kShape, rows);
      ops::ManifoldHyperConnection::addPost(g, streams, branch, mixes, kShape, rows);
    }) << " us\n";
    // Attention over a full window and selection.
    MetalBuffer q = bf16Buffer(size_t(rows) * kHeads * kLatent), window = bf16Buffer(size_t(kWindow) * kLatent),
                compressed = bf16Buffer(size_t(kCompressedPool) * kLatent), sinks = floatBuffer(kHeads, 1.0f);
    std::vector<uint32_t> indices(size_t(rows) * kKeyStride), counts(rows, kKeyStride);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t k = 0; k < kKeyStride; ++k)
        indices[size_t(r) * kKeyStride + k] = k < kWindow ? (k | LATENT_ATTENTION_WINDOW_KEY) : k * 3;
    MetalBuffer indexBuffer = upload(backend, indices), countBuffer = upload(backend, counts),
                scores = scratch(backend, uint64_t{rows} * kHeads * kKeyStride * 4),
                out = scratch(backend, uint64_t{rows} * kHeads * kLatent * 4);
    const uint32_t splits = ops::LatentAttention::splits(rows);
    MetalBuffer parts = splits > 1 ? scratch(backend, ops::LatentAttention::partBytes(kHeads, rows)) : out;
    std::cout << "latent attention over 640 keys, " << at << ": " << microseconds(backend, [&](CommandGraph &g) {
      ops::LatentAttention::add(g, q, window, compressed, indexBuffer, countBuffer, sinks, scores, parts, out, kHeads,
                                kKeyStride, 1.0f / std::sqrt(512.0f), rows);
    }) << " us\n";
    std::vector<uint32_t> positions(rows);
    for (uint32_t r = 0; r < rows; ++r) positions[r] = 70000 + r;
    MetalBuffer positionBuffer = upload(backend, positions), freqs = upload(backend, ropeFrequencies(true)),
                rotated = scratch(backend, uint64_t{rows} * kHeads * kLatent * 2);
    // The attention's passes alone.
    // Without a ring the window's latents stand in its binding, unread.
    const LatentAttentionParams params{rows, kHeads, kKeyStride, 1.0f / std::sqrt(512.0f), splits, 0, {}};
    std::cout << "  scores pass: " << microseconds(backend, [&](CommandGraph &g) {
      g.add("latent_attention_scores", {q, window, compressed, indexBuffer, countBuffer, scores, window}, params,
            {rows, (kKeyStride + LATENT_ATTENTION_KEY_BLOCK - 1) / LATENT_ATTENTION_KEY_BLOCK, 1},
            {LATENT_ATTENTION_THREADS, 1, 1});
    }) << " us, softmax pass: " << microseconds(backend, [&](CommandGraph &g) {
      g.add("latent_attention_softmax", {scores, countBuffer, sinks}, params, {rows, kHeads, 1}, {32, 1, 1});
    }) << " us, values pass: " << microseconds(backend, [&](CommandGraph &g) {
      g.add("latent_attention_values", {scores, window, compressed, indexBuffer, countBuffer, parts, window}, params,
            {rows, kHeads / LATENT_ATTENTION_HEAD_BLOCK, splits}, {LATENT_ATTENTION_VALUE_THREADS, 1, 1});
    }) << " us (" << splits << " parts)\n";
    std::cout << "query rotation (64 x 512), " << at << ": " << microseconds(backend, [&](CommandGraph &g) {
      ops::LatentRotation::add(g, q, false, norm, freqs, positionBuffer, rotated,
                               {64, 512, 32, false, ops::LatentRounding::None, false, 0.0f}, rows);
    }) << " us\n";
    std::cout << "output rotation back and FP8 (64 x 512), " << at << ": "
              << microseconds(backend, [&](CommandGraph &g) {
                   ops::LatentRotation::add(g, out, true, norm, freqs, positionBuffer, rotated,
                                            {64, 512, 32, true, ops::LatentRounding::Fp8, false, 0.0f}, rows);
                 })
              << " us\n";
    MetalBuffer kv = bf16Buffer(size_t(rows) * 5 * kHidden), weights = floatBuffer(size_t(kStreams) * kHidden, 1.0f);
    std::cout << "Engram gate, " << at << ": " << microseconds(backend, [&](CommandGraph &g) {
      ops::Engram::addGate(g, streams, kv, weights, kHidden, 1e-20f, rows);
    }) << " us\n";
  }
  // The indexer and the selection of a decode row at long contexts.
  for (uint32_t context : {8192u, 32768u, 131072u}) {
    for (uint32_t ratio : {2u, 1u}) {
      const uint32_t groups = context / ratio, stride = groups;
      MetalBuffer q = bf16Buffer(size_t(32) * LATENT_INDEX_DIM), weights = floatBuffer(32, 1.0f),
                  keys = bf16Buffer(size_t(groups) * LATENT_INDEX_DIM),
                  positions = upload(backend, std::vector<uint32_t>{context - 1}),
                  scores = scratch(backend, uint64_t{stride} * 4), selected = scratch(backend, 512 * 4);
      const std::string at = std::to_string(context / 1024) + "K tokens, ratio " + std::to_string(ratio);
      std::cout << "indexer scores, " << at << ": " << microseconds(backend, [&](CommandGraph &g) {
        ops::LatentIndex::addScores(g, q, weights, keys, positions, scores, 32, groups, stride, ratio, 1.0f / 64, 1);
      }) << " us\n";
      std::cout << "top-512 selection, " << at << ": " << microseconds(backend, [&](CommandGraph &g) {
        ops::LatentIndex::addSelect(g, scores, selected, groups, 512, stride, 512, 1);
      }) << " us\n";
    }
  }
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      const bool timing = argc == 3 && std::string(argv[2]) == "bench";
      if (argc != 2 && !timing) throw std::runtime_error("usage: deepseek41-kernels METALLIB [bench]");
      MetalBackend backend(argv[1]);
      if (timing) {
        bench(backend);
        return 0;
      }
      // One row, a mix threadgroup's rows exactly and across groups.
      for (uint32_t rows : {1u, 3u, 4u, 5u, 9u}) manifoldHyperTest(backend, rows);
      for (uint32_t rows : {1u, 8u, 33u}) routerTest(backend, rows);
      // A decode row's full keys; a row of its own latent alone; a verify
      // step's and a prefill chunk's rows of every count.
      latentAttentionTest(backend, {kKeyStride});
      latentAttentionTest(backend, {1});
      latentAttentionTest(backend, {129, 300, kKeyStride, 37});
      // A prefill's rows, in simdgroup matrices, of key counts about the
      // blocks of their passes (64 keys scored, 32 summed, at a time).
      latentAttentionTest(backend, {1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200, 255, 256, 257, 300, 511,
                                    512, 513, 600, kKeyStride});
      keyListTest(backend);
      streamInitTest(backend);
      streamMeanTest(backend);
      // A step inside DSpark's block, and its last.
      markovStepTest(backend, 2);
      markovStepTest(backend, 4);
      compressTest(backend, 1, 2, false);
      compressTest(backend, 37, 2, false);
      compressTest(backend, 5, 1, false);
      compressTest(backend, 37, 2, true);
      indexScoresTest(backend, 2);
      indexScoresTest(backend, 1);
      // A verify step's rows, and a prefill's, whose attention multiplies in
      // simdgroup matrices.
      pagedLatentTest(backend, 5);
      pagedLatentTest(backend, 24);
      selectTest(backend);
      // Blocks of 8 groups; keep the top 16 blocks; rows reaching 0, 9, 1000
      // and every group.
      candidateTest(backend, 1500, 1536, 8, 16, 2, {0, 19, 2001, 2999}, 4);
      // DeepSeek-V4.1-Flash's candidate layer (blocks of 8 groups of a token,
      // the top 2048) at a fresh 65,793-token prompt's last chunk: the first
      // piece of its first tile of 256 rows, as many as the indexer's scores
      // fit, whose 8,225 blocks a row take more than an eighth of their bytes.
      std::vector<uint32_t> piece(ops::LatentIndex::pieceRows(65'793, 256));
      std::iota(piece.begin(), piece.end(), 65'536u);
      candidateTest(backend, 65'793, 65'793, 8, 2048, 1, piece, 256);
      engramHashTest();
      for (uint32_t rows : {1u, 5u}) engramGateTest(backend, rows);
      // DeepSeek-V4.1-Flash's rotations, at positions up to past a million.
      const std::vector<uint32_t> positions{0, 1, 127, 65535, 65536, 1'048'575};
      latentRotationTest(backend, "window latents", {1, 512, 32, false, ops::LatentRounding::Fp8, true, 1e-20f},
                         false, true, positions);
      latentRotationTest(backend, "window latents, uncompressed layer",
                         {1, 512, 32, false, ops::LatentRounding::Fp8, true, 1e-20f}, false, false, positions);
      latentRotationTest(backend, "compressed latents",
                         {1, 512, 32, false, ops::LatentRounding::Fp4E4m3Scale, true, 1e-20f}, false, true,
                         {0, 2, 254, 65534, 1'048'574});
      latentRotationTest(backend, "indexer keys", {1, 128, 32, false, ops::LatentRounding::Fp4Pow2Scale, true, 1e-20f},
                         false, true, positions);
      latentRotationTest(backend, "indexer queries",
                         {32, 128, 32, false, ops::LatentRounding::Fp4Pow2Scale, false, 0.0f}, false, true, positions);
      latentRotationTest(backend, "queries", {64, 512, 32, false, ops::LatentRounding::None, false, 0.0f}, false, true,
                         positions);
      latentRotationTest(backend, "attention output, back", {64, 512, 32, true, ops::LatentRounding::Fp8, false, 0.0f},
                         true, true, positions);
      latentRotationTest(backend, "query latent, normalized alone",
                         {1, 1280, 0, false, ops::LatentRounding::None, true, 1e-20f}, false, true, positions);
    } catch (const std::exception &error) {
      std::cerr << "DeepSeek-V4.1 kernel test error: " << error.what() << '\n';
      return 1;
    }
  }
  if (failures) {
    std::cout << "DeepSeek-V4.1 kernel tests FAILED (" << failures << " failures)\n";
    return 1;
  }
  std::cout << "DeepSeek-V4.1 kernel tests PASS\n";
  return 0;
}
