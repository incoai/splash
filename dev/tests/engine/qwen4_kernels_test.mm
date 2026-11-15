// Qwen3.8-Flash-Next's kernels (kernels/shared/hyper_connection.metal,
// ngram_embedding.metal and block_indexer.metal) through ops::HyperConnection,
// ops::NgramEmbedding and ops::BlockIndexer, against fp64 CPU references of the
// model's computation:
// - the hyper-connection mix at Flash-Next's shape: per-stream RMS norm,
//   injection weights, the low-rank gate and the stream average; the combine
//   and the streams' start; the norm's two kernels against each other, bit
//   for bit;
// - the PLE gate, its dilated convolution over a history, the history after
//   a sequence's rows and after a verify step's retained rows;
// - QSA: the indexer's normalized, rotated queries (bf16) and raw keys,
//   pooled keys (bf16) of blocks prefill steps of several sizes and a verify
//   step complete, the pending keys a step and a verify's commit keep, a
//   block a rejected row completed, and each row's block bitmap (dense below
//   the budget; above it the top blocks, ties towards the earlier block, and
//   the tail block after the row's complete ones), with YaRN too, to the end
//   of its window; the scratch a target's window takes;
// - the capture of a layer's output for the DFlash2 draft (the mean of its
//   streams); the GPU PLE hash against the CPU's;
// - each buffer the hyper-connection and PLE kernels reach, at its extent and
//   one element short, and their refusal of norms not F32.
#include "ChainRows.hpp"
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/model/Qwen4Exp.hpp"
#include "../../../runtime/model/Qwen4Steps.hpp"
#include "../../../runtime/ops/BlockIndexer.hpp"
#include "../../../runtime/ops/HyperConnection.hpp"
#include "../../../runtime/ops/NgramEmbedding.hpp"
#include "../../../runtime/ops/RoPE.hpp"
#include "../../../runtime/ops/Sampling.hpp"
#include "LinearInputReference.hpp"
#include "TestBuffers.hpp"
#include "metal/abi/BlockIndexer.h"
#include "metal/abi/KvExtent.h"
#include "metal/abi/HyperConnection.h"
#include "metal/abi/NgramEmbedding.h"
#include "ops/LaneRows.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using splash::metal::BufferReach;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::NormWeights;
namespace model = splash::model;
namespace ops = splash::ops;

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
double silu(double x) { return x * sigmoid(x); }

template <class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = backend.allocateBuffer(std::max<uint64_t>(values.size() * sizeof(T), 16), BufferStorage::Shared,
                                              "qwen4-test");
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
std::vector<uint16_t> bf16s(const std::vector<float> &v) {
  std::vector<uint16_t> out(v.size());
  std::transform(v.begin(), v.end(), out.begin(), bf16);
  return out;
}
// |got - want| within a bf16 step of want plus a small absolute floor.
bool closeBf16(double got, double want, double floor = 1e-3) {
  return std::fabs(got - want) <= std::fabs(want) * (1.0 / 128) + floor;
}

// A shared scratch buffer of `bytes` bytes.
MetalBuffer scratch(MetalBackend &backend, uint64_t bytes) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, "qwen4-test");
}

void run(MetalBackend &backend, CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

// Lanes' rows of a verify step as its proposer writes them on the device
// (SplashLaneRows), each lane's rows as a chain (a SplashTreeRow a row), and
// the step of them.
MetalBuffer laneRowsOf(MetalBackend &backend, std::span<const ops::LaneRows> lanes) {
  return upload(backend, std::vector<ops::LaneRows>(lanes.begin(), lanes.end()));
}
MetalBuffer chainsOf(MetalBackend &backend, std::span<const ops::LaneRows> lanes) {
  std::vector<ops::TreeRow> tree(splash::test::rowsEnd(lanes));
  for (const ops::LaneRows &lane : lanes)
    for (uint32_t row = 0; row < lane.rows; ++row) tree[lane.first + row] = splash::test::chainRow(row);
  return upload(backend, tree);
}
ops::StepRows stepOf(std::span<const ops::LaneRows> lanes) {
  return {static_cast<uint32_t>(lanes.size()), static_cast<uint32_t>(splash::test::rowsEnd(lanes)), splash::test::mostRows(lanes)};
}

// Flash-Next's mix: HYPER_CONNECTION_MAXIMUM_STREAMS streams of HYPER_CONNECTION_NORM_HIDDEN
// values, the hyper-connections' low rank of 320.
constexpr uint32_t kHidden = HYPER_CONNECTION_NORM_HIDDEN, kStreams = HYPER_CONNECTION_MAXIMUM_STREAMS, kRank = 320;

void hyperTest(MetalBackend &backend) {
  std::mt19937 rng(7);
  const uint32_t rows = 5, hidden = kHidden, streams = kStreams, rank = kRank, width = hidden * streams;
  const ops::HyperShape shape{hidden, streams, rank};
  const auto x = randoms(rng, rows * width, 1.5f);
  const auto norm = randoms(rng, width, 0.3f);
  std::vector<float> normPlus(width);
  for (uint32_t i = 0; i < width; ++i) normPlus[i] = 1.0f + norm[i];
  const auto inject = randoms(rng, streams * width, 0.02f);
  MetalBuffer xs = upload(backend, x), normalized = scratch(backend, rows * width * 2),
              weights = scratch(backend, rows * streams * 4);
  const NormWeights nw{upload(backend, normPlus), true};
  CommandGraph graph;
  ops::HyperConnection::addNorm(graph, xs, nw, upload(backend, inject), normalized, weights, shape, rows);
  run(backend, graph);
  const auto gotN = download<uint16_t>(normalized, rows * width);
  const auto gotW = download<float>(weights, rows * streams);
  bool okN = true, okW = true;
  for (uint32_t r = 0; r < rows; ++r) {
    std::vector<double> xn(width);
    for (uint32_t s = 0; s < streams; ++s) {
      double sum = 0;
      for (uint32_t d = 0; d < hidden; ++d)
        sum += double(x[r * width + s * hidden + d]) * x[r * width + s * hidden + d];
      const double scale = 1.0 / std::sqrt(sum / hidden + 1e-6);
      for (uint32_t d = 0; d < hidden; ++d) {
        const uint32_t c = s * hidden + d;
        xn[c] = x[r * width + c] * scale * normPlus[c];
        okN &= closeBf16(f32(gotN[r * width + c]), xn[c]);
      }
    }
    for (uint32_t j = 0; j < streams; ++j) {
      double dot = 0;
      for (uint32_t c = 0; c < width; ++c) dot += inject[j * width + c] * xn[c];
      const double want = 2 * sigmoid(dot / streams);
      okW &= std::fabs(gotW[r * streams + j] - want) <= 1e-4;
    }
  }
  check(okN, "hyper norm: per-stream RMS norm times (1 + w)");
  check(okW, "hyper norm: injection weights 2 sigmoid(inject . xn / streams)");

  // low, mix
  const auto low = randoms(rng, rows * rank, 2.0f);
  MetalBuffer lowBuffer = upload(backend, bf16s(low));
  const auto gate = randoms(rng, rows * width, 1.0f);
  MetalBuffer gateBuffer = upload(backend, bf16s(gate)), mixed = scratch(backend, rows * hidden * 2);
  CommandGraph graph2;
  ops::HyperConnection::addLow(graph2, lowBuffer, shape, rows);
  ops::HyperConnection::addMix(graph2, normalized, gateBuffer, mixed, shape, rows);
  run(backend, graph2);
  const auto gotLow = download<uint16_t>(lowBuffer, rows * rank);
  bool okLow = true;
  for (uint32_t i = 0; i < rows * rank; ++i)
    okLow &= closeBf16(f32(gotLow[i]), silu(double(f32(bf16(low[i]))) / streams));
  check(okLow, "hyper low: silu(low / streams)");
  const auto gotMix = download<uint16_t>(mixed, rows * hidden);
  bool okMix = true;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t d = 0; d < hidden; ++d) {
      double sum = 0;
      for (uint32_t s = 0; s < streams; ++s)
        sum += double(f32(gotN[r * width + s * hidden + d])) * sigmoid(f32(bf16(gate[r * width + s * hidden + d])));
      okMix &= closeBf16(f32(gotMix[r * hidden + d]), sum / streams);
    }
  check(okMix, "hyper mix: mean of streams gated by sigmoid(up)");

  // combine and init
  const auto branch = randoms(rng, rows * hidden, 1.0f);
  MetalBuffer branchBuffer = upload(backend, bf16s(branch));
  CommandGraph graph3;
  ops::HyperConnection::addCombine(graph3, xs, branchBuffer, weights, shape, rows);
  run(backend, graph3);
  const auto gotX = download<float>(xs, rows * width);
  bool okC = true;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t s = 0; s < streams; ++s)
      for (uint32_t d = 0; d < hidden; ++d) {
        const double want =
            x[r * width + s * hidden + d] + double(f32(bf16(branch[r * hidden + d]))) * gotW[r * streams + s];
        okC &= std::fabs(gotX[r * width + s * hidden + d] - want) <= 1e-5 * (1 + std::fabs(want));
      }
  check(okC, "hyper combine: streams + branch x injection weight");
  MetalBuffer streamsOut = scratch(backend, rows * width * 4);
  CommandGraph graph4;
  ops::HyperConnection::addInit(graph4, branchBuffer, streamsOut, shape, rows);
  run(backend, graph4);
  const auto gotInit = download<float>(streamsOut, rows * width);
  bool okI = true;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t s = 0; s < streams; ++s)
      for (uint32_t d = 0; d < hidden; ++d)
        okI &= gotInit[r * width + s * hidden + d] == f32(bf16(branch[r * hidden + d]));
  check(okI, "hyper init: every stream the embedding");
}

// The table variants (low and mix into a register-tile projection's Table16
// table): bit for bit the plain kernel's output followed by the projection's
// own decode_linear_gguf_prepare, over two lanes of eight rows.
void hyperTableTest(MetalBackend &backend) {
  std::mt19937 rng(11);
  const uint32_t lanes = 2, rows = 8 * lanes, hidden = kHidden, streams = kStreams, rank = kRank,
                 width = hidden * streams;
  const ops::HyperShape shape{hidden, streams, rank};
  const auto zeroed = [&](uint64_t bytes) {
    MetalBuffer buffer = scratch(backend, bytes);
    std::memset(buffer.contents(), 0, bytes);
    return buffer;
  };
  const auto table = [&](uint32_t k) {
    return ops::LinearScratch{zeroed(ops::tableBytes(k, rows)),
                              zeroed(ops::tableSumsBytes(ops::LinearInput::Table16, k, rows)), {}, {}};
  };
  const auto prepare = [&](CommandGraph &graph, const MetalBuffer &input, const ops::LinearScratch &s, uint32_t k) {
    splash::test::addReferencePreparation(graph, ops::LinearInput::Table16, input, s.input, s.sums, k, lanes);
  };
  const auto same = [](const MetalBuffer &a, const MetalBuffer &b) {
    return a.sizeBytes() == b.sizeBytes() && !std::memcmp(a.contents(), b.contents(), a.sizeBytes());
  };
  const auto sameTables = [&](const ops::LinearScratch &a, const ops::LinearScratch &b) {
    return same(a.input, b.input) && same(a.sums, b.sums);
  };
  const auto x = randoms(rng, rows * width, 1.5f);
  std::vector<float> normPlus = randoms(rng, width, 0.3f);
  for (float &v : normPlus) v += 1.0f;
  const NormWeights nw{upload(backend, normPlus), true};
  const MetalBuffer xs = upload(backend, x), inject = upload(backend, randoms(rng, streams * width, 0.02f));
  MetalBuffer normalized = zeroed(rows * width * 2), weights = zeroed(rows * streams * 4);
  CommandGraph norm;
  ops::HyperConnection::addNorm(norm, xs, nw, inject, normalized, weights, shape, rows);
  run(backend, norm);

  const std::vector<uint16_t> low = bf16s(randoms(rng, rows * rank, 2.0f));
  MetalBuffer lowPlain = upload(backend, low), lowTable = upload(backend, low);
  const ops::LinearScratch lowRef = table(rank), lowGot = table(rank);
  CommandGraph lows;
  ops::HyperConnection::addLow(lows, lowPlain, shape, rows);
  prepare(lows, lowPlain, lowRef, rank);
  ops::HyperConnection::addLow(lows, lowTable, shape, rows, lowGot, ops::LinearInput::Table16);
  run(backend, lows);
  check(sameTables(lowRef, lowGot), "hyper low table: table and sums as the prepare's");
  check(download<uint16_t>(lowTable, rows * rank) == low, "hyper low table: leaves its input as it was");

  const MetalBuffer gate = upload(backend, bf16s(randoms(rng, rows * width, 1.0f)));
  MetalBuffer mixed = zeroed(rows * hidden * 2), mixedT = zeroed(rows * hidden * 2);
  const ops::LinearScratch mixRef = table(hidden), mixGot = table(hidden);
  CommandGraph mix;
  ops::HyperConnection::addMix(mix, normalized, gate, mixed, shape, rows);
  prepare(mix, mixed, mixRef, hidden);
  ops::HyperConnection::addMix(mix, normalized, gate, mixedT, shape, rows, mixGot, ops::LinearInput::Table16);
  run(backend, mix);
  check(same(mixed, mixedT), "hyper mix table: output as the plain mix's");
  check(sameTables(mixRef, mixGot), "hyper mix table: table and sums as the prepare's");
}

// The norm's kernels (kernels/shared/hyper_connection.metal):
// ops::HyperConnection::addNorm runs an injecting mix of up to
// HYPER_CONNECTION_NORM_QUARTER_ROWS rows on the quarters kernel and the rest
// on the loaded one, which sum alike: at decode's row counts and past them,
// the two write bit for bit the same normalized rows and injection weights,
// and a mix that injects nothing the same normalized rows. Shader
// validation's instrumentation compiles the kernels apart, which can move a
// bf16 value by its last bit; under it (`exact` false) they agree within
// that, and test-engine-metal runs the exact comparison uninstrumented
// (`qwen4-kernels METALLIB bitwise`).
void hyperNormKernelsTest(MetalBackend &backend, bool exact) {
  std::mt19937 rng(13);
  const uint32_t hidden = kHidden, streams = kStreams, rank = kRank, width = hidden * streams;
  const ops::HyperShape shape{hidden, streams, rank};
  const uint32_t maximumRows = 512;
  auto x = randoms(rng, maximumRows * width, 1.5f);
  // A row of tiny values, where the norm's epsilon counts.
  for (uint32_t c = 0; c < width; ++c) x[3 * width + c] *= 1e-4f;
  std::vector<float> normPlus = randoms(rng, width, 0.3f);
  for (float &v : normPlus) v += 1.0f;
  const NormWeights nw{upload(backend, normPlus), true};
  const MetalBuffer xs = upload(backend, x), inject = upload(backend, randoms(rng, streams * width, 0.02f));
  // Equal bytes or, inexact, bf16 values (`floats` false) within their last
  // bit and floats (sums of them) within 1e-4 relative.
  const auto same = [exact](const MetalBuffer &a, const MetalBuffer &b, uint64_t bytes, bool floats = false) {
    if (!std::memcmp(a.contents(), b.contents(), bytes)) return true;
    if (exact) return false;
    if (floats) {
      const auto *x = static_cast<const float *>(a.contents()), *y = static_cast<const float *>(b.contents());
      for (uint64_t i = 0; i < bytes / 4; ++i)
        if (!(std::fabs(x[i] - y[i]) <= 1e-4f * (1.0f + std::fabs(y[i])))) return false;
      return true;
    }
    const auto *x = static_cast<const uint16_t *>(a.contents()), *y = static_cast<const uint16_t *>(b.contents());
    for (uint64_t i = 0; i < bytes / 2; ++i)
      if (std::abs(int(x[i]) - int(y[i])) > 1) return false;
    return true;
  };
  const auto zeroed = [&](uint64_t bytes) {
    MetalBuffer buffer = scratch(backend, bytes);
    std::memset(buffer.contents(), 0, bytes);
    return buffer;
  };
  bool okKernels = true, okPlain = true;
  for (const uint32_t rows : {1u, 3u, 8u, 16u, 24u, 32u, 64u, 65u, 128u, 136u, 256u, 512u}) {
    // addNorm's kernel, and the other one on the same rows.
    const bool quarters = rows <= HYPER_CONNECTION_NORM_QUARTER_ROWS;
    const std::string other = quarters ? "hyper_connection_norm_loaded" : "hyper_connection_norm_quarters";
    const uint32_t otherThreads = quarters ? HYPER_CONNECTION_ROW_THREADS : HYPER_CONNECTION_NORM_QUARTER_THREADS;
    const HyperConnectionParams params{rows, hidden, streams, rank, 1};
    MetalBuffer normalized = zeroed(rows * width * 2), weights = zeroed(rows * streams * 4);
    MetalBuffer normalizedOther = zeroed(rows * width * 2), weightsOther = zeroed(rows * streams * 4);
    MetalBuffer normalizedPlain = zeroed(rows * width * 2);
    CommandGraph graph;
    ops::HyperConnection::addNorm(graph, xs, nw, inject, normalized, weights, shape, rows);
    graph.add(other, {xs, nw.buffer, inject, normalizedOther, weightsOther}, params, {rows, 1, 1},
              {otherThreads, 1, 1});
    ops::HyperConnection::addNorm(graph, xs, nw, MetalBuffer{}, normalizedPlain, MetalBuffer{}, shape, rows);
    run(backend, graph);
    const uint64_t rowBytes = uint64_t{rows} * width * 2, weightBytes = uint64_t{rows} * streams * 4;
    const bool kernels = same(normalized, normalizedOther, rowBytes) && same(weights, weightsOther, weightBytes, true);
    okKernels &= kernels;
    if (!kernels) std::cout << "  the kernels differ at " << rows << " rows\n";
    const bool plain = same(normalizedPlain, normalized, rowBytes);
    okPlain &= plain;
    if (!plain) std::cout << "  a mix without injection differs at " << rows << " rows\n";
  }
  const std::string how = exact ? "bit for bit" : "close to (instrumented)";
  check(okKernels, "hyper norm: the quarters and loaded kernels' output and weights " + how);
  check(okPlain, "hyper norm without injection: normalized rows " + how + " the injecting mix's");
}

void pleTest(MetalBackend &backend) {
  std::mt19937 rng(11);
  const uint32_t rows = 7, hidden = 256, streams = 4, taps = 4, dilation = 3, width = hidden * streams;
  const ops::NgramShape shape{hidden, streams, taps, dilation};
  const uint32_t history = shape.history();
  const auto key = randoms(rng, rows * width, 1.0f), value = randoms(rng, rows * hidden, 1.0f),
             x = randoms(rng, rows * width, 1.0f), kn = randoms(rng, width, 0.2f), qn = randoms(rng, width, 0.2f),
             cn = randoms(rng, width, 0.2f), tapsW = randoms(rng, width * taps, 0.4f),
             hist = randoms(rng, history * width, 1.0f);
  auto plus = [](std::vector<float> v) { for (float &e : v) e += 1.0f; return v; };
  const auto knP = plus(kn), qnP = plus(qn), cnP = plus(cn);
  // The convolution input of a verify step's rows, which the commit reads.
  MetalBuffer gated = scratch(backend, rows * width * 4),
              conv = scratch(backend, uint64_t{splash::ops::kNarrowLaneRows} * width * 2);
  MetalBuffer streamsBuffer = upload(backend, x), histIn = upload(backend, bf16s(hist)),
              histOut = scratch(backend, history * width * 2);
  CommandGraph graph;
  ops::NgramEmbedding::addGate(graph, upload(backend, bf16s(key)), upload(backend, bf16s(value)), streamsBuffer,
                               {upload(backend, knP), true}, {upload(backend, qnP), true}, {upload(backend, cnP), true},
                               gated, conv, shape, rows);
  run(backend, graph);
  const auto gotG = download<float>(gated, rows * width);
  const auto gotConv = download<uint16_t>(conv, rows * width);
  std::vector<double> wantG(rows * width), wantConv(rows * width);
  bool okG = true, okConvIn = true;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t s = 0; s < streams; ++s) {
      double ks = 0, qs = 0;
      for (uint32_t d = 0; d < hidden; ++d) {
        const double kv = f32(bf16(key[r * width + s * hidden + d])), qv = x[r * width + s * hidden + d];
        ks += kv * kv;
        qs += qv * qv;
      }
      ks = 1 / std::sqrt(ks / hidden + 1e-6);
      qs = 1 / std::sqrt(qs / hidden + 1e-6);
      double dot = 0;
      for (uint32_t d = 0; d < hidden; ++d) {
        const uint32_t c = s * hidden + d;
        dot += f32(bf16(key[r * width + c])) * ks * knP[c] * x[r * width + c] * qs * qnP[c];
      }
      const double score = dot / std::sqrt(double(hidden));
      const double g = sigmoid((score > 0 ? 1 : score < 0 ? -1 : 0) * std::sqrt(std::max(std::fabs(score), 1e-6)));
      double vs = 0;
      for (uint32_t d = 0; d < hidden; ++d) vs += std::pow(f32(bf16(value[r * hidden + d])) * g, 2);
      const double cs = 1 / std::sqrt(vs / hidden + 1e-6);
      for (uint32_t d = 0; d < hidden; ++d) {
        const uint32_t c = s * hidden + d;
        wantG[r * width + c] = f32(bf16(value[r * hidden + d])) * g;
        wantConv[r * width + c] = wantG[r * width + c] * cs * cnP[c];
        okG &= std::fabs(gotG[r * width + c] - wantG[r * width + c]) <= 1e-4 * (1 + std::fabs(wantG[r * width + c]));
        okConvIn &= closeBf16(f32(gotConv[r * width + c]), wantConv[r * width + c]);
      }
    }
  }
  check(okG, "PLE gate: gated values");
  check(okConvIn, "PLE gate: normalized convolution input");

  CommandGraph graph2;
  const std::array<ops::LaneRows, 1> sequence{{{0, rows}}};
  ops::NgramEmbedding::addConvolution(graph2, conv, std::span(&histIn, 1), std::span(&histOut, 1),
                                      upload(backend, tapsW), gated, streamsBuffer, shape,
                                      laneRowsOf(backend, sequence), {1, rows, rows});
  run(backend, graph2);
  const auto gotX = download<float>(streamsBuffer, rows * width);
  const auto gotH = download<uint16_t>(histOut, history * width);
  bool okX = true, okH = true;
  auto input = [&](int t, uint32_t c) -> double {
    return t >= 0 ? f32(gotConv[t * width + c]) : f32(bf16(hist[(history + t) * width + c]));
  };
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < width; ++c) {
      double sum = 0;
      for (uint32_t k = 0; k < taps; ++k)
        sum += tapsW[c * taps + k] * input(int(r) - int((taps - 1 - k) * dilation), c);
      const double want = x[r * width + c] + gotG[r * width + c] + silu(sum);
      okX &= std::fabs(gotX[r * width + c] - want) <= 1e-4 * (1 + std::fabs(want));
    }
  for (uint32_t j = 0; j < history; ++j)
    for (uint32_t c = 0; c < width; ++c) okH &= f32(gotH[j * width + c]) == input(int(rows) - int(history) + int(j), c);
  check(okX, "PLE convolution: streams + gated + silu(dilated taps over the history)");
  check(okH, "PLE convolution: the history after the sequence");

  // The history after a lane's accepted path of its chain (a row mask the
  // GPU holds) and after a sequence's first rows (a count the host knows).
  for (uint32_t retained = 1; retained <= rows; retained += 3) {
    MetalBuffer committed = scratch(backend, history * width * 2), captured = scratch(backend, history * width * 2);
    CommandGraph graph3;
    ops::NgramEmbedding::addCommit(graph3, conv, std::span(&histIn, 1), std::span(&committed, 1),
                                   upload(backend, std::vector<uint32_t>{splash::test::chainPath(retained)}), shape,
                                   laneRowsOf(backend, sequence), {1, rows, rows});
    ops::NgramEmbedding::addHistory(graph3, conv, histIn, captured, shape, retained);
    run(backend, graph3);
    const auto got = download<uint16_t>(committed, history * width);
    const auto gotHistory = download<uint16_t>(captured, history * width);
    bool ok = true, okHistory = true;
    for (uint32_t j = 0; j < history; ++j)
      for (uint32_t c = 0; c < width; ++c) {
        const double want = input(int(retained) - int(history) + int(j), c);
        ok &= f32(got[j * width + c]) == want;
        okHistory &= f32(gotHistory[j * width + c]) == want;
      }
    check(ok, "PLE commit: the history after " + std::to_string(retained) + " retained rows");
    check(okHistory, "PLE history: the history after a sequence's first " + std::to_string(retained) + " rows");
  }
}

// A verify block's PLE rows as a tree whose rows' packed predecessors are
// not their parents (row r follows parents[r], row 0 the anchor), at `taps`
// taps `dilation` rows apart: each row's convolution taps its path's
// convolution inputs (and the history before the anchor), as the chain of its
// path computes it, and the commit of the accepted path to each of `leaves`
// writes the history after that path's rows, a chain's history after the
// same tokens. With `whole`, the tree has rows deep enough that every tap is
// their path's, and a leaf's path holds the whole history, which the commit
// takes from the path alone.
void pleTreeTest(MetalBackend &backend, std::span<const uint32_t> parents, uint32_t taps, uint32_t dilation,
                 std::span<const uint32_t> leaves, bool whole) {
  std::mt19937 rng(41);
  constexpr uint32_t hidden = 256, streams = 4, width = hidden * streams;
  const auto rows = static_cast<uint32_t>(parents.size());
  const ops::NgramShape shape{hidden, streams, taps, dilation};
  const uint32_t history = shape.history();
  std::vector<splash::ops::TreeRow> tree(rows);
  for (uint32_t r = 0; r < rows; ++r)
    tree[r] = r ? splash::ops::TreeRow{tree[parents[r]].ancestors | 1u << r, static_cast<uint16_t>(parents[r]),
                                       static_cast<uint16_t>(tree[parents[r]].depth + 1)}
                : splash::test::chainRow(0);
  const std::string scope = "PLE tree of " + std::to_string(rows) + " rows at " + std::to_string(taps) + " taps " +
                            std::to_string(dilation) + " apart";
  const auto conv = randoms(rng, rows * width, 1.0f), hist = randoms(rng, history * width, 1.0f),
             gated = randoms(rng, rows * width, 1.0f), x = randoms(rng, rows * width, 1.0f),
             tapsW = randoms(rng, width * taps, 0.4f);
  const MetalBuffer convBuffer = upload(backend, bf16s(conv)), histIn = upload(backend, bf16s(hist)),
                    tapsBuffer = upload(backend, tapsW), gatedBuffer = upload(backend, gated),
                    streamsBuffer = upload(backend, x);
  CommandGraph graph;
  const std::array<ops::LaneRows, 1> block{{{0, rows}}};
  ops::NgramEmbedding::addConvolution(graph, convBuffer, std::span(&histIn, 1), {}, tapsBuffer, gatedBuffer,
                                      streamsBuffer, shape, laneRowsOf(backend, block), {1, rows, rows},
                                      upload(backend, tree));
  run(backend, graph);
  const auto gotX = download<float>(streamsBuffer, rows * width);
  // Input `depth` of row r's path, or before the anchor the history's.
  const auto input = [&](uint32_t r, int depth, uint32_t c) -> double {
    if (depth < 0) return f32(bf16(hist[(history + depth) * width + c]));
    uint32_t row = r;
    while (tree[row].depth > uint32_t(depth)) row = tree[row].parent;
    return f32(bf16(conv[row * width + c]));
  };
  bool okX = true, differs = false;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < width; ++c) {
      double sum = 0, packed = 0;
      for (uint32_t k = 0; k < taps; ++k) {
        const int back = int((taps - 1 - k) * dilation);
        sum += tapsW[c * taps + k] * input(r, int(tree[r].depth) - back, c);
        packed += tapsW[c * taps + k] *
                  (int(r) - back >= 0 ? f32(bf16(conv[(r - back) * width + c])) : f32(bf16(hist[(history + int(r) - back) * width + c])));
      }
      const double want = x[r * width + c] + gated[r * width + c] + silu(sum);
      okX &= std::fabs(gotX[r * width + c] - want) <= 1e-4 * (1 + std::fabs(want));
      differs |= std::fabs(silu(packed) - silu(sum)) > 1e-2;
    }
  check(differs, scope + ": a row's packed predecessors are not its path's");
  check(okX, scope + ": each row's dilated taps follow its path");
  if (whole) {
    const bool everyTap = std::ranges::any_of(tree, [&](const ops::TreeRow &row) { return row.depth >= history; });
    const bool wholeHistory = std::ranges::any_of(
        leaves, [&](uint32_t leaf) { return uint32_t(std::popcount(tree[leaf].ancestors)) >= history; });
    check(everyTap && wholeHistory,
          scope + ": rows whose every tap is their path's, and a path that holds the whole history");
  }
  for (const uint32_t leaf : leaves) {
    MetalBuffer committed = scratch(backend, history * width * 2);
    CommandGraph commit;
    ops::NgramEmbedding::addCommit(commit, convBuffer, std::span(&histIn, 1), std::span(&committed, 1),
                                   upload(backend, std::vector<uint32_t>{tree[leaf].ancestors}), shape,
                                   laneRowsOf(backend, block), {1, rows, rows});
    run(backend, commit);
    const auto got = download<uint16_t>(committed, history * width);
    bool ok = true;
    for (uint32_t j = 0; j < history; ++j)
      for (uint32_t c = 0; c < width; ++c)
        ok &= f32(got[j * width + c]) == input(leaf, int(tree[leaf].depth) + 1 - int(history) + int(j), c);
    check(ok, scope + ": the history after the accepted path to row " + std::to_string(leaf));
  }
}

// A QSA lane: its tokens' raw keys, its pages of the index in a shuffled
// table of its own, and its pending keys, which the GPU passes from one of
// its steps to the next.
struct QsaLane final {
  uint32_t tokens = 0, first = 0;
  std::vector<uint32_t> table; // the lane's page ids
  std::vector<float> raw;      // every token's raw key
  MetalBuffer tableBuffer;
  std::array<MetalBuffer, 2> pending; // pending[0] before the next step
};

constexpr uint32_t kQsaDimension = BLOCK_INDEXER_DIMENSION, kQsaPage = 32;
constexpr uint64_t kQsaPendingBytes = uint64_t{BLOCK_INDEXER_PENDING} * kQsaDimension * 2;

// The bf16 raw keys of a lane's tokens [from, from + rows).
std::vector<uint16_t> qsaKeys(const QsaLane &lane, uint32_t from, uint32_t rows) {
  std::vector<uint16_t> keys(uint64_t{rows} * kQsaDimension);
  for (uint64_t i = 0; i < keys.size(); ++i) keys[i] = bf16(lane.raw[uint64_t{from} * kQsaDimension + i]);
  return keys;
}

// The records of the blocks whose last token is one of `rows` from `from`
// (a step's, as model/Qwen4Steps writes them for text).
std::vector<BlockIndexerBlock> qsaRecords(uint32_t from, uint32_t rows) {
  std::vector<BlockIndexerBlock> records;
  for (uint32_t last = from; last < from + rows; ++last)
    if (last % 4 == 3) records.push_back({last - 3, {last - 3, last - 3, last - 3}});
  return records;
}

// Pools a lane's tokens [from, from + rows) as a prefill step of one lane
// does from the step's keys and the lane's pending keys, which it then
// replaces with the step's.
void qsaPrefillStep(MetalBackend &backend, CommandGraph &graph, QsaLane &lane, uint32_t from, uint32_t rows,
                    const NormWeights &keyNorm, const MetalBuffer &invFreq, float ropeScale, uint32_t indexOffset) {
  const MetalBuffer keys = upload(backend, qsaKeys(lane, from, rows));
  const std::vector<BlockIndexerBlock> records = qsaRecords(from, rows);
  ops::BlockIndexer::addPool(
      graph, {upload(backend, records), static_cast<uint32_t>(records.size()), keys, lane.pending[0]}, keyNorm,
      invFreq, ropeScale, {rows, from, lane.tableBuffer, indexOffset, 512, 0, 0});
  ops::BlockIndexer::addPending(graph, keys, lane.pending[0], lane.pending[1], from, rows);
  std::swap(lane.pending[0], lane.pending[1]);
}

// A QSA case: `lanes` lanes, lane l with a context of tokens + 61 l tokens,
// whose last `rows` are a prompt sequence's rows, which attend sparsely past
// `top` complete blocks; one projection of every lane's rows, then each
// sequence's pool and selection, its rows scored 8 at a time. Earlier steps
// pooled the tokens before them in prefill steps of `chunk` rows. The first
// `tied` blocks of each lane's tokens have one raw key, zero on the rotated
// dimensions, so that their pooled keys and every row's scores of them are
// equal. The queries rotate by the target's RoPE tables (rope_build_tables), with
// YaRN by 4 when `yarn`: its frequencies, and its attention factor scaling
// queries and pooled keys alike, as the reference does.
struct QsaCase final {
  uint32_t tokens;
  uint32_t rows;
  uint32_t top;
  uint32_t tied;
  uint32_t lanes;
  uint32_t chunk;
  bool yarn = false;
};

void qsaTest(MetalBackend &backend, QsaCase c) {
  std::mt19937 rng(13);
  constexpr uint32_t D = kQsaDimension, H = BLOCK_INDEXER_HEADS, page = kQsaPage;
  const uint32_t rows = c.rows, top = c.top, lanes = c.lanes, laneRows = lanes * rows;
  std::vector<QsaLane> lane(lanes);
  uint32_t pages = 0, mostTokens = 0;
  const auto tiedKey = randoms(rng, D, 1.0f);
  for (uint32_t l = 0; l < lanes; ++l) {
    QsaLane &x = lane[l];
    x.tokens = c.tokens + 61 * l;
    x.first = x.tokens - rows;
    const uint32_t lanePages = (x.tokens + page - 1) / page;
    x.table.resize(lanePages);
    std::iota(x.table.begin(), x.table.end(), pages);
    std::shuffle(x.table.begin(), x.table.end(), rng);
    pages += lanePages;
    mostTokens = std::max(mostTokens, x.tokens);
    x.raw = randoms(rng, uint64_t{x.tokens} * D, 1.0f);
    for (uint32_t t = 0; t < 4 * c.tied; ++t)
      for (uint32_t d = 0; d < D; ++d) x.raw[uint64_t{t} * D + d] = d < 64 ? 0.0f : tiedKey[d];
    for (MetalBuffer &pending : x.pending) pending = upload(backend, std::vector<uint16_t>(kQsaPendingBytes / 2, 0));
  }
  const uint32_t words = (mostTokens / 4 + 2 + 31) / 32, stride = words * 32;
  // Extents of at most the pages an entry indexes, back to back, each
  // holding the layer's index past a region of other bytes: the pooled keys
  // of its pages. Each lane's index table holds its pages' entries.
  constexpr uint32_t kExtentPages = SPLASH_KV_PAGE_INDEX_MASK + 1;
  const uint32_t indexOffset = 64 * 1024, extents = (pages + kExtentPages - 1) / kExtentPages;
  const uint64_t pageBytes = uint64_t{page / 4} * D * 2,
                 extentBytes = indexOffset + std::min(pages, kExtentPages) * pageBytes;
  MetalBuffer extent = backend.allocateBuffer(extents * extentBytes, BufferStorage::Shared, "qwen4-test-extent",
                                              BufferReach::Addressed);
  check(!((extent.gpuAddress() | (extents > 1 ? extentBytes : 0)) & SPLASH_KV_PAGE_INDEX_MASK),
        "QSA extents leave room for the page index");
  // A page's pooled keys, in bf16 values from the first extent's start.
  const auto pooledAt = [&](uint32_t index) {
    return (index / kExtentPages * extentBytes + indexOffset + index % kExtentPages * pageBytes) / 2;
  };
  std::vector<MetalBuffer> tables;
  for (QsaLane &x : lane) {
    std::vector<uint64_t> entries;
    for (uint32_t index : x.table)
      entries.push_back((extent.gpuAddress() + index / kExtentPages * extentBytes) | index % kExtentPages);
    x.tableBuffer = upload(backend, entries);
    backend.setTableReach(x.tableBuffer, std::span(&extent, 1));
    tables.push_back(x.tableBuffer);
  }
  const auto qn = randoms(rng, D, 0.2f), kn = randoms(rng, D, 0.2f);
  std::vector<float> qnP(D), knP(D), invFreq(32);
  for (uint32_t d = 0; d < D; ++d) qnP[d] = 1 + qn[d], knP[d] = 1 + kn[d];
  const model::RopeScaling scaling = c.yarn ? model::RopeScaling{4.0, 262144} : model::RopeScaling{};
  model::rotaryInverseFrequencies(10'000'000.0f, 32, scaling, invFreq);
  const float ropeScale = scaling.attentionFactor();
  const NormWeights keyNorm{upload(backend, knP), true};
  const MetalBuffer invFreqBuffer = upload(backend, invFreq);

  // The tokens before the step's rows, in prefill steps.
  CommandGraph history;
  for (QsaLane &x : lane)
    for (uint32_t from = 0; from < x.first; from += c.chunk)
      qsaPrefillStep(backend, history, x, from, std::min(c.chunk, x.first - from), keyNorm, invFreqBuffer,
                     ropeScale, indexOffset);
  if (!history.empty()) run(backend, history);

  // The step: every lane's rows in one projection, then each lane's sequence.
  const auto query = randoms(rng, uint64_t{laneRows} * H * D, 1.0f);
  std::vector<float> stepKeys(uint64_t{laneRows} * D);
  std::vector<uint32_t> ropePositions;
  for (uint32_t l = 0; l < lanes; ++l)
    for (uint32_t r = 0; r < rows; ++r) {
      const uint32_t row = l * rows + r, position = lane[l].first + r;
      for (uint32_t d = 0; d < D; ++d) stepKeys[uint64_t{row} * D + d] = lane[l].raw[uint64_t{position} * D + d];
      ropePositions.insert(ropePositions.end(), 3, position);
    }
  const MetalBuffer cosT = scratch(backend, uint64_t{laneRows} * 32 * 4), sinT = scratch(backend, uint64_t{laneRows} * 32 * 4),
                    draftRope = scratch(backend, SPLASH_DRAFT_ROPE_PAIRS * 4);
  MetalBuffer queries = scratch(backend, uint64_t{laneRows} * H * D * 2), keys = scratch(backend, uint64_t{laneRows} * D * 2);
  std::vector<ops::LaneRows> blocks;
  for (uint32_t l = 0; l < lanes; ++l) blocks.push_back({l * rows, rows});
  CommandGraph graph;
  ops::RoPE::addTables(graph, upload(backend, ropePositions), draftRope, invFreqBuffer, draftRope, cosT, sinT,
                       draftRope, draftRope, {laneRows, 0, ropeScale}, laneRows);
  ops::BlockIndexer::addProject(graph, upload(backend, query), upload(backend, stepKeys), {upload(backend, qnP), true},
                                cosT, sinT, queries, keys, laneRowsOf(backend, blocks), {lanes, laneRows, rows});
  constexpr uint32_t scoreRows = 8;
  // Poisoned bitmaps: the select writes every word a tile's lists read.
  MetalBuffer scores = scratch(backend, uint64_t{scoreRows} * stride * 4),
              mask = upload(backend, std::vector<uint32_t>(uint64_t{laneRows} * words, ~0u));
  for (uint32_t l = 0; l < lanes; ++l) {
    const auto rowsOfLane = [&](const MetalBuffer &buffer, uint64_t rowBytes) {
      return backend.view(buffer, uint64_t{l} * rows * rowBytes, uint64_t{rows} * rowBytes);
    };
    const ops::IndexerSequence sequence{rows, lane[l].first, tables[l], indexOffset, top, words, stride};
    const std::vector<BlockIndexerBlock> records = qsaRecords(lane[l].first, rows);
    ops::BlockIndexer::addPool(graph,
                               {upload(backend, records), static_cast<uint32_t>(records.size()),
                                rowsOfLane(keys, D * 2), lane[l].pending[0]},
                               keyNorm, invFreqBuffer, ropeScale, sequence);
    ops::BlockIndexer::addSelect(graph, rowsOfLane(queries, H * D * 2), scores, scoreRows,
                                 rowsOfLane(mask, uint64_t{words} * 4), sequence);
  }
  run(backend, graph);

  // A rotation's angle is the fp32 product the RoPE tables take (the
  // queries' and the pool's alike): at thousands of tokens its rounding moves
  // a value by more than a bf16 tolerance. YaRN scales the rotated
  // dimensions of queries and keys alike.
  auto rotate = [&](std::vector<double> v, uint32_t position) {
    for (uint32_t i = 0; i < 32; ++i) {
      const double a = float(position) * invFreq[i], c = std::cos(a) * ropeScale, s = std::sin(a) * ropeScale;
      const double x0 = v[i], x1 = v[i + 32];
      v[i] = x0 * c - x1 * s;
      v[i + 32] = x1 * c + x0 * s;
    }
    return v;
  };
  auto rmsNorm = [&](std::vector<double> v, const std::vector<float> &w) {
    double sum = 0;
    for (double e : v) sum += e * e;
    const double scale = 1 / std::sqrt(sum / D + 1e-6);
    for (uint32_t d = 0; d < D; ++d) v[d] *= scale * w[d];
    return v;
  };
  // Queries and pooled keys are stored bf16: within a rounding of the fp64
  // value, a relative 2^-8, past the fp32 kernel's own error.
  const auto nearBf16 = [](double got, double want) { return std::fabs(got - want) <= std::fabs(want) / 256 + 2e-4; };
  std::vector<float> gotQ;
  for (const uint16_t value : download<uint16_t>(queries, uint64_t{laneRows} * H * D)) gotQ.push_back(f32(value));
  std::vector<float> gotP;
  for (const uint16_t value : download<uint16_t>(extent, extents * extentBytes / 2)) gotP.push_back(f32(value));
  const auto gotKeys = download<uint16_t>(keys, uint64_t{laneRows} * D);
  const auto gotMask = download<uint32_t>(mask, uint64_t{laneRows} * words);
  bool okQ = true, okKeys = true, okP = true, okMask = true;
  for (uint32_t l = 0; l < lanes; ++l) {
    const QsaLane &x = lane[l];
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t h = 0; h < H; ++h) {
        const uint64_t head = (uint64_t{l} * rows + r) * H + h;
        std::vector<double> v(query.begin() + head * D, query.begin() + (head + 1) * D);
        const std::vector<double> want = rotate(rmsNorm(v, qnP), x.first + r);
        for (uint32_t d = 0; d < D; ++d) okQ &= nearBf16(gotQ[head * D + d], want[d]);
      }
    const std::vector<uint16_t> wantKeys = qsaKeys(x, x.first, rows);
    okKeys &= std::equal(wantKeys.begin(), wantKeys.end(), gotKeys.begin() + uint64_t{l} * rows * D);
    const auto pooledKey = [&](uint32_t b) { return &gotP[pooledAt(x.table[b / 8]) + (b % 8) * D]; };
    for (uint32_t b = 0; b < x.tokens / 4; ++b) {
      std::vector<double> mean(D, 0);
      for (uint32_t t = 4 * b; t < 4 * b + 4; ++t)
        for (uint32_t d = 0; d < D; ++d) mean[d] += f32(bf16(x.raw[uint64_t{t} * D + d])) / 4;
      const std::vector<double> want = rotate(rmsNorm(mean, knP), 4 * b);
      for (uint32_t d = 0; d < D; ++d) okP &= nearBf16(pooledKey(b)[d], want[d]);
    }
    for (uint32_t r = 0; r < rows; ++r) {
      const uint32_t row = l * rows + r, position = x.first + r, complete = (position + 1) / 4;
      const auto got = [&](uint32_t b) { return bool((gotMask[uint64_t{row} * words + b / 32] >> (b % 32)) & 1); };
      // The tail block and nothing past it up to two blocks past the tail
      // (a tile's last row's), in whole words; below the budget, every block.
      const uint32_t written = std::min(words, (complete + 2) / 32 + 1);
      for (uint32_t b = complete; b < written * 32; ++b) okMask &= got(b) == (b == complete);
      if (complete <= top) {
        for (uint32_t b = 0; b < complete; ++b) okMask &= got(b);
        continue;
      }
      std::vector<std::pair<double, uint32_t>> scored;
      for (uint32_t b = 0; b < complete; ++b) {
        double s = 0;
        for (uint32_t h = 0; h < H; ++h) {
          double dot = 0;
          for (uint32_t d = 0; d < D; ++d) dot += double(gotQ[(uint64_t{row} * H + h) * D + d]) * pooledKey(b)[d];
          s += std::max(dot, 0.0);
        }
        scored.push_back({s / std::sqrt(double(D)), b});
      }
      std::stable_sort(scored.begin(), scored.end(), [](auto a, auto b) { return a.first > b.first; });
      // The top blocks by the fp64 scores of the kernel's queries and keys:
      // its fp32 scores may order blocks within their rounding of the
      // threshold either way, but equal ones (equal keys) towards the earlier.
      const double threshold = scored[top - 1].first, near = 1e-4 * std::max(1.0, threshold);
      uint32_t taken = 0;
      for (uint32_t i = 0; i < complete; ++i) {
        const bool in = got(scored[i].second);
        taken += in;
        if (std::fabs(scored[i].first - threshold) > near) okMask &= in == (i < top);
        if (i && scored[i].first == scored[i - 1].first) okMask &= !in || got(scored[i - 1].second);
      }
      okMask &= taken == top;
    }
  }
  const std::string scope =
      std::to_string(lanes) + " lane(s) of " + std::to_string(c.tokens) + " tokens" + (c.yarn ? " with YaRN" : "");
  check(okQ, "QSA project of " + scope + ": normalized, rotated query heads");
  check(okKeys, "QSA project of " + scope + ": the rows' raw keys");
  check(okP, "QSA pool of " + scope + " in prefill steps of " + std::to_string(c.chunk) +
                 " rows: normalized mean keys rotated to the block start");
  check(okMask, "QSA select of " + scope + ": dense below the budget, top blocks and the tail block above it" +
                    (c.tied ? ", ties towards the earlier block" : ""));
}

// The history of a verify lane: `anchor` tokens of random raw keys, pooled
// by prefill steps into the index of a table of its own, `pages` shuffled
// pages of one buffer (an entry's page index its page), which hold its
// anchor's tokens and the step's after them; and its pending keys after
// them, in pending[0].
QsaLane qsaHistory(MetalBackend &backend, std::mt19937 &rng, uint32_t anchor, uint32_t pages, MetalBuffer &index,
                   const NormWeights &keyNorm, const MetalBuffer &invFreq) {
  QsaLane lane;
  lane.tokens = lane.first = anchor;
  lane.raw = randoms(rng, uint64_t{anchor} * kQsaDimension, 1.0f);
  lane.table.resize(pages);
  std::iota(lane.table.begin(), lane.table.end(), 0);
  std::shuffle(lane.table.begin(), lane.table.end(), rng);
  // Whole 16 KiB pages, so that the index is aligned for the page bits.
  const uint64_t bytes = (uint64_t{pages} * (kQsaPage / 4) * kQsaDimension * 2 + 16383) / 16384 * 16384;
  index = backend.allocateBuffer(bytes, BufferStorage::Shared, "qwen4-test-index", BufferReach::Addressed);
  std::memset(index.contents(), 0, index.sizeBytes());
  if (index.gpuAddress() & SPLASH_KV_PAGE_INDEX_MASK) throw std::runtime_error("a QSA test index leaves no page bits");
  std::vector<uint64_t> entries;
  for (uint32_t page : lane.table) entries.push_back(index.gpuAddress() | page);
  lane.tableBuffer = upload(backend, entries);
  backend.setTableReach(lane.tableBuffer, std::span(&index, 1));
  for (MetalBuffer &pending : lane.pending) pending = upload(backend, std::vector<uint16_t>(kQsaPendingBytes / 2, 0));
  CommandGraph graph;
  for (uint32_t from = 0; from < anchor; from += 2048)
    qsaPrefillStep(backend, graph, lane, from, std::min(2048u, anchor - from), keyNorm, invFreq, 1.0F, 0);
  if (!graph.empty()) run(backend, graph);
  return lane;
}

// A verify lane of the tree kernels: its rows' places in its block's tree
// from its anchor, and each row's indexer projections before the indexer's
// norm and RoPE, its query heads (heads x dimension a row) and raw key.
struct QsaTreeLane final {
  uint32_t anchor = 0;
  std::vector<ops::TreeRow> tree;
  std::vector<float> query, key;
};

// The tree whose row r follows parents[r] (the anchor's none, -1).
std::vector<ops::TreeRow> qsaTree(std::span<const int> parents) {
  std::vector<ops::TreeRow> tree;
  for (uint32_t row = 0; row < parents.size(); ++row) {
    const int parent = parents[row];
    tree.push_back(parent < 0 ? ops::TreeRow{1u, SPLASH_TREE_ROOT, 0}
                              : ops::TreeRow{tree[parent].ancestors | 1u << row, static_cast<uint16_t>(parent),
                                             static_cast<uint16_t>(tree[parent].depth + 1)});
  }
  return tree;
}

// The chain of `rows` rows.
std::vector<ops::TreeRow> qsaChain(uint32_t rows) {
  std::vector<ops::TreeRow> chain;
  for (uint32_t row = 0; row < rows; ++row) chain.push_back(splash::test::chainRow(row));
  return chain;
}

// The rows of a tree row's path from the anchor, its ancestors by depth.
std::vector<uint32_t> qsaPath(const ops::TreeRow &row) {
  std::vector<uint32_t> path;
  for (uint32_t rest = row.ancestors; rest; rest &= rest - 1) path.push_back(std::countr_zero(rest));
  return path;
}

// The lane whose rows are `rows` of `lane`'s, in that order, as a chain: a
// path of its tree run one row after another.
QsaTreeLane qsaChainOf(const QsaTreeLane &lane, std::span<const uint32_t> rows, uint32_t anchor) {
  constexpr uint32_t D = kQsaDimension, H = BLOCK_INDEXER_HEADS;
  QsaTreeLane chain{anchor, qsaChain(static_cast<uint32_t>(rows.size())), {}, {}};
  for (const uint32_t row : rows) {
    chain.query.insert(chain.query.end(), lane.query.begin() + uint64_t{row} * H * D,
                       lane.query.begin() + uint64_t{row + 1} * H * D);
    chain.key.insert(chain.key.end(), lane.key.begin() + uint64_t{row} * D, lane.key.begin() + uint64_t{row + 1} * D);
  }
  return chain;
}

// The tree kernels' outputs of a verify step: the lanes' rows of the step,
// and each row's queries and raw keys, block key, scores, bitmap and row mask;
// with a commit, each lane's pending keys after its accepted path (the index
// its history's).
struct QsaTreeResult final {
  std::vector<ops::LaneRows> lanes;
  uint32_t rows = 0;
  MetalBuffer queries, keys, blockKeys, scores, mask, rowMasks;
  std::vector<MetalBuffer> pendingAfter;
};

// The norms and RoPE of the QSA tree tests: the indexer's query and key norms,
// and the target's inverse frequencies without YaRN.
struct QsaWeights final {
  NormWeights queryNorm, keyNorm;
  MetalBuffer invFreq;
};

QsaWeights qsaWeights(MetalBackend &backend, std::mt19937 &rng) {
  const auto qn = randoms(rng, kQsaDimension, 0.2f), kn = randoms(rng, kQsaDimension, 0.2f);
  std::vector<float> qnP(kQsaDimension), knP(kQsaDimension), invFreq(32);
  for (uint32_t d = 0; d < kQsaDimension; ++d) qnP[d] = 1 + qn[d], knP[d] = 1 + kn[d];
  model::rotaryInverseFrequencies(10'000'000.0f, 32, model::RopeScaling{}, invFreq);
  return {{upload(backend, qnP), true}, {upload(backend, knP), true}, upload(backend, invFreq)};
}

// A row of the step's bitmap past the words of a lane's history, poison.
constexpr uint32_t kQsaPoison = 0xA5A5A5A5u;

// A verify step of the tree kernels over `lanes`, packed one after another
// from row 0, lane l over histories[l] (its index and pending keys), whose
// rows attend sparsely past `top` blocks, with bitmaps of `words` words and
// scores of 32 x words a row. With `paths`, each lane's accepted path, the
// commit after it.
QsaTreeResult qsaTreeStep(MetalBackend &backend, std::span<const QsaTreeLane> lanes,
                          std::span<const QsaLane *const> histories, uint32_t top, uint32_t words,
                          const QsaWeights &weights, std::span<const uint32_t> paths = {}) {
  constexpr uint32_t D = kQsaDimension, H = BLOCK_INDEXER_HEADS, kLaneRecords = 4;
  QsaTreeResult result;
  uint32_t most = 0;
  for (const QsaTreeLane &lane : lanes) {
    result.lanes.push_back({result.rows, static_cast<uint32_t>(lane.tree.size())});
    result.rows += static_cast<uint32_t>(lane.tree.size());
    most = std::max(most, static_cast<uint32_t>(lane.tree.size()));
  }
  const uint32_t rows = result.rows, stride = words * 32;
  std::vector<float> query, key;
  std::vector<ops::TreeRow> tree;
  std::vector<uint32_t> ropePositions, anchors, counts;
  std::vector<BlockIndexerBlock> records(lanes.size() * kLaneRecords);
  std::vector<MetalBuffer> tables, pending;
  for (uint32_t l = 0; l < lanes.size(); ++l) {
    const QsaTreeLane &lane = lanes[l];
    query.insert(query.end(), lane.query.begin(), lane.query.end());
    key.insert(key.end(), lane.key.begin(), lane.key.end());
    tree.insert(tree.end(), lane.tree.begin(), lane.tree.end());
    for (const ops::TreeRow &row : lane.tree) ropePositions.insert(ropePositions.end(), 3, lane.anchor + row.depth);
    // The blocks a verify block's most rows complete from the anchor on, as
    // Qwen4Steps writes them.
    const std::vector<BlockIndexerBlock> laneBlocks = qsaRecords(lane.anchor, SPLASH_TARGET_VERIFY_ROWS);
    std::ranges::copy(laneBlocks, records.begin() + uint64_t{l} * kLaneRecords);
    counts.push_back(static_cast<uint32_t>(laneBlocks.size()));
    anchors.push_back(lane.anchor);
    tables.push_back(histories[l]->tableBuffer);
    pending.push_back(histories[l]->pending[0]);
  }
  const MetalBuffer laneRows = upload(backend, result.lanes), treeRows = upload(backend, tree);
  const MetalBuffer cosT = scratch(backend, uint64_t{rows} * 32 * 4), sinT = scratch(backend, uint64_t{rows} * 32 * 4),
                    draftRope = scratch(backend, SPLASH_DRAFT_ROPE_PAIRS * 4);
  result.queries = scratch(backend, uint64_t{rows} * H * D * 2);
  result.keys = scratch(backend, uint64_t{rows} * D * 2);
  // A row that completes no block keeps no block key: poison, read by none.
  result.blockKeys = upload(backend, std::vector<uint16_t>(uint64_t{rows} * D, 0x7FC1));
  result.scores = upload(backend, std::vector<uint32_t>(uint64_t{rows} * stride, kQsaPoison));
  result.mask = upload(backend, std::vector<uint32_t>(uint64_t{rows} * words, kQsaPoison));
  result.rowMasks = upload(backend, std::vector<uint32_t>(rows, kQsaPoison));
  const uint32_t laneCap = most <= 8 ? 8u : 16u;
  const ops::IndexerTreeRows treeRowsOf{laneRows, treeRows, rows, laneCap, anchors, {tables, 0}, top, words, stride};
  CommandGraph graph;
  ops::RoPE::addTables(graph, upload(backend, ropePositions), draftRope, weights.invFreq, draftRope, cosT, sinT,
                       draftRope, draftRope, {rows, 0, 1.0F}, rows);
  ops::BlockIndexer::addProject(graph, upload(backend, query), upload(backend, key), weights.queryNorm, cosT, sinT,
                                result.queries, result.keys, laneRows,
                                {static_cast<uint32_t>(lanes.size()), rows, laneCap});
  ops::BlockIndexer::addTreePool(graph, {upload(backend, records), counts, kLaneRecords, result.keys, pending},
                                 result.blockKeys, weights.keyNorm, weights.invFreq, 1.0F, treeRowsOf);
  ops::BlockIndexer::addTreeSelect(graph, result.queries, result.blockKeys, result.scores, result.mask,
                                   result.rowMasks, treeRowsOf);
  if (!paths.empty()) {
    for (uint32_t l = 0; l < lanes.size(); ++l)
      result.pendingAfter.push_back(upload(backend, std::vector<uint16_t>(kQsaPendingBytes / 2, 0)));
    ops::BlockIndexer::addCommit(graph, {result.keys, result.blockKeys, 1, rows,
                                         upload(backend, std::vector<uint32_t>(paths.begin(), paths.end())), laneRows,
                                         anchors, pending, result.pendingAfter, tables, 0});
  }
  run(backend, graph);
  return result;
}

// Words `first` on of a row of a step's buffer of `width` words a row.
std::vector<uint32_t> qsaWords(const MetalBuffer &buffer, uint32_t width, uint32_t row, uint32_t first,
                               uint32_t count) {
  const auto all = download<uint32_t>(buffer, uint64_t{row + 1} * width);
  return {all.begin() + uint64_t{row} * width + first, all.begin() + uint64_t{row} * width + first + count};
}

bool qsaBit(const std::vector<uint32_t> &bits, uint32_t block) { return (bits[block / 32] >> (block % 32)) & 1u; }

// Whether a verify row's bitmap holds the blocks of an fp64 reference's top
// blocks of its `complete` complete blocks: its scores from its queries and
// each block's pooled key (`key(b)`), the top `top` within the fp32
// rounding of the threshold either way.
template <class Key>
bool qsaSelectsTop(const std::vector<float> &queries, const std::vector<uint32_t> &bits, uint32_t complete,
                   uint32_t top, Key key) {
  constexpr uint32_t D = kQsaDimension, H = BLOCK_INDEXER_HEADS;
  if (complete <= top) {
    for (uint32_t b = 0; b < complete; ++b)
      if (!qsaBit(bits, b)) return false;
    return true;
  }
  std::vector<std::pair<double, uint32_t>> scored;
  for (uint32_t b = 0; b < complete; ++b) {
    const float *k = key(b);
    double s = 0;
    for (uint32_t h = 0; h < H; ++h) {
      double dot = 0;
      for (uint32_t d = 0; d < D; ++d) dot += double(queries[h * D + d]) * k[d];
      s += std::max(dot, 0.0);
    }
    scored.push_back({s / std::sqrt(double(D)), b});
  }
  std::stable_sort(scored.begin(), scored.end(), [](auto a, auto b) { return a.first > b.first; });
  const double threshold = scored[top - 1].first, near = 1e-4 * std::max(1.0, threshold);
  uint32_t taken = 0;
  bool ok = true;
  for (uint32_t i = 0; i < complete; ++i) {
    const bool in = qsaBit(bits, scored[i].second);
    taken += in;
    if (std::fabs(scored[i].first - threshold) > near) ok &= in == (i < top);
  }
  return ok && taken == top;
}

// A verify block of 16 rows as a tree from an anchor past `anchor` tokens,
// against each of its paths run as a chain: rows 0 to 9 a chain, row 10 a
// second child of the anchor (whose packed predecessor, row 9, is not its
// parent) with a child of its own, rows 12 to 14 a branch off row 3, and row
// 15 a sibling of row 9. At an anchor of 2 past a block (completing depths 1,
// 5, 9 and 13) the anchor's block completes on three paths (rows 1 and 10),
// the next one on two (rows 5 and 13), and the one after it on two (rows 9
// and 15), each path's of its own tokens. Every row's block key, its scores
// of every complete block, its blocks below the anchor's, its row mask by
// depth and its bits from the anchor's block on are its path's chain's, bit
// for bit; each chain's selections are an fp64 reference's top blocks; and a
// tree row's bitmap from the anchor's block on names the blocks of its row
// mask's slots. Then the commit of one path writes the index and pending
// keys that path's chain's commit does. At an anchor 1 past a block the
// siblings that complete blocks are rows 2 and 11, and 6 and 14.
void qsaTreeAsChains(MetalBackend &backend, uint32_t anchor, uint32_t top) {
  constexpr uint32_t D = kQsaDimension, H = BLOCK_INDEXER_HEADS, rows = 16;
  std::mt19937 rng(41 + anchor);
  const QsaWeights weights = qsaWeights(backend, rng);
  MetalBuffer index;
  const uint32_t pages = (anchor + rows + kQsaPage - 1) / kQsaPage;
  QsaLane history = qsaHistory(backend, rng, anchor, pages, index, weights.keyNorm, weights.invFreq);
  const std::vector<uint8_t> before = download<uint8_t>(index, index.sizeBytes());
  constexpr std::array<int, rows> kParents{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 10, 3, 12, 13, 8};
  const QsaTreeLane lane{anchor, qsaTree(kParents), randoms(rng, uint64_t{rows} * H * D, 1.0f),
                         randoms(rng, uint64_t{rows} * D, 1.0f)};
  const uint32_t words = ((anchor + rows) / 4 + BLOCK_INDEXER_TILE_SLACK + 31) / 32 + 1, stride = words * 32;
  const std::array<const QsaLane *, 1> histories{&history};
  // The path committed: row 14's, which completes two blocks of its own.
  const std::vector<uint32_t> committed = qsaPath(lane.tree[14]);
  const std::array<uint32_t, 1> treePath{lane.tree[14].ancestors};
  const QsaTreeResult tree = qsaTreeStep(backend, std::span(&lane, 1), histories, top, words, weights, treePath);
  const std::vector<uint8_t> treeIndex = download<uint8_t>(index, index.sizeBytes());
  std::memcpy(index.contents(), before.data(), before.size());
  // Each leaf's path as a chain; a tree row's chain is the first leaf's that
  // holds it.
  std::vector<int> leafOf(rows, -1);
  std::vector<QsaTreeResult> chains;
  std::vector<std::vector<uint32_t>> leafPaths;
  for (uint32_t leaf = 0; leaf < rows; ++leaf) {
    if (std::ranges::find(kParents, int(leaf)) != kParents.end()) continue;
    leafPaths.push_back(qsaPath(lane.tree[leaf]));
    const QsaTreeLane chain = qsaChainOf(lane, leafPaths.back(), anchor);
    const bool commits = leafPaths.back() == std::vector<uint32_t>(committed.begin(), committed.end());
    const std::array<uint32_t, 1> chainPath{splash::test::chainPath(static_cast<uint32_t>(committed.size()))};
    chains.push_back(qsaTreeStep(backend, std::span(&chain, 1), histories, top, words, weights,
                                 commits ? std::span<const uint32_t>(chainPath) : std::span<const uint32_t>{}));
    if (commits) {
      check(download<uint8_t>(index, index.sizeBytes()) == treeIndex && treeIndex != before &&
                download<uint16_t>(chains.back().pendingAfter[0], kQsaPendingBytes / 2) ==
                    download<uint16_t>(tree.pendingAfter[0], kQsaPendingBytes / 2),
            "QSA tree at " + std::to_string(anchor) + ": the commit of a path writes its chain's index and pending keys");
      std::memcpy(index.contents(), before.data(), before.size());
    }
    for (const uint32_t row : leafPaths.back())
      if (leafOf[row] < 0) leafOf[row] = static_cast<int>(chains.size() - 1);
  }
  const uint32_t history0 = anchor / 4;
  const auto treeBlockKeys = download<uint16_t>(tree.blockKeys, uint64_t{rows} * D);
  const auto treeMasks = download<uint32_t>(tree.rowMasks, rows);
  std::vector<float> pooled;
  for (const uint16_t value : download<uint16_t>(index, index.sizeBytes() / 2)) pooled.push_back(f32(value));
  bool okKeys = true, okScores = true, okBits = true, okMasks = true, okReference = true;
  uint32_t sparseRows = 0, attendedOwn = 0, droppedOwn = 0, differingSiblings = 0;
  for (uint32_t row = 0; row < rows; ++row) {
    const ops::TreeRow place = lane.tree[row];
    const uint32_t depth = place.depth, complete = (anchor + depth + 1) / 4;
    const QsaTreeResult &chain = chains[leafOf[row]];
    const std::vector<uint32_t> path = qsaPath(place);
    // The block key of the row, and its scores of its complete blocks.
    const auto chainKeys = download<uint16_t>(chain.blockKeys, uint64_t{chain.rows} * D);
    if ((anchor + depth) % 4 == 3)
      okKeys &= std::equal(treeBlockKeys.begin() + uint64_t{row} * D, treeBlockKeys.begin() + uint64_t{row + 1} * D,
                           chainKeys.begin() + uint64_t{depth} * D);
    if (complete > top) {
      ++sparseRows;
      okScores &= qsaWords(tree.scores, stride, row, 0, complete) == qsaWords(chain.scores, stride, depth, 0, complete);
    }
    // Its bitmap: below the anchor's block its chain's; from there on the
    // blocks of its row mask's slots, which by depth are its chain's.
    const std::vector<uint32_t> treeBits = qsaWords(tree.mask, words, row, 0, words);
    const std::vector<uint32_t> chainBits = qsaWords(chain.mask, words, depth, 0, words);
    const uint32_t chainMask = download<uint32_t>(chain.rowMasks, chain.rows)[depth];
    for (uint32_t b = 0; b < history0; ++b) okBits &= qsaBit(treeBits, b) == qsaBit(chainBits, b);
    okMasks &= !(treeMasks[row] & ~place.ancestors) && !(chainMask >> (depth + 1));
    for (uint32_t k = 0; k <= depth; ++k) okMasks &= ((treeMasks[row] >> path[k]) & 1u) == ((chainMask >> k) & 1u);
    const uint32_t treeWritten = ((anchor + row + 1) / 4 + BLOCK_INDEXER_TILE_SLACK) / 32 + 1;
    for (uint32_t b = history0; b < treeWritten * 32; ++b) {
      bool named = false;
      for (uint32_t rest = treeMasks[row]; rest; rest &= rest - 1) named |= (anchor + std::countr_zero(rest)) / 4 == b;
      okBits &= qsaBit(treeBits, b) == named;
    }
    for (uint32_t b = history0; b < complete; ++b) {
      const bool attended = (chainMask >> (4 * b + 3 - anchor)) & 1u;
      attendedOwn += attended;
      droppedOwn += !attended;
    }
    // The chain's selection against the fp64 top blocks: below the anchor's
    // block the index's pooled keys, from it on its rows' block keys.
    std::vector<float> rowQueries;
    for (const uint32_t value : qsaWords(chain.queries, H * D / 2, depth, 0, H * D / 2)) {
      rowQueries.push_back(f32(uint16_t(value)));
      rowQueries.push_back(f32(uint16_t(value >> 16)));
    }
    std::vector<float> stepKeys;
    for (const uint16_t value : chainKeys) stepKeys.push_back(f32(value));
    okReference &= qsaSelectsTop(rowQueries, chainBits, complete, top, [&](uint32_t b) -> const float * {
      if (b >= history0) return &stepKeys[uint64_t{4 * b + 3 - anchor} * D];
      const uint32_t page = history.table[b / 8];
      return &pooled[(uint64_t{page} * 8 + b % 8) * D];
    });
  }
  // Rows of one depth that complete a block hold their own paths' tokens.
  uint32_t siblings = 0;
  for (uint32_t a = 0; a < rows; ++a)
    for (uint32_t b = a + 1; b < rows; ++b)
      if (lane.tree[a].depth == lane.tree[b].depth && (anchor + lane.tree[a].depth) % 4 == 3) {
        ++siblings;
        differingSiblings += !std::equal(treeBlockKeys.begin() + uint64_t{a} * D,
                                         treeBlockKeys.begin() + uint64_t{a + 1} * D, treeBlockKeys.begin() + uint64_t{b} * D);
      }
  const std::string scope = "QSA tree at " + std::to_string(anchor) + " past " + std::to_string(top) + " blocks";
  check(okKeys && siblings >= 2 && differingSiblings == siblings,
        scope + ": each block key its path's chain's, " + std::to_string(siblings) + " pairs of siblings' apart");
  check(okScores && sparseRows > 0, scope + ": each sparse row's scores its path's chain's (" +
                                        std::to_string(sparseRows) + " rows)");
  check(okBits && okMasks, scope + ": each row's selection its path's chain's, its bits its row mask's slots");
  check(okReference && attendedOwn > 0 && (top >= 512 || droppedOwn > 0),
        scope + ": each chain's top blocks the fp64 reference's, its own blocks among them (" +
            std::to_string(attendedOwn) + " attended, " + std::to_string(droppedOwn) + " not)");
}

// A chain of `rows` rows from an anchor past `anchor` tokens that accepts
// `accepted` of them, against the same tokens decoded one a step, each step's
// row committed: each row's block key, scores of its complete blocks and
// bitmap are the step's that decodes it, bit for bit, and the index and
// pending keys after the chain's commit are those after the last step's.
void qsaChainAsSequential(MetalBackend &backend, uint32_t anchor, uint32_t rows, uint32_t accepted, uint32_t top) {
  constexpr uint32_t D = kQsaDimension, H = BLOCK_INDEXER_HEADS;
  std::mt19937 rng(53 + anchor);
  const QsaWeights weights = qsaWeights(backend, rng);
  MetalBuffer index;
  const QsaLane history =
      qsaHistory(backend, rng, anchor, (anchor + rows + kQsaPage - 1) / kQsaPage, index, weights.keyNorm,
                 weights.invFreq);
  const std::vector<uint8_t> before = download<uint8_t>(index, index.sizeBytes());
  const QsaTreeLane chain{anchor, qsaChain(rows), randoms(rng, uint64_t{rows} * H * D, 1.0f),
                          randoms(rng, uint64_t{rows} * D, 1.0f)};
  const uint32_t words = ((anchor + rows) / 4 + BLOCK_INDEXER_TILE_SLACK + 31) / 32 + 1, stride = words * 32;
  const std::array<const QsaLane *, 1> histories{&history};
  const std::array<uint32_t, 1> path{splash::test::chainPath(accepted)};
  const QsaTreeResult whole = qsaTreeStep(backend, std::span(&chain, 1), histories, top, words, weights, path);
  const std::vector<uint8_t> chainIndex = download<uint8_t>(index, index.sizeBytes());
  std::memcpy(index.contents(), before.data(), before.size());
  QsaLane decoded = history;
  bool okKeys = true, okScores = true, okBits = true;
  for (uint32_t step = 0; step < accepted; ++step) {
    const std::array<uint32_t, 1> row{step};
    const QsaTreeLane one = qsaChainOf(chain, row, anchor + step);
    const std::array<const QsaLane *, 1> current{&decoded};
    const std::array<uint32_t, 1> anchorOnly{1};
    const QsaTreeResult single = qsaTreeStep(backend, std::span(&one, 1), current, top, words, weights, anchorOnly);
    const uint32_t complete = (anchor + step + 1) / 4;
    if ((anchor + step) % 4 == 3)
      okKeys &= qsaWords(single.blockKeys, D / 2, 0, 0, D / 2) == qsaWords(whole.blockKeys, D / 2, step, 0, D / 2);
    if (complete > top)
      okScores &= qsaWords(single.scores, stride, 0, 0, complete) == qsaWords(whole.scores, stride, step, 0, complete);
    okBits &= qsaWords(single.mask, words, 0, 0, words) == qsaWords(whole.mask, words, step, 0, words);
    decoded.pending[0] = single.pendingAfter[0];
  }
  const std::string scope = "QSA chain of " + std::to_string(rows) + " rows at " + std::to_string(anchor) + " past " +
                            std::to_string(top) + " blocks, " + std::to_string(accepted) + " accepted";
  check(okKeys && okScores && okBits, scope + ": each row's block key, scores and bitmap a step's of one row");
  check(download<uint8_t>(index, index.sizeBytes()) == chainIndex && chainIndex != before &&
            download<uint16_t>(decoded.pending[0], kQsaPendingBytes / 2) ==
                download<uint16_t>(whole.pendingAfter[0], kQsaPendingBytes / 2),
        scope + ": the commit's index and pending keys the steps' of one row");
}

// The commit of a verify step's two attention layers in one dispatch, each
// layer its own region of every page's index plane (indexLayerBytes apart)
// and its own history, raw keys, block keys and pending keys: after the
// accepted path of qsaTreeAsChains' tree to row 14, which completes two
// blocks of its own at an anchor 2 past a block, each layer's index region
// and pending keys are its path's chain's committed alone into that layer.
void qsaCommitLayers(MetalBackend &backend) {
  constexpr uint32_t D = kQsaDimension, H = BLOCK_INDEXER_HEADS, rows = 16, layers = 2, anchor = 2046, top = 512;
  std::mt19937 rng(61);
  const QsaWeights weights = qsaWeights(backend, rng);
  const uint32_t pages = (anchor + rows + kQsaPage - 1) / kQsaPage;
  // A layer's region of the index: whole 16 KiB pages, so that a table of
  // the region keeps its entries' page bits.
  const uint64_t layerBytes = (uint64_t{pages} * (kQsaPage / 4) * D * 2 + 16383) / 16384 * 16384;
  MetalBuffer index = backend.allocateBuffer(layers * layerBytes, BufferStorage::Shared, "qwen4-test-index",
                                             BufferReach::Addressed);
  std::memset(index.contents(), 0, index.sizeBytes());
  if (index.gpuAddress() & SPLASH_KV_PAGE_INDEX_MASK) throw std::runtime_error("a QSA test index leaves no page bits");
  std::vector<uint32_t> order(pages);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);
  // Each layer's history: its own raw keys of the anchor's tokens, pooled
  // through a table of the layer's region, and its pending keys after them.
  std::array<QsaLane, layers> history;
  for (uint32_t layer = 0; layer < layers; ++layer) {
    QsaLane &lane = history[layer];
    lane.tokens = lane.first = anchor;
    lane.raw = randoms(rng, uint64_t{anchor} * D, 1.0f);
    lane.table = order;
    std::vector<uint64_t> entries;
    for (const uint32_t page : order) entries.push_back((index.gpuAddress() + layer * layerBytes) | page);
    lane.tableBuffer = upload(backend, entries);
    backend.setTableReach(lane.tableBuffer, std::span(&index, 1));
    for (MetalBuffer &pending : lane.pending) pending = upload(backend, std::vector<uint16_t>(kQsaPendingBytes / 2, 0));
    CommandGraph graph;
    for (uint32_t from = 0; from < anchor; from += 2048)
      qsaPrefillStep(backend, graph, lane, from, std::min(2048u, anchor - from), weights.keyNorm, weights.invFreq,
                     1.0F, 0);
    run(backend, graph);
  }
  const std::vector<uint8_t> before = download<uint8_t>(index, index.sizeBytes());
  constexpr std::array<int, rows> kParents{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 10, 3, 12, 13, 8};
  const std::vector<ops::TreeRow> tree = qsaTree(kParents);
  const uint32_t leaf = 14, words = ((anchor + rows) / 4 + BLOCK_INDEXER_TILE_SLACK + 31) / 32 + 1;
  const std::vector<uint32_t> committed = qsaPath(tree[leaf]);
  // Each layer's verify step of the tree, its rows' raw keys and block keys
  // and its pending keys before it in its layer's place, then one commit of
  // the path in both layers.
  std::array<QsaTreeLane, layers> lanes;
  std::vector<uint16_t> keys, blockKeys, pendingIn;
  for (uint32_t layer = 0; layer < layers; ++layer) {
    lanes[layer] = {anchor, tree, randoms(rng, uint64_t{rows} * H * D, 1.0f), randoms(rng, uint64_t{rows} * D, 1.0f)};
    const std::array<const QsaLane *, 1> histories{&history[layer]};
    const QsaTreeResult step = qsaTreeStep(backend, std::span(&lanes[layer], 1), histories, top, words, weights);
    const std::vector<uint16_t> stepKeys = download<uint16_t>(step.keys, uint64_t{rows} * D),
                                stepBlockKeys = download<uint16_t>(step.blockKeys, uint64_t{rows} * D),
                                pending = download<uint16_t>(history[layer].pending[0], kQsaPendingBytes / 2);
    keys.insert(keys.end(), stepKeys.begin(), stepKeys.end());
    blockKeys.insert(blockKeys.end(), stepBlockKeys.begin(), stepBlockKeys.end());
    pendingIn.insert(pendingIn.end(), pending.begin(), pending.end());
  }
  const MetalBuffer pendingOut = upload(backend, std::vector<uint16_t>(layers * kQsaPendingBytes / 2, 0));
  const std::array<ops::LaneRows, 1> laneRows{{{0, rows}}};
  const std::array<uint32_t, 1> anchors{anchor};
  const std::array<MetalBuffer, 1> in{upload(backend, pendingIn)}, out{pendingOut}, tables{history[0].tableBuffer};
  CommandGraph graph;
  ops::BlockIndexer::addCommit(graph, {upload(backend, keys), upload(backend, blockKeys), layers, rows,
                                       upload(backend, std::vector<uint32_t>{tree[leaf].ancestors}),
                                       laneRowsOf(backend, laneRows), anchors, in, out, tables,
                                       static_cast<uint32_t>(layerBytes)});
  run(backend, graph);
  const std::vector<uint8_t> treeIndex = download<uint8_t>(index, index.sizeBytes());
  const std::vector<uint16_t> treePending = download<uint16_t>(pendingOut, layers * kQsaPendingBytes / 2);
  // Each layer's chain of the path, committed alone through its region's
  // table, from the history before the step.
  std::memcpy(index.contents(), before.data(), before.size());
  bool pendingSame = true;
  for (uint32_t layer = 0; layer < layers; ++layer) {
    const QsaTreeLane chain = qsaChainOf(lanes[layer], committed, anchor);
    const std::array<const QsaLane *, 1> histories{&history[layer]};
    const std::array<uint32_t, 1> chainPath{splash::test::chainPath(static_cast<uint32_t>(committed.size()))};
    const QsaTreeResult step = qsaTreeStep(backend, std::span(&chain, 1), histories, top, words, weights, chainPath);
    const std::vector<uint16_t> pending = download<uint16_t>(step.pendingAfter[0], kQsaPendingBytes / 2);
    pendingSame &= std::equal(pending.begin(), pending.end(),
                              treePending.begin() + uint64_t{layer} * kQsaPendingBytes / 2);
  }
  const std::vector<uint8_t> chainIndex = download<uint8_t>(index, index.sizeBytes());
  const auto region = [&](const std::vector<uint8_t> &bytes, uint32_t layer) {
    return std::vector<uint8_t>(bytes.begin() + layer * layerBytes, bytes.begin() + (layer + 1) * layerBytes);
  };
  check(treeIndex == chainIndex && region(treeIndex, 0) != region(before, 0) &&
            region(treeIndex, 1) != region(before, 1),
        "QSA commit of a tree's path in two layers: each layer's index region its path's chain's");
  check(pendingSame, "QSA commit of a tree's path in two layers: each layer's pending keys its path's chain's");
}

// Lanes whose verify blocks hold different row counts, packed one after
// another (5, 8, 1 and 6 rows from rows 0, 5, 13 and 14) in one dispatch of
// each tree kernel, around the sparse budget (lane l's anchor at 2045 + 61
// l), the second a tree with a branch off its anchor: each lane projects,
// pools, selects and commits a path as it does alone in a dispatch of its own
// from row 0, the same queries, keys, block keys, scores, bitmaps, row masks,
// pending keys and index, bit for bit.
void qsaPackedLanes(MetalBackend &backend) {
  constexpr uint32_t D = kQsaDimension, H = BLOCK_INDEXER_HEADS, top = 512, lanes = 4;
  std::mt19937 rng(31);
  const QsaWeights weights = qsaWeights(backend, rng);
  constexpr std::array<int, 8> kBranch{-1, 0, 1, 0, 3, 4, 2, 6};
  std::vector<QsaTreeLane> lane;
  std::vector<QsaLane> history;
  std::vector<MetalBuffer> indexes(lanes);
  std::vector<std::vector<uint8_t>> before;
  const std::array<uint32_t, lanes> sizes{5, 8, 1, 6};
  for (uint32_t l = 0; l < lanes; ++l) {
    const uint32_t anchor = 2045 + 61 * l;
    history.push_back(qsaHistory(backend, rng, anchor, (anchor + 16 + kQsaPage - 1) / kQsaPage, indexes[l],
                                 weights.keyNorm, weights.invFreq));
    before.push_back(download<uint8_t>(indexes[l], indexes[l].sizeBytes()));
    lane.push_back({anchor, l == 1 ? qsaTree(kBranch) : qsaChain(sizes[l]),
                    randoms(rng, uint64_t{sizes[l]} * H * D, 1.0f), randoms(rng, uint64_t{sizes[l]} * D, 1.0f)});
  }
  const uint32_t words = ((2045 + 61 * 3 + 16) / 4 + BLOCK_INDEXER_TILE_SLACK + 31) / 32 + 1, stride = words * 32;
  // Paths that complete a block: 3 rows of the first lane, the branch's
  // deepest row, the anchor alone and 4 rows of the last lane.
  const std::array<uint32_t, lanes> paths{splash::test::chainPath(3), lane[1].tree[5].ancestors, 1, splash::test::chainPath(4)};
  std::vector<const QsaLane *> all;
  for (const QsaLane &x : history) all.push_back(&x);
  const QsaTreeResult together = qsaTreeStep(backend, lane, all, top, words, weights, paths);
  std::vector<std::vector<uint8_t>> pooled;
  for (uint32_t l = 0; l < lanes; ++l) {
    pooled.push_back(download<uint8_t>(indexes[l], indexes[l].sizeBytes()));
    std::memcpy(indexes[l].contents(), before[l].data(), before[l].size());
  }
  bool ok = true;
  for (uint32_t l = 0; l < lanes; ++l) {
    const ops::LaneRows block = together.lanes[l];
    const QsaTreeResult alone =
        qsaTreeStep(backend, std::span(lane).subspan(l, 1), std::span(all).subspan(l, 1), top, words, weights,
                    std::span(paths).subspan(l, 1));
    const auto same = [&](const MetalBuffer &packedValues, const MetalBuffer &aloneValues, uint64_t width) {
      const auto a = download<uint32_t>(packedValues, packedValues.sizeBytes() / 4);
      const auto b = download<uint32_t>(aloneValues, aloneValues.sizeBytes() / 4);
      return std::equal(b.begin(), b.begin() + block.rows * width, a.begin() + block.first * width);
    };
    ok &= same(together.queries, alone.queries, H * D / 2) && same(together.keys, alone.keys, D / 2) &&
          same(together.blockKeys, alone.blockKeys, D / 2) && same(together.scores, alone.scores, stride) &&
          same(together.mask, alone.mask, words) && same(together.rowMasks, alone.rowMasks, 1);
    ok &= download<uint16_t>(together.pendingAfter[l], kQsaPendingBytes / 2) ==
          download<uint16_t>(alone.pendingAfter[0], kQsaPendingBytes / 2);
    ok &= download<uint8_t>(indexes[l], indexes[l].sizeBytes()) == pooled[l] && pooled[l] != before[l];
  }
  check(ok, "QSA tree kernels of packed lanes of 5, 8, 1 and 6 rows: each lane's as alone");
}

// The indexer's projection of a decode step's lanes of 1 to 16 rows, packed
// one after another at a cycle's caps of 8 and 16 rows a lane, reads each
// lane's rows from the layout on the device: each lane's queries and keys are
// its rows' projected alone from row 0, bit for bit, and the step's spare
// rows, past its rows to its lanes' caps, keep their fills.
void qsaProjectPackedLanes(MetalBackend &backend) {
  constexpr uint32_t D = kQsaDimension, H = BLOCK_INDEXER_HEADS, P = SPLASH_TARGET_ROPE_PAIRS;
  constexpr uint16_t kFill = 0x7FC1;
  std::mt19937 rng(43);
  const QsaWeights weights = qsaWeights(backend, rng);
  struct Projected final {
    std::vector<uint16_t> queries, keys;
  };
  // The rows `rows` of a step's rows of `width` values.
  const auto rowsOfStep = [](const std::vector<float> &values, ops::LaneRows rows, uint32_t width) {
    return std::vector<float>(values.begin() + uint64_t{rows.first} * width,
                              values.begin() + (uint64_t{rows.first} + rows.rows) * width);
  };
  // The projection of `lanes` at `cap` rows a lane, into queries and keys of
  // `slots` rows filled beforehand.
  const auto project = [&](std::span<const ops::LaneRows> lanes, uint32_t cap, uint32_t slots,
                           const std::vector<float> &query, const std::vector<float> &key,
                           const std::vector<float> &cos, const std::vector<float> &sin) {
    MetalBuffer queries = upload(backend, std::vector<uint16_t>(uint64_t{slots} * H * D, kFill)),
                keys = upload(backend, std::vector<uint16_t>(uint64_t{slots} * D, kFill));
    CommandGraph graph;
    ops::BlockIndexer::addProject(
        graph, upload(backend, query), upload(backend, key), weights.queryNorm, upload(backend, cos),
        upload(backend, sin), queries, keys, laneRowsOf(backend, lanes),
        {static_cast<uint32_t>(lanes.size()), static_cast<uint32_t>(splash::test::rowsEnd(lanes)), cap});
    run(backend, graph);
    return Projected{download<uint16_t>(queries, uint64_t{slots} * H * D), download<uint16_t>(keys, uint64_t{slots} * D)};
  };
  const auto filled = [&](const std::vector<uint16_t> &values, uint64_t from) {
    return std::all_of(values.begin() + from, values.end(), [&](uint16_t value) { return value == kFill; });
  };
  struct Case final {
    uint32_t cap;
    std::vector<uint32_t> sizes;
  };
  const std::array<Case, 3> cases{{{8, {3, 8, 1, 6}}, {16, {16, 1, 9, 12}}, {16, {7, 16}}}};
  for (const Case &c : cases) {
    std::vector<ops::LaneRows> lanes;
    uint32_t rows = 0;
    for (const uint32_t size : c.sizes) {
      lanes.push_back({rows, size});
      rows += size;
    }
    const uint32_t slots = static_cast<uint32_t>(lanes.size()) * c.cap;
    const auto query = randoms(rng, uint64_t{slots} * H * D, 1.0f), key = randoms(rng, uint64_t{slots} * D, 1.0f),
               cos = randoms(rng, uint64_t{slots} * P, 1.0f), sin = randoms(rng, uint64_t{slots} * P, 1.0f);
    const Projected packed = project(lanes, c.cap, slots, query, key, cos, sin);
    bool same = true;
    for (const ops::LaneRows &lane : lanes) {
      const std::array<ops::LaneRows, 1> own{{{0, lane.rows}}};
      const Projected alone = project(own, c.cap, c.cap, rowsOfStep(query, lane, H * D), rowsOfStep(key, lane, D),
                                      rowsOfStep(cos, lane, P), rowsOfStep(sin, lane, P));
      same &= std::equal(alone.queries.begin(), alone.queries.begin() + uint64_t{lane.rows} * H * D,
                         packed.queries.begin() + uint64_t{lane.first} * H * D) &&
              std::equal(alone.keys.begin(), alone.keys.begin() + uint64_t{lane.rows} * D,
                         packed.keys.begin() + uint64_t{lane.first} * D);
    }
    std::string sizes;
    for (const uint32_t size : c.sizes) sizes += (sizes.empty() ? "" : ", ") + std::to_string(size);
    const std::string scope = "QSA projection of lanes of " + sizes + " rows at a cap of " + std::to_string(c.cap);
    check(same, scope + ": each lane's queries and keys as alone, bit for bit");
    check(filled(packed.queries, uint64_t{rows} * H * D) && filled(packed.keys, uint64_t{rows} * D),
          scope + ": the step's spare rows keep their fills");
  }
}

// The PLE convolution of a command's sequences in one dispatch, each
// sequence's rows its entry of the layout on the device: a decode step's
// lanes of 1 to 16 rows packed at a cycle's caps of 8 and 16, trees among
// them whose rows' packed predecessors are not their parents, and a prompt
// chunk's sequences, which write their histories after their rows. Each
// sequence's streams and history are its own convolved alone from row 0 after
// its own history, bit for bit, and the command's spare rows, past its rows
// to its sequences' bound, keep their streams.
void plePackedLanes(MetalBackend &backend) {
  constexpr uint32_t hidden = 256, streams = 4, taps = 4, dilation = 3, width = hidden * streams;
  constexpr uint16_t kFill = 0x7FC1;
  const ops::NgramShape shape{hidden, streams, taps, dilation};
  const uint32_t history = shape.history();
  std::mt19937 rng(47);
  const MetalBuffer tapsBuffer = upload(backend, randoms(rng, uint64_t{width} * taps, 0.4f));
  struct Convolved final {
    std::vector<uint32_t> streams;
    std::vector<std::vector<uint16_t>> histories;
  };
  // The convolution of `lanes` at `bound` rows a sequence over `slots` rows
  // of `conv`, `gated` and `x`, sequence s after histories[s]; with `tree` (a
  // row a slot) along its rows' paths, else writing each sequence's history
  // after its rows: the streams' bits and the histories written.
  const auto convolve = [&](std::span<const ops::LaneRows> lanes, uint32_t bound, uint32_t slots,
                            const std::vector<float> &conv, const std::vector<float> &gated,
                            const std::vector<float> &x, std::span<const std::vector<float>> histories,
                            std::span<const ops::TreeRow> tree) {
    const bool inTree = !tree.empty();
    const MetalBuffer streamsBuffer = upload(backend, x);
    std::vector<MetalBuffer> in, out;
    for (const std::vector<float> &values : histories) {
      in.push_back(upload(backend, bf16s(values)));
      if (!inTree) out.push_back(upload(backend, std::vector<uint16_t>(uint64_t{history} * width, kFill)));
    }
    CommandGraph graph;
    ops::NgramEmbedding::addConvolution(
        graph, upload(backend, bf16s(conv)), in, out, tapsBuffer, upload(backend, gated), streamsBuffer, shape,
        laneRowsOf(backend, lanes),
        {static_cast<uint32_t>(lanes.size()), static_cast<uint32_t>(splash::test::rowsEnd(lanes)), bound},
        inTree ? upload(backend, std::vector<ops::TreeRow>(tree.begin(), tree.end())) : MetalBuffer{});
    run(backend, graph);
    Convolved result{download<uint32_t>(streamsBuffer, uint64_t{slots} * width), {}};
    for (const MetalBuffer &buffer : out) result.histories.push_back(download<uint16_t>(buffer, uint64_t{history} * width));
    return result;
  };
  // A verify step's lanes by their rows' parents (-1 the anchor's), or a
  // prompt chunk's sequences, chains of their rows.
  const auto chain = [](uint32_t rows) {
    std::vector<int> parents(rows);
    for (uint32_t row = 0; row < rows; ++row) parents[row] = int(row) - 1;
    return parents;
  };
  struct Case final {
    uint32_t bound;
    bool tree;
    std::vector<std::vector<int>> sequences;
  };
  const std::vector<Case> cases{
      {8, true, {chain(3), {-1, 0, 1, 2, 3, 1, 5, 0}, chain(1), {-1, 0, 0, 1, 2, 3}}},
      {16, true, {{-1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 0, 11, 12, 13, 14}, chain(1), {-1, 0, 1, 1, 3, 2, 5, 0, 7},
                  chain(12)}},
      {16, true, {{-1, 0, 0, 1, 2, 3, 4}, chain(16)}},
      {40, false, {chain(5), chain(40), chain(1), chain(17)}}};
  for (const Case &c : cases) {
    std::vector<ops::LaneRows> lanes;
    std::vector<ops::TreeRow> tree;
    uint32_t rows = 0;
    for (const std::vector<int> &parents : c.sequences) {
      lanes.push_back({rows, static_cast<uint32_t>(parents.size())});
      rows += static_cast<uint32_t>(parents.size());
      if (!c.tree) continue;
      const std::vector<ops::TreeRow> own = qsaTree(parents);
      tree.insert(tree.end(), own.begin(), own.end());
    }
    const uint32_t slots = static_cast<uint32_t>(lanes.size()) * c.bound;
    const auto conv = randoms(rng, uint64_t{slots} * width, 1.0f), gated = randoms(rng, uint64_t{slots} * width, 1.0f),
               x = randoms(rng, uint64_t{slots} * width, 1.0f);
    std::vector<std::vector<float>> histories;
    for (size_t s = 0; s < lanes.size(); ++s) histories.push_back(randoms(rng, uint64_t{history} * width, 1.0f));
    const Convolved packed = convolve(lanes, c.bound, slots, conv, gated, x, histories, tree);
    bool same = true;
    for (size_t s = 0; s < lanes.size(); ++s) {
      const ops::LaneRows lane = lanes[s];
      const auto rowsOfLane = [&](const std::vector<float> &values) {
        return std::vector<float>(values.begin() + uint64_t{lane.first} * width,
                                  values.begin() + (uint64_t{lane.first} + lane.rows) * width);
      };
      const std::array<ops::LaneRows, 1> own{{{0, lane.rows}}};
      const Convolved alone =
          convolve(own, c.bound, lane.rows, rowsOfLane(conv), rowsOfLane(gated), rowsOfLane(x),
                   std::span(histories).subspan(s, 1),
                   c.tree ? std::span<const ops::TreeRow>(tree).subspan(lane.first, lane.rows)
                          : std::span<const ops::TreeRow>{});
      same &= std::equal(alone.streams.begin(), alone.streams.end(),
                         packed.streams.begin() + uint64_t{lane.first} * width);
      if (!c.tree) same &= alone.histories[0] == packed.histories[s];
    }
    std::vector<uint32_t> xBits(x.size());
    std::memcpy(xBits.data(), x.data(), x.size() * sizeof(float));
    std::string sizes;
    for (const ops::LaneRows &lane : lanes) sizes += (sizes.empty() ? "" : ", ") + std::to_string(lane.rows);
    const std::string scope = std::string("PLE convolution of ") + (c.tree ? "tree lanes" : "prompt sequences") +
                              " of " + sizes + " rows at a bound of " + std::to_string(c.bound);
    check(same, scope + ": each one's streams" + (c.tree ? "" : " and history") + " as alone, bit for bit");
    check(std::equal(packed.streams.begin() + uint64_t{rows} * width, packed.streams.end(),
                     xBits.begin() + uint64_t{rows} * width),
          scope + ": the spare rows keep their streams");
  }
}

// The PLE commit of a decode step's lanes of 1 to 16 rows, packed one after
// another at a cycle's caps of 8 and 16, in one dispatch: each lane's history
// after its accepted path (a path of its tree, or the first rows of its
// chain) is its own committed alone from row 0 after its own history, bit for
// bit.
void plePackedCommit(MetalBackend &backend) {
  constexpr uint32_t hidden = 256, streams = 4, taps = 4, dilation = 3, width = hidden * streams;
  constexpr uint16_t kFill = 0x7FC1;
  const ops::NgramShape shape{hidden, streams, taps, dilation};
  const uint64_t historyValues = uint64_t{shape.history()} * width;
  std::mt19937 rng(53);
  // The histories `lanes` commit at `bound` rows a lane after `paths`, from
  // the rows of `conv` and each lane's history before.
  const auto commit = [&](std::span<const ops::LaneRows> lanes, uint32_t bound, const std::vector<float> &conv,
                          std::span<const std::vector<float>> histories, std::span<const uint32_t> paths) {
    std::vector<MetalBuffer> in, out;
    for (const std::vector<float> &values : histories) {
      in.push_back(upload(backend, bf16s(values)));
      out.push_back(upload(backend, std::vector<uint16_t>(historyValues, kFill)));
    }
    CommandGraph graph;
    ops::NgramEmbedding::addCommit(
        graph, upload(backend, bf16s(conv)), in, out, upload(backend, std::vector<uint32_t>(paths.begin(), paths.end())),
        shape, laneRowsOf(backend, lanes),
        {static_cast<uint32_t>(lanes.size()), static_cast<uint32_t>(splash::test::rowsEnd(lanes)), bound});
    run(backend, graph);
    std::vector<std::vector<uint16_t>> result;
    for (const MetalBuffer &buffer : out) result.push_back(download<uint16_t>(buffer, historyValues));
    return result;
  };
  // A lane's tree and the last row of the path it accepted.
  struct Lane final {
    std::vector<ops::TreeRow> tree;
    uint32_t leaf;
  };
  struct Case final {
    uint32_t cap;
    std::vector<Lane> lanes;
  };
  const std::vector<Case> cases{
      {8, {{qsaChain(3), 1}, {qsaTree(std::vector<int>{-1, 0, 1, 2, 3, 1, 5, 0}), 6}, {qsaChain(1), 0},
           {qsaTree(std::vector<int>{-1, 0, 0, 1, 2, 3}), 4}}},
      {16, {{qsaTree(std::vector<int>{-1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 0, 11, 12, 13, 14}), 10}, {qsaChain(1), 0},
            {qsaTree(std::vector<int>{-1, 0, 1, 1, 3, 2, 5, 0, 7}), 6}, {qsaChain(12), 11}}},
      {16, {{qsaTree(std::vector<int>{-1, 0, 0, 1, 2, 3, 4}), 0}, {qsaChain(16), 15}}}};
  for (const Case &c : cases) {
    std::vector<ops::LaneRows> lanes;
    std::vector<uint32_t> paths;
    uint32_t rows = 0;
    for (const Lane &lane : c.lanes) {
      lanes.push_back({rows, static_cast<uint32_t>(lane.tree.size())});
      rows += static_cast<uint32_t>(lane.tree.size());
      paths.push_back(lane.tree[lane.leaf].ancestors);
    }
    const auto conv = randoms(rng, uint64_t{rows} * width, 1.0f);
    std::vector<std::vector<float>> histories;
    for (size_t l = 0; l < lanes.size(); ++l) histories.push_back(randoms(rng, historyValues, 1.0f));
    const auto packed = commit(lanes, c.cap, conv, histories, paths);
    bool same = true;
    for (size_t l = 0; l < lanes.size(); ++l) {
      const ops::LaneRows lane = lanes[l];
      const std::array<ops::LaneRows, 1> own{{{0, lane.rows}}};
      const std::vector<float> laneConv(conv.begin() + uint64_t{lane.first} * width,
                                        conv.begin() + (uint64_t{lane.first} + lane.rows) * width);
      same &= commit(own, c.cap, laneConv, std::span(histories).subspan(l, 1), std::span(paths).subspan(l, 1))[0] ==
              packed[l];
    }
    std::string sizes;
    for (const ops::LaneRows &lane : lanes) sizes += (sizes.empty() ? "" : ", ") + std::to_string(lane.rows);
    check(same, "PLE commit of lanes of " + sizes + " rows at a cap of " + std::to_string(c.cap) +
                    ": each lane's history after its path as alone, bit for bit");
  }
}

// Lanes of different row counts packed one after another (5, 8, 1 and 6
// rows from rows 0, 5, 13 and 14) hash their rows as each lane does alone
// from row 0, each from its own tokens before it, bit for bit.
void pleHashPackedLanes(MetalBackend &backend) {
  const uint32_t heads = 16, eos = 248044;
  NgramHashParams params{};
  params.ngram = 3;
  params.heads_per_ngram = 8;
  params.eos = eos;
  const uint64_t multipliers[NGRAM_EMBEDDING_MAXIMUM_NGRAM] = {0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full,
                                                             0x165667B19E3779F9ull};
  std::copy(std::begin(multipliers), std::end(multipliers), params.multipliers);
  for (uint32_t h = 0; h < heads; ++h) {
    params.vocabularies[h] = 19999999 + h * 7919;
    params.offsets[h] = h * 20000000;
  }
  const std::array<ops::LaneRows, 4> packed{{{0, 5}, {5, 8}, {13, 1}, {14, 6}}};
  const std::vector<uint32_t> tokens = {5, 6, eos, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 248319,
                                        0, 12, eos, 13, 14, 15};
  const std::vector<uint32_t> before = {40, 41, NGRAM_EMBEDDING_NONE, NGRAM_EMBEDDING_NONE, eos, 77, 3, eos};
  MetalBuffer indices = scratch(backend, uint64_t{tokens.size()} * heads * 4);
  CommandGraph graph;
  ops::NgramEmbedding::addHash(graph,
                               {upload(backend, tokens), chainsOf(backend, packed), laneRowsOf(backend, packed),
                                upload(backend, before), indices},
                               params, stepOf(packed), heads);
  run(backend, graph);
  const auto got = download<uint32_t>(indices, tokens.size() * heads);
  bool ok = true;
  for (uint32_t lane = 0; lane < packed.size(); ++lane) {
    const ops::LaneRows block = packed[lane];
    const std::array<ops::LaneRows, 1> own{{{0, block.rows}}};
    MetalBuffer alone = scratch(backend, uint64_t{block.rows} * heads * 4);
    CommandGraph single;
    const std::vector<uint32_t> laneTokens(tokens.begin() + block.first, tokens.begin() + block.first + block.rows);
    const std::vector<uint32_t> laneBefore(before.begin() + lane * 2, before.begin() + lane * 2 + 2);
    ops::NgramEmbedding::addHash(single,
                                 {upload(backend, laneTokens), chainsOf(backend, own), laneRowsOf(backend, own),
                                  upload(backend, laneBefore), alone},
                                 params, stepOf(own), heads);
    run(backend, single);
    const auto want = download<uint32_t>(alone, uint64_t{block.rows} * heads);
    ok &= std::equal(want.begin(), want.end(), got.begin() + uint64_t{block.first} * heads);
  }
  check(ok, "PLE hash of packed lanes of 5, 8, 1 and 6 rows: each lane's rows as alone");
}

// A target's QSA scratch (model::Qwen4Steps) takes rows to the end of the
// window it serves: a prefill chunk's last rows, and a verify step's lanes
// of 8 rows, or of a verify block's 16 from the window's last anchor into
// the KV slots past it, at the native window, at the 1,010,000 tokens YaRN
// serves at most and at a window YaRN stretches to 300,020 tokens, where an
// 8-row tile's slack left a 16-row lane's bitmap a block short, by their
// ratio to the native window (model::servedFor), whose prefill scores keep
// the native window's scratch; the native window's scratch refuses rows at
// the end of YaRN's.
void qsaScratchCoversWindow(MetalBackend &backend) {
  constexpr uint32_t kChunk = 2048, kVerify = 8, kBlock = SPLASH_TARGET_VERIFY_ROWS, kLanes = 4;
  const model::Qwen4ExpWeights weights;
  const std::array<uint32_t, 1> captured{47};
  const model::QwenTargetGeometry native(model::Qwen4ExpLayout{}, captured);
  model::QwenTargetGeometry longest = native, odd = native;
  longest.ropeScaling = {1'010'000.0 / 262'144, 262'144};
  odd.ropeScaling = {300'020.0 / 262'144, 262'144};
  const model::QwenTargetGeometry &yarn = longest, &stretched = odd;
  const MetalBuffer table = scratch(backend, 8);
  // A prefill chunk's selection, or a verify step's over lanes of `rows`
  // rows on the device from their anchors at `firsts`.
  const auto selects = [&](const model::Qwen4Scratch &s, uint32_t rows, std::vector<uint32_t> firsts) {
    const std::vector<MetalBuffer> tables(firsts.size(), table);
    std::vector<ops::LaneRows> blocks;
    for (uint32_t lane = 0; lane < firsts.size(); ++lane) blocks.push_back({lane * rows, rows});
    CommandGraph graph;
    try {
      if (firsts.size() == 1)
        ops::BlockIndexer::addSelect(graph, s.qsaQueries, s.qsaScores, s.scoreRows, s.qsaMask,
                                     {rows, firsts[0], table, 0, 512, s.maskWords, s.scoreStride});
      else
        ops::BlockIndexer::addTreeSelect(
            graph, s.qsaQueries, s.qsaBlockKeys, s.qsaScores, s.qsaMask, s.qsaRowMasks,
            {upload(backend, blocks), upload(backend, std::vector<ops::TreeRow>(blocks.size() * rows)),
             static_cast<uint32_t>(blocks.size()) * rows, rows, firsts, {tables, 0}, 512, s.maskWords, s.scoreStride});
    } catch (const std::invalid_argument &) {
      return false;
    }
    return !graph.empty();
  };
  uint64_t nativeScores = 0;
  for (const auto &[geometry, window] :
       {std::pair{&native, 262'144U}, std::pair{&yarn, 1'010'000U}, std::pair{&stretched, 300'020U}}) {
    const model::Qwen4Steps steps(weights, *geometry, backend, 1);
    const model::Qwen4Scratch prefill = steps.prefillScratch(0), verify = steps.verifyScratch(kLanes * kBlock);
    const uint64_t scores = uint64_t{prefill.scoreRows} * prefill.scoreStride;
    if (!nativeScores) nativeScores = scores;
    check(geometry->contextTokens() == window && selects(prefill, kChunk, {window - kChunk}) &&
              selects(verify, kVerify, std::vector<uint32_t>(kLanes, window - kVerify)) &&
              selects(verify, kBlock, std::vector<uint32_t>(kLanes, window - 2)) && scores <= nativeScores,
          "QSA scratch of a " + std::to_string(window) + "-token window: its last rows, within the native scores");
  }
  const model::Qwen4Steps steps(weights, native, backend, 1);
  check(!selects(steps.prefillScratch(0), kChunk, {1'010'000 - kChunk}),
        "QSA scratch of the native window refuses the rows of YaRN's");
}

// A verify step whose rows complete a block and whose lane accepts fewer of
// them, then the next step with other keys for the rejected tokens: the
// block's pooled key is the next step's, from the pending keys the commit
// kept and the new rows', on either side of a page; the rejected rows' block
// never enters the index.
void qsaRollbackTest(MetalBackend &backend) {
  constexpr uint32_t D = kQsaDimension, H = BLOCK_INDEXER_HEADS, rows = 8;
  bool ok = true;
  for (const uint32_t first : {2046u, 2049u, 4093u}) {
    std::mt19937 rng(29 + first);
    const QsaWeights weights = qsaWeights(backend, rng);
    MetalBuffer index;
    QsaLane lane = qsaHistory(backend, rng, first, (first + 2 * rows + kQsaPage - 1) / kQsaPage, index,
                              weights.keyNorm, weights.invFreq);
    const std::vector<float> committed = randoms(rng, uint64_t{2 * rows} * D, 1.0f);
    const uint32_t words = ((first + 2 * rows) / 4 + BLOCK_INDEXER_TILE_SLACK + 31) / 32 + 1;
    // A step of `keys` from `from` that accepts `accepted` rows.
    const auto step = [&](uint32_t from, std::vector<float> keys, uint32_t accepted) {
      const QsaTreeLane chain{from, qsaChain(rows), randoms(rng, uint64_t{rows} * H * D, 1.0f), std::move(keys)};
      const std::array<const QsaLane *, 1> histories{&lane};
      const std::array<uint32_t, 1> path{splash::test::chainPath(accepted)};
      lane.pending[0] = qsaTreeStep(backend, std::span(&chain, 1), histories, 512, words, weights, path).pendingAfter[0];
    };
    // The draft's rows after the first, which holds the committed token, are
    // rejected.
    std::vector<float> draft(committed.begin(), committed.begin() + uint64_t{rows} * D);
    for (uint64_t i = D; i < draft.size(); ++i) draft[i] = -draft[i] + 0.5f;
    step(first, draft, 1);
    step(first + 1, std::vector<float>(committed.begin() + D, committed.begin() + uint64_t{rows + 1} * D), rows);
    // Every block the two steps complete, from the committed keys alone, and
    // the history's before them.
    const auto key = [&](uint32_t t) -> const float * {
      return t < first ? &lane.raw[uint64_t{t} * D] : &committed[uint64_t{t - first} * D];
    };
    const auto gotP = download<uint16_t>(index, index.sizeBytes() / 2);
    for (uint32_t b = first / 4; b < (first + 1 + rows) / 4; ++b) {
      std::vector<double> mean(D, 0);
      for (uint32_t t = 4 * b; t < 4 * b + 4; ++t)
        for (uint32_t d = 0; d < D; ++d) mean[d] += f32(bf16(key(t)[d])) / 4;
      double sum = 0;
      for (double e : mean) sum += e * e;
      const double scale = 1 / std::sqrt(sum / D + 1e-6);
      // Dimensions past the rotated ones are the normalized mean times the
      // key norm.
      const auto norm = download<float>(weights.keyNorm.buffer, D);
      for (uint32_t d = 64; d < D; ++d) {
        const double want = mean[d] * scale * norm[d], got = f32(gotP[(uint64_t{lane.table[b / 8]} * 8 + b % 8) * D + d]);
        ok &= std::fabs(got - want) <= std::fabs(want) / 256 + 2e-4;
      }
    }
  }
  check(ok, "QSA rollback: a block a rejected row completed pools the next step's keys");
}

// The selection refuses scores or bitmaps too narrow for its rows' history,
// past which its kernels would write: rows at 4092 on complete 1025 blocks,
// which take 1025 scores and, with the tail block, 1026 bits.
void qsaSelectRefusesNarrowScratch(MetalBackend &backend) {
  const MetalBuffer any = scratch(backend, 1 << 20);
  const auto refused = [&](uint32_t maskWords, uint32_t scoreStride) {
    CommandGraph graph;
    try {
      ops::BlockIndexer::addSelect(graph, any, any, 8, any, {8, 4092, any, 0, 512, maskWords, scoreStride});
    } catch (const std::invalid_argument &error) {
      return std::string(error.what()) == "indexer scores or bitmaps narrower than the rows' history" && graph.empty();
    }
    return false;
  };
  check(refused(32, 1056) && refused(33, 1024) && !refused(33, 1025),
        "QSA select refuses scores or bitmaps narrower than its rows' history");
}

// The capture of a layer's output for the DFlash2 draft: each row's streams'
// mean, bf16 rounded once from its fp32 sum, in its capture slot of wider
// rows from a later row, the rest of those rows as they were.
void streamMeanTest(MetalBackend &backend) {
  std::mt19937 rng(17);
  const uint32_t rows = 5, hidden = 256, streams = 4, width = hidden * streams;
  const uint32_t stride = 3 * hidden, column = hidden, firstRow = 2;
  const ops::HyperShape shape{hidden, streams, 0};
  const auto values = randoms(rng, rows * width, 2.0f);
  constexpr uint16_t kPoison = 0x7FC1;
  const std::vector<uint16_t> before((firstRow + rows) * stride, kPoison);
  MetalBuffer captured = upload(backend, before);
  CommandGraph graph;
  ops::HyperConnection::addStreamMean(graph, upload(backend, values), captured, {firstRow, stride, column}, shape,
                                      rows);
  run(backend, graph);
  const auto got = download<uint16_t>(captured, before.size());
  bool ok = true;
  for (uint32_t r = 0; r < firstRow + rows; ++r)
    for (uint32_t c = 0; c < stride; ++c) {
      const uint16_t value = got[r * stride + c];
      if (r < firstRow || c < column || c >= column + hidden) {
        ok &= value == kPoison;
        continue;
      }
      const uint32_t row = r - firstRow, d = c - column;
      float sum = 0.0f;
      for (uint32_t stream = 0; stream < streams; ++stream) sum += values[row * width + stream * hidden + d];
      ok &= value == bf16(sum / float(streams));
    }
  check(ok, "stream mean: each row's streams' mean, bf16, in its slot of wider rows");
}

// The PLE hash on the GPU against the same hash on the CPU, as
// model/Qwen4Steps.cpp's qwen4PleRows computes it, of three lanes of
// splash::ops::kNarrowLaneRows rows each.
void pleHashTest(MetalBackend &backend) {
  const uint32_t lanes = 3, laneRows = splash::ops::kNarrowLaneRows, ngram = 3, perNgram = 8, heads = 16, eos = 248044;
  NgramHashParams params{};
  params.ngram = ngram;
  params.heads_per_ngram = perNgram;
  params.eos = eos;
  const uint64_t multipliers[NGRAM_EMBEDDING_MAXIMUM_NGRAM] = {0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full,
                                                         0x165667B19E3779F9ull};
  std::copy(std::begin(multipliers), std::end(multipliers), params.multipliers);
  for (uint32_t h = 0; h < heads; ++h) {
    params.vocabularies[h] = 19999999 + h * 7919;
    params.offsets[h] = h * 20000000;
  }
  const std::vector<uint32_t> tokens = {5, 6, eos, 7, 8, 9, 10, 11,      1, 2, 3, 4, 5, 6, 7, 8,
                                        248319, 0, 12, 13, eos, eos, 14, 15};
  // The ngram - 1 tokens before each lane's rows, nearest first. Lane 0:
  // two tokens; lane 1: a sequence start; lane 2: EOS just before.
  const std::vector<uint32_t> before = {40, 41, NGRAM_EMBEDDING_NONE, NGRAM_EMBEDDING_NONE, eos, 77};
  MetalBuffer indices = scratch(backend, lanes * laneRows * heads * 4);
  const std::array<ops::LaneRows, lanes> blocks{{{0, laneRows}, {laneRows, laneRows}, {2 * laneRows, laneRows}}};
  CommandGraph graph;
  ops::NgramEmbedding::addHash(graph,
                               {upload(backend, tokens), chainsOf(backend, blocks), laneRowsOf(backend, blocks),
                                upload(backend, before), indices},
                               params, stepOf(blocks), heads);
  run(backend, graph);
  const auto got = download<uint32_t>(indices, lanes * laneRows * heads);
  bool ok = true;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t r = 0; r < laneRows; ++r) {
      uint64_t context[NGRAM_EMBEDDING_MAXIMUM_NGRAM] = {tokens[lane * laneRows + r]};
      bool cut = false;
      for (uint32_t back = 1; back < ngram; ++back) {
        const uint32_t previous =
            back <= r ? tokens[lane * laneRows + r - back] : before[lane * (ngram - 1) + back - r - 1];
        cut = cut || previous == NGRAM_EMBEDDING_NONE || previous == eos;
        context[back] = cut ? eos : previous;
      }
      for (uint32_t gram = 2; gram <= ngram; ++gram) {
        uint64_t mixed = context[0] * multipliers[0];
        for (uint32_t j = 1; j < gram; ++j) mixed ^= context[j] * multipliers[j];
        for (uint32_t g = 0; g < perNgram; ++g) {
          const uint32_t head = (gram - 2) * perNgram + g;
          ok &= got[(lane * laneRows + r) * heads + head] ==
                uint32_t(mixed % params.vocabularies[head] + params.offsets[head]);
        }
      }
    }
  check(ok, "PLE hash: GPU rows equal the CPU's");
}

// A verify block that is a tree hashes each row's n-grams along its path:
// its parents' tokens up to the anchor, then those before its lane's first
// row, whatever rows the step packs before it. Lane 0's six rows: 1 and 3
// follow the anchor, 2 and 5 row 1, 4 row 3, so rows 3 and 5 follow rows
// other than the rows before them; lane 1, from row 6, a chain of three.
// The GPU's rows equal a CPU walk of each row's parents, and rows 3 and 5
// differ from a hash of the rows before them.
void pleHashTreeRows(MetalBackend &backend) {
  const uint32_t ngram = 3, perNgram = 8, heads = 16, eos = 248044;
  NgramHashParams params{};
  params.ngram = ngram;
  params.heads_per_ngram = perNgram;
  params.eos = eos;
  const uint64_t multipliers[NGRAM_EMBEDDING_MAXIMUM_NGRAM] = {0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full,
                                                             0x165667B19E3779F9ull};
  std::copy(std::begin(multipliers), std::end(multipliers), params.multipliers);
  for (uint32_t h = 0; h < heads; ++h) {
    params.vocabularies[h] = 19999999 + h * 7919;
    params.offsets[h] = h * 20000000;
  }
  const std::array<ops::LaneRows, 2> lanes{{{0, 6}, {6, 3}}};
  const std::vector<uint16_t> parents{SPLASH_TREE_ROOT, 0, 1, 0, 3, 1, SPLASH_TREE_ROOT, 0, 1};
  const std::vector<uint32_t> tokens{100, 101, 102, 103, 104, 105, 200, 201, 202};
  const std::vector<uint32_t> before{40, 41, 50, NGRAM_EMBEDDING_NONE};
  std::vector<ops::TreeRow> tree(tokens.size());
  for (const ops::LaneRows &lane : lanes)
    for (uint32_t row = 0; row < lane.rows; ++row) {
      const uint16_t parent = parents[lane.first + row];
      const ops::TreeRow up = row ? tree[lane.first + parent] : ops::TreeRow{0, 0, 0};
      tree[lane.first + row] = {(row ? up.ancestors : 0U) | (1U << row), parent,
                                static_cast<uint16_t>(row ? up.depth + 1 : 0)};
    }
  MetalBuffer indices = scratch(backend, uint64_t{tokens.size()} * heads * 4);
  CommandGraph graph;
  ops::NgramEmbedding::addHash(
      graph, {upload(backend, tokens), upload(backend, tree), laneRowsOf(backend, lanes), upload(backend, before), indices},
      params, stepOf(lanes), heads);
  run(backend, graph);
  const auto got = download<uint32_t>(indices, tokens.size() * heads);
  // A row's heads from its token and the ngram - 1 before it, `previous`.
  const auto hashed = [&](uint32_t token, std::array<uint32_t, NGRAM_EMBEDDING_MAXIMUM_NGRAM - 1> previous) {
    std::array<uint64_t, NGRAM_EMBEDDING_MAXIMUM_NGRAM> context{token};
    bool cut = false;
    for (uint32_t back = 1; back < ngram; ++back) {
      cut = cut || previous[back - 1] == NGRAM_EMBEDDING_NONE || previous[back - 1] == eos;
      context[back] = cut ? eos : previous[back - 1];
    }
    std::vector<uint32_t> rowHeads(heads);
    for (uint32_t gram = 2; gram <= ngram; ++gram) {
      uint64_t mixed = context[0] * multipliers[0];
      for (uint32_t j = 1; j < gram; ++j) mixed ^= context[j] * multipliers[j];
      for (uint32_t g = 0; g < perNgram; ++g) {
        const uint32_t head = (gram - 2) * perNgram + g;
        rowHeads[head] = uint32_t(mixed % params.vocabularies[head] + params.offsets[head]);
      }
    }
    return rowHeads;
  };
  bool ok = true, branchesDiffer = true;
  for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
    const ops::LaneRows block = lanes[lane];
    for (uint32_t row = 0; row < block.rows; ++row) {
      std::array<uint32_t, NGRAM_EMBEDDING_MAXIMUM_NGRAM - 1> path{}, packed{};
      uint32_t at = row;
      for (uint32_t back = 1; back < ngram; ++back) {
        const uint32_t depth = tree[block.first + row].depth;
        if (back <= depth) at = parents[block.first + at];
        path[back - 1] = back <= depth ? tokens[block.first + at] : before[lane * (ngram - 1) + back - depth - 1];
        packed[back - 1] = back <= row ? tokens[block.first + row - back] : before[lane * (ngram - 1) + back - row - 1];
      }
      const std::vector<uint32_t> want = hashed(tokens[block.first + row], path);
      ok &= std::equal(want.begin(), want.end(), got.begin() + uint64_t{block.first + row} * heads);
      if (lane == 0 && (row == 3 || row == 5))
        branchesDiffer &= hashed(tokens[block.first + row], packed) != want;
    }
  }
  check(ok && branchesDiffer, "PLE hash of a tree: each row's n-grams along its path");
}

using Extents = std::initializer_list<splash::test::BufferExtent>;
using Buffers = std::vector<MetalBuffer>;

// Each buffer the hyper-connection kernels reach, at Flash-Next's shape over
// five rows: the streams (fp32), the norm's F32 weights, every stream's
// injection weights of the normalized rows and each row's weight per stream,
// the normalized and gate rows (bf16), the low rank's rows, the block's input
// and branch, and the stream mean's rows; and the norm's refusal of bf16
// weights.
void hyperExtentsTest(MetalBackend &backend) {
  const uint32_t rows = 5, hidden = kHidden, streams = kStreams, rank = kRank, width = hidden * streams;
  const ops::HyperShape shape{hidden, streams, rank};
  const uint64_t streamBytes = uint64_t{rows} * width * 4, weightBytes = uint64_t{rows} * streams * 4;
  const uint64_t normalizedBytes = uint64_t{rows} * width * 2, hiddenBytes = uint64_t{rows} * hidden * 2;
  splash::test::requireExtents(backend,
                               Extents{{0, hiddenBytes, 2, "hyper-connection embedding"},
                                       {1, streamBytes, 4, "hyper-connection streams"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 ops::HyperConnection::addInit(graph, b[0], b[1], shape, rows);
                               });
  splash::test::requireExtents(backend,
                               Extents{{0, streamBytes, 4, "hyper-connection streams"},
                                       {1, uint64_t{width} * 4, 4, "hyper-connection norm"},
                                       {2, uint64_t{streams} * width * 4, 4, "hyper-connection injection"},
                                       {3, normalizedBytes, 2, "hyper-connection normalized"},
                                       {4, weightBytes, 4, "hyper-connection weight"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 ops::HyperConnection::addNorm(graph, b[0], {b[1], true}, b[2], b[3], b[4], shape,
                                                               rows);
                               });
  splash::test::requireExtents(backend, Extents{{0, uint64_t{rows} * rank * 2, 2, "hyper-connection low"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 ops::HyperConnection::addLow(graph, b[0], shape, rows);
                               });
  splash::test::requireExtents(backend,
                               Extents{{0, normalizedBytes, 2, "hyper-connection normalized"},
                                       {1, normalizedBytes, 2, "hyper-connection gate"},
                                       {2, hiddenBytes, 2, "hyper-connection mix"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 ops::HyperConnection::addMix(graph, b[0], b[1], b[2], shape, rows);
                               });
  splash::test::requireExtents(backend,
                               Extents{{0, streamBytes, 4, "hyper-connection streams"},
                                       {1, hiddenBytes, 2, "hyper-connection branch"},
                                       {2, weightBytes, 4, "hyper-connection weight"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 ops::HyperConnection::addCombine(graph, b[0], b[1], b[2], shape, rows);
                               });
  // Into column `hidden` of rows of three hidden widths, from row 2.
  splash::test::requireExtents(backend,
                               Extents{{0, streamBytes, 4, "stream mean input"},
                                       {1, uint64_t{2 + rows} * 3 * hidden * 2, 2, "stream mean output"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 ops::HyperConnection::addStreamMean(graph, b[0], b[1], {2, 3 * hidden, hidden},
                                                                     shape, rows);
                               });
  CommandGraph refused;
  splash::test::rejects(
      [&] {
        ops::HyperConnection::addNorm(refused, scratch(backend, streamBytes), {scratch(backend, width * 4), false},
                                      scratch(backend, uint64_t{streams} * width * 4),
                                      scratch(backend, normalizedBytes), scratch(backend, weightBytes), shape, rows);
      },
      "the hyper-connection norm is not F32", "a hyper-connection norm of bf16 weights was encoded");
  check(refused.empty(), "hyper-connection buffers at their extents, and bf16 norm weights refused");
}

// Each buffer the PLE kernels reach, over seven rows of four streams of 256
// values and a history of three taps three rows apart: the gate's key, value
// and stream rows, its three F32 norms, gated values and convolution input;
// the convolution's input rows, history before and after, taps, gated values
// and streams; the history after a sequence's first rows and after a lane's
// retained rows, of a verify step's rows, and the retained count; the hash's
// tokens, each lane's tokens before it (19 rows: two whole lanes and three
// rows of a third) and its row ids; and the gate's refusal of each of its
// norms in bf16.
void pleExtentsTest(MetalBackend &backend) {
  const uint32_t rows = 7, hidden = 256, streams = 4, taps = 4, dilation = 3, width = hidden * streams;
  const ops::NgramShape shape{hidden, streams, taps, dilation};
  const uint64_t history = uint64_t{shape.history()} * width * 2;
  const uint64_t inputBytes = uint64_t{rows} * width * 2, floatBytes = uint64_t{rows} * width * 4;
  splash::test::requireExtents(backend,
                               Extents{{0, inputBytes, 2, "n-gram embedding key"},
                                       {1, uint64_t{rows} * hidden * 2, 2, "n-gram embedding value"},
                                       {2, floatBytes, 4, "n-gram embedding streams"},
                                       {3, uint64_t{width} * 4, 4, "n-gram embedding key norm"},
                                       {4, uint64_t{width} * 4, 4, "n-gram embedding query norm"},
                                       {5, uint64_t{width} * 4, 4, "n-gram embedding convolution norm"},
                                       {6, floatBytes, 4, "n-gram embedding gated"},
                                       {7, inputBytes, 2, "n-gram embedding convolution input"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 ops::NgramEmbedding::addGate(graph, b[0], b[1], b[2], {b[3], true}, {b[4], true},
                                                              {b[5], true}, b[6], b[7], shape, rows);
                               });
  // A prompt chunk's two sequences, each its history before and after.
  splash::test::requireExtents(backend,
                               Extents{{0, inputBytes, 2, "n-gram embedding convolution input"},
                                       {1, history, 2, "n-gram embedding history"},
                                       {2, history, 2, "n-gram embedding history"},
                                       {3, history, 2, "n-gram embedding next history"},
                                       {4, history, 2, "n-gram embedding next history"},
                                       {5, uint64_t{width} * taps * 4, 4, "n-gram embedding convolution taps"},
                                       {6, floatBytes, 4, "n-gram embedding gated"},
                                       {7, floatBytes, 4, "n-gram embedding streams"},
                                       {8, 2 * sizeof(ops::LaneRows), sizeof(ops::LaneRows),
                                        "n-gram embedding layout"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 const std::array<MetalBuffer, 2> in{b[1], b[2]}, out{b[3], b[4]};
                                 ops::NgramEmbedding::addConvolution(graph, b[0], in, out, b[5], b[6], b[7], shape,
                                                                     b[8], {2, rows, rows});
                               });
  // A verify block's convolution, along its rows' paths.
  splash::test::requireExtents(backend,
                               Extents{{0, inputBytes, 2, "n-gram embedding convolution input"},
                                       {1, history, 2, "n-gram embedding history"},
                                       {2, uint64_t{width} * taps * 4, 4, "n-gram embedding convolution taps"},
                                       {3, floatBytes, 4, "n-gram embedding gated"},
                                       {4, floatBytes, 4, "n-gram embedding streams"},
                                       {5, uint64_t{rows} * sizeof(splash::ops::TreeRow), 8,
                                        "n-gram embedding tree rows"},
                                       {6, sizeof(ops::LaneRows), sizeof(ops::LaneRows), "n-gram embedding layout"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 ops::NgramEmbedding::addConvolution(graph, b[0], std::span(&b[1], 1), {}, b[2], b[3],
                                                                     b[4], shape, b[6], {1, rows, rows}, b[5]);
                               });
  splash::test::requireExtents(backend,
                               Extents{{0, inputBytes, 2, "n-gram embedding convolution input"},
                                       {1, history, 2, "n-gram embedding history"},
                                       {2, history, 2, "n-gram embedding next history"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 ops::NgramEmbedding::addHistory(graph, b[0], b[1], b[2], shape, rows);
                               });
  // Two lanes' commit, each its history before and after its path.
  splash::test::requireExtents(
      backend,
      Extents{{0, uint64_t{2 * splash::ops::kNarrowLaneRows} * width * 2, 2, "n-gram embedding convolution input"},
              {1, history, 2, "n-gram embedding history"},
              {2, history, 2, "n-gram embedding history"},
              {3, history, 2, "n-gram embedding next history"},
              {4, history, 2, "n-gram embedding next history"},
              {5, 2 * 4, 4, "n-gram embedding accepted path"},
              {6, 2 * sizeof(ops::LaneRows), sizeof(ops::LaneRows), "n-gram embedding layout"}},
      [&](CommandGraph &graph, const Buffers &b) {
        const std::array<MetalBuffer, 2> in{b[1], b[2]}, out{b[3], b[4]};
        ops::NgramEmbedding::addCommit(graph, b[0], in, out, b[5], shape, b[6],
                                       {2, 2 * splash::ops::kNarrowLaneRows, splash::ops::kNarrowLaneRows});
      });
  NgramHashParams params{};
  params.ngram = 3;
  params.heads_per_ngram = 8;
  const uint32_t heads = 16;
  const std::array<ops::LaneRows, 3> blocks{{{0, splash::ops::kNarrowLaneRows},
                                             {splash::ops::kNarrowLaneRows, splash::ops::kNarrowLaneRows},
                                             {2 * splash::ops::kNarrowLaneRows, 3}}};
  const uint64_t hashRows = 2 * splash::ops::kNarrowLaneRows + 3;
  splash::test::requireExtents(backend,
                               Extents{{0, hashRows * 4, 4, "n-gram hash token"},
                                       {1, hashRows * sizeof(ops::TreeRow), sizeof(ops::TreeRow),
                                        "n-gram hash tree row"},
                                       {2, 3 * sizeof(ops::LaneRows), sizeof(ops::LaneRows), "n-gram hash lane rows"},
                                       {3, 3 * 2 * 4, 4, "n-gram hash prior token"},
                                       {4, hashRows * heads * 4, 4, "n-gram hash index"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 ops::NgramEmbedding::addHash(graph, {b[0], b[1], b[2], b[3], b[4]}, params,
                                                              stepOf(blocks), heads);
                               });
  CommandGraph refused;
  const MetalBuffer norm = scratch(backend, uint64_t{width} * 4);
  // The key, query and convolution norms in turn in bf16.
  for (uint32_t bfloat16 = 0; bfloat16 < 3; ++bfloat16)
    splash::test::rejects(
        [&] {
          ops::NgramEmbedding::addGate(refused, scratch(backend, inputBytes), scratch(backend, inputBytes),
                                       scratch(backend, floatBytes), {norm, bfloat16 != 0}, {norm, bfloat16 != 1},
                                       {norm, bfloat16 != 2}, scratch(backend, floatBytes),
                                       scratch(backend, inputBytes), shape, rows);
        },
        "the n-gram embedding norms are not F32", "a PLE gate of a bf16 norm was encoded");
  check(refused.empty(), "PLE buffers at their extents, and each bf16 norm refused");
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      const bool bitwise = argc == 3 && std::string(argv[2]) == "bitwise";
      if (argc != 2 && !bitwise) throw std::runtime_error("usage: qwen4-kernels METALLIB [bitwise]");
      MetalBackend backend(argv[1]);
      if (bitwise) {
        // Uninstrumented: the norm's kernels alone, exact.
        if (std::getenv("MTL_SHADER_VALIDATION")) throw std::runtime_error("bitwise runs without shader validation");
        hyperNormKernelsTest(backend, true);
      } else {
        hyperTest(backend);
        hyperTableTest(backend);
        hyperNormKernelsTest(backend, !std::getenv("MTL_SHADER_VALIDATION"));
        pleTest(backend);
        // A chain of four, a branch off row 1 and one off the anchor,
        // committing the paths of row 6 (rows 0, 1, 5 and 6) and row 4 (a
        // chain of five).
        pleTreeTest(backend, std::array<uint32_t, 8>{0, 0, 1, 2, 3, 1, 5, 0}, 4, 2, std::array<uint32_t, 2>{6, 4},
                    false);
        // Flash-Next's taps (a history of 9 rows) over a 16-row block: a chain
        // through row 3 that skips row 4 to run on from row 5 to depth 12, and
        // a branch off row 1 (rows 4, 14 and 15). Rows 10 to 13 take every tap
        // from their paths, and the path to row 13 (13 rows) holds the whole
        // history after it.
        pleTreeTest(backend,
                    std::array<uint32_t, 16>{0, 0, 1, 2, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 4, 14}, 4, 3,
                    std::array<uint32_t, 2>{13, 15}, true);
        plePackedLanes(backend);
        plePackedCommit(backend);
        qsaTest(backend, {150, 20, 16, 0, 1, 7});
        // A threshold inside a run of equal scores: fewer other blocks than
        // the budget.
        qsaTest(backend, {4100, 20, 600, 700, 1, 2048});
        // Sequences of 8 rows on four and three lanes, sparse and causal
        // rows among them.
        qsaTest(backend, {2050, 8, 512, 0, 4, 1021});
        qsaTest(backend, {700, 8, 160, 120, 3, 2});
        // Rows 2045 to 2060 at Flash-Next's budget, the indexer's semantics
        // across the dense-to-sparse transition: up to 2050 every block, from
        // 2051 (513 complete blocks) the top 512, the tail in either case;
        // lanes on both sides of it.
        qsaTest(backend, {2061, 16, 512, 0, 1, 1023});
        qsaTest(backend, {2047, 8, 512, 0, 4, 509});
        // Past 32 768 blocks, where each select thread owns two bitmap words.
        qsaTest(backend, {140000, 8, 512, 0, 1, 2047});
        // With YaRN: four lanes, and a row at the end of the 1,010,000
        // tokens it serves, past 252,000 blocks.
        qsaTest(backend, {2050, 8, 512, 0, 4, 1021, true});
        qsaTest(backend, {1'010'000, 8, 512, 0, 1, 2047, true});
        // A verify block's tree against its paths run as chains: past a
        // small budget, where rows drop blocks of their own, and around
        // Flash-Next's (2046: rows of depth 2 or less dense, deeper ones
        // sparse), at anchors 2 and 1 past a block.
        qsaTreeAsChains(backend, 66, 12);
        qsaTreeAsChains(backend, 602, 16);
        qsaTreeAsChains(backend, 2046, 512);
        qsaTreeAsChains(backend, 4093, 512);
        // A chain against its rows decoded one a step, on either side of
        // Flash-Next's budget and of a page, accepting all of it or part.
        qsaChainAsSequential(backend, 2045, 16, 16, 512);
        qsaChainAsSequential(backend, 4093, 8, 5, 512);
        qsaChainAsSequential(backend, 602, 16, 11, 16);
        qsaPackedLanes(backend);
        qsaCommitLayers(backend);
        qsaProjectPackedLanes(backend);
        qsaScratchCoversWindow(backend);
        qsaRollbackTest(backend);
        qsaSelectRefusesNarrowScratch(backend);
        streamMeanTest(backend);
        pleHashTest(backend);
        pleHashPackedLanes(backend);
        pleHashTreeRows(backend);
        hyperExtentsTest(backend);
        pleExtentsTest(backend);
      }
    } catch (const std::exception &error) {
      std::cerr << "qwen4 kernel test error: " << error.what() << '\n';
      return 1;
    }
  }
  if (failures) {
    std::cout << "Qwen4 kernel tests FAILED (" << failures << " failures)\n";
    return 1;
  }
  std::cout << "Qwen4 kernel tests PASS\n";
  return 0;
}
