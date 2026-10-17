// Qwen3.8-Flash-Next's own kernels (kernels/shared/qwen4.metal and
// qwen4_qsa.metal) through ops::Qwen4, against fp64 CPU references of the
// model's computation:
// - the hyper-connection mix at Flash-Next's shape: per-stream RMS norm,
//   injection weights, the low-rank gate and the stream average; the combine
//   and the streams' start; the norm's two kernels against each other, bit
//   for bit;
// - the PLE gate, its dilated convolution over a history, the history after
//   a sequence's rows and after a verify step's retained rows;
// - QSA: the indexer's normalized, rotated queries (bf16) and raw keys at
//   their KV page slots, pooled keys (bf16), and each row's block bitmap
//   (dense below the budget; above it the top blocks, ties towards the
//   earlier block, and the tail block after the row's complete ones);
// - the capture of a layer's output for the DFlash2 draft (the mean of its
//   streams); the GPU PLE hash against the CPU's.
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/ops/Qwen4.hpp"
#include "../../../runtime/ops/Sampling.hpp"
#include "LinearInputReference.hpp"
#include "metal/abi/KvExtent.h"
#include "metal/abi/Qwen4.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
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
#include <vector>

namespace {

using splash::metal::BufferReach;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::NormWeights;
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

// Flash-Next's mix: QWEN4_MAXIMUM_STREAMS streams of QWEN4_NORM_HIDDEN
// values, the hyper-connections' low rank of 320.
constexpr uint32_t kHidden = QWEN4_NORM_HIDDEN, kStreams = QWEN4_MAXIMUM_STREAMS, kRank = 320;

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
  ops::Qwen4::addNorm(graph, xs, nw, upload(backend, inject), normalized, weights, shape, rows);
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
  ops::Qwen4::addLow(graph2, lowBuffer, shape, rows);
  ops::Qwen4::addMix(graph2, normalized, gateBuffer, mixed, shape, rows);
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
  ops::Qwen4::addCombine(graph3, xs, branchBuffer, weights, shape, rows);
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
  ops::Qwen4::addInit(graph4, branchBuffer, streamsOut, shape, rows);
  run(backend, graph4);
  const auto gotInit = download<float>(streamsOut, rows * width);
  bool okI = true;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t s = 0; s < streams; ++s)
      for (uint32_t d = 0; d < hidden; ++d)
        okI &= gotInit[r * width + s * hidden + d] == f32(bf16(branch[r * hidden + d]));
  check(okI, "hyper init: every stream the embedding");
}

// The table variants (norm, low and mix into a register-tile projection's
// Table16 table): bit for bit the plain kernel's output followed by the
// projection's own decode_linear_gguf_prepare, over two lanes of eight rows.
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
  MetalBuffer normalizedT = zeroed(rows * width * 2), weightsT = zeroed(rows * streams * 4);
  const ops::LinearScratch normRef = table(width), normGot = table(width);
  CommandGraph norm;
  ops::Qwen4::addNorm(norm, xs, nw, inject, normalized, weights, shape, rows);
  prepare(norm, normalized, normRef, width);
  const ops::PreparedInput prepared =
      ops::Qwen4::addNorm(norm, xs, nw, inject, normalizedT, weightsT, shape, rows, normGot, ops::LinearInput::Table16);
  run(backend, norm);
  check(prepared.layout == ops::LinearInput::Table16 && prepared.source.sameView(normalizedT),
        "hyper norm table: names its output and table layout");
  check(same(normalized, normalizedT) && same(weights, weightsT),
        "hyper norm table: output and weights as the plain norm's");
  check(sameTables(normRef, normGot), "hyper norm table: table and sums as the prepare's");

  const std::vector<uint16_t> low = bf16s(randoms(rng, rows * rank, 2.0f));
  MetalBuffer lowPlain = upload(backend, low), lowTable = upload(backend, low);
  const ops::LinearScratch lowRef = table(rank), lowGot = table(rank);
  CommandGraph lows;
  ops::Qwen4::addLow(lows, lowPlain, shape, rows);
  prepare(lows, lowPlain, lowRef, rank);
  ops::Qwen4::addLow(lows, lowTable, shape, rows, lowGot, ops::LinearInput::Table16);
  run(backend, lows);
  check(sameTables(lowRef, lowGot), "hyper low table: table and sums as the prepare's");
  check(download<uint16_t>(lowTable, rows * rank) == low, "hyper low table: leaves its input as it was");

  const MetalBuffer gate = upload(backend, bf16s(randoms(rng, rows * width, 1.0f)));
  MetalBuffer mixed = zeroed(rows * hidden * 2), mixedT = zeroed(rows * hidden * 2);
  const ops::LinearScratch mixRef = table(hidden), mixGot = table(hidden);
  CommandGraph mix;
  ops::Qwen4::addMix(mix, normalized, gate, mixed, shape, rows);
  prepare(mix, mixed, mixRef, hidden);
  ops::Qwen4::addMix(mix, normalized, gate, mixedT, shape, rows, mixGot, ops::LinearInput::Table16);
  run(backend, mix);
  check(same(mixed, mixedT), "hyper mix table: output as the plain mix's");
  check(sameTables(mixRef, mixGot), "hyper mix table: table and sums as the prepare's");
}

// The norm's kernels (kernels/shared/qwen4.metal): ops::Qwen4::addNorm runs
// an injecting mix of up to QWEN4_NORM_QUARTER_ROWS rows on the quarters
// kernel and the rest on the loaded one, which sum alike: at decode's row
// counts and past them, the two write bit for bit the same normalized rows,
// injection weights and tables, and a mix that injects nothing the same
// normalized rows. Shader validation's instrumentation compiles the kernels
// apart, which can move a bf16 value by its last bit; under it (`exact`
// false) they agree within that, and test-engine-metal runs the exact
// comparison uninstrumented (`qwen4-kernels METALLIB bitwise`).
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
  const auto table = [&](uint32_t rows) {
    return ops::LinearScratch{zeroed(ops::tableBytes(width, rows)),
                              zeroed(ops::tableSumsBytes(ops::LinearInput::Table16, width, rows)), {}, {}};
  };
  bool okKernels = true, okPlain = true, okTables = true;
  for (const uint32_t rows : {1u, 3u, 8u, 16u, 24u, 32u, 64u, 65u, 128u, 136u, 256u, 512u}) {
    // addNorm's kernel, and the other one on the same rows.
    const bool quarters = rows <= QWEN4_NORM_QUARTER_ROWS;
    const std::string other = quarters ? "qwen4_hyper_norm_loaded" : "qwen4_hyper_norm_quarters";
    const uint32_t otherThreads = quarters ? QWEN4_ROW_THREADS : QWEN4_NORM_QUARTER_THREADS;
    const Qwen4HyperParams params{rows, hidden, streams, rank, 1};
    MetalBuffer normalized = zeroed(rows * width * 2), weights = zeroed(rows * streams * 4);
    MetalBuffer normalizedOther = zeroed(rows * width * 2), weightsOther = zeroed(rows * streams * 4);
    MetalBuffer normalizedPlain = zeroed(rows * width * 2);
    CommandGraph graph;
    ops::Qwen4::addNorm(graph, xs, nw, inject, normalized, weights, shape, rows);
    graph.add(other, {xs, nw.buffer, inject, normalizedOther, weightsOther}, params, {rows, 1, 1},
              {otherThreads, 1, 1});
    ops::Qwen4::addNorm(graph, xs, nw, MetalBuffer{}, normalizedPlain, MetalBuffer{}, shape, rows);
    // A decode step's rows also into the down projection's table.
    const bool tables = rows % 8 == 0;
    MetalBuffer normalizedT = zeroed(rows * width * 2), weightsT = zeroed(rows * streams * 4);
    MetalBuffer normalizedOtherT = zeroed(rows * width * 2), weightsOtherT = zeroed(rows * streams * 4);
    const ops::LinearScratch tableGot = table((rows + 7) / 8 * 8), tableOther = table((rows + 7) / 8 * 8);
    if (tables) {
      static_cast<void>(ops::Qwen4::addNorm(graph, xs, nw, inject, normalizedT, weightsT, shape, rows, tableGot,
                                            ops::LinearInput::Table16));
      graph.add(other + "_table16",
                {xs, nw.buffer, inject, normalizedOtherT, weightsOtherT, tableOther.input, tableOther.sums}, params,
                {rows, 1, 1}, {otherThreads, 1, 1});
    }
    run(backend, graph);
    const uint64_t rowBytes = uint64_t{rows} * width * 2, weightBytes = uint64_t{rows} * streams * 4;
    const bool kernels = same(normalized, normalizedOther, rowBytes) && same(weights, weightsOther, weightBytes, true);
    okKernels &= kernels;
    if (!kernels) std::cout << "  the kernels differ at " << rows << " rows\n";
    const bool plain = same(normalizedPlain, normalized, rowBytes);
    okPlain &= plain;
    if (!plain) std::cout << "  a mix without injection differs at " << rows << " rows\n";
    if (tables) {
      const bool equal = same(normalizedT, normalized, rowBytes) && same(weightsT, weights, weightBytes, true) &&
                         same(normalizedOtherT, normalized, rowBytes) &&
                         same(tableGot.input, tableOther.input, tableGot.input.sizeBytes()) &&
                         // The sums of the table's values, which inexact can move further.
                         (!exact || same(tableGot.sums, tableOther.sums, tableGot.sums.sizeBytes()));
      okTables &= equal;
      if (!equal) std::cout << "  the table kernels differ at " << rows << " rows\n";
    }
  }
  const std::string how = exact ? "bit for bit" : "close to (instrumented)";
  check(okKernels, "hyper norm: the quarters and loaded kernels' output and weights " + how);
  check(okPlain, "hyper norm without injection: normalized rows " + how + " the injecting mix's");
  check(okTables, "hyper norm tables: the quarters and loaded kernels' output, weights and tables " + how);
}

void pleTest(MetalBackend &backend) {
  std::mt19937 rng(11);
  const uint32_t rows = 7, hidden = 256, streams = 4, taps = 4, dilation = 3, width = hidden * streams;
  const ops::PleShape shape{hidden, streams, taps, dilation};
  const uint32_t history = shape.history();
  const auto key = randoms(rng, rows * width, 1.0f), value = randoms(rng, rows * hidden, 1.0f),
             x = randoms(rng, rows * width, 1.0f), kn = randoms(rng, width, 0.2f), qn = randoms(rng, width, 0.2f),
             cn = randoms(rng, width, 0.2f), tapsW = randoms(rng, width * taps, 0.4f),
             hist = randoms(rng, history * width, 1.0f);
  auto plus = [](std::vector<float> v) { for (float &e : v) e += 1.0f; return v; };
  const auto knP = plus(kn), qnP = plus(qn), cnP = plus(cn);
  MetalBuffer gated = scratch(backend, rows * width * 4), conv = scratch(backend, rows * width * 2);
  MetalBuffer streamsBuffer = upload(backend, x), histIn = upload(backend, bf16s(hist)),
              histOut = scratch(backend, history * width * 2);
  CommandGraph graph;
  ops::Qwen4::addPleGate(graph, upload(backend, bf16s(key)), upload(backend, bf16s(value)), streamsBuffer,
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
  ops::Qwen4::addPleConvolution(graph2, conv, histIn, histOut, upload(backend, tapsW), gated, streamsBuffer, shape,
                                rows);
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

  // The history after a lane's retained rows (a count the GPU holds) and
  // after a sequence's first rows (one the host knows).
  for (uint32_t retained = 1; retained <= rows; retained += 3) {
    MetalBuffer committed = scratch(backend, history * width * 2), captured = scratch(backend, history * width * 2);
    CommandGraph graph3;
    ops::Qwen4::addPleCommit(graph3, conv, histIn, committed, upload(backend, std::vector<uint32_t>{retained}), shape);
    ops::Qwen4::addPleHistory(graph3, conv, histIn, captured, shape, retained);
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

// A QSA case: `lanes` lanes in one dispatch of each kernel, lane l with a
// context of tokens + 61 l tokens over a shuffled page table of its own
// pages, whose last `rows` are the step's rows, which attend sparsely past
// `top` complete blocks. The first `tied` blocks of each lane's tokens have
// one raw key, zero on the rotated dimensions, so that their pooled keys and
// every row's scores of them are equal. One lane's rows are scored 8 at a
// time; several lanes' at once.
struct QsaCase final {
  uint32_t tokens;
  uint32_t rows;
  uint32_t top;
  uint32_t tied;
  uint32_t lanes;
};

void qsaTest(MetalBackend &backend, QsaCase c) {
  std::mt19937 rng(13);
  constexpr uint32_t D = QWEN4_QSA_DIMENSION, H = QWEN4_QSA_HEADS, page = 32;
  struct Lane final {
    uint32_t tokens, first, pageBase, pages;
    std::vector<uint32_t> table; // the lane's page ids in the extent
    std::vector<float> raw;      // every token's raw key
  };
  const uint32_t rows = c.rows, top = c.top, lanes = c.lanes, laneRows = lanes * rows;
  std::vector<Lane> lane(lanes);
  uint32_t pages = 0, mostTokens = 0;
  const auto tiedKey = randoms(rng, D, 1.0f);
  for (uint32_t l = 0; l < lanes; ++l) {
    Lane &x = lane[l];
    x.tokens = c.tokens + 61 * l;
    x.first = x.tokens - rows;
    x.pageBase = pages;
    x.pages = (x.tokens + page - 1) / page;
    pages += x.pages;
    mostTokens = std::max(mostTokens, x.tokens);
    x.table.resize(x.pages);
    std::iota(x.table.begin(), x.table.end(), x.pageBase);
    std::shuffle(x.table.begin(), x.table.end(), rng);
    x.raw = randoms(rng, uint64_t{x.tokens} * D, 1.0f);
    for (uint32_t t = 0; t < 4 * c.tied; ++t)
      for (uint32_t d = 0; d < D; ++d) x.raw[uint64_t{t} * D + d] = d < 64 ? 0.0f : tiedKey[d];
  }
  const uint32_t words = (mostTokens / 4 + 2 + 31) / 32, stride = words * 32;
  // Earlier tokens' raw keys are already in the store.
  std::vector<uint16_t> store(uint64_t{pages} * page * D, 0);
  for (const Lane &x : lane)
    for (uint32_t t = 0; t < x.first; ++t)
      for (uint32_t d = 0; d < D; ++d)
        store[(uint64_t{x.table[t / page]} * page + t % page) * D + d] = bf16(x.raw[uint64_t{t} * D + d]);
  // One KV extent holding the layer's index regions: the raw keys of every
  // page, then their pooled keys; each lane's table holds its pages' entries.
  const uint64_t keyBytes = uint64_t{pages} * page * D * 2;
  MetalBuffer extent = backend.allocateBuffer(keyBytes + uint64_t{pages} * (page / 4) * D * 2,
                                              BufferStorage::Shared, "qwen4-test-extent",
                                              BufferReach::Addressed);
  check(!(extent.gpuAddress() & SPLASH_KV_PAGE_INDEX_MASK), "QSA extent leaves room for the page index");
  std::memcpy(extent.contents(), store.data(), keyBytes);
  MetalBuffer rawKeys = backend.view(extent, 0, keyBytes),
              pooled = backend.view(extent, keyBytes, extent.sizeBytes() - keyBytes);
  std::vector<MetalBuffer> pageTables;
  std::vector<uint32_t> firstPositions, counts;
  for (const Lane &x : lane) {
    std::vector<uint64_t> entries;
    for (uint32_t index : x.table) entries.push_back(extent.gpuAddress() | index);
    pageTables.push_back(upload(backend, entries));
    firstPositions.push_back(x.first);
  }
  const ops::QsaIndex qsaIndex{0, static_cast<uint32_t>(keyBytes)};
  const auto query = randoms(rng, uint64_t{laneRows} * H * D, 1.0f), qn = randoms(rng, D, 0.2f),
             kn = randoms(rng, D, 0.2f);
  std::vector<float> stepKeys(uint64_t{laneRows} * D);
  std::vector<float> qnP(D), knP(D), invFreq(32), cosT(laneRows * 32), sinT(laneRows * 32);
  for (uint32_t d = 0; d < D; ++d) qnP[d] = 1 + qn[d], knP[d] = 1 + kn[d];
  for (uint32_t i = 0; i < 32; ++i) invFreq[i] = std::pow(10'000'000.0f, -float(i) / 32);
  for (uint32_t l = 0; l < lanes; ++l)
    for (uint32_t r = 0; r < rows; ++r) {
      const uint32_t row = l * rows + r, position = lane[l].first + r;
      for (uint32_t d = 0; d < D; ++d) stepKeys[uint64_t{row} * D + d] = lane[l].raw[uint64_t{position} * D + d];
      for (uint32_t i = 0; i < 32; ++i) {
        cosT[row * 32 + i] = std::cos(float(position) * invFreq[i]);
        sinT[row * 32 + i] = std::sin(float(position) * invFreq[i]);
      }
    }
  MetalBuffer queries = scratch(backend, uint64_t{laneRows} * H * D * 2);
  const ops::QsaRows qsaRows{rows, firstPositions, pageTables, top, words, stride};
  // Every block whose last token is in a lane's step, pooled; earlier blocks
  // pooled by an earlier step, which this test runs too: each lane's blocks
  // `laneRecords` records apart.
  const uint32_t laneRecords = mostTokens / 4;
  std::vector<Qwen4QsaBlock> blocks(uint64_t{lanes} * laneRecords);
  for (uint32_t l = 0; l < lanes; ++l) {
    counts.push_back(lane[l].tokens / 4);
    for (uint32_t b = 0; b < counts.back(); ++b) blocks[uint64_t{l} * laneRecords + b] = {4 * b, {4 * b, 4 * b, 4 * b}};
  }
  CommandGraph graph;
  ops::Qwen4::addQsaProject(graph, upload(backend, query), upload(backend, stepKeys), {upload(backend, qnP), true},
                            upload(backend, cosT), upload(backend, sinT), qsaIndex, queries, qsaRows);
  ops::Qwen4::addQsaPool(graph, {upload(backend, blocks), counts, laneRecords, pageTables},
                         {upload(backend, knP), true}, upload(backend, invFreq), qsaIndex);
  const uint32_t scoreRows = lanes > 1 ? laneRows : 8;
  MetalBuffer scores = scratch(backend, uint64_t{scoreRows} * stride * 4),
              mask = scratch(backend, uint64_t{laneRows} * words * 4);
  ops::Qwen4::addQsaSelect(graph, queries, qsaIndex, scores, scoreRows, mask, qsaRows);
  run(backend, graph);

  // A rotation's angle is the fp32 product the RoPE tables take (the host's
  // above, the pool's own): at thousands of tokens its rounding moves a value
  // by more than a bf16 tolerance.
  auto rotate = [&](std::vector<double> v, uint32_t position) {
    for (uint32_t i = 0; i < 32; ++i) {
      const double a = float(position) * invFreq[i], c = std::cos(a), s = std::sin(a);
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
  for (const uint16_t value : download<uint16_t>(pooled, uint64_t{pages} * (page / 4) * D)) gotP.push_back(f32(value));
  const auto gotRaw = download<uint16_t>(rawKeys, uint64_t{pages} * page * D);
  const auto gotMask = download<uint32_t>(mask, uint64_t{laneRows} * words);
  bool okQ = true, okRaw = true, okP = true, okMask = true;
  for (uint32_t l = 0; l < lanes; ++l) {
    const Lane &x = lane[l];
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t h = 0; h < H; ++h) {
        const uint64_t head = (uint64_t{l} * rows + r) * H + h;
        std::vector<double> v(query.begin() + head * D, query.begin() + (head + 1) * D);
        const std::vector<double> want = rotate(rmsNorm(v, qnP), x.first + r);
        for (uint32_t d = 0; d < D; ++d) okQ &= nearBf16(gotQ[head * D + d], want[d]);
      }
    for (uint32_t t = x.first; t < x.tokens; ++t)
      for (uint32_t d = 0; d < D; ++d)
        okRaw &= gotRaw[(uint64_t{x.table[t / page]} * page + t % page) * D + d] == bf16(x.raw[uint64_t{t} * D + d]);
    const auto pooledKey = [&](uint32_t b) { return &gotP[(uint64_t{x.table[b / 8]} * 8 + b % 8) * D]; };
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
      // The tail block and nothing past it; below the budget, every block.
      for (uint32_t b = complete; b < words * 32; ++b) okMask &= got(b) == (b == complete);
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
  const std::string scope = std::to_string(lanes) + " lane(s) of " + std::to_string(c.tokens) + " tokens";
  check(okQ, "QSA project of " + scope + ": normalized, rotated query heads");
  check(okRaw, "QSA project of " + scope + ": raw keys at their pages' slots");
  check(okP, "QSA pool of " + scope + ": normalized mean keys rotated to the block start");
  check(okMask, "QSA select of " + scope + ": dense below the budget, top blocks and the tail block above it" +
                    (c.tied ? ", ties towards the earlier block" : ""));
}

// The selection refuses scores or bitmaps too narrow for its rows' history,
// past which its kernels would write: rows at 4092 on complete 1025 blocks,
// which take 1025 scores and, with the tail block, 1026 bits.
void qsaSelectRefusesNarrowScratch(MetalBackend &backend) {
  const MetalBuffer any = scratch(backend, 1 << 20);
  const std::array<uint32_t, 1> firstPositions{4092};
  const std::array<MetalBuffer, 1> pageTables{any};
  const auto refused = [&](uint32_t maskWords, uint32_t scoreStride) {
    CommandGraph graph;
    try {
      ops::Qwen4::addQsaSelect(graph, any, {0, 0}, any, 8, any,
                               {8, firstPositions, pageTables, 512, maskWords, scoreStride});
    } catch (const std::invalid_argument &error) {
      return std::string(error.what()) == "QSA scores or bitmaps narrower than the rows' history" && graph.empty();
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
  ops::Qwen4::addStreamMean(graph, upload(backend, values), captured, {firstRow, stride, column}, shape, rows);
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

// The PLE hash on the GPU against the CPU's (Qwen4Steps::hashRows), of
// three lanes of SPLASH_TARGET_VERIFY_ROWS rows each.
void pleHashTest(MetalBackend &backend) {
  const uint32_t lanes = 3, laneRows = SPLASH_TARGET_VERIFY_ROWS, ngram = 3, perNgram = 8, heads = 16, eos = 248044;
  Qwen4PleHashParams params{};
  params.rows = lanes * laneRows;
  params.ngram = ngram;
  params.heads_per_ngram = perNgram;
  params.eos = eos;
  const uint64_t multipliers[QWEN4_PLE_MAXIMUM_NGRAM] = {0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full,
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
  const std::vector<uint32_t> before = {40, 41, QWEN4_PLE_NONE, QWEN4_PLE_NONE, eos, 77};
  MetalBuffer indices = scratch(backend, lanes * laneRows * heads * 4);
  CommandGraph graph;
  ops::Qwen4::addPleHash(graph, upload(backend, tokens), upload(backend, before), indices, params, heads);
  run(backend, graph);
  const auto got = download<uint32_t>(indices, lanes * laneRows * heads);
  bool ok = true;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t r = 0; r < laneRows; ++r) {
      uint64_t context[QWEN4_PLE_MAXIMUM_NGRAM] = {tokens[lane * laneRows + r]};
      bool cut = false;
      for (uint32_t back = 1; back < ngram; ++back) {
        const uint32_t previous =
            back <= r ? tokens[lane * laneRows + r - back] : before[lane * (ngram - 1) + back - r - 1];
        cut = cut || previous == QWEN4_PLE_NONE || previous == eos;
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
        qsaTest(backend, {150, 20, 16, 0, 1});
        // A threshold inside a run of equal scores: fewer other blocks than
        // the budget.
        qsaTest(backend, {4100, 20, 600, 700, 1});
        // A verify step's lanes, sparse and causal rows among them.
        qsaTest(backend, {2050, 8, 512, 0, 4});
        qsaTest(backend, {700, 8, 160, 120, 3});
        // Past 32 768 blocks, where each select thread owns two bitmap words.
        qsaTest(backend, {140000, 8, 512, 0, 1});
        qsaSelectRefusesNarrowScratch(backend);
        streamMeanTest(backend);
        pleHashTest(backend);
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
