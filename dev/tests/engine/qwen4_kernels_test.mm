// Qwen3.8-Flash-Next's own kernels (kernels/shared/qwen4.metal and
// qwen4_qsa.metal) through ops::Qwen4, against fp64 CPU references of
// llama.cpp's qwen4exp graph:
// - the hyper-connection mix: per-stream RMS norm, injection weights, the
//   low-rank gate and the stream average; the combine and the streams' start;
// - the PLE gate, its dilated convolution over a history, the history after
//   a sequence and after a verify step's retained rows;
// - QSA: the indexer's normalized, rotated queries and raw keys at their KV
//   page slots, pooled keys, and each row's block bitmap (dense below the
//   budget; above it the top blocks, ties towards the earlier block, and
//   the row's own block);
// - the MTP head's inputs (stream-major [enorm(e) ; hnorm_s(h_s)] with
//   shifted streams and repeated rows), the eh projection's widening, the
//   argmax proposals, the carried row; the GPU PLE hash against the CPU's.
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/ops/Qwen4.hpp"
#include "metal/abi/KvExtent.h"
#include "metal/abi/Qwen4.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

void hyperTest(MetalBackend &backend) {
  std::mt19937 rng(7);
  const uint32_t rows = 5, hidden = 512, streams = 4, rank = 64, width = hidden * streams;
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
      for (uint32_t d = 0; d < hidden; ++d) sum += double(x[r * width + s * hidden + d]) * x[r * width + s * hidden + d];
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
  for (uint32_t i = 0; i < rows * rank; ++i) okLow &= closeBf16(f32(gotLow[i]), silu(double(f32(bf16(low[i]))) / streams));
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
        const double want = x[r * width + s * hidden + d] + double(f32(bf16(branch[r * hidden + d]))) * gotW[r * streams + s];
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
      for (uint32_t d = 0; d < hidden; ++d) okI &= gotInit[r * width + s * hidden + d] == f32(bf16(branch[r * hidden + d]));
  check(okI, "hyper init: every stream the embedding");
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
                         {upload(backend, knP), true}, {upload(backend, qnP), true}, {upload(backend, cnP), true}, gated,
                         conv, shape, rows);
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
  ops::Qwen4::addPleConvolution(graph2, conv, histIn, histOut, upload(backend, tapsW), gated, streamsBuffer, shape, rows);
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
      for (uint32_t k = 0; k < taps; ++k) sum += tapsW[c * taps + k] * input(int(r) - int((taps - 1 - k) * dilation), c);
      const double want = x[r * width + c] + gotG[r * width + c] + silu(sum);
      okX &= std::fabs(gotX[r * width + c] - want) <= 1e-4 * (1 + std::fabs(want));
    }
  for (uint32_t j = 0; j < history; ++j)
    for (uint32_t c = 0; c < width; ++c) okH &= f32(gotH[j * width + c]) == input(int(rows) - int(history) + int(j), c);
  check(okX, "PLE convolution: streams + gated + silu(dilated taps over the history)");
  check(okH, "PLE convolution: the history after the sequence");

  for (uint32_t retained = 1; retained <= rows; retained += 3) {
    MetalBuffer committed = scratch(backend, history * width * 2);
    CommandGraph graph3;
    ops::Qwen4::addPleCommit(graph3, conv, histIn, committed, upload(backend, std::vector<uint32_t>{retained}), shape);
    run(backend, graph3);
    const auto got = download<uint16_t>(committed, history * width);
    bool ok = true;
    for (uint32_t j = 0; j < history; ++j)
      for (uint32_t c = 0; c < width; ++c) ok &= f32(got[j * width + c]) == input(int(retained) - int(history) + int(j), c);
    check(ok, "PLE commit: the history after " + std::to_string(retained) + " retained rows");
  }
}

void qsaTest(MetalBackend &backend) {
  std::mt19937 rng(13);
  constexpr uint32_t D = QWEN4_QSA_DIMENSION, H = QWEN4_QSA_HEADS, page = 32;
  const uint32_t top = 16, words = 8, stride = words * 32;
  // Context of 150 tokens over 5 pages in a shuffled page table; the step's
  // rows are its last 20 tokens.
  const uint32_t tokens = 150, rows = 20, first = tokens - rows, pages = 5;
  std::vector<uint32_t> table = {3, 0, 4, 1, 2};
  const auto rawAll = randoms(rng, tokens * D, 1.0f);
  // Earlier tokens' raw keys are already in the store.
  std::vector<uint16_t> store(pages * page * D, 0);
  for (uint32_t t = 0; t < first; ++t)
    for (uint32_t d = 0; d < D; ++d) store[(table[t / page] * page + t % page) * D + d] = bf16(rawAll[t * D + d]);
  // One KV extent holding the layer's index regions: the raw keys of every
  // page, then their pooled keys; the table holds the pages' entries.
  const uint64_t keyBytes = uint64_t{pages} * page * D * 2;
  MetalBuffer extent = backend.allocateBuffer(keyBytes + uint64_t{pages} * (page / 4) * D * 4,
                                              BufferStorage::Shared, "qwen4-test-extent");
  check(!(extent.gpuAddress() & SPLASH_KV_PAGE_INDEX_MASK), "QSA extent leaves room for the page index");
  std::memcpy(extent.contents(), store.data(), keyBytes);
  MetalBuffer rawKeys = backend.view(extent, 0, keyBytes),
              pooled = backend.view(extent, keyBytes, extent.sizeBytes() - keyBytes);
  std::vector<uint64_t> entries;
  for (uint32_t index : table) entries.push_back(extent.gpuAddress() | index);
  MetalBuffer pageTable = upload(backend, entries);
  const ops::QsaIndex qsaIndex{0, static_cast<uint32_t>(keyBytes)};
  const auto query = randoms(rng, rows * H * D, 1.0f), qn = randoms(rng, D, 0.2f), kn = randoms(rng, D, 0.2f);
  std::vector<float> stepKeys(rows * D);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t d = 0; d < D; ++d) stepKeys[r * D + d] = rawAll[(first + r) * D + d];
  std::vector<float> qnP(D), knP(D), invFreq(32), cosT(rows * 32), sinT(rows * 32);
  for (uint32_t d = 0; d < D; ++d) qnP[d] = 1 + qn[d], knP[d] = 1 + kn[d];
  for (uint32_t i = 0; i < 32; ++i) invFreq[i] = std::pow(10'000'000.0f, -float(i) / 32);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t i = 0; i < 32; ++i) {
      cosT[r * 32 + i] = std::cos(float(first + r) * invFreq[i]);
      sinT[r * 32 + i] = std::sin(float(first + r) * invFreq[i]);
    }
  MetalBuffer queries = scratch(backend, rows * H * D * 4);
  const ops::QsaRows qsaRows{rows, first, top, words, stride};
  // Every block whose last token is in the step, pooled; earlier blocks
  // pooled by an earlier step, which this test runs too.
  std::vector<Qwen4QsaBlock> blocks;
  for (uint32_t last = 3; last < tokens; last += 4) blocks.push_back({last - 3, {last - 3, last - 3, last - 3}});
  CommandGraph graph;
  ops::Qwen4::addQsaProject(graph, upload(backend, query), upload(backend, stepKeys), {upload(backend, qnP), true},
                            upload(backend, cosT), upload(backend, sinT), pageTable, qsaIndex, queries, qsaRows);
  ops::Qwen4::addQsaPool(graph, upload(backend, blocks), uint32_t(blocks.size()), {upload(backend, knP), true},
                         upload(backend, invFreq), pageTable, qsaIndex);
  MetalBuffer scores = scratch(backend, 8 * stride * 4), mask = scratch(backend, rows * words * 4);
  ops::Qwen4::addQsaSelect(graph, queries, qsaIndex, pageTable, scores, 8, mask, qsaRows);
  run(backend, graph);

  auto rotate = [&](std::vector<double> v, double position) {
    for (uint32_t i = 0; i < 32; ++i) {
      const double a = position * double(float(invFreq[i])), c = std::cos(a), s = std::sin(a);
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
  const auto gotQ = download<float>(queries, rows * H * D);
  bool okQ = true;
  std::vector<std::vector<double>> qRef(rows * H);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t h = 0; h < H; ++h) {
      std::vector<double> v(query.begin() + (r * H + h) * D, query.begin() + (r * H + h + 1) * D);
      qRef[r * H + h] = rotate(rmsNorm(v, qnP), first + r);
      for (uint32_t d = 0; d < D; ++d) okQ &= std::fabs(gotQ[(r * H + h) * D + d] - qRef[r * H + h][d]) <= 2e-4;
    }
  check(okQ, "QSA project: normalized, rotated query heads");
  const auto gotRaw = download<uint16_t>(rawKeys, pages * page * D);
  bool okRaw = true;
  for (uint32_t t = first; t < tokens; ++t)
    for (uint32_t d = 0; d < D; ++d) okRaw &= gotRaw[(table[t / page] * page + t % page) * D + d] == bf16(rawAll[t * D + d]);
  check(okRaw, "QSA project: raw keys at their pages' slots");
  const auto gotP = download<float>(pooled, pages * (page / 4) * D);
  bool okP = true;
  std::vector<std::vector<double>> kRef(tokens / 4);
  for (uint32_t b = 0; b < tokens / 4; ++b) {
    std::vector<double> mean(D, 0);
    for (uint32_t t = 4 * b; t < 4 * b + 4; ++t)
      for (uint32_t d = 0; d < D; ++d) mean[d] += f32(bf16(rawAll[t * D + d])) / 4;
    kRef[b] = rotate(rmsNorm(mean, knP), 4 * b);
    for (uint32_t d = 0; d < D; ++d)
      okP &= std::fabs(gotP[(table[b / 8] * 8 + b % 8) * D + d] - kRef[b][d]) <= 2e-4;
  }
  check(okP, "QSA pool: normalized mean keys rotated to the block start");
  const auto gotMask = download<uint32_t>(mask, rows * words);
  bool okMask = true;
  for (uint32_t r = 0; r < rows; ++r) {
    const uint32_t position = first + r, complete = (position + 1) / 4;
    std::vector<bool> want(words * 32, false);
    if (complete <= top) {
      for (uint32_t b = 0; b <= complete; ++b) want[b] = true;
    } else {
      std::vector<std::pair<double, uint32_t>> scored;
      for (uint32_t b = 0; b < complete; ++b) {
        double s = 0;
        for (uint32_t h = 0; h < H; ++h) {
          double dot = 0;
          for (uint32_t d = 0; d < D; ++d) dot += double(gotQ[(r * H + h) * D + d]) * gotP[(table[b / 8] * 8 + b % 8) * D + d];
          s += std::max(dot, 0.0);
        }
        scored.push_back({s / std::sqrt(double(D)), b});
      }
      std::stable_sort(scored.begin(), scored.end(), [](auto a, auto b) { return a.first > b.first; });
      for (uint32_t i = 0; i < top; ++i) want[scored[i].second] = true;
      want[complete] = true;
    }
    for (uint32_t b = 0; b < words * 32; ++b) okMask &= bool((gotMask[r * words + b / 32] >> (b % 32)) & 1) == want[b];
  }
  check(okMask, "QSA select: dense below the budget, top blocks and the row's block above it");
}

void mtpTest(MetalBackend &backend) {
  std::mt19937 rng(11);
  const uint32_t rows = 8, hidden = 256, streams = 4, width = hidden * streams, total = 16, offset = 8, limit = 3;
  const ops::HyperShape shape{hidden, streams, 0};
  const auto embedding = randoms(rng, rows * hidden, 1.0f);
  const auto carried = randoms(rng, width, 2.0f);
  const auto history = randoms(rng, rows * width, 2.0f);
  const auto enorm = randoms(rng, hidden, 1.0f);
  const auto hnorm = randoms(rng, width, 1.0f);
  const auto e16 = bf16s(embedding);
  MetalBuffer output = scratch(backend, uint64_t{streams} * total * 2 * hidden * 2);
  std::memset(output.contents(), 0, output.sizeBytes());
  CommandGraph graph;
  ops::Qwen4::addMtpInput(graph, upload(backend, e16), upload(backend, carried), upload(backend, history),
                          NormWeights{upload(backend, enorm), true}, NormWeights{upload(backend, hnorm), true}, output,
                          shape, rows, limit, offset, total);
  run(backend, graph);
  const auto got = download<uint16_t>(output, uint64_t{streams} * total * 2 * hidden);
  bool ok = true;
  for (uint32_t r = 0; r < rows; ++r) {
    const uint32_t source = std::min(r, limit - 1);
    double es = 0;
    for (uint32_t d = 0; d < hidden; ++d) es += double(f32(e16[source * hidden + d])) * f32(e16[source * hidden + d]);
    const double escale = 1.0 / std::sqrt(es / hidden + 1e-6);
    const float *h = source == 0 ? carried.data() : history.data() + (source - 1) * width;
    for (uint32_t s = 0; s < streams; ++s) {
      double hs = 0;
      for (uint32_t d = 0; d < hidden; ++d) hs += double(h[s * hidden + d]) * h[s * hidden + d];
      const double hscale = 1.0 / std::sqrt(hs / hidden + 1e-6);
      const uint16_t *o = got.data() + (uint64_t{s} * total + offset + r) * 2 * hidden;
      for (uint32_t d = 0; d < hidden; ++d) {
        ok &= closeBf16(f32(o[d]), f32(e16[source * hidden + d]) * escale * enorm[d]);
        ok &= closeBf16(f32(o[hidden + d]), h[s * hidden + d] * hscale * hnorm[s * hidden + d]);
      }
    }
  }
  // Rows of other lanes stay untouched.
  for (uint32_t s = 0; s < streams; ++s)
    for (uint32_t i = 0; i < offset * 2 * hidden; ++i) ok &= got[uint64_t{s} * total * 2 * hidden + i] == 0;
  check(ok, "MTP input: stream-major norms of the embedding and the shifted streams");

  // Widening: stream-major bf16 [streams][rows][hidden] to [rows][streams][hidden].
  const auto projected = bf16s(randoms(rng, rows * width, 1.0f));
  MetalBuffer widened = scratch(backend, rows * width * 4);
  CommandGraph widen;
  ops::Qwen4::addMtpWiden(widen, upload(backend, projected), widened, shape, rows);
  run(backend, widen);
  const auto w = download<float>(widened, rows * width);
  bool okWiden = true;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t s = 0; s < streams; ++s)
      for (uint32_t d = 0; d < hidden; ++d)
        okWiden &= w[(r * streams + s) * hidden + d] == f32(projected[(uint64_t{s} * rows + r) * hidden + d]);
  check(okWiden, "MTP widen: stream-major rows to fp32 streams");

  // Draft tokens, argmax proposals and the carried row.
  const uint32_t lanes = 2, laneRows = 8, proposals = 7, candidates = 16, vocabulary = 70001;
  std::vector<uint32_t> anchors(lanes * laneRows, 0), proposed(lanes * proposals, 0);
  anchors[0] = 11;
  anchors[laneRows] = 22;
  for (uint32_t i = 0; i < lanes * proposals; ++i) proposed[i] = 100 + i;
  MetalBuffer anchorBuffer = upload(backend, anchors), proposedBuffer = upload(backend, proposed),
              tokens = scratch(backend, lanes * laneRows * 4);
  CommandGraph tokenGraph;
  ops::Qwen4::addMtpTokens(tokenGraph, anchorBuffer, proposedBuffer, tokens, lanes, laneRows, proposals, 3);
  run(backend, tokenGraph);
  const auto t = download<uint32_t>(tokens, lanes * laneRows);
  bool okTokens = true;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t r = 0; r < laneRows; ++r) {
      const uint32_t row = std::min(r, 2u);
      okTokens &= t[lane * laneRows + r] == (row == 0 ? anchors[lane * laneRows] : proposed[lane * proposals + row - 1]);
    }
  check(okTokens, "MTP tokens: anchor, proposals so far, then repeats");

  auto logits = randoms(rng, uint64_t{lanes} * laneRows * vocabulary, 1.0f);
  logits[(0 * laneRows + 4) * uint64_t{vocabulary} + 69999] = 50.0f;
  logits[(1 * laneRows + 4) * uint64_t{vocabulary} + 5] = 50.0f;
  logits[(1 * laneRows + 4) * uint64_t{vocabulary} + 3] = 50.0f; // tie: the lower id
  std::vector<uint32_t> cand(lanes * proposals * candidates, 7);
  MetalBuffer candidateBuffer = upload(backend, cand);
  CommandGraph argmax;
  ops::Qwen4::addMtpArgmax(argmax, upload(backend, logits), proposedBuffer, candidateBuffer, lanes, laneRows,
                           vocabulary, proposals, candidates, 4, true);
  run(backend, argmax);
  const auto p = download<uint32_t>(proposedBuffer, lanes * proposals);
  const auto c = download<uint32_t>(candidateBuffer, lanes * proposals * candidates);
  bool okArgmax = true;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t i = 0; i < proposals; ++i) {
      const uint32_t want = i < 4 ? 100 + lane * proposals + i : (lane ? 3 : 69999);
      okArgmax &= p[lane * proposals + i] == want;
      okArgmax &= c[(lane * proposals + i) * candidates] == (i < 4 ? 7 : want);
      okArgmax &= c[(lane * proposals + i) * candidates + 1] == 7;
    }
  check(okArgmax, "MTP argmax: the row's argmax as its proposal and every later one");

  const auto streamRows = randoms(rng, laneRows * width, 1.0f);
  MetalBuffer carriedOut = scratch(backend, width * 4), carriedFixed = scratch(backend, width * 4);
  MetalBuffer streamBuffer = upload(backend, streamRows);
  CommandGraph carry;
  ops::Qwen4::addMtpCarry(carry, streamBuffer, carriedOut, upload(backend, std::vector<uint32_t>{5}), shape, laneRows);
  ops::Qwen4::addMtpCarry(carry, streamBuffer, carriedFixed, {}, shape, 3);
  run(backend, carry);
  const auto co = download<float>(carriedOut, width), cf = download<float>(carriedFixed, width);
  bool okCarry = true;
  for (uint32_t i = 0; i < width; ++i) {
    okCarry &= co[i] == streamRows[4 * width + i];
    okCarry &= cf[i] == streamRows[2 * width + i];
  }
  check(okCarry, "MTP carry: the last retained row, or the last row");
}

// The PLE hash on the GPU against Runtime.mm's pleRows.
void pleHashTest(MetalBackend &backend) {
  const uint32_t lanes = 3, laneRows = 8, ngram = 3, perNgram = 8, heads = 16, eos = 248044;
  Qwen4PleHashParams params{};
  params.rows = lanes * laneRows;
  params.lane_rows = laneRows;
  params.ngram = ngram;
  params.heads_per_ngram = perNgram;
  params.eos = eos;
  const uint64_t multipliers[4] = {0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full, 0x165667B19E3779F9ull,
                                   0x27D4EB2F165667C5ull};
  for (uint32_t i = 0; i < 4; ++i) params.multipliers[i] = multipliers[i];
  for (uint32_t h = 0; h < heads; ++h) {
    params.vocabularies[h] = 19999999 + h * 7919;
    params.offsets[h] = h * 20000000;
  }
  std::vector<uint32_t> tokens = {5, 6, eos, 7, 8, 9, 10, 11,      1, 2, 3, 4, 5, 6, 7, 8,
                                  248319, 0, 12, 13, eos, eos, 14, 15};
  // Lane 0: two tokens before; lane 1: a sequence start; lane 2: EOS just before.
  const std::vector<uint32_t> before = {40, 41, QWEN4_PLE_NONE, QWEN4_PLE_NONE, eos, 77};
  MetalBuffer indices = scratch(backend, lanes * laneRows * heads * 4);
  CommandGraph graph;
  ops::Qwen4::addPleHash(graph, upload(backend, tokens), upload(backend, before), indices, params, heads);
  run(backend, graph);
  const auto got = download<uint32_t>(indices, lanes * laneRows * heads);
  bool ok = true;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t r = 0; r < laneRows; ++r) {
      uint64_t context[4] = {tokens[lane * laneRows + r], eos, eos, eos};
      bool cut = false;
      for (uint32_t back = 1; back < ngram; ++back) {
        const uint32_t previous = back <= r ? tokens[lane * laneRows + r - back] : before[lane * 2 + back - r - 1];
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
      if (argc != 2) throw std::runtime_error("usage: qwen4-kernels METALLIB");
      MetalBackend backend(argv[1]);
      hyperTest(backend);
      pleTest(backend);
      qsaTest(backend);
      mtpTest(backend);
      pleHashTest(backend);
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
