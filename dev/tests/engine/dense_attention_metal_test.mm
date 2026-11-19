// ops::DenseAttention (kernels/shared/dense_attention.metal) against an fp64
// reference, row by row, in the vision tower's layout: per-head planes of
// heads of 72 (queries and keys padded to 80), every patch attending the
// whole image, the output in per-head planes or, as the tower writes it, in
// rows of every head. Token counts fall below,
// on and past the 32-row query tile and 128-key tile, up to 4,100 (33 key
// tiles, the last of four keys), and each runs three kinds of input: random
// queries, keys and values; keys whose scores rise tile by tile, so every
// row rescales its running output at every tile;
// and scores and values over a wide dynamic range, whose probabilities
// underflow. Each tensor is bound at the extent the operator requires and
// followed by a key tile of bf16 NaN: shader validation does not see MPP
// loads and stores past a binding, but a NaN read reaches its row and a
// write past the output changes its tail. Keys and values past the rows hold
// large finite values, which the masks must hide.
//
//   dense-attention METALLIB
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/DenseAttention.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::DenseAttention;
using splash::ops::DenseAttentionTensors;
using splash::ops::HeadTensor;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using splash::test::require;

constexpr uint32_t kHeads = 16;
constexpr uint32_t kKeyTile = 128;
constexpr uint16_t kNan = 0x7FC0;

enum class Kind { Random, Rising, Wide };

const char *kindName(Kind kind) {
  switch (kind) {
  case Kind::Random:
    return "random";
  case Kind::Rising:
    return "rising";
  case Kind::Wide:
    return "wide";
  }
  return "";
}

// `elements` bf16 values, the extent the kernel is bound to, in an
// allocation that follows them with `tail` values of NaN.
struct Tensor final {
  uint64_t elements = 0;
  uint64_t tail = 0;
  MetalBuffer allocation;
  MetalBuffer view;

  Tensor(MetalBackend &backend, uint64_t count, uint64_t tailCount)
      : elements(count), tail(tailCount), allocation(splash::test::sharedBuffer(backend, (elements + tail) * 2)),
        view(backend.view(allocation, 0, elements * 2)) {
    std::fill_n(data(), elements + tail, kNan);
  }
  [[nodiscard]] uint16_t *data() const { return static_cast<uint16_t *>(allocation.contents()); }
  [[nodiscard]] bool tailIntact() const {
    return std::all_of(data() + elements, data() + elements + tail, [](uint16_t value) { return value == kNan; });
  }
};

// One case: heads of `dimensions` in per-head planes, the output in planes
// too or in rows of every head (the context rows the tower's projection
// reads).
struct Case final {
  const char *name;
  uint32_t dimensions;
  bool outputRows;
};

uint32_t paddedDimensions(const Case &c) { return (c.dimensions + 15) / 16 * 16; }

// Where a tensor's element (head, row, dimension) lies.
uint64_t at(const HeadTensor &tensor, uint32_t head, uint32_t row, uint32_t dimension) {
  return uint64_t{head} * tensor.headStride + uint64_t{row} * tensor.rowStride + dimension;
}

// The tensors of one dispatch and the host addresses of their elements.
struct Inputs final {
  DenseAttentionTensors tensors;
  uint16_t *queries = nullptr, *keys = nullptr, *values = nullptr, *output = nullptr;
};

struct HeadCheck final {
  // The largest error over its bound, and where.
  double worst = 0.0;
  uint32_t worstRow = 0;
  // Rows and key tiles past the first whose maximum did not rise.
  uint64_t steadyTiles = 0;
  std::string failure;
};

// One head's output rows against the fp64 attention of its bf16 inputs with
// the kernel's scale. An output may differ by `relative` of W + |out|, W the
// row's probability-weighted mean of |v| in that dimension: bf16
// probabilities in PV and the bf16 output each round by up to 2^-8; fp32
// scores (up to 80 products, the scale and the maximum subtracted) change
// each probability by up to 2 epsilon; and fp32 sums over the keys and the
// rescales add a few units of 2^-24 per key.
HeadCheck checkHead(const Case &c, Kind kind, uint32_t rows, float scale, uint32_t head, const Inputs &in) {
  HeadCheck check;
  const uint32_t qk = paddedDimensions(c), d = c.dimensions;
  const DenseAttentionTensors &t = in.tensors;
  const auto q = [&](uint32_t row, uint32_t i) { return double(bf16ToFloat(in.queries[at(t.queries, head, row, i)])); };
  const auto k = [&](uint32_t row, uint32_t i) { return double(bf16ToFloat(in.keys[at(t.keys, head, row, i)])); };
  const auto v = [&](uint32_t row, uint32_t i) { return double(bf16ToFloat(in.values[at(t.values, head, row, i)])); };
  double largestKey = 0.0;
  for (uint32_t key = 0; key < rows; ++key) {
    double squares = 0.0;
    for (uint32_t i = 0; i < qk; ++i) squares += k(key, i) * k(key, i);
    largestKey = std::max(largestKey, std::sqrt(squares));
  }
  std::vector<double> logits(rows), sums(d), magnitudes(d);
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t first = 0, end = rows;
    double queryNorm = 0.0;
    for (uint32_t i = 0; i < qk; ++i) queryNorm += q(row, i) * q(row, i);
    queryNorm = std::sqrt(queryNorm);
    double maximum = -INFINITY, largestLogit = 0.0;
    for (uint32_t key = first; key < end; ++key) {
      double partial[8] = {};
      for (uint32_t i = 0; i < qk; i += 8)
        for (uint32_t lane = 0; lane < 8; ++lane) partial[lane] += q(row, i + lane) * k(key, i + lane);
      const double dot = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                         ((partial[4] + partial[5]) + (partial[6] + partial[7]));
      logits[key] = dot * scale;
      maximum = std::max(maximum, logits[key]);
      largestLogit = std::max(largestLogit, std::fabs(logits[key]));
    }
    if (kind == Kind::Rising) {
      double previous = -INFINITY;
      for (uint32_t start = 0; start < rows; start += kKeyTile) {
        const double tile =
            *std::max_element(logits.begin() + start, logits.begin() + std::min(rows, start + kKeyTile));
        if (start && !(tile > previous + 0.05)) ++check.steadyTiles;
        previous = std::max(previous, tile);
      }
    }

    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(magnitudes.begin(), magnitudes.end(), 0.0);
    double total = 0.0;
    for (uint32_t key = first; key < end; ++key) {
      const double probability = std::exp(logits[key] - maximum);
      total += probability;
      for (uint32_t i = 0; i < d; ++i) {
        sums[i] += probability * v(key, i);
        magnitudes[i] += probability * std::fabs(v(key, i));
      }
    }
    const double epsilon = scale * qk * 0x1p-24 * queryNorm * largestKey + 0x1p-22 * largestLogit;
    const double relative = 0x1p-8 + 2.5 * epsilon + (2.0 * rows + 512.0) * 0x1p-24 + 0x1p-17;
    for (uint32_t i = 0; i < d; ++i) {
      const double expected = sums[i] / total;
      const double bound = relative * (magnitudes[i] / total + std::fabs(expected));
      const double observed = bf16ToFloat(in.output[at(t.output, head, row, i)]);
      const double error = std::fabs(observed - expected);
      if (!(error <= bound)) {
        check.failure = "row " + std::to_string(row) + " dimension " + std::to_string(i) + ": " +
                        std::to_string(observed) + " vs " + std::to_string(expected) + " (bound " +
                        std::to_string(bound) + ")";
        return check;
      }
      if (error / bound > check.worst) {
        check.worst = error / bound;
        check.worstRow = row;
      }
    }
  }
  return check;
}

// The inputs of `rows` rows. Rows past them hold large finite values;
// queries and keys end in zeros past the head's dimensions. Rising keys
// carry their tile's offset in their first dimension, which a query's first
// dimension of one turns into scores that rise by 0.7 to 1.2 from tile to
// tile, more than the spread of the small rest. Wide inputs scale each
// query row, key and value by its own power of two.
void fill(const Case &c, Kind kind, uint32_t rows, uint32_t padded, const Inputs &in) {
  std::mt19937 random(0x7a11 + rows * 3 + static_cast<uint32_t>(kind) + c.dimensions);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::uniform_int_distribution<int> step(6, 10), queryExponent(-3, 3), keyExponent(-4, 3), valueExponent(-8, 8);
  std::vector<float> offset(padded / kKeyTile, 0.0F);
  for (size_t tile = 1; tile < offset.size(); ++tile) offset[tile] = offset[tile - 1] + static_cast<float>(step(random));
  const uint32_t qk = paddedDimensions(c);
  for (uint32_t head = 0; head < kHeads; ++head) {
    for (uint32_t token = 0; token < padded; ++token) {
      uint16_t *query = in.queries + at(in.tensors.queries, head, token, 0);
      uint16_t *key = in.keys + at(in.tensors.keys, head, token, 0);
      uint16_t *value = in.values + at(in.tensors.values, head, token, 0);
      if (token >= rows) {
        std::fill_n(query, qk, floatToBf16(16.0F));
        std::fill_n(key, qk, floatToBf16(16.0F));
        std::fill_n(value, c.dimensions, floatToBf16(1024.0F));
        continue;
      }
      float queryScale = 1.4F, keyScale = 1.4F, valueScale = 1.0F;
      if (kind == Kind::Rising) queryScale = keyScale = 0.25F;
      if (kind == Kind::Wide) {
        queryScale = std::ldexp(1.0F, queryExponent(random));
        keyScale = std::ldexp(1.0F, keyExponent(random));
        valueScale = std::ldexp(1.0F, valueExponent(random));
      }
      std::fill_n(query, qk, uint16_t{0});
      std::fill_n(key, qk, uint16_t{0});
      for (uint32_t i = 0; i < c.dimensions; ++i) {
        query[i] = floatToBf16(normal(random) * queryScale);
        key[i] = floatToBf16(normal(random) * keyScale);
        value[i] = floatToBf16(normal(random) * valueScale);
      }
      if (kind == Kind::Rising) {
        query[0] = floatToBf16(1.0F);
        key[0] = floatToBf16(offset[token / kKeyTile]);
      }
    }
  }
}

// One sequence of `rows` rows: the operator's dispatch over tensors at their
// extents, checked head by head. Queries, keys and values each their own
// tensor of every head's padded rows; the output such a tensor too, or rows
// of every head.
void runCase(MetalBackend &backend, const Case &c, Kind kind, uint32_t rows) {
  const uint32_t padded = DenseAttention::paddedKeys(rows);
  const uint32_t d = c.dimensions, qk = paddedDimensions(c);
  const uint64_t tail = uint64_t{kKeyTile} * kHeads * qk;
  std::vector<Tensor> buffers;
  Inputs in;
  buffers.emplace_back(backend, uint64_t{kHeads} * padded * qk, tail);
  buffers.emplace_back(backend, uint64_t{kHeads} * padded * qk, tail);
  buffers.emplace_back(backend, uint64_t{kHeads} * padded * d, tail);
  if (c.outputRows)
    buffers.emplace_back(backend, uint64_t{rows} * kHeads * d, tail);
  else
    buffers.emplace_back(backend, uint64_t{kHeads} * padded * d, tail);
  in.tensors = {{buffers[0].view, padded * qk, qk},
                {buffers[1].view, padded * qk, qk},
                {buffers[2].view, padded * d, d},
                c.outputRows ? HeadTensor{buffers[3].view, d, kHeads * d} : HeadTensor{buffers[3].view, padded * d, d}};
  in.queries = buffers[0].data();
  in.keys = buffers[1].data();
  in.values = buffers[2].data();
  in.output = buffers[3].data();
  fill(c, kind, rows, padded, in);
  const float scale = 1.0F / std::sqrt(static_cast<float>(d));
  CommandGraph graph;
  DenseAttention::add(graph, in.tensors, kHeads, d, rows, scale);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  const std::string label = std::string(c.name) + " " + kindName(kind) + " rows " + std::to_string(rows);
  for (const Tensor &tensor : buffers)
    require(tensor.tailIntact(), label + ": a write reached past a tensor's extent");
  std::vector<std::future<HeadCheck>> checks;
  for (uint32_t head = 0; head < kHeads; ++head)
    checks.push_back(std::async(std::launch::async, checkHead, std::cref(c), kind, rows, scale, head, std::cref(in)));
  double worst = 0.0;
  uint32_t worstHead = 0, worstRow = 0;
  uint64_t steadyTiles = 0;
  for (uint32_t head = 0; head < kHeads; ++head) {
    const HeadCheck check = checks[head].get();
    require(check.failure.empty(), label + " head " + std::to_string(head) + " " + check.failure);
    steadyTiles += check.steadyTiles;
    if (check.worst >= worst) {
      worst = check.worst;
      worstHead = head;
      worstRow = check.worstRow;
    }
  }
  require(kind != Kind::Rising || !steadyTiles,
          label + ": " + std::to_string(steadyTiles) + " row tiles of rising keys kept their maximum");
  std::printf("dense attention %s: worst error %.3f of its bound (head %u row %u)\n", label.c_str(), worst,
              worstHead, worstRow);
}

// The operator's checks of what it reads: each tensor at its extent, the
// output in rows of every head, refused one element short.
void requireExtents(MetalBackend &backend) {
  constexpr uint32_t rows = 45, d = 72, qk = 80;
  const uint32_t padded = DenseAttention::paddedKeys(rows);
  const MetalBuffer q = splash::test::sharedBuffer(backend, uint64_t{kHeads} * padded * qk * 2);
  const MetalBuffer k = splash::test::sharedBuffer(backend, uint64_t{kHeads} * padded * qk * 2);
  const MetalBuffer v = splash::test::sharedBuffer(backend, uint64_t{kHeads} * padded * d * 2);
  const MetalBuffer out = splash::test::sharedBuffer(backend, uint64_t{rows} * kHeads * d * 2);
  const auto encode = [&](CommandGraph &graph, const MetalBuffer &queries, const MetalBuffer &keys,
                          const MetalBuffer &values, const MetalBuffer &output) {
    DenseAttention::add(graph,
                        {{queries, padded * qk, qk}, {keys, padded * qk, qk}, {values, padded * d, d},
                         {output, d, kHeads * d}},
                        kHeads, d, rows, 0.125F);
  };
  const auto plane = [&](uint32_t extentRows, uint32_t width) {
    return ((uint64_t{kHeads} - 1) * padded * width + (uint64_t{extentRows} - 1) * width + width) * 2;
  };
  using splash::test::requireExtent;
  requireExtent(backend, q, plane(DenseAttention::paddedQueries(rows), qk), 2, "dense attention query",
                [&](CommandGraph &g, const MetalBuffer &view) { encode(g, view, k, v, out); });
  requireExtent(backend, k, plane(padded, qk), 2, "dense attention key",
                [&](CommandGraph &g, const MetalBuffer &view) { encode(g, q, view, v, out); });
  requireExtent(backend, v, plane(padded, d), 2, "dense attention value",
                [&](CommandGraph &g, const MetalBuffer &view) { encode(g, q, k, view, out); });
  requireExtent(backend, out, out.sizeBytes(), 2, "dense attention output",
                [&](CommandGraph &g, const MetalBuffer &view) { encode(g, q, k, v, view); });
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "usage: dense-attention METALLIB");
      MetalBackend backend(argv[1]);
      requireExtents(backend);
      const Case cases[] = {{"output planes", 72, false}, {"output rows", 72, true}};
      for (const Case &c : cases)
        for (const uint32_t rows : {4U, 24U, 36U, 124U, 128U, 132U, 144U, 260U, 4100U})
          for (const Kind kind : {Kind::Random, Kind::Rising, Kind::Wide})
            runCase(backend, c, kind, rows);
      std::cout << "dense_attention_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "dense_attention_metal_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
