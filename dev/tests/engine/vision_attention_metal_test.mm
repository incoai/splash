// vision_attention (kernels/shared/vision.metal) against an fp64 reference,
// row by row. Token counts fall below, on and past its 32-row query tile and
// 128-key tile, up to 4,100 (33 key tiles, the last of four keys), and each
// runs three kinds of input: random queries, keys and values; keys whose
// scores rise tile by tile, so every row rescales its running output at
// every tile; and scores and values over a wide dynamic range, whose
// probabilities underflow. Each tensor is bound at the extent the encoder's
// scratch gives it for that many patches (ops/Vision.cpp) and followed by a
// key tile of bf16 NaN: shader validation does not see MPP loads and stores
// past a binding, but a NaN read reaches its row and a write past the output
// changes its tail. The padded keys and values hold large finite values,
// which the softmax mask must hide.
//
//   vision-attention METALLIB
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Vision.h"
#include "ops/Vision.hpp"

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
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using splash::test::require;

constexpr splash::ops::VisionLayout kTower{};
constexpr uint32_t kHeads = kTower.heads;
constexpr uint32_t kHeadDimension = kTower.headDimension;
// The encoder's query and key rows hold the head dimension padded with
// zeros to the 16 a matmul's K extent takes, and every tensor's tokens are
// padded to whole key tiles (ops/Vision.cpp).
constexpr uint32_t kQkDimension = 80;
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

// `rows` rows of `width` bf16 values, the extent the kernel is bound to, in
// an allocation that follows them with a key tile of rows of NaN.
struct Tensor final {
  uint64_t elements = 0;
  uint64_t tail = 0;
  MetalBuffer allocation;
  MetalBuffer view;

  Tensor(MetalBackend &backend, uint64_t rows, uint32_t width)
      : elements(rows * width), tail(uint64_t{kKeyTile} * width),
        allocation(splash::test::sharedBuffer(backend, (elements + tail) * 2)),
        view(backend.view(allocation, 0, elements * 2)) {
    std::fill_n(data(), elements + tail, kNan);
  }
  [[nodiscard]] uint16_t *data() const { return static_cast<uint16_t *>(allocation.contents()); }
  [[nodiscard]] bool tailIntact() const {
    return std::all_of(data() + elements, data() + elements + tail, [](uint16_t value) { return value == kNan; });
  }
};

// The inputs of one image of `tokens` patches over `padded` rows of every
// head. A valid row's queries and keys end in the zeros vision_qkv_prepare
// writes past the head dimension; rows past the image hold large finite
// values. Rising keys carry their tile's offset in their first dimension,
// which a query's first dimension of one turns into scores that rise by 0.7
// to 1.2 from tile to tile, more than the spread of the small rest. Wide
// inputs scale each query row, key and value by its own power of two.
void fill(Kind kind, uint32_t tokens, uint32_t padded, const Tensor &queries, const Tensor &keys,
          const Tensor &values) {
  std::mt19937 random(0x7a11 + tokens * 3 + static_cast<uint32_t>(kind));
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::uniform_int_distribution<int> step(6, 10), queryExponent(-3, 3), keyExponent(-4, 3),
      valueExponent(-8, 8);
  std::vector<float> offset(padded / kKeyTile, 0.0F);
  for (size_t tile = 1; tile < offset.size(); ++tile)
    offset[tile] = offset[tile - 1] + static_cast<float>(step(random));
  for (uint32_t head = 0; head < kHeads; ++head) {
    for (uint32_t token = 0; token < padded; ++token) {
      const uint64_t row = uint64_t{head} * padded + token;
      uint16_t *query = queries.data() + row * kQkDimension;
      uint16_t *key = keys.data() + row * kQkDimension;
      uint16_t *value = values.data() + row * kHeadDimension;
      if (token >= tokens) {
        std::fill_n(query, kQkDimension, floatToBf16(16.0F));
        std::fill_n(key, kQkDimension, floatToBf16(16.0F));
        std::fill_n(value, kHeadDimension, floatToBf16(1024.0F));
        continue;
      }
      float queryScale = 1.4F, keyScale = 1.4F, valueScale = 1.0F;
      if (kind == Kind::Rising)
        queryScale = keyScale = 0.25F;
      if (kind == Kind::Wide) {
        queryScale = std::ldexp(1.0F, queryExponent(random));
        keyScale = std::ldexp(1.0F, keyExponent(random));
        valueScale = std::ldexp(1.0F, valueExponent(random));
      }
      std::fill_n(query, kQkDimension, uint16_t{0});
      std::fill_n(key, kQkDimension, uint16_t{0});
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        query[dimension] = floatToBf16(normal(random) * queryScale);
        key[dimension] = floatToBf16(normal(random) * keyScale);
        value[dimension] = floatToBf16(normal(random) * valueScale);
      }
      if (kind == Kind::Rising) {
        query[0] = floatToBf16(1.0F);
        key[0] = floatToBf16(offset[token / kKeyTile]);
      }
    }
  }
}

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
// scores (80 products, the scale and the maximum subtracted) change each
// probability by up to 2 epsilon; and fp32 sums over the keys and the
// rescales add a few units of 2^-24 per key.
HeadCheck checkHead(Kind kind, uint32_t tokens, uint32_t padded, float scale, uint32_t head,
                    const uint16_t *queries, const uint16_t *keys, const uint16_t *values,
                    const uint16_t *output) {
  HeadCheck check;
  const uint64_t first = uint64_t{head} * padded;
  std::vector<double> q(uint64_t{tokens} * kQkDimension), k(q.size());
  std::vector<double> v(uint64_t{tokens} * kHeadDimension);
  for (uint64_t index = 0; index < q.size(); ++index) {
    q[index] = bf16ToFloat(queries[first * kQkDimension + index]);
    k[index] = bf16ToFloat(keys[first * kQkDimension + index]);
  }
  for (uint64_t index = 0; index < v.size(); ++index)
    v[index] = bf16ToFloat(values[first * kHeadDimension + index]);
  double largestKey = 0.0;
  for (uint32_t key = 0; key < tokens; ++key) {
    double squares = 0.0;
    for (uint32_t dimension = 0; dimension < kQkDimension; ++dimension)
      squares += k[uint64_t{key} * kQkDimension + dimension] * k[uint64_t{key} * kQkDimension + dimension];
    largestKey = std::max(largestKey, std::sqrt(squares));
  }

  std::vector<double> logits(tokens), sums(kHeadDimension), magnitudes(kHeadDimension);
  for (uint32_t row = 0; row < tokens; ++row) {
    const double *query = q.data() + uint64_t{row} * kQkDimension;
    double queryNorm = 0.0;
    for (uint32_t dimension = 0; dimension < kQkDimension; ++dimension)
      queryNorm += query[dimension] * query[dimension];
    queryNorm = std::sqrt(queryNorm);
    double maximum = -INFINITY, largestLogit = 0.0;
    for (uint32_t key = 0; key < tokens; ++key) {
      const double *keyRow = k.data() + uint64_t{key} * kQkDimension;
      double partial[8] = {};
      for (uint32_t dimension = 0; dimension < kQkDimension; dimension += 8)
        for (uint32_t lane = 0; lane < 8; ++lane)
          partial[lane] += query[dimension + lane] * keyRow[dimension + lane];
      const double dot = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                         ((partial[4] + partial[5]) + (partial[6] + partial[7]));
      logits[key] = dot * scale;
      maximum = std::max(maximum, logits[key]);
      largestLogit = std::max(largestLogit, std::fabs(logits[key]));
    }
    if (kind == Kind::Rising) {
      double previous = -INFINITY;
      for (uint32_t start = 0; start < tokens; start += kKeyTile) {
        const double tile =
            *std::max_element(logits.begin() + start, logits.begin() + std::min(tokens, start + kKeyTile));
        if (start && !(tile > previous + 0.05))
          ++check.steadyTiles;
        previous = std::max(previous, tile);
      }
    }

    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(magnitudes.begin(), magnitudes.end(), 0.0);
    double total = 0.0;
    for (uint32_t key = 0; key < tokens; ++key) {
      const double probability = std::exp(logits[key] - maximum);
      const double *value = v.data() + uint64_t{key} * kHeadDimension;
      total += probability;
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        sums[dimension] += probability * value[dimension];
        magnitudes[dimension] += probability * std::fabs(value[dimension]);
      }
    }
    const double epsilon =
        scale * kQkDimension * 0x1p-24 * queryNorm * largestKey + 0x1p-22 * largestLogit;
    const double relative = 0x1p-8 + 2.5 * epsilon + (2.0 * tokens + 512.0) * 0x1p-24 + 0x1p-17;
    const uint16_t *actual = output + (first + row) * kHeadDimension;
    for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
      const double expected = sums[dimension] / total;
      const double bound = relative * (magnitudes[dimension] / total + std::fabs(expected));
      const double observed = bf16ToFloat(actual[dimension]);
      const double error = std::fabs(observed - expected);
      if (!(error <= bound)) {
        check.failure = "row " + std::to_string(row) + " dimension " + std::to_string(dimension) + ": " +
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

// One image of `tokens` patches: the encoder's attention dispatch over
// tensors at their extents, checked head by head.
void runCase(MetalBackend &backend, Kind kind, uint32_t tokens) {
  const uint32_t padded = (tokens + kKeyTile - 1) / kKeyTile * kKeyTile;
  const uint64_t rows = uint64_t{kHeads} * padded;
  const Tensor queries(backend, rows, kQkDimension), keys(backend, rows, kQkDimension);
  const Tensor values(backend, rows, kHeadDimension), output(backend, rows, kHeadDimension);
  fill(kind, tokens, padded, queries, keys, values);
  const VisionAttentionParams params{tokens, padded, 1.0F / std::sqrt(static_cast<float>(kHeadDimension))};
  CommandGraph graph;
  graph.add("vision_attention", {queries.view, keys.view, values.view, output.view}, params,
            {(tokens + SPLASH_VISION_ATTENTION_ROWS - 1) / SPLASH_VISION_ATTENTION_ROWS, kHeads, 1},
            {SPLASH_VISION_ATTENTION_THREADS, 1, 1});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  const std::string label = std::string(kindName(kind)) + " tokens " + std::to_string(tokens);
  require(queries.tailIntact() && keys.tailIntact() && values.tailIntact() && output.tailIntact(),
          label + ": a write reached past a tensor's extent");
  std::vector<std::future<HeadCheck>> checks;
  for (uint32_t head = 0; head < kHeads; ++head)
    checks.push_back(std::async(std::launch::async, checkHead, kind, tokens, padded, params.scale, head,
                                queries.data(), keys.data(), values.data(), output.data()));
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
  std::printf("vision_attention %s: worst error %.3f of its bound (head %u row %u)\n", label.c_str(), worst,
              worstHead, worstRow);
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "usage: vision-attention METALLIB");
      MetalBackend backend(argv[1]);
      for (const uint32_t tokens : {4U, 24U, 36U, 124U, 128U, 132U, 144U, 260U, 4100U})
        for (const Kind kind : {Kind::Random, Kind::Rising, Kind::Wide})
          runCase(backend, kind, tokens);
      std::cout << "vision_attention_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "vision_attention_metal_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
