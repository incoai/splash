// Ripple's mixed draft geometry through production encoders: sixteen rows,
// four lanes, RoPE/NoPE, 4K/16K rings and noncausal proposal attention.
#include "LinearNumerics.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
using namespace splash;
using test::bf16ToFloat;
using test::floatToBf16;
using test::require;
using metal::MetalBuffer;
constexpr uint32_t Rows = 16, Heads = 80, KvHeads = 8, Dim = 128, Width = Heads * Dim, KvWidth = KvHeads * Dim;
uint16_t *words(MetalBuffer b) { return static_cast<uint16_t *>(b.contents()); }
float value(MetalBuffer b, uint64_t i) { return bf16ToFloat(words(b)[i]); }

void run(metal::MetalBackend &backend, uint32_t window, bool rotated, uint32_t lanes) {
  const ops::DraftAttentionShape shape{64, draft_dynamic_width(64, 4), Width + 2 * KvWidth, Width,
                                        Heads, KvHeads, Dim, window, 4, rotated};
  const auto plan = ops::DraftAttention::plan(shape, lanes, Rows);
  auto allocate = [&](uint64_t bytes) {
    auto b = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "draft parity");
    std::memset(b.contents(), 0, bytes);
    return b;
  };
  const uint64_t count = uint64_t{lanes} * Rows;
  const auto qkv = allocate(count * shape.qkvSize * 2);
  const auto queries = allocate(plan.workspace().groupedQueriesBytes);
  const auto saved = allocate(count * Width * 2);
  const auto reordered = allocate(count * Width * 2);
  const auto queryKeys = allocate(plan.workspace().queryKeysBytes);
  const auto queryValues = allocate(plan.workspace().queryValuesBytes);
  const auto norm = allocate(Dim * 2);
  const auto cosine = allocate(count * Dim / 2 * 4), sine = allocate(count * Dim / 2 * 4);
  const auto context = allocate(count * 2 * KvWidth * 2);
  const auto retained = allocate(lanes * 4);
  std::vector<uint32_t> positions(lanes), lengths(lanes);
  std::array<MetalBuffer, 4> keys{}, values{};
  for (uint32_t dim = 0; dim < Dim; ++dim) words(norm)[dim] = floatToBf16(1.0F);
  for (uint64_t i = 0; i < count * shape.qkvSize; ++i)
    words(qkv)[i] = floatToBf16(0.25F * std::sin(float(i * 17 % 1009)));
  for (uint64_t i = 0; i < count * 2 * KvWidth; ++i)
    words(context)[i] = floatToBf16(0.5F * std::cos(float(i * 31 % 1013)));
  for (uint64_t i = 0; i < count * Dim / 2; ++i) {
    static_cast<float *>(cosine.contents())[i] = std::cos(float(i % 73));
    static_cast<float *>(sine.contents())[i] = std::sin(float(i % 73));
  }
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    keys[lane] = allocate(uint64_t{window} * KvWidth * 2);
    values[lane] = allocate(uint64_t{window} * KvWidth * 2);
    positions[lane] = 2 * window + 31 + lane;
    lengths[lane] = positions[lane] + Rows;
    static_cast<uint32_t *>(retained.contents())[lane] = Rows;
  }
  for (uint32_t lane = lanes; lane < 4; ++lane) {
    keys[lane] = keys[lanes - 1];
    values[lane] = values[lanes - 1];
  }
  metal::CommandGraph graph;
  ops::DraftAttention::addPrepare(graph, {qkv, queries, norm, norm, cosine, sine, queryKeys, queryValues}, plan);
  ops::RowCopy::add(graph, queries, {0, Width, 0}, saved, {0, Width, 0}, count, Width);
  ops::DraftAttention::addContextCommit(graph, context, norm, cosine, sine, keys, values, retained, positions, shape);
  ops::DraftAttention::addDecode(graph, {queries, keys, values, queryKeys, queryValues}, lengths, plan);
  ops::DraftAttention::addReorder(graph, queries, reordered, plan);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < Rows; ++row)
      for (uint32_t head = 0; head < Heads; ++head) {
        double sum = 0;
        const uint64_t source = (uint64_t{lane} * Rows + row) * shape.qkvSize + head * Dim;
        for (uint32_t dim = 0; dim < Dim; ++dim) sum += std::pow(value(qkv, source + dim), 2);
        const double inverse = 1 / std::sqrt(sum / Dim + 1e-6);
        for (uint32_t dim : {0u, 63u, 64u, 127u}) {
          const uint32_t pair = dim % (Dim / 2);
          const float a = bf16ToFloat(floatToBf16(value(qkv, source + pair) * inverse));
          const float b = bf16ToFloat(floatToBf16(value(qkv, source + pair + Dim / 2) * inverse));
          const uint64_t r = (uint64_t{lane} * Rows + row) * Dim / 2 + pair;
          const float c = static_cast<float *>(cosine.contents())[r], s = static_cast<float *>(sine.contents())[r];
          const float expected = rotated ? (dim < Dim / 2 ? a * c - b * s : b * c + a * s)
                                         : (dim < Dim / 2 ? a : b);
          const uint64_t at = ((uint64_t{lane} * Heads + head) * Rows + row) * Dim + dim;
          require(std::fabs(value(saved, at) - expected) < 0.015F,
                  "draft q/k prepare changed the normalized or unrotated heads");
        }
      }
  // Context keys use the same per-layer rotation policy as proposal keys.
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < Rows; ++row)
      for (uint32_t head = 0; head < KvHeads; ++head) {
        const uint64_t source = (uint64_t{lane} * Rows + row) * 2 * KvWidth + head * Dim;
        double sum = 0;
        for (uint32_t dim = 0; dim < Dim; ++dim) sum += std::pow(value(context, source + dim), 2);
        const double inv = 1 / std::sqrt(sum / Dim + 1e-6);
        const uint32_t slot = (positions[lane] + row) % window;
        for (uint32_t dim : {0u, 63u, 64u, 127u}) {
          const uint32_t pair = dim % (Dim / 2);
          const float a = bf16ToFloat(floatToBf16(value(context, source + pair) * inv));
          const float b = bf16ToFloat(floatToBf16(value(context, source + pair + Dim / 2) * inv));
          const uint64_t r = (uint64_t{lane} * Rows + row) * Dim / 2 + pair;
          const float c = static_cast<float *>(cosine.contents())[r], s = static_cast<float *>(sine.contents())[r];
          const float expected = rotated ? (dim < Dim / 2 ? a * c - b * s : b * c + a * s)
                                         : (dim < Dim / 2 ? a : b);
          require(std::fabs(value(keys[lane], (uint64_t{head} * window + slot) * Dim + dim) - expected) < 0.015F,
                  "draft context keys changed the layer's rotation policy");
          require(value(values[lane], (uint64_t{head} * Dim + dim) * window + slot) ==
                      value(context, source + KvWidth + dim), "draft context value landed in another ring slot");
        }
      }
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t head : {0u, 9u, 10u, 79u})
      for (uint32_t row : {0u, 8u, 15u}) {
        const uint32_t kvHead = head / 10;
        const uint64_t query = ((uint64_t{lane} * Heads + head) * Rows + row) * Dim;
        const uint32_t old = window - 1, first = lengths[lane] - old;
        std::vector<double> scores(old + Rows);
        double maximum = -INFINITY;
        for (uint32_t token = 0; token < old + Rows; ++token) {
          double score = 0;
          for (uint32_t dim = 0; dim < Dim; ++dim) {
            const float key = token < old
                ? value(keys[lane], (uint64_t{kvHead} * window + (first + token) % window) * Dim + dim)
                : value(queryKeys, ((uint64_t{lane} * KvHeads + kvHead) * Rows + token - old) * Dim + dim);
            score += value(saved, query + dim) * key;
          }
          scores[token] = score / std::sqrt(double(Dim));
          maximum = std::max(maximum, scores[token]);
        }
        double mass = 0;
        for (double &score : scores) { score = std::exp(score - maximum); mass += score; }
        for (uint32_t dim : {0u, 63u, 127u}) {
          double expected = 0;
          for (uint32_t token = 0; token < old + Rows; ++token) {
            const float v = token < old
                ? value(values[lane], (uint64_t{kvHead} * Dim + dim) * window + (first + token) % window)
                : value(queryValues, ((uint64_t{lane} * KvHeads + kvHead) * Dim + dim) * Rows + token - old);
            expected += scores[token] / mass * v;
          }
          const float got = value(queries, query + dim);
          require(std::isfinite(got) && std::fabs(got - expected) < 0.00005 + 0.03 * std::fabs(expected),
                  "draft attention diverged from its retained-history reference");
          require(value(reordered, (uint64_t{lane} * Rows + row) * Width + head * Dim + dim) == got,
                  "sixteen-row draft reorder crossed a lane boundary");
        }
      }
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: ripple-draft METALLIB");
    metal::MetalBackend backend(argv[1]);
    for (uint32_t window : {4096u, 16384u})
      for (bool rotated : {true, false})
        for (uint32_t lanes = 1; lanes <= 4; ++lanes) run(backend, window, rotated, lanes);
    std::cout << "ripple-draft: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "ripple-draft: FAIL: " << error.what() << '\n';
    return 1;
  }
}
