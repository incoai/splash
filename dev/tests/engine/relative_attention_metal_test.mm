#include "HostKvExtents.hpp"
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "TestFamilies.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "metal/abi/RelativeAttention.h"
#include "ops/KernelCalibration.hpp"
#include "ops/LaneRows.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/RelativeAttention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Relative attention (ops::RelativeAttention): 24 query heads over 8 KV heads
// of 128 dimensions, its queries and keys clipped heads, its values
// normalized ones, each score biased by the row's relative states mixed with
// the layer's table at the key's backward distance below 1024, a sink of the
// row's own for each query head, its output gated; each row over its whole
// history (the full layers', their queries and states scaled by the row's
// log scale) or a window of the 1023 keys before it. Each case runs the
// production encoders: the prepare, against fp64 of its rows, and the
// attention, against fp64 of what the prepare staged and the K/V the pages
// hold.

namespace {

using namespace splash;
using test::HostKvExtents;
using test::bf16ToFloat;
using test::floatToBf16;
using test::rejects;
using test::require;
using test::ulpBf16;
using ops::RelativeAttention;

constexpr uint32_t kQueryHeads = RelativeAttention::kQueryHeads, kKvHeads = RelativeAttention::kKvHeads;
constexpr uint32_t kGroup = kQueryHeads / kKvHeads, kHeadDimension = RelativeAttention::kHeadDimension;
constexpr uint32_t kStates = SPLASH_RELATIVE_ATTENTION_STATES, kExtent = SPLASH_RELATIVE_ATTENTION_EXTENT;
constexpr uint32_t kWindow = RelativeAttention::kWindowTokens;
// The projection's blocks, and its rows' stride (whole tiles of the
// projection's outputs: its padding columns hold NaNs, which no kernel reads).
constexpr uint64_t kWidth = RelativeAttention::kProjectionWidth, kStride = RelativeAttention::kProjectionStride;
static_assert(kStride == 6592, "the projection's rows are 103 tiles of 64 outputs");
constexpr uint32_t kGateOffset = kQueryHeads * kHeadDimension, kStateOffset = 2 * kQueryHeads * kHeadDimension;
constexpr uint32_t kSinkOffset = kStateOffset + kQueryHeads * kStates;
constexpr uint32_t kPageTokens = kv::kPageTokens;
constexpr uint32_t kLaneRows = ops::kNarrowLaneRows;
// The layer under test is the second of a pool's two.
constexpr uint32_t kLayer = 1;
// A sliding layer's variant and a full layer's.
constexpr ops::RelativeAttentionVariant kSliding{.windowTokens = kWindow, .positionScaled = false};
constexpr ops::RelativeAttentionVariant kFull{.windowTokens = 0, .positionScaled = true};
// Sinks below, inside and above the scores, and above every score: a row's
// head h takes kSinks[(h + row) % 4] and a quarter for each four heads before
// it, so that each row's and head's sink is its own.
constexpr std::array kSinks{-9.0f, 0.0f, 12.4f, 60.0f};

kv::Layout layoutOf(kv::Format format, uint32_t layers = 1) {
  return {.attentionLayers = layers, .kvHeads = kKvHeads, .headDimension = kHeadDimension, .format = format};
}

// A value in [-1, 1) that depends on every bit of `seed`.
float unit(uint64_t seed) {
  seed += 0x9e3779b97f4a7c15ULL;
  seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
  seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
  seed ^= seed >> 31;
  return float(seed >> 40) / float(1U << 23) - 1.0f;
}
uint64_t seedOf(uint32_t kind, uint32_t lane, uint32_t token, uint32_t column) {
  return ((uint64_t{kind} * 8 + lane) * 32768 + token) * 8192 + column;
}
float bf16(float value) { return bf16ToFloat(floatToBf16(value)); }

// A lane's row's inputs, by absolute row of the lane's rows: its projection
// row's column (queries of three magnitudes a head, so that the clip holds
// some heads and passes others; gates; relative states; sinks) and its
// convolved K and V rows (keys of growing magnitude a head, values of any:
// the norm takes it).
float projection(uint32_t lane, uint32_t row, uint32_t column) {
  if (column >= kSinkOffset) {
    const uint32_t head = column - kSinkOffset;
    return bf16(kSinks[(head + row) % kSinks.size()] + 0.25f * float(head / kSinks.size()));
  }
  const float amplitude = column < kGateOffset ? 1.0f + float(column / kHeadDimension % 3) : 2.0f;
  return bf16(amplitude * unit(seedOf(0, lane, row, column)));
}
float chunkKey(uint32_t lane, uint32_t row, uint32_t head, uint32_t dimension) {
  return bf16((0.5f + 0.5f * float(head)) * unit(seedOf(1, lane, row, head * kHeadDimension + dimension)));
}
float chunkValue(uint32_t lane, uint32_t row, uint32_t head, uint32_t dimension) {
  return bf16((0.25f + float(head)) * unit(seedOf(2, lane, row, head * kHeadDimension + dimension)));
}

// The layer's weights: query scales, relative-state scales (both around one,
// head by head apart) and the bias table.
float queryScale(uint32_t head, uint32_t dimension) {
  return 0.4f + 0.15f * float(head % 8) + 0.1f * unit(seedOf(3, 0, head, dimension));
}
float relativeScale(uint32_t head, uint32_t state) {
  return 0.5f + 0.4f * float(head % 6) + 0.1f * unit(seedOf(4, 0, head, state));
}
float biasOf(uint32_t distance, uint32_t state) { return 0.6f * unit(seedOf(5, 0, distance, state)); }

metal::MetalBuffer filled(metal::MetalBackend &backend, uint64_t bytes, int byte) {
  metal::MetalBuffer buffer = test::sharedBuffer(backend, std::max<uint64_t>(bytes, 1));
  std::memset(buffer.contents(), byte, buffer.sizeBytes());
  return buffer;
}
metal::MetalBuffer zeroed(metal::MetalBackend &backend, uint64_t bytes) { return filled(backend, bytes, 0); }
// Partials of NaNs: a reduce that combines a split which wrote nothing reads
// them.
metal::MetalBuffer poisoned(metal::MetalBackend &backend, uint64_t bytes) { return filled(backend, bytes, 0xff); }
template <class T> T *contents(const metal::MetalBuffer &buffer) { return static_cast<T *>(buffer.contents()); }

double logScale(uint32_t position) {
  return 1.0 + double(SPLASH_RELATIVE_ATTENTION_LOG_SCALE_ALPHA) *
                   std::log(std::max(double(position + 1) / SPLASH_RELATIVE_ATTENTION_LOG_SCALE_FLOOR, 1.0));
}
// The clip of a head to an L2 norm of at most `clip`, in fp64.
double clipFactor(std::span<const double> head, double clip) {
  double squares = 0;
  for (const double value : head) squares += value * value;
  return std::min(clip / (std::sqrt(squares) + SPLASH_RELATIVE_ATTENTION_CLIP_EPSILON), 1.0);
}

// One attention of `lanes` lanes after their histories, as
// attention_variants_metal_test.mm's cases: each lane's pages mixed over the
// extents of a pool of two layers, its history written into them directly;
// with a window, the entries of a lane's table below its first row's window
// name a page of NaNs.
struct Case final {
  kv::Layout layout;
  ops::RelativeAttentionVariant variant;
  bool int8Queries;
  uint32_t lanes;
  uint32_t rows;
  std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> histories;
  HostKvExtents pool;
  SplashKvLayer layer;
  std::array<std::vector<uint32_t>, SPLASH_MAXIMUM_BATCH_WIDTH> pages;
  std::array<metal::MetalBuffer, SPLASH_MAXIMUM_BATCH_WIDTH> tables;
  uint32_t poisoned = 0;
  ops::RelativeAttentionWeights weights;

  [[nodiscard]] uint32_t page(uint32_t lane, uint32_t token) const { return pages[lane][token / kPageTokens]; }
  [[nodiscard]] ops::AttentionCalibration attention() const {
    return test::attentionRow(int8Queries ? 1U : ops::kNever);
  }
  [[nodiscard]] double key(uint32_t lane, uint32_t head, uint32_t token, uint32_t dimension) const {
    const uint64_t index = splash_kv_key_element(head, token % kPageTokens, dimension, kHeadDimension);
    if (layout.format == kv::Format::BFloat16)
      return bf16ToFloat(pool.slab<uint16_t>(kLayer, SPLASH_KV_KEYS, page(lane, token))[index]);
    return double(pool.slab<int8_t>(kLayer, SPLASH_KV_KEYS, page(lane, token))[index]) *
           pool.slab<float>(kLayer, SPLASH_KV_KEY_SCALES, page(lane, token))[
               splash_kv_scale_element(head, token % kPageTokens)];
  }
  [[nodiscard]] double value(uint32_t lane, uint32_t head, uint32_t token, uint32_t dimension) const {
    const uint64_t index = pool.valueElement(head, token % kPageTokens, dimension);
    if (layout.format == kv::Format::BFloat16)
      return bf16ToFloat(pool.slab<uint16_t>(kLayer, SPLASH_KV_VALUES, page(lane, token))[index]);
    return double(pool.slab<int8_t>(kLayer, SPLASH_KV_VALUES, page(lane, token))[index]) *
           pool.slab<float>(kLayer, SPLASH_KV_VALUE_SCALES, page(lane, token))[
               splash_kv_scale_element(head, token % kPageTokens)];
  }
};

uint32_t windowStart(ops::RelativeAttentionVariant variant, uint32_t position) {
  return variant.windowTokens && position > variant.windowTokens ? position - variant.windowTokens : 0;
}

Case makeCase(metal::MetalBackend &backend, kv::Format format, bool int8Queries, ops::RelativeAttentionVariant variant,
              uint32_t lanes, uint32_t rows, std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> histories) {
  std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> pageCounts{};
  uint32_t allPages = 1;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    pageCounts[lane] = (histories[lane] + rows + kPageTokens - 1) / kPageTokens;
    allPages += pageCounts[lane];
  }
  const auto spread = HostKvExtents::spread(allPages + 2);
  const std::vector<uint32_t> ids = HostKvExtents::mixedPages(spread, allPages, histories[0] + rows + lanes);
  Case data{layoutOf(format), variant, int8Queries, lanes, rows, histories,
            HostKvExtents(backend, layoutOf(format, kLayer + 1), spread.extentPages, spread.extents),
            {}, {}, {}, 0,
            {zeroed(backend, kQueryHeads * kHeadDimension * sizeof(float)),
             zeroed(backend, kQueryHeads * kStates * sizeof(float)),
             zeroed(backend, kExtent * kStates * sizeof(float))}};
  data.layer = data.pool.layer(kLayer);
  for (uint32_t head = 0; head < kQueryHeads; ++head) {
    for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
      contents<float>(data.weights.queryScale)[head * kHeadDimension + dimension] = queryScale(head, dimension);
    for (uint32_t state = 0; state < kStates; ++state)
      contents<float>(data.weights.relativeScale)[head * kStates + state] = relativeScale(head, state);
  }
  for (uint32_t distance = 0; distance < kExtent; ++distance)
    for (uint32_t state = 0; state < kStates; ++state)
      contents<float>(data.weights.bias)[distance * kStates + state] = biasOf(distance, state);
  const bool quantized = format == kv::Format::Int8;
  const uint32_t nan = ids.back();
  for (uint32_t tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES; ++tensor) {
    const auto bytes = data.pool.layerSpans(nan, kLayer)[tensor];
    if (tensor % 2 == 0 && quantized)
      std::fill(bytes.begin(), bytes.end(), std::byte{0x7f});
    else
      for (size_t at = 0; at + 4 <= bytes.size(); at += 4) {
        const uint32_t bits = tensor % 2 ? 0x7fc00000u : 0x7fc07fc0u;
        std::memcpy(bytes.data() + at, &bits, 4);
      }
  }
  uint32_t firstPage = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    data.pages[lane].assign(ids.begin() + firstPage, ids.begin() + firstPage + pageCounts[lane]);
    firstPage += pageCounts[lane];
    std::vector<uint32_t> table = data.pages[lane];
    if (variant.windowTokens)
      for (uint32_t page = 0; page < windowStart(variant, histories[lane]) / kPageTokens; ++page) {
        table[page] = nan;
        ++data.poisoned;
      }
    data.tables[lane] = zeroed(backend, table.size() * sizeof(SplashKvPage));
    data.pool.writeTable(table, data.tables[lane]);
    for (uint32_t token = 0; token < histories[lane]; ++token)
      for (uint32_t head = 0; head < kKvHeads; ++head) {
        const uint32_t id = data.page(lane, token), place = token % kPageTokens;
        const uint64_t scale = splash_kv_scale_element(head, place);
        if (quantized) {
          data.pool.slab<float>(kLayer, SPLASH_KV_KEY_SCALES, id)[scale] = 0.02f * (1 + float((token + head) % 9) / 16);
          data.pool.slab<float>(kLayer, SPLASH_KV_VALUE_SCALES, id)[scale] =
              0.008f * (1 + float((token * 5 + head) % 11) / 16);
        }
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const float key = unit(seedOf(6, lane, token, head * kHeadDimension + dimension));
          const float value = unit(seedOf(7, lane, token, head * kHeadDimension + dimension));
          const uint64_t keyIndex = splash_kv_key_element(head, place, dimension, kHeadDimension);
          const uint64_t valueIndex = data.pool.valueElement(head, place, dimension);
          if (quantized) {
            data.pool.slab<int8_t>(kLayer, SPLASH_KV_KEYS, id)[keyIndex] = int8_t(std::lround(127 * key));
            data.pool.slab<int8_t>(kLayer, SPLASH_KV_VALUES, id)[valueIndex] = int8_t(std::lround(127 * value));
          } else {
            data.pool.slab<uint16_t>(kLayer, SPLASH_KV_KEYS, id)[keyIndex] = floatToBf16(2.5f * key);
            data.pool.slab<uint16_t>(kLayer, SPLASH_KV_VALUES, id)[valueIndex] = floatToBf16(value);
          }
        }
      }
  }
  for (uint32_t lane = lanes; lane < data.tables.size(); ++lane) data.tables[lane] = data.tables[0];
  return data;
}

// A verify block's rows as a tree (metal/abi/VerifyTree.h): each row's
// parent before it, the anchor's none. A chain's row r follows r - 1.
struct Tree final {
  std::vector<uint32_t> parent;

  [[nodiscard]] std::vector<uint32_t> path(uint32_t row) const {
    std::vector<uint32_t> rows{row};
    while (rows.front()) rows.insert(rows.begin(), parent[rows.front()]);
    return rows;
  }
  [[nodiscard]] SplashTreeRow place(uint32_t row) const {
    uint32_t ancestors = 0;
    for (const uint32_t on : path(row)) ancestors |= 1U << on;
    return {ancestors, static_cast<uint16_t>(row ? parent[row] : SPLASH_TREE_ROOT),
            static_cast<uint16_t>(path(row).size() - 1)};
  }
};

// The rows of a command: rows [first, first + rows) of each lane's, `stride`
// rows a KV head staged, each lane's rows after the one before it, or with
// trees each lane's block of rows as a tree; their inputs, staging and output
// rows.
struct Step final {
  uint32_t first;
  uint32_t rows;
  uint32_t stride;
  std::vector<Tree> trees;
  ops::RelativeAttentionRows inputs;
  ops::RelativeAttentionStaging staging;
  metal::MetalBuffer output, layout, tree;

  [[nodiscard]] uint64_t fused(uint32_t lane, uint32_t head, uint32_t row) const {
    return ((uint64_t{lane} * kKvHeads + head / kGroup) * stride + row) * kGroup + head % kGroup;
  }
  [[nodiscard]] uint64_t outputIndex(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) const {
    return ((uint64_t{lane} * rows + row) * kQueryHeads + head) * kHeadDimension + dimension;
  }
  // A lane's row's position after `history`, and its depth.
  [[nodiscard]] uint32_t depth(uint32_t lane, uint32_t row) const {
    return trees.empty() ? first + row : uint32_t(trees[lane].path(row).size() - 1);
  }
};

std::vector<kv::ChunkedPrefillParams> sequencesOf(const Case &data, const Step &step) {
  std::vector<kv::ChunkedPrefillParams> sequences;
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    sequences.push_back(ops::PagedAttention::chunkParams(data.histories[lane] + step.first, step.rows, step.stride,
                                                         uint32_t(data.pages[lane].size())));
  return sequences;
}

Step makeStep(metal::MetalBackend &backend, const Case &data, uint32_t first, uint32_t rows, uint32_t stride,
              std::vector<Tree> trees = {}) {
  const uint64_t packed = uint64_t{data.lanes} * rows;
  Step step{first, rows, stride, std::move(trees), {}, {}, {}, {}, {}};
  step.inputs = {poisoned(backend, packed * kStride * 2), zeroed(backend, packed * kKvHeads * kHeadDimension * 2),
                 zeroed(backend, packed * kKvHeads * kHeadDimension * 2)};
  const ops::RelativeAttentionStagingBytes bytes =
      RelativeAttention::stagingBytes(sequencesOf(data, step), data.layout);
  step.staging = {poisoned(backend, bytes.queries), poisoned(backend, bytes.states), poisoned(backend, bytes.sinks)};
  step.output = zeroed(backend, packed * kQueryHeads * kHeadDimension * 2);
  step.layout = zeroed(backend, uint64_t{data.lanes} * sizeof(SplashLaneRows));
  if (!step.trees.empty()) step.tree = zeroed(backend, packed * sizeof(SplashTreeRow));
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    contents<SplashLaneRows>(step.layout)[lane] = {lane * rows, rows};
    for (uint32_t row = 0; row < rows; ++row) {
      const uint64_t at = uint64_t{lane} * rows + row;
      for (uint32_t column = 0; column < kWidth; ++column)
        contents<uint16_t>(step.inputs.projection)[at * kStride + column] =
            floatToBf16(projection(lane, first + row, column));
      for (uint32_t head = 0; head < kKvHeads; ++head)
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const uint64_t element = (at * kKvHeads + head) * kHeadDimension + dimension;
          contents<uint16_t>(step.inputs.keys)[element] = floatToBf16(chunkKey(lane, first + row, head, dimension));
          contents<uint16_t>(step.inputs.values)[element] = floatToBf16(chunkValue(lane, first + row, head, dimension));
        }
      if (!step.trees.empty()) contents<SplashTreeRow>(step.tree)[at] = step.trees[lane].place(row);
    }
  }
  return step;
}

ops::RelativeAttentionPlan planOf(const Case &data, const Step &step) {
  return RelativeAttention::plan(sequencesOf(data, step), data.layout, data.attention(), ops::Int8Activations::On,
                                 RelativeAttention::workspace(data.lanes * ops::kAttentionSequenceSplits, data.layout),
                                 data.variant);
}

// The prepare and the attention of a step, as the runtime encodes them.
metal::CommandGraph encode(const Case &data, const Step &step, const ops::RelativeAttentionPlan &plan,
                           const metal::MetalBuffer &partials, const metal::MetalBuffer &statistics) {
  metal::CommandGraph graph;
  const std::vector<kv::ChunkedPrefillParams> sequences = sequencesOf(data, step);
  const ops::AttentionRows rows{step.layout, data.lanes * step.rows, step.tree};
  RelativeAttention::addProjection(graph, step.inputs, data.weights, step.staging, data.layer,
                                   std::span(data.tables).first(data.lanes), sequences, rows, data.layout,
                                   data.variant);
  std::vector<ops::AttentionHistory> histories;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) histories.push_back({data.layer, data.tables[lane]});
  RelativeAttention::addAttention(graph, plan, step.staging, data.weights, partials, statistics, sequences, histories,
                                  rows, {step.inputs.projection, step.output});
  return graph;
}

void run(metal::MetalBackend &backend, const Case &data, const Step &step, const ops::RelativeAttentionPlan &plan) {
  const metal::MetalBuffer partials = poisoned(backend, plan.workspace().partialsBytes);
  const metal::MetalBuffer statistics = poisoned(backend, plan.workspace().statisticsBytes);
  const metal::CommandGraph graph = encode(data, step, plan, partials, statistics);
  const auto &dispatches = graph.dispatches();
  const std::string format = data.layout.format == kv::Format::Int8 ? "q8" : "bf16";
  require(dispatches.size() == 3 &&
              dispatches[0].pipelineName ==
                  "relative_attention_" + format + "_prepare" + (data.variant.positionScaled ? "_scaled" : "") &&
              dispatches[2].threadsPerThreadgroup.x == kHeadDimension,
          "the prepare names another variant's kernel");
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    require(plan.sequences()[lane].splits == test::attentionSplits(step.rows, data.histories[lane] + step.first,
                                                                   false, data.variant.windowTokens),
            "a lane's splits are not the paged attention's");
  static_cast<void>(backend.submitCommandAsync(dispatches).wait());
}

// ---- The prepare -----------------------------------------------------------

// Within `ulps` bf16 steps of `expected` (the prepare rounds once, or twice
// where it scales).
void near(double actual, double expected, double ulps, const std::string &what) {
  if (!(std::fabs(actual - expected) <= ulps * ulpBf16(float(expected)) * (1 + 1e-6)))
    throw std::runtime_error(what + " is " + std::to_string(actual) + ", fp64 " + std::to_string(expected));
}

// The staged queries, relative states and sinks, and the stored keys and
// values of every row of a step, against fp64.
void checkPrepare(const Case &data, const Step &step, const std::string &what) {
  const double rounding = data.variant.positionScaled ? 2.01 : 1.01;
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    for (uint32_t row = 0; row < step.rows; ++row) {
      const uint32_t absolute = step.first + row;
      const uint32_t position = data.histories[lane] + step.depth(lane, row);
      const double scale = data.variant.positionScaled ? logScale(position) : 1.0;
      const std::string at = what + ": lane " + std::to_string(lane) + " row " + std::to_string(absolute);
      for (uint32_t head = 0; head < kQueryHeads; ++head) {
        std::vector<double> queries(kHeadDimension), states(kStates);
        for (uint32_t d = 0; d < kHeadDimension; ++d)
          queries[d] = double(projection(lane, absolute, head * kHeadDimension + d)) * double(queryScale(head, d));
        for (uint32_t k = 0; k < kStates; ++k)
          states[k] =
              double(projection(lane, absolute, kStateOffset + head * kStates + k)) * double(relativeScale(head, k));
        const double queryClip = clipFactor(queries, SPLASH_RELATIVE_ATTENTION_QK_CLIP);
        const double stateClip = clipFactor(states, SPLASH_RELATIVE_ATTENTION_STATE_CLIP);
        const uint64_t fused = step.fused(lane, head, row);
        for (uint32_t d = 0; d < kHeadDimension; ++d)
          near(bf16ToFloat(contents<uint16_t>(step.staging.queries)[fused * kHeadDimension + d]),
               queries[d] * queryClip * scale, rounding, at + " query head " + std::to_string(head));
        for (uint32_t k = 0; k < kStates; ++k)
          near(contents<float>(step.staging.states)[fused * kStates + k], states[k] * stateClip * scale, rounding,
               at + " relative state of head " + std::to_string(head));
        require(contents<float>(step.staging.sinks)[fused] == projection(lane, absolute, kSinkOffset + head),
                at + ": a sink is not its row's own");
      }
      // The row's K/V at the slot of its row.
      const uint32_t token = data.histories[lane] + absolute;
      for (uint32_t head = 0; head < kKvHeads; ++head) {
        std::vector<double> keys(kHeadDimension), values(kHeadDimension);
        double squares = 0;
        for (uint32_t d = 0; d < kHeadDimension; ++d) {
          keys[d] = chunkKey(lane, absolute, head, d);
          values[d] = chunkValue(lane, absolute, head, d);
          squares += values[d] * values[d];
        }
        const double keyClip = clipFactor(keys, SPLASH_RELATIVE_ATTENTION_QK_CLIP);
        const double inverse = 1 / std::sqrt(squares / kHeadDimension + 1e-6);
        double keyLargest = 0, valueLargest = 0;
        for (uint32_t d = 0; d < kHeadDimension; ++d) {
          keyLargest = std::max(keyLargest, std::fabs(keys[d] * keyClip));
          valueLargest = std::max(valueLargest, std::fabs(values[d] * inverse));
        }
        for (uint32_t d = 0; d < kHeadDimension; ++d) {
          const double key = keys[d] * keyClip, value = values[d] * inverse;
          if (data.layout.format == kv::Format::BFloat16) {
            near(data.key(lane, head, token, d), key, 1.01, at + " key");
            near(data.value(lane, head, token, d), value, 1.01, at + " value");
          } else {
            require(std::fabs(data.key(lane, head, token, d) - key) <= keyLargest / 127 * 0.51 + ulpBf16(float(key)) &&
                        std::fabs(data.value(lane, head, token, d) - value) <=
                            valueLargest / 127 * 0.51 + ulpBf16(float(value)),
                    at + ": a stored INT8 key or value is off its row");
          }
        }
      }
    }
}

// ---- The attention ---------------------------------------------------------

// A key a row attends: its slot and its backward distance from the row.
struct Key final {
  uint32_t token;
  uint32_t distance;
};

// The keys a lane's row attends: of a chain's row at position p, those of
// its window or its whole history to p, at distance p - t; of a tree's, at
// the anchor's position plus its depth, that position's window of the
// history and its path's rows of the block, each at the slot of its row and
// at the distance of their depths.
std::vector<Key> keysOf(const Case &data, const Step &step, uint32_t lane, uint32_t row) {
  const uint32_t history = data.histories[lane];
  std::vector<Key> keys;
  if (step.trees.empty()) {
    const uint32_t position = history + step.first + row;
    for (uint32_t token = windowStart(data.variant, position); token <= position; ++token)
      keys.push_back({token, position - token});
    return keys;
  }
  const std::vector<uint32_t> path = step.trees[lane].path(row);
  const uint32_t depth = uint32_t(path.size() - 1), position = history + depth;
  for (uint32_t token = windowStart(data.variant, position); token < history; ++token)
    keys.push_back({token, position - token});
  for (uint32_t on = 0; on < path.size(); ++on) keys.push_back({history + path[on], depth - on});
  return keys;
}

struct Reference final {
  std::array<double, kHeadDimension> output{};
  std::array<double, kHeadDimension> mass{};
};

// The stored K/V of a lane's tokens [first, end), dequantized once (exact in
// fp32: a bf16, or an INT8 code times its fp32 scale rounded once, well
// inside the bound): [token - first][KV head][dimension].
struct StoredKv final {
  uint32_t first = 0;
  std::vector<float> keys, values;

  [[nodiscard]] uint64_t index(uint32_t head, uint32_t token, uint32_t dimension) const {
    return (uint64_t{token - first} * kKvHeads + head) * kHeadDimension + dimension;
  }
};

StoredKv stored(const Case &data, uint32_t lane, uint32_t first, uint32_t end) {
  StoredKv kv{first, std::vector<float>(uint64_t{end - first} * kKvHeads * kHeadDimension), {}};
  kv.values.resize(kv.keys.size());
  for (uint32_t token = first; token < end; ++token)
    for (uint32_t head = 0; head < kKvHeads; ++head)
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        kv.keys[kv.index(head, token, dimension)] = float(data.key(lane, head, token, dimension));
        kv.values[kv.index(head, token, dimension)] = float(data.value(lane, head, token, dimension));
      }
  return kv;
}

// The fp64 attention of a lane's row and query head over the K/V the pages
// hold, from what the prepare staged: its queries (as INT8 codes times their
// row's scale where the case scores INT8 queries), its relative states, whose
// dot with the bias table's row of a key's distance below the extent adds to
// its score, and its sink, a key with no value.
Reference reference(const Case &data, const Step &step, const StoredKv &kv, uint32_t lane, uint32_t row,
                    uint32_t head) {
  const uint64_t fused = step.fused(lane, head, row);
  std::array<double, kHeadDimension> queries{};
  for (uint32_t d = 0; d < kHeadDimension; ++d)
    queries[d] = bf16ToFloat(contents<uint16_t>(step.staging.queries)[fused * kHeadDimension + d]);
  if (data.int8Queries) {
    float largest = 0;
    for (const double value : queries) largest = std::max(largest, float(std::fabs(value)));
    const test::Int8Block block = test::int8Block(largest);
    for (double &value : queries) value = double(test::int8Code(float(value), block)) * block.scale;
  }
  const float *states = &contents<float>(step.staging.states)[fused * kStates];
  const double sink = contents<float>(step.staging.sinks)[fused];
  const std::vector<Key> keys = keysOf(data, step, lane, row);
  std::vector<double> scores(keys.size());
  double maximum = sink;
  for (size_t i = 0; i < keys.size(); ++i) {
    double score = 0;
    const float *stored = &kv.keys[kv.index(head / kGroup, keys[i].token, 0)];
    for (uint32_t d = 0; d < kHeadDimension; ++d) score += queries[d] * stored[d];
    score /= std::sqrt(double(kHeadDimension));
    if (keys[i].distance < kExtent)
      for (uint32_t k = 0; k < kStates; ++k)
        score += double(states[k]) * contents<float>(data.weights.bias)[keys[i].distance * kStates + k];
    scores[i] = score;
    maximum = std::max(maximum, score);
  }
  double denominator = std::exp(sink - maximum);
  for (double &score : scores) {
    score = std::exp(score - maximum);
    denominator += score;
  }
  Reference result;
  for (size_t i = 0; i < keys.size(); ++i) {
    const double weight = scores[i] / denominator;
    const float *stored = &kv.values[kv.index(head / kGroup, keys[i].token, 0)];
    for (uint32_t d = 0; d < kHeadDimension; ++d) {
      const double value = stored[d];
      result.output[d] += weight * value;
      result.mass[d] += weight * std::fabs(value);
    }
  }
  return result;
}

// The rows of a step the reference checks: all of a short one's, and of a
// long one those of its first and last two tiles, those about the window's
// and the extent's ends, and a spread of the others.
std::vector<uint32_t> checkedRows(uint32_t position, uint32_t rows) {
  std::vector<uint32_t> checked;
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t tile = row / kv::kAttentionTileRows, tiles = kv::attentionTiles(rows);
    const bool edge =
        tile < 2 || tile + 2 >= tiles || (position + row >= kExtent - 16 && position + row < kExtent + 16);
    if (rows <= 64 || edge || row % 13 == 0) checked.push_back(row);
  }
  return checked;
}

// Each checked output against the reference, gated by its head's gate: within
// the bf16 rounding of the probabilities (2^-9 of the mass, twice over) and of
// the attention row, both times the gate, and of the gated row. Returns the
// largest error in units of that bound.
double checkOutput(const Case &data, const Step &step, const std::string &what) {
  double largest = 0;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    // Every key a checked row attends: from the first row's window (a tree's
    // anchor's) to the step's last row's slot.
    const uint32_t position = data.histories[lane] + step.first;
    const StoredKv kv = stored(data, lane, windowStart(data.variant, position), position + step.rows);
    for (const uint32_t row : checkedRows(position, step.rows))
      for (uint32_t head = 0; head < kQueryHeads; ++head) {
        const Reference expected = reference(data, step, kv, lane, row, head);
        for (uint32_t d = 0; d < kHeadDimension; ++d) {
          const double gate = 1 / (1 + std::exp(-double(projection(lane, step.first + row,
                                                                   kGateOffset + head * kHeadDimension + d))));
          const double exact = expected.output[d] * gate;
          const double actual = bf16ToFloat(contents<uint16_t>(step.output)[step.outputIndex(lane, head, row, d)]);
          const double bound =
              gate * (expected.mass[d] / 256 + ulpBf16(float(expected.output[d]))) + ulpBf16(float(exact));
          const double error = std::fabs(actual - exact);
          if (!(std::isfinite(actual) && error <= bound))
            throw std::runtime_error(what + ": lane " + std::to_string(lane) + " row " +
                                     std::to_string(step.first + row) + " head " + std::to_string(head) +
                                     " dimension " + std::to_string(d) + " is " +
                                     std::to_string(actual) + ", the reference " + std::to_string(exact) +
                                     " (bound " + std::to_string(bound) + ")");
          largest = std::max(largest, error / bound);
        }
      }
  }
  return largest;
}

std::string label(kv::Format format, bool int8Queries) {
  return std::string(kv::formatName(format)) + (int8Queries ? " with INT8 queries" : "");
}
std::string describe(const Case &data, std::string_view phase) {
  std::string text = std::string(phase) + " " + label(data.layout.format, data.int8Queries) +
                     (data.variant.windowTokens ? " window " + std::to_string(data.variant.windowTokens)
                                                : " whole history, position-scaled");
  for (uint32_t lane = 0; lane < data.lanes; ++lane) text += " " + std::to_string(data.histories[lane]);
  return text;
}

// A prefill of `rows` rows after a cached prefix of `history` tokens, as one
// chunk, and again as the chunks `split` lists, each over the K/V its
// predecessors stored: its prepare and its every checked row.
void checkPrefill(metal::MetalBackend &backend, kv::Format format, bool int8Queries,
                  ops::RelativeAttentionVariant variant, uint32_t history, uint32_t rows,
                  std::vector<uint32_t> split = {}) {
  const Case data = makeCase(backend, format, int8Queries, variant, 1, rows, {history, 0, 0, 0});
  const std::string what = describe(data, "prefill") + " rows " + std::to_string(rows);
  double largest = 0;
  split.insert(split.begin(), rows);
  for (const uint32_t chunk : split)
    for (uint32_t first = 0; first < rows; first += chunk) {
      const uint32_t chunkRows = std::min(chunk, rows - first);
      const Step step =
          makeStep(backend, data, first, chunkRows, (chunkRows + kPageTokens - 1) / kPageTokens * kPageTokens);
      run(backend, data, step, planOf(data, step));
      checkPrepare(data, step, what);
      largest = std::max(largest, checkOutput(data, step, what + " in chunks of " + std::to_string(chunk)));
    }
  std::cout << what << (split.size() > 1 ? " and in chunks" : "") << ": " << data.poisoned
            << " table entries of NaNs below the window, largest error " << largest << " of its bound PASS\n";
}

// A verify step of chains of a lane's eight rows each, one to a batch of
// lanes, each lane's history its own.
void checkVerify(metal::MetalBackend &backend, kv::Format format, bool int8Queries,
                 ops::RelativeAttentionVariant variant, std::span<const uint32_t> histories) {
  std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> lanes{};
  std::copy(histories.begin(), histories.end(), lanes.begin());
  const Case data = makeCase(backend, format, int8Queries, variant, uint32_t(histories.size()), kLaneRows, lanes);
  const Step step = makeStep(backend, data, 0, kLaneRows, kPageTokens);
  run(backend, data, step, planOf(data, step));
  const std::string what = describe(data, "verify");
  checkPrepare(data, step, what);
  std::cout << what << ": largest error " << checkOutput(data, step, what) << " of its bound PASS\n";
}

// A verify of lanes whose blocks of two tiles' rows are trees: a chain, a
// branch off the third row that runs through the second tile, two children a
// row, and a random tree, so that a row's ancestors sit at the slots of rows
// other than their depths and its bias takes their depths' distances, and its
// log scale its depth's position.
void checkTreeVerify(metal::MetalBackend &backend, kv::Format format, bool int8Queries,
                     ops::RelativeAttentionVariant variant, std::span<const uint32_t> histories) {
  constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
  static_assert(kRows == 2 * kv::kAttentionTileRows, "a block of two tiles");
  std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> lanes{};
  std::copy(histories.begin(), histories.end(), lanes.begin());
  const auto count = uint32_t(histories.size());
  const Case data = makeCase(backend, format, int8Queries, variant, count, kRows, lanes);
  std::vector<Tree> trees(count);
  for (uint32_t lane = 0; lane < count; ++lane)
    for (uint32_t row = 0; row < kRows; ++row) {
      const uint32_t back = std::min(row, 1 + uint32_t((unit(seedOf(8, lane, row, 0)) + 1) * 2) % 4);
      trees[lane].parent.push_back(!row            ? 0
                                   : lane % 4 == 0 ? row - 1
                                   : lane % 4 == 1 ? (row == 8 ? 2 : row - 1)
                                   : lane % 4 == 2 ? (row - 1) / 2
                                                   : row - back);
    }
  const Step step = makeStep(backend, data, 0, kRows, kPageTokens, trees);
  run(backend, data, step, planOf(data, step));
  const std::string what = describe(data, "tree verify");
  checkPrepare(data, step, what);
  std::cout << what << ": largest error " << checkOutput(data, step, what) << " of its bound PASS\n";
}

// Split and reduce derive the same pages for any split count, so every count
// the lane workspace holds gives the reference: a decode lane of each count,
// encoded as addAttention encodes it with the count set.
void checkSplitCounts(metal::MetalBackend &backend, kv::Format format, bool int8Queries,
                      ops::RelativeAttentionVariant variant, uint32_t history) {
  const Case data = makeCase(backend, format, int8Queries, variant, 1, kLaneRows, {history, 0, 0, 0});
  const Step step = makeStep(backend, data, 0, kLaneRows, kPageTokens);
  const ops::RelativeAttentionPlan plan = planOf(data, step);
  run(backend, data, step, plan);
  const ops::AttentionWorkspace workspace = RelativeAttention::workspace(kv::kAttentionMaximumSplits, data.layout);
  const metal::MetalBuffer partials = poisoned(backend, workspace.partialsBytes);
  const metal::MetalBuffer statistics = poisoned(backend, workspace.statisticsBytes);
  const kv::ChunkedPrefillParams sequence = sequencesOf(data, step)[0];
  for (uint32_t splits = 1; splits <= kv::kAttentionMaximumSplits; ++splits) {
    std::memset(partials.contents(), 0xff, partials.sizeBytes());
    std::memset(statistics.contents(), 0xff, statistics.sizeBytes());
    std::memset(step.output.contents(), 0xff, step.output.sizeBytes());
    SplashAttentionParams params{.sequences = 1, .first_sequence = 0, .rows = kLaneRows, .tiles = 1, .trees = 0,
                                 .last_rows = 0, .kv = data.layer, .sequence = {}};
    params.sequence[0] = {.committed_tokens = sequence.committed_tokens, .row_bound = kLaneRows, .layout_row = 0,
                          .stage_row = 0, .chunk_stride = kPageTokens,
                          .page_table_entries = sequence.page_table_entries, .split_count = splits,
                          .first_tile = 0, .first_slot = 0, .block_stride = 0, .block_list = 0};
    metal::CommandGraph graph;
    std::vector<metal::MetalBuffer> split{step.staging.queries, partials, statistics};
    split.insert(split.end(), data.tables.begin(), data.tables.end());
    split.insert(split.end(), {partials, step.layout, step.staging.states, data.weights.bias});
    graph.add(plan.splitPipeline(), std::move(split), params, {kKvHeads, 1, splits});
    graph.add(plan.reducePipeline(),
              {partials, statistics, step.inputs.projection, step.output, step.output, step.output, step.layout,
               step.staging.sinks},
              params, {kKvHeads, kLaneRows * kGroup, 1}, {kHeadDimension, 1, 1});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    checkOutput(data, step, describe(data, "lane") + " of " + std::to_string(splits) + " splits");
  }
  std::cout << "split counts: " << describe(data, "lane") << " of 1-" << kv::kAttentionMaximumSplits
            << " splits PASS\n";
}

// Where every row's window holds its whole history, the window changes
// nothing: the windowed kernels' outputs are the whole history's bit for bit
// (both unscaled), at the same split partition.
void checkWindowWithinHistory(metal::MetalBackend &backend, kv::Format format, bool int8Queries) {
  const auto outputs = [&](ops::RelativeAttentionVariant variant, bool verify, uint32_t lanes, uint32_t rows,
                           std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> histories) {
    const Case data = makeCase(backend, format, int8Queries, variant, lanes, rows, histories);
    const Step step = makeStep(backend, data, 0, rows, verify ? kPageTokens : (rows + 31) / 32 * 32);
    run(backend, data, step, planOf(data, step));
    const auto *begin = contents<uint16_t>(step.output);
    return std::vector<uint16_t>(begin, begin + step.output.sizeBytes() / 2);
  };
  constexpr ops::RelativeAttentionVariant whole{.windowTokens = 0, .positionScaled = false};
  constexpr std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> kWithin{0, 31, 500, 1015, 1, 300, 700, 1008};
  require(outputs(kSliding, true, kWithin.size(), kLaneRows, kWithin) ==
              outputs(whole, true, kWithin.size(), kLaneRows, kWithin),
          "a windowed verify inside its window differs from the whole history's");
  require(outputs(kSliding, false, 1, 600, {0, 0, 0, 0}) == outputs(whole, false, 1, 600, {0, 0, 0, 0}) &&
              outputs(kSliding, false, 1, 24, {900, 0, 0, 0}) == outputs(whole, false, 1, 24, {900, 0, 0, 0}),
          "a windowed prefill inside its window differs from the whole history's");
  std::cout << "windows within the history: " << label(format, int8Queries)
            << " bit for bit the whole history's PASS\n";
}

// The plans' kernels, and the layouts and windows they refuse.
void checkPlans() {
  const std::array sequence{ops::PagedAttention::chunkParams(5000, 300, 320, (5300 + 31) / 32)};
  const std::array lane{ops::PagedAttention::chunkParams(5000, kLaneRows, kPageTokens, (5008 + 31) / 32)};
  for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16}) {
    const kv::Layout layout = layoutOf(format);
    const std::string name = format == kv::Format::Int8 ? "q8" : "bf16";
    const ops::AttentionWorkspace chunk = RelativeAttention::workspace(ops::kAttentionSequenceSplits, layout);
    for (const auto variant : {kSliding, kFull}) {
      const std::string window = variant.windowTokens ? "_w1023" : "";
      const ops::RelativeAttentionPlan plan = RelativeAttention::plan(
          sequence, layout, test::attentionRow(ops::kNever), ops::Int8Activations::On, chunk, variant);
      require(plan.splitPipeline() == "relative_attention_" + name + "_split" + window &&
                  plan.reducePipeline() == "relative_attention_reduce" + window,
              "a plan names another variant's kernels");
      const ops::RelativeAttentionPlan codes = RelativeAttention::plan(
          lane, layout, test::attentionRow(1), ops::Int8Activations::On, chunk, variant);
      require(codes.splitPipeline() ==
                  "relative_attention_" + std::string(format == kv::Format::Int8 ? "q8q" : "bf16") + "_split" + window,
              "INT8 queries score INT8 pages alone");
      const ops::RelativeAttentionPlan last = RelativeAttention::lastTiles(plan);
      require(last.sequences()[0].firstTile + 1 == kv::attentionTiles(300), "the last tiles are not the last");
    }
  }
  kv::Layout wider = layoutOf(kv::Format::Int8);
  wider.kvHeads = 4;
  rejects([&] { static_cast<void>(RelativeAttention::workspace(1, wider)); }, "no relative attention kernel",
          "a layout of other heads was accepted");
  rejects(
      [&] {
        static_cast<void>(RelativeAttention::plan(sequence, layoutOf(kv::Format::Int8), test::attentionRow(1),
                                                  ops::Int8Activations::On,
                                                  RelativeAttention::workspace(256, layoutOf(kv::Format::Int8)),
                                                  {.windowTokens = 512}));
      },
      "no relative attention kernel", "another window was accepted");
  std::cout << "plans: kernels of each variant, and refusals PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: relative-attention METALLIB");
    metal::MetalBackend backend(argv[1]);
    checkPlans();
    for (const auto &[format, int8Queries] : {std::pair{kv::Format::Int8, false}, std::pair{kv::Format::Int8, true},
                                              std::pair{kv::Format::BFloat16, false}}) {
      // Sliding layers: prompt chunks from position 0 and after cached
      // prefixes on either side of the window (its start's page entries of
      // NaNs), whose windows and extents coincide; a chunk again as several.
      checkPrefill(backend, format, int8Queries, kSliding, 0, 2048);
      for (const uint32_t history : {31U, 1000U, 1023U, 1024U, 1025U, 1056U})
        for (const uint32_t rows : {1U, 8U, 33U})
          checkPrefill(backend, format, int8Queries, kSliding, history, rows);
      checkPrefill(backend, format, int8Queries, kSliding, 4103, 1100, {509});
      checkPrefill(backend, format, int8Queries, kSliding, 1000, 13, {1, 3});
      // Full layers: from position 0 past the extent, after prefixes that
      // straddle it, and past 8192 positions, where the log scale grows.
      checkPrefill(backend, format, int8Queries, kFull, 0, 1300, {257});
      checkPrefill(backend, format, int8Queries, kFull, 1010, 40, {7});
      checkPrefill(backend, format, int8Queries, kFull, 8180, 100, {31});
      checkPrefill(backend, format, int8Queries, kFull, 20000, 17);
      // Verify chains of one to four lanes and trees of four, before, at and
      // past the window and the extent, and past 8192.
      for (const auto variant : {kSliding, kFull}) {
        constexpr std::array<uint32_t, 4> kHistories{0, 1020, 1031, 9000};
        for (uint32_t lanes = 1; lanes <= 4; ++lanes)
          checkVerify(backend, format, int8Queries, variant, std::span(kHistories).first(lanes));
        checkTreeVerify(backend, format, int8Queries, variant, std::array<uint32_t, 4>{3, 1015, 2000, 8190});
        checkSplitCounts(backend, format, int8Queries, variant, 4103);
      }
      checkWindowWithinHistory(backend, format, int8Queries);
    }
    std::cout << "relative attention PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "relative attention FAIL: " << error.what() << "\n";
    return 1;
  }
}
