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
// hold; a decode step's single pass, against fp64 of the K/V it stores and
// of its rows over the K/V the pages hold from what the prepare stages (the
// bits it stages itself), and bit for bit against the prepare, split and
// reduce at its splits.

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
// The GPU cores a decode step's splits fill (AttentionCalibration::cores):
// a 20-core M5 Pro's.
constexpr uint32_t kCores = 20;
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
    ops::AttentionCalibration row = test::attentionRow(int8Queries ? 1U : ops::kNever);
    row.cores = kCores;
    return row;
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
// rows. A lane's layout entry may hold fewer rows than its bound (laneRows,
// as a decode step's proposer writes them): the rest of its packed rows are
// in no lane.
struct Step final {
  uint32_t first;
  uint32_t rows;
  uint32_t stride;
  std::vector<Tree> trees;
  std::vector<uint32_t> laneRows;
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
              std::vector<Tree> trees = {}, std::vector<uint32_t> laneRows = {}) {
  const uint64_t packed = uint64_t{data.lanes} * rows;
  if (laneRows.empty()) laneRows.assign(data.lanes, rows);
  Step step{first, rows, stride, std::move(trees), std::move(laneRows), {}, {}, {}, {}, {}};
  step.inputs = {poisoned(backend, packed * kStride * 2), zeroed(backend, packed * kKvHeads * kHeadDimension * 2),
                 zeroed(backend, packed * kKvHeads * kHeadDimension * 2)};
  const ops::RelativeAttentionStagingBytes bytes =
      RelativeAttention::stagingBytes(sequencesOf(data, step), data.layout);
  step.staging = {poisoned(backend, bytes.queries), poisoned(backend, bytes.states), poisoned(backend, bytes.sinks)};
  step.output = zeroed(backend, packed * kQueryHeads * kHeadDimension * 2);
  step.layout = zeroed(backend, uint64_t{data.lanes} * sizeof(SplashLaneRows));
  if (!step.trees.empty()) step.tree = zeroed(backend, packed * sizeof(SplashTreeRow));
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    contents<SplashLaneRows>(step.layout)[lane] = {lane * rows, step.laneRows[lane]};
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
  require(dispatches[0].pipelineName ==
              "relative_attention_" + format + "_prepare" + (data.variant.positionScaled ? "_scaled" : ""),
          "the prepare names another variant's kernel");
  // A dispatch of one split a sequence runs the direct kernel, any other the
  // split and the reduce.
  size_t at = 1;
  for (const ops::RelativeAttentionPlan::Dispatch &dispatch : plan.dispatches()) {
    const bool direct = plan.direct(dispatch);
    require(direct == (dispatch.splits == 1) && dispatches.size() >= at + (direct ? 1 : 2) &&
                (direct ? dispatches[at].pipelineName == plan.directPipeline()
                        : dispatches[at].pipelineName == plan.splitPipeline() &&
                              dispatches[at + 1].threadsPerThreadgroup.x == kHeadDimension),
            "an attention dispatch runs other kernels than its splits'");
    at += direct ? 1 : 2;
  }
  require(dispatches.size() == at, "an attention runs more dispatches than its plan's");
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    require(plan.sequences()[lane].splits == test::attentionSplits(step.rows, data.histories[lane] + step.first,
                                                                   false, data.variant.windowTokens),
            "a lane's splits are not the paged attention's");
  static_cast<void>(backend.submitCommandAsync(dispatches).wait());
}

// A decode step's plan of a case's lanes, with the runtime's lane workspaces.
ops::RelativeAttentionPlan decodePlanOf(const Case &data, const Step &step) {
  const ops::AttentionWorkspace lane = RelativeAttention::laneWorkspace(data.layout);
  return RelativeAttention::plan(sequencesOf(data, step), data.layout, data.attention(), ops::Int8Activations::On,
                                 {lane.partialsBytes * data.lanes, lane.statisticsBytes * data.lanes}, data.variant,
                                 ops::StepKind::Decode);
}

// A lane's single-pass splits: decodeSplits of the pages its tile reads.
uint32_t decodeLaneSplits(const Case &data, const Step &step, uint32_t lane) {
  const uint32_t pages = (data.histories[lane] + step.first + step.rows + kPageTokens - 1) / kPageTokens;
  return RelativeAttention::decodeSplits(
      data.variant.windowTokens ? std::min(pages, kv::windowAttentionPages(data.variant.windowTokens)) : pages,
      step.rows, data.attention().cores);
}

// A decode step's single pass on a poisoned workspace and zero counters, as
// the runtime encodes it: one dispatch of the plan's kernel, which leaves
// its counters zero.
void runDecode(metal::MetalBackend &backend, const Case &data, const Step &step,
               const ops::RelativeAttentionPlan &plan) {
  const ops::AttentionWorkspace bytes = plan.workspace();
  const ops::RelativeAttentionWorkspace workspace{poisoned(backend, bytes.partialsBytes),
                                                  poisoned(backend, bytes.statisticsBytes),
                                                  zeroed(backend, RelativeAttention::kCountersBytes)};
  metal::CommandGraph graph;
  RelativeAttention::addDecode(graph, plan, step.inputs, data.weights, data.layer,
                               std::span(data.tables).first(data.lanes), sequencesOf(data, step),
                               {step.layout, data.lanes * step.rows, step.tree}, workspace,
                               {step.inputs.projection, step.output});
  require(graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == plan.decodePipeline(),
          "a single pass runs another kernel than its plan's");
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    require(plan.sequences()[lane].splits == decodeLaneSplits(data, step, lane), "a lane's splits are not its own");
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *counters = contents<uint32_t>(workspace.counters);
  require(std::all_of(counters, counters + RelativeAttention::kCountersBytes / 4, [](uint32_t c) { return !c; }),
          "a single pass left a counter it arrived at");
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
    for (uint32_t row = 0; row < step.laneRows[lane]; ++row) {
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
    const StoredKv kv = stored(data, lane, windowStart(data.variant, position), position + step.laneRows[lane]);
    for (const uint32_t row : checkedRows(position, step.laneRows[lane]))
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

// The direct kernel writes the split and reduce's bits: a lane's prompt chunk
// whose plan runs it (one split), after its prepare, attended by it through
// addAttention and by the split and the reduce of the paged attention's tiles
// of 8 rows at the same split (as checkSplitCounts encodes them), each into
// plain output rows: the same output bits. It writes no out-projection table
// (such a chunk's out-projection reads plain rows): addAttention refuses a
// table input for it. Returns whether the plan runs it.
bool checkDirectBits(metal::MetalBackend &backend, const Case &data, const Step &step,
                     const ops::RelativeAttentionPlan &plan, const std::string &what) {
  if (data.lanes != 1 || plan.dispatches().size() != 1 || !plan.direct(plan.dispatches()[0])) return false;
  const std::vector<kv::ChunkedPrefillParams> sequences = sequencesOf(data, step);
  const ops::AttentionRows rows{step.layout, step.rows, step.tree};
  const std::vector<ops::AttentionHistory> histories{{data.layer, data.tables[0]}};
  constexpr uint32_t width = kQueryHeads * kHeadDimension;
  const ops::AttentionWorkspace workspace = plan.workspace();
  const auto attend = [&](bool direct) {
    const metal::MetalBuffer partials = poisoned(backend, workspace.partialsBytes);
    const metal::MetalBuffer statistics = poisoned(backend, workspace.statisticsBytes);
    const metal::MetalBuffer hidden = poisoned(backend, uint64_t{step.rows} * width * 2);
    metal::CommandGraph graph;
    if (direct) {
      RelativeAttention::addAttention(graph, plan, step.staging, data.weights, partials, statistics, sequences,
                                      histories, rows,
                                      {step.inputs.projection, hidden, ops::AttentionOutput::Every, {},
                                       ops::LinearInput::Plain});
      require(graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == plan.directPipeline(),
              what + ": a chunk of one split runs other kernels than the direct one");
    } else {
      SplashAttentionParams params{.sequences = 1, .first_sequence = 0, .rows = step.rows,
                                   .tiles = kv::attentionTiles(step.rows), .trees = 0, .last_rows = 0,
                                   .kv = data.layer, .sequence = {}};
      params.sequence[0] = {.committed_tokens = sequences[0].committed_tokens, .row_bound = step.rows,
                            .layout_row = 0, .stage_row = 0, .chunk_stride = step.stride,
                            .page_table_entries = sequences[0].page_table_entries, .split_count = 1,
                            .first_tile = 0, .first_slot = 0, .block_stride = 0, .block_list = 0};
      std::vector<metal::MetalBuffer> split{step.staging.queries, partials, statistics};
      split.insert(split.end(), data.tables.begin(), data.tables.end());
      split.insert(split.end(), {partials, step.layout, step.staging.states, data.weights.bias});
      graph.add(plan.splitPipeline(), std::move(split), params, {kKvHeads, kv::attentionTiles(step.rows), 1});
      graph.add(plan.reducePipeline(),
                {partials, statistics, step.inputs.projection, hidden, hidden, hidden, step.layout,
                 step.staging.sinks},
                params, {kKvHeads, kv::kAttentionTileRows * kGroup, kv::attentionTiles(step.rows)},
                {kHeadDimension, 1, 1});
    }
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *begin = contents<std::byte>(hidden);
    return std::vector<std::byte>(begin, begin + hidden.sizeBytes());
  };
  {
    const metal::MetalBuffer partials = poisoned(backend, workspace.partialsBytes);
    const metal::MetalBuffer statistics = poisoned(backend, workspace.statisticsBytes);
    const metal::MetalBuffer hidden = poisoned(backend, uint64_t{step.rows} * width * 2);
    const metal::MetalBuffer table =
        poisoned(backend, ops::tableBytes(width, uint64_t{ops::tableTiles(step.rows)} * GGUF_TABLE16_ROWS));
    const metal::MetalBuffer sums = poisoned(backend, ops::tableSumsBytes(ops::LinearInput::Table16, width, step.rows));
    metal::CommandGraph graph;
    bool refused = false;
    try {
      RelativeAttention::addAttention(graph, plan, step.staging, data.weights, partials, statistics, sequences,
                                      histories, rows,
                                      {step.inputs.projection, hidden, ops::AttentionOutput::Every,
                                       {table, sums, {}, {}}, ops::LinearInput::Table16});
    } catch (const std::invalid_argument &) {
      refused = true;
    }
    require(refused && graph.dispatches().empty(), what + ": the direct kernel took the out-projection's table");
  }
  const std::vector<std::byte> direct = attend(true), split = attend(false);
  if (direct == split) return true;
  // The first output element that differs, and how many do.
  std::string where;
  size_t differ = 0;
  for (size_t at = 0; at + 1 < direct.size(); at += 2)
    if (std::memcmp(&direct[at], &split[at], 2)) {
      if (!differ++) {
        uint16_t a = 0, b = 0;
        std::memcpy(&a, &direct[at], 2);
        std::memcpy(&b, &split[at], 2);
        const size_t element = at / 2;
        where = " (row " + std::to_string(element / width) + " head " +
                std::to_string(element % width / kHeadDimension) + " dimension " +
                std::to_string(element % kHeadDimension) + ": " + std::to_string(bf16ToFloat(a)) + " against " +
                std::to_string(bf16ToFloat(b)) + ")";
      }
    }
  throw std::runtime_error(what + ": the direct kernel wrote other bits than the split and reduce, " +
                           std::to_string(differ) + " output elements" + where);
}

// A prefill of `rows` rows after a cached prefix of `history` tokens, as one
// chunk, and again as the chunks `split` lists, each over the K/V its
// predecessors stored: its prepare and its every checked row, and the bits of
// each chunk of one split (checkDirectBits).
void checkPrefill(metal::MetalBackend &backend, kv::Format format, bool int8Queries,
                  ops::RelativeAttentionVariant variant, uint32_t history, uint32_t rows,
                  std::vector<uint32_t> split = {}) {
  const Case data = makeCase(backend, format, int8Queries, variant, 1, rows, {history, 0, 0, 0});
  const std::string what = describe(data, "prefill") + " rows " + std::to_string(rows);
  double largest = 0;
  uint32_t direct = 0, chunks = 0;
  split.insert(split.begin(), rows);
  for (const uint32_t chunk : split)
    for (uint32_t first = 0; first < rows; first += chunk) {
      const uint32_t chunkRows = std::min(chunk, rows - first);
      const Step step =
          makeStep(backend, data, first, chunkRows, (chunkRows + kPageTokens - 1) / kPageTokens * kPageTokens);
      const ops::RelativeAttentionPlan plan = planOf(data, step);
      const std::string at = what + " in chunks of " + std::to_string(chunk);
      run(backend, data, step, plan);
      checkPrepare(data, step, what);
      largest = std::max(largest, checkOutput(data, step, at));
      direct += checkDirectBits(backend, data, step, plan, at + " from row " + std::to_string(first));
      ++chunks;
    }
  std::cout << what << (split.size() > 1 ? " and in chunks" : "") << ": " << data.poisoned
            << " table entries of NaNs below the window, largest error " << largest << " of its bound, " << direct
            << " of " << chunks << " chunks direct, the split and reduce's bits PASS\n";
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

// Lanes' blocks of `rows` rows as trees, lane by lane in turn: a chain, a
// branch off the third row that runs through the second tile, two children a
// row, and a random tree, so that a row's ancestors sit at the slots of rows
// other than their depths and its bias takes their depths' distances, and its
// log scale its depth's position.
std::vector<Tree> treesOf(uint32_t lanes, uint32_t rows) {
  std::vector<Tree> trees(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < rows; ++row) {
      const uint32_t back = std::min(row, 1 + uint32_t((unit(seedOf(8, lane, row, 0)) + 1) * 2) % 4);
      trees[lane].parent.push_back(!row            ? 0
                                   : lane % 4 == 0 ? row - 1
                                   : lane % 4 == 1 ? (row == 8 ? 2 : row - 1)
                                   : lane % 4 == 2 ? (row - 1) / 2
                                                   : row - back);
    }
  return trees;
}

// A verify of lanes whose blocks of two tiles' rows are trees (treesOf).
void checkTreeVerify(metal::MetalBackend &backend, kv::Format format, bool int8Queries,
                     ops::RelativeAttentionVariant variant, std::span<const uint32_t> histories) {
  constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
  static_assert(kRows == 2 * kv::kAttentionTileRows, "a block of two tiles");
  std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> lanes{};
  std::copy(histories.begin(), histories.end(), lanes.begin());
  const auto count = uint32_t(histories.size());
  const Case data = makeCase(backend, format, int8Queries, variant, count, kRows, lanes);
  const Step step = makeStep(backend, data, 0, kRows, kPageTokens, treesOf(count, kRows));
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

// ---- The single pass -------------------------------------------------------

// Every byte of each lane's block rows' K/V in their pages, which `poison`
// overwrites: NaN values (BF16), or codes of -1 and NaN scales (INT8).
std::vector<std::byte> blockKv(const Case &data, const Step &step, bool poison) {
  std::vector<std::byte> bytes;
  const auto visit = [&](std::byte *at, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      bytes.push_back(at[i]);
      if (poison) at[i] = i % 2 && data.layout.format == kv::Format::BFloat16 ? std::byte{0x7f} : std::byte{0xff};
    }
  };
  const size_t element = data.layout.format == kv::Format::BFloat16 ? 2 : 1;
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    for (uint32_t row = 0; row < step.laneRows[lane]; ++row) {
      const uint32_t token = data.histories[lane] + step.first + row, page = data.page(lane, token);
      const uint32_t place = token % kPageTokens;
      const std::array<std::span<std::byte>, 4> spans = data.pool.layerSpans(page, kLayer);
      for (uint32_t head = 0; head < kKvHeads; ++head) {
        for (uint32_t d = 0; d < kHeadDimension; ++d) {
          visit(spans[SPLASH_KV_KEYS].data() + splash_kv_key_element(head, place, d, kHeadDimension) * element,
                element);
          visit(spans[SPLASH_KV_VALUES].data() + data.pool.valueElement(head, place, d) * element, element);
        }
        if (data.layout.format == kv::Format::Int8)
          for (const uint32_t tensor : {SPLASH_KV_KEY_SCALES, SPLASH_KV_VALUE_SCALES})
            visit(spans[tensor].data() + splash_kv_scale_element(head, place) * sizeof(float), sizeof(float));
      }
    }
  return bytes;
}

// A step's prepare alone, which stages what the reference reads (a single
// pass stages the same bits in its own slots: checkDecodeBits), then its
// block rows' K/V poisoned, which the single pass must store again.
void runPrepare(metal::MetalBackend &backend, const Case &data, const Step &step) {
  metal::CommandGraph graph;
  RelativeAttention::addProjection(graph, step.inputs, data.weights, step.staging, data.layer,
                                   std::span(data.tables).first(data.lanes), sequencesOf(data, step),
                                   {step.layout, data.lanes * step.rows, step.tree}, data.layout, data.variant);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  static_cast<void>(blockKv(data, step, true));
}

// A decode step's single pass (RelativeAttention::addDecode) over lanes of
// `bound` rows a lane (the step's lane cap), each holding laneRows of them as
// a decode step's proposer chooses, chains or trees (treesOf), after the
// prepare (runPrepare): the prepare's staging, the single pass's stores and
// every checked row against fp64, and no row past a lane's rows written.
void checkDecode(metal::MetalBackend &backend, kv::Format format, bool int8Queries,
                 ops::RelativeAttentionVariant variant, std::span<const uint32_t> histories, uint32_t bound,
                 std::span<const uint32_t> laneRows, bool trees) {
  std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> lanes{};
  std::copy(histories.begin(), histories.end(), lanes.begin());
  const auto count = uint32_t(histories.size());
  const Case data = makeCase(backend, format, int8Queries, variant, count, bound, lanes);
  const Step step = makeStep(backend, data, 0, bound, kPageTokens, trees ? treesOf(count, bound) : std::vector<Tree>{},
                             std::vector<uint32_t>(laneRows.begin(), laneRows.end()));
  runPrepare(backend, data, step);
  runDecode(backend, data, step, decodePlanOf(data, step));
  std::string what = describe(data, trees ? "tree decode" : "decode") + " rows";
  for (const uint32_t rows : laneRows) what += " " + std::to_string(rows);
  what += " of " + std::to_string(bound);
  checkPrepare(data, step, what);
  const double largest = checkOutput(data, step, what);
  for (uint32_t lane = 0; lane < count; ++lane)
    for (uint32_t row = step.laneRows[lane]; row < bound; ++row)
      for (uint32_t head = 0; head < kQueryHeads; ++head)
        for (uint32_t d = 0; d < kHeadDimension; ++d)
          require(!contents<uint16_t>(step.output)[step.outputIndex(lane, head, row, d)],
                  what + ": a row past its lane's rows was written");
  std::cout << what << ": largest error " << largest << " of its bound PASS\n";
}

// A lane's block run by the prepare, split and reduce at the single pass's
// splits, as checkSplitCounts encodes them: the split of the same queries'
// format (INT8 codes where the plan scores them).
void runMultiPass(metal::MetalBackend &backend, const Case &data, const Step &step,
                  const ops::RelativeAttentionPlan &decode) {
  const std::vector<kv::ChunkedPrefillParams> sequences = sequencesOf(data, step);
  const ops::AttentionRows rows{step.layout, step.rows, step.tree};
  const uint32_t splits = decode.sequences()[0].splits;
  const ops::AttentionWorkspace workspace = RelativeAttention::workspace(kv::kAttentionMaximumSplits, data.layout);
  const metal::MetalBuffer partials = poisoned(backend, workspace.partialsBytes);
  const metal::MetalBuffer statistics = poisoned(backend, workspace.statisticsBytes);
  SplashAttentionParams params{.sequences = 1, .first_sequence = 0, .rows = step.rows, .tiles = 1,
                               .trees = step.tree ? 1U : 0U, .last_rows = 0, .kv = data.layer, .sequence = {}};
  params.sequence[0] = {.committed_tokens = sequences[0].committed_tokens, .row_bound = step.rows, .layout_row = 0,
                        .stage_row = 0, .chunk_stride = step.stride,
                        .page_table_entries = sequences[0].page_table_entries, .split_count = splits,
                        .first_tile = 0, .first_slot = 0, .block_stride = 0, .block_list = 0};
  const std::string window = data.variant.windowTokens ? "_w1023" : "";
  const std::string format = decode.decodePipeline().starts_with("relative_attention_q8q") ? "q8q"
                             : data.layout.format == kv::Format::Int8                      ? "q8"
                                                                                           : "bf16";
  metal::CommandGraph graph;
  RelativeAttention::addProjection(graph, step.inputs, data.weights, step.staging, data.layer,
                                   std::span(data.tables).first(1), sequences, rows, data.layout, data.variant);
  std::vector<metal::MetalBuffer> split{step.staging.queries, partials, statistics};
  split.insert(split.end(), data.tables.begin(), data.tables.end());
  split.insert(split.end(), {step.tree ? step.tree : partials, step.layout, step.staging.states, data.weights.bias});
  graph.add("relative_attention_" + format + "_split" + window, std::move(split), params, {kKvHeads, 1, splits});
  graph.add("relative_attention_reduce" + window,
            {partials, statistics, step.inputs.projection, step.output, step.output, step.output, step.layout,
             step.staging.sinks},
            params, {kKvHeads, kLaneRows * kGroup, 1}, {kHeadDimension, 1, 1});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

// The single pass computes the prepare, split and reduce's bits: a lane of
// the narrow cap's rows (one tile of 24 fused rows either way), run by both
// at the single pass's splits, stores the same K/V and writes the same
// output, the single pass over its block rows' poisoned slots.
void checkDecodeBits(metal::MetalBackend &backend, kv::Format format, bool int8Queries,
                     ops::RelativeAttentionVariant variant, uint32_t history, uint32_t rows, bool trees) {
  const Case data = makeCase(backend, format, int8Queries, variant, 1, kLaneRows, {history, 0, 0, 0});
  const Step step = makeStep(backend, data, 0, kLaneRows, kPageTokens,
                             trees ? treesOf(1, kLaneRows) : std::vector<Tree>{}, {rows});
  const ops::RelativeAttentionPlan plan = decodePlanOf(data, step);
  runMultiPass(backend, data, step, plan);
  const std::vector<std::byte> stores = blockKv(data, step, true);
  const auto *output = contents<uint16_t>(step.output);
  const std::vector<uint16_t> attended(output, output + step.output.sizeBytes() / 2);
  std::memset(step.output.contents(), 0, step.output.sizeBytes());
  runDecode(backend, data, step, plan);
  const std::string what = describe(data, trees ? "tree decode" : "decode") + " rows " + std::to_string(rows);
  require(blockKv(data, step, false) == stores, what + ": the single pass stored other K/V bits than the prepare");
  require(std::equal(attended.begin(), attended.end(), output),
          what + ": the single pass wrote other bits than the split and reduce at its splits");
  std::cout << what << " at " << plan.sequences()[0].splits
            << " splits: the prepare, split and reduce's bits PASS\n";
}

// The single pass's splits partition the same pages for any count, so each
// count up to SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS gives the reference: a
// lane of `bound` rows of each count, encoded as addDecode encodes it with
// the count set, which leaves its counters zero.
void checkDecodeSplitCounts(metal::MetalBackend &backend, kv::Format format, bool int8Queries,
                            ops::RelativeAttentionVariant variant, uint32_t history, uint32_t bound) {
  constexpr uint32_t kSplits = SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS;
  const Case data = makeCase(backend, format, int8Queries, variant, 1, bound, {history, 0, 0, 0});
  const Step step = makeStep(backend, data, 0, bound, kPageTokens, treesOf(1, bound));
  const ops::RelativeAttentionPlan plan = decodePlanOf(data, step);
  const ops::AttentionWorkspace workspace =
      RelativeAttention::workspace(2 * kv::attentionTiles(bound) * kSplits, data.layout);
  const metal::MetalBuffer partials = poisoned(backend, workspace.partialsBytes);
  const metal::MetalBuffer statistics = poisoned(backend, workspace.statisticsBytes);
  const metal::MetalBuffer counters = zeroed(backend, RelativeAttention::kCountersBytes);
  const kv::ChunkedPrefillParams sequence = sequencesOf(data, step)[0];
  runPrepare(backend, data, step);
  for (uint32_t splits = 1; splits <= kSplits; ++splits) {
    std::memset(step.output.contents(), 0xff, step.output.sizeBytes());
    SplashAttentionParams params{.sequences = 1, .first_sequence = 0, .rows = bound,
                                 .tiles = kv::attentionTiles(bound), .trees = 1, .last_rows = 0, .kv = data.layer,
                                 .sequence = {}};
    params.sequence[0] = {.committed_tokens = sequence.committed_tokens, .row_bound = bound, .layout_row = 0,
                          .stage_row = 0, .chunk_stride = kPageTokens,
                          .page_table_entries = sequence.page_table_entries, .split_count = splits,
                          .first_tile = 0, .first_slot = 0, .block_stride = 0, .block_list = 0};
    std::vector<metal::MetalBuffer> bindings{step.inputs.projection, step.inputs.keys, step.inputs.values,
                                             data.weights.queryScale, data.weights.relativeScale, data.weights.bias,
                                             partials, statistics, counters, step.output, step.output, step.output};
    bindings.insert(bindings.end(), data.tables.begin(), data.tables.end());
    bindings.insert(bindings.end(), {step.tree, step.layout});
    metal::CommandGraph graph;
    graph.add(plan.decodePipeline(), std::move(bindings), params, {kKvHeads, 1, splits});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const std::string what = describe(data, "tree decode") + " of " + std::to_string(splits) + " splits";
    checkOutput(data, step, what);
    const auto *left = contents<uint32_t>(counters);
    require(std::all_of(left, left + RelativeAttention::kCountersBytes / 4, [](uint32_t c) { return !c; }),
            what + ": a counter left nonzero");
  }
  std::cout << "split counts: " << describe(data, "tree decode") << " rows " << bound << " of 1-" << kSplits
            << " splits PASS\n";
}

// Where every row's window holds its whole history, the window changes
// nothing: the windowed kernels' outputs are the whole history's bit for bit
// (both unscaled), at the same split partition. A decode step's single pass
// over the whole history is a full layer's, position-scaled (the variants it
// is compiled for), whose log scale is exactly 1 at these positions (below
// 8,192: relative_attention_log_scale).
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
  // A decode step's single pass of a batch of lanes of a verify block's
  // rows, as trees.
  const auto decoded = [&](ops::RelativeAttentionVariant variant) {
    const Case data = makeCase(backend, format, int8Queries, variant, kWithin.size(), SPLASH_TARGET_VERIFY_ROWS,
                               kWithin);
    const Step step = makeStep(backend, data, 0, SPLASH_TARGET_VERIFY_ROWS, kPageTokens,
                               treesOf(kWithin.size(), SPLASH_TARGET_VERIFY_ROWS));
    runDecode(backend, data, step, decodePlanOf(data, step));
    const auto *begin = contents<uint16_t>(step.output);
    return std::vector<uint16_t>(begin, begin + step.output.sizeBytes() / 2);
  };
  require(decoded(kSliding) == decoded(kFull), "a windowed decode inside its window differs from the whole history's");
  require(outputs(kSliding, false, 1, 600, {0, 0, 0, 0}) == outputs(whole, false, 1, 600, {0, 0, 0, 0}) &&
              outputs(kSliding, false, 1, 24, {900, 0, 0, 0}) == outputs(whole, false, 1, 24, {900, 0, 0, 0}),
          "a windowed prefill inside its window differs from the whole history's");
  std::cout << "windows within the history: " << label(format, int8Queries)
            << " bit for bit the whole history's PASS\n";
}

// The plans' kernels, and the layouts and windows they refuse; and an output
// of each sequence's last row alone, which a prompt chunk's attention and a
// decode step's single pass refuse.
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
                  plan.reducePipeline() == "relative_attention_reduce" + window &&
                  plan.directPipeline() == "relative_attention_" + name + "_direct" + window,
              "a plan names another variant's kernels");
      // 300 rows after 5,000 tokens: a sliding layer's window in one split,
      // whose dispatch runs the direct kernel over 19 tiles of 16 rows, a
      // full layer's history in more.
      require(plan.dispatches().size() == 1 &&
                  plan.direct(plan.dispatches()[0]) == (variant.windowTokens != 0) &&
                  plan.directGroups(plan.dispatches()[0]).x == kKvHeads &&
                  plan.directGroups(plan.dispatches()[0]).y == 19,
              "a plan runs the direct kernel other than for one split");
      const ops::RelativeAttentionPlan codes = RelativeAttention::plan(
          lane, layout, test::attentionRow(1), ops::Int8Activations::On, chunk, variant);
      require(codes.splitPipeline() ==
                      "relative_attention_" + std::string(format == kv::Format::Int8 ? "q8q" : "bf16") + "_split" +
                          window &&
                  codes.directPipeline() ==
                      "relative_attention_" + std::string(format == kv::Format::Int8 ? "q8q" : "bf16") + "_direct" +
                          window,
              "INT8 queries score INT8 pages alone");
    }
    // A decode step's single pass: the kernel of its lane cap (_r16 past the
    // narrow cap's rows), its layer's variant (a sliding layer's window, a
    // full layer's position scale) and queries' format, and a lane's splits
    // of its own pages.
    const ops::AttentionWorkspace lanes = RelativeAttention::laneWorkspace(layout);
    for (const uint32_t cap : {kLaneRows, uint32_t{SPLASH_TARGET_VERIFY_ROWS}})
      for (const uint32_t cores : {kCores, 32U})
        for (const auto variant : {kSliding, kFull}) {
          const std::array steps{ops::PagedAttention::chunkParams(5000, cap, kPageTokens, (5000 + cap + 31) / 32),
                                 ops::PagedAttention::chunkParams(70, cap, kPageTokens, (70 + cap + 31) / 32)};
          const auto decode = [&](uint32_t int8QueriesFromPages) {
            ops::AttentionCalibration row = test::attentionRow(int8QueriesFromPages);
            row.cores = cores;
            return RelativeAttention::plan(steps, layout, row, ops::Int8Activations::On,
                                           {2 * lanes.partialsBytes, 2 * lanes.statisticsBytes}, variant,
                                           ops::StepKind::Decode);
          };
          const std::string suffix = std::string(cap > kLaneRows ? "_r16" : "") +
                                     (variant.windowTokens ? "_w1023" : "") + (variant.positionScaled ? "_scaled" : "");
          const ops::RelativeAttentionPlan plan = decode(ops::kNever);
          require(plan.decodePipeline() == "relative_attention_" + name + "_decode" + suffix &&
                      plan.splitPipeline().empty() && plan.reducePipeline().empty() &&
                      plan.directPipeline().empty() && plan.dispatches().size() == 1 &&
                      !plan.direct(plan.dispatches()[0]),
                  "a decode plan names another kernel");
          for (size_t s = 0; s < steps.size(); ++s) {
            const uint32_t pages = (steps[s].committed_tokens + cap + kPageTokens - 1) / kPageTokens;
            require(plan.sequences()[s].splits ==
                        RelativeAttention::decodeSplits(
                            variant.windowTokens ? std::min(pages, kv::windowAttentionPages(kWindow)) : pages, cap,
                            cores),
                    "a decode lane's splits are not its own pages'");
          }
          require(decode(1).decodePipeline() ==
                      "relative_attention_" + std::string(format == kv::Format::Int8 ? "q8q" : "bf16") + "_decode" +
                          suffix,
                  "a decode plan's INT8 queries score INT8 pages alone");
        }
  }
  // A lane's splits: one per kDecodeCorePages / cores pages, but one a page
  // up to a verify block's most, at most decodeMostSplits, and as many as the
  // runs of pages they make. On 20 cores (most 12 and 7) 9 pages' 7 splits of
  // 2 pages make 5 runs, a window's 34 pages take 9 and a full layer's 65 11
  // runs of 6; on 32 cores (16 and 10) the window 12 runs of 3 and 65 pages
  // 13 of 5.
  constexpr uint32_t wide = SPLASH_TARGET_VERIFY_ROWS;
  const uint32_t window = kv::windowAttentionPages(kWindow);
  require(RelativeAttention::decodeMostSplits(kLaneRows, 20) == 12 &&
              RelativeAttention::decodeMostSplits(wide, 20) == 7 &&
              RelativeAttention::decodeMostSplits(kLaneRows, 32) == 16 &&
              RelativeAttention::decodeMostSplits(wide, 32) == 10 &&
              RelativeAttention::decodeMostSplits(kLaneRows, 1000) == SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS,
          "a decode lane's most splits follow another rule");
  bool pageEach = true;
  for (const uint32_t cores : {20U, 32U})
    for (uint32_t pages = 1; pages <= RelativeAttention::decodeMostSplits(wide, cores); ++pages)
      pageEach = pageEach && RelativeAttention::decodeSplits(pages, kLaneRows, cores) == pages &&
                 RelativeAttention::decodeSplits(pages, wide, cores) == pages;
  require(pageEach && window == 34 && RelativeAttention::decodeSplits(9, kLaneRows, 20) == 5 &&
              RelativeAttention::decodeSplits(window, kLaneRows, 20) == 9 &&
              RelativeAttention::decodeSplits(65, kLaneRows, 20) == 11 &&
              RelativeAttention::decodeSplits(1u << 20, kLaneRows, 20) == 12 &&
              RelativeAttention::decodeSplits(1u << 20, wide, 20) == 7 &&
              RelativeAttention::decodeSplits(window, kLaneRows, 32) == 12 &&
              RelativeAttention::decodeSplits(65, kLaneRows, 32) == 13 &&
              RelativeAttention::decodeSplits(1u << 20, kLaneRows, 32) == 16 &&
              RelativeAttention::decodeSplits(1u << 20, wide, 32) == 10,
          "a decode lane's splits follow another rule");
  rejects(
      [&] {
        static_cast<void>(RelativeAttention::plan(lane, layoutOf(kv::Format::Int8), test::attentionRow(1),
                                                  ops::Int8Activations::On,
                                                  RelativeAttention::workspace(256, layoutOf(kv::Format::Int8)),
                                                  kSliding, ops::StepKind::Decode));
      },
      "a calibration of none", "a decode step was planned for no device's cores");
  // The single pass is compiled for a full layer's variant and a sliding
  // layer's alone: the whole history unscaled and a window position-scaled
  // have no kernel.
  ops::AttentionCalibration calibration = test::attentionRow(1);
  calibration.cores = kCores;
  for (const ops::RelativeAttentionVariant variant :
       {ops::RelativeAttentionVariant{},
        ops::RelativeAttentionVariant{.windowTokens = kWindow, .positionScaled = true}})
    rejects(
        [&] {
          static_cast<void>(RelativeAttention::plan(lane, layoutOf(kv::Format::Int8), calibration,
                                                    ops::Int8Activations::On,
                                                    RelativeAttention::workspace(256, layoutOf(kv::Format::Int8)),
                                                    variant, ops::StepKind::Decode));
        },
        "no relative attention decode kernel", "a decode step of a variant no layer attends was planned");
  for (const uint32_t cap : {kLaneRows + 1, uint32_t{SPLASH_TARGET_VERIFY_ROWS} + 1})
    rejects(
        [&] {
          const std::array steps{ops::PagedAttention::chunkParams(5000, kLaneRows, kPageTokens, 158),
                                 ops::PagedAttention::chunkParams(70, cap, kPageTokens, 4)};
          static_cast<void>(RelativeAttention::plan(
              cap > SPLASH_TARGET_VERIFY_ROWS ? std::span(steps).last(1) : std::span(steps), layoutOf(kv::Format::Int8),
              test::attentionRow(1), ops::Int8Activations::On,
              RelativeAttention::workspace(256, layoutOf(kv::Format::Int8)), kSliding, ops::StepKind::Decode));
        },
        "share a lane cap", "a decode step of lanes of other caps, or past a verify block, was planned");
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
  // Before any buffer: relative attention writes every row.
  const ops::AttentionWorkspace capacity = RelativeAttention::workspace(256, layoutOf(kv::Format::Int8));
  const ops::RelativeAttentionPlan chunk = RelativeAttention::plan(
      sequence, layoutOf(kv::Format::Int8), test::attentionRow(1), ops::Int8Activations::On, capacity, kSliding);
  const ops::RelativeAttentionPlan step = RelativeAttention::plan(
      lane, layoutOf(kv::Format::Int8), calibration, ops::Int8Activations::On, capacity, kSliding,
      ops::StepKind::Decode);
  const ops::AttentionOutputRows last{{}, {}, ops::AttentionOutput::Last};
  metal::CommandGraph graph;
  rejects(
      [&] {
        static_cast<void>(RelativeAttention::addAttention(graph, chunk, {}, {}, {}, {}, sequence, {}, {}, last));
      },
      "writes every row", "a prompt chunk's attention of each sequence's last row was encoded");
  rejects([&] { static_cast<void>(RelativeAttention::addDecode(graph, step, {}, {}, {}, {}, lane, {}, {}, last)); },
          "writes every row", "a decode step's single pass of each sequence's last row was encoded");
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
      // Decode steps' single pass, at the same histories: chains of one to
      // four lanes of the narrow cap's rows, and of a row a lane (a cycle
      // without a draft); trees of the narrow cap and of a verify block's
      // rows, whole and of fewer rows on the device; a batch of eight
      // lanes; each split count; and the prepare, split and reduce's bits.
      for (const auto variant : {kSliding, kFull}) {
        constexpr std::array<uint32_t, 4> kHistories{0, 1020, 1031, 9000};
        constexpr std::array<uint32_t, 4> kTrees{3, 1015, 2000, 8190};
        constexpr std::array<uint32_t, 8> kWide{0, 1020, 1031, 9000, 3, 1015, 2000, 8190};
        constexpr std::array<uint32_t, 8> kEight{8, 8, 8, 8, 8, 8, 8, 8}, kOne{1, 1, 1, 1, 1, 1, 1, 1};
        constexpr std::array<uint32_t, 8> kSixteen{16, 16, 16, 16, 16, 16, 16, 16};
        for (uint32_t lanes = 1; lanes <= 4; ++lanes)
          checkDecode(backend, format, int8Queries, variant, std::span(kHistories).first(lanes), kLaneRows,
                      std::span(kEight).first(lanes), false);
        checkDecode(backend, format, int8Queries, variant, kHistories, kLaneRows, std::span(kOne).first(4), false);
        checkDecode(backend, format, int8Queries, variant, kTrees, kLaneRows, std::array<uint32_t, 4>{8, 1, 5, 3},
                    true);
        checkDecode(backend, format, int8Queries, variant, kTrees, SPLASH_TARGET_VERIFY_ROWS,
                    std::span(kSixteen).first(4), true);
        checkDecode(backend, format, int8Queries, variant, kTrees, SPLASH_TARGET_VERIFY_ROWS,
                    std::array<uint32_t, 4>{9, 1, 16, 12}, true);
        checkDecode(backend, format, int8Queries, variant, kWide, kLaneRows, kEight, true);
        for (const uint32_t cap : {kLaneRows, uint32_t{SPLASH_TARGET_VERIFY_ROWS}})
          checkDecodeSplitCounts(backend, format, int8Queries, variant, 4103, cap);
        checkDecodeBits(backend, format, int8Queries, variant, 1031, kLaneRows, false);
        checkDecodeBits(backend, format, int8Queries, variant, 2000, 1, false);
        checkDecodeBits(backend, format, int8Queries, variant, 8190, 5, true);
        checkDecodeBits(backend, format, int8Queries, variant, 3, kLaneRows, true);
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
