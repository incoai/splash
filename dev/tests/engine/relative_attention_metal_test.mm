#include "HostKvExtents.hpp"
#include "LinearNumerics.hpp"
#include "NormReference.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/KernelCalibration.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

// The relative geometry (ops::PagedAttention, metal/abi/PagedAttention.h):
// 24 query heads over 8 KV heads of 128 dimensions without RoPE, each row's
// prepare from its query side row and its convolved keys and values (clipped
// queries and keys, scaled and clipped relative states, RMS-normalized values,
// tau of the row's position on a position-scaled layer), and its attention
// with each score biased by its key's distance, each row's own sink and its
// output gates. A window layer attends the 1023 keys before each row and
// itself, unscaled; a global layer its whole history, scaled. Each prepare's
// outputs against fp64, and each attention through the production encoders
// against an fp64 reference of the queries, relative states and K/V the
// prepare wrote, in INT8 pages (with bf16 queries and INT8 ones) and BF16
// pages; and the whole of it against an fp64 reference of the inputs, whose
// largest errors each check reports.

namespace {

using namespace splash;
using test::HostKvExtents;
using test::bf16ToFloat;
using test::floatToBf16;
using test::rejects;
using test::require;
using test::ulpBf16;

constexpr uint32_t kQueryHeads = 24, kKvHeads = 8, kGroup = kQueryHeads / kKvHeads, kHeadDimension = 128;
constexpr uint32_t kWidth = ops::kRelativeWidth, kExtent = ops::kRelativeExtent;
constexpr uint32_t kSide = ops::relativeQuerySideWidth(kQueryHeads, kHeadDimension);
constexpr uint32_t kGates = SPLASH_RELATIVE_GATES(kQueryHeads, kHeadDimension);
constexpr uint32_t kStates = SPLASH_RELATIVE_STATES(kQueryHeads, kHeadDimension);
constexpr uint32_t kSinks = SPLASH_RELATIVE_SINKS(kQueryHeads, kHeadDimension);
constexpr uint32_t kKeyWidth = kKvHeads * kHeadDimension;
constexpr uint32_t kWindow = 1023, kPageTokens = kv::kPageTokens;
static_assert(kSide == 6592 && kGates == 3072 && kStates == 6144 && kSinks == 6528,
              "the query side row of Inkling 2's attention");
// The layer under test is the second of a pool's two.
constexpr uint32_t kLayer = 1;
constexpr ops::AttentionVariant kWindowed{.rotated = false, .windowTokens = kWindow, .positionScaled = false};
constexpr ops::AttentionVariant kGlobal{.rotated = false, .windowTokens = 0, .positionScaled = true};
constexpr double kQkClip = 22.627416997969522, kStateClip = 8.0, kEpsilon = 1e-6;

// A value in [-1, 1) that depends on every bit of `seed`.
float unit(uint64_t seed) {
  seed += 0x9e3779b97f4a7c15ULL;
  seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
  seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
  seed ^= seed >> 31;
  return float(seed >> 40) / float(1U << 23) - 1.0f;
}

uint64_t seedOf(uint32_t kind, uint32_t lane, uint32_t head, uint32_t token, uint32_t dimension) {
  return ((((uint64_t{kind} * 5 + lane) * 64 + head) * 32768 + token) << 8) + dimension;
}

metal::MetalBuffer filled(metal::MetalBackend &backend, uint64_t bytes, int byte) {
  metal::MetalBuffer buffer = test::sharedBuffer(backend, std::max<uint64_t>(bytes, 1));
  std::memset(buffer.contents(), byte, buffer.sizeBytes());
  return buffer;
}
metal::MetalBuffer zeroed(metal::MetalBackend &backend, uint64_t bytes) { return filled(backend, bytes, 0); }
// NaNs, where a read past what a kernel should read shows.
metal::MetalBuffer poisoned(metal::MetalBackend &backend, uint64_t bytes) { return filled(backend, bytes, 0xff); }

template <class T> T *contents(const metal::MetalBuffer &buffer) { return static_cast<T *>(buffer.contents()); }

kv::Layout layoutOf(kv::Format format, uint32_t layers = 1) {
  return {.attentionLayers = layers, .kvHeads = kKvHeads, .headDimension = kHeadDimension, .format = format};
}

// clip_c(x) = x min(1, c / (|x| + 1e-6)) of a head's vector, in place.
void clip(std::span<double> vector, double threshold) {
  double squares = 0;
  for (const double value : vector) squares += value * value;
  const double factor = std::min(1.0, threshold / (std::sqrt(squares) + kEpsilon));
  for (double &value : vector) value *= factor;
}

double tauOf(uint32_t position) {
  return 1.0 + 0.1 * std::log(std::max(1.0, (double(position) + 1) / 8192.0));
}

// A layer's F32 weights: the query scales [head][dimension], the relative
// states' [head][kWidth], and the distance table of kExtent rows, past which
// it holds a row of NaNs that a read past the extent would add.
struct Weights final {
  metal::MetalBuffer queryScale, relativeScale, table;
};

Weights makeWeights(metal::MetalBackend &backend, uint32_t seed) {
  Weights weights{zeroed(backend, kQueryHeads * kHeadDimension * 4), zeroed(backend, kQueryHeads * kWidth * 4),
                  poisoned(backend, (kExtent + 1) * kWidth * 4)};
  for (uint32_t i = 0; i < kQueryHeads * kHeadDimension; ++i)
    contents<float>(weights.queryScale)[i] = 1.25f + unit(seedOf(10, seed, 0, 0, 0) + i);
  for (uint32_t i = 0; i < kQueryHeads * kWidth; ++i)
    contents<float>(weights.relativeScale)[i] = 1.75f + 1.25f * unit(seedOf(11, seed, 0, 0, 0) + i);
  for (uint32_t i = 0; i < kExtent * kWidth; ++i)
    contents<float>(weights.table)[i] = 0.5f * unit(seedOf(12, seed, 0, 0, 0) + i);
  return weights;
}

// The sink logits of a query head: below, about and far above the scores,
// where the sink takes almost all of the row's weight (60), with a little of
// each row's own.
constexpr std::array kSinkLevels{-9.0f, 0.0f, 6.0f, 60.0f};

// A current row's inputs, by lane and absolute row (its position past the
// lane's history): its query side row, F32 (the pad NaN), and its keys and
// values after their convolutions. The scales of q, r and k span the clips'
// thresholds, so that some heads clip and some do not.
float queryInput(uint32_t lane, uint32_t row, uint32_t at) {
  if (at < kGates) return (1.5f + 0.5f * float(at / kHeadDimension % 3)) * unit(seedOf(0, lane, 0, row, at));
  if (at < kStates) return 3.0f * unit(seedOf(1, lane, 0, row, at));
  if (at < kSinks) return 2.0f * unit(seedOf(2, lane, 0, row, at));
  if (at < kSinks + kQueryHeads)
    return kSinkLevels[(at - kSinks) % kSinkLevels.size()] + 0.5f * unit(seedOf(3, lane, 0, row, at));
  return std::numeric_limits<float>::quiet_NaN();
}
float keyInput(uint32_t lane, uint32_t row, uint32_t at) {
  return (1.0f + 1.5f * float(at / kHeadDimension % 3)) * unit(seedOf(4, lane, 0, row, at));
}
float valueInput(uint32_t lane, uint32_t row, uint32_t at) {
  return 0.25f + 2.0f * unit(seedOf(5, lane, 0, row, at));
}

// A verify block's rows as a tree (metal/abi/VerifyTree.h): each row's
// parent before it, the anchor's none.
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

// One layer's attention of `lanes` lanes after their histories, each lane's
// pages mixed over the extents of a pool of two layers, its history written
// into them directly as a prepare stores it (clipped keys, normalized
// values). With a window, the entries of a lane's table below its window
// name a page of NaNs instead, so that a read through one shows.
struct Case final {
  kv::Layout layout;
  ops::AttentionVariant variant;
  bool int8Queries;
  uint32_t lanes;
  std::array<uint32_t, 4> histories;
  // Each lane's rows a tree's, or none (a prompt's rows, causal).
  std::array<Tree, 4> trees;
  bool tree = false;
  HostKvExtents pool;
  SplashKvLayer layer;
  std::array<std::vector<uint32_t>, 4> pages;
  std::array<metal::MetalBuffer, 4> tables;
  uint32_t poisoned = 0;
  Weights weights;

  [[nodiscard]] uint32_t page(uint32_t lane, uint32_t token) const { return pages[lane][token / kPageTokens]; }
  [[nodiscard]] ops::AttentionCalibration attention() const {
    ops::AttentionCalibration row = ops::kApple10Attention;
    row.int8QueriesFromPages = int8Queries ? 1U : ops::kNever;
    return row;
  }
  // A lane's stored K/V of a token: a BF16 page's, or an INT8 page's code
  // times its row's scale.
  [[nodiscard]] double stored(uint32_t lane, bool value, uint32_t head, uint32_t token, uint32_t dimension) const {
    const uint32_t id = page(lane, token), place = token % kPageTokens;
    const uint64_t index = value ? pool.valueElement(head, place, dimension)
                                 : splash_kv_key_element(head, place, dimension, kHeadDimension);
    const uint32_t tensor = value ? SPLASH_KV_VALUES : SPLASH_KV_KEYS;
    if (layout.format == kv::Format::BFloat16) return bf16ToFloat(pool.slab<uint16_t>(kLayer, tensor, id)[index]);
    return double(pool.slab<int8_t>(kLayer, tensor, id)[index]) *
           pool.slab<float>(kLayer, value ? SPLASH_KV_VALUE_SCALES : SPLASH_KV_KEY_SCALES,
                            id)[splash_kv_scale_element(head, place)];
  }
  [[nodiscard]] double scale(uint32_t lane, bool value, uint32_t head, uint32_t token) const {
    if (layout.format == kv::Format::BFloat16) return 0;
    return pool.slab<float>(kLayer, value ? SPLASH_KV_VALUE_SCALES : SPLASH_KV_KEY_SCALES,
                            page(lane, token))[splash_kv_scale_element(head, token % kPageTokens)];
  }
};

// Writes one token's key and value of a KV head as the store does.
void storeToken(Case &data, uint32_t lane, uint32_t head, uint32_t token, std::span<const double> key,
                std::span<const double> value) {
  const uint32_t id = data.page(lane, token), place = token % kPageTokens;
  const bool quantized = data.layout.format == kv::Format::Int8;
  for (const bool values : {false, true}) {
    const std::span<const double> source = values ? value : key;
    float largest = 0;
    for (const double element : source)
      largest = std::max(largest, std::fabs(bf16ToFloat(floatToBf16(float(element)))));
    for (uint32_t d = 0; d < kHeadDimension; ++d) {
      const uint64_t index = values ? data.pool.valueElement(head, place, d)
                                    : splash_kv_key_element(head, place, d, kHeadDimension);
      const uint32_t tensor = values ? SPLASH_KV_VALUES : SPLASH_KV_KEYS;
      const uint16_t rounded = floatToBf16(float(source[d]));
      if (quantized)
        data.pool.slab<int8_t>(kLayer, tensor, id)[index] =
            int8_t(std::clamp(std::lrint(bf16ToFloat(rounded) * 127.0f / largest), -127L, 127L));
      else
        data.pool.slab<uint16_t>(kLayer, tensor, id)[index] = rounded;
    }
    if (quantized)
      data.pool.slab<float>(kLayer, values ? SPLASH_KV_VALUE_SCALES : SPLASH_KV_KEY_SCALES,
                            id)[splash_kv_scale_element(head, place)] = largest / 127.0f;
  }
}

Case makeCase(metal::MetalBackend &backend, kv::Format format, bool int8Queries, ops::AttentionVariant variant,
              uint32_t lanes, uint32_t rows, std::array<uint32_t, 4> histories) {
  std::array<uint32_t, 4> pageCounts{};
  uint32_t allPages = 1;  // and the page of NaNs
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    pageCounts[lane] = (histories[lane] + rows + kPageTokens - 1) / kPageTokens;
    allPages += pageCounts[lane];
  }
  const auto spread = HostKvExtents::spread(allPages + 2);
  const std::vector<uint32_t> ids = HostKvExtents::mixedPages(spread, allPages, histories[0] + rows + lanes);
  Case data{layoutOf(format), variant, int8Queries, lanes, histories, {}, false,
            HostKvExtents(backend, layoutOf(format, kLayer + 1), spread.extentPages, spread.extents),
            {}, {}, {}, 0, makeWeights(backend, histories[0] + rows)};
  data.layer = data.pool.layer(kLayer);
  const uint32_t nan = ids.back();
  for (uint32_t tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES; ++tensor) {
    const auto bytes = data.pool.layerSpans(nan, kLayer)[tensor];
    if (tensor % 2 == 0 && format == kv::Format::Int8)
      std::fill(bytes.begin(), bytes.end(), std::byte{0x7f});
    else
      for (size_t at = 0; at + 4 <= bytes.size(); at += 4) {
        const uint32_t bits = tensor % 2 ? 0x7fc00000u : 0x7fc07fc0u;
        std::memcpy(bytes.data() + at, &bits, 4);
      }
  }
  for (uint32_t lane = 0, first = 0; lane < lanes; first += pageCounts[lane++]) {
    data.pages[lane].assign(ids.begin() + first, ids.begin() + first + pageCounts[lane]);
    std::vector<uint32_t> table = data.pages[lane];
    if (variant.windowTokens)
      for (uint32_t page = 0; page < (histories[lane] - std::min(histories[lane], kWindow)) / kPageTokens; ++page) {
        table[page] = nan;
        ++data.poisoned;
      }
    data.tables[lane] = zeroed(backend, table.size() * sizeof(SplashKvPage));
    data.pool.writeTable(table, data.tables[lane]);
    std::array<double, kHeadDimension> key{}, value{};
    for (uint32_t token = 0; token < histories[lane]; ++token)
      for (uint32_t head = 0; head < kKvHeads; ++head) {
        double squares = 0;
        for (uint32_t d = 0; d < kHeadDimension; ++d) {
          key[d] = 2.5 * unit(seedOf(6, lane, head, token, d));
          value[d] = 0.2 + unit(seedOf(7, lane, head, token, d));
          squares += value[d] * value[d];
        }
        clip(key, kQkClip);
        for (double &element : value) element /= std::sqrt(squares / kHeadDimension + kEpsilon);
        storeToken(data, lane, head, token, key, value);
      }
  }
  for (uint32_t lane = lanes; lane < 4; ++lane) data.tables[lane] = data.tables[0];
  return data;
}

// Rows [first, first + rows) of each lane's rows (each lane's block of
// `blocks`, packed by their first rows), their inputs as packed rows, their
// staging `stride` rows per KV head, the prepare's relative states, and the
// attention's output rows, a packed row each.
struct Staging final {
  uint32_t first;
  uint32_t rows;
  uint32_t stride;
  std::array<SplashLaneRows, 4> blocks;
  uint32_t packedRows;
  metal::MetalBuffer querySide, keys, values, queries, states, output, layout, tree;

  [[nodiscard]] uint64_t queryIndex(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) const {
    return (((uint64_t{lane} * kKvHeads + head / kGroup) * stride + row) * kGroup + head % kGroup) * kHeadDimension +
           dimension;
  }
  [[nodiscard]] uint64_t outputIndex(uint32_t packed, uint32_t head, uint32_t dimension) const {
    return (uint64_t{packed} * kQueryHeads + head) * kHeadDimension + dimension;
  }
};

Staging stage(metal::MetalBackend &backend, const Case &data, uint32_t first, uint32_t rows, uint32_t stride,
              std::span<const uint32_t> blockRows = {}) {
  Staging staging{first, rows, stride, {}, 0, {}, {}, {}, {}, {}, {}, {}, {}};
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    staging.blocks[lane] = {staging.packedRows, blockRows.empty() ? rows : blockRows[lane]};
    staging.packedRows += staging.blocks[lane].rows;
  }
  const uint32_t packedRows = staging.packedRows;
  staging.querySide = zeroed(backend, uint64_t{packedRows} * kSide * 4);
  staging.keys = zeroed(backend, uint64_t{packedRows} * kKeyWidth * 4);
  staging.values = zeroed(backend, uint64_t{packedRows} * kKeyWidth * 4);
  staging.queries = zeroed(backend, uint64_t{data.lanes} * kKvHeads * stride * kGroup * kHeadDimension * 2);
  staging.states = poisoned(backend, uint64_t{packedRows} * kQueryHeads * kWidth * 4);
  staging.output = poisoned(backend, uint64_t{packedRows} * kQueryHeads * kHeadDimension * 2);
  staging.layout = zeroed(backend, uint64_t{data.lanes} * sizeof(SplashLaneRows));
  staging.tree = data.tree ? poisoned(backend, uint64_t{packedRows} * sizeof(SplashTreeRow)) : metal::MetalBuffer{};
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    const SplashLaneRows block = staging.blocks[lane];
    contents<SplashLaneRows>(staging.layout)[lane] = block;
    for (uint32_t row = 0; row < block.rows; ++row) {
      const uint64_t packed = block.first + row;
      for (uint32_t at = 0; at < kSide; ++at)
        contents<float>(staging.querySide)[packed * kSide + at] = queryInput(lane, first + row, at);
      for (uint32_t at = 0; at < kKeyWidth; ++at) {
        contents<float>(staging.keys)[packed * kKeyWidth + at] = keyInput(lane, first + row, at);
        contents<float>(staging.values)[packed * kKeyWidth + at] = valueInput(lane, first + row, at);
      }
      if (data.tree) contents<SplashTreeRow>(staging.tree)[packed] = data.trees[lane].place(row);
    }
  }
  return staging;
}

// Each lane's sequence: its rows after its history and the rows the staging
// starts at, bound by the staging's rows (a ragged block holds fewer).
std::vector<kv::ChunkedPrefillParams> sequencesOf(const Case &data, const Staging &staging) {
  std::vector<kv::ChunkedPrefillParams> sequences;
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    sequences.push_back(ops::PagedAttention::chunkParams(data.histories[lane] + staging.first, staging.rows,
                                                         staging.stride, uint32_t(data.pages[lane].size())));
  return sequences;
}

ops::AttentionPlan planOf(const Case &data, const Staging &staging) {
  return ops::PagedAttention::plan(
      sequencesOf(data, staging), kQueryHeads, data.layout, data.attention(), ops::Int8Activations::On,
      ops::PagedAttention::workspace(data.lanes * ops::kAttentionSequenceSplits, kQueryHeads, data.layout),
      data.variant);
}

// The prepare and the attention as a target encodes them, into `output`
// (every row unless it says otherwise).
metal::CommandGraph encode(const Case &data, const Staging &staging, const ops::AttentionPlan &plan,
                           const metal::MetalBuffer &partials, const metal::MetalBuffer &statistics,
                           const ops::AttentionOutputRows *output = nullptr) {
  metal::CommandGraph graph;
  const std::vector<kv::ChunkedPrefillParams> sequences = sequencesOf(data, staging);
  const ops::AttentionRows rows{staging.layout, staging.packedRows, staging.tree};
  ops::PagedAttention::addRelativeProjection(
      graph, {staging.querySide, staging.keys, staging.values, data.weights.queryScale, data.weights.relativeScale},
      staging.queries, staging.states, data.layer, std::span(data.tables).first(data.lanes), sequences, rows,
      kQueryHeads, data.layout, data.variant);
  std::vector<ops::AttentionHistory> histories;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) histories.push_back({data.layer, data.tables[lane]});
  static_cast<void>(ops::PagedAttention::addAttention(
      graph, plan, staging.queries, partials, statistics, sequences, histories, rows,
      output ? *output : ops::AttentionOutputRows{staging.querySide, staging.output}, {},
      {staging.states, data.weights.table}));
  return graph;
}

// The kernels and bindings of a case's encoding: the variant's prepare, its
// split and reduce, the split with the relative states and table after a
// page table of each lane's and the tree rows and layout, the reduce reading
// the query side rows.
constexpr size_t kLaneTables = SPLASH_MAXIMUM_BATCH_WIDTH;
void checkGraph(const Case &data, const Staging &staging, const ops::AttentionPlan &plan,
                const metal::CommandGraph &graph) {
  const std::string window = data.variant.windowTokens ? "_w" + std::to_string(data.variant.windowTokens) : "";
  const std::string format = data.layout.format == kv::Format::Int8 ? "q8" : "bf16";
  require(plan.relative && !plan.sinks &&
              plan.splitPipeline() ==
                  "attention_" + (data.int8Queries ? std::string("q8q") : format) + "_split_kv8_g3_d128" + window &&
              plan.reducePipeline() == "attention_reduce_kv8_g3_d128" + window,
          "the plan names another variant's kernels");
  const auto &dispatches = graph.dispatches();
  require(dispatches.size() >= 3 &&
              dispatches[0].pipelineName ==
                  "attention_" + format + "_qkv_kv8_g3_d128" + (data.variant.positionScaled ? "_scaled" : "") &&
              dispatches[0].threadsPerThreadgroup.x == kHeadDimension &&
              dispatches[0].buffers.size() == 9 + kLaneTables && dispatches[1].buffers.size() == 7 + kLaneTables &&
              dispatches[1].buffers[5 + kLaneTables].buffer.sameView(staging.states) &&
              dispatches[1].buffers[6 + kLaneTables].buffer.sameView(data.weights.table) &&
              dispatches[2].buffers.size() == 7 &&
              dispatches[2].buffers[2].buffer.sameView(staging.querySide) &&
              dispatches[2].threadsPerThreadgroup.x == kHeadDimension,
          "the prepare, split and reduce run other kernels or bindings");
}

// The fp64 prepare of a current row (`row` of its lane's rows from the
// staging's first, at `position`): its queries and relative states, of
// every query head, and its keys and values, of every KV head.
struct Prepared final {
  std::vector<double> queries, states, keys, values;
};

Prepared prepare(const Case &data, uint32_t lane, uint32_t row, uint32_t position) {
  Prepared result{std::vector<double>(kGates), std::vector<double>(kQueryHeads * kWidth),
                  std::vector<double>(kKeyWidth), std::vector<double>(kKeyWidth)};
  const double tau = data.variant.positionScaled ? tauOf(position) : 1.0;
  const float *queryScale = contents<float>(data.weights.queryScale);
  const float *relativeScale = contents<float>(data.weights.relativeScale);
  for (uint32_t head = 0; head < kQueryHeads; ++head) {
    const std::span query(result.queries.data() + head * kHeadDimension, kHeadDimension);
    for (uint32_t d = 0; d < kHeadDimension; ++d)
      query[d] = double(queryScale[head * kHeadDimension + d]) * queryInput(lane, row, head * kHeadDimension + d);
    clip(query, kQkClip);
    for (double &element : query) element *= tau;
    const std::span state(result.states.data() + head * kWidth, kWidth);
    for (uint32_t j = 0; j < kWidth; ++j)
      state[j] = double(relativeScale[head * kWidth + j]) * queryInput(lane, row, kStates + head * kWidth + j);
    clip(state, kStateClip);
    for (double &element : state) element *= tau;
  }
  for (uint32_t head = 0; head < kKvHeads; ++head) {
    const std::span key(result.keys.data() + head * kHeadDimension, kHeadDimension);
    const std::span value(result.values.data() + head * kHeadDimension, kHeadDimension);
    double squares = 0;
    for (uint32_t d = 0; d < kHeadDimension; ++d) {
      key[d] = keyInput(lane, row, head * kHeadDimension + d);
      value[d] = valueInput(lane, row, head * kHeadDimension + d);
      squares += value[d] * value[d];
    }
    clip(key, kQkClip);
    for (double &element : value) element /= std::sqrt(squares / kHeadDimension + kEpsilon);
  }
  return result;
}

// The largest errors of a check: absolute, relative to the largest
// magnitude of the reference's head vector (a head's queries, relative state,
// key or value, or attention row), and in units of the check's bound.
struct Errors final {
  double absolute = 0, relative = 0, bound = 0;
  void add(double error, double scale, double bound) {
    absolute = std::max(absolute, error);
    if (scale > 0) relative = std::max(relative, error / scale);
    this->bound = std::max(this->bound, error / bound);
  }
  void add(const Errors &other) {
    absolute = std::max(absolute, other.absolute);
    relative = std::max(relative, other.relative);
    bound = std::max(bound, other.bound);
  }
  [[nodiscard]] std::string text() const {
    char line[96];
    std::snprintf(line, sizeof line, "abs %.3g rel %.3g", absolute, relative);
    return line;
  }
};

// The prepare's errors: its staged queries, relative states and stored K/V.
struct PrepareErrors final {
  Errors queries, states, kv;
  void add(const PrepareErrors &other) {
    queries.add(other.queries);
    states.add(other.states);
    kv.add(other.kv);
  }
  [[nodiscard]] std::string text() const {
    return "q " + queries.text() + ", r " + states.text() + ", K/V " + kv.text();
  }
};

// The largest magnitude of a head's vector.
double largest(std::span<const double> vector) {
  double result = 0;
  for (const double value : vector) result = std::max(result, std::fabs(value));
  return result;
}

// The prepare's outputs against fp64: each staged query the reference
// rounded once to bf16; each relative state within 2^-20 of its head's
// state's norm; each stored key and value a BF16 page's rounded once, an
// INT8 page's within half its scale of the bf16 rounding.
PrepareErrors checkPrepare(const Case &data, const Staging &staging, const std::string &what) {
  PrepareErrors errors;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    const SplashLaneRows block = staging.blocks[lane];
    for (uint32_t row = 0; row < block.rows; ++row) {
      const uint32_t depth = data.tree ? data.trees[lane].place(row).depth : staging.first + row;
      const uint32_t position = data.histories[lane] + depth, slot = data.histories[lane] + staging.first + row;
      const Prepared expected = prepare(data, lane, staging.first + row, position);
      const std::string at = what + ": lane " + std::to_string(lane) + " row " + std::to_string(staging.first + row);
      for (uint32_t head = 0; head < kQueryHeads; ++head) {
        const std::span query = std::span(expected.queries).subspan(head * kHeadDimension, kHeadDimension);
        for (uint32_t d = 0; d < kHeadDimension; ++d) {
          const uint16_t got = contents<uint16_t>(staging.queries)[staging.queryIndex(lane, head, row, d)];
          require(test::roundedOnceToBf16(got, query[d]), at + " head " + std::to_string(head) + " query " +
                                                              std::to_string(bf16ToFloat(got)) + ", the reference " +
                                                              std::to_string(query[d]));
          errors.queries.add(std::fabs(bf16ToFloat(got) - query[d]), largest(query), ulpBf16(float(query[d])));
        }
        const std::span state = std::span(expected.states).subspan(head * kWidth, kWidth);
        double norm = 0;
        for (const double value : state) norm += value * value;
        for (uint32_t j = 0; j < kWidth; ++j) {
          const double got =
              contents<float>(staging.states)[(uint64_t{block.first + row} * kQueryHeads + head) * kWidth + j];
          const double bound = std::sqrt(norm) * 0x1p-20;
          require(std::fabs(got - state[j]) <= bound, at + " head " + std::to_string(head) + " relative state " +
                                                          std::to_string(got) + ", the reference " +
                                                          std::to_string(state[j]));
          errors.states.add(std::fabs(got - state[j]), largest(state), bound);
        }
      }
      for (uint32_t head = 0; head < kKvHeads; ++head)
        for (const bool value : {false, true}) {
          const std::span exact =
              std::span(value ? expected.values : expected.keys).subspan(head * kHeadDimension, kHeadDimension);
          for (uint32_t d = 0; d < kHeadDimension; ++d) {
            const double got = data.stored(lane, value, head, slot, d);
            const double scale = data.scale(lane, value, head, slot);
            const double bound =
                scale ? 0.51 * scale + ulpBf16(float(exact[d])) : 0.5 * ulpBf16(float(exact[d])) * (1 + 1.0 / 256);
            require(std::fabs(got - exact[d]) <= bound, at + " KV head " + std::to_string(head) +
                                                            (value ? " value " : " key ") + std::to_string(got) +
                                                            ", the reference " + std::to_string(exact[d]));
            errors.kv.add(std::fabs(got - exact[d]), largest(exact), bound);
          }
        }
    }
  }
  return errors;
}

// A row's visible keys and their distances back from it: a prompt row's
// window (or whole history) and the rows before it; a tree row's window of
// the history before its anchor and its path's rows, at their slots, each
// its depth's distance.
struct Visible final {
  std::vector<uint32_t> slots, distances;
};

Visible visible(const Case &data, const Staging &staging, uint32_t lane, uint32_t row) {
  Visible keys;
  const uint32_t history = data.histories[lane] + (data.tree ? 0 : staging.first);
  const std::vector<uint32_t> path = data.tree ? data.trees[lane].path(row) : std::vector<uint32_t>{};
  const uint32_t position = data.tree ? history + uint32_t(path.size()) - 1 : history + row;
  const uint32_t start =
      data.variant.windowTokens && position > data.variant.windowTokens ? position - data.variant.windowTokens : 0;
  const uint32_t end = data.tree ? history : position + 1;
  for (uint32_t token = start; token < end; ++token) {
    keys.slots.push_back(token);
    keys.distances.push_back(position - token);
  }
  for (uint32_t depth = 0; depth < path.size(); ++depth) {
    keys.slots.push_back(history + path[depth]);
    keys.distances.push_back(uint32_t(path.size()) - 1 - depth);
  }
  return keys;
}

// A lane's K/V of slots [first, end) as its pages store them, dequantized
// once ([slot - first][KV head][dimension]), and of the staging's rows from
// `current` on, the fp64 prepare's, which the end-to-end reference reads.
struct LaneKv final {
  uint32_t first = 0, current = 0;
  std::vector<double> keys, values, exactKeys, exactValues;

  [[nodiscard]] const double *row(bool value, bool exact, uint32_t slot, uint32_t head) const {
    if (exact && slot >= current)
      return (value ? exactValues : exactKeys).data() + (uint64_t{slot - current} * kKvHeads + head) * kHeadDimension;
    return (value ? values : keys).data() + (uint64_t{slot - first} * kKvHeads + head) * kHeadDimension;
  }
};

LaneKv laneKv(const Case &data, uint32_t lane, uint32_t first, uint32_t current, std::span<const Prepared> prepared) {
  const uint32_t end = current + uint32_t(prepared.size());
  LaneKv kv{first, current, std::vector<double>(uint64_t{end - first} * kKeyWidth), {}, {}, {}};
  kv.values.resize(kv.keys.size());
  for (uint32_t slot = first; slot < end; ++slot)
    for (uint32_t head = 0; head < kKvHeads; ++head)
      for (uint32_t d = 0; d < kHeadDimension; ++d) {
        const uint64_t at = (uint64_t{slot - first} * kKvHeads + head) * kHeadDimension + d;
        kv.keys[at] = data.stored(lane, false, head, slot, d);
        kv.values[at] = data.stored(lane, true, head, slot, d);
      }
  for (const Prepared &row : prepared) {
    kv.exactKeys.insert(kv.exactKeys.end(), row.keys.begin(), row.keys.end());
    kv.exactValues.insert(kv.exactValues.end(), row.values.begin(), row.values.end());
  }
  return kv;
}

// The fp64 attention of a row and query head over its visible keys: scores
// q . k / sqrt(128) plus the bias of each key's distance (none past the
// table), the row's sink in the softmax's denominator, the result times the
// sigmoid of the head's gates. The K/V are as stored, or (`exact`) the fp64
// prepare's of the staging's rows; `mass` is the sum of each key's weight
// times its value's magnitude, which bounds the error of the kernels' bf16
// probabilities.
struct Output final {
  std::array<double, kHeadDimension> output{}, mass{}, gate{};
  // The sink's share of the softmax's weight.
  double sink = 0;
};

Output attend(const Case &data, const Staging &staging, const LaneKv &kv, bool exact, uint32_t lane, uint32_t row,
              uint32_t head, std::span<const double> query, std::span<const double> state) {
  const uint32_t kvHead = head / kGroup, packed = staging.blocks[lane].first + row;
  const Visible keys = visible(data, staging, lane, row);
  const float *table = contents<float>(data.weights.table);
  const float *side = contents<float>(staging.querySide) + uint64_t{packed} * kSide;
  const double sink = side[kSinks + head];
  std::vector<double> scores(keys.slots.size());
  double maximum = sink;
  for (size_t i = 0; i < keys.slots.size(); ++i) {
    const double *key = kv.row(false, exact, keys.slots[i], kvHead);
    double score = 0;
    for (uint32_t d = 0; d < kHeadDimension; ++d) score += query[d] * key[d];
    score /= std::sqrt(double(kHeadDimension));
    if (keys.distances[i] < kExtent)
      for (uint32_t j = 0; j < kWidth; ++j) score += state[j] * table[keys.distances[i] * kWidth + j];
    scores[i] = score;
    maximum = std::max(maximum, score);
  }
  double denominator = std::exp(sink - maximum);
  for (double &score : scores) denominator += score = std::exp(score - maximum);
  Output result;
  result.sink = std::exp(sink - maximum) / denominator;
  for (size_t i = 0; i < keys.slots.size(); ++i) {
    const double *value = kv.row(true, exact, keys.slots[i], kvHead);
    const double weight = scores[i] / denominator;
    for (uint32_t d = 0; d < kHeadDimension; ++d) {
      result.output[d] += weight * value[d];
      result.mass[d] += weight * std::fabs(value[d]);
    }
  }
  for (uint32_t d = 0; d < kHeadDimension; ++d) {
    result.gate[d] = 1.0 / (1.0 + std::exp(-double(side[kGates + head * kHeadDimension + d])));
    result.output[d] *= result.gate[d];
  }
  return result;
}

// The rows and query heads checked, and those whose sink takes more than
// 99% of the softmax's weight.
uint64_t gRows = 0, gSinkRows = 0;

// The rows a check covers: all of a short block's, of a long one its first
// and last two tiles' and a spread of the others.
std::vector<uint32_t> checkedRows(uint32_t rows) {
  std::vector<uint32_t> checked;
  for (uint32_t row = 0; row < rows; ++row)
    if (rows <= 64 || row < 16 || row + 16 >= rows || row % 7 == 0) checked.push_back(row);
  return checked;
}

// Each checked output against the fp64 attention of what the prepare wrote
// (the staged queries, as INT8 codes where the split scores them so, the
// relative states and the stored K/V): within the bf16 rounding of the
// probabilities (2^-9 of the mass, twice over) and of the ungated and gated
// outputs. Returns its errors, and in `whole` those against the fp64
// attention of the fp64 prepare, which bounds nothing.
Errors checkOutput(const Case &data, const Staging &staging, const std::string &what, Errors &whole) {
  Errors errors;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    const SplashLaneRows block = staging.blocks[lane];
    // The fp64 prepare of the staging's rows, and the lane's K/V from its
    // first row's window on.
    std::vector<Prepared> prepared;
    for (uint32_t row = 0; row < block.rows; ++row) {
      const uint32_t depth = data.tree ? data.trees[lane].place(row).depth : staging.first + row;
      prepared.push_back(prepare(data, lane, staging.first + row, data.histories[lane] + depth));
    }
    const uint32_t current = data.histories[lane] + staging.first;
    const LaneKv kv = laneKv(data, lane, visible(data, staging, lane, 0).slots.front(), current, prepared);
    for (const uint32_t row : checkedRows(block.rows))
      for (uint32_t head = 0; head < kQueryHeads; ++head) {
        std::array<double, kHeadDimension> query{};
        for (uint32_t d = 0; d < kHeadDimension; ++d)
          query[d] = bf16ToFloat(contents<uint16_t>(staging.queries)[staging.queryIndex(lane, head, row, d)]);
        if (data.int8Queries) {
          float largest = 0;
          for (const double value : query) largest = std::max(largest, float(std::fabs(value)));
          const test::Int8Block block8 = test::int8Block(largest);
          for (double &value : query) value = double(test::int8Code(float(value), block8)) * block8.scale;
        }
        std::array<double, kWidth> state{};
        for (uint32_t j = 0; j < kWidth; ++j)
          state[j] = contents<float>(staging.states)[(uint64_t{block.first + row} * kQueryHeads + head) * kWidth + j];
        const Output expected = attend(data, staging, kv, false, lane, row, head, query, state);
        const Output exact =
            attend(data, staging, kv, true, lane, row, head,
                   std::span(prepared[row].queries).subspan(head * kHeadDimension, kHeadDimension),
                   std::span(prepared[row].states).subspan(head * kWidth, kWidth));
        const double magnitude = largest(expected.output), exactMagnitude = largest(exact.output);
        gSinkRows += expected.sink > 0.99;
        ++gRows;
        for (uint32_t d = 0; d < kHeadDimension; ++d) {
          const double actual =
              bf16ToFloat(contents<uint16_t>(staging.output)[staging.outputIndex(block.first + row, head, d)]);
          const double ungated = expected.output[d] / expected.gate[d];
          const double bound = expected.gate[d] * (expected.mass[d] / 256 + ulpBf16(float(ungated))) +
                               ulpBf16(float(expected.output[d]));
          const double error = std::fabs(actual - expected.output[d]);
          if (!(std::isfinite(actual) && error <= bound))
            throw std::runtime_error(what + ": lane " + std::to_string(lane) + " row " +
                                     std::to_string(staging.first + row) + " head " + std::to_string(head) +
                                     " dimension " + std::to_string(d) + " is " + std::to_string(actual) +
                                     ", the reference " + std::to_string(expected.output[d]) + " (bound " +
                                     std::to_string(bound) + ")");
          errors.add(error, magnitude, bound);
          whole.add(std::fabs(actual - exact.output[d]), exactMagnitude, 1);
        }
      }
  }
  return errors;
}

void run(metal::MetalBackend &backend, const Case &data, const Staging &staging, const ops::AttentionPlan &plan) {
  const metal::MetalBuffer partials = poisoned(backend, plan.workspace().partialsBytes);
  const metal::MetalBuffer statistics = poisoned(backend, plan.workspace().statisticsBytes);
  const metal::CommandGraph graph = encode(data, staging, plan, partials, statistics);
  checkGraph(data, staging, plan, graph);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

std::string describe(const Case &data, std::string_view phase) {
  std::string text = std::string(phase) + " " + std::string(kv::formatName(data.layout.format)) +
                     (data.int8Queries ? " with INT8 queries" : "") +
                     (data.variant.windowTokens ? " window " + std::to_string(data.variant.windowTokens)
                                                : " global (tau)") +
                     " history";
  for (uint32_t lane = 0; lane < data.lanes; ++lane) text += " " + std::to_string(data.histories[lane]);
  return text;
}

// The errors of every check, for the summary: by KV format, and of the
// attention whatever its pages.
PrepareErrors gPrepare;
std::array<Errors, 2> gAttention, gWhole;

void report(const Case &data, const std::string &what, const PrepareErrors &prepared, const Errors &output,
            const Errors &whole) {
  const size_t format = data.layout.format == kv::Format::Int8 ? 0 : 1;
  gPrepare.add(prepared);
  gAttention[format].add(output);
  gWhole[format].add(whole);
  std::cout << what << ": prepare " << prepared.text() << "; attention " << output.text() << " (" << output.bound
            << " of its bound); end to end " << whole.text();
  if (data.poisoned) std::cout << "; " << data.poisoned << " table entries of NaNs below the windows";
  std::cout << " PASS\n";
}

// A prefill of `rows` rows after `history` tokens as one chunk, and again as
// chunks of `chunk` rows, each over the K/V its predecessors' prepares
// stored; and the chunk's last row alone (the plan of its last tiles), bit
// for bit its row among every row's.
void checkPrefill(metal::MetalBackend &backend, kv::Format format, bool int8Queries, ops::AttentionVariant variant,
                  uint32_t history, uint32_t rows, uint32_t chunk) {
  const Case data = makeCase(backend, format, int8Queries, variant, 1, rows, {history, 0, 0, 0});
  const std::string what = describe(data, "prefill") + " rows " + std::to_string(rows);
  PrepareErrors prepared;
  Errors output, whole;
  for (const uint32_t size : {rows, chunk}) {
    for (uint32_t first = 0; first < rows; first += size) {
      const uint32_t chunkRows = std::min(size, rows - first);
      const Staging staging =
          stage(backend, data, first, chunkRows, (chunkRows + kPageTokens - 1) / kPageTokens * kPageTokens);
      const ops::AttentionPlan plan = planOf(data, staging);
      run(backend, data, staging, plan);
      const std::string at = what + (size == rows ? "" : " in chunks of " + std::to_string(chunk));
      prepared.add(checkPrepare(data, staging, at));
      output.add(checkOutput(data, staging, at, whole));
      if (size != rows) continue;
      const metal::MetalBuffer last = poisoned(backend, uint64_t{kQueryHeads} * kHeadDimension * 2);
      const ops::AttentionPlan tails = ops::PagedAttention::lastTiles(plan);
      const metal::MetalBuffer partials = poisoned(backend, tails.workspace().partialsBytes);
      const metal::MetalBuffer statistics = poisoned(backend, tails.workspace().statisticsBytes);
      const ops::AttentionOutputRows lastRow{staging.querySide, last, ops::AttentionOutput::Last};
      const metal::CommandGraph graph = encode(data, staging, tails, partials, statistics, &lastRow);
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      require(!std::memcmp(last.contents(),
                           contents<uint16_t>(staging.output) + staging.outputIndex(chunkRows - 1, 0, 0),
                           last.sizeBytes()),
              what + ": the last row alone differs from its row among every row's");
    }
  }
  report(data, what + " and in chunks of " + std::to_string(chunk) + ", last row alone", prepared, output, whole);
}

// A decode step of single rows, one a lane, as tree blocks of their anchors
// alone and as rows without a tree.
void checkDecode(metal::MetalBackend &backend, kv::Format format, bool int8Queries, ops::AttentionVariant variant,
                 std::array<uint32_t, 4> histories) {
  PrepareErrors prepared;
  Errors output, whole;
  for (const bool tree : {false, true}) {
    Case data = makeCase(backend, format, int8Queries, variant, 4, 1, histories);
    data.tree = tree;
    for (Tree &lane : data.trees) lane.parent = {0};
    const std::string what = describe(data, "decode of single rows");
    const Staging staging = stage(backend, data, 0, 1, kPageTokens);
    run(backend, data, staging, planOf(data, staging));
    prepared.add(checkPrepare(data, staging, what));
    output.add(checkOutput(data, staging, what, whole));
    if (tree) report(data, what + ", with tree rows and without", prepared, output, whole);
  }
}

// A verify step of four lanes whose blocks are trees that branch, packed by
// their first rows (`blockRows`, ragged where they differ): a chain with a
// lookup branch off row 2 at row 8, two children a row, a random tree, and a
// chain, so that a tree row's distances and tau follow its depth, not its
// row, and rows of the second tile attend ancestors of the first.
void checkTreeVerify(metal::MetalBackend &backend, kv::Format format, bool int8Queries, ops::AttentionVariant variant,
                     std::array<uint32_t, 4> histories, std::array<uint32_t, 4> blockRows) {
  constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
  Case data = makeCase(backend, format, int8Queries, variant, 4, kRows, histories);
  data.tree = true;
  for (uint32_t lane = 0; lane < 4; ++lane)
    for (uint32_t row = 0; row < blockRows[lane]; ++row) {
      const uint32_t back = std::min(row, 1 + uint32_t((unit(seedOf(8, lane, 0, row, 0)) + 1) * 2) % 4);
      data.trees[lane].parent.push_back(!row       ? 0
                                        : lane == 0 ? (row == 8 ? 2 : row - 1)
                                        : lane == 1 ? (row - 1) / 2
                                        : lane == 2 ? row - back
                                                    : row - 1);
    }
  const Staging staging = stage(backend, data, 0, kRows, kPageTokens, blockRows);
  run(backend, data, staging, planOf(data, staging));
  const std::string what = describe(data, "tree verify") + " blocks " + std::to_string(blockRows[0]) + "," +
                           std::to_string(blockRows[1]) + "," + std::to_string(blockRows[2]) + "," +
                           std::to_string(blockRows[3]);
  Errors whole;
  const PrepareErrors prepared = checkPrepare(data, staging, what);
  const Errors output = checkOutput(data, staging, what, whole);
  report(data, what, prepared, output, whole);
}

// What the geometry's plans and encoders take and refuse: its two variants'
// kernels, a rotated variant, a position scale of another geometry, the
// other geometries' projection, and its attention without its bias or
// another's with one.
void checkPlans(metal::MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  const kv::Layout flint{.attentionLayers = 1, .kvHeads = 4, .headDimension = 128};
  constexpr ops::Int8Activations off = ops::Int8Activations::Off;
  const auto planOf = [&](uint32_t heads, const kv::Layout &of, ops::AttentionVariant variant) {
    const std::array sequence{ops::PagedAttention::chunkParams(0, 8, 32, 1)};
    return ops::PagedAttention::plan(sequence, heads, of, ops::kApple10Attention, off,
                                     ops::PagedAttention::workspace(ops::kAttentionSequenceSplits, heads, of), variant);
  };
  require(ops::PagedAttention::compiles(kQueryHeads, layout, true, 0, true) &&
              !ops::PagedAttention::compiles(kQueryHeads, layout, true, 0, false) &&
              !ops::PagedAttention::compiles(kQueryHeads, layout, false, 0, true) &&
              ops::PagedAttention::compiles(32, flint, false, 64, false) &&
              !ops::PagedAttention::compiles(32, flint, false, 64, true),
          "the relative geometry compiles as another, or another as it");
  require(planOf(kQueryHeads, layout, kGlobal).relative && planOf(kQueryHeads, layout, kWindowed).relative &&
              !planOf(32, flint, {}).relative &&
              planOf(kQueryHeads, layout, kGlobal).reducePipeline() == "attention_reduce_kv8_g3_d128" &&
              planOf(kQueryHeads, layout, kWindowed).splitPipeline() == "attention_q8_split_kv8_g3_d128_w1023",
          "the relative geometry's plans name other kernels");
  for (const ops::AttentionVariant variant : {ops::AttentionVariant{}, ops::AttentionVariant{.rotated = false,
                                                                                              .windowTokens = 512}})
    rejects([&] { (void)planOf(kQueryHeads, layout, variant); }, "no attention kernel for the layer's variant",
            "a variant the relative geometry has no kernels for was accepted");
  rejects([&] { (void)planOf(32, flint, {.rotated = true, .windowTokens = 0, .positionScaled = true}); },
          "no attention kernel for the layer's variant", "another geometry's position-scaled variant was accepted");
  const metal::MetalBuffer buffer = zeroed(backend, 1 << 23);
  const std::array chunk{ops::PagedAttention::chunkParams(0, 8, 32, 1)};
  const std::array tables{buffer};
  const ops::AttentionRows rows{buffer, 8};
  const ops::NormWeights norm{buffer, true};
  metal::CommandGraph graph;
  rejects([&] {
    ops::PagedAttention::addProjection(graph, buffer, norm, norm, {}, {}, buffer, {1, 0}, tables, chunk, rows,
                                       kQueryHeads, layout, kWindowed);
  }, "a relative geometry's rows take its relative projection", "a relative geometry took the q/k norms");
  rejects([&] {
    ops::PagedAttention::addRelativeProjection(graph, {buffer, buffer, buffer, buffer, buffer}, buffer, buffer, {1, 0},
                                               tables, chunk, rows, 32, flint, {.rotated = false});
  }, "a relative projection takes an unrotated relative geometry", "another geometry took a relative projection");
  const ops::RelativeProjection inputs{buffer, buffer, buffer, buffer, buffer};
  const metal::MetalBuffer shortStates = zeroed(backend, 8 * kQueryHeads * kWidth * 4 - 4);
  rejects([&] {
    ops::PagedAttention::addRelativeProjection(graph, inputs, buffer, shortStates, {1, 0}, tables, chunk, rows,
                                               kQueryHeads, layout, kGlobal);
  }, "attention relative states buffer holds", "a prepare with too few relative states was accepted");
  const std::array histories{ops::AttentionHistory{{1, 0}, buffer}};
  const ops::AttentionOutputRows output{buffer, buffer};
  rejects([&] {
    ops::PagedAttention::addAttention(graph, planOf(kQueryHeads, layout, kGlobal), buffer, buffer, buffer, chunk,
                                      histories, rows, output);
  }, "attention relative states buffer holds", "a relative plan encoded without its bias");
  rejects([&] {
    ops::PagedAttention::addAttention(graph, planOf(kQueryHeads, layout, kGlobal), buffer, buffer, buffer, chunk,
                                      histories, rows, output, {}, {buffer, shortStates});
  }, "attention relative table buffer holds", "a relative plan encoded with a short table");
  rejects([&] {
    ops::PagedAttention::addAttention(graph, planOf(32, flint, {}), buffer, buffer, buffer, chunk, histories, rows,
                                      output, zeroed(backend, 32 * 4), {buffer, buffer});
  }, "the attention kernels take no relative bias", "another plan encoded with a relative bias");
  require(graph.empty(), "a refused encoding added a dispatch");
  std::cout << "relative plans: kernels, variants and refusals PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: relative-attention METALLIB");
    metal::MetalBackend backend(argv[1]);
    checkPlans(backend);
    // INT8 pages score against bf16 queries and against INT8 ones in every
    // split (the q8q entries); BF16 pages against bf16 queries.
    for (const auto &[format, int8Queries] : {std::pair{kv::Format::Int8, false}, std::pair{kv::Format::Int8, true},
                                              std::pair{kv::Format::BFloat16, false}})
      for (const ops::AttentionVariant variant : {kWindowed, kGlobal}) {
        // From the history's start in several chunks; after a prefix past the
        // window and the bias's extent; across position 8192, where tau
        // leaves 1.
        checkPrefill(backend, format, int8Queries, variant, 0, 300, 64);
        checkPrefill(backend, format, int8Queries, variant, 1500, 100, 37);
        checkPrefill(backend, format, int8Queries, variant, 8150, 80, 33);
        // Single rows at and past the window's and the table's extents, and
        // past position 8192 (tau 1.009 and 1.159).
        checkDecode(backend, format, int8Queries, variant, {1023, 1025, 9000, 40000});
        // Tree blocks of two tiles, and ragged blocks packed by their first
        // rows, past position 8192, the window and over short histories.
        checkTreeVerify(backend, format, int8Queries, variant, {9000, 1100, 37, 0}, {16, 16, 16, 16});
        checkTreeVerify(backend, format, int8Queries, variant, {8190, 1020, 5, 2047}, {16, 5, 9, 12});
      }
    std::cout << "largest errors: prepare " << gPrepare.text() << " (q " << gPrepare.queries.bound << ", r "
              << gPrepare.states.bound << ", K/V " << gPrepare.kv.bound << " of their bounds)\n";
    std::cout << gRows << " rows and heads checked, " << gSinkRows << " whose sink takes over 99% of the weight\n";
    for (const size_t format : {0, 1})
      std::cout << "largest errors over " << (format ? "BF16" : "INT8") << " pages: attention "
                << gAttention[format].text() << " (" << gAttention[format].bound << " of its bound), end to end "
                << gWhole[format].text() << "\n";
    std::cout << "relative attention: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "relative attention: FAIL: " << error.what() << '\n';
    return 1;
  }
}
