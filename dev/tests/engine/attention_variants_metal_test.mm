#include "HostKvExtents.hpp"
#include "LinearInputReference.hpp"
#include "LinearNumerics.hpp"
#include "NormReference.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "model/RopeScaling.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/RoPE.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

// The attention variants of the compiled geometry that has them all
// (ops::PagedAttention, kCompiledAttention): 32 query heads over 4 KV heads of
// 128 dimensions, without an output gate, with a learned sink per query head,
// its queries and keys rotated by RoPE of 64 pairs or not (NoPE), and each row
// attending its whole history or a window of the 512 keys before it. Each runs
// through the production encoders against an fp64 reference of the K/V its
// own store wrote.

namespace {

using namespace splash;
using test::HostKvExtents;
using test::bf16ToFloat;
using test::floatToBf16;
using test::rejects;
using test::require;
using test::ulpBf16;

constexpr uint32_t kQueryHeads = 32, kKvHeads = 4, kGroup = kQueryHeads / kKvHeads, kHeadDimension = 128;
constexpr uint32_t kWindow = 512, kRotaryPairs = kHeadDimension / 2;
constexpr uint32_t kPageTokens = kv::kPageTokens;
// The attention layer under test is the second of a pool's two, so its region
// starts past the first one's in every extent.
constexpr uint32_t kLayer = 1;
// The two variants a layer of the geometry takes: rotated over a window, and
// unrotated over its whole history.
constexpr ops::AttentionVariant kWindowed{.rotated = true, .windowTokens = kWindow};
constexpr ops::AttentionVariant kWhole{.rotated = false, .windowTokens = 0};
// Sinks below, at and above the scores, and one above every score, which
// leaves its rows' outputs tiny but exact: query head h takes kSinks[h % 4]
// and a quarter for each four heads before it, so that each head's sink is
// its own and a row that takes another head's, of its group or of another
// group, shows.
constexpr std::array kSinks{-9.0f, 0.0f, 12.4f, 60.0f};
float sinkOf(uint32_t head) {
  return kSinks[head % kSinks.size()] + 0.25f * float(head / kSinks.size());
}

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

uint64_t seedOf(uint32_t kind, uint32_t lane, uint32_t head, uint32_t token, uint32_t dimension) {
  return ((((uint64_t{kind} * 5 + lane) * 64 + head) * 16384 + token) << 8) + dimension;
}

metal::MetalBuffer filled(metal::MetalBackend &backend, uint64_t bytes, int byte) {
  metal::MetalBuffer buffer = test::sharedBuffer(backend, std::max<uint64_t>(bytes, 1));
  std::memset(buffer.contents(), byte, buffer.sizeBytes());
  return buffer;
}

metal::MetalBuffer zeroed(metal::MetalBackend &backend, uint64_t bytes) { return filled(backend, bytes, 0); }

// A partial workspace of NaNs: a reduce that combines a split which wrote
// nothing (one past those the partition gives) reads them, where zeros
// would leave its output as it was.
metal::MetalBuffer poisoned(metal::MetalBackend &backend, uint64_t bytes) { return filled(backend, bytes, 0xff); }

template <class T> T *contents(const metal::MetalBuffer &buffer) { return static_cast<T *>(buffer.contents()); }

// One attention of `lanes` lanes after their histories: a prefill (one lane)
// of `rows` rows, or a verify step of every lane's eight. Each lane's pages
// are mixed over the extents of a pool of two layers, its history written
// into them directly, as a cached prefix's. With a window, the entries of a
// lane's table below its first row's window name a page of NaNs instead, so
// that a read through one shows in the output.
struct Case final {
  kv::Layout layout;
  ops::AttentionVariant variant;
  uint32_t lanes;
  uint32_t rows;
  std::array<uint32_t, 4> histories;
  HostKvExtents pool;
  SplashKvLayer layer;
  std::array<std::vector<uint32_t>, 4> pages;
  std::array<metal::MetalBuffer, 4> tables;
  uint32_t poisoned = 0;
  metal::MetalBuffer sinks;

  [[nodiscard]] uint32_t page(uint32_t lane, uint32_t token) const { return pages[lane][token / kPageTokens]; }

  // The stored K/V of a lane's token, which the reference attends.
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

// The current rows' keys, values and queries, by absolute row of a lane's
// rows, whatever chunk stages them.
float chunkKey(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) {
  return bf16ToFloat(floatToBf16(1.5f * unit(seedOf(0, lane, head, row, dimension))));
}
float chunkValue(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) {
  return bf16ToFloat(floatToBf16(unit(seedOf(1, lane, head, row, dimension))));
}
float query(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) {
  return bf16ToFloat(floatToBf16(2.0f * unit(seedOf(2, lane, head, row, dimension))));
}

Case makeCase(metal::MetalBackend &backend, kv::Format format, ops::AttentionVariant variant, uint32_t lanes,
              uint32_t rows, std::array<uint32_t, 4> histories) {
  std::array<uint32_t, 4> pageCounts{};
  uint32_t allPages = 1;  // and the page of NaNs
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    pageCounts[lane] = (histories[lane] + rows + kPageTokens - 1) / kPageTokens;
    allPages += pageCounts[lane];
  }
  const auto spread = HostKvExtents::spread(allPages + 2);
  const std::vector<uint32_t> ids = HostKvExtents::mixedPages(spread, allPages, histories[0] + rows + lanes);
  Case data{layoutOf(format), variant, lanes, rows, histories,
            HostKvExtents(backend, layoutOf(format, kLayer + 1), spread.extentPages, spread.extents),
            {}, {}, {}, 0, zeroed(backend, kQueryHeads * sizeof(float))};
  data.layer = data.pool.layer(kLayer);
  for (uint32_t head = 0; head < kQueryHeads; ++head) contents<float>(data.sinks)[head] = sinkOf(head);
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
      for (uint32_t page = 0; page < (histories[lane] - std::min(histories[lane], kWindow)) / kPageTokens; ++page) {
        table[page] = nan;
        ++data.poisoned;
      }
    data.tables[lane] = zeroed(backend, table.size() * sizeof(SplashKvPage));
    data.pool.writeTable(table, data.tables[lane].contents());
    for (uint32_t token = 0; token < histories[lane]; ++token)
      for (uint32_t head = 0; head < kKvHeads; ++head) {
        const uint32_t id = data.page(lane, token), place = token % kPageTokens;
        const uint64_t scale = splash_kv_scale_element(head, place);
        if (quantized) {
          // Scales differ from token to token, so that one read at another
          // token's place shows.
          data.pool.slab<float>(kLayer, SPLASH_KV_KEY_SCALES, id)[scale] = 0.012f * (1 + float((token + head) % 9) / 16);
          data.pool.slab<float>(kLayer, SPLASH_KV_VALUE_SCALES, id)[scale] = 0.008f * (1 + float((token * 5 + head) % 11) / 16);
        }
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const float key = unit(seedOf(3, lane, head, token, dimension)), value = unit(seedOf(4, lane, head, token, dimension));
          const uint64_t keyIndex = splash_kv_key_element(head, place, dimension, kHeadDimension);
          const uint64_t valueIndex = data.pool.valueElement(head, place, dimension);
          if (quantized) {
            data.pool.slab<int8_t>(kLayer, SPLASH_KV_KEYS, id)[keyIndex] = int8_t(std::lround(127 * key));
            data.pool.slab<int8_t>(kLayer, SPLASH_KV_VALUES, id)[valueIndex] = int8_t(std::lround(127 * value));
          } else {
            data.pool.slab<uint16_t>(kLayer, SPLASH_KV_KEYS, id)[keyIndex] = floatToBf16(1.5f * key);
            data.pool.slab<uint16_t>(kLayer, SPLASH_KV_VALUES, id)[valueIndex] = floatToBf16(value);
          }
        }
      }
  }
  for (uint32_t lane = lanes; lane < 4; ++lane) data.tables[lane] = data.tables[0];
  return data;
}

// The staging of rows [first, first + rows) of each lane's rows, `stride`
// rows per KV head: queries [lane][KV head][row][query head in group]
// [dimension], keys [lane][KV head][row][dimension], values [lane][KV head]
// [dimension][row].
struct Staging final {
  uint32_t first;
  uint32_t rows;
  uint32_t stride;
  metal::MetalBuffer keys, values, queries, output;

  [[nodiscard]] uint64_t queryIndex(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) const {
    return (((uint64_t{lane} * kKvHeads + head / kGroup) * stride + row) * kGroup + head % kGroup) * kHeadDimension +
           dimension;
  }
};

Staging stage(metal::MetalBackend &backend, const Case &data, uint32_t first, uint32_t rows, uint32_t stride) {
  const uint64_t kvElements = uint64_t{data.lanes} * kKvHeads * stride * kHeadDimension;
  Staging staging{first, rows, stride, zeroed(backend, kvElements * 2), zeroed(backend, kvElements * 2),
                  zeroed(backend, kvElements * kGroup * 2), zeroed(backend, kvElements * kGroup * 2)};
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t head = 0; head < kKvHeads; ++head)
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const uint64_t base = (uint64_t{lane} * kKvHeads + head) * stride;
          contents<uint16_t>(staging.keys)[(base + row) * kHeadDimension + dimension] =
              floatToBf16(chunkKey(lane, head, first + row, dimension));
          contents<uint16_t>(staging.values)[(base * kHeadDimension) + uint64_t{dimension} * stride + row] =
              floatToBf16(chunkValue(lane, head, first + row, dimension));
        }
      for (uint32_t head = 0; head < kQueryHeads; ++head)
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
          contents<uint16_t>(staging.queries)[staging.queryIndex(lane, head, row, dimension)] =
              floatToBf16(query(lane, head, first + row, dimension));
    }
  return staging;
}

// The first key a row at `position` attends.
uint32_t windowStart(ops::AttentionVariant variant, uint32_t position) {
  return variant.windowTokens && position > variant.windowTokens ? position - variant.windowTokens : 0;
}

// The stored K/V of tokens [first, end) of a lane, dequantized once:
// [token - first][KV head][dimension].
struct StoredKv final {
  uint32_t first = 0;
  std::vector<double> keys, values;

  [[nodiscard]] uint64_t index(uint32_t head, uint32_t token, uint32_t dimension) const {
    return (uint64_t{token - first} * kKvHeads + head) * kHeadDimension + dimension;
  }
};

StoredKv stored(const Case &data, uint32_t lane, uint32_t first, uint32_t end) {
  StoredKv kv{first, std::vector<double>(uint64_t{end - first} * kKvHeads * kHeadDimension), {}};
  kv.values.resize(kv.keys.size());
  for (uint32_t token = first; token < end; ++token)
    for (uint32_t head = 0; head < kKvHeads; ++head)
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        kv.keys[kv.index(head, token, dimension)] = data.key(lane, head, token, dimension);
        kv.values[kv.index(head, token, dimension)] = data.value(lane, head, token, dimension);
      }
  return kv;
}

// The fp64 attention of a lane's row (its position after the lane's
// history) and query head over the stored K/V of its window, or its whole
// history, and the head's sink: a key with no value. `mass` is the sum of
// each key's weight times its value's magnitude, which bounds the error of
// the kernels' bf16 probabilities.
struct Reference final {
  std::array<double, kHeadDimension> output{};
  std::array<double, kHeadDimension> mass{};
};

Reference reference(const Case &data, const StoredKv &kv, uint32_t lane, uint32_t row, uint32_t head) {
  const uint32_t position = data.histories[lane] + row, kvHead = head / kGroup;
  const uint32_t start = windowStart(data.variant, position);
  const double sink = contents<float>(data.sinks)[head];
  std::array<double, kHeadDimension> queries{};
  for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
    queries[dimension] = query(lane, head, row, dimension);
  std::vector<double> scores(position + 1 - start);
  double maximum = sink;
  for (uint32_t token = start; token <= position; ++token) {
    const double *keys = &kv.keys[kv.index(kvHead, token, 0)];
    double score = 0;
    for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) score += queries[dimension] * keys[dimension];
    scores[token - start] = score / std::sqrt(double(kHeadDimension));
    maximum = std::max(maximum, scores[token - start]);
  }
  double denominator = std::exp(sink - maximum);
  for (double &score : scores) {
    score = std::exp(score - maximum);
    denominator += score;
  }
  Reference result;
  for (uint32_t token = start; token <= position; ++token) {
    const double weight = scores[token - start] / denominator;
    const double *values = &kv.values[kv.index(kvHead, token, 0)];
    for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
      result.output[dimension] += weight * values[dimension];
      result.mass[dimension] += weight * std::fabs(values[dimension]);
    }
  }
  return result;
}

// The rows of a chunk the reference checks: all of a short one's, and of a
// long one those of its first and last two tiles, those whose windows
// begin about the history's start, and a spread of the others.
std::vector<uint32_t> checkedRows(uint32_t position, uint32_t rows) {
  std::vector<uint32_t> checked;
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t tile = row / kv::kPrefillAttentionTileRows, tiles = (rows + 7) / 8;
    const bool edge = tile < 2 || tile + 2 >= tiles || (position + row >= kWindow - 16 && position + row < kWindow + 16);
    if (rows <= 64 || edge || row % 13 == 0) checked.push_back(row);
  }
  return checked;
}

// Each checked output against the reference: within the bf16 rounding of
// the probabilities (2^-9 of the mass, twice over) and of the output.
// Returns the largest error in units of that bound.
double checkOutput(const Case &data, const Staging &staging, const std::string &what) {
  double largest = 0;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    const uint32_t position = data.histories[lane] + staging.first;
    const StoredKv kv = stored(data, lane, windowStart(data.variant, position), position + staging.rows);
    for (const uint32_t row : checkedRows(position, staging.rows))
      for (uint32_t head = 0; head < kQueryHeads; ++head) {
        const Reference expected = reference(data, kv, lane, staging.first + row, head);
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const double actual =
              bf16ToFloat(contents<uint16_t>(staging.output)[staging.queryIndex(lane, head, row, dimension)]);
          const double bound = expected.mass[dimension] / 256 + ulpBf16(float(expected.output[dimension]));
          const double error = std::fabs(actual - expected.output[dimension]);
          if (!(std::isfinite(actual) && error <= bound))
            throw std::runtime_error(what + ": lane " + std::to_string(lane) + " row " +
                                     std::to_string(staging.first + row) + " head " + std::to_string(head) +
                                     " dimension " + std::to_string(dimension) + " is " + std::to_string(actual) +
                                     ", the reference " + std::to_string(expected.output[dimension]) + " (bound " +
                                     std::to_string(bound) + ")");
          largest = std::max(largest, error / bound);
        }
      }
  }
  return largest;
}

// The store wrote each staged row to its page: BF16 bit for bit, INT8 within
// half a step of its row's scale, the head's largest magnitude over 127.
void checkStore(const Case &data, const Staging &staging, const std::string &what) {
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    for (uint32_t row = 0; row < staging.rows; ++row) {
      const uint32_t token = data.histories[lane] + staging.first + row;
      for (uint32_t head = 0; head < kKvHeads; ++head)
        for (const bool values : {false, true}) {
          float largest = 0;
          for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
            largest = std::max(largest, std::fabs(values ? chunkValue(lane, head, staging.first + row, dimension)
                                                         : chunkKey(lane, head, staging.first + row, dimension)));
          const double scale = largest / 127;
          for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
            const double source = values ? chunkValue(lane, head, staging.first + row, dimension)
                                         : chunkKey(lane, head, staging.first + row, dimension);
            const double stored = values ? data.value(lane, head, token, dimension) : data.key(lane, head, token, dimension);
            const bool kept = data.layout.format == kv::Format::BFloat16
                                  ? stored == source
                                  : std::fabs(stored - source) <= scale * (0.5 + 1e-5);
            require(kept, what + ": the store's row differs from its staging");
          }
        }
    }
}

// The plan of a case's chunk, and its encoding: the store, then the
// attention, as the runtime encodes them.
template <class Plan> metal::CommandGraph encode(const Case &data, const Staging &staging, const Plan &plan,
                                                 const metal::MetalBuffer &partials,
                                                 const metal::MetalBuffer &statistics) {
  metal::CommandGraph graph;
  if constexpr (std::is_same_v<Plan, ops::PrefillAttentionPlan>) {
    const kv::ChunkedPrefillParams chunk = ops::PagedAttention::prefillParams(
        data.histories[0] + staging.first, staging.rows, staging.stride, uint32_t(data.pages[0].size()));
    ops::PagedAttention::addPrefillStore(graph, data.layer, staging.keys, staging.values, data.tables[0], chunk,
                                         data.layout);
    ops::PagedAttention::addPrefill(graph, data.layer, staging.queries, staging.output, partials, statistics,
                                    data.tables[0], chunk, plan, {}, data.sinks);
  } else {
    std::array<kv::ChunkedPrefillParams, 4> chunks{};
    for (uint32_t lane = 0; lane < data.lanes; ++lane)
      chunks[lane] = ops::PagedAttention::verifyParams(data.histories[lane], uint32_t(data.pages[lane].size()));
    ops::PagedAttention::addVerify(graph, data.layer,
                                   {staging.keys, staging.values, staging.queries, partials, statistics,
                                    staging.output, data.tables, data.sinks},
                                   std::span(chunks).first(data.lanes), plan);
  }
  return graph;
}

// A plan's kernels and grids: the variant's split and reduce of the
// geometry, the reduce a thread per dimension, and the window's split
// counts.
template <class Plan> void checkPlan(const Case &data, const Plan &plan, const metal::CommandGraph &graph) {
  constexpr bool prefill = std::is_same_v<Plan, ops::PrefillAttentionPlan>;
  const std::string phase = prefill ? "prefill" : "verify";
  const std::string window = data.variant.windowTokens ? "_w" + std::to_string(data.variant.windowTokens) : "";
  const std::string format = data.layout.format == kv::Format::Int8 ? "q8" : "bf16";
  require(plan.sinks && plan.splitPipeline == phase + "_attention_" + format + "_split_kv4_g8_d128" + window &&
              plan.reducePipeline == phase + "_attention_reduce_kv4_g8_d128" + window + "_sinks",
          "the plan names another variant's kernels");
  const auto &dispatches = graph.dispatches();
  require(dispatches.size() == 3 && dispatches[0].pipelineName == phase + "_attention_" + format + "_store_kv4_d128" &&
              dispatches[0].threadsPerThreadgroup.x == kHeadDimension &&
              dispatches[2].threadsPerThreadgroup.x == kHeadDimension && dispatches[2].buffers.size() == 4 &&
              dispatches[2].buffers[3].buffer.sameView(data.sinks),
          "the store and reduce run a thread per dimension, the reduce with the sinks");
  const uint32_t pages = kv::windowAttentionPages(kWindow);
  if constexpr (prefill) {
    const uint32_t splits = std::clamp(32U / plan.tiles, 1U, 32U);
    require(plan.splits == (data.variant.windowTokens ? std::min(splits, pages) : splits),
            "a prefill's splits are not its window's");
  } else {
    for (uint32_t lane = 0; lane < data.lanes; ++lane) {
      const uint32_t splits = kv::verifyAttentionSplits(data.histories[lane]);
      require(plan.laneSplits[lane] == (data.variant.windowTokens ? std::min(splits, pages) : splits),
              "a verify lane's splits are not its window's");
    }
  }
}

template <class Plan> void run(metal::MetalBackend &backend, const Case &data, const Staging &staging, const Plan &plan) {
  const metal::MetalBuffer partials = poisoned(backend, plan.workspace.partialsBytes);
  const metal::MetalBuffer statistics = poisoned(backend, plan.workspace.statisticsBytes);
  const metal::CommandGraph graph = encode(data, staging, plan, partials, statistics);
  checkPlan(data, plan, graph);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

std::string describe(const Case &data, std::string_view phase) {
  std::string text = std::string(phase) + " " + std::string(kv::formatName(data.layout.format)) +
                     (data.variant.windowTokens ? " window " + std::to_string(data.variant.windowTokens) : " whole") +
                     " history";
  for (uint32_t lane = 0; lane < data.lanes; ++lane) text += " " + std::to_string(data.histories[lane]);
  return text;
}

// A prefill of `rows` rows after a cached prefix of `history` tokens, as one
// chunk, and again as the chunks `split` lists, each over the K/V its
// predecessors stored, each row of each against the reference.
void checkPrefill(metal::MetalBackend &backend, kv::Format format, ops::AttentionVariant variant, uint32_t history,
                  uint32_t rows, std::vector<uint32_t> split = {}) {
  const Case data = makeCase(backend, format, variant, 1, rows, {history, 0, 0, 0});
  const std::string what = describe(data, "prefill") + " rows " + std::to_string(rows);
  double largest = 0;
  split.insert(split.begin(), rows);
  bool whole = true;
  for (const uint32_t chunk : split) {
    for (uint32_t first = 0; first < rows; first += chunk) {
      const uint32_t chunkRows = std::min(chunk, rows - first);
      const Staging staging = stage(backend, data, first, chunkRows, (chunkRows + kPageTokens - 1) / kPageTokens * kPageTokens);
      run(backend, data, staging, ops::PagedAttention::prefillPlan(chunkRows, kQueryHeads, data.layout, variant));
      checkStore(data, staging, what);
      largest = std::max(largest, checkOutput(data, staging, what + (whole ? "" : " in chunks of " + std::to_string(chunk))));
    }
    whole = false;
  }
  std::cout << what << (split.size() > 1 ? " and in chunks" : "") << ": " << data.poisoned
            << " table entries of NaNs below the window, largest error " << largest << " of its bound PASS\n";
}

void checkVerify(metal::MetalBackend &backend, kv::Format format, ops::AttentionVariant variant,
                 std::span<const uint32_t> histories) {
  std::array<uint32_t, 4> lanes{};
  std::copy(histories.begin(), histories.end(), lanes.begin());
  const auto count = uint32_t(histories.size());
  const Case data = makeCase(backend, format, variant, count, kv::kVerifyRows, lanes);
  const Staging staging = stage(backend, data, 0, kv::kVerifyRows, kv::kVerifyChunkStride);
  run(backend, data, staging, ops::PagedAttention::verifyPlan(count, kQueryHeads, data.layout, histories, variant));
  const std::string what = describe(data, "verify");
  checkStore(data, staging, what);
  const double largest = checkOutput(data, staging, what);
  std::cout << what << ": " << data.poisoned << " table entries of NaNs below the windows, largest error "
            << largest << " of its bound PASS\n";
}

// Split and reduce derive the same pages from the window for any split
// count, so every count the partial workspace holds gives the reference,
// the reduce combining exactly the splits that wrote: each lane of a verify
// with its split count set, past the window's pages as well, and a prefill
// tile.
void checkSplitCounts(metal::MetalBackend &backend, kv::Format format) {
  const std::array<uint32_t, 3> histories{4097, 515, 10000};
  const Case data = makeCase(backend, format, kWindowed, 3, kv::kVerifyRows, {4097, 515, 10000, 0});
  const Staging staging = stage(backend, data, 0, kv::kVerifyRows, kv::kVerifyChunkStride);
  const auto plan = ops::PagedAttention::verifyPlan(3, kQueryHeads, data.layout, histories, kWindowed);
  run(backend, data, staging, plan);
  for (const uint32_t splits : {1U, 2U, 3U, 7U, 17U, 18U, 19U, 32U, 128U}) {
    const metal::MetalBuffer partials = poisoned(backend, plan.workspace.partialsBytes);
    const metal::MetalBuffer statistics = poisoned(backend, plan.workspace.statisticsBytes);
    std::memset(staging.output.contents(), 0xff, staging.output.sizeBytes());
    std::array<kv::VerifyAttentionParams, 4> params{};
    for (uint32_t lane = 0; lane < 3; ++lane)
      params[lane] = {histories[lane], uint32_t(data.pages[lane].size()), data.layer, splits, splits, 0, 0};
    metal::CommandGraph graph;
    graph.add(plan.splitPipeline,
              {staging.queries, partials, statistics, data.tables[0], data.tables[1], data.tables[2], data.tables[3]},
              params, {kKvHeads, splits, 3});
    graph.add(plan.reducePipeline, {partials, statistics, staging.output, data.sinks}, params,
              {kKvHeads, kv::kVerifyRows * kGroup, 3}, {kHeadDimension, 1, 1});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    checkOutput(data, staging, describe(data, "verify") + " of " + std::to_string(splits) + " splits");
  }
  const Case tile = makeCase(backend, format, kWindowed, 1, 8, {4103, 0, 0, 0});
  const Staging tileStaging = stage(backend, tile, 0, 8, kPageTokens);
  const auto tilePlan = ops::PagedAttention::prefillPlan(8, kQueryHeads, tile.layout, kWindowed);
  run(backend, tile, tileStaging, tilePlan);
  for (uint32_t splits = 1; splits <= SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS; ++splits) {
    const metal::MetalBuffer partials = poisoned(backend, uint64_t{splits} * kKvHeads * 64 * kHeadDimension * 4);
    const metal::MetalBuffer statistics = poisoned(backend, uint64_t{splits} * kKvHeads * 64 * 2 * 4);
    std::memset(tileStaging.output.contents(), 0xff, tileStaging.output.sizeBytes());
    const kv::PrefillAttentionParams params{4103, 8, kPageTokens, uint32_t(tile.pages[0].size()), tile.layer,
                                            splits, 0, 0};
    metal::CommandGraph graph;
    graph.add(tilePlan.splitPipeline, {tileStaging.queries, partials, statistics, tile.tables[0]}, params,
              {kKvHeads, 1, splits});
    graph.add(tilePlan.reducePipeline, {partials, statistics, tileStaging.output, tile.sinks}, params,
              {kKvHeads, 64, 1}, {kHeadDimension, 1, 1});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    checkOutput(tile, tileStaging, describe(tile, "prefill") + " of " + std::to_string(splits) + " splits");
  }
  std::cout << "split counts: " << kv::formatName(format) << " verify 1-128 and prefill 1-32 splits PASS\n";
}

// Where every row's window holds its whole history, the window changes
// nothing: the windowed kernels' outputs are the whole history's bit for bit
// at the same split partition (a page a split in a verify, a prefill's
// splits below the window's pages).
void checkWindowWithinHistory(metal::MetalBackend &backend, kv::Format format) {
  const auto outputs = [&](ops::AttentionVariant variant, bool verify, uint32_t lanes, uint32_t rows,
                           std::array<uint32_t, 4> histories) {
    const Case data = makeCase(backend, format, variant, lanes, rows, histories);
    const Staging staging = stage(backend, data, 0, rows, verify ? kv::kVerifyChunkStride : (rows + 31) / 32 * 32);
    if (verify)
      run(backend, data, staging,
          ops::PagedAttention::verifyPlan(lanes, kQueryHeads, data.layout, std::span(histories).first(lanes), variant));
    else
      run(backend, data, staging, ops::PagedAttention::prefillPlan(rows, kQueryHeads, data.layout, variant));
    const auto *begin = contents<uint16_t>(staging.output);
    return std::vector<uint16_t>(begin, begin + staging.output.sizeBytes() / 2);
  };
  const ops::AttentionVariant whole{.rotated = true, .windowTokens = 0};
  require(outputs(kWindowed, true, 4, 8, {0, 31, 200, 504}) == outputs(whole, true, 4, 8, {0, 31, 200, 504}),
          "a windowed verify inside its window differs from the whole history's");
  require(outputs(kWindowed, false, 1, 300, {0, 0, 0, 0}) == outputs(whole, false, 1, 300, {0, 0, 0, 0}) &&
              outputs(kWindowed, false, 1, 24, {200, 0, 0, 0}) == outputs(whole, false, 1, 24, {200, 0, 0, 0}),
          "a windowed prefill inside its window differs from the whole history's");
  std::cout << "windows within the history: " << kv::formatName(format) << " bit for bit the whole history's PASS\n";
}

// The plans of the variants, and what each refuses: a variant the geometry
// is not compiled for, and a reduce's sinks where the plan has none or the
// layer gives too few.
void checkPlans(metal::MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  const kv::Layout qwen{.attentionLayers = 1, .kvHeads = 4, .headDimension = 256};
  require(kv::windowAttentionPages(512) == 18 && kv::verifyAttentionSplits(0, 512) == 18 &&
              kv::verifyAttentionSplits(1'000'000, 512) == 18 && kv::verifyAttentionSplits(1'000'000) == 128,
          "a window of 512 keys reads at most 18 pages");
  for (const ops::AttentionVariant variant : {ops::AttentionVariant{.rotated = true, .windowTokens = 256},
                                              ops::AttentionVariant{.rotated = false, .windowTokens = 1024}})
    rejects([&] { (void)ops::PagedAttention::prefillPlan(8, kQueryHeads, layout, variant); },
            "no attention kernel for the layer's variant", "a window the geometry has no kernels for was accepted");
  for (const ops::AttentionVariant variant : {kWindowed, kWhole})
    rejects([&] { (void)ops::PagedAttention::prefillPlan(8, 24, qwen, variant); },
            "no attention kernel for the layer's variant", "a variant of a geometry without it was accepted");
  require(!ops::PagedAttention::prefillPlan(8, 24, qwen).sinks &&
              ops::PagedAttention::prefillPlan(8, 24, qwen).reducePipeline == "prefill_attention_reduce",
          "a geometry without sinks takes none");
  const metal::MetalBuffer buffer = zeroed(backend, 1 << 23), sinks = zeroed(backend, kQueryHeads * 4);
  const metal::MetalBuffer fewSinks = zeroed(backend, kQueryHeads * 4 - 4);
  const kv::ChunkedPrefillParams chunk = ops::PagedAttention::prefillParams(0, 8, 32, 1);
  const auto plan = ops::PagedAttention::prefillPlan(8, kQueryHeads, layout, kWindowed);
  metal::CommandGraph graph;
  rejects([&] { ops::PagedAttention::addPrefill(graph, {1, 0}, buffer, buffer, buffer, buffer, buffer, chunk, plan); },
          "attention sinks buffer holds", "a plan with sinks encoded without them");
  rejects([&] {
    ops::PagedAttention::addPrefill(graph, {1, 0}, buffer, buffer, buffer, buffer, buffer, chunk, plan, {}, fewSinks);
  }, "attention sinks buffer holds", "a plan with sinks encoded with too few");
  rejects([&] {
    ops::PagedAttention::addPrefill(graph, {1, 0}, buffer, buffer, buffer, buffer, buffer, chunk,
                                    ops::PagedAttention::prefillPlan(8, 24, qwen), {}, sinks);
  }, "the attention kernels take no sinks", "a plan without sinks encoded with some");
  require(graph.empty(), "a refused attention encoded a dispatch");
  std::cout << "variant plans PASS\n";
}

// The q/k norms and RoPE of the geometry's projection rows, without a gate:
// 32 query heads, then 4 K and 4 V heads, of 128. Rotated, each pair
// (i, i + 64) of the normalized head rotates by its row's table entry, NeoX's
// half split of the whole head; unrotated (NoPE) the head is its norm, and no
// table is bound. Against fp64 with norm weights in bf16 or F32, as
// paged_attention_plan_test's check of the gated geometries.
void checkProjection(metal::MetalBackend &backend, bool rotated, bool float32, bool verify) {
  const ops::AttentionVariant variant{.rotated = rotated, .windowTokens = 0};
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  const uint32_t lanes = verify ? 3 : 1, rows = verify ? 8 : 37, stride = verify ? kv::kVerifyChunkStride : 64;
  const uint32_t packedWidth = (kQueryHeads + 2 * kKvHeads) * kHeadDimension;
  const metal::MetalBuffer packed = zeroed(backend, uint64_t{lanes} * rows * packedWidth * 2);
  const metal::MetalBuffer ropeCos = rotated ? zeroed(backend, uint64_t{lanes} * rows * kRotaryPairs * 4) : metal::MetalBuffer{};
  const metal::MetalBuffer ropeSin = rotated ? zeroed(backend, ropeCos.sizeBytes()) : metal::MetalBuffer{};
  const metal::MetalBuffer queries = zeroed(backend, uint64_t{lanes} * kQueryHeads * stride * kHeadDimension * 2);
  const metal::MetalBuffer keys = zeroed(backend, uint64_t{lanes} * kKvHeads * stride * kHeadDimension * 2);
  const metal::MetalBuffer values = zeroed(backend, keys.sizeBytes());
  uint64_t seed = (rotated ? 100 : 200) + (float32 ? 10 : 0) + (verify ? 1 : 0);
  for (uint64_t i = 0; i < packed.sizeBytes() / 2; ++i) contents<uint16_t>(packed)[i] = floatToBf16(2 * unit(seed++ << 20));
  for (uint64_t i = 0; rotated && i < ropeCos.sizeBytes() / 4; ++i) {
    const double angle = 3.14159265358979 * unit(seed++ << 20);
    contents<float>(ropeCos)[i] = float(std::cos(angle));
    contents<float>(ropeSin)[i] = float(std::sin(angle));
  }
  const auto weight = [&](uint32_t) { return float(1.0 + 0.3 * unit(seed++ << 20)); };
  const ops::NormWeights queryNorm = test::makeNormWeights(backend, kHeadDimension, float32, weight);
  const ops::NormWeights keyNorm = test::makeNormWeights(backend, kHeadDimension, float32, weight);
  metal::CommandGraph graph;
  if (verify)
    ops::PagedAttention::addVerifyProjection(graph, packed, queryNorm, keyNorm, ropeCos, ropeSin, queries, keys,
                                             values, kQueryHeads, layout, lanes, variant);
  else
    ops::PagedAttention::addPrefillProjection(graph, packed, queryNorm, keyNorm, ropeCos, ropeSin, queries, keys,
                                              values, rows, stride, kQueryHeads, layout, variant);
  const std::string kernel = std::string(verify ? "verify" : "prefill") + "_attention_qkv_kv4_g8_d128" +
                             (rotated ? "" : "_nope") + (float32 ? "_f32" : "");
  require(graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == kernel &&
              graph.dispatches()[0].threadsPerThreadgroup.x == kHeadDimension &&
              graph.dispatches()[0].buffers.size() == (rotated ? 8U : 6U),
          "the projection runs another kernel or binds tables it does not read");
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < rows; ++row) {
      const uint64_t packedRow = (uint64_t{lane} * rows + row) * packedWidth;
      const uint64_t rope = (uint64_t{lane} * rows + row) * kRotaryPairs;
      for (uint32_t head = 0; head < kQueryHeads + kKvHeads; ++head) {
        const bool isQuery = head < kQueryHeads;
        const uint32_t h = isQuery ? head : head - kQueryHeads;
        const uint16_t *source =
            contents<uint16_t>(packed) + packedRow + (isQuery ? h : kQueryHeads + h) * kHeadDimension;
        const uint16_t *out =
            isQuery ? contents<uint16_t>(queries) +
                          (((uint64_t{lane} * kKvHeads + h / kGroup) * stride + row) * kGroup + h % kGroup) *
                              kHeadDimension
                    : contents<uint16_t>(keys) + ((uint64_t{lane} * kKvHeads + h) * stride + row) * kHeadDimension;
        const std::vector<double> normalized = test::rmsNorm(source, isQuery ? queryNorm : keyNorm, kHeadDimension);
        for (uint32_t d = 0; d < kRotaryPairs; ++d) {
          if (!rotated) {
            require(test::roundedOnceToBf16(out[d], normalized[d]) &&
                        test::roundedOnceToBf16(out[d + kRotaryPairs], normalized[d + kRotaryPairs]),
                    "the unrotated prepare is not the norm rounded once");
            continue;
          }
          const double first = bf16ToFloat(floatToBf16(float(normalized[d])));
          const double second = bf16ToFloat(floatToBf16(float(normalized[d + kRotaryPairs])));
          const double c = contents<float>(ropeCos)[rope + d], s = contents<float>(ropeSin)[rope + d];
          const double rotatedHalves[2] = {first * c - second * s, second * c + first * s};
          for (uint32_t half = 0; half < 2; ++half)
            require(std::fabs(bf16ToFloat(out[d + half * kRotaryPairs]) - rotatedHalves[half]) <=
                        ulpBf16(float(rotatedHalves[half])) +
                            ulpBf16(float(std::max(std::fabs(first), std::fabs(second)))),
                    "the prepare's rotation differs from NeoX's half split in fp64");
        }
        if (!isQuery)
          for (uint32_t d = 0; d < kHeadDimension; ++d)
            require(contents<uint16_t>(values)[((uint64_t{lane} * kKvHeads + h) * kHeadDimension + d) * stride + row] ==
                        source[kKvHeads * kHeadDimension + d],
                    "the prepare's value copy differs");
      }
    }
  std::cout << "attention prepare: " << (rotated ? "RoPE 64" : "NoPE") << (verify ? " verify" : " prefill")
            << (float32 ? " f32" : " bf16") << " norms PASS\n";
}

// The tables of 64 pairs a row: each row's angle, its position times the
// pair's inverse frequency of theta 1e4, NeoX's, to the KV ceiling.
void checkRopeTables(metal::MetalBackend &backend) {
  const std::vector<uint32_t> positions{0, 1, 511, 512, 4097, 262143, kv::kMaximumLogicalTokens - 1};
  const auto rows = uint32_t(positions.size());
  const metal::MetalBuffer targetPositions = zeroed(backend, rows * 3 * 4);
  for (uint32_t row = 0; row < rows; ++row)
    std::fill_n(contents<uint32_t>(targetPositions) + row * 3, 3, positions[row]);
  const metal::MetalBuffer inverse = zeroed(backend, kRotaryPairs * 4);
  model::rotaryInverseFrequencies(1.0e4F, kRotaryPairs, {}, {contents<float>(inverse), kRotaryPairs});
  const metal::MetalBuffer cosine = zeroed(backend, rows * kRotaryPairs * 4), sine = zeroed(backend, cosine.sizeBytes());
  const metal::MetalBuffer draft = zeroed(backend, SPLASH_DRAFT_ROPE_PAIRS * 4);
  metal::CommandGraph graph;
  ops::RoPE::addTables(graph, targetPositions, draft, inverse, draft, cosine, sine, draft, draft, {rows, 0, 1.0F},
                       2048, kRotaryPairs);
  require(graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == "rope_build_tables_p64",
          "tables of 64 pairs run another kernel");
  rejects([&] {
    ops::RoPE::addTables(graph, targetPositions, draft, inverse, draft, cosine, sine, draft, draft, {rows, 0, 1.0F},
                         2048, 48);
  }, "no RoPE table kernel for the target's rotary pairs", "tables of pairs no kernel builds were accepted");
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  double largest = 0;
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t pair = 0; pair < kRotaryPairs; ++pair) {
      const float angle = float(positions[row]) * contents<float>(inverse)[pair];
      require(std::fabs(double(contents<float>(inverse)[pair]) - std::pow(1.0e4, -double(pair) / kRotaryPairs)) <=
                  1e-6 * std::pow(1.0e4, -double(pair) / kRotaryPairs),
              "the inverse frequencies are not NeoX's of theta 1e4");
      largest = std::max({largest, std::fabs(contents<float>(cosine)[row * kRotaryPairs + pair] - std::cos(double{angle})),
                          std::fabs(contents<float>(sine)[row * kRotaryPairs + pair] - std::sin(double{angle}))});
    }
  require(largest < 1e-6, "the tables of 64 pairs lost precision");
  std::cout << "RoPE tables of 64 pairs to position " << positions.back() << ": largest error " << largest
            << " PASS\n";
}

// The ungated output gathers the staged attention rows into the
// out-projection's input rows as they are, prefill and verify, and verify's
// table variant writes the table the projection's own preparation writes of
// those rows.
void checkGather(metal::MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  const uint32_t width = kQueryHeads * kHeadDimension;
  for (const bool verify : {false, true}) {
    const uint32_t lanes = verify ? 3 : 1, rows = verify ? lanes * kv::kVerifyRows : 37,
                   stride = verify ? kv::kVerifyChunkStride : 64;
    const metal::MetalBuffer attention = zeroed(backend, uint64_t{lanes} * kQueryHeads * stride * kHeadDimension * 2);
    for (uint64_t i = 0; i < attention.sizeBytes() / 2; ++i) contents<uint16_t>(attention)[i] = floatToBf16(unit(i));
    const metal::MetalBuffer hidden = zeroed(backend, uint64_t{rows} * width * 2);
    // The gather reads no packed row, so it takes none.
    metal::CommandGraph graph;
    if (verify)
      require(ops::PagedAttention::addVerifyGate(graph, {}, attention, hidden, kQueryHeads, layout, lanes, {},
                                                 ops::LinearInput::Plain)
                      .layout == ops::LinearInput::Plain,
              "the plain gather wrote a table");
    else
      ops::PagedAttention::addPrefillGate(graph, {}, attention, hidden, rows, stride, kQueryHeads, layout);
    require(graph.dispatches().size() == 1 &&
                graph.dispatches()[0].pipelineName ==
                    std::string(verify ? "verify" : "prefill") + "_attention_gather_kv4_g8_d128",
            "the ungated output runs another kernel");
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    for (uint32_t row = 0; row < rows; ++row)
      for (uint32_t head = 0; head < kQueryHeads; ++head)
        for (uint32_t d = 0; d < kHeadDimension; ++d) {
          const uint32_t lane = row / kv::kVerifyRows, local = verify ? row % kv::kVerifyRows : row;
          const uint64_t staged =
              (((uint64_t{verify ? lane : 0} * kKvHeads + head / kGroup) * stride + local) * kGroup + head % kGroup) *
                  kHeadDimension +
              d;
          require(contents<uint16_t>(hidden)[(uint64_t{row} * kQueryHeads + head) * kHeadDimension + d] ==
                      contents<uint16_t>(attention)[staged],
                  "the gather moved an attention row elsewhere or changed it");
        }
    if (!verify) continue;
    const ops::LinearScratch scratch{
        zeroed(backend, ops::tableBytes(width, rows)),
        zeroed(backend, ops::tableSumsBytes(ops::LinearInput::Table16, width, rows)), {}, {}};
    const ops::LinearScratch expected{
        zeroed(backend, ops::tableBytes(width, rows)),
        zeroed(backend, ops::tableSumsBytes(ops::LinearInput::Table16, width, rows)), {}, {}};
    const metal::MetalBuffer tabled = zeroed(backend, hidden.sizeBytes());
    metal::CommandGraph tableGraph;
    const ops::PreparedInput prepared = ops::PagedAttention::addVerifyGate(
        tableGraph, {}, attention, tabled, kQueryHeads, layout, lanes, scratch, ops::LinearInput::Table16);
    require(prepared.layout == ops::LinearInput::Table16 &&
                tableGraph.dispatches()[0].pipelineName == "verify_attention_gather_table16_kv4_g8_d128",
            "the table gather reported no table");
    test::addReferencePreparation(tableGraph, ops::LinearInput::Table16, hidden, expected.input, expected.sums, width,
                                  lanes);
    static_cast<void>(backend.submitCommandAsync(tableGraph.dispatches()).wait());
    require(!std::memcmp(tabled.contents(), hidden.contents(), hidden.sizeBytes()) &&
                !std::memcmp(scratch.input.contents(), expected.input.contents(), scratch.input.sizeBytes()) &&
                !std::memcmp(scratch.sums.contents(), expected.sums.contents(), scratch.sums.sizeBytes()),
            "the table gather's rows or table differ from the plain gather's and its preparation");
  }
  std::cout << "ungated output: prefill and verify gathers, Table16 PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: attention-variants METALLIB");
    metal::MetalBackend backend(argv[1]);
    checkPlans(backend);
    checkRopeTables(backend);
    for (const bool rotated : {true, false})
      for (const bool float32 : {false, true})
        for (const bool verify : {false, true}) checkProjection(backend, rotated, float32, verify);
    checkGather(backend);
    for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16}) {
      // Prefill tiles whose rows straddle their window's start: from the
      // history's start, after a cached prefix on either side of a window
      // and of a page, and long chunks after a long prefix, whose first tiles
      // read none of the pages below their windows; a chunk again as several.
      checkPrefill(backend, format, kWindowed, 0, 2048);
      for (const uint32_t history : {31U, 32U, 511U, 512U, 513U, 544U})
        for (const uint32_t rows : {1U, 8U, 33U}) checkPrefill(backend, format, kWindowed, history, rows);
      checkPrefill(backend, format, kWindowed, 4103, 2048, {509});
      checkPrefill(backend, format, kWindowed, 700, 1057, {31, 512});
      checkPrefill(backend, format, kWindowed, 505, 13, {1, 3});
      checkPrefill(backend, format, kWhole, 0, 300);
      checkPrefill(backend, format, kWhole, 300, 517, {129});
      // Verify lanes of one to four, each with a history of its own: before,
      // at and past their windows, up to 10000 tokens.
      for (const std::vector<uint32_t> &histories : std::vector<std::vector<uint32_t>>{
               {0}, {511, 512}, {513, 544, 1023}, {0, 505, 4097, 10000}, {31, 32, 4096, 7}})
        checkVerify(backend, format, kWindowed, histories);
      checkVerify(backend, format, kWhole, std::vector<uint32_t>{0, 31, 1000, 2047});
      checkSplitCounts(backend, format);
      checkWindowWithinHistory(backend, format);
    }
    std::cout << "attention variants: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "attention variants: FAIL: " << error.what() << '\n';
    return 1;
  }
}
