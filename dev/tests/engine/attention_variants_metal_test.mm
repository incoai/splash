#include "HostKvExtents.hpp"
#include "LinearInputReference.hpp"
#include "LinearNumerics.hpp"
#include "NormReference.hpp"
#include "StagedKvStore.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "TestFamilies.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "model/RopeScaling.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/RoPE.hpp"
#include "ops/KernelCalibration.hpp"
#include "ops/LaneRows.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <utility>
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
// The geometry among the families' (TestFamilies.hpp): Flint's.
constexpr test::AttentionGeometry kGeometry{kQueryHeads, kKvHeads, kHeadDimension, kv::Reads::EveryPage};
constexpr uint32_t kWindow = 512, kRotaryPairs = kHeadDimension / 2;
constexpr uint32_t kPageTokens = kv::kPageTokens;
// A lane of the narrow cap's rows (a decode lane of up to eight), one
// attention tile.
constexpr uint32_t kLaneRows = ops::kNarrowLaneRows;
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
  // INT8 pages score against INT8 queries in every split (the q8q entries),
  // which the reference quantizes as the kernel does.
  bool int8Queries;
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
  // The plans' attention row: INT8 queries in every split, or none.
  [[nodiscard]] ops::AttentionCalibration attention() const {
    return test::attentionRow(int8Queries ? 1U : ops::kNever);
  }

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
  Case data{layoutOf(format), variant, int8Queries, lanes, rows, histories,
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
    data.pool.writeTable(table, data.tables[lane]);
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
// [dimension][row]; and the attention's output, the out-projection's input
// rows [lane][row][query head][dimension].
struct Staging final {
  uint32_t first;
  uint32_t rows;
  uint32_t stride;
  metal::MetalBuffer keys, values, queries, output;
  // Each lane's rows of the command, lane after lane (SplashLaneRows).
  metal::MetalBuffer layout;

  [[nodiscard]] uint64_t queryIndex(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) const {
    return (((uint64_t{lane} * kKvHeads + head / kGroup) * stride + row) * kGroup + head % kGroup) * kHeadDimension +
           dimension;
  }
  [[nodiscard]] uint64_t outputIndex(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) const {
    return ((uint64_t{lane} * rows + row) * kQueryHeads + head) * kHeadDimension + dimension;
  }
};

Staging stage(metal::MetalBackend &backend, const Case &data, uint32_t first, uint32_t rows, uint32_t stride) {
  const uint64_t kvElements = uint64_t{data.lanes} * kKvHeads * stride * kHeadDimension;
  Staging staging{first,
                  rows,
                  stride,
                  zeroed(backend, kvElements * 2),
                  zeroed(backend, kvElements * 2),
                  zeroed(backend, kvElements * kGroup * 2),
                  zeroed(backend, kvElements * kGroup * 2),
                  zeroed(backend, uint64_t{data.lanes} * sizeof(SplashLaneRows))};
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    contents<SplashLaneRows>(staging.layout)[lane] = {lane * rows, rows};
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
// the kernels' bf16 probabilities. With INT8 queries the reference's queries
// are the kernel's codes times their row's scale (test::int8Block).
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
  if (data.int8Queries) {
    float largest = 0;
    for (const double value : queries) largest = std::max(largest, float(std::fabs(value)));
    const test::Int8Block block = test::int8Block(largest);
    for (double &value : queries) value = double(test::int8Code(float(value), block)) * block.scale;
  }
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
    const uint32_t tile = row / kv::kAttentionTileRows, tiles = kv::attentionTiles(rows);
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
              bf16ToFloat(contents<uint16_t>(staging.output)[staging.outputIndex(lane, head, row, dimension)]);
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

// The sequences of a case's staged rows: each lane's after its history and
// the rows the staging starts at.
std::vector<kv::ChunkedPrefillParams> sequencesOf(const Case &data, const Staging &staging) {
  std::vector<kv::ChunkedPrefillParams> sequences;
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    sequences.push_back(ops::PagedAttention::chunkParams(data.histories[lane] + staging.first, staging.rows,
                                                         staging.stride, uint32_t(data.pages[lane].size())));
  return sequences;
}

// The plan of a case's staged rows, every lane in one dispatch.
ops::AttentionPlan planOf(const Case &data, const Staging &staging) {
  return ops::PagedAttention::plan(
      sequencesOf(data, staging), kQueryHeads, data.layout, data.attention(), ops::Int8Activations::On,
      ops::PagedAttention::workspace(data.lanes * ops::kAttentionSequenceSplits, kQueryHeads, data.layout),
      data.variant);
}

// The encoding of a case's staged rows: their K/V into the pages (the test
// libraries' store of staged rows, test::addStagedStore), then the attention
// as the runtime encodes it after the q/k projection, a verify's over its
// tree rows where it has them, into `output` (every row into the staging's
// output rows unless it says otherwise; the geometry reads no packed row).
// Returns the input the attention prepared with the graph.
std::pair<metal::CommandGraph, ops::PreparedInput> encode(const Case &data, const Staging &staging,
                                                          const ops::AttentionPlan &plan,
                                                          const metal::MetalBuffer &partials,
                                                          const metal::MetalBuffer &statistics,
                                                          const metal::MetalBuffer &tree = {},
                                                          const ops::AttentionOutputRows *output = nullptr) {
  metal::CommandGraph graph;
  const std::vector<kv::ChunkedPrefillParams> sequences = sequencesOf(data, staging);
  const ops::AttentionRows rows{staging.layout, data.lanes * staging.rows, tree};
  test::addStagedStore(graph, data.layer, staging.keys, staging.values, std::span(data.tables).first(data.lanes),
                       sequences, rows, data.layout);
  std::vector<ops::AttentionHistory> histories;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) histories.push_back({data.layer, data.tables[lane]});
  const ops::PreparedInput prepared =
      ops::PagedAttention::addAttention(graph, plan, staging.queries, partials, statistics, sequences, histories,
                                        rows, output ? *output : ops::AttentionOutputRows{{}, staging.output},
                                        data.sinks);
  return {std::move(graph), prepared};
}

// A plan's kernels and grids: the variant's split and reduce of the
// geometry, the reduce a thread per dimension, and each lane's split count,
// of its rows' tiles and the pages they read of its window.
void checkPlan(const Case &data, const Staging &staging, const ops::AttentionPlan &plan,
               const metal::CommandGraph &graph) {
  const std::string window = data.variant.windowTokens ? "_w" + std::to_string(data.variant.windowTokens) : "";
  const std::string format = data.layout.format == kv::Format::Int8 ? "q8" : "bf16";
  require(plan.sinks &&
              plan.splitPipeline() ==
                  "attention_" + (data.int8Queries ? std::string("q8q") : format) + "_split_kv4_g8_d128" + window &&
              plan.reducePipeline() == "attention_reduce_kv4_g8_d128" + window + "_sinks",
          "the plan names another variant's kernels");
  const auto &dispatches = graph.dispatches();
  require(dispatches.size() == 3 && dispatches[0].pipelineName == "test_staged_kv_store_" + format + "_kv4_d128" &&
              dispatches[0].threadsPerThreadgroup.x == kHeadDimension &&
              dispatches[2].threadsPerThreadgroup.x == kHeadDimension && dispatches[2].buffers.size() == 8 &&
              dispatches[2].buffers[3].buffer.sameView(staging.output) &&
              dispatches[2].buffers[7].buffer.sameView(data.sinks),
          "the store and reduce run a thread per dimension, the reduce with the sinks");
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    require(plan.sequences()[lane].splits == test::attentionSplits(staging.rows, data.histories[lane] + staging.first,
                                                                   false, data.variant.windowTokens),
            "a lane's splits are not its rows' and window's");
}

void run(metal::MetalBackend &backend, const Case &data, const Staging &staging, const ops::AttentionPlan &plan,
         const metal::MetalBuffer &tree = {}) {
  const metal::MetalBuffer partials = poisoned(backend, plan.workspace().partialsBytes);
  const metal::MetalBuffer statistics = poisoned(backend, plan.workspace().statisticsBytes);
  const metal::CommandGraph graph = encode(data, staging, plan, partials, statistics, tree).first;
  checkPlan(data, staging, plan, graph);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

// A case's KV format, and its queries where they are INT8.
std::string formatLabel(kv::Format format, bool int8Queries) {
  return std::string(kv::formatName(format)) + (int8Queries ? " with INT8 queries" : "");
}

std::string describe(const Case &data, std::string_view phase) {
  std::string text = std::string(phase) + " " + formatLabel(data.layout.format, data.int8Queries) +
                     (data.variant.windowTokens ? " window " + std::to_string(data.variant.windowTokens) : " whole") +
                     " history";
  for (uint32_t lane = 0; lane < data.lanes; ++lane) text += " " + std::to_string(data.histories[lane]);
  return text;
}

// A prefill of `rows` rows after a cached prefix of `history` tokens, as one
// chunk, and again as the chunks `split` lists, each over the K/V its
// predecessors stored, each row of each against the reference.
void checkPrefill(metal::MetalBackend &backend, kv::Format format, bool int8Queries, ops::AttentionVariant variant,
                  uint32_t history, uint32_t rows, std::vector<uint32_t> split = {}) {
  const Case data = makeCase(backend, format, int8Queries, variant, 1, rows, {history, 0, 0, 0});
  const std::string what = describe(data, "prefill") + " rows " + std::to_string(rows);
  double largest = 0;
  split.insert(split.begin(), rows);
  bool whole = true;
  for (const uint32_t chunk : split) {
    for (uint32_t first = 0; first < rows; first += chunk) {
      const uint32_t chunkRows = std::min(chunk, rows - first);
      const Staging staging = stage(backend, data, first, chunkRows, (chunkRows + kPageTokens - 1) / kPageTokens * kPageTokens);
      run(backend, data, staging, planOf(data, staging));
      checkStore(data, staging, what);
      largest = std::max(largest, checkOutput(data, staging, what + (whole ? "" : " in chunks of " + std::to_string(chunk))));
    }
    whole = false;
  }
  std::cout << what << (split.size() > 1 ? " and in chunks" : "") << ": " << data.poisoned
            << " table entries of NaNs below the window, largest error " << largest << " of its bound PASS\n";
}

void checkVerify(metal::MetalBackend &backend, kv::Format format, bool int8Queries, ops::AttentionVariant variant,
                 std::span<const uint32_t> histories) {
  std::array<uint32_t, 4> lanes{};
  std::copy(histories.begin(), histories.end(), lanes.begin());
  const auto count = uint32_t(histories.size());
  const Case data = makeCase(backend, format, int8Queries, variant, count, kLaneRows, lanes);
  const Staging staging = stage(backend, data, 0, kLaneRows, kPageTokens);
  run(backend, data, staging, planOf(data, staging));
  const std::string what = describe(data, "verify");
  checkStore(data, staging, what);
  const double largest = checkOutput(data, staging, what);
  std::cout << what << ": " << data.poisoned << " table entries of NaNs below the windows, largest error "
            << largest << " of its bound PASS\n";
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

// The fp64 attention of tree row `row` of a lane's block after its history,
// as reference() computes a chain's: its position is the anchor's plus its
// depth, it attends that position's window of the history and, of the
// block, its path's rows, whose K/V the store left at the slots of their
// rows.
Reference treeReference(const Case &data, const StoredKv &kv, uint32_t lane, const Tree &tree, uint32_t row,
                        uint32_t head) {
  const uint32_t history = data.histories[lane], kvHead = head / kGroup;
  const std::vector<uint32_t> path = tree.path(row);
  std::vector<uint32_t> tokens;
  for (uint32_t token = windowStart(data.variant, history + uint32_t(path.size()) - 1); token < history; ++token)
    tokens.push_back(token);
  for (const uint32_t on : path) tokens.push_back(history + on);
  const double sink = contents<float>(data.sinks)[head];
  std::array<double, kHeadDimension> queries{};
  for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
    queries[dimension] = query(lane, head, row, dimension);
  if (data.int8Queries) {
    float largest = 0;
    for (const double value : queries) largest = std::max(largest, float(std::fabs(value)));
    const test::Int8Block block = test::int8Block(largest);
    for (double &value : queries) value = double(test::int8Code(float(value), block)) * block.scale;
  }
  std::vector<double> scores(tokens.size());
  double maximum = sink;
  for (size_t i = 0; i < tokens.size(); ++i) {
    const double *keys = &kv.keys[kv.index(kvHead, tokens[i], 0)];
    double score = 0;
    for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) score += queries[dimension] * keys[dimension];
    scores[i] = score / std::sqrt(double(kHeadDimension));
    maximum = std::max(maximum, scores[i]);
  }
  double denominator = std::exp(sink - maximum);
  for (double &score : scores) {
    score = std::exp(score - maximum);
    denominator += score;
  }
  Reference result;
  for (size_t i = 0; i < tokens.size(); ++i) {
    const double weight = scores[i] / denominator;
    const double *values = &kv.values[kv.index(kvHead, tokens[i], 0)];
    for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
      result.output[dimension] += weight * values[dimension];
      result.mass[dimension] += weight * std::fabs(values[dimension]);
    }
  }
  return result;
}

// Each lane's tree rows, its block `blocks[lane]` of the step's rows, against
// treeReference, within checkOutput's bound. Returns the largest error in
// units of that bound.
double checkTreeRows(const Case &data, const Staging &staging, std::span<const Tree> trees,
                     std::span<const SplashLaneRows> blocks, const std::string &what) {
  double largest = 0;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    const uint32_t history = data.histories[lane];
    const StoredKv kv = stored(data, lane, windowStart(data.variant, history), history + blocks[lane].rows);
    for (uint32_t row = 0; row < blocks[lane].rows; ++row)
      for (uint32_t head = 0; head < kQueryHeads; ++head) {
        const Reference expected = treeReference(data, kv, lane, trees[lane], row, head);
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const double actual = bf16ToFloat(
              contents<uint16_t>(staging.output)[staging.outputIndex(0, head, blocks[lane].first + row, dimension)]);
          const double bound = expected.mass[dimension] / 256 + ulpBf16(float(expected.output[dimension]));
          const double error = std::fabs(actual - expected.output[dimension]);
          if (!(std::isfinite(actual) && error <= bound))
            throw std::runtime_error(what + ": lane " + std::to_string(lane) + " tree row " + std::to_string(row) +
                                     " head " + std::to_string(head) + " dimension " + std::to_string(dimension) +
                                     " is " + std::to_string(actual) + ", the reference " +
                                     std::to_string(expected.output[dimension]) + " (bound " +
                                     std::to_string(bound) + ")");
          largest = std::max(largest, error / bound);
        }
      }
  }
  return largest;
}

// A verify of lanes whose blocks of two tiles' rows are trees, each row
// against treeReference: a chain, a lookup branch off the draft's third row
// that runs through the second tile, two children a row, and a random tree,
// so that rows of the second tile attend ancestors of the first, whose K/V
// the store wrote before the attention. Without a window the chain's rows
// are bit for bit those of the same block attended causally.
void checkTreeVerify(metal::MetalBackend &backend, kv::Format format, bool int8Queries, ops::AttentionVariant variant,
                     std::span<const uint32_t> histories) {
  constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
  static_assert(kRows == 2 * kv::kAttentionTileRows, "a block of two tiles");
  std::array<uint32_t, 4> lanes{};
  std::copy(histories.begin(), histories.end(), lanes.begin());
  const auto count = uint32_t(histories.size());
  const Case data = makeCase(backend, format, int8Queries, variant, count, kRows, lanes);
  std::array<Tree, 4> trees;
  for (uint32_t row = 0; row < kRows; ++row) {
    trees[0].parent.push_back(row ? row - 1 : 0);
    trees[1].parent.push_back(row == 8 ? 2 : row ? row - 1 : 0);
    trees[2].parent.push_back(row ? (row - 1) / 2 : 0);
    const uint32_t back = std::min(row, 1 + uint32_t((unit(seedOf(4, 3, 0, row, 0)) + 1) * 2) % 4);
    trees[3].parent.push_back(row ? row - back : 0);
  }
  const metal::MetalBuffer tree = zeroed(backend, uint64_t{count} * kRows * sizeof(SplashTreeRow));
  for (uint32_t lane = 0; lane < count; ++lane)
    for (uint32_t row = 0; row < kRows; ++row) contents<SplashTreeRow>(tree)[lane * kRows + row] = trees[lane].place(row);
  const Staging staging = stage(backend, data, 0, kRows, kPageTokens);
  const ops::AttentionPlan plan = planOf(data, staging);
  run(backend, data, staging, plan, tree);
  const std::string what = describe(data, "tree verify");
  checkStore(data, staging, what);
  std::array<SplashLaneRows, 4> blocks{};
  for (uint32_t lane = 0; lane < count; ++lane) blocks[lane] = {lane * kRows, kRows};
  const double largest = checkTreeRows(data, staging, trees, std::span(blocks).first(count), what);
  if (!variant.windowTokens) {
    const std::vector<uint16_t> trees16(contents<uint16_t>(staging.output),
                                        contents<uint16_t>(staging.output) + staging.output.sizeBytes() / 2);
    run(backend, data, staging, plan);
    for (uint32_t row = 0; row < kRows; ++row)
      for (uint32_t head = 0; head < kQueryHeads; ++head)
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const uint64_t at = staging.outputIndex(0, head, row, dimension);
          require(trees16[at] == contents<uint16_t>(staging.output)[at],
                  what + ": a chain's tree rows differ from its causal rows");
        }
  }
  std::cout << what << ": largest error " << largest << " of its bound PASS\n";
}

// A verify step whose four lanes hold blocks of 16, 5, 9 and 12 rows packed
// by their first rows, 0, 16, 21 and 30, as a step's layout places each
// lane's rows after the lane's before it, each block a tree that branches: a
// chain with a lookup branch off row 2 at row 8, two children a row, a random
// tree and two children a row again. Each lane's rows against treeReference
// at their rows of the step. The tree rows of no lane's row hold 0xFF, whose
// ancestors read as a chain's, so that a lane that took its tree rows at
// another row than its first (lanes 2 and 3 at lane x 16) would attend
// another lane's tree or a chain: over the short histories `histories` gives
// those lanes, its rows would move past the bound.
void checkRaggedTreeVerify(metal::MetalBackend &backend, kv::Format format, bool int8Queries,
                           ops::AttentionVariant variant, const std::array<uint32_t, 4> &histories) {
  constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr std::array<uint32_t, 4> kBlockRows{16, 5, 9, 12};
  const Case data = makeCase(backend, format, int8Queries, variant, 4, kRows, histories);
  const Staging staging = stage(backend, data, 0, kRows, kPageTokens);
  const metal::MetalBuffer tree = poisoned(backend, uint64_t{4} * kRows * sizeof(SplashTreeRow));
  std::array<Tree, 4> trees;
  std::array<SplashLaneRows, 4> blocks{};
  for (uint32_t lane = 0; lane < 4; ++lane) {
    blocks[lane] = {lane ? blocks[lane - 1].first + blocks[lane - 1].rows : 0, kBlockRows[lane]};
    for (uint32_t row = 0; row < kBlockRows[lane]; ++row) {
      const uint32_t back = std::min(row, 1 + uint32_t((unit(seedOf(5, lane, 0, row, 0)) + 1) * 2) % 4);
      trees[lane].parent.push_back(!row       ? 0
                                   : lane == 0 ? (row == 8 ? 2 : row - 1)
                                   : lane == 2 ? row - back
                                               : (row - 1) / 2);
      contents<SplashTreeRow>(tree)[blocks[lane].first + row] = trees[lane].place(row);
    }
    contents<SplashLaneRows>(staging.layout)[lane] = blocks[lane];
  }
  run(backend, data, staging, planOf(data, staging), tree);
  const std::string what = describe(data, "ragged tree verify");
  const double largest = checkTreeRows(data, staging, trees, blocks, what);
  std::cout << what << ": lanes of 16, 5, 9 and 12 tree rows at rows 0, 16, 21 and 30, largest error " << largest
            << " of its bound PASS\n";
}

// Split and reduce derive the same pages from the window for any split
// count, so every count the partial workspace holds gives the reference,
// the reduce combining exactly the splits that wrote: each lane of a decode
// step with its split count set, past the window's pages as well, and a lane
// alone.
void checkSplitCounts(metal::MetalBackend &backend, kv::Format format, bool int8Queries) {
  const auto counts = [&](const Case &data, std::span<const uint32_t> splitCounts) {
    const Staging staging = stage(backend, data, 0, kLaneRows, kPageTokens);
    const ops::AttentionPlan plan = planOf(data, staging);
    run(backend, data, staging, plan);
    // The workspace production allocates for the lanes, which holds every
    // split count's slots, poisoned again before each.
    const ops::AttentionWorkspace workspace =
        ops::PagedAttention::workspace(data.lanes * kv::kAttentionMaximumSplits, kQueryHeads, data.layout);
    const metal::MetalBuffer partials = poisoned(backend, workspace.partialsBytes);
    const metal::MetalBuffer statistics = poisoned(backend, workspace.statisticsBytes);
    const std::vector<kv::ChunkedPrefillParams> sequences = sequencesOf(data, staging);
    for (const uint32_t splits : splitCounts) {
      std::memset(partials.contents(), 0xff, partials.sizeBytes());
      std::memset(statistics.contents(), 0xff, statistics.sizeBytes());
      std::memset(staging.output.contents(), 0xff, staging.output.sizeBytes());
      SplashAttentionParams params{.sequences = data.lanes, .first_sequence = 0, .rows = data.lanes * kLaneRows,
                                   .tiles = data.lanes, .trees = 0, .last_rows = 0, .kv = data.layer,
                                   .sequence = {}};
      for (uint32_t lane = 0; lane < data.lanes; ++lane)
        params.sequence[lane] = {.committed_tokens = sequences[lane].committed_tokens,
                                 .row_bound = kLaneRows,
                                 .layout_row = 0,
                                 .stage_row = lane * kKvHeads * kPageTokens,
                                 .chunk_stride = kPageTokens,
                                 .page_table_entries = sequences[lane].page_table_entries,
                                 .split_count = splits,
                                 .first_tile = lane,
                                 .first_slot = lane * kKvHeads * splits,
                                 .block_stride = 0,
                                 .block_list = 0};
      metal::CommandGraph graph;
      // Without tree rows the split binds another buffer in their place.
      graph.add(plan.splitPipeline(),
                {staging.queries, partials, statistics, data.tables[0], data.tables[1], data.tables[2], data.tables[3],
                 partials, staging.layout},
                params, {kKvHeads, data.lanes, splits});
      // The ungated plain reduce reads no packed row and writes no table, so
      // the output rows take those bindings too.
      graph.add(plan.reducePipeline(),
                {partials, statistics, staging.output, staging.output, staging.output, staging.output, staging.layout,
                 data.sinks},
                params, {kKvHeads, kLaneRows * kGroup, data.lanes}, {kHeadDimension, 1, 1});
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      checkOutput(data, staging, describe(data, "lanes") + " of " + std::to_string(splits) + " splits");
    }
  };
  constexpr std::array<uint32_t, 9> laneCounts{1, 2, 3, 7, 17, 18, 19, 32, 128};
  counts(makeCase(backend, format, int8Queries, kWindowed, 3, kLaneRows, {4097, 515, 10000, 0}), laneCounts);
  std::array<uint32_t, kv::kAttentionMaximumSplits> every{};
  std::iota(every.begin(), every.end(), 1U);
  counts(makeCase(backend, format, int8Queries, kWindowed, 1, kLaneRows, {4103, 0, 0, 0}), every);
  std::cout << "split counts: " << formatLabel(format, int8Queries) << " three lanes and one lane of 1-128 splits PASS\n";
}

// Where every row's window holds its whole history, the window changes
// nothing: the windowed kernels' outputs are the whole history's bit for bit
// at the same split partition (a page a split in a verify, a prefill's
// splits below the window's pages).
void checkWindowWithinHistory(metal::MetalBackend &backend, kv::Format format, bool int8Queries) {
  const auto outputs = [&](ops::AttentionVariant variant, bool verify, uint32_t lanes, uint32_t rows,
                           std::array<uint32_t, 4> histories) {
    const Case data = makeCase(backend, format, int8Queries, variant, lanes, rows, histories);
    const Staging staging = stage(backend, data, 0, rows, verify ? kPageTokens : (rows + 31) / 32 * 32);
    run(backend, data, staging, planOf(data, staging));
    const auto *begin = contents<uint16_t>(staging.output);
    return std::vector<uint16_t>(begin, begin + staging.output.sizeBytes() / 2);
  };
  const ops::AttentionVariant whole{.rotated = true, .windowTokens = 0};
  require(outputs(kWindowed, true, 4, 8, {0, 31, 200, 504}) == outputs(whole, true, 4, 8, {0, 31, 200, 504}),
          "a windowed verify inside its window differs from the whole history's");
  require(outputs(kWindowed, false, 1, 300, {0, 0, 0, 0}) == outputs(whole, false, 1, 300, {0, 0, 0, 0}) &&
              outputs(kWindowed, false, 1, 24, {200, 0, 0, 0}) == outputs(whole, false, 1, 24, {200, 0, 0, 0}),
          "a windowed prefill inside its window differs from the whole history's");
  std::cout << "windows within the history: " << formatLabel(format, int8Queries)
            << " bit for bit the whole history's PASS\n";
}

// A NaN in one row's query of a head and an infinity in another's turn those
// rows' scores NaN, with INT8 queries (the row's scale is NaN:
// kernels/common/int8_activations.h) as with bf16 queries: their softmax
// statistics are NaN, which the reduce writes as a row of zeros, as it writes
// a row without keys. Every other output stays as it was, bit for bit.
void checkNonFiniteQueries(metal::MetalBackend &backend, bool int8Queries) {
  constexpr uint32_t kRows = 24, kHistory = 300;
  const Case data = makeCase(backend, kv::Format::Int8, int8Queries, kWhole, 1, kRows, {kHistory, 0, 0, 0});
  const Staging staging = stage(backend, data, 0, kRows, kPageTokens);
  const ops::AttentionPlan plan = planOf(data, staging);
  const auto outputs = [&] {
    run(backend, data, staging, plan);
    const auto *begin = contents<uint16_t>(staging.output);
    return std::vector<uint16_t>(begin, begin + staging.output.sizeBytes() / 2);
  };
  const std::vector<uint16_t> clean = outputs();
  struct Poison final {
    uint32_t row, head, dimension;
    uint16_t bits;
  };
  constexpr std::array<Poison, 2> kPoisons{{{3, 5, 17, 0x7FC0}, {17, 26, 100, 0x7F80}}};
  for (const Poison &poison : kPoisons)
    contents<uint16_t>(staging.queries)[staging.queryIndex(0, poison.head, poison.row, poison.dimension)] =
        poison.bits;
  const std::vector<uint16_t> poisoned = outputs();
  for (uint32_t row = 0; row < kRows; ++row)
    for (uint32_t head = 0; head < kQueryHeads; ++head) {
      const bool hit = std::any_of(kPoisons.begin(), kPoisons.end(), [&](const Poison &poison) {
        return poison.row == row && poison.head == head;
      });
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        const uint64_t at = staging.outputIndex(0, head, row, dimension);
        require(hit ? bf16ToFloat(poisoned[at]) == 0.0f : poisoned[at] == clean[at],
                formatLabel(kv::Format::Int8, int8Queries) + ": row " + std::to_string(row) + " head " +
                    std::to_string(head) + " output " + std::to_string(bf16ToFloat(poisoned[at])) +
                    (hit ? " beside a non-finite query, not 0" : " changed beside another row's non-finite query"));
      }
    }
  std::cout << "non-finite queries: " << formatLabel(kv::Format::Int8, int8Queries)
            << " write their rows alone as zeros PASS\n";
}

// The plans of the variants, and what each refuses: a variant the geometry
// is not compiled for, and a reduce's sinks where the plan has none or the
// layer gives too few.
void checkPlans(metal::MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  const kv::Layout qwen{.attentionLayers = 1, .kvHeads = 4, .headDimension = 256};
  constexpr ops::Int8Activations off = ops::Int8Activations::Off;
  // The plan of one sequence of `rows` rows after `history` tokens.
  const auto planOf = [&](uint32_t rows, uint32_t history, uint32_t heads, const kv::Layout &of,
                          ops::AttentionVariant variant = {}) {
    const std::array sequence{ops::PagedAttention::chunkParams(history, rows, (rows + 31) / 32 * 32,
                                                               (history + rows + 31) / 32)};
    return ops::PagedAttention::plan(sequence, heads, of, ops::kApple10Attention, off,
                                     ops::PagedAttention::workspace(ops::kAttentionSequenceSplits, heads, of), variant);
  };
  const auto splitsAfter = [&](uint32_t rows, uint32_t history, ops::AttentionVariant variant) {
    return planOf(rows, history, kQueryHeads, layout, variant).sequences()[0].splits;
  };
  // A window of 512 keys reads at most 18 pages, so a lane takes no more
  // splits; a whole layer's lane takes up to 128.
  require(kv::windowAttentionPages(512) == 18 && splitsAfter(8, 0, kWindowed) == 18 &&
              splitsAfter(8, 1'000'000, kWindowed) == 18 && splitsAfter(8, 1'000'000, kWhole) == 128,
          "a window of 512 keys reads at most 18 pages");
  // A windowed tile reads its window's pages whatever its history, so a
  // windowed chunk keeps its splits as the history grows; a whole layer's
  // chunk takes more.
  require(splitsAfter(64, 1024, kWindowed) == splitsAfter(64, 65536, kWindowed) &&
              splitsAfter(64, 1024, kWhole) < splitsAfter(64, 65536, kWhole),
          "a windowed chunk's splits follow its history, or a whole chunk's do not");
  for (const ops::AttentionVariant variant : {ops::AttentionVariant{.rotated = true, .windowTokens = 256},
                                              ops::AttentionVariant{.rotated = false, .windowTokens = 1024}})
    rejects([&] { (void)planOf(8, 0, kQueryHeads, layout, variant); }, "no attention kernel for the layer's variant",
            "a window the geometry has no kernels for was accepted");
  for (const ops::AttentionVariant variant : {kWindowed, kWhole})
    rejects([&] { (void)planOf(8, 0, 24, qwen, variant); }, "no attention kernel for the layer's variant",
            "a variant of a geometry without it was accepted");
  require(!planOf(8, 0, 24, qwen).sinks && planOf(8, 0, 24, qwen).reducePipeline() == "attention_reduce",
          "a geometry without sinks takes none");
  const metal::MetalBuffer buffer = zeroed(backend, 1 << 23), sinks = zeroed(backend, kQueryHeads * 4);
  const metal::MetalBuffer fewSinks = zeroed(backend, kQueryHeads * 4 - 4);
  const std::array chunk{ops::PagedAttention::chunkParams(0, 8, 32, 1)};
  const ops::AttentionOutputRows output{buffer, buffer};
  const std::array histories{ops::AttentionHistory{{1, 0}, buffer}};
  const ops::AttentionRows rows{buffer, 8};
  const auto plan = planOf(8, 0, kQueryHeads, layout, kWindowed);
  metal::CommandGraph graph;
  rejects([&] {
    ops::PagedAttention::addAttention(graph, plan, buffer, buffer, buffer, chunk, histories, rows, output);
  }, "attention sinks buffer holds", "a plan with sinks encoded without them");
  rejects([&] {
    ops::PagedAttention::addAttention(graph, plan, buffer, buffer, buffer, chunk, histories, rows, output, fewSinks);
  }, "attention sinks buffer holds", "a plan with sinks encoded with too few");
  rejects([&] {
    ops::PagedAttention::addAttention(graph, planOf(8, 0, 24, qwen), buffer, buffer, buffer, chunk, histories, rows,
                                      output, sinks);
  }, "the attention kernels take no sinks", "a plan without sinks encoded with some");
  require(graph.empty(), "a refused attention encoded a dispatch");
  std::cout << "variant plans PASS\n";
}

// The q/k norms and RoPE of the geometry's projection rows, without a gate:
// 32 query heads, then 4 K and 4 V heads, of 128. Rotated, each pair
// (i, i + 64) of the normalized head rotates by its row's table entry, NeoX's
// half split of the whole head; unrotated (NoPE) the head is its norm, and no
// table is bound. Against fp64 with norm weights of each type the geometry's
// families store (test::familyNorms), as paged_attention_plan_test's check of
// the gated geometries; the queries staged, the keys and values in BF16 page
// slots, the values' bits as the rows hold them, and INT8 ones a key within
// half its scale of the bound and a value's code within one of its over the
// scale (paged_attention_plan_test's checkProjection).
void checkProjection(metal::MetalBackend &backend, kv::Format format, bool rotated, bool float32, bool verify) {
  const ops::AttentionVariant variant{.rotated = rotated, .windowTokens = 0};
  const kv::Layout layout = layoutOf(format);
  const bool int8 = format == kv::Format::Int8;
  const uint32_t lanes = verify ? 3 : 1, rows = verify ? kLaneRows : 37, stride = verify ? kPageTokens : 64;
  const uint32_t packedWidth = (kQueryHeads + 2 * kKvHeads) * kHeadDimension;
  const metal::MetalBuffer packed = zeroed(backend, uint64_t{lanes} * rows * packedWidth * 2);
  const metal::MetalBuffer ropeCos = rotated ? zeroed(backend, uint64_t{lanes} * rows * kRotaryPairs * 4) : metal::MetalBuffer{};
  const metal::MetalBuffer ropeSin = rotated ? zeroed(backend, ropeCos.sizeBytes()) : metal::MetalBuffer{};
  const metal::MetalBuffer queries = zeroed(backend, uint64_t{lanes} * kQueryHeads * stride * kHeadDimension * 2);
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
  // Each lane a sequence of `rows` rows, staged after the lanes before it,
  // its pages after the lanes' before it.
  std::vector<kv::ChunkedPrefillParams> sequences;
  const metal::MetalBuffer layoutRows = zeroed(backend, uint64_t{lanes} * sizeof(SplashLaneRows));
  uint32_t allPages = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    sequences.push_back(ops::PagedAttention::chunkParams(100 * lane, rows, stride, (100 * lane + rows + 31) / 32));
    contents<SplashLaneRows>(layoutRows)[lane] = {lane * rows, rows};
    allPages += sequences.back().page_table_entries;
  }
  const auto spread = HostKvExtents::spread(allPages);
  HostKvExtents pool(backend, layout, spread.extentPages, spread.extents);
  std::vector<std::vector<uint32_t>> pageIds;
  std::vector<metal::MetalBuffer> tables;
  for (uint32_t lane = 0, first = 0; lane < lanes; first += sequences[lane++].page_table_entries) {
    pageIds.emplace_back(sequences[lane].page_table_entries);
    std::iota(pageIds.back().begin(), pageIds.back().end(), first);
    tables.push_back(zeroed(backend, uint64_t{sequences[lane].page_table_entries} * sizeof(SplashKvPage)));
    pool.writeTable(pageIds.back(), tables.back());
  }
  metal::CommandGraph graph;
  ops::PagedAttention::addProjection(graph, packed, queryNorm, keyNorm, ropeCos, ropeSin, queries, pool.layer(0),
                                     tables, sequences, {layoutRows, lanes * rows}, kQueryHeads, layout, variant);
  const std::string kernel = std::string(int8 ? "attention_q8_qkv_kv4_g8_d128" : "attention_bf16_qkv_kv4_g8_d128") +
                             (rotated ? "" : "_nope") + (float32 ? "_f32" : "");
  require(graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == kernel &&
              graph.dispatches()[0].threadsPerThreadgroup.x == kHeadDimension &&
              graph.dispatches()[0].buffers.size() == (rotated ? 11U : 9U),
          "the projection runs another kernel or binds tables it does not read");
  // A row's stored element of a K/V tensor: a BF16 page's value, or an INT8
  // page's code times its row's scale, and the scale.
  const auto stored = [&](uint32_t lane, uint32_t row, bool value, uint32_t head, uint32_t d) {
    const uint32_t token = sequences[lane].committed_tokens + row, id = pageIds[lane][token / 32], slot = token % 32;
    const uint64_t index =
        value ? pool.valueElement(head, slot, d) : splash_kv_key_element(head, slot, d, kHeadDimension);
    if (!int8)
      return std::pair{double(bf16ToFloat(pool.slab<uint16_t>(0, value ? SPLASH_KV_VALUES : SPLASH_KV_KEYS, id)[index])),
                       0.0};
    const double scale =
        pool.slab<float>(0, value ? SPLASH_KV_VALUE_SCALES : SPLASH_KV_KEY_SCALES, id)[splash_kv_scale_element(head, slot)];
    return std::pair{double(pool.slab<int8_t>(0, value ? SPLASH_KV_VALUES : SPLASH_KV_KEYS, id)[index]) * scale, scale};
  };
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
            contents<uint16_t>(queries) +
            (((uint64_t{lane} * kKvHeads + h / kGroup) * stride + row) * kGroup + h % kGroup) * kHeadDimension;
        const std::vector<double> normalized = test::rmsNorm(source, isQuery ? queryNorm : keyNorm, kHeadDimension);
        // An output of the head: a query's staged bf16, or a key as its page
        // stores it, within half its scale (none for BF16) more.
        const auto output = [&](uint32_t d) {
          return isQuery ? std::pair{double(bf16ToFloat(out[d])), 0.0} : stored(lane, row, false, h, d);
        };
        for (uint32_t d = 0; d < kRotaryPairs; ++d) {
          if (!rotated) {
            for (const uint32_t at : {d, d + kRotaryPairs}) {
              const auto [actual, scale] = output(at);
              require(scale ? std::fabs(actual - normalized[at]) <= ulpBf16(float(normalized[at])) + 0.51 * scale
                            : test::roundedOnceToBf16(floatToBf16(float(actual)), normalized[at]),
                      "the unrotated prepare is not the norm rounded once");
            }
            continue;
          }
          const double first = bf16ToFloat(floatToBf16(float(normalized[d])));
          const double second = bf16ToFloat(floatToBf16(float(normalized[d + kRotaryPairs])));
          const double c = contents<float>(ropeCos)[rope + d], s = contents<float>(ropeSin)[rope + d];
          const double rotatedHalves[2] = {first * c - second * s, second * c + first * s};
          for (uint32_t half = 0; half < 2; ++half) {
            const auto [actual, scale] = output(d + half * kRotaryPairs);
            require(std::fabs(actual - rotatedHalves[half]) <=
                        ulpBf16(float(rotatedHalves[half])) +
                            ulpBf16(float(std::max(std::fabs(first), std::fabs(second)))) + 0.51 * scale,
                    "the prepare's rotation differs from NeoX's half split in fp64");
          }
        }
        if (!isQuery)
          for (uint32_t d = 0; d < kHeadDimension; ++d) {
            const double source_value = bf16ToFloat(source[kKvHeads * kHeadDimension + d]);
            const auto [actual, scale] = stored(lane, row, true, h, d);
            require(scale ? std::fabs(actual / scale - std::nearbyint(source_value / scale)) <= 1
                          : actual == source_value,
                    "the prepare's value store differs");
          }
      }
    }
  std::cout << "attention prepare: " << kv::formatName(format) << (rotated ? " RoPE 64" : " NoPE")
            << (verify ? " lanes" : " chunk") << (float32 ? " f32" : " bf16") << " norms PASS\n";
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

// The ungated reduce writes the attention's rows as the out-projection's
// input rows as they are, reading no packed row: every row of a chunk or of
// lanes, or each lane's last row alone as row `lane`, from the plan of its
// last tiles, bit for bit its row among every row's; and its table variant
// writes the same rows and the table the projection's own preparation
// writes of them.
void checkOutputRows(metal::MetalBackend &backend) {
  constexpr uint32_t kWidth = kQueryHeads * kHeadDimension;
  for (const bool lanesOfEight : {false, true}) {
    const uint32_t lanes = lanesOfEight ? 3 : 1, rows = lanesOfEight ? kLaneRows : 37;
    const Case data = makeCase(backend, kv::Format::Int8, false, kWindowed, lanes, rows,
                               lanesOfEight ? std::array<uint32_t, 4>{0, 505, 4097, 0}
                                            : std::array<uint32_t, 4>{600, 0, 0, 0});
    const Staging staging = stage(backend, data, 0, rows, lanesOfEight ? kPageTokens : 64);
    const ops::AttentionPlan plan = planOf(data, staging);
    run(backend, data, staging, plan);
    // The attention of `of` into `output`: its reduce's kernel and the input
    // it prepared.
    const auto attend = [&](const ops::AttentionPlan &of, const ops::AttentionOutputRows &output) {
      const metal::MetalBuffer partials = poisoned(backend, of.workspace().partialsBytes);
      const metal::MetalBuffer statistics = poisoned(backend, of.workspace().statisticsBytes);
      const auto [graph, prepared] = encode(data, staging, of, partials, statistics, {}, &output);
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      return std::pair{graph.dispatches().back().pipelineName, prepared.layout};
    };
    const std::string reduce(plan.reducePipeline());
    const metal::MetalBuffer last = poisoned(backend, uint64_t{lanes} * kWidth * 2);
    require(attend(ops::PagedAttention::lastTiles(plan), {{}, last, ops::AttentionOutput::Last}) ==
                std::pair{reduce, ops::LinearInput::Plain},
            "the last rows' reduce runs another kernel or wrote a table");
    for (uint32_t lane = 0; lane < lanes; ++lane)
      require(!std::memcmp(contents<uint16_t>(last) + uint64_t{lane} * kWidth,
                           contents<uint16_t>(staging.output) + staging.outputIndex(lane, 0, rows - 1, 0), kWidth * 2),
              "a lane's last row differs from its row among every row's");
    const uint32_t outputRows = lanes * rows, tiles = (outputRows + 7) / 8;
    const auto tableScratch = [&] {
      return ops::LinearScratch{zeroed(backend, ops::tableBytes(kWidth, uint64_t{tiles} * 8)),
                                zeroed(backend, ops::tableSumsBytes(ops::LinearInput::Table16, kWidth, outputRows)),
                                {},
                                {}};
    };
    const ops::LinearScratch scratch = tableScratch(), expected = tableScratch();
    const metal::MetalBuffer tabled = poisoned(backend, uint64_t{outputRows} * kWidth * 2);
    require(attend(plan, {{}, tabled, ops::AttentionOutput::Every, scratch, ops::LinearInput::Table16}) ==
                std::pair{reduce + "_table16", ops::LinearInput::Table16},
            "the table reduce runs another kernel or reported no table");
    metal::CommandGraph reference;
    test::addReferencePreparation(reference, ops::LinearInput::Table16, staging.output, expected.input,
                                  expected.sums, kWidth, tiles);
    static_cast<void>(backend.submitCommandAsync(reference.dispatches()).wait());
    require(!std::memcmp(tabled.contents(), staging.output.contents(), tabled.sizeBytes()),
            "the table reduce's rows differ from the plain reduce's");
    require(!std::memcmp(scratch.input.contents(), expected.input.contents(), scratch.input.sizeBytes()),
            "the table reduce's table differs from its rows' preparation");
    // The sums as values: the preparation's seed of a padding row (past the
    // output rows, in their last table tile) is -0, where no producer writes.
    const auto *sums = contents<float>(scratch.sums), *preparedSums = contents<float>(expected.sums);
    require(std::equal(sums, sums + scratch.sums.sizeBytes() / sizeof(float), preparedSums),
            "the table reduce's sums differ from its rows' preparation");
  }
  std::cout << "ungated output rows: every row, last rows and Table16 of a chunk and of lanes PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: attention-variants METALLIB");
    metal::MetalBackend backend(argv[1]);
    checkPlans(backend);
    checkRopeTables(backend);
    for (const bool rotated : {true, false})
      for (const bool float32 : test::familyNorms(kGeometry))
        for (const bool verify : {false, true})
          for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16})
            checkProjection(backend, format, rotated, float32, verify);
    checkOutputRows(backend);
    // INT8 pages score against bf16 queries and against INT8 ones in every
    // split (the q8q entries); BF16 pages against bf16 queries.
    for (const auto &[format, int8Queries] : {std::pair{kv::Format::Int8, false}, std::pair{kv::Format::Int8, true},
                                              std::pair{kv::Format::BFloat16, false}}) {
      // Prefill tiles whose rows straddle their window's start: from the
      // history's start, after a cached prefix on either side of a window
      // and of a page, and long chunks after a long prefix, whose first tiles
      // read none of the pages below their windows; a chunk again as several.
      checkPrefill(backend, format, int8Queries, kWindowed, 0, 2048);
      for (const uint32_t history : {31U, 32U, 511U, 512U, 513U, 544U})
        for (const uint32_t rows : {1U, 8U, 33U})
          checkPrefill(backend, format, int8Queries, kWindowed, history, rows);
      checkPrefill(backend, format, int8Queries, kWindowed, 4103, 2048, {509});
      checkPrefill(backend, format, int8Queries, kWindowed, 700, 1057, {31, 512});
      checkPrefill(backend, format, int8Queries, kWindowed, 505, 13, {1, 3});
      checkPrefill(backend, format, int8Queries, kWhole, 0, 300);
      checkPrefill(backend, format, int8Queries, kWhole, 300, 517, {129});
      // Verify lanes of one to four, each with a history of its own: before,
      // at and past their windows, up to 10000 tokens.
      for (const std::vector<uint32_t> &histories : std::vector<std::vector<uint32_t>>{
               {0}, {511, 512}, {513, 544, 1023}, {0, 505, 4097, 10000}, {31, 32, 4096, 7}})
        checkVerify(backend, format, int8Queries, kWindowed, histories);
      checkVerify(backend, format, int8Queries, kWhole, std::vector<uint32_t>{0, 31, 1000, 2047});
      // Tree blocks of two tiles: before, at and past their windows.
      for (const std::vector<uint32_t> &histories :
           std::vector<std::vector<uint32_t>>{{0, 505, 4097, 10000}, {511, 512, 513, 544}})
        checkTreeVerify(backend, format, int8Queries, kWindowed, histories);
      checkTreeVerify(backend, format, int8Queries, kWhole, std::vector<uint32_t>{0, 31, 1000, 2047});
      // Tree blocks of ragged lanes: lanes 0 and 1 past and at their windows,
      // lanes 2 and 3 over short histories.
      checkRaggedTreeVerify(backend, format, int8Queries, kWindowed, {4097, 512, 7, 0});
      checkRaggedTreeVerify(backend, format, int8Queries, kWhole, {1000, 31, 40, 2});
      checkSplitCounts(backend, format, int8Queries);
      checkWindowWithinHistory(backend, format, int8Queries);
    }
    for (const bool int8Queries : {false, true}) checkNonFiniteQueries(backend, int8Queries);
    std::cout << "attention variants: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "attention variants: FAIL: " << error.what() << '\n';
    return 1;
  }
}
