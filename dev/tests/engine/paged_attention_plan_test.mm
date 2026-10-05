#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/RoPE.hpp"
#include "tuning/HostKvExtents.hpp"
#include "tuning/LinearNumerics.hpp"

#include "NormReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

using namespace splash;
using ops::tuning::HostKvExtents;
using ops::tuning::bf16ToFloat;
using ops::tuning::floatToBf16;
using ops::tuning::ulpBf16;

static_assert(!std::is_aggregate_v<ops::PrefillAttentionPlan> &&
              !std::is_default_constructible_v<ops::PrefillAttentionPlan> &&
              !std::is_copy_assignable_v<ops::PrefillAttentionPlan>);
static_assert(!std::is_aggregate_v<ops::VerifyAttentionPlan> &&
              !std::is_default_constructible_v<ops::VerifyAttentionPlan> &&
              !std::is_copy_assignable_v<ops::VerifyAttentionPlan>);

using splash::test::rejects;
using splash::test::require;

void checkPrefillSlotOrientation(uint32_t queryHeads, kv::Layout layout) {
  bool unequalAxes = false, partialTile = false, multipleSplits = false;
  // Enumerate nonsquare grids and partial final tiles. Each logical partial
  // belongs to exactly one query tile, KV head and balanced history split.
  for (const uint32_t rows : {17U, 257U, 2048U}) {
    const auto plan = ops::PagedAttention::prefillPlan(rows, queryHeads, layout);
    const uint32_t tiles = (rows + 7) / 8;
    require(plan.splitGroups.x == layout.kvHeads &&
                plan.splitGroups.y == tiles && plan.splitGroups.z == plan.splits,
            "prefill split axes must be KV head, query tile, split");
    unequalAxes |= tiles != plan.splits;
    partialTile |= rows % 8 != 0;
    multipleSplits |= plan.splits > 1;
    const uint64_t slots = uint64_t{tiles} * layout.kvHeads * plan.splits;
    std::vector<uint8_t> visits(slots, 0);
    for (uint32_t z = 0; z < plan.splitGroups.z; ++z)
      for (uint32_t y = 0; y < plan.splitGroups.y; ++y)
        for (uint32_t x = 0; x < plan.splitGroups.x; ++x) {
          // group.y selects the query tile and group.z its balanced split;
          // scratch is [tile][KV head][split].
          const uint64_t slot = (uint64_t{y} * layout.kvHeads + x) * plan.splits + z;
          require(slot < visits.size() && visits[slot] == 0,
                  "prefill grid aliases or exceeds a logical partial slot");
          ++visits[slot];
          require(slot % plan.splits == z &&
                      (slot / plan.splits) % layout.kvHeads == x &&
                      slot / (uint64_t{plan.splits} * layout.kvHeads) == y,
                  "prefill dispatch orientation changed logical scratch ownership");
        }
    require(std::all_of(visits.begin(), visits.end(),
                        [](uint8_t count) { return count == 1; }),
            "prefill grid omitted a logical partial slot");
  }
  require(unequalAxes && partialTile && multipleSplits,
          "prefill slot orientation cases omitted unequal axes, partial tiles or splits");
}

void checkPlans(uint32_t queryHeads, kv::Layout layout) {
  const std::string geometrySuffix = layout.kvHeads == 4 ? "" : "_kv2_g8";
  const std::array<uint32_t, 1> zeroHistory{};
  const std::string prefillSplit = std::string(layout.format == kv::Format::Int8
      ? "prefill_attention_q8_split" : "prefill_attention_bf16_split") + geometrySuffix;
  const std::string prefillReduce = "prefill_attention_reduce" + geometrySuffix;
  checkPrefillSlotOrientation(queryHeads, layout);
  for (uint32_t rows = 1; rows <= 2048; ++rows) {
    const auto plan = ops::PagedAttention::prefillPlan(rows, queryHeads, layout);
    const uint32_t tiles = (rows + 7) / 8;
    const uint32_t splits = std::clamp(32U / tiles, 1U, 32U);
    require(plan.rows == rows && plan.splits == splits, "prefill plan lost actual rows");
    require(plan.splitPipeline == prefillSplit && plan.reducePipeline == prefillReduce,
            "prefill plan runs the wrong pipelines");
    require(plan.splitGroups.x == layout.kvHeads &&
                plan.splitGroups.y == tiles && plan.splitGroups.z == splits &&
                plan.reduceGroups.x == layout.kvHeads &&
                plan.reduceGroups.y == 8 * queryHeads / layout.kvHeads &&
                plan.reduceGroups.z == tiles,
            "prefill split/reduce geometry disagrees");
    const uint64_t fused = uint64_t{tiles} * splits * 8 * queryHeads;
    require(plan.workspace.partialsBytes == fused * 256 * 4 &&
                plan.workspace.statisticsBytes == fused * 2 * 4,
            "prefill split dispatch and exact scratch disagree");
    const auto bound = ops::PagedAttention::prefillWorkspace(rows, queryHeads, layout);
    require(bound.partialsBytes >= plan.workspace.partialsBytes &&
                bound.statisticsBytes >= plan.workspace.statisticsBytes,
            "prefill arena omitted a valid shorter plan");
  }
  const std::string verifySplit = std::string(layout.format == kv::Format::Int8
      ? "verify_attention_q8_split" : "verify_attention_bf16_split") + geometrySuffix;
  const std::string verifyReduce = "verify_attention_reduce" + geometrySuffix;
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const std::array<uint32_t, 4> histories{0, 31, 16384, 131072};
    const auto plan = ops::PagedAttention::verifyPlan(lanes, queryHeads, layout,
                                                      std::span(histories).first(lanes));
    require(plan.splitPipeline == verifySplit && plan.reducePipeline == verifyReduce,
            "verify plan runs the wrong pipelines");
    uint32_t maximum = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t expected = kv::verifyAttentionSplits(histories[lane]);
      require(plan.laneSplits[lane] == expected && expected >= kv::kVerifySplits &&
                  expected <= kv::kVerifyMaximumSplits,
              "verify lane split count does not follow its own history");
      maximum = std::max(maximum, expected);
    }
    const uint64_t fused =
        uint64_t{lanes} * 8 * kv::kVerifyMaximumSplits * queryHeads;
    require(plan.splits == maximum &&
                plan.workspace.partialsBytes == fused * 256 * 4 &&
                plan.workspace.statisticsBytes == fused * 2 * 4 &&
                plan.splitGroups.y == plan.splits &&
                plan.splitGroups.z == lanes &&
                plan.reduceGroups.y == 8 * queryHeads / layout.kvHeads &&
                plan.reduceGroups.z == lanes,
            "verify split/reduce/scratch disagree");
    require(plan.laneSplits[0] == 32 &&
                (lanes < 4 || plan.laneSplits[3] == kv::kVerifyMaximumSplits),
            "verify partition changed");
  }
  for (const uint32_t rows : {0U, 2049U})
    rejects([&] { (void)ops::PagedAttention::prefillPlan(rows, queryHeads, layout); },
            "invalid attention workspace rows", "a prefill plan of no rows or past the budget was accepted");
  for (const uint32_t lanes : {0U, 5U})
    rejects([&] { (void)ops::PagedAttention::verifyPlan(lanes, queryHeads, layout, zeroHistory); },
            "invalid attention workspace batch width",
            "a verify plan of no lanes or more than a batch was accepted");
  rejects(
      [&] {
        const std::array<uint32_t, 2> two{};
        (void)ops::PagedAttention::verifyPlan(3, queryHeads, layout, two);
      },
      "invalid verify attention history vector", "a verify plan took fewer histories than lanes");
  rejects(
      [&] {
        const std::array<uint32_t, 4> padded{};
        (void)ops::PagedAttention::verifyPlan(3, queryHeads, layout, padded);
      },
      "invalid verify attention history vector", "a verify plan took more histories than lanes");
  rejects(
      [&] {
        const std::array<uint32_t, 1> beyond{kv::kMaximumPhysicalTokens};
        (void)ops::PagedAttention::verifyPlan(1, queryHeads, layout, beyond);
      },
      "verify attention history exceeds physical context",
      "verify rows past the physical context were accepted");
  rejects([&] { (void)ops::PagedAttention::verifyPlan(1, queryHeads + 1, layout, zeroHistory); },
          "no paged-attention kernel for layout", "a query head count without a kernel was accepted");
}

// The attention layer under test is the second of a pool's two, so its region
// starts past the first one's in every extent.
constexpr uint32_t kLayer = 1;

struct Case final {
  uint32_t queryHeads;
  kv::Layout layout;
  uint32_t lanes;
  uint32_t rows;
  uint32_t stride;
  HostKvExtents pool;
  SplashKvLayer layer{};
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer queries;
  std::array<metal::MetalBuffer, 4> tables;
  // Each lane's page ids, which its table holds as entries.
  std::array<std::vector<uint32_t>, 4> pages;
  std::array<kv::ChunkedPrefillParams, 4> stores{};

  uint64_t queryIndex(uint32_t lane, uint32_t head, uint32_t row,
                      uint32_t dimension) const {
    const uint32_t group = queryHeads / layout.kvHeads;
    return (((uint64_t{lane} * layout.kvHeads + head / group) * stride + row) *
                group +
            head % group) *
               256 +
           dimension;
  }

  uint32_t page(uint32_t lane, uint32_t token) const { return pages[lane][token / 32]; }

  float key(uint32_t lane, uint32_t head, uint32_t token,
             uint32_t dimension) const {
    const uint32_t id = page(lane, token);
    const uint64_t index = splash_kv_key_element(head, token % 32, dimension);
    const uint64_t scale = splash_kv_scale_element(head, token % 32);
    if (layout.format == kv::Format::BFloat16)
      return bf16ToFloat(pool.slab<uint16_t>(kLayer, SPLASH_KV_KEYS, id)[index]);
    return pool.slab<int8_t>(kLayer, SPLASH_KV_KEYS, id)[index] *
           pool.slab<float>(kLayer, SPLASH_KV_KEY_SCALES, id)[scale];
  }

  float value(uint32_t lane, uint32_t head, uint32_t token,
               uint32_t dimension) const {
    const uint32_t id = page(lane, token);
    const uint64_t index = splash_kv_value_element(head, token % 32, dimension);
    const uint64_t scale = splash_kv_scale_element(head, token % 32);
    if (layout.format == kv::Format::BFloat16)
      return bf16ToFloat(pool.slab<uint16_t>(kLayer, SPLASH_KV_VALUES, id)[index]);
    return pool.slab<int8_t>(kLayer, SPLASH_KV_VALUES, id)[index] *
           pool.slab<float>(kLayer, SPLASH_KV_VALUE_SCALES, id)[scale];
  }
};

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  if (!bytes) return {};
  auto buffer = test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// The lanes' pages are mixed over three or more extents of the pool, boundary
// pages first; oneExtent puts the same pages in one extent, which must give
// the same bits.
Case makeCase(metal::MetalBackend &backend, uint32_t queryHeads,
               kv::Layout layout, uint32_t lanes, uint32_t rows,
               uint32_t history, bool verify, bool oneExtent = false) {
  std::array<uint32_t, 4> historyLengths{};
  std::array<uint32_t, 4> pageCounts{};
  uint32_t allPages = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    // Verify lanes differ by more than a split's worth of pages, so long
    // histories give each lane its own split count under one slot stride.
    historyLengths[lane] = history + (verify ? lane * 649 : 0);
    pageCounts[lane] = (historyLengths[lane] + rows + 31) / 32;
    allPages += pageCounts[lane];
  }
  const auto spread = HostKvExtents::spread(allPages + 2);
  const std::vector<uint32_t> ids =
      HostKvExtents::mixedPages(spread, allPages, history + rows + lanes);
  kv::Layout poolLayout = layout;
  poolLayout.attentionLayers = kLayer + 1;
  Case data{queryHeads, layout, lanes, rows, (rows + 31) / 32 * 32,
            oneExtent ? HostKvExtents(backend, poolLayout,
                                            spread.extentPages * spread.extents, 1)
                      : HostKvExtents(backend, poolLayout, spread.extentPages,
                                            spread.extents),
            {}, {}, {}, {}, {}, {}, {}};
  data.layer = data.pool.layer(kLayer);
  data.keys = allocate(backend, uint64_t{lanes} * layout.kvHeads * data.stride * 256 * 2);
  data.values = allocate(backend, data.keys.sizeBytes());
  data.queries = allocate(backend, uint64_t{lanes} * queryHeads * data.stride * 256 * 2);
  uint32_t firstPage = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    data.pages[lane].assign(ids.begin() + firstPage,
                            ids.begin() + firstPage + pageCounts[lane]);
    firstPage += pageCounts[lane];
    data.tables[lane] = allocate(backend, uint64_t{pageCounts[lane]} * sizeof(SplashKvPage));
    data.pool.writeTable(data.pages[lane], data.tables[lane].contents());
    data.stores[lane] =
        verify ? ops::PagedAttention::verifyParams(historyLengths[lane], pageCounts[lane])
               : ops::PagedAttention::prefillParams(historyLengths[lane], rows, data.stride,
                                                    pageCounts[lane]);
    for (uint32_t token = 0; token < historyLengths[lane]; ++token) {
      const uint32_t id = data.page(lane, token);
      for (uint32_t head = 0; head < layout.kvHeads; ++head) {
        const uint64_t slot = splash_kv_scale_element(head, token % 32);
        if (layout.format == kv::Format::Int8) {
          data.pool.slab<float>(kLayer, SPLASH_KV_KEY_SCALES, id)[slot] = 0.006f;
          data.pool.slab<float>(kLayer, SPLASH_KV_VALUE_SCALES, id)[slot] = 0.007f;
        }
        for (uint32_t dimension = 0; dimension < 256; ++dimension) {
          const int key = int((token * 37 + head * 101 + dimension * 17 +
                               token * dimension * 3 + lane * 7) % 255) - 127;
          const int value = int((token * 53 + head * 79 + dimension * 29 +
                                 token * dimension * 5 + lane * 19) % 255) - 127;
          const uint64_t keyIndex = splash_kv_key_element(head, token % 32, dimension);
          const uint64_t valueIndex = splash_kv_value_element(head, token % 32, dimension);
          if (layout.format == kv::Format::Int8) {
            data.pool.slab<int8_t>(kLayer, SPLASH_KV_KEYS, id)[keyIndex] = key;
            data.pool.slab<int8_t>(kLayer, SPLASH_KV_VALUES, id)[valueIndex] = value;
          } else {
            data.pool.slab<uint16_t>(kLayer, SPLASH_KV_KEYS, id)[keyIndex] =
                floatToBf16(key * 0.006f);
            data.pool.slab<uint16_t>(kLayer, SPLASH_KV_VALUES, id)[valueIndex] =
                floatToBf16(value * 0.007f);
          }
        }
      }
    }
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t head = 0; head < layout.kvHeads; ++head) {
        const uint64_t base = (uint64_t{lane} * layout.kvHeads + head) * data.stride * 256;
        for (uint32_t dimension = 0; dimension < 256; ++dimension) {
          static_cast<uint16_t *>(data.keys.contents())[base + row * 256 + dimension] =
              floatToBf16(float(int((row * 37 + head * 101 + dimension * 17) % 255) - 127) * 0.006f);
          static_cast<uint16_t *>(data.values.contents())[base + dimension * data.stride + row] =
              floatToBf16(float(int((row * 53 + head * 79 + dimension * 29) % 255) - 127) * 0.007f);
        }
      }
      for (uint32_t head = 0; head < queryHeads; ++head)
        for (uint32_t dimension = 0; dimension < 256; ++dimension)
          static_cast<uint16_t *>(data.queries.contents())[
              data.queryIndex(lane, head, row, dimension)] =
              floatToBf16(float(int((row * 43 + head * 67 + dimension * 11 +
                              head * dimension * 7) % 1019) - 509) / 1018.0f);
    }
  }
  for (uint32_t lane = lanes; lane < 4; ++lane) {
    data.tables[lane] = data.tables[0];
    data.pages[lane] = data.pages[0];
  }
  return data;
}

// Independent scalar softmax over the exact KV pages produced by the store.
// Sampling query rows bounds the full-2048 oracle cost. Each prefill candidate
// and host chunk is checked independently against its own causal KV history.
void checkReference(const Case &data, const std::vector<uint16_t> &actual) {
  double dot = 0, actualSquared = 0, expectedSquared = 0;
  float maximumError = 0;
  std::vector<uint32_t> selectedRows{0, std::min(7U, data.rows - 1),
                                     data.rows / 2, data.rows - 1};
  std::sort(selectedRows.begin(), selectedRows.end());
  selectedRows.erase(std::unique(selectedRows.begin(), selectedRows.end()), selectedRows.end());
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    for (uint32_t row : selectedRows) {
      for (uint32_t head : {0U, data.queryHeads - 1}) {
        const uint32_t kvHead = head / (data.queryHeads / data.layout.kvHeads);
        const uint32_t tokens = data.stores[lane].committed_tokens + row + 1;
        std::vector<float> scores(tokens);
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t token = 0; token < tokens; ++token) {
          float score = 0;
          for (uint32_t dimension = 0; dimension < 256; ++dimension)
            score += bf16ToFloat(static_cast<const uint16_t *>(data.queries.contents())[
                              data.queryIndex(lane, head, row, dimension)]) *
                     data.key(lane, kvHead, token, dimension);
          scores[token] = score * 0.0625f;
          maximum = std::max(maximum, scores[token]);
        }
        float denominator = 0;
        for (float &score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (uint32_t dimension = 0; dimension < 256; ++dimension) {
          float expected = 0;
          for (uint32_t token = 0; token < tokens; ++token)
            expected += scores[token] * data.value(lane, kvHead, token, dimension);
          expected /= denominator;
          const float value = bf16ToFloat(actual[data.queryIndex(lane, head, row, dimension)]);
          require(std::isfinite(value), "attention output is nonfinite");
          maximumError = std::max(maximumError, std::abs(value - expected));
          dot += value * expected;
          actualSquared += value * value;
          expectedSquared += expected * expected;
        }
      }
    }
  }
  const double cosine = dot / std::sqrt(actualSquared * expectedSquared);
  if (!(maximumError < 0.02f && cosine > 0.9995))
    throw std::runtime_error(
        "attention candidate failed scalar KV oracle: history=" +
        std::to_string(data.stores[0].committed_tokens) + " rows=" +
        std::to_string(data.rows) + " maximum_absolute_error=" +
        std::to_string(maximumError) + " cosine=" + std::to_string(cosine));
}

void checkEquivalent(const Case &data, const std::vector<uint16_t> &baseline,
                      const std::vector<uint16_t> &candidate) {
  float maximumError = 0;
  double dot = 0, baselineSquared = 0, candidateSquared = 0;
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    for (uint32_t head = 0; head < data.queryHeads; ++head)
      for (uint32_t row = 0; row < data.rows; ++row)
        for (uint32_t dimension = 0; dimension < 256; ++dimension) {
          const uint64_t index = data.queryIndex(lane, head, row, dimension);
          const float left = bf16ToFloat(baseline[index]);
          const float right = bf16ToFloat(candidate[index]);
          require(std::isfinite(right), "attention candidate is nonfinite");
          maximumError = std::max(maximumError, std::abs(left - right));
          dot += left * right;
          baselineSquared += left * left;
          candidateSquared += right * right;
        }
  const double cosine = dot / std::sqrt(baselineSquared * candidateSquared);
  if (!(maximumError < 0.02f && cosine > 0.9995))
    throw std::runtime_error(
        "attention candidate differs from baseline: history=" +
        std::to_string(data.stores[0].committed_tokens) + " rows=" +
        std::to_string(data.rows) + " maximum_absolute_error=" +
        std::to_string(maximumError) + " cosine=" + std::to_string(cosine));
}

// Construct expected extents from the source bits, independently of GPU
// stores: every byte of every extent but the stored rows' slots stays.
std::vector<std::vector<std::byte>> expectedBf16Store(const Case &data) {
  std::vector<std::vector<std::byte>> result;
  for (uint32_t extent = 0; extent < data.pool.extentCount(); ++extent) {
    const auto bytes = data.pool.bytes(extent);
    result.emplace_back(bytes.begin(), bytes.end());
  }
  for (unsigned tensor = 0; tensor < 2; ++tensor) {
    const auto *source = static_cast<const uint16_t *>(
        (tensor ? data.values : data.keys).contents());
    for (uint32_t lane = 0; lane < data.lanes; ++lane) {
      for (uint32_t row = 0; row < data.stores[lane].chunk_tokens; ++row) {
        const uint32_t token = data.stores[lane].committed_tokens + row;
        const uint32_t id = data.page(lane, token);
        const auto *page = data.pool.slab<uint16_t>(
            kLayer, tensor ? SPLASH_KV_VALUES : SPLASH_KV_KEYS, id);
        const uint32_t extent = id / data.pool.extentPages();
        for (uint32_t head = 0; head < data.layout.kvHeads; ++head)
          for (uint32_t d = 0; d < 256; ++d) {
            const uint64_t element = tensor ? splash_kv_value_element(head, token % 32, d)
                                            : splash_kv_key_element(head, token % 32, d);
            const auto offset = reinterpret_cast<const std::byte *>(page + element) -
                                data.pool.bytes(extent).data();
            const uint64_t base = (uint64_t{lane} * data.layout.kvHeads + head) * data.stride * 256;
            const uint64_t input = base + (tensor ? d * data.stride + row : row * 256 + d);
            std::memcpy(result[extent].data() + offset, source + input, sizeof(uint16_t));
          }
      }
    }
  }
  return result;
}

void checkBf16Store(const Case &data, const std::vector<std::vector<std::byte>> &expected) {
  for (uint32_t extent = 0; extent < data.pool.extentCount(); ++extent) {
    const auto bytes = data.pool.bytes(extent);
    require(std::equal(bytes.begin(), bytes.end(), expected[extent].begin()),
            "BF16 store changed source bits, history, another layer or an unused page slot");
  }
}

void checkBf16StoreEdges(metal::MetalBackend &backend) {
  for (uint32_t heads : {16U, 24U}) {
    auto data = makeCase(backend, heads, {1, heads == 24 ? 4U : 2U, 256,
                                         kv::Format::BFloat16}, 1, 33, 31, false);
    // Signed zero, subnormals, large finite values, infinities and NaN payloads.
    constexpr std::array<uint16_t, 12> bits{0, 0x8000, 1, 0x8001, 0x007f, 0x0080,
                                          0x4960, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc3};
    for (auto buffer : {data.keys, data.values}) {
      auto *source = static_cast<uint16_t *>(buffer.contents());
      for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i) source[i] = bits[i % bits.size()];
    }
    const auto expected = expectedBf16Store(data);
    metal::CommandGraph graph;
    ops::PagedAttention::addPrefillStore(graph, data.layer, data.keys, data.values,
                                        data.tables[0], data.stores[0], data.layout);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    checkBf16Store(data, expected);
  }
}

enum class Phase : uint8_t { Prefill, Verify };

template <Phase phase>
std::vector<uint16_t> run(metal::MetalBackend &backend, Case &data, bool testBounds) {
  constexpr bool prefill = phase == Phase::Prefill;
  const auto plan = [&] {
    if constexpr (prefill)
      return ops::PagedAttention::prefillPlan(data.rows, data.queryHeads, data.layout);
    else {
      std::vector<uint32_t> histories;
      for (uint32_t lane = 0; lane < data.lanes; ++lane)
        histories.push_back(data.stores[lane].committed_tokens);
      return ops::PagedAttention::verifyPlan(data.lanes, data.queryHeads, data.layout,
                                            histories);
    }
  }();
  constexpr uint64_t guardBytes = 256;
  const std::array sizes{plan.workspace.partialsBytes, plan.workspace.statisticsBytes,
                         data.queries.sizeBytes()};
  std::array<metal::MetalBuffer, 3> backing, views;
  for (size_t i = 0; i < sizes.size(); ++i) {
    backing[i] = allocate(backend, sizes[i] + 2 * guardBytes);
    std::memset(backing[i].contents(), 0xa5, sizes[i] + 2 * guardBytes);
    views[i] = backend.view(backing[i], guardBytes, sizes[i]);
    std::memset(views[i].contents(), 0, sizes[i]);
  }
  const auto partials = views[0], statistics = views[1], output = views[2];
  if constexpr (prefill) {
    if (testBounds) {
      metal::CommandGraph shortGraph;
      auto mismatch = data.stores[0];
      mismatch.chunk_tokens = plan.rows == 1 ? 2 : plan.rows - 1;
      rejects(
          [&] {
            ops::PagedAttention::addPrefill(shortGraph, data.layer, data.queries, output,
                                           partials, statistics, data.tables[0], mismatch, plan);
          },
          "prefill attention rows do not match plan", "a chunk of other rows than its plan was accepted");
      require(shortGraph.empty(), "mismatched prefill plan partially encoded a graph");
    }
  }
  metal::CommandGraph graph;
  if constexpr (prefill) {
    ops::PagedAttention::addPrefillStore(graph, data.layer, data.keys, data.values,
                                        data.tables[0], data.stores[0], data.layout);
    ops::PagedAttention::addPrefill(graph, data.layer, data.queries, output, partials, statistics,
                                    data.tables[0], data.stores[0], plan);
  } else {
    ops::PagedAttention::addVerify(graph, data.layer,
                                   {data.keys, data.values, data.queries, partials, statistics, output, data.tables},
                                   std::span(data.stores).first(plan.lanes), plan);
  }
  if constexpr (prefill) {
    require(graph.dispatches().size() == 3,
            "production prefill should encode store/split/reduce");
    const auto checkDispatch = [&](const auto &dispatch, auto groups, std::string_view pipeline) {
      require(dispatch.pipelineName == pipeline && dispatch.threadgroups.x == groups.x &&
                  dispatch.threadgroups.y == groups.y && dispatch.threadgroups.z == groups.z &&
                  dispatch.bytes.size() == 1 &&
                  dispatch.bytes[0].sizeBytes == sizeof(kv::PrefillAttentionParams),
              "production prefill dispatch departed from its plan");
      kv::PrefillAttentionParams params;
      std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
      require(params.committed_tokens == data.stores[0].committed_tokens &&
                  params.rows == data.rows &&
                  params.chunk_stride == data.stride &&
                  params.page_table_entries == data.stores[0].page_table_entries &&
                  params.kv.extent_pages == data.layer.extent_pages &&
                  params.kv.offset == data.layer.offset &&
                  params.split_count == plan.splits,
              "recorded prefill ABI does not describe the actual split plan");
    };
    checkDispatch(graph.dispatches()[1], plan.splitGroups, plan.splitPipeline);
    checkDispatch(graph.dispatches()[2], plan.reduceGroups, plan.reducePipeline);
  } else {
    require(graph.dispatches().size() == 3, "production verify should encode store/split/reduce");
    const auto &split = graph.dispatches()[1];
    const auto &reduce = graph.dispatches()[2];
    require(split.pipelineName == plan.splitPipeline && reduce.pipelineName == plan.reducePipeline &&
                split.threadgroups.y == plan.splits && split.threadgroups.z == plan.splitGroups.z &&
                reduce.threadgroups.y == plan.reduceGroups.y &&
                reduce.threadgroups.z == plan.reduceGroups.z,
            "production verify encoding departed from its plan");
  }
  const auto expected = data.layout.format == kv::Format::BFloat16
                            ? expectedBf16Store(data)
                            : std::vector<std::vector<std::byte>>{};
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  if (data.layout.format == kv::Format::BFloat16) checkBf16Store(data, expected);
  for (size_t i = 0; i < sizes.size(); ++i) {
    const auto *bytes = static_cast<const uint8_t *>(backing[i].contents());
    for (uint64_t byte = 0; byte < guardBytes; ++byte)
      require(bytes[byte] == 0xa5 && bytes[guardBytes + sizes[i] + byte] == 0xa5,
              "attention scratch/output write canary changed");
  }
  const auto *values = static_cast<const uint16_t *>(output.contents());
  return {values, values + output.sizeBytes() / 2};
}

void checkPrefill(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout,
                   uint32_t history, uint32_t rows) {
  auto data = makeCase(backend, heads, layout, 1, rows, history, false);
  const auto output = run<Phase::Prefill>(backend, data, true);
  checkReference(data, output);
  checkEquivalent(data, output, run<Phase::Prefill>(backend, data, false));
  auto oneExtent = makeCase(backend, heads, layout, 1, rows, history, false, true);
  require(run<Phase::Prefill>(backend, oneExtent, false) == output,
          "prefill attention over extents differs from one extent of the same pages");
  if (rows == 1057) {
    // Reuse the identical packed BF16 inputs and KV history across unaligned
    // host chunks, checking each path against its independent causal oracle.
    const auto copy = [](metal::MetalBuffer buffer) {
      const auto *begin = static_cast<const uint16_t *>(buffer.contents());
      return std::vector<uint16_t>(begin, begin + buffer.sizeBytes() / 2);
    };
    const auto keys = copy(data.keys), values = copy(data.values), queries = copy(data.queries);
    uint32_t offset = 0;
    for (uint32_t chunk : {3U, 5U, 31U, 509U, 509U}) {
      data.rows = chunk;
      data.stores[0].committed_tokens = history + offset;
      data.stores[0].chunk_tokens = chunk;
      for (uint32_t head = 0; head < layout.kvHeads; ++head)
        for (uint32_t row = 0; row < chunk; ++row)
          for (uint32_t d = 0; d < 256; ++d) {
            const uint64_t base = uint64_t{head} * data.stride * 256;
            static_cast<uint16_t *>(data.keys.contents())[base + row * 256 + d] =
                keys[base + (offset + row) * 256 + d];
            static_cast<uint16_t *>(data.values.contents())[base + d * data.stride + row] =
                values[base + d * data.stride + offset + row];
          }
      for (uint32_t head = 0; head < heads; ++head)
        for (uint32_t row = 0; row < chunk; ++row)
          for (uint32_t d = 0; d < 256; ++d)
            static_cast<uint16_t *>(data.queries.contents())[data.queryIndex(0, head, row, d)] =
                queries[data.queryIndex(0, head, offset + row, d)];
      const auto chunkOutput = run<Phase::Prefill>(backend, data, true);
      checkReference(data, chunkOutput);
      checkEquivalent(data, chunkOutput, run<Phase::Prefill>(backend, data, false));
      offset += chunk;
    }
    require(offset == rows, "chunk comparison dropped logical query rows");
  }
  std::cout << "paged prefill: format=" << kv::formatName(layout.format) << " q=" << heads << " history=" << history
            << " rows=" << rows << " PASS\n";
}

void checkVerify(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout,
                  uint32_t history, uint32_t lanes) {
  auto data = makeCase(backend, heads, layout, lanes, 8, history, true);
  const auto output = run<Phase::Verify>(backend, data, true);
  checkReference(data, output);
  auto oneExtent = makeCase(backend, heads, layout, lanes, 8, history, true, true);
  require(run<Phase::Verify>(backend, oneExtent, false) == output,
          "verify attention over extents differs from one extent of the same pages");
  std::cout << "paged verify: format=" << kv::formatName(layout.format) << " q=" << heads << " history=" << history
            << " lanes=" << lanes << " PASS\n";
}

// The q/k RMS norms, RoPE and V copy of the attention prepare, prefill and
// verify, against fp64 with norm weights in bf16 or F32 (a GGUF's). Past the
// rotary pairs a row holds the norm rounded once to bf16; each rotated value
// is within an ulp of the fp64 rotation of the bf16-rounded norms plus an ulp
// of the larger input, which covers the fp32 kernel rounding a norm to the
// other bf16 neighbour and a rotation that cancels.
void checkProjection(metal::MetalBackend &backend, uint32_t queryHeads, kv::Layout layout,
                     bool float32, bool verify) {
  constexpr uint32_t kDim = 256, kPairs = 32;
  const uint32_t kvHeads = layout.kvHeads, group = queryHeads / kvHeads;
  const uint32_t lanes = verify ? 3 : 1, rows = verify ? 8 : 37,
                 stride = verify ? kv::kVerifyChunkStride : 64;
  const uint32_t packedWidth = 2 * queryHeads * kDim + 2 * kvHeads * kDim;
  auto packed = allocate(backend, uint64_t{lanes} * rows * packedWidth * 2);
  auto ropeCos = allocate(backend, uint64_t{lanes} * rows * kPairs * 4);
  auto ropeSin = allocate(backend, ropeCos.sizeBytes());
  auto queries = allocate(backend, uint64_t{lanes} * queryHeads * stride * kDim * 2);
  auto keys = allocate(backend, uint64_t{lanes} * kvHeads * stride * kDim * 2);
  auto values = allocate(backend, keys.sizeBytes());
  uint32_t state = 0x2545F491U + queryHeads + (verify ? 7 : 0);
  const auto unit = [&] {
    state = state * 1664525U + 1013904223U;
    return double(state >> 8) / double(1U << 23) - 1.0;
  };
  auto *packedData = static_cast<uint16_t *>(packed.contents());
  for (uint64_t i = 0; i < packed.sizeBytes() / 2; ++i)
    packedData[i] = floatToBf16(float(2 * unit()));
  for (uint64_t i = 0; i < ropeCos.sizeBytes() / 4; ++i) {
    const double angle = 3.14159265358979 * unit();
    static_cast<float *>(ropeCos.contents())[i] = float(std::cos(angle));
    static_cast<float *>(ropeSin.contents())[i] = float(std::sin(angle));
  }
  const auto weight = [&](uint32_t) { return float(1.0 + 0.3 * unit()); };
  const ops::NormWeights queryNorm = test::makeNormWeights(backend, kDim, float32, weight);
  const ops::NormWeights keyNorm = test::makeNormWeights(backend, kDim, float32, weight);
  const auto addProjection = [&](metal::CommandGraph &graph, const ops::NormWeights &keyWeights) {
    if (verify)
      ops::PagedAttention::addVerifyProjection(graph, packed, queryNorm, keyWeights, ropeCos, ropeSin,
                                               queries, keys, values, queryHeads, layout, lanes);
    else
      ops::PagedAttention::addPrefillProjection(graph, packed, queryNorm, keyWeights, ropeCos, ropeSin,
                                                queries, keys, values, rows, stride, queryHeads,
                                                layout);
  };
  metal::CommandGraph graph;
  addProjection(graph, keyNorm);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();

  const auto *queryData = static_cast<const uint16_t *>(queries.contents());
  const auto *keyData = static_cast<const uint16_t *>(keys.contents());
  const auto *valueData = static_cast<const uint16_t *>(values.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < rows; ++row) {
      const uint64_t packedRow = (uint64_t{lane} * rows + row) * packedWidth;
      const uint64_t rope = (uint64_t{lane} * rows + row) * kPairs;
      for (uint32_t head = 0; head < queryHeads + kvHeads; ++head) {
        const bool query = head < queryHeads;
        const uint32_t h = query ? head : head - queryHeads;
        const uint16_t *source = packedData + packedRow +
                                 (query ? h * 2 * kDim : 2 * queryHeads * kDim + h * kDim);
        const uint16_t *out =
            query ? queryData + (((uint64_t{lane} * kvHeads + h / group) * stride + row) * group +
                                 h % group) * kDim
                  : keyData + ((uint64_t{lane} * kvHeads + h) * stride + row) * kDim;
        const std::vector<double> normalized =
            test::rmsNorm(source, query ? queryNorm : keyNorm, kDim);
        for (uint32_t d = 2 * kPairs; d < kDim; ++d)
          require(test::roundedOnceToBf16(out[d], normalized[d]),
                  "attention prepare norm differs from the fp64 reference");
        for (uint32_t d = 0; d < kPairs; ++d) {
          const double first = bf16ToFloat(floatToBf16(float(normalized[d])));
          const double second = bf16ToFloat(floatToBf16(float(normalized[d + kPairs])));
          const double c = static_cast<const float *>(ropeCos.contents())[rope + d];
          const double s = static_cast<const float *>(ropeSin.contents())[rope + d];
          const double rotated[2] = {first * c - second * s, second * c + first * s};
          for (uint32_t half = 0; half < 2; ++half)
            require(std::fabs(bf16ToFloat(out[d + half * kPairs]) - rotated[half]) <=
                        ulpBf16(float(rotated[half])) +
                            ulpBf16(float(std::max(std::fabs(first), std::fabs(second)))),
                    "attention prepare rotation differs from the fp64 reference");
        }
        if (!query)
          for (uint32_t d = 0; d < kDim; ++d)
            require(valueData[((uint64_t{lane} * kvHeads + h) * kDim + d) * stride + row] ==
                        source[kvHeads * kDim + d],
                    "attention prepare value copy differs");
      }
    }
  // One dispatch normalizes the queries and the keys, so it takes one norm type.
  if (float32) {
    metal::CommandGraph rejected;
    try {
      addProjection(rejected, {keyNorm.buffer, false});
      throw std::runtime_error("mixed q/k norm types were accepted");
    } catch (const std::invalid_argument &error) {
      require(std::string_view(error.what()) == "query and key norms differ in type",
              "mixed q/k norm types rejected for the wrong reason");
    }
  }
  std::cout << "attention prepare: q=" << queryHeads << (verify ? " verify" : " prefill")
            << (float32 ? " f32" : " bf16") << " norms PASS\n";
}

// Each buffer the projections, gates, store and attention reach, at its
// extent and one element short: the packed rows (the gates read each row's
// query heads only), the RoPE rows, the staging of each KV head's rows
// `stride` rows apart (values transposed), the page table of a chunk's
// visible tokens, and the plans' partials and statistics. Prefill stages a
// chunk of 37 tokens after 100 in 64 rows per KV head, its attention whole
// tiles of 8 rows; verify stages three lanes' 8 rows in 32 per KV head.
void bufferExtents(metal::MetalBackend &backend, uint32_t queryHeads, kv::Layout layout) {
  constexpr uint32_t dimension = 256, tokens = 37, stride = 64, committed = 100, lanes = 3;
  const uint32_t kvHeads = layout.kvHeads, group = queryHeads / kvHeads;
  const uint64_t packedWidth = uint64_t{2 * queryHeads + 2 * kvHeads} * dimension;
  const uint64_t queryWidth = uint64_t{group} * dimension, verifyRows = uint64_t{lanes} * kv::kVerifyRows;
  const uint64_t verifyHeads = uint64_t{lanes} * kvHeads;
  // `rows` rows of each of `heads` heads, `rowStride` rows apart; values
  // hold a row per dimension.
  const auto staged = [](uint64_t heads, uint64_t rowStride, uint64_t rows, uint64_t width) {
    return ((heads - 1) * rowStride + rows) * width * 2;
  };
  const auto stagedValues = [&](uint64_t heads, uint64_t rowStride, uint64_t rows) {
    return ((heads * dimension - 1) * rowStride + rows) * 2;
  };
  const auto gateRows = [&](uint64_t rows) {
    return ((rows - 1) * packedWidth + uint64_t{queryHeads} * 2 * dimension) * 2;
  };
  const auto pageTable = [](const kv::ChunkedPrefillParams &chunk) {
    return uint64_t{(chunk.committed_tokens + chunk.chunk_tokens + 31) / 32} * sizeof(SplashKvPage);
  };
  const ops::NormWeights queryNorm{allocate(backend, dimension * 2)}, keyNorm{allocate(backend, dimension * 2)};
  using Extents = std::initializer_list<test::BufferExtent>;
  // The q/k norms of both projections, whose other buffers hold every row.
  const uint64_t rows = std::max<uint64_t>(tokens, verifyRows);
  const metal::MetalBuffer packed = allocate(backend, rows * packedWidth * 2), rope = allocate(backend, rows * 32 * 4),
                           staging = allocate(backend, staged(verifyHeads, stride, stride, queryWidth));
  for (const bool verify : {false, true})
    for (const bool key : {false, true}) {
      test::requireExtent(backend, (key ? keyNorm : queryNorm).buffer, dimension * 2, 2, "norm weight",
                          [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                            const ops::NormWeights query{key ? queryNorm.buffer : view},
                                keys{key ? view : keyNorm.buffer};
                            if (verify)
                              ops::PagedAttention::addVerifyProjection(graph, packed, query, keys, rope, rope, staging,
                                                                       staging, staging, queryHeads, layout, lanes);
                            else
                              ops::PagedAttention::addPrefillProjection(graph, packed, query, keys, rope, rope,
                                                                        staging, staging, staging, tokens, stride,
                                                                        queryHeads, layout);
                          });
    }
  using Buffers = std::vector<metal::MetalBuffer>;
  test::requireExtents(backend,
                       Extents{{0, tokens * packedWidth * 2, 2, "attention q/k/v"},
                               {1, uint64_t{tokens} * 32 * 4, 4, "attention RoPE cosine"},
                               {2, uint64_t{tokens} * 32 * 4, 4, "attention RoPE sine"},
                               {3, staged(kvHeads, stride, tokens, queryWidth), 2, "attention query"},
                               {4, staged(kvHeads, stride, tokens, dimension), 2, "attention key"},
                               {5, stagedValues(kvHeads, stride, tokens), 2, "attention value"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         ops::PagedAttention::addPrefillProjection(graph, b[0], queryNorm, keyNorm, b[1], b[2], b[3],
                                                                   b[4], b[5], tokens, stride, queryHeads, layout);
                       });
  const auto verifyStaged = [&](uint64_t width) {
    return staged(verifyHeads, kv::kVerifyChunkStride, kv::kVerifyRows, width);
  };
  test::requireExtents(backend,
                       Extents{{0, verifyRows * packedWidth * 2, 2, "attention q/k/v"},
                               {1, verifyRows * 32 * 4, 4, "attention RoPE cosine"},
                               {2, verifyRows * 32 * 4, 4, "attention RoPE sine"},
                               {3, verifyStaged(queryWidth), 2, "attention query"},
                               {4, verifyStaged(dimension), 2, "attention key"},
                               {5, stagedValues(verifyHeads, kv::kVerifyChunkStride, kv::kVerifyRows), 2,
                                "attention value"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         ops::PagedAttention::addVerifyProjection(graph, b[0], queryNorm, keyNorm, b[1], b[2], b[3],
                                                                  b[4], b[5], queryHeads, layout, lanes);
                       });
  test::requireExtents(backend,
                       Extents{{0, gateRows(tokens), 2, "attention gate"},
                               {1, staged(kvHeads, stride, tokens, queryWidth), 2, "attention row"},
                               {2, uint64_t{tokens} * queryHeads * dimension * 2, 2, "attention hidden"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         ops::PagedAttention::addPrefillGate(graph, b[0], b[1], b[2], tokens, stride, queryHeads,
                                                             layout);
                       });
  const uint32_t width = queryHeads * dimension;
  test::requireExtents(backend,
                       Extents{{0, gateRows(verifyRows), 2, "attention gate"},
                               {1, verifyStaged(queryWidth), 2, "attention row"},
                               {2, verifyRows * width * 2, 2, "attention hidden"},
                               {3, ops::tableBytes(width, verifyRows), 2, "linear table"},
                               {4, ops::tableSumsBytes(ops::LinearInput::Table64, width, verifyRows), 4,
                                "linear table sums"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         (void)ops::PagedAttention::addVerifyGate(graph, b[0], b[1], b[2], queryHeads, layout, lanes,
                                                                  {b[3], b[4], {}, {}}, ops::LinearInput::Table64);
                       });

  const auto chunk = ops::PagedAttention::prefillParams(committed, tokens, stride, 8);
  test::requireExtents(backend,
                       Extents{{0, staged(kvHeads, stride, tokens, dimension), 2, "attention key"},
                               {1, stagedValues(kvHeads, stride, tokens), 2, "attention value"},
                               {2, pageTable(chunk), sizeof(SplashKvPage), "page table"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         ops::PagedAttention::addPrefillStore(graph, {}, b[0], b[1], b[2], chunk, layout);
                       });
  const auto prefill = ops::PagedAttention::prefillPlan(tokens, queryHeads, layout);
  const uint64_t tileRows = kv::prefillAttentionTiles(tokens) * kv::kPrefillAttentionTileRows;
  test::requireExtents(backend,
                       Extents{{0, staged(kvHeads, stride, tileRows, queryWidth), 2, "attention query"},
                               {1, staged(kvHeads, stride, tileRows, queryWidth), 2, "attention output"},
                               {2, prefill.workspace.partialsBytes, 4, "prefill attention partials"},
                               {3, prefill.workspace.statisticsBytes, 4, "prefill attention statistics"},
                               {4, pageTable(chunk), sizeof(SplashKvPage), "page table"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         ops::PagedAttention::addPrefill(graph, {}, b[0], b[1], b[2], b[3], b[4], chunk, prefill);
                       });
  const std::array<uint32_t, lanes> histories{committed, 2000, 0};
  std::array<kv::ChunkedPrefillParams, lanes> chunks{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    chunks[lane] = ops::PagedAttention::verifyParams(histories[lane], 64);
  const auto verify = ops::PagedAttention::verifyPlan(lanes, queryHeads, layout, histories);
  std::vector<test::BufferExtent> extents{
      {0, verifyStaged(dimension), 2, "attention key"},
      {1, stagedValues(verifyHeads, kv::kVerifyChunkStride, kv::kVerifyRows), 2, "attention value"},
      {2, verifyStaged(queryWidth), 2, "attention query"},
      {3, verify.workspace.partialsBytes, 4, "verify attention partials"},
      {4, verify.workspace.statisticsBytes, 4, "verify attention statistics"},
      {5, verifyStaged(queryWidth), 2, "attention output"}};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    extents.push_back({6 + lane, pageTable(chunks[lane]), sizeof(SplashKvPage), "page table"});
  test::requireExtents(backend, extents, [&](metal::CommandGraph &graph, Buffers b) {
    // A table for every lane of the batch; past the plan's lanes, unread.
    b.resize(6 + SPLASH_MAXIMUM_BATCH_WIDTH, b[6]);
    ops::PagedAttention::addVerify(graph, {}, {b[0], b[1], b[2], b[3], b[4], b[5], std::span(b).subspan(6)}, chunks,
                                   verify);
  });
}

// Each buffer the RoPE tables reach, at its extent and one element short: the
// target rows' three positions and the draft rows' one, each kind's inverse
// frequencies, and its rows of cosines and sines.
void ropeExtents(metal::MetalBackend &backend) {
  constexpr RopeTableParams rows{37, 24};
  test::requireExtents(backend,
                       std::initializer_list<test::BufferExtent>{
                           {0, rows.target_rows * 3 * 4, 4, "target RoPE position"},
                           {1, rows.draft_rows * 4, 4, "draft RoPE position"},
                           {2, SPLASH_TARGET_ROPE_PAIRS * 4, 4, "target inverse frequency"},
                           {3, SPLASH_DRAFT_ROPE_PAIRS * 4, 4, "draft inverse frequency"},
                           {4, rows.target_rows * SPLASH_TARGET_ROPE_PAIRS * 4, 4, "target RoPE cosine"},
                           {5, rows.target_rows * SPLASH_TARGET_ROPE_PAIRS * 4, 4, "target RoPE sine"},
                           {6, rows.draft_rows * SPLASH_DRAFT_ROPE_PAIRS * 4, 4, "draft RoPE cosine"},
                           {7, rows.draft_rows * SPLASH_DRAFT_ROPE_PAIRS * 4, 4, "draft RoPE sine"}},
                       [&](metal::CommandGraph &graph, const std::vector<metal::MetalBuffer> &b) {
                         ops::RoPE::addTables(graph, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], rows, 2048);
                       });
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc >= 1 && argc <= 3, "usage: paged-attention-plan [METALLIB [--long]]");
    for (auto format : {kv::Format::Int8, kv::Format::BFloat16})
      for (uint32_t heads : {24U, 16U})
        checkPlans(heads, {1, heads == 24 ? 4U : 2U, 256, format});
    if (argc == 1) {
      std::cout << "paged attention plans: CPU PASS\n";
      return 0;
    }
    metal::MetalBackend backend(argv[1]);
    checkBf16StoreEdges(backend);
    ropeExtents(backend);
    for (uint32_t heads : {24U, 16U})
      bufferExtents(backend, heads, {1, heads == 24 ? 4U : 2U, 256, kv::Format::Int8});
    // The prepare kernels do not depend on the KV format.
    for (uint32_t heads : {24U, 16U})
      for (bool float32 : {false, true})
        for (bool verify : {false, true})
          checkProjection(backend, heads, {1, heads == 24 ? 4U : 2U, 256, kv::Format::Int8}, float32,
                          verify);
    if (argc == 3 && std::string_view(argv[2]) == "--long") {
      for (auto format : {kv::Format::Int8, kv::Format::BFloat16})
        for (uint32_t heads : {24U, 16U})
          for (uint32_t history : {131072U, 260096U}) {
            const kv::Layout layout{1, heads == 24 ? 4U : 2U, 256, format};
            auto prefill = makeCase(backend, heads, layout, 1, 2048, history, false);
            checkReference(prefill, run<Phase::Prefill>(backend, prefill, true));
            auto verify = makeCase(backend, heads, layout, 4, 8, history, true);
            checkReference(verify, run<Phase::Verify>(backend, verify, true));
            std::cout << "long attention: format=" << kv::formatName(format)
                      << " q=" << heads << " history=" << history << " PASS\n" << std::flush;
          }
      std::cout << "long paged attention plans: PASS\n";
      return 0;
    }
    require(argc == 2, "usage: paged-attention-plan [METALLIB [--long]]");
    for (auto format : {kv::Format::Int8, kv::Format::BFloat16})
    for (uint32_t heads : {24U, 16U}) {
      const kv::Layout layout{1, heads == 24 ? 4U : 2U, 256, format};
      for (const auto [history, rows] :
           std::array<std::array<uint32_t, 2>, 10>{{{0, 1}, {33, 7}, {255, 17},
                                                  {1023, 8}, {0, 2048},
                                                  {4093, 1057}, {16383, 257},
                                                  {100, 77}, {8064, 130},
                                                  {6145, 2048}}})
        checkPrefill(backend, heads, layout, history, rows);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        checkVerify(backend, heads, layout, 0, lanes);
        checkVerify(backend, heads, layout, 1023, lanes);
      }
      checkVerify(backend, heads, layout, 16384, 2);
    }
    std::cout << "paged attention plans: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "paged attention plans: FAIL: " << error.what() << '\n';
    return 1;
  }
}
