#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace splash;

static_assert(!std::is_aggregate_v<ops::PrefillAttentionPlan> &&
              !std::is_default_constructible_v<ops::PrefillAttentionPlan> &&
              !std::is_copy_assignable_v<ops::PrefillAttentionPlan>);
static_assert(!std::is_aggregate_v<ops::VerifyAttentionPlan> &&
              !std::is_default_constructible_v<ops::VerifyAttentionPlan> &&
              !std::is_copy_assignable_v<ops::VerifyAttentionPlan>);

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid attention plan was accepted");
}

uint16_t bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}

float fp32(uint16_t value) {
  return std::bit_cast<float>(uint32_t{value} << 16);
}

void checkPrefillSlotOrientation(uint32_t queryHeads, kv::Layout layout,
                                ops::PrefillAttentionConfig config) {
  bool unequalAxes = false, partialTile = false, multipleSplits = false;
  // Enumerate nonsquare grids and partial final tiles. Each logical partial
  // belongs to exactly one query tile, KV head and balanced history split.
  for (const auto [history, rows] :
       std::array<std::array<uint32_t, 2>, 4>{{{4093, 17}, {16383, 257},
                                              {4095, 2048}, {131072, 17}}}) {
    const auto plan = ops::PagedAttention::prefillPlan(rows, queryHeads, layout,
                                                       history, config);
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
  const std::array<uint32_t, 4> zeroHistory{};
  for (const auto config : ops::PagedAttention::prefillCandidates()) {
    const std::string splitPipeline = std::string(layout.format == kv::Format::Int8 ? "prefill_attention_q8_split" : "prefill_attention_bf16_split") +
        (layout.format == kv::Format::Int8 && config.scalePlacement == ops::AttentionScalePlacement::Cooperative
             ? "_cooperative_scale" : "") + geometrySuffix;
    const std::string reducePipeline = "prefill_attention_q8_reduce" + geometrySuffix;
    checkPrefillSlotOrientation(queryHeads, layout, config);
    for (uint32_t rows = 1; rows <= 2048; ++rows)
      for (uint32_t history : {0U, 33U, 4095U, 4096U, 131072U,
                               kv::kMaximumPhysicalTokens - rows}) {
        const auto plan = ops::PagedAttention::prefillPlan(rows, queryHeads, layout,
                                                          history, config);
        const uint32_t tiles = (rows + 7) / 8;
        const uint32_t multiplier = static_cast<uint32_t>(config.splitMultiplier);
        const uint32_t splits = std::min(32U, multiplier * std::clamp(32U / tiles, 1U, 32U));
        require(plan.rows == rows && plan.historyTokens == history &&
                    plan.configuration == config && plan.splits == splits,
                "prefill plan lost actual rows or logical history");
        require(plan.splitPipeline == splitPipeline && plan.reducePipeline == reducePipeline,
                "prefill scale placement changed the wrong pipeline");
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
        const auto bound = ops::PagedAttention::prefillWorkspace(rows, queryHeads, layout, config);
        require(bound.partialsBytes >= plan.workspace.partialsBytes &&
                    bound.statisticsBytes >= plan.workspace.statisticsBytes,
                "prefill arena omitted a valid shorter/context-edge plan");
      }
  }
  for (const auto config : ops::PagedAttention::verifyCandidates()) {
    const std::string splitPipeline = std::string(layout.format == kv::Format::Int8 ? "verify_attention_q8_split" : "verify_attention_bf16_split") +
        (layout.format == kv::Format::Int8 && config.scalePlacement == ops::AttentionScalePlacement::Cooperative
             ? "_cooperative_scale" : "") + geometrySuffix;
    const std::string reducePipeline = "verify_attention_q8_reduce" + geometrySuffix;
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const std::array<uint32_t, 4> histories{0, 31, 16384, 131072};
      const auto plan =
          ops::PagedAttention::verifyPlan(lanes, queryHeads, layout, histories, config);
      require(plan.configuration == config && plan.splitPipeline == splitPipeline &&
                  plan.reducePipeline == reducePipeline,
              "verify scale or operand placement changed the wrong pipeline");
      const uint32_t base = static_cast<uint32_t>(config.splitCount);
      uint32_t maximum = 0;
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t expected = kv::q8VerifyAttentionSplits(base, histories[lane], 8);
        require(plan.laneSplits[lane] == expected && expected >= base &&
                    expected <= kv::kQ8VerifyMaximumSplits,
                "verify lane split count does not follow its own history");
        maximum = std::max(maximum, expected);
      }
      const uint64_t fused =
          uint64_t{lanes} * 8 * kv::kQ8VerifyMaximumSplits * queryHeads;
      require(plan.splits == maximum &&
                  plan.workspace.partialsBytes == fused * 256 * 4 &&
                  plan.workspace.statisticsBytes == fused * 2 * 4 &&
                  plan.splitGroups.y == plan.splits &&
                  plan.splitGroups.z == lanes &&
                  plan.reduceGroups.y == 8 * queryHeads / layout.kvHeads &&
                  plan.reduceGroups.z == lanes,
              "verify split/reduce/scratch disagree");
      if (config == ops::VerifyAttentionConfig{})
        require(plan.laneSplits[0] == 32 &&
                    (lanes < 4 || plan.laneSplits[3] == kv::kQ8VerifyMaximumSplits),
                "default verify partition changed");
    }
  }
  rejects([&] { (void)ops::PagedAttention::prefillPlan(0, queryHeads, layout, 0); });
  rejects([&] { (void)ops::PagedAttention::prefillPlan(2049, queryHeads, layout, 0); });
  rejects([&] { (void)ops::PagedAttention::prefillPlan(1, queryHeads, layout, kv::kMaximumPhysicalTokens); });
  rejects([&] { (void)ops::PagedAttention::prefillPlan(2048, queryHeads, layout, UINT32_MAX); });
  rejects([&] { (void)ops::PagedAttention::verifyPlan(0, queryHeads, layout, zeroHistory); });
  rejects([&] { (void)ops::PagedAttention::verifyPlan(5, queryHeads, layout, zeroHistory); });
  rejects([&] {
    const std::array<uint32_t, 2> two{};
    (void)ops::PagedAttention::verifyPlan(3, queryHeads, layout, two);
  });
  rejects([&] {
    const std::array<uint32_t, 1> beyond{kv::kMaximumPhysicalTokens};
    (void)ops::PagedAttention::verifyPlan(1, queryHeads, layout, beyond);
  });
  rejects([&] {
    (void)ops::PagedAttention::prefillPlan(
        8, queryHeads, layout, 0,
        {static_cast<ops::PrefillSplitMultiplier>(0)});
  });
  rejects([&] {
    (void)ops::PagedAttention::prefillPlan(
        8, queryHeads, layout, 0,
        {static_cast<ops::PrefillSplitMultiplier>(3)});
  });
  rejects([&] {
    (void)ops::PagedAttention::verifyPlan(
        1, queryHeads, layout, zeroHistory, {static_cast<ops::VerifySplitCount>(0)});
  });
  const auto invalidPlacement = static_cast<ops::AttentionScalePlacement>(2);
  rejects([&] {
    (void)ops::PagedAttention::prefillPlan(
        8, queryHeads, layout, 0, {ops::PrefillSplitMultiplier::One, invalidPlacement});
  });
  rejects([&] {
    (void)ops::PagedAttention::prefillWorkspace(
        8, queryHeads, layout, {ops::PrefillSplitMultiplier::One, invalidPlacement});
  });
  rejects([&] {
    (void)ops::PagedAttention::verifyPlan(
        1, queryHeads, layout, zeroHistory,
        {ops::VerifySplitCount::ThirtyTwo, invalidPlacement});
  });
  rejects([&] { (void)ops::PagedAttention::verifyPlan(1, queryHeads + 1, layout, zeroHistory); });
  kv::Q8VerifyAttentionParams params{0, 8, 32, 1, 1, 0, 0, 0};
  require(kv::q8VerifyAttentionValidationError(params) == "split_count_invalid",
          "zero split count is not a supported configuration");
  params = {0, 8, 32, 1, 1, 32, 16, 0};
  require(kv::q8VerifyAttentionValidationError(params) == "slot_splits_invalid",
          "a slot stride below the split count is not a valid partition");
}

struct Case final {
  uint32_t queryHeads;
  kv::Layout layout;
  uint32_t lanes;
  uint32_t rows;
  uint32_t stride;
  kv::LayerStorage layer;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer queries;
  std::array<metal::MetalBuffer, 4> tables;
  std::array<kv::Q8ChunkedPrefillParams, 4> stores{};
  std::array<kv::Q8VerifyAttentionParams, 4> attention{};

  uint64_t queryIndex(uint32_t lane, uint32_t head, uint32_t row,
                      uint32_t dimension) const {
    const uint32_t group = queryHeads / layout.kvHeads;
    return (((uint64_t{lane} * layout.kvHeads + head / group) * stride + row) *
                group +
            head % group) *
               256 +
           dimension;
  }

  uint64_t scaleIndex(uint32_t lane, uint32_t head, uint32_t token) const {
    const auto *table = static_cast<const uint32_t *>(tables[lane].contents());
    return (uint64_t{table[token / 32]} * layout.kvHeads + head) * 32 +
           token % 32;
  }

  float key(uint32_t lane, uint32_t head, uint32_t token,
             uint32_t dimension) const {
    const uint64_t index = scaleIndex(lane, head, token);
    if (layout.format == kv::Format::BFloat16)
      return fp32(static_cast<const uint16_t *>(layer.keyData.contents())[index * 256 + dimension]);
    return static_cast<const int8_t *>(layer.keyData.contents())[index * 256 +
                                                                 dimension] *
           static_cast<const float *>(layer.keyScales.contents())[index];
  }

  float value(uint32_t lane, uint32_t head, uint32_t token,
               uint32_t dimension) const {
    const uint64_t index = scaleIndex(lane, head, token);
    const uint64_t dataIndex = (index / 32 * 256 + dimension) * 32 + index % 32;
    if (layout.format == kv::Format::BFloat16)
      return fp32(static_cast<const uint16_t *>(layer.valueData.contents())[dataIndex]);
    return static_cast<const int8_t *>(layer.valueData.contents())[dataIndex] *
           static_cast<const float *>(layer.valueScales.contents())[index];
  }
};

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  if (!bytes) return {};
  auto buffer = backend.allocateBuffer(bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

Case makeCase(metal::MetalBackend &backend, uint32_t queryHeads,
               kv::Layout layout, uint32_t lanes, uint32_t rows,
               uint32_t history, bool verify) {
  Case data{queryHeads, layout, lanes, rows, (rows + 31) / 32 * 32,
            {}, {}, {}, {}, {}, {}, {}};
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
  const uint32_t physicalPages = 2 * allPages + 1;
  const uint64_t dataBytes = physicalPages * layout.dataBytesPerLayerPage();
  const uint64_t scaleBytes = physicalPages * layout.scaleBytesPerLayerPage();
  data.layer = {allocate(backend, dataBytes), allocate(backend, scaleBytes),
                allocate(backend, dataBytes), allocate(backend, scaleBytes), layout.format};
  data.keys = allocate(backend, uint64_t{lanes} * layout.kvHeads * data.stride * 256 * 2);
  data.values = allocate(backend, data.keys.sizeBytes());
  data.queries = allocate(backend, uint64_t{lanes} * queryHeads * data.stride * 256 * 2);
  uint32_t firstPage = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    data.tables[lane] = allocate(backend, uint64_t{pageCounts[lane]} * 4);
    auto *table = static_cast<uint32_t *>(data.tables[lane].contents());
    for (uint32_t page = 0; page < pageCounts[lane]; ++page)
      table[page] = 2 * (firstPage + page) + 1;
    firstPage += pageCounts[lane];
    data.stores[lane] = {historyLengths[lane], rows, data.stride,
                         pageCounts[lane], physicalPages, 0, 0, 0};
    data.attention[lane] = {historyLengths[lane], rows, data.stride,
                            pageCounts[lane], physicalPages, 32, 32, 0};
    if (verify)
      data.attention[lane].active_rows = std::array{8U, 1U, 3U, 7U}[lane];
    for (uint32_t token = 0; token < historyLengths[lane]; ++token) {
      for (uint32_t head = 0; head < layout.kvHeads; ++head) {
        const uint64_t scaleIndex = data.scaleIndex(lane, head, token);
        if (layout.format == kv::Format::Int8) {
          static_cast<float *>(data.layer.keyScales.contents())[scaleIndex] = 0.006f;
          static_cast<float *>(data.layer.valueScales.contents())[scaleIndex] = 0.007f;
        }
        for (uint32_t dimension = 0; dimension < 256; ++dimension) {
          const int key = int((token * 37 + head * 101 + dimension * 17 +
                               token * dimension * 3 + lane * 7) % 255) - 127;
          const int value = int((token * 53 + head * 79 + dimension * 29 +
                                 token * dimension * 5 + lane * 19) % 255) - 127;
          const uint64_t valueIndex =
              (scaleIndex / 32 * 256 + dimension) * 32 + scaleIndex % 32;
          if (layout.format == kv::Format::Int8) {
            static_cast<int8_t *>(data.layer.keyData.contents())[scaleIndex * 256 + dimension] = key;
            static_cast<int8_t *>(data.layer.valueData.contents())[valueIndex] = value;
          } else {
            static_cast<uint16_t *>(data.layer.keyData.contents())[scaleIndex * 256 + dimension] = bf16(key * 0.006f);
            static_cast<uint16_t *>(data.layer.valueData.contents())[valueIndex] = bf16(value * 0.007f);
          }
        }
      }
    }
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t head = 0; head < layout.kvHeads; ++head) {
        const uint64_t base = (uint64_t{lane} * layout.kvHeads + head) * data.stride * 256;
        for (uint32_t dimension = 0; dimension < 256; ++dimension) {
          static_cast<uint16_t *>(data.keys.contents())[base + row * 256 + dimension] =
              bf16(float(int((row * 37 + head * 101 + dimension * 17) % 255) - 127) * 0.006f);
          static_cast<uint16_t *>(data.values.contents())[base + dimension * data.stride + row] =
              bf16(float(int((row * 53 + head * 79 + dimension * 29) % 255) - 127) * 0.007f);
        }
      }
      for (uint32_t head = 0; head < queryHeads; ++head)
        for (uint32_t dimension = 0; dimension < 256; ++dimension)
          static_cast<uint16_t *>(data.queries.contents())[
              data.queryIndex(lane, head, row, dimension)] =
              bf16(float(int((row * 43 + head * 67 + dimension * 11 +
                              head * dimension * 7) % 1019) - 509) / 1018.0f);
    }
  }
  for (uint32_t lane = lanes; lane < 4; ++lane) {
    data.tables[lane] = data.tables[0];
    data.stores[lane] = data.stores[0];
    data.attention[lane] = data.attention[0];
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
      if (row >= data.attention[lane].active_rows)
        continue;
      for (uint32_t head : {0U, data.queryHeads - 1}) {
        const uint32_t kvHead = head / (data.queryHeads / data.layout.kvHeads);
        const uint32_t tokens = data.stores[lane].committed_tokens + row + 1;
        std::vector<float> scores(tokens);
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t token = 0; token < tokens; ++token) {
          float score = 0;
          for (uint32_t dimension = 0; dimension < 256; ++dimension)
            score += fp32(static_cast<const uint16_t *>(data.queries.contents())[
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
          const float value = fp32(actual[data.queryIndex(lane, head, row, dimension)]);
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
          const float left = fp32(baseline[index]);
          const float right = fp32(candidate[index]);
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

// Construct expected pages from the source bits, independently of GPU stores.
std::array<std::vector<uint16_t>, 2> expectedBf16Store(const Case &data) {
  std::array<std::vector<uint16_t>, 2> result;
  for (unsigned tensor = 0; tensor < 2; ++tensor) {
    const auto cache = tensor ? data.layer.valueData : data.layer.keyData;
    const auto *before = static_cast<const uint16_t *>(cache.contents());
    result[tensor].assign(before, before + cache.sizeBytes() / 2);
    const auto *source = static_cast<const uint16_t *>(
        (tensor ? data.values : data.keys).contents());
    for (uint32_t lane = 0; lane < data.lanes; ++lane) {
      const auto *table = static_cast<const uint32_t *>(data.tables[lane].contents());
      for (uint32_t row = 0; row < data.stores[lane].chunk_tokens; ++row) {
        const uint32_t token = data.stores[lane].committed_tokens + row;
        for (uint32_t head = 0; head < data.layout.kvHeads; ++head)
          for (uint32_t d = 0; d < 256; ++d) {
            const uint64_t pageHead = uint64_t{table[token / 32]} * data.layout.kvHeads + head;
            const uint64_t destination = tensor ? (pageHead * 256 + d) * 32 + token % 32
                                                : (pageHead * 32 + token % 32) * 256 + d;
            const uint64_t base = (uint64_t{lane} * data.layout.kvHeads + head) * data.stride * 256;
            const uint64_t input = base + (tensor ? d * data.stride + row : row * 256 + d);
            result[tensor][destination] = source[input];
          }
      }
    }
  }
  return result;
}

void checkBf16Store(const Case &data, const std::array<std::vector<uint16_t>, 2> &expected) {
  for (unsigned tensor = 0; tensor < 2; ++tensor) {
    const auto cache = tensor ? data.layer.valueData : data.layer.keyData;
    require(std::memcmp(cache.contents(), expected[tensor].data(), cache.sizeBytes()) == 0,
            "BF16 store changed source bits, history, or an unused page slot");
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
    (void)backend.submitCommand(graph.dispatches());
    checkBf16Store(data, expected);
  }
}

template <class Config>
std::vector<uint16_t> run(metal::MetalBackend &backend, Case &data,
                           Config config, bool testBounds) {
  const auto plan = [&] {
    if constexpr (std::is_same_v<Config, ops::PrefillAttentionConfig>)
      return ops::PagedAttention::prefillPlan(data.rows, data.queryHeads, data.layout,
                                             data.stores[0].committed_tokens, config);
    else {
      std::array<uint32_t, 4> histories{};
      for (uint32_t lane = 0; lane < data.lanes; ++lane)
        histories[lane] = data.attention[lane].committed_tokens;
      return ops::PagedAttention::verifyPlan(data.lanes, data.queryHeads, data.layout,
                                            histories, config);
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
  auto encode = [&](metal::CommandGraph &graph, metal::MetalBuffer partialBuffer,
                     metal::MetalBuffer statisticsBuffer) {
    if constexpr (std::is_same_v<Config, ops::PrefillAttentionConfig>) {
      ops::PagedAttention::addPrefill(graph, data.layer, data.queries, output,
                                      partialBuffer, statisticsBuffer, data.tables[0],
                                      data.stores[0], plan);
    } else {
      ops::PagedVerifyBuffers buffers{data.keys, data.values, data.queries,
                                      partialBuffer, statisticsBuffer, output, data.tables};
      ops::PagedAttention::addVerify(
          graph, data.layer, buffers, data.stores, data.attention, plan);
    }
  };
  if (testBounds) {
    metal::CommandGraph shortGraph;
    const auto originalFormat = data.layer.format;
    data.layer.format = originalFormat == kv::Format::Int8 ? kv::Format::BFloat16 : kv::Format::Int8;
    rejects([&] { encode(shortGraph, partials, statistics); });
    data.layer.format = originalFormat;
    require(shortGraph.empty(), "mismatched KV format partially encoded a graph");
    if constexpr (std::is_same_v<Config, ops::PrefillAttentionConfig>) {
      auto mismatch = data.stores[0];
      mismatch.chunk_tokens = plan.rows == 1 ? 2 : plan.rows - 1;
      rejects([&] {
        ops::PagedAttention::addPrefill(shortGraph, data.layer, data.queries, output,
                                       partials, statistics, data.tables[0], mismatch, plan);
      });
      require(shortGraph.empty(), "mismatched prefill plan partially encoded a graph");
      mismatch = data.stores[0];
      ++mismatch.committed_tokens;
      rejects([&] {
        ops::PagedAttention::addPrefill(shortGraph, data.layer, data.queries, output,
                                       partials, statistics, data.tables[0], mismatch, plan);
      });
      require(shortGraph.empty(), "mismatched history partially encoded a graph");
    }
    rejects([&] {
      encode(shortGraph, backend.view(partials, 0, partials.sizeBytes() - 4),
             statistics);
    });
    require(shortGraph.empty(), "undersized partial scratch partially encoded a graph");
    rejects([&] {
      encode(shortGraph, partials,
             backend.view(statistics, 0, statistics.sizeBytes() - 4));
    });
    require(shortGraph.empty(), "undersized statistic scratch partially encoded a graph");
  }
  metal::CommandGraph graph;
  if constexpr (std::is_same_v<Config, ops::PrefillAttentionConfig>)
    ops::PagedAttention::addPrefillStore(graph, data.layer, data.keys, data.values,
                                        data.tables[0], data.stores[0], data.layout);
  encode(graph, partials, statistics);
  if constexpr (std::is_same_v<Config, ops::PrefillAttentionConfig>) {
    require(graph.dispatches().size() == 3,
            "production prefill should encode store/split/reduce");
    const auto checkDispatch = [&](const auto &dispatch, auto groups, std::string_view pipeline) {
      require(dispatch.pipelineName == pipeline && dispatch.threadgroups.x == groups.x &&
                  dispatch.threadgroups.y == groups.y && dispatch.threadgroups.z == groups.z &&
                  dispatch.bytes.size() == 1 &&
                  dispatch.bytes[0].sizeBytes == sizeof(kv::Q8PrefillAttentionParams),
              "production prefill dispatch departed from its plan");
      kv::Q8PrefillAttentionParams params;
      std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
      require(params.committed_tokens == plan.historyTokens && params.rows == data.rows &&
                  params.chunk_stride == data.stride &&
                  params.page_table_entries == data.stores[0].page_table_entries &&
                  params.physical_page_count == data.stores[0].physical_page_count &&
                  params.split_count == plan.splits &&
                  params.reserved0 == 0 && params.reserved1 == 0,
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
                            : std::array<std::vector<uint16_t>, 2>{};
  (void)backend.submitCommand(graph.dispatches());
  if (data.layout.format == kv::Format::BFloat16) checkBf16Store(data, expected);
  for (size_t i = 0; i < sizes.size(); ++i) {
    const auto *bytes = static_cast<const uint8_t *>(backing[i].contents());
    for (uint64_t byte = 0; byte < guardBytes; ++byte)
      require(bytes[byte] == 0xa5 && bytes[guardBytes + sizes[i] + byte] == 0xa5,
              "attention scratch/output write canary changed");
  }
  const auto *values = static_cast<const uint16_t *>(output.contents());
  if constexpr (std::is_same_v<Config, ops::VerifyAttentionConfig>)
    for (uint32_t lane = 0; lane < data.lanes; ++lane)
      for (uint32_t row = data.attention[lane].active_rows; row < data.rows; ++row)
        for (uint32_t head = 0; head < data.queryHeads; ++head)
          for (uint32_t dimension = 0; dimension < 256; ++dimension)
            require(values[data.queryIndex(lane, head, row, dimension)] == 0,
                    "inactive verify rows were not zeroed");
  return {values, values + output.sizeBytes() / 2};
}

void checkPrefill(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout,
                   uint32_t history, uint32_t rows) {
  auto data = makeCase(backend, heads, layout, 1, rows, history, false);
  std::vector<uint16_t> defaultOutput;
  for (const auto config : ops::PagedAttention::prefillCandidates()) {
    const auto output = run(backend, data, config, true);
    checkReference(data, output);
    if (config == ops::PrefillAttentionConfig{})
      defaultOutput = output;
    checkEquivalent(data, output, run(backend, data, config, false));
  }
  require(!defaultOutput.empty(), "default prefill configuration was not tested");
  checkEquivalent(data, defaultOutput,
                  run(backend, data, ops::PrefillAttentionConfig{}, false));
  if (rows == 1057) {
    // Reuse the identical packed BF16 inputs and KV history across unaligned
    // host chunks, checking each path against its independent causal oracle.
    const auto copy = [](metal::MetalBuffer buffer) {
      const auto *begin = static_cast<const uint16_t *>(buffer.contents());
      return std::vector<uint16_t>(begin, begin + buffer.sizeBytes() / 2);
    };
    const auto keys = copy(data.keys), values = copy(data.values), queries = copy(data.queries);
    for (const auto config : ops::PagedAttention::prefillCandidates()) {
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
        const auto output = run(backend, data, config, true);
        checkReference(data, output);
        checkEquivalent(data, output, run(backend, data, config, false));
        offset += chunk;
      }
      require(offset == rows, "chunk comparison dropped logical query rows");
    }
  }
  std::cout << "paged prefill candidates: format=" << kv::formatName(layout.format) << " q=" << heads << " history=" << history
            << " rows=" << rows << " PASS\n";
}

void checkVerify(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout,
                  uint32_t history, uint32_t lanes) {
  auto data = makeCase(backend, heads, layout, lanes, 8, history, true);
  std::vector<uint16_t> baseline;
  for (const auto config : ops::PagedAttention::verifyCandidates()) {
    const auto output = run(backend, data, config, true);
    checkReference(data, output);
    if (baseline.empty())
      baseline = output;
    checkEquivalent(data, baseline, output);
  }
  std::cout << "paged verify candidates: format=" << kv::formatName(layout.format) << " q=" << heads << " history=" << history
            << " lanes=" << lanes << " PASS\n";
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
    if (argc == 3 && std::string_view(argv[2]) == "--long") {
      for (auto format : {kv::Format::Int8, kv::Format::BFloat16})
        for (uint32_t heads : {24U, 16U})
          for (uint32_t history : {131072U, 260096U}) {
            const kv::Layout layout{1, heads == 24 ? 4U : 2U, 256, format};
            auto prefill = makeCase(backend, heads, layout, 1, 2048, history, false);
            checkReference(prefill, run(backend, prefill, ops::PrefillAttentionConfig{}, true));
            auto verify = makeCase(backend, heads, layout, 4, 8, history, true);
            checkReference(verify, run(backend, verify, ops::VerifyAttentionConfig{}, true));
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
