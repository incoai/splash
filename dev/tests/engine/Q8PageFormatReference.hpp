#pragma once

#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace splash::kv {

using BFloat16Bits = uint16_t;

// This oracle targets the Qwen3.8 kernel specialization.
inline constexpr Layout kOracleLayout{16, 4, 256};
inline constexpr uint32_t kKvHeads = kOracleLayout.kvHeads;
inline constexpr uint32_t kHeadDimension = kOracleLayout.headDimension;
inline constexpr uint64_t kElementsPerLayerPage =
    kOracleLayout.elementsPerLayerPage();
inline constexpr uint64_t kScalesPerTensorLayerPage =
    kOracleLayout.scalesPerTensorLayerPage();
inline constexpr uint64_t kKeyDataBytesPerLayerPage =
    kOracleLayout.dataBytesPerLayerPage();
inline constexpr uint64_t kKeyScaleBytesPerLayerPage =
    kOracleLayout.scaleBytesPerLayerPage();
inline constexpr uint64_t kValueDataBytesPerLayerPage =
    kOracleLayout.dataBytesPerLayerPage();
inline constexpr uint64_t kValueScaleBytesPerLayerPage =
    kOracleLayout.scaleBytesPerLayerPage();
inline constexpr uint64_t kBytesPerLayerPage =
    kOracleLayout.bytesPerLayerPage();
inline constexpr uint64_t kBytesPerModelPage =
    kOracleLayout.bytesPerModelPage();

[[nodiscard]] constexpr StorageByteCounts
storageByteCounts(uint64_t pages) noexcept {
  return kOracleLayout.storageByteCounts(pages);
}

struct Q8LayerPage final {
  std::array<int8_t, kElementsPerLayerPage> keys{};
  std::array<float, kScalesPerTensorLayerPage> keyScales{};
  std::array<int8_t, kElementsPerLayerPage> values{};
  std::array<float, kScalesPerTensorLayerPage> valueScales{};
};

static_assert(sizeof(Q8LayerPage) == kBytesPerLayerPage);
static_assert(std::is_standard_layout_v<Q8LayerPage>);

struct Q8KVMetalPageParams final {
  uint32_t sourcePage = 0;
  uint32_t destinationPage = 0;
  uint32_t validTokens = kPageTokens;
  uint32_t reserved = 0;
};

static_assert(sizeof(Q8KVMetalPageParams) == 16);
static_assert(std::is_standard_layout_v<Q8KVMetalPageParams>);

constexpr uint64_t logicalIndex(uint32_t token, uint32_t head,
                                uint32_t dimension) {
  return (uint64_t{token} * kKvHeads + head) * kHeadDimension + dimension;
}

constexpr uint64_t keyDataIndex(uint32_t head, uint32_t token,
                                uint32_t dimension) {
  return (uint64_t{head} * kPageTokens + token) * kHeadDimension + dimension;
}

constexpr uint64_t keyScaleIndex(uint32_t head, uint32_t token) {
  return uint64_t{head} * kPageTokens + token;
}

constexpr uint64_t valueDataIndex(uint32_t head, uint32_t token,
                                  uint32_t dimension) {
  return (uint64_t{head} * kHeadDimension + dimension) * kPageTokens + token;
}

constexpr uint64_t valueScaleIndex(uint32_t head, uint32_t token) {
  return uint64_t{head} * kPageTokens + token;
}

inline BFloat16Bits floatToBFloat16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    uint16_t upper = static_cast<uint16_t>(bits >> 16);
    if ((bits & 0x007fffffU) && !(upper & 0x007fU))
      upper |= 1;
    return upper;
  }
  const uint32_t roundingBias = 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>((bits + roundingBias) >> 16);
}

inline float bfloat16ToFloat(BFloat16Bits value) {
  return std::bit_cast<float>(uint32_t{value} << 16);
}

namespace reference_detail {

constexpr uint64_t logicalElements(uint32_t validTokens) {
  return uint64_t{validTokens} * kKvHeads * kHeadDimension;
}

inline void validateInput(std::span<const float> values,
                          uint32_t validTokens, const char *name) {
  if (validTokens > kPageTokens)
    throw std::invalid_argument("Q8 KV valid token count exceeds one page");
  if (values.size() != logicalElements(validTokens))
    throw std::invalid_argument(std::string(name) +
                                " has the wrong logical page size");
  if (std::any_of(values.begin(), values.end(),
                  [](float value) { return !std::isfinite(value); }))
    throw std::invalid_argument(std::string(name) +
                                " contains a non-finite value");
}

inline void validateOutput(std::span<float> values, const char *name) {
  if (values.size() != kElementsPerLayerPage)
    throw std::invalid_argument(std::string(name) +
                                " has the wrong logical page size");
}

inline float storedScale(float maximum) {
  return maximum == 0.0F ? 0.0F : maximum / float(kQuantizedMaximum);
}

inline int8_t quantize(float value, float maximum) {
  if (maximum == 0.0F)
    return 0;
  long quantized = static_cast<long>(
      std::nearbyint(value * float(kQuantizedMaximum) / maximum));
  quantized =
      std::clamp<long>(quantized, kQuantizedMinimum, kQuantizedMaximum);
  return static_cast<int8_t>(quantized);
}

inline void checkElement(uint32_t head, uint32_t token, uint32_t dimension) {
  if (head >= kKvHeads || token >= kPageTokens || dimension >= kHeadDimension)
    throw std::out_of_range("Q8 KV element is outside the page");
}

} // namespace reference_detail

inline float dequantizeKey(const Q8LayerPage &source, uint32_t head,
                           uint32_t token, uint32_t dimension) {
  reference_detail::checkElement(head, token, dimension);
  return float(source.keys[keyDataIndex(head, token, dimension)]) *
         source.keyScales[keyScaleIndex(head, token)];
}

inline float dequantizeValue(const Q8LayerPage &source, uint32_t head,
                             uint32_t token, uint32_t dimension) {
  reference_detail::checkElement(head, token, dimension);
  return float(source.values[valueDataIndex(head, token, dimension)]) *
         source.valueScales[valueScaleIndex(head, token)];
}

inline void quantizeLayerPage(std::span<const float> logicalKeys,
                              std::span<const float> logicalValues,
                              uint32_t validTokens,
                              Q8LayerPage &destination) {
  reference_detail::validateInput(logicalKeys, validTokens, "logical keys");
  reference_detail::validateInput(logicalValues, validTokens,
                                  "logical values");
  destination = {};
  for (uint32_t head = 0; head < kKvHeads; ++head) {
    for (uint32_t token = 0; token < validTokens; ++token) {
      float keyMaximum = 0.0F;
      float valueMaximum = 0.0F;
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        keyMaximum = std::max(
            keyMaximum,
            std::abs(logicalKeys[logicalIndex(token, head, dimension)]));
        valueMaximum = std::max(
            valueMaximum,
            std::abs(logicalValues[logicalIndex(token, head, dimension)]));
      }
      destination.keyScales[keyScaleIndex(head, token)] =
          reference_detail::storedScale(keyMaximum);
      destination.valueScales[valueScaleIndex(head, token)] =
          reference_detail::storedScale(valueMaximum);
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        destination.keys[keyDataIndex(head, token, dimension)] =
            reference_detail::quantize(
                logicalKeys[logicalIndex(token, head, dimension)], keyMaximum);
        destination.values[valueDataIndex(head, token, dimension)] =
            reference_detail::quantize(
                logicalValues[logicalIndex(token, head, dimension)],
                valueMaximum);
      }
    }
  }
}

inline void dequantizeLayerPage(const Q8LayerPage &source,
                                uint32_t validTokens,
                                std::span<float> logicalKeys,
                                std::span<float> logicalValues) {
  if (validTokens > kPageTokens)
    throw std::invalid_argument("Q8 KV valid token count exceeds one page");
  reference_detail::validateOutput(logicalKeys, "logical keys");
  reference_detail::validateOutput(logicalValues, "logical values");
  std::fill(logicalKeys.begin(), logicalKeys.end(), 0.0F);
  std::fill(logicalValues.begin(), logicalValues.end(), 0.0F);
  for (uint32_t token = 0; token < validTokens; ++token) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        const uint64_t output = logicalIndex(token, head, dimension);
        logicalKeys[output] = dequantizeKey(source, head, token, dimension);
        logicalValues[output] =
            dequantizeValue(source, head, token, dimension);
      }
    }
  }
}

// Test-only guard checks: production compares guards through the runtime
// cache identity, so these helpers live with the tests that assert on them.
[[nodiscard]] inline bool isValidLayoutGuard(const LayoutGuard &guard) {
  const Layout layout{guard.attentionLayers, guard.kvHeads,
                        guard.headDimension};
  return layout.valid() &&
         guard.quantization == uint32_t(Quantization::SymmetricInt8) &&
         guard.scaleType == uint32_t(ScaleType::Float32) &&
         guard.keyLayout == uint32_t(KeyLayout::TokenMajor) &&
         guard.valueLayout == uint32_t(ValueLayout::DimensionMajor) &&
         guard.pageTokens == kPageTokens &&
         guard.elementsPerScale == layout.elementsPerScale() &&
         guard.quantizedMinimum == kQuantizedMinimum &&
         guard.quantizedMaximum == kQuantizedMaximum &&
         guard.bytesPerLayerPage == layout.bytesPerLayerPage() &&
         guard.bytesPerModelPage == layout.bytesPerModelPage();
}

[[nodiscard]] inline bool matchesLayout(const LayoutGuard &guard,
                                          Layout layout) {
  return isValidLayoutGuard(guard) &&
         guard.attentionLayers == layout.attentionLayers &&
         guard.kvHeads == layout.kvHeads &&
         guard.headDimension == layout.headDimension;
}

} // namespace splash::kv
