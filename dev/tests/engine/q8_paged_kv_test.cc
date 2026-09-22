#include "Q8PageFormatReference.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace splash::kv;

namespace {

void testByteAccounting() {
  constexpr Layout bf16{16, 4, 256, Format::BFloat16};
  static_assert(bf16.valid());
  static_assert(bf16.bytesPerModelPage() == 2'097'152);
  static_assert(bf16.sparseMappingBatchPages() == 1);
  static_assert(bf16.backingExtentPages() == 64);
  static_assert(bf16.storageByteCounts(4096).total == 8ULL * 1024 * 1024 * 1024);
  static_assert(bf16.scaleBytesPerLayerPage() == 0);
  constexpr Layout compact{10, 2, 256, Format::BFloat16};
  static_assert(compact.bytesPerModelPage() == 655'360);
  static_assert(compact.sparseMappingBatchPages() == 2);
  static_assert(compact.backingExtentPages() == 206);
  static_assert(!Layout{16, 4, 256, static_cast<Format>(0)}.valid());
  static_assert(kBytesPerModelPage == 1'064'960);
  StorageByteCounts one = storageByteCounts(1);
  assert(one.keyData == 512 * 1024);
  assert(one.keyScales == 8 * 1024);
  assert(one.valueData == 512 * 1024);
  assert(one.valueScales == 8 * 1024);
  assert(one.total == 1'064'960);
  StorageByteCounts production = storageByteCounts(4608);
  assert(production.total == 4'907'335'680ULL); // 147456 tokens.
}

void testLayouts() {
  assert(keyDataIndex(0, 0, 1) == keyDataIndex(0, 0, 0) + 1);
  assert(keyDataIndex(0, 1, 0) == keyDataIndex(0, 0, 0) + kHeadDimension);
  assert(valueDataIndex(0, 1, 0) == valueDataIndex(0, 0, 0) + 1);
  assert(valueDataIndex(0, 0, 1) == valueDataIndex(0, 0, 0) + kPageTokens);
  assert(keyDataIndex(kKvHeads - 1, kPageTokens - 1, kHeadDimension - 1) ==
         kElementsPerLayerPage - 1);
  assert(valueDataIndex(kKvHeads - 1, kPageTokens - 1, kHeadDimension - 1) ==
         kElementsPerLayerPage - 1);
}

void testBFloat16() {
  for (float value : {0.0f, -0.0f, 1.0f, -3.5f, 1.0e-20f, 65504.0f}) {
    float roundTrip = bfloat16ToFloat(floatToBFloat16(value));
    if (value == 0.0f) {
      assert(roundTrip == value);
    } else {
      assert(std::abs(roundTrip - value) <= std::abs(value) / 128.0f);
    }
  }
  assert(std::isinf(bfloat16ToFloat(
      floatToBFloat16(std::numeric_limits<float>::infinity()))));
  assert(std::isnan(bfloat16ToFloat(
      floatToBFloat16(std::numeric_limits<float>::quiet_NaN()))));
}

void testLayoutGuard() {
  std::array<uint8_t, 32> digest{};
  for (uint32_t i = 0; i < digest.size(); ++i)
    digest[i] = uint8_t(i);
  LayoutGuard first = makeLayoutGuard(kOracleLayout, digest);
  LayoutGuard second = makeLayoutGuard(kOracleLayout, digest);
  auto bf16 = kOracleLayout;
  bf16.format = Format::BFloat16;
  const auto bf16Guard = makeLayoutGuard(bf16, digest);
  assert(bf16Guard.quantization != first.quantization);
  assert(bf16Guard.scaleType == uint32_t(ScaleType::None));
  assert(bf16Guard.elementsPerScale == 0);
  assert(bf16Guard.quantizedMinimum == 0 && bf16Guard.quantizedMaximum == 0);
  assert(bf16Guard.modelArtifactSha256 == first.modelArtifactSha256);
  assert(matchesLayout(first, kOracleLayout));
  assert(first.modelArtifactSha256 == second.modelArtifactSha256);
  second.elementsPerScale = 32;
  assert(!isValidLayoutGuard(second));
  second = first;
  ++second.modelArtifactSha256[0];
  assert(isValidLayoutGuard(second));
  assert(first.modelArtifactSha256 != second.modelArtifactSha256);
}

void testQuantization(uint32_t validTokens) {
  uint64_t inputElements = uint64_t{validTokens} * kKvHeads * kHeadDimension;
  std::vector<float> keys(inputElements);
  std::vector<float> values(inputElements);
  for (uint32_t token = 0; token < validTokens; ++token) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        uint64_t index = logicalIndex(token, head, dimension);
        keys[index] =
            2.5f * std::sin(float(token * 17 + head * 31 + dimension) * 0.013f);
        values[index] =
            1.75f *
            std::cos(float(token * 29 + head * 11 + dimension) * 0.017f);
      }
    }
  }
  auto page = std::make_unique<Q8LayerPage>();
  quantizeLayerPage(keys, values, validTokens, *page);
  std::vector<float> decodedKeys(kElementsPerLayerPage);
  std::vector<float> decodedValues(kElementsPerLayerPage);
  dequantizeLayerPage(*page, validTokens, decodedKeys, decodedValues);

  for (uint32_t token = 0; token < validTokens; ++token) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        uint64_t logical = logicalIndex(token, head, dimension);
        float keyScale = page->keyScales[keyScaleIndex(head, token)];
        float valueScale = page->valueScales[valueScaleIndex(head, token)];
        assert(std::abs(decodedKeys[logical] - keys[logical]) <=
               keyScale * 0.51f + 1.0e-7f);
        assert(std::abs(decodedValues[logical] - values[logical]) <=
               valueScale * 0.51f + 1.0e-7f);
        assert(decodedKeys[logical] ==
               dequantizeKey(*page, head, token, dimension));
        assert(decodedValues[logical] ==
               dequantizeValue(*page, head, token, dimension));
      }
    }
  }
  for (uint32_t token = validTokens; token < kPageTokens; ++token) {
    uint64_t logical = logicalIndex(token, 0, 0);
    assert(decodedKeys[logical] == 0.0f);
    assert(decodedValues[logical] == 0.0f);
  }
}

void testZeroAndInvalidInputs() {
  constexpr uint32_t tokens = 1;
  std::vector<float> values(uint64_t{tokens} * kKvHeads * kHeadDimension);
  auto page = std::make_unique<Q8LayerPage>();
  quantizeLayerPage(values, values, tokens, *page);
  assert(std::all_of(page->keys.begin(), page->keys.end(),
                     [](int8_t value) { return value == 0; }));
  assert(std::all_of(page->values.begin(), page->values.end(),
                     [](int8_t value) { return value == 0; }));
  values[0] = std::numeric_limits<float>::quiet_NaN();
  bool rejected = false;
  try {
    quantizeLayerPage(values, values, tokens, *page);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}

} // namespace

int main() {
  testByteAccounting();
  testLayouts();
  testBFloat16();
  testLayoutGuard();
  testQuantization(kPageTokens);
  testQuantization(17);
  testZeroAndInvalidInputs();
  std::cout << "q8_paged_kv_test: ok\n";
}
