#include "Q8PageFormatReference.hpp"
#include "TestChecks.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace splash::kv;
using splash::test::rejects;
using splash::test::require;

namespace {

void testByteAccounting() {
  constexpr Layout bf16{16, 4, 256, Format::BFloat16};
  static_assert(bf16.valid());
  static_assert(bf16.bytesPerModelPage() == 2'097'152);
  static_assert(bf16.extentAlignmentPages() == 1);
  static_assert(bf16.minimumExtentPages() == 32 && bf16.maximumExtentPages() == 96);
  static_assert(4096 * bf16.bytesPerModelPage() == 8ULL * 1024 * 1024 * 1024);
  static_assert(bf16.scaleBytesPerLayerPage() == 0);
  constexpr Layout compact{10, 2, 256, Format::BFloat16};
  static_assert(compact.bytesPerModelPage() == 655'360);
  static_assert(compact.extentAlignmentPages() == 2);
  static_assert(compact.minimumExtentPages() == 104 && compact.maximumExtentPages() == 306);
  // A pool's extents hold whole alignment units, so that every tensor region
  // of an extent starts 64 KiB-aligned: the 512-byte-per-page scale regions
  // need 128 pages for four KV heads and 256 for two, BF16 one or two. The
  // size is chosen per pool between half and one and a half times the
  // 128 MiB target, leaving the fewest of its pages over, the one nearest the
  // target on a tie.
  static_assert(kExtentRegionAlignmentBytes == 64 * 1024);
  static_assert(kOracleLayout.extentAlignmentPages() == 128);
  static_assert(kOracleLayout.minimumExtentPages() == 128 &&
                kOracleLayout.maximumExtentPages() == 128);
  static_assert(kOracleLayout.extentPagesFor(127) == 0 && kOracleLayout.extentPagesFor(128) == 128 &&
                kOracleLayout.extentPagesFor(1000) == 128);
  constexpr Layout compactInt8{10, 2, 256};
  static_assert(compactInt8.bytesPerModelPage() == 332'800);
  static_assert(compactInt8.extentAlignmentPages() == 256);
  static_assert(compactInt8.minimumExtentPages() == 256 && compactInt8.maximumExtentPages() == 512);
  static_assert(compactInt8.extentPagesFor(255) == 0 && compactInt8.extentPagesFor(511) == 256);
  // Both sizes leave nothing over: 512 pages (162.5 MiB) is nearer the target
  // than 256 (81.25 MiB). With 256 pages over, only 256 leaves nothing.
  static_assert(compactInt8.extentPagesFor(10'240) == 512);
  static_assert(compactInt8.extentPagesFor(10'496) == 256);
  // 448 pages divide by 32, 56 and 64 (128 MiB, the target); 97 leaves one
  // page over 32, 48 and 96 extents, of which 48 (96 MiB) is nearest.
  static_assert(bf16.extentPagesFor(31) == 0 && bf16.extentPagesFor(448) == 64 &&
                bf16.extentPagesFor(97) == 48);
  static_assert(!Layout{16, 4, 256, static_cast<Format>(0)}.valid());
  static_assert(kBytesPerModelPage == 1'064'960);
  // One page's keys (or values) and their scales across the layers.
  static_assert(kOracleLayout.attentionLayers * kKeyDataBytesPerLayerPage == 512 * 1024);
  static_assert(kOracleLayout.attentionLayers * kKeyScaleBytesPerLayerPage == 8 * 1024);
  static_assert(kBytesPerModelPage == 2 * (512 + 8) * 1024);
  static_assert(4608 * kBytesPerModelPage == 4'907'335'680ULL); // 147456 tokens.
}

// A sliding window: windowedLayers names the layers whose rows read only the
// windowTokens keys before their own, and either is zero without the other.
// A page's regions hold the layers no window bounds first, then the windowed
// ones, each in layer order; without a window a layer's region is the layer,
// and a window changes no byte count.
void testWindows() {
  // Flint's 49 layers: all but 0, 1, 48 and every fourth from 5 windowed.
  constexpr uint64_t windowed = [] {
    uint64_t layers = (uint64_t{1} << 48) - 4;
    for (uint32_t layer = 5; layer < 48; layer += 4)
      layers &= ~(uint64_t{1} << layer);
    return layers;
  }();
  constexpr Layout flint{49, 4, 128, Format::Int8, 0, false, 512, windowed};
  static_assert(flint.valid() && std::popcount(windowed) == 35);
  static_assert(flint.bytesPerModelPage() == 1'655'808 &&
                flint.bytesPerModelPage() == Layout{49, 4, 128}.bytesPerModelPage());
  static_assert(flint.region(0) == 0 && flint.region(1) == 1 && flint.region(5) == 2 &&
                flint.region(45) == 12 && flint.region(48) == 13);
  static_assert(flint.region(2) == 14 && flint.region(3) == 15 && flint.region(6) == 17 &&
                flint.region(47) == 48);
  std::vector<bool> taken(flint.attentionLayers);
  for (uint32_t layer = 0; layer < flint.attentionLayers; ++layer) {
    const uint32_t region = flint.region(layer);
    require(region < flint.attentionLayers && !taken[region], "two layers share a region");
    taken[region] = true;
    for (uint32_t later = layer + 1; later < flint.attentionLayers; ++later)
      if (flint.windowed(later) == flint.windowed(layer))
        require(flint.region(later) > region, "a class of layers is out of layer order");
  }
  for (const Layout layout : {Layout{16, 4, 256}, Layout{12, 2, 256, Format::Int8, 128, true}})
    for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer)
      require(layout.region(layer) == layer && !layout.windowed(layer),
              "a layout without a window moved a layer");
  // Layers past the mask's 64 are never windowed and come before the
  // windowed ones.
  constexpr Layout wide{80, 2, 256, Format::Int8, 0, false, 64, 1};
  static_assert(wide.valid() && wide.region(0) == 79 && wide.region(1) == 0 &&
                wide.region(79) == 78);
  static_assert(!Layout{49, 4, 128, Format::Int8, 0, false, 512, 0}.valid() &&
                !Layout{49, 4, 128, Format::Int8, 0, false, 0, windowed}.valid() &&
                !Layout{48, 4, 128, Format::Int8, 0, false, 512, uint64_t{1} << 48}.valid());
}

void testLayouts() {
  require(splash_kv_key_element(0, 0, 1) == splash_kv_key_element(0, 0, 0) + 1,
          "a key's next dimension is not its next element");
  require(splash_kv_key_element(0, 1, 0) ==
              splash_kv_key_element(0, 0, 0) + kHeadDimension,
          "a key's next token does not follow its head dimension");
  require(splash_kv_value_element(0, 1, 0) ==
              splash_kv_value_element(0, 0, 0) + 1,
          "a value's next token is not its next element");
  require(splash_kv_value_element(0, 0, 1) ==
              splash_kv_value_element(0, 0, 0) + kPageTokens,
          "a value's next dimension does not follow the page's tokens");
  require(splash_kv_key_element(kKvHeads - 1, kPageTokens - 1,
                                kHeadDimension - 1) == kElementsPerLayerPage - 1,
          "the last key element is not the layer page's last");
  require(splash_kv_value_element(kKvHeads - 1, kPageTokens - 1,
                                  kHeadDimension - 1) == kElementsPerLayerPage - 1,
          "the last value element is not the layer page's last");
}

void testBFloat16() {
  for (float value : {0.0f, -0.0f, 1.0f, -3.5f, 1.0e-20f, 65504.0f}) {
    float roundTrip = bfloat16ToFloat(floatToBFloat16(value));
    if (value == 0.0f) {
      require(roundTrip == value, "a zero did not survive BF16");
    } else {
      require(std::abs(roundTrip - value) <= std::abs(value) / 128.0f,
              "a BF16 round trip moved a value by more than 1/128 of it");
    }
  }
  require(std::isinf(bfloat16ToFloat(
              floatToBFloat16(std::numeric_limits<float>::infinity()))),
          "an infinity did not survive BF16");
  require(std::isnan(bfloat16ToFloat(
              floatToBFloat16(std::numeric_limits<float>::quiet_NaN()))),
          "a NaN did not survive BF16");
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
        float keyScale = page->keyScales[splash_kv_scale_element(head, token)];
        float valueScale = page->valueScales[splash_kv_scale_element(head, token)];
        require(std::abs(decodedKeys[logical] - keys[logical]) <=
                    keyScale * 0.51f + 1.0e-7f,
                "a key is off by more than half its scale");
        require(std::abs(decodedValues[logical] - values[logical]) <=
                    valueScale * 0.51f + 1.0e-7f,
                "a value is off by more than half its scale");
        require(decodedKeys[logical] ==
                    dequantizeKey(*page, head, token, dimension),
                "the page's keys and one key's dequantization differ");
        require(decodedValues[logical] ==
                    dequantizeValue(*page, head, token, dimension),
                "the page's values and one value's dequantization differ");
      }
    }
  }
  for (uint32_t token = validTokens; token < kPageTokens; ++token) {
    uint64_t logical = logicalIndex(token, 0, 0);
    require(decodedKeys[logical] == 0.0f && decodedValues[logical] == 0.0f,
            "a row past the valid tokens is not zero");
  }
}

void testZeroAndInvalidInputs() {
  constexpr uint32_t tokens = 1;
  std::vector<float> values(uint64_t{tokens} * kKvHeads * kHeadDimension);
  auto page = std::make_unique<Q8LayerPage>();
  quantizeLayerPage(values, values, tokens, *page);
  require(std::all_of(page->keys.begin(), page->keys.end(),
                      [](int8_t value) { return value == 0; }) &&
              std::all_of(page->values.begin(), page->values.end(),
                          [](int8_t value) { return value == 0; }),
          "a zero page did not quantize to zeros");
  values[0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { quantizeLayerPage(values, values, tokens, *page); },
          "logical keys contains a non-finite value",
          "a page with a non-finite value was quantized");
}

} // namespace

int main() {
  try {
    testByteAccounting();
    testWindows();
    testLayouts();
    testBFloat16();
    testQuantization(kPageTokens);
    testQuantization(17);
    testZeroAndInvalidInputs();
    std::cout << "q8_paged_kv_test: ok\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "q8_paged_kv_test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
