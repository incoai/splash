#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/MetalBackend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace splash::kv {

// Selected once for a runtime and its entire page pool. Weight storage is
// independent of the KV format; requests never change it while serving.
enum class Format : uint32_t { Int8 = 1, BFloat16 = 2 };

[[nodiscard]] constexpr bool validFormat(Format format) noexcept {
  return format == Format::Int8 || format == Format::BFloat16;
}

[[nodiscard]] constexpr std::string_view formatName(Format format) noexcept {
  switch (format) {
  case Format::Int8: return "int8";
  case Format::BFloat16: return "bf16";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view storageFormatName(Format format) noexcept {
  switch (format) {
  case Format::Int8:
    return "q8s8_f32_scale_per_token_head_k_token_major_v_dimension_major";
  case Format::BFloat16:
    return "bf16_k_token_major_v_dimension_major";
  }
  return "invalid";
}

// Physical backing for the engine's page pool. Implementations provide Metal
// storage or deterministic test storage.
class Backing {
public:
  virtual ~Backing() = default;
  [[nodiscard]] virtual uint32_t pageCount() const noexcept = 0;
  [[nodiscard]] virtual uint64_t bytesPerPage() const noexcept = 0;
  [[nodiscard]] virtual bool isResident(uint32_t page) const = 0;
  [[nodiscard]] virtual metal::AllocationResult ensureResident(uint32_t page) = 0;
  [[nodiscard]] virtual bool releaseBackingForPage(uint32_t page) = 0;
  [[nodiscard]] virtual uint32_t extentFirstPage(uint32_t page) const = 0;
  [[nodiscard]] virtual uint32_t extentPageCount(uint32_t page) const = 0;
  // Physical release is asynchronous and paced: while a previous release is
  // still being torn down by the kernel, callers keep the next empty extent
  // resident instead of queueing more unmap work. Test backings are always
  // ready; awaitRelease() blocks only at startup and shutdown.
  [[nodiscard]] virtual bool releaseReady() const noexcept { return true; }
  virtual void awaitRelease() {}
};

struct LayerStorage final {
  metal::MetalBuffer keyData;
  metal::MetalBuffer keyScales;
  metal::MetalBuffer valueData;
  metal::MetalBuffer valueScales;
  Format format = Format::Int8;
};

// Shared cache format and execution limits; model dimensions live in Layout.
inline constexpr uint32_t kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
inline constexpr uint32_t kMaximumLogicalTokens =
    SPLASH_MAXIMUM_CONTEXT_TOKENS;
inline constexpr uint32_t kMaximumPhysicalTokens =
    SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS;
inline constexpr int32_t kQuantizedMinimum = -127;
inline constexpr int32_t kQuantizedMaximum = 127;
inline constexpr uint64_t kSparseMappingAlignmentBytes = 64 * 1024;
inline constexpr uint64_t kAllocationExtentTargetBytes =
    SPLASH_ALLOCATION_EXTENT_TARGET_BYTES;

struct StorageByteCounts final {
  uint64_t keyData = 0;
  uint64_t keyScales = 0;
  uint64_t valueData = 0;
  uint64_t valueScales = 0;
  uint64_t total = 0;
};

namespace detail {

[[nodiscard]] constexpr uint64_t gcd(uint64_t left, uint64_t right) noexcept {
  while (right) {
    const uint64_t remainder = left % right;
    left = right;
    right = remainder;
  }
  return left;
}

[[nodiscard]] constexpr uint64_t lcm(uint64_t left, uint64_t right) noexcept {
  return left && right ? left / gcd(left, right) * right : 0;
}

[[nodiscard]] constexpr uint64_t pagesForAlignedMapping(
    uint64_t bytesPerPage) noexcept {
  return bytesPerPage
             ? kSparseMappingAlignmentBytes /
                   gcd(kSparseMappingAlignmentBytes, bytesPerPage)
             : 0;
}

} // namespace detail

// Physical KV geometry: Page32, either BF16 or per-(token, head) symmetric INT8.
// Layer and head counts vary by target.
struct Layout final {
  uint32_t attentionLayers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  Format format = Format::Int8;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return attentionLayers && kvHeads && headDimension && validFormat(format);
  }
  [[nodiscard]] constexpr uint32_t elementsPerScale() const noexcept {
    return format == Format::Int8 ? headDimension : 0;
  }
  [[nodiscard]] constexpr uint64_t elementsPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} * kvHeads * headDimension;
  }
  [[nodiscard]] constexpr uint64_t scalesPerTensorLayerPage() const noexcept {
    return format == Format::Int8 ? uint64_t{kPageTokens} * kvHeads : 0;
  }
  // Keys and values share one data and one scale geometry per layer page.
  [[nodiscard]] constexpr uint64_t dataBytesPerLayerPage() const noexcept {
    return elementsPerLayerPage() * (format == Format::Int8 ? 1 : 2);
  }
  [[nodiscard]] constexpr uint64_t scaleBytesPerLayerPage() const noexcept {
    return scalesPerTensorLayerPage() * sizeof(float);
  }
  [[nodiscard]] constexpr uint64_t bytesPerLayerPage() const noexcept {
    return 2 * (dataBytesPerLayerPage() + scaleBytesPerLayerPage());
  }
  [[nodiscard]] constexpr uint64_t bytesPerModelPage() const noexcept {
    return uint64_t{attentionLayers} * bytesPerLayerPage();
  }

  // Metal sparse mappings must begin and end on 64-KiB tile boundaries. The
  // INT8 scale buffers are the tightest constraint: 4 heads require 128
  // pages and 2 heads require 256. BF16 needs only 1 or 2 pages. This is physical
  // allocation geometry; prefix matching remains Page32 in both cases.
  [[nodiscard]] constexpr uint32_t sparseMappingBatchPages() const noexcept {
    if (format == Format::BFloat16)
      return static_cast<uint32_t>(
          detail::pagesForAlignedMapping(dataBytesPerLayerPage()));
    return static_cast<uint32_t>(detail::lcm(
        detail::pagesForAlignedMapping(dataBytesPerLayerPage()),
        detail::pagesForAlignedMapping(scaleBytesPerLayerPage())));
  }

  [[nodiscard]] constexpr uint64_t sparseMappingBatchBytes() const noexcept {
    return uint64_t{sparseMappingBatchPages()} * bytesPerModelPage();
  }

  [[nodiscard]] constexpr uint32_t backingExtentPages() const noexcept {
    const uint64_t unit = sparseMappingBatchBytes();
    if (!unit)
      return 0;
    return static_cast<uint32_t>(
        ((kAllocationExtentTargetBytes + unit - 1) / unit) *
        sparseMappingBatchPages());
  }

  [[nodiscard]] constexpr StorageByteCounts
  storageByteCounts(uint64_t pageCount) const noexcept {
    const uint64_t data = pageCount * attentionLayers * dataBytesPerLayerPage();
    const uint64_t scale =
        pageCount * attentionLayers * scaleBytesPerLayerPage();
    return {data, scale, data, scale, pageCount * bytesPerModelPage()};
  }

  bool operator==(const Layout &) const = default;
};

enum class Quantization : uint32_t { SymmetricInt8 = 1, BFloat16 = 2 };
enum class ScaleType : uint32_t { None = 0, Float32 = 2 };
enum class KeyLayout : uint32_t { TokenMajor = 1 };
enum class ValueLayout : uint32_t { DimensionMajor = 1 };

// Stable metadata for rejecting incompatible cached blocks before any block is
// read. modelArtifactSha256 is the digest of the exact packed target artifact
// set; geometry/layout fields remain explicit so format changes cannot alias.
struct alignas(8) LayoutGuard final {
  uint32_t quantization = 0;
  uint32_t scaleType = 0;
  uint32_t keyLayout = 0;
  uint32_t valueLayout = 0;
  uint32_t pageTokens = 0;
  uint32_t elementsPerScale = 0;
  uint32_t attentionLayers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  int32_t quantizedMinimum = 0;
  int32_t quantizedMaximum = 0;
  uint64_t bytesPerLayerPage = 0;
  uint64_t bytesPerModelPage = 0;
  std::array<uint8_t, 32> modelArtifactSha256{};

  [[nodiscard]] Format format() const noexcept {
    switch (quantization) {
    case uint32_t(Quantization::SymmetricInt8): return Format::Int8;
    case uint32_t(Quantization::BFloat16): return Format::BFloat16;
    default: return static_cast<Format>(0);
    }
  }
};

static_assert(sizeof(LayoutGuard) == 96);
static_assert(std::is_standard_layout_v<LayoutGuard>);
static_assert(std::is_trivially_copyable_v<LayoutGuard>);

[[nodiscard]] inline LayoutGuard makeLayoutGuard(
    Layout layout,
    const std::array<uint8_t, 32> &modelArtifactSha256) {
  LayoutGuard result;
  const bool quantized = layout.format == Format::Int8;
  result.quantization = uint32_t(quantized ? Quantization::SymmetricInt8
                                         : Quantization::BFloat16);
  result.scaleType = uint32_t(quantized ? ScaleType::Float32 : ScaleType::None);
  result.keyLayout = uint32_t(KeyLayout::TokenMajor);
  result.valueLayout = uint32_t(ValueLayout::DimensionMajor);
  result.pageTokens = kPageTokens;
  result.elementsPerScale = layout.elementsPerScale();
  result.attentionLayers = layout.attentionLayers;
  result.kvHeads = layout.kvHeads;
  result.headDimension = layout.headDimension;
  result.quantizedMinimum = quantized ? kQuantizedMinimum : 0;
  result.quantizedMaximum = quantized ? kQuantizedMaximum : 0;
  result.bytesPerLayerPage = layout.bytesPerLayerPage();
  result.bytesPerModelPage = layout.bytesPerModelPage();
  result.modelArtifactSha256 = modelArtifactSha256;
  return result;
}

} // namespace splash::kv
