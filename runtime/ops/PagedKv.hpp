#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

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

// A page whose content goes to another page.
struct PageCopy final {
  uint32_t from = 0;
  uint32_t to = 0;
};

// The memory of the engine's page pool: extents of extentPages() pages,
// pageCount() a whole number of them. Only KvPool allocates and releases
// them. Implementations provide Metal storage or deterministic test storage.
class ExtentStorage {
public:
  virtual ~ExtentStorage() = default;
  [[nodiscard]] virtual uint32_t pageCount() const noexcept = 0;
  [[nodiscard]] virtual uint64_t bytesPerPage() const noexcept = 0;
  [[nodiscard]] virtual uint32_t extentPages() const noexcept = 0;
  // Allocates extent `extent`, which is not allocated: granted, or the
  // refusal's cause with nothing allocated. std::logic_error for an
  // allocated extent.
  [[nodiscard]] virtual metal::AllocationResult allocateExtent(uint32_t extent) = 0;
  // Releases allocated extent `extent` at once. std::logic_error, changing
  // nothing, for an unallocated extent or while a command is in flight (a
  // command reaches extents through its page tables without retaining them).
  virtual void releaseExtent(uint32_t extent) = 0;
  // Copies each page's content onto its destination, both in allocated
  // extents: how the pool moves the pages of an extent it empties.
  // std::logic_error, copying nothing, for a page of an unallocated extent
  // or while a command is in flight (it reaches both pages through its
  // tables and may still write the source).
  virtual void copyPages(std::span<const PageCopy> copies) = 0;
};

// Shared cache format and execution limits; model dimensions live in Layout.
inline constexpr uint32_t kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
inline constexpr uint32_t kMaximumLogicalTokens =
    SPLASH_MAXIMUM_CONTEXT_TOKENS;
inline constexpr uint32_t kMaximumPhysicalTokens =
    SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS;
inline constexpr int32_t kQuantizedMinimum = -127;
inline constexpr int32_t kQuantizedMaximum = 127;
// Every tensor region of an extent starts on this boundary. The attention
// kernels were tuned on it, and regions aligned to less cost Apple10's 35B
// verify kernel a fixed ~25 µs per dispatch.
inline constexpr uint64_t kExtentRegionAlignmentBytes = 64 * 1024;
// The size a pool aims its extents at (Layout::extentPagesFor stays within
// half and one and a half times it): each extent costs the serving loop an
// allocation and a release, and an extent returns memory only once all of
// its pages are free.
inline constexpr uint64_t kAllocationExtentTargetBytes = 128ull * 1024 * 1024;

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

// Pages whose bytes of one tensor fill whole region alignment units.
[[nodiscard]] constexpr uint64_t pagesForAlignedRegion(
    uint64_t bytesPerPage) noexcept {
  return bytesPerPage
             ? kExtentRegionAlignmentBytes /
                   gcd(kExtentRegionAlignmentBytes, bytesPerPage)
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
  // Qwen3.8-Flash-Next's QSA: each page also holds, per layer, its tokens'
  // indexer keys of this many values (bf16 [token][dimension]) and the
  // pooled keys of its blocks of kIndexBlockTokens tokens (fp32 [block]
  // [dimension]), in index regions after every layer's K/V region
  // (PageStorage::indexPlacement).
  uint32_t indexDimension = 0;
  static constexpr uint32_t kIndexBlockTokens = 4;

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
  [[nodiscard]] constexpr uint64_t indexKeyBytesPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} * indexDimension * 2;
  }
  [[nodiscard]] constexpr uint64_t indexPooledBytesPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} / kIndexBlockTokens * indexDimension * sizeof(float);
  }
  // The K/V bytes of one layer's page; the index bytes are apart.
  [[nodiscard]] constexpr uint64_t bytesPerLayerPage() const noexcept {
    return 2 * (dataBytesPerLayerPage() + scaleBytesPerLayerPage());
  }
  [[nodiscard]] constexpr uint64_t indexBytesPerLayerPage() const noexcept {
    return indexKeyBytesPerLayerPage() + indexPooledBytesPerLayerPage();
  }
  [[nodiscard]] constexpr uint64_t bytesPerModelPage() const noexcept {
    return uint64_t{attentionLayers} * (bytesPerLayerPage() + indexBytesPerLayerPage());
  }

  // An extent holds a whole number of these pages, so that every tensor
  // region starts 64 KiB-aligned. The INT8 scales are the tightest
  // constraint: 4 heads require 128 pages and 2 heads require 256. BF16
  // needs only 1 or 2 pages. This is allocation geometry only; prefix
  // matching remains Page32 in both cases.
  [[nodiscard]] constexpr uint32_t extentAlignmentPages() const noexcept {
    uint64_t pages = format == Format::BFloat16
                         ? detail::pagesForAlignedRegion(dataBytesPerLayerPage())
                         : detail::lcm(detail::pagesForAlignedRegion(dataBytesPerLayerPage()),
                                       detail::pagesForAlignedRegion(scaleBytesPerLayerPage()));
    if (indexDimension)
      pages = detail::lcm(pages, detail::lcm(detail::pagesForAlignedRegion(indexKeyBytesPerLayerPage()),
                                             detail::pagesForAlignedRegion(indexPooledBytesPerLayerPage())));
    return static_cast<uint32_t>(pages);
  }

  // Extents hold whole alignment units, between half and one and a half
  // times the allocation target; a unit larger than that is an extent on its
  // own. These are the smallest and the largest sizes.
  [[nodiscard]] constexpr uint32_t minimumExtentPages() const noexcept {
    const uint64_t unit = uint64_t{extentAlignmentPages()} * bytesPerModelPage();
    if (!unit)
      return 0;
    const uint64_t units =
        std::max<uint64_t>(1, (kAllocationExtentTargetBytes / 2 + unit - 1) / unit);
    return static_cast<uint32_t>(units * extentAlignmentPages());
  }
  [[nodiscard]] constexpr uint32_t maximumExtentPages() const noexcept {
    const uint64_t unit = uint64_t{extentAlignmentPages()} * bytesPerModelPage();
    if (!unit)
      return 0;
    const uint64_t units = kAllocationExtentTargetBytes * 3 / 2 / unit;
    return std::max(static_cast<uint32_t>(units * extentAlignmentPages()),
                    minimumExtentPages());
  }

  // The extent size of a pool that holds `pages` pages: all of a pool's
  // extents hold the same number of pages, because kernels find a layer's
  // region from that number. Of the sizes above that fit the pool, the one
  // that leaves the fewest pages over, and the one nearest the target on a
  // tie. Zero when the pool holds fewer pages than the smallest size.
  [[nodiscard]] constexpr uint32_t extentPagesFor(uint64_t pages) const noexcept {
    const uint32_t step = extentAlignmentPages();
    const uint64_t smallest = minimumExtentPages();
    if (!smallest || pages < smallest)
      return 0;
    const uint64_t largest = std::min<uint64_t>(maximumExtentPages(), pages);
    uint64_t best = 0, bestLeft = 0, bestDistance = 0;
    for (uint64_t extent = smallest; extent <= largest; extent += step) {
      const uint64_t bytes = extent * bytesPerModelPage();
      const uint64_t distance = bytes > kAllocationExtentTargetBytes
                                    ? bytes - kAllocationExtentTargetBytes
                                    : kAllocationExtentTargetBytes - bytes;
      if (!best || pages % extent < bestLeft ||
          (pages % extent == bestLeft && distance < bestDistance)) {
        best = extent;
        bestLeft = pages % extent;
        bestDistance = distance;
      }
    }
    return static_cast<uint32_t>(best);
  }

  bool operator==(const Layout &) const = default;
};

} // namespace splash::kv
