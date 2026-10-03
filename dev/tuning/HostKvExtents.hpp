#pragma once

#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::ops::tuning {

// The extents of a KV pool in CPU-visible memory, for the attention tuner,
// kernel tests and benchmarks that fill and read pages on the host. Page p
// sits in extent p / extentPages at index p % extentPages, and every
// layer's region where splash_kv_offset places it, as in production
// extents; kernels reach a page only through its entry. The constructors
// size each extent exactly, so an access past one fails under shader
// validation; contiguous() extents lie extentStride() apart in one region,
// and an access past one runs into its neighbour unchecked.
class HostKvExtents final {
public:
  struct Extent final {
    std::byte *contents = nullptr;
    uint64_t gpuAddress = 0;
  };

  [[nodiscard]] static uint64_t extentBytes(kv::Layout layout,
                                            uint32_t extentPages) noexcept {
    return uint64_t{extentPages} * layout.bytesPerModelPage();
  }
  // The distance between extents laid out one after another in a buffer:
  // every extent starts 16 KiB-aligned, as page entries need.
  [[nodiscard]] static uint64_t extentStride(kv::Layout layout,
                                             uint32_t extentPages) noexcept {
    constexpr uint64_t alignment = uint64_t{SPLASH_KV_PAGE_INDEX_MASK} + 1;
    return (extentBytes(layout, extentPages) + alignment - 1) / alignment * alignment;
  }

  // Extents of a test pool of at least `pages` pages: three or more, of a
  // page count that is not a power of two, so that a page table crosses
  // extents and an index computed with a shift or a mask would fail.
  struct Geometry final {
    uint32_t extentPages = 0;
    uint32_t extents = 0;
  };
  [[nodiscard]] static Geometry spread(uint32_t pages) noexcept {
    uint32_t extentPages = std::clamp((pages + 2) / 3, 3U, SPLASH_KV_PAGE_INDEX_MASK);
    if (std::has_single_bit(extentPages)) ++extentPages;
    return {extentPages, std::max(3U, (pages + extentPages - 1) / extentPages)};
  }
  // Extents of a pool of at least `pages` pages for measurements: whole
  // alignment units, so that every region starts 64 KiB-aligned as in
  // production, and as few extents as the page index allows.
  [[nodiscard]] static Geometry aligned(kv::Layout layout, uint32_t pages) noexcept {
    const uint32_t unit = layout.extentAlignmentPages();
    const uint32_t largest = SPLASH_KV_PAGE_INDEX_MASK / unit * unit;
    const uint32_t extentPages = std::min(largest, (pages + unit - 1) / unit * unit);
    return {extentPages, (pages + extentPages - 1) / extentPages};
  }

  // Shared buffers of the backend, zero-filled, resident for every command
  // like every backend buffer.
  HostKvExtents(metal::MetalBackend &backend, kv::Layout layout,
                uint32_t extentPages, uint32_t extents)
      : layout_(layout), extentPages_(extentPages) {
    for (uint32_t index = 0; index < extents; ++index) {
      metal::MetalBuffer buffer = backend.allocateBuffer(
          extentBytes(layout, extentPages), metal::BufferStorage::Shared,
          "host-kv-extent");
      auto *contents = static_cast<std::byte *>(buffer.contents());
      std::fill_n(contents, buffer.sizeBytes(), std::byte{0});
      extents_.push_back({contents, buffer.gpuAddress()});
      buffers_.push_back(std::move(buffer));
    }
    validate();
  }

  // Extents the caller allocated, sized by extentBytes(), and makes resident
  // for its own commands.
  HostKvExtents(kv::Layout layout, uint32_t extentPages,
                std::vector<Extent> extents)
      : layout_(layout), extentPages_(extentPages), extents_(std::move(extents)) {
    validate();
  }

  // `extents` extents one after another in a region the caller allocated,
  // extentStride() bytes each, and keeps resident for its commands.
  [[nodiscard]] static HostKvExtents contiguous(kv::Layout layout, uint32_t extentPages,
                                                uint32_t extents, std::byte *contents,
                                                uint64_t gpuAddress) {
    const uint64_t stride = extentStride(layout, extentPages);
    std::vector<Extent> placed;
    for (uint32_t index = 0; index < extents; ++index)
      placed.push_back({contents + index * stride, gpuAddress + index * stride});
    return {layout, extentPages, std::move(placed)};
  }

  [[nodiscard]] uint32_t extentPages() const noexcept { return extentPages_; }
  [[nodiscard]] uint32_t extentCount() const noexcept {
    return static_cast<uint32_t>(extents_.size());
  }
  [[nodiscard]] uint32_t pageCount() const noexcept {
    return extentPages_ * extentCount();
  }
  // Every byte of one extent.
  [[nodiscard]] std::span<std::byte> bytes(uint32_t index) const {
    return {extents_.at(index).contents, extentBytes(layout_, extentPages_)};
  }
  [[nodiscard]] SplashKvLayer layer(uint32_t index) const {
    if (index >= layout_.attentionLayers)
      throw std::out_of_range("host KV layer is outside the layout");
    return splash_kv_layer(extentPages_, dataBytes(), scaleBytes(), index);
  }
  // Writes the entries of `pages` to the start of a CPU-visible table.
  void writeTable(std::span<const uint32_t> pages, void *table) const {
    auto *entries = static_cast<SplashKvPage *>(table);
    for (size_t index = 0; index < pages.size(); ++index)
      entries[index] = entry(pages[index]);
  }
  // The host address of one tensor of one page in one layer.
  template <typename T>
  [[nodiscard]] T *slab(uint32_t layer, uint32_t tensor, uint32_t page) const {
    return reinterpret_cast<T *>(extent(page).contents +
                                 offset(layer, tensor, page % extentPages_));
  }
  // `count` distinct pages of a pool for a request's table: the first and
  // last page of every extent, then the others, each group shuffled.
  [[nodiscard]] static std::vector<uint32_t>
  mixedPages(Geometry geometry, uint32_t count, uint32_t seed) {
    std::vector<uint32_t> boundaries, interior;
    for (uint32_t page = 0; page < geometry.extentPages * geometry.extents; ++page) {
      const uint32_t index = page % geometry.extentPages;
      (index == 0 || index + 1 == geometry.extentPages ? boundaries : interior)
          .push_back(page);
    }
    std::mt19937 random(seed);
    std::shuffle(boundaries.begin(), boundaries.end(), random);
    std::shuffle(interior.begin(), interior.end(), random);
    boundaries.insert(boundaries.end(), interior.begin(), interior.end());
    if (count > boundaries.size())
      throw std::out_of_range("host KV pool holds fewer pages than requested");
    boundaries.resize(count);
    return boundaries;
  }

private:
  void validate() const {
    if (!layout_.valid() || !extentPages_ || extentPages_ > SPLASH_KV_PAGE_INDEX_MASK ||
        extents_.empty())
      throw std::invalid_argument("invalid host KV extent geometry");
    for (const Extent &extent : extents_) {
      if (!extent.contents || !extent.gpuAddress ||
          extent.gpuAddress & SPLASH_KV_PAGE_INDEX_MASK)
        throw std::invalid_argument("host KV extent is not visible or not 16 KiB-aligned");
    }
  }
  [[nodiscard]] const Extent &extent(uint32_t page) const {
    if (page >= pageCount())
      throw std::out_of_range("host KV page is outside the extents");
    return extents_[page / extentPages_];
  }
  [[nodiscard]] SplashKvPage entry(uint32_t page) const {
    return splash_kv_page_entry(extent(page).gpuAddress, page % extentPages_);
  }
  [[nodiscard]] uint32_t dataBytes() const noexcept {
    return static_cast<uint32_t>(layout_.dataBytesPerLayerPage());
  }
  [[nodiscard]] uint32_t scaleBytes() const noexcept {
    return static_cast<uint32_t>(layout_.scaleBytesPerLayerPage());
  }
  [[nodiscard]] uint64_t offset(uint32_t layer, uint32_t tensor, uint32_t index) const {
    return splash_kv_offset(extentPages_, dataBytes(), scaleBytes(), layer, tensor, index);
  }

  kv::Layout layout_;
  uint32_t extentPages_ = 0;
  std::vector<Extent> extents_;
  std::vector<metal::MetalBuffer> buffers_;
};

} // namespace splash::ops::tuning
