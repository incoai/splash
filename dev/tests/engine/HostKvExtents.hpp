#pragma once

#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::test {

// The extents of a KV pool in CPU-visible memory, for the kernel tests and
// benchmarks that fill and read pages on the host. Page p sits in extent
// p / extentPages at index p % extentPages, every layer's region where
// splash_kv_offset places the layer's kv::Layout::region and the index plane
// after them, as in production extents (kv::PageStorage); an extent of Slim
// pages holds the regions of the layers no window bounds and the index
// plane, and one of Index pages only the index plane. Kernels reach a page
// only through its entries. The constructors size each extent exactly, so
// an access past one fails under shader validation; contiguous() extents
// lie extentStride() apart in one region, and an access past one runs into
// its neighbour unchecked.
class HostKvExtents final {
public:
  struct Extent final {
    std::byte *contents = nullptr;
    uint64_t gpuAddress = 0;
    kv::PageForm form = kv::PageForm::Full;
  };

  [[nodiscard]] static uint64_t extentBytes(kv::Layout layout, uint32_t extentPages,
                                            kv::PageForm form = kv::PageForm::Full) noexcept {
    return uint64_t{extentPages} * layout.bytesPerModelPage(form);
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

  // Shared buffers of the backend, zero-filled and addressed
  // (metal::BufferReach), as the pool's extents: resident for every command
  // buffer. `extents` Full extents, or an extent of each form of `forms`.
  HostKvExtents(metal::MetalBackend &backend, kv::Layout layout,
                uint32_t extentPages, uint32_t extents)
      : HostKvExtents(backend, layout, extentPages, std::vector(extents, kv::PageForm::Full)) {}
  HostKvExtents(metal::MetalBackend &backend, kv::Layout layout,
                uint32_t extentPages, std::span<const kv::PageForm> forms)
      : layout_(layout), extentPages_(extentPages) {
    for (const kv::PageForm form : forms) {
      metal::MetalBuffer buffer = backend.allocateBuffer(
          extentBytes(layout, extentPages, form), metal::BufferStorage::Shared,
          "host-kv-extent", metal::BufferReach::Addressed);
      auto *contents = static_cast<std::byte *>(buffer.contents());
      std::fill_n(contents, buffer.sizeBytes(), std::byte{0});
      extents_.push_back({contents, buffer.gpuAddress(), form});
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
    const Extent &extent = extents_.at(index);
    return {extent.contents, extentBytes(layout_, extentPages_, extent.form)};
  }
  [[nodiscard]] SplashKvLayer layer(uint32_t index) const {
    if (index >= layout_.attentionLayers)
      throw std::out_of_range("host KV layer is outside the layout");
    return splash_kv_layer(extentPages_, dataBytes(), scaleBytes(), layout_.region(index));
  }
  // Writes the entries of `pages` to the start of a CPU-visible table, as
  // kv::PageStorage::writeEntries: SPLASH_KV_PAGE_ABSENT for an Index page.
  void writeTable(std::span<const uint32_t> pages, void *table) const {
    auto *entries = static_cast<SplashKvPage *>(table);
    for (size_t index = 0; index < pages.size(); ++index) {
      const Extent &placed = extent(pages[index]);
      entries[index] = placed.form == kv::PageForm::Index
                           ? SPLASH_KV_PAGE_ABSENT
                           : splash_kv_page_entry(placed.gpuAddress, pages[index] % extentPages_);
    }
  }
  // The same for an index table (kv::PageStorage::writeIndexEntries): the
  // entries that reach the pages' block index, in their extent's index
  // plane. std::logic_error without an index, and for a Full extent whose
  // plane leaves no room for the page index, which only extents of whole
  // alignment units (aligned()) rule out.
  void writeIndexTable(std::span<const uint32_t> pages, void *table) const {
    if (!layout_.indexDimension)
      throw std::logic_error("host KV pages of a layout without an index have no index entries");
    auto *entries = static_cast<SplashKvPage *>(table);
    for (size_t index = 0; index < pages.size(); ++index) {
      const Extent &placed = extent(pages[index]);
      const uint64_t plane = placed.gpuAddress + indexPlaneOffset(placed.form);
      if (plane & SPLASH_KV_PAGE_INDEX_MASK)
        throw std::logic_error("host KV index plane leaves no room for the page index");
      entries[index] = splash_kv_page_entry(plane, pages[index] % extentPages_);
    }
  }
  // Where a layer's block index sits in every extent's index plane, as
  // kv::PageStorage::indexOffset places it.
  [[nodiscard]] uint64_t indexOffset(uint32_t layer) const {
    if (!layout_.indexDimension || layer >= layout_.attentionLayers)
      throw std::out_of_range("host KV layer has no index region");
    return uint64_t{extentPages_} * layer * layout_.indexBytesPerLayerPage();
  }
  // A page's bytes as kv::PageStorage::layerSpans and indexSpans give them:
  // one layer's tensors, by SPLASH_KV_KEYS to SPLASH_KV_VALUE_SCALES (empty
  // scales for BF16), and each layer's block index.
  [[nodiscard]] std::array<std::span<std::byte>, 4> layerSpans(uint32_t page, uint32_t layer) const {
    if (layer >= layout_.attentionLayers)
      throw std::out_of_range("host KV layer is outside the layout");
    std::array<std::span<std::byte>, 4> result;
    for (uint32_t tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES; ++tensor)
      result[tensor] = {slab<std::byte>(layer, tensor, page), splash_kv_page_bytes(dataBytes(), scaleBytes(), tensor)};
    return result;
  }
  [[nodiscard]] std::vector<std::span<std::byte>> indexSpans(uint32_t page) const {
    std::vector<std::span<std::byte>> result;
    const Extent &placed = extent(page);
    const uint64_t bytes = layout_.indexBytesPerLayerPage();
    for (uint32_t layer = 0; layout_.indexDimension && layer < layout_.attentionLayers; ++layer)
      result.emplace_back(placed.contents + indexPlaneOffset(placed.form) + indexOffset(layer) +
                              page % extentPages_ * bytes,
                          bytes);
    return result;
  }
  // Where one value of a KV head sits in a page's slab of values, in the
  // layout's value order (kv::Layout::valuesTokenMajor).
  [[nodiscard]] uint64_t valueElement(uint32_t head, uint32_t token, uint32_t dimension) const noexcept {
    return layout_.valuesTokenMajor ? splash_kv_key_element(head, token, dimension, layout_.headDimension)
                                    : splash_kv_value_element(head, token, dimension, layout_.headDimension);
  }
  // The host address of one tensor of one page in one layer whose region
  // the page's form holds (kv::Layout::regions).
  template <typename T>
  [[nodiscard]] T *slab(uint32_t layer, uint32_t tensor, uint32_t page) const {
    const Extent &placed = extent(page);
    if (placed.form == kv::PageForm::Index)
      throw std::logic_error("host KV page holds only its index");
    if (layout_.region(layer) >= layout_.regions(placed.form))
      throw std::logic_error("host KV page holds no windowed layer's K/V");
    return reinterpret_cast<T *>(placed.contents + offset(layer, tensor, page % extentPages_));
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
      if (extent.form == kv::PageForm::Index && !layout_.spillable())
        throw std::invalid_argument("host KV pages of a layout that cannot spill are all Full");
      if (extent.form == kv::PageForm::Slim && !layout_.slimmable())
        throw std::invalid_argument("host KV pages of a layout without a window keep every layer");
    }
  }
  [[nodiscard]] const Extent &extent(uint32_t page) const {
    if (page >= pageCount())
      throw std::out_of_range("host KV page is outside the extents");
    return extents_[page / extentPages_];
  }
  // Where an extent's index plane starts: after the regions its form holds.
  [[nodiscard]] uint64_t indexPlaneOffset(kv::PageForm form) const noexcept {
    return uint64_t{extentPages_} * layout_.regions(form) * layout_.bytesPerLayerPage();
  }
  [[nodiscard]] uint32_t dataBytes() const noexcept {
    return static_cast<uint32_t>(layout_.dataBytesPerLayerPage());
  }
  [[nodiscard]] uint32_t scaleBytes() const noexcept {
    return static_cast<uint32_t>(layout_.scaleBytesPerLayerPage());
  }
  [[nodiscard]] uint64_t offset(uint32_t layer, uint32_t tensor, uint32_t index) const {
    return splash_kv_offset(extentPages_, dataBytes(), scaleBytes(), layout_.region(layer), tensor, index);
  }

  kv::Layout layout_;
  uint32_t extentPages_ = 0;
  std::vector<Extent> extents_;
  std::vector<metal::MetalBuffer> buffers_;
};

} // namespace splash::test
