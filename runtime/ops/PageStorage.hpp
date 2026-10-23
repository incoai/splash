#pragma once

#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::kv {

// Storage for paged KV in extents: ordinary shared Metal buffers of
// extentPages pages each, which KvPool alone allocates and releases: its
// runway when it is built, then an extent when it needs one of its pages.
// An extent whose pages are all free stays allocated until a reclaim pass
// between commands releases it (Engine::reclaimMemory); the cleanup after
// startup warmup keeps one. Inside an extent every attention layer has a
// region that holds the keys of all its pages, then their key scales, values
// and value scales (abi/KvExtent.h), in the order Layout::region gives them:
// a window's layers after the others. With a block index, the index plane
// follows them: per layer a region of every page's pooled keys
// (indexOffset). An extent holds pages of one form: a Full extent all of
// that, a Slim extent the regions of the layers no window bounds and the
// index plane, an Index extent only the index plane (kv::PageForm). A
// layer's region sits at one offset in every extent that holds it, so its
// SplashKvLayer serves a Full page and a Slim one alike. Kernels reach a
// page through its entries in a request's GPU tables, so extents are
// allocated metal::BufferReach::Addressed: the command queue's residency set
// keeps them resident for every command buffer. The host reaches the same
// bytes through the page's spans(). Active requests may overwrite slots at
// or beyond their logical commit index. Cached KV blocks reference only
// fully committed pages, which are immutable while shared.
class PageStorage final : public ExtentStorage {
public:
  // pageCount must be a whole number of extents of extentPages pages, a
  // whole number of the layout's alignment units (Layout::extentPagesFor).
  PageStorage(metal::MetalBackend &backend,
              metal::AllocationAdmission admitAllocation,
              Layout layout,
              uint32_t pageCount,
              uint32_t extentPages);

  PageStorage(const PageStorage &) = delete;
  PageStorage &operator=(const PageStorage &) = delete;

  [[nodiscard]] uint32_t pageCount() const noexcept override {
    return pageCount_;
  }
  [[nodiscard]] uint64_t bytesPerPage(PageForm form = PageForm::Full) const noexcept override {
    return layout_.bytesPerModelPage(form);
  }
  [[nodiscard]] Layout layout() const noexcept { return layout_; }
  [[nodiscard]] uint32_t extentPages() const noexcept override {
    return extentPages_;
  }
  [[nodiscard]] uint64_t extentBytes(PageForm form = PageForm::Full) const noexcept {
    return uint64_t{extentPages_} * layout_.bytesPerModelPage(form);
  }
  // The bytes of the extents that hold memory now: a measurement of the
  // buffers, which KvPool's record of its extents must match.
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;
  [[nodiscard]] bool isAllocated(uint32_t page) const;
  // std::out_of_range for an extent past the pool; std::invalid_argument for
  // Index pages of a layout that cannot spill (Layout::spillable) and Slim
  // pages of one whose pages cannot give up a window's layers
  // (Layout::slimmable).
  [[nodiscard]] metal::AllocationResult allocateExtent(uint32_t extent,
                                                       PageForm form = PageForm::Full) override;
  // The caller must prove that no active, prefix, reserved, or in-flight
  // reference remains anywhere in this extent. std::out_of_range for an
  // extent past the pool. KvPool never releases an extent that holds a page
  // a request or the cache holds; GPU page tables rely on it (Runtime's
  // PageTableBinding).
  void releaseExtent(uint32_t extent) override;
  // Copies what both pages hold: the K/V regions of the one that holds
  // fewer (Layout::regions) and the index, which is all of spans() between
  // pages of one form and indexSpans() between an Index page and another.
  // std::out_of_range for a page past the pool.
  void copyPages(std::span<const PageCopy> copies) override;
  // Zeroes everything each page holds (spans()). std::out_of_range for a
  // page past the pool.
  void clearPages(std::span<const uint32_t> pages) override;
  // Where each attention layer's K/V region sits in every extent that holds
  // it, by layer.
  [[nodiscard]] std::span<const SplashKvLayer> layers() const noexcept {
    return layers_;
  }
  // Where a layer's index region (Layout::indexDimension), the pooled keys of
  // every page, sits in every extent's index plane, in bytes: what an index
  // table's entries reach (writeIndexEntries). std::out_of_range without an
  // index or for a layer past the layout.
  [[nodiscard]] uint32_t indexOffset(uint32_t layer) const;

  // What the page holds: its extent's form. std::logic_error for a page
  // whose extent is not allocated.
  [[nodiscard]] PageForm form(uint32_t page) const;

  // Writes the entries kernels reach `pages`' K/V by, from index `first` on,
  // to a CPU-visible GPU page table, which must hold all of them; entries
  // before `first` are left as they are. An Index page's entry is
  // SPLASH_KV_PAGE_ABSENT; a Slim page's reaches the regions it holds, all
  // that a row its window has passed reads of it. Throws std::logic_error
  // for a table that is not CPU-visible or too small and for a page whose
  // extent is not allocated (a GPU table holds only pages of allocated
  // extents), std::invalid_argument for `first` past the pages.
  void writeEntries(std::span<const uint32_t> pages, uint32_t first,
                    const metal::MetalBuffer &table) const;
  // The same for an index table (ops::BlockIndex::tables), whose entries
  // reach the pages' block index, every page's whatever its form: the GPU
  // address of its extent's index plane, with the page's index in the low
  // bits (SPLASH_KV_PAGE_INDEX_MASK). std::logic_error also without an index.
  void writeIndexEntries(std::span<const uint32_t> pages, uint32_t first,
                         const metal::MetalBuffer &table) const;
  // The page's memory as the host reaches it, and the only code that names
  // it: the bytes of each tensor in each region the page's form holds,
  // region by region as keys, key scales, values and value scales, where
  // splash_kv_offset places them for the kernels, then each layer's pooled
  // keys (indexOffset): a Slim page's are the first spans of a Full page's
  // and its pooled keys, an Index page's its pooled keys alone. BF16 pages
  // have no scale bytes. Throws std::logic_error for a page whose extent is
  // not allocated.
  [[nodiscard]] std::vector<std::span<std::byte>> spans(uint32_t page) const;
  // The page's bytes of one layer's tensors, by SPLASH_KV_KEYS to
  // SPLASH_KV_VALUE_SCALES (empty scales for BF16), and of each layer's
  // pooled keys: the pieces of spans(). layerSpans() throws std::logic_error
  // for a layer whose region the page's form does not hold and
  // std::out_of_range for a layer past the layout; indexSpans() is empty
  // without an index.
  [[nodiscard]] std::array<std::span<std::byte>, 4> layerSpans(uint32_t page, uint32_t layer) const;
  [[nodiscard]] std::vector<std::span<std::byte>> indexSpans(uint32_t page) const;

private:
  struct Extent final {
    // Empty while the extent is not allocated.
    metal::MetalBuffer buffer;
    // The buffer's GPU address, zero while it is not allocated.
    uint64_t address = 0;
    PageForm form = PageForm::Full;
  };

  [[nodiscard]] size_t extentIndex(uint32_t page) const;
  // The page's allocated extent; std::logic_error while it is not allocated.
  [[nodiscard]] const Extent &allocatedExtent(uint32_t page) const;
  // Where an extent's index plane starts in its buffer: after the regions
  // its form holds.
  [[nodiscard]] uint64_t indexPlaneOffset(PageForm form) const noexcept {
    return uint64_t{extentPages_} * layout_.regions(form) * layout_.bytesPerLayerPage();
  }
  // The page's spans of its first `regions` regions, then of its index.
  [[nodiscard]] std::vector<std::span<std::byte>> spans(uint32_t page, uint32_t regions) const;
  template <class Entry>
  void writeTable(std::span<const uint32_t> pages, uint32_t first, const metal::MetalBuffer &table,
                  Entry entry) const;

  metal::MetalBackend &backend_;
  metal::AllocationAdmission admitAllocation_;
  Layout layout_;
  uint32_t pageCount_ = 0;
  uint32_t extentPages_ = 0;
  std::vector<SplashKvLayer> layers_;
  // The layer of each region, in region order (Layout::region).
  std::vector<uint32_t> regionLayers_;
  std::vector<Extent> extents_;
};

} // namespace splash::kv
