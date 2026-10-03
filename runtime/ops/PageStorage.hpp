#pragma once

#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/PagedKv.hpp"

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
// and value scales (abi/KvExtent.h). Kernels reach a page through its entry
// in a request's GPU page table, and the backend's residency set keeps every
// extent resident for every command; the host reaches the same bytes through
// the page's spans(). Active requests may overwrite slots at or beyond their
// logical commit index. Cached KV blocks reference only fully committed
// pages, which are immutable while shared.
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
  [[nodiscard]] uint64_t bytesPerPage() const noexcept override {
    return layout_.bytesPerModelPage();
  }
  [[nodiscard]] Layout layout() const noexcept { return layout_; }
  [[nodiscard]] uint32_t extentPages() const noexcept override {
    return extentPages_;
  }
  [[nodiscard]] uint64_t extentBytes() const noexcept {
    return uint64_t{extentPages_} * layout_.bytesPerModelPage();
  }
  // The bytes of the extents that hold memory now: a measurement of the
  // buffers, which KvPool's record of its extents must match.
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;
  [[nodiscard]] bool isAllocated(uint32_t page) const;
  // std::out_of_range for an extent past the pool.
  [[nodiscard]] metal::AllocationResult allocateExtent(uint32_t extent) override;
  // The caller must prove that no active, prefix, reserved, or in-flight
  // reference remains anywhere in this extent. std::out_of_range for an
  // extent past the pool. KvPool never releases an extent that holds a page
  // a request or the cache holds; GPU page tables rely on it (Runtime's
  // PageTableBinding).
  void releaseExtent(uint32_t extent) override;
  // Copies every tensor of each page in every layer (spans()).
  // std::out_of_range for a page past the pool.
  void copyPages(std::span<const PageCopy> copies) override;
  // Where each attention layer's region sits in every extent, by layer.
  [[nodiscard]] std::span<const SplashKvLayer> layers() const noexcept {
    return layers_;
  }

  // Writes the entries kernels reach `pages` by, from index `first` on, to a
  // CPU-visible GPU page table, which must hold all of them; entries before
  // `first` are left as they are. Throws std::logic_error for a table that is
  // not CPU-visible or too small and for a page whose extent is not allocated
  // (a GPU table holds only pages of allocated extents), std::invalid_argument
  // for `first` past the pages.
  void writeEntries(std::span<const uint32_t> pages, uint32_t first,
                    const metal::MetalBuffer &table) const;
  // The page's memory as the host reaches it, and the only code that names
  // it: the page's bytes of each tensor in every layer's region, layer by
  // layer as keys, key scales, values and value scales, where
  // splash_kv_offset places them for the kernels. BF16 pages have no scale
  // bytes. Throws std::logic_error for a page whose extent is not allocated.
  [[nodiscard]] std::vector<std::span<std::byte>> spans(uint32_t page) const;

private:
  [[nodiscard]] size_t extentIndex(uint32_t page) const;
  [[nodiscard]] SplashKvPage entry(uint32_t page) const;

  metal::MetalBackend &backend_;
  metal::AllocationAdmission admitAllocation_;
  Layout layout_;
  uint32_t pageCount_ = 0;
  uint32_t extentPages_ = 0;
  std::vector<SplashKvLayer> layers_;
  // Empty while the extent is not allocated.
  std::vector<metal::MetalBuffer> extents_;
  // Each extent's GPU address, zero while it is not allocated.
  std::vector<uint64_t> extentAddresses_;
};

} // namespace splash::kv
