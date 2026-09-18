#pragma once

#include "metal/MetalBackend.hpp"
#include "ops/PagedKv.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <stdexcept>
#include <vector>

namespace splash::kv {

// Physical storage for Q8 pages. Active requests may overwrite slots at or
// beyond their logical commit index. Cached KV blocks reference only fully
// committed pages, which are immutable while shared.
class Q8PageStorage final : public Backing {
public:
  Q8PageStorage(metal::MetalBackend &backend,
                metal::AllocationAdmission admitAllocation,
                Q8Layout layout,
                uint32_t pageCount);
  ~Q8PageStorage() override;

  Q8PageStorage(const Q8PageStorage &) = delete;
  Q8PageStorage &operator=(const Q8PageStorage &) = delete;

  [[nodiscard]] uint32_t pageCount() const noexcept override {
    return pageCount_;
  }
  [[nodiscard]] uint64_t bytesPerPage() const noexcept override {
    return layout_.bytesPerModelPage();
  }
  [[nodiscard]] Q8Layout layout() const noexcept { return layout_; }
  [[nodiscard]] uint32_t sparseMappingBatchPages() const noexcept {
    return layout_.sparseMappingBatchPages();
  }
  [[nodiscard]] uint32_t backingExtentPages() const noexcept {
    return layout_.backingExtentPages();
  }
  [[nodiscard]] uint64_t declaredBytes() const noexcept;
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;
  [[nodiscard]] uint32_t residentPages() const noexcept;
  [[nodiscard]] bool isResident(uint32_t page) const override;
  [[nodiscard]] metal::AllocationResult ensureResident(uint32_t page) override;
  // The caller must prove that no active, prefix, reserved, or in-flight
  // reference remains anywhere in this extent. The unmap is asynchronous:
  // the backend retains the heap until the kernel finishes, and the next
  // release waits for releaseReady().
  [[nodiscard]] bool releaseBackingForPage(uint32_t page) override;
  [[nodiscard]] bool releaseReady() const noexcept override;
  void awaitRelease() override;
  [[nodiscard]] uint32_t extentFirstPage(uint32_t page) const override;
  [[nodiscard]] uint32_t extentPageCount(uint32_t page) const override;
  [[nodiscard]] const Q8LayerStorage &layer(uint32_t index) const {
    if (index >= layers_.size())
      throw std::out_of_range("invalid attention layer index");
    return layers_[index];
  }

private:
  struct Extent {
    uint32_t firstPage = 0;
    uint32_t pageCount = 0;
    std::optional<metal::SparseHeap> heap;
  };

  [[nodiscard]] size_t extentIndex(uint32_t page) const;
  [[nodiscard]] std::vector<metal::SparseMapping>
  mappingsFor(const Extent &extent) const;

  metal::MetalBackend &backend_;
  metal::AllocationAdmission admitAllocation_;
  Q8Layout layout_;
  uint32_t pageCount_ = 0;
  std::vector<Q8LayerStorage> layers_;
  std::vector<Extent> extents_;
  uint64_t residentBackingBytes_ = 0;
  uint32_t residentPages_ = 0;
};

} // namespace splash::kv
