#pragma once

#include "model/DFlashDraft.hpp"
#include "model/SlotFile.hpp"

namespace splash::model {

// One BF16 page contains the same token range in every draft layer. Its
// layout matches an attention tile, so active requests and cached windows
// use the same backing without packing or restoring a separate ring.
class DraftKvCache final {
public:
  struct Page final {
    metal::MetalBuffer buffer;
    std::shared_ptr<StateAllocationTracker> tracker;
    uint64_t bytes = 0;
    ~Page();
  };
  struct Pool final {
    std::vector<std::shared_ptr<Page>> free;
    uint32_t allocatedPages = 0;
    bool open = true;
  };
  struct Block final {
    std::shared_ptr<Page> page;
    std::shared_ptr<Pool> pool;
    mutable std::weak_ptr<SlotFile::Slot> disk;
    ~Block();
  };
  using Window = std::vector<std::shared_ptr<Block>>;

  DraftKvCache(metal::MetalBackend &backend,
               metal::AllocationAdmission admission,
               std::shared_ptr<StateAllocationTracker> tracker,
               DraftStateLayout layout);
  ~DraftKvCache();
  [[nodiscard]] std::shared_ptr<Block>
  acquire(metal::AllocationFailure *failure = nullptr);
  void releaseIdle(uint32_t keep = 0) noexcept;
  [[nodiscard]] uint32_t idlePages() const noexcept {
    return pool_->free.size();
  }
  [[nodiscard]] std::vector<DFlashDraftRingLayer>
  views(const Window &pages) const;

  // Declare before the batch's page handles, so they return to the pool first.
  // A failed batch must not turn its partial allocations into apparent reclaim
  // progress: an admission retry would otherwise allocate/free them forever.
  class AllocationScope final {
  public:
    explicit AllocationScope(DraftKvCache &cache)
        : cache_(cache), keep_(cache.idlePages()) {}
    ~AllocationScope() {
      if (!committed_)
        cache_.releaseIdle(keep_);
    }
    AllocationScope(const AllocationScope &) = delete;
    AllocationScope &operator=(const AllocationScope &) = delete;
    void commit() noexcept { committed_ = true; }

  private:
    DraftKvCache &cache_;
    uint32_t keep_;
    bool committed_ = false;
  };

private:
  metal::MetalBackend &backend_;
  metal::AllocationAdmission admission_;
  std::shared_ptr<StateAllocationTracker> tracker_;
  DraftStateLayout layout_;
  std::shared_ptr<Pool> pool_ = std::make_shared<Pool>();
};

// A circular block table for the sliding window. Snapshots retain page
// references. Before a GPU write, shared pages are copied; exclusive pages
// stay in place. Query/verification scratch never enters this table.
class DFlashDraftRing final {
public:
  DraftKvCache::Window pages;
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept {
    uint64_t bytes = 0;
    for (const auto &block : pages)
      bytes += block->page->bytes;
    return bytes;
  }
};

} // namespace splash::model
