#pragma once

#include "ops/PagedKv.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace splash::engine {

struct KvPoolSnapshot {
  // Pages of allocated extents: all of them, those nothing holds, and those
  // requests and the cache hold.
  uint32_t pagesAllocated = 0;
  uint32_t pagesFree = 0;
  uint32_t pagesActive = 0;
  uint32_t pagesPrefix = 0;
  uint64_t allocatedBytes = 0;
  // The bytes of extents none of whose pages is held, which a reclaim
  // releases at once.
  uint64_t reclaimableBytes = 0;
  // Extents allocated and released through the pool, and the longest
  // allocation and release of one: what memory costs per extent. The counts
  // and the longest allocation include the runway the pool allocates when it
  // is built, before the serving loop runs. How long a whole reclaim pass
  // holds the loop shows in its longest tick.
  uint64_t extentAllocations = 0;
  uint64_t extentReleases = 0;
  double extentAllocateMaxMilliseconds = 0.0;
  double extentReleaseMaxMilliseconds = 0.0;
  // Extents emptied by moving their pages, and the pages moved.
  uint64_t extentCompactions = 0;
  uint64_t pagesMoved = 0;
};

// Where the pages of an emptied extent went. Whoever names one of its pages
// follows it.
struct KvPageMoves final {
  uint32_t firstPage = 0;
  // By offset in the extent: the page that took the page's content, or the
  // page itself where it was free.
  std::vector<uint32_t> destinations;

  [[nodiscard]] bool empty() const noexcept { return destinations.empty(); }
  // The page that holds what `page` held.
  [[nodiscard]] uint32_t follow(uint32_t page) const noexcept {
    return page >= firstPage && page - firstPage < destinations.size()
               ? destinations[page - firstPage]
               : page;
  }
};

struct KvPageAcquisition {
  std::vector<uint32_t> pages;
  // Why the pool could not grow by an extent; None when granted.
  metal::AllocationFailure failure = metal::AllocationFailure::None;

  [[nodiscard]] bool granted() const noexcept {
    return failure == metal::AllocationFailure::None;
  }
};

// Sole owner of KV page references and of which extents are allocated.
// Requests hold counted references to a page; the cache owns a page once,
// for the one block on it (prefixOwner). The pool alone allocates and
// releases extents, starting with the runway its constructor allocates.
// Resource policy may ask for pages or release references, but cannot
// directly allocate or release Metal memory. Free pages are handed out from
// the allocated extent with the most live pages first, so partially used
// extents fill up, empty extents are touched last, and cold extents drain to
// empty, the only state in which an extent can be released. Held pages are
// scattered over the extents all the same; compactExtent() empties one more
// extent whenever the free pages of the others cover it.
class KvPool final {
public:
  // Allocates the runway, the extents that hold pages [0, runwayPages), or
  // throws metal::MetalAllocationError with the budget's cause;
  // std::invalid_argument for a runway longer than the pool.
  KvPool(kv::ExtentStorage &storage, uint32_t runwayPages);

  // The pages come with one active reference each, the requester's.
  [[nodiscard]] KvPageAcquisition acquirePages(uint32_t count);
  void retainPage(uint32_t page, bool prefixOwner);
  void releasePage(uint32_t page, bool prefixOwner);

  [[nodiscard]] uint32_t pageCount() const noexcept;
  [[nodiscard]] uint64_t bytesPerPage() const noexcept;
  [[nodiscard]] uint32_t extentPages() const noexcept { return extentPages_; }
  // Free pages of allocated extents; acquisition hands these out first.
  [[nodiscard]] uint32_t freePageCount() const noexcept;
  // A request holds the page, as one of its own or as a page of a cached
  // chain it uses.
  [[nodiscard]] bool pageActive(uint32_t page) const;
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;
  // Pages of allocated extents no request holds: free, or held only by the
  // cache.
  [[nodiscard]] uint32_t reusablePages() const noexcept {
    return allocatedExtents_ * extentPages_ - activePages_;
  }

  // Releases completely unreferenced extents, at most `limit` of them, and
  // returns the bytes released. keepRunway keeps one empty extent warm, so
  // the next request does not wait for an allocation; unlike the runway the
  // constructor allocates, it is always a single extent.
  [[nodiscard]] uint64_t reclaimEmptyExtents(bool keepRunway, uint32_t limit);
  // Empties the allocated extent that holds the fewest pages, by moving each
  // of them to a free page of the other extents that hold pages, fullest
  // first: free pages scattered over the pool become an empty extent, which
  // a reclaim releases. References move with their pages, and the result
  // says where each went, for the caller to re-point whoever names them.
  // Nothing moves, and the result is empty, unless those free pages cover
  // the extent. An extent that holds a page of `fixed` is not emptied.
  [[nodiscard]] KvPageMoves compactExtent(std::span<const uint32_t> fixed);
  [[nodiscard]] KvPoolSnapshot snapshot() const;

private:
  static constexpr uint32_t noIndex = std::numeric_limits<uint32_t>::max();

  struct PageRecord {
    uint32_t activeReferences = 0;
    uint32_t previousFree = noIndex;
    uint32_t nextFree = noIndex;
    bool prefixOwned = false;
    bool onFreeList = false;
  };

  struct IndexList {
    uint32_t head = noIndex;
    uint32_t count = 0;
  };

  struct ExtentRecord {
    uint32_t usedPages = 0;
    uint32_t previousReclaimable = noIndex;
    uint32_t nextReclaimable = noIndex;
    // The extent's pages nothing holds; only an allocated extent lists any.
    IndexList freePages;
    bool allocated = false;
    bool reclaimable = false;
  };

  [[nodiscard]] uint32_t extentOf(uint32_t page) const noexcept {
    return page / extentPages_;
  }
  [[nodiscard]] uint32_t firstPage(uint32_t extent) const noexcept {
    return extent * extentPages_;
  }
  [[nodiscard]] bool pageFree(uint32_t page) const;
  // The lowest extent that is not allocated; noIndex when all of them are.
  [[nodiscard]] uint32_t unallocatedExtent() const noexcept;
  void insertFree(uint32_t page) noexcept;
  void removeFree(uint32_t page) noexcept;
  [[nodiscard]] uint32_t popFree() noexcept;
  [[nodiscard]] uint32_t packingExtent() noexcept;
  void markUsed(uint32_t page) noexcept;
  void markFree(uint32_t page) noexcept;
  void setExtentAllocated(uint32_t extent, bool allocated) noexcept;
  void setExtentReclaimable(uint32_t extent, bool reclaimable) noexcept;

  // Every allocation and release of an extent, timed and counted; the
  // extent's record follows the storage.
  [[nodiscard]] metal::AllocationResult allocateExtent(uint32_t extent);
  void releaseExtent(uint32_t extent);

  kv::ExtentStorage &storage_;
  uint32_t extentPages_ = 0;
  uint64_t extentAllocations_ = 0;
  uint64_t extentReleases_ = 0;
  double extentAllocateMaxMilliseconds_ = 0.0;
  double extentReleaseMaxMilliseconds_ = 0.0;
  uint64_t extentCompactions_ = 0;
  uint64_t pagesMoved_ = 0;
  std::vector<PageRecord> pages_;
  std::vector<ExtentRecord> extents_;
  uint32_t freePages_ = 0;
  // The extent currently being filled. Stays valid while only this extent
  // changes, so a burst of allocations rescans the extents once per extent
  // it moves into.
  uint32_t packingExtent_ = noIndex;
  IndexList reclaimableExtents_;
  uint32_t activePages_ = 0;
  uint32_t prefixPages_ = 0;
  uint32_t allocatedExtents_ = 0;
};

} // namespace splash::engine
