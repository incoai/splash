#pragma once

#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace splash::engine {

// The pages of the extents of one form (kv::PageForm), as KvPoolSnapshot
// counts them.
struct KvPoolFormSnapshot {
  uint32_t pagesAllocated = 0;
  uint32_t pagesFree = 0;
  uint32_t pagesActive = 0;
  uint32_t pagesPrefix = 0;
  uint64_t allocatedBytes = 0;
  uint64_t reclaimableBytes = 0;
};

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
  // The same pages and bytes of the extents of Index pages alone, which the
  // counts above include.
  KvPoolFormSnapshot index{};
};

// Where moved pages went: an emptied extent's, or pages moved to the other
// form. Whoever names one of them follows it.
struct KvPageMoves final {
  // Each moved page and the page that took its content, by moved page.
  std::vector<kv::PageCopy> moved;

  [[nodiscard]] bool empty() const noexcept { return moved.empty(); }
  // The page that holds what `page` held.
  [[nodiscard]] uint32_t follow(uint32_t page) const noexcept {
    const auto found = std::ranges::lower_bound(moved, page, {}, &kv::PageCopy::from);
    return found != moved.end() && found->from == page ? found->to : page;
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

// Pages moved to the other form (KvPool::movePages): where each went, or,
// with nothing moved, why the pool could not grow by an extent of the form.
struct KvPageMoveResult final {
  KvPageMoves moves;
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
// directly allocate or release Metal memory. Every extent holds pages of one
// form (kv::PageForm) from its allocation to its release, and the forms
// share one id space: a page's content goes to a page of the other form when
// its K/V leaves memory or comes back (movePages). Free pages of a form are
// handed out from the allocated extent of the form with the most live pages
// first, so partially used extents fill up, empty extents are touched last,
// and cold extents drain to empty, the only state in which an extent can be
// released. Held pages are scattered over the extents all the same;
// compactExtent() empties one more extent whenever the free pages of the
// others of its form cover it.
class KvPool final {
public:
  // Allocates the runway, the Full extents that hold pages [0, runwayPages),
  // or throws metal::MetalAllocationError with the budget's cause;
  // std::invalid_argument for a runway longer than the pool.
  KvPool(kv::ExtentStorage &storage, uint32_t runwayPages);

  // The pages, of the form, come with one active reference each, the
  // requester's.
  [[nodiscard]] KvPageAcquisition acquirePages(uint32_t count, kv::PageForm form = kv::PageForm::Full);
  void retainPage(uint32_t page, bool prefixOwner);
  void releasePage(uint32_t page, bool prefixOwner);
  // Zeroes a page its requester holds alone and has not written yet, so the
  // rows it never writes read zero rather than what the page held before.
  // std::logic_error for a page anyone else holds.
  void clearPage(uint32_t page);

  [[nodiscard]] uint32_t pageCount() const noexcept;
  [[nodiscard]] uint64_t bytesPerPage(kv::PageForm form = kv::PageForm::Full) const noexcept;
  [[nodiscard]] uint32_t extentPages() const noexcept { return extentPages_; }
  // Free pages of the form's allocated extents; acquisition hands these out
  // first.
  [[nodiscard]] uint32_t freePageCount(kv::PageForm form = kv::PageForm::Full) const noexcept;
  // What the page holds: its extent's form. std::logic_error for a page of
  // an unallocated extent.
  [[nodiscard]] kv::PageForm form(uint32_t page) const;
  // A request holds the page, as one of its own or as a page of a cached
  // chain it uses.
  [[nodiscard]] bool pageActive(uint32_t page) const;
  // Between commands: gives each page, of the other form, a page of form
  // `to` that takes its index (ExtentStorage::copyPages) and its references,
  // as compactExtent() moves them, and frees it; all of them or none. The
  // pages of form `to` come as acquirePages() hands them out, from the free
  // pages (the form's runway among them) and then from extents it allocates;
  // a refused extent moves nothing and stays for the retry. std::logic_error,
  // moving nothing, for a page that is free, already of form `to` or listed
  // twice, and when the storage refuses the copy.
  [[nodiscard]] KvPageMoveResult movePages(std::span<const uint32_t> pages, kv::PageForm to);
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;
  // Pages of the form's allocated extents no request holds: free, or held
  // only by the cache.
  [[nodiscard]] uint32_t reusablePages(kv::PageForm form = kv::PageForm::Full) const noexcept {
    const FormRecord &record = formRecord(form);
    return record.allocatedExtents * extentPages_ - record.activePages;
  }

  // Releases completely unreferenced extents, of the form or of either, at
  // most `limit` of them, and returns the bytes released. keepRunway keeps
  // one empty extent of each form warm, so the next request, or the next
  // spill, does not wait for an allocation; unlike the runway the
  // constructor allocates, it is always a single extent.
  [[nodiscard]] uint64_t reclaimEmptyExtents(bool keepRunway, uint32_t limit,
                                             std::optional<kv::PageForm> form = std::nullopt);
  // Allocates an empty extent of the form unless one is allocated already:
  // the runway the next pages of the form come from without an allocation,
  // as a spill at the memory wall needs its Index pages to free its Full
  // ones. Granted, or the budget's refusal with nothing allocated.
  [[nodiscard]] metal::AllocationResult fillRunway(kv::PageForm form);
  // Empties the allocated extent, of the form or of either, that holds the
  // fewest pages, by moving each of them to a free page of the other extents
  // of its form that hold pages, fullest first: free pages scattered over the
  // pool become an empty extent, which a reclaim releases. Of either form, a
  // Full extent goes before an Index one, as reclaimEmptyExtents() releases
  // them; a caller that needs the memory of one form names it. References
  // move with their pages, and the result says where each went, for the
  // caller to re-point whoever names them. Nothing moves, and the result is
  // empty, unless those free pages cover the extent. An extent that holds a
  // page of `fixed` is not emptied.
  [[nodiscard]] KvPageMoves compactExtent(std::span<const uint32_t> fixed,
                                          std::optional<kv::PageForm> form = std::nullopt);
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
    kv::PageForm form = kv::PageForm::Full;
    bool allocated = false;
    bool reclaimable = false;
  };

  // The extents of one form and their pages.
  struct FormRecord {
    uint32_t allocatedExtents = 0;
    uint32_t freePages = 0;
    uint32_t activePages = 0;
    uint32_t prefixPages = 0;
    // The extent currently being filled. Stays valid while only this extent
    // changes, so a burst of allocations rescans the extents once per extent
    // it moves into.
    uint32_t fillingExtent = noIndex;
    // The allocated extents none of whose pages is held, the one emptied
    // last first.
    IndexList reclaimableExtents;
  };

  [[nodiscard]] FormRecord &formRecord(kv::PageForm form) noexcept {
    return forms_[static_cast<size_t>(form)];
  }
  [[nodiscard]] const FormRecord &formRecord(kv::PageForm form) const noexcept {
    return forms_[static_cast<size_t>(form)];
  }
  [[nodiscard]] uint32_t extentOf(uint32_t page) const noexcept {
    return page / extentPages_;
  }
  [[nodiscard]] uint32_t firstPage(uint32_t extent) const noexcept {
    return extent * extentPages_;
  }
  [[nodiscard]] FormRecord &formOf(uint32_t page) noexcept {
    return formRecord(extents_[extentOf(page)].form);
  }
  [[nodiscard]] bool pageFree(uint32_t page) const;
  // The lowest extent that is not allocated; noIndex when all of them are.
  [[nodiscard]] uint32_t unallocatedExtent() const noexcept;
  // The extent of the form compactExtent() empties; noIndex when the free
  // pages of the form cover none.
  [[nodiscard]] uint32_t compactableExtent(kv::PageForm form, std::span<const uint32_t> fixed) const noexcept;
  // Takes `count` free pages of the form off the free lists, allocating
  // extents of the form when they run out; with a refusal, puts them back
  // and returns none.
  [[nodiscard]] KvPageAcquisition takeFreePages(uint32_t count, kv::PageForm form);
  // Gives a page's references to the free page its content was copied to,
  // and frees it.
  void moveReferences(const kv::PageCopy &copy) noexcept;
  [[nodiscard]] KvPoolFormSnapshot formSnapshot(kv::PageForm form) const noexcept;
  void insertFree(uint32_t page) noexcept;
  void removeFree(uint32_t page) noexcept;
  [[nodiscard]] uint32_t popFree(kv::PageForm form) noexcept;
  [[nodiscard]] uint32_t fillingExtent(kv::PageForm form) noexcept;
  void markUsed(uint32_t page) noexcept;
  void markFree(uint32_t page) noexcept;
  void setExtentAllocated(uint32_t extent, bool allocated) noexcept;
  void setExtentReclaimable(uint32_t extent, bool reclaimable) noexcept;

  // Every allocation and release of an extent, timed and counted; the
  // extent's record follows the storage.
  [[nodiscard]] metal::AllocationResult allocateExtent(uint32_t extent, kv::PageForm form);
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
  // By kv::PageForm.
  std::array<FormRecord, 2> forms_;
};

} // namespace splash::engine
