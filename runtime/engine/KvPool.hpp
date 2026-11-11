#pragma once

#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
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
  uint32_t extentsActive = 0;
  uint64_t activeBytes = 0;
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
  // The extents that hold a page a request holds, and their bytes: the
  // pool's extents commands reach through requests' tables, which extents
  // only the cache holds are not among.
  uint32_t extentsActive = 0;
  uint64_t activeBytes = 0;
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
  // The same pages and bytes of the extents of Index pages alone and of Slim
  // pages alone, which the counts above include.
  KvPoolFormSnapshot index{};
  KvPoolFormSnapshot slim{};
};

// Where moved pages went: an emptied extent's, or pages moved to another
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

// Pages moved to another form (KvPool::movePages): where each went, or,
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
// share one id space: a page's content goes to a page of another form when
// its K/V leaves memory or comes back, or when it gives up a window's layers
// (movePages). Free pages of a form are handed out from the extents of the
// form that hold a page a request holds first, then from the others, and
// among those from the one with the most live pages first: a request's new
// page joins an extent commands already reach rather than one only the
// cache holds, partially used extents fill up, empty extents are touched
// last, and cold extents drain to empty, the only state in which an extent
// can be released. Held pages are scattered
// over the extents all the same; compactExtent() empties one more extent
// whenever the free pages of the others of its form cover it. Commands reach
// a page only through the tables of the requests that hold it, which make
// resident only the extents of their pages (kv::PageStorage::writeEntries),
// so an extent only the cache holds costs commands nothing.
class KvPool final {
public:
  // Allocates the runway, the Full extents that hold pages [0, runwayPages),
  // or throws metal::MetalAllocationError with the budget's cause;
  // std::invalid_argument for a runway longer than the pool.
  KvPool(kv::ExtentStorage &storage, uint32_t runwayPages);

  // Holds the runway's pages, as a request holds its own, while it lives:
  // startup warmup runs on them without acquiring them, so the pool neither
  // hands them out nor releases their extents meanwhile. Rethrows, holding
  // none, when the pool refuses one.
  class RunwayHold final {
  public:
    explicit RunwayHold(KvPool &pool);
    ~RunwayHold();
    RunwayHold(const RunwayHold &) = delete;
    RunwayHold &operator=(const RunwayHold &) = delete;

  private:
    KvPool &pool_;
  };

  // The pages, of the form, come with one active reference each, the
  // requester's. `held` free pages of the form stay free, those waiting
  // claims ranked above the requester hold: the pages come from the free
  // pages beyond them, and then from extents it allocates.
  [[nodiscard]] KvPageAcquisition acquirePages(uint32_t count, kv::PageForm form = kv::PageForm::Full,
                                               uint32_t held = 0);
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
  // The one observer of pageActive: told of a page as a request's first
  // reference to it arrives or its last leaves, or as its references move
  // with its content (movePages, compactExtent), so that whoever keeps
  // counts of what the pages hold follows them (KvCache). Null for none;
  // std::logic_error for a second observer while one is set, whose counts
  // would stop following.
  void observeActivity(std::function<void(uint32_t page)> observer);
  // Between commands: gives each page, of another form, a page of form `to`
  // that takes what both hold (ExtentStorage::copyPages) and its references,
  // as compactExtent() moves them, and frees it; all of them or none. The
  // pages of form `to` come as acquirePages() hands them out, from the free
  // pages (the form's runway among them) beyond the `held` ones and then
  // from extents it allocates; a refused extent moves nothing and stays for
  // the retry. std::logic_error, moving nothing, for a page that is free,
  // already of form `to` or listed twice, and when the storage refuses the
  // copy.
  [[nodiscard]] KvPageMoveResult movePages(std::span<const uint32_t> pages, kv::PageForm to, uint32_t held = 0);
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;
  // Pages of the form's allocated extents no request holds: free, or held
  // only by the cache.
  [[nodiscard]] uint32_t reusablePages(kv::PageForm form = kv::PageForm::Full) const noexcept {
    const FormRecord &record = formRecord(form);
    return record.allocatedExtents * extentPages_ - record.activePages;
  }
  // The form's allocated extents, empty ones and the runway among them.
  [[nodiscard]] uint32_t allocatedExtents(kv::PageForm form) const noexcept {
    return formRecord(form).allocatedExtents;
  }
  // The form's allocated extents that hold no page, the runway among them.
  [[nodiscard]] uint32_t emptyExtents(kv::PageForm form) const noexcept {
    return formRecord(form).reclaimableExtents.count;
  }

  // Releases completely unreferenced extents, of the form or of any, at
  // most `limit` of them, and returns the bytes released. keepRunway keeps
  // one empty extent of each form warm, so the next request, spill or trim
  // does not wait for an allocation; unlike the runway the constructor
  // allocates, it is always a single extent.
  [[nodiscard]] uint64_t reclaimEmptyExtents(bool keepRunway, uint32_t limit,
                                             std::optional<kv::PageForm> form = std::nullopt);
  // Allocates an empty extent of the form unless one is allocated already:
  // the runway the next pages of the form come from without an allocation,
  // as a spill at the memory wall needs its Index pages to free its Full
  // ones, and a trim its Slim pages. Granted, or the budget's refusal with
  // nothing allocated.
  [[nodiscard]] metal::AllocationResult fillRunway(kv::PageForm form);
  // Empties the allocated extent, of the form or of any, that holds the
  // fewest pages, by moving each of them to a free page of the other extents
  // of its form that hold pages, fullest first: free pages scattered over the
  // pool become an empty extent, which a reclaim releases. Of any form, a
  // Full extent goes before a Slim one and a Slim one before an Index one, as
  // reclaimEmptyExtents() releases them; a caller that needs the memory of
  // one form names it. References move with their pages, and the result
  // says where each went, for the caller to re-point whoever names them.
  // Nothing moves, and the result is empty, unless those free pages cover
  // the extent. An extent that holds a page of `fixed` is not emptied.
  [[nodiscard]] KvPageMoves compactExtent(std::span<const uint32_t> fixed,
                                          std::optional<kv::PageForm> form = std::nullopt);
  // The form's extents that hold pages, none of `fixed`: those
  // compactExtent() may empty.
  [[nodiscard]] uint32_t unfixedExtents(std::span<const uint32_t> fixed, kv::PageForm form) const;
  // What compaction and release look at in an extent: its form, whether it
  // is allocated, and its used and free pages.
  struct ExtentCounts final {
    kv::PageForm form = kv::PageForm::Full;
    bool allocated = false;
    uint32_t usedPages = 0;
    uint32_t freePages = 0;
  };
  // Every extent's counts, by index.
  [[nodiscard]] std::vector<ExtentCounts> extentCounts() const;
  [[nodiscard]] KvPoolSnapshot snapshot() const;

private:
  static constexpr uint32_t noIndex = std::numeric_limits<uint32_t>::max();

  // The extent of the form compactExtent() empties among these: of those
  // that hold pages and no page of `fixed`, the one that holds the fewest,
  // ties to the lowest index, once the free pages of the other extents of
  // its form that hold pages cover it; none otherwise.
  [[nodiscard]] std::optional<uint32_t> compactable(std::span<const ExtentCounts> extents, kv::PageForm form,
                                                    std::span<const uint32_t> fixed) const noexcept;

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
    // The extent's pages a request holds: commands reach the extent while it
    // holds any.
    uint32_t activePages = 0;
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
    // The extents that hold a page a request holds.
    uint32_t activeExtents = 0;
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
  // Allocates an extent of the form at the lowest page ids no extent holds.
  // The page ids are the KV capacity the memory plan sized, which the budget
  // may outlast: with every extent allocated the pool refuses as the budget
  // does, and the claim's need of pages in place takes cached pages, which
  // hold ids (MemoryLedger::free).
  [[nodiscard]] metal::AllocationResult growExtent(kv::PageForm form);
  // Takes `count` free pages of the form off the free lists, leaving `held`
  // of them there, growing extents of the form when they run out; with a
  // refusal, puts them back and returns none.
  [[nodiscard]] KvPageAcquisition takeFreePages(uint32_t count, kv::PageForm form, uint32_t held = 0);
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
  // A page gains its first request's reference or loses its last: counted in
  // its form and its extent; when the extent's count leaves zero or reaches
  // it, the form counts the extent.
  void markActive(uint32_t page) noexcept;
  void markInactive(uint32_t page) noexcept;
  void setExtentAllocated(uint32_t extent, bool allocated) noexcept;
  void setExtentReclaimable(uint32_t extent, bool reclaimable) noexcept;

  // Every allocation and release of an extent, timed and counted; the
  // extent's record follows the storage.
  [[nodiscard]] metal::AllocationResult allocateExtent(uint32_t extent, kv::PageForm form);
  void releaseExtent(uint32_t extent);

  kv::ExtentStorage &storage_;
  uint32_t extentPages_ = 0;
  // The runway's pages, from page 0 (RunwayHold).
  uint32_t runwayPages_ = 0;
  uint64_t extentAllocations_ = 0;
  uint64_t extentReleases_ = 0;
  double extentAllocateMaxMilliseconds_ = 0.0;
  double extentReleaseMaxMilliseconds_ = 0.0;
  uint64_t extentCompactions_ = 0;
  uint64_t pagesMoved_ = 0;
  std::vector<PageRecord> pages_;
  std::vector<ExtentRecord> extents_;
  // By kv::PageForm.
  std::array<FormRecord, 3> forms_;
  std::function<void(uint32_t page)> activity_;
};

} // namespace splash::engine
