#include "engine/KvPool.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::engine {
namespace {

double millisecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

KvPool::KvPool(kv::ExtentStorage &storage, uint32_t runwayPages)
    : storage_(storage), extentPages_(storage.extentPages()),
      pages_(storage.pageCount()) {
  if (pages_.empty() || !storage_.bytesPerPage() || !extentPages_ ||
      pages_.size() % extentPages_) {
    throw std::invalid_argument("invalid KV extent storage");
  }
  if (runwayPages > pages_.size())
    throw std::invalid_argument("the KV runway exceeds the page pool");
  extents_.resize(pages_.size() / extentPages_);
  for (uint32_t extent = 0; firstPage(extent) < runwayPages; ++extent) {
    if (const metal::AllocationResult allocated = allocateExtent(extent);
        !allocated) {
      throw metal::MetalAllocationError(
          std::string("unable to allocate the KV runway: ") +
              metal::allocationFailureName(allocated.failure),
          allocated.failure);
    }
  }
}

KvPageAcquisition KvPool::acquirePages(uint32_t count) {
  if (!count)
    return {};

  std::vector<uint32_t> selected;
  selected.reserve(count);
  auto returnSelected = [&] {
    for (auto p = selected.rbegin(); p != selected.rend(); ++p)
      insertFree(*p);
  };
  while (selected.size() < count) {
    if (freePages_) {
      selected.push_back(popFree());
      continue;
    }
    // Page ids cover every extent the hard budget could hold, so the budget
    // refuses growth before they run out.
    const uint32_t extent = unallocatedExtent();
    if (extent == noIndex)
      throw std::logic_error("KV page ids ran out before the memory budget did");
    const metal::AllocationResult allocated = allocateExtent(extent);
    if (!allocated) {
      // The extents this acquisition allocated stay, reclaimable: the
      // budget admitted them, and the retry that follows a reclaim takes
      // their pages first instead of allocating them again. A reclaim pass
      // returns them if they stay unused.
      returnSelected();
      return {{}, allocated.failure};
    }
  }

  for (uint32_t page : selected) {
    markUsed(page);
    pages_[page].activeReferences = 1;
    ++activePages_;
  }
  return {std::move(selected)};
}

void KvPool::retainPage(uint32_t page, bool prefixOwner) {
  PageRecord &record = pages_.at(page);
  if (!extents_[extentOf(page)].allocated) {
    throw std::logic_error("cannot retain a KV page of an unallocated extent");
  }
  if (prefixOwner && record.prefixOwned)
    throw std::logic_error("KV page already belongs to a cached block");
  if (!prefixOwner && record.activeReferences == std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("KV page reference overflow");
  if (pageFree(page))
    markUsed(page);
  if (prefixOwner) {
    record.prefixOwned = true;
    ++prefixPages_;
  } else if (!record.activeReferences++) {
    ++activePages_;
  }
}

void KvPool::releasePage(uint32_t page, bool prefixOwner) {
  PageRecord &record = pages_.at(page);
  if (prefixOwner ? !record.prefixOwned : !record.activeReferences)
    throw std::logic_error("invalid KV page release");
  if (prefixOwner) {
    record.prefixOwned = false;
    --prefixPages_;
  } else if (!--record.activeReferences) {
    --activePages_;
  }
  if (pageFree(page))
    markFree(page);
}

uint32_t KvPool::pageCount() const noexcept {
  return static_cast<uint32_t>(pages_.size());
}

uint64_t KvPool::bytesPerPage() const noexcept {
  return storage_.bytesPerPage();
}

uint32_t KvPool::freePageCount() const noexcept { return freePages_; }

bool KvPool::pageActive(uint32_t page) const {
  return pages_.at(page).activeReferences != 0;
}

bool KvPool::pageFree(uint32_t page) const {
  const PageRecord &record = pages_.at(page);
  return !record.activeReferences && !record.prefixOwned;
}

uint64_t KvPool::allocatedBytes() const noexcept {
  return uint64_t{allocatedExtents_} * extentPages_ * bytesPerPage();
}

uint64_t KvPool::reclaimEmptyExtents(bool keepRunway, uint32_t limit) {
  uint32_t reclaimed = 0;
  bool kept = false;
  uint32_t extent = reclaimableExtents_.head;
  while (extent != noIndex && reclaimed < limit) {
    const uint32_t next = extents_[extent].nextReclaimable;
    if (keepRunway && !kept) {
      kept = true;
    } else {
      releaseExtent(extent);
      ++reclaimed;
    }
    extent = next;
  }
  return uint64_t{reclaimed} * extentPages_ * bytesPerPage();
}

metal::AllocationResult KvPool::allocateExtent(uint32_t extent) {
  const auto start = std::chrono::steady_clock::now();
  const metal::AllocationResult allocated = storage_.allocateExtent(extent);
  if (allocated) {
    ++extentAllocations_;
    extentAllocateMaxMilliseconds_ =
        std::max(extentAllocateMaxMilliseconds_, millisecondsSince(start));
    setExtentAllocated(extent, true);
  }
  return allocated;
}

void KvPool::releaseExtent(uint32_t extent) {
  const auto start = std::chrono::steady_clock::now();
  storage_.releaseExtent(extent);
  ++extentReleases_;
  extentReleaseMaxMilliseconds_ =
      std::max(extentReleaseMaxMilliseconds_, millisecondsSince(start));
  setExtentAllocated(extent, false);
}

KvPageMoves KvPool::compactExtent(std::span<const uint32_t> fixed) {
  // Pages move only into extents that hold pages already: an empty extent is
  // released as it is, never filled to release another.
  uint32_t freeInUse = 0;
  for (const ExtentRecord &extent : extents_) {
    if (extent.allocated && extent.usedPages)
      freeInUse += extent.freePages.count;
  }
  uint32_t emptied = noIndex;
  for (uint32_t index = 0; index < extents_.size(); ++index) {
    const ExtentRecord &extent = extents_[index];
    if (!extent.allocated || !extent.usedPages ||
        extent.usedPages > freeInUse - extent.freePages.count ||
        (emptied != noIndex && extent.usedPages >= extents_[emptied].usedPages))
      continue;
    if (std::none_of(fixed.begin(), fixed.end(),
                     [&](uint32_t page) { return extentOf(page) == index; }))
      emptied = index;
  }
  if (emptied == noIndex)
    return {};

  std::vector<uint32_t> receivers;
  for (uint32_t index = 0; index < extents_.size(); ++index) {
    const ExtentRecord &extent = extents_[index];
    if (index != emptied && extent.allocated && extent.usedPages &&
        extent.freePages.count)
      receivers.push_back(index);
  }
  std::stable_sort(receivers.begin(), receivers.end(),
                   [&](uint32_t left, uint32_t right) {
                     return extents_[left].usedPages > extents_[right].usedPages;
                   });
  const uint32_t first = firstPage(emptied);
  std::vector<kv::PageCopy> copies;
  copies.reserve(extents_[emptied].usedPages);
  auto receiver = receivers.begin();
  uint32_t to = extents_[*receiver].freePages.head;
  for (uint32_t page = first; page < first + extentPages_; ++page) {
    if (pageFree(page))
      continue;
    // The receivers' free pages cover the extent, so one always follows.
    while (to == noIndex)
      to = extents_[*++receiver].freePages.head;
    copies.push_back({page, to});
    to = pages_[to].nextFree;
  }

  storage_.copyPages(copies);
  KvPageMoves moves{first, std::vector<uint32_t>(extentPages_)};
  std::iota(moves.destinations.begin(), moves.destinations.end(), first);
  for (const kv::PageCopy &copy : copies) {
    markUsed(copy.to);
    pages_[copy.to].activeReferences =
        std::exchange(pages_[copy.from].activeReferences, 0);
    pages_[copy.to].prefixOwned = std::exchange(pages_[copy.from].prefixOwned, false);
    markFree(copy.from);
    moves.destinations[copy.from - moves.firstPage] = copy.to;
  }
  ++extentCompactions_;
  pagesMoved_ += copies.size();
  return moves;
}

KvPoolSnapshot KvPool::snapshot() const {
  KvPoolSnapshot result;
  result.pagesAllocated = allocatedExtents_ * extentPages_;
  result.pagesActive = activePages_;
  result.pagesPrefix = prefixPages_;
  result.pagesFree = freePages_;
  result.allocatedBytes = allocatedBytes();
  result.reclaimableBytes =
      uint64_t{reclaimableExtents_.count} * extentPages_ * bytesPerPage();
  result.extentAllocations = extentAllocations_;
  result.extentReleases = extentReleases_;
  result.extentAllocateMaxMilliseconds = extentAllocateMaxMilliseconds_;
  result.extentReleaseMaxMilliseconds = extentReleaseMaxMilliseconds_;
  result.extentCompactions = extentCompactions_;
  result.pagesMoved = pagesMoved_;
  return result;
}

uint32_t KvPool::unallocatedExtent() const noexcept {
  for (uint32_t extent = 0; extent < extents_.size(); ++extent) {
    if (!extents_[extent].allocated)
      return extent;
  }
  return noIndex;
}

void KvPool::insertFree(uint32_t page) noexcept {
  PageRecord &record = pages_[page];
  if (record.onFreeList || !pageFree(page))
    std::terminate();
  IndexList &list = extents_[extentOf(page)].freePages;
  record.previousFree = noIndex;
  record.nextFree = list.head;
  record.onFreeList = true;
  if (list.head != noIndex)
    pages_[list.head].previousFree = page;
  list.head = page;
  ++list.count;
  ++freePages_;
  packingExtent_ = noIndex;
}

void KvPool::removeFree(uint32_t page) noexcept {
  PageRecord &record = pages_[page];
  if (!record.onFreeList)
    std::terminate();
  IndexList &list = extents_[extentOf(page)].freePages;
  if (record.previousFree == noIndex) {
    if (list.head != page)
      std::terminate();
    list.head = record.nextFree;
  } else {
    pages_[record.previousFree].nextFree = record.nextFree;
  }
  if (record.nextFree != noIndex)
    pages_[record.nextFree].previousFree = record.previousFree;
  record.previousFree = noIndex;
  record.nextFree = noIndex;
  record.onFreeList = false;
  if (!list.count || !freePages_)
    std::terminate();
  --list.count;
  --freePages_;
}

uint32_t KvPool::popFree() noexcept {
  const uint32_t page = extents_[packingExtent()].freePages.head;
  if (page == noIndex)
    std::terminate();
  removeFree(page);
  return page;
}

// The allocated extent with the most live pages that still has a free page;
// ties go to the lowest index, and empty extents lose to any used one. The
// answer only changes when another extent gains or loses a page, so it is
// reused until then.
uint32_t KvPool::packingExtent() noexcept {
  if (packingExtent_ != noIndex && extents_[packingExtent_].freePages.count)
    return packingExtent_;
  uint32_t best = noIndex;
  for (uint32_t index = 0; index < extents_.size(); ++index) {
    const ExtentRecord &extent = extents_[index];
    if (extent.freePages.count &&
        (best == noIndex || extent.usedPages > extents_[best].usedPages)) {
      best = index;
    }
  }
  if (best == noIndex)
    std::terminate();
  packingExtent_ = best;
  return best;
}

void KvPool::markUsed(uint32_t page) noexcept {
  if (!pageFree(page))
    std::terminate();
  const uint32_t extentIndex = extentOf(page);
  if (pages_[page].onFreeList) {
    removeFree(page);
    if (extentIndex != packingExtent_)
      packingExtent_ = noIndex;
  }
  ExtentRecord &extent = extents_[extentIndex];
  if (!extent.usedPages)
    setExtentReclaimable(extentIndex, false);
  ++extent.usedPages;
}

void KvPool::markFree(uint32_t page) noexcept {
  const uint32_t extentIndex = extentOf(page);
  ExtentRecord &extent = extents_[extentIndex];
  if (!extent.usedPages)
    std::terminate();
  --extent.usedPages;
  insertFree(page);
  if (!extent.usedPages)
    setExtentReclaimable(extentIndex, true);
}

// An extent changes state only while none of its pages is held: its pages
// join the free lists when it is allocated, lowest page first, and leave
// them when it is released.
void KvPool::setExtentAllocated(uint32_t extentIndex, bool allocated) noexcept {
  ExtentRecord &extent = extents_[extentIndex];
  if (extent.allocated == allocated || extent.usedPages)
    std::terminate();
  const uint32_t first = firstPage(extentIndex);
  if (allocated) {
    extent.allocated = true;
    ++allocatedExtents_;
    for (uint32_t page = first + extentPages_; page > first;)
      insertFree(--page);
    setExtentReclaimable(extentIndex, true);
    return;
  }
  setExtentReclaimable(extentIndex, false);
  for (uint32_t page = first; page < first + extentPages_; ++page)
    removeFree(page);
  if (!allocatedExtents_)
    std::terminate();
  extent.allocated = false;
  --allocatedExtents_;
}

void KvPool::setExtentReclaimable(uint32_t extentIndex,
                                  bool reclaimable) noexcept {
  ExtentRecord &extent = extents_[extentIndex];
  if (extent.reclaimable == reclaimable)
    return;
  if (reclaimable) {
    if (!extent.allocated || extent.usedPages)
      std::terminate();
    extent.previousReclaimable = noIndex;
    extent.nextReclaimable = reclaimableExtents_.head;
    if (reclaimableExtents_.head != noIndex) {
      extents_[reclaimableExtents_.head].previousReclaimable = extentIndex;
    }
    reclaimableExtents_.head = extentIndex;
    ++reclaimableExtents_.count;
    extent.reclaimable = true;
    return;
  }
  if (extent.previousReclaimable == noIndex) {
    if (reclaimableExtents_.head != extentIndex)
      std::terminate();
    reclaimableExtents_.head = extent.nextReclaimable;
  } else {
    extents_[extent.previousReclaimable].nextReclaimable =
        extent.nextReclaimable;
  }
  if (extent.nextReclaimable != noIndex) {
    extents_[extent.nextReclaimable].previousReclaimable =
        extent.previousReclaimable;
  }
  extent.previousReclaimable = noIndex;
  extent.nextReclaimable = noIndex;
  extent.reclaimable = false;
  if (!reclaimableExtents_.count)
    std::terminate();
  --reclaimableExtents_.count;
}

} // namespace splash::engine
