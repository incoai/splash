#include "engine/KvPool.hpp"
#include "AwakeClock.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::engine {

namespace {

// The order extents of each form are compacted and released in.
constexpr std::array kForms{kv::PageForm::Full, kv::PageForm::Slim, kv::PageForm::Index};

} // namespace

KvPool::KvPool(kv::ExtentStorage &storage, uint32_t runwayPages)
    : storage_(storage), extentPages_(storage.extentPages()),
      pages_(storage.pageCount()) {
  if (pages_.empty() || !storage_.bytesPerPage(kv::PageForm::Full) || !extentPages_ ||
      pages_.size() % extentPages_) {
    throw std::invalid_argument("invalid KV extent storage");
  }
  if (runwayPages > pages_.size())
    throw std::invalid_argument("the KV runway exceeds the page pool");
  extents_.resize(pages_.size() / extentPages_);
  for (uint32_t extent = 0; firstPage(extent) < runwayPages; ++extent) {
    if (const metal::AllocationResult allocated = allocateExtent(extent, kv::PageForm::Full);
        !allocated) {
      throw metal::MetalAllocationError(
          std::string("unable to allocate the KV runway: ") +
              metal::allocationFailureName(allocated.failure),
          allocated.failure);
    }
  }
}

KvPageAcquisition KvPool::acquirePages(uint32_t count, kv::PageForm form) {
  KvPageAcquisition acquired = takeFreePages(count, form);
  for (uint32_t page : acquired.pages) {
    markUsed(page);
    pages_[page].activeReferences = 1;
    ++formRecord(form).activePages;
  }
  return acquired;
}

KvPageAcquisition KvPool::takeFreePages(uint32_t count, kv::PageForm form) {
  if (!count)
    return {};

  std::vector<uint32_t> selected;
  selected.reserve(count);
  auto returnSelected = [&] {
    for (auto p = selected.rbegin(); p != selected.rend(); ++p)
      insertFree(*p);
  };
  while (selected.size() < count) {
    if (formRecord(form).freePages) {
      selected.push_back(popFree(form));
      continue;
    }
    // Page ids cover every extent the hard budget could hold, so the budget
    // refuses growth before they run out.
    const uint32_t extent = unallocatedExtent();
    if (extent == noIndex) {
      returnSelected();
      throw std::logic_error("KV page ids ran out before the memory budget did");
    }
    const metal::AllocationResult allocated = allocateExtent(extent, form);
    if (!allocated) {
      // The extents this acquisition allocated stay, reclaimable: the
      // budget admitted them, and the retry that follows a reclaim takes
      // their pages first instead of allocating them again. A reclaim pass
      // returns them if they stay unused.
      returnSelected();
      return {{}, allocated.failure};
    }
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
    ++formOf(page).prefixPages;
  } else if (!record.activeReferences++) {
    ++formOf(page).activePages;
  }
}

void KvPool::releasePage(uint32_t page, bool prefixOwner) {
  PageRecord &record = pages_.at(page);
  if (prefixOwner ? !record.prefixOwned : !record.activeReferences)
    throw std::logic_error("invalid KV page release");
  if (prefixOwner) {
    record.prefixOwned = false;
    --formOf(page).prefixPages;
  } else if (!--record.activeReferences) {
    --formOf(page).activePages;
  }
  if (pageFree(page))
    markFree(page);
}

void KvPool::clearPage(uint32_t page) {
  const PageRecord &record = pages_.at(page);
  if (record.activeReferences != 1 || record.prefixOwned)
    throw std::logic_error("only a page its requester holds alone can be cleared");
  storage_.clearPages({&page, 1});
}

uint32_t KvPool::pageCount() const noexcept {
  return static_cast<uint32_t>(pages_.size());
}

uint64_t KvPool::bytesPerPage(kv::PageForm form) const noexcept {
  return storage_.bytesPerPage(form);
}

uint32_t KvPool::freePageCount(kv::PageForm form) const noexcept {
  return formRecord(form).freePages;
}

kv::PageForm KvPool::form(uint32_t page) const {
  static_cast<void>(pages_.at(page));
  const ExtentRecord &extent = extents_[extentOf(page)];
  if (!extent.allocated) {
    throw std::logic_error("KV page " + std::to_string(page) +
                           " is in an extent that is not allocated");
  }
  return extent.form;
}

KvPageMoveResult KvPool::movePages(std::span<const uint32_t> pages, kv::PageForm to) {
  std::vector<uint32_t> moving(pages.begin(), pages.end());
  std::ranges::sort(moving);
  if (std::ranges::adjacent_find(moving) != moving.end())
    throw std::logic_error("a KV page cannot move to the other form twice at once");
  for (const uint32_t page : moving) {
    if (form(page) == to || pageFree(page))
      throw std::logic_error("KV page " + std::to_string(page) +
                             " is free or of the form it would move to");
  }
  const KvPageAcquisition destinations = takeFreePages(static_cast<uint32_t>(moving.size()), to);
  if (!destinations.granted())
    return {{}, destinations.failure};

  KvPageMoves moves;
  moves.moved.reserve(moving.size());
  for (size_t index = 0; index < moving.size(); ++index)
    moves.moved.push_back({moving[index], destinations.pages[index]});
  try {
    storage_.copyPages(moves.moved);
  } catch (...) {
    for (auto page = destinations.pages.rbegin(); page != destinations.pages.rend(); ++page)
      insertFree(*page);
    throw;
  }
  for (const kv::PageCopy &copy : moves.moved)
    moveReferences(copy);
  return {std::move(moves)};
}

bool KvPool::pageActive(uint32_t page) const {
  return pages_.at(page).activeReferences != 0;
}

bool KvPool::pageFree(uint32_t page) const {
  const PageRecord &record = pages_.at(page);
  return !record.activeReferences && !record.prefixOwned;
}

uint64_t KvPool::allocatedBytes() const noexcept {
  uint64_t bytes = 0;
  for (const kv::PageForm form : kForms)
    bytes += formSnapshot(form).allocatedBytes;
  return bytes;
}

uint64_t KvPool::reclaimEmptyExtents(bool keepRunway, uint32_t limit, std::optional<kv::PageForm> form) {
  uint64_t reclaimedBytes = 0;
  for (const kv::PageForm each : kForms) {
    if (form && *form != each)
      continue;
    bool kept = false;
    uint32_t extent = formRecord(each).reclaimableExtents.head;
    while (extent != noIndex && limit) {
      const uint32_t next = extents_[extent].nextReclaimable;
      if (keepRunway && !kept) {
        kept = true;
      } else {
        releaseExtent(extent);
        reclaimedBytes += uint64_t{extentPages_} * bytesPerPage(each);
        --limit;
      }
      extent = next;
    }
  }
  return reclaimedBytes;
}

metal::AllocationResult KvPool::fillRunway(kv::PageForm form) {
  if (formRecord(form).reclaimableExtents.count)
    return {};
  const uint32_t extent = unallocatedExtent();
  if (extent == noIndex)
    throw std::logic_error("KV page ids ran out before the memory budget did");
  return allocateExtent(extent, form);
}

metal::AllocationResult KvPool::allocateExtent(uint32_t extent, kv::PageForm form) {
  const auto start = AwakeClock::now();
  const metal::AllocationResult allocated = storage_.allocateExtent(extent, form);
  if (allocated) {
    ++extentAllocations_;
    extentAllocateMaxMilliseconds_ =
        std::max(extentAllocateMaxMilliseconds_, millisecondsSince(start));
    extents_[extent].form = form;
    setExtentAllocated(extent, true);
  }
  return allocated;
}

void KvPool::releaseExtent(uint32_t extent) {
  const auto start = AwakeClock::now();
  storage_.releaseExtent(extent);
  ++extentReleases_;
  extentReleaseMaxMilliseconds_ =
      std::max(extentReleaseMaxMilliseconds_, millisecondsSince(start));
  setExtentAllocated(extent, false);
}

uint32_t KvPool::compactableExtent(kv::PageForm form, std::span<const uint32_t> fixed) const noexcept {
  // Pages move only into extents of their form that hold pages already: an
  // empty extent is released as it is, never filled to release another.
  uint32_t freeInUse = 0;
  for (const ExtentRecord &extent : extents_) {
    if (extent.allocated && extent.form == form && extent.usedPages)
      freeInUse += extent.freePages.count;
  }
  uint32_t emptied = noIndex;
  for (uint32_t index = 0; index < extents_.size(); ++index) {
    const ExtentRecord &extent = extents_[index];
    if (!extent.allocated || extent.form != form || !extent.usedPages ||
        extent.usedPages > freeInUse - extent.freePages.count ||
        (emptied != noIndex && extent.usedPages >= extents_[emptied].usedPages))
      continue;
    if (std::none_of(fixed.begin(), fixed.end(),
                     [&](uint32_t page) { return extentOf(page) == index; }))
      emptied = index;
  }
  return emptied;
}

KvPageMoves KvPool::compactExtent(std::span<const uint32_t> fixed, std::optional<kv::PageForm> form) {
  uint32_t emptied = noIndex;
  for (const kv::PageForm each : kForms) {
    if (emptied == noIndex && (!form || *form == each) && formRecord(each).allocatedExtents)
      emptied = compactableExtent(each, fixed);
  }
  if (emptied == noIndex)
    return {};

  std::vector<uint32_t> receivers;
  for (uint32_t index = 0; index < extents_.size(); ++index) {
    const ExtentRecord &extent = extents_[index];
    if (index != emptied && extent.allocated && extent.form == extents_[emptied].form &&
        extent.usedPages && extent.freePages.count)
      receivers.push_back(index);
  }
  std::stable_sort(receivers.begin(), receivers.end(),
                   [&](uint32_t left, uint32_t right) {
                     return extents_[left].usedPages > extents_[right].usedPages;
                   });
  const uint32_t first = firstPage(emptied);
  KvPageMoves moves;
  moves.moved.reserve(extents_[emptied].usedPages);
  auto receiver = receivers.begin();
  uint32_t to = extents_[*receiver].freePages.head;
  for (uint32_t page = first; page < first + extentPages_; ++page) {
    if (pageFree(page))
      continue;
    // The receivers' free pages cover the extent, so one always follows.
    while (to == noIndex)
      to = extents_[*++receiver].freePages.head;
    moves.moved.push_back({page, to});
    to = pages_[to].nextFree;
  }

  storage_.copyPages(moves.moved);
  for (const kv::PageCopy &copy : moves.moved)
    moveReferences(copy);
  ++extentCompactions_;
  pagesMoved_ += moves.moved.size();
  return moves;
}

KvPoolSnapshot KvPool::snapshot() const {
  KvPoolSnapshot result;
  result.index = formSnapshot(kv::PageForm::Index);
  result.slim = formSnapshot(kv::PageForm::Slim);
  for (const kv::PageForm form : kForms) {
    const KvPoolFormSnapshot each = formSnapshot(form);
    result.pagesAllocated += each.pagesAllocated;
    result.pagesActive += each.pagesActive;
    result.pagesPrefix += each.pagesPrefix;
    result.pagesFree += each.pagesFree;
    result.allocatedBytes += each.allocatedBytes;
    result.reclaimableBytes += each.reclaimableBytes;
  }
  result.extentAllocations = extentAllocations_;
  result.extentReleases = extentReleases_;
  result.extentAllocateMaxMilliseconds = extentAllocateMaxMilliseconds_;
  result.extentReleaseMaxMilliseconds = extentReleaseMaxMilliseconds_;
  result.extentCompactions = extentCompactions_;
  result.pagesMoved = pagesMoved_;
  return result;
}

KvPoolFormSnapshot KvPool::formSnapshot(kv::PageForm form) const noexcept {
  const FormRecord &record = formRecord(form);
  const uint64_t extentBytes = uint64_t{extentPages_} * bytesPerPage(form);
  return {record.allocatedExtents * extentPages_,
          record.freePages,
          record.activePages,
          record.prefixPages,
          record.allocatedExtents * extentBytes,
          record.reclaimableExtents.count * extentBytes};
}

uint32_t KvPool::unallocatedExtent() const noexcept {
  for (uint32_t extent = 0; extent < extents_.size(); ++extent) {
    if (!extents_[extent].allocated)
      return extent;
  }
  return noIndex;
}

void KvPool::moveReferences(const kv::PageCopy &copy) noexcept {
  PageRecord &from = pages_[copy.from];
  PageRecord &to = pages_[copy.to];
  FormRecord &fromForm = formOf(copy.from);
  FormRecord &toForm = formOf(copy.to);
  markUsed(copy.to);
  if (from.activeReferences) {
    --fromForm.activePages;
    ++toForm.activePages;
  }
  if (from.prefixOwned) {
    --fromForm.prefixPages;
    ++toForm.prefixPages;
  }
  to.activeReferences = std::exchange(from.activeReferences, 0);
  to.prefixOwned = std::exchange(from.prefixOwned, false);
  markFree(copy.from);
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
  FormRecord &form = formOf(page);
  ++form.freePages;
  form.fillingExtent = noIndex;
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
  FormRecord &form = formOf(page);
  if (!list.count || !form.freePages)
    std::terminate();
  --list.count;
  --form.freePages;
}

uint32_t KvPool::popFree(kv::PageForm form) noexcept {
  const uint32_t page = extents_[fillingExtent(form)].freePages.head;
  if (page == noIndex)
    std::terminate();
  removeFree(page);
  return page;
}

// The allocated extent of the form with the most live pages that still has
// a free page; ties go to the lowest index, and empty extents lose to any
// used one. The answer only changes when another extent of the form gains or
// loses a page, so it is reused until then.
uint32_t KvPool::fillingExtent(kv::PageForm form) noexcept {
  FormRecord &record = formRecord(form);
  if (const uint32_t filling = record.fillingExtent;
      filling != noIndex && extents_[filling].form == form && extents_[filling].freePages.count)
    return filling;
  uint32_t best = noIndex;
  for (uint32_t index = 0; index < extents_.size(); ++index) {
    const ExtentRecord &extent = extents_[index];
    if (extent.form == form && extent.freePages.count &&
        (best == noIndex || extent.usedPages > extents_[best].usedPages)) {
      best = index;
    }
  }
  if (best == noIndex)
    std::terminate();
  record.fillingExtent = best;
  return best;
}

void KvPool::markUsed(uint32_t page) noexcept {
  if (!pageFree(page))
    std::terminate();
  const uint32_t extentIndex = extentOf(page);
  if (pages_[page].onFreeList) {
    removeFree(page);
    if (FormRecord &form = formOf(page); extentIndex != form.fillingExtent)
      form.fillingExtent = noIndex;
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
  FormRecord &form = formRecord(extent.form);
  const uint32_t first = firstPage(extentIndex);
  if (allocated) {
    extent.allocated = true;
    ++form.allocatedExtents;
    for (uint32_t page = first + extentPages_; page > first;)
      insertFree(--page);
    setExtentReclaimable(extentIndex, true);
    return;
  }
  setExtentReclaimable(extentIndex, false);
  for (uint32_t page = first; page < first + extentPages_; ++page)
    removeFree(page);
  if (!form.allocatedExtents)
    std::terminate();
  extent.allocated = false;
  --form.allocatedExtents;
}

void KvPool::setExtentReclaimable(uint32_t extentIndex,
                                  bool reclaimable) noexcept {
  ExtentRecord &extent = extents_[extentIndex];
  if (extent.reclaimable == reclaimable)
    return;
  IndexList &list = formRecord(extent.form).reclaimableExtents;
  if (reclaimable) {
    if (!extent.allocated || extent.usedPages)
      std::terminate();
    extent.previousReclaimable = noIndex;
    extent.nextReclaimable = list.head;
    if (list.head != noIndex) {
      extents_[list.head].previousReclaimable = extentIndex;
    }
    list.head = extentIndex;
    ++list.count;
    extent.reclaimable = true;
    return;
  }
  if (extent.previousReclaimable == noIndex) {
    if (list.head != extentIndex)
      std::terminate();
    list.head = extent.nextReclaimable;
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
  if (!list.count)
    std::terminate();
  --list.count;
}

} // namespace splash::engine
