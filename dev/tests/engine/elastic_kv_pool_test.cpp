#include "TestChecks.hpp"
#include "TestKvPool.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using splash::engine::KvPool;
using splash::kv::PageCopy;
using splash::kv::PageForm;
using splash::metal::AllocationFailure;
using splash::test::TestKvStorage;

using splash::test::rejects;
using splash::test::require;

void release(KvPool &pool, const std::vector<uint32_t> &pages,
             bool prefix = false) {
    for (uint32_t page : pages) pool.releasePage(page, prefix);
}

// Every test pool has 4 pages of 100 bytes per extent, and with an index 4
// pages of 10 bytes per extent of Index pages.
constexpr uint64_t extentBytes = 4 * 100;
constexpr uint64_t indexExtentBytes = 4 * 10;

uint64_t reclaimEvery(KvPool &pool, bool keepRunway) {
    return pool.reclaimEmptyExtents(keepRunway, std::numeric_limits<uint32_t>::max());
}

void testGrowthFillsAllocatedExtents() {
    TestKvStorage storage(12, 100, 4);
    KvPool pool(storage, 4);
    auto pages = pool.acquirePages(5);
    require(pages.granted() && pages.pages.size() == 5,
            "elastic pool did not acquire requested pages");
    require(pages.pages[0] == 0 && pages.pages[3] == 3 &&
                pages.pages[4] == 4,
            "elastic pool did not fill its allocated runway first");
    auto live = pool.snapshot();
    require(live.pagesAllocated == 8 && live.pagesActive == 5 &&
                live.pagesFree == 3 &&
                live.allocatedBytes == 800,
            "elastic growth accounting is incorrect");

    release(pool, pages.pages);
    require(reclaimEvery(pool, true) == extentBytes,
            "reclaim did not retain exactly one warm runway");
    auto reclaimed = pool.snapshot();
    require(reclaimed.pagesAllocated == 4 &&
                reclaimed.reclaimableBytes == extentBytes &&
                storage.releasedExtents == 1,
            "empty extent was not returned exactly");
    require(reclaimed.extentAllocations == 2 && reclaimed.extentReleases == 1,
            "the pool did not count its runway, growth and release");
}

// The runway is allocated through the pool like every later extent, so the
// pool's counts cover all the storage holds. A runway the budget refuses
// fails with its cause, and one longer than the pool is invalid.
void testRunwayIsAllocatedThroughThePool() {
    TestKvStorage storage(16, 100, 4);
    KvPool pool(storage, 6);
    auto status = pool.snapshot();
    require(status.extentAllocations == 2 && status.pagesAllocated == 8 &&
                status.reclaimableBytes == 2 * extentBytes && storage.allocated(0) &&
                storage.allocated(1) && !storage.allocated(2),
            "the runway was not the extents of its pages, allocated through the pool");
    require(reclaimEvery(pool, true) == extentBytes, "an empty runway extent was not released");
    status = pool.snapshot();
    require(status.extentAllocations - status.extentReleases == 1 &&
                status.pagesAllocated == 4 && storage.allocatedPages() == 4,
            "the pool's extent counts differ from the extents allocated");

    TestKvStorage blocked(16, 100, 4);
    blocked.growthBlocked = true;
    bool refused = false;
    try {
        KvPool refusedPool(blocked, 4);
    } catch (const splash::metal::MetalAllocationError &error) {
        refused = error.failure() == AllocationFailure::EngineBudget;
    }
    require(refused, "a refused runway did not fail with the budget's cause");
    rejects([&] { KvPool longPool(blocked, 17); }, "the KV runway exceeds the page pool",
            "a runway longer than the pool was accepted");
}

// Page ids cover every extent the budget could hold, so running out of them
// is a broken invariant rather than a refusal, and takes no page.
void testIdExhaustionIsALogicError() {
    TestKvStorage storage(8, 100, 4);
    KvPool pool(storage, 8);
    rejects([&] { static_cast<void>(pool.acquirePages(9)); },
            "KV page ids ran out before the memory budget did",
            "running out of page ids was not a logic error");
    require(pool.freePageCount() == 8 && pool.acquirePages(8).granted(),
            "running out of page ids kept the pages it had taken");
}

// An extent the storage refuses denies the acquisition, which holds no page
// and keeps the extents it allocated, reclaimable: the retry takes their
// pages instead of allocating them again, and a reclaim pass returns them if
// nothing does.
void testFailedGrowthKeepsItsExtentsForTheRetry() {
    TestKvStorage storage(16, 100, 4);
    storage.growthAllowed = [](uint32_t extent) { return extent != 2; };
    KvPool pool(storage, 0);
    const auto before = pool.snapshot().extentReleases;
    auto pages = pool.acquirePages(9);
    auto status = pool.snapshot();
    require(!pages.granted() && pages.failure == AllocationFailure::EngineBudget &&
                storage.allocationAttempts == 3 &&
                storage.releasedExtents == 0 &&
                pool.snapshot().extentReleases == before &&
                status.pagesAllocated == 8 && status.reclaimableBytes == 2 * extentBytes &&
                status.pagesFree == 8 && status.pagesActive == 0 &&
                status.pagesPrefix == 0 && status.extentAllocations == 2,
            "a failed acquisition was not denied, held pages or did not "
            "keep its extents");
    storage.growthAllowed = nullptr;
    pages = pool.acquirePages(9);
    require(pages.granted() && pool.snapshot().extentAllocations == 3 &&
                storage.releasedExtents == 0,
            "the retry allocated again the extents it was denied with");
    release(pool, pages.pages);
    require(reclaimEvery(pool, false) == 3 * extentBytes &&
                pool.snapshot().pagesAllocated == 0,
            "a reclaim pass did not return the extents the retry left");
}

void testPressureReusesFreePagesAndDeniesGrowth() {
    TestKvStorage storage(8, 100, 4);
    KvPool pool(storage, 4);
    auto active = pool.acquirePages(2);
    require(active.granted() && active.pages.size() == 2,
            "pressure setup did not acquire active pages");
    storage.growthBlocked = true;
    auto reused = pool.acquirePages(1);
    require(reused.granted() && reused.pages.size() == 1 &&
                reused.pages.front() == 2,
            "critical pressure rejected a free page of an allocated extent");
    auto denied = pool.acquirePages(2);
    require(!denied.granted() && denied.failure == AllocationFailure::EngineBudget,
            "critical pressure admitted a new extent");
    require(pool.pageActive(active.pages[0]) && pool.pageActive(active.pages[1]) &&
                pool.pageActive(reused.pages.front()) &&
                pool.snapshot().pagesActive == 3,
            "critical pressure corrupted existing active references");
    release(pool, reused.pages);
    release(pool, active.pages);
    require(reclaimEvery(pool, false) == extentBytes &&
                pool.snapshot().pagesAllocated == 0,
            "pressure cleanup did not reclaim the empty extent");
}

// One pass releases every empty extent but the runway, however many there
// are; a pass without the runway releases that one too.
void testPassReleasesEveryEmptyExtent() {
    constexpr uint32_t extents = 200;
    TestKvStorage storage(4 * extents, 100, 4);
    KvPool pool(storage, 4 * extents);
    auto pages = pool.acquirePages(4 * extents);
    require(pages.granted() && pool.snapshot().pagesAllocated == 4 * extents,
            "release setup did not acquire every page");
    release(pool, pages.pages);
    require(pool.snapshot().reclaimableBytes == extents * extentBytes,
            "every empty extent was not reclaimable");
    const auto before = pool.snapshot().extentReleases;
    require(reclaimEvery(pool, true) == (extents - 1) * extentBytes &&
                storage.releasedExtents == extents - 1 &&
                pool.snapshot().extentReleases == before + extents - 1 &&
                pool.snapshot().reclaimableBytes == extentBytes,
            "a pass did not release every empty extent but the runway");
    require(reclaimEvery(pool, false) == extentBytes &&
                pool.snapshot().pagesAllocated == 0 &&
                pool.snapshot().extentReleases == extents,
            "a pass without the runway did not release it");
}

void testFullestExtentFillsFirstSoColdExtentsDrain() {
    TestKvStorage storage(12, 100, 4);
    KvPool pool(storage, 12);
    auto all = pool.acquirePages(12);
    require(all.granted() && all.pages.size() == 12,
            "fill setup did not acquire every page");
    // Leave extent 0 with three holes, extent 1 with one and extent 2 with two.
    release(pool, {0, 1, 2, 5, 8, 9});
    require(pool.snapshot().pagesFree == 6 &&
                pool.snapshot().reclaimableBytes == 0,
            "partial release accounting is incorrect");

    // New pages come from the fullest extents; the coldest keeps its holes.
    auto refill = pool.acquirePages(2);
    require(refill.granted() && refill.pages.size() == 2 &&
                refill.pages[0] == 5 &&
                (refill.pages[1] == 8 || refill.pages[1] == 9),
            "refill did not take pages from the fullest extents first");
    auto again = pool.acquirePages(1);
    require(again.granted() && again.pages.front() / 4 == 2,
            "allocation did not continue with the fullest extent");

    // Its last page going cold empties the extent so it can be released.
    release(pool, {3});
    require(pool.snapshot().reclaimableBytes == extentBytes &&
                reclaimEvery(pool, false) == extentBytes &&
                pool.snapshot().pagesAllocated == 8 &&
                storage.releasedExtents == 1,
            "drained extent was not released");
}

// A release the storage refuses, as PageStorage refuses one while a command
// is in flight, leaves the pool's record of the extent as it was.
void testRefusedReleaseChangesNothing() {
    TestKvStorage storage(8, 100, 4);
    KvPool pool(storage, 8);
    storage.commandInFlight = [] { return true; };
    rejects([&] { static_cast<void>(reclaimEvery(pool, false)); },
            "cannot release a KV extent while a command is in flight",
            "an extent was released while a command was in flight");
    const auto status = pool.snapshot();
    require(status.pagesAllocated == 8 && status.pagesFree == 8 &&
                status.reclaimableBytes == 2 * extentBytes && status.extentReleases == 0 &&
                storage.allocatedPages() == 8,
            "a refused release changed the pool's record of its extent");
    storage.commandInFlight = nullptr;
    require(reclaimEvery(pool, false) == 2 * extentBytes && pool.snapshot().pagesAllocated == 0,
            "the extents a refused release kept were not released afterwards");
}

void testPrefixAndActiveReferencesHoldTheExtent() {
    TestKvStorage storage(8, 100, 4);
    KvPool pool(storage, 4);
    auto active = pool.acquirePages(1);
    require(active.granted(), "shared reference setup failed");
    pool.retainPage(active.pages.front(), true);
    pool.releasePage(active.pages.front(), false);
    require(reclaimEvery(pool, false) == 0 &&
                pool.snapshot().pagesPrefix == 1,
            "prefix-owned extent was reclaimed while live");
    pool.releasePage(active.pages.front(), true);
    require(reclaimEvery(pool, false) == extentBytes,
            "last prefix release did not make extent reclaimable");
}

// The cache owns a page once, for the one block on it: a second claim or a
// second release is a broken invariant, refused before anything changes.
void testCacheOwnsAPageOnce() {
    TestKvStorage storage(4, 100, 4);
    KvPool pool(storage, 4);
    auto active = pool.acquirePages(1);
    require(active.granted(), "ownership setup failed");
    const uint32_t page = active.pages.front();
    pool.retainPage(page, true);
    require(pool.snapshot().pagesPrefix == 1, "the cache did not own the page");
    rejects([&] { pool.retainPage(page, true); }, "KV page already belongs to a cached block",
            "the cache owned a page twice");
    pool.releasePage(page, true);
    rejects([&] { pool.releasePage(page, true); }, "invalid KV page release",
            "the cache gave up a page it did not own");
    require(pool.snapshot().pagesPrefix == 0 && pool.pageActive(page),
            "a refused claim or release changed the page's references");
    pool.releasePage(page, false);
}

// A page is cleared only while its requester holds it alone: a second
// request or the cache may already read it.
void testOnlyAPageItsRequesterHoldsAloneIsCleared() {
    TestKvStorage storage(4, 100, 4);
    KvPool pool(storage, 4);
    auto acquired = pool.acquirePages(2);
    require(acquired.granted(), "clearing setup failed");
    const uint32_t alone = acquired.pages[0];
    const uint32_t shared = acquired.pages[1];
    storage.content[alone] = 7;
    storage.content[shared] = 8;
    pool.clearPage(alone);
    require(storage.content[alone] == 0 && storage.cleared == std::vector<uint32_t>{alone},
            "the page its requester holds alone was not cleared");
    pool.retainPage(shared, false);
    rejects([&] { pool.clearPage(shared); },
            "only a page its requester holds alone can be cleared",
            "a page two requests hold was cleared");
    pool.releasePage(shared, false);
    pool.retainPage(shared, true);
    rejects([&] { pool.clearPage(shared); },
            "only a page its requester holds alone can be cleared",
            "a page the cache holds was cleared");
    require(storage.content[shared] == 8 && storage.cleared.size() == 1,
            "a refused clear changed the page");
    pool.releasePage(shared, true);
    release(pool, acquired.pages);
}

// The pool with every page allocated and held by a request, less the pages
// in `released`. Every page starts with content of its own.
KvPool held(TestKvStorage &storage, const std::vector<uint32_t> &released) {
    for (uint32_t page = 0; page < storage.pageCount(); ++page)
        storage.content[page] = 100 + page;
    KvPool pool(storage, storage.pageCount());
    auto all = pool.acquirePages(storage.pageCount());
    require(all.granted(), "compaction setup did not acquire every page");
    release(pool, released);
    return pool;
}

// The free pages of the extents in use cover the extent with the fewest
// pages: its pages move to them, content and references, and it is empty.
void testCompactionEmptiesTheExtentWithTheFewestPages() {
    TestKvStorage storage(12, 100, 4);
    // Extent 0 keeps page 3, extent 1 pages 4 to 6, extent 2 pages 8 and 9.
    KvPool pool = held(storage, {0, 1, 2, 7, 10, 11});
    pool.retainPage(3, true);
    const auto moves = pool.compactExtent({});
    require(moves.moved == std::vector<PageCopy>{{3, 7}} && moves.follow(3) == 7 &&
                moves.follow(0) == 0 && moves.follow(8) == 8,
            "compaction did not move the emptiest extent's page to the fullest one");
    require(storage.copies.size() == 1 && storage.copies[0].from == 3 &&
                storage.copies[0].to == 7 && storage.content[7] == 103,
            "the moved page's content did not follow it");
    require(!pool.pageActive(3) && pool.pageActive(7),
            "the moved page's references did not follow it");
    const auto status = pool.snapshot();
    require(status.pagesActive == 6 && status.pagesPrefix == 1 &&
                status.pagesFree == 6 && status.pagesAllocated == 12 &&
                status.reclaimableBytes == extentBytes && status.extentCompactions == 1 &&
                status.pagesMoved == 1,
            "compaction changed what is held or did not empty its extent");
    // Both references release on the page it moved to.
    pool.releasePage(7, true);
    pool.releasePage(7, false);
    require(!pool.pageActive(7) && pool.snapshot().pagesPrefix == 0,
            "a moved reference was not released where it went");
    require(reclaimEvery(pool, false) == extentBytes &&
                pool.snapshot().pagesAllocated == 8 && storage.releasedExtents == 1,
            "the emptied extent was not released");
}

// Pages go to the fullest extents first and on to the next when one is full.
void testCompactionFillsTheFullestExtentsFirst() {
    TestKvStorage storage(16, 100, 4);
    // Extents 0 and 1 keep three pages each, extents 2 and 3 two each.
    KvPool pool = held(storage, {3, 7, 10, 11, 14, 15});
    const auto moves = pool.compactExtent({});
    require(moves.moved == std::vector<PageCopy>{{8, 3}, {9, 7}} &&
                storage.content[3] == 108 && storage.content[7] == 109,
            "compaction did not fill the fullest extents first");
    // Extents 0 and 1 are full now; extent 3's two pages have nowhere to go.
    require(pool.compactExtent({}).empty() && storage.copies.size() == 2 &&
                pool.snapshot().extentCompactions == 1,
            "compaction moved pages the free pages did not cover");
}

// Free pages count only in extents that hold pages: an empty extent is
// released as it is, never filled to release another.
void testCompactionNeedsFreePagesInExtentsInUse() {
    TestKvStorage storage(12, 100, 4);
    // Extent 0 keeps one page, extent 1 none, extent 2 all four.
    KvPool pool = held(storage, {1, 2, 3, 4, 5, 6, 7});
    require(pool.compactExtent({}).empty() && storage.copies.empty() &&
                pool.snapshot().reclaimableBytes == extentBytes &&
                pool.snapshot().extentCompactions == 0,
            "compaction filled an empty extent");
    // Each extent holds more than the other has free.
    TestKvStorage tight(8, 100, 4);
    KvPool tightPool = held(tight, {2, 3, 7});
    require(tightPool.compactExtent({}).empty() && tight.copies.empty(),
            "compaction moved an extent the free pages did not cover");
}

// An extent with a page that must stay where it is is not emptied; the
// extent with the next fewest pages is.
void testCompactionLeavesFixedPagesInPlace() {
    TestKvStorage storage(16, 100, 4);
    KvPool pool = held(storage, {3, 7, 10, 11, 14, 15});
    const std::vector<uint32_t> everywhere{0, 4, 9, 13};
    require(pool.compactExtent(everywhere).empty() && storage.copies.empty(),
            "compaction emptied an extent that holds a fixed page");
    const std::vector<uint32_t> fixed{9};
    const auto moves = pool.compactExtent(fixed);
    require(moves.moved == std::vector<PageCopy>{{12, 3}, {13, 7}} && moves.follow(9) == 9 &&
                pool.pageActive(9),
            "compaction did not pass over the extent with a fixed page");
}

// A storage that cannot copy now, while a command is in flight, throws
// before anything moved.
void testCompactionMovesNothingWhenTheStorageRefuses() {
    TestKvStorage storage(12, 100, 4);
    KvPool pool = held(storage, {0, 1, 2, 7, 10, 11});
    storage.commandInFlight = [] { return true; };
    rejects([&] { static_cast<void>(pool.compactExtent({})); },
            "cannot copy KV pages while a command is in flight",
            "a compaction went on without the copy the storage refused");
    const auto status = pool.snapshot();
    require(pool.pageActive(3) && !pool.pageActive(7) &&
                status.pagesActive == 6 && status.pagesFree == 6 &&
                status.reclaimableBytes == 0 && status.extentCompactions == 0,
            "a refused copy left pages moved");
    storage.commandInFlight = nullptr;
    require(pool.compactExtent({}).follow(3) == 7,
            "the pool did not compact after the storage refused");
}

// Pages of a form come only from extents of their form, the fullest first.
void testEachFormFillsItsOwnExtents() {
    TestKvStorage storage(24, 100, 4, 10);
    KvPool pool(storage, 0);
    auto index = pool.acquirePages(8, PageForm::Index);
    auto full = pool.acquirePages(4);
    require(index.granted() && full.granted() && index.pages.front() == 0 && full.pages.front() == 8 &&
                pool.form(7) == PageForm::Index && pool.form(8) == PageForm::Full,
            "pages of a form did not come from extents of their own");
    // Index extent 0 keeps one page, Index extent 1 three, Full extent 2 three.
    release(pool, {0, 1, 2, 4, 8});
    require(pool.freePageCount(PageForm::Index) == 4 && pool.freePageCount() == 1,
            "free pages were not counted by form");
    auto refill = pool.acquirePages(2, PageForm::Index);
    auto more = pool.acquirePages(1);
    require(refill.granted() && refill.pages[0] == 4 && refill.pages[1] / 4 == 0 && more.granted() &&
                more.pages[0] == 8,
            "pages did not come from the fullest extent of their form first");
    const auto status = pool.snapshot();
    require(status.pagesAllocated == 12 && status.pagesActive == 10 && status.pagesFree == 2 &&
                status.allocatedBytes == extentBytes + 2 * indexExtentBytes && status.index.pagesAllocated == 8 &&
                status.index.pagesActive == 6 && status.index.pagesFree == 2 &&
                status.index.allocatedBytes == 2 * indexExtentBytes &&
                pool.allocatedBytes() == storage.allocatedBytes() && pool.reusablePages() == 0 &&
                pool.reusablePages(PageForm::Index) == 2,
            "the pool did not account for its pages by form");
    rejects([&] { static_cast<void>(pool.form(12)); }, "KV page 12 is in an extent that is not allocated",
            "a page of an unallocated extent had a form");
}

// An extent released and allocated again holds pages of the form it is
// allocated for this time.
void testAnExtentTakesTheFormOfEachAllocation() {
    TestKvStorage storage(8, 100, 4, 10);
    KvPool pool(storage, 0);
    auto index = pool.acquirePages(1, PageForm::Index);
    require(index.granted(), "re-forming setup failed");
    release(pool, index.pages);
    require(reclaimEvery(pool, false) == indexExtentBytes, "the Index extent was not released");
    auto full = pool.acquirePages(1);
    auto spill = pool.acquirePages(1, PageForm::Index);
    require(full.granted() && full.pages[0] / 4 == 0 && pool.form(full.pages[0]) == PageForm::Full &&
                spill.granted() && spill.pages[0] / 4 == 1 && pool.form(spill.pages[0]) == PageForm::Index,
            "an extent kept the form of an earlier allocation");
}

// A page moves to the other form with its content and references, the
// cache's too, and the page it leaves is free; it moves back the same way.
void testPagesMoveToTheOtherFormAndBack() {
    TestKvStorage storage(16, 100, 4, 10);
    for (uint32_t page = 0; page < storage.pageCount(); ++page)
        storage.content[page] = 100 + page;
    KvPool pool(storage, 4);
    auto acquired = pool.acquirePages(3);
    require(acquired.granted() && pool.fillRunway(PageForm::Index), "move setup failed");
    // Page 1 belongs to the cache alone, page 2 to two requests.
    pool.retainPage(1, true);
    pool.releasePage(1, false);
    pool.retainPage(2, false);
    const auto spilled = pool.movePages(std::vector<uint32_t>{2, 1}, PageForm::Index);
    const uint32_t one = spilled.moves.follow(1), two = spilled.moves.follow(2);
    require(spilled.granted() && spilled.moves.moved == std::vector<PageCopy>{{1, 4}, {2, 5}} &&
                spilled.moves.follow(0) == 0 && pool.form(one) == PageForm::Index &&
                storage.content[one] == 101 && storage.content[two] == 102,
            "pages did not move to the Index runway with their content");
    const auto status = pool.snapshot();
    require(!pool.pageActive(2) && pool.pageActive(two) && pool.freePageCount() == 3 &&
                pool.freePageCount(PageForm::Index) == 2 && status.pagesActive == 2 && status.pagesPrefix == 1 &&
                status.index.pagesActive == 1 && status.index.pagesPrefix == 1 && status.pagesMoved == 0,
            "a moved page's references did not follow it, or the page it left was not freed");
    const auto restored = pool.movePages(std::vector<uint32_t>{two}, PageForm::Full);
    const uint32_t back = restored.moves.follow(two);
    require(restored.granted() && back / 4 == 0 && pool.form(back) == PageForm::Full &&
                storage.content[back] == 102 && pool.pageActive(back) && !pool.pageActive(two),
            "a page did not move back to a Full page");
    release(pool, {back, back, 0});
    pool.releasePage(one, true);
    require(reclaimEvery(pool, false) == extentBytes + indexExtentBytes && pool.snapshot().pagesAllocated == 0,
            "moved references were not released where they went");
}

// A move is all or nothing: a page it cannot move, an extent the budget
// refuses or a copy the storage refuses moves none of them.
void testRefusedMovesMoveNothing() {
    TestKvStorage storage(16, 100, 4, 10);
    KvPool pool(storage, 4);
    auto held = pool.acquirePages(6);
    require(held.granted() && pool.fillRunway(PageForm::Index), "move setup failed");
    const auto before = pool.snapshot();
    const auto unchanged = [&] {
        const auto now = pool.snapshot();
        return storage.copies.empty() && now.pagesAllocated == before.pagesAllocated &&
               now.pagesActive == before.pagesActive && now.pagesFree == before.pagesFree &&
               now.index.pagesFree == before.index.pagesFree && pool.pageActive(0) && pool.pageActive(5);
    };
    rejects([&] { static_cast<void>(pool.movePages(std::vector<uint32_t>{0, 6}, PageForm::Index)); },
            "KV page 6 is free or of the form it would move to", "a free page moved");
    rejects([&] { static_cast<void>(pool.movePages(std::vector<uint32_t>{0, 8}, PageForm::Index)); },
            "KV page 8 is free or of the form it would move to", "a page moved to its own form");
    rejects([&] { static_cast<void>(pool.movePages(std::vector<uint32_t>{1, 1}, PageForm::Index)); },
            "a KV page cannot move to the other form twice at once", "a page moved twice at once");
    rejects([&] { static_cast<void>(pool.movePages(std::vector<uint32_t>{0, 12}, PageForm::Index)); },
            "KV page 12 is in an extent that is not allocated", "a page of an unallocated extent moved");
    // Five pages need an Index extent past the runway's four pages.
    const std::vector<uint32_t> five{0, 1, 2, 3, 4};
    storage.growthBlocked = true;
    const auto denied = pool.movePages(five, PageForm::Index);
    require(!denied.granted() && denied.failure == AllocationFailure::EngineBudget && denied.moves.empty(),
            "a move the budget refused was granted");
    storage.growthBlocked = false;
    storage.commandInFlight = [] { return true; };
    rejects([&] { static_cast<void>(pool.movePages(std::span(five).first(4), PageForm::Index)); },
            "cannot copy KV pages while a command is in flight", "a move went on without its copy");
    storage.commandInFlight = nullptr;
    require(unchanged(), "a refused move changed the pool");
    require(pool.movePages(five, PageForm::Index).granted() && !pool.pageActive(0) &&
                pool.snapshot().index.pagesActive == 5,
            "the pool did not move the pages a refusal had kept");
}

// Each form keeps a runway: fillRunway() allocates an empty extent of the
// form unless one is, and a reclaim that keeps the runway keeps one empty
// extent of each form.
void testEachFormKeepsARunway() {
    TestKvStorage storage(32, 100, 4, 10);
    KvPool pool(storage, 4);
    const auto allocations = [&] { return pool.snapshot().extentAllocations; };
    require(pool.fillRunway(PageForm::Full) && allocations() == 1, "a second Full runway was allocated");
    require(pool.fillRunway(PageForm::Index) && pool.fillRunway(PageForm::Index) && allocations() == 2 &&
                pool.freePageCount(PageForm::Index) == 4,
            "the Index runway was not allocated once");
    // A page taken from the runway leaves no empty Index extent.
    auto spill = pool.acquirePages(1, PageForm::Index);
    require(spill.granted() && pool.fillRunway(PageForm::Index) && allocations() == 3 &&
                pool.snapshot().index.reclaimableBytes == indexExtentBytes,
            "the Index runway was not refilled once taken");
    auto taken = pool.acquirePages(4, PageForm::Index);
    storage.growthBlocked = true;
    const auto refused = pool.fillRunway(PageForm::Index);
    require(taken.granted() && !refused && refused.failure == AllocationFailure::EngineBudget && allocations() == 3,
            "a refused runway allocated an extent");
    storage.growthBlocked = false;
    release(pool, taken.pages);
    release(pool, spill.pages);
    auto full = pool.acquirePages(5);
    require(full.granted(), "runway setup did not grow the Full extents");
    release(pool, full.pages);
    // Two empty extents of each form.
    require(pool.reclaimEmptyExtents(true, std::numeric_limits<uint32_t>::max(), PageForm::Index) ==
                    indexExtentBytes &&
                pool.snapshot().reclaimableBytes == 2 * extentBytes + indexExtentBytes,
            "a reclaim of Index extents released the Index runway or a Full extent");
    require(reclaimEvery(pool, true) == extentBytes &&
                pool.snapshot().reclaimableBytes == extentBytes + indexExtentBytes,
            "a reclaim that keeps the runway did not keep one empty extent of each form");
    require(pool.reclaimEmptyExtents(false, 1) == extentBytes && reclaimEvery(pool, false) == indexExtentBytes &&
                pool.snapshot().pagesAllocated == 0 && storage.allocatedPages() == 0,
            "a reclaim of one extent did not release a Full extent before an Index one");

    TestKvStorage dense(8, 100, 4);
    KvPool densePool(dense, 4);
    rejects([&] { static_cast<void>(densePool.fillRunway(PageForm::Index)); },
            "test KV pages without an index are all Full", "a pool without an index allocated Index pages");
    rejects([&] { static_cast<void>(densePool.acquirePages(1, PageForm::Index)); },
            "test KV pages without an index are all Full", "a pool without an index acquired Index pages");
    require(densePool.freePageCount() == 4 && densePool.snapshot().extentAllocations == 1,
            "a refused Index acquisition changed the pool");
}

// At the memory wall the Index runway takes a spill without an allocation,
// and the Full pages the spill frees serve the next acquisition.
void testSpillAtTheWallGoesThroughTheIndexRunway() {
    TestKvStorage storage(16, 100, 4, 10);
    storage.budgetPages = 9;
    KvPool pool(storage, 8);
    auto held = pool.acquirePages(8);
    require(held.granted() && pool.fillRunway(PageForm::Index), "wall setup failed");
    const auto denied = pool.acquirePages(1);
    require(!denied.granted() && denied.failure == AllocationFailure::EngineBudget,
            "the pool grew past its budget");
    const auto spilled = pool.movePages(std::span(held.pages).first(4), PageForm::Index);
    const auto grown = pool.acquirePages(4);
    require(spilled.granted() && grown.granted() && pool.snapshot().extentAllocations == 3,
            "a spill at the wall did not free Full pages for the next acquisition");
}

// Compaction moves pages only to extents of their form: an extent whose
// form has no room elsewhere stays, however many pages the other form has
// free.
void testCompactionStaysWithinAForm() {
    TestKvStorage storage(16, 100, 4, 10);
    for (uint32_t page = 0; page < storage.pageCount(); ++page)
        storage.content[page] = 100 + page;
    KvPool pool(storage, 0);
    require(pool.acquirePages(8).granted() && pool.acquirePages(8, PageForm::Index).granted(),
            "compaction setup failed");
    // Full extent 0 keeps page 3, Full extent 1 all four; Index extent 2
    // keeps page 11, Index extent 3 pages 14 and 15.
    release(pool, {0, 1, 2, 8, 9, 10, 12, 13});
    const auto moves = pool.compactExtent({});
    require(moves.moved == std::vector<PageCopy>{{11, 13}} && storage.content[13] == 111 &&
                pool.form(13) == PageForm::Index && pool.pageActive(13),
            "compaction did not move the Index page to the other Index extent");
    require(pool.compactExtent({}).empty() && pool.snapshot().index.reclaimableBytes == indexExtentBytes &&
                reclaimEvery(pool, false) == indexExtentBytes && pool.pageActive(3),
            "compaction moved a page to an extent of the other form");
}

// A compaction for one form empties an extent of that form, however few
// pages an extent of the other holds; one for either empties a Full extent
// before an Index one.
void testCompactionEmptiesAnExtentOfTheFormAskedFor() {
    // Full extents 0 and 1 keep two pages each; Index extent 2 keeps page 8,
    // Index extent 3 pages 12 to 14.
    const auto spread = [](TestKvStorage &storage) {
        KvPool pool(storage, 0);
        require(pool.acquirePages(8).granted() && pool.acquirePages(8, PageForm::Index).granted(),
                "compaction setup failed");
        release(pool, {2, 3, 6, 7, 9, 10, 11, 15});
        return pool;
    };
    const std::vector<PageCopy> full{{0, 7}, {1, 6}}, index{{8, 15}};

    TestKvStorage forFullStorage(16, 100, 4, 10);
    KvPool forFull = spread(forFullStorage);
    require(forFull.compactExtent({}, PageForm::Full).moved == full &&
                forFull.snapshot().reclaimableBytes == extentBytes,
            "a compaction for Full pages did not empty a Full extent");
    require(forFull.compactExtent({}, PageForm::Full).empty() && forFull.pageActive(8) &&
                forFull.snapshot().index.reclaimableBytes == 0,
            "a compaction for Full pages emptied an Index extent");

    TestKvStorage forIndexStorage(16, 100, 4, 10);
    KvPool forIndex = spread(forIndexStorage);
    require(forIndex.compactExtent({}, PageForm::Index).moved == index &&
                forIndex.snapshot().reclaimableBytes == indexExtentBytes,
            "a compaction for Index pages did not empty only an Index extent");

    TestKvStorage eitherStorage(16, 100, 4, 10);
    KvPool either = spread(eitherStorage);
    require(either.compactExtent({}).moved == full && either.compactExtent({}).moved == index &&
                either.snapshot().reclaimableBytes == extentBytes + indexExtentBytes,
            "a compaction for either form did not empty a Full extent before an Index one");
}

}  // namespace

int main() {
    try {
        testGrowthFillsAllocatedExtents();
        testRunwayIsAllocatedThroughThePool();
        testIdExhaustionIsALogicError();
        testFailedGrowthKeepsItsExtentsForTheRetry();
        testPressureReusesFreePagesAndDeniesGrowth();
        testPassReleasesEveryEmptyExtent();
        testFullestExtentFillsFirstSoColdExtentsDrain();
        testRefusedReleaseChangesNothing();
        testPrefixAndActiveReferencesHoldTheExtent();
        testCacheOwnsAPageOnce();
        testOnlyAPageItsRequesterHoldsAloneIsCleared();
        testCompactionEmptiesTheExtentWithTheFewestPages();
        testCompactionFillsTheFullestExtentsFirst();
        testCompactionNeedsFreePagesInExtentsInUse();
        testCompactionLeavesFixedPagesInPlace();
        testCompactionMovesNothingWhenTheStorageRefuses();
        testEachFormFillsItsOwnExtents();
        testAnExtentTakesTheFormOfEachAllocation();
        testPagesMoveToTheOtherFormAndBack();
        testRefusedMovesMoveNothing();
        testEachFormKeepsARunway();
        testSpillAtTheWallGoesThroughTheIndexRunway();
        testCompactionStaysWithinAForm();
        testCompactionEmptiesAnExtentOfTheFormAskedFor();
        std::cout << "elastic KV pool tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "elastic KV pool test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
