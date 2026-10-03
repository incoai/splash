#include "TestChecks.hpp"
#include "TestKvPool.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using splash::engine::KvPool;
using splash::metal::AllocationFailure;
using splash::test::TestKvStorage;

using splash::test::require;

template <typename Error, typename Function>
void requireThrows(Function &&function, const char *message) {
    try {
        function();
    } catch (const Error &) {
        return;
    }
    throw std::runtime_error(message);
}

void release(KvPool &pool, const std::vector<uint32_t> &pages,
             bool prefix = false) {
    for (uint32_t page : pages) pool.releasePage(page, prefix);
}

// Every test pool has 4 pages of 100 bytes per extent.
constexpr uint64_t extentBytes = 4 * 100;

uint64_t reclaimEvery(KvPool &pool, bool keepRunway) {
    return pool.reclaimEmptyExtents(keepRunway, std::numeric_limits<uint32_t>::max());
}

void testGrowthPacksAllocatedExtents() {
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
    requireThrows<std::invalid_argument>([&] { KvPool longPool(blocked, 17); },
                                         "a runway longer than the pool was accepted");
}

// Page ids cover every extent the budget could hold, so running out of them
// is a broken invariant rather than a refusal.
void testIdExhaustionIsALogicError() {
    TestKvStorage storage(8, 100, 4);
    KvPool pool(storage, 8);
    requireThrows<std::logic_error>(
        [&] { static_cast<void>(pool.acquirePages(9)); },
        "running out of page ids was not a logic error");
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
    requireThrows<std::logic_error>([&] { static_cast<void>(reclaimEvery(pool, false)); },
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
    requireThrows<std::logic_error>([&] { pool.retainPage(page, true); },
                                    "the cache owned a page twice");
    pool.releasePage(page, true);
    requireThrows<std::logic_error>([&] { pool.releasePage(page, true); },
                                    "the cache gave up a page it did not own");
    require(pool.snapshot().pagesPrefix == 0 && pool.pageActive(page),
            "a refused claim or release changed the page's references");
    pool.releasePage(page, false);
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
    require(!moves.empty() && moves.firstPage == 0 &&
                moves.destinations.size() == 4 && moves.follow(3) == 7 &&
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
    require(moves.firstPage == 8 && moves.follow(8) == 3 && moves.follow(9) == 7 &&
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
    KvPool packed = held(tight, {2, 3, 7});
    require(packed.compactExtent({}).empty() && tight.copies.empty(),
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
    require(moves.firstPage == 12 && moves.follow(12) == 3 && moves.follow(13) == 7 &&
                moves.follow(9) == 9 && pool.pageActive(9),
            "compaction did not pass over the extent with a fixed page");
}

// A storage that cannot copy now, while a command is in flight, throws
// before anything moved.
void testCompactionMovesNothingWhenTheStorageRefuses() {
    TestKvStorage storage(12, 100, 4);
    KvPool pool = held(storage, {0, 1, 2, 7, 10, 11});
    storage.commandInFlight = [] { return true; };
    bool threw = false;
    try {
        static_cast<void>(pool.compactExtent({}));
    } catch (const std::logic_error &) {
        threw = true;
    }
    const auto status = pool.snapshot();
    require(threw && pool.pageActive(3) && !pool.pageActive(7) &&
                status.pagesActive == 6 && status.pagesFree == 6 &&
                status.reclaimableBytes == 0 && status.extentCompactions == 0,
            "a refused copy left pages moved");
    storage.commandInFlight = nullptr;
    require(pool.compactExtent({}).follow(3) == 7,
            "the pool did not compact after the storage refused");
}

}  // namespace

int main() {
    try {
        testGrowthPacksAllocatedExtents();
        testRunwayIsAllocatedThroughThePool();
        testIdExhaustionIsALogicError();
        testFailedGrowthKeepsItsExtentsForTheRetry();
        testPressureReusesFreePagesAndDeniesGrowth();
        testPassReleasesEveryEmptyExtent();
        testFullestExtentFillsFirstSoColdExtentsDrain();
        testRefusedReleaseChangesNothing();
        testPrefixAndActiveReferencesHoldTheExtent();
        testCacheOwnsAPageOnce();
        testCompactionEmptiesTheExtentWithTheFewestPages();
        testCompactionFillsTheFullestExtentsFirst();
        testCompactionNeedsFreePagesInExtentsInUse();
        testCompactionLeavesFixedPagesInPlace();
        testCompactionMovesNothingWhenTheStorageRefuses();
        std::cout << "elastic KV pool tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "elastic KV pool test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
