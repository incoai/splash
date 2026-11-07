#include "EngineTestSupport.hpp"
#include "FakeElasticPool.hpp"
#include "MemoryPasses.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "TestRegistry.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

using namespace splash;
using namespace splash::engine;
using splash::test::Events;
using splash::test::Executor;
using splash::test::guardReleases;
using splash::test::idle;
using splash::test::OffloadControl;
using splash::test::request;
using splash::test::runUntilIdle;
using splash::test::Spilling;

namespace {

using splash::test::require;

// A start that must spill while the SSD cache takes no writes, its projection past the resident
// context, fails naming the SSD cache, not --max-memory.
void testAClosedTierRefusesAStartPastTheResidentContext() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.writableFile = false;
  engine::Cache cache(pool, &tier, nullptr);
  Executor model;
  Events events;
  EngineConfig config = test::engineConfig();
  config.spill = {.capable = true,
                  .always = true,
                  .denseTokens = 2048,
                  .geometry = {.fullPageBytes = 423'936,
                               .indexPageBytes = 24'576,
                               .slotBytes = 425'984,
                               .residentTokens = 64,
                               .densePages = 64,
                               .windowPages = 64,
                               .groupPages = 256}};
  engine::Engine engine(config, cache, model, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(100, 3)));
  runUntilIdle(engine);
  require(events.failures == std::vector<std::string>{"capacity_exhausted"} &&
              engine.snapshot().spill.refusedClosed == 1 &&
              events.failureDetails.front().first.find("SSD cache") != std::string::npos &&
              events.failureDetails.front().first.find("--max-memory") == std::string::npos,
          "a request past the resident context did not fail naming the closed SSD cache");
}

// A request starts Resident where its projection (output clipped at 4,096 tokens) fits the room beside
// the lanes' growth, even past the resident context, Spill where it does not; dense history never spills.
void testAdmissionSpillsWhatMemoryCannotHold() {
  // Memory holds 200 pages, the resident context 100.
  Spilling spill(1024, false, 100 * 100, 0, {4}, {}, {.limitBytes = 200 * 100});
  // Eight pages of prompt and 32K tokens of output: the output is clipped at
  // 4,096 tokens, so it projects 137 pages, which memory holds.
  spill.engine.submit(Spilling::prompt(1, 1, 8, 32'768));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first request did not decode");
  // Four pages and 4,096 tokens project 133 pages more, which it does not.
  spill.engine.submit(Spilling::prompt(2, 2, 4, 4'096));
  spill.runUntil([&] { return spill.events.startIds.size() == 2; }, "the second request did not start");
  // Half a page of prompt and 8 tokens of output stay within the dense
  // history's page.
  EngineRequest dense = request(3, std::vector<uint32_t>(16, 3));
  dense.maxNewTokens = 8;
  spill.engine.submit(dense);
  spill.runUntil([&] { return spill.events.startIds.size() == 3; }, "the third request did not start");
  std::vector<KvMode> modes;
  for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
    modes.push_back(active.kvMode);
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(modes == std::vector<KvMode>{KvMode::Resident, KvMode::Spill, KvMode::Resident} &&
              counts.resident == 2 && counts.spill == 1,
          "admission did not keep in memory exactly the requests whose projection memory holds");
  spill.engine.cancel(1);
  spill.engine.cancel(2);
  spill.runUntilIdle();
  require(spill.committedSlots() == 0, "a request kept its commitment");
}

// A Spill start commits a slot for each projected page without a copy, its restored chain's
// included, and each page draws its slot as it is written.
void testSpillCommitmentCoversTheChainWithoutCopies() {
  Spilling spill(64);
  // The first turn's pages are written nowhere: the tier takes no writes
  // of Spill requests' pages.
  spill.allowSpillWrites(false);
  spill.engine.submit(Spilling::prompt(1, 1, 4, 1));
  spill.runUntilIdle();
  require(spill.tier.writes.empty() && spill.committedSlots() == 0,
          "the first turn wrote copies or kept its commitment");
  spill.allowSpillWrites(true);
  // The second turn restores the first's four pages and the state at their
  // end, and its projection holds two pages more: six, none with a copy.
  spill.engine.submit(Spilling::prompt(2, 1, 6, 1));
  spill.runUntil([&] { return spill.events.startIds.size() == 2; },
                 "the second turn did not start");
  require(spill.events.starts.back() == 4 * KvCache::pageTokens && spill.committedSlots() == 6,
          "the commitment left out the restored pages that have no copy");
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.tier.writes.size() == 6 &&
              std::ranges::all_of(spill.tier.draws,
                                  [](model::DiskBudget::Draw draw) {
                                    return draw == model::DiskBudget::Draw::Commitment;
                                  }) &&
              spill.committedSlots() == 0 && spill.engine.snapshot().spill.spill == 2,
          "the restored and new pages were not written on the commitment");
}

// A Resident lane at the wall converts where the quota holds its whole history, waits for its
// copies and grows into the freed pages; a quota short of the history refuses it, and it fails.
void testConversionCommitsTheHistoryWithoutCopies() {
  for (const uint32_t slots : {5U, 8U}) {
    Spilling spill(slots, false);
    // Two Full extents beside the Index runway, and one Index extent more.
    spill.storage.budgetPages = 9;
    // Seven pages of prompt and 40 tokens of output reach a ninth page past
    // 248 tokens; the projection then is eight whole pages, none copied.
    spill.engine.submit(Spilling::prompt(1, 1, 7, 40));
    spill.runUntilIdle();
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    if (slots == 5) {
      require(counts.converted == 0 && spill.tier.writes.empty() &&
                  spill.events.failures == std::vector<std::string>{"capacity_exhausted"} &&
                  spill.events.failureDetails.front().first.find("SSD cache") !=
                      std::string::npos,
              "a conversion the quota could not hold with the history went ahead");
    } else {
      require(counts.converted == 1 && counts.resident == 1 &&
                  spill.events.completedCount == 1 && spill.events.usage.at(1).second == 40 &&
                  spill.tier.writes.size() == 8 && spill.storage.allocatedBytes() <= 900,
              "the converted lane did not wait for its spilled pages and grow into them");
    }
    require(spill.committedSlots() == 0, "the lane kept its commitment");
  }
}

// A converting lane whose restored history has copies spills those pages at once and decodes its
// whole output rather than ending at the wall.
void testConversionSpillsAHistoryWithCopies() {
  // The room beside the lanes holds 20 pages, as does the resident context.
  Spilling spill(400, false, 2'000, 0, {4}, {}, {.limitBytes = 2'000});
  // Ten pages and 4,096 tokens of output do not fit: the first turn spills,
  // its pages written through, and is cancelled while it decodes.
  spill.engine.submit(Spilling::prompt(1, 1, 10, 4'096));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first turn did not decode");
  for (int step = 0; step < 50; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
  }
  spill.engine.cancel(1);
  spill.runUntilIdle();
  require(spill.engine.snapshot().spill.spill == 1 && spill.tier.writes.size() >= 10,
          "the first turn did not spill its prompt's pages");
  // With 120 tokens of output the second turn fits and starts Resident: it
  // restores the first's chain whole, with its copies, and meets the wall of
  // the 14 pages memory holds while it decodes (the first turn's spill gave
  // back the Full extents it emptied).
  spill.storage.budgetPages = 14;
  spill.engine.submit(Spilling::prompt(2, 1, 10, 120));
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(counts.converted == 1 && counts.lengthEnds == 0 && spill.events.usage.contains(2) &&
              spill.events.usage.at(2).second == 120 && spill.committedSlots() == 0,
          "the lane whose history had copies did not spill them when it converted");
}

// A converted lane preempted and resumed stays Spill under its output limit when the quota holds its
// projection; a quota short of it bounds the resume, and the bound and the end at it count as the
// converted lane's.
void testAConvertedLaneResumesUnderTheSpillRules() {
  for (const bool quotaHolds : {true, false}) {
    Spilling spill(16, false);
    // Two Full extents beside the Index runway, and one Index extent more.
    spill.storage.budgetPages = 9;
    // Seven pages of prompt and 100 tokens of output: the lane meets the
    // wall past its eighth page and converts.
    EngineRequest background = Spilling::prompt(1, 1, 7, 100);
    background.priority = RequestPriority::Background;
    spill.engine.submit(std::move(background));
    spill.runUntil([&] { return spill.engine.snapshot().spill.converted == 1 && spill.events.outputs[1].size() >= 10; },
                   "the lane did not convert at the wall");
    // A Normal start is refused memory while the lane is resident, which
    // suspends it.
    spill.model.beginGrowthBlocked = [&] {
      return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
    };
    spill.model.resumeDenied = true;
    spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
    spill.runUntil([&] { return spill.events.completedCount == 1; }, "the Normal request did not run");
    require(spill.model.suspensions == 1, "the converted lane was not suspended");
    // Short: the seven pages its conversion copied, whose copies the quota
    // counts again as the resume holds them, and two slots hold 288 tokens,
    // 63 of output.
    if (!quotaHolds)
      spill.budget->setActiveLimitBytes((7 + 2) * 100);
    spill.model.resumeDenied = false;
    spill.runUntil([&] { return spill.model.resumptions == 1; }, "the converted lane did not resume");
    const KvMode resumed = spill.engine.snapshot().activeRequests.front().kvMode;
    spill.runUntilIdle();
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    const uint32_t output = quotaHolds ? 100 : 63;
    require(resumed == KvMode::Spill && spill.events.completedCount == 2 && spill.events.failures.empty() &&
                spill.events.usage.at(1).second == output && spill.model.outputLimits.at(1) == output,
            quotaHolds ? "a converted lane did not resume Spill with its whole output"
                       : "a converted lane's resume was not bounded to what the quota holds");
    require(counts.converted == 1 && counts.convertedLengthEnds == counts.lengthEnds &&
                counts.convertedBounded == counts.bounded && counts.bounded == (quotaHolds ? 0U : 1U) &&
                counts.lengthEnds == counts.bounded,
            "the converted lane's bounded resume or its end at the bound was not counted, as a bound and as the "
            "converted lane's");
    require(spill.committedSlots() == 0, "the lane kept its commitment");
  }
}

// A lane that converts at the wall where a start is not admitted, so that landed moves would keep an
// empty Full extent as the runway, releases every Full extent its moves empty, compacted, and decodes
// without a yield.
void testAConvertedLanesMovesReleaseTheExtentsTheyEmpty() {
  Spilling spill(1024);
  const auto freeFullPages = [&] {
    const KvPoolSnapshot pool = spill.pool.snapshot();
    return pool.pagesFree - pool.index.pagesFree;
  };
  // Seven pages of prompt start Resident while the SSD cache takes no
  // writes, and convert at the wall of the extents they hold.
  spill.tier.writableFile = false;
  spill.engine.submit(Spilling::prompt(1, 1, 7, 400));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the lane did not decode");
  spill.tier.writableFile = true;
  spill.storage.budgetPages = static_cast<uint32_t>(spill.storage.allocatedBytes() / 100);
  spill.runUntil([&] { return spill.engine.snapshot().spill.converted == 1; }, "the lane did not convert");
  require(!spill.host.governor.admits(ClaimClass::Start), "a start was admitted at the wall");
  spill.storage.budgetPages = 256;
  // All its pages but the dense history and the window spill.
  spill.runUntil([&] { return spill.cache.spillSnapshot().spilledPages >= 5 && !spill.engine.commandInFlight(); },
                 "the converted lane's pages did not spill");
  require(freeFullPages() < spill.pool.extentPages(),
          "the converted lane's moves kept a Full extent they emptied at the wall");
  spill.runUntilIdle();
  require(spill.events.completedCount == 1 && spill.events.failedCount == 0 &&
              spill.engine.snapshot().resourceSuspensions == 0,
          "the converted lane failed or yielded");
}

// Flash-Next's proportions: 256-page extents, 423,936 B Full and 24,576 B Index pages, a 390 MiB hot
// block floor, and a dense history and tail window of a page each.
struct FlashNextLanes {
  static constexpr Spilling::Pages kPages{.count = 8192, .fullBytes = 423'936, .indexBytes = 24'576};
  Spilling spill{16'384, false, 0, 200'000, Spilling::Extent{256}, {}, {}, false, kPages};
  FlashNextLanes() { spill.model.spilledReadFloorBytes = 390ULL << 20; }
  // No Full extent more: the bytes held, in whole Full pages.
  void wall() { spill.storage.budgetPages = static_cast<uint32_t>(spill.storage.allocatedBytes() / kPages.fullBytes); }
  [[nodiscard]] KvMode mode(uint64_t id) const {
    for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
      if (active.id == id)
        return active.kvMode;
    return KvMode::Resident;
  }
};

// A lone 64K Resident lane at the wall converts itself, its pages freeing more than the hot block
// floor, and decodes its whole output.
void testALongDecodingLaneConvertsItselfAtTheWall() {
  FlashNextLanes lanes;
  Spilling &spill = lanes.spill;
  // 2,047 pages: one page's room left in its eighth extent.
  spill.engine.submit(Spilling::prompt(1, 1, 2'046, 200));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the lane did not decode");
  require(lanes.mode(1) == KvMode::Resident, "the lane did not start Resident");
  lanes.wall();
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(spill.events.completedCount == 1 && spill.events.failures.empty() &&
              spill.events.usage.at(1).second == 200,
          "the lane did not decode its whole output");
  require(counts.converted == 1 && spill.cache.spillSnapshot().spilledPages > 0,
          "the lane did not convert itself at the wall");
}

// With no free Index page to land its copies in, the 64K lane does not convert and writes nothing.
void testALaneConvertsOnlyWithAnIndexRunway() {
  FlashNextLanes lanes;
  Spilling &spill = lanes.spill;
  spill.engine.submit(Spilling::prompt(1, 1, 2'046, 200));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the lane did not decode");
  require(spill.pool.reclaimEmptyExtents(false, 1, kv::PageForm::Index) != 0 &&
              !spill.pool.freePageCount(kv::PageForm::Index),
          "the Index runway was not released");
  lanes.wall();
  spill.runUntilIdle();
  require(spill.engine.snapshot().spill.converted == 0 && spill.tier.writes.empty(),
          "a lane converted with no Index page to land its copies in");
}

// A growth only the host refuses never converts: the host exemption grows the lane Resident.
void testAGrowthTheHostRefusesDoesNotConvert() {
  Spilling spill{16'384, false, 0, 200'000, Spilling::Extent{256}, {}, {}, true, FlashNextLanes::kPages};
  spill.model.spilledReadFloorBytes = 390ULL << 20;
  spill.engine.submit(Spilling::prompt(1, 1, 2'046, 200));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the lane did not decode");
  spill.host.setPressure(MemoryPressure::Warning);
  spill.runUntilIdle();
  require(spill.engine.snapshot().spill.converted == 0 && spill.events.completedCount == 1 &&
              spill.events.failures.empty() && spill.events.usage.at(1).second == 200,
          "a growth the host refused converted its lane, or the lane did not decode");
}

// Beside another lane the 64K lane converts itself rather than yield, and both decode whole.
void testALongLaneConvertsItselfBesideAnother() {
  FlashNextLanes lanes;
  Spilling &spill = lanes.spill;
  spill.engine.submit(Spilling::prompt(1, 1, 2'046, 400));
  spill.engine.submit(Spilling::prompt(2, 2, 1, 600));
  // A command in which the long lane needs a page and the short lane none.
  const auto needsPage = [&](uint64_t id, uint32_t promptPages) {
    const uint64_t tokens = promptPages * KvCache::pageTokens + 1 + spill.events.outputs[id].size() +
                            model::ExecutionLimits::targetVerifyRows;
    return tokens > uint64_t{spill.cache.pageTable(id).pages.size()} * KvCache::pageTokens;
  };
  spill.runUntil(
      [&] {
        return !spill.engine.commandInFlight() && spill.events.outputs.contains(1) && needsPage(1, 2'046) &&
               !needsPage(2, 1);
      },
      "the long lane did not come to need a page alone");
  require(lanes.mode(1) == KvMode::Resident && lanes.mode(2) == KvMode::Resident, "the lanes did not start Resident");
  lanes.wall();
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  spill.runUntil([&] { return lanes.mode(1) == KvMode::Spill; }, "the long lane did not convert itself");
  for (const uint32_t page : held.pages)
    spill.pool.releasePage(page, false);
  spill.runUntilIdle();
  const EngineSnapshot snapshot = spill.engine.snapshot();
  require(held.granted() && snapshot.spill.converted == 1 && snapshot.resourceSuspensions == 0 &&
              spill.events.completedCount == 2 && spill.events.failures.empty() &&
              spill.events.usage.at(1).second == 400 && spill.events.usage.at(2).second == 600,
          "a lane that could convert itself yielded, or a lane did not decode its whole output");
}

// Pages a Resident lane shares count for no conversion: a 40K shared prefix blocks it, a 4K one
// does not.
void testASharedPrefixCountsForNoConversion() {
  for (const uint32_t shared : {1'250U, 125U}) {
    FlashNextLanes lanes;
    Spilling &spill = lanes.spill;
    spill.engine.submit(Spilling::prompt(2, 1, shared, 600));
    spill.runUntil([&] { return spill.events.outputs.contains(2); }, "the sharing lane did not decode");
    std::vector<uint32_t> prompt(shared * KvCache::pageTokens, 1);
    prompt.resize(2'046 * KvCache::pageTokens + 1, 3);
    EngineRequest longLane = request(1, std::move(prompt));
    longLane.maxNewTokens = 400;
    spill.engine.submit(std::move(longLane));
    const auto needsPage = [&](uint64_t id, uint32_t promptPages) {
      const uint64_t tokens = promptPages * KvCache::pageTokens + 1 + spill.events.outputs[id].size() +
                              model::ExecutionLimits::targetVerifyRows;
      return tokens > uint64_t{spill.cache.pageTable(id).pages.size()} * KvCache::pageTokens;
    };
    spill.runUntil(
        [&] {
          return !spill.engine.commandInFlight() && spill.events.outputs.contains(1) && needsPage(1, 2'046) &&
                 !needsPage(2, shared);
        },
        "the long lane did not come to need a page alone");
    require(lanes.mode(1) == KvMode::Resident && spill.cache.pageTable(1).pages.front() ==
                                                     spill.cache.pageTable(2).pages.front(),
            "the lanes did not share their prefix's pages, the long lane Resident");
    lanes.wall();
    const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
    const bool converts = shared == 125;
    require(held.granted() && (lanes.mode(1) == KvMode::Spill) == converts &&
                spill.engine.snapshot().spill.converted == (converts ? 1U : 0U),
            converts ? "a lane whose own suffix passes the gate did not convert"
                     : "a lane converted on pages a Resident lane shares");
    for (const uint32_t page : held.pages)
      spill.pool.releasePage(page, false);
    spill.storage.budgetPages = FlashNextLanes::kPages.count;
    spill.runUntilIdle();
    require(spill.events.completedCount == 2 && spill.events.failures.empty(), "the lanes did not complete");
  }
}

// A 32K lane, which frees less than the hot block floor, and a prefilling lane never convert;
// alone at the wall each fails as on a target that cannot spill.
void testAShortOrPrefillingLaneDoesNotConvert() {
  for (const bool prefilling : {true, false}) {
    FlashNextLanes lanes;
    Spilling &spill = lanes.spill;
    if (prefilling) {
      spill.model.spilledReadFloorBytes = 0;
      // A 64K prompt that starts Resident, as while the SSD cache takes no
      // writes, and whose prefill meets the wall halfway.
      spill.tier.writableFile = false;
      spill.storage.budgetPages = 1'024;
      spill.engine.submit(Spilling::prompt(1, 1, 2'046, 200));
      spill.runUntil([&] { return spill.model.prefillChunks[1] >= 1; }, "the prompt did not start");
      spill.tier.writableFile = true;
    } else {
      // 1,023 pages: one page's room left in its fourth extent.
      spill.engine.submit(Spilling::prompt(1, 1, 1'022, 200));
      spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the lane did not decode");
      lanes.wall();
    }
    spill.runUntilIdle();
    require(spill.engine.snapshot().spill.converted == 0 && spill.cache.spillSnapshot().spilledPages == 0,
            std::string("a ") + (prefilling ? "prefilling" : "short") + " lane converted to Spill");
    require(spill.events.failures == std::vector<std::string>{"capacity_exhausted"},
            "the lone lane did not fail at the wall");
  }
}

// A lane's growth converts no other lane: a short lane refused at the wall leaves a 64K neighbour
// Resident.
void testALanesGrowthConvertsNoOtherLane() {
  FlashNextLanes lanes;
  Spilling &spill = lanes.spill;
  spill.model.spilledReadFloorBytes = 0;
  spill.engine.submit(Spilling::prompt(1, 1, 2'046, 400));
  spill.engine.submit(Spilling::prompt(2, 2, 1, 600));
  // A command in which the short lane needs a page and the long lane none.
  const auto needsPage = [&](uint64_t id, uint32_t promptPages) {
    const uint64_t tokens = promptPages * KvCache::pageTokens + 1 + spill.events.outputs[id].size() +
                            model::ExecutionLimits::targetVerifyRows;
    return tokens > uint64_t{spill.cache.pageTable(id).pages.size()} * KvCache::pageTokens;
  };
  spill.runUntil(
      [&] {
        return !spill.engine.commandInFlight() && spill.events.outputs.contains(1) && needsPage(2, 1) &&
               !needsPage(1, 2'046);
      },
      "the short lane did not come to need a page alone");
  require(lanes.mode(1) == KvMode::Resident && lanes.mode(2) == KvMode::Resident, "the lanes did not start Resident");
  // At the wall: no free Full page, and no extent more.
  lanes.wall();
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  static_cast<void>(spill.engine.tick(spill.now++));
  require(held.granted() && lanes.mode(1) == KvMode::Resident, "a lane's growth converted another lane");
  for (const uint32_t page : held.pages)
    spill.pool.releasePage(page, false);
  spill.storage.budgetPages = FlashNextLanes::kPages.count;
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.failures.empty(), "the lanes did not complete");
}

// A lane whose spill would free about 101 MiB past the hot block floor, short of the 103.5 MiB extent
// its growth lacks, does not convert; a store unit meets the extent and it decodes Resident.
void testALaneFreeingLessThanItsNeedDoesNotConvert() {
  Spilling spill{16'384, false, 0, 200'000, Spilling::Extent{256}, {}, {}, true, FlashNextLanes::kPages};
  spill.model.spilledReadFloorBytes = 390ULL << 20;
  test::FakeElasticPool experts("experts", spill.host.memory, 128ULL << 20, 0, 1, 1, 1);
  static_cast<void>(spill.host.governor.ledger().addPool(experts));
  // 1,279 pages: one page's room left in its fifth extent.
  spill.engine.submit(Spilling::prompt(1, 1, 1'278, 200));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the lane did not decode");
  // The engine's limit leaves no room past what memory holds.
  const uint64_t limit = test::GovernorLimits{}.limitBytes;
  spill.host.memory.hold("limit", limit - std::min(limit, spill.host.memory.total()));
  spill.runUntilIdle();
  require(spill.engine.snapshot().spill.converted == 0 && experts.shrinks == 1,
          "a lane whose spill freed less than the floor and its need converted");
  require(spill.events.completedCount == 1 && spill.events.failures.empty() && spill.events.usage.at(1).second == 200,
          "the lane did not decode its whole output");
}

// Lane demand counts the hot block floor a prefilling Spill lane waits for, and none once it holds
// it.
void testLaneDemandCountsTheHotFloorALaneWaitsFor() {
  Spilling spill(Spilling::Governed{}, 1024, true);
  spill.model.spilledReadFloorBytes = 50'000;
  spill.engine.submit(Spilling::prompt(1, 1, 40, 64));
  uint64_t waiting = 0;
  spill.runUntil(
      [&] {
        if (!spill.events.outputs.contains(1))
          waiting = std::max(waiting, spill.host.governor.ledger().laneDemandBytes());
        return spill.model.spilledReadFloors.contains(1) && !spill.engine.commandInFlight();
      },
      "the Spill lane did not take its floor");
  static_cast<void>(spill.engine.tick(spill.now++));
  require(waiting >= 50'000, "the floor a Spill lane in prefill waits for was not counted");
  require(spill.host.governor.ledger().laneDemandBytes() < 50'000, "a floor the lane holds was counted");
  spill.runUntilIdle();
  require(spill.events.failedCount == 0, "the lane failed");
}

// A start refused memory waits without converting, then starts in the mode its attempt placed,
// counted once.
void testAStartRefusedMemoryCountsNoConversion() {
  Spilling spill(64, false);
  spill.model.deniedReservations = 1000;
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  // The first turn's state goes to the disk: the second restores it.
  spill.engine.submit(Spilling::prompt(1, 1, 4, 1));
  spill.runUntilIdle();
  spill.model.deniedReservations = 0;
  // A resident lane decodes within its last page.
  spill.engine.submit(Spilling::prompt(3, 3, 3, 30));
  spill.runUntil([&] { return spill.events.outputs.contains(3); }, "the resident lane did not decode");
  // The second turn's restore is refused memory for 2 s.
  spill.storage.growthBlocked = true;
  spill.engine.submit(Spilling::prompt(2, 1, 6, 1));
  for (int step = 0; step < 20; ++step) {
    spill.now += 100;
    static_cast<void>(spill.engine.tick(spill.now));
  }
  require(spill.events.startIds.size() == 2 && spill.tier.writes.empty() &&
              spill.engine.snapshot().spill.converted == 0,
          "a start refused memory converted, or counted a conversion");
  // Once memory grows again it starts Resident, its attempt's mode.
  spill.storage.growthBlocked = false;
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(spill.events.completedCount == 3 && counts.converted == 0 && counts.spill == 0 &&
              counts.resident == 3,
          "a start was not counted once, by the mode it started in");
}

// A Spill start short of quota is bounded to what it holds, waits for other commitments behind
// a later arrival that leads it, or is refused (kv_disk_quota) when the quota cannot hold it alone.
void testShortQuotaBoundsWaitsOrRefusesASpillStart() {
  {
    // 3 pages of prompt and 200 tokens of output project 9 pages: 5 hold
    // the prompt and 63 tokens.
    Spilling spill(5);
    spill.engine.submit(Spilling::prompt(1, 1, 3, 200));
    spill.runUntilIdle();
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    require(spill.events.completedCount == 1 && spill.events.usage.at(1).second == 63 &&
                spill.model.outputLimits.at(1) == 63 && counts.bounded == 1 &&
                counts.lengthEnds == 1 && spill.events.boundedIds == std::vector<uint64_t>{1} &&
                spill.tier.writes.size() == 5 && spill.committedSlots() == 0,
            "a start the quota holds with a page of output was not bounded to it");
  }
  {
    Spilling spill(10);
    // The first request commits 8 of the 10 slots.
    spill.engine.submit(Spilling::prompt(1, 1, 2, 191));
    spill.runUntil([&] { return spill.events.outputs.contains(1); },
                   "the first request did not decode");
    // The second needs 4 for its prompt and a page of output, and waits; the
    // third, which arrives then, leads it with one page of prompt and fits in
    // the 2 left.
    spill.engine.submit(Spilling::prompt(2, 2, 3, 200));
    spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1; },
                   "the second request did not wait for the quota");
    spill.engine.submit(Spilling::prompt(3, 3, 1, 31));
    for (int step = 0; step < 5; ++step) {
      static_cast<void>(spill.engine.tick(spill.now++));
      spill.tier.complete();
    }
    const ResourceWaitSnapshot waits = spill.engine.resourceWaitSnapshot(spill.now);
    require(spill.events.startIds == std::vector<uint64_t>{1, 3} && waits.diskQuota == 1 &&
                waits.memory == 0 && waits.heldBehindRefusal == 0,
            "a start waiting for the quota held back a later arrival that leads it");
    spill.runUntilIdle();
    require(spill.events.startIds == std::vector<uint64_t>{1, 3, 2} &&
                spill.events.completedCount == 3 && spill.events.usage.at(2).second == 200 &&
                spill.events.usage.at(3).second == 31 && spill.events.boundedIds.empty() &&
                spill.committedSlots() == 0,
            "the waiting start did not run once the quota it waited for came back, or ended bounded");
  }
  {
    Spilling spill(3);
    spill.engine.submit(Spilling::prompt(1, 1, 3, 200));
    spill.runUntilIdle();
    require(spill.events.failures == std::vector<std::string>{"kv_disk_quota"} &&
                !spill.events.failureDetails.front().second &&
                spill.events.failureDetails.front().first.find("--max-cache-disk") !=
                    std::string::npos &&
                spill.engine.snapshot().spill.refusedQuota == 1 && spill.committedSlots() == 0,
            "a start the quota cannot hold even alone was not refused naming it");
  }
}

// A start waiting for the SSD cache's quota holds back only the starts it leads that compete for
// the quota (competesForQuota): a later Resident start it leads, whose chain has no unheld copies,
// starts beside it.
void testAQuotaWaitHoldsBackOnlyStartsCompetingForIt() {
  // The room beside the lanes holds 20 pages, so a projection past it spills;
  // one within a page stays Resident, and none fits the resident context
  // alone.
  Spilling spill(10, false, 0, KvCache::pageTokens, {4}, {}, {.limitBytes = 20 * 100});
  const auto submit = [&](uint64_t id, uint32_t tokens, uint32_t output) {
    EngineRequest value = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.maxNewTokens = output;
    spill.engine.submit(std::move(value));
  };
  const auto mode = [&](uint64_t id) {
    for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
      if (active.id == id)
        return std::optional<KvMode>(active.kvMode);
    return std::optional<KvMode>();
  };
  // The first request commits the whole quota, its output bounded to it; the
  // second, refused the page its prompt needs, waits.
  submit(1, 33, 4000);
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first request did not decode");
  submit(2, 8, 4000);
  spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1; },
                 "the second request did not wait for the quota");
  // Both led by the second, the third Resident and the fourth Spill.
  std::vector<uint64_t> tried;
  spill.model.beginObserver = [&] { tried.push_back(spill.model.lastBeginId); };
  submit(3, 12, 1);
  submit(4, 12, 4000);
  for (int step = 0; step < 5; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
  }
  const ResourceWaitSnapshot waits = spill.engine.resourceWaitSnapshot(spill.now);
  require(mode(1) == KvMode::Spill && mode(2) == KvMode::Spill && std::ranges::find(tried, 3) != tried.end() &&
              std::ranges::find(tried, 4) == tried.end() && waits.diskQuota == 1 && waits.heldBehindRefusal == 1,
          "a start waiting for the quota held back a Resident start, or let a Spill start be tried");
  spill.runUntilIdle();
  const auto started = [&](uint64_t id) {
    return std::ranges::find(spill.events.startIds, id) - spill.events.startIds.begin();
  };
  require(spill.events.completedCount == 4 && spill.events.failedCount == 0 && started(3) < started(2) &&
              started(2) < started(4) && spill.committedSlots() == 0,
          "the starts did not run in the order the quota wait allows");
}

// A start refused the SSD cache's quota stays a quota refusal while a higher priority keeps it
// waiting for scheduling: the pass its retry starts it, it holds back only the starts it leads
// that compete for the quota (competesForQuota), and a Resident start whose chain has no unheld
// copies starts beside it.
void testADeclinedQuotaRefusalKeepsItsScope() {
  // As in testAQuotaWaitHoldsBackOnlyStartsCompetingForIt: the first
  // request commits the whole quota and the second waits for it.
  Spilling spill(10, false, 0, KvCache::pageTokens, {4}, {}, {.limitBytes = 20 * 100});
  const auto submit = [&](uint64_t id, uint32_t tokens, uint32_t output,
                          RequestPriority priority = RequestPriority::Normal) {
    EngineRequest value = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.maxNewTokens = output;
    value.priority = priority;
    spill.engine.submit(std::move(value));
  };
  submit(1, 33, 4000);
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first request did not decode");
  submit(2, 8, 4000);
  spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1; },
                 "the second request did not wait for the quota");
  // A higher priority runs; the second, the third (Resident) and the fourth
  // (Spill) wait for scheduling.
  submit(5, 12, 400, RequestPriority::Foreground);
  spill.runUntil([&] { return spill.events.outputs.contains(5); }, "the higher priority did not decode");
  submit(3, 12, 1);
  submit(4, 12, 4000);
  for (int step = 0; step < 2; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
  }
  // The higher priority and the first end in the same pass: the second's
  // retry gets the quota.
  std::vector<uint64_t> tried;
  spill.model.beginObserver = [&] { tried.push_back(spill.model.lastBeginId); };
  spill.engine.cancel(5);
  spill.engine.cancel(1);
  spill.runUntil([&] { return !tried.empty(); }, "nothing started once the higher priority ended");
  require(tried == std::vector<uint64_t>{2, 3} && spill.events.failedCount == 0,
          "a declined quota refusal held back a Resident start, or let a Spill start be tried");
}

// A Resident start submitted after a start refused the SSD cache's quota, holding none of the
// quota, does not keep that start's wait open though admitted ahead of it: the wait expires at its
// limit while the Resident start still works.
void testAResidentStartDoesNotHoldAQuotaWaitOpen() {
  // As in testAQuotaWaitHoldsBackOnlyStartsCompetingForIt: the first
  // request commits the whole quota and the second waits for it.
  Spilling spill(10, false, 0, KvCache::pageTokens, {4}, {}, {.limitBytes = 20 * 100});
  const auto submit = [&](uint64_t id, uint32_t tokens, uint32_t output, RequestPriority priority) {
    EngineRequest value = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.maxNewTokens = output;
    value.priority = priority;
    value.deadlineMilliseconds = 1e9;
    spill.engine.submit(std::move(value));
  };
  // A second apart: the second's backoff is over at every pass, so its
  // retries are refused beside the lanes admitted ahead of it.
  const auto run = [&](const std::function<bool()> &done, const char *message) {
    for (uint32_t step = 0; step < 1000 && !done(); ++step) {
      static_cast<void>(spill.engine.tick(spill.now += 1000));
      spill.tier.complete();
    }
    require(done(), message);
  };
  // The quota's holder runs below the others' priority: it does not run
  // while they do.
  submit(1, 33, 4000, RequestPriority::Background);
  run([&] { return spill.events.outputs.contains(1); }, "the first request did not decode");
  submit(2, 8, 4000, RequestPriority::Normal);
  run([&] { return spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1; },
      "the second request did not wait for the quota");
  // A shorter Resident prompt leads the waiting one and starts beside it; it
  // decodes for twice the wait's limit.
  submit(3, 1, KvCache::pageTokens - 1, RequestPriority::Normal);
  run([&] { return spill.events.outputs.contains(3); }, "the Resident start did not decode");
  const double startedAt = spill.now;
  const size_t held = spill.events.outputs.at(1).size();
  run([&] { return spill.events.failedCount != 0 || spill.events.usage.contains(3); },
      "neither the wait nor the Resident start ended");
  require(spill.events.failures == std::vector<std::string>{"resource_timeout"} &&
              spill.events.failureDetails.back().first.find("SSD cache's room for active KV") != std::string::npos &&
              spill.events.failureDetails.back().second,
          "a wait for the quota outlived its limit while a Resident start admitted beside it worked");
  require(spill.now <= startedAt + kResourceWaitTimeoutMilliseconds && !spill.events.usage.contains(3) &&
              spill.events.outputs.at(1).size() == held,
          "the wait did not expire at its limit, or the quota's holder ran meanwhile");
  spill.engine.cancel(1);
  spill.engine.cancel(3);
  spill.runUntilIdle();
  require(spill.committedSlots() == 0, "the quota was not returned");
}

// A Resident start that restores a chain with copies holds them as active KV, which the quota counts:
// admitted ahead of a Spill start refused the quota, it keeps that start's wait open while it works,
// past the wait's limit, and the Spill start runs once it ends.
void testALaneHoldingCopiesHoldsAQuotaWaitOpen() {
  Spilling spill(12, false, 2'000, 0, {4}, {}, {.limitBytes = 2'000});
  const auto mode = [&](uint64_t id) {
    for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
      if (active.id == id)
        return std::optional<KvMode>(active.kvMode);
    return std::optional<KvMode>();
  };
  // The first request spills its 10 pages, whose copies stay cached.
  spill.engine.submit(Spilling::prompt(1, 1, 10, 4'096));
  spill.runUntilIdle();
  // A Spill start and a Resident one that restores the first request's
  // chain arrive together. The Resident one leads, and holds 10 of the 12
  // slots of active KV while it decodes: the Spill start is refused the
  // quota.
  EngineRequest spilling = Spilling::prompt(2, 2, 4, 4'000);
  spilling.deadlineMilliseconds = 1e9;
  EngineRequest resident = Spilling::prompt(3, 1, 10, 60);
  resident.deadlineMilliseconds = 1e9;
  spill.engine.submit(std::move(spilling));
  spill.engine.submit(std::move(resident));
  const double submittedAt = spill.now;
  std::optional<double> residentEnded;
  bool waited = false;
  for (uint32_t step = 0; step < 1000 && !spill.events.outputs.contains(2) && !spill.events.failedCount; ++step) {
    static_cast<void>(spill.engine.tick(spill.now += 1000));
    spill.tier.complete();
    // Each lane counts the copies it holds as they land and leave.
    spill.cache.checkInvariants();
    if (mode(3) && spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1) {
      waited = true;
      require(mode(3) == KvMode::Resident && spill.cache.spillSnapshot().heldBytes >= 1'000,
              "the Spill start did not wait beside a Resident lane holding its chain's copies");
    }
    if (!residentEnded && spill.events.usage.contains(3))
      residentEnded = spill.now;
  }
  require(waited && spill.events.failedCount == 0 && spill.events.outputs.contains(2) && residentEnded &&
              *residentEnded > submittedAt + kResourceWaitTimeoutMilliseconds,
          "a quota wait expired while a lane holding copies of active KV worked");
  require(spill.events.startIds == std::vector<uint64_t>{1, 3, 2} && mode(2) == KvMode::Spill,
          "the Spill start did not run once the lane holding the copies ended");
  spill.engine.cancel(2);
  spill.runUntilIdle();
  require(spill.committedSlots() == 0, "the quota was not returned");
}

// A Resident start that restores a chain with copies no request holds would take the quota they
// count as once it held them: a quota refusal it follows holds it back, as it would a Spill
// start, while a Resident start whose chain has no copies starts beside the refusal.
void testAQuotaRefusalHoldsBackAStartRestoringUnheldCopies() {
  // Active KV may take 12 of the 48 slots: cached copies keep the rest.
  Spilling spill(48, false, 2'000, 0, {4}, {}, {.limitBytes = 2'000});
  spill.budget->setActiveLimitBytes(12 * 100);
  // Only Spill requests write copies: a Resident turn's chain stays without.
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Persist)] = 0;
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Reclaim)] = 0;
  const auto submit = [&](uint64_t id, std::vector<uint32_t> tokens, uint32_t output) {
    EngineRequest value = request(id, std::move(tokens));
    value.maxNewTokens = output;
    value.deadlineMilliseconds = 1e9;
    spill.engine.submit(std::move(value));
  };
  const auto continued = [](std::vector<uint32_t> tokens, uint32_t token) {
    tokens.insert(tokens.end(), KvCache::pageTokens, token);
    return tokens;
  };
  const auto started = [&](uint64_t id) {
    return std::ranges::find(spill.events.startIds, id) - spill.events.startIds.begin();
  };
  // A Spill turn leaves its chain cached with copies, a Resident one its
  // chain without.
  const std::vector<uint32_t> spilled(10 * KvCache::pageTokens + 1, 1);
  const std::vector<uint32_t> resident(2 * KvCache::pageTokens + 1, 6);
  submit(1, spilled, 4'096);
  spill.runUntilIdle();
  submit(6, resident, 1);
  spill.runUntilIdle();
  require(spill.cache.probe(spilled, {}).cachedTokens() >= 10 * KvCache::pageTokens &&
              spill.cache.probe(resident, {}).cachedTokens() && spill.cache.spillSnapshot().heldBytes == 0,
          "the turns' chains were not cached");
  // A Spill lane commits the whole quota; a Spill start of one token waits
  // for it, and leads every later start.
  submit(4, std::vector<uint32_t>(KvCache::pageTokens + 1, 4), 4'000);
  spill.runUntil([&] { return spill.events.outputs.contains(4); }, "the Spill lane did not decode");
  submit(2, {2}, 700);
  spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1; },
                 "the Spill start did not wait for the quota");
  // Two Resident starts continue the turns' chains: the first would hold
  // the cached copies of its chain as active KV.
  submit(3, continued(spilled, 3), 1);
  submit(5, continued(resident, 5), 1);
  for (int step = 0; step < 5; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
    spill.cache.checkInvariants();
  }
  const ResourceWaitSnapshot waits = spill.engine.resourceWaitSnapshot(spill.now);
  require(started(5) < started(2) && started(3) == std::ssize(spill.events.startIds) && waits.diskQuota == 1 &&
              waits.heldBehindRefusal == 1,
          "a quota refusal did not hold back the start restoring unheld copies alone");
  // The quota grows: the Spill start starts, then the one it held back.
  spill.budget->setActiveLimitBytes(48 * 100);
  spill.runUntil([&] { return started(3) < std::ssize(spill.events.startIds); }, "the start held back did not start");
  spill.cache.checkInvariants();
  require(started(2) < started(3) && spill.events.failedCount == 0,
          "the start held back did not start after the Spill start");
  spill.engine.cancel(4);
  spill.engine.cancel(2);
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.committedSlots() == 0, "the quota was not returned");
}

// A Resident start whose chain's copies running lanes hold already takes none of the quota: the
// quota counts a held copy once, however many lanes hold its block. A quota refusal it follows does
// not hold it back, and it starts beside the refusal.
void testAStartRestoringHeldCopiesStartsBesideAQuotaRefusal() {
  Spilling spill(12, false, 2'000, 0, {4}, {}, {.limitBytes = 2'000});
  const auto submit = [&](uint64_t id, std::vector<uint32_t> tokens, uint32_t output) {
    EngineRequest value = request(id, std::move(tokens));
    value.maxNewTokens = output;
    value.deadlineMilliseconds = 1e9;
    spill.engine.submit(std::move(value));
  };
  const auto started = [&](uint64_t id) {
    return std::ranges::find(spill.events.startIds, id) != spill.events.startIds.end();
  };
  // A Spill turn leaves its chain cached with copies; a Resident lane
  // restores it and holds 10 of the 12 slots of active KV while it decodes.
  const std::vector<uint32_t> spilled(10 * KvCache::pageTokens + 1, 1);
  submit(1, spilled, 4'096);
  spill.runUntilIdle();
  submit(3, spilled, 300);
  spill.runUntil([&] { return spill.events.outputs.contains(3); }, "the Resident lane did not decode");
  require(spill.cache.spillSnapshot().heldBytes == 1'000, "the Resident lane did not hold its chain's copies");
  // A Spill start is refused the quota and leads every later start.
  submit(2, std::vector<uint32_t>(4 * KvCache::pageTokens + 1, 2), 4'000);
  spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1; },
                 "the Spill start did not wait for the quota");
  // A Resident start restores the same chain, and more of its own.
  std::vector<uint32_t> continued = spilled;
  continued.insert(continued.end(), 5 * KvCache::pageTokens, 4);
  submit(4, std::move(continued), 1);
  for (int step = 0; step < 5 && !started(4); ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
    spill.cache.checkInvariants();
  }
  const ResourceWaitSnapshot waits = spill.engine.resourceWaitSnapshot(spill.now);
  require(started(4) && !started(2) && waits.diskQuota == 1 && !waits.heldBehindRefusal &&
              spill.cache.spillSnapshot().heldBytes == 1'000,
          "a quota refusal held back a start whose chain's copies lanes hold already");
  for (const uint64_t id : {2, 3})
    spill.engine.cancel(id);
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.committedSlots() == 0, "the quota was not returned");
}

// A Spill start that restores a chain with copies no request holds takes them into the quota of
// active KV as it holds them, beside its commitment: it starts only with room for both, its output
// bounded to what the quota holds, and active KV stays within its limit, revoking no commitment.
void testASpillStartCountsTheCopiesItRestores() {
  // Active KV may take 12 of the 48 slots: cached copies keep the rest.
  Spilling spill(48, false, 2'000, 0, {4}, {}, {.limitBytes = 2'000});
  spill.budget->setActiveLimitBytes(12 * 100);
  const auto submit = [&](uint64_t id, std::vector<uint32_t> tokens, uint32_t output) {
    EngineRequest value = request(id, std::move(tokens));
    value.maxNewTokens = output;
    value.deadlineMilliseconds = 1e9;
    spill.engine.submit(std::move(value));
  };
  // A Spill turn leaves its chain cached with 10 copies or more, which no
  // request holds.
  const std::vector<uint32_t> spilled(10 * KvCache::pageTokens + 1, 1);
  submit(1, spilled, 4'096);
  spill.runUntilIdle();
  require(spill.cache.probe(spilled, {}).cachedTokens() >= 10 * KvCache::pageTokens &&
              spill.cache.spillSnapshot().heldBytes == 0,
          "the turn's chain was not cached");
  // A Spill start continues it: its 10 copies and its commitment share the
  // 12 slots.
  std::vector<uint32_t> continued = spilled;
  continued.insert(continued.end(), KvCache::pageTokens, 2);
  submit(2, std::move(continued), 4'000);
  spill.runUntil(
      [&] {
        const KvSpillSnapshot snapshot = spill.cache.spillSnapshot();
        require(snapshot.heldBytes + snapshot.committedBytes <= snapshot.activeLimitBytes &&
                    !snapshot.revokedCommitments,
                "active KV passed its limit as the Spill start held the copies it restored");
        return idle(spill.engine);
      },
      "the Spill start did not end");
  require(spill.events.failedCount == 0 && spill.events.completedCount == 2 && spill.events.outputs.contains(2) &&
              spill.committedSlots() == 0,
          "the Spill start did not run within the quota");
}

// A start refused memory keeps the probe it was placed with while it backs off, and the cache may
// take the chain that probe found meanwhile: a quota refusal it follows reads that chain only as
// far as the cache still has it, and the start waits on.
void testAQuotaRefusalReadsOnlyTheChainAStartKeeps() {
  Spilling spill(10, false, 0, KvCache::pageTokens, {4}, {}, {.limitBytes = 20 * 100});
  // No cached copy is written: a cached chain stays without copies.
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Persist)] = 0;
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Reclaim)] = 0;
  const auto submit = [&](uint64_t id, std::vector<uint32_t> tokens, uint32_t output) {
    EngineRequest value = request(id, std::move(tokens));
    value.maxNewTokens = output;
    value.deadlineMilliseconds = 1e9;
    spill.engine.submit(std::move(value));
  };
  const std::vector<uint32_t> turn(KvCache::pageTokens + 1, 50);
  // An earlier turn ends; its chain stays cached.
  submit(5, turn, 1);
  spill.runUntil([&] { return spill.events.usage.contains(5); }, "the earlier turn did not end");
  require(spill.cache.probe(turn, {}).cachedTokens() != 0, "the earlier turn's chain was not cached");
  // The first request commits the whole quota; the second waits for it.
  submit(1, std::vector<uint32_t>(KvCache::pageTokens + 1, 1), 4'000);
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first request did not decode");
  submit(2, {2}, 4'000);
  spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1; },
                 "the second request did not wait for the quota");
  // The earlier turn sent again: the host refuses its start memory.
  spill.model.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  spill.model.beginGrowthBlocked = [&] { return spill.model.lastBeginId == 3; };
  submit(3, turn, 1);
  spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).memory == 1; },
                 "the resend was not refused memory");
  // Reclaims take the chain its probe found.
  for (int step = 0; step < 40 && spill.cache.lookup(turn, {}).kvBoundary; ++step) {
    static_cast<void>(test::evictStep(spill.cache));
    static_cast<void>(test::stateStep(spill.cache));
  }
  require(!spill.cache.lookup(turn, {}).kvBoundary, "the chain was not taken");
  for (int step = 0; step < 5; ++step)
    static_cast<void>(spill.engine.tick(spill.now++));
  const ResourceWaitSnapshot waits = spill.engine.resourceWaitSnapshot(spill.now);
  require(waits.memory == 1 && waits.diskQuota == 1 && spill.events.failedCount == 0,
          "the starts did not wait on beside a chain the cache took");
  spill.model.beginGrowthBlocked = {};
  for (const uint64_t id : {1, 2, 3})
    spill.engine.cancel(id);
  spill.runUntilIdle();
  require(spill.committedSlots() == 0, "the quota was not returned");
}

// A start that cannot spill, the tier closed or the quota short, starts Resident where its
// projection fits the resident context alone.
void testAStartThatCannotSpillStartsResidentWithinTheResidentContext() {
  for (const bool closed : {true, false}) {
    // The room beside the lanes holds 20 pages, so placement spills; the resident context is 50.
    Spilling spill(8, false, 0, 50 * KvCache::pageTokens, {4}, {}, {.limitBytes = 20 * 100});
    spill.tier.writableFile = !closed;
    // 40 pages of prompt and 64 tokens of output project 43 pages; the 8
    // slots of the quota cannot hold them.
    spill.engine.submit(Spilling::prompt(1, 1, 40, 64));
    spill.runUntilIdle();
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    require(spill.events.completedCount == 1 && spill.events.usage.at(1).second == 64 &&
                counts.resident == 1 && counts.spill == 0 && counts.refusedClosed == 0 &&
                counts.refusedQuota == 0 && spill.tier.writes.empty(),
            closed ? "a start that fits the resident context was refused while the tier took no writes"
                   : "a start that fits the resident context was refused for a short quota");
  }
}

// A Spill start the write budget cannot cover is refused (kv_write_budget) with a Retry-After of
// when the budget will cover it.
void testWriteBudgetRefusalSaysWhenToRetry() {
  // Every start spills, and none fits in memory alone.
  Spilling spill(64, true, 100'000, KvCache::pageTokens);
  // 1,600 bytes a day, 400 at once: four slots.
  spill.budget->setWriteBudget(1'600);
  // The first request commits the eight pages of its projection and writes
  // none of them.
  spill.allowSpillWrites(false);
  spill.engine.submit(Spilling::prompt(1, 1, 2, 191));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first request did not decode");
  require(spill.committedSlots() == 8, "the first request did not commit its projection");
  // The budget covers the 800 bytes committed once it refilled 400 more,
  // a quarter of a day's: in 6 hours.
  spill.engine.submit(Spilling::prompt(2, 2, 1, 31));
  spill.runUntil([&] { return spill.events.failedCount == 1; }, "the second request was not refused");
  require(spill.events.failures == std::vector<std::string>{"kv_write_budget"} &&
              spill.events.failureDetails.front().second &&
              spill.events.retryAfters == std::vector<uint32_t>{21'600} &&
              spill.engine.snapshot().spill.refusedWriteBudget == 1,
          "the write budget's refusal did not say when a retry may succeed");
  spill.engine.cancel(1);
  spill.runUntilIdle();
  require(spill.committedSlots() == 0, "the first request kept its commitment");
}

// A Spill prompt whose history holds Index pages runs as prefill groups; a Resident one as chunks.
void testPrefillGroupsFollowIndexPages() {
  for (const bool spills : {false, true}) {
    Spilling spill(256, spills, 100'000, 0, {4}, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget));
    spill.model.time(test::heavyHistoryTarget());
    // 128 pages and a token: three chunks, or fewer commands with groups.
    spill.engine.submit(Spilling::prompt(1, 1, 128, 1));
    spill.runUntilIdle();
    require(spill.events.completedCount == 1 &&
                (spills ? spill.model.prefillChunks.at(1) < 3u : spill.model.prefillChunks.at(1) == 3u),
            spills ? "a Spill request's prompt did not run as prefill groups"
                   : "a Resident request's prompt ran as a prefill group");
  }
}

// A Resident lane that comes to hold a Spill request's blocks as it prefills runs the rest of its
// prompt as prefill groups: it shares that request's prefix but no state on it, so it starts from
// nothing, and the blocks its first chunk computes take the Spill request's Index pages
// (KvCache::insert).
void testALaneThatTakesSpilledBlocksAsItPrefillsRunsPrefillGroups() {
  Spilling spill(1024, true, 100'000, 0, {4}, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget));
  spill.model.time(test::heavyHistoryTarget());
  // Eight pages, four more of its own and a token: its pages but its dense
  // and window pages spill.
  std::vector<uint32_t> prompt(8 * KvCache::pageTokens, 1);
  prompt.resize(prompt.size() + 4 * KvCache::pageTokens + 1, 9);
  EngineRequest spilling = request(1, prompt);
  spilling.maxNewTokens = 4'096;
  spill.engine.submit(std::move(spilling));
  spill.runUntil([&] { return spill.cache.spillSnapshot().spilledPages == 10 && !spill.engine.commandInFlight(); },
                 "the Spill request's pages did not spill");
  // Its eight pages, then 120 pages and a token: three chunks, the first of
  // which takes the seven spilled pages among them. It starts Resident while
  // the SSD cache takes no writes.
  prompt.resize(8 * KvCache::pageTokens);
  prompt.resize(prompt.size() + 120 * KvCache::pageTokens + 1, 2);
  EngineRequest continuing = request(2, std::move(prompt));
  continuing.maxNewTokens = 1;
  spill.tier.writableFile = false;
  spill.engine.submit(std::move(continuing));
  spill.runUntil([&] { return spill.events.startIds.size() == 2; }, "the Resident request did not start");
  spill.tier.writableFile = true;
  spill.runUntil([&] { return spill.events.usage.contains(2); }, "the Resident request did not run");
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  // The two chunks left run as one group.
  require(counts.resident == 1 && counts.spill == 1 && spill.events.starts.back() == 0 &&
              spill.model.prefillChunks.at(2) == 2,
          "a Resident lane that took spilled blocks as it prefilled ran the rest a chunk a command");
  spill.engine.cancel(1);
  spill.runUntilIdle();
}

// Full extents a spill empties are released, all but one runway while a spilling prompt grows or
// the host holds growth back.
void testSpilledFullExtentsLeaveOnceNoPromptGrows() {
  for (const bool paused : {false, true}) {
    Spilling spill(1024);
    const auto emptyFullExtents = [&] {
      const KvPoolSnapshot pool = spill.pool.snapshot();
      return (pool.reclaimableBytes - pool.index.reclaimableBytes) /
             (uint64_t{spill.pool.extentPages()} * spill.pool.bytesPerPage());
    };
    // 70 pages and a token: prefill commands after the first's pages
    // spilled, then 64 tokens of output.
    spill.host.pause(paused);
    spill.engine.submit(Spilling::prompt(1, 1, 70, 64));
    uint64_t prompting = 0;
    spill.runUntil(
        [&] {
          if (spill.events.outputs.contains(1)) return true;
          prompting = std::max(prompting, emptyFullExtents());
          return false;
        },
        "the request did not decode");
    // All its pages but the dense history and the window: 68.
    spill.runUntil([&] { return spill.cache.spillSnapshot().spilledPages >= 68; },
                   "the prompt's pages did not spill");
    require(spill.model.prefillChunks.at(1) >= 2 && prompting <= 1 && emptyFullExtents() == (paused ? 1u : 0u),
            paused ? "the Full extents spills emptied did not keep the runway while the host held growth back"
                   : "the Full extents spills emptied stayed, or the runway went while the prompt grew");
    spill.host.pause(false);
    spill.runUntilIdle();
  }
}

// Under the host's pause a Spill prompt's landed pages still move to Index pages, so a prompt that
// fits only spilled completes.
void testSpillMovesThroughTheHostPause() {
  Spilling spill(Spilling::Governed{}, 1024);
  // The governor's host refuses growth unless a request in service needs
  // it, then the engine's limit: 90 Full pages.
  spill.limit(90 * 100);
  spill.host.pause(true);
  // As Full pages, the prompt's 121 outgrow the limit.
  spill.engine.submit(Spilling::prompt(1, 1, 120, 8));
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 1 &&
              spill.events.usage.at(1).second == 8,
          "a Spill prompt kept its landed pages Full under the host's pause until the engine's limit");
}

// A suspended Spill request returns its commitment and commits again on resume; a short quota
// bounds the resumed output, and each attempt bounds it afresh.
void testSpillResumeCommitsAgain() {
  for (const bool roomLater : {false, true}) {
    Spilling spill(20);
    // No copies: the request holds its whole history without any.
    spill.allowSpillWrites(false);
    EngineRequest background = Spilling::prompt(1, 1, 2, 100);
    background.priority = RequestPriority::Background;
    spill.engine.submit(std::move(background));
    spill.runUntil([&] { return spill.events.outputs[1].size() >= 10; },
                   "the request did not decode");
    // A Normal start is refused memory while it is resident, which
    // suspends it.
    spill.model.beginGrowthBlocked = [&] {
      return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
    };
    spill.model.resumeDenied = true;
    spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
    spill.runUntil([&] { return spill.events.completedCount == 1; },
                   "the Normal request did not run");
    require(spill.model.suspensions == 1 && spill.committedSlots() == 0,
            "the suspended request kept its commitment");
    // Its projection is 5 pages, none copied; 4 hold its history and a page
    // of output, 63 tokens in all.
    spill.allowSpillWrites(true);
    spill.budget->setActiveLimitBytes(4 * 100);
    if (roomLater) {
      const uint32_t attempts = spill.model.resumeAttempts;
      spill.runUntil([&] { return spill.model.resumeAttempts > attempts; },
                     "the request did not try to resume");
      require(spill.model.outputLimits.at(1) == 63, "the refused resume was not bounded");
      spill.budget->setActiveLimitBytes(20 * 100);
    }
    spill.model.resumeDenied = false;
    spill.runUntil([&] { return spill.model.resumptions == 1; }, "the request did not resume");
    require(spill.committedSlots() == (roomLater ? 5 : 4),
            "the resumed request did not commit for what it still projects");
    spill.runUntilIdle();
    const uint32_t output = roomLater ? 100 : 63;
    require(spill.events.completedCount == 2 && spill.events.usage.at(1).second == output &&
                spill.model.outputLimits.at(1) == output && spill.committedSlots() == 0,
            roomLater ? "a bound of a resume that did not start outlived it"
                      : "the resumed request was not bounded to what the quota holds");
  }
}

// A request bounded at its start and bounded again on resume counts as one bounded request.
void testABoundAtStartAndResumeCountsOnce() {
  // 2 pages of prompt and 200 tokens of output project 9 pages: 5 hold the
  // prompt and 95 tokens.
  Spilling spill(20);
  spill.budget->setActiveLimitBytes(5 * 100);
  // No copies: the request holds its whole history without any.
  spill.allowSpillWrites(false);
  EngineRequest background = Spilling::prompt(1, 1, 2, 200);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 10; }, "the request did not decode");
  require(spill.model.outputLimits.at(1) == 95, "the start was not bounded");
  // Room for a Normal start, which is refused memory while the request is
  // resident and so suspends it.
  spill.budget->setActiveLimitBytes(20 * 100);
  spill.model.beginGrowthBlocked = [&] {
    return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
  };
  spill.model.resumeDenied = true;
  spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
  spill.runUntil([&] { return spill.events.completedCount == 1; }, "the Normal request did not run");
  require(spill.model.suspensions == 1, "the request was not suspended");
  spill.budget->setActiveLimitBytes(5 * 100);
  spill.model.resumeDenied = false;
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(spill.model.resumptions == 1 && spill.events.usage.at(1).second == 95 &&
              spill.model.outputLimits.at(1) == 95 && spill.events.boundedIds == std::vector<uint64_t>{1},
          "the resume was not bounded again");
  require(counts.bounded == 1 && counts.lengthEnds == 1 && counts.convertedBounded == 0,
          "a request bounded at its start and on resume was not counted once");
}

// A Spill request suspended before the tier closed resumes and completes: the closure refuses only
// new Spill starts.
void testSpillResumeOnAClosedTier() {
  Spilling spill(20);
  // Its prompt's second and third pages spill.
  EngineRequest background = Spilling::prompt(1, 1, 4, 100);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 10; },
                 "the request did not decode");
  // A Normal start is refused memory while it is resident, which
  // suspends it.
  spill.model.beginGrowthBlocked = [&] {
    return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
  };
  spill.model.resumeDenied = true;
  spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
  spill.runUntil([&] { return spill.events.completedCount == 1; },
                 "the Normal request did not run");
  require(spill.model.suspensions == 1 && spill.cache.spillSnapshot().spilledPages != 0,
          "the request was not suspended with spilled pages");
  spill.tier.writableFile = false;
  spill.model.resumeDenied = false;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 2 &&
              spill.model.resumptions == 1 && spill.events.usage.at(1).second == 100 &&
              spill.committedSlots() == 0,
          "a Spill request suspended before the tier closed did not resume");
}

// A resume's quota bound leaves output past its anchor: no room past it ends the request at its
// length, one more page bounds it to that page.
void testSpillResumeBoundLeavesOutputPastItsAnchor() {
  for (const uint32_t slots : {3U, 4U}) {
    Spilling spill(20);
    // No copies: the request holds its whole history without any.
    spill.allowSpillWrites(false);
    EngineRequest background = Spilling::prompt(1, 1, 2, 100);
    background.priority = RequestPriority::Background;
    spill.engine.submit(std::move(background));
    // 65 tokens of prompt and 30 of output end a token short of the third
    // page, which the anchor ends.
    spill.runUntil([&] { return spill.events.outputs[1].size() == 30; },
                   "the request did not decode");
    // A Normal start is refused memory while it is resident, which
    // suspends it.
    spill.model.beginGrowthBlocked = [&] {
      return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
    };
    spill.model.resumeDenied = true;
    spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
    spill.runUntil([&] { return spill.events.completedCount == 1; },
                   "the Normal request did not run");
    spill.budget->setActiveLimitBytes(uint64_t{slots} * 100);
    spill.model.resumeDenied = false;
    spill.runUntilIdle();
    const bool ends = slots == 3;
    require(spill.model.suspensions == 1 && spill.model.resumptions == (ends ? 0u : 1u) &&
                spill.events.completedCount == 2 &&
                spill.events.usage.at(1).second == (ends ? 30u : 63u) &&
                spill.events.boundedIds == std::vector<uint64_t>{1} &&
                spill.engine.snapshot().spill.lengthEnds == 1 && spill.committedSlots() == 0,
            ends ? "a resume whose quota held no output past its anchor did not end at its length"
                 : "a resume was not bounded to the output past its anchor that the quota holds");
  }
}

// A Spill request suspended in its prompt holds no anchor: a quota of its prompt's pages bounds it
// to the one token its prompt's last step selects.
void testSpillResumeInItsPromptHoldsNoAnchor() {
  Spilling spill(20, true, 100'000, 0, {4}, test::commandShape(KvCache::pageTokens));
  // No copies: the request holds its whole history without any.
  spill.allowSpillWrites(false);
  // 95 tokens of prompt end a token short of the third page, and each
  // prefill command takes a page of them.
  EngineRequest background = request(1, std::vector<uint32_t>(3 * KvCache::pageTokens - 1, 1));
  background.maxNewTokens = 100;
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntil([&] { return spill.model.prefillChunks[1] == 1; }, "the prompt did not start");
  // A Normal start is refused memory while it is resident, which suspends
  // it.
  spill.model.beginGrowthBlocked = [&] {
    return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
  };
  spill.model.resumeDenied = true;
  spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
  spill.runUntil([&] { return spill.events.completedCount == 1; },
                 "the Normal request did not run");
  require(spill.model.suspensions == 1 &&
              spill.engine.snapshot().activeRequests.front().promptProcessed <
                  3 * KvCache::pageTokens - 1,
          "the request was not suspended in its prompt");
  spill.budget->setActiveLimitBytes(3 * 100);
  spill.model.resumeDenied = false;
  spill.runUntilIdle();
  require(spill.model.resumptions == 1 && spill.model.outputLimits.at(1) == 1 &&
              spill.events.completedCount == 2 && spill.events.usage.at(1).second == 1 &&
              spill.events.boundedIds == std::vector<uint64_t>{1} && spill.committedSlots() == 0,
          "a resume in its prompt was not bounded to the token its prompt's pages leave");
}

// A resumed Spill request takes its replay's Full pages a command at a time, so a replay larger
// than memory as Full pages completes.
void testSpillResumeTakesItsReplaysPagesAsItRuns() {
  Spilling spill(1024);
  // 100 Full pages: a prefill command's 64 fit beside the Index pages of the
  // request's spilled history, and its 125 pages do not.
  spill.storage.budgetPages = 100;
  spill.model.decodeTokens = 8;
  EngineRequest background = Spilling::prompt(1, 1, 4, 4'096);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 3'840; },
                 "the request did not decode");
  // A Normal start is refused memory while it is resident, which suspends
  // it.
  spill.model.beginGrowthBlocked = [&] {
    return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
  };
  spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
  spill.runUntilIdle();
  require(spill.model.suspensions == 1 && spill.model.resumptions == 1 &&
              spill.events.failedCount == 0 && spill.events.completedCount == 2 &&
              spill.events.usage.at(1).second == 4'096,
          "a Spill request whose replay memory holds only a command at a time as Full pages "
          "did not resume");
}

// While a Resident resume waits for memory the pools do not grow into it: it resumes Resident on
// that room, with no store take-back and no conversion.
void testAWaitingResumeFindsTheMemoryItsKvHeldFree() {
  Spilling spill(Spilling::Governed{}, 1024, false);
  test::FakeElasticPool experts("experts", spill.host.memory, 1'000, 0, 16, 16, 0);
  static_cast<void>(spill.host.governor.ledger().addPool(experts));
  // The engine's limit holds the projection of a request with 4,096 tokens
  // of output, 261 Full pages, beside the Index runway.
  spill.limit(270 * 100);
  spill.model.decodeTokens = 8;
  EngineRequest background = Spilling::prompt(1, 1, 4, 4'096);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  // 2,400 tokens of output: 80 pages, all Full.
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 2'400; },
                 "the request did not decode");
  // A Normal start is refused memory while it is resident, which suspends
  // it, and its resume is refused until the store has grown.
  spill.model.beginGrowthBlocked = [&] {
    return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
  };
  spill.model.resumeDenied = true;
  spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
  spill.runUntil([&] { return spill.events.completedCount == 1; },
                 "the Normal request did not run");
  // A critical pass releases what the cache held; the control pass then
  // leaves the room to the resume that waits for it.
  static_cast<void>(test::criticalPass(spill.host.governor));
  static_cast<void>(spill.host.governor.controlPass(spill.now));
  require(experts.units() == 0, "the store grew " + std::to_string(experts.units()) +
                                    " units into the room a waiting resume waits for");
  const uint64_t shrinks = experts.shrinks;
  spill.model.resumeDenied = false;
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(counts.resident == 2 && counts.converted == 0 && spill.events.failedCount == 0 &&
              spill.events.completedCount == 2 && spill.events.usage.at(1).second == 4'096 &&
              spill.model.resumptions == 1 && experts.shrinks == shrinks,
          "a Resident request's resume did not run on the room it waited for without a take back");
}

// Decode leaves out a lane refused its hot block floor, which waits rather than fails.
void testDecodeLeavesOutALaneRefusedItsHotFloor() {
  Spilling spill(64);
  spill.model.refusedSpilledReads = {2};
  spill.engine.submit(Spilling::prompt(1, 1, 5, 20));
  spill.engine.submit(Spilling::prompt(2, 2, 5, 20));
  spill.runUntil([&] { return spill.events.usage.contains(1); },
                 "the lane given its floor did not finish");
  require(spill.cache.holdsIndexPages(2) && !spill.events.outputs.contains(2) &&
              spill.events.failedCount == 0,
          "a lane refused its hot floor decoded or failed beside a lane that ran");
  spill.model.refusedSpilledReads.clear();
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.usage.at(2).second == 20,
          "the lane refused its hot floor did not run once it had it");
}

// A Spill lane refused its floor beside a decoding lane takes the prefill lane's memory, and does
// not wait for write-throughs that free no extent.
void testARefusedDecodeTakesThePrefillLanesMemory() {
  for (const bool writesInFlight : {false, true}) {
    // With writes in flight, extents of 64 pages, which the requests share.
    Spilling spill(1024, true, 100'000, 0, Spilling::Extent{writesInFlight ? 64u : 4u});
    // Lane 3's floor fits only while the prompt's lane is not resident.
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      const auto prompt = spill.model.requests.find(2);
      return id != 3 || prompt == spill.model.requests.end() || !prompt->second.resident;
    };
    // Prefill takes 4 ms a row: beside a decoding lane, a command is cut to
    // the floor of 512 rows.
    spill.model.time(test::fourMillisecondRows());
    // With writes in flight, the tier lands every transfer but the writes of
    // the pages the prompt's lane holds, which stay in flight. No restore
    // runs: its writes and transfers pair up in order.
    std::unordered_set<uint32_t> promptPages;
    const auto run = [&](const std::function<bool()> &done, const char *message) {
      for (uint32_t step = 0; step < 2000 && !done(); ++step) {
        static_cast<void>(spill.engine.tick(spill.now++));
        const auto prompt = spill.model.requests.find(2);
        if (writesInFlight && prompt != spill.model.requests.end() && prompt->second.resident)
          for (const uint32_t page : spill.cache.pageTable(2).pages)
            promptPages.insert(page);
        for (size_t index = 0; index < spill.tier.writes.size(); ++index)
          spill.tier.transfers.at(index)->ready =
              !promptPages.contains(spill.tier.writes[index].page);
      }
      require(done(), message);
    };
    spill.engine.submit(Spilling::prompt(1, 1, 5, 200));
    run([&] { return spill.events.outputs.contains(1); }, "the decoding lane did not start");
    // The prompt runs two commands, of 512 and 768 rows; the next request
    // arrives between them, while the prompt's lane holds its pages.
    spill.engine.submit(Spilling::prompt(2, 2, 40, 1));
    run([&] { return spill.model.prefillChunks[2] >= 1; }, "the prompt did not run");
    spill.engine.submit(Spilling::prompt(3, 3, 5, 8));
    run(
        [&] {
          return spill.events.outputs.contains(3) || spill.events.usage.contains(1) ||
                 spill.events.usage.contains(2);
        },
        "no request decoded or ended");
    require(spill.events.outputs.contains(3) && !spill.events.usage.contains(1) &&
                !spill.events.usage.contains(2) &&
                spill.engine.snapshot().resourceSuspensions == 1,
            writesInFlight
                ? "a lane refused its hot floor waited for writes that return no memory"
                : "a lane refused its hot floor waited for the decoding lane or the prompt to end");
    spill.runUntilIdle();
    require(spill.events.completedCount == 3 && spill.events.failedCount == 0 &&
                spill.events.usage.at(3).second == 8,
            "the lanes did not all complete once the prompt had yielded");
  }
}

// A refused floor evicts no KV that cannot empty an extent, and asks again only once memory came
// back.
void testRefusedModelMemoryTakesWhatGivesMemoryBack() {
  // Extents of 64 pages, which the requests share.
  Spilling spill(1024, true, 100'000, 0, Spilling::Extent{64});
  // An earlier conversation leaves its spilled chain cached, its pages
  // holding their index.
  const std::vector<uint32_t> earlier(40 * KvCache::pageTokens + 1, 1);
  spill.engine.submit(request(1, earlier));
  spill.runUntilIdle();
  const uint32_t indexPages = spill.cache.spillSnapshot().spilledPages;
  require(spill.cache.lookup(earlier, {}).kvBoundary == 40 * KvCache::pageTokens && indexPages != 0,
          "the earlier conversation did not leave its spilled chain cached");
  // A lane decodes beside one refused its floor.
  uint32_t attempts = 0;
  bool room = false;
  spill.model.spilledReadsRoom = [&](uint64_t id) {
    attempts += id == 3;
    return id != 3 || room;
  };
  spill.engine.submit(Spilling::prompt(2, 2, 5, 200));
  spill.engine.submit(Spilling::prompt(3, 3, 5, 8));
  spill.runUntil([&] { return attempts != 0; }, "the lane did not ask for its floor");
  require(attempts <= 4 && spill.cache.spillSnapshot().spilledPages >= indexPages,
          "a refused floor evicted Index pages, asking for it again after each");
  room = true;
  spill.runUntilIdle();
  require(spill.events.completedCount == 3 && spill.events.failedCount == 0,
          "the lane refused its floor did not run once it had it");
}

// A lone Spill lane refused its floor waits, rather than fails, while a finished or cancelled
// neighbour's chain is written out leaf by leaf.
void testALoneLaneWaitsForTheChainANeighbourLeft() {
  for (const bool restored : {true, false}) {
    // Extents of 16 pages, which the requests share. Every state goes
    // straight to the SSD. The quota holds the Spill lane's commitment but
    // not the neighbour's, which stays Resident.
    Spilling spill(Spilling::Governed{}, 128, true, Spilling::Extent{16});
    spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
    spill.model.stateTier = std::make_shared<OffloadControl>();
    spill.model.stateTier->ready = true;
    spill.model.restoreControl->ready = true;
    // An earlier turn leaves its chain spilled, its copies written through.
    if (restored) {
      spill.engine.submit(Spilling::prompt(1, 1, 40, 1));
      spill.runUntilIdle();
    }
    // The neighbour starts Resident, as a request does while the SSD cache
    // takes no writes: it restores that chain into Full pages, its copies
    // kept, or none, and runs eight pages of prompt past it.
    spill.tier.writableFile = false;
    spill.engine.submit(Spilling::prompt(2, 1, 48, 10000));
    spill.runUntil([&] { return spill.events.outputs.contains(2); }, "the neighbour did not decode");
    spill.tier.writableFile = true;
    // The Spill lane's floor, three extents' bytes, finds an extent's room
    // when it first asks for it: it fits once two extents have left memory.
    bool asked = false;
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      if (id != 3 || spill.model.spilledReadFloors.contains(id))
        return true;
      if (!std::exchange(asked, true))
        spill.limit(spill.held() + 16 * 100);
      return static_cast<bool>(spill.host.reserve(3 * 16 * 100));
    };
    spill.engine.submit(Spilling::prompt(3, 3, 20, 8));
    spill.runUntil([&] { return asked; }, "the Spill lane did not ask for its floor");
    // The neighbour finishes, or is cancelled.
    if (restored)
      spill.model.decodeFinishes = true;
    else
      spill.engine.cancel(2);
    spill.runUntilIdle();
    require(spill.events.failedCount == 0 && spill.events.outputs.contains(3),
            restored ? "a lone lane refused its floor failed while a finished neighbour's chain was written"
                     : "a lone lane refused its floor failed while a cancelled neighbour's chain was written");
    require(std::ranges::count_if(spill.tier.writes,
                                  [](const test::TestKvTier::Write &write) {
                                    return write.write == WriteClass::Reclaim;
                                  }) > 1,
            "the neighbour's chain left memory without writes one after another");
  }
}

// Memory on its way back to a lane waiting for its floor is not taken by a prompt ranked below it;
// a command between its passes goes on, and no new one starts.
void testAWaitingFloorKeepsTheMemoryOnItsWayBack() {
  for (const uint32_t passes : {1u, 4u}) {
    // Extents of 16 pages. Every state goes straight to the SSD. The quota
    // holds the Spill lane's commitment but not the prompt's, which stays
    // Resident.
    Spilling spill(Spilling::Governed{}, 128, true, Spilling::Extent{16},
                   test::commandShape(model::ExecutionLimits::prefillTokenBudget, passes, passes > 1));
    spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
    spill.model.stateTier = std::make_shared<OffloadControl>();
    spill.model.stateTier->ready = true;
    spill.model.restoreControl->ready = true;
    std::vector<std::pair<uint32_t, uint32_t>> promptPasses;
    spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
      if (plan.kind == WorkKind::Prefill && plan.items.front().requestId == 3)
        promptPasses.emplace_back(plan.pass, plan.passes);
      return test::CommandTimes{};
    };
    spill.model.time(test::passingTarget());
    // While the SSD cache takes no writes, an earlier turn leaves its chain
    // cached without copies, and the prompt starts Resident.
    spill.tier.writableFile = false;
    spill.engine.submit(Spilling::prompt(1, 1, 48, 1));
    spill.runUntilIdle();
    spill.engine.submit(Spilling::prompt(3, 3, 120, 4096));
    spill.runUntil([&] { return spill.model.prefillChunks[3] != 0; }, "the prompt did not run");
    spill.tier.writableFile = true;
    // The Spill lane's first decode command reads; from its second on,
    // which follows the prompt's next command or its first pass, its floor,
    // three extents' bytes, finds an extent's room: it fits once two extents
    // have left memory since it was first refused.
    uint32_t asks = 0;
    bool limited = false;
    size_t refused = 0;
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      if (id != 2 || ++asks == 1)
        return true;
      if (!std::exchange(limited, true)) {
        spill.limit(spill.held() + 16 * 100);
        refused = promptPasses.size();
      }
      return static_cast<bool>(spill.host.reserve(3 * 16 * 100));
    };
    spill.engine.submit(Spilling::prompt(2, 2, 20, 8));
    spill.runUntil(
        [&] {
          return (spill.events.outputs.contains(2) && spill.events.outputs.at(2).size() > 1) ||
                 spill.events.outputs.contains(3);
        },
        "neither the Spill lane nor the prompt decoded");
    require(limited && spill.engine.snapshot().spill.converted == 0,
            "the fixture's floor or quota is off");
    require(spill.events.outputs.contains(2) && !spill.events.outputs.contains(3),
            "a prompt that yields before a lane waiting for its floor took the memory on its way "
            "back");
    const auto waited = std::span(promptPasses).subspan(refused);
    if (passes > 1)
      require(std::ranges::any_of(waited, [](const auto &each) { return each.first != 0; }) &&
                  std::ranges::none_of(waited, [](const auto &each) { return each.first == 0; }),
              "a prompt between its command's passes stopped, or started a command, while a lane it "
              "yields before waited for its floor");
    spill.limit(Spilling::kGovernedLimitBytes);
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    require(spill.events.failedCount == 0 && spill.events.completedCount == 3,
            "the lanes did not all complete");
  }
}

// Of two Spill lanes short of floors, the one that yields first leaves the memory to the waiter,
// and a tick that runs nothing leaves a wakeup that is not yet due.
void testAHeldSpillLaneLeavesTheFloorToTheWaiter() {
  for (const bool holdsFloor : {false, true}) {
    Spilling spill(Spilling::Governed{}, 1024, true, Spilling::Extent{16});
    spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
    spill.model.stateTier = std::make_shared<OffloadControl>();
    spill.model.stateTier->ready = true;
    spill.model.restoreControl->ready = true;
    const auto prompt = [](uint64_t id, uint32_t token, uint32_t pages) {
      EngineRequest result = Spilling::prompt(id, token, pages, 400);
      result.deadlineMilliseconds = 1e9;
      return result;
    };
    spill.tier.writableFile = false;
    spill.engine.submit(prompt(1, 1, 48));
    spill.runUntilIdle();
    spill.tier.writableFile = true;
    // A floor takes an extent beside the KV pool, which the governor admits.
    // Once the floors run short, the first needs two extents back and the
    // second one more.
    constexpr uint64_t floorBytes = 16 * 100;
    spill.model.memory = &spill.host.memory;
    spill.model.admission = spill.host.governor.allocationAdmission();
    spill.model.spilledReadFloorBytes = floorBytes;
    bool budget = false;
    uint32_t refusals = 0;
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      if (spill.model.spilledReadFloors.contains(id))
        return true;
      if (!budget) {
        if (holdsFloor && id == 2)
          return true;
        budget = true;
        spill.limit(spill.held() - floorBytes);
      }
      const bool fits = static_cast<bool>(spill.host.reserve(floorBytes));
      refusals += !fits;
      return fits;
    };
    // The lane with less history is submitted, and so planned, first. Both
    // decode, their pages' writes in flight.
    spill.engine.submit(prompt(2, 2, 4));
    spill.engine.submit(prompt(3, 3, 6));
    for (uint32_t step = 0; step < 100 && !(spill.events.outputs.contains(2) &&
                                            spill.events.outputs.contains(3));
         ++step)
      static_cast<void>(spill.engine.tick(spill.now += 50));
    require(spill.events.outputs.contains(2) && spill.events.outputs.contains(3) && !budget,
            "the lanes did not decode before their pages spilled");
    std::optional<double> shortFloor;
    std::optional<double> longFloor;
    size_t shortOutputs = 0;
    std::optional<size_t> shortOutputsAtRefusal;
    for (uint32_t step = 0; step < 4000 && !(shortFloor && longFloor); ++step) {
      const bool ran = spill.engine.tick(spill.now += 50);
      if (!ran && !spill.engine.commandInFlight()) {
        const std::optional<double> wakeup = spill.engine.nextWakeupMilliseconds();
        require(!wakeup || *wakeup > spill.now,
                "a tick that ran nothing left the loop a wakeup that was due");
      }
      // Writes land every eighth tick.
      if (step % 8 == 0)
        spill.tier.complete();
      // From the refusal that follows the first, a write is on its way.
      if (refusals > 1 && !shortOutputsAtRefusal)
        shortOutputsAtRefusal = spill.events.outputs.at(2).size();
      if (!shortFloor && spill.model.spilledReadFloors.contains(2))
        shortFloor = spill.now;
      if (!longFloor && spill.model.spilledReadFloors.contains(3)) {
        longFloor = spill.now;
        shortOutputs = spill.events.outputs.at(2).size() - shortOutputsAtRefusal.value_or(0);
      }
    }
    require(budget && shortFloor && longFloor && spill.events.failedCount == 0,
            "the lanes did not both get their floors");
    if (!holdsFloor) {
      require(*longFloor < *shortFloor,
              "a Spill lane that yields before one waiting for its floor took the memory on its "
              "way back");
    } else {
      require(shortOutputs > 1, "a lane that holds its floor did not decode while another waited");
    }
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    require(spill.events.failedCount == 0 && spill.events.completedCount == 3,
            "the lanes did not all complete");
  }
}

// A start refused memory on its way back holds no lane: a Resident prompt beside it goes on.
void testAWaitingStartHoldsNoLane() {
  Spilling spill(Spilling::Governed{}, 128, true, Spilling::Extent{16});
  spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  spill.tier.writableFile = false;
  spill.engine.submit(Spilling::prompt(1, 1, 48, 1));
  spill.runUntilIdle();
  spill.engine.submit(Spilling::prompt(3, 3, 120, 4096));
  spill.runUntil([&] { return spill.model.prefillChunks[3] != 0; }, "the prompt did not run");
  spill.tier.writableFile = true;
  // The tier writes one demotion at a time, so that the chain's memory
  // comes back over many ticks and the start still waits for it when the
  // prompt's next command runs.
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Reclaim)] = 1;
  // The start's state, three extents' bytes, finds an extent's room when it
  // is first refused: it fits once two extents have left memory since.
  bool limit = false;
  spill.model.beginGrowthBlocked = [&] {
    if (spill.model.lastBeginId != 2)
      return false;
    if (!std::exchange(limit, true))
      spill.limit(spill.held() + 16 * 100);
    spill.model.beginAllocationFailure = spill.host.reserve(3 * 16 * 100).failure;
    return spill.model.beginAllocationFailure != metal::AllocationFailure::None;
  };
  const size_t pages = spill.cache.pageTable(3).pages.size();
  const auto started = [&] {
    const auto found = spill.model.requests.find(2);
    return found != spill.model.requests.end() && found->second.resident;
  };
  spill.engine.submit(Spilling::prompt(2, 2, 20, 8));
  spill.runUntil([&] { return started() || spill.cache.pageTable(3).pages.size() > pages; },
                 "neither the start nor the prompt went on");
  require(limit && !started() && spill.cache.pageTable(3).pages.size() > pages,
          "a start waiting for memory on its way back held a resident prompt");
  spill.limit(Spilling::kGovernedLimitBytes);
  spill.model.decodeFinishes = true;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 3,
          "the requests did not all complete");
}

// A lane held for a waiting floor ends its command at the state boundary its pages reach, taking
// no page past its capacity.
void testAHeldLaneEndsItsCommandAtItsCapacity() {
  Spilling spill(Spilling::Governed{}, 1024, true, Spilling::Extent{16}, test::commandShape(498));
  spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  const auto prompt = [](uint64_t id, uint32_t token, uint32_t tokens) {
    EngineRequest result = request(id, std::vector<uint32_t>(tokens, token));
    result.maxNewTokens = 400;
    result.deadlineMilliseconds = 1e9;
    return result;
  };
  // An earlier turn leaves its chain cached without copies, its state on
  // the SSD; the states of the turns after it have buffers.
  spill.tier.writableFile = false;
  spill.engine.submit(prompt(1, 1, 48 * KvCache::pageTokens + 1));
  spill.runUntilIdle();
  spill.tier.writableFile = true;
  spill.model.deniedReservations = 0;
  const auto processed = [&](uint64_t id) {
    for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
      if (active.id == id)
        return active.promptProcessed;
    return 0u;
  };
  // The Spill lane's floor, an extent's bytes, finds no room at the engine's
  // limit, which from its first refusal leaves none, until the limit goes.
  bool limited = false;
  spill.model.spilledReadsRoom = [&](uint64_t id) {
    if (id != 2)
      return true;
    if (!std::exchange(limited, true))
      spill.limit(spill.held());
    return static_cast<bool>(spill.host.reserve(16 * 100));
  };
  // The Spill lane decodes, its pages' writes in flight.
  spill.engine.submit(prompt(2, 2, 4 * KvCache::pageTokens + 1));
  for (uint32_t step = 0; step < 100 && !spill.events.outputs.contains(2); ++step)
    static_cast<void>(spill.engine.tick(spill.now += 50));
  require(spill.events.outputs.contains(2) && !spill.cache.holdsIndexPages(2),
          "the Spill lane did not decode before its pages spilled");
  // A prompt starts Resident: its first command runs 498 rows, inside the
  // page that ends at its replay point, 512.
  spill.tier.writableFile = false;
  spill.engine.submit(prompt(4, 4, 524));
  for (uint32_t step = 0; step < 100 && processed(4) == 0; ++step)
    static_cast<void>(spill.engine.tick(spill.now += 50));
  spill.tier.writableFile = true;
  require(processed(4) == 498 && spill.cache.pageTable(4).pages.size() == 16,
          "the long prompt's first command did not end inside its replay point's page");
  // The Spill lane's pages spill; from its next decode command it waits for
  // its floor while a cached leaf's write, which does not land, is in flight.
  spill.tier.complete();
  for (uint32_t step = 0; step < 100 && processed(4) == 498; ++step) {
    static_cast<void>(spill.engine.tick(spill.now += 50));
    for (const auto &transfer : spill.tier.transfers)
      if (transfer->submitted && transfer->write != WriteClass::Reclaim)
        transfer->ready = true;
  }
  require(spill.cache.holdsIndexPages(2) && spill.events.outputs.at(2).size() == 1,
          "the Spill lane did not wait for its floor");
  require(processed(4) == 512 && spill.cache.pageTable(4).pages.size() == 16,
          "a held lane's command took KV pages past its capacity");
  spill.limit(Spilling::kGovernedLimitBytes);
  spill.model.decodeFinishes = true;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 3,
          "the requests did not complete");
}

// An arrival held back for a waiting floor leaves a running command its passes, also when the
// command's lane is cancelled between them.
void testAJoinerHeldBackLeavesACommandItsPasses() {
  for (const bool cancelled : {false, true}) {
    Spilling spill(Spilling::Governed{}, 1024, true, Spilling::Extent{16},
                   test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 4, true));
    spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
    spill.model.stateTier = std::make_shared<OffloadControl>();
    spill.model.stateTier->ready = true;
    spill.model.restoreControl->ready = true;
    std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> prefills;
    spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
      if (plan.kind == WorkKind::Prefill)
        for (const ModelBatchItem &item : items)
          prefills.emplace_back(item.requestId, plan.pass, plan.passes);
      return plan.kind == WorkKind::Decode
                 ? test::CommandTimes{.wallMilliseconds = 100.0}
                 : test::CommandTimes{4900.0 / plan.passes, 3400.0 / plan.passes};
    };
    const auto prompt = [](uint64_t id, uint32_t token, uint32_t tokens) {
      EngineRequest result = request(id, std::vector<uint32_t>(tokens, token));
      result.maxNewTokens = 400;
      result.deadlineMilliseconds = 1e9;
      return result;
    };
    spill.tier.writableFile = false;
    spill.engine.submit(prompt(1, 1, 48 * KvCache::pageTokens + 1));
    spill.runUntilIdle();
    spill.tier.writableFile = true;
    spill.model.deniedReservations = 0;
    // The Spill lane's floor, three extents' bytes, finds no room at the
    // engine's limit, which from its first refusal leaves none, until the
    // limit goes: no more than an extent comes back meanwhile.
    bool limited = false;
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      if (id != 2)
        return true;
      if (!std::exchange(limited, true))
        spill.limit(spill.held());
      return static_cast<bool>(spill.host.reserve(3 * 16 * 100));
    };
    spill.engine.submit(prompt(2, 2, 4 * KvCache::pageTokens + 1));
    for (uint32_t step = 0; step < 100 && !spill.events.outputs.contains(2); ++step)
      static_cast<void>(spill.engine.tick(spill.now += 50));
    require(spill.events.outputs.contains(2) && !spill.cache.holdsIndexPages(2),
            "the Spill lane did not decode before its pages spilled");
    // A short prompt starts Resident and runs its command's first pass.
    spill.model.time(test::passingTarget());
    spill.tier.writableFile = false;
    spill.engine.submit(prompt(3, 3, 60));
    const auto passesOf = [&](uint64_t id, uint32_t pass) {
      return std::ranges::count_if(prefills, [&](const auto &each) {
        return std::get<0>(each) == id && std::get<1>(each) == pass;
      });
    };
    for (uint32_t step = 0; step < 100 && !passesOf(3, 0); ++step)
      static_cast<void>(spill.engine.tick(spill.now += 50));
    require(std::get<2>(prefills.back()) == 4, "the prompt's command did not run in passes");
    spill.tier.writableFile = true;
    // The Spill lane's pages spill; from its next decode command it waits for
    // its floor while a cached leaf's write, which does not land, is in flight.
    spill.tier.complete();
    const size_t outputs = spill.events.outputs.at(2).size();
    spill.engine.submit(prompt(4, 4, 300));
    const auto step = [&] {
      static_cast<void>(spill.engine.tick(spill.now += 50));
      for (const auto &transfer : spill.tier.transfers)
        if (transfer->submitted && transfer->write != WriteClass::Reclaim)
          transfer->ready = true;
    };
    for (uint32_t tick = 0; tick < 200 && !passesOf(3, 1) && passesOf(3, 0) < 2; ++tick)
      step();
    require(passesOf(3, 0) == 1 && passesOf(3, 1) == 1,
            "a command beside a held-back arrival did not keep its passes: its first ran " +
                std::to_string(passesOf(3, 0)) + " times");
    if (cancelled) {
      step();
      spill.engine.cancel(3);
      require(spill.engine.snapshot().cancelled == 1, "a command's only lane cancelled between its passes stayed");
      for (uint32_t tick = 0; tick < 8; ++tick)
        step();
    } else {
      for (uint32_t tick = 0; tick < 200 && !passesOf(3, 3); ++tick)
        step();
      require(passesOf(3, 0) == 1 && passesOf(3, 3) == 1,
              "a command beside a held-back arrival did not run its passes to its last without it");
    }
    require(spill.cache.holdsIndexPages(2) && spill.events.outputs.at(2).size() == outputs,
            "the Spill lane did not wait for its floor");
    require(!passesOf(4, 0), "an arrival took pages while a lane it yields before waited for its floor");
    spill.limit(Spilling::kGovernedLimitBytes);
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    const EngineSnapshot counts = spill.engine.snapshot();
    require(passesOf(4, 0) != 0 && counts.resourceSuspensions == 0 && counts.failed == 0 &&
                counts.cancelled == cancelled && spill.events.completedCount == 4,
            "the arrival did not run once the lane had its floor, or a lane yielded");
  }
}

// A wait for memory that never lands holds the lanes ranked below it for the wait limit, no longer.
void testAWaitThatNeverLandsHoldsLanesUpToItsLimit() {
  Spilling spill(Spilling::Governed{}, 128, true, Spilling::Extent{16});
  spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  // Prefill takes 4 ms a row: beside a decoding lane, a command is cut.
  spill.model.time(test::fourMillisecondRows());
  // No request reaches its deadline here.
  const auto prompt = [](uint64_t id, uint32_t token, uint32_t pages, uint32_t output) {
    EngineRequest result = Spilling::prompt(id, token, pages, output);
    result.deadlineMilliseconds = 1e9;
    return result;
  };
  spill.tier.writableFile = false;
  spill.engine.submit(prompt(1, 1, 40, 1));
  spill.runUntilIdle();
  spill.engine.submit(prompt(4, 4, 24, 100'000));
  spill.runUntil([&] { return spill.events.outputs.contains(4); }, "the decoding lane did not start");
  spill.engine.submit(prompt(3, 3, 120, 4096));
  spill.runUntil([&] { return spill.model.prefillChunks[3] != 0; }, "the prompt did not run");
  spill.tier.writableFile = true;
  // From the Spill lane's second decode command on, its floor, six extents'
  // bytes, finds five extents' room, which the lanes beside it grow into:
  // it fits only once an extent has left memory, which only the cached
  // chain's writes, which never land, would give back. Their landings cover
  // its lack with an extent to spare for the decoding lane, ranked above
  // it and so not held, to grow into over the wait.
  uint32_t asks = 0;
  std::optional<double> refusedAt;
  spill.model.spilledReadsRoom = [&](uint64_t id) {
    if (id != 2 || ++asks == 1)
      return true;
    if (!refusedAt) {
      spill.limit(spill.held() + 5 * 16 * 100);
      refusedAt = spill.now;
    }
    return static_cast<bool>(spill.host.reserve(6 * 16 * 100));
  };
  // Ticks 50 ms apart, landing every transfer but those writes.
  const auto run = [&](const std::function<bool()> &done, const char *message) {
    for (uint32_t step = 0; step < 4000 && !done(); ++step) {
      static_cast<void>(spill.engine.tick(spill.now += 50));
      for (const auto &transfer : spill.tier.transfers)
        if (transfer->submitted && transfer->write != WriteClass::Reclaim)
          transfer->ready = true;
    }
    require(done(), message);
  };
  spill.engine.submit(prompt(2, 2, 20, 8));
  run([&] { return refusedAt.has_value(); }, "the Spill lane was not refused its floor");
  const double wait = kResourceWaitTimeoutMilliseconds;
  const auto held = [&] { return spill.cache.pageTable(3).pages.size(); };
  const size_t pages = held();
  run([&] { return spill.now >= *refusedAt + wait - 100 || held() != pages; },
      "time did not pass");
  require(held() == pages && spill.events.outputs.at(2).size() == 1 &&
              spill.events.failedCount == 0,
          "the prompt grew, or the Spill lane decoded or failed, within the wait's limit");
  run([&] { return held() != pages; },
      "a wait that never landed held the prompt past its limit");
  require(spill.now <= *refusedAt + wait + 1000 && spill.events.outputs.at(2).size() == 1 &&
              spill.events.failedCount == 0,
          "the prompt grew late, or only once the Spill lane's wait had ended");
  spill.tier.complete();
  spill.limit(Spilling::kGovernedLimitBytes);
  spill.model.decodeFinishes = true;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 4,
          "the lanes did not all complete");
}

// A lone Spill lane refused its floor waits for the copies it owes rather than fail: with landings
// the governor collects, behind its dense history's copy, and at a wall short of an Index extent.
void testALoneSpillLaneWaitsForItsOwnCopies() {
  struct Case final {
    uint32_t writes;
    bool governorCollects;
    bool wall;
    const char *message;
  };
  for (const Case &each :
       {Case{4, false, false, "a lone Spill lane failed while its own copies were freeing its memory"},
        Case{4, true, false, "a lone Spill lane failed once the governor's pass collected its copies"},
        Case{1, false, false, "a lone Spill lane failed while its dense history's copy was written"},
        Case{4, false, true, "a lone Spill lane failed at a wall the Index runway could not refill at"}}) {
    // Extents of 16 pages. Every state goes straight to the SSD.
    Spilling spill(Spilling::Governed{}, 1024, true, Spilling::Extent{16});
    spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
    spill.model.stateTier = std::make_shared<OffloadControl>();
    spill.model.stateTier->ready = true;
    spill.model.restoreControl->ready = true;
    // The tier writes a few of the lane's pages at a time, and lands none of
    // them before the lane decodes.
    spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = each.writes;
    bool land = false;
    const auto run = [&](const std::function<bool()> &done, const char *message) {
      for (uint32_t step = 0; step < 4000 && !done(); ++step) {
        static_cast<void>(spill.engine.tick(spill.now++));
        if (!land)
          continue;
        spill.tier.complete();
        if (each.governorCollects && !spill.engine.commandInFlight())
          spill.engine.collectLandings(spill.now);
      }
      require(done(), message);
    };
    // Its floor, two extents' bytes, finds no room when it first asks for
    // it: it fits once two extents have left memory.
    bool asked = false;
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      if (spill.model.spilledReadFloors.contains(id))
        return true;
      if (!std::exchange(asked, true))
        spill.limit(spill.held());
      return static_cast<bool>(spill.host.reserve(2 * 16 * 100));
    };
    spill.engine.submit(Spilling::prompt(1, 1, 60, 64));
    run([&] { return spill.events.outputs.contains(1); }, "the lane did not decode");
    require(!asked && !spill.cache.holdsIndexPages(1), "a page spilled before its copy landed");
    // At the wall, the memory left beside its Full extents and the Index
    // runway holds less than another Index extent.
    if (each.wall)
      spill.limit(spill.held() + 16 * 10 / 2);
    land = true;
    run([&] { return idle(spill.engine); }, "the lane did not end");
    require(asked && spill.events.failedCount == 0 && spill.events.usage.at(1).second == 64, each.message);
  }
}

// A turn that restores a Spill chain without all its copies, refused staging before its first
// command, writes that chain and waits rather than fail.
void testALaneRefusedBeforeItsFirstCommandWritesItsChain() {
  // Extents of 16 pages. Every state goes straight to the SSD. The model's
  // staging of a prompt's history takes memory the governor admits: that of
  // 60 pages is more than an extent's.
  Spilling spill(Spilling::Governed{}, 1024, true, Spilling::Extent{16});
  spill.model.memory = &spill.host.memory;
  spill.model.admission = spill.host.governor.allocationAdmission();
  spill.model.stagingPageBytes = 30;
  spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = 4;
  // The first turn ends after its first token, its copies owed no more, and
  // the writes it started land.
  spill.engine.submit(Spilling::prompt(1, 1, 60, 1));
  spill.runUntilIdle();
  for (uint32_t step = 0; step < 5; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
  }
  require(spill.cache.spillSnapshot().writeThroughInFlight == 0,
          "the first turn's writes were still in flight");
  // The next turn's staging finds no room: it fits once an extent has left
  // memory.
  spill.limit(spill.held());
  spill.engine.submit(Spilling::prompt(2, 1, 60, 64));
  for (uint32_t step = 0; step < 4000 && !idle(spill.engine); ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
  }
  require(idle(spill.engine), "the requests did not end");
  require(spill.events.failedCount == 0 && spill.events.usage.at(2).second == 64,
          "a lone lane refused its staging failed while the chain it restored was written");
}

// A page whose copy has not landed holds back no page after it from spilling.
void testAPageWaitingForItsCopyHoldsBackNoneAfterIt() {
  Spilling spill(1024);
  spill.engine.submit(Spilling::prompt(1, 1, 12, 64));
  // Every write lands but that of the lane's first page past its dense
  // history.
  std::optional<uint32_t> held;
  for (uint32_t step = 0; step < 2000; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    const auto lane = spill.model.requests.find(1);
    if (!held && lane != spill.model.requests.end() && lane->second.resident &&
        spill.cache.pageTable(1).pages.size() > 1)
      held = spill.cache.pageTable(1).pages[1];
    for (size_t index = 0; index < spill.tier.writes.size(); ++index)
      if (spill.tier.transfers.at(index)->submitted && spill.tier.writes[index].page != held)
        spill.tier.transfers.at(index)->ready = true;
    if (spill.events.outputs.contains(1) && spill.cache.spillSnapshot().spilledPages >= 9)
      break;
  }
  require(held && spill.pool.form(*held) == kv::PageForm::Full &&
              spill.cache.spillSnapshot().spilledPages >= 9,
          "a page whose copy had not landed held back the pages after it");
  spill.engine.cancel(1);
  for (uint32_t step = 0; step < 2000 && !idle(spill.engine); ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
  }
  require(idle(spill.engine), "the lane did not end");
}

// A conversion that spills pages a Spill lane shares replans the command, so both lanes read them
// through their floors.
void testAConversionLeavesNoLaneWithoutItsFloor() {
  Spilling spill(1024);
  // The Spill lane writes its copies only once the Resident lane holds them.
  spill.allowSpillWrites(false);
  spill.engine.submit(Spilling::prompt(1, 1, 4, 200));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the Spill lane did not decode");
  // The Resident lane starts while the SSD cache takes no writes.
  spill.tier.writableFile = false;
  spill.engine.submit(Spilling::prompt(2, 1, 4, 200));
  spill.runUntil([&] { return spill.events.outputs.contains(2); }, "the Resident lane did not decode");
  spill.tier.writableFile = true;
  spill.allowSpillWrites(true);
  // Lane 2's next command needs a page that lane 1's does not.
  const auto needsPage = [&](uint64_t id) {
    const uint64_t tokens = 4 * KvCache::pageTokens + 1 + spill.events.outputs[id].size() +
                            model::ExecutionLimits::targetVerifyRows;
    return tokens > uint64_t{spill.cache.pageTable(id).pages.size()} * KvCache::pageTokens;
  };
  spill.runUntil(
      [&] {
        return !spill.engine.commandInFlight() &&
               spill.cache.chainCopies(spill.cache.blockAt(1, 4 * KvCache::pageTokens)).copied == 4 &&
               needsPage(2) && !needsPage(1);
      },
      "lane 2 did not come to need a page that lane 1 does not");
  require(spill.engine.snapshot().spill.spill == 1 && spill.engine.snapshot().spill.resident == 1 &&
              !spill.cache.holdsIndexPages(1) &&
              spill.cache.pageTable(1).pages.front() == spill.cache.pageTable(2).pages.front(),
          "the lanes did not share the prompt, its pages Full beside their copies");
  // At the wall: no free Full page, and no growth.
  spill.storage.growthBlocked = true;
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  const uint64_t replans = spill.engine.snapshot().replans;
  static_cast<void>(spill.engine.tick(spill.now++));
  require(held.granted() && spill.engine.commandInFlight() && spill.engine.snapshot().replans > replans &&
              spill.engine.snapshot().spill.converted == 1 && spill.cache.holdsIndexPages(1) &&
              spill.model.spilledReadFloors.contains(1) && spill.model.spilledReadFloors.contains(2),
          "a lane whose pages a later lane's conversion spilled read them without its floor");
  for (const uint32_t page : held.pages)
    spill.pool.releasePage(page, false);
  spill.storage.growthBlocked = false;
  spill.model.decodeFinishes = true;
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.failedCount == 0,
          "the lanes did not complete after the conversion");
}

// A lone Spill lane the tier takes no more pages of: decoding, it ends with Length at its bound;
// prefilling, it fails naming the SSD cache.
void testLoneSpillLaneEndsAtTheWall() {
  for (const bool decoding : {true, false}) {
    Spilling spill(64);
    // Two Full extents beside the Index runway: eight Full pages.
    spill.storage.budgetPages = 9;
    spill.allowSpillWrites(false);
    spill.engine.submit(Spilling::prompt(1, 1, decoding ? 4 : 10, 200));
    spill.runUntilIdle();
    if (decoding) {
      const uint32_t output =
          spill.events.usage.contains(1) ? spill.events.usage.at(1).second : 0;
      require(spill.events.failedCount == 0 && output > 0 &&
                  4 * KvCache::pageTokens + 1 + output <= 8 * KvCache::pageTokens &&
                  spill.engine.snapshot().spill.lengthEnds == 1 &&
                  spill.events.boundedIds == std::vector<uint64_t>{1},
              "a lone Spill lane at the wall did not end with Length at its bound");
    } else {
      require(spill.events.failures == std::vector<std::string>{"capacity_exhausted"} &&
                  spill.events.failureDetails.front().first.find("SSD cache") !=
                      std::string::npos,
              "a lone Spill prefill at the wall did not fail naming the SSD cache");
    }
    require(spill.committedSlots() == 0, "the lane kept its commitment");
  }
}

// A floor only the host refuses is taken as memory a request in service needs, and the lane
// decodes.
void testLaneDecodesThroughHostRefusedHotFloor() {
  Spilling spill(64);
  spill.model.spilledReadsRefusal = metal::AllocationFailure::HostPressure;
  spill.model.spilledReadsRoom = [&](uint64_t) { return static_cast<bool>(spill.host.reserve(100)); };
  spill.host.pause(true);
  spill.engine.submit(Spilling::prompt(1, 1, 5, 20));
  spill.runUntilIdle();
  require(spill.events.completedCount == 1 && spill.events.failedCount == 0 &&
              spill.events.usage.at(1).second == 20,
          "the lane did not decode through the host's refusal of its hot floor");
}

// Unreadable KV fails only its lane (kv_unreadable) before it emits; the others go on, and the
// block's copy leaves the lookup.
void testUnreadableKvEndsOnlyItsLane() {
  Spilling spill(64);
  // A first request leaves the state at 128 tokens that the failing lane
  // restores.
  spill.engine.submit(Spilling::prompt(30, 3, 4, 2));
  spill.runUntil([&] { return spill.events.outputs.contains(30); },
                 "the first request did not decode");
  const uint64_t unreadable = spill.cache.blockAt(30, 2 * KvCache::pageTokens);
  spill.runUntilIdle();
  const uint64_t invalidations = spill.cache.snapshot().stateCache.invalidations;
  // The failing lane's first step after its restore reads the block, and
  // cannot.
  spill.model.unreadableBlocks[3] = {unreadable};
  for (uint64_t id = 1; id <= 4; ++id)
    spill.engine.submit(Spilling::prompt(id, static_cast<uint32_t>(id), id == 3 ? 6 : 4, 4));
  spill.runUntilIdle();
  require(spill.events.failures == std::vector<std::string>{"kv_unreadable"} &&
              spill.events.failureDetails.front().second && spill.events.completedCount == 4 &&
              !spill.events.outputs.contains(3) &&
              std::ranges::all_of(std::array<uint64_t, 3>{1, 2, 4},
                                  [&](uint64_t id) {
                                    return spill.events.usage.at(id).second == 4;
                                  }),
          "unreadable KV failed other lanes, or its own lane emitted its step");
  require(spill.cache.snapshot().stateCache.invalidations == invalidations,
          "the state the failing lane restored was taken for the failure");
  const std::vector<uint32_t> prompt(6 * KvCache::pageTokens + 1, 3);
  require(spill.cache.lookup(prompt, {}).kvBoundary == KvCache::pageTokens &&
              spill.committedSlots() == 0,
          "a lookup still matched the block whose copy could not be read");
}

// A Spill start commits before its restore; a failed restore returns the commitment, and the start
// runs later from the prefix that remains.
void testFailedRestoreReturnsTheCommitment() {
  Spilling spill(64);
  // The first request's state at 128 tokens goes to the disk tier: no cache
  // slot takes it.
  spill.model.deniedReservations = 1000;
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.engine.submit(Spilling::prompt(1, 1, 4, 1));
  spill.runUntilIdle();
  require(spill.engine.snapshot().diskStatePublications == 1,
          "fixture did not write the replay state to disk");
  spill.engine.submit(Spilling::prompt(2, 1, 6, 1));
  spill.runUntil([&] { return spill.model.diskReads == 1; }, "the start did not restore its state");
  require(spill.committedSlots() > 0, "the start did not commit before its restore");
  // The read fails; the start's next attempt waits for the host.
  bool hostRefuses = true;
  spill.model.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  spill.model.beginGrowthBlocked = [&] { return hostRefuses; };
  spill.model.restoreControl->ready = true;
  spill.model.restoreControl->success = false;
  static_cast<void>(spill.engine.tick(spill.now++));
  require(spill.committedSlots() == 0 && spill.events.startIds.size() == 1,
          "the failed restore kept its commitment");
  hostRefuses = false;
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.starts.back() == 0 &&
              spill.committedSlots() == 0,
          "the start did not run from the prefix that remained");
}

// A KV need takes cached pages of its form in place, with no extent emptied. A Spill lane's landed
// copies free pages in its extents, which a short Resident prompt beside it takes; the prompt
// yields or ends, and the lane's next page comes from the prompt's cached pages: it runs to its end.
void testALaneTakesCachedPagesInPlace() {
  for (const uint64_t room : {0, 50, 100}) {
    Spilling spill(Spilling::Governed{}, 1024, false, Spilling::Extent{4}, test::commandShape(KvCache::pageTokens));
    spill.limit(spill.held() + 1'100);
    EngineRequest lane = Spilling::prompt(2, 2, 12, 200);
    lane.deadlineMilliseconds = 1e9;
    spill.engine.submit(std::move(lane));
    // Its copies stay in flight until the prompt arrives.
    for (int tick = 0; tick < 27; ++tick)
      static_cast<void>(spill.engine.tick(spill.now++));
    spill.limit(spill.held() + room);
    spill.engine.submit(Spilling::prompt(1, 1, 3, 1));
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    require(spill.events.failedCount == 0 && spill.events.completedCount == 2,
            "beside a prompt, in room for " + std::to_string(room) +
                " B, a lane failed while cached pages fit its next page");
  }
}

// With no Spill lane in service a refused Index runway refill is asked only in the control pass,
// not at every tick or command.
void testWithNoSpillLaneTheRunwayRefillsInTheControlPass() {
  Spilling spill(1024, false);
  require(spill.pool.reclaimEmptyExtents(false, 1, kv::PageForm::Index) &&
              !spill.pool.allocatedExtents(kv::PageForm::Index),
          "fixture: the Index runway stayed");
  uint32_t refills = 0;
  spill.storage.growthAllowed = [&](uint32_t) {
    for (const auto &[_, state] : spill.host.governor.ledger().claims())
      if (!state.spec.requestId && state.spec.cls == ClaimClass::Optional &&
          state.spec.timing == NeedTiming::Immediate) {
        ++refills;
        return false;
      }
    return true;
  };
  spill.engine.submit(Spilling::prompt(1, 1, 3, 16));
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(spill.events.completedCount == 1 && counts.resident == 1 && counts.spill == 0,
          "fixture: the request did not run Resident");
  require(!refills, "with no Spill lane in service the refused Index runway refill was asked " +
                        std::to_string(refills) + " times in ticks and commands");
}

// A disk-only chain restored into Full and Index pages at the wall keeps the free pages its refused
// attempt took, and the start runs from its cached prefix.
void testARestoreOfTwoFormsKeepsTheFreePagesItTakes() {
  Spilling spill(Spilling::Governed{}, 1024, true, Spilling::Extent{16});
  spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  // The first turn's chain is written through and its state goes to the SSD;
  // a critical pass leaves the chain only on disk.
  spill.engine.submit(Spilling::prompt(1, 1, 40, 1));
  spill.runUntilIdle();
  static_cast<void>(test::criticalPass(spill.host.governor));
  // An idle chain that fills two Full extents, no copy of which the SSD
  // cache took: no Full page is free.
  spill.tier.writableFile = false;
  EngineRequest idleChain = request(3, std::vector<uint32_t>(32 * KvCache::pageTokens, 3));
  idleChain.maxNewTokens = 1;
  spill.engine.submit(std::move(idleChain));
  spill.runUntilIdle();
  spill.tier.writableFile = true;
  static_cast<void>(spill.pool.reclaimEmptyExtents(false, std::numeric_limits<uint32_t>::max(), kv::PageForm::Full));
  require(!spill.pool.freePageCount(kv::PageForm::Full) && spill.pool.allocatedExtents(kv::PageForm::Full) == 2,
          "fixture: " + std::to_string(spill.pool.freePageCount(kv::PageForm::Full)) + " free Full pages in " +
              std::to_string(spill.pool.allocatedExtents(kv::PageForm::Full)) + " extents");
  // Room for the Full extent the restore's dense history takes and half an
  // Index extent beyond the Index pages free.
  spill.limit(spill.held() + 1600 + 80);
  const uint64_t idle = spill.held();
  spill.engine.submit(Spilling::prompt(2, 1, 40, 1));
  spill.runUntilIdle();
  const auto start = std::ranges::find(spill.events.startIds, 2u);
  const uint32_t matched =
      start == spill.events.startIds.end() ? 0 : spill.events.starts[start - spill.events.startIds.begin()];
  require(spill.events.failedCount == 0 && matched == 40 * KvCache::pageTokens,
          "a start whose chain restores in two forms at the wall did not run from its cached prefix: matched " +
              std::to_string(matched) + " tokens, " + std::to_string(spill.events.failedCount) +
              " failures, idle memory " + std::to_string(idle) + " B");
}

// A Spill prompt preempted past its first checkpoint resumes from its durable prefix, never from 0,
// though its copies land a tick late.
void testPreemptedSpillPrefillResumesFromItsDurablePrefix() {
  Spilling spill(Spilling::Governed{}, 1024, true, Spilling::Extent{4}, test::commandShape(KvCache::pageTokens));
  spill.model.memory = &spill.host.memory;
  spill.model.admission = spill.host.governor.allocationAdmission();
  // Each lane's state takes buffers of its own from the governor.
  spill.model.laneFootprintBytes = 1000;
  // Copies started before the tick before this one land now.
  size_t landing = 0;
  const auto tick = [&] {
    for (size_t index = 0; index < landing; ++index)
      if (!spill.tier.transfers[index]->finished && spill.tier.transfers[index]->submitted)
        spill.tier.transfers[index]->ready = true;
    landing = spill.tier.transfers.size();
    static_cast<void>(spill.engine.tick(spill.now++));
  };
  const auto runUntil = [&](const std::function<bool()> &done, const char *message) {
    for (uint32_t step = 0; step < 4000 && !done(); ++step)
      tick();
    require(done(), message);
  };
  EngineRequest background = Spilling::prompt(1, 1, 200, 4);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  runUntil([&] { return spill.model.prefillChunks[1] == 130 && !spill.engine.commandInFlight(); },
           "the prompt did not run 130 commands");
  // Room for the Foreground request's states but not for its lane's: that
  // comes only from the prompt, which yields it.
  spill.limit(spill.held() + 200);
  EngineRequest foreground = Spilling::prompt(2, 2, 1, 1);
  foreground.priority = RequestPriority::Foreground;
  spill.engine.submit(std::move(foreground));
  runUntil([&] { return spill.events.usage.contains(2); }, "the Foreground request did not run");
  require(spill.model.suspensions == 1, "the prompt was not suspended for the Foreground start");
  const uint64_t restored = spill.model.restored;
  const uint32_t commands = spill.model.prefillChunks[1];
  spill.limit(Spilling::kGovernedLimitBytes);
  runUntil([&] { return idle(spill.engine); }, "the requests did not end");
  require(spill.model.resumptions == 1 && spill.model.restored - restored >= 4096,
          "the preempted prompt did not resume from its durable prefix: it restored " +
              std::to_string(spill.model.restored - restored) + " tokens (resume asked " +
              std::to_string(spill.model.lastRestoredTokens) + "), and ran " +
              std::to_string(spill.model.prefillChunks[1] - commands) + " more prefill commands");
  require(spill.events.failedCount == 0 && spill.events.completedCount == 2 && spill.events.usage.at(1).second == 4,
          "a request failed");
}

// Placement counts no expert store unit: 17 Full pages in 1,000 bytes of room spill, though a
// 2,000-byte unit above the store's floor would hold them.
void testPlacementCountsNoExpertStore() {
  Spilling governed(Spilling::Governed{}, 1024, false);
  test::FakeElasticPool experts("experts", governed.host.memory, 2'000, 0, 1, 1, 1);
  static_cast<void>(governed.host.governor.ledger().addPool(experts));
  governed.limit(governed.held() + 1'000);
  governed.engine.submit(Spilling::prompt(1, 1, 16, 1));
  governed.runUntilIdle();
  const SpillRequestsSnapshot placed = governed.engine.snapshot().spill;
  require(placed.resident == 0 && placed.spill == 1 && governed.events.failedCount == 0,
          "a prompt whose Resident KV only the expert store made room for started Resident");
}

// A prompt whose Spill footprint is no smaller than its Resident one starts Resident.
void testPlacementNeverSpillsWhenSpillTakesMore() {
  // Resident KV holds 2 pages; five pages and a token of prompt and a token
  // of output project 6 pages, all of which a Spill lane keeps Full.
  Spilling spill(1024, false, 2 * 100);
  spill.engine.submit(Spilling::prompt(1, 1, 5, 1));
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(counts.resident == 1 && counts.spill == 0 && spill.events.failedCount == 0,
          "a prompt whose Spill footprint is no smaller than its Resident one spilled");
}

// Placement prices a Spill start's hot block floor in full, also while idle extents above the floors
// would hold it, as a start may take those: a prompt whose Spill floor beside its hot floor takes more
// than its Resident KV starts Resident in a room in which it spills without a hot floor.
void testPlacementCountsTheHotFloorIdleExtentsWouldHold() {
  struct Case final {
    uint64_t floor, stale;
    bool spills;
  };
  for (const Case &each : {Case{0, 0, true}, Case{1'000, 0, false}, Case{1'000, 1'000, false}}) {
    Spilling spill(Spilling::Governed{}, 1024, false);
    spill.model.spilledReadFloorBytes = each.floor;
    spill.model.staleHotBytes = each.stale;
    spill.limit(spill.held() + 1'500);
    // Ten pages of prompt and ten of output.
    spill.engine.submit(Spilling::prompt(1, 1, 9, 10 * KvCache::pageTokens));
    spill.runUntil([&] { return std::ranges::find(spill.events.startIds, 1) != spill.events.startIds.end(); },
                   "the prompt did not start");
    const SpillRequestsSnapshot placed = spill.engine.snapshot().spill;
    spill.limit(Spilling::kGovernedLimitBytes);
    spill.runUntilIdle();
    const std::string name = "a " + std::to_string(each.floor) + "-byte hot floor beside " +
                             std::to_string(each.stale) + " idle bytes";
    require(spill.events.failedCount == 0 && spill.events.completedCount == 1, "the prompt with " + name + " failed");
    require(placed.spill == (each.spills ? 1U : 0U) && placed.resident == (each.spills ? 0U : 1U),
            "the prompt with " + name + (each.spills ? " started Resident" : " spilled"));
  }
}

// An admission that changes the ledger places the next candidate again: of two prompts that fit
// one at a time, the second starts Spill.
void testAnAdmissionPlacesTheNextCandidateAgain() {
  Spilling spill(Spilling::Governed{}, 1024, false);
  spill.limit(spill.held() + 2'500);
  spill.engine.submit(Spilling::prompt(1, 1, 16, 1));
  spill.engine.submit(Spilling::prompt(2, 2, 16, 1));
  spill.runUntil(
      [&] {
        const SpillRequestsSnapshot started = spill.engine.snapshot().spill;
        return started.resident + started.spill == 2;
      },
      "the prompts did not start");
  spill.limit(Spilling::kGovernedLimitBytes);
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(counts.resident == 1 && counts.spill == 1 && spill.events.failedCount == 0,
          "a candidate started on a placement made before another start took its lane: " +
              std::to_string(counts.resident) + " Resident, " + std::to_string(counts.spill) + " Spill");
}

// Placement counts the idle state buffers a start frees: 17 Full pages fit 1,500 bytes of room
// beside 1,000 idle bytes.
void testPlacementCountsIdleStateBuffers() {
  Spilling spill(Spilling::Governed{}, 1024, false);
  spill.host.memory.hold("states", 1'000);
  spill.model.heldStateBytes += 1'000;
  spill.model.reclaimableIdleStateBytes = 1'000;
  spill.limit(spill.held() + 1'500);
  spill.engine.submit(Spilling::prompt(1, 1, 16, 1));
  spill.runUntil(
      [&] {
        const SpillRequestsSnapshot started = spill.engine.snapshot().spill;
        return started.resident + started.spill == 1;
      },
      "the prompt did not start");
  spill.limit(Spilling::kGovernedLimitBytes);
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(counts.resident == 1 && counts.spill == 0 && spill.events.failedCount == 0,
          "a prompt the room held beside the idle state buffers spilled");
}

// Placement counts what the lane's claims take, its KV in whole extents
// and its lane's states, each of which alone decides a case of its own in a
// room of 15.5 Flash-Next extents: a prompt either passes the room spills
// and finishes, and one both leave room for runs Resident. Placed by its
// pages alone, a prompt of the first kind ran Resident and failed.
void testPlacementCountsWholeExtentsAndTheLanesStates() {
  constexpr uint64_t extent = 256 * FlashNextLanes::kPages.fullBytes;
  constexpr uint64_t laneStates = 118'751'232;
  struct Case final {
    uint32_t pages;
    uint64_t states;
    bool spills;
  };
  // 15 whole extents beside the states; 15.25 extents of pages without them; 14 and the states.
  for (const Case &each : {Case{3'839, laneStates, true}, Case{3'903, 0, true}, Case{3'583, laneStates, false}}) {
    Spilling spill{16'384, false, 0, 200'000, Spilling::Extent{256}, {}, {.limitBytes = 4ULL << 30}, true,
                   FlashNextLanes::kPages};
    spill.model.laneFootprintBytes = each.states;
    spill.host.memory.hold("limit", (4ULL << 30) - spill.held() - 31 * extent / 2);
    spill.engine.submit(Spilling::prompt(1, 1, each.pages, 1));
    spill.runUntil([&] { return std::ranges::find(spill.events.startIds, 1) != spill.events.startIds.end(); },
                   "the prompt did not start");
    const SpillRequestsSnapshot placed = spill.engine.snapshot().spill;
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    const std::string name =
        std::to_string(each.pages) + " pages beside " + std::to_string(each.states) + " B of states";
    require(spill.events.failedCount == 0 && spill.events.completedCount == 1, "a prompt of " + name + " failed");
    require(placed.spill == (each.spills ? 1U : 0U) && placed.resident == (each.spills ? 0U : 1U),
            "a prompt of " + name + (each.spills ? " started Resident" : " spilled"));
  }
}

// Placement counts the states a start's claims evict against the pool the
// start leaves: they refill the lane's buffers it took before any of theirs
// frees memory. Beside three cached conversations' states and a lane's
// buffers in the pool, a 20-page prompt prefilled a page a command spills in
// a room short of what its start takes, and runs Resident in that room plus
// the lane's buffers and a state. Counted against the pool before the start,
// it started Resident in the first room and failed at its KV.
void testPlacementCountsTheLanesBuffersAStartTakesFromThePool() {
  constexpr uint64_t lane = 128, state = 64, shortRoom = 1'100;
  for (const uint64_t room : {shortRoom, shortRoom + lane + state}) {
    Spilling spill(Spilling::Governed{}, 1024, false, Spilling::Extent{4}, test::commandShape(KvCache::pageTokens));
    spill.model.laneFootprintBytes = lane;
    spill.model.decodeFinishes = true;
    for (uint32_t each = 2; each < 5; ++each) {
      spill.engine.submit(Spilling::prompt(each, each, 3, 1));
      spill.runUntilIdle();
    }
    require(spill.model.pooledLaneBytes == lane && spill.host.memory.held("states") == lane + 3 * state,
            "the conversations did not leave a lane's buffers and three states");
    spill.limit(spill.held() + room);
    spill.engine.submit(Spilling::prompt(1, 1, 20, 1));
    spill.runUntil([&] { return std::ranges::find(spill.events.startIds, 1) != spill.events.startIds.end(); },
                   "the prompt did not start");
    const SpillRequestsSnapshot placed = spill.engine.snapshot().spill;
    spill.runUntilIdle();
    const std::string name = "the prompt in room for " + std::to_string(room) + " B";
    require(spill.events.failedCount == 0 && spill.events.completedCount == 4, name + " failed");
    require(placed.spill == (room == shortRoom ? 1U : 0U), name + (room == shortRoom ? " started Resident" : " spilled"));
  }
}

// A conversation's next turn starts from the pool its last turn left, a
// lane's buffers, and its claims evict no state: placement charges it the
// state it captures alone, and counts as room only the idle buffers beyond
// those its start takes. A 20-page next turn prefilled a page a command,
// beside no idle buffers and beside a second lane's, runs Resident in room
// for its pages and that state, and spills in room for its pages alone.
// Charged the lane's buffers whole, it spilled in both rooms; counting the
// idle buffers its start takes as room, it started Resident in the second.
void testPlacementCountsANextTurnsStartAtWhatItAllocates() {
  constexpr uint64_t lane = 128, state = 64;
  for (const uint64_t idle : {uint64_t{0}, lane})
    for (const bool fits : {false, true}) {
      Spilling spill(Spilling::Governed{}, 1024, false, Spilling::Extent{4}, test::commandShape(KvCache::pageTokens));
      spill.model.laneFootprintBytes = lane;
      spill.model.decodeFinishes = true;
      spill.engine.submit(Spilling::prompt(2, 2, 3, 1));
      spill.runUntilIdle();
      require(spill.model.pooledLaneBytes == lane && spill.host.memory.held("states") == lane + state,
              "the first turn did not leave a lane's buffers and its state");
      spill.host.memory.hold("states", idle);
      spill.model.heldStateBytes += idle;
      spill.model.reclaimableIdleStateBytes = idle;
      spill.limit(spill.held() + 20 * spill.pool.bytesPerPage() + (fits ? state : 0));
      spill.engine.submit(Spilling::prompt(1, 2, 20, 1));
      spill.runUntil([&] { return std::ranges::find(spill.events.startIds, 1) != spill.events.startIds.end(); },
                     "the next turn did not start");
      const SpillRequestsSnapshot placed = spill.engine.snapshot().spill;
      spill.runUntilIdle();
      const std::string name =
          std::string(fits ? "the next turn in room for its state" : "the next turn without room for its state") +
          (idle ? " beside idle buffers" : "");
      require(spill.events.failedCount == 0 && spill.events.completedCount == 2, name + " failed");
      require(placed.spill == (fits ? 0U : 1U), name + (fits ? " spilled" : " started Resident"));
    }
}

// Placement counts the Resident lanes' projected growth with the prompt's,
// in whole extents past the free pages, as their claims take both from the
// same extents: beside a lane that projects 6 pages more, a prompt that
// fills whole extents spills in room half an extent short of the extents
// both take, and runs Resident in room for them, each beside the state its
// start captures. Counted in pages beside the prompt's extents, the lane's
// growth fit the first room.
void testPlacementCountsTheLanesGrowthInTheSameExtents() {
  constexpr uint64_t state = 64, shortRoom = 4'600 + state;
  for (const uint64_t room : {shortRoom, 4'800 + state}) {
    Spilling spill(Spilling::Governed{}, 1024, false);
    spill.model.time(test::passingTarget());
    EngineRequest lane = Spilling::prompt(3, 3, 7, 200);
    lane.deadlineMilliseconds = 1e9;
    spill.engine.submit(std::move(lane));
    spill.runUntil([&] { return spill.events.outputs.contains(3) && !spill.engine.commandInFlight(); },
                   "the lane did not decode");
    require(spill.pool.freePageCount(kv::PageForm::Full) == 0, "the lane left free pages");
    spill.limit(spill.held() + room);
    spill.engine.submit(Spilling::prompt(1, 1, 39, 1));
    spill.runUntil(
        [&] {
          const SpillRequestsSnapshot started = spill.engine.snapshot().spill;
          return started.resident + started.spill == 2;
        },
        "the prompt did not start");
    const SpillRequestsSnapshot placed = spill.engine.snapshot().spill;
    spill.limit(Spilling::kGovernedLimitBytes);
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    require(spill.events.failedCount == 0, "a request failed");
    require(room == shortRoom ? placed.spill == 1 : placed.resident == 2,
            "beside a lane's growth, a prompt in room for " + std::to_string(room) + " B " +
                (room == shortRoom ? "started Resident" : "spilled"));
  }
}

// Placement needs only the KV its restored chain does not hold in memory, and counts that chain as
// no room: a cached follow-up starts Resident, uncached or past the room it spills.
void testPlacementCountsTheChainItRestoresInMemory() {
  for (const bool cached : {true, false}) {
    Spilling spill(1024, false, 60 * 100, 0, {4}, {}, {.limitBytes = Spilling::kGovernedLimitBytes}, true);
    if (cached) {
      spill.engine.submit(Spilling::prompt(1, 1, 32, 1));
      spill.runUntilIdle();
    }
    spill.limit(spill.held() + 26 * 100);
    spill.engine.submit(Spilling::prompt(2, 1, 36, 1));
    spill.runUntil([&] { return std::ranges::find(spill.events.startIds, 2) != spill.events.startIds.end(); },
                   "the turn did not start");
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    if (cached)
      require(counts.resident == 2 && counts.spill == 0 && spill.events.starts.back() == 32 * KvCache::pageTokens,
              "a follow-up turn spilled though the chain it restored held its first 32 pages in memory");
    else
      require(counts.resident == 0 && counts.spill == 1, "a prompt with nothing cached started Resident beside room "
                                                          "for less than its projection");
    spill.engine.cancel(2);
    spill.runUntilIdle();
  }
  {
    Spilling spill(1024, false, 60 * 100, 0, {4}, {}, {.limitBytes = Spilling::kGovernedLimitBytes}, true);
    auto states = std::make_shared<OffloadControl>();
    states->ready = true;
    spill.model.offloadTier = states;
    spill.engine.submit(Spilling::prompt(1, 1, 32, 1));
    spill.runUntilIdle();
    const EngineRequest turn = Spilling::prompt(2, 1, 56, 1);
    const CacheProbe probe = spill.cache.probe(turn.prompt, {});
    require(probe.restoredChain().size() == 32, "the follow-up turn did not find its first 32 pages cached");
    const uint64_t point = probe.restoredChain().back();
    PersistStatus written = spill.cache.persist(point);
    for (int step = 0; step < 8 && written != PersistStatus::Durable; ++step) {
      spill.tier.complete();
      static_cast<void>(spill.cache.pollTransfers());
      written = spill.cache.persist(point);
    }
    require(written == PersistStatus::Durable, "the chain's point was not written ahead");
    spill.limit(spill.held() + 20 * 100);
    spill.engine.submit(turn);
    spill.runUntil([&] { return std::ranges::find(spill.events.startIds, 2) != spill.events.startIds.end(); },
                   "the turn did not start");
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    require(counts.resident == 1 && counts.spill == 1 && counts.converted == 0,
            "a follow-up turn started Resident on the room its own chain's copies would free");
    spill.engine.cancel(2);
    spill.runUntilIdle();
  }
}

} // namespace

std::span<const test::TestCase> spillTests() {
  static constexpr test::TestCase kTests[] = {
      {"testPlacementCountsNoExpertStore", testPlacementCountsNoExpertStore},
      {"testPlacementNeverSpillsWhenSpillTakesMore", testPlacementNeverSpillsWhenSpillTakesMore},
      {"testPlacementCountsTheHotFloorIdleExtentsWouldHold", testPlacementCountsTheHotFloorIdleExtentsWouldHold},
      {"testAnAdmissionPlacesTheNextCandidateAgain", testAnAdmissionPlacesTheNextCandidateAgain},
      {"testPlacementCountsIdleStateBuffers", testPlacementCountsIdleStateBuffers},
      {"testPlacementCountsTheChainItRestoresInMemory", testPlacementCountsTheChainItRestoresInMemory},
      {"testPlacementCountsWholeExtentsAndTheLanesStates", testPlacementCountsWholeExtentsAndTheLanesStates},
      {"testPlacementCountsTheLanesGrowthInTheSameExtents", testPlacementCountsTheLanesGrowthInTheSameExtents},
      {"testPlacementCountsTheLanesBuffersAStartTakesFromThePool", testPlacementCountsTheLanesBuffersAStartTakesFromThePool},
      {"testPlacementCountsANextTurnsStartAtWhatItAllocates", testPlacementCountsANextTurnsStartAtWhatItAllocates},
      {"testAClosedTierRefusesAStartPastTheResidentContext", testAClosedTierRefusesAStartPastTheResidentContext},
      {"testAdmissionSpillsWhatMemoryCannotHold", testAdmissionSpillsWhatMemoryCannotHold},
      {"testSpillCommitmentCoversTheChainWithoutCopies", testSpillCommitmentCoversTheChainWithoutCopies},
      {"testConversionCommitsTheHistoryWithoutCopies", testConversionCommitsTheHistoryWithoutCopies},
      {"testConversionSpillsAHistoryWithCopies", testConversionSpillsAHistoryWithCopies},
      {"testAConvertedLaneResumesUnderTheSpillRules", testAConvertedLaneResumesUnderTheSpillRules},
      {"testAConvertedLanesMovesReleaseTheExtentsTheyEmpty", testAConvertedLanesMovesReleaseTheExtentsTheyEmpty},
      {"testALongDecodingLaneConvertsItselfAtTheWall", testALongDecodingLaneConvertsItselfAtTheWall},
      {"testALaneConvertsOnlyWithAnIndexRunway", testALaneConvertsOnlyWithAnIndexRunway},
      {"testAGrowthTheHostRefusesDoesNotConvert", testAGrowthTheHostRefusesDoesNotConvert},
      {"testALaneFreeingLessThanItsNeedDoesNotConvert", testALaneFreeingLessThanItsNeedDoesNotConvert},
      {"testALongLaneConvertsItselfBesideAnother", testALongLaneConvertsItselfBesideAnother},
      {"testASharedPrefixCountsForNoConversion", testASharedPrefixCountsForNoConversion},
      {"testAShortOrPrefillingLaneDoesNotConvert", testAShortOrPrefillingLaneDoesNotConvert},
      {"testALanesGrowthConvertsNoOtherLane", testALanesGrowthConvertsNoOtherLane},
      {"testLaneDemandCountsTheHotFloorALaneWaitsFor", testLaneDemandCountsTheHotFloorALaneWaitsFor},
      {"testAStartRefusedMemoryCountsNoConversion", testAStartRefusedMemoryCountsNoConversion},
      {"testShortQuotaBoundsWaitsOrRefusesASpillStart", testShortQuotaBoundsWaitsOrRefusesASpillStart},
      {"testAQuotaWaitHoldsBackOnlyStartsCompetingForIt", testAQuotaWaitHoldsBackOnlyStartsCompetingForIt},
      {"testADeclinedQuotaRefusalKeepsItsScope", testADeclinedQuotaRefusalKeepsItsScope},
      {"testAResidentStartDoesNotHoldAQuotaWaitOpen", testAResidentStartDoesNotHoldAQuotaWaitOpen},
      {"testALaneHoldingCopiesHoldsAQuotaWaitOpen", testALaneHoldingCopiesHoldsAQuotaWaitOpen},
      {"testAQuotaRefusalHoldsBackAStartRestoringUnheldCopies",
       testAQuotaRefusalHoldsBackAStartRestoringUnheldCopies},
      {"testAQuotaRefusalReadsOnlyTheChainAStartKeeps", testAQuotaRefusalReadsOnlyTheChainAStartKeeps},
      {"testAStartRestoringHeldCopiesStartsBesideAQuotaRefusal",
       testAStartRestoringHeldCopiesStartsBesideAQuotaRefusal},
      {"testASpillStartCountsTheCopiesItRestores", testASpillStartCountsTheCopiesItRestores},
      {"testAStartThatCannotSpillStartsResidentWithinTheResidentContext",
       testAStartThatCannotSpillStartsResidentWithinTheResidentContext},
      {"testWriteBudgetRefusalSaysWhenToRetry", testWriteBudgetRefusalSaysWhenToRetry},
      {"testPrefillGroupsFollowIndexPages", testPrefillGroupsFollowIndexPages},
      {"testALaneThatTakesSpilledBlocksAsItPrefillsRunsPrefillGroups",
       testALaneThatTakesSpilledBlocksAsItPrefillsRunsPrefillGroups},
      {"testSpilledFullExtentsLeaveOnceNoPromptGrows", testSpilledFullExtentsLeaveOnceNoPromptGrows},
      {"testSpillMovesThroughTheHostPause", testSpillMovesThroughTheHostPause},
      {"testSpillResumeCommitsAgain", testSpillResumeCommitsAgain},
      {"testABoundAtStartAndResumeCountsOnce", testABoundAtStartAndResumeCountsOnce},
      {"testSpillResumeOnAClosedTier", testSpillResumeOnAClosedTier},
      {"testSpillResumeBoundLeavesOutputPastItsAnchor", testSpillResumeBoundLeavesOutputPastItsAnchor},
      {"testSpillResumeInItsPromptHoldsNoAnchor", testSpillResumeInItsPromptHoldsNoAnchor},
      {"testSpillResumeTakesItsReplaysPagesAsItRuns", testSpillResumeTakesItsReplaysPagesAsItRuns},
      {"testAWaitingResumeFindsTheMemoryItsKvHeldFree", testAWaitingResumeFindsTheMemoryItsKvHeldFree},
      {"testDecodeLeavesOutALaneRefusedItsHotFloor", testDecodeLeavesOutALaneRefusedItsHotFloor},
      {"testARefusedDecodeTakesThePrefillLanesMemory", testARefusedDecodeTakesThePrefillLanesMemory},
      {"testRefusedModelMemoryTakesWhatGivesMemoryBack", testRefusedModelMemoryTakesWhatGivesMemoryBack},
      {"testALoneLaneWaitsForTheChainANeighbourLeft", testALoneLaneWaitsForTheChainANeighbourLeft},
      {"testAWaitingFloorKeepsTheMemoryOnItsWayBack", testAWaitingFloorKeepsTheMemoryOnItsWayBack},
      {"testAWaitThatNeverLandsHoldsLanesUpToItsLimit", testAWaitThatNeverLandsHoldsLanesUpToItsLimit},
      {"testAHeldSpillLaneLeavesTheFloorToTheWaiter", testAHeldSpillLaneLeavesTheFloorToTheWaiter},
      {"testAWaitingStartHoldsNoLane", testAWaitingStartHoldsNoLane},
      {"testAHeldLaneEndsItsCommandAtItsCapacity", testAHeldLaneEndsItsCommandAtItsCapacity},
      {"testAJoinerHeldBackLeavesACommandItsPasses", testAJoinerHeldBackLeavesACommandItsPasses},
      {"testALoneSpillLaneWaitsForItsOwnCopies", testALoneSpillLaneWaitsForItsOwnCopies},
      {"testALaneRefusedBeforeItsFirstCommandWritesItsChain", testALaneRefusedBeforeItsFirstCommandWritesItsChain},
      {"testAPageWaitingForItsCopyHoldsBackNoneAfterIt", testAPageWaitingForItsCopyHoldsBackNoneAfterIt},
      {"testAConversionLeavesNoLaneWithoutItsFloor", testAConversionLeavesNoLaneWithoutItsFloor},
      {"testLoneSpillLaneEndsAtTheWall", testLoneSpillLaneEndsAtTheWall},
      {"testLaneDecodesThroughHostRefusedHotFloor", testLaneDecodesThroughHostRefusedHotFloor},
      {"testUnreadableKvEndsOnlyItsLane", testUnreadableKvEndsOnlyItsLane},
      {"testFailedRestoreReturnsTheCommitment", testFailedRestoreReturnsTheCommitment},
      {"testARestoreOfTwoFormsKeepsTheFreePagesItTakes", testARestoreOfTwoFormsKeepsTheFreePagesItTakes},
      {"testALaneTakesCachedPagesInPlace", testALaneTakesCachedPagesInPlace},
      {"testWithNoSpillLaneTheRunwayRefillsInTheControlPass", testWithNoSpillLaneTheRunwayRefillsInTheControlPass},
      {"testPreemptedSpillPrefillResumesFromItsDurablePrefix", testPreemptedSpillPrefillResumesFromItsDurablePrefix},
  };
  return kTests;
}
