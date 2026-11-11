#include "EngineTestSupport.hpp"
#include "FakeElasticPool.hpp"
#include "MemoryPasses.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "TestRegistry.hpp"
#include "TestWriteBudget.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

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
// copies and grows into the freed pages; a quota short of the history refuses it, and it fails,
// naming its lane Resident in decode: none of its KV went to the SSD cache.
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
                  spill.events.failures == std::vector<std::string>{"capacity_exhausted"},
              "a conversion the quota could not hold with the history went ahead");
      const std::string &message = spill.events.failureDetails.front().first;
      require(message.find(", a Resident lane of ") != std::string::npos &&
                  message.find(" tokens in decode, ") != std::string::npos &&
                  message.find("SSD cache") == std::string::npos,
              "the lane the quota refused did not fail naming its mode and phase alone: " + message);
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

// The KV mode of the request's lane, Resident before it has one.
KvMode kvMode(const Spilling &spill, uint64_t id) {
  for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
    if (active.id == id)
      return active.kvMode;
  return KvMode::Resident;
}

// Flash-Next's proportions: 256-page extents, 423,936 B Full and 24,576 B Index pages, a 390 MiB hot
// block floor, and a dense history and tail window of a page each. Its page ids hold every extent the
// lanes reach, so memory limits the pool and never the ids, as production's hold every extent the limit
// could hold of Index pages: 8,192 held 32 extents, and the runway's refill failed for ids.
struct FlashNextLanes {
  static constexpr Spilling::Pages kPages{.count = 65'536, .fullBytes = 423'936, .indexBytes = 24'576};
  Spilling spill{16'384, false, 0, 200'000, Spilling::Extent{256}, {}, {}, false, kPages};
  FlashNextLanes() { spill.model.spilledReadFloorBytes = 390ULL << 20; }
  // No Full extent more: the bytes held, in whole Full pages.
  void wall() { spill.storage.budgetPages = static_cast<uint32_t>(spill.storage.allocatedBytes() / kPages.fullBytes); }
};

// Flash-Next's proportions (FlashNextLanes) under the governor's 40 GiB limit, with a prefill's history staging of
// 36,864 B a page: a request spills where placement says, its pages written through and kept Full until a claim
// releases them.
struct FlashNextGoverned {
  // The room a Spill start finds: its first command's extent and state, not its Resident KV's extents.
  static constexpr uint64_t kStartRoomBytes = 300ULL << 20;
  explicit FlashNextGoverned(const std::function<void(Executor &)> &shape = {},
                             const Spilling::Pages &pages = FlashNextLanes::kPages)
      : spill{16'384, false, 0, 200'000, Spilling::Extent{256}, shape, {}, true, pages} {
    spill.model.spilledReadFloorBytes = 390ULL << 20;
    spill.model.stagingPageBytes = 36'864;
  }
  // Starts `request` Spill (Engine::placement): the room it finds holds its first command but not its Resident KV,
  // and the limit leaves the engine ample room again once it has its lane.
  void startSpill(EngineRequest request) {
    const uint64_t id = request.id;
    spill.host.leaveRoom(kStartRoomBytes);
    spill.engine.submit(std::move(request));
    spill.runUntil([&] { return std::ranges::count(spill.events.startIds, id) != 0; }, "the start did not start");
    spill.host.leaveRoom(std::nullopt);
    require(kvMode(spill, id) == KvMode::Spill, "the start was not placed Spill");
  }
  // The memory wall: the engine holds what it does now and no more (GovernedHost::leaveRoom).
  void wall() { spill.host.leaveRoom(0); }
  // Ticks until the lane has decoded `tokens` tokens, between commands.
  void decode(uint64_t id, size_t tokens) {
    spill.runUntil([&] { return spill.events.outputs[id].size() >= tokens && !spill.engine.commandInFlight(); },
                   "the lane did not decode");
  }

  Spilling spill;
};

// A lone 64K Resident lane at the engine's limit converts itself, its pages freeing more than the hot
// block floor, and decodes its whole output.
void testALongDecodingLaneConvertsItselfAtTheWall() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  // 2,047 pages: one page's room left in its eighth extent.
  spill.engine.submit(Spilling::prompt(1, 1, 2'046, 200));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the lane did not decode");
  require(kvMode(spill, 1) == KvMode::Resident, "the lane did not start Resident");
  lanes.wall();
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(spill.events.completedCount == 1 && spill.events.failures.empty() &&
              spill.events.usage.at(1).second == 200,
          "the lane did not decode its whole output");
  require(counts.converted == 1 && spill.cache.spillSnapshot().spilledPages > 0,
          "the lane did not convert itself at the wall");
}

// A lone 64K prompt that starts Resident and meets the engine's limit at 56K tokens, its pages there
// freeing more than the hot block floor and its growth, converts itself between its prefill commands,
// prefills on as a Spill lane and decodes the output it decodes without the limit.
void testAPrefillingLaneConvertsItselfAtTheWall() {
  struct Run final {
    uint64_t converted = 0, spilledPages = 0;
    std::vector<std::string> failures;
    std::vector<uint32_t> output;
  };
  const auto run = [](bool walled) {
    FlashNextGoverned lanes;
    Spilling &spill = lanes.spill;
    spill.engine.submit(Spilling::prompt(1, 1, 2'046, 200));
    // 1,792 pages: seven whole extents.
    spill.runUntil(
        [&] {
          return spill.model.prefillChunks[1] != 0 && spill.cache.pageTable(1).pages.size() >= 1'792 &&
                 !spill.engine.commandInFlight();
        },
        "the prompt did not prefill 56K tokens");
    require(kvMode(spill, 1) == KvMode::Resident, "the prompt did not start Resident");
    if (walled) {
      lanes.wall();
      spill.runUntil([&] { return kvMode(spill, 1) == KvMode::Spill; }, "the prompt did not convert itself");
      require(!spill.events.outputs.contains(1), "the prompt converted only once it decoded");
    }
    spill.runUntilIdle();
    return Run{.converted = spill.engine.snapshot().spill.converted,
               .spilledPages = spill.cache.spillSnapshot().spilledPages,
               .failures = spill.events.failures,
               .output = spill.events.outputs[1]};
  };
  const Run roomy = run(false), walled = run(true);
  require(roomy.converted == 0 && roomy.failures.empty() && roomy.output.size() == 200,
          "the prompt without the limit did not decode its output Resident");
  require(walled.converted == 1 && walled.spilledPages > 0 && walled.failures.empty(),
          "a prefilling lane at the limit did not convert itself and complete");
  require(walled.output == roomy.output, "the converted lane decoded another output");
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
  require(kvMode(spill, 1) == KvMode::Resident && kvMode(spill, 2) == KvMode::Resident,
          "the lanes did not start Resident");
  lanes.wall();
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  spill.runUntil([&] { return kvMode(spill, 1) == KvMode::Spill; }, "the long lane did not convert itself");
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
    require(kvMode(spill, 1) == KvMode::Resident &&
                spill.cache.pageTable(1).pages.front() == spill.cache.pageTable(2).pages.front(),
            "the lanes did not share their prefix's pages, the long lane Resident");
    lanes.wall();
    const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
    const bool converts = shared == 125;
    require(held.granted() && (kvMode(spill, 1) == KvMode::Spill) == converts &&
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

// A lane of 32K tokens at most, whose pages free less than the hot block floor, does not convert,
// decoding or prefilling; alone at the wall it fails as on a target that cannot spill, naming its
// lane Resident in its phase and not the SSD cache, which holds none of its KV.
void testAShortLaneDoesNotConvert() {
  for (const bool prefilling : {true, false}) {
    FlashNextLanes lanes;
    Spilling &spill = lanes.spill;
    if (prefilling) {
      // A 64K prompt that starts Resident, as while the SSD cache takes no
      // writes, and whose prefill meets the wall before 32K tokens.
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
            std::string("a short ") + (prefilling ? "prefilling" : "decoding") + " lane converted to Spill");
    require(spill.events.failures == std::vector<std::string>{"capacity_exhausted"},
            "the lone lane did not fail at the wall");
    const std::string &message = spill.events.failureDetails.front().first;
    require(message.find(", a Resident lane of ") != std::string::npos &&
                message.find(prefilling ? " tokens in prefill, " : " tokens in decode, ") != std::string::npos &&
                message.find("SSD cache") == std::string::npos,
            "the lone lane's failure did not name its mode and phase alone: " + message);
  }
}

// A lone Resident prefill at the engine's limit, Metal memory outside the engine's buffers taking
// the room its KV would grow into, fails naming its lane's mode, phase and tokens and those bytes,
// and not the SSD cache, which holds none of its KV.
void testALoneResidentWallNamesItsModeAndPhase() {
  const test::GovernorLimits limits;
  Spilling spill{16'384, false, 0, 200'000, Spilling::Extent{256}, {}, limits, true, FlashNextLanes::kPages};
  // A 64K prompt that starts Resident, as while the SSD cache takes no writes.
  spill.tier.writableFile = false;
  spill.engine.submit(Spilling::prompt(1, 1, 2'046, 200));
  spill.runUntil([&] { return spill.model.prefillChunks[1] >= 1; }, "the prompt did not start");
  spill.tier.writableFile = true;
  // Memory outside the engine's buffers takes the room the limit leaves.
  const uint64_t untracked = limits.limitBytes - std::min(limits.limitBytes, spill.host.memory.total());
  spill.host.memory.holdUntracked(untracked);
  spill.runUntilIdle();
  require(spill.events.failures == std::vector<std::string>{"capacity_exhausted"},
          "the lone prefill did not fail at the limit");
  const std::string &message = spill.events.failureDetails.front().first;
  require(message.find("; the request, a Resident lane of 65473 tokens in prefill, ") != std::string::npos &&
              message.find(" beside device_untracked_bytes=" + std::to_string(untracked) + " of Metal memory") !=
                  std::string::npos &&
              message.find("SSD cache") == std::string::npos,
          "the lone prefill's failure did not name its lane and the memory outside the engine's buffers alone: " +
              message);
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
  require(kvMode(spill, 1) == KvMode::Resident && kvMode(spill, 2) == KvMode::Resident,
          "the lanes did not start Resident");
  // At the wall: no free Full page, and no extent more.
  lanes.wall();
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  static_cast<void>(spill.engine.tick(spill.now++));
  require(held.granted() && kvMode(spill, 1) == KvMode::Resident, "a lane's growth converted another lane");
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

// The pooled buffers the next lane starts from (Executor::pooledLaneBytes), `bytes` of them, which
// only KV in service takes: a lane's start takes none of them.
void poolTheNextLanesBuffers(Spilling &spill, uint64_t bytes) {
  spill.host.memory.hold("states", bytes);
  spill.model.heldStateBytes += bytes;
  spill.model.pooledLaneBytes += bytes;
}

// Memory beside a Flash-Next prompt of 2,046 pages and its 828 MiB of Resident KV, on a run whose SSD
// cache holds `slots` pages: the engine's limit leaves `room` past what memory holds, `pooled` bytes
// of it held as the next lane's pooled buffers.
struct ShortOfBoth {
  explicit ShortOfBoth(uint64_t room, uint64_t pooled = 0, uint32_t slots = 1'000)
      : spill{slots, false, 0, 200'000, Spilling::Extent{256}, {}, {}, true, FlashNextLanes::kPages} {
    if (pooled)
      poolTheNextLanesBuffers(spill, pooled);
    const uint64_t limit = test::GovernorLimits{}.limitBytes;
    spill.host.memory.hold("limit", limit - std::min(limit, spill.host.memory.total() + room));
  }
  Spilling spill;
};

// The ways the SSD cache cannot take a start: its room short of the projection, as ShortOfBoth's 1,000
// slots are of a 64K prompt's 2,047 pages; the tier closed; or the daily write budget spent, which a
// write of another writer's (a state in use) overdrew until it refills in 6 hours.
enum class Unspillable : uint8_t { Quota, Closed, WriteBudget };
constexpr std::array kUnspillable{Unspillable::Quota, Unspillable::Closed, Unspillable::WriteBudget};
void cannotSpill(Spilling &spill, Unspillable way) {
  if (way == Unspillable::Closed)
    spill.tier.writableFile = false;
  if (way == Unspillable::WriteBudget) {
    spill.budget->setWriteBudget(2 * kHostPageBytes);
    test::spendWriteBudget(spill.budget);
  }
}
std::string unspillable(Unspillable way) {
  return way == Unspillable::Quota    ? " for a short quota"
         : way == Unspillable::Closed ? " while the tier took no writes"
                                      : " while the write budget was spent";
}
std::string failureDetail(const Events &events) {
  return events.failureDetails.empty() ? std::string() : ": " + events.failureDetails.front().first;
}

// What a 64K start refused at its start for want of memory and of the SSD cache names of memory, as
// capacity_exhausted does: its Resident lane of 65,473 tokens at its start, the Metal memory outside the
// engine's buffers, and the bytes its least start needs, its prompt and the rows of the decode step after
// it in eight Full extents with its state, less than a page, beside fewer that memory has for it.
bool namesTheMemoryShortfall(const std::string &message) {
  const uint64_t extents = 8 * 256 * FlashNextLanes::kPages.fullBytes;
  const std::string lane = "; the request, a Resident lane of 65473 tokens at its start, does not fit in the "
                           "memory this server may use beside device_untracked_bytes=";
  const std::string needs = ": it needs memory_needed_bytes=", has = ", where memory_available_bytes=";
  const size_t at = message.find(needs), there = message.find(has);
  if (message.find(lane) == std::string::npos || at == std::string::npos || there == std::string::npos)
    return false;
  const uint64_t needed = std::stoull(message.substr(at + needs.size()));
  return needed >= extents && needed < extents + FlashNextLanes::kPages.fullBytes &&
         std::stoull(message.substr(there + has.size())) < needed;
}

// A start alone that cannot spill and whose prompt memory does not hold now is refused at its start,
// naming what memory and the SSD cache have for it, rather than start Resident and fail partway through
// its prefill: kv_disk_quota for a short quota, capacity_exhausted for a closed tier, kv_write_budget
// with the 6 hours until the budget covers what spilling requests were promised for a spent one.
void testALoneStartMemoryCannotHoldIsRefusedAtItsStart() {
  for (const Unspillable way : kUnspillable) {
    ShortOfBoth both(400ULL << 20);
    Spilling &spill = both.spill;
    cannotSpill(spill, way);
    spill.engine.submit(Spilling::prompt(1, 1, 2'046, 200));
    spill.runUntilIdle();
    const std::string message = spill.events.failureDetails.empty() ? "" : spill.events.failureDetails.front().first;
    const std::string code = way == Unspillable::Quota    ? "kv_disk_quota"
                             : way == Unspillable::Closed ? "capacity_exhausted"
                                                          : "kv_write_budget";
    require(spill.events.failures == std::vector<std::string>{code} && spill.model.prefillChunks[1] == 0 &&
                spill.events.retryAfters == std::vector<uint32_t>{way == Unspillable::WriteBudget ? 21'600U : 0U} &&
                namesTheMemoryShortfall(message),
            "a start alone that neither memory nor the SSD cache holds was not refused at its start naming both" +
                unspillable(way) + failureDetail(spill.events));
  }
}

// A start alone that cannot spill, whose prompt memory holds but not its projection, starts Resident
// under the bound of the KV memory holds, as the quota bounds a Spill start's output, and ends at it with
// Length: memory holds eight Full extents, 2,048 pages, of the 2,056 the 64K prompt and its 300 tokens of
// output project, and so 56 tokens of output with the 7 rows past them its last decode step verifies.
// The parent refused it at its start.
void testALoneStartTakesMemorysBound() {
  for (const Unspillable way : kUnspillable) {
    ShortOfBoth both(900ULL << 20);
    Spilling &spill = both.spill;
    cannotSpill(spill, way);
    spill.engine.submit(Spilling::prompt(1, 1, 2'046, 300));
    spill.runUntilIdle();
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    require(spill.events.completedCount == 1 && spill.events.failures.empty() && counts.resident == 1 &&
                counts.spill == 0 && counts.bounded == 1 && counts.lengthEnds == 1 &&
                spill.model.outputLimits.at(1) == 56 && spill.events.usage.at(1).second == 56 &&
                spill.events.boundedIds == std::vector<uint64_t>{1},
            "a start alone whose prompt memory holds did not start Resident and end at memory's bound" +
                unspillable(way) + failureDetail(spill.events));
  }
}

// That bound holds through a suspension: a Resident resume meets no admission that sets it again, and
// without it the lane would run into the wall past what it has streamed and fail there.
void testAResumeKeepsMemorysBound() {
  ShortOfBoth both(900ULL << 20);
  Spilling &spill = both.spill;
  cannotSpill(spill, Unspillable::Closed);
  EngineRequest background = Spilling::prompt(1, 1, 2'046, 300);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 10; }, "the request did not decode");
  require(spill.model.outputLimits.at(1) == 56, "the request did not start under memory's bound");
  // A Normal start is refused memory while it is resident, which suspends it.
  spill.model.beginGrowthBlocked = [&] {
    return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
  };
  spill.model.resumeDenied = true;
  spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
  spill.runUntil([&] { return spill.model.suspensions == 1 && spill.events.completedCount == 1; },
                 "the Normal request did not run beside the suspended one");
  spill.model.resumeDenied = false;
  spill.runUntilIdle();
  require(spill.model.resumptions == 1 && spill.events.completedCount == 2 && spill.events.failures.empty() &&
              spill.model.outputLimits.at(1) == 56 && spill.events.usage.at(1).second == 56 &&
              spill.events.boundedIds == std::vector<uint64_t>{1},
          "a resume lost the bound memory set at its start: limit " +
              std::to_string(spill.model.outputLimits.at(1)) + failureDetail(spill.events));
}

// A guard: the same start beside the next lane's pooled buffers, which cover what memory lacks,
// starts Resident and decodes its output. The claim it makes in service takes them, as its growth
// does; a start's would keep them, and it would not start.
void testALoneStartCountsTheNextLanesBuffers() {
  ShortOfBoth both(400ULL << 20, 768ULL << 20);
  Spilling &spill = both.spill;
  spill.engine.submit(Spilling::prompt(1, 1, 2'046, 200));
  spill.runUntilIdle();
  require(spill.events.completedCount == 1 && spill.events.failures.empty() &&
              spill.events.usage.at(1).second == 200 && spill.model.pooledLaneBytes == 0,
          "a start alone that the next lane's buffers hold did not take them and decode: " +
              (spill.events.failureDetails.empty() ? std::string() : spill.events.failureDetails.front().first));
}

// A start alone that cannot spill and whose Resident KV memory does not hold now, but whose SSD
// cache holds its prompt and some of its output, takes the quota's bound on its output and starts
// Spill, as a Spill start short of its projection does: 2,050 slots hold the 64K prompt's 2,047 pages
// and 127 tokens of its 300.
void testALoneStartMemoryCannotHoldTakesTheQuotasBound() {
  ShortOfBoth both(400ULL << 20, 0, 2'050);
  Spilling &spill = both.spill;
  spill.engine.submit(Spilling::prompt(1, 1, 2'046, 300));
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(spill.events.completedCount == 1 && spill.events.failures.empty() && counts.spill == 1 &&
              counts.resident == 0 && spill.model.outputLimits.at(1) == 127 &&
              spill.events.usage.at(1).second == 127 && spill.events.boundedIds == std::vector<uint64_t>{1},
          "a start alone that memory does not hold did not start Spill under the quota's bound: " +
              (spill.events.failureDetails.empty() ? std::string() : spill.events.failureDetails.front().first));
}

// A start alone within the context the memory plan advertises starts. At the resident bound
// (ContextBounds::resident: 65,529 tokens, eight of Flash-Next's Full extents), in the room the plan
// leaves those extents past its lane's buffers, which a warmup left pooled, and its Index runway, and in
// that room and up to an extent less a byte more, a prompt with 32 tokens of output that fills it and
// cannot spill (its quota short, its tier closed or its write budget spent) starts Resident and decodes
// its output: its claim counts what the plan counts, its KV in whole extents and the lane buffers the
// pool lacks, and no state its prefill captures, which takes buffers only where memory has them.
// Charged a captured state's 118.75 MB beside its KV, a lone start on Qwen3.8-Flash-Next was refused an
// extent of the context the plan advertised.
void testALoneStartWithinTheResidentContextStarts() {
  constexpr uint64_t extent = 256 * FlashNextLanes::kPages.fullBytes, lane = 276'496'384;
  for (const uint64_t slack : {uint64_t{0}, extent / 2, extent - 1})
    for (const Unspillable way : kUnspillable) {
      Spilling spill{1'000, false, 0, 65'529, Spilling::Extent{256}, {}, {}, true, FlashNextLanes::kPages};
      spill.model.laneFootprintBytes = lane;
      spill.host.memory.hold("states", lane);
      spill.model.heldStateBytes = lane;
      spill.model.pooledLaneBytes = lane;
      const uint64_t limit = test::GovernorLimits{}.limitBytes;
      spill.host.memory.hold("limit", limit - std::min(limit, spill.host.memory.total() + 8 * extent + slack));
      cannotSpill(spill, way);
      EngineRequest alone = request(1, std::vector<uint32_t>(65'497, 1));
      alone.maxNewTokens = 32;
      spill.engine.submit(std::move(alone));
      spill.runUntilIdle();
      const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
      require(spill.events.completedCount == 1 && spill.events.failures.empty() && counts.resident == 1 &&
                  counts.spill == 0 && spill.events.usage.at(1).second == 32 && spill.events.boundedIds.empty(),
              "a start alone at the resident context in room for its extents and " + std::to_string(slack) +
                  " B did not start Resident and decode its output" + unspillable(way) + failureDetail(spill.events));
    }
}

// Beside a lane in service, a start that cannot spill, its quota short, its tier closed or its write
// budget spent, and whose Resident KV that lane keeps memory from holding waits for it to end, then starts
// and decodes: it never prefills beside it.
void testAStartMemoryCannotHoldBesideALaneWaitsForIt() {
  for (const Unspillable way : kUnspillable) {
    ShortOfBoth both(1'200ULL << 20);
    Spilling &spill = both.spill;
    cannotSpill(spill, way);
    spill.engine.submit(Spilling::prompt(1, 1, 1'000, 100));
    spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first lane did not decode");
    spill.engine.submit(Spilling::prompt(2, 2, 2'046, 50));
    bool beside = false;
    spill.runUntil(
        [&] {
          beside = beside || (spill.events.completedCount == 0 && spill.model.prefillChunks[2] > 0);
          return idle(spill.engine);
        },
        "the requests did not end");
    require(!beside && spill.events.completedCount == 2 && spill.events.failures.empty() &&
                spill.events.usage.at(2).second == 50,
            "a start memory could not hold beside a lane prefilled beside it, or did not decode after it" +
                unspillable(way) + failureDetail(spill.events));
  }
}

// Beside a lane in service, a start the closed tier refuses whose prompt memory does not hold even
// alone waits for that lane and is decided once it ends: refused at its start, naming what memory has
// for it, never prefilled.
void testAStartMemoryNeverHoldsIsDecidedOnceTheLaneEnds() {
  ShortOfBoth both(800ULL << 20);
  Spilling &spill = both.spill;
  spill.tier.writableFile = false;
  spill.engine.submit(Spilling::prompt(1, 1, 1'000, 100));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first lane did not decode");
  spill.engine.submit(Spilling::prompt(2, 2, 2'046, 50));
  spill.runUntil([&] { return spill.events.failedCount != 0; }, "the start was not refused");
  require(spill.events.completedCount == 1 && spill.events.usage.at(1).second == 100 &&
              spill.events.failures == std::vector<std::string>{"capacity_exhausted"} &&
              spill.model.prefillChunks[2] == 0 &&
              namesTheMemoryShortfall(spill.events.failureDetails.front().first),
          "a start memory does not hold even alone was not refused at its start once the lane beside it ended" +
              failureDetail(spill.events));
  spill.runUntilIdle();
}

// Beside a lane of a lower priority, the same start makes that lane yield, as a start refused memory
// does, and prefills only once it has; it decodes, and the lane decodes its output once it resumes.
void testAStartMemoryCannotHoldBesideALowerLaneMakesItYield() {
  ShortOfBoth both(1'200ULL << 20);
  Spilling &spill = both.spill;
  EngineRequest background = Spilling::prompt(1, 1, 1'000, 100);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the Background lane did not decode");
  EngineRequest foreground = Spilling::prompt(2, 2, 2'046, 50);
  foreground.priority = RequestPriority::Foreground;
  spill.engine.submit(std::move(foreground));
  bool beside = false;
  spill.runUntil(
      [&] {
        beside = beside || (spill.engine.snapshot().prioritySuspensions == 0 && spill.model.prefillChunks[2] > 0);
        return idle(spill.engine);
      },
      "the requests did not end");
  require(!beside && spill.engine.snapshot().prioritySuspensions == 1 && spill.events.completedCount == 2 &&
              spill.events.failures.empty() && spill.events.usage.at(2).second == 50 &&
              spill.events.usage.at(1).second == 100,
          "a start memory could not hold prefilled beside a lower priority's lane before that lane yielded to "
          "it, or the two did not decode");
}

// Beside a Foreground lane whose KV and projected growth keep memory from holding it, a Foreground start
// that cannot spill waits for that lane and suspends none of the Background lanes beside it, which would
// bring it no closer to running: they keep their lanes and KV. Once that lane ends it starts, and all four
// decode. The parent suspended both Background lanes while the Foreground lane decoded.
void testAStartWaitingForALaneOfItsPriorityLeavesLowerLanesRunning() {
  ShortOfBoth both(1'200ULL << 20);
  Spilling &spill = both.spill;
  const auto submit = [&](uint64_t id, uint32_t pages, uint32_t output, RequestPriority priority) {
    EngineRequest request = Spilling::prompt(id, static_cast<uint32_t>(id), pages, output);
    request.priority = priority;
    spill.engine.submit(std::move(request));
  };
  submit(2, 100, 300, RequestPriority::Background);
  submit(3, 100, 300, RequestPriority::Background);
  spill.runUntil([&] { return spill.events.outputs.contains(2) && spill.events.outputs.contains(3); },
                 "the Background lanes did not decode");
  submit(1, 1'000, 100, RequestPriority::Foreground);
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the Foreground lane did not decode");
  submit(4, 2'046, 50, RequestPriority::Foreground);
  // While the Foreground lane is in service: the lanes suspended, and the start's prefill chunks.
  uint64_t suspended = 0;
  uint32_t chunks = 0;
  spill.runUntil(
      [&] {
        if (!spill.events.usage.contains(1)) {
          const EngineSnapshot snapshot = spill.engine.snapshot();
          suspended = snapshot.prioritySuspensions + snapshot.resourceSuspensions;
          chunks = spill.model.prefillChunks[4];
        }
        return idle(spill.engine);
      },
      "the requests did not end");
  require(suspended == 0 && chunks == 0 && spill.events.completedCount == 4 && spill.events.failures.empty() &&
              spill.events.usage.at(4).second == 50 && spill.events.usage.at(2).second == 300 &&
              spill.events.usage.at(3).second == 300,
          "beside a lane of its priority, a start that waits for it suspended " + std::to_string(suspended) +
              " lanes and ran " + std::to_string(chunks) + " prefill chunks, or the requests did not decode" +
              failureDetail(spill.events));
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

// A start refused memory keeps the probe it was placed with while it backs off, and the host's
// pass at critical pressure may take the chain that probe found: placed again in the tick after
// that control pass, it counts the chain only as far as the cache still has it, and waits on. Its
// 16 pages Resident take more than Spill's floor, so at the memory wall placement counts what a
// start may take beside the chain it restores (Engine::residentRoom), whose walk follows the chain
// up from the state the probe found: with that state's block taken, it threw "unknown KV cache
// block" out of the tick.
void testAPlacementCountsOnlyTheChainAStartKeeps() {
  Spilling spill(64, false, 0, 64 * KvCache::pageTokens);
  const auto submit = [&](uint64_t id, std::vector<uint32_t> tokens, uint32_t output) {
    EngineRequest value = request(id, std::move(tokens));
    value.maxNewTokens = output;
    value.deadlineMilliseconds = 1e9;
    spill.engine.submit(std::move(value));
  };
  const std::vector<uint32_t> turn(16 * KvCache::pageTokens + 1, 50);
  // An earlier turn ends; its chain stays cached.
  submit(5, turn, 1);
  spill.runUntil([&] { return spill.events.usage.contains(5); }, "the earlier turn did not end");
  require(spill.cache.probe(turn, {}).cachedTokens() != 0, "the earlier turn's chain was not cached");
  // The earlier turn sent again: the host refuses its start memory.
  spill.model.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  spill.model.beginGrowthBlocked = [&] { return spill.model.lastBeginId == 3; };
  submit(3, turn, 1);
  spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).memory == 1; },
                 "the resend was not refused memory");
  // The control pass at critical pressure brings the probe up to date (Engine::collectLandings),
  // then the host's pass takes the cache: the chain's state, which this model's states write to
  // no SSD cache, and its blocks, of which no copy was written, leaf first.
  spill.host.setPressure(MemoryPressure::Critical);
  spill.engine.collectLandings(spill.now);
  static_cast<void>(spill.host.governor.controlPass(spill.now));
  require(!spill.cache.lookup(turn, {}).kvBoundary, "the host's pass did not take the chain");
  spill.host.leaveRoom(0);
  for (int step = 0; step < 5; ++step)
    static_cast<void>(spill.engine.tick(spill.now++));
  require(spill.engine.resourceWaitSnapshot(spill.now).memory == 1 && spill.events.failedCount == 0,
          "the start did not wait on beside a chain the cache took");
  // Memory back, it starts without the chain and completes.
  spill.host.leaveRoom(std::nullopt);
  spill.host.setPressure(MemoryPressure::Normal);
  spill.model.beginGrowthBlocked = {};
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.failedCount == 0,
          "the start did not complete once memory came back");
}

// A start alone that cannot spill, the tier closed or the quota short, and whose projection fits
// the resident context starts Resident where memory holds it now, the next lane's pooled buffers
// among it, and is refused at its start where it does not.
void testAStartThatCannotSpillStartsResidentWhereMemoryHoldsIt() {
  for (const bool closed : {true, false})
    for (const bool buffers : {true, false}) {
      Spilling spill(8, false, 0, 50 * KvCache::pageTokens, {4}, {}, {.limitBytes = Spilling::kGovernedLimitBytes},
                     true);
      spill.tier.writableFile = !closed;
      // The room holds 20 pages, so placement spills; the resident context is 50, and the next lane's
      // pooled buffers, where it has them, hold 60 more.
      if (buffers)
        poolTheNextLanesBuffers(spill, 60 * 100);
      spill.limit(spill.held() + 20 * 100);
      // 40 pages of prompt and 64 tokens of output project 43 pages; the 8
      // slots of the quota cannot hold them.
      spill.engine.submit(Spilling::prompt(1, 1, 40, 64));
      spill.runUntilIdle();
      const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
      const std::string when = std::string(closed ? " while the tier took no writes" : " for a short quota");
      if (buffers)
        require(spill.events.completedCount == 1 && spill.events.usage.at(1).second == 64 && counts.resident == 1 &&
                    counts.spill == 0 && counts.refusedClosed == 0 && counts.refusedQuota == 0 &&
                    spill.model.pooledLaneBytes == 0,
                "a start the next lane's buffers hold was refused, or did not take them," + when);
      else
        require(spill.events.failures ==
                        std::vector<std::string>{closed ? "capacity_exhausted" : "kv_disk_quota"} &&
                    (closed ? counts.refusedClosed : counts.refusedQuota) == 1 && spill.model.prefillChunks[1] == 0,
                "a start memory did not hold started" + when);
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

// A governed Spill lane in Flash-Next's proportions that decodes with its kPages pages past the dense history and
// the window, three extents of them, not yet copied: their copies land once a test allows spill writes.
struct LandingLane {
  static constexpr uint32_t kPages = 3 * 256;
  LandingLane() {
    spill.allowSpillWrites(false);
    spill.engine.submit(Spilling::prompt(1, 1, kPages + 2, 4'096));
    spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the lane did not decode");
  }
  [[nodiscard]] uint64_t fullBytes() const {
    const KvPoolSnapshot pool = spill.pool.snapshot();
    return pool.allocatedBytes - pool.index.allocatedBytes;
  }
  Spilling spill{4'096, true, 100'000, 0, Spilling::Extent{256}, {}, {}, true, FlashNextLanes::kPages};
};

// Under critical pressure, which refuses every allocation the host's rule judges, a Spill lane's landed pages
// still move past the Index runway, and so they do with Metal memory outside the engine's buffers holding the charge
// over the engine's limit: the Full extents each runway's moves empty leave and pay for the next Index extent past
// both (MemoryGovernor::fund), and the runway stands again once the moves end.
void testLandedPagesSpillAtCriticalPressure() {
  for (const bool overLimit : {false, true}) {
    LandingLane lane;
    Spilling &spill = lane.spill;
    const uint64_t full = lane.fullBytes();
    // Over the limit by all the lane's Full extents hold, the charge stays
    // over it once the moves have released three of them.
    if (overLimit) {
      const uint64_t limit = test::GovernorLimits{}.limitBytes;
      spill.host.memory.holdUntracked(limit - std::min(limit, spill.host.memory.total()) + full);
    }
    spill.host.setPressure(MemoryPressure::Critical);
    spill.allowSpillWrites(true);
    const std::string where = overLimit ? ", the charge over the engine's limit" : "";
    const std::string stayed =
        "under critical pressure the landed pages stayed Full once the Index runway was used" + where;
    spill.runUntil([&] { return spill.cache.spillSnapshot().spilledPages >= LandingLane::kPages; }, stayed.c_str());
    require(lane.fullBytes() + LandingLane::kPages * FlashNextLanes::kPages.fullBytes <= full &&
                spill.pool.freePageCount(kv::PageForm::Index) >= spill.pool.extentPages() &&
                spill.events.failures.empty() && spill.engine.snapshot().resourceSuspensions == 0,
            "the moves kept the Full extents they emptied or left the Index runway short, or the lane failed or "
            "yielded" + where);
  }
}

// A refill the driver denies keeps what the moves released: the allocation's next attempt, once the ledger has freed
// what costs nothing, pays for it with those bytes under critical pressure, and the moves go on past the runway.
void testADriverDeniedRefillKeepsWhatTheMovesReleased() {
  LandingLane lane;
  Spilling &spill = lane.spill;
  // The next lane's pooled buffers, which free() takes at once under
  // critical pressure: the denied refill's allocation tries again.
  poolTheNextLanesBuffers(spill, 64ULL << 20);
  // The driver denies the first refill of an Index runway the moves used up.
  bool denied = false;
  spill.storage.driverDenies = [&](kv::PageForm form) {
    return form == kv::PageForm::Index && !spill.pool.freePageCount(kv::PageForm::Index) &&
           !std::exchange(denied, true);
  };
  spill.host.setPressure(MemoryPressure::Critical);
  // Every copy lands at once, so the first moves use the whole runway.
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = 1024;
  spill.runUntil([&] { return spill.cache.spillSnapshot().spilledPages >= LandingLane::kPages; },
                 "a refill the driver denied lost what the moves had released: the landed pages stayed Full");
  require(denied && spill.events.failures.empty() && spill.engine.snapshot().resourceSuspensions == 0,
          "the driver denied no refill, or the lane failed or yielded");
}

// Under critical pressure a runway whose moves empty no Full extent, a write in flight holding a page in each, pays
// for no refill: the moves wait, and once those writes land and the extents can empty, every landed page moves, with
// no failure or yield.
void testMovesThatEmptyNoFullExtentWaitAtCriticalPressure() {
  LandingLane lane;
  Spilling &spill = lane.spill;
  // Past the dense history, the first page of the lane in each extent.
  std::set<uint32_t> held, extents;
  const std::span<const uint32_t> pages = spill.cache.pageTable(1).pages;
  for (size_t index = 1; index + 1 < pages.size(); ++index)
    if (extents.insert(pages[index] / spill.pool.extentPages()).second)
      held.insert(pages[index]);
  // Ticks on, landing every transfer but the writes of the held pages.
  const auto runUntil = [&](const std::function<bool()> &done, const char *message) {
    for (uint32_t step = 0; step < 2000 && !done(); ++step) {
      static_cast<void>(spill.engine.tick(spill.now++));
      size_t write = 0;
      for (const auto &transfer : spill.tier.transfers) {
        const bool heldWrite = transfer->write && held.contains(spill.tier.writes[write++].page);
        if (!transfer->finished && transfer->submitted && !heldWrite)
          transfer->ready = true;
      }
    }
    require(done(), message);
  };
  spill.host.setPressure(MemoryPressure::Critical);
  // Every write starts at once, the held pages' among them.
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = 1024;
  runUntil([&] { return spill.cache.spillSnapshot().spilledPages >= spill.pool.extentPages(); },
           "the landed pages did not take the Index runway");
  // A hundred ticks on, the moves still wait.
  const double waited = spill.now + 100;
  runUntil([&] { return spill.now >= waited; }, "the engine stopped ticking");
  require(spill.cache.spillSnapshot().spilledPages == spill.pool.extentPages() && extents.size() == 4,
          "the moves went past the Index runway with no Full extent emptied to pay for it: " +
              std::to_string(spill.cache.spillSnapshot().spilledPages) + " spilled, extents " +
              std::to_string(extents.size()));
  held.clear();
  runUntil([&] { return spill.cache.spillSnapshot().spilledPages >= LandingLane::kPages; },
           "once the writes in flight landed, the landed pages stayed Full under critical pressure");
  require(spill.events.failures.empty() && spill.engine.snapshot().resourceSuspensions == 0,
          "the lane failed or yielded while its moves waited");
}

// Under critical pressure a prefilling Spill prompt's moves keep no empty Full extent for it to grow into: only the
// extents they release pay for the next Index extent (Engine::landedExtents), so its landed pages move past the
// Index runway. Once memory refuses its KV the prompt yields, and it resumes and completes once the pressure lifts.
void testASpillingPromptKeepsNoFullExtentAtCriticalPressure() {
  Spilling spill{4'096, true, 100'000, 0, Spilling::Extent{256}, {}, {}, true, FlashNextLanes::kPages};
  const uint64_t extent = uint64_t{spill.pool.extentPages()} * FlashNextLanes::kPages.fullBytes;
  const auto emptyFullExtents = [&] {
    const KvPoolSnapshot pool = spill.pool.snapshot();
    return (pool.reclaimableBytes - pool.index.reclaimableBytes) / extent;
  };
  // A 64K prompt prefills three extents with no copies.
  spill.allowSpillWrites(false);
  spill.engine.submit(Spilling::prompt(1, 1, 2'046, 8));
  spill.runUntil(
      [&] { return spill.model.prefillChunks[1] && spill.cache.pageTable(1).pages.size() >= LandingLane::kPages; },
      "the prompt did not prefill three extents");
  spill.host.setPressure(MemoryPressure::Critical);
  // Until it yields, the empty Full extents its moves leave.
  uint64_t kept = 0;
  const auto yielded = [&] {
    if (spill.engine.snapshot().resourceSuspensions)
      return true;
    kept = std::max(kept, emptyFullExtents());
    return false;
  };
  // Its copies land at once, and then no more: its KV outgrows its Full
  // pages, which critical pressure grows by none.
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = 1024;
  spill.runUntil([&] { return yielded() || spill.cache.spillSnapshot().spilledPages > spill.pool.extentPages(); },
                 "the prompt's landed pages did not take the Index runway");
  const uint32_t moved = spill.cache.spillSnapshot().spilledPages;
  spill.allowSpillWrites(false);
  spill.runUntil(yielded, "the prompt did not yield under critical pressure");
  spill.host.setPressure(MemoryPressure::Normal);
  spill.allowSpillWrites(true);
  spill.runUntilIdle();
  require(moved > spill.pool.extentPages() && kept == 0,
          "under critical pressure the prompt's landed pages stayed Full once the Index runway was used, or its "
          "moves kept a Full extent they emptied: " +
              std::to_string(moved) + " moved, " + std::to_string(kept) + " kept");
  require(spill.events.completedCount == 1 && spill.events.failures.empty(),
          "the prompt that yielded under critical pressure failed or did not complete");
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

// A request suspended before it has any output, whose prompt the quota can no longer hold even alone
// (the volume filled meanwhile), is refused as a start is. Ending at its length would answer with
// nothing, and a score that ends so is one the server takes for a broken engine.
void testASpillResumeWithNoOutputIsRefusedOutOfQuota() {
  for (const bool score : {false, true}) {
    Spilling spill(20, true, 100'000, 0, {4}, test::commandShape(KvCache::pageTokens));
    // No copies: the request holds its whole history without any.
    spill.allowSpillWrites(false);
    // Three pages of prompt, and each prefill command takes a page of them.
    EngineRequest background = request(1, std::vector<uint32_t>(3 * KvCache::pageTokens - 1, 1));
    background.maxNewTokens = score ? 0 : 100;
    if (score)
      background.scoreTokens = {1, 2};
    background.priority = RequestPriority::Background;
    spill.engine.submit(std::move(background));
    spill.runUntil([&] { return spill.model.prefillChunks[1] == 1; }, "the prompt did not start");
    // A Normal start is refused memory while it is resident, which suspends it.
    spill.model.beginGrowthBlocked = [&] {
      return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
    };
    spill.model.resumeDenied = true;
    spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
    spill.runUntil([&] { return spill.events.completedCount == 1; }, "the Normal request did not run");
    require(spill.model.suspensions == 1, "the request was not suspended in its prompt");
    spill.budget->setActiveLimitBytes(100);
    spill.model.resumeDenied = false;
    spill.runUntilIdle();
    require(spill.model.resumptions == 0 && spill.events.completedCount == 1 &&
                spill.events.failures == std::vector<std::string>{"kv_disk_quota"} &&
                spill.engine.snapshot().spill.lengthEnds == 0 && spill.committedSlots() == 0,
            std::string(score ? "a score" : "a prompt") +
                " that the quota could no longer hold at its resume was not refused: " +
                std::to_string(spill.engine.snapshot().spill.lengthEnds) + " length ends" +
                failureDetail(spill.events));
  }
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

// A Spill lane keeps the pages whose copies landed Full while memory lasts: it writes every page through, yet holds
// no Index page and reads nothing back, taking neither history staging nor a hot block floor.
void testASpillLaneKeepsLandedPagesWhileMemoryLasts() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  lanes.startSpill(Spilling::prompt(1, 1, 2'046, 200));
  uint32_t index = 0;
  bool reads = false;
  spill.runUntil(
      [&] {
        if (spill.events.usage.contains(1))
          return true;
        index = std::max(index, spill.cache.indexPages(1));
        reads = reads || spill.model.spilledReadFloors.contains(1) || spill.model.historyStaging.contains(1);
        return false;
      },
      "the lane did not end");
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 200 &&
              spill.cache.spillSnapshot().writeThroughPages >= 2'046,
          "the Spill lane did not write its pages through and decode its whole output");
  require(index == 0 && !reads, "a Spill lane with memory to spare moved " + std::to_string(index) +
                                    " pages to the index form, or took memory to read them back");
}

// A Spill lane in prefill at the memory wall releases at once only what it lacks: its oldest pages whose copies
// landed, in the whole extents that hold its growth's extent and its history's staging through its next command,
// 512 pages for 108.5 MB and 40.1 MB on Qwen3.8-Flash-Next, the staging held by the same command. No claim waits.
void testASpillLaneAtTheWallReleasesOnlyWhatItLacks() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  lanes.startSpill(Spilling::prompt(1, 1, 2'046, 8));
  spill.runUntil([&] { return spill.cache.pageTable(1).pages.size() >= 1'024 && !spill.engine.commandInFlight(); },
                 "the prompt did not reach its 1,024th page");
  lanes.wall();
  uint32_t released = 0;
  bool waited = false;
  spill.runUntil(
      [&] {
        waited = waited || spill.host.governor.snapshot().claims.waiting != 0;
        released = spill.cache.indexPages(1);
        return released && !spill.engine.commandInFlight();
      },
      "the lane at the wall released none of its pages");
  require(released == 512 && !waited && spill.model.historyStaging.contains(1),
          "the lane at the wall released " + std::to_string(released) + " pages, not the 512 of the two extents its "
              "growth and staging lack, waited, or read without its staging");
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 8, "the lane did not complete");
}

// A Spill lane prefilling at the memory wall in commands of 8,192 rows, a checkpoint inside each, releases its own
// landed pages for the KV past the checkpoint, as its next command would: wherever the wall falls, from its 1,024th or
// 1,280th page with up to two extents of room left, no command is cut there and the lane decodes the output of a run
// with room to spare. On Qwen3.8-Flash-Next each cut cost a pass more of the routed experts, which stream: 14 cuts in
// a 237K prompt, its first token 8% later.
void testASpillLaneAtTheWallCutsNoCommand() {
  constexpr uint64_t extentBytes = uint64_t{256} * FlashNextLanes::kPages.fullBytes;
  struct Run final {
    std::vector<uint32_t> output;
    std::vector<std::string> failures;
    uint64_t cuts = 0, released = 0;
  };
  // The wall from `wallPage` on with `room` left, until the prompt is prefilled.
  const auto run = [](uint32_t wallPage, std::optional<uint64_t> room) {
    FlashNextGoverned lanes(test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true));
    Spilling &spill = lanes.spill;
    lanes.startSpill(Spilling::prompt(1, 1, 2'046, 8));
    const auto prefilled = [&](uint32_t pages) {
      return spill.events.usage.contains(1) || !spill.events.failures.empty() ||
             (spill.cache.pageTable(1).pages.size() >= pages && !spill.engine.commandInFlight());
    };
    spill.runUntil([&] { return prefilled(wallPage); }, "the prompt did not reach the wall's page");
    Run result;
    if (room) {
      spill.host.leaveRoom(*room);
      const uint64_t cuts = spill.engine.snapshot().captureCuts;
      const uint64_t released = spill.cache.spillSnapshot().releasedPages;
      spill.runUntil([&] { return prefilled(2'046); }, "the prompt did not prefill at the wall");
      result.cuts = spill.engine.snapshot().captureCuts - cuts;
      result.released = spill.cache.spillSnapshot().releasedPages - released;
      spill.host.leaveRoom(std::nullopt);
    }
    spill.runUntilIdle();
    result.output = spill.events.outputs[1];
    result.failures = spill.events.failures;
    return result;
  };
  const Run roomy = run(1'024, std::nullopt);
  std::string failures;
  for (const uint32_t wallPage : {1'024u, 1'280u})
    for (uint64_t quarters = 0; quarters <= 8; ++quarters) {
      const Run walled = run(wallPage, quarters * extentBytes / 4);
      if (!walled.failures.empty() || walled.output != roomy.output || !walled.released || walled.cuts)
        failures += "; from page " + std::to_string(wallPage) + " with " + std::to_string(quarters) +
                    "/4 extent left: " + std::to_string(walled.failures.size()) + " failures, " +
                    std::to_string(walled.released) + " pages released, " + std::to_string(walled.cuts) + " cuts" +
                    (walled.output == roomy.output ? "" : ", another output");
    }
  require(failures.empty(),
          "a lane at the wall failed, released nothing, decoded another output or cut a command" + failures);
}

// A lane prefilling in commands of 6,144 rows meets the wall where its extent holds the next command's rows up to the
// checkpoint and no more, while it holds no Index page: the KV past the checkpoint makes the lane's first release, the
// command is planned again to stage the lane's history (Engine::prepare), and it runs whole, the lane decoding the
// output of a run with room to spare. Every command reads spilled K/V through memory reserved for it (TestExecutor).
void testTheKvPastACheckpointMakesALanesFirstRelease() {
  struct Run final {
    std::vector<uint32_t> output;
    std::vector<std::string> failures;
    uint64_t cuts = 0, replans = 0;
    bool released = false;
  };
  const auto run = [](bool walled) {
    FlashNextGoverned lanes(test::commandShape(3 * model::ExecutionLimits::prefillTokenBudget, 6, true));
    Spilling &spill = lanes.spill;
    lanes.startSpill(Spilling::prompt(1, 1, 2'046, 8));
    // Past 1,024 pages, the first boundary whose extent leaves 128 to 255 Full pages free: the next command's rows up
    // to its checkpoint fit, the KV past it needs an extent more.
    spill.runUntil(
        [&] {
          const uint32_t free = spill.pool.freePageCount(kv::PageForm::Full);
          return spill.cache.pageTable(1).pages.size() >= 1'024 && !spill.engine.commandInFlight() && free >= 128 &&
                 free < 256;
        },
        "no boundary left part of an extent free");
    require(!spill.cache.holdsIndexPages(1), "the lane released its pages before the wall");
    Run result;
    if (walled) {
      const engine::EngineSnapshot before = spill.engine.snapshot();
      lanes.wall();
      spill.runUntil(
          [&] {
            return !spill.events.failures.empty() ||
                   (spill.cache.pageTable(1).pages.size() >= 2'046 && !spill.engine.commandInFlight());
          },
          "the prompt did not prefill at the wall");
      const engine::EngineSnapshot after = spill.engine.snapshot();
      result.cuts = after.captureCuts - before.captureCuts;
      result.replans = after.replans - before.replans;
      result.released = spill.events.failures.empty() && spill.cache.holdsIndexPages(1);
      spill.host.leaveRoom(std::nullopt);
    }
    spill.runUntilIdle();
    result.output = spill.events.outputs[1];
    result.failures = spill.events.failures;
    return result;
  };
  const Run roomy = run(false), walled = run(true);
  require(walled.failures.empty() && walled.output == roomy.output && walled.released,
          "the lane at the wall failed, decoded another output or released nothing");
  require(walled.cuts == 0 && walled.replans > 0,
          "the KV past the checkpoint cut " + std::to_string(walled.cuts) + " commands, or no command was planned "
              "again for the lane's first release");
}

// Two Spill lanes on one conversation that diverge, an agent and its subagent on one context, complete at the limits
// where each completes alone, as two lanes of their own contexts do: the shorter decodes first and the longer yields
// its floor to it, resumes on pages that release moved to the index form, and decodes beside the shorter's tail the
// cache keeps Full. Counted in whole extents of its own pages, the release for the longer's 390 MiB hot block floor
// gave one extent of its 511 pages beside 3 free ones, where compaction empties two (Cache::releaseBeside), and the
// longer lane failed capacity_exhausted from 650 to 725 MB, where lane 1 alone and the two of their own contexts
// complete from 650 and lane 2 alone from 625; on Qwen3.8-Flash-Next at --max-memory 9.38G and 9.43G a 105,672-token
// request failed so beside a 97,849-token one on its first 89,792, where it completes alone and 1.6.0 served both.
void testTwoLanesThatDivergeFromOnePrefixCompleteWhereEachDoesAlone() {
  // From an idle engine with `megabytes` of room, lane 1, 1,534 pages of one context and 512 of its own, and lane 2,
  // the context (or as many pages of its own, `distinct`) and 256 more: the failures.
  const auto failures = [](uint64_t megabytes, bool first, bool second, bool distinct) {
    FlashNextGoverned lanes(test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true));
    Spilling &spill = lanes.spill;
    spill.host.leaveRoom(megabytes * 1'000'000);
    const auto branch = [&](uint64_t id, uint32_t own, uint32_t token) {
      EngineRequest request = Spilling::prompt(id, distinct && id == 2 ? 2 : 1, 1'534, 8);
      request.prompt.insert(request.prompt.end(), own * KvCache::pageTokens, token);
      return request;
    };
    if (first)
      spill.engine.submit(branch(1, 512, 5));
    if (second)
      spill.engine.submit(branch(2, 256, 6));
    spill.runUntilIdle();
    std::string found;
    for (const auto &[message, retryable] : spill.events.failureDetails)
      found += ": " + message.substr(0, 100);
    return found;
  };
  std::string failed;
  const auto check = [&](const char *lanes, uint64_t megabytes, bool first, bool second, bool distinct) {
    if (const std::string found = failures(megabytes, first, second, distinct); !found.empty())
      failed += "; " + std::string(lanes) + " at " + std::to_string(megabytes) + " MB" + found;
  };
  check("lane 2 alone", 625, false, true, false);
  for (const uint64_t megabytes : {650, 675}) {
    check("lane 1 alone", megabytes, true, false, false);
    check("lane 2 alone", megabytes, false, true, false);
  }
  check("two lanes of their own contexts", 650, true, true, true);
  for (const uint64_t megabytes : {650, 675, 700, 725})
    check("two lanes on one context", megabytes, true, true, false);
  require(failed.empty(), "lanes failed at limits where each lane completes alone" + failed);
}

// Two distinct Spill prompts at the memory wall complete where each completes alone. The second, leading the next
// command (Scheduler::prefillAdmissionOrder), prefills beside the first; at its decode the first yields, and the
// second, then alone, counted the pages the first left cached and its own release in whole extents of each, short of
// what their takes free together: having taken nothing, it failed capacity_exhausted from 690 MB, where each completes
// alone. A lone claim takes what the order gives whatever the count (ClaimSpec::lone).
void testTwoDistinctSpillPromptsAtTheWallCompleteWhereEachDoesAlone() {
  Spilling::Pages pages = FlashNextLanes::kPages;
  pages.groupPages = 4 * model::ExecutionLimits::prefillTokenBudget / KvCache::pageTokens;
  // The failures from an idle engine with `megabytes` of room: of a 2,740-page prompt (`first`) and of a prompt of
  // `second` pages (none: no such prompt) submitted while the first's first command runs.
  const auto failures = [&](uint64_t megabytes, bool first, uint32_t second) {
    FlashNextGoverned lanes(test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true), pages);
    Spilling &spill = lanes.spill;
    spill.host.leaveRoom(megabytes * 1'000'000);
    if (first) {
      spill.engine.submit(Spilling::prompt(1, 1, 2'740, 8));
      spill.runUntil([&] { return spill.engine.commandInFlight(); }, "the first prompt did not start");
    }
    if (second)
      spill.engine.submit(Spilling::prompt(2, 2, second, 8));
    spill.runUntilIdle();
    std::string found;
    for (const auto &[message, retryable] : spill.events.failureDetails)
      found += ": " + message.substr(0, 100);
    return found;
  };
  std::string failed;
  const auto check = [&](const char *lanes, uint64_t megabytes, bool first, uint32_t second) {
    if (const std::string found = failures(megabytes, first, second); !found.empty())
      failed += "; " + std::string(lanes) + " at " + std::to_string(megabytes) + " MB" + found;
  };
  for (const uint64_t megabytes : {690, 700, 710, 720}) {
    check("the first alone", megabytes, true, 0);
    check("the second alone", megabytes, false, 2'483);
    check("the second one page longer", megabytes, true, 2'484);
    check("the pair", megabytes, true, 2'483);
  }
  require(failed.empty(), "two Spill prompts failed at limits where each completes alone" + failed);
}

// A Spill lane alone on the chain another lane left cached completes where it completes from an idle engine. Its claim
// for its hot block floor counted the cached pages and its own release in whole extents of each, short of what it
// lacked, and having taken nothing it failed capacity_exhausted at 670 MB: a lone claim takes what the order gives
// whatever the count (ClaimSpec::lone).
void testALaneAloneOnTheChainAnotherLeftCompletesWhereItDoesFromIdle() {
  Spilling::Pages pages = FlashNextLanes::kPages;
  pages.groupPages = 4 * model::ExecutionLimits::prefillTokenBudget / KvCache::pageTokens;
  const auto branch = [](uint64_t id, uint32_t own) {
    EngineRequest request = Spilling::prompt(id, 1, 2'241, 8);
    request.prompt.insert(request.prompt.end(), own * KvCache::pageTokens, static_cast<uint32_t>(id + 1));
    return request;
  };
  // Lane 2's failures from an idle engine with `megabytes` of room, after lane 1 alone (`first`) or not.
  const auto failures = [&](uint64_t megabytes, bool first) {
    FlashNextGoverned lanes(test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true), pages);
    Spilling &spill = lanes.spill;
    spill.host.leaveRoom(megabytes * 1'000'000);
    if (first) {
      spill.engine.submit(branch(1, 501));
      spill.runUntilIdle();
    }
    const size_t before = spill.events.failureDetails.size();
    spill.engine.submit(branch(2, 251));
    spill.runUntilIdle();
    std::string found;
    for (size_t index = before; index < spill.events.failureDetails.size(); ++index)
      found += ": " + spill.events.failureDetails[index].first.substr(0, 100);
    return found;
  };
  std::string failed;
  for (const uint64_t megabytes : {670, 680, 690})
    for (const bool first : {false, true})
      if (const std::string found = failures(megabytes, first); !found.empty())
        failed += "; lane 2 " + std::string(first ? "after lane 1" : "alone") + " at " + std::to_string(megabytes) +
                  " MB" + found;
  require(failed.empty(), "a lane failed alone at limits where it completes from an idle engine" + failed);
}

// A lone Spill lane fails capacity_exhausted only past what it holds alone, and then without waiting out its resource
// wait: its claim takes what the order gives and waits while its writes are in flight, after which nothing else holds
// memory it could get. Where its copies never land, its wait ends at the resource wait's limit; where they fail, the
// SSD cache's file takes no more writes, and it fails at once. The prompt completes from 690 MB.
void testALoneLaneFailsOnlyPastWhatItHoldsAlone() {
  Spilling::Pages pages = FlashNextLanes::kPages;
  pages.groupPages = 4 * model::ExecutionLimits::prefillTokenBudget / KvCache::pageTokens;
  enum class Copies : uint8_t { Land, Stall, Fail };
  // The lane's failures from an idle engine with `megabytes` of room, the copies it writes once it decodes landing,
  // never landing, or failing, ticking 100 ms at a time for up to two resource waits within its own deadline.
  const auto failures = [&](uint64_t megabytes, Copies copies) {
    FlashNextGoverned lanes(test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true), pages);
    Spilling &spill = lanes.spill;
    spill.host.leaveRoom(megabytes * 1'000'000);
    EngineRequest request = Spilling::prompt(1, 1, 2'740, 8);
    request.deadlineMilliseconds = 3 * engine::kResourceWaitTimeoutMilliseconds;
    spill.engine.submit(std::move(request));
    if (copies == Copies::Land) {
      spill.runUntilIdle();
      return spill.events.failures;
    }
    spill.runUntil([&] { return spill.engine.snapshot().activeRequests.front().phase == Phase::Decode; },
                   "the lane did not decode");
    for (const double end = spill.now + 2 * engine::kResourceWaitTimeoutMilliseconds;
         spill.now < end && !idle(spill.engine); spill.now += 100) {
      static_cast<void>(spill.engine.tick(spill.now));
      if (copies == Copies::Fail && spill.tier.inFlight()) {
        spill.tier.fail(WriteClass::Spill);
        spill.tier.writableFile = false;
      }
    }
    require(idle(spill.engine), "the lane's wait for its copies did not end");
    return spill.events.failures;
  };
  const auto check = [&](const char *what, uint64_t megabytes, Copies copies,
                         const std::vector<std::string> &expected) {
    const std::vector<std::string> found = failures(megabytes, copies);
    std::string codes;
    for (const std::string &code : found)
      codes += " " + code;
    require(found == expected, "a lone lane " + std::string(what) + " at " + std::to_string(megabytes) +
                                   " MB ended with failures:" + (codes.empty() ? " none" : codes));
  };
  check("alone", 690, Copies::Land, {});
  check("alone", 680, Copies::Land, {"capacity_exhausted"});
  check("whose copies never land", 680, Copies::Stall, {"resource_timeout"});
  check("whose copies fail", 680, Copies::Fail, {"capacity_exhausted"});
}

// Two Spill lanes on one prompt at the memory wall complete with the memory the longer completes with alone, as on
// 1.6.0: the wall from lane 1's 1,024th page with up to four extents left, held until the lanes end, beside lane 2,
// lane 1's first 1,534 pages, which waits for lane 1 to reach them and then decodes on them at once, taking its hot
// block floor at the wall. Submitted beside lane 1, lane 2 completes from the room lane 1 alone completes from (8/4 of
// an extent); started as a Spill start (FlashNextGoverned::startSpill), it holds that start's room while lane 1
// prefills to its 1,024th page, the wall falls there, and the two complete from 11/4: 653.9 MB as from an idle
// engine, where lane 1 alone completes from 651.2 MB. Counted by the room left at the wall, the second looked like a
// pair failing where a lone lane completes.
void testTwoLanesOnOnePromptAtTheWallNeedNoMoreThanTheLongerAlone() {
  constexpr uint64_t extentBytes = uint64_t{256} * FlashNextLanes::kPages.fullBytes;
  struct Run final {
    uint64_t memory = 0;  // the room at the wall as from an idle engine
    bool completed = false;
  };
  // Lane 1, 2,046 pages, and lane 2 (`two`), its first 1,534, started as a Spill start (`started`) or submitted;
  // the wall from lane 1's 1,024th page with `quarters` of an extent left.
  const auto run = [](uint64_t quarters, bool two, bool started) {
    FlashNextGoverned lanes(test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true));
    Spilling &spill = lanes.spill;
    const uint64_t idle = spill.host.governor.snapshot().headroomBytes;
    lanes.startSpill(Spilling::prompt(1, 1, 2'046, 8));
    if (two && started)
      lanes.startSpill(Spilling::prompt(2, 1, 1'534, 8));
    else if (two)
      spill.engine.submit(Spilling::prompt(2, 1, 1'534, 8));
    const auto processed = [&] {
      for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
        if (active.id == 1)
          return active.promptProcessed;
      return uint32_t{0};
    };
    spill.runUntil([&] { return processed() >= 1'024 * KvCache::pageTokens && !spill.engine.commandInFlight(); },
                   "lane 1 did not reach its 1,024th page");
    const uint64_t wall = spill.host.governor.snapshot().headroomBytes;
    spill.host.leaveRoom(quarters * extentBytes / 4);
    spill.runUntil(
        [&] {
          return !spill.events.failures.empty() ||
                 (spill.events.usage.contains(1) && (!two || spill.events.usage.contains(2)));
        },
        "the lanes did not end at the wall");
    const Run result{.memory = quarters * extentBytes / 4 + idle - wall, .completed = spill.events.failures.empty()};
    spill.host.leaveRoom(std::nullopt);
    spill.runUntilIdle();
    return result;
  };
  uint64_t alone = std::numeric_limits<uint64_t>::max();
  for (uint64_t quarters = 0; quarters <= 16; ++quarters)
    if (const Run each = run(quarters, false, false); each.completed)
      alone = std::min(alone, each.memory);
  require(alone != std::numeric_limits<uint64_t>::max(), "lane 1 alone did not complete with four extents left");
  std::string failed;
  for (const bool started : {false, true})
    for (uint64_t quarters = 0; quarters <= 16; ++quarters)
      if (const Run pair = run(quarters, true, started); pair.memory >= alone && !pair.completed)
        failed += "; " + std::string(started ? "started" : "submitted") + " with " + std::to_string(quarters) +
                  "/4 extent left, " + std::to_string(pair.memory) + " B";
  require(failed.empty(), "two lanes on one prompt failed at the wall with the memory lane 1 alone completes with (" +
                              std::to_string(alone) + " B)" + failed);
}

// A decoding Spill lane takes its hot block floor only at its first release, which frees the floor beside what its
// growth lacks, and the same prepare reserves it: the command after the release reads with it. A Background start
// refused at the same wall, which releases no Normal lane, takes none of it.
void testTheFirstReleaseFundsTheLanesReads() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  // 2,047 pages: one page's room left in its eighth extent.
  lanes.startSpill(Spilling::prompt(1, 1, 2'046, 400));
  lanes.decode(1, 10);
  require(!spill.cache.holdsIndexPages(1) && !spill.model.spilledReadFloors.contains(1),
          "a decoding Spill lane held Index pages or its hot block floor before any release");
  lanes.wall();
  EngineRequest background = Spilling::prompt(2, 2, 1, 4);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  bool floor = false;
  spill.runUntil(
      [&] {
        if (!spill.cache.holdsIndexPages(1))
          return false;
        floor = spill.model.spilledReadFloors.contains(1);
        return true;
      },
      "the decoding lane released none of its pages");
  require(floor && !spill.events.startIds.empty() && std::ranges::count(spill.events.startIds, 2) == 0,
          "the release's prepare did not reserve the lane's floor, or the Background start took memory it freed");
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 400 &&
              spill.engine.snapshot().resourceSuspensions == 0,
          "the lanes did not complete, or one yielded");
}

// A start at the memory wall releases the landed pages of a lane of its own priority rather than wait for it to end:
// a 2K Normal prompt starts beside a Normal Spill lane in decode, which yields nothing and decodes on.
void testAStartReleasesALaneOfItsPriority() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  // 2,047 pages: one page's room left in its eighth extent, the 16 tokens of output within it.
  lanes.startSpill(Spilling::prompt(1, 1, 2'046, 16));
  lanes.decode(1, 4);
  require(!spill.cache.holdsIndexPages(1), "the decoding lane held Index pages before the wall");
  lanes.wall();
  spill.engine.submit(Spilling::prompt(2, 2, 64, 4));
  spill.runUntil([&] { return std::ranges::count(spill.events.startIds, 2) != 0; }, "the start did not start");
  require(spill.cache.holdsIndexPages(1) && !spill.events.usage.contains(1) &&
              spill.engine.snapshot().resourceSuspensions == 0,
          "the start did not release the lane's pages, waited for the lane of its priority to end, or the lane "
          "yielded");
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.completedCount == 2, "the requests did not complete");
}

// A start never releases the pages of a lane of a higher priority: a Normal prompt at the wall waits for a
// Foreground Spill lane to end, the lane's pages left Full.
void testAStartNeverReleasesAHigherPriorityLane() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  // 2,047 pages: one page's room left in its eighth extent, the 16 tokens of output within it.
  EngineRequest foreground = Spilling::prompt(1, 1, 2'046, 16);
  foreground.priority = RequestPriority::Foreground;
  lanes.startSpill(std::move(foreground));
  lanes.decode(1, 4);
  lanes.wall();
  spill.engine.submit(Spilling::prompt(2, 2, 64, 4));
  bool released = false;
  spill.runUntil(
      [&] {
        if (spill.events.usage.contains(1))
          return true;
        released = released || spill.cache.holdsIndexPages(1);
        return false;
      },
      "the Foreground lane did not end");
  require(!released && std::ranges::count(spill.events.startIds, 2) == 0,
          "the Foreground lane's pages left memory before it ended, or the Normal start started first");
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.completedCount == 2, "the requests did not complete");
}

// A decoding Spill lane that never held its hot block floor needs no memory until a release gives it Index pages.
// Under macOS's critical pressure, which refuses every allocation, what the host's pass freed pays for the floor, or
// a lane that needed nothing would stop and time out.
void testACriticalPassPaysForTheFirstReadsOfALaneItReleases() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  lanes.startSpill(Spilling::prompt(1, 1, 3'070, 1'000));
  lanes.decode(1, 10);
  require(!spill.cache.holdsIndexPages(1) && !spill.model.spilledReadFloors.contains(1),
          "the lane held Index pages or its floor before any release");
  spill.host.setPressure(MemoryPressure::Critical);
  const uint64_t extents = 8 * 256 * (FlashNextLanes::kPages.fullBytes - FlashNextLanes::kPages.indexBytes);
  static_cast<void>(test::hostPass(spill.host.governor, extents, true));
  require(spill.cache.holdsIndexPages(1), "the host's pass released none of the lane's pages");
  require(spill.model.spilledReadFloors.contains(1), "the release left the lane without its floor");
  // As far as the pages it has hold; a new page would need memory critical pressure refuses.
  lanes.decode(1, spill.events.outputs[1].size() + 32);
  require(spill.engine.snapshot().resourceSuspensions == 0, "the lane yielded under critical pressure");
  spill.host.setPressure(MemoryPressure::Normal);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.completedCount == 1, "the lane did not complete");
}

// A request alone at the wall waits for the copies a finished Spill lane left in flight, which free the lane's pages
// as they land: refused then, it would be told to shrink its context, where a retry a moment later succeeds.
void testALoneRequestWaitsForTheCopiesAFinishedSpillLaneLeft() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = 32;
  bool land = false;
  const auto run = [&](const std::function<bool()> &done, const char *message) {
    for (uint32_t step = 0; step < 6000 && !done(); ++step) {
      static_cast<void>(spill.engine.tick(spill.now++));
      if (land)
        spill.tier.complete();
    }
    require(done(), message);
  };
  spill.host.leaveRoom(FlashNextGoverned::kStartRoomBytes);
  spill.engine.submit(Spilling::prompt(1, 1, 3'070, 1));
  run([&] { return std::ranges::count(spill.events.startIds, uint64_t{1}) != 0; }, "the lane did not start");
  spill.host.leaveRoom(std::nullopt);
  require(kvMode(spill, 1) == KvMode::Spill, "the lane was not placed Spill");
  run([&] { return spill.events.completedCount == 1; }, "the lane did not finish");
  require(spill.tier.inFlight() != 0, "no copy of the lane's was in flight as it finished");
  require(spill.cache.transfersInFlight(), "the finished lane's copies in flight counted as nothing on its way back");
  lanes.wall();
  spill.engine.submit(Spilling::prompt(2, 2, 3'070, 4));
  for (uint32_t step = 0; step < 300; ++step)
    static_cast<void>(spill.engine.tick(spill.now++));
  require(spill.events.failures.empty(), "a lone request failed while a finished lane's copies were in flight");
  land = true;
  spill.host.leaveRoom(std::nullopt);
  run([&] { return spill.events.completedCount == 2 || !spill.events.failures.empty(); }, "the request did not end");
  require(spill.events.failures.empty(), "the request failed once the copies landed");
}

// A decoding Spill lane of 3,070 pages under macOS's critical pressure, which refuses every allocation, beside an
// expert store of `chunks` 1 GiB chunks over a floor of two: a release at an earlier wall left it Index pages, its hot
// block floor and five extents of pages whose copies landed.
struct CriticalLane {
  explicit CriticalLane(uint32_t chunks = 2)
      : experts{"experts", spill.host.memory, 1ULL << 30, 2, chunks, chunks, chunks} {
    static_cast<void>(spill.host.governor.ledger().addPool(experts));
    lanes.startSpill(Spilling::prompt(1, 1, 3'070, 1'000));
    lanes.decode(1, 10);
    lanes.wall();
    spill.runUntil(
        [&] {
          return spill.cache.holdsIndexPages(1) && spill.model.spilledReadFloors.contains(1) &&
                 !spill.engine.commandInFlight();
        },
        "the lane did not release its pages at the wall");
    spill.host.setPressure(MemoryPressure::Critical);
  }
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  test::FakeElasticPool experts;
};

// Under critical pressure the host's pass releases a lane's pages whose copies landed before any lane yields and
// before the expert store: with a chunk above its floor, the store gives nothing to a pass of two extents' pages,
// 204 MB on Qwen3.8-Flash-Next, which the lane's release covers.
void testCriticalReleasesCleanPagesBeforeAnyLaneYields() {
  CriticalLane critical(3);
  Spilling &spill = critical.spill;
  const uint32_t before = spill.cache.indexPages(1);
  const uint64_t extents = 2 * 256 * (FlashNextLanes::kPages.fullBytes - FlashNextLanes::kPages.indexBytes);
  const FreeResult freed = test::hostPass(spill.host.governor, extents, true);
  require(freed.nowBytes >= extents && spill.cache.indexPages(1) > before && critical.experts.shrinks == 0,
          "the host's pass under critical pressure freed " + std::to_string(freed.nowBytes) + " B, released " +
              std::to_string(spill.cache.indexPages(1) - before) + " of the lane's pages and took " +
              std::to_string(critical.experts.shrinks) + " store chunks");
  critical.lanes.decode(1, spill.events.outputs[1].size() + 64);
  require(spill.engine.snapshot().resourceSuspensions == 0, "the lane yielded under critical pressure");
  spill.host.setPressure(MemoryPressure::Normal);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 1'000, "the lane did not complete");
}

// Under macOS's warning pressure the host's pass for its deficit releases the pages whose copies landed of the Spill
// lanes, the lowest-ranked first, as its pass under critical pressure does: the Full extents they empty go back to the
// host, as 1.6.0's lanes gave them back once their copies landed. At normal pressure the same deficit leaves them
// Full. Either way the lane decodes on and completes. On the parent the pass at warning released none.
void testAWarningDeficitReleasesALanesLandedPages() {
  for (const MemoryPressure pressure : {MemoryPressure::Normal, MemoryPressure::Warning}) {
    FlashNextGoverned lanes;
    Spilling &spill = lanes.spill;
    lanes.startSpill(Spilling::prompt(1, 1, 3'070, 1'000));
    spill.runUntil(
        [&] {
          const KvSpillSnapshot written = spill.cache.spillSnapshot();
          return spill.events.outputs[1].size() >= 10 && written.writeThroughPages >= 3'000 &&
                 written.writeThroughInFlight == 0 && !spill.engine.commandInFlight();
        },
        "the lane's copies did not land");
    const auto fullBytes = [&] {
      const KvPoolSnapshot pool = spill.pool.snapshot();
      return pool.allocatedBytes - pool.index.allocatedBytes;
    };
    const uint64_t full = fullBytes();
    // Other applications leave the host half its margin: a deficit.
    spill.host.pause(true);
    spill.host.setPressure(pressure);
    static_cast<void>(spill.host.governor.controlPass(spill.now));
    const uint64_t released = spill.cache.spillSnapshot().releasedPages;
    const uint64_t extent = 256 * FlashNextLanes::kPages.fullBytes;
    const bool warning = pressure == MemoryPressure::Warning;
    require(warning ? released >= 2 * 256 && fullBytes() + 2 * extent <= full
                    : released == 0 && !spill.cache.holdsIndexPages(1) && fullBytes() == full,
            std::string(warning ? "under warning pressure" : "at normal pressure") + " the host's deficit pass released " +
                std::to_string(released) + " of the lane's pages and took its Full extents from " +
                std::to_string(full) + " B to " + std::to_string(fullBytes()) + " B");
    spill.host.setPressure(MemoryPressure::Normal);
    spill.host.pause(false);
    spill.runUntilIdle();
    require(spill.events.failures.empty() && spill.events.usage.at(1).second == 1'000 &&
                spill.engine.snapshot().resourceSuspensions == 0,
            "the lane did not complete without a yield after the host's pass" + failureDetail(spill.events));
  }
}

// With Metal memory outside the engine's buffers holding the charge over the engine's limit by more than a Full
// extent, the host's pass under critical pressure still releases two extents of a lane's pages, 204 MB on
// Qwen3.8-Flash-Next: each runway's emptied Full extent pays for the next Index extent past the limit as past the
// host (MemoryGovernor::fund), and the runway stands again once the release ends. Refused by the limit, the first
// refill ended the release at one runway, 108,527,680 B, and left no free Index page for the next.
void testACriticalPassReleasesPastTheEnginesLimit() {
  CriticalLane critical;
  Spilling &spill = critical.spill;
  const uint64_t extent = 256 * FlashNextLanes::kPages.fullBytes;
  spill.host.memory.holdUntracked(spill.host.governor.snapshot().headroomBytes + extent + extent / 2);
  const uint32_t before = spill.cache.indexPages(1);
  const uint64_t extents = 2 * 256 * (FlashNextLanes::kPages.fullBytes - FlashNextLanes::kPages.indexBytes);
  const FreeResult freed = test::hostPass(spill.host.governor, extents, true);
  const uint32_t released = spill.cache.indexPages(1) - before;
  require(freed.nowBytes >= extents && released >= 2 * 256 &&
              spill.pool.freePageCount(kv::PageForm::Index) >= spill.pool.extentPages(),
          "over the engine's limit the host's pass freed " + std::to_string(freed.nowBytes) + " B, released " +
              std::to_string(released) + " of the lane's pages and left " +
              std::to_string(spill.pool.freePageCount(kv::PageForm::Index)) + " free Index pages");
  spill.host.memory.holdUntracked(0);
  spill.host.setPressure(MemoryPressure::Normal);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 1'000, "the lane did not complete");
}

// A take a refused allocation cuts short is no overcount: the driver denies the Index extent the host's pass under
// critical pressure refills after its first runway, so the release moves one runway of the two extents it counted,
// and /status counts neither an overcounted take nor a short release (MemoryLedger::CountChecks), as for the host's
// refusal or the engine's limit. On the parent the driver's denial counted as both, an accounting defect to report.
void testARefillTheDriverDeniesIsNoOvercount() {
  CriticalLane critical;
  Spilling &spill = critical.spill;
  bool denied = false;
  spill.storage.driverDenies = [&](kv::PageForm form) {
    return form == kv::PageForm::Index && !std::exchange(denied, true);
  };
  const uint32_t before = spill.cache.indexPages(1);
  const uint64_t extents = 2 * 256 * (FlashNextLanes::kPages.fullBytes - FlashNextLanes::kPages.indexBytes);
  const FreeResult freed = test::hostPass(spill.host.governor, extents, true);
  const uint32_t released = spill.cache.indexPages(1) - before;
  const MemoryLedger::CountChecks checks = spill.host.governor.snapshot().countChecks;
  require(denied && freed.nowBytes < extents && released == 256 && checks.coveredRuns != 0 &&
              checks.overcountedRuns == 0 && checks.shortReleases == 0,
          "a pass whose refill the driver denied released " + std::to_string(released) + " pages, freed " +
              std::to_string(freed.nowBytes) + " B and counted " + std::to_string(checks.overcountedRuns) +
              " overcounted takes and " + std::to_string(checks.shortReleases) + " short releases");
  spill.storage.driverDenies = {};
  spill.host.setPressure(MemoryPressure::Normal);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 1'000, "the lane did not complete");
}

// Under critical pressure a lane's growth the host refuses releases its own lane's pages whose copies landed, in
// place: into free Index pages, the Full pages they free taken by the growth, so the lane decodes on rather than
// yield.
void testAHostRefusedGrowthReleasesItsOwnLane() {
  CriticalLane critical;
  Spilling &spill = critical.spill;
  // No free Full page: the growth asks for an extent.
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  const uint32_t before = spill.cache.indexPages(1);
  critical.lanes.decode(1, spill.events.outputs[1].size() + 64);
  require(held.granted() && spill.engine.snapshot().resourceSuspensions == 0 && spill.cache.indexPages(1) > before,
          "a growth the host refused under critical pressure did not release its own lane's pages, or the lane "
          "yielded");
  for (const uint32_t page : held.pages)
    spill.pool.releasePage(page, false);
  spill.host.setPressure(MemoryPressure::Normal);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 1'000, "the lane did not complete");
}

// A release in place keeps the Index runway, which under critical pressure only the bytes a release frees refill: a
// growth the host refuses with no free Full page releases an extent of its lane's pages more than it takes, whose
// emptied Full extent refills the runway. The host's next pass then still releases two extents of the lane's pages,
// 204 MB on Qwen3.8-Flash-Next, with the expert store at its floor, and leaves an extent of free Index pages.
void testAReleaseInPlaceKeepsTheIndexRunway() {
  CriticalLane critical;
  Spilling &spill = critical.spill;
  // No free Full page: the growth asks for an extent.
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  critical.lanes.decode(1, spill.events.outputs[1].size() + 64);
  const uint32_t before = spill.cache.indexPages(1);
  const uint64_t extents = 2 * 256 * (FlashNextLanes::kPages.fullBytes - FlashNextLanes::kPages.indexBytes);
  const FreeResult freed = test::hostPass(spill.host.governor, extents, true);
  const uint32_t runway = spill.pool.freePageCount(kv::PageForm::Index);
  require(held.granted() && spill.engine.snapshot().resourceSuspensions == 0 && freed.nowBytes >= extents &&
              spill.cache.indexPages(1) >= before + 512 && runway >= spill.pool.extentPages() &&
              critical.experts.shrinks == 0,
          "after a release in place under critical pressure the host's pass freed " + std::to_string(freed.nowBytes) +
              " B, released " + std::to_string(spill.cache.indexPages(1) - before) + " of the lane's pages, took " +
              std::to_string(critical.experts.shrinks) + " store chunks and left " + std::to_string(runway) +
              " free Index pages");
  for (const uint32_t page : held.pages)
    spill.pool.releasePage(page, false);
  spill.host.setPressure(MemoryPressure::Normal);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 1'000, "the lane did not complete");
}

// A release in place keeps the Index runway at every pressure level, since the room may hold no Index extent to refill
// it, 6.3 MB on Qwen3.8-Flash-Next. A first wall's release leaves a Background lane 162 pages whose copies landed, too
// few for an extent more: at a second wall, with no free Full page and no room, a Normal lane's growth takes none of
// them in place but releases its own lane's pages. The host's pass under critical pressure then still empties two Full
// extents, of the lanes' pages and the free Full pages beside them (Cache::releaseBeside).
void testAReleaseInPlaceKeepsTheRunwayTheRoomCannotRefill() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  EngineRequest background = Spilling::prompt(3, 3, 1'700, 400);
  background.priority = RequestPriority::Background;
  lanes.startSpill(std::move(background));
  lanes.decode(3, 4);
  lanes.startSpill(Spilling::prompt(1, 1, 3'070, 600));
  lanes.decode(1, 64);
  require(!spill.cache.holdsIndexPages(1) && !spill.cache.holdsIndexPages(3),
          "a lane held Index pages before the wall");
  // At each wall no Full page is free: the first releases the Background lane's pages for the Normal lane's growth.
  lanes.wall();
  const KvPageAcquisition first = spill.pool.acquirePages(spill.pool.freePageCount());
  spill.runUntil([&] { return spill.cache.holdsIndexPages(3) && !spill.engine.commandInFlight(); },
                 "the Background lane released none of its pages at the first wall");
  lanes.wall();
  const KvPageAcquisition second = spill.pool.acquirePages(spill.pool.freePageCount());
  const uint32_t backgroundIndex = spill.cache.indexPages(3), normalIndex = spill.cache.indexPages(1);
  lanes.decode(1, spill.events.outputs[1].size() + 64);
  require(spill.cache.indexPages(3) == backgroundIndex && spill.cache.indexPages(1) > normalIndex,
          "at the second wall the Normal lane's growth moved " +
              std::to_string(spill.cache.indexPages(3) - backgroundIndex) + " of the Background lane's pages and " +
              std::to_string(spill.cache.indexPages(1) - normalIndex) + " of its own");
  const uint32_t runway = spill.pool.freePageCount(kv::PageForm::Index);
  spill.host.setPressure(MemoryPressure::Critical);
  const uint32_t before = spill.cache.indexPages(1), backgroundBefore = spill.cache.indexPages(3);
  const uint32_t fullExtents = spill.pool.allocatedExtents(kv::PageForm::Full);
  const uint64_t extents = 2 * 256 * (FlashNextLanes::kPages.fullBytes - FlashNextLanes::kPages.indexBytes);
  const FreeResult freed = test::hostPass(spill.host.governor, extents, true);
  const uint32_t emptied = fullExtents - spill.pool.allocatedExtents(kv::PageForm::Full);
  require(first.granted() && second.granted() && runway >= spill.pool.extentPages() && freed.nowBytes >= extents &&
              emptied >= 2,
          "after growths at a wall with no room the runway held " + std::to_string(runway) +
              " free Index pages, and the host's pass under critical pressure freed " +
              std::to_string(freed.nowBytes) + " B, released " + std::to_string(spill.cache.indexPages(1) - before) +
              " of the Normal lane's pages and " + std::to_string(spill.cache.indexPages(3) - backgroundBefore) +
              " of the Background lane's, and emptied " + std::to_string(emptied) + " Full extents");
  for (const KvPageAcquisition *held : {&first, &second})
    for (const uint32_t page : held->pages)
      spill.pool.releasePage(page, false);
  spill.host.setPressure(MemoryPressure::Normal);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.completedCount == 2, "the lanes did not complete");
}

// A growth takes its page in place before a release for bytes, which moves more pages and may still leave it short:
// at a wall with no free Full page and no room, a lone decoding lane whose first release left it fewer than two
// extents of pages whose copies landed moves its page and an extent more in place, where a release of one extent for
// bytes would free 102 MB of the 108.5 MB its Full extent lacks and leave too few pages to move in place.
void testAGrowthTakesItsPageInPlaceBeforeAReleaseForBytes() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  lanes.startSpill(Spilling::prompt(1, 1, 2'000, 600));
  lanes.decode(1, 64);
  require(!spill.cache.holdsIndexPages(1), "the lane held Index pages before the wall");
  // At each wall no Full page is free: the first releases the lane's pages for its growth and floor.
  lanes.wall();
  const KvPageAcquisition first = spill.pool.acquirePages(spill.pool.freePageCount());
  spill.runUntil([&] { return spill.cache.holdsIndexPages(1) && !spill.engine.commandInFlight(); },
                 "the lane released none of its pages at the first wall");
  lanes.wall();
  const KvPageAcquisition second = spill.pool.acquirePages(spill.pool.freePageCount());
  const uint32_t before = spill.cache.indexPages(1);
  const size_t decoded = spill.events.outputs[1].size() + 64;
  uint32_t released = 0;
  spill.runUntil(
      [&] {
        if (spill.events.usage.contains(1))
          return true;
        released = spill.cache.indexPages(1) - before;
        return spill.events.outputs[1].size() >= decoded && !spill.engine.commandInFlight();
      },
      "the lane neither decoded nor ended");
  require(first.granted() && second.granted() && !spill.events.usage.contains(1) && released < 512,
          "the lane at the second wall released " + std::to_string(released) + " pages, or ended after " +
              std::to_string(spill.events.outputs[1].size()) + " tokens");
  for (const KvPageAcquisition *held : {&first, &second})
    for (const uint32_t page : held->pages)
      spill.pool.releasePage(page, false);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 600, "the lane did not complete");
}

// A lane whose command is between its passes reads its history as the command's first pass cut it, so it gains no
// Index page there: no claim releases it, and its pages do not move as their copies land. A start of its priority at
// the wall leaves its pages Full until the command's last pass.
void testNoReleaseOfALaneBetweenItsPasses() {
  FlashNextGoverned lanes(test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true));
  Spilling &spill = lanes.spill;
  std::vector<std::pair<uint32_t, uint32_t>> passes;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill)
      passes.emplace_back(plan.pass, plan.passes);
    return test::CommandTimes{};
  };
  const auto between = [&] { return !passes.empty() && passes.back().first + 1 < passes.back().second; };
  lanes.startSpill(Spilling::prompt(1, 1, 2'046, 8));
  spill.runUntil(
      [&] { return spill.cache.pageTable(1).pages.size() >= 1'024 && between() && !spill.engine.commandInFlight(); },
      "the prompt did not reach its 1,024th page between a command's passes");
  lanes.wall();
  spill.engine.submit(Spilling::prompt(2, 2, 64, 4));
  bool released = false;
  spill.runUntil(
      [&] {
        released = released || (between() && spill.cache.holdsIndexPages(1));
        return !between();
      },
      "the command did not reach its last pass");
  require(!released, "a lane gained Index pages between its command's passes");
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.completedCount == 2, "the requests did not complete");
}

// Requests on one prefix share its pages, so a release moves a page only where its claim may take it from every lane
// that holds it: neither a decoding Spill lane's growth at the wall nor a start of its priority releases the lane's
// conversation while a longer prompt on it, which holds those pages, is between its command's passes, and all three
// requests complete.
void testNoReleaseOfAPageALaneBetweenItsPassesHolds() {
  FlashNextGoverned lanes(test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true));
  Spilling &spill = lanes.spill;
  std::vector<std::pair<uint32_t, uint32_t>> passes;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill)
      passes.emplace_back(plan.pass, plan.passes);
    return test::CommandTimes{};
  };
  const auto between = [&] { return !passes.empty() && passes.back().first + 1 < passes.back().second; };
  lanes.startSpill(Spilling::prompt(1, 1, 2'046, 4'096));
  lanes.decode(1, 4);
  // Lane 2's prompt is lane 1's and 2,048 pages more: it holds lane 1's pages and prefills the rest in passes. Lane 1
  // then decodes eight tokens a step, its verify rows', so its growth meets the wall within four steps.
  lanes.startSpill(Spilling::prompt(2, 1, 4'094, 8));
  spill.model.decodeTokens = model::ExecutionLimits::targetVerifyRows;
  spill.runUntil([&] { return between() && !spill.engine.commandInFlight(); },
                 "lane 2's prompt did not come between a command's passes");
  require(!spill.cache.holdsIndexPages(1) && !spill.cache.holdsIndexPages(2),
          "a lane held Index pages before the wall");
  lanes.wall();
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  spill.engine.submit(Spilling::prompt(3, 3, 64, 8));
  // The command goes on to its last pass, or lane 2 yields, which gives it up.
  const auto inCommand = [&] { return between() && spill.model.requests.at(2).resident; };
  bool released = false;
  spill.runUntil(
      [&] {
        released = released || (inCommand() && spill.cache.holdsIndexPages(2));
        return !inCommand();
      },
      "the command did not reach its last pass");
  require(held.granted() && !released, "a page a lane between its command's passes holds left memory");
  for (const uint32_t page : held.pages)
    spill.pool.releasePage(page, false);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.completedCount == 3, "the requests did not complete");
}

// Guard: a lone decoding Spill lane at the engine's limit takes no Index page whose hot block floor (390 MiB) its
// release cannot free beside its growth: each page frees its Full bytes less its Index page's, so even four extents of
// pages whose copies landed free the floor alone. With 1,021 such pages it releases none and ends with Length at the
// wall, rather than fail for its reads.
void testALoneLaneWhoseReleaseCannotPayItsReadsEndsAtItsLength() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  // 1,023 pages and a token: 1,021 pages past the dense history and window, under four extents.
  lanes.startSpill(Spilling::prompt(1, 1, 1'023, 4'096));
  lanes.decode(1, 64);
  lanes.wall();
  // No free Full page: the growth asks for an extent.
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  spill.runUntil([&] { return idle(spill.engine); }, "the lane did not end at the wall");
  require(held.granted() && spill.events.failures.empty() && spill.events.usage.contains(1) &&
              spill.engine.snapshot().spill.lengthEnds == 1,
          "a lone Spill lane whose release could not pay for its reads did not end with Length at the wall: " +
              (spill.events.failures.empty() ? std::string("no failure") : spill.events.failureDetails.front().first) +
              ", " + std::to_string(spill.engine.snapshot().spill.lengthEnds) + " length ends");
  for (const uint32_t page : held.pages)
    spill.pool.releasePage(page, false);
}

// Guard: a lone Spill lane at the wall whose copies are still being written waits for them, then releases them, and
// prefills on with no resource_timeout and no failure.
void testALoneLaneWaitsForItsLastWritesThenReleasesThem() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  spill.allowSpillWrites(false);
  lanes.startSpill(Spilling::prompt(1, 1, 2'046, 8));
  spill.runUntil([&] { return spill.cache.pageTable(1).pages.size() >= 1'024 && !spill.engine.commandInFlight(); },
                 "the prompt did not reach its 1,024th page");
  lanes.wall();
  spill.allowSpillWrites(true);
  bool waited = false;
  spill.runUntil(
      [&] {
        waited = waited || spill.host.governor.snapshot().claims.waiting != 0;
        return spill.cache.holdsIndexPages(1);
      },
      "the lane released none of its pages once their copies landed");
  require(waited, "the lane at the wall did not wait for its copies in flight");
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 8,
          "the lone lane failed at the wall while its copies were landing");
}

// A decoding lane whose pages another lane's claim released reads them with its hot block floor from its next
// command: the release frees the floor beside what the claim lacks, and the lane's next prepare reserves it. Until
// then it decodes on its pages in memory, with no floor.
void testAReleaseLeavesNoLaneWithoutItsFloor() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  // 2,047 pages: one page's room left in its eighth extent, the 16 tokens of output within it.
  EngineRequest background = Spilling::prompt(1, 1, 2'046, 16);
  background.priority = RequestPriority::Background;
  lanes.startSpill(std::move(background));
  lanes.decode(1, 4);
  require(!spill.cache.holdsIndexPages(1) && !spill.model.spilledReadFloors.contains(1),
          "a decoding Spill lane no claim released held Index pages or a hot block floor");
  // A Normal start at the wall releases the lane's pages.
  lanes.wall();
  spill.engine.submit(Spilling::prompt(2, 2, 64, 600));
  spill.runUntil([&] { return spill.cache.holdsIndexPages(1); }, "no claim released the Spill lane's pages");
  const size_t decoded = spill.events.outputs[1].size();
  lanes.decode(1, decoded + 1);
  require(spill.model.spilledReadFloors.contains(1), "a lane whose pages a release took decoded without its floor");
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.completedCount == 2 &&
              spill.engine.snapshot().resourceSuspensions == 0,
          "the lanes did not complete after the release, or one yielded");
}

// Two Spill lanes that hold no Index page read no spilled history: they prefill together as Resident lanes do, the
// longer one's 193 rows past its whole commands beside the shorter.
void testTwoSpillLanesWithoutIndexPagesPrefillTogether() {
  // Flash-Next's pages, and an engine's limit short of one extent: every start spills.
  Spilling spill{16'384, false, 0, 200'000, Spilling::Extent{256}, {}, {.limitBytes = 50ULL << 20}, false,
                 FlashNextLanes::kPages};
  std::vector<std::vector<uint64_t>> prefills;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill) {
      std::vector<uint64_t> lanes;
      for (const BatchItem &item : plan.items)
        lanes.push_back(item.requestId);
      prefills.push_back(std::move(lanes));
    }
    return test::CommandTimes{};
  };
  spill.engine.submit(Spilling::prompt(1, 1, 15, 4));
  spill.engine.submit(Spilling::prompt(2, 2, 70, 4));
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.engine.snapshot().spill.spill == 2 &&
              std::ranges::any_of(prefills, [](const std::vector<uint64_t> &lanes) { return lanes.size() == 2; }),
          "two Spill lanes that hold no Index page prefilled apart");
}

// Guard: a start on a decoding Spill lane's conversation counts the pages of the chain it restores once, as its own
// in place, and not again as the lane's to release, since the start then holds them: with 200 MB of room its 1,001
// pages past the chain lack four extents (434 MB) memory does not hold, so it starts Spill, and both complete with no
// yield.
void testAStartOnALanesConversationCountsItsChainOnce() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  lanes.startSpill(Spilling::prompt(1, 1, 2'046, 400));
  lanes.decode(1, 4);
  spill.host.leaveRoom(200'000'000);
  spill.engine.submit(Spilling::prompt(2, 1, 3'046, 8));
  spill.runUntil([&] { return std::ranges::count(spill.events.startIds, 2) != 0; }, "the start did not start");
  require(kvMode(spill, 2) == KvMode::Spill,
          "a start counted the chain it restores as its own and as the lane's to release, and started Resident");
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.completedCount == 2 &&
              spill.engine.snapshot().resourceSuspensions == 0,
          "the requests did not complete, or one yielded");
}

// A Resident lane reads its history Full, and a Resident start's count holds the chain it restores in place as its
// own: a start on the first 1,500 pages of a decoding Spill lane's conversation, placed Resident at the wall for the
// lane's other pages its claims release, releases none of that chain in its own claims, and restores it with no Index
// page.
void testAResidentStartOnALanesPrefixRestoresItWhole() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  // A lane's start allocates 200 MB of state buffers beside those the pool keeps.
  spill.model.laneFootprintBytes = 200'000'000;
  // Request 10 leaves a 1,500-page chain cached; lane 1 restores it and prefills 3,500 pages more.
  spill.engine.submit(Spilling::prompt(10, 1, 1'500, 1));
  spill.runUntilIdle();
  lanes.startSpill(Spilling::prompt(1, 1, 5'000, 400));
  lanes.decode(1, 4);
  require(!spill.cache.holdsIndexPages(1), "the decoding lane held Index pages before the wall");
  lanes.wall();
  // Request 10's prompt again: its chain is lane 1's first 1,500 pages.
  spill.engine.submit(Spilling::prompt(2, 1, 1'500, 8));
  spill.runUntil([&] { return std::ranges::count(spill.events.startIds, 2) != 0; }, "the start did not start");
  require(kvMode(spill, 2) == KvMode::Resident && spill.cache.holdsIndexPages(1),
          "the start was not placed Resident, or released none of the lane's pages");
  require(!spill.cache.holdsIndexPages(2),
          "a Resident start's claims released the chain its count held, which it restored with " +
              std::to_string(spill.cache.indexPages(2)) + " Index pages");
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.completedCount == 3 &&
              spill.engine.snapshot().resourceSuspensions == 0,
          "the requests did not complete, or one yielded");
}

// A release of pages several lanes share gives each of them Index pages, so it frees the first reads of every one
// that holds none yet beside what its claim lacks: a start on a decoding Spill lane's conversation, its 1,001 pages
// past the chain at 50 MB of room, releases their shared prefix for its growth and staging and for the decoding
// lane's 390 MiB hot block floor, which memory then holds for that lane's next token.
void testAReleaseOfASharedPrefixFundsEveryLanesReads() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  lanes.startSpill(Spilling::prompt(1, 1, 2'046, 400));
  lanes.decode(1, 4);
  require(!spill.cache.holdsIndexPages(1), "the decoding lane held Index pages before any release");
  spill.host.leaveRoom(50'000'000);
  spill.engine.submit(Spilling::prompt(2, 1, 3'046, 8));
  spill.runUntil([&] { return spill.cache.holdsIndexPages(1) && !spill.engine.commandInFlight(); },
                 "no release gave the decoding lane Index pages");
  const uint64_t headroom = spill.host.governor.snapshot().headroomBytes;
  require(spill.model.spilledReadFloors.contains(1) || headroom >= spill.model.spilledReadFloorBytes,
          "a release of a shared prefix left the decoding lane " + std::to_string(headroom) +
              " B for its 390 MiB hot block floor");
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.completedCount == 2 &&
              spill.engine.snapshot().resourceSuspensions == 0,
          "the requests did not complete, or one yielded");
}

// The first take that moves a page several lanes share frees it, so a count of their releases counts it once: two
// Normal Spill lanes on one conversation, of 3,000 pages and of those and 1,000 more, give a start at the wall what
// the longer one's release frees, 715.7 MB on Qwen3.8-Flash-Next. A Normal start of 2,300 pages that cannot spill,
// the SSD cache closed, so waits for the shorter lane to end, rather than start beside both, release every page they
// hold and yield.
void testACountOfReleasesCountsASharedPageOnce() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  lanes.startSpill(Spilling::prompt(1, 1, 3'000, 400));
  lanes.decode(1, 4);
  lanes.startSpill(Spilling::prompt(2, 1, 4'000, 400));
  lanes.decode(2, 4);
  require(!spill.cache.holdsIndexPages(1) && !spill.cache.holdsIndexPages(2),
          "a lane held Index pages before the wall");
  spill.tier.writableFile = false;
  lanes.wall();
  spill.engine.submit(Spilling::prompt(3, 3, 2'300, 8));
  bool beside = false;
  spill.runUntil(
      [&] {
        beside = beside || (std::ranges::count(spill.events.startIds, 3) != 0 && !spill.events.usage.contains(1));
        return idle(spill.engine);
      },
      "the requests did not end");
  require(!beside && spill.events.failures.empty() && spill.events.completedCount == 3 &&
              spill.engine.snapshot().resourceSuspensions == 0,
          "a start that cannot spill started beside two lanes on one conversation, or a request failed or yielded (" +
              std::to_string(spill.engine.snapshot().resourceSuspensions) + " yields)");
}

// A count of releases holds for the takes made as the copies stand; copies that land before them can lower it, as a
// lane's landed pages then join, in its whole extents, shared pages a later lane's unit counted. Two Normal Spill
// lanes on a 2,433-page prefix, of 769 and 897 pages more, count at most an extent of pages, 102 MB on
// Qwen3.8-Flash-Next, less once the shorter one's copies have landed, with nothing taken.
void testACountOfReleasesFallsByAtMostAnExtentAsCopiesLand() {
  FlashNextGoverned lanes;
  Spilling &spill = lanes.spill;
  // Request 10 leaves the prefix cached; each lane restores it and prefills pages of its own.
  spill.engine.submit(Spilling::prompt(10, 1, 2'433, 1));
  spill.runUntilIdle();
  const auto branch = [](uint64_t id, uint32_t pages) {
    EngineRequest result = Spilling::prompt(id, 1, 2'433, 4'000);
    result.prompt.insert(result.prompt.end(), pages * KvCache::pageTokens, static_cast<uint32_t>(id + 1));
    return result;
  };
  lanes.startSpill(branch(2, 897));
  spill.runUntil(
      [&] {
        return spill.events.outputs[2].size() >= 4 && !spill.cache.owesCopies(2) && spill.tier.inFlight() == 0 &&
               !spill.engine.commandInFlight();
      },
      "the longer lane's copies did not land");
  require(!spill.cache.holdsIndexPages(2), "the longer lane held Index pages before any release");
  lanes.startSpill(branch(1, 769));
  // The shorter lane prefills its own pages, none of their copies landing.
  for (uint32_t step = 0; step < 4'000 && (spill.events.outputs[1].empty() || spill.engine.commandInFlight()); ++step)
    static_cast<void>(spill.engine.tick(spill.now++));
  require(!spill.cache.holdsIndexPages(1) && spill.tier.inFlight() != 0,
          "the shorter lane held Index pages, or its copies landed, before the wall");
  const ClaimSpec start{.cls = ClaimClass::Start, .priority = RequestPriority::Normal};
  lanes.wall();
  const uint64_t landing = spill.host.governor.available(start, {.bytes = 20ULL << 30});
  spill.host.leaveRoom(std::nullopt);
  spill.runUntil(
      [&] { return !spill.cache.owesCopies(1) && spill.tier.inFlight() == 0 && !spill.engine.commandInFlight(); },
      "the shorter lane's copies did not land");
  lanes.wall();
  const uint64_t landed = spill.host.governor.available(start, {.bytes = 20ULL << 30});
  const uint64_t extent =
      uint64_t{spill.pool.extentPages()} * (FlashNextLanes::kPages.fullBytes - FlashNextLanes::kPages.indexBytes);
  require(spill.cache.spillSnapshot().releases == 0 && landed + extent >= landing,
          "the count of releases fell from " + std::to_string(landing) + " B to " + std::to_string(landed) +
              " B as copies landed, by more than an extent of pages, or a claim took some");
}

// A lone Spill prefill at the engine's limit, Metal memory outside the engine's buffers taking the room and 1 GiB
// more, names why its KV gives no more memory back: where the SSD cache took no copy of its pages, as while the tier
// writes none of them, the SSD cache; where it took them all, that even a release down to its floor does not cover
// what it lacks, and not the SSD cache, which holds its KV.
void testALoneSpillWallNamesWhyItsKvGaveNoMoreMemory() {
  for (const bool written : {true, false}) {
    FlashNextGoverned lanes;
    Spilling &spill = lanes.spill;
    spill.allowSpillWrites(written);
    lanes.startSpill(Spilling::prompt(1, 1, 2'046, 8));
    spill.runUntil([&] { return spill.cache.pageTable(1).pages.size() >= 1'024 && !spill.engine.commandInFlight(); },
                   "the prompt did not reach its 1,024th page");
    const uint64_t untracked = spill.host.governor.snapshot().headroomBytes + (1ULL << 30);
    spill.host.memory.holdUntracked(untracked);
    spill.runUntilIdle();
    require(spill.events.failures == std::vector<std::string>{"capacity_exhausted"},
            "the lone Spill prefill did not fail at the limit");
    const std::string &message = spill.events.failureDetails.front().first;
    const std::string_view named = written ? "even with its KV released down to its floor"
                                           : "the SSD cache took no more of its KV";
    const std::string_view other = written ? "SSD cache" : "down to its floor";
    require(message.find(", a Spill lane of ") != std::string::npos &&
                message.find(" tokens in prefill, ") != std::string::npos &&
                message.find(named) != std::string::npos && message.find(other) == std::string::npos &&
                message.find("device_untracked_bytes=" + std::to_string(untracked)) != std::string::npos,
            std::string("a lone Spill prefill whose copies ") + (written ? "landed" : "the tier did not take") +
                " did not fail naming its lane, \"" + std::string(named) + "\" alone and the memory outside the "
                "engine's buffers: " + message);
  }
}

// A lone Spill lane the tier takes no more pages of: decoding, it ends with Length at its bound;
// prefilling, its first command's ten pages, more than the eight memory holds, fail naming its lane
// Spill in prefill and that even a release down to its floor does not cover them: it published no
// page whose copy the SSD cache did not take.
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
      require(spill.events.failures == std::vector<std::string>{"capacity_exhausted"},
              "a lone Spill prefill at the wall did not fail");
      const std::string &message = spill.events.failureDetails.front().first;
      require(message.find("; the request, a Spill lane of 321 tokens in prefill, ") != std::string::npos &&
                  message.find("; even with its KV released down to its floor") != std::string::npos,
              "a lone Spill prefill at the wall did not fail naming its lane and its floor: " + message);
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

// A KV need takes cached pages of its form in place, with no extent emptied. Under --kv-spill always,
// since its premise is pages that move as their copies land, a Spill lane's landed copies free pages
// in its extents, which a short prompt beside it, spilling too, takes; the prompt yields or ends,
// and the lane's next page comes from the prompt's cached pages: it runs to its end.
void testALaneTakesCachedPagesInPlace() {
  for (const uint64_t room : {0, 50, 100}) {
    Spilling spill(Spilling::Governed{}, 1024, true, Spilling::Extent{4}, test::commandShape(KvCache::pageTokens));
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
// the lane's buffers. Counted against the pool before the start, it started
// Resident in the first room and failed at its KV.
void testPlacementCountsTheLanesBuffersAStartTakesFromThePool() {
  constexpr uint64_t lane = 128, state = 64, shortRoom = 1'100;
  for (const uint64_t room : {shortRoom, shortRoom + lane}) {
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
// lane's buffers, and its claims evict no state: placement charges its start
// nothing, as the memory plan charges a request its lane's buffers once, nor
// the states its prefill captures, and counts as room only the idle buffers
// beyond those its start takes. A 20-page next turn prefilled a page a
// command, beside no idle buffers and beside a second lane's, runs Resident
// in room for its pages, and spills in room a state's 64 bytes short of
// them. Charged the lane's buffers whole, or the state its prefill captures,
// it spilled in the first room; counting the idle buffers its start takes as
// room, it started Resident in the second.
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
      spill.limit(spill.held() + 20 * spill.pool.bytesPerPage() - (fits ? 0 : state));
      spill.engine.submit(Spilling::prompt(1, 2, 20, 1));
      spill.runUntil([&] { return std::ranges::find(spill.events.startIds, 1) != spill.events.startIds.end(); },
                     "the next turn did not start");
      const SpillRequestsSnapshot placed = spill.engine.snapshot().spill;
      spill.runUntilIdle();
      const std::string name =
          std::string(fits ? "the next turn in room for its pages" : "the next turn short of room for its pages") +
          (idle ? " beside idle buffers" : "");
      require(spill.events.failedCount == 0 && spill.events.completedCount == 2, name + " failed");
      require(placed.spill == (fits ? 0U : 1U), name + (fits ? " spilled" : " started Resident"));
    }
}

// Placement counts the Resident lanes' projected growth with the prompt's,
// in whole extents past the free pages, as their claims take both from the
// same extents: beside a lane that projects 6 pages more, a prompt that
// fills whole extents spills in room half an extent short of the extents
// both take, and runs Resident in room for them. Counted in pages beside the
// prompt's extents, the lane's growth fit the first room.
void testPlacementCountsTheLanesGrowthInTheSameExtents() {
  constexpr uint64_t shortRoom = 4'600;
  for (const uint64_t room : {shortRoom, uint64_t{4'800}}) {
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
      {"testAPrefillingLaneConvertsItselfAtTheWall", testAPrefillingLaneConvertsItselfAtTheWall},
      {"testALaneConvertsOnlyWithAnIndexRunway", testALaneConvertsOnlyWithAnIndexRunway},
      {"testAGrowthTheHostRefusesDoesNotConvert", testAGrowthTheHostRefusesDoesNotConvert},
      {"testALaneFreeingLessThanItsNeedDoesNotConvert", testALaneFreeingLessThanItsNeedDoesNotConvert},
      {"testALongLaneConvertsItselfBesideAnother", testALongLaneConvertsItselfBesideAnother},
      {"testASharedPrefixCountsForNoConversion", testASharedPrefixCountsForNoConversion},
      {"testAShortLaneDoesNotConvert", testAShortLaneDoesNotConvert},
      {"testALoneResidentWallNamesItsModeAndPhase", testALoneResidentWallNamesItsModeAndPhase},
      {"testALoneStartMemoryCannotHoldIsRefusedAtItsStart", testALoneStartMemoryCannotHoldIsRefusedAtItsStart},
      {"testALoneStartCountsTheNextLanesBuffers", testALoneStartCountsTheNextLanesBuffers},
      {"testALoneStartTakesMemorysBound", testALoneStartTakesMemorysBound},
      {"testAResumeKeepsMemorysBound", testAResumeKeepsMemorysBound},
      {"testALoneStartMemoryCannotHoldTakesTheQuotasBound", testALoneStartMemoryCannotHoldTakesTheQuotasBound},
      {"testALoneStartWithinTheResidentContextStarts", testALoneStartWithinTheResidentContextStarts},
      {"testAStartMemoryCannotHoldBesideALaneWaitsForIt", testAStartMemoryCannotHoldBesideALaneWaitsForIt},
      {"testAStartMemoryNeverHoldsIsDecidedOnceTheLaneEnds", testAStartMemoryNeverHoldsIsDecidedOnceTheLaneEnds},
      {"testAStartMemoryCannotHoldBesideALowerLaneMakesItYield",
       testAStartMemoryCannotHoldBesideALowerLaneMakesItYield},
      {"testALanesGrowthConvertsNoOtherLane", testALanesGrowthConvertsNoOtherLane},
      {"testAStartWaitingForALaneOfItsPriorityLeavesLowerLanesRunning",
       testAStartWaitingForALaneOfItsPriorityLeavesLowerLanesRunning},
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
      {"testAPlacementCountsOnlyTheChainAStartKeeps", testAPlacementCountsOnlyTheChainAStartKeeps},
      {"testAStartRestoringHeldCopiesStartsBesideAQuotaRefusal",
       testAStartRestoringHeldCopiesStartsBesideAQuotaRefusal},
      {"testASpillStartCountsTheCopiesItRestores", testASpillStartCountsTheCopiesItRestores},
      {"testAStartThatCannotSpillStartsResidentWhereMemoryHoldsIt",
       testAStartThatCannotSpillStartsResidentWhereMemoryHoldsIt},
      {"testWriteBudgetRefusalSaysWhenToRetry", testWriteBudgetRefusalSaysWhenToRetry},
      {"testPrefillGroupsFollowIndexPages", testPrefillGroupsFollowIndexPages},
      {"testALaneThatTakesSpilledBlocksAsItPrefillsRunsPrefillGroups",
       testALaneThatTakesSpilledBlocksAsItPrefillsRunsPrefillGroups},
      {"testSpilledFullExtentsLeaveOnceNoPromptGrows", testSpilledFullExtentsLeaveOnceNoPromptGrows},
      {"testSpillMovesThroughTheHostPause", testSpillMovesThroughTheHostPause},
      {"testLandedPagesSpillAtCriticalPressure", testLandedPagesSpillAtCriticalPressure},
      {"testADriverDeniedRefillKeepsWhatTheMovesReleased", testADriverDeniedRefillKeepsWhatTheMovesReleased},
      {"testMovesThatEmptyNoFullExtentWaitAtCriticalPressure", testMovesThatEmptyNoFullExtentWaitAtCriticalPressure},
      {"testASpillingPromptKeepsNoFullExtentAtCriticalPressure",
       testASpillingPromptKeepsNoFullExtentAtCriticalPressure},
      {"testSpillResumeCommitsAgain", testSpillResumeCommitsAgain},
      {"testABoundAtStartAndResumeCountsOnce", testABoundAtStartAndResumeCountsOnce},
      {"testSpillResumeOnAClosedTier", testSpillResumeOnAClosedTier},
      {"testSpillResumeBoundLeavesOutputPastItsAnchor", testSpillResumeBoundLeavesOutputPastItsAnchor},
      {"testSpillResumeInItsPromptHoldsNoAnchor", testSpillResumeInItsPromptHoldsNoAnchor},
      {"testASpillResumeWithNoOutputIsRefusedOutOfQuota", testASpillResumeWithNoOutputIsRefusedOutOfQuota},
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
      {"testASpillLaneKeepsLandedPagesWhileMemoryLasts", testASpillLaneKeepsLandedPagesWhileMemoryLasts},
      {"testASpillLaneAtTheWallReleasesOnlyWhatItLacks", testASpillLaneAtTheWallReleasesOnlyWhatItLacks},
      {"testASpillLaneAtTheWallCutsNoCommand", testASpillLaneAtTheWallCutsNoCommand},
      {"testTheKvPastACheckpointMakesALanesFirstRelease", testTheKvPastACheckpointMakesALanesFirstRelease},
      {"testTwoLanesThatDivergeFromOnePrefixCompleteWhereEachDoesAlone",
       testTwoLanesThatDivergeFromOnePrefixCompleteWhereEachDoesAlone},
      {"testTwoDistinctSpillPromptsAtTheWallCompleteWhereEachDoesAlone",
       testTwoDistinctSpillPromptsAtTheWallCompleteWhereEachDoesAlone},
      {"testALaneAloneOnTheChainAnotherLeftCompletesWhereItDoesFromIdle",
       testALaneAloneOnTheChainAnotherLeftCompletesWhereItDoesFromIdle},
      {"testALoneLaneFailsOnlyPastWhatItHoldsAlone", testALoneLaneFailsOnlyPastWhatItHoldsAlone},
      {"testTwoLanesOnOnePromptAtTheWallNeedNoMoreThanTheLongerAlone",
       testTwoLanesOnOnePromptAtTheWallNeedNoMoreThanTheLongerAlone},
      {"testTheFirstReleaseFundsTheLanesReads", testTheFirstReleaseFundsTheLanesReads},
      {"testAStartReleasesALaneOfItsPriority", testAStartReleasesALaneOfItsPriority},
      {"testAStartNeverReleasesAHigherPriorityLane", testAStartNeverReleasesAHigherPriorityLane},
      {"testACriticalPassPaysForTheFirstReadsOfALaneItReleases", testACriticalPassPaysForTheFirstReadsOfALaneItReleases},
      {"testALoneRequestWaitsForTheCopiesAFinishedSpillLaneLeft", testALoneRequestWaitsForTheCopiesAFinishedSpillLaneLeft},
      {"testCriticalReleasesCleanPagesBeforeAnyLaneYields", testCriticalReleasesCleanPagesBeforeAnyLaneYields},
      {"testACriticalPassReleasesPastTheEnginesLimit", testACriticalPassReleasesPastTheEnginesLimit},
      {"testARefillTheDriverDeniesIsNoOvercount", testARefillTheDriverDeniesIsNoOvercount},
      {"testAWarningDeficitReleasesALanesLandedPages", testAWarningDeficitReleasesALanesLandedPages},
      {"testAHostRefusedGrowthReleasesItsOwnLane", testAHostRefusedGrowthReleasesItsOwnLane},
      {"testAReleaseInPlaceKeepsTheIndexRunway", testAReleaseInPlaceKeepsTheIndexRunway},
      {"testAReleaseInPlaceKeepsTheRunwayTheRoomCannotRefill", testAReleaseInPlaceKeepsTheRunwayTheRoomCannotRefill},
      {"testAGrowthTakesItsPageInPlaceBeforeAReleaseForBytes", testAGrowthTakesItsPageInPlaceBeforeAReleaseForBytes},
      {"testNoReleaseOfALaneBetweenItsPasses", testNoReleaseOfALaneBetweenItsPasses},
      {"testNoReleaseOfAPageALaneBetweenItsPassesHolds", testNoReleaseOfAPageALaneBetweenItsPassesHolds},
      {"testALoneLaneWhoseReleaseCannotPayItsReadsEndsAtItsLength",
       testALoneLaneWhoseReleaseCannotPayItsReadsEndsAtItsLength},
      {"testALoneLaneWaitsForItsLastWritesThenReleasesThem", testALoneLaneWaitsForItsLastWritesThenReleasesThem},
      {"testAReleaseLeavesNoLaneWithoutItsFloor", testAReleaseLeavesNoLaneWithoutItsFloor},
      {"testTwoSpillLanesWithoutIndexPagesPrefillTogether", testTwoSpillLanesWithoutIndexPagesPrefillTogether},
      {"testAStartOnALanesConversationCountsItsChainOnce", testAStartOnALanesConversationCountsItsChainOnce},
      {"testAResidentStartOnALanesPrefixRestoresItWhole", testAResidentStartOnALanesPrefixRestoresItWhole},
      {"testAReleaseOfASharedPrefixFundsEveryLanesReads", testAReleaseOfASharedPrefixFundsEveryLanesReads},
      {"testACountOfReleasesCountsASharedPageOnce", testACountOfReleasesCountsASharedPageOnce},
      {"testACountOfReleasesFallsByAtMostAnExtentAsCopiesLand", testACountOfReleasesFallsByAtMostAnExtentAsCopiesLand},
      {"testALoneSpillWallNamesWhyItsKvGaveNoMoreMemory", testALoneSpillWallNamesWhyItsKvGaveNoMoreMemory},
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
