// The memory ledger's rules as the engine applies them: who a claim may take memory from, what a waiting claim
// holds, and what is no claim at all.

#include "EngineHarness.hpp"
#include "EngineTestSupport.hpp"
#include "FakeElasticPool.hpp"
#include "GovernedHost.hpp"
#include "LandingUnit.hpp"
#include "MemoryPasses.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestRegistry.hpp"
#include "engine/Engine.hpp"
#include "engine/WriteBehind.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;
using splash::test::Events;
using splash::test::Executor;
using splash::test::HarnessOptions;
using splash::test::guardReleases;
using splash::test::idle;
using splash::test::LandingUnit;
using splash::test::request;
using splash::test::runUntilIdle;
using splash::test::Spilling;

namespace {

using splash::test::require;

// Whether the engine lists the request with its KV in `mode`.
bool listedAs(const Engine &engine, uint64_t id, KvMode mode) {
  for (const ActiveRequestSnapshot &active : engine.snapshot().activeRequests)
    if (active.id == id)
      return active.kvMode == mode;
  return false;
}

// A Foreground request of `prompt` whose two tokens each wait for a token
// mask: once its prompt has run it waits for its first, in service beside
// the lanes of a lower priority, which run meanwhile.
EngineRequest maskedForeground(uint64_t id, std::vector<uint32_t> prompt) {
  EngineRequest result = request(id, prompt);
  result.priority = RequestPriority::Foreground;
  result.constraint = ConstraintMode::TokenMask;
  result.maxNewTokens = 2;
  result.deadlineMilliseconds = 1e9;
  return result;
}

// Answers every mask the request asks for until it ends.
void answerMasks(Events &events, Engine &engine, uint64_t id, const std::function<void()> &tick) {
  const std::array<uint32_t, 1> mask{1};
  size_t answered = 0;
  for (uint32_t step = 0; step < 2000 && !events.usage.contains(id); ++step) {
    for (; answered < events.maskRequests.size(); ++answered)
      if (events.maskRequests[answered].first == id)
        engine.provideMask(id, mask);
    tick();
  }
  require(events.usage.contains(id), "the masked request did not finish");
}

// A claim takes nothing a higher-priority lane needs: no conversion of that lane and no pool unit while it serves;
// the lower lane gives way itself.
void testALowerPriorityClaimTakesNothingAHigherPriorityLaneNeeds() {
  {
    // The store holds four units above its floor; beside it the limit
    // leaves the KV eight extents. The two lanes start with thirteen pages
    // each; the Background lane's twentieth would take a ninth extent.
    constexpr uint64_t kUnit = 16 * 4096;
    test::EngineHarness harness(
        {.pages = 128, .memory = test::FakeMemory::Config{}, .limits = {.limitBytes = 5 * kUnit + 32 * 4096 + 4096}});
    test::FakeElasticPool experts("experts", *harness.memory, kUnit, 1, 5, 8, 5);
    harness.model.experts = &experts;
    static_cast<void>(harness.governor->ledger().addPool(experts));
    harness.model.decodeFinishes = false;
    EngineRequest background = harness.request(1, std::vector<uint32_t>(400, 1));
    background.priority = RequestPriority::Background;
    background.maxNewTokens = 400;
    background.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(background));
    harness.runUntil([&] { return harness.events.outputs.contains(1); }, "the Background lane did not decode");
    harness.engine.submit(maskedForeground(2, std::vector<uint32_t>(400, 2)));
    harness.runUntil([&] { return harness.engine.snapshot().resourceSuspensions != 0 || experts.shrinks != 0; },
                     "the Background lane's growth was not refused beside the waiting Foreground lane");
    require(experts.shrinks == 0 && !harness.events.outputs.contains(2) && harness.events.maskRequests.size() == 1,
            "a Background lane's growth took a unit of the expert store beside a Foreground lane in service: "
            "store units " + std::to_string(experts.units()));
    answerMasks(harness.events, harness.engine, 2, [&] {
      if (!harness.tick())
        harness.advanceTo(harness.now() + 1.0);
    });
    harness.runUntilIdle();
    require(harness.events.completedCount == 2 && harness.events.failedCount == 0 &&
                harness.events.outputs[1].size() == 400 && experts.shrinks == 0,
            "the Background request did not finish in the room the Foreground one left: completed " +
                std::to_string(harness.events.completedCount) + ", failed " + std::to_string(harness.events.failedCount) +
                ", outputs " + std::to_string(harness.events.outputs[1].size()) + ", store shrinks " +
                std::to_string(experts.shrinks) +
                (harness.events.failureDetails.empty() ? "" : " (" + harness.events.failureDetails.front().first + ")"));
  }
  {
    // A Background lane of two pages decodes; the Foreground lane of 22
    // pages waits for its mask beside it, and no extent is allocated past
    // what the two hold.
    Spilling spill(1024, false);
    EngineRequest background = request(1, std::vector<uint32_t>(64, 1));
    background.priority = RequestPriority::Background;
    background.maxNewTokens = 400;
    background.deadlineMilliseconds = 1e9;
    spill.engine.submit(std::move(background));
    spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the Background lane did not decode");
    spill.engine.submit(maskedForeground(2, std::vector<uint32_t>(680, 2)));
    spill.runUntil([&] { return spill.events.maskRequests.size() == 1; },
                   "the Foreground lane did not wait for its mask");
    require(listedAs(spill.engine, 2, KvMode::Resident), "the Foreground lane is not Resident");
    spill.storage.budgetPages = static_cast<uint32_t>(spill.storage.allocatedBytes() / 100);
    spill.runUntil(
        [&] {
          return spill.engine.snapshot().resourceSuspensions != 0 || listedAs(spill.engine, 1, KvMode::Spill) ||
                 listedAs(spill.engine, 2, KvMode::Spill);
        },
        "the Background lane's growth was not refused beside the waiting Foreground lane");
    require(listedAs(spill.engine, 2, KvMode::Resident) && spill.events.failedCount == 0,
            "a Background lane's growth converted the Foreground lane beside it to Spill");
    spill.storage.budgetPages = std::numeric_limits<uint32_t>::max();
    answerMasks(spill.events, spill.engine, 2, [&] {
      static_cast<void>(spill.engine.tick(spill.now++));
      spill.tier.complete();
    });
    spill.runUntilIdle();
    require(spill.events.completedCount == 2 && spill.events.failedCount == 0,
            "the Background request did not finish after the Foreground one");
  }
}

// An engine whose prompts run in passes (test::commandShape), its `pages`
// KV pages of 16 bytes and the model's states of 64 allocated through
// FakeMemory under a governor of `limitBytes` (GovernedHost); the prefill
// commands' first lane and pass, in order.
struct PassesUnderALimit {
  explicit PassesUnderALimit(uint64_t limitBytes, uint32_t pages = 512)
      : host({.limitBytes = limitBytes}), storage(pages, 16, 4) {
    model.memory = &host.memory;
    model.admission = host.governor.allocationAdmission();
    model.decodeFinishes = false;
    guardReleases(storage, engine);
    model.time(test::passingTarget());
    model.commandTimes = [this](const BatchPlan &plan, std::span<const ModelBatchItem>) {
      if (plan.kind == WorkKind::Prefill)
        prefills.emplace_back(plan.items.front().requestId, plan.pass);
      return test::CommandTimes{};
    };
  }
  void submit(uint64_t id, uint32_t tokens, RequestPriority priority) {
    EngineRequest value = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.priority = priority;
    value.deadlineMilliseconds = 1e9;
    engine.submit(std::move(value));
  }
  void runUntil(const std::function<bool()> &done, const char *message) {
    test::tickUntil(engine, now, done, message);
  }

  test::GovernedHost host;
  test::TestKvStorage storage;
  KvPool pool{(host.through(storage), storage), 0};
  engine::Cache cache{pool, nullptr, nullptr};
  Executor model;
  bool shaped = (test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true)(model), true);
  Events events;
  engine::Engine engine{host.connect({}), cache, model, events};
  std::vector<std::pair<uint64_t, uint32_t>> prefills;
  double now = 1;
};

// The Index pages a Spill lane's landed copies move into are claimed at that lane's priority, so beside a
// higher-priority lane they take no pool unit.
void testASpillLanesIndexPagesAreClaimedAtItsPriority() {
  Spilling spill(1024);
  std::vector<RequestPriority> claimed;
  spill.storage.growthAllowed = [&](uint32_t) {
    for (const auto &[_, state] : spill.host.governor.ledger().claims())
      if (!state.spec.requestId && state.spec.cls == ClaimClass::InService &&
          state.spec.timing == NeedTiming::Immediate)
        claimed.push_back(state.spec.priority);
    return true;
  };
  EngineRequest background = Spilling::prompt(1, 1, 20, 8);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntilIdle();
  require(spill.events.completedCount == 1 && spill.cache.spillSnapshot().spilledPages != 0,
          "the Background Spill prompt did not run and spill");
  require(!claimed.empty() && std::ranges::all_of(claimed, [](RequestPriority priority) {
            return priority == RequestPriority::Background;
          }),
          "the Index pages of a Background Spill lane were claimed at another priority: " +
              std::to_string(std::ranges::count(claimed, RequestPriority::Foreground)) + " of " +
              std::to_string(claimed.size()) + " allocations at Foreground");
}

// A prompt whose work supersedes the command between its passes takes that command's state buffers rather than
// suspend a lane; a lane of it that ended meanwhile is settled after the take.
void testSupersedingClaimTakesThePassCommand() {
  for (const bool ended : {false, true}) {
    // Two Background prompts of five pages share a command in passes: their
    // KV takes three extents (192 bytes) and their replay points two states
    // (128). The Foreground prompt's first extent passes the limit beside
    // them; without the states, its KV and its replay point fit.
    PassesUnderALimit passes(352);
    passes.submit(1, 129, RequestPriority::Background);
    passes.submit(2, 129, RequestPriority::Background);
    passes.runUntil([&] { return !passes.prefills.empty() && !passes.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
    require(passes.prefills.back().second == 0 && passes.model.heldStateBytes == 128,
            "the Background prompts' command did not hold its states between passes");
    if (ended)
      passes.engine.cancel(2);
    passes.submit(3, 129, RequestPriority::Foreground);
    passes.runUntil([&] { return passes.events.usage.contains(3); }, "the Foreground prompt did not finish");
    passes.runUntil([&] { return idle(passes.engine); }, "the requests did not end");
    const EngineSnapshot counts = passes.engine.snapshot();
    require(counts.resourceSuspensions == 0 && counts.prioritySuspensions == 0 && counts.cancelled == ended &&
                passes.events.completedCount == 3 && passes.events.failedCount == 0,
            std::string("a superseding prompt did not take the memory of the command between its passes") +
                (ended ? " with a lane that ended between them" : "") + ": suspensions " +
                std::to_string(counts.resourceSuspensions) + "+" + std::to_string(counts.prioritySuspensions) +
                ", cancelled " + std::to_string(counts.cancelled) + ", completed " +
                std::to_string(passes.events.completedCount) + ", failed " +
                std::to_string(passes.events.failedCount));
  }
}

// A claim that takes the command between its passes outside any plan's preparation, as any claim of in-service
// work may, takes no lane with it: the lane of it that was cancelled ends once, at the next tick, and gives back
// its claim and its KV.
void testALaneOfACommandTakenOutsidePreparationEndsAtTheNextTick() {
  PassesUnderALimit passes(uint64_t{1} << 30);
  passes.submit(1, 129, RequestPriority::Background);
  passes.submit(2, 129, RequestPriority::Background);
  passes.runUntil([&] { return !passes.prefills.empty() && !passes.engine.commandInFlight(); },
                  "the Background prompts did not run a pass");
  require(passes.prefills.back().second == 0 && passes.model.heldStateBytes == 128,
          "the Background prompts' command is not between its passes");
  passes.engine.cancel(2);
  require(!passes.events.usage.contains(2), "the lane cancelled between passes ended before its command did");
  const uint64_t abandoned = passes.engine.snapshot().scheduler.abandonedPassCommands;
  const size_t lanePages = passes.cache.pageTable(2).pages.size();
  const uint32_t freePages = passes.pool.freePageCount(kv::PageForm::Full);
  MemoryLedger &ledger = passes.host.governor.ledger();
  const ClaimId claim = ledger.open({.cls = ClaimClass::InService, .supersedes = true});
  const FreeResult freed = passes.host.governor.free(claim, {.stateBuffers = 2});
  ledger.close(claim);
  require(freed.outcome == FreeOutcome::Met &&
              passes.engine.snapshot().scheduler.abandonedPassCommands == abandoned + 1,
          "the claim did not take the command between its passes");
  require(!passes.events.usage.contains(2), "the claim's take ended the lane inside the ledger's free");
  static_cast<void>(passes.engine.tick(passes.now++));
  const auto holdsClaim = [&](uint64_t id) {
    return std::ranges::any_of(ledger.claims(), [&](const auto &entry) { return entry.second.spec.requestId == id; });
  };
  require(passes.events.usage.contains(2) && passes.engine.snapshot().cancelled == 1 && !holdsClaim(2) &&
              lanePages != 0 && passes.pool.freePageCount(kv::PageForm::Full) >= freePages + lanePages,
          "the lane cancelled in the command a claim took did not end at the next tick with its claim and KV");
  passes.runUntil([&] { return idle(passes.engine); }, "the requests did not end");
  const EngineSnapshot counts = passes.engine.snapshot();
  require(counts.cancelled == 1 && passes.events.completedCount == 2 && passes.events.failedCount == 0 &&
              counts.scheduler.abandonedPassCommands == abandoned + 1,
          "the cancelled lane did not end exactly once beside the other prompt's completion: cancelled " +
              std::to_string(counts.cancelled) + ", completed " + std::to_string(passes.events.completedCount) +
              ", abandoned " + std::to_string(counts.scheduler.abandonedPassCommands - abandoned));
}

// A lane cancelled between its command's passes ends at once though a higher-priority decode holds the command off;
// the command gives way whole to the decode.
void testACancelBetweenPassesEndsBesideAHigherPrioritysDecode() {
  PassesUnderALimit passes(uint64_t{1} << 30);
  EngineRequest foreground = request(3, std::vector<uint32_t>(64, 3));
  foreground.priority = RequestPriority::Foreground;
  foreground.constraint = ConstraintMode::TokenMask;
  foreground.maxNewTokens = 1000;
  foreground.deadlineMilliseconds = 1e9;
  passes.engine.submit(std::move(foreground));
  passes.runUntil([&] { return !passes.events.maskRequests.empty(); },
                  "the Foreground lane did not ask for its first mask");
  passes.submit(1, 129, RequestPriority::Background);
  passes.submit(2, 129, RequestPriority::Background);
  passes.runUntil(
      [&] {
        return std::ranges::any_of(passes.prefills, [](const auto &prefill) { return prefill.first != 3; }) &&
               !passes.engine.commandInFlight();
      },
      "the Background prompts did not run a pass");
  require(passes.prefills.back().second == 0 && passes.prefills.back().first != 3,
          "the Background prompts' command is not between its passes");
  const std::array<uint32_t, 1> mask{1};
  size_t answered = 0;
  const auto answer = [&] {
    for (; answered < passes.events.maskRequests.size(); ++answered)
      if (passes.events.maskRequests[answered].first == 3)
        passes.engine.provideMask(3, mask);
  };
  answer();
  passes.engine.cancel(2);
  for (uint32_t step = 0; step < 50 && !passes.events.usage.contains(2); ++step) {
    answer();
    static_cast<void>(passes.engine.tick(passes.now++));
  }
  require(passes.events.usage.contains(2) && !passes.events.usage.contains(3),
          "the cancelled lane did not end while the Foreground lane decoded");
}

// Beside a decode of the command's own priority, a lane cancelled between passes ends at the command's last pass
// and the other lanes keep the passes they ran.
void testACancelBetweenPassesBesideItsOwnPrioritysDecodeKeepsThePasses() {
  PassesUnderALimit passes(uint64_t{1} << 30);
  // Each prefill pass's lanes, pass and passes.
  struct Pass final {
    std::vector<uint64_t> lanes;
    uint32_t pass = 0;
    uint32_t passes = 0;
  };
  std::vector<Pass> ran;
  passes.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill) {
      passes.prefills.emplace_back(plan.items.front().requestId, plan.pass);
      Pass &unit = ran.emplace_back(Pass{.pass = plan.pass, .passes = plan.passes});
      for (const BatchItem &item : plan.items)
        unit.lanes.push_back(item.requestId);
    }
    return test::CommandTimes{};
  };
  const auto holds = [](const Pass &unit, uint64_t id) {
    return std::ranges::find(unit.lanes, id) != unit.lanes.end();
  };
  const auto passesOf = [&](uint64_t id) {
    std::string result;
    for (const Pass &unit : ran)
      if (holds(unit, id))
        result += (result.empty() ? "" : ",") + std::to_string(unit.pass);
    return result;
  };
  EngineRequest decoder = request(3, std::vector<uint32_t>(64, 3));
  decoder.maxNewTokens = 1000;
  decoder.deadlineMilliseconds = 1e9;
  passes.engine.submit(std::move(decoder));
  passes.runUntil([&] { return passes.events.outputs.contains(3); }, "the Normal lane did not decode");
  passes.submit(1, 129, RequestPriority::Normal);
  passes.submit(2, 129, RequestPriority::Normal);
  passes.runUntil(
      [&] {
        return std::ranges::any_of(ran, [&](const Pass &unit) { return !holds(unit, 3); }) &&
               !passes.engine.commandInFlight();
      },
      "the Normal prompts did not run a pass");
  require(ran.back().pass == 0 && ran.back().passes > 1 && holds(ran.back(), 1) && holds(ran.back(), 2),
          "the Normal prompts do not share a command between its passes");
  const uint32_t commandPasses = ran.back().passes;
  passes.engine.cancel(2);
  for (uint32_t step = 0; step < 200 && !passes.events.usage.contains(2); ++step)
    static_cast<void>(passes.engine.tick(passes.now++));
  const Pass ending = ran.back();
  const uint64_t abandonedAtEnd = passes.engine.snapshot().scheduler.abandonedPassCommands;
  passes.runUntil([&] { return passes.events.outputs.contains(1); }, "the prompt left did not decode");
  std::string expected;
  for (uint32_t pass = 0; pass < commandPasses; ++pass)
    expected += (pass ? "," : "") + std::to_string(pass);
  require(passes.events.usage.contains(2) && passes.events.outputs.contains(3) && abandonedAtEnd == 0 &&
              ending.pass + 1 == commandPasses && holds(ending, 2) && passesOf(1) == expected &&
              passes.engine.snapshot().scheduler.abandonedPassCommands == 0,
          "a lane cancelled between the passes beside a decode of its command's priority did not end at the "
          "command's last pass, its other lane keeping its passes: abandoned commands " +
              std::to_string(passes.engine.snapshot().scheduler.abandonedPassCommands) +
              ", the cancelled lane ended after pass " + std::to_string(ending.pass) + " of " +
              std::to_string(ending.passes) + (holds(ending, 2) ? " holding it" : " without it") +
              ", the other lane ran passes " + passesOf(1) + " (expected " + expected + ")");
}

// A start the host refused beside a resident lane is freed for by the host's passes, from a rebuild-only pool and
// never the expert store, and then runs.
void testAHostRefusedStartIsFreedForByThePass() {
  test::EngineHarness harness({.pages = 64, .memory = test::FakeMemory::Config{}});
  test::FakeElasticPool pool("vision_arena", *harness.memory, 64 * 1024, 1, 3, 4, 3);
  pool.rebuildOnly = true;
  static_cast<void>(harness.governor->ledger().addPool(pool));
  harness.model.laneFootprintBytes = 64 * 1024;
  harness.engine.submit(harness.request(1, std::vector<uint32_t>(513, 1)));
  harness.runUntilIdle();
  harness.engine.submit(maskedForeground(2, std::vector<uint32_t>(64, 2)));
  harness.runUntil([&] { return harness.events.maskRequests.size() == 1; },
                   "the Foreground lane did not wait for its mask");
  test::FakeMemory &memory = *harness.memory;
  memory.setHost(memory.hostBytes() - memory.total() - test::GovernorLimits{}.hostSpareReserveBytes -
                     kHostMarginBytes - 48 * 1024,
                 MemoryPressure::Normal);
  harness.engine.submit(harness.request(3, std::vector<uint32_t>(64, 3)));
  double pass = harness.now();
  for (const double end = harness.now() + 3000.0; harness.now() < end && !harness.events.outputs.contains(3);) {
    if (harness.now() >= pass) {
      static_cast<void>(harness.governor->controlPass(harness.now()));
      pass += 500.0;
    }
    if (!harness.tick())
      harness.advanceTo(harness.now() + 1.0);
  }
  const MemoryGovernorSnapshot governor = harness.governor->snapshot();
  require(harness.events.outputs.contains(3),
          "a start the host refused beside a resident lane was never freed for by the host's passes: begins " +
              std::to_string(harness.model.beginAttempts) + ", host headroom " +
              std::to_string(governor.hostHeadroomBytes) + ", waiting claims " +
              std::to_string(governor.claims.waiting) + ", states " + std::to_string(memory.held("states")) +
              ", kv " + std::to_string(memory.held("kv")) + ", host refused " +
              std::to_string(harness.governor->ledger().hostRefusedWaitingBytes()));
}

// A request that has not started ranks by the work its start restores: a cold start neither outranks a lane in
// prefill of its priority nor holds room that lane waits for.
void testAStartRanksByTheWorkItHolds() {
  test::EngineHarness harness({.pages = 256, .memory = test::FakeMemory::Config{}});
  EngineRequest lane = request(1, std::vector<uint32_t>(4000, 1));
  lane.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(lane));
  harness.runUntil(
      [&] {
        const EngineSnapshot counts = harness.engine.snapshot();
        return !counts.activeRequests.empty() && counts.activeRequests.front().promptProcessed != 0 &&
               !harness.engine.commandInFlight();
      },
      "the long prompt did not run its first command");
  const auto rank = [&](uint64_t id) -> std::optional<ClaimRank> {
    for (const auto &[_, state] : harness.governor->ledger().claims())
      if (state.spec.requestId == id)
        return state.spec.rank;
    return std::nullopt;
  };
  std::optional<ClaimRank> started;
  harness.model.beginObserver = [&] {
    if (harness.model.lastBeginId == 2 && !started)
      started = rank(2);
  };
  EngineRequest start = request(2, std::vector<uint32_t>(3000, 2));
  start.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(start));
  harness.runUntil([&] { return started.has_value(); }, "the start was never attempted");
  require(started->completedTokens == 0,
          "a cold start's claim counted " + std::to_string(started->completedTokens) +
              " tokens of its prompt as completed work");
}

// Five cached one-page blocks under states on disk in a pool the budget
// holds to two extents, three pages free: evicting a block starts a
// demotion whose page comes back once its copy lands (tier.complete()).
struct LandingPages {
  LandingPages() {
    storage.budgetPages = 8;
    tier.transferLimit = 8;
    model.decodeFinishes = false;
    guardReleases(storage, engine);
    test::cacheBlocksUnderDiskStates(cache, 5);
    require(pool.freePageCount() == 3 && tier.demotions == 0, "the fixture's pages are off");
  }
  // A decoding lane of two pages whose fourth page waits for a demotion,
  // its claim holding the page that lands.
  void waitingDecoder(uint64_t id) {
    EngineRequest waiter = request(id, std::vector<uint32_t>(64, static_cast<uint32_t>(id)));
    waiter.maxNewTokens = 64;
    waiter.deadlineMilliseconds = 1e9;
    engine.submit(std::move(waiter));
    test::tickUntil(engine, now, [&] { return tier.demotions != 0 && !engine.commandInFlight(); },
                    "the decoding lane's growth did not wait for a demotion");
    require(pool.freePageCount() == 0 && governor().ledger().claimWaiting() && events.outputs[id].size() < 64,
            "the decoding lane does not wait on its claim for the page landing");
  }
  [[nodiscard]] MemoryGovernor &governor() const { return *config.governor; }
  // The KV pages the request holds, none before its lane begins.
  [[nodiscard]] size_t pages(uint64_t id) const {
    for (const ActiveRequestSnapshot &active : engine.snapshot().activeRequests)
      if (active.id == id && active.phase != Phase::Queued && active.phase != Phase::WaitingResources)
        return cache.pageTable(id).pages.size();
    return 0;
  }
  // Whether the request is listed as waiting for resources (suspended).
  [[nodiscard]] bool suspended(uint64_t id) const {
    for (const ActiveRequestSnapshot &active : engine.snapshot().activeRequests)
      if (active.id == id)
        return active.phase == Phase::WaitingResources;
    return false;
  }

  test::TestKvStorage storage{16, 4096, 4};
  KvPool pool{storage, 0};
  test::TestKvTier tier;
  engine::Cache cache{pool, &tier, test::ampleDiskBudget()};
  Executor model;
  Events events;
  const EngineConfig config = test::engineConfig({.maxContext = 102400});
  engine::Engine engine{config, cache, model, events};
  double now = 1;
};

// Memory on its way to a waiting claim lands for it: a claim ranked below cannot take it while it waits, one ranked
// above can, so prefills cannot starve waiting decoders.
void testLandingPaysItsOwnerFirst() {
  {
    // A prefill ranks below a decoding lane of its priority: arriving while
    // the decoder waits for its landing, it takes none of it, and the
    // decoder never gives way.
    LandingPages landing;
    landing.waitingDecoder(1);
    landing.engine.submit(request(2, std::vector<uint32_t>(31, 2)));
    bool decoderSuspended = false;
    for (uint32_t step = 0; step < 400 && !idle(landing.engine); ++step) {
      static_cast<void>(landing.engine.tick(landing.now++));
      landing.tier.complete();
      decoderSuspended = decoderSuspended || landing.suspended(1);
      require(landing.pages(2) == 0 || landing.pages(1) >= 4 || landing.pool.freePageCount() != 0,
              "a prefill ranked below the waiting decoder took the page that landed for it");
    }
    require(idle(landing.engine) && !decoderSuspended && landing.events.failedCount == 0 &&
                landing.events.outputs[1].size() == 64 && landing.events.usage.contains(2),
            "the waiting decoder gave way to a prefill ranked below it, or the requests did not end");
  }
  {
    // A Foreground prompt ranks above a Normal decoder that waits: the room
    // the decoder's claim holds does not hold it back, and it runs first.
    LandingPages landing;
    landing.waitingDecoder(1);
    const size_t decoded = landing.events.outputs[1].size();
    EngineRequest foreground = request(2, std::vector<uint32_t>(31, 2));
    foreground.priority = RequestPriority::Foreground;
    foreground.deadlineMilliseconds = 1e9;
    landing.engine.submit(std::move(foreground));
    for (uint32_t step = 0; step < 400 && !landing.events.usage.contains(2); ++step) {
      static_cast<void>(landing.engine.tick(landing.now++));
      landing.tier.complete();
    }
    require(landing.events.usage.contains(2) && landing.events.outputs[1].size() == decoded,
            "a Foreground prompt waited behind the room a Normal decoder's claim holds");
    for (uint32_t step = 0; step < 400 && !idle(landing.engine); ++step) {
      static_cast<void>(landing.engine.tick(landing.now++));
      landing.tier.complete();
    }
    require(idle(landing.engine) && landing.events.failedCount == 0 && landing.events.outputs[1].size() == 64,
            "the decoder did not finish after the Foreground prompt");
  }
}

// A start that waits for the SSD cache's quota is no memory claim: nothing holds room for it, and the expert store
// regrows after the quiet period.
void testQuotaWaitIsNotAMemoryClaim() {
  Spilling spill(10);
  test::FakeElasticPool experts("experts", spill.host.memory, 4096, 1, 4, 8, 3);
  spill.model.experts = &experts;
  static_cast<void>(spill.host.governor.ledger().addPool(experts));
  // The first request commits 8 of the 10 slots; the second needs 4 for its
  // prompt and a page of output, and waits for them.
  spill.engine.submit(Spilling::prompt(1, 1, 2, 191));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first request did not decode");
  spill.engine.submit(Spilling::prompt(2, 2, 3, 200));
  spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1; },
                 "the second request did not wait for the quota");
  for (int step = 0; step < 8; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
    const MemoryGovernorSnapshot memory = spill.host.governor.snapshot();
    require(spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1 &&
                !spill.host.governor.ledger().claimWaiting() && memory.claims.waiting == 0,
            "a start waiting for the SSD cache's quota waits for memory");
    if (!spill.engine.commandInFlight())
      static_cast<void>(spill.host.governor.controlPass(spill.now));
  }
  static_cast<void>(spill.host.governor.controlPass(spill.now + 1000.0 * kRegrowQuietSeconds));
  require(experts.shrinks == 0 && experts.units() == 4,
          "the expert store did not grow back while only the quota kept a start waiting: units " +
              std::to_string(experts.units()));
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.failedCount == 0,
          "the start did not run once the quota it waited for came back");
}

// A state only on disk at `prompt`'s 64 tokens, over KV the cache holds in
// memory, built in a start's Floor claim as a start that adopts a
// persistent cache builds it.
void cacheDiskState(test::EngineHarness &harness, const std::vector<uint32_t> &prompt) {
  const test::ClaimScope start(*harness.governor, ClaimClass::Floor);
  harness.cache.beginRequest(999);
  require(harness.cache.ensureTokens(999, 64).granted(), "the disk state's KV was not allocated");
  test::demoteState(harness.cache, test::publishBlocks(harness.cache, 999, prompt, 64));
  harness.cache.endRequest(999);
}

// A state restored from disk takes its RAM copy in an Optional claim of its request, between commands, so the next
// request on the prompt reads no disk.
void testARestoredStateIsPromotedInAClaim() {
  const std::vector<uint32_t> prompt(65, 17);
  {
    test::EngineHarness harness({.memory = test::FakeMemory::Config{}});
    cacheDiskState(harness, prompt);
    harness.model.restoreControl->ready = true;
    harness.engine.submit(harness.request(1, prompt));
    harness.runUntilIdle();
    harness.engine.submit(harness.request(2, prompt));
    harness.runUntilIdle();
    const StateCacheSnapshot states = harness.cache.snapshot().stateCache;
    require(harness.model.diskReads == 1 && states.promotions == 1 && harness.model.heldStateBytes == 64 &&
                harness.events.completedCount == 2 && harness.events.failedCount == 0,
            "a restored state was not promoted into memory its claim was admitted: disk reads " +
                std::to_string(harness.model.diskReads) + ", promotions " + std::to_string(states.promotions) +
                ", state bytes " + std::to_string(harness.model.heldStateBytes));
  }
  {
    test::EngineHarness harness({.memory = test::FakeMemory::Config{}});
    cacheDiskState(harness, prompt);
    const std::shared_ptr<std::atomic<bool>> release = harness.model.holdCommands();
    harness.engine.submit(harness.request(1, prompt));
    harness.engine.submit(harness.request(2, std::vector<uint32_t>(31, 2)));
    harness.runUntil([&] { return harness.engine.commandInFlight() && harness.model.diskReads == 1; },
                     "the restore did not start beside a command in flight", 100);
    harness.memory->hold("limit", harness.governor->snapshot().headroomBytes - 32);
    harness.model.restoreControl->ready = true;
    static_cast<void>(harness.tick());
    const StateCacheSnapshot landed = harness.cache.snapshot().stateCache;
    require(harness.engine.commandInFlight() && landed.promotions == 0 && landed.promotionsSkipped == 0,
            "a restore that landed beside a command in flight gave up its copy before the command ended");
    harness.memory->release("limit", harness.memory->held("limit"));
    *release = true;
    harness.runUntilIdle();
    const StateCacheSnapshot states = harness.cache.snapshot().stateCache;
    require(harness.events.completedCount == 2 && harness.events.failedCount == 0 && states.promotions == 1 &&
                states.promotionsSkipped == 0,
            "a restore that landed beside a command in flight was not promoted once the command ended");
  }
}

// A lane below a waiting claim, refused an extent by the host, takes in place the pages it lacks beyond those held
// for the claim, and keeps its lane.
void testALaneBelowAWaitingClaimTakesWhatItLacksBeyondTheHeldPages() {
  test::EngineHarness harness({.memory = test::FakeMemory::Config{}});
  constexpr uint32_t page = KvCache::pageTokens;
  harness.engine.submit(harness.request(1, std::vector<uint32_t>(8 * page + 1, 7)));
  harness.runUntilIdle();
  harness.model.decodeFinishes = false;
  EngineRequest lane = harness.request(2, std::vector<uint32_t>(page + 1, 3));
  lane.maxNewTokens = 4 * page;
  harness.engine.submit(std::move(lane));
  harness.runUntil([&] { return harness.events.outputs.contains(2) && !harness.engine.commandInFlight(); },
                   "the lane did not decode");
  // The claim takes the landing alone: the cached chain, which costs
  // nothing at the harness's rates, is set aside meanwhile.
  MemoryLedger &ledger = harness.governor->ledger();
  LandingUnit landing(8 * harness.cache.bytesPerPage(kv::PageForm::Full));
  ledger.removeSource(harness.cache);
  ledger.addSource(landing);
  const ClaimId waiter = ledger.open({.cls = ClaimClass::InService, .rank = {.priority = 2}});
  const ClaimId below = ledger.open({.cls = ClaimClass::InService, .rank = {.priority = 1}});
  require(ledger.free(waiter, {.bytes = 8 * harness.cache.bytesPerPage(kv::PageForm::Full)}, 0, MemoryPressure::Normal)
                  .outcome == FreeOutcome::Pending &&
              ledger.claim(waiter).waiting && ledger.heldPages(below, 0)[0] == 8,
          "the Foreground claim did not wait for 8 pages' bytes");
  ledger.close(below);
  ledger.removeSource(landing);
  ledger.addSource(harness.cache);
  harness.memory->setHost(harness.memory->hostBytes() - (2ULL << 30) - kHostMarginBytes / 2, MemoryPressure::Normal);
  harness.runUntil(
      [&] {
        return harness.events.outputs.at(2).size() >= 2 * page || harness.engine.snapshot().resourceSuspensions;
      },
      "the lane neither decoded past its next page nor was suspended");
  const uint64_t suspensions = harness.engine.snapshot().resourceSuspensions;
  const uint32_t free = harness.pool.freePageCount(kv::PageForm::Full);
  require(suspensions == 0 && free >= 8,
          "a lane below a waiting claim refused an extent by the host did not take what it lacked beyond the held "
          "pages, or took them: " +
              std::to_string(suspensions) + " suspensions, " + std::to_string(free) + " free pages");
  ledger.close(waiter);
  harness.memory->setHost(test::FakeMemory::Config{}.otherBytes, MemoryPressure::Normal);
  harness.runUntilIdle();
  require(harness.events.completedCount == 2 && harness.events.failedCount == 0, "the lane did not finish");
}

// A command that captures a checkpoint and the prompt's replay point reserves both in the replay point's claim, and
// takes the checkpoint's buffers only when refused.
void testAReplayPointBesideACheckpointIsReservedInAClaim() {
  PassesUnderALimit passes(1ULL << 30);
  passes.submit(1, 6145, RequestPriority::Normal);
  passes.runUntil([&] { return idle(passes.engine); }, "the prompt did not finish");
  const EngineSnapshot counts = passes.engine.snapshot();
  require(test::capturedStates(passes.model, 1) ==
                  std::vector<std::pair<uint32_t, uint64_t>>{{4096, 6145}, {6144, 6145}} &&
              counts.checkpointPublications == 1 && counts.replayStatePublications == 1 &&
              passes.model.heldStateBytes == 128 && passes.events.usage.contains(1) &&
              passes.events.failedCount == 0,
          "the command did not capture the checkpoint and the replay point in buffers of their own: "
          "checkpoints " + std::to_string(counts.checkpointPublications) + ", replay points " +
              std::to_string(counts.replayStatePublications) + ", state bytes " +
              std::to_string(passes.model.heldStateBytes));
}

// Engines on one governor offer a shared cache to the ledger once, and an engine that ended leaves nothing of its
// own.
void testEnginesOnOneGovernorOfferTheCacheOnce() {
  test::GovernedHost host;
  test::TestKvStorage storage{64, 100, 4};
  KvPool pool{storage, 0};
  engine::Cache cache{pool, nullptr, nullptr};
  Executor model;
  Events events;
  // A cached chain of two extents, one of which the host's claim may take.
  const std::vector<uint32_t> prompt(8 * KvCache::pageTokens, 9);
  cache.beginRequest(900);
  require(cache.ensureTokens(900, prompt.size()).granted(), "the cached chain's KV was not allocated");
  static_cast<void>(test::publishBlocks(cache, 900, prompt, static_cast<uint32_t>(prompt.size())));
  cache.endRequest(900);
  const ClaimSpec spec{.cls = ClaimClass::Host};
  const auto found = [&] { return host.governor.available(spec, {.bytes = std::numeric_limits<uint64_t>::max()}); };
  engine::Engine first{host.connect({}), cache, model, events};
  const uint64_t once = found();
  require(once != 0, "the cache offered the host nothing");
  {
    engine::Engine second{host.connect({}), cache, model, events};
    require(found() == once, "a second engine over the cache made its units count twice: " + std::to_string(found()) +
                                 " bytes for " + std::to_string(once));
  }
  require(found() == once, "an engine that ended left what it registered with the ledger");
}

// A request's TTFT splits into the commands that held it, each interval once by precedence; the parts are what
// those commands ran and never exceed the TTFT.
void testTtftPartsSumToTtft() {
  test::EngineHarness harness({.pages = 256, .timing = &test::kM6FlashNextQ2});
  harness.model.decodeFinishes = false;
  EngineRequest decoder = harness.request(1, std::vector<uint32_t>(40, 1));
  decoder.maxNewTokens = 64;
  decoder.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(decoder));
  harness.runUntil([&] { return harness.events.outputs.contains(1); }, "the decoder did not decode");
  const double arrived = harness.now();
  for (const uint64_t id : {2, 3}) {
    EngineRequest prompt = harness.request(id, std::vector<uint32_t>(3000, static_cast<uint32_t>(id)));
    prompt.maxNewTokens = 2;
    prompt.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(prompt));
  }
  harness.runUntil([&] { return harness.events.tokenTimes.contains(2) && harness.events.tokenTimes.contains(3); },
                   "the prompts did not answer");
  for (const uint64_t id : {2, 3}) {
    const double first = harness.events.tokenTimes.at(id).front();
    TtftParts expected{};
    for (const test::EngineHarness::Command &command : harness.commands()) {
      const double from = std::max(command.startMilliseconds, arrived);
      const double to = std::min(command.startMilliseconds + command.wallMilliseconds, first);
      if (to <= from)
        continue;
      const bool own = std::ranges::find(command.lanes, id) != command.lanes.end();
      const TtftPart part = command.kind == WorkKind::Decode ? TtftPart::Decode
                            : own                            ? TtftPart::OwnPrefill
                                                             : TtftPart::OtherPrefill;
      expected[static_cast<size_t>(part)] += to - from;
    }
    const TtftParts &parts = harness.events.ttftParts.at(id);
    double counted = 0.0;
    for (size_t part = 0; part < kTtftParts; ++part) {
      require(std::abs(parts[part] - expected[part]) < 1e-6,
              "request " + std::to_string(id) + "'s TTFT part " + std::to_string(part) + " was " +
                  std::to_string(parts[part]) + " ms, not the " + std::to_string(expected[part]) +
                  " ms its commands ran");
      counted += parts[part];
    }
    require(parts[static_cast<size_t>(TtftPart::OwnPrefill)] > 0.0 &&
                parts[static_cast<size_t>(TtftPart::Decode)] > 0.0 && counted <= first - arrived + 1e-6,
            "the TTFT parts did not cover the commands that held the prompt within its TTFT");
  }
}

// Two prompts whose claims wait for `landing`: the first runs a command, then the second, of fewer rows than the first
// has left, leads, its plans leaving the first out.
struct LeftOutLane final {
  test::EngineHarness harness{{.pages = 256, .memory = test::FakeMemory::Config{}, .timing = &test::kM6FlashNextQ2}};
  MemoryLedger &ledger = harness.governor->ledger();
  LandingUnit landing{1ULL << 30};

  LeftOutLane() {
    EngineRequest first = harness.request(1, std::vector<uint32_t>(5048, 1));
    first.maxNewTokens = 2;
    first.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(first));
    harness.runUntil([&] { return harness.commands().size() == 1 && !harness.engine.commandInFlight(); },
                     "the first prompt did not run its first command");
    harness.model.beginObserver = [this] {
      harness.memory->hold("limit", harness.governor->snapshot().headroomBytes - 32);
      ledger.addSource(landing);
    };
    EngineRequest second = harness.request(2, std::vector<uint32_t>(2900, 2));
    second.maxNewTokens = 2;
    second.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(second));
    harness.runUntil([&] { return waiting(1) && waiting(2); }, "the claims did not wait for the landing");
  }
  ~LeftOutLane() { ledger.removeSource(landing); }
  LeftOutLane(const LeftOutLane &) = delete;
  LeftOutLane &operator=(const LeftOutLane &) = delete;

  [[nodiscard]] bool waiting(uint64_t id) const {
    return std::ranges::any_of(ledger.claims(), [&](const auto &entry) {
      return entry.second.spec.requestId == id && entry.second.waiting;
    });
  }
};

// A lane a prefill command leaves out while it is ready to retry waits for that command, not on its claim: its claim
// holds no room the command's lanes must take for it, and its TTFT counts the command as another's prefill.
void testALaneAPlanLeavesOutWaitsForItsCommandNotOnItsClaim() {
  LeftOutLane lanes;
  test::EngineHarness &harness = lanes.harness;
  bool heldWhileLeftOut = false;
  harness.model.submitObserver = [&] { heldWhileLeftOut = heldWhileLeftOut || lanes.waiting(1); };
  harness.memory->release("limit", harness.memory->held("limit"));
  lanes.landing.landed = true;
  const size_t before = harness.commands().size();
  harness.runUntil([&] { return harness.events.tokenTimes.contains(1); }, "the first prompt did not answer");
  harness.model.submitObserver = nullptr;
  const test::EngineHarness::Command &leftOut = harness.commands().at(before);
  require(leftOut.lanes == std::vector<uint64_t>{2}, "the second prompt's command did not leave the first out");
  const TtftParts &parts = harness.events.ttftParts.at(1);
  require(!heldWhileLeftOut && parts[static_cast<size_t>(TtftPart::OtherPrefill)] >= leftOut.wallMilliseconds &&
              parts[static_cast<size_t>(TtftPart::ClaimWait)] < leftOut.wallMilliseconds,
          "a lane the command left out waited on its claim: claim wait " +
              std::to_string(parts[static_cast<size_t>(TtftPart::ClaimWait)]) + " ms, other prefill " +
              std::to_string(parts[static_cast<size_t>(TtftPart::OtherPrefill)]) + " ms, the command " +
              std::to_string(leftOut.wallMilliseconds) + " ms");
  harness.runUntilIdle();
  require(harness.events.completedCount == 2 && harness.events.failedCount == 0, "the prompts did not finish");
}

// A lane each retry's plan leaves out before it retries it in turn holds room for the resource wait's limit from its
// first refusal that waited, not from its latest.
void testALeftOutLaneHoldsRoomForTheLimitFromItsFirstWait() {
  LeftOutLane lanes;
  test::EngineHarness &harness = lanes.harness;
  const double waited = harness.now();
  double lastHeld = waited;
  harness.runUntil(
      [&] {
        if (lanes.waiting(1))
          lastHeld = harness.now();
        return harness.now() >= waited + 2 * kResourceWaitTimeoutMilliseconds || harness.events.failedCount == 2;
      },
      "the waits did not end", 10'000'000);
  require(lastHeld < waited + kResourceWaitTimeoutMilliseconds + kResourceRetryBackoffMilliseconds,
          "the left-out lane's claim held room " + std::to_string(lastHeld - waited) + " ms after it first waited");
}

// A lane alone in service grows into the pooled buffers a lane starts from once nothing else meets its KV: the memory
// plan gives one lane the whole context beside its own state, not beside a second lane's.
void testALoneLaneGrowsIntoThePooledLaneBuffers() {
  constexpr uint64_t kPageBytes = 4096, kLane = 64 * 1024;
  constexpr uint32_t kPages = 256;
  test::EngineHarness harness(
      {.pages = kPages, .memory = test::FakeMemory::Config{}, .limits = {.limitBytes = kLane + kPages * kPageBytes}});
  harness.model.laneFootprintBytes = kLane;
  // Request 1 runs beside request 2's first command and ends, leaving its
  // lane's buffers in the pool; request 2's KV then needs them.
  bool beside = false;
  harness.model.beginObserver = [&] {
    beside = beside || (harness.model.lastBeginId == 2 && harness.model.requests.contains(1));
  };
  for (const auto &[id, tokens] : {std::pair<uint64_t, uint32_t>{1, 40}, {2, 7800}}) {
    EngineRequest value = harness.request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(value));
  }
  harness.runUntilIdle();
  require(beside, "request 2 did not start beside request 1");
  require(harness.events.completedCount == 2 && harness.events.failedCount == 0,
          "a lone lane whose context fits beside its own state failed: " +
              std::to_string(harness.events.failedCount) + " failed, " +
              std::to_string(harness.model.pooledLaneBytes) + " bytes of a lane's buffers kept in the pool");
}

// The budget of one lane and the whole KV, as the memory plan sizes it, and the page ids of that KV: a lane alone
// at the whole context leaves its pages cached, with every extent allocated and the budget not reached. The next
// request's pages are refused as the budget refuses them, and take those cached pages: both complete.
void testTheRequestAfterAWholeContextLaneTakesItsCachedPages() {
  constexpr uint64_t kPageBytes = 4096, kLane = 64 * 1024;
  constexpr uint32_t kPages = 256, kExtentPages = 32;
  test::EngineHarness harness({.pages = kPages,
                               .extentPages = kExtentPages,
                               .engine = {.maxContext = kPages * kv::kPageTokens - 7},
                               .memory = test::FakeMemory::Config{},
                               .limits = {.limitBytes = kLane + kPages * kPageBytes}});
  harness.model.laneFootprintBytes = kLane;
  uint64_t completed = 0;
  for (const auto &[id, tokens] : {std::pair<uint64_t, uint32_t>{1, 8100}, {2, 2000}}) {
    EngineRequest value = harness.request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.deadlineMilliseconds = 1e12;
    harness.engine.submit(std::move(value));
    harness.runUntilIdle();
    require(harness.events.completedCount == ++completed && harness.events.failedCount == 0,
            "request " + std::to_string(id) + " did not complete: " + std::to_string(harness.events.failedCount) +
                " failed");
  }
}

// Eight requests near the whole context that share a prefix arrive at once under the budget of one lane and the
// whole KV: each fits alone, so each completes, whatever extent and lane sizes and prefix.
void testRequestsNearTheWholeContextOnASharedPrefixAllComplete() {
  constexpr uint64_t kPageBytes = 4096;
  constexpr uint32_t kPages = 256;
  struct Shape final {
    uint32_t extentPages;
    uint64_t laneBytes;
    uint32_t prefix;
  };
  for (const Shape shape : {Shape{32, 64 * 1024, 1024}, Shape{32, 300 * 1024, 2048}, Shape{4, 96 * 1024, 0},
                            Shape{32, 0, 1024}}) {
    test::EngineHarness harness({.pages = kPages,
                                 .extentPages = shape.extentPages,
                                 .engine = {.maxContext = kPages * kv::kPageTokens - 7},
                                 .memory = test::FakeMemory::Config{},
                                 .limits = {.limitBytes = shape.laneBytes + kPages * kPageBytes}});
    harness.model.laneFootprintBytes = shape.laneBytes;
    uint64_t id = 1;
    for (const uint32_t tokens : {7800U, 3000U, 6000U, 8100U, 2000U, 7000U, 5000U, 7900U}) {
      std::vector<uint32_t> prompt(tokens, 7);
      for (uint32_t index = shape.prefix; index < tokens; ++index)
        prompt[index] = static_cast<uint32_t>(1000 + id * 13 + index % 11);
      EngineRequest value = harness.request(id++, std::move(prompt));
      value.deadlineMilliseconds = 1e12;
      harness.engine.submit(std::move(value));
    }
    harness.runUntilIdle();
    require(harness.events.completedCount == 8 && harness.events.failedCount == 0,
            "of eight requests that each fit alone (extents of " + std::to_string(shape.extentPages) +
                " pages, a lane of " + std::to_string(shape.laneBytes) + " bytes, a prefix of " +
                std::to_string(shape.prefix) + " tokens), " + std::to_string(harness.events.failedCount) +
                " failed");
  }
}

// A start refused memory that lands later holds no room on its claim from the pass in which the scheduler declines
// it, while a prompt that leads it prefills. Wanted and refused again, its claim waits again, its wait's age kept from
// its first refusal, within the resource wait limit from its first wait only.
void testADeclinedRefusalHoldsNoClaimRoom() {
  for (const bool pastLimit : {false, true}) {
    LandingUnit landing{1ULL << 30};
    landing.landed = true;
    test::EngineHarness harness({.pages = 4096, .memory = test::FakeMemory::Config{}});
    MemoryLedger &ledger = harness.governor->ledger();
    ledger.addSource(landing);
    harness.model.laneFootprintBytes = 64 << 10;
    // Each start of request 2 finds memory short of its state until the
    // landing lands.
    bool refuses = true;
    harness.model.beginObserver = [&] {
      if (refuses && harness.model.lastBeginId == 2 && !harness.memory->held("limit")) {
        harness.memory->hold("limit", harness.governor->snapshot().headroomBytes - 32);
        landing.landed = false;
      }
    };
    const auto land = [&] {
      harness.memory->release("limit", harness.memory->held("limit"));
      landing.landed = true;
    };
    const auto waiting = [&] {
      return std::ranges::any_of(ledger.claims(), [](const auto &entry) {
        return entry.second.spec.requestId == 2 && entry.second.waiting;
      });
    };
    const auto submit = [&](uint64_t id, uint32_t tokens) {
      EngineRequest value = harness.request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
      value.deadlineMilliseconds = 1e9;
      harness.engine.submit(std::move(value));
    };
    // Request 1 finds no free lane at its first attempt, so request 2,
    // submitted after it, is tried alone.
    harness.model.maximumLanes = 0;
    submit(1, 8193);
    static_cast<void>(harness.tick());
    harness.model.maximumLanes = model::ExecutionLimits::maximumBatchWidth;
    submit(2, 16385);
    static_cast<void>(harness.tick());
    require(waiting() && harness.engine.resourceWaitSnapshot(harness.now()).memory == 1,
            "the start was not refused memory that lands later");
    const double refused = harness.now();
    land();
    // Request 1 starts at its retry and leads request 2.
    harness.runUntil([&] { return harness.model.requests.contains(1); }, "the first request did not start");
    require(!waiting() && harness.engine.resourceWaitSnapshot(harness.now()).memory == 0,
            "a start the scheduler declined held room on its claim");
    if (pastLimit)
      harness.advanceTo(refused + kResourceWaitTimeoutMilliseconds);
    // Request 1's last command wants request 2 again, which is refused again.
    harness.runUntil([&] { return harness.engine.resourceWaitSnapshot(harness.now()).memory == 1; },
                     "the declined start was not tried again");
    require(waiting() == !pastLimit &&
                harness.engine.resourceWaitSnapshot(harness.now()).oldestWaitMilliseconds >= harness.now() - refused,
            pastLimit ? "a declined start's claim held room again past the limit from its first wait"
                      : "a declined start's claim did not wait again once wanted, or its wait's age restarted");
    refuses = false;
    land();
    harness.runUntilIdle();
    require(harness.events.completedCount == 2 && harness.events.failedCount == 0, "the prompts did not finish");
  }
}

// The time a start waits for the host to admit it counts as its claim's wait, not as the decode commands' that ran
// beside it.
void testAHostRefusedStartWaitsOnItsClaim() {
  test::EngineHarness harness({.pages = 256, .memory = test::FakeMemory::Config{}, .timing = &test::kM6FlashNextQ2});
  harness.model.decodeFinishes = false;
  harness.model.laneFootprintBytes = 64 << 10;
  EngineRequest decoder = harness.request(1, std::vector<uint32_t>(40, 1));
  decoder.maxNewTokens = 100'000;
  decoder.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(decoder));
  harness.runUntil([&] { return harness.events.outputs.contains(1); }, "the decoder did not decode");
  const test::GovernorLimits limits;
  const uint64_t other = harness.memory->hostBytes() - limits.hostSpareReserveBytes - harness.memory->total();
  harness.memory->setHost(other - kHostMarginBytes + (1ULL << 20), MemoryPressure::Normal);
  const double arrived = harness.now();
  EngineRequest prompt = harness.request(2, std::vector<uint32_t>(300, 2));
  prompt.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(prompt));
  harness.runUntil([&] { return harness.now() >= arrived + 2000.0; }, "the clock did not move");
  require(!harness.events.tokenTimes.contains(2), "the host admitted a start short of its margin");
  harness.memory->setHost(other - 4 * kHostMarginBytes, MemoryPressure::Normal);
  const double released = harness.now();
  harness.runUntil([&] { return harness.events.tokenTimes.contains(2); }, "the prompt did not answer");
  const TtftParts &parts = harness.events.ttftParts.at(2);
  require(parts[static_cast<size_t>(TtftPart::ClaimWait)] > 0.5 * (released - arrived),
          "a start the host refused waited " + std::to_string(released - arrived) + " ms, of which " +
              std::to_string(parts[static_cast<size_t>(TtftPart::ClaimWait)]) + " ms counted as its claim's wait and " +
              std::to_string(parts[static_cast<size_t>(TtftPart::Decode)]) + " ms as decode commands'");
}

// A start beside a decoding lane gives back an idle rebuild-only arena it does not hold, but not the expert store
// the decode reads.
void testAStartReleasesAnIdleArenaBesideADecodingLane() {
  constexpr uint64_t kExtent = 4 * 4096, kState = 64 * 1024, kChunk = 64 * 1024, kArena = 256 * 1024;
  test::EngineHarness harness(
      {.pages = 64,
       .memory = test::FakeMemory::Config{},
       .limits = {.limitBytes = 3 * kChunk + kArena + kState + 2 * kExtent + kExtent}});
  // The store: three units, one its floor, of the four that would hold every
  // expert, which nothing misses. The arena: one unit, which the next image's
  // start builds again.
  test::FakeElasticPool experts("experts", *harness.memory, kChunk, 1, 3, 4, 3);
  test::FakeElasticPool arena("vision_arena", *harness.memory, kArena, 0, 1, 1, 1);
  arena.rebuildOnly = true;
  static_cast<void>(harness.governor->ledger().addPool(experts));
  static_cast<void>(harness.governor->ledger().addPool(arena));
  harness.model.laneFootprintBytes = kState;
  harness.model.decodeFinishes = false;
  EngineRequest text = harness.request(1, std::vector<uint32_t>(40, 1));
  text.maxNewTokens = 30;
  text.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(text));
  harness.runUntil([&] { return harness.events.outputs.contains(1); }, "the text request did not decode");
  EngineRequest second = harness.request(2, std::vector<uint32_t>(16, 2));
  second.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(second));
  harness.runUntil([&] { return harness.events.outputs.contains(2); }, "the second request did not answer");
  require(!harness.events.usage.contains(1) && arena.shrinks == 1,
          "the second request started once the text request's decode ended (" +
              std::to_string(harness.events.outputs[1].size()) + " of its 30 tokens), not by giving back the "
              "idle arena (" + std::to_string(arena.shrinks) + " units)");
  require(experts.shrinks == 0, "a start took a unit of the store");
  harness.runUntilIdle();
  require(harness.events.completedCount == 2 && harness.events.failedCount == 0 && experts.shrinks == 0,
          "the text request did not finish beside the second request");
}

// An image start beside a decode keeps the encoder arena its image fits and gives back the idle embedding rows; the
// expert store keeps its units.
void testAnImageStartKeepsItsArenaAndGivesBackIdleRows() {
  constexpr uint64_t kExtent = 4 * 4096, kState = 64 * 1024, kChunk = 64 * 1024, kCache = 256 * 1024;
  test::EngineHarness harness(
      {.pages = 64,
       .memory = test::FakeMemory::Config{},
       .limits = {.limitBytes = 3 * kChunk + 2 * kCache + kState + 2 * kExtent + kExtent}});
  // The store as testAStartReleasesAnIdleArenaBesideADecodingLane's. The
  // arena and the rows: one unit each, built again by an image's start and
  // encode.
  test::FakeElasticPool experts("experts", *harness.memory, kChunk, 1, 3, 4, 3);
  test::FakeElasticPool arena("vision_arena", *harness.memory, kCache, 0, 1, 1, 1);
  test::FakeElasticPool rows("embeddings", *harness.memory, kCache, 0, 1, 1, 1);
  arena.rebuildOnly = rows.rebuildOnly = true;
  static_cast<void>(harness.governor->ledger().addPool(experts));
  static_cast<void>(harness.governor->ledger().addPool(arena));
  static_cast<void>(harness.governor->ledger().addPool(rows));
  harness.model.laneFootprintBytes = kState;
  harness.model.decodeFinishes = false;
  // The image request's activation holds the arena from its first attempt.
  harness.model.beginObserver = [&] {
    if (harness.model.lastBeginId == 2) arena.hold(1);
  };
  EngineRequest text = harness.request(1, std::vector<uint32_t>(40, 1));
  text.maxNewTokens = 30;
  text.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(text));
  harness.runUntil([&] { return harness.events.outputs.contains(1); }, "the text request did not decode");
  EngineRequest image = harness.request(2, std::vector<uint32_t>(16, 2));
  image.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(image));
  harness.runUntil([&] { return harness.events.outputs.contains(2); }, "the image request did not answer");
  require(!harness.events.usage.contains(1) && rows.shrinks == 1,
          "the image request started once the text request's decode ended (" +
              std::to_string(harness.events.outputs[1].size()) + " of its 30 tokens), not by giving back the "
              "embedding rows nothing holds (" + std::to_string(rows.shrinks) + " units)");
  require(arena.shrinks == 0 && experts.shrinks == 0, "a start took the arena its image holds or a unit of the store");
  harness.runUntilIdle();
  require(harness.events.completedCount == 2 && harness.events.failedCount == 0 && experts.shrinks == 0,
          "the text request did not finish beside the image request");
}

// A decode gap past the scheduler's bound counts once, under the first cause that held in it; a gap within the
// bound counts nowhere.
void testADecodeGapNamesItsCause() {
  test::EngineHarness harness({.pages = 1024, .timing = &test::kM6FlashNextQ2});
  harness.model.decodeFinishes = false;
  EngineRequest decoder = harness.request(1, std::vector<uint32_t>(40, 1));
  decoder.maxNewTokens = 4000;
  decoder.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(decoder));
  harness.runUntil([&] { return harness.events.outputs[1].size() > 8; }, "the decoder did not decode");
  const auto causes = [&] { return harness.engine.snapshot().decodeGapCauses; };
  require(causes().gaps == std::array<uint64_t, kGapCauses>{} &&
              causes().milliseconds == std::array<double, kGapCauses>{},
          "a decode gap within the bound counted under a cause");
  EngineRequest prompt = harness.request(2, std::vector<uint32_t>(16384, 2));
  prompt.maxNewTokens = 1;
  prompt.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(prompt));
  harness.runUntil([&] { return harness.events.tokenTimes.contains(2); }, "the prompt did not answer");
  const DecodeGapCausesSnapshot counted = causes();
  const auto at = [](GapCause cause) { return static_cast<size_t>(cause); };
  const double target = harness.engine.snapshot().scheduler.decodeGapTargetMilliseconds;
  require(counted.gaps[at(GapCause::PrefillUnit)] > 0 &&
              counted.milliseconds[at(GapCause::PrefillUnit)] > target * counted.gaps[at(GapCause::PrefillUnit)] &&
              counted.prefillMilliseconds > 0.0,
          "the decode gaps the prompt's prefill units made past the bound did not count under a prefill unit: " +
              std::to_string(counted.gaps[at(GapCause::PrefillUnit)]) + " gaps");
  for (const GapCause cause : {GapCause::ClaimWait, GapCause::SsdRead, GapCause::PoolRelease, GapCause::Replan,
                               GapCause::Other})
    require(!counted.gaps[at(cause)], "a decode gap counted under a cause that did not hold in it");
}

// A decode gap ended by a command that reads experts the store missed counts as an SSD wait, for every lane of the
// command.
void testADecodeGapOnExpertReadsIsAnSsdWait() {
  test::EngineHarness harness({.pages = 1024, .memory = test::FakeMemory::Config{}, .timing = &test::kM6FlashNextQ2});
  test::FakeElasticPool experts("experts", *harness.memory, 1ULL << 20, 1, 1, 32, 1);
  harness.model.experts = &experts;
  harness.model.time(test::kM6FlashNextQ2);
  harness.model.decodeFinishes = false;
  for (uint64_t id = 1; id <= 4; ++id) {
    EngineRequest decoder = harness.request(id, std::vector<uint32_t>(40, static_cast<uint32_t>(id)));
    decoder.maxNewTokens = 64;
    decoder.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(decoder));
  }
  harness.runUntilIdle();
  const EngineSnapshot snapshot = harness.engine.snapshot();
  const DecodeGapCausesSnapshot &counted = snapshot.decodeGapCauses;
  const auto at = [](GapCause cause) { return static_cast<size_t>(cause); };
  const double bound = snapshot.scheduler.decodeGapTargetMilliseconds;
  require(harness.longestDecodeGap() > bound && snapshot.scheduler.decodeBatchesByWidth[3] > 1,
          "four lanes did not decode in commands longer than the bound, " + std::to_string(bound) + " ms");
  require(counted.gaps[at(GapCause::SsdRead)] > 0 && !counted.gaps[at(GapCause::Other)] &&
              !counted.gaps[at(GapCause::Replan)] && !counted.gaps[at(GapCause::PoolRelease)],
          "decode gaps past the bound that commands reading missed experts ended counted " +
              std::to_string(counted.gaps[at(GapCause::SsdRead)]) + " times as an SSD wait, " +
              std::to_string(counted.gaps[at(GapCause::PoolRelease)]) + " as a pool release, " +
              std::to_string(counted.gaps[at(GapCause::Replan)]) + " as a replan and " +
              std::to_string(counted.gaps[at(GapCause::Other)]) + " under no cause");
}

// A decode gap ended by a command that read spilled K/V from the SSD, hot blocks its lane missed, counts as an SSD
// wait; one ended by a command whose lane found every hot block in memory counts as none, however long the
// command took.
void testADecodeGapIsAnSsdWaitOnlyWhenItsHotBlocksMissed() {
  for (const bool missed : {false, true}) {
    test::EngineHarness harness({.pages = 1024, .timing = &test::kM6FlashNextQ2});
    harness.model.decodeFinishes = false;
    const double bound = harness.engine.snapshot().scheduler.decodeGapTargetMilliseconds;
    harness.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
      test::CommandTimes times;
      if (plan.kind == WorkKind::Decode) {
        times.wallMilliseconds = 2.0 * bound;
        times.sample.work.spilledKvReadBytes = missed ? 1ULL << 20 : 0;
      }
      return times;
    };
    EngineRequest decoder = harness.request(1, std::vector<uint32_t>(40, 1));
    decoder.maxNewTokens = 4000;
    decoder.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(decoder));
    harness.runUntil([&] { return harness.events.tokenTimes[1].size() > 8; }, "the decoder did not decode");
    const DecodeGapCausesSnapshot counted = harness.engine.snapshot().decodeGapCauses;
    const auto at = [](GapCause cause) { return static_cast<size_t>(cause); };
    const uint64_t gaps = std::accumulate(counted.gaps.begin(), counted.gaps.end(), uint64_t{0});
    require(harness.longestDecodeGap() > bound && gaps > 0 &&
                counted.gaps[at(missed ? GapCause::SsdRead : GapCause::Other)] == gaps,
            std::string("decode gaps past the bound whose commands read ") + (missed ? "missed" : "no") +
                " hot blocks counted " + std::to_string(counted.gaps[at(GapCause::SsdRead)]) +
                " times as an SSD wait and " + std::to_string(counted.gaps[at(GapCause::Other)]) +
                " under no cause, of " + std::to_string(gaps));
  }
}

// Copies ahead for a pool's regrowth stay within about the unit the try lacked (one 16-page batch past it at most),
// and stop once the store is at its cap.
void testCopiesAheadForRegrowthStayWithinAUnit() {
  constexpr uint64_t pageBytes = 4096;
  constexpr uint32_t unitPages = 64;
  for (const bool atCap : {false, true}) {
    test::TestKvStorage storage(1024, pageBytes, 4);
    KvPool pool(storage, 0);
    test::TestKvTier tier(false);
    tier.transferLimit = 256;
    tier.capacity = 1024;
    engine::Cache cache(pool, &tier, test::ampleDiskBudget());
    Executor executor;
    executor.keptTier = std::make_shared<test::OffloadControl>();
    executor.keptTier->ready = true;
    Events events;
    test::GovernedHost host;
    engine::Engine engine{host.connect({.maxContext = 102400}), cache, executor, events};
    guardReleases(storage, engine);
    test::FakeElasticPool experts("experts", host.memory, unitPages * pageBytes, 1, 2, 3, atCap ? 2 : 1);
    static_cast<void>(host.governor.ledger().addPool(experts));
    double now = 1;
    for (uint64_t id = 1; id <= 4; ++id) {
      engine.submit(request(id, std::vector<uint32_t>(2060, static_cast<uint32_t>(id))));
      test::tickUntil(engine, now, [&] { return idle(engine); }, "a conversation did not finish");
    }
    require(!tier.demotions, "a temporary tier copied before anything asked for copies");
    host.leaveRoom(0);
    now += 2 * WriteBehind::kDelayMilliseconds;
    static_cast<void>(host.governor.controlPass(1000.0 * (kRegrowQuietSeconds + 1.0)));
    require(experts.units() == (atCap ? 2u : 1u) &&
                host.governor.ledger().growthLackBytes() == (atCap ? 0 : unitPages * pageBytes),
            "the regrowth try did not lack its unit from uncopied cache");
    // An Immediate claim reached uncopied cache: copies ahead pay
    // (MemoryLedger::copyDemand), the lanes asking for none.
    MemoryLedger &ledger = host.governor.ledger();
    const ClaimId reached = ledger.open({.cls = ClaimClass::Optional, .requestId = 1, .timing = NeedTiming::Immediate});
    static_cast<void>(ledger.free(reached, {.pages = 1}, 0, MemoryPressure::Normal));
    ledger.close(reached);
    require(host.governor.copyDemand() && !tier.demotions, "no claim reached uncopied cache");
    for (uint32_t step = 0; step < 32; ++step) {
      tier.complete();
      static_cast<void>(cache.pollTransfers());
      static_cast<void>(engine.tick(++now));
    }
    if (atCap)
      require(!tier.demotions, "write-behind copied ahead with the store at its cap");
    else
      require(tier.demotions >= unitPages && tier.demotions <= unitPages + 16,
              "write-behind copied " + std::to_string(tier.demotions) + " pages ahead for a " +
                  std::to_string(unitPages) + "-page unit");
  }
}

} // namespace

std::span<const test::TestCase> rulesTests() {
  static constexpr test::TestCase kTests[] = {
      {"testALowerPriorityClaimTakesNothingAHigherPriorityLaneNeeds",
       testALowerPriorityClaimTakesNothingAHigherPriorityLaneNeeds},
      {"testASpillLanesIndexPagesAreClaimedAtItsPriority", testASpillLanesIndexPagesAreClaimedAtItsPriority},
      {"testSupersedingClaimTakesThePassCommand", testSupersedingClaimTakesThePassCommand},
      {"testALaneOfACommandTakenOutsidePreparationEndsAtTheNextTick",
       testALaneOfACommandTakenOutsidePreparationEndsAtTheNextTick},
      {"testLandingPaysItsOwnerFirst", testLandingPaysItsOwnerFirst},
      {"testQuotaWaitIsNotAMemoryClaim", testQuotaWaitIsNotAMemoryClaim},
      {"testALaneBelowAWaitingClaimTakesWhatItLacksBeyondTheHeldPages",
       testALaneBelowAWaitingClaimTakesWhatItLacksBeyondTheHeldPages},
      {"testARestoredStateIsPromotedInAClaim", testARestoredStateIsPromotedInAClaim},
      {"testAReplayPointBesideACheckpointIsReservedInAClaim", testAReplayPointBesideACheckpointIsReservedInAClaim},
      {"testEnginesOnOneGovernorOfferTheCacheOnce", testEnginesOnOneGovernorOfferTheCacheOnce},
      {"testTtftPartsSumToTtft", testTtftPartsSumToTtft},
      {"testAHostRefusedStartWaitsOnItsClaim", testAHostRefusedStartWaitsOnItsClaim},
      {"testAStartReleasesAnIdleArenaBesideADecodingLane", testAStartReleasesAnIdleArenaBesideADecodingLane},
      {"testAnImageStartKeepsItsArenaAndGivesBackIdleRows", testAnImageStartKeepsItsArenaAndGivesBackIdleRows},
      {"testADecodeGapNamesItsCause", testADecodeGapNamesItsCause},
      {"testADecodeGapOnExpertReadsIsAnSsdWait", testADecodeGapOnExpertReadsIsAnSsdWait},
      {"testADecodeGapIsAnSsdWaitOnlyWhenItsHotBlocksMissed", testADecodeGapIsAnSsdWaitOnlyWhenItsHotBlocksMissed},
      {"testACancelBetweenPassesEndsBesideAHigherPrioritysDecode",
       testACancelBetweenPassesEndsBesideAHigherPrioritysDecode},
      {"testACancelBetweenPassesBesideItsOwnPrioritysDecodeKeepsThePasses",
       testACancelBetweenPassesBesideItsOwnPrioritysDecodeKeepsThePasses},
      {"testAHostRefusedStartIsFreedForByThePass", testAHostRefusedStartIsFreedForByThePass},
      {"testAStartRanksByTheWorkItHolds", testAStartRanksByTheWorkItHolds},
      {"testCopiesAheadForRegrowthStayWithinAUnit", testCopiesAheadForRegrowthStayWithinAUnit},
      {"testALaneAPlanLeavesOutWaitsForItsCommandNotOnItsClaim",
       testALaneAPlanLeavesOutWaitsForItsCommandNotOnItsClaim},
      {"testALeftOutLaneHoldsRoomForTheLimitFromItsFirstWait", testALeftOutLaneHoldsRoomForTheLimitFromItsFirstWait},
      {"testADeclinedRefusalHoldsNoClaimRoom", testADeclinedRefusalHoldsNoClaimRoom},
      {"testALoneLaneGrowsIntoThePooledLaneBuffers", testALoneLaneGrowsIntoThePooledLaneBuffers},
      {"testTheRequestAfterAWholeContextLaneTakesItsCachedPages",
       testTheRequestAfterAWholeContextLaneTakesItsCachedPages},
      {"testRequestsNearTheWholeContextOnASharedPrefixAllComplete",
       testRequestsNearTheWholeContextOnASharedPrefixAllComplete},
  };
  return kTests;
}
