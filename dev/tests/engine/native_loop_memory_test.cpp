#include "FakeElasticPool.hpp"
#include "LoopFixture.hpp"
#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestExecutor.hpp"
#include "TestMetalStatistics.hpp"
#include "TestRegistry.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryControl.hpp"
#include "engine/NativeRuntime.hpp"

#include <algorithm>
#include <optional>
#include <utility>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::Executor;
using splash::test::hostMemory;
using splash::test::LoopFixture;
using splash::test::PoolShape;
using splash::test::releaseAnswer;
using splash::test::request;
using splash::test::require;
using splash::test::runUntilIdle;

// The reporter logs a change in what requests wait for, never a retry or the
// depth of a queue.
void testMemoryStatusReporterLogsTransitionsOnly() {
  ResourceWaitSnapshot wait{.memory = 2, .concurrency = 1, .heldBehindRefusal = 4,
                            .restoring = 1, .suspended = 1,
                            .oldestWaitMilliseconds = 1250.0, .draining = true};
  MemoryStatusReporter reporter;
  require(reporter.update({}, true).empty(), "healthy idle engine logged pressure");
  require(!reporter.update(wait, false).empty(), "pressure transition was silent");
  ++wait.memory;
  wait.oldestWaitMilliseconds += 1000;
  require(reporter.update(wait, false).empty(), "pressure retries flooded the log");
  wait = {};
  wait.concurrency = 4;
  require(!reporter.update(wait, true).empty(), "end of memory wait was silent");
  require(reporter.update(wait, true).empty(), "concurrency queue logged pressure");
  // The refused request is deferred for scheduling, out of the memory
  // count, while it still holds the others back.
  wait.heldBehindRefusal = 2;
  const std::string held = reporter.update(wait, true);
  require(held.find("held=2") != std::string::npos &&
              held.find("cleared") == std::string::npos,
          "requests held behind a refusal were reported as no wait");
}

// The store's account in the governor's snapshot.
MemoryAccountSnapshot storeAccount(const MemoryGovernor &governor) {
  const MemoryGovernorSnapshot snapshot = governor.snapshot();
  const auto found = std::ranges::find(snapshot.accounts, std::string("experts"), &MemoryAccountSnapshot::name);
  require(found != snapshot.accounts.end(), "the store has no account");
  return *found;
}

// One control pass at the fixture's clock `elapsed` milliseconds on, at
// FakeMemory's pressure then.
void controlPassAfter(LoopFixture &fixture, double elapsed) {
  fixture.monotonic += elapsed;
  fixture.memory->advanceTo(fixture.monotonic);
  static_cast<void>(fixture.control->run(fixture.memory->pressure()));
}

// A store that came back at its floor, as the weights' restore leaves it,
// grows back to its cap through the control pass's growth into free room,
// a unit each kRegrowQuietSeconds through the ledger, while the host admits
// it: nothing else grows it.
void testRestoreBringsTheFloorThenGrowthTheRest() {
  LoopFixture fixture({}, {}, {}, metal::kResidencyKeepAliveSeconds,
                      test::LoopOptions{.memory = test::FakeMemory::Config{}});
  constexpr uint64_t unit = 1ULL << 30;
  test::FakeElasticPool store("experts", *fixture.memory, unit, 1, 4, 8, 1);
  static_cast<void>(fixture.governor->ledger().addPool(store));
  for (uint32_t pass = 0; pass < 4 && store.state().currentBytes < store.state().capBytes; ++pass)
    controlPassAfter(fixture, 1000.0 * kRegrowQuietSeconds);
  const MemoryAccountSnapshot account = storeAccount(*fixture.governor);
  require(store.state().currentBytes == 4 * unit && account.grows == 3 && account.shrinks == 0,
          "the control pass did not grow the store back from its floor to its cap a unit at a time");
}

// Critical pressure takes the margin once a settle second, here a store
// unit, nothing else being there; once the pressure is over, the store takes
// the free room back, a unit each kRegrowQuietSeconds.
void testStoreRetakesFreeRoomAfterCriticalPressure() {
  LoopFixture fixture({}, {}, {}, metal::kResidencyKeepAliveSeconds,
                      test::LoopOptions{.memory = test::FakeMemory::Config{
                                            .script = {{1'000.0, 8ULL << 30, MemoryPressure::Critical},
                                                       {2'000.0, 8ULL << 30, MemoryPressure::Normal}}}});
  constexpr uint64_t unit = 1ULL << 30;
  test::FakeElasticPool store("experts", *fixture.memory, unit, 1, 4, 8, 4);
  static_cast<void>(fixture.governor->ledger().addPool(store));
  controlPassAfter(fixture, 1'000.0);
  require(store.state().currentBytes == 3 * unit, "critical pressure did not take the margin from the store");
  for (uint32_t pass = 0; pass < 5 && store.state().currentBytes < store.state().capBytes; ++pass)
    controlPassAfter(fixture, 1000.0 * kRegrowQuietSeconds);
  require(store.state().currentBytes == 4 * unit,
          "the store did not take the free room back once the pressure was over");
}

// Borrowed 3(d): host readings that show a release only 0.5-1 s late cause
// no second release. Other applications leave the idle engine's host half
// its margin short; the control pass, every 100 ms, gives back a unit of a
// pool that costs only its rebuild (the host's claim takes none of the
// expert store's short of critical pressure, owner 11-03 17:30), which
// covers that, and the passes while the host's readings still show the
// shortfall take nothing more, nor take the unit back.
void testLaggingHostReadingsCauseNoSecondRelease() {
  constexpr uint64_t kGiB = 1ULL << 30;
  for (const double lag : {500.0, 1000.0}) {
    test::LoopFixture fixture({}, {}, {}, metal::kResidencyKeepAliveSeconds,
                              test::LoopOptions{.memory = test::FakeMemory::Config{.readingLagMilliseconds = lag}});
    test::FakeElasticPool arena("vision_arena", *fixture.memory, kGiB, 1, 4, 8, 4);
    arena.rebuildOnly = true;
    static_cast<void>(fixture.governor->ledger().addPool(arena));
    fixture.loop.announceReady(protocol::peer::kTestReady);
    // The control pass every 100 ms, the loop idle: past the lag first, so
    // that the readings show what the engine holds.
    const auto passes = [&](double milliseconds) {
      for (const double end = fixture.monotonic + milliseconds; fixture.monotonic < end; fixture.monotonic += 100.0) {
        fixture.memory->advanceTo(fixture.monotonic);
        static_cast<void>(fixture.loop.tick());
        static_cast<void>(fixture.control->run(fixture.memory->pressure()));
      }
    };
    passes(lag + 1000.0);
    require(!arena.shrinks && !arena.grows, "an ample host moved the pool");
    const test::GovernorLimits limits;
    fixture.memory->setHost(fixture.memory->hostBytes() - limits.hostSpareReserveBytes - fixture.memory->total() -
                                kHostMarginBytes / 2,
                            MemoryPressure::Normal);
    passes(3000.0);
    require(arena.shrinks == 1 && arena.grows == 0,
            "host readings " + std::to_string(lag) + " ms late took " + std::to_string(arena.shrinks) +
                " units of the pool and gave back " + std::to_string(arena.grows) + ", not one and none");
  }
}

} // namespace

std::span<const test::TestCase> memoryTests() {
  static constexpr test::TestCase kTests[] = {
      {"testMemoryStatusReporterLogsTransitionsOnly", testMemoryStatusReporterLogsTransitionsOnly},
      {"testRestoreBringsTheFloorThenGrowthTheRest", testRestoreBringsTheFloorThenGrowthTheRest},
      {"testStoreRetakesFreeRoomAfterCriticalPressure", testStoreRetakesFreeRoomAfterCriticalPressure},
      {"testLaggingHostReadingsCauseNoSecondRelease", testLaggingHostReadingsCauseNoSecondRelease},
  };
  return kTests;
}
