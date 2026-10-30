#include "EngineHarness.hpp"
#include "EngineTestSupport.hpp"
#include "MemoryPasses.hpp"
#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "TestRegistry.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

using namespace splash;
using namespace splash::engine;
using splash::test::cacheBlocksUnderDiskStates;
using splash::test::demoteState;
using splash::test::Events;
using splash::test::Executor;
using splash::test::guardReleases;
using splash::test::idle;
using splash::test::OffloadControl;
using splash::test::OffloadState;
using splash::test::OffloadTicket;
using splash::test::request;
using splash::test::RestoreControl;
using splash::test::runUntilIdle;
using splash::test::State;
using splash::test::tickUntil;

namespace {

using splash::test::require;

// With no cache slot a state goes straight to disk, one write at a time, and the next lane restores
// it; a rolling checkpoint goes to disk the same way.
void testStateWithoutACacheSlotGoesToDisk() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  executor.deniedReservations = 1000;
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> first(65, 1);
  auto decoding = request(1, first);
  decoding.maxNewTokens = 4;
  engine.submit(std::move(decoding));
  double now = 1;
  tickUntil(engine, now, [&] { return engine.snapshot().diskStatePublications == 1; },
            "the first lane did not publish its state");
  auto counters = engine.snapshot();
  auto states = cache.snapshot().stateCache;
  require(executor.diskSnapshots == 1 && counters.replayStatePublications == 1 &&
              counters.diskStatePublications == 1 &&
              counters.replayStatePublicationFailures == 0 &&
              counters.recycledStatePublications == 0 && states.entries == 1 &&
              states.bytes == 0 && states.diskBytes == 64 && states.offloads == 1,
          "the state did not go straight to disk");
  // The write is still in flight: the next boundary finds the staging
  // buffer busy and is not retried.
  engine.submit(request(2, std::vector<uint32_t>(65, 2)));
  tickUntil(engine, now,
            [&] { return engine.snapshot().replayStatePublicationFailures == 1; },
            "the second lane did not reach its boundary");
  require(events.completedCount == 0, "the first lane ended before the second boundary");
  runUntilIdle(engine);
  counters = engine.snapshot();
  require(executor.diskSnapshots == 1 && counters.diskStatePublications == 1 &&
              counters.replayStatePublicationFailures == 1 &&
              events.completedCount == 2 && events.failedCount == 0,
          "a second write started beside the one in flight");
  executor.stateTier->ready = true;
  static_cast<void>(engine.tick(1000));
  states = cache.snapshot().stateCache;
  require(states.entries == 1 && states.diskBytes == 64 && states.offloadFailures == 0,
          "the write did not land");
  // The prefix comes back from disk.
  engine.submit(request(3, first));
  static_cast<void>(engine.tick(1001));
  require(executor.diskReads == 1 && executor.prefillRows == 130,
          "the disk state was not read");
  executor.restoreControl->ready = true;
  runUntilIdle(engine);
  require(executor.restored == 64 && executor.prefillRows == 131 &&
              events.starts.back() == 64 &&
              events.failedCount == 0,
          "the prefix was not restored from the disk state");
  // A rolling checkpoint denied a cache slot goes to disk like any state,
  // and stays there beside the replay state of the same prompt, as the
  // prompt's last checkpoint.
  engine.submit(request(4, std::vector<uint32_t>(2 * kPrefillCheckpointTokens + 1, 4)));
  runUntilIdle(engine);
  counters = engine.snapshot();
  require(counters.checkpointPublications == 1 &&
              counters.checkpointPublicationFailures == 0 &&
              counters.diskStatePublications == 3 && executor.diskSnapshots == 3 &&
              counters.resources.stateCache.checkpointEntries == 1 &&
              counters.resources.stateCache.checkpointRetirements == 0 &&
              events.completedCount == 4 && events.failedCount == 0,
          "the checkpoint did not go to disk and stay there");
}

// A state already on disk at the same block is deduplicated, not written again.
void testStateAlreadyOnDiskIsDeduplicated() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  executor.deniedReservations = 1000;
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.stateTier->ready = true;
  Events events;
  engine::Engine engine(test::engineConfig(), cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(5001, 7);
  auto producer = request(1, prompt);
  producer.priority = RequestPriority::Background;
  engine.submit(std::move(producer));
  static_cast<void>(engine.tick(0));
  auto duplicate = request(2, prompt);
  duplicate.priority = RequestPriority::Foreground;
  engine.submit(std::move(duplicate));
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              executor.prefillRows == 2 * prompt.size(),
          "both lanes did not compute the prompt");
  const auto counters = engine.snapshot();
  require(executor.diskSnapshots == 1 && counters.diskStatePublications == 1 &&
              counters.replayStatePublications == 1 &&
              counters.deduplicatedStatePublications == 1,
          "a state already on disk was counted as written");
}

// A long prefill's disk checkpoint outlives its cancelled request: a branch restores it rather than
// replay, and keeps it.
void testCancelledPrefillRecoversFromItsDiskCheckpoint() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor(1);
  executor.deniedReservations = 1000;
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.stateTier->ready = true;
  executor.restoreControl->promotionDenied = true;
  Events events;
  engine::Engine engine(test::engineConfig(), cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> donor(2 * kPrefillCheckpointTokens + 1, 61);
  engine.submit(request(600, donor));
  for (uint32_t step = 0; step < 128; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (!engine.commandInFlight() && executor.requests.at(600).position == 6144)
      break;
  }
  auto counters = engine.snapshot();
  require(!engine.commandInFlight() && executor.requests.at(600).position == 6144 &&
              counters.checkpointPublications == 1 && counters.diskStatePublications == 1 &&
              counters.resources.stateCache.checkpointEntries == 1 &&
              counters.resources.stateCache.bytes == 0,
          "the checkpoint did not go to disk");
  engine.cancel(600);
  runUntilIdle(engine);
  require(engine.snapshot().cancelled == 1 &&
              cache.snapshot().stateCache.checkpointEntries == 1,
          "cancellation removed the disk checkpoint");
  std::vector<uint32_t> branch(donor.begin(), donor.begin() + 6145);
  branch.back() = 62;
  engine.submit(request(601, branch));
  static_cast<void>(engine.tick(1000));
  require(executor.diskReads == 1, "the disk checkpoint was not read");
  executor.restoreControl->ready = true;
  runUntilIdle(engine);
  counters = engine.snapshot();
  require(events.starts.back() == 4096 &&
              executor.restored == 4096 && executor.prefillRows == 6144 + 2049 &&
              counters.completed == 1 && events.failedCount == 0 &&
              counters.resources.stateCache.checkpointEntries == 1 &&
              counters.resources.stateCache.checkpointRetirements == 0,
          "the branch did not resume from the disk checkpoint and keep it");
}

// A final state refused beside another lane's write leaves the disk checkpoint, and the next turn
// resumes from it, whether the request was cancelled or completed.
void testFailedFinalStateKeepsTheDiskCheckpoint() {
  for (bool cancel : {true, false}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    executor.deniedReservations = 1000;
    executor.stateTier = std::make_shared<OffloadControl>();
    executor.stateTier->ready = true;
    executor.restoreControl->promotionDenied = true;
    Events events;
    engine::Engine engine(test::engineConfig(), cache, executor, events);
    guardReleases(storage, engine);
    const std::vector<uint32_t> prompt(2 * kPrefillCheckpointTokens + 1, 65);
    EngineRequest decoding = request(610, prompt);
    decoding.maxNewTokens = 64;
    engine.submit(decoding);
    for (uint32_t step = 0; step < 128; ++step) {
      static_cast<void>(engine.tick(step + 1));
      if (!engine.commandInFlight() && executor.requests.at(610).position == 6144)
        break;
    }
    auto counters = engine.snapshot();
    require(!engine.commandInFlight() && executor.requests.at(610).position == 6144 &&
                counters.checkpointPublications == 1 && counters.diskStatePublications == 1 &&
                counters.resources.stateCache.checkpointEntries == 1,
            "the checkpoint did not go to disk");
    executor.stateTier->ready = false;
    engine.submit(request(611, std::vector<uint32_t>(65, 66)));
    for (uint32_t step = 0; step < 128 && !engine.snapshot().replayStatePublicationFailures;
         ++step)
      static_cast<void>(engine.tick(1000 + step));
    counters = engine.snapshot();
    require(counters.replayStatePublications == 1 &&
                counters.replayStatePublicationFailures == 1 &&
                counters.diskStatePublications == 2 && executor.diskSnapshots == 2 &&
                events.failedCount == 0,
            "the final state was not refused beside the other lane's write");
    require(counters.resources.stateCache.checkpointEntries == 1 &&
                counters.resources.stateCache.checkpointRetirements == 0,
            "the final boundary retired the checkpoint it could not replace");
    if (cancel)
      engine.cancel(610);
    executor.stateTier->ready = true;
    runUntilIdle(engine);
    require(engine.snapshot().completed == (cancel ? 1U : 2U) &&
                cache.snapshot().stateCache.checkpointEntries == 1,
            "the request's end removed the checkpoint its final state did not replace");
    engine.submit(request(612, prompt));
    static_cast<void>(engine.tick(2000));
    executor.restoreControl->ready = true;
    runUntilIdle(engine);
    counters = engine.snapshot();
    require(events.starts.back() == 4096 &&
                executor.restored == 4096 && counters.cancelled == (cancel ? 1U : 0U) &&
                counters.completed == (cancel ? 2U : 3U) && events.failedCount == 0 &&
                counters.resources.stateCache.checkpointEntries == 1 &&
                counters.resources.stateCache.checkpointRetirements == 0,
            "the next turn did not resume from the disk checkpoint and keep it");
  }
}

// A disk checkpoint stays the recovery point until its successor is published; a retry resumes
// from it.
void testDiskCheckpointStaysUntilItsSuccessorIsPublished() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  executor.deniedReservations = 1000;
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.stateTier->ready = true;
  executor.restoreControl->promotionDenied = true;
  Events events;
  engine::Engine engine(test::engineConfig(), cache, executor, events);
  guardReleases(storage, engine);
  // Checkpoints at 4096 and 8192, the replay point at 12288.
  const std::vector<uint32_t> prompt(3 * kPrefillCheckpointTokens + 1, 67);
  engine.submit(request(620, prompt));
  for (uint32_t step = 0; step < 128; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (!engine.commandInFlight() && executor.requests.at(620).position == 6144)
      break;
  }
  auto counters = engine.snapshot();
  require(!engine.commandInFlight() && executor.requests.at(620).position == 6144 &&
              counters.checkpointPublications == 1 && counters.diskStatePublications == 1 &&
              counters.resources.stateCache.checkpointEntries == 1 &&
              counters.resources.stateCache.bytes == 0,
          "the checkpoint did not go to disk");
  executor.stateTier->ready = false;
  engine.submit(request(621, std::vector<uint32_t>(65, 68)));
  for (uint32_t step = 0; step < 128 && !engine.snapshot().checkpointPublicationFailures; ++step)
    static_cast<void>(engine.tick(1000 + step));
  counters = engine.snapshot();
  require(counters.checkpointPublicationFailures == 1 && counters.diskStatePublications == 2 &&
              counters.checkpointPublications == 1 && events.failedCount == 0,
          "the next checkpoint was not refused beside the other lane's write");
  require(counters.resources.stateCache.checkpointEntries == 1 &&
              counters.resources.stateCache.checkpointRetirements == 0 &&
              cache.probe(prompt, {}).cachedTokens() == kPrefillCheckpointTokens,
          "a checkpoint that was not published retired the one on disk");
  engine.cancel(620);
  executor.stateTier->ready = true;
  runUntilIdle(engine);
  engine.submit(request(622, prompt));
  static_cast<void>(engine.tick(2000));
  executor.restoreControl->ready = true;
  runUntilIdle(engine);
  require(events.starts.back() == kPrefillCheckpointTokens &&
              executor.restored == kPrefillCheckpointTokens && events.failedCount == 0,
          "the retry did not resume from the checkpoint on disk");
}

// No checkpoint is planned within a prefill chunk of the replay boundary; one a chunk or more away
// is published, in RAM or on disk, beside the replay state.
void testNoCheckpointWithinAChunkOfTheReplayBoundary() {
  const uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  for (bool denyRam : {false, true}) {
    for (uint32_t remaining : {32u, chunk - 32, chunk}) {
      test::TestKvStorage storage(512, 4096, 4);
      KvPool pool(storage, 0);
      engine::Cache cache(pool, nullptr, nullptr);
      Executor executor;
      executor.deniedReservations = denyRam ? 1000 : 0;
      executor.stateTier = std::make_shared<OffloadControl>();
      executor.stateTier->ready = true;
      Events events;
      engine::Engine engine(test::engineConfig(), cache, executor, events);
      guardReleases(storage, engine);
      const std::vector<uint32_t> prompt(kPrefillCheckpointTokens + remaining + 1, 71);
      engine.submit(request(1, prompt));
      runUntilIdle(engine);
      const auto counters = engine.snapshot();
      const uint64_t checkpoint = remaining >= chunk;
      require(counters.checkpointPublications == checkpoint &&
                  counters.checkpointPublicationFailures == 0 &&
                  counters.replayStatePublications == 1 &&
                  counters.replayStatePublicationFailures == 0 &&
                  executor.diskSnapshots == (denyRam ? 1 + checkpoint : 0) &&
                  counters.resources.stateCache.entries == 1 + checkpoint &&
                  counters.resources.stateCache.checkpointEntries == checkpoint &&
                  events.completedCount == 1 && events.failedCount == 0,
              "a checkpoint within a chunk of the replay boundary was planned or the final "
              "state was lost");
      executor.restoreControl->ready = true;
      engine.submit(request(2, prompt));
      runUntilIdle(engine);
      require(events.starts.back() == prompt.size() - 1 &&
                  events.completedCount == 2 && events.failedCount == 0,
              "skipping a checkpoint lost the final reusable prefix");
    }
  }
}

// A lane cancelled past a skipped near-final checkpoint resumes from the checkpoint before it, in
// RAM or on disk.
void testSkippedCheckpointKeepsPreviousRecoveryPoint() {
  for (bool disk : {false, true}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor(1);
    executor.deniedReservations = disk ? 1000 : 0;
    executor.stateTier = std::make_shared<OffloadControl>();
    executor.stateTier->ready = true;
    executor.restoreControl->ready = true;
    executor.restoreControl->promotionDenied = disk;
    Events events;
    engine::Engine engine(test::engineConfig(), cache, executor, events);
    guardReleases(storage, engine);
    const std::vector<uint32_t> prompt(2 * kPrefillCheckpointTokens + 33, 73);
    engine.submit(request(1, prompt));
    for (uint32_t step = 0; step < 128; ++step) {
      static_cast<void>(engine.tick(step + 1));
      if (!engine.commandInFlight() &&
          executor.requests.at(1).position == 2 * kPrefillCheckpointTokens)
        break;
    }
    const auto before = engine.snapshot();
    require(!engine.commandInFlight() &&
                executor.requests.at(1).position == 2 * kPrefillCheckpointTokens &&
                std::none_of(executor.plans.at(1).boundaries.begin(),
                             executor.plans.at(1).boundaries.end(),
                             [](const DraftBoundaryPlan &boundary) {
                               return boundary.boundary == 2 * kPrefillCheckpointTokens;
                             }) &&
                before.checkpointPublications == 1 &&
                before.checkpointPublicationFailures == 0 &&
                before.resources.stateCache.checkpointEntries == 1 &&
                before.resources.stateCache.checkpointRetirements == 0 &&
                executor.diskSnapshots == (disk ? 1 : 0),
            "a near-final checkpoint was planned or replaced its predecessor");
    engine.cancel(1);
    runUntilIdle(engine);
    engine.submit(request(2, prompt));
    runUntilIdle(engine);
    require(events.starts.back() == kPrefillCheckpointTokens &&
                engine.snapshot().cancelled == 1 && engine.snapshot().completed == 1 &&
                events.failedCount == 0 &&
                cache.snapshot().stateCache.checkpointEntries == 1,
            "cancellation after a skipped checkpoint lost the earlier recovery point");
  }
}

// A lone lane whose growth needs a cached state written out waits for the write in flight, also
// under host pressure, rather than fail or yield, and runs on the freed pages.
void testGrowthWaitsForTheStateWriteInFlight() {
  for (bool paused : {false, true}) {
    test::TestKvStorage storage(16, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    test::GovernedHost host;
    host.pause(paused);
    engine::Engine engine(host.connect({.maxContext = 102400}), cache, executor, events);
    guardReleases(storage, engine);
    // A write in flight from a lane still running: its block is no leaf to evict.
    auto writing = std::make_shared<OffloadControl>();
    cache.beginRequest(999);
    require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
    const auto held = test::publishBlocks(cache, 999, std::vector<uint32_t>(64, 12), 64);
    require(cache.publishStateToDisk(held, [&](std::function<void()>) {
              return std::make_unique<OffloadTicket>(writing);
            }, false),
            "fixture write did not start");
    // A cached state in RAM whose eviction must wait for that write.
    auto cached = std::make_shared<OffloadControl>();
    cache.beginRequest(998);
    require(cache.ensureTokens(998, 64).granted(), "fixture KV failed");
    const auto cachedBlock = test::publishBlocks(cache, 998, std::vector<uint32_t>(64, 13), 64);
    cache.publishCompositeState(cachedBlock, std::make_shared<OffloadState>(cached), false);
    cache.endRequest(998);
    // Every further page needs an extent the budget, or the host, refuses.
    storage.allocationFailure = paused ? metal::AllocationFailure::HostPressure
                                       : metal::AllocationFailure::EngineBudget;
    storage.growthBlocked = true;
    engine.submit(request(1, std::vector<uint32_t>(33, 17)));
    static_cast<void>(engine.tick(1));
    static_cast<void>(engine.tick(2));
    require(events.failedCount == 0 && executor.prefillRows == 0 && executor.suspensions == 0,
            "the lane failed or yielded while the tier was busy");
    writing->ready = true;
    static_cast<void>(engine.tick(3));
    require(cache.snapshot().stateCache.offloads == 2 && cache.snapshot().stateCache.bytes == 0 &&
                cached->released,
            "the cached state was not written once the staging buffer was free");
    cached->ready = true;
    for (uint32_t step = 4; step < 20 && !idle(engine); ++step)
      static_cast<void>(engine.tick(step));
    require(events.completedCount == 1 && events.failedCount == 0 && executor.suspensions == 0 &&
                executor.prefillRows == 33,
            "the lane did not run on the pages the written state gave up");
  }
}

// A lane waiting for pages always leaves a bounded wakeup, so a missed transfer wake cannot stall
// it.
void testWaitingLaneAlwaysNamesAWakeup() {
  test::TestKvStorage storage(12, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  cacheBlocksUnderDiskStates(cache, 6);
  engine.submit(request(1, std::vector<uint32_t>(97, 7)));
  static_cast<void>(engine.tick(1));
  require(tier.demotions == 1 && executor.prefillRows == 0,
          "the lane did not wait for a page");
  const auto wakeup = engine.nextWakeupMilliseconds();
  require(wakeup.has_value() && *wakeup <= 1.0 + 100.0,
          "a waiting lane left the engine without a bounded wakeup");
  // The wait ends on its own once the pages are back.
  for (uint32_t step = 2; step < 40 && !idle(engine); ++step) {
    tier.complete();
    static_cast<void>(engine.tick(step));
  }
  require(idle(engine) && events.outputs.contains(1) && events.failedCount == 0,
          "the lane never ran");
}

// Lanes waiting for pages leave the tick to lanes that need none: a decode beside a waiting
// prefill, a restored prefill beside a waiting decode.
void testWaitingPlanDoesNotIdleRunnableLanes() {
  const auto runWithTransfers = [](engine::Engine &engine, test::TestKvTier &tier,
                                   double &now, const std::function<bool()> &done) {
    for (uint32_t step = 0; step < 200 && !done(); ++step) {
      tier.complete();
      static_cast<void>(engine.tick(now++));
    }
    require(done(), "the waiting lane did not run once its pages landed");
  };
  {
    test::TestKvStorage storage(12, 4096, 4);
    storage.budgetPages = 8;
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    tier.transferLimit = 8;
    engine::Cache cache(pool, &tier, test::ampleDiskBudget());
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
    guardReleases(storage, engine);
    cacheBlocksUnderDiskStates(cache, 6);
    // A stream decodes within its one page, beside a prefill short of two.
    auto stream = request(2, std::vector<uint32_t>(1, 5));
    stream.maxNewTokens = 1000;
    engine.submit(std::move(stream));
    double now = 1;
    tickUntil(engine, now, [&] { return events.outputs.contains(2); },
              "the stream did not decode");
    engine.submit(request(1, std::vector<uint32_t>(97, 7)));
    const uint64_t decodes = engine.snapshot().scheduler.decodeBatches;
    tickUntil(engine, now, [&] { return tier.demotions > 0; },
              "the prefill did not wait for pages");
    require(engine.commandInFlight() && engine.snapshot().scheduler.decodeBatches == decodes + 1 &&
                executor.prefillRows == 1,
            "the stream idled while a prefill waited for its pages");
    runWithTransfers(engine, tier, now, [&] { return events.outputs.contains(1); });
    engine.cancel(2);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
    require(events.failedCount == 0 && executor.prefillRows == 98,
            "the lanes did not finish their prompts");
  }
  {
    test::TestKvStorage storage(72, 4096, 4);
    storage.budgetPages = 8;
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    tier.transferLimit = 8;
    engine::Cache cache(pool, &tier, test::ampleDiskBudget());
    Executor executor;
    Events events;
    engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
    guardReleases(storage, engine);
    cacheBlocksUnderDiskStates(cache, 6);
    // A prompt that goes on from the first block's 32 tokens by 8 more
    // restores that block's state from disk: it holds the block's page, and
    // takes the page its 8 rows need as it starts (Engine::admit), so its
    // prefill needs none. The read lands when the test says.
    std::vector<uint32_t> continuing(KvCache::pageTokens, 900);
    continuing.resize(KvCache::pageTokens + 8, 7);
    engine.submit(request(1, continuing));
    double now = 1;
    static_cast<void>(engine.tick(now++));
    // A short prompt takes the last of the eight pages memory holds, and its
    // first decode needs one more, which only a block's demotion frees.
    engine.submit(request(2, std::vector<uint32_t>(30, 8)));
    tickUntil(engine, now, [&] { return tier.demotions > 0; }, "the decode did not wait for a page");
    require(executor.diskReads == 1 && executor.prefillRows == 30 && pool.freePageCount() == 0 &&
                !engine.commandInFlight() && !events.outputs.contains(2),
            "fixture pages are off");
    // Once the read has landed, the restored prompt's rows run while the
    // decode waits for its page.
    executor.restoreControl->ready = true;
    const uint64_t prefills = engine.snapshot().scheduler.prefillBatches;
    tickUntil(engine, now, [&] { return engine.snapshot().scheduler.prefillBatches != prefills; },
              "the restored prompt did not run");
    require(engine.commandInFlight() && engine.snapshot().scheduler.prefillBatches == prefills + 1 &&
                executor.prefillRows == 38 && !events.outputs.contains(2),
            "the restored prompt idled while a decode waited for its page");
    runWithTransfers(engine, tier, now, [&] { return idle(engine); });
    require(events.completedCount == 2 && events.failedCount == 0 && executor.prefillRows == 38,
            "the lanes did not finish");
  }
}

// Memory runs short once: a claim demotes the leaf a written state needs, and it lands. A temporary
// tier's copies ahead follow the memory's demand for a while after (MemoryGovernor::copyDemand).
void runShortOfMemory(engine::Cache &cache, test::TestKvTier &tier, engine::MemoryGovernor &governor) {
  cacheBlocksUnderDiskStates(cache, 1);
  engine::MemoryLedger &ledger = governor.ledger();
  const engine::ClaimId claim = ledger.open({});
  const engine::FreeResult freed = ledger.free(claim, {.pages = 1}, 0, false);
  ledger.close(claim);
  require(freed.outcome == engine::FreeOutcome::Pending && governor.copyDemand(),
          "the written state's leaf was not demoted as a unit that lands later");
  tier.complete();
  require(cache.pollTransfers() && !cache.snapshot().pool.pagesPrefix, "the demotion did not land");
}

// Ticks the lane ahead through its long prompt, landing the tier's transfers, until done().
void tickBesideALaneAhead(engine::Engine &engine, test::TestKvTier &tier, double &now,
                          const std::function<bool()> &done) {
  for (uint32_t step = 0; step < 64 && !done(); ++step) {
    tier.complete();
    static_cast<void>(engine.tick(++now));
  }
}

// A replay point is copied behind its request: a persistent tier's after its delay, a temporary
// tier's only on the memory's demand; a clean stop writes only a persistent tier's.
void testReplayPointsAreWrittenBehind() {
  struct Fixture final {
    Fixture(bool persistent, bool shortOfMemory)
        : tier(persistent), cache(pool, &tier, test::ampleDiskBudget()),
          config(test::engineConfig({.maxContext = 102400})), engine(config, cache, executor, events) {
      tier.transferLimit = 64;
      tier.capacity = 128;
      executor.keptTier = std::make_shared<OffloadControl>();
      executor.keptTier->ready = true;
      guardReleases(storage, engine);
      if (shortOfMemory)
        runShortOfMemory(cache, tier, *config.governor);
      writes = tier.demotions;
      diskBytes = cache.snapshot().stateCache.diskBytes;
      engine.submit(request(1, std::vector<uint32_t>(2060, 7)));
      tickUntil(engine, now, [this] { return idle(engine); }, "the request did not finish");
    }
    // The chain's 64 pages and the state, written while they stay in RAM.
    [[nodiscard]] bool durableInRam() const {
      const CacheSnapshot resources = cache.snapshot();
      return engine.snapshot().writeBehind.durable == 1 && tier.demotions == writes + 64 &&
             resources.kvTier.copies == 64 && resources.pool.pagesPrefix >= 64 &&
             resources.stateCache.bytes >= 64 && resources.stateCache.diskBytes == diskBytes + 64;
    }
    test::TestKvStorage storage{1024, 4096, 4};
    KvPool pool{storage, 0};
    test::TestKvTier tier;
    engine::Cache cache;
    Executor executor;
    Events events;
    EngineConfig config;
    engine::Engine engine;
    double now = 1;
    // What the tier held before the request.
    uint32_t writes = 0;
    uint64_t diskBytes = 0;
  };
  {
    Fixture fixture(true, false);
    const std::optional<double> due = fixture.engine.nextWakeupMilliseconds();
    require(fixture.engine.snapshot().writeBehind.waiting == 1 && fixture.tier.demotions == fixture.writes &&
                due && *due > fixture.now && *due <= fixture.now + WriteBehind::kDelayMilliseconds,
            "the replay point did not wait for its delay");
    static_cast<void>(fixture.engine.tick(*due - 1));
    require(fixture.tier.demotions == fixture.writes, "the replay point was written before its delay");
    double now = *due;
    static_cast<void>(fixture.engine.tick(now));
    require(fixture.tier.demotions == fixture.writes + 16,
            "the replay point's copies did not start a few at a time");
    for (uint32_t step = 0; step < 16 && !fixture.durableInRam(); ++step) {
      fixture.tier.complete();
      static_cast<void>(fixture.engine.tick(++now));
    }
    require(fixture.durableInRam(), "the replay point did not become durable in RAM");
  }
  {
    Fixture fixture(false, true);
    require(fixture.engine.snapshot().writeBehind.waiting == 1 && !fixture.engine.nextWakeupMilliseconds(),
            "a temporary tier's point waited for a time while nothing asked for copies");
    static_cast<void>(fixture.engine.tick(fixture.now + 2 * WriteBehind::kDelayMilliseconds));
    require(fixture.tier.demotions == fixture.writes && fixture.engine.snapshot().writeBehind.waiting == 1,
            "a temporary tier wrote a point ahead while no lane asked for memory");
    double now = fixture.now + 2 * WriteBehind::kDelayMilliseconds;
    EngineRequest ahead = request(2, std::vector<uint32_t>(16384, 9));
    ahead.deadlineMilliseconds = 1e9;
    fixture.engine.submit(std::move(ahead));
    for (uint32_t step = 0; step < 8 && fixture.tier.demotions == fixture.writes; ++step)
      static_cast<void>(fixture.engine.tick(++now));
    require(fixture.tier.demotions == fixture.writes + 16,
            "the replay point's copies did not start a few at a time once the lane ahead asked for memory");
    tickBesideALaneAhead(fixture.engine, fixture.tier, now, [&] { return fixture.durableInRam(); });
    require(fixture.durableInRam(), "the replay point did not become durable in RAM on the memory's demand");
    fixture.engine.cancel(2);
    tickUntil(fixture.engine, now, [&] { return idle(fixture.engine); }, "the lane ahead did not end");
  }
  {
    Fixture fixture(true, false);
    require(!fixture.engine.flushRestorePoints() && fixture.tier.demotions == 16,
            "a clean stop did not write the point that had not waited");
    bool flushed = false;
    for (uint32_t step = 0; step < 16 && !flushed; ++step) {
      fixture.tier.complete();
      flushed = fixture.engine.flushRestorePoints();
    }
    require(flushed && fixture.durableInRam(), "a clean stop did not leave the point durable");
  }
  {
    Fixture fixture(false, false);
    static_cast<void>(fixture.engine.tick(fixture.now + 2 * WriteBehind::kDelayMilliseconds));
    require(fixture.engine.snapshot().writeBehind.waiting == 1 && !fixture.tier.demotions &&
                !fixture.engine.nextWakeupMilliseconds(),
            "a temporary tier wrote a point ahead while memory had not run short");
    require(fixture.engine.flushRestorePoints() && !fixture.tier.demotions,
            "a clean stop wrote a temporary tier's point");
  }
}

// A temporary tier copies a point once its request ends, a persistent one while it runs; once
// copied, a reclaim frees the conversation at once, writing nothing.
void testFinishedConversationsLeaveAtOnce() {
  for (const bool persistent : {false, true}) {
    test::TestKvStorage storage{1024, 4096, 4};
    KvPool pool{storage, 0};
    test::TestKvTier tier(persistent);
    tier.transferLimit = 64;
    tier.capacity = 128;
    engine::Cache cache(pool, &tier, test::ampleDiskBudget());
    Executor executor;
    executor.keptTier = std::make_shared<OffloadControl>();
    executor.keptTier->ready = true;
    executor.decodeFinishes = false;
    Events events;
    const EngineConfig config = test::engineConfig({.maxContext = 102400});
    engine::Engine engine(config, cache, executor, events);
    guardReleases(storage, engine);
    runShortOfMemory(cache, tier, *config.governor);
    const CacheSnapshot before = cache.snapshot();
    const uint32_t writes = tier.demotions;
    EngineRequest decoding = request(1, std::vector<uint32_t>(2060, 7));
    decoding.maxNewTokens = 10'000;
    decoding.deadlineMilliseconds = 1e9;
    engine.submit(std::move(decoding));
    double now = 1;
    tickUntil(engine, now, [&] { return events.outputs.contains(1); }, "the request did not decode");
    now += WriteBehind::kDelayMilliseconds;
    for (uint32_t step = 0; step < 4; ++step)
      static_cast<void>(engine.tick(now++));
    require(engine.snapshot().writeBehind.waiting == 1 && !events.completedCount &&
                tier.demotions == writes + (persistent ? 16u : 0u),
            persistent ? "a persistent tier did not write a point in use"
                       : "a temporary tier wrote a point its running request holds");
    engine.cancel(1);
    tickUntil(engine, now, [&] { return idle(engine); }, "the request did not end");
    if (!persistent) {
      EngineRequest ahead = request(2, std::vector<uint32_t>(16384, 9));
      ahead.deadlineMilliseconds = 1e9;
      engine.submit(std::move(ahead));
    }
    tickBesideALaneAhead(engine, tier, now, [&] { return engine.snapshot().writeBehind.durable == 1; });
    const CacheSnapshot copied = cache.snapshot();
    require(engine.snapshot().writeBehind.durable == 1 && tier.demotions == writes + 64 &&
                copied.kvTier.copies == 64 &&
                copied.stateCache.diskBytes == before.stateCache.diskBytes + 64,
            "the finished conversation's point was not copied behind it");
    if (!persistent) {
      engine.cancel(2);
      tickUntil(engine, now, [&] { return idle(engine); }, "the lane ahead did not end");
      continue;
    }
    const bool reclaimed = test::releaseStep(cache, ReclaimClass::Ordinary).gave();
    while (test::releaseStep(cache, ReclaimClass::Ordinary).gave()) {
    }
    const CacheSnapshot left = cache.snapshot();
    require(reclaimed && !cache.transfersInFlight() && tier.demotions == writes + 64 &&
                left.kvTier.demotions == before.kvTier.demotions && !left.pool.pagesPrefix &&
                !left.stateCache.bytes &&
                left.stateCache.diskBytes == before.stateCache.diskBytes + 64 &&
                left.kvTier.diskBlocks == before.kvTier.diskBlocks + 64,
            "a reclaim of the finished conversation wrote it or waited for a write");
  }
}

// Copies ahead start nothing while a lane refused beside others waits for memory on its way back,
// and go on once it has its page.
void testCopiesAheadWaitForALaneRefusedBesideOthers() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier(true);
  tier.transferLimit = 64;
  tier.capacity = 256;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.keptTier = std::make_shared<OffloadControl>();
  executor.keptTier->ready = true;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // The leaf the refusal demotes, under a state on the SSD.
  cacheBlocksUnderDiskStates(cache, 1);
  engine.submit(request(9, std::vector<uint32_t>(2060, 9)));
  double now = 1;
  tickUntil(engine, now, [&] { return idle(engine); }, "the conversation did not finish");
  const std::optional<double> due = engine.nextWakeupMilliseconds();
  require(due && engine.snapshot().writeBehind.waiting == 1 && !tier.demotions,
          "the finished conversation's point did not wait for its delay");
  // Lane 1's decode needs a fourth page two tokens in, lane 2's 24 tokens in.
  executor.decodeFinishes = false;
  const auto promptTokens = [](uint64_t id) { return id == 1 ? 87U : 65U; };
  for (const uint32_t id : {1U, 2U}) {
    auto decoding = request(id, std::vector<uint32_t>(promptTokens(id), id));
    decoding.maxNewTokens = 1000;
    decoding.deadlineMilliseconds = 1e9;
    engine.submit(std::move(decoding));
  }
  tickUntil(engine, now, [&] { return events.outputs.contains(1) && events.outputs.contains(2); },
            "the lanes did not decode");
  const auto needsPage = [&](uint64_t id) {
    return promptTokens(id) + events.outputs[id].size() + model::ExecutionLimits::targetVerifyRows >
           uint64_t{cache.pageTable(id).pages.size()} * KvCache::pageTokens;
  };
  tickUntil(engine, now, [&] { return !engine.commandInFlight() && needsPage(1) && !needsPage(2); },
            "lane 1 did not come to need a page that lane 2 does not");
  require(now < *due, "the point fell due before the lanes decoded");
  storage.growthBlocked = true;
  const KvPageAcquisition held = pool.acquirePages(pool.freePageCount());
  static_cast<void>(engine.tick(now++));
  const uint32_t demotions = tier.demotions;
  require(held.granted() && engine.commandInFlight() && demotions == 1 && !executor.suspensions,
          "lane 1 was not refused its page beside lane 2 while the leaf's demotion is in flight");
  static_cast<void>(engine.tick(*due));
  const std::optional<double> wakeup = engine.nextWakeupMilliseconds();
  require(tier.demotions == demotions && engine.snapshot().writeBehind.waiting == 1 &&
              (!wakeup || *wakeup > *due),
          "the point was copied, or woke the engine, while a lane beside others waited for memory");
  const size_t decoded = events.outputs[1].size();
  tier.complete();
  now = *due + 1;
  tickUntil(engine, now, [&] { return tier.demotions != demotions; }, "the point was not copied");
  require(events.outputs[1].size() > decoded, "the point was copied before the lane had its page");
  for (const uint32_t page : held.pages)
    pool.releasePage(page, false);
  storage.growthBlocked = false;
  executor.decodeFinishes = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "the lanes did not finish");
  require(events.failedCount == 0, "a lane failed");
}

// Copies ahead wait for memory on its way back to a refused lane no longer than the resource
// wait's limit: once the demotion lands past it, a due point's copy starts before the lane decodes.
void testCopiesAheadWaitNoLongerThanTheWaitsLimit() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier(true);
  tier.transferLimit = 64;
  tier.capacity = 256;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.keptTier = std::make_shared<OffloadControl>();
  executor.keptTier->ready = true;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // The leaf the refusal demotes, under a state on the SSD.
  cacheBlocksUnderDiskStates(cache, 1);
  engine.submit(request(9, std::vector<uint32_t>(2060, 9)));
  double now = 1;
  tickUntil(engine, now, [&] { return idle(engine); }, "the conversation did not finish");
  const std::optional<double> due = engine.nextWakeupMilliseconds();
  require(due && engine.snapshot().writeBehind.waiting == 1 && !tier.demotions,
          "the finished conversation's point did not wait for its delay");
  // Lane 1's decode needs a fourth page two tokens in, lane 2's 24 tokens in.
  executor.decodeFinishes = false;
  const auto promptTokens = [](uint64_t id) { return id == 1 ? 87U : 65U; };
  for (const uint32_t id : {1U, 2U}) {
    auto decoding = request(id, std::vector<uint32_t>(promptTokens(id), id));
    decoding.maxNewTokens = 1000;
    decoding.deadlineMilliseconds = 1e9;
    engine.submit(std::move(decoding));
  }
  tickUntil(engine, now, [&] { return events.outputs.contains(1) && events.outputs.contains(2); },
            "the lanes did not decode");
  const auto needsPage = [&](uint64_t id) {
    return promptTokens(id) + events.outputs[id].size() + model::ExecutionLimits::targetVerifyRows >
           uint64_t{cache.pageTable(id).pages.size()} * KvCache::pageTokens;
  };
  tickUntil(engine, now, [&] { return !engine.commandInFlight() && needsPage(1) && !needsPage(2); },
            "lane 1 did not come to need a page that lane 2 does not");
  require(now < *due, "the point fell due before the lanes decoded");
  storage.growthBlocked = true;
  const KvPageAcquisition held = pool.acquirePages(pool.freePageCount());
  const double refusedAt = now;
  static_cast<void>(engine.tick(now++));
  const uint32_t demotions = tier.demotions;
  const size_t decoded = events.outputs[1].size();
  require(held.granted() && engine.commandInFlight() && demotions == 1 && !executor.suspensions,
          "lane 1 was not refused its page beside lane 2 while the leaf's demotion is in flight");
  static_cast<void>(engine.tick(*due));
  const std::optional<double> wakeup = engine.nextWakeupMilliseconds();
  require(tier.demotions == demotions && engine.snapshot().writeBehind.waiting == 1 &&
              (!wakeup || *wakeup > *due),
          "the point was copied, or woke the engine, while a lane beside others waited for memory");
  static_cast<void>(engine.tick(refusedAt + kResourceWaitTimeoutMilliseconds - 1));
  require(tier.demotions == demotions, "the point was copied while the demotion was in flight");
  // The demotion lands past the limit.
  tier.complete();
  static_cast<void>(engine.tick(refusedAt + kResourceWaitTimeoutMilliseconds + 1));
  require(tier.demotions > demotions && events.outputs[1].size() == decoded,
          "the point waited past the limit of a wait for memory on its way back");
  for (const uint32_t page : held.pages)
    pool.releasePage(page, false);
  storage.growthBlocked = false;
  executor.decodeFinishes = true;
  now = refusedAt + kResourceWaitTimeoutMilliseconds + 2;
  tickUntil(engine, now, [&] { return idle(engine); }, "the lanes did not finish");
  require(events.failedCount == 0, "a lane failed");
}

// A refused lane below the priority the scheduler plans holds no copy ahead back.
void testCopiesAheadDoNotWaitForALaneBelowThePriorityThatRuns() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier(true);
  tier.transferLimit = 64;
  tier.capacity = 256;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.keptTier = std::make_shared<OffloadControl>();
  executor.keptTier->ready = true;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // The leaf the refusal demotes, under a state on the SSD.
  cacheBlocksUnderDiskStates(cache, 1);
  engine.submit(request(9, std::vector<uint32_t>(2060, 9)));
  double now = 1;
  tickUntil(engine, now, [&] { return idle(engine); }, "the conversation did not finish");
  const std::optional<double> due = engine.nextWakeupMilliseconds();
  require(due && engine.snapshot().writeBehind.waiting == 1 && !tier.demotions,
          "the finished conversation's point did not wait for its delay");
  executor.decodeFinishes = false;
  const auto decoding = [&](uint64_t id, uint32_t tokens, RequestPriority priority) {
    auto value = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.priority = priority;
    value.maxNewTokens = 1000;
    value.deadlineMilliseconds = 1e9;
    engine.submit(std::move(value));
  };
  decoding(1, 87, RequestPriority::Normal);
  tickUntil(engine, now, [&] { return events.outputs.contains(1); }, "the Normal lane did not decode");
  const auto needsPage = [&] {
    return 87 + events.outputs[1].size() + model::ExecutionLimits::targetVerifyRows >
           uint64_t{cache.pageTable(1).pages.size()} * KvCache::pageTokens;
  };
  tickUntil(engine, now, [&] { return !engine.commandInFlight() && needsPage(); },
            "the Normal lane did not come to need a page");
  storage.growthBlocked = true;
  const KvPageAcquisition held = pool.acquirePages(pool.freePageCount());
  static_cast<void>(engine.tick(now++));
  require(held.granted() && tier.demotions == 1 && !engine.commandInFlight(),
          "the Normal lane was not refused alone while the leaf's demotion is in flight");
  for (const uint32_t page : held.pages)
    pool.releasePage(page, false);
  storage.growthBlocked = false;
  decoding(2, 9, RequestPriority::Foreground);
  tier.complete();
  const size_t decoded = events.outputs[1].size();
  tickUntil(engine, now, [&] { return events.outputs[2].size() >= 2; },
            "the Foreground request did not decode");
  require(now < *due && events.outputs[1].size() == decoded && tier.demotions == 1,
          "the Normal lane decoded beside the Foreground request, or the point was copied early");
  static_cast<void>(engine.tick(*due));
  require(tier.demotions > 1, "the point waited for a lane of a priority the scheduler does not plan");
  executor.decodeFinishes = true;
  now = *due + 1;
  tickUntil(engine, now, [&] { return idle(engine); }, "the lanes did not finish");
  require(events.failedCount == 0, "a lane failed");
}

void publishDiskState(engine::Cache &cache, const std::vector<uint32_t> &prompt);

// A refusal after progress restarts the lane's wait limit, also while lanes beside it run.
void testARefusalAfterProgressRestartsTheWaitLimit() {
  constexpr double kLimit = 400;
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400, .resourceWaitTimeoutMilliseconds = kLimit}),
                        cache, executor, events);
  guardReleases(storage, engine);
  // The leaf a refusal demotes, under a state on the SSD, and a later
  // prefix whose state is on the SSD too.
  cacheBlocksUnderDiskStates(cache, 1);
  const std::vector<uint32_t> prefix(65, 2);
  publishDiskState(cache, prefix);
  executor.restoreControl->ready = true;
  auto first = request(1, std::vector<uint32_t>(87, 1));
  first.maxNewTokens = 1000;
  first.deadlineMilliseconds = 1e9;
  engine.submit(std::move(first));
  double now = 1;
  tickUntil(engine, now, [&] { return events.outputs.contains(1); }, "the lane did not decode");
  const auto needsPage = [&] {
    return 87 + events.outputs[1].size() + model::ExecutionLimits::targetVerifyRows >
           uint64_t{cache.pageTable(1).pages.size()} * KvCache::pageTokens;
  };
  tickUntil(engine, now, [&] { return !engine.commandInFlight() && needsPage(); },
            "the lane did not come to need a page");
  storage.growthBlocked = true;
  std::vector<uint32_t> held = pool.acquirePages(pool.freePageCount()).pages;
  const double refused = now;
  static_cast<void>(engine.tick(now));
  require(tier.demotions == 1 && !engine.commandInFlight(), "the lane was not refused alone beside a demotion");
  // A later request restores that prefix into a page growth gives it, the
  // rest of which is held, and runs beside the lane within that page: it
  // yields before the lane, so it takes no page past it while the lane
  // waits for the demotion (prepare).
  auto later = request(2, prefix);
  later.maxNewTokens = 1000;
  later.deadlineMilliseconds = 1e9;
  engine.submit(std::move(later));
  const double step = kLimit / 8;
  now += step;
  storage.growthBlocked = false;
  static_cast<void>(engine.tick(now));
  storage.growthBlocked = true;
  const std::vector<uint32_t> spare = pool.acquirePages(pool.freePageCount()).pages;
  held.insert(held.end(), spare.begin(), spare.end());
  tickUntil(engine, now, [&] { return events.outputs.contains(2) && engine.commandInFlight(); },
            "the later request did not run beside the waiting lane");
  // The demotion lands halfway through the limit, and the page it frees
  // goes elsewhere.
  tier.complete();
  const double landed = refused + kLimit / 2;
  now = std::max(now, landed);
  static_cast<void>(engine.tick(now));
  const std::vector<uint32_t> taken = pool.acquirePages(pool.freePageCount()).pages;
  held.insert(held.end(), taken.begin(), taken.end());
  for (; now < landed + kLimit + 2 * step && !events.failedCount; now += step) {
    static_cast<void>(engine.tick(now));
    require(!events.failedCount || now >= landed + kLimit, "the lane ended before its restarted limit");
  }
  require(events.failures == std::vector<std::string>{"resource_timeout"} && now >= landed + kLimit &&
              now > refused + kLimit + step,
          "the lane refused beside a running one after progress did not end at its restarted limit");
  for (const uint32_t page : held)
    pool.releasePage(page, false);
  storage.growthBlocked = false;
  engine.cancel(2);
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
}

// A suspended request held behind another's refusal during a recovery holds no copy ahead back.
void testCopiesAheadDoNotWaitForARequestHeldBehindARefusal() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier(true);
  tier.transferLimit = 64;
  tier.capacity = 512;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.keptTier = std::make_shared<OffloadControl>();
  executor.keptTier->ready = true;
  executor.decodeFinishes = false;
  executor.beginGrowthBlocked = [&] { return executor.lastBeginId == 6; };
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // The leaf whose demotion the resume waits for, under a state on the SSD.
  cacheBlocksUnderDiskStates(cache, 1);
  const auto lane = [&](uint64_t id, uint32_t tokens, RequestPriority priority, uint32_t output = 1000) {
    auto decoding = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    decoding.priority = priority;
    decoding.maxNewTokens = output;
    decoding.deadlineMilliseconds = 1e9;
    engine.submit(std::move(decoding));
  };
  // A Normal lane, then three Foreground ones beside it, the first with a
  // point that waits for its delay, fill the lanes; a Foreground arrival
  // suspends the Normal one.
  lane(4, 5, RequestPriority::Normal);
  double now = 1;
  tickUntil(engine, now, [&] { return events.outputs.contains(4); }, "the Normal lane did not decode");
  lane(1, 2060, RequestPriority::Foreground);
  lane(2, 9, RequestPriority::Foreground);
  lane(3, 9, RequestPriority::Foreground);
  tickUntil(engine, now, [&] { return events.outputs.contains(1) && events.outputs.contains(3); },
            "the Foreground lanes did not decode");
  const std::optional<double> due = engine.nextWakeupMilliseconds();
  require(due && engine.snapshot().writeBehind.waiting == 1, "the first lane's point did not wait for its delay");
  lane(5, 9, RequestPriority::Foreground, 2);
  tickUntil(engine, now, [&] { return events.usage.contains(5) && !engine.commandInFlight(); },
            "the arrival did not finish");
  require(engine.snapshot().prioritySuspensions == 1 && !executor.requests.at(4).resident,
          "the arrival did not suspend the Normal lane");
  // Its resume is refused its page while the leaf's demotion it started is
  // in flight, and a Foreground request arrives, which recovery tries first.
  storage.growthBlocked = true;
  const std::vector<uint32_t> held = pool.acquirePages(pool.freePageCount()).pages;
  tickUntil(engine, now, [&] { return tier.inFlight() && !engine.commandInFlight(); },
            "the resume did not wait for the leaf's demotion");
  for (const uint32_t page : held)
    pool.releasePage(page, false);
  storage.growthBlocked = false;
  lane(6, 9, RequestPriority::Foreground);
  // The demotion and what the Foreground request's refusals write land; it
  // is refused with nothing on its way back, and the resume is held behind it.
  for (uint32_t step = 0; step < 32; ++step) {
    tier.complete();
    static_cast<void>(engine.tick(now++));
  }
  require(now < *due && !tier.inFlight() && !executor.requests.at(4).resident && !events.starts.empty() &&
              events.startIds.back() != 6 && engine.resourceWaitSnapshot(now).heldBehindRefusal == 1,
          "the resume was not held behind the Foreground request's refusal");
  const uint32_t demotions = tier.demotions;
  static_cast<void>(engine.tick(*due));
  require(tier.demotions > demotions, "the point waited for a request held behind a refusal");
  executor.beginGrowthBlocked = {};
  executor.decodeFinishes = true;
  now = *due + 1;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
}

// A lane refused memory during a recovery, its priority running, holds copies ahead back until it
// has its page.
void testCopiesAheadWaitForALaneRefusedDuringARecovery() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier(true);
  tier.transferLimit = 64;
  tier.capacity = 256;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.keptTier = std::make_shared<OffloadControl>();
  executor.keptTier->ready = true;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // The leaf the refusal demotes, under a state on the SSD.
  cacheBlocksUnderDiskStates(cache, 1);
  engine.submit(request(9, std::vector<uint32_t>(2060, 9)));
  double now = 1;
  tickUntil(engine, now, [&] { return idle(engine); }, "the conversation did not finish");
  const std::optional<double> due = engine.nextWakeupMilliseconds();
  require(due && engine.snapshot().writeBehind.waiting == 1 && !tier.demotions,
          "the finished conversation's point did not wait for its delay");
  executor.decodeFinishes = false;
  const auto promptTokens = [](uint64_t id) { return id == 1 ? 87U : id == 2 ? 65U : 9U; };
  const auto decoding = [&](uint64_t id, RequestPriority priority, uint32_t output) {
    auto value = request(id, std::vector<uint32_t>(promptTokens(id), static_cast<uint32_t>(id)));
    value.priority = priority;
    value.maxNewTokens = output;
    value.deadlineMilliseconds = 1e9;
    engine.submit(std::move(value));
  };
  for (const uint32_t id : {1U, 2U, 3U, 4U})
    decoding(id, RequestPriority::Normal, 1000);
  tickUntil(engine, now, [&] { return events.outputs.contains(1) && events.outputs.contains(4); },
            "the Normal lanes did not decode");
  decoding(5, RequestPriority::Foreground, 2);
  tickUntil(engine, now, [&] { return engine.snapshot().prioritySuspensions != 0; },
            "the Foreground arrival suspended no Normal lane");
  executor.resumeDenied = true;
  tickUntil(engine, now, [&] { return events.usage.contains(5) && executor.resumeAttempts != 0; },
            "the arrival did not finish, or the suspended lane did not try to resume");
  require(engine.snapshot().prioritySuspensions == 1 && !executor.requests.at(4).resident,
          "the arrival did not suspend lane 4, or it resumed");
  const auto needsPage = [&](uint64_t id) {
    return promptTokens(id) + events.outputs[id].size() + model::ExecutionLimits::targetVerifyRows >
           uint64_t{cache.pageTable(id).pages.size()} * KvCache::pageTokens;
  };
  tickUntil(engine, now, [&] { return !engine.commandInFlight() && needsPage(1) && !needsPage(2); },
            "lane 1 did not come to need a page that lane 2 does not");
  require(now < *due, "the point fell due before lane 1 needed its page");
  storage.growthBlocked = true;
  const KvPageAcquisition held = pool.acquirePages(pool.freePageCount());
  static_cast<void>(engine.tick(now++));
  const uint32_t demotions = tier.demotions;
  require(held.granted() && engine.commandInFlight() && demotions == 1 && executor.suspensions == 1,
          "lane 1 was not refused its page beside lane 2 while the leaf's demotion is in flight");
  static_cast<void>(engine.tick(*due));
  require(tier.demotions == demotions && engine.snapshot().writeBehind.waiting == 1,
          "the point was copied while a lane refused during a recovery waited for memory");
  const size_t decoded = events.outputs[1].size();
  tier.complete();
  now = *due + 1;
  tickUntil(engine, now, [&] { return tier.demotions != demotions; }, "the point was not copied");
  require(events.outputs[1].size() > decoded, "the point was copied before the lane had its page");
  for (const uint32_t page : held.pages)
    pool.releasePage(page, false);
  storage.growthBlocked = false;
  executor.resumeDenied = false;
  executor.decodeFinishes = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "the lanes did not finish");
  require(events.failedCount == 0, "a lane failed");
}

// With the tier unwritable and nothing in flight, a lane short of pages is answered, not parked.
void testNothingInFlightIsNotPending() {
  test::TestKvStorage storage(12, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.writableFile = false;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  for (uint64_t id = 900; id < 908; ++id) {
    std::vector<uint32_t> prompt(32, static_cast<uint32_t>(id));
    cache.beginRequest(id);
    if (!cache.ensureTokens(id, 32).granted()) {
      cache.endRequest(id);
      break;
    }
    cache.publishCommittedBlocks(id, prompt, 32, {});
    cache.endRequest(id);
  }
  engine.submit(request(1, std::vector<uint32_t>(97, 7)));
  runUntilIdle(engine);
  require(idle(engine) && events.outputs.contains(1) && events.failedCount == 0 &&
              tier.demotions == 0 && executor.prefillRows == 97,
          "the lane did not run on leaves the unwritable tier had to drop");
}

// One pass demotes as many leaves as the page shortfall needs, and the lane waits for all of them
// without demoting more.
void testPageShortfallDemotesInBulk() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  tier.capacity = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  cacheBlocksUnderDiskStates(cache, 6);
  // The cached blocks fill two extents but two of their pages: those two
  // are free, no other extent is allocated, and the budget admits nothing
  // more. The KV up to the replay boundary, 160 tokens, falls three pages
  // short; the last token one more.
  require(pool.freePageCount() == 2 && pool.snapshot().pagesAllocated == 8,
          "fixture storage geometry changed");
  storage.growthBlocked = true;
  engine.submit(request(1, std::vector<uint32_t>(161, 7)));
  static_cast<void>(engine.tick(1));
  require(tier.demotions == 3 && executor.prefillRows == 0 && executor.suspensions == 0 &&
              events.failedCount == 0,
          "a shortfall of three pages did not start three demotions in one pass");
  static_cast<void>(engine.tick(2));
  require(tier.demotions == 3, "waiting demoted more");
  for (uint32_t step = 3; step < 40 && !idle(engine); ++step) {
    tier.complete();
    static_cast<void>(engine.tick(step));
  }
  require(idle(engine) && events.outputs.contains(1) && executor.prefillRows == 161 &&
              executor.suspensions == 0 && events.failedCount == 0 && tier.demotions == 4 &&
              cache.snapshot().kvTier.diskBlocks == 4,
          "the lane did not run on the pages the demotions gave back");
}

// KV growth refused at the engine's limit takes a cached state, whose write frees its buffers at
// once: the lane grows before the write lands.
void testKvGrowthProceedsThroughDemotion() {
  test::TestKvStorage storage(128, State().bytes() / 4, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  test::GovernedHost host;
  engine::Engine engine(host.connect({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  host.bookStates(executor);
  auto transfer = std::make_shared<OffloadControl>();
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 32).granted(), "offload fixture KV failed");
  const auto block = test::publishBlocks(cache, 999, std::vector<uint32_t>(32, 12), 32);
  // The cached state's buffers are the model's, booked as its states.
  host.memory.hold("states", State().bytes());
  executor.heldStateBytes += State().bytes();
  cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer, executor.evictedStateBytes), false);
  cache.endRequest(999);
  {
    auto lookup = cache.lookup(std::vector<uint32_t>(33, 12), {});
    cache.recordLookup(lookup);
  }
  host.leaveRoom(0);
  storage.growthAllowed = [&](uint32_t) {
    if (transfer->released)
      return true;
    storage.allocationFailure = host.reserve(State().bytes()).failure;
    return false;
  };
  engine.submit(request(1, std::vector<uint32_t>(256, 17)));
  runUntilIdle(engine);
  // The demoted state's RAM served KV growth before its write finished.
  require(transfer->released && !transfer->ready && executor.prefillRows == 256 &&
              executor.suspensions == 0 && events.failedCount == 0,
          "KV growth waited for the write or replayed the lane");
  require(cache.snapshot().stateCache.diskBytes == 64, "demoted state lost its disk copy");
  transfer->ready = true;
  static_cast<void>(engine.tick(1000));
  const auto stats = cache.snapshot().stateCache;
  require(stats.offloads == 1 && stats.offloadFailures == 0 && stats.diskBytes == 64,
          "completed write was not consumed");
}

// A request a recovery leaves untried holds no copy ahead back.
void testCopiesAheadDoNotWaitForARequestARecoveryLeavesUntried() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier(true);
  tier.transferLimit = 64;
  tier.capacity = 256;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.keptTier = std::make_shared<OffloadControl>();
  executor.keptTier->ready = true;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // The Normal request's prefix, its state and KV on the SSD alone.
  const std::vector<uint32_t> prefix(65, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prefix, 64));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 2; ++written) {
    require(test::evictStep(cache).gave(),
            "KV block was not written");
    tier.complete();
    require(cache.pollTransfers(), "written block did not land");
  }
  require(test::releaseStep(cache).gave() &&
              storage.allocatedPages() == 0,
          "the prefix did not move to disk whole");
  // The leaves the refusal demotes, under states on the SSD.
  cacheBlocksUnderDiskStates(cache, 3);
  const uint32_t written = tier.demotions;
  engine.submit(request(9, std::vector<uint32_t>(2060, 9)));
  double now = 1;
  tickUntil(engine, now, [&] { return idle(engine); }, "the conversation did not finish");
  const std::optional<double> due = engine.nextWakeupMilliseconds();
  require(due && engine.snapshot().writeBehind.waiting == 1 && tier.demotions == written,
          "the finished conversation's point did not wait for its delay");
  executor.decodeFinishes = false;
  const auto decoding = [&](uint64_t id, const std::vector<uint32_t> &prompt, RequestPriority priority) {
    auto value = request(id, prompt);
    value.priority = priority;
    value.maxNewTokens = 1000;
    value.deadlineMilliseconds = 1e9;
    engine.submit(std::move(value));
  };
  for (const uint32_t id : {1U, 2U, 3U})
    decoding(id, std::vector<uint32_t>(9, id), RequestPriority::Normal);
  tickUntil(engine, now, [&] { return events.outputs.contains(3) && !engine.commandInFlight(); },
            "the Normal lanes did not decode");
  storage.growthBlocked = true;
  const KvPageAcquisition held = pool.acquirePages(pool.freePageCount());
  decoding(4, prefix, RequestPriority::Normal);
  static_cast<void>(engine.tick(now++));
  require(held.granted() && tier.demotions > written && cache.transfersInFlight() &&
              engine.resourceWaitSnapshot(now).memory == 1 && !executor.requests.contains(4),
          "the Normal request was not refused its pages while the leaves' demotions are in flight");
  for (const uint32_t page : held.pages)
    pool.releasePage(page, false);
  storage.growthBlocked = false;
  decoding(5, std::vector<uint32_t>(9, 5), RequestPriority::Foreground);
  tickUntil(engine, now, [&] { return executor.requests.contains(5); },
            "the first Foreground arrival did not take the last lane");
  decoding(6, std::vector<uint32_t>(9, 6), RequestPriority::Foreground);
  tickUntil(engine, now, [&] { return engine.snapshot().prioritySuspensions != 0; },
            "the second Foreground arrival suspended no Normal lane");
  require(engine.snapshot().prioritySuspensions == 1 && cache.transfersInFlight() &&
              !executor.requests.contains(4),
          "the second Foreground arrival suspended more than a lane, or the demotions landed meanwhile");
  tier.complete();
  tickUntil(engine, now, [&] { return events.outputs[6].size() >= 2; },
            "the second Foreground arrival did not decode");
  const uint32_t demotions = tier.demotions;
  require(now < *due && !cache.transfersInFlight() && !executor.requests.contains(4),
          "the demotions did not land, or the Normal request started, before the point's time");
  static_cast<void>(engine.tick(*due));
  require(tier.demotions > demotions, "the point waited for a request the recovery leaves untried");
  executor.decodeFinishes = true;
  executor.restoreControl->ready = true;
  now = *due + 1;
  for (uint32_t step = 0; step < 1000 && !idle(engine); ++step) {
    tier.complete();
    static_cast<void>(engine.tick(now++));
  }
  require(idle(engine) && events.failedCount == 0, "a request failed, or the engine did not reach idle");
}

void publishDiskState(engine::Cache &cache, const std::vector<uint32_t> &prompt) {
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "disk fixture KV allocation failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 64));
  cache.endRequest(999);
}

// A prompt refused what nothing returns, beside a lane waiting for a page on its way back, takes
// none of it and yields its lane; the page pays the waiting lane first, not a start arriving then.
void testABatchRefusalOfNothingOnItsWayBackHoldsNoOne() {
  test::TestKvStorage storage(256, 4096, 1);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  const EngineConfig config = test::engineConfig({.maxContext = 102400});
  engine::Engine engine(config, cache, executor, events);
  guardReleases(storage, engine);
  // A cached block under a state on disk, whose page frees once its copy is
  // written, and a later prefix whose state is on disk.
  cacheBlocksUnderDiskStates(cache, 1);
  const std::vector<uint32_t> prefix(65, 17);
  publishDiskState(cache, prefix);
  const auto prompt = [](uint64_t id, std::vector<uint32_t> tokens) {
    EngineRequest result = request(id, tokens);
    result.maxNewTokens = 1000;
    result.deadlineMilliseconds = 1e9;
    return result;
  };
  const auto pages = [&](uint64_t id) { return cache.pageTable(id).pages.size(); };
  const auto outputs = [&](uint64_t id) {
    return events.outputs.contains(id) ? events.outputs.at(id).size() : 0;
  };
  const auto waiting = [&] { return config.governor->snapshot().claims.waiting; };
  // The lane beside it holds four pages; the waiting lane holds two, which
  // it fills first.
  engine.submit(prompt(1, std::vector<uint32_t>(97, 1)));
  engine.submit(prompt(2, std::vector<uint32_t>(50, 2)));
  double now = 1;
  tickUntil(engine, now, [&] { return outputs(1) && outputs(2); }, "the lanes did not decode");
  // Memory holds what the lanes and the cache have now. Ticks are 150 ms
  // apart, so the waiting lane retries at each.
  storage.budgetPages = storage.allocatedPages();
  for (uint32_t step = 0; step < 100 && !tier.demotions; ++step)
    static_cast<void>(engine.tick(now += 150));
  require(tier.demotions == 1 && pages(2) == 2 && waiting() == 1,
          "the lane did not wait on its claim for a page on its way back");
  engine.submit(prompt(3, std::vector<uint32_t>(4097, 3)));
  for (uint32_t step = 0; step < 4; ++step)
    static_cast<void>(engine.tick(now += 150));
  require(!executor.requests.at(3).resident && engine.snapshot().resourceSuspensions == 1 && pages(2) == 2 &&
              waiting() == 1,
          "the prompt refused what nothing returns took the page held for the waiting lane, waited, or did "
          "not yield its lane");
  // The page lands as the start arrives.
  std::vector<uint32_t> restoring = prefix;
  restoring.insert(restoring.end(), 30, 4);
  engine.submit(prompt(4, restoring));
  tier.complete();
  static_cast<void>(engine.tick(now += 150));
  require(pages(2) == 3 && !waiting() && !executor.requests.contains(4) &&
              engine.snapshot().resourceSuspensions == 1,
          "the landed page did not pay the lane that waited for it first");
  executor.decodeFinishes = true;
  executor.restoreControl->ready = true;
  storage.budgetPages = std::numeric_limits<uint32_t>::max();
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.failedCount == 0 && events.completedCount == 4, "the requests did not complete");
}

// A disk restore holds its lane until the read lands, and completion, cancellation, expiry, a
// failed read or a read at the deadline each end without leaks or output.
void testAsyncRestoreLifecycle() {
  for (int outcome = 0; outcome < 5; ++outcome) {
    test::TestKvStorage storage(128, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
    guardReleases(storage, engine);
    std::vector<uint32_t> prompt(65, 17);
    publishDiskState(cache, prompt);
    engine.submit(request(1, prompt));
    static_cast<void>(engine.tick(1));
    require(executor.diskReads == 1 && executor.prefillRows == 0 && events.starts.empty(),
            "disk restore ran before IO completion");
    require(cache.snapshot().stateCache.pinned == 1 && executor.requests.contains(1),
            "restore did not retain admitted state and KV");
    // A separate lane can run while this restore is pending.
    engine.submit(request(2, std::vector<uint32_t>(3, 99)));
    for (int tick = 2; tick < 8; ++tick) static_cast<void>(engine.tick(tick));
    require(events.outputs.contains(2), "restore blocked independent model work");
    if (outcome == 1) engine.cancel(1);
    if (outcome == 2) {
      // Expired, it keeps its waiting phase until the read drains.
      static_cast<void>(engine.tick(10001));
      const auto &active = engine.snapshot().activeRequests;
      require(active.size() == 1 && active.front().id == 1 &&
                  active.front().phase != engine::Phase::Failed && events.failures.empty(),
              "an expired restore failed before its read drained");
    }
    if (outcome == 3) executor.restoreControl->success = false;
    require(executor.requests.contains(1), "cancellation reused IO destination before drain");
    executor.restoreControl->ready = true;
    // The read lands at the tick the deadline passes: the request ends
    // without starting on it.
    if (outcome == 4) static_cast<void>(engine.tick(10001));
    runUntilIdle(engine);
    require(cache.snapshot().stateCache.pinned == 0 && cache.snapshot().activeRequests == 0 &&
                executor.requests.empty(), "restore leaked active resources");
    if (outcome == 0)
      require(executor.restored == 64 && executor.prefillRows == 4, "disk hit recomputed prefix");
    else if (outcome == 3)
      require(executor.restored == 0 && executor.prefillRows == 68 && executor.diskReads == 1,
              "failed read did not fall back once to cold prefill");
    else
      require(executor.restored == 0 && !events.outputs.contains(1),
              "cancelled or expired restore emitted output");
    if (outcome == 2 || outcome == 4)
      require(events.failures == std::vector<std::string>{"deadline_exceeded"} &&
                  events.failureDetails.front().first == "request deadline exceeded" &&
                  std::ranges::find(events.startIds, 1) == events.startIds.end(),
              "an expired restore lost its outcome or message, or started");
  }
}

// A stop of a lane that has output and is restoring its state from disk
// waits for the read to drain, then finishes as Stop.
void testStopWhileRestoringFinishesAfterTheRestore() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  // The prompt's state stays on disk only: no lane captures another, and the
  // restore keeps no copy in memory, so the lane's resumption reads it again.
  executor.deniedReservations = std::numeric_limits<uint32_t>::max();
  executor.restoreControl->ready = true;
  executor.restoreControl->promotionDenied = true;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 17);
  publishDiskState(cache, prompt);
  // Lane 1, with the shortest prompt the least work, yields its lane to a
  // Foreground arrival once all the lanes are full.
  for (uint64_t id = 1; id <= 4; ++id) {
    EngineRequest lane = request(id, id == 1 ? prompt : std::vector<uint32_t>(129, id));
    lane.maxNewTokens = 1000;
    engine.submit(std::move(lane));
  }
  double now = 1;
  tickUntil(engine, now,
            [&] {
              return std::ranges::all_of(std::array<uint64_t, 4>{1, 2, 3, 4},
                                         [&](uint64_t id) { return !events.outputs[id].empty(); });
            },
            "the lanes did not decode");
  // Once the arrival finishes, lane 1 resumes into a restore that is held:
  // the second read of the only state on disk.
  executor.restoreControl = std::make_shared<RestoreControl>();
  executor.restoreControl->promotionDenied = true;
  EngineRequest arriving = request(5, std::vector<uint32_t>(9, 5));
  arriving.priority = RequestPriority::Foreground;
  engine.submit(std::move(arriving));
  tickUntil(engine, now, [&] { return events.usage.contains(5) && executor.diskReads == 2; },
            "the suspended lane did not resume into a disk restore");
  require(engine.snapshot().prioritySuspensions == 1 && !events.reasons.contains(1),
          "the arrival did not suspend the lane");

  engine.stop(1);
  static_cast<void>(engine.tick(now++));
  require(!events.reasons.contains(1) && executor.restoreControl->cancelled,
          "a stop while restoring finished before the restore drained");
  executor.restoreControl->ready = true;
  tickUntil(engine, now, [&] { return events.reasons.contains(1); }, "the stopped request did not end");
  for (uint64_t id = 2; id <= 4; ++id)
    engine.cancel(id);
  tickUntil(engine, now, [&] { return idle(engine); }, "the engine did not become idle");

  const EngineSnapshot counts = engine.snapshot();
  require(events.reasons.at(1) == EngineFinishReason::Stop && counts.completed == 2 &&
              counts.cancelled == 3 && counts.failed == 0,
          "a stop while restoring did not finish as Stop after the restore");
}

// A disk hit refused memory drops its own pin and activates without the prefix, then finds the
// state again and reads it only once it holds its lane.
void testDiskHitWithNoActiveMemory() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65, 17);
  publishDiskState(cache, prompt);
  executor.restoreControl->ready = true;
  std::vector<uint32_t> activations;
  executor.beginObserver = [&] {
    require(!executor.diskReads, "the disk was read before the request held its lane");
    activations.push_back(executor.lastRestoredTokens);
  };
  executor.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
  executor.beginGrowthBlocked = [&] { return executor.lastRestoredTokens != 0; };
  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  require(activations == std::vector<uint32_t>{64, 0} && executor.diskReads == 1 && executor.restored == 64 &&
              executor.prefillRows == 1 && events.failedCount == 0 && events.completedCount == 1,
          "disk hit bypassed admission or deadlocked on its own pin");
}

// A prefix hit on disk again and again is promoted to memory: the disk is read once.
void testRepeatedDiskHitPromotesToMemory() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65, 17);
  publishDiskState(cache, prompt);
  executor.restoreControl->ready = true;
  for (uint64_t id = 1; id <= 4; ++id) {
    engine.submit(request(id, prompt));
    runUntilIdle(engine);
  }
  require(executor.diskReads == 1 && cache.snapshot().stateCache.promotions == 1 &&
              cache.snapshot().lookup.stateDiskHits == 1 && events.failedCount == 0,
          "hot restored prefix continued to read from disk");
}

// A failed disk read falls back to the next match, a shallower state in RAM, not a cold start.
void testFailedDiskRestoreKeepsShallowerState() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  cache.publishCommittedBlocks(999, prompt, 64, {});
  demoteState(cache, cache.blockAt(999, 64));
  cache.publishCompositeState(cache.blockAt(999, 32), std::make_shared<State>(), false);
  cache.endRequest(999);
  executor.restoreControl->ready = true;
  executor.restoreControl->success = false;
  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  require(executor.diskReads == 1 && events.failedCount == 0 && executor.restored == 32 &&
              executor.prefillRows == 33,
          "failed disk restore discarded the shallower RAM state");
}

// A prefix whose state and KV are on disk comes back, state and pages, before the lane runs.
void testDiskKvPrefixIsRestoredBeforeTheLaneRuns() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65, 17);
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  const auto block = test::publishBlocks(cache, 999, prompt, 64);
  cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer), false);
  cache.endRequest(999);
  require(test::evictStep(cache).stateBuffers && cache.pollTransfers(),
          "state was not demoted");
  for (uint32_t written = 1; written <= 2; ++written) {
    require(test::evictStep(cache).gave() && tier.demotions == written,
            "KV block was not written");
    tier.complete();
    require(cache.pollTransfers(), "written block did not land");
  }
  require(cache.snapshot().pool.pagesPrefix == 0 && cache.snapshot().kvTier.diskBlocks == 2,
          "prefix did not move to disk whole");

  engine.submit(request(1, prompt));
  static_cast<void>(engine.tick(1));
  require(executor.diskReads == 1 && tier.restores == 2 && executor.prefillRows == 0 &&
              events.starts.empty(),
          "lane ran before its pages came back");
  // The state read lands first; the lane still waits for its pages.
  executor.restoreControl->ready = true;
  static_cast<void>(engine.tick(2));
  static_cast<void>(engine.tick(3));
  require(executor.prefillRows == 0 && events.starts.empty(),
          "lane ran with pages still on the way");
  tier.complete();
  runUntilIdle(engine);
  require(executor.restored == 64 && executor.prefillRows == 1 && events.starts.size() == 1 &&
              events.starts[0] == 64 && events.failedCount == 0,
          "restored prefix was not used");
  const auto stats = cache.snapshot();
  require(stats.kvTier.restores == 2 &&
              stats.pool.pagesPrefix == 2 && stats.kvTier.diskBlocks == 2,
          "restore accounting is off");
}

// Siblings wait for a producer restoring their shared prefix: the state is read once and the
// shared span prefilled once.
void testSharedPrefillWaitsForARestoringProducer() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // 64 tokens whose state is on disk, 128 more both share, 34 of their own.
  std::vector<uint32_t> prompt(226);
  std::iota(prompt.begin(), prompt.end(), 1);
  publishDiskState(cache, prompt);
  executor.restoreControl->ready = false;
  engine.submit(request(1, prompt));
  std::fill(prompt.begin() + 192, prompt.end(), 7);
  engine.submit(request(2, prompt));
  static_cast<void>(engine.tick(1));
  require(executor.diskReads == 1 && engine.snapshot().scheduler.waitingPrefix == 1,
          "the sibling did not wait for the restoring producer");
  executor.restoreControl->ready = true;
  runUntilIdle(engine);
  require(executor.diskReads == 1 &&
              events.starts.back() == 192 &&
              executor.prefillRows == (192 - 64) + 2 * 34 && events.completedCount == 2,
          "the sibling read the state again or prefilled the shared span");
}

// A cancelled restoring producer publishes nothing; its waiting sibling starts at once and restores
// the prefix itself.
void testCancelledRestoringProducerReleasesItsWaiter() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(226);
  std::iota(prompt.begin(), prompt.end(), 1);
  publishDiskState(cache, prompt);
  const std::shared_ptr<RestoreControl> producerRead = executor.restoreControl;
  engine.submit(request(1, prompt));
  std::fill(prompt.begin() + 192, prompt.end(), 7);
  engine.submit(request(2, prompt));
  static_cast<void>(engine.tick(1));
  require(executor.diskReads == 1 && engine.snapshot().scheduler.waitingPrefix == 1,
          "the sibling did not wait for the restoring producer");
  engine.cancel(1);
  executor.restoreControl = std::make_shared<RestoreControl>();
  static_cast<void>(engine.tick(2));
  require(executor.diskReads == 2 && engine.resourceWaitSnapshot(2).restoring == 2 &&
              executor.requests.contains(1),
          "the sibling waited for a cancelled producer's read to drain");
  producerRead->ready = true;
  executor.restoreControl->ready = true;
  runUntilIdle(engine);
  require(events.starts.back() == 64 &&
              engine.snapshot().cancelled == 1 && events.completedCount == 2,
          "the sibling did not restore the prefix itself");
}

// A restoring start holds back only a later prompt it would still lead; one its first command ends,
// or leaves cheaper, runs during the read.
void testALaterPromptWaitsForARestoringPromptThatLeadsIt() {
  constexpr uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  // The rows the restoring prompt has past the 64 tokens whose state is on
  // disk, the later prompt's, and whether the later one runs during the
  // read: 1500 rows end in one command (beside 640 a plan of both would not
  // hold them), 3100 leave 1052 against 3000, 5200 leave 3152.
  for (const auto &[restoring, later, meanwhile] :
       {std::tuple{640u, 1500u, true}, {3000u, 3100u, true}, {3000u, 5200u, false}}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
    guardReleases(storage, engine);
    std::vector<uint32_t> prompt(64 + restoring);
    std::iota(prompt.begin(), prompt.end(), 1);
    publishDiskState(cache, prompt);
    executor.restoreControl->ready = false;
    engine.submit(request(1, prompt));
    std::vector<uint32_t> second(later);
    std::iota(second.begin(), second.end(), 1u << 20);
    engine.submit(request(2, second));
    double now = 1;
    for (; now <= 6; ++now)
      static_cast<void>(engine.tick(now));
    const std::string sizes = std::to_string(later) + " rows beside " + std::to_string(restoring) + " restoring";
    if (meanwhile)
      require(executor.diskReads == 1 && executor.restored == 0 && events.startIds == std::vector<uint64_t>{2} &&
                  executor.prefillChunks.contains(2),
              "a later prompt of " + sizes + " did not run while the read was in flight");
    else
      require(executor.diskReads == 1 && events.startIds.empty(),
              "a later prompt of " + sizes + " started while a prompt that leads it restored its prefix");
    executor.restoreControl->ready = true;
    // Whether the later prompt started while the restored one had more than
    // its last command left.
    bool early = false;
    tickUntil(
        engine, now,
        [&] {
          if (std::ranges::count(events.startIds, 2u) == 0)
            return false;
          for (const ActiveRequestSnapshot &active : engine.snapshot().activeRequests)
            early = early || (active.id == 1 && active.phase == Phase::Prefill &&
                              active.promptTokens - active.promptProcessed > chunk);
          return true;
        },
        "the later prompt did not start");
    runUntilIdle(engine);
    require((meanwhile || (!early && events.startIds.front() == 1)) && events.completedCount == 2 &&
                events.failedCount == 0,
            "a later prompt of " + sizes + " started before the restored one's last command, which it waited for");
  }
}

// A request that skips the cache waits for no restoring producer, and no producer plans a junction
// for it.
void testSkipCacheRequestDoesNotWaitForAProducer() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // The state at 64 and both blocks of KV under it are on disk.
  std::vector<uint32_t> prompt(226);
  std::iota(prompt.begin(), prompt.end(), 1);
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  const auto block = test::publishBlocks(cache, 999, prompt, 64);
  cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer), false);
  cache.endRequest(999);
  require(test::evictStep(cache).stateBuffers && cache.pollTransfers(),
          "state was not demoted");
  for (uint32_t written = 1; written <= 2; ++written) {
    require(test::evictStep(cache).gave() && tier.demotions == written,
            "KV block was not written");
    tier.complete();
    require(cache.pollTransfers(), "written block did not land");
  }
  while (pool.snapshot().pagesAllocated)
    require(test::releaseStep(cache).gave(),
            "an empty extent was not released");
  // Alone, with no page the budget grants, its restore fails: it skips the
  // cache from then on.
  storage.growthBlocked = true;
  engine.submit(request(1, prompt));
  static_cast<void>(engine.tick(1));
  require(engine.resourceWaitSnapshot(1).memory == 1 && executor.diskReads == 0,
          "the restore that could not fit did not wait for memory");
  storage.growthBlocked = false;
  // A foreground request sharing its first 192 tokens is admitted first and
  // restores the prefix. It plans a junction at 160 for a sibling sharing
  // that much with it, and none at 192 for the request that skips the cache.
  std::vector<uint32_t> producer = prompt;
  std::fill(producer.begin() + 192, producer.end(), 7);
  auto foreground = request(2, producer);
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  std::vector<uint32_t> sibling = producer;
  std::fill(sibling.begin() + 160, sibling.end(), 9);
  engine.submit(request(3, sibling));
  // While the producer restores, the request that skips the cache starts:
  // the producer runs in no command before its read lands, and the request's
  // prompt ends in its own. It never waits for the
  // producer's prefix.
  double now = 101;
  bool prefixWait = false;
  tickUntil(engine, now,
            [&] {
              for (const ActiveRequestSnapshot &active : engine.snapshot().activeRequests)
                prefixWait = prefixWait || (active.id == 1 && active.phase == Phase::WaitingPrefix);
              return std::find(events.startIds.begin(), events.startIds.end(), 1) !=
                     events.startIds.end();
            },
            "the request that skips the cache did not start");
  require(!prefixWait && events.starts.front() == 0 &&
              executor.diskReads == 1 && executor.restored == 0,
          "the request that skips the cache waited for the restoring producer, or for its prefix");
  executor.restoreControl->ready = true;
  for (; now < 300 && !idle(engine); ++now) {
    static_cast<void>(engine.tick(now));
    tier.complete();
  }
  require(idle(engine) && events.completedCount == 3 && events.failedCount == 0,
          "the requests did not finish");
  const auto &boundaries = executor.plans.at(2).boundaries;
  require(events.starts.back() == 160 &&
              std::none_of(boundaries.begin(), boundaries.end(),
                           [](const DraftBoundaryPlan &boundary) {
                             return boundary.boundary == 192;
                           }),
          "the producer planned a junction for the request that skips the cache");
}

// A request whose prefix restore cannot fit alone is not probed again, and starts cold.
void testSkipCacheCandidateIsNotProbed() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  const auto hashed = [&] { return cache.snapshot().lookup.probeHashedBlocks; };
  const std::vector<uint32_t> prompt(65, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 64));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 2; ++written) {
    require(test::evictStep(cache).gave(), "KV block was not written");
    tier.complete();
    require(cache.pollTransfers(), "written block did not land");
  }
  require(test::releaseStep(cache).gave() &&
              storage.allocatedPages() == 0 && cache.snapshot().kvTier.diskBlocks == 2,
          "the prefix did not move to disk whole");
  // No page can be allocated until the request starts again.
  storage.growthBlocked = true;
  executor.beginObserver = [&] { storage.growthBlocked = executor.beginAttempts < 2; };
  engine.submit(request(1, prompt));
  static_cast<void>(engine.tick(1));
  require(hashed() == 2 && events.starts.empty() && engine.resourceWaitSnapshot(1).memory == 1,
          "the restore that cannot fit alone did not wait");
  double now = 2;
  tickUntil(engine, now, [&] { return !events.starts.empty(); },
            "the request did not start without its prefix");
  require(hashed() == 2 && events.starts[0] == 0,
          "a request that ignores the cache was probed or ranked by its prefix");
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 1 && executor.prefillRows == 65 && tier.restores == 0,
          "the request did not finish cold");
}

// A restore refused pages suspends a lower-priority lane when nothing is on its way back, and waits
// without suspending when a write will free them.
void testRefusedPrefixRestorePreemptsLowerResident() {
  for (const bool written : {false, true}) {
    test::TestKvStorage storage(64, 4096, 1);
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    engine::Cache cache(pool, written ? &tier : nullptr, written ? test::ampleDiskBudget() : nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
    guardReleases(storage, engine);
    // The prefix's state is on disk and its two pages in memory. Beside it,
    // another state on disk keeps the page of its block, which a reclaim
    // writes to disk before it frees it.
    const std::vector<uint32_t> prompt(65, 17);
    publishDiskState(cache, prompt);
    if (written) {
      cache.beginRequest(998);
      require(cache.ensureTokens(998, 32).granted(), "fixture KV failed");
      demoteState(cache, test::publishBlocks(cache, 998, std::vector<uint32_t>(33, 5), 32));
      cache.endRequest(998);
    }
    // The Background lane's three pages fill the engine's limit.
    storage.budgetPages = storage.allocatedPages() + 3;
    auto background = request(1, std::vector<uint32_t>(65, 1));
    background.priority = RequestPriority::Background;
    background.maxNewTokens = 1000;
    engine.submit(std::move(background));
    double now = 1;
    tickUntil(engine, now, [&] { return events.outputs.contains(1); },
              "the Background lane did not decode");
    // Its state read from disk, the request needs a page beyond its prefix.
    engine.submit(request(2, prompt));
    tickUntil(engine, now, [&] { return executor.beginAttempts == 2; },
              "the request did not try to start");
    if (written) {
      require(tier.demotions == 1 && engine.snapshot().prioritySuspensions == 0 &&
                  executor.requests.at(1).resident,
              "a lane yielded memory on its way back");
      tier.complete();
    } else {
      require(engine.snapshot().prioritySuspensions == 1 && !executor.requests.at(1).resident,
              "the refused restore did not suspend the lower lane");
    }
    tickUntil(engine, now, [&] { return executor.diskReads == 1; },
              "the request was not admitted into its restore");
    require(executor.beginAttempts == 3, "the request did not start at its next attempt");
    executor.restoreControl->ready = true;
    tickUntil(engine, now, [&] { return events.startIds.back() == 2; },
              "the restored request did not start");
    require(events.starts.back() == 64 &&
                engine.snapshot().prioritySuspensions == (written ? 0U : 1U),
            "the request did not start from its restored prefix");
    for (uint64_t id : {1, 2})
      engine.cancel(id);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
    require(events.failedCount == 0, "a request failed");
  }
}

// A request reading its prefix from disk counts as restoring, not as waiting for memory.
void testRestoringRequestIsNotWaitingForMemory() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  bool hostRefuses = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return hostRefuses; };
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 17);
  publishDiskState(cache, prompt);
  engine.submit(request(1, prompt));
  static_cast<void>(engine.tick(1));
  require(engine.resourceWaitSnapshot(1).memory == 1, "the host's refusal did not wait for memory");
  hostRefuses = false;
  double now = 101;
  static_cast<void>(engine.tick(now));
  const auto restoring = engine.resourceWaitSnapshot(now);
  require(executor.diskReads == 1 && restoring.restoring == 1 && restoring.memory == 0,
          "a request reading its prefix from disk was counted as waiting for memory");
  executor.restoreControl->ready = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 1 && executor.restored == 64 &&
              engine.resourceWaitSnapshot(now).restoring == 0,
          "the restored request did not finish");
}

// A cancelled disk prefix restore stops its queued reads and keeps the disk pages for a retry.
void testCancelledDiskPrefixStopsQueuedReads() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 1;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(129, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 128).granted(), "fixture KV failed");
  const auto block = test::publishBlocks(cache, 999, prompt, 128);
  demoteState(cache, block);
  cache.endRequest(999);
  for (int page = 0; page < 4; ++page) {
    require(test::evictStep(cache).gave(),
            "fixture KV did not demote");
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  }
  engine.submit(request(1, prompt));
  static_cast<void>(engine.tick(1));
  require(tier.restores == 1 && executor.diskReads == 1,
          "restore did not start within the tier's limit");
  engine.cancel(1);
  executor.restoreControl->ready = true;
  static_cast<void>(engine.tick(2));
  for (int tick = 3; tick < 16; ++tick) {
    tier.complete();
    static_cast<void>(engine.tick(tick));
  }
  const auto stats = cache.snapshot();
  require(tier.restores == 1 && !tier.inFlight() && idle(engine) &&
              stats.stateCache.pinned == 0 && stats.activeRequests == 0 &&
              executor.requests.empty() && !events.outputs.contains(1),
          "cancelled prefix read unused pages or retained active resources");
  require(stats.kvTier.diskBlocks == 4 && stats.pool.pagesPrefix == 1,
          "cancellation destroyed reusable disk pages");
  executor.restoreControl = std::make_shared<RestoreControl>();
  executor.restoreControl->ready = true;
  engine.submit(request(2, prompt));
  for (int tick = 16; tick < 64; ++tick) {
    tier.complete();
    static_cast<void>(engine.tick(tick));
  }
  require(tier.restores == 4 && executor.restored == 128 &&
              events.outputs.contains(2) && events.failedCount == 0 &&
              cache.snapshot().activeRequests == 0,
          "retry did not reuse the cancelled request's disk prefix");
}

// A start refused pages on their way back keeps its claim waiting for the wait limit from its first
// pending refusal, and no longer.
void testAPendingRefusalHoldsItsClaimForTheWaitLimit() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  tier.capacity = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  const EngineConfig config = test::engineConfig({.maxContext = 102400});
  engine::Engine engine(config, cache, executor, events);
  guardReleases(storage, engine);
  cacheBlocksUnderDiskStates(cache, 6);
  storage.growthBlocked = true;
  EngineRequest start = request(1, std::vector<uint32_t>(161, 7));
  start.deadlineMilliseconds = 1e9;
  engine.submit(std::move(start));
  static_cast<void>(engine.tick(1));
  require(tier.demotions == 3 && config.governor->ledger().claimWaiting(),
          "the start refused pages on their way back did not wait on its claim");
  // The demotions never land.
  static_cast<void>(engine.tick(1 + kResourceWaitTimeoutMilliseconds - 1000));
  require(config.governor->ledger().claimWaiting() && events.failedCount == 0,
          "the claim stopped waiting before the wait limit from its first pending refusal");
  static_cast<void>(engine.tick(1 + kResourceWaitTimeoutMilliseconds + 500));
  require(!config.governor->ledger().claimWaiting(), "the claim held its room past the wait limit");
}

// A retry past the wait limit holds no room again, though another lane's end renewed its deadline.
void testARetryPastTheWaitLimitHoldsNothing() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  tier.capacity = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  const EngineConfig config = test::engineConfig({.maxContext = 102400});
  engine::Engine engine(config, cache, executor, events);
  guardReleases(storage, engine);
  cacheBlocksUnderDiskStates(cache, 6);
  storage.growthBlocked = true;
  EngineRequest lane = request(1, std::vector<uint32_t>(161, 7));
  lane.deadlineMilliseconds = 1e9;
  engine.submit(std::move(lane));
  static_cast<void>(engine.tick(1));
  require(tier.demotions == 3 && config.governor->ledger().claimWaiting(),
          "the lane refused pages on their way back did not wait on its claim");
  // The demotions never land. Another lane starts and ends before the
  // limit, which renews the waiting lane's deadline at its next refusal.
  EngineRequest other = request(2, std::vector<uint32_t>(1, 3));
  other.deadlineMilliseconds = 1e9;
  engine.submit(std::move(other));
  static_cast<void>(engine.tick(29'000));
  engine.cancel(2);
  static_cast<void>(engine.tick(29'001));
  static_cast<void>(engine.tick(29'002));
  static_cast<void>(engine.tick(1 + kResourceWaitTimeoutMilliseconds + 500));
  require(events.failedCount == 0 && !config.governor->ledger().claimWaiting(),
          "a retry past the wait limit from the first pending refusal held its room again");
}

// A start queued for scheduling between refusals keeps its first refusal's time, which bounds its
// claim.
void testAStartQueuedBetweenRefusalsHoldsFromTheFirst() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  tier.capacity = 16;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  const EngineConfig config = test::engineConfig({.maxContext = 102400});
  engine::Engine engine(config, cache, executor, events);
  guardReleases(storage, engine);
  // The chain the start restores, on disk, and cached blocks whose eviction
  // starts demotions; no free page, no extent more.
  const std::vector<uint32_t> prompt(129, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 128).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 128));
  cache.endRequest(999);
  for (int page = 0; page < 4; ++page) {
    require(test::evictStep(cache).gave(),
            "fixture KV did not demote");
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  }
  cacheBlocksUnderDiskStates(cache, 6);
  storage.growthBlocked = true;
  const KvPageAcquisition held = pool.acquirePages(pool.freePageCount());
  EngineRequest start = request(1, prompt);
  start.deadlineMilliseconds = 1e9;
  engine.submit(std::move(start));
  static_cast<void>(engine.tick(1));
  require(held.granted() && tier.demotions != 0 && config.governor->ledger().claimWaiting() &&
              !engine.snapshot().activeRequests.empty() &&
              engine.snapshot().activeRequests.front().phase == Phase::WaitingResources,
          "the start was not refused the pages of its chain while demotions are in flight");
  // A Foreground lane starts, and the start waits for scheduling below it
  // until it ends; then it is refused again.
  EngineRequest foreground = request(2, std::vector<uint32_t>(1, 3));
  foreground.priority = RequestPriority::Foreground;
  foreground.deadlineMilliseconds = 1e9;
  engine.submit(std::move(foreground));
  static_cast<void>(engine.tick(2));
  static_cast<void>(engine.tick(4'999));
  engine.cancel(2);
  static_cast<void>(engine.tick(5'000));
  static_cast<void>(engine.tick(1 + kResourceWaitTimeoutMilliseconds + 500));
  require(events.failedCount == 0 && !config.governor->ledger().claimWaiting(),
          "a start queued between pending refusals held its room past the wait limit from its first");
}

// A burst of completions queues their points together, copies them all, and delays no decode
// command.
void testACompletionBurstIsCopiedWithoutDelayingDecode() {
  test::EngineHarness harness(
      {.pages = 256,
       .tier = test::TierOptions{.capacity = 256, .transferLimit = 4, .bytesPerMillisecond = 1.0, .persistent = true},
       .memory = test::FakeMemory::Config{},
       .timing = &test::kM6FlashNextQ2});
  harness.model.keptTier = std::make_shared<OffloadControl>();
  harness.model.keptTier->ready = true;
  harness.model.decodeFinishes = false;
  EngineRequest decoding = harness.request(1, std::vector<uint32_t>(65, 1));
  decoding.maxNewTokens = 100'000;
  decoding.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(decoding));
  // A prompt deep enough to be copied behind (WriteBehind::kMinimumTokens),
  // cached with its replay point at 2048.
  std::vector<uint32_t> shared(2060);
  std::iota(shared.begin(), shared.end(), 10'000);
  EngineRequest first = harness.request(2, shared);
  first.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(first));
  harness.runUntil([&] { return harness.events.usage.contains(2); }, "the first prompt did not finish");
  for (uint32_t id = 3; id <= 5; ++id) {
    std::vector<uint32_t> prompt(shared.begin(), shared.begin() + 2048);
    for (uint32_t token = 0; token < 60; ++token)
      prompt.push_back(id * 1000 + token);
    EngineRequest ending = harness.request(id, prompt);
    ending.maxNewTokens = 4;
    ending.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(ending));
  }
  uint64_t mostWaiting = 0;
  harness.runUntil(
      [&] {
        mostWaiting = std::max<uint64_t>(mostWaiting, harness.engine.snapshot().writeBehind.waiting);
        return harness.events.completedCount == 4 && harness.engine.snapshot().writeBehind.durable == 4;
      },
      "the burst's points were not all copied behind it");
  require(mostWaiting >= 3, "the three points did not wait at once: most waiting " + std::to_string(mostWaiting));
  const std::vector<test::EngineHarness::Command> &commands = harness.commands();
  for (size_t index = 1; index < commands.size(); ++index)
    require(commands[index].startMilliseconds ==
                commands[index - 1].startMilliseconds + commands[index - 1].wallMilliseconds,
            "a command waited after the one before it ended");
  require(!harness.events.usage.contains(1) && harness.events.failedCount == 0,
          "the decoding lane did not decode throughout");
  harness.engine.cancel(1);
  harness.runUntilIdle();
}

} // namespace

std::span<const test::TestCase> diskTests() {
  static constexpr test::TestCase kTests[] = {
      {"testAPendingRefusalHoldsItsClaimForTheWaitLimit", testAPendingRefusalHoldsItsClaimForTheWaitLimit},
      {"testARetryPastTheWaitLimitHoldsNothing", testARetryPastTheWaitLimitHoldsNothing},
      {"testAStartQueuedBetweenRefusalsHoldsFromTheFirst", testAStartQueuedBetweenRefusalsHoldsFromTheFirst},
      {"testDiskKvPrefixIsRestoredBeforeTheLaneRuns", testDiskKvPrefixIsRestoredBeforeTheLaneRuns},
      {"testSkipCacheCandidateIsNotProbed", testSkipCacheCandidateIsNotProbed},
      {"testRefusedPrefixRestorePreemptsLowerResident", testRefusedPrefixRestorePreemptsLowerResident},
      {"testRestoringRequestIsNotWaitingForMemory", testRestoringRequestIsNotWaitingForMemory},
      {"testSharedPrefillWaitsForARestoringProducer", testSharedPrefillWaitsForARestoringProducer},
      {"testCancelledRestoringProducerReleasesItsWaiter", testCancelledRestoringProducerReleasesItsWaiter},
      {"testALaterPromptWaitsForARestoringPromptThatLeadsIt", testALaterPromptWaitsForARestoringPromptThatLeadsIt},
      {"testSkipCacheRequestDoesNotWaitForAProducer", testSkipCacheRequestDoesNotWaitForAProducer},
      {"testCancelledDiskPrefixStopsQueuedReads", testCancelledDiskPrefixStopsQueuedReads},
      {"testFailedDiskRestoreKeepsShallowerState", testFailedDiskRestoreKeepsShallowerState},
      {"testRepeatedDiskHitPromotesToMemory", testRepeatedDiskHitPromotesToMemory},
      {"testGrowthWaitsForTheStateWriteInFlight", testGrowthWaitsForTheStateWriteInFlight},
      {"testPageShortfallDemotesInBulk", testPageShortfallDemotesInBulk},
      {"testWaitingLaneAlwaysNamesAWakeup", testWaitingLaneAlwaysNamesAWakeup},
      {"testReplayPointsAreWrittenBehind", testReplayPointsAreWrittenBehind},
      {"testFinishedConversationsLeaveAtOnce", testFinishedConversationsLeaveAtOnce},
      {"testCopiesAheadWaitForALaneRefusedBesideOthers", testCopiesAheadWaitForALaneRefusedBesideOthers},
      {"testCopiesAheadWaitNoLongerThanTheWaitsLimit", testCopiesAheadWaitNoLongerThanTheWaitsLimit},
      {"testCopiesAheadDoNotWaitForALaneBelowThePriorityThatRuns",
       testCopiesAheadDoNotWaitForALaneBelowThePriorityThatRuns},
      {"testARefusalAfterProgressRestartsTheWaitLimit", testARefusalAfterProgressRestartsTheWaitLimit},
      {"testCopiesAheadDoNotWaitForARequestHeldBehindARefusal", testCopiesAheadDoNotWaitForARequestHeldBehindARefusal},
      {"testCopiesAheadWaitForALaneRefusedDuringARecovery", testCopiesAheadWaitForALaneRefusedDuringARecovery},
      {"testCopiesAheadDoNotWaitForARequestARecoveryLeavesUntried",
       testCopiesAheadDoNotWaitForARequestARecoveryLeavesUntried},
      {"testWaitingPlanDoesNotIdleRunnableLanes", testWaitingPlanDoesNotIdleRunnableLanes},
      {"testNothingInFlightIsNotPending", testNothingInFlightIsNotPending},
      {"testKvGrowthProceedsThroughDemotion", testKvGrowthProceedsThroughDemotion},
      {"testAsyncRestoreLifecycle", testAsyncRestoreLifecycle},
      {"testStopWhileRestoringFinishesAfterTheRestore", testStopWhileRestoringFinishesAfterTheRestore},
      {"testDiskHitWithNoActiveMemory", testDiskHitWithNoActiveMemory},
      {"testABatchRefusalOfNothingOnItsWayBackHoldsNoOne", testABatchRefusalOfNothingOnItsWayBackHoldsNoOne},
      {"testStateWithoutACacheSlotGoesToDisk", testStateWithoutACacheSlotGoesToDisk},
      {"testStateAlreadyOnDiskIsDeduplicated", testStateAlreadyOnDiskIsDeduplicated},
      {"testCancelledPrefillRecoversFromItsDiskCheckpoint", testCancelledPrefillRecoversFromItsDiskCheckpoint},
      {"testFailedFinalStateKeepsTheDiskCheckpoint", testFailedFinalStateKeepsTheDiskCheckpoint},
      {"testDiskCheckpointStaysUntilItsSuccessorIsPublished", testDiskCheckpointStaysUntilItsSuccessorIsPublished},
      {"testNoCheckpointWithinAChunkOfTheReplayBoundary", testNoCheckpointWithinAChunkOfTheReplayBoundary},
      {"testSkippedCheckpointKeepsPreviousRecoveryPoint", testSkippedCheckpointKeepsPreviousRecoveryPoint},
      {"testACompletionBurstIsCopiedWithoutDelayingDecode", testACompletionBurstIsCopiedWithoutDelayingDecode},
  };
  return kTests;
}
