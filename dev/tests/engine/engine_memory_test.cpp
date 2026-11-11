#include "EngineHarness.hpp"
#include "EngineTestSupport.hpp"
#include "FakeElasticPool.hpp"
#include "GovernedHost.hpp"
#include "MemoryPasses.hpp"
#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "TestMetalStatistics.hpp"
#include "TestRegistry.hpp"
#include "benchmarks/PrefillWork.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

using namespace splash;
using namespace splash::engine;
using benchmark::draftContextRows;
using splash::test::cacheBlocksUnderDiskStates;
using splash::test::capturedStates;
using splash::test::constrainedRequest;
using splash::test::demoteState;
using splash::test::Events;
using splash::test::Executor;
using splash::test::GovernedHost;
using splash::test::guardReleases;
using splash::test::idle;
using splash::test::OffloadControl;
using splash::test::OffloadState;
using splash::test::request;
using splash::test::runUntilIdle;
using splash::test::runUntilStatesHeld;
using splash::test::Spilling;
using splash::test::State;
using splash::test::tickUntil;

namespace {

using splash::test::rejects;
using splash::test::require;

// The engine's limit reached: the rest of it holds other memory (the
// weights), so that every allocation through `host`'s governor is refused
// there for want of room, its claim lacking what it asked (MemoryGovernor::
// need). For a fake that admits its own allocations once what the ledger
// freed made room, and asks the governor for the refusal's books.
void reachTheLimit(GovernedHost &host) { host.memory.hold("weights", host.governor.snapshot().headroomBytes); }

// A fake's refusal of `bytes` at that limit, its cause, the need booked.
metal::AllocationFailure refusedAtTheLimit(GovernedHost &host, uint64_t bytes) {
  const metal::AllocationResult result = host.reserve(bytes);
  if (result)
    throw std::logic_error("the governor admitted what its limit was to refuse");
  return result.failure;
}

// A publication's expected refusals are values; an exception is a broken invariant and ends the engine instead of
// counting as a failed publication.
void testPublicationInvariantFailureIsFatal() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.reservationObserver = [] { throw std::logic_error("reservation invariant"); };
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(3, std::vector<uint32_t>(65, 7)));
  rejects(
      [&] {
        for (double now = 1; now < 32; ++now)
          static_cast<void>(engine.tick(now));
      },
      "reservation invariant", "a broken publication invariant did not fail the engine");
  require(engine.snapshot().replayStatePublicationFailures == 0,
          "a broken publication invariant was counted as a failed publication");
}

// A boundary whose buffers the model refuses costs one attempt: the command ends there, the failure is counted and
// the request is served.
void testRefusedBuffersCostOnlyThatAttempt() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.deniedReservations = 100;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65, 7);

  engine.submit(request(3, prompt));
  runUntilIdle(engine);
  const DraftContextPlan &plan = executor.plans.at(3);
  require(plan.boundaries.size() == 2 && plan.boundaries[0].boundary == 64 &&
              plan.boundaries[1].boundary == 65 &&
              draftContextRows(plan) == prompt.size(),
          "the replay boundary left the plan");
  require(executor.reservationAttempts == 1 && executor.reservations == 0 &&
              engine.snapshot().captureCuts == 1 &&
              engine.snapshot().replayStatePublicationFailures == 1 &&
              engine.snapshot().junctionMaterializationFailures == 0 &&
              engine.snapshot().recycledStatePublications == 0 &&
              resources.snapshot().stateCache.evictions == 0 &&
              events.completedCount == 1,
          "latest replay-state denial was counted incorrectly");

  // The first pass left a usable KV chain but no state. The second request
  // plans its replay state there again; its refused buffers are counted as
  // a replay-state failure while the request replays through the KV prefix
  // normally.
  engine.submit(request(4, prompt));
  runUntilIdle(engine);
  require(executor.reservationAttempts == 2 && executor.reservations == 0 &&
              engine.snapshot().junctionMaterializationFailures == 0 &&
              engine.snapshot().replayStatePublicationFailures == 2 &&
              events.starts.back() == 0 &&
              executor.prefillRows == 130 && events.completedCount == 2 &&
              events.failedCount == 0,
          "denied replay state failed the request or was not counted");

  // The denial is transient: the next lane over the same prefix lands it.
  executor.deniedReservations = 0;
  engine.submit(request(5, prompt));
  runUntilIdle(engine);
  require(executor.reservations == 1 &&
              engine.snapshot().replayStatePublications == 1 &&
              resources.snapshot().stateCache.entries == 1,
          "replay state was not published once its buffers were granted");
}

// Buffers refused once for a replay state come from recycling the least recently used cached state, whose slot then
// holds the new lane's state.
void testRefusedBuffersRecycleLruStateAndRetry() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> older(65, 1);
  const std::vector<uint32_t> newer(65, 2);
  engine.submit(request(1, older));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.stateCache.evictions == 0,
          "recycle fixture did not cache the older state");

  executor.deniedReservations = 1;
  engine.submit(request(2, newer));
  runUntilIdle(engine);
  const auto recycled = engine.snapshot();
  require(executor.reservationAttempts == 3 && executor.reservations == 2 &&
              recycled.recycledStatePublications == 1 &&
              recycled.replayStatePublications == 2 &&
              recycled.replayStatePublicationFailures == 0 &&
              events.completedCount == 2,
          "refused buffers were not retried after recycling a state");
  require(recycled.resources.stateCache.evictions == 1 &&
              recycled.resources.stateCache.entries == 1 &&
              recycled.resources.stateCache.publications == 2 &&
              recycled.resources.stateCache.bytes == cached.stateCache.bytes,
          "recycling changed the cached state footprint");

  // The surviving state belongs to the new lane: re-sending its prompt is a
  // state hit, while the older prompt is back to KV without a state.
  engine.submit(request(3, newer));
  runUntilIdle(engine);
  require(executor.restored == 64 && events.starts.back() == 64,
          "recycled slot does not hold the new lane's state");
  engine.submit(request(4, older));
  runUntilIdle(engine);
  require(executor.restored == 64 &&
              events.starts.back() == 0 &&
              engine.snapshot().replayStatePublications == 3,
          "recycled state was not the least recently used one");
}

// A state recycled for a capture's buffers counts as a recycled publication only once the capture is published, not
// for a cancelled command.
void testRecycledStateCountsOnlyForAPublishedCapture() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(65, 1)));
  runUntilIdle(engine);
  executor.deniedReservations = 1;
  executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
  engine.submit(request(2, std::vector<uint32_t>(65, 2)));
  require(engine.tick(1) && engine.commandInFlight() && executor.reservationAttempts == 3 &&
              resources.snapshot().stateCache.evictions == 1,
          "the capture's buffers did not recycle the cached state");
  engine.cancel(2);
  *executor.holdPrefillUntil = true;
  runUntilIdle(engine);
  const auto counters = engine.snapshot();
  require(counters.recycledStatePublications == 0 && counters.replayStatePublications == 1 &&
              counters.resources.stateCache.entries == 0 && counters.cancelled == 1,
          "a cancelled capture counted as a recycled publication");
}

// A boundary refused again after recycling fails having paid one cached state; persistent refusal never drains the
// cache.
void testPersistentRefusalRecyclesAtMostOneState() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {1, 2, 3}) {
    engine.submit(request(id, std::vector<uint32_t>(65, id)));
    runUntilIdle(engine);
  }
  require(resources.snapshot().stateCache.entries == 3,
          "persistent-denial fixture did not cache three states");

  executor.deniedReservations = 100;
  engine.submit(request(4, std::vector<uint32_t>(65, 4)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(executor.reservationAttempts == 5 && executor.reservations == 3 &&
              executor.deniedReservations == 98 &&
              after.replayStatePublicationFailures == 1 &&
              after.recycledStatePublications == 0 &&
              events.completedCount == 4 && events.failedCount == 0,
          "persistently refused buffers were retried more than once");
  require(after.resources.stateCache.evictions == 1 &&
              after.resources.stateCache.entries == 2,
          "persistent refusal recycled more than one cached state");
}

void testLongSuffixSkipsDraftRestore() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prefix(65);
  for (uint32_t index = 0; index < prefix.size(); ++index) {
    prefix[index] = index + 1;
  }
  engine.submit(request(10, prefix));
  runUntilIdle(engine);

  std::vector<uint32_t> extended = prefix;
  extended.resize(4097, 99);
  engine.submit(request(11, extended));
  runUntilIdle(engine);
  require(!executor.restoredDraft &&
              !executor.plans.at(11).restoresDraftState,
          "long suffix restored draft state that its final window overwrites");
}

// A lane cancelled while its boundary's command is in flight publishes nothing; no failure is counted and earlier
// blocks stay cached.
void testCancellationInFlightAtBoundaryPublishesNoState() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(4097, 11);

  engine.submit(request(4, prompt));
  require(engine.tick(1) && engine.tick(2),
          "long request did not complete its first prefill command");
  require(resources.snapshot().pool.pagesPrefix == 64 &&
              executor.reservationAttempts == 0,
          "first prefill command changed the state cache");
  require(engine.tick(3) && engine.commandInFlight(),
          "second prefill command was not launched");
  engine.cancel(4);
  require(engine.commandInFlight() && executor.requests.size() == 1,
          "in-flight cancellation released resources owned by the command");
  runUntilIdle(engine);
  // The command captured the replay state; it goes back to the pool.
  const auto snapshot = engine.snapshot();
  require(executor.captures.size() == 1 && *executor.evictedStateBytes == 64 &&
              snapshot.resources.stateCache.entries == 0 &&
              snapshot.resources.stateCache.publications == 0 &&
              snapshot.replayStatePublications == 0 &&
              snapshot.replayStatePublicationFailures == 0 &&
              snapshot.cancelled == 1,
          "cancelled command published or counted its boundary");
  require(snapshot.resources.pool.pagesPrefix == 64 &&
              snapshot.resources.pool.pagesActive == 0 &&
              snapshot.resources.activeRequests == 0 &&
              executor.requests.empty(),
          "cancellation leaked active resources or dropped committed KV");
}

// A start refused its state's buffers retries once the ledger freed what it lacks: a cached state's buffers, or
// under the host the idle buffers.
void testLaneStateGrowthReclaimsCachedStateAndRetries() {
  for (bool hostPressure : {false, true}) {
    test::TestKvStorage storage(16, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    GovernedHost host;
    engine::Engine engine(host.connect({}), resources, executor, events);
    guardReleases(storage, engine);

    engine.submit(request(20, std::vector<uint32_t>(65, 7)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();
    require(cached.stateCache.entries == 1,
            "state-reclaim setup did not publish a composite state");

    if (hostPressure) {
      storage.growthBlocked = true;
      executor.kvGrowthBlocked = &storage.growthBlocked;
      executor.reclaimableIdleStateBytes = 350'224'384;
      host.pause(true);
      host.admitBegins(executor, 4096);
    } else {
      executor.statesLacked = [&] { return resources.snapshot().stateCache.entries != 0 ? 1U : 0U; };
      executor.beginGrowthBlocked = [&] { return executor.statesLacked() != 0; };
    }
    const uint32_t attempts = executor.beginAttempts;
    engine.submit(request(21, {8}));
    runUntilIdle(engine);
    const auto after = resources.snapshot();
    require(executor.beginAttempts == attempts + 2 &&
                events.completedCount == 2,
            "lane state growth did not reclaim memory and retry");
    if (hostPressure) {
      require(executor.reclaimedIdleStateBytes == 350'224'384 &&
                  !storage.growthBlocked &&
                  after.stateCache.entries == cached.stateCache.entries &&
                  after.stateCache.evictions == cached.stateCache.evictions,
              "host-pressure state admission evicted cache or ignored idle memory");
    } else {
      require(after.stateCache.entries == 0,
              "lane state growth did not reclaim cached state");
    }
  }
}

void testKvGrowthReclaimsIdleStateBeforeCache() {
  for (auto failure : {metal::AllocationFailure::EngineBudget,
                       metal::AllocationFailure::HostPressure}) {
    test::TestKvStorage storage(12, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);

    engine.submit(request(25, std::vector<uint32_t>(65, 25)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();
    require(cached.stateCache.entries == 1,
            "idle-state reclaim setup did not retain cache state");

    storage.growthBlocked = true;
    storage.allocationFailure = failure;
    executor.kvGrowthBlocked = &storage.growthBlocked;
    executor.unblockGrowthOnSuspend = false;
    executor.reclaimableIdleStateBytes = 350'224'384;
    engine.submit(request(26, std::vector<uint32_t>(161, 26)));
    runUntilIdle(engine);

    const auto after = resources.snapshot();
    require(executor.reclaimedIdleStateBytes == 350'224'384 &&
                !storage.growthBlocked && events.completedCount == 2 &&
                engine.snapshot().resourceSuspensions == 0,
            "KV growth did not reclaim model-owned idle state before suspension");
    // The growth denial neither evicts the cached state nor touches the growing
    // lane's own boundary: its replay state still lands.
    require(after.stateCache.evictions == cached.stateCache.evictions &&
                after.stateCache.entries == cached.stateCache.entries + 1 &&
                engine.snapshot().replayStatePublications == 2 &&
                engine.snapshot().recycledStatePublications == 0,
            "KV growth evicted useful composite state before idle state memory");
  }
}

// KV growth the cache covers keeps the pooled buffers the next lane starts from, so a request arriving under host
// pressure still starts.
void testKvGrowthKeepsTheRingAWaitingLaneStartsFrom() {
  constexpr uint64_t kRing = 4096;
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({}), resources, executor, events);
  guardReleases(storage, engine);
  executor.pooledLaneBytes = kRing;
  executor.statesLacked = [&]() -> std::optional<uint32_t> {
    if (executor.pooledLaneBytes)
      return 0;
    return std::nullopt;
  };
  executor.beginGrowthBlocked = [&] {
    if (executor.pooledLaneBytes >= kRing) {
      executor.pooledLaneBytes -= kRing;
      return false;
    }
    executor.beginAllocationFailure = host.reserve(kRing).failure;
    return executor.beginAllocationFailure != metal::AllocationFailure::None;
  };
  executor.laneReleased = [&] { executor.pooledLaneBytes += kRing; };
  const auto resident = [&](uint64_t id) {
    const auto found = executor.requests.find(id);
    return found != executor.requests.end() && found->second.resident;
  };
  const auto done = [&](uint64_t id) { return events.usage.contains(id); };
  const auto submit = [&](uint64_t id, uint32_t tokens) {
    EngineRequest next = request(id, std::vector<uint32_t>(65, static_cast<uint32_t>(id)));
    next.maxNewTokens = tokens;
    next.deadlineMilliseconds = 1e9;
    engine.submit(next);
  };

  double now = 1;
  submit(40, 8);
  submit(41, 400);
  tickUntil(engine, now, [&] { return done(40); }, "the first request did not finish");
  require(resident(41) && executor.pooledLaneBytes == kRing,
          "the finished lane's ring did not return to the pool");

  host.pause(true);
  storage.growthBlocked = true;
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  executor.reclaimableIdleStateBytes = 1U << 20;
  tickUntil(engine, now, [&] { return !storage.growthBlocked; }, "KV growth was never denied");
  require(executor.pooledLaneBytes == kRing, "KV growth reclaimed the ring a lane starts from");

  submit(42, 8);
  tickUntil(engine, now, [&] { return resident(42) || done(42); }, "the waiting request never started");
  require(resident(41), "the waiting request started only once the growing one ended");
  tickUntil(engine, now, [&] { return done(41) && done(42); }, "a request did not finish");
  require(events.failures.empty(), "a request failed");
}

// Two lanes prefill in one command: the KV growth denial is met by reclaiming idle model state, never by dropping
// a lane's boundary, and both replay states land.
void testKvGrowthDenialKeepsEveryLaneReplayState() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  storage.growthBlocked = true;
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.reclaimableIdleStateBytes = 1U << 20;

  engine.submit(request(300, std::vector<uint32_t>(161, 300)));
  engine.submit(request(301, std::vector<uint32_t>(161, 301)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(executor.reclaimedIdleStateBytes == (1U << 20) &&
              !storage.growthBlocked && executor.suspensions == 0 &&
              events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "a growth denial suspended or failed a lane");
  require(!executor.prefillWidths.empty() && executor.prefillWidths[0] == 2,
          "both lanes did not prefill in one command");
  require(after.replayStatePublications == 2 &&
              after.replayStatePublicationFailures == 0 &&
              after.recycledStatePublications == 0 &&
              after.resources.stateCache.entries == 2 &&
              after.resources.stateCache.evictions == 0,
          "KV growth denial cost a lane its replay state");
}

// The shared prefix of 2112 tokens, then 161 of a conversation's own from
// `first` on: a branch off a cached conversation plans its junction at 2112
// and its replay point at 2272, both inside its command [2048, 2273).
std::vector<uint32_t> branchingConversation(uint32_t first) {
  std::vector<uint32_t> prompt(2112 + 161);
  std::iota(prompt.begin(), prompt.begin() + 2112, 100'000);
  std::iota(prompt.begin() + 2112, prompt.end(), first);
  return prompt;
}

// A prefill command captures each state boundary it reaches on the way: checkpoints, replay points and a branch's
// junction.
void testBoundaryStatesAreCapturedInsideCommands() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(8001);
  std::iota(prompt.begin(), prompt.end(), 1);
  // Its replay point at 1056 is where the long prompt resumes.
  engine.submit(request(1, std::vector<uint32_t>(prompt.begin(), prompt.begin() + 1057)));
  runUntilIdle(engine);
  require(capturedStates(executor, 1) == std::vector<std::pair<uint32_t, uint64_t>>{{1056, 1057}},
          "a prompt's one command did not capture its replay point");
  // Its commands end at 3104, 5152, 7200 and 8001.
  engine.submit(request(2, prompt));
  runUntilIdle(engine);
  require(events.starts.back() == 1056 && executor.prefillChunks.at(2) == 4 &&
              capturedStates(executor, 2) ==
                  std::vector<std::pair<uint32_t, uint64_t>>{{4096, 5152}, {8000, 8001}},
          "a checkpoint or a replay point inside a command was not captured there");
  auto counters = engine.snapshot();
  require(counters.checkpointPublications == 1 && counters.replayStatePublications == 2 &&
              counters.stateCaptures == 3 && counters.captureCuts == 0 &&
              resources.lookup(std::vector<uint32_t>(prompt.begin(), prompt.begin() + 4097), {})
                      .resumeBoundary() == 4096 &&
              resources.lookup(prompt, {}).resumeBoundary() == 8000,
          "the captured states were not published");

  engine.submit(request(3, branchingConversation(10'000)));
  runUntilIdle(engine);
  engine.submit(request(4, branchingConversation(20'000)));
  runUntilIdle(engine);
  require(executor.prefillChunks.at(4) == 2 &&
              capturedStates(executor, 4) ==
                  std::vector<std::pair<uint32_t, uint64_t>>{{2112, 2273}, {2272, 2273}},
          "a junction and a replay point in one command were not both captured");
  counters = engine.snapshot();
  require(counters.junctionMaterializations == 1 && counters.replayStatePublications == 4 &&
              counters.stateCaptures == 6 && counters.captureCuts == 0,
          "the branch's states were not published");
  engine.submit(request(5, branchingConversation(30'000)));
  runUntilIdle(engine);
  require(events.starts.back() == 2112, "a third branch did not resume from the junction");
}

// Refused buffers for a boundary inside a command end the command there, after the earlier states are captured.
void testRefusedBuffersEndTheCommandAtTheirBoundary() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.stateTier->ready = true;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(1, branchingConversation(10'000)));
  runUntilIdle(engine);
  executor.denyReservationAt = 2272;
  const std::vector<uint32_t> branch = branchingConversation(20'000);
  engine.submit(request(2, branch));
  runUntilIdle(engine);
  const auto counters = engine.snapshot();
  require(executor.prefillChunks.at(2) == 3 &&
              capturedStates(executor, 2) ==
                  std::vector<std::pair<uint32_t, uint64_t>>{{2112, 2272}} &&
              counters.captureCuts == 1 && counters.junctionMaterializations == 1,
          "refused buffers did not end the command at their boundary");
  require(executor.diskSnapshots == 1 && counters.diskStatePublications == 1 &&
              counters.replayStatePublications == 2 &&
              counters.replayStatePublicationFailures == 0,
          "the refused boundary's state did not go to disk");
  std::vector<uint32_t> next = branch;
  next.resize(next.size() + 40, 7);
  executor.restoreControl->ready = true;
  engine.submit(request(3, next));
  runUntilIdle(engine);
  require(events.starts.back() == 2272 && executor.diskReads == 1,
          "the next turn did not resume from the state on disk");
}

// KV past a boundary is optional to a command: refused, the command ends at the boundary and captures it, and the
// lane neither waits nor yields.
void testRefusedKvPastABoundaryEndsTheCommandThere() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  // The extent the last token's page needs comes once the first command ran.
  storage.growthAllowed = [&](uint32_t extent) {
    return extent == 0 || executor.prefillChunks.contains(1);
  };
  engine.submit(request(1, std::vector<uint32_t>(129, 1)));
  runUntilIdle(engine);
  const auto counters = engine.snapshot();
  require(executor.prefillChunks.at(1) == 2 &&
              capturedStates(executor, 1) ==
                  std::vector<std::pair<uint32_t, uint64_t>>{{128, 128}} &&
              counters.captureCuts == 1 && counters.stateCaptures == 0,
          "refused KV did not end the command at the boundary");
  require(executor.suspensions == 0 && counters.resourceSuspensions == 0 &&
              counters.replayStatePublications == 1 && events.completedCount == 1 &&
              events.failedCount == 0,
          "refused KV past a boundary made the lane wait, yield or fail");
}

// A checkpoint a command passes on its way to a state that retires it is not captured, and that state takes buffers
// of its own class.
void testPassedCheckpointLeavesTheJunctionItsOwnClass() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.stateTier->ready = true;
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({.prefillCheckpointTokens = 3072}), resources, executor, events);
  guardReleases(storage, engine);
  bool memoryFull = false;
  executor.reservationRoom = [&] {
    if (*executor.evictedStateBytes >= 64) {
      *executor.evictedStateBytes -= 64;
      return true;
    }
    return !memoryFull;
  };
  // Another conversation leaves the prompt at 3584 for a replay point of its
  // own at 3680, so the request plans a junction there, past its checkpoint
  // at 3072: both lie in its command [2048, 4096).
  std::vector<uint32_t> prompt(6001);
  std::iota(prompt.begin(), prompt.end(), 1);
  std::vector<uint32_t> other(prompt.begin(), prompt.begin() + 3584);
  other.resize(3685, 0);
  engine.submit(request(1, other));
  runUntilIdle(engine);
  engine.submit(request(3, std::vector<uint32_t>(200, 7)));
  runUntilIdle(engine);
  // Memory fills: a state's buffers come only from a state the cache lets
  // go, and while the engine may not grow, from no free pages.
  host.pause(true);
  memoryFull = true;
  engine.submit(request(2, prompt));
  double now = 100;
  tickUntil(engine, now,
            [&] {
              return executor.prefillChunks.contains(2) && executor.prefillChunks.at(2) == 2 &&
                     !engine.commandInFlight();
            },
            "the request did not run its second command");
  const auto counters = engine.snapshot();
  require(capturedStates(executor, 2) == std::vector<std::pair<uint32_t, uint64_t>>{{3584, 4096}} &&
              counters.captureCuts == 0 && counters.junctionMaterializations == 1 &&
              counters.recycledStatePublications == 1 && counters.checkpointPublications == 0 &&
              executor.diskSnapshots == 0,
          "a passed checkpoint's refusal ended the command, or it was written to the SSD");
  host.pause(false);
  memoryFull = false;
  tickUntil(engine, now, [&] { return idle(engine); }, "the request did not finish");
  require(events.completedCount == 3 && events.failedCount == 0 &&
              resources.probe(prompt, {}).cachedTokens() == 5984,
          "the request did not publish its replay point");
}

// A command refused the KV past a checkpoint it would pass ends there and captures it, so the lane's progress stays
// in memory.
void testRefusedKvPastAPassedCheckpointCapturesIt() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.stateTier->ready = true;
  Events events;
  engine::Engine engine(test::engineConfig({.prefillCheckpointTokens = 3072}), resources,
                        executor, events);
  guardReleases(storage, engine);
  // A queued Background sibling shares the prompt up to 3584: the request
  // plans a junction there for it, past its checkpoint at 3072, both in its
  // command [2048, 4096). The pages past 3072 come once that command ran.
  std::vector<uint32_t> prompt(6001);
  std::iota(prompt.begin(), prompt.end(), 1);
  std::vector<uint32_t> sibling(prompt.begin(), prompt.begin() + 3584);
  sibling.resize(3685, 0);
  const auto commands = [&] {
    const auto found = executor.prefillChunks.find(1);
    return found == executor.prefillChunks.end() ? 0U : found->second;
  };
  storage.growthAllowed = [&](uint32_t) {
    return storage.allocatedPages() < 3072 / KvCache::pageTokens || commands() >= 2;
  };
  engine.submit(request(1, prompt));
  auto background = request(2, sibling);
  background.priority = RequestPriority::Background;
  engine.submit(std::move(background));
  double now = 1;
  tickUntil(engine, now, [&] { return commands() == 2 && !engine.commandInFlight(); },
            "the request did not run its second command");
  const auto counters = engine.snapshot();
  require(capturedStates(executor, 1) == std::vector<std::pair<uint32_t, uint64_t>>{{3072, 3072}} &&
              counters.captureCuts == 1 && counters.checkpointPublications == 1 &&
              executor.diskSnapshots == 0,
          "the command did not end at the checkpoint and capture it");
  tickUntil(engine, now, [&] { return idle(engine); }, "the requests did not finish");
  require(events.starts.back() == 3584 && events.completedCount == 2 &&
              events.failedCount == 0 && engine.snapshot().junctionMaterializations == 1,
          "the sibling did not resume from the junction");
}

// A command with a non-finite row commits nothing: its captured states go back to the pool unpublished.
void testNonFiniteCommandDropsItsCaptures() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.nonFinitePrefills.insert(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(65, 1)));
  runUntilIdle(engine);
  const auto counters = engine.snapshot();
  require(executor.reservations == 1 && executor.captures.empty() &&
              *executor.evictedStateBytes == 64 &&
              counters.resources.stateCache.entries == 0 &&
              counters.replayStatePublications == 0 &&
              counters.replayStatePublicationFailures == 0 &&
              events.failures == std::vector<std::string>{"model_result_invalid"},
          "a non-finite command published its state or kept its buffers");
}

// Two lanes computing the same prompt both capture its replay state; the
// later one finds it cached and drops its own, whose buffers go back.
void testDuplicateCaptureIsDropped() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(5001, 7);
  auto producer = request(1, prompt);
  producer.priority = RequestPriority::Background;
  engine.submit(std::move(producer));
  static_cast<void>(engine.tick(0));
  // It does not wait for a producer of a lower priority.
  auto duplicate = request(2, prompt);
  duplicate.priority = RequestPriority::Foreground;
  engine.submit(std::move(duplicate));
  runUntilIdle(engine);
  const auto counters = engine.snapshot();
  require(executor.prefillRows == 2 * prompt.size() && executor.captures.size() == 2 &&
              counters.stateCaptures == 2 && counters.replayStatePublications == 1 &&
              counters.deduplicatedStatePublications == 1 &&
              counters.resources.stateCache.entries == 1 &&
              *executor.evictedStateBytes == 64,
          "a duplicate capture was published or kept its buffers");
}

// Two lanes in one command capture three states: one lane its branch's
// junction and replay point, the other its replay point.
void testOneCommandCapturesSeveralLanesStates() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(1, branchingConversation(10'000)));
  runUntilIdle(engine);
  engine.submit(request(2, branchingConversation(20'000)));
  double now = 1;
  tickUntil(engine, now,
            [&] { return executor.prefillChunks.contains(2) && !engine.commandInFlight(); },
            "the branch did not run its first command");
  engine.submit(request(3, std::vector<uint32_t>(65, 3)));
  runUntilIdle(engine);
  require(executor.prefillWidths.back() == 2 &&
              capturedStates(executor, 2) ==
                  std::vector<std::pair<uint32_t, uint64_t>>{{2112, 2273}, {2272, 2273}} &&
              capturedStates(executor, 3) == std::vector<std::pair<uint32_t, uint64_t>>{{64, 65}},
          "one command did not capture both lanes' states");
  const auto counters = engine.snapshot();
  require(counters.junctionMaterializations == 1 && counters.replayStatePublications == 3 &&
              counters.resources.stateCache.entries == 4 && events.completedCount == 3,
          "the states one command captured were not all published");
}

// While every lane is resident a waiting request is a concurrency wait, with no cache probe or admission attempt,
// and no memory wait counts against its limit.
void testFullLanesSkipAdmissionAttempts() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  // The first pass starts the three short prompts and tries request 5, the
  // fourth of its batch, which meets pressure. Request 4 arrived before it
  // with a longer prompt: request 5 leads it and holds it back, and takes the
  // last lane at its retry.
  executor.beginGrowthBlocked = [&] { return executor.beginAttempts == 4; };
  Events events;
  engine::Engine engine(test::engineConfig({.resourceWaitTimeoutMilliseconds = 1000.0}),
                        resources, executor, events);
  guardReleases(storage, engine);
  const auto submit = [&](uint64_t id, uint32_t promptTokens) {
    auto value = request(id, std::vector<uint32_t>(promptTokens, id));
    value.maxNewTokens = 100'000;
    value.deadlineMilliseconds = 100'000;
    engine.submit(std::move(value));
  };
  for (uint64_t id = 1; id <= 3; ++id)
    submit(id, 33);
  submit(4, 129);
  submit(5, 64);
  require(engine.tick(1) && engine.resourceWaitSnapshot(1).memory == 1 &&
              executor.lastBeginId == 5,
          "fixture did not leave one request waiting for memory");
  for (double now = 2; now <= 6; ++now)
    static_cast<void>(engine.tick(now));
  require(executor.requests.size() == 3 && executor.beginAttempts == 4,
          "the request the refused one leads was tried while it backed off");
  static_cast<void>(engine.tick(200));
  require(executor.requests.size() == 4 && executor.requests.contains(5) &&
              executor.beginAttempts == 5,
          "fixture did not make every lane resident");

  for (double now : {201.0, 1200.0, 1201.0})
    static_cast<void>(engine.tick(now));
  const auto waiting = engine.resourceWaitSnapshot(1201);
  require(executor.beginAttempts == 5 && events.failedCount == 0 &&
              waiting.concurrency == 1 && waiting.memory == 0,
          "full lanes retried admission or kept the memory wait limit");
  engine.cancel(1);
  require(engine.tick(1202) && executor.beginAttempts == 6 &&
              events.startIds.back() == 4,
          "a released lane did not admit the waiting request");
  for (uint64_t id : {2, 3, 4, 5})
    engine.cancel(id);
  for (double now = 1203; now < 1220 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && resources.snapshot().activeRequests == 0,
          "full-lane fixture leaked its lanes");
}

void testConcurrencyLimitDoesNotEvictCache() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(request(22, std::vector<uint32_t>(65, 22)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1,
          "concurrency-limit setup did not retain a cache state");

  engine.submit(request(23, std::vector<uint32_t>(4097, 23)));
  require(engine.tick(1) && engine.tick(2),
          "resident request did not start long prefill");
  engine.submit(request(24, {24}));
  require(engine.tick(3), "saturated engine made no resident progress");
  const auto waiting = engine.resourceWaitSnapshot(13.0);
  require(waiting.concurrency == 1 && waiting.memory == 0 &&
              waiting.suspended == 0 && waiting.oldestWaitMilliseconds == 10.0,
          "concurrency wait diagnostics were confused with memory pressure");

  const auto saturated = resources.snapshot();
  require(saturated.stateCache.entries == cached.stateCache.entries &&
              saturated.stateCache.evictions == cached.stateCache.evictions,
          "lane saturation was mistaken for memory pressure");

  engine.cancel(23);
  engine.cancel(24);
  runUntilIdle(engine);
}

// Under host pressure, refused buffers still recycle one cached state (memory-neutral); the state footprint does
// not grow and KV is untouched.
void testHostPressureStillRecyclesLruStateForRefusedBuffers() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({}), resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(210, std::vector<uint32_t>(65, 210)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  const uint32_t attempts = executor.reservationAttempts;

  host.setPressure(MemoryPressure::Warning);
  executor.deniedReservations = 1;
  engine.submit(request(211, std::vector<uint32_t>(65, 211)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(idle(engine) && events.completedCount == 2 &&
              executor.reservationAttempts == attempts + 2 &&
              after.recycledStatePublications == 1 &&
              after.replayStatePublications == 2 &&
              after.replayStatePublicationFailures == 0,
          "refused buffers under host pressure did not come from recycling");
  require(after.resources.stateCache.evictions ==
                  cached.stateCache.evictions + 1 &&
              after.resources.stateCache.entries == cached.stateCache.entries &&
              after.resources.stateCache.bytes == cached.stateCache.bytes &&
              after.resources.pool.pagesPrefix == cached.pool.pagesPrefix + 2,
          "recycling under host pressure grew state or drained KV cache");
}

// The governor's host pause as the engine meets it: other applications leave
// the host short of the governor's margin (GovernedHost::pause), and the
// governor refuses KV growth and a lane's state for the host unless they are
// memory a request in service needs. The engine's limit is the governor's.
struct HostPause final {
  explicit HostPause(const test::GovernorLimits &limits = {}) : host(limits) {}
  void pause() { host.pause(true); }
  [[nodiscard]] MemoryGovernor &governor() noexcept { return host.governor; }

  GovernedHost host;
  // The pages no request held each time the pause let a request grow.
  std::vector<uint32_t> reusableAtGrowth;

  void attach(EngineConfig &config, test::TestKvStorage &storage, KvPool &pool) {
    config = host.connect(std::move(config));
    storage.through(host.memory, [this, &pool, admission = host.governor.allocationAdmission()](
                                     uint64_t bytes, const std::function<void()> &allocate) {
      const KvPoolSnapshot pages = pool.snapshot();
      const metal::AllocationResult result = admission(bytes, allocate);
      if (result && host.paused())
        reusableAtGrowth.push_back(pages.pagesAllocated - pages.pagesActive);
      return result;
    });
  }
  // A start allocates a lane's state buffers of a page.
  void attach(Executor &executor) { host.admitBegins(executor, 4096); }
};

// Host pressure pauses growth, not a request in service: it takes cached pages no request holds first, then grows,
// and is never suspended for the host.
void testRequestInServiceGrowsThroughTheHostPause() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  HostPause host;
  EngineConfig config = test::engineConfig();
  host.attach(config, storage, pool);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  // A finished prompt leaves two cached blocks under a state.
  engine.submit(request(1, std::vector<uint32_t>(65, 1)));
  runUntilIdle(engine);
  require(resources.snapshot().pool.pagesPrefix == 2 &&
              resources.snapshot().stateCache.entries == 1,
          "host pause fixture did not cache the first prompt");
  executor.decodeFinishes = false;
  auto running = request(2, std::vector<uint32_t>(65, 2));
  running.maxNewTokens = 400;
  running.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(running));
  double now = 1;
  while (now < 100 && events.outputs[2].empty())
    static_cast<void>(engine.tick(now++));
  require(!events.outputs[2].empty(), "the request did not start decoding");

  host.pause();
  const uint32_t allocated = pool.snapshot().pagesAllocated;
  while (now < 2000 && events.completedCount < 2)
    static_cast<void>(engine.tick(now++));
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.outputs[2].size() == 400 && executor.suspensions == 0 &&
              engine.snapshot().resourceSuspensions == 0,
          "host pressure stopped a request in service");
  require(pool.snapshot().pagesAllocated > allocated && !host.reusableAtGrowth.empty() &&
              std::all_of(host.reusableAtGrowth.begin(), host.reusableAtGrowth.end(),
                          [](uint32_t pages) { return pages == 0; }),
          "the request grew while cached pages no request held could serve it");
  require(resources.lookup(std::vector<uint32_t>(65, 1), {}).kvBoundary == 0,
          "the paused request did not take the idle cached pages first");
}

// A growth the host refuses takes only what frees now: a leaf whose copies are on their way to the
// SSD cache stays and no write starts for it, and the host exemption grows the lane.
void testAHostRefusedGrowthTakesOnlyWhatFreesNow() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier(true);
  tier.transferLimit = 64;
  tier.capacity = 128;
  engine::Cache resources(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.keptTier = std::make_shared<OffloadControl>();
  executor.keptTier->ready = true;
  Events events;
  HostPause host;
  EngineConfig config = test::engineConfig({.maxContext = 102400});
  host.attach(config, storage, pool);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  // A finished conversation's 64 pages, the first 16 of whose copies start
  // after the persistent tier's delay and stay in flight.
  const std::vector<uint32_t> first(2060, 7);
  engine.submit(request(1, first));
  tickUntil(engine, now, [&] { return idle(engine); }, "the first request did not finish");
  now = engine.nextWakeupMilliseconds().value_or(now);
  static_cast<void>(engine.tick(now++));
  require(tier.inFlight() == 16 && resources.lookup(first, {}).kvBoundary == 2048,
          "the leaf's copies did not start");
  executor.decodeFinishes = false;
  auto running = request(2, std::vector<uint32_t>(65, 2));
  running.maxNewTokens = 400;
  running.deadlineMilliseconds = 1e9;
  engine.submit(std::move(running));
  tickUntil(engine, now, [&] { return !events.outputs[2].empty(); }, "the lane did not decode");
  host.pause();
  const uint32_t writes = tier.demotions;
  const uint32_t allocated = pool.snapshot().pagesAllocated;
  for (uint32_t step = 0; step < 3000 && events.completedCount < 2; ++step)
    static_cast<void>(engine.tick(now++));
  require(events.completedCount == 2 && events.failedCount == 0 && events.outputs[2].size() == 400 &&
              engine.snapshot().resourceSuspensions == 0 && pool.snapshot().pagesAllocated > allocated,
          "the host exemption did not grow the lane through the pause");
  require(tier.demotions == writes && resources.lookup(first, {}).kvBoundary == 2048,
          "a growth the host refused took the leaf on its way to the SSD cache, or wrote it");
}

// With nothing in service the first request starts through the host pause; one beside it waits without evicting and
// starts once the first is done.
void testFirstRequestStartsThroughTheHostPause() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  HostPause host;
  host.pause();
  EngineConfig config = test::engineConfig();
  host.attach(config, storage, pool);
  host.attach(executor);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  auto first = request(1, std::vector<uint32_t>(65, 1));
  first.maxNewTokens = 40;
  first.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(first));
  double now = 1;
  while (now < 100 && events.outputs[1].empty())
    static_cast<void>(engine.tick(now++));
  require(!events.outputs[1].empty() && executor.beginAttempts == 2 &&
              pool.snapshot().pagesAllocated >= 4,
          "the first request did not start through the host pause");

  auto second = request(2, std::vector<uint32_t>(65, 2));
  second.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(second));
  const CacheSnapshot before = resources.snapshot();
  static_cast<void>(engine.tick(now++));
  static_cast<void>(engine.tick(now++));
  require(!executor.requests.contains(2) && events.startIds.size() == 1 &&
              engine.resourceWaitSnapshot(now).memory == 1 &&
              resources.snapshot().pool.pagesPrefix >= before.pool.pagesPrefix &&
              resources.snapshot().stateCache.evictions == before.stateCache.evictions,
          "a second request grew, or evicted for the host, beside one in service");
  while (now < 1000 && events.completedCount < 2)
    static_cast<void>(engine.tick(now++));
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.startIds == std::vector<uint64_t>({1, 2}) &&
              executor.suspensions == 0,
          "the waiting request did not start once nothing was in service");
}

// A request whose start needs no new memory still starts beside one in
// service, as before, and then grows like it.
void testRequestThatNeedsNoGrowthStartsUnderTheHostPause() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  HostPause host;
  EngineConfig config = test::engineConfig();
  host.attach(config, storage, pool);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  auto first = request(1, std::vector<uint32_t>(33, 1));
  first.maxNewTokens = 300;
  first.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(first));
  double now = 1;
  while (now < 100 && events.outputs[1].empty())
    static_cast<void>(engine.tick(now++));
  host.pause();
  // The lane's buffers are pooled and the first extent has free pages.
  auto second = request(2, std::vector<uint32_t>(33, 2));
  second.maxNewTokens = 300;
  second.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(second));
  while (now < 100 && events.outputs[2].empty())
    static_cast<void>(engine.tick(now++));
  require(!events.outputs[2].empty() && host.reusableAtGrowth.empty(),
          "a request that needed no new memory waited under the host pause");
  while (now < 2000 && events.completedCount < 2)
    static_cast<void>(engine.tick(now++));
  require(events.completedCount == 2 && events.failedCount == 0 &&
              executor.suspensions == 0 && !host.reusableAtGrowth.empty(),
          "requests in service did not grow through the host pause");
}

// A suspended request resumes like a new one: beside a request in service
// its pages wait for the host, and it resumes once that request has finished.
void testSuspendedRequestWaitsForTheHostBesideOneInService() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  // The engine's limit is two extents.
  HostPause host({.limitBytes = 8 * 4096});
  EngineConfig config = test::engineConfig();
  config.resourceWaitTimeoutMilliseconds = 100.0;
  host.attach(config, storage, pool);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {1, 2}) {
    auto value = request(id, std::vector<uint32_t>(id == 1 ? 65 : 33, id));
    value.maxNewTokens = 150;
    value.deadlineMilliseconds = 1'000'000;
    engine.submit(std::move(value));
  }
  double now = 1;
  while (now < 1000 && !executor.suspensions)
    static_cast<void>(engine.tick(now += 5));
  require(engine.snapshot().resourceSuspensions == 1 &&
              !executor.requests.at(2).resident && executor.requests.at(1).resident,
          "the limit did not suspend the shorter request");

  host.pause();
  while (now < 5000 && events.completedCount < 1) {
    require(engine.snapshot().resourceResumptions == 0,
            "a suspended request grew through the pause beside one in service");
    static_cast<void>(engine.tick(now += 5));
  }
  // Its lane resumed and gave way again each time its pages were refused.
  require(events.completedCount == 1 && executor.resumeAttempts > 0 &&
              events.failedCount == 0 && events.outputs[1].size() == 150,
          "the request in service did not finish, or the suspended one never retried");
  while (now < 10000 && events.completedCount < 2)
    static_cast<void>(engine.tick(now += 5));
  require(events.completedCount == 2 && events.failedCount == 0 &&
              engine.snapshot().resourceResumptions == 1 &&
              events.outputs[2].size() == 150,
          "the suspended request did not resume once nothing was in service");
}

// The engine's own limit still binds a request in service under the pause: alone with nothing to reclaim it fails
// at once as at capacity.
void testEngineLimitBindsThroughTheHostPause() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  // The host refuses first unless a request in service needs the pages, then
  // the engine's limit of two extents.
  HostPause host({.limitBytes = 8 * 4096});
  EngineConfig config = test::engineConfig();
  host.attach(config, storage, pool);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  auto value = request(1, std::vector<uint32_t>(65, 1));
  value.maxNewTokens = 1000;
  value.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(value));
  double now = 1;
  while (now < 100 && events.outputs[1].empty())
    static_cast<void>(engine.tick(now++));
  host.pause();
  while (now < 1000 && !events.failedCount && !events.completedCount)
    static_cast<void>(engine.tick(now++));
  require(events.failures == std::vector<std::string>{"capacity_exhausted"} &&
              executor.suspensions == 0 && pool.snapshot().pagesAllocated == 8,
          "a lone request at the engine's limit waited for the host instead of failing");
  const std::string &message = events.failureDetails.front().first;
  require(message.starts_with("could not allocate ") &&
              message.find("--max-memory") != std::string::npos &&
              message.find("--max-context") != std::string::npos,
          "a lone request at the engine's limit did not name the limits that would fit it");
}

// Under the host pause a lane in service short of a page takes cached KV, not cached states, whose buffers would
// give it no page.
void testPausedPageShortageKeepsCachedStates() {
  test::TestKvStorage storage(64, 4096, 1);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  HostPause host;
  EngineConfig config = test::engineConfig();
  host.attach(config, storage, pool);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  // Cached: a checkpoint, an ordinary state below it, and KV below both,
  // allocated as a start's.
  {
    const test::ClaimScope fixture(host.governor(), ClaimClass::Floor);
    resources.beginRequest(900);
    require(resources.ensureTokens(900, 128).granted(), "fixture KV failed");
    resources.publishCommittedBlocks(900, std::vector<uint32_t>(129, 9), 128, {});
    resources.publishCompositeState(resources.blockAt(900, 32), std::make_shared<State>(), true);
    resources.publishCompositeState(resources.blockAt(900, 64), std::make_shared<State>(), false);
    resources.endRequest(900);
  }
  // The decode needs one more page, at its 57th token.
  auto running = request(1, std::vector<uint32_t>(33, 1));
  running.maxNewTokens = 30;
  engine.submit(std::move(running));
  double now = 1;
  tickUntil(engine, now, [&] { return !events.outputs[1].empty(); },
            "the request did not decode");
  host.pause();
  const StateCacheSnapshot before = resources.snapshot().stateCache;
  tickUntil(engine, now, [&] { return idle(engine); }, "the request did not finish");
  const StateCacheSnapshot after = resources.snapshot().stateCache;
  require(events.completedCount == 1 && engine.snapshot().resourceSuspensions == 0 &&
              host.reusableAtGrowth.empty(),
          "the lane did not take a cached page under the pause");
  require(after.entries == before.entries && after.evictions == before.evictions &&
              after.checkpointEvictions == 0,
          "a page shortage under the pause evicted cached states");
}

// A request that cannot start while an earlier lane holds memory waits for that lane however long it works; then
// its wait limit runs as usual.
void testAdmissionWaitsOutEarlierLanes() {
  for (const bool hostRecovers : {true, false}) {
    test::TestKvStorage storage(32, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    EngineConfig config = test::engineConfig();
    config.resourceWaitTimeoutMilliseconds = 100.0;
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    auto running = request(310, {310});
    running.maxNewTokens = 1000;
    engine.submit(std::move(running));
    static_cast<void>(engine.tick(1));
    // The host has no memory for a second lane while the first runs.
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    executor.beginGrowthBlocked = [&] { return !hostRecovers || !events.completedCount; };
    engine.submit(request(311, {311}));
    double now = 1;
    while (now < 500)
      static_cast<void>(engine.tick(now += 50));
    require(!events.completedCount && events.failedCount == 0,
            "a request waiting for a resident lane's memory timed out while it ran");
    executor.decodeFinishes = true;
    while (!events.completedCount && now < 1000)
      static_cast<void>(engine.tick(now += 10));
    for (const double end = now + 150; now < end && !idle(engine);)
      static_cast<void>(engine.tick(now += 10));
    if (hostRecovers)
      require(events.completedCount == 2 && events.failedCount == 0,
              "the waiting request did not start once the lane finished");
    else
      require(events.completedCount == 1 &&
                  events.failures == std::vector<std::string>{"resource_timeout"},
              "a wait that no resident lane could end did not expire");
    require(idle(engine) && executor.requests.empty(), "the resource wait leaked a request");
  }
}

// A lane in service refused memory counts the lanes admitted before its first refusal: a start submitted after it and
// admitted after that refusal does not extend its wait, however often the lane is refused again while that start runs.
// The wait fails at its limit, retryably, while the start's command is still in flight.
void testStartsAdmittedBesideALaneInServiceDoNotExtendItsWait() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  tier.capacity = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // Two pages free beside the cached blocks, and no extent more. The lane's
  // prompt needs the pages their demotions return, which never land: it
  // keeps its lane and waits for them.
  cacheBlocksUnderDiskStates(cache, 6);
  storage.growthBlocked = true;
  EngineRequest lane = request(1, std::vector<uint32_t>(161, 7));
  lane.deadlineMilliseconds = 1e9;
  engine.submit(std::move(lane));
  double now = 1;
  static_cast<void>(engine.tick(now));
  require(tier.demotions == 3 && executor.requests.contains(1) && executor.prefillRows == 0,
          "the lane in service was not refused the pages on their way back");
  // A start whose prompt continues the newest cached block, which no
  // demotion took, outranks the lane by the tokens it has done: its claim
  // takes the room the lane's waits for, and its state comes back from disk
  // at once. It decodes beside the lane, which is retried, and refused,
  // several times meanwhile.
  executor.restoreControl->ready = true;
  EngineRequest beside = request(2, std::vector<uint32_t>(33, 905));
  beside.maxNewTokens = 100000;
  beside.deadlineMilliseconds = 1e9;
  engine.submit(std::move(beside));
  tickUntil(engine, now, [&] { return events.outputs.contains(2); }, "the start beside the lane did not decode");
  const double decodedAt = now;
  for (const double end = now + 3 * kResourceRetryBackoffMilliseconds; now < end;)
    static_cast<void>(engine.tick(now += 10));
  require(executor.requests.contains(1) && !executor.prefillChunks[1] && executor.suspensions == 0 &&
              events.failedCount == 0,
          "the lane did not keep waiting in service beside the start");
  // The start's next decode command stays in flight past the lane's limit.
  executor.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
  while (!events.failedCount && now < decodedAt + 2 * kResourceWaitTimeoutMilliseconds)
    static_cast<void>(engine.tick(now += 100));
  require(events.failures == std::vector<std::string>{"resource_timeout"} && events.failureDetails.back().second &&
              now <= decodedAt + kResourceWaitTimeoutMilliseconds + 100,
          "a start admitted beside a lane in service after its refusal extended its wait");
  require(engine.commandInFlight() && executor.requests.contains(2) && !events.completedCount,
          "the start's command was not in flight as the wait expired");
  *executor.holdDecodeUntil = true;
  engine.cancel(2);
  for (const double end = now + 1000; now < end && !idle(engine);)
    static_cast<void>(engine.tick(now += 10));
  require(idle(engine) && executor.requests.empty(), "the resource wait leaked a request");
}

// A lane admitted in the pass that refused an earlier request holds memory that wait is for, so its work restarts
// the wait's limit.
void testLaneAdmittedBeforeARefusalHoldsTheWaitOpen() {
  for (const bool lanesFull : {false, true}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    executor.beginGrowthBlocked = [&] { return executor.lastBeginId == 330; };
    Events events;
    EngineConfig config = test::engineConfig();
    config.resourceWaitTimeoutMilliseconds = 50.0;
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    double now = 1;
    // Occupants in every lane: both requests wait for one until they are
    // cancelled.
    const uint64_t occupants = lanesFull ? model::ExecutionLimits::maximumBatchWidth : 0;
    for (uint64_t id = 1; id <= occupants; ++id) {
      auto occupant = request(id, {static_cast<uint32_t>(id)});
      occupant.maxNewTokens = 1000;
      engine.submit(std::move(occupant));
    }
    tickUntil(engine, now, [&] { return executor.requests.size() == occupants; },
              "the occupying lanes did not start");
    engine.submit(request(330, std::vector<uint32_t>(1025, 330)));
    auto later = request(331, {331});
    later.maxNewTokens = 1000;
    engine.submit(std::move(later));
    if (lanesFull) {
      tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).concurrency == 2; },
                "the requests did not wait for a lane");
      for (uint64_t id = 1; id <= occupants; ++id)
        engine.cancel(id);
    }
    tickUntil(engine, now, [&] { return executor.requests.contains(331); },
              "the later request did not start");
    require(!executor.requests.contains(330) && engine.resourceWaitSnapshot(now).memory == 1,
            "the later request did not start in the pass that refused the first");
    for (const double end = now + 300; now < end;)
      static_cast<void>(engine.tick(now += 10));
    require(events.failedCount == 0,
            "a lane admitted before the refusal did not hold the wait open");
    engine.cancel(331);
    const double cancelledAt = now;
    while (!events.failedCount && now < cancelledAt + 1000)
      static_cast<void>(engine.tick(now += 10));
    require(now <= cancelledAt + 70 &&
                events.failures == std::vector<std::string>{"resource_timeout"},
            "the wait did not expire once the lane admitted before it was gone");
  }
}

// Shorter starts the scheduler leads ahead of a refused start, admitted after its first refusal, hold memory that wait
// is for: it waits while they work, one after another, past its limit, and starts once their memory frees; with none
// freed, its wait expires at its limit once they are gone.
void testStartsAdmittedAheadOfARefusedStartHoldItsWaitOpen() {
  for (const bool frees : {true, false}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    // The host has no memory for the long prompt while the shorter ones run.
    bool held = true;
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    executor.beginGrowthBlocked = [&] { return executor.lastBeginId == 340 && (held || !frees); };
    Events events;
    EngineConfig config = test::engineConfig();
    // Longer than the retry backoff: the refused start is tried again while
    // the shorter ones run.
    config.resourceWaitTimeoutMilliseconds = 300.0;
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    double now = 1;
    engine.submit(request(340, std::vector<uint32_t>(1025, 340)));
    static_cast<void>(engine.tick(now));
    require(engine.resourceWaitSnapshot(now).memory == 1 && !executor.requests.contains(340),
            "the long prompt did not wait for memory");
    // Each shorter prompt leads the refused one and starts beside its wait;
    // the second takes over from the first.
    for (const uint64_t id : {341, 342}) {
      auto shorter = request(id, {static_cast<uint32_t>(id)});
      shorter.maxNewTokens = 100000;
      engine.submit(std::move(shorter));
      tickUntil(engine, now, [&] { return executor.requests.contains(id); }, "a shorter prompt did not start");
      if (id == 342)
        engine.cancel(341);
      for (const double end = now + 400; now < end;)
        static_cast<void>(engine.tick(now += 10));
    }
    require(events.failedCount == 0 && !executor.requests.contains(340) &&
                engine.resourceWaitSnapshot(now).memory == 1,
            "a wait expired while the starts admitted ahead of it worked");
    engine.cancel(342);
    held = false;
    const double cancelledAt = now;
    while (!events.failedCount && !executor.requests.contains(340) && now < cancelledAt + 1000)
      static_cast<void>(engine.tick(now += 10));
    if (frees)
      require(executor.requests.contains(340) && events.failedCount == 0,
              "the refused start did not start once the memory freed");
    else
      require(now <= cancelledAt + 320 && events.failures == std::vector<std::string>{"resource_timeout"},
              "the wait did not expire once the starts admitted ahead of it were gone");
    engine.cancel(340);
    for (const double end = now + 200; now < end && !idle(engine);)
      static_cast<void>(engine.tick(now += 10));
    require(idle(engine) && executor.requests.empty(), "the resource wait leaked a request");
  }
}

void testSingletonHostPressureWaitRecoversOrTerminates() {
  for (uint32_t outcome = 0; outcome < 4; ++outcome) {
    test::TestKvStorage storage(32, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    GovernedHost host;
    EngineConfig config = host.connect({});
    config.resourceWaitTimeoutMilliseconds = outcome == 3 ? 100.0 : 30000.0;
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    engine.submit(request(220, std::vector<uint32_t>(65, 220)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();

    host.setPressure(MemoryPressure::Warning);
    storage.growthBlocked = true;
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    auto value = request(221, std::vector<uint32_t>(161, 221));
    value.deadlineMilliseconds = 300.0;
    engine.submit(std::move(value));
    static_cast<void>(engine.tick(20.0));
    static_cast<void>(engine.tick(21.0));
    // Suspension drops the lane's planned boundary silently: it is neither a
    // reservation attempt nor a publication failure.
    require(executor.suspensions == 1 && events.capacityExhaustedCount == 0 &&
                resources.snapshot().pool.pagesPrefix == cached.pool.pagesPrefix &&
                resources.snapshot().stateCache.evictions ==
                    cached.stateCache.evictions &&
                executor.reservationAttempts == 1 &&
                engine.snapshot().replayStatePublicationFailures == 0,
            "singleton host pressure killed active work or drained its cache");
    const uint32_t attempts = executor.resumeAttempts;
    static_cast<void>(engine.tick(50.0));
    require(executor.resumeAttempts == attempts,
            "suspended host-pressure request ignored resource backoff");
    const auto waiting = engine.resourceWaitSnapshot(50.0);
    require(waiting.memory == 1 && waiting.concurrency == 0 &&
                waiting.suspended == 1 && waiting.oldestWaitMilliseconds >= 29.0,
            "resource diagnostics lost suspension reason or original wait time");

    if (outcome == 0) {
      host.setPressure(MemoryPressure::Normal);
      storage.growthBlocked = false;
      for (double now = 121.0; now < 140.0 && !idle(engine); ++now)
        static_cast<void>(engine.tick(now));
      // The resumed lane re-arms its replay boundary and publishes normally.
      require(events.completedCount == 2 && events.failedCount == 0 &&
                  engine.snapshot().replayStatePublications == 2 &&
                  engine.snapshot().replayStatePublicationFailures == 0,
              "singleton request did not recover after pressure cleared");
    } else if (outcome == 1) {
      engine.cancel(221);
      static_cast<void>(engine.tick(60.0));
      require(engine.snapshot().cancelled == 1,
              "host-pressure wait ignored cancellation");
    } else if (outcome == 3) {
      static_cast<void>(engine.tick(150.0));
      require(events.failures == std::vector<std::string>{"resource_timeout"} &&
                  events.failureDetails.back().second &&
                  events.failureDetails.back().first ==
                      "memory did not become available within the resource wait "
                      "limit: macOS is short of memory; close memory-heavy "
                      "applications",
              "persistent memory pressure did not fail with a retryable timeout "
              "that names the shortage");
    } else {
      static_cast<void>(engine.tick(301.0));
      require(events.completedCount + events.failedCount == 2,
              "host-pressure wait ignored deadline");
    }
    require(idle(engine) && executor.requests.empty() &&
                resources.snapshot().activeRequests == 0 &&
                resources.snapshot().pool.pagesActive == 0 &&
                events.capacityExhaustedCount == 0 &&
                resources.snapshot().stateCache.entries ==
                    cached.stateCache.entries + (outcome == 0 ? 1U : 0U),
            "host-pressure wait leaked request state or active KV");
    const auto cleared = engine.resourceWaitSnapshot(400.0);
    require(cleared.memory == 0 && cleared.concurrency == 0 &&
                cleared.suspended == 0 && cleared.oldestWaitMilliseconds == 0.0,
            "terminal work retained resource wait diagnostics");
  }
}

void testKvPressureNarrowsTheRealBatch() {
  test::TestKvStorage storage(4, 4096, 1);
  storage.budgetPages = 3;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(request(30, std::vector<uint32_t>(33, 30)));
  engine.submit(request(31, std::vector<uint32_t>(33, 31)));
  runUntilIdle(engine);

  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "KV pressure failed a lane that could run in a narrower batch");
  require(std::find(executor.prefillWidths.begin(),
                    executor.prefillWidths.end(),
                    1) != executor.prefillWidths.end(),
          "resource pressure did not degrade B2 prefill to real B1 commands");
}

// Required KV growth takes the least recently used cached state from the shared budget instead of failing or
// suspending, and still publishes its replay state.
void testKvGrowthReclaimsCachedStateWhenBudgetIsShared() {
  // An extent of four pages takes the bytes of a state's buffers.
  test::TestKvStorage storage(8, State().bytes() / 4, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({}), resources, executor, events);
  guardReleases(storage, engine);
  host.bookStates(executor);
  // The engine's limit refuses an extent's bytes while the cached state
  // holds the memory the KV shares with it.
  bool limited = false;
  storage.growthAllowed = [&](uint32_t) {
    if (!limited || resources.snapshot().stateCache.entries == 0)
      return true;
    storage.allocationFailure = refusedAtTheLimit(host, State().bytes());
    return false;
  };
  engine.submit(request(230, std::vector<uint32_t>(65, 230)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.pool.pagesAllocated == 4,
          "shared-budget fixture did not cache one state in one extent");
  reachTheLimit(host);
  limited = true;

  engine.submit(request(231, std::vector<uint32_t>(161, 231)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0 &&
              executor.suspensions == 0 && executor.prefillRows == 226,
          "required KV growth failed or suspended behind cached state");
  require(after.resources.stateCache.evictions == 1 &&
              after.resources.stateCache.entries == 1 &&
              after.replayStatePublications == 2 &&
              after.recycledStatePublications == 0 &&
              after.resources.pool.pagesAllocated == 8,
          "KV growth did not take exactly the older cached state");
  require(after.resources.pool.pagesActive == 0 &&
              after.resources.activeRequests == 0,
          "grown request leaked its active resources");
}

// What the ledger frees now for a lane does not cost it its wait for the rest: it waits for the pages that land and
// then runs.
void testWhatFreesNowKeepsALaneWaitingForTheRest() {
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
  // Two pages free beside the cached blocks, and no extent more.
  cacheBlocksUnderDiskStates(cache, 6);
  require(pool.freePageCount() == 2 && pool.snapshot().pagesAllocated == 8, "fixture storage geometry changed");
  storage.growthBlocked = true;
  executor.reclaimableIdleStateBytes = State().bytes();
  engine.submit(request(1, std::vector<uint32_t>(161, 7)));
  static_cast<void>(engine.tick(1));
  require(executor.reclaimedIdleStateBytes == State().bytes() && tier.demotions == 3 &&
              executor.prefillRows == 0 && events.failedCount == 0,
          "a lane that freed idle buffers did not wait for the pages on their way back");
  for (uint32_t step = 2; step < 40 && !idle(engine); ++step) {
    tier.complete();
    static_cast<void>(engine.tick(step));
  }
  require(idle(engine) && executor.prefillRows == 161 && executor.suspensions == 0 && events.failedCount == 0,
          "the lane did not run on the pages the demotions gave back");
}

void testRequiredWorkDoesNotReserveAnExtraPage() {
  for (uint32_t promptTokens : {1U, 24U}) {
    test::TestKvStorage storage(2, 4096, 1);
    storage.budgetPages = 1;
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);
    engine.submit(request(231, std::vector<uint32_t>(promptTokens, 231)));
    runUntilIdle(engine);
    require(events.completedCount == 1 && events.emitted == 1 &&
                events.capacityExhaustedCount == 0 &&
                engine.snapshot().scheduler.decodeBatchesByWidth[0] == 1,
            "unused KV runway rejected a request whose full verify fits");
  }
}

void testAdmissionPinsDesiredStateAndCountsOnlySuccess() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> desired(65, 240);
  engine.submit(request(240, desired));
  runUntilIdle(engine);
  engine.submit(request(241, std::vector<uint32_t>(65, 241)));
  runUntilIdle(engine);
  const auto before = engine.snapshot();
  require(before.resources.stateCache.entries == 2,
          "pinning test did not publish both resident states");
  uint32_t pinnedAttempts = 0;
  executor.beginObserver = [&] {
    if (resources.snapshot().stateCache.pinned == 1) ++pinnedAttempts;
  };
  // The start lacks the buffers of a state the cache holds beside the one
  // it pins, which the ledger takes before it retries once.
  executor.statesLacked = [&] { return resources.snapshot().stateCache.entries > 1 ? 1U : 0U; };
  executor.beginGrowthBlocked = [&] { return executor.statesLacked() != 0; };
  const uint32_t attempts = executor.beginAttempts;
  engine.submit(request(242, desired));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(pinnedAttempts == 2 && executor.beginAttempts == attempts + 2 && events.completedCount == 3 &&
              after.resources.stateCache.entries == 1 &&
              after.resources.stateCache.pinned == 0 &&
              events.starts.back() == 64 &&
              after.cacheHits == before.cacheHits + 1 &&
              after.coldMisses == before.coldMisses,
          "successful retry failed to restore/count the protected cache state");
}

void testAdmissionCanDropItsOwnCachePinToMakeProgress() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 245);
  engine.submit(request(245, prompt));
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 1,
          "self-pin test did not retain its initial state");
  // Under the new ceiling, one active request fits only after releasing the
  // sole cached state, whose buffers its activation lacks. Its lookup lease
  // must not create a permanent deadlock.
  executor.statesLacked = [&] { return resources.snapshot().stateCache.entries != 0 ? 1U : 0U; };
  executor.beginGrowthBlocked = [&] { return executor.statesLacked() != 0; };
  engine.submit(request(246, prompt));
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0 && executor.restored == 0 &&
              executor.prefillRows == 130 &&
              events.starts.back() == 0 &&
              engine.snapshot().coldMisses == 2 && engine.snapshot().cacheHits == 0,
          "admission waited on its own cache pin instead of recomputing cold");
}

// Activation learns how much of the prompt a restore brings back; an attempt that gives up its prefix activates
// with nothing restored.
void testActivationReceivesTheRestoreBoundary() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 247);
  engine.submit(request(247, prompt));
  runUntilIdle(engine);
  require(executor.lastRestoredTokens == 0, "a cold start activated with a restored prefix");
  engine.submit(request(248, prompt));
  runUntilIdle(engine);
  require(executor.lastRestoredTokens == 64 && executor.restored == 64,
          "a prefix hit did not activate with its restore boundary");
  std::vector<uint32_t> attempts;
  executor.beginObserver = [&] { attempts.push_back(executor.lastRestoredTokens); };
  executor.statesLacked = [&] { return resources.snapshot().stateCache.entries != 0 ? 1U : 0U; };
  executor.beginGrowthBlocked = [&] { return executor.statesLacked() != 0; };
  engine.submit(request(249, prompt));
  runUntilIdle(engine);
  require(attempts.size() >= 2 && attempts.front() == 64 && attempts.back() == 0 &&
              events.starts.back() == 0,
          "the retry without its prefix activated with the dropped restore boundary");
}

// A start refused memory beside a resident lane waits with its prompt's state still cached, rather than release its
// lease to the reclaim.
void testRefusedStartKeepsItsLeaseBesideAResidentLane() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  const std::vector<uint32_t> prompt(65, 1);
  engine.submit(request(1, prompt));
  tickUntil(engine, now, [&] { return idle(engine); }, "the first request did not finish");
  executor.decodeFinishes = false;
  auto lane = request(2, std::vector<uint32_t>(65, 2));
  lane.maxNewTokens = 1000;
  engine.submit(std::move(lane));
  tickUntil(engine, now, [&] { return events.outputs[2].size() >= 2; },
            "the resident lane did not decode");
  // The budget refuses the start while the lane is resident.
  executor.beginGrowthBlocked = [&] {
    return executor.lastBeginId == 3 && executor.requests.contains(2);
  };
  const uint32_t prefillRows = executor.prefillRows;
  engine.submit(request(3, prompt));
  for (int step = 0; step < 5; ++step)
    static_cast<void>(engine.tick(now++));
  require(resources.probe(prompt, {}).cachedTokens() == 64 &&
              engine.resourceWaitSnapshot(now).memory == 1 &&
              executor.prefillRows == prefillRows,
          "a start held back by a resident lane gave up its own cached state");
  executor.decodeFinishes = true;
  const uint32_t restored = executor.restored;
  tickUntil(engine, now, [&] { return idle(engine); }, "the held start did not run");
  require(events.starts.back() == 64 &&
              executor.restored == restored + 64 && events.failedCount == 0,
          "the held start did not resume from its cached state");
}

// A start refused while a state write is in flight keeps its lease and waits for the write; then the state is
// written, leaves, and the start restores it.
void testRefusedStartKeepsItsLeaseWhileMemoryIsPending() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({}), resources, executor, events);
  guardReleases(storage, engine);
  host.bookStates(executor);
  executor.offloadTier = std::make_shared<OffloadControl>();
  executor.offloadTier->ready = true;
  executor.restoreControl->ready = true;
  auto write = std::make_shared<OffloadControl>();
  resources.beginRequest(999);
  require(resources.ensureTokens(999, 64).granted(), "fixture KV failed");
  resources.publishCompositeState(
      test::publishBlocks(resources, 999, std::vector<uint32_t>(64, 9), 64),
      std::make_shared<OffloadState>(write), false);
  resources.endRequest(999);
  require(test::stateStep(resources).gave() &&
              resources.snapshot().stateCache.bytes == 0,
          "the fixture state was not written");
  double now = 1;
  const std::vector<uint32_t> prompt(65, 1);
  engine.submit(request(1, prompt));
  tickUntil(engine, now, [&] { return idle(engine); }, "the first request did not finish");
  reachTheLimit(host);
  host.admitBegins(executor, State().bytes());
  engine.submit(request(2, prompt));
  for (int step = 0; step < 5; ++step)
    static_cast<void>(engine.tick(now++));
  require(resources.probe(prompt, {}).cachedTokens() == 64 &&
              engine.resourceWaitSnapshot(now).memory == 1 && events.startIds.size() == 1,
          "a start waiting for a write in flight gave up its own cached state");
  write->ready = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "the waiting start did not run");
  require(events.starts.back() == 64 &&
              events.failedCount == 0,
          "the waiting start did not resume from its cached state");
}

// A lone start with nothing else to free releases its lease, the state goes to disk, and the start restores that
// copy instead of recomputing.
void testDroppedLeaseLooksUpAgain() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 17);
  auto write = std::make_shared<OffloadControl>();
  write->ready = true;
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  cache.publishCompositeState(test::publishBlocks(cache, 999, prompt, 64),
                              std::make_shared<OffloadState>(write), false);
  cache.endRequest(999);
  // Its activation lacks the buffers of the state in RAM.
  executor.statesLacked = [&] { return cache.snapshot().stateCache.bytes != 0 ? 1U : 0U; };
  executor.beginGrowthBlocked = [&] { return executor.statesLacked() != 0; };
  executor.restoreControl->ready = true;
  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  require(executor.diskReads == 1 && executor.restored == 64 && executor.prefillRows == 1 &&
              events.starts.back() == 64 &&
              events.failedCount == 0,
          "a start that dropped its lease recomputed the state the reclaim wrote");
}

// A host refusal never makes a start drop its lease: alone it waits for the pressure to lift and then resumes from
// its cached state.
void testHostRefusedStartKeepsItsLease() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({}), cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 1);
  double now = 1;
  engine.submit(request(1, prompt));
  tickUntil(engine, now, [&] { return idle(engine); }, "the first request did not finish");
  host.pause(true);
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return host.paused(); };
  engine.submit(request(2, prompt));
  for (int step = 0; step < 5; ++step)
    static_cast<void>(engine.tick(now++));
  require(cache.probe(prompt, {}).cachedTokens() == 64 && events.startIds.size() == 1 &&
              events.failedCount == 0,
          "a start the host refused gave up its cached state");
  host.pause(false);
  tickUntil(engine, now, [&] { return idle(engine); }, "the refused start did not run");
  require(events.starts.back() == 64 && executor.prefillRows == 65 + 1,
          "the refused start did not resume from its cached state");
}

void testSingletonCapacityFailureTerminatesCleanly() {
  test::TestKvStorage storage(2, 4096, 1);
  storage.budgetPages = 1;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(request(40, std::vector<uint32_t>(33, 40)));
  runUntilIdle(engine);
  require(events.completedCount == 0 && events.failedCount == 1 &&
              events.capacityExhaustedCount == 1 && idle(engine),
          "B1 capacity failure did not emit exactly one terminal event");
  // It names the limits that would fit it: --max-memory only lowers the
  // budget, so a larger one only where one lowered it.
  require(events.failureDetails[0].first.ends_with(
              ": restart the server with a smaller --max-context, or with a larger --max-memory "
              "where one lowered it"),
          "the capacity failure did not name the limits that would fit the request");
  require(resources.snapshot().activeRequests == 0 && executor.requests.empty(),
          "B1 capacity failure leaked backend resources");
}

void testQueuedLongPrefillsLeaveRoomForShortWork() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(8193, id)));
  require(engine.tick(1) && engine.tick(2) &&
              executor.requests.size() == 1 && executor.prefillRows == 2048,
          "long prefills reserved lanes without executable work");
  engine.submit(request(5, std::vector<uint32_t>(65, 5)));
  require(engine.tick(3) && executor.requests.contains(5) &&
              executor.requests.at(5).position > 0 && events.completedCount == 0,
          "short arrival waited for a long prefill to finish");
  for (uint32_t now = 4; now < 200 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && events.completedCount == 5 &&
              events.failedCount == 0 && events.capacityExhaustedCount == 0 &&
              executor.prefillRows == 4 * 8193 + 65 && executor.suspensions == 0,
          "admission lost work, introduced replay, or stranded queued requests");
}

void testAdmissionUsesCachedRemainingWork() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> warm(4097, 47);
  engine.submit(request(1, warm));
  runUntilIdle(engine);
  engine.submit(request(2, std::vector<uint32_t>(8193, 48)));
  require(engine.tick(1) && engine.tick(2), "cold prefill did not start");
  engine.submit(request(3, warm));
  require(engine.tick(3) && executor.requests.contains(3) &&
              executor.restored == 4096 && executor.requests.at(3).position == 4097,
          "cached prompt was scheduled by total length instead of remaining work");
  runUntilIdle(engine);
  require(events.completedCount == 3 && events.failedCount == 0,
          "cache-aware admission failed to finish");
}

// A request that cannot be selected while a higher priority decodes is queued without hashing its prompt; once
// selectable, one probe finds its prefix.
void testUnselectableCandidatesAreNotProbed() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  const auto hashed = [&] { return resources.snapshot().lookup.probeHashedBlocks; };
  const std::vector<uint32_t> cached(129, 5);
  engine.submit(request(1, cached));
  runUntilIdle(engine);
  executor.decodeFinishes = false;
  auto urgent = request(2, std::vector<uint32_t>(33, 6));
  urgent.priority = RequestPriority::Foreground;
  urgent.maxNewTokens = 1000;
  engine.submit(std::move(urgent));
  double now = 100;
  tickUntil(engine, now, [&] { return events.outputs.contains(2); },
            "the higher priority did not decode");
  std::vector<uint32_t> prompt(cached.begin(), cached.begin() + 128);
  prompt.resize(161, 8);
  engine.submit(request(3, prompt));
  const uint64_t before = hashed();
  for (uint32_t step = 0; step < 5; ++step)
    static_cast<void>(engine.tick(now++));
  require(hashed() == before && !executor.requests.contains(3) &&
              engine.snapshot().scheduler.queued == 1,
          "admission probed a request it could not select");
  engine.cancel(2);
  tickUntil(engine, now, [&] { return executor.requests.contains(3); },
            "the waiting request did not start once the higher priority left");
  require(hashed() - before == 5 && events.startIds.back() == 3 &&
              events.starts.back() == 128,
          "the request was not probed once, or missed its cached prefix");
}

// A request waiting beside a resident prefill keeps its probe between passes and hashes only the page past its
// match again.
void testWaitingCandidateProbeIsRefreshedNotRepeated() {
  test::TestKvStorage storage(4096, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  const auto hashed = [&] { return resources.snapshot().lookup.probeHashedBlocks; };
  const std::vector<uint32_t> cached(257, 3);
  engine.submit(request(1, cached));
  runUntilIdle(engine);
  engine.submit(request(2, std::vector<uint32_t>(20'000, 2)));
  double now = 100;
  tickUntil(engine, now, [&] { return executor.requests.contains(2); },
            "the resident prefill did not start");
  std::vector<uint32_t> prompt(cached.begin(), cached.begin() + 256);
  prompt.resize(40'000, 4);
  engine.submit(request(3, prompt));
  const uint64_t before = hashed();
  tickUntil(engine, now, [&] { return hashed() != before; },
            "the waiting request was not probed");
  require(hashed() - before == 9, "the first probe did not hash its prefix and the page past it");
  const uint64_t probed = hashed();
  const size_t commands = executor.prefillWidths.size();
  for (uint32_t step = 0; step < 6; ++step)
    static_cast<void>(engine.tick(now++));
  const size_t passes = executor.prefillWidths.size() - commands;
  require(passes >= 2 && hashed() - probed == passes && !executor.requests.contains(3),
          "a kept probe hashed more than the page past its match");
}

// The host refuses the first start twice: as the governor's rule for the
// only request in service while the host is short of its margin, and once
// that claim passes the host, as the driver may (MemoryGovernor::
// allocationAdmission), which nothing the ledger frees ends. Later starts
// pass.
void refuseTheExemptRetry(Executor &executor, GovernedHost &host) {
  executor.beginGrowthBlocked = [&executor, &host] {
    if (executor.beginAttempts == 1) {
      executor.beginAllocationFailure = host.reserve(4096).failure;
      return executor.beginAllocationFailure != metal::AllocationFailure::None;
    }
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    return executor.beginAttempts == 2;
  };
}

// A request refused memory closes admission in the same pass: the request that arrived after it is not tried, and
// starts once the first has.
void testMemoryWaitHoldsBackLaterArrivals() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({}), resources, executor, events);
  guardReleases(storage, engine);
  host.pause(true);
  refuseTheExemptRetry(executor, host);
  engine.submit(request(1, std::vector<uint32_t>(4097, 47)));
  engine.submit(request(2, std::vector<uint32_t>(8193, 48)));
  static_cast<void>(engine.tick(1));
  require(executor.beginAttempts == 2 && executor.requests.empty() &&
              engine.resourceWaitSnapshot(1).memory == 1,
          "a later arrival was tried while an earlier request waits for memory");
  double now = 102;
  tickUntil(engine, now, [&] { return executor.requests.contains(2); },
            "the later arrival did not start after the request it waited behind");
  require(events.startIds == std::vector<uint64_t>{1, 2},
          "the requests did not start in the order they arrived");
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 2 && events.failedCount == 0,
          "a request held back behind a memory wait did not finish");
}

// A request refused memory beside a resident lane holds back only the starts it leads: a later, shorter arrival,
// which leads it, starts; a later, longer one the scheduler wants after it waits, untried, until it starts; a higher
// priority still starts.
void testARefusedStartHoldsOnlyWhatItLeads() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  bool refused = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return refused && executor.lastBeginId == 2; };
  std::vector<uint64_t> tried;
  executor.beginObserver = [&] { tried.push_back(executor.lastBeginId); };
  double now = 1;
  engine.submit(request(1, std::vector<uint32_t>(16385, 1)));
  require(engine.tick(now++) && executor.requests.contains(1), "the first request did not start");
  engine.submit(request(2, std::vector<uint32_t>(129, 2)));
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
            "the refused request is not waiting for memory");
  // Request 3 leads the refused one; request 4 is led by it, and the command
  // that ends its prompt would end request 4's too.
  engine.submit(request(3, std::vector<uint32_t>(65, 3)));
  engine.submit(request(4, std::vector<uint32_t>(257, 4)));
  for (uint32_t step = 0; step < 4; ++step)
    static_cast<void>(engine.tick(now++));
  require((executor.requests.contains(3) || events.usage.contains(3)) &&
              std::ranges::find(tried, 4) == tried.end() && engine.resourceWaitSnapshot(now).memory == 1 &&
              engine.resourceWaitSnapshot(now).heldBehindRefusal == 1 && events.failedCount == 0,
          "a refused request held back a later arrival that leads it, or let one it leads be tried");
  auto urgent = request(5, std::vector<uint32_t>(65, 5));
  urgent.priority = RequestPriority::Foreground;
  engine.submit(std::move(urgent));
  tickUntil(engine, now, [&] { return executor.requests.contains(5) || events.usage.contains(5); },
            "a refused request held back a higher priority");
  require(!executor.requests.contains(4) && !events.usage.contains(4),
          "a later arrival the refused request leads started ahead of it");
  refused = false;
  now += 100;
  tickUntil(engine, now, [&] { return executor.requests.contains(2); },
            "the waiting request did not start once its memory was there");
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  const auto started = [&](uint64_t id) {
    return std::find(events.startIds.begin(), events.startIds.end(), id) - events.startIds.begin();
  };
  require(events.completedCount == 5 && events.failedCount == 0 && started(2) < started(4) &&
              executor.prefillRows == 16385 + 129 + 65 + 257 + 65,
          "the held request lost work or started ahead of the refused one");
}

// Younger, shorter prompts that lead a refused start overtake it by at most the aging bound: once
// maximumBatchWidth - 1 commands in a row served them and not it, it leads them and holds them back until it starts.
void testAgingBoundsHowFarShorterArrivalsPassARefusedStart() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  bool refused = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return refused && executor.lastBeginId == 1; };
  constexpr uint32_t kRefusedRows = 129, kShortRows = 33;
  double now = 1;
  engine.submit(request(1, std::vector<uint32_t>(kRefusedRows, 1)));
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
            "the first request is not waiting for memory");
  // One shorter prompt at a time, the next once the last has ended.
  uint64_t latest = 100;
  engine.submit(request(latest, std::vector<uint32_t>(kShortRows, static_cast<uint32_t>(latest))));
  for (uint32_t step = 0; step < 400; ++step) {
    if (events.usage.contains(latest)) {
      ++latest;
      engine.submit(request(latest, std::vector<uint32_t>(kShortRows, static_cast<uint32_t>(latest))));
    }
    static_cast<void>(engine.tick(now++));
  }
  const auto overtaken = static_cast<uint32_t>(std::ranges::count_if(events.startIds, [](uint64_t id) { return id >= 100; }));
  const uint32_t bound = model::ExecutionLimits::maximumBatchWidth - 1;
  require(overtaken != 0 && overtaken <= bound && !executor.requests.contains(latest) &&
              engine.resourceWaitSnapshot(now).heldBehindRefusal == 1,
          "shorter prompts overtook the refused start " + std::to_string(overtaken) +
              " times, not within the aging bound of " + std::to_string(bound));
  refused = false;
  now += 100;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  const auto started = [&](uint64_t id) {
    return std::find(events.startIds.begin(), events.startIds.end(), id) - events.startIds.begin();
  };
  require(events.failedCount == 0 && started(1) < started(latest),
          "the refused start did not start ahead of the prompts it held back");
}

// The refused request is cancelled: the start it held back starts without it.
void testAHeldStartProceedsOnceTheRefusedOneIsCancelled() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return executor.lastBeginId == 2; };
  double now = 1;
  engine.submit(request(1, std::vector<uint32_t>(16385, 1)));
  require(engine.tick(now++) && executor.requests.contains(1), "the first request did not start");
  engine.submit(request(2, std::vector<uint32_t>(129, 2)));
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
            "the refused request is not waiting for memory");
  // Led by the refused request, which the command that ends its prompt would
  // join.
  engine.submit(request(3, std::vector<uint32_t>(257, 3)));
  for (uint32_t step = 0; step < 4; ++step)
    static_cast<void>(engine.tick(now++));
  require(!executor.requests.contains(3) && !events.usage.contains(3),
          "the refused request did not hold back a later arrival it leads");
  engine.cancel(2);
  tickUntil(engine, now, [&] { return executor.requests.contains(3) || events.usage.contains(3); },
            "the held request did not start once the refused one was cancelled");
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  // The cancelled request is reported as ended, without a row of its own.
  require(events.completedCount == 3 && events.failedCount == 0 &&
              engine.snapshot().cancelled == 1 && executor.prefillRows == 16385 + 257,
          "a cancelled wait left work behind");
}

// A refused request that then waits for a prefix holds back nothing: a later short request starts while it waits.
void testARefusedRequestWaitingForAPrefixHoldsBackNothing() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  // Two lanes: the first two requests take both, so the producer waits for
  // a lane while the host refuses request 4.
  Executor executor(2);
  bool hostRefuses = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return hostRefuses && executor.lastBeginId == 4; };
  // A prefill row takes 0.1 ms: a contended command holds a chunk's rows.
  executor.commandTimes = [](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    double rows = 0.0;
    for (const BatchItem &item : plan.items)
      rows += item.tokenCount;
    return test::CommandTimes{.wallMilliseconds = 0.1 * rows};
  };
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  // The producer and request 4 share 960 tokens.
  std::vector<uint32_t> producer(1000, 7);
  std::vector<uint32_t> refused(1000, 7);
  std::fill(producer.begin() + 960, producer.end(), 8);
  std::fill(refused.begin() + 960, refused.end(), 9);
  engine.submit(request(1, {1}));
  engine.submit(request(2, {2}));
  engine.submit(request(3, producer));
  engine.submit(request(4, refused));
  static_cast<void>(engine.tick(1));
  const auto waits = engine.resourceWaitSnapshot(1);
  require(executor.requests.size() == 2 && waits.memory == 1 && waits.concurrency == 1,
          "the producer did not wait for a lane beside a refused request");
  hostRefuses = false;
  require(engine.tick(2), "the first requests did not prefill");
  engine.cancel(2);
  // The producer starts in the freed lane with a junction for request 4,
  // which then waits for it; a decode runs before the producer's prefill.
  require(engine.tick(3) && executor.requests.contains(3) &&
              engine.snapshot().scheduler.waitingPrefix == 1,
          "the refused request did not wait for the producer's prefix");
  engine.submit(request(5, {5}));
  double now = 4;
  tickUntil(engine, now, [&] { return executor.requests.contains(5) || events.usage.contains(5); },
            "a later arrival waited behind a refused request that waits for a prefix");
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  const auto started = [&](uint64_t id) {
    return std::find(events.startIds.begin(), events.startIds.end(), id) - events.startIds.begin();
  };
  require(events.completedCount == 5 && events.failedCount == 0 && started(5) < started(4) &&
              executor.restored == 960,
          "the refused request did not start from the producer's junction after the later arrival");
}

// Suspended requests resume one at a time in admission order: a refused resume holds back later ones even if they
// would fit.
void testRefusedResumeHoldsBackLaterSuspendedLanes() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig({.resourceWaitTimeoutMilliseconds = 1000.0}),
                        resources, executor, events);
  guardReleases(storage, engine);
  // The resident lane fills the first extent and decodes within it.
  auto resident = request(1, std::vector<uint32_t>(97, 1));
  resident.maxNewTokens = 1000;
  engine.submit(std::move(resident));
  double now = 1;
  tickUntil(engine, now, [&] { return events.emitted != 0; }, "the resident lane did not decode");
  // The host refuses growth: both later requests are suspended at their
  // first prefill, request 2 to resume with five pages, request 3 with one.
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  const std::vector<uint32_t> first(161, 2);
  const std::vector<uint32_t> second(33, 3);
  engine.submit(request(2, first));
  engine.submit(request(3, second));
  tickUntil(engine, now, [&] { return executor.suspensions == 2; },
            "the later requests were not suspended");
  // One more extent: room for request 3's page, not for request 2's. The
  // refusal of request 2's resume drains the resident lane for the wait
  // limit; its retries after the drain are refused too.
  storage.growthBlocked = false;
  storage.allocationFailure = metal::AllocationFailure::EngineBudget;
  storage.budgetPages = 8;
  for (const double end = now + 1500; now < end; now += 101) {
    static_cast<void>(engine.tick(now));
    require(engine.snapshot().resourceResumptions == 0 && !executor.requests.at(3).resident &&
                events.failedCount == 0,
            "a suspended request resumed ahead of an earlier one refused memory");
  }
  require(executor.resumeAttempts >= 3 && executor.requests.at(1).resident,
          "the refused resume was not retried beside the resident lane");
  executor.decodeFinishes = true;
  for (const double end = now + 2000; now < end && !idle(engine); now += 101)
    static_cast<void>(engine.tick(now));
  // The executor records every state it grants, request 2's refused
  // resumes included: request 3's is the last.
  const auto &resumed = executor.resumedPrompts;
  require(idle(engine) && events.completedCount == 3 && events.failedCount == 0 &&
              engine.snapshot().resourceResumptions == 2 &&
              std::find(resumed.begin(), resumed.end(), second) == resumed.end() - 1,
          "the suspended requests did not resume in admission order");
}

// During recovery admission admits one request a pass: the starts above every suspended request, then the
// resumes.
void testRecoveryAdmitsOneRequestAPass() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig({.resourceWaitTimeoutMilliseconds = 1000.0}),
                        resources, executor, events);
  guardReleases(storage, engine);
  auto resident = request(1, std::vector<uint32_t>(97, 1));
  resident.maxNewTokens = 1000;
  engine.submit(std::move(resident));
  double now = 1;
  tickUntil(engine, now, [&] { return events.emitted != 0; }, "the resident lane did not decode");
  // The host refuses growth: both later requests are suspended at their
  // first prefill. Then it allows it, and two higher priorities arrive.
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  engine.submit(request(2, std::vector<uint32_t>(33, 2)));
  engine.submit(request(3, std::vector<uint32_t>(33, 3)));
  tickUntil(engine, now, [&] { return executor.suspensions == 2; },
            "the later requests were not suspended");
  storage.growthBlocked = false;
  for (const uint64_t id : {4, 5}) {
    EngineRequest value = request(id, {static_cast<uint32_t>(id)});
    value.priority = RequestPriority::Foreground;
    engine.submit(std::move(value));
  }
  const auto admitted = [&] { return events.startIds.size() + engine.snapshot().resourceResumptions; };
  std::vector<uint64_t> order;
  for (const double end = now + 5000; now < end && engine.snapshot().resourceResumptions < 2; now += 101) {
    const size_t before = admitted();
    static_cast<void>(engine.tick(now));
    require(admitted() <= before + 1, "recovery admitted more than one request in a pass");
  }
  require(engine.snapshot().resourceResumptions == 2 && events.failedCount == 0 &&
              std::ranges::find(events.startIds, 4) != events.startIds.end() &&
              std::ranges::find(events.startIds, 5) != events.startIds.end(),
          "recovery did not start the higher priorities and resume the suspended requests");
}

// A suspended request held behind a refused resume is not tried, so it keeps no retry time and does not run out its
// wait.
void testHeldSuspendedRequestNeitherWakesNorExpires() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  Events events;
  engine::Engine engine(test::engineConfig({.resourceWaitTimeoutMilliseconds = 300.0}),
                        resources, executor, events);
  guardReleases(storage, engine);
  // The host refuses growth, so both requests are suspended at their first
  // prefill, and then it refuses every resume.
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  engine.submit(request(1, {1}));
  engine.submit(request(2, {2}));
  require(engine.tick(1) && engine.tick(2) && executor.suspensions == 2 &&
              !engine.commandInFlight(),
          "the requests were not suspended");
  storage.growthBlocked = false;
  executor.resumeDenied = true;
  // As the native loop does, sleep until the next wake-up.
  double now = 2;
  while (!events.failedCount) {
    const std::optional<double> wakeup = engine.nextWakeupMilliseconds();
    require(wakeup && *wakeup > now,
            "a request held behind a refused resume woke the loop with nothing to do");
    now = *wakeup;
    static_cast<void>(engine.tick(now));
  }
  require(events.failures == std::vector<std::string>{"resource_timeout"},
          "the refused resume did not end at its wait limit alone");
  executor.resumeDenied = false;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 1 && events.failedCount == 1 &&
              engine.snapshot().resourceResumptions == 1,
          "a request held behind a refused resume ran out its wait meanwhile");
}

// /status counts the starts a refused one holds back while the scheduler wants it, never the refused one itself; its
// wait age runs from its first refusal across a pass that declines it.
void testStatusCountsRequestsHeldBehindARefusal() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  bool refused = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return refused && executor.lastBeginId == 2; };
  double now = 1;
  engine.submit(request(1, std::vector<uint32_t>(16385, 1)));
  require(engine.tick(now++) && executor.requests.contains(1), "the first request did not start");
  engine.submit(request(2, std::vector<uint32_t>(129, 2)));
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
            "the refused request is not waiting for memory");
  const double refusedBy = now;
  // Both led by the refused request, which the command that ends its prompt
  // would join.
  engine.submit(request(3, std::vector<uint32_t>(257, 3)));
  engine.submit(request(4, std::vector<uint32_t>(257, 4)));
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).heldBehindRefusal == 2; },
            "the requests behind a refusal were not counted");
  auto wait = engine.resourceWaitSnapshot(now);
  require(wait.memory == 1, "the refused request left its memory wait");
  // A higher priority arrives by the refused request's next retry and starts
  // in that pass, which defers the refused request for scheduling: it leaves
  // its memory wait and holds back nothing.
  auto urgent = request(5, std::vector<uint32_t>(65, 5));
  urgent.priority = RequestPriority::Foreground;
  engine.submit(std::move(urgent));
  now += 100;
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 0; },
            "the refused request was not deferred for scheduling");
  wait = engine.resourceWaitSnapshot(now);
  require(wait.heldBehindRefusal == 0 && !executor.requests.contains(3),
          "a refused request deferred for scheduling, or what it leads, was counted as held");
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
            "the refused request did not wait for memory again");
  wait = engine.resourceWaitSnapshot(now);
  require(wait.heldBehindRefusal == 2 && wait.oldestWaitMilliseconds >= now - refusedBy,
          "the refused request's wait age restarted when a pass deferred it");
  refused = false;
  now += 100;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  wait = engine.resourceWaitSnapshot(now);
  require(events.completedCount == 5 && events.failedCount == 0 && wait.heldBehindRefusal == 0,
          "the requests held behind a refusal did not finish");
}

// Requests held behind a refusal count only while it stands, not once the refused request is cancelled or started.
void testHeldRequestsCountWhileTheirRefusalStands() {
  for (const bool cancel : {true, false}) {
    test::TestKvStorage storage(2048, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);
    bool refused = true;
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    executor.beginGrowthBlocked = [&] { return refused && executor.lastBeginId == 2; };
    double now = 1;
    engine.submit(request(1, std::vector<uint32_t>(16385, 1)));
    require(engine.tick(now++) && executor.requests.contains(1), "the first request did not start");
    engine.submit(request(2, std::vector<uint32_t>(129, 2)));
    tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
              "the refused request is not waiting for memory");
    // Both led by the refused request, which the command that ends its
    // prompt would join.
    engine.submit(request(3, std::vector<uint32_t>(257, 3)));
    engine.submit(request(4, std::vector<uint32_t>(257, 4)));
    tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).heldBehindRefusal == 2; },
              "the requests behind a refusal were not counted");
    if (cancel) {
      engine.cancel(2);
      require(engine.resourceWaitSnapshot(now).heldBehindRefusal == 0,
              "requests still counted as held behind a cancelled refusal");
    } else {
      refused = false;
      now += 100;
      tickUntil(engine, now, [&] { return executor.requests.contains(2); },
                "the refused request did not start");
      require(engine.resourceWaitSnapshot(now).heldBehindRefusal == 0,
              "requests still counted as held behind a refused request that started");
    }
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
    require(events.failedCount == 0, "a request held behind a refusal failed");
  }
}

// During recovery /status counts the suspended requests held behind the first refused resume, not those behind a
// refused start.
void testStatusCountsSuspendedRequestsHeldDuringRecovery() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  bool hostRefuses = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return hostRefuses && executor.lastBeginId == 2; };
  Events events;
  engine::Engine engine(test::engineConfig({.resourceWaitTimeoutMilliseconds = 1000.0}),
                        resources, executor, events);
  guardReleases(storage, engine);
  // The resident lane fills the first extent and decodes within it.
  auto resident = request(1, std::vector<uint32_t>(97, 1));
  resident.maxNewTokens = 1000;
  engine.submit(std::move(resident));
  double now = 1;
  tickUntil(engine, now, [&] { return events.emitted != 0; }, "the resident lane did not decode");
  // The host refuses request 2's state and all growth. Requests 3 and 4 start
  // in the pass that refuses request 2 and are suspended at their first
  // prefill. Request 3's resume is refused too, and request 4 is held behind
  // it without an attempt.
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  engine.submit(request(2, std::vector<uint32_t>(4097, 2)));
  engine.submit(request(3, std::vector<uint32_t>(33, 3)));
  engine.submit(request(4, std::vector<uint32_t>(33, 4)));
  tickUntil(engine, now, [&] { return executor.suspensions == 2; },
            "the later requests were not suspended");
  for (const double end = now + 1500;
       now < end && !engine.resourceWaitSnapshot(now).heldBehindRefusal; now += 101)
    static_cast<void>(engine.tick(now));
  const auto wait = engine.resourceWaitSnapshot(now);
  require(wait.suspended == 2 && wait.heldBehindRefusal == 1,
          "recovery counted requests it does not hold back behind a refused resume");
  hostRefuses = false;
  storage.growthBlocked = false;
  executor.decodeFinishes = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 4 && events.failedCount == 0,
          "the requests did not finish once memory was there");
}

void testSchedulingWaitDoesNotConsumeMemoryTimeout() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({.resourceWaitTimeoutMilliseconds = 1000.0}), resources, executor, events);
  guardReleases(storage, engine);
  host.pause(true);
  refuseTheExemptRetry(executor, host);
  engine.submit(request(1, std::vector<uint32_t>(8193, 47)));
  static_cast<void>(engine.tick(1));
  require(executor.beginAttempts == 2 && !executor.requests.contains(1) &&
              engine.resourceWaitSnapshot(1).memory == 1,
          "the first request did not wait for memory");
  auto urgent = request(2, std::vector<uint32_t>(4097, 48));
  urgent.priority = RequestPriority::Foreground;
  engine.submit(std::move(urgent));
  // Prefill admission takes only the top priority tier: the first request
  // waits for scheduling alone.
  require(engine.tick(102) && engine.resourceWaitSnapshot(102).memory == 0 &&
              executor.requests.contains(2) && !executor.requests.contains(1),
          "scheduler delay retained a stale memory wait");
  for (uint32_t now = 1100; now < 1200 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && events.completedCount == 2 && events.failedCount == 0,
          "scheduling delay triggered the memory wait timeout");
}

void testUnadmittedRequestsHonorCancellationAndDeadline() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(8193, 1)));
  engine.submit(request(2, std::vector<uint32_t>(8193, 2)));
  auto expiring = request(3, std::vector<uint32_t>(8193, 3));
  expiring.deadlineMilliseconds = 3;
  engine.submit(std::move(expiring));
  require(engine.tick(1) && engine.tick(2), "long prefill did not start");
  engine.cancel(2);
  static_cast<void>(engine.tick(3));
  runUntilIdle(engine);
  require(engine.snapshot().cancelled == 1 && events.failedCount == 1 &&
              events.completedCount == 2 && executor.beginAttempts == 1,
          "queued cancellation or deadline allocated resources or failed cleanup");
  require(events.failures == std::vector<std::string>{"deadline_exceeded"} &&
              events.failureDetails.front().first == "request deadline exceeded",
          "a queued request's deadline lost its outcome or message");
}

void testGrowthKeepsPrefillProgressWhenAnUnstartedPeerCanYield() {
  for (RequestPriority priority : {RequestPriority::Normal,
                                   RequestPriority::Background}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(2);
    Events events;
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);
    storage.growthAllowed = [&](uint32_t) {
      const auto started = executor.requests.find(48);
      return started == executor.requests.end() ||
             started->second.position < 2048 ||
             std::count_if(executor.requests.begin(), executor.requests.end(),
                           [](const auto &entry) { return entry.second.resident; }) < 2;
    };
    // Its final command leaves rows of the budget to a peer whose last chunk
    // of 96 rows they spare a command (Scheduler: a lane joins with those).
    engine.submit(request(48, std::vector<uint32_t>(4000, 48)));
    auto peer = request(49, std::vector<uint32_t>(8192 + 96, 49));
    peer.priority = priority;
    engine.submit(std::move(peer));
    require(engine.tick(1) && engine.tick(2) &&
                executor.requests.at(48).position == 2048 &&
                !executor.requests.contains(49),
            "unstarted prefill reserved a lane before it had scheduled work");
    require(engine.tick(3) && executor.requests.at(48).resident,
            "KV growth discarded completed prefill");
    if (priority == RequestPriority::Normal) {
      // The final prefill slice admits a peer into its remaining row budget;
      // if that lane prevents KV growth, the unstarted peer must yield.
      require(executor.suspensions == 1 && !executor.requests.at(49).resident,
              "KV growth did not yield the unstarted peer");
    } else {
      require(executor.suspensions == 0 && !executor.requests.contains(49),
              "lower-priority work reserved a lane before its dispatch");
    }
    runUntilIdle(engine);
    require(events.completedCount == 2 && events.failedCount == 0 &&
                events.capacityExhaustedCount == 0 && executor.prefillRows == 4000 + 8192 + 96,
            "work-aware admission lost prefill work or failed to complete");
  }
}

void testGrowthYieldsLowerPriorityResidentOutsideBatch() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  storage.growthAllowed = [&](uint32_t) {
    return std::count_if(executor.requests.begin(), executor.requests.end(),
                         [](const auto &entry) { return entry.second.resident; }) < 2;
  };
  engine.submit(request(48, std::vector<uint32_t>(4096, 48)));
  require(engine.tick(1) && engine.tick(2) &&
              executor.requests.at(48).position == 2048,
          "priority fixture did not advance the initial request");
  auto foreground = request(49, std::vector<uint32_t>(8192, 49));
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  require(engine.tick(3) && executor.suspensions == 1 &&
              !executor.requests.at(48).resident &&
              executor.requests.at(49).resident,
          "resource recovery preempted foreground work for a lower-priority peer");
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "priority preemption failed to finish both requests");
}

void testPrefillGrowthPreservesAnActiveDecodePeer() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(48, std::vector<uint32_t>(8192, 48)));
  require(engine.tick(1) && engine.tick(2), "prefill setup did not progress");
  auto decoding = request(49, {49});
  decoding.maxNewTokens = 32;
  engine.submit(std::move(decoding));
  require(engine.tick(3) && engine.tick(4) &&
              executor.requests.at(49).position == 1,
          "short peer did not enter decode");
  storage.growthBlocked = true;
  for (uint32_t step = 5; step < 15 && !executor.suspensions; ++step)
    static_cast<void>(engine.tick(step));
  require(executor.suspensions == 1 && !executor.requests.at(48).resident &&
              executor.requests.at(49).resident,
          "prefill growth interrupted an equal-priority decode stream");
  storage.growthBlocked = false;
  executor.decodeFinishes = true;
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "mixed-phase pressure failed to finish both requests");
}

// A decoding lane refused a page is left out of a command that still runs; the prompt's lane outside it yields only
// once its own command is refused.
void testARefusedDecodeLeavesThePromptOutsideItsCommand() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  // Each command's kind and lanes.
  std::vector<std::pair<WorkKind, std::vector<uint64_t>>> commands;
  executor.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
    std::vector<uint64_t> lanes;
    for (const ModelBatchItem &item : items)
      lanes.push_back(item.requestId);
    commands.emplace_back(plan.kind, std::move(lanes));
    return test::CommandTimes{};
  };
  // Lane 1's decode needs a fourth page two tokens in, lane 2's 24 tokens in.
  const auto promptTokens = [](uint64_t id) { return id == 1 ? 87U : 65U; };
  for (const uint32_t id : {1U, 2U}) {
    auto decoding = request(id, std::vector<uint32_t>(promptTokens(id), id));
    decoding.maxNewTokens = 1000;
    engine.submit(std::move(decoding));
  }
  double now = 1;
  tickUntil(engine, now, [&] { return events.outputs.contains(1) && events.outputs.contains(2); },
            "the lanes did not decode");
  // A long prompt's commands run between theirs.
  engine.submit(request(3, std::vector<uint32_t>(8 * model::ExecutionLimits::prefillTokenBudget, 3)));
  const auto needsPage = [&](uint64_t id) {
    return promptTokens(id) + events.outputs[id].size() + model::ExecutionLimits::targetVerifyRows >
           uint64_t{resources.pageTable(id).pages.size()} * KvCache::pageTokens;
  };
  tickUntil(
      engine, now,
      [&] {
        return !engine.commandInFlight() && !commands.empty() &&
               commands.back().first == WorkKind::Prefill && needsPage(1) && !needsPage(2);
      },
      "lane 1 did not come to need a page that lane 2 does not");
  // At the wall, with nothing in flight: no free page, and no growth.
  storage.growthBlocked = true;
  const KvPageAcquisition held = pool.acquirePages(pool.freePageCount());
  static_cast<void>(engine.tick(now++));
  require(held.granted() && engine.commandInFlight() &&
              commands.back() == std::pair{WorkKind::Decode, std::vector<uint64_t>{2}} &&
              executor.suspensions == 0 && executor.requests.at(3).resident,
          "a lane refused its KV beside a lane that fit suspended the prompt outside their command");
  // The prompt's own command, refused next, suspends its lane.
  const size_t submitted = commands.size();
  tickUntil(engine, now, [&] { return executor.suspensions != 0 || commands.size() != submitted; },
            "nothing ran or yielded after the decode");
  require(executor.suspensions == 1 && !executor.requests.at(3).resident &&
              commands.size() == submitted,
          "the prompt's lane did not yield once its own command was refused");
  for (const uint32_t page : held.pages)
    pool.releasePage(page, false);
  storage.growthBlocked = false;
  executor.decodeFinishes = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "the lanes did not finish");
  require(events.completedCount == 3 && events.failedCount == 0, "the lanes did not all complete");
}

void testKvPressureSuspendsInsteadOfKillingActiveWork() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.kvGrowthBlocked = &storage.growthBlocked;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);

  storage.growthBlocked = true;
  engine.submit(request(50, {50}));
  engine.submit(request(51, {51}));
  runUntilIdle(engine);

  require(executor.suspensions == 1 && executor.resumptions == 1 &&
              engine.snapshot().resourceSuspensions == 1 &&
              engine.snapshot().resourceResumptions == 1,
          "KV pressure did not suspend and resume one lane");
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "recoverable pressure killed an active request");
}

void testPressureRetryIsBackedOffWithoutProgress() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  Events events;
  GovernedHost host;
  host.setPressure(MemoryPressure::Warning);
  EngineConfig config = host.connect({});
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);

  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  engine.submit(request(52, {52}));
  engine.submit(request(53, {53}));
  require(engine.tick(1.0) && engine.tick(2.0),
          "pressure did not suspend both blocked lanes");
  const uint32_t attemptsBeforeBackoff = executor.resumeAttempts;
  require(executor.suspensions == 2 && attemptsBeforeBackoff == 0,
          "suspended lanes were retried before a resource wake-up");
  require(!engine.tick(50.0) &&
              executor.resumeAttempts == attemptsBeforeBackoff,
          "resource wait retried on an unrelated scheduler tick");

  storage.growthBlocked = false;
  host.setPressure(MemoryPressure::Normal);
  require(engine.tick(102.0), "resource retry timer did not wake the engine");
  for (double now = 103.0; now < 120.0 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.resumptions == 2 &&
              events.completedCount == 2 && events.failedCount == 0,
          "backed-off resource requests did not recover cleanly");
}

// A waiting request's retry time wakes the native loop only when tick() can retry it; otherwise the loop polls
// without blocking.
void testAdmissionRetryWakesOnlyWhenTickCanRetry() {
  {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    executor.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
    Events events;
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);
    engine.submit(request(1, {1}));
    require(engine.tick(1) && engine.tick(2), "resident request did not prefill");
    engine.submit(request(2, {2}));
    require(engine.tick(3) && engine.commandInFlight() &&
                engine.resourceWaitSnapshot(3).concurrency == 1,
            "second request did not wait for the resident lane's command");
    require(!engine.tick(150) &&
                engine.nextWakeupMilliseconds() == 1150.0,
            "a lane wait asked for a wake-up while a command was in flight");
    *executor.holdDecodeUntil = true;
    for (double now = 151; now < 170 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    require(idle(engine) && events.completedCount == 2,
            "lane wait did not complete after the command");
  }
  {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    GovernedHost host;
    host.pause(true);
    EngineConfig config = host.connect({});
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    storage.growthBlocked = true;
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    engine.submit(request(1, {1}));
    engine.submit(request(2, {2}));
    require(engine.tick(1) && executor.suspensions == 1 &&
                engine.resourceWaitSnapshot(1).concurrency == 1,
            "fixture did not suspend one request behind a lane wait");
    require(!engine.tick(101) && executor.resumeAttempts == 1 &&
                engine.nextWakeupMilliseconds() == 201.0,
            "a lane wait asked for a wake-up only the suspended request gets");
    host.pause(false);
    storage.growthBlocked = false;
    for (double now = 201; now < 220 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    require(idle(engine) && events.completedCount == 2,
            "lane wait did not complete after the suspended request");
  }
}

void testRecoveryDrainDoesNotConsumeResourceWaitBudget() {
  for (bool expireRequest : {false, true}) {
    test::TestKvStorage storage(3, 4096, 1);
    storage.budgetPages = 2;
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(2);
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);
    // The resident's held command is its last one.
    for (uint64_t id : {250, 251}) {
      auto value = request(id, std::vector<uint32_t>(24, id));
      value.maxNewTokens = 2;
      value.deadlineMilliseconds = expireRequest ? 30000 : 1000000;
      engine.submit(std::move(value));
    }
    for (double now = 1; now <= 5; ++now)
      require(engine.tick(now), "drain fixture made no progress");
    require(executor.suspensions == 1, "drain fixture did not preempt a lane");
    executor.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
    require(engine.tick(6) && engine.commandInFlight(),
            "resident peer did not start its command");
    static_cast<void>(engine.tick(20006));
    require(engine.resourceWaitSnapshot(20006).draining,
            "recovery did not drain behind the resident peer");
    // The drain ends at the resource wait limit, and its time did not count
    // against the suspended request's own resource wait.
    static_cast<void>(engine.tick(30006));
    require(executor.resumeAttempts == 0,
            "recovery retried while a resident peer was still running");
    if (expireRequest) {
      require(!events.failures.empty() &&
                  std::all_of(events.failures.begin(), events.failures.end(),
                              [](const auto &code) { return code == "deadline_exceeded"; }),
              "draining suppressed the request deadline");
    } else {
      require(events.failedCount == 0 && engine.nextWakeupMilliseconds() > 30006,
              "deliberate drain consumed the resource wait limit");
      const auto wait = engine.resourceWaitSnapshot(30006);
      require(!wait.draining && wait.suspended == 1 && wait.memory == 1 &&
                  wait.oldestWaitMilliseconds >= 30000,
              "drain outlived its limit or hid the elapsed resource wait");
    }
    *executor.holdDecodeUntil = true;
    for (double now = 30007; now < 30100 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    require(idle(engine) && resources.snapshot().activeRequests == 0,
            "drain fixture did not release its resources");
    if (!expireRequest)
      require(events.completedCount == 2 && events.failedCount == 0 &&
                  executor.resumptions == 1,
              "suspended work did not resume after its peer completed");
  }
}

void testDecodePreemptionReplaysCommittedHistoryWithoutRepeatingOutput() {
  test::TestKvStorage storage(3, 4096, 1);
  storage.budgetPages = 2;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {250, 251}) {
    auto value = request(id, std::vector<uint32_t>(24, id));
    value.maxNewTokens = 4;
    value.returnProgress = true;
    engine.submit(std::move(value));
  }
  for (double now = 1; now <= 4; ++now)
    require(engine.tick(now), "decode replay setup made no progress");
  require(events.emitted == 2 && engine.tick(5) && executor.suspensions == 1,
          "full active KV did not preempt a lane at the verify boundary");
  uint64_t preempted = 0;
  for (const auto &[id, state] : executor.requests) {
    if (!state.resident) preempted = id;
  }
  require(preempted && resources.snapshot().activeRequests == 1 &&
              resources.snapshot().pool.pagesActive == 1,
          "preempted decode retained its active KV ownership");

  // New admission and the elapsed retry timer must not disturb the resident
  // lane while it uses the released headroom to finish.
  executor.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
  engine.submit(request(252, {252}));
  require(engine.tick(6) && engine.commandInFlight(),
          "surviving lane could not launch after preemption");
  require(!engine.tick(150) && engine.nextWakeupMilliseconds() == 1150.0 &&
              events.startIds.size() == 2 && executor.resumeAttempts == 0,
          "draining recovery busy-woke or admitted a competing request");
  *executor.holdDecodeUntil = true;
  for (double now = 151; now < 210 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.resumptions == 2 &&
              events.completedCount == 3 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "decode preemption did not finish all original/new requests");
  require(events.progress.at(250) == std::vector<uint32_t>({0, 24}) &&
              events.progress.at(251) == std::vector<uint32_t>({0, 24}),
          "decode recovery reported generated history as prompt progress");
  std::vector<uint32_t> committedHistory(24, preempted);
  committedHistory.push_back(42);
  // The newly admitted request also yields once because the recovering
  // continuation has reserved both pages. Neither replays generated output.
  require(executor.resumedPrompts ==
              std::vector<std::vector<uint32_t>>{committedHistory, {252}} &&
              engine.snapshot().resourceReplayTokens == committedHistory.size() + 1,
          "replay omitted or duplicated committed generated history");
  require(events.outputs.at(250) == std::vector<uint32_t>(4, 42) &&
              events.outputs.at(251) == std::vector<uint32_t>(4, 42) &&
              events.usage.at(250) == std::pair<uint32_t, uint32_t>{24, 4} &&
              events.usage.at(251) == std::pair<uint32_t, uint32_t>{24, 4} &&
              events.startIds.size() == 3 &&
              events.maskRequests.empty() &&
              engine.snapshot().cacheHits == 0 && engine.snapshot().coldMisses == 3,
          "resource replay emitted prior output or changed request usage/hits");
  require(executor.requests.empty() && executor.reservationAttempts == 0 &&
              resources.snapshot().pool.pagesActive == 0 &&
              resources.snapshot().activeRequests == 0,
          "decode replay leaked active resources");
}

void testLongDecodePreemptionPlansTheCurrentReplayBoundary() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.decodeFinishes = false;
  executor.deniedReservations = std::numeric_limits<uint32_t>::max();
  Events events;
  GovernedHost host;
  EngineConfig config = test::engineConfig();
  // Keep the old 4096-token replay boundary distinct from checkpoints.
  config.prefillCheckpointTokens = 8192;
  config = host.connect(std::move(config));
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);

  constexpr uint64_t id = 255;
  // Each decode step drafts one token and accepts none: one token a step,
  // as without a draft.
  executor.speculativeCounts = {{id, {1, 0}}};
  const std::vector<uint32_t> prompt(4097, 5);
  auto value = request(id, prompt);
  value.maxNewTokens = 10'000;
  value.deadlineMilliseconds = 100'000;
  engine.submit(std::move(value));
  double now = 1;
  for (; now < 20'000 && executor.suspensions == 0; ++now) {
    if (events.emitted >= 4200) {
      storage.growthBlocked = true;
      storage.allocationFailure = metal::AllocationFailure::HostPressure;
      host.setPressure(MemoryPressure::Warning);
    }
    static_cast<void>(engine.tick(now));
  }
  require(executor.suspensions == 1 && !engine.commandInFlight(),
          "long decode did not suspend at KV growth pressure");
  const uint32_t draftedBeforeSuspension = events.draftedTokens;
  const std::vector<uint32_t> emitted = events.outputs.at(id);
  std::vector<uint32_t> history = prompt;
  history.insert(history.end(), emitted.begin(), emitted.end());
  require(resources.snapshot().stateCache.entries == 0,
          "refused buffers left a composite state to restore");

  // Retain the lane's KV one draft window past the prompt's replay boundary,
  // whose state is gone. That KV ends there, so it is no junction: the
  // resumed lane rebuilds the state where the conversation's next turn
  // resumes, then captures the generated history's end.
  constexpr uint32_t retainedKvTokens = 6144;
  while (resources.snapshot().pool.pagesPrefix >
         retainedKvTokens / KvCache::pageTokens) {
    require(test::evictStep(resources).gave(),
            "could not reclaim the suspended decode's KV tail");
  }
  {
    auto lookup = resources.lookup(history, {});
    require(lookup.kvBoundary == retainedKvTokens &&
                lookup.resumeBoundary() == 0,
            "long replay did not retain the intended stateless KV");
  }

  storage.growthBlocked = false;
  host.setPressure(MemoryPressure::Normal);
  executor.deniedReservations = 0;
  executor.decodeFinishes = true;
  now += 101;
  require(engine.tick(now++), "long replay did not resume after pressure eased");
  const auto &plan = executor.plans.at(id);
  require(plan.replayEnd == history.size() && plan.captureSpans.size() == 2 &&
              plan.boundaries.size() == 3 && plan.boundaries.front().boundary == 4096,
          "resumed draft plan did not rebuild the prompt's replay point first");
  const double finishBy = now + 100;
  for (; now < finishBy && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.resumptions == 1 &&
              events.completedCount == 1 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "long decode did not complete after partial KV reclamation");
  require(executor.resumedPrompts ==
              std::vector<std::vector<uint32_t>>{history} &&
              engine.snapshot().resourceReplayTokens == history.size(),
          "long replay omitted or duplicated committed history");
  auto expectedOutput = emitted;
  expectedOutput.push_back(42);
  require(events.outputs.at(id) == expectedOutput &&
              events.usage.at(id) == std::pair<uint32_t, uint32_t>{
                                         prompt.size(), expectedOutput.size()} &&
              events.startIds == std::vector<uint64_t>{id},
          "long replay repeated output or changed original request usage");
  // The request reports the decode steps before its suspension and after,
  // as /status counts them; the replay drafts nothing.
  require(draftedBeforeSuspension > 0 &&
              events.draftedTokens > draftedBeforeSuspension &&
              events.speculativeCounts.at(id) ==
                  SpeculativeCounts{events.draftedTokens, 0, events.draftedTokens},
          "suspension lost the request's speculative counts");
  require(executor.requests.empty() &&
              resources.snapshot().pool.pagesActive == 0 &&
              resources.snapshot().activeRequests == 0,
          "long replay leaked active resources");
  // The conversation's next turn resumes from the rebuilt point.
  std::vector<uint32_t> next = prompt;
  next.resize(next.size() + 40, 9);
  require(resources.lookup(next, {}).resumeBoundary() == 4096 &&
              resources.snapshot().stateCache.inUse == 0,
          "the resumed lane did not rebuild the prompt's replay point");
}

// A resumed producer plans the junction at a sibling's shared boundary, so the sibling resumes there instead of
// prefilling the shared prefix.
void testResumedProducerPlansJunctionsForSiblings() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  GovernedHost host;
  EngineConfig config = host.connect({});
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(20001);
  std::iota(prompt.begin(), prompt.end(), 1);
  engine.submit(request(1, prompt));
  double now = 1;
  tickUntil(engine, now,
            [&] { return !engine.commandInFlight() && executor.prefillRows >= 10'000; },
            "the producer did not pass its checkpoint at 8192");
  // The host refuses the producer's next pages: it yields its lane.
  host.pause(true);
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  tickUntil(engine, now, [&] { return executor.suspensions == 1; },
            "the producer was not suspended");
  std::vector<uint32_t> sibling(prompt.begin(), prompt.begin() + 18'000);
  sibling.resize(18'065, 0);
  engine.submit(request(2, sibling));
  const uint32_t rows = executor.prefillRows;
  host.pause(false);
  storage.growthBlocked = false;
  now += 101;
  tickUntil(engine, now, [&] { return engine.snapshot().scheduler.waitingPrefix == 1; },
            "the sibling did not wait for the resumed producer");
  const auto &boundaries = executor.plans.at(1).boundaries;
  require(executor.resumptions == 1 && executor.restored == 8192 &&
              std::any_of(boundaries.begin(), boundaries.end(),
                          [](const DraftBoundaryPlan &boundary) {
                            return boundary.boundary == 17'984;
                          }),
          "the resumed producer planned no junction at the shared boundary");
  tickUntil(engine, now, [&] { return idle(engine); }, "the requests did not finish");
  require(events.starts.back() == 17'984 &&
              executor.prefillRows - rows == (prompt.size() - 8192) + (sibling.size() - 17'984) &&
              events.completedCount == 2 && events.failedCount == 0,
          "the sibling did not resume from the resumed producer's junction");
}

// The field case of a lane suspended mid-decode: it resumes with its
// prompt's replay point lost, and the only memory left is another
// conversation's cached KV, two whole extents with no state to recycle.
struct LostReplayPoint {
  LostReplayPoint() {
    executor.decodeFinishes = false;
    executor.deniedReservations = std::numeric_limits<uint32_t>::max();
    guardReleases(storage, engine);
    auto value = request(id, prompt);
    value.maxNewTokens = 10'000;
    value.deadlineMilliseconds = 100'000;
    engine.submit(std::move(value));
    for (; now < 20'000 && executor.suspensions == 0; ++now) {
      if (events.emitted >= 200) {
        storage.growthBlocked = true;
        storage.allocationFailure = metal::AllocationFailure::HostPressure;
        host.setPressure(MemoryPressure::Warning);
      }
      static_cast<void>(engine.tick(now));
    }
    require(executor.suspensions == 1 && !engine.commandInFlight() &&
                resources.snapshot().stateCache.entries == 0,
            "the decode was not suspended without its point");
    // Its KV goes too, and so do the extents it held.
    while (test::releaseStep(resources).gave()) {
    }
    require(resources.snapshot().pool.pagesPrefix == 0 && pool.snapshot().pagesAllocated == 0,
            "the suspended lane's KV stayed");
    storage.growthBlocked = false;
    host.setPressure(MemoryPressure::Normal);
    // Another conversation's KV, with no state, fills two whole extents.
    resources.beginRequest(900);
    require(resources.ensureTokens(900, 256).granted(), "the other conversation got no pages");
    resources.publishCommittedBlocks(900, other, 256, {});
    resources.endRequest(900);
    releases = storage.releasedExtents;
  }


  // Resumes the lane, whose replay point is planned again, and runs it to
  // its end. A state's buffers need memory that only released extents give,
  // and an extent goes only while no command is in flight (guardReleases).
  void finish() {
    executor.deniedReservations = 0;
    executor.decodeFinishes = true;
    now += 101;
    for (const double end = now + 1000; now < end && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
  }

  static constexpr uint64_t id = 256;
  const std::vector<uint32_t> prompt = std::vector<uint32_t>(4097, 5);
  const std::vector<uint32_t> other = std::vector<uint32_t>(257, 6);
  test::TestKvStorage storage{1024, 4096, 4};
  KvPool pool{storage, 0};
  engine::Cache resources{pool, nullptr, nullptr};
  Executor executor{1};
  Events events;
  GovernedHost host;
  engine::Engine engine{host.connect({.prefillCheckpointTokens = 8192}), resources, executor, events};
  double now = 1;
  // The extents released before the lane resumes.
  uint32_t releases = 0;
};

// A resumed lane's rebuilt point takes another conversation's KV oldest first until an extent empties, a second one
// if a state needs it.
void testResumedLaneRebuildsItsPointFromCachedKv(uint32_t extentsPerState) {
  LostReplayPoint fixture;
  test::TestKvStorage &storage = fixture.storage;
  engine::Engine &engine = fixture.engine;
  engine::Cache &resources = fixture.resources;
  const uint32_t releases = fixture.releases;
  const uint64_t stateBytes = extentsPerState * 4 * 4096;
  reachTheLimit(fixture.host);
  fixture.executor.reservationRoom = [&] {
    if (storage.releasedExtents >= releases + extentsPerState)
      return true;
    fixture.executor.reservationFailure = refusedAtTheLimit(fixture.host, stateBytes);
    return false;
  };
  fixture.finish();
  const Events &events = fixture.events;
  const Executor &executor = fixture.executor;
  const auto after = engine.snapshot();
  require(idle(engine) && executor.resumptions == 1 && events.completedCount == 1 &&
              events.failedCount == 0 && after.recycledStatePublications == 1 &&
              after.resources.stateCache.inUseEvictions == 0 &&
              after.resources.stateCache.inUse == 0,
          "the resumed lane did not publish its rebuilt point");
  // For one extent the older KV paid one page: the lane's last page, alone
  // in its extent, moved there and its extent went. A second extent costs
  // the older KV four pages more. The next turn resumes from the point.
  std::vector<uint32_t> next = fixture.prompt;
  next.resize(next.size() + 40, 9);
  require(resources.lookup(fixture.other, {}).kvBoundary == (extentsPerState == 1 ? 224u : 96u) &&
              storage.copies.size() == extentsPerState &&
              storage.releasedExtents == releases + extentsPerState &&
              resources.lookup(next, {}).resumeBoundary() == 4096,
          "the rebuilt point did not come from the older KV");
}

// Buffers that extents cannot fit stop taking extents once one state's worth is released; neither point is
// published.
void testRefusedBuffersTakeAtMostOneStateOfExtents() {
  LostReplayPoint fixture;
  const uint64_t stateBytes = 4 * 4096;
  reachTheLimit(fixture.host);
  fixture.executor.reservationRoom = [&] {
    fixture.executor.reservationFailure = refusedAtTheLimit(fixture.host, stateBytes);
    return false;
  };
  fixture.finish();
  std::vector<uint32_t> next = fixture.prompt;
  next.resize(next.size() + 40, 9);
  require(idle(fixture.engine) && fixture.events.completedCount == 1 &&
              fixture.storage.releasedExtents == fixture.releases + 1 &&
              fixture.resources.lookup(fixture.other, {}).kvBoundary == 256 - 32 &&
              fixture.resources.probe(next, {}).cachedTokens() == 0,
          "refused buffers took more than one state's worth of extents");
}

// A start refused its state's extent takes another conversation's leaves until an extent has left memory, and asks
// again once then, not after each leaf.
void testRefusedStateAsksAgainOnceAnExtentLeft() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({}), resources, executor, events);
  guardReleases(storage, engine);
  // Another conversation's KV, with no state, fills two whole extents.
  const std::vector<uint32_t> other(257, 6);
  resources.beginRequest(900);
  require(resources.ensureTokens(900, 256).granted(), "the other conversation got no pages");
  resources.publishCommittedBlocks(900, other, 256, {});
  resources.endRequest(900);
  // The start's state fits only once an extent has left memory.
  const uint32_t releases = storage.releasedExtents;
  reachTheLimit(host);
  executor.beginGrowthBlocked = [&] {
    if (storage.releasedExtents != releases)
      return false;
    executor.beginAllocationFailure = refusedAtTheLimit(host, 4 * 4096);
    return true;
  };
  const uint32_t attempts = executor.beginAttempts;
  engine.submit(request(1, std::vector<uint32_t>(65, 7)));
  runUntilIdle(engine);
  require(events.completedCount == 1 && events.failedCount == 0 &&
              executor.beginAttempts == attempts + 2 && storage.releasedExtents == releases + 1 &&
              resources.lookup(other, {}).kvBoundary == 128,
          "a refused state was asked for again after each leaf, or took more than an extent");
}

// Under the host pause a released extent gives a state nothing, so an unpublishable replay point leaves other
// cached KV in place.
void testPausedPublicationInUseTakesNoKv() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  GovernedHost host;
  EngineConfig config = host.connect({});
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> other(257, 6);
  resources.beginRequest(900);
  require(resources.ensureTokens(900, 256).granted(), "the other conversation got no pages");
  resources.publishCommittedBlocks(900, other, 256, {});
  resources.endRequest(900);

  executor.reservationRoom = [] { return false; };
  const uint32_t releases = storage.releasedExtents;
  engine.submit(request(1, std::vector<uint32_t>(4097, 5)));
  double now = 1;
  require(engine.tick(now++) && executor.requests.contains(1), "the request did not start");
  host.pause(true);
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 1 && events.failedCount == 0 &&
              engine.snapshot().replayStatePublicationFailures == 1 &&
              storage.releasedExtents == releases && storage.copies.empty() &&
              resources.lookup(other, {}).kvBoundary == 256,
          "a paused publication took cached KV for buffers it could not allocate");
}

void testPreemptedDecodeRestoresItsResidentCompositeState() {
  test::TestKvStorage storage(8, 4096, 2);
  storage.budgetPages = 6;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {260, 261}) {
    auto value = request(id, std::vector<uint32_t>(65, id));
    value.maxNewTokens = 30;
    engine.submit(std::move(value));
  }
  // Lookups keep both prompts' states at 64 resident while the lanes contend
  // for pages: the preempted lane then resumes from its own.
  const std::vector<CacheLookup> held = runUntilStatesHeld(engine, resources, {260, 261});
  for (double now = 100; now < 400 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.suspensions == 1 &&
              executor.resumptions == 1 && executor.restored == 64 &&
              executor.resumedPrompts.size() == 1 &&
              executor.resumedPrompts.front().size() == 89 &&
              engine.snapshot().resourceReplayTokens == 25,
          "preempted decode did not use its cached state plus exact suffix");
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0 && events.startIds.size() == 2 &&
              events.outputs.at(260) == std::vector<uint32_t>(30, 42) &&
              events.outputs.at(261) == std::vector<uint32_t>(30, 42) &&
              events.usage.at(260) == std::pair<uint32_t, uint32_t>{65, 30} &&
              engine.snapshot().cacheHits == 0 &&
              engine.snapshot().coldMisses == 2,
          "internal cache restore changed output or request accounting");
}

// A resumed lane that replays generated history keeps its own recovery point at its last whole page, past the
// generation prompt.
void testPreemptedDecodeReplayBoundaryIgnoresTheGenerationPrompt() {
  test::TestKvStorage storage(8, 4096, 2);
  storage.budgetPages = 6;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {262, 263}) {
    auto value = request(id, std::vector<uint32_t>(65, id));
    // Long enough to move the boundary if it applied to the 89-token history
    // too: the prompt's state lands at 32, the history's at 64.
    value.generationPromptTokens = 30;
    value.maxNewTokens = 30;
    engine.submit(std::move(value));
  }
  const std::vector<CacheLookup> held = runUntilStatesHeld(engine, resources, {262, 263});
  for (double now = 100; now < 400 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.suspensions == 1 &&
              executor.resumedPrompts.size() == 1 &&
              executor.resumedPrompts.front().size() == 89 &&
              events.completedCount == 2,
          "the decode was not preempted after 24 generated tokens");
  // Prompt tokens are the request id.
  const DraftContextPlan &plan =
      executor.plans.at(executor.resumedPrompts.front().front());
  require(executor.restored == 32 && plan.replayEnd == 89 &&
              plan.boundaries.size() == 2 && plan.boundaries[0].boundary == 64,
          "the resumed decode's replay boundary applied its generation prompt");
}

void testRepeatedPreemptionRespectsBackoffAndCancellation() {
  for (bool cancel : {false, true}) {
    test::TestKvStorage storage(4, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    executor.kvGrowthBlocked = &storage.growthBlocked;
    executor.unblockGrowthOnSuspend = false;
    Events events;
    GovernedHost host;
    host.pause(true);
    engine::Engine engine(host.connect({}), resources, executor, events);
    guardReleases(storage, engine);
    storage.growthBlocked = true;
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    auto value = request(270, {270});
    value.deadlineMilliseconds = 350;
    engine.submit(std::move(value));
    require(engine.tick(1) && executor.suspensions == 1,
            "singleton host pressure did not preempt safely");
    require(!engine.tick(50) && !engine.tick(101) &&
                executor.resumeAttempts == 1 &&
                !engine.tick(150) && !engine.tick(201) &&
                executor.resumeAttempts == 2 &&
                engine.snapshot().resourceSuspensions == 1 &&
                engine.snapshot().resourceResumptions == 0 &&
                executor.prefillRows == 0,
            "failed recovery replayed work or ignored retry backoff");
    require(engine.nextWakeupMilliseconds() == 301 &&
                events.startIds.size() == 1 &&
                engine.snapshot().cacheHits + engine.snapshot().coldMisses == 1 &&
                resources.snapshot().pool.pagesActive == 0 &&
                resources.snapshot().activeRequests == 0,
            "repeated preemption retained KV or duplicated admission counters");
    if (cancel) {
      engine.cancel(270);
      static_cast<void>(engine.tick(202));
    } else {
      static_cast<void>(engine.tick(350));
    }
    require(idle(engine) && executor.requests.empty() &&
                events.completedCount == (cancel ? 1U : 0U) &&
                events.failedCount == (cancel ? 0U : 1U) &&
                events.capacityExhaustedCount == 0 && events.emitted == 0,
            "repeated preemption ignored its terminal deadline/cancel");
  }
}

// A budget denial frees the empty extents that cover what it lacks, all at once, and retries once; with nothing
// more to free a lone request fails in the same tick.
void testBudgetDenialRetriesAfterRelease() {
  for (bool recovers : {true, false}) {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    auto pages = pool.acquirePages(8);
    require(pages.granted(), "could not seed the KV extents");
    for (uint32_t page : pages.pages)
      pool.releasePage(page, false);
    Executor executor(1);
    Events events;
    GovernedHost host;
    engine::Engine engine(host.connect({.resourceWaitTimeoutMilliseconds = 500.0}), cache, executor, events);
    guardReleases(storage, engine);
    // The start lacks the storage's two extents of bytes.
    reachTheLimit(host);
    executor.beginGrowthBlocked = [&] {
      if (pool.snapshot().pagesAllocated == 0 && recovers)
        return false;
      executor.beginAllocationFailure = refusedAtTheLimit(host, 8 * 4096);
      return true;
    };
    engine.submit(request(284, {284}));
    static_cast<void>(engine.tick(1));
    // The denial released both empty extents in one free() and retried
    // once.
    require(executor.beginAttempts == 2,
            "a denied admission did not retry once after the extents it released");
    if (recovers) {
      require(executor.requests.size() == 1 && events.failedCount == 0,
              "the retry after the release did not admit the request");
      for (double now = 2; now < 400 && !idle(engine); ++now)
        static_cast<void>(engine.tick(now));
      require(events.completedCount == 1 && events.failedCount == 0,
              "state admission did not proceed once its release made room");
    } else {
      require(pool.snapshot().pagesAllocated == 0 && idle(engine) &&
                  events.failures == std::vector<std::string>{"capacity_exhausted"},
              "a budget that nothing more frees did not fail as exhausted "
              "capacity at once");
    }
    require(idle(engine) && cache.snapshot().activeRequests == 0 &&
                executor.requests.empty(),
            "a budget denial leaked request ownership");
  }
}

// A lane's start keeps the pooled buffers a lane starts from: a need of state buffers would only allocate them
// again. Its KV the engine's limit refused takes them once nothing else meets it, and fits; under critical pressure
// KV takes them first, and its need, which the storage refuses again, logs short, not a met refused again.
void testAStartKeepsThePooledLaneBuffersItsKvTakesThemLast() {
  constexpr uint64_t kExtent = 4 * 4096;
  enum class Refused { State, KvAtTheLimit, KvCritical };
  for (const Refused refused : {Refused::State, Refused::KvAtTheLimit, Refused::KvCritical}) {
    test::TestKvStorage storage(8, 4096, 4);
    GovernedHost host;
    if (refused == Refused::KvAtTheLimit)
      host.through(storage);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    engine::Engine engine(host.connect({}), cache, executor, events);
    guardReleases(storage, engine);
    // The pool holds an extent's worth of the buffers the next lane starts
    // from, beside a lane that takes none.
    host.bookStates(executor);
    host.memory.hold("states", kExtent);
    executor.heldStateBytes = kExtent;
    executor.pooledLaneBytes = kExtent;
    if (refused == Refused::State) {
      executor.beginGrowthBlocked = [] { return true; };
      executor.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
    } else if (refused == Refused::KvAtTheLimit) {
      reachTheLimit(host);
    } else {
      storage.growthBlocked = true;
      host.setPressure(MemoryPressure::Critical);
    }
    std::vector<std::string> lines;
    host.governor.ledger().setDecisionLog([&](std::string_view line) { lines.emplace_back(line); });
    engine.submit(request(286, {286}));
    runUntilIdle(engine);
    if (refused == Refused::KvCritical) {
      const auto logs = [&](std::string_view text) {
        return std::ranges::any_of(lines, [&](const std::string &line) { return line.find(text) != std::string::npos; });
      };
      require(logs("claim of request 286 needs 0.0 MiB, 1 pages: short") && !logs("yet refused again"),
              "a KV need free() found short logged as met: " + (lines.empty() ? std::string() : lines.back()));
    }
    const bool completes = refused == Refused::KvAtTheLimit;
    require(idle(engine) && (events.completedCount == 1) == completes &&
                events.failures == (completes ? std::vector<std::string>{}
                                              : std::vector<std::string>{"capacity_exhausted"}) &&
                (executor.pooledLaneBytes == kExtent) == (refused == Refused::State),
            refused == Refused::State ? "a lane's start released the pooled buffers it starts from"
            : completes               ? "KV at the engine's limit kept the pooled buffers with nothing else to take"
                                      : "KV kept the pooled buffers under critical pressure");
  }
}

// Under the host pause a lane short of state buffers takes the oldest cached state only when the states in RAM can
// cover what it lacks.
void testPausedStateAdmissionReusesCachedStates() {
  for (const std::optional<uint32_t> lacked : {std::optional<uint32_t>{1}, std::optional<uint32_t>{2},
                                               std::optional<uint32_t>{}}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    GovernedHost host;
    EngineConfig config = host.connect({});
    engine::Engine engine(config, cache, executor, events);
    guardReleases(storage, engine);
    engine.submit(request(287, std::vector<uint32_t>(65, 287)));
    runUntilIdle(engine);
    const auto cached = cache.snapshot().stateCache;
    require(cached.entries == 1, "fixture did not cache one state");

    host.pause(true);
    // Each state the cache gives up returns what the pool lacks of one.
    executor.statesLacked = [&]() -> std::optional<uint32_t> {
      if (!lacked)
        return std::nullopt;
      const uint64_t given = cache.snapshot().stateCache.evictions - cached.evictions;
      return *lacked > given ? static_cast<uint32_t>(*lacked - given) : 0;
    };
    executor.beginGrowthBlocked = [&] { return executor.statesLacked() != std::optional<uint32_t>{0}; };
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    engine.submit(request(288, {288}));
    for (double now = 10; now < 20 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    const auto after = cache.snapshot().stateCache;
    if (lacked == 1) {
      require(idle(engine) && events.completedCount == 2 && events.failedCount == 0 &&
                  after.evictions == cached.evictions + 1,
              "a paused admission did not take the cached state's buffers");
    } else {
      require(!idle(engine) && events.completedCount == 1 &&
                  engine.snapshot().scheduler.waitingResources == 1 &&
                  after.entries == 1 && after.evictions == cached.evictions,
              "a paused admission evicted a state that could not cover its lane");
    }
  }
}

// A prefill chunk denied growth keeps the extents its attempt allocated and reuses them on its retry instead of
// allocating them again.
void testDeniedGrowthAllocatesEachExtentOnce() {
  constexpr uint32_t cachedExtents = 24;
  constexpr uint32_t budgetExtents = 32;
  test::TestKvStorage storage(4 * 128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), cache, executor, events);
  guardReleases(storage, engine);
  for (uint32_t chain = 0; chain < cachedExtents; ++chain) {
    const uint64_t id = 5000 + chain;
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 128).granted(), "could not seed a cached prefix");
    cache.publishCommittedBlocks(id, std::vector<uint32_t>(129, 7000 + chain), 128, {});
    cache.endRequest(id);
  }
  storage.growthAllowed = [&](uint32_t) {
    return pool.snapshot().pagesAllocated / 4 < budgetExtents;
  };
  const KvPoolSnapshot before = pool.snapshot();
  const uint32_t attemptsBefore = storage.allocationAttempts;
  const uint32_t blocksBefore = cache.snapshot().pool.pagesPrefix;
  // The first chunk's 64 pages need 16 new extents; the budget has room for
  // 8, and evicting cached prefixes frees the other 32 pages.
  engine.submit(request(286, std::vector<uint32_t>(16 * 4 * 32 + 1, 286)));
  static_cast<void>(engine.tick(1));
  const KvPoolSnapshot after = pool.snapshot();
  require(executor.prefillRows == 2048 && events.failedCount == 0,
          "the chunk was not admitted once evictions made room");
  require(after.extentAllocations - before.extentAllocations ==
                  budgetExtents - cachedExtents &&
              after.extentReleases == before.extentReleases,
          "denied attempts allocated and released the same extents again");
  // One denied attempt, one reclaim step that evicts the shortfall and no
  // more, and a retry the free pages cover.
  require(storage.allocationAttempts - attemptsBefore ==
                  budgetExtents - cachedExtents + 1 &&
              blocksBefore - cache.snapshot().pool.pagesPrefix == 32,
          "the denial was retried per evicted page or evicted more than it lacked");
}

// The control pass runs only between commands: while one is in flight its landings and host claim throw and take
// nothing.
void testControlPassRefusesACommandInFlight() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  auto pages = pool.acquirePages(8);
  require(pages.granted(), "could not seed the KV extents");
  for (uint32_t page : pages.pages)
    pool.releasePage(page, false);
  Executor executor(1);
  executor.prefillAnchor = true;
  executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
  executor.reclaimableIdleStateBytes = 64;
  Events events;
  const EngineConfig config = test::engineConfig();
  engine::Engine engine(config, cache, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(289, {289}));
  require(engine.tick(1) && engine.commandInFlight() &&
              pool.snapshot().reclaimableBytes == 4 * 4096,
          "the prefill did not stay in flight beside an empty extent");

  rejects([&] { engine.collectLandings(1); }, "memory landings collected while a command is in flight",
          "the control pass collected landings while a command was in flight");
  rejects([&] { static_cast<void>(test::criticalPass(*config.governor)); },
          "the engine's victims taken while a command is in flight",
          "the host claim took memory while a command was in flight");
  require(executor.reclaimableIdleStateBytes == 64 &&
              pool.snapshot().extentReleases == 0 &&
              pool.snapshot().reclaimableBytes == 4 * 4096,
          "a refused reclaim released memory");

  *executor.holdPrefillUntil = true;
  require(engine.tick(2) && !engine.commandInFlight() &&
              events.completedCount == 1,
          "the held prefill did not complete");
  static_cast<void>(test::criticalPass(*config.governor));
  require(executor.reclaimedIdleStateBytes == 64 &&
              pool.snapshot().extentReleases == 2 &&
              pool.snapshot().pagesAllocated == 0,
          "the reclaim after the command did not take the idle memory");
}

// Three prompts of three blocks each fill pages 0 to 8; the first is used
// again, and the second prompt's state and blocks go, so extent 0 keeps a
// free page and extent 1 two. A fourth request continues the third prompt
// and keeps decoding; its new rows fill extent 1, which holds its pages
// already, so extent 2 holds its third block alone. The fake model marks the
// pages it writes and checks them on every step (Executor::kv).
struct ScatteredPages {
  ScatteredPages() {
    model.kv = &storage;
    guardReleases(storage, engine);
    scatter(engine, cache);
    EngineRequest running = request(4, prompt(3000, 129));
    running.maxNewTokens = 4;
    model.decodeFinishes = false;
    engine.submit(running);
    for (; now < 16 && events.outputs[4].empty(); ++now)
      static_cast<void>(engine.tick(now));
    const PageTableState table = cache.pageTable(4);
    const CacheSnapshot cached = cache.snapshot();
    require(!events.outputs[4].empty() &&
                std::vector<uint32_t>(table.pages.begin(), table.pages.end()) ==
                    std::vector<uint32_t>{6, 7, 8, 4, 5} &&
                cached.pool.pagesAllocated == 12 && cached.pool.pagesFree == 4,
            "compaction setup did not leave extent 2 with one held page");
  }

  // The three prompts, the first used again (a lookup), and the second's
  // state and blocks evicted.
  static void scatter(engine::Engine &engine, engine::Cache &cache) {
    for (uint32_t id = 1; id <= 3; ++id) {
      engine.submit(request(id, prompt(1000 * id, 97)));
      runUntilIdle(engine);
    }
    static_cast<void>(cache.lookup(prompt(1000, 97), {}));
    for (uint32_t victim = 0; victim < 4; ++victim) {
      require(test::evictStep(cache).gave(),
              "the second prompt was not evicted");
    }
  }

  static std::vector<uint32_t> prompt(uint32_t first, uint32_t tokens) {
    std::vector<uint32_t> result(tokens);
    std::iota(result.begin(), result.end(), first);
    return result;
  }

  // A fifth lane that fits only once the pool has given an extent back:
  // its state lacks an extent's bytes at the engine's limit.
  void submitLaneTheBudgetRefuses() {
    reachTheLimit(host);
    model.beginGrowthBlocked = [this] {
      if (model.lastBeginId != 5 || pool.snapshot().pagesAllocated <= 8)
        return false;
      model.beginAllocationFailure = refusedAtTheLimit(host, 4 * 4096);
      return true;
    };
    engine.submit(request(5, prompt(5000, 33)));
  }

  test::TestKvStorage storage{16, 4096, 4};
  KvPool pool{storage, 0};
  engine::Cache cache{pool, nullptr, nullptr};
  Executor model;
  Events events;
  GovernedHost host;
  engine::Engine engine{host.connect({}), cache, model, events};
  // The time of the next tick.
  double now = 1;
};

// A lane the budget refuses gathers the pool's free pages by compacting an extent before evicting any cached block.
void testStateStartGathersFreePagesBeforeEvicting() {
  ScatteredPages fixture;
  engine::Cache &cache = fixture.cache;
  Events &events = fixture.events;
  const uint64_t revision = cache.pageTable(4).revision;
  const CacheSnapshot cached = cache.snapshot();
  fixture.submitLaneTheBudgetRefuses();
  for (double now = fixture.now; now < 32 && events.startIds.back() != 5; ++now)
    static_cast<void>(fixture.engine.tick(now));
  const CacheSnapshot started = cache.snapshot();
  require(events.startIds.back() == 5 && fixture.storage.copies.size() == 1 &&
              fixture.storage.copies[0].from == 8 && fixture.storage.copies[0].to == 3 &&
              started.pool.extentCompactions == 1 && started.pool.extentReleases == 1 &&
              started.pool.pagesPrefix >= cached.pool.pagesPrefix &&
              started.stateCache.entries >= cached.stateCache.entries,
          "the lane's start evicted instead of gathering the free pages");
  const PageTableState moved = cache.pageTable(4);
  require(moved.pages[2] == 3 && moved.pages[0] == 6 && moved.pages[4] == 5 &&
              moved.revision > revision,
          "the running request did not follow its moved page");
  runUntilIdle(fixture.engine);
  require(events.completedCount == 5 && events.failedCount == 0 &&
              events.outputs[4].size() == 4,
          "a request did not complete after its page moved");
}

// A command reads its page table after its states' buffers are reserved, so it names pages that compaction moved
// for them.
void testCommandReadsItsPageTableAfterRoomForItsStates() {
  test::TestKvStorage storage{16, 4096, 4};
  KvPool pool{storage, 0};
  engine::Cache cache{pool, nullptr, nullptr};
  Executor model;
  model.kv = &storage;
  Events events;
  GovernedHost host;
  engine::Engine engine{host.connect({}), cache, model, events};
  guardReleases(storage, engine);
  ScatteredPages::scatter(engine, cache);
  // The governor admits the buffers of the continuation's replay point, an
  // extent's bytes at its limit, only once an extent has gone.
  const uint32_t releases = storage.releasedExtents;
  reachTheLimit(host);
  model.reservationRoom = [&] {
    if (storage.releasedExtents > releases)
      return true;
    model.reservationFailure = refusedAtTheLimit(host, 4 * 4096);
    return false;
  };
  engine.submit(request(4, ScatteredPages::prompt(3000, 129)));
  runUntilIdle(engine);
  require(storage.copies.size() == 1 && storage.copies[0].from == 8 &&
              pool.snapshot().extentCompactions == 1 && storage.releasedExtents == releases + 1,
          "the replay point's buffers did not empty the extent of the lane's page");
  require(model.captures.size() == 4 && events.completedCount == 4 && events.failedCount == 0 &&
              engine.snapshot().replayStatePublications == 4,
          "the command did not run on the page its block moved to");
}

// Pages move only between commands; a lane refused beside a held decode waits for it, then the pool compacts for
// it.
void testCompactionWaitsForTheCommandInFlight() {
  ScatteredPages fixture;
  engine::Engine &engine = fixture.engine;
  Events &events = fixture.events;
  double now = fixture.now;
  fixture.model.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
  for (; now < 32 && !engine.commandInFlight(); ++now)
    static_cast<void>(engine.tick(now));
  fixture.submitLaneTheBudgetRefuses();
  for (const double end = now + 10; now < end; ++now)
    static_cast<void>(engine.tick(now));
  require(engine.commandInFlight() && fixture.storage.copies.empty() &&
              fixture.pool.snapshot().extentCompactions == 0 &&
              events.startIds.back() != 5,
          "pages moved while a command was in flight");
  *fixture.model.holdDecodeUntil = true;
  for (const double end = now + 16; now < end && events.startIds.back() != 5; ++now)
    static_cast<void>(engine.tick(now));
  require(events.startIds.back() == 5 && fixture.storage.copies.size() == 1 &&
              fixture.pool.snapshot().extentCompactions == 1,
          "the lane did not start by emptying an extent once the decode completed");
  runUntilIdle(engine);
  require(events.completedCount == 5 && events.failedCount == 0,
          "a request did not complete after the held decode");
}

// Trimming windowed layers' K/V of pages every window has passed changes no output: steps read only Full pages
// inside their windows.
void testWindowTrimChangesNoOutput() {
  struct Outcome {
    std::unordered_map<uint64_t, std::vector<uint32_t>> outputs;
    std::vector<uint32_t> starts;
    uint64_t digest = 0;
    uint64_t trimmed = 0;
    uint32_t slimPages = 0;
  };
  const auto run = [](bool trim) {
    test::TestKvStorage storage(512, 4096, 4, 0, 1024);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr, std::nullopt, CacheWindowPolicy{2, trim});
    Executor model;
    model.kv = &storage;
    model.windowPages = 2;
    model.decodeFinishes = false;
    model.decodeTokens = 4;
    Events events;
    engine::Engine engine(test::engineConfig(), cache, model, events);
    guardReleases(storage, engine);
    double now = 1;
    std::vector<uint32_t> prompt = ScatteredPages::prompt(1000, 3000);
    EngineRequest first = request(1, prompt);
    first.maxNewTokens = 40;
    engine.submit(first);
    tickUntil(engine, now, [&] { return idle(engine); }, "the first request did not end");
    // The fake marks no decoded rows, so the next turn shares the prompt
    // alone.
    for (uint32_t token = 0; token < 300; ++token)
      prompt.push_back(90000 + token);
    EngineRequest second = request(2, prompt);
    second.maxNewTokens = 20;
    engine.submit(second);
    tickUntil(engine, now, [&] { return idle(engine); }, "the continuation did not end");
    const CacheSnapshot cached = cache.snapshot();
    return Outcome{events.outputs, events.starts, model.windowDigest, cached.window->trimmedPages,
                   cached.pool.slim.pagesAllocated - cached.pool.slim.pagesFree};
  };
  const Outcome trimmed = run(true), full = run(false);
  require(trimmed.outputs == full.outputs && trimmed.starts == full.starts && trimmed.digest == full.digest &&
              trimmed.starts.size() == 2 && trimmed.starts[1] > 0 && trimmed.outputs.at(2).size() == 20,
          "trimming changed what the steps read or emitted");
  require(trimmed.trimmed > 90 && trimmed.slimPages > 90 && full.trimmed == 0 && full.slimPages == 0,
          "the pages the windows passed did not give up their windowed layers, or did with trimming off");
}

// Under the host pause a windowed target's trims count as memory the request in service needs, so a long prompt
// completes.
void testWindowTrimServesThroughTheHostPause() {
  test::TestKvStorage storage(512, 4096, 4, 0, 1024);
  // The engine's limit is 128 Full pages, 4096 tokens; the prompt is 8000.
  // The host refuses growth unless a request in service needs it, then the
  // engine's limit binds.
  GovernedHost host({.limitBytes = 128 * 4096});
  host.pause(true);
  host.through(storage);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr, std::nullopt, CacheWindowPolicy{2, true});
  Executor model;
  model.kv = &storage;
  model.windowPages = 2;
  model.decodeFinishes = false;
  Events events;
  engine::Engine engine(host.connect({}), cache, model, events);
  guardReleases(storage, engine);
  EngineRequest prompt = request(1, ScatteredPages::prompt(1000, 8000));
  prompt.maxNewTokens = 8;
  engine.submit(prompt);
  double now = 1;
  tickUntil(engine, now, [&] { return idle(engine); }, "the long prompt did not end");
  const CacheSnapshot cached = cache.snapshot();
  require(events.completedCount == 1 && events.failures.empty() && events.outputs.at(1).size() == 8,
          "a prompt the trims fit failed under the host's pause");
  require(cached.window->trimmedPages > 128 && cached.pool.slim.pagesAllocated > 128,
          "the pages the prompt's windows passed stayed Full under the host's pause");
}

// Ledger takes run outside any allocation's scope: trims met by the Slim runway's last free pages allocate no
// extent.
void testTrimsAtTheSlimRunwaysLastPagesAllocateNothing() {
  test::TestKvStorage storage(64, 4096, 4, 0, 1024);
  GovernedHost host;
  host.through(storage);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr, std::nullopt, CacheWindowPolicy{2, true});
  MemoryLedger &ledger = host.governor.ledger();
  ledger.addSource(cache);
  ledger.setReturning([&cache] { return cache.returning(); },
                      [&cache](std::span<const uint64_t> requestIds) { return cache.returningFor(requestIds); });
  // Two conversations cached under a start's Floor claim: the first's eight
  // pages trim, which leaves the Slim runway an extent of free pages, and
  // the second's four Full pages wait to trim.
  const auto converse = [&](uint64_t id, uint32_t pages) {
    std::vector<uint32_t> prompt(pages * KvCache::pageTokens + 1);
    std::iota(prompt.begin(), prompt.end(), 1000 * static_cast<uint32_t>(id));
    cache.beginRequest(id);
    require(cache.ensureTokens(id, prompt.size()).granted(), "fixture KV failed");
    cache.publishCommittedBlocks(id, prompt, pages * KvCache::pageTokens, {});
    cache.endRequest(id);
  };
  {
    const test::ClaimScope start(host.governor, ClaimClass::Floor);
    converse(1, 8);
    require(cache.trimWindows() == 8, "the first conversation's pages did not trim");
    converse(2, 4);
  }
  const uint32_t fullPages = pool.freePageCount(kv::PageForm::Full);
  const uint64_t allocated = storage.allocatedBytes();
  require(pool.freePageCount(kv::PageForm::Slim) == 4, "the Slim runway is not an extent of free pages");
  const ClaimId claim = ledger.open(test::kInServiceGrowth);
  const FreeResult freed = host.governor.free(claim, {.pages = 4});
  ledger.close(claim);
  require(freed.outcome == FreeOutcome::Met && pool.freePageCount(kv::PageForm::Slim) == 0 &&
              pool.freePageCount(kv::PageForm::Full) == fullPages + 4 && storage.allocatedBytes() == allocated,
          "a page need's trims allocated, or did not take the Slim runway's pages");
  {
    const test::ClaimScope trims(host.governor, ClaimClass::InService);
    static_cast<void>(cache.trimWindows());
  }
  require(pool.freePageCount(kv::PageForm::Slim) == 4, "the Slim runway did not refill in a claim's scope");
}

void testAllocationCausesRemainDistinct() {
  for (bool stateAllocation : {false, true}) {
    for (auto reason : {metal::AllocationFailure::HostPressure,
                        metal::AllocationFailure::EngineBudget,
                        metal::AllocationFailure::DriverRejected}) {
      test::TestKvStorage storage(8, 4096, 4);
      KvPool pool(storage, 0);
      engine::Cache cache(pool, nullptr, nullptr);
      Executor executor(1);
      Events events;
      engine::Engine engine(test::engineConfig(), cache, executor, events);
      guardReleases(storage, engine);
      storage.growthBlocked = !stateAllocation;
      storage.allocationFailure = reason;
      executor.beginGrowthBlocked = [stateAllocation] { return stateAllocation; };
      executor.beginAllocationFailure = reason;
      engine.submit(request(285, {285}));
      static_cast<void>(engine.tick(1));
      if (reason == metal::AllocationFailure::HostPressure) {
        require(!idle(engine) && events.failedCount == 0 &&
                    executor.prefillRows == 0,
                "temporary host admission failure became a terminal error");
        engine.cancel(285);
        static_cast<void>(engine.tick(2));
      } else {
        require(idle(engine) && events.failures ==
                    std::vector<std::string>{"capacity_exhausted"} &&
                    events.failureDetails.size() == 1 &&
                    !events.failureDetails[0].second &&
                    events.failureDetails[0].first.find(
                        metal::allocationFailureName(reason)) != std::string::npos,
                "allocation failure lost its cause or became retryable");
      }
      require(idle(engine) && executor.requests.empty() &&
                  cache.snapshot().activeRequests == 0,
              "allocation denial leaked request ownership");
    }
  }
}

void testRecoveryAdmitsFailedKvTargetBeforeReplaying() {
  for (bool sharePrefix : {false, true}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(2);
    executor.unblockGrowthOnSuspend = false;
    executor.kvGrowthBlocked = &storage.growthBlocked;
    Events events;
    // The storage models the request-sized headroom denial while global
    // pressure is Normal: test::engineConfig's host pauses no growth.
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);
    const std::vector<uint32_t> prompt(129, 280);
    if (sharePrefix) {
      engine.submit(request(279, std::vector<uint32_t>(65, 280)));
      runUntilIdle(engine);
    }
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    // The host refuses the next extent, which the last token's page needs:
    // the first dispatch ends at the page-aligned replay state (128 tokens).
    storage.growthAllowed = [](uint32_t extent) { return extent == 0; };
    auto input = request(280, prompt);
    input.returnProgress = true;
    engine.submit(std::move(input));
    require(engine.tick(1) && engine.tick(2) && engine.snapshot().captureCuts == 1,
            "recovery fixture did not prefill up to its replay state");
    const auto cached = resources.snapshot();
    const uint32_t restored = executor.restored;
    {
      auto lookup = resources.lookup(prompt, {});
      require(lookup.resumeBoundary() == 128 &&
                  cached.pool.pagesAllocated == cached.pool.pagesActive,
              "recovery fixture needs a state backed only by active prefix pages");
    }
    storage.growthAllowed = {};
    storage.growthBlocked = true;
    require(engine.tick(3) && engine.snapshot().resourceSuspensions == 1,
            "request-sized host denial was treated as permanent capacity");
    require(resources.snapshot().stateCache.entries == cached.stateCache.entries &&
                resources.snapshot().stateCache.evictions == cached.stateCache.evictions,
            "active prefix pages were mistaken for idle pages and lost their state");
    const uint64_t rows = executor.prefillRows;
    const uint64_t replay = engine.snapshot().resourceReplayTokens;
    for (double now : {103.0, 203.0, 303.0}) {
      require(!engine.tick(now) && !engine.commandInFlight() &&
                  executor.prefillRows == rows && executor.restored == restored &&
                  engine.snapshot().resourceReplayTokens == replay &&
                  engine.snapshot().resourceResumptions == 0 &&
                  resources.snapshot().activeRequests == 0 &&
                  resources.snapshot().pool.pagesActive == 0,
              "insufficient KV replayed committed rows or retained active leases");
    }
    const auto reported = events.progress.at(280);
    require(reported.back() == 128, "prefill recovery lost completed progress");
    storage.growthBlocked = false;
    require(engine.tick(403), "recovery did not resume when KV became available");
    for (double now = 404; now < 425 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    auto expected = reported;
    expected.push_back(129);
    require(events.progress.at(280) == expected,
            "prefill recovery duplicated or regressed progress");
    require(idle(engine) && engine.snapshot().resourceResumptions == 1 &&
                executor.restored == restored + 128 &&
                engine.snapshot().resourceReplayTokens == 1 &&
                events.outputs.at(280) == std::vector<uint32_t>{42} &&
                resources.snapshot().activeRequests == 0 &&
                resources.snapshot().pool.pagesActive == 0 &&
                events.failedCount == 0 && events.capacityExhaustedCount == 0,
            "KV recovery failed to continue cleanly after headroom recovered");
  }
}

// A resumption whose disk state fails to load waits again with the same KV target, still needing the pages of the
// dispatch that suspended it.
void testFailedResumeRestoreKeepsTheKvTarget() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  executor.deniedReservations = 1000;
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.stateTier->ready = true;
  Events events;
  const EngineConfig config = test::engineConfig();
  engine::Engine engine(config, cache, executor, events);
  guardReleases(storage, engine);
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  const std::vector<uint32_t> prompt(129, 290);
  engine.submit(request(290, prompt));
  // Refused buffers for the replay state at 128, the first dispatch ends
  // there and the state goes to disk.
  require(engine.tick(1) && engine.tick(2) && executor.prefillRows == 128 &&
              engine.snapshot().diskStatePublications == 1,
          "fixture did not write the replay state to disk");
  storage.growthBlocked = true;
  require(engine.tick(3) && engine.snapshot().resourceSuspensions == 1,
          "the last prompt token's page did not suspend the request");
  storage.growthBlocked = false;
  static_cast<void>(engine.tick(103));
  require(executor.diskReads == 1 && executor.prefillRows == 128 &&
              engine.snapshot().resourceResumptions == 0,
          "resumption did not wait for its disk state");
  // The read fails and the host refuses the retry its lane; the pressure
  // controller's pass then empties the cache and releases the KV extents.
  // Afterwards one extent (128 tokens) can come back: room for replay to
  // start, not for the dispatch that suspended the request.
  executor.resumeDenied = true;
  executor.restoreControl->ready = true;
  executor.restoreControl->success = false;
  static_cast<void>(engine.tick(104));
  static_cast<void>(test::criticalPass(*config.governor));
  require(cache.snapshot().stateCache.entries == 0 && pool.snapshot().pagesAllocated == 0 &&
              engine.snapshot().resourceResumptions == 0,
          "fixture did not release the failed resumption's memory");
  executor.resumeDenied = false;
  storage.growthAllowed = [&](uint32_t) { return pool.snapshot().pagesAllocated == 0; };
  for (double now : {204.0, 304.0, 404.0}) {
    static_cast<void>(engine.tick(now));
    require(!engine.commandInFlight() && executor.prefillRows == 128 &&
                engine.snapshot().resourceResumptions == 0 &&
                engine.snapshot().resourceSuspensions == 1 &&
                cache.snapshot().activeRequests == 0,
            "replay started without the KV of the dispatch that suspended it");
  }
  storage.growthAllowed = {};
  for (double now = 504; now < 530 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.prefillRows == 128 + 129 &&
              engine.snapshot().resourceResumptions == 1 &&
              events.outputs.at(290) == std::vector<uint32_t>{42} &&
              events.failedCount == 0 && cache.snapshot().activeRequests == 0,
          "request did not replay cleanly once its KV target fit");
}

void testAdmissionReopensAfterLastSuspendedRequestResumes() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  GovernedHost host;
  host.setPressure(MemoryPressure::Warning);
  EngineConfig config = host.connect({});
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);

  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  auto longRequest = request(271, {271});
  longRequest.maxNewTokens = 1000;
  engine.submit(std::move(longRequest));
  require(engine.tick(1) && executor.suspensions == 1,
          "recovery admission fixture did not suspend its only lane");
  host.setPressure(MemoryPressure::Normal);
  storage.growthBlocked = false;
  require(engine.tick(101) && engine.tick(102) && executor.resumptions == 1,
          "recovery admission fixture did not resume its last waiting lane");

  engine.submit(request(272, {272}));
  for (double now = 103; now < 115 && events.startIds.size() < 2; ++now)
    static_cast<void>(engine.tick(now));
  require(events.startIds == std::vector<uint64_t>({271, 272}) &&
              executor.requests.contains(271) &&
              engine.snapshot().resourceSuspensions == 1,
          "finished resource recovery blocked new work until decode completed");
  engine.cancel(271);
  engine.cancel(272);
  for (double now = 115; now < 125 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && resources.snapshot().activeRequests == 0,
          "recovery admission fixture did not release its lanes");
}

void testAdmissionRespectsPriorityBeforeHashOrder() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);

  EngineRequest background = request(100, {1});
  background.priority = RequestPriority::Background;
  EngineRequest foreground = request(1, {2});
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(background));
  engine.submit(std::move(foreground));
  require(engine.tick(2), "priority admission made no progress");
  require(!events.startIds.empty() && events.startIds.front() == 1,
          "unordered request storage admitted background work first");
  runUntilIdle(engine);
}

// Two Background lanes: one waits for its initial mask without a command in
// flight; the other fills the first extent and is suspended at 7, when it
// must allocate another. The host refuses its growth even as a request in
// service's or, at the hard limit, the engine's budget refuses it. The
// resource wait limit, and so the drain, is 1000 ms.
struct SuspendedBesideAMaskWait {
  explicit SuspendedBesideAMaskWait(bool hostPause) {
    guardReleases(storage, engine);
    auto resident = constrainedRequest(1, 100'000);
    resident.priority = RequestPriority::Background;
    engine.submit(std::move(resident));
    auto growing = request(2, std::vector<uint32_t>(95, 2));
    growing.priority = RequestPriority::Background;
    growing.maxNewTokens = 1000;
    growing.deadlineMilliseconds = 100'000;
    engine.submit(std::move(growing));
    // One command prefills both prompts.
    for (double now = 1; now <= 2; ++now)
      static_cast<void>(engine.tick(now));
    require(events.maskRequests.size() == 1 &&
                engine.snapshot().scheduler.decoding == 1,
            "fixture lanes did not reach their mask wait and decode");
    host.pause(hostPause);
    storage.growthBlocked = true;
    storage.allocationFailure = hostPause ? metal::AllocationFailure::HostPressure
                                          : metal::AllocationFailure::EngineBudget;
    require(engine.tick(7) && executor.suspensions == 1 &&
                executor.requests.at(1).resident,
            "denied growth did not suspend the lane beside the resident");
  }

  test::TestKvStorage storage{64, 4096, 4};
  KvPool pool{storage, 0};
  engine::Cache resources{pool, nullptr, nullptr};
  Executor executor;
  Events events;
  GovernedHost host;
  engine::Engine engine{host.connect({.resourceWaitTimeoutMilliseconds = 1000.0}), resources, executor, events};
};

// After a suspension, admission waits for resident lanes only while memory stays short, and at most the resource
// wait limit.
void testRecoveryDrainEndsWithItsCause() {
  enum class Cause {
    PressureClears,
    KvStillShort,
    StateStillShort,
    PressurePersists,
    HardLimit
  };
  for (Cause cause : {Cause::PressureClears, Cause::KvStillShort,
                      Cause::StateStillShort, Cause::PressurePersists,
                      Cause::HardLimit}) {
    SuspendedBesideAMaskWait fixture(cause != Cause::HardLimit);
    test::TestKvStorage &storage = fixture.storage;
    engine::Cache &resources = fixture.resources;
    Executor &executor = fixture.executor;
    Events &events = fixture.events;
    engine::Engine &engine = fixture.engine;
    // A request of the suspended lane's priority waits behind it.
    auto later = request(3, {3});
    later.priority = RequestPriority::Background;
    later.deadlineMilliseconds = 100'000;
    engine.submit(std::move(later));

    const bool pauseLifts = cause == Cause::PressureClears ||
                            cause == Cause::KvStillShort ||
                            cause == Cause::StateStillShort;
    if (pauseLifts)
      fixture.host.pause(false);
    if (cause == Cause::PressureClears || cause == Cause::StateStillShort)
      storage.growthBlocked = false;
    executor.resumeDenied = cause == Cause::StateStillShort;
    if (cause == Cause::PressureClears) {
      for (double now = 8; now < 120 && events.startIds.size() < 3; ++now)
        static_cast<void>(engine.tick(now));
      require(events.startIds == std::vector<uint64_t>({1, 2, 3}) &&
                  executor.resumptions == 1 && executor.requests.at(1).resident,
              "cleared pressure kept new work waiting for the residents");
    } else {
      // Only a lifted pause lets the suspended request retry at 107; a
      // failure there resumes the drain.
      for (double now = 8; now <= 107; ++now)
        static_cast<void>(engine.tick(now));
      const uint32_t retried = executor.resumeAttempts;
      require((retried != 0) == pauseLifts,
              "suspended request retried while the drain held");
      for (double now = 108; now < 1007; ++now)
        static_cast<void>(engine.tick(now));
      require(engine.resourceWaitSnapshot(1006).draining &&
                  executor.resumeAttempts == retried &&
                  engine.nextWakeupMilliseconds() == 1007.0,
              "drain did not hold while memory stayed short");
      static_cast<void>(engine.tick(1007));
      require(!engine.resourceWaitSnapshot(1007).draining &&
                  executor.resumeAttempts > retried &&
                  events.startIds.size() == 2,
              "drain outlasted the resource wait limit");
      for (double now = 1008; now < 2010 && events.startIds.size() < 3; ++now)
        static_cast<void>(engine.tick(now));
      require(events.failures == std::vector<std::string>{"resource_timeout"} &&
                  events.capacityExhaustedCount == 0 &&
                  events.startIds.back() == 3,
              "suspended request's wait was not bounded after the drain");
    }
    for (uint64_t id : {1, 2, 3})
      engine.cancel(id);
    for (double now = 2010; now < 2030 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    require(idle(engine) && resources.snapshot().activeRequests == 0,
            "recovery drain fixture leaked its lanes");
  }
}

// The recovery drain stops waiting once a resident lane gives memory back: the suspended request retries at once.
void testRecoveryDrainEndsWhenAResidentReleasesMemory() {
  for (const bool enough : {true, false}) {
    test::TestKvStorage storage(32, 4096, 4);
    storage.budgetPages = 16;
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    EngineConfig config = test::engineConfig();
    config.resourceWaitTimeoutMilliseconds = 1000.0;
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    // The residents hold one page and four: the large one's pages go to the
    // suspended request once it ends, the small one's page is too little.
    auto small = constrainedRequest(1, 100'000);
    auto large = constrainedRequest(3, 100'000);
    large.prompt.assign(100, 3);
    engine.submit(std::move(small));
    engine.submit(std::move(large));
    double now = 1;
    tickUntil(engine, now, [&] { return events.maskRequests.size() == 2; },
              "the residents did not wait for their masks");
    // Its first command, 416 rows, needs thirteen pages of the eleven left
    // under the engine's limit.
    auto suspended = request(2, std::vector<uint32_t>(420, 2));
    suspended.deadlineMilliseconds = 100'000;
    engine.submit(std::move(suspended));
    tickUntil(engine, now, [&] { return executor.suspensions == 1; },
              "the prefill was not suspended at the limit");
    require(engine.resourceWaitSnapshot(now).draining && executor.resumeAttempts == 0,
            "the suspension did not start the drain");
    now += 2;
    engine.cancel(enough ? 3 : 1);
    for (uint32_t step = 0; step < 4; ++step)
      static_cast<void>(engine.tick(now++));
    if (enough) {
      require(engine.snapshot().resourceResumptions == 1 && executor.resumptions == 1 &&
                  !engine.resourceWaitSnapshot(now).draining,
              "the drain held after a resident released the memory it waited for");
    } else {
      require(executor.resumeAttempts == 1 && engine.snapshot().resourceResumptions == 0 &&
                  engine.resourceWaitSnapshot(now).draining,
              "a refusal after a resident ended did not resume the drain");
    }
    for (uint64_t id : {1, 2, 3})
      engine.cancel(id);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  }
}

// A state whose buffers are refused leaves no work waiting: once the host pause lifts, the drain ends and the
// suspended request resumes, though a resident's replay point was refused during the drain.
void testRefusedStateDoesNotHoldTheDrain() {
  SuspendedBesideAMaskWait fixture(true);
  engine::Engine &engine = fixture.engine;
  Executor &executor = fixture.executor;
  // While the pause holds, a resident of a higher priority than the
  // suspended one starts, is refused the buffers of its replay point, at 64,
  // with nothing to free for them, and waits for its mask.
  fixture.storage.growthBlocked = false;
  executor.deniedReservations = 100;
  const uint32_t granted = executor.reservations;
  auto refused = request(3, std::vector<uint32_t>(65, 3));
  refused.priority = RequestPriority::Foreground;
  refused.constraint = ConstraintMode::TokenMask;
  refused.maxNewTokens = 2;
  refused.deadlineMilliseconds = 100'000;
  engine.submit(std::move(refused));
  double now = 8;
  tickUntil(engine, now, [&] { return fixture.events.maskRequests.size() == 2; },
            "the higher priority did not reach its mask wait");
  require(executor.reservations == granted && engine.snapshot().replayStatePublicationFailures == 1 &&
              executor.requests.at(3).resident && engine.resourceWaitSnapshot(now).draining &&
              executor.resumeAttempts == 0,
          "the replay point was not refused during the drain");
  executor.deniedReservations = 0;
  fixture.host.pause(false);
  tickUntil(engine, now, [&] { return executor.resumptions == 1; }, "the suspended request did not resume");
  require(now < 120, "a refused state held the drain after the host pause lifted");
  for (uint64_t id : {1, 2, 3})
    engine.cancel(id);
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
}

// A request of a higher priority than every suspended one starts at once, held by neither the drain nor the
// suspended lanes.
void testHigherPriorityArrivalIsNotHeldByRecovery() {
  for (const bool hostPause : {true, false}) {
    SuspendedBesideAMaskWait fixture(hostPause);
    engine::Engine &engine = fixture.engine;
    auto urgent = request(3, {3});
    urgent.priority = RequestPriority::Foreground;
    urgent.deadlineMilliseconds = 100'000;
    engine.submit(std::move(urgent));
    double now = 8;
    tickUntil(engine, now, [&] { return fixture.events.usage.contains(3); },
              "the higher priority did not finish");
    require(now < 20 && fixture.events.startIds == std::vector<uint64_t>({1, 2, 3}) &&
                fixture.events.failedCount == 0 && !fixture.executor.requests.at(2).resident,
            "recovery held back a request above every suspended one");
    for (uint64_t id : {1, 2})
      engine.cancel(id);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  }
}

// With every lane full, a Foreground arrival suspends the lane that has done least and takes its place; an arrival
// of the residents' priority waits.
void testForegroundArrivalPreemptsWhenLanesAreFull() {
  for (const RequestPriority arrival : {RequestPriority::Foreground, RequestPriority::Normal}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);
    // Lane 3's prompt is the shortest, and the lanes decode together.
    for (uint64_t id = 1; id <= 4; ++id) {
      auto lane = request(id, std::vector<uint32_t>(id == 3 ? 33 : 65, static_cast<uint32_t>(id)));
      lane.maxNewTokens = 1000;
      engine.submit(std::move(lane));
    }
    double now = 1;
    tickUntil(engine, now, [&] { return engine.snapshot().scheduler.decoding == 4; },
              "the lanes did not decode");
    auto arriving = request(5, std::vector<uint32_t>(9, 5));
    arriving.priority = arrival;
    engine.submit(std::move(arriving));
    if (arrival == RequestPriority::Normal) {
      for (uint32_t step = 0; step < 8; ++step)
        static_cast<void>(engine.tick(now++));
      require(engine.snapshot().prioritySuspensions == 0 && !executor.requests.contains(5) &&
                  engine.resourceWaitSnapshot(now).concurrency == 1,
              "an arrival of the residents' priority took a lane");
    } else {
      tickUntil(engine, now, [&] { return events.usage.contains(5); },
                "the Foreground arrival did not finish");
      require(engine.snapshot().prioritySuspensions == 1 &&
                  engine.snapshot().resourceSuspensions == 0 && !executor.requests.at(3).resident &&
                  !engine.resourceWaitSnapshot(now).draining,
              "the arrival did not take the lane of the least work");
      tickUntil(engine, now, [&] { return engine.snapshot().resourceResumptions == 1; },
                "the suspended lane did not resume after the arrival finished");
    }
    for (uint64_t id = 1; id <= 5; ++id)
      engine.cancel(id);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
    require(events.failedCount == 0, "a lane failed");
  }
}

// A request waiting for a prefix another lane computes takes no lane until the junction lands, then takes a
// lower-priority lane.
void testPrefixWaiterTakesNoLaneUntilThePrefixLands() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id = 1; id <= 3; ++id) {
    auto lane = request(id, {static_cast<uint32_t>(id)});
    lane.priority = RequestPriority::Background;
    lane.maxNewTokens = 1000;
    engine.submit(std::move(lane));
  }
  double now = 1;
  tickUntil(engine, now, [&] { return engine.snapshot().scheduler.decoding == 3; },
            "the Background lanes did not decode");
  std::vector<uint32_t> producer(5000, 7);
  std::vector<uint32_t> waiter(5000, 7);
  std::fill(producer.begin() + 4800, producer.end(), 8);
  std::fill(waiter.begin() + 4800, waiter.end(), 9);
  auto producing = request(4, producer);
  producing.maxNewTokens = 1000;
  engine.submit(std::move(producing));
  engine.submit(request(5, waiter));
  // The producer's command ends at the junction the request waits for, and
  // is the last before it lands.
  tickUntil(engine, now, [&] { return executor.prefillRows == 3 + 4800; },
            "the producer did not reach the junction");
  require(engine.snapshot().prioritySuspensions == 0 && !executor.requests.contains(5) &&
              engine.snapshot().captureCuts == 1,
          "the request took a lane while it waited for a prefix");
  tickUntil(engine, now, [&] { return events.startIds.back() == 5; },
            "the request did not start once the prefix landed");
  require(engine.snapshot().prioritySuspensions == 1 &&
              events.starts.back() == 4800,
          "the request did not take a lane to start from the prefix");
  for (uint64_t id = 1; id <= 5; ++id)
    engine.cancel(id);
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.failedCount == 0, "a lane failed");
}

// A request below a running lane's priority takes no lane: it suspends no lower resident and leaves a free lane to
// a suspended request.
void testRequestBelowARunningPriorityTakesNoLane() {
  for (const bool recovering : {false, true}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);
    // Lane 3's prompt is the shortest.
    const uint64_t lanes = recovering ? 4 : 3;
    for (uint64_t id = 1; id <= lanes; ++id) {
      auto lane = request(id, std::vector<uint32_t>(id == 3 ? 33 : 65, static_cast<uint32_t>(id)));
      lane.priority = RequestPriority::Background;
      lane.maxNewTokens = 1000;
      engine.submit(std::move(lane));
    }
    double now = 1;
    tickUntil(engine, now, [&] { return engine.snapshot().scheduler.decoding == lanes; },
              "the Background lanes did not decode");
    // A Foreground stream takes the last lane, during recovery lane 3's.
    auto stream = request(5, {5});
    stream.priority = RequestPriority::Foreground;
    stream.maxNewTokens = 1000;
    engine.submit(std::move(stream));
    tickUntil(engine, now, [&] { return events.outputs.contains(5); },
              "the Foreground stream did not decode");
    const uint64_t suspended = recovering ? 1 : 0;
    require(engine.snapshot().prioritySuspensions == suspended,
            "the Foreground stream did not take its lane");
    engine.submit(request(6, {6}));
    if (recovering)
      engine.cancel(1);
    for (uint32_t step = 0; step < 8; ++step)
      static_cast<void>(engine.tick(now++));
    require(!executor.requests.contains(6) &&
                engine.snapshot().prioritySuspensions == suspended &&
                engine.snapshot().resourceResumptions == suspended,
            "a request below a running priority took a lane");
    engine.cancel(5);
    tickUntil(engine, now, [&] { return events.outputs.contains(6); },
              "the request did not start once the higher priority was done");
    require(engine.snapshot().prioritySuspensions == suspended,
            "the request took a lane from a lower resident");
    for (uint64_t id = 1; id <= 6; ++id)
      engine.cancel(id);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
    require(events.failedCount == 0, "a lane failed");
  }
}

// A lane short of pages waits for the copies of demoted blocks to land and
// runs on their pages: no lane yields, and no more is written than needed.
void testPagesReturnFromDemotionWithoutSuspending() {
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
  // Six cached blocks, each under a state on disk, hold six of eight pages.
  cacheBlocksUnderDiskStates(cache, 6);
  require(pool.freePageCount() == 2 && tier.demotions == 0, "fixture pages are off");

  // The first prefill command's KV reaches the replay boundary first, 96
  // tokens: three pages against two free ones.
  engine.submit(request(1, std::vector<uint32_t>(97, 7)));
  static_cast<void>(engine.tick(1));
  require(tier.demotions == 1 && executor.suspensions == 0 &&
              executor.prefillRows == 0 && events.failedCount == 0,
          "the lane yielded or more than one block was written");
  static_cast<void>(engine.tick(2));
  static_cast<void>(engine.tick(3));
  require(tier.demotions == 1 && executor.prefillRows == 0, "waiting demoted more");
  // The lane lacks two pages in all: one up to its replay boundary, where
  // its first command ends, and one for the last token. Exactly two blocks
  // are written.
  for (uint32_t step = 4; step < 40 && !idle(engine); ++step) {
    tier.complete();
    static_cast<void>(engine.tick(step));
  }
  require(idle(engine), "engine did not reach idle");
  require(events.outputs.contains(1) && executor.prefillRows == 97 && executor.suspensions == 0 &&
              events.failedCount == 0 && tier.demotions == 2,
          "lane did not run on the returned pages, or more was written than it lacked");
  const auto stats = cache.snapshot();
  require(stats.kvTier.diskBlocks == 2 && stats.kvTier.pendingPages == 0 &&
              stats.stateCache.diskBytes == 6 * 64,
          "tier accounting after the wait is off");
}

// A request refused the pages to restore its prefix closes admission behind it, so a later short request cannot
// take those pages.
void testRefusedRestoreClosesAdmission() {
  enum class Cause { GrowthBlocked, Pending };
  for (const Cause cause : {Cause::GrowthBlocked, Cause::Pending}) {
    test::TestKvStorage storage(16, 4096, 4);
    storage.budgetPages = 8;
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    tier.transferLimit = 8;
    engine::Cache cache(pool, &tier, test::ampleDiskBudget());
    Executor executor;
    executor.decodeFinishes = false;
    executor.restoreControl->ready = true;
    Events events;
    engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
    guardReleases(storage, engine);
    // Request 2's prompt: its state and both of its KV blocks are on disk.
    const std::vector<uint32_t> prompt(65, 17);
    cache.beginRequest(999);
    require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
    demoteState(cache, test::publishBlocks(cache, 999, prompt, 64));
    cache.endRequest(999);
    for (uint32_t written = 1; written <= 2; ++written) {
      require(test::evictStep(cache).gave(),
              "KV block was not written");
      tier.complete();
      require(cache.pollTransfers(), "written block did not land");
    }
    // Pending: cached blocks under states on disk take the first extent, whose
    // pages come back only once their copies are written.
    if (cause == Cause::Pending) {
      for (uint64_t id = 900; id < 904; ++id) {
        cache.beginRequest(id);
        require(cache.ensureTokens(id, 32).granted(), "fixture KV failed");
        demoteState(cache, test::publishBlocks(
                               cache, id, std::vector<uint32_t>(32, static_cast<uint32_t>(id)), 32));
        cache.endRequest(id);
      }
    }
    // Lane 1 takes the remaining pages and decodes within them.
    auto resident = request(1, std::vector<uint32_t>(97, 1));
    resident.maxNewTokens = 1000;
    engine.submit(std::move(resident));
    double now = 1;
    tickUntil(engine, now, [&] { return events.emitted != 0; }, "the resident lane did not decode");
    if (cause == Cause::GrowthBlocked) {
      storage.growthBlocked = true;
      storage.allocationFailure = metal::AllocationFailure::EngineBudget;
    }
    require(pool.freePageCount() == 0, "the fixture left pages free");
    engine.submit(request(2, prompt));
    tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
              "the restore's pages were not refused");
    require(cause == Cause::GrowthBlocked ? tier.demotions == 2 : tier.inFlight() != 0,
            "the refusal did not come from the intended cause");
    engine.submit(request(3, {3}));
    // The next admission pass holds it back.
    tickUntil(engine, now,
              [&] { return engine.resourceWaitSnapshot(now).heldBehindRefusal == 1; },
              "a later arrival was not held back behind a restore refused its pages");
    for (uint32_t step = 0; step < 10; ++step, now += 101) {
      static_cast<void>(engine.tick(now));
      require(!executor.requests.contains(3) && !events.usage.contains(3) &&
                  engine.resourceWaitSnapshot(now).heldBehindRefusal == 1 &&
                  events.failedCount == 0,
              "a later arrival started ahead of a restore refused its pages");
    }
    if (cause == Cause::Pending) {
      engine.cancel(2);
      tickUntil(engine, now, [&] { return executor.requests.contains(3) || events.usage.contains(3); },
                "admission stayed closed after the refused restore was cancelled");
    }
    executor.decodeFinishes = true;
    storage.growthBlocked = false;
    for (uint32_t step = 0; step < 100 && !idle(engine); ++step) {
      tier.complete();
      static_cast<void>(engine.tick(now++));
    }
    // A cancelled request is reported as ended.
    require(idle(engine) && events.completedCount == 3 && events.failedCount == 0,
            "the requests did not finish");
    if (cause == Cause::GrowthBlocked)
      require(events.startIds == std::vector<uint64_t>{1, 2, 3} && executor.restored == 64,
              "the refused restore did not start from its prefix before the request behind it");
  }
}

// A command between passes yields to a waiting higher-priority prompt only when that prompt's command runs, not on
// every retry.
void testAHigherPriorityThatWaitsLeavesACommandItsPasses() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.capacity = 64;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor model;
  model.groupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
  model.streamedExperts = true;
  model.passes = 6;
  // The passes of the Background prompts' commands.
  std::vector<uint32_t> passes;
  model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
    if (plan.kind == WorkKind::Prefill && plan.items.front().requestId != 3)
      passes.push_back(plan.pass);
    uint32_t rows = 0;
    for (const ModelBatchItem &item : items)
      rows += item.tokenCount;
    return test::CommandTimes{(3400.0 + rows * 1500.0 / 2048) / plan.passes, 3400.0 / plan.passes};
  };
  Events events;
  engine::Engine engine(test::engineConfig(), cache, model, events);
  guardReleases(storage, engine);
  model.time(test::passingTarget());
  for (uint64_t id = 1; id <= 2; ++id) {
    EngineRequest background = request(id, std::vector<uint32_t>(300, static_cast<uint32_t>(id)));
    background.priority = RequestPriority::Background;
    engine.submit(std::move(background));
  }
  double now = 1;
  tickUntil(engine, now, [&] { return passes.size() == 1 && !engine.commandInFlight(); },
            "the Background prompts did not run a pass");
  // Cached blocks under states on disk take the pages the engine's limit
  // leaves; they come back only once their copies are written.
  storage.budgetPages = 32;
  for (uint64_t id = 900;; ++id) {
    cache.beginRequest(id);
    if (!cache.ensureTokens(id, 32).granted()) {
      cache.endRequest(id);
      break;
    }
    demoteState(cache, test::publishBlocks(cache, id, std::vector<uint32_t>(32, static_cast<uint32_t>(id)), 32));
    cache.endRequest(id);
  }
  EngineRequest foreground = request(3, std::vector<uint32_t>(100, 3));
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  for (uint32_t step = 0; step < 50 && !(events.usage.contains(1) && events.usage.contains(2)); ++step, now += 101)
    static_cast<void>(engine.tick(now));
  require(events.usage.contains(1) && events.usage.contains(2) && !events.usage.contains(3) && tier.demotions != 0,
          "the Background prompts' command did not end while a Foreground prompt waited for copies");
  require(passes == std::vector<uint32_t>{0, 1, 2, 3, 4, 5},
          "a higher priority that waited for memory abandoned a command between its passes");
  for (uint32_t step = 0; step < 100 && !idle(engine); ++step, ++now) {
    tier.complete();
    static_cast<void>(engine.tick(now));
  }
  require(idle(engine) && events.completedCount == 3 && events.failedCount == 0,
          "the Foreground prompt did not run once its memory landed");
}

// A lane whose pages keep landing is never failed for waiting: the resource limit measures time without progress.
void testWaitWithProgressOutlivesTheResourceLimit() {
  test::TestKvStorage storage(12, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  tier.budget = test::ampleDiskBudget();
  engine::Cache cache(pool, &tier, tier.budget);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // A three-block prefix and its state move to disk entirely.
  std::vector<uint32_t> prompt(97, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 96).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 96));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 3; ++written) {
    require(test::evictStep(cache).gave() && tier.demotions == written,
            "prefix block was not written");
    tier.complete();
    require(cache.pollTransfers(), "prefix block did not land");
  }
  // Six cached blocks under disk states hold six of the eight pages.
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  for (uint64_t id = 900; id < 906; ++id) {
    std::vector<uint32_t> filler(32, static_cast<uint32_t>(id));
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 32).granted(), "filler KV failed");
    cache.publishCompositeState(test::publishBlocks(cache, id, filler, 32),
                                std::make_shared<OffloadState>(transfer), false);
    cache.endRequest(id);
    require(test::stateStep(cache).gave() && cache.pollTransfers(),
            "filler state was not demoted");
  }
  require(pool.freePageCount() == 2, "fixture pages are off");

  tier.transferLimit = 1;
  executor.restoreControl->ready = true;
  EngineRequest waiting = request(1, prompt);
  waiting.deadlineMilliseconds = 1e9;
  engine.submit(std::move(waiting));
  double now = 1.0;
  for (int round = 0; round < 40 && !idle(engine); ++round) {
    static_cast<void>(engine.tick(now));
    tier.complete();
    now += 20000.0;
  }
  require(idle(engine), "lane did not finish");
  require(now > 90000.0, "the wait did not cross the resource limit");
  require(events.failedCount == 0 && events.starts.size() == 1 &&
              events.starts[0] == 96 && executor.restored == 96 &&
              executor.prefillRows == 1 && executor.suspensions == 0,
          "a lane whose pages kept landing was failed or lost its prefix");
  require(cache.snapshot().kvTier.restores == 3,
          "resource retries lost or repeated completed KV restores");
}

// An admission rolled back in the same pass is no progress: the request fails at the limit of its first refused
// attempt.
void testRolledBackAdmissionKeepsItsWaitLimit() {
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
  // A three-block prefix and its state move to disk entirely.
  std::vector<uint32_t> prompt(97, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 96).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 96));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 3; ++written) {
    require(test::evictStep(cache).gave() && tier.demotions == written,
            "prefix block was not written");
    tier.complete();
    require(cache.pollTransfers(), "prefix block did not land");
  }
  // Six cached blocks under disk states hold six of the eight pages.
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  for (uint64_t id = 900; id < 906; ++id) {
    std::vector<uint32_t> filler(32, static_cast<uint32_t>(id));
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 32).granted(), "filler KV failed");
    cache.publishCompositeState(test::publishBlocks(cache, id, filler, 32),
                                std::make_shared<OffloadState>(transfer), false);
    cache.endRequest(id);
    require(test::stateStep(cache).gave() && cache.pollTransfers(),
            "filler state was not demoted");
  }
  require(pool.freePageCount() == 2, "fixture pages are off");

  // The demotion that would free the restore's third page never lands.
  tier.transferLimit = 1;
  EngineRequest waiting = request(1, prompt);
  waiting.deadlineMilliseconds = 1e9;
  engine.submit(std::move(waiting));
  constexpr double firstAttempt = 1.0;
  double failedAt = 0.0;
  for (double now = firstAttempt; now < firstAttempt + 40000.0 && !events.failedCount;
       now += 100.0) {
    static_cast<void>(engine.tick(now));
    failedAt = now;
  }
  require(events.failures == std::vector<std::string>{"resource_timeout"} &&
              failedAt - firstAttempt >= 30000.0 && failedAt - firstAttempt <= 30200.0 &&
              executor.suspensions == 0,
          "a rolled-back admission moved its own wait limit out");
}

// A wait limit that landed pages pushed past, until the next attempt, is no wake-up: tick() does not act on it.
void testLimitOutlivedByProgressDoesNotWakeTheLoop() {
  test::TestKvStorage storage(16, 4096, 2);
  storage.budgetPages = 14;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.capacity = 64;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // An eight-block prefix and its state move to disk entirely.
  std::vector<uint32_t> prompt(257, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 256).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 256));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 8; ++written) {
    require(test::evictStep(cache).gave() && tier.demotions == written,
            "prefix block was not written");
    tier.complete();
    require(cache.pollTransfers(), "prefix block did not land");
  }
  // Six cached blocks under disk states.
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  for (uint64_t id = 900; id < 906; ++id) {
    std::vector<uint32_t> filler(32, static_cast<uint32_t>(id));
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 32).granted(), "filler KV failed");
    cache.publishCompositeState(test::publishBlocks(cache, id, filler, 32),
                                std::make_shared<OffloadState>(transfer), false);
    cache.endRequest(id);
    require(test::stateStep(cache).gave() && cache.pollTransfers(),
            "filler state was not demoted");
  }
  EngineRequest running = request(1, std::vector<uint32_t>(33, 5));
  running.deadlineMilliseconds = 1e9;
  engine.submit(std::move(running));
  double now = 1.0;
  // One command prefills its prompt, capturing its replay state.
  for (int step = 0; step < 2; ++step, now += 100.0)
    static_cast<void>(engine.tick(now));

  // The running lane's next decode command is held, and the lane restoring
  // the prefix waits for pages the tier is still writing.
  executor.restoreControl->ready = true;
  auto hold = std::make_shared<std::atomic<bool>>(false);
  executor.holdDecodeUntil = hold;
  EngineRequest waiting = request(2, prompt);
  waiting.deadlineMilliseconds = 1e9;
  engine.submit(std::move(waiting));
  for (int step = 0; step < 3; ++step, now += 100.0)
    static_cast<void>(engine.tick(now));
  require(engine.commandInFlight() && engine.resourceWaitSnapshot(now).memory == 1 &&
              tier.demotions == 10 && events.starts.size() == 1,
          "the lane did not wait for pages behind the command");
  // The pages land, and the command is still held when the limit passes.
  tier.complete();
  static_cast<void>(engine.tick(now));
  now += 31000.0;
  require(!engine.tick(now) && events.failedCount == 0 && engine.commandInFlight(),
          "the lane failed although pages landed since its last attempt");
  const auto wakeup = engine.nextWakeupMilliseconds();
  require(wakeup.has_value() && *wakeup > now,
          "a limit tick() does not act on woke the loop");

  // The command's completion wakes the loop, and the lane's next attempt
  // takes the pages.
  *hold = true;
  executor.decodeFinishes = true;
  for (int step = 0; step < 40 && !idle(engine); ++step, now += 1.0) {
    static_cast<void>(engine.tick(now));
    tier.complete();
  }
  require(idle(engine) && events.completedCount == 2 && events.failedCount == 0 &&
              events.starts.size() == 2 &&
              events.starts[1] == 256 && executor.restored == 256,
          "the lane did not run on its prefix after the command");
}

// A lane admitted while another waited, resident at that one's suspension, holds memory its resume waits for, so
// its work restarts the limit.
void testLaneAdmittedBeforeASuspensionHoldsTheWaitOpen() {
  test::TestKvStorage storage(16, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(
      test::engineConfig({.maxContext = 102400, .resourceWaitTimeoutMilliseconds = 50.0}),
      cache, executor, events);
  guardReleases(storage, engine);
  // Cached blocks under states on disk take the first extent, whose pages
  // come back only once their copies are written.
  for (uint64_t id = 900; id < 904; ++id) {
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 32).granted(), "fixture KV failed");
    demoteState(cache, test::publishBlocks(
                           cache, id, std::vector<uint32_t>(32, static_cast<uint32_t>(id)), 32));
    cache.endRequest(id);
  }
  // Request 1's KV up to its replay boundary needs seven of the eight pages;
  // its last token all eight.
  engine.submit(request(1, std::vector<uint32_t>(225, 1)));
  double now = 1;
  static_cast<void>(engine.tick(now++));
  require(executor.requests.contains(1) && tier.inFlight() != 0 && executor.prefillRows == 0,
          "the lane did not wait for the pages being demoted");
  // Request 2 takes two of the pages left free.
  auto later = request(2, std::vector<uint32_t>(40, 2));
  later.maxNewTokens = 1000;
  engine.submit(std::move(later));
  tickUntil(engine, now, [&] { return events.emitted != 0; },
            "the later request did not run beside the waiting lane");
  require(tier.inFlight() != 0 && executor.prefillRows == 40 && executor.suspensions == 0,
          "the waiting lane ran or yielded before its pages landed");
  for (uint32_t step = 0; step < 100 && !executor.suspensions; ++step) {
    tier.complete();
    static_cast<void>(engine.tick(now++));
  }
  require(executor.suspensions == 1 && executor.prefillRows == 40,
          "the waiting lane was not suspended once the pages landed");
  for (const double end = now + 400; now < end;) {
    static_cast<void>(engine.tick(now += 10));
    require(events.failedCount == 0 && engine.snapshot().resourceResumptions == 0,
            "a lane resident at a suspension did not hold the wait open");
  }
  // Once the lane beside it ends, the suspended lane's page is the oldest
  // cached leaf's, which leaves RAM once its copy lands.
  executor.decodeFinishes = true;
  for (uint32_t step = 0; step < 1000 && !idle(engine); ++step) {
    tier.complete();
    static_cast<void>(engine.tick(now++));
  }
  require(idle(engine), "engine did not reach idle");
  require(events.completedCount == 2 && events.failedCount == 0 &&
              engine.snapshot().resourceResumptions == 1,
          "the suspended lane did not resume once the lane beside it finished");
}

// A restoring lane whose pages are all held by a resident lane waits for
// that lane instead of giving up its prefix or failing for capacity.
void testRestoringLaneWaitsForResidentLanes() {
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
  // A two-block prefix and its state move to disk.
  std::vector<uint32_t> prompt(65, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 64));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 2; ++written) {
    require(test::evictStep(cache).gave() && tier.demotions == written,
            "block was not written");
    tier.complete();
    require(cache.pollTransfers(), "block did not land");
  }
  // A long lane held in decode occupies the pool.
  auto hold = std::make_shared<std::atomic<bool>>(false);
  executor.holdDecodeUntil = hold;
  EngineRequest holder = request(2, std::vector<uint32_t>(200, 5));
  holder.deadlineMilliseconds = 1e9;
  engine.submit(std::move(holder));
  for (int step = 1; step < 6; ++step) static_cast<void>(engine.tick(step));
  require(pool.freePageCount() < 3 && engine.commandInFlight(), "the holder did not take the pool");

  executor.restoreControl->ready = true;
  EngineRequest waiting = request(1, prompt);
  waiting.deadlineMilliseconds = 1e9;
  engine.submit(std::move(waiting));
  for (int step = 10; step < 16; ++step) static_cast<void>(engine.tick(step));
  require(events.failedCount == 0 && executor.suspensions == 0 && events.starts.size() == 1,
          "the restoring lane failed or yielded instead of waiting");
  *hold = true;
  for (int step = 20; step < 60 && !idle(engine); ++step) {
    static_cast<void>(engine.tick(step));
    tier.complete();
  }
  require(idle(engine) && events.failedCount == 0 && executor.suspensions == 0 &&
              events.starts.size() == 2 && events.starts[1] == 64 &&
              executor.restored == 64,
          "the restoring lane did not run on its prefix once pages returned");
}

// A restore completes beside a tool-using lane that keeps the model busy every tick.
void testRestoreCompletesWhileAConstrainedLaneDecodes() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  // A two-block prefix and its state move to disk.
  std::vector<uint32_t> prompt(65, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 64));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 2; ++written) {
    require(test::evictStep(cache).gave() && tier.demotions == written,
            "block was not written");
    tier.complete();
    require(cache.pollTransfers(), "block did not land");
  }

  // The frontend answers every mask request on the tick that made it, and
  // transfers land between ticks.
  const std::array<uint32_t, 1> mask{1};
  uint32_t answered = 0;
  auto step = [&](double now) {
    static_cast<void>(engine.tick(now));
    for (; answered < events.maskRequests.size(); ++answered)
      engine.provideMask(events.maskRequests[answered].first, mask);
    tier.complete();
  };
  EngineRequest decoding = constrainedRequest(1, 1e9);
  decoding.maxNewTokens = 1000;
  engine.submit(std::move(decoding));
  double now = 1;
  for (; now < 20; ++now) step(now);
  require(events.emitted > 1 && events.completedCount == 0, "the constrained lane is not decoding");

  executor.restoreControl->ready = true;
  EngineRequest restoring = request(2, prompt);
  restoring.deadlineMilliseconds = 1e9;
  engine.submit(std::move(restoring));
  for (; now < 100 && events.startIds.size() < 2; ++now) step(now);
  require(events.startIds.size() == 2 && events.starts[1] == 64 &&
              executor.restored == 64,
          "the restore waited for the constrained lane to stop decoding");
  require(events.completedCount == 0 && events.failedCount == 0 &&
              cache.snapshot().kvTier.restores == 2,
          "the constrained lane stopped, or the restore did not land whole");
}

// A running request's replay point is in use: growth reclaim takes newer state and KV before it, so the next turn
// does not replay its prompt.
void testRunningRequestKeepsItsReplayPoint() {
  test::TestKvStorage storage(16, 4096, 4);
  storage.budgetPages = 12;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;

  const std::vector<uint32_t> first(65, 1);
  auto conversation = request(1, first);
  conversation.maxNewTokens = 60;
  engine.submit(std::move(conversation));
  tickUntil(engine, now, [&] { return events.outputs[1].size() >= 4; },
            "the conversation did not decode");
  require(resources.snapshot().stateCache.inUse == 1,
          "the decode did not use its replay point");
  // A newer request runs to completion: its state and KV are newer than the
  // decode's replay point.
  engine.submit(request(2, std::vector<uint32_t>(97, 2)));
  tickUntil(engine, now, [&] { return events.usage.contains(2); },
            "the newer request did not finish");
  // Growth now needs cached pages back.
  storage.growthBlocked = true;
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  engine.submit(request(3, std::vector<uint32_t>(97, 3)));
  tickUntil(engine, now, [&] { return events.usage.contains(1) && events.usage.contains(3); },
            "the conversation or the growing request did not finish");
  require(executor.suspensions == 0 && events.failedCount == 0,
          "the fixture suspended or failed a lane");

  std::vector<uint32_t> next = first;
  next.insert(next.end(), events.outputs[1].begin(), events.outputs[1].end());
  next.resize(next.size() + 40, 7);
  engine.submit(request(4, next));
  tickUntil(engine, now, [&] { return idle(engine); }, "the next turn did not finish");
  require(events.starts.back() == 64,
          "the next turn lost the replay point its predecessor ran from");
  const auto state = resources.snapshot().stateCache;
  require(state.inUse == 0 && state.inUseEvictions == 0,
          "a finished request still used its replay point");
}

// A lane suspended under host pressure keeps its replay point; another peer's growth takes a finished peer's newer
// state and KV instead.
void testSuspendedRequestKeepsItsReplayPoint() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(3);
  executor.decodeFinishes = false;
  Events events;
  GovernedHost host;
  EngineConfig config = host.connect({});
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  // Every lane needs its next page at the same step: the shortest history
  // yields, the next one finishes first.
  const std::vector<uint32_t> first(81, 1);
  const std::vector<std::pair<std::vector<uint32_t>, uint32_t>> lanes{
      {first, 40}, {std::vector<uint32_t>(113, 2), 20}, {std::vector<uint32_t>(145, 3), 210}};
  for (uint64_t id = 1; id <= lanes.size(); ++id) {
    auto value = request(id, lanes[id - 1].first);
    value.maxNewTokens = lanes[id - 1].second;
    engine.submit(std::move(value));
  }
  tickUntil(engine, now, [&] { return !events.outputs[3].empty(); }, "the lanes did not decode");
  host.pause(true);
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  tickUntil(engine, now, [&] { return executor.suspensions == 1; }, "no lane was suspended");
  // The pool grows again while the engine still drains.
  storage.growthBlocked = false;
  tickUntil(engine, now, [&] { return events.usage.contains(2); }, "the peer did not finish");
  require(executor.resumptions == 0 && resources.snapshot().stateCache.inUse == 2,
          "the suspended lane and the running peer did not both use their points");
  storage.growthBlocked = true;
  tickUntil(engine, now, [&] { return events.usage.contains(3); },
            "the growing peer did not finish");
  std::vector<uint32_t> history = first;
  history.insert(history.end(), events.outputs[1].begin(), events.outputs[1].end());
  const auto state = resources.snapshot().stateCache;
  require(executor.resumptions == 0 && state.evictions == 1 && state.inUseEvictions == 0 &&
              resources.probe(history, {}).cachedTokens() == 64,
          "the growing peer took the suspended lane's replay point");
  host.pause(false);
  storage.growthBlocked = false;
  tickUntil(engine, now, [&] { return idle(engine); }, "the suspended lane did not finish");
  require(executor.restored == 64, "the suspended lane did not resume from its replay point");
  std::vector<uint32_t> next = first;
  next.insert(next.end(), events.outputs[1].begin(), events.outputs[1].end());
  next.resize(next.size() + 40, 7);
  engine.submit(request(4, next));
  tickUntil(engine, now, [&] { return idle(engine); }, "the next turn did not finish");
  require(events.starts.back() == 64 &&
              resources.snapshot().stateCache.inUse == 0,
          "the next turn lost the replay point its predecessor resumed from");
}

// A resumed lane keeps using its prompt's replay point; the point inside its generated history is a checkpoint,
// reclaimed first once the lane ends.
void testResumedLaneKeepsThePromptReplayPoint() {
  test::TestKvStorage storage(8, 4096, 2);
  storage.budgetPages = 6;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {264, 265}) {
    auto value = request(id, std::vector<uint32_t>(65, id));
    // The prompt's replay point lands at 32, the history's at 64.
    value.generationPromptTokens = 30;
    value.maxNewTokens = 30;
    engine.submit(std::move(value));
  }
  // Lookups keep both prompts' points resident while the lanes contend for
  // pages: the reclaim that ends in a yield takes what is in use too.
  std::vector<CacheLookup> held = runUntilStatesHeld(engine, resources, {264, 265});
  double now = 100;
  for (; now < 200 && executor.suspensions == 0; ++now)
    static_cast<void>(engine.tick(now));
  held.clear();
  // Each prompt's point, then the resumed lane's history point.
  for (; now < 400 && resources.snapshot().stateCache.entries < 3; ++now)
    static_cast<void>(engine.tick(now));
  require(executor.resumptions == 1 && executor.restored == 32 &&
              resources.snapshot().stateCache.entries == 3 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "the resumed lane did not publish its history's point as a checkpoint");
  // Prompt tokens are the request id.
  const std::vector<uint32_t> history = executor.resumedPrompts.front();
  const uint64_t id = history.front();
  require(!events.usage.contains(id) && resources.probe(history, {}).cachedTokens() == 64,
          "the resumed lane finished early");
  for (; now < 400 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine), "the resumed lane did not finish");
  // The first state reclaim takes is the history's checkpoint.
  while (resources.snapshot().stateCache.entries == 3) {
    require(test::evictStep(resources).gave(),
            "the cache could not be reclaimed");
  }
  require(resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.probe(history, {}).cachedTokens() == 32,
          "the history's point outlasted the prompt's");
  // A shrink that keeps the resume point keeps the prompt's point.
  while (test::takeStep(resources, {.cls = ClaimClass::Host}, {.bytes = 1}).gave()) {
  }
  // The next turn keeps the text before the generation prompt.
  std::vector<uint32_t> next(35, static_cast<uint32_t>(id));
  next.resize(80, 7);
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.probe(next, {}).cachedTokens() == 32,
          "the shrink did not keep the prompt's replay point");
  engine.submit(request(266, next));
  for (; now < 500 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && events.starts.back() == 32,
          "the next turn did not resume from the prompt's replay point");
}

// A lane resumed from the checkpoint at the end of its generated history keeps it a checkpoint, reclaimed before
// the prompt's replay point.
void testRestoredHistoryCheckpointStaysDisposable() {
  test::TestKvStorage storage(64, 4096, 1);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  GovernedHost host;
  EngineConfig config = host.connect({});
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  const auto hostPressure = [&](bool refused) {
    host.pause(refused);
    storage.growthBlocked = refused;
    storage.allocationFailure = refused ? metal::AllocationFailure::HostPressure
                                        : metal::AllocationFailure::EngineBudget;
  };
  auto value = request(1, std::vector<uint32_t>(65, 5));
  // The prompt's replay point lands at 32; a history of 89 tokens ends at 64.
  value.generationPromptTokens = 30;
  value.maxNewTokens = 30;
  value.priority = RequestPriority::Background;
  engine.submit(std::move(value));
  double now = 1;
  tickUntil(engine, now, [&] { return !events.outputs[1].empty(); }, "the lane did not decode");
  // The page its 24th token needs is refused: the lane yields with a history
  // of 89 tokens, resumes from the prompt's point and checkpoints at 64.
  hostPressure(true);
  tickUntil(engine, now, [&] { return executor.suspensions == 1; }, "the lane was not suspended");
  hostPressure(false);
  tickUntil(engine, now, [&] { return engine.snapshot().checkpointPublications == 1; },
            "the resumed lane did not checkpoint its history's end");
  // A foreground start refused memory makes it yield again at once.
  hostPressure(true);
  auto foreground = request(2, std::vector<uint32_t>(65, 9));
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  tickUntil(engine, now, [&] { return executor.suspensions == 2; },
            "the lane did not yield to the foreground start");
  hostPressure(false);
  tickUntil(engine, now, [&] { return idle(engine); }, "the requests did not finish");
  const std::vector<uint32_t> history = executor.resumedPrompts.back();
  require(executor.resumptions == 2 && executor.restored == 32 + 64 && history.size() == 89 &&
              events.completedCount == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "the lane did not resume from its history's checkpoint, or made it ordinary");
  const uint32_t entries = resources.snapshot().stateCache.entries;
  while (resources.snapshot().stateCache.entries == entries) {
    require(test::evictStep(resources).gave(),
            "the cache could not be reclaimed");
  }
  require(resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.probe(history, {}).cachedTokens() == 32,
          "the history's checkpoint outlasted the prompt's replay point");
}

// Requests with the same prompt resume from the same replay point and each
// use it: the first to end leaves it in use.
void testSharedReplayPointCountsEachRequest() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {1, 2}) {
    auto value = request(id, std::vector<uint32_t>(65, 7));
    value.maxNewTokens = id == 1 ? 10 : 40;
    engine.submit(std::move(value));
  }
  double now = 1;
  for (; now < 100 && !events.usage.contains(1); ++now)
    static_cast<void>(engine.tick(now));
  require(events.startIds.size() == 2 && executor.restored == 64 &&
              events.usage.contains(1) && !events.usage.contains(2),
          "the fixture did not overlap two requests on one replay point");
  require(resources.snapshot().stateCache.inUse == 1,
          "the first request to end released its peer's replay point");
  for (; now < 200 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && resources.snapshot().stateCache.inUse == 0,
          "the last request to end kept its replay point in use");
}

// A prompt sent again restores the replay point its first run published and
// uses it while it runs.
void testRestoredEndpointIsInUse() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 7);
  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  executor.decodeFinishes = false;
  auto again = request(2, prompt);
  again.maxNewTokens = 10;
  engine.submit(std::move(again));
  double now = 100;
  for (; now < 120 && events.outputs[2].size() < 2; ++now)
    static_cast<void>(engine.tick(now));
  require(events.starts.back() == 64 &&
              resources.snapshot().stateCache.inUse == 1,
          "a restored replay point was not in use");
  for (; now < 200 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && resources.snapshot().stateCache.inUse == 0,
          "the restored replay point stayed in use");
}

// A request's replay point is in use once reached; with no slot free, its publication recycles another running
// request's older point.
void testReplayPointRecyclesAnOlderPointInUse() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  const auto decode = [&](uint64_t id) {
    auto value = request(id, std::vector<uint32_t>(65, id));
    value.maxNewTokens = 1000;
    engine.submit(std::move(value));
    tickUntil(engine, now, [&] { return events.outputs[id].size() >= 2; },
              "a request did not decode");
  };
  decode(1);
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.inUse == 1,
          "the older request did not cache its replay point");
  // No cache slot is free for the newer request's point.
  executor.deniedReservations = 1;
  decode(2);
  const auto snapshot = engine.snapshot();
  require(snapshot.recycledStatePublications == 1 &&
              snapshot.replayStatePublicationFailures == 0 &&
              snapshot.resources.stateCache.entries == 1 &&
              snapshot.resources.stateCache.inUse == 2 &&
              snapshot.resources.stateCache.inUseEvictions == 1,
          "the newer replay point did not recycle the older point in use");
}

// A start held back by a resident lane takes nothing in use for its memory; it waits for the lane and runs once the
// lane is done.
void testHeldBackStartTakesNothingInUse() {
  enum class Refused { State, PausedState, Restore };
  for (const Refused refused : {Refused::State, Refused::PausedState, Refused::Restore}) {
    test::TestKvStorage storage(16, 4096, 4);
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    engine::Cache resources(pool, &tier, test::ampleDiskBudget());
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);
    const std::vector<uint32_t> held(65, 2);
    if (refused == Refused::Restore) {
      // The start's prompt matches a prefix only the disk holds, and the
      // budget holds no extent beyond the one the running lane fills.
      resources.beginRequest(999);
      require(resources.ensureTokens(999, 64).granted(), "fixture KV failed");
      demoteState(resources, test::publishBlocks(resources, 999, held, 64));
      resources.endRequest(999);
      for (uint32_t written = 1; written <= 2; ++written) {
        require(test::evictStep(resources).gave() &&
                    tier.demotions == written,
                "block was not written");
        tier.complete();
        require(resources.pollTransfers(), "block did not land");
      }
      storage.budgetPages = 4;
    }
    double now = 1;
    const std::vector<uint32_t> first(65, 1);
    auto running = request(1, first);
    running.maxNewTokens = 1000;
    engine.submit(std::move(running));
    tickUntil(engine, now, [&] { return events.outputs[1].size() >= 2; },
              "the running request did not decode");
    // Nothing but what is in use is left to reclaim.
    while (test::evictStep(resources, ReclaimClass::Ordinary).gave()) {
    }
    // The one state in RAM is the running request's replay point.
    require(resources.snapshot().stateCache.inUse == 1 &&
                resources.snapshot().stateCache.bytes == 64,
            "the running request did not use its replay point");
    if (refused == Refused::State)
      executor.deniedBegins = 1000;
    // Refused by the host, the lane takes a cached state's buffers instead
    // (reuseCachedStateWhilePaused); the one cached state is in use.
    bool hostRefuses = refused == Refused::PausedState;
    executor.statesLacked = [&] { return hostRefuses ? 1U : 0U; };
    executor.beginGrowthBlocked = [&] { return hostRefuses; };
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    engine.submit(request(2, held));
    for (int step = 0; step < 20; ++step)
      static_cast<void>(engine.tick(now++));
    std::vector<uint32_t> next = first;
    next.insert(next.end(), events.outputs[1].begin(), events.outputs[1].end());
    next.resize(next.size() + 40, 7);
    require(resources.snapshot().stateCache.inUseEvictions == 0 &&
                resources.probe(next, {}).cachedTokens() == 64 &&
                engine.snapshot().scheduler.waitingResources == 1 &&
                !executor.requests.contains(2),
            "a start held back by a resident lane took the lane's replay point");
    executor.decodeFinishes = true;
    executor.deniedBegins = 0;
    hostRefuses = false;
    executor.restoreControl->ready = true;
    for (const double end = now + 200; now < end && !idle(engine); ++now) {
      static_cast<void>(engine.tick(now));
      tier.complete();
    }
    require(idle(engine) && events.completedCount == 2 && events.failedCount == 0 &&
                events.startIds.back() == 2 &&
                executor.diskReads == (refused == Refused::Restore ? 1U : 0U),
            "the held-back start did not run once the lane finished");
  }
}

// A lane that would be the first to yield takes nothing in use before it is suspended, so a decoding lane's replay
// point survives.
void testLaneThatWouldYieldTakesNothingInUse() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig(), resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  const std::vector<uint32_t> first(65, 1);
  auto conversation = request(1, first);
  conversation.maxNewTokens = 1000;
  engine.submit(std::move(conversation));
  tickUntil(engine, now, [&] { return events.outputs[1].size() >= 2; },
            "the conversation did not decode");
  require(resources.snapshot().stateCache.inUse == 1,
          "the decode did not use its replay point");
  while (test::evictStep(resources, ReclaimClass::Ordinary).gave()) {
  }
  storage.growthBlocked = true;
  engine.submit(request(2, std::vector<uint32_t>(65, 2)));
  tickUntil(engine, now, [&] { return executor.suspensions == 1; },
            "the starting lane was not suspended");
  std::vector<uint32_t> next = first;
  next.resize(first.size() + 40, 7);
  require(resources.snapshot().stateCache.inUseEvictions == 0 &&
              resources.probe(next, {}).cachedTokens() == 64,
          "the lane that yielded took the decoding conversation's replay point");
  require(!executor.requests.at(2).resident && executor.requests.at(1).resident &&
              !events.usage.contains(1),
          "the starting lane did not yield to the decoding one");
  executor.decodeFinishes = true;
  storage.growthBlocked = false;
  tickUntil(engine, now, [&] { return idle(engine); }, "the suspended lane did not finish");
  require(events.completedCount == 2 && events.failedCount == 0,
          "the lanes did not both complete");
}

// Short of critical pressure the host claim takes nothing in use, whatever its target: a running request's replay
// point stays.
void testPacedPassKeepsInServiceReplayPoint() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  const EngineConfig config = test::engineConfig();
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  // Each turn is the one before, its reply and a new message.
  std::vector<uint32_t> prompt(65);
  std::iota(prompt.begin(), prompt.end(), 1);
  const auto follow = [&](uint64_t id, size_t tokens) {
    prompt.insert(prompt.end(), events.outputs[id].begin(), events.outputs[id].end());
    prompt.resize(tokens, static_cast<uint32_t>(id) + 1000);
  };
  engine.submit(request(1, prompt));
  tickUntil(engine, now, [&] { return idle(engine); }, "the first turn did not finish");
  follow(1, 140);
  executor.decodeFinishes = false;
  auto running = request(2, prompt);
  running.maxNewTokens = 1000;
  engine.submit(std::move(running));
  tickUntil(engine, now, [&] { return events.outputs[2].size() >= 2; },
            "the second turn did not decode");
  require(events.starts.back() == 64 && resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.inUse == 1,
          "the second turn did not restore the first turn's point and publish its own");

  // Short of its margin the host claim may take what lanes hold beside
  // the cache.
  static_cast<void>(test::hostPass(*config.governor));
  require(resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.inUseEvictions == 0,
          "a paced pass nothing waited for took what a running request uses");

  executor.decodeFinishes = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "the second turn did not finish");
  follow(2, 220);
  engine.submit(request(3, prompt));
  tickUntil(engine, now, [&] { return idle(engine); }, "the third turn did not finish");
  require(events.starts.back() == 128 && events.completedCount == 3 && events.failedCount == 0,
          "the third turn did not resume from the replay point the second published");
}

// A replay point the host alone refuses, with nothing left its class may take, passes the host as its request's KV
// would.
void testReplayPointAdmittedInMarginBand() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t stateBytes = 160ULL << 20;
  constexpr uint64_t reserve = 2ULL << 30;
  // The host's spare memory falls by the state buffers it gives.
  uint64_t given = 0;
  MemoryGovernor governor(backend, 40ULL << 30, reserve, reserve, [&] {
    const uint64_t spare = reserve + kHostMarginBytes + stateBytes / 2 - given;
    return std::optional<HostMemory>{{spare, spare}};
  }, 0);
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  // A state's buffers come from the model's pool, which the states the
  // cache lets go of refill, or else from one admission.
  executor.reservationRoom = [&] {
    if (*executor.evictedStateBytes) {
      *executor.evictedStateBytes -= State().bytes();
      return true;
    }
    const metal::AllocationResult admitted =
        governor.allocationAdmission()(stateBytes, [&] { given += stateBytes; });
    executor.reservationFailure = admitted.failure;
    return static_cast<bool>(admitted);
  };
  Events events;
  EngineConfig config;
  connectToGovernor(config, governor);
  engine::Engine engine(test::engineConfig(std::move(config)), resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  std::vector<uint32_t> prompt(65);
  std::iota(prompt.begin(), prompt.end(), 1);
  engine.submit(request(1, prompt));
  tickUntil(engine, now, [&] { return idle(engine); }, "the first turn did not finish");
  require(engine.snapshot().replayStatePublications == 1 &&
              engine.snapshot().replayStatePublicationFailures == 0 && given == stateBytes &&
              governor.snapshot().hostHeadroomBytes < kHostMarginBytes,
          "the host's margin refused the replay point of the request in service");

  // The next turn resumes there, and its own point recycles the first's.
  prompt.insert(prompt.end(), events.outputs[1].begin(), events.outputs[1].end());
  prompt.resize(140, 7);
  engine.submit(request(2, prompt));
  tickUntil(engine, now, [&] { return idle(engine); }, "the second turn did not finish");
  require(events.starts.back() == 64 && engine.snapshot().replayStatePublications == 2 &&
              engine.snapshot().recycledStatePublications == 1 && given == stateBytes &&
              events.completedCount == 2 && events.failedCount == 0,
          "the second turn did not resume from the first turn's replay point");
}

// Every end of a decoding request releases its replay point: cancellation, failure, deadline and capacity
// exhaustion.
void testEveryEndReleasesTheReplayPoint() {
  enum class End { Cancel, Failure, Deadline, Capacity };
  for (const End end : {End::Cancel, End::Failure, End::Deadline, End::Capacity}) {
    // Without a peer or cached KV to take, the decode's growth past three
    // pages exhausts the capacity. The pages it lacks are no state's: its
    // own point stays cached once its end releases it, as nothing that
    // frees no page is taken for a page need.
    test::TestKvStorage storage(64, 4096, 1);
    if (end == End::Capacity)
      storage.budgetPages = 3;
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine(test::engineConfig(), resources, executor, events);
    guardReleases(storage, engine);
    auto value = request(1, std::vector<uint32_t>(65, 7));
    value.maxNewTokens = 1000;
    value.deadlineMilliseconds = end == End::Deadline ? 50 : 10000;
    engine.submit(std::move(value));
    double now = 1;
    for (; now < 40 && (events.outputs[1].size() < 2 || engine.commandInFlight()); ++now)
      static_cast<void>(engine.tick(now));
    require(resources.snapshot().stateCache.inUse == 1,
            "the decode did not use its replay point");
    if (end == End::Cancel)
      engine.cancel(1);
    else if (end == End::Failure)
      engine.failRequest(1, LaneOutcome::InvalidMask, "the request failed");
    require(end == End::Deadline || end == End::Capacity ||
                resources.snapshot().stateCache.inUse == 0,
            "the request's end did not release its replay point at once");
    for (now = 100; now < 200 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    const auto state = resources.snapshot().stateCache;
    require(idle(engine) && state.inUse == 0 &&
                state.entries == 1 &&
                events.capacityExhaustedCount == (end == End::Capacity ? 1U : 0U),
            "a request's end did not release its replay point");
  }
}

// A suspended lane keeps its replay point while it waits for and fails a disk read; cancellation or the deadline
// then release it.
void testWaitingEndsReleaseTheReplayPoint() {
  enum class End { CancelDuringRead, DeadlineDuringRead, CancelAfterFailedRead };
  for (const End end : {End::CancelDuringRead, End::DeadlineDuringRead,
                        End::CancelAfterFailedRead}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    executor.deniedReservations = 1000;
    executor.stateTier = std::make_shared<OffloadControl>();
    executor.stateTier->ready = true;
    Events events;
    engine::Engine engine(test::engineConfig(), cache, executor, events);
    guardReleases(storage, engine);
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    auto value = request(290, std::vector<uint32_t>(129, 290));
    value.deadlineMilliseconds = 1000;
    engine.submit(std::move(value));
    // Refused buffers for the replay state at 128, the first command ends
    // there and the state goes to disk; the last prompt token's page
    // suspends the request, and its resumption waits for the read.
    require(engine.tick(1) && engine.tick(2) &&
                engine.snapshot().diskStatePublications == 1,
            "fixture did not write the replay state to disk");
    storage.growthBlocked = true;
    require(engine.tick(3) && engine.snapshot().resourceSuspensions == 1,
            "the last prompt token's page did not suspend the request");
    storage.growthBlocked = false;
    static_cast<void>(engine.tick(103));
    require(executor.diskReads == 1 && cache.snapshot().stateCache.inUse == 1,
            "the resumption did not wait for its replay point");
    executor.restoreControl->ready = true;
    if (end == End::CancelDuringRead) {
      engine.cancel(290);
      static_cast<void>(engine.tick(104));
    } else if (end == End::DeadlineDuringRead) {
      static_cast<void>(engine.tick(1000));
    } else {
      executor.resumeDenied = true;
      executor.restoreControl->success = false;
      static_cast<void>(engine.tick(104));
      require(!idle(engine) && !engine.commandInFlight() &&
                  cache.snapshot().stateCache.inUse == 1,
              "a failed read ended the replay point's use");
      engine.cancel(290);
      require(cache.snapshot().stateCache.inUse == 0,
              "cancelling the waiting lane did not release its replay point");
      static_cast<void>(engine.tick(105));
    }
    require(idle(engine) && cache.snapshot().stateCache.inUse == 0 &&
                events.completedCount + events.failedCount == 1,
            "a waiting request's end did not release its replay point");
  }
}

// A conversation that finished and one that decodes: an ordinary replay
// point and one in use, beside the store's chunks above its floor; the KV
// in extents of `extentPages` pages, of `pages` pages in all.
struct StoreFixture {
  explicit StoreFixture(uint32_t lanes = 2, uint32_t extentPages = 4, uint32_t pages = 16)
      : storage(pages, 4096, extentPages), executor(lanes) {
    guardReleases(storage, engine);
    engine.submit(request(1, std::vector<uint32_t>(65, 1)));
    tickUntil(engine, now, [&] { return idle(engine); }, "the first conversation did not finish");
    executor.decodeFinishes = false;
    auto running = request(2, std::vector<uint32_t>(65, 2));
    running.maxNewTokens = 1000;
    engine.submit(std::move(running));
    tickUntil(engine, now, [&] { return events.outputs[2].size() >= 2; },
              "the long decode did not start");
    require(resources.snapshot().stateCache.entries == 2 &&
                resources.snapshot().stateCache.inUse == 1,
            "the fixture did not cache an ordinary replay point and one in use");
  }
  // Lets the decode finish.
  void finish() {
    executor.decodeFinishes = true;
    tickUntil(engine, now, [&] { return idle(engine); }, "the decode did not finish");
    require(events.failedCount == 0, "a request failed");
  }

  test::TestKvStorage storage;
  KvPool pool{storage, 0};
  engine::Cache resources{pool, nullptr, nullptr};
  Executor executor;
  Events events;
  GovernedHost host;
  engine::Engine engine{host.connect({}), resources, executor, events};
  double now = 1;
};

// The expert store as the ledger's pool over `host`'s memory: units of an
// extent's bytes, of which `above` above its floor of one; registered as a
// start registers the model's pools (Bootstrap).
struct Store {
  Store(GovernedHost &host, Executor &executor, uint32_t above)
      : experts("experts", host.memory, 4 * 4096, 1, 1 + above, 8, 1 + above), executor(executor) {
    executor.experts = &experts;
    static_cast<void>(host.governor.ledger().addPool(experts));
  }
  ~Store() { executor.experts = nullptr; }
  Store(const Store &) = delete;
  Store &operator=(const Store &) = delete;
  test::FakeElasticPool experts;
  Executor &executor;
};

// A copy ahead of the leaf a refusal would free holds it back as a demotion does; a host-refused need waits for no
// copy and is short instead.
void testCopyAheadOfALeafHoldsARefusalBack() {
  for (const bool hostRefused : {false, true}) {
    test::TestKvStorage storage(4, 100, 1);
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    engine::Cache cache(pool, &tier, test::ampleDiskBudget());
    auto transfer = std::make_shared<OffloadControl>();
    transfer->ready = true;
    // A finished conversation's leaf with its state in RAM, and a state in
    // use on a chain of its own.
    const auto cached = [&](uint64_t id) {
      std::vector<uint32_t> prompt(KvCache::pageTokens + 1);
      std::iota(prompt.begin(), prompt.end(), 1000 * static_cast<uint32_t>(id));
      cache.beginRequest(id);
      require(cache.ensureTokens(id, KvCache::pageTokens).granted(), "fixture KV failed");
      const uint64_t block = test::publishBlocks(cache, id, prompt, KvCache::pageTokens);
      cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer), false);
      cache.endRequest(id);
      return block;
    };
    const uint64_t leaf = cached(1), used = cached(2);
    StateUse use = cache.useState(used);
    // The leaf's state copy lands; its page's stays in flight.
    require(cache.persist(leaf) == PersistStatus::Started && cache.pollTransfers() && tier.inFlight() == 1 &&
                cache.stateResident(leaf),
            "the fixture's copies did not start");
    const uint32_t pages = pool.freePageCount() + 1;
    // A page scan is the need of a refusal the host refused.
    const Need need{.hostRefused = hostRefused, .pages = pages};
    FreeResult refused;
    do
      refused = test::freeFromCache(cache, test::kInServiceGrowth, need);
    while (refused.outcome == FreeOutcome::Met);
    require(refused.outcome == (hostRefused ? FreeOutcome::Short : FreeOutcome::Pending) &&
                (hostRefused || !cache.stateResident(leaf)) && cache.stateResident(used) &&
                !cache.snapshot().stateCache.inUseEvictions && pool.snapshot().pagesPrefix == 2 &&
                tier.inFlight() == 1,
            "a refusal took what is in use while the leaf it frees next waited for its copy");
    tier.complete();
    require(cache.pollTransfers() && test::freeFromCache(cache, test::kInServiceGrowth, need).outcome == FreeOutcome::Met &&
                !cache.stateResident(leaf) && cache.stateResident(used) && pool.snapshot().pagesPrefix == 1 &&
                tier.demotions == 1,
            "the refusal that followed did not free the copied leaf, writing nothing");
  }
}

// A lone lane whose growth needs the pages of a cached chain whose copies ahead are in flight above its leaves
// takes the leaves and waits for the copies to land: it does not end as if at its capacity. On the M6 a 32K prompt
// that fit alone failed 24,576 rows in, its claim short while a restore point's chain was copied ahead.
void testLoneGrowthWaitsForCopiesAheadAboveTheLeaves() {
  test::TestKvStorage storage(16, 4096, 4);
  GovernedHost host;
  host.through(storage);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.capacity = 64;
  tier.transferLimit = 8;
  // The copies ahead of the chain's three top blocks stay in flight; those below have not started.
  tier.writeLimits[static_cast<size_t>(WriteClass::Persist)] = 3;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  // A finished conversation's eight blocks, two extents, its state at the leaf, the blocks without copies.
  const auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  std::vector<uint32_t> cached(8 * KvCache::pageTokens + 1);
  std::iota(cached.begin(), cached.end(), 9000);
  uint64_t leaf = 0;
  {
    const test::ClaimScope fixture(host.governor, ClaimClass::Floor);
    cache.beginRequest(900);
    require(cache.ensureTokens(900, 8 * KvCache::pageTokens).granted(), "fixture KV failed");
    leaf = test::publishBlocks(cache, 900, cached, 8 * KvCache::pageTokens);
    cache.publishCompositeState(leaf, std::make_shared<OffloadState>(transfer), false);
    cache.endRequest(900);
  }
  require(cache.persist(leaf) == PersistStatus::Started && cache.pollTransfers() && tier.inFlight() == 3,
          "the copies ahead of the chain did not start");
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(host.connect({}), cache, executor, events);
  guardReleases(storage, engine);
  // The engine's limit leaves one more extent: the lane's eleven pages are its four, the five below the copies,
  // which their demotions free, and two of the three the copies hold, which go as they land.
  host.leaveRoom(4 * 4096);
  auto value = request(1, std::vector<uint32_t>(11 * KvCache::pageTokens - 8, 1));
  value.maxNewTokens = 4;
  engine.submit(std::move(value));
  // Every write but the copies ahead lands as it is due.
  const auto landWrites = [&] {
    for (const auto &each : tier.transfers)
      if (each->write != WriteClass::Persist && !each->finished)
        each->ready = true;
  };
  double now = 1;
  for (int tick = 0; tick < 200 && !events.failedCount && !idle(engine); ++tick) {
    landWrites();
    static_cast<void>(engine.tick(now += 5));
  }
  require(events.failedCount == 0 && !idle(engine) && tier.inFlight() == 3,
          "a lone lane short of pages the copies in flight held ended (" +
              (events.failureDetails.empty() ? std::string("no failure") : events.failureDetails.front().first) +
              "), or did not wait for them");
  tickUntil(engine, now, [&] { tier.complete(); return idle(engine); }, "the request did not finish");
  require(events.completedCount == 1 && events.failedCount == 0 && events.outputs[1].size() == 4,
          "the lane did not take the copied chain's pages once the copies landed");
}

// A lone request's prefix restore that only a cached state's buffers make room for takes them, though they refill the
// buffers a next lane starts from: it restores its prefix rather than start cold.
void testLoneRestoreTakesTheBuffersALaneStartsFrom() {
  // An extent of four pages takes the bytes of a state's buffers, and so does
  // a lane's footprint.
  test::TestKvStorage storage(16, State().bytes() / 4, 4);
  GovernedHost host;
  host.through(storage);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.capacity = 8;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor(1);
  executor.laneFootprintBytes = State().bytes();
  Events events;
  engine::Engine engine(host.connect({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  host.bookStates(executor);
  executor.restoreControl->ready = true;
  // The prefix, four blocks, its KV and its state on the SSD.
  const std::vector<uint32_t> prefix(129, 17);
  {
    const test::ClaimScope fixture(host.governor, ClaimClass::Floor);
    cache.beginRequest(999);
    require(cache.ensureTokens(999, 128).granted(), "fixture KV failed");
    test::demoteState(cache, test::publishBlocks(cache, 999, prefix, 128));
    cache.endRequest(999);
    for (uint32_t written = 1; written <= 4; ++written) {
      require(test::evictStep(cache).gave(), "KV block was not written");
      tier.complete();
      require(cache.pollTransfers(), "written block did not land");
    }
    require(test::releaseStep(cache).gave() && storage.allocatedPages() == 0 &&
                cache.snapshot().kvTier.diskBlocks == 4,
            "the prefix did not move to the SSD whole");
  }
  // Another conversation's state in RAM.
  const std::vector<uint32_t> other(33, 23);
  engine.submit(request(1, other));
  runUntilIdle(engine);
  require(cache.snapshot().stateCache.bytes != 0, "the other conversation's state is not in RAM");
  // The engine's limit leaves the restore's five pages two extents only once
  // that state's buffers left memory.
  host.leaveRoom(0);
  engine.submit(request(2, prefix));
  double now = 1;
  tickUntil(engine, now, [&] { tier.complete(); return idle(engine); }, "the request did not finish");
  require(events.completedCount == 2 && events.failedCount == 0 && events.starts.size() == 2 &&
              events.starts[1] == 128 && executor.prefillRows == 33 + 1,
          "a lone restore that a cached state's buffers make room for started cold or failed");
}

// KV a prefill command takes past its first boundary, refused at the limit, takes neither the store nor what is in
// use; the command ends at the boundary.
void testCaptureKvPastABoundaryTakesNothingInUse() {
  StoreFixture fixture(2, 1);
  Executor &executor = fixture.executor;
  engine::Engine &engine = fixture.engine;
  while (test::evictStep(fixture.resources, ReclaimClass::Ordinary).gave()) {
  }
  static_cast<void>(fixture.resources.releaseEmptyExtents(false));
  Store store(fixture.host, executor, 1);
  // The engine's limit refuses growth from the replay point's reservation
  // to the command's submission.
  bool refusing = false;
  executor.reservationObserver = [&] { refusing = true; };
  executor.submitObserver = [&] { refusing = false; };
  fixture.storage.growthAllowed = [&](uint32_t) { return !refusing; };
  const uint64_t cuts = engine.snapshot().captureCuts;
  engine.submit(request(3, std::vector<uint32_t>(129, 3)));
  tickUntil(engine, fixture.now, [&] { return engine.snapshot().captureCuts > cuts; },
            "the command did not end at its boundary");
  require(fixture.resources.snapshot().stateCache.inUseEvictions == 0 && store.experts.shrinks == 0,
          "the KV past a boundary took the expert store or what is in use");
  executor.reservationObserver = {};
  executor.submitObserver = {};
  fixture.storage.growthAllowed = {};
  fixture.finish();
}

// KV past a command's first boundary that only the host's margin refuses passes the host as the lane's growth does,
// so the command is not cut.
void testCaptureKvPastTheMarginCutsNoCommand() {
  constexpr uint64_t pageBytes = 423'936;
  constexpr uint32_t extentPages = 256;
  struct Case final {
    uint32_t length;
    uint64_t cuts;
  };
  std::string failures;
  // 18,000 tokens end at a replay point whose own buffers the margin refuses while the lane holds a checkpoint.
  for (const Case &expected : {Case{12'000, 0}, Case{18'000, 1}}) {
    test::EngineHarness harness({.pages = 4 * extentPages,
                                 .pageBytes = pageBytes,
                                 .extentPages = extentPages,
                                 .memory = test::FakeMemory::Config{},
                                 .timing = &test::kM6FlashNextQ2,
                                 .shape = test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget,
                                                             test::kM6FlashNextQ2.layers,
                                                             test::streamsExperts(test::kM6FlashNextQ2))});
    harness.model.time(test::kM6FlashNextQ2);
    // The host leaves one extent and 60 MiB beside its margin.
    const test::GovernorLimits limits;
    harness.memory->setHost(harness.memory->hostBytes() - limits.hostSpareReserveBytes - harness.memory->total() -
                                kHostMarginBytes - (60ULL << 20) - pageBytes * extentPages,
                            MemoryPressure::Normal);
    EngineRequest prompt = harness.request(1, std::vector<uint32_t>(expected.length, 1));
    prompt.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(prompt));
    harness.runUntilIdle();
    require(harness.events.completedCount == 1 && harness.events.failedCount == 0,
            "a prompt of " + std::to_string(expected.length) + " tokens did not complete");
    const uint64_t cuts = harness.engine.snapshot().captureCuts;
    if (cuts != expected.cuts)
      failures += "; " + std::to_string(expected.length) + " tokens: " + std::to_string(cuts) + " capture cuts";
  }
  require(failures.empty(), "the KV past a checkpoint the host's margin refused cut a lone prompt's command" + failures);
}

// A replay point takes a store unit above its floor before a state in use, and the state in use only once the store
// is at its floor.
void testReplayPointTakesTheExpertStoreBeforeWhatIsInUse() {
  for (const bool atFloor : {false, true}) {
    StoreFixture fixture;
    Executor &executor = fixture.executor;
    while (test::evictStep(fixture.resources, ReclaimClass::Ordinary).gave()) {
    }
    static_cast<void>(fixture.resources.releaseEmptyExtents(false));
    Store store(fixture.host, executor, atFloor ? 0 : 2);
    reachTheLimit(fixture.host);
    // Refused an extent's bytes at the limit until a unit went, or the state
    // in use did.
    executor.reservationRoom = [&] {
      if (store.experts.shrinks || fixture.resources.snapshot().stateCache.inUseEvictions)
        return true;
      executor.reservationFailure = refusedAtTheLimit(fixture.host, 4 * 4096);
      return false;
    };
    const uint64_t publications = fixture.engine.snapshot().replayStatePublications;
    fixture.engine.submit(request(3, std::vector<uint32_t>(65, 3)));
    tickUntil(fixture.engine, fixture.now,
              [&] { return fixture.engine.snapshot().replayStatePublications > publications; },
              "the replay point was not published");
    require(store.experts.shrinks == (atFloor ? 0U : 1U) &&
                fixture.resources.snapshot().stateCache.inUseEvictions == (atFloor ? 1U : 0U),
            atFloor ? "the replay point did not take a state in use with the store at its floor"
                    : "the replay point took a state in use before a unit of the expert store");
    executor.reservationRoom = {};
    fixture.finish();
  }
}

// A replay point refused at the limit takes cache before the store, starting copies of unwritten cache; meanwhile
// the store keeps its units.
void testReplayPointCopiesCacheBeforeTheExpertStore() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  // The tier takes the copies of an extent's pages at once.
  tier.transferLimit = 4;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({}), cache, executor, events);
  guardReleases(storage, engine);
  // The request holds what it takes for the time its work takes.
  executor.time(test::passingTarget());
  std::vector<uint32_t> prompt(129);
  std::iota(prompt.begin(), prompt.end(), 1000);
  cache.beginRequest(1);
  require(cache.ensureTokens(1, 128).granted(), "fixture KV failed");
  const uint64_t leaf = test::publishBlocks(cache, 1, prompt, 128);
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  cache.publishCompositeState(
      leaf, std::make_shared<OffloadState>(transfer, executor.evictedStateBytes), false);
  cache.endRequest(1);
  // The state goes first, written to the disk; its pages stay.
  require(test::evictStep(cache, ReclaimClass::Ordinary).gave(),
          "the fixture's state did not go");
  static_cast<void>(cache.pollTransfers());
  require(!cache.stateResident(leaf) && pool.snapshot().pagesPrefix != 0 && tier.demotions == 0,
          "the fixture did not leave its pages in RAM without copies");
  // A unit of the store above its floor.
  Store store(host, executor, 1);
  reachTheLimit(host);
  // The replay point is refused an extent's bytes at the limit until the
  // conversation's pages left RAM or a unit of the store went.
  executor.reservationRoom = [&] {
    if (pool.snapshot().pagesPrefix == 0 || store.experts.shrinks)
      return true;
    executor.reservationFailure = refusedAtTheLimit(host, 4 * 4096);
    return false;
  };
  engine.submit(request(3, std::vector<uint32_t>(65, 3)));
  double now = 1;
  for (int step = 0; step < 200 && !idle(engine); ++step) {
    static_cast<void>(engine.tick(now++));
    tier.complete();
  }
  require(idle(engine) && events.failedCount == 0, "the request did not finish");
  require(!store.experts.shrinks && tier.demotions != 0,
          "the replay point took a unit of the expert store before the cache that needed a copy");
  executor.reservationRoom = {};
}

// A lone lane's replay point that no take can meet takes nothing: refused, it goes on without buffers of its own
// (Refusal::Settled), so its claim is not lone (ClaimSpec::lone) and takes only what meets its need. Refused twice the
// bytes of the expert store's unit above its floor, until its command is cut there, the store keeps that unit.
void testALoneReplayPointThatCannotBeMetTakesNothing() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({}), cache, executor, events);
  guardReleases(storage, engine);
  Store store(host, executor, 1);
  reachTheLimit(host);
  executor.reservationRoom = [&] {
    if (engine.snapshot().captureCuts)
      return true;
    executor.reservationFailure = refusedAtTheLimit(host, 2 * 4 * 4096);
    return false;
  };
  engine.submit(request(3, std::vector<uint32_t>(65, 3)));
  double now = 1;
  tickUntil(engine, now, [&] { return idle(engine); }, "the request did not finish");
  require(events.completedCount == 1 && engine.snapshot().captureCuts == 1 && store.experts.shrinks == 0,
          "a lone replay point no take could meet took a unit of the expert store: " +
              std::to_string(store.experts.shrinks) + " shrinks, " +
              std::to_string(engine.snapshot().captureCuts) + " capture cuts");
  executor.reservationRoom = {};
}

// A replay point the host refuses while the only takeable state is still being copied waits for that copy and then
// passes the host, so a lone prompt runs as one command.
void testAHostRefusedReplayPointCutsNoCommand() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, test::ampleDiskBudget());
  Executor executor;
  Events events;
  GovernedHost host;
  engine::Engine engine(host.connect({}), cache, executor, events);
  guardReleases(storage, engine);
  // A finished conversation's state in RAM, copied ahead to the SSD: a copy
  // that never lands here, which reads the RAM copy.
  std::vector<uint32_t> prompt(129);
  std::iota(prompt.begin(), prompt.end(), 1000);
  cache.beginRequest(1);
  require(cache.ensureTokens(1, 128).granted(), "fixture KV failed");
  const uint64_t leaf = test::publishBlocks(cache, 1, prompt, 128);
  cache.publishCompositeState(
      leaf, std::make_shared<OffloadState>(std::make_shared<OffloadControl>(), executor.evictedStateBytes), false);
  cache.endRequest(1);
  require(cache.persist(leaf) == PersistStatus::Started && cache.stateResident(leaf), "the state's copy did not start");
  // No idle state buffer: a state's buffers take one admission, which the
  // host refuses under its warning.
  executor.reservationRoom = [&] {
    executor.reservationFailure = host.reserve(State().bytes()).failure;
    return executor.reservationFailure == metal::AllocationFailure::None;
  };
  host.setPressure(MemoryPressure::Warning);
  std::vector<uint32_t> rows;
  executor.commandTimed = [&](const BatchPlan &plan, const test::CommandTimes &) {
    for (const BatchItem &item : plan.items)
      if (plan.kind == WorkKind::Prefill && item.requestId == 3)
        rows.push_back(item.tokenCount);
  };
  const uint64_t captures = engine.snapshot().stateCaptures;
  engine.submit(request(3, std::vector<uint32_t>(65, 3)));
  double now = 1;
  tickUntil(engine, now, [&] { return idle(engine); }, "the request did not finish");
  std::string shown;
  for (const uint32_t each : rows)
    shown += " " + std::to_string(each);
  require(events.completedCount == 1 && events.failedCount == 0 && rows == std::vector<uint32_t>{65} &&
              engine.snapshot().captureCuts == 0 && engine.snapshot().stateCaptures == captures + 1 &&
              host.governor.snapshot().claims.waiting == 0,
          "a lone prompt's replay point, refused by the host with only a copying state to free, cut its command: "
          "commands" + shown + ", " + std::to_string(engine.snapshot().captureCuts) + " capture cuts");
  executor.reservationRoom = {};
  executor.commandTimed = {};
}

// Priority preemption fires when memory, not width, is what a higher-priority start lacks: it takes the lower
// lane's memory and runs first.
void testMemoryNotWidthTriggersPriorityPreemption() {
  // The engine's limit: seven extents of four pages, the states besides.
  test::EngineHarness harness({.memory = test::FakeMemory::Config{}, .limits = {.limitBytes = 28 * 4096}});
  harness.model.decodeFinishes = false;
  EngineRequest background = harness.request(1, std::vector<uint32_t>(65, 1));
  background.priority = RequestPriority::Background;
  background.maxNewTokens = 600;
  background.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(background));
  harness.runUntil([&] { return harness.events.outputs[1].size() >= 450; }, "the background lane did not decode");
  // 515 tokens in five extents; the Foreground prompt's 16 pages need four
  // more than the three its last extent has free, and two fit.
  require(harness.model.maximumLanes > 1 && harness.storage.allocatedPages() == 20,
          "the fixture's lane does not hold five extents with lanes to spare");
  EngineRequest foreground = harness.request(2, std::vector<uint32_t>(500, 2));
  foreground.maxNewTokens = 8;
  foreground.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(foreground));
  harness.runUntil([&] { return harness.events.usage.contains(2); }, "the foreground request did not finish");
  require(harness.events.failedCount == 0 && !harness.events.usage.contains(1) &&
              harness.engine.snapshot().resourceSuspensions == 1,
          "a foreground start the limit refused beside free lanes did not take the background lane's memory: "
          "suspensions " + std::to_string(harness.engine.snapshot().resourceSuspensions) + ", failed " +
              std::to_string(harness.events.failedCount) + ", background done " +
              std::to_string(harness.events.usage.contains(1)) + ", outputs " +
              std::to_string(harness.events.outputs[1].size()) + ", pages " +
              std::to_string(harness.storage.allocatedPages()));
  harness.runUntilIdle();
  require(harness.events.completedCount == 2 && harness.events.failedCount == 0 &&
              harness.events.outputs[1].size() == 600,
          "the background request did not finish after it gave way");
}

// A lane short of half a store unit, with no cache to take, takes back one store unit and the store keeps the rest.
void testSmallShortfallFreesAtMostOneStoreUnitMore() {
  constexpr uint64_t kUnit = 16 * 4096;
  // The store holds four units above its floor of one; beside it the limit
  // leaves the KV four extents and a page, and the lane needs more than
  // that and less than a unit more.
  test::EngineHarness harness(
      {.pages = 128, .memory = test::FakeMemory::Config{}, .limits = {.limitBytes = 5 * kUnit + 16 * 4096 + 4096}});
  test::FakeElasticPool experts("experts", *harness.memory, kUnit, 1, 5, 8, 5);
  harness.model.experts = &experts;
  static_cast<void>(harness.governor->ledger().addPool(experts));
  harness.model.decodeFinishes = false;
  EngineRequest request = harness.request(1, std::vector<uint32_t>(65, 1));
  request.maxNewTokens = 700;
  request.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(request));
  harness.runUntilIdle();
  require(harness.events.completedCount == 1 && harness.events.failedCount == 0 &&
              harness.storage.allocatedPages() > 17 && harness.storage.allocatedPages() <= 32,
          "the lane did not grow past its room by less than a unit: pages " +
              std::to_string(harness.storage.allocatedPages()) + ", store units " + std::to_string(experts.units()) +
              ", failed " + std::to_string(harness.events.failedCount) +
              (harness.events.failures.empty() ? "" : " (" + harness.events.failureDetails.front().first + ")"));
  require(experts.shrinks == 1 && experts.units() == 4,
          "a shortfall of half a store unit took more than one unit of the store");
}

// A conversation's KV chain and the state that resumes it leave together, so its next turn starts where both reach.
void testKvChainAndItsCheckpointLeaveTogether() {
  test::EngineHarness harness({.memory = test::FakeMemory::Config{}, .limits = {.limitBytes = 40 * 4096 + 4096}});
  const auto turn = [&](uint64_t id, const std::vector<uint32_t> &prompt) {
    EngineRequest next = harness.request(id, prompt);
    next.maxNewTokens = 1;
    harness.engine.submit(std::move(next));
    harness.runUntil([&] { return harness.events.usage.contains(id); }, "a turn did not finish");
  };
  std::vector<uint32_t> first(129);
  std::iota(first.begin(), first.end(), 1);
  turn(1, first);
  for (uint32_t id = 2; id <= 13; ++id) {
    std::vector<uint32_t> other(129);
    std::iota(other.begin(), other.end(), id * 1000);
    turn(id, other);
  }
  uint32_t kv = 0, resume = 0;
  {
    const CacheLookup lookup = harness.cache.lookup(first, {});
    kv = lookup.kvBoundary;
    resume = lookup.resumeBoundary();
  }
  require(harness.engine.snapshot().resources.stateCache.evictions > 0 || harness.storage.releasedExtents > 0,
          "twelve conversations did not press the cache");
  require(kv == resume, "the cache kept " + std::to_string(kv) + " tokens of KV past the state at " +
                            std::to_string(resume) + " that resumes them");
  std::vector<uint32_t> next = first;
  next.insert(next.end(), harness.events.outputs[1].begin(), harness.events.outputs[1].end());
  next.resize(200, 7);
  turn(14, next);
  require(harness.events.starts.back() == resume && harness.events.failedCount == 0,
          "the next turn did not start where its chain and its state both reach");
}

} // namespace

std::span<const test::TestCase> memoryTests() {
  static constexpr test::TestCase kTests[] = {
      {"testRunningRequestKeepsItsReplayPoint", testRunningRequestKeepsItsReplayPoint},
      {"testSuspendedRequestKeepsItsReplayPoint", testSuspendedRequestKeepsItsReplayPoint},
      {"testResumedLaneKeepsThePromptReplayPoint", testResumedLaneKeepsThePromptReplayPoint},
      {"testRestoredHistoryCheckpointStaysDisposable", testRestoredHistoryCheckpointStaysDisposable},
      {"testSharedReplayPointCountsEachRequest", testSharedReplayPointCountsEachRequest},
      {"testRestoredEndpointIsInUse", testRestoredEndpointIsInUse},
      {"testReplayPointRecyclesAnOlderPointInUse", testReplayPointRecyclesAnOlderPointInUse},
      {"testHeldBackStartTakesNothingInUse", testHeldBackStartTakesNothingInUse},
      {"testLaneThatWouldYieldTakesNothingInUse", testLaneThatWouldYieldTakesNothingInUse},
      {"testPacedPassKeepsInServiceReplayPoint", testPacedPassKeepsInServiceReplayPoint},
      {"testReplayPointTakesTheExpertStoreBeforeWhatIsInUse", testReplayPointTakesTheExpertStoreBeforeWhatIsInUse},
      {"testReplayPointCopiesCacheBeforeTheExpertStore", testReplayPointCopiesCacheBeforeTheExpertStore},
      {"testALoneReplayPointThatCannotBeMetTakesNothing", testALoneReplayPointThatCannotBeMetTakesNothing},
      {"testAHostRefusedReplayPointCutsNoCommand", testAHostRefusedReplayPointCutsNoCommand},
      {"testCopyAheadOfALeafHoldsARefusalBack", testCopyAheadOfALeafHoldsARefusalBack},
      {"testLoneGrowthWaitsForCopiesAheadAboveTheLeaves", testLoneGrowthWaitsForCopiesAheadAboveTheLeaves},
      {"testLoneRestoreTakesTheBuffersALaneStartsFrom", testLoneRestoreTakesTheBuffersALaneStartsFrom},
      {"testCaptureKvPastABoundaryTakesNothingInUse", testCaptureKvPastABoundaryTakesNothingInUse},
      {"testCaptureKvPastTheMarginCutsNoCommand", testCaptureKvPastTheMarginCutsNoCommand},
      {"testReplayPointAdmittedInMarginBand", testReplayPointAdmittedInMarginBand},
      {"testEveryEndReleasesTheReplayPoint", testEveryEndReleasesTheReplayPoint},
      {"testWaitingEndsReleaseTheReplayPoint", testWaitingEndsReleaseTheReplayPoint},
      {"testRestoringLaneWaitsForResidentLanes", testRestoringLaneWaitsForResidentLanes},
      {"testRestoreCompletesWhileAConstrainedLaneDecodes", testRestoreCompletesWhileAConstrainedLaneDecodes},
      {"testWaitWithProgressOutlivesTheResourceLimit", testWaitWithProgressOutlivesTheResourceLimit},
      {"testRolledBackAdmissionKeepsItsWaitLimit", testRolledBackAdmissionKeepsItsWaitLimit},
      {"testLimitOutlivedByProgressDoesNotWakeTheLoop", testLimitOutlivedByProgressDoesNotWakeTheLoop},
      {"testLaneAdmittedBeforeASuspensionHoldsTheWaitOpen", testLaneAdmittedBeforeASuspensionHoldsTheWaitOpen},
      {"testPagesReturnFromDemotionWithoutSuspending", testPagesReturnFromDemotionWithoutSuspending},
      {"testRefusedRestoreClosesAdmission", testRefusedRestoreClosesAdmission},
      {"testAHigherPriorityThatWaitsLeavesACommandItsPasses", testAHigherPriorityThatWaitsLeavesACommandItsPasses},
      {"testPublicationInvariantFailureIsFatal", testPublicationInvariantFailureIsFatal},
      {"testRefusedBuffersCostOnlyThatAttempt", testRefusedBuffersCostOnlyThatAttempt},
      {"testRefusedBuffersRecycleLruStateAndRetry", testRefusedBuffersRecycleLruStateAndRetry},
      {"testRecycledStateCountsOnlyForAPublishedCapture", testRecycledStateCountsOnlyForAPublishedCapture},
      {"testPersistentRefusalRecyclesAtMostOneState", testPersistentRefusalRecyclesAtMostOneState},
      {"testLongSuffixSkipsDraftRestore", testLongSuffixSkipsDraftRestore},
      {"testCancellationInFlightAtBoundaryPublishesNoState", testCancellationInFlightAtBoundaryPublishesNoState},
      {"testLaneStateGrowthReclaimsCachedStateAndRetries", testLaneStateGrowthReclaimsCachedStateAndRetries},
      {"testKvGrowthReclaimsIdleStateBeforeCache", testKvGrowthReclaimsIdleStateBeforeCache},
      {"testKvGrowthKeepsTheRingAWaitingLaneStartsFrom", testKvGrowthKeepsTheRingAWaitingLaneStartsFrom},
      {"testKvGrowthDenialKeepsEveryLaneReplayState", testKvGrowthDenialKeepsEveryLaneReplayState},
      {"testBoundaryStatesAreCapturedInsideCommands", testBoundaryStatesAreCapturedInsideCommands},
      {"testRefusedBuffersEndTheCommandAtTheirBoundary", testRefusedBuffersEndTheCommandAtTheirBoundary},
      {"testRefusedKvPastABoundaryEndsTheCommandThere", testRefusedKvPastABoundaryEndsTheCommandThere},
      {"testPassedCheckpointLeavesTheJunctionItsOwnClass", testPassedCheckpointLeavesTheJunctionItsOwnClass},
      {"testRefusedKvPastAPassedCheckpointCapturesIt", testRefusedKvPastAPassedCheckpointCapturesIt},
      {"testNonFiniteCommandDropsItsCaptures", testNonFiniteCommandDropsItsCaptures},
      {"testDuplicateCaptureIsDropped", testDuplicateCaptureIsDropped},
      {"testOneCommandCapturesSeveralLanesStates", testOneCommandCapturesSeveralLanesStates},
      {"testFullLanesSkipAdmissionAttempts", testFullLanesSkipAdmissionAttempts},
      {"testConcurrencyLimitDoesNotEvictCache", testConcurrencyLimitDoesNotEvictCache},
      {"testHostPressureStillRecyclesLruStateForRefusedBuffers",
       testHostPressureStillRecyclesLruStateForRefusedBuffers},
      {"testRequestInServiceGrowsThroughTheHostPause", testRequestInServiceGrowsThroughTheHostPause},
      {"testAHostRefusedGrowthTakesOnlyWhatFreesNow", testAHostRefusedGrowthTakesOnlyWhatFreesNow},
      {"testFirstRequestStartsThroughTheHostPause", testFirstRequestStartsThroughTheHostPause},
      {"testRequestThatNeedsNoGrowthStartsUnderTheHostPause", testRequestThatNeedsNoGrowthStartsUnderTheHostPause},
      {"testSuspendedRequestWaitsForTheHostBesideOneInService", testSuspendedRequestWaitsForTheHostBesideOneInService},
      {"testEngineLimitBindsThroughTheHostPause", testEngineLimitBindsThroughTheHostPause},
      {"testPausedPageShortageKeepsCachedStates", testPausedPageShortageKeepsCachedStates},
      {"testSingletonHostPressureWaitRecoversOrTerminates", testSingletonHostPressureWaitRecoversOrTerminates},
      {"testAdmissionWaitsOutEarlierLanes", testAdmissionWaitsOutEarlierLanes},
      {"testStartsAdmittedBesideALaneInServiceDoNotExtendItsWait",
       testStartsAdmittedBesideALaneInServiceDoNotExtendItsWait},
      {"testLaneAdmittedBeforeARefusalHoldsTheWaitOpen", testLaneAdmittedBeforeARefusalHoldsTheWaitOpen},
      {"testStartsAdmittedAheadOfARefusedStartHoldItsWaitOpen",
       testStartsAdmittedAheadOfARefusedStartHoldItsWaitOpen},
      {"testKvPressureNarrowsTheRealBatch", testKvPressureNarrowsTheRealBatch},
      {"testKvGrowthReclaimsCachedStateWhenBudgetIsShared", testKvGrowthReclaimsCachedStateWhenBudgetIsShared},
      {"testWhatFreesNowKeepsALaneWaitingForTheRest", testWhatFreesNowKeepsALaneWaitingForTheRest},
      {"testRequiredWorkDoesNotReserveAnExtraPage", testRequiredWorkDoesNotReserveAnExtraPage},
      {"testAdmissionPinsDesiredStateAndCountsOnlySuccess", testAdmissionPinsDesiredStateAndCountsOnlySuccess},
      {"testAdmissionCanDropItsOwnCachePinToMakeProgress", testAdmissionCanDropItsOwnCachePinToMakeProgress},
      {"testActivationReceivesTheRestoreBoundary", testActivationReceivesTheRestoreBoundary},
      {"testRefusedStartKeepsItsLeaseBesideAResidentLane", testRefusedStartKeepsItsLeaseBesideAResidentLane},
      {"testRefusedStartKeepsItsLeaseWhileMemoryIsPending", testRefusedStartKeepsItsLeaseWhileMemoryIsPending},
      {"testDroppedLeaseLooksUpAgain", testDroppedLeaseLooksUpAgain},
      {"testHostRefusedStartKeepsItsLease", testHostRefusedStartKeepsItsLease},
      {"testSingletonCapacityFailureTerminatesCleanly", testSingletonCapacityFailureTerminatesCleanly},
      {"testQueuedLongPrefillsLeaveRoomForShortWork", testQueuedLongPrefillsLeaveRoomForShortWork},
      {"testAdmissionUsesCachedRemainingWork", testAdmissionUsesCachedRemainingWork},
      {"testUnselectableCandidatesAreNotProbed", testUnselectableCandidatesAreNotProbed},
      {"testWaitingCandidateProbeIsRefreshedNotRepeated", testWaitingCandidateProbeIsRefreshedNotRepeated},
      {"testMemoryWaitHoldsBackLaterArrivals", testMemoryWaitHoldsBackLaterArrivals},
      {"testARefusedStartHoldsOnlyWhatItLeads", testARefusedStartHoldsOnlyWhatItLeads},
      {"testAHeldStartProceedsOnceTheRefusedOneIsCancelled", testAHeldStartProceedsOnceTheRefusedOneIsCancelled},
      {"testAgingBoundsHowFarShorterArrivalsPassARefusedStart", testAgingBoundsHowFarShorterArrivalsPassARefusedStart},
      {"testARefusedRequestWaitingForAPrefixHoldsBackNothing", testARefusedRequestWaitingForAPrefixHoldsBackNothing},
      {"testRefusedResumeHoldsBackLaterSuspendedLanes", testRefusedResumeHoldsBackLaterSuspendedLanes},
      {"testRecoveryAdmitsOneRequestAPass", testRecoveryAdmitsOneRequestAPass},
      {"testHeldSuspendedRequestNeitherWakesNorExpires", testHeldSuspendedRequestNeitherWakesNorExpires},
      {"testStatusCountsRequestsHeldBehindARefusal", testStatusCountsRequestsHeldBehindARefusal},
      {"testHeldRequestsCountWhileTheirRefusalStands", testHeldRequestsCountWhileTheirRefusalStands},
      {"testStatusCountsSuspendedRequestsHeldDuringRecovery", testStatusCountsSuspendedRequestsHeldDuringRecovery},
      {"testSchedulingWaitDoesNotConsumeMemoryTimeout", testSchedulingWaitDoesNotConsumeMemoryTimeout},
      {"testUnadmittedRequestsHonorCancellationAndDeadline", testUnadmittedRequestsHonorCancellationAndDeadline},
      {"testGrowthKeepsPrefillProgressWhenAnUnstartedPeerCanYield",
       testGrowthKeepsPrefillProgressWhenAnUnstartedPeerCanYield},
      {"testGrowthYieldsLowerPriorityResidentOutsideBatch", testGrowthYieldsLowerPriorityResidentOutsideBatch},
      {"testPrefillGrowthPreservesAnActiveDecodePeer", testPrefillGrowthPreservesAnActiveDecodePeer},
      {"testARefusedDecodeLeavesThePromptOutsideItsCommand", testARefusedDecodeLeavesThePromptOutsideItsCommand},
      {"testKvPressureSuspendsInsteadOfKillingActiveWork", testKvPressureSuspendsInsteadOfKillingActiveWork},
      {"testRecoveryDrainDoesNotConsumeResourceWaitBudget", testRecoveryDrainDoesNotConsumeResourceWaitBudget},
      {"testRecoveryDrainEndsWithItsCause", testRecoveryDrainEndsWithItsCause},
      {"testRecoveryDrainEndsWhenAResidentReleasesMemory", testRecoveryDrainEndsWhenAResidentReleasesMemory},
      {"testRefusedStateDoesNotHoldTheDrain", testRefusedStateDoesNotHoldTheDrain},
      {"testHigherPriorityArrivalIsNotHeldByRecovery", testHigherPriorityArrivalIsNotHeldByRecovery},
      {"testForegroundArrivalPreemptsWhenLanesAreFull", testForegroundArrivalPreemptsWhenLanesAreFull},
      {"testPrefixWaiterTakesNoLaneUntilThePrefixLands", testPrefixWaiterTakesNoLaneUntilThePrefixLands},
      {"testRequestBelowARunningPriorityTakesNoLane", testRequestBelowARunningPriorityTakesNoLane},
      {"testPressureRetryIsBackedOffWithoutProgress", testPressureRetryIsBackedOffWithoutProgress},
      {"testAdmissionRetryWakesOnlyWhenTickCanRetry", testAdmissionRetryWakesOnlyWhenTickCanRetry},
      {"testDecodePreemptionReplaysCommittedHistoryWithoutRepeatingOutput",
       testDecodePreemptionReplaysCommittedHistoryWithoutRepeatingOutput},
      {"testLongDecodePreemptionPlansTheCurrentReplayBoundary", testLongDecodePreemptionPlansTheCurrentReplayBoundary},
      {"testResumedProducerPlansJunctionsForSiblings", testResumedProducerPlansJunctionsForSiblings},
      {"testResumedLaneRebuildsItsPointFromCachedKv(1)", [] { testResumedLaneRebuildsItsPointFromCachedKv(1); }},
      {"testResumedLaneRebuildsItsPointFromCachedKv(2)", [] { testResumedLaneRebuildsItsPointFromCachedKv(2); }},
      {"testRefusedBuffersTakeAtMostOneStateOfExtents", testRefusedBuffersTakeAtMostOneStateOfExtents},
      {"testRefusedStateAsksAgainOnceAnExtentLeft", testRefusedStateAsksAgainOnceAnExtentLeft},
      {"testPausedPublicationInUseTakesNoKv", testPausedPublicationInUseTakesNoKv},
      {"testPreemptedDecodeRestoresItsResidentCompositeState", testPreemptedDecodeRestoresItsResidentCompositeState},
      {"testPreemptedDecodeReplayBoundaryIgnoresTheGenerationPrompt",
       testPreemptedDecodeReplayBoundaryIgnoresTheGenerationPrompt},
      {"testRepeatedPreemptionRespectsBackoffAndCancellation", testRepeatedPreemptionRespectsBackoffAndCancellation},
      {"testAdmissionReopensAfterLastSuspendedRequestResumes", testAdmissionReopensAfterLastSuspendedRequestResumes},
      {"testRecoveryAdmitsFailedKvTargetBeforeReplaying", testRecoveryAdmitsFailedKvTargetBeforeReplaying},
      {"testFailedResumeRestoreKeepsTheKvTarget", testFailedResumeRestoreKeepsTheKvTarget},
      {"testBudgetDenialRetriesAfterRelease", testBudgetDenialRetriesAfterRelease},
      {"testAStartKeepsThePooledLaneBuffersItsKvTakesThemLast", testAStartKeepsThePooledLaneBuffersItsKvTakesThemLast},
      {"testPausedStateAdmissionReusesCachedStates", testPausedStateAdmissionReusesCachedStates},
      {"testDeniedGrowthAllocatesEachExtentOnce", testDeniedGrowthAllocatesEachExtentOnce},
      {"testControlPassRefusesACommandInFlight", testControlPassRefusesACommandInFlight},
      {"testStateStartGathersFreePagesBeforeEvicting", testStateStartGathersFreePagesBeforeEvicting},
      {"testCommandReadsItsPageTableAfterRoomForItsStates", testCommandReadsItsPageTableAfterRoomForItsStates},
      {"testCompactionWaitsForTheCommandInFlight", testCompactionWaitsForTheCommandInFlight},
      {"testWindowTrimChangesNoOutput", testWindowTrimChangesNoOutput},
      {"testWindowTrimServesThroughTheHostPause", testWindowTrimServesThroughTheHostPause},
      {"testTrimsAtTheSlimRunwaysLastPagesAllocateNothing", testTrimsAtTheSlimRunwaysLastPagesAllocateNothing},
      {"testAllocationCausesRemainDistinct", testAllocationCausesRemainDistinct},
      {"testAdmissionRespectsPriorityBeforeHashOrder", testAdmissionRespectsPriorityBeforeHashOrder},
      {"testMemoryNotWidthTriggersPriorityPreemption", testMemoryNotWidthTriggersPriorityPreemption},
      {"testSmallShortfallFreesAtMostOneStoreUnitMore", testSmallShortfallFreesAtMostOneStoreUnitMore},
      {"testKvChainAndItsCheckpointLeaveTogether", testKvChainAndItsCheckpointLeaveTogether},
  };
  return kTests;
}
