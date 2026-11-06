#include "EngineHarness.hpp"
#include "EngineTestSupport.hpp"
#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestRegistry.hpp"
#include "benchmarks/PrefillWork.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

using namespace splash;
using namespace splash::engine;
using benchmark::draftContextRows;
using splash::test::capturedStates;
using splash::test::Events;
using splash::test::GovernedHost;
using splash::test::Executor;
using splash::test::guardReleases;
using splash::test::idle;
using splash::test::OffloadControl;
using splash::test::request;
using splash::test::State;

namespace {

using splash::test::require;

// The harness ticked at a test's instant: its clock moves there first and
// never back (a test written for 0 ticks at the harness's start).
bool tickAt(test::EngineHarness &h, double now) {
  if (now > h.now())
    h.advanceTo(now);
  return h.tick();
}

// Ticks on from `now` until done() holds, as the harness runs
// (EngineHarness::runUntil); `now` follows its clock.
void tickUntil(test::EngineHarness &h, double &now, const std::function<bool()> &done, const char *message) {
  if (now > h.now())
    h.advanceTo(now);
  h.runUntil(done, message, 1000);
  now = h.now() + 1;
}

// Ticks as the harness runs until the engine has published `publications`
// checkpoints.
void runUntilCheckpoint(test::EngineHarness &h, uint64_t publications) {
  h.runUntil([&] { return h.engine.snapshot().checkpointPublications >= publications; },
             "engine did not publish the expected checkpoint");
}

void testConcurrentProgressRetainsAtMostOnePointPerLane() {
  test::EngineHarness h({.pages = 2048, .shape = [](Executor &shape) { shape.maximumLanes = 2; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> prompt(25001, 20);
  engine.submit(request(410, prompt));
  engine.submit(request(411, prompt));
  uint32_t maximumEntries = 0;
  for (uint32_t step = 0; step < 128 && !idle(engine); ++step) {
    static_cast<void>(tickAt(h, step + 1));
    maximumEntries =
        std::max(maximumEntries, resources.snapshot().stateCache.entries);
  }
  // 24576 lies within one prefill chunk of the replay boundary at 24992; the
  // last checkpoint, at 20480, stays beside the replay state.
  const uint32_t checkpoints = 5;
  require(idle(engine) && events.completedCount == 2 &&
              engine.snapshot().checkpointPublications >= checkpoints &&
              engine.snapshot().checkpointPublications <= 2 * checkpoints &&
              engine.snapshot().cacheHits == 1 &&
              executor.prefillRows == prompt.size() * 2 - 24992 &&
              maximumEntries <= 2 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "concurrent prompts accumulated progress states beyond their active lanes");
}

void testSharedCheckpointSurvivesPeerRollingReplacement() {
  test::EngineHarness h({.pages = 2048, .shape = [](Executor &shape) { shape.maximumLanes = 2; }});
  engine::Cache &resources = h.cache;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  std::vector<uint32_t> original(25001, 20);
  auto background = request(412, original);
  background.priority = RequestPriority::Background;
  engine.submit(std::move(background));
  runUntilCheckpoint(h, 1);
  std::vector<uint32_t> branch = original;
  std::fill(branch.begin() + kPrefillCheckpointTokens, branch.end(), 21);
  auto foreground = request(413, branch);
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  runUntilCheckpoint(h, 2);
  require(resources.lookup(original, {}).resumeBoundary() == kPrefillCheckpointTokens,
          "rolling a shared checkpoint retired the paused peer's recovery point");
  engine.cancel(412);
  engine.cancel(413);
  h.runUntilIdle();
  require(events.failedCount == 0, "shared checkpoint cancellation failed");
}

void testRepeatedRetriesRollTheRestoredCheckpoint() {
  test::EngineHarness h({.pages = 2048, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> prompt(40001, 31);
  for (uint32_t attempt = 0; attempt < 3; ++attempt) {
    engine.submit(request(500 + attempt, prompt));
    runUntilCheckpoint(h, attempt + 1);
    require(events.starts.back() == attempt * kPrefillCheckpointTokens &&
                resources.snapshot().stateCache.entries == 1 &&
                resources.snapshot().stateCache.checkpointEntries == 1,
            "retry promoted or accumulated intermediate checkpoints");
    engine.cancel(500 + attempt);
    h.runUntilIdle();
  }
  engine.submit(request(503, prompt));
  h.runUntilIdle();
  // Of its checkpoints only the last, at 36864, stays beside its replay state.
  require(events.starts.back() == 3 * kPrefillCheckpointTokens &&
              executor.prefillRows == prompt.size() &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1 &&
              resources.lookup(std::vector<uint32_t>(36865, 31), {}).resumeBoundary() == 36864,
          "successful retry recomputed or retained superseded recovery states");
}

void testRetryCancelledBeforeNextCheckpointKeepsItsSource() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> prompt(25001, 32);
  engine.submit(request(510, prompt));
  runUntilCheckpoint(h, 1);
  engine.cancel(510);
  h.runUntilIdle();
  for (uint32_t id : {511U, 512U}) {
    engine.submit(request(id, prompt));
    require(tickAt(h, 1) && engine.commandInFlight() &&
                events.starts.back() == kPrefillCheckpointTokens,
            "retry did not restore the previous progress point");
    engine.cancel(id);
    h.runUntilIdle();
    require(resources.lookup(prompt, {}).resumeBoundary() ==
                    kPrefillCheckpointTokens &&
                resources.snapshot().stateCache.checkpointEntries == 1,
            "cancelled retry lost or promoted its unchanged recovery point");
  }
}

// A restored endpoint becomes ordinary where it is, on disk too when no
// cache slot takes its RAM copy, and its loss is then a lost state.
void testRestoredCheckpointAtReplayEndBecomesOrdinary() {
  for (bool disk : {false, true}) {
    for (uint32_t suffix : {1U, 31U}) {
      test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
      engine::Cache &resources = h.cache;
      Executor &executor = h.model;
      Events &events = h.events;
      engine::Engine &engine = h.engine;
      if (disk) {
        executor.deniedReservations = 1000;
        executor.stateTier = std::make_shared<OffloadControl>();
        executor.stateTier->ready = true;
        executor.restoreControl->ready = true;
        executor.restoreControl->promotionDenied = true;
      }
      const std::vector<uint32_t> prompt(25001, 33);
      engine.submit(request(520, prompt));
      runUntilCheckpoint(h, 1);
      engine.cancel(520);
      h.runUntilIdle();
      const uint32_t taken = executor.reservations + executor.diskSnapshots;
      std::vector<uint32_t> shorter(
          prompt.begin(), prompt.begin() + kPrefillCheckpointTokens + suffix);
      engine.submit(request(521, shorter));
      h.runUntilIdle();
      const auto states = resources.snapshot().stateCache;
      require(events.starts.back() == kPrefillCheckpointTokens &&
                  executor.reservations + executor.diskSnapshots == taken &&
                  engine.snapshot().deduplicatedStatePublications == 1 &&
                  states.entries == 1 && states.checkpointEntries == 0 &&
                  states.promotionsSkipped == (disk ? 1 : 0) &&
                  resources.lookup(shorter, {}).resumeBoundary() ==
                      kPrefillCheckpointTokens,
              "restored replay endpoint was copied or retired as temporary");
      auto held = resources.lookup(shorter, {});
      const uint64_t block = held.state->kvBlock();
      const CompositeState *copy = held.state->state().get();
      held = {};
      resources.discardState(block, copy);
      require(resources.lookup(shorter, {}).lostState,
              "the loss of a restored replay endpoint was not a lost state");
    }
  }
}

void testRetryRetiresCheckpointAtDeeperJunction() {
  test::EngineHarness h({.pages = 2048,
                         .engine = {.prefillCheckpointTokens = 8192},
                         .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> prompt(30001, 34);
  engine.submit(request(530, prompt));
  runUntilCheckpoint(h, 2);
  require(tickAt(h, 50) && tickAt(h, 51) && !engine.commandInFlight() &&
              executor.requests.at(530).position == 18432,
          "fixture did not commit past the last recovery point");
  engine.cancel(530);
  h.runUntilIdle();
  // Another conversation leaves the prompt at 18432 and goes on to a state
  // of its own, so the retry's matched KV ends at a branch point.
  std::vector<uint32_t> branch(prompt.begin(), prompt.begin() + 18432);
  branch.resize(18432 + KvCache::pageTokens, 35);
  resources.beginRequest(999);
  require(resources.ensureTokens(999, branch.size()).granted(), "branch fixture KV failed");
  resources.publishCompositeState(
      test::publishBlocks(resources, 999, branch, static_cast<uint32_t>(branch.size())),
      std::make_shared<State>(), false);
  resources.endRequest(999);
  engine.submit(request(531, prompt));
  require(tickAt(h, 1) && tickAt(h, 2) &&
              events.starts.back() == 16384 &&
              engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "retry kept an earlier recovery point after publishing its junction");
  // Past the junction it checkpoints at 24576, the last before its replay
  // boundary, which stays.
  h.runUntilIdle();
  require(resources.snapshot().stateCache.entries == 4 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "retry did not retain its junction, its replay state and its last checkpoint");
}

void testPinnedCheckpointSkipsReplacementButNotOrdinaryState() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> prompt(18001, 35);
  engine.submit(request(540, prompt));
  runUntilCheckpoint(h, 1);
  auto pinned = resources.lookup(prompt, {});
  require(pinned.resumeBoundary() == kPrefillCheckpointTokens,
          "fixture did not pin its checkpoint");
  h.runUntilIdle();
  // The checkpoints at 8192 and 12288 cannot retire the pinned one: their
  // commands capture them in pooled buffers, which go back unpublished.
  // 16384 lies within one prefill chunk of the replay boundary at 17984.
  require(engine.snapshot().checkpointPublications == 1 &&
              engine.snapshot().checkpointPublicationFailures == 2 &&
              executor.reservationAttempts == 4 && executor.captures.size() == 4 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1 &&
              resources.lookup(prompt, {}).resumeBoundary() == 17984,
          "pinned recovery point was overwritten or blocked ordinary publication");
  pinned = {};
  require(test::stateStep(resources).gave() &&
              resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.lookup(prompt, {}).resumeBoundary() == 17984,
          "released recovery pin did not rejoin the lower-priority queue");
}

// A checkpoint whose predecessor a restore pins is not published, so it
// takes buffers only from the pool or an admission: refused them, it
// recycles no other lane's checkpoint, and its command ends there.
void testPinnedPredecessorTakesNoRoomFromOtherStates() {
  test::EngineHarness h({.pages = 2048, .shape = [](Executor &shape) { shape.maximumLanes = 2; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  engine::Engine &engine = h.engine;
  // A cancelled prefill leaves its checkpoint at 4096 cached.
  const std::vector<uint32_t> other(25001, 41);
  engine.submit(request(1, other));
  runUntilCheckpoint(h, 1);
  engine.cancel(1);
  h.runUntilIdle();
  const std::vector<uint32_t> prompt(18001, 42);
  engine.submit(request(2, prompt));
  runUntilCheckpoint(h, 2);
  auto pinned = resources.lookup(prompt, {});
  require(pinned.resumeBoundary() == kPrefillCheckpointTokens &&
              resources.lookup(other, {}).resumeBoundary() == kPrefillCheckpointTokens,
          "fixture did not pin its checkpoint beside the other one");
  // Memory is full: buffers come only from states the cache lets go.
  *executor.evictedStateBytes = 0;
  executor.reservationRoom = [&] {
    if (*executor.evictedStateBytes < 64)
      return false;
    *executor.evictedStateBytes -= 64;
    return true;
  };
  double now = 1000;
  tickUntil(h, now, [&] { return engine.snapshot().checkpointPublicationFailures == 2; },
            "the checkpoints after the pinned one did not fail");
  const auto counters = engine.snapshot();
  require(counters.checkpointPublications == 2 &&
              counters.resources.stateCache.checkpointEvictions == 0 &&
              resources.lookup(other, {}).resumeBoundary() == kPrefillCheckpointTokens,
          "a checkpoint that could not be published recycled another lane's");
  pinned = {};
  engine.cancel(2);
  h.runUntilIdle();
}

// A checkpoint whose predecessor a reader pins takes no room before its command:
// unpinned while the command runs, it takes the predecessor's buffers after it;
// still pinned, it is not published.
void testPinnedPredecessorReleasedDuringTheCommand() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  bool memoryFull = false;
  executor.reservationRoom = [&] {
    if (*executor.evictedStateBytes >= 64) {
      *executor.evictedStateBytes -= 64;
      return true;
    }
    return !memoryFull;
  };
  // Checkpoints at 4096 and 8192: the command [6144, 8192) ends at the
  // second.
  const std::vector<uint32_t> prompt(3 * kPrefillCheckpointTokens + 1, 43);
  engine.submit(request(1, prompt));
  double now = 1;
  tickUntil(h, now,
            [&] {
              return executor.requests.contains(1) &&
                     executor.requests.at(1).position == 6144 && !engine.commandInFlight();
            },
            "the request did not reach the command before its next checkpoint");
  CacheLookup pinned = resources.lookup(prompt, {});
  require(pinned.resumeBoundary() == kPrefillCheckpointTokens,
          "fixture did not pin the lane's checkpoint");
  // Memory is full: buffers come only from a state the cache lets go.
  memoryFull = true;
  *executor.evictedStateBytes = 0;
  executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
  require(tickAt(h, now++) && engine.commandInFlight() &&
              executor.requests.at(1).position == 2 * kPrefillCheckpointTokens,
          "the command did not run to the next checkpoint");
  pinned = {};
  *executor.holdPrefillUntil = true;
  tickUntil(h, now, [&] { return !engine.commandInFlight(); },
            "the command did not complete");
  const auto counters = engine.snapshot();
  require(counters.checkpointPublications == 2 && counters.checkpointPublicationFailures == 0 &&
              executor.snapshots == 1 &&
              counters.resources.stateCache.checkpointRetirements == 1 &&
              counters.resources.stateCache.evictions == 0 &&
              resources.probe(prompt, {}).cachedTokens() == 2 * kPrefillCheckpointTokens,
          "the checkpoint was not taken in its predecessor's buffers once the pin went");
  executor.holdPrefillUntil.reset();
  memoryFull = false;
  tickUntil(h, now, [&] { return idle(engine); }, "the request did not finish");
  require(events.completedCount == 1 && events.failedCount == 0, "the request failed");
}

void testFailedReplacementContinuesWithoutRecoveryPoint() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> prompt(25001, 36);
  engine.submit(request(550, prompt));
  runUntilCheckpoint(h, 1);
  executor.denyReservationAt = 2 * kPrefillCheckpointTokens;
  for (uint32_t step = 0; step < 32 &&
       !engine.snapshot().checkpointPublicationFailures; ++step)
    static_cast<void>(tickAt(h, step + 1));
  require(engine.snapshot().checkpointPublicationFailures == 1 &&
              resources.snapshot().stateCache.entries == 0 &&
              events.failedCount == 0,
          "denied replacement kept the retired checkpoint or failed inference");
  // The checkpoints from 12288 on land; the last, at 20480, stays.
  h.runUntilIdle();
  require(events.completedCount == 1 && executor.prefillRows == prompt.size() &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "inference did not recover from a skipped checkpoint publication");
}

void testRollingHandleCannotRetirePromotedState() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> prompt(25001, 37);
  engine.submit(request(560, prompt));
  runUntilCheckpoint(h, 1);
  {
    auto existing = resources.lookup(prompt, {});
    require(resources.reuseCompositeState(existing.state->kvBlock(), false),
            "shared ordinary boundary could not reuse its checkpoint");
  }
  h.runUntilIdle();
  const std::vector<uint32_t> prefix(
      prompt.begin(), prompt.begin() + kPrefillCheckpointTokens + 1);
  // Beside the promoted state: the replay state and the last checkpoint.
  require(resources.snapshot().stateCache.entries == 3 &&
              resources.snapshot().stateCache.checkpointEntries == 1 &&
              resources.lookup(prefix, {}).resumeBoundary() ==
                  kPrefillCheckpointTokens,
          "old rolling handle deleted a state promoted by another request");
}

// With room for it, the next checkpoint takes buffers of its own before its
// command, beside the lane's checkpoint, and its command captures it; the
// lane's checkpoint retires once the next one is published.
void testCheckpointRetiresOnceItsSuccessorIsPublished() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  engine::Engine &engine = h.engine;
  engine.submit(request(430, std::vector<uint32_t>(25001, 23)));
  runUntilCheckpoint(h, 1);
  executor.reservationObserver = [&] {
    require(resources.snapshot().stateCache.checkpointEntries == 1 &&
                resources.snapshot().stateCache.checkpointRetirements == 0,
            "the next checkpoint's buffers retired the lane's checkpoint before its command");
  };
  runUntilCheckpoint(h, 2);
  require(capturedStates(executor, 430) ==
                  std::vector<std::pair<uint32_t, uint64_t>>{{4096, 4096}, {8192, 8192}} &&
              executor.snapshots == 0 && engine.snapshot().captureCuts == 0,
          "the next checkpoint was not captured in its command");
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.evictions == 0 &&
              resources.snapshot().stateCache.checkpointRetirements == 1 &&
              engine.snapshot().checkpointPublicationFailures == 0,
          "the lane's checkpoint did not retire once the next one was published");
  engine.cancel(430);
  h.runUntilIdle();
}

// A command that captures the lane's next checkpoint and is cancelled, expires or
// fails publishes nothing and leaves the lane its checkpoint, even with memory
// full; a retry resumes there.
void testInterruptedCommandKeepsTheLanesCheckpoint() {
  enum class End { Cancelled, Expired, Failed };
  for (const bool full : {false, true}) {
    for (const End end : {End::Cancelled, End::Expired, End::Failed}) {
      test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
      engine::Cache &resources = h.cache;
      Executor &executor = h.model;
      Events &events = h.events;
      engine::Engine &engine = h.engine;
      bool memoryFull = false;
      executor.reservationRoom = [&] {
        if (*executor.evictedStateBytes >= 64) {
          *executor.evictedStateBytes -= 64;
          return true;
        }
        return !memoryFull;
      };
      // Checkpoints at 4096, 8192 and 12288: the command [6144, 8192)
      // captures the one at 8192.
      const std::vector<uint32_t> prompt(18001, 24);
      auto first = request(1, prompt);
      first.deadlineMilliseconds = 1'000;
      engine.submit(std::move(first));
      double now = 1;
      tickUntil(h, now,
                [&] {
                  return executor.requests.contains(1) &&
                         executor.requests.at(1).position == 6144 && !engine.commandInFlight();
                },
                "the request did not reach the command before its next checkpoint");
      memoryFull = full;
      if (full)
        *executor.evictedStateBytes = 0;
      executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
      if (end == End::Failed)
        executor.nonFinitePrefills.insert(1);
      require(tickAt(h, now++) && engine.commandInFlight() &&
                  engine.snapshot().checkpointPublications == 1 &&
                  resources.snapshot().stateCache.checkpointEntries == 1,
              "the command took the lane's checkpoint before it ran");
      if (end == End::Cancelled)
        engine.cancel(1);
      if (end == End::Expired)
        static_cast<void>(tickAt(h, now = 1'000));
      *executor.holdPrefillUntil = true;
      tickUntil(h, now, [&] { return idle(engine); }, "the request did not end");
      require(end == End::Cancelled
                  ? engine.snapshot().cancelled == 1
                  : events.failures == std::vector<std::string>{end == End::Expired
                                                                    ? "deadline_exceeded"
                                                                    : "model_result_invalid"},
              "the request did not end as its command did");
      require(engine.snapshot().checkpointPublications == 1 &&
                  resources.snapshot().stateCache.checkpointRetirements == 0 &&
                  resources.snapshot().stateCache.evictions == 0 &&
                  resources.lookup(prompt, {}).resumeBoundary() == 4096,
              "an interrupted command cost the lane its checkpoint");
      executor.holdPrefillUntil.reset();
      executor.nonFinitePrefills.clear();
      const uint32_t completed = events.completedCount;
      engine.submit(request(2, prompt));
      tickUntil(h, now, [&] { return idle(engine); }, "the retry did not finish");
      require(events.starts.back() == 4096 && events.completedCount == completed + 1,
              "the retry did not resume from the lane's checkpoint");
    }
  }
}

// With memory full, a junction, checkpoint or replay point refused buffers before
// its command takes no room a reclaim could make of the lane's checkpoint: the
// command ends at its boundary and the state then takes the checkpoint's buffers.
// No other state goes, and a cancel there leaves the checkpoint.
void testRefusedStateTakesTheCheckpointAfterItsCommand() {
  enum class Kind { Junction, Checkpoint, ReplayPoint };
  for (const Kind kind : {Kind::Junction, Kind::Checkpoint, Kind::ReplayPoint}) {
    for (const bool cancelled : {false, true}) {
      test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
      engine::Cache &resources = h.cache;
      Executor &executor = h.model;
      Events &events = h.events;
      engine::Engine &engine = h.engine;
      bool memoryFull = false;
      executor.reservationRoom = [&] {
        if (*executor.evictedStateBytes >= 64) {
          *executor.evictedStateBytes -= 64;
          return true;
        }
        return !memoryFull;
      };
      // States a reclaim could take: another conversation's checkpoint, the
      // state the prompt resumes from at 64 and, for the junction, the state
      // of the conversation that leaves the prompt at 6016.
      const std::vector<uint32_t> other(25001, 41);
      engine.submit(request(1, other));
      runUntilCheckpoint(h, 1);
      engine.cancel(1);
      h.runUntilIdle();
      // Resuming at 64, the prompt's commands end at 2112, 4160, 6208 and
      // 8256: the checkpoint at 4096 lies in the second, and the junction at
      // 6016, the checkpoint at 8192 or the replay point at 6976 in the next.
      const uint32_t length =
          kind == Kind::Junction ? 8001 : kind == Kind::Checkpoint ? 12001 : 7001;
      const uint32_t boundary =
          kind == Kind::Junction ? 6016 : kind == Kind::Checkpoint ? 8192 : 6976;
      const uint32_t start = kind == Kind::Junction ? 4160 : 6208;
      std::vector<uint32_t> prompt(length);
      std::iota(prompt.begin(), prompt.end(), 1);
      const std::vector<uint32_t> resumed(prompt.begin(), prompt.begin() + 65);
      engine.submit(request(2, resumed));
      h.runUntilIdle();
      std::vector<uint32_t> branch(prompt.begin(), prompt.begin() + 6016);
      branch.resize(6016 + KvCache::pageTokens, 0);
      if (kind == Kind::Junction) {
        resources.beginRequest(999);
        require(resources.ensureTokens(999, branch.size()).granted(),
                "junction fixture could not allocate its KV prefix");
        resources.publishCompositeState(
            test::publishBlocks(resources, 999, branch, static_cast<uint32_t>(branch.size())),
            std::make_shared<State>(), false);
        resources.endRequest(999);
      }
      engine.submit(request(3, prompt));
      double now = 100;
      tickUntil(h, now,
                [&] {
                  return executor.requests.contains(3) &&
                         executor.requests.at(3).position == start &&
                         !engine.commandInFlight();
                },
                "the prompt did not reach the command of its state");
      require(events.starts.back() == 64 &&
                  capturedStates(executor, 3) ==
                      std::vector<std::pair<uint32_t, uint64_t>>{{4096, 4160}} &&
                  resources.probe(prompt, {}).cachedTokens() == 4096,
              "fixture did not leave the lane its checkpoint at 4096");
      // Memory fills: the idle buffers in the pool went back to the host.
      memoryFull = true;
      *executor.evictedStateBytes = 0;
      if (cancelled)
        executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
      require(tickAt(h, now++) && executor.requests.at(3).position == boundary &&
                  engine.snapshot().captureCuts == 1 &&
                  capturedStates(executor, 3).size() == 1 &&
                  resources.snapshot().stateCache.checkpointRetirements == 0 &&
                  resources.probe(prompt, {}).cachedTokens() == 4096,
              "the refused state did not end its command, or took the lane's checkpoint");
      std::vector<uint32_t> pastBranch = branch;
      pastBranch.push_back(0);
      const auto cached = [&] {
        return resources.probe(other, {}).cachedTokens() == 4096 &&
               resources.probe(resumed, {}).cachedTokens() == 64 &&
               (kind != Kind::Junction ||
                resources.probe(pastBranch, {}).cachedTokens() == branch.size()) &&
               resources.snapshot().stateCache.evictions == 0;
      };
      if (cancelled) {
        engine.cancel(3);
        *executor.holdPrefillUntil = true;
        tickUntil(h, now, [&] { return idle(engine); }, "the request did not end");
        require(executor.snapshots == 0 &&
                    resources.snapshot().stateCache.checkpointRetirements == 0 &&
                    resources.probe(prompt, {}).cachedTokens() == 4096 && cached(),
                "a cancelled command cost the lane its checkpoint or another state");
        continue;
      }
      tickUntil(h, now, [&] { return !engine.commandInFlight(); },
                "the command did not complete");
      require(executor.snapshots == 1 &&
                  resources.snapshot().stateCache.checkpointRetirements == 1 &&
                  resources.probe(prompt, {}).cachedTokens() == boundary && cached(),
              "the state was not taken in the lane's checkpoint's buffers after its command");
      const auto counters = engine.snapshot();
      require((kind == Kind::Junction ? counters.junctionMaterializations
               : kind == Kind::Checkpoint ? counters.checkpointPublications - 2
                                          : counters.replayStatePublications - 1) == 1 &&
                  counters.recycledStatePublications == 0,
              "the state was not published");
      memoryFull = false;
      tickUntil(h, now, [&] { return idle(engine); }, "the request did not finish");
      // Reported complete: the cancelled conversation and both prompts.
      require(events.completedCount == 3 && events.failedCount == 0, "a request failed");
    }
  }
}

// A junction is an ordinary state: publishing one retires the lane's
// earlier progress point. Past it, the replay state finds no checkpoint of
// the lane's to keep.
void testJunctionRetiresEarlierProgressPoint() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> prompt(20001, 29);
  // Another conversation leaves the prompt at 18432 and goes on to a state
  // of its own: the request plans a junction at that branch point, past its
  // checkpoint at 16384.
  std::vector<uint32_t> branch(prompt.begin(), prompt.begin() + 18432);
  branch.resize(18432 + KvCache::pageTokens, 30);
  resources.beginRequest(470);
  require(resources.ensureTokens(470, branch.size()).granted(),
          "junction fixture could not allocate its KV prefix");
  resources.publishCompositeState(
      test::publishBlocks(resources, 470, branch, static_cast<uint32_t>(branch.size())),
      std::make_shared<State>(), false);
  resources.endRequest(470);
  engine.submit(request(471, prompt));
  double now = 1;
  tickUntil(h, now, [&] { return engine.snapshot().junctionMaterializations == 1; },
            "the request did not publish its junction");
  require(engine.snapshot().checkpointPublications == 4 &&
              resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.snapshot().stateCache.entries == 2,
          "the junction left a superseded progress checkpoint resident");
  h.runUntilIdle();
  require(resources.snapshot().stateCache.entries == 3 &&
              resources.lookup(prompt, {}).resumeBoundary() == 20000,
          "the request did not publish its replay state after the junction");
}

// Conversations sharing a long system prompt: the second resumes from the
// first's kept last checkpoint, pins a junction where they part, which retires
// that checkpoint, and the third resumes at the junction.
void testSecondConversationResumesFromTheLastCheckpoint() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  // A 12000-token system prompt, then 201 tokens of the conversation's own:
  // checkpoints at 4096 and 8192, the replay state at 12192.
  std::vector<uint32_t> system(12000);
  std::iota(system.begin(), system.end(), 1);
  const auto conversation = [&](uint32_t first) {
    std::vector<uint32_t> prompt = system;
    prompt.resize(system.size() + 201);
    std::iota(prompt.begin() + system.size(), prompt.end(), first);
    return prompt;
  };
  engine.submit(request(1, conversation(100'000)));
  h.runUntilIdle();
  auto counters = engine.snapshot();
  require(counters.checkpointPublications == 2 &&
              counters.resources.stateCache.checkpointRetirements == 1 &&
              counters.resources.stateCache.checkpointEntries == 1 &&
              counters.resources.stateCache.checkpointBytes == 64 &&
              counters.resources.stateCache.entries == 2,
          "the first conversation did not keep its last checkpoint");
  // Only the last stays: a prompt that leaves the system prompt before 8192
  // finds no state.
  std::vector<uint32_t> early(system.begin(), system.begin() + 6000);
  early.resize(6100, 7);
  require(resources.probe(early, {}).cachedTokens() == 0,
          "a checkpoint outlived the one that replaced it");

  const uint32_t computed = executor.prefillRows;
  engine.submit(request(2, conversation(200'000)));
  h.runUntilIdle();
  counters = engine.snapshot();
  require(events.starts.back() == 8192 && executor.restored == 8192 &&
              executor.prefillRows - computed == 12201 - 8192 &&
              counters.junctionMaterializations == 1,
          "the second conversation did not resume from the kept checkpoint");
  require(counters.resources.stateCache.checkpointEntries == 0 &&
              counters.resources.stateCache.checkpointRetirements == 2 &&
              counters.resources.stateCache.entries == 3,
          "the junction did not retire the checkpoint it supersedes");

  const uint32_t restored = executor.restored;
  engine.submit(request(3, conversation(300'000)));
  h.runUntilIdle();
  require(events.starts.back() == 12000 && executor.restored == restored + 12000 &&
              events.completedCount == 3 && events.failedCount == 0,
          "the third conversation did not resume from the junction");
}

// A prompt that shares less than one checkpoint interval with a cached one
// finds no checkpoint below where they part, and one shorter than an
// interval plans none: such a request starts from the start and keeps no
// checkpoint.
void testShortSharedPrefixFindsAndKeepsNoCheckpoint() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  // Checkpoints at 4096 and 8192; the one at 8192 stays.
  std::vector<uint32_t> cached(12201);
  std::iota(cached.begin(), cached.end(), 1);
  engine.submit(request(1, cached));
  h.runUntilIdle();
  std::vector<uint32_t> shorter(cached.begin(), cached.begin() + 3000);
  shorter.resize(4001, 7);
  engine.submit(request(2, shorter));
  h.runUntilIdle();
  const auto counters = engine.snapshot();
  require(events.starts.back() == 0 && counters.checkpointPublications == 2 &&
              counters.resources.stateCache.checkpointEntries == 1 &&
              events.completedCount == 2 && events.failedCount == 0,
          "a short prompt resumed from or kept a checkpoint");
}

// The checkpoint a prompt keeps is as disposable as any: reclaim takes it
// before an older ordinary state, and a later publication that finds no room
// recycles it rather than an ordinary state, and is published.
void testKeptCheckpointGoesFirstUnderPressure() {
  GovernedHost host;
  test::EngineHarness h({.pages = 1024,
                         .engine = host.connect({}),
                         .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> hot(65, 27);
  engine.submit(request(1, hot));
  h.runUntilIdle();
  // Each keeps its checkpoint at 8192 beside its replay state at 12192.
  const auto keep = [&](uint64_t id, uint32_t token) {
    const std::vector<uint32_t> prompt(12201, token);
    engine.submit(request(id, prompt));
    h.runUntilIdle();
    require(resources.snapshot().stateCache.checkpointEntries == 1,
            "the prompt did not keep its last checkpoint");
    return prompt;
  };
  const std::vector<uint32_t> first = keep(2, 28);
  require(test::evictStep(resources).gave() &&
              resources.snapshot().stateCache.checkpointEvictions == 1 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.lookup(hot, {}).resumeBoundary() == 64 &&
              resources.lookup(first, {}).resumeBoundary() == 12192,
          "reclaim took an ordinary state before the kept checkpoint");

  const std::vector<uint32_t> second = keep(3, 29);
  // While the engine may not grow, a state's buffers take no free pages: the
  // next replay point, refused them once, recycles a state.
  host.pause(true);
  executor.deniedReservations = 1;
  const std::vector<uint32_t> next(65, 30);
  engine.submit(request(4, next));
  h.runUntilIdle();
  const auto counters = engine.snapshot();
  require(counters.recycledStatePublications == 1 &&
              counters.replayStatePublicationFailures == 0 &&
              counters.resources.stateCache.checkpointEvictions == 2 &&
              counters.resources.stateCache.checkpointEntries == 0 &&
              counters.resources.stateCache.entries == 4 &&
              resources.lookup(hot, {}).resumeBoundary() == 64 &&
              resources.lookup(first, {}).resumeBoundary() == 12192 &&
              resources.lookup(second, {}).resumeBoundary() == 12192 &&
              resources.lookup(next, {}).resumeBoundary() == 64,
          "a publication without room took an ordinary state before the kept checkpoint");
}

// Two lanes holding one checkpoint at 4096 as their recovery point: the first's
// next checkpoint retires it only once the second's replay point is published;
// refused, it stays the second's recovery point.
void testSharedCheckpointSurvivesPeerReplayPoint() {
  for (const bool refused : {false, true}) {
    test::EngineHarness h({.pages = 2048, .shape = [](Executor &shape) { shape.maximumLanes = 2; }});
    engine::Cache &resources = h.cache;
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    executor.decodeFinishes = false;
    // Checkpoints at 4096 and 8192, the replay state at 12192.
    std::vector<uint32_t> first(12201);
    std::iota(first.begin(), first.end(), 1);
    engine.submit(request(1, first));
    runUntilCheckpoint(h, 1);
    // The second follows the first to 6000, and its replay point lands at
    // 6176: no state the first plans lies in what they share past 4096, so
    // it does not wait for the first.
    std::vector<uint32_t> second(first.begin(), first.begin() + 6000);
    second.resize(6201, 0);
    EngineRequest decoding = request(2, second);
    decoding.maxNewTokens = 1000;
    // A lookup holds the shared checkpoint while the second's replay point
    // is refused, so that the refusal cannot recycle it.
    std::optional<CacheLookup> held;
    if (refused) {
      held = resources.lookup(first, {});
      executor.denyReservationAt = 6176;
    }
    engine.submit(std::move(decoding));
    double now = 200;
    tickUntil(h, now,
              [&] {
                const auto counters = engine.snapshot();
                return counters.replayStatePublications +
                           counters.replayStatePublicationFailures ==
                       1;
              },
              "the second lane did not reach its replay point");
    require(events.starts.back() == 4096 &&
                engine.snapshot().replayStatePublicationFailures == (refused ? 1U : 0U) &&
                engine.snapshot().checkpointPublications == 1 &&
                resources.probe(first, {}).cachedTokens() == 4096,
            "the second lane's replay point retired the first lane's recovery point");
    held.reset();
    executor.denyReservationAt.reset();
    tickUntil(h, now, [&] { return engine.snapshot().checkpointPublications == 2; },
              "the first lane did not reach its next checkpoint");
    // Refused, the second lane still resumes from the shared checkpoint;
    // otherwise its replay point does, and the first's own progress replaced
    // the shared one.
    require(resources.probe(second, {}).cachedTokens() == (refused ? 4096U : 6176U) &&
                resources.snapshot().stateCache.checkpointEntries == (refused ? 2U : 1U),
            "the first lane's next checkpoint retired the second lane's recovery point");
    executor.decodeFinishes = true;
    tickUntil(h, now, [&] { return idle(engine); }, "the lanes did not finish");
    const auto counters = engine.snapshot();
    require(events.completedCount == 2 && events.failedCount == 0 &&
                counters.resources.activeRequests == 0 &&
                counters.resources.stateCache.pinned == 0 &&
                counters.resources.stateCache.inUse == 0 &&
                counters.resources.stateCache.entries == 3 &&
                counters.resources.stateCache.checkpointEntries == (refused ? 2U : 1U) &&
                resources.probe(first, {}).cachedTokens() == 12192,
            "the lanes did not both finish, or left something behind");
  }
}

// With memory full, a replay point refused buffers of its own takes its lane's
// checkpoint's, not the other lane's older checkpoint; with room, both stay.
void testReplayPointWithoutRoomTakesItsOwnCheckpoint() {
  for (const bool full : {true, false}) {
    GovernedHost host;
    test::EngineHarness h({.pages = 4096,
                           .engine = host.connect({}),
                           .shape = [](Executor &shape) { shape.maximumLanes = 2; }});
    engine::Cache &resources = h.cache;
    Executor &executor = h.model;
    engine::Engine &engine = h.engine;
    bool memoryFull = false;
    executor.reservationRoom = [&] {
      if (*executor.evictedStateBytes >= 64) {
        *executor.evictedStateBytes -= 64;
        return true;
      }
      return !memoryFull;
    };
    // Checkpoints at 4096 and 8192, the replay state at 12192; the other
    // prompt checkpoints every 4096 tokens until it ends.
    std::vector<uint32_t> prompt(12201);
    std::iota(prompt.begin(), prompt.end(), 1'000'000);
    std::vector<uint32_t> other(40001);
    std::iota(other.begin(), other.end(), 1);
    // Leaves the prompt past its checkpoint at 8192.
    std::vector<uint32_t> branch(prompt.begin(), prompt.begin() + 9000);
    branch.resize(9100, 7);
    engine.submit(request(2, other));
    double now = 1;
    tickUntil(h, now, [&] { return engine.snapshot().checkpointPublications == 1; },
              "the other lane did not publish its first checkpoint");
    engine.submit(request(1, prompt));
    // Its commands up to the replay point capture no other state.
    tickUntil(h, now, [&] { return resources.probe(branch, {}).cachedTokens() == 8192; },
              "the lane did not publish its checkpoint at 8192");
    require(resources.probe(other, {}).cachedTokens() == 4096,
            "fixture did not leave each lane its checkpoint");
    host.pause(true);
    memoryFull = full;
    // Memory fills: the idle buffers the retired checkpoint at 4096 left in
    // the pool went back to the host.
    if (full)
      *executor.evictedStateBytes = 0;
    tickUntil(h, now,
              [&] {
                const auto counters = engine.snapshot();
                return counters.replayStatePublications +
                           counters.replayStatePublicationFailures ==
                       1;
              },
              "the lane did not reach its replay point");
    require(engine.snapshot().replayStatePublications == 1 &&
                resources.snapshot().stateCache.checkpointEvictions == 0 &&
                resources.probe(prompt, {}).cachedTokens() == 12192 &&
                resources.probe(other, {}).cachedTokens() == 4096,
            "the replay point took the other lane's recovery point");
    require(resources.probe(branch, {}).cachedTokens() == (full ? 0U : 8192U),
            full ? "a replay point refused room kept its checkpoint"
                 : "a replay point with room retired its checkpoint");
    engine.cancel(2);
    tickUntil(h, now, [&] { return idle(engine); }, "the lanes did not end");
  }
}

// The cache loses the deepest state on the prompt's chain.
void dropDeepestState(engine::Cache &cache, const std::vector<uint32_t> &prompt) {
  CacheLookup lookup = cache.lookup(prompt, {});
  require(lookup.state.has_value(), "the prompt's chain holds no state");
  const uint64_t block = lookup.state->kvBlock();
  lookup = {};
  cache.invalidateState(block);
}

// A Background lane decodes `history` tokens past its prompt, and a Normal
// request the host refuses memory beside it then takes its lane; the lane's
// prompt replay point is lost while it waits, so it resumes from the
// checkpoint before it.
void suspendWithoutTheReplayPoint(test::EngineHarness &h, double &now, const std::vector<uint32_t> &prompt,
                                  uint32_t history) {
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&executor] {
    return executor.lastBeginId == 2 && executor.requests.at(1).resident;
  };
  auto background = request(1, prompt);
  background.priority = RequestPriority::Background;
  background.maxNewTokens = 10'000;
  engine.submit(std::move(background));
  tickUntil(h, now, [&] { return events.outputs[1].size() == history; }, "the lane did not decode its history");
  engine.submit(request(2, std::vector<uint32_t>(65, 2)));
  tickUntil(h, now, [&] { return executor.suspensions == 1; }, "the Normal request did not take the lane");
  dropDeepestState(h.cache, prompt);
}

// A resumed lane's command that captures its replay point and the end of its
// generated history past it keeps the checkpoint it resumed from cached beside
// both: the state past the replay point takes no checkpoint's buffers.
void testStatePastTheReplayPointKeepsTheCheckpoint() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  engine::Engine &engine = h.engine;
  executor.decodeFinishes = false;
  // Checkpoints at 4096 and 8192, the replay point at 12192. A history of 64
  // tokens ends at 12256: the lane resumes from 8192, and its command
  // [10240, 12265) captures both.
  std::vector<uint32_t> prompt(12201);
  std::iota(prompt.begin(), prompt.end(), 1);
  double now = 1;
  suspendWithoutTheReplayPoint(h, now, prompt, 64);
  tickUntil(h, now,
            [&] { return executor.prefillChunks.at(1) == 8 && !engine.commandInFlight(); },
            "the resumed lane did not reach its history's end");
  std::vector<uint32_t> branch(prompt.begin(), prompt.begin() + 9000);
  branch.resize(9100, 0);
  require(executor.restored == 8192 &&
              capturedStates(executor, 1) == std::vector<std::pair<uint32_t, uint64_t>>{
                                                 {4096, 4096},
                                                 {8192, 8192},
                                                 {12192, 12201},
                                                 {12192, 12265},
                                                 {12256, 12265}} &&
              engine.snapshot().checkpointPublications == 3,
          "the resumed lane did not capture its replay point and its history's end");
  require(resources.probe(branch, {}).cachedTokens() == 8192 &&
              resources.snapshot().stateCache.checkpointEntries == 2,
          "a state past the replay point retired the checkpoint the replay point keeps");
  executor.decodeFinishes = true;
  tickUntil(h, now, [&] { return idle(engine); }, "the lane did not finish");
}

// The prompt's replay point keeps the checkpoint before it beside buffers of
// its own; refused them, it takes that checkpoint's rather than another
// conversation's state, also when its own command captures the checkpoint.
void testReplayPointWithoutRoomTakesTheCheckpointItsCommandCaptures() {
  GovernedHost host;
  test::EngineHarness h({.pages = 1024,
                         .engine = host.connect({.prefillCheckpointTokens = 3072}),
                         .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.decodeFinishes = false;
  executor.decodeTokens = 8;
  bool memoryFull = false;
  executor.reservationRoom = [&] {
    if (*executor.evictedStateBytes >= 64) {
      *executor.evictedStateBytes -= 64;
      return true;
    }
    return !memoryFull;
  };
  // A checkpoint at 3072, the replay point at 6656. A history of 1600 tokens
  // ends at 8256: the lane resumes from 3072, and its command [5120, 7168)
  // captures its checkpoint at 6144 and the replay point.
  std::vector<uint32_t> prompt(6657);
  std::iota(prompt.begin(), prompt.end(), 1);
  double now = 1;
  suspendWithoutTheReplayPoint(h, now, prompt, 1600);
  tickUntil(h, now, [&] { return events.usage.contains(2); },
            "the Normal request did not finish");
  // Memory fills: a state's buffers come only from a state the cache lets
  // go, and while the engine may not grow, from no free pages. One state's
  // buffers are idle in the pool, and the checkpoint takes them.
  host.pause(true);
  memoryFull = true;
  *executor.evictedStateBytes = 64;
  const uint64_t evictions = resources.snapshot().stateCache.evictions;
  tickUntil(h, now,
            [&] { return executor.prefillChunks.at(1) == 6 && !engine.commandInFlight(); },
            "the resumed lane did not reach its replay point");
  const auto counters = engine.snapshot();
  require(executor.restored == 3072 &&
              capturedStates(executor, 1).back() == std::pair<uint32_t, uint64_t>{6656, 7168} &&
              counters.replayStatePublications == 3 && counters.checkpointPublications == 1 &&
              counters.resources.stateCache.evictions == evictions &&
              resources.probe(std::vector<uint32_t>(65, 2), {}).cachedTokens() == 64 &&
              resources.probe(prompt, {}).cachedTokens() == 6656,
          "the replay point refused room recycled another conversation's state");
  host.pause(false);
  memoryFull = false;
  executor.decodeFinishes = true;
  tickUntil(h, now, [&] { return idle(engine); }, "the lane did not finish");
  require(events.completedCount == 2 && events.failedCount == 0, "a request failed");
}

void testShortSuffixContinuesCheckpointDraftState() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> prompt(18001, 30);
  engine.submit(request(480, prompt));
  runUntilCheckpoint(h, 1);
  engine.cancel(480);
  h.runUntilIdle();
  std::vector<uint32_t> shorter(
      prompt.begin(), prompt.begin() + kPrefillCheckpointTokens + 209);
  engine.submit(request(481, shorter));
  h.runUntilIdle();
  require(events.starts.back() == kPrefillCheckpointTokens &&
              executor.restoredDraft &&
              draftContextRows(executor.plans.at(481)) == 209 &&
              executor.plans.at(481).restoresDraftState,
          "short checkpoint suffix discarded or rebuilt its restored draft window");
}

void testDefaultCheckpointRestoresLatestCommittedPrefix() {
  test::EngineHarness h({.pages = 2048, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  const std::vector<uint32_t> donor(32769, 61);
  engine.submit(request(600, donor));
  for (uint32_t step = 0; step < 128; ++step) {
    static_cast<void>(tickAt(h, step + 1));
    if (!engine.commandInFlight() &&
        executor.requests.at(600).position == 22528)
      break;
  }
  const auto committed = engine.snapshot();
  require(!engine.commandInFlight() && executor.prefillRows == 22528 &&
              executor.requests.at(600).position == 22528 &&
              committed.resources.pool.pagesPrefix == 22528 / KvCache::pageTokens,
          "fixture did not commit 22528 prompt tokens before cancellation");
  require(committed.checkpointPublications == 5 &&
              committed.checkpointPublicationFailures == 0 &&
              committed.resources.stateCache.entries == 1 &&
              committed.resources.stateCache.checkpointEntries == 1 &&
              committed.resources.stateCache.checkpointRetirements == 4,
          "default checkpoints accumulated or lost temporary states");
  {
    const auto lookup = resources.lookup(donor, {});
    require(lookup.kvBoundary == 22528 && lookup.resumeBoundary() == 20480,
            "completed KV tail did not retain its preceding 4K checkpoint");
  }
  engine.cancel(600);
  h.runUntilIdle();
  require(engine.snapshot().cancelled == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "cancellation removed its completed recovery point");

  std::vector<uint32_t> branch(donor.begin(), donor.begin() + 22529);
  branch.back() = 62;
  engine.submit(request(601, branch));
  h.runUntilIdle();
  const auto finished = engine.snapshot();
  require(events.startIds.back() == 601 &&
              events.starts.back() == 20480 &&
              executor.prefillRows - 22528 == 2049 &&
              finished.completed == 1 && finished.cancelled == 1 &&
              events.failedCount == 0,
          "branch replayed before its latest checkpoint or failed completion");
  // The checkpoint the branch resumed from is its prompt's last: it stays
  // beside the replay state.
  require(finished.checkpointPublications == 5 &&
              finished.replayStatePublications == 1 &&
              finished.resources.stateCache.entries == 2 &&
              finished.resources.stateCache.checkpointEntries == 1 &&
              finished.resources.stateCache.checkpointRetirements == 4 &&
              resources.lookup(branch, {}).resumeBoundary() == 22528,
          "branch completion did not keep its last checkpoint beside its replay state");
}

// A checkpoint costs a cached state, its capture and its draft window, so
// none is planned within one prefill chunk of where a request resumes or of
// its replay boundary: not for a follow-up resuming at 2976 whose replay
// boundary is 4192, nor for one resuming at 4064.
void testCheckpointsSkipNearResumeAndReplayBoundaries() {
  test::EngineHarness h({.pages = 1024, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  const auto checkpointPlanned = [&](uint64_t id) {
    const auto &boundaries = executor.plans.at(id).boundaries;
    return std::any_of(boundaries.begin(), boundaries.end(),
                       [](const DraftBoundaryPlan &boundary) {
                         return boundary.boundary == kPrefillCheckpointTokens;
                       });
  };
  std::vector<uint32_t> prompt(3009);
  std::iota(prompt.begin(), prompt.end(), 1);
  EngineRequest first = request(1, prompt);
  // Its replay point lands at 2976, before the generation prompt.
  first.generationPromptTokens = 7;
  engine.submit(std::move(first));
  h.runUntilIdle();
  const uint32_t reservations = executor.reservations;
  prompt.resize(3002);
  prompt.resize(4201, 500);
  engine.submit(request(2, prompt));
  h.runUntilIdle();
  require(events.starts.back() == 2976 &&
              !checkpointPlanned(2) && engine.snapshot().checkpointPublications == 0 &&
              executor.reservations == reservations + 1,
          "a follow-up checkpointed within a chunk of its resume point or replay boundary");

  std::vector<uint32_t> resumed(4065);
  std::iota(resumed.begin(), resumed.end(), 10'000);
  engine.submit(request(3, resumed));
  h.runUntilIdle();
  resumed.resize(8193, 600);
  engine.submit(request(4, resumed));
  h.runUntilIdle();
  require(events.starts.back() == 4064 &&
              !checkpointPlanned(4) && engine.snapshot().checkpointPublications == 0,
          "a request checkpointed within a chunk of the state it resumed from");
}

} // namespace

std::span<const test::TestCase> checkpointsTests() {
  static constexpr test::TestCase kTests[] = {
      {"testConcurrentProgressRetainsAtMostOnePointPerLane", testConcurrentProgressRetainsAtMostOnePointPerLane},
      {"testSharedCheckpointSurvivesPeerRollingReplacement", testSharedCheckpointSurvivesPeerRollingReplacement},
      {"testRepeatedRetriesRollTheRestoredCheckpoint", testRepeatedRetriesRollTheRestoredCheckpoint},
      {"testRetryCancelledBeforeNextCheckpointKeepsItsSource", testRetryCancelledBeforeNextCheckpointKeepsItsSource},
      {"testRestoredCheckpointAtReplayEndBecomesOrdinary", testRestoredCheckpointAtReplayEndBecomesOrdinary},
      {"testRetryRetiresCheckpointAtDeeperJunction", testRetryRetiresCheckpointAtDeeperJunction},
      {"testPinnedCheckpointSkipsReplacementButNotOrdinaryState",
       testPinnedCheckpointSkipsReplacementButNotOrdinaryState},
      {"testPinnedPredecessorTakesNoRoomFromOtherStates", testPinnedPredecessorTakesNoRoomFromOtherStates},
      {"testPinnedPredecessorReleasedDuringTheCommand", testPinnedPredecessorReleasedDuringTheCommand},
      {"testFailedReplacementContinuesWithoutRecoveryPoint", testFailedReplacementContinuesWithoutRecoveryPoint},
      {"testRollingHandleCannotRetirePromotedState", testRollingHandleCannotRetirePromotedState},
      {"testCheckpointRetiresOnceItsSuccessorIsPublished", testCheckpointRetiresOnceItsSuccessorIsPublished},
      {"testInterruptedCommandKeepsTheLanesCheckpoint", testInterruptedCommandKeepsTheLanesCheckpoint},
      {"testRefusedStateTakesTheCheckpointAfterItsCommand", testRefusedStateTakesTheCheckpointAfterItsCommand},
      {"testJunctionRetiresEarlierProgressPoint", testJunctionRetiresEarlierProgressPoint},
      {"testSecondConversationResumesFromTheLastCheckpoint", testSecondConversationResumesFromTheLastCheckpoint},
      {"testShortSharedPrefixFindsAndKeepsNoCheckpoint", testShortSharedPrefixFindsAndKeepsNoCheckpoint},
      {"testKeptCheckpointGoesFirstUnderPressure", testKeptCheckpointGoesFirstUnderPressure},
      {"testSharedCheckpointSurvivesPeerReplayPoint", testSharedCheckpointSurvivesPeerReplayPoint},
      {"testReplayPointWithoutRoomTakesItsOwnCheckpoint", testReplayPointWithoutRoomTakesItsOwnCheckpoint},
      {"testStatePastTheReplayPointKeepsTheCheckpoint", testStatePastTheReplayPointKeepsTheCheckpoint},
      {"testReplayPointWithoutRoomTakesTheCheckpointItsCommandCaptures",
       testReplayPointWithoutRoomTakesTheCheckpointItsCommandCaptures},
      {"testShortSuffixContinuesCheckpointDraftState", testShortSuffixContinuesCheckpointDraftState},
      {"testDefaultCheckpointRestoresLatestCommittedPrefix", testDefaultCheckpointRestoresLatestCommittedPrefix},
      {"testCheckpointsSkipNearResumeAndReplayBoundaries", testCheckpointsSkipNearResumeAndReplayBoundaries},
  };
  return kTests;
}
