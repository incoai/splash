#include "EngineHarness.hpp"
#include "EngineTestSupport.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestRegistry.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>

using namespace splash;
using namespace splash::engine;
using splash::test::constrainedRequest;
using splash::test::Events;
using splash::test::GovernedHost;
using splash::test::Executor;
using splash::test::guardReleases;
using splash::test::idle;
using splash::test::request;

namespace {

using splash::test::rejects;
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

void testConcurrentColdPrefixesComputeOnce() {
  test::EngineHarness h({.pages = 64});
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  for (uint32_t id = 1; id <= 4; ++id) {
    std::vector<uint32_t> prompt(193, 7);
    std::fill(prompt.begin() + 160, prompt.end(), id + 10);
    engine.submit(request(id, prompt));
  }
  static_cast<void>(tickAt(h, 0));
  require(model.requests.size() == 1,
          "shared cold prefix allocated redundant lanes");
  h.runUntilIdle();
  require(model.prefillRows == 160 + 4 * 33 && model.restored == 3 * 160,
          "concurrent cold requests recomputed their shared prefix");
  require(events.completedCount == 4 && events.failedCount == 0 &&
              events.emitted == 4,
          "shared prefill lost independent completions");
}

// A cold request whose first blocks match cached ones computes them, then
// reads the cached blocks' pages in its later commands: it reports those
// tokens when it completes, though it restored nothing.
void testColdRequestReportsTheBlocksItReadFromTheCache() {
  test::EngineHarness h({.pages = 64});
  engine::Cache &cache = h.cache;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  std::vector<uint32_t> first(100);
  std::iota(first.begin(), first.end(), 1);
  engine.submit(request(1, first));
  h.runUntilIdle();
  // Two full blocks in common and no state where the prompts part.
  std::vector<uint32_t> second = first;
  std::fill(second.begin() + 2 * KvCache::pageTokens + 5, second.end(), 7);
  require(cache.lookup(second, {}).kvBoundary == 2 * KvCache::pageTokens &&
              cache.lookup(second, {}).resumeBoundary() == 0,
          "fixture did not share KV without a state");
  engine.submit(request(2, second));
  h.runUntilIdle();
  require(events.starts == std::vector<uint32_t>{0, 0} && events.completedCount == 2,
          "the second request did not start cold");
  require(events.deduplicatedTokens.at(1) == 0 &&
              events.deduplicatedTokens.at(2) == 2 * KvCache::pageTokens &&
              engine.snapshot().deduplicatedKvTokens == 2 * KvCache::pageTokens,
          "a cold request did not report the cached blocks it read");
}

// Requests of different cache salts share nothing: concurrent ones each
// compute their prefix instead of waiting for another's state, and a later
// one restores only a state of its own salt and swaps no other salt's
// block in.
void testCacheSaltsShareNothing() {
  test::EngineHarness h({.pages = 64});
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  const auto salted = [](uint64_t id, const std::vector<uint32_t> &prompt, uint64_t salt) {
    EngineRequest value = request(id, prompt);
    value.cacheSalt = {salt, 0};
    return value;
  };
  for (uint32_t id = 1; id <= 4; ++id) {
    std::vector<uint32_t> prompt(193, 7);
    std::fill(prompt.begin() + 160, prompt.end(), id + 10);
    engine.submit(salted(id, prompt, id));
  }
  static_cast<void>(tickAt(h, 0));
  require(model.requests.size() == 4, "requests of different salts waited for another's prefix");
  h.runUntilIdle();
  require(model.prefillRows == 4 * 193 && model.restored == 0 && events.completedCount == 4,
          "concurrent requests of different salts shared a prefix");
  const uint32_t computed = model.prefillRows;
  std::vector<uint32_t> prompt(193, 7);
  std::fill(prompt.begin() + 160, prompt.end(), 30);
  engine.submit(salted(5, prompt, 99));
  h.runUntilIdle();
  require(events.starts.back() == 0 && events.deduplicatedTokens.at(5) == 0 &&
              model.prefillRows - computed == 193,
          "a request read another salt's cached blocks");
  engine.submit(salted(6, prompt, 99));
  h.runUntilIdle();
  require(events.starts.back() > 0, "a request missed its own salt's cached prefix");
}

// A request whose last command fills blocks found cached never reads their
// pages, so it reports none of them.
void testBlocksFoundCachedAfterTheLastCommandAreNotReported() {
  test::EngineHarness h({.pages = 64});
  engine::Cache &cache = h.cache;
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  std::vector<uint32_t> first(100);
  std::iota(first.begin(), first.end(), 1);
  engine.submit(request(1, first));
  h.runUntilIdle();
  // A generation prompt that starts in the first block leaves the second
  // request no state boundary, and its prompt's command selects its only
  // token: that command is its last.
  const std::vector<uint32_t> second(first.begin(), first.begin() + 70);
  EngineRequest last = request(2, second);
  last.generationPromptTokens = 40;
  require(cache.lookup(second, {}).kvBoundary == 2 * KvCache::pageTokens &&
              cache.lookup(second, {}).resumeBoundary() == 0,
          "fixture did not share KV without a state");
  model.prefillAnchor = true;
  engine.submit(std::move(last));
  h.runUntilIdle();
  require(events.starts == std::vector<uint32_t>{0, 0} && events.completedCount == 2 &&
              model.prefillChunks.at(2) == 1 && events.outputs.at(2) == std::vector<uint32_t>{42},
          "the second request did not end with the command of its prompt");
  require(events.deduplicatedTokens.at(2) == 0 && engine.snapshot().deduplicatedKvTokens == 0,
          "a request reported cached blocks no command of it read");
}

// The engine refuses a thinking budget the target could not end, as the
// protocol does: one without end tokens, and one with an end token before the
// last that would end thinking itself.
void testUnusableThinkingBudgetIsRefused() {
  test::EngineHarness h({.pages = 16, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Engine &engine = h.engine;
  EngineRequest empty = request(1, std::vector<uint32_t>(8, 7));
  empty.thinkingBudget = ThinkingBudget{4, {}, 9};
  rejects([&] { engine.submit(std::move(empty)); }, "needs end tokens",
          "a thinking budget without end tokens was accepted");
  EngineRequest early = request(2, std::vector<uint32_t>(8, 7));
  early.thinkingBudget = ThinkingBudget{4, {11, 12, 11}, 9};
  rejects([&] { engine.submit(std::move(early)); }, "only the last thinking end token",
          "a thinking budget whose end closes thinking early was accepted");
  EngineRequest usable = request(3, std::vector<uint32_t>(8, 7));
  usable.thinkingBudget = ThinkingBudget{4, {12, 11}, 9};
  engine.submit(std::move(usable));
}

void testSharedPrefillRebuildsTheMissingJunctionOnce() {
  test::EngineHarness h({.pages = 512});
  engine::Cache &cache = h.cache;
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  engine.submit(request(1, std::vector<uint32_t>(6530, 7)));
  h.runUntilIdle();
  std::vector<uint32_t> branch(6575, 7);
  std::fill(branch.begin() + 6517, branch.end(), 8);
  {
    // The prompt kept its last checkpoint, at 4096; the branch point has KV
    // and no state.
    auto hit = cache.lookup(branch, {});
    require(hit.kvBoundary == 6496 && hit.resumeBoundary() == 4096,
            "fixture did not recreate an internal branch past the kept checkpoint");
  }
  const uint32_t before = model.prefillRows;
  for (uint32_t id = 2; id <= 5; ++id) {
    std::fill(branch.begin() + 6517, branch.end(), id + 10);
    engine.submit(request(id, branch));
  }
  h.runUntilIdle();
  require(model.prefillRows - before == (6496 - 4096) + 4 * (6575 - 6496) &&
              model.restored == 4096 + 3 * 6496 && events.completedCount == 5,
          "concurrent internal branches each rebuilt the missing GDN state");
}

void testSharedPrefillReleasesDifferentJunctionsIndependently() {
  test::EngineHarness h({.pages = 64});
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  engine.submit(request(1, std::vector<uint32_t>(193, 7)));
  for (uint32_t id = 2; id <= 3; ++id) {
    std::vector<uint32_t> branch(193, 7);
    std::fill(branch.begin() + (id == 2 ? 96 : 160), branch.end(), id + 10);
    engine.submit(request(id, branch));
  }
  static_cast<void>(tickAt(h, 0));
  static_cast<void>(tickAt(h, 1));
  static_cast<void>(tickAt(h, 2));
  require(model.requests.size() == 2 && events.startIds.back() == 2,
          "a ready junction waited for the producer's longer shared prefix");
  h.runUntilIdle();
  require(model.prefillRows == 193 + 97 + 33 && events.completedCount == 3,
          "different shared boundaries were not restored independently");
}

void testSharedPrefillEvictedPublicationFallsBack() {
  test::EngineHarness h({.pages = 64});
  engine::Cache &cache = h.cache;
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  engine.submit(request(1, std::vector<uint32_t>(193, 7)));
  engine.submit(request(2, std::vector<uint32_t>(193, 7)));
  static_cast<void>(tickAt(h, 0));
  static_cast<void>(tickAt(h, 1));
  // The producer's replay point is in use, so pressure takes it, not an
  // ordinary publication.
  while (test::releaseStep(cache).gave()) {
  }
  require(cache.snapshot().stateCache.entries == 0,
          "published shared prefix was pinned against pressure reclamation");
  h.runUntilIdle();
  require(events.completedCount == 2 && model.prefillRows == 386 &&
              cache.snapshot().activeRequests == 0,
          "evicted shared publication stranded its waiter");
}

void testSharedPrefillProducerFailureReleasesWaiters() {
  for (bool cancelled : {false, true}) {
    test::EngineHarness h({.pages = 64});
    engine::Cache &cache = h.cache;
    Executor &model = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    engine.submit(request(1, std::vector<uint32_t>(193, 7)));
    engine.submit(request(2, std::vector<uint32_t>(193, 7)));
    static_cast<void>(tickAt(h, 0));
    require(engine.snapshot().scheduler.waitingPrefix == 1 &&
                engine.resourceWaitSnapshot(0).memory == 0 &&
                model.requests.size() == 1,
            "shared prefill waiter was admitted before its prefix was ready");
    if (cancelled)
      engine.cancel(1);
    else
      engine.failRequest(1, LaneOutcome::InvalidMask, "producer failed");
    h.runUntilIdle();
    require(events.outputs[2] == std::vector<uint32_t>{42} &&
                cache.snapshot().activeRequests == 0 && model.requests.empty(),
            "failed prefix producer stranded a waiter or leaked resources");
  }
}

void testSharedPrefillWaiterCancellationAndDeadline() {
  for (bool cancelled : {false, true}) {
    test::EngineHarness h({.pages = 64});
    Executor &model = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    engine.submit(request(1, std::vector<uint32_t>(193, 7)));
    auto waiter = request(2, std::vector<uint32_t>(193, 7));
    // Alive at the first tick, past its deadline at the next.
    waiter.deadlineMilliseconds = h.now() + 1;
    engine.submit(std::move(waiter));
    static_cast<void>(tickAt(h, 0));
    if (cancelled)
      engine.cancel(2);
    // Its deadline passes while it waits for the prefix.
    h.advanceTo(h.now() + 1);
    h.runUntilIdle();
    require(model.beginAttempts == 1 && events.outputs[2].empty() &&
                events.outputs[1] == std::vector<uint32_t>{42} &&
                engine.snapshot().scheduler.waitingPrefix == 0,
            "expired prefix waiter allocated a lane or interrupted its producer");
    require(!cancelled || events.usage.at(2) == std::pair<uint32_t, uint32_t>{193, 0},
            "a request cancelled before it started reported other usage");
  }
}

void testSharedPrefillFailedPublicationFallsBack() {
  test::EngineHarness h({.pages = 64});
  engine::Cache &cache = h.cache;
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  model.deniedReservations = 100;
  for (uint32_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(193, 7)));
  h.runUntilIdle();
  require(events.completedCount == 4 && events.failedCount == 0 &&
              model.prefillRows == 4 * 193 && cache.snapshot().activeRequests == 0,
          "a missing prefix state stranded dependent requests");
}

void testSharedPrefillDoesNotBlockUnrelatedWork() {
  for (bool images : {false, true}) {
    test::EngineHarness h({.pages = 64});
    Executor &model = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    for (uint32_t id = 1; id <= 4; ++id) {
      auto value = request(id, std::vector<uint32_t>(193, images ? 7 : id));
      if (images) {
        value.images = {{0, 16, 8, 8, id, id}};
        value.imagePixels.assign(value.images.front().pixelBytes(), 1);
      }
      engine.submit(std::move(value));
    }
    static_cast<void>(tickAt(h, 0));
    require(model.requests.size() == 4 &&
                engine.snapshot().scheduler.waitingPrefix == 0,
            "unrelated token or image prefixes were serialized");
    h.runUntilIdle();
    require(model.prefillRows == 4 * 193 && events.completedCount == 4,
            "unrelated work incorrectly reused a shared state");
  }
}

void testSharedPrefillHonorsPriorityAndLateArrival() {
  for (bool foreground : {false, true}) {
    test::EngineHarness h({.pages = 512});
    Executor &model = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    auto producer = request(1, std::vector<uint32_t>(5001, 7));
    producer.priority = RequestPriority::Background;
    engine.submit(std::move(producer));
    static_cast<void>(tickAt(h, 0));
    auto waiter = request(2, std::vector<uint32_t>(5001, 7));
    waiter.priority = foreground ? RequestPriority::Foreground
                                : RequestPriority::Background;
    engine.submit(std::move(waiter));
    static_cast<void>(tickAt(h, 1));
    static_cast<void>(tickAt(h, 2));
    require(model.requests.size() == (foreground ? 2u : 1u),
            "prefix admission ignored priority or a late arrival");
    h.runUntilIdle();
    require(events.completedCount == 2 && events.failedCount == 0,
            "late prefix waiter did not finish");
    if (!foreground)
      require(model.prefillRows == 5010 && model.restored == 4992,
              "late arrival missed the producer's planned replay point");
  }
}

// Late siblings sharing the producer's prompt up to 5984 extend its plan once
// its command ends at its checkpoint at 4096, refused or not, where its draft
// state is complete.
void testLateSharedPrefillExtendsTheProducerPlan() {
  for (bool denyCheckpoint : {false, true}) {
    test::EngineHarness h({.pages = 512});
    Executor &model = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    if (denyCheckpoint)
      model.denyReservationAt = 4096;
    engine.submit(request(1, std::vector<uint32_t>(6601, 7)));
    static_cast<void>(tickAt(h, 0));
    for (uint32_t id = 2; id <= 4; ++id) {
      std::vector<uint32_t> branch(6601, 7);
      std::fill(branch.begin() + 6000, branch.end(), id + 10);
      engine.submit(request(id, branch));
    }
    h.runUntilIdle();
    require(events.completedCount == 4 && events.failedCount == 0 &&
                model.prefillRows == 6601 + 3 * (6601 - 5984) &&
                model.restored == 3 * 5984,
            "late siblings recomputed the prefix after the producer's checkpoint");
  }
}

// A late sibling gets a new boundary only where a whole draft window can still be
// captured for it: past the producer's command end (7232) or a boundary planned
// before it (6976), not at 5600; sharing a planned checkpoint (8192) makes that
// one a junction. (The producer resumes off the chunk grid, at 1056.)
void testLateSiblingBoundaryNeedsAWholeDraftWindow() {
  test::EngineHarness h({.pages = 2048});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  std::vector<uint32_t> prompt(12001);
  std::iota(prompt.begin(), prompt.end(), 1);
  const auto branch = [&](uint32_t shared, uint32_t first) {
    std::vector<uint32_t> result(prompt.begin(), prompt.begin() + shared);
    result.resize(shared + 200);
    std::iota(result.begin() + shared, result.end(), first);
    return result;
  };
  // Its replay point is where the producer resumes, and another
  // conversation's KV goes on with the prompt to 6400, where the producer
  // plans a junction; that conversation's checkpoint at 4096 is evicted.
  engine.submit(request(1, std::vector<uint32_t>(prompt.begin(), prompt.begin() + 1057)));
  h.runUntilIdle();
  engine.submit(request(2, branch(6400, 900'000)));
  h.runUntilIdle();
  require(test::evictStep(resources).gave() &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "the other conversation's checkpoint was not evicted");
  engine.submit(request(3, prompt));
  double now = 1;
  tickUntil(h, now,
            [&] {
              return !engine.commandInFlight() && executor.requests.contains(3) &&
                     executor.requests.at(3).position == 3104;
            },
            "the producer did not run its first command");
  executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
  static_cast<void>(tickAt(h, now++));
  require(engine.commandInFlight() && executor.requests.at(3).position == 5152,
          "the producer's command did not pass its checkpoint");
  const std::array<std::pair<uint64_t, uint32_t>, 4> siblings{
      {{4, 5600}, {5, 6976}, {6, 7232}, {7, 8192}}};
  for (const auto &[id, shared] : siblings)
    engine.submit(request(id, branch(shared, 1'000'000 * static_cast<uint32_t>(id))));
  // The command's completion plans the siblings' boundaries.
  *executor.holdPrefillUntil = true;
  static_cast<void>(tickAt(h, now++));
  const auto &boundaries = executor.plans.at(3).boundaries;
  const auto planned = [&](uint32_t tokens) {
    return std::any_of(boundaries.begin(), boundaries.end(),
                       [&](const DraftBoundaryPlan &boundary) {
                         return boundary.boundary == tokens;
                       });
  };
  require(!engine.commandInFlight() && !planned(5600) && planned(6400) && planned(6976) &&
              planned(7232) && planned(8192),
          "late siblings' boundaries ignored the draft window");
  tickUntil(h, now, [&] { return idle(engine); }, "the requests did not finish");
  const auto start = [&](uint64_t id) {
    const auto found = std::find(events.startIds.begin(), events.startIds.end(), id);
    return events.starts.at(static_cast<size_t>(found - events.startIds.begin()));
  };
  require(start(3) == 1056 && start(4) == 4096 && start(5) == 6976 && start(6) == 7232 &&
              start(7) == 8192,
          "a late sibling did not resume from the producer's state it shares");
  const auto counters = engine.snapshot();
  require(counters.checkpointPublications == 2 && events.completedCount == 7 &&
              events.failedCount == 0 &&
              resources.lookup(branch(8192, 2'000'000), {}).resumeBoundary() == 8192,
          "the checkpoint a late sibling shares was not published as a junction");
}

// A late sibling sharing the producer's prompt up to its planned checkpoint
// (6144) or past it (6784) gets that checkpoint, or a junction after it, as its
// junction; the producer's next command ends there, and a junction past the
// checkpoint takes its place, so the command does not capture the checkpoint.
void testLateSiblingJunctionAtOrPastAPlannedCheckpoint() {
  for (const uint32_t shared : {6144u, 6784u}) {
    test::EngineHarness h({.pages = 2048, .engine = {.prefillCheckpointTokens = 2048}});
    engine::Cache &resources = h.cache;
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    std::vector<uint32_t> prompt(12001);
    std::iota(prompt.begin(), prompt.end(), 1);
    const auto branch = [&](uint32_t first) {
      std::vector<uint32_t> result(prompt.begin(), prompt.begin() + shared);
      result.resize(shared + 200);
      std::iota(result.begin() + shared, result.end(), first);
      return result;
    };
    engine.submit(request(1, std::vector<uint32_t>(prompt.begin(), prompt.begin() + 1057)));
    h.runUntilIdle();
    engine.submit(request(2, prompt));
    double now = 1;
    tickUntil(h, now,
              [&] {
                return !engine.commandInFlight() && executor.requests.contains(2) &&
                       executor.requests.at(2).position == 3104;
              },
              "the producer did not run its first command");
    executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
    static_cast<void>(tickAt(h, now++));
    require(engine.commandInFlight() && executor.requests.at(2).position == 5152,
            "the producer's command did not pass its checkpoint");
    engine.submit(request(3, branch(1'000'000)));
    *executor.holdPrefillUntil = true;
    tickUntil(h, now, [&] { return idle(engine); }, "the requests did not finish");
    const auto captured = [&](uint32_t tokens) {
      return std::count_if(executor.captures.begin(), executor.captures.end(),
                           [&](const Executor::Capture &capture) {
                             return capture.requestId == 2 && capture.tokens == tokens;
                           });
    };
    require(captured(6144) == (shared == 6144 ? 1 : 0) && captured(shared) == 1,
            "the producer captured a checkpoint its junction retires at once");
    const auto counters = engine.snapshot();
    require(events.completedCount == 3 && events.failedCount == 0 &&
                events.starts.back() == shared && counters.captureCuts == 1 &&
                counters.junctionMaterializations == 1 && counters.checkpointPublications == 2 &&
                counters.checkpointPublicationFailures == 0 &&
                resources.lookup(branch(2'000'000), {}).resumeBoundary() == shared,
            "the sibling did not resume from the producer's junction after one command");
  }
}

// A sibling's shared boundary on one of the producer's checkpoints makes that
// checkpoint a state the sibling resumes from: it is published as a junction
// and outlives the producer's later progress instead of rolling with it.
void testSharedJunctionAtACheckpointIsReusable() {
  test::EngineHarness h({.pages = 512});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  // The producer checkpoints at 4096; the sibling shares its first 4100
  // tokens.
  std::vector<uint32_t> prompt(8193);
  std::iota(prompt.begin(), prompt.end(), 1);
  auto producer = request(1, prompt);
  // Admitted first, it plans the sibling's junction on its own checkpoint.
  producer.priority = RequestPriority::Foreground;
  engine.submit(std::move(producer));
  std::vector<uint32_t> sibling(prompt.begin(), prompt.begin() + 4100);
  sibling.resize(4200, 0);
  engine.submit(request(2, sibling));
  h.runUntilIdle();
  const auto counters = engine.snapshot();
  require(events.starts.at(1) == 4096 &&
              executor.restored == 4096,
          "the sibling did not resume from the producer's shared boundary");
  require(counters.junctionMaterializations == 1 && counters.checkpointPublications == 0 &&
              counters.resources.stateCache.checkpointRetirements == 0 &&
              counters.resources.stateCache.checkpointEntries == 0,
          "the shared boundary was published as a rolling checkpoint");
  std::vector<uint32_t> next(prompt.begin(), prompt.begin() + 4100);
  next.resize(4140, 1);
  require(resources.lookup(next, {}).resumeBoundary() == 4096,
          "the producer's later progress retired the shared junction");
}

void testSharedPrefillCapacityFailureDoesNotDeadlock() {
  test::EngineHarness h({.pages = 8});
  h.storage.budgetPages = 4;
  engine::Cache &cache = h.cache;
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  for (uint32_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(193, 7)));
  for (uint32_t step = 0; step < 64 && !idle(engine); ++step)
    static_cast<void>(tickAt(h, step * 1000));
  require(idle(engine) && cache.snapshot().activeRequests == 0 &&
              model.requests.empty() && events.emitted == 0,
          "capacity failure stranded a shared prefix producer or waiter");
  engine.submit(request(5, std::vector<uint32_t>(33, 9)));
  h.runUntilIdle();
  require(events.outputs[5] == std::vector<uint32_t>{42},
          "capacity failure prevented subsequent service");
}

// A request that fails in an admission pass, naming its lane at its start,
// stays listed until the pass ends; a sibling admitted after it in the same
// pass plans no junction for it.
void testSiblingFailedInTheAdmissionPassGetsNoJunction() {
  test::EngineHarness h({.pages = 64});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  // 64 shared tokens, then 33 of each one's own.
  std::vector<uint32_t> prompt(97);
  std::iota(prompt.begin(), prompt.end(), 1);
  // Alone, the first cannot get its state: it fails at once.
  executor.deniedBegins = 1;
  engine.submit(request(1, prompt));
  std::fill(prompt.begin() + 64, prompt.end(), 7);
  engine.submit(request(2, prompt));
  static_cast<void>(tickAt(h, 1));
  require(events.capacityExhaustedCount == 1 && events.startIds == std::vector<uint64_t>{2},
          "the two requests were not admitted in one pass");
  const std::string &message = events.failureDetails.front().first;
  require(message.starts_with("could not allocate request state: ") &&
              message.find("; the request, a Resident lane of 97 tokens at its start, ") != std::string::npos,
          "the request alone without its state did not fail naming its lane at its start: " + message);
  h.runUntilIdle();
  const auto &boundaries = executor.plans.at(2).boundaries;
  require(std::none_of(boundaries.begin(), boundaries.end(),
                       [](const DraftBoundaryPlan &boundary) {
                         return boundary.boundary == 64;
                       }) &&
              engine.snapshot().junctionMaterializations == 0 && events.completedCount == 1,
          "a sibling planned a junction for a request that failed before it");
}

// A cold request publishes its replay state; once that state is gone, the
// next request over the same prompt rebuilds it where the KV still matches,
// as a replay state, and the one after resumes from it.
void testColdPublishesReplayStateAndRebuildsALostOne() {
  test::EngineHarness h({.pages = 32, .engine = {.maxContext = 102400}});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  require(engine.snapshot().maximumContextTokens == 102400,
          "snapshot lost the configured context limit");
  std::vector<uint32_t> prompt(65);
  for (uint32_t i = 0; i < prompt.size(); ++i)
    prompt[i] = i + 1;

  engine.submit(request(1, prompt));
  h.runUntilIdle();
  require(executor.prefillRows == 65 && executor.reservations == 1 &&
              engine.snapshot().replayStatePublications == 1,
          "cold request did not retain its latest Page32 replay state");
  require(events.starts.size() == 1 && events.starts[0] == 0,
          "cold request reported a cache hit");

  require(test::stateStep(resources, ReclaimClass::InUse).gave(),
          "test could not remove the latest replay state");
  engine.submit(request(2, prompt));
  h.runUntilIdle();
  require(executor.restored == 0 && executor.prefillRows == 130 &&
              executor.reservations == 2 &&
              engine.snapshot().replayStatePublications == 2 &&
              engine.snapshot().junctionMaterializations == 0,
          "second request did not rebuild the lost replay state");
  require(events.starts.size() == 2 &&
              events.starts[1] == 0,
          "KV-only replay was reported as a state hit");

  engine.submit(request(3, prompt));
  h.runUntilIdle();
  require(executor.restored == 64 && executor.prefillRows == 131,
          "cache hit did not replay exactly one real input token");
  require(events.starts.size() == 3 && events.starts[2] == 64,
          "state-backed hit accounting is wrong");
  require(events.completedCount == 3 && events.failedCount == 0 &&
              events.emitted == 3,
          "request lifecycle did not complete cleanly");
}

void testConcurrentDuplicateStateIsCapturedOnce() {
  test::EngineHarness h({.pages = 32});
  Executor &executor = h.model;
  engine::Engine &engine = h.engine;
  std::vector<uint32_t> prompt(65);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;

  engine.submit(request(101, prompt));
  engine.submit(request(102, prompt));
  h.runUntilIdle();

  // The second request restores the first publication without reserving a
  // redundant lane or capturing the same state again.
  const auto snapshot = engine.snapshot();
  require(executor.reservationAttempts == 1 && executor.reservations == 1,
          "duplicate state publication captured the state twice");
  require(snapshot.resources.stateCache.entries == 1 &&
              snapshot.resources.stateCache.publications == 1 &&
              snapshot.cacheHits == 1 && executor.prefillRows == 66 &&
              snapshot.replayStatePublications == 1 &&
              snapshot.replayStatePublicationFailures == 0,
          "duplicate state publication was not reused and accounted");
}

// A prompt's replay state is its last whole page before its generation
// prompt, or before its last token when that is unknown.
void testReplayStateEndsBeforeTheGenerationPrompt() {
  test::EngineHarness h({.pages = 32});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  engine::Engine &engine = h.engine;
  struct Case {
    uint32_t tokens;
    uint32_t generationPromptTokens;
    uint32_t replayBoundary;
  };
  uint64_t id = 0;
  for (const Case &test : {Case{97, 0, 96}, Case{97, 7, 64}, Case{103, 7, 96},
                           Case{33, 7, 0}}) {
    std::vector<uint32_t> prompt(test.tokens);
    std::iota(prompt.begin(), prompt.end(), static_cast<uint32_t>(++id * 1000));
    EngineRequest value = request(id, prompt);
    value.generationPromptTokens = test.generationPromptTokens;
    engine.submit(std::move(value));
    h.runUntilIdle();
    require(resources.lookup(prompt, {}).resumeBoundary() == test.replayBoundary,
            "the replay state did not end before the generation prompt");
  }
  require(executor.reservations == 3,
          "a generation prompt within the first page left a replay state");
}

// A next turn that renders the generation prompt differently diverges inside
// it (at N-4, as Qwen3.6 without reasoning does). Only a replay state before
// the generation prompt serves it.
void testFollowUpResumesBeforeTheGenerationPrompt() {
  const auto followUp = [](uint32_t generationPromptTokens) {
    test::EngineHarness h({.pages = 32});
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    std::vector<uint32_t> prompt(97);
    std::iota(prompt.begin(), prompt.end(), 1);
    EngineRequest turn = request(1, prompt);
    turn.generationPromptTokens = generationPromptTokens;
    engine.submit(std::move(turn));
    h.runUntilIdle();
    prompt.resize(93);
    prompt.resize(133, 500);
    engine.submit(request(2, prompt));
    h.runUntilIdle();
    return std::pair{executor.plans.at(1), events.starts.at(1)};
  };
  const auto [plan, start] = followUp(7);
  require(plan.boundaries.size() == 2 && plan.boundaries[0].boundary == 64 &&
              plan.boundaries[1].boundary == 97,
          "the replay state was not planned before the generation prompt");
  require(start == 64,
          "the follow-up did not resume before the generation prompt");
  require(followUp(0).second == 0,
          "a replay state past the divergence served the follow-up");
}

// An identical retry, in sequence or concurrently, resumes from the replay
// state and publishes no second state inside the generation prompt.
void testRetryPublishesNoStateInsideTheGenerationPrompt() {
  for (bool concurrent : {false, true}) {
    test::EngineHarness h({.pages = 32});
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    std::vector<uint32_t> prompt(97);
    std::iota(prompt.begin(), prompt.end(), 1);
    for (uint64_t id : {1, 2}) {
      EngineRequest value = request(id, prompt);
      value.generationPromptTokens = 7;
      engine.submit(std::move(value));
      if (!concurrent)
        h.runUntilIdle();
    }
    h.runUntilIdle();
    const auto snapshot = engine.snapshot();
    require(executor.reservationAttempts == 1 &&
                snapshot.junctionMaterializations == 0 &&
                snapshot.resources.stateCache.entries == 1 &&
                events.starts.at(1) == 64,
            "a retry published a state inside the generation prompt");
  }
}

// A shared junction serves the waiter's next turn too, so it also ends before
// the waiter's generation prompt.
void testSharedJunctionEndsBeforeTheGenerationPrompt() {
  test::EngineHarness h({.pages = 64});
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  std::vector<uint32_t> prompt(200);
  std::iota(prompt.begin(), prompt.end(), 1);
  EngineRequest producer = request(1, prompt);
  // Admitted first, it plans the junction for the waiter's whole prompt.
  producer.priority = RequestPriority::Foreground;
  producer.generationPromptTokens = 7;
  engine.submit(std::move(producer));
  prompt.resize(97);
  EngineRequest waiter = request(2, prompt);
  waiter.generationPromptTokens = 7;
  engine.submit(std::move(waiter));
  h.runUntilIdle();
  prompt.resize(93);
  prompt.resize(133, 500);
  engine.submit(request(3, prompt));
  h.runUntilIdle();
  require(events.starts.size() == 3 && events.starts[1] == 64 &&
              events.starts[2] == 64,
          "the shared junction was not the waiter's reusable state");
}

void testImageSpansKeyPrefixIdentity() {
  test::EngineHarness h({.pages = 32});
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  // Image runs render as one placeholder id, so two prompts with different
  // images are token-identical; only the span digests differ.
  std::vector<uint32_t> prompt(65, 248056);
  auto withImage = [&](uint64_t id, uint64_t digest) {
    EngineRequest result = request(id, prompt);
    result.images = {{8, 16, 8, 8, digest, digest ^ 0xabcdULL}};
    result.imagePixels.assign(result.images[0].pixelBytes(), 1);
    return result;
  };

  engine.submit(withImage(1, 0x1111));
  h.runUntilIdle();
  require(events.starts.size() == 1 && events.starts[0] == 0,
          "image producer unexpectedly hit the cache");
  engine.submit(withImage(2, 0x2222));
  h.runUntilIdle();
  require(events.starts.size() == 2 && events.starts[1] == 0,
          "a different image falsely matched the cached prefix");
  engine.submit(withImage(3, 0x1111));
  h.runUntilIdle();
  require(events.starts.size() == 3 && events.starts[2] == 64,
          "an identical image did not reuse the cached prefix");
  engine.submit(request(4, prompt));
  h.runUntilIdle();
  require(events.starts.size() == 4 && events.starts[3] == 0,
          "a text-only prompt matched an image-keyed prefix");
  require(events.completedCount == 4 && events.failedCount == 0,
          "image request lifecycle did not complete cleanly");
}

// Two token-identical prompts share their prefix up to the block that holds
// the start of the first image on which they differ: the later one waits for
// the junction there and resumes from it.
void testSharedPrefillBoundaryStopsAtTheFirstDifferentImage() {
  const ImageSpan a{40, 16, 8, 8, 0x1111, 0x2222};
  const ImageSpan b{200, 16, 8, 8, 0x3333, 0x4444};
  const ImageSpan c{200, 16, 8, 8, 0x5555, 0x6666};
  ImageSpan otherA = a;
  otherA.digestLo ^= 1;
  struct Shape {
    std::vector<ImageSpan> producer;
    std::vector<ImageSpan> waiter;
    uint32_t shared = 0;
  };
  for (const Shape &shape : {Shape{{a, b}, {a, c}, 192}, Shape{{a}, {otherA}, 32}}) {
    test::EngineHarness h({.pages = 64});
    Executor &model = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    const auto withImages = [](uint64_t id, const std::vector<ImageSpan> &images) {
      EngineRequest result = request(id, std::vector<uint32_t>(289, 248056));
      result.images = images;
      for (const ImageSpan &image : images)
        result.imagePixels.resize(result.imagePixels.size() + image.pixelBytes(), 1);
      return result;
    };
    engine.submit(withImages(1, shape.producer));
    engine.submit(withImages(2, shape.waiter));
    static_cast<void>(tickAt(h, 0));
    require(model.requests.size() == 1 && engine.snapshot().scheduler.waitingPrefix == 1,
            "the later request did not wait for the shared prefix");
    h.runUntilIdle();
    require(events.starts.size() == 2 &&
                events.starts[1] == shape.shared &&
                model.prefillRows == 289 + 289 - shape.shared && events.completedCount == 2,
            "the shared prefix did not end at the block of the first different image");
  }
}

// One prefill chunk that every caller's prompt shares, then 65 tokens of
// `token`: the replay state lands at 2112. A prompt branching off a cached
// one at the end of the chunk plans a junction there, a draft window past
// the start.
std::vector<uint32_t> branchPrompt(uint32_t token) {
  std::vector<uint32_t> prompt(model::ExecutionLimits::prefillTokenBudget);
  std::iota(prompt.begin(), prompt.end(), 1);
  prompt.resize(prompt.size() + 65, token);
  return prompt;
}

// A request branching off a cached conversation that goes on to a state of
// its own publishes the junction at the branch point, then its own latest
// replay state.
void testOneRequestPublishesJunctionAndLatestReplayState() {
  test::EngineHarness h({.pages = 256, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  Executor &executor = h.model;
  engine::Engine &engine = h.engine;

  engine.submit(request(5, branchPrompt(777)));
  h.runUntilIdle();
  engine.submit(request(6, branchPrompt(888)));
  h.runUntilIdle();

  const DraftContextPlan &plan = executor.plans.at(6);
  const auto snapshot = engine.snapshot();
  require(plan.boundaries.size() == 3 && plan.boundaries[0].boundary == 2048 &&
              plan.boundaries[1].boundary == 2112 &&
              plan.boundaries[2].boundary == 2113,
          "junction, latest replay state, and active end were not ordered");
  require(executor.reservations == 3 && snapshot.junctionMaterializations == 1 &&
              snapshot.replayStatePublications == 2 &&
              snapshot.resources.stateCache.entries == 3,
          "one request did not retain both sparse composite states");
}

// A junction costs a cached state, its capture and a draft window of
// draft context, so it is planned only where it saves a later request at
// least a draft window of prefill: at a branch point that far past the state
// the request resumes from.
void testLazyJunctionNeedsADraftWindowOfGain() {
  test::EngineHarness h({.pages = 256, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  std::vector<uint32_t> shared(2112);
  std::iota(shared.begin(), shared.end(), 1);
  // The shared prefix, then 161 tokens of the conversation's own from
  // `first` on: its replay state lands at 2272.
  const auto conversation = [&](uint32_t first) {
    std::vector<uint32_t> prompt = shared;
    prompt.resize(shared.size() + 161);
    std::iota(prompt.begin() + shared.size(), prompt.end(), first);
    return prompt;
  };
  const std::vector<uint32_t> cached = conversation(10'000);
  engine.submit(request(1, cached));
  h.runUntilIdle();
  engine.submit(request(2, conversation(20'000)));
  h.runUntilIdle();
  require(engine.snapshot().junctionMaterializations == 1,
          "a branch off the cached conversation did not publish its junction");
  const uint32_t restored = executor.restored;
  engine.submit(request(3, conversation(30'000)));
  h.runUntilIdle();
  require(executor.restored == restored + 2112 &&
              events.starts.back() == 2112,
          "a third branch did not resume from the junction");
  // This one follows the cached conversation 96 tokens past the junction,
  // where that one goes on to its state, and then branches.
  std::vector<uint32_t> follower = cached;
  std::iota(follower.begin() + shared.size() + 96, follower.end(), 40'000);
  engine.submit(request(4, follower));
  h.runUntilIdle();
  const auto &boundaries = executor.plans.at(4).boundaries;
  require(events.starts.back() == 2112 &&
              std::none_of(boundaries.begin(), boundaries.end(),
                           [](const DraftBoundaryPlan &boundary) {
                             return boundary.boundary == 2112 + 96;
                           }) &&
              engine.snapshot().junctionMaterializations == 1,
          "a branch point 96 tokens past the resumed state planned a junction");
}

// Buffers refused for the latest replay state come from recycling the least
// recently used cached state, which is an older unrelated state, not the
// junction state this same request published one command earlier.
void testLatestReplayDenialRecyclesOlderStateNotTheJunction() {
  GovernedHost host;
  test::EngineHarness h({.pages = 256,
                         .engine = host.connect({}),
                         .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;

  engine.submit(request(70, std::vector<uint32_t>(65, 7000)));
  h.runUntilIdle();
  engine.submit(request(7, branchPrompt(777)));
  h.runUntilIdle();
  const auto older = resources.snapshot();
  require(older.stateCache.entries == 2 && older.stateCache.evictions == 0,
          "older unrelated state was not left in the cache");

  engine.submit(request(8, branchPrompt(888)));
  require(tickAt(h, 1) && tickAt(h, 2),
          "request did not publish its junction state");
  require(engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 3,
          "lazy junction was not materialized in the first command");
  // The pages this request held for the cached prefix are free; while the
  // engine may not grow, the replay state's buffers take none of them and
  // recycle a state.
  host.pause(true);
  executor.deniedReservations = 1;
  h.runUntilIdle();
  host.pause(false);

  const DraftContextPlan &plan = executor.plans.at(8);
  const auto snapshot = engine.snapshot();
  require(plan.boundaries.size() == 3 && plan.boundaries[0].boundary == 2048 &&
              plan.boundaries[1].boundary == 2112 &&
              plan.boundaries[2].boundary == 2113,
          "the plan did not keep its boundaries");
  require(executor.deniedReservations == 0 &&
              snapshot.recycledStatePublications == 1 &&
              snapshot.replayStatePublications == 3 &&
              snapshot.replayStatePublicationFailures == 0 &&
              snapshot.resources.stateCache.evictions == 1 &&
              snapshot.resources.stateCache.entries == 3,
          "latest-state denial did not recycle exactly one older state");

  // The junction state survived the recycle: another branch resumes from
  // it, while the recycled prompt is back to KV without a state and
  // publishes its replay state again.
  const uint32_t restored = executor.restored;
  engine.submit(request(9, branchPrompt(999)));
  h.runUntilIdle();
  require(executor.restored == restored + 2048 &&
              events.starts.back() == 2048,
          "latest-state denial discarded the independent junction state");
  engine.submit(request(71, std::vector<uint32_t>(65, 7000)));
  h.runUntilIdle();
  require(events.starts.back() == 0 &&
              engine.snapshot().replayStatePublications == 5 &&
              engine.snapshot().junctionMaterializations == 1,
          "recycled state was not the older unrelated one");
}

void testCancellationAfterJunctionDiscardsLaterState() {
  test::EngineHarness h({.pages = 256, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  Executor &executor = h.model;
  engine::Engine &engine = h.engine;

  engine.submit(request(9, branchPrompt(777)));
  h.runUntilIdle();
  engine.submit(request(10, branchPrompt(999)));
  require(tickAt(h, 1) && tickAt(h, 2),
          "request did not publish its first sparse state");
  const auto published = engine.snapshot();
  require(published.resources.stateCache.entries == 2 &&
              published.junctionMaterializations == 1 &&
              executor.reservationAttempts == 2,
          "junction publication did not land in the first command");
  engine.cancel(10);
  h.runUntilIdle();
  // The 2112 boundary is dropped with the lane: no reservation, no failure
  // counted, and the junction state stays cached.
  const auto cancelled = engine.snapshot();
  require(executor.reservationAttempts == 2 &&
              cancelled.resources.stateCache.entries == 2 &&
              cancelled.resources.stateCache.publications == 2 &&
              cancelled.replayStatePublications == 1 &&
              cancelled.replayStatePublicationFailures == 0 &&
              cancelled.cancelled == 1 && executor.requests.empty() &&
              cancelled.resources.activeRequests == 0,
          "cancellation leaked or removed the wrong sparse state");
}

// A shared prefill's junction is an ordinary publication: with no cached
// state to recycle, it makes room from another conversation's cached KV, an
// extent at a time, and the sibling resumes from it rather than recompute
// the shared prefix.
void testSharedJunctionPublishesFromCachedKv() {
  GovernedHost host;
  test::EngineHarness h({.pages = 64, .engine = host.connect({})});
  test::TestKvStorage &storage = h.storage;
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  // Another conversation's KV, with no state, fills two whole extents.
  resources.beginRequest(900);
  require(resources.ensureTokens(900, 256).granted(), "the other conversation got no pages");
  resources.publishCommittedBlocks(900, std::vector<uint32_t>(257, 6), 256, {});
  resources.endRequest(900);
  // A state's buffers need memory that only a released extent gives: until
  // one is released, each asks the engine's limit for one state's buffers,
  // which it leaves no room for.
  const uint32_t releases = storage.releasedExtents;
  executor.reservationRoom = [&] {
    return storage.releasedExtents > releases || static_cast<bool>(host.reserve(64));
  };
  host.leaveRoom(0);
  for (uint32_t id = 1; id <= 2; ++id) {
    std::vector<uint32_t> prompt(193, 7);
    std::fill(prompt.begin() + 160, prompt.end(), id + 10);
    engine.submit(request(id, prompt));
  }
  h.runUntilIdle();
  const auto snapshot = engine.snapshot();
  require(snapshot.junctionMaterializations == 1 &&
              snapshot.junctionMaterializationFailures == 0 && storage.releasedExtents > releases &&
              executor.restored == 160 && executor.prefillRows == 193 + 33 && events.completedCount == 2,
          "the shared junction did not make room from cached KV");
}

// Ticks the harness at 1 to 6 ms: the constrained request's initial mask,
// its first token, then its first cycle's overlapped verify.
void advanceToOverlappedVerify(test::EngineHarness &h, uint64_t requestId) {
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  require(tickAt(h, 1) && tickAt(h, 2), "constrained request did not reach its initial mask");
  require(events.maskRequests.size() == 1 &&
              events.maskRequests[0].first == requestId &&
              events.maskRequests[0].second.empty(),
          "initial constrained mask request is malformed");
  const std::array<uint32_t, 1> initialMask{1};
  engine.provideMask(requestId, initialMask);
  // Its first token is selected, then its first cycle drafts.
  require(tickAt(h, 3) && tickAt(h, 4) && tickAt(h, 5) && tickAt(h, 6),
          "constrained request did not launch overlapped verification");
  require(engine.commandInFlight() && executor.overlap &&
              events.maskRequests.size() == 2 &&
              events.maskRequests[1].first == requestId &&
              events.maskRequests[1].second.size() == 8,
          "target forward did not retain its batch while requesting a mask");
}

// Once its first token is selected, a constrained lane decodes in the command
// of an unconstrained lane of its priority. The command asks for the
// constrained lane's verify mask alone, a mask response for the other lane
// does not reach the model, and the one mask completes the command for both.
void testConstrainedLaneSharesDecodeCommand() {
  test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 2; }});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.decodeFinishes = false;

  engine.submit(constrainedRequest(1, 100'000));
  EngineRequest unconstrained = request(2, {2});
  unconstrained.maxNewTokens = 100;
  unconstrained.deadlineMilliseconds = 100'000;
  engine.submit(std::move(unconstrained));
  double now = 1;
  tickUntil(h, now, [&] { return events.maskRequests.size() == 1; },
            "the constrained prompt did not ask for its first mask");
  const std::array<uint32_t, 1> mask{1};
  engine.provideMask(1, mask);
  tickUntil(h, now, [&] { return events.maskRequests.size() == 2; },
            "the constrained lane's first cycle did not ask for its verify mask");
  require(engine.commandInFlight() && executor.overlap && executor.overlap->requestId == 1 &&
              events.maskRequests[1].first == 1,
          "the shared command did not wait for the constrained lane's mask");
  const uint32_t provided = executor.providedMasks;
  const size_t unconstrainedTokens = events.outputs[2].size();
  engine.provideMask(2, mask);
  require(executor.providedMasks == provided && events.failedCount == 0,
          "a mask for the lane that asked for none reached the model");
  engine.provideMask(1, mask);
  require(tickAt(h, now++) && !engine.commandInFlight() && events.outputs[1] == std::vector<uint32_t>{42} &&
              events.outputs[2].size() > unconstrainedTokens &&
              engine.snapshot().scheduler.decodeBatchesByWidth[1] != 0,
          "the constrained lane's mask did not complete the command for both lanes");
  for (const uint64_t id : {1, 2})
    engine.cancel(id);
  tickUntil(h, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.failedCount == 0, "a lane failed");
}

// A command's engine cycle runs from the previous command's retirement while
// the engine stays busy, so it covers the host work between commands, and
// from the command's plan after a tick that found nothing to do.
void testDecodeCycleCoversHostWork() {
  test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.decodeFinishes = false;

  EngineRequest greedy = request(1, {1});
  greedy.maxNewTokens = 2;
  engine.submit(std::move(greedy));
  // Each command is planned 5 ms after the previous one retires and
  // retires 10 ms after its plan.
  for (double now : {10.0, 20.0, 25.0, 35.0, 40.0, 50.0})
    require(tickAt(h, now), "the greedy request stalled");
  using Cycles = std::vector<std::pair<WorkKind, double>>;
  require(events.completedCount == 1 &&
              events.cycles == Cycles{{WorkKind::Prefill, 10.0},
                                      {WorkKind::Decode, 15.0},
                                      {WorkKind::Decode, 15.0}},
          "back-to-back commands were not timed from retirement to retirement");
  require(!tickAt(h, 60.0), "a finished request kept the engine busy");

  engine.submit(constrainedRequest(2));
  require(tickAt(h, 100.0) && tickAt(h, 110.0) &&
              events.maskRequests.size() == 1 && !tickAt(h, 130.0),
          "a lane waiting for its first mask kept the engine busy");
  const std::array<uint32_t, 1> mask{1};
  engine.provideMask(2, mask);
  require(tickAt(h, 140.0) && tickAt(h, 150.0) &&
              events.cycles.back() == std::pair{WorkKind::Decode, 10.0},
          "a decode planned after an idle tick was not timed from its plan");
}

// A constrained prompt asks for its first mask when its prefill completes
// and takes no decode slot until the mask arrives; its first token is then
// selected by a plan of its own, and drafting starts with the next.
void testConstrainedPrefillRequestsMaskWithoutDecodeSlot() {
  test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;

  engine.submit(constrainedRequest(1));
  require(tickAt(h, 1) && tickAt(h, 2) &&
              events.maskRequests.size() == 1 &&
              events.maskRequests[0].first == 1 &&
              events.maskRequests[0].second.empty(),
          "the prefill completion did not ask for the first mask");
  require(events.cycles == std::vector<std::pair<WorkKind, double>>{
                               {WorkKind::Prefill, 1.0}},
          "the first mask cost a command");
  for (double now = 3; now < 10; ++now)
    require(!tickAt(h, now) && !engine.commandInFlight(),
            "a decode plan ran before the first mask arrived");
  const std::array<uint32_t, 1> mask{1};
  engine.provideMask(1, mask);
  require(tickAt(h, 10) && tickAt(h, 11) && events.emitted == 0 &&
              events.cycles.back().first == WorkKind::Decode &&
              !executor.overlap,
          "the first token's selection drafted or emitted a token");
  require(tickAt(h, 12) && executor.overlap && engine.commandInFlight(),
          "the first draft did not follow the selection");
}

void testConstraintMaskWaitHonorsCancelAndDeadline() {
  {
    test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    engine.submit(constrainedRequest(201));
    advanceToOverlappedVerify(h, 201);
    engine.cancel(201);
    const std::array<uint32_t, 1> lateMask{1};
    engine.provideMask(201, lateMask);
    engine.failRequest(201, LaneOutcome::InvalidMask, "late mask");
    require(executor.overlap->abandoned && tickAt(h, 7) && idle(engine) &&
                events.completedCount == 1 && events.failedCount == 0 &&
                events.emitted == 0,
            "cancelled mask wait left an active scheduler batch");
  }

  {
    test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    engine.submit(constrainedRequest(202, 100.0));
    advanceToOverlappedVerify(h, 202);
    require(tickAt(h, 100.0) && executor.overlap->abandoned &&
                idle(engine) && events.failedCount == 1,
            "deadline did not terminate an in-flight host mask wait");
  }
}

// A verify mask the server leaves unanswered fails its request alone once
// the limit passes: the command commits without the mask, a mask that comes
// after the limit is ignored, and the lane goes to the request waiting for
// it.
void testUnansweredVerifyMaskFailsOnlyItsRequest() {
  test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  engine.submit(constrainedRequest(203, 100'000.0));
  // The verify mask is requested at tick 6.
  advanceToOverlappedVerify(h, 203);
  engine.submit(request(204, {2}));
  static_cast<void>(tickAt(h, 5005.0));
  require(engine.commandInFlight() && !executor.overlap->abandoned &&
              events.failedCount == 0,
          "a verify mask wait ended before its limit");
  // The command still runs after the limit, so a late mask finds the
  // request in flight.
  executor.overlap->held = true;
  require(tickAt(h, 5006.0) && engine.commandInFlight() &&
              executor.overlap->abandoned && events.failedCount == 0,
          "an unanswered verify mask wait did not end at its limit");
  const std::array<uint32_t, 1> lateMask{1};
  engine.provideMask(203, lateMask);
  require(!executor.overlap->provided,
          "a verify mask that came after the limit reached the model");
  executor.overlap->held = false;
  require(tickAt(h, 5007.0) && !engine.commandInFlight() &&
              events.failures == std::vector<std::string>{"mask_timeout"} &&
              events.failureDetails.front().second && events.emitted == 0 &&
              events.completedCount == 0,
          "an unanswered verify mask did not fail its request as retryable");
  require(tickAt(h, 5008.0) && events.startIds.back() == 204,
          "the timed-out request's lane did not go to the waiting request");
  h.runUntilIdle();
  require(events.completedCount == 1 && events.failedCount == 1,
          "the waiting request did not complete after the mask timeout");
}

// An initial mask the server leaves unanswered fails its request when the
// limit passes, which the engine's next wakeup names; an answered mask
// clears the limit.
void testUnansweredInitialMaskFails() {
  for (const bool answered : {false, true}) {
    test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    engine.submit(constrainedRequest(205, 100'000.0));
    require(tickAt(h, 1) && tickAt(h, 2) &&
                events.maskRequests.size() == 1,
            "constrained request did not reach its initial mask");
    require(engine.nextWakeupMilliseconds() == 5002.0,
            "the initial mask wait did not schedule its limit");
    if (answered) {
      const std::array<uint32_t, 1> initialMask{1};
      engine.provideMask(205, initialMask);
      require(engine.nextWakeupMilliseconds() == 100'000.0 &&
                  tickAt(h, 5002.0) && events.failedCount == 0,
              "an answered initial mask kept its limit");
      continue;
    }
    static_cast<void>(tickAt(h, 5001.0));
    require(events.failedCount == 0, "an initial mask wait ended early");
    require(tickAt(h, 5002.0) &&
                events.failures == std::vector<std::string>{"mask_timeout"} &&
                events.failureDetails.front().second &&
                executor.requests.empty() && idle(engine),
            "an unanswered initial mask did not fail its request");
  }
}

void testDecodeNearContextCeilingCoversVerifyRows() {
  test::EngineHarness h(
      {.pages = 8, .engine = {.maxContext = 64}, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.decodeFinishes = false;

  // prompt + max_tokens fill a context that is a whole number of pages; the
  // last decode cycles store verify rows past the logical ceiling.
  EngineRequest value = request(10, std::vector<uint32_t>(57, 10));
  value.maxNewTokens = 7;
  engine.submit(value);
  h.runUntilIdle();
  require(events.completedCount == 1 && events.failedCount == 0 &&
              events.emitted == 7,
          "decode at the context ceiling lost its verify-row page coverage");
}

void testExpiredMaskWaitFinalizesWhileAnotherCommandRuns() {
  test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 2; }});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;

  engine.submit(constrainedRequest(300, 100.0));
  require(tickAt(h, 1) && tickAt(h, 2),
          "constrained request did not reach its initial mask wait");
  require(events.maskRequests.size() == 1, "initial mask was not requested");

  executor.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
  engine.submit(request(301, {2}));
  require(tickAt(h, 5) && tickAt(h, 6) && tickAt(h, 7) &&
              engine.commandInFlight(),
          "peer decode command was not held in flight");

  // The waiting request expires while the peer command is still running. It
  // must be finalized now; its late mask then finds no live request.
  require(tickAt(h, 150.0) && events.failedCount == 1 &&
              engine.commandInFlight(),
          "expired mask wait was not finalized behind an in-flight command");
  *executor.holdDecodeUntil = true;
  for (double now = 151.0; now < 160.0 && !idle(engine); ++now)
    static_cast<void>(tickAt(h, now));
  require(idle(engine) && events.completedCount == 1 &&
              events.failedCount == 1,
          "peer request did not complete after the expired wait was removed");
}

void testOrdinaryInFlightDeadlineDrainsWithoutPublishingOrOutput() {
  for (WorkKind heldKind : {WorkKind::Prefill, WorkKind::Decode}) {
    test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
    engine::Cache &resources = h.cache;
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    auto released = std::make_shared<std::atomic<bool>>(false);
    if (heldKind == WorkKind::Prefill)
      executor.holdPrefillUntil = released;
    else
      executor.holdDecodeUntil = released;
    auto value = request(310, std::vector<uint32_t>(65, 310));
    value.deadlineMilliseconds = 100;
    engine.submit(std::move(value));
    // One command prefills the prompt, capturing its replay point at 64.
    const uint32_t launchTick = heldKind == WorkKind::Prefill ? 1 : 3;
    for (uint32_t tick = 1; tick <= launchTick; ++tick)
      require(tickAt(h, tick), "request did not launch the held command");
    require(engine.commandInFlight(), "test command did not remain in flight");
    const auto before = resources.snapshot();
    require(tickAt(h, 100) && engine.commandInFlight() &&
                executor.requests.size() == 1 && events.failedCount == 0 &&
                resources.snapshot().pool.pagesActive == before.pool.pagesActive,
            "ordinary deadline released resources owned by in-flight Metal");
    require(engine.nextWakeupMilliseconds() == 1100.0 && !tickAt(h, 101),
            "expired in-flight request lost its bounded health wakeup");
    engine.cancel(310); // A later cancel must not replace the deadline failure.
    *released = true;
    // A held prefill expires with the state it captured, which goes back to
    // the pool; a held decode follows the prefill that published it.
    require(tickAt(h, 102) && idle(engine) && events.failedCount == 1 &&
                events.failures.front() == "deadline_exceeded" &&
                events.failureDetails.front().first ==
                    "request deadline exceeded" &&
                events.completedCount == 0 && events.emitted == 0 &&
                executor.captures.size() == 1 && executor.requests.empty(),
            "expired ordinary command emitted output or lost its deadline");
    const auto after = resources.snapshot();
    require(after.stateCache.publications == before.stateCache.publications &&
                after.pool.pagesPrefix == before.pool.pagesPrefix &&
                after.pool.pagesActive == 0 && after.activeRequests == 0,
            "expired ordinary command published cache state or leaked resources");
  }
}

void testStalledSuspensionFailsWithCapacity() {
  test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 2; }});
  test::TestKvStorage &storage = h.storage;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.unblockGrowthOnSuspend = false;

  storage.growthBlocked = true;
  engine.submit(request(60, {60}));
  engine.submit(request(61, {61}));
  require(tickAt(h, 1.0) && tickAt(h, 2.0) && executor.suspensions == 1 &&
              events.capacityExhaustedCount == 1,
          "preemption did not identify that the remaining singleton cannot fit");

  // Nothing completes between suspension and retry, so suspending again would
  // only rotate the same two lanes forever: the suspended lane fails at its
  // resume.
  for (double now = 102.0; now < 140.0 && !idle(engine); ++now)
    static_cast<void>(tickAt(h, now));
  require(idle(engine) && events.capacityExhaustedCount == 2 &&
              engine.snapshot().resourceSuspensions == 1 &&
              engine.snapshot().resourceResumptions == 0 && executor.prefillRows == 0,
          "stalled suspension did not converge to a capacity failure");
  const std::string &message = events.failureDetails.back().first;
  require(message.find(", a Resident lane of ") != std::string::npos &&
              message.find(" tokens at its resume, ") != std::string::npos,
          "the suspended lane did not fail naming its lane at its resume: " + message);
}

void testTerminalAnchorWithoutKvIsNotCached() {
  for (uint32_t withoutKv : {1U, 0U}) {
    test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
    engine::Cache &resources = h.cache;
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    executor.decodeTokensWithoutKv = withoutKv;

    // 31 prompt tokens plus the emitted token complete one Page32 block only
    // when that token has a stored KV row.
    engine.submit(request(70, std::vector<uint32_t>(31, 70)));
    h.runUntilIdle();
    require(events.completedCount == 1 && events.emitted == 1,
            "terminal anchor request did not complete");
    require(resources.snapshot().pool.pagesPrefix == (withoutKv ? 0U : 1U),
            "token without a KV row changed the cached block count");
  }
}

// An unconstrained request's first token reaches the client when the prefill
// selects it, before the first decode cycle that commits it; that cycle's
// event leaves it out, and the usage counts what the client got, also when the
// request is cancelled before that cycle.
void testFirstTokenIsSentAtPrefill() {
  for (const bool cancel : {false, true}) {
    test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    executor.firstTokens = true;
    executor.decodeFinishes = false;
    executor.decodeTokens = 4;
    executor.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);

    EngineRequest value = request(91, std::vector<uint32_t>(40, 91));
    value.maxNewTokens = 12;
    engine.submit(std::move(value));
    // The first cycle is held in flight: the client has the first token.
    double now = 1;
    tickUntil(h, now, [&] { return !events.outputs[91].empty() && engine.commandInFlight(); },
              "the first cycle did not start after the prompt");
    require(events.outputs[91] == std::vector<uint32_t>{42},
            "the first token did not reach the client before the first cycle");
    if (cancel)
      engine.cancel(91);
    *executor.holdDecodeUntil = true;
    tickUntil(h, now, [&] { return idle(engine); }, "the request did not end");
    const uint32_t expected = cancel ? 1 : 12;
    require(events.completedCount == 1 && events.outputs[91].size() == expected &&
                events.usage[91] == std::pair<uint32_t, uint32_t>{40, expected},
            cancel ? "a cancelled request's usage left out the first token it was sent"
                   : "the stream and the usage disagree on the tokens sent");
  }
}

// A request suspended after its first token was sent and before its first
// decode cycle keeps that token pending: it resumes with its prompt alone, and
// the cycle that commits the token does not send it again. /status counts the
// token meanwhile.
void testFirstTokenSentBeforeASuspensionIsNotSentAgain() {
  test::EngineHarness h({.pages = 512});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.firstTokens = true;
  executor.decodeFinishes = false;
  executor.decodeTokens = 4;
  // A Normal start is refused memory while the Background lane is resident,
  // which suspends that lane for it.
  executor.beginGrowthBlocked = [&] {
    return executor.lastBeginId == 2 && executor.requests.at(1).resident;
  };

  const std::vector<uint32_t> prompt(40, 1);
  auto background = request(1, prompt);
  background.priority = RequestPriority::Background;
  background.maxNewTokens = 12;
  engine.submit(std::move(background));
  double now = 1;
  tickUntil(h, now, [&] { return events.outputs.contains(1); },
            "the first token was not sent at the prefill");
  auto normal = request(2, std::vector<uint32_t>(9, 2));
  normal.maxNewTokens = 4;
  engine.submit(std::move(normal));
  tickUntil(h, now, [&] { return !executor.requests.at(1).resident; },
            "the Background lane was not suspended");
  const EngineSnapshot suspended = engine.snapshot();
  require(events.outputs.at(1) == std::vector<uint32_t>{42} &&
              suspended.activeRequests.front().id == 1 &&
              suspended.activeRequests.front().generatedTokens == 1,
          "the suspended lane did not keep the first token it was sent");
  tickUntil(h, now, [&] { return idle(engine); }, "the requests did not end");
  require(executor.resumedPrompts == std::vector<std::vector<uint32_t>>{prompt} &&
              events.completedCount == 2 && events.failedCount == 0 &&
              events.outputs.at(1) == std::vector<uint32_t>(12, 42) &&
              events.usage.at(1) == std::pair<uint32_t, uint32_t>{40, 12},
          "the resumed lane replayed its first token or sent it again");
}

// A lane suspended in its decode takes new pages past the chain it restores
// when it resumes, which its replay writes in chunks that may end inside any
// of them: they all come cleared, so rows not yet written read zero whoever
// held them before.
void testResumedLaneClearsThePagesPastItsRestoredChain() {
  test::EngineHarness h({.pages = 512});
  test::TestKvStorage &storage = h.storage;
  engine::Cache &resources = h.cache;
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.firstTokens = true;
  executor.decodeFinishes = false;
  executor.decodeTokens = 4;
  executor.beginGrowthBlocked = [&] {
    return executor.lastBeginId == 2 && executor.requests.at(1).resident;
  };

  auto background = request(1, std::vector<uint32_t>(40, 1));
  background.priority = RequestPriority::Background;
  background.maxNewTokens = 80;
  engine.submit(std::move(background));
  double now = 1;
  tickUntil(h, now,
            [&] { return events.outputs.contains(1) && events.outputs.at(1).size() >= 41; },
            "the lane did not decode");
  auto normal = request(2, std::vector<uint32_t>(9, 2));
  normal.maxNewTokens = 4;
  engine.submit(std::move(normal));
  tickUntil(h, now, [&] { return !executor.requests.at(1).resident; },
            "the Background lane was not suspended");
  const size_t cleared = storage.cleared.size();
  tickUntil(h, now, [&] { return executor.requests.at(1).resident; },
            "the Background lane did not resume");
  const auto replayEnd = static_cast<uint32_t>(executor.resumedPrompts.at(0).size());
  const PageTableState table = resources.pageTable(1);
  const std::vector<uint32_t> pages(table.pages.begin(), table.pages.end());
  const uint32_t restored = executor.lastRestoredTokens / KvCache::pageTokens;
  require(replayEnd % KvCache::pageTokens && replayEnd / KvCache::pageTokens < pages.size() &&
              restored + 1 < pages.size() && restored <= replayEnd / KvCache::pageTokens,
          "fixture did not take pages past its restored chain");
  for (size_t page = restored; page < pages.size(); ++page)
    require(std::find(storage.cleared.begin() + cleared, storage.cleared.end(), pages[page]) !=
                storage.cleared.end(),
            "a page past the resumed lane's restored chain was not cleared");
  tickUntil(h, now, [&] { return idle(engine); }, "the requests did not end");
  require(events.completedCount == 2 && events.failedCount == 0,
          "the resumed lane did not complete");
}

// A lane suspended after it published a block past its latest state replays
// that block when it resumes and finds it cached: its own, not counted as KV
// another command computed.
void testReplayFindsItsOwnBlocks() {
  test::EngineHarness h({.pages = 512});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.firstTokens = true;
  executor.decodeFinishes = false;
  executor.decodeTokens = 4;
  executor.beginGrowthBlocked = [&] {
    return executor.lastBeginId == 2 && executor.requests.at(1).resident;
  };

  const std::vector<uint32_t> prompt(40, 1);
  auto background = request(1, prompt);
  background.priority = RequestPriority::Background;
  background.maxNewTokens = 40;
  engine.submit(std::move(background));
  double now = 1;
  // Its KV reaches past the second block; its latest state is at the first.
  tickUntil(h, now,
            [&] { return events.outputs.contains(1) && events.outputs.at(1).size() >= 28; },
            "the lane did not decode past its second block");
  auto normal = request(2, std::vector<uint32_t>(9, 2));
  normal.maxNewTokens = 4;
  engine.submit(std::move(normal));
  tickUntil(h, now, [&] { return !executor.requests.at(1).resident; },
            "the Background lane was not suspended");
  tickUntil(h, now, [&] { return idle(engine); }, "the requests did not end");
  require(executor.lastRestoredTokens == KvCache::pageTokens &&
              executor.resumedPrompts.size() == 1 &&
              executor.resumedPrompts.front().size() > 2 * KvCache::pageTokens,
          "the lane did not replay its second block");
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.usage.at(1) == std::pair<uint32_t, uint32_t>{40, 40},
          "the resumed lane did not complete");
  require(events.deduplicatedTokens.at(1) == 0 && engine.snapshot().deduplicatedKvTokens == 0,
          "a replay counted the request's own block as another's");
}

// A lane suspended in its prefill resumes from a state another request
// cached past the blocks it published: it reads the KV of the blocks it
// restores there, which the other request computed, and counts it though it
// started cold.
void testResumePastItsBlocksCountsAnotherRequestsKv() {
  test::EngineHarness h({.pages = 1024});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.beginGrowthBlocked = [&] {
    return executor.lastBeginId == 2 && executor.requests.at(1).resident;
  };

  std::vector<uint32_t> prompt(8000);
  std::iota(prompt.begin(), prompt.end(), 1);
  auto background = request(1, prompt);
  background.priority = RequestPriority::Background;
  engine.submit(std::move(background));
  double now = 1;
  tickUntil(h, now,
            [&] { return !engine.commandInFlight() && executor.prefillRows == 2048; },
            "the lane did not prefill its first chunk");
  // A Normal request of the same prompt suspends the lane, which resumes
  // only once that request has computed the prompt and completed.
  executor.resumeDenied = true;
  engine.submit(request(2, prompt));
  tickUntil(h, now, [&] { return events.usage.contains(2); },
            "the Normal request did not complete");
  executor.resumeDenied = false;
  tickUntil(h, now, [&] { return idle(engine); }, "the lane did not complete");
  const uint32_t replayPoint = (8000 - 1) / KvCache::pageTokens * KvCache::pageTokens;
  require(executor.suspensions == 1 && executor.resumptions == 1 &&
              executor.lastRestoredTokens == replayPoint &&
              events.starts == std::vector<uint32_t>{0, 0} && events.completedCount == 2 &&
              events.failedCount == 0,
          "the lane did not resume from the Normal request's replay point");
  // The Normal request read the lane's first chunk. The lane read the blocks
  // the Normal request computed past it: those it restored, and the last
  // one, which it computed and then found cached.
  require(events.deduplicatedTokens.at(2) == 2048 &&
              events.deduplicatedTokens.at(1) == prompt.size() - 2048 &&
              engine.snapshot().deduplicatedKvTokens == prompt.size(),
          "a resume did not count the blocks another request computed");
}

void testPrefillCanCompleteTheRequest() {
  for (bool stop : {false, true}) {
    test::EngineHarness h({.pages = 8, .shape = [](Executor &shape) { shape.maximumLanes = 1; }});
    Executor &executor = h.model;
    Events &events = h.events;
    engine::Engine &engine = h.engine;
    executor.prefillAnchor = stop;

    EngineRequest value = request(90, std::vector<uint32_t>(40, 90));
    value.maxNewTokens = stop ? 8 : 1;
    engine.submit(std::move(value));
    h.runUntilIdle();
    require(events.completedCount == 1 && events.emitted == 1 &&
                events.failedCount == 0,
            "request finished by prefill did not complete cleanly");
    require(engine.snapshot().scheduler.decodeBatchesByWidth[0] == 0,
            "request finished by prefill was decoded");
  }
}

void testOutOfVocabularyOutputFailsLaneOnly() {
  test::EngineHarness h({.pages = 64, .engine = {.vocabularySize = 1000}});
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  // The sampling kernels leave 0xffffffff when a logit row is entirely
  // non-finite; the engine must fail that lane before the sentinel reaches
  // the token history while the peer request completes normally.
  model.poisonRequest = 1;
  model.poisonToken = 0xFFFFFFFFU;
  engine.submit(request(1, {7, 7, 7}));
  engine.submit(request(2, {7, 7, 7}));
  h.runUntilIdle();
  require(events.failedCount == 1,
          "out-of-vocabulary model output did not fail the lane");
  require(events.failures.size() == 1 &&
              events.failures[0] == "model_result_invalid",
          "out-of-vocabulary failure carried the wrong code");
  require(events.completedCount == 1,
          "poisoned lane took down the rest of the batch");
  require(events.outputs[1].empty() &&
              events.outputs[2] == std::vector<uint32_t>{42},
          "out-of-vocabulary token reached the event sink");
  require(events.usage.count(2) == 1,
          "clean peer request did not complete");
}

// A damaged state is one way to a result that is not finite: a request whose
// first step from the state it restored is not finite takes the state out of
// the cache, and the next request with the prompt computes it again. The KV
// blocks below it and what the step published before stay.
void testNonFiniteFirstStepAfterARestoreInvalidatesTheState() {
  test::EngineHarness h({.pages = 32});
  engine::Cache &cache = h.cache;
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  std::vector<uint32_t> prompt(129);
  std::iota(prompt.begin(), prompt.end(), 1u);

  // A first turn publishes the state at the prompt's second page.
  engine.submit(request(1, std::vector<uint32_t>(prompt.begin(), prompt.begin() + 65)));
  h.runUntilIdle();
  model.nonFinitePrefills = {2};
  engine.submit(request(2, prompt));
  h.runUntilIdle();
  require(events.starts == std::vector<uint32_t>{0, 64} &&
              events.failures == std::vector<std::string>{"model_result_invalid"},
          "the request did not resume from the state and fail its first step");
  const CacheLookup lookup = cache.lookup(prompt, {});
  require(lookup.kvBoundary == 64 && !lookup.state &&
              cache.snapshot().stateCache.invalidations == 1,
          "the state the failed step computed from stayed, or the blocks below it went");

  model.nonFinitePrefills.clear();
  engine.submit(request(3, prompt));
  h.runUntilIdle();
  require(events.starts.back() == 0 && events.completedCount == 2,
          "the next request resumed from the state a failed step computed from");
}

// A step from the restored state that was finite proves it: a later step
// that is not finite leaves it, and the next request resumes from it.
void testNonFiniteLaterStepKeepsTheRestoredState() {
  test::EngineHarness h({.pages = 32, .engine = {.vocabularySize = 1000}});
  engine::Cache &cache = h.cache;
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  std::vector<uint32_t> prompt(129);
  std::iota(prompt.begin(), prompt.end(), 1u);

  engine.submit(request(1, std::vector<uint32_t>(prompt.begin(), prompt.begin() + 65)));
  h.runUntilIdle();
  // Its prompt's steps from the state succeed; its first decode fails.
  model.poisonRequest = 2;
  model.poisonToken = 0xFFFFFFFFU;
  engine.submit(request(2, prompt));
  h.runUntilIdle();
  require(events.starts == std::vector<uint32_t>{0, 64} &&
              events.failures == std::vector<std::string>{"model_result_invalid"},
          "the request did not resume from the state and fail in its decode");
  require(cache.lookup(prompt, {}).resumeBoundary() == 128 &&
              cache.snapshot().stateCache.invalidations == 0,
          "a failure after a finite step from the state took a state out of the cache");
}

// A result that is not finite can drain after a cancellation has already
// ended its request: the cancellation stands as the end, and the state the
// failed step computed from goes all the same.
void testNonFiniteStepAfterCancellationInvalidatesTheState() {
  test::EngineHarness h({.pages = 32});
  engine::Cache &cache = h.cache;
  Executor &model = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  std::vector<uint32_t> prompt(129);
  std::iota(prompt.begin(), prompt.end(), 1u);

  engine.submit(request(1, std::vector<uint32_t>(prompt.begin(), prompt.begin() + 65)));
  h.runUntilIdle();
  model.nonFinitePrefills = {2};
  model.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
  engine.submit(request(2, prompt));
  for (double now = 1; now < 100 && !engine.commandInFlight(); ++now)
    static_cast<void>(tickAt(h, now));
  require(engine.commandInFlight(), "the resumed request did not launch its held prefill");
  engine.cancel(2);
  *model.holdPrefillUntil = true;
  h.runUntilIdle();
  require(engine.snapshot().cancelled == 1 && events.failedCount == 0,
          "the cancellation did not stand as the request's end");
  require(!cache.lookup(prompt, {}).state && cache.snapshot().stateCache.invalidations == 1,
          "a cancelled request's step that was not finite left the state it computed from");
}

// The engine refuses a config the bootstrap has not completed (no context
// window, vocabulary or governor) or whose decode share is negative or not finite.
void testInvalidConfigIsRefused() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  const auto refused = [&](const auto &change, std::string_view error) {
    EngineConfig config = test::engineConfig();
    change(config);
    rejects([&] { engine::Engine(config, resources, executor, events); }, error,
            "an invalid engine config was accepted");
  };
  refused([](EngineConfig &config) { config.maxContext = 0; },
          "context and vocabulary sizes must be positive");
  refused([](EngineConfig &config) { config.vocabularySize = 0; },
          "context and vocabulary sizes must be positive");
  refused([](EngineConfig &config) { config.governor = nullptr; }, "the engine needs its memory governor");
  refused([](EngineConfig &config) { config.spill.capable = true; },
          "a target that can spill needs its page and slot sizes");
  for (const double share : {-0.5, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    refused([share](EngineConfig &config) { config.decodeShare = share; },
            "decode share must be nonnegative and finite");
}

void testSpeculativeCountsStayWithEachConcurrentRequest() {
  test::EngineHarness h({.pages = 128});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.decodeFinishes = false;
  executor.speculativeCounts = {{1, {7, 2}}, {2, {7, 6}}};
  auto prose = request(1, std::vector<uint32_t>(65, 11));
  prose.maxNewTokens = 6;
  auto arithmetic = request(2, std::vector<uint32_t>(65, 12));
  arithmetic.maxNewTokens = 21;
  engine.submit(std::move(prose));
  engine.submit(std::move(arithmetic));
  h.runUntilIdle();
  require(events.speculativeCounts.at(1) == SpeculativeCounts{14, 4, 2} &&
              events.speculativeCounts.at(2) == SpeculativeCounts{21, 18, 3} &&
              events.draftedTokens == 14 + 21 &&
              events.acceptedDraftTokens == 4 + 18 &&
              events.outputs.at(1).size() == 6 && events.outputs.at(2).size() == 21,
          "concurrent requests mixed or dropped their speculative counts");
  auto score = request(3, std::vector<uint32_t>(65, 13));
  score.maxNewTokens = 0;
  score.scoreTokens = {1, 2};
  engine.submit(std::move(score));
  h.runUntilIdle();
  require(events.speculativeCounts.at(3) == SpeculativeCounts{},
          "score-only request inherited speculative generation activity");
}

void testImpossibleSpeculativeCountsFailOnlyTheirOwnLane() {
  test::EngineHarness h({.pages = 128});
  Executor &executor = h.model;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  executor.speculativeCounts = {{1, {7, 9}}, {2, {7, 1}}};
  engine.submit(request(1, std::vector<uint32_t>(65, 11)));
  auto healthy = request(2, std::vector<uint32_t>(65, 12));
  healthy.maxNewTokens = 2;
  engine.submit(std::move(healthy));
  h.runUntilIdle();
  require(events.failedCount == 1 && events.completedCount == 1 &&
              !events.speculativeCounts.contains(1) &&
              events.speculativeCounts.at(2) == SpeculativeCounts{7, 1, 1},
          "invalid speculative counts poisoned another request's completion");
}

// A client stop finishes a request that has output as Stop, counted
// completed; one that has generated nothing yet is cancelled; a repeated or
// unknown id is ignored.
void testStopFinishesARequestAsStopped() {
  test::EngineHarness h;
  h.model.decodeFinishes = false;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  EngineRequest generating = h.request(1, std::vector<uint32_t>(40, 3));
  generating.maxNewTokens = 64;
  engine.submit(generating);
  h.runUntil([&] { return !events.outputs[1].empty(); }, "the request produced no output", 1000);

  engine.stop(1);
  engine.submit(h.request(2, std::vector<uint32_t>(40, 5)));
  engine.stop(2);
  h.runUntilIdle(1000);

  require(events.reasons.at(1) == EngineFinishReason::Stop &&
              events.usage.at(1).second < generating.maxNewTokens,
          "a stopped request did not finish early as Stop");
  require(events.reasons.at(2) == EngineFinishReason::Cancelled,
          "a request stopped before any output was not cancelled");
  const EngineSnapshot counts = engine.snapshot();
  require(counts.completed == 1 && counts.cancelled == 1 && counts.failed == 0,
          "a stopped request was not counted completed");

  engine.stop(1);
  engine.stop(999);

  const EngineSnapshot after = engine.snapshot();
  require(events.completedCount == 2 && events.failedCount == 0 && after.completed == 1 &&
              after.cancelled == 1 && after.failed == 0,
          "a stop of a finished or unknown id was not ignored");
}

// A stop of a lane inside the in-flight command waits for the command to
// drain, then finishes as Stop without emitting that command's output.
void testStopInFlightFinishesAfterTheCommand() {
  test::EngineHarness h;
  h.model.decodeFinishes = false;
  Events &events = h.events;
  engine::Engine &engine = h.engine;
  EngineRequest generating = h.request(1, std::vector<uint32_t>(40, 3));
  generating.maxNewTokens = 64;
  engine.submit(generating);
  h.runUntil([&] { return !events.outputs[1].empty(); }, "the request produced no output", 1000);
  h.model.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
  h.runUntil([&] { return engine.commandInFlight(); }, "the request did not launch a held decode", 1000);
  const size_t emittedBefore = events.outputs.at(1).size();

  engine.stop(1);
  static_cast<void>(h.tick());
  require(engine.commandInFlight() && events.completedCount == 0,
          "an in-flight stop finished before its command drained");
  *h.model.holdDecodeUntil = true;
  h.runUntilIdle(1000);

  const EngineSnapshot counts = engine.snapshot();
  require(events.reasons.at(1) == EngineFinishReason::Stop && counts.completed == 1 &&
              counts.cancelled == 0 && events.outputs.at(1).size() == emittedBefore,
          "an in-flight stop did not finish as Stop after its command");
}

} // namespace

std::span<const test::TestCase> requestsTests() {
  static constexpr test::TestCase kTests[] = {
      {"testSpeculativeCountsStayWithEachConcurrentRequest", testSpeculativeCountsStayWithEachConcurrentRequest},
      {"testImpossibleSpeculativeCountsFailOnlyTheirOwnLane", testImpossibleSpeculativeCountsFailOnlyTheirOwnLane},
      {"testUnusableThinkingBudgetIsRefused", testUnusableThinkingBudgetIsRefused},
      {"testConcurrentColdPrefixesComputeOnce", testConcurrentColdPrefixesComputeOnce},
      {"testColdRequestReportsTheBlocksItReadFromTheCache", testColdRequestReportsTheBlocksItReadFromTheCache},
      {"testBlocksFoundCachedAfterTheLastCommandAreNotReported",
       testBlocksFoundCachedAfterTheLastCommandAreNotReported},
      {"testCacheSaltsShareNothing", testCacheSaltsShareNothing},
      {"testSharedPrefillRebuildsTheMissingJunctionOnce", testSharedPrefillRebuildsTheMissingJunctionOnce},
      {"testSharedPrefillReleasesDifferentJunctionsIndependently",
       testSharedPrefillReleasesDifferentJunctionsIndependently},
      {"testSharedPrefillEvictedPublicationFallsBack", testSharedPrefillEvictedPublicationFallsBack},
      {"testSharedPrefillProducerFailureReleasesWaiters", testSharedPrefillProducerFailureReleasesWaiters},
      {"testSharedPrefillWaiterCancellationAndDeadline", testSharedPrefillWaiterCancellationAndDeadline},
      {"testSharedPrefillFailedPublicationFallsBack", testSharedPrefillFailedPublicationFallsBack},
      {"testSharedPrefillDoesNotBlockUnrelatedWork", testSharedPrefillDoesNotBlockUnrelatedWork},
      {"testSharedPrefillHonorsPriorityAndLateArrival", testSharedPrefillHonorsPriorityAndLateArrival},
      {"testLateSharedPrefillExtendsTheProducerPlan", testLateSharedPrefillExtendsTheProducerPlan},
      {"testLateSiblingBoundaryNeedsAWholeDraftWindow", testLateSiblingBoundaryNeedsAWholeDraftWindow},
      {"testLateSiblingJunctionAtOrPastAPlannedCheckpoint", testLateSiblingJunctionAtOrPastAPlannedCheckpoint},
      {"testSharedJunctionAtACheckpointIsReusable", testSharedJunctionAtACheckpointIsReusable},
      {"testSharedPrefillCapacityFailureDoesNotDeadlock", testSharedPrefillCapacityFailureDoesNotDeadlock},
      {"testSiblingFailedInTheAdmissionPassGetsNoJunction", testSiblingFailedInTheAdmissionPassGetsNoJunction},
      {"testInvalidConfigIsRefused", testInvalidConfigIsRefused},
      {"testColdPublishesReplayStateAndRebuildsALostOne", testColdPublishesReplayStateAndRebuildsALostOne},
      {"testConcurrentDuplicateStateIsCapturedOnce", testConcurrentDuplicateStateIsCapturedOnce},
      {"testReplayStateEndsBeforeTheGenerationPrompt", testReplayStateEndsBeforeTheGenerationPrompt},
      {"testFollowUpResumesBeforeTheGenerationPrompt", testFollowUpResumesBeforeTheGenerationPrompt},
      {"testRetryPublishesNoStateInsideTheGenerationPrompt", testRetryPublishesNoStateInsideTheGenerationPrompt},
      {"testSharedJunctionEndsBeforeTheGenerationPrompt", testSharedJunctionEndsBeforeTheGenerationPrompt},
      {"testImageSpansKeyPrefixIdentity", testImageSpansKeyPrefixIdentity},
      {"testSharedPrefillBoundaryStopsAtTheFirstDifferentImage",
       testSharedPrefillBoundaryStopsAtTheFirstDifferentImage},
      {"testOneRequestPublishesJunctionAndLatestReplayState", testOneRequestPublishesJunctionAndLatestReplayState},
      {"testLazyJunctionNeedsADraftWindowOfGain", testLazyJunctionNeedsADraftWindowOfGain},
      {"testLatestReplayDenialRecyclesOlderStateNotTheJunction",
       testLatestReplayDenialRecyclesOlderStateNotTheJunction},
      {"testCancellationAfterJunctionDiscardsLaterState", testCancellationAfterJunctionDiscardsLaterState},
      {"testSharedJunctionPublishesFromCachedKv", testSharedJunctionPublishesFromCachedKv},
      {"testConstrainedLaneSharesDecodeCommand", testConstrainedLaneSharesDecodeCommand},
      {"testConstraintMaskWaitHonorsCancelAndDeadline", testConstraintMaskWaitHonorsCancelAndDeadline},
      {"testUnansweredVerifyMaskFailsOnlyItsRequest", testUnansweredVerifyMaskFailsOnlyItsRequest},
      {"testUnansweredInitialMaskFails", testUnansweredInitialMaskFails},
      {"testDecodeCycleCoversHostWork", testDecodeCycleCoversHostWork},
      {"testConstrainedPrefillRequestsMaskWithoutDecodeSlot", testConstrainedPrefillRequestsMaskWithoutDecodeSlot},
      {"testDecodeNearContextCeilingCoversVerifyRows", testDecodeNearContextCeilingCoversVerifyRows},
      {"testExpiredMaskWaitFinalizesWhileAnotherCommandRuns", testExpiredMaskWaitFinalizesWhileAnotherCommandRuns},
      {"testOrdinaryInFlightDeadlineDrainsWithoutPublishingOrOutput",
       testOrdinaryInFlightDeadlineDrainsWithoutPublishingOrOutput},
      {"testStalledSuspensionFailsWithCapacity", testStalledSuspensionFailsWithCapacity},
      {"testTerminalAnchorWithoutKvIsNotCached", testTerminalAnchorWithoutKvIsNotCached},
      {"testPrefillCanCompleteTheRequest", testPrefillCanCompleteTheRequest},
      {"testFirstTokenIsSentAtPrefill", testFirstTokenIsSentAtPrefill},
      {"testFirstTokenSentBeforeASuspensionIsNotSentAgain", testFirstTokenSentBeforeASuspensionIsNotSentAgain},
      {"testResumedLaneClearsThePagesPastItsRestoredChain", testResumedLaneClearsThePagesPastItsRestoredChain},
      {"testReplayFindsItsOwnBlocks", testReplayFindsItsOwnBlocks},
      {"testResumePastItsBlocksCountsAnotherRequestsKv", testResumePastItsBlocksCountsAnotherRequestsKv},
      {"testOutOfVocabularyOutputFailsLaneOnly", testOutOfVocabularyOutputFailsLaneOnly},
      {"testNonFiniteFirstStepAfterARestoreInvalidatesTheState",
       testNonFiniteFirstStepAfterARestoreInvalidatesTheState},
      {"testNonFiniteLaterStepKeepsTheRestoredState", testNonFiniteLaterStepKeepsTheRestoredState},
      {"testNonFiniteStepAfterCancellationInvalidatesTheState", testNonFiniteStepAfterCancellationInvalidatesTheState},
      {"testStopFinishesARequestAsStopped", testStopFinishesARequestAsStopped},
      {"testStopInFlightFinishesAfterTheCommand", testStopInFlightFinishesAfterTheCommand},
  };
  return kTests;
}
