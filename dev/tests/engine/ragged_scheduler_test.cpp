#include "TestChecks.hpp"
#include "engine/Scheduler.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

engine::RequestSpec
request(uint64_t id, uint32_t prompt, bool constrained = false,
        RequestPriority priority = RequestPriority::Normal) {
  return {id, priority, constrained, prompt, 10'000.0};
}

void completePrefill(engine::Scheduler &scheduler,
                     const BatchPlan &plan, double wallMilliseconds = 0.0,
                     bool representativePrefillTiming = true) {
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items) {
    results.push_back(
        {item.requestId, item.tokenCount, false, DecodeStage::Regular});
  }
  scheduler.commit(plan, {});
  scheduler.complete(plan, results, wallMilliseconds,
                      representativePrefillTiming);
}

void completeDecode(engine::Scheduler &scheduler, bool finished = false,
                    double wallMilliseconds = 0.0) {
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Decode, "expected a decode command");
  scheduler.commit(plan, {});
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items)
    results.push_back({item.requestId, 0, finished, DecodeStage::Regular});
  scheduler.complete(plan, results, wallMilliseconds, true);
}

// Prefills a constrained request's prompt, whose end waits for the mask of
// its first token.
void awaitFirstMask(engine::Scheduler &scheduler, uint64_t id) {
  const BatchPlan prompt = *scheduler.next({});
  require(prompt.kind == WorkKind::Prefill && prompt.width() == 1 &&
              prompt.items[0].requestId == id,
          "expected the constrained prompt");
  scheduler.commit(prompt, {});
  const std::array result{StepResult{id, prompt.items[0].tokenCount, false,
                                     DecodeStage::ApplyInitialMask}};
  scheduler.complete(prompt, result, 0.0, true);
  require(scheduler.phase(id) == engine::Phase::WaitingMask,
          "a constrained prompt did not wait for its first mask");
}

void testAdmissionSharesDispatchOrderAndBudget() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 8193));
  scheduler.resourcesReady(1, 0);
  scheduler.submit(request(2, 8193));
  scheduler.submit(request(3, 4097));
  scheduler.submit(request(4, 65));
  const std::array candidates{PrefillAdmission{2, 0},
                              PrefillAdmission{3, 4096},
                              PrefillAdmission{4, 0}};
  require(scheduler.prefillAdmissionOrder(candidates) ==
              std::vector<uint64_t>({3, 4}),
          "admission did not pack cached and short work ahead of cold work");
  scheduler.resourcesReady(3, 4096);
  scheduler.resourcesReady(4, 0);
  const auto plan = *scheduler.next({});
  require(plan.items.size() == 3 && plan.items[0].requestId == 3 &&
              plan.items[1].requestId == 4 && plan.items[2].requestId == 1 &&
              plan.items[2].tokenCount == 1982,
          "dispatch disagreed with admission work accounting");

  Scheduler shortPrompts(0.0);
  std::vector<PrefillAdmission> many;
  for (uint64_t id = 1; id <= 8; ++id) {
    shortPrompts.submit(request(id, 65));
    many.push_back({id, 0});
  }
  require(shortPrompts.prefillAdmissionOrder(many) ==
              std::vector<uint64_t>({1, 2, 3, 4}),
          "short prefill admission lost batching or exceeded the real width");
}

void testAdmissionRespectsContendedBudgetAndDecodePriority() {
  Scheduler scheduler(0.0);
  scheduler.observePrefill(2048, 6144.0);
  std::vector<PrefillAdmission> candidates;
  for (uint64_t id = 1; id <= 4; ++id) {
    scheduler.submit(request(id, 65));
    candidates.push_back({id, 0});
  }
  require(scheduler.prefillAdmissionOrder(candidates) ==
              std::vector<uint64_t>({1, 2}),
          "admission ignored the contended actual-row budget");
  scheduler.submit(request(5, 1, false, RequestPriority::Foreground));
  scheduler.resourcesReady(5, 1);
  require(scheduler.prefillAdmissionOrder(candidates).empty(),
          "lower-priority prefill reserved lanes ahead of runnable foreground decode");
  scheduler.cancel(5);
  require(!scheduler.prefillAdmissionOrder(candidates).empty(),
          "prefill admission did not resume after foreground decode left");
}

// The tier admission can select is set by the lanes that prefill or decode,
// not by those waiting for a mask or for admission.
void testHighestRunnablePriority() {
  Scheduler scheduler(0.0);
  require(!scheduler.highestRunnablePriority(), "an idle scheduler had a runnable tier");
  scheduler.submit(request(1, 1, true, RequestPriority::Foreground));
  scheduler.resourcesReady(1, 0);
  awaitFirstMask(scheduler, 1);
  scheduler.submit(request(2, 1, false, RequestPriority::Background));
  scheduler.resourcesReady(2, 1);
  scheduler.submit(request(3, 100));
  scheduler.resourcesReady(3, 0);
  scheduler.submit(request(4, 100, false, RequestPriority::Foreground));
  require(scheduler.phase(1) == Phase::WaitingMask &&
              scheduler.highestRunnablePriority() == RequestPriority::Normal,
          "a mask wait or a queued request set the runnable tier");
}

void testQueuedPrefillCannotBeOvertakenIndefinitely() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 8193));
  for (uint64_t id = 2; id <= 4; ++id) {
    scheduler.submit(request(id, 2048));
    const std::array candidates{PrefillAdmission{1, 0}, PrefillAdmission{id, 0}};
    require(scheduler.prefillAdmissionOrder(candidates) == std::vector<uint64_t>{id},
            "short work did not overtake queued long work");
    scheduler.resourcesReady(id, 0);
    completePrefill(scheduler, *scheduler.next({}));
    scheduler.cancel(id);
    scheduler.remove(id);
  }
  scheduler.submit(request(5, 65));
  const std::array candidates{PrefillAdmission{1, 0}, PrefillAdmission{5, 0}};
  require(scheduler.prefillAdmissionOrder(candidates) == std::vector<uint64_t>{1},
          "queued long prefill starved behind short arrivals");
}

void testWarmupTimingSeedsFirstContendedCommand() {
  for (double sample : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN(), 6144.0}) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, sample);
    scheduler.observePrefill(1, 10000.0);
    scheduler.submit(request(1, 8193));
    scheduler.resourcesReady(1, 0);
    require(scheduler.next({})->items[0].tokenCount == 2048,
            "warmup shrank an uncontended prefill");
    scheduler.submit(request(2, 1));
    scheduler.resourcesReady(2, 1);
    completeDecode(scheduler);
    const auto first = *scheduler.next({});
    require(first.kind == WorkKind::Prefill &&
                first.items[0].tokenCount == (sample == 6144.0 ? 128 : 2048),
            "first contended prefill ignored warmup or accepted invalid timing");
  }
}

void testShortestRemainingFirstUsesActualRows() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 17));
  scheduler.submit(request(2, 1000));
  scheduler.submit(request(3, 3000));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  scheduler.resourcesReady(3, 0);
  BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 3,
          "ragged prefill did not pack all ready sequences");
  uint32_t rows = 0;
  for (const BatchItem &item : plan.items)
    rows += item.tokenCount;
  require(rows == model::ExecutionLimits::prefillTokenBudget,
          "ragged prefill did not use the exact actual-row budget");
  require(plan.items[0].requestId == 1 && plan.items[0].tokenCount == 17 &&
              plan.items[1].requestId == 2 &&
              plan.items[1].tokenCount == 1000 &&
              plan.items[2].requestId == 3 &&
              plan.items[2].tokenCount == 1031,
          "prefill did not serve the shortest remaining sequences first");
  require(std::all_of(plan.items.begin(), plan.items.end(),
                      [](const BatchItem &item) {
                        return item.promptOffset == 0;
                      }),
          "initial prefill plan did not carry its absolute token offset");
}

void testPerRequestBoundary() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 4096));
  scheduler.submit(request(2, 4096));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  scheduler.setPrefillBoundary(1, 32);
  BatchPlan plan = *scheduler.next({});
  require(plan.items[0].tokenCount == 32 && plan.items[1].tokenCount == 2016,
          "one sequence boundary incorrectly padded or shortened its peer");
  completePrefill(scheduler, plan);
  require(scheduler.phase(1) == engine::Phase::Prefill,
          "materialization boundary ended the request prefill");
  // The peer now has fewer remaining rows and takes the next command alone;
  // the capped lane resumes after it from its scheduler-owned offset.
  const BatchPlan peer = *scheduler.next({});
  require(peer.items.size() == 1 && peer.items[0].requestId == 2,
          "the shorter remaining peer did not take the next command");
  completePrefill(scheduler, peer);
  // The peer's 32-row tail still sorts first; the capped lane follows from
  // its scheduler-owned offset.
  const BatchPlan resumed = *scheduler.next({});
  require(resumed.items.size() == 2 && resumed.items[0].requestId == 2 &&
              resumed.items[0].tokenCount == 32 &&
              resumed.items[1].requestId == 1 &&
              resumed.items[1].promptOffset == 32 &&
              resumed.items[1].tokenCount == 2016,
          "resumed prefill lost the scheduler-owned absolute offset");
}

// complete() consumes a boundary its command reached, so the next one is
// armed against the progress after that command: a boundary the request has
// reached is refused.
void testReachedBoundaryIsConsumedBeforeTheNextIsArmed() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 4096));
  scheduler.resourcesReady(1, 0);
  scheduler.setPrefillBoundary(1, 64);
  const BatchPlan reaching = *scheduler.next({});
  require(reaching.items[0].tokenCount == 64, "the boundary did not cap the command");
  completePrefill(scheduler, reaching);
  bool refused = false;
  try {
    scheduler.setPrefillBoundary(1, 64);
  } catch (const std::invalid_argument &) {
    refused = true;
  }
  require(refused, "a boundary the request has reached was armed again");
  scheduler.setPrefillBoundary(1, 128);
  const BatchPlan next = *scheduler.next({});
  require(next.items[0].promptOffset == 64 && next.items[0].tokenCount == 64,
          "the next boundary did not cap the next command");
}

void testEqualPromptsFinishInArrivalOrder() {
  // Equal cold prompts are served oldest first, one whole budget at a time,
  // instead of an equal water-fill share that finishes them all at once.
  engine::Scheduler scheduler(0.0);
  for (uint64_t id = 1; id <= 3; ++id) {
    scheduler.submit(request(id, 3000));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "equal prompts were water-filled instead of served oldest first");
}

void testShortArrivalPrecedesLongColdPrompt() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 3000));
  scheduler.submit(request(2, 700));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  const BatchPlan plan = *scheduler.next({});
  require(plan.items.size() == 2 && plan.items[0].requestId == 2 &&
              plan.items[0].tokenCount == 700 &&
              plan.items[1].requestId == 1 &&
              plan.items[1].tokenCount == 1348,
          "a later short prompt waited behind an earlier long one");
}

void testBoundaryCapsDispatchWithoutChangingPriority() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 4096));
  scheduler.submit(request(2, 500));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  scheduler.setPrefillBoundary(1, 64);
  const BatchPlan plan = *scheduler.next({});
  require(plan.items.size() == 2 && plan.items[0].requestId == 2 &&
              plan.items[0].tokenCount == 500 &&
              plan.items[1].requestId == 1 &&
              plan.items[1].tokenCount == 64,
          "state capture changed priority or lost its dispatch boundary");

  engine::Scheduler checkpoints(0.0);
  checkpoints.submit(request(1, 25000));
  checkpoints.submit(request(2, 10000));
  checkpoints.resourcesReady(1, 2048);
  checkpoints.resourcesReady(2, 0);
  checkpoints.setPrefillBoundary(1, 4096);
  checkpoints.setPrefillBoundary(2, 4096);
  const BatchPlan next = *checkpoints.next({});
  require(next.items.size() == 1 && next.items[0].requestId == 2 &&
              next.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "rolling checkpoints placed a long prompt before a shorter arrival");
}

void testEqualLanesRunInArrivalOrderWithoutOvertaking() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 5000));
  scheduler.submit(request(2, 5000));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  const BatchPlan first = *scheduler.next({});
  require(first.items.size() == 1 && first.items[0].requestId == 1 &&
              first.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "first command did not give the whole budget to the oldest lane");
  completePrefill(scheduler, first);
  // Lane 2 was left out by an older lane, which is not overtaking: lane 1
  // keeps the budget until it finishes, as under FIFO.
  const BatchPlan second = *scheduler.next({});
  require(second.items.size() == 1 && second.items[0].requestId == 1,
          "an older lane lost the budget to a younger equal lane");
}

void testLaneOvertakenThreeTimesLeadsTheNextCommand() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 6000));
  scheduler.resourcesReady(1, 0);
  // Three later short arrivals each take the whole budget ahead of lane 1;
  // each stops in prefill so no decode command interleaves.
  const auto finishShort = [&](const BatchPlan &plan) {
    scheduler.commit(plan, {});
    const std::vector<StepResult> results{
        {plan.items[0].requestId, plan.items[0].tokenCount, true,
         DecodeStage::Regular}};
    scheduler.complete(plan, results, 0.0, true);
  };
  for (uint64_t id = 2; id <= 4; ++id) {
    scheduler.submit(request(id, model::ExecutionLimits::prefillTokenBudget));
    scheduler.resourcesReady(id, 0);
    const BatchPlan plan = *scheduler.next({});
    require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
                plan.items[0].requestId == id,
            "a short arrival did not run ahead of the long lane");
    finishShort(plan);
  }
  scheduler.submit(request(5, model::ExecutionLimits::prefillTokenBudget));
  scheduler.resourcesReady(5, 0);
  const BatchPlan overdue = *scheduler.next({});
  require(overdue.items.size() == 1 && overdue.items[0].requestId == 1 &&
              overdue.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "a lane overtaken three times did not lead the next command");
  completePrefill(scheduler, overdue);
  // Served once, the long lane yields to short arrivals again.
  const BatchPlan resumed = *scheduler.next({});
  require(resumed.items.size() == 1 && resumed.items[0].requestId == 5,
          "a served lane kept its overdue priority");
}

void testServedLaneResetsOvertaking() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 3000));
  scheduler.submit(request(2, 700));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  const BatchPlan first = *scheduler.next({});
  require(first.items.size() == 2,
          "both ready lanes did not share the command");
  scheduler.commit(first, {});
  // The short prompt stops in prefill; the long one keeps its remainder.
  std::vector<StepResult> results;
  for (const BatchItem &item : first.items) {
    results.push_back({item.requestId, item.tokenCount, item.requestId == 2,
                       DecodeStage::Regular});
  }
  scheduler.complete(first, results, 0.0, true);
  scheduler.submit(request(3, 200));
  scheduler.resourcesReady(3, 0);
  const BatchPlan second = *scheduler.next({});
  require(second.items.size() == 2 && second.items[0].requestId == 3 &&
              second.items[0].tokenCount == 200 &&
              second.items[1].requestId == 1 &&
              second.items[1].tokenCount == 1652,
          "a lane that received rows was treated as overtaken");
}

void testRealDecodeWidths() {
  for (uint32_t width = 1; width <= 4; ++width) {
    engine::Scheduler scheduler(0.0);
    for (uint32_t lane = 0; lane < width; ++lane) {
      scheduler.submit(request(lane + 1, 1));
      scheduler.resourcesReady(lane + 1, 1);
    }
    BatchPlan plan = *scheduler.next({});
    require(plan.kind == WorkKind::Decode && plan.width() == width,
            "ready decode width was delayed or rewritten");
    for (const BatchItem &item : plan.items) {
      require(item.tokenCount == 0, "decode plan carried a variable row count");
    }
    scheduler.commit(plan, {});
    std::vector<StepResult> results;
    for (const BatchItem &item : plan.items) {
      results.push_back({item.requestId, 0, false, DecodeStage::Regular});
    }
    scheduler.complete(plan, results, 0.0, true);
    require(scheduler.snapshot().decodeBatchesByWidth[width - 1] == 1,
            "decode width counter did not record the real command");
  }
}

void testUnconstrainedLanesShareDecode() {
  Scheduler scheduler(0.0);
  for (uint64_t id = 1; id <= 4; ++id) {
    scheduler.submit(request(id, 1));
    scheduler.resourcesReady(id, 1);
  }
  const BatchPlan plan = *scheduler.next({});
  require(plan.width() == 4 && !plan.constrained,
          "unconstrained requests did not share one decode command");
}

void testRejectedCommitCountsNothing() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 1);
  const BatchPlan stale = *scheduler.next({});
  scheduler.cancel(2);
  bool rejected = false;
  try {
    scheduler.commit(stale, {});
  } catch (const std::logic_error &) {
    rejected = true;
  }
  require(rejected && scheduler.snapshot().decodeBatches == 0,
          "rejected commit incremented decode counters");
  completeDecode(scheduler);
  require(scheduler.snapshot().decodeBatches == 1,
          "the remaining request's decode was not counted");
}

void testConstrainedDecodeRemainsSeparate() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1, true));
  scheduler.resourcesReady(1, 0);
  awaitFirstMask(scheduler, 1);
  scheduler.maskReady(1);
  completeDecode(scheduler);
  scheduler.submit(request(2, 1));
  scheduler.submit(request(3, 1));
  scheduler.resourcesReady(2, 1);
  scheduler.resourcesReady(3, 1);
  const BatchPlan unconstrained = *scheduler.next({});
  require(unconstrained.width() == 2 && !unconstrained.constrained,
          "unconstrained decode included a constrained lane");
  completeDecode(scheduler);
  const BatchPlan constrained = *scheduler.next({});
  require(constrained.width() == 1 && constrained.items[0].requestId == 1 &&
              constrained.constrained,
          "constrained decode lost its independent mask pipeline");
}

void testPrefillAndDecodeAlternateWithoutStarvation() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 10'000));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 0);

  BatchPlan decode = *scheduler.next({});
  require(decode.kind == WorkKind::Decode && decode.items[0].requestId == 1,
          "decode did not retain the first ready command");
  scheduler.commit(decode, {});
  const std::array decodeResult{
      StepResult{1, 0, false, DecodeStage::Regular}};
  scheduler.complete(decode, decodeResult, 0.0, true);

  BatchPlan prefill = *scheduler.next({});
  require(prefill.kind == WorkKind::Prefill &&
              prefill.items[0].requestId == 2,
          "continuous decode starved a newly admitted prefill");
  completePrefill(scheduler, prefill);

  BatchPlan nextDecode = *scheduler.next({});
  require(nextDecode.kind == WorkKind::Decode &&
              nextDecode.items[0].requestId == 1,
          "prefill did not yield the next command back to decode");
}

// Lane 1 decodes beside lane 2's long prompt: one decode command, then one
// 500 ms prefill command.
void runContendedPrefill(engine::Scheduler &scheduler) {
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 20'000));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 0);
  completeDecode(scheduler, false, 50.0);
  const BatchPlan prefill = *scheduler.next({});
  require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 2,
          "the long prompt did not follow the first decode command");
  completePrefill(scheduler, prefill, 500.0);
}

void testDecodeRepaysItsShareOfContendedPrefill() {
  for (const double share : {0.0, 0.5}) {
    engine::Scheduler scheduler(share);
    runContendedPrefill(scheduler);
    // At 0.5 the prefill owes 250 ms, five 50 ms decode commands; with no
    // share the kinds alternate one command each.
    for (uint32_t step = 0; step < (share > 0.0 ? 5u : 1u); ++step)
      completeDecode(scheduler, false, 50.0);
    require(scheduler.next({})->kind == WorkKind::Prefill,
            "decode did not return the next command once its share was repaid");
  }
}

// A constrained lane waits for its first mask while an equal-priority prompt
// prefills: decode could not have used that time, so once the mask arrives
// the kinds alternate without banked debt.
void testMaskWaitAccruesNoDecodeDebt() {
  engine::Scheduler scheduler(0.5);
  scheduler.submit(request(1, 1, true));
  scheduler.resourcesReady(1, 0);
  awaitFirstMask(scheduler, 1);
  scheduler.submit(request(2, 20'000));
  scheduler.resourcesReady(2, 0);
  for (uint32_t command = 0; command < 3; ++command)
    completePrefill(scheduler, *scheduler.next({}), 500.0);
  scheduler.maskReady(1);
  completeDecode(scheduler, false, 50.0);
  require(scheduler.next({})->kind == WorkKind::Prefill,
          "prefill beside a lane waiting for its mask banked decode debt");
}

// Lanes that wait for memory on its way back are left out of the plan, so
// the other kind or the other lanes run. A prefill that ran meanwhile owes
// the decoders it was planned without nothing, and a prefill lane left out
// was blocked, not overtaken.
void testExcludedLanesAreNotPlanned() {
  engine::Scheduler scheduler(0.5);
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 1));
  scheduler.submit(request(3, 20'000));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 1);
  scheduler.resourcesReady(3, 0);
  const std::array<uint64_t, 1> first{1};
  const BatchPlan decode = *scheduler.next(first);
  require(decode.kind == WorkKind::Decode && decode.width() == 1 &&
              decode.items[0].requestId == 2,
          "a decode plan included a lane left out of it");
  const std::array<uint64_t, 2> decoders{1, 2};
  const BatchPlan prefill = *scheduler.next(decoders);
  require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 3,
          "the prefill did not run while every decoder was left out");
  scheduler.commit(prefill, decoders);
  const std::array result{StepResult{3, prefill.items[0].tokenCount, false, DecodeStage::Regular}};
  scheduler.complete(prefill, result, 500.0, true);
  completeDecode(scheduler, false, 50.0);
  require(scheduler.next({})->kind == WorkKind::Prefill,
          "a prefill owed decode time to decoders it was planned without");

  engine::Scheduler blocked(0.0);
  blocked.submit(request(1, 6000));
  blocked.resourcesReady(1, 0);
  const std::array<uint64_t, 1> waiting{1};
  for (uint64_t id = 2; id <= 4; ++id) {
    blocked.submit(request(id, model::ExecutionLimits::prefillTokenBudget));
    blocked.resourcesReady(id, 0);
    const BatchPlan plan = *blocked.next(waiting);
    blocked.commit(plan, waiting);
    const std::array finished{
        StepResult{id, plan.items[0].tokenCount, true, DecodeStage::Regular}};
    blocked.complete(plan, finished, 0.0, true);
  }
  blocked.submit(request(5, model::ExecutionLimits::prefillTokenBudget));
  blocked.resourcesReady(5, 0);
  require(blocked.next({})->items[0].requestId == 5,
          "a lane left out of three commands was treated as overtaken");
}

void testDecodeDebtLeavesWithTheLastDecoder() {
  for (const bool finished : {true, false}) {
    engine::Scheduler scheduler(0.5);
    runContendedPrefill(scheduler);
    // Lane 1 leaves still owed 200 ms: its decode command finishes it, or it
    // is cancelled between commands.
    completeDecode(scheduler, finished, 50.0);
    if (!finished)
      scheduler.cancel(1);
    scheduler.submit(request(3, 1));
    scheduler.resourcesReady(3, 1);
    require(scheduler.next({})->kind == WorkKind::Prefill,
            "a later decoder inherited debt owed to one that left");
  }
}

void testHigherPriorityPrefillPrecedesDecodeDebt() {
  engine::Scheduler scheduler(0.5);
  runContendedPrefill(scheduler);
  scheduler.submit(request(3, 20'000, false, RequestPriority::Foreground));
  scheduler.resourcesReady(3, 0);
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 3,
          "decode debt overrode a higher-priority prefill");
}

void testMeasuredBudgetOnlyLimitsContendedWork() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  const BatchPlan first = *scheduler.next({});
  require(first.items[0].tokenCount == 2048,
          "unmeasured isolated work lost the full prefill budget");
  completePrefill(scheduler, first, 4096.0);
  require(scheduler.next({})->items[0].tokenCount == 2048,
          "slow isolated work lost the full prefill budget");

  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler, false, 1e9);
  const BatchPlan contended = *scheduler.next({});
  require(contended.kind == WorkKind::Prefill &&
              contended.items[0].tokenCount == 128,
          "measured slow prefill was not shortened for a decoding peer");
  completePrefill(scheduler, contended, 256.0);
  completeDecode(scheduler, true);
  require(scheduler.next({})->items[0].tokenCount == 2048,
          "prefill did not recover isolated throughput after its peer ended");
}

void testAuxiliaryWorkDoesNotTrainTextPrefillTiming() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  completePrefill(scheduler, *scheduler.next({}), 1e9, false);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler);
  require(scheduler.next({})->items.front().tokenCount == 128,
          "auxiliary encoding latency contaminated the text throughput estimate");
}

void testMeasuredBudgetUsesActualRowsAndRecovers() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 128));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 512.0);
  scheduler.submit(request(2, 20'000));
  scheduler.resourcesReady(2, 0);
  completeDecode(scheduler);
  BatchPlan plan = *scheduler.next({});
  require(plan.items[0].tokenCount == 64,
          "prefill timing was normalized by capacity instead of actual rows");

  for (uint32_t sample = 0; sample < 8; ++sample) {
    completePrefill(scheduler, plan, 0.25 * plan.items[0].tokenCount);
    completeDecode(scheduler);
    plan = *scheduler.next({});
  }
  require(plan.items[0].tokenCount > 64 && plan.items[0].tokenCount < 2048,
          "prefill budget did not adapt when measured execution became faster");
}

void testMeasuredBudgetPreservesPriorityAndStateBoundaries() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000, false, RequestPriority::Foreground));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  scheduler.submit(request(2, 1, false, RequestPriority::Background));
  scheduler.resourcesReady(2, 1);
  require(scheduler.next({})->items[0].tokenCount == 2048,
          "lower-priority work reduced foreground throughput");

  scheduler.submit(request(3, 20'000, false, RequestPriority::Foreground));
  scheduler.resourcesReady(3, 0);
  scheduler.submit(request(4, 1, false, RequestPriority::Foreground));
  scheduler.resourcesReady(4, 1);
  completeDecode(scheduler);
  scheduler.setPrefillBoundary(1, 2080);
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 2 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].promptOffset == 2048 &&
              plan.items[0].tokenCount == 32 &&
              plan.items[1].requestId == 3 &&
              plan.items[1].tokenCount == 96,
          "adaptive prefill lost a state boundary or exceeded its shared budget");
}

void testUnavailableTimingAndMinimumBudget() {
  for (double wallMilliseconds :
       {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(), 1.0, 1e9}) {
    engine::Scheduler scheduler(0.0);
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    completePrefill(scheduler, *scheduler.next({}), wallMilliseconds);
    scheduler.submit(request(2, 1));
    scheduler.resourcesReady(2, 1);
    completeDecode(scheduler);
    const BatchPlan plan = *scheduler.next({});
    require(plan.items[0].tokenCount ==
                (wallMilliseconds == 1e9 ? 64u : 2048u),
            "missing timing or extreme sample produced an invalid prefill budget");
  }
}

void testMeasuredBudgetDoesNotCountBlockedPeers() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  scheduler.submit(request(2, 20'000));
  scheduler.waitForResources(2);
  scheduler.submit(request(3, 20'000));
  require(scheduler.next({})->items[0].tokenCount == 2048,
          "queued or resource-blocked work reduced resident throughput");
}

void testMeasuredBudgetPreservesPurePrefillPacking() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  for (uint64_t id = 2; id <= 4; ++id) {
    scheduler.submit(request(id, 20'000));
    scheduler.resourcesReady(id, 0);
  }
  scheduler.setPrefillBoundary(1, 2080);
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.width() == 2 &&
              plan.items[0].tokenCount == 32 &&
              plan.items[1].tokenCount == 2016,
          "pure prefill contention lost throughput or stopped packing peers");
}

void testTinyTailDoesNotDistortPrefillThroughput() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 2056));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 2048.0);
  completePrefill(scheduler, *scheduler.next({}), 800.0);
  scheduler.submit(request(2, 20'000));
  scheduler.resourcesReady(2, 0);
  completeDecode(scheduler);
  require(scheduler.next({})->items[0].tokenCount == 256,
          "fixed overhead from a tiny tail distorted the prefill estimate");
}

void testMeasuredBudgetFinishesShortPrefillPromptly() {
  for (const auto priority : {RequestPriority::Normal,
                              RequestPriority::Background}) {
    engine::Scheduler scheduler(0.0);
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    completePrefill(scheduler, *scheduler.next({}), 4096.0);
    scheduler.submit(request(2, 31, false, priority));
    scheduler.resourcesReady(2, 0);
    const BatchPlan plan = *scheduler.next({});
    if (priority == RequestPriority::Background) {
      require(plan.width() == 1 && plan.items[0].requestId == 1 &&
                  plan.items[0].tokenCount == 2048,
              "a lower-priority short prefill throttled foreground work");
      continue;
    }
    require(plan.width() == 2 && plan.items[0].requestId == 2 &&
                plan.items[0].tokenCount == 31 &&
                plan.items[1].requestId == 1 &&
                plan.items[1].tokenCount == 97,
            "a finishing short prefill waited for a full long-prefill batch");
    completePrefill(scheduler, plan, 256.0);
    const BatchPlan next = *scheduler.next({});
    require(next.kind == WorkKind::Decode && next.items[0].requestId == 2,
            "the completed short prefill did not get its next decode turn");
  }
}

// At a measured 2.75 ms per row the slice is 128 rows. An arrival that
// finishes within the full budget, though not within one slice, takes a
// command of its own rows instead of a full one shared with a long prompt;
// one that does not finish in it still packs a full command.
void testShortArrivalBesideLongPrefillEndsAtItsLastRow() {
  const auto beside = [](std::initializer_list<uint32_t> arrivals) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 5632.0);
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    completePrefill(scheduler, *scheduler.next({}), 5632.0);
    uint64_t id = 2;
    for (uint32_t prompt : arrivals) {
      scheduler.submit(request(id, prompt));
      scheduler.resourcesReady(id++, 0);
    }
    return scheduler;
  };
  Scheduler chat = beside({300});
  const BatchPlan plan = *chat.next({});
  require(plan.width() == 1 && plan.items[0].requestId == 2 && plan.items[0].tokenCount == 300,
          "a short arrival shared a full command with a long prefill");
  completePrefill(chat, plan, 825.0);
  completeDecode(chat);
  const BatchPlan slice = *chat.next({});
  require(slice.width() == 1 && slice.items[0].requestId == 1 &&
              slice.items[0].tokenCount == 128,
          "the long prefill did not return to slices beside the decoding arrival");

  const BatchPlan packed = *beside({3000}).next({});
  require(packed.width() == 1 && packed.items[0].requestId == 2 &&
              packed.items[0].tokenCount == 2048,
          "an arrival that does not finish in one command lost the full budget");

  const BatchPlan first = *beside({300, 300}).next({});
  require(first.width() == 1 && first.items[0].requestId == 2 &&
              first.items[0].tokenCount == 300,
          "two short arrivals shared a command instead of finishing in turn");
}

void testMeasuredBudgetRetainsOvertakingBound() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  scheduler.submit(request(100, 1));
  scheduler.resourcesReady(100, 1);
  for (uint64_t id = 2; id <= 4; ++id) {
    completeDecode(scheduler);
    scheduler.submit(request(id, 128));
    scheduler.resourcesReady(id, 0);
    const BatchPlan plan = *scheduler.next({});
    require(plan.items.size() == 1 && plan.items[0].requestId == id,
            "adaptive prefill did not serve a short arrival first");
    scheduler.commit(plan, {});
    const std::array result{StepResult{id, 128, true, DecodeStage::Regular}};
    scheduler.complete(plan, result, 256.0, true);
  }
  scheduler.submit(request(5, 128));
  scheduler.resourcesReady(5, 0);
  completeDecode(scheduler);
  const BatchPlan overdue = *scheduler.next({});
  require(overdue.items.size() == 1 && overdue.items[0].requestId == 1 &&
              overdue.items[0].tokenCount == 128,
          "adaptive prefill allowed short arrivals to starve an older lane");
}

void testPriorityPrecedesWorkKindAlternation() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1, false, RequestPriority::Background));
  scheduler.submit(request(2, 100, false, RequestPriority::Foreground));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 0);

  BatchPlan foreground = *scheduler.next({});
  require(foreground.kind == WorkKind::Prefill &&
              foreground.items[0].requestId == 2,
          "work-kind alternation overrode request priority");
}

void testPrefillCommandContainsOnePriorityTier() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 4096, false, RequestPriority::Foreground));
  scheduler.submit(request(2, 4096, false, RequestPriority::Background));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);

  BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "ragged prefill mixed priority tiers in one command");
}

void testDecodeCommandContainsOnePriorityTier() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1, false, RequestPriority::Foreground));
  scheduler.submit(request(2, 1, false, RequestPriority::Background));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 1);

  BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Decode && plan.width() == 1 &&
              plan.items[0].requestId == 1,
          "a decode command mixed priority tiers");
  scheduler.commit(plan, {});
  const std::array result{
      StepResult{1, 0, true, DecodeStage::Regular}};
  scheduler.complete(plan, result, 0.0, true);

  BatchPlan background = *scheduler.next({});
  require(background.kind == WorkKind::Decode && background.width() == 1 &&
              background.items[0].requestId == 2,
          "background decode did not run after foreground completion");
}

void testDecodeLanesRotate() {
  engine::Scheduler scheduler(0.0);
  for (uint64_t id = 1; id <= 6; ++id) {
    scheduler.submit(request(id, 1));
    scheduler.resourcesReady(id, 1);
  }

  BatchPlan first = *scheduler.next({});
  require(first.kind == WorkKind::Decode && first.width() == 4 &&
              !first.constrained,
          "the first unconstrained decode did not use the real B4 graph");
  scheduler.commit(first, {});
  std::vector<StepResult> firstResults;
  for (const BatchItem &item : first.items) {
    firstResults.push_back(
        {item.requestId, 0, false, DecodeStage::Regular});
  }
  scheduler.complete(first, firstResults, 0.0, true);

  BatchPlan second = *scheduler.next({});
  require(second.kind == WorkKind::Decode && second.width() == 4 &&
              second.items[0].requestId == 5 && second.items[1].requestId == 6,
          "decode did not prioritize lanes omitted by the previous batch");
}

// Compare repeated dispatches with the previous full-sort policy across
// queue depths, submission orders, priority tiers, constraints and exclusions.
void testDecodeSelectionMatchesFullSort() {
  struct Lane {
    uint64_t id;
    RequestPriority priority;
    bool constrained;
    uint64_t order;
    uint64_t lastDispatch = 0;
    bool ready;
  };
  std::mt19937 random(106);
  for (uint32_t count : {0, 1, 4, 5, 16, 65}) {
    for (uint32_t trial = 0; trial < 16; ++trial) {
      Scheduler scheduler(0.0);
      std::vector<uint64_t> ids;
      for (uint64_t id = 1; id <= count; ++id)
        ids.push_back(id);
      std::shuffle(ids.begin(), ids.end(), random);
      std::vector<Lane> lanes;
      for (uint64_t id : ids) {
        const auto priority = static_cast<RequestPriority>(random() % 3);
        const bool constrained = random() % 2;
        const bool ready = random() % 5 != 0;
        lanes.push_back({id, priority, constrained, lanes.size(), 0, ready});
        scheduler.submit(request(id, 1, constrained, priority));
        if (ready)
          scheduler.resourcesReady(id, 1);
      }
      uint64_t dispatch = 0;
      for (uint32_t step = 0; step < 64; ++step) {
        std::shuffle(ids.begin(), ids.end(), random);
        const std::span<const uint64_t> excluded(
            ids.data(), std::min<size_t>(random() % 5, ids.size()));
        std::vector<Lane *> sorted;
        for (Lane &lane : lanes)
          if (lane.ready && std::find(excluded.begin(), excluded.end(), lane.id) ==
                                excluded.end())
            sorted.push_back(&lane);
        std::sort(sorted.begin(), sorted.end(), [](const Lane *a, const Lane *b) {
          if (a->priority != b->priority)
            return a->priority < b->priority;
          if (a->lastDispatch != b->lastDispatch)
            return a->lastDispatch < b->lastDispatch;
          return a->order < b->order;
        });
        const auto plan = scheduler.next(excluded);
        require(plan.has_value() == !sorted.empty(),
                "bounded selection changed decode readiness");
        if (!plan)
          continue;
        const Lane &leader = *sorted.front();
        std::vector<Lane *> expected;
        for (Lane *lane : sorted) {
          if (lane->priority == leader.priority &&
              lane->constrained == leader.constrained)
            expected.push_back(lane);
          if (expected.size() == model::ExecutionLimits::maximumBatchWidth)
            break;
        }
        require(plan->kind == WorkKind::Decode &&
                    plan->constrained == leader.constrained &&
                    plan->decodeStage == DecodeStage::Regular &&
                    plan->width() == expected.size(),
                "bounded selection changed the full-sort batch shape");
        std::vector<StepResult> results;
        ++dispatch;
        for (size_t index = 0; index < expected.size(); ++index) {
          Lane &lane = *expected[index];
          require(plan->items[index].requestId == lane.id &&
                      plan->items[index].tokenCount == 0 &&
                      plan->items[index].promptOffset == 0,
                  "bounded selection changed full-sort dispatch order");
          lane.lastDispatch = dispatch;
          const bool finished = random() % 7 == 0;
          lane.ready = !finished;
          results.push_back({lane.id, 0, finished, DecodeStage::Regular});
        }
        scheduler.commit(*plan, excluded);
        scheduler.complete(*plan, results, 0.0, true);
      }
    }
  }
}

void testMaskStagesNeverMix() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1, true));
  scheduler.submit(request(2, 1, true));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);

  BatchPlan prompts = *scheduler.next({});
  require(prompts.kind == WorkKind::Prefill && prompts.width() == 2,
          "constrained prompts did not form one prefill");
  scheduler.commit(prompts, {});
  const std::array promptResults{
      StepResult{1, 1, false, DecodeStage::ApplyInitialMask},
      StepResult{2, 1, false, DecodeStage::ApplyInitialMask},
  };
  scheduler.complete(prompts, promptResults, 0.0, true);
  scheduler.maskReady(1);

  BatchPlan initialSelection = *scheduler.next({});
  require(initialSelection.width() == 1 &&
              initialSelection.items[0].requestId == 1 &&
              initialSelection.decodeStage == DecodeStage::ApplyInitialMask,
          "a ready first mask did not get its selection plan");
  scheduler.commit(initialSelection, {});
  const std::array initialSelectionResult{
      StepResult{1, 0, false, DecodeStage::Regular}};
  scheduler.complete(initialSelection, initialSelectionResult, 0.0, true);

  scheduler.maskReady(2);
  BatchPlan otherInitial = *scheduler.next({});
  require(otherInitial.width() == 1 &&
              otherInitial.items[0].requestId == 2 &&
              otherInitial.decodeStage == DecodeStage::ApplyInitialMask,
          "a first token's selection mixed with a drafting lane");
  scheduler.commit(otherInitial, {});
  const std::array otherInitialResult{
      StepResult{2, 0, true, DecodeStage::Regular}};
  scheduler.complete(otherInitial, otherInitialResult, 0.0, true);

  BatchPlan verify = *scheduler.next({});
  require(verify.width() == 1 && verify.items[0].requestId == 1 &&
              verify.decodeStage == DecodeStage::Regular,
          "a drafting lane mixed with a first token's selection");
  scheduler.commit(verify, {});
  const std::array verifyResult{
      StepResult{1, 0, true, DecodeStage::Regular}};
  scheduler.complete(verify, verifyResult, 0.0, true);
}

// The first tokens of constrained requests of one priority whose masks have
// arrived are selected in one plan.
void testInitialSelectionsBatch() {
  engine::Scheduler scheduler(0.0);
  for (uint64_t id = 1; id <= 6; ++id) {
    scheduler.submit(request(id, 1, true));
    scheduler.resourcesReady(id, 0);
  }
  for (uint32_t batch = 0; batch < 2; ++batch) {
    const BatchPlan prompts = *scheduler.next({});
    scheduler.commit(prompts, {});
    std::vector<StepResult> results;
    for (const BatchItem &item : prompts.items)
      results.push_back({item.requestId, 1, false, DecodeStage::ApplyInitialMask});
    scheduler.complete(prompts, results, 0.0, true);
  }
  for (uint64_t id = 1; id <= 6; ++id)
    scheduler.maskReady(id);
  const std::array<uint64_t, 1> excluded{1};
  const BatchPlan selection = *scheduler.next(excluded);
  require(selection.kind == WorkKind::Decode && selection.width() == 4 &&
              selection.decodeStage == DecodeStage::ApplyInitialMask &&
              selection.constrained,
          "first-token selections of one priority did not share a full plan");
  for (size_t index = 0; index < selection.items.size(); ++index)
    require(selection.items[index].requestId == index + 2,
            "first-token selection changed arrival order or included an excluded lane");
}

void testWaitingMaskExpiresAtRequestDeadline() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1, true));
  scheduler.resourcesReady(1, 0);
  awaitFirstMask(scheduler, 1);
  require(scheduler.expireDeadlines(10'000.0) &&
              scheduler.phase(1) == engine::Phase::Failed,
          "waiting mask survived its request deadline");

  bool rejectedLateMask = false;
  try {
    scheduler.maskReady(1);
  } catch (const std::logic_error &) {
    rejectedLateMask = true;
  }
  require(rejectedLateMask, "late mask revived an expired request");
}

void testWaitingMaskBoundsPeerPrefill() {
  for (const RequestPriority priority : {RequestPriority::Foreground,
                                         RequestPriority::Normal,
                                         RequestPriority::Background}) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4096.0);
    scheduler.submit(request(1, 1, true, priority));
    scheduler.resourcesReady(1, 0);
    awaitFirstMask(scheduler, 1);

    scheduler.submit(request(2, 20'000));
    scheduler.resourcesReady(2, 0);
    const BatchPlan prefill = *scheduler.next({});
    const bool protectedPeer = priority <= RequestPriority::Normal;
    require(prefill.kind == WorkKind::Prefill &&
                prefill.items[0].requestId == 2 &&
                prefill.items[0].tokenCount == (protectedPeer ? 128u : 2048u),
            "mask wait lost prefill latency protection or priority ordering");
    // A mask arriving during the command must get the next decode turn.
    scheduler.maskReady(1);
    completePrefill(scheduler, prefill);
    if (protectedPeer) {
      const BatchPlan decode = *scheduler.next({});
      require(decode.kind == WorkKind::Decode &&
                  decode.items[0].requestId == 1,
              "ready mask did not resume after bounded prefill");
    }
    scheduler.cancel(1);
    require(scheduler.next({})->items[0].tokenCount == 2048,
            "cancelled mask request kept isolated prefill throttled");
  }
}

void testResourceSuspensionReplaysFromCache() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 4096));
  scheduler.resourcesReady(1, 0);
  BatchPlan first = *scheduler.next({});
  completePrefill(scheduler, first);
  require(!scheduler.suspended(1), "a running request was suspended");
  scheduler.suspendForResources(1);
  require(scheduler.phase(1) == engine::Phase::WaitingResources && scheduler.suspended(1),
          "resident prefill did not enter resource wait");
  scheduler.resumeFromResources(1, 32, 4096);
  require(!scheduler.suspended(1), "a resumed request stayed suspended");
  BatchPlan resumed = *scheduler.next({});
  require(resumed.kind == WorkKind::Prefill &&
              resumed.items[0].promptOffset == 32,
          "resource resume did not start from its acquired cache boundary");

  // A request that ends while suspended stays so until it is removed, and
  // submission order follows the submits.
  engine::Scheduler ended(0.0);
  ended.submit(request(3, 64));
  ended.submit(request(4, 64));
  require(ended.submissionOrder(3) < ended.submissionOrder(4),
          "submission order did not follow the submits");
  ended.resourcesReady(3, 0);
  completePrefill(ended, *ended.next({}));
  ended.suspendForResources(3);
  ended.cancel(3);
  require(ended.phase(3) == engine::Phase::Cancelled && ended.suspended(3),
          "a cancelled request forgot it was suspended");
  ended.remove(3);
}

// A request suspended while it holds its first mask replays its history and
// selects its first token under that mask, without asking for another: the
// replay's prefill reports no mask wait.
void testReplayKeepsHeldInitialMask() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(2, 1, true));
  scheduler.resourcesReady(2, 0);
  awaitFirstMask(scheduler, 2);
  scheduler.maskReady(2);
  scheduler.suspendForResources(2);
  scheduler.resumeFromResources(2, 32, 40);
  const auto replay = *scheduler.next({});
  require(replay.kind == WorkKind::Prefill &&
              replay.items[0].promptOffset == 32 &&
              replay.items[0].tokenCount == 8,
          "decode resume did not replay its committed token history");
  completePrefill(scheduler, replay);
  require(scheduler.phase(2) == engine::Phase::Decode,
          "the replay waited for a second first mask");
  const auto continuation = *scheduler.next({});
  require(continuation.kind == WorkKind::Decode &&
              continuation.decodeStage == DecodeStage::ApplyInitialMask,
          "the replay dropped the held first mask's selection");
}

} // namespace

int main() {
  try {
    testAdmissionSharesDispatchOrderAndBudget();
    testAdmissionRespectsContendedBudgetAndDecodePriority();
    testHighestRunnablePriority();
    testQueuedPrefillCannotBeOvertakenIndefinitely();
    testWarmupTimingSeedsFirstContendedCommand();
    testShortestRemainingFirstUsesActualRows();
    testPerRequestBoundary();
    testReachedBoundaryIsConsumedBeforeTheNextIsArmed();
    testEqualPromptsFinishInArrivalOrder();
    testShortArrivalPrecedesLongColdPrompt();
    testBoundaryCapsDispatchWithoutChangingPriority();
    testEqualLanesRunInArrivalOrderWithoutOvertaking();
    testLaneOvertakenThreeTimesLeadsTheNextCommand();
    testServedLaneResetsOvertaking();
    testRealDecodeWidths();
    testUnconstrainedLanesShareDecode();
    testRejectedCommitCountsNothing();
    testConstrainedDecodeRemainsSeparate();
    testPrefillAndDecodeAlternateWithoutStarvation();
    testDecodeRepaysItsShareOfContendedPrefill();
    testMaskWaitAccruesNoDecodeDebt();
    testExcludedLanesAreNotPlanned();
    testDecodeDebtLeavesWithTheLastDecoder();
    testHigherPriorityPrefillPrecedesDecodeDebt();
    testMeasuredBudgetOnlyLimitsContendedWork();
    testAuxiliaryWorkDoesNotTrainTextPrefillTiming();
    testMeasuredBudgetUsesActualRowsAndRecovers();
    testMeasuredBudgetPreservesPriorityAndStateBoundaries();
    testUnavailableTimingAndMinimumBudget();
    testMeasuredBudgetDoesNotCountBlockedPeers();
    testMeasuredBudgetPreservesPurePrefillPacking();
    testTinyTailDoesNotDistortPrefillThroughput();
    testMeasuredBudgetFinishesShortPrefillPromptly();
    testShortArrivalBesideLongPrefillEndsAtItsLastRow();
    testMeasuredBudgetRetainsOvertakingBound();
    testPriorityPrecedesWorkKindAlternation();
    testPrefillCommandContainsOnePriorityTier();
    testDecodeCommandContainsOnePriorityTier();
    testDecodeLanesRotate();
    testDecodeSelectionMatchesFullSort();
    testMaskStagesNeverMix();
    testInitialSelectionsBatch();
    testWaitingMaskExpiresAtRequestDeadline();
    testWaitingMaskBoundsPeerPrefill();
    testResourceSuspensionReplaysFromCache();
    testReplayKeepsHeldInitialMask();
    std::cout << "ragged scheduler tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ragged scheduler tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
