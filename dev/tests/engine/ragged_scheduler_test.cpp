#include "engine/Scheduler.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

engine::RequestSpec request(uint64_t id, uint32_t prompt,
                             BatchCohort cohort = BatchCohort::Greedy,
                             RequestPriority priority = RequestPriority::Normal) {
  return {id, priority, cohort, prompt, 10'000.0};
}

void completePrefill(engine::Scheduler &scheduler,
                     const BatchPlan &plan, double wallMilliseconds = 0.0,
                     bool representativePrefillTiming = true) {
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items) {
    results.push_back(
        {item.requestId, item.tokenCount, false, DecodeStage::Regular});
  }
  scheduler.commit(plan);
  scheduler.complete(plan, results, wallMilliseconds,
                      representativePrefillTiming);
}

void completeDecode(engine::Scheduler &scheduler, bool finished = false,
                    double wallMilliseconds = 0.0) {
  const BatchPlan plan = *scheduler.next();
  require(plan.kind == WorkKind::Decode, "expected a decode command");
  scheduler.commit(plan);
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items)
    results.push_back({item.requestId, 0, finished, DecodeStage::Regular});
  scheduler.complete(plan, results, wallMilliseconds);
}

void testAdmissionSharesDispatchOrderAndBudget() {
  Scheduler scheduler;
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
  const auto plan = *scheduler.next();
  require(plan.items.size() == 3 && plan.items[0].requestId == 3 &&
              plan.items[1].requestId == 4 && plan.items[2].requestId == 1 &&
              plan.items[2].tokenCount == 1982,
          "dispatch disagreed with admission work accounting");

  Scheduler shortPrompts;
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
  Scheduler scheduler;
  scheduler.observePrefill(2048, 6144.0);
  std::vector<PrefillAdmission> candidates;
  for (uint64_t id = 1; id <= 4; ++id) {
    scheduler.submit(request(id, 65));
    candidates.push_back({id, 0});
  }
  require(scheduler.prefillAdmissionOrder(candidates) ==
              std::vector<uint64_t>({1, 2}),
          "admission ignored the contended actual-row budget");
  scheduler.submit(request(5, 1, BatchCohort::Greedy, RequestPriority::Foreground));
  scheduler.resourcesReady(5, 1);
  require(scheduler.prefillAdmissionOrder(candidates).empty(),
          "lower-priority prefill reserved cells ahead of runnable foreground decode");
  scheduler.cancel(5);
  require(!scheduler.prefillAdmissionOrder(candidates).empty(),
          "prefill admission did not resume after foreground decode left");
}

void testQueuedPrefillCannotBeOvertakenIndefinitely() {
  Scheduler scheduler;
  scheduler.submit(request(1, 8193));
  for (uint64_t id = 2; id <= 4; ++id) {
    scheduler.submit(request(id, 2048));
    const std::array candidates{PrefillAdmission{1, 0}, PrefillAdmission{id, 0}};
    require(scheduler.prefillAdmissionOrder(candidates) == std::vector<uint64_t>{id},
            "short work did not overtake queued long work");
    scheduler.resourcesReady(id, 0);
    completePrefill(scheduler, *scheduler.next());
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
    Scheduler scheduler;
    scheduler.observePrefill(2048, sample);
    scheduler.observePrefill(1, 10000.0);
    scheduler.submit(request(1, 8193));
    scheduler.resourcesReady(1, 0);
    require(scheduler.next()->items[0].tokenCount == 2048,
            "warmup shrank an uncontended prefill");
    scheduler.submit(request(2, 1));
    scheduler.resourcesReady(2, 1);
    completeDecode(scheduler);
    const auto first = *scheduler.next();
    require(first.kind == WorkKind::Prefill &&
                first.items[0].tokenCount == (sample == 6144.0 ? 128 : 2048),
            "first contended prefill ignored warmup or accepted invalid timing");
  }
}

void testShortestRemainingFirstUsesActualRows() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 17));
  scheduler.submit(request(2, 1000));
  scheduler.submit(request(3, 3000));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  scheduler.resourcesReady(3, 0);
  BatchPlan plan = *scheduler.next();
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
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 4096));
  scheduler.submit(request(2, 4096));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  scheduler.setPrefillBoundary(1, 32);
  BatchPlan plan = *scheduler.next();
  require(plan.items[0].tokenCount == 32 && plan.items[1].tokenCount == 2016,
          "one sequence boundary incorrectly padded or shortened its peer");
  completePrefill(scheduler, plan);
  require(scheduler.phase(1) == engine::Phase::Prefill,
          "materialization boundary ended the request prefill");
  // The peer now has fewer remaining rows and takes the next command alone;
  // the capped lane resumes after it from its scheduler-owned offset.
  const BatchPlan peer = *scheduler.next();
  require(peer.items.size() == 1 && peer.items[0].requestId == 2,
          "the shorter remaining peer did not take the next command");
  completePrefill(scheduler, peer);
  // The peer's 32-row tail still sorts first; the capped lane follows from
  // its scheduler-owned offset.
  const BatchPlan resumed = *scheduler.next();
  require(resumed.items.size() == 2 && resumed.items[0].requestId == 2 &&
              resumed.items[0].tokenCount == 32 &&
              resumed.items[1].requestId == 1 &&
              resumed.items[1].promptOffset == 32 &&
              resumed.items[1].tokenCount == 2016,
          "resumed prefill lost the scheduler-owned absolute offset");
}

void testEqualPromptsFinishInArrivalOrder() {
  // Equal cold prompts are served oldest first, one whole budget at a time,
  // instead of an equal water-fill share that finishes them all at once.
  engine::Scheduler scheduler;
  for (uint64_t id = 1; id <= 3; ++id) {
    scheduler.submit(request(id, 3000));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan plan = *scheduler.next();
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "equal prompts were water-filled instead of served oldest first");
}

void testShortArrivalPrecedesLongColdPrompt() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 3000));
  scheduler.submit(request(2, 700));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  const BatchPlan plan = *scheduler.next();
  require(plan.items.size() == 2 && plan.items[0].requestId == 2 &&
              plan.items[0].tokenCount == 700 &&
              plan.items[1].requestId == 1 &&
              plan.items[1].tokenCount == 1348,
          "a later short prompt waited behind an earlier long one");
}

void testBoundaryCapsDispatchWithoutChangingPriority() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 4096));
  scheduler.submit(request(2, 500));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  scheduler.setPrefillBoundary(1, 64);
  const BatchPlan plan = *scheduler.next();
  require(plan.items.size() == 2 && plan.items[0].requestId == 2 &&
              plan.items[0].tokenCount == 500 &&
              plan.items[1].requestId == 1 &&
              plan.items[1].tokenCount == 64,
          "state capture changed priority or lost its dispatch boundary");

  engine::Scheduler checkpoints;
  checkpoints.submit(request(1, 25000));
  checkpoints.submit(request(2, 10000));
  checkpoints.resourcesReady(1, 2048);
  checkpoints.resourcesReady(2, 0);
  checkpoints.setPrefillBoundary(1, 4096);
  checkpoints.setPrefillBoundary(2, 4096);
  const BatchPlan next = *checkpoints.next();
  require(next.items.size() == 1 && next.items[0].requestId == 2 &&
              next.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "rolling checkpoints placed a long prompt before a shorter arrival");
}

void testEqualLanesRunInArrivalOrderWithoutOvertaking() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 5000));
  scheduler.submit(request(2, 5000));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  const BatchPlan first = *scheduler.next();
  require(first.items.size() == 1 && first.items[0].requestId == 1 &&
              first.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "first command did not give the whole budget to the oldest lane");
  completePrefill(scheduler, first);
  // Lane 2 was left out by an older lane, which is not overtaking: lane 1
  // keeps the budget until it finishes, as under FIFO.
  const BatchPlan second = *scheduler.next();
  require(second.items.size() == 1 && second.items[0].requestId == 1,
          "an older lane lost the budget to a younger equal lane");
}

void testLaneOvertakenThreeTimesLeadsTheNextCommand() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 6000));
  scheduler.resourcesReady(1, 0);
  // Three later short arrivals each take the whole budget ahead of lane 1;
  // each stops in prefill so no decode command interleaves.
  const auto finishShort = [&](const BatchPlan &plan) {
    scheduler.commit(plan);
    const std::vector<StepResult> results{
        {plan.items[0].requestId, plan.items[0].tokenCount, true,
         DecodeStage::Regular}};
    scheduler.complete(plan, results);
  };
  for (uint64_t id = 2; id <= 4; ++id) {
    scheduler.submit(request(id, model::ExecutionLimits::prefillTokenBudget));
    scheduler.resourcesReady(id, 0);
    const BatchPlan plan = *scheduler.next();
    require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
                plan.items[0].requestId == id,
            "a short arrival did not run ahead of the long lane");
    finishShort(plan);
  }
  scheduler.submit(request(5, model::ExecutionLimits::prefillTokenBudget));
  scheduler.resourcesReady(5, 0);
  const BatchPlan overdue = *scheduler.next();
  require(overdue.items.size() == 1 && overdue.items[0].requestId == 1 &&
              overdue.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "a lane overtaken three times did not lead the next command");
  completePrefill(scheduler, overdue);
  // Served once, the long lane yields to short arrivals again.
  const BatchPlan resumed = *scheduler.next();
  require(resumed.items.size() == 1 && resumed.items[0].requestId == 5,
          "a served lane kept its overdue priority");
}

void testServedLaneResetsOvertaking() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 3000));
  scheduler.submit(request(2, 700));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  const BatchPlan first = *scheduler.next();
  require(first.items.size() == 2,
          "both ready lanes did not share the command");
  scheduler.commit(first);
  // The short prompt stops in prefill; the long one keeps its remainder.
  std::vector<StepResult> results;
  for (const BatchItem &item : first.items) {
    results.push_back({item.requestId, item.tokenCount, item.requestId == 2,
                       DecodeStage::Regular});
  }
  scheduler.complete(first, results);
  scheduler.submit(request(3, 200));
  scheduler.resourcesReady(3, 0);
  const BatchPlan second = *scheduler.next();
  require(second.items.size() == 2 && second.items[0].requestId == 3 &&
              second.items[0].tokenCount == 200 &&
              second.items[1].requestId == 1 &&
              second.items[1].tokenCount == 1652,
          "a lane that received rows was treated as overtaken");
}

void testRealDecodeWidths() {
  for (uint32_t width = 1; width <= 4; ++width) {
    engine::Scheduler scheduler;
    for (uint32_t lane = 0; lane < width; ++lane) {
      scheduler.submit(request(lane + 1, 1));
      scheduler.resourcesReady(lane + 1, 1);
    }
    BatchPlan plan = *scheduler.next();
    require(plan.kind == WorkKind::Decode && plan.width() == width,
            "ready decode width was delayed or rewritten");
    for (const BatchItem &item : plan.items) {
      require(item.tokenCount == 0, "decode plan carried a variable row count");
    }
    scheduler.commit(plan);
    std::vector<StepResult> results;
    for (const BatchItem &item : plan.items) {
      results.push_back({item.requestId, 0, false, DecodeStage::Regular});
    }
    scheduler.complete(plan, results);
    require(scheduler.snapshot().decodeBatchesByWidth[width - 1] == 1,
            "decode width counter did not record the real command");
  }
}

void testNoCrossCohortBatch() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 1, BatchCohort::Greedy));
  scheduler.submit(request(2, 1, BatchCohort::Sampling));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 1);
  BatchPlan plan = *scheduler.next();
  require(plan.width() == 1,
          "decode mixed incompatible consumer policies in one graph");
}

void testPrefillAndDecodeAlternateWithoutStarvation() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 10'000));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 0);

  BatchPlan decode = *scheduler.next();
  require(decode.kind == WorkKind::Decode && decode.items[0].requestId == 1,
          "decode did not retain the first ready command");
  scheduler.commit(decode);
  const std::array decodeResult{
      StepResult{1, 0, false, DecodeStage::Regular}};
  scheduler.complete(decode, decodeResult);

  BatchPlan prefill = *scheduler.next();
  require(prefill.kind == WorkKind::Prefill &&
              prefill.items[0].requestId == 2,
          "continuous decode starved a newly admitted prefill");
  completePrefill(scheduler, prefill);

  BatchPlan nextDecode = *scheduler.next();
  require(nextDecode.kind == WorkKind::Decode &&
              nextDecode.items[0].requestId == 1,
          "prefill did not yield the next command back to decode");
}

void testMeasuredBudgetOnlyLimitsContendedWork() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  const BatchPlan first = *scheduler.next();
  require(first.items[0].tokenCount == 2048,
          "unmeasured isolated work lost the full prefill budget");
  completePrefill(scheduler, first, 4096.0);
  require(scheduler.next()->items[0].tokenCount == 2048,
          "slow isolated work lost the full prefill budget");

  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler, false, 1e9);
  const BatchPlan contended = *scheduler.next();
  require(contended.kind == WorkKind::Prefill &&
              contended.items[0].tokenCount == 128,
          "measured slow prefill was not shortened for a decoding peer");
  completePrefill(scheduler, contended, 256.0);
  completeDecode(scheduler, true);
  require(scheduler.next()->items[0].tokenCount == 2048,
          "prefill did not recover isolated throughput after its peer ended");
}

void testAuxiliaryWorkDoesNotTrainTextPrefillTiming() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next(), 4096.0);
  completePrefill(scheduler, *scheduler.next(), 1e9, false);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler);
  require(scheduler.next()->items.front().tokenCount == 128,
          "auxiliary encoding latency contaminated the text throughput estimate");
}

void testMeasuredBudgetUsesActualRowsAndRecovers() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 128));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next(), 512.0);
  scheduler.submit(request(2, 20'000));
  scheduler.resourcesReady(2, 0);
  completeDecode(scheduler);
  BatchPlan plan = *scheduler.next();
  require(plan.items[0].tokenCount == 64,
          "prefill timing was normalized by capacity instead of actual rows");

  for (uint32_t sample = 0; sample < 8; ++sample) {
    completePrefill(scheduler, plan, 0.25 * plan.items[0].tokenCount);
    completeDecode(scheduler);
    plan = *scheduler.next();
  }
  require(plan.items[0].tokenCount > 64 && plan.items[0].tokenCount < 2048,
          "prefill budget did not adapt when measured execution became faster");
}

void testMeasuredBudgetPreservesPriorityAndStateBoundaries() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 20'000, BatchCohort::Greedy,
                           RequestPriority::Foreground));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next(), 4096.0);
  scheduler.submit(request(2, 1, BatchCohort::Greedy,
                           RequestPriority::Background));
  scheduler.resourcesReady(2, 1);
  require(scheduler.next()->items[0].tokenCount == 2048,
          "lower-priority work reduced foreground throughput");

  scheduler.submit(request(3, 20'000, BatchCohort::Greedy,
                           RequestPriority::Foreground));
  scheduler.resourcesReady(3, 0);
  scheduler.submit(request(4, 1, BatchCohort::Greedy,
                           RequestPriority::Foreground));
  scheduler.resourcesReady(4, 1);
  completeDecode(scheduler);
  scheduler.setPrefillBoundary(1, 2080);
  const BatchPlan plan = *scheduler.next();
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
    engine::Scheduler scheduler;
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    completePrefill(scheduler, *scheduler.next(), wallMilliseconds);
    scheduler.submit(request(2, 1));
    scheduler.resourcesReady(2, 1);
    completeDecode(scheduler);
    const BatchPlan plan = *scheduler.next();
    require(plan.items[0].tokenCount ==
                (wallMilliseconds == 1e9 ? 64u : 2048u),
            "missing timing or extreme sample produced an invalid prefill budget");
  }
}

void testMeasuredBudgetDoesNotCountBlockedPeers() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next(), 4096.0);
  scheduler.submit(request(2, 20'000));
  scheduler.waitForResources(2);
  scheduler.submit(request(3, 20'000));
  require(scheduler.next()->items[0].tokenCount == 2048,
          "queued or resource-blocked work reduced resident throughput");
}

void testMeasuredBudgetPreservesPurePrefillPacking() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next(), 4096.0);
  for (uint64_t id = 2; id <= 4; ++id) {
    scheduler.submit(request(id, 20'000));
    scheduler.resourcesReady(id, 0);
  }
  scheduler.setPrefillBoundary(1, 2080);
  const BatchPlan plan = *scheduler.next();
  require(plan.kind == WorkKind::Prefill && plan.width() == 2 &&
              plan.items[0].tokenCount == 32 &&
              plan.items[1].tokenCount == 2016,
          "pure prefill contention lost throughput or stopped packing peers");
}

void testTinyTailDoesNotDistortPrefillThroughput() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 2056));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next(), 2048.0);
  completePrefill(scheduler, *scheduler.next(), 800.0);
  scheduler.submit(request(2, 20'000));
  scheduler.resourcesReady(2, 0);
  completeDecode(scheduler);
  require(scheduler.next()->items[0].tokenCount == 256,
          "fixed overhead from a tiny tail distorted the prefill estimate");
}

void testMeasuredBudgetFinishesShortPrefillPromptly() {
  for (const auto priority : {RequestPriority::Normal,
                              RequestPriority::Background}) {
    engine::Scheduler scheduler;
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    completePrefill(scheduler, *scheduler.next(), 4096.0);
    scheduler.submit(request(2, 31, BatchCohort::Greedy, priority));
    scheduler.resourcesReady(2, 0);
    const BatchPlan plan = *scheduler.next();
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
    const BatchPlan next = *scheduler.next();
    require(next.kind == WorkKind::Decode && next.items[0].requestId == 2,
            "the completed short prefill did not get its next decode turn");
  }
}

void testMeasuredBudgetRetainsOvertakingBound() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next(), 4096.0);
  scheduler.submit(request(100, 1));
  scheduler.resourcesReady(100, 1);
  for (uint64_t id = 2; id <= 4; ++id) {
    completeDecode(scheduler);
    scheduler.submit(request(id, 128));
    scheduler.resourcesReady(id, 0);
    const BatchPlan plan = *scheduler.next();
    require(plan.items.size() == 1 && plan.items[0].requestId == id,
            "adaptive prefill did not serve a short arrival first");
    scheduler.commit(plan);
    const std::array result{StepResult{id, 128, true, DecodeStage::Regular}};
    scheduler.complete(plan, result, 256.0);
  }
  scheduler.submit(request(5, 128));
  scheduler.resourcesReady(5, 0);
  completeDecode(scheduler);
  const BatchPlan overdue = *scheduler.next();
  require(overdue.items.size() == 1 && overdue.items[0].requestId == 1 &&
              overdue.items[0].tokenCount == 128,
          "adaptive prefill allowed short arrivals to starve an older lane");
}

void testPriorityPrecedesWorkKindAlternation() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 1, BatchCohort::Greedy,
                           RequestPriority::Background));
  scheduler.submit(request(2, 100, BatchCohort::Greedy,
                           RequestPriority::Foreground));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 0);

  BatchPlan foreground = *scheduler.next();
  require(foreground.kind == WorkKind::Prefill &&
              foreground.items[0].requestId == 2,
          "work-kind alternation overrode request priority");
}

void testPrefillCommandContainsOnePriorityTier() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 4096, BatchCohort::Greedy,
                           RequestPriority::Foreground));
  scheduler.submit(request(2, 4096, BatchCohort::Greedy,
                           RequestPriority::Background));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);

  BatchPlan plan = *scheduler.next();
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "ragged prefill mixed priority tiers in one command");
}

void testDecodeCommandContainsOnePriorityTier() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 1, BatchCohort::Greedy,
                           RequestPriority::Foreground));
  scheduler.submit(request(2, 1, BatchCohort::Greedy,
                           RequestPriority::Background));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 1);

  BatchPlan plan = *scheduler.next();
  require(plan.kind == WorkKind::Decode && plan.width() == 1 &&
              plan.items[0].requestId == 1,
          "fixed-eight decode mixed priority tiers in one command");
  scheduler.commit(plan);
  const std::array result{
      StepResult{1, 0, true, DecodeStage::Regular}};
  scheduler.complete(plan, result);

  BatchPlan background = *scheduler.next();
  require(background.kind == WorkKind::Decode && background.width() == 1 &&
              background.items[0].requestId == 2,
          "background decode did not run after foreground completion");
}

void testDecodeCohortsAndLanesRotate() {
  engine::Scheduler scheduler;
  for (uint64_t id = 1; id <= 5; ++id) {
    scheduler.submit(request(id, 1, BatchCohort::Greedy));
    scheduler.resourcesReady(id, 1);
  }
  scheduler.submit(request(6, 1, BatchCohort::Sampling));
  scheduler.resourcesReady(6, 1);

  BatchPlan first = *scheduler.next();
  require(first.kind == WorkKind::Decode && first.width() == 4 &&
              first.cohort == BatchCohort::Greedy,
          "first compatible decode cohort did not use the real B4 graph");
  scheduler.commit(first);
  std::vector<StepResult> firstResults;
  for (const BatchItem &item : first.items) {
    firstResults.push_back(
        {item.requestId, 0, false, DecodeStage::Regular});
  }
  scheduler.complete(first, firstResults);

  BatchPlan second = *scheduler.next();
  require(second.kind == WorkKind::Decode &&
              std::any_of(second.items.begin(), second.items.end(),
                          [](const BatchItem &item) {
                            return item.requestId == 5;
                          }),
          "an unscheduled lane was starved by the previous B4 members");
  scheduler.commit(second);
  std::vector<StepResult> secondResults;
  for (const BatchItem &item : second.items) {
    secondResults.push_back(
        {item.requestId, 0, false, DecodeStage::Regular});
  }
  scheduler.complete(second, secondResults);

  BatchPlan third = *scheduler.next();
  require(third.kind == WorkKind::Decode && third.width() == 1 &&
              third.items[0].requestId == 6 &&
              third.cohort == BatchCohort::Sampling,
          "a different decode cohort was starved by a continuous leader");
}

void testMaskStagesNeverMix() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 1, BatchCohort::Constrained));
  scheduler.submit(request(2, 1, BatchCohort::Constrained));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 1);

  BatchPlan initialRequest = *scheduler.next();
  require(initialRequest.width() == 2 &&
              initialRequest.decodeStage == DecodeStage::RequestInitialMask,
          "initial mask requests did not form one compatible batch");
  scheduler.commit(initialRequest);
  const std::array initialResults{
      StepResult{1, 0, false, DecodeStage::ApplyInitialMask},
      StepResult{2, 0, false, DecodeStage::ApplyInitialMask},
  };
  scheduler.complete(initialRequest, initialResults);
  scheduler.maskReady(1);

  BatchPlan initialResume = *scheduler.next();
  require(initialResume.width() == 1 && initialResume.items[0].requestId == 1 &&
              initialResume.decodeStage == DecodeStage::ApplyInitialMask,
          "initial-mask continuation lost its executor stage");
  scheduler.commit(initialResume);
  // Draft, target forward, host-mask wait, and commit are one scheduler-owned
  // model ticket. The scheduler therefore sees the next ordinary decode
  // stage only after the entire constrained cycle completes.
  const std::array initialResumeResult{
      StepResult{1, 0, false, DecodeStage::Regular}};
  scheduler.complete(initialResume, initialResumeResult);

  scheduler.maskReady(2);
  BatchPlan otherInitial = *scheduler.next();
  require(otherInitial.width() == 1 &&
              otherInitial.items[0].requestId == 2 &&
              otherInitial.decodeStage == DecodeStage::ApplyInitialMask,
          "initial-anchor selections were batched before classification");
  scheduler.commit(otherInitial);
  const std::array otherInitialResult{
      StepResult{2, 0, true, DecodeStage::Regular}};
  scheduler.complete(otherInitial, otherInitialResult);

  BatchPlan verify = *scheduler.next();
  require(verify.width() == 1 && verify.items[0].requestId == 1 &&
              verify.decodeStage == DecodeStage::Regular,
          "completed constraint cycle mixed with an initial-mask continuation");
  scheduler.commit(verify);
  const std::array verifyResult{
      StepResult{1, 0, true, DecodeStage::Regular}};
  scheduler.complete(verify, verifyResult);
}

void testWaitingMaskExpiresAtRequestDeadline() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 1, BatchCohort::Constrained));
  scheduler.resourcesReady(1, 1);

  const BatchPlan plan = *scheduler.next();
  scheduler.commit(plan);
  const std::array result{
      StepResult{1, 0, false, DecodeStage::ApplyInitialMask}};
  scheduler.complete(plan, result);
  require(scheduler.phase(1) == engine::Phase::WaitingMask,
          "constrained request did not wait for its CPU mask");
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

void testResourceSuspensionReplaysFromCacheAndPreservesDecodeStage() {
  engine::Scheduler scheduler;
  scheduler.submit(request(1, 4096));
  scheduler.resourcesReady(1, 0);
  BatchPlan first = *scheduler.next();
  completePrefill(scheduler, first);
  scheduler.suspendForResources(1);
  require(scheduler.phase(1) == engine::Phase::WaitingResources,
          "resident prefill did not enter resource wait");
  scheduler.resumeFromResources(1, 32, 4096);
  BatchPlan resumed = *scheduler.next();
  require(resumed.kind == WorkKind::Prefill &&
              resumed.items[0].promptOffset == 32,
          "resource resume did not start from its acquired cache boundary");

  engine::Scheduler decode;
  decode.submit(request(2, 1, BatchCohort::Constrained));
  decode.resourcesReady(2, 1);
  auto initial = *decode.next();
  decode.commit(initial);
  const std::array initialResult{
      StepResult{2, 0, false, DecodeStage::ApplyInitialMask}};
  decode.complete(initial, initialResult);
  decode.maskReady(2);
  decode.suspendForResources(2);
  decode.resumeFromResources(2, 32, 40);
  const auto replay = *decode.next();
  require(replay.kind == WorkKind::Prefill &&
              replay.items[0].promptOffset == 32 &&
              replay.items[0].tokenCount == 8,
          "decode resume did not replay its committed token history");
  completePrefill(decode, replay);
  const auto continuation = *decode.next();
  require(continuation.kind == WorkKind::Decode &&
              continuation.decodeStage == DecodeStage::ApplyInitialMask,
          "replay reset the consumer's existing decode/mask stage");
}

} // namespace

int main() {
  try {
    testAdmissionSharesDispatchOrderAndBudget();
    testAdmissionRespectsContendedBudgetAndDecodePriority();
    testQueuedPrefillCannotBeOvertakenIndefinitely();
    testWarmupTimingSeedsFirstContendedCommand();
    testShortestRemainingFirstUsesActualRows();
    testPerRequestBoundary();
    testEqualPromptsFinishInArrivalOrder();
    testShortArrivalPrecedesLongColdPrompt();
    testBoundaryCapsDispatchWithoutChangingPriority();
    testEqualLanesRunInArrivalOrderWithoutOvertaking();
    testLaneOvertakenThreeTimesLeadsTheNextCommand();
    testServedLaneResetsOvertaking();
    testRealDecodeWidths();
    testNoCrossCohortBatch();
    testPrefillAndDecodeAlternateWithoutStarvation();
    testMeasuredBudgetOnlyLimitsContendedWork();
    testAuxiliaryWorkDoesNotTrainTextPrefillTiming();
    testMeasuredBudgetUsesActualRowsAndRecovers();
    testMeasuredBudgetPreservesPriorityAndStateBoundaries();
    testUnavailableTimingAndMinimumBudget();
    testMeasuredBudgetDoesNotCountBlockedPeers();
    testMeasuredBudgetPreservesPurePrefillPacking();
    testTinyTailDoesNotDistortPrefillThroughput();
    testMeasuredBudgetFinishesShortPrefillPromptly();
    testMeasuredBudgetRetainsOvertakingBound();
    testPriorityPrecedesWorkKindAlternation();
    testPrefillCommandContainsOnePriorityTier();
    testDecodeCommandContainsOnePriorityTier();
    testDecodeCohortsAndLanesRotate();
    testMaskStagesNeverMix();
    testWaitingMaskExpiresAtRequestDeadline();
    testResourceSuspensionReplaysFromCacheAndPreservesDecodeStage();
    std::cout << "ragged scheduler tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ragged scheduler tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
