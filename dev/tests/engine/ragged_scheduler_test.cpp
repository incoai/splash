// The scheduler (engine::Scheduler): its phases, decode commands and debt,
// and the prefill rules: grouping, passes and cuts sized from the last
// command's time, the per-layer pair bound, joins, order and aging, and a
// higher priority abandoning a command between its passes. Prefill targets
// are the fakes' fitted profiles (TimingProfile.cpp).

#include "TestChecks.hpp"
#include "TestRegistry.hpp"
#include "TimingProfile.hpp"
#include "engine/Scheduler.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using test::rejects;
using test::require;

constexpr uint32_t kChunk = model::ExecutionLimits::prefillTokenBudget;
constexpr uint32_t kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
// The prefill commands in a row that may serve later arrivals of a request's
// priority and not it before it is overdue (Scheduler.cpp).
constexpr uint32_t kOvertakes = model::ExecutionLimits::maximumBatchWidth - 1;
constexpr double kGap = 500.0;

model::CommandLimits limits(uint32_t commandRows = kChunk, uint32_t passes = 1, uint32_t indexPageLanes = 1,
                            uint64_t attentionPairs = std::numeric_limits<uint64_t>::max()) {
  return {.commandRows = commandRows,
          .layers = passes,
          .maximumPasses = passes,
          .lanesPastChunk = 1,
          .indexPageLanes = indexPageLanes,
          .attentionPairs = attentionPairs};
}

// The limits the target reports: whether its experts stream is the profile's.
model::CommandLimits reported(const test::TimingProfile &profile, model::CommandLimits commandLimits) {
  commandLimits.streamsWeights = test::streamsExperts(profile);
  return commandLimits;
}

RequestSpec request(uint64_t id, uint32_t prompt, RequestPriority priority = RequestPriority::Normal) {
  return {id, priority, prompt};
}

// The turnarounds u the pass tests run at (TimingProfile.cpp: the profiles'
// 60 ms is the middle one).
constexpr std::array<double, 3> kTurnarounds{5.0, 60.0, 150.0};

test::TimingProfile withTurnaround(const test::TimingProfile &profile, double turnaround) {
  test::TimingProfile result = profile;
  result.turnaroundMilliseconds = turnaround;
  return result;
}

std::vector<test::TimingProfile> atEveryTurnaround(std::initializer_list<const test::TimingProfile *> profiles) {
  std::vector<test::TimingProfile> result;
  for (const test::TimingProfile *profile : profiles)
    for (const double turnaround : kTurnarounds)
      result.push_back(withTurnaround(*profile, turnaround));
  return result;
}

std::string at(const test::TimingProfile &profile) {
  return std::string(profile.name) + " at u " + std::to_string(int(profile.turnaroundMilliseconds)) + " ms";
}

// A target, its scheduler and what the engine gives the scheduler: the
// lanes' shapes, the lanes left out, the command between its passes, which
// run() keeps as the engine's PassCommand does, and for admission the starts
// restoring their prefix. Commands take the profile's time on a clock.
class Sim {
public:
  explicit Sim(const test::TimingProfile &profile, model::CommandLimits commandLimits = limits(),
               double decodeShare = 0.5, std::function<double()> residentShare = {})
      : work(profile, std::move(residentShare)), commandLimits(reported(profile, commandLimits)),
        scheduler({.decodeShare = decodeShare, .decodeGapMilliseconds = kGap, .limits = this->commandLimits}, work) {}

  test::TimedWork work;
  model::CommandLimits commandLimits;
  Scheduler scheduler;
  std::vector<LaneShape> shapes;
  std::vector<uint64_t> excluded;
  std::optional<PassProgress> passes;
  std::vector<PrefillAdmission> restoring;
  double now = 0.0;

  [[nodiscard]] PlanInputs in() const { return {excluded, shapes, passes ? &*passes : nullptr, restoring}; }
  [[nodiscard]] std::optional<BatchPlan> next() const { return scheduler.next(in()); }
  [[nodiscard]] BatchPlan plan() const {
    const std::optional<BatchPlan> result = next();
    require(result.has_value(), "the scheduler planned nothing");
    return *result;
  }
  // A resident prefill lane at `position` with `remaining` rows to go, and a
  // decoding lane.
  void lane(uint64_t id, uint32_t position, uint32_t remaining, RequestPriority priority = RequestPriority::Normal) {
    scheduler.submit(request(id, position + remaining, priority));
    scheduler.resourcesReady(id, position);
  }
  void decoder(uint64_t id, RequestPriority priority = RequestPriority::Normal) {
    scheduler.submit(request(id, 1, priority));
    scheduler.resourcesReady(id, 1);
  }
  void shape(uint64_t id, uint32_t indexPages, bool gainsIndexPages = false) {
    std::erase_if(shapes, [id](const LaneShape &shape) { return shape.requestId == id; });
    shapes.push_back({.requestId = id, .indexPages = indexPages, .gainsIndexPages = gainsIndexPages});
  }
  [[nodiscard]] std::vector<model::PrefillLane> lanes(std::span<const BatchItem> items) const {
    std::vector<model::PrefillLane> result;
    for (const BatchItem &item : items) {
      const auto found = std::ranges::find(shapes, item.requestId, &LaneShape::requestId);
      result.push_back({item.promptOffset, item.tokenCount, found == shapes.end() ? 0 : found->indexPages});
    }
    return result;
  }
  // Runs `plan` as the engine does, its time the profile's (times `scale`,
  // its sample as `sampled` leaves it): a prefill's lanes consume their rows
  // at its last pass, a decode's commit eight tokens each.
  double run(const BatchPlan &plan, double scale = 1.0) {
    // A plan that supersedes the command between its passes abandons it
    // first (Engine::tick).
    if (scheduler.abandonsPasses(plan, in()))
      abandon();
    scheduler.commit(plan, in());
    if (plan.kind == WorkKind::Prefill && plan.passes > 1 && !plan.pass)
      passes = PassProgress{.plan = plan};
    test::CommandTimes times;
    std::vector<StepResult> results;
    const bool last = plan.pass + 1 == plan.passes;
    if (plan.kind == WorkKind::Prefill) {
      times = work.times(lanes(plan.items), plan.layer, plan.passes - plan.pass);
      for (const BatchItem &item : plan.items)
        results.push_back({item.requestId, last ? item.tokenCount : 0, false, DecodeStage::Regular});
    } else {
      times = test::decodeTimes(work.profile(), plan.width(), plan.width() * 8, work.residentShare(), 0);
      for (const BatchItem &item : plan.items)
        results.push_back({item.requestId, 0, false, DecodeStage::Regular, 8});
    }
    times.wallMilliseconds *= scale;
    model::CommandSample sample = test::sampleOf(times);
    if (sampled)
      sampled(sample, plan);
    scheduler.complete(plan, results, sample);
    if (plan.kind == WorkKind::Prefill && passes) {
      if (last) {
        passes.reset();
      } else {
        passes->plan = scheduler.nextPass(passes->plan);
      }
    }
    now += times.wallMilliseconds;
    return times.wallMilliseconds;
  }
  double step(double scale = 1.0) { return run(plan(), scale); }
  std::function<void(model::CommandSample &, const BatchPlan &)> sampled;
  // Abandons the command between its passes, as PassCommand::yield does.
  void abandon() {
    scheduler.abandoned();
    passes.reset();
  }
};

// A dense 27B target of one chunk a command.
class DenseSim final : public Sim {
public:
  explicit DenseSim(double decodeShare = 0.0) : Sim(test::kDense27B, limits(), decodeShare) {}
};

uint32_t rowsOf(const BatchPlan &plan) {
  uint32_t rows = 0;
  for (const BatchItem &item : plan.items)
    rows += item.tokenCount;
  return rows;
}

void completeDecode(Sim &sim, bool finished = false) {
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Decode, "expected a decode command");
  sim.scheduler.commit(plan, sim.in());
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items)
    results.push_back({item.requestId, 0, finished, DecodeStage::Regular, 1});
  sim.scheduler.complete(plan, results, {.wallMilliseconds = 50.0});
}

// Commits a prefill unit and completes it in `sample`'s time, its lanes
// consuming their rows at its last pass.
void completePrefill(Sim &sim, const BatchPlan &plan, const model::CommandSample &sample = {}) {
  require(plan.kind == WorkKind::Prefill, "expected a prefill command");
  sim.scheduler.commit(plan, sim.in());
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items)
    results.push_back(
        {item.requestId, plan.pass + 1 == plan.passes ? item.tokenCount : 0, false, DecodeStage::Regular});
  sim.scheduler.complete(plan, results, sample);
}

// Rounds that measure turns' commands: an 8K prefix, then a 300-row turn on
// it alone and two together, each leaving at its first token.
void measureTurns(Sim &sim) {
  struct Round final {
    uint32_t position, rows, turns;
  };
  uint64_t id = 1000;
  for (const Round round : {Round{0, 8192, 1}, Round{8192, 300, 1}, Round{8192, 300, 2}}) {
    std::vector<uint64_t> waiting;
    for (uint32_t turn = 0; turn < round.turns; ++turn) {
      sim.lane(id, round.position, round.rows);
      waiting.push_back(id++);
    }
    while (!waiting.empty()) {
      sim.step();
      std::erase_if(waiting, [&](uint64_t lane) {
        if (sim.scheduler.phase(lane) == Phase::Prefill)
          return false;
        sim.scheduler.cancel(lane);
        sim.scheduler.remove(lane);
        return true;
      });
    }
  }
}

// Lane 1 decodes beside lane 2's long prompt: a decode command, then a
// prefill unit of 500 ms, which owes decode its share of that.
void runContendedPrefill(Sim &sim) {
  sim.decoder(1);
  sim.lane(2, 0, 20'000);
  completeDecode(sim);
  const BatchPlan prefill = sim.plan();
  require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 2,
          "the long prompt did not follow the first decode command");
  completePrefill(sim, prefill, {.wallMilliseconds = 500.0});
}

// Prefills a constrained request's prompt, whose end waits for the mask of
// its first token.
void awaitFirstMask(Sim &sim, uint64_t id) {
  BatchPlan prompt = sim.plan();
  require(prompt.kind == WorkKind::Prefill && prompt.width() == 1 && prompt.items[0].requestId == id,
          "expected the constrained prompt");
  while (prompt.pass + 1 < prompt.passes) {
    sim.run(prompt);
    prompt = sim.plan();
  }
  sim.scheduler.commit(prompt, sim.in());
  const std::array result{StepResult{id, prompt.items[0].tokenCount, false, DecodeStage::ApplyInitialMask}};
  sim.scheduler.complete(prompt, result, {});
  sim.passes.reset();
  require(sim.scheduler.phase(id) == Phase::WaitingMask, "a constrained prompt did not wait for its first mask");
}

// --- Phases, decode and debt -------------------------------------------------------------------------------

void testRealDecodeWidths() {
  for (uint32_t width = 1; width <= 4; ++width) {
    DenseSim sim;
    for (uint32_t lane = 0; lane < width; ++lane)
      sim.decoder(lane + 1);
    const BatchPlan plan = sim.plan();
    require(plan.kind == WorkKind::Decode && plan.width() == width && rowsOf(plan) == 0,
            "ready decode width was delayed or rewritten");
    sim.run(plan);
    require(sim.scheduler.snapshot().decodeBatchesByWidth[width - 1] == 1,
            "decode width counter did not record the real command");
  }
}

void testRejectedCommitCountsNothing() {
  DenseSim sim;
  sim.decoder(1);
  sim.decoder(2);
  const BatchPlan stale = sim.plan();
  sim.scheduler.cancel(2);
  rejects([&] { static_cast<void>(sim.scheduler.commit(stale, sim.in())); },
          "batch no longer matches scheduler state", "a commit of a cancelled request's batch was accepted");
  require(sim.scheduler.snapshot().decodeBatches == 0, "rejected commit incremented decode counters");
  completeDecode(sim);
  require(sim.scheduler.snapshot().decodeBatches == 1, "the remaining request's decode was not counted");
}

void testDecodeLanesRotate() {
  DenseSim sim;
  for (uint64_t id = 1; id <= 6; ++id)
    sim.decoder(id);
  const BatchPlan first = sim.plan();
  require(first.kind == WorkKind::Decode && first.width() == 4, "the first decode did not use the real B4 graph");
  sim.run(first);
  const BatchPlan second = sim.plan();
  require(second.width() == 4 && second.items[0].requestId == 5 && second.items[1].requestId == 6,
          "decode did not prioritize lanes omitted by the previous batch");
}

void testMaskStagesNeverMix() {
  DenseSim sim;
  sim.lane(1, 0, 1);
  sim.lane(2, 0, 1);
  const BatchPlan prompts = sim.plan();
  require(prompts.kind == WorkKind::Prefill && prompts.width() == 2, "constrained prompts did not form one prefill");
  sim.scheduler.commit(prompts, sim.in());
  const std::array promptResults{StepResult{1, 1, false, DecodeStage::ApplyInitialMask},
                                 StepResult{2, 1, false, DecodeStage::ApplyInitialMask}};
  sim.scheduler.complete(prompts, promptResults, {});
  sim.scheduler.maskReady(1);
  const BatchPlan initial = sim.plan();
  require(initial.width() == 1 && initial.items[0].requestId == 1 &&
              initial.decodeStage == DecodeStage::ApplyInitialMask,
          "a ready first mask did not get its selection plan");
  sim.scheduler.commit(initial, sim.in());
  const std::array selected{StepResult{1, 0, false, DecodeStage::Regular}};
  sim.scheduler.complete(initial, selected, {});
  sim.scheduler.maskReady(2);
  const BatchPlan other = sim.plan();
  require(other.width() == 1 && other.items[0].requestId == 2 && other.decodeStage == DecodeStage::ApplyInitialMask,
          "a first token's selection mixed with a drafting lane");
}

void testPrefillAndDecodeAlternateWithoutStarvation() {
  DenseSim sim;
  sim.decoder(1);
  sim.lane(2, 0, 10'000);
  require(sim.plan().kind == WorkKind::Decode, "decode did not retain the first ready command");
  completeDecode(sim);
  const BatchPlan prefill = sim.plan();
  require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 2,
          "continuous decode starved a newly admitted prefill");
  sim.run(prefill);
  require(sim.plan().kind == WorkKind::Decode, "prefill did not yield the next command back to decode");
}

// Lane 1 decodes beside lane 2's long prompt: one decode command, then one
// prefill unit of 500 ms owes half of it at a share of 0.5.
void testDecodeRepaysItsShareOfContendedPrefill() {
  for (const double share : {0.0, 0.5}) {
    DenseSim sim(share);
    sim.decoder(1);
    sim.lane(2, 0, 20'000);
    completeDecode(sim);
    const BatchPlan prefill = sim.plan();
    sim.scheduler.commit(prefill, sim.in());
    std::vector<StepResult> results;
    for (const BatchItem &item : prefill.items)
      results.push_back({item.requestId, prefill.pass + 1 == prefill.passes ? item.tokenCount : 0, false,
                         DecodeStage::Regular});
    sim.scheduler.complete(prefill, results, {.wallMilliseconds = 500.0});
    for (uint32_t step = 0; step < (share > 0.0 ? 5u : 1u); ++step)
      completeDecode(sim);
    require(sim.plan().kind == WorkKind::Prefill, "decode did not return the next command once its share was repaid");
  }
}

// Lanes left out of a plan are not planned; a prefill owes nothing to the
// decoders it was planned without.
void testExcludedLanesAreNotPlanned() {
  DenseSim sim(0.5);
  sim.decoder(1);
  sim.decoder(2);
  sim.lane(3, 0, 20'000);
  sim.excluded = {1};
  const BatchPlan decode = sim.plan();
  require(decode.kind == WorkKind::Decode && decode.width() == 1 && decode.items[0].requestId == 2,
          "a decode plan included a lane left out of it");
  sim.excluded = {1, 2};
  const BatchPlan prefill = sim.plan();
  require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 3,
          "the prefill did not run while every decoder was left out");
  sim.run(prefill);
  sim.excluded.clear();
  completeDecode(sim);
  require(sim.plan().kind == WorkKind::Prefill, "a prefill owed decode time to decoders it was planned without");
}

void testPriorityPrecedesWorkKindAlternation() {
  DenseSim sim;
  sim.decoder(1, RequestPriority::Background);
  sim.lane(2, 0, 100, RequestPriority::Foreground);
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 2,
          "work-kind alternation overrode request priority");
}

void testPrefillCommandContainsOnePriorityTier() {
  DenseSim sim;
  sim.lane(1, 0, 100, RequestPriority::Foreground);
  sim.lane(2, 0, 100, RequestPriority::Background);
  const BatchPlan plan = sim.plan();
  require(plan.width() == 1 && plan.items[0].requestId == 1, "a prefill command mixed priority tiers");
}

// The tier admission can select is set by the lanes that prefill or decode,
// not by those waiting for a mask or for admission.
void testHighestRunnablePriority() {
  DenseSim sim;
  require(!sim.scheduler.highestRunnablePriority(), "an idle scheduler had a runnable tier");
  sim.lane(1, 0, 1, RequestPriority::Foreground);
  awaitFirstMask(sim, 1);
  sim.decoder(2, RequestPriority::Background);
  sim.lane(3, 0, 100);
  sim.scheduler.submit(request(4, 100, RequestPriority::Foreground));
  require(sim.scheduler.highestRunnablePriority() == RequestPriority::Normal,
          "a mask wait or a queued request set the runnable tier");
}

void testResourceSuspensionReplaysFromCache() {
  DenseSim sim;
  sim.lane(1, 0, 4096);
  sim.step();
  sim.scheduler.suspendForResources(1);
  require(sim.scheduler.phase(1) == Phase::WaitingResources && sim.scheduler.suspended(1),
          "resident prefill did not enter resource wait");
  sim.scheduler.resumeFromResources(1, 32, 4096);
  const BatchPlan resumed = sim.plan();
  require(resumed.kind == WorkKind::Prefill && resumed.items[0].promptOffset == 32,
          "resource resume did not start from its acquired cache boundary");
}

// Admission starts only the candidates the next command serves.
void testAdmissionStartsTheLanesOfTheNextCommand() {
  DenseSim sim;
  sim.lane(1, 0, 41'000);
  sim.scheduler.submit(request(2, 38'000));
  sim.scheduler.submit(request(3, 30'000));
  const std::array queued{PrefillAdmission{2, 0}, PrefillAdmission{3, 0}};
  require(sim.scheduler.prefillAdmissionOrder(queued, sim.in()) == std::vector<uint64_t>{3},
          "admission did not start the candidate the next command leads with");
}

// Admission starts at most a full batch: of eight measured turns that share a
// command, the first four.
void testAdmissionStartsAtMostAFullBatch() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  measureTurns(sim);
  std::vector<PrefillAdmission> turns;
  for (uint64_t id = 1; id <= 8; ++id) {
    sim.scheduler.submit(request(id, 8192 + 300));
    turns.push_back({id, 8192});
  }
  require(sim.scheduler.prefillAdmissionOrder(turns, sim.in()) == std::vector<uint64_t>({1, 2, 3, 4}),
          "admission of short prompts lost their sharing or exceeded the real width");
}

// Admission ranks candidates against every decoder: a higher priority's
// decode holds a lower priority's prompts back until it leaves.
void testAdmissionWaitsForAHigherPrioritysDecode() {
  DenseSim sim;
  std::vector<PrefillAdmission> candidates;
  for (uint64_t id = 1; id <= 2; ++id) {
    sim.scheduler.submit(request(id, 65));
    candidates.push_back({id, 0});
  }
  require(!sim.scheduler.prefillAdmissionOrder(candidates, sim.in()).empty(), "admission started nothing");
  sim.decoder(5, RequestPriority::Foreground);
  require(sim.scheduler.prefillAdmissionOrder(candidates, sim.in()).empty(),
          "a lower priority's prefill started ahead of a runnable Foreground decode");
  sim.scheduler.cancel(5);
  require(!sim.scheduler.prefillAdmissionOrder(candidates, sim.in()).empty(),
          "prefill admission did not resume after the Foreground decode left");
}

// Two constrained prompts wait for their first masks; their first tokens'
// selections share a plan of their stage, and once selected they decode with
// the other lanes.
void testMaskedLanesShareDecode() {
  for (const bool selected : {false, true}) {
    DenseSim sim;
    sim.lane(1, 0, 1);
    sim.lane(2, 0, 1);
    const BatchPlan prompts = sim.plan();
    sim.scheduler.commit(prompts, sim.in());
    const std::array promptResults{StepResult{1, 1, false, DecodeStage::ApplyInitialMask},
                                   StepResult{2, 1, false, DecodeStage::ApplyInitialMask}};
    sim.scheduler.complete(prompts, promptResults, {});
    sim.scheduler.maskReady(1);
    sim.scheduler.maskReady(2);
    if (!selected) {
      const BatchPlan selection = sim.plan();
      require(selection.kind == WorkKind::Decode && selection.width() == 2 &&
                  selection.decodeStage == DecodeStage::ApplyInitialMask,
              "first-token selections of one priority did not share a plan");
      continue;
    }
    completeDecode(sim);
    sim.decoder(3);
    sim.decoder(4);
    const BatchPlan plan = sim.plan();
    require(plan.kind == WorkKind::Decode && plan.width() == 4 && plan.decodeStage == DecodeStage::Regular,
            "lanes that waited for their first masks decoded apart from the others");
  }
}

// A prefill beside a lane waiting for its mask owes decode nothing: the lane
// could not have decoded meanwhile.
void testMaskWaitAccruesNoDecodeDebt() {
  DenseSim sim(0.5);
  sim.lane(1, 0, 1);
  awaitFirstMask(sim, 1);
  sim.lane(2, 0, 20'000);
  for (uint32_t command = 0; command < 3; ++command)
    completePrefill(sim, sim.plan(), {.wallMilliseconds = 500.0});
  sim.scheduler.maskReady(1);
  completeDecode(sim);
  require(sim.plan().kind == WorkKind::Prefill, "prefill beside a lane waiting for its mask banked decode debt");
}

// Debt is owed only to lanes that decode: once the last leaves, by its
// decode command's end or a cancel, a later decoder starts without it.
void testDecodeDebtLeavesWithTheLastDecoder() {
  for (const bool finished : {true, false}) {
    DenseSim sim(0.5);
    runContendedPrefill(sim);
    completeDecode(sim, finished);
    if (!finished)
      sim.scheduler.cancel(1);
    sim.decoder(3);
    require(sim.plan().kind == WorkKind::Prefill, "a later decoder inherited debt owed to one that left");
  }
}

void testHigherPriorityPrefillPrecedesDecodeDebt() {
  DenseSim sim(0.5);
  runContendedPrefill(sim);
  sim.lane(3, 0, 20'000, RequestPriority::Foreground);
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 3,
          "decode debt overrode a higher-priority prefill");
}

void testDecodeCommandContainsOnePriorityTier() {
  DenseSim sim;
  sim.decoder(1, RequestPriority::Foreground);
  sim.decoder(2, RequestPriority::Background);
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Decode && plan.width() == 1 && plan.items[0].requestId == 1,
          "a decode command mixed priority tiers");
  completeDecode(sim, true);
  const BatchPlan background = sim.plan();
  require(background.kind == WorkKind::Decode && background.width() == 1 && background.items[0].requestId == 2,
          "background decode did not run after foreground completion");
}

// A request suspended while it holds its first mask replays its history and
// selects its first token under that mask, without asking for another.
void testReplayKeepsHeldInitialMask() {
  DenseSim sim;
  sim.lane(2, 0, 1);
  awaitFirstMask(sim, 2);
  sim.scheduler.maskReady(2);
  sim.scheduler.suspendForResources(2);
  sim.scheduler.resumeFromResources(2, 32, 40);
  const BatchPlan replay = sim.plan();
  require(replay.kind == WorkKind::Prefill && replay.items[0].promptOffset == 32 && replay.items[0].tokenCount == 8,
          "decode resume did not replay its committed token history");
  completePrefill(sim, replay);
  require(sim.scheduler.phase(2) == Phase::Decode, "the replay waited for a second first mask");
  const BatchPlan continuation = sim.plan();
  require(continuation.kind == WorkKind::Decode && continuation.decodeStage == DecodeStage::ApplyInitialMask,
          "the replay dropped the held first mask's selection");
}

// A lane waiting for its mask, of the prefill's priority or a higher one,
// bounds the prefill beside it as a decoder does; a lower one does not. A
// mask that arrives during the unit gets the next command; cancelled, the
// lane leaves the prefill unbounded.
void testWaitingMaskBoundsPeerPrefill() {
  for (const RequestPriority priority :
       {RequestPriority::Foreground, RequestPriority::Normal, RequestPriority::Background}) {
    DenseSim sim;
    sim.lane(1, 0, 1, priority);
    awaitFirstMask(sim, 1);
    sim.lane(2, 0, 20'000);
    const BatchPlan prefill = sim.plan();
    const bool bounded = priority <= RequestPriority::Normal;
    require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 2 &&
                (prefill.items[0].tokenCount < kChunk) == bounded,
            "a mask wait lost prefill latency protection or priority ordering");
    sim.scheduler.maskReady(1);
    completePrefill(sim, prefill);
    if (bounded) {
      const BatchPlan decode = sim.plan();
      require(decode.kind == WorkKind::Decode && decode.items[0].requestId == 1,
              "a ready mask did not resume after a bounded prefill");
    }
    sim.scheduler.cancel(1);
    require(sim.plan().items[0].tokenCount == kChunk, "a cancelled mask request kept the prefill bounded");
  }
}

// --- Contention --------------------------------------------------------------------------------------------

// Uncontended work runs whole; beside a decoder of its priority it is
// bounded, and once the decoder ends it runs whole again.
void testOnlyContendedUnitsAreBounded() {
  DenseSim sim;
  sim.lane(1, 0, 20'000);
  require(sim.plan().items[0].tokenCount == kChunk, "uncontended work lost its rows");
  sim.step();
  sim.decoder(2);
  completeDecode(sim);
  const BatchPlan contended = sim.plan();
  require(contended.kind == WorkKind::Prefill && contended.passes == 1 && contended.items[0].tokenCount < kChunk &&
              contended.items[0].tokenCount >= 512,
          "a contended dense unit was not cut to its floor or more");
  sim.run(contended);
  completeDecode(sim, true);
  require(sim.plan().items[0].tokenCount == kChunk, "prefill did not run whole again once its peer ended");
}

// Dense and resident-expert commands never run in passes. Contended, a unit
// is cut toward G from the floor, max(512, the model's least rows): the floor
// before any command ran, then the rows the last command ran in G, never
// fewer than the floor but at the prompt's end.
void testContendedCutsApproachGFromTheFloor() {
  for (const auto &[profile, least] : std::initializer_list<std::pair<const test::TimingProfile *, uint32_t>>{
           {&test::kDense27B, 1}, {&test::kDense27B, 850}, {&test::kStoreLessSpillFlashNext, 1}}) {
    model::CommandLimits commandLimits = limits(kChunk, profile->layers);
    commandLimits.minimumUnitRows = least;
    Sim sim(*profile, commandLimits);
    const uint32_t floor = std::max(512u, least);
    const std::string of = std::string(profile->name) + " of least rows " + std::to_string(least) + ": ";
    sim.decoder(1);
    sim.lane(2, 0, 40'000);
    uint32_t units = 0;
    while (sim.scheduler.phase(2) == Phase::Prefill) {
      const BatchPlan plan = sim.plan();
      if (plan.kind != WorkKind::Prefill) {
        sim.run(plan);
        continue;
      }
      const uint32_t rows = plan.items[0].tokenCount, left = 40'000 - sim.scheduler.promptProcessed(2);
      require(plan.passes == 1, of + "a contended unit ran in passes");
      require(units || rows == floor, of + "the first contended unit ran " + std::to_string(rows) + " rows");
      require(rows >= floor || rows == left, of + "a contended unit ran " + std::to_string(rows) + " rows");
      const double wall = sim.run(plan);
      ++units;
      require(units <= 2 || rows == left ||
                  ((rows <= floor || wall <= 1.25 * kGap) && (rows >= kChunk || wall >= 0.5 * kGap)),
              of + "unit " + std::to_string(units) + " of " + std::to_string(rows) + " rows took " +
                  std::to_string(wall) + " ms");
    }
  }
}

// A cut sizes a deep lane by the pairs its layers reach: on Flint a lane at 64K
// runs its first contended unit within 2G after units measured near the start. A
// rest below the floor runs with its unit.
void testACutFollowsItsLanesDepth() {
  {
    Sim sim(test::kFlint, limits(kChunk, test::kFlint.layers));
    sim.decoder(1);
    sim.lane(2, 0, 40'000);
    while (sim.scheduler.promptProcessed(2) < 8192)
      sim.step();
    sim.scheduler.cancel(2);
    sim.scheduler.remove(2);
    sim.lane(3, 65536, 20'000);
    BatchPlan plan = sim.plan();
    for (uint32_t decodes = 0; decodes < 100 && plan.kind == WorkKind::Decode; ++decodes) {
      sim.run(plan);
      plan = sim.plan();
    }
    require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 3, "the deep lane did not lead a unit");
    const double wall = sim.run(plan);
    require(wall <= 2.0 * kGap, "the first unit at 64K ran " + std::to_string(plan.items[0].tokenCount) +
                                    " rows in " + std::to_string(wall) + " ms");
  }
  Sim sim(test::kDense27B, limits(kChunk, test::kDense27B.layers));
  sim.decoder(1);
  sim.lane(2, 0, 612);
  BatchPlan plan = sim.plan();
  if (plan.kind == WorkKind::Decode) {
    sim.run(plan);
    plan = sim.plan();
  }
  require(plan.kind == WorkKind::Prefill && plan.items[0].tokenCount == 612,
          "a contended lane of 612 rows ran " + std::to_string(plan.items[0].tokenCount) + " rows of its unit");
}

// A group sizes its passes by its rows and pairs over the command it was
// sized from: after a 300-row turn or an 8192-row group near the start, an
// 8192-row group at 128K on the 35B at 18G runs every pass within G (at the
// first command's time a layer, unscaled, a pass would hold several of its
// attention layers).
void testADeeperGroupRunsMorePasses() {
  for (const uint32_t rows : {300u, 8192u}) {
    Sim sim(test::k35BStudio18G, limits(8192, test::k35BStudio18G.layers));
    sim.lane(1, 0, rows);
    for (sim.run(sim.plan()); sim.passes;)
      sim.run(sim.passes->plan);
    sim.scheduler.cancel(1);
    sim.scheduler.remove(1);
    sim.lane(2, 131072, 8192);
    const BatchPlan group = sim.plan();
    double longest = sim.run(group);
    while (sim.passes)
      longest = std::max(longest, sim.run(sim.passes->plan));
    require(group.items[0].tokenCount == 8192 && longest <= kGap,
            "a deep group after " + std::to_string(rows) + " rows ran " +
                std::to_string(group.items[0].tokenCount) + " rows from " + std::to_string(group.passes) +
                " passes, the longest " + std::to_string(longest) + " ms");
  }
}

// A group runs in about its time over G passes: the first command after
// start, measured by nothing, and a 2039-row turn after a 292-row one, 7x
// the rows and on Flash-Next's dense attention about 49x a layer's pairs,
// both of which begin a layer a pass. Once a pass of a command ran an
// attention layer, its other passes are sized from its own.
void testAGroupRunsItsTimeOverGPasses() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, test::kM6FlashNextQ2.layers));
  const auto runTurn = [&sim](uint64_t id, uint32_t rows) {
    sim.lane(id, 0, rows);
    const BatchPlan first = sim.plan();
    require(first.items[0].tokenCount == rows, "a turn did not run in one command");
    uint32_t passes = 1;
    double milliseconds = sim.run(first);
    for (; sim.passes; ++passes)
      milliseconds += sim.run(sim.passes->plan);
    sim.scheduler.cancel(id);
    sim.scheduler.remove(id);
    // Its time over G, and the first pass, sized before its own time.
    const double expected = std::ceil(milliseconds / kGap) + 1.0;
    require(passes <= 2.0 * expected, std::to_string(rows) + " rows ran in " + std::to_string(passes) + " passes of " +
                                          std::to_string(milliseconds) + " ms in all");
  };
  runTurn(1, 292);
  runTurn(2, 2039);
}

// No lane joins a contended cut unit past its rows: before any command was
// measured a cut unit holds the floor's 512 rows, which two short turns
// share; a longer one waits.
void testJoinersStayWithinACutUnit() {
  Sim sim(test::kDense27B);
  sim.decoder(1);
  sim.lane(2, 0, 300);
  sim.lane(3, 0, 200);
  sim.lane(4, 0, 400);
  BatchPlan plan = sim.plan();
  if (plan.kind == WorkKind::Decode) {
    sim.run(plan);
    plan = sim.plan();
  }
  require(plan.kind == WorkKind::Prefill && plan.width() == 2 && rowsOf(plan) == 500 &&
              std::ranges::count(plan.items, 4u, &BatchItem::requestId) == 0,
          "a cut unit of the floor's 512 rows held " + std::to_string(rowsOf(plan)) + " rows of " +
              std::to_string(plan.width()) + " lanes");
}

// A lower priority's decode does not bound a prefill; one of its priority
// does.
void testALowerPrioritysDecodeDoesNotBoundAPrefill() {
  DenseSim sim;
  sim.lane(1, 0, 20'000, RequestPriority::Foreground);
  sim.decoder(2, RequestPriority::Background);
  require(sim.plan().items[0].tokenCount == kChunk, "lower-priority work bounded a Foreground prefill");
  sim.decoder(3, RequestPriority::Foreground);
  completeDecode(sim);
  require(sim.plan().items[0].tokenCount < kChunk, "a Foreground decoder left its priority's prefill unbounded");
}

// Neither lanes waiting for admission or memory nor lanes that only prefill
// bound a prefill: nothing of theirs waits for its units.
void testWaitingOrPrefillingPeersDoNotBoundAPrefill() {
  DenseSim sim;
  sim.lane(1, 0, 20'000);
  sim.scheduler.submit(request(2, 20'000));
  sim.scheduler.waitForResources(2);
  sim.scheduler.submit(request(3, 20'000));
  require(sim.plan().items[0].tokenCount == kChunk, "queued or resource-blocked work bounded a resident prefill");
  for (uint64_t id = 4; id <= 6; ++id)
    sim.lane(id, 0, 20'000);
  const BatchPlan plan = sim.plan();
  require(plan.width() == 1 && plan.items[0].requestId == 1 && plan.items[0].tokenCount == kChunk &&
              plan.passes == 1,
          "lanes that only prefill bounded their leader");
}

// contended_prefill_ms counts the units planning bounded, those committed
// while a lane of their priority decoded: none of a lone prompt's, its last
// among them, which leaves its lane decoding, and one beside that decoder (a
// prompt of 100 rows, one unit within the bound).
void testContendedPrefillCountsTheUnitsPlanningBounds() {
  DenseSim sim;
  sim.lane(1, 0, 3000);
  while (sim.scheduler.phase(1) == Phase::Prefill)
    sim.step();
  require(sim.scheduler.snapshot().contendedPrefill.count == 0,
          "a lone prompt's units counted as contended, its last ending beside its own decode");
  sim.lane(2, 0, 100);
  while (sim.scheduler.phase(2) == Phase::Prefill)
    sim.step();
  require(sim.scheduler.snapshot().contendedPrefill.count == 1,
          "a unit beside a decoder of its priority was not counted once");
}

// contended_prefill_ms' quantiles are the units' own times to within a
// millisecond, never above the longest: 17 units of 510 ms and 3 of 600 ms sit
// either side of the log-2 microsecond edge at 524.288 ms, where reading a
// bucket's upper edge put p95 at 1048.576 ms.
void testContendedPrefillQuantilesAreTheUnitsTimes() {
  DenseSim sim;
  sim.decoder(1);
  sim.lane(2, 0, 200'000);
  for (uint32_t unit = 0; unit < 20; ++unit) {
    while (sim.plan().kind == WorkKind::Decode)
      completeDecode(sim);
    completePrefill(sim, sim.plan(), {.wallMilliseconds = unit < 17 ? 510.0 : 600.0});
  }
  const model::LatencyStats units = sim.scheduler.snapshot().contendedPrefill;
  require(units.count == 20 && units.maximumMilliseconds == 600.0 && units.p50Milliseconds >= 510.0 &&
              units.p50Milliseconds <= 511.0 && units.p95Milliseconds == 600.0 && units.p99Milliseconds == 600.0,
          "contended units of 510 and 600 ms read p50 " + std::to_string(units.p50Milliseconds) + ", p95 " +
              std::to_string(units.p95Milliseconds) + ", p99 " + std::to_string(units.p99Milliseconds) +
              ", max " + std::to_string(units.maximumMilliseconds) + " over " + std::to_string(units.count));
}

// Units past contended_prefill_ms' millisecond buckets (4.094 s) read as the
// longest.
void testContendedPrefillPastItsBucketsReadsTheLongest() {
  DenseSim sim;
  sim.decoder(1);
  sim.lane(2, 0, 200'000);
  for (const double wall : {1.0, 4500.0, 6000.0}) {
    while (sim.plan().kind == WorkKind::Decode)
      completeDecode(sim);
    completePrefill(sim, sim.plan(), {.wallMilliseconds = wall});
  }
  const model::LatencyStats units = sim.scheduler.snapshot().contendedPrefill;
  require(units.count == 3 && units.p50Milliseconds == 6000.0 && units.p95Milliseconds == 6000.0 &&
              units.maximumMilliseconds == 6000.0,
          "contended units of 1, 4500 and 6000 ms read p50 " + std::to_string(units.p50Milliseconds) + ", p95 " +
              std::to_string(units.p95Milliseconds) + ", max " + std::to_string(units.maximumMilliseconds));
}

// A ready peer of its priority, with no decoder, does not bound a lane that
// streams its history: the Spill lane leads a group's rows beside a long
// Resident prompt as it does alone.
void testAReadyPeerDoesNotBoundAStreamingLeader() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.lane(1, 45056, 40'000);
  sim.shape(1, 45056 / kPageTokens, true);
  const BatchPlan alone = sim.plan();
  sim.lane(2, 0, 200'000);
  const BatchPlan beside = sim.plan();
  require(beside.items[0].requestId == 1 && beside.items[0].tokenCount == alone.items[0].tokenCount &&
              beside.items[0].tokenCount == 8192,
          "a streaming leader beside a ready peer ran " + std::to_string(beside.items[0].tokenCount) +
              " rows, alone " + std::to_string(alone.items[0].tokenCount));
}

// A command sizes the next group from its rows' time alone: one whose time
// is not positive or not finite, whose completion added work, or that was
// abandoned between its passes sizes nothing, and an image encode ahead of
// its first pass adds nothing to it.
void testUntimedUnitsSizeNothing() {
  using Sampled = std::function<void(model::CommandSample &, const BatchPlan &)>;
  // The passes of the group after a 300-row turn whose samples `sampled`
  // changes, or that a Foreground turn abandons after its first pass.
  const auto nextPasses = [](const Sampled &sampled, bool abandoned = false) {
    Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
    sim.sampled = sampled;
    sim.lane(1, 8192, 300, RequestPriority::Background);
    const BatchPlan plan = sim.plan();
    require(plan.passes == 48, "a group before any command was measured was not planned in a pass a layer");
    sim.run(plan);
    if (!abandoned) {
      while (sim.passes)
        sim.run(sim.passes->plan);
      sim.scheduler.cancel(1);
      sim.scheduler.remove(1);
    }
    sim.sampled = nullptr;
    sim.lane(2, 8192, 300, abandoned ? RequestPriority::Foreground : RequestPriority::Background);
    return sim.plan().passes;
  };
  const uint32_t timed = nextPasses({});
  require(timed < 48, "a timed command left the next group every pass");
  for (const double wall : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()})
    require(nextPasses([wall](model::CommandSample &sample, const BatchPlan &) { sample.wallMilliseconds = wall; }) ==
                48,
            "a command of " + std::to_string(wall) + " ms sized the next group");
  require(nextPasses([](model::CommandSample &sample, const BatchPlan &) { sample.representative = false; }) == 48,
          "a command whose completion added work sized the next group");
  require(nextPasses({}, true) == 48, "a command abandoned after its first pass sized the next group");
  const uint32_t encoded = nextPasses([](model::CommandSample &sample, const BatchPlan &plan) {
    if (!plan.pass) {
      sample.wallMilliseconds += 5000.0;
      sample.imageEncodeMilliseconds = 5000.0;
    }
  });
  require(encoded == timed, "an image encode ahead of a command left the next group " + std::to_string(encoded) +
                                " passes, without it " + std::to_string(timed));
}

// One lane of a command may run past a chunk (model::CommandLimits::
// lanesPastChunk), and a command holds at most the model's command rows,
// whatever its lanes' depths and lengths.
void testACommandHoldsOneLanePastAChunk() {
  for (const uint32_t depth : {0u, 70'000u, 150'000u, 300'000u})
    for (const uint32_t remaining : {300u, 3000u, 9000u, 20'000u})
      for (const uint32_t peer : {0u, 100'000u})
        for (const uint32_t shortest : {1000u, remaining + 1000}) {
          Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
          sim.lane(1, depth, remaining);
          sim.lane(2, peer, shortest);
          sim.lane(3, peer, remaining + 2000);
          sim.lane(4, 0, remaining + 3000);
          const BatchPlan plan = sim.plan();
          uint32_t past = 0, rows = 0;
          for (const BatchItem &item : plan.items) {
            past += rows / kChunk != (rows + item.tokenCount - 1) / kChunk;
            rows += item.tokenCount;
          }
          require(past <= 1 && rows <= 8192, "a command held " + std::to_string(past) +
                                                 " lanes past a chunk or " + std::to_string(rows) + " rows");
        }
}

// --- Groups, the bound, passes -----------------------------------------------------------------------------

// A dense prompt alone runs whole: nothing waits for its units.
void testALoneDenseCommandRunsWhole() {
  Sim sim(test::kDense27B, limits(kChunk, test::kDense27B.layers));
  sim.lane(1, 0, 4096);
  const BatchPlan plan = sim.plan();
  require(plan.items[0].tokenCount == kChunk && plan.passes == 1, "a lone dense command did not run whole");
}

// A lone streamed request: a long prompt takes a group of the model's command
// rows, and a turn runs in passes each within G once a command was measured.
void testALoneStreamedRequestRunsPassesWithinG() {
  for (const test::TimingProfile &profile : atEveryTurnaround({&test::kM6FlashNextQ2})) {
    Sim sim(profile, limits(8192, 48));
    sim.lane(1, 0, 20'000);
    require(sim.plan().items[0].tokenCount == 8192, at(profile) + ": a lone long prompt did not take a group");
    for (sim.run(sim.plan()); sim.passes;)
      sim.run(sim.passes->plan);
    sim.scheduler.cancel(1);
    sim.scheduler.remove(1);
    sim.lane(2, 8192, 300);
    const BatchPlan turn = sim.plan();
    double longest = 0.0;
    for (longest = sim.run(turn); sim.passes;)
      longest = std::max(longest, sim.run(sim.passes->plan));
    require(turn.items[0].tokenCount == 300 && turn.passes > 1 && longest <= kGap,
            at(profile) + ": a lone turn ran in " + std::to_string(turn.passes) + " passes, the longest " +
                std::to_string(longest) + " ms");
    // A turn of the same shape, sized from that one, splits no finer than G
    // needs: each of its passes takes at least G/2.
    sim.scheduler.cancel(2);
    sim.scheduler.remove(2);
    sim.lane(3, 8192, 300);
    const BatchPlan again = sim.plan();
    double shortest = 0.0;
    for (shortest = sim.run(again); sim.passes;)
      shortest = std::min(shortest, sim.run(sim.passes->plan));
    require(shortest >= kGap / 2, at(profile) + ": the next turn ran in " + std::to_string(again.passes) +
                                      " passes, the shortest " + std::to_string(shortest) + " ms");
  }
}

// The pair bound is a row rule: one attention layer's pairs stay within the
// cap only by fewer rows (8192 rows fit only to 61,183, so a lane at 70,000
// halves to 4096); a block-sparse layer's rows keep the command's.
void testThePairBoundIsPerLayerAndOnlyRowsMeetIt() {
  {
    Sim sim(test::kDense27B, limits(kChunk, test::kDense27B.layers, 1,
                                    model::denseAttentionPairs(131072 - kChunk, kChunk)));
    sim.lane(1, 131072 + 8192, 20'000);
    require(sim.plan().items[0].tokenCount == kChunk / 2, "a dense lane past its window did not halve once");
  }
  {
    test::TimingProfile full = test::kM6FlashNextQ2;
    full.reach = {};
    Sim sim(full, limits(8192, 19, 1, model::denseAttentionPairs(61183, 8192)));
    sim.lane(1, 70000, 20'000);
    require(sim.plan().items[0].tokenCount == 4096, "passes kept rows the per-layer pair bound refuses");
  }
  for (const uint32_t position : {0u, 131072u, 1u << 20}) {
    Sim sim(test::kM6FlashNextQ2, limits(8192, 48, 1, model::denseAttentionPairs(262144 - kChunk, kChunk)));
    sim.lane(1, position, 20'000);
    require(sim.plan().items[0].tokenCount == 8192, "a block-sparse lane lost rows to the pair bound");
  }
}

// Every contended unit past the first commands takes at most G and its
// work's growth since the command it was sized from.
void testEveryContendedUnitWithinItsBound() {
  for (const test::TimingProfile &tuned :
       atEveryTurnaround({&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2})) {
    const test::TimingProfile *profile = &tuned;
    Sim sim(*profile, limits(8192, profile->layers));
    sim.decoder(1);
    sim.lane(2, 0, 120'000);
    uint32_t units = 0;
    double longest = 0.0;
    while (sim.scheduler.phase(2) == Phase::Prefill && units < 400) {
      const BatchPlan plan = sim.plan();
      const double wall = sim.run(plan);
      // A unit of one layer can be no shorter.
      if (plan.kind == WorkKind::Prefill && ++units > 40 && plan.passes < profile->layers)
        longest = std::max(longest, wall / kGap);
    }
    require(longest <= 1.0, at(*profile) + ": a contended unit took " + std::to_string(longest) + " G");
  }
}

// A contended Spill lane leads commands of its group's rows, bounded by
// passes, never cut.
void testAContendedSpillLaneKeepsItsGroupRows() {
  for (const test::TimingProfile *profile : {&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2}) {
    Sim sim(*profile, limits(8192, 48));
    sim.decoder(1);
    sim.decoder(2);
    sim.lane(3, 45056, 40'000);
    sim.shape(3, 45056 / kPageTokens, true);
    for (uint32_t command = 0; command < 30 && sim.scheduler.phase(3) == Phase::Prefill; ++command) {
      const BatchPlan plan = sim.plan();
      if (plan.kind == WorkKind::Prefill && !plan.pass) {
        const uint32_t remaining = 85056 - plan.items[0].promptOffset;
        require(plan.items[0].tokenCount == std::min(remaining, 8192u),
                std::string(profile->name) + ": a contended Spill lane lost its group's rows");
      }
      sim.run(plan);
    }
  }
}

// Beside a Spill lane streaming its history, a decoder's gaps between its
// decode commands stay within a unit's bound and one decode command, at
// every turnaround.
void testDecodeGapsBesideAStreamingLaneStayWithinTheBound() {
  for (const test::TimingProfile *base : {&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2}) {
    for (const double turnaround : kTurnarounds) {
      const test::TimingProfile profile = withTurnaround(*base, turnaround);
      Sim sim(profile, limits(8192, profile.layers));
      sim.decoder(1);
      sim.lane(2, 45056, 80'000);
      sim.shape(2, 45056 / kPageTokens, true);
      std::optional<double> lastDecode;
      uint32_t units = 0;
      double worst = 0.0, oneLayer = 0.0;
      while (sim.scheduler.phase(2) == Phase::Prefill && units < 800) {
        const BatchPlan plan = sim.plan();
        const double wall = sim.run(plan);
        if (plan.kind == WorkKind::Prefill) {
          // A unit of one layer can be no shorter.
          if (plan.passes == profile.layers)
            oneLayer = std::max(oneLayer, wall / kGap);
          ++units;
          continue;
        }
        if (lastDecode && units > 40)
          worst = std::max(worst, (sim.now - *lastDecode - wall) / kGap);
        lastDecode = sim.now;
      }
      require(units > 40 && worst <= std::max(1.0, oneLayer),
              at(profile) + ": a decode gap beside a streaming lane reached " + std::to_string(worst) + " G");
    }
  }
}

// The TTFT simulation on the M6's profile: agent turns of 300 rows at 8K,
// arriving at `arrivals` after three rounds of as many turns together have
// sized their passes; each turn's first token ends its prompt (no admission
// here), after which it leaves, as what it waited for is what this measures.
// Returns each turn's TTFT, and in `first` the first plan.
std::vector<double> runTurns(std::span<const double> arrivals, std::optional<BatchPlan> *first = nullptr) {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  for (uint64_t round = 0, id = 100; round < 3; ++round) {
    const uint64_t from = id;
    for (; id < from + arrivals.size(); ++id)
      sim.lane(id, 8192, 300);
    while (sim.scheduler.phase(from) == Phase::Prefill)
      sim.step();
    for (uint64_t lane = from; lane < id; ++lane) {
      sim.scheduler.cancel(lane);
      sim.scheduler.remove(lane);
    }
  }
  sim.now = 0.0;
  std::vector<double> ttft(arrivals.size(), -1.0);
  size_t arrived = 0, done = 0;
  while (done < arrivals.size()) {
    for (; arrived < arrivals.size() && arrivals[arrived] <= sim.now; ++arrived)
      sim.lane(arrived + 1, 8192, 300);
    const std::optional<BatchPlan> plan = sim.next();
    if (!plan) {
      sim.now = arrivals[arrived];
      continue;
    }
    if (first && !*first)
      *first = plan;
    sim.run(*plan);
    for (size_t index = 0; index < arrived; ++index)
      if (ttft[index] < 0.0 && sim.scheduler.phase(index + 1) != Phase::Prefill) {
        ttft[index] = sim.now - arrivals[index];
        sim.scheduler.cancel(index + 1);
        sim.scheduler.remove(index + 1);
        ++done;
      }
  }
  return ttft;
}

// Turns together (N = 1, 2, 4) share one command, the worst TTFT within 1.25
// times that command; turns apart wait at most for the command in flight and
// one command of all of them.
void testTurnsTtftAloneTogetherAndApart() {
  const test::TimingProfile &profile = test::kM6FlashNextQ2;
  test::TimedWork work(profile);
  const auto whole = [&](uint32_t turns) {
    std::vector<model::PrefillLane> lanes(turns, model::PrefillLane{8192, 300, 0});
    return work.times(lanes, 0, 1).wallMilliseconds;
  };
  const std::array lone{0.0};
  const double own = runTurns(lone)[0];
  for (const uint32_t count : {1u, 2u, 4u}) {
    const std::vector<double> together(count, 0.0);
    std::optional<BatchPlan> first;
    const std::vector<double> ttft = runTurns(together, &first);
    require(first && first->width() == count, std::to_string(count) + " turns together did not share a command");
    require(*std::ranges::max_element(ttft) <= 1.25 * whole(count),
            std::to_string(count) + " turns together: worst TTFT " +
                std::to_string(*std::ranges::max_element(ttft)) + " ms against their command's " +
                std::to_string(whole(count)) + " ms");
  }
  const std::array split{0.0, 1030.0};
  const std::vector<double> ttft = runTurns(split);
  require(ttft[1] <= (own - split[1]) + own + 1e-6 && ttft[0] + ttft[1] <= own + (own - split[1]) + own + 1e-6,
          "turns 1.03 s apart took " + std::to_string(ttft[0]) + " and " + std::to_string(ttft[1]) +
              " ms, past the rest of the first's command and their own, " + std::to_string(own) + " ms each");
  for (const uint32_t count : {2u, 4u})
    for (const double gap : {500.0, 1000.0, 1500.0}) {
      std::vector<double> arrivals;
      for (uint32_t index = 0; index < count; ++index)
        arrivals.push_back(index * gap);
      const std::vector<double> apart = runTurns(arrivals);
      for (size_t index = 0; index < apart.size(); ++index)
        require(apart[index] <= 1.25 * (own + whole(count)),
                std::to_string(count) + " turns " + std::to_string(int(gap)) + " ms apart: turn " +
                    std::to_string(index + 1) + " took " + std::to_string(apart[index]) + " ms");
    }
}

// --- Joins --------------------------------------------------------------------------------------------------

// A long lane's sparing rows are those past its whole commands of the
// model's command rows, however evenly its own commands would split them:
// 8192 + 300 rows alone run as 6144 and 2348, 16384 + 116 as 6144, 6144 and
// 4212, and 300 or 116 rows spare it a command beside a turn.
void testALongLaneSparesACommandWithTheRowsPastItsWholeCommands() {
  for (const uint32_t sparing : {300u, 116u}) {
    const uint32_t remaining = (sparing == 300 ? 8192 : 16384) + sparing;
    Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
    sim.lane(1, 8192, 300);
    sim.lane(2, 0, remaining);
    const BatchPlan plan = sim.plan();
    require(plan.width() == 2 && plan.items[0].requestId == 1 && plan.items[1].tokenCount == sparing,
            "a lane of " + std::to_string(remaining) + " rows did not join a turn with its " +
                std::to_string(sparing) + " sparing rows: " + std::to_string(plan.width()) + " lanes, " +
                std::to_string(plan.items.back().tokenCount) + " rows last");
  }
}

// A lane that streams its history shares a command: a Spill lane's last
// rows join a short turn's command, which runs in passes.
void testAStreamingLaneSharesACommand() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.lane(1, 8192, 300);
  sim.lane(2, 45056, 200);
  sim.shape(2, 45056 / kPageTokens);
  const BatchPlan plan = sim.plan();
  require(plan.items == std::vector<BatchItem>{{2, 200, 45056}, {1, 300, 8192}},
          "a lane with Index pages did not share a turn's command");
  require(plan.passes > 1, "the shared command did not run in passes");
}

// One lane whose history holds or may gain Index pages per command: a lane
// that may gain them does not join a command in passes beside one that holds
// them, and is counted; in one pass it may.
void testTwoSpillLanesNeverShareAMultiPassCommand() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.decoder(9);
  sim.lane(1, 45056, 300);
  sim.shape(1, 45056 / kPageTokens);
  sim.lane(2, 45056, 200);
  sim.shape(2, 0, true);
  BatchPlan plan = sim.plan();
  while (plan.kind != WorkKind::Prefill) {
    sim.run(plan);
    plan = sim.plan();
  }
  require(plan.passes > 1 && plan.width() == 1, "two lanes with Index pages shared a command in passes");
  sim.run(plan);
  require(sim.scheduler.snapshot().indexPageConflicts == 1, "the lane left out for Index pages was not counted");
}

// Commit counts what next() planned, not a plan made again without the lanes
// prepare refused: the refused lane 3 would lead and leave no conflict.
void testTheCountersAreThoseOfThePlanCommitted() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.decoder(9);
  sim.lane(1, 45056, 300);
  sim.shape(1, 45056 / kPageTokens);
  sim.lane(2, 45056, 200);
  sim.shape(2, 0, true);
  sim.lane(3, 8192, 100);
  completeDecode(sim);
  sim.excluded = {3};
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId != 3 && plan.width() == 1,
          "the plan without the refused lane did not leave a lane out for Index pages");
  sim.excluded.clear();
  sim.run(plan);
  require(sim.scheduler.snapshot().indexPageConflicts == 1,
          "the committed plan's Index-page conflict was not counted");
}

// --- Order and aging ----------------------------------------------------------------------------------------

// The fewest remaining rows lead: 11K, 13K, 35K whatever their order.
void testFewestRemainingRowsLead() {
  DenseSim sim;
  sim.lane(1, 0, 35'000);
  sim.lane(2, 0, 13'000);
  sim.lane(3, 0, 11'000);
  require(sim.plan().items[0].requestId == 3, "the fewest remaining rows did not lead");
}

// Of two near-equal long Spill prompts, the older, started alone, prefills to
// its end, its last command's passes included, before the younger is
// admitted: the younger is never overtaken. The shorter starts when both
// wait, the younger here; the older then waits K of its commands, starts
// overdue, and from then on no more than K commands in a row pass over it.
void testNearEqualLongPromptsAgeByCommands() {
  Sim alone(test::kM6FlashNextQ2, limits(8192, 48));
  alone.lane(1, 0, 71'254);
  alone.shapes.push_back({.requestId = 1, .spill = true});
  alone.step();
  require(alone.passes.has_value(), "the older prompt's first command did not run in passes");
  alone.scheduler.submit(request(2, 71'220));
  const std::array younger{PrefillAdmission{2, 0}};
  while (alone.scheduler.phase(1) == Phase::Prefill) {
    require(alone.scheduler.prefillAdmissionOrder(younger, alone.in()).empty(),
            "the younger prompt was admitted while the older prefilled, at " +
                std::to_string(alone.scheduler.promptProcessed(1)) + " of its rows" +
                (alone.passes ? " and a pass of its command" : ""));
    alone.step();
  }
  require(alone.scheduler.prefillAdmissionOrder(younger, alone.in()) == std::vector<uint64_t>{2},
          "the younger prompt did not start once the older's prompt ended");
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.scheduler.submit(request(1, 71'254));
  sim.scheduler.submit(request(2, 71'220));
  const std::array both{PrefillAdmission{1, 0}, PrefillAdmission{2, 0}};
  require(sim.scheduler.prefillAdmissionOrder(both, sim.in()) == std::vector<uint64_t>{2},
          "admission did not start the shorter of two near-equal prompts");
  sim.scheduler.resourcesReady(2, 0);
  sim.shapes.push_back({.requestId = 2, .spill = true});
  const std::array older{PrefillAdmission{1, 0}};
  uint32_t commands = 0;
  while (sim.scheduler.prefillAdmissionOrder(older, sim.in()).empty()) {
    require(commands < kOvertakes, "the older prompt was not admitted after K commands of the younger");
    const BatchPlan plan = sim.plan();
    commands += !plan.pass;
    sim.run(plan);
  }
  require(commands == kOvertakes, "the older prompt was admitted after " + std::to_string(commands) +
                                      " commands of the younger, not K");
  sim.scheduler.resourcesReady(1, 0);
  sim.shapes.push_back({.requestId = 1, .spill = true});
  while (sim.passes)
    sim.step();
  require(sim.plan().items[0].requestId == 1, "the overdue older prompt did not lead the next command");
  uint32_t passedOver = 0, led = 0;
  while (sim.scheduler.phase(1) == Phase::Prefill) {
    const BatchPlan plan = sim.plan();
    if (plan.kind == WorkKind::Prefill && !plan.pass) {
      const bool served = std::ranges::count(plan.items, 1u, &BatchItem::requestId) != 0;
      passedOver = served ? 0 : passedOver + 1;
      led += plan.items[0].requestId == 1;
      require(passedOver <= kOvertakes, "more than K commands in a row passed over the older prompt");
    }
    sim.run(plan);
  }
  require(led > 1 && sim.scheduler.phase(2) != Phase::Prefill,
          "the two prompts did not share the commands until the younger's prompt ended");
}

// A queued prompt starts only when the next command serves it and leaves it
// waiting for no other prompt: the command ends its prompt, it leads, it is
// overdue, or every lane before it ends its prompt there and none is a Spill
// lane (two Spill lanes prefill together only as the later one leads). So the
// 57,664 prompt stays queued though its 320 rows past its whole commands fit
// beside the Spill lane's last command.
void testALongPromptStartsOnlyToLead() {
  const auto spill = [](Sim &sim, uint64_t id) { sim.shapes.push_back({.requestId = id, .spill = true}); };
  const auto round = [&](Sim &sim, uint32_t second, uint32_t third) {
    sim.lane(3, 0, 59'043);
    while (sim.scheduler.phase(3) == Phase::Prefill)
      sim.step();
    sim.lane(1, 0, second);
    spill(sim, 1);
    sim.scheduler.submit(request(2, third));
    const std::array later{PrefillAdmission{2, 0}};
    while (sim.scheduler.phase(1) == Phase::Prefill) {
      require(sim.scheduler.prefillAdmissionOrder(later, sim.in()).empty(),
              "the third prompt, " + std::to_string(third) + " tokens, started to join the second's command at " +
                  std::to_string(sim.scheduler.promptProcessed(1)) + " of its rows");
      sim.step();
    }
    require(sim.scheduler.prefillAdmissionOrder(later, sim.in()) == std::vector<uint64_t>{2},
            "the third prompt, " + std::to_string(third) + " tokens, did not start once it led");
  };
  Sim fits(test::kStudio18GFlashNextQ2, limits(8192, 48));
  round(fits, 57'644, 57'664);
  Sim sim(test::kStudio18GFlashNextQ2, limits(8192, 48));
  round(sim, 59'020, 59'054);
  sim.scheduler.cancel(2);
  sim.scheduler.remove(2);
  sim.lane(4, 0, 6144);
  sim.scheduler.submit(request(5, 3 * 8192 + 1710));
  const std::array next{PrefillAdmission{5, 0}};
  require(sim.scheduler.prefillAdmissionOrder(next, sim.in()) == std::vector<uint64_t>{5},
          "a prompt that leads once the command ends the Resident one ahead of it did not start beside it");
  spill(sim, 4);
  require(sim.scheduler.prefillAdmissionOrder(next, sim.in()).empty(),
          "a prompt that leads once the command ends the Spill one ahead of it started beside it");
  Sim chunk(test::kStudio18GFlashNextQ2, limits(8192, 48));
  chunk.lane(1, 0, 1000);
  chunk.scheduler.submit(request(2, 1048));
  const std::array whole{PrefillAdmission{2, 0}};
  require(chunk.scheduler.prefillAdmissionOrder(whole, chunk.in()) == std::vector<uint64_t>{2},
          "a prompt the next command ends did not start beside its leader");
}

// A start restoring its prefix from the SSD cache runs in no command before
// its read lands, and holds back only a prompt that it would then lead: beside
// 3000 rows to go, a prompt of 3100, left 1052 by its first command, starts;
// one of 5200, left 3152, stays queued.
void testARestoringStartHoldsBackWhatItWouldLead() {
  // Whether a cold prompt of `prompt` rows starts beside a start of
  // `priority` restoring to `cached` with `restoring` rows to go.
  const auto starts = [](Sim &sim, uint32_t cached, uint32_t restoring, uint32_t prompt,
                         RequestPriority priority = RequestPriority::Normal) {
    sim.scheduler.submit(request(1, cached + restoring, priority));
    sim.restoring.push_back({1, cached});
    sim.scheduler.submit(request(2, prompt));
    const std::array candidate{PrefillAdmission{2, 0}};
    return sim.scheduler.prefillAdmissionOrder(candidate, sim.in()) == std::vector<uint64_t>{2};
  };
  for (const auto &[restoring, prompt, priority, expected] :
       {std::tuple{640u, 1500u, RequestPriority::Normal, true},
        {3000u, 3100u, RequestPriority::Normal, true},
        {3000u, 5200u, RequestPriority::Normal, false},
        {162u, 226u, RequestPriority::Foreground, true},
        {162u, 3100u, RequestPriority::Foreground, false}}) {
    DenseSim sim;
    require(starts(sim, 64, restoring, prompt, priority) == expected,
            "a prompt of " + std::to_string(prompt) + " rows " + (expected ? "did not start" : "started") +
                " beside a start of priority " + std::to_string(static_cast<int>(priority)) + " restoring with " +
                std::to_string(restoring) + " rows to go");
  }
  Sim pair(test::kStudio18GFlashNextQ2, limits(8192, 48));
  pair.shapes.push_back({.requestId = 1});
  require(!starts(pair, 56'644, 1000, 57'664), "the second of the near-equal pair started beside the first's restore");
}

// A long lane under a stream of short arrivals of its priority waits behind
// at most K = 3 commands that serve them, then leads one: its prompt ends
// within (1 + K) times its own time plus a unit.
void testAgingBoundsALongLaneUnderArrivals() {
  DenseSim sim;
  sim.lane(1, 0, 6 * kChunk);
  double cost = 0.0;
  for (uint32_t command = 0; command < 6; ++command) {
    const std::array item{BatchItem{1, kChunk, command * kChunk}};
    cost += sim.work.times(sim.lanes(item), 0, 1).wallMilliseconds;
  }
  // A shorter prompt arrives as each one before it ends.
  uint64_t arrival = 0;
  double longest = 0.0;
  while (sim.scheduler.phase(1) == Phase::Prefill && arrival < 1000) {
    if (!arrival || sim.scheduler.phase(arrival) != Phase::Prefill) {
      if (arrival) {
        sim.scheduler.cancel(arrival);
        sim.scheduler.remove(arrival);
      }
      sim.lane(arrival = arrival ? arrival + 1 : 100, 0, 1500);
    }
    longest = std::max(longest, sim.step());
  }
  require(sim.scheduler.phase(1) != Phase::Prefill && sim.now <= 4.0 * cost + longest,
          "the long lane's prompt ended after " + std::to_string(sim.now) + " ms, past (1 + K) of its " +
              std::to_string(cost) + " ms");
}

// A prompt waiting for a lane is overtaken as a lane is: after K commands
// that end later arrivals' short prompts, a few hundred rows each, the 30K
// prompt, whose rows past its whole chunks do not fit beside them, is overdue
// and starts, ahead of the next short arrival, and leads the next command;
// served, it counts again, and the short arrival leads.
void testAWaitingPromptIsOverdueAfterKOvertakes() {
  DenseSim sim;
  sim.scheduler.submit(request(1, 14 * kChunk + 1900));
  const std::array waiting{PrefillAdmission{1, 0}};
  for (uint64_t id = 10; id < 10 + kOvertakes; ++id) {
    sim.lane(id, 0, 300);
    require(sim.scheduler.prefillAdmissionOrder(waiting, sim.in()).empty(),
            "the waiting prompt started after " + std::to_string(id - 10) + " commands of later arrivals");
    sim.step();
    require(sim.scheduler.phase(id) == Phase::Decode, "a short arrival did not end its prompt in one command");
    sim.scheduler.cancel(id);
    sim.scheduler.remove(id);
  }
  sim.scheduler.submit(request(20, 300));
  const std::array both{PrefillAdmission{1, 0}, PrefillAdmission{20, 0}};
  require(sim.scheduler.prefillAdmissionOrder(both, sim.in()).front() == 1,
          "the waiting prompt did not start ahead of a short arrival after K overtakes");
  sim.scheduler.resourcesReady(1, 0);
  sim.scheduler.resourcesReady(20, 0);
  const BatchPlan first = sim.plan();
  require(first.items == std::vector<BatchItem>{{1, kChunk, 0}}, "the overdue prompt did not lead the next command");
  sim.run(first);
  require(sim.plan().items[0].requestId == 20, "served, the prompt stayed overdue");
}

// Equal prompts are served oldest first, the oldest leading to its prompt's
// end.
void testEqualPromptsFinishInArrivalOrder() {
  DenseSim sim;
  for (uint64_t id = 1; id <= 3; ++id)
    sim.lane(id, 0, 5000);
  for (uint32_t command = 0; command < 3; ++command) {
    const BatchPlan plan = sim.plan();
    require(plan.items[0].requestId == 1 &&
                plan.items[0].tokenCount == std::min(kChunk, 5000 - command * kChunk),
            "equal prompts were not served oldest first");
    sim.run(plan);
  }
  require(sim.scheduler.phase(1) == Phase::Decode, "the oldest prompt did not end in three commands");
  sim.scheduler.cancel(1);
  require(sim.plan().items[0].requestId == 2, "the next oldest equal prompt did not lead once the oldest's ended");
}

// A later short prompt leads an earlier long one.
void testShortArrivalPrecedesLongColdPrompt() {
  DenseSim sim;
  sim.lane(1, 0, 3000);
  sim.lane(2, 0, 700);
  const BatchPlan plan = sim.plan();
  require(plan.items[0].requestId == 2 && plan.items[0].tokenCount == 700,
          "a later short prompt waited behind an earlier long one");
}

// A higher priority's commands do not age a lower one: once they end, a
// later shorter lane of its priority still leads.
void testHigherTierCommandsDoNotAgeALowerTier() {
  DenseSim sim;
  sim.lane(1, 0, 4096, RequestPriority::Background);
  for (uint64_t id = 10; id < 30; ++id) {
    sim.lane(id, 0, 1500, RequestPriority::Foreground);
    while (sim.scheduler.phase(id) == Phase::Prefill)
      sim.step();
    sim.scheduler.cancel(id);
    sim.scheduler.remove(id);
  }
  sim.lane(2, 0, 1000, RequestPriority::Background);
  require(sim.plan().items[0].requestId == 2, "a higher priority's commands aged a lower lane");
}

// --- A command between its passes ----------------------------------------------------------------------------

// A same-priority arrival neither cuts nor drops the rows a command between its
// passes started: it leads the next plan, with the long lane's remaining rows,
// once the command ends, early or late in its passes.
void testASameTierArrivalLeavesTheRowsACommandStarted() {
  for (const double elapsed : {1000.0, 3000.0}) {
    {
      Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
      sim.lane(1, 0, 9500);
      const BatchPlan command = sim.plan();
      require(command.width() == 1 && command.items[0].tokenCount == 6144 && command.passes > 2,
              "the long prompt did not lead a command of 6144 rows in passes");
      double ran = sim.run(command);
      while (sim.passes && ran < elapsed)
        ran += sim.run(sim.passes->plan);
      require(sim.passes.has_value(), "the command ended before W");
      const std::string after = "after W " + std::to_string(ran) + " ms: ";
      sim.lane(2, 8192, 300);
      const BatchPlan fresh = [&] {
        const PassProgress kept = *sim.passes;
        sim.passes.reset();
        const BatchPlan result = sim.plan();
        sim.passes = kept;
        return result;
      }();
      require(fresh.items == std::vector<BatchItem>{{2, 300, 8192}, {1, 1308, 0}},
              after + "the turn did not lead a plan with the long lane's sparing rows");
      require(!sim.scheduler.abandonsPasses(sim.plan(), sim.in()), after + "the turn superseded the command");
      while (sim.passes)
        sim.step();
      const BatchPlan next = sim.plan();
      require(next.kind == WorkKind::Prefill && std::ranges::count(next.items, 2u, &BatchItem::requestId) == 1 &&
                  sim.scheduler.snapshot().abandonedPassCommands == 0,
              after + "the turn did not run in the command after the long lane's");
    }
  }
}

// Admission plans the next command without a lane that ended between its
// command's passes, as planning does: a prompt that joins the live lane
// starts, though the ended lane, overdue, would lead with rows that leave it
// no room.
void testAdmissionPlansWithoutALaneThatEnded() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.lane(1, 0, 8000);
  sim.lane(2, 20'000, 192);
  const BatchPlan command = sim.plan();
  require(command.width() == 2 && command.passes > 1, "two lanes did not share a command in passes");
  sim.run(command);
  sim.shapes.push_back({.requestId = 1, .ended = true});
  sim.scheduler.submit(request(3, 300));
  const std::array candidate{PrefillAdmission{3, 0}};
  require(sim.scheduler.prefillAdmissionOrder(candidate, sim.in()) == std::vector<uint64_t>{3},
          "admission planned the next command with the lane that ended");
}

// A command between its passes holds its lanes to its last pass: a pass
// before the last consumes no rows. It is not abandoned while a pass of it
// runs; abandoned, it leaves its lanes their rows, which they run again
// from a first pass, a lane suspended meanwhile too once it resumes.
void testCommandBetweenPassesHoldsItsLanes() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.lane(1, 20'000, 300);
  sim.lane(2, 20'000, 300);
  const BatchPlan first = sim.plan();
  require(first.width() == 2 && first.passes > 1, "two turns did not share a command in passes");
  sim.scheduler.commit(first, sim.in());
  PassProgress progress{.plan = first};
  test::rejects([&] { sim.scheduler.abandoned(); }, "cannot be abandoned",
                "a command was abandoned while a pass of it ran");
  const std::array results{StepResult{1, 0, false, DecodeStage::Regular},
                           StepResult{2, 0, false, DecodeStage::Regular}};
  sim.scheduler.complete(first, results, {.wallMilliseconds = 500.0});
  require(sim.scheduler.phase(1) == Phase::Prefill && sim.scheduler.promptProcessed(1) == 20'000,
          "a pass before a command's last consumed its rows");
  sim.scheduler.abandoned();
  BatchPlan plan = sim.plan();
  require(plan.pass == 0 && plan.passes == first.passes && plan.items == first.items,
          "an abandoned command's lanes did not run its rows again from a first pass");
  sim.run(plan);
  sim.abandon();
  sim.scheduler.suspendForResources(1);
  sim.scheduler.resumeFromResources(1, 20'000, 20'300);
  plan = sim.plan();
  require(plan.pass == 0 && plan.items == first.items,
          "a lane suspended between its command's passes did not run its rows again with the others");
}

// prefill_rows counts a command's rows once, with its first pass: not again
// when its lanes rerun them after it was abandoned, but again when a resumed
// lane replays rows that ran before its suspension.
void testAnAbandonedCommandsRowsCountOnce() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.lane(1, 0, 3 * 8192);
  sim.run(sim.plan());
  require(sim.passes.has_value(), "the long command did not run in passes");
  sim.lane(2, 8192, 300, RequestPriority::Foreground);
  require(sim.scheduler.abandonsPasses(sim.plan(), sim.in()), "the turn did not supersede the long command");
  // The turn ends with its prompt, as its higher priority's decode would
  // keep the long prompt waiting.
  while (sim.scheduler.phase(1) == Phase::Prefill || sim.scheduler.phase(2) == Phase::Prefill) {
    sim.step();
    if (sim.scheduler.phase(2) == Phase::Decode)
      sim.scheduler.finish(2);
  }
  const SchedulerSnapshot counts = sim.scheduler.snapshot();
  require(counts.abandonedPassCommands > 0 && counts.prefillRows == 3 * 8192 + 300,
          std::to_string(counts.prefillRows) + " rows counted for the prompts' " + std::to_string(3 * 8192 + 300) +
              " after " + std::to_string(counts.abandonedPassCommands) + " abandoned commands");
  sim.lane(3, 0, 8192);
  BatchPlan plan = sim.plan();
  while (plan.kind != WorkKind::Prefill) {
    sim.run(plan);
    plan = sim.plan();
  }
  sim.run(plan);
  require(sim.passes.has_value(), "the contended command did not run in passes");
  const uint64_t counted = sim.scheduler.snapshot().prefillRows;
  sim.abandon();
  sim.scheduler.suspendForResources(3);
  sim.scheduler.resumeFromResources(3, 0, 8192);
  while (sim.scheduler.phase(3) == Phase::Prefill)
    sim.step();
  require(sim.scheduler.snapshot().prefillRows == counted,
          "a lane suspended in its first command's passes counted " +
              std::to_string(sim.scheduler.snapshot().prefillRows - counted) +
              " of its 8192 rows again once it resumed, which never ran before");
  // A lane whose first command ended, suspended between the passes of its
  // second, resumes from its prompt's start: the first command's rows count
  // again, the second's not.
  sim.lane(4, 0, 2 * 8192);
  plan = sim.plan();
  while (plan.kind != WorkKind::Prefill || !sim.scheduler.promptProcessed(4)) {
    sim.run(plan);
    plan = sim.plan();
  }
  require(plan.width() == 1 && plan.items[0].requestId == 4 && plan.passes > 1,
          "the second command did not run in passes");
  const uint32_t ran = sim.scheduler.promptProcessed(4);
  const uint32_t reached = plan.items[0].promptOffset + plan.items[0].tokenCount;
  sim.run(plan);
  const uint64_t before = sim.scheduler.snapshot().prefillRows;
  sim.abandon();
  sim.scheduler.suspendForResources(4);
  sim.scheduler.resumeFromResources(4, 0, 2 * 8192);
  while (sim.scheduler.phase(4) == Phase::Prefill)
    sim.step();
  const uint64_t replayed = sim.scheduler.snapshot().prefillRows - before;
  require(replayed == ran + (2 * 8192 - reached),
          "a replay from the prompt's start counted " + std::to_string(replayed) + " rows, not the " +
              std::to_string(ran) + " that ran before its suspension and the " + std::to_string(2 * 8192 - reached) +
              " no command counted");
  // The same suspension resumed past the rows counted, where the cache
  // holds the prompt's blocks another request published: the replay's rows
  // all count, none of them counted before.
  sim.lane(5, 0, 3 * 8192);
  plan = sim.plan();
  while (plan.kind != WorkKind::Prefill || !sim.scheduler.promptProcessed(5)) {
    sim.run(plan);
    plan = sim.plan();
  }
  require(plan.width() == 1 && plan.items[0].requestId == 5 && plan.passes > 1,
          "the second command did not run in passes");
  const uint32_t resumed = plan.items[0].promptOffset + plan.items[0].tokenCount + 2048;
  sim.run(plan);
  const uint64_t past = sim.scheduler.snapshot().prefillRows;
  sim.abandon();
  sim.scheduler.suspendForResources(5);
  sim.scheduler.resumeFromResources(5, resumed, 3 * 8192);
  while (sim.scheduler.phase(5) == Phase::Prefill)
    sim.step();
  require(sim.scheduler.snapshot().prefillRows - past == 3 * 8192 - resumed,
          "a replay from " + std::to_string(resumed) + ", past the rows counted, counted " +
              std::to_string(sim.scheduler.snapshot().prefillRows - past) + " of the " +
              std::to_string(3 * 8192 - resumed) + " rows no command counted");
  // A lane admitted on a cached prefix and suspended before its first
  // command replays from the prompt's start: every row of the replay ran
  // in no command of its own, so all count.
  sim.lane(6, 4096, 8192);
  const uint64_t admitted = sim.scheduler.snapshot().prefillRows;
  sim.scheduler.suspendForResources(6);
  sim.scheduler.resumeFromResources(6, 0, 4096 + 8192);
  while (sim.scheduler.phase(6) == Phase::Prefill)
    sim.step();
  require(sim.scheduler.snapshot().prefillRows - admitted == 4096 + 8192,
          "a lane admitted at 4096 and suspended before its first command counted " +
              std::to_string(sim.scheduler.snapshot().prefillRows - admitted) + " of the " +
              std::to_string(4096 + 8192) + " rows its replay ran");
}

// A higher priority's decode runs between a command's passes without abandoning
// it; a higher priority's prompt supersedes the command, cannot commit beside
// it, and runs once it is abandoned, the lanes rerunning from a first pass.
void testPassesYieldToAHigherPriority() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.lane(3, 0, 100, RequestPriority::Foreground);
  awaitFirstMask(sim, 3);
  sim.lane(1, 20'000, 300, RequestPriority::Background);
  sim.lane(2, 20'000, 300, RequestPriority::Background);
  const BatchPlan first = sim.plan();
  require(first.width() == 2 && first.pass == 0 && first.passes > 1,
          "two Background prompts did not share a command in passes");
  sim.run(first);
  sim.scheduler.maskReady(3);
  BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Decode && plan.items[0].requestId == 3 &&
              !sim.scheduler.abandonsPasses(plan, sim.in()),
          "a higher priority's decode waited for a command between its passes, or abandoned it");
  completeDecode(sim, true);
  plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.pass == 1 && plan.items == first.items,
          "a command did not go on with its next pass after a higher priority's decode");
  sim.lane(5, 0, 100, RequestPriority::Foreground);
  plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 5 && plan.width() == 1 &&
              sim.scheduler.abandonsPasses(plan, sim.in()),
          "a higher priority's prompt waited for a command between its passes");
  test::rejects([&] { static_cast<void>(sim.scheduler.commit(plan, sim.in())); }, "batch no longer matches",
                "a higher priority's prompt committed beside a command between its passes");
  sim.abandon();
  require(!sim.scheduler.abandonsPasses(plan, sim.in()), "an abandoned command stayed between its passes");
  while (sim.scheduler.phase(5) == Phase::Prefill)
    sim.step();
  completeDecode(sim, true);
  plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.pass == 0 && plan.items == first.items,
          "an abandoned command's lanes did not run its rows again from a first pass");
}

constexpr test::TestCase kTests[] = {
    {"testRealDecodeWidths", testRealDecodeWidths},
    {"testRejectedCommitCountsNothing", testRejectedCommitCountsNothing},
    {"testDecodeLanesRotate", testDecodeLanesRotate},
    {"testMaskStagesNeverMix", testMaskStagesNeverMix},
    {"testPrefillAndDecodeAlternateWithoutStarvation", testPrefillAndDecodeAlternateWithoutStarvation},
    {"testDecodeRepaysItsShareOfContendedPrefill", testDecodeRepaysItsShareOfContendedPrefill},
    {"testExcludedLanesAreNotPlanned", testExcludedLanesAreNotPlanned},
    {"testPriorityPrecedesWorkKindAlternation", testPriorityPrecedesWorkKindAlternation},
    {"testPrefillCommandContainsOnePriorityTier", testPrefillCommandContainsOnePriorityTier},
    {"testHighestRunnablePriority", testHighestRunnablePriority},
    {"testResourceSuspensionReplaysFromCache", testResourceSuspensionReplaysFromCache},
    {"testAdmissionStartsTheLanesOfTheNextCommand", testAdmissionStartsTheLanesOfTheNextCommand},
    {"testAdmissionStartsAtMostAFullBatch", testAdmissionStartsAtMostAFullBatch},
    {"testAdmissionWaitsForAHigherPrioritysDecode", testAdmissionWaitsForAHigherPrioritysDecode},
    {"testMaskedLanesShareDecode", testMaskedLanesShareDecode},
    {"testMaskWaitAccruesNoDecodeDebt", testMaskWaitAccruesNoDecodeDebt},
    {"testDecodeDebtLeavesWithTheLastDecoder", testDecodeDebtLeavesWithTheLastDecoder},
    {"testHigherPriorityPrefillPrecedesDecodeDebt", testHigherPriorityPrefillPrecedesDecodeDebt},
    {"testDecodeCommandContainsOnePriorityTier", testDecodeCommandContainsOnePriorityTier},
    {"testReplayKeepsHeldInitialMask", testReplayKeepsHeldInitialMask},
    {"testWaitingMaskBoundsPeerPrefill", testWaitingMaskBoundsPeerPrefill},
    {"testOnlyContendedUnitsAreBounded", testOnlyContendedUnitsAreBounded},
    {"testContendedCutsApproachGFromTheFloor", testContendedCutsApproachGFromTheFloor},
    {"testACutFollowsItsLanesDepth", testACutFollowsItsLanesDepth},
    {"testADeeperGroupRunsMorePasses", testADeeperGroupRunsMorePasses},
    {"testAGroupRunsItsTimeOverGPasses", testAGroupRunsItsTimeOverGPasses},
    {"testJoinersStayWithinACutUnit", testJoinersStayWithinACutUnit},
    {"testALowerPrioritysDecodeDoesNotBoundAPrefill", testALowerPrioritysDecodeDoesNotBoundAPrefill},
    {"testWaitingOrPrefillingPeersDoNotBoundAPrefill", testWaitingOrPrefillingPeersDoNotBoundAPrefill},
    {"testContendedPrefillCountsTheUnitsPlanningBounds", testContendedPrefillCountsTheUnitsPlanningBounds},
    {"testContendedPrefillQuantilesAreTheUnitsTimes", testContendedPrefillQuantilesAreTheUnitsTimes},
    {"testContendedPrefillPastItsBucketsReadsTheLongest", testContendedPrefillPastItsBucketsReadsTheLongest},
    {"testAReadyPeerDoesNotBoundAStreamingLeader", testAReadyPeerDoesNotBoundAStreamingLeader},
    {"testUntimedUnitsSizeNothing", testUntimedUnitsSizeNothing},
    {"testACommandHoldsOneLanePastAChunk", testACommandHoldsOneLanePastAChunk},
    {"testALoneDenseCommandRunsWhole", testALoneDenseCommandRunsWhole},
    {"testALoneStreamedRequestRunsPassesWithinG", testALoneStreamedRequestRunsPassesWithinG},
    {"testThePairBoundIsPerLayerAndOnlyRowsMeetIt", testThePairBoundIsPerLayerAndOnlyRowsMeetIt},
    {"testEveryContendedUnitWithinItsBound", testEveryContendedUnitWithinItsBound},
    {"testAContendedSpillLaneKeepsItsGroupRows", testAContendedSpillLaneKeepsItsGroupRows},
    {"testDecodeGapsBesideAStreamingLaneStayWithinTheBound", testDecodeGapsBesideAStreamingLaneStayWithinTheBound},
    {"testTurnsTtftAloneTogetherAndApart", testTurnsTtftAloneTogetherAndApart},
    {"testALongLaneSparesACommandWithTheRowsPastItsWholeCommands",
     testALongLaneSparesACommandWithTheRowsPastItsWholeCommands},
    {"testAStreamingLaneSharesACommand", testAStreamingLaneSharesACommand},
    {"testTwoSpillLanesNeverShareAMultiPassCommand", testTwoSpillLanesNeverShareAMultiPassCommand},
    {"testTheCountersAreThoseOfThePlanCommitted", testTheCountersAreThoseOfThePlanCommitted},
    {"testFewestRemainingRowsLead", testFewestRemainingRowsLead},
    {"testNearEqualLongPromptsAgeByCommands", testNearEqualLongPromptsAgeByCommands},
    {"testALongPromptStartsOnlyToLead", testALongPromptStartsOnlyToLead},
    {"testARestoringStartHoldsBackWhatItWouldLead", testARestoringStartHoldsBackWhatItWouldLead},
    {"testAgingBoundsALongLaneUnderArrivals", testAgingBoundsALongLaneUnderArrivals},
    {"testAWaitingPromptIsOverdueAfterKOvertakes", testAWaitingPromptIsOverdueAfterKOvertakes},
    {"testEqualPromptsFinishInArrivalOrder", testEqualPromptsFinishInArrivalOrder},
    {"testShortArrivalPrecedesLongColdPrompt", testShortArrivalPrecedesLongColdPrompt},
    {"testHigherTierCommandsDoNotAgeALowerTier", testHigherTierCommandsDoNotAgeALowerTier},
    {"testASameTierArrivalLeavesTheRowsACommandStarted", testASameTierArrivalLeavesTheRowsACommandStarted},
    {"testAdmissionPlansWithoutALaneThatEnded", testAdmissionPlansWithoutALaneThatEnded},
    {"testCommandBetweenPassesHoldsItsLanes", testCommandBetweenPassesHoldsItsLanes},
    {"testAnAbandonedCommandsRowsCountOnce", testAnAbandonedCommandsRowsCountOnce},
    {"testPassesYieldToAHigherPriority", testPassesYieldToAHigherPriority},
};

} // namespace

int main(int argc, char **argv) {
  const test::TestGroup groups[] = {{"scheduler", kTests}};
  return test::runTests("ragged-scheduler", groups, argc, argv);
}
