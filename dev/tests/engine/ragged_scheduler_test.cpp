#include "TestChecks.hpp"
#include "engine/Scheduler.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::rejects;
using splash::test::require;

engine::RequestSpec
request(uint64_t id, uint32_t prompt,
        RequestPriority priority = RequestPriority::Normal) {
  return {id, priority, prompt, 10'000.0};
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
          "admission did not take cached and short work ahead of cold work");
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
  scheduler.submit(request(5, 1, RequestPriority::Foreground));
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
  scheduler.submit(request(1, 1, RequestPriority::Foreground));
  scheduler.resourcesReady(1, 0);
  awaitFirstMask(scheduler, 1);
  scheduler.submit(request(2, 1, RequestPriority::Background));
  scheduler.resourcesReady(2, 1);
  scheduler.submit(request(3, 100));
  scheduler.resourcesReady(3, 0);
  scheduler.submit(request(4, 100, RequestPriority::Foreground));
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

// A prefill command's attention stays within a full chunk's at the end of a
// 262,144-token native window: below it every command takes a whole chunk;
// past it, at depths only YaRN serves, a lane's rows halve until they fit
// (512 at a million tokens, after a cached prefix too), and another deep lane
// joins the command only within what is left. Without a bound every command
// takes a whole chunk.
void testDeepPrefillKeepsANativeChunksAttention() {
  const uint64_t native = Scheduler::attentionPairs(262'144 - 2048, 2048);
  const auto rows = [](uint64_t bound, std::initializer_list<uint32_t> positions) {
    Scheduler scheduler(0.0, bound);
    uint64_t id = 0;
    for (const uint32_t position : positions) {
      scheduler.submit(request(++id, 1'010'000));
      scheduler.resourcesReady(id, position);
    }
    const BatchPlan plan = *scheduler.next({});
    std::vector<uint32_t> result;
    for (const BatchItem &item : plan.items)
      result.push_back(item.tokenCount);
    return result;
  };
  using Rows = std::vector<uint32_t>;
  require(rows(native, {0}) == Rows{2048} && rows(native, {260'096}) == Rows{2048} &&
              rows(native, {300'000}) == Rows{1024} && rows(native, {1'000'000}) == Rows{512} &&
              rows(std::numeric_limits<uint64_t>::max(), {1'000'000}) == Rows{2048},
          "a prefill command's attention left a native chunk's");
  require(rows(native, {600'000, 600'000}) == Rows({512, 192}),
          "a second deep lane overran the command's attention");
}

// Every lane but a prefill command's last lies within its first chunk, so
// that at most one lane's rows run past a chunk however the engine cuts or
// drops lanes, and a command holds at most each lane's group rows: a deep
// group leader whose rows halve to more than a chunk runs alone, one halved
// to at most a chunk leaves the rest of its group to a peer, a lane without
// groups (a prompt with images) joins only a command of at most a chunk, and
// no estimate, depth or peer breaks either.
void testPrefillGroupHasOneSpanningLane() {
  constexpr uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  const uint64_t native = Scheduler::attentionPairs(262'144 - 2048, 2048);
  // A command whose 2048 rows took wallMilliseconds, hostMilliseconds of
  // it waiting on expert reads; none before the first command.
  struct Sample final {
    double wallMilliseconds = 0.0;
    double hostMilliseconds = 0.0;
  };
  // A lane with `remaining` rows after `position`, of 8192-row groups or,
  // like a prompt with images, of chunks.
  struct Lane final {
    uint32_t position = 0;
    uint32_t remaining = 0;
    uint32_t groupRows = 4 * chunk;
  };
  using Rows = std::vector<uint32_t>;
  // The rows of the first command's lanes, and their group rows.
  struct Planned final {
    Rows rows;
    Rows groupRows;
  };
  const auto plan = [&](Sample sample, std::initializer_list<Lane> lanes) {
    Scheduler scheduler(0.0, native);
    if (sample.wallMilliseconds > 0.0)
      scheduler.observePrefill(chunk, sample.wallMilliseconds, sample.hostMilliseconds);
    uint64_t id = 0;
    for (const Lane &lane : lanes) {
      RequestSpec spec = request(++id, lane.position + lane.remaining);
      spec.prefillGroupRows = lane.groupRows;
      scheduler.submit(spec);
      scheduler.resourcesReady(id, lane.position);
    }
    const BatchPlan command = *scheduler.next({});
    Planned result;
    for (const BatchItem &item : command.items) {
      result.rows.push_back(item.tokenCount);
      result.groupRows.push_back(lanes.begin()[item.requestId - 1].groupRows);
    }
    return result;
  };
  // The lanes whose rows run past a chunk as the runtime packs them.
  const auto spanning = [&](const Rows &rows) {
    uint32_t count = 0, offset = 0;
    for (const uint32_t value : rows) {
      count += value && offset / chunk != (offset + value - 1) / chunk;
      offset += value;
    }
    return count;
  };
  const Sample streamed{4900.0, 3400.0};
  // 70K deep, the leader's group of 8192 rows halves to 4096 for its
  // attention; the long peer would have taken the other 4096.
  require(plan(streamed, {{70'000, 20'000}, {0, 40'000}}).rows == Rows{4096},
          "a leader past a chunk shared its command");
  // From 128.5K deep it halves to a chunk, and the peer takes the rest of
  // the group, past the chunk's end.
  require(plan(streamed, {{130'000, 20'000}, {0, 40'000}}).rows == Rows({2048, 6144}),
          "a peer of a leader of one chunk lost the rest of the group");
  // 300K deep it halves to 1024: the peer takes the rest of the group, and
  // the next lane none.
  require(plan(streamed, {{300'000, 20'000}, {0, 40'000}, {0, 50'000}}).rows == Rows({1024, 7168}),
          "a lane joined after one ran past a chunk");
  // A peer of chunks takes only the chunk's other 1024 rows, and keeps the
  // next lane to them too.
  require(plan(streamed, {{300'000, 20'000}, {0, 40'000, chunk}, {0, 50'000}}).rows == Rows({1024, 1024}),
          "a lane without prefill groups ran in a group");
  require(plan(streamed, {{130'000, 20'000}, {0, 40'000, chunk}, {0, 50'000}}).rows == Rows{2048},
          "a lane without prefill groups ran in a group");
  for (const Sample sample : {Sample{}, Sample{600.0}, streamed, Sample{20'000.0, 19'000.0}})
    for (const uint32_t depth : {0u, 70'000u, 150'000u, 255'000u, 300'000u, 600'000u})
      for (const uint32_t remaining : {300u, 3000u, 9000u, 20'000u})
        for (const uint32_t peer : {0u, 100'000u})
          for (const uint32_t peerGroupRows : {chunk, 4 * chunk})
            for (const uint32_t shortest : {1000u, remaining + 1000}) {
              const auto [rows, groupRows] = plan(sample, {{depth, remaining},
                                                           {peer, shortest, peerGroupRows},
                                                           {peer, remaining + 2000},
                                                           {0, remaining + 3000, peerGroupRows}});
              // The engine runs any of the lanes, in order, each with its rows
              // or fewer: here all of them or half.
              for (uint32_t kept = 1; kept < 1u << rows.size(); ++kept)
                for (uint32_t halved = 0; halved < 1u << rows.size(); ++halved) {
                  Rows run;
                  for (uint32_t lane = 0; lane < rows.size(); ++lane)
                    if (kept >> lane & 1)
                      run.push_back(halved >> lane & 1 ? rows[lane] / 2 : rows[lane]);
                  require(spanning(run) <= 1, "several lanes of a prefill command ran past a chunk");
                }
              uint32_t total = 0;
              for (const uint32_t value : rows)
                total += value;
              require(total <= *std::min_element(groupRows.begin(), groupRows.end()),
                      "a prefill command held more than a lane's group rows");
            }
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

// A command's wait on its host steps (a streamed target's expert reads) is
// fixed whatever its rows: the contended slice fits fixed plus per-row cost,
// and grows to twice the fixed part, so a command's rows take at least as
// long as its fixed part and the slice has no step where the fixed part
// alone fills it.
void testFixedPrefillCostSizesContendedSlice() {
  const auto slice = [](double wallMilliseconds, double hostMilliseconds) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, wallMilliseconds, hostMilliseconds);
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    scheduler.submit(request(2, 1));
    scheduler.resourcesReady(2, 1);
    completeDecode(scheduler);
    const BatchPlan plan = *scheduler.next({});
    require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 1,
            "expected the contended prefill");
    return plan.items[0].tokenCount;
  };
  // 0.3 ms a row after the fixed part.
  const auto wall = [](double fixed) { return fixed + 2048 * 0.3; };
  require(slice(wall(0.0), 0.0) == 1024, "a prefill without host waits lost its per-row slice");
  require(slice(wall(200.0), 200.0) == 512, "the contended slice ignored the fixed prefill cost");
  require(slice(wall(490.0), 490.0) == 1024 && slice(wall(510.0), 510.0) == 1024,
          "the contended slice jumped where the fixed part fills it");
  require(slice(4900.0, 3400.0) == 2048, "a streamed prefill was cut below its fixed cost");
}

// A lane whose commands may be prefill groups takes one alone when nothing
// contends: its remaining chunks split evenly over the fewest groups that
// hold them. Beside a decoding peer it runs chunks, and a lane without groups
// (a model without them, or a request with images) keeps to chunks.
void testPrefillGroupsRunUncontended() {
  const auto first = [](uint32_t prompt, uint32_t groupRows, bool contended) {
    Scheduler scheduler(0.0);
    // A streamed target's command: 3.4 s of its 4.9 s waiting on expert reads.
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    RequestSpec spec = request(1, prompt);
    spec.prefillGroupRows = groupRows;
    scheduler.submit(spec);
    scheduler.resourcesReady(1, 0);
    if (contended) {
      scheduler.submit(request(2, 1));
      scheduler.resourcesReady(2, 1);
      completeDecode(scheduler);
    }
    const BatchPlan plan = *scheduler.next({});
    require(plan.kind == WorkKind::Prefill && plan.width() == 1 && plan.items[0].requestId == 1,
            "expected the long prefill alone");
    return plan.items[0].tokenCount;
  };
  require(first(10'000, 8192, false) == 6144, "five chunks did not split into two even prefill groups");
  require(first(8192, 8192, false) == 8192, "four chunks did not run as one prefill group");
  require(first(3000, 8192, false) == 3000, "a prompt of two chunks did not run as one prefill group");
  require(first(2048, 8192, false) == 2048, "a prompt of one chunk did not run as a chunk");
  require(first(10'000, 2048, false) == 2048, "a lane without prefill groups took more than a chunk");
  require(first(10'000, 8192, true) == 2048, "a prefill group ran beside a decoding peer");
}

// A lane of 100,000 rows whose commands may be prefill groups and run in up
// to `passes` passes over its model's layers.
RequestSpec passingLane(uint64_t id, uint32_t passes) {
  RequestSpec spec = request(id, 100'000);
  spec.prefillGroupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
  spec.prefillPasses = passes;
  return spec;
}

// Completes a pass of a prefill command of one lane, which consumes its rows
// if it is the last, and else none of them.
void completePass(engine::Scheduler &scheduler, const BatchPlan &plan, double wallMilliseconds,
                  double hostMilliseconds = 0.0) {
  require(plan.kind == WorkKind::Prefill && plan.width() == 1, "expected a lane's pass alone");
  scheduler.commit(plan, {});
  const uint32_t consumed = plan.pass + 1 == plan.passes ? plan.items[0].tokenCount : 0;
  const std::array result{StepResult{plan.items[0].requestId, consumed, false, DecodeStage::Regular}};
  scheduler.complete(plan, result, wallMilliseconds, true, hostMilliseconds);
}

// Where its model runs a prefill command in passes over its layers, a lane
// keeps its group whatever contends, and runs it alone in as many passes as
// keep each within the contended slice by the command estimate, up to its
// model's. Its later passes come before any other prefill, and before any
// admission, with a decoding peer's commands between them; only the last
// consumes the rows, and the estimate learns from the passes' time
// together. A lane that streams its history runs its commands whole, and so
// does one whose model has no passes or before any estimate.
void testPrefillPassesBoundCommands() {
  Scheduler scheduler(0.5);
  // A streamed target's chunk: 4.9 s, 3.4 s of it on expert reads. A group of
  // four chunks: 3.4 s + 4 x 1.5 s = 9.4 s, 19 passes of the slice.
  scheduler.observePrefill(2048, 4900.0, 3400.0);
  scheduler.submit(passingLane(1, 48));
  scheduler.resourcesReady(1, 0);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  scheduler.submit(request(3, 200'000));
  scheduler.resourcesReady(3, 0);
  scheduler.submit(request(4, 100));
  completeDecode(scheduler);
  BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.width() == 1 && plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount == 8192 && plan.pass == 0 && plan.passes == 19,
          "a lane whose model runs passes did not run its group alone in passes within the slice");
  for (uint32_t pass = 0; pass < plan.passes; ++pass) {
    require(plan.items[0].requestId == 1 && plan.items[0].tokenCount == 8192 && plan.pass == pass &&
                plan.passes == 19 && scheduler.promptProcessed(1) == 0,
            "a command's next pass did not go on, or an earlier one consumed its rows");
    require(!pass || scheduler.prefillAdmissionOrder(std::array{PrefillAdmission{4, 0}}).empty(),
            "a start was admitted between a command's passes");
    completePass(scheduler, plan, 500.0, 300.0);
    // The decoding peer runs what each pass owes it, 250 ms.
    completeDecode(scheduler, false, 250.0);
    plan = *scheduler.next({});
  }
  require(scheduler.promptProcessed(1) == 8192 && scheduler.phase(1) == engine::Phase::Prefill &&
              plan.items[0].requestId == 1 && plan.passes == 19 && plan.pass == 0,
          "the last pass did not consume the command's rows, or the estimate learned from each pass alone");
  require(scheduler.snapshot().prefillBatches == 1 && scheduler.snapshot().prefillRows == 8192,
          "a command in passes counted more than once");
  scheduler.setStreamsHistory(1, true);
  plan = *scheduler.next({});
  require(plan.items[0].requestId == 1 && plan.passes == 1, "a lane that streams its history ran passes");

  for (const uint32_t passes : {1u, 48u}) {
    Scheduler fresh(0.0);
    if (passes == 1)
      fresh.observePrefill(2048, 4900.0, 3400.0);
    fresh.submit(passingLane(1, passes));
    fresh.resourcesReady(1, 0);
    require(fresh.next({})->passes == 1, passes == 1 ? "a model without passes ran passes"
                                                     : "a command ran passes before any estimate");
  }
}

// A command between its passes ends with its lane: suspended, the lane
// replays its rows from its first pass; cancelled, another lane's prefill
// runs next; and a pass that ends its request, as one cancelled while the
// pass ran, leaves the rest.
void testCommandBetweenPassesEndsWithItsLane() {
  Scheduler scheduler(0.0);
  scheduler.observePrefill(2048, 4900.0, 3400.0);
  scheduler.submit(passingLane(1, 48));
  scheduler.resourcesReady(1, 0);
  scheduler.submit(request(2, 200'000));
  scheduler.resourcesReady(2, 0);
  BatchPlan plan = *scheduler.next({});
  completePass(scheduler, plan, 500.0);
  scheduler.suspendForResources(1);
  scheduler.resumeFromResources(1, 0, 100'000);
  plan = *scheduler.next({});
  require(plan.items[0].requestId == 1 && plan.pass == 0 && plan.passes > 1,
          "a lane suspended between its command's passes did not replay from the first");
  completePass(scheduler, plan, 500.0);
  scheduler.cancel(1);
  plan = *scheduler.next({});
  require(plan.items[0].requestId == 2 && plan.pass == 0 && plan.passes == 1,
          "a cancelled command's passes went on");

  Scheduler ending(0.0);
  ending.observePrefill(2048, 4900.0, 3400.0);
  ending.submit(passingLane(1, 48));
  ending.resourcesReady(1, 0);
  plan = *ending.next({});
  ending.commit(plan, {});
  const std::array ended{StepResult{1, 0, true, DecodeStage::Regular}};
  ending.complete(plan, ended, 500.0, true);
  require(ending.phase(1) == engine::Phase::Completed && !ending.next({}),
          "a pass that ended its request left the rest of its command to run");
}

// Completes a prefill command of one lane that streamed its history, which
// waited streamMilliseconds of its hostMilliseconds on the reads.
void completeStreamedPrefill(engine::Scheduler &scheduler, const BatchPlan &plan,
                             double wallMilliseconds, double hostMilliseconds,
                             double streamMilliseconds) {
  require(plan.kind == WorkKind::Prefill && plan.width() == 1, "expected a streaming lane alone");
  scheduler.commit(plan, {});
  const std::array result{
      StepResult{plan.items[0].requestId, plan.items[0].tokenCount, false, DecodeStage::Regular}};
  scheduler.complete(plan, result, wallMilliseconds, true, hostMilliseconds, streamMilliseconds);
}

// A lane that streams its history runs alone: leading, it takes no peers
// into its command; behind a leader, it is passed over while the others
// share the command, until later arrivals have overtaken it often enough
// that it leads one.
void testStreamingLaneRunsAlone() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 100));
  scheduler.submit(request(2, 300));
  scheduler.submit(request(3, 500));
  for (uint64_t id = 1; id <= 3; ++id)
    scheduler.resourcesReady(id, 0);
  scheduler.setStreamsHistory(1, true);
  BatchPlan plan = *scheduler.next({});
  require(plan.width() == 1 && plan.items[0].requestId == 1 && plan.items[0].tokenCount == 100,
          "a streaming leader shared its command");
  scheduler.setStreamsHistory(1, false);
  scheduler.setStreamsHistory(2, true);
  plan = *scheduler.next({});
  require(plan.width() == 2 && plan.items[0].requestId == 1 && plan.items[1].requestId == 3,
          "a streaming lane joined a command it did not lead");

  Scheduler overtaken(0.0);
  overtaken.submit(request(1, 4000));
  overtaken.resourcesReady(1, 0);
  overtaken.setStreamsHistory(1, true);
  uint64_t id = 2;
  for (; id <= model::ExecutionLimits::maximumBatchWidth; ++id) {
    overtaken.submit(request(id, 100));
    overtaken.resourcesReady(id, 0);
    plan = *overtaken.next({});
    require(plan.width() == 1 && plan.items[0].requestId == id,
            "a passed-over streaming lane led or joined a command");
    completePrefill(overtaken, plan);
    overtaken.cancel(id);
  }
  overtaken.submit(request(id, 100));
  overtaken.resourcesReady(id, 0);
  plan = *overtaken.next({});
  require(plan.width() == 1 && plan.items[0].requestId == 1,
          "an overtaken streaming lane did not lead a command of its own");
}

// Contended, a lane that streams its history takes the fewest rows, halving
// from its prefill group's, whose time still covers its estimated wait on
// the reads: fewer would not end the command sooner. It never takes fewer
// than another contended command, which it takes before it has streamed.
// Uncontended, it takes its group.
void testContendedStreamingLaneHalvesFromItsGroup() {
  // At 0.5 ms a row, a contended command takes 512 rows.
  const auto first = [](std::optional<double> streamMilliseconds, bool contended) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 1024.0);
    RequestSpec spec = request(1, 100'000);
    spec.prefillGroupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
    scheduler.submit(spec);
    scheduler.resourcesReady(1, 0);
    scheduler.setStreamsHistory(1, true);
    if (streamMilliseconds) {
      const BatchPlan streamed = *scheduler.next({});
      completeStreamedPrefill(scheduler, streamed,
                              streamed.items[0].tokenCount * 0.5 + *streamMilliseconds,
                              *streamMilliseconds, *streamMilliseconds);
    }
    if (contended) {
      scheduler.submit(request(2, 1));
      scheduler.resourcesReady(2, 1);
      completeDecode(scheduler);
    }
    const BatchPlan plan = *scheduler.next({});
    require(plan.kind == WorkKind::Prefill && plan.width() == 1 && plan.items[0].requestId == 1,
            "expected the streaming lane alone");
    return plan.items[0].tokenCount;
  };
  require(first(std::nullopt, true) == 512,
          "a streaming lane's first contended command took more than another's");
  require(first(2800.0, true) == 8192, "a contended stream that dominates its group was cut");
  require(first(1000.0, true) == 2048,
          "a contended streaming lane did not take the fewest rows covering its stream");
  require(first(0.0, true) == 512, "a stream nothing waited on cut a command below another's");
  require(first(2800.0, false) == 8192, "an uncontended streaming lane did not take its group");
}

// A lane that streams its history may run commands that read none of it,
// before its first pages spill: they leave its estimate of the wait alone,
// which its first command that read the history sets.
void testStreamEstimateIgnoresCommandsThatReadNothing() {
  Scheduler scheduler(0.0);
  scheduler.observePrefill(2048, 1024.0);
  RequestSpec spec = request(1, 100'000);
  spec.prefillGroupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
  scheduler.submit(spec);
  scheduler.resourcesReady(1, 0);
  scheduler.setStreamsHistory(1, true);
  const BatchPlan unread = *scheduler.next({});
  completePrefill(scheduler, unread, unread.items[0].tokenCount * 0.5);
  const BatchPlan streamed = *scheduler.next({});
  completeStreamedPrefill(scheduler, streamed, streamed.items[0].tokenCount * 0.5 + 2800.0,
                          2800.0, 2800.0);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler);
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount == 8192,
          "a command that read no history lowered its lane's estimate of the reads");
}

// A streaming command's wait on its history is its lane's own cost: the
// command estimate other commands' slices come from leaves it out, and the
// rows it ran train the per-row estimate as any command's do.
void testStreamWaitStaysOutOfTheCommandEstimate() {
  Scheduler scheduler(0.0);
  scheduler.observePrefill(2048, 1024.0);
  scheduler.submit(request(1, 4096));
  scheduler.resourcesReady(1, 0);
  scheduler.setStreamsHistory(1, true);
  // 3 s of reads beside 2048 rows at 0.5 ms each.
  completeStreamedPrefill(scheduler, *scheduler.next({}), 4024.0, 3000.0, 3000.0);
  scheduler.setStreamsHistory(1, false);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler);
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items[0].tokenCount == 512,
          "a streamed command's wait on its reads widened other commands' slices");
}

// Each gap a decoding lane sees between its commands across a streaming
// prefill command is recorded, and the command counts as contended; gaps
// across other commands are not recorded.
void testDecodeGapAcrossStreamingPrefill() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1));
  scheduler.resourcesReady(1, 1);
  scheduler.submit(request(2, 4096));
  scheduler.resourcesReady(2, 0);
  completeDecode(scheduler, false, 40.0);
  completePrefill(scheduler, *scheduler.next({}), 1000.0);
  completeDecode(scheduler, false, 40.0);
  require(scheduler.snapshot().decodeGap.count == 0 &&
              scheduler.snapshot().contendedStreamingPrefillBatches == 0,
          "a gap across an ordinary prefill was recorded");
  scheduler.setStreamsHistory(2, true);
  completeStreamedPrefill(scheduler, *scheduler.next({}), 3000.0, 2500.0, 2500.0);
  completeDecode(scheduler, false, 40.0);
  const SchedulerSnapshot counts = scheduler.snapshot();
  require(counts.decodeGap.count == 1 && counts.decodeGap.maximumMilliseconds == 3040.0 &&
              counts.contendedStreamingPrefillBatches == 1,
          "the gap across a streaming prefill was not recorded");
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
          "ragged prefill did not batch all ready sequences");
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

// The engine ends a lane's command before its planned rows at a state
// boundary whose state or KV it could not reserve (Engine::prepare): the
// scheduler takes the rows the shortened item ran, and the lane resumes from
// them.
void testShortenedPrefillResumesFromItsRows() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 4096));
  scheduler.submit(request(2, 1000));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  BatchPlan plan = *scheduler.next({});
  require(plan.items.size() == 2 && plan.items[0].requestId == 2 &&
              plan.items[1].requestId == 1 && plan.items[1].tokenCount == 1048,
          "the long lane did not take the rest of the budget");
  plan.items[1].tokenCount = 32;
  completePrefill(scheduler, plan);
  require(scheduler.promptProcessed(1) == 32 && scheduler.phase(1) == engine::Phase::Prefill,
          "a shortened command did not keep the rows it ran");
  completeDecode(scheduler, true);
  const BatchPlan resumed = *scheduler.next({});
  require(resumed.items.size() == 1 && resumed.items[0].requestId == 1 &&
              resumed.items[0].promptOffset == 32 &&
              resumed.items[0].tokenCount == model::ExecutionLimits::prefillTokenBudget,
          "the shortened lane did not resume from the rows it ran");
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
  require(plan.width() == 4,
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
  rejects([&] { scheduler.commit(stale, {}); }, "batch no longer matches scheduler state",
          "a commit of a cancelled request's batch was accepted");
  require(scheduler.snapshot().decodeBatches == 0,
          "rejected commit incremented decode counters");
  completeDecode(scheduler);
  require(scheduler.snapshot().decodeBatches == 1,
          "the remaining request's decode was not counted");
}

// Once its first token is selected, a lane that waited for its first mask
// decodes in the commands of the lanes that never did: two of each share one.
void testMaskedLanesShareDecode() {
  Scheduler scheduler(0.0);
  for (uint64_t id : {1, 2}) {
    scheduler.submit(request(id, 1));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan prompts = *scheduler.next({});
  scheduler.commit(prompts, {});
  const std::array promptResults{
      StepResult{1, 1, false, DecodeStage::ApplyInitialMask},
      StepResult{2, 1, false, DecodeStage::ApplyInitialMask},
  };
  scheduler.complete(prompts, promptResults, 0.0, true);
  scheduler.maskReady(1);
  scheduler.maskReady(2);
  completeDecode(scheduler);
  for (uint64_t id : {3, 4}) {
    scheduler.submit(request(id, 1));
    scheduler.resourcesReady(id, 1);
  }
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Decode && plan.width() == 4 &&
              plan.decodeStage == DecodeStage::Regular,
          "lanes that waited for their first masks decoded apart from the others");
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
  scheduler.submit(request(1, 1));
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
  scheduler.submit(request(3, 20'000, RequestPriority::Foreground));
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

void testMeasuredBudgetPreservesPriority() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000, RequestPriority::Foreground));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  scheduler.submit(request(2, 1, RequestPriority::Background));
  scheduler.resourcesReady(2, 1);
  require(scheduler.next({})->items[0].tokenCount == 2048,
          "lower-priority work reduced foreground throughput");

  scheduler.submit(request(3, 20'000, RequestPriority::Foreground));
  scheduler.resourcesReady(3, 0);
  scheduler.submit(request(4, 1, RequestPriority::Foreground));
  scheduler.resourcesReady(4, 1);
  completeDecode(scheduler);
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].promptOffset == 2048 &&
              plan.items[0].tokenCount == 128,
          "adaptive prefill exceeded its shared budget");
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

// Peers that only prefill, none finishing within the measured slice, do not
// contend: the leader keeps the full budget.
void testMeasuredBudgetPreservesPurePrefillThroughput() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  for (uint64_t id = 2; id <= 4; ++id) {
    scheduler.submit(request(id, 20'000));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.width() == 1 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount == model::ExecutionLimits::prefillTokenBudget,
          "pure prefill contention lost throughput");
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
    scheduler.submit(request(2, 31, priority));
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
// one that does not finish in it still takes a full command.
void testShortArrivalBesideLongPrefillEndsAtItsLastRow() {
  const auto beside = [](Scheduler &scheduler, std::initializer_list<uint32_t> arrivals) {
    scheduler.observePrefill(2048, 5632.0);
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    completePrefill(scheduler, *scheduler.next({}), 5632.0);
    uint64_t id = 2;
    for (uint32_t prompt : arrivals) {
      scheduler.submit(request(id, prompt));
      scheduler.resourcesReady(id++, 0);
    }
    return *scheduler.next({});
  };
  Scheduler chat(0.0);
  const BatchPlan plan = beside(chat, {300});
  require(plan.width() == 1 && plan.items[0].requestId == 2 && plan.items[0].tokenCount == 300,
          "a short arrival shared a full command with a long prefill");
  completePrefill(chat, plan, 825.0);
  completeDecode(chat);
  const BatchPlan slice = *chat.next({});
  require(slice.width() == 1 && slice.items[0].requestId == 1 &&
              slice.items[0].tokenCount == 128,
          "the long prefill did not return to slices beside the decoding arrival");

  Scheduler longer(0.0);
  const BatchPlan full = beside(longer, {3000});
  require(full.width() == 1 && full.items[0].requestId == 2 &&
              full.items[0].tokenCount == 2048,
          "an arrival that does not finish in one command lost the full budget");

  Scheduler two(0.0);
  const BatchPlan first = beside(two, {300, 300});
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
  scheduler.submit(request(1, 1, RequestPriority::Background));
  scheduler.submit(request(2, 100, RequestPriority::Foreground));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 0);

  BatchPlan foreground = *scheduler.next({});
  require(foreground.kind == WorkKind::Prefill &&
              foreground.items[0].requestId == 2,
          "work-kind alternation overrode request priority");
}

void testPrefillCommandContainsOnePriorityTier() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 4096, RequestPriority::Foreground));
  scheduler.submit(request(2, 4096, RequestPriority::Background));
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
  scheduler.submit(request(1, 1, RequestPriority::Foreground));
  scheduler.submit(request(2, 1, RequestPriority::Background));
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
  require(first.kind == WorkKind::Decode && first.width() == 4,
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

void testMaskStagesNeverMix() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 1));
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
  for (uint64_t id : {1, 2}) {
    scheduler.submit(request(id, 1));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan prompts = *scheduler.next({});
  scheduler.commit(prompts, {});
  const std::array promptResults{
      StepResult{1, 1, false, DecodeStage::ApplyInitialMask},
      StepResult{2, 1, false, DecodeStage::ApplyInitialMask},
  };
  scheduler.complete(prompts, promptResults, 0.0, true);
  scheduler.maskReady(1);
  scheduler.maskReady(2);
  const BatchPlan selection = *scheduler.next({});
  require(selection.kind == WorkKind::Decode && selection.width() == 2 &&
              selection.decodeStage == DecodeStage::ApplyInitialMask,
          "first-token selections of one priority did not share a plan");
}

void testWaitingMaskExpiresAtRequestDeadline() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1));
  scheduler.resourcesReady(1, 0);
  awaitFirstMask(scheduler, 1);
  require(scheduler.expireDeadlines(10'000.0) &&
              scheduler.phase(1) == engine::Phase::Failed,
          "waiting mask survived its request deadline");

  rejects([&] { scheduler.maskReady(1); }, "request is not waiting for a mask",
          "late mask revived an expired request");
}

void testWaitingMaskBoundsPeerPrefill() {
  for (const RequestPriority priority : {RequestPriority::Foreground,
                                         RequestPriority::Normal,
                                         RequestPriority::Background}) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4096.0);
    scheduler.submit(request(1, 1, priority));
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
  scheduler.submit(request(2, 1));
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
    testFixedPrefillCostSizesContendedSlice();
    testPrefillGroupHasOneSpanningLane();
    testPrefillGroupsRunUncontended();
    testPrefillPassesBoundCommands();
    testCommandBetweenPassesEndsWithItsLane();
    testDeepPrefillKeepsANativeChunksAttention();
    testStreamingLaneRunsAlone();
    testContendedStreamingLaneHalvesFromItsGroup();
    testStreamEstimateIgnoresCommandsThatReadNothing();
    testStreamWaitStaysOutOfTheCommandEstimate();
    testDecodeGapAcrossStreamingPrefill();
    testShortestRemainingFirstUsesActualRows();
    testShortenedPrefillResumesFromItsRows();
    testEqualPromptsFinishInArrivalOrder();
    testShortArrivalPrecedesLongColdPrompt();
    testEqualLanesRunInArrivalOrderWithoutOvertaking();
    testLaneOvertakenThreeTimesLeadsTheNextCommand();
    testServedLaneResetsOvertaking();
    testRealDecodeWidths();
    testUnconstrainedLanesShareDecode();
    testRejectedCommitCountsNothing();
    testMaskedLanesShareDecode();
    testPrefillAndDecodeAlternateWithoutStarvation();
    testDecodeRepaysItsShareOfContendedPrefill();
    testMaskWaitAccruesNoDecodeDebt();
    testExcludedLanesAreNotPlanned();
    testDecodeDebtLeavesWithTheLastDecoder();
    testHigherPriorityPrefillPrecedesDecodeDebt();
    testMeasuredBudgetOnlyLimitsContendedWork();
    testAuxiliaryWorkDoesNotTrainTextPrefillTiming();
    testMeasuredBudgetUsesActualRowsAndRecovers();
    testMeasuredBudgetPreservesPriority();
    testUnavailableTimingAndMinimumBudget();
    testMeasuredBudgetDoesNotCountBlockedPeers();
    testMeasuredBudgetPreservesPurePrefillThroughput();
    testTinyTailDoesNotDistortPrefillThroughput();
    testMeasuredBudgetFinishesShortPrefillPromptly();
    testShortArrivalBesideLongPrefillEndsAtItsLastRow();
    testMeasuredBudgetRetainsOvertakingBound();
    testPriorityPrecedesWorkKindAlternation();
    testPrefillCommandContainsOnePriorityTier();
    testDecodeCommandContainsOnePriorityTier();
    testDecodeLanesRotate();
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
