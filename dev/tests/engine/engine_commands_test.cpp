#include "EngineTestSupport.hpp"
#include "MemoryPasses.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestRegistry.hpp"
#include "engine/Engine.hpp"
#include "engine/PassCommand.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;
using splash::test::Events;
using splash::test::Executor;
using splash::test::guardReleases;
using splash::test::idle;
using splash::test::request;
using splash::test::runUntilIdle;
using splash::test::Spilling;
using splash::test::tickUntil;

namespace {

using splash::test::require;

// A command in three passes over lanes 1 and 2, the lanes in `ended` ended,
// on a scheduler that counts the commands given up.
class PassesOfTwo final {
public:
  PassesOfTwo() {
    BatchPlan first{.kind = WorkKind::Prefill, .items = {{.requestId = 1}, {.requestId = 2}}};
    first.passes = 3;
    std::vector<ModelBatchItem> items(2);
    items[0].requestId = 1;
    items[1].requestId = 2;
    command.start(first, std::move(items), {{}, {}});
  }
  // The plan of its next pass.
  [[nodiscard]] BatchPlan next() const { return command.progress()->plan; }
  [[nodiscard]] uint64_t abandoned() const { return scheduler.snapshot().abandonedPassCommands; }

  struct NoWork final : model::PrefillWorkSource {
    [[nodiscard]] model::CommandWork prefillWork(std::span<const model::PrefillLane>, uint32_t,
                                                 uint32_t) const override {
      return {};
    }
  } work;
  Scheduler scheduler{{.limits = {.commandRows = 2048, .layers = 3, .maximumPasses = 3}}, work};
  std::set<uint64_t> ended;
  PassCommand command{scheduler, [this](uint64_t id) { return ended.contains(id); }};
};

using Released = std::vector<uint64_t>;

// A lane that ends while another goes on stays in the command, through its passes, until the command is given up.
void testAPassCommandReleasesAnEndedLaneOnlyWhenGivenUp() {
  PassesOfTwo passes;
  passes.ended = {2};
  const Released atEnd = passes.command.laneEnded(2, false);
  const Released atPass = passes.command.passCompleted(passes.next());
  require(atEnd.empty() && atPass.empty() && passes.command.holds(2) && passes.abandoned() == 0,
          "a lane that ended beside one that goes on left its command before the command was given up");
  require(passes.command.yield() == Released{2} && !passes.command.active() && passes.abandoned() == 1,
          "the command given up did not release its lane that ended, alone");
}

// With no pass in flight, the end of its last lane gives the command up at once with every lane.
void testAPassCommandWhoseLanesAllEndedIsGivenUp() {
  PassesOfTwo passes;
  passes.ended = {1};
  const Released first = passes.command.laneEnded(1, false);
  passes.ended = {1, 2};
  require(first.empty() && passes.command.laneEnded(2, false) == Released{1, 2} && !passes.command.active() &&
              passes.abandoned() == 1,
          "the command whose lanes all ended between its passes was not given up with them");
}

// A lane that ends while a pass of its command runs waits for that pass: its completion gives the command up.
void testALaneEndingWithAPassInFlightWaitsForThePass() {
  PassesOfTwo passes;
  passes.ended = {1, 2};
  require(passes.command.laneEnded(2, true).empty() && passes.command.active() && passes.abandoned() == 0,
          "the command was given up while a pass of it was in flight");
  require(passes.command.passCompleted(passes.next()) == Released{1, 2} && !passes.command.active() &&
              passes.abandoned() == 1,
          "the pass's completion did not give up the command whose lanes all ended");
}

// The last pass ends the command and gives up nothing: the lanes that ended end with its results.
void testTheLastPassReleasesNoLane() {
  PassesOfTwo passes;
  for (uint32_t pass = 1; pass < 3; ++pass)
    require(passes.command.passCompleted(passes.next()).empty() && passes.next().pass == pass,
            "a pass did not move the command to its next");
  passes.ended = {1, 2};
  require(passes.command.laneEnded(1, true).empty() && passes.command.passCompleted(passes.next()).empty() &&
              !passes.command.active() && passes.abandoned() == 0,
          "the last pass released lanes or gave its command up");
}

// Yielding without a command gives nothing up.
void testYieldingNoPassCommandDoesNothing() {
  PassesOfTwo passes;
  static_cast<void>(passes.command.yield());
  require(passes.command.yield().empty() && passes.abandoned() == 1,
          "yielding no command gave something up");
}

// A prompt with images runs prefill groups as a text prompt does: an image
// anywhere in a conversation leaves its long prompts' commands groups.
void testImagePromptsRunPrefillGroups() {
  for (bool images : {false, true}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    model.groupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
    model.streamedExperts = true;
    Events events;
    engine::Engine engine(test::engineConfig(), cache, model, events);
    guardReleases(storage, engine);
    model.time(test::passingTarget());
    auto value = request(1, std::vector<uint32_t>(8193, 7));
    if (images) {
      value.images = {{4000, 16, 8, 8, 1, 1}};
      value.imagePixels.assign(value.images.front().pixelBytes(), 1);
    }
    engine.submit(std::move(value));
    runUntilIdle(engine);
    require(events.completedCount == 1 && model.prefillChunks.at(1) == 2,
            images ? "a prompt with an image did not run prefill groups"
                   : "a text prompt did not run prefill groups");
  }
}

// A ticket of several commands, as a prefill command whose images are
// encoded ahead of it, runs each on the engine's wakes (advance()): the
// second only once the first has completed, however many wakes come before.
void testTicketAdvancesOnTheEngineWakes() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  model.stagePrefillUntil = std::make_shared<std::atomic<bool>>(false);
  Events events;
  engine::Engine engine(test::engineConfig(), cache, model, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(100, 7)));
  double now = 1;
  for (uint32_t tick = 0; tick < 8; ++tick)
    static_cast<void>(engine.tick(now++));
  require(engine.commandInFlight() && events.completedCount == 0,
          "a prefill command of two ran its second before its first completed");
  *model.stagePrefillUntil = true;
  tickUntil(engine, now, [&] { return idle(engine); },
            "a prefill command of two did not run its second on the engine's wakes");
  require(events.completedCount == 1 && events.failedCount == 0,
          "a prefill command of two did not complete its request");
}

// A long prompt's commands run in passes with a decoding lane's steps between
// them, each pass with its own page table, and only the last commits the rows; a
// cancel between passes ends the request at once, the rest of its command unrun.
void testPrefillPassesInterleaveDecodeAndEndAtACancel() {
  for (const bool cancel : {false, true}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    model.groupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
    model.streamedExperts = true;
    model.passes = 48;
    model.decodeFinishes = false;
    // Each command in order: its kind, first lane, pass and passes.
    struct Command final {
      WorkKind kind;
      uint64_t requestId;
      uint32_t pass, passes, rows;
    };
    std::vector<Command> commands;
    model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
      commands.push_back({plan.kind, plan.items.front().requestId, plan.pass, plan.passes,
                          plan.items.front().tokenCount});
      return plan.kind == WorkKind::Decode ? test::CommandTimes{.wallMilliseconds = 100.0}
                                           : test::CommandTimes{500.0, 300.0};
    };
    Events events;
    engine::Engine engine(test::engineConfig({.decodeShare = 0.5}), cache, model, events);
    guardReleases(storage, engine);
    model.time(test::passingTarget());
    EngineRequest decoding = request(1, std::vector<uint32_t>(100, 3));
    decoding.maxNewTokens = 10'000;
    engine.submit(std::move(decoding));
    double now = 1;
    tickUntil(engine, now, [&] { return events.outputs.contains(1); }, "the short request did not decode");
    // Two groups, each in the passes that bound it beside the decoding lane.
    engine.submit(request(2, std::vector<uint32_t>(8193, 7)));
    const auto prefill = [](const Command &command) {
      return command.kind == WorkKind::Prefill && command.requestId == 2;
    };
    const auto passes = [&] { return std::ranges::count_if(commands, prefill); };
    if (cancel) {
      tickUntil(engine, now, [&] { return passes() == 3 && !engine.commandInFlight(); },
                "the long prompt did not run three passes");
      engine.cancel(2);
      require(engine.snapshot().cancelled == 1, "a cancel between passes did not end the request at once");
      const size_t ran = commands.size();
      for (uint32_t step = 0; step < 8; ++step) static_cast<void>(engine.tick(now++));
      require(passes() == 3 && commands.size() > ran && !model.prefillChunks.contains(2),
              "a cancelled command's passes ran on, or the decoding lane stopped");
    } else {
      tickUntil(engine, now, [&] { return events.outputs.contains(2); }, "the long prompt did not end");
      require(model.prefillChunks.at(2) == 2 && model.prefillRows == 100 + 8193,
              "a command in passes committed its rows other than once, with its last pass");
    }
    std::vector<Command> prefills;
    for (size_t index = 0; index < commands.size(); ++index) {
      const Command &command = commands[index];
      if (!prefill(command)) continue;
      // The command it is a pass of: its first, or once that ended its
      // second.
      const bool second = !prefills.empty() && prefills.front().passes <= prefills.size();
      const uint32_t pass = static_cast<uint32_t>(second ? prefills.size() - prefills.front().passes : prefills.size());
      prefills.push_back(command);
      const Command &first = prefills[second ? prefills.front().passes : 0];
      require(command.passes > 1 && command.rows == first.rows && command.pass == pass &&
                  command.passes == first.passes,
              "a long prompt's command did not run its rows in its passes in order");
      require(!command.pass || (commands[index - 1].kind == WorkKind::Decode && commands[index - 1].requestId == 1),
              "the decoding lane did not step between two passes");
    }
    engine.cancel(1);
    if (!cancel) engine.cancel(2);
    runUntilIdle(engine);
  }
}

// An engine on a streamed target whose prefill commands may run in six passes
// (test::passingTarget), recording the commands it ran. A command takes 3.4 s of
// expert reads plus 1.5 s per 2048 rows (a measured chunk), split over its
// passes, or with targetTimes the target's own; a decode takes 100 ms. A
// governed one runs under `host`'s governor, whose limit a test may close
// (leaveRoom).
struct PassingEngine {
  struct Command final {
    WorkKind kind;
    std::vector<uint64_t> lanes;
    uint32_t pass, passes;
    // The states each lane's item captures.
    std::vector<size_t> captures;
  };
  explicit PassingEngine(bool governed = false)
      : engine(governed ? host.connect({.decodeShare = 0.5}) : test::engineConfig({.decodeShare = 0.5}), cache,
               model, events) {
    model.decodeFinishes = false;
    model.commandTimes = [this](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
      Command command{plan.kind, {}, plan.pass, plan.passes, {}};
      std::vector<model::PrefillLane> lanes;
      for (const ModelBatchItem &item : items) {
        command.lanes.push_back(item.requestId);
        command.captures.push_back(item.stateCaptures.size());
        lanes.push_back({static_cast<uint32_t>(item.logicalPosition), item.tokenCount, 0});
      }
      commands.push_back(std::move(command));
      if (plan.kind == WorkKind::Decode)
        return test::CommandTimes{.wallMilliseconds = 100.0};
      if (targetTimes)
        return test::CommandTimes{.wallMilliseconds =
                                      model.timing->times(lanes, plan.layer, plan.passes - plan.pass).wallMilliseconds};
      uint32_t rows = 0;
      for (const model::PrefillLane &lane : lanes)
        rows += lane.rows;
      return test::CommandTimes{(3400.0 + rows * 1500.0 / 2048) / plan.passes, 3400.0 / plan.passes};
    };
    guardReleases(storage, engine);
    model.time(test::passingTarget());
  }
  // A request of `tokens` prompt tokens of its id, of `priority`.
  void submit(uint64_t id, uint32_t tokens, RequestPriority priority = RequestPriority::Normal) {
    EngineRequest value = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.priority = priority;
    engine.submit(std::move(value));
  }
  // The prefill commands run so far, a pass each.
  [[nodiscard]] size_t prefills() const {
    return std::ranges::count_if(commands, [](const Command &command) { return command.kind == WorkKind::Prefill; });
  }
  void tickUntil(const std::function<bool()> &done, const char *message) {
    ::tickUntil(engine, now, done, message);
  }

  test::TestKvStorage storage{512, 4096, 4};
  KvPool pool{storage, 0};
  engine::Cache cache{pool, nullptr, nullptr};
  Executor model;
  // What a command may hold, which the engine takes as it is built.
  bool shaped = (test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true)(model), true);
  Events events;
  test::GovernedHost host;
  engine::Engine engine;
  std::vector<Command> commands;
  bool targetTimes = false;
  double now = 1;
};

// Short prompts that arrive together share one command in passes: every pass
// runs all their items and captures, and the last commits their rows. A lane
// cancelled between passes rides to the last pass and publishes nothing; a
// command whose every lane is cancelled ends at once, or after its pass in flight.
void testShortPromptsShareACommandInPasses() {
  for (const auto [cancelled, inFlight] : {std::pair{0u, false}, {1u, false}, {3u, false}, {3u, true}}) {
    PassingEngine passing;
    for (uint64_t id = 1; id <= 3; ++id)
      passing.submit(id, 300);
    if (cancelled) {
      passing.tickUntil([&] { return passing.prefills() == 2 && passing.engine.commandInFlight() == inFlight; },
                        "the short prompts did not run two passes");
      for (uint64_t id = 1; id <= cancelled; ++id)
        passing.engine.cancel(id);
      require(passing.engine.snapshot().cancelled == (cancelled == 3 && !inFlight ? 3u : 0u),
              cancelled == 3 && !inFlight ? "a command whose lanes were all cancelled went on"
                                          : "a lane cancelled during its command's passes left the command");
    }
    passing.tickUntil([&] { return idle(passing.engine); }, "the short prompts did not end");
    require(passing.prefills() == (cancelled == 3 ? 2u : 6u) &&
                std::ranges::all_of(passing.commands,
                                    [](const PassingEngine::Command &command) {
                                      return command.kind != WorkKind::Prefill ||
                                             (command.lanes == std::vector<uint64_t>{1, 2, 3} &&
                                              command.passes == 6 &&
                                              command.captures == std::vector<size_t>{1, 1, 1});
                                    }),
            "short prompts did not share one command in passes, each pass with every lane's capture");
    const EngineSnapshot counts = passing.engine.snapshot();
    require(counts.cancelled == cancelled && passing.events.completedCount == 3 &&
                counts.replayStatePublications == 3 - cancelled,
            "a lane cancelled between its command's passes committed, or a lane that ran did not");
  }
}

// A command between its passes does not hold back a higher priority: a
// Foreground request that arrives between them starts at once, and its
// prompt runs before the command's next pass, which yields to it,
// abandoned. Its lanes then run their rows again from a first pass.
void testAHigherPriorityIsNotHeldBehindPasses() {
  PassingEngine passing;
  for (uint64_t id = 1; id <= 2; ++id)
    passing.submit(id, 300, RequestPriority::Background);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
  passing.submit(3, 100, RequestPriority::Foreground);
  const size_t arrived = passing.commands.size();
  passing.tickUntil([&] { return passing.events.usage.contains(3); }, "the Foreground request did not end");
  require(passing.events.startIds.back() == 3 && passing.commands.at(arrived).lanes == std::vector<uint64_t>{3},
          "a higher priority's start waited for a command between its passes");
  passing.tickUntil([&] { return idle(passing.engine); }, "the Background requests did not end");
  const auto rerun = std::ranges::find_if(passing.commands.begin() + static_cast<std::ptrdiff_t>(arrived),
                                          passing.commands.end(), [](const PassingEngine::Command &command) {
                                            return command.lanes == std::vector<uint64_t>{1, 2};
                                          });
  require(rerun != passing.commands.end() && rerun->pass == 0 && passing.events.completedCount == 3 &&
              passing.events.failedCount == 0,
          "an abandoned command's lanes did not run their rows again from a first pass");
}

// Cancelled lanes hold no arrival back: a higher priority's arrival gives up a
// command whose lanes were cancelled halfway through its passes, they end with
// it, and the live lane runs its rows again.
void testLanesCancelledBetweenPassesHoldNoArrivalBack() {
  PassingEngine passing;
  for (uint64_t id = 1; id <= 3; ++id)
    passing.submit(id, 300, RequestPriority::Background);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
  const PassingEngine::Command first = passing.commands.back();
  require(first.kind == WorkKind::Prefill && first.lanes == std::vector<uint64_t>{1, 2, 3} && first.passes >= 4,
          "the Background prompts did not share a command in passes");
  passing.tickUntil([&] { return passing.prefills() == first.passes / 2 && !passing.engine.commandInFlight(); },
                    "the Background prompts did not run half their passes");
  for (uint64_t id = 2; id <= 3; ++id)
    passing.engine.cancel(id);
  passing.submit(4, 100, RequestPriority::Foreground);
  const size_t arrived = passing.commands.size();
  passing.tickUntil([&] { return passing.events.usage.contains(4); }, "the Foreground request did not end");
  require(passing.commands.at(arrived).lanes == std::vector<uint64_t>{4},
          "a higher priority's arrival waited for a command whose other lanes were cancelled halfway through its " +
              std::to_string(first.passes) + " passes");
  passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
  const EngineSnapshot counts = passing.engine.snapshot();
  require(counts.completed == 2 && counts.cancelled == 2 && passing.events.failedCount == 0,
          "the requests did not end as two completed and two cancelled");
}

// A lane suspended between its command's passes abandons the command: the other
// lanes rerun from a first pass, the suspended one once it resumes. A lane
// cancelled there instead gives the arrival its lane: the command is given up,
// which ends it, and no lane is suspended.
void testALaneSuspendedBetweenPassesAbandonsItsCommand() {
  for (const bool cancelled : {false, true}) {
    PassingEngine passing;
    // The Foreground start, of a higher priority, abandons the command
    // (Scheduler::yieldsTo).
    passing.targetTimes = true;
    for (uint64_t id = 1; id <= 4; ++id)
      passing.submit(id, 300, RequestPriority::Background);
    const auto lanes = [](const PassingEngine::Command &command, std::vector<uint64_t> ids) {
      return command.kind == WorkKind::Prefill && command.lanes == ids;
    };
    passing.tickUntil(
        [&] {
          return std::ranges::count_if(passing.commands, [&](const PassingEngine::Command &command) {
                   return lanes(command, {1, 2, 3, 4});
                 }) == 1 &&
                 !passing.engine.commandInFlight();
        },
        "the Background prompts did not run a pass of their command");
    if (cancelled) {
      passing.engine.cancel(4);
      require(passing.engine.snapshot().cancelled == 0, "a lane cancelled between its command's passes left it");
    }
    passing.submit(5, 100, RequestPriority::Foreground);
    passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
    std::vector<PassingEngine::Command> prefills;
    for (const PassingEngine::Command &command : passing.commands)
      if (command.kind == WorkKind::Prefill)
        prefills.push_back(command);
    // The lanes that run again, and those once the suspended one resumes.
    const std::vector<uint64_t> again{1, 2, 3};
    const std::vector<uint64_t> resumed = cancelled ? std::vector<uint64_t>{1, 2, 3} : std::vector<uint64_t>{1, 2, 3, 4};
    require(passing.engine.snapshot().prioritySuspensions == !cancelled &&
                passing.engine.snapshot().cancelled == cancelled && prefills.size() > 2 &&
                prefills[1].lanes == again && prefills[1].pass == 0 &&
                std::ranges::count_if(prefills, [&](const PassingEngine::Command &command) {
                  return lanes(command, resumed) && command.pass;
                }) == 5 &&
                passing.events.completedCount == 5 && passing.events.failedCount == 0,
            "a command whose lane was suspended between its passes ran on, or its lanes did not run again");
  }
}

// A stop that reaches a lane between its command's passes ends it with that
// command: a lane with output is there only when it resumes from a
// suspension and replays its tokens. It finishes as Stop, counted completed,
// and emits nothing more.
void testAStopBetweenPassesFinishesAsStop() {
  PassingEngine passing;
  // Lane 1, with the shortest prompt the least work, yields its lane to a
  // Foreground arrival.
  for (uint64_t id = 1; id <= 4; ++id) {
    EngineRequest lane = request(id, std::vector<uint32_t>(id == 1 ? 100 : 300, static_cast<uint32_t>(id)));
    lane.priority = RequestPriority::Background;
    lane.maxNewTokens = 1000;
    passing.engine.submit(std::move(lane));
  }
  passing.tickUntil(
      [&] {
        return std::ranges::all_of(std::array<uint64_t, 4>{1, 2, 3, 4},
                                   [&](uint64_t id) { return !passing.events.outputs[id].empty(); });
      },
      "the Background lanes did not decode");
  passing.submit(5, 100, RequestPriority::Foreground);
  const size_t arrived = passing.commands.size();
  const auto replayBetweenPasses = [&] {
    if (passing.engine.commandInFlight() || passing.commands.size() == arrived)
      return false;
    const PassingEngine::Command &last = passing.commands.back();
    return last.kind == WorkKind::Prefill && last.lanes == std::vector<uint64_t>{1} && last.pass + 1 < last.passes;
  };
  passing.tickUntil([&] { return passing.events.usage.contains(5) && replayBetweenPasses(); },
                    "the suspended lane did not resume into a replay in passes");
  require(passing.engine.snapshot().prioritySuspensions == 1, "the arrival did not suspend the lane");
  const size_t emitted = passing.events.outputs.at(1).size();

  passing.engine.stop(1);
  passing.tickUntil([&] { return passing.events.reasons.contains(1); }, "the stopped lane did not end");
  for (uint64_t id = 2; id <= 4; ++id)
    passing.engine.cancel(id);
  passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
  const EngineSnapshot counts = passing.engine.snapshot();
  require(passing.events.reasons.at(1) == EngineFinishReason::Stop && passing.events.outputs.at(1).size() == emitted &&
              counts.completed == 2 && counts.cancelled == 3 && counts.failed == 0,
          "a stop between a replay's passes did not finish as Stop with nothing more emitted");
}

// A higher-priority prompt refused its KV takes the memory of the Background
// command between its passes, which a suspension would abandon anyway, before
// any lane yields: no lane is suspended, and the command reruns from a first pass.
void testAHigherPriorityTakesWhatACommandBetweenPassesHolds() {
  PassingEngine passing(true);
  passing.host.bookStates(passing.model);
  // While growth is blocked an extent asks the engine's limit for one
  // state's buffers, which it leaves no room for, and the buffers going
  // back unblock it (Executor::kvGrowthBlocked).
  bool blocked = false;
  passing.model.kvGrowthBlocked = &blocked;
  passing.storage.growthAllowed = [&](uint32_t) {
    return !blocked || static_cast<bool>(passing.host.reserve(64));
  };
  for (uint64_t id = 1; id <= 2; ++id)
    passing.submit(id, 300, RequestPriority::Background);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
  blocked = true;
  passing.host.leaveRoom(0);
  passing.submit(3, 100, RequestPriority::Foreground);
  const size_t arrived = passing.commands.size();
  passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
  const auto rerun = std::ranges::find_if(passing.commands.begin() + static_cast<std::ptrdiff_t>(arrived),
                                          passing.commands.end(), [](const PassingEngine::Command &command) {
                                            return command.lanes == std::vector<uint64_t>{1, 2};
                                          });
  const EngineSnapshot counts = passing.engine.snapshot();
  require(counts.resourceSuspensions == 0 && counts.prioritySuspensions == 0 &&
              passing.model.reclaimedIdleStateBytes != 0 &&
              passing.commands.at(arrived).lanes == std::vector<uint64_t>{3} && rerun != passing.commands.end() &&
              rerun->pass == 0 && passing.events.completedCount == 3 && passing.events.failedCount == 0,
          "a higher priority's refused KV suspended a lane of a command between its passes");
}

// Each later pass of a command in passes runs every lane's page table cut
// anew: a memory-control pass between passes that empties an extent moves
// the second lane's page out of it, and the next pass finds the page where
// it went.
void testLaterPassesFollowEveryLanesMovedPages() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  model.kv = &storage;
  model.groupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
  model.streamedExperts = true;
  model.passes = 6;
  std::vector<std::pair<std::vector<uint64_t>, uint32_t>> prefills;
  model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
    std::vector<uint64_t> lanes;
    uint32_t rows = 0;
    for (const ModelBatchItem &item : items) {
      lanes.push_back(item.requestId);
      rows += item.tokenCount;
    }
    if (plan.kind == WorkKind::Prefill)
      prefills.emplace_back(std::move(lanes), plan.pass);
    return test::CommandTimes{(3400.0 + rows * 1500.0 / 2048) / plan.passes, 3400.0 / plan.passes};
  };
  Events events;
  const EngineConfig config = test::engineConfig();
  engine::Engine engine(config, cache, model, events);
  guardReleases(storage, engine);
  model.time(test::passingTarget());
  double now = 1;
  engine.submit(request(1, std::vector<uint32_t>(65, 1)));
  tickUntil(engine, now, [&] { return idle(engine); }, "the first prompt did not end");
  prefills.clear();
  for (const auto &[id, tokens] : {std::pair{uint64_t{2}, 129u}, {uint64_t{3}, 50u}}) {
    EngineRequest prompt = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    prompt.maxNewTokens = 2;
    engine.submit(std::move(prompt));
  }
  tickUntil(engine, now, [&] { return prefills.size() == 1 && !engine.commandInFlight(); },
            "the two prompts did not run a pass of their command");
  require(prefills.back() == std::pair{std::vector<uint64_t>{3, 2}, 0u}, "the two prompts did not share a command");
  const uint64_t revision = cache.pageTable(2).revision;
  static_cast<void>(test::criticalPass(*config.governor));
  const CacheSnapshot reclaimed = cache.snapshot();
  require(reclaimed.pool.extentCompactions == 1 && reclaimed.pool.extentReleases == 1 &&
              cache.pageTable(2).revision > revision,
          "the memory-control pass did not move the second lane's page");
  tickUntil(engine, now, [&] { return idle(engine); }, "the two prompts did not end");
  require(events.completedCount == 3 && events.failedCount == 0,
          "a later pass did not follow a lane's moved page");
}

// A higher-priority start beside a Background command between its passes is a
// start, not a superseding claim: refused its state, it takes memory from the
// lane that yields first, whose suspension gives the command up, and no lane is
// suspended for memory.
void testAHigherPriorityStartBesideACommandBetweenPassesIsAStart() {
  PassingEngine passing(true);
  for (uint64_t id = 1; id <= 2; ++id)
    passing.submit(id, 300, RequestPriority::Background);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
  // Until state buffers come back, its state asks the engine's limit for
  // one state's, which it leaves no room for.
  passing.model.beginGrowthBlocked = [&] {
    return passing.model.lastBeginId == 3 && *passing.model.evictedStateBytes == 0 &&
           !passing.host.reserve(64);
  };
  passing.host.leaveRoom(0);
  passing.submit(3, 100, RequestPriority::Foreground);
  passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
  const EngineSnapshot counts = passing.engine.snapshot();
  require(counts.prioritySuspensions == 1 && counts.resourceSuspensions == 0 &&
              passing.events.completedCount == 3 && passing.events.failedCount == 0,
          "a higher priority's refused start took the command between its passes, or a lane yielded for memory");
}

// A lane of several that ends at its deadline during its command's passes
// ends once: while the command's next pass is in flight, the engine has
// nothing more to do for it.
void testALaneThatExpiredBetweenPassesEndsOnce() {
  PassingEngine passing;
  passing.submit(1, 300);
  EngineRequest expiring = request(2, std::vector<uint32_t>(300, 2));
  expiring.deadlineMilliseconds = 5000;
  passing.engine.submit(std::move(expiring));
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the prompts did not run a pass of their command");
  const auto hold = std::make_shared<std::atomic<bool>>(false);
  passing.model.holdPrefillUntil = hold;
  passing.tickUntil([&] { return passing.engine.commandInFlight(); }, "the command's next pass did not start");
  passing.now = 5000;
  require(passing.engine.tick(passing.now++) && !passing.engine.tick(passing.now++),
          "a lane that ended during its command's passes kept the engine busy");
  *hold = true;
  passing.model.holdPrefillUntil.reset();
  passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
  require(passing.events.failedCount == 1 && passing.events.completedCount == 1,
          "the expired lane did not fail once, or the other did not complete");
}

// The lane that yields is chosen after the ladder: a superseding prompt refused
// its KV gives up the command between passes, which ends the command's cancelled
// lane, and then suspends the live one, nothing ending twice.
void testTheLaneThatYieldsIsChosenAfterTheLadder() {
  PassingEngine passing;
  for (uint64_t id = 1; id <= 2; ++id)
    passing.submit(id, 300, RequestPriority::Background);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
  passing.engine.cancel(2);
  passing.storage.growthBlocked = true;
  passing.submit(3, 3000, RequestPriority::Foreground);
  passing.tickUntil([&] { return passing.engine.snapshot().resourceSuspensions == 1; },
                    "the Foreground prompt's refused KV did not suspend the live Background lane");
  require(passing.engine.snapshot().cancelled == 1, "the cancelled lane did not end with its command");
  passing.storage.growthBlocked = false;
  passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
  const EngineSnapshot counts = passing.engine.snapshot();
  require(counts.cancelled == 1 && counts.completed == 2 && counts.failed == 0,
          "a lane the ladder ended was chosen to yield");
}

// A lane that ended between its command's passes, cancelled or expired, is never
// the lane that yields: refused memory gives the command up instead, which ends
// that lane, suspends none, and reruns the live lane from a first pass.
void testALaneThatEndedBetweenPassesIsNotSuspended() {
  for (const uint64_t ending : {uint64_t{3}, uint64_t{2}}) {
    for (const bool cancelled : {true, false}) {
      PassingEngine passing;
      passing.model.decodeTokens = 8;
      EngineRequest decoding = request(1, std::vector<uint32_t>(100, 1));
      decoding.maxNewTokens = 400;
      passing.engine.submit(std::move(decoding));
      passing.tickUntil([&] { return passing.events.outputs.contains(1); }, "the decoding lane did not decode");
      for (const uint64_t id : {uint64_t{2}, uint64_t{3}}) {
        EngineRequest prompt = request(id, std::vector<uint32_t>(300, static_cast<uint32_t>(id)));
        if (id == ending)
          prompt.deadlineMilliseconds = 5000;
        passing.engine.submit(std::move(prompt));
      }
      const size_t submitted = passing.commands.size();
      const size_t prefills = passing.prefills();
      // A decoding step runs between the passes.
      passing.tickUntil(
          [&] { return passing.prefills() == prefills + 1 && passing.commands.back().kind == WorkKind::Decode; },
          "the prompts did not run a pass of their command");
      const auto pass = std::ranges::find_if(passing.commands.begin() + static_cast<std::ptrdiff_t>(submitted),
                                             passing.commands.end(), [](const PassingEngine::Command &command) {
                                               return command.kind == WorkKind::Prefill;
                                             });
      require(pass->lanes == std::vector<uint64_t>{2, 3} && pass->passes > 1,
              "the prompts did not share a command in passes");
      if (cancelled) {
        passing.engine.cancel(ending);
      } else {
        passing.now = 5000;
        static_cast<void>(passing.engine.tick(passing.now++));
      }
      const EngineSnapshot before = passing.engine.snapshot();
      require(before.cancelled + before.failed == 0, "a lane that ended between its command's passes left it");
      passing.storage.growthBlocked = true;
      const size_t ended = passing.commands.size();
      passing.tickUntil(
          [&] {
            const EngineSnapshot counts = passing.engine.snapshot();
            return counts.cancelled + counts.failed == 1 || counts.resourceSuspensions;
          },
          "the lane that ended did not end");
      passing.storage.growthBlocked = false;
      passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
      // The command never runs its last pass: it is given up, and the live
      // lane's rows run again from a first pass.
      const auto lanes = [&](std::vector<uint64_t> ids) {
        return std::ranges::find_if(passing.commands.begin() + static_cast<std::ptrdiff_t>(ended),
                                    passing.commands.end(), [&](const PassingEngine::Command &command) {
                                      return command.kind == WorkKind::Prefill && command.lanes == ids &&
                                             (ids.size() == 1 || command.pass + 1 == command.passes);
                                    });
      };
      const auto rerun = lanes({ending == 3 ? uint64_t{2} : uint64_t{3}});
      const EngineSnapshot counts = passing.engine.snapshot();
      require(counts.resourceSuspensions == 0 && counts.prioritySuspensions == 0 &&
                  counts.cancelled == cancelled && counts.failed == !cancelled && counts.completed == 2 &&
                  lanes({2, 3}) == passing.commands.end() && rerun != passing.commands.end() && rerun->pass == 0,
              "a lane that ended between its command's passes was suspended, or the command not given up");
    }
  }
}

// A lane of a command in passes ends at its deadline as it does at a
// cancel: between the passes, while a decoding lane's step runs there, a
// lone lane's command ends at once, its next pass unrun; during a pass, the
// command ends after it.
void testADeadlineEndsALaneOfACommandInPasses() {
  enum class When { BetweenPasses, InAPass };
  for (const When when : {When::BetweenPasses, When::InAPass}) {
    PassingEngine passing;
    if (when == When::BetweenPasses) {
      EngineRequest decoding = request(1, std::vector<uint32_t>(100, 1));
      decoding.maxNewTokens = 10'000;
      decoding.deadlineMilliseconds = 1e9;
      passing.engine.submit(std::move(decoding));
      passing.tickUntil([&] { return passing.events.outputs.contains(1); }, "the decoding lane did not decode");
    }
    EngineRequest expiring = request(2, std::vector<uint32_t>(8193, 2));
    expiring.deadlineMilliseconds = 5000;
    passing.engine.submit(std::move(expiring));
    const auto expiringPasses = [&] {
      return std::ranges::count_if(passing.commands, [](const PassingEngine::Command &command) {
        return command.kind == WorkKind::Prefill && std::ranges::count(command.lanes, 2u);
      });
    };
    passing.tickUntil(
        [&] {
          return expiringPasses() == 1 + (when == When::InAPass) && passing.engine.commandInFlight() &&
                 (when != When::BetweenPasses || passing.commands.back().kind == WorkKind::Decode);
        },
        "the expiring prompt did not run its passes");
    passing.now = 5000;
    static_cast<void>(passing.engine.tick(passing.now++));
    const std::optional<double> wakeup = passing.engine.nextWakeupMilliseconds();
    require(passing.events.failedCount == 1 && (!wakeup || *wakeup >= passing.now),
            "a lane whose deadline passed during its command's passes did not end there");
    if (when == When::BetweenPasses)
      passing.engine.cancel(1);
    passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
    require(expiringPasses() == 1 + (when == When::InAPass) && passing.events.failedCount == 1,
            "a lane whose deadline passed during its command's passes was run again");
  }
}

// During recovery admission waits for no command's passes either: a
// Background lane a Normal start's refusal suspended resumes between them,
// and its replay, of a lower priority, runs once the Normal prompt's command
// has run its last.
void testRecoveryAdmitsBetweenPasses() {
  PassingEngine passing;
  passing.model.beginGrowthBlocked = [&] {
    return passing.model.lastBeginId == 2 && passing.model.requests.at(1).resident;
  };
  EngineRequest background = request(1, std::vector<uint32_t>(65, 1));
  background.priority = RequestPriority::Background;
  background.maxNewTokens = 1000;
  passing.engine.submit(std::move(background));
  passing.tickUntil([&] { return passing.events.outputs.contains(1); }, "the Background lane did not decode");
  passing.submit(2, 300);
  const auto normal = [](const PassingEngine::Command &command) {
    return command.kind == WorkKind::Prefill && command.lanes == std::vector<uint64_t>{2};
  };
  passing.tickUntil([&] { return std::ranges::count_if(passing.commands, normal) == 2 && !passing.engine.commandInFlight(); },
                    "the Normal prompt did not run two passes");
  require(passing.engine.snapshot().prioritySuspensions == 1 && passing.engine.snapshot().resourceResumptions == 1,
          "a suspended request did not resume between a command's passes");
  // The first command of the Normal prompt, and the Background lane's replay
  // after it, if it ran.
  const auto replay = [&] {
    const auto first = std::ranges::find_if(passing.commands, normal);
    return std::pair{first, std::find_if(first, passing.commands.end(), [](const PassingEngine::Command &command) {
                       return command.kind == WorkKind::Prefill && command.lanes == std::vector<uint64_t>{1};
                     })};
  };
  passing.tickUntil([&] { return replay().second != passing.commands.end(); }, "the suspended request did not replay");
  const auto [first, replayed] = replay();
  require(std::count_if(first, replayed, normal) == 6, "a lower priority's replay ran between a command's passes");
  passing.engine.cancel(1);
  passing.tickUntil([&] { return idle(passing.engine); }, "engine did not reach idle");
  require(passing.events.failedCount == 0, "a request failed");
}

// A Resident request that continues a Spill request's prefix holds its
// spilled blocks spilled too (Cache::restoreRequest), so its prompt reads
// its history from the SSD cache as a Spill one does, and runs as prefill
// groups, which read it once for their chunks.
void testAResidentLaneHoldingSpilledBlocksRunsPrefillGroups() {
  // Resident KV holds 120 pages, and the engine's limit no more: a request
  // whose projection does not fit in the room left starts Spill
  // (Engine::placement).
  Spilling spill(256, false, 120 * 100, 0, {4}, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget),
                 {.limitBytes = 120 * 100});
  spill.model.time(test::heavyHistoryTarget());
  // Eight pages and 4,096 tokens of output project 136 pages: it spills,
  // and decodes below the next request's priority.
  EngineRequest spilling = Spilling::prompt(1, 1, 8, 4'096);
  spilling.priority = RequestPriority::Background;
  spill.engine.submit(std::move(spilling));
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 10; },
                 "the Spill request did not decode");
  // Its eight pages, then 64 pages and a token, two chunks or a group: 73
  // pages in all with its output, which memory holds.
  std::vector<uint32_t> prompt(8 * KvCache::pageTokens, 1);
  prompt.resize(prompt.size() + 64 * KvCache::pageTokens + 1, 2);
  EngineRequest continuing = request(2, std::move(prompt));
  continuing.maxNewTokens = 1;
  spill.engine.submit(std::move(continuing));
  spill.runUntil([&] { return spill.events.usage.contains(2); },
                 "the Resident request did not run");
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(counts.spill == 1 && counts.resident == 1 &&
              spill.events.starts.back() == 8 * KvCache::pageTokens &&
              spill.model.prefillChunks.at(2) == 1,
          "a Resident request holding spilled blocks ran its prompt a chunk a command");
  spill.engine.cancel(1);
  spill.runUntilIdle();
}

// The rule for lanes that ended holds for refused spilled-KV reads too: a
// decoding Spill lane refused its hot block cache, or a superseding prompt
// refused its history's staging, gives the command up, ending its cancelled
// lane, and only a live lane yields.
void testASpilledReadsRefusalFollowsTheRuleForLanesThatEnded() {
  struct Lanes final {
    std::vector<uint64_t> ids;
    uint32_t pass, passes;
  };
  const auto setUp = [](Spilling &spill, std::vector<Lanes> &prefills) {
    spill.model.commandTimes = [&prefills](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
      if (plan.kind == WorkKind::Decode)
        return test::CommandTimes{.wallMilliseconds = 100.0};
      Lanes command{{}, plan.pass, plan.passes};
      uint32_t rows = 0;
      for (const ModelBatchItem &item : items) {
        command.ids.push_back(item.requestId);
        rows += item.tokenCount;
      }
      prefills.push_back(std::move(command));
      return test::CommandTimes{(3400.0 + rows * 1500.0 / 2048) / plan.passes, 3400.0 / plan.passes};
    };
    spill.model.time(test::passingTarget());
  };
  const auto sharing = [](const Spilling &spill, const std::vector<Lanes> &prefills) {
    return !prefills.empty() && prefills.back().ids.size() == 2 && prefills.back().pass + 1 < prefills.back().passes &&
           !spill.engine.commandInFlight();
  };
  {
    // The engine's limit holds 200 pages of Resident KV: the first request
    // projects 209 and starts Spill (Engine::placement).
    Spilling spill(1024, false, 200 * 100, 0, {4}, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true),
                   {.limitBytes = 200 * 100});
    std::vector<Lanes> prefills;
    setUp(spill, prefills);
    spill.engine.submit(Spilling::prompt(1, 1, 80, 4096));
    spill.runUntil([&] { return spill.events.outputs.contains(1) && spill.cache.holdsIndexPages(1); },
                   "the Spill lane did not decode with Index pages");
    spill.engine.submit(Spilling::prompt(2, 2, 9, 1));
    spill.engine.submit(Spilling::prompt(3, 3, 9, 1));
    spill.runUntil([&] { return sharing(spill, prefills); }, "the two prompts did not share a command in passes");
    spill.engine.cancel(3);
    require(spill.engine.snapshot().cancelled == 0, "the cancelled lane left its command");
    spill.model.refusedSpilledReads = {1};
    spill.runUntil([&] { return spill.engine.snapshot().cancelled == 1; }, "the cancelled lane did not end");
    spill.model.refusedSpilledReads.clear();
    require(spill.engine.snapshot().resourceSuspensions == 0, "a lane yielded beside one that gave its command up");
    spill.engine.cancel(1);
    spill.runUntilIdle();
    require(spill.events.completedCount == 3 && spill.events.failedCount == 0, "the requests did not end");
  }
  {
    Spilling spill(1024, true, 100'000, 0, Spilling::Extent{64}, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true));
    std::vector<Lanes> prefills;
    setUp(spill, prefills);
    const std::vector<uint32_t> earlier(40 * KvCache::pageTokens + 1, 7);
    spill.engine.submit(request(1, earlier));
    spill.runUntilIdle();
    for (uint64_t id = 2; id <= 3; ++id) {
      EngineRequest background = request(id, std::vector<uint32_t>(300, static_cast<uint32_t>(id)));
      background.priority = RequestPriority::Background;
      spill.engine.submit(std::move(background));
    }
    spill.runUntil([&] { return sharing(spill, prefills); }, "the Background prompts did not share a command in passes");
    spill.engine.cancel(3);
    spill.model.refusedStaging = {4};
    std::vector<uint32_t> prompt = earlier;
    prompt.insert(prompt.end(), 300, 9);
    EngineRequest foreground = request(4, prompt);
    foreground.priority = RequestPriority::Foreground;
    spill.engine.submit(std::move(foreground));
    spill.runUntil([&] { return spill.engine.snapshot().resourceSuspensions == 1; },
                   "the refused prompt did not take its memory from the live lane");
    require(spill.engine.snapshot().cancelled == 1, "the cancelled lane did not end with its command");
    spill.model.refusedStaging.clear();
    spill.runUntilIdle();
    const EngineSnapshot counts = spill.engine.snapshot();
    require(counts.cancelled == 1 && spill.events.failedCount == 0 && spill.events.completedCount == 4,
            "the requests did not end once each");
  }
}

// On a target that can spill, a Foreground prompt the engine's limit refuses
// its KV takes the Background command between its passes: the command, which
// the prompt's own command gives up anyway, costs its claim nothing, and the
// buffers of the states it would capture are what the prompt's KV fits in.
void testASupersedingPromptTakesACommandsMemoryBeforeSpilling() {
  Spilling spill(64, false, 100'000, 0, {4}, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true));
  spill.host.bookStates(spill.model);
  std::vector<std::pair<uint64_t, uint32_t>> prefills;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill)
      prefills.emplace_back(plan.items.front().requestId, plan.pass);
    return test::CommandTimes{};
  };
  // Once the prompt has started Resident, an extent asks the engine's limit
  // for one state's buffers, which it leaves no room for, until buffers go
  // back (Executor::kvGrowthBlocked).
  bool blocked = false;
  spill.model.kvGrowthBlocked = &blocked;
  spill.storage.growthAllowed = [&](uint32_t) {
    if (blocked)
      return static_cast<bool>(spill.host.reserve(64));
    spill.host.leaveRoom(std::nullopt);
    return true;
  };
  spill.model.beginObserver = [&] {
    if (spill.model.lastBeginId == 3) {
      blocked = true;
      spill.host.leaveRoom(0);
    }
  };
  spill.model.time(test::passingTarget());
  for (uint64_t id = 1; id <= 2; ++id) {
    EngineRequest background = Spilling::prompt(id, static_cast<uint32_t>(id), 4, 1);
    background.priority = RequestPriority::Background;
    spill.engine.submit(std::move(background));
  }
  spill.runUntil([&] { return !prefills.empty() && !spill.engine.commandInFlight(); },
                 "the Background prompts did not run a pass");
  EngineRequest foreground = Spilling::prompt(3, 3, 4, 1);
  foreground.priority = RequestPriority::Foreground;
  spill.engine.submit(std::move(foreground));
  spill.runUntilIdle();
  const EngineSnapshot counts = spill.engine.snapshot();
  require(counts.spill.spill == 0 && counts.resourceSuspensions == 0 &&
              spill.model.reclaimedIdleStateBytes != 0 && spill.events.completedCount == 3 &&
              spill.events.failedCount == 0,
          "a superseding prompt did not take the memory of the command between its passes");
}

// A lane that ended between its command's passes, whose KV leaves with that
// command, is no part of a start's projection: a turn that would fit beside
// the lanes that go on starts Resident, not Spill for life.
void testAnEndedLaneLeavesAStartResident() {
  Spilling spill(64, false, 3000, 0, {4}, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6, true));
  std::vector<std::pair<uint64_t, uint32_t>> prefills;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill)
      prefills.emplace_back(plan.items.front().requestId, plan.pass);
    return test::CommandTimes{};
  };
  spill.model.time(test::passingTarget());
  spill.engine.submit(Spilling::prompt(1, 1, 20, 1));
  spill.engine.submit(Spilling::prompt(2, 2, 2, 1));
  spill.runUntil([&] { return !prefills.empty() && !spill.engine.commandInFlight(); },
                 "the prompts did not run a pass of their command");
  require(spill.engine.snapshot().spill.spill == 0 && prefills.back() == std::pair{2UL, 0U},
          "the prompts did not share a command");
  spill.engine.cancel(1);
  spill.engine.submit(Spilling::prompt(3, 3, 6, 1));
  spill.runUntilIdle();
  require(spill.engine.snapshot().spill.spill == 0 && spill.events.failedCount == 0 &&
              spill.events.completedCount == 3,
          "a start counted a lane that had ended between its command's passes");
}

// A lane joins a command with the rows that spare it a command of its own,
// never to fill the leader's chunk, which delays the leader and advances no
// one's completion: beside a prompt of 30 rows, one of 2060 takes 12, and its
// other 2048 run as a chunk of their own.
void testALaneJoinsWithTheRowsThatSpareItACommand() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  // Each prefill command's rows, by lane in submission order.
  std::vector<std::vector<uint32_t>> commands;
  model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill) {
      std::vector<uint32_t> rows(2);
      for (const BatchItem &item : plan.items)
        rows.at(item.requestId - 1) = item.tokenCount;
      commands.push_back(std::move(rows));
    }
    return test::CommandTimes{};
  };
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(2060, 7)));
  engine.submit(request(2, std::vector<uint32_t>(30, 8)));
  runUntilIdle(engine);
  require(events.completedCount == 2 &&
              commands == std::vector<std::vector<uint32_t>>{{12, 30}, {2048, 0}},
          "a lane joined a command with more rows than spare it a command of its own");
}

// A prefill command of each lane's rows as the model ran it.
using RanRows = std::vector<std::vector<std::pair<uint64_t, uint32_t>>>;

void recordPrefills(Executor &model, RanRows &commands) {
  model.commandTimes = [out = &commands](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
    if (plan.kind == WorkKind::Prefill && !plan.pass) {
      std::vector<std::pair<uint64_t, uint32_t>> rows;
      for (size_t index = 0; index < plan.items.size(); ++index) {
        require(items[index].requestId == plan.items[index].requestId &&
                    items[index].tokenCount == plan.items[index].tokenCount,
                "a command reached the model other than as planned");
        rows.emplace_back(plan.items[index].requestId, plan.items[index].tokenCount);
      }
      out->push_back(std::move(rows));
    }
    return test::CommandTimes{};
  };
}

// A capture refused at a boundary inside a lane's command cuts the command there
// (LaneShape::rowCap), and the next command goes on from the boundary: a group of
// 7000 rows crosses its checkpoint at 4096, which the model refuses.
void testACaptureCutReplansWithARowCap() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 1, true)(model);
  RanRows commands;
  recordPrefills(model, commands);
  model.denyReservationAt = engine::kPrefillCheckpointTokens;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
  guardReleases(storage, engine);
  model.time(test::passingTarget());
  engine.submit(request(1, std::vector<uint32_t>(7000, 7)));
  runUntilIdle(engine);
  const EngineSnapshot counts = engine.snapshot();
  require(events.completedCount == 1 && !commands.empty() &&
              commands.front() == std::vector<std::pair<uint64_t, uint32_t>>{{1, engine::kPrefillCheckpointTokens}},
          "the command did not end at the boundary whose capture was refused");
  require(counts.captureCuts == 1 && counts.replans >= 1,
          "the capture cut was not planned again with the lane's rows capped");
  require(commands.size() >= 2 && commands[1].front() == std::pair<uint64_t, uint32_t>{1, 7000 - 4096},
          "the next command did not go on from the boundary");
}

// A capture cut's row cap holds for its tick only: the next tick plans the lane's
// command uncut and admits no prompt that command cannot hold (a long prompt's
// 24 sparing rows fit beside the cut's 4096 rows, not the next command's 8192).
void testARowCapEndsWithItsTick() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 1, true)(model);
  RanRows commands;
  recordPrefills(model, commands);
  model.denyReservationAt = engine::kPrefillCheckpointTokens;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
  guardReleases(storage, engine);
  model.time(test::kM6FlashNextQ2);
  engine.submit(request(1, std::vector<uint32_t>(18'096, 7)));
  double now = 1;
  static_cast<void>(engine.tick(now++));
  require(commands == RanRows{{{1, engine::kPrefillCheckpointTokens}}} && engine.snapshot().captureCuts == 1,
          "the prompt's first group did not end at the boundary whose capture was refused");
  engine.submit(request(2, std::vector<uint32_t>(24'600, 8)));
  tickUntil(engine, now, [&] { return commands.size() == 2; }, "the prompt's next command did not run");
  require(commands[1] == std::vector<std::pair<uint64_t, uint32_t>>{{1, 8192}} &&
              events.startIds == std::vector<uint64_t>{1},
          "admission started a prompt the next command did not serve, planned beside the cut command's rows");
  tickUntil(engine, now, [&] { return idle(engine); }, "the prompts did not end");
  require(events.completedCount == 2 && events.failedCount == 0, "the prompts did not complete");
}

// A lane refused its KV leaves the plan, the others keeping what they were
// granted, and runs once memory frees (the pool holds the short prompt's pages
// but not the long one's beside them).
void testAPartialAdmissionReplansWithoutTheDeniedLane() {
  test::TestKvStorage storage(512, 4096, 4);
  storage.budgetPages = 52;
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  RanRows commands;
  recordPrefills(model, commands);
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(300, 7)));
  engine.submit(request(2, std::vector<uint32_t>(1500, 8)));
  static_cast<void>(engine.tick(1));
  require(commands.size() == 1 && commands.front() == std::vector<std::pair<uint64_t, uint32_t>>{{1, 300}} &&
              engine.snapshot().replans >= 1,
          "the plan was not made again without the lane refused its KV");
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 && model.prefillChunks.at(1) == 1 &&
              model.prefillRows == 1800,
          "the lanes did not each run their rows once");
}

// However prepare's refusals fall, a tick's planning ends: each replan leaves a
// lane out, caps its rows at an earlier boundary or follows a shape change, so a
// tick plans again at most once a lane for refusals, once a lane per boundary its
// command may cross and once for a command between its passes it takes.
void testReplansPerTickAreBounded() {
  constexpr uint32_t groupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
  constexpr std::array<uint32_t, 4> prompts{300, 7000, 9500, 13'000};
  constexpr uint64_t most = prompts.size() * (1 + groupRows / engine::kPrefillCheckpointTokens + 1) + 1;
  uint64_t replanned = 0;
  for (uint32_t seed = 1; seed <= 20; ++seed) {
    std::mt19937 random(seed);
    test::TestKvStorage storage(2048, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    test::commandShape(groupRows, 1, true)(model);
    model.reservationRoom = [&] { return random() % 3 != 0; };
    Events events;
    engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
    guardReleases(storage, engine);
    model.time(test::passingTarget());
    for (uint64_t id = 1; id <= prompts.size(); ++id)
      engine.submit(request(id, std::vector<uint32_t>(prompts[id - 1], static_cast<uint32_t>(id))));
    double now = 1;
    for (uint32_t step = 0; step < 2000 && !idle(engine); ++step) {
      const uint64_t before = engine.snapshot().replans;
      static_cast<void>(engine.tick(now++));
      const uint64_t replans = engine.snapshot().replans - before;
      replanned += replans;
      require(replans <= most, "seed " + std::to_string(seed) + ": a tick planned again " +
                                   std::to_string(replans) + " times");
    }
    require(idle(engine) && events.completedCount == prompts.size(),
            "seed " + std::to_string(seed) + ": the prompts did not end");
  }
  require(replanned > 0, "no refusal sent a plan back");
}

// The rows a request has left to prefill: none once its prompt ended.
uint32_t rowsLeft(const engine::Engine &engine, uint64_t id) {
  const EngineSnapshot counts = engine.snapshot();
  const auto found = std::ranges::find(counts.activeRequests, id, &ActiveRequestSnapshot::id);
  return found != counts.activeRequests.end() && found->phase == Phase::Prefill
             ? found->promptTokens - found->promptProcessed
             : 0;
}

// A higher-priority start takes a lower resident's lane only as the scheduler
// admits it, in ordinary admission and in recovery: with every lane resident,
// a Foreground prompt that would neither lead nor end in the next command
// suspends no lane until the Foreground prompt ahead of it ends; one that
// leads suspends a lane, and the suspended Normal lanes stay so meanwhile.
void testAPreemptingStartWaitsUntilItLeads() {
  constexpr uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  // The Normal decoders resident, the second prompt's tokens, whether it is
  // submitted once the first ran a command, and the lanes suspended before
  // the first's last command.
  for (const auto &[normals, second, afterCommand, suspended] :
       {std::tuple{3u, 3 * chunk + 400, false, 0u}, {4u, 3 * chunk + 400, false, 1u}, {4u, 2 * chunk + 4, true, 2u}}) {
    test::TestKvStorage storage(1024, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
    guardReleases(storage, engine);
    const auto submit = [&](uint64_t id, uint32_t tokens, RequestPriority priority, uint32_t output) {
      EngineRequest value = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
      value.priority = priority;
      value.maxNewTokens = output;
      value.deadlineMilliseconds = 1e9;
      engine.submit(std::move(value));
    };
    double now = 1;
    for (uint64_t id = 1; id <= normals; ++id)
      submit(id, 300, RequestPriority::Normal, 100'000);
    tickUntil(engine, now, [&] { return engine.snapshot().scheduler.decoding == normals; },
              "the Normal requests did not decode");
    submit(5, 3 * chunk + 100, RequestPriority::Foreground, 1);
    tickUntil(engine, now, [&] { return std::ranges::count(events.startIds, 5u) == 1; },
              "the first Foreground prompt did not start");
    if (afterCommand)
      tickUntil(engine, now, [&] { return rowsLeft(engine, 5) == 2 * chunk + 100; },
                "the first Foreground prompt did not run a command");
    submit(6, second, RequestPriority::Foreground, 1);
    std::string early;
    tickUntil(
        engine, now,
        [&] {
          const EngineSnapshot counts = engine.snapshot();
          const bool started = std::ranges::count(events.startIds, 6u) != 0;
          if (early.empty() &&
              ((rowsLeft(engine, 5) > chunk && (started || counts.prioritySuspensions > suspended)) ||
               (!started && counts.resourceResumptions)))
            early = std::to_string(counts.prioritySuspensions) + " lanes suspended, " +
                    std::to_string(counts.resourceResumptions) + " resumed, the second " +
                    (started ? "started" : "not started");
          return started;
        },
        "the second Foreground prompt did not start");
    require(early.empty(), "before the first Foreground prompt's last command, of " + std::to_string(second) +
                               " tokens beside " + std::to_string(normals) + " Normal lanes: " + early);
  }
}

// Foreground prompts that arrive together are judged together, in ordinary
// admission and in recovery: the shorter suspends a Normal lane and leads, and
// the longer suspends no other lane until the shorter's last command.
void testPreemptingStartsAreJudgedTogether() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, executor, events);
  guardReleases(storage, engine);
  const auto submit = [&](uint64_t id, uint32_t tokens, RequestPriority priority, uint32_t output) {
    EngineRequest value = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.priority = priority;
    value.maxNewTokens = output;
    value.deadlineMilliseconds = 1e9;
    engine.submit(std::move(value));
  };
  double now = 1;
  for (uint64_t id = 1; id <= 4; ++id)
    submit(id, 300, RequestPriority::Normal, 100'000);
  tickUntil(engine, now, [&] { return engine.snapshot().scheduler.decoding == 4; },
            "the Normal requests did not decode");
  constexpr uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  submit(5, 4 * chunk + 400, RequestPriority::Foreground, 1);
  submit(6, 2 * chunk + 100, RequestPriority::Foreground, 1);
  std::string early;
  tickUntil(
      engine, now,
      [&] {
        const EngineSnapshot counts = engine.snapshot();
        const bool started = std::ranges::count(events.startIds, 5u) != 0;
        if (early.empty() && (std::ranges::count(events.startIds, 6u) == 0 || rowsLeft(engine, 6) > chunk) &&
            (started || counts.prioritySuspensions > 1))
          early = std::to_string(counts.prioritySuspensions) + " lanes suspended, the longer " +
                  (started ? "started" : "not started");
        return started;
      },
      "the longer Foreground prompt did not start");
  require(early.empty(), "before the shorter Foreground prompt's last command: " + early);
}

// Under --kv-spill always, two prompts tried together are judged as the Spill
// lanes they would start as: the longer, whose 193 rows past its whole commands
// would fit beside the shorter, starts only once it leads.
void testSpillPromptsTriedTogetherPrefillApart() {
  Spilling spill(1024);
  std::vector<std::vector<uint64_t>> prefills;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill) {
      std::vector<uint64_t> lanes;
      for (const BatchItem &item : plan.items)
        lanes.push_back(item.requestId);
      prefills.push_back(std::move(lanes));
    }
    return test::CommandTimes{};
  };
  spill.engine.submit(Spilling::prompt(1, 1, 15, 4));
  spill.engine.submit(Spilling::prompt(2, 2, 70, 4));
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.failedCount == 0 &&
              std::ranges::none_of(prefills, [](const std::vector<uint64_t> &lanes) { return lanes.size() > 1; }),
          "two Spill prompts tried together prefilled together, the later one behind");
}

} // namespace

std::span<const test::TestCase> commandsTests() {
  static constexpr test::TestCase kTests[] = {
      {"testImagePromptsRunPrefillGroups", testImagePromptsRunPrefillGroups},
      {"testAPassCommandReleasesAnEndedLaneOnlyWhenGivenUp", testAPassCommandReleasesAnEndedLaneOnlyWhenGivenUp},
      {"testAPassCommandWhoseLanesAllEndedIsGivenUp", testAPassCommandWhoseLanesAllEndedIsGivenUp},
      {"testALaneEndingWithAPassInFlightWaitsForThePass", testALaneEndingWithAPassInFlightWaitsForThePass},
      {"testTheLastPassReleasesNoLane", testTheLastPassReleasesNoLane},
      {"testYieldingNoPassCommandDoesNothing", testYieldingNoPassCommandDoesNothing},
      {"testTicketAdvancesOnTheEngineWakes", testTicketAdvancesOnTheEngineWakes},
      {"testPrefillPassesInterleaveDecodeAndEndAtACancel", testPrefillPassesInterleaveDecodeAndEndAtACancel},
      {"testShortPromptsShareACommandInPasses", testShortPromptsShareACommandInPasses},
      {"testAHigherPriorityIsNotHeldBehindPasses", testAHigherPriorityIsNotHeldBehindPasses},
      {"testLanesCancelledBetweenPassesHoldNoArrivalBack", testLanesCancelledBetweenPassesHoldNoArrivalBack},
      {"testALaneSuspendedBetweenPassesAbandonsItsCommand", testALaneSuspendedBetweenPassesAbandonsItsCommand},
      {"testAStopBetweenPassesFinishesAsStop", testAStopBetweenPassesFinishesAsStop},
      {"testAHigherPriorityTakesWhatACommandBetweenPassesHolds",
       testAHigherPriorityTakesWhatACommandBetweenPassesHolds},
      {"testAHigherPriorityStartBesideACommandBetweenPassesIsAStart",
       testAHigherPriorityStartBesideACommandBetweenPassesIsAStart},
      {"testALaneThatExpiredBetweenPassesEndsOnce", testALaneThatExpiredBetweenPassesEndsOnce},
      {"testTheLaneThatYieldsIsChosenAfterTheLadder", testTheLaneThatYieldsIsChosenAfterTheLadder},
      {"testALaneThatEndedBetweenPassesIsNotSuspended", testALaneThatEndedBetweenPassesIsNotSuspended},
      {"testLaterPassesFollowEveryLanesMovedPages", testLaterPassesFollowEveryLanesMovedPages},
      {"testADeadlineEndsALaneOfACommandInPasses", testADeadlineEndsALaneOfACommandInPasses},
      {"testRecoveryAdmitsBetweenPasses", testRecoveryAdmitsBetweenPasses},
      {"testAResidentLaneHoldingSpilledBlocksRunsPrefillGroups",
       testAResidentLaneHoldingSpilledBlocksRunsPrefillGroups},
      {"testASpilledReadsRefusalFollowsTheRuleForLanesThatEnded",
       testASpilledReadsRefusalFollowsTheRuleForLanesThatEnded},
      {"testASupersedingPromptTakesACommandsMemoryBeforeSpilling",
       testASupersedingPromptTakesACommandsMemoryBeforeSpilling},
      {"testAnEndedLaneLeavesAStartResident", testAnEndedLaneLeavesAStartResident},
      {"testALaneJoinsWithTheRowsThatSpareItACommand", testALaneJoinsWithTheRowsThatSpareItACommand},
      {"testACaptureCutReplansWithARowCap", testACaptureCutReplansWithARowCap},
      {"testARowCapEndsWithItsTick", testARowCapEndsWithItsTick},
      {"testAPartialAdmissionReplansWithoutTheDeniedLane", testAPartialAdmissionReplansWithoutTheDeniedLane},
      {"testReplansPerTickAreBounded", testReplansPerTickAreBounded},
      {"testAPreemptingStartWaitsUntilItLeads", testAPreemptingStartWaitsUntilItLeads},
      {"testPreemptingStartsAreJudgedTogether", testPreemptingStartsAreJudgedTogether},
      {"testSpillPromptsTriedTogetherPrefillApart", testSpillPromptsTriedTogetherPrefillApart},
  };
  return kTests;
}
