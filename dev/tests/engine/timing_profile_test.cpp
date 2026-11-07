// The fakes' time: the profiles against the commands they are fitted to,
// the structure every profile keeps, the work the fake model reports, and
// the harness clock that runs commands for the profile's time.

#include "EngineHarness.hpp"
#include "TestChecks.hpp"
#include "TestImmediateTicket.hpp"
#include "TestMetalMemory.hpp"
#include "TestRegistry.hpp"
#include "TimingProfile.hpp"
#include "model/WorkFormulas.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace splash;

namespace {

using test::MeasuredCommand;
using test::MeasuredPrompt;
using test::require;
using test::TimedWork;
using test::TimingProfile;

// What a profile takes for a recorded command of one lane from position 0
// at the residency it was measured at.
test::CommandTimes timesOf(const TimingProfile &profile, const MeasuredCommand &command) {
  const TimedWork work(profile);
  if (command.warmup)
    return work.warmupTimes(command.rows);
  const model::PrefillLane lane{.rows = command.rows};
  return work.times({&lane, 1}, 0, 1);
}

// The worst and mean relative error of a profile's walls on its commands.
std::pair<double, double> errors(const TimingProfile &profile, std::span<const MeasuredCommand> commands) {
  double worst = 0.0, sum = 0.0;
  for (const MeasuredCommand &command : commands) {
    const double error =
        std::abs(timesOf(profile, command).wallMilliseconds - command.wallMilliseconds) / command.wallMilliseconds;
    worst = std::max(worst, error);
    sum += error;
  }
  return {worst, sum / commands.size()};
}

// A prompt as the engine runs it on a target without passes: commands of
// one chunk's rows from its position.
double promptMilliseconds(const TimingProfile &profile, const MeasuredPrompt &prompt) {
  const TimedWork work(profile);
  double total = 0.0;
  for (uint32_t done = 0; done < prompt.rows;) {
    const model::PrefillLane lane{.position = prompt.position + done,
                                  .rows = std::min(prompt.rows - done, model::ExecutionLimits::prefillTokenBudget)};
    total += work.times({&lane, 1}, 0, 1).wallMilliseconds;
    done += lane.rows;
  }
  return total;
}

// The M6 profile keeps within its fit's tolerance on the eleven commands it
// is fitted to: worst 20 %, mean 8 %.
void testTheM6ProfileFitsItsCommands() {
  const auto [worst, mean] = errors(test::kM6FlashNextQ2, test::kM6FlashNextQ2Commands);
  require(worst <= 0.20 && mean <= 0.08,
          "the M6 profile's walls are off by " + std::to_string(worst) + " worst, " + std::to_string(mean) + " mean");
}

// The Studio 18G profile keeps within 10 % of each of its four commands.
void testTheStudioProfileFitsItsCommands() {
  const auto [worst, mean] = errors(test::kStudio18GFlashNextQ2, test::kStudio18GFlashNextQ2Commands);
  require(worst <= 0.10, "the Studio profile's walls are off by " + std::to_string(worst));
  static_cast<void>(mean);
}

// The M6's 2048-row warm-up, a prompt of token 0 that routes ten experts a
// layer, takes its recorded 3213 ms within 10 %.
void testTheWarmupIsCalibrated() {
  const double wall = TimedWork(test::kM6FlashNextQ2).warmupTimes(2048).wallMilliseconds;
  require(std::abs(wall - 3213.0) <= 321.3, "the warm-up takes " + std::to_string(wall) + " ms");
}

// For every recorded command of 300 rows or more, the part of its wall its
// rows do not add exceeds its host wait: the fixed part is GPU work and
// turnaround as well, not the host steps alone. A turn of 300 rows waits on
// the host more than twice as long as a 6144-row command on the M6.
void testTheFixedPartExceedsTheHostWait() {
  for (const auto &[profile, commands] :
       {std::pair{&test::kM6FlashNextQ2, std::span<const MeasuredCommand>(test::kM6FlashNextQ2Commands)},
        std::pair{&test::kStudio18GFlashNextQ2, std::span<const MeasuredCommand>(test::kStudio18GFlashNextQ2Commands)}}) {
    for (const MeasuredCommand &command : commands) {
      if (command.rows < 300)
        continue;
      const test::CommandTimes times = timesOf(*profile, command);
      require(times.wallMilliseconds - command.rows * profile->rowMilliseconds > times.hostWaitMilliseconds,
              std::string(profile->name) + ": a command's fixed part is only its host wait");
    }
  }
  const double turn = timesOf(test::kM6FlashNextQ2, {.rows = 300}).hostWaitMilliseconds;
  const double group = timesOf(test::kM6FlashNextQ2, {.rows = 6144}).hostWaitMilliseconds;
  require(turn > 2 * group, "a short turn does not wait on the host longer than a group");
}

// Halving the experts' residency raises a 2048-row command's wall by the
// read time compute no longer hides and a decode command's by its added
// reads, on every profile whose store streams.
void testHalvingTheResidencyAddsItsReadTime() {
  for (const TimingProfile *profile :
       {&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2, &test::kStudio36FlashNextQ2, &test::k35BStudio18G}) {
    const double share = profile->measuredResidentShare;
    const model::PrefillLane lane{.rows = 2048};
    const TimedWork full(*profile, [&] { return share; });
    const TimedWork half(*profile, [&] { return share / 2; });
    const model::CommandWork fullWork = full.prefillWork({&lane, 1}, 0, 1);
    const model::CommandWork halfWork = half.prefillWork({&lane, 1}, 0, 1);
    const test::CommandTimes before = full.times({&lane, 1}, 0, 1);
    const test::CommandTimes after = half.times({&lane, 1}, 0, 1);
    require(halfWork.weightReadBytes > fullWork.weightReadBytes && after.wallMilliseconds > before.wallMilliseconds &&
                std::abs((after.wallMilliseconds - before.wallMilliseconds) -
                         (after.hostWaitMilliseconds - before.hostWaitMilliseconds)) < 1e-6,
            std::string(profile->name) + ": a smaller store did not cost a prefill its reads");
    const double decodeAdded = test::decodeTimes(*profile, 1, 8, share / 2, 0).wallMilliseconds -
                               test::decodeTimes(*profile, 1, 8, share, 0).wallMilliseconds;
    const double expected = (test::decodeBytesPerToken(*profile, share / 2) - test::decodeBytesPerToken(*profile, share)) *
                            8 / profile->decodeWeightBytesPerMillisecond;
    require(decodeAdded > 0 && std::abs(decodeAdded - expected) < 1e-6,
            std::string(profile->name) + ": a smaller store did not cost a decode its reads");
  }
}

// A store-less target streams no experts: its commands wait on no host
// steps, and a lane's history reads lengthen its command by the read time
// compute does not hide, from its first Index page (one lane of 2048 rows
// at 64K).
void testAStoreLessCommandPaysItsHistoryReads() {
  const TimingProfile &profile = test::kStoreLessSpillFlashNext;
  const TimedWork work(profile);
  const auto times = [&](uint32_t indexPages) {
    const model::PrefillLane lane{.position = 65'536, .rows = 2048, .indexPages = indexPages};
    return work.times({&lane, 1}, 0, 1);
  };
  const test::CommandTimes resident = times(0);
  require(resident.hostWaitMilliseconds == 0,
          "a store-less command waits " + std::to_string(resident.hostWaitMilliseconds) + " ms on the host");
  const double compute = resident.wallMilliseconds - profile.turnaroundMilliseconds;
  for (const uint32_t pages : {1u, 2048u}) {
    const test::CommandTimes spilled = times(pages);
    const double unhidden =
        spilled.historyWaitMilliseconds - profile.hiddenShare * std::min(spilled.historyWaitMilliseconds, compute);
    require(spilled.historyWaitMilliseconds > 0 &&
                std::abs(spilled.wallMilliseconds - resident.wallMilliseconds - unhidden) < 1e-6,
            std::to_string(pages) + " Index pages add " +
                std::to_string(spilled.wallMilliseconds - resident.wallMilliseconds) + " ms, not their reads' " +
                std::to_string(unhidden));
  }
}

// Routing is the measured one: a Flash-Next command of 300 rows routes
// 14.5K of the 24,576 experts over its 48 layers; the warm-up routes ten a
// layer.
void testRoutingIsMeasured() {
  const double routed = 48 * test::distinctExperts(test::kM6FlashNextQ2, 300);
  require(std::abs(routed - 14'500.0) <= 145.0,
          "a 300-row command routes " + std::to_string(routed) + " experts");
  require(test::distinctExperts(test::kM6FlashNextQ2, 1) == 10.0 && test::distinctExperts(test::kDense27B, 300) == 0.0,
          "routing exceeds the rows' or routes a dense target's experts");
}

// The dense 27B profile reproduces its benchmark's prompts within 1 %, and
// Flint's within half its records' 0.1 s resolution.
void testTheDenseAndFlintProfilesFitTheirPrompts() {
  for (const MeasuredPrompt &prompt : test::kDense27BPrompts) {
    const double wall = promptMilliseconds(test::kDense27B, prompt);
    require(std::abs(wall - prompt.wallMilliseconds) <= 0.01 * prompt.wallMilliseconds,
            "27B: a " + std::to_string(prompt.rows) + "-row prompt takes " + std::to_string(wall) + " ms");
  }
  for (const MeasuredPrompt &prompt : test::kFlintPrompts) {
    const double wall = promptMilliseconds(test::kFlint, prompt);
    require(std::abs(wall - prompt.wallMilliseconds) <= 50.0,
            "Flint: a " + std::to_string(prompt.rows) + "-row prompt takes " + std::to_string(wall) + " ms");
  }
}

// A command's passes split its layers and what they count: their layers,
// pairs, block-index pairs, expert reads and history reads add up to the
// whole command's.
void testThePassesSplitTheLayersAndTheirWork() {
  for (const TimingProfile *profile : {&test::kM6FlashNextQ2, &test::kFlint}) {
    const TimedWork work(*profile);
    const model::PrefillLane lanes[] = {{.position = 70'000, .rows = 6000, .indexPages = 100}, {.rows = 2000}};
    const test::TimedCommand whole = work.command(lanes, 0, 1);
    test::TimedCommand sum;
    for (uint32_t pass = 0; pass < profile->layers; ++pass) {
      const test::TimedCommand part = work.command(lanes, pass, profile->layers - pass);
      require(part.rows == whole.rows && part.chunks == whole.chunks, "a pass does not run the command's rows");
      sum.layers += part.layers;
      sum.work.attentionPairs += part.work.attentionPairs;
      sum.indexPairs += part.indexPairs;
      sum.work.weightReadBytes += part.work.weightReadBytes;
      sum.work.spilledKvReadBytes += part.work.spilledKvReadBytes;
    }
    require(sum.layers == whole.layers && sum.work.attentionPairs == whole.work.attentionPairs &&
                sum.indexPairs == whole.indexPairs && sum.work.weightReadBytes <= whole.work.weightReadBytes &&
                whole.work.weightReadBytes - sum.work.weightReadBytes < 64 &&
                sum.work.spilledKvReadBytes == whole.work.spilledKvReadBytes,
            std::string(profile->name) + ": the passes' work does not add up to the command's");
  }
  // Flint: the full-attention layers compute every key, the windowed ones at
  // most 513 a row.
  const model::PrefillLane deep{.position = 100'000, .rows = 2048};
  const model::CommandWork flint = TimedWork(test::kFlint).prefillWork({&deep, 1}, 0, 1);
  require(flint.layerAttentionPairs == model::denseAttentionPairs(100'000, 2048) &&
              flint.attentionPairs ==
                  13 * model::denseAttentionPairs(100'000, 2048) + 36 * model::attentionPairs({.windowTokens = 513}, 100'000, 2048),
          "Flint's full and windowed layers are not counted at their reach");
}

// Every pass of a command reports the pairs of the target's heaviest
// attention layer, the whole command's (amend/I.md B1): a pass does not
// split a layer, and the bound on a layer's pairs holds rows to it whatever
// the pass. That includes a pass that runs no attention layer (Flash-Next's
// first of 48, its layer 0) or no full-attention one (Flint's second of 49,
// a windowed layer).
void testEveryPassReportsTheTargetsHeaviestLayer() {
  const model::PrefillLane lanes[] = {{.position = 70'000, .rows = 6000}, {.rows = 2000}};
  for (const TimingProfile *profile : {&test::kM6FlashNextQ2, &test::kFlint}) {
    const TimedWork work(*profile);
    const uint64_t whole = work.prefillWork(lanes, 0, 1).layerAttentionPairs;
    for (const uint32_t passes : {2u, 19u, profile->layers})
      for (uint32_t pass = 0, first = 0; pass < passes; ++pass) {
        const model::CommandWork part = work.prefillWork(lanes, first, passes - pass);
        require(part.layerAttentionPairs == whole,
                std::string(profile->name) + ": pass " + std::to_string(pass) + " of " + std::to_string(passes) +
                    " reports " + std::to_string(part.layerAttentionPairs) +
                    " pairs in a layer, not the target's heaviest layer's " + std::to_string(whole));
        first = model::passLayers(first, passes - pass, profile->layers).end;
      }
  }
  require(TimedWork(test::kM6FlashNextQ2).prefillWork(lanes, 0, 1).layerAttentionPairs ==
                  model::attentionPairs(test::kM6FlashNextQ2.reach, 70'000, 6000) +
                      model::attentionPairs(test::kM6FlashNextQ2.reach, 0, 2000) &&
              TimedWork(test::kFlint).prefillWork(lanes, 0, 1).layerAttentionPairs ==
                  model::denseAttentionPairs(70'000, 6000) + model::denseAttentionPairs(0, 2000),
          "the heaviest layer's pairs are not the lanes' at the heaviest reach");
}

// Every profile is whole: layers that attend, a decode command of four lanes
// slower than one of one, residency and hits within their ranges, and the routing of a
// streamed target rising with its rows.
void testEveryProfileIsWhole() {
  for (const TimingProfile *profile :
       {&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2, &test::kStudio36FlashNextQ2,
        &test::kStoreLessSpillFlashNext, &test::kDense27B, &test::kFlint, &test::k35BStudio18G}) {
    const std::string name(profile->name);
    require(!name.empty() && profile->layers && profile->attentionEvery && profile->turnaroundMilliseconds > 0,
            name + " lacks its layers or turnaround");
    require(profile->decodeMilliseconds[0] > 0 && profile->decodeMilliseconds[3] > profile->decodeMilliseconds[0],
            name + ": four lanes decode no slower than one");
    require(profile->measuredResidentShare > 0 && profile->measuredResidentShare <= 1, name + ": residency out of range");
    if (!profile->experts.experts)
      continue;
    const double hit = test::prefillHit(*profile, profile->measuredResidentShare);
    require(hit >= 0 && hit <= 1 && test::decodeBytesPerToken(*profile, profile->measuredResidentShare) >= 0,
            name + ": hits or reads out of range");
    require(test::distinctExperts(*profile, 300) < test::distinctExperts(*profile, 8192), name + ": routing does not rise");
  }
}

// A ticket's sample is its times' work with its wall; a timed ticket is not
// ready before the clock reaches its end and cannot complete before.
void testTicketsSampleTheirWorkAndWaitForTheirEnd() {
  const test::CommandTimes times{.wallMilliseconds = 40.0, .sample = {.work = {.attentionPairs = 300}}};
  const model::CommandSample sample = test::ImmediateTicket({}, times).sample();
  require(sample.work.attentionPairs == 300 && sample.wallMilliseconds == 40.0, "a ticket's sample is not its times");
  double now = 10.0;
  test::TimedTicket ticket({}, times, [&] { return now; }, 50.0);
  require(!ticket.ready(), "a timed ticket was ready before its end");
  test::rejects([&] { static_cast<void>(ticket.wait()); }, "before its end", "a timed ticket completed early");
  now = 50.0;
  require(ticket.ready() && ticket.sample().work.attentionPairs == 300, "a timed ticket was not ready at its end");
}

// On the harness's clock a command takes the profile's time: the first
// prefill runs from the start for prefillTimes' wall, each next command
// starts at the end of the one before it, the record has them as the model
// received them, and the request's tokens arrive a decode command apart.
void testTheHarnessClockRunsCommandsForTheirTime() {
  test::EngineHarness harness({.timing = &test::kM6FlashNextQ2});
  harness.model.decodeFinishes = false;
  engine::EngineRequest request = harness.request(1, std::vector<uint32_t>(305, 7));
  request.maxNewTokens = 3;
  request.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(request));
  harness.runUntilIdle();
  const auto &commands = harness.commands();
  require(commands.size() == 4 && commands[0].kind == WorkKind::Prefill && commands[0].rows == std::vector<uint32_t>{305},
          "the record is not the prefill and three decodes");
  const model::PrefillLane lane{.rows = 305};
  const double prefill = TimedWork(test::kM6FlashNextQ2).times({&lane, 1}, 0, 1).wallMilliseconds;
  require(commands[0].startMilliseconds == 1.0 && commands[0].wallMilliseconds == prefill,
          "the prefill did not take the profile's time");
  for (size_t index = 1; index < commands.size(); ++index)
    require(commands[index].kind == WorkKind::Decode && commands[index].wallMilliseconds == 149.5 &&
                commands[index].startMilliseconds ==
                    commands[index - 1].startMilliseconds + commands[index - 1].wallMilliseconds,
            "a command did not start at the end of the one before it");
  require(harness.longestDecodeGap(1) == 149.5 && harness.events.tokenTimes.at(1).front() == 1.0 + prefill + 149.5,
          "the tokens did not arrive at the commands' ends");
}

// With a rate, the harness's tier lands a transfer a slot's bytes at that
// rate after it starts, a restore once submitted (the cache submits its
// restores as it ticks), and the harness clock moves to the next landing.
void testTheTierLandsTransfersAtItsRate() {
  test::EngineHarness harness({.tier = test::TierOptions{.bytesPerMillisecond = 0.5}});
  test::TestKvTier &tier = *harness.tier;
  const auto slot = [&] { return tier.acquireSlot(model::DiskBudget::Draw::Cache); };
  const std::unique_ptr<engine::KvTransfer> write = tier.demote(3, slot(), engine::WriteClass::Reclaim, {});
  const std::unique_ptr<engine::KvTransfer> read = tier.restore(slot(), 4, {});
  require(tier.nextLanding() == 201.0, "the write is not due a slot's bytes at the rate after it started");
  harness.advanceTo(200.0);
  require(!write->ready(), "a write landed before it was due");
  harness.advanceTo(201.0);
  require(write->ready() && !read->ready(), "the write did not land when due, or an unsubmitted restore did");
  tier.submitRestores();
  harness.advanceTo(201.0);
  require(read->ready(), "a submitted restore that was due did not land");
  require(write->finish() && read->finish(), "a landed transfer failed");
  const std::unique_ptr<engine::KvTransfer> next = tier.demote(5, slot(), engine::WriteClass::Reclaim, {});
  harness.runUntil([&] { return next->ready(); }, "the next write did not land");
  require(harness.now() == 401.0, "the clock moved to " + std::to_string(harness.now()) + ", not the landing");
}

// Noise is seeded: one seed gives the same walls run after run, another
// different ones.
void testNoiseIsSeeded() {
  const auto walls = [](uint64_t seed) {
    TimingProfile profile = test::kM6FlashNextQ2;
    profile.noise = 0.1;
    profile.seed = seed;
    test::EngineHarness harness({.timing = &profile});
    for (uint64_t id = 1; id <= 3; ++id)
      harness.engine.submit(harness.request(id, std::vector<uint32_t>(400, static_cast<uint32_t>(id))));
    harness.runUntilIdle();
    std::vector<double> result;
    for (const auto &command : harness.commands())
      result.push_back(command.wallMilliseconds);
    return result;
  };
  require(walls(3) == walls(3) && walls(3) != walls(4), "noise is not seeded");
}

} // namespace

int main(int argc, char **argv) {
  static constexpr test::TestCase kTests[] = {
      {"theM6ProfileFitsItsCommands", testTheM6ProfileFitsItsCommands},
      {"theStudioProfileFitsItsCommands", testTheStudioProfileFitsItsCommands},
      {"theWarmupIsCalibrated", testTheWarmupIsCalibrated},
      {"theFixedPartExceedsTheHostWait", testTheFixedPartExceedsTheHostWait},
      {"halvingTheResidencyAddsItsReadTime", testHalvingTheResidencyAddsItsReadTime},
      {"aStoreLessCommandPaysItsHistoryReads", testAStoreLessCommandPaysItsHistoryReads},
      {"routingIsMeasured", testRoutingIsMeasured},
      {"theDenseAndFlintProfilesFitTheirPrompts", testTheDenseAndFlintProfilesFitTheirPrompts},
      {"thePassesSplitTheLayersAndTheirWork", testThePassesSplitTheLayersAndTheirWork},
      {"everyPassReportsTheTargetsHeaviestLayer", testEveryPassReportsTheTargetsHeaviestLayer},
      {"everyProfileIsWhole", testEveryProfileIsWhole},
      {"ticketsSampleTheirWorkAndWaitForTheirEnd", testTicketsSampleTheirWorkAndWaitForTheirEnd},
      {"theHarnessClockRunsCommandsForTheirTime", testTheHarnessClockRunsCommandsForTheirTime},
      {"theTierLandsTransfersAtItsRate", testTheTierLandsTransfersAtItsRate},
      {"noiseIsSeeded", testNoiseIsSeeded},
  };
  const test::TestGroup groups[] = {{"timing-profile", kTests}};
  return test::runTests("timing-profile", groups, argc, argv, [] { test::metalStatistics() = {}; });
}
