// What a start makes of the prefill FFN's Neural Engine split (engine/AneFfnStartup.cpp) on a model the test plays:
// each outcome, the memory plan it adopts, the automatic context, the calibration a start makes, remembers and checks
// again, and the line it logs.

#include "TestChecks.hpp"
#include "TestModel.hpp"
#include "TestStderr.hpp"
#include "ane/Program.hpp"
#include "engine/AneFfnStartup.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace splash;
using namespace splash::engine;
using Kind = AneFfnOutcome::Kind;
using splash::ops::ane_ffn::Calibration;
using splash::ops::ane_ffn::Check;
using splash::ops::ane_ffn::ChunkTimings;
using splash::ops::ane_ffn::Timings;
using splash::test::rejects;
using splash::test::require;

constexpr std::string_view kSettingUp =
    "Setting up the Neural Engine FFN split (splash serve --disable-ane keeps the FFN on the GPU).";

DeviceCapabilities device() {
  DeviceCapabilities result;
  result.deviceName = "test";
  result.appleGpuFamily = 9;
  result.macosMajor = 26;
  result.macosMinor = 4;
  result.physicalMemoryBytes = 16 * kGiB;
  result.recommendedMaxWorkingSetBytes = 12 * kGiB;
  result.maxBufferLengthBytes = 8 * kGiB;
  result.maxThreadgroupMemoryBytes = 32 * 1024;
  result.maxThreadgroupWidth = 1024;
  result.hasUnifiedMemory = true;
  return result;
}

// The memory plan of a model of 4 GiB of weights with `bytes` set aside for the split.
EngineMemoryPlanResult planMemory(uint64_t bytes) {
  ModelMemoryProfile profile = test::modelMemoryProfile(2 * kGiB, kGiB, kGiB);
  profile.footprint.aneFfnBytes = bytes;
  return evaluateEngineMemoryPlan(device(), profile, 0);
}
uint32_t contextWith(uint64_t bytes) {
  const EngineMemoryPlanResult result = planMemory(bytes);
  return result.plan ? result.plan->maximumContextTokens() : 0;
}

// `tokens` with its thousands grouped, as the log prints a context.
std::string grouped(uint32_t tokens) {
  std::string digits = std::to_string(tokens);
  for (size_t at = digits.size(); at > 3; at -= 3) digits.insert(at - 3, ",");
  return digits;
}

// An M5 Max's timings of Qwen3.8-27B's 34 units, which choose 8 of them.
Timings measured() { return {{14.0 / 34, 11.699, 23.484}, {27.0 / 34, 5.033, 44.683}, 18.990}; }

// The chunks of a split timed over its functions of 2048 down to 512 rows: the GPU alone 5 ms at 512 rows and 20 at
// 2048, each function's split layer 0.8 of the GPU's at its rows but function 2048's at `most` ms and function 640's
// at 5.5, which splits its chunks from 553 rows (ane_ffn_calibration_test); the Neural Engine's part 12 ms of function
// 2048's, 0.8 of each other function's split layer.
ChunkTimings timedChunks(double most) {
  const auto gpu = [](uint32_t rows) { return 5.0 + (rows - 512) * 15.0 / 1536; };
  ChunkTimings chunks;
  chunks.gpu = {{{512, gpu(512)}, {2048, gpu(2048)}}};
  for (uint32_t rows = 2048; rows >= 512; rows -= 128) {
    const double split = rows == 2048 ? most : rows == 640 ? 5.5 : 0.8 * gpu(rows);
    chunks.functions.push_back({rows == 512 ? 512 : rows - 127, rows, split, rows == 2048 ? 12.0 : 0.8 * split});
  }
  return chunks;
}
// A check that matches the calibration of timedChunks(15.5).
constexpr Check kMatching{20.0, 15.5, 12.0};

// A model the test plays: a dense one of 34 units, each `unitBytes` of the split's memory, what its timings and its
// prepared split return, and what the start asked of it.
struct Fake {
  bool dense = true;
  std::string unsupported;
  std::optional<std::string> unavailable;
  uint64_t unitBytes = 256 * kMiB;
  std::function<Timings()> time = measured;
  // What a check of the split times, whatever split prepare() returns.
  Check check = kMatching;
  std::function<AneFfnPrepared(uint32_t, AneFfnTiming)> prepare = [](uint32_t, AneFfnTiming timing) {
    AneFfnPrepared prepared;
    if (timing == AneFfnTiming::Chunks) prepared.chunks = timedChunks(15.5);
    prepared.error = 0.025;
    return prepared;
  };
  bool healthy = true;
  // What this Mac remembers, as lines of text by the most units a plan held.
  std::map<uint32_t, std::string> memory;
  // The calls the start made: of unavailable() and time(), and of prepare() with its arguments.
  uint32_t asked = 0, timings = 0;
  std::optional<std::pair<uint32_t, AneFfnTiming>> prepared;

  [[nodiscard]] uint64_t bytes(uint32_t aneUnits) const { return aneUnits * unitBytes; }
  // What the plan holds with the largest split the model could take, the most units whose plan holds any: the
  // context a start whose calibration fails assumes.
  [[nodiscard]] uint32_t largestSplitContext() const {
    for (uint32_t aneUnits = 33; aneUnits > 0; --aneUnits)
      if (const uint32_t context = contextWith(bytes(aneUnits))) return context;
    return 0;
  }

  AneFfnModel model() {
    AneFfnModel result;
    result.dense = dense;
    result.unsupported = unsupported;
    result.unavailable = [this] {
      ++asked;
      return unavailable;
    };
    result.units = 34;
    result.plannedBytes = [this](uint32_t aneUnits) { return bytes(aneUnits); };
    result.time = [this] {
      ++timings;
      return time();
    };
    result.prepare = [this](uint32_t aneUnits, AneFfnTiming timing) {
      prepared = {aneUnits, timing};
      AneFfnPrepared result = prepare(aneUnits, timing);
      if (timing == AneFfnTiming::Check) result.check = check;
      return result;
    };
    result.recall = [this](uint32_t maxAneUnits) -> std::optional<Calibration> {
      const auto found = memory.find(maxAneUnits);
      return found == memory.end() ? std::nullopt : Calibration::parse(found->second);
    };
    result.remember = [this](uint32_t maxAneUnits, const Calibration &calibration) {
      memory[maxAneUnits] = calibration.text();
    };
    result.forget = [this](uint32_t maxAneUnits) { memory.erase(maxAneUnits); };
    result.healthy = [this] { return healthy; };
    return result;
  }
};

struct Started {
  AneFfnStart start;
  std::string log;
};
Started start(Fake &fake, AneFfnSetting setting = {}, uint32_t context = 0, std::function<bool()> cancelled = {}) {
  Started result;
  result.log = test::capturedStderr(
      [&] { result.start = startAneFfn(fake.model(), setting, context, planMemory, cancelled); });
  return result;
}

bool has(const std::string &log, std::string_view text) { return log.find(text) != std::string::npos; }
// Whether `log` is one line that holds `text`.
bool line(const std::string &log, std::string_view text) {
  return has(log, text) && log.find('\n') == log.size() - 1;
}
bool nothingStarted(const Fake &fake, const Started &started) {
  return !fake.timings && !fake.prepared && !started.start.split &&
         !started.start.plan &&
         !has(started.log, kSettingUp);
}

// --disable-ane leaves the GPU alone, at the context the plan holds without the split, without asking whether this Mac
// could run it, and says so where the split could run.
void testOff() {
  Fake fake;
  Started started = start(fake, {.enabled = false});
  require(started.start.outcome.kind == Kind::Off && nothingStarted(fake, started) && !fake.asked &&
              started.start.outcome.context == contextWith(0) &&
              started.start.outcome.contextWithout == contextWith(0) &&
              line(started.log, "The GPU runs the prefill FFN alone, as given."),
          "--disable-ane was not taken as given");
  fake.unsupported = "ANE FFN split needs unrotated quantized projections";
  started = start(fake, {.enabled = false});
  require(started.start.outcome.kind == Kind::Off && started.log.empty(),
          "--disable-ane was logged for a model the split does not take");
}

// A model of no dense FFN layers logs nothing; one whose layers the split does not take, or on a Mac without the
// private Neural Engine interface, says why. Each serves the context the plan holds without the split.
void testUnsupported() {
  Fake fake;
  fake.dense = false;
  Started started = start(fake);
  require(started.start.outcome.kind == Kind::Unsupported && nothingStarted(fake, started) && started.log.empty() &&
              started.start.outcome.context == contextWith(0),
          "a MoE model was logged");
  fake = {};
  fake.unsupported = "ANE FFN split needs unrotated quantized projections";
  started = start(fake);
  require(started.start.outcome.kind == Kind::Unsupported && nothingStarted(fake, started) && !fake.asked &&
              started.start.outcome.context == contextWith(0) &&
              line(started.log, "The GPU runs the prefill FFN alone: " + fake.unsupported + "."),
          "a dense model the split does not take did not say why");
  fake = {};
  fake.unavailable = "AppleNeuralEngine lacks _ANEClient";
  started = start(fake);
  require(started.start.outcome.kind == Kind::Unsupported && nothingStarted(fake, started) && fake.asked == 1 &&
              started.start.outcome.context == contextWith(0) &&
              line(started.log, "The GPU runs the prefill FFN alone: AppleNeuralEngine lacks _ANEClient."),
          "a Mac without the Neural Engine's interface did not say so: " + started.log);
}

// A --max-context no split leaves: the GPU runs alone, nothing is timed or prepared, and a warning names the most
// any split leaves and the way out. One the GPU alone does not hold either fails the start on its own.
void testRefusedContext() {
  Fake fake;
  const uint32_t without = contextWith(0), most = contextWith(fake.bytes(1));
  require(most < without, "the split's memory does not come out of the KV cache");
  Started started = start(fake, {}, most + 1);
  require(started.start.outcome.kind == Kind::Refused && started.start.outcome.context == most &&
              started.start.outcome.contextWithout == without && nothingStarted(fake, started),
          "a context no split leaves did not refuse the split");
  require(line(started.log, "Warning · Neural Engine FFN split off: --max-context " + std::to_string(most + 1) +
                                " cannot be held with it, and any split leaves at most ") &&
              has(started.log, "; pass --max-context " + std::to_string(most) +
                                   " or less to run it, or --disable-ane, which also silences this line."),
          "the refusal did not warn with the way out: " + started.log);
  started = start(fake, {}, without + 1);
  require(started.start.outcome.kind == Kind::Refused && nothingStarted(fake, started) && started.log.empty(),
          "a context the GPU alone does not hold was warned of");
  // A given share is refused on its own units.
  const uint32_t given = contextWith(fake.bytes(7));
  started = start(fake, {.given = AneFfnSetting::Given{0.2}}, given + 1);
  require(started.start.outcome.kind == Kind::Refused && started.start.outcome.context == given &&
              has(started.log, ", and the given share leaves at most "),
          "a given share was not refused on its own units: " + started.log);
}

// Without --max-context a split whose fewest units leave no context is refused, with a warning, before anything is
// timed, and the engine serves the context the plan holds without it.
void testRefusedAutomatic() {
  Fake fake;
  fake.unitBytes = 16 * kGiB;
  const Started started = start(fake);
  require(started.start.outcome.kind == Kind::Refused && started.start.outcome.context == contextWith(0) &&
              nothingStarted(fake, started) &&
              line(started.log, "Warning · Neural Engine FFN split off: it leaves no memory for context; "
                                "--disable-ane silences this line."),
          "a split that leaves no context was not refused: " + started.log);
}

// Without --max-context the split takes its memory from the KV cache: the start adopts the plan with it, and the
// engine serves the context the plan holds with the split, which the log names beside the GPU's alone.
void testAutomaticContext() {
  Fake fake;
  const Started started = start(fake);
  const AneFfnOutcome &outcome = started.start.outcome;
  require(outcome.kind == Kind::Split && fake.timings == 1 && fake.prepared == std::pair(8u, AneFfnTiming::Chunks),
          "the split of 8 units was not calibrated and prepared");
  require(started.start.plan && started.start.plan->breakdown().aneFfnBytes == fake.bytes(8) &&
              started.start.plan->maximumContextTokens() == contextWith(fake.bytes(8)) &&
              outcome.context == contextWith(fake.bytes(8)) && outcome.context < contextWith(0) &&
              outcome.contextWithout == contextWith(0),
          "the plan with the split was not adopted at the automatic context");
  const size_t split = started.log.find("Neural Engine FFN split at share 0.24 for chunks of 553 rows or more: 15.5 ms "
                                        "per 2048-row FFN layer against 20.0 on the GPU alone, 2.5% from the GPU alone "
                                        "on the Neural Engine's part (calibrated in ");
  require(started.log.find(kSettingUp) < split && split != std::string::npos &&
              has(started.log, "; context " + grouped(contextWith(fake.bytes(8))) + " tokens (" +
                                   grouped(contextWith(0)) + " with --disable-ane)."),
          "the split was not set up and logged with both contexts: " + started.log);
}

// A start serves the automatic context of its own calibration: the chosen split's units, none for a calibration that
// found no gain, and the most units the model could take for one that failed. The plan of each holds it.
void testServedContext() {
  const auto noGain = [] {
    Timings timings = measured();
    timings.gpuAlone = 14.0;
    return timings;
  };
  const auto slowChunks = [](uint32_t, AneFfnTiming) {
    AneFfnPrepared prepared;
    prepared.chunks = timedChunks(21.0);
    return prepared;
  };
  const auto fails = [](uint32_t, AneFfnTiming) -> AneFfnPrepared { throw std::runtime_error("an evaluation failed"); };
  const uint32_t most = Fake{}.largestSplitContext();
  require(most && most < contextWith(Fake{}.bytes(1)), "the largest split holds no less context");
  struct Case {
    Fake fake;
    AneFfnSetting setting;
    Kind kind;
    uint32_t context;
  };
  std::vector<Case> cases(6);
  cases[1].fake.time = noGain;
  cases[2].fake.prepare = slowChunks;
  cases[3].fake.time = []() -> Timings { throw std::runtime_error("calibration timings are not usable"); };
  cases[4].fake.prepare = fails;
  cases[5].setting = {.given = AneFfnSetting::Given{0.2}};
  const auto split = [](uint32_t aneUnits) { return contextWith(Fake{}.bytes(aneUnits)); };
  const std::array<std::pair<Kind, uint32_t>, 6> expected{{{Kind::Split, split(8)},
                                                           {Kind::NoGain, contextWith(0)},
                                                           {Kind::NoGain, contextWith(0)},
                                                           {Kind::Unavailable, most},
                                                           {Kind::Unavailable, most},
                                                           {Kind::Split, split(7)}}};
  for (size_t index = 0; index < cases.size(); ++index) {
    Case &test = cases[index];
    const auto &[kind, context] = expected[index];
    const Started started = start(test.fake, test.setting);
    const AneFfnOutcome &outcome = started.start.outcome;
    const bool named = has(started.log, "; context " + grouped(context) + " tokens (" + grouped(contextWith(0)) +
                                            " with --disable-ane)");
    require(outcome.kind == kind && outcome.context == context && outcome.contextWithout == contextWith(0) &&
                (!started.start.plan || started.start.plan->maximumContextTokens() >= context) &&
                (kind == Kind::Split) == started.start.plan.has_value() && named == (context != contextWith(0)),
            "start " + std::to_string(index) + " did not serve its calibration's automatic context: " + started.log);
  }
  // With --max-context the start holds the context asked for, which the log does not name.
  Fake fake;
  const Started started = start(fake, {}, 4096);
  require(started.start.outcome.kind == Kind::Split && started.start.plan->maximumContextTokens() >= 4096 &&
              !has(started.log, "with --disable-ane"),
          "a start given --max-context named the automatic context");
}

// With --max-context the split takes at most the units whose plan still holds it.
void testExplicitContext() {
  Fake fake;
  const uint32_t context = contextWith(fake.bytes(5));
  const Started started = start(fake, {}, context);
  require(started.start.outcome.kind == Kind::Split && fake.prepared == std::pair(5u, AneFfnTiming::Chunks) && started.start.plan &&
              started.start.plan->maximumContextTokens() >= context && !has(started.log, "with --disable-ane"),
          "the split took more units than the plan holds with the context asked for");
}

// A failure of the timings or of the prepared split leaves the GPU alone and the plan as it was, and warns;
// cancellation, an interrupted wait and a backend that stopped serving end the start.
void testFailures() {
  Fake fake;
  fake.time = []() -> Timings { throw std::runtime_error("calibration timings are not usable"); };
  Started started = start(fake);
  require(started.start.outcome.kind == Kind::Unavailable && !started.start.plan && !started.start.split &&
              !fake.prepared,
          "a timing that failed did not leave the GPU alone");
  require(has(started.log, "Warning · Neural Engine FFN split unavailable (calibration timings are not usable); the "
                           "GPU runs the prefill FFN alone; context ") &&
              has(started.log, " with --disable-ane). --disable-ane silences this line."),
          "a timing that failed did not warn: " + started.log);
  fake = {};
  fake.prepare = [](uint32_t, AneFfnTiming) -> AneFfnPrepared {
    throw std::runtime_error("ANE FFN split of 640 rows failed (the Neural Engine's output or its scales were not "
                             "finite)");
  };
  started = start(fake);
  require(started.start.outcome.kind == Kind::Unavailable && !started.start.plan &&
              has(started.log, "unavailable (ANE FFN split of 640 rows failed"),
          "a split that failed verify() was not unavailable");
  // Chunks that were not timed are not read.
  fake.prepare = [](uint32_t, AneFfnTiming) { return AneFfnPrepared{}; };
  started = start(fake);
  require(started.start.outcome.kind == Kind::Unavailable && !started.start.plan &&
              has(started.log, "unavailable (the split's chunks were not timed)"),
          "a split of no chunks timed was not unavailable: " + started.log);

  bool cancelled = false;
  fake.prepare = [&](uint32_t, AneFfnTiming) -> AneFfnPrepared {
    cancelled = true;
    throw std::runtime_error("the Metal command was not submitted");
  };
  rejects([&] { static_cast<void>(start(fake, {}, 0, [&] { return cancelled; })); }, "not submitted",
          "a cancelled start went on");
  fake.prepare = [](uint32_t, AneFfnTiming) -> AneFfnPrepared { throw ane::Interrupted("the wait was interrupted"); };
  rejects([&] { static_cast<void>(start(fake)); }, "interrupted", "an interrupted wait went on");
  fake.prepare = [&](uint32_t, AneFfnTiming) -> AneFfnPrepared {
    fake.healthy = false;
    throw std::runtime_error("the backend stopped serving");
  };
  rejects([&] { static_cast<void>(start(fake)); }, "stopped serving", "a start went on on a backend that stopped");
}

// No share predicted to gain enough, or no chunk timed to, leaves the GPU alone.
void testNoGain() {
  Fake fake;
  fake.time = [] {
    Timings timings = measured();
    timings.gpuAlone = 14.5;
    return timings;
  };
  Started started = start(fake);
  require(started.start.outcome.kind == Kind::NoGain && !fake.prepared && !started.start.plan &&
              has(started.log, "The GPU runs the prefill FFN alone: no Neural Engine split beats its 14.5 ms per "
                               "2048-row FFN layer by enough (calibrated in "),
          "a share predicted to gain too little split: " + started.log);
  fake = {};
  fake.prepare = [](uint32_t, AneFfnTiming) {
    AneFfnPrepared prepared;
    prepared.chunks = timedChunks(21.0);
    return prepared;
  };
  started = start(fake);
  require(started.start.outcome.kind == Kind::NoGain && !started.start.plan &&
              has(started.log, "no chunk of the Neural Engine split beats it by enough, 21.0 ms per 2048-row FFN "
                               "layer against 20.0 (calibrated in "),
          "a split of no chunk that gains split: " + started.log);
}

// A start takes this Mac's calibration, once remembered, while a check of it matches: a second start of the model
// prepares the split the first chose, checked, times nothing else, and serves it over the same chunks at the same
// context. A check whose GPU alone or split layer moved more than 5% either way, or whose Neural Engine ran more than
// 10% faster, calibrates again and remembers what it finds; one whose Neural Engine runs slower, or that moved less,
// takes the calibration.
void testRemembered() {
  Fake fake;
  const Started first = start(fake);
  require(first.start.outcome.kind == Kind::Split && fake.timings == 1 && fake.memory.size() == 1,
          "the first start did not remember its calibration");
  const Started second = start(fake);
  require(second.start.outcome.kind == Kind::Split && fake.timings == 1 &&
              fake.prepared == std::pair(8u, AneFfnTiming::Check) &&
              second.start.outcome.context == first.start.outcome.context &&
              has(second.log, "Neural Engine FFN split at share 0.24 for chunks of 553 rows or more: 15.5 ms per "
                              "2048-row FFN layer against 20.0 on the GPU alone, 2.5% from the GPU alone on the "
                              "Neural Engine's part (checked against this Mac's calibration in "),
          "a second start did not take the calibration it checked: " + second.log);
  struct Drift {
    Check check;
    std::string_view said;
  };
  for (const Drift &drift :
       {Drift{{21.1, 15.5, 12.0}, "the GPU alone's full-chunk layer took 21.1 ms against 20.0"},
        Drift{{20.0, 14.7, 12.0}, "the split's full-chunk layer took 14.7 ms against 15.5"},
        Drift{{20.0, 15.5, 10.7}, "the Neural Engine's part of the split's full-chunk layer took 10.7 ms against 12.0"}}) {
    Fake drifted;
    static_cast<void>(start(drifted));
    drifted.check = drift.check;
    const Started again = start(drifted);
    require(again.start.outcome.kind == Kind::Split && drifted.timings == 2 &&
                drifted.prepared == std::pair(8u, AneFfnTiming::Chunks) && drifted.memory.size() == 1 &&
                has(again.log, "Calibrating the Neural Engine FFN split again: " + std::string(drift.said) +
                                   " when this Mac calibrated it.") &&
                has(again.log, "(calibrated in "),
            "a check that drifted did not calibrate again: " + again.log);
  }
  for (const Check &close : {Check{20.9, 15.5, 12.0}, Check{20.0, 16.2, 12.0}, Check{20.0, 15.5, 14.0}}) {
    Fake kept;
    static_cast<void>(start(kept));
    kept.check = close;
    const Started again = start(kept);
    require(again.start.outcome.kind == Kind::Split && kept.timings == 1 && has(again.log, "(checked against "),
            "a check within the drift did not take the calibration: " + again.log);
  }
}

// A start remembers under the most units its plan holds, and takes nothing remembered under others; a calibration
// that finds no gain forgets what was remembered, and a given share neither takes nor remembers one. A record of more
// units than the plan holds is not taken.
void testForgotten() {
  Fake fake;
  static_cast<void>(start(fake));
  const uint32_t context = contextWith(fake.bytes(5));
  const Started fewer = start(fake, {}, context);
  require(fewer.start.outcome.kind == Kind::Split && fake.timings == 2 &&
              fake.prepared == std::pair(5u, AneFfnTiming::Chunks) && fake.memory.size() == 2,
          "a start of fewer units took a calibration of more: " + fewer.log);
  fake.memory.erase(fake.memory.begin(), std::next(fake.memory.begin()));
  fake.check = Check{25.0, 15.5, 12.0};
  fake.time = [] {
    Timings timings = measured();
    timings.gpuAlone = 14.5;
    return timings;
  };
  const Started none = start(fake);
  require(none.start.outcome.kind == Kind::NoGain && fake.memory.empty(),
          "a calibration that found no gain did not forget the one remembered: " + none.log);

  Fake given;
  static_cast<void>(start(given));
  const std::map<uint32_t, std::string> remembered = given.memory;
  const Started set = start(given, {.given = AneFfnSetting::Given{0.2}});
  require(set.start.outcome.kind == Kind::Split && given.prepared == std::pair(7u, AneFfnTiming::None) &&
              given.memory == remembered,
          "a given share took or changed this Mac's calibration: " + set.log);

  // A record of more units than the plan holds, which the fake remembers under fewer.
  Fake more;
  static_cast<void>(start(more));
  const uint32_t five = contextWith(more.bytes(5));
  more.memory[5] = more.memory.begin()->second;
  const Started held = start(more, {}, five);
  require(held.start.outcome.kind == Kind::Split && more.timings == 2 &&
              more.prepared == std::pair(5u, AneFfnTiming::Chunks),
          "a start took a calibration of more units than its plan holds: " + held.log);
}

// A given share runs the units nearest it, untimed, over the chunks given; a given split of no units or of rows no
// function holds is refused.
void testGivenShare() {
  Fake fake;
  Started started = start(fake, {.given = AneFfnSetting::Given{0.2}});
  require(started.start.outcome.kind == Kind::Split && !fake.timings && fake.prepared == std::pair(7u, AneFfnTiming::None) &&
              started.start.plan &&
              started.start.plan->breakdown().aneFfnBytes == fake.bytes(7) &&
              has(started.log, "Neural Engine FFN split at the given share 0.21 for chunks of 512 rows or more, 2.5% "
                               "from the GPU alone on the Neural Engine's part"),
          "the given share did not run as given: " + started.log);
  started = start(fake, {.given = AneFfnSetting::Given{0.2, 832}});
  require(started.start.outcome.kind == Kind::Split && has(started.log, "for chunks of 832 rows or more"),
          "the given least chunk did not run as given: " + started.log);
  for (const AneFfnSetting::Given given : {AneFfnSetting::Given{0.0}, AneFfnSetting::Given{1.0},
                                           AneFfnSetting::Given{0.2, 511}, AneFfnSetting::Given{0.2, 2049}})
    rejects([&] { static_cast<void>(start(fake, {.given = given})); }, "ANE FFN",
            "a given split of share " + std::to_string(given.share) + " from " + std::to_string(given.minimumRows) +
                " rows ran");
  // A dev tool's options: none calibrates, a share of 0 runs the GPU alone, and least rows need a share.
  const AneFfnSetting calibrated = AneFfnSetting::fromGiven(std::nullopt, std::nullopt),
                      off = AneFfnSetting::fromGiven(0.0, 832), given = AneFfnSetting::fromGiven(0.3, 832);
  require(calibrated.enabled && !calibrated.given && !off.enabled && given.enabled && given.given &&
              given.given->share == 0.3 && given.given->minimumRows == 832 &&
              AneFfnSetting::fromGiven(0.3, std::nullopt).given->minimumRows == ops::AneFfn::kMinimumRows,
          "a dev tool's options did not give their setting");
  rejects([] { static_cast<void>(AneFfnSetting::fromGiven(std::nullopt, 832)); }, "without a share",
          "least rows without a share were taken");
}

} // namespace

int main() {
  try {
    testOff();
    testUnsupported();
    testRefusedContext();
    testRefusedAutomatic();
    testAutomaticContext();
    testServedContext();
    testExplicitContext();
    testFailures();
    testNoGain();
    testRemembered();
    testForgotten();
    testGivenShare();
    std::cout << "ane ffn startup tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ane ffn startup tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
