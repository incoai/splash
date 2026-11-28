// What a start makes of the prefill FFN's Neural Engine split (engine/AneFfnStartup.cpp) on a model the test plays:
// each outcome, the memory plan it adopts, the automatic context and the context a relaunch keeps, the split it leaves
// pending until the engine has prefilled a full chunk, the calibration finishAneFfn makes from that prefill or takes
// again, the line each logs and what a start a command of the split ends reports.

#include "Supervision.hpp"
#include "TestChecks.hpp"
#include "TestModel.hpp"
#include "TestStderr.hpp"
#include "ane/Program.hpp"
#include "engine/AneFfnStartup.hpp"
#include "engine/Bootstrap.hpp"
#include "metal/MetalBackend.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
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
using splash::ops::AneFfn;
using splash::ops::ane_ffn::Calibration;
using splash::ops::ane_ffn::ChunkTimings;
using splash::ops::ane_ffn::Timing;
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

bool near(double value, double expected) { return std::abs(value - expected) < 1e-9; }

// The mixer a calibration remembers beside its own GPU alone's full chunk, its chunks' 20 ms (timedChunks), as a
// later start reads its own beside it, of a prefill whose mixer beside time()'s 18.99 ms is `mixer`.
double remembered(double mixer) { return 18.990 + mixer - 20.0; }

// An M5 Max's timings of Qwen3.8-27B's 34 units, which choose 8 of them in one step. Its half step at the high share
// (two steps' H) is set at 70 ms, three times the M5 Max's 22.3, so that two steps gain only beside a mixer of more
// than about 2 ms: none beside the 2 ms a Fake's prefill gives by default, 4 units beside 10 ms.
Timings measured() { return {{14.0 / 34, 11.699, 23.484}, {27.0 / 34, 5.033, 44.683}, 18.990, 70.0}; }
// The same with the M5 Max's own half step, 22.3 ms, whose two steps beside a mixer of 10 ms choose 13 units.
Timings pipelined() {
  Timings timings = measured();
  timings.highHalf = 22.3415;
  return timings;
}
// Other M5 Max timings, which choose 6 units and, refitted on a full chunk of 13.84 ms, 7 (ane_ffn_calibration_test);
// with the half step `highHalf`, by default slow enough that two steps gain nothing beside a mixer of 2 ms (their own
// 23.0 ms chooses 11 units of two steps beside one of 10 ms).
Timings refitting(double highHalf = 70.0) {
  return {{14.0 / 34, 9.9673, 23.929}, {27.0 / 34, 4.7526, 46.034}, 15.834, highHalf};
}
// An M5 Pro's timings, which choose 15 units.
Timings m5Pro() { return {{14.0 / 34, 24.244, 26.277}, {27.0 / 34, 10.147, 45.679}, 38.829, 22.8395}; }

// The chunks of a split timed over its one-step functions of `top` rows down to 512, as Measurement::chunks times
// them: of 2048 rows for a split of one share (13), of 1024 for one of both shares (5), whose function of 1024 rows
// holds the chunks of 897 to 1024. The GPU alone takes 5 ms at 512 rows and 20 at 2048, each function's split layer
// 0.8 of the GPU's at its rows but function `top`'s at `most` ms and function 640's at 5.5, which splits its chunks
// from 553 rows (ane_ffn_calibration_test).
ChunkTimings timedChunks(double most, uint32_t top = AneFfn::kMaximumRows) {
  const auto gpu = [](uint32_t rows) { return 5.0 + (rows - 512) * 15.0 / 1536; };
  ChunkTimings chunks;
  chunks.gpu = {{{512, gpu(512)}, {2048, gpu(2048)}}};
  for (uint32_t rows = top; rows >= 512; rows -= 128)
    chunks.functions.push_back({rows == 512 ? 512 : rows - 127, rows,
                                rows == top ? most : rows == 640 ? 5.5 : 0.8 * gpu(rows)});
  return chunks;
}

// A split a start prepared: its shares, its rows and whether its chunks were timed.
struct Preparation {
  uint32_t oneStep = 0, twoSteps = 0, oneStepRows = 0, twoStepRows = 0;
  bool timed = false;
  bool operator==(const Preparation &) const = default;
};
std::string text(const std::vector<Preparation> &preparations) {
  std::string result;
  for (const Preparation &preparation : preparations)
    result += "{" + std::to_string(preparation.oneStep) + ", " + std::to_string(preparation.twoSteps) + ", " +
              std::to_string(preparation.oneStepRows) + ", " + std::to_string(preparation.twoStepRows) +
              (preparation.timed ? ", timed} " : ", untimed} ");
  return result;
}

// A model the test plays: a dense one of 34 units and `layers` FFN layers, each unit `unitBytes` of the split's
// memory, what its timings and its prepared split return, the rest of the layer of its prefill on the GPU alone
// (`mixer`, prefillMilliseconds), and what the start and its finish asked of it.
struct Fake {
  bool dense = true;
  std::string unsupported;
  std::optional<std::string> unavailable;
  uint64_t unitBytes = 256 * kMiB;
  uint32_t layers = 64;
  double mixer = 2.0;
  std::function<Timings()> time = measured;
  // The split's full chunk at the units calibration first chooses, timed on the layers time() timed: by default as
  // the first layers' chunks below time it.
  std::function<double(uint32_t)> timeAt = [](uint32_t) { return 15.5; };
  // By default the chunks of the functions the split's program holds: one share's to 2048 rows, both shares' to 1024.
  std::function<AneFfnPrepared(AneFfn::Shares, bool)> prepare = [](AneFfn::Shares shares, bool timed) {
    AneFfnPrepared prepared;
    if (timed) prepared.chunks = shares.twoSteps ? timedChunks(8.0, AneFfn::kTwoStepRows) : timedChunks(15.5);
    prepared.error = 0.025;
    return prepared;
  };
  // This Mac's calibration as the model keeps it, whatever the most units; the most units the start recalled one
  // for, and the calibration it remembered and under which most units.
  std::optional<Calibration> stored, remembered;
  std::optional<uint32_t> recalled, rememberedMost;
  // Whether remember() keeps what it is given, as a cache directory that takes writes does.
  bool keeps = true;
  bool healthy = true;
  // The calls the start made: of unavailable() and time(), the units of timeAt()'s, and of prepare() the one-step
  // units and whether timed of the last, the one-step units of each, and each in full.
  uint32_t asked = 0, timings = 0;
  std::vector<uint32_t> timedUnits;
  std::optional<std::pair<uint32_t, bool>> prepared;
  std::vector<uint32_t> preparedUnits;
  std::vector<Preparation> preparations;

  [[nodiscard]] uint64_t bytes(uint32_t aneUnits) const { return aneUnits * unitBytes; }
  // The largest split the model could take, the most units whose plan holds any context, and what that plan holds:
  // the context every start that may split serves.
  [[nodiscard]] uint32_t largestUnits() const {
    for (uint32_t aneUnits = 33; aneUnits > 0; --aneUnits)
      if (contextWith(bytes(aneUnits))) return aneUnits;
    return 0;
  }
  [[nodiscard]] uint32_t largestSplitContext() const { return contextWith(bytes(largestUnits())); }

  AneFfnModel model() {
    AneFfnModel result;
    result.dense = dense;
    result.unsupported = unsupported;
    result.unavailable = [this] {
      ++asked;
      return unavailable;
    };
    result.units = 34;
    result.layers = layers;
    result.plannedBytes = [this](uint32_t aneUnits) { return bytes(aneUnits); };
    result.time = [this] {
      ++timings;
      Timings timed = time();
      // The model's timings carry no mixer: finishAneFfn takes it from the prefill.
      timed.mixer = 0.0;
      return timed;
    };
    // Beside the same GPU alone as time()'s high share.
    result.timeAt = [this](uint32_t aneUnits) {
      timedUnits.push_back(aneUnits);
      return Timing{aneUnits / 34.0, 0.0, timeAt(aneUnits), time().gpuAlone};
    };
    result.prepare = [this](AneFfn::Shares shares, AneFfn::Rows rows, bool timeChunks) {
      prepared = {shares.oneStep, timeChunks};
      preparedUnits.push_back(shares.oneStep);
      preparations.push_back({shares.oneStep, shares.twoSteps, rows.oneStep, rows.twoSteps, timeChunks});
      return prepare(shares, timeChunks);
    };
    result.recall = [this](uint32_t maxAneUnits) {
      recalled = maxAneUnits;
      return stored;
    };
    result.remember = [this](uint32_t maxAneUnits, const Calibration &calibration) {
      if (!keeps) return false;
      rememberedMost = maxAneUnits;
      stored = remembered = calibration;
      return true;
    };
    result.healthy = [this] { return healthy; };
    return result;
  }
};

// The engine's prefill of a full chunk on the GPU alone that `start` waited for, in milliseconds: each of the
// model's layers its FFN and the model's mixer beside it. The FFN is the GPU alone's full chunk as the calibration the
// start recalled timed it, so that the prefill gives the model's mixer beside it (a recalled calibration whose mixer
// reads more than a tenth longer calibrates again); else as time() times it, none where time() fails, as the
// finish's own timing then does.
double prefillMilliseconds(Fake &fake, const AneFfnStart &start) {
  if (start.calibration && start.calibration->aneUnits)
    return (start.calibration->gpu[1].milliseconds + fake.mixer) * fake.layers;
  try {
    return (fake.time().gpuAlone + fake.mixer) * fake.layers;
  } catch (const std::exception &) {
    return 0.0;
  }
}

struct Started {
  AneFfnStart start;
  std::string log;
};
Started start(Fake &fake, AneFfnSetting setting = {}, RequestedContext context = {},
              std::function<bool()> cancelled = {}) {
  Started result;
  result.log = test::capturedStderr(
      [&] { result.start = startAneFfn(fake.model(), setting, context, planMemory, cancelled); });
  return result;
}

struct Finished {
  AneFfnFinished finished;
  std::string log;
};
Finished finish(Fake &fake, const Started &started, double prefill, std::function<bool()> cancelled = {}) {
  Finished result;
  result.log = test::capturedStderr(
      [&] { result.finished = finishAneFfn(fake.model(), started.start, prefill, cancelled); });
  return result;
}

// A start, and its finish where the start left the split pending, after the prefill the model gives
// (prefillMilliseconds): the outcome is the finish's where there is one, the log both.
struct Ran {
  Started started;
  std::optional<Finished> finished;
  std::string log;
  [[nodiscard]] const AneFfnOutcome &outcome() const {
    return finished ? finished->finished.outcome : started.start.outcome;
  }
  [[nodiscard]] bool split() const {
    return finished ? finished->finished.split != nullptr : started.start.split != nullptr;
  }
};
Ran run(Fake &fake, AneFfnSetting setting = {}, RequestedContext context = {},
        std::function<bool()> cancelled = {}) {
  Ran result;
  result.started = start(fake, setting, context, cancelled);
  result.log = result.started.log;
  if (result.started.start.outcome.kind == Kind::Pending) {
    result.finished = finish(fake, result.started, prefillMilliseconds(fake, result.started.start), cancelled);
    result.log += result.finished->log;
  }
  return result;
}

bool has(const std::string &log, std::string_view text) { return log.find(text) != std::string::npos; }
// Whether `log` is one line that holds `text`.
bool line(const std::string &log, std::string_view text) {
  return has(log, text) && log.find('\n') == log.size() - 1;
}
// Whether the first line of `log` says that the split is being set up.
bool settingUpFirst(const std::string &log) { return log.find(kSettingUp) < log.find('\n'); }
bool nothingStarted(const Fake &fake, const Started &started) {
  return !fake.timings && !fake.prepared && !fake.recalled && !fake.remembered && !started.start.split &&
         !started.start.plan && !has(started.log, kSettingUp);
}
// Whether `started` left the split pending within `maxAneUnits`, having done nothing but recall: nothing timed,
// prepared, remembered or logged, and the plan of those units adopted, whose context it serves.
bool pending(const Fake &fake, const Started &started, uint32_t maxAneUnits) {
  const AneFfnStart &start = started.start;
  return start.outcome.kind == Kind::Pending && !fake.timings && !fake.prepared && !fake.remembered &&
         fake.recalled == maxAneUnits && started.log.empty() && !start.split && start.maxAneUnits == maxAneUnits &&
         start.plan && start.plan->breakdown().aneFfnBytes == fake.bytes(maxAneUnits) &&
         start.outcome.context == start.plan->maximumContextTokens() &&
         start.outcome.context == contextWith(fake.bytes(maxAneUnits)) &&
         start.outcome.contextSplitBytes == fake.bytes(maxAneUnits) && start.outcome.contextWithout == contextWith(0);
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
  Started started = start(fake, {}, {most + 1});
  require(started.start.outcome.kind == Kind::Refused && started.start.outcome.context == most &&
              started.start.outcome.contextWithout == without && nothingStarted(fake, started),
          "a context no split leaves did not refuse the split");
  require(line(started.log, "Warning · Neural Engine FFN split off: --max-context " + std::to_string(most + 1) +
                                " cannot be held with it, and any split leaves at most ") &&
              has(started.log, "; pass --max-context " + std::to_string(most) +
                                   " or less to run it, or --disable-ane, which also silences this line."),
          "the refusal did not warn with the way out: " + started.log);
  started = start(fake, {}, {without + 1});
  require(started.start.outcome.kind == Kind::Refused && nothingStarted(fake, started) && started.log.empty(),
          "a context the GPU alone does not hold was warned of");
  // A given share is refused on its own units.
  const uint32_t given = contextWith(fake.bytes(7));
  started = start(fake, {.given = AneFfnSetting::Given{0.2}}, {given + 1});
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

// Whether the plan refuses the split is known before anything is set up: a start finds it refused where even the
// least share leaves less than the context asked for, or with none asked no context. Memory the plan sets aside for
// something else can refuse a split the plan without it holds, which a start weighs before it decides the split (the
// SSD cache's automatic staging gives way to the split). A split that does not run is never refused.
void testRefusalIsArithmetic() {
  Fake fake;
  // The plan with `other` bytes set aside beside the split's.
  const auto beside = [](uint64_t other) { return [other](uint64_t bytes) { return planMemory(bytes + other); }; };
  const uint32_t least = contextWith(fake.bytes(1)), lessBeside = contextWith(fake.bytes(1) + kGiB);
  require(!aneFfnRefused(fake.model(), {}, least, planMemory) &&
              aneFfnRefused(fake.model(), {}, least + 1, planMemory) && lessBeside < least &&
              aneFfnRefused(fake.model(), {}, least, beside(kGiB)) &&
              !aneFfnRefused(fake.model(), {}, lessBeside, beside(kGiB)),
          "the refusal of a context did not follow what the least split leaves");
  const AneFfnSetting given{.given = AneFfnSetting::Given{0.2}};
  const uint32_t givenLeaves = contextWith(fake.bytes(7));
  require(!aneFfnRefused(fake.model(), given, givenLeaves, planMemory) &&
              aneFfnRefused(fake.model(), given, givenLeaves + 1, planMemory),
          "a given share was not refused on its own units");
  // Without --max-context: a least share that leaves some context, but none beside a GiB more.
  uint64_t holds = 0, none = 16 * kGiB;
  while (none - holds > kMiB) {
    const uint64_t middle = holds + (none - holds) / 2;
    if (contextWith(middle))
      holds = middle;
    else
      none = middle;
  }
  fake.unitBytes = holds;
  require(!aneFfnRefused(fake.model(), {}, 0, planMemory) && aneFfnRefused(fake.model(), {}, 0, beside(kGiB)),
          "a split that leaves no context beside other memory was not refused, or one that leaves some was");
  require(!fake.timings && !fake.prepared && !fake.recalled && !fake.remembered,
          "weighing the refusal set something up");
  // --disable-ane, a MoE target, layers the split does not take, and a Mac without the interface.
  Fake moe, unsupported, unavailable;
  moe.dense = false;
  unsupported.unsupported = "ANE FFN split needs unrotated quantized projections";
  unavailable.unavailable = "AppleNeuralEngine lacks _ANEClient";
  require(!aneFfnRefused(fake.model(), {.enabled = false}, 0, beside(16 * kGiB)) &&
              !aneFfnRefused(moe.model(), {}, 0, beside(16 * kGiB)) &&
              !aneFfnRefused(unsupported.model(), {}, 0, beside(16 * kGiB)) &&
              !aneFfnRefused(unavailable.model(), {}, 0, beside(16 * kGiB)),
          "a split that does not run was refused");
}

// A start that may split only decides: it times, prepares, remembers and logs nothing, and leaves the split pending
// with the plan of the largest split the model could take, whose context it serves, the units that plan holds, this
// Mac's calibration as recalled within them and the context asked for, which finishAneFfn takes. A remembered split
// of fewer units, of either kind of command, and a record of no split the Neural Engine refused once, which calibrates
// again, serve the same context and carry the record. It planned the remembered split's units, and set the split up.
void testPending() {
  const std::array<ChunkTimings::Gpu, 2> gpu{{{512, 5.0}, {2048, 20.0}}};
  Fake fake;
  const uint32_t most = fake.largestUnits();
  require(most > 13 && fake.largestSplitContext() < contextWith(fake.bytes(13)),
          "the largest split holds no less context than a calibrated one");
  Started started = start(fake);
  require(pending(fake, started, most) && !started.start.calibration && !started.start.requested.tokens &&
              started.start.outcome.context == fake.largestSplitContext() &&
              started.start.outcome.reason == "set up once the engine has prefilled a full chunk" &&
              aneFfnOutcomeName(Kind::Pending) == "pending",
          "a start that may split did more than decide: " + started.log);
  for (const Calibration &record :
       {Calibration{9, 700, gpu}, Calibration{9, 700, gpu, 0, 13, 2.0}, Calibration{.refusals = 1}}) {
    fake = {};
    fake.stored = record;
    started = start(fake);
    require(pending(fake, started, most) && started.start.calibration &&
                started.start.calibration->text() == record.text() &&
                started.start.outcome.context == fake.largestSplitContext(),
            "a start that recalled " + record.text() + " did not leave it pending at the largest split's context");
  }
  // With --max-context: the most units whose plan holds it, and the context asked for, which the log weighs.
  fake = {};
  const uint32_t context = contextWith(fake.bytes(5));
  started = start(fake, {}, {context});
  require(pending(fake, started, 5) && started.start.requested.tokens == context && !started.start.requested.relaunch &&
              started.start.plan->maximumContextTokens() >= context,
          "a start given --max-context did not leave the split of the units it holds pending");
  fake = {};
  started = start(fake, {}, {context, true});
  require(pending(fake, started, 5) && started.start.requested.tokens == context && started.start.requested.relaunch,
          "a relaunch's start did not carry the context it was asked to leave");
}

// finishAneFfn sets up only a split its start left pending: a start that ran the GPU alone, was refused, set a given
// split up or took a record of no split is not finished again, and nothing more is timed or prepared.
void testFinishingAStartNotPendingThrows() {
  struct Case {
    std::string name;
    Fake fake;
    AneFfnSetting setting;
    RequestedContext context;
  };
  std::vector<Case> cases(5);
  cases[0].name = "--disable-ane";
  cases[0].setting = {.enabled = false};
  cases[1].name = "a MoE model";
  cases[1].fake.dense = false;
  cases[2].name = "a refused context";
  cases[2].context = {contextWith(0) - 1};
  cases[3].name = "a given split";
  cases[3].setting = {.given = AneFfnSetting::Given{0.2}};
  cases[4].name = "a record of two refusals";
  cases[4].fake.stored = Calibration{.refusals = Calibration::kRefusals};
  for (Case &test : cases) {
    const Started started = start(test.fake, test.setting, test.context);
    const size_t preparations = test.fake.preparations.size();
    require(started.start.outcome.kind != Kind::Pending, test.name + " left the split pending");
    rejects([&] { static_cast<void>(finish(test.fake, started, prefillMilliseconds(test.fake, started.start))); },
            "was not left to set up", "the finish of " + test.name + " went on");
    require(!test.fake.timings && test.fake.preparations.size() == preparations,
            "the finish of " + test.name + " timed or prepared a split");
  }
}

// Without --max-context the split takes its memory from the KV cache: the start adopts the plan of the largest split
// the model could take, and its finish calibrates one step's 8 units, prepared as one share, its chunks timed, and
// remembers them, under the units the start recalled for; the engine serves the context the plan holds, which the
// finish's log names beside the GPU's alone after the line that it sets the split up.
void testAutomaticContext() {
  Fake fake;
  const Ran ran = run(fake);
  const AneFfnOutcome &outcome = ran.outcome();
  const uint32_t most = fake.largestUnits();
  require(outcome.kind == Kind::Split && fake.timings == 1 && fake.prepared == std::pair(8u, true) &&
              fake.preparations == std::vector<Preparation>{{8, 0, 512, 0, true}} && fake.remembered &&
              fake.remembered->aneUnits == 8 && fake.remembered->minimumRows == 553 &&
              !fake.remembered->twoStepUnits && fake.remembered->gpu[1].milliseconds == 20.0 &&
              fake.recalled == most && fake.rememberedMost == most,
          "the split of 8 units was not prepared and remembered: " + text(fake.preparations));
  const AneFfnStart &start = ran.started.start;
  require(start.plan && start.plan->breakdown().aneFfnBytes == fake.bytes(most) &&
              start.plan->maximumContextTokens() == contextWith(fake.bytes(most)) &&
              outcome.context == contextWith(fake.bytes(most)) && outcome.context < contextWith(0) &&
              outcome.contextWithout == contextWith(0) && outcome.contextSplitBytes == fake.bytes(most),
          "the plan of the largest split was not adopted at the automatic context");
  const std::string &log = ran.finished->log;
  const size_t split = log.find("Neural Engine FFN split at share 0.24 for chunks of 553 rows or more: 15.5 ms per "
                                "2048-row FFN layer against 20.0 on the GPU alone, 2.5% from the GPU alone on the "
                                "Neural Engine's part (calibrated in ");
  require(ran.started.log.empty() && settingUpFirst(log) && split != std::string::npos &&
              !has(log, "two steps") &&
              has(log, "; context " + grouped(outcome.context) + " tokens (" + grouped(contextWith(0)) +
                           " with --disable-ane)."),
          "the split was not set up and logged with both contexts: " + ran.log);
}

// Every start of the model on a Mac that may split serves the context of the largest split the model could take,
// whatever it or its finish makes of the split: calibrated, set up as remembered, gaining too little, or failing. A
// start after a calibration that found no gain, or after the Neural Engine refused the split twice, serves the GPU
// alone's, and a given share its own units'. The start's plan holds it, the outcome carries the split bytes it rests
// on, and the log names it beside the GPU alone's.
void testDeterministicContext() {
  const auto noGain = [] {
    Timings timings = measured();
    timings.gpuAlone = 14.0;
    return timings;
  };
  const auto slowChunks = [](AneFfn::Shares, bool) {
    AneFfnPrepared prepared;
    prepared.chunks = timedChunks(21.0);
    return prepared;
  };
  const auto fails = [](AneFfn::Shares, bool) -> AneFfnPrepared { throw std::runtime_error("an evaluation failed"); };
  const Calibration nine{9, 700, {{{512, 5.0}, {2048, 20.0}}}};
  const uint32_t most = Fake{}.largestSplitContext();
  require(most && most < contextWith(Fake{}.bytes(1)), "the largest split holds no less context");
  struct Case {
    Fake fake;
    AneFfnSetting setting;
  };
  std::vector<Case> cases(9);
  cases[1].fake.stored = nine;
  cases[2].fake.time = noGain;
  cases[3].fake.prepare = slowChunks;
  cases[4].fake.time = []() -> Timings { throw std::runtime_error("calibration timings are not usable"); };
  cases[5].fake.prepare = fails;
  cases[6].setting = {.given = AneFfnSetting::Given{0.2}};
  cases[7].fake.stored = nine;
  cases[7].fake.prepare = fails;
  cases[8].fake.stored = Calibration{.refusals = Calibration::kRefusals};
  const auto split = [](uint32_t aneUnits) { return contextWith(Fake{}.bytes(aneUnits)); };
  const std::array<std::pair<Kind, uint32_t>, 9> expected{{{Kind::Split, most},
                                                            {Kind::Split, most},
                                                            {Kind::NoGain, most},
                                                            {Kind::NoGain, most},
                                                            {Kind::Unavailable, most},
                                                            {Kind::Unavailable, most},
                                                            {Kind::Split, split(7)},
                                                            {Kind::Unavailable, most},
                                                            {Kind::Unavailable, contextWith(0)}}};
  for (size_t index = 0; index < cases.size(); ++index) {
    Case &test = cases[index];
    const auto &[kind, context] = expected[index];
    const Ran ran = run(test.fake, test.setting);
    const AneFfnOutcome &outcome = ran.outcome();
    const AneFfnStart &start = ran.started.start;
    const bool named = has(ran.log, "; context " + grouped(context) + " tokens (" + grouped(contextWith(0)) +
                                        " with --disable-ane)");
    require(outcome.kind == kind && outcome.context == context && outcome.contextWithout == contextWith(0) &&
                start.outcome.context == context && contextWith(outcome.contextSplitBytes) == context &&
                (!start.plan || start.plan->maximumContextTokens() == context) &&
                start.plan.has_value() == (context != contextWith(0)) && named == (context != contextWith(0)),
            "start " + std::to_string(index) + " did not serve the largest split's context: " + ran.log);
  }
  // The start after one whose calibration found no gain calibrates again, at the largest split's context.
  Fake &found = cases[2].fake;
  const Ran after = run(found);
  require(after.outcome().kind == Kind::NoGain && after.finished && found.timings == 2 && !found.remembered &&
              after.outcome().context == most && after.started.start.plan,
          "the start after a calibration of no gain did not calibrate again at the split's context: " + after.log);
  // A start that takes the calibration a start remembered serves the context that start served.
  Fake fake;
  const uint32_t first = run(fake).outcome().context;
  const Ran again = run(fake);
  require(fake.prepared == std::pair(8u, false) && again.outcome().context == first,
          "a start that took the remembered calibration served another context");
  // With --max-context the start holds the context asked for, which the log does not name.
  fake = {};
  const Ran ran = run(fake, {}, {4096});
  require(ran.outcome().kind == Kind::Split && ran.started.start.plan->maximumContextTokens() >= 4096 &&
              !has(ran.log, "with --disable-ane"),
          "a start given --max-context named the automatic context");
}

// With --max-context the split takes at most the units whose plan still holds it.
void testExplicitContext() {
  Fake fake;
  const uint32_t context = contextWith(fake.bytes(5));
  const Ran ran = run(fake, {}, {context});
  require(ran.outcome().kind == Kind::Split && fake.prepared == std::pair(5u, true) && fake.remembered &&
              fake.remembered->aneUnits == 5 && fake.recalled == 5u && fake.rememberedMost == 5u &&
              ran.started.start.plan && ran.started.start.plan->maximumContextTokens() >= context &&
              !has(ran.log, "with --disable-ane"),
          "the split took more units than the plan holds with the context asked for");
}

// A relaunch holds the context the server's first engine served: the largest split's, whatever that engine's
// calibration made of the split, so that one made anew, as by another build of the engine, serves it too. Its
// split takes at most the units whose plan holds that context, calibrated within them and remembered under them,
// and the log names no automatic context. Where no split leaves it, as after a first engine that took a record of no
// gain, the GPU runs alone, untimed, with a plain line instead of the way out a refused --max-context is warned of;
// a context the GPU alone does not hold either is the start's to report.
void testRelaunchContext() {
  const auto relaunch = [](uint32_t context) { return RequestedContext{.tokens = context, .relaunch = true}; };
  // The most units whose plan holds `context`.
  const auto mostHolding = [](uint32_t context) {
    uint32_t aneUnits = 33;
    while (aneUnits && contextWith(Fake{}.bytes(aneUnits)) < context) --aneUnits;
    return aneUnits;
  };
  const auto relaunched = [](const Ran &ran, uint32_t context) {
    return ran.outcome().kind == Kind::Split && ran.started.start.plan &&
           ran.started.start.plan->maximumContextTokens() >= context && !has(ran.log, "with --disable-ane") &&
           !has(ran.log, "--max-context");
  };
  // The first engine took 8 units; a calibration made anew on an M5 Pro's timings chooses 15, at the same context.
  Fake fake;
  const uint32_t first = run(fake).outcome().context;
  fake = {};
  fake.time = m5Pro;
  Fake automatic = fake;
  require(first == fake.largestSplitContext() && run(automatic).outcome().context == first &&
              automatic.remembered && automatic.remembered->aneUnits == 15,
          "a calibration made anew changed the automatic context");
  Ran ran = run(fake, {}, relaunch(first));
  require(relaunched(ran, first) && fake.timings == 1 && fake.recalled == mostHolding(first) &&
              ran.started.start.maxAneUnits == mostHolding(first) && fake.prepared &&
              fake.prepared->first <= mostHolding(first) && fake.rememberedMost == mostHolding(first),
          "a relaunch did not hold the first engine's context: " + ran.log);
  // The first engine's calibration failed, at the same context; the relaunch's chooses 8 units.
  fake = {};
  fake.time = []() -> Timings { throw std::runtime_error("calibration timings are not usable"); };
  const uint32_t failed = run(fake).outcome().context;
  fake = {};
  require(failed == fake.largestSplitContext(), "a calibration that failed changed the automatic context");
  ran = run(fake, {}, relaunch(failed));
  require(relaunched(ran, failed) && fake.prepared == std::pair(8u, true),
          "a relaunch did not hold the first engine's context after its calibration failed: " + ran.log);
  // The first engine found no gain and served the GPU alone's context, which no split leaves.
  fake = {};
  const uint32_t without = contextWith(0);
  Started started = start(fake, {}, relaunch(without));
  const std::string reason = "the first engine's context of " + grouped(without) + " tokens cannot be held with it";
  require(started.start.outcome.kind == Kind::Refused && started.start.outcome.reason == reason &&
              started.start.outcome.contextWithout == without && nothingStarted(fake, started) &&
              line(started.log, "Neural Engine FFN split off: " + reason + ".") && !has(started.log, "Warning"),
          "a relaunch's context no split leaves did not leave the GPU alone with a plain line: " + started.log);
  started = start(fake, {}, relaunch(without + 1));
  require(started.start.outcome.kind == Kind::Refused && nothingStarted(fake, started) && started.log.empty() &&
              started.start.outcome.reason == "the GPU alone does not hold the first engine's context of " +
                                                  grouped(without + 1) + " tokens either",
          "a relaunch's context the GPU alone does not hold was logged: " + started.log);
}

// A failure of the timings or of the prepared split leaves the GPU alone, the start's plan as it was and nothing
// remembered, and warns; cancellation, an interrupted wait and a backend that stopped serving end the finish.
void testFailures() {
  Fake fake;
  fake.time = []() -> Timings { throw std::runtime_error("calibration timings are not usable"); };
  Ran ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && ran.finished && !ran.split() && ran.started.start.plan &&
              ran.started.start.plan->breakdown().aneFfnBytes == fake.bytes(fake.largestUnits()) && !fake.prepared &&
              !fake.remembered,
          "a timing that failed did not leave the GPU alone");
  require(has(ran.log, "Warning · Neural Engine FFN split unavailable (calibration timings are not usable); the "
                       "GPU runs the prefill FFN alone; context ") &&
              has(ran.log, " with --disable-ane). --disable-ane silences this line."),
          "a timing that failed did not warn: " + ran.log);
  fake = {};
  fake.prepare = [](AneFfn::Shares, bool) -> AneFfnPrepared {
    throw std::runtime_error("ANE FFN split of 640 rows failed (the Neural Engine's output or its scales were not "
                             "finite)");
  };
  ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && !ran.split() && !fake.remembered &&
              has(ran.log, "unavailable (ANE FFN split of 640 rows failed"),
          "a split that failed verify() was not unavailable");
  // Chunks that were not timed are not read.
  fake.prepare = [](AneFfn::Shares, bool) { return AneFfnPrepared{}; };
  ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && !ran.split() && !fake.remembered &&
              has(ran.log, "unavailable (the split's chunks were not timed)"),
          "a split of no chunks timed was not unavailable: " + ran.log);

  bool cancelled = false;
  fake.prepare = [&](AneFfn::Shares, bool) -> AneFfnPrepared {
    cancelled = true;
    throw std::runtime_error("the Metal command was not submitted");
  };
  rejects([&] { static_cast<void>(run(fake, {}, {}, [&] { return cancelled; })); }, "not submitted",
          "a cancelled finish went on");
  fake.prepare = [](AneFfn::Shares, bool) -> AneFfnPrepared { throw ane::Interrupted("the wait was interrupted"); };
  rejects([&] { static_cast<void>(run(fake)); }, "interrupted", "an interrupted wait went on");
  fake.prepare = [&](AneFfn::Shares, bool) -> AneFfnPrepared {
    fake.healthy = false;
    throw std::runtime_error("the backend stopped serving");
  };
  rejects([&] { static_cast<void>(run(fake)); }, "stopped serving", "a finish went on on a backend that stopped");
}

// A command of the split's calibration or setup that ran out of GPU memory ends the start, and resource assembly fails
// on it as on any Metal command of its steps (RuntimeResources::create): the start reports CommandMemory, which no
// start retries, as for a warmup's command. Another process holds what the commands need, and a retry, in the
// process or the server's relaunch, would take it from that process. So does a command of the load, the expert
// store's fill. A command that failed otherwise is Internal at any step: the load's too, whose files the loader took
// (it was Model, which the server never relaunches after). A load the loader refused is still Model.
void testACommandOutOfMemoryIsCommandMemory() {
  using FailureKind = protocol::StartFailureKind;
  for (const bool setup : {false, true}) {
    for (const bool outOfMemory : {true, false}) {
      Fake fake;
      // A failed command stops the backend serving (MetalBackend's markUnhealthy).
      const auto failed = [&] {
        fake.healthy = false;
        return metal::MetalBackendError("Metal command 2 failed: Insufficient Memory", outOfMemory);
      };
      if (setup)
        fake.prepare = [&](AneFfn::Shares, bool) -> AneFfnPrepared { throw failed(); };
      else
        fake.time = [&]() -> Timings { throw failed(); };
      const std::string what = std::string(setup ? "setup" : "calibration") + "'s command that failed " +
                               (outOfMemory ? "for want of GPU memory" : "otherwise");
      try {
        static_cast<void>(run(fake));
        throw std::runtime_error("a start went on after its split's " + what);
      } catch (const metal::MetalBackendError &error) {
        const RuntimeBootstrapReport report =
            RuntimeBootstrapError(RuntimeResourcesError(RuntimeResourceStage::StorageAllocation, error, "budget"))
                .report();
        const FailureKind kind = startFailedEvent(report).kind;
        require(kind == (outOfMemory ? FailureKind::CommandMemory : FailureKind::Internal) &&
                    report.commandOutOfMemory == outOfMemory &&
                    report.resourceStage == RuntimeResourceStage::StorageAllocation &&
                    report.resourceFailure == RuntimeResourceFailure::Other &&
                    report.message == (outOfMemory ? startMemoryRefusal("Metal command 2 failed: Insufficient Memory")
                                                   : "Metal command 2 failed: Insufficient Memory") &&
                    report.budgetDescription == "budget",
                "a start whose split's " + what + " failed as kind " +
                    std::to_string(static_cast<int>(kind)) + (report.commandOutOfMemory ? ", out of memory" : ""));
        StartupRetryWindow window(supervision::kStartRetryWindow);
        require(!window.retryUntil(report, StartupRetryWindow::Clock::now()),
                "a start whose split's " + what + " was retried in-process");
      }
    }
  }
  for (const bool outOfMemory : {true, false}) {
    const RuntimeBootstrapReport report =
        RuntimeBootstrapError(RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                                    metal::MetalBackendError("Metal command 1 failed", outOfMemory)))
            .report();
    require(startFailedEvent(report).kind == (outOfMemory ? FailureKind::CommandMemory : FailureKind::Internal),
            std::string("a load whose command failed ") + (outOfMemory ? "for want of GPU memory" : "otherwise") +
                " was not classed by it");
  }
  const RuntimeBootstrapReport refused =
      RuntimeBootstrapError(RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                                  "the target's safetensors header is truncated"))
          .report();
  require(startFailedEvent(refused).kind == FailureKind::Model, "a load the loader refused was not Model");
}

// No share predicted to gain enough, or no chunk timed to, leaves the GPU alone and remembers nothing, so the next
// start calibrates again, at the plan's context: one start beside other work (an M3 Ultra beside two serving servers
// timed 84.0 ms a split layer against its GPU alone's 62.8, where a quiet start splits 0.41) or on a GPU whose pace
// moved would otherwise decide for every later start.
void testNoGain() {
  Fake fake;
  fake.time = [] {
    Timings timings = measured();
    timings.gpuAlone = 14.5;
    return timings;
  };
  Ran ran = run(fake);
  require(ran.outcome().kind == Kind::NoGain && !fake.prepared && !fake.remembered && !ran.split() &&
              has(ran.log, "The GPU runs the prefill FFN alone: no Neural Engine split beats its 14.5 ms per "
                           "2048-row FFN layer by enough (calibrated in ") &&
              has(ran.log, "; the next start calibrates again)"),
          "a share predicted to gain too little split, or was remembered: " + ran.log);
  ran = run(fake);
  require(ran.outcome().kind == Kind::NoGain && ran.finished && fake.timings == 2 && !fake.remembered &&
              ran.outcome().context == fake.largestSplitContext() && ran.started.start.plan,
          "the start after a calibration of no gain did not calibrate again at the plan's context: " + ran.log);
  fake = {};
  fake.prepare = [](AneFfn::Shares, bool) {
    AneFfnPrepared prepared;
    prepared.chunks = timedChunks(21.0);
    return prepared;
  };
  ran = run(fake);
  require(ran.outcome().kind == Kind::NoGain && fake.preparedUnits == std::vector{8u} && !fake.remembered &&
              !ran.split() &&
              has(ran.log, "no chunk of the Neural Engine split beats it by enough, 21.0 ms per 2048-row FFN "
                           "layer against 20.0 on the GPU alone (calibrated in "),
          "a split of no chunk that gains split: " + ran.log);
}

// The mixer is the rest of the layer of the engine's prefill of a full chunk on the GPU alone: its milliseconds over
// the model's FFN layers, less the GPU alone's FFN layer as timed, none where that is less. The same timings choose 4
// units of two steps beside a mixer of 10 ms, both shares prepared in one program and remembered with the mixer beside
// the record's own GPU alone (remembered()), and none beside one of 2 ms, which prepares one share; the same prefill
// over half the layers is a mixer of 22.99 ms, which takes two steps too; a mixer of none still prices them by their
// half step, and takes none.
void testTheMixerComesFromThePrefill() {
  Fake fake;
  Started started = start(fake);
  Finished finished = finish(fake, started, (18.990 + 10.0) * 64);
  require(finished.finished.outcome.kind == Kind::Split &&
              fake.preparations == std::vector<Preparation>{{8, 4, 512, 1024, true}} && fake.remembered &&
              fake.remembered->aneUnits == 8 && fake.remembered->twoStepUnits == 4 &&
              near(fake.remembered->mixer, remembered(10.0)) &&
              has(finished.log, "Neural Engine FFN split at share 0.24 for chunks of 553 rows or more and 0.12 in two "
                                "steps from 1024 rows: 8.0 ms per 1024-row FFN layer against 10.0 on the GPU alone; "
                                "two steps ") &&
              has(finished.log, " ms per 2048-row layer by the fit against 29.0, 2.5% from the GPU alone"),
          "a prefill of a 10 ms mixer did not take two steps: " + text(fake.preparations) + finished.log);
  fake = {};
  started = start(fake);
  finished = finish(fake, started, (18.990 + 2.0) * 64);
  require(finished.finished.outcome.kind == Kind::Split &&
              fake.preparations == std::vector<Preparation>{{8, 0, 512, 0, true}} && fake.remembered &&
              !fake.remembered->twoStepUnits && near(fake.remembered->mixer, remembered(2.0)) &&
              !has(finished.log, "two steps"),
          "a prefill of a 2 ms mixer took two steps: " + text(fake.preparations) + finished.log);
  fake = {};
  fake.layers = 32;
  started = start(fake);
  finished = finish(fake, started, (18.990 + 2.0) * 64);
  require(finished.finished.outcome.kind == Kind::Split && fake.remembered && fake.remembered->twoStepUnits &&
              near(fake.remembered->mixer, remembered(2 * (18.990 + 2.0) - 18.990)),
          "the prefill was not taken over the model's layers: " + text(fake.preparations));
  // A prefill no longer than the GPU alone's FFN layers is a mixer of none, not less, beside which two steps are still
  // priced by their timed half step, and gain nothing here. They were priced as one step, and took its 8 units.
  fake = {};
  started = start(fake);
  finished = finish(fake, started, 18.990 * 64 / 2);
  require(finished.finished.outcome.kind == Kind::Split &&
              fake.preparations == std::vector<Preparation>{{8, 0, 512, 0, true}} && fake.remembered &&
              fake.remembered->mixer == 0.0 && !fake.remembered->twoStepUnits,
          "a prefill shorter than its FFN layers was a mixer of less than none, or priced two steps as one: " +
              text(fake.preparations) + finished.log);
}

// Two steps take every command from AneFfn::kTwoStepRows, 1,024, whatever one step's least rows. The M5 Max's own
// half step beside a mixer of 10 ms chooses 13 units of two steps beside one step's 8, prepared together in one
// program whose one-step functions, to 1,024 rows, are timed: one step from its least rows, 553, two from 1,024, and
// remembered so; a start that takes the record prepares the same split, untimed. One step's chunks splitting only from
// 897 rows still leave two steps from 1,024. It took two steps from twice one step's least rows.
void testTwoStepsFromFixedRows() {
  require(Fake{}.largestUnits() >= 13, "the plan does not hold the 13 units two steps choose");
  Fake fake;
  fake.time = pipelined;
  fake.mixer = 10.0;
  Ran ran = run(fake);
  require(ran.outcome().kind == Kind::Split &&
              fake.preparations == std::vector<Preparation>{{8, 13, 512, 1024, true}} && fake.remembered &&
              fake.remembered->aneUnits == 8 && fake.remembered->minimumRows == 553 &&
              fake.remembered->twoStepUnits == 13 && near(fake.remembered->mixer, remembered(10.0)) &&
              fake.remembered->gpu[0].rows == 512 && fake.remembered->gpu[1].rows == 2048 &&
              has(ran.log, "at share 0.24 for chunks of 553 rows or more and 0.38 in two steps from 1024 rows: "),
          "two steps were not prepared and remembered from 1024 rows: " + text(fake.preparations) + ran.log);
  fake.preparations.clear();
  ran = run(fake);
  require(ran.outcome().kind == Kind::Split && fake.timings == 1 &&
              fake.preparations == std::vector<Preparation>{{8, 13, 553, 1024, false}} &&
              has(ran.log, "at share 0.24 for chunks of 553 rows or more and 0.38 in two steps from 1024 rows, 2.5% "
                           "from the GPU alone on the Neural Engine's part (set up as calibrated before"),
          "a start did not set up the remembered two steps: " + text(fake.preparations) + ran.log);
  fake = {};
  fake.time = pipelined;
  fake.mixer = 10.0;
  fake.prepare = [](AneFfn::Shares shares, bool timed) {
    AneFfnPrepared prepared;
    if (timed) {
      prepared.chunks = timedChunks(8.0, shares.twoSteps ? AneFfn::kTwoStepRows : AneFfn::kMaximumRows);
      // Functions of 896 rows or fewer as slow as twice the GPU's FFN.
      for (ChunkTimings::Function &function : prepared.chunks.functions)
        if (function.rows <= 896) function.split = 2.0 * function.split;
    }
    return prepared;
  };
  ran = run(fake);
  require(ran.outcome().kind == Kind::Split && fake.remembered && fake.remembered->minimumRows == 897 &&
              fake.remembered->twoStepUnits == 13 &&
              has(ran.log, "for chunks of 897 rows or more and 0.38 in two steps from 1024 rows"),
          "one step's least rows of 897 moved two steps' rows: " + ran.log);
  fake.preparations.clear();
  ran = run(fake);
  require(ran.outcome().kind == Kind::Split &&
              fake.preparations == std::vector<Preparation>{{8, 13, 897, 1024, false}},
          "a start did not set up two steps from 1024 rows beside one step from 897: " + text(fake.preparations));
}

// Two steps run only where a command of them queues no more evaluations than the Neural Engine holds in flight
// (AneFfn::fits): a model of 64 FFN layers takes them beside a mixer of 10 ms, one of 65 does not, and its split,
// one share in the program of one share, timed to 2048 rows, records none. Without two steps one step may take
// chunks from 1,024 rows or more: here from 1,100.
void testTwoStepsOnlyWhereTheyFit() {
  for (const uint32_t layers : {64u, 65u}) {
    Fake fake;
    fake.layers = layers;
    fake.mixer = 10.0;
    const Ran ran = run(fake);
    const bool two = layers == 64;
    require(ran.outcome().kind == Kind::Split &&
                fake.preparations ==
                    std::vector<Preparation>{two ? Preparation{8, 4, 512, 1024, true} : Preparation{8, 0, 512, 0, true}} &&
                fake.remembered && fake.remembered->twoStepUnits == (two ? 4u : 0u) &&
                near(fake.remembered->mixer, remembered(10.0)) && has(ran.log, "two steps") == two,
            "a model of " + std::to_string(layers) + " layers took two steps " + (two ? "nowhere" : "") + ": " +
                text(fake.preparations) + ran.log);
  }
  Fake fake;
  fake.layers = 65;
  fake.mixer = 10.0;
  fake.prepare = [](AneFfn::Shares, bool timed) {
    AneFfnPrepared prepared;
    if (timed) {
      prepared.chunks = timedChunks(15.5);
      // Function 1152's chunks split from 1,100 rows (1.02 of the GPU alone's 10.742 ms is 10.957, of 1,099 rows'
      // 10.947).
      prepared.chunks.functions[7].split = 10.952;
    }
    return prepared;
  };
  const Ran ran = run(fake);
  require(ran.outcome().kind == Kind::Split && fake.remembered && fake.remembered->minimumRows == 1100 &&
              !fake.remembered->twoStepUnits && has(ran.log, "for chunks of 1100 rows or more: "),
          "one step without two did not take chunks from 1100 rows: " + ran.log);
}

// Each kind of command stands on its own: a split of both shares whose one-step chunks split only from 1,024 rows, or
// none of them, takes no command in one step and serves two steps alone from 1,024 rows, which it remembers; the next
// start sets that up as remembered, and a start whose mixer reads the record's busier calibrates it again. So does a
// split whose one-step fit chose no units, two steps' naming its program. Neither kind gaining leaves the GPU alone
// and remembers none; a split of one share takes chunks from 1,024 rows.
void testTwoStepsStandAlone() {
  // The function of 1024 rows takes 10.195 ms, within 1.02 of the GPU alone's 10.0 at 1,024 rows, not at 1,023.
  const auto from1024 = [](AneFfn::Shares shares, bool timed) {
    AneFfnPrepared prepared;
    if (!timed) return prepared;
    if (shares.twoSteps) {
      prepared.chunks = timedChunks(10.195, AneFfn::kTwoStepRows);
    } else {
      prepared.chunks = timedChunks(15.5);
      prepared.chunks.functions[8].split = 10.195;
    }
    return prepared;
  };
  // Of 1024 rows 11.0 ms: no chunk.
  const auto splitsNone = [](AneFfn::Shares, bool timed) {
    AneFfnPrepared prepared;
    if (timed) prepared.chunks = timedChunks(11.0, AneFfn::kTwoStepRows);
    prepared.error = 0.025;
    return prepared;
  };
  for (const bool none : {false, true}) {
    Fake fake;
    fake.mixer = 10.0;
    if (none) fake.prepare = splitsNone;
    else
      fake.prepare = [&](AneFfn::Shares shares, bool timed) {
        AneFfnPrepared prepared = from1024(shares, timed);
        prepared.error = 0.025;
        return prepared;
      };
    const std::string form = none ? "none of whose one-step chunks splits" : "whose one step splits from 1024 rows";
    Ran ran = run(fake);
    require(ran.outcome().kind == Kind::Split &&
                fake.preparations == std::vector<Preparation>{{8, 4, 512, 1024, true}} && fake.remembered &&
                fake.remembered->aneUnits == 8 && !fake.remembered->minimumRows &&
                fake.remembered->twoStepUnits == 4 && near(fake.remembered->mixer, remembered(10.0)) &&
                has(ran.log, std::string("Neural Engine FFN split at share 0.12 in two steps from 1024 rows: ") +
                                 (none ? "11.0" : "10.2") + " ms per 1024-row FFN layer against 10.0 on the GPU "
                                                            "alone; two steps "),
            "a split of both shares " + form + " did not serve two steps alone: " + text(fake.preparations) + ran.log);
    fake.preparations.clear();
    ran = run(fake);
    require(ran.outcome().kind == Kind::Split && fake.timings == 1 &&
                fake.preparations == std::vector<Preparation>{{8, 4, 0, 1024, false}} &&
                has(ran.log, "Neural Engine FFN split at share 0.12 in two steps from 1024 rows, 2.5% from the GPU "
                             "alone on the Neural Engine's part (set up as calibrated before"),
            "a start did not set up the remembered two steps alone: " + text(fake.preparations) + ran.log);
    fake.mixer = 5.0;
    ran = run(fake);
    require(fake.timings == 2 && has(ran.log, "taken on a busier Mac (its mixer 9.0 ms a full chunk's layer, 5.0 "
                                              "now); the start calibrates it again."),
            "a busier record of two steps alone was not calibrated again: " + ran.log);
  }
  // Where one step's fit chooses no units, two steps are still asked: their units name the program, here beside a
  // mixer of 10 ms where one step's fit, an ANE slower than the GPU's part, gains 4.5% at most.
  {
    Fake fake;
    fake.mixer = 10.0;
    fake.time = [] { return Timings{{14.0 / 34, 14.0, 23.484}, {27.0 / 34, 5.033, 56.0}, 18.990, 12.0}; };
    fake.prepare = splitsNone;
    const Ran ran = run(fake);
    require(ran.outcome().kind == Kind::Split && fake.preparations.size() == 1 &&
                fake.preparations.front().oneStep == fake.preparations.front().twoSteps && fake.remembered &&
                fake.remembered->aneUnits == fake.preparations.front().twoSteps && !fake.remembered->minimumRows &&
                fake.remembered->twoStepUnits == fake.preparations.front().twoSteps &&
                has(ran.log, " in two steps from 1024 rows: 11.0 ms per 1024-row FFN layer"),
            "a split whose one-step fit chose no units did not serve two steps alone: " + text(fake.preparations) +
                ran.log);
  }
  // Beside the mixer of 2 ms two steps gain nothing either.
  Fake fake;
  fake.prepare = splitsNone;
  Ran ran = run(fake);
  require(ran.outcome().kind == Kind::NoGain && !fake.remembered &&
              has(ran.log, "The GPU runs the prefill FFN alone: no chunk of the Neural Engine split beats it by "
                           "enough, 11.0 ms per 1024-row FFN layer against 10.0 on the GPU alone (calibrated in "),
          "a split that gains in neither kind split: " + ran.log);
  fake = {};
  fake.layers = 65;
  fake.mixer = 10.0;
  fake.prepare = from1024;
  ran = run(fake);
  require(ran.outcome().kind == Kind::Split && fake.remembered && fake.remembered->minimumRows == 1024 &&
              !fake.remembered->twoStepUnits && has(ran.log, "for chunks of 1024 rows or more: "),
          "a split of one share did not take chunks from 1024 rows: " + ran.log);
}

// The chosen units' full chunk, timed nearest T's least as the fit's timings were and on their layers, fits the
// bandwidth's term where the low share could not: the M5 Max's timings choose 6 units, whose split layer of 13.84 ms
// chooses 7 (ane_ffn_calibration_test), whose split is prepared, timed, served and remembered instead where it takes
// no chunk longer, and otherwise the split of 6 serves, prepared again. One whose refitted model keeps its units is
// prepared once. The first layers' chunks do not refit: a model whose first layers' formats are slower, as mixed
// quantizations' can be, would bias it.
void testRefitMovesTheChoice() {
  Fake fake;
  fake.time = [] { return refitting(); };
  fake.timeAt = [](uint32_t) { return 13.84; };
  fake.prepare = [](AneFfn::Shares shares, bool timed) {
    AneFfnPrepared prepared;
    if (timed) prepared.chunks = timedChunks(shares.oneStep == 6 ? 13.84 : 13.55);
    prepared.error = 0.025;
    return prepared;
  };
  Ran ran = run(fake);
  require(ran.outcome().kind == Kind::Split && fake.timings == 1 && fake.timedUnits == std::vector{6u} &&
              fake.preparations == std::vector<Preparation>{{6, 0, 512, 0, true}, {7, 0, 512, 0, true}} &&
              fake.remembered && fake.remembered->aneUnits == 7 && fake.remembered->minimumRows == 553 &&
              has(ran.log, "Neural Engine FFN split at share 0.21 for chunks of 553 rows or more: 13.6 ms per "
                           "2048-row FFN layer against 20.0 on the GPU alone"),
          "the refitted choice of 7 units did not serve: " + text(fake.preparations) + ran.log);
  // Seven units whose 640-row function takes 6.0 ms, against six's 5.5, split chunks from 603 rows, not 553, and the
  // split of six serves, prepared again.
  fake = {};
  fake.time = [] { return refitting(); };
  fake.timeAt = [](uint32_t) { return 13.84; };
  fake.prepare = [](AneFfn::Shares shares, bool timed) {
    AneFfnPrepared prepared;
    if (timed) {
      prepared.chunks = timedChunks(shares.oneStep == 6 ? 13.84 : 13.55);
      if (shares.oneStep == 7) prepared.chunks.functions[11].split = 6.0;
    }
    prepared.error = 0.025;
    return prepared;
  };
  ran = run(fake);
  require(ran.outcome().kind == Kind::Split &&
              fake.preparations ==
                  std::vector<Preparation>{{6, 0, 512, 0, true}, {7, 0, 512, 0, true}, {6, 0, 512, 0, false}} &&
              fake.prepared == std::pair(6u, false) && fake.remembered && fake.remembered->aneUnits == 6 &&
              fake.remembered->minimumRows == 553 &&
              has(ran.log, "Neural Engine FFN split at share 0.18 for chunks of 553 rows or more: 13.8 ms per "
                           "2048-row FFN layer against 20.0 on the GPU alone"),
          "a refitted choice that took chunks longer served: " + text(fake.preparations) + ran.log);
  fake.preparedUnits.clear();
  fake.stored.reset();
  fake.time = measured;
  fake.timeAt = [](uint32_t) { return 15.5; };
  ran = run(fake);
  require(ran.outcome().kind == Kind::Split && fake.preparedUnits == std::vector{8u} && fake.remembered &&
              fake.remembered->aneUnits == 8,
          "a choice its refitted model keeps was prepared again");
  // Six units' full chunk takes 13.84 ms over the first layers, which would choose 7, but 13.19 over the fit's, which
  // keeps 6; and the other way about, 7, whose full chunk takes 13.0 ms.
  for (const bool fitLayersMove : {false, true}) {
    fake = {};
    fake.time = [] { return refitting(); };
    fake.timeAt = [=](uint32_t) { return fitLayersMove ? 13.84 : 13.19; };
    fake.prepare = [=](AneFfn::Shares shares, bool timed) {
      AneFfnPrepared prepared;
      if (timed) prepared.chunks = timedChunks(shares.oneStep == 6 ? (fitLayersMove ? 13.19 : 13.84) : 13.0);
      prepared.error = 0.025;
      return prepared;
    };
    ran = run(fake);
    require(ran.outcome().kind == Kind::Split &&
                fake.preparedUnits == (fitLayersMove ? std::vector{6u, 7u} : std::vector{6u}) && fake.remembered &&
                fake.remembered->aneUnits == (fitLayersMove ? 7u : 6u),
            "the first layers' chunks, not the split timed on the fit's layers, refitted the choice: " + ran.log);
  }
  // A refit whose split cannot be timed leaves the first units, whose split is prepared, served and remembered.
  fake = {};
  fake.time = [] { return refitting(); };
  fake.timeAt = [](uint32_t) -> double { throw std::runtime_error("the Neural Engine refused the program"); };
  fake.prepare = [](AneFfn::Shares, bool timed) {
    AneFfnPrepared prepared;
    if (timed) prepared.chunks = timedChunks(13.84);
    prepared.error = 0.025;
    return prepared;
  };
  ran = run(fake);
  require(ran.outcome().kind == Kind::Split && fake.preparedUnits == std::vector{6u} && fake.remembered &&
              fake.remembered->aneUnits == 6 &&
              has(ran.log, "refitted share was not found (the Neural Engine refused the program); its first "
                           "share serves"),
          "a refit whose split could not be timed did not leave the first units to serve: " + ran.log);
}

// The refit moves one step's units in the one program of both shares: the M5 Max's timings with their own half step
// beside a mixer of 10 ms choose 6 units of one step, refitted to 7, and 11 of two steps, and the refit's candidate is
// prepared and timed with both shares, as is the first split prepared again where the candidate takes a chunk longer.
void testTheRefitKeepsBothShares() {
  for (const bool slower : {false, true}) {
    Fake fake;
    fake.time = [] { return refitting(23.017); };
    fake.mixer = 10.0;
    fake.timeAt = [](uint32_t) { return 13.84; };
    fake.prepare = [=](AneFfn::Shares shares, bool timed) {
      AneFfnPrepared prepared;
      if (timed) {
        prepared.chunks = timedChunks(shares.oneStep == 6 ? 8.0 : 7.9, AneFfn::kTwoStepRows);
        // Seven units' 640-row function at 6.0 ms splits chunks from 603 rows, not 553.
        if (slower && shares.oneStep == 7) prepared.chunks.functions[3].split = 6.0;
      }
      prepared.error = 0.025;
      return prepared;
    };
    const Ran ran = run(fake);
    std::vector<Preparation> expected{{6, 11, 512, 1024, true}, {7, 11, 512, 1024, true}};
    if (slower) expected.push_back({6, 11, 512, 1024, false});
    require(ran.outcome().kind == Kind::Split && fake.preparations == expected && fake.remembered &&
                fake.remembered->aneUnits == (slower ? 6u : 7u) && fake.remembered->twoStepUnits == 11 &&
                fake.remembered->minimumRows == 553 &&
                has(ran.log, std::string("at share ") + (slower ? "0.18" : "0.21") +
                                 " for chunks of 553 rows or more and 0.32 in two steps from 1024 rows: "),
            std::string("the refit's ") + (slower ? "first split" : "candidate") +
                " was not prepared at both shares: " + text(fake.preparations) + ran.log);
  }
}

// A refitted split that cannot be prepared leaves the first, which was: the M5 Max's timings choose 6 units and then
// 7 (testRefitMovesTheChoice), whose program the Neural Engine refuses, and the split of 6 serves, prepared again, and
// is remembered, the log saying why. It ended the start with the GPU alone and nothing remembered, so that every
// later start calibrated again. On a backend that stopped serving, or in a finish that was cancelled, the failure
// ends the finish as any other does.
void testARefittedSplitThatFailsKeepsTheFirst() {
  Fake fake;
  fake.time = [] { return refitting(); };
  fake.timeAt = [](uint32_t) { return 13.84; };
  fake.prepare = [](AneFfn::Shares shares, bool timed) {
    if (shares.oneStep == 7) throw std::runtime_error("the Neural Engine refused the program of 7 units");
    AneFfnPrepared prepared;
    if (timed) prepared.chunks = timedChunks(13.84);
    prepared.error = 0.025;
    return prepared;
  };
  const Ran ran = run(fake);
  require(ran.outcome().kind == Kind::Split &&
              fake.preparations ==
                  std::vector<Preparation>{{6, 0, 512, 0, true}, {7, 0, 512, 0, true}, {6, 0, 512, 0, false}} &&
              fake.prepared == std::pair(6u, false) && fake.remembered && fake.remembered->aneUnits == 6 &&
              fake.remembered->minimumRows == 553 &&
              has(ran.log, "refitted share was not prepared (the Neural Engine refused the program of 7 units); "
                           "its first share serves") &&
              has(ran.log, "Neural Engine FFN split at share 0.18 for chunks of 553 rows or more: 13.8 ms per "
                           "2048-row FFN layer against 20.0 on the GPU alone"),
          "a refitted split that failed to prepare did not leave the first to serve: " + ran.log);
  fake = {};
  fake.time = [] { return refitting(); };
  fake.timeAt = [](uint32_t) { return 13.84; };
  fake.prepare = [&](AneFfn::Shares shares, bool timed) {
    if (shares.oneStep == 7) {
      fake.healthy = false;
      throw std::runtime_error("the backend stopped serving");
    }
    AneFfnPrepared prepared;
    if (timed) prepared.chunks = timedChunks(13.84);
    return prepared;
  };
  rejects([&] { static_cast<void>(run(fake)); }, "stopped serving",
          "a finish went on past a refitted split's failure on a backend that stopped");
  fake = {};
  fake.time = [] { return refitting(); };
  fake.timeAt = [](uint32_t) { return 13.84; };
  bool cancelled = false;
  fake.prepare = [&](AneFfn::Shares shares, bool timed) {
    if (shares.oneStep == 7) {
      cancelled = true;
      throw std::runtime_error("ANE program was not submitted");
    }
    AneFfnPrepared prepared;
    if (timed) prepared.chunks = timedChunks(13.84);
    return prepared;
  };
  rejects([&] { static_cast<void>(run(fake, {}, {}, [&] { return cancelled; })); }, "not submitted",
          "a cancelled finish went on past a refitted split's failure");
}

// A first split that does not prepare again after the refitted one lost leaves the GPU alone to this start, its
// context the start's, and the calibration of the first units remembered, which the next start sets up: the M5 Max's
// timings choose 6 units and then 7, whose 640-row function takes 6.0 ms (testRefitMovesTheChoice), and the Neural
// Engine refuses the program of 6 the second time. It ended the start with the GPU alone and nothing remembered, so
// that every later start calibrated again.
void testAFirstSplitThatDoesNotPrepareAgainIsRemembered() {
  Fake fake;
  fake.time = [] { return refitting(); };
  fake.timeAt = [](uint32_t) { return 13.84; };
  uint32_t preparedSix = 0;
  fake.prepare = [&](AneFfn::Shares shares, bool timed) {
    if (shares.oneStep == 6 && ++preparedSix == 2)
      throw std::runtime_error("the Neural Engine refused the program of 6 units");
    AneFfnPrepared prepared;
    if (timed) {
      prepared.chunks = timedChunks(shares.oneStep == 6 ? 13.84 : 13.55);
      if (shares.oneStep == 7) prepared.chunks.functions[11].split = 6.0;
    }
    prepared.error = 0.025;
    return prepared;
  };
  Ran ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && !ran.split() && fake.preparedUnits == std::vector{6u, 7u, 6u} &&
              fake.remembered && fake.remembered->aneUnits == 6 && fake.remembered->minimumRows == 553 &&
              ran.outcome().context == fake.largestSplitContext() &&
              has(ran.log, "its calibrated share's split was not prepared again: the Neural Engine refused the "
                           "program of 6 units; the next start sets it up as calibrated"),
          "a first split that did not prepare again left nothing remembered: " + ran.log);
  ran = run(fake);
  require(ran.outcome().kind == Kind::Split && fake.timings == 1 && fake.prepared == std::pair(6u, false) &&
              has(ran.log, "set up as calibrated before"),
          "the next start did not set up the remembered first units: " + ran.log);
}

// A split the Neural Engine refuses (ane::Refused) counts beside the calibration, and the second refusal leaves the
// GPU alone, uncalibrated, until the calibration's key changes: it was forgotten at each refusal, so that a split the
// Neural Engine refused each time was calibrated every other start. A calibration whose first split does not prepare
// again is kept with the refusal and refused again by the next start (testAFirstSplitThatDoesNotPrepareAgainIsRemembered's
// timings); a remembered split refused once is calibrated again, and that split refused; a split refused while it is
// calibrated is calibrated once more. The log says what the next start does, as what stayed remembered says.
void testARefusedSplitIsNotCalibratedInALoop() {
  const auto refusedUntimed = [](AneFfn::Shares shares, bool timed) {
    if (!timed)
      throw ane::Refused("ANE load failed: the Neural Engine refused the program of " +
                         std::to_string(shares.oneStep) + " units");
    AneFfnPrepared prepared;
    prepared.chunks = timedChunks(shares.oneStep == 6 ? 13.84 : 13.55);
    if (shares.oneStep == 7) prepared.chunks.functions[11].split = 6.0;
    return prepared;
  };
  constexpr std::string_view kLeft = "starts leave the GPU alone until Splash, macOS or the model changes";
  Fake fake;
  fake.time = [] { return refitting(); };
  fake.timeAt = [](uint32_t) { return 13.84; };
  fake.prepare = refusedUntimed;
  Ran ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && fake.stored && fake.stored->aneUnits == 6 &&
              fake.stored->refusals == 1 &&
              has(ran.log, "refused the program of 6 units; the next start sets it up as calibrated"),
          "a calibration whose first split was refused again was not kept with the refusal: " + ran.log);
  ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && fake.stored && !fake.stored->aneUnits &&
              fake.stored->refusals == 2 && has(ran.log, std::string("refused the program of 6 units; ") + kLeft.data()),
          "a split refused twice did not leave the GPU alone: " + ran.log);
  for (int later = 0; later < 2; ++later) {
    fake.preparedUnits.clear();
    ran = run(fake);
    require(ran.outcome().kind == Kind::Unavailable && !ran.finished && fake.timings == 1 &&
                fake.preparedUnits.empty() && ran.outcome().context == contextWith(0) &&
                has(ran.log, std::string("the Neural Engine refused its split twice; ") + kLeft.data()),
            "a start after a split refused twice calibrated or set a split up: " + ran.log);
  }

  const std::array<ChunkTimings::Gpu, 2> gpu{{{512, 5.0}, {2048, 20.0}}};
  fake = {};
  fake.stored = Calibration{9, 700, gpu};
  fake.prepare = [](AneFfn::Shares shares, bool timed) -> AneFfnPrepared {
    if (shares.oneStep == 9) throw ane::Refused("ANE compilation failed");
    return Fake{}.prepare(shares, timed);
  };
  fake.keeps = false;
  ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && fake.stored && fake.stored->aneUnits == 9 &&
              has(ran.log, "ANE compilation failed; the next start sets it up as calibrated"),
          "a refusal that could not be remembered said the next start would not set the split up: " + ran.log);
  fake.keeps = true;
  ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && fake.stored && !fake.stored->aneUnits &&
              fake.stored->refusals == 1 && has(ran.log, "ANE compilation failed; the next start calibrates again"),
          "a remembered split refused once was not left to calibrate again: " + ran.log);
  ran = run(fake);
  require(ran.outcome().kind == Kind::Split && fake.timings == 1 && fake.stored && fake.stored->aneUnits == 8 &&
              fake.stored->refusals == 1,
          "the start after a refusal did not calibrate again, keeping the refusal: " + ran.log);
  fake.prepare = [](AneFfn::Shares, bool) -> AneFfnPrepared { throw ane::Refused("ANE load failed"); };
  ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && fake.stored && fake.stored->refusals == 2 &&
              has(ran.log, std::string("ANE load failed; ") + kLeft.data()),
          "a split calibrated again after a refusal and refused did not leave the GPU alone: " + ran.log);

  fake = {};
  fake.prepare = [](AneFfn::Shares, bool) -> AneFfnPrepared { throw ane::Refused("ANE load failed"); };
  ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && fake.stored && fake.stored->refusals == 1 &&
              has(ran.log, "ANE load failed; the next start calibrates again"),
          "a split refused while it was calibrated was not counted: " + ran.log);
  ran = run(fake);
  ran = run(fake);
  require(fake.timings == 2 && fake.stored && fake.stored->refusals == 2 &&
              has(ran.log, std::string("the Neural Engine refused its split twice; ") + kLeft.data()),
          "a split refused each time it was calibrated was calibrated more than twice: " + ran.log);
}

// A remembered split that fails to prepare for another reason than the Neural Engine's refusal, as Metal refusing an
// arena under memory pressure, keeps its calibration: that start runs the GPU alone and the next sets the split up.
// It was forgotten, and the next start calibrated again.
void testAFailureThatIsNotARefusalKeepsTheCalibration() {
  const std::array<ChunkTimings::Gpu, 2> gpu{{{512, 5.0}, {2048, 20.0}}};
  Fake fake;
  fake.stored = Calibration{9, 700, gpu};
  fake.prepare = [](AneFfn::Shares, bool) -> AneFfnPrepared {
    throw metal::MetalAllocationError("the prefill arena was not allocated", metal::AllocationFailure::DriverRejected);
  };
  Ran ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && !ran.split() && fake.stored && fake.stored->aneUnits == 9 &&
              fake.stored->minimumRows == 700 && !fake.stored->refusals &&
              has(ran.log, "its remembered share's split was not prepared: the prefill arena was not allocated; "
                           "the next start sets it up as calibrated"),
          "a remembered split that failed for another reason than a refusal lost its calibration: " + ran.log);
  fake.prepare = Fake{}.prepare;
  ran = run(fake);
  require(ran.outcome().kind == Kind::Split && !fake.timings && fake.prepared == std::pair(9u, false),
          "the start after a failure that was not a refusal did not set the calibration up: " + ran.log);
}

// A start takes this Mac's calibration once remembered, and leaves it to its finish: the start sets nothing up, and
// the finish neither times nor remembers, prepares the remembered units untimed with the remembered least chunk, of
// one step and of two from 1,024 rows where the record has a share of two, or of two steps alone, and says so. A
// remembered record of no split, of two refusals, leaves the GPU alone at once. One of more units than the plan
// holds, of either kind, of least rows no function of one step holds, below two steps' 1,024 where it has a share of
// two, or of neither least rows nor two steps, is calibrated again.
void testRemembered() {
  const std::array<ChunkTimings::Gpu, 2> gpu{{{512, 5.0}, {2048, 20.0}}};
  Fake fake;
  fake.stored = Calibration{9, 700, gpu};
  const Started started = start(fake);
  require(started.start.outcome.kind == Kind::Pending && !fake.prepared && !fake.timings && started.log.empty(),
          "a start set the remembered split up before its runtime's prefill: " + started.log);
  const Finished finished = finish(fake, started, prefillMilliseconds(fake, started.start));
  require(finished.finished.outcome.kind == Kind::Split && !fake.timings &&
              fake.preparations == std::vector<Preparation>{{9, 0, 700, 0, false}} && !fake.remembered &&
              fake.recalled && settingUpFirst(finished.log) &&
              has(finished.log, "Neural Engine FFN split at share 0.26 for chunks of 700 rows or more, 2.5% from the "
                                "GPU alone on the Neural Engine's part (set up as calibrated before"),
          "the remembered calibration was not taken: " + finished.log);
  // Of both kinds, and at the edges of the rows a record may hold.
  for (const Calibration &record : {Calibration{9, 700, gpu, 0, 4, 2.0}, Calibration{9, 1023, gpu, 0, 4, 2.0},
                                    Calibration{9, AneFfn::kMaximumRows, gpu}}) {
    fake = {};
    fake.stored = record;
    const Ran ran = run(fake);
    require(ran.outcome().kind == Kind::Split && !fake.timings && !fake.remembered &&
                fake.preparations ==
                    std::vector<Preparation>{{9, record.twoStepUnits, record.minimumRows,
                                              record.twoStepUnits ? AneFfn::kTwoStepRows : 0, false}} &&
                has(ran.log, "for chunks of " + std::to_string(record.minimumRows) + " rows or more" +
                                 (record.twoStepUnits ? " and 0.12 in two steps from 1024 rows, " : ", ")),
            "the remembered calibration " + record.text() + " was not taken: " + text(fake.preparations) + ran.log);
  }
  fake = {};
  fake.stored = Calibration{9, 0, gpu, 0, 4, 2.0};
  Ran alone = run(fake);
  require(alone.outcome().kind == Kind::Split && !fake.timings && !fake.remembered &&
              fake.preparations == std::vector<Preparation>{{9, 4, 0, AneFfn::kTwoStepRows, false}} &&
              has(alone.log, "at share 0.12 in two steps from 1024 rows, 2.5% from the GPU alone"),
          "the remembered two steps alone were not taken: " + text(fake.preparations) + alone.log);
  fake = {};
  fake.stored = Calibration{.refusals = Calibration::kRefusals};
  Ran ran = run(fake);
  require(ran.outcome().kind == Kind::Unavailable && !ran.finished && !fake.timings && !fake.prepared &&
              !fake.remembered && !ran.started.start.plan &&
              has(ran.log, "the Neural Engine refused its split twice; starts leave the GPU alone until Splash, "
                           "macOS or the model changes"),
          "a remembered calibration of no split did not leave the GPU alone: " + ran.log);
  for (const Calibration &unusable :
       {Calibration{34, 700, gpu}, Calibration{9, AneFfn::kMinimumRows - 1, gpu},
        Calibration{9, AneFfn::kMaximumRows + 1, gpu}, Calibration{9, AneFfn::kTwoStepRows, gpu, 0, 4, 2.0},
        Calibration{4, 700, gpu, 0, 34, 2.0}, Calibration{9, 0, gpu}}) {
    fake = {};
    fake.stored = unusable;
    const Started pending = start(fake);
    require(pending.start.outcome.kind == Kind::Pending && !pending.start.calibration,
            "a start carried the unusable calibration " + unusable.text());
    const Finished calibrated = finish(fake, pending, prefillMilliseconds(fake, pending.start));
    require(calibrated.finished.outcome.kind == Kind::Split && fake.timings == 1 &&
                fake.prepared == std::pair(8u, true) && fake.remembered && fake.remembered->aneUnits == 8,
            "the remembered calibration " + unusable.text() + " was taken");
  }
}

// A given share runs the units nearest it, set up by the start, untimed, over the chunks given and, where two steps
// fit the model's layers, in two steps at the same share from 1,024 rows; it is not recalled or remembered. A given
// split of no units, or of least rows below 512 or at or past two steps' 1,024 (past 2,048 where two steps do not
// fit), is refused.
void testGivenShare() {
  Fake fake;
  Started started = start(fake, {.given = AneFfnSetting::Given{0.2}});
  require(started.start.outcome.kind == Kind::Split && !fake.timings && fake.prepared == std::pair(7u, false) &&
              fake.preparations == std::vector<Preparation>{{7, 7, 512, 1024, false}} && !fake.recalled &&
              !fake.remembered && started.start.plan && started.start.plan->breakdown().aneFfnBytes == fake.bytes(7) &&
              settingUpFirst(started.log) &&
              has(started.log, "Neural Engine FFN split at the given share 0.21 for chunks of 512 rows or more and "
                               "0.21 in two steps from 1024 rows, 2.5% from the GPU alone on the Neural Engine's part"),
          "the given share did not run as given: " + text(fake.preparations) + started.log);
  // Two steps from 1,024 rows whatever the given least rows.
  for (const uint32_t rows : {512u, 832u, 1023u}) {
    fake.preparations.clear();
    started = start(fake, {.given = AneFfnSetting::Given{0.2, rows}});
    require(started.start.outcome.kind == Kind::Split &&
                fake.preparations == std::vector<Preparation>{{7, 7, rows, 1024, false}} &&
                has(started.log, "at the given share 0.21 for chunks of " + std::to_string(rows) +
                                     " rows or more and 0.21 in two steps from 1024 rows,"),
            "the given least chunk did not run as given: " + text(fake.preparations) + started.log);
  }
  for (const AneFfnSetting::Given given :
       {AneFfnSetting::Given{0.0}, AneFfnSetting::Given{1.0}, AneFfnSetting::Given{0.2, AneFfn::kMinimumRows - 1},
        AneFfnSetting::Given{0.2, AneFfn::kTwoStepRows}})
    rejects([&] { static_cast<void>(start(fake, {.given = given})); }, "ANE FFN",
            "a given split of share " + std::to_string(given.share) + " from " + std::to_string(given.minimumRows) +
                " rows ran");
  // A model of 65 layers runs one step alone, from any least rows a function of it holds.
  fake = {};
  fake.layers = 65;
  for (const uint32_t rows : {512u, 1024u, AneFfn::kMaximumRows}) {
    fake.preparations.clear();
    started = start(fake, {.given = AneFfnSetting::Given{0.2, rows}});
    require(started.start.outcome.kind == Kind::Split &&
                fake.preparations == std::vector<Preparation>{{7, 0, rows, 0, false}} &&
                has(started.log, "at the given share 0.21 for chunks of " + std::to_string(rows) + " rows or more, ") &&
                !has(started.log, "two steps"),
            "a given split of a model two steps do not fit took them: " + text(fake.preparations) + started.log);
  }
  for (const uint32_t rows : {AneFfn::kMinimumRows - 1, AneFfn::kMaximumRows + 1})
    rejects([&] { static_cast<void>(start(fake, {.given = AneFfnSetting::Given{0.2, rows}})); },
            "ANE FFN split has no function of " + std::to_string(rows) + " rows",
            "a given split of a model two steps do not fit ran from " + std::to_string(rows) + " rows");
  // A dev tool's options: none calibrates, a share of 0 runs the GPU alone, least rows need a share, and a share
  // without them takes one step's least function.
  const AneFfnSetting calibrated = AneFfnSetting::fromGiven(std::nullopt, std::nullopt),
                      off = AneFfnSetting::fromGiven(0.0, 832), given = AneFfnSetting::fromGiven(0.3, 832);
  require(calibrated.enabled && !calibrated.given && !off.enabled && given.enabled && given.given &&
              given.given->share == 0.3 && given.given->minimumRows == 832 &&
              AneFfnSetting::fromGiven(0.3, std::nullopt).given->minimumRows == AneFfn::kMinimumRows,
          "a dev tool's options did not give their setting");
  rejects([] { static_cast<void>(AneFfnSetting::fromGiven(std::nullopt, 832)); }, "without a share",
          "least rows without a share were taken");
}

// A recalled calibration taken on a busier Mac, as its mixer shows, is calibrated again at that start: one whose mixer
// reads more than a tenth longer than the one the start's prefill gives beside the calibration's GPU alone. The real
// case, a 20-core M5 Pro's Qwen3.8-27B UD-Q4_K_M under load (11-20): its mixer 23.84 ms against 19.36 idle, the GPU
// alone's FFN 36.95 against 36.92, one step from 1,798 rows and no two steps against 0.41 from 512 and 0.56 in two from
// 1,127. Within a tenth, and on a start busier than its own, the calibration is set up as recalled, nothing timed: a
// busy start never drops a quiet calibration.
void testABusierCalibrationIsTakenAgain() {
  const Calibration underLoad{13, 1798, {{{512, 9.5659375}, {2048, 36.9545938}}}, 0, 0, 23.8388789};
  // The start's prefill of a full chunk whose mixer reads `mixer` beside the calibration's GPU alone.
  const auto prefill = [&](const Fake &fake, double mixer) { return (underLoad.gpu[1].milliseconds + mixer) * fake.layers; };
  const auto recalled = [&](double mixer) {
    struct Result {
      Fake fake;
      Started started;
      Finished finished;
    };
    auto result = std::make_unique<Result>();
    result->fake.stored = underLoad;
    result->started = start(result->fake);
    require(pending(result->fake, result->started, result->fake.largestUnits()) && result->started.start.calibration &&
                result->started.start.calibration->minimumRows == 1798,
            "a start that recalled a calibration did not leave it to the finish");
    result->finished = finish(result->fake, result->started, prefill(result->fake, mixer));
    return result;
  };
  // Idle now: the M5 Pro's idle prefill, 56.29 ms a layer, gives 19.33 ms beside the record's 36.95.
  const auto idle = recalled(36.9225625 + 19.3630671 - underLoad.gpu[1].milliseconds);
  require(has(idle->finished.log, "The Neural Engine FFN split's calibration was taken on a busier Mac (its mixer 23.8 "
                                  "ms a full chunk's layer, 19.3 now); the start calibrates it again.") &&
              idle->fake.timings == 1 && idle->fake.remembered && idle->fake.remembered->minimumRows != 1798 &&
              idle->finished.finished.outcome.kind == Kind::Split && has(idle->finished.log, "(calibrated in"),
          "a calibration taken on a busier Mac was not calibrated again: " + idle->finished.log);
  // The edge: 23.84 against 21.7, a tenth less 0.01, is set up; against 21.6 calibrated again.
  const auto within = recalled(21.7), past = recalled(21.6);
  require(!within->fake.timings && !within->fake.remembered &&
              within->fake.preparations == std::vector<Preparation>{{13, 0, 1798, 0, false}} &&
              has(within->finished.log, "set up as calibrated before, its mixer 23.8 ms a full chunk's layer, 21.7 now") &&
              past->fake.timings == 1 && has(past->finished.log, "busier Mac"),
          "a calibration within a tenth of the start's mixer was not set up, or one past it was");
  // A later start whose prefill reads as the calibrating one's sets its record up: the record keeps its mixer beside
  // its own GPU alone (the chunks' 20 ms), which the check reads beside, not beside time()'s 18.99, which would read
  // the same prefill a mixer of 2 against 0.99 and calibrate again at every start.
  Fake same;
  const double samePrefill = (18.990 + 2.0) * same.layers;
  Started first = start(same);
  static_cast<void>(finish(same, first, samePrefill));
  Started second = start(same);
  Finished again = finish(same, second, samePrefill);
  require(same.timings == 1 && again.finished.outcome.kind == Kind::Split &&
              has(again.log, "set up as calibrated before, its mixer 1.0 ms a full chunk's layer, 1.0 now"),
          "a start whose prefill read as its calibration's calibrated again: " + again.log);
  // A busier start sets the quiet calibration up.
  const auto busier = recalled(30.0);
  require(!busier->fake.timings && !busier->fake.remembered && busier->finished.finished.outcome.kind == Kind::Split &&
              has(busier->finished.log, "its mixer 23.8 ms a full chunk's layer, 30.0 now"),
          "a busier start dropped a quiet calibration: " + busier->finished.log);
}

// A start told to calibrate again (splash serve --recalibrate-ane) reads no record: one of a split, as testRemembered
// sets up without the option, and one of two refusals, which leaves the GPU alone without it, are calibrated as a start
// without a record is, and the split found replaces them, its refusals counted from none.
void testRecalibrate() {
  for (const Calibration &record :
       {Calibration{9, 700, {{{512, 5.0}, {2048, 20.0}}}}, Calibration{.refusals = Calibration::kRefusals}}) {
    Fake fake;
    fake.stored = record;
    const Ran ran = run(fake, {.recalibrate = true});
    require(!fake.recalled && !ran.started.start.calibration && ran.outcome().kind == Kind::Split &&
                fake.timings == 1 && fake.preparations == std::vector<Preparation>{{8, 0, 512, 0, true}} &&
                fake.stored && fake.stored->aneUnits == 8 && fake.stored->minimumRows == 553 &&
                !fake.stored->refusals && has(ran.log, "(calibrated in "),
            "a start told to calibrate again took the record " + record.text() + ": " + ran.log);
  }
}

} // namespace

int main() {
  try {
    testOff();
    testUnsupported();
    testRefusedContext();
    testRefusedAutomatic();
    testRefusalIsArithmetic();
    testPending();
    testFinishingAStartNotPendingThrows();
    testAutomaticContext();
    testDeterministicContext();
    testExplicitContext();
    testRelaunchContext();
    testFailures();
    testACommandOutOfMemoryIsCommandMemory();
    testNoGain();
    testTheMixerComesFromThePrefill();
    testTwoStepsFromFixedRows();
    testTwoStepsOnlyWhereTheyFit();
    testTwoStepsStandAlone();
    testRefitMovesTheChoice();
    testTheRefitKeepsBothShares();
    testARefittedSplitThatFailsKeepsTheFirst();
    testAFirstSplitThatDoesNotPrepareAgainIsRemembered();
    testARefusedSplitIsNotCalibratedInALoop();
    testAFailureThatIsNotARefusalKeepsTheCalibration();
    testRemembered();
    testABusierCalibrationIsTakenAgain();
    testRecalibrate();
    testGivenShare();
    std::cout << "ane ffn startup tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ane ffn startup tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
