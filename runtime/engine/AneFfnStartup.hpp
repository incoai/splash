#pragma once

#include "engine/MemoryPlan.hpp"
#include "ops/AneFfn.hpp"
#include "ops/AneFfnCalibration.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace splash::engine {

// Whether the prefill FFN's Neural Engine split may run (splash serve
// --disable-ane turns it off), and the split a dev tool runs instead of
// calibrating one.
struct AneFfnSetting final {
  // The ANE takes the whole channel units nearest `share` of them
  // (ops::ane_ffn::nearestUnits), over chunks of `minimumRows` rows or more.
  struct Given final {
    double share = 0.0;
    uint32_t minimumRows = ops::AneFfn::kMinimumRows; // one step's least function, as calibration's
  };
  bool enabled = true;
  std::optional<Given> given;

  // The setting of a dev tool's --ane-ffn-share and --ane-ffn-minimum-rows:
  // none calibrates the split, and a share of 0 runs the GPU alone. Throws
  // for least rows without a share.
  [[nodiscard]] static AneFfnSetting fromGiven(std::optional<double> share, std::optional<uint32_t> minimumRows);
};

// The split of a model, prepared: the split, taking the commands it was
// prepared for; the chunks of its share of one step as timed, if they were;
// and what AneFfn::verify returned.
struct AneFfnPrepared final {
  std::unique_ptr<ops::AneFfn> split;
  ops::ane_ffn::ChunkTimings chunks;
  double error = 0.0;
};

// The model's side of the split, as functions, so that a test can stand in
// for it (engine::aneFfnModel() in RuntimeResources.hpp).
struct AneFfnModel final {
  // Whether the target's FFN layers are all dense, which the split could
  // take; a MoE target's are not.
  bool dense = false;
  // Why the split does not take them; empty when it does.
  std::string unsupported;
  // Why this Mac cannot run the split (ane::unavailable), asked only of a
  // split that may run; none when it can.
  std::function<std::optional<std::string>()> unavailable;
  // The channel units of the split (ops::AneFfn::units), and its FFN layers,
  // one a layer of the target.
  uint32_t units = 0, layers = 0;
  // The split's Metal memory with the ANE taking at most `aneUnits` of them
  // in either kind of command.
  std::function<uint64_t(uint32_t aneUnits)> plannedBytes;
  // Calibration's timings (ops::ane_ffn::Measurement::time), all but the
  // mixer, and the same of the split with the ANE taking `aneUnits`, on the
  // layers they timed, each beside the GPU alone read right before it.
  std::function<ops::ane_ffn::Timings()> time;
  std::function<ops::ane_ffn::Timing(uint32_t aneUnits)> timeAt;
  // The split that serves with the ANE taking `shares` in commands of
  // `rows`, verified, with the chunks of its share of one step timed on its
  // program first (ops::ane_ffn::Measurement::chunks) when `timeChunks`. A
  // model a test plays prepares no split.
  std::function<AneFfnPrepared(ops::AneFfn::Shares shares, ops::AneFfn::Rows rows, bool timeChunks)> prepare;
  // This Mac's calibration of the split within `maxAneUnits` units, if one
  // is remembered; and remembering one, true once kept.
  std::function<std::optional<ops::ane_ffn::Calibration>(uint32_t maxAneUnits)> recall;
  std::function<bool(uint32_t maxAneUnits, const ops::ane_ffn::Calibration &calibration)> remember;
  // Whether the Metal backend still serves.
  std::function<bool()> healthy;
};

// The context a start asks the split to leave: --max-context's, or with
// `relaunch` a relaunch's, the context the server's first engine served
// (RuntimeResourcesConfig::relaunchContext). Zero tokens ask for none: the
// start serves the automatic context (AneFfnOutcome::context).
struct RequestedContext final {
  uint32_t tokens = 0;
  bool relaunch = false;
};

// How a start's split came out: off as given, not taken by the model or this
// Mac, refused by the memory plan, gaining too little, failed, running, or
// set up once the engine's runtime has prefilled a full chunk on the GPU
// alone (finishAneFfn), and why or how. Startup logs it (startAneFfn,
// finishAneFfn).
struct AneFfnOutcome final {
  enum class Kind : uint8_t { Off, Unsupported, Refused, NoGain, Unavailable, Split, Pending };
  Kind kind = Kind::Off;
  std::string reason;
  // The context the engine serves when no context is asked for, and the one
  // the memory plan holds without the split. While the split may run, the
  // first is what the plan holds with the largest split the model could take,
  // whatever the start makes of it, so that every start of the model on this
  // Mac serves the same; once this Mac's calibration found no gain, the GPU
  // alone's. Otherwise both are the same. For a split refused a context asked
  // for, the first is the most context a split leaves instead.
  // The split's bytes the plan of the first holds, none where it is the GPU
  // alone's.
  uint32_t context = 0, contextWithout = 0;
  uint64_t contextSplitBytes = 0;
};
// The kind's name in lowercase, as "no_gain".
[[nodiscard]] std::string_view aneFfnOutcomeName(AneFfnOutcome::Kind kind) noexcept;

// The prefill FFN's Neural Engine split (RuntimeResources::aneFfnSnapshot).
struct AneFfnSnapshot {
  // Off: the start left the GPU alone; Split: the split serves; Stopped: it
  // stopped while serving, and the GPU runs alone until the engine restarts.
  enum class State : uint8_t { Off, Split, Stopped };
  State state = State::Off;
  // The start's outcome (AneFfnOutcome), or why the split stopped.
  std::string reason;
  // The split's share and least chunk rows; 0 when off.
  double share = 0.0;
  uint32_t minimumRows = 0;
  // What it ran (ops::AneFfn::Served).
  uint64_t commands = 0, evaluations = 0;
  double aneMilliseconds = 0.0;
};

// The split a start runs and the memory plan that sets its memory aside,
// both none without one, and the outcome. A start whose split is Pending
// holds the largest split's memory and sets the split up after its
// runtime's prefill of a full chunk (finishAneFfn): within `maxAneUnits`, the
// units the plan holds, from this Mac's `calibration` of the model as
// recalled, if one.
struct AneFfnStart final {
  std::unique_ptr<ops::AneFfn> split;
  std::optional<EngineMemoryPlan> plan;
  AneFfnOutcome outcome;
  uint32_t maxAneUnits = 0;
  std::optional<ops::ane_ffn::Calibration> calibration;
  // The context the start was asked to leave, which the log line weighs.
  RequestedContext requested;
};

// The prefill FFN's Neural Engine split of `model` as `setting` allows,
// within the memory plan `planMemory` makes with the split's bytes set aside:
// one whose plan still holds the context `requested` asks for, or with none
// the automatic context (AneFfnOutcome::context). Every step is arithmetic
// but a given split's, which is set up here and takes the given least chunk,
// no breaker, and nothing remembered. Where this Mac's calibration of the
// model, within the units the plan holds, found no gain or the Neural Engine
// refused its split twice, the GPU runs alone; otherwise the split is
// Pending: the plan holds the most units, and finishAneFfn sets it up. Logs
// the outcome, a split a relaunch's context refuses in a plain line: the
// server, not the user, asked for that context. A failure leaves the GPU
// alone, unless `cancelled` returns true, the wait for the ANE was
// interrupted or the backend no longer serves, which the failure's exception
// reports. Throws std::invalid_argument for a given split of no units or of
// least rows no function holds.
[[nodiscard]] AneFfnStart
startAneFfn(const AneFfnModel &model, const AneFfnSetting &setting, RequestedContext requested,
            const std::function<EngineMemoryPlanResult(uint64_t aneFfnBytes)> &planMemory,
            const std::function<bool()> &cancelled);

// The split a start left Pending (startAneFfn), once the engine's runtime has
// prefilled a full chunk on the GPU alone in `prefillMilliseconds` of GPU
// time, and how it came out, which it logs; the context and plan stay the
// start's. It takes this Mac's calibration of the model, as recalled, unless
// the calibration's mixer reads more than a tenth longer than the one this
// start's prefill gives: one taken on a busier Mac. Else it is calibrated
// (AneFfnStartup.cpp), the mixer the rest of that prefill's layer, and the
// outcome remembered, a split or none. The split judges itself against the
// GPU alone as calibrated (ops::ane_ffn::Breaker), and a split that stops for
// losing keeps the calibration; the Neural Engine's refusals count
// (ops::ane_ffn::Calibration). Fails as startAneFfn does.
struct AneFfnFinished final {
  std::unique_ptr<ops::AneFfn> split;
  AneFfnOutcome outcome;
};
[[nodiscard]] AneFfnFinished finishAneFfn(const AneFfnModel &model, const AneFfnStart &start,
                                          double prefillMilliseconds, const std::function<bool()> &cancelled);

// Whether startAneFfn would find the split `setting` lets `model` run refused
// by the plan `planMemory` makes: the split may run on this Mac, and even its
// least share leaves less than `requestedContextTokens`, or with zero no
// context. Nothing is timed, set up or logged, so a start can weigh other
// memory the plan sets aside against the split first.
[[nodiscard]] bool aneFfnRefused(const AneFfnModel &model, const AneFfnSetting &setting,
                                 uint32_t requestedContextTokens,
                                 const std::function<EngineMemoryPlanResult(uint64_t aneFfnBytes)> &planMemory);

} // namespace splash::engine
