#include "engine/AneFfnStartup.hpp"

#include "AwakeClock.hpp"
#include "StderrLine.hpp"
#include "ane/Program.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::engine {
namespace {

namespace ane_ffn = ops::ane_ffn;
using Calibration = ane_ffn::Calibration;
using Kind = AneFfnOutcome::Kind;

// How much longer a recalled calibration's mixer may read than the one a
// start's prefill gives, beside the calibration's GPU alone's FFN layer,
// before the start calibrates again: a Mac busy beside a calibration reads
// its mixer long (a 20-core M5 Pro's Qwen3.8-27B UD-Q4_K_M 23.84 ms under
// load against 19.36 idle, +23%), or its GPU alone's FFN, beside which the
// start's mixer then reads shorter still; it repeats within 0.7% from one
// start to the next.
constexpr double kBusier = 0.1;

// `tokens` with its thousands grouped, as the server prints a context.
std::string grouped(uint32_t tokens) {
  std::string digits = std::to_string(tokens);
  for (size_t at = digits.size(); at > 3; at -= 3) digits.insert(at - 3, ",");
  return digits;
}

uint32_t contextOf(const EngineMemoryPlanResult &result) {
  return result.plan ? result.plan->maximumContextTokens() : 0;
}

// Why the split does not run on `model` as `setting` allows, whatever the
// memory plan holds: off as given, or not taken by the model or this Mac; none
// when it may run.
std::optional<std::pair<Kind, std::string>> notRunning(const AneFfnModel &model, const AneFfnSetting &setting) {
  if (!model.dense) return std::pair{Kind::Unsupported, std::string("the target's FFN layers are not all dense")};
  if (!setting.enabled) return std::pair{Kind::Off, std::string("as given")};
  if (!model.unsupported.empty()) return std::pair{Kind::Unsupported, model.unsupported};
  if (std::optional<std::string> reason = model.unavailable()) return std::pair{Kind::Unsupported, std::move(*reason)};
  return std::nullopt;
}

// The units of the least split `setting` lets `model` take: the given share's,
// else one.
uint32_t leastUnits(const AneFfnModel &model, const AneFfnSetting &setting) {
  return setting.given ? ane_ffn::nearestUnits(setting.given->share, model.units) : 1;
}

// Logs `outcome` of a start of `model` as `setting` allows: a model whose FFN
// layers are not all dense and one --disable-ane leaves alone without its
// layers said, and a refusal of a context the GPU alone does not hold either,
// which the start reports on its own (ServingRuntime::start), log nothing. A
// refusal of a relaunch's context is a plain line, without the way out the
// warning of a refused --max-context gives. An automatic context below the GPU
// alone's is named beside it.
void logOutcome(const AneFfnOutcome &outcome, const AneFfnModel &model, const AneFfnSetting &setting,
                RequestedContext requested) {
  const std::string context = !requested.tokens && outcome.context != outcome.contextWithout
                                  ? "; context " + grouped(outcome.context) + " tokens (" +
                                        grouped(outcome.contextWithout) + " with --disable-ane)"
                                  : "";
  switch (outcome.kind) {
  case Kind::Off:
    if (model.dense && model.unsupported.empty()) logLine("The GPU runs the prefill FFN alone, ", outcome.reason, ".");
    return;
  case Kind::Unsupported:
    if (model.dense) logLine("The GPU runs the prefill FFN alone: ", outcome.reason, ".");
    return;
  case Kind::NoGain:
    logLine("The GPU runs the prefill FFN alone: ", outcome.reason, context, ".");
    return;
  case Kind::Refused:
    if (requested.tokens > outcome.contextWithout) return;
    if (requested.relaunch)
      logLine("Neural Engine FFN split off: ", outcome.reason, ".");
    else if (requested.tokens && outcome.context)
      logWarning("Neural Engine FFN split off: ", outcome.reason, ", and ",
                 setting.given ? "the given share" : "any split", " leaves at most ", grouped(outcome.context),
                 " tokens; pass --max-context ", outcome.context,
                 " or less to run it, or --disable-ane, which also silences this line.");
    else
      logWarning("Neural Engine FFN split off: ", outcome.reason, "; --disable-ane silences this line.");
    return;
  case Kind::Unavailable:
    logWarning("Neural Engine FFN split unavailable (", outcome.reason, "); the GPU runs the prefill FFN alone",
               context, ". --disable-ane silences this line.");
    return;
  case Kind::Split:
    logLine("Neural Engine FFN split ", outcome.reason, context, ".");
    return;
  case Kind::Pending:
    return;
  }
}

// Milliseconds as a log line prints them.
std::string fixed(double value) {
  std::ostringstream text;
  text << std::fixed << std::setprecision(1) << value;
  return text.str();
}

// What a start of a split logs of its time: since `started`, in seconds.
std::string secondsSince(AwakeClock::time_point started) {
  std::ostringstream text;
  text << std::fixed << std::setprecision(1) << std::chrono::duration<double>(AwakeClock::now() - started).count()
       << " s";
  return text.str();
}

// How the split of `shares` in commands of `rows` came out, set up `how` with
// what calibration `timed` of it and the error verify() found, for a model of
// `units`: "at share ... (how in seconds)".
std::string splitReason(const AneFfnPrepared &prepared, ops::AneFfn::Shares shares, ops::AneFfn::Rows rows,
                        uint32_t units, bool given, const std::string &timed, const std::string &how) {
  std::ostringstream reason;
  reason << std::fixed << std::setprecision(2) << "at " << (given ? "the given share " : "share ");
  if (rows.oneStep)
    reason << static_cast<double>(shares.oneStep) / units << " for chunks of " << rows.oneStep << " rows or more";
  if (rows.oneStep && rows.twoSteps) reason << " and ";
  if (rows.twoSteps)
    reason << static_cast<double>(shares.twoSteps) / units << " in two steps from " << rows.twoSteps << " rows";
  reason << timed << ", " << fixed(100.0 * prepared.error) << "% from the GPU alone on the Neural Engine's part ("
         << how << ")";
  return reason.str();
}

// Why `step` failed on a backend that serves and in a start that goes on, or
// none, and in `refused` whether the Neural Engine refused it; a cancelled
// start, an interrupted wait for the ANE or a backend that stopped rethrows.
std::optional<std::string> failed(const AneFfnModel &model, const std::function<bool()> &cancelled,
                                  const std::function<void()> &step, bool &refused) {
  try {
    step();
    return std::nullopt;
  } catch (const ane::Interrupted &) {
    throw;
  } catch (const std::exception &error) {
    if ((cancelled && cancelled()) || !model.healthy()) throw;
    refused = dynamic_cast<const ane::Refused *>(&error);
    return error.what();
  }
}

// The rows of the commands of each kind a split of a record's takes: one
// step from its least rows where it has them, two from kTwoStepRows where it
// has a share of two.
ops::AneFfn::Rows rowsOf(const Calibration &record) {
  return {record.minimumRows, record.twoStepUnits ? ops::AneFfn::kTwoStepRows : 0};
}

// A record a start can take: units the plan holds, and least rows of one
// step its functions hold, below two steps' where it has a share of two, or
// none beside two steps.
bool takeable(const Calibration &record, uint32_t maxAneUnits) {
  const ops::AneFfn::Shares shares{record.aneUnits, record.twoStepUnits};
  return shares.most() <= maxAneUnits &&
         (!record.aneUnits || (!record.minimumRows && record.twoStepUnits) ||
          (record.minimumRows >= ops::AneFfn::kMinimumRows &&
           record.minimumRows < (record.twoStepUnits ? ops::AneFfn::kTwoStepRows : ops::AneFfn::kMaximumRows + 1)));
}

AneFfnStart decide(const AneFfnModel &model, const AneFfnSetting &setting, RequestedContext requested,
                   const std::function<EngineMemoryPlanResult(uint64_t)> &planMemory,
                   const std::function<bool()> &cancelled) {
  const uint32_t without = contextOf(planMemory(0));
  const auto gpuAlone = [&](Kind kind, std::string reason, uint32_t context, uint64_t contextSplitBytes = 0) {
    AneFfnStart start;
    start.outcome = {kind, std::move(reason), context, without, contextSplitBytes};
    start.requested = requested;
    return start;
  };
  if (std::optional<std::pair<Kind, std::string>> off = notRunning(model, setting))
    return gpuAlone(off->first, std::move(off->second), without);
  const std::optional<uint32_t> given =
      setting.given ? std::optional(ane_ffn::nearestUnits(setting.given->share, model.units)) : std::nullopt;
  // Two steps where a command of them queues no more evaluations than the
  // Neural Engine holds in flight.
  const bool twoSteps = ops::AneFfn::fits(model.layers, ops::AneFfn::Steps::Two);
  if (setting.given && (setting.given->minimumRows < ops::AneFfn::kMinimumRows ||
                        setting.given->minimumRows >= (twoSteps ? ops::AneFfn::kTwoStepRows : ops::AneFfn::kMaximumRows + 1)))
    throw std::invalid_argument("ANE FFN split has no function of " + std::to_string(setting.given->minimumRows) +
                                " rows");

  // Feasibility. The split's memory comes out of the KV cache, more of it
  // with each unit the ANE takes: the plan must still hold the context asked
  // for, or with none asked some context, which becomes the automatic one.
  const auto planWith = [&](uint32_t aneUnits) { return planMemory(model.plannedBytes(aneUnits)); };
  const auto contextWith = [&](uint32_t aneUnits) { return contextOf(planWith(aneUnits)); };
  // The most units from `low`, whose plan holds `context`, to `high` whose
  // plan holds it.
  const auto most = [&](uint32_t low, uint32_t high, uint32_t context) {
    while (low < high) {
      const uint32_t middle = low + (high - low + 1) / 2;
      if (contextWith(middle) >= context)
        low = middle;
      else
        high = middle - 1;
    }
    return low;
  };
  const uint32_t least = leastUnits(model, setting);
  if (const uint32_t leaves = contextWith(least); leaves < std::max<uint32_t>(requested.tokens, 1)) {
    if (!requested.tokens) return gpuAlone(Kind::Refused, "it leaves no memory for context", without);
    const std::string asked = requested.relaunch
                                  ? "the first engine's context of " + grouped(requested.tokens) + " tokens"
                                  : "--max-context " + std::to_string(requested.tokens);
    return gpuAlone(Kind::Refused,
                    requested.tokens > without ? "the GPU alone does not hold " + asked + " either"
                                               : asked + " cannot be held with it",
                    leaves);
  }
  // The most units whose plan holds the context asked for, or with none
  // asked the most whose plan holds any.
  const uint32_t maxAneUnits =
      given ? least : most(least, model.units - 1, std::max<uint32_t>(requested.tokens, 1));
  // A split the start sets up: the plan with its units, and the automatic
  // context, what that plan holds.
  const auto planned = [&](uint32_t aneUnits, Kind kind, std::string reason) {
    EngineMemoryPlanResult plan = planWith(aneUnits);
    if (!plan.plan) throw std::logic_error("the memory plan does not hold the split it chose");
    AneFfnStart start;
    start.outcome = {kind, std::move(reason), contextOf(plan), without, model.plannedBytes(aneUnits)};
    start.plan = std::move(plan.plan);
    start.requested = requested;
    return start;
  };

  // A given share takes commands of one step from the given rows and of two,
  // at the same share, from kTwoStepRows, as a calibrated split does.
  if (given) {
    logLine("Setting up the Neural Engine FFN split (splash serve --disable-ane keeps the FFN on the GPU).");
    const auto started = AwakeClock::now();
    const ops::AneFfn::Rows rows{setting.given->minimumRows, twoSteps ? ops::AneFfn::kTwoStepRows : 0};
    const ops::AneFfn::Shares shares{*given, twoSteps ? *given : 0};
    AneFfnPrepared prepared;
    bool refused = false;
    if (const std::optional<std::string> reason =
            failed(model, cancelled, [&] { prepared = model.prepare(shares, rows, false); }, refused))
      return gpuAlone(Kind::Unavailable, *reason, contextWith(*given), model.plannedBytes(*given));
    AneFfnStart start = planned(*given, Kind::Split,
                                splitReason(prepared, shares, rows, model.units, true, "",
                                            "set up in " + secondsSince(started)));
    start.split = std::move(prepared.split);
    return start;
  }
  // This Mac's calibration of the model within the units the plan holds: of
  // no split, refused twice, the GPU alone; refused once, calibrated again
  // (ane_ffn::Calibration); none recalled where the setting calibrates again.
  std::optional<Calibration> calibration = setting.recalibrate ? std::nullopt : model.recall(maxAneUnits);
  if (calibration && !takeable(*calibration, maxAneUnits)) calibration.reset();
  if (calibration && !calibration->aneUnits && calibration->refusals != 1)
    return gpuAlone(Kind::Unavailable, calibration->withoutSplit(), without);
  AneFfnStart start = planned(maxAneUnits, Kind::Pending, "set up once the engine has prefilled a full chunk");
  start.maxAneUnits = maxAneUnits;
  start.calibration = std::move(calibration);
  return start;
}

} // namespace

std::string_view aneFfnOutcomeName(AneFfnOutcome::Kind kind) noexcept {
  switch (kind) {
  case Kind::Off:
    return "off";
  case Kind::Unsupported:
    return "unsupported";
  case Kind::Refused:
    return "refused";
  case Kind::NoGain:
    return "no_gain";
  case Kind::Unavailable:
    return "unavailable";
  case Kind::Split:
    return "split";
  case Kind::Pending:
    return "pending";
  }
  return "unknown";
}

AneFfnSetting AneFfnSetting::fromGiven(std::optional<double> share, std::optional<uint32_t> minimumRows) {
  if (!share) {
    if (minimumRows) throw std::invalid_argument("ANE FFN least chunk rows given without a share");
    return {};
  }
  if (*share == 0.0) return {.enabled = false};
  return {.given = Given{*share, minimumRows.value_or(Given{}.minimumRows)}};
}

AneFfnStart startAneFfn(const AneFfnModel &model, const AneFfnSetting &setting, RequestedContext requested,
                        const std::function<EngineMemoryPlanResult(uint64_t aneFfnBytes)> &planMemory,
                        const std::function<bool()> &cancelled) {
  AneFfnStart start = decide(model, setting, requested, planMemory, cancelled);
  logOutcome(start.outcome, model, setting, requested);
  return start;
}

AneFfnFinished finishAneFfn(const AneFfnModel &model, const AneFfnStart &start, double prefillMilliseconds,
                            const std::function<bool()> &cancelled) {
  if (start.outcome.kind != Kind::Pending) throw std::logic_error("the Neural Engine split was not left to set up");
  const uint32_t maxAneUnits = start.maxAneUnits;
  const std::optional<Calibration> &calibration = start.calibration;
  logLine("Setting up the Neural Engine FFN split (splash serve --disable-ane keeps the FFN on the GPU).");
  const auto started = AwakeClock::now();
  // How it came out, which the start's context and plan keep.
  const auto finished = [&](Kind kind, std::string reason, std::unique_ptr<ops::AneFfn> split = {}) {
    AneFfnFinished result{std::move(split), start.outcome};
    result.outcome.kind = kind;
    result.outcome.reason = std::move(reason);
    logOutcome(result.outcome, model, {}, start.requested);
    return result;
  };
  // What the next start does once this one tried to remember `record`.
  const auto keep = [&](const Calibration &record) {
    return Calibration::nextStart(model.remember(maxAneUnits, record) ? record : calibration);
  };
  const auto sharesOf = [](const Calibration &record) {
    return ops::AneFfn::Shares{record.aneUnits, record.twoStepUnits};
  };
  // The split takes the calibration's least chunk, and judges itself against
  // the GPU alone as calibrated.
  const auto take = [&](AneFfnPrepared &prepared, const Calibration &record) {
    if (!prepared.split) return;
    prepared.split->setRows(rowsOf(record));
    prepared.split->setBreaker(ane_ffn::Breaker(record.gpu, record.mixer));
  };
  const auto serve = [&](AneFfnPrepared prepared, const Calibration &record, const std::string &timed,
                         const std::string &how) {
    std::string reason = splitReason(prepared, sharesOf(record), rowsOf(record), model.units, false, timed,
                                     how + " in " + secondsSince(started));
    return finished(Kind::Split, std::move(reason), std::move(prepared.split));
  };
  bool refusal = false;
  const auto failure = [&](const std::function<void()> &step) { return failed(model, cancelled, step, refusal); };
  try {
    // The mixer: the rest of the prefill's layer of a full chunk on the GPU
    // alone, beside its FFN layer `gpuAlone`.
    const auto mixerBeside = [&](double gpuAlone) {
      return std::max(0.0, prefillMilliseconds / model.layers - gpuAlone);
    };
    // A recalled calibration serves unless it was taken on a busier Mac than
    // this one is now: a quieter one serves however busy this start is.
    if (calibration && calibration->aneUnits) {
      const double now = mixerBeside(calibration->gpu[1].milliseconds);
      const std::string mixers = "its mixer " + fixed(calibration->mixer) + " ms a full chunk's layer, " +
                                 fixed(now) + " now";
      if (calibration->mixer <= (1.0 + kBusier) * now) {
        // The Neural Engine's refusal counts in the split's place; another
        // failure keeps the calibration.
        AneFfnPrepared prepared;
        if (const auto reason =
                failure([&] { prepared = model.prepare(sharesOf(*calibration), rowsOf(*calibration), false); }))
          return finished(Kind::Unavailable, "its remembered share's split was not prepared: " + *reason + "; " +
                                                 keep(refusal ? ane_ffn::refused(calibration) : *calibration));
        take(prepared, *calibration);
        return serve(std::move(prepared), *calibration, "", "set up as calibrated before, " + mixers);
      }
      logLine("The Neural Engine FFN split's calibration was taken on a busier Mac (", mixers,
              "); the start calibrates it again.");
    }

    ane_ffn::Timings timings = model.time();
    timings.mixer = mixerBeside(timings.gpuAlone);
    const auto choice = [&](const ane_ffn::Model &fitted, double alone) {
      return ane_ffn::choose(fitted, alone, model.units, maxAneUnits);
    };
    const std::string layer = " ms per " + std::to_string(ops::AneFfn::kMaximumRows) + "-row FFN layer";
    // No split gains enough: the start keeps nothing of it, so the next one
    // calibrates again (ane_ffn::Calibration).
    const auto noGain = [&](std::string reason) {
      return finished(Kind::NoGain, std::move(reason) + " (calibrated in " + secondsSince(started) +
                                        "; the next start calibrates again)");
    };
    const ane_ffn::Model fitted = ane_ffn::fit(timings.low, timings.high);
    const std::optional<uint32_t> first = choice(fitted, timings.gpuAlone);
    // Two steps from the same fit, whatever one step's: the ANE's evaluation
    // of a half beside the mixer of the other (ane_ffn::twoSteps), against
    // the GPU alone's whole layer.
    const ane_ffn::Model pipelined = ane_ffn::twoSteps(fitted, timings);
    const std::optional<uint32_t> two = ops::AneFfn::fits(model.layers, ops::AneFfn::Steps::Two)
                                            ? choice(pipelined, timings.gpuAlone + timings.mixer)
                                            : std::nullopt;
    if (!first && !two)
      return noGain("no Neural Engine split beats its " + fixed(timings.gpuAlone) + layer + " by enough");
    // One step's units name the program, two steps' where one step's fit
    // chooses none: its one-step functions then serve where their chunks gain.
    const uint32_t units = first.value_or(*two);
    // One program serves both kinds of command (ops::AneFfn::programRows),
    // its one-step functions timed on it before its split is made.
    const auto sharesAt = [&](uint32_t aneUnits) { return ops::AneFfn::Shares{aneUnits, two.value_or(0)}; };
    const ops::AneFfn::Rows anyRows{ops::AneFfn::kMinimumRows, two ? 2 * ops::AneFfn::kMinimumRows : 0};
    const auto timedSplit = [&](uint32_t aneUnits) {
      AneFfnPrepared split = model.prepare(sharesAt(aneUnits), anyRows, true);
      if (split.chunks.functions.empty()) throw std::logic_error("the split's chunks were not timed");
      return split;
    };
    // One step, as a split of one share calibrates. The split's full chunk at
    // the first units, timed nearest T's least as the fit's were and on their
    // layers, before their split is prepared, so no two splits are held at
    // once, fits u again (ane_ffn::refit); the units that model chooses serve
    // where their split is prepared and takes no chunk longer
    // (ane_ffn::slowerNowhere), else the first ones, whose split prepared: a
    // refit whose split cannot be timed or prepared, on a backend that serves
    // and in a start that goes on, says nothing against them.
    const auto refitting = [&](const char *failed, const auto &step) {
      if (const std::optional<std::string> reason = failure(step))
        logLine("The Neural Engine FFN split's refitted share was not ", failed, " (", *reason,
                "); its first share serves.");
    };
    std::optional<uint32_t> refitted;
    if (first)
      refitting("found", [&] {
        const ane_ffn::Timing at = model.timeAt(units);
        refitted = choice(ane_ffn::refit(fitted, at.share, at.split), at.alone);
      });
    AneFfnPrepared prepared = timedSplit(units);
    uint32_t chosen = units;
    std::optional<std::string> unprepared;
    if (refitted && *refitted != units) {
      const ane_ffn::ChunkTimings firstChunks = std::move(prepared.chunks);
      prepared.split.reset();
      std::optional<AneFfnPrepared> candidate;
      refitting("prepared", [&] { candidate = timedSplit(*refitted); });
      if (candidate && ane_ffn::slowerNowhere(candidate->chunks, firstChunks)) {
        prepared = std::move(*candidate);
        chosen = *refitted;
      } else {
        candidate.reset();
        unprepared = failure([&] { prepared = model.prepare(sharesAt(units), anyRows, false); });
        prepared.chunks = firstChunks;
      }
    }
    // The largest one-step chunk timed, against the GPU alone's line.
    const ane_ffn::ChunkTimings &chunks = prepared.chunks;
    const ane_ffn::ChunkTimings::Function &largest = chunks.functions.front();
    const auto &[low, high] = chunks.gpu;
    const double alone = low.milliseconds + (high.milliseconds - low.milliseconds) * (largest.rows - low.rows) /
                                                (high.rows - low.rows);
    const std::string timed = fixed(largest.split) + " ms per " + std::to_string(largest.rows) +
                              "-row FFN layer against " + fixed(alone) + " on the GPU alone";
    // Each kind where it gains: one step from its least rows, below the
    // commands two steps take, and two steps from theirs, alone where one
    // step splits no chunk below them; the GPU alone where neither gains.
    std::optional<uint32_t> least = ane_ffn::minimumRows(chunks);
    if (two && least && *least >= ops::AneFfn::kTwoStepRows) least.reset();
    if (!least && !two) return noGain("no chunk of the Neural Engine split beats it by enough, " + timed);
    // The record keeps the mixer beside its own GPU alone's full chunk, as a
    // later start reads its own beside it.
    const Calibration made{chosen, least.value_or(0), chunks.gpu, calibration ? calibration->refusals : 0,
                           two.value_or(0), mixerBeside(chunks.gpu[1].milliseconds)};
    if (unprepared)
      return finished(Kind::Unavailable, "its calibrated share's split was not prepared again: " + *unprepared + "; " +
                                             keep(refusal ? ane_ffn::refused(calibration, made) : made));
    take(prepared, made);
    const std::string pipelinedTimed =
        two ? "; two steps " + fixed(pipelined(static_cast<double>(*two) / model.units)) + " ms per " +
                  std::to_string(ops::AneFfn::kMaximumRows) + "-row layer by the fit against " +
                  fixed(timings.gpuAlone + timings.mixer)
            : std::string();
    AneFfnFinished result = serve(std::move(prepared), made, ": " + timed + pipelinedTimed, "calibrated");
    model.remember(maxAneUnits, made);
    return result;
  } catch (const ane::Interrupted &) {
    throw;
  } catch (const std::exception &error) {
    if ((cancelled && cancelled()) || !model.healthy()) throw;
    if (!dynamic_cast<const ane::Refused *>(&error)) return finished(Kind::Unavailable, error.what());
    return finished(Kind::Unavailable, error.what() + ("; " + keep(ane_ffn::refused(calibration))));
  }
}

bool aneFfnRefused(const AneFfnModel &model, const AneFfnSetting &setting, uint32_t requestedContextTokens,
                   const std::function<EngineMemoryPlanResult(uint64_t aneFfnBytes)> &planMemory) {
  return !notRunning(model, setting) &&
         contextOf(planMemory(model.plannedBytes(leastUnits(model, setting)))) <
             std::max<uint32_t>(requestedContextTokens, 1);
}

} // namespace splash::engine
