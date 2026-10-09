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
using Kind = AneFfnOutcome::Kind;

// `tokens` with its thousands grouped, as the server prints a context.
std::string grouped(uint32_t tokens) {
  std::string digits = std::to_string(tokens);
  for (size_t at = digits.size(); at > 3; at -= 3) digits.insert(at - 3, ",");
  return digits;
}

uint32_t contextOf(const EngineMemoryPlanResult &result) {
  return result.plan ? result.plan->maximumContextTokens() : 0;
}

// Logs `outcome` of a start of `model` as `setting` allows: a model without
// dense FFN layers and one --disable-ane leaves alone without its layers said, and
// a refusal of a context the GPU alone does not hold either, which fails the
// start on its own, log nothing. An automatic context below the GPU alone's
// is named beside it.
void logOutcome(const AneFfnOutcome &outcome, const AneFfnModel &model, const AneFfnSetting &setting,
                uint32_t requestedContextTokens) {
  const std::string context = !requestedContextTokens && outcome.context != outcome.contextWithout
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
    if (requestedContextTokens > outcome.contextWithout) return;
    if (requestedContextTokens && outcome.context)
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
  }
}

AneFfnStart decide(const AneFfnModel &model, const AneFfnSetting &setting, uint32_t requestedContextTokens,
                   const std::function<EngineMemoryPlanResult(uint64_t)> &planMemory,
                   const std::function<bool()> &cancelled) {
  const uint32_t without = contextOf(planMemory(0));
  const auto gpuAlone = [&](Kind kind, std::string reason, uint32_t context) {
    return AneFfnStart{{}, {}, {kind, std::move(reason), context, without}};
  };
  if (!model.dense) return gpuAlone(Kind::Unsupported, "the target has no dense FFN layers", without);
  if (!setting.enabled) return gpuAlone(Kind::Off, "as given", without);
  if (!model.unsupported.empty()) return gpuAlone(Kind::Unsupported, model.unsupported, without);
  if (const std::optional<std::string> reason = model.unavailable())
    return gpuAlone(Kind::Unsupported, *reason, without);
  const std::optional<uint32_t> given =
      setting.given ? std::optional(ane_ffn::nearestUnits(setting.given->share, model.units)) : std::nullopt;
  if (setting.given && (setting.given->minimumRows < ops::AneFfn::kMinimumRows ||
                        setting.given->minimumRows > ops::AneFfn::kMaximumRows))
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
  const uint32_t least = given.value_or(1);
  if (const uint32_t leaves = contextWith(least); leaves < std::max<uint32_t>(requestedContextTokens, 1)) {
    if (!requestedContextTokens) return gpuAlone(Kind::Refused, "it leaves no memory for context", without);
    return gpuAlone(Kind::Refused,
                    requestedContextTokens > without
                        ? "the GPU alone does not hold --max-context " + std::to_string(requestedContextTokens) +
                              " either"
                        : "--max-context " + std::to_string(requestedContextTokens) + " cannot be held with it",
                    leaves);
  }
  // The most units whose plan holds the context asked for, or with none
  // asked the most whose plan holds any.
  const uint32_t maxAneUnits =
      given ? least : most(least, model.units - 1, std::max<uint32_t>(requestedContextTokens, 1));
  // The automatic context of a start whose split takes `aneUnits`, the
  // GPU alone's for none: what the plan holds with them.
  const auto automaticWith = [&](uint32_t aneUnits) { return aneUnits ? contextWith(aneUnits) : without; };
  // The units the automatic context assumes: this Mac's calibration once
  // remembered, so that every start that takes it serves the same context,
  // whatever it makes of the split, and until then the most.
  uint32_t assumed = maxAneUnits;
  const auto gpuAloneServing = [&](Kind kind, std::string reason) {
    return gpuAlone(kind, std::move(reason), automaticWith(assumed));
  };

  logLine("Setting up the Neural Engine FFN split (splash serve --disable-ane keeps the FFN on the GPU).");
  const auto started = AwakeClock::now();
  const auto seconds = [&] {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << std::chrono::duration<double>(AwakeClock::now() - started).count()
         << " s";
    return text.str();
  };
  const std::string layer = " ms per " + std::to_string(ops::AneFfn::kMaximumRows) + "-row FFN layer";
  const auto fixed = [](double value) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << value;
    return text.str();
  };
  // The split that serves, with what the start found of it: its share, its
  // least chunk, the timings that chose them if it calibrated, the error
  // verify() found and how it was set up.
  const auto serve = [&](AneFfnPrepared prepared, uint32_t aneUnits, uint32_t minimumRows, const std::string &timed,
                         const char *how) {
    EngineMemoryPlanResult plan = planWith(aneUnits);
    if (!plan.plan) throw std::logic_error("the memory plan does not hold the split it chose");
    std::ostringstream reason;
    reason << std::fixed << std::setprecision(2) << "at " << (given ? "the given share " : "share ")
           << static_cast<double>(aneUnits) / model.units << " for chunks of " << minimumRows << " rows or more"
           << timed << ", " << fixed(100.0 * prepared.error) << "% from the GPU alone on the Neural Engine's part ("
           << how << " in " << seconds() << ")";
    return AneFfnStart{std::move(prepared.split), std::move(plan.plan),
                       {Kind::Split, reason.str(), automaticWith(aneUnits), without}};
  };
  try {
    if (given) {
      AneFfnPrepared prepared = model.prepare(*given, false);
      if (prepared.split) prepared.split->setMinimumRows(setting.given->minimumRows);
      return serve(std::move(prepared), *given, setting.given->minimumRows, "", "set up");
    }
    // The split takes the calibration's least chunk, and judges itself
    // against the GPU alone as calibrated, forgetting the calibration once it
    // loses.
    const auto take = [&](AneFfnPrepared &prepared, const ane_ffn::Calibration &calibration) {
      if (!prepared.split) return;
      prepared.split->setMinimumRows(calibration.minimumRows);
      prepared.split->setBreaker(ane_ffn::Breaker(calibration.gpu),
                                 [forget = model.forget, maxAneUnits] { forget(maxAneUnits); });
    };
    // No split gains enough: the start remembers so, and the GPU alone serves
    // its own context.
    const auto noGain = [&](std::string reason) {
      model.remember(maxAneUnits, {});
      assumed = 0;
      return gpuAloneServing(Kind::NoGain, std::move(reason));
    };

    std::optional<ane_ffn::Calibration> calibration = model.recall(maxAneUnits);
    if (calibration && calibration->aneUnits &&
        (calibration->aneUnits > maxAneUnits || calibration->minimumRows < ops::AneFfn::kMinimumRows ||
         calibration->minimumRows > ops::AneFfn::kMaximumRows))
      calibration.reset();
    if (calibration) {
      assumed = calibration->aneUnits;
      if (!calibration->aneUnits)
        return gpuAloneServing(Kind::NoGain, "no Neural Engine split beat it by enough when this Mac calibrated it");
      AneFfnPrepared prepared = model.prepare(calibration->aneUnits, false);
      take(prepared, *calibration);
      return serve(std::move(prepared), calibration->aneUnits, calibration->minimumRows, "",
                   "set up as calibrated before");
    }

    const ane_ffn::Timings timings = model.time();
    const std::optional<uint32_t> chosen =
        ane_ffn::choose(ane_ffn::fit(timings.low, timings.high), timings.gpuAlone, model.units, maxAneUnits);
    if (!chosen)
      return noGain("no Neural Engine split beats its " + fixed(timings.gpuAlone) + layer +
                    " by enough (calibrated in " + seconds() + ")");
    AneFfnPrepared prepared = model.prepare(*chosen, true);
    const ane_ffn::ChunkTimings &chunks = prepared.chunks;
    if (chunks.functions.empty()) throw std::logic_error("the split's chunks were not timed");
    const std::string timed =
        fixed(chunks.functions.front().split) + layer + " against " + fixed(chunks.gpu[1].milliseconds);
    const std::optional<uint32_t> minimumRows = ane_ffn::minimumRows(chunks);
    if (!minimumRows)
      return noGain("no chunk of the Neural Engine split beats it by enough, " + timed + " (calibrated in " +
                    seconds() + ")");
    const ane_ffn::Calibration made{*chosen, *minimumRows, chunks.gpu};
    take(prepared, made);
    AneFfnStart start = serve(std::move(prepared), *chosen, *minimumRows, ": " + timed + " on the GPU alone",
                              "calibrated");
    model.remember(maxAneUnits, made);
    return start;
  } catch (const ane::Interrupted &) {
    throw;
  } catch (const std::exception &error) {
    if ((cancelled && cancelled()) || !model.healthy()) throw;
    return gpuAloneServing(Kind::Unavailable, error.what());
  }
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
  }
  return "unknown";
}

AneFfnSetting AneFfnSetting::fromGiven(std::optional<double> share, std::optional<uint32_t> minimumRows) {
  if (!share) {
    if (minimumRows) throw std::invalid_argument("ANE FFN least chunk rows given without a share");
    return {};
  }
  if (*share == 0.0) return {.enabled = false};
  return {.given = Given{*share, minimumRows.value_or(ops::AneFfn::kMinimumRows)}};
}

AneFfnStart startAneFfn(const AneFfnModel &model, const AneFfnSetting &setting, uint32_t requestedContextTokens,
                        const std::function<EngineMemoryPlanResult(uint64_t aneFfnBytes)> &planMemory,
                        const std::function<bool()> &cancelled) {
  AneFfnStart start = decide(model, setting, requestedContextTokens, planMemory, cancelled);
  logOutcome(start.outcome, model, setting, requestedContextTokens);
  return start;
}

} // namespace splash::engine
