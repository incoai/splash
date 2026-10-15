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
    uint32_t minimumRows = ops::AneFfn::kMinimumRows;
  };
  bool enabled = true;
  std::optional<Given> given;

  // The setting of a dev tool's --ane-ffn-share and --ane-ffn-minimum-rows:
  // none calibrates the split, and a share of 0 runs the GPU alone. Throws
  // for least rows without a share.
  [[nodiscard]] static AneFfnSetting fromGiven(std::optional<double> share, std::optional<uint32_t> minimumRows);
};

// The split of a model, prepared: the split, taking every chunk its
// functions hold; its chunks as timed, if they were; and what
// AneFfn::verify returned.
struct AneFfnPrepared final {
  std::unique_ptr<ops::AneFfn> split;
  ops::ane_ffn::ChunkTimings chunks;
  double error = 0.0;
};

// The model's side of the split, as functions, so that a test can stand in
// for it (engine::aneFfnModel() in RuntimeResources.hpp).
struct AneFfnModel final {
  // Whether the target has dense FFN layers, which the split could take; a
  // MoE target has none.
  bool dense = false;
  // Why the split does not take them; empty when it does.
  std::string unsupported;
  // Why this Mac cannot run the split (ane::unavailable), asked only of a
  // split that may run; none when it can.
  std::function<std::optional<std::string>()> unavailable;
  // The channel units of the split (ops::AneFfn::units).
  uint32_t units = 0;
  // The split's Metal memory with the ANE taking `aneUnits` of them.
  std::function<uint64_t(uint32_t aneUnits)> plannedBytes;
  // Calibration's timings (ops::ane_ffn::Measurement::time).
  std::function<ops::ane_ffn::Timings()> time;
  // Whether the ANE's service holds the program of a split of `aneUnits`
  // (ops::AneFfn::compiled), which calibration's choice prefers within its
  // timings' noise (ops::ane_ffn::choose); none for no preference.
  std::function<bool(uint32_t aneUnits)> compiled;
  // The split that serves with the ANE taking `aneUnits`, verified, with its
  // chunks timed (ops::ane_ffn::Measurement::chunks) when `timeChunks`. A
  // model a test plays prepares no split.
  std::function<AneFfnPrepared(uint32_t aneUnits, bool timeChunks)> prepare;
  // Whether the Metal backend still serves.
  std::function<bool()> healthy;
};

// How a start's split came out: off as given, not taken by the model or this
// Mac, refused by the memory plan, gaining too little, failed, or running,
// and why or how. Startup logs it (startAneFfn).
struct AneFfnOutcome final {
  enum class Kind : uint8_t { Off, Unsupported, Refused, NoGain, Unavailable, Split };
  Kind kind = Kind::Off;
  std::string reason;
  // The context the engine serves without --max-context, and the one the
  // memory plan holds without the split. While the split may run, the first
  // is what the plan holds with the split the start's calibration chose,
  // none if it found no gain, and with the largest split the model could take
  // if the calibration failed. Otherwise both are the same. For a split
  // refused a --max-context, the first is the most context a split leaves
  // instead.
  uint32_t context = 0, contextWithout = 0;
};
// The kind's name in lowercase, as "no_gain".
[[nodiscard]] std::string_view aneFfnOutcomeName(AneFfnOutcome::Kind kind) noexcept;

// The split a start runs and the memory plan that sets its memory aside,
// both none without one, and the outcome.
struct AneFfnStart final {
  std::unique_ptr<ops::AneFfn> split;
  std::optional<EngineMemoryPlan> plan;
  AneFfnOutcome outcome;
};

// The prefill FFN's Neural Engine split of `model` as `setting` allows,
// within the memory plan `planMemory` makes with the split's bytes set aside:
// one whose plan still holds `requestedContextTokens`, or with zero the
// automatic context (AneFfnOutcome::context). Every step is arithmetic until
// the plan holds a split; then it logs that it sets the split up. Every start
// calibrates the split within the units the plan holds: times it, chooses its
// share (ops::ane_ffn::choose), prepares the split that serves, times its
// chunks and finds the least it takes (ops::ane_ffn::minimumRows). The split
// then judges itself against the GPU alone as calibrated
// (ops::ane_ffn::Breaker); a given split takes the given least chunk and no
// breaker. Logs the outcome. A failure leaves the GPU alone, unless `cancelled` returns true,
// the wait for the ANE was interrupted or the backend no longer serves, which
// the failure's exception reports. Throws std::invalid_argument for a given
// split of no units or of least rows no function holds.
[[nodiscard]] AneFfnStart
startAneFfn(const AneFfnModel &model, const AneFfnSetting &setting, uint32_t requestedContextTokens,
            const std::function<EngineMemoryPlanResult(uint64_t aneFfnBytes)> &planMemory,
            const std::function<bool()> &cancelled);

} // namespace splash::engine
