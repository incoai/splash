#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// How the prefill FFN's Neural Engine split (ops/AneFfn.hpp) runs on this
// device, from what ane_ffn::Measurement times: the ANE's share of a full
// chunk's FFN, the least chunk the split takes, and when a split that serves
// stops for losing to the GPU alone. Nothing here touches Metal or the ANE.
namespace splash::ops::ane_ffn {

// The split's per-layer milliseconds of a full chunk with the ANE taking
// `share` of the intermediate channels (its channel units over all of
// them), as the split runs them: its GPU part alone, and the split layer.
struct Timing final {
  double share = 0.0, gpu = 0.0, split = 0.0;
};
// What calibration times: the split at two shares, a full chunk's FFN layer
// on the GPU alone, the high share's split layer of a step of half a full
// chunk's rows, and the rest of the GPU alone's layer of a full chunk: the
// mixer, its norms and its share of the command, which the engine's prefill
// of a full chunk before the split joins it gives (engine::finishAneFfn).
struct Timings final {
  Timing low, high;
  double gpuAlone = 0.0, highHalf = 0.0, mixer = 0.0;
};

// `milliseconds`, which a measurement takes only as a duration it measured:
// finite and positive. Throws "calibration timings are not usable"
// otherwise.
[[nodiscard]] double usable(double milliseconds);

// The ANE's channel units nearest `share`, in (0, 1), of the `units` the
// intermediate channels hold: from one to all but one of them. Throws for
// another share or fewer than two units.
[[nodiscard]] uint32_t nearestUnits(double share, uint32_t units);

struct Line final {
  double at0 = 0.0, slope = 0.0;
  [[nodiscard]] double operator()(double share) const noexcept { return at0 + slope * share; }
};
// A full chunk's split FFN layer of one step at share s takes
//   T(s) = max(A(s), G(s) + u A(s)):
// the ANE's part A(s), which the layer waits on where it takes the longer,
// or the GPU's part G(s), slowed by the memory bandwidth the ANE's part takes
// beside it, u A(s). G is close to linear in s, and A to proportional. Of two
// steps, the halves of the chunk pipelined (AneFfn), the whole layer takes
//   T(s) = max(H(s), M + G(s) + u H(s)):
// the ANE's two evaluations of a half, H(s), proportional too, or the GPU's
// work over the layer, its mixer M beside the other step's evaluation.
struct Model final {
  Line gpu, ane, halves;
  double u = 0.0, mixer = 0.0;
  [[nodiscard]] double operator()(double share) const noexcept;
};
// The model of one step through the timings of a low and a high share
// (AneFfnCalibration.cpp). Throws unless the low share is the lower.
[[nodiscard]] Model fit(const Timing &low, const Timing &high);
// `oneStep`, fitted on `timings`, as two steps: H through the high share's
// step of half the rows, and the mixer. Throws unless the mixer is of none or
// more milliseconds.
[[nodiscard]] Model twoSteps(Model oneStep, const Timings &timings);
// `model` with its bandwidth's term u fit to `split`, a full chunk's split
// layer timed at `share`, where the GPU's part takes the longer there; the
// same model where the ANE's part does (AneFfnCalibration.cpp). Calibration
// fits u so at the low share, then at the units the model first chooses,
// nearest T's least.
[[nodiscard]] Model refit(Model model, double share, double split);

// The ANE's channel units, of the `units` the intermediate channels hold, at
// which chunks split, at most `maxAneUnits` of them: those at which T, or T a
// unit either way, is least at its longest (AneFfnCalibration.cpp); none if
// `model` predicts no gain worth the split there on `gpuAlone` milliseconds
// of the GPU alone.
[[nodiscard]] std::optional<uint32_t> choose(const Model &model, double gpuAlone, uint32_t units,
                                             uint32_t maxAneUnits);

// A chunk's FFN layer in milliseconds: the GPU's alone at the least rows and
// a full chunk's, and the split's of each of its functions of one step, from
// the most rows down, which chunks of `least` to `rows` rows run.
struct ChunkTimings final {
  struct Gpu final {
    uint32_t rows = 0;
    double milliseconds = 0.0;
  };
  struct Function final {
    uint32_t least = 0, rows = 0;
    double split = 0.0;
  };
  std::array<Gpu, 2> gpu{};
  std::vector<Function> functions;
};
// The least rows of a chunk the split takes: those from which every larger
// chunk of the functions splits (AneFfnCalibration.cpp); none when not even
// a chunk of the most rows does.
[[nodiscard]] std::optional<uint32_t> minimumRows(const ChunkTimings &timings);
// Whether the split `candidate` times splits every chunk the split
// `incumbent` times splits, none of them more than the timings' noise longer
// (AneFfnCalibration.cpp).
[[nodiscard]] bool slowerNowhere(const ChunkTimings &candidate, const ChunkTimings &incumbent);

// What one calibration of the split on this Mac decided, which later starts
// take again (engine::finishAneFfn): the ANE's units of a command of one
// step, which name the split's program, one step's fit's or two steps' where
// that chose none, and the least rows of a chunk the split takes in one step,
// none where one step splits no chunk below two steps' commands; the ANE's
// units of a command of two steps, which take every command from
// ops::AneFfn::kTwoStepRows, none where two would not gain; none of any when
// no split gained enough; the GPU alone's FFN layer at the least rows and a
// full chunk's, and the rest of its layer of a full chunk beside that (the
// mixer), which the breaker judges the split against. Beside it, how often
// the Neural Engine refused a split of the model (ane::Refused) since the
// start that first calibrated it: a record of no split and one refusal
// calibrates again, and the next refusal leaves no split and kRefusals, the
// GPU alone, so that a split the Neural Engine refuses each time costs two
// calibrations at most, not one every other start.
struct Calibration final {
  static constexpr uint32_t kRefusals = 2;
  uint32_t aneUnits = 0, minimumRows = 0;
  std::array<ChunkTimings::Gpu, 2> gpu{};
  uint32_t refusals = 0;
  uint32_t twoStepUnits = 0;
  double mixer = 0.0;

  // What the next start does with `record`, as remembered, in a log's words:
  // sets its split up, calibrates again or leaves the GPU alone.
  [[nodiscard]] static std::string nextStart(const std::optional<Calibration> &record);
  // Why a start that takes this record, of no split, leaves the GPU alone, in
  // a log's words: no split gained enough, or the Neural Engine refused one.
  [[nodiscard]] std::string withoutSplit() const;
  // As one line of text, and back: none for text that is not such a line.
  [[nodiscard]] std::string text() const;
  [[nodiscard]] static std::optional<Calibration> parse(std::string_view text);
};
// The record `recalled` once the Neural Engine refused the split a start
// prepared: `kept`, the calibration the next start sets up, or none, with the
// refusal counted, and none at Calibration::kRefusals.
[[nodiscard]] Calibration refused(const std::optional<Calibration> &recalled, Calibration kept = {});

// When a split that serves loses to the GPU alone (AneFfnCalibration.cpp):
// over each kWindow commands it finishes, the Neural Engine's milliseconds
// against the GPU alone's of as many layers.
class Breaker final {
public:
  static constexpr uint32_t kWindow = 8;

  // One that never trips, for a split whose GPU alone was not timed.
  Breaker() = default;
  // One against the GPU alone's FFN layer as timed at the least and the most
  // rows of the split's functions, and `mixer`, the rest of its layer of the
  // most rows. Throws unless those are ascending rows of usable milliseconds
  // and a mixer of none or more.
  explicit Breaker(const std::array<ChunkTimings::Gpu, 2> &gpu, double mixer);

  // A command the split finished: over each of its `layers` layers,
  // evaluations of functions of `rows` rows in all, its steps two
  // (`pipelined`) or one, took the Neural Engine `milliseconds` in all. Why the
  // split loses to the GPU alone, once a window of commands shows it; empty
  // otherwise.
  [[nodiscard]] std::string add(uint32_t rows, uint32_t layers, bool pipelined, double milliseconds);

private:
  // The GPU alone's milliseconds of a layer by its rows, and the most rows.
  std::optional<Line> gpu_;
  double mixer_ = 0.0;
  uint32_t mostRows_ = 0;
  // The window's commands, the Neural Engine's milliseconds over them and the
  // GPU alone's of their layers.
  uint32_t commands_ = 0;
  double ane_ = 0.0, alone_ = 0.0;
};

} // namespace splash::ops::ane_ffn
