#pragma once

#include "AwakeClock.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
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
// them), as the split runs them: its GPU part alone, and the split layer;
// and the GPU's FFN layer alone read right before it.
struct Timing final {
  double share = 0.0, gpu = 0.0, split = 0.0, alone = 0.0;
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

// What calibration reads, in the order time() below reads it: a split of
// `aneUnits` made ready to time, its program of one function of a full
// chunk's rows and, with `half`, of half as many too (`load`); that split's
// GPU part alone (`gpuPart`); the GPU's FFN layer alone of a step of `rows`
// rows (`alone`); and the split's layer of `rows` rows after its stand-in,
// less `standIn` (`split`): Measurement's Metal commands, or a test's own.
struct Readers final {
  std::function<void(uint32_t aneUnits, bool half)> load;
  std::function<double()> gpuPart;
  std::function<double(uint32_t rows)> alone;
  std::function<double(uint32_t rows, double standIn)> split;
};
// The split's timings at the channel units of `units` nearest two shares,
// each split layer less the GPU alone read right before it, in the state its
// command finds the GPU in: the GPU's pace moves as the ANE's programs
// compile (an M4 MacBook Air read 352.9 ms a full chunk's FFN layer first and
// 470 once it settled), and a GPU alone read earlier made the split layers
// longer by the difference. The GPU alone the fit's gate takes is the high
// share's, read last, beside the timings that set the units it chooses.
// `rows` are a full chunk's.
[[nodiscard]] Timings time(uint32_t units, uint32_t rows, const Readers &read);
// The same of a split of `aneUnits` alone, with the GPU alone read beside it.
[[nodiscard]] Timing time(uint32_t aneUnits, uint32_t units, uint32_t rows, const Readers &read);

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
// each command it finishes, the Neural Engine's milliseconds over it against
// the GPU alone's of its layers.
class Breaker final {
public:
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
  // split lost to the GPU alone on it, when it took the Neural Engine as long
  // as the GPU alone takes those layers; empty otherwise.
  [[nodiscard]] std::string judge(uint32_t rows, uint32_t layers, bool pipelined, double milliseconds) const;

private:
  // The GPU alone's milliseconds of a layer by its rows, and the most rows.
  std::optional<Line> gpu_;
  double mixer_ = 0.0;
  uint32_t mostRows_ = 0;
};

// What a split does once its breaker finds a command losing
// (AneFfn::setBreaker): it pauses for `length`, the GPU running every command
// alone until the first that finishes past the pause; the next command it
// splits probes, and a probe that loses stops it for good. One loss says the
// Neural Engine was slow then (another of its clients, heat), not that the
// calibration is wrong; a probe that loses too says it stays slow, and one
// that wins says it passed, so a later loss pauses the split as long again.
class Pause final {
public:
  // What a command changes: nothing, a pause begins, or the split stops.
  enum class Verdict : uint8_t { Kept, Paused, Stopped };
  static constexpr AwakeClock::duration kLength = std::chrono::minutes(5);

  explicit Pause(AwakeClock::duration length = kLength) noexcept : length_(length) {}
  [[nodiscard]] bool paused() const noexcept { return until_.has_value(); }
  // When the pause ends; only while paused().
  [[nodiscard]] AwakeClock::time_point until() const { return until_.value(); }
  // A command finished at `now`, split or on the GPU alone, and its breaker
  // found it `losing` or not.
  Verdict finished(AwakeClock::time_point now, bool split, bool losing);

private:
  std::optional<AwakeClock::time_point> until_;
  AwakeClock::duration length_;
  bool probing_ = false;
};

// Who was late in a split command whose longest buffer took more than 1 s and
// ten times its buffers' median (metal::CommandTiming): the Neural Engine when
// its longest evaluation, from the notice of its `ready` to its report, took
// half the excess or more, the GPU otherwise; none for a command without such
// a buffer (AneFfn::finish). The breaker counts the GPU's wait on a late
// evaluation already, in that evaluation's time; a GPU late on its own, the
// GPU alone would have been as late.
enum class Late : uint8_t { NeuralEngine, Gpu };
[[nodiscard]] std::optional<Late> late(double longestBuffer, double medianBuffer, double longestEvaluation) noexcept;

} // namespace splash::ops::ane_ffn
