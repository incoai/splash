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
// them), as the split runs them: its GPU part alone, and the split layer;
// and for a split of slots, the ANE's time a chunk's layer as a pipelined
// command keeps it busy (0 otherwise).
struct Timing final {
  double share = 0.0, gpu = 0.0, split = 0.0, ane = 0.0;
};
// What calibration times: the split at two shares, a full chunk's FFN layer
// on the GPU alone, and for a split that pipelines its chunks the layer's
// mixer on the GPU alone (0 otherwise).
struct Timings final {
  Timing low, high;
  double gpuAlone = 0.0, mixer = 0.0;
};

// `milliseconds`, which a measurement takes only as a duration it measured:
// finite and positive. Throws "calibration timings are not usable"
// otherwise.
[[nodiscard]] double usable(double milliseconds);

// The ANE's channel units nearest `share`, in (0, 1], of the `units` the
// intermediate channels hold: from one to all but one of them, or all of
// them at a share of 1, which a pipelined split runs on the ANE whole
// (AneFfn::fusesNorm). Throws for another share or fewer than two units.
[[nodiscard]] uint32_t nearestUnits(double share, uint32_t units);

struct Line final {
  double at0 = 0.0, slope = 0.0;
  [[nodiscard]] double operator()(double share) const noexcept { return at0 + slope * share; }
};
// A full chunk's split FFN layer at share s takes
//   T(s) = max(A(s), M + G(s) + u A(s)):
// the ANE's part A(s), which the layer waits on where it takes the longer,
// or the GPU's part G(s), slowed by the memory bandwidth the ANE's part takes
// beside it, u A(s). G is close to linear in s, and A to proportional. A
// split that runs each chunk's FFN after its mixer has no M; one that
// pipelines a command's chunks (AneFfn::start) runs the ANE's part beside
// the GPU's mixers of other chunks, whose time M joins the GPU's side.
struct Model final {
  Line gpu, ane;
  double u = 0.0, mixer = 0.0;
  [[nodiscard]] double operator()(double share) const noexcept;
};
// The model through the timings of a low and a high share
// (AneFfnCalibration.cpp). Throws unless the low share is the lower.
[[nodiscard]] Model fit(const Timing &low, const Timing &high);
// The same of calibration's timings, with the mixers and the ANE's pipelined
// time a split that pipelines its chunks timed (Timings::mixer, Timing::ane).
[[nodiscard]] Model fit(const Timings &timings);
// `model` with its bandwidth's term u fit to `split`, a full chunk's split
// layer timed at `share`, where the GPU's part takes the longer there; the
// same model where the ANE's part does (AneFfnCalibration.cpp). Calibration
// fits u so at the low share, then at the units the model first chooses,
// nearest T's least.
[[nodiscard]] Model refit(Model model, double share, double split);
// refit() at `at`'s share and split layer, its pipelined ANE time, if timed,
// giving A anew.
[[nodiscard]] Model refit(Model model, const Timing &at);

// The ANE's channel units, of the `units` the intermediate channels hold, at
// which chunks split, at most `maxAneUnits` of them, all of them only for a
// model with a mixer: those at which T, or T a unit either way, is least at
// its longest (AneFfnCalibration.cpp); none if `model` predicts no gain
// worth the split there on `gpuAlone` milliseconds of the GPU's FFN alone
// after the model's mixer.
[[nodiscard]] std::optional<uint32_t> choose(const Model &model, double gpuAlone, uint32_t units,
                                             uint32_t maxAneUnits);

// A chunk's FFN layer in milliseconds: the GPU's alone at the least and the
// most rows the split takes, and the split's of each count of its pieces
// (AneFfn::kEvaluationRows), from the most rows down, which chunks of
// `least` to `rows` rows run, the last piece padded.
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
// The chunks a split takes, as a start's log says: alone from `minimumRows`
// (none past `mostRows`), and pipelined from `pipelinedRows` where it
// `pipelines`.
[[nodiscard]] std::string chunksTaken(uint32_t minimumRows, uint32_t mostRows, bool pipelines,
                                      uint32_t pipelinedRows);
// The least rows of a chunk the split takes alone: those from which every
// larger chunk timed splits (AneFfnCalibration.cpp); none when not even a
// chunk of the most rows does.
[[nodiscard]] std::optional<uint32_t> minimumRows(const ChunkTimings &timings);
// Whether the split `candidate` times splits every chunk the split
// `incumbent` times splits, none of them more than the timings' noise longer
// (AneFfnCalibration.cpp).
[[nodiscard]] bool slowerNowhere(const ChunkTimings &candidate, const ChunkTimings &incumbent);

// What one calibration of the split on this Mac decided, which later starts
// take again (engine::startAneFfn): the ANE's units and the least rows of a
// chunk the split takes, none of either when no split gained enough; and the
// GPU alone's layer at the least and the most rows the split takes,
// which the breaker judges the split against. Beside it, how often the
// Neural Engine refused a split of the model (ane::Refused) since the start
// that first calibrated it: a record of no split and one refusal calibrates
// again, and the next refusal leaves no split and kRefusals, the GPU alone,
// so that a split the Neural Engine refuses each time costs two calibrations
// at most, not one every other start.
struct Calibration final {
  static constexpr uint32_t kRefusals = 2;
  uint32_t aneUnits = 0, minimumRows = 0;
  std::array<ChunkTimings::Gpu, 2> gpu{};
  uint32_t refusals = 0;
  // A layer of the most rows' mixer projections (Timings::mixer), which a
  // pipelined split's breaker counts beside the GPU alone's FFN; 0 for a
  // split that does not pipeline.
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
// against the GPU alone's of as many layers, their mixers' too where the
// command pipelined its chunks.
class Breaker final {
public:
  static constexpr uint32_t kWindow = 8;

  // One that never trips, for a split whose GPU alone was not timed.
  Breaker() = default;
  // One against the GPU alone's layer as timed at the least and the most rows
  // of the split's chunks, and `mixer`, the mixer projections of a layer of
  // the most rows, which scale with rows. Throws unless those are ascending
  // rows of usable milliseconds and a mixer of none or more.
  explicit Breaker(const std::array<ChunkTimings::Gpu, 2> &gpu, double mixer = 0.0);

  // A command the split finished: `evaluations` of chunks of `rows` rows,
  // from the least to the most rows timed, took the Neural Engine
  // `milliseconds` in all, its chunks `pipelined` or not. Why the split loses
  // to the GPU alone, once a window of commands shows it; empty otherwise.
  [[nodiscard]] std::string add(uint32_t rows, uint32_t evaluations, double milliseconds, bool pipelined = false);

private:
  // The GPU alone's milliseconds of a layer by its rows, and the most rows.
  std::optional<Line> gpu_;
  uint32_t mostRows_ = 0;
  double mixer_ = 0.0;
  // The window's commands, the Neural Engine's milliseconds over them and the
  // GPU alone's of their layers.
  uint32_t commands_ = 0;
  double ane_ = 0.0, alone_ = 0.0;
  // Whether a command of the window pipelined its chunks.
  bool pipelined_ = false;
};

} // namespace splash::ops::ane_ffn
