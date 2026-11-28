#include "ops/AneFfnCalibration.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace splash::ops::ane_ffn {
namespace {

// The split's predicted gain below which the GPU runs the FFN alone.
constexpr double kMinimumGain = 0.05;
// How much longer than the GPU's alone a chunk's split layer may take and
// the chunk still split: six timings of each function's split layer of
// Qwen3.8-27B on an M5 Max spread over ±3%, and on the layers calibration
// fits (Measurement::chunks) over ±1.6% for the median function and ±3% at
// most, and the FFN takes about half of a prefill, so a chunk that splits
// within this prefills at most about 1% slower.
constexpr double kChunkNoise = 0.02;
// The first word of a Calibration's text, which a change to its fields
// renumbers.
constexpr std::string_view kCalibrationTag = "ane-ffn-calibration-4";

// The GPU alone's milliseconds of a chunk's FFN layer by its rows: the line
// through its timings, which the GPU follows within 1% between them.
Line gpuAlone(const std::array<ChunkTimings::Gpu, 2> &gpu) {
  const auto &[low, high] = gpu;
  const double slope = (high.milliseconds - low.milliseconds) / (high.rows - low.rows);
  return {low.milliseconds - slope * low.rows, slope};
}

// The split layer of the function that holds a step of `rows` rows; infinite
// where none does.
double functionLayer(const ChunkTimings &timings, uint32_t rows) {
  const auto function = std::ranges::find_if(timings.functions, [&](const ChunkTimings::Function &timed) {
    return timed.least <= rows && rows <= timed.rows;
  });
  return function == timings.functions.end() ? std::numeric_limits<double>::infinity() : function->split;
}

// The shares calibration times the split at.
constexpr std::array<double, 2> kTimedShares{0.4, 0.8};

// A share's timing: its split's GPU part, and its layer less the GPU alone
// read right before it. The high share's program also holds a function of
// half the rows.
Timing share(uint32_t aneUnits, uint32_t units, uint32_t rows, bool half, const Readers &read) {
  read.load(aneUnits, half);
  const double gpu = read.gpuPart(), alone = read.alone(rows);
  return {static_cast<double>(aneUnits) / units, gpu, read.split(rows, alone), alone};
}

} // namespace

Timings time(uint32_t units, uint32_t rows, const Readers &read) {
  Timings timings;
  timings.low = share(nearestUnits(kTimedShares[0], units), units, rows, false, read);
  timings.high = share(nearestUnits(kTimedShares[1], units), units, rows, true, read);
  timings.gpuAlone = timings.high.alone;
  timings.highHalf = read.split(rows / 2, read.alone(rows / 2));
  return timings;
}

Timing time(uint32_t aneUnits, uint32_t units, uint32_t rows, const Readers &read) {
  return share(aneUnits, units, rows, false, read);
}

double usable(double milliseconds) {
  if (!(milliseconds > 0.0 && std::isfinite(milliseconds)))
    throw std::runtime_error("calibration timings are not usable");
  return milliseconds;
}

uint32_t nearestUnits(double share, uint32_t units) {
  if (!(share > 0.0 && share < 1.0) || units < 2)
    throw std::invalid_argument("ANE FFN share must lie in (0, 1) of two units or more");
  return static_cast<uint32_t>(std::clamp<long>(std::lround(share * units), 1, units - 1));
}

// Not a number where either part is not. A model of two steps has its
// halves' line, whatever its mixer.
double Model::operator()(double share) const noexcept {
  const double a = halves.slope ? halves(share) : ane(share), g = mixer + gpu(share) + u * a;
  return a > g ? a : g;
}

// The model, from the split's layers as they run between stand-ins for the
// mixer (Measurement). The GPU's part alone gives G at both shares. At the
// high share, near 0.8, the ANE's part takes the longer on every Mac measured
// (Qwen3.8-27B's fastest split gives the ANE 7 to 27 of its 34 units): the
// split layer there is A, which scales with the ANE's channels. The low
// share's split layer gives u where the GPU's part takes the longer there
// (refit). Where the ANE's part does not take the longer at the high share,
// A comes out longer than it is, and the choice smaller.
Model fit(const Timing &low, const Timing &high) {
  if (!(low.share < high.share)) throw std::invalid_argument("ANE FFN calibration needs a low and a high share");
  const double slope = (high.gpu - low.gpu) / (high.share - low.share);
  return refit({{low.gpu - slope * low.share, slope}, {0.0, high.split / high.share}, {}}, low.share, low.split);
}

// The high share's step of half the rows gives H as its full chunk gives A:
// a half costs the ANE more than half a full chunk (its 0.97 share of
// Qwen3.8-27B's FFN 29.7 ms at 1,024 rows against 54.5 at 2,048 on a 20-core
// M5 Pro).
Model twoSteps(Model oneStep, const Timings &timings) {
  if (!(timings.mixer >= 0.0 && std::isfinite(timings.mixer)))
    throw std::invalid_argument("ANE FFN calibration's mixer is not usable");
  oneStep.halves = {0.0, 2.0 * usable(timings.highHalf) / timings.high.share};
  oneStep.mixer = timings.mixer;
  return oneStep;
}

// Where the GPU's part takes the longer at `share`, the split layer there
// exceeds it by the bandwidth the ANE's part takes beside it, which gives u (a
// tenth to a third on an M6, at the low share). Where the ANE's part takes
// the longer there, the split layer's excess over A is the ANE's own, which
// says nothing of u. On an M5 Max the ANE's part takes the longer from below
// 0.4, so the low share leaves u at none, and T a unit short of T's least
// comes out shorter than it is: Qwen3.8-27B's G(6/34) of 13.2 ms against a
// split layer of 13.8 there (u = 0.06), which chose 6 units in half the
// calibrations against the 7 a sweep of given shares found fastest.
Model refit(Model model, double share, double split) {
  const double gpu = model.gpu(share), ane = model.ane(share);
  if (gpu > ane) model.u = std::clamp((split - gpu) / ane, 0.0, 1.0);
  return model;
}

// The units whose T, or T a unit either way, is least at its longest: the
// timings can put T's least a unit off, and a unit past where the ANE's part
// takes the longer costs the ANE's time per unit, a unit short of it the
// GPU's. Where the ANE's is the longer (M5 Max: 1.7 ms a unit of Qwen3.8-27B's
// full-chunk layer against the GPU's 0.6), the choice keeps below T's least;
// where the GPU's is (M6), it need not. Chunks of fewer rows, which the ANE
// pads up to its functions' rows, move T's least lower still; minimumRows
// keeps the split from the chunks it loses. The choice runs only if T there
// gains kMinimumGain on the GPU alone.
std::optional<uint32_t> choose(const Model &model, double gpuAlone, uint32_t units, uint32_t maxAneUnits) {
  if (!maxAneUnits || maxAneUnits >= units) throw std::invalid_argument("ANE FFN choice of no feasible units");
  std::vector<double> at(units);
  for (uint32_t aneUnits = 1; aneUnits < units; ++aneUnits) {
    at[aneUnits] = model(static_cast<double>(aneUnits) / units);
    if (!(at[aneUnits] > 0.0 && std::isfinite(at[aneUnits]))) return std::nullopt;
  }
  const auto longest = [&](uint32_t aneUnits) {
    return std::max({at[aneUnits], at[std::max(aneUnits - 1, 1u)], at[std::min(aneUnits + 1, units - 1)]});
  };
  uint32_t chosen = 1;
  for (uint32_t aneUnits = 2; aneUnits <= maxAneUnits; ++aneUnits)
    if (longest(aneUnits) < longest(chosen)) chosen = aneUnits;
  if (!(at[chosen] <= (1.0 - kMinimumGain) * gpuAlone)) return std::nullopt;
  return chosen;
}

// A chunk splits while its function's split layer takes at most kChunkNoise
// longer than the GPU's layer alone of the chunk's rows (gpuAlone). Every
// chunk a function runs takes about as long as the function's own rows: the
// ANE's part runs them and takes the longer. The functions do not cost the
// ANE in proportion to their rows (on an M5 Max 640 rows took 93% of 768's,
// 512 rows 73%), so each is timed, and each row count it runs checked: a
// batch's chunk sums its lanes' rows.
std::optional<uint32_t> minimumRows(const ChunkTimings &timings) {
  const auto &[low, high] = timings.gpu;
  const std::vector<ChunkTimings::Function> &functions = timings.functions;
  bool ordered = !functions.empty() && low.rows < high.rows;
  for (size_t index = 0; ordered && index < functions.size(); ++index)
    ordered = functions[index].least && functions[index].least <= functions[index].rows &&
              (!index || functions[index].rows + 1 == functions[index - 1].least);
  if (!ordered) throw std::invalid_argument("ANE FFN chunks are not those of its functions");
  const Line gpu = gpuAlone(timings.gpu);
  std::optional<uint32_t> minimum;
  for (const ChunkTimings::Function &function : functions)
    for (uint32_t rows = function.rows; rows >= function.least; --rows) {
      if (!(function.split <= (1.0 + kChunkNoise) * gpu(rows))) return minimum;
      minimum = rows;
    }
  return minimum;
}

// The candidate must split every chunk the incumbent splits, each of them in
// at most kChunkNoise longer, both split layers timed alike: more units can
// cost chunks a full one does not show. Qwen3.8-27B UD-Q4_K_M on an M5 Max
// took 10 of 34 units on the refitted model, whose full chunk prefilled
// 1.5-2.0% faster than 9's but which split from 780 rows against 575, so that
// chunks of 640 and 672 rows ran 12% slower. Below the incumbent's least
// rows the candidate splits only what its own least rows allow.
bool slowerNowhere(const ChunkTimings &candidate, const ChunkTimings &incumbent) {
  const std::optional<uint32_t> candidateLeast = minimumRows(candidate), incumbentLeast = minimumRows(incumbent);
  if (!incumbentLeast) return true;
  if (!candidateLeast || *candidateLeast > *incumbentLeast) return false;
  for (uint32_t rows = *incumbentLeast; rows <= incumbent.functions.front().rows; ++rows)
    if (!(functionLayer(candidate, rows) <= (1.0 + kChunkNoise) * functionLayer(incumbent, rows))) return false;
  return true;
}

Calibration refused(const std::optional<Calibration> &recalled, Calibration kept) {
  kept.refusals = (recalled ? recalled->refusals : 0) + 1;
  return kept.refusals < Calibration::kRefusals ? kept : Calibration{.refusals = Calibration::kRefusals};
}

std::string Calibration::nextStart(const std::optional<Calibration> &record) {
  if (record && record->aneUnits) return "the next start sets it up as calibrated";
  if (record && record->refusals == kRefusals)
    return "starts leave the GPU alone until Splash, macOS or the model changes";
  return "the next start calibrates again";
}

std::string Calibration::withoutSplit() const {
  return "the Neural Engine refused its split twice; " + nextStart(*this);
}

std::string Calibration::text() const {
  std::ostringstream text;
  text << std::setprecision(9) << kCalibrationTag << ' ' << aneUnits << ' ' << minimumRows;
  for (const ChunkTimings::Gpu &point : gpu) text << ' ' << point.rows << ' ' << point.milliseconds;
  text << ' ' << refusals << ' ' << twoStepUnits << ' ' << mixer;
  return text.str();
}

// A record of units holds least rows of one step or units of two steps, or
// both, the GPU alone at ascending rows of usable milliseconds, and fewer
// than kRefusals refusals; one of none holds neither, and one refusal or
// kRefusals.
std::optional<Calibration> Calibration::parse(std::string_view text) {
  std::istringstream in{std::string(text)};
  std::string tag;
  Calibration calibration;
  in >> tag >> calibration.aneUnits >> calibration.minimumRows;
  for (ChunkTimings::Gpu &point : calibration.gpu) in >> point.rows >> point.milliseconds;
  in >> calibration.refusals >> calibration.twoStepUnits >> calibration.mixer;
  std::string rest;
  if (!in || tag != kCalibrationTag || (in >> rest) || calibration.refusals > kRefusals) return std::nullopt;
  const auto &[low, high] = calibration.gpu;
  const auto usableMilliseconds = [](double value) { return value > 0.0 && std::isfinite(value); };
  if (!calibration.aneUnits)
    return calibration.minimumRows || calibration.twoStepUnits || !calibration.refusals
               ? std::nullopt
               : std::optional(Calibration{.refusals = calibration.refusals});
  if (calibration.refusals == kRefusals) return std::nullopt;
  if ((!calibration.minimumRows && !calibration.twoStepUnits) || !(low.rows < high.rows) ||
      !usableMilliseconds(low.milliseconds) || !usableMilliseconds(high.milliseconds) ||
      !(calibration.mixer >= 0.0 && std::isfinite(calibration.mixer)))
    return std::nullopt;
  return calibration;
}

// Each layer the GPU waits for the Neural Engine's evaluations before it joins
// the ANE's part to its own: one step's evaluations that take as long as the
// GPU alone's FFN layers, or two pipelined steps' as long as its whole layers,
// beside which they run, make the split slower than the GPU alone, whatever
// the rest of the layer takes: the breaker trips there, and never short of it. An
// evaluation counts against the GPU alone's layer of its function's rows,
// which the chunk pads up to, so the GPU alone would take no longer over the
// chunk; the reason scales the command's ratio to a layer of the most rows.
// At the units calibration chooses, at or just below where the parts meet
// (choose), an evaluation takes at most about as long as the GPU's part
// (Qwen3.8-27B on an M5 Max: 13.2 ms against the GPU alone's 19.0). An M5
// Pro's commands of two steps took 0.66-0.67 of the GPU alone's layers
// through seven minutes of prefill; beside another client of the ANE at their
// priority, 1.77; behind one at a higher priority, 12.7. Each command is
// judged on its own: one slow evaluation, as a function's first after its
// program loads again, weighs little among a command's.
Breaker::Breaker(const std::array<ChunkTimings::Gpu, 2> &gpu, double mixer) {
  const auto &[low, high] = gpu;
  if (!(low.rows < high.rows)) throw std::invalid_argument("ANE FFN breaker needs the GPU alone at two row counts");
  if (!(mixer >= 0.0 && std::isfinite(mixer))) throw std::invalid_argument("ANE FFN breaker's mixer is not usable");
  static_cast<void>(usable(low.milliseconds));
  static_cast<void>(usable(high.milliseconds));
  gpu_ = gpuAlone(gpu);
  mixer_ = mixer;
  mostRows_ = high.rows;
}

std::optional<double> Breaker::alone(uint32_t rows, uint32_t layers, bool pipelined) const {
  if (!gpu_) return std::nullopt;
  return layers * ((*gpu_)(rows) + (pipelined ? mixer_ * rows / mostRows_ : 0.0));
}

std::string Breaker::judge(uint32_t rows, uint32_t layers, bool pipelined, double milliseconds) const {
  if (!gpu_) return {};
  const double ratio = milliseconds / *alone(rows, layers, pipelined), most = (*gpu_)(mostRows_);
  if (!(ratio >= 1.0)) return {};
  std::ostringstream reason;
  reason << std::fixed << std::setprecision(1) << "losing to the GPU alone (" << ratio * most
         << " ms on the Neural Engine per " << mostRows_ << "-row layer against " << most << " ms on the GPU alone)";
  return reason.str();
}

Pause::Verdict Pause::finished(AwakeClock::time_point now, bool split, bool losing) {
  if (until_ && now >= *until_) {
    until_.reset();
    probing_ = true;
  }
  if (!split) return Verdict::Kept;
  if (!losing) {
    probing_ = false;
    return Verdict::Kept;
  }
  if (std::exchange(probing_, false)) return Verdict::Stopped;
  until_ = now + length_;
  return Verdict::Paused;
}

std::optional<Late> late(double longestBuffer, double medianBuffer, double longestEvaluation) noexcept {
  if (!(longestBuffer > 1.0 && longestBuffer > 10.0 * medianBuffer)) return std::nullopt;
  return longestEvaluation >= (longestBuffer - medianBuffer) / 2 ? Late::NeuralEngine : Late::Gpu;
}

// Split commands of Qwen3.8-27B MLX 4-bit on a 20-core M5 Pro took the GPU
// 0.80-1.33 of the GPU alone's time over their layers through 1,179 commands
// of cold prompts beside display sleep and wake, the screen saver, every core
// busy and memory at the system's warning, the most a 640-row command in one
// step, whose GPU alone counts no mixer; beside another process keeping the
// GPU busy with compute buffers of 1 s, 10.6-21.1. Three times sits over
// twice the first and under a third of the second. The image encodes ahead of
// a command are no work of its layers.
bool shared(double gpuMilliseconds, double encodeMilliseconds, double aloneMilliseconds) noexcept {
  return gpuMilliseconds - encodeMilliseconds >= 3.0 * aloneMilliseconds;
}

} // namespace splash::ops::ane_ffn
