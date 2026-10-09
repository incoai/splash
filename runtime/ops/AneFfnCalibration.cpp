#include "ops/AneFfnCalibration.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::ops::ane_ffn {
namespace {

// The split's predicted gain below which the GPU runs the FFN alone.
constexpr double kMinimumGain = 0.05;
// How much longer than the GPU's alone a chunk's split layer may take and
// the chunk still split: six timings of each function's split layer of
// Qwen3.8-27B on an M5 Max spread over ±3%, and the FFN takes about half of
// a prefill, so a chunk that splits within this prefills at most about 1%
// slower.
constexpr double kChunkNoise = 0.02;
// The first word of a Calibration's text, which a change to its fields
// renumbers.
constexpr std::string_view kCalibrationTag = "ane-ffn-calibration-1";

// The GPU alone's milliseconds of a chunk's FFN layer by its rows: the line
// through its timings, which the GPU follows within 1% between them.
Line gpuAlone(const std::array<ChunkTimings::Gpu, 2> &gpu) {
  const auto &[low, high] = gpu;
  const double slope = (high.milliseconds - low.milliseconds) / (high.rows - low.rows);
  return {low.milliseconds - slope * low.rows, slope};
}

} // namespace

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

// Not a number where either part is not.
double Model::operator()(double share) const noexcept {
  const double a = ane(share), g = gpu(share) + u * a;
  return a > g ? a : g;
}

// The model, from the split's layers as they run between stand-ins for the
// mixer (Measurement). The GPU's part alone gives G at both shares. At the
// high share, near 0.8, the ANE's part takes the longer on every Mac the
// split was measured on, whose fastest full chunks of Qwen3.8-27B's split
// layers take 8-9 of its 34 units on an M5 Max, 14-15 on an M5 Pro and 26-27
// on an M6: the split layer there is A, which scales with the ANE's channels.
// Where the GPU's part takes the longer at the low share, the split layer
// there exceeds it by the bandwidth the ANE's part takes beside it, which
// gives u: on an M6 the GPU's part runs a tenth to a third longer beside the
// ANE's, on an M5 Pro 3-7% near its fastest share. Where the ANE's part takes
// the longer there, as on an M5 Max, the split layer's excess over A is the
// ANE's own. Where the ANE's part does not take the longer at the high share,
// A comes out longer than it is, and the choice smaller.
Model fit(const Timing &low, const Timing &high) {
  if (!(low.share < high.share)) throw std::invalid_argument("ANE FFN calibration needs a low and a high share");
  const double slope = (high.gpu - low.gpu) / (high.share - low.share);
  Model model{{low.gpu - slope * low.share, slope}, {0.0, high.split / high.share}};
  const double gpu = model.gpu(low.share), ane = model.ane(low.share);
  if (gpu > ane) model.u = std::clamp((low.split - gpu) / ane, 0.0, 1.0);
  return model;
}

// The units whose T, or T a unit either way, is least at its longest: the
// timings can put T's least a unit off, and a unit past where the ANE's part
// takes the longer costs the ANE's time per unit, a unit short of it the
// GPU's. On an M5 Max and an M5 Pro the ANE's is the longer, 1.7 ms a unit of
// Qwen3.8-27B's full-chunk layer against the GPU's 0.6 and 1.1 ms, and the
// choice keeps below T's least; on an M6 the GPU's is, 1.9 ms against 1.2,
// and it need not. Chunks of fewer rows, which the ANE pads up to its
// functions' rows, move T's least lower still; minimumRows keeps the split
// from the chunks it loses. The choice runs only if T there gains
// kMinimumGain on the GPU alone.
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

std::string Calibration::text() const {
  std::ostringstream text;
  text << std::setprecision(9) << kCalibrationTag << ' ' << aneUnits << ' ' << minimumRows;
  for (const ChunkTimings::Gpu &point : gpu) text << ' ' << point.rows << ' ' << point.milliseconds;
  return text.str();
}

// A record of units holds least rows and the GPU alone at ascending rows of
// usable milliseconds; one of none holds neither.
std::optional<Calibration> Calibration::parse(std::string_view text) {
  std::istringstream in{std::string(text)};
  std::string tag;
  Calibration calibration;
  in >> tag >> calibration.aneUnits >> calibration.minimumRows;
  for (ChunkTimings::Gpu &point : calibration.gpu) in >> point.rows >> point.milliseconds;
  std::string rest;
  if (!in || tag != kCalibrationTag || (in >> rest)) return std::nullopt;
  const auto &[low, high] = calibration.gpu;
  const auto usableMilliseconds = [](double value) { return value > 0.0 && std::isfinite(value); };
  if (!calibration.aneUnits) return calibration.minimumRows ? std::nullopt : std::optional(Calibration{});
  if (!calibration.minimumRows || !(low.rows < high.rows) || !usableMilliseconds(low.milliseconds) ||
      !usableMilliseconds(high.milliseconds))
    return std::nullopt;
  return calibration;
}

// Each layer the GPU waits for the Neural Engine's evaluation before it joins
// the ANE's part to its own, so evaluations that take as long as the GPU's
// whole layers alone make the split slower than the GPU alone, whatever the
// rest of the layer takes: the breaker trips there, and never short of it. An
// evaluation counts against the GPU alone's layer of its function's rows,
// which the chunk pads up to, so the GPU alone would take no longer over the
// chunk; the reason scales the window's ratio to a layer of the most rows.
// At the units calibration chooses, at or just below where the parts meet
// (choose), an evaluation takes at most about as long as the GPU's part: on
// an M5 Max, calibration's fit of Qwen3.8-27B puts it at 13.2 ms against the
// GPU alone's 19.0 at 8 of 34 units, and the M5 Pro's 0.44 and the M6's 0.76
// leave more room.
// Measured, another process keeping the ANE busy beside the split costs it
// 2-8% on an M5 Max, an M5 Pro and an M6, and a minute of prefill slows the
// M6's ANE by a third: far short of the trip point. Judging kWindow commands
// at once, it weighs one slow evaluation, as a function's first after its
// program loads again, among all of theirs.
Breaker::Breaker(const std::array<ChunkTimings::Gpu, 2> &gpu) {
  const auto &[low, high] = gpu;
  if (!(low.rows < high.rows)) throw std::invalid_argument("ANE FFN breaker needs the GPU alone at two row counts");
  static_cast<void>(usable(low.milliseconds));
  static_cast<void>(usable(high.milliseconds));
  gpu_ = gpuAlone(gpu);
  mostRows_ = high.rows;
}

std::string Breaker::add(uint32_t rows, uint32_t evaluations, double milliseconds) {
  if (!gpu_) return {};
  ane_ += milliseconds;
  alone_ += evaluations * (*gpu_)(rows);
  if (++commands_ < kWindow) return {};
  const double ratio = ane_ / alone_, most = (*gpu_)(mostRows_);
  commands_ = 0;
  ane_ = alone_ = 0.0;
  if (!(ratio >= 1.0)) return {};
  std::ostringstream reason;
  reason << std::fixed << std::setprecision(1) << "losing to the GPU alone (" << ratio * most
         << " ms on the Neural Engine per " << mostRows_ << "-row layer against " << most << " ms on the GPU alone)";
  return reason.str();
}

} // namespace splash::ops::ane_ffn
