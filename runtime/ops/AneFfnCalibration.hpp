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
// What calibration times: the split at two shares, and a full chunk's FFN
// layer on the GPU alone.
struct Timings final {
  Timing low, high;
  double gpuAlone = 0.0;
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
// A full chunk's split FFN layer at share s takes
//   T(s) = max(A(s), G(s) + u A(s)):
// the ANE's part A(s), which the layer waits on where it takes the longer,
// or the GPU's part G(s), slowed by the memory bandwidth the ANE's part takes
// beside it, u A(s). G is close to linear in s, and A to proportional.
struct Model final {
  Line gpu, ane;
  double u = 0.0;
  [[nodiscard]] double operator()(double share) const noexcept;
};
// The model through the timings of a low and a high share
// (AneFfnCalibration.cpp). Throws unless the low share is the lower.
[[nodiscard]] Model fit(const Timing &low, const Timing &high);

// The ANE's channel units, of the `units` the intermediate channels hold, at
// which chunks split, at most `maxAneUnits` of them: those at which T, or T a
// unit either way, is least at its longest (AneFfnCalibration.cpp); none if
// `model` predicts no gain worth the split there on `gpuAlone` milliseconds
// of the GPU alone.
[[nodiscard]] std::optional<uint32_t> choose(const Model &model, double gpuAlone, uint32_t units,
                                             uint32_t maxAneUnits);

// A chunk's FFN layer in milliseconds: the GPU's alone at the least and the
// most rows of the split's functions, and the split's of each function, from
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

// What one calibration of the split on this Mac decided, which later starts
// take again (engine::startAneFfn): the ANE's units and the least rows of a
// chunk the split takes, none of either when no split gained enough; and the
// GPU alone's layer at the least and the most rows of the split's functions,
// which the breaker judges the split against.
struct Calibration final {
  uint32_t aneUnits = 0, minimumRows = 0;
  std::array<ChunkTimings::Gpu, 2> gpu{};

  // As one line of text, and back: none for text that is not such a line.
  [[nodiscard]] std::string text() const;
  [[nodiscard]] static std::optional<Calibration> parse(std::string_view text);
};

// When a split that serves loses to the GPU alone (AneFfnCalibration.cpp):
// over each kWindow commands it finishes, the Neural Engine's milliseconds
// against the GPU alone's of as many layers.
class Breaker final {
public:
  static constexpr uint32_t kWindow = 8;

  // One that never trips, for a split whose GPU alone was not timed.
  Breaker() = default;
  // One against the GPU alone's layer as timed at the least and the most rows
  // of the split's functions. Throws unless those are ascending rows of
  // usable milliseconds.
  explicit Breaker(const std::array<ChunkTimings::Gpu, 2> &gpu);

  // A command the split finished: `evaluations` of its function of `rows`
  // rows, from the least to the most rows timed, took the Neural Engine
  // `milliseconds` in all. Why the split loses to the GPU alone, once a
  // window of commands shows it; empty otherwise.
  [[nodiscard]] std::string add(uint32_t rows, uint32_t evaluations, double milliseconds);

private:
  // The GPU alone's milliseconds of a layer by its rows, and the most rows.
  std::optional<Line> gpu_;
  uint32_t mostRows_ = 0;
  // The window's commands, the Neural Engine's milliseconds over them and the
  // GPU alone's of their layers.
  uint32_t commands_ = 0;
  double ane_ = 0.0, alone_ = 0.0;
};

} // namespace splash::ops::ane_ffn
