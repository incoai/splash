#include "ops/AneFfnMeasurement.hpp"

#include "AwakeClock.hpp"

#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops::ane_ffn {
namespace {

// The shares calibration times the split at, a low one where the GPU's part
// takes the longer and a high one where the ANE's does (fit()): with the
// GPU's part on the bf16 kernels, and in int8, which runs it 1.6 times as
// fast on an M5 Max and moves the best share from about 0.3 to about 0.2
// there (0.15 and 0.5 also straddle the best int8 share of an M5 Pro's and
// an M5's slower GPUs).
constexpr std::array<double, 2> kTimedShares{0.4, 0.8}, kInt8TimedShares{0.15, 0.5};
// The layers time() spreads over the model; each of its commands takes all
// but the last, which gives the last one's next weights to stage.
constexpr uint32_t kSampledLayers = 5;
// The rows of layers of each command chunks() times, and the runs of each
// command.
constexpr uint32_t kCommandRows = 2 * AneFfn::kMaximumRows, kRuns = 3;

// The median of kRuns of `run`'s milliseconds, which passes over a first run
// that wires a function.
double median(const std::function<double()> &run) {
  std::array<double, kRuns> runs{};
  for (double &milliseconds : runs) milliseconds = run();
  std::ranges::nth_element(runs, runs.begin() + kRuns / 2);
  return runs[kRuns / 2];
}

} // namespace

Measurement::Measurement(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers,
                         const PrefillFfnBuffers &ffn, const std::array<metal::MetalBuffer, 2> &hidden,
                         std::function<bool()> interrupted, Int8Ffn *int8)
    : backend_(backend), layers_(layers), ffn_(ffn), hidden_(hidden), interrupted_(std::move(interrupted)),
      int8_(int8), linear_(backend.capabilities()) {
  if (const char *reason = AneFfn::unsupported(layers)) throw std::invalid_argument(reason);
  if (layers.size() < 2) throw std::invalid_argument("ANE FFN split timed on fewer than two layers");
  // Timing does not depend on the values, but on their being finite.
  AneFfn::fillNormalized(ffn_.normalized);
  for (const metal::MetalBuffer &buffer : hidden_)
    if (void *data = buffer.contents()) std::memset(data, 0, buffer.sizeBytes());
}

// Commands of layers in the order a prefill runs them. Each split layer
// follows the GPU alone's FFN of the same layer into the other hidden rows,
// which stands in for the mixer before it in a prefill: the ANE idles as long
// between its evaluations, and a layer that hands its inputs to the ANE after
// little GPU work can start the next command buffer late, by tens of
// milliseconds at random on the M5 Max with Qwen3.8-27B resident, which would
// time Metal's scheduling instead. `standIn`, the GPU's layer alone of those
// rows timed alike, takes away the stand-in and with it the command's
// submission. Each command takes `count` layers from the first, over which
// the jitter of the handoffs and the staging of the first layer's weights,
// which a prefill's command does once, average out, and the median of kRuns
// gives a layer's milliseconds. Each layer timed stages the next one's
// weights. The stand-in runs on the bf16 kernels also when the split's GPU
// part runs in int8, sharing the int8 FFN's scratch, which its own layers
// in the same command would overwrite.
double Measurement::layer(std::span<const SwiGluProjections> layers, uint32_t count, uint32_t rows, AneFfn *split,
                          double standIn, Int8Ffn *int8) {
  if (!count || count >= layers.size()) throw std::invalid_argument("ANE FFN layers timed without one to stage");
  const auto command = [&] {
    metal::CommandGraph graph;
    if (split && !split->begin()) throw std::runtime_error("ANE FFN split stopped (" + split->reason() + ")");
    for (uint32_t index = 0; index < count; ++index) {
      if (split) {
        linear_.addPrefillSwiGlu(graph, layers[index], ffn_, hidden_[0], hidden_[1], rows);
        split->add(graph, index, ffn_, hidden_[1], hidden_[0], rows);
      } else if (int8) {
        int8->add(graph, index, ffn_, hidden_[1], hidden_[0], rows);
      } else {
        linear_.addPrefillSwiGlu(graph, layers[index], ffn_, hidden_[1], hidden_[0], rows);
      }
    }
    const auto start = AwakeClock::now();
    if (split) {
      static_cast<void>(split->commit(graph, {}).wait());
      if (!split->completed())
        throw std::runtime_error("ANE FFN split of " + std::to_string(rows) + " rows failed (" + split->reason() +
                                 ")");
    } else {
      static_cast<void>(backend_.submitCommandAsync(graph.command()).wait());
    }
    return millisecondsSince(start) / count;
  };
  return usable(median(command) - standIn);
}

// Commands of all but the last of `layers` and of the first alone, of the
// split's GPU part of full chunks, the median of kRuns each: their
// difference cancels each command's fixed costs, its submission and the
// staging of its first layer's weights, which a prefill's command does once
// for all its layers: about 1 ms a command on an M5 Max, against a GPU part
// of 5-12 ms a layer.
double Measurement::gpuPart(std::span<const SwiGluProjections> layers, AneFfn &split) {
  const auto command = [&](uint32_t count) {
    return median([&] {
      metal::CommandGraph graph;
      if (!split.begin()) throw std::runtime_error("ANE FFN split stopped (" + split.reason() + ")");
      for (uint32_t index = 0; index < count; ++index)
        split.encode(graph, index, ffn_, hidden_[index & 1], hidden_[(index & 1) ^ 1], AneFfn::kMaximumRows,
                     AneFfn::Parts::Gpu);
      const auto start = AwakeClock::now();
      static_cast<void>(backend_.submitCommandAsync(graph.command()).wait());
      return millisecondsSince(start);
    });
  };
  const auto count = static_cast<uint32_t>(layers.size() - 1);
  return usable((command(count) - command(1)) / (count - 1));
}

Timings Measurement::time() {
  const uint32_t units = AneFfn::units(layers_);
  // A model of fewer layers times some twice.
  std::vector<SwiGluProjections> sampled;
  for (uint32_t index = 0; index < kSampledLayers; ++index)
    sampled.push_back(layers_[index * (layers_.size() - 1) / (kSampledLayers - 1)]);
  constexpr uint32_t kRows = AneFfn::kMaximumRows, kCount = kSampledLayers - 1;
  // The int8 FFN of the sampled layers, sharing the model's one's scratch.
  std::optional<Int8Ffn> int8;
  if (int8_) int8.emplace(backend_, sampled, int8_);
  Timings timings;
  // The split layer's stand-in, on the bf16 kernels (layer()).
  const double standIn = layer(sampled, kCount, kRows);
  timings.gpuAlone = int8 ? layer(sampled, kCount, kRows, nullptr, 0.0, &*int8) : standIn;
  const auto measure = [&](double share) {
    const uint32_t aneUnits = nearestUnits(share, units);
    AneFfn split(backend_, sampled, aneUnits, interrupted_, std::array{kRows}, int8 ? &*int8 : nullptr);
    return Timing{static_cast<double>(aneUnits) / units, gpuPart(sampled, split),
                  layer(sampled, kCount, kRows, &split, standIn)};
  };
  const std::array<double, 2> &shares = int8 ? kInt8TimedShares : kTimedShares;
  timings.low = measure(shares[0]);
  timings.high = measure(shares[1]);
  return timings;
}

ChunkTimings Measurement::chunks(AneFfn &split) {
  std::vector<uint32_t> rows;
  for (const AneFfn::Evaluation &evaluation : split.evaluations_) rows.push_back(evaluation.rows);
  // About kCommandRows rows of layers to a command, more layers of fewer rows.
  const auto count = [&](uint32_t chunk) {
    return static_cast<uint32_t>(std::clamp<size_t>(kCommandRows / chunk, 1, layers_.size() - 1));
  };
  const auto gpu = [&](uint32_t chunk, Int8Ffn *int8) {
    return ChunkTimings::Gpu{chunk, layer(layers_, count(chunk), chunk, nullptr, 0.0, int8)};
  };
  ChunkTimings timings{{gpu(rows.front(), int8_), gpu(rows.back(), int8_)}, {}};
  // The stand-in's line, on the bf16 kernels (layer()).
  const std::array<ChunkTimings::Gpu, 2> standIn =
      int8_ ? std::array{gpu(rows.front(), nullptr), gpu(rows.back(), nullptr)} : timings.gpu;
  const auto &[low, high] = standIn;
  const double slope = (high.milliseconds - low.milliseconds) / (high.rows - low.rows);
  for (size_t function = rows.size(); function-- > 0;)
    timings.functions.push_back({function ? rows[function - 1] + 1 : rows.front(), rows[function],
                                 layer(layers_, count(rows[function]), rows[function], &split,
                                       low.milliseconds + slope * (rows[function] - low.rows))});
  return timings;
}

} // namespace splash::ops::ane_ffn
