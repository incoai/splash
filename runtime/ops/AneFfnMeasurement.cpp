#include "ops/AneFfnMeasurement.hpp"

#include "AwakeClock.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops::ane_ffn {
namespace {

// The layers time() and chunks() spread over the model; each of their
// commands takes all but the last, which gives the last one's next weights
// to stage.
constexpr uint32_t kSampledLayers = 5;
// The runs of each command.
constexpr uint32_t kRuns = 3;

// The median of kRuns of `run`'s milliseconds, which passes over a first run
// that wires a function.
double median(const std::function<double()> &run) {
  std::array<double, kRuns> runs{};
  for (double &milliseconds : runs) milliseconds = run();
  std::ranges::nth_element(runs, runs.begin() + kRuns / 2);
  return runs[kRuns / 2];
}

} // namespace

Measurement::Measurement(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
                         const PrefillFfnBuffers &ffn, const std::array<metal::MetalBuffer, 2> &hidden,
                         std::function<bool()> interrupted)
    : backend_(backend), layers_(layers), ffn_(ffn), hidden_(hidden), interrupted_(std::move(interrupted)),
      linear_(linear) {
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
// weights.
double Measurement::layer(std::span<const SwiGluProjections> layers, uint32_t count, uint32_t rows, AneFfn *split,
                          double standIn) {
  if (!count || count >= layers.size()) throw std::invalid_argument("ANE FFN layers timed without one to stage");
  const auto command = [&] {
    metal::CommandGraph graph;
    if (split && !split->begin(split->steps_))
      throw std::runtime_error("ANE FFN split stopped (" + split->reason() + ")");
    for (uint32_t index = 0; index < count; ++index) {
      if (split) {
        linear_.addSwiGlu(graph, layers[index], ffn_, hidden_[0], hidden_[1], rows);
        split->add(graph, index, ffn_, hidden_[1], hidden_[0], rows);
        split->join(graph);
      } else {
        linear_.addSwiGlu(graph, layers[index], ffn_, hidden_[1], hidden_[0], rows);
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
      if (!split.begin(split.steps_)) throw std::runtime_error("ANE FFN split stopped (" + split.reason() + ")");
      for (uint32_t index = 0; index < count; ++index) {
        split.encode(graph, index, ffn_, hidden_[index & 1], hidden_[(index & 1) ^ 1], AneFfn::kMaximumRows,
                     AneFfn::Parts::Gpu);
        split.join(graph);
      }
      const auto start = AwakeClock::now();
      static_cast<void>(backend_.submitCommandAsync(graph.command()).wait());
      return millisecondsSince(start);
    });
  };
  const auto count = static_cast<uint32_t>(layers.size() - 1);
  return usable((command(count) - command(1)) / (count - 1));
}

std::vector<SwiGluProjections> Measurement::sampled() const {
  // A model of fewer layers times some twice.
  std::vector<SwiGluProjections> layers;
  for (uint32_t index = 0; index < kSampledLayers; ++index)
    layers.push_back(layers_[index * (layers_.size() - 1) / (kSampledLayers - 1)]);
  return layers;
}

Readers Measurement::readers(std::span<const SwiGluProjections> layers, std::unique_ptr<AneFfn> &split) {
  constexpr uint32_t kRows = AneFfn::kMaximumRows, kCount = kSampledLayers - 1;
  return {[this, layers, &split](uint32_t aneUnits, bool half) {
            // No two splits are held at once.
            split.reset();
            split.reset(new AneFfn(backend_, linear_, layers, {aneUnits, 0}, interrupted_,
                                   {half ? std::vector{kRows / 2, kRows} : std::vector{kRows}, {}}));
          },
          [this, layers, &split] { return gpuPart(layers, *split); },
          [this, layers](uint32_t rows) { return layer(layers, kCount, rows); },
          [this, layers, &split](uint32_t rows, double standIn) {
            return layer(layers, kCount, rows, split.get(), standIn);
          }};
}

Timings Measurement::time() {
  const std::vector<SwiGluProjections> layers = sampled();
  std::unique_ptr<AneFfn> split;
  return ane_ffn::time(AneFfn::units(layers_), AneFfn::kMaximumRows, readers(layers, split));
}

Timing Measurement::time(uint32_t aneUnits) {
  const std::vector<SwiGluProjections> layers = sampled();
  std::unique_ptr<AneFfn> split;
  return ane_ffn::time(aneUnits, AneFfn::units(layers_), AneFfn::kMaximumRows, readers(layers, split));
}

// On the layers time() timed, as many to a command, so that the splits
// calibration compares (slowerNowhere) are timed where it fitted them. Six
// calibrations of Qwen3.8-27B MLX 4-bit on an M5 Max timed each function's
// split layer within a range of 6.1% of its mean (3.2% the median function)
// this way, against 10.9% (3.6%) over the model's first layers at about
// 4,096 rows of layers to a command, and chose the same 7 units from about
// the same least rows (842 the median against 833).
// The GPU alone's line runs through the least rows and a full chunk, which
// it follows within 1% between them; from 256 rows it moved Qwen3.8-27B MLX
// 4-bit's least rows on a 20-core M5 Pro from 1.7.0's 533 to 653, a command
// of 544 to 640 rows 5-12% slower on the GPU alone.
ChunkTimings Measurement::chunks(AneFfn::Shares shares) {
  const std::vector<SwiGluProjections> layers = sampled();
  const std::array<std::vector<uint32_t>, 2> functionRows = AneFfn::programRows(shares);
  const std::vector<uint32_t> &rows = functionRows[static_cast<size_t>(AneFfn::Steps::One)];
  if (rows.empty()) throw std::invalid_argument("ANE FFN chunks timed of a split without one step's share");
  AneFfn split(backend_, linear_, layers, shares, interrupted_, functionRows);
  constexpr uint32_t kCount = kSampledLayers - 1;
  const auto gpu = [&](uint32_t chunk) { return ChunkTimings::Gpu{chunk, layer(layers, kCount, chunk)}; };
  ChunkTimings timings{{gpu(AneFfn::kMinimumRows), gpu(AneFfn::kMaximumRows)}, {}};
  const auto &[low, high] = timings.gpu;
  const double slope = (high.milliseconds - low.milliseconds) / (high.rows - low.rows);
  for (size_t function = rows.size(); function-- > 0;)
    timings.functions.push_back({function ? rows[function - 1] + 1 : rows.front(), rows[function],
                                 layer(layers, kCount, rows[function], &split,
                                       low.milliseconds + slope * (rows[function] - low.rows))});
  return timings;
}

} // namespace splash::ops::ane_ffn
