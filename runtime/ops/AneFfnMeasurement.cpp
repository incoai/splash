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

// The shares calibration times the split at.
constexpr std::array<double, 2> kTimedShares{0.4, 0.8};
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
                         std::function<bool()> interrupted, uint32_t slots,
                         std::span<const MixerProjections> mixers)
    : backend_(backend), layers_(layers), ffn_(ffn), hidden_(hidden), interrupted_(std::move(interrupted)),
      mixers_(mixers), slots_(slots), linear_(linear) {
  if (!slots) throw std::invalid_argument("ANE FFN split timed without a slot");
  if (const char *reason = AneFfn::unsupported(layers)) throw std::invalid_argument(reason);
  if (layers.size() < 2) throw std::invalid_argument("ANE FFN split timed on fewer than two layers");
  if (!mixers.empty() && mixers.size() != layers.size())
    throw std::invalid_argument("ANE FFN split timed with mixers of other layers");
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
    if (split && !split->begin()) throw std::runtime_error("ANE FFN split stopped (" + split->reason() + ")");
    for (uint32_t index = 0; index < count; ++index) {
      if (split) {
        linear_.addSwiGlu(graph, layers[index], ffn_, hidden_[0], hidden_[1], rows);
        split->add(graph, index, ffn_, hidden_[1], hidden_[0], rows);
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

std::vector<SwiGluProjections> Measurement::sampled() const {
  // A model of fewer layers times some twice.
  std::vector<SwiGluProjections> layers;
  for (uint32_t index = 0; index < kSampledLayers; ++index)
    layers.push_back(layers_[index * (layers_.size() - 1) / (kSampledLayers - 1)]);
  return layers;
}

// A mixer's GPU work over its projections': the scan or attention, the norms
// and the rest a prefill's layer runs beside them. Qwen3.8-27B's mixers took
// 34.7 ms a full chunk's layer at 8K of context on a 12-core M6, of which
// their projections 26 (dev/benchmarks' decode-profile), and 13.4 ms on a
// 32-core M5 Max (the GPU alone's 36.4 ms layer less its FFN's 23.0), of
// which their projections 9.3.
constexpr double kMixerOverProjections = 4.0 / 3.0;

// Each sampled layer's input projection from the normalized rows into the
// intermediate rows, and its output projection from gate's output rows into
// the hidden rows, as a prefill's mixer runs them on a full chunk, in
// commands of all but the last layer, the median of kRuns, times
// kMixerOverProjections.
double Measurement::mixer() {
  if (mixers_.empty()) return 0.0;
  std::vector<MixerProjections> mixers;
  for (uint32_t index = 0; index < kSampledLayers; ++index)
    mixers.push_back(mixers_[index * (mixers_.size() - 1) / (kSampledLayers - 1)]);
  constexpr uint32_t kRows = AneFfn::kMaximumRows, kCount = kSampledLayers - 1;
  const auto fits = [&](const metal::MetalBuffer &buffer, uint32_t width) {
    return uint64_t{kRows} * width * sizeof(uint16_t) <= buffer.sizeBytes();
  };
  for (const MixerProjections &projections : mixers)
    if (!fits(ffn_.intermediate, projections.input->outputSize) ||
        !fits(ffn_.gateScratch, projections.output->inputSize))
      throw std::invalid_argument("ANE FFN split timed with mixer projections wider than its FFN's");
  return usable(median([&] {
    metal::CommandGraph graph;
    for (uint32_t index = 0; index < kCount; ++index) {
      linear_.addProjection(graph, ffn_.normalized, *mixers[index].input, ffn_.intermediate, kRows, ffn_.scratch);
      linear_.addProjection(graph, ffn_.gateScratch, *mixers[index].output, hidden_[1], kRows, ffn_.scratch);
    }
    const auto start = AwakeClock::now();
    static_cast<void>(backend_.submitCommandAsync(graph.command()).wait());
    return millisecondsSince(start) / kCount;
  })) * kMixerOverProjections;
}

// Each layer starts every slot's chunk, the slot's previous layer joined
// first, and stages the next layer once all have started, as a pipelined
// prefill runs them without the mixers between: the ANE runs one evaluation
// after another, and its time over them (AneFfn::completed), the median of
// kRuns, is what a pipelined command takes of it, without the handoff's wait
// for each layer's GPU work that a split layer timed alone adds (7% of the
// ANE's whole FFN of Qwen3.8-27B on an M5 Max).
double Measurement::pipelinedAne(std::span<const SwiGluProjections> layers, AneFfn &split) {
  constexpr uint32_t kRows = AneFfn::kMaximumRows;
  const auto count = static_cast<uint32_t>(layers.size() - 1), slots = split.slots();
  return usable(median([&] {
    metal::CommandGraph graph;
    if (!split.begin()) throw std::runtime_error("ANE FFN split stopped (" + split.reason() + ")");
    for (uint32_t index = 0; index < count; ++index) {
      for (uint32_t slot = 0; slot < slots; ++slot) {
        if (index) split.join(graph, slot);
        split.start(graph, index, slot, ffn_, hidden_[index & 1], hidden_[(index & 1) ^ 1], kRows);
      }
      split.stageNext(graph, index);
    }
    for (uint32_t slot = 0; slot < slots; ++slot) split.join(graph, slot);
    static_cast<void>(split.commit(graph, {}).wait());
    const std::optional<AwakeClock::duration> ran = split.completed();
    if (!ran) throw std::runtime_error("ANE FFN split failed while timed (" + split.reason() + ")");
    return std::chrono::duration<double, std::milli>(*ran).count() / (count * slots);
  }));
}

Timings Measurement::time() {
  const uint32_t units = AneFfn::units(layers_);
  Timings timings;
  timings.gpuAlone = layer(sampled(), kSampledLayers - 1, AneFfn::kMaximumRows);
  if (slots_ > 1) timings.mixer = mixer();
  timings.low = time(nearestUnits(kTimedShares[0], units), timings.gpuAlone);
  timings.high = time(nearestUnits(kTimedShares[1], units), timings.gpuAlone);
  return timings;
}

Timing Measurement::time(uint32_t aneUnits, double gpuAlone) {
  const std::vector<SwiGluProjections> layers = sampled();
  constexpr uint32_t kRows = AneFfn::kMaximumRows;
  AneFfn split(backend_, linear_, layers, aneUnits, interrupted_, slots_);
  return Timing{static_cast<double>(aneUnits) / AneFfn::units(layers_), gpuPart(layers, split),
                layer(layers, kSampledLayers - 1, kRows, &split, gpuAlone),
                slots_ > 1 ? pipelinedAne(layers, split) : 0.0};
}

// On the layers time() timed, as many to a command, so that the splits
// calibration compares (slowerNowhere) are timed where it fitted them. Six
// calibrations of Qwen3.8-27B MLX 4-bit on an M5 Max timed each function's
// split layer within a range of 6.1% of its mean (3.2% the median function)
// this way, against 10.9% (3.6%) over the model's first layers at about
// 4,096 rows of layers to a command, and chose the same 7 units from about
// the same least rows (842 the median against 833).
ChunkTimings Measurement::chunks(uint32_t aneUnits) {
  const std::vector<SwiGluProjections> layers = sampled();
  AneFfn split(backend_, linear_, layers, aneUnits, interrupted_);
  // Chunks of each count of pieces, which a chunk of fewer rows costs as (AneFfn::kEvaluationRows).
  std::vector<uint32_t> rows;
  for (uint32_t chunk = AneFfn::kEvaluationRows; chunk <= AneFfn::kMaximumRows; chunk += AneFfn::kEvaluationRows)
    rows.push_back(chunk);
  constexpr uint32_t kCount = kSampledLayers - 1;
  const auto gpu = [&](uint32_t chunk) { return ChunkTimings::Gpu{chunk, layer(layers, kCount, chunk)}; };
  ChunkTimings timings{{gpu(rows.front()), gpu(rows.back())}, {}};
  const auto &[low, high] = timings.gpu;
  const double slope = (high.milliseconds - low.milliseconds) / (high.rows - low.rows);
  for (size_t function = rows.size(); function-- > 0;)
    timings.functions.push_back({function ? rows[function - 1] + 1 : rows.front(), rows[function],
                                 layer(layers, kCount, rows[function], &split,
                                       low.milliseconds + slope * (rows[function] - low.rows))});
  return timings;
}

} // namespace splash::ops::ane_ffn
