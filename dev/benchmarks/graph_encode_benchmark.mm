// Times the host side of a command without model weights: building a
// CommandGraph whose dispatches each bind k buffers and one parameter struct,
// submitCommandAsync up to its return (validation, encoding and commit), and
// the graph's destruction, for the binding counts production dispatches use.
// Every dispatch runs one no-op kernel, so GPU time stays out of the
// measurement. Each dispatch's buffers are a braced list, as production call
// sites write them, and its pipeline name is longer than a std::string holds
// inline, as production names are.
//
// It also times DecodeArena::batchSlice, copied out as its callers do, over
// every non-empty tensor and lane count of the decode arena of a Qwen3.8-27B
// target and a two-layer DFlash draft.
//
// Each figure is the median over --repeat graphs, after untimed warm-up
// graphs, in nanoseconds per dispatch (per call for batchSlice).
//
// usage: graph-encode-benchmark METALLIB [--dispatches N] [--repeat N]
// METALLIB is build/engine-tests/metal-backend.metallib, which holds the
// no-op kernel.
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/DFlashDraft.hpp"
#include "model/Qwen3_8.hpp"
#include "model/RuntimeArenas.hpp"
#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace splash;
using Clock = std::chrono::steady_clock;
using metal::CommandGraph;
using metal::MetalBuffer;

constexpr const char *kPipeline = "graph_encode_benchmark_noop_dispatch";
constexpr metal::DispatchSize kGroups{1, 1, 1};
constexpr uint32_t kWarmupGraphs = 3;
// The binding counts timed: the common projection and norm shapes, a sparse
// attention split's 18, and the batched GDN decode, short convolution and
// block indexer commit of eight lanes (25 to 28).
constexpr std::array<uint32_t, 8> kBindingCounts{3, 6, 9, 12, 18, 22, 25, 28};
constexpr uint32_t kMostBindings = 28;
static_assert(*std::max_element(kBindingCounts.begin(), kBindingCounts.end()) == kMostBindings);

// A production-sized parameter struct.
struct Params {
  std::array<uint32_t, 16> values{};
};

using Buffers = std::array<MetalBuffer, kMostBindings>;

template <size_t... Index>
void addDispatch(CommandGraph &graph, const Buffers &buffers, const Params &params,
                 std::index_sequence<Index...>) {
  graph.add(kPipeline, {buffers[Index]...}, params, kGroups);
}

template <size_t Bindings>
void addDispatches(CommandGraph &graph, const Buffers &buffers, uint32_t dispatches) {
  Params params;
  for (uint32_t dispatch = 0; dispatch < dispatches; ++dispatch) {
    params.values[0] = dispatch;
    addDispatch(graph, buffers, params, std::make_index_sequence<Bindings>{});
  }
}

void addDispatches(uint32_t bindings, CommandGraph &graph, const Buffers &buffers,
                   uint32_t dispatches) {
  switch (bindings) {
  case 3: return addDispatches<3>(graph, buffers, dispatches);
  case 6: return addDispatches<6>(graph, buffers, dispatches);
  case 9: return addDispatches<9>(graph, buffers, dispatches);
  case 12: return addDispatches<12>(graph, buffers, dispatches);
  case 18: return addDispatches<18>(graph, buffers, dispatches);
  case 22: return addDispatches<22>(graph, buffers, dispatches);
  case 25: return addDispatches<25>(graph, buffers, dispatches);
  case 28: return addDispatches<28>(graph, buffers, dispatches);
  default: throw std::logic_error("untimed binding count");
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double nanoseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::nano>(duration).count();
}

struct GraphFigures {
  double build = 0, submit = 0, destroy = 0;
};

GraphFigures timeGraphs(metal::MetalBackend &backend, const Buffers &buffers, uint32_t bindings,
                        uint32_t dispatches, uint32_t repeat) {
  std::vector<double> build, submit, destroy;
  for (uint32_t graphIndex = 0; graphIndex < kWarmupGraphs + repeat; ++graphIndex) {
    const Clock::time_point start = Clock::now();
    auto graph = std::make_unique<CommandGraph>();
    addDispatches(bindings, *graph, buffers, dispatches);
    const Clock::time_point built = Clock::now();
    metal::CommandTicket ticket = backend.submitCommandAsync(graph->dispatches());
    const Clock::time_point submitted = Clock::now();
    static_cast<void>(ticket.wait());
    const Clock::time_point released = Clock::now();
    graph.reset();
    const Clock::time_point destroyed = Clock::now();
    if (graphIndex < kWarmupGraphs)
      continue;
    build.push_back(nanoseconds(built - start) / dispatches);
    submit.push_back(nanoseconds(submitted - built) / dispatches);
    destroy.push_back(nanoseconds(destroyed - released) / dispatches);
  }
  return {median(build), median(submit), median(destroy)};
}

model::RuntimeGeometry qwen27bGeometry() {
  constexpr std::array<uint32_t, 2> captured{1, 3};
  const model::Qwen3_8Layout layout;
  model::RuntimeGeometry geometry;
  geometry.target = model::QwenTargetGeometry(layout, captured);
  geometry.target.kvLayout = layout.kvLayout();
  geometry.draft = model::draftLayout(layout.hiddenSize, layout.vocabularySize, 2, 1024, 10'000'000.0F, captured);
  return geometry;
}

double timeBatchSlices(metal::MetalBackend &backend, uint32_t repeat) {
  const ops::ExecutionPlans operators(backend.capabilities(), ops::Int8Activations::On);
  const model::RuntimeGeometry geometry = qwen27bGeometry();
  const model::DecodeArena arena(backend, geometry, operators);
  const uint32_t lanes = model::decodeArenaLanes(geometry, operators);
  std::vector<std::pair<model::DecodeTensor, uint32_t>> slices;
  for (uint32_t tensor = 0; tensor < model::decodeTensorCount; ++tensor) {
    const auto decodeTensor = static_cast<model::DecodeTensor>(tensor);
    if (!arena.batchSlice(decodeTensor, 1))
      continue;
    for (uint32_t width = 1; width <= lanes; ++width)
      slices.emplace_back(decodeTensor, width);
  }
  constexpr uint32_t kRounds = 100;
  std::vector<double> perCall;
  for (uint32_t round = 0; round < kWarmupGraphs + repeat; ++round) {
    const Clock::time_point start = Clock::now();
    for (uint32_t pass = 0; pass < kRounds; ++pass) {
      for (const auto &[tensor, width] : slices) {
        MetalBuffer slice = arena.batchSlice(tensor, width);
        asm volatile("" : : "r"(&slice) : "memory");
      }
    }
    if (round >= kWarmupGraphs)
      perCall.push_back(nanoseconds(Clock::now() - start) / (double(kRounds) * slices.size()));
  }
  return median(perCall);
}

uint32_t number(std::string_view text, std::string_view option) {
  uint32_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || !value)
    throw std::invalid_argument(std::string(option) + " takes a positive integer");
  return value;
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    if (argc < 2) {
      std::cerr << "usage: graph-encode-benchmark METALLIB [--dispatches N] [--repeat N]\n";
      return 64;
    }
    uint32_t dispatches = 1000, repeat = 25;
    for (int index = 2; index < argc; ++index) {
      const std::string_view option(argv[index]);
      if (index + 1 >= argc)
        throw std::invalid_argument(std::string(option) + " needs a value");
      if (option == "--dispatches")
        dispatches = number(argv[++index], option);
      else if (option == "--repeat")
        repeat = number(argv[++index], option);
      else
        throw std::invalid_argument("unknown option " + std::string(option));
    }
    metal::MetalBackend backend(argv[1]);
    Buffers buffers;
    for (MetalBuffer &buffer : buffers)
      buffer = backend.allocateBuffer(256, metal::BufferStorage::Shared, "graph-encode-benchmark");

    std::cout << std::fixed << std::setprecision(1) << dispatches << " dispatches a graph, median of "
              << repeat << " graphs, ns per dispatch\n"
              << "bindings     build    submit   destroy     total\n";
    for (const uint32_t bindings : kBindingCounts) {
      const GraphFigures figures = timeGraphs(backend, buffers, bindings, dispatches, repeat);
      std::cout << std::setw(8) << bindings << std::setw(10) << figures.build << std::setw(10)
                << figures.submit << std::setw(10) << figures.destroy << std::setw(10)
                << figures.build + figures.submit + figures.destroy << '\n';
    }
    std::cout << "batchSlice " << timeBatchSlices(backend, repeat) << " ns per call\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "graph-encode-benchmark: " << error.what() << '\n';
    return 1;
  }
}
