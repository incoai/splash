// The draft's attention core (DraftAttention::addDecode: the split and the
// reduce), timed as a decode cycle runs it: a stream of `layers` layers of
// distinct rings in one command, for each lane count and split count, so that
// the attention law's costs of the draft's ring can be fitted on every core
// count.
// Kernel timings: the rings hold random values, every lane's cache at
// `cache` tokens (past the window, its ring is full and wrapped).
//
//   draft-attention-sweep METALLIB [--lanes 1,2,4,8] [--splits 1,2,...,16]
//                         [--cache 4096] [--layers 5] [--rounds 15]
//
// Prints for each lane count and split count the median and the least fused
// GPU microseconds a layer over `rounds` rounds that run every split count
// in turn, after a warm-up. A cycle's other kernels pass far more bytes between its
// attention layers than a GPU's caches hold, so a run of many layers
// (--layers 20) reads its rings from memory as a cycle does.

#include "metal/MetalBackend.hpp"
#include "metal/abi/DraftAttention.h"
#include "ops/DraftAttention.hpp"
#include "ops/KernelCalibration.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

namespace {

constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kWindow = SPLASH_DRAFT_SLIDING_WINDOW;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;

std::vector<uint32_t> parseList(const std::string &text) {
  std::vector<uint32_t> values;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t end = text.find(',', start);
    const std::string part = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (part.empty()) throw std::invalid_argument("an empty list entry");
    values.push_back(static_cast<uint32_t>(std::stoul(part)));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return values;
}

// Small bf16 values of a fixed sequence: 0.5 + k/1024 for k in [0, 64).
MetalBuffer bfloatBuffer(MetalBackend &backend, uint64_t count, uint32_t seed, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(count * sizeof(uint16_t), BufferStorage::Shared, label);
  auto *values = static_cast<uint16_t *>(buffer.contents());
  uint32_t state = seed * 2654435761u + 1;
  for (uint64_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    const float value = (static_cast<float>(state >> 26) / 64.0f - 0.5f) * 0.25f;
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    values[index] = static_cast<uint16_t>(bits >> 16);
  }
  return buffer;
}

struct Layer final {
  MetalBuffer queries;
  std::vector<MetalBuffer> keys;
  std::vector<MetalBuffer> values;
  MetalBuffer queryKeys;
  MetalBuffer queryValues;
};

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument("usage: draft-attention-sweep METALLIB [--lanes L,...] [--splits S,...] "
                                  "[--cache N] [--layers N] [--rounds N]");
    std::vector<uint32_t> laneCounts{1, 2, 4, 8};
    std::vector<uint32_t> splitCounts;
    for (uint32_t splits = 1; splits <= SPLASH_DRAFT_ATTENTION_MOST_SPLITS; ++splits) splitCounts.push_back(splits);
    uint32_t cache = 2 * kWindow, layers = 5, rounds = 15;
    for (int index = 2; index < argc; index += 2) {
      const std::string option = argv[index];
      if (index + 1 >= argc) throw std::invalid_argument(option + " takes a value");
      const std::string value = argv[index + 1];
      if (option == "--lanes") laneCounts = parseList(value);
      else if (option == "--splits") splitCounts = parseList(value);
      else if (option == "--cache") cache = static_cast<uint32_t>(std::stoul(value));
      else if (option == "--layers") layers = static_cast<uint32_t>(std::stoul(value));
      else if (option == "--rounds") rounds = static_cast<uint32_t>(std::stoul(value));
      else throw std::invalid_argument("unknown option " + option);
    }
    for (const uint32_t lanes : laneCounts)
      if (!lanes || lanes > kLanes) throw std::invalid_argument("--lanes takes 1 to a batch's lanes");
    for (const uint32_t splits : splitCounts)
      if (!splits || splits > SPLASH_DRAFT_ATTENTION_MOST_SPLITS)
        throw std::invalid_argument("--splits takes 1 to the ring's tiles");
    if (!layers || !rounds) throw std::invalid_argument("--layers and --rounds take at least 1");

    MetalBackend backend(argv[1]);
    const auto &device = backend.capabilities();
    const KernelCalibration calibration = KernelCalibration::of(device);
    const uint32_t cores = calibration.cores;
    std::cout << "device " << device.deviceName << ", " << cores << " GPU cores, cache " << cache << " tokens, "
              << layers << " layers a command, " << rounds << " rounds\n";

    constexpr uint32_t hidden = 2304;
    const DraftAttentionShape shape{hidden,
                                    draft_dynamic_width(hidden),
                                    SPLASH_DRAFT_QKV_WIDTH,
                                    SPLASH_DRAFT_ATTENTION_WIDTH,
                                    SPLASH_DRAFT_QUERY_HEADS,
                                    SPLASH_DRAFT_KV_HEADS,
                                    SPLASH_DRAFT_HEAD_DIMENSION};
    const uint64_t ringElements = uint64_t{SPLASH_DRAFT_KV_HEADS} * kWindow * SPLASH_DRAFT_HEAD_DIMENSION;
    // One ring tile past the ring, as the cycle's rings are followed by other
    // tensors: a tile read past it would not fault.
    const uint64_t tailElements = uint64_t{SPLASH_DRAFT_ATTENTION_TILE} * SPLASH_DRAFT_HEAD_DIMENSION;

    for (const uint32_t lanes : laneCounts) {
      const uint64_t queryBytes =
          DraftAttention::plan(shape, lanes, SPLASH_DRAFT_ATTENTION_MOST_SPLITS).workspace().groupedQueriesBytes;
      std::vector<Layer> stream(layers);
      for (uint32_t layer = 0; layer < layers; ++layer) {
        Layer &l = stream[layer];
        l.queries = bfloatBuffer(backend, queryBytes / 2, layer * 31 + lanes, "draft queries");
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
          // Lanes past the step's bind its last lane's rings, as the cycle's do.
          if (lane < lanes) {
            l.keys.push_back(bfloatBuffer(backend, ringElements + tailElements, layer * 977 + lane * 7 + 1, "draft keys"));
            l.values.push_back(
                bfloatBuffer(backend, ringElements + tailElements, layer * 977 + lane * 7 + 2, "draft values"));
          } else {
            l.keys.push_back(l.keys[lanes - 1]);
            l.values.push_back(l.values[lanes - 1]);
          }
        }
        l.queryKeys = bfloatBuffer(backend, uint64_t{lanes} * SPLASH_DRAFT_KV_HEADS * kRows * SPLASH_DRAFT_HEAD_DIMENSION,
                                   layer * 13 + 3, "draft query keys");
        l.queryValues = bfloatBuffer(backend, uint64_t{lanes} * SPLASH_DRAFT_KV_HEADS * SPLASH_DRAFT_HEAD_DIMENSION * kRows,
                                     layer * 13 + 4, "draft query values");
      }
      const std::vector<uint32_t> cacheLengths(lanes, cache);
      std::vector<CommandGraph> graphs;
      for (const uint32_t splits : splitCounts) {
        const DraftAttentionPlan plan = DraftAttention::plan(shape, lanes, splits);
        CommandGraph graph;
        for (const Layer &l : stream)
          DraftAttention::addDecode(graph, {l.queries, l.keys, l.values, l.queryKeys, l.queryValues}, cacheLengths,
                                    plan);
        graphs.push_back(std::move(graph));
      }
      // Every split count each round, after a warm-up of at least half a
      // second of GPU time, so that the clocks have risen before the first
      // measurement and drift reaches every count alike.
      double warm = 0;
      for (uint32_t pass = 0; pass < 2 || warm < 0.5; ++pass)
        for (const CommandGraph &graph : graphs) warm += backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
      std::vector<std::vector<double>> microseconds(graphs.size());
      for (uint32_t round = 0; round < rounds; ++round)
        for (size_t index = 0; index < graphs.size(); ++index)
          microseconds[index].push_back(backend.submitCommandAsync(graphs[index].dispatches()).wait().gpuSeconds *
                                        1e6 / layers);
      for (size_t index = 0; index < graphs.size(); ++index) {
        std::vector<double> &times = microseconds[index];
        std::sort(times.begin(), times.end());
        std::printf("lanes %u splits %2u: median %8.1f us a layer, least %8.1f\n", lanes, splitCounts[index],
                    times[times.size() / 2], times.front());
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft-attention-sweep: " << error.what() << '\n';
    return 1;
  }
}
