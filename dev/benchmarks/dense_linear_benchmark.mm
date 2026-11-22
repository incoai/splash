// GPU time of every vision tower projection on each tile of ops::DenseLinear
// (kernels/shared/dense_linear.metal), the measurements behind the dense tile
// rule of runtime/ops/KernelCalibration.hpp:
//   dense-linear-benchmark <metallib> [rounds]
// The tower's projections (patch embedding, qkv, output, up, down, the
// merger's two) at the rows the images of 6x8, 8x8, 16x24, 32x32 and 64x64
// patches give them (a quarter for the merger's), each on the three tiles in
// ABBA order after three warm-up rounds. Printed are the medians of `rounds` (9)
// commands, each running the projection 8 times, in microseconds per
// projection, the tile the device's rule picks marked with *.
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/DenseLinear.h"
#include "ops/DenseLinear.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::DenseEpilogue;
using splash::ops::DenseLinear;

struct Tile final {
  const char *name;
  uint32_t rows, columns, simdgroups;
};
constexpr Tile kTiles[] = {{"dense_linear_m64n64s8", 64, 64, 8},
                           {"dense_linear_m64n64s4", 64, 64, 4},
                           {"dense_linear_m64n128s4", 64, 128, 4}};

struct Projection final {
  const char *name;
  uint32_t outputs, inputs;
  DenseEpilogue epilogue;
  bool merger;
};
constexpr Projection kProjections[] = {{"patch", 1152, 1536, DenseEpilogue::Residual, false},
                                       {"qkv", 3456, 1152, DenseEpilogue::None, false},
                                       {"output", 1152, 1152, DenseEpilogue::Residual, false},
                                       {"up", 4352, 1152, DenseEpilogue::GeluTanh, false},
                                       {"down", 1152, 4352, DenseEpilogue::Residual, false},
                                       {"merger up", 4608, 4608, DenseEpilogue::GeluErf, true},
                                       {"merger 2048", 2048, 4608, DenseEpilogue::None, true},
                                       {"merger 5120", 5120, 4608, DenseEpilogue::None, true}};
constexpr uint32_t kPatches[] = {48, 64, 384, 1024, 4096};
constexpr uint32_t kRepeats = 8;

const char *suffix(DenseEpilogue epilogue) {
  switch (epilogue) {
  case DenseEpilogue::None: return "";
  case DenseEpilogue::GeluTanh: return "_gelu_tanh";
  case DenseEpilogue::GeluErf: return "_gelu_erf";
  case DenseEpilogue::Residual: return "_residual";
  case DenseEpilogue::ScaledResidual: return "_scaled_residual";
  }
  return "";
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 2 || argc > 3) {
        std::cerr << "usage: dense-linear-benchmark <metallib> [rounds]\n";
        return 2;
      }
      const uint32_t rounds = argc == 3 ? static_cast<uint32_t>(std::stoul(argv[2])) : 9;
      MetalBackend backend(argv[1]);
      const DenseLinear dense(backend.capabilities());
      std::printf("dense-linear-benchmark: %u cores\n", backend.capabilities().plannedGpuCores());
      for (const Projection &p : kProjections) {
        // One set of buffers a projection, for its most rows: a new
        // allocation's first commands run slower.
        const auto allocate = [&](uint64_t bytes) {
          return backend.allocateBuffer(bytes, BufferStorage::Private, "dense benchmark");
        };
        const uint32_t most = p.merger ? 4096 / 4 : 4096;
        const MetalBuffer input = allocate(uint64_t{DenseLinear::paddedRows(most)} * p.inputs * 2);
        const MetalBuffer weight = allocate(uint64_t{p.outputs} * p.inputs * 2);
        const MetalBuffer bias = allocate(uint64_t{p.outputs} * 2);
        const MetalBuffer output =
            allocate(uint64_t{most} * p.outputs * (p.epilogue == DenseEpilogue::Residual ? 4 : 2));
        for (const uint32_t patches : kPatches) {
          const uint32_t rows = p.merger ? patches / 4 : patches;
          // The rule's pick, by the pipeline its dispatch names.
          CommandGraph probe;
          dense.add(probe, {weight, bias}, p.outputs, p.inputs, input, output, rows, p.epilogue);
          const std::string picked = probe.dispatches().front().pipelineName;
          const auto time = [&](const Tile &tile) {
            CommandGraph graph;
            for (uint32_t copy = 0; copy < kRepeats; ++copy)
              graph.add(std::string(tile.name) + suffix(p.epilogue), {input, weight, bias, output},
                        DenseLinearParams{rows, p.outputs, p.inputs, 0},
                        {(rows + tile.rows - 1) / tile.rows, p.outputs / tile.columns, 1},
                        {tile.simdgroups * 32, 1, 1});
            return backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds * 1e6 / kRepeats;
          };
          for (uint32_t warm = 0; warm < 3; ++warm)
            for (const Tile &tile : kTiles) static_cast<void>(time(tile));
          std::vector<std::vector<double>> samples(std::size(kTiles));
          for (uint32_t round = 0; round < rounds; ++round)
            for (size_t i = 0; i < std::size(kTiles); ++i) {
              const size_t tile = round % 2 ? std::size(kTiles) - 1 - i : i;
              samples[tile].push_back(time(kTiles[tile]));
            }
          std::printf("%-11s rows %5u (%ux%u):", p.name, rows, p.outputs, p.inputs);
          for (size_t i = 0; i < std::size(kTiles); ++i) {
            std::sort(samples[i].begin(), samples[i].end());
            const bool chosen = picked == std::string(kTiles[i].name) + suffix(p.epilogue);
            std::printf("  %s %9.1f%s", kTiles[i].name + 13, samples[i][samples[i].size() / 2], chosen ? "*" : " ");
          }
          std::printf("\n");
        }
      }
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "dense-linear-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}
