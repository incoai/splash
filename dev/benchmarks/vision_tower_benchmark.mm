// GPU time of the vision tower's encode (ops::Vision) at five image sizes,
// on weights of zeros, for a Mac that holds no model:
//   vision-tower-benchmark <metallib> [encodes]
// Each image of 6x8, 8x8, 16x24, 32x32 and 64x64 patches is encoded once to
// warm up, then `encodes` (9) times; printed is the median GPU time in
// milliseconds. The arithmetic does not depend on the values, so the times
// are those of a model's tower.
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Vision.hpp"

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
using splash::ops::ImageGrid;
using splash::ops::Vision;
using splash::ops::VisionLayout;
using splash::ops::VisionWeights;

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 2 || argc > 3) {
        std::cerr << "usage: vision-tower-benchmark <metallib> [encodes]\n";
        return 2;
      }
      const uint32_t encodes = argc == 3 ? static_cast<uint32_t>(std::stoul(argv[2])) : 9;
      MetalBackend backend(argv[1]);
      const auto zeros = [&](uint64_t elements) {
        return backend.allocateBuffer(elements * 2, BufferStorage::Private, "vision benchmark weights");
      };
      VisionLayout layout;
      layout.outputHiddenSize = 2048;
      const uint64_t hidden = layout.hiddenSize, merged = layout.mergedHiddenSize;
      const auto affine = [&](uint64_t outputs, uint64_t inputs) {
        return decltype(VisionWeights{}.patchEmbedding){zeros(outputs * inputs), zeros(outputs)};
      };
      const auto norm = [&] { return decltype(VisionWeights{}.mergerNorm){zeros(hidden), zeros(hidden)}; };
      VisionWeights weights;
      weights.layout = layout;
      weights.patchEmbedding = affine(hidden, layout.patchDimension);
      weights.positionTable = zeros(uint64_t{layout.positionGridSide} * layout.positionGridSide * hidden);
      for (uint32_t block = 0; block < layout.depth; ++block)
        weights.blocks.push_back({norm(), affine(3 * hidden, hidden), affine(hidden, hidden), norm(),
                                  affine(layout.paddedIntermediateSize, hidden),
                                  affine(hidden, layout.paddedIntermediateSize)});
      weights.mergerNorm = norm();
      weights.mergerUpProjection = affine(merged, merged);
      weights.mergerDownProjection = affine(layout.outputHiddenSize, merged);
      const ImageGrid grids[] = {{6, 8}, {8, 8}, {16, 24}, {32, 32}, {64, 64}};
      const Vision encoder(backend, weights, 64 * 64);
      for (const ImageGrid &grid : grids) {
        const MetalBuffer pixels = backend.allocateBuffer(grid.pixelBytes(layout.patchSize), BufferStorage::Shared, "pixels");
        // Whole 32-row tiles of embeddings, as any version of the tower takes.
        const MetalBuffer embeddings =
            zeros((uint64_t{grid.mergedTokens()} + 31) / 32 * 32 * layout.outputHiddenSize);
        const auto encode = [&] {
          CommandGraph graph;
          encoder.encode(graph, grid, pixels, embeddings);
          return backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds * 1e3;
        };
        static_cast<void>(encode());
        std::vector<double> times;
        for (uint32_t run = 0; run < encodes; ++run) times.push_back(encode());
        std::sort(times.begin(), times.end());
        std::printf("tower %ux%u: median %.3f ms, least %.3f\n", grid.height, grid.width, times[times.size() / 2],
                    times.front());
      }
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "vision-tower-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}
