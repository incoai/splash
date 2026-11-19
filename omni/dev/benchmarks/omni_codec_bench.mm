// Times the speech tokenizer decoder's convolutions (kernels/omni/codec.metal,
// omni_codec_conv_half) per decoder stage, at the rows a chunk of FRAMES
// frames gives one stream (or several streams' chunks, --streams):
//   omni-codec-bench METALLIB [--frames F] [--streams S]
// Each case is a chain of 8 dispatches in one command, best of 6 after 2
// warm-up commands; the figure is TFLOPS of the convolution's products.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/OmniCodec.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace splash;

namespace {

struct Stage {
  const char *name;
  uint32_t rowsPerFrame, in, out, taps, dilation;
};

struct Tile {
  const char *pipeline;
  uint32_t rows, columns, depth, threads;
};

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: omni-codec-bench METALLIB [--frames F] [--streams S]\n");
    return 2;
  }
  uint32_t frames = 25, streams = 1;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    if (flag == "--frames") frames = std::stoul(argv[i + 1]);
    else if (flag == "--streams") streams = std::stoul(argv[i + 1]);
  }
  try {
    metal::MetalBackend backend(argv[1]);
    // The decoder's convolutions (speech_tokenizer decoder_config): rows per
    // frame of each stage's input, channels, taps, dilation; transposed
    // convolutions as their phase-stacked form.
    const std::vector<Stage> stages{
        {"initial conv 1024->1536 k7", 4, 1024, 1536, 7, 1},
        {"block1 transposed 1536->8x768", 4, 1536, 6144, 2, 1},
        {"block1 conv1 768 k7 d3", 32, 768, 768, 7, 3},
        {"block1 conv2 768 k1", 32, 768, 768, 1, 1},
        {"block2 transposed 768->5x384", 32, 768, 1920, 2, 1},
        {"block2 conv1 384 k7 d3", 160, 384, 384, 7, 3},
        {"block2 conv2 384 k1", 160, 384, 384, 1, 1},
        {"block3 transposed 384->4x192", 160, 384, 768, 2, 1},
        {"block3 conv1 192 k7 d3", 640, 192, 192, 7, 3},
        {"block3 conv2 192 k1", 640, 192, 192, 1, 1},
        {"block4 transposed 192->3x96", 640, 192, 288, 2, 1},
        {"block4 conv1 96 k7 d3", 1920, 96, 96, 7, 3},
        {"block4 conv2 96 k1", 1920, 96, 96, 1, 1},
    };
    // The production tile (ops/OmniCodec.cpp); wider ones are gone (round 3
    // of docs/omni-performance.md: never ahead beyond noise).
    const std::vector<Tile> tiles{
        {"omni_codec_conv_half", OMNI_CODEC_CONV_TILE_ROWS, OMNI_CODEC_CONV_TILE_COLUMNS, OMNI_CODEC_CONV_TILE_DEPTH,
         OMNI_CODEC_CONV_THREADS},
    };
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> value(-1.0F, 1.0F);
    std::printf("%u frames x %u streams per dispatch; TFLOPS of each tile\n", frames, streams);
    double total[16] = {0};
    for (const Stage &stage : stages) {
      const uint32_t rows = stage.rowsPerFrame * frames * streams;
      const uint64_t weightCount = uint64_t{stage.taps} * stage.in * stage.out;
      metal::MetalBuffer input = backend.allocateBuffer(uint64_t{rows} * stage.in * 4, metal::BufferStorage::Shared, "in");
      metal::MetalBuffer output = backend.allocateBuffer(uint64_t{rows} * stage.out * 4, metal::BufferStorage::Shared, "out");
      metal::MetalBuffer weight = backend.allocateBuffer(weightCount * 2, metal::BufferStorage::Shared, "w");
      metal::MetalBuffer bias = backend.allocateBuffer(uint64_t{stage.out} * 4, metal::BufferStorage::Shared, "b");
      auto *x = static_cast<float *>(input.contents());
      for (uint64_t i = 0; i < uint64_t{rows} * stage.in; ++i) x[i] = value(rng);
      auto *w = static_cast<_Float16 *>(weight.contents());
      for (uint64_t i = 0; i < weightCount; ++i) w[i] = _Float16(value(rng) * 0.05F);
      std::memset(bias.contents(), 0, uint64_t{stage.out} * 4);
      OmniCodecConvParams p{};
      p.rows = rows;
      p.input_rows = rows;
      p.in_channels = stage.in;
      p.out_channels = stage.out;
      p.taps = stage.taps;
      p.dilation = stage.dilation;
      p.pad = int32_t((stage.taps - 1) * stage.dilation);
      p.input_stride = stage.in;
      p.output_stride = stage.out;
      p.flags = OMNI_CODEC_CONV_FLAG_BIAS;
      const double flops = 2.0 * rows * stage.in * stage.out * stage.taps;
      std::vector<double> results(tiles.size(), 0.0);
      for (size_t t = 0; t < tiles.size(); ++t) {
        const Tile &tile = tiles[t];
        if (stage.in % tile.depth) continue;
        double best = 1e9;
        for (int i = 0; i < 8; ++i) {
          metal::CommandGraph graph;
          for (int j = 0; j < 8; ++j)
            graph.add(tile.pipeline, {input, weight, bias, output, bias, output, input}, p,
                      {(rows + tile.rows - 1) / tile.rows, (stage.out + tile.columns - 1) / tile.columns, 1},
                      {tile.threads, 1, 1});
          const metal::CommandTiming timing = backend.submitCommandAsync(graph.command()).wait();
          if (i >= 2) best = std::min(best, timing.gpuSeconds / 8);
        }
        results[t] = flops / best / 1e12;
        total[t] += best;
      }
      std::printf("%-32s rows %6u:", stage.name, rows);
      for (size_t t = 0; t < tiles.size(); ++t) std::printf(" %5.1f", results[t]);
      std::printf("\n");
    }
    std::printf("%-45s", "per-tile total ms:");
    for (size_t t = 0; t < tiles.size(); ++t) std::printf(" %6.2f", total[t] * 1e3);
    std::printf("\n");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "omni codec bench: %s\n", error.what());
    return 1;
  }
  return 0;
}
