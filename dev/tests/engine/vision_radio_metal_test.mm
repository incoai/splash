// The kernels Nemotron 3.5 Bolt's RADIO tower runs (kernels/shared/vision.metal)
// against fp64 references, each dispatched as ops/Vision.cpp dispatches it:
// patchify (CLIP mean and std, block-major tokens); positions, HF's bilinear
// resample of the 128x128 table to the square of the grid's longer side and
// its crop, on grids that shrink, crop either side and stretch the table past
// its 128 cells; the prefix rows written past a patch embedding's last tile,
// at patch counts whose prefix crosses a 64-row tile; q/k/v preparation and
// the gather for heads of 80, past a 128-key tile; the merger's RMS norm; and
// the GEMMs' erf GELU and ReLU-squared epilogues. Buffers past what a kernel
// writes hold a sentinel it must leave.
//
//   vision-radio METALLIB
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Vision.h"
#include "ops/Vision.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::ImageGrid;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using splash::test::require;

constexpr splash::ops::VisionLayout kRadio = splash::ops::kRadioVisionLayout;
constexpr uint32_t kHidden = kRadio.hiddenSize;
constexpr uint32_t kHeadDimension = kRadio.headDimension;
constexpr uint32_t kHeads = kRadio.heads;
constexpr uint32_t kSide = kRadio.positionGridSide;
constexpr uint32_t kPrefix = kRadio.prefixTokens;
constexpr double kMean[3] = {0.48145466, 0.4578275, 0.40821073};
constexpr double kStd[3] = {0.26862954, 0.26130258, 0.27577711};
constexpr float kSentinel = 7.0F;

uint32_t roundUp(uint32_t value, uint32_t multiple) { return (value + multiple - 1) / multiple * multiple; }

template <class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = splash::test::sharedBuffer(backend, values.size() * sizeof(T));
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}

// The bf16 bits of values that are bf16 already (each rounded by bf16()).
std::vector<uint16_t> bf16Bits(const std::vector<float> &values) {
  std::vector<uint16_t> bits(values.size());
  std::transform(values.begin(), values.end(), bits.begin(), floatToBf16);
  return bits;
}

float bf16(float value) { return bf16ToFloat(floatToBf16(value)); }

template <class T> const T *read(const MetalBuffer &buffer) { return static_cast<const T *>(buffer.contents()); }

void run(MetalBackend &backend, CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

// The block-major (row, column) of a patch token (vision_patch_position).
std::pair<uint32_t, uint32_t> patchPosition(uint32_t token, uint32_t gridWidth) {
  const uint32_t blocks = gridWidth / 2, block = token / 4, inner = token % 4;
  return {(block / blocks) * 2 + inner / 2, (block % blocks) * 2 + inner % 2};
}

// A value against its fp64 reference: within `relative` of its magnitude
// plus `absolute`.
void requireNear(double actual, double expected, double relative, double absolute, const std::string &what) {
  if (!(std::fabs(actual - expected) <= relative * std::fabs(expected) + absolute))
    throw std::runtime_error(what + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}

void checkPatchify(MetalBackend &backend, ImageGrid grid) {
  const uint32_t width = grid.width * 16;
  std::mt19937 random(grid.height * 31 + grid.width);
  std::vector<uint8_t> pixels(grid.pixelBytes());
  for (uint8_t &pixel : pixels) pixel = static_cast<uint8_t>(random());
  const auto patches = static_cast<uint32_t>(grid.patches());
  std::vector<uint16_t> init(uint64_t{patches + 1} * kRadio.patchDimension, floatToBf16(kSentinel));
  const MetalBuffer image = upload(backend, pixels), output = upload(backend, init);
  CommandGraph graph;
  graph.add("vision_patchify_radio", {image, output}, VisionGridParams{grid.height, grid.width}, {patches, 1, 1});
  run(backend, graph);
  const uint16_t *rows = read<uint16_t>(output);
  for (uint32_t token = 0; token < patches; ++token) {
    const auto [row, column] = patchPosition(token, grid.width);
    for (uint32_t feature = 0; feature < kRadio.patchDimension; ++feature) {
      const uint32_t channel = feature / 256, y = feature % 256 / 16, x = feature % 16;
      const uint8_t pixel = pixels[((uint64_t{row} * 16 + y) * width + column * 16 + x) * 3 + channel];
      // bf16 rounding, and float's of the scale, mean and division.
      requireNear(bf16ToFloat(rows[uint64_t{token} * kRadio.patchDimension + feature]),
                  (pixel / 255.0 - kMean[channel]) / kStd[channel], 0x1p-8, 1e-6,
                  "patchify token " + std::to_string(token) + " feature " + std::to_string(feature));
    }
  }
  for (uint32_t feature = 0; feature < kRadio.patchDimension; ++feature)
    require(rows[uint64_t{patches} * kRadio.patchDimension + feature] == floatToBf16(kSentinel),
            "patchify wrote past the patches");
  std::printf("patchify %ux%u: passed\n", grid.height, grid.width);
}

// torch's bilinear source taps (align_corners=False) of one output index.
void taps(uint32_t index, uint32_t size, uint32_t *tap, double *weight) {
  const double source = std::max((index + 0.5) * (double(kSide) / size) - 0.5, 0.0);
  tap[0] = std::min(static_cast<uint32_t>(source), kSide - 1);
  tap[1] = std::min(tap[0] + 1, kSide - 1);
  weight[1] = source - tap[0];
  weight[0] = 1.0 - weight[1];
}

void checkPositions(MetalBackend &backend, const MetalBuffer &table, const std::vector<float> &values,
                    ImageGrid grid) {
  const auto patches = static_cast<uint32_t>(grid.patches());
  const std::vector<float> init(uint64_t{patches + 1} * kHidden, kSentinel);
  const MetalBuffer hidden = upload(backend, init);
  CommandGraph graph;
  graph.add("vision_prepare_positions_radio", {table, hidden}, VisionGridParams{grid.height, grid.width},
            {patches, 1, 1});
  run(backend, graph);
  const float *rows = read<float>(hidden);
  const uint32_t side = std::max(grid.height, grid.width);
  for (uint32_t token = 0; token < patches; ++token) {
    const auto [row, column] = patchPosition(token, grid.width);
    uint32_t rowTaps[2], columnTaps[2];
    double rowWeights[2], columnWeights[2];
    taps(row, side, rowTaps, rowWeights);
    taps(column, side, columnTaps, columnWeights);
    for (uint32_t dim = 0; dim < kHidden; ++dim) {
      double expected = 0.0;
      for (int h = 0; h < 2; ++h)
        for (int w = 0; w < 2; ++w)
          expected += rowWeights[h] * columnWeights[w] *
                      values[(uint64_t{rowTaps[h]} * kSide + columnTaps[w]) * kHidden + dim];
      // The fp32 source coordinate, below 128, moves the weights by a few
      // units of 2^-23 of it, times neighbors up to 2 apart (torch computes
      // it in fp32 too); a wrong tap or weight is off by tenths.
      requireNear(rows[uint64_t{token} * kHidden + dim], expected, 0.0, kSide * 0x1p-21,
                  "positions " + std::to_string(grid.height) + "x" + std::to_string(grid.width) + " token " +
                      std::to_string(token) + " dim " + std::to_string(dim));
    }
  }
  require(std::all_of(rows + uint64_t{patches} * kHidden, rows + uint64_t{patches + 1} * kHidden,
                      [](float value) { return value == kSentinel; }),
          "positions wrote past the patches");
  std::printf("positions %ux%u (side %u): passed\n", grid.height, grid.width, side);
}

void checkPrefix(MetalBackend &backend, ImageGrid grid) {
  const auto patches = static_cast<uint32_t>(grid.patches());
  const uint32_t end = roundUp(patches + kPrefix, 64);
  std::vector<float> prefix(uint64_t{kPrefix} * kHidden);
  for (uint64_t index = 0; index < prefix.size(); ++index) prefix[index] = bf16(float(index % 977) * 0.01F - 4.0F);
  const std::vector<float> init(uint64_t{end + 64} * kHidden, kSentinel);
  const MetalBuffer tokens = upload(backend, bf16Bits(prefix)), hidden = upload(backend, init);
  CommandGraph graph;
  graph.add("vision_prefix_radio", {tokens, hidden}, VisionGridParams{grid.height, grid.width},
            {end - patches, 1, 1});
  run(backend, graph);
  const float *rows = read<float>(hidden);
  for (uint32_t row = 0; row < end + 64; ++row) {
    for (uint32_t dim = 0; dim < kHidden; ++dim) {
      const float expected = row < patches || row >= end      ? kSentinel
                             : row < patches + kPrefix ? prefix[uint64_t{row - patches} * kHidden + dim]
                                                       : 0.0F;
      require(rows[uint64_t{row} * kHidden + dim] == expected,
              "prefix of " + std::to_string(patches) + " patches: row " + std::to_string(row));
    }
  }
  std::printf("prefix after %u patches (rows to %u): passed\n", patches, end);
}

// q/k/v preparation of `tokens` tokens and the gather of an attention output
// of the same layout back to rows.
void checkQkvAndGather(MetalBackend &backend, uint32_t tokens) {
  const uint32_t padded = roundUp(tokens, 128), tiles = padded / 128;
  std::mt19937 random(tokens);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<uint16_t> qkv(uint64_t{tokens} * 3 * kHidden);
  for (uint16_t &value : qkv) value = floatToBf16(normal(random));
  const uint64_t plane = uint64_t{kHeads} * padded * kHeadDimension;
  const std::vector<uint16_t> sentinel(plane + kHeadDimension, floatToBf16(kSentinel));
  const MetalBuffer rows = upload(backend, qkv);
  const MetalBuffer queries = upload(backend, sentinel), keys = upload(backend, sentinel),
                    values = upload(backend, sentinel);
  const VisionQkvParams params{tokens, padded};
  CommandGraph graph;
  graph.add("vision_qkv_prepare_radio", {rows, queries, keys, values}, params, {padded, 1, 1});
  run(backend, graph);
  for (uint32_t head = 0; head < kHeads; ++head) {
    for (uint32_t token = 0; token < padded; ++token) {
      for (uint32_t dim = 0; dim < kHeadDimension; ++dim) {
        const uint64_t qk = (uint64_t{head} * padded + token) * kHeadDimension + dim;
        const uint64_t v = ((uint64_t{head} * tiles + token / 128) * 128 + token % 128) * kHeadDimension + dim;
        const uint64_t source = uint64_t{token} * 3 * kHidden + head * kHeadDimension + dim;
        const bool valid = token < tokens;
        require(read<uint16_t>(queries)[qk] == (valid ? qkv[source] : 0) &&
                    read<uint16_t>(keys)[qk] == (valid ? qkv[source + kHidden] : 0) &&
                    read<uint16_t>(values)[v] == (valid ? qkv[source + 2 * kHidden] : 0),
                "q/k/v of " + std::to_string(tokens) + " tokens: head " + std::to_string(head) + " token " +
                    std::to_string(token));
      }
    }
  }
  for (const MetalBuffer *buffer : {&queries, &keys, &values})
    require(read<uint16_t>(*buffer)[plane] == floatToBf16(kSentinel), "q/k/v preparation wrote past a plane");

  std::vector<uint16_t> attention(plane);
  for (uint16_t &value : attention) value = floatToBf16(normal(random));
  const MetalBuffer padded_output = upload(backend, attention);
  const MetalBuffer context = upload(backend, std::vector<uint16_t>(uint64_t{tokens + 1} * kHidden,
                                                                     floatToBf16(kSentinel)));
  CommandGraph gather;
  gather.add("vision_attention_gather_d80", {padded_output, context}, params, {tokens, 1, 1});
  run(backend, gather);
  for (uint32_t token = 0; token <= tokens; ++token)
    for (uint32_t index = 0; index < kHidden; ++index)
      require(read<uint16_t>(context)[uint64_t{token} * kHidden + index] ==
                  (token < tokens ? attention[(uint64_t{index / kHeadDimension} * padded + token) * kHeadDimension +
                                              index % kHeadDimension]
                                  : floatToBf16(kSentinel)),
              "gather of " + std::to_string(tokens) + " tokens: token " + std::to_string(token));
  std::printf("q/k/v preparation and gather of %u tokens: passed\n", tokens);
}

void checkRmsNorm(MetalBackend &backend) {
  constexpr uint32_t rows = 3, width = kRadio.mergedHiddenSize;
  std::mt19937 random(5);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> input(rows * width), weight(width);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t i = 0; i < width; ++i) input[row * width + i] = normal(random) * std::ldexp(1.0F, 4 * row - 4);
  for (float &value : weight) value = bf16(1.0F + 0.25F * normal(random));
  const MetalBuffer in = upload(backend, input), w = upload(backend, bf16Bits(weight));
  const MetalBuffer out = upload(backend, std::vector<uint16_t>((rows + 1) * width, floatToBf16(kSentinel)));
  CommandGraph graph;
  graph.add("vision_rms_norm", {in, w, out}, VisionNormParams{width}, {rows, 1, 1});
  run(backend, graph);
  for (uint32_t row = 0; row < rows; ++row) {
    double squares = 0.0;
    for (uint32_t i = 0; i < width; ++i) squares += double(input[row * width + i]) * input[row * width + i];
    const double inverse = 1.0 / std::sqrt(squares / width + SPLASH_VISION_MERGER_RMS_EPSILON);
    for (uint32_t i = 0; i < width; ++i)
      requireNear(bf16ToFloat(read<uint16_t>(out)[row * width + i]), input[row * width + i] * inverse * weight[i],
                  0x1p-8 + 1e-5, 1e-6, "rms norm row " + std::to_string(row) + " value " + std::to_string(i));
  }
  for (uint32_t i = 0; i < width; ++i)
    require(read<uint16_t>(out)[rows * width + i] == floatToBf16(kSentinel), "rms norm wrote past its rows");
  std::printf("merger rms norm: passed\n");
}

// One GEMM of `rows` rows through its epilogue, whose tiles cover whole
// tileRows-row tiles of input and output.
void checkGemm(MetalBackend &backend, const char *pipeline, uint32_t rows, uint32_t output, uint32_t input,
               uint32_t tileRows, uint32_t tileColumns, bool relu) {
  const uint32_t allocated = roundUp(rows, tileRows);
  std::mt19937 random(rows + output);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<uint16_t> a(uint64_t{allocated} * input), w(uint64_t{output} * input), bias(output);
  for (uint16_t &value : a) value = floatToBf16(normal(random));
  for (uint16_t &value : w) value = floatToBf16(normal(random) * 0.1F);
  for (uint16_t &value : bias) value = floatToBf16(normal(random) * 0.5F);
  const MetalBuffer in = upload(backend, a), weights = upload(backend, w), b = upload(backend, bias);
  const MetalBuffer out = upload(backend, std::vector<uint16_t>(uint64_t{allocated} * output, 0));
  CommandGraph graph;
  graph.add(pipeline, {in, weights, b, out, out}, VisionGemmParams{output, input},
            {allocated / tileRows, output / tileColumns, 1});
  run(backend, graph);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < output; ++column) {
      double x = bf16ToFloat(bias[column]), magnitude = std::fabs(x);
      for (uint32_t k = 0; k < input; ++k) {
        const double product =
            double(bf16ToFloat(a[uint64_t{row} * input + k])) * bf16ToFloat(w[uint64_t{column} * input + k]);
        x += product;
        magnitude += std::fabs(product);
      }
      // An fp32 accumulation's error moves the activation by its slope.
      const double accumulation = input * 0x1p-23 * magnitude;
      double expected, slope;
      if (relu) {
        expected = x > 0 ? x * x : 0.0;
        slope = 2.0 * std::fabs(x) + accumulation;
      } else {
        expected = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        slope = 1.2;
      }
      requireNear(bf16ToFloat(read<uint16_t>(out)[uint64_t{row} * output + column]), expected, 0x1p-8,
                  slope * accumulation + 1e-6,
                  std::string(pipeline) + " row " + std::to_string(row) + " column " + std::to_string(column));
    }
  }
  std::printf("%s (%u rows): passed\n", pipeline, rows);
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "usage: vision-radio METALLIB");
      MetalBackend backend(argv[1]);
      for (const ImageGrid grid : {ImageGrid{6, 4}, ImageGrid{2, 2}})
        checkPatchify(backend, grid);

      std::mt19937 random(11);
      std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
      std::vector<float> table(uint64_t{kSide} * kSide * kHidden);
      for (float &value : table) value = bf16(uniform(random));
      const MetalBuffer positions = upload(backend, bf16Bits(table));
      // Shrunk and cropped to rows, to columns, square, and stretched past 128.
      for (const ImageGrid grid : {ImageGrid{6, 4}, ImageGrid{4, 6}, ImageGrid{32, 32}, ImageGrid{4, 132},
                                   ImageGrid{130, 2}})
        checkPositions(backend, positions, table, grid);

      // 56 + 10 crosses a 64-row tile, 120 + 10 a 128-key tile.
      for (const ImageGrid grid : {ImageGrid{2, 2}, ImageGrid{4, 14}, ImageGrid{10, 12}, ImageGrid{8, 8}})
        checkPrefix(backend, grid);
      for (const uint32_t tokens : {14U, 66U, 130U})
        checkQkvAndGather(backend, tokens);
      checkRmsNorm(backend);
      checkGemm(backend, "vision_gemm_m64n128_gelu_erf", 70, 256, 256, 64, 128, false);
      checkGemm(backend, "vision_gemm_m32n256_relu2", 40, 512, 256, 32, 256, true);
      std::cout << "vision_radio_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "vision_radio_metal_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
