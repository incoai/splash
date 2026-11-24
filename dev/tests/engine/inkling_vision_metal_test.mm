// Inkling 2's vision kernels (kernels/shared/inkling_vision.metal) against
// CPU references in double: patchify, the position and class rows with their
// rotary tables, the q/k/v preparation and the merger's rows. Every output is
// allocated NaN past what the kernel writes and checked to hold it. Then the
// encoder's checks of what it reads (ops/InklingVision.cpp): every weight at
// its extent and refused one element short, and an image's pixels and
// embedding rows. The shared dense operators its encoder runs, the scaled
// residual, the LayerNorm into fp32 rows in place and the attention of heads
// of 64 among them, are dense_linear_metal_test's and
// dense_attention_metal_test's.
//
//   inkling-vision-metal METALLIB
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Vision.h"
#include "ops/InklingVision.hpp"

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

using namespace splash;
using metal::CommandGraph;
using metal::MetalBackend;
using metal::MetalBuffer;
using test::bf16ToFloat;
using test::floatToBf16;
using test::require;

constexpr ops::VisionLayout kLayout = ops::visionLayout(ops::VisionTower::Inkling, 2304);
constexpr uint32_t kHidden = kLayout.hiddenSize;
constexpr uint32_t kHeadDim = kLayout.headDimension;
constexpr uint32_t kPatch = kLayout.patchSize;
constexpr uint32_t kPatchDim = kLayout.paddedPatchDimension;
constexpr uint32_t kSide = kLayout.positionGridSide;
constexpr uint16_t kBf16Nan = 0x7FC0;

uint32_t roundUp(uint32_t value, uint32_t multiple) { return (value + multiple - 1) / multiple * multiple; }

MetalBuffer bf16Buffer(MetalBackend &backend, uint64_t elements, uint16_t fill = kBf16Nan) {
  MetalBuffer buffer = test::sharedBuffer(backend, elements * 2);
  std::fill_n(static_cast<uint16_t *>(buffer.contents()), elements, fill);
  return buffer;
}

MetalBuffer floatBuffer(MetalBackend &backend, uint64_t elements, float fill = NAN) {
  MetalBuffer buffer = test::sharedBuffer(backend, elements * 4);
  std::fill_n(static_cast<float *>(buffer.contents()), elements, fill);
  return buffer;
}

uint16_t *words(const MetalBuffer &buffer) { return static_cast<uint16_t *>(buffer.contents()); }
float *floats(const MetalBuffer &buffer) { return static_cast<float *>(buffer.contents()); }

void run(MetalBackend &backend, CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

// Random bf16 values of a normal distribution scaled by `scale`.
void randomize(const MetalBuffer &buffer, uint64_t elements, std::mt19937 &random, float scale) {
  std::normal_distribution<float> normal(0.0F, scale);
  for (uint64_t index = 0; index < elements; ++index) words(buffer)[index] = floatToBf16(normal(random));
}

// Block-major (row, column) of a patch, as the kernels order them.
std::pair<uint32_t, uint32_t> position(uint32_t token, uint32_t gridWidth) {
  const uint32_t block = token / 4, inner = token % 4, blocks = gridWidth / 2;
  return {block / blocks * 2 + inner / 2, block % blocks * 2 + inner % 2};
}

// Patchify: each patch's 1200 values [channel, row, column] x / 127.5 - 1,
// zero to 1280, a row per patch; a pixel past the image's content its
// padding, -1 / 127.5 - 1, whatever the pixels hold there.
void checkPatchify(MetalBackend &backend, ops::ImageGrid grid) {
  const uint32_t patches = grid.height * grid.width, rows = patches;
  const uint32_t width = grid.width * kPatch;
  const uint32_t contentHeight = grid.contentHeight ? grid.contentHeight : grid.height * kPatch;
  const uint32_t contentWidth = grid.contentWidth ? grid.contentWidth : width;
  const MetalBuffer pixels = test::sharedBuffer(backend, grid.pixelBytes(kPatch));
  std::mt19937 random(patches);
  for (uint64_t index = 0; index < pixels.sizeBytes(); ++index)
    static_cast<uint8_t *>(pixels.contents())[index] = static_cast<uint8_t>(random());
  const MetalBuffer patchRows = bf16Buffer(backend, uint64_t{rows + 1} * kPatchDim);
  CommandGraph graph;
  graph.add("inkling_vision_patchify", {pixels, patchRows},
            InklingPatchifyParams{grid.height, grid.width, contentHeight, contentWidth}, {rows, 1, 1});
  run(backend, graph);
  const auto *pixel = static_cast<const uint8_t *>(pixels.contents());
  for (uint32_t token = 0; token < rows; ++token) {
    const auto [row, column] = position(token, grid.width);
    for (uint32_t feature = 0; feature < kPatchDim; ++feature) {
      float expected = 0.0F;
      if (feature < 3 * kPatch * kPatch) {
        const uint32_t channel = feature / (kPatch * kPatch), within = feature % (kPatch * kPatch);
        const uint32_t y = row * kPatch + within / kPatch, x = column * kPatch + within % kPatch;
        const uint64_t at = (uint64_t{y} * width + x) * 3 + channel;
        expected = (y < contentHeight && x < contentWidth ? float(pixel[at]) : -1.0F) / 127.5F - 1.0F;
      }
      const float actual = bf16ToFloat(words(patchRows)[uint64_t{token} * kPatchDim + feature]);
      require(std::fabs(actual - expected) <= 0x1p-8F * std::fabs(expected),
              "patchify token " + std::to_string(token) + " feature " + std::to_string(feature) + ": " +
                  std::to_string(actual) + " vs " + std::to_string(expected));
    }
  }
  require(words(patchRows)[uint64_t{rows} * kPatchDim] == kBf16Nan, "patchify wrote past its rows");
  std::printf("inkling_vision_patchify %ux%u, content %ux%u: passed\n", grid.height, grid.width, contentHeight,
              contentWidth);
}

// The positions resampled from the 32x32 table (half-pixel bilinear, edges
// clamped), the class row after the patches, and the rotary tables: the
// first 32 dimensions by the 1-based column, the last 32 by the row, the
// class token's at (0, 0).
void checkPositions(MetalBackend &backend, ops::ImageGrid grid) {
  const uint32_t patches = grid.height * grid.width, rows = patches + 1;
  std::mt19937 random(patches * 3);
  const MetalBuffer table = bf16Buffer(backend, uint64_t{kSide} * kSide * kHidden);
  randomize(table, uint64_t{kSide} * kSide * kHidden, random, 0.5F);
  const MetalBuffer classToken = bf16Buffer(backend, 2 * kHidden);
  randomize(classToken, 2 * kHidden, random, 1.0F);
  const MetalBuffer hidden = floatBuffer(backend, uint64_t{rows + 1} * kHidden);
  const MetalBuffer cosines = floatBuffer(backend, uint64_t{patches + 2} * kHeadDim);
  const MetalBuffer sines = floatBuffer(backend, uint64_t{patches + 2} * kHeadDim);
  CommandGraph graph;
  graph.add("inkling_vision_prepare_positions", {table, classToken, hidden, cosines, sines},
            VisionGridParams{grid.height, grid.width}, {rows, 1, 1});
  run(backend, graph);

  const auto taps = [](uint32_t index, uint32_t size, uint32_t *tap, double *weight) {
    const double source = (index + 0.5) * kSide / size - 0.5, lower = std::floor(source);
    for (uint32_t offset = 0; offset < 2; ++offset) {
      tap[offset] = static_cast<uint32_t>(std::clamp(lower + offset, 0.0, double(kSide - 1)));
      weight[offset] = std::max(1.0 - std::fabs(source - lower - offset), 0.0);
    }
  };
  for (uint32_t token = 0; token < rows; ++token) {
    const float *row = floats(hidden) + uint64_t{token} * kHidden;
    uint32_t rowPosition = 0, columnPosition = 0;
    for (uint32_t dim = 0; dim < kHidden; ++dim) {
      double expected = 0.0, magnitude = 0.0;
      if (token < patches) {
        const auto [r, c] = position(token, grid.width);
        rowPosition = r + 1, columnPosition = c + 1;
        uint32_t rowTaps[2], columnTaps[2];
        double rowWeights[2], columnWeights[2];
        taps(r, grid.height, rowTaps, rowWeights);
        taps(c, grid.width, columnTaps, columnWeights);
        for (uint32_t h = 0; h < 2; ++h)
          for (uint32_t w = 0; w < 2; ++w) {
            const double term = rowWeights[h] * columnWeights[w] *
                                bf16ToFloat(words(table)[(uint64_t{rowTaps[h]} * kSide + columnTaps[w]) * kHidden + dim]);
            expected += term, magnitude += std::fabs(term);
          }
      } else {
        expected = double(bf16ToFloat(words(classToken)[dim])) + bf16ToFloat(words(classToken)[kHidden + dim]);
        magnitude = std::fabs(expected);
      }
      require(std::fabs(row[dim] - expected) <= 2e-5 * magnitude + 1e-6,
              "positions token " + std::to_string(token) + " dim " + std::to_string(dim) + ": " +
                  std::to_string(row[dim]) + " vs " + std::to_string(expected));
    }
    for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
      const uint32_t pair = dim % 32 / 2, axis = dim < 32 ? columnPosition : rowPosition;
      const double angle = axis * std::pow(10000.0, -2.0 * pair / 32.0);
      const uint64_t at = uint64_t{token} * kHeadDim + dim;
      require(std::fabs(floats(cosines)[at] - std::cos(angle)) <= 1e-5 &&
                  std::fabs(floats(sines)[at] - std::sin(angle)) <= 1e-5,
              "rotary table token " + std::to_string(token) + " dim " + std::to_string(dim));
    }
  }
  require(std::isnan(floats(hidden)[uint64_t{rows} * kHidden]), "positions wrote past their rows");
  require(std::isnan(floats(cosines)[uint64_t{patches + 1} * kHeadDim]), "rotary tables written past the class token");
  std::printf("inkling_vision_prepare_positions %ux%u: passed\n", grid.height, grid.width);
}

// Queries and keys rotated by interleaved pairs, values copied, all three
// scattered into (heads, padded, 64) and zero from `tokens` on.
void checkQkvPrepare(MetalBackend &backend, uint32_t tokens) {
  const uint32_t padded = roundUp(tokens, 128), heads = kLayout.heads;
  std::mt19937 random(tokens);
  const MetalBuffer qkv = bf16Buffer(backend, uint64_t{padded} * 3 * kHidden);
  randomize(qkv, uint64_t{padded} * 3 * kHidden, random, 2.0F);
  const MetalBuffer cosines = floatBuffer(backend, uint64_t{tokens} * kHeadDim);
  const MetalBuffer sines = floatBuffer(backend, uint64_t{tokens} * kHeadDim);
  std::uniform_real_distribution<float> angle(-3.2F, 3.2F);
  for (uint64_t index = 0; index < uint64_t{tokens} * kHeadDim; index += 2) {
    const float a = angle(random);
    floats(cosines)[index] = floats(cosines)[index + 1] = std::cos(a);
    floats(sines)[index] = floats(sines)[index + 1] = std::sin(a);
  }
  const uint64_t planes = uint64_t{heads} * padded * kHeadDim;
  const MetalBuffer queries = bf16Buffer(backend, planes + 1), keys = bf16Buffer(backend, planes + 1),
                    values = bf16Buffer(backend, planes + 1);
  CommandGraph graph;
  graph.add("inkling_vision_qkv_prepare", {qkv, cosines, sines, queries, keys, values},
            VisionQkvParams{tokens, padded}, {padded, 1, 1});
  run(backend, graph);
  for (uint32_t token = 0; token < padded; ++token)
    for (uint32_t index = 0; index < kHidden; ++index) {
      const uint32_t head = index / kHeadDim, dim = index % kHeadDim;
      const uint64_t plane = (uint64_t{head} * padded + token) * kHeadDim + dim;
      double expected[3] = {0.0, 0.0, 0.0};
      if (token < tokens) {
        const uint16_t *row = words(qkv) + uint64_t{token} * 3 * kHidden;
        const double c = floats(cosines)[uint64_t{token} * kHeadDim + dim];
        const double s = floats(sines)[uint64_t{token} * kHeadDim + dim];
        const double sign = dim & 1 ? 1.0 : -1.0;
        for (uint32_t part = 0; part < 2; ++part)
          expected[part] = bf16ToFloat(row[part * kHidden + index]) * c +
                           sign * bf16ToFloat(row[part * kHidden + (index ^ 1)]) * s;
        expected[2] = bf16ToFloat(row[2 * kHidden + index]);
      }
      const MetalBuffer *outputs[3] = {&queries, &keys, &values};
      for (uint32_t part = 0; part < 3; ++part) {
        const double actual = bf16ToFloat(words(*outputs[part])[plane]);
        require(std::fabs(actual - expected[part]) <= 0x1p-8 * std::fabs(expected[part]) + 1e-6,
                "qkv prepare token " + std::to_string(token) + " index " + std::to_string(index) + " part " +
                    std::to_string(part));
      }
    }
  require(words(queries)[planes] == kBf16Nan && words(keys)[planes] == kBf16Nan && words(values)[planes] == kBf16Nan,
          "qkv prepare wrote past its planes");
  std::printf("inkling_vision_qkv_prepare %u tokens: passed\n", tokens);
}

// The merger's bf16 rows of the fp32 residual rows, a row per patch.
void checkPatchRows(MetalBackend &backend, uint32_t rows) {
  std::mt19937 random(rows * 7);
  const MetalBuffer hidden = floatBuffer(backend, uint64_t{rows + 1} * kHidden);
  std::normal_distribution<float> normal(0.5F, 4.0F);
  for (uint64_t index = 0; index < uint64_t{rows + 1} * kHidden; ++index) floats(hidden)[index] = normal(random);
  const MetalBuffer patchRows = bf16Buffer(backend, uint64_t{rows + 1} * kHidden);
  CommandGraph graph;
  graph.add("inkling_vision_patch_rows", {hidden, patchRows}, {rows, 1, 1});
  run(backend, graph);
  for (uint64_t at = 0; at < uint64_t{rows} * kHidden; ++at)
    require(words(patchRows)[at] == floatToBf16(floats(hidden)[at]), "merger rows value " + std::to_string(at));
  require(words(patchRows)[uint64_t{rows} * kHidden] == kBf16Nan, "the merger's rows were written past the patches");
  std::printf("inkling_vision_patch_rows %u rows: passed\n", rows);
}

// The encoder's reads: every weight at its extent and one element short,
// the class token's two rows, a tower of one block too few or too many, and
// an image's pixels (20-pixel patches) and embedding rows.
void checkEncoderExtents(MetalBackend &backend) {
  using ops::DenseWeights;
  using ops::LayerNormWeights;
  using ops::VisionBlock;
  using ops::VisionWeights;
  constexpr uint64_t kBf16Bytes = 2;
  constexpr ops::ImageGrid kGrid{2, 2};
  const uint64_t hidden = kLayout.hiddenSize, merged = kLayout.mergedHiddenSize;
  const MetalBuffer storage = test::sharedBuffer(backend, merged * merged * kBf16Bytes);
  const auto tensor = [&](uint64_t elements) { return backend.view(storage, 0, elements * kBf16Bytes); };
  const auto affine = [&](uint64_t output, uint64_t input) { return DenseWeights{tensor(output * input), tensor(output)}; };
  const LayerNormWeights norm{tensor(hidden), tensor(hidden)};
  // The output projections with their LayerScales.
  DenseWeights projection = affine(hidden, hidden), down = affine(hidden, kLayout.intermediateSize);
  projection.scale = down.scale = tensor(hidden);
  VisionWeights model;
  model.layout = kLayout;
  model.patchEmbedding = affine(hidden, kLayout.paddedPatchDimension);
  model.positionTable = tensor(uint64_t{kSide} * kSide * hidden);
  model.classToken = tensor(2 * hidden);
  model.inputNorm = norm;
  model.blocks.assign(kLayout.depth, VisionBlock{norm, affine(3 * hidden, hidden), projection, norm,
                                                 affine(kLayout.intermediateSize, hidden), down});
  model.mergerUpProjection = affine(merged, merged);
  model.mergerDownProjection = affine(kLayout.outputHiddenSize, merged);
  const MetalBuffer pixels = test::sharedBuffer(backend, kGrid.pixelBytes(kPatch));
  const MetalBuffer embeddings =
      test::sharedBuffer(backend, uint64_t{kGrid.mergedTokens()} * kLayout.outputHiddenSize * kBf16Bytes);
  const auto encode = [&](CommandGraph &graph, const VisionWeights &weights, const MetalBuffer &image,
                          const MetalBuffer &output) {
    const ops::InklingVision vision(backend, weights, kGrid.patches());
    vision.encode(graph, kGrid, image, output);
  };
  const auto requireWeight = [&](const MetalBuffer &buffer, uint64_t elements, const std::string &name,
                                 const auto &with) {
    test::requireExtent(backend, buffer, elements * kBf16Bytes, kBf16Bytes, name,
                        [&](CommandGraph &graph, const MetalBuffer &view) {
                          VisionWeights changed = model;
                          with(changed, view);
                          encode(graph, changed, pixels, embeddings);
                        });
  };
  const auto requireAffine = [&](DenseWeights VisionWeights::*member, uint64_t output, uint64_t input,
                                 const std::string &name) {
    requireWeight((model.*member).weight, output * input, name + " weight",
                  [&](VisionWeights &changed, const MetalBuffer &view) { (changed.*member).weight = view; });
    requireWeight((model.*member).bias, output, name + " bias",
                  [&](VisionWeights &changed, const MetalBuffer &view) { (changed.*member).bias = view; });
  };
  requireAffine(&VisionWeights::patchEmbedding, hidden, kLayout.paddedPatchDimension, "vision patch embedding");
  requireWeight(model.positionTable, uint64_t{kSide} * kSide * hidden, "vision position table",
                [](VisionWeights &changed, const MetalBuffer &view) { changed.positionTable = view; });
  requireWeight(model.classToken, 2 * hidden, "vision class token",
                [](VisionWeights &changed, const MetalBuffer &view) { changed.classToken = view; });
  for (const auto &[field, part] :
       {std::pair{&LayerNormWeights::weight, " weight"}, std::pair{&LayerNormWeights::bias, " bias"}})
    requireWeight(model.inputNorm.*field, hidden, std::string("vision input norm") + part,
                  [&](VisionWeights &changed, const MetalBuffer &view) { changed.inputNorm.*field = view; });
  for (const auto &[member, name] : {std::pair{&VisionBlock::projection, "vision attention projection scale"},
                                     std::pair{&VisionBlock::downProjection, "vision down projection scale"}})
    requireWeight((model.blocks.back().*member).scale, hidden, name,
                  [&](VisionWeights &changed, const MetalBuffer &view) { (changed.blocks.back().*member).scale = view; });
  for (const auto &[member, output, input, name] :
       {std::tuple{&VisionBlock::qkv, 3 * hidden, hidden, "vision q/k/v"},
        std::tuple{&VisionBlock::projection, hidden, hidden, "vision attention projection"},
        std::tuple{&VisionBlock::upProjection, uint64_t{kLayout.intermediateSize}, hidden, "vision up projection"},
        std::tuple{&VisionBlock::downProjection, hidden, uint64_t{kLayout.intermediateSize}, "vision down projection"}})
    for (const auto &[field, part, elements] :
         {std::tuple{&DenseWeights::weight, " weight", output * input}, std::tuple{&DenseWeights::bias, " bias", output}})
      requireWeight(model.blocks.back().*member.*field, elements, std::string(name) + part,
                    [&](VisionWeights &changed, const MetalBuffer &view) { changed.blocks.back().*member.*field = view; });
  requireAffine(&VisionWeights::mergerUpProjection, merged, merged, "vision merger up projection");
  requireAffine(&VisionWeights::mergerDownProjection, kLayout.outputHiddenSize, merged, "vision merger down projection");
  for (const uint32_t depth : {kLayout.depth - 1, kLayout.depth + 1}) {
    VisionWeights changed = model;
    changed.blocks.resize(depth, model.blocks.back());
    test::rejects([&] { const ops::InklingVision vision(backend, changed, kGrid.patches()); },
                  "vision tower holds " + std::to_string(depth) + " blocks, needs " + std::to_string(kLayout.depth),
                  "a tower of " + std::to_string(depth) + " blocks was accepted");
  }
  test::rejects([&] { static_cast<void>(ops::InklingVision::scratchBytes(ops::VisionLayout{.outputHiddenSize = 2304}, 4)); },
                "Inkling vision kernels are specialized", "a Qwen layout was accepted");
  test::requireExtent(backend, pixels, kGrid.pixelBytes(kPatch), 1, "image pixel",
                      [&](CommandGraph &graph, const MetalBuffer &view) { encode(graph, model, view, embeddings); });
  test::requireExtent(backend, embeddings, embeddings.sizeBytes(), kBf16Bytes, "image embedding",
                      [&](CommandGraph &graph, const MetalBuffer &view) { encode(graph, model, pixels, view); });
  std::printf("InklingVision extents: passed\n");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "usage: inkling-vision-metal METALLIB");
      MetalBackend backend(argv[1]);
      // Whole grids, and images padded right, bottom or both.
      for (const ops::ImageGrid grid : {ops::ImageGrid{2, 2}, ops::ImageGrid{4, 6}, ops::ImageGrid{10, 2},
                                        ops::ImageGrid{2, 2, 33, 37}, ops::ImageGrid{4, 6, 61, 120},
                                        ops::ImageGrid{10, 2, 200, 21}, ops::ImageGrid{2, 2, 1, 1}})
        checkPatchify(backend, grid);
      // Smaller than the table (downsampled), larger (upsampled), its size
      // and thin.
      for (const ops::ImageGrid grid :
           {ops::ImageGrid{6, 10}, ops::ImageGrid{36, 40}, ops::ImageGrid{32, 32}, ops::ImageGrid{2, 64}})
        checkPositions(backend, grid);
      for (const uint32_t tokens : {5U, 127U, 129U, 261U}) checkQkvPrepare(backend, tokens);
      for (const uint32_t rows : {4U, 36U}) checkPatchRows(backend, rows);
      checkEncoderExtents(backend);
      std::cout << "inkling_vision_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "inkling_vision_metal_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
