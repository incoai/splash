// Inkling 2's vision kernels (kernels/shared/inkling_vision.metal) against
// CPU references in double: patchify, the position and class rows with their
// rotary tables, the q/k/v preparation and the merger's rows. Every output is
// allocated NaN past what the kernel writes and checked to hold it. Then the
// encoder's checks of what it reads (ops/InklingVision.cpp): every weight at
// its extent and refused one element short, a LayerScale it lacks, and an
// image's pixels and embedding rows. The shared dense operators its encoder
// runs, the scaled residual, the LayerNorm into fp32 rows in place and the
// attention of heads of 64 among them, are dense_linear_metal_test's and
// dense_attention_metal_test's. Last the whole tower of Q8_0 projections
// (an mmproj's, DenseWeights::quantized) against the tower of the same
// values in bf16, which every Q8_0 value of a power-of-two scale is, within
// the bf16 rounding of a sum in another order (relative RMS 1e-2), at 17 and
// 101 tokens; and so the towers of the production image's writer and reader
// (vision::writeVision, readInklingVision) of a Q8_0 mmproj, its q, k and v
// stacked into one projection's planes, and of the same values in BF16.
//
//   inkling-vision-metal METALLIB
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Vision.h"
#include "TestFiles.hpp"
#include "metal/abi/QuantFormat.h"
#include "model/InklingVision.hpp"
#include "model/VisionPreparation.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightSource.hpp"
#include "model/WeightStore.hpp"
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
  // Each output projection's LayerScale, at its extent, and absent: a tower
  // without one is refused when its encoder is built, not at its first image.
  for (const auto &[member, name] : {std::pair{&VisionBlock::projection, "vision attention projection scale"},
                                     std::pair{&VisionBlock::downProjection, "vision down projection scale"}}) {
    requireWeight((model.blocks.back().*member).scale, hidden, name,
                  [&](VisionWeights &changed, const MetalBuffer &view) { (changed.blocks.back().*member).scale = view; });
    VisionWeights without = model;
    (without.blocks.back().*member).scale = {};
    test::rejects([&] { const ops::InklingVision vision(backend, without, kGrid.patches()); }, name,
                  std::string("a tower without its ") + name + " was accepted");
  }
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

// A tower of random weights whose projections are Q8_0 (codes in
// [-127, 127], a scale of 2^-11 or 2^-12 a block, whose products are exact
// in bf16 and fp16) or, `q8` false, the same values in bf16; LayerNorms of
// weight 1 and bias 0, LayerScales of 0.1 to 0.3.
ops::VisionWeights randomTower(MetalBackend &backend, bool q8, uint32_t seed) {
  std::mt19937 random(seed);
  const auto bf16Values = [&](uint64_t count, float scale) {
    MetalBuffer buffer = test::sharedBuffer(backend, count * 2);
    randomize(buffer, count, random, scale);
    return buffer;
  };
  const auto constant = [&](uint64_t count, float value) { return bf16Buffer(backend, count, floatToBf16(value)); };
  const auto projection = [&](uint32_t outputs, uint32_t inputs) {
    const float d = inputs > 1024 ? 0x1p-12F : 0x1p-11F;
    const uint32_t groups = inputs / 32;
    std::uniform_int_distribution<int> code(-127, 127);
    ops::DenseWeights weights{{}, bf16Values(outputs, 0.05F), {}};
    if (q8) {
      MetalBuffer plane0 = test::sharedBuffer(backend, uint64_t{outputs} * inputs);
      MetalBuffer meta = test::sharedBuffer(backend, uint64_t{outputs} * groups * 2);
      const _Float16 scale = static_cast<_Float16>(d);
      for (uint32_t n = 0; n < outputs; ++n)
        for (uint32_t g = 0; g < groups; ++g) {
          const uint64_t record = quant_tile_index(n, g, groups);
          std::memcpy(static_cast<uint8_t *>(meta.contents()) + record * 2, &scale, 2);
          for (uint32_t e = 0; e < 32; ++e)
            static_cast<int8_t *>(plane0.contents())[record * 32 + quant_slot(e)] = static_cast<int8_t>(code(random));
        }
      weights.quantized = ops::QuantizedSegment::planes(GGUF_FMT_Q80, outputs, inputs, plane0, {}, meta);
    } else {
      weights.weight = test::sharedBuffer(backend, uint64_t{outputs} * inputs * 2);
      for (uint64_t i = 0; i < uint64_t{outputs} * inputs; ++i)
        words(weights.weight)[i] = floatToBf16(d * static_cast<float>(code(random)));
    }
    return weights;
  };
  const uint32_t hidden = kHidden, merged = kLayout.mergedHiddenSize;
  const auto norm = [&] { return ops::LayerNormWeights{constant(hidden, 1.0F), constant(hidden, 0.0F)}; };
  const auto layerScale = [&] {
    std::uniform_real_distribution<float> uniform(0.1F, 0.3F);
    MetalBuffer buffer = test::sharedBuffer(backend, uint64_t{hidden} * 2);
    for (uint32_t i = 0; i < hidden; ++i) words(buffer)[i] = floatToBf16(uniform(random));
    return buffer;
  };
  ops::VisionWeights model;
  model.layout = kLayout;
  model.patchEmbedding = {bf16Values(uint64_t{hidden} * kPatchDim, 0.03F), constant(hidden, 0.0F)};
  model.positionTable = bf16Values(uint64_t{kSide} * kSide * hidden, 0.5F);
  model.classToken = bf16Values(2 * hidden, 0.5F);
  model.inputNorm = norm();
  for (uint32_t i = 0; i < kLayout.depth; ++i) {
    ops::VisionBlock block{norm(), projection(3 * hidden, hidden), projection(hidden, hidden), norm(),
                           projection(kLayout.intermediateSize, hidden), projection(hidden, kLayout.intermediateSize)};
    block.projection.scale = layerScale();
    block.downProjection.scale = layerScale();
    model.blocks.push_back(std::move(block));
  }
  model.mergerUpProjection = projection(merged, merged);
  model.mergerDownProjection = projection(kLayout.outputHiddenSize, merged);
  return model;
}

// The Q8_0 tower's embeddings of an image against the bf16 tower's: their
// RMS difference within `bound` of the bf16 tower's RMS.
void checkQ8Tower(MetalBackend &backend, const ops::VisionWeights &q8, const ops::VisionWeights &bf16,
                  ops::ImageGrid grid, double bound) {
  const MetalBuffer pixels = test::sharedBuffer(backend, grid.pixelBytes(kPatch));
  std::mt19937 random(5);
  for (uint64_t i = 0; i < grid.pixelBytes(kPatch); ++i)
    static_cast<uint8_t *>(pixels.contents())[i] = static_cast<uint8_t>(random());
  const uint64_t count = uint64_t{grid.mergedTokens()} * kLayout.outputHiddenSize;
  const auto encode = [&](const ops::VisionWeights &weights) {
    const ops::InklingVision vision(backend, weights, grid.patches());
    const MetalBuffer embeddings = bf16Buffer(backend, count);
    CommandGraph graph;
    vision.encode(graph, grid, pixels, embeddings);
    run(backend, graph);
    std::vector<double> values(count);
    for (uint64_t i = 0; i < count; ++i) values[i] = bf16ToFloat(words(embeddings)[i]);
    return values;
  };
  const std::vector<double> got = encode(q8), expected = encode(bf16);
  double difference = 0.0, magnitude = 0.0;
  for (uint64_t i = 0; i < count; ++i) {
    require(std::isfinite(got[i]), "the Q8_0 tower's embedding is not finite");
    difference += (got[i] - expected[i]) * (got[i] - expected[i]);
    magnitude += expected[i] * expected[i];
  }
  const double relative = std::sqrt(difference / magnitude);
  const std::string label = "Q8_0 tower " + std::to_string(grid.height) + "x" + std::to_string(grid.width);
  std::printf("%s: relative RMS difference %.2e (bound %.0e)\n", label.c_str(), relative, bound);
  require(relative <= bound, label + ": the embeddings differ from the bf16 tower's");
}

// The production image's tower (vision::writeVision, readInklingVision) of a
// Q8_0 mmproj, its q, k and v three tensors stacked into one projection's
// planes, against the image of the same values, every one BF16 (power-of-two
// scales), from the same writer and reader.
void checkPreparedQ8Tower(MetalBackend &backend) {
  const test::TemporaryDirectory directory("inkling-q8-image");
  const model::vision::Plan base = model::inklingVisionPlan(kLayout);
  std::mt19937 random(9);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::uniform_int_distribution<int> code(-127, 127);
  std::vector<uint8_t> file;
  const auto append = [&](const void *data, uint64_t bytes) {
    const uint64_t at = file.size();
    file.insert(file.end(), static_cast<const uint8_t *>(data), static_cast<const uint8_t *>(data) + bytes);
    return at;
  };
  // Each section's tensors: a projection's as Q8_0 blocks and as the same
  // values in BF16, the others in BF16 for both; the zero biases none.
  struct Tensor {
    size_t section;
    uint32_t rows, columns;
    uint64_t q8 = 0, bf16 = 0;
  };
  std::vector<Tensor> tensors;
  for (size_t index = 0; index < base.sections.size(); ++index) {
    const model::vision::Section &s = base.sections[index];
    if (s.gguf == "patch embedding bias" || s.gguf == "mm.1.bias" || s.gguf == "mm.2.bias") continue;
    const uint32_t parts = s.stacked ? (s.projection ? 3 : 2) : 1;
    const bool norm = s.gguf.find("ln") != std::string::npos && s.gguf.ends_with(".weight") && s.columns == 1;
    const bool layerScale = s.gguf.find(".ls") != std::string::npos;
    for (uint32_t part = 0; part < parts; ++part) {
      Tensor t{index, s.rows / parts, s.columns};
      std::vector<uint16_t> values(uint64_t{t.rows} * t.columns);
      if (s.projection) {
        const float d = t.columns > 1024 ? 0x1p-12F : 0x1p-11F;
        const _Float16 scale = static_cast<_Float16>(d);
        std::vector<uint8_t> blocks;
        for (uint64_t i = 0; i < values.size(); i += 32) {
          blocks.insert(blocks.end(), reinterpret_cast<const uint8_t *>(&scale), reinterpret_cast<const uint8_t *>(&scale) + 2);
          for (uint32_t e = 0; e < 32; ++e) {
            const int q = code(random);
            blocks.push_back(static_cast<uint8_t>(static_cast<int8_t>(q)));
            values[i + e] = floatToBf16(d * static_cast<float>(q));
          }
        }
        t.q8 = append(blocks.data(), blocks.size());
      } else {
        for (uint16_t &v : values)
          v = floatToBf16(norm ? 1.0F + 0.05F * normal(random) : layerScale ? 0.2F : 0.05F * normal(random));
      }
      t.bf16 = append(values.data(), values.size() * 2);
      tensors.push_back(t);
    }
  }
  const std::filesystem::path path = directory.path() / "mmproj.bin";
  test::writeFile(path, file);
  const model::WeightSource source(path);
  const auto tower = [&](bool q8) {
    model::vision::Plan plan = base;
    for (const Tensor &t : tensors) {
      const bool quantized = q8 && base.sections[t.section].projection;
      const uint64_t bytes = quantized ? uint64_t{t.rows} * t.columns / 32 * 34 : uint64_t{t.rows} * t.columns * 2;
      plan.sections[t.section].inputs.push_back(
          {"tensor", {&source, quantized ? "Q8_0" : "BF16", {t.columns, t.rows}, quantized ? t.q8 : t.bf16, bytes}});
    }
    model::vision::placeQ8Matrices(plan);
    MetalBuffer image = backend.allocateBuffer(plan.bytes, metal::BufferStorage::Shared, "vision image");
    model::vision::writeVision({static_cast<uint8_t *>(image.contents()), plan.bytes}, plan);
    model::WeightFile weights(backend, image, "vision/model.bin", model::kVisionMagic, kLayout.depth, 0, plan.bytes,
                              "test");
    ops::VisionWeights result = model::readInklingVision(weights, kLayout, plan);
    weights.finish();
    return std::pair{plan, std::move(result)};
  };
  const auto [q8Plan, q8] = tower(true);
  const auto [bf16Plan, bf16] = tower(false);
  uint32_t planes = 0;
  for (const model::vision::Section &s : q8Plan.sections) planes += s.q8Planes;
  require(planes == 4 * kLayout.depth + 2 && q8.blocks.front().qkv.isQuantized() && !bf16.blocks.front().qkv.isQuantized(),
          "the prepared Q8_0 tower does not hold its projections as planes");
  for (const ops::ImageGrid grid : {ops::ImageGrid{4, 4}, ops::ImageGrid{10, 10}})
    checkQ8Tower(backend, q8, bf16, grid, 1e-2);
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
      {
        const ops::VisionWeights q8 = randomTower(backend, true, 3), bf16 = randomTower(backend, false, 3);
        for (const ops::ImageGrid grid : {ops::ImageGrid{4, 4}, ops::ImageGrid{10, 10}})
          checkQ8Tower(backend, q8, bf16, grid, 1e-2);
      }
      checkPreparedQ8Tower(backend);
      std::cout << "inkling_vision_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "inkling_vision_metal_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
