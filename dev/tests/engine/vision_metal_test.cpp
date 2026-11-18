// The vision encoder's checks of what it reads, each buffer at its extent and
// refused one element short: every weight of the tower when the encoder is
// built, and an image's pixels and embedding rows when it encodes; and the
// tower's blocks, refused one fewer or one more than its layout names. It
// checks both towers the kernels compile, Qwen3.5's and RADIO's (prefix tokens,
// a merger RMS weight, no patch or merger biases). The
// test only encodes; every tensor is a view of one unwritten buffer.
//
// It also runs EVS (ops::VideoPruning) on three small tubelets with ties and
// checks the span's rows against the processor's order computed here.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/NemotronH.hpp"
#include "model/Qwen3_8.hpp"
#include "ops/Vision.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>
#include <iostream>
#include <string>
#include <tuple>

namespace {

using namespace splash;
using ops::VisionAffine;
using ops::VisionBlock;
using ops::VisionNorm;
using ops::VisionWeights;
using test::requireExtent;

constexpr uint64_t kBf16Bytes = 2;
constexpr ops::ImageGrid kGrid{2, 2};

void run(metal::MetalBackend &backend, const ops::VisionLayout &layout, bool radio) {
  const uint64_t hidden = layout.hiddenSize, merged = layout.mergedHiddenSize;
  const uint64_t intermediate = layout.mergerIntermediateSize;
  const uint64_t positions = uint64_t{layout.positionGridSide} * layout.positionGridSide * hidden;
  // The largest tensor: a merger projection or the position table.
  const metal::MetalBuffer storage = test::sharedBuffer(
      backend, std::max({uint64_t{layout.outputHiddenSize} * intermediate * kBf16Bytes,
                         intermediate * merged * kBf16Bytes, positions * kBf16Bytes}));
  const auto tensor = [&](uint64_t elements) { return backend.view(storage, 0, elements * kBf16Bytes); };
  const auto affine = [&](uint64_t output, uint64_t input) {
    return VisionAffine{tensor(output * input), tensor(output)};
  };
  const auto unbiased = [&](uint64_t output, uint64_t input) { return VisionAffine{tensor(output * input), {}}; };
  const VisionNorm norm{tensor(hidden), tensor(hidden)};
  VisionWeights model;
  model.layout = layout;
  model.patchEmbedding = radio ? unbiased(hidden, layout.patchDimension) : affine(hidden, layout.patchDimension);
  if (radio) model.videoPatchEmbedding = unbiased(hidden, 2 * layout.patchDimension);
  model.positionTable = tensor(positions);
  model.blocks.assign(layout.depth, VisionBlock{norm, affine(3 * hidden, hidden), affine(hidden, hidden), norm,
                                                affine(layout.paddedIntermediateSize, hidden),
                                                affine(hidden, layout.paddedIntermediateSize)});
  model.mergerNorm = radio ? VisionNorm{tensor(merged), {}} : norm;
  model.mergerUpProjection = radio ? unbiased(intermediate, merged) : affine(intermediate, merged);
  model.mergerDownProjection = radio ? unbiased(layout.outputHiddenSize, intermediate)
                                     : affine(layout.outputHiddenSize, intermediate);
  if (radio) model.prefixTokens = tensor(uint64_t{layout.prefixTokens} * hidden);
  const metal::MetalBuffer pixels = test::sharedBuffer(backend, kGrid.pixelBytes());
  const metal::MetalBuffer embeddings = test::sharedBuffer(
      backend, uint64_t{ops::Vision::embeddingRows(kGrid)} * layout.outputHiddenSize * kBf16Bytes);
  const auto encode = [&](metal::CommandGraph &graph, const VisionWeights &weights, const metal::MetalBuffer &image,
                          const metal::MetalBuffer &output) {
    const ops::Vision vision(backend, weights, kGrid.patches());
    vision.encode(graph, kGrid, image, output);
  };

  // A weight `with` puts in place of its buffer, the last block's for each
  // block's.
  const auto requireWeightExtent = [&](const metal::MetalBuffer &buffer, uint64_t elements, const std::string &name,
                                       const auto &with) {
    requireExtent(backend, buffer, elements * kBf16Bytes, kBf16Bytes, name,
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    VisionWeights changed = model;
                    with(changed, view);
                    encode(graph, changed, pixels, embeddings);
                  });
  };
  const auto requireAffineExtent = [&](VisionAffine VisionWeights::*member, uint64_t output, uint64_t input,
                                       const std::string &name) {
    requireWeightExtent((model.*member).weight, output * input, name + " weight",
                        [&](VisionWeights &changed, const metal::MetalBuffer &view) { (changed.*member).weight = view; });
    if ((model.*member).bias)
      requireWeightExtent((model.*member).bias, output, name + " bias",
                          [&](VisionWeights &changed, const metal::MetalBuffer &view) { (changed.*member).bias = view; });
  };
  requireAffineExtent(&VisionWeights::patchEmbedding, hidden, layout.patchDimension, "vision patch embedding");
  requireWeightExtent(
      model.positionTable, positions, "vision position table",
      [](VisionWeights &changed, const metal::MetalBuffer &view) { changed.positionTable = view; });
  if (radio)
    requireWeightExtent(
        model.prefixTokens, uint64_t{layout.prefixTokens} * hidden, "vision prefix tokens",
        [](VisionWeights &changed, const metal::MetalBuffer &view) { changed.prefixTokens = view; });
  for (const auto &[member, name] : {std::pair{&VisionBlock::norm1, "vision attention norm"},
                                     std::pair{&VisionBlock::norm2, "vision MLP norm"}})
    for (const auto &[field, part] : {std::pair{&VisionNorm::weight, " weight"}, std::pair{&VisionNorm::bias, " bias"}})
      requireWeightExtent(model.blocks.back().*member.*field, hidden, std::string(name) + part,
                          [&](VisionWeights &changed, const metal::MetalBuffer &view) {
                            changed.blocks.back().*member.*field = view;
                          });
  for (const auto &[member, output, input, name] :
       {std::tuple{&VisionBlock::qkv, 3 * hidden, hidden, "vision q/k/v"},
        std::tuple{&VisionBlock::projection, hidden, hidden, "vision attention projection"},
        std::tuple{&VisionBlock::upProjection, uint64_t{layout.paddedIntermediateSize}, hidden, "vision up projection"},
        std::tuple{&VisionBlock::downProjection, hidden, uint64_t{layout.paddedIntermediateSize},
                   "vision down projection"}})
    for (const auto &[field, part, elements] :
         {std::tuple{&VisionAffine::weight, " weight", output * input}, std::tuple{&VisionAffine::bias, " bias", output}})
      requireWeightExtent(model.blocks.back().*member.*field, elements, std::string(name) + part,
                          [&](VisionWeights &changed, const metal::MetalBuffer &view) {
                            changed.blocks.back().*member.*field = view;
                          });
  if (radio)
    requireWeightExtent(
        model.mergerNorm.weight, merged, "vision merger norm weight",
        [&](VisionWeights &changed, const metal::MetalBuffer &view) { changed.mergerNorm.weight = view; });
  else
    for (const auto &[field, part] : {std::pair{&VisionNorm::weight, " weight"}, std::pair{&VisionNorm::bias, " bias"}})
      requireWeightExtent(model.mergerNorm.*field, hidden, std::string("vision merger norm") + part,
                          [&](VisionWeights &changed, const metal::MetalBuffer &view) { changed.mergerNorm.*field = view; });
  requireAffineExtent(&VisionWeights::mergerUpProjection, intermediate, merged, "vision merger up projection");
  requireAffineExtent(&VisionWeights::mergerDownProjection, layout.outputHiddenSize, intermediate,
                      "vision merger down projection");
  for (const uint32_t depth : {layout.depth - 1, layout.depth + 1}) {
    VisionWeights changed = model;
    changed.blocks.resize(depth, model.blocks.back());
    test::rejects([&] { const ops::Vision vision(backend, changed, kGrid.patches()); },
                  "vision tower holds " + std::to_string(depth) + " blocks, needs " + std::to_string(layout.depth),
                  "a tower of " + std::to_string(depth) + " blocks was accepted");
  }

  requireExtent(backend, pixels, kGrid.pixelBytes(), 1, "image pixel",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  encode(graph, model, view, embeddings);
                });
  requireExtent(backend, embeddings, embeddings.sizeBytes(), kBf16Bytes, "image embedding",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  encode(graph, model, pixels, view);
                });
}

// Round to bf16, to nearest even, as Metal's bfloat conversion does.
float bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000U;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint16_t bf16Bits(float value) {
  uint32_t bits = 0;
  value = bf16(value);
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

// `kept` 7 keeps the most changed rows; 11 drops one of three tied rows,
// the last. A NaN row sorts first, as in torch, and still exactly `kept`
// rows are kept.
void pruning(metal::MetalBackend &backend, uint32_t kept, bool nan = false) {
  // Each row (1, s): its tubelets' slopes, with a repeat and two ties.
  std::vector<std::vector<float>> slopes{
      {0.0F, 0.5F, 1.0F, 2.0F}, {0.0F, 0.75F, 0.5F, 4.0F}, {0.0F, 0.75F, 2.0F, 0.25F}};
  if (nan) slopes[1][2] = std::numeric_limits<float>::quiet_NaN();
  const ops::VideoPruningShape shape{3, {4, 4, 2}, 256, kept, 5};
  const std::vector<uint32_t> cuts{0, 2, 4};
  const uint32_t rows = shape.grid.mergedTokens(), stride = ops::Vision::embeddingRows(shape.grid);
  const uint32_t total = shape.tubelets * rows, width = shape.width;
  const auto bytes = [&](uint64_t count) { return count * width * kBf16Bytes; };
  const metal::MetalBuffer encoded = test::sharedBuffer(backend, bytes(uint64_t{shape.tubelets} * stride));
  const metal::MetalBuffer text = test::sharedBuffer(backend, bytes(shape.textTokens));
  const metal::MetalBuffer cutBuffer = test::sharedBuffer(backend, cuts.size() * sizeof(uint32_t));
  const metal::MetalBuffer dissimilarity = test::sharedBuffer(backend, ops::VideoPruning::flagBytes(shape));
  const metal::MetalBuffer keep = test::sharedBuffer(backend, ops::VideoPruning::flagBytes(shape));
  const metal::MetalBuffer output = test::sharedBuffer(backend, bytes(shape.textTokens + shape.kept));
  auto *values = static_cast<uint16_t *>(encoded.contents());
  std::memset(values, 0, encoded.sizeBytes());
  for (uint32_t t = 0; t < shape.tubelets; ++t)
    for (uint32_t p = 0; p < rows; ++p) {
      values[uint64_t{t * stride + p} * width] = bf16Bits(1.0F);
      values[uint64_t{t * stride + p} * width + 1] = bf16Bits(slopes[t][p]);
    }
  auto *textValues = static_cast<uint16_t *>(text.contents());
  for (uint32_t j = 0; j < shape.textTokens; ++j)
    std::fill_n(textValues + uint64_t{j} * width, width, bf16Bits(100.0F + j));
  std::memcpy(cutBuffer.contents(), cuts.data(), cuts.size() * sizeof(uint32_t));

  metal::CommandGraph graph;
  ops::VideoPruning::add(graph, shape, encoded, text, cutBuffer, dissimilarity, keep, output);
  backend.preparePipelines(graph.dispatches());
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  // The processor's order: 1 - cosine of each row with its tubelet before,
  // each step rounded to bf16; 255 for the first tubelet; stable descending.
  std::vector<float> d(total, 255.0F);
  for (uint32_t t = 1; t < shape.tubelets; ++t)
    for (uint32_t p = 0; p < rows; ++p) {
      const float a = slopes[t][p], b = slopes[t - 1][p];
      const float an = bf16(std::sqrt(1.0F + a * a)), bn = bf16(std::sqrt(1.0F + b * b));
      const float dot = bf16(bf16(bf16(1.0F / an) * bf16(1.0F / bn)) + bf16(bf16(a / an) * bf16(b / bn)));
      d[t * rows + p] = bf16(1.0F - dot);
    }
  std::vector<uint32_t> order(total);
  std::iota(order.begin(), order.end(), 0U);
  std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
    return std::isnan(d[x]) ? !std::isnan(d[y]) : d[x] > d[y];
  });
  std::vector<bool> keeps(total);
  for (uint32_t i = 0; i < shape.kept; ++i) keeps[order[i]] = true;
  // The span's rows: text up to each tubelet's cut, its kept rows, the rest.
  // Each is the bits of a text row's value, or of a tubelet row's slope.
  std::vector<std::pair<bool, uint16_t>> expected;
  uint32_t token = 0;
  for (uint32_t t = 0; t < shape.tubelets; ++t) {
    for (; token < cuts[t]; ++token) expected.emplace_back(true, bf16Bits(100.0F + token));
    for (uint32_t p = 0; p < rows; ++p)
      if (keeps[t * rows + p]) expected.emplace_back(false, bf16Bits(slopes[t][p]));
  }
  for (; token < shape.textTokens; ++token) expected.emplace_back(true, bf16Bits(100.0F + token));
  test::require(expected.size() == shape.textTokens + shape.kept, "the reference lays out the wrong row count");
  const auto *out = static_cast<const uint16_t *>(output.contents());
  for (uint32_t row = 0; row < expected.size(); ++row) {
    const uint16_t *got = out + uint64_t{row} * width;
    const auto [isText, bits] = expected[row];
    const bool matches = isText ? got[0] == bits && got[width - 1] == bits
                                : got[0] == bf16Bits(1.0F) && got[1] == bits && got[width - 1] == 0;
    test::require(matches, "video pruning row " + std::to_string(row) + " is not the processor's");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    test::require(argc == 2, "usage: vision-metal METALLIB");
    metal::MetalBackend backend(argv[1]);
    run(backend, ops::VisionLayout{.outputHiddenSize = model::Qwen3_8Layout{}.hiddenSize}, false);
    ops::VisionLayout radio = ops::kRadioVisionLayout;
    radio.outputHiddenSize = model::NemotronHLayout{}.hiddenSize;
    run(backend, radio, true);
    pruning(backend, 7);
    pruning(backend, 11);
    pruning(backend, 7, true);
    // A tower the kernels do not compile.
    radio.heads = 20;
    test::rejects([&] { static_cast<void>(ops::Vision::scratchBytes(radio, 4)); }, "no vision kernels for the tower",
                  "an uncompiled tower was accepted");
    std::cout << "vision_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "vision_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
