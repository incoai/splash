// The vision encoder's checks of what it reads, each buffer at its extent and
// refused one element short: every weight of the tower when the encoder is
// built, and an image's pixels and embedding rows when it encodes; and the
// tower's blocks, refused one fewer or one more than its layout names. The
// test only encodes; every tensor is a view of one unwritten buffer.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/Qwen3_8.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
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

void run(metal::MetalBackend &backend) {
  const ops::VisionLayout layout = model::kQwen3_8VisionLayout;
  const uint64_t hidden = layout.hiddenSize, merged = layout.mergedHiddenSize;
  // The largest tensor is the merger's down projection.
  const metal::MetalBuffer storage =
      test::sharedBuffer(backend, uint64_t{layout.outputHiddenSize} * merged * kBf16Bytes);
  const auto tensor = [&](uint64_t elements) { return backend.view(storage, 0, elements * kBf16Bytes); };
  const auto affine = [&](uint64_t output, uint64_t input) {
    return VisionAffine{tensor(output * input), tensor(output)};
  };
  const VisionNorm norm{tensor(hidden), tensor(hidden)};
  VisionWeights model;
  model.layout = layout;
  model.patchEmbedding = affine(hidden, layout.patchDimension);
  model.positionTable = tensor(uint64_t{layout.positionGridSide} * layout.positionGridSide * hidden);
  model.blocks.assign(layout.depth, VisionBlock{norm, affine(3 * hidden, hidden), affine(hidden, hidden), norm,
                                                affine(layout.paddedIntermediateSize, hidden),
                                                affine(hidden, layout.paddedIntermediateSize)});
  model.mergerNorm = norm;
  model.mergerUpProjection = affine(merged, merged);
  model.mergerDownProjection = affine(layout.outputHiddenSize, merged);
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
    requireWeightExtent((model.*member).bias, output, name + " bias",
                        [&](VisionWeights &changed, const metal::MetalBuffer &view) { (changed.*member).bias = view; });
  };
  requireAffineExtent(&VisionWeights::patchEmbedding, hidden, layout.patchDimension, "vision patch embedding");
  requireWeightExtent(model.positionTable, uint64_t{layout.positionGridSide} * layout.positionGridSide * hidden,
                      "vision position table",
                      [](VisionWeights &changed, const metal::MetalBuffer &view) { changed.positionTable = view; });
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
  for (const auto &[field, part] : {std::pair{&VisionNorm::weight, " weight"}, std::pair{&VisionNorm::bias, " bias"}})
    requireWeightExtent(model.mergerNorm.*field, hidden, std::string("vision merger norm") + part,
                        [&](VisionWeights &changed, const metal::MetalBuffer &view) { changed.mergerNorm.*field = view; });
  requireAffineExtent(&VisionWeights::mergerUpProjection, merged, merged, "vision merger up projection");
  requireAffineExtent(&VisionWeights::mergerDownProjection, layout.outputHiddenSize, merged,
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

} // namespace

int main(int argc, char **argv) {
  try {
    test::require(argc == 2, "usage: vision-metal METALLIB");
    metal::MetalBackend backend(argv[1]);
    run(backend);
    std::cout << "vision_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "vision_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
