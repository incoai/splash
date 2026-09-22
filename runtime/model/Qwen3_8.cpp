#include "model/Qwen3_8.hpp"

#include "model/GgufTarget.hpp"

#include <algorithm>
#include <cstring>
#include <string_view>
#include <variant>

#include "metal/abi/ExecutionGeometry.h"

namespace splash::model {
namespace {

constexpr std::string_view kHeadMagic = "MDFL0002";

void validateLayout(const Qwen3_8Layout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.intermediateSize || !layout.attentionQueryHeads ||
      !layout.attentionKvHeads || !layout.attentionHeadDimension ||
      !layout.rotaryPairs || !(layout.rotaryTheta > 0.0F) ||
      !layout.fullAttentionPeriod) {
    throw WeightStoreError("Qwen3.8 layout contains a zero dimension");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
  if (layout.attentionQueryHeads * layout.attentionHeadDimension !=
      layout.attentionWidth) {
    throw WeightStoreError("Qwen3.8 attention layout is inconsistent");
  }
}

} // namespace

Qwen3_8Weights loadQwen3_8Weights(metal::MetalBackend &backend,
                                  const std::filesystem::path &directory,
                                  Qwen3_8Layout layout, bool kquant) {
  validateLayout(layout);
  auto readFfn = [&](WeightFile &file, Qwen3_8LayerWeights &layer) {
    if (kquant) {
      layer.gateProjection = readKQuantProjection(file, "mlp-gate");
      layer.upProjection = readKQuantProjection(file, "mlp-up");
      layer.downProjection = readKQuantProjection(file, "mlp-down");
      return;
    }
    layer.gateProjection = readQ4Projection(
        file, backend, layout.intermediateSize, layout.hiddenSize,
        "mlp-gate");
    layer.upProjection = readQ4Projection(
        file, backend, layout.intermediateSize, layout.hiddenSize,
        "mlp-up");
    layer.downProjection = readQ4Projection(
        file, backend, layout.hiddenSize, layout.intermediateSize,
        "mlp-down");
  };
  Qwen3_8Weights weights;
  if (kquant) {
    // The target directory holds the llama.cpp GGUF; every layer image is
    // repacked into memory as it is read.
    gguf::TargetGeometry geometry;
    geometry.layers = layout.layers;
    geometry.hiddenSize = layout.hiddenSize;
    geometry.vocabularySize = layout.vocabularySize;
    geometry.intermediateSize = layout.intermediateSize;
    geometry.gdnKeyHeads = layout.gdnKeyHeads;
    geometry.gdnValueHeads = layout.gdnValueHeads;
    geometry.gdnHeadDimension = layout.gdnHeadDimension;
    geometry.convolutionDimension = layout.convolutionDimension;
    geometry.attentionWidth = layout.attentionWidth;
    geometry.attentionHeadDimension = layout.attentionHeadDimension;
    geometry.fullAttentionPeriod = layout.fullAttentionPeriod;
    GgufTargetLoader loader(backend, findTargetGguf(directory), geometry);
    struct GgufFiles {
      GgufTargetLoader &loader;
      WeightFile layer(uint32_t index, bool) { return loader.layer(index); }
      WeightFile head(uint32_t) { return loader.head(); }
      WeightFile embedding(uint32_t, uint32_t) { return loader.embedding(); }
    };
    weights = readQwenTargetWeights<Qwen3_8Weights>(backend, layout, GgufFiles{loader},
                                                    readFfn, true);
  } else {
    weights = loadQwenTargetWeights<Qwen3_8Weights>(backend, directory, layout, kHeadMagic,
                                                    readFfn);
  }
  if (!kquant) return weights;
  // Shared scratch for the K-quant kernels: split-K partials (8 splits x 32 rows x
  // widest projection), permuted GDN out_proj activations for the prefill budget,
  // and the grouped -> tiled value-head permutation.
  const uint32_t widest = std::max({layout.packedGdnWidth, layout.packedFullWidth,
                                    layout.intermediateSize, layout.hiddenSize});
  metal::MetalBuffer partials = backend.allocateBuffer(
      uint64_t{8} * 32 * widest * 4, metal::BufferStorage::Shared, "kquant-partials");
  metal::MetalBuffer permuted = backend.allocateBuffer(
      uint64_t{SPLASH_PREFILL_TOKEN_BUDGET} * layout.attentionWidth * 2,
      metal::BufferStorage::Shared, "kquant-permuted");
  const uint32_t heads = layout.gdnValueHeads, keyHeads = layout.gdnKeyHeads;
  const uint32_t perHead = heads / keyHeads;
  metal::MetalBuffer permutation = backend.allocateBuffer(
      uint64_t{heads} * 4, metal::BufferStorage::Shared, "kquant-permutation");
  metal::MetalBuffer counters = backend.allocateBuffer(
      uint64_t{widest / 64} * 4, metal::BufferStorage::Shared, "kquant-counters");
  std::memset(counters.contents(), 0, counters.sizeBytes());
  auto *table = static_cast<uint32_t *>(permutation.contents());
  for (uint32_t tiled = 0; tiled < heads; ++tiled)
    table[tiled] = (tiled % keyHeads) * perHead + tiled / keyHeads;
  const auto attach = [&](ops::Q4Projection &p) {
    p.kqPartials = partials;
    p.kqPermuted = permuted;
    p.kqPermutation = permutation;
    p.kqCounters = counters;
  };
  for (Qwen3_8LayerWeights &layer : weights.layers) {
    std::visit([&](auto &mixer) { attach(mixer.inputProjection); attach(mixer.outputProjection); },
               layer.mixer);
    attach(layer.gateProjection);
    attach(layer.upProjection);
    attach(layer.downProjection);
  }
  attach(weights.logitsProjection);
  return weights;
}

} // namespace splash::model
