#pragma once

#include "model/GgufTarget.hpp"
#include "model/MlxTarget.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/QwenTarget.hpp"
#include "model/QwenTargetFiles.hpp"
#include "model/WeightStore.hpp"
#include "ops/GDN.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string_view>
#include <variant>

namespace splash::model {

// How a target's block images store its tensors; loadQwenTarget pairs each
// source's images with their format. Block images hold each tensor as one
// block-quantized segment, a fused projection as its tensors in output column
// order. A GGUF's keep its F32 norms and the GDN output projection's input
// columns in llama.cpp's tiled value-head order, so the GDN writes its output
// in it (a rotated Prism ML GGUF keeps them grouped, and rotateInputs,
// Qwen3_8.cpp, switches its GDN to that order); an MLX target's
// (model/MlxImage.hpp) keep its bf16 norms and grouped value heads.
struct BlockTargetFormat final {
  bool float32Norms = true;
  ops::GdnHeadOrder gdnOutputOrder = ops::GdnHeadOrder::Tiled;

  [[nodiscard]] ops::NormWeights norm(WeightFile &file, uint32_t width, std::string_view label) const {
    return readNorm(file, width, float32Norms, label);
  }
  [[nodiscard]] ops::Projection projection(WeightFile &file, uint32_t outputSize,
                                           uint32_t inputSize, std::string_view label) const {
    return readBlockProjection(file, outputSize, inputSize, label);
  }
  // The tensors, which may leave padding columns past the last one
  // (LinearGguf.cpp requireSegments).
  [[nodiscard]] ops::Projection fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                      std::initializer_list<std::string_view> tensors) const;
  [[nodiscard]] ops::EmbeddingWeights embedding(WeightFile &file, uint32_t outputSize,
                                                uint32_t inputSize) const {
    return readBlockEmbedding(file, outputSize, inputSize, "embedding");
  }
};

// Reads the mixer sections that follow a layer's input norm, in file order.
[[nodiscard]] QwenMixerWeights readQwenMixer(WeightFile &file, const BlockTargetFormat &format,
                                             const QwenTargetDimensions &target,
                                             bool fullAttention);

// Reads a target through the files of its images, written from an upstream
// source, in their format: per layer the input norm, mixer,
// post-attention norm and the architecture's FFN through readFfn, then the
// head and the token embedding. Weights is the architecture's weight struct.
template <class Weights, class Layout, class Files, class ReadFfn>
[[nodiscard]] Weights
readQwenTargetWeights(metal::MetalBackend &backend, const Layout &layout, Files &&files,
                      const BlockTargetFormat &format, ReadFfn readFfn) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Weights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const bool fullAttention = layout.isFullAttentionLayer(layerIndex);
    WeightFile file = files.layer(layerIndex);
    auto &layer = result.layers.emplace_back();
    layer.inputNorm = format.norm(file, layout.hiddenSize, "input-norm");
    layer.mixer = readQwenMixer(file, format, layout, fullAttention);
    layer.postAttentionNorm = format.norm(file, layout.hiddenSize, "post-attention-norm");
    readFfn(file, layer, format);
    file.finish();
    result.files.push_back(file.record());
  }

  {
    WeightFile file = files.head();
    result.finalNorm = format.norm(file, layout.hiddenSize, "final-norm");
    result.logitsProjection =
        format.projection(file, layout.vocabularySize, layout.hiddenSize, "logits");
    // bf16 logits would round near-ties together: their spacing is 0.125 at
    // logits of 16 to 32.
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = files.embedding();
    result.tokenEmbedding = format.embedding(file, layout.vocabularySize, layout.hiddenSize);
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

// Throws unless every dimension of a family's layout is set and the
// dimensions agree with each other. The image planners hold each tensor to
// whole plane tiles.
template <class Layout> void requireQwenLayout(const Layout &layout) {
  const auto zero = [](auto... dimensions) { return ((dimensions == 0) || ...); };
  const bool dense = layout.ffnKind == QwenFfnKind::Dense;
  const uint32_t ffnWidth = dense ? layout.intermediateSize : layout.expertIntermediateSize;
  const bool ffnZero = dense ? zero(ffnWidth) : zero(layout.experts, layout.expertsPerToken, ffnWidth);
  const bool routingInconsistent = !dense && layout.expertsPerToken > layout.experts;
  if (ffnZero || !(layout.rotaryTheta > 0.0F) ||
      zero(layout.maximumContextTokens, layout.layers, layout.hiddenSize, layout.vocabularySize,
           layout.packedGdnWidth, layout.packedFullWidth, layout.convolutionDimension, layout.gdnKeyHeads,
           layout.gdnValueHeads, layout.gdnHeadDimension, layout.attentionWidth, layout.attentionQueryHeads,
           layout.attentionKvHeads, layout.attentionHeadDimension, layout.rotaryPairs,
           layout.fullAttentionPeriod))
    throw WeightStoreError("Qwen target layout contains a zero dimension");
  if (routingInconsistent || layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension != (2 * layout.gdnKeyHeads + layout.gdnValueHeads) * layout.gdnHeadDimension ||
      layout.attentionWidth != layout.attentionQueryHeads * layout.attentionHeadDimension ||
      // The GDN value rows are sized with attentionWidth throughout.
      layout.gdnValueHeads * layout.gdnHeadDimension != layout.attentionWidth ||
      layout.packedFullWidth !=
          2 * layout.attentionWidth + 2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      std::ranges::any_of(layout.hiddenCaptureLayers, [&](uint32_t layer) { return layer >= layout.layers; }) ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid())
    throw WeightStoreError("Qwen target layout is inconsistent");
}

// Checks the layout and loads a target from its files. The architecture
// reads its FFN through readFfn, called with the file, the layer and the
// format.
template <class Weights, class Layout, class ReadFfn>
[[nodiscard]] Weights
loadQwenTarget(metal::MetalBackend &backend, const Layout &layout, const QwenTargetFiles &files,
               ReadFfn readFfn) {
  requireQwenLayout(layout);
  if (const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files))
    return readQwenTargetWeights<Weights>(backend, layout, gguf->get(), BlockTargetFormat{}, readFfn);
  return readQwenTargetWeights<Weights>(backend, layout,
                                        std::get<std::reference_wrapper<MlxTargetLoader>>(files).get(),
                                        BlockTargetFormat{false, ops::GdnHeadOrder::Grouped}, readFfn);
}

} // namespace splash::model
