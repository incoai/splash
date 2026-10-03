#pragma once

#include "model/AffineTarget.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/QwenTarget.hpp"
#include "model/QwenTargetFiles.hpp"
#include "model/WeightStore.hpp"
#include "ops/GDN.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <variant>

namespace splash::model {

// How a target's files store its tensors; loadQwenTarget pairs each source's
// files with their format. Affine files, packed or prepared from MLX, hold
// every projection, a fused one too, as one affine Q4 tensor, and bf16 norms.
struct AffineTargetFormat final {
  static constexpr ops::GdnHeadOrder gdnOutputOrder = ops::GdnHeadOrder::Grouped;

  [[nodiscard]] ops::NormWeights norm(WeightFile &file, uint32_t width, std::string_view label) const {
    return readNorm(file, width, false, label);
  }
  [[nodiscard]] ops::Projection projection(WeightFile &file, uint32_t outputSize,
                                           uint32_t inputSize, std::string_view label) const {
    return readAffineProjection(file, outputSize, inputSize, label);
  }
  // The tensor `label`; block images keep the projection as `tensors`.
  [[nodiscard]] ops::Projection fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                      std::string_view label,
                                      std::initializer_list<std::string_view>) const {
    return projection(file, outputSize, inputSize, label);
  }
  [[nodiscard]] ops::EmbeddingWeights embedding(WeightFile &file, uint32_t outputSize,
                                                uint32_t inputSize) const {
    return readAffineEmbedding(file, outputSize, inputSize, "embedding");
  }
};

// Prepared GGUF images hold each GGUF tensor as one block-quantized segment,
// a fused projection as its tensors in output column order, and the GGUF's
// F32 norms. The GGUF keeps the GDN output projection's input columns in
// llama.cpp's tiled value-head order, so the GDN writes its output in it; a
// rotated Prism ML GGUF keeps them grouped, and rotateInputs (Qwen3_8.cpp)
// switches its GDN to that order.
struct BlockTargetFormat final {
  static constexpr ops::GdnHeadOrder gdnOutputOrder = ops::GdnHeadOrder::Tiled;

  [[nodiscard]] ops::NormWeights norm(WeightFile &file, uint32_t width, std::string_view label) const {
    return readNorm(file, width, true, label);
  }
  [[nodiscard]] ops::Projection projection(WeightFile &file, uint32_t outputSize,
                                           uint32_t inputSize, std::string_view label) const {
    return readBlockProjection(file, outputSize, inputSize, label);
  }
  // The tensors, which may leave padding columns past the last one
  // (LinearGguf.cpp requireSegments); affine files keep one tensor.
  [[nodiscard]] ops::Projection fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                      std::string_view,
                                      std::initializer_list<std::string_view> tensors) const;
  [[nodiscard]] ops::EmbeddingWeights embedding(WeightFile &file, uint32_t outputSize,
                                                uint32_t inputSize) const {
    return readBlockEmbedding(file, outputSize, inputSize, "embedding");
  }
};

// Reads the mixer sections that follow a layer's input norm, in file order
// (instantiated for both formats).
template <class Format>
[[nodiscard]] QwenMixerWeights readQwenMixer(WeightFile &file, const Format &format,
                                             const QwenMixerGeometry &geometry,
                                             bool fullAttention);

// Opens the packed files of a target directory: one per hybrid layer, head.bin
// and embedding.bin.
template <class Layout> struct PackedTargetFiles final {
  metal::MetalBackend &backend;
  std::filesystem::path directory;
  const Layout &layout;
  [[nodiscard]] WeightFile layer(uint32_t index) const {
    const std::string filename = "layer-" + std::to_string(index) + ".bin";
    return WeightFile(backend, directory / filename, "target/" + filename, Layout::layerMagic, index,
                      layout.isFullAttentionLayer(index) ? 1U : 0U);
  }
  [[nodiscard]] WeightFile head() const {
    return WeightFile(backend, directory / "head.bin", "target/head.bin", Layout::headMagic, layout.layers, 2);
  }
  [[nodiscard]] WeightFile embedding() const {
    return WeightFile(backend, directory / "embedding.bin", "target/embedding.bin", kEmbeddingMagic,
                      layout.vocabularySize, layout.hiddenSize);
  }
};

// Reads a target through immutable WeightFiles, packaged or prepared locally,
// in their format: per layer the input norm, mixer, post-attention norm and
// the architecture's FFN through readFfn, then the head and the token
// embedding. Weights is the architecture's weight struct.
template <class Weights, class Layout, class Files, class Format, class ReadFfn>
[[nodiscard]] Weights
readQwenTargetWeights(metal::MetalBackend &backend, const Layout &layout, Files &&files,
                      const Format &format, ReadFfn readFfn) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Weights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const bool fullAttention = layout.isFullAttentionLayer(layerIndex);
    WeightFile file = files.layer(layerIndex);
    auto &layer = result.layers.emplace_back();
    layer.inputNorm = format.norm(file, layout.hiddenSize, "input-norm");
    layer.mixer = readQwenMixer(file, format, layout.mixerGeometry(), fullAttention);
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

// Throws unless every dimension of a family's layout is set, the dimensions
// agree with each other and every projection fits the Q4 storage tiles.
template <class Layout> void requireQwenLayout(const Layout &layout) {
  const auto zero = [](auto... dimensions) { return ((dimensions == 0) || ...); };
  uint32_t ffnWidth = 0;
  bool ffnZero = false;
  bool routingInconsistent = false;
  if constexpr (Layout::ffnKind == QwenFfnKind::Dense) {
    ffnWidth = layout.intermediateSize;
    ffnZero = zero(ffnWidth);
  } else {
    ffnWidth = layout.expertIntermediateSize;
    ffnZero = zero(layout.experts, layout.expertsPerToken, ffnWidth);
    routingInconsistent = layout.expertsPerToken > layout.experts;
  }
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
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(ffnWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, ffnWidth);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

// Checks the layout and loads a target from its files. The architecture
// reads its FFN through readFfn, called with the file, the layer and the
// format.
template <class Weights, class Layout, class ReadFfn>
[[nodiscard]] Weights
loadQwenTarget(metal::MetalBackend &backend, const Layout &layout, const QwenTargetFiles<Layout> &files,
               ReadFfn readFfn) {
  requireQwenLayout(layout);
  if (const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files))
    return readQwenTargetWeights<Weights>(backend, layout, gguf->get(), BlockTargetFormat{}, readFfn);
  const AffineTargetFormat affine{};
  if (const auto *mlx = std::get_if<std::reference_wrapper<AffineTargetLoader>>(&files))
    return readQwenTargetWeights<Weights>(backend, layout, mlx->get(), affine, readFfn);
  return readQwenTargetWeights<Weights>(backend, layout, std::get<PackedTargetFiles<Layout>>(files), affine,
                                        readFfn);
}

} // namespace splash::model
