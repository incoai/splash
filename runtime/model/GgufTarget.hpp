#pragma once

// Source adapter for a Qwen GGUF. Preparation writes immutable cached files;
// serving uses the same read-only WeightFile mappings as packaged weights.

#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/PreparedFiles.hpp"
#include "model/QwenHybridLayout.hpp"

namespace splash::model {

// The single .gguf in a target directory (shards are not supported).
[[nodiscard]] std::filesystem::path findTargetGguf(const std::filesystem::path &directory);

class GgufTargetLoader final {
public:
  // Plans every image from the GGUF's metadata once.
  GgufTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &path,
                   const gguf::TargetGeometry &geometry, PreparationCheck admitConversion = {});
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

  // Every image's cache identity and size, layers first, for the model's
  // disk check before the first image is written.
  [[nodiscard]] std::span<const PreparedWeight> weights() const noexcept { return weights_; }
  // Writes every missing image and maps none.
  void prepare();

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // The input rotation of a Prism ML GGUF, which planImages checked names
  // every quantized tensor of the target and its token table.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return rotation_; }

private:
  [[nodiscard]] WeightWriter writer(size_t index);
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightSource source_;
  std::vector<gguf::Image> images_; // layers, head, embedding
  std::optional<GgufRotation> rotation_;
  std::vector<PreparedWeight> weights_;
  PreparedFiles files_;
};

// The GGUF geometry of a Qwen layout with its family's dense or sparse MoE
// FFN.
template <class Layout>
[[nodiscard]] gguf::TargetGeometry ggufTargetGeometry(const Layout &layout) {
  gguf::TargetGeometry geometry;
  geometry.layers = layout.layers;
  geometry.hiddenSize = layout.hiddenSize;
  geometry.vocabularySize = layout.vocabularySize;
  geometry.gdnKeyHeads = layout.gdnKeyHeads;
  geometry.gdnValueHeads = layout.gdnValueHeads;
  geometry.gdnHeadDimension = layout.gdnHeadDimension;
  geometry.convolutionDimension = layout.convolutionDimension;
  geometry.attentionWidth = layout.attentionWidth;
  geometry.attentionKvHeads = layout.attentionKvHeads;
  geometry.attentionHeadDimension = layout.attentionHeadDimension;
  geometry.rotaryPairs = layout.rotaryPairs;
  geometry.rotaryTheta = layout.rotaryTheta;
  geometry.fullAttentionPeriod = layout.fullAttentionPeriod;
  if constexpr (Layout::ffnKind == QwenFfnKind::SparseMoe) {
    geometry.experts = layout.experts;
    geometry.expertsPerToken = layout.expertsPerToken;
    geometry.expertIntermediateSize = layout.expertIntermediateSize;
  } else {
    geometry.intermediateSize = layout.intermediateSize;
  }
  return geometry;
}

} // namespace splash::model
