#include "model/Qwen3_8.hpp"

#include <string_view>

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
                                  Qwen3_8Layout layout) {
  validateLayout(layout);
  return loadQwenTargetWeights<Qwen3_8Weights>(
      backend, directory, layout, kHeadMagic,
      [&](WeightFile &file, Qwen3_8LayerWeights &layer) {
        layer.gateProjection = readQ4Projection(
            file, backend, layout.intermediateSize, layout.hiddenSize,
            "mlp-gate");
        layer.upProjection = readQ4Projection(
            file, backend, layout.intermediateSize, layout.hiddenSize,
            "mlp-up");
        layer.downProjection = readQ4Projection(
            file, backend, layout.hiddenSize, layout.intermediateSize,
            "mlp-down");
      });
}

void validateLayout(const Qwen3_8Q8Layout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.intermediateSize || !layout.attentionQueryHeads ||
      !layout.attentionKvHeads || !layout.attentionHeadDimension ||
      !layout.rotaryPairs || !(layout.rotaryTheta > 0.0F) ||
      !layout.fullAttentionPeriod) {
    throw WeightStoreError("Qwen3.8 Q8 layout contains a zero dimension");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
  if (layout.attentionQueryHeads * layout.attentionHeadDimension !=
      layout.attentionWidth) {
    throw WeightStoreError("Qwen3.8 Q8 attention layout is inconsistent");
  }
}

Qwen3_8Q8Weights loadQwen3_8Q8Weights(metal::MetalBackend &backend,
                                      const std::filesystem::path &directory,
                                      Qwen3_8Q8Layout layout) {
  validateLayout(layout);
  return loadQwenTargetWeights<Qwen3_8Q8Weights>(
      backend, directory, layout, Qwen3_8Q8Layout::headMagic,
      [&](WeightFile &file, Qwen3_8Q8LayerWeights &layer) {
        layer.gateProjection = readQ8Projection(
            file, backend, layout.intermediateSize, layout.hiddenSize,
            "mlp-gate");
        layer.upProjection = readQ8Projection(
            file, backend, layout.intermediateSize, layout.hiddenSize,
            "mlp-up");
        layer.downProjection = readQ8Projection(
            file, backend, layout.hiddenSize, layout.intermediateSize,
            "mlp-down");
      });
}

} // namespace splash::model
