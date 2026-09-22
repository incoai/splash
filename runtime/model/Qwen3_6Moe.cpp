#include "Qwen3_6Moe.hpp"

#include <string_view>

namespace splash::model {
namespace {

constexpr std::string_view kHeadMagic = "MDFM0002";

void requireLayout(const Qwen3_6MoeLayout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.attentionQueryHeads || !layout.attentionKvHeads ||
      !layout.attentionHeadDimension || !layout.rotaryPairs ||
      !(layout.rotaryTheta > 0.0F) || !layout.fullAttentionPeriod ||
      !layout.experts || !layout.expertsPerToken ||
      !layout.expertIntermediateSize) {
    throw WeightStoreError("Qwen3.6 MoE layout contains a zero dimension");
  }
  if (layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension !=
          (2 * layout.gdnKeyHeads + layout.gdnValueHeads) *
              layout.gdnHeadDimension ||
      layout.attentionWidth !=
          layout.attentionQueryHeads * layout.attentionHeadDimension ||
      layout.packedFullWidth !=
          2 * layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      layout.expertsPerToken > layout.experts ||
      layout.hiddenCaptureLayers.back() >= layout.layers ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid()) {
    throw WeightStoreError("Qwen3.6 MoE layout is inconsistent");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.expertIntermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.expertIntermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

} // namespace

Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend,
                      const std::filesystem::path &directory,
                      Qwen3_6MoeLayout layout) {
  requireLayout(layout);
  return loadQwenTargetWeights<Qwen3_6MoeWeights>(
      backend, directory, layout, kHeadMagic,
      [&](WeightFile &file, Qwen3_6MoeLayerWeights &layer) {
        layer.ffn.router = readQ8Projection(
            file, backend, layout.experts, layout.hiddenSize, "router");
        layer.ffn.expertGate = readExpertQ4Projection(
            file, layout.experts, layout.expertIntermediateSize,
            layout.hiddenSize, "experts-gate");
        layer.ffn.expertUp = readExpertQ4Projection(
            file, layout.experts, layout.expertIntermediateSize,
            layout.hiddenSize, "experts-up");
        layer.ffn.expertDown = readExpertQ4Projection(
            file, layout.experts, layout.hiddenSize,
            layout.expertIntermediateSize, "experts-down");
        layer.ffn.sharedGate = readExpertQ4Projection(
            file, 1, layout.expertIntermediateSize, layout.hiddenSize,
            "shared-expert-gate");
        layer.ffn.sharedUp = readExpertQ4Projection(
            file, 1, layout.expertIntermediateSize, layout.hiddenSize,
            "shared-expert-up");
        layer.ffn.sharedDown = readExpertQ4Projection(
            file, 1, layout.hiddenSize, layout.expertIntermediateSize,
            "shared-expert-down");
        layer.ffn.sharedExpertGate = readQ8Projection(
            file, backend, kQ4StorageN, layout.hiddenSize,
            "shared-expert-scalar-gate");
      });
}

} // namespace splash::model
