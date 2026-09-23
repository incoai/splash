#include "Qwen3_6Moe.hpp"

#include "model/GgufTarget.hpp"

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
      !layout.q8Layout().valid() || !layout.gdnStateLayout().valid()) {
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
                      Qwen3_6MoeLayout layout, bool ggufTarget) {
  requireLayout(layout);
  if (ggufTarget) {
    // The target directory holds the llama.cpp GGUF; every layer image is
    // repacked into memory as it is read (model/GgufImage.hpp).
    GgufTargetLoader loader(backend, findTargetGguf(directory),
                            ggufTargetGeometry(layout));
    return readQwenTargetWeights<Qwen3_6MoeWeights>(
        backend, layout, GgufTargetFiles{loader},
        [](WeightFile &file, Qwen3_6MoeLayerWeights &layer) {
          ops::GgufMoeWeights ffn;
          ffn.router = readGgufSegment(file, "router");
          ffn.gate.routed = readGgufSegment(file, "experts-gate");
          ffn.up.routed = readGgufSegment(file, "experts-up");
          ffn.down.routed = readGgufSegment(file, "experts-down");
          ffn.gate.shared = readGgufSegment(file, "shared-expert-gate");
          ffn.up.shared = readGgufSegment(file, "shared-expert-up");
          ffn.down.shared = readGgufSegment(file, "shared-expert-down");
          ffn.sharedExpertGate =
              readGgufSegment(file, "shared-expert-scalar-gate");
          layer.ffn.gguf = std::move(ffn);
        },
        true);
  }
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
