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
                                  Qwen3_8Layout layout, bool ggufTarget) {
  validateLayout(layout);
  auto readFfn = [&](WeightFile &file, Qwen3_8LayerWeights &layer) {
    if (ggufTarget) {
      layer.gateProjection = readGgufProjection(file, "mlp-gate");
      layer.upProjection = readGgufProjection(file, "mlp-up");
      layer.downProjection = readGgufProjection(file, "mlp-down");
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
  if (ggufTarget) {
    // The target directory holds the llama.cpp GGUF; every layer image is
    // repacked into memory as it is read.
    GgufTargetLoader loader(backend, findTargetGguf(directory), ggufTargetGeometry(layout));
    weights = readQwenTargetWeights<Qwen3_8Weights>(backend, layout, GgufTargetFiles{loader},
                                                    readFfn, true);
  } else {
    weights = loadQwenTargetWeights<Qwen3_8Weights>(backend, directory, layout, kHeadMagic,
                                                    readFfn);
  }
  return weights;
}

} // namespace splash::model
