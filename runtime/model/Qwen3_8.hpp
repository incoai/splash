#pragma once

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/RoPE.h"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <string_view>

namespace splash::model {

struct Qwen3_8Layout final : QwenHybridLayout {
  // The family's name, as the installer pairs its draft (install/families.py).
  static constexpr std::string_view family = "Qwen3.8-27B";
  // Its config's text model type (transformers') and its GGUFs' architecture
  // (llama.cpp's); a safetensors checkpoint of it loads too.
  static constexpr std::string_view modelType = "qwen3_5_text";
  static constexpr std::string_view ggufArchitecture = "qwen35";
  static constexpr bool loadsSafetensors = true;

  constexpr Qwen3_8Layout()
      : QwenHybridLayout{{.maximumContextTokens = 262'144,
                          .layers = 64,
                          .hiddenSize = 5120,
                          .vocabularySize = 248320,
                          .packedGdnWidth = 16640,
                          .packedFullWidth = 14336,
                          .convolutionDimension = 10240,
                          .gdnKeyHeads = 16,
                          .gdnValueHeads = 48,
                          .gdnHeadDimension = 128,
                          .attentionWidth = 6144,
                          .attentionQueryHeads = 24,
                          .attentionKvHeads = 4,
                          .attentionHeadDimension = 256,
                          .rotaryPairs = SPLASH_TARGET_ROPE_PAIRS,
                          .rotaryTheta = 10'000'000.0F,
                          .fullAttentionPeriod = 4,
                          .maskToken = 248070,
                          .stopTokens = {248044, 248046},
                          .ffnKind = QwenFfnKind::Dense,
                          .intermediateSize = 17408}} {}
  bool operator==(const Qwen3_8Layout &) const = default;
};

struct Qwen3_8LayerWeights final {
  ops::NormWeights inputNorm;
  QwenMixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

using Qwen3_8Weights = QwenTargetWeights<Qwen3_8Layout, Qwen3_8LayerWeights>;

[[nodiscard]] Qwen3_8Weights
loadQwen3_8Weights(metal::MetalBackend &backend, Qwen3_8Layout layout,
                   const QwenTargetFiles &files);

} // namespace splash::model
