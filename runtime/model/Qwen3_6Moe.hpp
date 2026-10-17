#pragma once

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/RoPE.h"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <string_view>

namespace splash::model {

struct Qwen3_6MoeLayout final : QwenHybridLayout {
  // The family's name, as the installer pairs its draft (install/families.py).
  static constexpr std::string_view family = "Qwen3.6-35B-A3B";
  // Its config's text model type (transformers') and its GGUFs' architecture
  // (llama.cpp's); a safetensors checkpoint of it loads too.
  static constexpr std::string_view modelType = "qwen3_5_moe_text";
  static constexpr std::string_view ggufArchitecture = "qwen35moe";
  static constexpr bool loadsSafetensors = true;

  constexpr Qwen3_6MoeLayout()
      : QwenHybridLayout{{.maximumContextTokens = 262'144,
                          .layers = 40,
                          .hiddenSize = 2048,
                          .vocabularySize = 248320,
                          .packedGdnWidth = 12544,
                          .packedFullWidth = 9216,
                          .convolutionDimension = 8192,
                          .gdnKeyHeads = 16,
                          .gdnValueHeads = 32,
                          .gdnHeadDimension = 128,
                          .attentionWidth = 4096,
                          .attentionQueryHeads = 16,
                          .attentionKvHeads = 2,
                          .attentionHeadDimension = 256,
                          .rotaryPairs = SPLASH_TARGET_ROPE_PAIRS,
                          .rotaryTheta = 10'000'000.0F,
                          .fullAttentionPeriod = 4,
                          .maskToken = 248077,
                          .stopTokens = {248044, 248046},
                          .ffnKind = QwenFfnKind::SparseMoe,
                          .experts = 256,
                          .expertsPerToken = 8,
                          .expertIntermediateSize = 512}} {}
  bool operator==(const Qwen3_6MoeLayout &) const = default;
};

struct Qwen3_6MoeLayerWeights final {
  ops::NormWeights inputNorm;
  QwenMixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::BlockMoeWeights ffn;
};

using Qwen3_6MoeWeights = QwenTargetWeights<Qwen3_6MoeLayout, Qwen3_6MoeLayerWeights>;

[[nodiscard]] Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend, Qwen3_6MoeLayout layout,
                      const QwenTargetFiles &files);

} // namespace splash::model
