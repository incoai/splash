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

// The Qwen3.5 MoE architecture, whose families are sizes of it: Qwen3.6-35B-A3B
// and Qwen3.5-122B-A10B (kFamilies).
struct Qwen3_6MoeLayout final : QwenHybridLayout {
  // Its config's text model type and its GGUFs' architecture; a safetensors
  // checkpoint of it loads too.
  static constexpr std::string_view modelType = "qwen3_5_moe_text";
  static constexpr std::string_view ggufArchitecture = "qwen35moe";
  static constexpr bool loadsSafetensors = true;

  // Qwen3.6-35B-A3B's, its first family (kFamilies).
  constexpr Qwen3_6MoeLayout()
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
                          .yarnContextTokens = 1'010'000,
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
                          .expertIntermediateSize = 512},
                         "Qwen3.6-35B-A3B"} {}
  // A family of the architecture.
  constexpr explicit Qwen3_6MoeLayout(const QwenHybridLayout &family) : QwenHybridLayout(family) {}
  bool operator==(const Qwen3_6MoeLayout &) const = default;
};

// Qwen3.5-122B-A10B and its fine-tunes, such as Ornith-1.5-122B.
inline constexpr Qwen3_6MoeLayout kQwen3_5_122BLayout{
    QwenHybridLayout{{.nativeContextTokens = 262'144,
                      .yarnContextTokens = 1'010'000,
                      .layers = 48,
                      .hiddenSize = 3072,
                      .vocabularySize = 248320,
                      .packedGdnWidth = 20736,
                      .packedFullWidth = 17408,
                      .convolutionDimension = 12288,
                      .gdnKeyHeads = 16,
                      .gdnValueHeads = 64,
                      .gdnHeadDimension = 128,
                      .attentionWidth = 8192,
                      .attentionQueryHeads = 32,
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
                      .expertIntermediateSize = 1024},
                     "Qwen3.5-122B-A10B"}};

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
