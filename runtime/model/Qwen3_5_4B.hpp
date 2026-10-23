#pragma once

#include "model/Qwen3_8.hpp"

namespace splash::model {

struct Qwen3_5_4BLayout final : QwenHybridLayout {
  static constexpr std::string_view family = "Qwen3.5-4B";
  static constexpr std::string_view modelType = "qwen3_5_text";
  static constexpr std::string_view ggufArchitecture = "qwen35";
  static constexpr bool loadsSafetensors = true;

  constexpr Qwen3_5_4BLayout()
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
                          .layers = 32,
                          .hiddenSize = 2560,
                          .vocabularySize = 248320,
                          .packedGdnWidth = 12544,
                          .packedFullWidth = 10240,
                          .convolutionDimension = 8192,
                          .gdnKeyHeads = 16,
                          .gdnValueHeads = 32,
                          .gdnHeadDimension = 128,
                          .attentionWidth = 4096,
                          .attentionQueryHeads = 16,
                          .attentionKvHeads = 4,
                          .attentionHeadDimension = 256,
                          .rotaryPairs = SPLASH_TARGET_ROPE_PAIRS,
                          .rotaryTheta = 10'000'000.0F,
                          .fullAttentionPeriod = 4,
                          .maskToken = 248077,
                          .stopTokens = {248044, 248046},
                          .ffnKind = QwenFfnKind::Dense,
                          .intermediateSize = 9216,
                          .tiedEmbeddings = true}} {}
  bool operator==(const Qwen3_5_4BLayout &) const = default;
};

using Qwen3_5_4BWeights = QwenTargetWeights<Qwen3_5_4BLayout, Qwen3_8LayerWeights>;

[[nodiscard]] Qwen3_5_4BWeights
loadQwen3_5_4BWeights(metal::MetalBackend &backend, Qwen3_5_4BLayout layout,
                     const QwenTargetFiles &files);

} // namespace splash::model
