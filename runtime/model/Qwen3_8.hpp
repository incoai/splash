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

// The Qwen3.5 dense architecture, whose families are sizes of it: Qwen3.8-27B
// and Qwen3.5-4B (kFamilies).
struct Qwen3_8Layout final : QwenHybridLayout {
  // Its config's text model type and its GGUFs' architecture; a safetensors
  // checkpoint of it loads too.
  static constexpr std::string_view modelType = "qwen3_5_text";
  static constexpr std::string_view ggufArchitecture = "qwen35";
  static constexpr bool loadsSafetensors = true;

  // Qwen3.8-27B's, its first family (kFamilies).
  constexpr Qwen3_8Layout()
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
                          .yarnContextTokens = 1'010'000,
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
                          .intermediateSize = 17408},
                         "Qwen3.8-27B"} {}
  // A family of the architecture.
  constexpr explicit Qwen3_8Layout(const QwenHybridLayout &family) : QwenHybridLayout(family) {}
  bool operator==(const Qwen3_8Layout &) const = default;
};

// Qwen3.5-4B, whose head is its token table.
inline constexpr Qwen3_8Layout kQwen3_5_4BLayout{
    QwenHybridLayout{{.nativeContextTokens = 262'144,
                      .yarnContextTokens = 1'010'000,
                      .layers = 32,
                      .hiddenSize = 2560,
                      .vocabularySize = 248320,
                      .tiedEmbeddings = true,
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
                      .intermediateSize = 9216},
                     "Qwen3.5-4B"}};

struct Qwen3_8LayerWeights final {
  ops::NormWeights inputNorm;
  QwenMixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

// Its every layer's FFN is dense (QwenTarget.hpp's qwenMoeBlock).
[[nodiscard]] const ops::BlockMoeWeights *qwenMoeBlock(const Qwen3_8LayerWeights &layer) noexcept;
[[nodiscard]] std::optional<ops::SwiGluProjections> denseFfn(const Qwen3_8LayerWeights &layer) noexcept;

using Qwen3_8Weights = QwenTargetWeights<Qwen3_8Layout, Qwen3_8LayerWeights>;

[[nodiscard]] Qwen3_8Weights
loadQwen3_8Weights(metal::MetalBackend &backend, Qwen3_8Layout layout,
                   const QwenTargetFiles &files);

} // namespace splash::model
