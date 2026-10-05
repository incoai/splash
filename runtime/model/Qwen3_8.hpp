#pragma once

#include "DFlashDraft.hpp"
#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/RoPE.h"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <string_view>

namespace splash::model {

struct Qwen3_8Layout final : QwenHybridLayout<5> {
  static constexpr std::string_view layerMagic = "MDFL0006";
  static constexpr std::string_view headMagic = "MDFL0002";

  constexpr Qwen3_8Layout()
      : QwenHybridLayout{{.layers = 64,
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
                         /* hiddenCaptureLayers */ {5, 19, 33, 47, 61}} {}
  bool operator==(const Qwen3_8Layout &) const = default;
};

// The DFlash2 draft of Qwen3.8-27B.
inline constexpr DFlashDraftLayout kQwen3_8DraftLayout{.layers = 5,
                                                       .hiddenSize = 5120,
                                                       .vocabularySize = 248320,
                                                       .dynamicSize = draft_dynamic_width(5120),
                                                       .qkvSize = SPLASH_DRAFT_QKV_WIDTH,
                                                       .attentionSize = SPLASH_DRAFT_ATTENTION_WIDTH,
                                                       .intermediateSize = 17408,
                                                       .attentionHeadDimension = SPLASH_DRAFT_HEAD_DIMENSION,
                                                       .rotaryTheta = 10'000'000.0F,
                                                       .targetHiddenSize = 25600,
                                                       .selectorRank = SPLASH_DRAFT_SELECTOR_RANK,
                                                       .kvHeads = SPLASH_DRAFT_KV_HEADS};

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
                   const QwenTargetFiles<Qwen3_8Layout> &files);

} // namespace splash::model
