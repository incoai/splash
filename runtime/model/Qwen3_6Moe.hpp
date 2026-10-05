#pragma once

#include "DFlashDraft.hpp"
#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/RoPE.h"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <string_view>

namespace splash::model {

struct Qwen3_6MoeLayout final : QwenHybridLayout<8> {
  // The family's name, as the installer pairs its draft (install/families.py).
  static constexpr std::string_view family = "Qwen3.6-35B-A3B";
  static constexpr std::string_view layerMagic = "MDFM0001";
  static constexpr std::string_view headMagic = "MDFM0002";

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
                          .expertIntermediateSize = 512},
                         /* hiddenCaptureLayers */ {1, 6, 11, 16, 22, 27, 32, 37}} {}
  bool operator==(const Qwen3_6MoeLayout &) const = default;
};

// The DFlash2 draft of Qwen3.6-35B-A3B.
inline constexpr DFlashDraftLayout kQwen3_6MoeDraftLayout{.layers = 6,
                                                          .hiddenSize = 2048,
                                                          .vocabularySize = 248320,
                                                          .dynamicSize = draft_dynamic_width(2048),
                                                          .qkvSize = SPLASH_DRAFT_QKV_WIDTH,
                                                          .attentionSize = SPLASH_DRAFT_ATTENTION_WIDTH,
                                                          .intermediateSize = 6144,
                                                          .attentionHeadDimension = SPLASH_DRAFT_HEAD_DIMENSION,
                                                          .rotaryTheta = 10'000'000.0F,
                                                          .targetHiddenSize = 16384,
                                                          .selectorRank = SPLASH_DRAFT_SELECTOR_RANK,
                                                          .kvHeads = SPLASH_DRAFT_KV_HEADS};

// The vision tower of Qwen3.6-35B-A3B, projecting into its hidden width.
inline constexpr ops::VisionLayout kQwen3_6MoeVisionLayout{.outputHiddenSize = Qwen3_6MoeLayout{}.hiddenSize};

struct Qwen3_6MoeLayerWeights final {
  ops::NormWeights inputNorm;
  QwenMixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::MoeWeights ffn;
};

using Qwen3_6MoeWeights = QwenTargetWeights<Qwen3_6MoeLayout, Qwen3_6MoeLayerWeights>;

[[nodiscard]] Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend, Qwen3_6MoeLayout layout,
                      const QwenTargetFiles<Qwen3_6MoeLayout> &files);

} // namespace splash::model
