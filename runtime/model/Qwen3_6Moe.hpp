#pragma once

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <string_view>

namespace splash::model {

struct Qwen3_6MoeLayout final : QwenHybridLayout<8> {
  static constexpr std::string_view layerMagic = "MDFM0001";
  static constexpr std::string_view headMagic = "MDFM0002";
  static constexpr QwenFfnKind ffnKind = QwenFfnKind::SparseMoe;

  uint32_t experts = 256;
  uint32_t expertsPerToken = 8;
  uint32_t expertIntermediateSize = 512;

  constexpr Qwen3_6MoeLayout()
      : QwenHybridLayout{.layers = 40,
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
                         .rotaryPairs = 32,
                         .rotaryTheta = 10'000'000.0F,
                         .fullAttentionPeriod = 4,
                         .maskToken = 248077,
                         .stopTokens = {248044, 248046},
                         .hiddenCaptureLayers = {1, 6, 11, 16, 22, 27, 32, 37}} {}
  bool operator==(const Qwen3_6MoeLayout &) const = default;
};

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
