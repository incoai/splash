#pragma once

// Ripple (ricursiveAI; GGUF architecture ricursive, model type ricursive): a
// pre-norm decoder of 80 blocks of four RMSNorms, one before and one after
// each mixer and each FFN. What it changes beside the Qwen hybrid's blocks:
//  - Attention of 80 query heads over 8 KV heads of 128, with the Qwen3.5
//    output gate (a separate attn_gate projection, which the load fuses into
//    the packed q as the Qwen3.5's own fused gate rows are) and per-head
//    q/k norms; 64 of its layers slide over windows of 1024 keys rotated by
//    RoPE (theta 1e4) and the other 16 attend every key without RoPE.
//  - A value residual: each block's v input is
//    sigmoid(lambda) * attn_norm(x) + (1 - sigmoid(lambda)) * scaled
//    embeddings, the scaled token embedding rows the block mixes back in by
//    its learned lambda (attn_vres_lambda, one scalar a block).
//  - A dense SwiGLU FFN of 40960 whose gate and up activations clamp at 20
//    (ffn_clamp_threshold), and a norm after it (post_ffw_norm).
//  - Token embeddings the square root of the hidden size scales
//    (embed_scale), which the value residual's mix shares.
// Its weights load from a GGUF only. Its target (model/RippleTarget) and
// weights load are to land; loading refuses until then.

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace splash::model {

// Ripple's sliding layers, a bit per layer: four of five slide, the fifth
// (every fifth, from layer 4) attends every key without RoPE.
inline constexpr kv::LayerMask kRippleSlidingLayers = [] {
  kv::LayerMask layers;
  for (uint32_t layer = 0; layer < 80; ++layer)
    if (layer % 5 != 4) layers.set(layer, true);
  return layers;
}();
inline constexpr kv::LayerMask kRippleUnrotatedLayers = [] {
  kv::LayerMask layers;
  for (uint32_t layer = 0; layer < 80; ++layer)
    if (!kRippleSlidingLayers.test(layer)) layers.set(layer, true);
  return layers;
}();

struct RippleLayout final : QwenHybridLayout {
  static constexpr std::string_view modelType = "ricursive";
  static constexpr bool flatConfig = true;
  static constexpr std::string_view ggufArchitecture = "ricursive";
  static constexpr bool loadsSafetensors = false;

  // Ripple-65B's, its first family (kFamilies). Its full layers have no RoPE
  // to stretch, so the family serves its native window and no YaRN, and a
  // config's max_position_embeddings states the context the model claims,
  // which the family need not serve (configNativeContextTokens).
  constexpr RippleLayout()
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
                          .yarnContextTokens = 0,
                          .layers = 80,
                          .hiddenSize = 5120,
                          .vocabularySize = 100'352,
                          .packedFullWidth = 22'528,
                          .attentionWidth = 10'240,
                          .attentionQueryHeads = 80,
                          .attentionKvHeads = 8,
                          .attentionHeadDimension = 128,
                          .rotaryPairs = 64,
                          .rotaryTheta = 10'000.0F,
                          .slidingWindow = 1024,
                          .slidingLayers = kRippleSlidingLayers,
                          .unrotatedLayers = kRippleUnrotatedLayers,
                          .fullAttentionPeriod = 1,
                          .maskToken = 100'278,
                          .stopTokens = {100'265, 100'257},
                          .ffnKind = QwenFfnKind::Dense,
                          .intermediateSize = 40'960,
                          .configNativeContextTokens = 2'097'152,
                          .embeddingScale = 71.5541763305664F,
                          .swigluClamp = 20.0F,
                          .valueResidual = true,
                          .postFfnNorm = true},
                         "Ripple-65B"} {}
  // A family of the architecture.
  constexpr explicit RippleLayout(const QwenHybridLayout &family) : QwenHybridLayout(family) {}
  bool operator==(const RippleLayout &) const = default;
};

// A layer's dense SwiGLU.
struct RippleDenseFfnWeights final {
  ops::Projection gate;
  ops::Projection up;
  ops::Projection down;

  [[nodiscard]] ops::SwiGluProjections projections() const noexcept { return {&gate, &up, &down}; }
};

// A layer's weights: its four norms, its mixer (whose packed q holds the
// gate rows, as the Qwen3.5's does) with its value-residual lambda, and its
// dense SwiGLU, whose activations clamp at the family's swigluClamp.
struct RippleLayerWeights final {
  ops::NormWeights inputNorm;
  QwenMixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  RippleDenseFfnWeights ffn;
  ops::NormWeights postFfnNorm;
  float valueResidualLambda = 0.0F;
};

using RippleWeights = QwenTargetWeights<RippleLayout, RippleLayerWeights>;

[[nodiscard]] RippleWeights loadRippleWeights(metal::MetalBackend &backend, RippleLayout layout,
                                              const QwenTargetFiles &files);
// Its layers' dense FFNs, which the geometry takes whole (the shared MoE
// hook sees none), and its target (Ripple.cpp, whose making is to land).
void includeFamilyProjections(QwenTargetGeometry &geometry, const RippleWeights &weights);
[[nodiscard]] const ops::BlockMoeWeights *qwenMoeBlock(const RippleLayerWeights &layer) noexcept;
[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const RippleWeights &weights,
                                                         const QwenTargetGeometry &geometry,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         uint32_t prefillChunks);

} // namespace splash::model
