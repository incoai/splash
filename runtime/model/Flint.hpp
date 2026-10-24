#pragma once

// Flint (RecursiveSI/flint; GGUF architecture flint, model type flint): a
// pre-norm decoder of the Qwen MoE's blocks whose layers all attend, as a
// layout of the Qwen hybrid's with no GDN layers (QwenHybridLayout). What it
// changes are values of the blocks the Qwen targets share:
//  - Attention of 32 query heads over 4 KV heads of 128, with no output
//    gate, a learned sink per query head (a logit in the softmax's
//    denominator that adds no value) and, on 35 of its 49 layers, a sliding
//    window of the 513 keys that end at a row, rotated by RoPE over whole
//    heads (theta 1e4); its other 14 layers attend every key with no RoPE.
//  - Plain RMS norms (x * rsqrt(mean(x^2) + eps) * w), as a GGUF stores them.
//  - Its first two layers' FFN is a dense SwiGLU; the others' a MoE that
//    weights its top 6 of 256 experts by their sigmoid scores over the sum
//    of theirs, beside an ungated shared expert twice as wide as a routed
//    one, which runs as two slices of the routed width (ops::MoeShape).
//  - Four tokens end a turn, and no mask token is its own: a DFlash2 draft
//    chooses one of its reserved special tokens.
// Its target (model/FlintTarget.hpp) is a QwenTarget over those blocks. Its
// weights load from a GGUF only.

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <memory>
#include <string_view>
#include <variant>

namespace splash::model {

// Flint's sliding layers, a bit per layer (every layer attends): those from
// layer 2 to 47 but every fourth from layer 5 (5, 9, ..., 45), which with
// layers 0, 1 and 48 attend every key, without RoPE.
inline constexpr uint64_t kFlintSlidingLayers = [] {
  uint64_t layers = 0;
  for (uint32_t layer = 2; layer < 48; ++layer)
    if (layer % 4 != 1) layers |= uint64_t{1} << layer;
  return layers;
}();

struct FlintLayout final : QwenHybridLayout {
  // Its config's model type, whose keys its config.json holds at its top
  // level, and its GGUFs' architecture; no safetensors checkpoint of it loads.
  static constexpr std::string_view modelType = "flint";
  static constexpr bool flatConfig = true;
  static constexpr std::string_view ggufArchitecture = "flint";
  static constexpr bool loadsSafetensors = false;

  // Flint's, its first family (kFamilies). Its context is the native one:
  // its full layers have no RoPE to stretch.
  constexpr FlintLayout()
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
                          .layers = 49,
                          .hiddenSize = 2048,
                          .vocabularySize = 150144,
                          .packedFullWidth = 5120,
                          .attentionWidth = 4096,
                          .attentionQueryHeads = 32,
                          .attentionKvHeads = 4,
                          .attentionHeadDimension = 128,
                          .attentionGate = false,
                          .rotaryPairs = 64,
                          .rotaryTheta = 10'000.0F,
                          .slidingWindow = 513,
                          .slidingLayers = kFlintSlidingLayers,
                          .unrotatedLayers = ~kFlintSlidingLayers & ((uint64_t{1} << 49) - 1),
                          .fullAttentionPeriod = 1,
                          .firstDraftMaskToken = 150019,
                          .lastDraftMaskToken = 150143,
                          .stopTokens = {150001, 150003, 150005, 150018},
                          .ffnKind = QwenFfnKind::SparseMoe,
                          .intermediateSize = 6144,
                          .experts = 256,
                          .expertsPerToken = 6,
                          .expertIntermediateSize = 512,
                          .moeRouter = ops::MoeRouter::SigmoidNormalized,
                          .sharedExpertSlices = 2,
                          .sharedExpertGate = false,
                          .leadingDenseLayers = 2},
                         "Flint"} {}
  // A family of the architecture.
  constexpr explicit FlintLayout(const QwenHybridLayout &family) : QwenHybridLayout(family) {}
  bool operator==(const FlintLayout &) const = default;
};

// A leading layer's dense SwiGLU.
struct FlintDenseFfnWeights final {
  ops::Projection gate;
  ops::Projection up;
  ops::Projection down;

  [[nodiscard]] ops::SwiGluProjections projections() const noexcept { return {&gate, &up, &down}; }
};

// A layer: its attention with its sinks, the norms before it and before the
// FFN, and the FFN, dense on the leading layers and a MoE block on the
// others (QwenTargetDimensions::isMoeLayer).
struct FlintLayerWeights final {
  ops::NormWeights inputNorm;
  QwenAttentionWeights mixer;
  ops::NormWeights postAttentionNorm;
  std::variant<FlintDenseFfnWeights, ops::BlockMoeWeights> ffn;
};

using FlintWeights = QwenTargetWeights<FlintLayout, FlintLayerWeights>;

[[nodiscard]] FlintWeights loadFlintWeights(metal::MetalBackend &backend, FlintLayout layout,
                                            const QwenTargetFiles &files);

// A layer's MoE block, null for a leading dense layer's FFN.
[[nodiscard]] const ops::BlockMoeWeights *qwenMoeBlock(const FlintLayerWeights &layer) noexcept;
// The projections its dense layers dispatch (qwenTargetGeometry).
void includeFamilyProjections(QwenTargetGeometry &geometry, const FlintWeights &weights);
// Its target (model/FlintTarget.hpp), which keeps no step scratch of its own.
[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const FlintWeights &weights,
                                                         const QwenTargetGeometry &geometry,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         uint32_t prefillChunks);

} // namespace splash::model
