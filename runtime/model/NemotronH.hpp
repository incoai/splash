#pragma once

// Nemotron 3.5 Bolt and Super (GGUF architecture nemotron_h_moe, model type
// nemotron_h_moe_text): 88 layers of one block each, behind its own RMS norm,
// in the same order.
//  - Mamba-2 (40): in projection to z | x B C | dt, a 4-tap causal
//    convolution with bias over x B C and SiLU, the selective scan of 128
//    heads of 64 rows over a 128-column state shared by 8 groups, a D skip,
//    and a norm of 8 groups gated by SiLU(z) before the out projection.
//  - NoPE attention (8): grouped-query attention without rotary positions,
//    q/k norms or an output gate.
//  - Latent MoE (40): the router (sigmoid, a selection bias, top-9 normalized
//    and scaled by 5) reads the hidden row; the routed experts (up and down,
//    ReLU squared, no gate) run in a 1024-wide latent between two dense
//    projections, and a shared expert of its own width (ReLU squared) reads
//    the hidden row. Each of Bolt's MoE layers has its own expert count and
//    width; Super's all route top-22 of 512 experts 2688 wide.
// The layer numbering is the GGUF's: a draft's target_layer_ids name these
// blocks. Its weights load from a GGUF only.

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/RoPE.h"
#include "ops/Linear.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Mamba2.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <variant>

namespace splash::model {

enum class NemotronBlock : uint8_t { Mamba, Attention, Moe };

// Bolt's blocks: M Mamba-2, A attention, E MoE.
inline constexpr std::string_view kBoltBlocks =
    "MEMEMEMAEMEMEMEMAEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMAEMEMEMEME";
inline constexpr uint32_t kBoltLayers = 88;
static_assert(kBoltBlocks.size() == kBoltLayers);

// Each MoE layer's routed experts and their width (0 elsewhere), as the
// GGUF's router shapes and expert_feed_forward_length state them.
inline constexpr std::array<uint16_t, kBoltLayers> kBoltExperts = {
    0,   240, 0, 240, 0, 240, 0, 0,   304, 0, 240, 0, 240, 0, 240, 0, 0,   304, 0, 304, 0, 432,
    0,   432, 0, 0,   240, 0, 432, 0, 304, 0, 304, 0, 432, 0, 0,   368, 0, 368, 0, 368, 0, 432,
    0,   368, 0, 0,   496, 0, 432, 0, 240, 0, 368, 0, 368, 0, 0,   304, 0, 240, 0, 240, 0, 240,
    0,   240, 0, 0,   368, 0, 368, 0, 432, 0, 432, 0, 0,   496, 0, 432, 0, 432, 0, 496, 0, 368};
inline constexpr std::array<uint16_t, kBoltLayers> kBoltExpertWidths = {
    0,    2048, 0, 1536, 0, 2048, 0, 0,    2304, 0, 2304, 0, 2048, 0, 2048, 0, 0,    2304, 0, 2048, 0, 2560,
    0,    2304, 0, 0,    2048, 0, 2304, 0, 2560, 0, 2304, 0, 2304, 0, 0,    2304, 0, 2048, 0, 2304, 0, 2560,
    0,    2560, 0, 0,    2560, 0, 2560, 0, 2304, 0, 2304, 0, 2560, 0, 0,    2048, 0, 2304, 0, 2304, 0, 2048,
    0,    2048, 0, 0,    2048, 0, 2304, 0, 2304, 0, 2560, 0, 0,    2560, 0, 2560, 0, 2560, 0, 2560, 0, 2560};

constexpr NemotronBlock boltBlock(uint32_t layer) noexcept {
  return kBoltBlocks[layer] == 'M' ? NemotronBlock::Mamba
         : kBoltBlocks[layer] == 'A' ? NemotronBlock::Attention
                                     : NemotronBlock::Moe;
}

constexpr std::array<uint64_t, 2> boltMask(char block) noexcept {
  std::array<uint64_t, 2> mask{};
  for (uint32_t layer = 0; layer < kBoltLayers; ++layer)
    if (kBoltBlocks[layer] == block) mask[layer / 64] |= uint64_t{1} << (layer % 64);
  return mask;
}

struct NemotronHLayout final : QwenHybridLayout {
  static constexpr std::string_view modelType = "nemotron_h_moe_text";
  static constexpr std::string_view ggufArchitecture = "nemotron_h_moe";
  static constexpr bool loadsSafetensors = false;

  // Its Mamba-2 state, whose cached states hold it as it runs: StateCodec's
  // codes are GDN layers'.
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    GdnStateLayout layout = QwenHybridLayout::gdnStateLayout();
    layout.cachedCodes = false;
    return layout;
  }

  // Mamba-2: the in projection's rows (z | x B C | dt) and the inner width.
  uint32_t mambaInnerSize = 8192;
  uint32_t mambaInProjectionRows = 18560;
  // The MoE: the latent the routed experts run in, the shared expert's width
  // and the routed weights' scale; the most experts and the widest of any
  // layer, which size the plans' scratch (QwenTargetDimensions::experts and
  // expertIntermediateSize hold them).
  uint32_t moeLatentSize = 1024;
  uint32_t sharedIntermediateSize = 5376;
  float routedScale = 5.0F;
  // The RMS norms' epsilon (the Qwen families' SPLASH_RMS_EPSILON is 1e-6),
  // the double the norm kernels compile.
  double rmsEpsilon = SPLASH_RMS_EPSILON_E5;

  constexpr NemotronHLayout()
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
                          .layers = kBoltLayers,
                          .hiddenSize = 4096,
                          .vocabularySize = 131072,
                          // q | k | v: 4096 + 2 * 2 heads * 128.
                          .packedFullWidth = 4608,
                          // x | B | C: 8192 + 2 * 8 groups * 128.
                          .convolutionDimension = 10240,
                          .gdnKeyHeads = 8,
                          .gdnValueHeads = 128,
                          .gdnHeadDimension = 128,
                          .attentionWidth = 4096,
                          .attentionQueryHeads = 32,
                          .attentionKvHeads = 2,
                          .attentionHeadDimension = 128,
                          .attentionGate = false,
                          // NoPE (unrotatedLayers): the runtime builds RoPE
                          // tables of these for every step, which no
                          // Nemotron-H kernel reads.
                          .rotaryPairs = SPLASH_TARGET_ROPE_PAIRS,
                          .rotaryTheta = 10'000'000.0F,
                          // Every attention layer.
                          .unrotatedLayers = 0xFF,
                          // <SPECIAL_990>, the LiLiCorr draft's mask token.
                          .maskToken = 990,
                          // <|im_end|>, </s>.
                          .stopTokens = {11, 2},
                          .ffnKind = QwenFfnKind::SparseMoe,
                          .experts = 496,
                          .expertsPerToken = 9,
                          .expertIntermediateSize = 2560,
                          .attentionMask = boltMask('A'),
                          .recurrentMask = boltMask('M'),
                          .recurrentHeadDimension = 64},
                         "Nemotron-3.5-Bolt"} {}
  // A family of the architecture.
  constexpr explicit NemotronHLayout(const QwenHybridLayout &family) : QwenHybridLayout(family) {}
  bool operator==(const NemotronHLayout &) const = default;

  // Its MoE blocks' shape, of the most experts and the widest of any layer:
  // its expert store's, which each layer's narrows to its own (this hides
  // QwenHybridLayout's, of the Qwen families' MoE).
  [[nodiscard]] constexpr ops::MoeShape moeBlockShape() const noexcept {
    return {.hiddenSize = moeLatentSize,
            .experts = experts,
            .expertsPerToken = expertsPerToken,
            .expertIntermediateSize = expertIntermediateSize,
            .router = ops::MoeRouter::SigmoidBiased,
            .sharedSlices = 0,
            .sharedGate = false,
            .activation = ops::MoeActivation::ReluSquared,
            .routerInputSize = hiddenSize,
            .routedScale = routedScale};
  }
  [[nodiscard]] static constexpr NemotronBlock block(uint32_t layer) noexcept { return boltBlock(layer); }
  // A MoE layer's routed experts and their width, 0 elsewhere: Bolt's own
  // (kBoltExperts, kBoltExpertWidths), told apart by the experts a token
  // takes; any other family's every MoE layer has the target's.
  [[nodiscard]] static constexpr bool boltTables(const QwenTargetDimensions &target) noexcept {
    return target.expertsPerToken == 9;
  }
  [[nodiscard]] static constexpr uint32_t layerExperts(const QwenTargetDimensions &target, uint32_t layer) noexcept {
    if (block(layer) != NemotronBlock::Moe) return 0;
    return boltTables(target) ? kBoltExperts[layer] : target.experts;
  }
  [[nodiscard]] static constexpr uint32_t layerExpertWidth(const QwenTargetDimensions &target,
                                                           uint32_t layer) noexcept {
    if (block(layer) != NemotronBlock::Moe) return 0;
    return boltTables(target) ? kBoltExpertWidths[layer] : target.expertIntermediateSize;
  }
};

// Nemotron 3.5 Super: Bolt's blocks and widths but for its experts and a
// 1M-token window, served to kMaximumServedContextTokens.
inline constexpr NemotronHLayout kNemotronSuperLayout{[] {
  QwenHybridLayout family = NemotronHLayout{};
  family.nativeContextTokens = 1'048'576;
  family.experts = 512;
  family.expertsPerToken = 22;
  family.expertIntermediateSize = 2688;
  family.family = "Nemotron-3.5-Super";
  return family;
}()};
static_assert(!NemotronHLayout::boltTables(kNemotronSuperLayout) && kNemotronSuperLayout.moeBlockShape().valid());
static_assert(kNemotronSuperLayout.maximumContextTokens() == kMaximumServedContextTokens);

static_assert(NemotronHLayout::boltTables(NemotronHLayout{}) && NemotronHLayout{}.moeBlockShape().valid());
static_assert(NemotronHLayout{}.attentionLayerCount() == 8 && NemotronHLayout{}.recurrentLayerCount() == 40);
static_assert(NemotronHLayout{}.gdnStateLayout().recurrentRows == 64 &&
                  NemotronHLayout{}.gdnStateLayout().recurrentColumns == 128,
              "a Mamba-2 head's state: 64 rows of the 128-column state");
static_assert(
    [] {
      for (uint32_t layer = 0; layer < kBoltLayers; ++layer)
        if ((boltBlock(layer) == NemotronBlock::Moe) != (kBoltExperts[layer] != 0) ||
            (kBoltExperts[layer] != 0) != (kBoltExpertWidths[layer] != 0) ||
            kBoltExperts[layer] > NemotronHLayout{}.experts ||
            kBoltExpertWidths[layer] > NemotronHLayout{}.expertIntermediateSize || kBoltExpertWidths[layer] % 64)
          return false;
      return true;
    }(),
    "every MoE layer, and only those, has experts of a whole 64-column width within the plans'");

// The Mamba-2 mixer of a target of `dimensions` (ops::Mamba2): its groups,
// heads of recurrentHeadDimension rows and gdnHeadDimension state columns.
[[nodiscard]] constexpr ops::Mamba2Shape mamba2Shape(const QwenTargetDimensions &dimensions) noexcept {
  return {dimensions.gdnKeyHeads, dimensions.gdnValueHeads, dimensions.recurrentHeadDimension,
          dimensions.gdnHeadDimension};
}

// A Mamba-2 block's weights, in file order after its norm: the packed in
// projection (z | x B C | dt), the mixer's F32 tensors and the out
// projection.
struct NemotronMambaWeights final {
  ops::Projection in;
  ops::Mamba2Weights mixer;
  ops::Projection out;
};

// A latent MoE block (ops::MoE of the layer's shape): the latent's down
// projection, the routed experts behind the router of the hidden rows and
// its selection bias, the latent's up projection and the shared expert's up
// and down, ReLU^2 between them.
struct NemotronMoeWeights final {
  ops::MoeShape shape;
  ops::BlockMoeWeights experts;
  ops::Projection latentDown;
  ops::Projection latentUp;
  ops::Projection sharedUp;
  ops::Projection sharedDown;
};

// Its attention blocks are the Qwen families' (q | k | v fused in that column
// order, no q/k norms, no gate).
struct NemotronHLayerWeights final {
  ops::NormWeights norm;
  std::variant<NemotronMambaWeights, QwenAttentionWeights, NemotronMoeWeights> block;
};

using NemotronHWeights = QwenTargetWeights<NemotronHLayout, NemotronHLayerWeights>;

struct SparseRowStoreOptions;

// A Mamba-2 layer's decode tape for a target of `dimensions`.
[[nodiscard]] inline uint64_t mamba2TapeLayerBytes(const QwenTargetDimensions &dimensions) {
  return ops::mamba2TapeLayerBytes(mamba2Shape(dimensions));
}

// From a GGUF only (GgufTarget's NemotronHGguf plan).
[[nodiscard]] NemotronHWeights loadNemotronHWeights(metal::MetalBackend &backend, const NemotronHLayout &layout,
                                                    const QwenTargetFiles &files);
// The family's geometry: its blocks' projections and the layers its draft
// captures.
[[nodiscard]] QwenTargetGeometry qwenTargetGeometry(const NemotronHWeights &weights,
                                                    std::span<const uint32_t> captureLayers);
[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const NemotronHWeights &weights,
                                                         const QwenTargetGeometry &geometry,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         uint32_t prefillChunks);
[[nodiscard]] SparseRowStoreOptions streamedTableRows(const NemotronHLayout &layout);
// The decode scratch its blocks need beyond the runtime's arena
// (NemotronHTarget) and its prefill scratch.
[[nodiscard]] QwenTargetScratchBytes qwenTargetScratchBytes(const NemotronHLayout &layout,
                                                            const QwenTargetGeometry &geometry,
                                                            uint32_t prefillChunks, uint64_t tableRowBytes);

} // namespace splash::model
