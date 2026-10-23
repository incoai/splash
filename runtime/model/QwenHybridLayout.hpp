#pragma once

#include "model/RopeScaling.hpp"
#include "model/StateLayout.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedKv.hpp"
#include "ops/StopTokens.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace splash::model {

enum class QwenFfnKind : uint8_t { Dense, SparseMoe };

// The dimensions and tokens of a Qwen hybrid target: GDN layers, every
// fullAttentionPeriod-th layer an attention layer instead (every layer with
// a period of 1, a target without GDN), each followed by a dense FFN or a
// sparse MoE. A family's layout (QwenHybridLayout) sets them, and every view
// of a target reads them from there: the target loaders, the GGUF planner
// and the runtime's QwenTargetGeometry.
struct QwenTargetDimensions {
  // The native window, the context max_position_embeddings states, and the
  // most context YaRN stretches it to (servedFor), as the family's model
  // card validates it: zero for a family YaRN does not serve. Both fit the
  // runtime's KV ceiling (kv::kMaximumLogicalTokens).
  uint32_t nativeContextTokens = 0;
  uint32_t yarnContextTokens = 0;
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedFullWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  // Whether the GDN output gate is sigmoid(z) (Qwen3.8-Flash-Next) rather
  // than silu(z).
  bool gdnSigmoidGate = false;
  uint32_t attentionWidth = 0;
  uint32_t attentionQueryHeads = 0;
  uint32_t attentionKvHeads = 0;
  uint32_t attentionHeadDimension = 0;
  // Whether the attention's q projection holds each query head's output gate
  // after its queries, by whose sigmoid the attention's output rows multiply
  // (Qwen3.5's attn_output_gate); else it holds the queries alone.
  bool attentionGate = true;
  // The rotated dimension pairs of each attention head and their RoPE base.
  uint32_t rotaryPairs = 0;
  float rotaryTheta = 0.0F;
  // How the attention layers attend beyond reading every key up to their row
  // through the RoPE above, each a bit per attention layer, bit a for the
  // a-th (kv::Layout::windowedLayers), zero for none: those of slidingLayers
  // read the slidingWindow keys that end at their row, its own included (a
  // config's sliding_window), and those of unrotatedLayers take no RoPE
  // (NoPE). The attention kernels compile each such variant
  // (ops::AttentionVariant, QwenTargetGeometry::attentionVariant).
  uint32_t slidingWindow = 0;
  uint64_t slidingLayers = 0;
  uint64_t unrotatedLayers = 0;
  uint32_t fullAttentionPeriod = 0;
  // The token whose row a DFlash2 draft's mask rows read
  // (dflash_config.mask_token_id). A family whose vocabulary keeps none for
  // it (Flint's) names instead the tokens its drafts choose theirs among,
  // its reserved special tokens firstDraftMaskToken to lastDraftMaskToken:
  // maskToken is then its draft's (ModelDescriptor.mm describeSourceModel),
  // zero before a draft names one.
  uint32_t maskToken = 0;
  uint32_t firstDraftMaskToken = 0;
  uint32_t lastDraftMaskToken = 0;
  ops::StopTokens stopTokens{};
  QwenFfnKind ffnKind = QwenFfnKind::Dense;
  // The dense FFN's width.
  uint32_t intermediateSize = 0;
  // The sparse MoE's routed experts, those each token takes, and the width of
  // each; how its router weights them, and its shared expert: as wide as
  // sharedExpertSlices routed experts and weighted by the sigmoid of the
  // block's scalar gate or, without sharedExpertGate, by 1 (ops::MoeShape).
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  ops::MoeRouter moeRouter = ops::MoeRouter::Softmax;
  uint32_t sharedExpertSlices = 1;
  bool sharedExpertGate = true;
  // The first layers of a sparse MoE target, which run a dense FFN of
  // intermediateSize instead (a config's mlp_only_layers); its MoE blocks
  // are those of the layers after them, in layer order.
  uint32_t leadingDenseLayers = 0;
  // Qwen3.8-Flash-Next's additions (model/Qwen4Exp.hpp), which its family
  // takes together and the Qwen3.5 families leave zero: hyper-connected
  // streams (hyperConnected), the residual's streams and the mixes' low rank;
  // a QSA indexer (hasIndexer), its heads of indexerHeadDimension values, past
  // indexerTopBlocks blocks of indexerBlockTokens tokens of which a row
  // attends sparsely; and a PLE n-gram embedding (hasPle), its layer, n-gram
  // size, heads per n-gram, head width and convolution taps. Shared code
  // reads them only where zero means none: the KV pages' index region, a
  // state cell's PLE history and QSA pending keys, and the rows a mixer adds
  // to (kvLayout, gdnStateLayout, QwenTargetGeometry::mixersAddToHidden).
  uint32_t hyperConnections = 0;
  uint32_t hyperRank = 0;
  uint32_t indexerHeads = 0;
  uint32_t indexerHeadDimension = 0;
  uint32_t indexerTopBlocks = 0;
  uint32_t indexerBlockTokens = 0;
  uint32_t pleLayer = 0;
  uint32_t pleNgram = 0;
  uint32_t pleHeadsPerNgram = 0;
  uint32_t pleHeadDimension = 0;
  uint32_t pleConvolutionTaps = 0;

  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    return fullAttentionPeriod && (layer + 1) % fullAttentionPeriod == 0;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    return fullAttentionPeriod ? layers / fullAttentionPeriod : 0;
  }
  [[nodiscard]] constexpr uint32_t gdnLayerCount() const noexcept { return layers - attentionLayerCount(); }
  // Whether its drafts choose their mask token (firstDraftMaskToken).
  [[nodiscard]] constexpr bool draftChoosesMaskToken() const noexcept { return lastDraftMaskToken != 0; }
  // Whether layer `layer`'s FFN is a MoE block, and the block's index among
  // the target's (its ExpertStore block).
  [[nodiscard]] constexpr bool isMoeLayer(uint32_t layer) const noexcept {
    return ffnKind == QwenFfnKind::SparseMoe && layer >= leadingDenseLayers;
  }
  [[nodiscard]] constexpr uint32_t moeBlock(uint32_t layer) const noexcept { return layer - leadingDenseLayers; }
  // The shape of the MoE blocks but for the formats of their expert weights,
  // which the loaded weights give (QwenTargetGeometry::moeShape).
  [[nodiscard]] constexpr ops::MoeShape moeBlockShape() const noexcept {
    return {.hiddenSize = hiddenSize,
            .experts = experts,
            .expertsPerToken = expertsPerToken,
            .expertIntermediateSize = expertIntermediateSize,
            .router = moeRouter,
            .sharedSlices = sharedExpertSlices,
            .sharedGate = sharedExpertGate};
  }
  [[nodiscard]] constexpr uint32_t actualGdnWidth() const noexcept {
    return convolutionDimension + attentionWidth + 2 * gdnValueHeads;
  }
  // The window the target serves with `scaling`: the native one, or YaRN's,
  // the native one times its factor, rounded to the nearest token so that a
  // factor taken as a context over the native window (servedFor) serves that
  // context.
  [[nodiscard]] uint32_t servedContextTokens(const RopeScaling &scaling) const noexcept {
    if (!scaling.yarn())
      return nativeContextTokens;
    return static_cast<uint32_t>(std::lround(nativeContextTokens * scaling.factor));
  }
  // Whether the residual is hyperConnections streams, which a mix joins
  // before each mixer, each FFN and the logits in place of a norm.
  [[nodiscard]] constexpr bool hyperConnected() const noexcept { return hyperConnections != 0; }
  // Whether each full-attention layer has a QSA indexer.
  [[nodiscard]] constexpr bool hasIndexer() const noexcept { return indexerHeads != 0; }
  // Whether a PLE n-gram embedding feeds layer pleLayer from a table of its
  // own.
  [[nodiscard]] constexpr bool hasPle() const noexcept { return pleNgram != 0; }
  [[nodiscard]] constexpr uint32_t streamWidth() const noexcept { return hyperConnections * hiddenSize; }
  [[nodiscard]] constexpr uint32_t pleHeads() const noexcept {
    return pleNgram ? (pleNgram - 1) * pleHeadsPerNgram : 0;
  }
  [[nodiscard]] constexpr uint32_t pleWidth() const noexcept { return pleHeads() * pleHeadDimension; }
  // The PLE convolution's history: (taps - 1) * dilation rows, the dilation
  // being the n-gram size.
  [[nodiscard]] constexpr uint32_t pleHistory() const noexcept {
    return pleConvolutionTaps ? (pleConvolutionTaps - 1) * pleNgram : 0;
  }
  // Its bytes in a state cell: bf16 rows of the streams.
  [[nodiscard]] constexpr uint64_t pleHistoryBytes() const noexcept {
    return uint64_t{pleHistory()} * streamWidth() * GdnStateLayout::bfloat16Bytes;
  }
  // The tokens a QSA row attends to sparsely past (the indexer's top_k).
  [[nodiscard]] constexpr uint32_t indexerTokens() const noexcept { return indexerTopBlocks * indexerBlockTokens; }
  // The QSA indexer's pending keys in a state cell, after the PLE history:
  // per attention layer the raw keys (bf16) of the tokens of the block a
  // sequence has not completed, all but its last, which no page holds
  // (kv::Layout::indexDimension).
  [[nodiscard]] constexpr uint64_t indexerPendingBytes() const noexcept {
    return indexerBlockTokens ? uint64_t{attentionLayerCount()} * (indexerBlockTokens - 1) * indexerHeadDimension *
                                    GdnStateLayout::bfloat16Bytes
                              : 0;
  }
  bool operator==(const QwenTargetDimensions &) const = default;
};

// A family's layout: its name, its dimensions and the cache layouts they
// make. A layout type is an architecture, whose code its families share; each
// family is a value of it (model::kFamilies) that sets every value its FFN
// uses. The type states its config's model type, the architecture of its GGUFs
// and whether a safetensors checkpoint of it loads (modelType,
// ggufArchitecture, loadsSafetensors), and, where its config.json holds the
// text model's keys at its top level instead of in a text_config (a
// multimodal Qwen checkpoint's), flatConfig.
struct QwenHybridLayout : QwenTargetDimensions {
  static constexpr bool flatConfig = false;

  // The family's name, as the installer pairs its draft (install/families.py).
  std::string_view family;

  // The KV pages, which also hold a QSA indexer's pooled keys; a QSA target's
  // sparse attention gathers blocks of tokens, so its values are token-major.
  // A sliding layer reads the slidingWindow - 1 keys before its row's.
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    return {.attentionLayers = attentionLayerCount(),
            .kvHeads = attentionKvHeads,
            .headDimension = attentionHeadDimension,
            .indexDimension = indexerHeadDimension,
            .valuesTokenMajor = hasIndexer(),
            .windowTokens = slidingWindow ? slidingWindow - 1 : 0,
            .windowedLayers = slidingLayers};
  }
  // The GDN layers' state and, after it in a cell, a PLE convolution's
  // history and the QSA indexer's pending keys.
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {.layers = gdnLayerCount(),
            .convolutionHistory = gdnLayerCount() ? kGdnConvolutionTaps - 1 : 0,
            .convolutionChannels = convolutionDimension,
            .recurrentGroups = gdnValueHeads,
            .recurrentRows = gdnHeadDimension,
            .recurrentColumns = gdnHeadDimension,
            .auxiliaryBytes = pleHistoryBytes() + indexerPendingBytes()};
  }
  bool operator==(const QwenHybridLayout &) const = default;
};

} // namespace splash::model
