#pragma once

#include "model/StateLayout.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace splash::model {

enum class QwenFfnKind : uint8_t { Dense, SparseMoe };

// The dimensions and tokens of a Qwen hybrid target: GDN layers, every
// fullAttentionPeriod-th layer full attention instead, each followed by a
// dense FFN or a sparse MoE. A family's layout (QwenHybridLayout) sets them,
// and every view of a target reads them from there: the target loaders, the
// GGUF planner and the runtime's QwenTargetGeometry.
struct QwenTargetDimensions {
  // The native window, the context max_position_embeddings states, which
  // must fit the runtime's KV ceiling (kv::kMaximumLogicalTokens).
  uint32_t maximumContextTokens = 0;
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedFullWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionQueryHeads = 0;
  uint32_t attentionKvHeads = 0;
  uint32_t attentionHeadDimension = 0;
  // The rotated dimension pairs of each attention head and their RoPE base.
  uint32_t rotaryPairs = 0;
  float rotaryTheta = 0.0F;
  uint32_t fullAttentionPeriod = 0;
  uint32_t maskToken = 0;
  std::array<uint32_t, 2> stopTokens{};
  QwenFfnKind ffnKind = QwenFfnKind::Dense;
  // The dense FFN's width.
  uint32_t intermediateSize = 0;
  // The sparse MoE's routed experts, those each token takes, and the width of
  // each and of its shared expert.
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  // Qwen3.8-Flash-Next (model/Qwen4Exp.hpp) when hyperConnections is set: its
  // residual's streams and the mixes' low rank; the QSA indexer's heads of
  // indexerHeadDimension values, past indexerTopBlocks blocks of
  // indexerBlockTokens tokens of which a row attends sparsely; and the PLE
  // n-gram embedding's layer, n-gram size, heads per n-gram, head width and
  // convolution taps.
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
  [[nodiscard]] constexpr uint32_t actualGdnWidth() const noexcept {
    return convolutionDimension + attentionWidth + 2 * gdnValueHeads;
  }
  [[nodiscard]] constexpr bool qwen4() const noexcept { return hyperConnections != 0; }
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
  // The tokens a QSA row attends to sparsely past (the indexer's top_k).
  [[nodiscard]] constexpr uint32_t indexerTokens() const noexcept { return indexerTopBlocks * indexerBlockTokens; }
  bool operator==(const QwenTargetDimensions &) const = default;
};

// A family's layout: its dimensions, the CaptureLayers layers whose hidden
// states the draft reads, and the cache layouts they make. A family sets
// every value its FFN uses and adds its file magics.
template <size_t CaptureLayers> struct QwenHybridLayout : QwenTargetDimensions {
  std::array<uint32_t, CaptureLayers> hiddenCaptureLayers{};

  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    return {attentionLayerCount(), attentionKvHeads,
            attentionHeadDimension};
  }
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {layers - attentionLayerCount(), kGdnConvolutionTaps - 1,
            convolutionDimension,
            gdnValueHeads, gdnHeadDimension, gdnHeadDimension};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * hiddenCaptureLayers.size();
  }
  bool operator==(const QwenHybridLayout &) const = default;
};

} // namespace splash::model
