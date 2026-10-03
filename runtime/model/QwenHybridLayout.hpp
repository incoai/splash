#pragma once

#include "model/StateLayout.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace splash::model {

enum class QwenFfnKind : uint8_t { Dense, SparseMoe };

// The magic of a Qwen target's packed embedding file, whatever its family.
inline constexpr std::string_view kEmbeddingMagic = "MDFE0001";

// Sizes of the mixer sections in a packed layer file.
struct QwenMixerGeometry final {
  uint32_t hiddenSize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedFullWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionHeadDimension = 0;
};

// The dimensions and tokens of a Qwen hybrid target: GDN layers, every
// fullAttentionPeriod-th layer full attention instead, each followed by the
// family's FFN, and the CaptureLayers layers whose hidden states the draft
// reads. A family sets every value and adds its FFN sizes, its file magics
// and its ffnKind.
template <size_t CaptureLayers> struct QwenHybridLayout {
  uint32_t maximumContextTokens = kv::kMaximumLogicalTokens;
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
  uint32_t rotaryPairs = 0;
  float rotaryTheta = 0.0F;
  uint32_t fullAttentionPeriod = 0;
  uint32_t maskToken = 0;
  std::array<uint32_t, 2> stopTokens{};
  std::array<uint32_t, CaptureLayers> hiddenCaptureLayers{};

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
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    return {attentionLayerCount(), attentionKvHeads,
            attentionHeadDimension};
  }
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {layers - attentionLayerCount(), kGdnConvolutionTaps - 1,
            convolutionDimension,
            gdnValueHeads, gdnHeadDimension, gdnHeadDimension};
  }
  [[nodiscard]] constexpr QwenMixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize,     packedGdnWidth, packedFullWidth,
            convolutionDimension, gdnValueHeads,  gdnHeadDimension,
            attentionWidth, attentionHeadDimension};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * hiddenCaptureLayers.size();
  }
  bool operator==(const QwenHybridLayout &) const = default;
};

} // namespace splash::model
