#pragma once

#include "QwenTarget.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

struct Qwen3_6MoeLayout final {
  static constexpr std::string_view layerMagic = "MDFM0001";
  static constexpr std::array<uint32_t, 8> hiddenCaptureLayers{
      1, 6, 11, 16, 22, 27, 32, 37};

  uint32_t maximumContextTokens = kv::kMaximumLogicalTokens;
  uint32_t layers = 40;
  uint32_t hiddenSize = 2048;
  uint32_t vocabularySize = 248320;
  uint32_t packedGdnWidth = 12544;
  uint32_t packedFullWidth = 9216;
  uint32_t convolutionDimension = 8192;
  uint32_t gdnKeyHeads = 16;
  uint32_t gdnValueHeads = 32;
  uint32_t gdnHeadDimension = 128;
  uint32_t attentionWidth = 4096;
  uint32_t attentionQueryHeads = 16;
  uint32_t attentionKvHeads = 2;
  uint32_t attentionHeadDimension = 256;
  uint32_t rotaryPairs = 32;
  float rotaryTheta = 10'000'000.0F;
  uint32_t fullAttentionPeriod = 4;
  uint32_t experts = 256;
  uint32_t expertsPerToken = 8;
  uint32_t expertIntermediateSize = 512;
  uint32_t maskToken = 248077;
  std::array<uint32_t, 2> stopTokens{248044, 248046};

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
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * hiddenCaptureLayers.size();
  }
  [[nodiscard]] constexpr QwenMixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize,     packedGdnWidth, packedFullWidth,
            convolutionDimension, gdnValueHeads,  gdnHeadDimension,
            attentionWidth, attentionHeadDimension};
  }

  bool operator==(const Qwen3_6MoeLayout &) const = default;
};

struct Qwen3_6MoeLayerWeights final {
  metal::MetalBuffer inputNorm;
  QwenMixerWeights mixer;
  metal::MetalBuffer postAttentionNorm;
  ops::MoeWeights ffn;
};

struct Qwen3_6MoeWeights final {
  Qwen3_6MoeLayout layout;
  std::vector<Qwen3_6MoeLayerWeights> layers;
  metal::MetalBuffer finalNorm;
  ops::Q4Projection logitsProjection;
  ops::Q4Projection tokenEmbedding;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
};

[[nodiscard]] Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend,
                      const std::filesystem::path &directory,
                      Qwen3_6MoeLayout layout = {});

} // namespace splash::model
