#pragma once

#include "Model.hpp"
#include "QwenTarget.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

struct Qwen3_8Layout final {
  static constexpr std::string_view layerMagic = "MDFL0006";
  static constexpr std::array<uint32_t, 5> hiddenCaptureLayers{
      5, 19, 33, 47, 61};

  uint32_t maximumContextTokens = kv::kMaximumLogicalTokens;
  uint32_t layers = 64;
  uint32_t hiddenSize = 5120;
  uint32_t vocabularySize = 248320;
  uint32_t packedGdnWidth = 16640;
  uint32_t packedFullWidth = 14336;
  uint32_t convolutionDimension = 10240;
  uint32_t gdnKeyHeads = 16;
  uint32_t gdnValueHeads = 48;
  uint32_t gdnHeadDimension = 128;
  uint32_t attentionWidth = 6144;
  uint32_t intermediateSize = 17408;
  uint32_t attentionQueryHeads = 24;
  uint32_t attentionKvHeads = 4;
  uint32_t attentionHeadDimension = 256;
  uint32_t rotaryPairs = 32;
  float rotaryTheta = 10'000'000.0F;
  uint32_t fullAttentionPeriod = 4;
  uint32_t maskToken = 248070;
  std::array<uint32_t, 2> stopTokens{248044, 248046};

  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    return fullAttentionPeriod && (layer + 1) % fullAttentionPeriod == 0;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    return fullAttentionPeriod ? layers / fullAttentionPeriod : 0;
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
  bool operator==(const Qwen3_8Layout &) const = default;
};

struct Qwen3_8LayerWeights final {
  metal::MetalBuffer inputNorm;
  QwenMixerWeights mixer;
  metal::MetalBuffer postAttentionNorm;
  ops::Q4Projection gateProjection;
  ops::Q4Projection upProjection;
  ops::Q4Projection downProjection;
};

struct Qwen3_8Weights final {
  Qwen3_8Layout layout;
  std::vector<Qwen3_8LayerWeights> layers;
  metal::MetalBuffer finalNorm;
  ops::Q4Projection logitsProjection;
  ops::Q4Projection tokenEmbedding;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
};

[[nodiscard]] Qwen3_8Weights
loadQwen3_8Weights(metal::MetalBackend &backend,
                   const std::filesystem::path &directory,
                   Qwen3_8Layout layout = {});

} // namespace splash::model
