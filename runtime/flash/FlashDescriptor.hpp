#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace splash::flash {

enum class FlashLayerKind : uint8_t { GatedDeltaNet, SparseAttention };
// Applies to Qwen4 RMSNorm modules only. GDN's gated norm and vision norms
// always consume direct gamma independently of the checkpoint-wide audit.
enum class NormConvention : uint8_t { OnePlusWeight, DirectGamma };

struct FlashNormAudit final {
  NormConvention convention = NormConvention::OnePlusWeight;
  uint32_t anchors = 0;
  double medianMean = 0.0;
  double onesCenteredFraction = 0.0;
};

// The first native route deliberately supports the inspected local checkpoint
// geometry. Unsupported architecture/configuration fields fail before runtime
// construction; no existing Splash model descriptor is changed.
struct FlashDescriptor final {
  uint32_t layers = 48;
  uint32_t hiddenSize = 2560;
  uint32_t vocabularySize = 248320;
  uint32_t maximumContextTokens = 262144;
  uint32_t hcCount = 4;
  uint32_t hcLowRank = 320;
  uint32_t experts = 512;
  uint32_t expertsPerToken = 10;
  uint32_t expertIntermediateSize = 640;
  uint32_t sharedIntermediateSize = 640;
  uint32_t linearKeyHeads = 16;
  uint32_t linearValueHeads = 48;
  uint32_t linearKeyDimension = 128;
  uint32_t linearValueDimension = 128;
  uint32_t linearConvolutionTaps = 4;
  uint32_t attentionHeads = 24;
  uint32_t attentionKvHeads = 2;
  uint32_t attentionHeadDimension = 256;
  uint32_t rotaryDimensions = 64;
  std::array<uint32_t, 3> mropeSections{11, 11, 10};
  double rotaryTheta = 10000000.0;
  double normEpsilon = 1e-6;
  uint32_t indexerHeads = 4;
  uint32_t indexerKvHeads = 1;
  uint32_t indexerHeadDimension = 128;
  uint32_t indexerBudget = 2048;
  uint32_t indexerCompression = 4;
  uint32_t pleEmbeddingSize = 2560;
  uint32_t pleNgramSize = 3;
  uint32_t pleHeadsPerNgram = 8;
  uint32_t pleVocabularyBase = 20000000;
  uint32_t pleVocabularyAlignment = 128;
  uint32_t pleParts = 128;
  uint32_t pleConvolutionTaps = 4;
  uint32_t pleHistoryEos = 248044;
  uint32_t mtpLayers = 1;
  uint32_t visionLayers = 27;
  uint32_t visionHiddenSize = 1152;
  uint32_t visionIntermediateSize = 4304;
  uint32_t visionHeads = 16;
  uint32_t visionPatchSize = 16;
  uint32_t visionTemporalPatchSize = 2;
  uint32_t visionSpatialMerge = 2;
  uint32_t visionPositionCount = 2304;
  uint32_t imageToken = 248056;
  uint32_t videoToken = 248057;
  uint32_t visionStartToken = 248053;
  uint32_t visionEndToken = 248054;
  std::array<uint32_t, 2> stopTokens{248046, 248044};
  std::array<FlashLayerKind, 48> layerKinds{};
  std::vector<uint32_t> pleLayerIndices{1};
  // Filled from the three stored I64 arrays and shared BF16 scalar by the
  // aligned weight loader. Config parsing never regenerates these values.
  bool pleParametersLoaded = false;
  std::array<int64_t, 3> pleLayerMultipliers{};
  std::array<int64_t, 16> pleHeadVocabularySizes{};
  std::array<int64_t, 16> pleHeadOffsets{};
  float pleSharedWeightScale = 0.0F;
  uint64_t pleTableRows = 0;

  [[nodiscard]] uint32_t hyperHiddenSize() const noexcept {
    return hcCount * hiddenSize;
  }
  [[nodiscard]] uint32_t pleHeadCount() const noexcept {
    return (pleNgramSize - 1) * pleHeadsPerNgram;
  }
  [[nodiscard]] uint32_t pleHeadDimension() const noexcept {
    return pleEmbeddingSize / pleHeadCount();
  }
  [[nodiscard]] uint32_t pleConvolutionStateRows() const noexcept {
    return (pleConvolutionTaps - 1) * pleNgramSize;
  }
  void validate() const;
  [[nodiscard]] static FlashDescriptor
  fromConfig(const std::filesystem::path &path);
};

} // namespace splash::flash
