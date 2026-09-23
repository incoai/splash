#pragma once

#include "Model.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/Sampling.hpp"

#include <cstdint>
#include <filesystem>
#include <array>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

struct DFlashDraftRingLayer final {
  // K is [head][ring_position][dimension].
  metal::MetalBuffer keys;
  // V is [head][dimension][ring_position].
  metal::MetalBuffer values;
};

// Draft-owned persistent context, paired with target state in composite caches.
class DFlashDraftRing final {
public:
  DFlashDraftRing(metal::MetalBackend &backend,
                  std::shared_ptr<StateAllocationTracker> tracker,
                  DraftStateLayout layout,
                  std::string_view label);
  ~DFlashDraftRing();
  DFlashDraftRing(const DFlashDraftRing &) = delete;
  DFlashDraftRing &operator=(const DFlashDraftRing &) = delete;

  [[nodiscard]] const std::vector<DFlashDraftRingLayer> &layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept {
    return actualAllocatedBytes_;
  }

private:
  std::shared_ptr<StateAllocationTracker> tracker_;
  std::vector<DFlashDraftRingLayer> layers_;
  uint64_t actualAllocatedBytes_ = 0;
};

struct DFlashDraftLayout final {
  uint32_t layers = 5;
  uint32_t hiddenSize = 5120;
  uint32_t vocabularySize = 248320;
  uint32_t dynamicSize = 1280;
  uint32_t qkvSize = 6144;
  uint32_t attentionSize = 4096;
  uint32_t intermediateSize = 17408;
  uint32_t attentionHeadDimension = 128;
  float rotaryTheta = 10'000'000.0F;
  uint32_t targetHiddenSize = 25600;
  uint32_t selectorRank = 256;
  uint32_t kvHeads = 8;

  [[nodiscard]] constexpr DraftStateLayout stateLayout() const noexcept {
    return {layers, kvHeads, ExecutionLimits::draftContextTokens,
            attentionHeadDimension};
  }
  [[nodiscard]] constexpr ops::DraftAttentionShape attentionShape() const noexcept {
    return {hiddenSize, dynamicSize, qkvSize, attentionSize,
            attentionSize / attentionHeadDimension, kvHeads,
            attentionHeadDimension};
  }

  bool operator==(const DFlashDraftLayout &) const = default;
};

struct DFlashDecodeBuffers final {
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer dynamic;
  metal::MetalBuffer convolved;
  metal::MetalBuffer proposalQkv;
  metal::MetalBuffer attention;
  metal::MetalBuffer projected;
  metal::MetalBuffer residual;
  metal::MetalBuffer intermediate;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer queryKeys;
  metal::MetalBuffer queryValues;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer gateScratch;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
};

struct DFlashContextBuffers final {
  ops::LinearScratch linearScratch{};
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer qkv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer retainedCounts;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
};

struct DFlashPrefillSpan final {
  uint32_t compactRow = 0;
  uint32_t rows = 0;
  uint32_t startPosition = 0;
  std::span<const DFlashDraftRingLayer> ring;
};

struct DFlashPrefillBuffers final {
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer projectionSums;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer qkv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
};

struct DFlashSelectionBuffers final {
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
};

struct DFlashDraftLayerWeights final {
  ops::NormWeights inputNorm;
  metal::MetalBuffer attentionConvolution;
  ops::Q4Projection attentionDynamic;
  ops::Q4Projection qkvProjection;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  ops::Q4Projection outputProjection;
  ops::NormWeights postAttentionNorm;
  metal::MetalBuffer mlpConvolution;
  ops::Q4Projection mlpDynamic;
  ops::Q4Projection gateProjection;
  ops::Q4Projection upProjection;
  ops::Q4Projection downProjection;
};

struct DFlashDraftWeights final {
  DFlashDraftLayout layout;
  std::vector<DFlashDraftLayerWeights> layers;
  ops::Q4Projection contextProjection;
  ops::NormWeights hiddenNorm;
  ops::NormWeights finalNorm;
  ops::Q4Projection selectorProjection;
  metal::MetalBuffer predecessorCodebook;
  metal::MetalBuffer successorCodebook;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
};

inline constexpr std::string_view kDFlashLayerMagic = "MDFD0004";

[[nodiscard]] DFlashDraftWeights
loadDFlashDraftWeights(metal::MetalBackend &backend,
                       const std::filesystem::path &directory,
                       DFlashDraftLayout layout = {});

// Builds the draft layer graph from packed buffers and persistent context.
// Sampling and acceptance policy remain outside the model.
class DFlashDraft final {
public:
  DFlashDraft(const DFlashDraftWeights &weights, metal::MetalBackend &backend,
               const ops::ExecutionPlans &operators);

  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans) const;

  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                 const ops::Q4Projection &vocabularyProjection,
                 std::span<const uint32_t> cacheLengths, uint32_t lanes,
                 ops::Q4DispatchStats &stats) const;
  void addSelection(metal::CommandGraph &graph,
                    DFlashSelectionBuffers buffers,
                    std::span<const uint32_t> anchors,
                    std::span<const ops::SamplingPolicy> policies,
                    uint32_t proposalTokens) const;
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions,
                        uint32_t lanes,
                        ops::Q4DispatchStats &stats) const;

private:
  const DFlashDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  ops::Sampling selector_;
};

} // namespace splash::model
