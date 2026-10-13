#pragma once

#include "Model.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <array>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace splash::model {

struct DFlashDraftRingLayer final {
  // K is [head][ring_position][dimension].
  metal::MetalBuffer keys;
  // V is [head][dimension][ring_position].
  metal::MetalBuffer values;
};

// A lane's draft rings: every draft layer's keys and values of the last
// SPLASH_DRAFT_SLIDING_WINDOW positions, which its decode attends to. Cached
// states keep their context window (DFlashContextWindow) instead.
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

private:
  std::shared_ptr<StateAllocationTracker> tracker_;
  std::vector<DFlashDraftRingLayer> layers_;
  uint64_t actualAllocatedBytes_ = 0;
};

// The context window of a lane or a cached state: the context rows the
// rings' keys and values are computed from (DraftStateLayout::windowBytes),
// which a cached state keeps in place of the rings.
class DFlashContextWindow final {
public:
  DFlashContextWindow(metal::MetalBackend &backend,
                      std::shared_ptr<StateAllocationTracker> tracker,
                      DraftStateLayout layout, std::string_view label);
  ~DFlashContextWindow();
  DFlashContextWindow(const DFlashContextWindow &) = delete;
  DFlashContextWindow &operator=(const DFlashContextWindow &) = delete;

  [[nodiscard]] const metal::MetalBuffer &buffer() const noexcept { return buffer_; }

private:
  std::shared_ptr<StateAllocationTracker> tracker_;
  metal::MetalBuffer buffer_;
  uint64_t actualAllocatedBytes_ = 0;
};

// The dimensions of a DFlash2 draft. Each target's draft is defined beside
// the target (Qwen3_8.hpp, Qwen3_6Moe.hpp).
struct DFlashDraftLayout final {
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t dynamicSize = 0;
  uint32_t qkvSize = 0;
  uint32_t attentionSize = 0;
  uint32_t intermediateSize = 0;
  uint32_t attentionHeadDimension = 0;
  float rotaryTheta = 0.0F;
  uint32_t targetHiddenSize = 0;
  uint32_t selectorRank = 0;
  uint32_t kvHeads = 0;

  [[nodiscard]] constexpr DraftStateLayout stateLayout() const noexcept {
    return {layers, kvHeads, attentionHeadDimension, hiddenSize};
  }
  [[nodiscard]] constexpr ops::DraftAttentionShape attentionShape() const noexcept {
    return {hiddenSize, dynamicSize, qkvSize, attentionSize,
            attentionSize / attentionHeadDimension, kvHeads,
            attentionHeadDimension};
  }
  // The key and value columns of the fused QKV projection, all the context
  // writers read.
  [[nodiscard]] constexpr uint32_t contextKvSize() const noexcept {
    return qkvSize - attentionSize;
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
  metal::MetalBuffer contextKv;
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
  // The lane's context window, which keeps the span's context rows.
  metal::MetalBuffer window;
};

struct DFlashPrefillBuffers final {
  metal::MetalBuffer capturedTargetHidden;
  // The split partials and counters of the context projections' chunks of
  // up to a decode batch (Linear::addPrefill).
  ops::LinearScratch linearScratch;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer contextKv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
};

struct DFlashDraftLayerWeights final {
  ops::NormWeights inputNorm;
  metal::MetalBuffer attentionConvolution;
  ops::Projection attentionDynamic;
  ops::Projection qkvProjection;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  ops::Projection outputProjection;
  ops::NormWeights postAttentionNorm;
  metal::MetalBuffer mlpConvolution;
  ops::Projection mlpDynamic;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

struct DFlashDraftWeights final {
  DFlashDraftLayout layout;
  std::vector<DFlashDraftLayerWeights> layers;
  ops::Projection contextProjection;
  ops::NormWeights hiddenNorm;
  ops::NormWeights finalNorm;
  ops::Projection selectorProjection;
  metal::MetalBuffer predecessorCodebook;
  metal::MetalBuffer successorCodebook;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

class DraftCheckpointLoader;

// Reads a draft from the images DraftCheckpointLoader writes from a DFlash2
// checkpoint.
[[nodiscard]] DFlashDraftWeights
loadDFlashDraftWeights(metal::MetalBackend &backend, DraftCheckpointLoader &files,
                       DFlashDraftLayout layout);

// Builds the draft layer graph and its proposal selection
// (ops::DraftSelector) from batch buffers and persistent context; the
// target's sampling and acceptance policy remain outside the model.
class DFlashDraft final {
public:
  DFlashDraft(const DFlashDraftWeights &weights, metal::MetalBackend &backend,
               const ops::ExecutionPlans &operators);

  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans) const;
  // Computes the rings' keys and values of `rows` positions from
  // startPosition on again from the context window they were kept in, as
  // addContextPrefill writes them from their context rows. The rows pass
  // through the context buffers, and the RoPE tables must hold the rows'
  // positions.
  void addContextRebuild(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers,
                         const metal::MetalBuffer &window,
                         std::span<const DFlashDraftRingLayer> ring,
                         uint32_t rows, uint32_t startPosition) const;

  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                 const ops::Projection &vocabularyProjection,
                 std::span<const uint32_t> cacheLengths) const;
  void addSelection(metal::CommandGraph &graph,
                    const ops::DraftSelectorBuffers &buffers,
                    std::span<const uint32_t> anchors,
                    std::span<const ops::SamplingPolicy> policies) const;
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions) const;

private:
  // Each layer's context keys and values of the `rows` context rows in the
  // buffers' hidden rows, written to the spans' rings.
  void addContextKv(metal::CommandGraph &graph, const DFlashPrefillBuffers &buffers,
                    uint32_t rows, std::span<const DFlashPrefillSpan> spans) const;

  const DFlashDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  ops::DraftSelector selector_;
  // Each layer's key and value rows of its QKV projection, views of its
  // planes, which the context writers project with.
  std::vector<ops::Projection> contextKvProjections_;
};

} // namespace splash::model
