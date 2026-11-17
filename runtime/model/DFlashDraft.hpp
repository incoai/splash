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

// The dimensions of a DFlash2 draft, as its checkpoint's config.json gives
// them (ModelDescriptor.mm readDraftLayout): every draft's attention widths
// and selector are those the kernels are built for; its layers, intermediate
// size, rotary base and the target layers it reads are its own, and its
// hidden size and vocabulary its target's.
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
  uint32_t selectorRank = 0;
  uint32_t kvHeads = 0;
  // The target layers whose output rows the draft reads, ascending: the
  // first targetLayerCount of targetLayers, targetHiddenSize() values a row.
  std::array<uint32_t, ExecutionLimits::maximumDraftTargetLayers> targetLayers{};
  uint32_t targetLayerCount = 0;

  [[nodiscard]] constexpr DraftStateLayout stateLayout() const noexcept {
    return {layers, kvHeads, attentionHeadDimension, hiddenSize};
  }
  [[nodiscard]] constexpr std::span<const uint32_t> targetLayerIds() const noexcept {
    return {targetLayers.data(), targetLayerCount};
  }
  // A row of the target rows it reads: one of hiddenSize values per layer.
  [[nodiscard]] constexpr uint32_t targetHiddenSize() const noexcept { return hiddenSize * targetLayerCount; }
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

// The layout of a DFlash2 draft of a target of `hiddenSize` and
// `vocabularySize` with `layers` (at least one) layers of `intermediateSize`
// (a positive multiple of 64), rotary base `rotaryTheta` (positive), reading
// the target's layers `targetLayers` (ascending, 1 to
// ExecutionLimits::maximumDraftTargetLayers): its attention widths and
// selector are those the kernels are built for.
[[nodiscard]] DFlashDraftLayout draftLayout(uint32_t hiddenSize, uint32_t vocabularySize, uint32_t layers,
                                            uint32_t intermediateSize, float rotaryTheta,
                                            std::span<const uint32_t> targetLayers);

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
  // Each lane's retained rows of its rows of the step, a mask
  // (acceptance's), and its rows of the step (SplashLaneRows, which the
  // step's proposer writes).
  metal::MetalBuffer retainedRows;
  metal::MetalBuffer laneRows;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
  // Each lane's context window, which keeps the rows the lane retained as
  // its rings do; none for a lane whose decode leaves its window behind
  // (Model::capturesInDecode).
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> windows;
};

struct DFlashPrefillSpan final {
  uint32_t compactRow = 0;
  uint32_t rows = 0;
  uint32_t startPosition = 0;
  std::span<const DFlashDraftRingLayer> ring;
  // The lane's context window, which keeps the span's context rows.
  metal::MetalBuffer window;
};

// Context rows of a span a prefill also stores in a cached state's context
// window: the rows below a state boundary inside the command
// (ModelBatchItem::stateCaptures), whose window the command's later rows
// overwrite in the lane's.
struct DFlashWindowCapture final {
  uint32_t compactRow = 0;
  uint32_t rows = 0;
  uint32_t startPosition = 0;
  metal::MetalBuffer window;
};

struct DFlashPrefillBuffers final {
  metal::MetalBuffer capturedTargetHidden;
  // What the context projections' plans take beside their buffers: split
  // partials and counters, a table they prepare (Linear::addProjection).
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
  // The mask token's BF16 embedding the checkpoint ships, or none
  // (kDraftMaskEmbedding).
  metal::MetalBuffer maskEmbedding;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

class DraftCheckpointLoader;

// Reads a draft from the images DraftCheckpointLoader writes from a DFlash2
// checkpoint.
[[nodiscard]] DFlashDraftWeights
loadDFlashDraftWeights(metal::MetalBackend &backend, DraftCheckpointLoader &files,
                       DFlashDraftLayout layout);

// A draft decode step (DFlashDraft::step): each lane's cache length and its
// tree, which grows from its anchor, and the one vocabulary the step proposes
// from, whose columns its head writes (DFlashDraft::addDecode) and its
// selection reads (addSelection).
struct DFlashDraftStep final {
  std::span<const uint32_t> cacheLengths;
  std::span<const ops::DraftTreeLane> lanes;
  ops::DraftVocabulary vocabulary;
};

// Builds the draft layer graph and its proposal selection
// (ops::DraftSelector) from batch buffers and persistent context; the
// target's sampling and acceptance policy remain outside the model. Its
// logits are its target's vocabulary head's, `head`, over the whole
// vocabulary or over `subset` (QwenTargetDimensions::draftVocabulary), a view
// of the head's rows that reads that share of its weights.
class DFlashDraft final {
public:
  DFlashDraft(const DFlashDraftWeights &weights, metal::MetalBackend &backend,
               const ops::ExecutionPlans &operators, const ops::Projection &head,
               ops::DraftVocabulary subset);

  // A lane's tokens that choose the vocabulary of its steps: the last of the
  // tokens it read and its anchor (vocabulary).
  static constexpr uint32_t kVocabularyWindow = 64;
  // The vocabulary a step proposes from: `subset` while each lane's last
  // kVocabularyWindow tokens, the tail of `tokens[lane]` and `anchors[lane]`,
  // lie in it, else the whole vocabulary ({}). A token outside it ends the
  // draft's proposals there (a lane's lookup chain alone may hold it), so
  // text that leaves it drafts over the whole vocabulary until a window
  // passes in it: on 1.6.2, whose cycles verified the draft's chain, French
  // text, a tenth of whose tokens lie past the Qwen families' subset, kept
  // 0.2% fewer tokens a cycle than over the whole vocabulary through a window
  // of 64 (1.2% through 32, 22% with no window), and English agent cycles
  // drafted 93-100% of theirs over the subset.
  [[nodiscard]] static ops::DraftVocabulary vocabulary(ops::DraftVocabulary subset,
                                                       std::span<const std::span<const uint32_t>> tokens,
                                                       std::span<const uint32_t> anchors);
  // The step of lanes of `cacheLengths` and the trees `lanes` whose tokens
  // before their anchors are `tokens`, over the vocabulary they choose of
  // this draft's subset (vocabulary).
  [[nodiscard]] DFlashDraftStep step(std::span<const uint32_t> cacheLengths,
                                     std::span<const std::span<const uint32_t>> tokens,
                                     std::span<const ops::DraftTreeLane> lanes) const;

  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans,
                         std::span<const DFlashWindowCapture> windowCaptures) const;
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

  // Lanes of storage the draft's own projections, and its view of the head's
  // rows of its subset, bind in a decode step of `lanes` lanes
  // (ops::Linear::storageRows of a draft's block a lane; the whole head's are
  // QwenTarget::headStorageLanes). A context commit of a step's rows, up to a
  // verify block's a lane, binds as many lanes of its row tensors, which hold
  // a verify block's rows a lane: at least the staged tile's rows of any step
  // of `lanes` lanes (ops::stagedTileRows(T) <= 16 x lanes).
  [[nodiscard]] uint32_t decodeStorageLanes(uint32_t lanes) const;
  // A step's draft layers and its logits over its vocabulary, then its
  // proposals from them.
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers, const DFlashDraftStep &step) const;
  void addSelection(metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
                    const DFlashDraftStep &step) const;
  // Sets each of `lanes` lanes' mask rows of the draft input `rows` (all but
  // its anchor's, row 0 of its draftQueryRows) to the mask embedding the
  // checkpoint ships; without one they keep the target table's row.
  void addMaskRows(metal::CommandGraph &graph, metal::MetalBuffer rows, uint32_t lanes) const;

  // A decode step's context rows (`step`, among the target's captured rows):
  // those each lane retained, in path order, to its rings from its start
  // position on, rotated by the lane's RoPE rows (its block's positions), and
  // to its window when it has one (DFlashContextBuffers::windows). Its
  // normalized rows stay in buffers.hidden, which a state the step captures
  // stores in its own window.
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions,
                        const ops::StepRows &step) const;

private:
  // Each layer's context keys and values of the `rows` context rows in the
  // buffers' hidden rows, written to the spans' rings.
  void addContextKv(metal::CommandGraph &graph, const DFlashPrefillBuffers &buffers,
                    uint32_t rows, std::span<const DFlashPrefillSpan> spans) const;

  // The head of the logits over `vocabulary`.
  [[nodiscard]] const ops::Projection &head(ops::DraftVocabulary vocabulary) const;

  const DFlashDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  ops::DraftSelector selector_;
  const ops::Projection &head_;
  ops::DraftVocabulary subset_;
  // The head's rows of the subset's tokens.
  ops::Projection subsetHead_;
  // Each layer's key and value rows of its QKV projection, views of its
  // planes, which the context writers project with.
  std::vector<ops::Projection> contextKvProjections_;
};

} // namespace splash::model
