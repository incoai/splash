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
  // The draft's own state of the lane after its rings
  // (DraftStateLayout::auxiliaryBytes: a DSpark draft's f_rnn row), on the
  // first layer's entry alone; none on the others, or for a draft of none.
  metal::MetalBuffer auxiliary{};
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

// The kind of a model's draft: a DFlash2 checkpoint's (DFlashDraft), or a
// DSpark draft's of a GGUF (model/DSparkDraft.hpp, Inkling 2 Mini's).
enum class DraftKind : uint8_t { DFlash2, DSpark };

// The dimensions of a model's draft. A DFlash2 draft's, as its checkpoint's
// config.json gives them (ModelDescriptor.mm readDraftLayout): every draft's
// attention widths and selector are those the kernels are built for; its
// layers, intermediate size, rotary base and the target layers it reads are
// its own, and its hidden size and vocabulary its target's. A DSpark draft's,
// as its GGUF's header gives them (dsparkLayout): its q/k/v rows hold its 24
// query heads (qkvSize, attentionSize), it has no convolutions (dynamicSize
// 0), its selector rank is its Markov head's, and it adds its layers'
// windows, the tokens it proposes and the cap of its logits. A model
// installed without a draft (--draft-model none) has the zero layout: no
// draft weights, state or work, and its decode cycles verify each lane's
// anchor and host chain alone.
struct DFlashDraftLayout final {
  DraftKind kind = DraftKind::DFlash2;
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
  // A DSpark draft's last is the target's layer count: its final normed rows.
  std::array<uint32_t, ExecutionLimits::maximumDraftTargetLayers> targetLayers{};
  uint32_t targetLayerCount = 0;
  // A DSpark draft's: the layers whose rows see the slidingWindow context
  // positions before their block's anchor alone (bit l for layer l; the
  // others see their rings' every key), the tokens it proposes, the first
  // proposedVocabulary of the vocabulary, and the cap of its logits,
  // logitCap * tanh(x / logitCap).
  uint32_t slidingLayers = 0;
  uint32_t slidingWindow = 0;
  uint32_t proposedVocabulary = 0;
  float logitCap = 0.0F;
  // The format its projections load in, which its decode plans store rows
  // by (draftStepLanes): a DFlash2 checkpoint's af4g64 planes
  // (DraftCheckpoint), a DSpark GGUF's own, every projection's.
  uint32_t projectionFormat = GGUF_FMT_AF4G64;

  // Whether the model decodes with a draft: a draft has layers.
  [[nodiscard]] constexpr bool present() const noexcept { return layers != 0; }
  // A DSpark draft's lane keeps each layer's context row of a slot in its
  // window, three windows of one row each side by side, and its f_rnn row
  // (bf16) beside its rings.
  [[nodiscard]] constexpr DraftStateLayout stateLayout() const noexcept {
    if (kind == DraftKind::DSpark)
      return {layers, kvHeads, attentionHeadDimension, layers * hiddenSize,
              hiddenSize * DraftStateLayout::bfloat16Bytes};
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

// What a DSpark draft's GGUF states of it (dsparkLayout): its target's
// hidden size and vocabulary, which must be its target's; its layers,
// intermediate size, rotary base, the target features it reads (as the
// GGUF names them: layer visits whose inputs it reads, the last the visit
// count, the final norm), its windowed layers and window, its Markov rank,
// the tokens it proposes, its logits' cap and its projections' format.
struct DSparkHeader final {
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t layers = 0;
  uint32_t intermediateSize = 0;
  float rotaryTheta = 0.0F;
  std::vector<uint32_t> targetFeatures;
  uint32_t slidingLayers = 0;
  uint32_t slidingWindow = 0;
  uint32_t markovRank = 0;
  uint32_t proposedVocabulary = 0;
  float logitCap = 0.0F;
  uint32_t projectionFormat = GGUF_FMT_COUNT;
};

// The layout of a DSpark draft of `header` over a target of `hiddenSize`,
// `vocabularySize` and `targetLayers` layers: its attention of the widths the
// kernels take with 24 query heads, its Markov rank theirs, its windows
// inside a ring, and its features ascending, the last the target's final
// norm (targetLayers), each other one the input of a target layer, which it
// reads as the output of the layer before (its target layers).
[[nodiscard]] DFlashDraftLayout dsparkLayout(uint32_t hiddenSize, uint32_t vocabularySize, uint32_t targetLayers,
                                             const DSparkHeader &header);

struct DFlashDecodeBuffers final {
  ops::LinearScratch linearScratch{};
  // Each lane's draft state beside its rings (DFlashDraftRingLayer::
  // auxiliary), for a draft that keeps one.
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> laneAuxiliary{};
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
  // Each lane's draft state beside its rings, as DFlashDecodeBuffers'.
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> laneAuxiliary{};
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

struct DSparkDraftWeights;

struct DFlashDraftWeights final {
  DFlashDraftLayout layout;
  // A DSpark draft's weights (model/DSparkDraft.hpp) in place of the DFlash2
  // fields below, which it leaves empty; its files and bytes are these'.
  std::shared_ptr<const DSparkDraftWeights> dspark;
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

// A model's draft as the runtime drives it (a DFlash2 draft, DFlashDraft, or
// a DSpark draft, model/DSparkDraft.hpp): its context from its target's
// captured rows, kept in each lane's rings and window, its block's layers
// and logits, and its proposals into the selector's trees. DFlashDraft's
// comments describe each method.
class DraftModel {
public:
  DraftModel() = default;
  DraftModel(const DraftModel &) = delete;
  DraftModel &operator=(const DraftModel &) = delete;
  virtual ~DraftModel() = default;

  [[nodiscard]] virtual DFlashDraftStep step(std::span<const uint32_t> cacheLengths,
                                             std::span<const std::span<const uint32_t>> tokens,
                                             std::span<const ops::DraftTreeLane> lanes) const = 0;
  virtual void addContextPrefill(metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
                                 std::span<const DFlashPrefillSpan> spans,
                                 std::span<const DFlashWindowCapture> windowCaptures) const = 0;
  virtual void addContextRebuild(metal::CommandGraph &graph, DFlashPrefillBuffers buffers,
                                 const metal::MetalBuffer &window, std::span<const DFlashDraftRingLayer> ring,
                                 uint32_t rows, uint32_t startPosition) const = 0;
  virtual void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                         const DFlashDraftStep &step) const = 0;
  virtual void addSelection(metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
                            const DFlashDraftStep &step) const = 0;
  virtual void addMaskRows(metal::CommandGraph &graph, metal::MetalBuffer rows, uint32_t lanes) const = 0;
  virtual void addContextCommit(metal::CommandGraph &graph, DFlashContextBuffers buffers,
                                std::span<const uint32_t> startPositions, const ops::StepRows &step) const = 0;
  // Work the draft runs once before its first step, with the decode
  // arena's linear scratch and `rows`, scratch of the vocabulary head's
  // fp32 output of a decode step of `stepRows` rows, the most of the
  // runtime's (QwenTargetGeometry::stepRows): none but a DSpark draft's
  // fold of its Markov head into the head.
  virtual void addSetup(metal::CommandGraph &, ops::LinearScratch, metal::MetalBuffer, uint32_t) const {}
};

// Builds the draft layer graph and its proposal selection
// (ops::DraftSelector) from batch buffers and persistent context; the
// target's sampling and acceptance policy remain outside the model. Its
// logits are its target's vocabulary head's, `head`, over the whole
// vocabulary or over `subset` (QwenTargetDimensions::draftVocabulary), a view
// of the head's rows that reads that share of its weights.
class DFlashDraft final : public DraftModel {
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
                                     std::span<const ops::DraftTreeLane> lanes) const override;

  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans,
                         std::span<const DFlashWindowCapture> windowCaptures) const override;
  // Computes the rings' keys and values of `rows` positions from
  // startPosition on again from the context window they were kept in, as
  // addContextPrefill writes them from their context rows. The rows pass
  // through the context buffers, and the RoPE tables must hold the rows'
  // positions.
  void addContextRebuild(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers,
                         const metal::MetalBuffer &window,
                         std::span<const DFlashDraftRingLayer> ring,
                         uint32_t rows, uint32_t startPosition) const override;

  // A step's draft layers and its logits over its vocabulary, then its
  // proposals from them.
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers, const DFlashDraftStep &step) const override;
  void addSelection(metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
                    const DFlashDraftStep &step) const override;
  // Sets each of `lanes` lanes' mask rows of the draft input `rows` (all but
  // its anchor's, row 0 of its draftQueryRows) to the mask embedding the
  // checkpoint ships; without one they keep the target table's row.
  void addMaskRows(metal::CommandGraph &graph, metal::MetalBuffer rows, uint32_t lanes) const override;

  // A decode step's context rows (`step`, among the target's captured rows):
  // those each lane retained, in path order, to its rings from its start
  // position on, rotated by the lane's RoPE rows (its block's positions), and
  // to its window when it has one (DFlashContextBuffers::windows). Its
  // normalized rows stay in buffers.hidden, which a state the step captures
  // stores in its own window.
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions,
                        const ops::StepRows &step) const override;

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
