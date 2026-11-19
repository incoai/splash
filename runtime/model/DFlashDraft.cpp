#include "DFlashDraft.hpp"
#include "Checked.hpp"
#include "DraftCheckpoint.hpp"
#include "metal/abi/DraftAttention.h"
#include "ops/Embedding.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

void requireLayout(const DFlashDraftLayout &layout) {
  if (!layout.layers || !layout.hiddenSize || !layout.vocabularySize ||
      !layout.dynamicSize || !layout.qkvSize || !layout.attentionSize ||
      !layout.intermediateSize || !layout.attentionHeadDimension ||
      !(layout.rotaryTheta > 0.0F) ||
      !layout.targetLayerCount || !layout.selectorRank ||
      !layout.kvHeads) {
    throw WeightStoreError("DFlash draft layout contains a zero dimension");
  }
  if (layout.selectorRank != SPLASH_DRAFT_SELECTOR_RANK) {
    throw WeightStoreError("draft selector kernels are compiled for rank " +
                           std::to_string(SPLASH_DRAFT_SELECTOR_RANK));
  }
}

// The key and value rows of each layer's fused QKV projection, a view of its
// rows (ops::Projection::rows) without a copy.
std::vector<ops::Projection> contextKvRows(const metal::MetalBackend &backend,
                                           const DFlashDraftWeights &weights) {
  const DFlashDraftLayout &layout = weights.layout;
  std::vector<ops::Projection> result;
  result.reserve(weights.layers.size());
  for (const DFlashDraftLayerWeights &layer : weights.layers)
    result.push_back(layer.qkvProjection.rows(backend, {{layout.attentionSize, layout.contextKvSize()}}));
  return result;
}

// The head's rows of the subset's tokens (ops::DraftVocabulary::rows); the
// head itself for the whole vocabulary.
ops::Projection subsetRows(const metal::MetalBackend &backend, const ops::Projection &head,
                           ops::DraftVocabulary subset) {
  if (subset == ops::DraftVocabulary{}) return head;
  const std::array<ops::Projection::RowRange, 2> rows = subset.rows(head.outputSize);
  return head.rows(backend, {rows[0], rows[1]});
}

} // namespace

DFlashDraftLayout draftLayout(uint32_t hiddenSize, uint32_t vocabularySize, uint32_t layers,
                              uint32_t intermediateSize, float rotaryTheta, std::span<const uint32_t> targetLayers,
                              bool mixesRows) {
  constexpr uint32_t most = ExecutionLimits::maximumDraftTargetLayers;
  if (targetLayers.empty() || targetLayers.size() > most ||
      std::ranges::adjacent_find(targetLayers, std::ranges::greater_equal{}) != targetLayers.end())
    throw std::invalid_argument("a DFlash2 draft reads 1 to " + std::to_string(most) +
                                " target layers, ascending");
  // A layer is a bit of causalLayers.
  if (!layers || layers > 64) throw std::invalid_argument("a DFlash2 draft has 1 to 64 layers");
  // Its projections quantize in groups of 64 columns (DraftCheckpoint.cpp).
  if (!intermediateSize || intermediateSize % 64)
    throw std::invalid_argument("a DFlash2 draft's intermediate size is a positive multiple of 64");
  if (!(rotaryTheta > 0)) throw std::invalid_argument("a DFlash2 draft's rotary theta is positive");
  DFlashDraftLayout layout{.mixesRows = mixesRows,
                           .layers = layers,
                           .hiddenSize = hiddenSize,
                           .vocabularySize = vocabularySize,
                           .dynamicSize = draft_dynamic_width(hiddenSize),
                           .qkvSize = SPLASH_DRAFT_QKV_WIDTH,
                           .attentionSize = SPLASH_DRAFT_ATTENTION_WIDTH,
                           .intermediateSize = intermediateSize,
                           .attentionHeadDimension = SPLASH_DRAFT_HEAD_DIMENSION,
                           .rotaryTheta = rotaryTheta,
                           .selectorRank = SPLASH_DRAFT_SELECTOR_RANK,
                           .kvHeads = SPLASH_DRAFT_KV_HEADS,
                           .targetLayerCount = static_cast<uint32_t>(targetLayers.size())};
  std::ranges::copy(targetLayers, layout.targetLayers.begin());
  requireLayout(layout);
  return layout;
}

DFlashDraftRing::DFlashDraftRing(
    metal::MetalBackend &backend, std::shared_ptr<StateAllocationTracker> tracker,
    DraftStateLayout layout, std::string_view label)
    : tracker_(std::move(tracker)), layers_(layout.layers) {
  if (!tracker_)
    throw std::invalid_argument("draft state allocation tracker is empty");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  const metal::MetalBuffer base = backend.allocateBuffer(
      layout.ringBytes(), metal::BufferStorage::Shared, label);
  uint64_t cursor = 0;
  for (DFlashDraftRingLayer &layer : layers_) {
    layer.keys = backend.view(base, cursor, layout.tensorBytes());
    cursor += layout.tensorBytes();
    layer.values = backend.view(base, cursor, layout.tensorBytes());
    cursor += layout.tensorBytes();
  }
  if (cursor != layout.ringBytes())
    throw std::logic_error("draft ring accounting mismatch");
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (actualAllocatedBytes_ < layout.ringBytes())
    throw std::logic_error("draft ring allocation is below declared bytes");
  tracker_->bytes.fetch_add(actualAllocatedBytes_, std::memory_order_relaxed);
}

DFlashDraftRing::~DFlashDraftRing() {
  tracker_->bytes.fetch_sub(actualAllocatedBytes_, std::memory_order_relaxed);
}

DFlashContextWindow::DFlashContextWindow(
    metal::MetalBackend &backend, std::shared_ptr<StateAllocationTracker> tracker,
    DraftStateLayout layout, std::string_view label)
    : tracker_(std::move(tracker)) {
  if (!tracker_)
    throw std::invalid_argument("draft state allocation tracker is empty");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  buffer_ = backend.allocateBuffer(layout.windowBytes(), metal::BufferStorage::Shared, label);
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (actualAllocatedBytes_ < layout.windowBytes())
    throw std::logic_error("context window allocation is below declared bytes");
  tracker_->bytes.fetch_add(actualAllocatedBytes_, std::memory_order_relaxed);
}

DFlashContextWindow::~DFlashContextWindow() {
  tracker_->bytes.fetch_sub(actualAllocatedBytes_, std::memory_order_relaxed);
}

DFlashDraft::DFlashDraft(const DFlashDraftWeights &weights,
                         metal::MetalBackend &backend,
                         const ops::ExecutionPlans &operators,
                         const ops::Projection &head, ops::DraftVocabulary subset)
    : weights_(weights), backend_(backend), operators_(operators),
      selector_(weights.layout.vocabularySize), head_(head), subset_(subset),
      subsetHead_(subsetRows(backend, head, subset)),
      contextKvProjections_(contextKvRows(backend, weights)) {
  if (head.outputSize != weights.layout.vocabularySize)
    throw std::invalid_argument("a draft's head is its vocabulary's");
}

ops::DraftVocabulary DFlashDraft::vocabulary(ops::DraftVocabulary subset,
                                             std::span<const std::span<const uint32_t>> tokens,
                                             std::span<const uint32_t> anchors) {
  if (tokens.size() != anchors.size())
    throw std::invalid_argument("a draft step's lanes each have tokens and an anchor");
  const auto holds = [&](uint32_t token) { return subset.holds(token); };
  for (size_t lane = 0; lane < anchors.size(); ++lane) {
    const std::span<const uint32_t> read = tokens[lane];
    if (!holds(anchors[lane]) ||
        !std::ranges::all_of(read.last(std::min<size_t>(read.size(), kVocabularyWindow - 1)), holds))
      return {};
  }
  return subset;
}

DFlashDraftStep DFlashDraft::step(std::span<const uint32_t> cacheLengths,
                                  std::span<const std::span<const uint32_t>> tokens,
                                  std::span<const ops::DraftTreeLane> lanes) const {
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> anchors{};
  if (cacheLengths.size() != lanes.size() || lanes.size() > anchors.size())
    throw std::invalid_argument("a draft step's lanes, a batch's at most, each have a cache length and a tree");
  std::ranges::transform(lanes, anchors.begin(), &ops::DraftTreeLane::anchor);
  return {cacheLengths, lanes, vocabulary(subset_, tokens, std::span(anchors).first(lanes.size()))};
}

const ops::Projection &DFlashDraft::head(ops::DraftVocabulary vocabulary) const {
  if (vocabulary == subset_) return subsetHead_;
  if (vocabulary == ops::DraftVocabulary{}) return head_;
  throw std::invalid_argument("a draft proposes from the whole vocabulary or its subset");
}

void DFlashDraft::addSelection(metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
                               const DFlashDraftStep &step) const {
  const ops::DraftCodebooks codebooks{weights_.predecessorCodebook, weights_.successorCodebook};
  selector_.add(graph, buffers, weights_.layout.mixesRows ? &codebooks : nullptr, step.vocabulary, step.lanes);
}

void DFlashDraft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
    std::span<const DFlashPrefillSpan> spans,
    std::span<const DFlashWindowCapture> windowCaptures) const {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget || spans.empty())
    throw std::invalid_argument("invalid draft context prefill");
  const DFlashDraftLayout &layout = weights_.layout;
  for (const DFlashPrefillSpan &span : spans) {
    if (span.ring.size() != layout.layers)
      throw std::invalid_argument("draft prefill ring layer mismatch");
  }
  operators_.draftLinear().addProjection(graph, buffers.capturedTargetHidden, weights_.contextProjection,
                                         buffers.projected, rows, buffers.linearScratch);
  ops::Normalization::addRms(graph, buffers.projected, weights_.hiddenNorm, buffers.hidden, layout.hiddenSize,
                             rows);
  const uint64_t rowBytes = uint64_t{layout.hiddenSize} * sizeof(uint16_t);
  const auto store = [&](uint32_t compactRow, uint32_t count, uint32_t startPosition,
                         const metal::MetalBuffer &window) {
    ops::DraftAttention::addWindowStore(graph,
                                        backend_.view(buffers.hidden, compactRow * rowBytes, count * rowBytes),
                                        window, count, startPosition, layout.hiddenSize);
  };
  for (const DFlashPrefillSpan &span : spans)
    store(span.compactRow, span.rows, span.startPosition, span.window);
  // Only the window: a cached state keeps no rings.
  for (const DFlashWindowCapture &capture : windowCaptures)
    store(capture.compactRow, capture.rows, capture.startPosition, capture.window);
  addContextKv(graph, buffers, rows, spans);
}

void DFlashDraft::addContextRebuild(metal::CommandGraph &graph,
                                    DFlashPrefillBuffers buffers,
                                    const metal::MetalBuffer &window,
                                    std::span<const DFlashDraftRingLayer> ring,
                                    uint32_t rows, uint32_t startPosition) const {
  static_assert(ExecutionLimits::draftContextTokens <= ExecutionLimits::prefillTokenBudget,
                "a whole draft window passes through buffers of one prefill chunk's rows");
  const DFlashDraftLayout &layout = weights_.layout;
  if (!rows || rows > ExecutionLimits::draftContextTokens || ring.size() != layout.layers)
    throw std::invalid_argument("invalid draft context rebuild");
  ops::DraftAttention::addWindowLoad(graph, window, buffers.hidden, rows, startPosition,
                                     layout.hiddenSize);
  const DFlashPrefillSpan span{0, rows, startPosition, ring, window};
  addContextKv(graph, buffers, rows, {&span, 1});
}

void DFlashDraft::addContextKv(metal::CommandGraph &graph,
                               const DFlashPrefillBuffers &buffers, uint32_t rows,
                               std::span<const DFlashPrefillSpan> spans) const {
  const DFlashDraftLayout &layout = weights_.layout;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    operators_.draftLinear().addProjection(graph, buffers.hidden, contextKvProjections_[layer], buffers.contextKv,
                                           rows, buffers.linearScratch);
    for (const DFlashPrefillSpan &span : spans) {
      const uint64_t kvOffset =
          uint64_t{span.compactRow} * layout.contextKvSize() * sizeof(uint16_t);
      const uint64_t ropeOffset =
          uint64_t{span.compactRow} * (layout.attentionHeadDimension / 2) * sizeof(float);
      ops::DraftAttention::addContextPrefill(
          graph,
          backend_.view(buffers.contextKv, kvOffset,
                        uint64_t{span.rows} * layout.contextKvSize() *
                            sizeof(uint16_t)),
          weights_.layers[layer].keyNorm,
          backend_.view(buffers.ropeCos, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          backend_.view(buffers.ropeSin, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          span.ring[layer].keys, span.ring[layer].values, span.rows,
          span.startPosition, layout.attentionShape());
    }
  }
}

uint32_t DFlashDraft::decodeStorageLanes(uint32_t lanes) const {
  const ops::Linear &linear = operators_.draftLinear();
  const uint32_t rows = lanes * ExecutionLimits::draftQueryRows;
  uint32_t storageRows = rows;
  const auto include = [&](const ops::Projection &p) {
    storageRows = std::max(storageRows, linear.storageRows(rows, p.shape()));
  };
  for (const DFlashDraftLayerWeights &layer : weights_.layers) {
    for (const ops::Projection *p : {&layer.qkvProjection, &layer.outputProjection, &layer.gateProjection,
                                     &layer.upProjection, &layer.downProjection})
      include(*p);
    if (weights_.layout.mixesRows) {
      include(layer.attentionDynamic);
      include(layer.mlpDynamic);
    }
  }
  include(weights_.contextProjection);
  if (weights_.layout.mixesRows) include(weights_.selectorProjection);
  for (const ops::Projection &p : contextKvProjections_) include(p);
  // Its view of the head's rows, on the target's plans as the head (addDecode).
  storageRows = std::max(storageRows, operators_.linear().storageRows(rows, subsetHead_.shape()));
  return storageRows / ExecutionLimits::draftQueryRows;
}

void DFlashDraft::addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                            const DFlashDraftStep &step) const {
  const std::span<const uint32_t> cacheLengths = step.cacheLengths;
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft decode batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::draftQueryRows;
  const auto attentionPlan =
      operators_.draftAttention(layout.attentionShape(), lanes);
  const ops::Linear &linear = operators_.draftLinear();
  const ops::LinearScratch &scratch = buffers.linearScratch;

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const DFlashDraftLayerWeights &weights = weights_.layers[layer];
    if (!layout.mixesRows) {
      addResidualLayer(graph, buffers, weights, attentionPlan, cacheLengths, layer);
      continue;
    }
    const ops::LinearPlan attentionDynamicPlan = linear.plan(weights.attentionDynamic, rows);
    const ops::PreparedInput attentionNormalized = ops::Normalization::addRms(
        graph, buffers.hidden[current], weights.inputNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, attentionDynamicPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.dynamic, .scratch = scratch,
                .prepared = attentionNormalized},
               weights.attentionDynamic, attentionDynamicPlan);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    linear.add(graph, {.input = buffers.convolved, .output = buffers.proposalQkv, .scratch = scratch},
               weights.qkvProjection, linear.plan(weights.qkvProjection, rows));
    ops::DraftAttention::addPrepare(
        graph,
        {buffers.proposalQkv, buffers.attention, weights.queryNorm,
         weights.keyNorm, buffers.ropeCos, buffers.ropeSin, buffers.queryKeys,
         buffers.queryValues},
        attentionPlan);
    ops::DraftAttention::addDecode(
        graph,
        {buffers.attention, buffers.persistentKeys[layer],
         buffers.persistentValues[layer], buffers.queryKeys,
         buffers.queryValues},
        cacheLengths, attentionPlan);
    linear.add(graph, {.input = buffers.attention, .output = buffers.projected, .scratch = scratch},
               weights.outputProjection, linear.plan(weights.outputProjection, rows));
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.residual},
        attentionPlan, ops::DraftConvolutionStage::Residual);
    const ops::LinearPlan mlpDynamicPlan = linear.plan(weights.mlpDynamic, rows);
    const ops::PreparedInput mlpNormalized = ops::Normalization::addRms(
        graph, buffers.residual, weights.postAttentionNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, mlpDynamicPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.dynamic, .scratch = scratch,
                .prepared = mlpNormalized},
               weights.mlpDynamic, mlpDynamicPlan);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    linear.add(graph,
               {.input = buffers.convolved, .output = buffers.intermediate, .gateScratch = buffers.gateScratch,
                .scratch = scratch},
               weights.upProjection,
               linear.plan(weights.upProjection, rows, ops::LinearEpilogue::GateUp, &weights.gateProjection),
               &weights.gateProjection);
    linear.add(graph, {.input = buffers.intermediate, .output = buffers.projected, .scratch = scratch},
               weights.downProjection, linear.plan(weights.downProjection, rows));
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.hidden[next]},
        attentionPlan, ops::DraftConvolutionStage::Residual);
  }

  // The final norm feeds the shared vocabulary head and then the selector;
  // the selector reuses whatever table the head leaves when the layouts match.
  // The head is the target's weights, so it runs on the target's plans, as
  // its scratch is sized (DecodeArena::linearScratchSize): only the draft's
  // own projections keep bf16 activations.
  const ops::Linear &target = operators_.linear();
  const ops::Projection &vocabularyProjection = head(step.vocabulary);
  const ops::LinearPlan headPlan = target.plan(vocabularyProjection, rows);
  const ops::PreparedInput finalHidden = ops::Normalization::addRms(
      graph, buffers.hidden[weights_.layout.layers & 1], weights_.finalNorm,
      buffers.finalHidden, layout.hiddenSize, rows, scratch, headPlan.input());
  const ops::PreparedInput afterHead = target.add(
      graph, {.input = buffers.finalHidden, .output = buffers.logits, .scratch = scratch, .prepared = finalHidden},
      vocabularyProjection, headPlan);
  if (!layout.mixesRows) return;
  linear.add(graph,
             {.input = buffers.finalHidden, .output = buffers.selectorHidden, .scratch = scratch,
              .prepared = afterHead},
             weights_.selectorProjection, linear.plan(weights_.selectorProjection, rows));
}

// A DFlash layer: its attention's and its FFN's output each added to the
// residual by their last projection.
void DFlashDraft::addResidualLayer(metal::CommandGraph &graph, const DFlashDecodeBuffers &buffers,
                                   const DFlashDraftLayerWeights &weights, const ops::DraftAttentionPlan &attentionPlan,
                                   std::span<const uint32_t> cacheLengths, uint32_t layer) const {
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  const uint32_t rows = lanes * ExecutionLimits::draftQueryRows;
  const ops::Linear &linear = operators_.draftLinear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  const metal::MetalBuffer &hidden = buffers.hidden[layer & 1];
  const ops::LinearPlan qkvPlan = linear.plan(weights.qkvProjection, rows);
  const ops::PreparedInput attentionNormalized = ops::Normalization::addRms(
      graph, hidden, weights.inputNorm, buffers.normalized, layout.hiddenSize, rows, scratch, qkvPlan.input());
  linear.add(graph,
             {.input = buffers.normalized, .output = buffers.proposalQkv, .scratch = scratch,
              .prepared = attentionNormalized},
             weights.qkvProjection, qkvPlan);
  ops::DraftAttention::addPrepare(graph,
                                  {buffers.proposalQkv, buffers.attention, weights.queryNorm, weights.keyNorm,
                                   buffers.ropeCos, buffers.ropeSin, buffers.queryKeys, buffers.queryValues},
                                  attentionPlan);
  ops::DraftAttention::addDecode(graph,
                                 {buffers.attention, buffers.persistentKeys[layer], buffers.persistentValues[layer],
                                  buffers.queryKeys, buffers.queryValues},
                                 cacheLengths, attentionPlan, layout.causal(layer));
  linear.add(graph,
             {.input = buffers.attention, .output = buffers.residual, .residual = hidden, .scratch = scratch},
             weights.outputProjection, linear.plan(weights.outputProjection, rows, ops::LinearEpilogue::Residual));
  const ops::LinearPlan gateUpPlan =
      linear.plan(weights.upProjection, rows, ops::LinearEpilogue::GateUp, &weights.gateProjection);
  const ops::PreparedInput mlpNormalized =
      ops::Normalization::addRms(graph, buffers.residual, weights.postAttentionNorm, buffers.normalized,
                                 layout.hiddenSize, rows, scratch, gateUpPlan.input());
  linear.add(graph,
             {.input = buffers.normalized, .output = buffers.intermediate, .gateScratch = buffers.gateScratch,
              .scratch = scratch, .prepared = mlpNormalized},
             weights.upProjection, gateUpPlan, &weights.gateProjection);
  linear.add(graph,
             {.input = buffers.intermediate, .output = buffers.hidden[(layer & 1) ^ 1], .residual = buffers.residual,
              .scratch = scratch},
             weights.downProjection, linear.plan(weights.downProjection, rows, ops::LinearEpilogue::Residual));
}

void DFlashDraft::addMaskRows(metal::CommandGraph &graph, metal::MetalBuffer rows, uint32_t lanes) const {
  if (!weights_.maskEmbedding) return;
  ops::Embedding::addDraftMaskRows(graph, weights_.maskEmbedding, std::move(rows), weights_.layout.hiddenSize,
                                   lanes);
}

void DFlashDraft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions, const ops::StepRows &step) const {
  const uint32_t lanes = static_cast<uint32_t>(startPositions.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || step.lanes != lanes ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft context batch");
  }
  ops::requireStepRows(step);
  const DFlashDraftLayout &layout = weights_.layout;
  // The context projections run over the step's rows.
  const uint32_t rows = step.rows;
  const ops::Linear &linear = operators_.draftLinear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  linear.add(graph, {.input = buffers.capturedTargetHidden, .output = buffers.projected, .scratch = scratch},
             weights_.contextProjection, linear.plan(weights_.contextProjection, rows));
  // Every layer's key and value projection reads the same normalized rows,
  // prepared for the first layer's plan.
  ops::LinearPlan kvPlan = linear.plan(contextKvProjections_[0], rows);
  ops::PreparedInput hidden = ops::Normalization::addRms(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden, layout.hiddenSize, rows,
      scratch, kvPlan.input());
  // A lane holds at most the cap's rows and the step's: a one-lane step of
  // 9 to 15 rows takes the cap of 16.
  for (uint32_t lane = 0; lane < lanes; ++lane)
    if (buffers.windows[lane])
      ops::DraftAttention::addWindowCommit(
          graph, buffers.hidden, buffers.windows[lane],
          backend_.view(buffers.retainedRows, lane * sizeof(uint32_t), sizeof(uint32_t)),
          backend_.view(buffers.laneRows, lane * sizeof(ops::LaneRows), sizeof(ops::LaneRows)), rows,
          std::min(step.laneCap, rows), startPositions[lane], layout.hiddenSize);

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const ops::Projection &projection = contextKvProjections_[layer];
    if (layer) kvPlan = linear.plan(projection, rows);
    hidden = linear.add(graph,
                        {.input = buffers.hidden, .output = buffers.contextKv, .scratch = scratch,
                         .prepared = hidden},
                        projection, kvPlan);
    ops::DraftAttention::addContextCommit(
        graph, buffers.contextKv, weights_.layers[layer].keyNorm, buffers.ropeCos,
        buffers.ropeSin, buffers.persistentKeys[layer],
        buffers.persistentValues[layer], buffers.retainedRows, buffers.laneRows,
        startPositions, step, layout.attentionShape());
  }
}

// Reads a draft's images in their section order: each layer, then model.bin.
DFlashDraftWeights loadDFlashDraftWeights(metal::MetalBackend &backend,
                                          DraftCheckpointLoader &files,
                                          DFlashDraftLayout layout) {
  requireLayout(layout);
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  DFlashDraftWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const uint64_t convolutionBytes = checkedMultiply<WeightStoreError>(
      checkedMultiply<WeightStoreError>(SPLASH_DRAFT_CONVOLUTION_STAGES * SPLASH_DRAFT_CONVOLUTION_TAPS,
                                        layout.hiddenSize, "draft convolution elements"),
      kBFloat16Bytes, "draft convolution bytes");
  const uint64_t headNormBytes = checkedMultiply<WeightStoreError>(
      layout.attentionHeadDimension, kBFloat16Bytes,
      "draft head norm bytes");

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    WeightFile file = files.layer(layerIndex);
    DFlashDraftLayerWeights layer;
    layer.inputNorm = readNorm(file, layout.hiddenSize, false, "input-norm");
    if (layout.mixesRows) {
      layer.attentionConvolution = file.section(convolutionBytes, "attention-convolution");
      layer.attentionDynamic = readBlockProjection(file, layout.dynamicSize, layout.hiddenSize, "attention-dynamic");
    }
    layer.qkvProjection = readBlockProjection(
        file, layout.qkvSize, layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readBlockProjection(
        file, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        readNorm(file, layout.hiddenSize, false, "post-attention-norm");
    if (layout.mixesRows) {
      layer.mlpConvolution = file.section(convolutionBytes, "mlp-convolution");
      layer.mlpDynamic = readBlockProjection(file, layout.dynamicSize, layout.hiddenSize, "mlp-dynamic");
    }
    layer.gateProjection = readBlockProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = readBlockProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = readBlockProjection(
        file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
    file.finish();
    result.files.push_back(file.record());
    result.layers.push_back(std::move(layer));
  }

  {
    WeightFile file = files.model();
    result.contextProjection = readBlockProjection(
        file, layout.hiddenSize, layout.targetHiddenSize(),
        "context-projection");
    result.hiddenNorm = readNorm(file, layout.hiddenSize, false, "hidden-norm");
    result.finalNorm = readNorm(file, layout.hiddenSize, false, "final-norm");
    if (layout.mixesRows) {
      result.selectorProjection = readBlockProjection(file, layout.selectorRank, layout.hiddenSize, "selector");
      const uint64_t codebookBytes = checkedMultiply<WeightStoreError>(
          checkedMultiply<WeightStoreError>(layout.vocabularySize, layout.selectorRank, "draft codebook elements"),
          kBFloat16Bytes, "draft codebook bytes");
      result.predecessorCodebook = file.section(codebookBytes, "predecessor-codebook");
      result.successorCodebook = file.section(codebookBytes, "successor-codebook");
    }
    if (files.hasMaskEmbedding())
      result.maskEmbedding = file.section(uint64_t{layout.hiddenSize} * kBFloat16Bytes, "mask-embedding");
    file.finish();
    result.files.push_back(file.record());
  }

  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
