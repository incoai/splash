#include "DFlashDraft.hpp"
#include "Checked.hpp"
#include "DraftCheckpoint.hpp"
#include "metal/abi/DraftAttention.h"

#include <algorithm>
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
      !layout.kvHeads || !layout.queryRows || !layout.queryHeads) {
    throw WeightStoreError("DFlash draft layout contains a zero dimension");
  }
  const bool legacy = layout.queryRows == SPLASH_DRAFT_QUERY_ROWS &&
                      layout.queryHeads == SPLASH_DRAFT_QUERY_HEADS &&
                      layout.kvHeads == SPLASH_DRAFT_KV_HEADS &&
                      layout.attentionHeadDimension == SPLASH_DRAFT_HEAD_DIMENSION;
  const bool qwen4 = layout.queryRows == SPLASH_DRAFT4_QUERY_ROWS &&
                     layout.queryHeads == SPLASH_DRAFT4_QUERY_HEADS &&
                     layout.kvHeads == SPLASH_DRAFT4_KV_HEADS &&
                     layout.attentionHeadDimension == SPLASH_DRAFT4_HEAD_DIMENSION;
  if (!legacy && !qwen4)
    throw WeightStoreError("DFlash draft attention shape is not compiled");
  if (layout.selectorRank != SPLASH_DRAFT_SELECTOR_RANK) {
    throw WeightStoreError("draft selector kernels are compiled for rank " +
                           std::to_string(SPLASH_DRAFT_SELECTOR_RANK));
  }
}

// The key and value rows of each layer's fused QKV projection, without a
// copy: planes store whole tiles of QUANT_TILE_ROWS rows in row order
// (metal/abi/QuantFormat.h), so rows from a tile boundary on are one range of
// each plane.
std::vector<ops::Projection> contextKvRows(metal::MetalBackend &backend,
                                           const DFlashDraftWeights &weights) {
  const DFlashDraftLayout &layout = weights.layout;
  if (layout.attentionSize >= layout.qkvSize ||
      layout.attentionSize % QUANT_TILE_ROWS) {
    throw std::invalid_argument(
        "draft key and value rows do not start at a storage tile");
  }
  std::vector<ops::Projection> result;
  result.reserve(weights.layers.size());
  for (const DFlashDraftLayerWeights &layer : weights.layers) {
    const ops::QuantizedSegment &fused = layer.qkvProjection.blocks().segments.front();
    const auto rows = [&](const metal::MetalBuffer &plane) {
      if (!plane) return metal::MetalBuffer{};
      const uint64_t rowBytes = plane.sizeBytes() / layout.qkvSize;
      return backend.view(plane, uint64_t{layout.attentionSize} * rowBytes,
                          uint64_t{layout.contextKvSize()} * rowBytes);
    };
    result.emplace_back(layout.contextKvSize(), layout.hiddenSize,
                        ops::BlockWeights{{ops::QuantizedSegment::planes(
                            fused.formatId, layout.contextKvSize(), layout.hiddenSize, rows(fused.plane0),
                            rows(fused.plane1), rows(fused.meta))}});
  }
  return result;
}

} // namespace

DFlashDraftLayout draftLayout(uint32_t hiddenSize, uint32_t vocabularySize, uint32_t layers,
                              uint32_t intermediateSize, float rotaryTheta, std::span<const uint32_t> targetLayers) {
  return draftLayout(hiddenSize, vocabularySize, layers, intermediateSize, rotaryTheta, targetLayers, {});
}

DFlashDraftLayout draftLayout(uint32_t hiddenSize, uint32_t vocabularySize, uint32_t layers,
                              uint32_t intermediateSize, float rotaryTheta, std::span<const uint32_t> targetLayers,
                              DFlashDraftAttention attention) {
  constexpr uint32_t most = ExecutionLimits::maximumDraftTargetLayers;
  if (targetLayers.empty() || targetLayers.size() > most ||
      std::ranges::adjacent_find(targetLayers, std::ranges::greater_equal{}) != targetLayers.end())
    throw std::invalid_argument("a DFlash2 draft reads 1 to " + std::to_string(most) +
                                " target layers, ascending");
  if (!layers) throw std::invalid_argument("a DFlash2 draft has at least one layer");
  // Its projections quantize in groups of 64 columns (DraftCheckpoint.cpp).
  if (!intermediateSize || intermediateSize % 64)
    throw std::invalid_argument("a DFlash2 draft's intermediate size is a positive multiple of 64");
  if (!(rotaryTheta > 0)) throw std::invalid_argument("a DFlash2 draft's rotary theta is positive");
  if (!attention.queryRows || attention.queryRows % SPLASH_TARGET_VERIFY_ROWS ||
      attention.queryRows > SPLASH_DRAFT_MAXIMUM_QUERY_ROWS)
    throw std::invalid_argument("a DFlash2 draft block must hold 8 or 16 rows");
  const uint64_t attentionSize = uint64_t{attention.queryHeads} * attention.headDimension;
  const uint64_t kvSize = uint64_t{attention.kvHeads} * attention.headDimension;
  if (!attentionSize || !kvSize || attentionSize > UINT32_MAX ||
      attentionSize + 2 * kvSize > UINT32_MAX)
    throw std::invalid_argument("a DFlash2 draft attention width is invalid");
  DFlashDraftLayout layout{.layers = layers,
                           .hiddenSize = hiddenSize,
                           .vocabularySize = vocabularySize,
                           .dynamicSize = draft_dynamic_width(hiddenSize),
                           .qkvSize = static_cast<uint32_t>(attentionSize + 2 * kvSize),
                           .attentionSize = static_cast<uint32_t>(attentionSize),
                           .intermediateSize = intermediateSize,
                           .attentionHeadDimension = attention.headDimension,
                           .rotaryTheta = rotaryTheta,
                           .selectorRank = SPLASH_DRAFT_SELECTOR_RANK,
                           .kvHeads = attention.kvHeads,
                           .queryRows = attention.queryRows,
                           .queryHeads = attention.queryHeads,
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
                         const ops::ExecutionPlans &operators)
    : weights_(weights), backend_(backend), operators_(operators),
      selector_(weights.layout.vocabularySize),
      contextKvProjections_(contextKvRows(backend, weights)) {}

void DFlashDraft::addSelection(
    metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
    std::span<const uint32_t> anchors,
    std::span<const ops::SamplingPolicy> policies) const {
  selector_.add(graph, buffers,
                {weights_.predecessorCodebook, weights_.successorCodebook},
                anchors, policies, weights_.layout.queryRows);
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
  operators_.linear().addPrefill(graph, buffers.capturedTargetHidden, weights_.contextProjection,
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
    operators_.linear().addPrefill(graph, buffers.hidden, contextKvProjections_[layer], buffers.contextKv, rows,
                                   buffers.linearScratch);
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
  const ops::Linear &linear = operators_.linear();
  const uint32_t queryRows = weights_.layout.queryRows;
  const uint32_t chunkLanes =
      queryRows == SPLASH_DRAFT_QUERY_ROWS ? lanes : std::min(lanes, 2U);
  uint32_t storageLanes = lanes;
  const auto include = [&](const ops::Projection &p, uint32_t offset,
                           uint32_t count) {
    const uint32_t rows = count * queryRows;
    const uint32_t storageRows =
        std::max(rows, linear.decodeStorageRows(rows, p.shape()));
    const uint32_t needed =
        offset + (storageRows + queryRows - 1) / queryRows;
    storageLanes = std::max(storageLanes, needed);
  };
  for (uint32_t offset = 0; offset < lanes; offset += chunkLanes) {
    const uint32_t count = std::min(chunkLanes, lanes - offset);
    for (const DFlashDraftLayerWeights &layer : weights_.layers)
      for (const ops::Projection *p :
           {&layer.attentionDynamic, &layer.qkvProjection,
            &layer.outputProjection, &layer.mlpDynamic,
            &layer.gateProjection, &layer.upProjection,
            &layer.downProjection})
        include(*p, offset, count);
    include(weights_.contextProjection, offset, count);
    include(weights_.selectorProjection, offset, count);
    for (const ops::Projection &p : contextKvProjections_)
      include(p, offset, count);
  }
  return storageLanes;
}

void DFlashDraft::addDecode(
    metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
    const ops::Projection &vocabularyProjection,
    std::span<const uint32_t> cacheLengths) const {
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft decode batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const auto attentionPlan =
      operators_.draftAttention(layout.attentionShape(), lanes);
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  // The linear kernels take at most four eight-row tiles. A 16-row draft
  // block of more than two lanes is computed as two contiguous lane chunks;
  // all rows remain in the lane-major buffers for the block-wide attention.
  const uint32_t chunkLanes =
      layout.queryRows == SPLASH_DRAFT_QUERY_ROWS ? lanes : std::min(lanes, 2U);
  const auto forChunks = [&](auto &&fn) {
    for (uint32_t lane = 0; lane < lanes; lane += chunkLanes) {
      const uint32_t count = std::min(chunkLanes, lanes - lane);
      fn(lane * layout.queryRows, count * layout.queryRows, count);
    }
  };
  const auto bf16Rows = [&](const metal::MetalBuffer &buffer,
                            uint32_t firstRow, uint32_t count,
                            uint32_t width) {
    const uint64_t rowBytes = uint64_t{width} * sizeof(uint16_t);
    return backend_.view(buffer, uint64_t{firstRow} * rowBytes,
                         uint64_t{count} * rowBytes);
  };
  const auto floatRows = [&](const metal::MetalBuffer &buffer,
                             uint32_t firstRow, uint32_t count,
                             uint32_t width) {
    const uint64_t rowBytes = uint64_t{width} * sizeof(float);
    return backend_.view(buffer, uint64_t{firstRow} * rowBytes,
                         uint64_t{count} * rowBytes);
  };

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const DFlashDraftLayerWeights &weights = weights_.layers[layer];
    forChunks([&](uint32_t first, uint32_t count, uint32_t countLanes) {
      const ops::LinearPlan plan = linear.decodePlan(
          weights.attentionDynamic,
          countLanes * layout.queryRows / SPLASH_TARGET_VERIFY_ROWS);
      const metal::MetalBuffer hidden =
          bf16Rows(buffers.hidden[current], first, count, layout.hiddenSize);
      const metal::MetalBuffer normalized =
          bf16Rows(buffers.normalized, first, count, layout.hiddenSize);
      const ops::PreparedInput prepared = ops::Normalization::addRms(
          graph, hidden, weights.inputNorm, normalized, layout.hiddenSize,
          count, scratch, plan.input());
      linear.add(graph,
                 {.input = normalized,
                  .output = bf16Rows(buffers.dynamic, first, count,
                                     layout.dynamicSize),
                  .scratch = scratch, .prepared = prepared},
                 weights.attentionDynamic, plan);
    });
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    forChunks([&](uint32_t first, uint32_t count, uint32_t countLanes) {
      linear.add(
          graph,
          {.input = bf16Rows(buffers.convolved, first, count,
                             layout.hiddenSize),
           .output = bf16Rows(buffers.proposalQkv, first, count,
                              layout.qkvSize),
           .scratch = scratch},
          weights.qkvProjection,
          linear.decodePlan(weights.qkvProjection,
                            countLanes * layout.queryRows /
                                SPLASH_TARGET_VERIFY_ROWS));
    });
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
    ops::DraftAttention::addReorder(graph, buffers.attention,
                                    buffers.proposalQkv, attentionPlan);
    forChunks([&](uint32_t first, uint32_t count, uint32_t countLanes) {
      linear.add(
          graph,
          {.input = bf16Rows(buffers.proposalQkv, first, count,
                             layout.attentionSize),
           .output = bf16Rows(buffers.projected, first, count,
                              layout.hiddenSize),
           .scratch = scratch},
          weights.outputProjection,
          linear.decodePlan(weights.outputProjection,
                            countLanes * layout.queryRows /
                                SPLASH_TARGET_VERIFY_ROWS));
    });
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.residual},
        attentionPlan, ops::DraftConvolutionStage::Residual);
    forChunks([&](uint32_t first, uint32_t count, uint32_t countLanes) {
      const ops::LinearPlan plan = linear.decodePlan(
          weights.mlpDynamic,
          countLanes * layout.queryRows / SPLASH_TARGET_VERIFY_ROWS);
      const metal::MetalBuffer residual =
          bf16Rows(buffers.residual, first, count, layout.hiddenSize);
      const metal::MetalBuffer normalized =
          bf16Rows(buffers.normalized, first, count, layout.hiddenSize);
      const ops::PreparedInput prepared = ops::Normalization::addRms(
          graph, residual, weights.postAttentionNorm, normalized,
          layout.hiddenSize, count, scratch, plan.input());
      linear.add(graph,
                 {.input = normalized,
                  .output = bf16Rows(buffers.dynamic, first, count,
                                     layout.dynamicSize),
                  .scratch = scratch, .prepared = prepared},
                 weights.mlpDynamic, plan);
    });
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    forChunks([&](uint32_t first, uint32_t count, uint32_t countLanes) {
      const uint32_t planLanes =
          countLanes * layout.queryRows / SPLASH_TARGET_VERIFY_ROWS;
      linear.add(
          graph,
          {.input = bf16Rows(buffers.convolved, first, count,
                             layout.hiddenSize),
           .output = bf16Rows(buffers.intermediate, first, count,
                              layout.intermediateSize),
           .gateScratch = bf16Rows(buffers.gateScratch, first, count,
                                   layout.intermediateSize),
           .scratch = scratch},
          weights.upProjection,
          linear.decodePlan(weights.upProjection, planLanes,
                            ops::LinearEpilogue::GateUp,
                            &weights.gateProjection),
          &weights.gateProjection);
      linear.add(
          graph,
          {.input = bf16Rows(buffers.intermediate, first, count,
                             layout.intermediateSize),
           .output = bf16Rows(buffers.projected, first, count,
                              layout.hiddenSize),
           .scratch = scratch},
          weights.downProjection,
          linear.decodePlan(weights.downProjection, planLanes));
    });
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.hidden[next]},
        attentionPlan, ops::DraftConvolutionStage::Residual);
  }

  // The final norm feeds the shared vocabulary head and then the selector;
  // the selector reuses whatever table the head leaves when the layouts match.
  forChunks([&](uint32_t first, uint32_t count, uint32_t countLanes) {
    const uint32_t planLanes =
        countLanes * layout.queryRows / SPLASH_TARGET_VERIFY_ROWS;
    const ops::LinearPlan headPlan =
        linear.decodePlan(vocabularyProjection, planLanes);
    const metal::MetalBuffer hidden = bf16Rows(
        buffers.hidden[weights_.layout.layers & 1], first, count,
        layout.hiddenSize);
    const metal::MetalBuffer finalHidden =
        bf16Rows(buffers.finalHidden, first, count, layout.hiddenSize);
    const ops::PreparedInput prepared = ops::Normalization::addRms(
        graph, hidden, weights_.finalNorm, finalHidden, layout.hiddenSize,
        count, scratch, headPlan.input());
    const ops::PreparedInput afterHead = linear.add(
        graph,
        {.input = finalHidden,
         .output = floatRows(buffers.logits, first, count,
                             layout.vocabularySize),
         .scratch = scratch, .prepared = prepared},
        vocabularyProjection, headPlan);
    linear.add(
        graph,
        {.input = finalHidden,
         .output = bf16Rows(buffers.selectorHidden, first, count,
                            layout.selectorRank),
         .scratch = scratch, .prepared = afterHead},
        weights_.selectorProjection,
        linear.decodePlan(weights_.selectorProjection, planLanes));
  });
}

void DFlashDraft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions) const {
  const uint32_t lanes = static_cast<uint32_t>(startPositions.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft context batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  linear.add(graph, {.input = buffers.capturedTargetHidden, .output = buffers.projected, .scratch = scratch},
             weights_.contextProjection, linear.decodePlan(weights_.contextProjection, lanes));
  // Every layer's key and value projection reads the same normalized rows,
  // prepared for the first layer's plan.
  ops::LinearPlan kvPlan = linear.decodePlan(contextKvProjections_[0], lanes);
  ops::PreparedInput hidden = ops::Normalization::addRms(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden, layout.hiddenSize, rows,
      scratch, kvPlan.input());

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const ops::Projection &projection = contextKvProjections_[layer];
    if (layer) kvPlan = linear.decodePlan(projection, lanes);
    hidden = linear.add(graph,
                        {.input = buffers.hidden, .output = buffers.contextKv, .scratch = scratch,
                         .prepared = hidden},
                        projection, kvPlan);
    ops::DraftAttention::addContextCommit(
        graph, buffers.contextKv, weights_.layers[layer].keyNorm, buffers.ropeCos,
        buffers.ropeSin, buffers.persistentKeys[layer],
        buffers.persistentValues[layer], buffers.retainedCounts,
        startPositions, layout.attentionShape());
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
    layer.attentionConvolution =
        file.section(convolutionBytes, "attention-convolution");
    layer.attentionDynamic = readBlockProjection(
        file, layout.dynamicSize, layout.hiddenSize,
        "attention-dynamic");
    layer.qkvProjection = readBlockProjection(
        file, layout.qkvSize, layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readBlockProjection(
        file, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        readNorm(file, layout.hiddenSize, false, "post-attention-norm");
    layer.mlpConvolution = file.section(convolutionBytes, "mlp-convolution");
    layer.mlpDynamic = readBlockProjection(
        file, layout.dynamicSize, layout.hiddenSize, "mlp-dynamic");
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
    result.selectorProjection = readBlockProjection(
        file, layout.selectorRank, layout.hiddenSize, "selector");
    const uint64_t codebookBytes = checkedMultiply<WeightStoreError>(
        checkedMultiply<WeightStoreError>(layout.vocabularySize,
                                          layout.selectorRank,
                                          "draft codebook elements"),
        kBFloat16Bytes, "draft codebook bytes");
    result.predecessorCodebook =
        file.section(codebookBytes, "predecessor-codebook");
    result.successorCodebook =
        file.section(codebookBytes, "successor-codebook");
    file.finish();
    result.files.push_back(file.record());
  }

  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
