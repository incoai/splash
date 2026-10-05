#include "DFlashDraft.hpp"
#include "Checked.hpp"
#include "DraftCheckpoint.hpp"
#include "metal/abi/DraftAttention.h"

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
      !layout.targetHiddenSize || !layout.selectorRank ||
      !layout.kvHeads) {
    throw WeightStoreError("DFlash draft layout contains a zero dimension");
  }
  if (layout.selectorRank != SPLASH_DRAFT_SELECTOR_RANK) {
    throw WeightStoreError("draft selector kernels are compiled for rank " +
                           std::to_string(SPLASH_DRAFT_SELECTOR_RANK));
  }
  validateQ4Layout(layout.dynamicSize, layout.hiddenSize);
  validateQ4Layout(layout.qkvSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionSize);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.hiddenSize, layout.targetHiddenSize);
  validateQ4Layout(layout.selectorRank, layout.hiddenSize);
}

// The key and value rows of each layer's fused QKV projection, without a
// copy: affine planes store whole tiles of kQ4StorageN rows in row order
// (AffinePreparation), so rows from a tile boundary on are one range of
// each plane.
std::vector<ops::Projection> contextKvRows(metal::MetalBackend &backend,
                                           const DFlashDraftWeights &weights) {
  const DFlashDraftLayout &layout = weights.layout;
  if (layout.attentionSize >= layout.qkvSize ||
      layout.attentionSize % kQ4StorageN) {
    throw std::invalid_argument(
        "draft key and value rows do not start at a storage tile");
  }
  std::vector<ops::Projection> result;
  result.reserve(weights.layers.size());
  for (const DFlashDraftLayerWeights &layer : weights.layers) {
    const ops::AffineWeights &fused = layer.qkvProjection.affine();
    const auto rows = [&](const metal::MetalBuffer &plane) {
      const uint64_t rowBytes = plane.sizeBytes() / layout.qkvSize;
      return backend.view(plane, uint64_t{layout.attentionSize} * rowBytes,
                          uint64_t{layout.contextKvSize()} * rowBytes);
    };
    result.emplace_back(layout.contextKvSize(), layout.hiddenSize,
                        ops::AffineWeights{rows(fused.weights),
                                           rows(fused.scales),
                                           rows(fused.biases)});
  }
  return result;
}

} // namespace

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
                anchors, policies);
}

void DFlashDraft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
    std::span<const DFlashPrefillSpan> spans) const {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget || spans.empty())
    throw std::invalid_argument("invalid draft context prefill");
  const DFlashDraftLayout &layout = weights_.layout;
  for (const DFlashPrefillSpan &span : spans) {
    if (span.ring.size() != layout.layers)
      throw std::invalid_argument("draft prefill ring layer mismatch");
  }
  operators_.linear().addPrefillSums(graph, buffers.capturedTargetHidden, buffers.projectionSums,
                                     weights_.contextProjection, rows);
  operators_.linear().addPrefill(graph, buffers.capturedTargetHidden, weights_.contextProjection,
                                 buffers.projected, buffers.projectionSums, rows);
  ops::Normalization::addRmsWithQ4Sums(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden,
      buffers.projectionSums, layout.hiddenSize, rows);

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    operators_.linear().addPrefill(graph, buffers.hidden,
                      contextKvProjections_[layer],
                      buffers.contextKv, buffers.projectionSums, rows);
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
  const uint32_t rows = lanes * ExecutionLimits::draftQueryRows;
  const auto attentionPlan =
      operators_.draftAttention(layout.attentionShape(), lanes);
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const DFlashDraftLayerWeights &weights = weights_.layers[layer];
    const ops::LinearPlan attentionDynamicPlan = linear.decodePlan(weights.attentionDynamic, lanes);
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
               weights.qkvProjection, linear.decodePlan(weights.qkvProjection, lanes));
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
    linear.add(graph, {.input = buffers.proposalQkv, .output = buffers.projected, .scratch = scratch},
               weights.outputProjection, linear.decodePlan(weights.outputProjection, lanes));
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.residual},
        attentionPlan, ops::DraftConvolutionStage::Residual);
    const ops::LinearPlan mlpDynamicPlan = linear.decodePlan(weights.mlpDynamic, lanes);
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
               linear.decodePlan(weights.upProjection, lanes, ops::LinearEpilogue::GateUp, &weights.gateProjection),
               &weights.gateProjection);
    linear.add(graph, {.input = buffers.intermediate, .output = buffers.projected, .scratch = scratch},
               weights.downProjection, linear.decodePlan(weights.downProjection, lanes));
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.hidden[next]},
        attentionPlan, ops::DraftConvolutionStage::Residual);
  }

  // The final norm feeds the shared vocabulary head and then the selector;
  // the selector reuses whatever table the head leaves when the layouts match.
  const ops::LinearPlan headPlan = linear.decodePlan(vocabularyProjection, lanes);
  const ops::PreparedInput finalHidden = ops::Normalization::addRms(
      graph, buffers.hidden[weights_.layout.layers & 1], weights_.finalNorm,
      buffers.finalHidden, layout.hiddenSize, rows, scratch, headPlan.input());
  const ops::PreparedInput afterHead = linear.add(
      graph, {.input = buffers.finalHidden, .output = buffers.logits, .scratch = scratch, .prepared = finalHidden},
      vocabularyProjection, headPlan);
  linear.add(graph,
             {.input = buffers.finalHidden, .output = buffers.selectorHidden, .scratch = scratch,
              .prepared = afterHead},
             weights_.selectorProjection, linear.decodePlan(weights_.selectorProjection, lanes));
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

namespace {

// Reads a draft's files in their section order: each layer, then model.bin.
template <class Files>
DFlashDraftWeights readDraft(metal::MetalBackend &backend, Files &files,
                             const DFlashDraftLayout &layout) {
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
    layer.attentionDynamic = readAffineProjection(
        file, layout.dynamicSize, layout.hiddenSize,
        "attention-dynamic");
    layer.qkvProjection = readAffineProjection(
        file, layout.qkvSize, layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readAffineProjection(
        file, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        readNorm(file, layout.hiddenSize, false, "post-attention-norm");
    layer.mlpConvolution = file.section(convolutionBytes, "mlp-convolution");
    layer.mlpDynamic = readAffineProjection(
        file, layout.dynamicSize, layout.hiddenSize, "mlp-dynamic");
    layer.gateProjection = readAffineProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = readAffineProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = readAffineProjection(
        file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
    file.finish();
    result.files.push_back(file.record());
    result.layers.push_back(std::move(layer));
  }

  {
    WeightFile file = files.model();
    result.contextProjection = readAffineProjection(
        file, layout.hiddenSize, layout.targetHiddenSize,
        "context-projection");
    result.hiddenNorm = readNorm(file, layout.hiddenSize, false, "hidden-norm");
    result.finalNorm = readNorm(file, layout.hiddenSize, false, "final-norm");
    result.selectorProjection = readAffineProjection(
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

} // namespace

WeightFile PackedDraftFiles::layer(uint32_t index) const {
  const std::string filename = "layer-" + std::to_string(index) + ".bin";
  return images.load(packedImage(directory / filename, "draft/" + filename,
                                 kDFlashLayerMagic, index, 0));
}

WeightFile PackedDraftFiles::model() const {
  return images.load(packedImage(directory / "model.bin", "draft/model.bin",
                                 kDFlashLayerMagic, layout.layers, 1));
}

DFlashDraftWeights loadDFlashDraftWeights(metal::MetalBackend &backend,
                                          const DraftFiles &files,
                                          DFlashDraftLayout layout) {
  requireLayout(layout);
  if (const auto *checkpoint =
          std::get_if<std::reference_wrapper<DraftCheckpointLoader>>(&files))
    return readDraft(backend, checkpoint->get(), layout);
  return readDraft(backend, std::get<PackedDraftFiles>(files), layout);
}

} // namespace splash::model
