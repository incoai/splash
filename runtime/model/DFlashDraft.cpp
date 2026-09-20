#include "DFlashDraft.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

void validateLayout(const DFlashDraftLayout &layout) {
  if (!layout.layers || !layout.hiddenSize || !layout.vocabularySize ||
      !layout.dynamicSize || !layout.qkvSize || !layout.attentionSize ||
      !layout.intermediateSize || !layout.attentionHeadDimension ||
      !(layout.rotaryTheta > 0.0F) ||
      !layout.targetHiddenSize || !layout.selectorRank ||
      !layout.kvHeads) {
    throw WeightStoreError("DFlash draft layout contains a zero dimension");
  }
  if (layout.selectorRank != 256) {
    throw WeightStoreError(
        "draft selector kernels are compiled for rank 256");
  }
  validateQ4Layout(layout.dynamicSize, layout.hiddenSize);
  validateQ4Layout(layout.qkvSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionSize);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.hiddenSize, layout.targetHiddenSize);
  validateQ4Layout(layout.selectorRank, layout.hiddenSize);
}

} // namespace

DFlashDraftRing::DFlashDraftRing(
    metal::MetalBackend &backend, std::shared_ptr<StateAllocationTracker> tracker,
    DraftStateLayout layout, std::string_view label)
    : tracker_(std::move(tracker)), layers_(layout.layers) {
  if (!tracker_)
    throw std::invalid_argument("draft state allocation tracker is empty");
  if (!layout.valid() ||
      layout.tokens != ExecutionLimits::draftContextTokens) {
    throw std::invalid_argument("draft state layout is invalid");
  }
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
      selector_(backend, weights.layout.vocabularySize,
                ExecutionLimits::draftQueryRows) {
  validateLayout(weights_.layout);
  if (weights_.layers.size() != weights_.layout.layers ||
      !weights_.layout.stateLayout().valid()) {
    throw std::invalid_argument("draft weights do not match state geometry");
  }
}

void DFlashDraft::addSelection(
    metal::CommandGraph &graph, DFlashSelectionBuffers buffers,
    std::span<const uint32_t> anchors,
    std::span<const ops::SamplingPolicy> policies, uint32_t proposalTokens) const {
  selector_.addDraftSelector(
      graph,
      {std::move(buffers.logits), std::move(buffers.partialIds),
       std::move(buffers.partialValues), std::move(buffers.candidates),
       std::move(buffers.unary), std::move(buffers.selectorHidden),
       weights_.predecessorCodebook, weights_.successorCodebook,
       std::move(buffers.uniforms), std::move(buffers.proposedTokens),
       std::move(buffers.proposalProbabilities)},
      anchors, policies, proposalTokens);
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
  const ops::LinearMatrix context{layout.hiddenSize,
                                    layout.targetHiddenSize};
  operators_.linear().addPrefillSums(graph, buffers.capturedTargetHidden,
                     buffers.projectionSums, context, rows);
  operators_.linear().addPrefill(graph, buffers.capturedTargetHidden,
                 weights_.contextProjection, buffers.projected,
                 buffers.projectionSums, context, rows);
  ops::Normalization::addRmsWithQ4Sums(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden,
      buffers.projectionSums, layout.hiddenSize, rows);

  const ops::LinearMatrix qkv{layout.qkvSize, layout.hiddenSize};
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    operators_.linear().addPrefill(graph, buffers.hidden,
                      weights_.layers[layer].qkvProjection, buffers.qkv,
                      buffers.projectionSums, qkv, rows);
    for (const DFlashPrefillSpan &span : spans) {
      const uint64_t qkvOffset =
          uint64_t{span.compactRow} * layout.qkvSize * sizeof(uint16_t);
      const uint64_t ropeOffset =
          uint64_t{span.compactRow} * (layout.attentionHeadDimension / 2) * sizeof(float);
      ops::DraftAttention::addContextPrefill(
          graph,
          backend_.view(buffers.qkv, qkvOffset,
                        uint64_t{span.rows} * layout.qkvSize *
                            sizeof(uint16_t)),
          weights_.layers[layer].keyNorm,
          backend_.view(buffers.ropeCos, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          backend_.view(buffers.ropeSin, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          span.ring[layer].keys, span.ring[layer].values, span.rows,
          layout.stateLayout().tokens, span.startPosition,
          layout.attentionShape());
    }
  }
}

void DFlashDraft::addDecode(
    metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
    const ops::VocabularyProjection &vocabularyProjection,
    std::span<const uint32_t> cacheLengths, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      cacheLengths.size() != ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft decode batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::draftQueryRows;
  const auto attentionPlan =
      operators_.draftAttention(layout.attentionShape(), lanes);
  const ops::LinearMatrix qkv{layout.qkvSize, layout.hiddenSize};
  const ops::LinearMatrix dynamic{layout.dynamicSize, layout.hiddenSize};
  const ops::LinearMatrix output{layout.hiddenSize, layout.attentionSize};
  const ops::LinearMatrix gateUp{layout.intermediateSize, layout.hiddenSize};
  const ops::LinearMatrix down{layout.hiddenSize, layout.intermediateSize};

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const DFlashDraftLayerWeights &weights = weights_.layers[layer];
    ops::Normalization::addRms(graph, buffers.hidden[current],
                               weights.inputNorm,
                               buffers.normalized, layout.hiddenSize, rows, buffers.linearScratch);
    operators_.linear().addDecodeBatch(graph,
                       buffers.normalized, weights.attentionDynamic,
                       buffers.dynamic, dynamic, lanes, stats, buffers.linearScratch, true);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    operators_.linear().addDecodeBatch(graph, buffers.convolved,
                       weights.qkvProjection, buffers.proposalQkv, qkv, lanes,
                       stats, buffers.linearScratch);
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
        cacheLengths, layout.stateLayout().tokens, attentionPlan);
    ops::DraftAttention::addReorder(graph, buffers.attention,
                                    buffers.proposalQkv, attentionPlan);
    operators_.linear().addDecodeBatch(graph,
                       buffers.proposalQkv, weights.outputProjection,
                       buffers.projected, output, lanes, stats, buffers.linearScratch);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.residual},
        attentionPlan, ops::DraftConvolutionStage::Residual);
    ops::Normalization::addRms(graph, buffers.residual,
                               weights.postAttentionNorm, buffers.normalized,
                               layout.hiddenSize, rows, buffers.linearScratch);
    operators_.linear().addDecodeBatch(graph,
                       buffers.normalized, weights.mlpDynamic, buffers.dynamic,
                       dynamic, lanes, stats, buffers.linearScratch, true);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    operators_.linear().addGateUpBatch(graph, buffers.convolved, weights.gateProjection,
                       weights.upProjection, buffers.gateScratch,
                       buffers.intermediate, gateUp, lanes, stats, buffers.linearScratch);
    operators_.linear().addDecodeBatch(graph,
                       buffers.intermediate, weights.downProjection,
                       buffers.projected, down, lanes, stats, buffers.linearScratch);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.hidden[next]},
        attentionPlan, ops::DraftConvolutionStage::Residual);
  }

  ops::Normalization::addRms(graph,
                             buffers.hidden[weights_.layout.layers & 1],
                             weights_.finalNorm,
                             buffers.finalHidden, layout.hiddenSize, rows, buffers.linearScratch);
  const ops::LinearMatrix head{layout.vocabularySize, layout.hiddenSize};
  std::visit([&](const auto &vocab) {
    operators_.linear().addDecodeBatch(graph, buffers.finalHidden,
                       vocab, buffers.logits, head, lanes, stats, buffers.linearScratch, true);
  }, vocabularyProjection);
  const ops::LinearMatrix selector{layout.selectorRank, layout.hiddenSize};
  operators_.linear().addDecodeBatch(graph,
                     buffers.finalHidden, weights_.selectorProjection,
                     buffers.selectorHidden, selector, lanes, stats, buffers.linearScratch, true);
}

void DFlashDraft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      startPositions.size() != ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft context batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  const ops::LinearMatrix context{layout.hiddenSize, layout.targetHiddenSize};
  operators_.linear().addDecodeBatch(graph,
                     buffers.capturedTargetHidden, weights_.contextProjection,
                     buffers.projected, context, lanes, stats, buffers.linearScratch);
  ops::Normalization::addRms(graph, buffers.projected, weights_.hiddenNorm,
                             buffers.hidden, layout.hiddenSize, rows, buffers.linearScratch);

  const ops::LinearMatrix qkv{layout.qkvSize, layout.hiddenSize};
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    operators_.linear().addDecodeBatch(graph, buffers.hidden,
                       weights_.layers[layer].qkvProjection, buffers.qkv, qkv,
                       lanes, stats, buffers.linearScratch, true);
    ops::DraftAttention::addContextCommit(
        graph, buffers.qkv, weights_.layers[layer].keyNorm, buffers.ropeCos,
        buffers.ropeSin, buffers.persistentKeys[layer],
        buffers.persistentValues[layer], buffers.retainedCounts,
        startPositions, layout.stateLayout().tokens, layout.attentionShape(),
        lanes);
  }
}

DFlashDraftWeights
loadDFlashDraftWeights(metal::MetalBackend &backend,
                       const std::filesystem::path &directory,
                       DFlashDraftLayout layout) {
  validateLayout(layout);
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  DFlashDraftWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const uint64_t hiddenBytes = checkedWeightMultiply(
      layout.hiddenSize, kBFloat16Bytes, "draft norm bytes");
  const uint64_t convolutionBytes = checkedWeightMultiply(
      checkedWeightMultiply(4, layout.hiddenSize,
                            "draft convolution elements"),
      kBFloat16Bytes, "draft convolution bytes");
  const uint64_t headNormBytes = checkedWeightMultiply(
      layout.attentionHeadDimension, kBFloat16Bytes,
      "draft head norm bytes");

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const std::string filename =
        "layer-" + std::to_string(layerIndex) + ".bin";
    WeightFile file(backend, directory / filename, "draft/" + filename,
                    kDFlashLayerMagic, layerIndex, 0);
    DFlashDraftLayerWeights layer;
    layer.inputNorm = file.section(hiddenBytes, "input-norm");
    layer.attentionConvolution =
        file.section(convolutionBytes, "attention-convolution");
    layer.attentionDynamic = readQ4Projection(
        file, backend, layout.dynamicSize, layout.hiddenSize,
        "attention-dynamic");
    layer.qkvProjection = readQ4Projection(
        file, backend, layout.qkvSize, layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readQ4Projection(
        file, backend, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        file.section(hiddenBytes, "post-attention-norm");
    layer.mlpConvolution = file.section(convolutionBytes, "mlp-convolution");
    layer.mlpDynamic = readQ4Projection(
        file, backend, layout.dynamicSize, layout.hiddenSize, "mlp-dynamic");
    layer.gateProjection = readQ4Projection(
        file, backend, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = readQ4Projection(
        file, backend, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = readQ4Projection(file, backend, layout.hiddenSize,
                                            layout.intermediateSize,
                                            "mlp-down");
    file.finish();
    result.files.push_back(file.record());
    result.layers.push_back(std::move(layer));
  }

  {
    WeightFile file(backend, directory / "model.bin", "draft/model.bin",
                    kDFlashLayerMagic, layout.layers, 1);
    result.contextProjection = readQ4Projection(
        file, backend, layout.hiddenSize, layout.targetHiddenSize,
        "context-projection");
    result.hiddenNorm = file.section(hiddenBytes, "hidden-norm");
    result.finalNorm = file.section(hiddenBytes, "final-norm");
    result.selectorProjection = readQ4Projection(
        file, backend, layout.selectorRank, layout.hiddenSize, "selector");
    const uint64_t codebookBytes = checkedWeightMultiply(
        checkedWeightMultiply(layout.vocabularySize, layout.selectorRank,
                              "draft codebook elements"),
        kBFloat16Bytes, "draft codebook bytes");
    result.predecessorCodebook =
        file.section(codebookBytes, "predecessor-codebook");
    result.successorCodebook =
        file.section(codebookBytes, "successor-codebook");
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
