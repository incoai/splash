#include "model/QwenTarget.hpp"

#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/Qwen4Exp.hpp"
#include "ops/Qwen4.hpp"
#include "metal/abi/Qwen4.h"
#include "model/WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace splash::model {
namespace {

template <class Layout>
QwenTargetGeometry commonGeometry(const Layout &layout) {
  static_assert(std::tuple_size_v<decltype(Layout::hiddenCaptureLayers)> <=
                QwenTargetGeometry::maximumCaptureLayers);
  QwenTargetGeometry result;
  result.maximumContextTokens = layout.maximumContextTokens;
  result.layers = layout.layers;
  result.hiddenSize = layout.hiddenSize;
  result.vocabularySize = layout.vocabularySize;
  result.packedGdnWidth = layout.packedGdnWidth;
  result.packedFullWidth = layout.packedFullWidth;
  result.convolutionDimension = layout.convolutionDimension;
  result.attentionWidth = layout.attentionWidth;
  result.attentionQueryHeads = layout.attentionQueryHeads;
  result.attentionKvHeads = layout.attentionKvHeads;
  result.attentionHeadDimension = layout.attentionHeadDimension;
  result.rotaryPairs = layout.rotaryPairs;
  result.rotaryTheta = layout.rotaryTheta;
  result.gdnKeyHeads = layout.gdnKeyHeads;
  result.gdnValueHeads = layout.gdnValueHeads;
  result.gdnHeadDimension = layout.gdnHeadDimension;
  result.maskToken = layout.maskToken;
  result.stopTokens = layout.stopTokens;
  result.ffnKind = Layout::ffnKind;
  result.kvLayout = layout.kvLayout();
  result.stateLayout = layout.gdnStateLayout();
  result.captureLayerCount =
      static_cast<uint32_t>(layout.hiddenCaptureLayers.size());
  std::copy(layout.hiddenCaptureLayers.begin(),
            layout.hiddenCaptureLayers.end(),
            result.captureLayerValues.begin());
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_8Layout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_6MoeLayout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.experts = layout.experts;
  result.expertsPerToken = layout.expertsPerToken;
  result.expertIntermediateSize = layout.expertIntermediateSize;
  return result;
}

template <class Weights>
void requireWeights(const Weights &weights,
                    const QwenTargetGeometry &geometry) {
  const uint32_t attentionLayers = static_cast<uint32_t>(std::count_if(
      weights.layers.begin(), weights.layers.end(), [](const auto &layer) {
        return std::holds_alternative<QwenAttentionWeights>(layer.mixer);
      }));
  if (!geometry.valid() || weights.layers.size() != geometry.layers ||
      attentionLayers + geometry.mtpLayers != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument(
        "Qwen target weights do not match execution geometry");
  }
}

} // namespace

template <class Layout, class Layer>
QwenTarget::QwenTarget(const QwenTargetWeights<Layout, Layer> &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators, kv::Format format)
    : weights_(&weights), weightsBase_(weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators) {
  geometry_.kvLayout.format = format;
  requireWeights(weights, geometry_);
}

template QwenTarget::QwenTarget(const Qwen3_8Weights &, metal::MetalBackend &, const ops::ExecutionPlans &,
                                kv::Format);
template QwenTarget::QwenTarget(const Qwen3_6MoeWeights &, metal::MetalBackend &,
                                const ops::ExecutionPlans &, kv::Format);

namespace {

void includeProjection(QwenTargetGeometry &geometry, const ops::Projection &projection) {
  geometry.decodeProjections.push_back(projection.shape());
}

// The projections each layer's FFN dispatches, from the first layer on.
void includeFfn(QwenTargetGeometry &geometry, const Qwen3_8LayerWeights &layer, bool) {
  if (layer.gateProjection.shape() != layer.upProjection.shape())
    throw WeightStoreError("fused gate/up projections must have matching shapes and layouts");
  includeProjection(geometry, layer.gateProjection);
  includeProjection(geometry, layer.upProjection);
  includeProjection(geometry, layer.downProjection);
  geometry.gateUpProjections.push_back(layer.upProjection.shape());
}
// No source mixes MoE layouts, so one plan runs every block of a step.
void includeFfn(QwenTargetGeometry &geometry, const Qwen3_6MoeLayerWeights &layer, bool first) {
  if (first) geometry.moeLayout = layer.ffn.layout();
  if (layer.ffn.layout() != geometry.moeLayout)
    throw WeightStoreError("the MoE blocks of a target must share one weight layout");
}

// The format of most routed expert weights of a GGUF target's MoE blocks,
// GGUF_FMT_COUNT for none (ops::MoeShape::expertFormat).
uint32_t routedExpertFormat(std::span<const Qwen3_8LayerWeights>) { return GGUF_FMT_COUNT; }
uint32_t routedExpertFormat(std::span<const Qwen3_6MoeLayerWeights> layers) {
  std::array<uint64_t, GGUF_FMT_COUNT> weights{};
  for (const Qwen3_6MoeLayerWeights &layer : layers) {
    if (layer.ffn.layout() != ops::WeightLayout::Block32) return GGUF_FMT_COUNT;
    const ops::BlockMoeWeights &block = layer.ffn.blocks();
    for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
      if (!projection->routed.isFloat())
        weights[projection->routed.formatId] += uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
  }
  const auto most = std::max_element(weights.begin(), weights.end());
  return *most ? uint32_t(most - weights.begin()) : GGUF_FMT_COUNT;
}

} // namespace

template <class Layout, class Layer>
QwenTargetGeometry qwenTargetGeometry(const QwenTargetWeights<Layout, Layer> &weights) {
  auto geometry = geometryFor(weights.layout);
  for (const auto &layer : weights.layers) {
    std::visit([&](const auto &mixer) {
      includeProjection(geometry, mixer.inputProjection);
      includeProjection(geometry, mixer.outputProjection);
    }, layer.mixer);
    includeFfn(geometry, layer, &layer == &weights.layers.front());
  }
  geometry.moeExpertFormat = routedExpertFormat(weights.layers);
  geometry.prefillProjections = geometry.decodeProjections;
  includeProjection(geometry, weights.logitsProjection);
  for (auto *shapes : {&geometry.prefillProjections, &geometry.decodeProjections,
                       &geometry.gateUpProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
  return geometry;
}

template QwenTargetGeometry qwenTargetGeometry(const Qwen3_8Weights &);
template QwenTargetGeometry qwenTargetGeometry(const Qwen3_6MoeWeights &);

QwenTargetGeometry qwenTargetGeometry(const Qwen4ExpWeights &weights) {
  const Qwen4ExpLayout &layout = weights.layout;
  QwenTargetGeometry geometry = commonGeometry(layout);
  geometry.experts = layout.experts;
  geometry.expertsPerToken = layout.expertsPerToken;
  geometry.expertIntermediateSize = layout.expertIntermediateSize;
  geometry.hyperConnections = layout.hyperConnections;
  geometry.hyperRank = layout.hyperRank;
  geometry.pleLayer = layout.pleLayer;
  geometry.pleHeads = layout.pleHeads();
  geometry.pleHeadDimension = layout.pleHeadDimension;
  geometry.pleTaps = layout.pleConvolutionTaps;
  geometry.pleDilation = layout.pleNgram;
  geometry.mtpLayers = layout.mtpLayers;
  const auto hyper = [&](const Qwen4HyperWeights &mix) {
    includeProjection(geometry, mix.down);
    includeProjection(geometry, mix.up);
  };
  std::array<uint64_t, GGUF_FMT_COUNT> expertWeights{};
  for (const Qwen4ExpLayerWeights &layer : weights.layers) {
    hyper(layer.mixerHyper);
    hyper(layer.ffnHyper);
    std::visit([&](const auto &mixer) {
      includeProjection(geometry, mixer.inputProjection);
      includeProjection(geometry, mixer.outputProjection);
    }, layer.mixer);
    if (layer.indexer) {
      includeProjection(geometry, layer.indexer->query);
      includeProjection(geometry, layer.indexer->key);
    }
    if (layer.ple) {
      includeProjection(geometry, layer.ple->key);
      includeProjection(geometry, layer.ple->value);
    }
    if (&layer == &weights.layers.front()) geometry.moeLayout = layer.ffn.layout();
    if (layer.ffn.layout() != geometry.moeLayout || geometry.moeLayout != ops::WeightLayout::Block32)
      throw WeightStoreError("the MoE blocks of a target must share one GGUF weight layout");
    const ops::BlockMoeWeights &block = layer.ffn.blocks();
    for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
      if (!projection->routed.isFloat())
        expertWeights[projection->routed.formatId] +=
            uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
  }
  hyper(weights.finalHyper);
  if (weights.mtp) {
    const Qwen4MtpWeights &mtp = *weights.mtp;
    hyper(mtp.block.mixerHyper);
    hyper(mtp.block.ffnHyper);
    hyper(mtp.head);
    std::visit([&](const auto &mixer) {
      includeProjection(geometry, mixer.inputProjection);
      includeProjection(geometry, mixer.outputProjection);
    }, mtp.block.mixer);
    includeProjection(geometry, mtp.block.indexer->query);
    includeProjection(geometry, mtp.block.indexer->key);
    includeProjection(geometry, mtp.embedHidden);
  }
  const auto most = std::max_element(expertWeights.begin(), expertWeights.end());
  geometry.moeExpertFormat = *most ? uint32_t(most - expertWeights.begin()) : GGUF_FMT_COUNT;
  geometry.prefillProjections = geometry.decodeProjections;
  includeProjection(geometry, weights.logitsProjection);
  for (auto *shapes : {&geometry.prefillProjections, &geometry.decodeProjections, &geometry.gateUpProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
  return geometry;
}

QwenTarget::QwenTarget(const Qwen4ExpWeights &weights, metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators, kv::Format format)
    : weights_(&weights), weightsBase_(weights), geometry_(qwenTargetGeometry(weights)), backend_(backend),
      operators_(operators) {
  geometry_.kvLayout.format = format;
  requireWeights(weights, geometry_);
}

const Qwen4ExpWeights *QwenTarget::qwen4() const noexcept {
  const auto *weights = std::get_if<const Qwen4ExpWeights *>(&weights_);
  return weights ? *weights : nullptr;
}

const ops::Projection &QwenTarget::vocabularyProjection() const noexcept {
  return weightsBase_.logitsProjection;
}

uint32_t QwenTarget::decodeStorageLanes(uint32_t lanes) const {
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  uint32_t storageRows = rows;
  for (const auto &shape : geometry_.decodeProjections)
    storageRows = std::max(storageRows, operators_.linear().decodeStorageRows(rows, shape));
  return storageRows / ExecutionLimits::targetVerifyRows;
}

namespace {

// Rows [begin, begin + count) of a row-major buffer of `width` values of T.
template <class T>
metal::MetalBuffer rowsOf(metal::MetalBackend &backend, const metal::MetalBuffer &buffer, uint32_t begin,
                          uint32_t count, uint32_t width) {
  return backend.view(buffer, uint64_t{begin} * width * sizeof(T), uint64_t{count} * width * sizeof(T));
}

ops::HyperShape hyperShape(const QwenTargetGeometry &g) { return {g.hiddenSize, g.hyperConnections, g.hyperRank}; }
ops::PleShape pleShape(const QwenTargetGeometry &g) {
  return {g.hiddenSize, g.hyperConnections, g.pleTaps, g.pleDilation};
}

// The QSA indexer's fp32 projections of `rows` rows: one float segment each
// (Qwen4Exp.cpp readIndexer), at any row count.
void addIndexerProjection(metal::CommandGraph &graph, const ops::Linear &linear, metal::MetalBuffer input,
                          const ops::Projection &projection, metal::MetalBuffer output, uint32_t rows) {
  const auto &segments = projection.blocks().segments;
  if (segments.size() != 1 || !segments.front().isFloat())
    throw std::invalid_argument("the QSA indexer projections are single float segments");
  ops::addGgufFloat(graph, std::move(input), segments.front(), std::move(output), rows, projection.outputSize, 0,
                    ops::FloatOutput::Float32, linear.ggufFloatTile(rows, projection.outputSize));
}

// Rows `begin`.. of a row-major buffer of `width` values of T, to the end of
// the buffer: a projection's tile may read and write rows past its own, which
// for the MTP head's stream-major rows are the next stream's, projected after.
template <class T>
metal::MetalBuffer rowsFrom(metal::MetalBackend &backend, const metal::MetalBuffer &buffer, uint32_t begin,
                            uint32_t width) {
  const uint64_t offset = uint64_t{begin} * width * sizeof(T);
  return backend.view(buffer, offset, buffer.sizeBytes() - offset);
}

void requireLayerPartition(const QwenTargetGeometry &geometry, uint32_t gdnLayers, uint32_t attentionLayers) {
  if (gdnLayers != geometry.stateLayout.layers ||
      attentionLayers + geometry.mtpLayers != geometry.kvLayout.attentionLayers)
    throw std::logic_error("Qwen target layer partition mismatch");
}

} // namespace

// The state a prefill command's layers share: its inputs and the next GDN
// and attention layer of the step.
struct QwenTarget::PrefillStep {
  metal::CommandGraph &graph;
  const QwenTargetPrefillBuffers &buffers;
  std::span<const QwenTargetPrefillSequence> sequences;
  uint32_t rows;
  std::span<const kv::LayerStorage> kvLayers;
  std::optional<ops::MoePlan> moe{};
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
};

struct QwenTarget::VerifyStep {
  metal::CommandGraph &graph;
  const QwenTargetVerifyBuffers &buffers;
  std::span<const kv::LayerStorage> kvLayers;
  std::span<const kv::Q8ChunkedPrefillParams> q8;
  std::span<const kv::Q8VerifyAttentionParams> verify;
  uint32_t lanes;
  uint32_t rows;
  ops::LinearDispatchStats &stats;
  ops::VerifyAttentionPlan attention;
  std::optional<ops::MoePlan> moe{};
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
};

metal::MetalBuffer QwenTarget::addPrefill(
    metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const kv::LayerStorage> kvLayers) const {
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen packed prefill batch");
  }
  for (const QwenTargetPrefillSequence &sequence : sequences) {
    if (sequence.convolutionIn.size() != geometry_.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry_.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry_.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry_.stateLayout.layers) {
      throw std::invalid_argument("Qwen prefill state layer mismatch");
    }
  }
  PrefillStep step{graph, buffers, sequences, rows, kvLayers};
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moePrefill(geometry_.moeShape(), rows);
  if (const Qwen4ExpWeights *qwen4Weights = qwen4()) return addQwen4Prefill(step, *qwen4Weights);
  std::visit([&](const auto *weights) {
    if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(*weights)>, Qwen4ExpWeights>)
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      const auto &layer = weights->layers[index];
      const metal::MetalBuffer input = buffers.hidden[index & 1];
      const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
      const metal::MetalBuffer residual = std::visit(
          [&](const auto &mixer) { return addPrefillMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
      addPrefillFfn(step, layer, residual, output);
      if (const auto slot = geometry_.captureSlot(index))
        for (const QwenTargetPrefillSequence &sequence : sequences)
          for (uint32_t capture = 0; capture < sequence.captureCount; ++capture) {
            const QwenTargetPrefillCapture &c = sequence.captures[capture];
            ops::DraftAttention::captureTargetHidden(graph, output, buffers.captured, c.rows, *slot,
                                                     c.sourceStart, c.destinationStart, geometry_.hiddenSize,
                                                     geometry_.capturedHiddenSize());
          }
    }
  }, weights_);
  requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  return buffers.hidden[geometry_.layers & 1];
}

// An affine prefill projection reads the Q4 input sums of its rows, which the
// norm writes beside them; a block projection reads none.
void QwenTarget::addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm,
                                ops::WeightLayout consumer) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  if (consumer == ops::WeightLayout::Affine64)
    ops::Normalization::addRmsWithQ4Sums(step.graph, input, norm, b.normalized, b.projectionSums,
                                         geometry_.hiddenSize, step.rows);
  else
    ops::Normalization::addRms(step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows);
}

// The mixer output projection adds the mixer's rows to `input`.
void QwenTarget::addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                                  metal::MetalBuffer input, metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  if (projection.layout() == ops::WeightLayout::Affine64)
    operators_.linear().addPrefillSums(step.graph, hidden, b.projectionSums, projection, step.rows);
  operators_.linear().addPrefillResidual(step.graph, hidden, projection, input, output, b.projectionSums,
                                         step.rows, b.linearScratch);
}

metal::MetalBuffer QwenTarget::addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.gdnLayer++;
  addPrefillNorm(step, input, norm, mixer.inputProjection.layout());
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked, b.projectionSums,
                                 step.rows, b.linearScratch);
  for (const QwenTargetPrefillSequence &sequence : step.sequences) {
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<float>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    ops::GDN::addPrefill(
        step.graph,
        {u16(b.gdnPacked, geometry_.packedGdnWidth), mixer.convolutionWeights, sequence.convolutionIn[layer],
         sequence.convolutionOut[layer], u16(b.gdnQueries, geometry_.gdnKeyWidth()),
         u16(b.gdnKeys, geometry_.gdnKeyWidth()), u16(b.gdnValues, geometry_.attentionWidth), mixer.decay,
         mixer.timeBias, f32(b.gdnDecay, geometry_.gdnValueHeads), u16(b.gdnBeta, geometry_.gdnValueHeads),
         sequence.recurrentIn[layer], sequence.recurrentOut[layer], u16(b.recurrent, geometry_.attentionWidth),
         mixer.mixerNorm, u16(b.gdnHidden, geometry_.attentionWidth)},
        geometry_.gdnShape(), sequence.rows, mixer.outputHeadOrder);
  }
  addPrefillOutput(step, b.gdnHidden, mixer.outputProjection, input, b.gdnOutput);
  return b.gdnOutput;
}

metal::MetalBuffer QwenTarget::addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.attentionLayer++;
  addPrefillNorm(step, input, norm, mixer.inputProjection.layout());
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, b.projectionSums,
                                 step.rows, b.linearScratch);
  for (const QwenTargetPrefillSequence &sequence : step.sequences) {
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<float>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const uint64_t headBytes = uint64_t{sequence.attentionStride} * geometry_.attentionHeadDimension * sizeof(uint16_t);
    const uint64_t queryBytes = geometry_.attentionQueryHeads * headBytes;
    const uint64_t kvBytes = geometry_.attentionKvHeads * headBytes;
    const metal::MetalBuffer queries = backend_.view(b.fullQueries, sequence.queryOffset, queryBytes);
    const metal::MetalBuffer attentionRows = backend_.view(b.fullAttention, sequence.queryOffset, queryBytes);
    const metal::MetalBuffer keys = backend_.view(b.chunkKeys, sequence.kvOffset, kvBytes);
    const metal::MetalBuffer values = backend_.view(b.chunkValues, sequence.kvOffset, kvBytes);
    ops::PagedAttention::addPrefillProjection(
        step.graph, u16(b.fullPacked, geometry_.packedFullWidth), mixer.queryNorm, mixer.keyNorm,
        f32(b.ropeCos, geometry_.rotaryPairs), f32(b.ropeSin, geometry_.rotaryPairs), queries, keys, values,
        sequence.rows, sequence.attentionStride, sequence.attentionStride, geometry_.attentionQueryHeads,
        geometry_.kvLayout);
    ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[layer], keys, values, sequence.pageTable,
                                         sequence.q8, geometry_.kvLayout);
    ops::PagedAttention::addPrefill(
        step.graph, step.kvLayers[layer], queries, attentionRows, b.attentionPartials, b.attentionStatistics,
        sequence.pageTable, sequence.q8,
        operators_.prefillAttention(sequence.rows, geometry_.attentionQueryHeads, geometry_.kvLayout,
                                    sequence.q8.committed_tokens));
    ops::PagedAttention::addPrefillGate(
        step.graph, u16(b.fullPacked, geometry_.packedFullWidth), attentionRows,
        u16(b.attentionHidden, geometry_.attentionWidth), sequence.rows, sequence.attentionStride,
        sequence.attentionStride, geometry_.attentionQueryHeads, geometry_.kvLayout);
  }
  addPrefillOutput(step, b.attentionHidden, mixer.outputProjection, input, b.attentionOutput);
  return b.attentionOutput;
}

void QwenTarget::addPrefillFfn(PrefillStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                               metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  addPrefillNorm(step, residual, layer.postAttentionNorm, layer.gateProjection.layout());
  linear.addPrefill(step.graph, b.normalized, layer.gateProjection, b.denseGateScratch, b.projectionSums,
                    step.rows, b.linearScratch);
  linear.addPrefillUpWithGate(step.graph, b.normalized, layer.upProjection, b.denseGateScratch,
                              b.denseIntermediate, b.projectionSums, b.downProjectionSums, step.rows,
                              b.linearScratch);
  linear.addPrefillResidual(step.graph, b.denseIntermediate, layer.downProjection, residual, output,
                            b.downProjectionSums, step.rows, b.linearScratch);
}

void QwenTarget::addPrefillFfn(PrefillStep &step, const Qwen3_6MoeLayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                             step.rows);
  ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, layer.ffn, *step.moe);
}

void QwenTarget::addVerify(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const kv::LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
    ops::LinearDispatchStats &stats) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      q8.size() != ExecutionLimits::maximumBatchWidth ||
      verify.size() != ExecutionLimits::maximumBatchWidth ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry_.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry_.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry_.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry_.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = verify[lane].committed_tokens;
  VerifyStep step{graph, buffers, kvLayers, q8, verify, lanes, rows, stats,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.kvLayout, histories)};
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moeDecode(geometry_.moeShape(), lanes);
  if (const Qwen4ExpWeights *qwen4Weights = qwen4()) {
    addQwen4Verify(step, *qwen4Weights);
    return;
  }
  const ops::Linear &linear = operators_.linear();
  std::visit([&](const auto *weights) {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*weights)>, Qwen4ExpWeights>) return;
    else {
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      const auto &layer = weights->layers[index];
      const metal::MetalBuffer input = buffers.hidden[index & 1];
      const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
      const metal::MetalBuffer residual = std::visit(
          [&](const auto &mixer) { return addVerifyMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
      addVerifyFfn(step, layer, residual, output);
      if (const auto slot = geometry_.captureSlot(index))
        ops::DraftAttention::captureTargetHidden(graph, output, buffers.capturedTargetHidden, rows, *slot, 0, 0,
                                                 geometry_.hiddenSize, geometry_.capturedHiddenSize());
    }
    requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
    const ops::PreparedInput finalHidden = ops::Normalization::addRms(
        graph, buffers.hidden[geometry_.layers & 1], weights->finalNorm, buffers.finalHidden,
        geometry_.hiddenSize, rows, buffers.linearScratch,
        linear.decodePlan(weights->logitsProjection, lanes).input());
    linear.addDecodeBatch(graph, buffers.finalHidden, weights->logitsProjection, buffers.logits, lanes, stats,
                          buffers.linearScratch, finalHidden);
    }
  }, weights_);
}

// Each producer emits the table (if any) its consumer's plan reads.
metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const QwenGdnWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.gdnLayer++;
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch,
      linear.decodePlan(mixer.inputProjection, step.lanes).input());
  linear.addDecodeBatch(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked[layer], step.lanes,
                        step.stats, b.linearScratch, normalized);
  const ops::PreparedInput hidden = ops::GDN::addDecode(
      step.graph,
      {b.gdnPacked[layer], mixer.convolutionWeights, b.currentGdnStates, b.nextGdnStates, b.gdnMixed[layer],
       mixer.decay, mixer.timeBias, b.gdnDecay[layer], b.gdnBeta[layer], b.recurrent, mixer.mixerNorm,
       b.gdnHidden, b.arrived, b.generation, b.linearScratch},
      geometry_.gdnShape(), step.lanes, layer,
      {geometry_.stateLayout.convolutionLayerBytes(), geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()},
      mixer.outputHeadOrder,
      linear.decodePlan(mixer.outputProjection, step.lanes, ops::LinearEpilogue::Residual).input());
  linear.addResidualBatch(step.graph, b.gdnHidden, mixer.outputProjection, input, b.gdnOutput, step.lanes,
                          step.stats, b.linearScratch, hidden);
  return b.gdnOutput;
}

metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  constexpr uint32_t tileRows = kv::kPageTokens;
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer++;
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch,
      linear.decodePlan(mixer.inputProjection, step.lanes).input());
  linear.addDecodeBatch(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, step.lanes, step.stats,
                        b.linearScratch, normalized);
  ops::PagedAttention::addVerifyProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                           b.ropeSin, b.fullQueries, b.chunkKeys[layer], b.chunkValues[layer],
                                           ExecutionLimits::targetVerifyRows, tileRows, tileRows,
                                           geometry_.attentionQueryHeads, geometry_.kvLayout, step.lanes);
  ops::PagedAttention::addVerify(step.graph, step.kvLayers[layer],
                                 {b.chunkKeys[layer], b.chunkValues[layer], b.fullQueries, b.attentionPartials,
                                  b.attentionStatistics, b.fullAttention, b.pageTables},
                                 step.q8, step.verify, step.attention);
  const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
      step.graph, b.fullPacked, b.fullAttention, b.attentionHidden, ExecutionLimits::targetVerifyRows, tileRows,
      tileRows, geometry_.attentionQueryHeads, geometry_.kvLayout, step.lanes, b.linearScratch,
      linear.decodePlan(mixer.outputProjection, step.lanes, ops::LinearEpilogue::Residual).input());
  linear.addResidualBatch(step.graph, b.attentionHidden, mixer.outputProjection, input, b.attentionOutput,
                          step.lanes, step.stats, b.linearScratch, hidden);
  return b.attentionOutput;
}

void QwenTarget::addVerifyFfn(VerifyStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize, step.rows,
      b.linearScratch,
      linear.decodePlan(layer.upProjection, step.lanes, ops::LinearEpilogue::GateUp, &layer.gateProjection).input());
  linear.addGateUpBatch(step.graph, b.normalized, layer.gateProjection, layer.upProjection, b.denseGateScratch,
                        b.denseIntermediate, step.lanes, step.stats, b.linearScratch, normalized);
  linear.addResidualBatch(step.graph, b.denseIntermediate, layer.downProjection, residual, output, step.lanes,
                          step.stats, b.linearScratch);
}

void QwenTarget::addVerifyFfn(VerifyStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                             step.rows);
  ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, layer.ffn, *step.moe);
}

void QwenTarget::addHead(metal::CommandGraph &graph,
                         metal::MetalBuffer hidden,
                         metal::MetalBuffer finalHidden,
                         metal::MetalBuffer logits,
                         uint32_t normalizedRows, ops::LinearScratch scratch) const {
  if (!normalizedRows ||
      normalizedRows > ExecutionLimits::targetVerifyRows) {
    throw std::invalid_argument("invalid Qwen head row count");
  }
  // Qwen3.8-Flash-Next's hidden rows are its final mix, which the head
  // reads as they are.
  if (qwen4()) finalHidden = std::move(hidden);
  else
    ops::Normalization::addRms(graph, std::move(hidden), weightsBase_.finalNorm, finalHidden,
                               geometry_.hiddenSize, normalizedRows);
  operators_.linear().addDecode(graph,
                std::move(finalHidden), vocabularyProjection(),
                std::move(logits), scratch);
}

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, std::move(hidden),
                      rows);
}

void QwenTarget::addStateCommit(metal::CommandGraph &graph,
                                QwenTargetCommitBuffers buffers,
                                uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  if (qwen4()) {
    // Each lane's PLE history after its retained rows.
    const ops::PleShape shape{geometry_.hiddenSize, geometry_.hyperConnections, geometry_.pleTaps,
                              geometry_.pleDilation};
    const uint64_t laneBytes = uint64_t{ExecutionLimits::targetVerifyRows} * shape.width() * sizeof(uint16_t);
    for (uint32_t lane = 0; lane < lanes; ++lane)
      ops::Qwen4::addPleCommit(graph, backend_.view(buffers.pleConvolution, lane * laneBytes, laneBytes),
                               buffers.pleHistoriesIn[lane], buffers.pleHistoriesOut[lane],
                               backend_.view(buffers.retainedCounts, lane * sizeof(uint32_t), sizeof(uint32_t)),
                               shape);
    // With the MTP head, each lane carries its last retained row's streams.
    if (hasMtp()) {
      const uint32_t laneRows = ExecutionLimits::targetVerifyRows;
      for (uint32_t lane = 0; lane < lanes; ++lane)
        ops::Qwen4::addMtpCarry(
            graph, rowsOf<float>(backend_, buffers.streams, lane * laneRows, laneRows, geometry_.streamWidth()),
            mtpCarried(buffers.pleHistoriesOut[lane]),
            backend_.view(buffers.retainedCounts, lane * sizeof(uint32_t), sizeof(uint32_t)),
            hyperShape(geometry_), laneRows);
    }
  }
  ops::GDN::addCommit(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed),
       std::move(buffers.decay), std::move(buffers.beta), buffers.currentStates,
       buffers.nextStates, std::move(buffers.retainedCounts)},
      geometry_.gdnShape(), geometry_.stateLayout.layers, lanes,
      {geometry_.stateLayout.convolutionLayerBytes(),
       geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()});
}


// Qwen3.8-Flash-Next (model/Qwen4Exp.hpp). Every block reads the mix of the
// fp32 residual streams and adds its output back to each stream; the PLE
// layer adds its gated n-gram values before its mixer's mix.

metal::MetalBuffer QwenTarget::addQwen4Prefill(PrefillStep &step, const Qwen4ExpWeights &weights) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const QwenHyperBuffers &h = b.hyper;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape(geometry_);
  const uint32_t rows = step.rows;
  const auto project = [&](metal::MetalBuffer input, const ops::Projection &projection, metal::MetalBuffer output) {
    linear.addPrefill(step.graph, std::move(input), projection, std::move(output), b.projectionSums, rows,
                      b.linearScratch);
  };
  // The mix of the streams into b.normalized and, when it injects, the
  // injection weights into h.weights.
  const auto mix = [&](const Qwen4HyperWeights &hyper, metal::MetalBuffer output,
                       const metal::MetalBuffer &streams) {
    ops::Qwen4::addNorm(step.graph, streams, hyper.norm, hyper.inject, h.normalized, h.weights, hs, rows);
    project(h.normalized, hyper.down, h.low);
    ops::Qwen4::addLow(step.graph, h.low, hs, rows);
    project(h.low, hyper.up, h.gate);
    ops::Qwen4::addMix(step.graph, h.normalized, h.gate, std::move(output), hs, rows);
  };
  const auto f32Rows = [&](const metal::MetalBuffer &buffer, uint32_t begin, uint32_t count, uint32_t width) {
    return rowsOf<float>(backend_, buffer, begin, count, width);
  };
  const auto u16Rows = [&](const metal::MetalBuffer &buffer, uint32_t begin, uint32_t count, uint32_t width) {
    return rowsOf<uint16_t>(backend_, buffer, begin, count, width);
  };
  ops::Qwen4::addInit(step.graph, b.hidden[0], h.streams, hs, rows);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const Qwen4ExpLayerWeights &layer = weights.layers[index];
    if (layer.ple) {
      const Qwen4PleWeights &ple = *layer.ple;
      const ops::PleShape ps = pleShape(geometry_);
      ops::Embedding::add(step.graph, h.pleIndices, weights.pleTable, h.pleRows, rows * geometry_.pleHeads);
      project(h.pleRows, ple.key, h.gate);
      project(h.pleRows, ple.value, h.pleValue);
      ops::Qwen4::addPleGate(step.graph, h.gate, h.pleValue, h.streams, ple.keyNorm, ple.queryNorm,
                             ple.convolutionNorm, h.pleGated, h.pleConvolution, ps, rows);
      for (const QwenTargetPrefillSequence &sequence : step.sequences)
        ops::Qwen4::addPleConvolution(
            step.graph, u16Rows(h.pleConvolution, sequence.rowBegin, sequence.rows, ps.width()),
            sequence.pleHistoryIn, sequence.pleHistoryOut, ple.convolution,
            f32Rows(h.pleGated, sequence.rowBegin, sequence.rows, ps.width()),
            f32Rows(h.streams, sequence.rowBegin, sequence.rows, ps.width()), ps, sequence.rows);
    }
    mix(layer.mixerHyper, b.normalized, h.streams);
    std::visit(
        [&](const auto &mixer) {
          using Mixer = std::remove_cvref_t<decltype(mixer)>;
          if constexpr (std::is_same_v<Mixer, QwenGdnWeights>) {
            const uint32_t gdnLayer = step.gdnLayer++;
            project(b.normalized, mixer.inputProjection, b.gdnPacked);
            for (const QwenTargetPrefillSequence &sequence : step.sequences) {
              const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
                return u16Rows(buffer, sequence.rowBegin, sequence.rows, width);
              };
              const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
                return f32Rows(buffer, sequence.rowBegin, sequence.rows, width);
              };
              ops::GDN::addPrefill(
                  step.graph,
                  {u16(b.gdnPacked, geometry_.packedGdnWidth), mixer.convolutionWeights,
                   sequence.convolutionIn[gdnLayer], sequence.convolutionOut[gdnLayer],
                   u16(b.gdnQueries, geometry_.gdnKeyWidth()), u16(b.gdnKeys, geometry_.gdnKeyWidth()),
                   u16(b.gdnValues, geometry_.attentionWidth), mixer.decay, mixer.timeBias,
                   f32(b.gdnDecay, geometry_.gdnValueHeads), u16(b.gdnBeta, geometry_.gdnValueHeads),
                   sequence.recurrentIn[gdnLayer], sequence.recurrentOut[gdnLayer],
                   u16(b.recurrent, geometry_.attentionWidth), mixer.mixerNorm,
                   u16(b.gdnHidden, geometry_.attentionWidth)},
                  geometry_.gdnShape(), sequence.rows, mixer.outputHeadOrder);
            }
            project(b.gdnHidden, mixer.outputProjection, h.branch);
          } else {
            const uint32_t attentionLayer = step.attentionLayer++;
            project(b.normalized, mixer.inputProjection, b.fullPacked);
            const Qwen4IndexerWeights &indexer = *layer.indexer;
            addIndexerProjection(step.graph, linear, b.normalized, indexer.query, h.indexerQuery, rows);
            addIndexerProjection(step.graph, linear, b.normalized, indexer.key, h.indexerKey, rows);
            for (const QwenTargetPrefillSequence &sequence : step.sequences) {
              const ops::QsaRows qsa{sequence.rows, static_cast<uint32_t>(sequence.q8.committed_tokens),
                                     weights.layout.indexerTopBlocks, h.qsaMaskWords, h.qsaMaskWords * 32};
              constexpr uint32_t indexWidth = QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION;
              const metal::MetalBuffer queries = f32Rows(h.qsaQueries, sequence.rowBegin, sequence.rows, indexWidth);
              ops::Qwen4::addQsaProject(step.graph, f32Rows(h.indexerQuery, sequence.rowBegin, sequence.rows, indexWidth),
                                        f32Rows(h.indexerKey, sequence.rowBegin, sequence.rows, QWEN4_QSA_DIMENSION),
                                        indexer.queryNorm,
                                        f32Rows(b.ropeCos, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs),
                                        f32Rows(b.ropeSin, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs),
                                        sequence.pageTable, h.qsaRawKeys[attentionLayer], queries, qsa);
              ops::Qwen4::addQsaPool(step.graph, sequence.qsaBlocks, sequence.qsaBlockCount,
                                     h.qsaRawKeys[attentionLayer], indexer.keyNorm, h.inverseFrequencies,
                                     sequence.pageTable, h.qsaPooledKeys[attentionLayer]);
            }
            for (const QwenTargetPrefillSequence &sequence : step.sequences) {
              const uint32_t first = static_cast<uint32_t>(sequence.q8.committed_tokens);
              const bool sparse = ops::Qwen4::qsaSparse(first, sequence.rows, weights.layout.indexerTopBlocks);
              const ops::QsaRows qsa{sequence.rows, first, weights.layout.indexerTopBlocks, h.qsaMaskWords,
                                     h.qsaMaskWords * 32};
              constexpr uint32_t indexWidth = QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION;
              if (sparse)
                ops::Qwen4::addQsaSelect(step.graph, f32Rows(h.qsaQueries, sequence.rowBegin, sequence.rows, indexWidth),
                                         h.qsaPooledKeys[attentionLayer], sequence.pageTable, h.qsaScores,
                                         h.qsaScoreRows,
                                         rowsOf<uint32_t>(backend_, h.qsaMask, sequence.rowBegin, sequence.rows,
                                                          h.qsaMaskWords),
                                         qsa);
              const ops::QsaMask mask{h.qsaMask, sparse ? h.qsaMaskWords : 0, sequence.rowBegin};
              const uint64_t headBytes =
                  uint64_t{sequence.attentionStride} * geometry_.attentionHeadDimension * sizeof(uint16_t);
              const uint64_t queryBytes = geometry_.attentionQueryHeads * headBytes;
              const uint64_t kvBytes = geometry_.attentionKvHeads * headBytes;
              const metal::MetalBuffer queries = backend_.view(b.fullQueries, sequence.queryOffset, queryBytes);
              const metal::MetalBuffer attentionRows = backend_.view(b.fullAttention, sequence.queryOffset, queryBytes);
              const metal::MetalBuffer keys = backend_.view(b.chunkKeys, sequence.kvOffset, kvBytes);
              const metal::MetalBuffer values = backend_.view(b.chunkValues, sequence.kvOffset, kvBytes);
              ops::PagedAttention::addPrefillProjection(
                  step.graph, u16Rows(b.fullPacked, sequence.rowBegin, sequence.rows, geometry_.packedFullWidth),
                  mixer.queryNorm, mixer.keyNorm, f32Rows(b.ropeCos, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs),
                  f32Rows(b.ropeSin, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs), queries, keys, values,
                  sequence.rows, sequence.attentionStride, sequence.attentionStride, geometry_.attentionQueryHeads,
                  geometry_.kvLayout);
              ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[attentionLayer], keys, values,
                                                   sequence.pageTable, sequence.q8, geometry_.kvLayout);
              ops::PagedAttention::addPrefill(
                  step.graph, step.kvLayers[attentionLayer], queries, attentionRows, b.attentionPartials,
                  b.attentionStatistics, sequence.pageTable, sequence.q8,
                  operators_.prefillAttention(sequence.rows, geometry_.attentionQueryHeads, geometry_.kvLayout,
                                              sequence.q8.committed_tokens),
                  mask);
              ops::PagedAttention::addPrefillGate(
                  step.graph, u16Rows(b.fullPacked, sequence.rowBegin, sequence.rows, geometry_.packedFullWidth),
                  attentionRows, u16Rows(b.attentionHidden, sequence.rowBegin, sequence.rows, geometry_.attentionWidth),
                  sequence.rows, sequence.attentionStride, sequence.attentionStride, geometry_.attentionQueryHeads,
                  geometry_.kvLayout);
            }
            project(b.attentionHidden, mixer.outputProjection, h.branch);
          }
        },
        layer.mixer);
    ops::Qwen4::addCombine(step.graph, h.streams, h.branch, h.weights, hs, rows);
    mix(layer.ffnHyper, b.normalized, h.streams);
    ops::MoE::add(step.graph, {b.normalized, h.zeros, h.branch, b.moe}, layer.ffn, *step.moe);
    ops::Qwen4::addCombine(step.graph, h.streams, h.branch, h.weights, hs, rows);
  }
  requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  // The final mix, which the head reads as it is.
  mix(weights.finalHyper, b.hidden[1], h.streams);
  if (weights.mtp) {
    // The MTP head's KV at the rows: row r pairs the trunk's streams of the
    // position before (row 0: the sequence's carried streams) with the row's
    // token; each sequence carries its last row's streams.
    const Qwen4MtpWeights &mtp = *weights.mtp;
    const uint32_t hidden = geometry_.hiddenSize, width = geometry_.streamWidth();
    for (const QwenTargetPrefillSequence &sequence : step.sequences) {
      ops::Qwen4::addMtpInput(step.graph, u16Rows(b.hidden[0], sequence.rowBegin, sequence.rows, hidden),
                              mtpCarried(sequence.pleHistoryIn),
                              f32Rows(h.streams, sequence.rowBegin, sequence.rows, width), mtp.embedNorm,
                              mtp.hiddenNorm, h.mtpInput, hs, sequence.rows, sequence.rows, sequence.rowBegin, rows);
      ops::Qwen4::addMtpCarry(step.graph, f32Rows(h.streams, sequence.rowBegin, sequence.rows, width),
                              mtpCarried(sequence.pleHistoryOut), {}, hs, sequence.rows);
    }
    for (uint32_t s = 0; s < hs.streams; ++s)
      project(rowsFrom<uint16_t>(backend_, h.mtpInput, s * rows, 2 * hidden), mtp.embedHidden,
              rowsFrom<uint16_t>(backend_, h.gate, s * rows, hidden));
    ops::Qwen4::addMtpWiden(step.graph, h.gate, h.mtpStreams, hs, rows);
    mix(mtp.block.mixerHyper, b.normalized, h.mtpStreams);
    const auto &mixer = std::get<QwenAttentionWeights>(mtp.block.mixer);
    const Qwen4IndexerWeights &indexer = *mtp.block.indexer;
    const uint32_t attentionLayer = step.attentionLayer;
    project(b.normalized, mixer.inputProjection, b.fullPacked);
    addIndexerProjection(step.graph, linear, b.normalized, indexer.query, h.indexerQuery, rows);
    addIndexerProjection(step.graph, linear, b.normalized, indexer.key, h.indexerKey, rows);
    constexpr uint32_t indexWidth = QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION;
    for (const QwenTargetPrefillSequence &sequence : step.sequences) {
      const ops::QsaRows qsa{sequence.rows, static_cast<uint32_t>(sequence.q8.committed_tokens),
                             weights.layout.indexerTopBlocks, h.qsaMaskWords, h.qsaMaskWords * 32};
      ops::Qwen4::addQsaProject(step.graph, f32Rows(h.indexerQuery, sequence.rowBegin, sequence.rows, indexWidth),
                                f32Rows(h.indexerKey, sequence.rowBegin, sequence.rows, QWEN4_QSA_DIMENSION),
                                indexer.queryNorm,
                                f32Rows(b.ropeCos, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs),
                                f32Rows(b.ropeSin, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs),
                                sequence.pageTable, h.qsaRawKeys[attentionLayer],
                                f32Rows(h.qsaQueries, sequence.rowBegin, sequence.rows, indexWidth), qsa);
      ops::Qwen4::addQsaPool(step.graph, sequence.qsaBlocks, sequence.qsaBlockCount, h.qsaRawKeys[attentionLayer],
                             indexer.keyNorm, h.inverseFrequencies, sequence.pageTable,
                             h.qsaPooledKeys[attentionLayer]);
      const uint64_t headBytes =
          uint64_t{sequence.attentionStride} * geometry_.attentionHeadDimension * sizeof(uint16_t);
      const uint64_t queryBytes = geometry_.attentionQueryHeads * headBytes;
      const uint64_t kvBytes = geometry_.attentionKvHeads * headBytes;
      const metal::MetalBuffer keys = backend_.view(b.chunkKeys, sequence.kvOffset, kvBytes);
      const metal::MetalBuffer values = backend_.view(b.chunkValues, sequence.kvOffset, kvBytes);
      ops::PagedAttention::addPrefillProjection(
          step.graph, u16Rows(b.fullPacked, sequence.rowBegin, sequence.rows, geometry_.packedFullWidth),
          mixer.queryNorm, mixer.keyNorm, f32Rows(b.ropeCos, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs),
          f32Rows(b.ropeSin, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs),
          backend_.view(b.fullQueries, sequence.queryOffset, queryBytes), keys, values, sequence.rows,
          sequence.attentionStride, sequence.attentionStride, geometry_.attentionQueryHeads, geometry_.kvLayout);
      ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[attentionLayer], keys, values,
                                           sequence.pageTable, sequence.q8, geometry_.kvLayout);
    }
  }
  return b.hidden[1];
}

void QwenTarget::addQwen4VerifyMix(VerifyStep &step, const Qwen4HyperWeights &hyper, metal::MetalBuffer streams,
                                   metal::MetalBuffer output) const {
  const QwenHyperBuffers &h = step.buffers.hyper;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape(geometry_);
  const auto project = [&](metal::MetalBuffer input, const ops::Projection &projection, metal::MetalBuffer out) {
    linear.addDecodeBatch(step.graph, std::move(input), projection, std::move(out), step.lanes, step.stats,
                          step.buffers.linearScratch);
  };
  ops::Qwen4::addNorm(step.graph, std::move(streams), hyper.norm, hyper.inject, h.normalized, h.weights, hs,
                      step.rows);
  project(h.normalized, hyper.down, h.low);
  ops::Qwen4::addLow(step.graph, h.low, hs, step.rows);
  project(h.low, hyper.up, h.gate);
  ops::Qwen4::addMix(step.graph, h.normalized, h.gate, std::move(output), hs, step.rows);
}

void QwenTarget::addQwen4VerifyAttention(VerifyStep &step, const Qwen4ExpLayerWeights &layer,
                                         const QwenAttentionWeights &mixer, uint32_t attentionLayer) const {
  constexpr uint32_t tileRows = kv::kPageTokens;
  constexpr uint32_t laneRows = ExecutionLimits::targetVerifyRows;
  constexpr uint32_t indexWidth = QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION;
  const QwenTargetVerifyBuffers &b = step.buffers;
  const QwenHyperBuffers &h = b.hyper;
  const ops::Linear &linear = operators_.linear();
  const auto project = [&](metal::MetalBuffer input, const ops::Projection &projection, metal::MetalBuffer output,
                           ops::PreparedInput prepared = {}) {
    linear.addDecodeBatch(step.graph, std::move(input), projection, std::move(output), step.lanes, step.stats,
                          b.linearScratch, std::move(prepared));
  };
  const uint32_t topBlocks = qwen4()->layout.indexerTopBlocks;
  project(b.normalized, mixer.inputProjection, b.fullPacked);
  const Qwen4IndexerWeights &indexer = *layer.indexer;
  addIndexerProjection(step.graph, linear, b.normalized, indexer.query, h.indexerQuery, step.rows);
  addIndexerProjection(step.graph, linear, b.normalized, indexer.key, h.indexerKey, step.rows);
  bool sparse = false;
  for (uint32_t lane = 0; lane < step.lanes; ++lane)
    sparse = sparse || ops::Qwen4::qsaSparse(step.verify[lane].committed_tokens, laneRows, topBlocks);
  for (uint32_t lane = 0; lane < step.lanes; ++lane) {
    const uint32_t first = step.verify[lane].committed_tokens;
    const ops::QsaRows qsa{laneRows, first, topBlocks, h.qsaMaskWords, h.qsaMaskWords * 32};
    const metal::MetalBuffer queries = rowsOf<float>(backend_, h.qsaQueries, lane * laneRows, laneRows, indexWidth);
    ops::Qwen4::addQsaProject(step.graph, rowsOf<float>(backend_, h.indexerQuery, lane * laneRows, laneRows, indexWidth),
                              rowsOf<float>(backend_, h.indexerKey, lane * laneRows, laneRows, QWEN4_QSA_DIMENSION),
                              indexer.queryNorm,
                              rowsOf<float>(backend_, b.ropeCos, lane * laneRows, laneRows, geometry_.rotaryPairs),
                              rowsOf<float>(backend_, b.ropeSin, lane * laneRows, laneRows, geometry_.rotaryPairs),
                              b.pageTables[lane], h.qsaRawKeys[attentionLayer], queries, qsa);
    ops::Qwen4::addQsaPool(step.graph, b.qsaBlocks[lane], b.qsaBlockCounts[lane], h.qsaRawKeys[attentionLayer],
                           indexer.keyNorm, h.inverseFrequencies, b.pageTables[lane], h.qsaPooledKeys[attentionLayer]);
  }
  if (sparse)
    for (uint32_t lane = 0; lane < step.lanes; ++lane)
      ops::Qwen4::addQsaSelect(
          step.graph, rowsOf<float>(backend_, h.qsaQueries, lane * laneRows, laneRows, indexWidth),
          h.qsaPooledKeys[attentionLayer], b.pageTables[lane], h.qsaScores, h.qsaScoreRows,
          rowsOf<uint32_t>(backend_, h.qsaMask, lane * laneRows, laneRows, h.qsaMaskWords),
          {laneRows, step.verify[lane].committed_tokens, topBlocks, h.qsaMaskWords, h.qsaMaskWords * 32});
  ops::PagedAttention::addVerifyProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                           b.ropeSin, b.fullQueries, b.chunkKeys[attentionLayer],
                                           b.chunkValues[attentionLayer], laneRows, tileRows, tileRows,
                                           geometry_.attentionQueryHeads, geometry_.kvLayout, step.lanes);
  ops::PagedAttention::addVerify(step.graph, step.kvLayers[attentionLayer],
                                 {b.chunkKeys[attentionLayer], b.chunkValues[attentionLayer], b.fullQueries,
                                  b.attentionPartials, b.attentionStatistics, b.fullAttention, b.pageTables},
                                 step.q8, step.verify, step.attention,
                                 ops::QsaMask{h.qsaMask, sparse ? h.qsaMaskWords : 0, 0});
  const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
      step.graph, b.fullPacked, b.fullAttention, b.attentionHidden, laneRows, tileRows, tileRows,
      geometry_.attentionQueryHeads, geometry_.kvLayout, step.lanes, b.linearScratch,
      linear.decodePlan(mixer.outputProjection, step.lanes).input());
  project(b.attentionHidden, mixer.outputProjection, h.branch, hidden);
}

void QwenTarget::addQwen4Verify(VerifyStep &step, const Qwen4ExpWeights &weights) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const QwenHyperBuffers &h = b.hyper;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape(geometry_);
  const uint32_t rows = step.rows;
  const auto project = [&](metal::MetalBuffer input, const ops::Projection &projection, metal::MetalBuffer output,
                           ops::PreparedInput prepared = {}) {
    linear.addDecodeBatch(step.graph, std::move(input), projection, std::move(output), step.lanes, step.stats,
                          b.linearScratch, std::move(prepared));
  };
  ops::Qwen4::addInit(step.graph, b.hidden[0], h.streams, hs, rows);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const Qwen4ExpLayerWeights &layer = weights.layers[index];
    if (layer.ple) {
      const Qwen4PleWeights &ple = *layer.ple;
      const ops::PleShape ps = pleShape(geometry_);
      ops::Embedding::add(step.graph, h.pleIndices, weights.pleTable, h.pleRows, rows * geometry_.pleHeads);
      project(h.pleRows, ple.key, h.gate);
      project(h.pleRows, ple.value, h.pleValue);
      ops::Qwen4::addPleGate(step.graph, h.gate, h.pleValue, h.streams, ple.keyNorm, ple.queryNorm,
                             ple.convolutionNorm, h.pleGated, h.pleConvolution, ps, rows);
      // Each lane's eight rows after its history; the commit writes the
      // history after its retained rows.
      constexpr uint32_t laneRows = ExecutionLimits::targetVerifyRows;
      for (uint32_t lane = 0; lane < step.lanes; ++lane)
        ops::Qwen4::addPleConvolution(
            step.graph, rowsOf<uint16_t>(backend_, h.pleConvolution, lane * laneRows, laneRows, ps.width()),
            b.pleHistories[lane], {}, ple.convolution,
            rowsOf<float>(backend_, h.pleGated, lane * laneRows, laneRows, ps.width()),
            rowsOf<float>(backend_, h.streams, lane * laneRows, laneRows, ps.width()), ps, laneRows);
    }
    addQwen4VerifyMix(step, layer.mixerHyper, h.streams, b.normalized);
    std::visit(
        [&](const auto &mixer) {
          using Mixer = std::remove_cvref_t<decltype(mixer)>;
          if constexpr (std::is_same_v<Mixer, QwenGdnWeights>) {
            const uint32_t gdnLayer = step.gdnLayer++;
            project(b.normalized, mixer.inputProjection, b.gdnPacked[gdnLayer]);
            const ops::PreparedInput hidden = ops::GDN::addDecode(
                step.graph,
                {b.gdnPacked[gdnLayer], mixer.convolutionWeights, b.currentGdnStates, b.nextGdnStates,
                 b.gdnMixed[gdnLayer], mixer.decay, mixer.timeBias, b.gdnDecay[gdnLayer], b.gdnBeta[gdnLayer],
                 b.recurrent, mixer.mixerNorm, b.gdnHidden, b.arrived, b.generation, b.linearScratch},
                geometry_.gdnShape(), step.lanes, gdnLayer,
                {geometry_.stateLayout.convolutionLayerBytes(), geometry_.stateLayout.recurrentLayerBytes(),
                 geometry_.stateLayout.convolutionBytes()},
                mixer.outputHeadOrder, linear.decodePlan(mixer.outputProjection, step.lanes).input());
            project(b.gdnHidden, mixer.outputProjection, h.branch, hidden);
          } else {
            addQwen4VerifyAttention(step, layer, mixer, step.attentionLayer++);
          }
        },
        layer.mixer);
    ops::Qwen4::addCombine(step.graph, h.streams, h.branch, h.weights, hs, rows);
    addQwen4VerifyMix(step, layer.ffnHyper, h.streams, b.normalized);
    ops::MoE::add(step.graph, {b.normalized, h.zeros, h.branch, b.moe}, layer.ffn, *step.moe);
    ops::Qwen4::addCombine(step.graph, h.streams, h.branch, h.weights, hs, rows);
  }
  requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  addQwen4VerifyMix(step, weights.finalHyper, h.streams, b.finalHidden);
  project(b.finalHidden, weights.logitsProjection, b.logits);
  // The MTP head's KV at the step's rows: row r pairs the trunk's streams of
  // the position before with the row's token. The head reads them as its
  // context at the next steps; the rows past each lane's accepted ones are
  // overwritten before any later step reads them.
  if (weights.mtp) addQwen4MtpVerify(step, weights, b.hidden[0], h.streams, ExecutionLimits::targetVerifyRows, false);
}

metal::MetalBuffer QwenTarget::mtpCarried(const metal::MetalBuffer &auxiliary) const {
  const uint64_t pleBytes = uint64_t{geometry_.pleHistory()} * geometry_.streamWidth() * sizeof(uint16_t);
  return backend_.view(auxiliary, pleBytes, uint64_t{geometry_.streamWidth()} * sizeof(float));
}

void QwenTarget::addQwen4MtpVerify(VerifyStep &step, const Qwen4ExpWeights &weights, metal::MetalBuffer embedding,
                                   metal::MetalBuffer history, uint32_t limit, bool full) const {
  constexpr uint32_t laneRows = ExecutionLimits::targetVerifyRows;
  const Qwen4MtpWeights &mtp = *weights.mtp;
  const QwenTargetVerifyBuffers &b = step.buffers;
  const QwenHyperBuffers &h = b.hyper;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape(geometry_);
  const uint32_t rows = step.rows, hidden = geometry_.hiddenSize, width = geometry_.streamWidth();
  for (uint32_t lane = 0; lane < step.lanes; ++lane)
    ops::Qwen4::addMtpInput(step.graph, rowsOf<uint16_t>(backend_, embedding, lane * laneRows, laneRows, hidden),
                            mtpCarried(b.pleHistories[lane]),
                            rowsOf<float>(backend_, history, lane * laneRows, laneRows, width), mtp.embedNorm,
                            mtp.hiddenNorm, h.mtpInput, hs, laneRows, limit, lane * laneRows, rows);
  // The eh projection of each stream's rows, then the fp32 streams.
  for (uint32_t s = 0; s < hs.streams; ++s)
    linear.addDecodeBatch(step.graph, rowsFrom<uint16_t>(backend_, h.mtpInput, s * rows, 2 * hidden),
                          mtp.embedHidden, rowsFrom<uint16_t>(backend_, h.gate, s * rows, hidden), step.lanes,
                          step.stats, b.linearScratch);
  ops::Qwen4::addMtpWiden(step.graph, h.gate, h.mtpStreams, hs, rows);
  const uint32_t attentionLayer = geometry_.kvLayout.attentionLayers - geometry_.mtpLayers;
  addQwen4VerifyMix(step, mtp.block.mixerHyper, h.mtpStreams, b.normalized);
  addQwen4VerifyAttention(step, mtp.block, std::get<QwenAttentionWeights>(mtp.block.mixer), attentionLayer);
  if (!full) return;
  ops::Qwen4::addCombine(step.graph, h.mtpStreams, h.branch, h.weights, hs, rows);
  addQwen4VerifyMix(step, mtp.block.ffnHyper, h.mtpStreams, b.normalized);
  ops::MoE::add(step.graph, {b.normalized, h.zeros, h.branch, b.moe}, mtp.block.ffn, *step.moe);
  ops::Qwen4::addCombine(step.graph, h.mtpStreams, h.branch, h.weights, hs, rows);
  addQwen4VerifyMix(step, mtp.head, h.mtpStreams, b.finalHidden);
  linear.addDecodeBatch(step.graph, b.finalHidden, weights.logitsProjection, b.logits, step.lanes, step.stats,
                        b.linearScratch);
}

void QwenTarget::addMtpDraft(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
                             std::span<const kv::LayerStorage> kvLayers,
                             std::span<const kv::Q8ChunkedPrefillParams> q8,
                             std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes, uint32_t steps,
                             ops::LinearDispatchStats &stats) const {
  const Qwen4ExpWeights *weights = qwen4();
  constexpr uint32_t laneRows = ExecutionLimits::targetVerifyRows;
  if (!weights || !weights->mtp || !lanes || lanes > ExecutionLimits::maximumBatchWidth || !steps ||
      steps > buffers.proposalsPerLane || buffers.proposalsPerLane + 1 > laneRows ||
      q8.size() != ExecutionLimits::maximumBatchWidth || verify.size() != ExecutionLimits::maximumBatchWidth ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers)
    throw std::invalid_argument("invalid MTP draft batch");
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane) histories[lane] = verify[lane].committed_tokens;
  VerifyStep step{graph, buffers, kvLayers, q8, verify, lanes, lanes * laneRows, stats,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.kvLayout, histories)};
  step.moe = operators_.moeDecode(geometry_.moeShape(), lanes);
  const QwenHyperBuffers &h = buffers.hyper;
  // Step j reads rows 0..j-1 (the anchor and the proposals so far, each with
  // the streams of the row before) and drafts proposal j - 1 from row j - 1.
  // The rows before it recompute what the steps before wrote; the later rows
  // repeat it, so they route to its experts.
  for (uint32_t j = 1; j <= steps; ++j) {
    ops::Qwen4::addMtpTokens(graph, buffers.anchors, buffers.proposals, h.mtpTokens, lanes, laneRows,
                             buffers.proposalsPerLane, j);
    ops::Embedding::add(graph, h.mtpTokens, weightsBase_.tokenEmbedding, buffers.hidden[1], step.rows);
    addQwen4MtpVerify(step, *weights, buffers.hidden[1], h.mtpStreams, j, true);
    ops::Qwen4::addMtpArgmax(graph, buffers.logits, buffers.proposals, buffers.candidates, lanes, laneRows,
                             geometry_.vocabularySize, buffers.proposalsPerLane, buffers.candidatesPerProposal,
                             j - 1, j == steps);
  }
}

void QwenTarget::addPleHash(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer before,
                            metal::MetalBuffer indices, uint32_t lanes) const {
  const Qwen4ExpWeights *weights = qwen4();
  if (!weights || !lanes || lanes > ExecutionLimits::maximumBatchWidth) throw std::invalid_argument("invalid PLE hash");
  const Qwen4ExpLayout &layout = weights->layout;
  const Qwen4PleHash &hash = weights->pleHash;
  Qwen4PleHashParams params{};
  params.rows = lanes * ExecutionLimits::targetVerifyRows;
  params.lane_rows = ExecutionLimits::targetVerifyRows;
  params.ngram = layout.pleNgram;
  params.heads_per_ngram = layout.pleHeadsPerNgram;
  params.eos = layout.pleEosToken;
  const uint32_t heads = layout.pleHeads();
  if (heads > QWEN4_PLE_MAXIMUM_HEADS || hash.headVocabularies.size() != heads || hash.headOffsets.size() != heads)
    throw std::logic_error("PLE hash geometry mismatch");
  for (uint32_t i = 0; i < 4; ++i) params.multipliers[i] = hash.multipliers[i];
  for (uint32_t head = 0; head < heads; ++head) {
    params.vocabularies[head] = hash.headVocabularies[head];
    params.offsets[head] = hash.headOffsets[head];
  }
  ops::Qwen4::addPleHash(graph, std::move(tokens), std::move(before), std::move(indices), params, heads);
}

} // namespace splash::model
