#include "model/QwenTarget.hpp"

#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
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
      attentionLayers != geometry.kvLayout.attentionLayers) {
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

void requireLayerPartition(const QwenTargetGeometry &geometry, uint32_t gdnLayers, uint32_t attentionLayers) {
  if (gdnLayers != geometry.stateLayout.layers || attentionLayers != geometry.kvLayout.attentionLayers)
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
  std::visit([&](const auto *weights) {
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
  const ops::Linear &linear = operators_.linear();
  std::visit([&](const auto *weights) {
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

} // namespace splash::model
