#include "model/QwenTarget.hpp"

#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/WeightStore.hpp"
#include "ops/AneFfn.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace splash::model {
namespace {

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
                       const QwenTargetGeometry &geometry,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), weightsBase_(weights), geometry_(geometry),
      backend_(backend), operators_(operators) {
  requireWeights(weights, geometry_);
}

template QwenTarget::QwenTarget(const Qwen3_8Weights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template QwenTarget::QwenTarget(const Qwen3_6MoeWeights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);

namespace {

void includeProjection(QwenTargetGeometry &geometry, const ops::Projection &projection) {
  geometry.decodeProjections.push_back(projection.shape());
}

// The projections a dense layer's FFN dispatches; a MoE block runs its own
// plans (ops::MoeShape). Its gate and up run as one gate/up plan, which takes
// them of one size and rotation in any formats.
void includeFfn(QwenTargetGeometry &geometry, const Qwen3_8LayerWeights &layer) {
  const ops::ProjectionShape gate = layer.gateProjection.shape(), up = layer.upProjection.shape();
  if (gate.outputSize != up.outputSize || gate.inputSize != up.inputSize || gate.rotated != up.rotated)
    throw WeightStoreError("fused gate/up projections must have matching shapes");
  includeProjection(geometry, layer.gateProjection);
  includeProjection(geometry, layer.upProjection);
  includeProjection(geometry, layer.downProjection);
  geometry.gateUpProjections.push_back(up);
}
void includeFfn(QwenTargetGeometry &, const Qwen3_6MoeLayerWeights &) {}

// The format of most routed expert weights of a target's MoE blocks,
// GGUF_FMT_COUNT for none (ops::MoeShape::expertFormat).
uint32_t routedExpertFormat(std::span<const Qwen3_8LayerWeights>) { return GGUF_FMT_COUNT; }
uint32_t routedExpertFormat(std::span<const Qwen3_6MoeLayerWeights> layers) {
  std::array<uint64_t, GGUF_FMT_COUNT> weights{};
  for (const Qwen3_6MoeLayerWeights &layer : layers) {
    const ops::BlockMoeWeights &block = layer.ffn;
    for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
      if (!projection->routed.isFloat())
        weights[projection->routed.formatId] += uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
  }
  const auto most = std::max_element(weights.begin(), weights.end());
  return *most ? uint32_t(most - weights.begin()) : GGUF_FMT_COUNT;
}
// Whether the register tile takes every routed and shared expert weight of a
// target's MoE blocks (ops::MoeShape::registerFormats).
bool expertRegisterFormats(std::span<const Qwen3_8LayerWeights>) { return true; }
bool expertRegisterFormats(std::span<const Qwen3_6MoeLayerWeights> layers) {
  for (const Qwen3_6MoeLayerWeights &layer : layers)
    for (const ops::BlockExpertProjection *projection : {&layer.ffn.gate, &layer.ffn.up, &layer.ffn.down})
      for (const ops::QuantizedSegment *s : {&projection->routed, &projection->shared})
        if (!s->isFloat() && !quant_register_tensor(s->formatId, s->inputSize)) return false;
  return true;
}

} // namespace

template <class Layout, class Layer>
QwenTargetGeometry qwenTargetGeometry(const QwenTargetWeights<Layout, Layer> &weights,
                                      std::span<const uint32_t> captureLayers) {
  const Layout &layout = weights.layout;
  QwenTargetGeometry geometry(layout, captureLayers);
  geometry.kvLayout = layout.kvLayout();
  geometry.stateLayout = layout.gdnStateLayout();
  for (const auto &layer : weights.layers) {
    std::visit([&](const auto &mixer) {
      includeProjection(geometry, mixer.inputProjection);
      includeProjection(geometry, mixer.outputProjection);
    }, layer.mixer);
    includeFfn(geometry, layer);
  }
  geometry.moeExpertFormat = routedExpertFormat(weights.layers);
  geometry.moeRegisterFormats = expertRegisterFormats(weights.layers);
  geometry.prefillProjections = geometry.decodeProjections;
  includeProjection(geometry, weights.logitsProjection);
  for (auto *shapes : {&geometry.prefillProjections, &geometry.decodeProjections,
                       &geometry.gateUpProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
  return geometry;
}

template QwenTargetGeometry qwenTargetGeometry(const Qwen3_8Weights &, std::span<const uint32_t>);
template QwenTargetGeometry qwenTargetGeometry(const Qwen3_6MoeWeights &, std::span<const uint32_t>);

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

// Copies `rows` rows of a capture layer's output, from row `sourceRow`, into
// capture slot `slot` of the captured hidden rows from row `destinationRow`.
void addCapture(metal::CommandGraph &graph, const QwenTargetGeometry &geometry, uint32_t slot,
                metal::MetalBuffer output, uint32_t sourceRow, metal::MetalBuffer captured,
                uint32_t destinationRow, uint32_t rows) {
  const uint32_t width = geometry.hiddenSize, capturedWidth = geometry.capturedHiddenSize();
  if (slot >= capturedWidth / width)
    throw std::logic_error("Qwen target capture slot past the captured hidden rows");
  ops::RowCopy::add(graph, std::move(output), {sourceRow, width, 0}, std::move(captured),
                    {destinationRow, capturedWidth, slot * width}, rows, width);
}

// A layer's MoE block, null for a dense FFN.
const ops::BlockMoeWeights *moeOf(const Qwen3_8LayerWeights &) { return nullptr; }
template <class Layer> const ops::BlockMoeWeights *moeOf(const Layer &layer) { return &layer.ffn; }

} // namespace

// The state a prefill command's layers share: its inputs, the buffers the
// layer's MoE routes and runs through, and the next GDN and attention layer
// of the step.
struct QwenTarget::PrefillStep {
  metal::CommandGraph &graph;
  const QwenTargetPrefillBuffers &buffers;
  std::span<const QwenTargetPrefillSequence> sequences;
  uint32_t rows;
  std::span<const SplashKvLayer> kvLayers;
  // Each sequence's attention plan, which every attention layer runs.
  std::vector<ops::PrefillAttentionPlan> attention{};
  std::optional<ops::MoePlan> moe{};
  // The layer's MoE buffers, which its routing (addPrefillLayer), the layer's
  // load and its routed work (addPrefillExperts) share.
  ops::MoeBuffers ffn{};
  ops::AneFfn *aneFfn = nullptr;
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
};

struct QwenTarget::VerifyStep {
  metal::CommandGraph &graph;
  const QwenTargetVerifyBuffers &buffers;
  std::span<const SplashKvLayer> kvLayers;
  std::span<const kv::ChunkedPrefillParams> chunks;
  uint32_t lanes;
  uint32_t rows;
  ops::VerifyAttentionPlan attention;
  std::optional<ops::MoePlan> moe{};
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
};

metal::MetalBuffer QwenTarget::addPrefill(
    metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const SplashKvLayer> kvLayers, ops::AneFfn *aneFfn) const {
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen ragged prefill batch");
  }
  const auto everyLayer = [&](std::span<const metal::MetalBuffer> views) {
    return views.size() == geometry_.stateLayout.layers;
  };
  for (const QwenTargetPrefillSequence &sequence : sequences) {
    if (!everyLayer(sequence.convolutionIn) || !everyLayer(sequence.convolutionOut) ||
        !everyLayer(sequence.recurrentIn) || !everyLayer(sequence.recurrentOut) ||
        !std::ranges::all_of(sequence.stateCaptures, [&](const QwenTargetStateCapture &capture) {
          return everyLayer(capture.convolution) && everyLayer(capture.recurrent);
        })) {
      throw std::invalid_argument("Qwen prefill state layer mismatch");
    }
  }
  PrefillStep step{graph, buffers, sequences, rows, kvLayers};
  for (const QwenTargetPrefillSequence &sequence : sequences)
    step.attention.push_back(operators_.prefillAttention(
        sequence.rows, geometry_.attentionQueryHeads, geometry_.kvLayout));
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moePrefill(geometry_.moeShape(), rows);
  if (aneFfn && aneFfn->splits(rows)) step.aneFfn = aneFfn;
  std::visit([&](const auto *weights) {
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      addPrefillLayer(step, *weights, index);
      // Streamed experts: the load of those the step's rows route.
      if (const ops::BlockMoeWeights *ffn = moeOf(weights->layers[index]); ffn && ffn->stream) {
        const ops::MoeLoadPart part{&step.ffn, &*step.moe};
        ffn->stream->addLoad(graph, {&part, 1}, ffn->streamBlock);
      }
      addPrefillExperts(step, *weights, index);
    }
  }, weights_);
  requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  return buffers.hidden[geometry_.layers & 1];
}

// Qwen3.5 and Qwen3.6 (model/Qwen3_8.hpp, model/Qwen3_6Moe.hpp). Layer l
// reads its input rows from buffers.hidden[l & 1] and adds its mixer's and
// its FFN's output to them into the other; the layers the draft reads copy
// their output rows into buffers.captured.
template <class Layout, class Layer>
void QwenTarget::addPrefillLayer(PrefillStep &step, const QwenTargetWeights<Layout, Layer> &weights,
                                 uint32_t index) const {
  const Layer &layer = weights.layers[index];
  const QwenTargetPrefillBuffers &b = step.buffers;
  const metal::MetalBuffer input = b.hidden[index & 1];
  const metal::MetalBuffer residual = std::visit(
      [&](const auto &mixer) { return addPrefillMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
  addPrefillFfn(step, index, layer, residual, b.hidden[(index & 1) ^ 1]);
}

template <class Layout, class Layer>
void QwenTarget::addPrefillExperts(PrefillStep &step, const QwenTargetWeights<Layout, Layer> &weights,
                                   uint32_t index) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  if (const ops::BlockMoeWeights *ffn = moeOf(weights.layers[index]))
    ops::MoE::addRouted(step.graph, step.ffn, *ffn, *step.moe);
  if (const auto slot = geometry_.captureSlot(index))
    for (const QwenTargetPrefillSequence &sequence : step.sequences)
      for (uint32_t capture = 0; capture < sequence.captureCount; ++capture) {
        const QwenTargetPrefillCapture &c = sequence.captures[capture];
        addCapture(step.graph, geometry_, *slot, b.hidden[(index & 1) ^ 1], c.sourceStart, b.captured,
                   c.destinationStart, c.rows);
      }
}

// The normalized rows a layer's prefill projections read.
void QwenTarget::addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm) const {
  ops::Normalization::addRms(step.graph, input, norm, step.buffers.normalized, geometry_.hiddenSize, step.rows);
}

// The mixer output projection adds the mixer's rows to `input`.
void QwenTarget::addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                                  metal::MetalBuffer input, metal::MetalBuffer output) const {
  operators_.linear().addPrefillResidual(step.graph, hidden, projection, input, output, step.rows,
                                         step.buffers.linearScratch);
}

metal::MetalBuffer QwenTarget::addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.gdnLayer++;
  addPrefillNorm(step, input, norm);
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked, step.rows,
                                 b.linearScratch);
  for (const QwenTargetPrefillSequence &sequence : step.sequences) {
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<float>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    std::vector<ops::GdnPrefillCapture> captures;
    captures.reserve(sequence.stateCaptures.size());
    for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
      captures.push_back({capture.row, capture.convolution[layer], capture.recurrent[layer]});
    ops::GDN::addPrefill(
        step.graph, backend_,
        {u16(b.gdnPacked, geometry_.packedGdnWidth), mixer.convolutionWeights, sequence.convolutionIn[layer],
         sequence.convolutionOut[layer], u16(b.gdnQueries, geometry_.gdnKeyWidth()),
         u16(b.gdnKeys, geometry_.gdnKeyWidth()), u16(b.gdnValues, geometry_.attentionWidth), mixer.decay,
         mixer.timeBias, f32(b.gdnDecay, geometry_.gdnValueHeads), u16(b.gdnBeta, geometry_.gdnValueHeads),
         sequence.recurrentIn[layer], sequence.recurrentOut[layer], u16(b.recurrent, geometry_.attentionWidth),
         mixer.mixerNorm, u16(b.gdnHidden, geometry_.attentionWidth)},
        geometry_.gdnShape(), sequence.rows, geometry_.stateLayout.format, mixer.outputHeadOrder, captures);
  }
  addPrefillOutput(step, b.gdnHidden, mixer.outputProjection, input, b.gdnOutput);
  return b.gdnOutput;
}

metal::MetalBuffer QwenTarget::addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.attentionLayer++;
  addPrefillNorm(step, input, norm);
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, step.rows,
                                 b.linearScratch);
  for (size_t index = 0; index < step.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
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
        sequence.rows, sequence.attentionStride, geometry_.attentionQueryHeads, geometry_.kvLayout);
    ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[layer], keys, values, sequence.pageTable,
                                         sequence.chunk, geometry_.kvLayout);
    ops::PagedAttention::addPrefill(
        step.graph, step.kvLayers[layer], queries, attentionRows, b.attentionPartials, b.attentionStatistics,
        sequence.pageTable, sequence.chunk, step.attention[index]);
    ops::PagedAttention::addPrefillGate(
        step.graph, u16(b.fullPacked, geometry_.packedFullWidth), attentionRows,
        u16(b.attentionHidden, geometry_.attentionWidth), sequence.rows, sequence.attentionStride,
        geometry_.attentionQueryHeads, geometry_.kvLayout);
  }
  addPrefillOutput(step, b.attentionHidden, mixer.outputProjection, input, b.attentionOutput);
  return b.attentionOutput;
}

void QwenTarget::addPrefillFfn(PrefillStep &step, uint32_t index, const Qwen3_8LayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  addPrefillNorm(step, residual, layer.postAttentionNorm);
  const ops::PrefillFfnBuffers ffn = step.buffers.ffn();
  if (step.aneFfn)
    step.aneFfn->add(step.graph, index, ffn, residual, output, step.rows);
  else
    operators_.linear().addPrefillSwiGlu(step.graph, {&layer.gateProjection, &layer.upProjection,
                                                      &layer.downProjection},
                                         ffn, residual, output, step.rows);
}

void QwenTarget::addPrefillFfn(PrefillStep &step, uint32_t, const Qwen3_6MoeLayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  addPrefillNorm(step, residual, layer.postAttentionNorm);
  step.ffn = {b.normalized, residual, output, b.moe};
  ops::MoE::addRouting(step.graph, step.ffn, layer.ffn, *step.moe);
}

void QwenTarget::addVerify(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const SplashKvLayer> kvLayers,
    std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || chunks.size() != lanes ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = chunks[lane].committed_tokens;
  VerifyStep step{graph, buffers, kvLayers, chunks, lanes, rows,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.kvLayout,
                                             std::span(histories).first(lanes))};
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moeDecode(geometry_.moeShape(), lanes);
  std::visit([&](const auto *weights) {
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      const auto &layer = weights->layers[index];
      const metal::MetalBuffer input = buffers.hidden[index & 1];
      const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
      const metal::MetalBuffer residual = std::visit(
          [&](const auto &mixer) { return addVerifyMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
      addVerifyFfn(step, layer, residual, output);
      if (const auto slot = geometry_.captureSlot(index))
        addCapture(graph, geometry_, *slot, output, 0, buffers.capturedTargetHidden, 0, rows);
    }
    requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  }, weights_);
  addHeadBatch(graph, buffers.hidden[geometry_.layers & 1], buffers.finalHidden, buffers.logits, lanes,
               buffers.linearScratch);
}

// Each producer emits the table (if any) its consumer's plan reads.
metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const QwenGdnWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.gdnLayer++;
  const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.lanes);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch, inputPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.gdnPacked, .scratch = b.linearScratch,
              .prepared = normalized},
             mixer.inputProjection, inputPlan);
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.lanes, ops::LinearEpilogue::Residual);
  const ops::PreparedInput hidden = ops::GDN::addDecode(
      step.graph,
      {b.gdnPacked, mixer.convolutionWeights, b.currentGdnStates, b.nextGdnStates, b.gdnTape,
       std::span(b.gdnTapeLanes).first(step.lanes), mixer.decay, mixer.timeBias, mixer.mixerNorm, b.gdnHidden,
       b.linearScratch},
      geometry_.gdnShape(), step.lanes, layer, geometry_.stateLayout.strides(),
      mixer.outputHeadOrder, outputPlan.input());
  linear.add(step.graph,
             {.input = b.gdnHidden, .output = b.gdnOutput, .residual = input, .scratch = b.linearScratch,
              .prepared = hidden},
             mixer.outputProjection, outputPlan);
  return b.gdnOutput;
}

metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer++;
  const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.lanes);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch, inputPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.fullPacked, .scratch = b.linearScratch, .prepared = normalized},
             mixer.inputProjection, inputPlan);
  ops::PagedAttention::addVerifyProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                           b.ropeSin, b.fullQueries, b.chunkKeys[layer], b.chunkValues[layer],
                                           geometry_.attentionQueryHeads, geometry_.kvLayout, step.lanes);
  ops::PagedAttention::addVerify(step.graph, step.kvLayers[layer],
                                 {b.chunkKeys[layer], b.chunkValues[layer], b.fullQueries, b.attentionPartials,
                                  b.attentionStatistics, b.fullAttention, b.pageTables},
                                 step.chunks, step.attention);
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.lanes, ops::LinearEpilogue::Residual);
  const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
      step.graph, b.fullPacked, b.fullAttention, b.attentionHidden, geometry_.attentionQueryHeads,
      geometry_.kvLayout, step.lanes, b.linearScratch, outputPlan.input());
  linear.add(step.graph,
             {.input = b.attentionHidden, .output = b.attentionOutput, .residual = input,
              .scratch = b.linearScratch, .prepared = hidden},
             mixer.outputProjection, outputPlan);
  return b.attentionOutput;
}

void QwenTarget::addVerifyFfn(VerifyStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan gateUpPlan =
      linear.decodePlan(layer.upProjection, step.lanes, ops::LinearEpilogue::GateUp, &layer.gateProjection);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize, step.rows,
      b.linearScratch, gateUpPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.denseIntermediate, .gateScratch = b.denseGateScratch,
              .scratch = b.linearScratch, .prepared = normalized},
             layer.upProjection, gateUpPlan, &layer.gateProjection);
  linear.add(step.graph,
             {.input = b.denseIntermediate, .output = output, .residual = residual, .scratch = b.linearScratch},
             layer.downProjection,
             linear.decodePlan(layer.downProjection, step.lanes, ops::LinearEpilogue::Residual));
}

void QwenTarget::addVerifyFfn(VerifyStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                             step.rows);
  ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, layer.ffn, *step.moe);
}

void QwenTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                              metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
                              ops::LinearScratch scratch) const {
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan logitsPlan = linear.decodePlan(vocabularyProjection(), lanes);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      graph, std::move(hidden), weightsBase_.finalNorm, finalHidden, geometry_.hiddenSize,
      lanes * ExecutionLimits::targetVerifyRows, scratch, logitsPlan.input());
  linear.add(graph,
             {.input = std::move(finalHidden), .output = std::move(logits), .scratch = scratch,
              .prepared = normalized},
             vocabularyProjection(), logitsPlan);
}

void QwenTarget::addVerifyInput(metal::CommandGraph &graph,
                                metal::MetalBuffer draftInput,
                                metal::MetalBuffer proposals,
                                metal::MetalBuffer verifyInput,
                                uint32_t lanes) const {
  ops::Embedding::addVerifyInput(graph, std::move(draftInput),
                                 std::move(proposals), std::move(verifyInput),
                                 geometry_.vocabularySize, lanes);
}

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, std::move(hidden),
                      rows);
}

} // namespace splash::model
