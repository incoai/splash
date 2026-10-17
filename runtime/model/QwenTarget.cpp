#include "model/QwenTarget.hpp"

#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/WeightStore.hpp"
#include "ops/AneFfn.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"
#include "ops/PageStorage.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace splash::model {
namespace {

// The rows a decode step verifies of each of its lanes.
constexpr uint32_t kVerifyRows = ExecutionLimits::targetVerifyRows;

// Whether the register tile takes every routed and shared expert weight of a
// MoE block (ops::MoeShape::registerFormats).
bool registerTakes(const ops::BlockMoeWeights &block) {
  for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
    for (const ops::QuantizedSegment *s : {&projection->routed, &projection->shared})
      if (!s->isFloat() && !quant_register_tensor(s->formatId, s->inputSize)) return false;
  return true;
}

// A prefill sequence binds every GDN layer's state cells, in and out, and so
// does each state it captures; with auxiliary state, its cells too.
void requirePrefillStates(const QwenTargetGeometry &geometry, const QwenTargetPrefillSequence &sequence) {
  const auto everyLayer = [&](std::span<const metal::MetalBuffer> views) {
    return views.size() == geometry.stateLayout.layers;
  };
  const bool auxiliary = geometry.stateLayout.auxiliaryBytes != 0;
  if (!everyLayer(sequence.convolutionIn) || !everyLayer(sequence.convolutionOut) ||
      !everyLayer(sequence.recurrentIn) || !everyLayer(sequence.recurrentOut) ||
      (auxiliary && (!sequence.auxiliaryIn || !sequence.auxiliaryOut)) ||
      !std::ranges::all_of(sequence.stateCaptures, [&](const QwenTargetStateCapture &capture) {
        return everyLayer(capture.convolution) && everyLayer(capture.recurrent) && (capture.auxiliary || !auxiliary);
      }))
    throw std::invalid_argument("Qwen prefill state layer mismatch");
}

} // namespace

void QwenTargetGeometry::finish(std::span<const ops::BlockMoeWeights *const> moeBlocks,
                                const ops::Projection &logits) {
  // The format of most routed expert weights, GGUF_FMT_COUNT for none.
  std::array<uint64_t, GGUF_FMT_COUNT> weights{};
  moeRegisterFormats = true;
  for (const ops::BlockMoeWeights *block : moeBlocks) {
    moeRegisterFormats &= registerTakes(*block);
    for (const ops::BlockExpertProjection *projection : {&block->gate, &block->up, &block->down})
      if (!projection->routed.isFloat())
        weights[projection->routed.formatId] += uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
  }
  const auto most = std::max_element(weights.begin(), weights.end());
  moeExpertFormat = *most ? uint32_t(most - weights.begin()) : GGUF_FMT_COUNT;
  prefillProjections = decodeProjections;
  include(logits);
  for (auto *shapes : {&prefillProjections, &decodeProjections, &gateUpProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
}

const ops::BlockMoeWeights *qwenMoeBlock(const Qwen3_8LayerWeights &) noexcept { return nullptr; }

// A dense layer's FFN runs its gate and up as one gate/up plan, which takes
// them of one size and rotation in any formats.
void includeFamilyProjections(QwenTargetGeometry &geometry, const Qwen3_8Weights &weights) {
  for (const Qwen3_8LayerWeights &layer : weights.layers) {
    const ops::ProjectionShape gate = layer.gateProjection.shape(), up = layer.upProjection.shape();
    if (gate.outputSize != up.outputSize || gate.inputSize != up.inputSize || gate.rotated != up.rotated)
      throw WeightStoreError("fused gate/up projections must have matching shapes");
    geometry.include(layer.gateProjection);
    geometry.include(layer.upProjection);
    geometry.include(layer.downProjection);
    geometry.gateUpProjections.push_back(up);
  }
}
void includeFamilyProjections(QwenTargetGeometry &, const Qwen3_6MoeWeights &) {}

QwenTarget::QwenTarget(const QwenTargetWeightsBase &weights, const QwenTargetGeometry &geometry,
                       metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
    : weightsBase_(weights), geometry_(geometry), backend_(backend), operators_(operators) {
  if (!geometry_.valid()) throw std::invalid_argument("Qwen target weights do not match execution geometry");
}

QwenTarget::~QwenTarget() = default;

QwenTargetScratchBytes QwenTarget::allocatedScratchBytes() const noexcept { return {}; }

const ops::Projection &QwenTarget::vocabularyProjection() const noexcept {
  return weightsBase_.logitsProjection;
}

uint32_t QwenTarget::decodeStorageLanes(uint32_t lanes) const {
  const uint32_t rows = lanes * kVerifyRows;
  uint32_t storageRows = rows;
  for (const auto &shape : geometry_.decodeProjections)
    storageRows = std::max(storageRows, operators_.linear().decodeStorageRows(rows, shape));
  return storageRows / kVerifyRows;
}

void QwenTarget::prepareVerifyLane(uint32_t, uint64_t, const QwenTargetRequest &) {}

std::vector<uint32_t> QwenTarget::verifyTableRows(uint32_t) const { return {}; }

void QwenTarget::addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer draftInput,
                                metal::MetalBuffer proposals, metal::MetalBuffer verifyInput,
                                uint32_t lanes) const {
  ops::Embedding::addVerifyInput(graph, std::move(draftInput), std::move(proposals), std::move(verifyInput),
                                 geometry_.vocabularySize, lanes);
}

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, std::move(hidden),
                      rows);
}

void QwenTarget::addStateCommit(metal::CommandGraph &, QwenTargetCommitBuffers buffers, uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  if (geometry_.stateLayout.auxiliaryBytes)
    for (uint32_t lane = 0; lane < lanes; ++lane)
      if (!buffers.auxiliaryIn[lane] || !buffers.auxiliaryOut[lane])
        throw std::invalid_argument("Qwen state commit lane without its auxiliary state");
}

void QwenTarget::requireLayers(size_t layers, size_t attentionLayers) const {
  if (layers != geometry_.layers || attentionLayers != geometry_.kvLayout.attentionLayers)
    throw std::invalid_argument("Qwen target weights do not match execution geometry");
}

void QwenTarget::requireLayerPartition(uint32_t gdnLayers, uint32_t attentionLayers) const {
  if (gdnLayers != geometry_.stateLayout.layers || attentionLayers != geometry_.kvLayout.attentionLayers)
    throw std::logic_error("Qwen target layer partition mismatch");
}

void QwenTarget::requirePrefill(std::span<const QwenTargetPrefillChunk> chunks, const kv::PageStorage &kv,
                                const ops::AneFfn *aneFfn) const {
  // The Neural Engine splits a dense target's commands, which are one chunk.
  if (chunks.empty() || kv.layers().size() != geometry_.kvLayout.attentionLayers || (aneFfn && chunks.size() > 1))
    throw std::invalid_argument("invalid Qwen prefill");
}

QwenTarget::PrefillStep QwenTarget::prefillStep(metal::CommandGraph &graph, const QwenTargetPrefillChunk &chunk,
                                                const kv::PageStorage &kv, ops::AneFfn *aneFfn) const {
  if (chunk.sequences.empty() || chunk.sequences.size() > ExecutionLimits::maximumBatchWidth || !chunk.rows ||
      chunk.rows > ExecutionLimits::prefillTokenBudget)
    throw std::invalid_argument("invalid Qwen prefill chunk");
  PrefillStep step{graph, chunk.buffers, chunk.sequences, chunk.rows, kv.layers()};
  for (const QwenTargetPrefillSequence &sequence : chunk.sequences) {
    requirePrefillStates(geometry_, sequence);
    step.attention.push_back(
        operators_.prefillAttention(sequence.rows, geometry_.attentionQueryHeads, geometry_.kvLayout));
  }
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moePrefill(geometry_.moeShape(), chunk.rows);
  if (aneFfn && aneFfn->splits(chunk.rows)) step.aneFfn = aneFfn;
  return step;
}

QwenTarget::VerifyStep QwenTarget::verifyStep(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &buffers,
                                              const kv::PageStorage &kv,
                                              std::span<const kv::ChunkedPrefillParams> chunks,
                                              uint32_t lanes) const {
  const std::span<const SplashKvLayer> kvLayers = kv.layers();
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || chunks.size() != lanes ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (geometry_.stateLayout.auxiliaryBytes && !buffers.auxiliary[lane])
      throw std::invalid_argument("Qwen verify lane without its auxiliary state");
    histories[lane] = chunks[lane].committed_tokens;
  }
  VerifyStep step{graph, buffers, kvLayers, chunks, lanes, lanes * kVerifyRows,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.kvLayout,
                                             std::span(histories).first(lanes))};
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moeDecode(geometry_.moeShape(), lanes);
  return step;
}

namespace {

// A mixer's output projection of a prefill step's `hidden` rows.
void addPrefillOutput(const ops::Linear &linear, metal::CommandGraph &graph, metal::MetalBuffer hidden,
                      const ops::Projection &projection, uint32_t rows, ops::LinearScratch scratch,
                      metal::MetalBuffer output, metal::MetalBuffer residual) {
  if (residual)
    linear.addPrefillResidual(graph, std::move(hidden), projection, std::move(residual), std::move(output), rows,
                              scratch);
  else
    linear.addPrefill(graph, std::move(hidden), projection, std::move(output), rows, scratch);
}

} // namespace

void QwenTarget::addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer, BlockOutput output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.gdnLayer++;
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked, step.rows,
                                 b.linearScratch);
  for (const QwenTargetPrefillSequence &sequence : step.sequences) {
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(buffer, sequence.rowBegin, sequence.rows, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<float>(buffer, sequence.rowBegin, sequence.rows, width);
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
  addPrefillOutput(operators_.linear(), step.graph, b.gdnHidden, mixer.outputProjection, step.rows, b.linearScratch,
                   std::move(output.rows), std::move(output.residual));
}

// A sequence with blocks attends through the lists of its tiles of rows.
void QwenTarget::addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer, BlockOutput output,
                                 std::span<const ops::AttentionBlocks> blocks) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.attentionLayer++;
  if (!blocks.empty() && blocks.size() != step.sequences.size())
    throw std::invalid_argument("Qwen prefill attention blocks do not match its sequences");
  const auto u16Rows = [&](const metal::MetalBuffer &buffer, const QwenTargetPrefillSequence &sequence,
                           uint32_t width) {
    return rowsOf<uint16_t>(buffer, sequence.rowBegin, sequence.rows, width);
  };
  const auto f32Rows = [&](const metal::MetalBuffer &buffer, const QwenTargetPrefillSequence &sequence,
                           uint32_t width) {
    return rowsOf<float>(buffer, sequence.rowBegin, sequence.rows, width);
  };
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, step.rows,
                                 b.linearScratch);
  for (size_t index = 0; index < step.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
    const uint64_t headBytes = uint64_t{sequence.attentionStride} * geometry_.attentionHeadDimension * sizeof(uint16_t);
    const uint64_t queryBytes = geometry_.attentionQueryHeads * headBytes;
    const uint64_t kvBytes = geometry_.attentionKvHeads * headBytes;
    const metal::MetalBuffer queries = backend_.view(b.fullQueries, sequence.queryOffset, queryBytes);
    const metal::MetalBuffer attentionRows = backend_.view(b.fullAttention, sequence.queryOffset, queryBytes);
    const metal::MetalBuffer keys = backend_.view(b.chunkKeys, sequence.kvOffset, kvBytes);
    const metal::MetalBuffer values = backend_.view(b.chunkValues, sequence.kvOffset, kvBytes);
    ops::PagedAttention::addPrefillProjection(
        step.graph, u16Rows(b.fullPacked, sequence, geometry_.packedFullWidth), mixer.queryNorm, mixer.keyNorm,
        f32Rows(b.ropeCos, sequence, geometry_.rotaryPairs), f32Rows(b.ropeSin, sequence, geometry_.rotaryPairs),
        queries, keys, values, sequence.rows, sequence.attentionStride, geometry_.attentionQueryHeads,
        geometry_.kvLayout);
    ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[layer], keys, values, sequence.pageTable,
                                         sequence.chunk, geometry_.kvLayout);
    ops::PagedAttention::addPrefill(step.graph, step.kvLayers[layer], queries, attentionRows, b.attentionPartials,
                                    b.attentionStatistics, sequence.pageTable, sequence.chunk, step.attention[index],
                                    blocks.empty() ? ops::AttentionBlocks{} : blocks[index]);
    ops::PagedAttention::addPrefillGate(
        step.graph, u16Rows(b.fullPacked, sequence, geometry_.packedFullWidth), attentionRows,
        u16Rows(b.attentionHidden, sequence, geometry_.attentionWidth), sequence.rows, sequence.attentionStride,
        geometry_.attentionQueryHeads, geometry_.kvLayout);
  }
  addPrefillOutput(operators_.linear(), step.graph, b.attentionHidden, mixer.outputProjection, step.rows,
                   b.linearScratch, std::move(output.rows), std::move(output.residual));
}

ops::LinearInput QwenTarget::decodeInput(const ops::Projection &projection, uint32_t lanes) const {
  return operators_.linear().decodePlan(projection, lanes).input();
}

void QwenTarget::addDecodeProjection(metal::CommandGraph &graph, metal::MetalBuffer input,
                                     const ops::Projection &projection, metal::MetalBuffer output, uint32_t lanes,
                                     ops::LinearScratch scratch, ops::PreparedInput prepared) const {
  const ops::Linear &linear = operators_.linear();
  linear.add(graph,
             {.input = std::move(input), .output = std::move(output), .scratch = scratch,
              .prepared = std::move(prepared)},
             projection, linear.decodePlan(projection, lanes));
}

// Each producer emits the table (if any) its consumer's plan reads.
void QwenTarget::addVerifyMixer(VerifyStep &step, const QwenGdnWeights &mixer, ops::PreparedInput normalized,
                                BlockOutput output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.gdnLayer++;
  addDecodeProjection(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked, step.lanes, b.linearScratch,
                      std::move(normalized));
  const ops::LinearPlan outputPlan = linear.decodePlan(
      mixer.outputProjection, step.lanes, output.residual ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None);
  const ops::PreparedInput hidden = ops::GDN::addDecode(
      step.graph,
      {b.gdnPacked, mixer.convolutionWeights, b.currentGdnStates, b.nextGdnStates, b.gdnTape,
       std::span(b.gdnTapeLanes).first(step.lanes), mixer.decay, mixer.timeBias, mixer.mixerNorm, b.gdnHidden,
       b.linearScratch},
      geometry_.gdnShape(), step.lanes, layer, geometry_.stateLayout.strides(),
      mixer.outputHeadOrder, outputPlan.input());
  linear.add(step.graph,
             {.input = b.gdnHidden, .output = std::move(output.rows), .residual = std::move(output.residual),
              .scratch = b.linearScratch, .prepared = hidden},
             mixer.outputProjection, outputPlan);
}

// With `blocks`, each lane's rows attend through its list (ops::AttentionBlocks).
void QwenTarget::addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer, ops::PreparedInput normalized,
                                BlockOutput output, const ops::AttentionBlocks &blocks) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer++;
  addDecodeProjection(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, step.lanes, b.linearScratch,
                      std::move(normalized));
  ops::PagedAttention::addVerifyProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                           b.ropeSin, b.fullQueries, b.chunkKeys[layer], b.chunkValues[layer],
                                           geometry_.attentionQueryHeads, geometry_.kvLayout, step.lanes);
  ops::PagedAttention::addVerify(step.graph, step.kvLayers[layer],
                                 {b.chunkKeys[layer], b.chunkValues[layer], b.fullQueries, b.attentionPartials,
                                  b.attentionStatistics, b.fullAttention, b.pageTables},
                                 step.chunks, step.attention, blocks);
  const ops::LinearPlan outputPlan = linear.decodePlan(
      mixer.outputProjection, step.lanes, output.residual ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None);
  const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
      step.graph, b.fullPacked, b.fullAttention, b.attentionHidden, geometry_.attentionQueryHeads,
      geometry_.kvLayout, step.lanes, b.linearScratch, outputPlan.input());
  linear.add(step.graph,
             {.input = b.attentionHidden, .output = std::move(output.rows), .residual = std::move(output.residual),
              .scratch = b.linearScratch, .prepared = hidden},
             mixer.outputProjection, outputPlan);
}

namespace {

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

// The rows a mixer's output projection adds to the layer's input rows into.
template <class Buffers> metal::MetalBuffer mixerOutput(const Buffers &b, const QwenGdnWeights &) {
  return b.gdnOutput;
}
template <class Buffers> metal::MetalBuffer mixerOutput(const Buffers &b, const QwenAttentionWeights &) {
  return b.attentionOutput;
}

// The Qwen3.5 architecture: Qwen3.8-27B (model/Qwen3_8.hpp, a dense FFN) and
// Qwen3.6-35B-A3B (model/Qwen3_6Moe.hpp, a sparse MoE). Layer l reads its
// input rows from buffers.hidden[l & 1]; its mixer and its FFN each read the
// RMS norm of the rows before them and add their output to them, into the
// other buffer. The layers the draft reads copy their output rows into the
// captured rows, and the head reads the final norm of the last layer's.
template <class Layout, class Layer> class Qwen3_5Target final : public QwenTarget {
public:
  Qwen3_5Target(const QwenTargetWeights<Layout, Layer> &weights, const QwenTargetGeometry &geometry,
                metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
      : QwenTarget(weights, geometry, backend, operators), weights_(weights) {
    requireLayers(std::span<const Layer>(weights.layers));
  }

  std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph, std::span<const QwenTargetPrefillChunk> chunks,
                                             const kv::PageStorage &kv, ops::AneFfn *aneFfn) override {
    std::vector<PrefillStep> steps = prefillSteps(graph, chunks, kv, aneFfn);
    return addPrefillLayers(std::span(steps), *this);
  }

  void addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers, const kv::PageStorage &kv,
                 std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes) const override {
    VerifyStep step = verifyStep(graph, buffers, kv, chunks, lanes);
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      const Layer &layer = weights_.layers[index];
      const metal::MetalBuffer input = buffers.hidden[index & 1];
      const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
      const metal::MetalBuffer residual = std::visit(
          [&](const auto &mixer) {
            const ops::PreparedInput normalized = ops::Normalization::addRms(
                step.graph, input, layer.inputNorm, buffers.normalized, geometry_.hiddenSize, step.rows,
                buffers.linearScratch, decodeInput(mixer.inputProjection, step.lanes));
            const BlockOutput block{mixerOutput(buffers, mixer), input};
            addVerifyMixer(step, mixer, normalized, block);
            return block.rows;
          },
          layer.mixer);
      addVerifyFfn(step, layer, residual, output);
      if (const auto slot = geometry_.captureSlot(index))
        addCapture(step.graph, geometry_, *slot, output, 0, buffers.capturedTargetHidden, 0, step.rows);
    }
    requireLayerPartition(step.gdnLayer, step.attentionLayer);
    addHeadBatch(step.graph, buffers.hidden[geometry_.layers & 1], buffers.finalHidden, buffers.logits, step.lanes,
                 buffers.linearScratch);
  }

  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t lanes, ops::LinearScratch scratch) const override {
    const ops::PreparedInput normalized = ops::Normalization::addRms(
        graph, std::move(hidden), weights_.finalNorm, finalHidden, geometry_.hiddenSize, lanes * kVerifyRows,
        scratch, decodeInput(vocabularyProjection(), lanes));
    addDecodeProjection(graph, std::move(finalHidden), vocabularyProjection(), std::move(logits), lanes, scratch,
                        normalized);
  }

private:
  // The shared prefill's layer loop runs the family's (QwenTarget::addPrefillLayers).
  friend class QwenTarget;

  void addPrefillBegin(PrefillStep &) const {}
  void addPrefillLayer(PrefillStep &step, uint32_t index) const {
    const Layer &layer = weights_.layers[index];
    const QwenTargetPrefillBuffers &b = step.buffers;
    const metal::MetalBuffer input = b.hidden[index & 1];
    ops::Normalization::addRms(step.graph, input, layer.inputNorm, b.normalized, geometry_.hiddenSize, step.rows);
    const metal::MetalBuffer residual = std::visit(
        [&](const auto &mixer) {
          const BlockOutput block{mixerOutput(b, mixer), input};
          addPrefillMixer(step, mixer, block);
          return block.rows;
        },
        layer.mixer);
    addPrefillFfn(step, index, layer, residual, b.hidden[(index & 1) ^ 1]);
  }
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t index) const { return qwenMoeBlock(weights_.layers[index]); }
  void addPrefillExperts(PrefillStep &step, uint32_t index) const {
    const QwenTargetPrefillBuffers &b = step.buffers;
    if (const ops::BlockMoeWeights *ffn = moe(index)) ops::MoE::addRouted(step.graph, step.ffn, *ffn, *step.moe);
    forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t rows) {
      addCapture(step.graph, geometry_, slot, b.hidden[(index & 1) ^ 1], source, b.captured, destination, rows);
    });
  }
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const {
    return step.buffers.hidden[geometry_.layers & 1];
  }

  // The FFN of layer `index`, of the RMS norm of the mixer's `residual` rows,
  // adding to them into `output`: a dense FFN, or a MoE's routing, whose
  // routed work addPrefillExperts adds.
  void addPrefillFfn(PrefillStep &step, uint32_t index, const Qwen3_8LayerWeights &layer,
                     metal::MetalBuffer residual, metal::MetalBuffer output) const {
    ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, step.buffers.normalized,
                               geometry_.hiddenSize, step.rows);
    const ops::PrefillFfnBuffers ffn = step.buffers.ffn();
    if (step.aneFfn)
      step.aneFfn->add(step.graph, index, ffn, residual, output, step.rows);
    else
      operators_.linear().addPrefillSwiGlu(step.graph, {&layer.gateProjection, &layer.upProjection,
                                                        &layer.downProjection},
                                           ffn, residual, output, step.rows);
  }
  void addPrefillFfn(PrefillStep &step, uint32_t, const Qwen3_6MoeLayerWeights &layer,
                     metal::MetalBuffer residual, metal::MetalBuffer output) const {
    const QwenTargetPrefillBuffers &b = step.buffers;
    ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                               step.rows);
    step.ffn = {b.normalized, residual, output, b.moe};
    ops::MoE::addRouting(step.graph, step.ffn, layer.ffn, *step.moe);
  }

  void addVerifyFfn(VerifyStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
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
  void addVerifyFfn(VerifyStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                    metal::MetalBuffer output) const {
    const QwenTargetVerifyBuffers &b = step.buffers;
    ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                               step.rows);
    ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, layer.ffn, *step.moe);
  }

  const QwenTargetWeights<Layout, Layer> &weights_;
};

} // namespace

template <class Layout, class Layer>
std::unique_ptr<QwenTarget> makeQwenTarget(const QwenTargetWeights<Layout, Layer> &weights,
                                           const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                                           const ops::ExecutionPlans &operators, uint32_t) {
  return std::make_unique<Qwen3_5Target<Layout, Layer>>(weights, geometry, backend, operators);
}

template std::unique_ptr<QwenTarget> makeQwenTarget(const Qwen3_8Weights &, const QwenTargetGeometry &,
                                                    metal::MetalBackend &, const ops::ExecutionPlans &, uint32_t);
template std::unique_ptr<QwenTarget> makeQwenTarget(const Qwen3_6MoeWeights &, const QwenTargetGeometry &,
                                                    metal::MetalBackend &, const ops::ExecutionPlans &, uint32_t);

} // namespace splash::model
