#include "model/QwenTarget.hpp"

#include "model/Flint.hpp"
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
      }) ||
      (!sequence.endCodes.empty() && !everyLayer(sequence.endCodes)))
    throw std::invalid_argument("Qwen prefill state layer mismatch");
}

} // namespace

void QwenTargetGeometry::finish(std::span<const ops::BlockMoeWeights *const> moeBlocks,
                                const ops::Projection &logits) {
  moeRouterFormat = ops::moeRouterFormat(moeBlocks);
  moeInt8Experts = ops::moeInt8Experts(moeBlocks);
  prefillProjections = decodeProjections;
  include(logits);
  // The draft's head over its subset of the vocabulary: a view of the head's
  // rows (DFlashDraft), the head's shape over fewer of them.
  ops::ProjectionShape subset = logits.shape();
  subset.outputSize = draftVocabulary.columns(vocabularySize);
  decodeProjections.push_back(subset);
  for (auto *shapes : {&prefillProjections, &decodeProjections, &gateUpProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
}

void includeDenseFfn(QwenTargetGeometry &geometry, ops::SwiGluProjections ffn) {
  const ops::ProjectionShape gate = ffn.gate->shape(), up = ffn.up->shape();
  if (gate.outputSize != up.outputSize || gate.inputSize != up.inputSize || gate.rotated != up.rotated)
    throw WeightStoreError("fused gate/up projections must have matching shapes");
  geometry.include(*ffn.gate);
  geometry.include(*ffn.up);
  geometry.include(*ffn.down);
  geometry.gateUpProjections.push_back(up);
}

QwenTarget::QwenTarget(const QwenTargetWeightsBase &weights, const QwenTargetGeometry &geometry,
                       metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
    : weightsBase_(weights), geometry_(geometry), backend_(backend), operators_(operators) {
  if (!geometry_.valid()) throw std::invalid_argument("Qwen target weights do not match execution geometry");
  if (!ops::PagedAttention::compiles(geometry_.attentionQueryHeads, geometry_.kvLayout, geometry_.attentionGate,
                                     geometry_.rotaryPairs))
    throw std::invalid_argument("no attention kernel for the target's output gate and RoPE");
}

QwenTarget::~QwenTarget() = default;

QwenTargetScratchBytes QwenTarget::allocatedScratchBytes() const noexcept { return {}; }

const ops::Projection &QwenTarget::vocabularyProjection() const noexcept {
  return weightsBase_.logitsProjection;
}

void QwenTarget::addInsideCapture(metal::CommandGraph &, metal::MetalBuffer, metal::MetalBuffer) const {}

void QwenTarget::prepareVerifyLane(uint32_t, uint64_t, const QwenTargetRequest &) {}

std::vector<uint32_t> QwenTarget::verifyTableRows(const ops::LaneRows &) const { return {}; }

void QwenTarget::addVerifyInput(metal::CommandGraph &, const QwenVerifyInput &) const {}

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, std::move(hidden),
                      rows);
}

void QwenTarget::addStateCommit(metal::CommandGraph &, QwenTargetCommitBuffers buffers,
                                const ops::StepRows &step) const {
  if (!step.lanes || step.lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  if (geometry_.stateLayout.auxiliaryBytes)
    for (uint32_t lane = 0; lane < step.lanes; ++lane)
      if (!buffers.auxiliaryIn[lane] || !buffers.auxiliaryOut[lane])
        throw std::invalid_argument("Qwen state commit lane without its auxiliary state");
}

void QwenTarget::addStateFold(metal::CommandGraph &graph, QwenTargetFoldBuffers buffers, uint32_t rows) const {
  if (geometry_.stateLayout.auxiliaryBytes)
    throw std::logic_error("a target with auxiliary state folds no state in decode");
  if (!geometry_.stateLayout.layers)
    return;
  ops::GDN::addFold(graph,
                    {std::move(buffers.state), std::move(buffers.gdnTape), buffers.stepSlot,
                     std::move(buffers.retainedRows), std::move(buffers.fold)},
                    geometry_.gdnShape(), geometry_.stateLayout.layers, rows, geometry_.stateLayout.strides(),
                    geometry_.stateLayout.cachedRecurrentLayerBytes(), geometry_.stateLayout.cachedCodes);
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
                                QwenPrefillLayers layers, const ops::AneFfn *aneFfn) const {
  // The Neural Engine splits a dense target's commands, which are one chunk
  // and run whole.
  const bool whole = !layers.first && layers.end == geometry_.layers;
  if (chunks.empty() || kv.layers().size() != geometry_.kvLayout.attentionLayers ||
      (aneFfn && (chunks.size() > 1 || !whole)) || layers.first >= layers.end || layers.end > geometry_.layers)
    throw std::invalid_argument("invalid Qwen prefill");
}

QwenTarget::PrefillStep QwenTarget::prefillStep(metal::CommandGraph &graph, const QwenTargetPrefillChunk &chunk,
                                                const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                ops::AneFfn *aneFfn) const {
  if (chunk.sequences.empty() || chunk.sequences.size() > ExecutionLimits::maximumBatchWidth || !chunk.rows ||
      chunk.rows > ExecutionLimits::prefillTokenBudget)
    throw std::invalid_argument("invalid Qwen prefill chunk");
  PrefillStep step{graph, chunk.buffers, chunk.sequences, chunk.rows, kv.layers(), spilled};
  step.decode = chunk.decode;
  if (step.decode && (step.decode->step.lanes != chunk.sequences.size() || step.decode->step.rows != chunk.rows))
    throw std::invalid_argument("a Qwen decode step's lanes or rows do not match its chunk");
  for (size_t index = 0; index < chunk.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = chunk.sequences[index];
    // A decode lane's states are the step's (QwenTargetDecodeBuffers).
    if (!step.decode) requirePrefillStates(geometry_, sequence);
    else if (geometry_.stateLayout.auxiliaryBytes && !sequence.auxiliaryIn)
      throw std::invalid_argument("Qwen decode lane without its auxiliary state");
    step.chunks[index] = sequence.chunk;
    step.pageTables[index] = sequence.pageTable;
  }
  const ops::AttentionWorkspace workspace{chunk.buffers.attentionPartials.sizeBytes(),
                                          chunk.buffers.attentionStatistics.sizeBytes()};
  step.attention.emplace(operators_.attention(step.attentionSequences(), geometry_.attentionQueryHeads,
                                              geometry_.kvLayout, workspace));
  if (const uint32_t window = geometry_.kvLayout.windowTokens)
    step.windowedAttention.emplace(operators_.attention(step.attentionSequences(), geometry_.attentionQueryHeads,
                                                        geometry_.kvLayout, workspace, {.windowTokens = window}));
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe)
    step.moe = operators_.moe(geometry_.moeShape(), geometry_.moeRouterFormat, chunk.rows,
                              chunk.decode ? ops::StepKind::Decode : ops::StepKind::Prefill);
  if (aneFfn && aneFfn->splits(chunk.rows)) step.aneFfn = aneFfn;
  return step;
}

void QwenTarget::planOutputRows(PrefillStep &step, uint32_t index) const {
  step.readRows.clear();
  // A decode step reads every row's logits.
  if (step.decode || index + 1 != geometry_.layers || !geometry_.isFullAttentionLayer(index) ||
      geometry_.captureSlot(index) || step.sequences.size() == step.rows)
    return;
  for (const QwenTargetPrefillSequence &sequence : step.sequences)
    step.readRows.push_back(sequence.rowBegin + sequence.rows - 1);
  if (step.moe)
    step.moe = operators_.moe(geometry_.moeShape(), geometry_.moeRouterFormat, step.outputRows(), ops::StepKind::Prefill);
}

void QwenTarget::addReadRows(const PrefillStep &step, const metal::MetalBuffer &compact,
                             const metal::MetalBuffer &output) const {
  const uint32_t width = geometry_.hiddenSize;
  for (uint32_t row = 0; row < step.readRows.size(); ++row)
    ops::RowCopy::add(step.graph, compact, {row, width, 0}, output, {step.readRows[row], width, 0}, 1, width);
}

// The mixer reads the normalized rows as its input projection's plan takes
// them (`normalized`, which the norm wrote), and its output projection reads
// the hidden rows as the decode GDN or the attention gate wrote them for its
// plan. A decode step's GDN scans each lane's rows along its tree from the
// lane's state into its tape (ops::GDN::addDecode); a prompt sequence's scans
// its rows in order from its states into theirs, capturing those inside.
void QwenTarget::addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer, ops::PreparedInput normalized,
                                 BlockOutput output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = this->linear(step);
  const uint32_t layer = step.gdnLayer++;
  linear.add(step.graph,
             {.input = b.normalized, .output = b.gdnPacked, .scratch = b.linearScratch,
              .prepared = std::move(normalized)},
             mixer.inputProjection, linear.plan(mixer.inputProjection, step.rows));
  const ops::LinearPlan outputPlan = linear.plan(
      mixer.outputProjection, step.rows, output.residual ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None);
  ops::PreparedInput hidden;
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    hidden = ops::GDN::addDecode(
        step.graph,
        {b.gdnPacked, mixer.convolutionWeights, d->currentGdnStates, d->nextGdnStates, d->gdnTape,
         std::span(d->gdnTapeLanes).first(d->step.lanes), b.laneRows, step.rows, d->step.laneCap, mixer.decay,
         mixer.timeBias, mixer.mixerNorm, b.gdnHidden, d->tree, b.linearScratch},
        geometry_.gdnShape(), layer, geometry_.stateLayout.strides(), mixer.outputHeadOrder, outputPlan.input());
  } else {
    for (const QwenTargetPrefillSequence &sequence : step.sequences) {
      const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
        return rowsOf<uint16_t>(buffer, sequence.rowBegin, sequence.rows, width);
      };
      const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
        return rowsOf<float>(buffer, sequence.rowBegin, sequence.rows, width);
      };
      std::vector<ops::GdnPrefillCapture> captures;
      captures.reserve(sequence.stateCaptures.size() + 1);
      for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
        captures.push_back({capture.row, capture.convolution[layer], capture.recurrent[layer]});
      if (!sequence.endCodes.empty())
        captures.push_back({sequence.rows, {}, sequence.endCodes[layer]});
      ops::GDN::addPrefill(
          step.graph, backend_,
          {u16(b.gdnPacked, geometry_.packedGdnWidth), mixer.convolutionWeights, sequence.convolutionIn[layer],
           sequence.convolutionOut[layer], u16(b.gdnQueries, geometry_.gdnKeyWidth()),
           u16(b.gdnKeys, geometry_.gdnKeyWidth()), u16(b.gdnValues, geometry_.attentionWidth), mixer.decay,
           mixer.timeBias, f32(b.gdnDecay, geometry_.gdnValueHeads), u16(b.gdnBeta, geometry_.gdnValueHeads),
           sequence.recurrentIn[layer], sequence.recurrentOut[layer], u16(b.recurrent, geometry_.attentionWidth),
           mixer.mixerNorm, u16(b.gdnHidden, geometry_.attentionWidth)},
          geometry_.gdnShape(), sequence.rows, geometry_.stateLayout.format, mixer.outputHeadOrder, captures,
          geometry_.stateLayout.cachedCodes);
    }
  }
  linear.add(step.graph,
             {.input = b.gdnHidden, .output = std::move(output.rows), .residual = std::move(output.residual),
              .scratch = b.linearScratch, .prepared = std::move(hidden)},
             mixer.outputProjection, outputPlan);
}

// A sequence with blocks attends through the lists of its tiles of rows. A
// prompt sequence whose history holds Index pages attends staging, which the
// step's reads of spilled K/V fill with the layer's history and the chunk's
// pages once stored; a decode step whose lanes hold Index pages attends
// through its lists, which those reads locate (SpilledKvReads::verify): its
// rows that read their page tables read only Full pages (SparseKvShape).
void QwenTarget::addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer, ops::PreparedInput normalized,
                                 BlockOutput output, std::span<const ops::AttentionBlocks> blocks) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = this->linear(step);
  const uint32_t layer = step.attentionLayer++;
  const ops::AttentionVariant variant = geometry_.attentionVariant(layer);
  if (!blocks.empty() && blocks.size() != step.sequences.size())
    throw std::invalid_argument("Qwen attention blocks do not match its sequences");
  // A layer that outputs its read rows alone adds its output to their
  // residual rows, which it gathers.
  if (!step.readRows.empty() && !output.residual)
    throw std::logic_error("a Qwen prefill layer that outputs its read rows alone has no residual rows");
  const std::span<const kv::ChunkedPrefillParams> sequences = step.attentionSequences();
  const ops::AttentionRows rows{b.laneRows, step.rows, step.decode ? step.decode->tree : metal::MetalBuffer{}};
  const std::span<const metal::MetalBuffer> pageTables = std::span(step.pageTables).first(sequences.size());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.fullPacked, .scratch = b.linearScratch,
              .prepared = std::move(normalized)},
             mixer.inputProjection, linear.plan(mixer.inputProjection, step.rows));
  // The q/k projection stores each row's key and value in its page slot.
  ops::PagedAttention::addProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos, b.ropeSin,
                                     b.fullQueries, step.kvLayers[layer], pageTables, sequences, rows,
                                     geometry_.attentionQueryHeads, geometry_.kvLayout, variant);
  std::array<ops::AttentionHistory, ExecutionLimits::maximumBatchWidth> histories{};
  if (step.decode) {
    std::array<ops::AttentionBlocks, ExecutionLimits::maximumBatchWidth> attended{};
    std::copy(blocks.begin(), blocks.end(), attended.begin());
    // The lists' located blocks resolve from spilled history pages, which
    // the projection's stores into the step's own slots do not touch.
    if (step.spilled.active()) {
      if (blocks.empty() || !blocks.front().stride)
        throw std::logic_error("a decode step over spilled K/V attends without block lists");
      // The lanes' lists lie in one buffer, the lists of each lane's tiles at
      // the lane cap after the lanes before it (Qwen4ExpTarget::addDecodeQsa),
      // which the reads locate at once.
      ops::AttentionBlocks command = blocks.front();
      command.list = 0;
      const ops::AttentionBlocks located =
          step.spilled.verify(step.graph, layer, command, step.attention->sequences().front().tiles());
      for (size_t index = 0; index < blocks.size(); ++index) {
        attended[index] = located;
        attended[index].list = blocks[index].list;
      }
    }
    for (size_t index = 0; index < sequences.size(); ++index)
      histories[index] = {step.kvLayers[layer], pageTables[index], attended[index]};
  } else {
    for (size_t index = 0; index < step.sequences.size(); ++index) {
      const QwenTargetPrefillSequence &sequence = step.sequences[index];
      const uint32_t firstPage = sequence.chunk.committed_tokens / kv::kPageTokens;
      const PrefillKv history =
          step.spilled.prefill(step.graph, layer, sequence.lane, firstPage,
                               kv::chunkedPrefillRequiredPages(sequence.chunk) - firstPage,
                               {sequence.pageTable, step.kvLayers[layer]});
      histories[index] = {history.kv, history.table, blocks.empty() ? ops::AttentionBlocks{} : blocks[index]};
    }
  }
  // Only each sequence's last row is read (planOutputRows): the tile that
  // holds it attends as in the chunk (lastTiles), and its gated row is
  // written row `index`.
  const bool lastRows = !step.readRows.empty();
  const ops::AttentionPlan &plan = variant.windowTokens ? *step.windowedAttention : *step.attention;
  const ops::LinearPlan outputPlan =
      linear.plan(mixer.outputProjection, step.outputRows(),
                  output.residual ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None, nullptr, step.rows);
  const ops::PreparedInput hidden = ops::PagedAttention::addAttention(
      step.graph, lastRows ? ops::PagedAttention::lastTiles(plan) : plan, b.fullQueries, b.attentionPartials,
      b.attentionStatistics, sequences, std::span(histories).first(sequences.size()), rows,
      {b.fullPacked, b.attentionHidden, lastRows ? ops::AttentionOutput::Last : ops::AttentionOutput::Every,
       b.linearScratch, lastRows ? ops::LinearInput::Plain : outputPlan.input()},
      mixer.sinks);
  // The read rows' residual rows, gathered into the normalized rows the
  // input projection has read.
  metal::MetalBuffer residual = std::move(output.residual);
  if (lastRows) {
    for (uint32_t row = 0; row < step.readRows.size(); ++row)
      ops::RowCopy::add(step.graph, residual, {step.readRows[row], geometry_.hiddenSize, 0}, b.normalized,
                        {row, geometry_.hiddenSize, 0}, 1, geometry_.hiddenSize);
    residual = b.normalized;
  }
  linear.add(step.graph,
             {.input = b.attentionHidden, .output = std::move(output.rows), .residual = std::move(residual),
              .scratch = b.linearScratch, .prepared = hidden},
             mixer.outputProjection, outputPlan);
}

// The norm writes the rows as the FFN's first plan takes them (swiGluInput):
// a decode step's and a short chunk's gate/up plan, a longer chunk's gate.
void QwenTarget::addPrefillDenseFfn(PrefillStep &step, uint32_t index, const ops::NormWeights &norm,
                                    ops::SwiGluProjections ffn, metal::MetalBuffer residual,
                                    metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = this->linear(step);
  const uint32_t rows = step.outputRows();
  const ops::PrefillFfnBuffers buffers = b.ffn();
  // The split takes whole chunks; read rows alone run on the GPU.
  if (step.aneFfn && step.readRows.empty()) {
    ops::Normalization::addRms(step.graph, residual, norm, b.normalized, geometry_.hiddenSize, rows);
    step.aneFfn->add(step.graph, index, buffers, std::move(residual), std::move(output), rows);
    return;
  }
  const ops::PreparedInput normalized =
      ops::Normalization::addRms(step.graph, residual, norm, b.normalized, geometry_.hiddenSize, rows,
                                 b.linearScratch, linear.swiGluInput(ffn, rows, step.rows));
  linear.addSwiGlu(step.graph, ffn, buffers, std::move(residual), std::move(output), rows, step.rows, normalized);
}

// A decode step's lanes route their rows alone (ops::MoeBuffers::lanes).
void QwenTarget::addPrefillMoeRouting(PrefillStep &step, const ops::NormWeights &norm,
                                      const ops::BlockMoeWeights &ffn, metal::MetalBuffer residual,
                                      metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  ops::Normalization::addRms(step.graph, residual, norm, b.normalized, geometry_.hiddenSize, step.outputRows());
  step.ffn = {.input = b.normalized, .residual = std::move(residual), .output = std::move(output), .scratch = b.moe};
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    step.ffn.lanes = d->step.lanes;
    step.ffn.laneRows = b.laneRows;
  }
  ops::MoE::addRouting(step.graph, step.ffn, ffn, *step.moe);
}

void QwenTarget::addNormalizedHead(metal::CommandGraph &graph, const ops::Linear &linear,
                                   const ops::NormWeights &finalNorm, metal::MetalBuffer hidden,
                                   metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t rows,
                                   ops::LinearScratch scratch) const {
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      graph, std::move(hidden), finalNorm, finalHidden, geometry_.hiddenSize, rows, scratch, headInput(linear, rows));
  addHead(graph, linear, std::move(finalHidden), std::move(logits), rows, scratch, normalized);
}

ops::LinearInput QwenTarget::headInput(const ops::Linear &linear, uint32_t rows) const {
  return rows <= ops::kMaximumDecodeTileRows ? linear.plan(vocabularyProjection(), rows).input()
                                             : ops::LinearInput::Plain;
}

void QwenTarget::addHead(metal::CommandGraph &graph, const ops::Linear &linear, metal::MetalBuffer input,
                         metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch,
                         ops::PreparedInput prepared) const {
  const ops::Projection &head = vocabularyProjection();
  if (rows <= ops::kMaximumDecodeTileRows) {
    linear.add(graph, {.input = std::move(input), .output = std::move(logits), .scratch = scratch, .prepared = prepared},
               head, linear.plan(head, rows));
    return;
  }
  // Each pass's views hold its plan's storage rows, which a step's row
  // tensors hold (decodeStorageRows): the first 64 rows and the tile of the
  // rest, at most 96 rows at five and six lanes and 128 at seven and eight.
  for (uint32_t first = 0; first < rows; first += ops::kMaximumDecodeTileRows) {
    const ops::LinearPlan plan = linear.plan(head, std::min(rows - first, ops::kMaximumDecodeTileRows));
    linear.add(graph,
               {.input = rowsOf<uint16_t>(input, first, plan.storageRows(), geometry_.hiddenSize),
                .output = rowsOf<float>(logits, first, plan.storageRows(), geometry_.vocabularySize),
                .scratch = scratch},
               head, plan);
  }
}

void QwenTarget::addCapture(metal::CommandGraph &graph, uint32_t slot, metal::MetalBuffer output,
                            uint32_t sourceRow, metal::MetalBuffer captured, uint32_t destinationRow,
                            uint32_t rows) const {
  const uint32_t width = geometry_.hiddenSize, capturedWidth = geometry_.capturedHiddenSize();
  if (slot >= capturedWidth / width)
    throw std::logic_error("Qwen target capture slot past the captured hidden rows");
  ops::RowCopy::add(graph, std::move(output), {sourceRow, width, 0}, std::move(captured),
                    {destinationRow, capturedWidth, slot * width}, rows, width);
}

namespace {

// The rows a mixer's output projection adds to the layer's input rows into.
template <class Buffers> metal::MetalBuffer mixerOutput(const Buffers &b, const QwenGdnWeights &) {
  return b.gdnOutput;
}
template <class Buffers> metal::MetalBuffer mixerOutput(const Buffers &b, const QwenAttentionWeights &) {
  return b.attentionOutput;
}

// The Qwen3.5 architecture: Qwen3.8-27B (model/Qwen3_8.hpp, a dense FFN),
// Qwen3.6-35B-A3B (model/Qwen3_6Moe.hpp, a sparse MoE) and Flint
// (model/Flint.hpp: every layer attends, with its sinks, and its FFN is dense
// on the leading layers and a MoE block on the others). Layer l reads its
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
                                             const kv::PageStorage &kv, SpilledKvReads &spilled,
                                             QwenPrefillLayers layers, ops::AneFfn *aneFfn) override {
    std::vector<PrefillStep> steps = prefillSteps(graph, chunks, kv, spilled, layers, aneFfn);
    return addPrefillLayers(std::span(steps), *this, layers);
  }

  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const override {
    addNormalizedHead(graph, operators_.linear(), weights_.finalNorm, std::move(hidden), std::move(finalHidden),
                      std::move(logits), rows, scratch);
  }

private:
  // The shared prefill's layer loop runs the family's (QwenTarget::addPrefillLayers).
  friend class QwenTarget;

  void addPrefillBegin(PrefillStep &) const {}
  [[nodiscard]] bool attends(uint32_t index) const {
    return std::holds_alternative<QwenAttentionWeights>(weights_.layers[index].mixer);
  }
  // A layer whose output work runs on read rows alone (planOutputRows) ends
  // them over its input's first rows, which its mixer has read, and
  // addPrefillExperts places them.
  void addPrefillLayer(PrefillStep &step, uint32_t index) const {
    const Layer &layer = weights_.layers[index];
    const QwenTargetPrefillBuffers &b = step.buffers;
    const metal::MetalBuffer input = b.hidden[index & 1];
    planOutputRows(step, index);
    const metal::MetalBuffer residual = std::visit(
        [&](const auto &mixer) {
          const ops::PreparedInput normalized = ops::Normalization::addRms(
              step.graph, input, layer.inputNorm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch,
              linear(step).plan(mixer.inputProjection, step.rows).input());
          const BlockOutput block{mixerOutput(b, mixer), input};
          addPrefillMixer(step, mixer, normalized, block);
          return block.rows;
        },
        layer.mixer);
    addPrefillFfn(step, index, layer, residual, step.readRows.empty() ? b.hidden[(index & 1) ^ 1] : input);
  }
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t index) const { return qwenMoeBlock(weights_.layers[index]); }
  void addPrefillExperts(PrefillStep &step, uint32_t index) const {
    const QwenTargetPrefillBuffers &b = step.buffers;
    if (const ops::BlockMoeWeights *ffn = moe(index)) ops::MoE::addRouted(step.graph, step.ffn, *ffn, *step.moe);
    addReadRows(step, b.hidden[index & 1], b.hidden[(index & 1) ^ 1]);
    // The draft reads a decode step's every row of the layers it captures.
    if (step.decode) {
      if (const auto slot = geometry_.captureSlot(index))
        addCapture(step.graph, *slot, b.hidden[(index & 1) ^ 1], 0, step.decode->capturedTargetHidden, 0, step.rows);
      return;
    }
    forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t rows) {
      addCapture(step.graph, slot, b.hidden[(index & 1) ^ 1], source, b.captured, destination, rows);
    });
  }
  // A decode step's head reads every row's final norm.
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const {
    const metal::MetalBuffer hidden = step.buffers.hidden[geometry_.layers & 1];
    if (const QwenTargetDecodeBuffers *d = step.decode)
      addNormalizedHead(step.graph, linear(step), weights_.finalNorm, hidden, d->finalHidden, d->logits, step.rows,
                        step.buffers.linearScratch);
    return hidden;
  }

  // The FFN of layer `index` (QwenTarget::addPrefillDenseFfn and its kin): a
  // dense FFN, or a MoE's routing, whose routed work addPrefillExperts adds.
  void addPrefillFfn(PrefillStep &step, uint32_t index, const Layer &layer, metal::MetalBuffer residual,
                     metal::MetalBuffer output) const {
    if (const std::optional<ops::SwiGluProjections> dense = denseFfn(layer))
      addPrefillDenseFfn(step, index, layer.postAttentionNorm, *dense, std::move(residual), std::move(output));
    else
      addPrefillMoeRouting(step, layer.postAttentionNorm, *qwenMoeBlock(layer), std::move(residual),
                           std::move(output));
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
template std::unique_ptr<QwenTarget> makeQwenTarget(const FlintWeights &, const QwenTargetGeometry &,
                                                    metal::MetalBackend &, const ops::ExecutionPlans &, uint32_t);

} // namespace splash::model
