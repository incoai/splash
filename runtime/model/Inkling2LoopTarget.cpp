#include "model/Inkling2LoopTarget.hpp"

#include "model/Inkling2Loop.hpp"
#include "ops/AneFfn.hpp"
#include "ops/BlockResidualMix.hpp"
#include "ops/LogitFinish.hpp"
#include "ops/MessageGate.hpp"
#include "ops/Normalization.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace splash::model {
namespace {

// A prompt chunk's rows as its layout holds them
// (QwenTargetPrefillBuffers::laneRows): its sequences' `rows` rows, each at
// most the longest's.
ops::LayoutRows promptRows(std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows) {
  uint32_t longest = 0;
  for (const QwenTargetPrefillSequence &sequence : sequences) longest = std::max(longest, sequence.rows);
  return {static_cast<uint32_t>(sequences.size()), rows, longest};
}

// A message gate's projections, each one float segment (Inkling2Loop.cpp).
ops::MessageGateWeights gateWeights(const Inkling2LoopGateWeights &gate) {
  const auto segment = [](const ops::Projection &projection) -> const ops::QuantizedSegment & {
    const auto &segments = projection.blocks().segments;
    if (segments.size() != 1 || !segments.front().isFloat())
      throw std::invalid_argument("Inkling-2-Mini's message gates are single float segments");
    return segments.front();
  };
  return {segment(gate.down), segment(gate.forget), segment(gate.gain)};
}

} // namespace

Inkling2LoopTarget::Inkling2LoopTarget(const Inkling2LoopWeights &weights, const QwenTargetGeometry &geometry,
                                       metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                       uint32_t prefillChunks)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights), schedule_(weights.layout.schedule()),
      steps_(geometry_, schedule_.shape, backend, prefillChunks) {
  static_assert(!Inkling2LoopLayout::sharedAttention, "the layout states the attention its target's layers run");
  // A step runs an application a layer, each with its KV layer, and the
  // weights hold a layer per weight layer; the relative attention stores
  // the KV layout's pages.
  requireLayers(schedule_.applications(), schedule_.applications());
  if (weights_.layers.size() != schedule_.shape.weightLayers || !ops::RelativeAttention::stores(geometry_.kvLayout))
    throw std::invalid_argument("Inkling-2-Mini weights do not match its loop and attention");
  // Its draft's captures and rows are those its producer trained it on: the
  // rows passed into its applications and its normed token rows.
  requireDraftRows(true, true);
}

QwenTargetScratchBytes Inkling2LoopTarget::allocatedScratchBytes() const noexcept { return steps_.allocatedBytes(); }

std::vector<metal::MetalBuffer> Inkling2LoopTarget::addPrefill(metal::CommandGraph &graph,
                                                               std::span<const QwenTargetPrefillChunk> chunks,
                                                               const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                               QwenPrefillLayers layers, ops::AneFfn *aneFfn) {
  std::vector<PrefillStep> steps = prefillSteps<PrefillStep>(graph, chunks, kv, spilled, layers, aneFfn);
  for (uint32_t index = 0; index < steps.size(); ++index) {
    PrefillStep &step = steps[index];
    step.scratch = step.decode ? steps_.verifyScratch() : steps_.prefillScratch(index);
    planRelativeAttention(step);
    if (const QwenTargetDecodeBuffers *d = step.decode) {
      step.layoutRows = {d->step.lanes, d->step.rows, d->step.laneCap};
      step.ffn.lanes = d->step.lanes;
      step.ffn.laneRows = step.buffers.laneRows;
    } else {
      step.layoutRows = promptRows(step.sequences, step.rows);
    }
  }
  return addPrefillLayers(std::span(steps), *this, layers);
}

void Inkling2LoopTarget::planRelativeAttention(PrefillStep &step) const {
  const ops::AttentionWorkspace workspace{step.buffers.attentionPartials.sizeBytes(),
                                          step.buffers.attentionStatistics.sizeBytes()};
  const ops::Linear &linear = this->linear(step);
  const auto plan = [&](ops::RelativeAttentionVariant variant) {
    return ops::RelativeAttention::plan(step.attentionSequences(), geometry_.kvLayout,
                                        operators_.calibration().attention, linear.int8Activations(), workspace,
                                        variant);
  };
  // The full layers scale their queries and relative states by their rows'
  // positions; the sliding ones read their window.
  step.fullAttention.emplace(plan({.windowTokens = 0, .positionScaled = true}));
  step.slidingAttention.emplace(plan({.windowTokens = geometry_.kvLayout.windowTokens, .positionScaled = false}));
}

metal::MetalBuffer Inkling2LoopTarget::rows(const PrefillStep &step, Inkling2LoopRows where) const {
  const Inkling2LoopScratch &s = step.scratch;
  switch (where.kind) {
  case Inkling2LoopRowsKind::Hidden:
    return step.buffers.hidden[where.index];
  case Inkling2LoopRowsKind::Stack:
    return rowsOf<uint16_t>(s.stack, where.index * s.planeRows, s.planeRows, geometry_.hiddenSize);
  case Inkling2LoopRowsKind::Previous:
    return rowsOf<uint16_t>(s.previous, where.index * s.planeRows, s.planeRows, geometry_.hiddenSize);
  }
  throw std::logic_error("unknown Inkling-2-Mini rows");
}

void Inkling2LoopTarget::addPrefillBegin(PrefillStep &step) const {
  ops::Normalization::addRms(step.graph, step.buffers.hidden[0], weights_.embeddingNorm,
                             rows(step, schedule_.input(0)), geometry_.hiddenSize, step.rows);
}

const ops::BlockMoeWeights *Inkling2LoopTarget::moe(uint32_t application) const {
  return &weights_.layers[schedule_.weightLayer(application)].ffn;
}

// The planes are contiguous, so the pass before's block states are one copy
// of the step's rows of each plane.
void Inkling2LoopTarget::addRestart(const PrefillStep &step) const {
  const uint32_t width = geometry_.hiddenSize;
  for (uint32_t plane = 0; plane + 1 < schedule_.previousPlanes(); ++plane)
    ops::RowCopy::add(step.graph, rows(step, {Inkling2LoopRowsKind::Stack, schedule_.firstPassPlane() + plane}),
                      {0, width, 0}, rows(step, {Inkling2LoopRowsKind::Previous, plane}), {0, width, 0}, step.rows,
                      width);
}

// The source's layer forward (modeling_inkling2_loop.py
// Inkling2Loop*DecoderLayer): the mixer's inputs, its attention and the
// convolution of its output; at a block boundary the residual is that output
// alone (the layer's input is its stack plane already), otherwise the input
// plus it, in fp32; then the FFN's input, the mix of the stack, its own push
// included, and the residual, gated by its message in a later pass, through
// ffn_norm, and its routing.
void Inkling2LoopTarget::addPrefillLayer(PrefillStep &step, uint32_t application) const {
  const Inkling2LoopLayerWeights &layer = weights_.layers[schedule_.weightLayer(application)];
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Inkling2LoopScratch &s = step.scratch;
  const uint32_t hidden = geometry_.hiddenSize;
  if (schedule_.restarts(application)) addRestart(step);
  const metal::MetalBuffer input = rows(step, schedule_.input(application));
  // Its drafts read the rows passed into an application, its block
  // residual's partial sum, not the mixes its norms read of them.
  addDraftCapture(step, application, input);
  addAttention(step, application, addMixerInputs(step, application, input));
  addConvolution(step, application, Inkling2LoopConvolution::Mixer,
                 {.input = b.attentionOutput,
                  .taps = layer.mixer.outputConvolution,
                  .residual = schedule_.boundary(application) ? metal::MetalBuffer{} : input,
                  .output = s.residual},
                 {.channels = hidden}, ops::ShortConvOutput::Sum);
  const ops::BlockResidualStates states{
      .stack = s.stack, .planeRows = s.planeRows, .current = s.residual, .currentFloat = true};
  const ops::BlockResidualShape shape{.hidden = hidden, .stack = schedule_.ffnStates(application), .queries = 1};
  if (schedule_.gated(application)) {
    ops::BlockResidualMix::addMix(step.graph, states, layer.ffnQuery, s.mixed, shape, step.rows);
    addGate(step, application, 1, 0, 3, layer.ffnNorm, b.normalized);
  } else {
    ops::BlockResidualMix::addMixNorm(step.graph, states, layer.ffnQuery, std::span(&layer.ffnNorm, 1),
                                      std::span(&b.normalized, 1), shape, step.rows);
  }
  // The MoE's weighted sum alone (added to zero rows), into the rows its
  // convolution reads after the layer's load (addPrefillExperts).
  step.ffn.input = b.normalized;
  step.ffn.residual = s.zeros;
  step.ffn.output = b.attentionOutput;
  step.ffn.scratch = b.moe;
  ops::MoE::addRouting(step.graph, step.ffn, layer.ffn, *step.moe);
}

// Layer 0 reads its input itself; a sliding layer one mix of it, which its
// one norm takes for its q, k and v inputs alike; a full layer three, each
// through its own norm. In a later pass each input is first gated by its
// message, the pass before's states mixed by the layer's message queries
// (Inkling2LoopTextModel.forward's message_input).
std::array<metal::MetalBuffer, 3> Inkling2LoopTarget::addMixerInputs(PrefillStep &step, uint32_t application,
                                                                    metal::MetalBuffer input) const {
  const uint32_t weightLayer = schedule_.weightLayer(application);
  const Inkling2LoopLayerWeights &layer = weights_.layers[weightLayer];
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Inkling2LoopScratch &s = step.scratch;
  const uint32_t hidden = geometry_.hiddenSize, queries = weights_.layout.mixerQueries(weightLayer);
  const bool full = schedule_.full(application);
  const std::array<ops::NormWeights, 3> norms = std::visit(
      [](const auto &inputs) -> std::array<ops::NormWeights, 3> {
        if constexpr (std::is_same_v<std::decay_t<decltype(inputs)>, Inkling2LoopFullInputs>)
          return {inputs.queryNorm, inputs.keyNorm, inputs.valueNorm};
        else
          return {inputs.norm, inputs.norm, inputs.norm};
      },
      layer.inputs);
  const std::array<metal::MetalBuffer, 3> separate{b.normalized, s.keyInput, s.valueInput};
  const std::array<metal::MetalBuffer, 3> shared{b.normalized, b.normalized, b.normalized};
  const ops::BlockResidualStates states{.stack = s.stack, .planeRows = s.planeRows, .current = input};
  const ops::BlockResidualShape shape{.hidden = hidden, .stack = schedule_.mixerStates(application),
                                      .queries = queries};
  if (schedule_.gated(application)) {
    // The pass before's block states, then its output, the mix's current rows.
    const uint32_t planes = schedule_.previousPlanes();
    ops::BlockResidualMix::addMix(step.graph,
                                  {.stack = s.previous,
                                   .planeRows = s.planeRows,
                                   .current = rows(step, {Inkling2LoopRowsKind::Previous, planes - 1})},
                                  layer.messages->queries, s.messages,
                                  {.hidden = hidden, .stack = planes - 1, .queries = 4}, step.rows);
    ops::BlockResidualMix::addMix(step.graph, states, layer.mixerQueries, s.mixed, shape, step.rows);
    for (uint32_t index = 0; index < 3; ++index)
      addGate(step, application, queries, full ? index : 0, index, norms[index], separate[index]);
    return separate;
  }
  if (!queries) {
    ops::Normalization::addRms(step.graph, input, norms[0], b.normalized, hidden, step.rows);
    return shared;
  }
  ops::BlockResidualMix::addMixNorm(step.graph, states, layer.mixerQueries, std::span(norms).first(queries),
                                    std::span(separate).first(queries), shape, step.rows);
  return full ? separate : shared;
}

void Inkling2LoopTarget::addGate(const PrefillStep &step, uint32_t application, uint32_t queries, uint32_t column,
                                 uint32_t message, const ops::NormWeights &norm, metal::MetalBuffer output) const {
  const Inkling2LoopGateWeights &gate = weights_.layers[schedule_.weightLayer(application)].messages->gates[message];
  const Inkling2LoopScratch &s = step.scratch;
  const uint32_t hidden = geometry_.hiddenSize;
  const ops::MessageGateWeights weights = gateWeights(gate);
  ops::MessageGate::add(step.graph,
                        {.x = s.mixed,
                         .xStride = queries * hidden,
                         .xColumn = column * hidden,
                         .message = s.messages,
                         .messageStride = 4 * hidden,
                         .messageColumn = message * hidden,
                         .concat = s.gateConcat,
                         .low = s.gateLow,
                         .forget = s.gateForget,
                         .gain = s.gateGain,
                         .output = std::move(output)},
                        weights, norm, hidden, step.rows,
                        linear(step).ggufFloatTile(step.rows, hidden, weights.forget.formatId));
}

// A prompt sequence's history through the command's reads of spilled K/V
// (QwenTarget::addPrefillMixer); a decode lane's from its pages, which this
// target's dense layout never spills.
void Inkling2LoopTarget::addAttention(PrefillStep &step, uint32_t application,
                                      const std::array<metal::MetalBuffer, 3> &inputs) const {
  const Inkling2LoopAttentionWeights &mixer = weights_.layers[schedule_.weightLayer(application)].mixer;
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Inkling2LoopScratch &s = step.scratch;
  const ops::Linear &linear = this->linear(step);
  const bool full = schedule_.full(application);
  const uint32_t rows = step.rows, kvLayer = step.attentionLayer++;
  linear.addProjection(step.graph, inputs[0], mixer.inputProjection, b.fullPacked, rows, b.linearScratch);
  linear.addProjection(step.graph, inputs[1], mixer.key, s.keys, rows, b.linearScratch);
  linear.addProjection(step.graph, inputs[2], mixer.value, s.values, rows, b.linearScratch);
  const ops::ShortConvShape kv{.channels = schedule_.shape.kvWidth,
                               .dynamicHeads = full ? geometry_.attentionKvHeads : 0};
  metal::MetalBuffer keyTaps, valueTaps;
  uint32_t tapsStride = 0;
  if (full) {
    const auto &taps = std::get<Inkling2LoopFullInputs>(weights_.layers[schedule_.weightLayer(application)].inputs);
    linear.addProjection(step.graph, inputs[1], taps.keyTaps, s.keyTaps, rows, b.linearScratch);
    linear.addProjection(step.graph, inputs[2], taps.valueTaps, s.valueTaps, rows, b.linearScratch);
    keyTaps = s.keyTaps;
    valueTaps = s.valueTaps;
    // The taps projections' rows, whole tiles wide (Inkling2LoopFullInputs).
    tapsStride = taps.keyTaps.outputSize;
  }
  addConvolution(step, application, Inkling2LoopConvolution::Key,
                 {.input = s.keys,
                  .taps = mixer.keyConvolution,
                  .dynamicTaps = keyTaps,
                  .dynamicStride = tapsStride,
                  .output = s.convolvedKeys},
                 kv, ops::ShortConvOutput::Plain);
  addConvolution(step, application, Inkling2LoopConvolution::Value,
                 {.input = s.values,
                  .taps = mixer.valueConvolution,
                  .dynamicTaps = valueTaps,
                  .dynamicStride = tapsStride,
                  .output = s.convolvedValues},
                 kv, ops::ShortConvOutput::Plain);

  const std::span<const kv::ChunkedPrefillParams> sequences = step.attentionSequences();
  const std::span<const metal::MetalBuffer> pageTables = std::span(step.pageTables).first(sequences.size());
  const ops::AttentionRows attentionRows{b.laneRows, rows, step.decode ? step.decode->tree : metal::MetalBuffer{}};
  const ops::RelativeAttentionWeights weights{mixer.queryScale, mixer.relativeScale, mixer.relativeBias};
  const ops::RelativeAttentionStaging staging{b.fullQueries, s.relativeStates, s.rowSinks};
  const ops::RelativeAttentionPlan &plan = full ? *step.fullAttention : *step.slidingAttention;
  ops::RelativeAttention::addProjection(step.graph, {b.fullPacked, s.convolvedKeys, s.convolvedValues}, weights,
                                        staging, step.kvLayers[kvLayer], pageTables, sequences, attentionRows,
                                        geometry_.kvLayout,
                                        {.windowTokens = full ? 0 : geometry_.kvLayout.windowTokens,
                                         .positionScaled = full});
  std::array<ops::AttentionHistory, ExecutionLimits::maximumBatchWidth> histories{};
  for (size_t index = 0; index < step.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
    if (step.decode) {
      if (step.spilled.active()) throw std::logic_error("an Inkling-2-Mini decode step over spilled K/V");
      histories[index] = {step.kvLayers[kvLayer], pageTables[index], {}};
      continue;
    }
    const uint32_t firstPage = sequence.chunk.committed_tokens / kv::kPageTokens;
    const PrefillKv history =
        step.spilled.prefill(step.graph, kvLayer, sequence.lane, firstPage,
                             kv::chunkedPrefillRequiredPages(sequence.chunk) - firstPage,
                             {sequence.pageTable, step.kvLayers[kvLayer]});
    histories[index] = {history.kv, history.table, {}};
  }
  const ops::LinearPlan outputPlan = linear.plan(mixer.outputProjection, rows);
  const ops::PreparedInput gated = ops::RelativeAttention::addAttention(
      step.graph, plan, staging, weights, b.attentionPartials, b.attentionStatistics, sequences,
      std::span(histories).first(sequences.size()), attentionRows,
      {b.fullPacked, b.attentionHidden, ops::AttentionOutput::Every, b.linearScratch, outputPlan.input()});
  linear.add(step.graph,
             {.input = b.attentionHidden, .output = b.attentionOutput, .scratch = b.linearScratch, .prepared = gated},
             mixer.outputProjection, outputPlan);
}

// A prompt sequence's state after its rows goes to its auxiliaryOut and a
// state captured inside them to the capture's cell; a decode lane's rows'
// inputs go to the step's tape, which the commit reads.
void Inkling2LoopTarget::addConvolution(const PrefillStep &step, uint32_t application,
                                        Inkling2LoopConvolution convolution, const ops::ShortConvBuffers &buffers,
                                        ops::ShortConvShape shape, ops::ShortConvOutput output) const {
  std::vector<metal::MetalBuffer> stateIn, stateOut;
  std::vector<ops::ShortConvCapture> captures;
  for (uint32_t index = 0; index < step.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
    stateIn.push_back(sequence.auxiliaryIn);
    if (step.decode) continue;
    stateOut.push_back(sequence.auxiliaryOut);
    for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
      captures.push_back({index, capture.row, capture.auxiliary});
  }
  const ops::ShortConvState state{.column = schedule_.convolutionColumn(application, convolution),
                                  .stride = schedule_.stateColumns()};
  ops::ShortConvolution::addConvolution(step.graph, buffers, shape, output, stateIn, stateOut, state,
                                        step.buffers.laneRows, step.layoutRows, captures,
                                        step.decode ? step.decode->tree : metal::MetalBuffer{},
                                        step.decode ? step.scratch.tape : metal::MetalBuffer{});
}

void Inkling2LoopTarget::addPrefillExperts(PrefillStep &step, uint32_t application) const {
  const Inkling2LoopLayerWeights &layer = weights_.layers[schedule_.weightLayer(application)];
  ops::MoE::addRouted(step.graph, step.ffn, layer.ffn, *step.moe);
  const metal::MetalBuffer output = rows(step, schedule_.output(application));
  addConvolution(step, application, Inkling2LoopConvolution::Ffn,
                 {.input = step.buffers.attentionOutput,
                  .taps = layer.ffnConvolution,
                  .residual = step.scratch.residual,
                  .output = output},
                 {.channels = geometry_.hiddenSize}, ops::ShortConvOutput::RoundedSum);
}

// A decode step's every row, and a prompt chunk's rows of its sequences'
// captures (forEachPrefillCapture).
// A decode step's head reads every row of the final norm.
metal::MetalBuffer Inkling2LoopTarget::addPrefillEnd(PrefillStep &step) const {
  const Inkling2LoopScratch &s = step.scratch;
  const metal::MetalBuffer output = step.decode ? step.decode->finalHidden : step.buffers.hidden[1];
  const metal::MetalBuffer last = rows(step, schedule_.output(schedule_.applications() - 1));
  ops::BlockResidualMix::addMixNorm(
      step.graph, {.stack = s.stack, .planeRows = s.planeRows, .current = last},
      weights_.outputQuery, std::span(&weights_.finalNorm, 1), std::span(&output, 1),
      {.hidden = geometry_.hiddenSize, .stack = schedule_.headStates(), .queries = 1}, step.rows);
  addDraftCapture(step, schedule_.applications(), output);
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    addHead(step.graph, linear(step), d->finalHidden, d->logits, step.rows, step.buffers.linearScratch, {});
    ops::LogitFinish::add(step.graph, d->logits, geometry_.vocabularySize, step.rows, geometry_.logitFinish());
  }
  return output;
}

void Inkling2LoopTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer,
                                      metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const {
  addHead(graph, operators_.linear(), std::move(hidden), logits, rows, scratch, {});
  ops::LogitFinish::add(graph, std::move(logits), geometry_.vocabularySize, rows, geometry_.logitFinish());
}

void Inkling2LoopTarget::addDraftEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                                           metal::MetalBuffer hidden, uint32_t rows) const {
  addEmbedding(graph, std::move(tokens), hidden, rows);
  ops::Normalization::addRms(graph, hidden, weights_.embeddingNorm, hidden, geometry_.hiddenSize, rows);
}

void Inkling2LoopTarget::addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                                        const ops::StepRows &step) const {
  QwenTarget::addStateCommit(graph, buffers, step);
  const uint32_t lanes = step.lanes;
  ops::ShortConvolution::addCommit(graph, steps_.verifyScratch().tape, std::span(buffers.auxiliaryIn).first(lanes),
                                   std::span(buffers.auxiliaryOut).first(lanes), buffers.retainedRows,
                                   schedule_.stateColumns(), buffers.laneRows, {lanes, step.rows, step.laneCap});
}

void Inkling2LoopTarget::addKvPathCommit(metal::CommandGraph &graph, kv::Layout layout, uint32_t extentPages,
                                         std::span<const metal::MetalBuffer> pageTables,
                                         std::span<const uint32_t> committed, uint32_t laneRows,
                                         const metal::MetalBuffer &paths) const {
  ops::RelativeAttention::addPathCommit(graph, layout, extentPages, pageTables, committed, laneRows, paths);
}

std::unique_ptr<QwenTarget> makeQwenTarget(const Inkling2LoopWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t prefillChunks) {
  return std::make_unique<Inkling2LoopTarget>(weights, geometry, backend, operators, prefillChunks);
}

QwenTargetScratchBytes qwenTargetScratchBytes(const Inkling2LoopLayout &layout, const QwenTargetGeometry &geometry,
                                              uint32_t prefillChunks, uint64_t) {
  return Inkling2LoopSteps::scratchBytes(geometry, layout.loopShape(), prefillChunks);
}

ops::AttentionWorkspace qwenTargetAttentionWorkspace(const Inkling2LoopLayout &, const ops::ExecutionPlans &,
                                                     const QwenTargetGeometry &geometry, ops::StepKind step) {
  return step == ops::StepKind::Decode ? ops::RelativeAttention::laneWorkspace(geometry.kvLayout)
                                       : ops::RelativeAttention::sequenceWorkspace(geometry.kvLayout);
}

} // namespace splash::model
