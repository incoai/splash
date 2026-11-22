#include "model/InklingTarget.hpp"

#include "metal/abi/ShortConvolution.h"
#include "model/Inkling.hpp"
#include "model/WeightStore.hpp"
#include "ops/AneFfn.hpp"
#include "ops/BlockResidual.hpp"
#include "ops/Normalization.hpp"
#include "ops/PageStorage.hpp"
#include "ops/Sampling.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

constexpr ops::FloatElement kF32 = ops::FloatElement::Float32;
constexpr ops::FloatElement kBf16 = ops::FloatElement::BFloat16;
constexpr uint32_t kLanes = ExecutionLimits::maximumBatchWidth;
// The convolutions' histories and tape keep their inputs as
// kShortConvolutionStateValueBytes elements.
constexpr ops::FloatElement kStateElement = kShortConvolutionStateValueBytes == 2 ? kBf16 : kF32;

// A prompt chunk's rows as its layout holds them
// (QwenTargetPrefillBuffers::laneRows): its sequences' `rows` rows, each at
// most the longest's.
ops::LayoutRows promptRows(std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows) {
  uint32_t longest = 0;
  for (const QwenTargetPrefillSequence &sequence : sequences) longest = std::max(longest, sequence.rows);
  return {static_cast<uint32_t>(sequences.size()), rows, longest};
}

// A loop gate's weights as the loop control reads them: its projections'
// rows as the GGUF stores them, F32 or BF16, all three of one type.
ops::LoopControlWeights loopControlWeights(const InklingLoopGateWeights &gate) {
  const auto rows = [](const ops::Projection &projection) -> const ops::QuantizedSegment & {
    const std::vector<ops::QuantizedSegment> &segments = projection.blocks().segments;
    if (segments.size() != 1 || !segments.front().isFloat() ||
        (segments.front().formatId != ops::QuantizedSegment::kFloat32 && segments.front().formatId != GGUF_FMT_BF16))
      throw WeightStoreError("an Inkling loop gate's weights are F32 or BF16 rows");
    return segments.front();
  };
  const ops::QuantizedSegment &down = rows(gate.down), &forget = rows(gate.forget), &gain = rows(gate.gain);
  if (forget.formatId != down.formatId || gain.formatId != down.formatId)
    throw WeightStoreError("an Inkling loop gate's weights are of one type");
  return {down.plane0, forget.plane0, gain.plane0, down.formatId == ops::QuantizedSegment::kFloat32 ? kF32 : kBf16};
}

} // namespace

InklingTarget::InklingTarget(const InklingWeights &weights, const QwenTargetGeometry &geometry,
                             metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                             uint32_t prefillChunks)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights),
      steps_(geometry_, backend, prefillChunks) {
  // Every layer attends with K/V of its own, a weight layer's for each that
  // runs it; the sides are the relative geometry's rows.
  requireLayers(geometry_.layers, geometry_.layers);
  const InklingSides sides = InklingSides::of(geometry_);
  if (weights_.layers.size() != geometry_.weightLayers() || geometry_.relativeDimension != ops::kRelativeWidth ||
      geometry_.relativeExtent != ops::kRelativeExtent ||
      sides.query != ops::relativeQuerySideWidth(geometry_.attentionQueryHeads, geometry_.attentionHeadDimension) ||
      geometry_.shortConvolutionTaps != SHORT_CONVOLUTION_TAPS)
    throw std::invalid_argument("Inkling target weights do not match execution geometry");
  for (uint32_t index = 0; index < weights_.layers.size(); ++index) {
    const InklingLayerWeights &layer = weights_.layers[index];
    const bool core = index >= geometry_.loopStart && index < geometry_.loopEnd;
    if (core != layer.loop.has_value() || !index != !layer.mixer.residualQueries)
      throw std::invalid_argument("Inkling target weights do not match execution geometry");
    if (layer.loop) {
      auto &gates = loopGates_.emplace_back();
      for (uint32_t stream = 0; stream < gates.size(); ++stream)
        gates[stream] = loopControlWeights(layer.loop->gates[stream]);
    }
  }
}

QwenTargetScratchBytes InklingTarget::allocatedScratchBytes() const noexcept { return steps_.allocatedBytes(); }

std::vector<metal::MetalBuffer> InklingTarget::addPrefill(metal::CommandGraph &graph,
                                                          std::span<const QwenTargetPrefillChunk> chunks,
                                                          const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                          QwenPrefillLayers layers, ops::AneFfn *aneFfn) {
  std::vector<PrefillStep> steps = prefillSteps<PrefillStep>(graph, chunks, kv, spilled, layers, aneFfn);
  for (uint32_t index = 0; index < steps.size(); ++index) {
    PrefillStep &step = steps[index];
    if (const QwenTargetDecodeBuffers *d = step.decode) {
      step.scratch = steps_.decodeScratch();
      step.layoutRows = {d->step.lanes, d->step.rows, d->step.laneCap};
    } else {
      step.scratch = steps_.prefillScratch(index);
      step.layoutRows = promptRows(step.sequences, step.rows);
    }
    if (step.rows > step.scratch.rows) throw std::invalid_argument("an Inkling step past its scratch's rows");
  }
  return addPrefillLayers(std::span(steps), *this, layers);
}

ops::StridedRows InklingTarget::stateRows(const PrefillStep &step, uint32_t state) const {
  return {step.scratch.states.at(state), geometry_.hiddenSize};
}

// x = RMSNorm(E[token]) into the embedding's state, in fp32.
void InklingTarget::addPrefillBegin(PrefillStep &step) const {
  ops::Normalization::addRmsRows(step.graph, step.buffers.hidden[0], kBf16, weights_.embeddingNorm,
                                 step.scratch.states[kInklingEmbeddingState], kF32, geometry_.hiddenSize, step.rows);
}

// The query, key and value inputs: layer 0's is the embedding's
// rows; another layer's the depth mix of the blocks before it and its input
// with its queries, one of a sliding layer, the query's, key's and value's of
// a global one. On the loop's passes after its first each goes through its
// gate with its message: a global layer's in place, a sliding layer's, whose
// three share one mix, over its message's rows. Each is normed with its
// stream's norm (a sliding layer's one for all three) into bf16 rows: a
// sliding layer without messages norms one mix for all three.
std::array<metal::MetalBuffer, 3> InklingTarget::addAttentionInputs(PrefillStep &step, uint32_t layer,
                                                                    const InklingLayerStates &states) const {
  const InklingScratch &s = step.scratch;
  const uint32_t weight = geometry_.weightLayer(layer), hidden = geometry_.hiddenSize, rows = step.rows;
  const InklingAttentionWeights &mixer = weights_.layers[weight].mixer;
  const bool global = inklingGlobalLayer(geometry_, weight);
  std::array<metal::MetalBuffer, 3> inputs;
  if (!layer) {
    inputs.fill(s.states[kInklingEmbeddingState]);
  } else {
    std::vector<ops::StridedRows> mixed;
    for (uint32_t block = 0; block < states.attentionBlocks; ++block)
      mixed.push_back(stateRows(step, states.blocks[block]));
    mixed.push_back(stateRows(step, states.input));
    const uint32_t queries = global ? 3 : 1;
    ops::BlockResidual::add(step.graph, mixed, mixer.residualQueries, std::span(s.inputs).first(queries), hidden,
                            hidden, rows);
    for (uint32_t stream = 0; stream < inputs.size(); ++stream) inputs[stream] = s.inputs[global ? stream : 0];
  }
  if (states.messages) {
    const auto &gates = loopGates_.at(weight - geometry_.loopStart);
    std::array<ops::LoopControlStream, 3> streams;
    for (uint32_t stream = 0; stream < streams.size(); ++stream) {
      const metal::MetalBuffer output = global ? inputs[stream] : s.messages[stream];
      streams[stream] = {&gates[stream], inputs[stream], s.messages[stream], output, s.low[stream]};
      inputs[stream] = output;
    }
    ops::LoopControl::add(step.graph, streams, {hidden, InklingLayout::loopGateRank, hidden, hidden, hidden}, rows);
  }
  const QwenTargetPrefillBuffers &b = step.buffers;
  std::array<metal::MetalBuffer, 3> normalized{b.normalized, s.keyInput, s.valueInput};
  const uint32_t streams = global || states.messages ? 3 : 1;
  for (uint32_t stream = 0; stream < streams; ++stream)
    ops::Normalization::addRmsRows(step.graph, inputs[stream], kF32, mixer.inputNorms[global ? stream : 0],
                                   normalized[stream], kBf16, hidden, rows);
  if (streams == 1) normalized[1] = normalized[2] = normalized[0];
  return normalized;
}

void InklingTarget::addConvolution(PrefillStep &step, uint32_t layer, InklingConvolution convolution,
                                   const ops::ShortConvolutionBuffers &buffers,
                                   ops::ShortConvolutionShape shape) const {
  const size_t sequences = step.sequences.size();
  std::array<metal::MetalBuffer, kLanes> in, out;
  for (size_t index = 0; index < sequences; ++index) {
    in[index] = steps_.history(step.sequences[index].auxiliaryIn, layer, convolution);
    if (!step.decode) out[index] = steps_.history(step.sequences[index].auxiliaryOut, layer, convolution);
  }
  shape.state = kStateElement;
  const std::span<const metal::MetalBuffer> before = std::span(in).first(sequences);
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    ops::ShortConvolution::addConvolution(step.graph, buffers, {before, {}, steps_.historyStride()}, shape,
                                          step.buffers.laneRows, step.layoutRows,
                                          {d->tree, steps_.tape(step.scratch, layer, convolution)});
    return;
  }
  ops::ShortConvolution::addConvolution(step.graph, buffers,
                                        {before, std::span(out).first(sequences), steps_.historyStride()}, shape,
                                        step.buffers.laneRows, step.layoutRows);
  // A state captured inside a sequence's rows keeps the history after its
  // rows before the capture.
  const uint64_t elementBytes = ops::elementBytes(shape.input);
  for (size_t index = 0; index < sequences; ++index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
    const uint64_t first = uint64_t{sequence.rowBegin} * buffers.input.stride * elementBytes;
    const ops::StridedRows rows{backend_.view(buffers.input.buffer, first, buffers.input.buffer.sizeBytes() - first),
                                buffers.input.stride};
    for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
      ops::ShortConvolution::addHistory(step.graph, rows, in[index],
                                        steps_.history(capture.auxiliary, layer, convolution), steps_.historyStride(),
                                        shape, capture.row);
  }
}

// One layer: its messages on the loop's later passes; its
// attention inputs; its sides; its keys' and values' convolutions; the
// relative prepare, which stores the layer's K/V; the attention, its output
// projection and convolution, which starts the block's stream or adds to
// it; the MoE's input mix, gate and norm; and the MoE's routing, whose
// routed work addPrefillExperts adds.
void InklingTarget::addPrefillLayer(PrefillStep &step, uint32_t index) const {
  const uint32_t layer = step.attentionLayer++;
  if (layer != index) throw std::logic_error("an Inkling layer attends K/V of its own");
  const uint32_t weight = geometry_.weightLayer(layer), hidden = geometry_.hiddenSize, rows = step.rows;
  const InklingLayerWeights &w = weights_.layers[weight];
  const InklingLayerStates states = inklingLayerStates(geometry_, layer);
  const bool global = inklingGlobalLayer(geometry_, weight);
  const InklingSides sides = InklingSides::of(geometry_);
  const InklingScratch &s = step.scratch;
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = this->linear(step);
  // A pass's messages to each of the layer's four inputs, from the states
  // the pass before left.
  if (states.messages) {
    const std::array<ops::StridedRows, 2> bank{stateRows(step, states.bank[0]), stateRows(step, states.bank[1])};
    ops::BlockResidual::add(step.graph, bank, w.loop->messageQueries, s.messages, hidden, hidden, rows);
  }
  const std::array<metal::MetalBuffer, 3> inputs = addAttentionInputs(step, layer, states);
  // The sides, the query's first; a key or value side that reads the query's
  // input takes the table its plan prepared.
  const auto side = [&](const metal::MetalBuffer &input, const ops::Projection &projection,
                        const metal::MetalBuffer &output, ops::PreparedInput prepared) {
    return linear.add(step.graph,
                      {.input = input, .output = output, .scratch = b.linearScratch, .prepared = std::move(prepared)},
                      projection, linear.plan(projection, rows));
  };
  const ops::PreparedInput query = side(inputs[0], w.mixer.querySide, s.querySide, {});
  side(inputs[1], w.mixer.keySide, s.keySide, inputs[1].sameView(inputs[0]) ? query : ops::PreparedInput{});
  side(inputs[2], w.mixer.valueSide, s.valueSide, inputs[2].sameView(inputs[0]) ? query : ops::PreparedInput{});
  // The keys' and values' convolutions, with a global layer's
  // dynamic taps from the sides' rows past them.
  const uint32_t sideRows = sides.keyRows(global), kvWidth = sides.lags;
  const uint32_t headChannels = global ? geometry_.attentionHeadDimension : 0;
  const auto lags = [&](const metal::MetalBuffer &sideBuffer) {
    if (!global) return ops::StridedRows{};
    const uint64_t at = uint64_t{sides.lags} * sizeof(float);
    return ops::StridedRows{backend_.view(sideBuffer, at, sideBuffer.sizeBytes() - at), sideRows};
  };
  addConvolution(step, layer, InklingConvolution::Key,
                 {{s.keySide, sideRows}, lags(s.keySide), w.mixer.keyConvolution, {s.keys, kvWidth}, false},
                 {kvWidth, headChannels, kF32});
  addConvolution(step, layer, InklingConvolution::Value,
                 {{s.valueSide, sideRows}, lags(s.valueSide), w.mixer.valueConvolution, {s.values, kvWidth}, false},
                 {kvWidth, headChannels, kF32});
  // The relative prepare stores each row's key and value in its page slot;
  // a global layer scales its queries and relative states by tau of their
  // positions.
  ops::AttentionVariant variant = geometry_.attentionVariant(layer);
  variant.positionScaled = global;
  const std::span<const kv::ChunkedPrefillParams> sequences = step.attentionSequences();
  const ops::AttentionRows attentionRows{b.laneRows, rows, step.decode ? step.decode->tree : metal::MetalBuffer{}};
  ops::PagedAttention::addRelativeProjection(
      step.graph, {s.querySide, s.keys, s.values, w.mixer.queryClip, w.mixer.relativeClip}, b.fullQueries,
      s.relative, step.kvLayers[layer], std::span(step.pageTables).first(sequences.size()), sequences, attentionRows,
      geometry_.attentionQueryHeads, geometry_.kvLayout, variant);
  const auto histories = attentionHistories(step, layer);
  const ops::AttentionPlan &plan = variant.windowTokens ? *step.windowedAttention : *step.attention;
  const ops::LinearPlan outputPlan = linear.plan(w.mixer.outputProjection, rows);
  const ops::PreparedInput attended = ops::PagedAttention::addAttention(
      step.graph, plan, b.fullQueries, b.attentionPartials, b.attentionStatistics, sequences,
      std::span(histories).first(sequences.size()), attentionRows,
      {s.querySide, b.attentionHidden, ops::AttentionOutput::Every, b.linearScratch, outputPlan.input()}, {},
      {s.relative, w.mixer.relativeBias});
  linear.add(step.graph,
             {.input = b.attentionHidden, .output = b.attentionOutput, .scratch = b.linearScratch,
              .prepared = attended},
             w.mixer.outputProjection, outputPlan);
  // The attention's output convolution: a block's first layer
  // starts its stream with it, and its input joins the blocks; another adds
  // it to its stream.
  addConvolution(step, layer, InklingConvolution::Attention,
                 {{b.attentionOutput, hidden}, {}, w.mixer.outputConvolution,
                  stateRows(step, states.output), !states.startsBlock()},
                 {hidden, 0, kBf16});
  // The MoE's input: the depth mix of the blocks and the stream,
  // through its gate on a later pass, normed into the MoE's input rows.
  std::vector<ops::StridedRows> mixed;
  for (uint32_t block = 0; block < states.ffnBlocks; ++block) mixed.push_back(stateRows(step, states.blocks[block]));
  mixed.push_back(stateRows(step, states.output));
  ops::BlockResidual::add(step.graph, mixed, w.ffnResidualQuery, std::span(s.inputs).first(1), hidden, hidden, rows);
  if (states.messages) {
    const ops::LoopControlStream stream{&loopGates_.at(weight - geometry_.loopStart).back(), s.inputs[0],
                                        s.messages.back(), s.inputs[0], s.low.back()};
    ops::LoopControl::add(step.graph, std::span(&stream, 1),
                          {hidden, InklingLayout::loopGateRank, hidden, hidden, hidden}, rows);
  }
  ops::Normalization::addRmsRows(step.graph, s.inputs[0], kF32, w.ffnNorm, b.normalized, kBf16, hidden, rows);
  // The MoE adds its output to zeros into the attention output's rows, which
  // its convolution reads; a decode step's lanes route their rows alone.
  step.ffn = {.input = b.normalized, .residual = s.zeros, .output = b.attentionOutput, .scratch = b.moe};
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    step.ffn.lanes = d->step.lanes;
    step.ffn.laneRows = b.laneRows;
  }
  ops::MoE::addRouting(step.graph, step.ffn, w.ffn, *step.moe);
}

// The weight layers' MoE blocks, which every pass of the loop shares.
const ops::BlockMoeWeights *InklingTarget::moe(uint32_t index) const {
  return &weights_.layers[geometry_.weightLayer(index)].ffn;
}

// The routed experts, then the MoE output's convolution added to the stream.
void InklingTarget::addPrefillExperts(PrefillStep &step, uint32_t index) const {
  const InklingLayerWeights &w = weights_.layers[geometry_.weightLayer(index)];
  ops::MoE::addRouted(step.graph, step.ffn, w.ffn, *step.moe);
  const uint32_t hidden = geometry_.hiddenSize;
  addConvolution(step, index, InklingConvolution::Ffn,
                 {{step.buffers.attentionOutput, hidden}, {}, w.ffnConvolution,
                  stateRows(step, inklingLayerStates(geometry_, index).output), true},
                 {hidden, 0, kBf16});
}

// The head's input: the depth mix of every block and the coda's
// stream, normed into bf16 rows; a decode step's head reads every row of
// them.
metal::MetalBuffer InklingTarget::addPrefillEnd(PrefillStep &step) const {
  const InklingScratch &s = step.scratch;
  const uint32_t hidden = geometry_.hiddenSize;
  std::array<ops::StridedRows, kInklingFinalStates.size()> states;
  std::ranges::transform(kInklingFinalStates, states.begin(), [&](uint32_t state) { return stateRows(step, state); });
  ops::BlockResidual::add(step.graph, states, weights_.outputQuery, std::span(s.inputs).first(1), hidden, hidden,
                          step.rows);
  const QwenTargetDecodeBuffers *d = step.decode;
  const metal::MetalBuffer normalized = d ? d->finalHidden : step.buffers.hidden[1];
  ops::Normalization::addRmsRows(step.graph, s.inputs[0], kF32, weights_.finalNorm, normalized, kBf16, hidden,
                                 step.rows);
  if (d) addHead(step.graph, linear(step), normalized, d->logits, step.rows, step.buffers.linearScratch, {});
  return normalized;
}

void InklingTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer,
                                 metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const {
  addHead(graph, operators_.linear(), std::move(hidden), std::move(logits), rows, scratch, {});
}

void InklingTarget::addInsideCapture(metal::CommandGraph &graph, metal::MetalBuffer auxiliaryIn,
                                     metal::MetalBuffer captured) const {
  const InklingScratch scratch = steps_.prefillScratch(0);
  const InklingSides sides = InklingSides::of(geometry_);
  const uint32_t stride = steps_.historyStride();
  ops::ShortConvolution::addHistory(graph, {scratch.keySide, sides.globalKey},
                                    steps_.history(auxiliaryIn, 0, InklingConvolution::Key),
                                    steps_.history(captured, 0, InklingConvolution::Key), stride,
                                    {sides.lags, 0, kF32, kStateElement}, 1);
  ops::ShortConvolution::addHistory(graph, {scratch.keyInput, geometry_.hiddenSize},
                                    steps_.history(auxiliaryIn, 0, InklingConvolution::Attention),
                                    steps_.history(captured, 0, InklingConvolution::Attention), stride,
                                    {geometry_.hiddenSize, 0, kBf16, kStateElement}, 1);
}

// The tape holds each layer's rows of its four convolutions side by side, as
// a history does: one unit a layer.
void InklingTarget::addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                                   const ops::StepRows &step) const {
  QwenTarget::addStateCommit(graph, buffers, step);
  std::array<metal::MetalBuffer, kLanes> in, out;
  for (uint32_t lane = 0; lane < step.lanes; ++lane) {
    in[lane] = steps_.histories(buffers.auxiliaryIn[lane]);
    out[lane] = steps_.histories(buffers.auxiliaryOut[lane]);
  }
  const InklingScratch scratch = steps_.decodeScratch();
  const uint32_t stride = steps_.historyStride();
  ops::ShortConvolution::addCommit(graph, scratch.tape, std::span(in).first(step.lanes),
                                   std::span(out).first(step.lanes), buffers.retainedRows,
                                   {stride, geometry_.layers, stride, SHORT_CONVOLUTION_HISTORY * stride, stride,
                                    scratch.rows * stride, kStateElement},
                                   buffers.laneRows, {step.lanes, step.rows, step.laneCap});
}

std::unique_ptr<QwenTarget> makeQwenTarget(const InklingWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t prefillChunks) {
  return std::make_unique<InklingTarget>(weights, geometry, backend, operators, prefillChunks);
}

QwenTargetScratchBytes qwenTargetScratchBytes(const InklingLayout &, const QwenTargetGeometry &geometry,
                                              uint32_t prefillChunks, uint64_t) {
  return InklingSteps::scratchBytes(geometry, prefillChunks);
}

} // namespace splash::model
