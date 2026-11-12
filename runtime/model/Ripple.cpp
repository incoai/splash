#include "Ripple.hpp"

#include "QwenTargetLoader.hpp"
#include "ops/Embedding.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/RowCopy.hpp"
#include "ops/ValueMix.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace splash::model {

// A Ripple mixer: its fused projection carries the queries, their gate and
// the keys (packedFullWidth past the value rows), and its value projection
// the value rows alone, which the mixed input projects through.
QwenMixerWeights readRippleMixer(WeightFile &file, const BlockTargetFormat &format,
                                 const QwenTargetDimensions &target, bool) {
  QwenAttentionWeights attention;
  // The fused projection's rows sit at the packed rows' heads, whose tail
  // the placed value projection writes: the projection spans the packed rows'
  // whole width, which its dispatches take as the destination's row stride.
  attention.inputProjection = ops::Projection::placed(
      target.packedFullWidth, 0,
      format.fused(file, target.packedFullWidth - target.attentionKvHeads * target.attentionHeadDimension,
                   target.hiddenSize, {"attn-q", "attn-k"}));
  attention.valueProjection =
      ops::Projection::placed(target.packedFullWidth,
                              target.packedFullWidth - target.attentionKvHeads * target.attentionHeadDimension,
                              format.projection(file, target.attentionKvHeads * target.attentionHeadDimension,
                                                target.hiddenSize, "attn-v"));
  attention.queryNorm = format.norm(file, target.attentionHeadDimension, "query-norm");
  attention.keyNorm = format.norm(file, target.attentionHeadDimension, "key-norm");
  attention.outputProjection =
      format.projection(file, target.hiddenSize, target.attentionWidth, "attention-output");
  return attention;
}

RippleWeights loadRippleWeights(metal::MetalBackend &backend, RippleLayout layout,
                                const QwenTargetFiles &files) {
  // A block's FFN and what surrounds it, in its image's order: the
  // value-residual lambda, the dense SwiGLU, and the norms after it.
  const auto readFfn = [&](WeightFile &file, RippleLayerWeights &layer, const BlockTargetFormat &format, uint32_t) {
    layer.valueResidualLambda = file.section(kFloat32Bytes, "value-residual-lambda");
    layer.ffn.gate = format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.ffn.up = format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.ffn.down = format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
    layer.ffnNorm = format.norm(file, layout.hiddenSize, "ffn-norm");
    layer.postFfnNorm = format.norm(file, layout.hiddenSize, "post-ffn-norm");
  };
  return loadQwenTarget<RippleWeights>(
      backend, layout, files, [&](auto &loader, const BlockTargetFormat &format, const MoeExperts &) {
        return readQwenTargetWeights<RippleWeights>(layout, loader, format, readFfn, readRippleMixer);
      });
}

// The mixers of the family's split value: the fused projection's rows into
// the packed rows' queries, gate and keys; the block's mix of its normed
// input and the scaled embeddings; and the value rows at the packed rows'
// tails. The attention and its gate read the packed rows as every family's
// do; the output projection runs without a residual, which the block's
// post-attention norm and add apply.
void RippleTarget::addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer,
                                   const metal::MetalBuffer &lambda) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.attentionLayer++;
  const ops::AttentionVariant variant = geometry_.attentionVariant(layer);
  const std::vector<ops::PrefillAttentionPlan> &plans =
      variant.windowTokens ? step.windowedAttention : step.attention;
  const ops::Linear &linear = operators_.linear();
  const auto u16Rows = [&](const metal::MetalBuffer &buffer, const QwenTargetPrefillSequence &sequence,
                           uint32_t width) {
    return rowsOf<uint16_t>(buffer, sequence.rowBegin, sequence.rows, width);
  };
  const auto f32Rows = [&](const metal::MetalBuffer &buffer, const QwenTargetPrefillSequence &sequence,
                           uint32_t width) {
    return rowsOf<float>(buffer, sequence.rowBegin, sequence.rows, width);
  };
  linear.addPrefill(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, step.rows, b.linearScratch);
  ops::ValueMix::add(step.graph, b.normalized, embedRows_, lambda, b.denseIntermediate, step.rows,
                     geometry_.hiddenSize);
  linear.addPrefill(step.graph, b.denseIntermediate, mixer.valueProjection, b.fullPacked, step.rows,
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
        geometry_.kvLayout, variant);
    ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[layer], keys, values, sequence.pageTable,
                                         sequence.chunk, geometry_.kvLayout);
    const uint32_t firstPage = sequence.chunk.committed_tokens / kv::kPageTokens;
    const PrefillKv attended =
        step.spilled.prefill(step.graph, layer, sequence.lane, firstPage,
                             kv::chunkedPrefillRequiredPages(sequence.chunk) - firstPage,
                             {sequence.pageTable, step.kvLayers[layer]});
    ops::PagedAttention::addPrefill(step.graph, attended.kv, queries, attentionRows, b.attentionPartials,
                                    b.attentionStatistics, attended.table, sequence.chunk, plans[index],
                                    ops::AttentionBlocks{}, mixer.sinks);
    ops::PagedAttention::addPrefillGate(
        step.graph, u16Rows(b.fullPacked, sequence, geometry_.packedFullWidth), attentionRows,
        u16Rows(b.attentionHidden, sequence, geometry_.attentionWidth), sequence.rows, sequence.attentionStride,
        geometry_.attentionQueryHeads, geometry_.kvLayout);
  }
  linear.addPrefill(step.graph, b.attentionHidden, mixer.outputProjection, b.attentionOutput, step.rows,
                    b.linearScratch);
}

// The verify mixers, whose fused and value projections run per lane batch
// (decodeInput), over the normalized rows the table describes.
void RippleTarget::addVerifyProjectionPair(VerifyStep &step, const QwenAttentionWeights &mixer,
                                           const metal::MetalBuffer &lambda, ops::PreparedInput normalized) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  addDecodeProjection(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, step.lanes,
                      b.linearScratch, std::move(normalized));
  ops::ValueMix::add(step.graph, b.normalized, embedRows_, lambda, b.denseIntermediate, step.rows,
                     geometry_.hiddenSize);
  addDecodeProjection(step.graph, b.denseIntermediate, mixer.valueProjection, b.fullPacked, step.lanes,
                      b.linearScratch);
}

void RippleTarget::addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer,
                                  const metal::MetalBuffer &lambda, ops::PreparedInput normalized,
                                  BlockOutput output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const uint32_t layer = step.attentionLayer++;
  const ops::AttentionVariant variant = geometry_.attentionVariant(layer);
  addVerifyProjectionPair(step, mixer, lambda, std::move(normalized));
  ops::PagedAttention::addVerifyProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                           b.ropeSin, b.fullQueries, b.chunkKeys[layer], b.chunkValues[layer],
                                           geometry_.attentionQueryHeads, geometry_.kvLayout, step.lanes, variant);
  ops::AttentionBlocks attended;
  if (step.spilled.active()) attended = step.spilled.verify(step.graph, layer, ops::AttentionBlocks{});
  ops::PagedAttention::addVerify(step.graph, step.kvLayers[layer],
                                 {b.chunkKeys[layer], b.chunkValues[layer], b.fullQueries, b.attentionPartials,
                                  b.attentionStatistics, b.fullAttention, b.pageTables, mixer.sinks},
                                 step.chunks, variant.windowTokens ? *step.windowedAttention : step.attention,
                                 attended);
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan outputPlan = linear.decodePlan(mixer.outputProjection, step.lanes);
  const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
      step.graph, b.fullPacked, b.fullAttention, b.attentionHidden, geometry_.attentionQueryHeads,
      geometry_.kvLayout, step.lanes, b.linearScratch, outputPlan.input());
  linear.add(step.graph,
             {.input = b.attentionHidden, .output = std::move(output.rows), .scratch = b.linearScratch,
              .prepared = hidden},
             mixer.outputProjection, outputPlan);
}

// A block's dense SwiGLU, whose gate and up activations clamp at the
// family's threshold, and the norms and adds of the post-norm arrangement
// around it: the FFN reads the norm of the residual the post-attention norm
// made, and its output's norm adds back to that residual, which the
// attention's norm and add made.
void RippleTarget::addPrefillFfn(PrefillStep &step, const RippleLayerWeights &layer, metal::MetalBuffer ffnIn,
                                 metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  ops::Normalization::addRms(step.graph, ffnIn, layer.ffnNorm, b.normalized, geometry_.hiddenSize, step.rows);
  linear.addPrefill(step.graph, b.normalized, layer.ffn.gate, b.denseGateScratch, step.rows, b.linearScratch);
  linear.add(step.graph,
             {.input = b.normalized, .output = b.denseIntermediate, .gateScratch = b.denseGateScratch,
              .scratch = b.linearScratch, .clamp = geometry_.swigluClamp},
             layer.ffn.up, linear.prefillPlan(layer.ffn.up, step.rows, ops::LinearEpilogue::ClampedUpWithGate));
  linear.addPrefill(step.graph, b.denseIntermediate, layer.ffn.down, output, step.rows, b.linearScratch);
  ops::Normalization::addRms(step.graph, output, layer.postFfnNorm, b.normalized, geometry_.hiddenSize, step.rows);
  ops::ValueMix::sum(step.graph, b.normalized, ffnIn, output, step.rows, geometry_.hiddenSize);
}

void RippleTarget::addVerifyFfn(VerifyStep &step, const RippleLayerWeights &layer, metal::MetalBuffer ffnIn,
                                metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan gateUpPlan =
      linear.decodePlan(layer.ffn.up, step.lanes, ops::LinearEpilogue::ClampedGateUp, &layer.ffn.gate);
  const ops::PreparedInput normalized =
      ops::Normalization::addRms(step.graph, ffnIn, layer.ffnNorm, b.normalized, geometry_.hiddenSize, step.rows,
                                 b.linearScratch, gateUpPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.denseIntermediate, .gateScratch = b.denseGateScratch,
              .scratch = b.linearScratch, .prepared = normalized, .clamp = geometry_.swigluClamp},
             layer.ffn.up, gateUpPlan, &layer.ffn.gate);
  linear.add(step.graph, {.input = b.denseIntermediate, .output = output, .scratch = b.linearScratch},
             layer.ffn.down, linear.decodePlan(layer.ffn.down, step.lanes));
  ops::Normalization::addRms(step.graph, output, layer.postFfnNorm, b.normalized, geometry_.hiddenSize, step.rows);
  ops::ValueMix::sum(step.graph, b.normalized, ffnIn, output, step.rows, geometry_.hiddenSize);
}

// The geometry of a Ripple target's projections: its dense FFNs' gate and up
// run as one gate/up plan each, and its layers keep no MoE blocks.
void includeFamilyProjections(QwenTargetGeometry &geometry, const RippleWeights &weights) {
  for (const RippleLayerWeights &layer : weights.layers)
    includeDenseFfn(geometry, layer.ffn.projections());
}

const ops::BlockMoeWeights *qwenMoeBlock(const RippleLayerWeights &) noexcept { return nullptr; }

RippleTarget::RippleTarget(const RippleWeights &weights, const QwenTargetGeometry &geometry,
                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights) {
  requireLayers(weights.layers.size(), weights.layers.size());
  // The family's SwiGLU clamp, one F32 the clamped FFN epilogues read.
  if (!(geometry_.swigluClamp > 0.0F))
    throw std::invalid_argument("Ripple's SwiGLU clamp is not a positive threshold");
  clamp_ = backend.allocateBuffer(4, metal::BufferStorage::Shared, "ripple clamp");
  *static_cast<float *>(clamp_.contents()) = geometry_.swigluClamp;
  // The scaled token embedding rows, which every block's mix reads back: a
  // prefill chunk's or a verify batch's, written once per gather.
  embedRows_ = backend.allocateBuffer(uint64_t{SPLASH_PREFILL_TOKEN_BUDGET} * geometry_.hiddenSize * 2,
                                      metal::BufferStorage::Shared, "ripple embeddings");
}

RippleTarget::~RippleTarget() = default;

QwenTargetScratchBytes RippleTarget::allocatedScratchBytes() const noexcept {
  return {embedRows_.sizeBytes(), 0};
}

void RippleTarget::addEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                                metal::MetalBuffer hidden, uint32_t rows) const {
  // The gather writes the scaled rows the blocks mix back in, and the block
  // input reads their copy. A vision family's image rows would write the
  // block input's placeholders after the copy; the family serves text only.
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, embedRows_, rows,
                      geometry_.embeddingScale);
  ops::RowCopy::add(graph, embedRows_, {0, geometry_.hiddenSize, 0}, hidden, {0, geometry_.hiddenSize, 0},
                    rows, geometry_.hiddenSize);
}

std::vector<metal::MetalBuffer> RippleTarget::addPrefill(metal::CommandGraph &graph,
                                                         std::span<const QwenTargetPrefillChunk> chunks,
                                                         const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                         QwenPrefillLayers layers, ops::AneFfn *aneFfn) {
  if (aneFfn)
    throw std::invalid_argument("Ripple's FFN does not split to the Neural Engine");
  std::vector<PrefillStep> steps = prefillSteps(graph, chunks, kv, spilled, layers, aneFfn);
  return addPrefillLayers(std::span(steps), *this, layers);
}

void RippleTarget::addPrefillLayer(PrefillStep &step, uint32_t index) const {
  const RippleLayerWeights &layer = weights_.layers[index];
  const QwenTargetPrefillBuffers &b = step.buffers;
  const metal::MetalBuffer input = b.hidden[index & 1], output = b.hidden[(index & 1) ^ 1];
  ops::Normalization::addRms(step.graph, input, layer.inputNorm, b.normalized, geometry_.hiddenSize, step.rows);
  addPrefillMixer(step, std::get<QwenAttentionWeights>(layer.mixer), layer.valueResidualLambda);
  // The attention branch: its output's norm, then the residual it adds to.
  ops::Normalization::addRms(step.graph, b.attentionOutput, layer.postAttentionNorm, b.normalized,
                             geometry_.hiddenSize, step.rows);
  ops::ValueMix::sum(step.graph, b.normalized, input, b.attentionOutput, step.rows, geometry_.hiddenSize);
  addPrefillFfn(step, layer, b.attentionOutput, output);
}

const ops::BlockMoeWeights *RippleTarget::moe(uint32_t) const noexcept { return nullptr; }

void RippleTarget::addPrefillExperts(PrefillStep &step, uint32_t index) const {
  forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t rows) {
    addCapture(step.graph, slot, step.buffers.hidden[(index & 1) ^ 1], source, step.buffers.captured,
               destination, rows);
  });
}

metal::MetalBuffer RippleTarget::addPrefillEnd(PrefillStep &step) const {
  return step.buffers.hidden[geometry_.layers & 1];
}

void RippleTarget::addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers, const kv::PageStorage &kv,
                             SpilledKvReads &spilled, std::span<const kv::ChunkedPrefillParams> chunks,
                             uint32_t lanes, uint32_t blockRows) const {
  VerifyStep step = verifyStep(graph, buffers, kv, spilled, chunks, lanes);
  step.blockRows = blockRows;
  const bool wholeBlock = blockRows == SPLASH_TARGET_VERIFY_ROWS;
  const uint32_t headRows = wholeBlock ? step.rows : lanes * blockRows;
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const RippleLayerWeights &layer = weights_.layers[index];
    const metal::MetalBuffer input = buffers.hidden[index & 1], output = buffers.hidden[(index & 1) ^ 1];
    const ops::PreparedInput normalized = ops::Normalization::addRms(
        step.graph, input, layer.inputNorm, buffers.normalized, geometry_.hiddenSize, step.rows,
        buffers.linearScratch, decodeInput(std::get<QwenAttentionWeights>(layer.mixer).inputProjection,
                                           step.lanes));
    addVerifyMixer(step, std::get<QwenAttentionWeights>(layer.mixer), layer.valueResidualLambda,
                   std::move(normalized), {buffers.attentionOutput, input});
    // The attention branch: its output's norm, then the residual it adds to.
    ops::Normalization::addRms(step.graph, buffers.attentionOutput, layer.postAttentionNorm, buffers.normalized,
                               geometry_.hiddenSize, step.rows);
    ops::ValueMix::sum(step.graph, buffers.normalized, input, buffers.attentionOutput, step.rows,
                       geometry_.hiddenSize);
    addVerifyFfn(step, layer, buffers.attentionOutput, output);
    if (const auto slot = geometry_.captureSlot(index))
      addCapture(step.graph, *slot, output, 0, buffers.capturedTargetHidden,
                 step.rowBase, step.rows, {SPLASH_TARGET_VERIFY_ROWS, blockRows});
  }
  requireLayerPartition(step.gdnLayer, step.attentionLayer);
  // The pass's rows join the lanes' whole-block rows for the head, which the
  // pass that completes the block runs over all of them.
  if (!wholeBlock)
    ops::RowCopy::add(step.graph, buffers.hidden[geometry_.layers & 1], {0, geometry_.hiddenSize, 0},
                      buffers.headInput, {step.rowBase, geometry_.hiddenSize, 0}, step.rows,
                      geometry_.hiddenSize, {SPLASH_TARGET_VERIFY_ROWS, blockRows});
  if (chunks.front().row_base + SPLASH_TARGET_VERIFY_ROWS >= blockRows)
    addHeadBatch(step.graph, wholeBlock ? buffers.hidden[geometry_.layers & 1] : buffers.headInput,
                 buffers.finalHidden, buffers.logits, step.lanes, headRows, buffers.linearScratch);
}

void RippleTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                                metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
                                uint32_t rows, ops::LinearScratch scratch) const {
  addNormalizedHead(graph, weights_.finalNorm, std::move(hidden), std::move(finalHidden), std::move(logits),
                    lanes, rows, scratch);
}

QwenTargetScratchBytes qwenTargetScratchBytes(const RippleWeights &, const QwenTargetGeometry &geometry,
                                              uint32_t, uint64_t) {
  // The target's own scratch in the prefill path: the scaled embedding rows
  // the blocks mix back in (RippleTarget's embed rows).
  return {uint64_t{ExecutionLimits::prefillTokenBudget} * geometry.hiddenSize * 2, 0};
}

std::unique_ptr<QwenTarget> makeQwenTarget(const RippleWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t) {
  return std::make_unique<RippleTarget>(weights, geometry, backend, operators);
}

} // namespace splash::model
