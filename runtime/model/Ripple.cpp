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
  ops::ValueMix::add(step.graph, b.normalized, b.embeddings, lambda, b.denseIntermediate, step.rows,
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
    if (step.readRows.empty()) {
      ops::PagedAttention::addPrefill(step.graph, attended.kv, queries, attentionRows, b.attentionPartials,
                                    b.attentionStatistics, attended.table, sequence.chunk, plans[index],
                                    ops::AttentionBlocks{}, mixer.sinks);
      ops::PagedAttention::addPrefillGate(
          step.graph, u16Rows(b.fullPacked, sequence, geometry_.packedFullWidth), attentionRows,
          u16Rows(b.attentionHidden, sequence, geometry_.attentionWidth), sequence.rows, sequence.attentionStride,
        geometry_.attentionQueryHeads, geometry_.kvLayout);
      continue;
    }
    const uint32_t last = sequence.rows - 1;
    const auto tail = ops::PagedAttention::prefillTail(plans[index], last);
    const auto from = [&](const metal::MetalBuffer &buffer, uint32_t row) {
      return backend_.view(buffer, tail.offsetBytes(row), queryBytes - tail.offsetBytes(row));
    };
    ops::PagedAttention::addPrefill(step.graph, attended.kv, from(queries, tail.first),
                                   from(attentionRows, tail.first), b.attentionPartials, b.attentionStatistics,
                                   attended.table, tail.chunk(sequence.chunk), tail.plan, {}, mixer.sinks);
    ops::PagedAttention::addPrefillGate(
        step.graph, rowsOf<uint16_t>(b.fullPacked, sequence.rowBegin + last, 1, geometry_.packedFullWidth),
        from(attentionRows, last), rowsOf<uint16_t>(b.attentionHidden, index, 1, geometry_.attentionWidth),
        1, sequence.attentionStride, geometry_.attentionQueryHeads, geometry_.kvLayout);
  }
  linear.addPrefill(step.graph, b.attentionHidden, mixer.outputProjection, b.attentionOutput, step.outputRows(),
                    b.linearScratch, step.rows);
}

// The verify mixers, whose fused and value projections run per lane batch
// (decodeInput), over the normalized rows the table describes.
void RippleTarget::addVerifyProjectionPair(VerifyStep &step, const QwenAttentionWeights &mixer,
                                           const metal::MetalBuffer &lambda, ops::PreparedInput normalized) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  addDecodeProjection(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, step.lanes,
                      b.linearScratch, std::move(normalized), step.rows);
  ops::ValueMix::add(step.graph, b.normalized, embedRows_, lambda, b.denseIntermediate, step.rows,
                     geometry_.hiddenSize);
  addDecodeProjection(step.graph, b.denseIntermediate, mixer.valueProjection, b.fullPacked, step.lanes,
                      b.linearScratch, {}, step.rows);
}

void RippleTarget::addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer,
                                  const metal::MetalBuffer &lambda, ops::PreparedInput normalized,
                                  BlockOutput output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const uint32_t layer = step.attentionLayer++;
  const ops::AttentionVariant variant = geometry_.attentionVariant(layer);
  addVerifyProjectionPair(step, mixer, lambda, std::move(normalized));
  const uint32_t rows = step.rows / step.lanes;
  const uint32_t stride = kv::kVerifyChunkStride;
  const uint64_t queryBytes = uint64_t{stride} * geometry_.attentionWidth * sizeof(uint16_t);
  const uint64_t kvBytes = uint64_t{stride} * geometry_.attentionKvHeads *
                           geometry_.attentionHeadDimension * sizeof(uint16_t);
  const auto laneRows = [&](metal::MetalBuffer buffer, uint32_t lane, uint32_t width, uint32_t bytes) {
    const uint64_t extent = uint64_t{rows} * width * bytes;
    return backend_.view(buffer, lane * extent, extent);
  };
  for (uint32_t lane = 0; lane < step.lanes; ++lane) {
    const auto packed = laneRows(b.fullPacked, lane, geometry_.packedFullWidth, 2);
    const auto queries = backend_.view(b.fullQueries, lane * queryBytes, queryBytes);
    const auto attended = backend_.view(b.fullAttention, lane * queryBytes, queryBytes);
    const auto keys = backend_.view(b.chunkKeys[layer], lane * kvBytes, kvBytes);
    const auto values = backend_.view(b.chunkValues[layer], lane * kvBytes, kvBytes);
    ops::PagedAttention::addPrefillProjection(
        step.graph, packed, mixer.queryNorm, mixer.keyNorm,
        laneRows(b.ropeCos, lane, geometry_.rotaryPairs, 4),
        laneRows(b.ropeSin, lane, geometry_.rotaryPairs, 4), queries, keys, values,
        rows, stride, geometry_.attentionQueryHeads, geometry_.kvLayout, variant);
    ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[layer], keys, values,
                                        b.pageTables[lane], step.chunks[lane], geometry_.kvLayout);
    const auto plan = operators_.prefillAttention(rows, step.chunks[lane].committed_tokens,
                                                 geometry_.attentionQueryHeads, geometry_.kvLayout, variant);
    ops::PagedAttention::addPrefill(step.graph, step.kvLayers[layer], queries, attended,
                                   b.attentionPartials, b.attentionStatistics, b.pageTables[lane],
                                   step.chunks[lane], plan, {}, mixer.sinks);
    ops::PagedAttention::addPrefillGate(step.graph, packed, attended,
                                       laneRows(b.attentionHidden, lane, geometry_.attentionWidth, 2),
                                       rows, stride, geometry_.attentionQueryHeads, geometry_.kvLayout);
  }
  const ops::Linear &linear = operators_.linear();
  linear.add(step.graph, {.input = b.attentionHidden, .output = std::move(output.rows),
                          .scratch = b.linearScratch}, mixer.outputProjection,
             linear.decodePlan(mixer.outputProjection, step.lanes, ops::LinearEpilogue::None, nullptr, step.rows));
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
  ops::Normalization::addRms(step.graph, ffnIn, layer.ffnNorm, b.normalized, geometry_.hiddenSize, step.outputRows());
  const auto prepared = linear.add(step.graph,
      {.input = b.normalized, .output = b.denseGateScratch, .scratch = b.linearScratch},
      layer.ffn.gate, linear.prefillPlan(layer.ffn.gate, step.outputRows(), ops::LinearEpilogue::None, step.rows));
  linear.add(step.graph,
             {.input = b.normalized, .output = b.denseIntermediate, .gateScratch = b.denseGateScratch,
              .scratch = b.linearScratch, .prepared = prepared, .clamp = geometry_.swigluClamp},
             layer.ffn.up, linear.prefillPlan(layer.ffn.up, step.outputRows(),
                                             ops::LinearEpilogue::ClampedUpWithGate, step.rows));
  linear.addPrefill(step.graph, b.denseIntermediate, layer.ffn.down, output, step.outputRows(), b.linearScratch, step.rows);
  ops::Normalization::addRms(step.graph, output, layer.postFfnNorm, b.normalized, geometry_.hiddenSize, step.outputRows());
  ops::ValueMix::sum(step.graph, b.normalized, ffnIn, output, step.outputRows(), geometry_.hiddenSize);
}

void RippleTarget::addVerifyFfn(VerifyStep &step, const RippleLayerWeights &layer, metal::MetalBuffer ffnIn,
                                metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan gateUpPlan =
      linear.decodePlan(layer.ffn.up, step.lanes, ops::LinearEpilogue::ClampedGateUp, &layer.ffn.gate, step.rows);
  const ops::PreparedInput normalized =
      ops::Normalization::addRms(step.graph, ffnIn, layer.ffnNorm, b.normalized, geometry_.hiddenSize, step.rows,
                                 b.linearScratch, gateUpPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.denseIntermediate, .gateScratch = b.denseGateScratch,
              .scratch = b.linearScratch, .prepared = normalized, .clamp = geometry_.swigluClamp},
             layer.ffn.up, gateUpPlan, &layer.ffn.gate);
  linear.add(step.graph, {.input = b.denseIntermediate, .output = output, .scratch = b.linearScratch},
             layer.ffn.down, linear.decodePlan(layer.ffn.down, step.lanes, ops::LinearEpilogue::None, nullptr, step.rows));
  ops::Normalization::addRms(step.graph, output, layer.postFfnNorm, b.normalized, geometry_.hiddenSize, step.rows);
  ops::ValueMix::sum(step.graph, b.normalized, ffnIn, output, step.rows, geometry_.hiddenSize);
}

// The geometry of a Ripple target's projections: its dense FFNs' gate and up
// run as one gate/up plan each, and its layers keep no MoE blocks.
void includeFamilyProjections(QwenTargetGeometry &geometry, const RippleWeights &weights) {
  geometry.verifyRows = SPLASH_DRAFT_BLOCK_ROWS_LIMIT;
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
  // A verify block's immutable scaled embeddings; prefill keeps each
  // chunk's in its own arena slice while the chunks advance together.
  embedRows_ = backend.allocateBuffer(uint64_t{ExecutionLimits::maximumBatchWidth} * geometry_.verifyRows *
                                      geometry_.hiddenSize * 2,
                                      metal::BufferStorage::Shared, "ripple embeddings");
}

RippleTarget::~RippleTarget() = default;

QwenTargetScratchBytes RippleTarget::allocatedScratchBytes() const noexcept {
  return {0, embedRows_.sizeBytes()};
}

void RippleTarget::addEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                                metal::MetalBuffer hidden, uint32_t rows) const {
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, hidden, rows,
                      geometry_.embeddingScale);
}

void RippleTarget::addPrefillBegin(PrefillStep &step) const {
  ops::RowCopy::add(step.graph, step.buffers.hidden[0], {0, geometry_.hiddenSize, 0},
                    step.buffers.embeddings, {0, geometry_.hiddenSize, 0}, step.rows, geometry_.hiddenSize);
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
  planOutputRows(step, index);
  ops::Normalization::addRms(step.graph, input, layer.inputNorm, b.normalized, geometry_.hiddenSize, step.rows);
  addPrefillMixer(step, std::get<QwenAttentionWeights>(layer.mixer), layer.valueResidualLambda);
  metal::MetalBuffer residual = input;
  if (!step.readRows.empty()) {
    residual = output;
    for (uint32_t row = 0; row < step.readRows.size(); ++row)
      ops::RowCopy::add(step.graph, input, {step.readRows[row], geometry_.hiddenSize, 0},
                        residual, {row, geometry_.hiddenSize, 0}, 1, geometry_.hiddenSize);
  }
  ops::Normalization::addRms(step.graph, b.attentionOutput, layer.postAttentionNorm, b.normalized,
                             geometry_.hiddenSize, step.outputRows());
  ops::ValueMix::sum(step.graph, b.normalized, residual, b.attentionOutput, step.outputRows(), geometry_.hiddenSize);
  addPrefillFfn(step, layer, b.attentionOutput, step.readRows.empty() ? output : input);
}

const ops::BlockMoeWeights *RippleTarget::moe(uint32_t) const noexcept { return nullptr; }

void RippleTarget::addPrefillExperts(PrefillStep &step, uint32_t index) const {
  addReadRows(step, step.buffers.hidden[index & 1], step.buffers.hidden[(index & 1) ^ 1]);
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
  if (blockRows > geometry_.verifyRows || chunks.front().row_base)
    throw std::invalid_argument("Ripple verify requires one complete target traversal");
  step.rows = lanes * blockRows;
  ops::RowCopy::add(graph, buffers.hidden[0], {0, geometry_.hiddenSize, 0}, embedRows_,
                    {0, geometry_.hiddenSize, 0}, step.rows, geometry_.hiddenSize);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const RippleLayerWeights &layer = weights_.layers[index];
    const metal::MetalBuffer input = buffers.hidden[index & 1], output = buffers.hidden[(index & 1) ^ 1];
    const ops::PreparedInput normalized = ops::Normalization::addRms(
        step.graph, input, layer.inputNorm, buffers.normalized, geometry_.hiddenSize, step.rows,
        buffers.linearScratch, decodeInput(std::get<QwenAttentionWeights>(layer.mixer).inputProjection,
                                           step.lanes, step.rows));
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
                 0, step.rows);
  }
  requireLayerPartition(step.gdnLayer, step.attentionLayer);
  addHeadBatch(step.graph, buffers.hidden[geometry_.layers & 1], buffers.finalHidden,
               buffers.logits, step.lanes, step.rows, buffers.linearScratch);
}

void RippleTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                                metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
                                uint32_t rows, ops::LinearScratch scratch) const {
  addNormalizedHead(graph, weights_.finalNorm, std::move(hidden), std::move(finalHidden), std::move(logits),
                    lanes, rows, scratch);
}

QwenTargetScratchBytes qwenTargetScratchBytes(const RippleWeights &, const QwenTargetGeometry &geometry,
                                              uint32_t, uint64_t) {
  // The verify batch's immutable scaled embeddings; prefill's belong to
  // its arena so grouped chunks keep disjoint rows.
  return {0, uint64_t{ExecutionLimits::maximumBatchWidth} * geometry.verifyRows * geometry.hiddenSize * 2};
}

std::unique_ptr<QwenTarget> makeQwenTarget(const RippleWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t) {
  return std::make_unique<RippleTarget>(weights, geometry, backend, operators);
}

} // namespace splash::model
