#include "Ripple.hpp"

#include "QwenTargetLoader.hpp"
#include "ops/AneFfn.hpp"
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
                                   const metal::MetalBuffer &lambda, ops::PreparedInput normalized) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  linear.add(step.graph,
             {.input = b.normalized, .output = b.fullPacked, .scratch = b.linearScratch,
              .prepared = std::move(normalized)},
             mixer.inputProjection, linear.plan(mixer.inputProjection, step.rows));
  ops::ValueMix::add(step.graph, b.normalized, b.embeddings, lambda, b.denseIntermediate, step.rows,
                     geometry_.hiddenSize);
  linear.add(step.graph, {.input = b.denseIntermediate, .output = b.fullPacked, .scratch = b.linearScratch},
             mixer.valueProjection, linear.plan(mixer.valueProjection, step.rows));
  addPrefillAttention(step, mixer, {b.attentionOutput, {}});
}

// A block's dense SwiGLU, whose gate and up activations clamp at the
// family's threshold, and the norms and adds of the post-norm arrangement
// around it: the FFN reads the norm of the residual the post-attention norm
// made, and its output's norm adds back to that residual, which the
// attention's norm and add made.
void RippleTarget::addPrefillFfn(PrefillStep &step, uint32_t index, const RippleLayerWeights &layer, metal::MetalBuffer ffnIn,
                                 metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const ops::SwiGluProjections ffn{&layer.ffn.gate, &layer.ffn.up, &layer.ffn.down, geometry_.swigluClamp};
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, ffnIn, layer.ffnNorm, b.normalized, geometry_.hiddenSize, step.outputRows(), b.linearScratch,
      step.aneFfn ? ops::LinearInput::Plain : linear.swiGluInput(ffn, step.outputRows(), step.rows));
  if (step.aneFfn && step.readRows.empty())
    step.aneFfn->add(step.graph, index, b.ffn(), {}, output, step.rows);
  else
    linear.addSwiGlu(step.graph, ffn, b.ffn(), {}, output, step.outputRows(), step.rows, normalized);
  ops::Normalization::addRms(step.graph, output, layer.postFfnNorm, b.normalized, geometry_.hiddenSize, step.outputRows());
  ops::ValueMix::sum(step.graph, b.normalized, ffnIn, output, step.outputRows(), geometry_.hiddenSize);
}

std::optional<ops::SwiGluProjections> denseFfn(const RippleLayerWeights &layer) noexcept {
  return layer.ffn.projections();
}

const ops::BlockMoeWeights *qwenMoeBlock(const RippleLayerWeights &) noexcept { return nullptr; }

RippleTarget::RippleTarget(const RippleWeights &weights, const QwenTargetGeometry &geometry,
                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights) {
  requireLayers(weights.layers.size(), weights.layers.size());
  // The family's SwiGLU clamp, one F32 the clamped FFN epilogues read.
  if (!(geometry_.swigluClamp > 0.0F))
    throw std::invalid_argument("Ripple's SwiGLU clamp is not a positive threshold");
}

RippleTarget::~RippleTarget() = default;

void RippleTarget::addPrefillBegin(PrefillStep &step) const {
  ops::RowCopy::add(step.graph, step.buffers.hidden[0], {0, geometry_.hiddenSize, 0},
                    step.buffers.embeddings, {0, geometry_.hiddenSize, 0}, step.rows, geometry_.hiddenSize);
}

std::vector<metal::MetalBuffer> RippleTarget::addPrefill(metal::CommandGraph &graph,
                                                         std::span<const QwenTargetPrefillChunk> chunks,
                                                         const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                         QwenPrefillLayers layers, ops::AneFfn *aneFfn) {
  std::vector<PrefillStep> steps = prefillSteps(graph, chunks, kv, spilled, layers, aneFfn);
  return addPrefillLayers(std::span(steps), *this, layers);
}

void RippleTarget::addPrefillLayer(PrefillStep &step, uint32_t index) const {
  const RippleLayerWeights &layer = weights_.layers[index];
  const QwenTargetPrefillBuffers &b = step.buffers;
  const metal::MetalBuffer input = b.hidden[index & 1], output = b.hidden[(index & 1) ^ 1];
  planOutputRows(step, index);
  const QwenAttentionWeights &mixer = std::get<QwenAttentionWeights>(layer.mixer);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, layer.inputNorm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch,
      operators_.linear().plan(mixer.inputProjection, step.rows).input());
  addPrefillMixer(step, mixer, layer.valueResidualLambda, normalized);
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
  addPrefillFfn(step, index, layer, b.attentionOutput, step.readRows.empty() ? output : input);
}

const ops::BlockMoeWeights *RippleTarget::moe(uint32_t) const noexcept { return nullptr; }

void RippleTarget::addPrefillExperts(PrefillStep &step, uint32_t index) const {
  addReadRows(step, step.buffers.hidden[index & 1], step.buffers.hidden[(index & 1) ^ 1]);
  if (step.decode) {
    if (const auto slot = geometry_.captureSlot(index))
      addCapture(step.graph, *slot, step.buffers.hidden[(index & 1) ^ 1], 0,
                 step.decode->capturedTargetHidden, 0, step.rows);
    return;
  }
  forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t rows) {
    addCapture(step.graph, slot, step.buffers.hidden[(index & 1) ^ 1], source, step.buffers.captured,
               destination, rows);
  });
}

metal::MetalBuffer RippleTarget::addPrefillEnd(PrefillStep &step) const {
  const metal::MetalBuffer hidden = step.buffers.hidden[geometry_.layers & 1];
  if (const QwenTargetDecodeBuffers *d = step.decode)
    addNormalizedHead(step.graph, weights_.finalNorm, hidden, d->finalHidden, d->logits, step.rows,
                      step.buffers.linearScratch);
  return hidden;
}

void RippleTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                                metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
                                uint32_t rows, ops::LinearScratch scratch) const {
  addNormalizedHead(graph, weights_.finalNorm, std::move(hidden), std::move(finalHidden), std::move(logits), rows,
                    scratch);
}

std::unique_ptr<QwenTarget> makeQwenTarget(const RippleWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t) {
  return std::make_unique<RippleTarget>(weights, geometry, backend, operators);
}

} // namespace splash::model
