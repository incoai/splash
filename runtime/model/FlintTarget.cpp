#include "model/FlintTarget.hpp"

#include "model/Flint.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <utility>
#include <variant>

namespace splash::model {

FlintTarget::FlintTarget(const FlintWeights &weights, const QwenTargetGeometry &geometry,
                         metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights) {
  // Every layer attends.
  requireLayers(weights.layers.size(), weights.layers.size());
}

std::vector<metal::MetalBuffer> FlintTarget::addPrefill(metal::CommandGraph &graph,
                                                        std::span<const QwenTargetPrefillChunk> chunks,
                                                        const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                        QwenPrefillLayers layers, ops::AneFfn *aneFfn) {
  std::vector<PrefillStep> steps = prefillSteps(graph, chunks, kv, spilled, layers, aneFfn);
  return addPrefillLayers(std::span(steps), *this, layers);
}

void FlintTarget::addPrefillLayer(PrefillStep &step, uint32_t index) const {
  const FlintLayerWeights &layer = weights_.layers[index];
  const QwenTargetPrefillBuffers &b = step.buffers;
  const metal::MetalBuffer input = b.hidden[index & 1], output = b.hidden[(index & 1) ^ 1];
  ops::Normalization::addRms(step.graph, input, layer.inputNorm, b.normalized, geometry_.hiddenSize, step.rows);
  addPrefillMixer(step, layer.mixer, {b.attentionOutput, input});
  if (const auto *dense = std::get_if<FlintDenseFfnWeights>(&layer.ffn))
    addPrefillDenseFfn(step, index, layer.postAttentionNorm, dense->projections(), b.attentionOutput, output);
  else
    addPrefillMoeRouting(step, layer.postAttentionNorm, std::get<ops::BlockMoeWeights>(layer.ffn),
                         b.attentionOutput, output);
}

const ops::BlockMoeWeights *FlintTarget::moe(uint32_t index) const { return qwenMoeBlock(weights_.layers[index]); }

void FlintTarget::addPrefillExperts(PrefillStep &step, uint32_t index) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  if (const ops::BlockMoeWeights *ffn = moe(index)) ops::MoE::addRouted(step.graph, step.ffn, *ffn, *step.moe);
  forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t rows) {
    addCapture(step.graph, slot, b.hidden[(index & 1) ^ 1], source, b.captured, destination, rows);
  });
}

metal::MetalBuffer FlintTarget::addPrefillEnd(PrefillStep &step) const {
  return step.buffers.hidden[geometry_.layers & 1];
}

void FlintTarget::addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers, const kv::PageStorage &kv,
                            SpilledKvReads &spilled, std::span<const kv::ChunkedPrefillParams> chunks,
                            uint32_t lanes) const {
  VerifyStep step = verifyStep(graph, buffers, kv, spilled, chunks, lanes);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const FlintLayerWeights &layer = weights_.layers[index];
    const metal::MetalBuffer input = buffers.hidden[index & 1], output = buffers.hidden[(index & 1) ^ 1];
    const ops::PreparedInput normalized = ops::Normalization::addRms(
        step.graph, input, layer.inputNorm, buffers.normalized, geometry_.hiddenSize, step.rows,
        buffers.linearScratch, decodeInput(layer.mixer.inputProjection, step.lanes));
    addVerifyMixer(step, layer.mixer, normalized, {buffers.attentionOutput, input});
    if (const auto *dense = std::get_if<FlintDenseFfnWeights>(&layer.ffn))
      addVerifyDenseFfn(step, layer.postAttentionNorm, dense->projections(), buffers.attentionOutput, output);
    else
      addVerifyMoe(step, layer.postAttentionNorm, std::get<ops::BlockMoeWeights>(layer.ffn), buffers.attentionOutput,
                   output);
    if (const auto slot = geometry_.captureSlot(index))
      addCapture(step.graph, *slot, output, 0, buffers.capturedTargetHidden, 0, step.rows);
  }
  requireLayerPartition(step.gdnLayer, step.attentionLayer);
  addHeadBatch(step.graph, buffers.hidden[geometry_.layers & 1], buffers.finalHidden, buffers.logits, step.lanes,
               buffers.linearScratch);
}

void FlintTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                               metal::MetalBuffer logits, uint32_t lanes, ops::LinearScratch scratch) const {
  addNormalizedHead(graph, weights_.finalNorm, std::move(hidden), std::move(finalHidden), std::move(logits), lanes,
                    scratch);
}

std::unique_ptr<QwenTarget> makeQwenTarget(const FlintWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t) {
  return std::make_unique<FlintTarget>(weights, geometry, backend, operators);
}

} // namespace splash::model
