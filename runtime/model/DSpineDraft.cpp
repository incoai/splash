#include "model/DSpineDraft.hpp"

namespace splash::model {

void DSpineDraft::addDecode(metal::CommandGraph &graph, const DFlashDraftWeights &weights,
                            const ops::ExecutionPlans &operators, DFlashDecodeBuffers b,
                            const ops::Projection &head, std::span<const uint32_t> cacheLengths) {
  const auto &layout = weights.layout;
  if (!layout.isDSpine() || weights.bridge.read.size() != layout.layers ||
      weights.bridge.write.size() != layout.layers)
    throw std::invalid_argument("DSpine bridge layers do not match the checkpoint layout");
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  const uint32_t rows = lanes * operators.geometry().rows();
  const DSpineParams shape{layout.hiddenSize, layout.bridgeCodeSize, layout.bridgeRank, rows};
  const auto attention = operators.draftAttention(layout.attentionShape(), lanes);
  const auto &linear = operators.linear();
  const auto project = [&](const metal::MetalBuffer &input, const ops::Projection &weight,
                            const metal::MetalBuffer &output) {
    linear.add(graph, {.input = input, .output = output, .scratch = b.linearScratch},
                weight, linear.decodePlan(weight, lanes));
  };
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1, next = current ^ 1;
    const auto &w = weights.layers[layer];
    ops::Normalization::addRms(graph, b.hidden[current], w.inputNorm, b.normalized, layout.hiddenSize, rows);
    project(b.normalized, w.qkvProjection, b.proposalQkv);
    ops::DraftAttention::addPrepare(graph, {b.proposalQkv, b.attention, w.queryNorm, w.keyNorm,
        b.ropeCos, b.ropeSin, b.queryKeys, b.queryValues}, attention);
    ops::DraftAttention::addDecode(graph, {b.attention, b.persistentKeys[layer], b.persistentValues[layer],
        b.queryKeys, b.queryValues}, cacheLengths, attention);
    ops::DraftAttention::addReorder(graph, b.attention, b.proposalQkv, attention);
    project(b.proposalQkv, w.outputProjection, b.projected);
    ops::DSpine::addResidual(graph, b.projected, b.hidden[current], b.residual, layout.hiddenSize, rows);
    const auto upPlan = linear.decodePlan(w.upProjection, lanes, ops::LinearEpilogue::GateUp, &w.gateProjection);
    const auto prepared = ops::Normalization::addRms(graph, b.residual, w.postAttentionNorm, b.normalized,
        layout.hiddenSize, rows, b.linearScratch, upPlan.input());
    linear.add(graph, {.input = b.normalized, .output = b.intermediate, .gateScratch = b.gateScratch,
        .scratch = b.linearScratch, .prepared = prepared}, w.upProjection, upPlan, &w.gateProjection);
    project(b.intermediate, w.downProjection, b.projected);
    ops::DSpine::addResidual(graph, b.projected, b.residual, b.hidden[next], layout.hiddenSize, rows);

    ops::DSpine::addReadInput(graph, b.hidden[next], b.normalized, b.bridge, shape, operators.geometry());
    project(b.normalized, weights.bridge.read[layer], b.bridge.codes);
    ops::DSpine::addCodes(graph, b.inputTokens, weights.bridge, b.bridge, shape, operators.geometry());
    project(b.bridge.codes, weights.bridge.packed, b.bridge.packed);
    ops::DSpine::addMessage(graph, weights.bridge, b.bridge, shape, operators.geometry());
    project(b.bridge.message, weights.bridge.write[layer], b.projected);
    ops::DSpine::addWrite(graph, b.projected, b.hidden[next], b.bridge, shape, operators.geometry());
  }
  const auto headPlan = linear.decodePlan(head, lanes);
  const auto prepared = ops::Normalization::addRms(graph, b.hidden[layout.layers & 1], weights.finalNorm,
      b.finalHidden, layout.hiddenSize, rows, b.linearScratch, headPlan.input());
  linear.add(graph, {.input = b.finalHidden, .output = b.logits, .scratch = b.linearScratch, .prepared = prepared},
              head, headPlan);
}

} // namespace splash::model
