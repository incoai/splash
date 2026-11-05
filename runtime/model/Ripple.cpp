#include "Ripple.hpp"

#include "QwenTargetLoader.hpp"

#include <stdexcept>

namespace splash::model {

// A Ripple mixer: its fused projection carries the queries, their gate and
// the keys (packedFullWidth past the value rows), and its value projection
// the value rows alone, which the mixed input projects through.
QwenMixerWeights readRippleMixer(WeightFile &file, const BlockTargetFormat &format,
                                 const QwenTargetDimensions &target, bool) {
  QwenAttentionWeights attention;
  attention.inputProjection =
      format.fused(file, target.packedFullWidth - target.attentionKvHeads * target.attentionHeadDimension,
                   target.hiddenSize, {"attn-q", "attn-k"});
  attention.valueProjection =
      format.projection(file, target.attentionWidth, target.hiddenSize, "attn-v");
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
    layer.postFfnNorm = format.norm(file, layout.hiddenSize, "post-ffn-norm");
  };
  return loadQwenTarget<RippleWeights>(
      backend, layout, files, [&](auto &loader, const BlockTargetFormat &format, const MoeExperts &) {
        return readQwenTargetWeights<RippleWeights>(layout, loader, format, readFfn, readRippleMixer);
      });
}

std::unique_ptr<QwenTarget> makeQwenTarget(const RippleWeights &, const QwenTargetGeometry &,
                                           metal::MetalBackend &, const ops::ExecutionPlans &, uint32_t) {
  throw std::invalid_argument("Ripple-65B's target is not in this build");
}

// A Ripple layer's FFN is dense: its geometry takes the gate and up of every
// layer, whose gate and up run as one gate/up plan each, and the family's
// clamp is theirs to apply.
void includeFamilyProjections(QwenTargetGeometry &geometry, const RippleWeights &weights) {
  for (const RippleLayerWeights &layer : weights.layers)
    includeDenseFfn(geometry, layer.ffn.projections());
}

const ops::BlockMoeWeights *qwenMoeBlock(const RippleLayerWeights &) noexcept { return nullptr; }

} // namespace splash::model
