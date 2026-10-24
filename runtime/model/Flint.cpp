#include "model/Flint.hpp"

#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"

namespace splash::model {

// A layer image holds its input norm, its attention (readQwenAttention), the
// attention's sinks, the norm before its FFN and the FFN: a leading layer's
// dense gate, up and down, another's MoE block (readMoeBlock), whose blocks
// are the MoE layers' in order (GgufImage.cpp FlintGguf).
FlintWeights loadFlintWeights(metal::MetalBackend &backend, FlintLayout layout, const QwenTargetFiles &files) {
  return loadQwenTarget<FlintWeights>(
      backend, layout, files, [&](GgufTargetLoader &gguf, const BlockTargetFormat &format, const MoeExperts &experts) {
        FlintWeights result;
        result.layout = layout;
        result.layers.reserve(layout.layers);
        for (uint32_t index = 0; index < layout.layers; ++index) {
          WeightFile file = gguf.layer(index);
          FlintLayerWeights &layer = result.layers.emplace_back();
          layer.inputNorm = format.norm(file, layout.hiddenSize, "input-norm");
          layer.mixer = readQwenAttention(file, format, layout);
          layer.mixer.sinks = file.section(uint64_t{layout.attentionQueryHeads} * kFloat32Bytes, "attention-sinks");
          layer.postAttentionNorm = format.norm(file, layout.hiddenSize, "post-attention-norm");
          if (layout.isMoeLayer(index)) {
            layer.ffn = readMoeBlock(file, layout, experts, layout.moeBlock(index));
          } else {
            layer.ffn = FlintDenseFfnWeights{
                format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate"),
                format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up"),
                format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down")};
          }
          file.finish();
          result.files.push_back(file.record());
        }
        readQwenTargetHead(result, layout, gguf, format);
        return result;
      });
}

const ops::BlockMoeWeights *qwenMoeBlock(const FlintLayerWeights &layer) noexcept {
  return std::get_if<ops::BlockMoeWeights>(&layer.ffn);
}

void includeFamilyProjections(QwenTargetGeometry &geometry, const FlintWeights &weights) {
  for (const FlintLayerWeights &layer : weights.layers)
    if (const auto *dense = std::get_if<FlintDenseFfnWeights>(&layer.ffn))
      includeDenseFfn(geometry, dense->projections());
}

} // namespace splash::model
