#include "model/QwenTargetLoader.hpp"
#include "Checked.hpp"

#include <utility>

namespace splash::model {

ops::Projection BlockTargetFormat::fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                         std::initializer_list<std::string_view> tensors) const {
  ops::BlockWeights weights;
  uint32_t offset = 0;
  for (std::string_view tensor : tensors) {
    ops::QuantizedSegment s = readQuantizedSegment(file, tensor);
    s.columnOffset = offset;
    offset += s.outputSize;
    weights.segments.push_back(std::move(s));
  }
  return {outputSize, inputSize, std::move(weights)};
}

QwenMixerWeights readQwenMixer(WeightFile &file, const BlockTargetFormat &format,
                               const QwenTargetDimensions &target, bool fullAttention) {
  constexpr uint64_t kFloat32Bytes = 4;
  if (fullAttention) {
    QwenAttentionWeights attention;
    attention.inputProjection =
        format.fused(file, target.packedFullWidth, target.hiddenSize, {"attn-q", "attn-k", "attn-v"});
    attention.queryNorm = format.norm(file, target.attentionHeadDimension, "query-norm");
    attention.keyNorm = format.norm(file, target.attentionHeadDimension, "key-norm");
    attention.outputProjection =
        format.projection(file, target.hiddenSize, target.attentionWidth, "attention-output");
    return attention;
  }
  QwenGdnWeights gdn;
  gdn.inputProjection = format.fused(file, target.packedGdnWidth, target.hiddenSize, {"gdn-qkv", "gdn-z", "gdn-ab"});
  gdn.convolutionWeights = file.section(
      checkedMultiply<WeightStoreError>(
          checkedMultiply<WeightStoreError>(target.convolutionDimension,
                                            kGdnConvolutionTaps,
                                            "convolution elements"),
          kBFloat16Bytes, "convolution bytes"),
      "gdn-convolution");
  gdn.decay = file.section(
      checkedMultiply<WeightStoreError>(target.gdnValueHeads, kFloat32Bytes,
                                        "GDN decay bytes"),
      "gdn-decay");
  gdn.timeBias = file.section(
      checkedMultiply<WeightStoreError>(target.gdnValueHeads, kBFloat16Bytes,
                                        "GDN time bias bytes"),
      "gdn-time-bias");
  gdn.mixerNorm = format.norm(file, target.gdnHeadDimension, "gdn-norm");
  gdn.outputProjection =
      format.projection(file, target.hiddenSize, target.attentionWidth, "gdn-output");
  gdn.outputHeadOrder = format.gdnOutputOrder;
  return gdn;
}

MoeExperts openMoeExperts(metal::MetalBackend &backend, const QwenTargetDimensions &target) {
  return {residentExpertSlots(backend, target.experts), target.experts};
}

ops::BlockMoeWeights readMoeBlock(WeightFile &file, const MoeExperts &experts) {
  ops::BlockMoeWeights ffn;
  ffn.router = readQuantizedSegment(file, "router");
  ffn.gate.routed = readQuantizedSegment(file, "experts-gate");
  ffn.up.routed = readQuantizedSegment(file, "experts-up");
  ffn.down.routed = readQuantizedSegment(file, "experts-down");
  ffn.slotTable = experts.resident;
  ffn.slots = experts.experts;
  ffn.gate.shared = readQuantizedSegment(file, "shared-expert-gate");
  ffn.up.shared = readQuantizedSegment(file, "shared-expert-up");
  ffn.down.shared = readQuantizedSegment(file, "shared-expert-down");
  ffn.sharedScalarGate = readQuantizedSegment(file, "shared-expert-scalar-gate");
  return ffn;
}

} // namespace splash::model
