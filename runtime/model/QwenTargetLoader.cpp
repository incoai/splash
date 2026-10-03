#include "model/QwenTargetLoader.hpp"
#include "Checked.hpp"

#include <utility>

namespace splash::model {

ops::Projection BlockTargetFormat::fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                         std::string_view,
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

template <class Format>
QwenMixerWeights readQwenMixer(WeightFile &file, const Format &format,
                               const QwenMixerGeometry &geometry, bool fullAttention) {
  constexpr uint64_t kFloat32Bytes = 4;
  if (fullAttention) {
    QwenAttentionWeights attention;
    attention.inputProjection =
        format.fused(file, geometry.packedFullWidth, geometry.hiddenSize, "attention-input",
                     {"attn-q", "attn-k", "attn-v"});
    attention.queryNorm = format.norm(file, geometry.attentionHeadDimension, "query-norm");
    attention.keyNorm = format.norm(file, geometry.attentionHeadDimension, "key-norm");
    attention.outputProjection =
        format.projection(file, geometry.hiddenSize, geometry.attentionWidth, "attention-output");
    return attention;
  }
  QwenGdnWeights gdn;
  gdn.inputProjection = format.fused(file, geometry.packedGdnWidth, geometry.hiddenSize,
                                     "gdn-input", {"gdn-qkv", "gdn-z", "gdn-ab"});
  gdn.convolutionWeights = file.section(
      checkedMultiply<WeightStoreError>(
          checkedMultiply<WeightStoreError>(geometry.convolutionDimension,
                                            kGdnConvolutionTaps,
                                            "convolution elements"),
          kBFloat16Bytes, "convolution bytes"),
      "gdn-convolution");
  gdn.decay = file.section(
      checkedMultiply<WeightStoreError>(geometry.gdnValueHeads, kFloat32Bytes,
                                        "GDN decay bytes"),
      "gdn-decay");
  gdn.timeBias = file.section(
      checkedMultiply<WeightStoreError>(geometry.gdnValueHeads, kBFloat16Bytes,
                                        "GDN time bias bytes"),
      "gdn-time-bias");
  gdn.mixerNorm = format.norm(file, geometry.gdnHeadDimension, "gdn-norm");
  gdn.outputProjection =
      format.projection(file, geometry.hiddenSize, geometry.attentionWidth, "gdn-output");
  gdn.outputHeadOrder = Format::gdnOutputOrder;
  return gdn;
}

template QwenMixerWeights readQwenMixer(WeightFile &, const AffineTargetFormat &,
                                        const QwenMixerGeometry &, bool);
template QwenMixerWeights readQwenMixer(WeightFile &, const BlockTargetFormat &,
                                        const QwenMixerGeometry &, bool);

} // namespace splash::model
