#include "Qwen3_6Moe.hpp"
#include "model/QwenTargetLoader.hpp"

#include <utility>

namespace splash::model {
namespace {

// Affine files keep a Q8 router and shared-expert scalar gate and one Q4 slab
// per expert projection; the shared expert is a one-expert slab.
void readFfn(WeightFile &file, Qwen3_6MoeLayerWeights &layer, const Qwen3_6MoeLayout &layout,
             const AffineTargetFormat &) {
  const uint32_t hidden = layout.hiddenSize, width = layout.expertIntermediateSize;
  layer.ffn = ops::AffineMoeWeights{
      .router = readAffineQ8Projection(file, layout.experts, hidden, "router"),
      .expertGate = readAffineExpertProjection(file, layout.experts, width, hidden, "experts-gate"),
      .expertUp = readAffineExpertProjection(file, layout.experts, width, hidden, "experts-up"),
      .expertDown = readAffineExpertProjection(file, layout.experts, hidden, width, "experts-down"),
      .sharedGate = readAffineExpertProjection(file, 1, width, hidden, "shared-expert-gate"),
      .sharedUp = readAffineExpertProjection(file, 1, width, hidden, "shared-expert-up"),
      .sharedDown = readAffineExpertProjection(file, 1, hidden, width, "shared-expert-down"),
      .sharedScalarGate =
          readAffineQ8Projection(file, kQ4StorageN, hidden, "shared-expert-scalar-gate"),
  };
}

// GGUF images keep the tensors as the GGUF stores them, the router and the
// shared-expert scalar gate in F32.
void readFfn(WeightFile &file, Qwen3_6MoeLayerWeights &layer, const Qwen3_6MoeLayout &,
             const BlockTargetFormat &) {
  ops::BlockMoeWeights ffn;
  ffn.router = readQuantizedSegment(file, "router");
  ffn.gate.routed = readQuantizedSegment(file, "experts-gate");
  ffn.up.routed = readQuantizedSegment(file, "experts-up");
  ffn.down.routed = readQuantizedSegment(file, "experts-down");
  ffn.gate.shared = readQuantizedSegment(file, "shared-expert-gate");
  ffn.up.shared = readQuantizedSegment(file, "shared-expert-up");
  ffn.down.shared = readQuantizedSegment(file, "shared-expert-down");
  ffn.sharedScalarGate = readQuantizedSegment(file, "shared-expert-scalar-gate");
  layer.ffn = std::move(ffn);
}

} // namespace

Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend, Qwen3_6MoeLayout layout,
                      const QwenTargetFiles<Qwen3_6MoeLayout> &files) {
  return loadQwenTarget<Qwen3_6MoeWeights>(
      backend, layout, files, [&](WeightFile &file, Qwen3_6MoeLayerWeights &layer, const auto &format) {
        readFfn(file, layer, layout, format);
      });
}

} // namespace splash::model
