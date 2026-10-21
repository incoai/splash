#include "Qwen3_6Moe.hpp"
#include "model/QwenTargetLoader.hpp"

#include <utility>

namespace splash::model {

// Block images keep the routed experts of each projection as one tensor, the
// shared expert's as another, and the router and the shared-expert scalar
// gate in F32 (readMoeBlock).
Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend, Qwen3_6MoeLayout layout,
                      const QwenTargetFiles &files) {
  return loadQwenTarget<Qwen3_6MoeWeights>(
      backend, layout, files, [&](auto &loader, const BlockTargetFormat &format, const MoeExperts &experts) {
        return readQwenTargetWeights<Qwen3_6MoeWeights>(
            layout, loader, format,
            [&](WeightFile &file, Qwen3_6MoeLayerWeights &layer, const BlockTargetFormat &, uint32_t index) {
              layer.ffn = readMoeBlock(file, experts, index);
            });
      });
}

} // namespace splash::model
