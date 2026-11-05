#include "Ripple.hpp"

#include <stdexcept>

namespace splash::model {

RippleWeights loadRippleWeights(metal::MetalBackend &, RippleLayout layout, const QwenTargetFiles &) {
  throw std::invalid_argument(std::string(layout.family) + "'s weight load is not in this build");
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
