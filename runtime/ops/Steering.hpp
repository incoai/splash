#pragma once

// The projective steer of a layer's residual rows (kernels/shared/
// steer_project.metal): h <- h - alpha (h . d) d, per row, with d the layer's
// unit-norm direction (model/GlpVector.hpp). fp32, as the residual stream is.

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::ops {

class Steering final {
public:
  // Projects `rowCount` rows of `hidden` fp32 values off `direction`, `hidden`
  // unit-norm fp32 values, at strength `alpha`.
  static void addProject(metal::CommandGraph &graph, metal::MetalBuffer rows, metal::MetalBuffer direction,
                         float alpha, uint32_t rowCount, uint32_t hidden);
};

} // namespace splash::ops
