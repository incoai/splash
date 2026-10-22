#pragma once

// The control vector on a Qwen3.5-family target's bf16 residual
// (kernels/shared/residual_steer.metal, model/ControlVector.hpp): after each
// steered layer's FFN, before the DFlash2 capture.

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::ops {

// A control vector's change after a steered layer (addResidualSteer): row
// `layer` of its table of directions, added `scale` times, or, with project,
// the residual's component along that unit direction taken off `scale`
// times.
struct ResidualSteer final {
  uint32_t layer = 0;
  bool project = false;
  float scale = 1.0F;
};

// Steers `rows` rows of `residual`, rows of `hiddenSize` bf16 values, by row
// `steer.layer` of `directions`, a table of fp32 rows of `hiddenSize` values
// the target owns (model/ControlVector.hpp). The rows past `rows` are not
// read or written.
void addResidualSteer(metal::CommandGraph &graph, metal::MetalBuffer residual, metal::MetalBuffer directions,
                      ResidualSteer steer, uint32_t hiddenSize, uint32_t rows);

}  // namespace splash::ops
