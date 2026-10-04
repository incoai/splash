#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/RoPE.h"

#include <cstdint>

namespace splash::ops {

class RoPE final {
public:
  static void addTables(
      metal::CommandGraph &graph, metal::MetalBuffer targetPositions,
      metal::MetalBuffer draftPositions,
      metal::MetalBuffer targetInverseFrequencies,
      metal::MetalBuffer draftInverseFrequencies,
      metal::MetalBuffer targetCosine, metal::MetalBuffer targetSine,
      metal::MetalBuffer draftCosine, metal::MetalBuffer draftSine,
      RopeTableParams rows, uint32_t maximumRows);
};

} // namespace splash::ops
