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
  // The draft tables addTables writes, for `rows` rows of the consecutive
  // positions from startPosition on.
  static void addDraftRangeTables(metal::CommandGraph &graph,
                                  metal::MetalBuffer draftInverseFrequencies,
                                  metal::MetalBuffer draftCosine,
                                  metal::MetalBuffer draftSine, uint32_t rows,
                                  uint32_t startPosition, uint32_t maximumRows);
};

} // namespace splash::ops
