#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/RoPE.h"
#include "ops/LaneRows.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

class RoPE final {
public:
  // The cosine and sine tables of the target's and the draft's rows at their
  // positions; the target's scaled by rows.target_scale, of targetPairs pairs
  // a row: those its attention rotates, as many as its inverse frequencies
  // (SPLASH_TARGET_ROPE_PAIRS, or 64). Without target rows (a decode step's
  // cycle start, whose target rows its selector lays out later:
  // addTreeTables) the target's buffers may be empty, and so may they for a
  // target without RoPE (no pairs), which has no table: it then writes the
  // draft's rows alone, or nothing.
  static void addTables(
      metal::CommandGraph &graph, metal::MetalBuffer targetPositions,
      metal::MetalBuffer draftPositions,
      metal::MetalBuffer targetInverseFrequencies,
      metal::MetalBuffer draftInverseFrequencies,
      metal::MetalBuffer targetCosine, metal::MetalBuffer targetSine,
      metal::MetalBuffer draftCosine, metal::MetalBuffer draftSine,
      RopeTableParams rows, uint32_t maximumRows,
      uint32_t targetPairs = SPLASH_TARGET_ROPE_PAIRS);
  // The target's table rows of a decode step's rows (`step`), which its
  // selector lays out on the device: lane l's rows (`layout`, a
  // SplashLaneRows a lane) each at its anchor's RoPE position anchors[l]
  // plus the row's depth (`tree`, a SplashTreeRow a packed row), at its
  // packed row, as addTables makes a row at that position; rows past the
  // step's are not written. Nothing for a target without RoPE.
  static void addTreeTables(metal::CommandGraph &graph, metal::MetalBuffer layout, metal::MetalBuffer tree,
                            std::span<const uint32_t> anchors, metal::MetalBuffer targetInverseFrequencies,
                            metal::MetalBuffer targetCosine, metal::MetalBuffer targetSine, const StepRows &step,
                            float scale, uint32_t targetPairs = SPLASH_TARGET_ROPE_PAIRS);
  // The draft tables addTables writes, for `rows` rows of the consecutive
  // positions from startPosition on.
  static void addDraftRangeTables(metal::CommandGraph &graph,
                                  metal::MetalBuffer draftInverseFrequencies,
                                  metal::MetalBuffer draftCosine,
                                  metal::MetalBuffer draftSine, uint32_t rows,
                                  uint32_t startPosition, uint32_t maximumRows);
};

} // namespace splash::ops
