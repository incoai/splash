#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/LaneRows.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

class Embedding final {
public:
  static void add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                  const EmbeddingWeights &table, metal::MetalBuffer output,
                  uint32_t rows);
  // Each of `lanes` lanes' mask rows of its SPLASH_DRAFT_QUERY_ROWS draft
  // input rows of `width` values, all but row 0, its anchor's, set to the bf16
  // row `mask`.
  static void addDraftMaskRows(metal::CommandGraph &graph, metal::MetalBuffer mask,
                               metal::MetalBuffer rows, uint32_t width, uint32_t lanes);
};

} // namespace splash::ops
