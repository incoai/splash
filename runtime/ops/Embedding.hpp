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
  // Whether add gathers a table from the planes of a projection of this
  // format (EmbeddingWeights' planes): the linear formats of four bits or
  // more and Q8_0, whose plane values decode as their native rows' do.
  [[nodiscard]] static bool gathersPlanes(uint32_t formatId) noexcept;
  // Each of `lanes` lanes' mask rows of its SPLASH_DRAFT_QUERY_ROWS draft
  // input rows of `width` values, all but row 0, its anchor's, set to the bf16
  // row `mask`.
  static void addDraftMaskRows(metal::CommandGraph &graph, metal::MetalBuffer mask,
                               metal::MetalBuffer rows, uint32_t width, uint32_t lanes);
};

} // namespace splash::ops
