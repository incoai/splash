#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>

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
  // A verify step's input tokens, SPLASH_TARGET_VERIFY_ROWS per lane: the
  // lane's anchor, row 0 of its draft input rows, then the draft's
  // SPLASH_DRAFT_PROPOSAL_TOKENS proposals, each clamped into the
  // vocabulary.
  static void addVerifyInput(metal::CommandGraph &graph,
                             metal::MetalBuffer draftInputTokens,
                             metal::MetalBuffer proposedTokens,
                             metal::MetalBuffer verifyInputTokens,
                             uint32_t vocabulary, uint32_t lanes);
  // Each of `lanes` lanes' mask rows of its SPLASH_TARGET_VERIFY_ROWS draft
  // input rows of `width` values, all but row 0, its anchor's, set to the bf16
  // row `mask`.
  static void addDraftMaskRows(metal::CommandGraph &graph, metal::MetalBuffer mask,
                               metal::MetalBuffer rows, uint32_t width, uint32_t lanes);
};

} // namespace splash::ops
