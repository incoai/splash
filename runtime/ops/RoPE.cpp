#include "RoPE.hpp"

#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace splash::ops {

void RoPE::addTables(
    metal::CommandGraph &graph, metal::MetalBuffer targetPositions,
    metal::MetalBuffer draftPositions,
    metal::MetalBuffer targetInverseFrequencies,
    metal::MetalBuffer draftInverseFrequencies,
    metal::MetalBuffer targetCosine, metal::MetalBuffer targetSine,
    metal::MetalBuffer draftCosine, metal::MetalBuffer draftSine,
    RopeTableParams rows, uint32_t maximumRows) {
  if (!rows.target_rows || rows.target_rows > maximumRows ||
      rows.draft_rows > maximumRows || !rows.draft_pairs ||
      rows.draft_pairs > SPLASH_DRAFT4_ROPE_PAIRS) {
    throw std::invalid_argument("invalid RoPE table row count");
  }
  if (!(rows.target_scale > 0.0F))
    throw std::invalid_argument("invalid target RoPE scale");
  const uint64_t targetTable = uint64_t{rows.target_rows} * SPLASH_TARGET_ROPE_PAIRS * sizeof(float);
  const uint64_t draftTable = uint64_t{rows.draft_rows} * rows.draft_pairs * sizeof(float);
  // Target rows carry three positions (t, h, w): Qwen3.5's M-RoPE.
  requireBytes(targetPositions, uint64_t{rows.target_rows} * 3 * sizeof(uint32_t), "target RoPE position");
  requireBytes(draftPositions, uint64_t{rows.draft_rows} * sizeof(uint32_t), "draft RoPE position");
  requireBytes(targetInverseFrequencies, SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "target inverse frequency");
  requireBytes(draftInverseFrequencies, rows.draft_rows ? rows.draft_pairs * sizeof(float) : 0,
               "draft inverse frequency");
  requireBytes(targetCosine, targetTable, "target RoPE cosine");
  requireBytes(targetSine, targetTable, "target RoPE sine");
  requireBytes(draftCosine, draftTable, "draft RoPE cosine");
  requireBytes(draftSine, draftTable, "draft RoPE sine");
  const uint64_t elements =
      std::max<uint64_t>(uint64_t{rows.target_rows} * SPLASH_TARGET_ROPE_PAIRS,
                         uint64_t{rows.draft_rows} * rows.draft_pairs);
  graph.add("rope_build_tables",
            {std::move(targetPositions), std::move(draftPositions),
             std::move(targetInverseFrequencies),
             std::move(draftInverseFrequencies), std::move(targetCosine),
             std::move(targetSine), std::move(draftCosine),
             std::move(draftSine)},
            rows, {(elements + 255) / 256, 1, 1}, {256, 1, 1});
}

void RoPE::addDraftRangeTables(metal::CommandGraph &graph,
                               metal::MetalBuffer draftInverseFrequencies,
                               metal::MetalBuffer draftCosine,
                               metal::MetalBuffer draftSine, uint32_t rows,
                                  uint32_t startPosition, uint32_t maximumRows,
                                  uint32_t draftPairs) {
  if (!rows || rows > maximumRows || !draftPairs ||
      draftPairs > SPLASH_DRAFT4_ROPE_PAIRS)
    throw std::invalid_argument("invalid RoPE table row count");
  const uint64_t elements = uint64_t{rows} * draftPairs;
  requireBytes(draftInverseFrequencies, draftPairs * sizeof(float), "draft inverse frequency");
  requireBytes(draftCosine, elements * sizeof(float), "draft RoPE cosine");
  requireBytes(draftSine, elements * sizeof(float), "draft RoPE sine");
  graph.add("rope_build_draft_range_tables",
            {std::move(draftInverseFrequencies), std::move(draftCosine), std::move(draftSine)},
            RopeRangeParams{rows, startPosition, draftPairs},
            {(elements + 255) / 256, 1, 1}, {256, 1, 1});
}

} // namespace splash::ops
