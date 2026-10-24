#include "RoPE.hpp"

#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {

namespace {

// The table kernel of a target of `pairs` rotary pairs a row
// (kernels/shared/rope.metal).
std::string tableKernel(uint32_t pairs) {
  if (pairs == SPLASH_TARGET_ROPE_PAIRS) return "rope_build_tables";
  if (pairs == 64) return "rope_build_tables_p64";
  throw std::invalid_argument("no RoPE table kernel for the target's rotary pairs");
}

} // namespace

void RoPE::addTables(
    metal::CommandGraph &graph, metal::MetalBuffer targetPositions,
    metal::MetalBuffer draftPositions,
    metal::MetalBuffer targetInverseFrequencies,
    metal::MetalBuffer draftInverseFrequencies,
    metal::MetalBuffer targetCosine, metal::MetalBuffer targetSine,
    metal::MetalBuffer draftCosine, metal::MetalBuffer draftSine,
    RopeTableParams rows, uint32_t maximumRows, uint32_t targetPairs) {
  const std::string kernel = tableKernel(targetPairs);
  if (!rows.target_rows || rows.target_rows > maximumRows ||
      rows.draft_rows > maximumRows) {
    throw std::invalid_argument("invalid RoPE table row count");
  }
  if (!(rows.target_scale > 0.0F))
    throw std::invalid_argument("invalid target RoPE scale");
  const uint64_t targetTable = uint64_t{rows.target_rows} * targetPairs * sizeof(float);
  const uint64_t draftTable = uint64_t{rows.draft_rows} * SPLASH_DRAFT_ROPE_PAIRS * sizeof(float);
  // Target rows carry three positions (t, h, w): Qwen3.5's M-RoPE.
  requireBytes(targetPositions, uint64_t{rows.target_rows} * 3 * sizeof(uint32_t), "target RoPE position");
  requireBytes(draftPositions, uint64_t{rows.draft_rows} * sizeof(uint32_t), "draft RoPE position");
  requireBytes(targetInverseFrequencies, targetPairs * sizeof(float), "target inverse frequency");
  requireBytes(draftInverseFrequencies, rows.draft_rows ? SPLASH_DRAFT_ROPE_PAIRS * sizeof(float) : 0,
               "draft inverse frequency");
  requireBytes(targetCosine, targetTable, "target RoPE cosine");
  requireBytes(targetSine, targetTable, "target RoPE sine");
  requireBytes(draftCosine, draftTable, "draft RoPE cosine");
  requireBytes(draftSine, draftTable, "draft RoPE sine");
  const uint64_t elements =
      std::max<uint64_t>(uint64_t{rows.target_rows} * targetPairs,
                         uint64_t{rows.draft_rows} * SPLASH_DRAFT_ROPE_PAIRS);
  graph.add(kernel,
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
                               uint32_t startPosition, uint32_t maximumRows) {
  if (!rows || rows > maximumRows)
    throw std::invalid_argument("invalid RoPE table row count");
  const uint64_t elements = uint64_t{rows} * SPLASH_DRAFT_ROPE_PAIRS;
  requireBytes(draftInverseFrequencies, SPLASH_DRAFT_ROPE_PAIRS * sizeof(float), "draft inverse frequency");
  requireBytes(draftCosine, elements * sizeof(float), "draft RoPE cosine");
  requireBytes(draftSine, elements * sizeof(float), "draft RoPE sine");
  graph.add("rope_build_draft_range_tables",
            {std::move(draftInverseFrequencies), std::move(draftCosine), std::move(draftSine)},
            RopeRangeParams{rows, startPosition}, {(elements + 255) / 256, 1, 1}, {256, 1, 1});
}

} // namespace splash::ops
