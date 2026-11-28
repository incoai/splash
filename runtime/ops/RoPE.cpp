#include "RoPE.hpp"

#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {

namespace {

// The target table kernel `name` of a target of `pairs` rotary pairs a row
// (kernels/shared/rope.metal).
std::string tableKernel(std::string name, uint32_t pairs) {
  if (pairs == SPLASH_TARGET_ROPE_PAIRS) return name;
  if (pairs == 64) return name + "_p64";
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
  // A target's pairs choose the kernel of its rows; draft rows alone, of a
  // target without RoPE among them, take the default kernel's.
  const std::string kernel = tableKernel("rope_build_tables", rows.target_rows ? targetPairs : SPLASH_TARGET_ROPE_PAIRS);
  if ((!rows.target_rows && !rows.draft_rows) || rows.target_rows > maximumRows ||
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
  requireBytes(targetInverseFrequencies, rows.target_rows ? targetPairs * sizeof(float) : 0,
               "target inverse frequency");
  requireBytes(draftInverseFrequencies, rows.draft_rows ? SPLASH_DRAFT_ROPE_PAIRS * sizeof(float) : 0,
               "draft inverse frequency");
  requireBytes(targetCosine, targetTable, "target RoPE cosine");
  requireBytes(targetSine, targetTable, "target RoPE sine");
  requireBytes(draftCosine, draftTable, "draft RoPE cosine");
  requireBytes(draftSine, draftTable, "draft RoPE sine");
  // Without target rows the kernel reads no target buffer: the draft's
  // stand in their places.
  if (!rows.target_rows) {
    targetPositions = draftPositions;
    targetInverseFrequencies = draftInverseFrequencies;
    targetCosine = draftCosine;
    targetSine = draftSine;
  }
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

void RoPE::addTreeTables(metal::CommandGraph &graph, metal::MetalBuffer layout, metal::MetalBuffer tree,
                         std::span<const uint32_t> anchors, metal::MetalBuffer targetInverseFrequencies,
                         metal::MetalBuffer targetCosine, metal::MetalBuffer targetSine, const StepRows &step,
                         float scale, uint32_t targetPairs) {
  const std::string kernel = tableKernel("rope_build_tree_tables", targetPairs);
  requireStepRows(step);
  if (anchors.size() != step.lanes)
    throw std::invalid_argument("a decode step's RoPE rows take an anchor a lane");
  if (!(scale > 0.0F))
    throw std::invalid_argument("invalid target RoPE scale");
  RopeTreeParams params{.lanes = step.lanes, .rows = step.rows, .target_scale = scale, .anchors = {}};
  std::copy(anchors.begin(), anchors.end(), params.anchors);
  const uint64_t table = uint64_t{step.rows} * targetPairs * sizeof(float);
  requireBytes(layout, uint64_t{step.lanes} * sizeof(SplashLaneRows), "RoPE lane rows");
  requireBytes(tree, uint64_t{step.rows} * sizeof(SplashTreeRow), "RoPE tree rows");
  requireBytes(targetInverseFrequencies, targetPairs * sizeof(float), "target inverse frequency");
  requireBytes(targetCosine, table, "target RoPE cosine");
  requireBytes(targetSine, table, "target RoPE sine");
  graph.add(kernel,
            {std::move(layout), std::move(tree), std::move(targetInverseFrequencies), std::move(targetCosine),
             std::move(targetSine)},
            params, {step.laneCap, step.lanes, 1}, {targetPairs, 1, 1});
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
