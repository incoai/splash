#pragma once

#include "engine/MemoryPlan.hpp"

#include <stdexcept>
#include <utility>

namespace splash::test {

// Synthetic nonzero allocations for planner/status tests. Production obtains
// these values from the loaded model and plannedRuntimeMemory().
[[nodiscard]] inline engine::ModelMemoryFootprint
modelMemoryFootprint(uint64_t targetWeightsBytes,
                     uint64_t draftWeightsBytes,
                     uint64_t visionWeightsBytes) {
  return {targetWeightsBytes, draftWeightsBytes, visionWeightsBytes,
          {350'224'384, 734'396'416, 160'669'696}, 0};
}

[[nodiscard]] inline engine::ModelMemoryProfile
modelMemoryProfile(uint64_t targetWeightsBytes,
                   uint64_t draftWeightsBytes,
                   uint64_t visionWeightsBytes) {
  // Qwen3.8-27B's native window.
  return {"Qwen3.8-27B", 262'144,
          kv::Layout{16, 4, 256},
          modelMemoryFootprint(targetWeightsBytes, draftWeightsBytes,
                               visionWeightsBytes)};
}

// A block-sparse target's KV, Qwen3.8-Flash-Next's: 12 attention layers of 2
// KV heads of 256 in int8, a 128-wide block index and token-major values, so
// its pages can spill (kv::Layout::spillable); each verify row attends at
// most 512 blocks of 4 tokens, and rows with up to 2,048 tokens of history
// read their page table.
[[nodiscard]] constexpr kv::Layout sparseKvLayout() noexcept {
  return {12, 2, 256, kv::Format::Int8, 128, true};
}
[[nodiscard]] constexpr model::SparseKvShape sparseKvShape() noexcept {
  return {.blockTokens = 4, .selectedBlocks = 512, .verifyRows = 8, .denseHistoryTokens = 2048,
          .sinkTokens = 0, .localTokens = 0};
}
static_assert(sparseKvLayout().spillable());

// The plan for a profile the test expects to fit; throws with the planner's
// verdict otherwise.
[[nodiscard]] inline engine::EngineMemoryPlan
requireMemoryPlan(const DeviceCapabilities &device,
                  const engine::ModelMemoryProfile &model,
                  uint64_t maximumMemoryBytes = 0) {
  engine::EngineMemoryPlanResult result =
      engine::evaluateEngineMemoryPlan(device, model, maximumMemoryBytes);
  if (!result.plan)
    throw std::runtime_error(result.status.describe());
  return std::move(*result.plan);
}

} // namespace splash::test
