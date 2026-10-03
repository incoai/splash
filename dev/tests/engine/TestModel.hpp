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
  return {"Qwen3.8-27B", kv::kMaximumLogicalTokens,
          kv::Layout{16, 4, 256},
          modelMemoryFootprint(targetWeightsBytes, draftWeightsBytes,
                               visionWeightsBytes)};
}

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
