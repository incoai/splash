#pragma once

#include "engine/MemoryPlan.hpp"

namespace splash::test {

// Synthetic nonzero allocations for planner/status tests. Production obtains
// these values from the loaded model and plannedRuntimeMemory().
[[nodiscard]] inline engine::ModelMemoryFootprint
modelMemoryFootprint(uint64_t targetWeightsBytes,
                     uint64_t draftWeightsBytes,
                     uint64_t visionWeightsBytes) {
  return {targetWeightsBytes, draftWeightsBytes, visionWeightsBytes,
          350'224'384, 734'396'416,
          160'669'696, 256ULL * 1024 * 1024, 512ULL * 1024 * 1024};
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

} // namespace splash::test
