#pragma once

#include "TestAdmission.hpp"
#include "TestChecks.hpp"
#include "engine/Engine.hpp"
#include "model/WeightMemory.hpp"
#include "ops/PagedKv.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace splash::test {

// An engine config completed as the bootstrap completes one, for tests: what
// the test set, and otherwise the whole logical context, every 32-bit token
// in the vocabulary and a host that never pauses growth.
[[nodiscard]] inline engine::EngineConfig engineConfig(engine::EngineConfig config = {}) {
  if (!config.maxContext)
    config.maxContext = kv::kMaximumLogicalTokens;
  if (!config.vocabularySize)
    config.vocabularySize = std::numeric_limits<uint32_t>::max();
  if (!config.growthPaused)
    config.growthPaused = [] { return false; };
  if (!config.serving)
    config.serving = [](bool) {};
  return config;
}

// The weights of a native loop under test: three images of a gibibyte that
// count what the loop does with them, each restored under the admission the
// loop passes.
class Weights final : public model::WeightMemory {
public:
  static constexpr uint32_t kImages = 3;
  static constexpr uint64_t kImageBytes = 1ULL << 30;
  [[nodiscard]] bool released() const noexcept override {
    return restored_ < kImages;
  }
  void release() override { restored_ = 0; }
  bool restore(const metal::AllocationAdmission &admit) override {
    if (failRestore)
      throw std::runtime_error("weights restore test");
    require(released(), "held weights were restored");
    if (const metal::AllocationResult admitted = admit(kImageBytes, [] {}); !admitted)
      throw metal::MetalAllocationError("weights restore test refused", admitted.failure);
    ++restores;
    return ++restored_ == kImages;
  }
  // The images in memory.
  [[nodiscard]] uint32_t held() const noexcept { return restored_; }
  bool failRestore = false;
  // The images written back.
  uint32_t restores = 0;

private:
  uint32_t restored_ = kImages;
};

} // namespace splash::test
