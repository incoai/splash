#pragma once

#include "metal/MetalBackend.hpp"
#include "model/WeightMemory.hpp"

#include <cstdint>
#include <functional>
#include <thread>

namespace splash::test {

// Admits every allocation (metal::AllocationAdmission), as nothing governs a
// test's memory.
inline metal::AllocationResult admitAll(uint64_t, const std::function<void()> &allocate) {
  allocate();
  return {};
}

// Waits while a part of `weights` is written on its thread
// (model::WeightMemory::writing).
inline void awaitWritten(const model::WeightMemory &weights) {
  while (weights.writing())
    std::this_thread::yield();
}

// Takes every part of `weights` back under `admit`, as the engine does: each
// restore() once the part before it is written.
inline void restoreWeights(model::WeightMemory &weights, const metal::AllocationAdmission &admit = admitAll) {
  for (awaitWritten(weights); !weights.restore(admit); awaitWritten(weights)) {
  }
}

} // namespace splash::test
