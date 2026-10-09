#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <functional>

namespace splash::test {

// Admits every allocation (metal::AllocationAdmission), as nothing governs a
// test's memory.
inline metal::AllocationResult admitAll(uint64_t, const std::function<void()> &allocate) {
  allocate();
  return {};
}

} // namespace splash::test
