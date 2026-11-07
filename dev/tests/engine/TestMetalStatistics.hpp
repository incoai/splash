#pragma once

#include "metal/MetalBackend.hpp"

namespace splash::test {

// The memory statistics the stand-in backend (TestMetalMemory.hpp) reports.
[[nodiscard]] inline metal::MetalMemoryStats &metalStatistics() noexcept {
  static metal::MetalMemoryStats statistics;
  return statistics;
}

} // namespace splash::test
