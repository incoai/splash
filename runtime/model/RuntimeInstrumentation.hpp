#pragma once

#include "model/Runtime.hpp"

#include <cstdint>
#include <vector>

namespace splash::model {

// What tests impose on a Runtime beyond its serving API. Only the
// instrumented build of Runtime.cpp (SPLASH_RUNTIME_INSTRUMENTATION) defines
// these; production binaries never link it.
class RuntimeInstrumentation final {
public:
  // The PLE rows the last decode step gathered for batch lane `lane`'s
  // verify rows, each row's heads' in order (Qwen3.8-Flash-Next), until the
  // next decode step; none for a target without them.
  [[nodiscard]] static std::vector<uint32_t> lastPleRows(const Runtime &runtime, uint32_t lane);
};

} // namespace splash::model
