#pragma once

#include "engine/MemoryGovernor.hpp"
#include "engine/NativeRuntime.hpp"
#include "metal/MetalBackend.hpp"

#include <string>

namespace splash::engine {

// Emits only transitions; retry counts and queue depth do not produce logs.
class MemoryStatusReporter final {
public:
  [[nodiscard]] std::string update(const ResourceWaitSnapshot &wait,
                                    bool hostGrowthAllowed);
private:
  unsigned state_ = 0;
};

// The engine's memory control between commands: it records the system's
// pressure, logs a change in what requests wait for, and runs the reclaim
// pass the pressure policy asks for, telling the governor what it found.
class MemoryControl final {
public:
  MemoryControl(MemoryGovernor &governor, metal::MetalBackend &backend,
                NativeRuntime &loop) noexcept
      : governor_(governor), backend_(backend), loop_(loop) {}

  // One memory-control pass at a command-free point. True while transfers
  // in flight hold back part of the target: the transport runs it again at
  // the next command-free point.
  [[nodiscard]] bool run(MemoryPressure pressure);

private:
  MemoryGovernor &governor_;
  metal::MetalBackend &backend_;
  NativeRuntime &loop_;
  MemoryStatusReporter reporter_;
  MemoryPressurePolicy policy_;
};

} // namespace splash::engine
