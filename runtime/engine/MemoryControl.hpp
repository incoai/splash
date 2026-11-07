#pragma once

#include "engine/Engine.hpp"
#include "engine/MemoryGovernor.hpp"
#include "metal/MetalBackend.hpp"

#include <string>

namespace splash::engine {

class NativeRuntime;

// Emits only transitions; retry counts and queue depth do not produce logs.
class MemoryStatusReporter final {
public:
  [[nodiscard]] std::string update(const ResourceWaitSnapshot &wait,
                                    bool hostGrowthAllowed);
private:
  unsigned state_ = 0;
};

// The engine's memory control between commands: it releases the weights of
// an engine left without a request (NativeRuntime::releaseIdleWeights),
// records the system's pressure, lands finished transfers and gives the
// ledger the lanes' demand and the wanted chains
// (NativeRuntime::collectLandings), runs the governor's control pass
// (MemoryGovernor::controlPass) and logs a change in what requests wait for.
class MemoryControl final {
public:
  MemoryControl(MemoryGovernor &governor, metal::MetalBackend &backend,
                NativeRuntime &loop) noexcept
      : governor_(governor), backend_(backend), loop_(loop) {}

  // One memory-control pass at a command-free point. True while the host
  // claim waits for memory on its way back: the transport runs it again at
  // the next command-free point (MemoryGovernor::controlPass).
  [[nodiscard]] bool run(MemoryPressure pressure);

private:
  MemoryGovernor &governor_;
  metal::MetalBackend &backend_;
  NativeRuntime &loop_;
  MemoryStatusReporter reporter_;
};

} // namespace splash::engine
