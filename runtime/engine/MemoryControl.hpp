#pragma once

#include "engine/MemoryGovernor.hpp"
#include "engine/NativeRuntime.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <functional>
#include <optional>
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

// The engine's memory control between commands: it releases the weights of
// an engine left without a request (NativeRuntime::releaseIdleWeights),
// records the system's pressure, holds the host's debt for compression the
// engine's growth caused (CompressionGuard), logs a change in what requests
// wait for, and runs the reclaim pass the pressure policy asks for, telling
// the governor what it found.
class MemoryControl final {
public:
  using CompressorContent = std::function<std::optional<uint64_t>()>;

  MemoryControl(MemoryGovernor &governor, metal::MetalBackend &backend,
                NativeRuntime &loop,
                CompressorContent compressorContent = queryCompressorContent)
      : governor_(governor), backend_(backend), loop_(loop),
        compressorContent_(std::move(compressorContent)) {}

  // One memory-control pass at a command-free point. True while transfers
  // in flight hold back part of the target: the transport runs it again at
  // the next command-free point.
  [[nodiscard]] bool run(MemoryPressure pressure);

private:
  // Samples the compressor for the guard while the weights stay resident,
  // and gives the governor the debt it holds. True when the debt changed.
  bool guardCompression(const MemoryGovernorSnapshot &memory, double now);

  MemoryGovernor &governor_;
  metal::MetalBackend &backend_;
  NativeRuntime &loop_;
  CompressorContent compressorContent_;
  MemoryStatusReporter reporter_;
  MemoryPressurePolicy policy_;
  CompressionGuard guard_;
};

} // namespace splash::engine
