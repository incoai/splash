#pragma once

#include "metal/MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace splash::metal {

// GPU time of one dispatch committed as its own command while profiling.
struct DispatchTiming {
  std::string pipelineName;
  double gpuSeconds = 0.0;
};

// Host and GPU instants in system mach time, in seconds. A command with
// event steps spans the first through last Metal command buffer.
struct CommandTrace {
  struct Segment { double begin = 0.0, end = 0.0; };
  std::vector<Segment> segments;
  uint64_t sequence = 0;
  uint64_t dispatches = 0;
  double prepareStart = 0.0;
  double encodeStart = 0.0;
  double commit = 0.0;
  double complete = 0.0;
  double gpuStart = 0.0;
  double gpuEnd = 0.0;
};

// What tests and benchmarks observe of a MetalBackend beyond its serving
// API. Only the instrumented build of MetalBackend.mm
// (SPLASH_BACKEND_INSTRUMENTATION) defines these; production binaries never
// link it.
class BackendInstrumentation final {
public:
  // Commands that passed the submission gate since construction.
  [[nodiscard]] static uint64_t submittedCommands(const MetalBackend &backend);
  [[nodiscard]] static size_t cachedPipelines(const MetalBackend &backend);
  // While profiling, submitCommandAsync() commits every dispatch as its own
  // command and waits for it, recording its GPU time, then invokes
  // completion inline and returns an already-completed ticket with the
  // summed timing; it refuses a command with event steps (EventStep).
  // takeDispatchProfile() reads and clears the timings.
  static void setDispatchProfiling(MetalBackend &backend, bool enabled);
  [[nodiscard]] static std::vector<DispatchTiming>
  takeDispatchProfile(MetalBackend &backend);
  // Optional ordinary-command tracing; it does not split dispatches or
  // wait for additional work. Take only after the measured tickets complete.
  static void setCommandTracing(MetalBackend &backend, bool enabled);
  [[nodiscard]] static std::vector<CommandTrace>
  takeCommandTrace(MetalBackend &backend);
  [[nodiscard]] static uint64_t droppedCommandTraces(MetalBackend &backend);
};

} // namespace splash::metal
