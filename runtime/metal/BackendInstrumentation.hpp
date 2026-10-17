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
  // summed timing. It runs a command's host steps between its dispatches and
  // refuses a command with any other event step (EventStep).
  // takeDispatchProfile() reads and clears the timings.
  static void setDispatchProfiling(MetalBackend &backend, bool enabled);
  [[nodiscard]] static std::vector<DispatchTiming>
  takeDispatchProfile(MetalBackend &backend);
  // Whether awake host steps spin (EventStep::awake): the backend's check
  // that a GPU spin sees the CPU end it, run now if no command ran it.
  [[nodiscard]] static bool awakeHostStepsSpin(MetalBackend &backend);
};

} // namespace splash::metal
