#pragma once

#include "metal/MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
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
  // summed timing. takeDispatchProfile() reads and clears the timings.
  static void setDispatchProfiling(MetalBackend &backend, bool enabled);
  [[nodiscard]] static std::vector<DispatchTiming>
  takeDispatchProfile(MetalBackend &backend);
  // While profiling, called after each replayed dispatch completes and before
  // the next runs, so benchmarks can read Shared scratch before it is reused.
  // The dispatch and its bytes bindings remain valid during the call. An
  // empty observer clears it; unprofiled commands never call it.
  using DispatchObserver = std::function<void(const ComputeDispatch &)>;
  static void setDispatchObserver(MetalBackend &backend,
                                 DispatchObserver observer);
};

} // namespace splash::metal
