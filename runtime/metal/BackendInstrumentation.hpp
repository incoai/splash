#pragma once

#include "metal/MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace splash::metal {

// GPU time of one dispatch committed as its own command while profiling.
struct DispatchTiming {
  std::string pipelineName;
  double gpuSeconds = 0.0;
};

// A copy taken after a dispatch while snapshotting every dispatch of a
// pipeline family (BackendInstrumentation::snapshotEvery): its pipeline, its
// first parameter bytes and the view it binds at the chosen binding.
struct DispatchSnapshot {
  std::string pipelineName;
  std::vector<uint8_t> parameters;
  std::vector<uint8_t> buffer;
};

// A command's dispatches as it was submitted while capturing, its event steps
// dropped and each dispatch's parameter bytes copied into `payloads`, which
// its bytes bindings point into: any of its dispatches can be submitted again,
// alone or with others, while the buffers they bind live. Move-only, so the
// bindings keep pointing into its own payloads.
struct CapturedCommand {
  std::vector<ComputeDispatch> dispatches;
  std::vector<std::vector<uint8_t>> payloads;

  CapturedCommand() = default;
  CapturedCommand(CapturedCommand &&) noexcept = default;
  CapturedCommand &operator=(CapturedCommand &&) noexcept = default;
  CapturedCommand(const CapturedCommand &) = delete;
  CapturedCommand &operator=(const CapturedCommand &) = delete;
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
  // Whether the library holds function `name`, which a pipeline of that name
  // needs, looked up without compiling anything.
  [[nodiscard]] static bool hasFunction(const MetalBackend &backend, std::string_view name);
  // The Metal command buffers the last command submitted takes: one, and one
  // more per buffer its event steps add, which the queue holds 512 of.
  [[nodiscard]] static uint64_t lastCommandBuffers(const MetalBackend &backend);
  // While profiling, submitCommandAsync() commits every dispatch as its own
  // command and waits for it, recording its GPU time, then invokes
  // completion inline and returns an already-completed ticket with the
  // summed timing. It runs a command's host steps between its dispatches and
  // refuses a command with any other event step (EventStep).
  // takeDispatchProfile() reads and clears the timings.
  static void setDispatchProfiling(MetalBackend &backend, bool enabled);
  [[nodiscard]] static std::vector<DispatchTiming>
  takeDispatchProfile(MetalBackend &backend);
  // While set, a command in two parts (MetalBackend::submitCommandPartsAsync)
  // sends nothing ahead: its second part is encoded behind its first in the
  // same buffer, and the GPU starts only once both are encoded, as it starts
  // a command built whole. For comparing the two: they run the same
  // dispatches in the same order, and the host does the same work for both.
  // Profiling runs the first part's dispatches and then the second's.
  static void submitPartsWhole(MetalBackend &backend, bool enabled);
  // While capturing, every command submitted (a command in two parts as one,
  // its first part's dispatches and then its second's) runs as usual and is
  // also kept as a CapturedCommand; takeCapturedCommands() reads and clears
  // them. decode-profile's census times parts of a captured cycle with them.
  static void captureCommands(MetalBackend &backend, bool enabled);
  [[nodiscard]] static std::vector<CapturedCommand> takeCapturedCommands(MetalBackend &backend);
  // Whether awake host steps spin (HostStepGpu::Awake): the backend's check
  // that a GPU spin sees the CPU end it, run now unless one decided; nothing
  // while no check decided (every spin started after its release, the GPU
  // busy with other work).
  [[nodiscard]] static std::optional<bool> awakeHostStepsSpin(MetalBackend &backend);
  // What the backend's checks decided so far, running none: true once a spin
  // was seen, false once awake steps were found not to spin, nothing while
  // no check decided (awake steps then run as plain ones).
  [[nodiscard]] static std::optional<bool> hostStepSpinDecision(const MetalBackend &backend);
  // Every pipeline the backend creates from now on whose function declares
  // function constant kMoeStagedOnly defines it true: the GGUF expert passes
  // run every tile on the staged tile and none on the column tile
  // (kernels/shared/moe_gguf.metal), so that a route's output depends on its
  // own row alone, not on how many rows its step routes to its expert.
  // Production defines it nowhere. Throws once the backend has created a
  // pipeline of its cache, which would keep its own.
  static void stageExpertTiles(MetalBackend &backend);
  // The next command submitted whole (submitCommandAsync) that holds a
  // dispatch of pipeline `pipeline` runs a host step after the first such
  // dispatch that copies the views that
  // dispatch binds at `bindings`, as the work before the step left them.
  // takeSnapshot() reads and clears the copies, a byte vector a binding (none
  // until such a command ran), and disarms the step.
  static void snapshotAfter(MetalBackend &backend, std::string pipeline, std::vector<uint32_t> bindings);
  [[nodiscard]] static std::vector<std::vector<uint8_t>> takeSnapshot(MetalBackend &backend);
  // While `prefix` is not empty, every command submitted (a command in two
  // parts as one) runs a host step after each of its dispatches whose
  // pipeline name starts with `prefix`, which copies that dispatch's first
  // parameter bytes and the view of its buffer binding `fromLast` bindings
  // before its last (a host-visible view); takeSnapshots() reads and clears
  // the copies, in order. decode-profile --experts counts a cycle's routed
  // experts with it.
  static void snapshotEvery(MetalBackend &backend, std::string prefix, uint32_t fromLast);
  [[nodiscard]] static std::vector<DispatchSnapshot> takeSnapshots(MetalBackend &backend);
};

} // namespace splash::metal
