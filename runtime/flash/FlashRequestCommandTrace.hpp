#pragma once

#include "engine/Json.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/ProfilingJson.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <fcntl.h>
#include <unistd.h>

namespace splash::flash {
struct FlashRequestTraceLane {
  uint64_t requestId = 0, generation = 0;
  uint32_t inputRows = 0;
};
struct FlashRequestTraceInfo {
  bool enabled = false;
  uint64_t records = 0, missingProfiles = 0, unexpectedProfiles = 0;
};
struct FlashRequestTraceClockComparison {
  bool hardwareTimestampsValid = false, crossClockValid = false;
  double uncertaintySeconds = 0, submitToGPUStart = 0, commitBeginToGPUStart = 0,
      commitEndToGPUStart = 0, GPUEndToCompletedCallback = 0;
  const char *reason = "hardware timestamps unavailable";
};
inline FlashRequestTraceClockComparison compareFlashRequestTraceClocks(
    const metal::CommandDispatchProfile &p) noexcept {
  FlashRequestTraceClockComparison result;
  const auto finite = [](double v) { return std::isfinite(v); };
  result.hardwareTimestampsValid = finite(p.commandGpuStartSeconds) && finite(p.commandGpuEndSeconds) &&
      p.commandGpuStartSeconds > 0 && p.commandGpuEndSeconds >= p.commandGpuStartSeconds;
  if (!result.hardwareTimestampsValid) return result;
  const auto &before = p.commitClockBridge; const auto &after = p.completedClockBridge;
  if (!before.valid || !after.valid || !finite(before.steadyMinusMachSeconds) ||
      !finite(after.steadyMinusMachSeconds) || !finite(before.uncertaintySeconds) ||
      !finite(after.uncertaintySeconds) || before.uncertaintySeconds < 0 || after.uncertaintySeconds < 0) {
    result.reason = "valid Mach/steady clock bridges unavailable"; return result;
  }
  result.uncertaintySeconds = before.uncertaintySeconds + after.uncertaintySeconds;
  const double drift = std::abs(after.steadyMinusMachSeconds - before.steadyMinusMachSeconds);
  if (drift > result.uncertaintySeconds + 0.000005) {
    result.reason = "clock bridge offsets changed across command"; return result;
  }
  const double GPUStartSteady = p.commandGpuStartSeconds + before.steadyMinusMachSeconds;
  const double GPUEndSteady = p.commandGpuEndSeconds + after.steadyMinusMachSeconds;
  if (!finite(p.hostSubmissionStartSeconds) || !finite(p.hostCommitBeginSeconds) ||
      !finite(p.hostCommitEndSeconds) || !finite(p.hostCompletedSeconds) ||
      p.hostSubmissionStartSeconds <= 0 || p.hostCommitBeginSeconds < p.hostSubmissionStartSeconds ||
      p.hostCommitEndSeconds < p.hostCommitBeginSeconds || p.hostCompletedSeconds <= 0 ||
      GPUStartSteady + result.uncertaintySeconds < p.hostCommitBeginSeconds ||
      GPUEndSteady + result.uncertaintySeconds < GPUStartSteady ||
      p.hostCompletedSeconds + result.uncertaintySeconds < GPUEndSteady) {
    result.reason = "host/hardware command boundaries are inconsistent"; return result;
  }
  result.crossClockValid = true; result.reason = "valid command hardware timestamps with consistent Mach/steady bridges";
  result.submitToGPUStart = GPUStartSteady - p.hostSubmissionStartSeconds;
  result.commitBeginToGPUStart = GPUStartSteady - p.hostCommitBeginSeconds;
  // Negative values are retained: GPU work can start while commit() is running.
  result.commitEndToGPUStart = GPUStartSteady - p.hostCommitEndSeconds;
  result.GPUEndToCompletedCallback = p.hostCompletedSeconds - GPUEndSteady;
  return result;
}
inline void writeFlashRequestCommandTraceRecord(std::ostream &out, uint64_t instance,
    const char *phase, const char *role, std::span<const FlashRequestTraceLane> lanes,
    const metal::CommandDispatchProfile *p, bool expectedSubmission) {
  out << "{\"schema\":\"splash-request-command-trace-v1\",\"instrumentation_on\":true"
      << ",\"metadata_overhead_is_diagnostic\":true,\"labels_are_not_wire_causality\":true"
      << ",\"instance_id\":" << instance << ",\"phase\":" << json::quote(phase)
      << ",\"role\":" << json::quote(role) << ",\"lanes\":" << lanes.size() << ",\"requests\":[";
  uint64_t rows = 0;
  for (size_t index = 0; index < lanes.size(); ++index) {
    if (index) out << ','; const auto &lane = lanes[index]; rows += lane.inputRows;
    out << "{\"request_id\":" << lane.requestId << ",\"generation\":" << lane.generation
        << ",\"input_rows\":" << lane.inputRows << '}';
  }
  out << "],\"actual_rows\":" << rows << ",\"submission_expected\":" << (expectedSubmission ? "true" : "false")
      << ",\"profile_present\":" << (p ? "true" : "false");
  if (!p) { out << ",\"event\":" << json::quote(expectedSubmission ? "profile_missing" : "resolved_without_gpu_submit") << "}\n"; return; }
  const auto clocks = compareFlashRequestTraceClocks(*p);
  out << ",\"command_sequence\":" << p->sequence << ",\"profiling_mode\":"
      << json::quote(metal::commandDispatchProfilingModeName(p->mode)) << ",\"profile_status\":"
      << json::quote(metal::commandDispatchProfileStatusName(p->status))
      << ",\"encoder_boundaries_altered\":" << (p->encoderBoundariesAltered ? "true" : "false")
      << ",\"sampling_barriers\":" << (p->samplingBarriers ? "true" : "false")
      << ",\"dispatch_count\":" << p->dispatchCount
      << ",\"dispatch_metadata_truncated\":" << (p->dispatchMetadataTruncated ? "true" : "false")
      << ",\"dropped_profiles_before\":" << p->droppedProfilesBefore;
  const auto field = [&](const char *name, double value, bool valid = true) {
    out << ',' << json::quote(name) << ':'; if (valid) profiling::writeNumber(out, value); else out << "null";
  };
  field("gpu_hardware_start_mach_seconds", p->commandGpuStartSeconds, clocks.hardwareTimestampsValid);
  field("gpu_hardware_end_mach_seconds", p->commandGpuEndSeconds, clocks.hardwareTimestampsValid);
  out << ",\"hardware_timestamps_valid\":" << (clocks.hardwareTimestampsValid ? "true" : "false");
  const bool kernelTimingValid = p->commandKernelTimingValid &&
      std::isfinite(p->commandKernelStartSeconds) && std::isfinite(p->commandKernelEndSeconds) &&
      p->commandKernelStartSeconds > 0 && p->commandKernelEndSeconds >= p->commandKernelStartSeconds;
  out << ",\"driver_kernel_timing_valid\":" << (kernelTimingValid ? "true" : "false")
      << ",\"driver_kernel_timing_scope\":\"Metal command kernelStartTime/kernelEndTime; driver processing, not GPU execution\"";
  field("driver_kernel_start_mach_seconds", p->commandKernelStartSeconds, kernelTimingValid);
  field("driver_kernel_end_mach_seconds", p->commandKernelEndSeconds, kernelTimingValid);
  field("driver_kernel_processing_seconds", p->commandKernelEndSeconds - p->commandKernelStartSeconds, kernelTimingValid);
  field("host_submit_entry_steady_seconds", p->hostSubmissionStartSeconds);
  field("host_commit_begin_steady_seconds", p->hostCommitBeginSeconds);
  field("host_commit_end_steady_seconds", p->hostCommitEndSeconds);
  field("host_scheduled_callback_steady_seconds", p->hostScheduledSeconds);
  field("host_completed_callback_steady_seconds", p->hostCompletedSeconds);
  field("host_ready_steady_seconds", p->hostReadySeconds);
  field("gpu_seconds", p->timing.gpuSeconds); field("command_wall_seconds", p->timing.wallSeconds);
  out << ",\"clock_bridges\":{\"commit\":"; profiling::writeJson(out, p->commitClockBridge);
  out << ",\"completed\":"; profiling::writeJson(out, p->completedClockBridge); out << '}';
  out << ",\"cross_clock_valid\":" << (clocks.crossClockValid ? "true" : "false")
      << ",\"cross_clock_reason\":" << json::quote(clocks.reason);
  field("cross_clock_uncertainty_seconds", clocks.uncertaintySeconds, clocks.crossClockValid);
  field("backend_submit_entry_to_gpu_start_seconds", clocks.submitToGPUStart, clocks.crossClockValid);
  field("commit_begin_to_gpu_start_seconds", clocks.commitBeginToGPUStart, clocks.crossClockValid);
  field("commit_end_to_gpu_start_seconds", clocks.commitEndToGPUStart, clocks.crossClockValid);
  field("hardware_gpu_end_to_completed_callback_seconds", clocks.GPUEndToCompletedCallback, clocks.crossClockValid);
  out << ",\"callback_timestamp_is_not_hardware_timestamp\":true}\n";
}
// Single Worker owner; there is no cross-thread sink or GPU synchronization.
class FlashRequestCommandTrace final {
public:
  [[nodiscard]] static std::unique_ptr<FlashRequestCommandTrace> fromEnvironment() {
    const char *path = std::getenv("SPLASH_FLASH_REQUEST_COMMAND_TRACE");
    if (!path) return {};
    return std::unique_ptr<FlashRequestCommandTrace>(new FlashRequestCommandTrace(path));
  }
  ~FlashRequestCommandTrace() { if (fd_ >= 0) ::close(fd_); }
  FlashRequestCommandTrace(const FlashRequestCommandTrace &) = delete;
  FlashRequestCommandTrace &operator=(const FlashRequestCommandTrace &) = delete;
  [[nodiscard]] FlashRequestTraceInfo info() const noexcept { return info_; }
  void afterCall(metal::MetalBackend &backend, uint64_t instance, const char *phase,
      const char *role, std::span<const FlashRequestTraceLane> lanes, const metal::CommandTiming &timing) {
    const bool expected = timing.wallSeconds > 0 || timing.gpuSeconds > 0;
    const auto profiles = backend.takeCommandDispatchProfiles();
    if (expected && profiles.empty()) ++info_.missingProfiles;
    if (profiles.size() > (expected ? 1u : 0u)) info_.unexpectedProfiles += profiles.size() - (expected ? 1u : 0u);
    if (profiles.empty()) emit(instance, phase, role, lanes, nullptr, expected);
    for (const auto &profile : profiles) emit(instance, phase, role, lanes, &profile, expected);
  }
private:
  explicit FlashRequestCommandTrace(const char *path) {
    if (!*path || !std::filesystem::path(path).is_absolute())
      throw std::invalid_argument("SPLASH_FLASH_REQUEST_COMMAND_TRACE requires a fresh absolute local JSONL path");
    fd_ = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "cannot create fresh request command trace");
    info_.enabled = true;
  }
  void emit(uint64_t instance, const char *phase, const char *role,
      std::span<const FlashRequestTraceLane> lanes, const metal::CommandDispatchProfile *profile, bool expected) {
    std::ostringstream line; writeFlashRequestCommandTraceRecord(line, instance, phase, role, lanes, profile, expected);
    const auto bytes = line.str(); size_t written = 0;
    while (written < bytes.size()) {
      const auto count = ::write(fd_, bytes.data() + written, bytes.size() - written);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) throw std::system_error(count < 0 ? errno : EIO, std::generic_category(), "request command trace write failed");
      written += size_t(count);
    }
    ++info_.records;
  }
  int fd_ = -1;
  FlashRequestTraceInfo info_;
};
} // namespace splash::flash
