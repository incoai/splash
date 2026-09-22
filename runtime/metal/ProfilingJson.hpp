#pragma once

#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <ostream>
#include <string_view>

// Explicit metadata-only serialization. No MetalBuffer contents, inline
// parameter bytes, request objects, prompts, token IDs or weight data cross
// this interface. Vector limits retain counts and report truncation.
namespace splash::profiling {
inline constexpr size_t kMaximumJsonDispatches = 4096;
inline constexpr size_t kMaximumJsonBindings = 32;

class ScopedJsonFormat final {
public:
  explicit ScopedJsonFormat(std::ostream &out)
      : out_(out), flags_(out.flags()), width_(out.width()), locale_(out.getloc()),
        changedLocale_(locale_ != std::locale::classic()) {
    out_.setf(std::ios::dec, std::ios::basefield);
    out_.unsetf(std::ios::showbase | std::ios::showpos);
    out_.width(0);
    if (changedLocale_) out_.imbue(std::locale::classic());
  }
  ~ScopedJsonFormat() {
    out_.flags(flags_);out_.width(width_);
    if (changedLocale_) { try { out_.imbue(locale_); } catch (...) {} }
  }
private:
  std::ostream &out_;
  std::ios::fmtflags flags_;
  std::streamsize width_;
  std::locale locale_;
  bool changedLocale_;
};

inline void writeNumber(std::ostream &out, double value) {
  if (!std::isfinite(value)) { out << "null"; return; }
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
      std::chars_format::general, std::numeric_limits<double>::max_digits10);
  if (result.ec != std::errc{}) { out << "null"; return; }
  out.write(buffer, result.ptr - buffer);
}
inline void writeBoolean(std::ostream &out, bool value) {
  out << (value ? "true" : "false");
}
inline void writeString(std::ostream &out, std::string_view value) {
  out << json::quote(value);
}
inline void writeJson(std::ostream &out, const metal::DispatchSize &value) {
  ScopedJsonFormat format(out);
  out << '[' << value.x << ',' << value.y << ',' << value.z << ']';
}
inline void writeJson(std::ostream &out,
                      const metal::CommandDispatchProfilingCapability &value) {
  ScopedJsonFormat format(out);
  out << "{\"timestamp_counter_set\":"; writeBoolean(out,value.timestampCounterSet);
  out << ",\"dispatch_boundary\":"; writeBoolean(out,value.dispatchBoundary);
  out << ",\"stage_boundary\":"; writeBoolean(out,value.stageBoundary);
  out << ",\"reason\":"; writeString(out,value.reason); out << '}';
}
inline void writeJson(std::ostream &out,
                      const metal::DeviceMemorySampleTiming &value) {
  ScopedJsonFormat format(out);
  out << "{\"began_steady_seconds\":";
  if(value.beganSteadySeconds>0)writeNumber(out,value.beganSteadySeconds);else out << "null";
  out << ",\"ended_steady_seconds\":";
  if(value.valid)writeNumber(out,value.endedSteadySeconds);else out << "null";
  out << ",\"seconds\":";
  if(value.valid)writeNumber(out,value.seconds);else out << "null";
  out << ",\"valid\":";writeBoolean(out,value.valid);out << '}';
}
inline void writeJson(std::ostream &out,
                      const metal::MachSteadyClockBridge &value) {
  ScopedJsonFormat format(out);
  out << "{\"began_steady_seconds\":";
  if(value.beganSteadySeconds>0)writeNumber(out,value.beganSteadySeconds);else out << "null";
  out << ",\"ended_steady_seconds\":";
  if(value.endedSteadySeconds>0)writeNumber(out,value.endedSteadySeconds);else out << "null";
  out << ",\"mach_seconds\":";
  if(value.valid)writeNumber(out,value.machSeconds);else out << "null";
  out << ",\"steady_minus_mach_seconds\":";
  if(value.valid)writeNumber(out,value.steadyMinusMachSeconds);else out << "null";
  out << ",\"uncertainty_seconds\":";
  if(value.valid)writeNumber(out,value.uncertaintySeconds);else out << "null";
  out << ",\"mach_absolute_timestamp\":";
  if(value.valid)out << value.machAbsoluteTimestamp;else out << "null";
  out << ",\"timebase_numer\":" << value.timebaseNumer
      << ",\"timebase_denom\":" << value.timebaseDenom << ",\"valid\":";
  writeBoolean(out,value.valid);out << '}';
}
inline void writeJson(std::ostream &out,
                      const metal::CommandDispatchTimestamp &value) {
  ScopedJsonFormat format(out);
  out << "{\"index\":" << value.index << ",\"pipeline\":"; writeString(out,value.pipelineName);
  out << ",\"threadgroups\":"; writeJson(out,value.threadgroups);
  out << ",\"threads_per_threadgroup\":"; writeJson(out,value.threadsPerThreadgroup);
  out << ",\"execution_width\":" << value.executionWidth
      << ",\"max_threads_per_threadgroup\":" << value.maxTotalThreadsPerThreadgroup
      << ",\"static_threadgroup_memory_bytes\":" << value.staticThreadgroupMemoryBytes
      << ",\"gpu_start_timestamp\":" << value.gpuStartTimestamp
      << ",\"gpu_end_timestamp\":" << value.gpuEndTimestamp
      << ",\"timestamps_valid\":"; writeBoolean(out,value.timestampsValid);
  out << ",\"calibrated_start_seconds\":";
  if(value.timestampsValid)writeNumber(out,value.calibratedStartSeconds);else out << "null";
  out << ",\"calibrated_end_seconds\":";
  if(value.timestampsValid)writeNumber(out,value.calibratedEndSeconds);else out << "null";
  out << ",\"gpu_seconds\":";
  if(value.timestampsValid)writeNumber(out,value.gpuSeconds);else out << "null";
  const size_t count=std::min(value.bindings.size(),kMaximumJsonBindings);
  out << ",\"bindings_total\":" << value.bindings.size() << ",\"bindings_truncated\":";
  writeBoolean(out,count<value.bindings.size());out << ",\"bindings\":[";
  for(size_t i=0;i<count;++i) {
    if(i)out << ',';const auto &binding=value.bindings[i];
    out << "{\"index\":" << binding.index << ",\"size_bytes\":" << binding.sizeBytes
        << ",\"inline_bytes\":";writeBoolean(out,binding.inlineBytes);out << '}';
  }
  out << "]}";
}
inline void writeJson(std::ostream &out, const metal::CommandDispatchProfile &value) {
  ScopedJsonFormat format(out);
  out << "{\"sequence\":" << value.sequence << ",\"mode\":";
  writeString(out,metal::commandDispatchProfilingModeName(value.mode));
  out << ",\"status\":";writeString(out,metal::commandDispatchProfileStatusName(value.status));
  out << ",\"reason\":";writeString(out,value.reason);
  out << ",\"encoder_boundaries_altered\":";writeBoolean(out,value.encoderBoundariesAltered);
  out << ",\"sampling_barriers\":";writeBoolean(out,value.samplingBarriers);
  out << ",\"dropped_profiles_before\":" << value.droppedProfilesBefore;
#define SPLASH_PROFILE_JSON_NUMBER(Key,Field) out << ",\"" Key "\":"; writeNumber(out,value.Field)
  SPLASH_PROFILE_JSON_NUMBER("gpu_seconds",timing.gpuSeconds);
  SPLASH_PROFILE_JSON_NUMBER("wall_seconds",timing.wallSeconds);
  SPLASH_PROFILE_JSON_NUMBER("command_gpu_start_seconds",commandGpuStartSeconds);
  SPLASH_PROFILE_JSON_NUMBER("command_gpu_end_seconds",commandGpuEndSeconds);
  out << ",\"command_kernel_timing_valid\":";writeBoolean(out,value.commandKernelTimingValid);
  out << ",\"command_kernel_start_seconds\":";
  if(value.commandKernelTimingValid)writeNumber(out,value.commandKernelStartSeconds);else out << "null";
  out << ",\"command_kernel_end_seconds\":";
  if(value.commandKernelTimingValid)writeNumber(out,value.commandKernelEndSeconds);else out << "null";
  SPLASH_PROFILE_JSON_NUMBER("host_preparation_seconds",hostPreparationSeconds);
  SPLASH_PROFILE_JSON_NUMBER("host_encoding_seconds",hostEncodingSeconds);
  SPLASH_PROFILE_JSON_NUMBER("sparse_dependency_wait_seconds",sparseDependencyWaitSeconds);
  SPLASH_PROFILE_JSON_NUMBER("host_commit_seconds",hostCommitSeconds);
  out << ",\"host_commit_timing_valid\":";writeBoolean(out,value.hostCommitTimingValid);
  SPLASH_PROFILE_JSON_NUMBER("host_submission_start_seconds",hostSubmissionStartSeconds);
  SPLASH_PROFILE_JSON_NUMBER("host_encoding_start_seconds",hostEncodingStartSeconds);
  SPLASH_PROFILE_JSON_NUMBER("host_encoding_end_seconds",hostEncodingEndSeconds);
  SPLASH_PROFILE_JSON_NUMBER("host_commit_begin_seconds",hostCommitBeginSeconds);
  SPLASH_PROFILE_JSON_NUMBER("host_commit_end_seconds",hostCommitEndSeconds);
  SPLASH_PROFILE_JSON_NUMBER("host_scheduled_seconds",hostScheduledSeconds);
  SPLASH_PROFILE_JSON_NUMBER("host_completed_seconds",hostCompletedSeconds);
  SPLASH_PROFILE_JSON_NUMBER("host_ready_seconds",hostReadySeconds);
#undef SPLASH_PROFILE_JSON_NUMBER
  out << ",\"device_memory_samples\":{\"pre_commit\":";
  writeJson(out,value.preCommitDeviceMemorySample);
  out << ",\"post_commit\":";writeJson(out,value.postCommitDeviceMemorySample);
  out << ",\"scheduled\":";writeJson(out,value.scheduledDeviceMemorySample);
  out << ",\"completed\":";writeJson(out,value.completedDeviceMemorySample);out << '}';
  out << ",\"clock_bridges\":{\"commit\":";writeJson(out,value.commitClockBridge);
  out << ",\"completed\":";writeJson(out,value.completedClockBridge);out << '}';
  out << ",\"calibration\":{\"cpu_start\":" << value.calibrationCpuStart
      << ",\"gpu_start\":" << value.calibrationGpuStart << ",\"cpu_end\":" << value.calibrationCpuEnd
      << ",\"gpu_end\":" << value.calibrationGpuEnd << '}';
  const size_t count=std::min(value.dispatches.size(),kMaximumJsonDispatches);
  out << ",\"dispatch_count\":" << value.dispatchCount
      << ",\"dispatch_metadata_truncated\":";
  writeBoolean(out,value.dispatchMetadataTruncated);
  out << ",\"dispatches_total\":" << value.dispatchCount
      << ",\"dispatches_metadata_count\":" << value.dispatches.size()
      << ",\"dispatches_emitted\":" << count << ",\"dispatches_truncated\":";
  writeBoolean(out,value.dispatchMetadataTruncated || count<value.dispatches.size() ||
      count<value.dispatchCount);out << ",\"dispatches\":[";
  for(size_t i=0;i<count;++i){if(i)out << ',';writeJson(out,value.dispatches[i]);}
  out << "]}";
}
} // namespace splash::profiling
