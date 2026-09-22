// CPU-only diagnostic contract tests. No Metal device, backend, model or GPU.
#include "flash/FlashRequestCommandTrace.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <bit>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <string_view>
#include <sys/stat.h>

namespace {
using splash::flash::FlashRequestCommandTrace;
using splash::flash::FlashRequestTraceLane;
using splash::metal::CommandDispatchProfile;
uint64_t checks = 0, clockCases = 0, parsedRecords = 0, fileCases = 0;
constexpr const char *flag = "SPLASH_FLASH_REQUEST_COMMAND_TRACE";

void require(bool value, const char *message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
void closeEnough(double actual, double expected, const char *message) {
  require(std::isfinite(actual) && std::abs(actual - expected) < 1e-12, message);
}
template <class Exception, class Operation>
void rejects(Operation operation, const char *message) {
  bool rejected = false;
  try { operation(); } catch (const Exception &) { rejected = true; }
  require(rejected, message);
}
void environment(const char *value) {
  require(value ? ::setenv(flag, value, 1) == 0 : ::unsetenv(flag) == 0,
          "could not update test-process environment");
}
struct EnvironmentRestore {
  bool present = std::getenv(flag) != nullptr;
  std::string value = present ? std::getenv(flag) : "";
  ~EnvironmentRestore() {
    if (present) ::setenv(flag, value.c_str(), 1); else ::unsetenv(flag);
  }
};
struct TemporaryDirectory {
  std::filesystem::path path;
  TemporaryDirectory() {
    std::array<char, 80> pattern{};
    const std::string text = "/tmp/splash-request-command-trace-tests-XXXXXX";
    std::copy(text.begin(), text.end(), pattern.begin());
    auto *created = ::mkdtemp(pattern.data());
    require(created != nullptr, "could not create test temporary directory");
    path = created;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

CommandDispatchProfile validProfile() {
  CommandDispatchProfile p;
  p.sequence = 73;
  p.mode = splash::metal::CommandDispatchProfilingMode::Command;
  p.status = splash::metal::CommandDispatchProfileStatus::Complete;
  p.dispatchCount = 5132;
  p.dispatchMetadataTruncated = true;
  p.droppedProfilesBefore = 9;
  // Independent, exactly representable spans on two different clock axes.
  p.commandGpuStartSeconds = 100.0625;
  p.commandGpuEndSeconds = 100.125;
  p.hostSubmissionStartSeconds = 110;
  p.hostCommitBeginSeconds = 110.03125;
  p.hostCommitEndSeconds = 110.09375;
  p.hostScheduledSeconds = 110.078125;
  p.hostCompletedSeconds = 110.15625;
  p.hostReadySeconds = 110.1875;
  p.timing.gpuSeconds = 0.0625;
  p.timing.wallSeconds = 0.15625;
  for (auto *bridge : {&p.commitClockBridge, &p.completedClockBridge}) {
    bridge->valid = true;
    bridge->machAbsoluteTimestamp = 100000000000;
    bridge->timebaseNumer = 1;
    bridge->timebaseDenom = 1;
    bridge->machSeconds = 100;
    bridge->steadyMinusMachSeconds = 10;
    bridge->uncertaintySeconds = 0.000001;
    bridge->beganSteadySeconds = 109.999999;
    bridge->endedSteadySeconds = 110.000001;
  }
  return p;
}

std::string record(const CommandDispatchProfile *p, bool expected = true,
                   std::span<const FlashRequestTraceLane> lanes = {},
                   const char *phase = "prefill", const char *role = "target") {
  std::ostringstream out;
  splash::flash::writeFlashRequestCommandTraceRecord(out, 44, phase, role,
                                                    lanes, p, expected);
  const auto text = out.str();
  require(!text.empty() && text.back() == '\n', "trace is not a complete JSONL record");
  require(std::count(text.begin(), text.end(), '\n') == 1,
          "record contains an unescaped interior newline");
  return text;
}
NSDictionary *deserialize(const std::string &text) {
  NSData *data = [NSData dataWithBytes:text.data() length:text.size()];
  NSError *error = nil;
  id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error == nil && [object isKindOfClass:[NSDictionary class]],
          "trace does not deserialize as a JSON dictionary");
  ++parsedRecords;
  return object;
}
double number(NSDictionary *object, NSString *key) {
  id value = object[key];
  require([value isKindOfClass:[NSNumber class]], "expected numeric trace field");
  return [value doubleValue];
}
bool boolean(NSDictionary *object, NSString *key) {
  id value = object[key];
  require([value isKindOfClass:[NSNumber class]], "expected boolean trace field");
  return [value boolValue];
}
void nullComparisons(NSDictionary *object) {
  require(!boolean(object, @"cross_clock_valid"), "invalid clocks claim valid cross-clock attribution");
  for (NSString *key in @[@"cross_clock_uncertainty_seconds",
                          @"backend_submit_entry_to_gpu_start_seconds",
                          @"commit_begin_to_gpu_start_seconds",
                          @"commit_end_to_gpu_start_seconds",
                          @"hardware_gpu_end_to_completed_callback_seconds"])
    require(object[key] == [NSNull null], "invalid cross-clock attribution invented a numeric delay");
  require([object[@"cross_clock_reason"] length] > 0, "missing invalid-clock explanation");
}

void clockContract() {
  auto p = validProfile();
  auto comparison = splash::flash::compareFlashRequestTraceClocks(p);
  ++clockCases;
  require(comparison.hardwareTimestampsValid && comparison.crossClockValid,
          "consistent independent Mach/steady axes were rejected");
  closeEnough(comparison.submitToGPUStart, 0.0625, "wrong submit-entry to GPU-start delay");
  closeEnough(comparison.commitBeginToGPUStart, 0.03125, "wrong commit-begin to GPU-start delay");
  closeEnough(comparison.commitEndToGPUStart, -0.03125,
              "legitimate GPU-start overlap with commit() was clamped or rejected");
  closeEnough(comparison.GPUEndToCompletedCallback, 0.03125,
              "completed callback was mistaken for the hardware GPU-end timestamp");
  NSDictionary *json = deserialize(record(&p));
  closeEnough(number(json, @"backend_submit_entry_to_gpu_start_seconds"), 0.0625,
              "JSON changed submit-entry delay");
  closeEnough(number(json, @"commit_end_to_gpu_start_seconds"), -0.03125,
              "JSON lost negative commit overlap");
  closeEnough(number(json, @"hardware_gpu_end_to_completed_callback_seconds"), 0.03125,
              "JSON lost callback arrival delay");
  require(boolean(json, @"callback_timestamp_is_not_hardware_timestamp"),
          "callback/hardware distinction missing");
  closeEnough(number(json, @"gpu_hardware_start_mach_seconds"), 100.0625,
              "hardware timestamp incorrectly relabeled onto the steady axis");
  closeEnough(number(json, @"host_submit_entry_steady_seconds"), 110,
              "host timestamp incorrectly relabeled onto the Mach axis");

  using Mutation = std::function<void(CommandDispatchProfile &)>;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::array<Mutation, 31> invalid{{
      [](auto &q) { q.commandGpuStartSeconds = 0; },
      [](auto &q) { q.commandGpuEndSeconds = 0; },
      [](auto &q) { q.commandGpuStartSeconds = -1; },
      [nan](auto &q) { q.commandGpuStartSeconds = nan; },
      [nan](auto &q) { q.commandGpuEndSeconds = nan; },
      [inf](auto &q) { q.commandGpuStartSeconds = inf; },
      [inf](auto &q) { q.commandGpuEndSeconds = inf; },
      [](auto &q) { q.commandGpuEndSeconds = q.commandGpuStartSeconds - 0.125; },
      [](auto &q) { q.commitClockBridge.valid = false; },
      [](auto &q) { q.completedClockBridge.valid = false; },
      [nan](auto &q) { q.commitClockBridge.steadyMinusMachSeconds = nan; },
      [inf](auto &q) { q.completedClockBridge.steadyMinusMachSeconds = inf; },
      [nan](auto &q) { q.commitClockBridge.uncertaintySeconds = nan; },
      [inf](auto &q) { q.completedClockBridge.uncertaintySeconds = inf; },
      [](auto &q) { q.commitClockBridge.uncertaintySeconds = -0.01; },
      [](auto &q) { q.completedClockBridge.uncertaintySeconds = -0.01; },
      [](auto &q) { q.completedClockBridge.steadyMinusMachSeconds += 0.001; },
      [nan](auto &q) { q.hostSubmissionStartSeconds = nan; },
      [inf](auto &q) { q.hostSubmissionStartSeconds = inf; },
      [](auto &q) { q.hostSubmissionStartSeconds = 0; },
      [](auto &q) { q.hostSubmissionStartSeconds = -1; },
      [nan](auto &q) { q.hostCommitBeginSeconds = nan; },
      [inf](auto &q) { q.hostCommitEndSeconds = inf; },
      [](auto &q) { q.hostCommitBeginSeconds = q.hostSubmissionStartSeconds - 0.125; },
      [](auto &q) { q.hostCommitEndSeconds = q.hostCommitBeginSeconds - 0.125; },
      [](auto &q) { q.commandGpuStartSeconds = 100; },
      [](auto &q) { q.hostCompletedSeconds = 110.0625; },
      [](auto &q) { q.hostCompletedSeconds = 0; },
      [](auto &q) { q.hostCompletedSeconds = -1; },
      [nan](auto &q) { q.hostCompletedSeconds = nan; },
      [inf](auto &q) { q.hostCompletedSeconds = inf; },
  }};
  for (const auto &mutate : invalid) {
    auto q = validProfile();
    mutate(q);
    ++clockCases;
    require(!splash::flash::compareFlashRequestTraceClocks(q).crossClockValid,
            "missing, nonfinite, drifted or inconsistent clocks accepted");
    nullComparisons(deserialize(record(&q)));
  }
  // A scheduled callback is not evidence of GPU scheduling; missing it cannot
  // invalidate independently consistent hardware/commit/completion boundaries.
  p.hostScheduledSeconds = nan;
  p.hostReadySeconds = inf;
  ++clockCases;
  json = deserialize(record(&p));
  require(boolean(json, @"cross_clock_valid"), "callback arrival substituted for hardware timing");
  require(json[@"host_scheduled_callback_steady_seconds"] == [NSNull null] &&
              json[@"host_ready_steady_seconds"] == [NSNull null],
          "nonfinite diagnostic host fields produced invalid JSON");
  // A small bridge offset difference within its measured error is acceptable.
  p = validProfile();
  p.completedClockBridge.steadyMinusMachSeconds += 0.000001;
  ++clockCases;
  require(splash::flash::compareFlashRequestTraceClocks(p).crossClockValid,
          "bridge drift inside measured uncertainty was rejected");

  // Sub-millisecond attribution must survive JSON at realistic long uptime.
  // Compare decimal delays to the independent intended intervals, rather than
  // reusing the trace's arithmetic as the expected result.
  p = validProfile();
  constexpr double uptime = 1000000, offset = 742.125;
  p.commandGpuStartSeconds = uptime + 0.000250;
  p.commandGpuEndSeconds = uptime + 0.125250;
  p.hostSubmissionStartSeconds = uptime + offset;
  p.hostCommitBeginSeconds = uptime + offset + 0.000125;
  p.hostCommitEndSeconds = uptime + offset + 0.000375;
  p.hostCompletedSeconds = uptime + offset + 0.126000;
  p.hostScheduledSeconds = uptime + offset + 0.000500;
  p.hostReadySeconds = uptime + offset + 0.126125;
  p.commitClockBridge.steadyMinusMachSeconds = offset;
  p.completedClockBridge.steadyMinusMachSeconds = offset;
  ++clockCases;
  json = deserialize(record(&p));
  require(boolean(json, @"cross_clock_valid"), "long-uptime clock attribution was rejected");
  for (const auto &[key, expected] : std::array<std::pair<NSString *, double>, 4>{{
           {@"backend_submit_entry_to_gpu_start_seconds", 0.000250},
           {@"commit_begin_to_gpu_start_seconds", 0.000125},
           {@"commit_end_to_gpu_start_seconds", -0.000125},
           {@"hardware_gpu_end_to_completed_callback_seconds", 0.000750}}})
    require(std::abs(number(json, key) - expected) < 0.000000001,
            "sub-millisecond attribution lost more than 1ns at 1e6-second uptime");
  require(std::bit_cast<uint64_t>(number(json, @"gpu_hardware_start_mach_seconds")) ==
              std::bit_cast<uint64_t>(p.commandGpuStartSeconds) &&
              std::bit_cast<uint64_t>(number(json, @"host_submit_entry_steady_seconds")) ==
              std::bit_cast<uint64_t>(p.hostSubmissionStartSeconds),
          "JSON parsing lost absolute timestamp double bits at long uptime");
}

void metadataContract() {
  auto p = validProfile();
  p.reason = "DO_NOT_EXPORT_REQUEST_TEXT /private/request/path token=248044";
  splash::metal::CommandDispatchTimestamp privateDispatch;
  privateDispatch.pipelineName = "DO_NOT_EXPORT_PIPELINE_OR_TOKENS";
  p.dispatches.push_back(privateDispatch);
  std::array<FlashRequestTraceLane, 4> original{{
      {9007199254740993ULL, 100, 512}, {17, 3, 1}, {18, 4, 128}, {19, 5, 7}}};
  const auto ownedCookies = original;
  original[0] = {999, 200, 9};
  const auto bytes = record(&p, true, ownedCookies, "mtp_verify", "target_joint");
  NSDictionary *json = deserialize(bytes);
  require([json[@"schema"] isEqualToString:@"splash-request-command-trace-v1"],
          "wrong trace schema");
  require([json[@"phase"] isEqualToString:@"mtp_verify"] &&
              [json[@"role"] isEqualToString:@"target_joint"],
          "fixed phase/role labels changed");
  require(boolean(json, @"instrumentation_on") &&
              boolean(json, @"metadata_overhead_is_diagnostic") &&
              boolean(json, @"labels_are_not_wire_causality"),
          "trace failed to disclose instrumentation or causal labeling scope");
  require(number(json, @"instance_id") == 44 && number(json, @"lanes") == 4 &&
              number(json, @"actual_rows") == 648,
          "ragged lane cardinality or real row count changed");
  NSArray *requests = json[@"requests"];
  require([requests isKindOfClass:[NSArray class]] && requests.count == 4,
          "request cookie list missing");
  for (size_t i = 0; i < ownedCookies.size(); ++i) {
    NSDictionary *lane = requests[i];
    require([lane[@"request_id"] unsignedLongLongValue] == ownedCookies[i].requestId &&
                [lane[@"generation"] unsignedLongLongValue] == ownedCookies[i].generation &&
                [lane[@"input_rows"] unsignedLongLongValue] == ownedCookies[i].inputRows,
            "copied request identity/generation or real lane rows changed");
    require(lane.count == 3, "request dictionary exported more than cookies and cardinality");
  }
  require(number(json, @"command_sequence") == 73 &&
              number(json, @"dispatch_count") == 5132 &&
              number(json, @"dropped_profiles_before") == 9 &&
              boolean(json, @"dispatch_metadata_truncated"),
          "dispatch truncation/count/drop provenance missing");
  require([json[@"profiling_mode"] isEqualToString:@"command"] &&
              [json[@"profile_status"] isEqualToString:@"complete"] &&
              !boolean(json, @"encoder_boundaries_altered") &&
              !boolean(json, @"sampling_barriers"),
          "command mode mislabeled as a modified encoder/counter workload");
  require(!boolean(json, @"driver_kernel_timing_valid") &&
              json[@"driver_kernel_processing_seconds"] == [NSNull null],
          "unavailable driver processing timestamps were presented as valid");
  p.commandKernelTimingValid = true;
  p.commandKernelStartSeconds = 1000000.125;
  p.commandKernelEndSeconds = 1000001.625;
  json = deserialize(record(&p));
  require(boolean(json, @"driver_kernel_timing_valid") &&
              number(json, @"driver_kernel_processing_seconds") == 1.5 &&
              number(json, @"driver_kernel_start_mach_seconds") == p.commandKernelStartSeconds,
          "driver processing duration or absolute timestamp serialization changed");
  p.commandKernelEndSeconds = p.commandKernelStartSeconds - 1;
  json = deserialize(record(&p));
  require(!boolean(json, @"driver_kernel_timing_valid") &&
              json[@"driver_kernel_start_mach_seconds"] == [NSNull null],
          "reversed driver processing timestamps were accepted");
  p.commandKernelTimingValid = false;
  for (auto forbidden : {"DO_NOT_EXPORT", "/private/request/path", "248044", "prompt",
                         "request_text", "tokens", "weight", "dispatches", "pipeline"})
    require(bytes.find(forbidden) == std::string::npos,
            "trace exported request text, token IDs, paths, weights or dispatch internals");
  p.mode = splash::metal::CommandDispatchProfilingMode::StagePerDispatch;
  p.status = splash::metal::CommandDispatchProfileStatus::SampleLimitExceeded;
  p.encoderBoundariesAltered = true;
  p.samplingBarriers = true;
  json = deserialize(record(&p));
  require([json[@"profiling_mode"] isEqualToString:@"stage"] &&
              [json[@"profile_status"] isEqualToString:@"sample_limit_exceeded"] &&
              boolean(json, @"encoder_boundaries_altered") &&
              boolean(json, @"sampling_barriers"),
          "altered-encoder stage trace hid its sampling provenance");
  for (bool expected : {true, false}) {
    json = deserialize(record(nullptr, expected, ownedCookies, "rollback", "target_joint"));
    require(!boolean(json, @"profile_present") &&
                boolean(json, @"submission_expected") == expected,
            "missing profile was mistaken for GPU execution");
    require([json[@"event"] isEqualToString:expected ? @"profile_missing" : @"resolved_without_gpu_submit"],
            "missing submission and missing profile were conflated");
    require(json[@"gpu_seconds"] == nil && json[@"cross_clock_valid"] == nil &&
                json[@"command_sequence"] == nil,
            "record without a command profile fabricated command timing");
  }
  json = deserialize(record(&p, true, {}, "quote\"\nphase", "role\\\tname"));
  require([json[@"phase"] isEqualToString:@"quote\"\nphase"] &&
              [json[@"role"] isEqualToString:@"role\\\tname"],
          "metadata escaping broke JSONL boundaries");
}

void fileContract() {
  EnvironmentRestore restore;
  TemporaryDirectory temporary;
  environment(nullptr);
  ++fileCases;
  require(!FlashRequestCommandTrace::fromEnvironment(), "absent flag enabled diagnostic I/O");
  require(std::filesystem::is_empty(temporary.path), "absent flag created a file");
  for (const char *bad : {"", "relative.jsonl", "./relative.jsonl"}) {
    environment(bad);
    ++fileCases;
    rejects<std::invalid_argument>([] { (void)FlashRequestCommandTrace::fromEnvironment(); },
                                  "empty or relative trace path accepted");
    require(std::filesystem::is_empty(temporary.path), "rejected flag created a file");
  }
  const auto fresh = temporary.path / "fresh.jsonl";
  environment(fresh.c_str());
  int openedFD = -1;
  struct stat created{};
  {
    ++fileCases;
    auto trace = FlashRequestCommandTrace::fromEnvironment();
    require(trace && trace->info().enabled && trace->info().records == 0 &&
                trace->info().missingProfiles == 0 && trace->info().unexpectedProfiles == 0,
            "new trace source started with fabricated command/profile counts");
    require(::stat(fresh.c_str(), &created) == 0 && S_ISREG(created.st_mode) &&
                (created.st_mode & 0777) == 0600 && created.st_size == 0,
            "new trace is not an empty private regular file");
    for (int fd = 0; fd < 256; ++fd) {
      struct stat candidate{};
      if (::fstat(fd, &candidate) == 0 && candidate.st_dev == created.st_dev &&
          candidate.st_ino == created.st_ino) {
        require(openedFD == -1, "trace file has multiple open descriptors");
        openedFD = fd;
      }
    }
    require(openedFD >= 0 && (::fcntl(openedFD, F_GETFD) & FD_CLOEXEC),
            "trace descriptor can escape through exec()");
    require((::fcntl(openedFD, F_GETFL) & O_ACCMODE) == O_WRONLY,
            "trace descriptor has unexpected access mode");
    rejects<std::system_error>([] { (void)FlashRequestCommandTrace::fromEnvironment(); },
                              "concurrent trace source replaced an existing inode");
  }
  require(::fcntl(openedFD, F_GETFD) == -1 && errno == EBADF,
          "trace destructor did not close its own descriptor");
  const auto existing = temporary.path / "existing.jsonl";
  constexpr std::string_view sentinel = "preserve existing trace and source\n";
  { std::ofstream stream(existing); stream << sentinel; }
  require(::chmod(existing.c_str(), 0640) == 0, "could not set sentinel permissions");
  struct stat before{};
  require(::stat(existing.c_str(), &before) == 0, "cannot stat existing fixture");
  environment(existing.c_str());
  ++fileCases;
  rejects<std::system_error>([] { (void)FlashRequestCommandTrace::fromEnvironment(); },
                            "existing trace path was overwritten");
  struct stat after{};
  require(::stat(existing.c_str(), &after) == 0 && before.st_ino == after.st_ino &&
              before.st_size == after.st_size && before.st_mode == after.st_mode,
          "existing trace inode, size or permissions changed");
  { std::ifstream stream(existing); const std::string text((std::istreambuf_iterator<char>(stream)), {});
    require(text == sentinel, "existing trace bytes changed"); }
  const auto symlink = temporary.path / "linked.jsonl";
  require(::symlink(existing.c_str(), symlink.c_str()) == 0, "cannot create symlink fixture");
  environment(symlink.c_str());
  ++fileCases;
  rejects<std::system_error>([] { (void)FlashRequestCommandTrace::fromEnvironment(); },
                            "symlink trace destination followed or replaced");
  require(std::filesystem::is_symlink(symlink), "trace creation replaced a symlink");
  const auto missingParent = temporary.path / "missing-parent" / "trace.jsonl";
  environment(missingParent.c_str());
  ++fileCases;
  rejects<std::system_error>([] { (void)FlashRequestCommandTrace::fromEnvironment(); },
                            "trace silently created missing parent directories");
  require(!std::filesystem::exists(missingParent.parent_path()),
          "rejected destination created parent directories");
}
} // namespace

int main() {
  @autoreleasepool {
    try {
      clockContract();
      metadataContract();
      fileContract();
      std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
                << ",\"clock_cases\":" << clockCases << ",\"parsed_json_records\":" << parsedRecords
                << ",\"file_cases\":" << fileCases
                << ",\"uptime_seconds\":1000000,\"json_delay_tolerance_ns\":1"
                << ",\"metal_backend_constructions\":0,\"gpu_commands\":0}\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "flash_request_command_trace_test: " << error.what() << '\n';
      return 1;
    }
  }
}
