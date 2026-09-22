#include "engine/PrefillTrace.hpp"
#include "metal/ProfilingJson.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace {
void require(bool value,const char *message) {
  if(!value)throw std::runtime_error(message);
}
struct Grouping final : std::numpunct<char> {
  char do_thousands_sep() const override { return ':'; }
  std::string do_grouping() const override { return "\3"; }
};
void testEnvironment() {
  using namespace splash::engine;
  ::unsetenv("SPLASH_PREFILL_TRACE_FILE");
  ::setenv("SPLASH_PREFILL_TRACE_GPU_MODE","stage",1);
  require(NativePrefillTraceOptions::fromEnvironment().file.empty(),
          "GPU mode alone enabled trace");
  ::setenv("SPLASH_PREFILL_TRACE_FILE","",1);
  require(NativePrefillTraceOptions::fromEnvironment().file.empty(),
          "empty FILE enabled trace");
  ::setenv("SPLASH_PREFILL_TRACE_FILE","unused-local-fixture-path",1);
  ::unsetenv("SPLASH_PREFILL_TRACE_GPU_MODE");
  auto value=NativePrefillTraceOptions::fromEnvironment();
  require(value.validMode && value.gpuMode==splash::metal::CommandDispatchProfilingMode::Command,
          "FILE alone must choose command");
  for(const auto mode : {"off","command","dispatch","stage"}) {
    ::setenv("SPLASH_PREFILL_TRACE_GPU_MODE",mode,1);
    value=NativePrefillTraceOptions::fromEnvironment();
    require(value.validMode && std::string(splash::metal::commandDispatchProfilingModeName(value.gpuMode))==mode,
            "explicit GPU mode changed");
  }
  ::setenv("SPLASH_PREFILL_TRACE_GPU_MODE","bad-mode",1);
  require(!NativePrefillTraceOptions::fromEnvironment().validMode,
          "invalid mode silently selected fallback");
  ::unsetenv("SPLASH_PREFILL_TRACE_FILE");
  ::unsetenv("SPLASH_PREFILL_TRACE_GPU_MODE");
}
void testJson(bool emit) {
  using namespace splash;
  metal::CommandDispatchProfile profile;
  profile.sequence=1234567;
  profile.mode=metal::CommandDispatchProfilingMode::Command;
  profile.status=metal::CommandDispatchProfileStatus::Complete;
  profile.reason="metadata \"quoted\"\\path\nnext";
  profile.timing.gpuSeconds=.125;
  profile.timing.wallSeconds=std::numeric_limits<double>::infinity();
  profile.hostReadySeconds=15.25;
  profile.preCommitDeviceMemorySample={4.0,4.125,.125,true};
  profile.postCommitDeviceMemorySample={5.0,0.0,0.0,false};
  profile.commandKernelStartSeconds=1.5;
  profile.commandKernelEndSeconds=2.0;
  profile.commandKernelTimingValid=true;
  profile.commitClockBridge.beganSteadySeconds=19.9;
  profile.commitClockBridge.endedSteadySeconds=20.1;
  profile.commitClockBridge.machSeconds=7.0;
  profile.commitClockBridge.steadyMinusMachSeconds=13.0;
  profile.commitClockBridge.uncertaintySeconds=.1;
  profile.commitClockBridge.machAbsoluteTimestamp=7'000'000'000ULL;
  profile.commitClockBridge.timebaseNumer=1;
  profile.commitClockBridge.timebaseDenom=1;
  profile.commitClockBridge.valid=true;
  profile.dispatchCount=9001;
  profile.dispatchMetadataTruncated=true;
  profile.dispatches.resize(4200);
  auto &dispatch=profile.dispatches.front();
  dispatch.gpuStartTimestamp=UINT64_MAX;
  dispatch.pipelineName="metadata_pipeline";
  dispatch.threadgroups={1234567,2,1};
  dispatch.timestampsValid=false;
  dispatch.calibratedStartSeconds=std::numeric_limits<double>::quiet_NaN();
  dispatch.bindings.resize(35);
  std::ostringstream out;
  const std::locale grouped(std::locale::classic(),new Grouping);
  out.imbue(grouped);out << std::hex << std::showbase << std::showpos << std::setw(13);
  const auto flags=out.flags();const auto width=out.width();
  profiling::writeJson(out,profile);
  require(out.flags()==flags && out.width()==width && out.getloc()==grouped,
          "JSON serializer changed caller formatting");
  const auto json=out.str();
  require(json.find("\"sequence\":1234567")!=std::string::npos,
          "JSON inherited hex/grouping/showpos formatting");
  require(json.find("\"wall_seconds\":null")!=std::string::npos,
          "nonfinite duration was not null");
  require(json.find("\"dispatches_total\":9001")!=std::string::npos &&
          json.find("\"dispatches_emitted\":4096")!=std::string::npos,
          "true dispatch counts or bounded output lost");
  require(json.find("\"bindings_total\":35,\"bindings_truncated\":true")!=std::string::npos,
          "binding truncation not explicit");
  require(json.find("18446744073709551615")!=std::string::npos,
          "GPU timestamp integer precision lost");
  if (emit) std::cout << json << '\n';
  model::ModelPhaseProfile phase;
  phase.commandSequence=1234567;phase.lanes=8;phase.rows=32;
  phase.beganSteadySeconds=std::numeric_limits<double>::quiet_NaN();
  phase.logicalBegin[0]=10;phase.logicalEnd[0]=42;
  if (emit) { profiling::writeJson(std::cout,phase);std::cout << '\n'; }
  metal::CommandDispatchProfilingCapability capability;
  capability.reason="unsupported \"counter\"";
  if (emit) { profiling::writeJson(std::cout,capability);std::cout << '\n'; }
}
}
int main(int argc,char **argv) {
  try {
    const bool emit=argc==2&&std::string_view(argv[1])=="--emit-json";
    testEnvironment();testJson(emit);
    if(!emit)std::cout << "profiling JSON/environment tests passed\n";
    return EXIT_SUCCESS;
  }
  catch(const std::exception &error){std::cerr << error.what() << '\n';return EXIT_FAILURE;}
}
