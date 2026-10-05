// Offline kernel measurement. Loads an installed model, measures each
// projection key's tuning candidates (tuning::linearCandidates) against the
// policy default in runtime/ops through the production encoders, and prints
// one line per key: the winner with its paired GPU/wall gain, or "default
// kept". With --candidates every timed candidate is listed, so a policy rule
// can be judged by what it costs.
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/ModelFactory.hpp"
#include "tuning/LinearTuning.hpp"
#include "tuning/TuningWorkloads.hpp"

#import <Foundation/Foundation.h>

#include <charconv>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef SPLASH_BUILD_ID
#error "tune-kernels requires the generated build identity"
#endif

namespace {
using namespace splash;
using namespace splash::ops;
using namespace splash::ops::tuning;

constexpr std::string_view kUsage =
    "usage: tune-kernels METALLIB MODEL_ROOT [--seconds PER_KEY] [--pairs N]\n"
    "                    [--candidates]\n"
    "  --seconds  wall budget per operator key (default 10)\n"
    "  --pairs    paired samples per candidate, 12..64 (default 12)\n"
    "  --candidates  after each Linear key, list every timed candidate with its\n"
    "             median GPU/wall gain over the default, best first\n";

volatile std::sig_atomic_t interrupted = 0;
void stopSignal(int) { interrupted = 1; }

struct Options final {
  MeasurementOptions measurement;
  bool candidates = false;
};

double positiveNumber(std::string_view value, std::string_view option) {
  double result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
      !std::isfinite(result) || result <= 0)
    throw std::invalid_argument(std::string(option) + " requires a positive number");
  return result;
}

size_t pairCount(std::string_view value, std::string_view option) {
  size_t result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
      result < kMinPairedSamples || result > kMaxPairedSamples)
    throw std::invalid_argument(std::string(option) + " requires an integer between 12 and 64");
  return result;
}

Options parse(int argc, char **argv) {
  if (argc < 3) throw std::invalid_argument(std::string(kUsage));
  Options options;
  options.measurement.maximumWallSeconds = 10;
  for (int i = 3; i < argc; ++i) {
    const std::string_view option = argv[i];
    const bool hasValue = i + 1 < argc && argv[i + 1][0] != '-';
    if (option == "--seconds" && hasValue) {
      options.measurement.maximumWallSeconds = positiveNumber(argv[++i], option);
    } else if (option == "--pairs" && hasValue) {
      options.measurement.samplePairs = pairCount(argv[++i], option);
    } else if (option == "--candidates") {
      options.candidates = true;
    } else {
      throw std::invalid_argument("unknown option or missing value: " + std::string(option) +
                                  "\n" + std::string(kUsage));
    }
  }
  return options;
}

// An enumerator prints as the token of its own case label: a rename changes
// both, and -Wswitch catches a new one.
#define ENUMERATOR_NAME(enumerator) \
  case enumerator:                  \
    return #enumerator
[[noreturn]] void unnamed() { throw std::logic_error("value outside its enumeration"); }

std::string_view name(LinearTile tile) {
  switch (tile) {
    ENUMERATOR_NAME(LinearTile::N128);
    ENUMERATOR_NAME(LinearTile::N256);
    ENUMERATOR_NAME(LinearTile::Paired128);
    ENUMERATOR_NAME(LinearTile::Split128);
    ENUMERATOR_NAME(LinearTile::Paired256);
    ENUMERATOR_NAME(LinearTile::Q4Register);
    ENUMERATOR_NAME(LinearTile::GgufStaged);
    ENUMERATOR_NAME(LinearTile::GgufPrefill);
    ENUMERATOR_NAME(LinearTile::GgufRegister);
  }
  unnamed();
}
std::string_view name(LinearPhase phase) {
  switch (phase) {
    ENUMERATOR_NAME(LinearPhase::Prefill);
    ENUMERATOR_NAME(LinearPhase::Decode);
  }
  unnamed();
}
std::string_view name(LinearEpilogue epilogue) {
  switch (epilogue) {
    ENUMERATOR_NAME(LinearEpilogue::None);
    ENUMERATOR_NAME(LinearEpilogue::Residual);
    ENUMERATOR_NAME(LinearEpilogue::GateUp);
    ENUMERATOR_NAME(LinearEpilogue::UpWithGate);
  }
  unnamed();
}
std::string_view name(LinearSimdgroups groups) {
  switch (groups) {
    ENUMERATOR_NAME(LinearSimdgroups::Four);
    ENUMERATOR_NAME(LinearSimdgroups::Eight);
  }
  unnamed();
}
#undef ENUMERATOR_NAME

std::string describe(const LinearConfig &c) {
  std::ostringstream out;
  out << "{" << name(c.tile) << ", " << c.groups << ", " << name(c.simdgroups) << ", " << c.splits << "}";
  return out.str();
}

std::string describe(const LinearWorkload &w) {
  std::ostringstream out;
  out << "{{" << w.matrix.outputSize << ", " << w.matrix.inputSize << "}, " << w.rows << ", "
      << name(w.phase) << ", " << name(w.epilogue) << "}";
  return out.str();
}

std::string percent(double gain) {
  std::ostringstream out;
  out << std::showpos << std::fixed << std::setprecision(1) << gain * 100 << '%';
  return out.str();
}

// The winner's own paired evidence: median GPU/wall gain over the baseline,
// taken from the completed measurement whose candidate ID selected it.
std::string evidence(std::span<const MeasurementResult> measurements,
                     std::optional<CandidateId> winner) {
  if (!winner) return "";
  for (const auto &m : measurements) {
    if (m.candidate == *winner && m.status == MeasurementStatus::Completed) {
      return "  gpu " + percent(m.gpuAssessment.medianPairedGain) + "  wall " +
             percent(m.wallAssessment.medianPairedGain) + "  (" +
             std::to_string(m.pairCount) + " pairs)";
    }
  }
  return "";
}

std::optional<CandidateId> candidateOf(std::span<const LinearPlan> plans, const LinearConfig &config) {
  for (size_t index = 0; index < plans.size(); ++index)
    if (plans[index].configuration() == config) return CandidateId{uint32_t(index)};
  return std::nullopt;
}

void outcome(const std::string &workload, bool complete, bool changed,
             const std::string &chosen, const std::string &proof,
             std::exception_ptr failure) {
  std::cout << "  " << workload << "\n    ";
  if (failure) {
    try { std::rethrow_exception(failure); }
    catch (const std::exception &error) { std::cout << "FAILED: " << error.what(); }
    catch (...) { std::cout << "FAILED"; }
  } else if (!complete) {
    std::cout << "incomplete (budget, pressure or interrupt); default kept";
  } else if (changed) {
    std::cout << "-> " << chosen << proof;
  } else {
    std::cout << "default kept";
  }
  std::cout << '\n';
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << kUsage;
        return 0;
      }
      const Options options = parse(argc, argv);
      std::signal(SIGINT, stopSignal);
      std::signal(SIGTERM, stopSignal);
      std::cout << std::unitbuf;  // progress lines reach a log as they happen
      const std::filesystem::path metallib = argv[1], modelRoot = argv[2];

      uint32_t measured = 0, changed = 0, incomplete = 0, failed = 0;
      metal::MetalBackend backend(metallib.string());
      const auto &device = backend.capabilities();
      if (const auto error = device.validationError()) throw std::runtime_error(*error);
      const uint64_t budget =
          engine::EngineMemoryPolicy::hardBudgetBytes(device.recommendedMaxWorkingSetBytes, 0);
      if (!budget)
        throw std::runtime_error("device working set does not cover its protected margin");
      engine::MemoryGovernor governor(
          backend, budget,
          engine::EngineMemoryPolicy::hostAvailableReserveBytes(device.physicalMemoryBytes),
          engine::queryHostAvailableMemory, 0);
      const MeasurementStop underPressure = [&] {
        const auto state = governor.snapshot();
        return state.pressure != engine::MemoryPressure::Normal ||
               !state.hostGrowthAllowed || !state.headroomBytes ||
               NSProcessInfo.processInfo.thermalState >= NSProcessInfoThermalStateSerious;
      };
      const MeasurementStop stop = [] { return interrupted != 0; };
      const auto governed = governor.allocationAdmission();
      const metal::AllocationAdmission admit =
          [&](uint64_t bytes, const auto &allocate) -> metal::AllocationResult {
        if (interrupted || underPressure())
          return metal::AllocationFailure::HostPressure;
        return governed(bytes, allocate);
      };

      const auto descriptor = model::inspectModelPackage(modelRoot);
      std::optional<model::ModelPackage> package;
      if (!admit(model::modelWeightBytes(modelRoot, descriptor),
                 [&] { package.emplace(model::loadModelPackage(backend, modelRoot, descriptor)); }))
        throw std::runtime_error("model package memory admission denied or interrupted");
      const auto workloads =
          model::collectTuningWorkloads(*package, kPrefillProbeRows, kDecodeProbeWidths);

      std::cout << "tune-kernels: " << device.deviceName << " (Apple GPU family "
                << device.appleGpuFamily << "), model " << package->name() << ", build "
                << SPLASH_BUILD_ID << "\n  " << options.measurement.samplePairs
                << " pairs per candidate, " << options.measurement.maximumWallSeconds
                << " s per key\n";
      // The workloads keep only Affine64 weights, and a GGUF source prepares
      // every target projection and expert as Block32.
      if (descriptor.targetSource == model::TargetSource::Gguf)
        std::cout << "  GGUF target: its projections and experts follow the device policy; "
                     "only the draft's projections are measured\n";
      std::cout << '\n';

      for (const auto &input : workloads) {
        if (interrupted) break;
        const auto result = tuneLinear(backend, admit, input, options.measurement, underPressure, stop);
        const auto plans = linearCandidates(device, input.workload);
        const auto baseline = plans.front().configuration();
        const bool didChange = result.complete && result.configuration != baseline;
        outcome(describe(input.workload), result.complete, didChange,
                describe(result.configuration),
                evidence(result.measurements, candidateOf(plans, result.configuration)),
                result.failure);
        if (options.candidates) {
          // Every candidate's own paired evidence against the default, so a
          // policy rule can be judged by what it costs, not only by who won.
          std::vector<std::pair<double, std::string>> rows;
          for (const auto &m : result.measurements) {
            if (m.candidate.value >= plans.size()) continue;
            const bool timed = m.status == MeasurementStatus::Completed ||
                               m.status == MeasurementStatus::Rejected;
            rows.emplace_back(timed ? m.gpuAssessment.medianPairedGain : -1.0,
                describe(plans[m.candidate.value].configuration()) +
                (timed ? "  gpu " + percent(m.gpuAssessment.medianPairedGain) + "  wall " +
                             percent(m.wallAssessment.medianPairedGain) + "  " +
                             std::string(timingVerdictName(m.gpuAssessment.verdict))
                       : std::string("  not timed")));
          }
          std::sort(rows.begin(), rows.end(),
                    [](const auto &a, const auto &b) { return a.first > b.first; });
          std::cout << "      default " << describe(baseline) << '\n';
          for (const auto &row : rows) std::cout << "      " << row.second << '\n';
        }
        ++measured;
        changed += didChange;
        incomplete += !result.complete && !result.failure;
        failed += bool(result.failure);
        if (!backend.healthy()) throw std::runtime_error("Metal backend became unhealthy");
      }

      std::cout << "\nmeasured " << measured << " keys: " << changed << " changed, "
                << incomplete << " incomplete, " << failed << " failed"
                << (interrupted ? ", interrupted" : "") << '\n';
      return failed || interrupted ? 1 : 0;
    } catch (const std::exception &error) {
      std::cerr << "tune-kernels: " << error.what() << '\n';
      return 1;
    }
  }
}
