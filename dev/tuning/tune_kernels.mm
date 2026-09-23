// Offline kernel measurement. Loads an installed model, measures every
// precompiled operator candidate against the policy default in runtime/ops
// through the production encoders, and prints one line per key: the winner
// with its paired GPU/wall gain, or "default kept". With --candidates every
// timed candidate is listed, so a policy rule can be judged by what it costs.
// --confirm times the complete prefill/decode graphs, defaults versus winners.
#include "engine/Bootstrap.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/ModelFactory.hpp"
#include "tuning/AttentionTuning.hpp"
#include "tuning/DraftAttentionTuning.hpp"
#include "tuning/LinearTuning.hpp"
#include "tuning/MoeTuning.hpp"
#include "tuning/TuningWorkloads.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
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
    "                    [--confirm [PAIRS]] [--candidates]\n"
    "  --seconds  wall budget per operator key (default 10; attention gets 4x)\n"
    "  --pairs    paired samples per candidate, 12..64 (default 12)\n"
    "  --confirm  also time the complete prefill/decode graphs, defaults vs\n"
    "             winners, with PAIRS pairs each (default 12, the minimum)\n"
    "  --candidates  after each Linear key, list every timed candidate with its\n"
    "             median GPU/wall gain over the default, best first\n";

volatile std::sig_atomic_t interrupted = 0;
void stopSignal(int) { interrupted = 1; }

struct Options final {
  MeasurementOptions measurement;
  std::optional<size_t> confirmPairs;
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
    } else if (option == "--confirm") {
      options.confirmPairs =
          hasValue ? pairCount(argv[++i], option) : kMinPairedSamples;
    } else {
      throw std::invalid_argument("unknown option or missing value: " + std::string(option) +
                                  "\n" + std::string(kUsage));
    }
  }
  return options;
}

uint64_t packageBytes(const std::filesystem::path &root) {
  uint64_t total = 0;
  for (const std::string_view directory : {"target", "draft", "vision"}) {
    for (const auto &entry : std::filesystem::recursive_directory_iterator(root / directory)) {
      if (!entry.is_regular_file()) continue;
      constexpr uint64_t padding = model::kWeightFileAlignment - 1;
      total += (entry.file_size() + padding) & ~padding;
    }
  }
  if (!total) throw std::invalid_argument("empty model package");
  return total;
}

std::string_view name(LinearTile tile) {
  switch (tile) {
  case LinearTile::N128: return "LinearTile::N128";
  case LinearTile::N256: return "LinearTile::N256";
  case LinearTile::Paired128: return "LinearTile::Paired128";
  case LinearTile::Simdgroup: return "Simdgroup";
  case LinearTile::Split32: return "LinearTile::Split32";
  case LinearTile::Split64: return "LinearTile::Split64";
  case LinearTile::Paired256: return "LinearTile::Paired256";
  case LinearTile::GgufStaged: return "LinearTile::GgufStaged";
  case LinearTile::GgufSimdgroup: return "LinearTile::GgufSimdgroup";
  }
  return "LinearTile::N128";
}
std::string_view name(LinearPhase phase) {
  return phase == LinearPhase::Prefill ? "LinearPhase::Prefill" : "LinearPhase::Decode";
}
std::string_view name(LinearEpilogue epilogue) {
  switch (epilogue) {
  case LinearEpilogue::None: return "LinearEpilogue::None";
  case LinearEpilogue::Residual: return "LinearEpilogue::Residual";
  case LinearEpilogue::GateUp: return "LinearEpilogue::GateUp";
  case LinearEpilogue::UpWithGate: return "LinearEpilogue::UpWithGate";
  }
  return "LinearEpilogue::None";
}
std::string_view name(LinearSimdgroups groups) {
  return groups == LinearSimdgroups::Two    ? "LinearSimdgroups::Two"
         : groups == LinearSimdgroups::Four ? "LinearSimdgroups::Four"
                                            : "LinearSimdgroups::Eight";
}
std::string_view name(AttentionScalePlacement placement) {
  return placement == AttentionScalePlacement::Softmax ? "AttentionScalePlacement::Softmax"
                                                       : "AttentionScalePlacement::Cooperative";
}
std::string_view name(PrefillSplitMultiplier value) {
  return value == PrefillSplitMultiplier::One ? "PrefillSplitMultiplier::One"
                                              : "PrefillSplitMultiplier::Two";
}
std::string name(VerifySplitCount value) {
  return "VerifySplitCount(" + std::to_string(static_cast<uint32_t>(value)) + ")";
}
std::string_view name(MoePhase phase) {
  return phase == MoePhase::Prefill ? "MoePhase::Prefill" : "MoePhase::Decode";
}
std::string name(MoeExpertTile tile) {
  return "MoeExpertTile::M" + std::to_string(static_cast<uint32_t>(tile));
}

std::string describe(const LinearConfig &c) {
  std::ostringstream out;
  out << "{" << name(c.tile) << ", " << c.groups << ", " << name(c.simdgroups) << ", " << c.splits << "}";
  return out.str();
}
std::string describe(const PrefillAttentionConfig &c) {
  return "{" + std::string(name(c.splitMultiplier)) + ", " + std::string(name(c.scalePlacement)) + "}";
}
std::string describe(const VerifyAttentionConfig &c) {
  return "{" + name(c.splitCount) + ", " + std::string(name(c.scalePlacement)) + "}";
}
std::string describe(const DraftAttentionConfiguration &c) {
  return "{" + (c.groups ? std::to_string(c.groups) : std::string("full")) + "}";
}
std::string describe(const MoeConfig &c) { return "{" + name(c.expertTile) + "}"; }

std::string describe(const AttentionShape &s) {
  std::ostringstream out;
  out << "{" << s.queryHeads << ", " << s.kvHeads << ", " << s.headDimension << "}";
  return out.str();
}
std::string describe(const LinearWorkload &w) {
  std::ostringstream out;
  out << "{{" << w.matrix.outputSize << ", " << w.matrix.inputSize << "}, " << w.rows << ", "
      << name(w.phase) << ", " << name(w.epilogue) << "}";
  return out.str();
}
std::string describe(const PrefillAttentionPolicy &w) {
  return "{" + describe(w.shape) + ", " + std::to_string(w.rows) + "}";
}
std::string describe(const VerifyAttentionPolicy &w) {
  return "{" + describe(w.shape) + ", " + std::to_string(w.lanes) + "}";
}
std::string describe(const DraftAttentionWorkload &w) {
  const auto &s = w.shape;
  std::ostringstream out;
  out << "{{" << s.hiddenSize << ", " << s.dynamicSize << ", " << s.qkvSize << ", "
      << s.attentionSize << ", " << s.queryHeads << ", " << s.kvHeads << ", "
      << s.headDimension << "}, " << w.lanes << "}";
  return out.str();
}
std::string describe(const MoeWorkload &w) {
  const auto &s = w.shape;
  std::ostringstream out;
  out << "{{" << s.hiddenSize << ", " << s.experts << ", " << s.expertsPerToken << ", "
      << s.expertIntermediateSize << "}, " << w.rows << ", " << name(w.phase) << "}";
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

template <class Plans, class Config>
std::optional<CandidateId> candidateOf(const Plans &plans, const Config &config) {
  for (size_t index = 0; index < plans.size(); ++index) {
    if constexpr (requires { plans[index].configuration(); }) {
      if (plans[index].configuration() == config) return CandidateId{uint32_t(index)};
    } else if constexpr (requires { plans[index].config(); }) {
      if (plans[index].config() == config) return CandidateId{uint32_t(index)};
    } else {
      if (plans[index] == config) return CandidateId{uint32_t(index)};
    }
  }
  return std::nullopt;
}

void outcome(std::string_view family, const std::string &workload, bool complete,
             bool changed, const std::string &chosen, const std::string &proof,
             std::exception_ptr failure) {
  std::cout << "  " << std::left << std::setw(18) << family << workload << "\n    ";
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

struct Confirmation final {
  std::string graph;
  TimingAssessment gpu;
  TimingAssessment wall;
};

// Complete-graph timing, defaults versus winners, through the production
// bootstrap. Arenas are sized for both because the winners are supplied as
// the creation-time choices; only those two tables are ever installed.
std::vector<Confirmation> confirm(const std::filesystem::path &metallib,
                                  const std::filesystem::path &modelRoot,
                                  const OperatorChoices &winners, size_t pairs) {
  engine::RuntimeBootstrapConfig config;
  config.resources.metallibPath = metallib;
  config.resources.modelRoot = modelRoot;
  config.resources.model = model::inspectModelPackage(modelRoot);
  config.resources.buildId = SPLASH_BUILD_ID;
  config.resources.operatorChoices = winners;
  const auto &capabilities = config.resources.model.capabilities;
  const uint32_t maskWordsPerToken = (capabilities.vocabularySize + 31) / 32;
  config.nativeLoop.maskWordsPerToken = maskWordsPerToken;
  config.protocolLimits.maxTokenBatch = model::ExecutionLimits::maximumStepTokens;
  config.protocolLimits.maxSimulationTokens = capabilities.draftQueryRows;
  config.protocolLimits.maxMaskWords = maskWordsPerToken * (capabilities.draftQueryRows + 1);
  auto bootstrap = engine::RuntimeBootstrap::start(
      std::move(config), [](std::span<const uint8_t>) {},
      []() -> std::string { throw std::logic_error("no status requests during tuning"); });
  if (!bootstrap->report().ready)
    throw std::runtime_error("production bootstrap did not reach Ready");
  auto &resources = bootstrap->resources();
  auto &runtime = bootstrap->modelRuntime();
  const OperatorChoices &choices = winners;
  if (choices.empty()) throw std::runtime_error("no kernel choices to confirm");

  std::vector<Confirmation> results;
  auto measure = [&](const std::string &graph, auto &&run) {
    std::vector<PairedTiming> gpu, wall;
    for (size_t pair = 0; pair < pairs && !interrupted; ++pair) {
      const MeasurementOrder order = measurementOrder(pair);
      double gpuSeconds[2]{};
      std::array<model::WarmupStepResult, 2> steps;
      for (int slot = 0; slot < 2; ++slot) {
        const bool baseline = (slot == 0) == (order == MeasurementOrder::BaselineFirst);
        resources.installOperatorChoices(baseline ? OperatorChoices{} : choices);
        auto [g, step] = run();
        gpuSeconds[baseline ? 0 : 1] = g;
        steps[baseline ? 0 : 1] = std::move(step);
      }
      try {
        const auto [gpuPair, wallPair] = pairWarmupMeasurements(
            steps[0], gpuSeconds[0], steps[1], gpuSeconds[1], order);
        gpu.push_back(gpuPair);
        wall.push_back(wallPair);
      } catch (const std::runtime_error &error) {
        throw std::runtime_error(graph + ": " + error.what());
      }
    }
    resources.installOperatorChoices(choices);
    results.push_back({graph, evaluate(gpu), evaluate(wall)});
  };
  measure("prefill 2048 rows", [&] {
    auto step = runtime.warmupPrefill(model::ExecutionLimits::prefillTokenBudget);
    return std::pair{runtime.telemetry().lastPrefillGpuSeconds, std::move(step)};
  });
  for (uint32_t width = 1; width <= model::ExecutionLimits::maximumBatchWidth; ++width) {
    measure("decode B" + std::to_string(width), [&] {
      auto step = runtime.warmupDecodeBatch(width);
      return std::pair{runtime.telemetry().lastDecodeGpuSeconds, std::move(step)};
    });
  }
  return results;
}

void printAssessment(std::string_view label, const TimingAssessment &value) {
  std::cout << "  " << label << ' ' << std::fixed << std::setprecision(2)
            << value.baselineMedianSeconds * 1000 << " -> "
            << value.candidateMedianSeconds * 1000 << " ms, "
            << timingVerdictName(value.verdict);
  if (value.verdict == TimingVerdict::Improved || value.verdict == TimingVerdict::Stable ||
      value.verdict == TimingVerdict::Uncertain || value.verdict == TimingVerdict::Regressed)
    std::cout << " (median gain " << percent(value.medianPairedGain) << ")";
  std::cout << '\n';
}

std::string today() {
  char buffer[32];
  std::time_t now = std::time(nullptr);
  std::strftime(buffer, sizeof buffer, "%Y-%m-%d", std::localtime(&now));
  return buffer;
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

      OperatorChoices choices;
      uint32_t measured = 0, changed = 0, incomplete = 0, failed = 0;
      std::string deviceName, modelName;
      uint32_t gpuFamily = 0;
      {
      metal::MetalBackend backend(metallib.string());
      const auto &device = backend.capabilities();
      deviceName = device.deviceName;
      gpuFamily = device.appleGpuFamily;
      if (const auto error = device.validationError()) throw std::runtime_error(*error);
      const uint64_t margin =
          engine::EngineMemoryPolicy::workingSetMarginBytes(device.recommendedMaxWorkingSetBytes);
      if (device.recommendedMaxWorkingSetBytes <= margin)
        throw std::runtime_error("device working set does not cover its protected margin");
      engine::MemoryGovernor governor(
          backend, device.recommendedMaxWorkingSetBytes - margin,
          engine::EngineMemoryPolicy::hostAvailableReserveBytes(device.physicalMemoryBytes));
      const MeasurementStop underPressure = [&] {
        const auto state = governor.snapshot();
        return state.pressure != engine::MemoryPressure::Normal || !state.growthAllowed ||
               NSProcessInfo.processInfo.thermalState >= NSProcessInfoThermalStateSerious;
      };
      const MeasurementStop stop = [] { return interrupted != 0; };
      const auto governed = governor.allocationAdmission();
      const metal::AllocationAdmission admit = [&](uint64_t bytes, const auto &allocate) {
        return !interrupted && !underPressure() && governed(bytes, allocate);
      };

      std::optional<model::ModelPackage> package;
      if (!admit(packageBytes(modelRoot),
                 [&] { package.emplace(model::loadModelPackage(backend, modelRoot)); }))
        throw std::runtime_error("model package memory admission denied or interrupted");
      const auto workloads =
          model::collectTuningWorkloads(*package, kPrefillProbeRows, kDecodeProbeWidths);
      modelName = package->name();

      std::cout << "tune-kernels: " << device.deviceName << " (Apple GPU family "
                << device.appleGpuFamily << "), model " << package->name() << ", build "
                << SPLASH_BUILD_ID << "\n  " << options.measurement.samplePairs
                << " pairs per candidate, " << options.measurement.maximumWallSeconds
                << " s per key (attention " << options.measurement.maximumWallSeconds * 4
                << " s per policy)\n\n";

      MeasurementOptions attention = options.measurement;
      attention.maximumWallSeconds = options.measurement.maximumWallSeconds * 4;
      auto account = [&](bool complete, bool didChange, std::exception_ptr failure) {
        ++measured;
        changed += complete && didChange;
        incomplete += !complete && !failure;
        failed += bool(failure);
        if (!backend.healthy()) throw std::runtime_error("Metal backend became unhealthy");
      };

      const Q4Linear linear(device);
      const auto verifyBaseline = VerifyAttentionConfig{};
      for (const auto &input : workloads.linear) {
        if (interrupted) break;
        const auto result = tuneLinear(backend, admit, input, options.measurement, underPressure, stop);
        const auto baseline = linear.plan(input.workload).configuration();
        const bool didChange = result.complete && result.choice.configuration != baseline;
        if (didChange) choices.linear.push_back(result.choice);
        outcome("linear", describe(input.workload), result.complete, didChange,
                describe(result.choice.configuration),
                evidence(result.measurements,
                         candidateOf(linear.candidates(input.workload), result.choice.configuration)),
                result.failure);
        if (options.candidates) {
          // Every candidate's own paired evidence against the default, so a
          // policy rule can be judged by what it costs, not only by who won.
          const auto plans = linear.candidates(input.workload);
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
        account(result.complete, didChange, result.failure);
      }
      if (!interrupted) {
        const PrefillAttentionPolicy policy{workloads.targetAttention};
        const auto result = tunePrefillAttentionPolicy(backend, admit, policy, attention, underPressure, stop);
        const bool didChange = result.complete && result.choice.configuration != PrefillAttentionConfig{};
        if (didChange) choices.prefillAttention.push_back(result.choice);
        std::string proof;
        for (const auto &probe : result.probes)
          proof += "\n      history " + std::to_string(probe.choice.workload.historyTokens) +
                   evidence(probe.measurements,
                            candidateOf(PagedAttention::prefillCandidates(), result.choice.configuration));
        outcome("prefill attention", describe(policy), result.complete, didChange,
                describe(result.choice.configuration), proof, result.failure);
        account(result.complete, didChange, result.failure);
      }
      for (uint32_t width : kDecodeProbeWidths) {
        if (interrupted) break;
        const VerifyAttentionPolicy policy{workloads.targetAttention, width};
        const auto result = tuneVerifyAttentionPolicy(backend, admit, policy, attention, underPressure, stop);
        const bool didChange = result.complete && result.choice.configuration != verifyBaseline;
        if (didChange) choices.verifyAttention.push_back(result.choice);
        std::string proof;
        for (const auto &probe : result.probes)
          proof += "\n      probe" + evidence(probe.measurements,
                       candidateOf(verifyAttentionTuningCandidates(verifyBaseline),
                                   result.choice.configuration));
        outcome("verify attention", describe(policy), result.complete, didChange,
                describe(result.choice.configuration), proof, result.failure);
        account(result.complete, didChange, result.failure);
      }
      for (uint32_t width : kDecodeProbeWidths) {
        if (interrupted) break;
        const DraftAttentionWorkload workload{workloads.draftAttention, width};
        const auto result = tuneDraftAttention(backend, admit, workload, options.measurement, underPressure, stop);
        const bool didChange = result.complete && result.choice.configuration != DraftAttentionConfiguration{};
        if (didChange) choices.draftAttention.push_back(result.choice);
        outcome("draft attention", describe(workload), result.complete, didChange,
                describe(result.choice.configuration),
                evidence(result.measurements,
                         candidateOf(DraftAttention::candidates(workload.shape), result.choice.configuration)),
                result.failure);
        account(result.complete, didChange, result.failure);
      }
      for (const auto &input : workloads.moe) {
        if (interrupted) break;
        const auto result = tuneMoe(backend, admit, input, options.measurement, underPressure, stop);
        const auto candidates = ExecutionPlans(backend.capabilities()).moeCandidates(input.workload);
        const MoeConfig baseline = candidates.front().config();
        const bool didChange = result.complete && result.choice.configuration != baseline;
        if (didChange) choices.moe.push_back(result.choice);
        outcome("moe", describe(input.workload), result.complete, didChange,
                describe(result.choice.configuration),
                evidence(result.measurements, candidateOf(candidates, result.choice.configuration)),
                result.failure);
        account(result.complete, didChange, result.failure);
      }

      std::cout << "\nmeasured " << measured << " keys: " << changed << " changed, "
                << incomplete << " incomplete, " << failed << " failed"
                << (interrupted ? ", interrupted" : "") << "\n\n";
      }  // sweep scope: release the model, fixtures and backend before confirmation

      std::cout << "// Apple GPU family " << gpuFamily << " (" << deviceName
                << "), " << modelName
                << ", tune-kernels " << today() << ", build " << SPLASH_BUILD_ID << "\n"
                << "// " << changed << " of " << measured
                << " keys have a candidate that beat the policy default; if a margin\n"
                << "// matters, change the rules in runtime/ops, not a table.\n";
      for (const auto &c : choices.linear)
        std::cout << "choices.linear.push_back({" << describe(c.workload) << ", "
                  << describe(c.configuration) << "});\n";
      for (const auto &c : choices.prefillAttention)
        std::cout << "choices.prefillAttention.push_back({" << describe(c.workload) << ", "
                  << describe(c.configuration) << "});\n";
      for (const auto &c : choices.verifyAttention)
        std::cout << "choices.verifyAttention.push_back({" << describe(c.workload) << ", "
                  << describe(c.configuration) << "});\n";
      for (const auto &c : choices.draftAttention)
        std::cout << "choices.draftAttention.push_back({" << describe(c.workload) << ", "
                  << describe(c.configuration) << "});\n";
      for (const auto &c : choices.moe)
        std::cout << "choices.moe.push_back({" << describe(c.workload) << ", "
                  << describe(c.configuration) << "});\n";
      if (choices.empty()) std::cout << "// (no entry: every measured key kept its default)\n";

      if (options.confirmPairs && !interrupted && !choices.empty()) {
        std::cout << "confirming complete graphs (" << *options.confirmPairs
                  << " pairs each; defaults -> winners)...\n";
        for (const auto &row : confirm(metallib, modelRoot, choices, *options.confirmPairs)) {
          std::cout << row.graph << '\n';
          printAssessment("GPU ", row.gpu);
          printAssessment("wall", row.wall);
        }
        std::cout << '\n';
      }

      return failed || interrupted ? 1 : 0;
    } catch (const std::exception &error) {
      std::cerr << "tune-kernels: " << error.what() << '\n';
      return 1;
    }
  }
}
