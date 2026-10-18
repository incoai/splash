#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestCheckpoint.hpp"
#include "TestStderr.hpp"
#include "engine/Bootstrap.hpp"
#include "engine/RuntimeResources.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryPlan.hpp"
#include "model/RuntimeArenas.hpp"

#import <Foundation/Foundation.h>

#include <sys/time.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace splash;
using namespace splash::engine;

using splash::test::rejects;
using splash::test::require;

// A small model whose weights load and whose geometry the runtime plans:
// four target and two draft layers of the 2048-wide hidden state a compiled
// draft attention takes, with Qwen3.8-27B's attention and GDN heads and a
// two-block vision tower.
struct PlannedModel final {
  model::Qwen3_8Layout target;
  model::DFlashDraftLayout draft = model::kQwen3_8DraftLayout;
  ops::VisionLayout vision;
};

PlannedModel plannedModel() {
  PlannedModel result;
  model::Qwen3_8Layout &target = result.target;
  target.layers = 4;
  target.hiddenSize = 2048;
  target.vocabularySize = 1024;
  target.intermediateSize = 1024;
  target.hiddenCaptureLayers.fill(target.layers - 1);
  model::DFlashDraftLayout &draft = result.draft;
  draft.layers = 2;
  draft.hiddenSize = target.hiddenSize;
  draft.vocabularySize = target.vocabularySize;
  draft.dynamicSize = 512;
  draft.intermediateSize = 1024;
  draft.targetHiddenSize = target.capturedHiddenSize();
  ops::VisionLayout &vision = result.vision;
  vision.depth = 2;
  vision.hiddenSize = 128;
  vision.patchDimension = 1536;
  vision.intermediateSize = 200;
  vision.paddedIntermediateSize = 256;
  vision.mergedHiddenSize = 512;
  vision.outputHiddenSize = target.hiddenSize;
  vision.heads = 2;
  vision.headDimension = 64;
  vision.positionGridSide = 4;
  return result;
}

// The target and draft checkpoints of `model` under root, which weight
// planning reads; its vision tower stays absent, so loading stops there.
void writeCheckpoints(const std::filesystem::path &root, const model::ModelDescriptor &model) {
  std::visit([&](const auto &target) { test::writeSyntheticCheckpoints(root, target, model.draft); },
             model.target);
}

RuntimeResourcesConfig budgetConfig(const char *metallibPath, const std::filesystem::path &root) {
  const PlannedModel planned = plannedModel();
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor("budget-test", planned.target, planned.draft, planned.vision,
                                            model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  config.buildId = "budget-test";
  writeCheckpoints(root, config.model);
  return config;
}

// A model under root whose weights load, which nothing but the host
// refuses: `name` names its descriptor and build.
RuntimeResourcesConfig loadableConfig(const char *metallibPath, const std::filesystem::path &root,
                                      const std::string &name) {
  const PlannedModel planned = plannedModel();
  static_cast<void>(test::writeSyntheticModel(root, planned.target, planned.draft, planned.vision));
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor(name, planned.target, planned.draft, planned.vision,
                                            model::TargetSource::Safetensors,
                                            model::VisionSource::Safetensors);
  config.model.sourceIdentity = "synthetic";
  config.buildId = name;
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{64 * kGiB, 64 * kGiB}); };
  return config;
}

// A local volume of 1000 GiB with `availableBytes` free, which leaves the
// automatic SSD cache room from 12 GiB on.
DiskVolume localVolume(uint64_t availableBytes) {
  return {"/Volumes/Test", availableBytes, 1000 * kGiB, true};
}

// A volume probe that reads `volume` for every directory.
std::function<std::optional<DiskVolume>(const std::filesystem::path &)>
reading(std::optional<DiskVolume> volume) {
  return [volume](const std::filesystem::path &) { return volume; };
}

// Startup fails at the loader, whose vision weight files are deliberately
// absent. Reaching it is the assertion: everything the engine checks before
// loading the weights let this configuration through.
void requireReachesModelLoader(RuntimeResourcesConfig config,
                               const std::filesystem::path &root,
                               const char *message) {
  try {
    auto resources = RuntimeResources::create(config, 0);
    throw std::runtime_error("a model without weights unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    const std::string_view text = error.what();
    require(error.failure() == RuntimeResourceFailure::Other &&
                error.stage() == RuntimeResourceStage::ModelLoading &&
                text.find(root.string()) != std::string_view::npos,
            message);
  }
}

// Beside its weights a model needs at least the runtime reserves, one lane's
// state and the KV runway.
uint64_t minimumBytes(const RuntimeResourcesConfig &config) {
  return minimumRequiredBytes(model::modelWeightBytes(config.modelRoot, config.model) +
                                  model::kPipelineReserveBytes + model::kRuntimeOverheadReserveBytes,
                              config.model.stateLayout.laneBytes(),
                              config.model.targetKvLayout)
      .value();
}

// A persistent cache's directory is named for what its copies hold: the same
// models and layouts name the same one under any build, and a change to any
// of them, or to the RoPE scaling its keys went through, names another.
void testPersistentCacheNamespace() {
  const model::ModelDescriptor model = model::makeModelDescriptor(
      "namespace-test", model::Qwen3_8Layout{}, model::kQwen3_8DraftLayout,
      model::kQwen3_8VisionLayout, model::TargetSource::Safetensors,
      model::VisionSource::Safetensors);
  const auto identity = [&](char models, std::string_view build, kv::Format format) {
    kv::Layout layout = model.targetKvLayout;
    layout.format = format;
    return makeRuntimeCacheIdentity(std::string(64, models), std::string(64, 'b'), build,
                                    layout, {});
  };
  const std::string name =
      persistentCacheNamespace(identity('a', "build", kv::Format::Int8), model.stateLayout);
  require(name.size() == 32 && name.find_first_not_of("0123456789abcdef") == std::string::npos,
          "a cache namespace is not 32 lowercase hex digits");
  require(persistentCacheNamespace(identity('a', "another build", kv::Format::Int8),
                                   model.stateLayout) == name,
          "the build changed the cache namespace");
  model::CompositeStateLayout states = model.stateLayout, bfloatStates = model.stateLayout;
  ++states.draft.layers;
  bfloatStates.target.format = ops::GdnStateFormat::BFloat16;
  require(persistentCacheNamespace(identity('c', "build", kv::Format::Int8), model.stateLayout) !=
                  name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::BFloat16),
                                       model.stateLayout) != name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), states) != name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), bfloatStates) != name,
          "other models, layouts or state formats shared a cache namespace");
  const auto scaled = [&](double factor) {
    kv::Layout layout = model.targetKvLayout;
    layout.format = kv::Format::Int8;
    return persistentCacheNamespace(makeRuntimeCacheIdentity(std::string(64, 'a'), std::string(64, 'b'), "build",
                                                             layout, {factor, 262144}),
                                    model.stateLayout);
  };
  require(scaled(1.0) == name && scaled(4.0) != name && scaled(2.0) != name && scaled(2.0) != scaled(4.0),
          "YaRN's keys shared a cache namespace with another scaling's");
  // Factors that print alike to six digits still rotate keys differently.
  require(scaled(3.9999999) != scaled(4.0) && scaled(1.0000001) != scaled(1.0000004),
          "factors alike to six digits shared a cache namespace");
}

void testWeightBudgetBeforeLoading(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());

  {
    config.memoryPressure = [] { return MemoryPressure::Critical; };
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("model load ignored system pressure");
    } catch (const RuntimeResourcesError &error) {
      require(error.failure() == RuntimeResourceFailure::HostCapacity,
              "startup pressure did not remain retryable");
      require(std::string_view(error.what()).find("not enough free memory") !=
                  std::string_view::npos,
              "startup pressure reached the weight loader");
    }
  }
  config.memoryPressure = [] { return MemoryPressure::Warning; };

  // The low ceiling is one byte short of the minimum, so admission must
  // charge modelWeightBytes and all a model needs beside its weights. The
  // other ceilings must reach the real loader, whose expected weight files
  // are deliberately absent.
  const uint64_t minimum = minimumBytes(config);
  for (uint64_t ceiling : {minimum - 1, minimum, uint64_t{0}}) {
    config.maximumMemoryBytes = ceiling;
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("a model without weights unexpectedly loaded");
    } catch (const RuntimeResourcesError &error) {
      if (ceiling == minimum - 1) {
        require(error.failure() == RuntimeResourceFailure::EngineCapacity,
                "hard weight budget lost its engine-capacity classification");
        const std::string_view text = error.what();
        require(error.stage() == RuntimeResourceStage::MemoryPlanning &&
                    text.find("require " + std::to_string(minimum) + " bytes") !=
                        std::string_view::npos &&
                    text.find("budget is " + std::to_string(minimum - 1) +
                              " bytes") != std::string_view::npos,
                "weight loading began before checking the memory ceiling");
      } else {
        require(error.failure() == RuntimeResourceFailure::Other,
                "missing model file was misclassified as allocation pressure");
      }
    }
  }
  config.maximumMemoryBytes = 0;
  requireReachesModelLoader(config, root.path(),
                            "a sufficient weight budget did not reach the "
                            "model loader");
}

// A state's write to the disk tier stages through a state-sized buffer the
// plan sets aside only when the tier starts: a quota that holds no state
// leaves the tier off and the budget to KV. An explicit quota's staging is
// planned whatever it costs, and the automatic one never costs a start. The
// notices and the refusal name the tier as users know it, the SSD cache.
void testStateStagingNeedsAStartedTier(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  const uint64_t minimum = minimumBytes(config);
  const uint64_t stateBytes = config.model.stateLayout.cachedBytes();
  config.maximumMemoryBytes = minimum;
  config.cacheDisk = {.bytes = stateBytes - 1};
  const std::string notice = test::capturedStderr([&] {
    requireReachesModelLoader(config, root.path(),
                              "a quota that holds no state set staging aside");
  });
  require(notice.find("SSD cache off: its quota holds no state of " +
                      std::to_string(stateBytes >> 20) + " MiB") != std::string::npos,
          "the notice of a quota that holds no state did not name the SSD cache");

  // The automatic tier lets a budget with no room for its staging start,
  // and says why it is off.
  config.cacheDisk = {.automatic = true};
  const std::string automatic = test::capturedStderr([&] {
    requireReachesModelLoader(config, root.path(),
                              "the automatic tier's staging refused a start");
  });
  require(automatic.find("SSD cache off: the memory budget has no room for its " +
                         std::to_string(stateBytes >> 20) +
                         " MiB of state staging beside the model") != std::string::npos,
          "the automatic tier did not say the memory budget turned it off");

  config.cacheDisk = {.bytes = stateBytes};
  try {
    auto resources = RuntimeResources::create(config, 0);
    throw std::runtime_error("a model without weights unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    require(error.failure() == RuntimeResourceFailure::EngineCapacity &&
                error.stage() == RuntimeResourceStage::MemoryPlanning &&
                std::string_view(error.what())
                        .find("SSD cache state staging require " +
                              std::to_string(minimum + stateBytes) +
                              " bytes") != std::string_view::npos,
            "a started tier's staging was not counted by name before loading");
  }
  config.maximumMemoryBytes = minimum + stateBytes;
  requireReachesModelLoader(config, root.path(),
                            "a budget with room for the staging did not "
                            "reach the model loader");
}

// The image patch limit is a positive multiple of four up to the
// protocol's per-image ceiling; anything else is refused before the backend
// is created.
void testImagePatchCapIsBounded(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  for (const uint32_t patches : {ops::kMaximumImagePatches + 4, 6U}) {
    config.maximumImagePatches = patches;
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("an image patch cap outside the protocol's was accepted");
    } catch (const RuntimeResourcesError &error) {
      require(error.stage() == RuntimeResourceStage::Configuration,
              "an invalid image patch cap was not a configuration error");
    }
  }
  config.maximumImagePatches = 4096;
  requireReachesModelLoader(config, root.path(),
                            "a smaller image patch cap did not reach the model loader");
}

// Startup and the governor sample host memory through the config's probe,
// and the automatic SSD cache its volume through another: without either,
// the config is refused before the backend is created.
void testHostMemoryProbeIsRequired(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  for (const bool memory : {true, false}) {
    RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
    if (memory)
      config.hostMemory = {};
    else
      config.diskVolume = {};
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("resources were assembled without a host memory or volume probe");
    } catch (const RuntimeResourcesError &error) {
      require(error.stage() == RuntimeResourceStage::Configuration,
              memory ? "a missing host memory probe was not a configuration error"
                     : "a missing volume probe was not a configuration error");
    }
  }
}

// The 27B under a budget half a GiB above its weights: the weights alone fit,
// but not with what the runtime needs beside them. Startup refuses it before
// any weight is loaded.
void testModelBeyondBudgetIsRefusedBeforeLoading(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  config.model = model::makeModelDescriptor("budget-test", model::Qwen3_8Layout{}, model::kQwen3_8DraftLayout,
                                            model::kQwen3_8VisionLayout, model::TargetSource::Safetensors,
                                            model::VisionSource::Safetensors);
  writeCheckpoints(root.path(), config.model);
  config.maximumMemoryBytes = model::modelWeightBytes(root.path(), config.model) + kGiB / 2;
  try {
    auto resources = RuntimeResources::create(config, 0);
    throw std::runtime_error("a model that cannot fit unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    require(error.failure() == RuntimeResourceFailure::EngineCapacity &&
                error.stage() == RuntimeResourceStage::MemoryPlanning,
            "a model that cannot fit reached the weight loader");
  }
}

// The rule that keeps users off the startup floor: admission weighs
// reclaimable memory against the macOS reserve, never against the model.
// A model far larger than everything reclaimable still starts, because the
// operation guard checks each image's allocation as loading builds the
// residency up. A quarter of the 27B's layers keeps it within the automatic
// budget of a 24 GB Mac.
void testStartupAdmissionIgnoresModelSize(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  model::Qwen3_8Layout target;
  target.layers = 16;
  target.hiddenCaptureLayers = {1, 4, 7, 10, 13};
  config.model = model::makeModelDescriptor("budget-test", target, model::kQwen3_8DraftLayout,
                                            model::kQwen3_8VisionLayout, model::TargetSource::Safetensors,
                                            model::VisionSource::Safetensors);
  writeCheckpoints(root.path(), config.model);
  require(model::modelWeightBytes(root.path(), config.model) > 3 * kGiB, "the model must exceed the sample");
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{3 * kGiB, 3 * kGiB}); };
  requireReachesModelLoader(config, root.path(),
                            "a model larger than reclaimable host memory "
                            "refused to start");

  // Below the reserve macOS is the one at risk, so startup waits instead.
  // Unmeasurable telemetry waits the same way.
  for (std::optional<uint64_t> available :
       {std::optional<uint64_t>(64 * kMiB), std::optional<uint64_t>()}) {
    config.hostMemory = [available] {
      return available ? std::optional<HostMemory>(HostMemory{*available, *available}) : std::nullopt;
    };
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("model load ignored the macOS reserve");
    } catch (const RuntimeResourcesError &error) {
      require(error.failure() == RuntimeResourceFailure::HostCapacity,
              "exhausted host memory did not remain retryable");
      require(std::string_view(error.what()).find("not enough free memory") !=
                  std::string_view::npos,
              "exhausted host memory reached the weight loader");
    }
  }
}

// The memory plan takes the vision category from what loaded, so a model
// with vision whose loader produced no vision bytes must stop here.
void testLoadedVisionIsRequiredOnlyWithVision() {
  model::LoadedModel loaded;
  loaded.descriptor = model::makeModelDescriptor(
      "loaded-test", model::Qwen3_8Layout{}, model::kQwen3_8DraftLayout,
      model::kQwen3_8VisionLayout, model::TargetSource::Safetensors,
      model::VisionSource::Safetensors);
  model::Qwen3_8Weights target;
  target.actualAllocatedBytes = 1;
  target.manifestFingerprintSha256 = "target";
  loaded.target = std::move(target);
  loaded.draft.actualAllocatedBytes = 1;
  loaded.manifestFingerprintSha256 = "model";
  require(loaded.descriptor.hasVision(), "the test model has no vision");
  rejects([&] { requireLoadedModel(loaded); },
          "loaded model has incomplete allocation accounting",
          "a multimodal model without loaded vision weights was accepted");
  loaded.vision.actualAllocatedBytes = 1;
  requireLoadedModel(loaded);
  loaded.vision.actualAllocatedBytes = 0;
  loaded.descriptor.visionSource = model::VisionSource::None;
  requireLoadedModel(loaded);
}

// Resource assembly names the step it fails at, at its real throw site: a
// host with less than its reserve refuses the start before the weights are
// opened, and the startup retry window retries it. Past the weights the host
// refuses only what loading would (testStartGrowsThroughHostPressure).
void testFailedStepIsNamed(const char *metallibPath) {
  using namespace std::chrono_literals;
  const test::TemporaryDirectory root("splash-assembly");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "assembly-test");
  const auto host = [&](uint64_t available) {
    config.hostMemory = [available] { return std::optional<HostMemory>(HostMemory{available, available}); };
  };
  // Nothing but the host refuses this model. The suite's one complete start
  // also reports the SSD cache a disk quota gives it, by that name.
  host(64 * kGiB);
  config.cacheDisk = {.bytes = kGiB};
  const std::string started =
      test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); });
  require(started.find("SSD cache: up to ") != std::string::npos &&
              started.find(" of temporary files in ") != std::string::npos &&
              started.find("; KV pages of ") != std::string::npos &&
              started.find("--max-cache-disk 0 turns it off.") != std::string::npos,
          "a start did not report its SSD cache by that name");
  config.cacheDisk = {};
  const auto failure = [&](uint64_t available) {
    host(available);
    try {
      static_cast<void>(RuntimeResources::create(config, 0));
    } catch (const RuntimeResourcesError &error) {
      return RuntimeBootstrapError(error).report();
    }
    throw std::runtime_error("resources were assembled on a host short of memory");
  };
  const uint64_t reserve = EngineMemoryPolicy::hostAvailableReserveBytes(
      metal::probeDeviceCapabilities().physicalMemoryBytes);
  const RuntimeBootstrapReport before = failure(reserve / 2);
  require(before.resourceStage == RuntimeResourceStage::ModelLoading &&
              before.resourceFailure == RuntimeResourceFailure::HostCapacity &&
              before.message.find("not enough free memory to start") != std::string::npos,
          "a start the host refused before the weights loaded named another step");
  const auto first = StartupRetryWindow::Clock::time_point{};
  require(StartupRetryWindow(30s).retryUntil(before, first) == first + 30s,
          "a start the host refused was not retried");
}

// What a start allocates, the KV runway and then the arenas and the warmup as
// RuntimeBootstrap::start allocates them, is memory a request in service
// needs: neither the warning margin above the spare reserve nor macOS's
// warning refuses it, only what refuses loading, and an allocation the
// engine marks meanwhile leaves the start's mark in place. Once serving
// begins, growth that no request in service needs keeps to both again.
void testStartGrowsThroughHostPressure(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-start-pressure");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "start-pressure-test");
  const uint64_t reserve = EngineMemoryPolicy::hostAvailableReserveBytes(
      metal::probeDeviceCapabilities().physicalMemoryBytes);
  // A host with its reserve but not the warning margin beyond it, then one
  // with ample room while macOS warns.
  for (const bool warning : {false, true}) {
    const uint64_t available = warning ? 64 * kGiB : reserve + kHostWarningMarginBytes / 2;
    config.hostMemory = [available] {
      return std::optional<HostMemory>(HostMemory{available, available});
    };
    config.memoryPressure = [warning] {
      return warning ? MemoryPressure::Warning : MemoryPressure::Normal;
    };
    const std::string host = warning ? "under macOS's warning" : "short of the warning margin";
    std::unique_ptr<RuntimeResources> resources;
    static_cast<void>(
        test::capturedStderr([&] { resources = RuntimeResources::create(config, 0); }));
    const metal::AllocationAdmission admit = resources->memoryGovernor().allocationAdmission();
    const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
    std::unique_ptr<model::RuntimeModel> runtime;
    require(static_cast<bool>(admit(budget.sharedPrefillBytes + budget.sharedDecodeBytes, [&] {
              runtime = model::createRuntime(resources->modelContext());
            })),
            "the start's arenas waited " + host);
    EngineConfig engine;
    connectToGovernor(engine, resources->memoryGovernor());
    {
      const Serving allocation(engine.serving, true);
    }
    // The warmup's lane state; it throws if refused.
    static_cast<void>(runtime->warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
    static_cast<void>(runtime->warmupDecodeBatch(1));
    resources->beginServing();
    const metal::AllocationResult grown = admit(kMiB, [] {});
    require(!grown && grown.failure == metal::AllocationFailure::HostPressure,
            "growth that no request in service needs went on " + host + " once serving began");
  }
}

// A model served with YaRN starts and runs as any other, with YaRN all the
// way into its runtime: the start names the window it serves, the runtime's
// geometry serves that window, its arenas hold YaRN's rotary frequencies, and
// its prefill and decode build the target's RoPE tables with the attention
// factor.
void testYarnRuntime(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-yarn");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "yarn-test");
  config.model = model::withYarn(std::move(config.model), 4.0);
  std::unique_ptr<RuntimeResources> resources;
  const std::string started =
      test::capturedStderr([&] { resources = RuntimeResources::create(config, 0); });
  require(started.find("YaRN scales the target's RoPE by 4: a window of 1010000 tokens instead of the native "
                       "262144.") != std::string::npos,
          "a start with YaRN did not name the window it serves");
  const model::RuntimeContext context = resources->modelContext();
  const auto geometry = model::RuntimeGeometry::from(context.model, kv::Format::Int8, ops::GdnStateFormat::Float32);
  require(geometry.target.ropeScaling == config.model.ropeScaling && geometry.target.contextTokens() == 1'010'000,
          "the runtime's geometry lost YaRN or its window");
  std::array<float, SPLASH_TARGET_ROPE_PAIRS> expected{};
  model::rotaryInverseFrequencies(geometry.target.rotaryTheta, geometry.target.rotaryPairs,
                                  geometry.target.ropeScaling, expected);
  const metal::AllocationAdmission admit = resources->memoryGovernor().allocationAdmission();
  const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
  std::unique_ptr<model::RuntimeModel> runtime;
  require(static_cast<bool>(admit(budget.sharedPrefillBytes + budget.sharedDecodeBytes, [&] {
            const model::PrefillArena arena(context.backend, geometry, context.operators);
            require(std::memcmp(arena.get(model::PrefillTensor::TargetInverseFrequencies).contents(),
                                expected.data(), sizeof(expected)) == 0,
                    "the prefill arena does not hold YaRN's frequencies");
          })),
          "the YaRN prefill arena was refused");
  require(static_cast<bool>(admit(budget.sharedPrefillBytes + budget.sharedDecodeBytes,
                                  [&] { runtime = model::createRuntime(context); })),
          "the YaRN runtime's arenas were refused");
  static_cast<void>(runtime->warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
  static_cast<void>(runtime->warmupDecodeBatch(1));
}

// The SSD cache holds no write back unless a write budget is set; one that
// is set starts with a quarter of a day's.
void testWriteBudgetOnlyWhenSet(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-write-budget");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "write-budget-test");
  config.cacheDisk = {.bytes = kGiB};
  const auto tier = [&] {
    KvTierSnapshot snapshot;
    static_cast<void>(test::capturedStderr([&] {
      snapshot = RuntimeResources::create(config, 0)->cache().snapshot().kvTier;
    }));
    return snapshot;
  };
  const KvTierSnapshot unset = tier();
  require(unset.capacityBytes == kGiB && !unset.writeBudgetBytes &&
              unset.writesGated == DiskWritesGate::No,
          "the SSD cache had a write budget nobody set");
  config.cacheDiskWritesPerDay = 8 * kGiB;
  const KvTierSnapshot set = tier();
  require(set.writeBudgetBytes == 2 * kGiB && set.writesGated == DiskWritesGate::No,
          "a write budget that was set did not start with a quarter of a day's");
}

// Every start removes the caches under its root that nobody used for two
// weeks and keeps the others, one without --persistent-cache or the SSD
// cache too.
void testStaleCachesGoAtEveryStart(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-stale-model"), caches("splash-stale");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "stale-test");
  config.cacheRoot = caches.path() / "prefix-cache";
  const std::string stale(CacheDirectory::kNamespaceDigits, 'a'),
      fresh(CacheDirectory::kNamespaceDigits, 'b');
  for (const std::string &name : {stale, fresh}) {
    auto directory = CacheDirectory::open(config.cacheRoot, name, std::chrono::milliseconds(0));
    std::ofstream(directory->kvSlots()) << "cache";
  }
  const timeval old[2] = {{0, 0}, {1000000000, 0}};
  for (const char *file : {"lock", "kv.slots"})
    require(::utimes((config.cacheRoot / stale / file).c_str(), old) == 0,
            "a cache file could not be aged");
  static_cast<void>(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); }));
  require(!std::filesystem::exists(config.cacheRoot / stale) &&
              std::filesystem::exists(config.cacheRoot / fresh / "kv.slots"),
          "a start without a persistent cache kept a stale cache under its root, or removed another");
}

// A cache root the engine makes for a persistent cache stays out of Time
// Machine backups; one that was there keeps its own setting.
void testMadeCacheRootStaysOutOfBackups(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-backups-model"), caches("splash-backups");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "backups-test");
  config.cacheDisk = {.bytes = kGiB};
  config.persistentCache = true;
  const auto excluded = [](const std::filesystem::path &path) {
    NSURL *url = [NSURL fileURLWithPath:@(path.c_str()) isDirectory:YES];
    NSNumber *value = nil;
    return [url getResourceValue:&value forKey:NSURLIsExcludedFromBackupKey error:nil] &&
           value.boolValue;
  };
  config.cacheRoot = caches.path() / "made" / "prefix-cache";
  static_cast<void>(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); }));
  require(std::filesystem::is_directory(config.cacheRoot) && excluded(config.cacheRoot) &&
              !excluded(caches.path() / "made"),
          "the cache root the engine made was not excluded from backups, or its parent was");
  config.cacheRoot = caches.path() / "there";
  std::filesystem::create_directory(config.cacheRoot);
  static_cast<void>(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); }));
  require(!excluded(config.cacheRoot), "a cache root that was there was excluded from backups");
}

// The automatic tier keeps its staging only where the plan holds the
// requested context beside it. Asked for the context the plan holds without
// the staging but not with it, a start turns the tier off with one line and
// plans as without it.
void testAutomaticTierYieldsToTheContext(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-automatic");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "automatic-test");
  config.diskVolume = reading(localVolume(500 * kGiB));
  uint64_t minimumRequired = 0;
  const auto start = [&](uint32_t requestedContextTokens, uint32_t &contextTokens,
                         bool &tier) {
    return test::capturedStderr([&] {
      auto resources = RuntimeResources::create(config, requestedContextTokens);
      contextTokens = resources->memoryPlan().maximumContextTokens();
      minimumRequired = resources->memoryPlan().breakdown().minimumRequiredBytes;
      tier = resources->cache().snapshot().kvTier.capacityBytes != 0;
    });
  };
  uint32_t without = 0, with = 0, yielded = 0;
  bool tier = false;
  static_cast<void>(start(0, without, tier));
  // A budget whose KV the staging shortens: the plan's least, the staging
  // and a little KV.
  config.maximumMemoryBytes =
      minimumRequired + config.model.stateLayout.cachedBytes() + 64 * kMiB;
  static_cast<void>(start(0, without, tier));
  require(!tier && without, "a start without the SSD cache had one, or no context");
  config.cacheDisk = {.automatic = true};
  static_cast<void>(start(0, with, tier));
  require(tier && with < without,
          "the automatic tier did not start, or its staging cost the plan no context");
  const std::string off = start(without, yielded, tier);
  require(!tier && yielded == without &&
              off.find("SSD cache off: the memory plan cannot hold a " + std::to_string(without) +
                       "-token context beside its state staging") != std::string::npos,
          "the automatic tier kept staging the requested context needs");
}

// The automatic tier's staging is decided by memory alone, so a start serves
// the same context whatever room the volume its files would go to has: a
// relaunch after something else filled the disk announces the context the
// server serves. The volume decides only whether the files open. A volume
// without room, one that cannot be measured or one that is not local leaves
// the tier off with one line and the staging the plan set aside unused; a
// persistent cache whose root has no room keeps temporary files.
void testAutomaticContextIgnoresTheVolume(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-volume"), caches("splash-volume-root");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "volume-test");
  struct Started {
    uint32_t context = 0, planContext = 0;
    uint64_t staging = 0, minimumRequired = 0;
    bool tier = false, persistent = false;
    std::string log;
  };
  const auto start = [&] {
    Started started;
    started.log = test::capturedStderr([&] {
      auto resources = RuntimeResources::create(config, 0);
      started.context = resources->aneFfnOutcome().context;
      started.planContext = resources->memoryPlan().maximumContextTokens();
      started.staging = resources->memoryPlan().breakdown().stateStagingBytes;
      started.minimumRequired = resources->memoryPlan().breakdown().minimumRequiredBytes;
      const CacheSnapshot cache = resources->cache().snapshot();
      started.tier = cache.kvTier.capacityBytes != 0;
      started.persistent = cache.persistent;
    });
    return started;
  };
  const auto offOnce = [](const Started &started, std::string_view reason) {
    const size_t at = started.log.find("SSD cache off: ");
    return at != std::string::npos && started.log.find("SSD cache off", at + 1) == std::string::npos &&
           started.log.find(reason, at) != std::string::npos &&
           started.log.find("; the state staging the plan set aside stays unused.", at) !=
               std::string::npos;
  };
  // A budget whose KV the staging shortens: the plan's least, the staging
  // and a little KV.
  config.maximumMemoryBytes =
      start().minimumRequired + config.model.stateLayout.cachedBytes() + 64 * kMiB;
  const Started off = start();
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  const Started room = start();
  require(!off.tier && room.tier && room.staging == config.model.stateLayout.cachedBytes() &&
              room.context == room.planContext && room.context < off.context,
          "the automatic tier did not start beside its staging, or the staging cost no context");
  const std::array<std::pair<std::optional<DiskVolume>, std::string>, 3> refusals{{
      {localVolume(11 * kGiB), "11.0 GiB available on /Volumes/Test is too little beyond the 10.0 GiB"},
      {std::nullopt, "its volume cannot be measured"},
      {DiskVolume{"/Volumes/Share", 500 * kGiB, 1000 * kGiB, false}, "/Volumes/Share is not a local volume"},
  }};
  for (const auto &[volume, reason] : refusals) {
    config.diskVolume = reading(volume);
    const Started refused = start();
    require(!refused.tier && refused.staging == room.staging && refused.context == room.context &&
                refused.planContext == room.planContext && offOnce(refused, reason),
            "a volume without room changed the automatic context, or did not leave the tier off "
            "once (" + reason + ")");
  }
  // A persistent cache's root without room leaves the tier temporary files;
  // with no room for those either, it is off.
  config.persistentCache = true;
  config.cacheRoot = caches.path() / "prefix-cache";
  config.diskVolume = [&](const std::filesystem::path &directory) {
    return std::optional(localVolume(directory == config.cacheRoot ? 11 * kGiB : 500 * kGiB));
  };
  const Started temporary = start();
  require(temporary.tier && !temporary.persistent && temporary.context == room.context &&
              temporary.log.find("Persistent cache off: 11.0 GiB available on /Volumes/Test") !=
                  std::string::npos &&
              temporary.log.find("SSD cache off") == std::string::npos,
          "a persistent cache whose root had no room did not keep temporary files");
  config.diskVolume = reading(localVolume(11 * kGiB));
  const Started neither = start();
  require(!neither.tier && neither.context == room.context &&
              offOnce(neither, "for its root, 11.0 GiB available on /Volumes/Test"),
          "a persistent cache with no room anywhere did not leave the tier off once");
}

// The engine asks the governor whether the host pauses growth, and its
// serving mark lets a request in service grow through that pause.
void testEngineFollowsTheGovernor(const char *metallibPath) {
  metal::MetalBackend backend(metallibPath);
  constexpr uint64_t kGiB = 1ULL << 30;
  constexpr uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + 8 * kGiB;
  const metal::MetalMemoryStats memory = backend.memoryStats();
  MemoryGovernor governor(
      backend, std::max(memory.allocatedBytes, memory.deviceCurrentAllocatedBytes) + kGiB,
      hostReserve, hostReserve,
      [&available] { return std::optional<HostMemory>(HostMemory{*available, *available}); }, 0);
  EngineConfig config;
  connectToGovernor(config, governor);
  require(config.growthPaused && config.serving && !config.growthPaused(),
          "the engine was not connected to the governor");
  const auto admits = [admit = governor.allocationAdmission()] {
    return static_cast<bool>(admit(1024, [] {}));
  };
  // Inside the warning margin the host pauses growth that no request in
  // service needs.
  available = hostReserve + kGiB / 2;
  require(config.growthPaused() && !admits(), "the engine did not see the host's pause");
  config.serving(true);
  require(admits(), "the serving mark did not reach the governor");
  config.serving(false);
  require(!admits(), "the serving mark was not cleared");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "expected metallib path");
      testLoadedVisionIsRequiredOnlyWithVision();
      testPersistentCacheNamespace();
      testWeightBudgetBeforeLoading(argv[1]);
      testStateStagingNeedsAStartedTier(argv[1]);
      testImagePatchCapIsBounded(argv[1]);
      testHostMemoryProbeIsRequired(argv[1]);
      testModelBeyondBudgetIsRefusedBeforeLoading(argv[1]);
      testStartupAdmissionIgnoresModelSize(argv[1]);
      testFailedStepIsNamed(argv[1]);
      testStartGrowsThroughHostPressure(argv[1]);
      testYarnRuntime(argv[1]);
      testWriteBudgetOnlyWhenSet(argv[1]);
      testStaleCachesGoAtEveryStart(argv[1]);
      testMadeCacheRootStaysOutOfBackups(argv[1]);
      testAutomaticTierYieldsToTheContext(argv[1]);
      testAutomaticContextIgnoresTheVolume(argv[1]);
      testEngineFollowsTheGovernor(argv[1]);
      std::cout << "runtime resources tests passed\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "runtime resources tests failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
