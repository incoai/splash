#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestCheckpoint.hpp"
#include "TestStderr.hpp"
#include "engine/Bootstrap.hpp"
#include "engine/RuntimeResources.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryPlan.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

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
                                            model::TargetSource::Mlx, model::VisionSource::Mlx);
  config.buildId = "budget-test";
  writeCheckpoints(root, config.model);
  return config;
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
// of them names another.
void testPersistentCacheNamespace() {
  const model::ModelDescriptor model = model::makeModelDescriptor(
      "namespace-test", model::Qwen3_8Layout{}, model::kQwen3_8DraftLayout,
      model::kQwen3_8VisionLayout, model::TargetSource::Mlx,
      model::VisionSource::Mlx);
  const auto identity = [&](char models, std::string_view build, kv::Format format) {
    kv::Layout layout = model.targetKvLayout;
    layout.format = format;
    return makeRuntimeCacheIdentity(std::string(64, models), std::string(64, 'b'), build,
                                    layout);
  };
  const std::string name =
      persistentCacheNamespace(identity('a', "build", kv::Format::Int8), model.stateLayout);
  require(name.size() == 32 && name.find_first_not_of("0123456789abcdef") == std::string::npos,
          "a cache namespace is not 32 lowercase hex digits");
  require(persistentCacheNamespace(identity('a', "another build", kv::Format::Int8),
                                   model.stateLayout) == name,
          "the build changed the cache namespace");
  model::CompositeStateLayout states = model.stateLayout;
  ++states.draft.layers;
  require(persistentCacheNamespace(identity('c', "build", kv::Format::Int8), model.stateLayout) !=
                  name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::BFloat16),
                                       model.stateLayout) != name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), states) != name,
          "other models or layouts shared a cache namespace");
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
// leaves the tier off and the budget to KV. The notice and the refusal name
// the tier as users know it, the SSD cache.
void testStateStagingNeedsAStartedTier(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  const uint64_t minimum = minimumBytes(config);
  const uint64_t stateBytes = config.model.stateLayout.cachedBytes();
  config.maximumMemoryBytes = minimum;
  config.maximumCacheDiskBytes = stateBytes - 1;
  const std::string notice = test::capturedStderr([&] {
    requireReachesModelLoader(config, root.path(),
                              "a quota that holds no state set staging aside");
  });
  require(notice.find("SSD cache disabled (") != std::string::npos,
          "the notice of a quota that holds no state did not name the SSD cache");

  config.maximumCacheDiskBytes = stateBytes;
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

// Startup and the governor sample host memory through the config's probe:
// without one, the config is refused before the backend is created.
void testHostMemoryProbeIsRequired(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  config.hostAvailableMemory = {};
  try {
    auto resources = RuntimeResources::create(config, 0);
    throw std::runtime_error("resources were assembled without a host memory probe");
  } catch (const RuntimeResourcesError &error) {
    require(error.stage() == RuntimeResourceStage::Configuration,
            "a missing host memory probe was not a configuration error");
  }
}

// The 27B under a budget half a GiB above its weights: the weights alone fit,
// but not with what the runtime needs beside them. Startup refuses it before
// any weight is loaded.
void testModelBeyondBudgetIsRefusedBeforeLoading(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  config.model = model::makeModelDescriptor("budget-test", model::Qwen3_8Layout{}, model::kQwen3_8DraftLayout,
                                            model::kQwen3_8VisionLayout, model::TargetSource::Mlx,
                                            model::VisionSource::Mlx);
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
                                            model::kQwen3_8VisionLayout, model::TargetSource::Mlx,
                                            model::VisionSource::Mlx);
  writeCheckpoints(root.path(), config.model);
  require(model::modelWeightBytes(root.path(), config.model) > 3 * kGiB, "the model must exceed the sample");
  config.hostAvailableMemory = [] { return std::optional<uint64_t>(3 * kGiB); };
  requireReachesModelLoader(config, root.path(),
                            "a model larger than reclaimable host memory "
                            "refused to start");

  // Below the reserve macOS is the one at risk, so startup waits instead.
  // Unmeasurable telemetry waits the same way.
  for (std::optional<uint64_t> available :
       {std::optional<uint64_t>(64 * kMiB), std::optional<uint64_t>()}) {
    config.hostAvailableMemory = [available] { return available; };
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
      model::kQwen3_8VisionLayout, model::TargetSource::Mlx,
      model::VisionSource::Mlx);
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

// Resource assembly names the step it fails at, at its real throw sites on
// either side of the weight load: a host with less than its reserve refuses
// the start before the weights are opened, and one with its reserve but not
// the warning margin beyond it refuses the KV runway once the weights have
// loaded. The startup retry window counts the later step as progress.
void testFailedStepIsNamed(const char *metallibPath) {
  using namespace std::chrono_literals;
  const test::TemporaryDirectory root("splash-assembly");
  const PlannedModel planned = plannedModel();
  static_cast<void>(test::writeSyntheticModel(root.path(), planned.target, planned.draft, planned.vision));
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root.path();
  config.model = model::makeModelDescriptor("assembly-test", planned.target, planned.draft,
                                            planned.vision, model::TargetSource::Mlx,
                                            model::VisionSource::Mlx);
  config.model.sourceIdentity = "synthetic";
  config.buildId = "assembly-test";
  const auto host = [&](uint64_t available) {
    config.hostAvailableMemory = [available] { return std::optional<uint64_t>(available); };
  };
  // Nothing but the host refuses this model. The suite's one complete start
  // also reports the SSD cache a disk quota gives it, by that name.
  host(64 * kGiB);
  config.maximumCacheDiskBytes = kGiB;
  require(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); })
                  .find("SSD cache: 1024 MiB for KV pages of ") != std::string::npos,
          "a start did not report its SSD cache by that name");
  config.maximumCacheDiskBytes = 0;
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
  const RuntimeBootstrapReport after = failure(reserve + kHostWarningMarginBytes / 2);
  require(after.resourceStage == RuntimeResourceStage::StorageAllocation &&
              after.resourceFailure == RuntimeResourceFailure::HostCapacity &&
              after.message.find("unable to allocate the KV runway") != std::string::npos,
          "a KV runway the host refused after the weights loaded named another step");
  StartupRetryWindow window(30s);
  const auto first = StartupRetryWindow::Clock::time_point{};
  require(window.retryUntil(before, first) == first + 30s &&
              window.retryUntil(after, first + 20s) == first + 50s,
          "a failure after the weights loaded did not count as progress");
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
      hostReserve, [&available] { return available; }, 0);
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
      testEngineFollowsTheGovernor(argv[1]);
      std::cout << "runtime resources tests passed\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "runtime resources tests failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
