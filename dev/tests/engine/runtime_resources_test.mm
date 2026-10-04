#include "ScopedTestConfig.hpp"
#include "TestChecks.hpp"
#include "engine/RuntimeResources.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryPlan.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using namespace splash;
using namespace splash::engine;

using splash::test::require;

class TemporaryModelRoot final {
public:
  // Placeholders are sparse: a package larger than the machine's memory
  // costs three extents on disk.
  explicit TemporaryModelRoot(uint64_t bytesPerComponent = 16 * 1024)
      : fileBytes(bytesPerComponent), packageBytes(3 * bytesPerComponent) {
    path = std::filesystem::temp_directory_path() /
           ("splash-budget-" +
            std::string([NSUUID UUID].UUIDString.UTF8String));
    for (const char *component : {"target", "draft", "vision"}) {
      std::filesystem::create_directories(path / component);
      const auto file = path / component / "placeholder.bin";
      std::ofstream(file).put('\0');
      std::filesystem::resize_file(file, fileBytes);
    }
  }

  ~TemporaryModelRoot() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }

  uint64_t fileBytes;
  uint64_t packageBytes;
  std::filesystem::path path;
};

RuntimeResourcesConfig budgetConfig(const char *metallibPath,
                                    const TemporaryModelRoot &root) {
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root.path;
  config.model = model::makeModelDescriptor(
      "budget-test", model::Qwen3_8Layout{}, model::DFlashDraftLayout{},
      ops::VisionLayout{});
  config.buildId = "budget-test";
  return config;
}

// Startup fails at the loader, whose weight files are deliberately absent.
// Reaching it is the assertion: everything the engine checks before opening
// the package let this configuration through.
void requireReachesModelLoader(RuntimeResourcesConfig config,
                               const std::filesystem::path &root,
                               const char *message) {
  try {
    auto resources = RuntimeResources::create(config);
    throw std::runtime_error("placeholder model unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    require(error.failure() == RuntimeResourceFailure::Other &&
                std::string(error.what()).find("[model_loading]") !=
                    std::string::npos &&
                error.message().find("unable to open") != std::string::npos &&
                error.message().find((root / "target").string()) !=
                    std::string::npos,
            message);
  }
}

// Beside its weights a model needs at least the runtime reserves, one lane's
// state and the KV runway.
uint64_t minimumBytes(const RuntimeResourcesConfig &config,
                      const TemporaryModelRoot &root) {
  return minimumRequiredBytes(root.packageBytes + model::kPipelineReserveBytes +
                                  model::kRuntimeOverheadReserveBytes,
                              config.model.stateLayout.laneBytes(),
                              config.model.targetKvLayout)
      .value();
}

// A persistent cache's directory is named for what its copies hold: the same
// models and layouts name the same one under any build, and a change to any
// of them names another.
void testPersistentCacheNamespace() {
  const model::ModelDescriptor model = model::makeModelDescriptor(
      "namespace-test", model::Qwen3_8Layout{}, model::DFlashDraftLayout{},
      ops::VisionLayout{});
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
  TemporaryModelRoot root;
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root);

  {
    config.memoryPressure = [] { return MemoryPressure::Critical; };
    try {
      auto resources = RuntimeResources::create(config);
      throw std::runtime_error("model load ignored system pressure");
    } catch (const RuntimeResourcesError &error) {
      require(error.failure() == RuntimeResourceFailure::HostCapacity,
              "startup pressure did not remain retryable");
      require(error.message().find("not enough free memory") !=
                  std::string::npos,
              "startup pressure reached the weight loader");
    }
  }
  config.memoryPressure = [] { return MemoryPressure::Warning; };

  // The low ceiling is one byte short of the minimum, so every directory
  // must be counted. The other ceilings must reach the real loader, whose
  // expected weight files are deliberately absent. No actual model package is
  // needed for this test.
  const uint64_t minimum = minimumBytes(config, root);
  for (uint64_t ceiling : {minimum - 1, minimum, uint64_t{0}}) {
    config.maximumMemoryBytes = ceiling;
    try {
      auto resources = RuntimeResources::create(config);
      throw std::runtime_error("placeholder model unexpectedly loaded");
    } catch (const RuntimeResourcesError &error) {
      if (ceiling == minimum - 1) {
        require(error.failure() == RuntimeResourceFailure::EngineCapacity,
                "hard weight budget lost its engine-capacity classification");
        require(std::string(error.what()).find("[memory_planning]") !=
                    std::string::npos &&
                    error.message().find(
                        "require " + std::to_string(minimum) + " bytes") !=
                        std::string::npos &&
                    error.message().find(
                        "budget is " + std::to_string(minimum - 1) +
                        " bytes") != std::string::npos,
                "weight loading began before checking the memory ceiling");
      } else {
        require(error.failure() == RuntimeResourceFailure::Other,
                "missing model file was misclassified as allocation pressure");
      }
    }
  }
  config.maximumMemoryBytes = 0;
  requireReachesModelLoader(config, root.path,
                            "a sufficient weight budget did not reach the "
                            "model loader");
}

// A state's write to the disk tier stages through a state-sized buffer the
// plan sets aside only when the tier starts: a quota that holds no state
// leaves the tier off and the budget to KV.
void testStateStagingNeedsAStartedTier(const char *metallibPath) {
  TemporaryModelRoot root;
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root);
  const uint64_t minimum = minimumBytes(config, root);
  const uint64_t stateBytes = config.model.stateLayout.cachedBytes();
  config.maximumMemoryBytes = minimum;
  config.maximumCacheDiskBytes = stateBytes - 1;
  requireReachesModelLoader(config, root.path,
                            "a quota that holds no state set staging aside");

  config.maximumCacheDiskBytes = stateBytes;
  try {
    auto resources = RuntimeResources::create(config);
    throw std::runtime_error("placeholder model unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    require(error.failure() == RuntimeResourceFailure::EngineCapacity &&
                std::string(error.what()).find("[memory_planning]") !=
                    std::string::npos &&
                error.message().find(
                    "require " + std::to_string(minimum + stateBytes) +
                    " bytes") != std::string::npos,
            "a started tier's staging was not counted before loading");
  }
  config.maximumMemoryBytes = minimum + stateBytes;
  requireReachesModelLoader(config, root.path,
                            "a budget with room for the staging did not "
                            "reach the model loader");
}

// The image patch limit is a positive multiple of four up to the
// protocol's per-image ceiling; anything else is refused before the backend
// is created.
void testImagePatchCapIsBounded(const char *metallibPath) {
  TemporaryModelRoot root;
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root);
  for (const uint32_t patches : {ops::kMaximumImagePatches + 4, 6U}) {
    config.maximumImagePatches = patches;
    try {
      auto resources = RuntimeResources::create(config);
      throw std::runtime_error("an image patch cap outside the protocol's was accepted");
    } catch (const RuntimeResourcesError &error) {
      require(std::string(error.what()).find("[configuration]") != std::string::npos,
              "an invalid image patch cap was not a configuration error");
    }
  }
  config.maximumImagePatches = 4096;
  requireReachesModelLoader(config, root.path,
                            "a smaller image patch cap did not reach the model loader");
}

// A 34.5 GiB model under a 35 GiB budget: the weights alone fit, but not
// with what the runtime needs beside them. Startup refuses it before any
// weight is loaded.
void testModelBeyondBudgetIsRefusedBeforeLoading(const char *metallibPath) {
  TemporaryModelRoot root(23 * kGiB / 2);
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root);
  config.maximumMemoryBytes = 35 * kGiB;
  try {
    auto resources = RuntimeResources::create(config);
    throw std::runtime_error("placeholder model unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    require(error.failure() == RuntimeResourceFailure::EngineCapacity &&
                std::string(error.what()).find("[memory_planning]") !=
                    std::string::npos,
            "a model that cannot fit reached the weight loader");
  }
}

// The rule that keeps users off the startup floor: admission weighs
// reclaimable memory against the macOS reserve, never against the model.
// A package far larger than everything reclaimable still starts, because
// the operation guard checks each image's allocation as loading builds the
// residency up.
void testStartupAdmissionIgnoresPackageSize(const char *metallibPath) {
  TemporaryModelRoot root(2 * kGiB);
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root);
  require(root.packageBytes > 3 * kGiB, "the package must exceed the sample");
  {
    const test::ScopedTestConfig seam(
        {.hostAvailableMemory = [] { return std::optional<uint64_t>(3 * kGiB); }});
    requireReachesModelLoader(config, root.path,
                              "a package larger than reclaimable host memory "
                              "refused to start");
  }

  // Below the reserve macOS is the one at risk, so startup waits instead.
  // Unmeasurable telemetry waits the same way.
  for (std::optional<uint64_t> available :
       {std::optional<uint64_t>(64 * kMiB), std::optional<uint64_t>()}) {
    const test::ScopedTestConfig seam(
        {.hostAvailableMemory = [available] { return available; }});
    try {
      auto resources = RuntimeResources::create(config);
      throw std::runtime_error("model load ignored the macOS reserve");
    } catch (const RuntimeResourcesError &error) {
      require(error.failure() == RuntimeResourceFailure::HostCapacity,
              "exhausted host memory did not remain retryable");
      require(error.message().find("not enough free memory") !=
                  std::string::npos,
              "exhausted host memory reached the weight loader");
    }
  }
}

// The memory plan takes the vision category from what loaded, so a model
// with vision whose loader produced no vision bytes must stop here.
void testLoadedVisionIsRequiredOnlyWithVision() {
  model::ModelPackage package;
  package.descriptor = model::makeModelDescriptor(
      "loaded-test", model::Qwen3_8Layout{}, model::DFlashDraftLayout{},
      ops::VisionLayout{});
  model::Qwen3_8Weights target;
  target.actualAllocatedBytes = 1;
  target.manifestFingerprintSha256 = "target";
  package.target = std::move(target);
  package.draft.actualAllocatedBytes = 1;
  package.manifestFingerprintSha256 = "package";
  bool rejected = false;
  try {
    requireLoadedModel(package);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected && package.descriptor.hasVision(),
          "a multimodal model without loaded vision weights was accepted");
  package.vision.actualAllocatedBytes = 1;
  requireLoadedModel(package);
  package.vision.actualAllocatedBytes = 0;
  package.descriptor.visionSource = model::VisionSource::None;
  requireLoadedModel(package);
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
      testModelBeyondBudgetIsRefusedBeforeLoading(argv[1]);
      testStartupAdmissionIgnoresPackageSize(argv[1]);
      testEngineFollowsTheGovernor(argv[1]);
      std::cout << "runtime resources tests passed\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "runtime resources tests failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
