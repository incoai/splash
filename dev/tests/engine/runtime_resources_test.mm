#include "engine/RuntimeResources.hpp"
#include "engine/MemoryPlan.hpp"

#import <Foundation/Foundation.h>

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

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

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

  // Each low ceiling fits two components, so every directory must be counted.
  // The other ceilings must reach the real loader, whose expected weight files
  // are deliberately absent. No actual model package is needed for this test.
  for (uint64_t ceiling : {root.packageBytes - 1, root.packageBytes,
                           uint64_t{0}}) {
    config.maximumMemoryBytes = ceiling;
    try {
      auto resources = RuntimeResources::create(config);
      throw std::runtime_error("placeholder model unexpectedly loaded");
    } catch (const RuntimeResourcesError &error) {
      if (ceiling == root.packageBytes - 1) {
        require(error.failure() == RuntimeResourceFailure::EngineCapacity,
                "hard weight budget lost its engine-capacity classification");
        require(std::string(error.what()).find("[memory_planning]") !=
                    std::string::npos &&
                    error.message().find("model weights require 49152 bytes") !=
                        std::string::npos &&
                    error.message().find("budget is 49151 bytes") !=
                        std::string::npos,
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

// The rule that keeps users off the startup floor: admission weighs
// reclaimable memory against the macOS reserve, never against the model.
// A package far larger than everything reclaimable still starts, because
// mapped weights become resident page by page under the operation guard.
void testStartupAdmissionIgnoresPackageSize(const char *metallibPath) {
  TemporaryModelRoot root(2 * kGiB);
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root);
  require(root.packageBytes > 3 * kGiB, "the package must exceed the sample");
  config.hostAvailableMemory = [] {
    return std::optional<uint64_t>(3 * kGiB);
  };
  requireReachesModelLoader(config, root.path,
                            "a package larger than reclaimable host memory "
                            "refused to start");

  // Below the reserve macOS is the one at risk, so startup waits instead.
  // Unmeasurable telemetry waits the same way.
  for (std::optional<uint64_t> available :
       {std::optional<uint64_t>(64 * kMiB), std::optional<uint64_t>()}) {
    config.hostAvailableMemory = [available] { return available; };
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

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "expected metallib path");
      testWeightBudgetBeforeLoading(argv[1]);
      testStartupAdmissionIgnoresPackageSize(argv[1]);
      std::cout << "runtime resources tests passed\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "runtime resources tests failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
