#pragma once

#include "ops/Vision.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/KvPageTier.hpp"
#include "ops/PageStorage.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "engine/MemoryAudit.hpp"
#include "ops/ExecutionPlans.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::engine {

struct EngineConfig;

enum class RuntimeResourceStage {
  Configuration,
  BackendCreation,
  CapabilityValidation,
  ModelLoading,
  MemoryPlanning,
  StorageAllocation,
};

[[nodiscard]] std::string_view
runtimeResourceStageName(RuntimeResourceStage stage);

// What /status reports of the loaded model, the build and the KV pages.
// Both digests are SHA-256 in lowercase hex.
struct RuntimeCacheIdentity {
  // The combined manifest of every model the runtime loaded.
  std::string modelLayoutSha256;
  std::string buildId;
  kv::Layout kvLayout;
  // The target model's manifest.
  std::string targetModelSha256;
};

[[nodiscard]] RuntimeCacheIdentity
makeRuntimeCacheIdentity(std::string_view combinedManifestSha256,
                         std::string_view targetManifestSha256,
                         std::string_view buildId,
                         kv::Layout targetKvLayout);

// The memory plan counts each weight category from the loaded package, so
// every category the model has must report its allocation and identity.
void requireLoadedModel(const model::ModelPackage &package);

struct RuntimeResourcesConfig {
  kv::Format kvFormat = kv::Format::Int8;
  std::filesystem::path metallibPath;
  std::filesystem::path modelRoot;
  model::ModelDescriptor model;
  std::string buildId;
  uint64_t maximumMemoryBytes = 0;
  // Disk quota shared by cached KV pages and states; zero disables the tier.
  uint64_t maximumCacheDiskBytes = 0;
  // Patches per image, from --max-image-patches: the engine admits images up
  // to it when the model loaded vision and none otherwise. The wire parser
  // keeps the protocol ceiling.
  uint32_t maximumImagePatches = ops::kMaximumImagePatches;
  // The process's existing pressure observer runs before resource assembly;
  // it only publishes a level. Bootstrap checks it at Metal operation
  // boundaries; after Ready the transport control handler keeps it current.
  std::function<MemoryPressure()> memoryPressure;
  std::function<bool()> cancelled;
};

enum class RuntimeResourceFailure {
  Other,
  HostCapacity,
  EngineCapacity,
  DriverAllocation,
};

[[nodiscard]] constexpr RuntimeResourceFailure resourceAllocationFailure(
    metal::AllocationFailure failure) noexcept {
  switch (failure) {
  case metal::AllocationFailure::HostPressure:
    return RuntimeResourceFailure::HostCapacity;
  case metal::AllocationFailure::EngineBudget:
    return RuntimeResourceFailure::EngineCapacity;
  case metal::AllocationFailure::DriverRejected:
    return RuntimeResourceFailure::DriverAllocation;
  default:
    return RuntimeResourceFailure::Other;
  }
}

class RuntimeResourcesError final : public std::runtime_error {
public:
  RuntimeResourcesError(RuntimeResourceStage stage, std::string message,
                        std::string statusJson = {},
                        std::string budgetDescription = {},
                        RuntimeResourceFailure failure =
                            RuntimeResourceFailure::Other);

  [[nodiscard]] RuntimeResourceFailure failure() const noexcept {
    return failure_;
  }
  [[nodiscard]] const std::string &message() const noexcept { return message_; }
  [[nodiscard]] const std::string &statusJson() const noexcept {
    return statusJson_;
  }
  [[nodiscard]] const std::string &budgetDescription() const noexcept {
    return budgetDescription_;
  }

private:
  RuntimeResourceFailure failure_;
  std::string message_;
  std::string statusJson_;
  std::string budgetDescription_;
};

// Owns every process-wide native resource exactly once. Members go in
// reverse declaration order: Cache -> KV pool -> KV disk tier -> state
// storage -> KV page storage -> governor -> model package -> Metal backend.
// The KV disk tier must go before the KV page storage: its IO worker reads
// and writes pages in place in the extents, and its destructor waits for
// every transfer in flight.
class RuntimeResources final {
public:
  [[nodiscard]] static std::unique_ptr<RuntimeResources>
  create(const RuntimeResourcesConfig &config);

  RuntimeResources(const RuntimeResources &) = delete;
  RuntimeResources &operator=(const RuntimeResources &) = delete;

  [[nodiscard]] metal::MetalBackend &backend() noexcept { return *backend_; }
  [[nodiscard]] const EngineMemoryPlan &memoryPlan() const noexcept {
    return memoryPlan_;
  }
  [[nodiscard]] MemoryGovernor &memoryGovernor() noexcept {
    return *memoryGovernor_;
  }
  [[nodiscard]] engine::Cache &cache() noexcept {
    return *cache_;
  }
  [[nodiscard]] const RuntimeCacheIdentity &cacheIdentity() const noexcept {
    return cacheIdentity_;
  }
  // What other applications left, measured before the engine took any;
  // empty when the host could not be measured.
  [[nodiscard]] std::optional<uint64_t> hostAvailableAtStart() const noexcept {
    return hostAvailableAtStart_;
  }

  [[nodiscard]] model::RuntimeContext modelContext() noexcept;
  [[nodiscard]] ActualMemoryReport
  actualMemoryReport(const model::ModelMemoryActual &modelMemory) const;

private:

  RuntimeResources(std::unique_ptr<metal::MetalBackend> backend,
                   model::ModelPackage model, ops::ExecutionPlans operators,
                   EngineMemoryPlan memoryPlan,
                   RuntimeCacheIdentity cacheIdentity,
                   std::unique_ptr<MemoryGovernor> memoryGovernor,
                   std::unique_ptr<kv::PageStorage> kvPages,
                   std::unique_ptr<model::QwenStateStorage> stateStorage,
                   std::unique_ptr<KvPageTier> kvTier,
                   std::unique_ptr<KvPool> kvPool,
                   std::unique_ptr<engine::Cache> cache,
                   std::optional<uint64_t> hostAvailableAtStart);

  std::unique_ptr<metal::MetalBackend> backend_;
  model::ModelPackage model_;
  ops::ExecutionPlans operators_;
  EngineMemoryPlan memoryPlan_;
  RuntimeCacheIdentity cacheIdentity_;
  std::unique_ptr<MemoryGovernor> memoryGovernor_;
  std::unique_ptr<kv::PageStorage> kvPages_;
  std::unique_ptr<model::QwenStateStorage> stateStorage_;
  std::unique_ptr<KvPageTier> kvTier_;
  std::unique_ptr<KvPool> kvPool_;
  std::unique_ptr<engine::Cache> cache_;
  std::optional<uint64_t> hostAvailableAtStart_;
};

// Connects an engine to the governor that admits its memory: the engine asks
// it whether the host pauses growth, and marks the allocations a request in
// service makes. Every engine that runs against a governor connects through it.
void connectToGovernor(EngineConfig &config, MemoryGovernor &governor);

} // namespace splash::engine
