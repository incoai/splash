#pragma once

#include "ops/Vision.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "ops/Q8PageStorage.hpp"
#include "model/ModelFactory.hpp"
#include "engine/MemoryAudit.hpp"
#include "ops/ExecutionPlans.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::engine {

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

inline constexpr std::string_view kQ8FormatName =
    "q8s8_f32_scale_per_token_head_k_token_major_v_dimension_major";

[[nodiscard]] inline std::string
digestHex(const std::array<uint8_t, 32> &digest) {
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (uint8_t byte : digest) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

struct RuntimeCacheIdentity {
  std::string modelLayoutSha256;
  std::string buildId;
  // One process-wide content namespace. KV blocks never copy model/build
  // strings or physical layout metadata.
  CacheNamespace cacheNamespace;
  // Stable binary layout guard for physical Q8 pages.
  kv::Q8LayoutGuard q8Layout;
  // SHA-256 of the versioned model/build/Q8 compatibility tuple.
  std::string namespaceSha256;
};

[[nodiscard]] RuntimeCacheIdentity
makeRuntimeCacheIdentity(std::string_view combinedManifestSha256,
                         std::string_view targetManifestSha256,
                         std::string_view buildId,
                         kv::Q8Layout targetKvLayout);

struct RuntimeResourcesConfig {
  std::filesystem::path metallibPath;
  std::filesystem::path modelRoot;
  model::ModelDescriptor model;
  std::string buildId;
  uint64_t maximumMemoryBytes = 0;
  // Patches per image the vision scratch covers; the protocol limit and the
  // engine's request validation use the same value.
  uint32_t maximumImagePatches = ops::kMaximumImagePatches;
  // The process's existing pressure observer runs before resource assembly;
  // it only publishes a level. Bootstrap checks it at Metal operation
  // boundaries; after Ready the transport control handler keeps it current.
  std::function<MemoryPressure()> memoryPressure;
  std::function<bool()> cancelled;
  // Reclaimable host memory, sampled at every Metal operation during startup
  // and by the governor afterwards. Empty means the live vm_statistics64
  // estimate; tests substitute a fixed value.
  MemoryGovernor::HostAvailableMemoryProvider hostAvailableMemory;
  // Kernel choices to install over the operator policy. Used by the offline
  // measurement tool and tests; production leaves it empty. Arenas are sized
  // for the operator defaults plus these choices.
  std::optional<ops::OperatorChoices> operatorChoices;
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

// Owns every process-wide native resource exactly once. Destruction order is
// Cache -> logical KV pool -> state -> Q8 backing -> governor ->
// model package -> Metal backend.
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
  [[nodiscard]] const model::ModelMemoryPlan &
  modelMemoryPlan() const noexcept {
    return modelMemoryPlan_;
  }
  [[nodiscard]] MemoryGovernor &memoryGovernor() noexcept {
    return *memoryGovernor_;
  }
  [[nodiscard]] model::StateStorage &stateStorage() noexcept {
    return *stateStorage_;
  }
  [[nodiscard]] engine::Cache &cache() noexcept {
    return *cache_;
  }
  [[nodiscard]] const RuntimeCacheIdentity &cacheIdentity() const noexcept {
    return cacheIdentity_;
  }

  [[nodiscard]] model::RuntimeContext modelContext() noexcept;
  [[nodiscard]] ActualMemoryReport
  actualMemoryReport(const model::ModelMemoryActual &modelMemory,
                     uint64_t estimatedWarmupPeakBytes) const;
  // Offline tuning tool only, before any request: swaps between the operator
  // defaults and the choices this instance was created with. Arenas were
  // sized for exactly those two configurations, so nothing else may be
  // installed after creation.
  void installOperatorChoices(const ops::OperatorChoices &choices) {
    operators_.install(choices);
  }

private:

  RuntimeResources(std::unique_ptr<metal::MetalBackend> backend,
                   model::ModelPackage model, ops::ExecutionPlans operators,
                   EngineMemoryPlan memoryPlan,
                   model::ModelMemoryPlan modelMemoryPlan,
                   RuntimeCacheIdentity cacheIdentity,
                   std::unique_ptr<MemoryGovernor> memoryGovernor,
                   std::unique_ptr<kv::Q8PageStorage> kvPages,
                   std::unique_ptr<model::StateStorage> stateStorage,
                   std::unique_ptr<KvPool> kvPool,
                   std::unique_ptr<engine::Cache> cache,
                   uint32_t maximumImagePatches);

  std::unique_ptr<metal::MetalBackend> backend_;
  model::ModelPackage model_;
  ops::ExecutionPlans operators_;
  EngineMemoryPlan memoryPlan_;
  model::ModelMemoryPlan modelMemoryPlan_;
  RuntimeCacheIdentity cacheIdentity_;
  std::unique_ptr<MemoryGovernor> memoryGovernor_;
  std::unique_ptr<kv::Q8PageStorage> kvPages_;
  std::unique_ptr<model::StateStorage> stateStorage_;
  std::unique_ptr<KvPool> kvPool_;
  std::unique_ptr<engine::Cache> cache_;
  uint32_t maximumImagePatches_ = 0;
};

} // namespace splash::engine
