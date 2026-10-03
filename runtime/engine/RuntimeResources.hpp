#pragma once

#include "ops/Vision.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/Cache.hpp"
#include "engine/CacheDirectory.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/KvPageTier.hpp"
#include "ops/PageStorage.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "engine/MemoryAudit.hpp"
#include "ops/ExecutionPlans.hpp"

#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

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

// The name of the directory a persistent cache tier keeps its files in:
// what its copies hold, from the models that computed them, the KV and
// state layouts and the format of the cache's files. The build is not part
// of it; a change to what a copy holds bumps the cache's format instead.
[[nodiscard]] std::string
persistentCacheNamespace(const RuntimeCacheIdentity &identity,
                         const model::CompositeStateLayout &states);

// A persistent cache tier's directory, and the files of its KV pages and
// states there.
struct PersistentCacheFiles final {
  std::unique_ptr<CacheDirectory> directory;
  std::shared_ptr<model::SlotFile> kv;
  std::shared_ptr<model::SlotFile> states;
};

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
  // Where the tier keeps its files for the next process to take back
  // (--cache-dir): one directory per persistentCacheNamespace(). Empty for
  // temporary files that go with the process.
  std::filesystem::path persistentCacheRoot;
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
// storage -> KV page storage -> governor -> model package -> Metal backend
// -> a persistent tier's directory, whose lock goes last. The KV disk tier
// must go before the KV page storage: its IO worker reads and writes pages
// in place in the extents, and its destructor waits for every transfer in
// flight. A persistent tier's files are sealed first, however the process
// ends, so that the copies the cache lets go of stay for the next process.
class RuntimeResources final {
public:
  // A persistent tier takes back what the last process left before create()
  // returns.
  [[nodiscard]] static std::unique_ptr<RuntimeResources>
  create(const RuntimeResourcesConfig &config);

  RuntimeResources(const RuntimeResources &) = delete;
  RuntimeResources &operator=(const RuntimeResources &) = delete;
  ~RuntimeResources();

  // The engine is Ready: a persistent tier's directory marks this process
  // serving, on probation for its first minute after an unclean end
  // (CacheDirectory::beginServing).
  void beginServing();
  // At a clean stop, after the engine's flush (Engine::flushRestorePoints):
  // each state copy's label takes its state's current recency, both files
  // reach the drive and the directory records a clean end. False when they
  // did not; true at once without a persistent tier.
  [[nodiscard]] bool closePersistentCache();

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

  RuntimeResources(PersistentCacheFiles persistentCache,
                   std::unique_ptr<metal::MetalBackend> backend,
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
  // Takes back the restore points the last process left in a persistent
  // tier (Cache::adopt) and opens its files for this one.
  void adoptPersistentCache();
  void stopProbation() noexcept;

  PersistentCacheFiles persistentCache_;
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
  // Ends a probation once it has lasted, unless the process stops first.
  std::thread probation_;
  std::mutex probationMutex_;
  std::condition_variable probationWake_;
  bool probationStopped_ = false;
};

// Connects an engine to the governor that admits its memory: the engine asks
// it whether the host pauses growth, and marks the allocations a request in
// service makes. Every engine that runs against a governor connects through it.
void connectToGovernor(EngineConfig &config, MemoryGovernor &governor);

} // namespace splash::engine
