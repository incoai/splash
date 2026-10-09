#pragma once

#include "ops/Vision.hpp"
#include "engine/AneFfnStartup.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/Cache.hpp"
#include "engine/CacheDirectory.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/KvPageTier.hpp"
#include "ops/PageStorage.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "engine/MemoryAudit.hpp"
#include "engine/ReleasableMemory.hpp"
#include "engine/Status.hpp"
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

// The memory plan counts each weight category from the loaded model, so
// every category the model has must report its allocation and identity.
void requireLoadedModel(const model::LoadedModel &loaded);

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
  // How long the engine keeps its memory without work (--idle-release): every
  // buffer stays wired that long after the last command (the backend's
  // residency keep-alive), and the weights' memory is freed once that long
  // passes without a request (NativeRuntime::releaseIdleWeights). Infinite
  // keeps both while the engine runs.
  double idleReleaseSeconds = metal::kResidencyKeepAliveSeconds;
  // Reclaimable host memory, sampled at every Metal operation during startup
  // and by the governor afterwards.
  MemoryGovernor::HostAvailableMemoryProvider hostAvailableMemory =
      queryHostAvailableMemory;
  // The prefill FFN's Neural Engine split (startAneFfn): off with
  // --disable-ane, and a benchmark round runs the split of the first so that
  // rounds repeat one another.
  AneFfnSetting aneFfn;
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

// A step of resource assembly that failed, with what() its message. Startup
// reports it with its step (RuntimeBootstrapReport::describe).
class RuntimeResourcesError final : public std::runtime_error {
public:
  RuntimeResourcesError(RuntimeResourceStage stage, std::string message,
                        std::string budgetDescription = {},
                        RuntimeResourceFailure failure =
                            RuntimeResourceFailure::Other);

  [[nodiscard]] RuntimeResourceStage stage() const noexcept { return stage_; }
  [[nodiscard]] RuntimeResourceFailure failure() const noexcept {
    return failure_;
  }
  [[nodiscard]] const std::string &budgetDescription() const noexcept {
    return budgetDescription_;
  }

private:
  RuntimeResourceStage stage_;
  RuntimeResourceFailure failure_;
  std::string budgetDescription_;
};

// Owns every process-wide native resource exactly once. Members go in
// reverse declaration order: what the engine gives back while idle -> Neural
// Engine split -> Cache -> KV pool -> KV disk tier -> state storage -> KV
// page storage -> governor -> loaded model -> Metal backend -> a persistent
// tier's directory, whose lock goes last.
// The KV disk tier must go before the KV page storage: its IO worker reads
// and writes pages in place in the extents, and its destructor waits for
// every transfer in flight. A persistent tier's files are sealed first,
// however the process ends, so that the copies the cache lets go of stay for
// the next process.
class RuntimeResources final {
public:
  // `requestedContextTokens` is the context the engine is asked to hold
  // (--max-context), zero for the automatic context (AneFfnOutcome::context),
  // which the Neural Engine split must leave. A persistent tier takes back
  // what the last process left before create() returns.
  [[nodiscard]] static std::unique_ptr<RuntimeResources>
  create(const RuntimeResourcesConfig &config, uint32_t requestedContextTokens);

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
  // The share the prefill FFN's Neural Engine split runs at and the least
  // rows of a chunk it takes; 0 for none.
  [[nodiscard]] double aneFfnShare() const noexcept { return aneFfn_ ? aneFfn_->share() : 0.0; }
  [[nodiscard]] uint32_t aneFfnMinimumRows() const noexcept { return aneFfn_ ? aneFfn_->minimumRows() : 0; }
  // How the start's split came out, and the automatic context.
  [[nodiscard]] const AneFfnOutcome &aneFfnOutcome() const noexcept { return aneFfnOutcome_; }
  // The split as it stands now, for /status.
  [[nodiscard]] AneFfnSnapshot aneFfnSnapshot() const;

  [[nodiscard]] model::RuntimeContext modelContext() noexcept;
  // What the engine gives back while idle (NativeLoopConfig::weights): the
  // weight images, and the Neural Engine split's program if it runs one.
  [[nodiscard]] ReleasableMemory &releasableMemory() noexcept { return releasableMemory_; }
  [[nodiscard]] ActualMemoryReport
  actualMemoryReport(const model::ModelMemoryActual &modelMemory) const;

private:

  RuntimeResources(PersistentCacheFiles persistentCache,
                   std::unique_ptr<metal::MetalBackend> backend,
                   model::LoadedModel model, ops::ExecutionPlans operators,
                   EngineMemoryPlan memoryPlan,
                   RuntimeCacheIdentity cacheIdentity,
                   std::unique_ptr<MemoryGovernor> memoryGovernor,
                   std::unique_ptr<kv::PageStorage> kvPages,
                   std::unique_ptr<model::QwenStateStorage> stateStorage,
                   std::unique_ptr<KvPageTier> kvTier,
                   std::unique_ptr<KvPool> kvPool,
                   std::unique_ptr<engine::Cache> cache,
                   std::unique_ptr<ops::AneFfn> aneFfn, AneFfnOutcome aneFfnOutcome,
                   std::optional<uint64_t> hostAvailableAtStart);
  // Takes back the restore points the last process left in a persistent
  // tier (Cache::adopt) and opens its files for this one.
  void adoptPersistentCache();
  void stopProbation() noexcept;

  PersistentCacheFiles persistentCache_;
  std::unique_ptr<metal::MetalBackend> backend_;
  model::LoadedModel model_;
  ops::ExecutionPlans operators_;
  EngineMemoryPlan memoryPlan_;
  RuntimeCacheIdentity cacheIdentity_;
  std::unique_ptr<MemoryGovernor> memoryGovernor_;
  std::unique_ptr<kv::PageStorage> kvPages_;
  std::unique_ptr<model::QwenStateStorage> stateStorage_;
  std::unique_ptr<KvPageTier> kvTier_;
  std::unique_ptr<KvPool> kvPool_;
  std::unique_ptr<engine::Cache> cache_;
  std::unique_ptr<ops::AneFfn> aneFfn_;
  // Over model_'s images and aneFfn_.
  ReleasableMemory releasableMemory_;
  AneFfnOutcome aneFfnOutcome_;
  std::optional<uint64_t> hostAvailableAtStart_;
  // Ends a probation once it has lasted, unless the process stops first.
  std::thread probation_;
  std::mutex probationMutex_;
  std::condition_variable probationWake_;
  bool probationStopped_ = false;
};

// The model's side of the split of `loaded`'s target on `backend`, valid
// while `backend`, `loaded` and `operators` are: its timing and prepared split
// run on a prefill arena of their own, its programs wait for the ANE's
// service until `cancelled` returns true, and its calibrations are
// remembered beside the programs (ane::recall) under its layers' shapes and
// formats, the device, the macOS version, the engine's `buildId` and the
// most units the plan holds.
[[nodiscard]] AneFfnModel aneFfnModel(metal::MetalBackend &backend, const model::LoadedModel &loaded,
                                      const ops::ExecutionPlans &operators, kv::Format format,
                                      std::string_view buildId, std::function<bool()> cancelled);

// Connects an engine to the governor that admits its memory: the engine asks
// it whether the host pauses growth, and marks the allocations a request in
// service makes. Every engine that runs against a governor connects through it.
void connectToGovernor(EngineConfig &config, MemoryGovernor &governor);

} // namespace splash::engine
