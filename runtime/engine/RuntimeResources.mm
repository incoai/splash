#include "engine/RuntimeResources.hpp"
#include "AwakeClock.hpp"
#include "Checked.hpp"
#include "StderrLine.hpp"
#include "engine/DiskLabels.hpp"
#include "engine/Engine.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/WeightStore.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <chrono>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>

namespace splash::engine {
namespace {

static_assert(model::ExecutionLimits::maximumBatchWidth ==
              SPLASH_MAXIMUM_BATCH_WIDTH);
static_assert(model::ExecutionLimits::prefillTokenBudget ==
              SPLASH_PREFILL_TOKEN_BUDGET);
static_assert(model::ExecutionLimits::draftQueryRows ==
              SPLASH_DRAFT_QUERY_ROWS);
static_assert(model::ExecutionLimits::draftProposalTokens ==
              SPLASH_DRAFT_PROPOSAL_TOKENS);
static_assert(model::ExecutionLimits::targetVerifyRows ==
              SPLASH_TARGET_VERIFY_ROWS);
static_assert(model::ExecutionLimits::draftContextTokens ==
              SPLASH_DRAFT_SLIDING_WINDOW);
static_assert(model::ExecutionLimits::speculativeScratchTokens ==
              SPLASH_SPECULATIVE_SCRATCH_TOKENS);

uint8_t hexNibble(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<uint8_t>(value - 'a' + 10);
  }
  if (value >= 'A' && value <= 'F') {
    return static_cast<uint8_t>(value - 'A' + 10);
  }
  throw std::invalid_argument("manifest SHA-256 is not hexadecimal");
}

uint64_t mebibytes(uint64_t bytes) noexcept { return bytes / kMiB; }

// The startup admission rule. Deliberately independent of the model size:
// it is checked here again at every Metal operation, each image's
// allocation included, as loading and warmup build the residency up.
void requireStartupHeadroom(
    const MemoryGovernor::HostAvailableMemoryProvider &hostAvailableMemory,
    uint64_t reserveBytes, MemoryPressure pressure) {
  const std::optional<uint64_t> available = hostAvailableMemory();
  if (!available || *available <= reserveBytes ||
      pressure == MemoryPressure::Critical) {
    std::ostringstream message;
    message << "not enough free memory to start: ";
    if (!available)
      message << "reclaimable host memory cannot be measured";
    else
      message << mebibytes(*available) << " MiB reclaimable, "
              << mebibytes(reserveBytes) << " MiB protected for macOS, system "
              << "pressure " << memoryPressureName(pressure);
    message << "; close memory-heavy applications and retry";
    throw metal::MetalAllocationError(message.str(),
                                      metal::AllocationFailure::HostPressure);
  }
}

// The format of a persistent cache's files and of what its copies hold. A
// change to either, such as a fix to a kernel that writes KV or state, bumps
// it, so that no start takes back copies of the old one.
constexpr uint32_t kPersistentCacheFormat = 1;
// How long a start waits for the cache directory another process holds: one
// that is closing has finished its flush, or been stopped by the server, by
// then (server/runtime.py `_shutdown_grace_seconds`).
constexpr std::chrono::seconds kCacheLockWait{20};
// Another model's cache nobody opened for this long is removed.
constexpr std::chrono::hours kStaleCache{24 * 14};
// How long a start that follows an unclean end serves on probation: the
// server counts a failure within that time toward a crash loop
// (server/backend.py `CRASH_LOOP_WINDOW_SECONDS`).
constexpr std::chrono::seconds kProbation{60};

// A persistent tier's directory under root and its files, empty when another
// process holds the directory or it cannot be used: the tier then keeps
// temporary files, as without one.
PersistentCacheFiles openPersistentCache(const std::filesystem::path &root,
                                         const std::string &cacheNamespace,
                                         uint64_t kvSlotBytes, uint64_t stateSlotBytes,
                                         const std::shared_ptr<model::DiskBudget> &budget,
                                         const std::function<bool()> &cancelled) {
  try {
    PersistentCacheFiles files;
    files.directory = CacheDirectory::open(root, cacheNamespace, kCacheLockWait,
                                           kStaleCache, cancelled);
    if (!files.directory) {
      if (!cancelled || !cancelled())
        logLine("Persistent cache ", (root / cacheNamespace).string(),
                " is in use by another process; this one keeps a temporary cache.");
      return {};
    }
    const std::vector<std::byte> tag(reinterpret_cast<const std::byte *>(cacheNamespace.data()),
                                     reinterpret_cast<const std::byte *>(cacheNamespace.data()) +
                                         cacheNamespace.size());
    files.kv = std::make_shared<model::SlotFile>(
        kvSlotBytes, budget,
        model::SlotFile::Persistence{files.directory->kvSlots(), files.directory->kvRecords(),
                                     tag, sizeof(KvBlockLabel)});
    files.states = std::make_shared<model::SlotFile>(
        stateSlotBytes, budget,
        model::SlotFile::Persistence{files.directory->stateSlots(),
                                     files.directory->stateRecords(), tag, sizeof(StateLabel)});
    return files;
  } catch (const std::exception &error) {
    logLine("Persistent cache disabled (", error.what(), "); this process keeps a temporary cache.");
    return {};
  }
}

std::array<uint8_t, 32> parseSha256(std::string_view value) {
  if (value.size() != 64) {
    throw std::invalid_argument(
        "manifest SHA-256 must contain exactly 64 hex characters");
  }
  std::array<uint8_t, 32> result{};
  for (size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<uint8_t>((hexNibble(value[index * 2]) << 4) |
                                         hexNibble(value[index * 2 + 1]));
  }
  return result;
}

} // namespace

void requireLoadedModel(const model::ModelPackage &package) {
  if (!package.targetActualAllocatedBytes() ||
      !package.draft.actualAllocatedBytes ||
      (package.descriptor.hasVision() && !package.vision.actualAllocatedBytes) ||
      package.manifestFingerprintSha256.empty() ||
      package.targetManifestFingerprint().empty()) {
    throw std::invalid_argument(
        "loaded model package has incomplete allocation accounting");
  }
}

std::string_view runtimeResourceStageName(RuntimeResourceStage stage) {
  switch (stage) {
  case RuntimeResourceStage::Configuration:
    return "configuration";
  case RuntimeResourceStage::BackendCreation:
    return "backend_creation";
  case RuntimeResourceStage::CapabilityValidation:
    return "capability_validation";
  case RuntimeResourceStage::ModelLoading:
    return "model_loading";
  case RuntimeResourceStage::MemoryPlanning:
    return "memory_planning";
  case RuntimeResourceStage::StorageAllocation:
    return "storage_allocation";
  }
  return "unknown";
}

RuntimeCacheIdentity
makeRuntimeCacheIdentity(std::string_view combinedManifestSha256,
                         std::string_view targetManifestSha256,
                         std::string_view buildId,
                         kv::Layout targetKvLayout) {
  if (buildId.empty()) {
    throw std::invalid_argument("runtime build id is required");
  }
  if (!targetKvLayout.valid()) {
    throw std::invalid_argument("runtime target KV layout is invalid");
  }
  // Parsing rejects a malformed manifest digest before the KV pool and the
  // cache are built.
  RuntimeCacheIdentity result;
  result.modelLayoutSha256 = model::digestHex(parseSha256(combinedManifestSha256));
  result.buildId = buildId;
  result.kvLayout = targetKvLayout;
  result.targetModelSha256 = model::digestHex(parseSha256(targetManifestSha256));
  return result;
}

std::string persistentCacheNamespace(const RuntimeCacheIdentity &identity,
                                     const model::CompositeStateLayout &states) {
  const kv::Layout &kv = identity.kvLayout;
  const model::GdnStateLayout &gdn = states.target;
  const model::DraftStateLayout &draft = states.draft;
  std::ostringstream canonical;
  canonical << "splash-persistent-cache-v" << kPersistentCacheFormat << '\n'
            << identity.modelLayoutSha256 << '\n'
            << "kv " << kv::kPageTokens << ' ' << kv.attentionLayers << ' ' << kv.kvHeads << ' '
            << kv.headDimension << ' ' << kv::formatName(kv.format) << '\n'
            << "gdn " << gdn.layers << ' ' << gdn.convolutionHistory << ' '
            << gdn.convolutionChannels << ' ' << gdn.recurrentGroups << ' ' << gdn.recurrentRows
            << ' ' << gdn.recurrentColumns << '\n'
            << "draft " << draft.layers << ' ' << draft.kvHeads << ' ' << draft.headDimension << ' '
            << SPLASH_DRAFT_SLIDING_WINDOW << '\n';
  // 128 bits name it.
  return model::weightDigest(canonical.str()).substr(0, 32);
}

RuntimeResourcesError::RuntimeResourcesError(RuntimeResourceStage stage,
                                             std::string message,
                                             std::string budgetDescription,
                                             RuntimeResourceFailure failure)
    : std::runtime_error(std::move(message)), stage_(stage), failure_(failure),
      budgetDescription_(std::move(budgetDescription)) {}

RuntimeResources::RuntimeResources(
    PersistentCacheFiles persistentCache,
    std::unique_ptr<metal::MetalBackend> backend, model::ModelPackage model,
    ops::ExecutionPlans operators, EngineMemoryPlan memoryPlan,
    RuntimeCacheIdentity cacheIdentity,
    std::unique_ptr<MemoryGovernor> memoryGovernor,
    std::unique_ptr<kv::PageStorage> kvPages,
    std::unique_ptr<model::QwenStateStorage> stateStorage,
    std::unique_ptr<KvPageTier> kvTier,
    std::unique_ptr<KvPool> kvPool, std::unique_ptr<engine::Cache> cache,
    std::optional<uint64_t> hostAvailableAtStart)
    : persistentCache_(std::move(persistentCache)),
      backend_(std::move(backend)), model_(std::move(model)),
      operators_(std::move(operators)),
      memoryPlan_(std::move(memoryPlan)),
      cacheIdentity_(std::move(cacheIdentity)),
      memoryGovernor_(std::move(memoryGovernor)), kvPages_(std::move(kvPages)),
      stateStorage_(std::move(stateStorage)), kvTier_(std::move(kvTier)),
      kvPool_(std::move(kvPool)),
      cache_(std::move(cache)), hostAvailableAtStart_(hostAvailableAtStart) {}

std::unique_ptr<RuntimeResources>
RuntimeResources::create(const RuntimeResourcesConfig &config) {
  if (config.metallibPath.empty() || config.modelRoot.empty() ||
      !kv::validFormat(config.kvFormat) ||
      !config.model.valid() || !config.hostAvailableMemory ||
      config.buildId.empty() || !config.maximumImagePatches ||
      config.maximumImagePatches % 4 ||
      config.maximumImagePatches > ops::kMaximumImagePatches) {
    throw RuntimeResourcesError(
        RuntimeResourceStage::Configuration,
        "metallib path, model root, a valid model layout and KV format, a "
        "host memory probe, build id, and a merge-aligned image patch limit no "
        "larger than the protocol's are required");
  }
  std::unique_ptr<metal::MetalBackend> backend;
  try {
    backend = std::make_unique<metal::MetalBackend>(config.metallibPath.string(),
                                                    config.idleReleaseSeconds);
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::BackendCreation,
                                error.what(), {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::BackendCreation,
                                error.what());
  }

  const DeviceCapabilities &device = backend->capabilities();
  if (auto error = device.validationError()) {
    throw RuntimeResourcesError(RuntimeResourceStage::CapabilityValidation,
                                *error);
  }

  const uint64_t hostReserveBytes =
      EngineMemoryPolicy::hostAvailableReserveBytes(device.physicalMemoryBytes);
  // What other applications leave, measured before the engine takes any.
  const std::optional<uint64_t> hostAvailableAtStart =
      config.hostAvailableMemory();
  // Startup work stops on cancellation and keeps its reserve of host memory.
  const auto throwIfCancelled = [cancelled = config.cancelled] {
    if (cancelled && cancelled())
      throw metal::MetalBackendError("startup cancelled");
  };
  const auto currentPressure = [pressure = config.memoryPressure] {
    return pressure ? pressure() : MemoryPressure::Normal;
  };
  const auto admitMetalOperation = [throwIfCancelled, currentPressure,
                                    hostAvailableMemory = config.hostAvailableMemory,
                                    hostReserveBytes] {
    throwIfCancelled();
    requireStartupHeadroom(hostAvailableMemory, hostReserveBytes,
                           currentPressure());
  };
  backend->setOperationGuard(admitMetalOperation);
  // A synchronous command wait gives up when the process shuts down, also
  // after startup.
  backend->setWaitInterrupt(config.cancelled);
  // One disk quota serves KV pages and states. Without room for a state,
  // disk KV cannot preserve a restorable prefix, so the tier stays off, and
  // no state's write needs the staging buffer the plan would set aside.
  std::shared_ptr<model::DiskBudget> diskBudget;
  std::shared_ptr<model::SlotFile> stateFile;
  const uint64_t stateBytes = config.model.stateLayout.cachedBytes();
  if (config.maximumCacheDiskBytes) {
    diskBudget = std::make_shared<model::DiskBudget>(config.maximumCacheDiskBytes);
    try {
      stateFile = std::make_shared<model::SlotFile>(
          model::SlotFile::slotBytesFor(stateBytes), diskBudget);
    } catch (const std::exception &error) {
      diskBudget.reset();
      logLine("Cache disk tier disabled (", error.what(),
              "); no state staging is set aside.");
    }
  }
  // A state's write to the disk tier stages through one buffer of a state's
  // size. It is the backend's like every other, so the governor charges it
  // beside the weights and the plan sets it aside before it sizes KV.
  const uint64_t stateStagingBytes = stateFile ? stateBytes : 0;
  try {
    const uint64_t hardBudgetBytes = EngineMemoryPolicy::hardBudgetBytes(
        device.recommendedMaxWorkingSetBytes, config.maximumMemoryBytes);
    // Reject a model that cannot fit before loading its weights. Beside
    // them the plan needs at least the runtime reserves, one lane's state,
    // the KV runway and any disk tier state staging; the full plan below adds
    // the arenas.
    kv::Layout kvLayout = config.model.targetKvLayout;
    kvLayout.format = config.kvFormat;
    uint64_t fixedBytes = 0;
    for (const uint64_t bytes :
         {model::modelWeightBytes(config.modelRoot, config.model),
          model::kPipelineReserveBytes, model::kRuntimeOverheadReserveBytes,
          stateStagingBytes}) {
      if (!checkedAdd(fixedBytes, bytes, fixedBytes))
        fixedBytes = std::numeric_limits<uint64_t>::max();
    }
    const uint64_t requiredBytes =
        minimumRequiredBytes(fixedBytes, config.model.stateLayout.laneBytes(),
                             kvLayout)
            .value_or(std::numeric_limits<uint64_t>::max());
    if (requiredBytes > hardBudgetBytes) {
      throw RuntimeResourcesError(
          RuntimeResourceStage::MemoryPlanning,
          "model weights with the runtime reserves, one lane's state, the KV "
          "runway and any disk tier state staging require " +
              std::to_string(requiredBytes) +
              " bytes but the Metal memory budget is " +
              std::to_string(hardBudgetBytes) + " bytes",
          {}, RuntimeResourceFailure::EngineCapacity);
    }
    // Fail before opening the package when the machine has no headroom at
    // all; the guard installed above keeps checking as residency grows.
    admitMetalOperation();
  } catch (const RuntimeResourcesError &) {
    throw;
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what());
  }

  model::ModelPackage package;
  try {
    const auto started = AwakeClock::now();
    package = model::loadModelPackage(*backend, config.modelRoot, config.model);
    requireLoadedModel(package);
    const std::chrono::duration<double> loading = AwakeClock::now() - started;
    logLine("Weights loaded in ", std::fixed, std::setprecision(2),
            loading.count(), " s.");
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what());
  }

  // One plan owner is used both before allocation and during encoding. The
  // engine lends it to model execution without inspecting its plans.
  ops::ExecutionPlans operators(device);
  model::ModelMemoryPlan modelMemoryPlan;
  try {
    modelMemoryPlan = model::plannedRuntimeMemory(device, package, operators, config.kvFormat);
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(
        RuntimeResourceStage::MemoryPlanning,
        std::string("model allocated-size plan is invalid: ") + error.what());
  }

  ModelMemoryFootprint footprint{
      package.targetActualAllocatedBytes(),
      package.draft.actualAllocatedBytes,
      package.vision.actualAllocatedBytes,
      modelMemoryPlan,
      stateStagingBytes,
  };

  ModelMemoryProfile modelProfile{
      package.name(), package.maximumContextTokens(),
      package.targetKvLayout(config.kvFormat), footprint};
  EngineMemoryPlanResult planResult =
      evaluateEngineMemoryPlan(device, modelProfile, config.maximumMemoryBytes);
  if (!planResult.plan) {
    throw RuntimeResourcesError(RuntimeResourceStage::MemoryPlanning,
                                planResult.status.message,
                                planResult.status.describe());
  }
  EngineMemoryPlan memoryPlan = std::move(*planResult.plan);

  RuntimeCacheIdentity cacheIdentity;
  try {
    cacheIdentity = makeRuntimeCacheIdentity(
        package.manifestFingerprintSha256,
        package.targetManifestFingerprint(), config.buildId,
        package.targetKvLayout(config.kvFormat));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(),
                                memoryPlan.breakdown().describe());
  }

  try {
    const EngineMemoryBreakdown &budget = memoryPlan.breakdown();
    // The governor holds the complete Metal footprint to the hard budget. The
    // plan budgets pipelines and driver allocations inside the pipeline and
    // allocator reserves, so memory outside the backend's buffers is charged
    // only beyond them, and elastic state and KV never grow into them.
    auto memoryGovernor = std::make_unique<MemoryGovernor>(
        *backend, budget.hardBudgetBytes, hostReserveBytes, config.hostAvailableMemory,
        budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes);
    if (config.memoryPressure)
      memoryGovernor->setPressure(config.memoryPressure());
    logLine("Kernel policy for GPU family ", device.appleGpuFamily,
            " with ", device.gpuCoreCount, " cores.");

    // Page ids for every extent the hard budget could hold: the governor,
    // never the id range, limits the pool.
    const uint64_t poolExtents = std::min<uint64_t>(
        (budget.hardBudgetBytes + budget.kvExtentBytes - 1) / budget.kvExtentBytes,
        std::numeric_limits<uint32_t>::max() / budget.kvExtentPages);
    auto kvPages = std::make_unique<kv::PageStorage>(
        *backend, memoryGovernor->allocationAdmission(), package.targetKvLayout(config.kvFormat),
        static_cast<uint32_t>(poolExtents * budget.kvExtentPages), budget.kvExtentPages);
    auto kvPool = std::make_unique<KvPool>(*kvPages, model::ExecutionLimits::warmupKvPages);
    // A persistent tier keeps its files in a cache directory of its own.
    // When another process holds it or it cannot be used, the tier keeps
    // temporary files, as without one.
    PersistentCacheFiles persistentCache;
    if (stateFile && !config.persistentCacheRoot.empty()) {
      persistentCache = openPersistentCache(
          config.persistentCacheRoot,
          persistentCacheNamespace(cacheIdentity, package.stateLayout()),
          model::SlotFile::slotBytesFor(kvPages->bytesPerPage()), stateFile->slotBytes(),
          diskBudget, config.cancelled);
      if (persistentCache.directory)
        stateFile = persistentCache.states;
    }
    auto stateStorage = std::make_unique<model::QwenStateStorage>(
        *backend, memoryGovernor->allocationAdmission(), package.stateLayout(),
        stateFile);
    std::unique_ptr<KvPageTier> kvTier;
    if (diskBudget) {
      try {
        const uint64_t slotBytes = model::SlotFile::slotBytesFor(kvPages->bytesPerPage());
        kvTier = std::make_unique<KvPageTier>(
            *kvPages, persistentCache.kv
                          ? persistentCache.kv
                          : std::make_shared<model::SlotFile>(slotBytes, diskBudget));
        logLine(persistentCache.directory ? "Persistent cache tier: " : "Cache disk tier: ",
                config.maximumCacheDiskBytes / kMiB, " MiB for KV pages of ",
                slotBytes / 1024, " KiB and states of ", stateBytes / kMiB,
                " MiB; a state's write stages through ", stateStagingBytes / kMiB,
                " MiB of the memory plan.");
      } catch (const std::exception &error) {
        // The persistent files are open by now; without the tier nothing
        // would keep or replace their copies.
        if (persistentCache.directory)
          throw;
        logLine("Cache disk KV storage disabled; state storage remains enabled (",
                error.what(), ").");
      }
    }
    auto cache = std::make_unique<engine::Cache>(*kvPool, kvTier.get(), diskBudget);

    if (stateStorage->actualAllocatedBytes() != 0) {
      throw std::runtime_error("lane state was allocated eagerly");
    }
    metal::MetalMemoryStats memory = backend->memoryStats();
    if (!backend->healthy()) {
      throw std::runtime_error(
          "Metal backend became unhealthy during resource allocation: " +
          backend->unhealthyReason());
    }
    if (memory.allocatedBytes > budget.hardBudgetBytes ||
        memory.deviceCurrentAllocatedBytes > budget.hardBudgetBytes) {
      throw std::runtime_error(
          "base Metal allocation exceeds immutable hard budget");
    }

    auto result = std::unique_ptr<RuntimeResources>(new RuntimeResources(
        std::move(persistentCache), std::move(backend), std::move(package),
        std::move(operators), std::move(memoryPlan), std::move(cacheIdentity),
        std::move(memoryGovernor), std::move(kvPages), std::move(stateStorage),
        std::move(kvTier), std::move(kvPool), std::move(cache),
        hostAvailableAtStart));
    result->adoptPersistentCache();
    return result;
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::StorageAllocation,
                                error.what(),
                                memoryPlan.breakdown().describe(),
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::StorageAllocation,
                                error.what(),
                                memoryPlan.breakdown().describe());
  }
}

RuntimeResources::~RuntimeResources() {
  stopProbation();
  if (persistentCache_.directory) {
    persistentCache_.kv->seal();
    persistentCache_.states->seal();
  }
}

void RuntimeResources::adoptPersistentCache() {
  if (!persistentCache_.directory)
    return;
  CacheDirectory &directory = *persistentCache_.directory;
  if (!directory.coldReason().empty())
    logLine("Persistent cache emptied: ", directory.coldReason(), ".");
  std::vector<PersistedKv> blocks;
  for (const model::SlotRecord &record : persistentCache_.kv->records())
    blocks.push_back({record.label, [this, record] { return kvTier_->adopt(record); }});
  std::vector<PersistedState> states;
  for (const model::SlotRecord &record : persistentCache_.states->records())
    states.push_back({record.label, [this, record](uint32_t tokens) {
                        return stateStorage_->adopt(record, tokens);
                      }});
  const CacheAdoption adoption = cache_->adopt(std::move(blocks), std::move(states));
  persistentCache_.kv->finishAdoption();
  persistentCache_.states->finishAdoption();
  logLine("Persistent cache ", directory.path().string(), ": took back ", adoption.states,
          " restore points over ", adoption.blocks, " KV blocks (", adoption.bytes / kMiB,
          " MiB); left ", adoption.dropped, " copies behind.",
          directory.uncleanExit()
              ? " The last process did not stop cleanly: this one serves on probation for "
                "its first minute."
              : "");
}

void RuntimeResources::beginServing() {
  CacheDirectory *directory = persistentCache_.directory.get();
  if (!directory)
    return;
  const bool probation = directory->uncleanExit();
  try {
    directory->beginServing(probation);
  } catch (const std::exception &error) {
    logLine("Persistent cache cannot mark this process serving (", error.what(),
            "); the next start will not know how it ended.");
    return;
  }
  if (!probation)
    return;
  probation_ = std::thread([this, directory] {
    std::unique_lock lock(probationMutex_);
    if (probationWake_.wait_until(lock, AwakeClock::now() + kProbation,
                                  [this] { return probationStopped_; }))
      return;
    try {
      directory->endProbation();
    } catch (const std::exception &error) {
      logLine("Persistent cache probation did not end (", error.what(), ").");
    }
  });
}

void RuntimeResources::stopProbation() noexcept {
  {
    std::lock_guard lock(probationMutex_);
    probationStopped_ = true;
  }
  probationWake_.notify_all();
  if (probation_.joinable())
    probation_.join();
}

bool RuntimeResources::closePersistentCache() {
  if (!persistentCache_.directory)
    return true;
  stopProbation();
  cache_->relabelStates();
  // Each file runs its operations in order: the writes and labels submitted
  // before land first.
  const auto kv = persistentCache_.kv->synchronize({});
  const auto states = persistentCache_.states->synchronize({});
  const bool kvSynchronized = kv->wait();
  if (!states->wait() || !kvSynchronized)
    return false;
  try {
    persistentCache_.directory->close();
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

model::RuntimeContext RuntimeResources::modelContext() noexcept {
  return {
      *backend_,
      model_,
      *kvPages_,
      *stateStorage_,
      operators_,
  };
}

ActualMemoryReport RuntimeResources::actualMemoryReport(
    const model::ModelMemoryActual &modelMemory) const {
  ActualMemoryReport report;
  report.targetWeightsBytes = model_.targetActualAllocatedBytes();
  report.draftWeightsBytes = model_.draft.actualAllocatedBytes;
  report.visionWeightsBytes = model_.vision.actualAllocatedBytes;
  report.stateAllocatedBytes = modelMemory.stateActualAllocatedBytes;
  report.sharedPrefillBytes = modelMemory.sharedPrefillActualAllocatedBytes;
  report.sharedDecodeBytes = modelMemory.sharedDecodeActualAllocatedBytes;
  if (kvPool_->allocatedBytes() != kvPages_->actualAllocatedBytes()) {
    throw std::logic_error("the KV pool and its storage disagree on allocated extents");
  }
  report.kvAllocatedBytes = kvPool_->allocatedBytes();
  report.stateStagingBytes = modelMemory.stateStagingBytes;
  // Optional warmup may end with a rolled-back allocation and no subsequent
  // command. Refresh the current counts after that rollback; peaks stay intact.
  metal::MetalMemoryStats memory = backend_->refreshMemoryStats();
  report.backendAllocatedBytes = memory.allocatedBytes;
  report.deviceCurrentAllocatedBytes = memory.deviceCurrentAllocatedBytes;
  report.devicePeakAllocatedBytes = memory.devicePeakAllocatedBytes;
  // A capacity-limited warmup can roll back a partial allocation before it
  // returns a result; the backend's own high-water mark keeps it.
  report.backendPeakAllocatedBytes = memory.peakAllocatedBytes;
  return report;
}

void connectToGovernor(EngineConfig &config, MemoryGovernor &governor) {
  config.growthPaused = [&governor] {
    return !governor.snapshot().hostGrowthAllowed;
  };
  config.serving = [&governor](bool serving) { governor.setServing(serving); };
}

} // namespace splash::engine
