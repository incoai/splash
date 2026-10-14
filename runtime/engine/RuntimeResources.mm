#include "engine/RuntimeResources.hpp"
#include "AwakeClock.hpp"
#include "Checked.hpp"
#include "StderrLine.hpp"
#include "engine/DiskLabels.hpp"
#include "engine/Engine.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/WeightStore.hpp"
#include "ops/AneFfnMeasurement.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <chrono>
#include <iomanip>
#include <limits>
#include <optional>
#include <span>
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
    const MemoryGovernor::HostMemoryProvider &hostMemory, uint64_t reserveBytes,
    MemoryPressure pressure) {
  const std::optional<HostMemory> host = hostMemory();
  const std::optional<uint64_t> available =
      host ? std::optional<uint64_t>(host->available) : std::nullopt;
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
// change to either, such as a fix to a kernel that writes KV or state or a
// change to the block identities its labels record (blockIdentity), bumps
// it, so that no start takes back copies of the old one.
constexpr uint32_t kPersistentCacheFormat = 2;
// How long a start waits for the cache directory another process holds: one
// that is closing has finished its flush, or been stopped by the server, by
// then (server/runtime.py `_shutdown_grace_seconds`).
constexpr std::chrono::seconds kCacheLockWait{20};
// A cache nobody opened or wrote for this long is removed at the next start,
// whether or not that start keeps a cache itself.
constexpr std::chrono::hours kStaleCache{24 * 14};
// How long a start that follows an unclean end serves on probation: the
// server counts a failure within that time toward a crash loop
// (server/backend.py `CRASH_LOOP_WINDOW_SECONDS`).
constexpr std::chrono::seconds kProbation{60};

// A cache root the engine made stays out of Time Machine backups, as
// ~/Library/Caches and $TMPDIR do: its files change with every write.
void excludeFromBackups(const std::filesystem::path &root) {
  NSURL *url = [NSURL fileURLWithPath:@(root.c_str()) isDirectory:YES];
  NSError *error = nil;
  if (![url setResourceValue:@YES forKey:NSURLIsExcludedFromBackupKey error:&error])
    logLine("Persistent cache root ", root.string(), " stays in Time Machine backups (",
            error.localizedDescription.UTF8String, ").");
}

// A persistent tier's directory under root and its files, empty when another
// process holds the directory or it cannot be used: the tier then keeps
// temporary files, as without one. A root it makes is excluded from
// backups.
PersistentCacheFiles openPersistentCache(const std::filesystem::path &root,
                                         const std::string &cacheNamespace,
                                         uint64_t kvSlotBytes, uint64_t stateSlotBytes,
                                         const std::shared_ptr<model::DiskBudget> &budget,
                                         const std::function<bool()> &cancelled) {
  try {
    if (root.empty())
      throw std::runtime_error("no home directory to keep it in");
    std::error_code ignored;
    const bool made = !std::filesystem::exists(root, ignored);
    PersistentCacheFiles files;
    files.directory = CacheDirectory::open(root, cacheNamespace, kCacheLockWait, cancelled);
    if (made && std::filesystem::exists(root, ignored))
      excludeFromBackups(root);
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

// The disk tier's startup line: where its files are, its limit and why, its
// write budget and what passes it when one is set, what it holds and stages
// through, and how to turn it off.
void reportDiskTier(const DiskSpace &space, const model::DiskBudget &budget, bool automatic,
                    uint64_t writesPerDay, std::string_view files, uint64_t kvSlotBytes,
                    uint64_t stateBytes) {
  std::ostringstream line;
  line << "SSD cache: up to " << gibibytes(budget.limitBytes()) << " of " << files;
  if (space.othersBytes())
    line << " beside " << gibibytes(space.othersBytes()) << " of other caches there";
  const std::optional<DiskVolume> &volume = space.volume();
  line << " (";
  if (!space.followed()) {
    line << "its quota, on a volume whose free space it does not follow";
  } else {
    const uint64_t floor = diskFloorBytes(volume->sizeBytes);
    if (budget.limitBytes() < budget.capacityBytes())
      line << "a quarter of the "
           << gibibytes(diskRoomBytes(volume->availableBytes, space.heldBytes(), floor))
           << " its volume has";
    else
      line << (automatic ? "its automatic quota" : "its quota") << ", while its volume has room";
    line << " beyond the " << gibibytes(floor) << " it keeps free";
  }
  line << ")";
  if (writesPerDay)
    line << "; a write budget of " << gibibytes(writesPerDay)
         << " a day, which the states conversations resume from and KV pages pass, starting with "
            "a quarter of a day's";
  line << "; ";
  if (kvSlotBytes)
    line << "KV pages of " << kvSlotBytes / 1024 << " KiB and ";
  line << "states of " << mebibytes(stateBytes) << " MiB, a state's write staging through "
       << mebibytes(stateBytes) << " MiB of the memory plan; --max-cache-disk 0 turns it off.";
  logLine(line.str());
}

// The layers' part of what ane::recall() keeps a calibration under: their
// shapes and formats.
std::string calibrationKey(std::span<const ops::SwiGluProjections> layers) {
  std::string key = "ane-ffn calibration";
  for (const ops::SwiGluProjections &layer : layers)
    for (const ops::Projection *projection : {layer.gate, layer.up, layer.down})
      key += " " + std::to_string(projection->outputSize) + "x" + std::to_string(projection->inputSize) + ":g" +
             std::to_string(projection->blocks().segments.front().formatId);
  return key;
}

// The split's part of what the engine gives back while idle, if it runs one.
std::optional<ReleasableMemory::Split> idleSplit(ops::AneFfn *split) {
  if (!split)
    return std::nullopt;
  return ReleasableMemory::Split{[split] { return split->release(); },
                                 [split] { split->restore(); }};
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

void requireLoadedModel(const model::LoadedModel &loaded) {
  if (!loaded.targetActualAllocatedBytes() ||
      (loaded.hasDraft() && !loaded.draft.actualAllocatedBytes) ||
      (loaded.descriptor.hasVision() && !loaded.vision.actualAllocatedBytes) ||
      loaded.manifestFingerprintSha256.empty() ||
      loaded.targetManifestFingerprint().empty()) {
    throw std::invalid_argument(
        "loaded model has incomplete allocation accounting");
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
            << ' ' << gdn.recurrentColumns
            // fp32 states keep the line they had before states had a format,
            // so the caches they wrote stay valid.
            << (gdn.format == ops::GdnStateFormat::BFloat16 ? " bf16" : "") << '\n'
            << "draft " << draft.layers << ' ' << draft.kvHeads << ' ' << draft.headDimension << ' '
            << SPLASH_DRAFT_SLIDING_WINDOW << '\n'
            << "context " << draft.contextWidth << " q4 " << SPLASH_DRAFT_CONTEXT_GROUP << '\n';
  // 128 bits name it.
  return model::weightDigest(canonical.str()).substr(0, CacheDirectory::kNamespaceDigits);
}

RuntimeResourcesError::RuntimeResourcesError(RuntimeResourceStage stage,
                                             std::string message,
                                             std::string budgetDescription,
                                             RuntimeResourceFailure failure)
    : std::runtime_error(std::move(message)), stage_(stage), failure_(failure),
      budgetDescription_(std::move(budgetDescription)) {}

RuntimeResources::RuntimeResources(
    PersistentCacheFiles persistentCache,
    std::unique_ptr<metal::MetalBackend> backend, model::LoadedModel model,
    ops::ExecutionPlans operators, EngineMemoryPlan memoryPlan,
    RuntimeCacheIdentity cacheIdentity,
    std::unique_ptr<MemoryGovernor> memoryGovernor,
    std::unique_ptr<kv::PageStorage> kvPages,
    std::unique_ptr<model::QwenStateStorage> stateStorage,
    std::unique_ptr<KvPageTier> kvTier, std::unique_ptr<DiskSpace> diskSpace,
    std::unique_ptr<KvPool> kvPool, std::unique_ptr<engine::Cache> cache,
    std::unique_ptr<ops::AneFfn> aneFfn, AneFfnOutcome aneFfnOutcome)
    : persistentCache_(std::move(persistentCache)),
      backend_(std::move(backend)), model_(std::move(model)),
      operators_(std::move(operators)),
      memoryPlan_(std::move(memoryPlan)),
      cacheIdentity_(std::move(cacheIdentity)),
      memoryGovernor_(std::move(memoryGovernor)), kvPages_(std::move(kvPages)),
      stateStorage_(std::move(stateStorage)), kvTier_(std::move(kvTier)),
      diskSpace_(std::move(diskSpace)), kvPool_(std::move(kvPool)),
      cache_(std::move(cache)), aneFfn_(std::move(aneFfn)),
      releasableMemory_(*model_.images, idleSplit(aneFfn_.get())),
      aneFfnOutcome_(std::move(aneFfnOutcome)) {}

std::unique_ptr<RuntimeResources>
RuntimeResources::create(const RuntimeResourcesConfig &config,
                         uint32_t requestedContextTokens) {
  if (config.metallibPath.empty() || config.modelRoot.empty() ||
      !kv::validFormat(config.kvFormat) || !ops::validFormat(config.stateFormat) ||
      !config.model.valid() || !config.hostMemory || !config.diskVolume ||
      config.buildId.empty() || !config.maximumImagePatches ||
      config.maximumImagePatches % 4 ||
      config.maximumImagePatches > ops::kMaximumImagePatches) {
    throw RuntimeResourcesError(
        RuntimeResourceStage::Configuration,
        "metallib path, model root, a valid model layout, KV and state format, "
        "host memory and volume probes, build id, and a merge-aligned image patch "
        "limit no larger than the protocol's are required");
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
  // Startup work stops on cancellation and keeps its reserve of host memory.
  const auto throwIfCancelled = [cancelled = config.cancelled] {
    if (cancelled && cancelled())
      throw metal::MetalBackendError("startup cancelled");
  };
  const auto currentPressure = [pressure = config.memoryPressure] {
    return pressure ? pressure() : MemoryPressure::Normal;
  };
  const auto admitMetalOperation = [throwIfCancelled, currentPressure,
                                    hostMemory = config.hostMemory, hostReserveBytes] {
    throwIfCancelled();
    requireStartupHeadroom(hostMemory, hostReserveBytes, currentPressure());
  };
  backend->setOperationGuard(admitMetalOperation);
  // A synchronous command wait gives up when the process shuts down, also
  // after startup.
  backend->setWaitInterrupt(config.cancelled);
  // One disk quota serves KV pages and states. Without room for a state,
  // disk KV cannot preserve a restorable prefix, so the tier stays off, and
  // no state's write needs the staging buffer the plan would set aside. The
  // plan never depends on the disk, so that every start of a model serves
  // the same context: an explicit quota that holds a state sets the staging
  // aside, and the automatic one where the plan holds the model and the
  // requested context beside it and the Neural Engine split does not need
  // its memory (below). The disk decides only whether the tier's files open,
  // once the plan is made: where they do not, the tier is off and the staging
  // stays unused. The volume sizes the tier once its files are open
  // (DiskSpace).
  model::CompositeStateLayout stateLayout = config.model.stateLayout;
  stateLayout.target.format = config.stateFormat;
  const uint64_t stateBytes = stateLayout.cachedBytes();
  const bool automaticDisk = config.cacheDisk.automatic;
  const uint64_t cacheDiskBytes = automaticDisk ? kAutomaticCacheDiskBytes : config.cacheDisk.bytes;
  // A state's write to the disk tier stages through one buffer of a state's
  // size. It is the backend's like every other, so the governor charges it
  // beside the weights and the plan sets it aside before it sizes KV.
  uint64_t stateStagingBytes = 0;
  if (cacheDiskBytes >= model::SlotFile::slotBytesFor(stateBytes))
    stateStagingBytes = stateBytes;
  else if (cacheDiskBytes)
    logLine("SSD cache off: its quota holds no state of ", mebibytes(stateBytes),
            " MiB; no state staging is set aside.");
  // The automatic tier turns off before anything is planned with it.
  const auto diskOff = [&](std::string_view reason) {
    logLine("SSD cache off: ", reason, "; no state staging is set aside.");
    stateStagingBytes = 0;
  };
  try {
    const uint64_t hardBudgetBytes = EngineMemoryPolicy::hardBudgetBytes(
        device.recommendedMaxWorkingSetBytes, config.maximumMemoryBytes);
    // Reject a model that cannot fit before loading its weights. Beside
    // them the plan needs at least the runtime reserves, one lane's state,
    // the KV runway and any disk tier state staging; the full plan below adds
    // the arenas.
    kv::Layout kvLayout = config.model.targetKvLayout;
    kvLayout.format = config.kvFormat;
    const uint64_t weightBytes = model::modelWeightBytes(config.modelRoot, config.model);
    const auto requiredFor = [&](uint64_t stagingBytes) {
      uint64_t fixedBytes = 0;
      for (const uint64_t bytes : {weightBytes, model::kPipelineReserveBytes,
                                   model::kRuntimeOverheadReserveBytes, stagingBytes}) {
        if (!checkedAdd(fixedBytes, bytes, fixedBytes))
          fixedBytes = std::numeric_limits<uint64_t>::max();
      }
      return minimumRequiredBytes(fixedBytes, stateLayout.laneBytes(), kvLayout)
          .value_or(std::numeric_limits<uint64_t>::max());
    };
    if (stateStagingBytes && automaticDisk && requiredFor(stateStagingBytes) > hardBudgetBytes)
      diskOff("the memory budget has no room for its " + std::to_string(mebibytes(stateBytes)) +
              " MiB of state staging beside the model");
    const uint64_t requiredBytes = requiredFor(stateStagingBytes);
    if (requiredBytes > hardBudgetBytes) {
      throw RuntimeResourcesError(
          RuntimeResourceStage::MemoryPlanning,
          "model weights with the runtime reserves, one lane's state, the KV "
          "runway and any SSD cache state staging require " +
              std::to_string(requiredBytes) +
              " bytes but the Metal memory budget is " +
              std::to_string(hardBudgetBytes) + " bytes",
          {}, RuntimeResourceFailure::EngineCapacity);
    }
    // Fail before loading the model when the machine has no headroom at
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

  model::LoadedModel loaded;
  try {
    const auto started = AwakeClock::now();
    loaded = model::loadModel(*backend, config.modelRoot, config.model);
    requireLoadedModel(loaded);
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
    modelMemoryPlan = model::plannedRuntimeMemory(loaded, operators, config.kvFormat,
                                                  config.stateFormat);
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(
        RuntimeResourceStage::MemoryPlanning,
        std::string("model allocated-size plan is invalid: ") + error.what());
  }
  // The engine's memory plan, with `stagingBytes` set aside for the disk
  // tier's state staging and `aneFfnBytes` for the prefill FFN's Neural Engine
  // split.
  const auto planWith = [&](uint64_t stagingBytes, uint64_t aneFfnBytes) {
    ModelMemoryFootprint footprint{
        loaded.targetActualAllocatedBytes(),
        loaded.draft.actualAllocatedBytes,
        loaded.vision.actualAllocatedBytes,
        modelMemoryPlan,
        stagingBytes,
        aneFfnBytes,
    };
    ModelMemoryProfile modelProfile{
        loaded.name(), loaded.maximumContextTokens(),
        loaded.targetKvLayout(config.kvFormat), footprint, config.stateFormat};
    return evaluateEngineMemoryPlan(device, modelProfile, config.maximumMemoryBytes);
  };
  // The plan with the staging the tier keeps.
  const auto planMemory = [&](uint64_t aneFfnBytes) { return planWith(stateStagingBytes, aneFfnBytes); };
  // The model's side of the split (startAneFfn).
  const AneFfnModel splitModel =
      aneFfnModel(*backend, loaded, operators, config.kvFormat, config.buildId, config.cancelled);
  EngineMemoryPlanResult planResult = planMemory(0);
  // The automatic tier keeps its staging only where the plan holds the model
  // and the requested context beside it. The split wins over it: where the
  // plan refuses the split beside the staging but not without it, the tier
  // turns off before the split is decided. A relaunch decides it as its first
  // engine, an automatic start, did, so that it keeps the staging that engine
  // kept; only its split leaves that engine's context (below).
  if (stateStagingBytes && automaticDisk) {
    const auto withoutStaging = [&](uint64_t aneFfnBytes) { return planWith(0, aneFfnBytes); };
    if (!planResult.plan ||
        planResult.plan->maximumContextTokens() < std::max<uint32_t>(1, requestedContextTokens))
      diskOff(requestedContextTokens
                  ? "the memory plan cannot hold a " + std::to_string(requestedContextTokens) +
                        "-token context beside its state staging"
                  : std::string("the memory plan cannot hold the model beside its state staging"));
    else if (aneFfnRefused(splitModel, config.aneFfn, requestedContextTokens, planMemory) &&
             !aneFfnRefused(splitModel, config.aneFfn, requestedContextTokens, withoutStaging))
      diskOff(requestedContextTokens
                  ? "the memory plan cannot hold the Neural Engine FFN split and a " +
                        std::to_string(requestedContextTokens) +
                        "-token context beside its state staging"
                  : std::string("the memory plan cannot hold the Neural Engine FFN split and any "
                                "context beside its state staging"));
    if (!stateStagingBytes)
      planResult = planMemory(0);
  }
  if (!planResult.plan) {
    throw RuntimeResourcesError(RuntimeResourceStage::MemoryPlanning,
                                planResult.status.message,
                                planResult.status.describe());
  }
  EngineMemoryPlan memoryPlan = std::move(*planResult.plan);

  RuntimeCacheIdentity cacheIdentity;
  try {
    cacheIdentity = makeRuntimeCacheIdentity(
        loaded.manifestFingerprintSha256,
        loaded.targetManifestFingerprint(), config.buildId,
        loaded.targetKvLayout(config.kvFormat));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(),
                                memoryPlan.breakdown().describe());
  }

  try {
    // The governor holds the complete Metal footprint to the hard budget. The
    // plan budgets pipelines and driver allocations inside the pipeline and
    // allocator reserves, so memory outside the backend's buffers is charged
    // only beyond them, and elastic state and KV never grow into them. The
    // split only moves bytes from KV to fixed runtime memory, so the governor
    // is the same with it or without.
    auto memoryGovernor = std::make_unique<MemoryGovernor>(
        *backend, memoryPlan.breakdown().hardBudgetBytes, hostReserveBytes,
        EngineMemoryPolicy::hostSpareReserveBytes(device.physicalMemoryBytes), config.hostMemory,
        memoryPlan.breakdown().pipelineReserveBytes + memoryPlan.breakdown().runtimeOverheadReserveBytes);
    if (config.memoryPressure)
      memoryGovernor->setPressure(config.memoryPressure());
    // What the start allocates, the KV runway, the arenas and the warmup, is
    // memory a request in service needs: like loading, it needs only the
    // reserve macOS keeps short of critical pressure, which every Metal
    // operation checks. The host's margins and macOS's warning hold back
    // only what serving adds (beginServing).
    memoryGovernor->setServing(true);
    logLine("Kernel policy for GPU family ", device.appleGpuFamily,
            " with ", device.gpuCoreCount, " cores.");
    // The split leaves the context asked for, else a relaunch's.
    const RequestedContext splitContext =
        requestedContextTokens ? RequestedContext{.tokens = requestedContextTokens}
                               : RequestedContext{.tokens = config.relaunchContext,
                                                  .relaunch = config.relaunchContext != 0};
    AneFfnStart aneFfn =
        startAneFfn(splitModel, config.aneFfn, splitContext, planMemory, config.cancelled);
    if (aneFfn.plan)
      memoryPlan = std::move(*aneFfn.plan);
    const EngineMemoryBreakdown &budget = memoryPlan.breakdown();

    // Page ids for every extent the hard budget could hold: the governor,
    // never the id range, limits the pool.
    const uint64_t poolExtents = std::min<uint64_t>(
        (budget.hardBudgetBytes + budget.kvExtentBytes - 1) / budget.kvExtentBytes,
        std::numeric_limits<uint32_t>::max() / budget.kvExtentPages);
    auto kvPages = std::make_unique<kv::PageStorage>(
        *backend, memoryGovernor->allocationAdmission(), loaded.targetKvLayout(config.kvFormat),
        static_cast<uint32_t>(poolExtents * budget.kvExtentPages), budget.kvExtentPages);
    auto kvPool = std::make_unique<KvPool>(*kvPages, model::ExecutionLimits::warmupKvPages);
    // The tier's files open now, where the disk takes them: the plan set the
    // staging aside without asking it. A persistent tier keeps its files in a
    // cache directory of its own; when another process holds it or it cannot
    // be used, the tier keeps temporary files, as without one. The automatic
    // tier opens files only on a volume where its limit would hold two states
    // and 1 GiB of KV: a persistent tier's root, whose caches nobody holds
    // count as the tier's, else the temporary files'. Where it opens none,
    // the tier is off and the staging stays unused.
    std::shared_ptr<model::DiskBudget> diskBudget;
    std::shared_ptr<model::SlotFile> stateFile;
    PersistentCacheFiles persistentCache;
    const std::string cacheNamespace =
        persistentCacheNamespace(cacheIdentity, loaded.stateLayout(config.stateFormat));
    // The cache root's caches nobody holds, this start's own included, looked
    // at once: they are room an automatic persistent tier may take, the stale
    // ones go once this start holds its own, and the rest share the root's
    // limit with a persistent tier (DiskSpace).
    const std::vector<CacheDirectory::Idle> rootCaches =
        config.cacheRoot.empty() ? std::vector<CacheDirectory::Idle>{}
                                 : CacheDirectory::idle(config.cacheRoot);
    if (stateStagingBytes) {
      // A write budget, when one is set, refills on the awake clock the
      // engine schedules its wake-ups on (awakeMilliseconds), so a write it
      // holds back is taken up again when the budget admits it. Without one
      // every write passes.
      diskBudget = std::make_shared<model::DiskBudget>(cacheDiskBytes);
      if (config.cacheDiskWritesPerDay)
        diskBudget->setWriteBudget(config.cacheDiskWritesPerDay);
      const uint64_t stateSlotBytes = model::SlotFile::slotBytesFor(stateBytes);
      std::optional<std::string> rootRefusal, temporaryRefusal;
      if (automaticDisk) {
        const uint64_t minimumBytes = 2 * stateSlotBytes + (uint64_t{1} << 30);
        std::error_code ignored;
        temporaryRefusal = automaticDiskRefusal(
            config.diskVolume(std::filesystem::temp_directory_path(ignored)), 0, minimumBytes);
        if (config.persistentCache && !config.cacheRoot.empty()) {
          uint64_t heldBytes = 0;
          for (const CacheDirectory::Idle &cache : rootCaches)
            heldBytes += cache.bytes;
          rootRefusal =
              automaticDiskRefusal(config.diskVolume(config.cacheRoot), heldBytes, minimumBytes);
        }
      }
      if (config.persistentCache && !rootRefusal) {
        persistentCache = openPersistentCache(
            config.cacheRoot, cacheNamespace, model::SlotFile::slotBytesFor(kvPages->bytesPerPage()),
            stateSlotBytes, diskBudget, config.cancelled);
        stateFile = persistentCache.states;
      }
      std::optional<std::string> off;
      if (!stateFile && temporaryRefusal) {
        if (rootRefusal)
          off = "for its root, " + *rootRefusal + "; for temporary files, " + *temporaryRefusal;
        else
          off = config.persistentCache ? "for temporary files, " + *temporaryRefusal : *temporaryRefusal;
      } else if (!stateFile) {
        if (rootRefusal)
          logLine("Persistent cache off: ", *rootRefusal, "; the SSD cache keeps temporary files.");
        try {
          stateFile = std::make_shared<model::SlotFile>(stateSlotBytes, diskBudget);
        } catch (const std::exception &error) {
          off = error.what();
        }
      }
      if (off) {
        logLine("SSD cache off: ", *off, "; the state staging the plan set aside stays unused.");
        diskBudget.reset();
      }
    }
    // After the open, which holds this start's own cache: a held one stays.
    // The stale ones go, and the others share the root's limit with the
    // tier.
    std::vector<CacheDirectory::Idle> otherCaches =
        CacheDirectory::removeStale(rootCaches, kStaleCache);
    std::erase_if(otherCaches, [&](const CacheDirectory::Idle &cache) {
      return cache.path.filename() == cacheNamespace;
    });
    auto stateStorage = std::make_unique<model::QwenStateStorage>(
        *backend, memoryGovernor->allocationAdmission(), loaded.stateLayout(config.stateFormat),
        stateFile);
    std::unique_ptr<KvPageTier> kvTier;
    std::unique_ptr<DiskSpace> diskSpace;
    if (diskBudget) {
      const uint64_t slotBytes = model::SlotFile::slotBytesFor(kvPages->bytesPerPage());
      std::shared_ptr<model::SlotFile> kvFile = persistentCache.kv;
      try {
        if (!kvFile)
          kvFile = std::make_shared<model::SlotFile>(slotBytes, diskBudget);
        kvTier = std::make_unique<KvPageTier>(*kvPages, kvFile);
      } catch (const std::exception &error) {
        // The persistent files are open by now; without the tier nothing
        // would keep or replace their copies.
        if (persistentCache.directory)
          throw;
        kvFile.reset();
        logLine("SSD cache KV storage disabled; state storage remains enabled (",
                error.what(), ").");
      }
      // The volume sizes the tier before it takes anything back. A
      // persistent tier's root holds the other caches nobody holds too.
      std::vector<std::shared_ptr<model::SlotFile>> files{stateFile};
      if (kvFile)
        files.push_back(kvFile);
      diskSpace = std::make_unique<DiskSpace>(
          diskBudget, std::move(files),
          persistentCache.directory ? std::move(otherCaches) : std::vector<CacheDirectory::Idle>{});
      reportDiskTier(*diskSpace, *diskBudget, automaticDisk, config.cacheDiskWritesPerDay,
                     persistentCache.directory
                         ? "persistent files in " + persistentCache.directory->path().string()
                         : "temporary files in " +
                               std::filesystem::temp_directory_path().string(),
                     kvFile ? slotBytes : 0, stateBytes);
    }
    auto cache = std::make_unique<engine::Cache>(*kvPool, kvTier.get(), diskBudget);
    if (diskSpace)
      cache->setDiskMeasure([space = diskSpace.get()] { space->measure(); });

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
        std::move(persistentCache), std::move(backend), std::move(loaded),
        std::move(operators), std::move(memoryPlan), std::move(cacheIdentity),
        std::move(memoryGovernor), std::move(kvPages), std::move(stateStorage),
        std::move(kvTier), std::move(diskSpace), std::move(kvPool), std::move(cache),
        std::move(aneFfn.split), std::move(aneFfn.outcome)));
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
  memoryGovernor_->setServing(false);
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
      aneFfn_.get(),
  };
}

AneFfnSnapshot RuntimeResources::aneFfnSnapshot() const {
  if (!aneFfn_)
    return {.reason = aneFfnOutcome_.reason};
  const bool stopped = aneFfn_->retired();
  const ops::AneFfn::Served &served = aneFfn_->served();
  return {stopped ? AneFfnSnapshot::State::Stopped : AneFfnSnapshot::State::Split,
          stopped ? aneFfn_->reason() : aneFfnOutcome_.reason,
          aneFfn_->share(),
          aneFfn_->minimumRows(),
          served.commands,
          served.evaluations,
          served.milliseconds};
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
  report.aneFfnBytes = aneFfn_ ? aneFfn_->allocatedBytes() : 0;
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

AneFfnModel aneFfnModel(metal::MetalBackend &backend, const model::LoadedModel &loaded,
                        const ops::ExecutionPlans &operators, kv::Format format, std::string_view buildId,
                        std::function<bool()> cancelled) {
  auto layers = std::make_shared<const std::vector<ops::SwiGluProjections>>(model::aneFfnLayers(loaded));
  AneFfnModel model;
  model.dense = !layers->empty();
  if (!model.dense) return model;
  if (const char *reason = ops::AneFfn::unsupported(*layers)) {
    model.unsupported = reason;
    return model;
  }
  model.unavailable = [] { return ane::unavailable(); };
  model.units = ops::AneFfn::units(*layers);
  model.plannedBytes = [layers](uint32_t aneUnits) { return ops::AneFfn::plannedBytes(*layers, aneUnits); };
  const auto onArena = [&backend, &loaded, &operators, format](const auto &use) {
    model::withPrefillArena(backend, loaded, operators, format, use);
  };
  model.time = [&backend, layers, cancelled, onArena] {
    ops::ane_ffn::Timings timings;
    onArena([&](const ops::PrefillFfnBuffers &ffn, const std::array<metal::MetalBuffer, 2> &hidden) {
      timings = ops::ane_ffn::Measurement(backend, *layers, ffn, hidden, cancelled).time();
    });
    return timings;
  };
  model.prepare = [&backend, layers, cancelled, onArena](uint32_t aneUnits, bool timeChunks) {
    AneFfnPrepared prepared;
    onArena([&](const ops::PrefillFfnBuffers &ffn, const std::array<metal::MetalBuffer, 2> &hidden) {
      prepared.split = std::make_unique<ops::AneFfn>(backend, *layers, aneUnits, cancelled);
      prepared.error = prepared.split->verify(*layers, ffn, hidden);
      if (timeChunks)
        prepared.chunks =
            ops::ane_ffn::Measurement(backend, *layers, ffn, hidden, cancelled).chunks(*prepared.split);
    });
    return prepared;
  };
  const std::string key = calibrationKey(*layers) + " device " + backend.capabilities().deviceName + " macos " +
                          [[NSProcessInfo processInfo] operatingSystemVersionString].UTF8String + " build " +
                          std::string(buildId) + " most ";
  model.recall = [key](uint32_t maxAneUnits) -> std::optional<ops::ane_ffn::Calibration> {
    const std::optional<std::string> line = ane::recall(key + std::to_string(maxAneUnits));
    return line ? ops::ane_ffn::Calibration::parse(*line) : std::nullopt;
  };
  model.remember = [key](uint32_t maxAneUnits, const ops::ane_ffn::Calibration &calibration) {
    ane::remember(key + std::to_string(maxAneUnits), calibration.text());
  };
  model.forget = [key](uint32_t maxAneUnits) { ane::forget(key + std::to_string(maxAneUnits)); };
  model.healthy = [&backend] { return backend.healthy(); };
  return model;
}

void connectToGovernor(EngineConfig &config, MemoryGovernor &governor) {
  config.growthPaused = [&governor] {
    return !governor.snapshot().hostGrowthAllowed;
  };
  config.serving = [&governor](bool serving) { return governor.setServing(serving); };
}

} // namespace splash::engine
