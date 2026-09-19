#include "engine/RuntimeResources.hpp"
#include "metal/abi/ExecutionGeometry.h"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <ctime>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>

namespace splash::engine {
namespace {

template <typename... Parts>
void logKernelStartup(const Parts &...parts) noexcept {
  try {
    std::ostringstream text;
    (text << ... << parts);
    const std::string message = text.str();
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    char timestamp[9] = "--:--:--";
    if (localtime_r(&now, &local))
      std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", &local);
    std::ostringstream line;
    line << timestamp << ' ';
    // Native stderr is inherited by serve. Keep each optional startup notice
    // bounded and on one line, including messages from caught exceptions.
    for (unsigned char character : std::string_view(message).substr(0, 768))
      line << (character < 32 || character == 127 ? ' ' : char(character));
    if (message.size() > 768) line << "...";
    std::cerr << line.str() << '\n';
  } catch (...) {
    // Optional diagnostics must not affect startup or serving.
  }
}

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

std::string errorText(RuntimeResourceStage stage, std::string_view message,
                      std::string_view budgetDescription) {
  std::ostringstream out;
  out << "runtime resource assembly failed [" << runtimeResourceStageName(stage)
      << "]: " << message;
  if (!budgetDescription.empty()) {
    out << '\n' << budgetDescription;
  }
  return out.str();
}

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

uint64_t checkedAdd(uint64_t left, uint64_t right, std::string_view label) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    throw std::overflow_error(std::string(label) + " byte count overflows");
  }
  return left + right;
}

uint64_t packedModelFileBytes(const std::filesystem::path &root) {
  uint64_t bytes = 0;
  for (std::string_view directory : {"target", "draft", "vision"}) {
    const std::filesystem::path package = root / directory;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(package)) {
      if (!entry.is_regular_file())
        continue;
      bytes = checkedAdd(bytes, entry.file_size(), "model package");
    }
  }
  if (!bytes) {
    throw std::invalid_argument("model package contains no regular files");
  }
  return bytes;
}

uint64_t mebibytes(uint64_t bytes) noexcept { return bytes / kMiB; }

// The startup admission rule. Deliberately independent of the model size:
// weights are mapped, not copied, so the package never has to fit in
// reclaimable memory at once. Residency is what must fit, and it is checked
// here again at every Metal operation as loading and warmup build it up.
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

std::string sha256(std::string_view value) {
  if (value.size() > std::numeric_limits<CC_LONG>::max()) {
    throw std::overflow_error("runtime cache identity is too large to hash");
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256(value.data(), static_cast<CC_LONG>(value.size()),
                 digest.data())) {
    throw std::runtime_error("runtime cache identity SHA-256 failed");
  }
  return digestHex(digest);
}

std::string
canonicalRuntimeCacheNamespace(const RuntimeCacheIdentity &identity) {
  // Length-prefix the unconstrained strings; every other field has a fixed
  // name and decimal representation. This is the one semantic cache tuple,
  // never a hash of compiler padding or native struct bytes.
  std::ostringstream canonical;
  canonical << "splash.runtime-cache-identity\n"
            << "loaded_model_layout_sha256=" << identity.modelLayoutSha256
            << '\n'
            << "build_id_bytes=" << identity.buildId.size() << '\n'
            << "build_id=" << identity.buildId << '\n'
            << "dtype=" << kQ8FormatName << '\n'
            << "page_tokens=" << identity.q8Layout.pageTokens << '\n'
            << "elements_per_scale=" << identity.q8Layout.elementsPerScale
            << '\n'
            << "target_model_sha256="
            << digestHex(identity.q8Layout.modelArtifactSha256) << '\n'
            << "q8_quantization=" << identity.q8Layout.quantization << '\n'
            << "q8_scale_type=" << identity.q8Layout.scaleType << '\n'
            << "q8_key_layout=" << identity.q8Layout.keyLayout << '\n'
            << "q8_value_layout=" << identity.q8Layout.valueLayout << '\n'
            << "q8_attention_layers=" << identity.q8Layout.attentionLayers
            << '\n'
            << "q8_kv_heads=" << identity.q8Layout.kvHeads << '\n'
            << "q8_head_dimension=" << identity.q8Layout.headDimension << '\n'
            << "q8_quantized_minimum=" << identity.q8Layout.quantizedMinimum
            << '\n'
            << "q8_quantized_maximum=" << identity.q8Layout.quantizedMaximum
            << '\n'
            << "q8_bytes_per_layer_page=" << identity.q8Layout.bytesPerLayerPage
            << '\n'
            << "q8_bytes_per_model_page=" << identity.q8Layout.bytesPerModelPage
            << '\n';
  return sha256(canonical.str());
}

void requireLoadedModel(const model::ModelPackage &package) {
  if (!package.targetActualAllocatedBytes() ||
      !package.draft.actualAllocatedBytes ||
      !package.vision.actualAllocatedBytes ||
      package.manifestFingerprintSha256.empty() ||
      package.targetManifestFingerprint().empty()) {
    throw std::invalid_argument(
        "loaded model package has incomplete allocation accounting");
  }
}

} // namespace

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
                         kv::Q8Layout targetKvLayout) {
  if (buildId.empty()) {
    throw std::invalid_argument("runtime build id is required");
  }
  if (!targetKvLayout.valid()) {
    throw std::invalid_argument("runtime target KV layout is invalid");
  }
  // Parse both digests even though only the target digest belongs in the
  // physical-page ABI. This rejects malformed combined manifests early.
  std::array<uint8_t, 32> combinedDigest = parseSha256(combinedManifestSha256);
  std::array<uint8_t, 32> targetDigest = parseSha256(targetManifestSha256);
  RuntimeCacheIdentity result;
  result.modelLayoutSha256 = digestHex(combinedDigest);
  result.buildId = buildId;
  result.q8Layout = kv::makeQ8LayoutGuard(targetKvLayout, targetDigest);
  result.namespaceSha256 = canonicalRuntimeCacheNamespace(result);
  result.cacheNamespace.digest = parseSha256(result.namespaceSha256);
  return result;
}

RuntimeResourcesError::RuntimeResourcesError(RuntimeResourceStage stage,
                                             std::string message,
                                             std::string statusJson,
                                             std::string budgetDescription,
                                             RuntimeResourceFailure failure)
    : std::runtime_error(errorText(stage, message, budgetDescription)),
      failure_(failure),
      message_(std::move(message)), statusJson_(std::move(statusJson)),
      budgetDescription_(std::move(budgetDescription)) {}

RuntimeResources::RuntimeResources(
    std::unique_ptr<metal::MetalBackend> backend, model::ModelPackage model,
    ops::ExecutionPlans operators,
    EngineMemoryPlan memoryPlan, model::ModelMemoryPlan modelMemoryPlan,
    RuntimeCacheIdentity cacheIdentity,
    std::unique_ptr<MemoryGovernor> memoryGovernor,
    std::unique_ptr<kv::Q8PageStorage> kvPages,
    std::unique_ptr<model::StateStorage> stateStorage,
    std::unique_ptr<KvPool> kvPool, std::unique_ptr<engine::Cache> cache,
    uint32_t maximumImagePatches)
    : backend_(std::move(backend)), model_(std::move(model)),
      operators_(std::move(operators)),
      memoryPlan_(std::move(memoryPlan)),
      modelMemoryPlan_(std::move(modelMemoryPlan)),
      cacheIdentity_(std::move(cacheIdentity)),
      memoryGovernor_(std::move(memoryGovernor)), kvPages_(std::move(kvPages)),
      stateStorage_(std::move(stateStorage)), kvPool_(std::move(kvPool)),
      cache_(std::move(cache)), maximumImagePatches_(maximumImagePatches) {}

std::unique_ptr<RuntimeResources>
RuntimeResources::create(const RuntimeResourcesConfig &config) {
  if (config.metallibPath.empty() || config.modelRoot.empty() ||
      !config.model.valid() ||
      config.buildId.empty() || !config.maximumImagePatches ||
      config.maximumImagePatches % 4) {
    throw RuntimeResourcesError(
        RuntimeResourceStage::Configuration,
        "metallib path, model root, build id, and a merge-aligned image "
        "patch limit are required");
  }
  std::unique_ptr<metal::MetalBackend> backend;
  try {
    backend =
        std::make_unique<metal::MetalBackend>(config.metallibPath.string());
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::BackendCreation,
                                error.what(), {}, {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::BackendCreation,
                                error.what());
  }

  const DeviceCapabilities &device = backend->capabilities();
  if (auto error = device.validationError()) {
    throw RuntimeResourcesError(RuntimeResourceStage::CapabilityValidation,
                                *error, deviceStatusJson(device));
  }

  const uint64_t hostReserveBytes =
      EngineMemoryPolicy::hostAvailableReserveBytes(device.physicalMemoryBytes);
  MemoryGovernor::HostAvailableMemoryProvider hostAvailableMemory =
      config.hostAvailableMemory ? config.hostAvailableMemory
                                 : queryHostAvailableMemory;
  backend->setOperationGuard(
      [cancelled = config.cancelled, pressure = config.memoryPressure,
       hostAvailableMemory, hostReserveBytes] {
        if (cancelled && cancelled())
          throw metal::MetalBackendError("Metal operation cancelled");
        requireStartupHeadroom(hostAvailableMemory, hostReserveBytes,
            pressure ? pressure() : MemoryPressure::Normal);
      });
  try {
    const uint64_t modelBytes = packedModelFileBytes(config.modelRoot);
    const uint64_t hardBudgetBytes = EngineMemoryPolicy::hardBudgetBytes(
        device.recommendedMaxWorkingSetBytes, config.maximumMemoryBytes);
    // Reject an impossible weight budget before registering model buffers.
    // The full plan below still uses measured allocations and runtime costs.
    if (modelBytes > hardBudgetBytes) {
      throw RuntimeResourcesError(
          RuntimeResourceStage::MemoryPlanning,
          "model weights require " + std::to_string(modelBytes) +
              " bytes but the Metal memory budget is " +
              std::to_string(hardBudgetBytes) + " bytes",
          deviceStatusJson(device), {}, RuntimeResourceFailure::EngineCapacity);
    }
    // Fail before opening the package when the machine has no headroom at
    // all; the guard installed above keeps checking as residency grows.
    requireStartupHeadroom(hostAvailableMemory, hostReserveBytes,
        config.memoryPressure ? config.memoryPressure() : MemoryPressure::Normal);
  } catch (const RuntimeResourcesError &) {
    throw;
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), deviceStatusJson(device), {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), deviceStatusJson(device));
  }

  model::ModelPackage package;
  try {
    package = model::loadModelPackage(*backend, config.modelRoot, config.model);
    requireLoadedModel(package);
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), deviceStatusJson(device), {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), deviceStatusJson(device));
  }

  // One selection owner is used both before allocation and during encoding.
  // The engine lends it to model execution without inspecting kernel choices.
  ops::ExecutionPlans operators(device);
  model::ModelMemoryPlan modelMemoryPlan;
  auto prepareMemory = [&]() -> EngineMemoryPlan {
    try {
      modelMemoryPlan = model::plannedRuntimeMemory(device, package, operators);
      if (auto error = modelMemoryPlan.validationError()) {
        throw std::invalid_argument(*error);
      }
    } catch (const std::exception &error) {
      throw RuntimeResourcesError(
          RuntimeResourceStage::MemoryPlanning,
          std::string("model allocated-size plan is invalid: ") + error.what(),
          deviceStatusJson(device));
    }

    ModelMemoryFootprint footprint{
        package.targetActualAllocatedBytes(),
        package.draft.actualAllocatedBytes,
        package.vision.actualAllocatedBytes,
        modelMemoryPlan.activeStateCellPlannedAllocatedBytes,
        modelMemoryPlan.sharedPrefillPlannedAllocatedBytes,
        modelMemoryPlan.sharedDecodePlannedAllocatedBytes,
        modelMemoryPlan.pipelineReserveBytes,
        modelMemoryPlan.runtimeOverheadReserveBytes,
    };

    ModelMemoryProfile modelProfile{
        package.name(), package.maximumContextTokens(),
        package.targetKvLayout(), footprint};
    EngineMemoryPlanResult planResult =
        evaluateEngineMemoryPlan(device, modelProfile, config.maximumMemoryBytes);
    if (!planResult.plan) {
      throw RuntimeResourcesError(
          RuntimeResourceStage::MemoryPlanning, planResult.status.message,
          planResult.status.toStatusJson(), planResult.status.describe());
    }
    EngineMemoryPlan memoryPlan = std::move(*planResult.plan);
    return memoryPlan;
  };
  // Establish the serving baseline and the one real memory governor before
  // installing the shipped choices. Selected workspace never gets a separate
  // allowance or replaces the immutable engine-wide ceiling.
  EngineMemoryPlan memoryPlan = prepareMemory();

  RuntimeCacheIdentity cacheIdentity;
  try {
    cacheIdentity = makeRuntimeCacheIdentity(
        package.manifestFingerprintSha256,
        package.targetManifestFingerprint(), config.buildId,
        package.targetKvLayout());
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), memoryPlan.toStatusJson(),
                                memoryPlan.breakdown().describe());
  }

  try {
    const auto baselineMemoryPlan = memoryPlan;
    const auto baselineModelMemoryPlan = modelMemoryPlan;
    const EngineMemoryBreakdown &baselineBudget = baselineMemoryPlan.breakdown();
    const uint64_t runtimeReserve =
        baselineBudget.pipelineReserveBytes + baselineBudget.runtimeOverheadReserveBytes;
    if (baselineBudget.hardBudgetBytes <= runtimeReserve) {
      throw std::logic_error("runtime reserves consume the Metal budget");
    }
    // The governor observes the complete Metal footprint. Keeping explicit
    // pipeline/allocator reserves outside its growth ceiling prevents elastic
    // state and KV from silently consuming the startup safety margin.
    const uint64_t elasticGrowthCeiling =
        baselineBudget.hardBudgetBytes - runtimeReserve;
    auto memoryGovernor = std::make_unique<MemoryGovernor>(
        *backend, elasticGrowthCeiling, hostReserveBytes, hostAvailableMemory);
    if (config.memoryPressure)
      memoryGovernor->setPressure(config.memoryPressure());
    std::string rejected;
    auto adoptChoices = [&](const ops::OperatorChoices &choices) {
      try {
        operators.install(choices);
        auto selectedMemoryPlan = prepareMemory();
        const auto &selected = selectedMemoryPlan.breakdown();
        if (selected.pipelineReserveBytes + selected.runtimeOverheadReserveBytes !=
                runtimeReserve || selected.hardBudgetBytes != baselineBudget.hardBudgetBytes)
          throw std::logic_error("operator choices changed the memory governor ceiling");
        memoryPlan = std::move(selectedMemoryPlan);
        return true;
      } catch (const std::exception &error) {
        // An illegal table entry or a host/user limit that no longer fits the
        // selected scratch keeps the operator defaults, never a partial table.
        rejected = error.what();
        operators.install({});
        modelMemoryPlan = baselineModelMemoryPlan;
        memoryPlan = baselineMemoryPlan;
        return false;
      }
    };
    logKernelStartup("Kernel policy for GPU family ", device.appleGpuFamily,
                     " with ", device.gpuCoreCount, " cores.");
    if (config.operatorChoices && !config.operatorChoices->empty()) {
      if (adoptChoices(*config.operatorChoices))
        logKernelStartup("Installed supplied kernel choices.");
      else
        logKernelStartup("Supplied kernel choices rejected (", rejected,
                         "); using the kernel policy.");
    }

    const EngineMemoryBreakdown &budget = memoryPlan.breakdown();
    auto kvPages = std::make_unique<kv::Q8PageStorage>(
        *backend, memoryGovernor->allocationAdmission(), package.targetKvLayout(),
        budget.kvVirtualPages);
    auto stateStorage = model::createStateStorage(
        *backend, memoryGovernor->allocationAdmission(), package);
    if (!stateStorage) {
      throw std::runtime_error("model factory returned no state storage");
    }
    auto kvPool = std::make_unique<KvPool>(*kvPages);
    auto cache =
        std::make_unique<engine::Cache>(*kvPool, cacheIdentity.cacheNamespace);

    if (kvPages->declaredBytes() != budget.kvVirtualBytes ||
        kvPages->actualAllocatedBytes() > budget.kvVirtualBytes) {
      throw std::runtime_error(
          "actual Q8 page storage exceeds its planned category");
    }
    if (stateStorage->actualAllocatedBytes() != 0) {
      throw std::runtime_error("state cells were allocated eagerly");
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
        std::move(backend), std::move(package), std::move(operators),
        std::move(memoryPlan),
        std::move(modelMemoryPlan), std::move(cacheIdentity),
        std::move(memoryGovernor), std::move(kvPages), std::move(stateStorage),
        std::move(kvPool), std::move(cache), config.maximumImagePatches));
    return result;
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::StorageAllocation,
                                error.what(), memoryPlan.toStatusJson(),
                                memoryPlan.breakdown().describe(),
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::StorageAllocation,
                                error.what(), memoryPlan.toStatusJson(),
                                memoryPlan.breakdown().describe());
  }
}

model::RuntimeContext RuntimeResources::modelContext() noexcept {
  const EngineMemoryBreakdown &budget = memoryPlan_.breakdown();
  return {
      *backend_,
      memoryGovernor_->allocationAdmission(),
      model_,
      *kvPages_,
      *stateStorage_,
      operators_,
      maximumImagePatches_,
      budget.pipelineReserveBytes,
      budget.runtimeOverheadReserveBytes,
  };
}

ActualMemoryReport RuntimeResources::actualMemoryReport(
    const model::ModelMemoryActual &modelMemory,
    uint64_t estimatedWarmupPeakBytes) const {
  ActualMemoryReport report;
  report.targetWeightsBytes = model_.targetActualAllocatedBytes();
  report.draftWeightsBytes = model_.draft.actualAllocatedBytes;
  report.visionWeightsBytes = model_.vision.actualAllocatedBytes;
  report.stateResidentBytes = modelMemory.stateActualAllocatedBytes;
  report.sharedPrefillBytes = modelMemory.sharedPrefillActualAllocatedBytes;
  report.sharedDecodeBytes = modelMemory.sharedDecodeActualAllocatedBytes;
  report.kvResidentBytes = kvPages_->actualAllocatedBytes();
  // Optional warmup may end with a rolled-back allocation and no subsequent
  // command. Refresh current residency after that rollback; peaks stay intact.
  metal::MetalMemoryStats memory = backend_->refreshMemoryStats();
  if (memory.sparseResidentBytes >
      std::numeric_limits<uint64_t>::max() - memory.allocatedBytes) {
    throw std::overflow_error("backend memory accounting overflows");
  }
  report.backendAllocatedBytes =
      memory.allocatedBytes + memory.sparseResidentBytes;
  report.deviceCurrentAllocatedBytes = memory.deviceCurrentAllocatedBytes;
  report.devicePeakAllocatedBytes = memory.devicePeakAllocatedBytes;
  // A capacity-limited warmup can roll back a partial allocation before it
  // returns a result. Preserve that tracked high-water mark independently of
  // the device-wide measurement used by the audit.
  const auto &budget = memoryPlan_.breakdown();
  const uint64_t reserves =
      budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes;
  if (memory.peakResidentBytes >
      std::numeric_limits<uint64_t>::max() - reserves) {
    throw std::overflow_error("warmup memory estimate overflows");
  }
  report.estimatedWarmupPeakBytes =
      std::max(estimatedWarmupPeakBytes, memory.peakResidentBytes + reserves);
  return report;
}

} // namespace splash::engine
