#include "TestModel.hpp"
#include "engine/RuntimeResources.hpp"
#include "engine/Status.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

EngineMemoryPlan plan() {
  DeviceCapabilities device;
  device.deviceName = "test";
  device.appleGpuFamily = 9;
  device.macosMajor = 26;
  device.macosMinor = 4;
  device.physicalMemoryBytes = 32 * kGiB;
  device.recommendedMaxWorkingSetBytes = 24 * kGiB;
  device.maxBufferLengthBytes = 16 * kGiB;
  device.maxThreadgroupMemoryBytes = 32 * 1024;
  device.maxThreadgroupWidth = 1024;
  device.hasUnifiedMemory = true;
  device.supportsPlacementSparse = true;
  return requireEngineMemoryPlan(
      device, test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB));
}

MemoryAuditResult audit(const EngineMemoryPlan &memoryPlan) {
  const auto &b = memoryPlan.breakdown();
  ActualMemoryReport actual;
  actual.targetWeightsBytes = b.targetWeightsBytes;
  actual.draftWeightsBytes = b.draftWeightsBytes;
  actual.visionWeightsBytes = b.visionWeightsBytes;
  actual.stateResidentBytes = b.activeStateCellBytes;
  actual.sharedPrefillBytes = b.sharedPrefillBytes;
  actual.sharedDecodeBytes = b.sharedDecodeBytes;
  actual.kvResidentBytes = b.kvExtentBytes;
  actual.backendAllocatedBytes =
      actual.targetWeightsBytes + actual.draftWeightsBytes +
      actual.visionWeightsBytes + actual.stateResidentBytes +
      actual.sharedPrefillBytes + actual.sharedDecodeBytes +
      actual.kvResidentBytes;
  actual.deviceCurrentAllocatedBytes = actual.backendAllocatedBytes;
  actual.devicePeakAllocatedBytes = actual.backendAllocatedBytes;
  actual.estimatedWarmupPeakBytes = actual.backendAllocatedBytes;
  return auditActualMemory(memoryPlan, actual);
}

void testCleanRuntimeStatus() {
  EngineMemoryPlan memoryPlan = plan();
  engine::EngineSnapshot engine;
  engine.maximumContextTokens = 102400;
  require(memoryPlan.maximumContextTokens() > engine.maximumContextTokens,
          "context regression needs a model capacity larger than the limit");
  engine.submitted = 3;
  engine.completed = 2;
  engine.cacheHits = 1;
  engine.coldMisses = 2;
  engine.reusedTokens = 64;
  engine.junctionMaterializations = 1;
  engine.checkpointPublications = 3;
  engine.checkpointPublicationFailures = 1;
  engine.resourceReplayTokens = 1234;
  engine.deduplicatedStatePublications = 2;
  engine.recycledStatePublications = 1;
  engine.scheduler.waitingPrefix = 3;
  engine.scheduler.prefillBatches = 4;
  engine.scheduler.prefillRows = 4096;
  engine.scheduler.decodeBatches = 4;
  engine.scheduler.decodeBatchesByWidth = {1, 1, 1, 1};
  engine.scheduler.decodeMixedGreedySamplingBatches = 2;
  engine.resources.pool = {256, 200, 24, 32, 128, 72, 1, 128 * 4096ULL,
                           32 * 4096ULL};
  engine.resources.kvCache = {32, 32 * 4096ULL};
  engine.resources.stateCache = {2, 0, 128, 1, 1, 2, 0};
  engine.resources.stateCache.checkpointEntries = 1;
  engine.resources.stateCache.checkpointBytes = 64;
  engine.resources.stateCache.checkpointEvictions = 4;
  engine.resources.stateCache.checkpointRetirements = 3;
  engine.resources.lookup = {3, 128, 64, 1};
  engine.resources.activeRequests = 1;

  WarmupReport warmup;
  warmup.maximumPrefill = WarmupStepStatus::Complete;
  warmup.decodeBatches.fill(WarmupStepStatus::Complete);
  warmup.draftVerifyCommit = WarmupStepStatus::Complete;
  warmup.compositeStateRestore = WarmupStepStatus::Complete;
  warmup.memoryBudgetValidated = true;
  warmup.maximumPrefillDetail = "packed_rows=2048";

  RuntimeMetricsSnapshot metrics;
  metrics.prefillInputTokens = 4096;
  metrics.decodeOutputTokens = 32;
  metrics.draftedTokens = 28;
  metrics.acceptedDraftTokens = 20;
  metrics.draftAcceptanceRate = 20.0 / 28.0;
  metrics.currentDecodeBatch = {true, 3, 0, 4, 21, 15, 1.5, 2666.0};

  engine::RuntimeCacheIdentity identity;
  identity.modelLayoutSha256 = std::string(64, 'a');
  identity.buildId = "build";
  identity.kvLayout = kv::makeLayoutGuard({16, 4, 256}, {});
  identity.kvLayout.modelArtifactSha256.fill(0xbc);
  identity.namespaceSha256 = std::string(64, 'd');

  metal::MetalMemoryStats metal;
  metal.allocatedBytes = 4 * kGiB;
  metal.peakAllocatedBytes = metal.allocatedBytes;
  metal.sparseVirtualBytes = memoryPlan.breakdown().kvVirtualBytes;
  metal.sparseResidentBytes = memoryPlan.breakdown().kvExtentBytes;
  metal.peakSparseResidentBytes = metal.sparseResidentBytes;
  metal.peakResidentBytes = metal.allocatedBytes + metal.sparseResidentBytes;
  metal.deviceCurrentAllocatedBytes = metal.allocatedBytes;
  metal.devicePeakAllocatedBytes = metal.allocatedBytes;
  metal.sparseTileBytes = 65536;
  metal.pendingSparseUnmaps = 1;
  metal.pendingSparseUnmapSeconds = 0.0125;
  metal.completedSparseUnmaps = 7;
  metal.lastSparseUnmapSeconds = 0.05;
  metal.maxSparseUnmapSeconds = 0.3;
  metal.sparseMapWaitEvent = 149;
  metal.pendingSparseMapWaitSeconds = 0.5;
  metal.lastSparseMapWaitSeconds = 0.25;
  metal.maxSparseMapWaitSeconds = 0.75;

  MemoryGovernorSnapshot governor;
  governor.limitBytes = memoryPlan.breakdown().hardBudgetBytes;
  governor.observedResidentBytes = metal.allocatedBytes;
  governor.headroomBytes = governor.limitBytes - governor.observedResidentBytes;
  governor.hostMeasurementValid = true;
  governor.hostAvailableBytes = 8 * kGiB;
  governor.hostReserveBytes = 2 * kGiB;
  governor.hostHeadroomBytes = 6 * kGiB;
  governor.growthAllowed = true;

  model::ModelTelemetry executorTelemetry;
  executorTelemetry.stateResidentBytes = 350'224'384;
  executorTelemetry.warmIdleStateCells = 1;
  executorTelemetry.targetPrefillRows = 10000;
  executorTelemetry.draftContextRowsActive = 2048;
  executorTelemetry.draftContextRowsMaterialization = 31;
  executorTelemetry.draftContextRowsAvoided = 7921;
  executorTelemetry.draftStateRestoreSkipped = 1;
  executorTelemetry.draftStateResets = 2;
  executorTelemetry.constrainedMaskOverlapBatches = 5;
  executorTelemetry.constrainedMaskOverlapRequests = 8;
  executorTelemetry.lastConstrainedTargetForwardGpuSeconds = 0.0725;
  executorTelemetry.totalConstrainedTargetForwardGpuSeconds = 0.25;
  executorTelemetry.lastConstrainedMaskWaitSeconds = 0.0015;
  executorTelemetry.totalConstrainedMaskWaitSeconds = 0.012;
  executorTelemetry.lastPrefillGpuSeconds = 5.25;
  executorTelemetry.lastPrefillWallSeconds = 116.921479;
  executorTelemetry.totalPrefillGpuSeconds = 500.5;
  executorTelemetry.totalPrefillWallSeconds = 700.25;
  executorTelemetry.lastDecodeGpuSeconds = 0.0725;
  executorTelemetry.lastDecodeWallSeconds = 0.083;
  executorTelemetry.totalDecodeGpuSeconds = 1.125;
  executorTelemetry.totalDecodeWallSeconds = 1.5;
  executorTelemetry.imageEncodes = 3;
  executorTelemetry.imageEmbeddingReuses = 4;
  const std::string json =
      runtimeStatusJson(memoryPlan, engine, metal, warmup, audit(memoryPlan),
                        metrics, executorTelemetry, identity, governor, true);
  require(json.find("\"kv\":{\"target_model_sha256\"") != std::string::npos &&
              json.find("\"q8\":{\"target_model_sha256\"") != std::string::npos,
          "INT8 status lost its generic or legacy identity");
  auto bf16Identity = identity;
  bf16Identity.kvLayout = kv::makeLayoutGuard({16, 4, 256, kv::Format::BFloat16}, {});
  const auto bf16Status = runtimeStatusJson(memoryPlan, engine, metal, warmup, audit(memoryPlan),
                        metrics, executorTelemetry, bf16Identity, governor, true);
  require(bf16Status.find("\"format\":\"bf16\"") != std::string::npos &&
              bf16Status.find("\"scale_type\":\"none\"") != std::string::npos &&
              bf16Status.find("\"q8\":") == std::string::npos,
          "BF16 cache identity advertised INT8 storage");
  require(json.find("\"schema_version\":5") != std::string::npos &&
              json.find("\"ready\":true") != std::string::npos,
          "status readiness/schema is wrong");
  require(json.find("\"ready\":true,\"maximum_context_tokens\":102400,") !=
              std::string::npos,
          "status advertised model capacity instead of the active engine limit");
  require(json.find("\"images\":{\"encodes\":3,\"embedding_reuses\":4}") !=
              std::string::npos,
          "image telemetry is missing from status");
  require(json.find("\"model_timing\":{\"scope\":\"model_lifetime\","
                    "\"prefill\":{\"last_gpu_ms\":5250,\"last_wall_ms\":116921.479,"
                    "\"total_gpu_ms\":500500,\"total_wall_ms\":700250},"
                    "\"decode\":{\"last_gpu_ms\":72.5,\"last_wall_ms\":83,"
                    "\"total_gpu_ms\":1125,\"total_wall_ms\":1500}}") !=
              std::string::npos,
          "status lost model GPU/wall timing, scope, or milliseconds units");

  const std::string unmeasured =
      runtimeStatusJson(memoryPlan, engine, metal, warmup, audit(memoryPlan),
                        metrics, {}, identity, governor, true);
  require(unmeasured.find("\"model_timing\":{\"scope\":\"model_lifetime\","
                          "\"prefill\":{\"last_gpu_ms\":0,\"last_wall_ms\":0,"
                          "\"total_gpu_ms\":0,\"total_wall_ms\":0},"
                          "\"decode\":{\"last_gpu_ms\":0,\"last_wall_ms\":0,"
                          "\"total_gpu_ms\":0,\"total_wall_ms\":0}}") !=
              std::string::npos,
          "status invented model timings from request metrics");
  require(json.find("\"pages_free\":200,\"pages_free_resident\":72,") !=
              std::string::npos,
          "status confused free virtual KV pages with resident free pages");
  require(json.find("\"sparse_tile_bytes\":65536,\"pending_unmaps\":1,"
                    "\"pending_unmap_ms\":12.5,\"unmaps_completed\":7,"
                    "\"unmap_last_ms\":50,\"unmap_max_ms\":300,"
                    "\"map_wait_event\":149,\"pending_map_wait_ms\":500,"
                    "\"map_wait_last_ms\":250,\"map_wait_max_ms\":750}") !=
              std::string::npos,
          "status lost the paced sparse release diagnostics");
  require(json.find("\"system_pressure\":\"normal\"") != std::string::npos &&
              json.find("\"host_measurement_valid\":true") != std::string::npos &&
              json.find("\"host_headroom_bytes\":" + std::to_string(6 * kGiB)) !=
                  std::string::npos,
          "status omitted the host-side growth constraints");
  require(json.find("\"resident_bytes\":350224384") != std::string::npos &&
              json.find("\"warm_idle_cells\":1") != std::string::npos &&
              json.find("\"scope\":\"startup_warmup\"") != std::string::npos,
          "live state residency or audit scope is missing from status");
  require(json.find("\"checkpoint_entries\":1,\"checkpoint_bytes\":64,"
                    "\"checkpoint_evictions\":4,\"checkpoint_retirements\":3") !=
              std::string::npos,
          "temporary state occupancy and retirement are missing from status");
  require(json.find("\"block_tokens\":32") != std::string::npos &&
              json.find("\"decode_batches_by_width\":{\"b1\":1,\"b2\":1,\"b3\":"
                        "1,\"b4\":1}") != std::string::npos,
          "Page32 or real B3 status is missing");
  require(json.find("\"decode_mixed_greedy_sampling_batches\":2}") !=
              std::string::npos,
          "status lost the mixed greedy/sampling decode count");
  auto idleEngine = engine;
  idleEngine.scheduler = {};
  const auto idleJson = runtimeStatusJson(
      memoryPlan, idleEngine, metal, warmup, audit(memoryPlan), metrics,
      executorTelemetry, identity, governor, true);
  require(idleJson.find("\"decode_mixed_greedy_sampling_batches\":0}") !=
              std::string::npos,
          "status omitted the zero mixed decode count");
  require(
      json.find("\"dynamic_budget_bytes\"") != std::string::npos &&
          json.find("\"resource_replay_tokens\":1234") != std::string::npos &&
          json.find("\"waiting_prefix\":3") != std::string::npos &&
          json.find("\"deduplicated_state_publications\":2,"
                    "\"recycled_state_publications\":1,") !=
              std::string::npos &&
          json.find("\"lazy_junctions\":1") != std::string::npos &&
          json.find("\"checkpoint_publications\":3,"
                    "\"checkpoint_publication_failures\":1") !=
              std::string::npos &&
          json.find("\"draft_context\":{\"target_prefill_rows\":10000,\"active_"
                    "rows\":2048,\"materialization_rows\":31,\"avoided_rows\":"
                    "7921,\"restore_skipped\":1,\"resets\":2}") !=
              std::string::npos &&
          json.find("\"constraint_masks\":{\"overlap_batches\":5,\"overlap_"
                    "requests\":8,\"last_target_forward_gpu_ms\":72.5,"
                    "\"total_target_forward_gpu_ms\":250,\"last_residual_"
                    "wait_ms\":1.5,\"total_residual_wait_ms\":12}") !=
              std::string::npos,
      "elastic KV-first status is incomplete");

  warmup.decodeBatches[2] = WarmupStepStatus::Pending;
  const std::string incomplete =
      runtimeStatusJson(memoryPlan, engine, metal, warmup, audit(memoryPlan),
                        metrics, executorTelemetry, identity, governor, true);
  require(incomplete.find("\"ready\":false") != std::string::npos,
          "missing native B3 warmup did not fail readiness");
}

void testCurrentReadinessAndSimultaneousPeak() {
  const EngineMemoryPlan memoryPlan = plan();
  require(memoryPlan.breakdown().hardBudgetBytes == 23 * kGiB,
          "readiness regression requires a 23 GiB budget");
  WarmupReport warmup;
  warmup.maximumPrefill = WarmupStepStatus::Complete;
  warmup.decodeBatches.fill(WarmupStepStatus::Complete);
  warmup.draftVerifyCommit = WarmupStepStatus::Complete;
  warmup.compositeStateRestore = WarmupStepStatus::Complete;
  warmup.memoryBudgetValidated = true;
  MemoryGovernorSnapshot governor;
  governor.hostMeasurementValid = true;
  governor.hostAvailableBytes = 8 * kGiB;
  governor.hostReserveBytes = 2 * kGiB;
  metal::MetalMemoryStats memory;
  // Dense usage previously reached 20 GiB with 2 GiB of KV. It then shrank
  // to 18 GiB while KV grew to 4 GiB: the simultaneous peak stayed 22 GiB.
  memory.allocatedBytes = 18 * kGiB;
  memory.peakAllocatedBytes = 20 * kGiB;
  memory.sparseResidentBytes = 4 * kGiB;
  memory.peakSparseResidentBytes = 4 * kGiB;
  memory.peakResidentBytes = 22 * kGiB;
  memory.deviceCurrentAllocatedBytes = 22 * kGiB;
  memory.devicePeakAllocatedBytes = 22 * kGiB;
  auto status = [&] {
    return runtimeStatusJson(memoryPlan, {}, memory, warmup, audit(memoryPlan),
                             {}, {}, {}, governor, true);
  };
  const std::string healthy = status();
  require(healthy.find("\"ready\":true") != std::string::npos &&
              healthy.find("\"peak_bytes\":" + std::to_string(22 * kGiB)) !=
                  std::string::npos,
          "disjoint dense/sparse peaks falsely exceeded the budget");

  memory.allocatedBytes = 20 * kGiB;
  memory.peakResidentBytes = 24 * kGiB;
  memory.deviceCurrentAllocatedBytes = 24 * kGiB;
  memory.devicePeakAllocatedBytes = 24 * kGiB;
  require(status().find("\"ready\":false") != std::string::npos,
          "current over-budget allocation was marked ready");
  memory.allocatedBytes = 18 * kGiB;
  memory.deviceCurrentAllocatedBytes = 22 * kGiB;
  const std::string recovered = status();
  require(recovered.find("\"ready\":true") != std::string::npos &&
              recovered.find("\"peak_bytes\":" + std::to_string(24 * kGiB)) !=
                  std::string::npos,
          "historical overage permanently poisoned recovered readiness");

  // Reaching the host reserve is a warning: growth pauses and cache is shed
  // while requests keep running, so the server stays ready. Only the
  // governor's critical verdict marks it not ready.
  governor.hostAvailableBytes = 2 * kGiB;
  governor.pressure = MemoryPressure::Warning;
  governor.growthAllowed = false;
  require(status().find("\"ready\":true") != std::string::npos,
          "reaching the host reserve under warning marked the server not ready");
  governor.hostAvailableBytes = 1 * kGiB;
  governor.pressure = MemoryPressure::Critical;
  require(status().find("\"ready\":false") != std::string::npos,
          "critical memory pressure was marked ready");
  governor.hostAvailableBytes = 8 * kGiB;
  governor.pressure = MemoryPressure::Normal;
  governor.growthAllowed = true;
}

void testWarmupStatesPreserveReadinessAndMeasurementTruth() {
  const EngineMemoryPlan memoryPlan = plan();
  WarmupReport warmup;
  require(!warmup.ready(), "unexecuted warmup was ready");
  warmup.maximumPrefill = WarmupStepStatus::Complete;
  warmup.decodeBatches.fill(WarmupStepStatus::Complete);
  warmup.draftVerifyCommit = WarmupStepStatus::Complete;
  warmup.compositeStateRestore = WarmupStepStatus::Complete;
  warmup.memoryBudgetValidated = true;
  require(warmup.ready(), "complete warmup was not ready");

  for (WarmupStepStatus *required : {&warmup.maximumPrefill,
                                    &warmup.decodeBatches[0],
                                    &warmup.draftVerifyCommit}) {
    for (WarmupStepStatus state : {WarmupStepStatus::Pending,
                                  WarmupStepStatus::MemoryLimited}) {
      *required = state;
      require(!warmup.ready(), "required execution was skipped for readiness");
    }
    *required = WarmupStepStatus::Complete;
  }

  MemoryGovernorSnapshot governor;
  governor.hostMeasurementValid = true;
  governor.hostAvailableBytes = 8 * kGiB;
  governor.hostReserveBytes = 2 * kGiB;
  auto status = [&] {
    return runtimeStatusJson(memoryPlan, {}, {}, warmup, audit(memoryPlan),
                             {}, {}, {}, governor, true);
  };
  require(status().find("\"memory_limited_steps\":[]") != std::string::npos,
          "fully measured warmup listed a memory-limited step");

  struct OptionalStep {
    WarmupStepStatus *state;
    const char *name;
  };
  const OptionalStep optional[] = {
      {&warmup.decodeBatches[1], "decode_b2"},
      {&warmup.decodeBatches[2], "decode_b3"},
      {&warmup.decodeBatches[3], "decode_b4"},
      {&warmup.compositeStateRestore, "composite_state_restore"},
  };
  for (const auto &step : optional) {
    const std::string key = std::string("\"") + step.name + "\":";
    *step.state = WarmupStepStatus::Pending;
    const std::string pending = status();
    require(!warmup.ready() &&
                pending.find("\"ready\":false") != std::string::npos &&
                pending.find(key + "false") != std::string::npos &&
                pending.find("\"memory_limited_steps\":[]") != std::string::npos,
            "unexecuted optional warmup was treated as memory-limited");

    *step.state = WarmupStepStatus::MemoryLimited;
    const std::string limited = status();
    require(warmup.ready() &&
                limited.find("\"ready\":true") != std::string::npos &&
                limited.find(key + "false") != std::string::npos &&
                limited.find(std::string("\"memory_limited_steps\":[\"") +
                             step.name + "\"]") != std::string::npos,
            "memory-limited warmup did not preserve readiness and measurement truth");

    *step.state = WarmupStepStatus::Complete;
    require(status().find(key + "true") != std::string::npos,
            "completed optional warmup was not reported as measured");
  }
  for (const auto &step : optional)
    *step.state = WarmupStepStatus::MemoryLimited;
  const std::string limited = status();
  require(warmup.ready() &&
              limited.find("\"memory_limited_steps\":[\"decode_b2\",\"decode_b3\","
                           "\"decode_b4\",\"composite_state_restore\"]") !=
                  std::string::npos &&
              limited.find("\"decode_b1\":true") != std::string::npos,
          "single-lane readiness omitted or mislabeled memory-limited steps");
  warmup.memoryBudgetValidated = false;
  require(!warmup.ready(), "memory-limited warmup bypassed the memory audit");
  warmup.memoryBudgetValidated = true;
  warmup.error = "injected failure";
  require(!warmup.ready(), "memory-limited warmup ignored an execution error");
}

void testMemoryPressureTelemetry() {
  const EngineMemoryPlan memoryPlan = plan();
  MemoryGovernorSnapshot governor;
  governor.limitBytes = 10 * kGiB;
  governor.headroomBytes = 5 * kGiB;
  governor.hostMeasurementValid = true;
  governor.hostAvailableBytes = 2 * kGiB;
  governor.hostReserveBytes = 2 * kGiB;
  governor.pressure = MemoryPressure::Critical;
  governor.growthAllowed = false;
  auto status = [&] {
    return runtimeStatusJson(memoryPlan, {}, {}, {}, {}, {}, {}, {}, governor, true);
  };
  const std::string hostLimited = status();
  require(hostLimited.find("\"memory_pressure\":\"critical\"") !=
                  std::string::npos &&
              hostLimited.find("\"system_pressure\":\"normal\"") !=
                  std::string::npos &&
              hostLimited.find("\"headroom_bytes\":" + std::to_string(5 * kGiB)) !=
                  std::string::npos &&
              hostLimited.find("\"host_headroom_bytes\":0") != std::string::npos &&
              hostLimited.find("\"growth_allowed\":false") != std::string::npos,
          "host shortage was hidden by unused engine budget or OS Normal");
  governor.systemPressure = MemoryPressure::Warning;
  governor.pressure = MemoryPressure::Warning;
  require(status().find("\"system_pressure\":\"warning\"") != std::string::npos,
          "OS pressure was not exposed independently");
  governor.hostMeasurementValid = false;
  governor.hostAvailableBytes = 0;
  governor.pressure = MemoryPressure::Critical;
  require(status().find("\"host_measurement_valid\":false") != std::string::npos,
          "missing host measurement was reported as a valid zero");
}

void testResourceWaitDiagnostics() {
  ResourceWaitSnapshot wait{.memory = 2, .concurrency = 1, .suspended = 1,
                            .oldestWaitMilliseconds = 1250.0, .draining = true};
  const auto memoryPlan = plan();
  const std::string json = runtimeStatusJson(
      memoryPlan, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, wait);
  require(json.find("\"admission\":{\"waiting\":3,\"waiting_memory\":2,"
                    "\"waiting_concurrency\":1,\"suspended\":1,\"draining\":true,"
                    "\"oldest_wait_ms\":1250}") != std::string::npos,
          "resource wait summary is missing or inaccurate");
  MemoryStatusReporter reporter;
  require(reporter.update({}, true).empty(), "healthy idle engine logged pressure");
  require(!reporter.update(wait, false).empty(), "pressure transition was silent");
  ++wait.memory;
  wait.oldestWaitMilliseconds += 1000;
  require(reporter.update(wait, false).empty(), "pressure retries flooded the log");
  wait = {};
  wait.concurrency = 4;
  require(!reporter.update(wait, true).empty(), "end of memory wait was silent");
  require(reporter.update(wait, true).empty(), "concurrency queue logged pressure");
}

} // namespace

int main() {
  try {
    testCleanRuntimeStatus();
    testCurrentReadinessAndSimultaneousPeak();
    testWarmupStatesPreserveReadinessAndMeasurementTruth();
    testMemoryPressureTelemetry();
    testResourceWaitDiagnostics();
    std::cout << "runtime status tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "runtime status tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
