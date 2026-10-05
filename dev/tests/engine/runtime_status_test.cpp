#include "TestChecks.hpp"
#include "TestModel.hpp"
#include "TestStderr.hpp"
#include "engine/RuntimeResources.hpp"
#include "StderrLine.hpp"
#include "engine/Status.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::capturedStderr;
using splash::test::require;

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
  return test::requireMemoryPlan(
      device, test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB));
}

MemoryAuditResult audit(const EngineMemoryPlan &memoryPlan) {
  const auto &b = memoryPlan.breakdown();
  ActualMemoryReport actual;
  actual.targetWeightsBytes = b.targetWeightsBytes;
  actual.draftWeightsBytes = b.draftWeightsBytes;
  actual.visionWeightsBytes = b.visionWeightsBytes;
  actual.stateAllocatedBytes = b.laneStateBytes;
  actual.sharedPrefillBytes = b.sharedPrefillBytes;
  actual.sharedDecodeBytes = b.sharedDecodeBytes;
  actual.kvAllocatedBytes = b.kvExtentBytes;
  actual.backendAllocatedBytes =
      actual.targetWeightsBytes + actual.draftWeightsBytes +
      actual.visionWeightsBytes + actual.stateAllocatedBytes +
      actual.sharedPrefillBytes + actual.sharedDecodeBytes +
      actual.kvAllocatedBytes;
  actual.deviceCurrentAllocatedBytes = actual.backendAllocatedBytes;
  actual.devicePeakAllocatedBytes = actual.backendAllocatedBytes;
  actual.backendPeakAllocatedBytes = actual.backendAllocatedBytes;
  return auditActualMemory(memoryPlan, actual);
}

// A representative status document, every section of it set, and the golden
// copy of the whole document that the server's reading of it is tested on
// (dev/tests/engine/test_status_contract.py).
void testCleanRuntimeStatus(const char *goldenPath) {
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
  engine.resourceSuspensions = 5;
  engine.prioritySuspensions = 2;
  engine.deduplicatedStatePublications = 2;
  engine.recycledStatePublications = 1;
  engine.scheduler.waitingPrefix = 3;
  engine.diskStatePublications = 4;
  engine.scheduler.prefillBatches = 4;
  engine.scheduler.prefillRows = 4096;
  engine.scheduler.decodeBatches = 4;
  engine.scheduler.decodeBatchesByWidth = {1, 1, 1, 1};
  engine.resources.pool = {128, 72, 24, 32, 128 * 4096ULL,
                           32 * 4096ULL, 5, 3, 2.5, 0.75, 2, 9};
  engine.resources.extentCompactMaxMilliseconds = 1.25;
  engine.resources.stateCache = {2, 0, 128, 1, 1, 2, 0};
  engine.resources.stateCache.checkpointEntries = 1;
  engine.resources.stateCache.checkpointBytes = 64;
  engine.resources.stateCache.checkpointEvictions = 4;
  engine.resources.stateCache.checkpointRetirements = 3;
  engine.resources.stateCache.inUse = 2;
  engine.resources.stateCache.inUseEvictions = 5;
  engine.resources.lookup = {.probeHashedBlocks = 7,
                             .kvHitTokens = 128,
                             .stateDiskHits = 1,
                             .lazyJunctions = 1};
  engine.resources.activeRequests = 1;
  engine.resources.kvTier.restores = 3;
  engine.resources.kvTier.readBytes = 12345;
  engine.resources.kvTier.writtenBytes = 67890;
  engine.resources.kvTier.fileBytes = 24680;
  engine.resources.persistent = true;
  engine.resources.kvTier.copies = 64;
  engine.resources.adoption = {2, 96, 4096, 1};
  engine.writeBehind = {1, 5, 3, 0};

  WarmupReport warmup;
  warmup.maximumPrefill = WarmupStepStatus::Complete;
  warmup.decodeBatches.fill(WarmupStepStatus::Complete);
  warmup.compositeStateRestore = WarmupStepStatus::Complete;
  warmup.maximumPrefillDetail = "packed_rows=2048";

  RuntimeMetricsSnapshot metrics;
  metrics.prefillInputTokens = 4096;
  metrics.decodeOutputTokens = 32;
  metrics.decodeWallMilliseconds = 10.0;
  metrics.decodeCycleMilliseconds = 12.5;
  metrics.draftedTokens = 28;
  metrics.acceptedDraftTokens = 20;
  metrics.draftAcceptanceRate = 20.0 / 28.0;
  metrics.currentDecodeBatch = {true, 3, 0, 4, 21, 15, 1.5, 2666.0};

  engine::RuntimeCacheIdentity identity;
  identity.modelLayoutSha256 = std::string(64, 'a');
  identity.buildId = "build";
  identity.kvLayout = kv::Layout{16, 4, 256};
  identity.targetModelSha256 = std::string(64, 'b');

  metal::MetalMemoryStats metal;
  metal.allocatedBytes = 4 * kGiB;
  metal.peakAllocatedBytes = metal.allocatedBytes;
  metal.deviceCurrentAllocatedBytes = metal.allocatedBytes;
  metal.devicePeakAllocatedBytes = metal.allocatedBytes;

  MemoryGovernorSnapshot governor;
  governor.limitBytes = memoryPlan.breakdown().hardBudgetBytes;
  governor.chargedBytes = metal.allocatedBytes;
  governor.headroomBytes = governor.limitBytes - governor.chargedBytes;
  governor.hostMeasurementValid = true;
  governor.hostAvailableBytes = 8 * kGiB;
  governor.hostReserveBytes = 2 * kGiB;
  governor.hostHeadroomBytes = 6 * kGiB;
  governor.hostGrowthAllowed = true;

  model::ModelTelemetry executorTelemetry;
  executorTelemetry.stateAllocatedBytes = 350'224'384;
  executorTelemetry.idleGdnCells = 1;
  executorTelemetry.idleDraftRings = 2;
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
  executorTelemetry.visionArenaBytes = 5;
  executorTelemetry.embeddingCacheBytes = 6;
  executorTelemetry.stateHeldImageBytes = 7;
  executorTelemetry.imageRowsBytes = 8;
  const ResourceWaitSnapshot wait{.memory = 2, .concurrency = 1, .heldBehindRefusal = 4,
                                  .restoring = 1, .suspended = 1,
                                  .oldestWaitMilliseconds = 1250.0, .draining = true};
  const std::string json = runtimeStatusJson(
      memoryPlan, engine, metal, warmup, audit(memoryPlan), metrics, executorTelemetry,
      identity, governor, true, {}, wait, NativeLoopTiming{1843.25}, {600.0, false, 2});
  std::ifstream golden(goldenPath);
  const std::string expected{std::istreambuf_iterator<char>(golden), {}};
  require(golden && json + '\n' == expected,
          std::string(goldenPath) + " is not the status document, which is now:\n" + json);
  require(json.find("\"kv_disk_hit_tokens\":96") != std::string::npos &&
              json.find("\"kv_restores\":3") != std::string::npos,
          "disk token accounting must include transfers completed before admission retries");
  require(json.find("\"read_bytes\":12345") != std::string::npos &&
              json.find("\"written_bytes\":67890") != std::string::npos &&
              json.find("\"file_bytes\":24680,") != std::string::npos,
          "disk byte accounting was not exposed");
  require(json.find("\"persistent\":true,\"kv_copies\":64,\"kv_copy_failures\":0,"
                    "\"taken_back\":{\"states\":2,\"kv_blocks\":96,\"bytes\":4096,"
                    "\"left_behind\":1},\"write_behind\":{\"waiting\":1,\"durable\":5,"
                    "\"unneeded\":3,\"refused\":0}}") != std::string::npos,
          "status lost the persistent tier");
  require(json.find("\"state_staging_bytes\":0,\"fixed_runtime_bytes\"") !=
              std::string::npos,
          "the memory plan status omitted the disk tier's state staging");
  const std::string kvIdentity =
      "{\"target_model_sha256\":\"" + std::string(64, 'b') +
      "\",\"format\":\"int8\",\"quantization\":\"symmetric_int8\","
      "\"scale_type\":\"float32\",\"key_layout\":\"token_major\","
      "\"value_layout\":\"dimension_major\"}";
  require(json.find("\"kv\":" + kvIdentity) != std::string::npos,
          "INT8 status lost its KV identity");
  require(json.find("\"cache\":{\"loaded_model_layout_sha256\":\"" + std::string(64, 'a') +
                    "\",\"build_id\":\"build\",\"dtype\":\"q8s8_f32_scale_per_token_"
                    "head_k_token_major_v_dimension_major\",\"block_tokens\":32}") !=
                  std::string::npos &&
              json.find("runtime_cache_namespace") == std::string::npos,
          "status lost the cache identity or reported a cache namespace");
  auto bf16Identity = identity;
  bf16Identity.kvLayout = kv::Layout{16, 4, 256, kv::Format::BFloat16};
  const auto bf16Status = runtimeStatusJson(memoryPlan, engine, metal, warmup, audit(memoryPlan),
                        metrics, executorTelemetry, bf16Identity, governor, true,
                        {}, {}, {}, {});
  require(bf16Status.find("\"format\":\"bf16\"") != std::string::npos &&
              bf16Status.find("\"scale_type\":\"none\"") != std::string::npos,
          "BF16 cache identity advertised INT8 storage");
  require(json.find("\"schema_version\":6") != std::string::npos &&
              json.find("\"ready\":true") != std::string::npos,
          "status readiness/schema is wrong");
  require(json.find("\"ready\":true,\"maximum_context_tokens\":102400,") !=
              std::string::npos,
          "status advertised model capacity instead of the active engine limit");
  require(json.find("\"images\":{\"encodes\":3,\"embedding_reuses\":4,"
                    "\"arena_bytes\":5,\"cached_bytes\":6,\"state_held_bytes\":7,"
                    "\"rows_bytes\":8}") !=
              std::string::npos,
          "image telemetry is missing from status");
  require(json.find("\"decode_wall_ms\":10,\"decode_cycle_ms\":12.5,") !=
              std::string::npos,
          "status lost the decode command wall or the engine's decode cycle");
  require(json.find("\"model_timing\":{\"scope\":\"model_lifetime\","
                    "\"prefill\":{\"last_gpu_ms\":5250,\"last_wall_ms\":116921.479,"
                    "\"total_gpu_ms\":500500,\"total_wall_ms\":700250},"
                    "\"decode\":{\"last_gpu_ms\":72.5,\"last_wall_ms\":83,"
                    "\"total_gpu_ms\":1125,\"total_wall_ms\":1500}}") !=
              std::string::npos,
          "status lost model GPU/wall timing, scope, or milliseconds units");

  const std::string unmeasured =
      runtimeStatusJson(memoryPlan, engine, metal, warmup, audit(memoryPlan),
                        metrics, {}, identity, governor, true, {}, {}, {}, {});
  require(unmeasured.find("\"model_timing\":{\"scope\":\"model_lifetime\","
                          "\"prefill\":{\"last_gpu_ms\":0,\"last_wall_ms\":0,"
                          "\"total_gpu_ms\":0,\"total_wall_ms\":0},"
                          "\"decode\":{\"last_gpu_ms\":0,\"last_wall_ms\":0,"
                          "\"total_gpu_ms\":0,\"total_wall_ms\":0}}") !=
              std::string::npos,
          "status invented model timings from request metrics");
  require(json.find("\"kv\":{\"block_tokens\":32,"
                    "\"pages_allocated\":128,\"pages_active\":24,"
                    "\"pages_cache\":32,\"pages_free\":72,") !=
              std::string::npos,
          "status lost the KV pool's page counts");
  require(json.find("\"extent_allocations\":5,\"extent_releases\":3,"
                    "\"extent_allocate_max_ms\":2.5,\"extent_release_max_ms\":0.75,"
                    "\"extent_compactions\":2,\"pages_moved\":9,"
                    "\"extent_compact_max_ms\":1.25}") !=
              std::string::npos,
          "status lost the KV extent growth and release diagnostics");
  require(json.find("\"system_pressure\":\"normal\"") != std::string::npos &&
              json.find("\"host_measurement_valid\":true") != std::string::npos &&
              json.find("\"host_headroom_bytes\":" + std::to_string(6 * kGiB)) !=
                  std::string::npos,
          "status omitted the host-side growth constraints");
  require(json.find("\"serving_footprint_bytes\"") == std::string::npos &&
              json.find("\"reserved_bytes\"") == std::string::npos,
          "status reported a governor field nothing reads");
  require(json.find("\"allocated_bytes\":350224384") != std::string::npos &&
              json.find("\"idle_gdn_cells\":1,\"idle_draft_rings\":2,") !=
                  std::string::npos &&
              json.find("\"active_lanes\":") != std::string::npos &&
              json.find("\"cell_ceiling\"") == std::string::npos &&
              json.find("\"scope\":\"startup_warmup\"") != std::string::npos,
          "live state memory or audit scope is missing from status");
  require(json.find("\"checkpoint_entries\":1,\"checkpoint_bytes\":64,"
                    "\"checkpoint_evictions\":4,\"checkpoint_retirements\":3") !=
              std::string::npos,
          "temporary state occupancy and retirement are missing from status");
  require(json.find("\"pinned\":0,\"in_use\":2,\"in_use_evictions\":5,") !=
              std::string::npos,
          "states unfinished requests use are missing from status");
  require(json.find("\"cache\":{\"probe_hashed_blocks\":7,\"hits\":1,\"cold_misses\":2,") !=
              std::string::npos &&
              json.find("\"disk_hits\":1,") != std::string::npos &&
              json.find("\"misses\"") == std::string::npos &&
              json.find("\"lookups\"") == std::string::npos &&
              json.find("\"state_hit_tokens\"") == std::string::npos &&
              json.find("\"deduplicated_publications\"") == std::string::npos,
          "status reported a lookup counter twice");
  require(json.find("\"block_tokens\":32") != std::string::npos &&
              json.find("\"decode_batches_by_width\":{\"b1\":1,\"b2\":1,\"b3\":"
                        "1,\"b4\":1}}") != std::string::npos,
          "Page32 or real B3 status is missing");
  require(
      json.find("\"dynamic_budget_bytes\"") != std::string::npos &&
          json.find("\"resource_replay_tokens\":1234") != std::string::npos &&
          json.find("\"resource_suspensions\":5,\"priority_suspensions\":2,") !=
              std::string::npos &&
          json.find("\"waiting_prefix\":3") != std::string::npos &&
          json.find("\"deduplicated_state_publications\":2,"
                    "\"recycled_state_publications\":1,"
                    "\"disk_state_publications\":4,") !=
              std::string::npos &&
          json.find("\"lazy_junctions\":1") != std::string::npos &&
          json.find("\"checkpoint_publications\":3,"
                    "\"checkpoint_publication_failures\":1") !=
              std::string::npos &&
          json.find("\"draft_context\":{\"target_prefill_rows\":10000,\"prompt_"
                    "end_rows\":2048,\"materialization_rows\":31,\"avoided_rows\":"
                    "7921,\"restore_skipped\":1,\"resets\":2}") !=
              std::string::npos &&
          json.find("\"constraint_masks\":{\"overlap_batches\":5,\"overlap_"
                    "requests\":8,\"last_target_forward_gpu_ms\":72.5,"
                    "\"total_target_forward_gpu_ms\":250,\"last_residual_"
                    "wait_ms\":1.5,\"total_residual_wait_ms\":12}") !=
              std::string::npos,
      "status lacks the engine's memory, cache or constraint counters");
}

void testCurrentReadinessAndSimultaneousPeak() {
  const EngineMemoryPlan memoryPlan = plan();
  require(memoryPlan.breakdown().hardBudgetBytes == 23 * kGiB,
          "readiness regression requires a 23 GiB budget");
  WarmupReport warmup;
  warmup.maximumPrefill = WarmupStepStatus::Complete;
  warmup.decodeBatches.fill(WarmupStepStatus::Complete);
  warmup.compositeStateRestore = WarmupStepStatus::Complete;
  MemoryGovernorSnapshot governor;
  governor.hostMeasurementValid = true;
  governor.hostAvailableBytes = 8 * kGiB;
  governor.hostReserveBytes = 2 * kGiB;
  metal::MetalMemoryStats memory;
  // KV extents are ordinary allocations: their bytes are part of the
  // backend's current and peak bytes.
  memory.allocatedBytes = 22 * kGiB;
  memory.peakAllocatedBytes = 22 * kGiB;
  memory.deviceCurrentAllocatedBytes = 22 * kGiB;
  memory.devicePeakAllocatedBytes = 22 * kGiB;
  auto status = [&] {
    return runtimeStatusJson(memoryPlan, {}, memory, warmup, audit(memoryPlan),
                             {}, {}, {}, governor, true, {}, {}, {}, {});
  };
  const std::string healthy = status();
  require(healthy.find("\"ready\":true") != std::string::npos &&
              healthy.find("\"peak_bytes\":" + std::to_string(22 * kGiB)) !=
                  std::string::npos,
          "allocations within the budget were not ready or lost their peak");

  memory.allocatedBytes = 24 * kGiB;
  memory.peakAllocatedBytes = 24 * kGiB;
  memory.deviceCurrentAllocatedBytes = 24 * kGiB;
  memory.devicePeakAllocatedBytes = 24 * kGiB;
  require(status().find("\"ready\":false") != std::string::npos,
          "current over-budget allocation was marked ready");
  memory.allocatedBytes = 22 * kGiB;
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
  governor.hostGrowthAllowed = false;
  require(status().find("\"ready\":true") != std::string::npos,
          "reaching the host reserve under warning marked the server not ready");
  governor.hostAvailableBytes = 1 * kGiB;
  governor.pressure = MemoryPressure::Critical;
  require(status().find("\"ready\":false") != std::string::npos,
          "critical memory pressure was marked ready");
  governor.hostAvailableBytes = 8 * kGiB;
  governor.pressure = MemoryPressure::Normal;
  governor.hostGrowthAllowed = true;
}

void testWarmupStepsReportMeasurementTruth() {
  const EngineMemoryPlan memoryPlan = plan();
  WarmupReport warmup;
  warmup.maximumPrefill = WarmupStepStatus::Complete;
  warmup.decodeBatches.fill(WarmupStepStatus::Complete);
  warmup.compositeStateRestore = WarmupStepStatus::Complete;
  MemoryGovernorSnapshot governor;
  governor.hostMeasurementValid = true;
  governor.hostAvailableBytes = 8 * kGiB;
  governor.hostReserveBytes = 2 * kGiB;
  auto status = [&] {
    return runtimeStatusJson(memoryPlan, {}, {}, warmup, audit(memoryPlan),
                             {}, {}, {}, governor, true, {}, {}, {}, {});
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
    require(pending.find(key + "false") != std::string::npos &&
                pending.find("\"memory_limited_steps\":[]") != std::string::npos,
            "unexecuted optional warmup was treated as memory-limited");

    *step.state = WarmupStepStatus::MemoryLimited;
    const std::string limited = status();
    require(limited.find(key + "false") != std::string::npos &&
                limited.find(std::string("\"memory_limited_steps\":[\"") +
                             step.name + "\"]") != std::string::npos,
            "memory-limited warmup was reported as measured or not listed");

    *step.state = WarmupStepStatus::Complete;
    require(status().find(key + "true") != std::string::npos,
            "completed optional warmup was not reported as measured");
  }
  for (const auto &step : optional)
    *step.state = WarmupStepStatus::MemoryLimited;
  const std::string limited = status();
  require(limited.find("\"memory_limited_steps\":[\"decode_b2\",\"decode_b3\","
                       "\"decode_b4\",\"composite_state_restore\"]") !=
                  std::string::npos &&
              limited.find("\"decode_b1\":true") != std::string::npos,
          "single-lane status omitted or mislabeled memory-limited steps");
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
  governor.hostGrowthAllowed = false;
  auto status = [&] {
    return runtimeStatusJson(memoryPlan, {}, {}, {}, {}, {}, {}, {}, governor, true, {},
                             {}, {}, {});
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
  ResourceWaitSnapshot wait{.memory = 2, .concurrency = 1, .heldBehindRefusal = 4,
                            .restoring = 1, .suspended = 1,
                            .oldestWaitMilliseconds = 1250.0, .draining = true};
  const auto memoryPlan = plan();
  const std::string json = runtimeStatusJson(
      memoryPlan, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, wait, {}, {});
  require(json.find("\"admission\":{\"waiting\":3,\"waiting_memory\":2,"
                    "\"waiting_concurrency\":1,\"held_behind_refusal\":4,\"restoring\":1,"
                    "\"suspended\":1,\"draining\":true,"
                    "\"oldest_wait_ms\":1250},\"loop\":{\"max_tick_ms\":0}") !=
              std::string::npos,
          "resource wait summary is missing or inaccurate");
  const std::string ticked = runtimeStatusJson(
      memoryPlan, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, wait,
      NativeLoopTiming{1843.25}, {});
  require(ticked.find("\"loop\":{\"max_tick_ms\":1843.25}") != std::string::npos &&
              ticked.find("\"schema_version\":6") != std::string::npos,
          "the loop's longest tick is missing, or changed the status schema");
}

// The weights' idle release, null with --idle-release off, whether they are
// released and how often they were written back.
void testWeightsStatus() {
  const auto memoryPlan = plan();
  const auto status = [&](WeightsSnapshot weights) {
    return runtimeStatusJson(memoryPlan, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, {}, {},
                             weights);
  };
  require(status({600.0, true, 2})
                  .find("\"weights\":{\"idle_release_seconds\":600,\"released\":true,"
                        "\"restores\":2}") != std::string::npos,
          "the weights' status is missing or inaccurate");
  require(status({1234567.5, false, 0}).find("\"idle_release_seconds\":1234567.5,") !=
              std::string::npos,
          "the idle release lost precision");
  require(status({std::numeric_limits<double>::infinity(), false, 0})
                  .find("\"weights\":{\"idle_release_seconds\":null,\"released\":false,"
                        "\"restores\":0}") != std::string::npos,
          "an idle release that is off is not null");
}

// The server and the runtime share stderr, as `serve > log 2>&1` does: a
// line written from any thread arrives whole.
void testStderrLinesStayWhole() {
  std::istringstream lines(capturedStderr([] {
    std::vector<std::thread> writers;
    for (int writer = 0; writer < 8; ++writer)
      writers.emplace_back([writer] {
        for (int line = 0; line < 300; ++line)
          writeStderrLine("writer " + std::to_string(writer) + " line " +
                          std::to_string(line));
      });
    for (std::thread &writer : writers)
      writer.join();
  }));
  const std::regex whole("writer [0-7] line [0-9]+");
  int count = 0;
  for (std::string line; std::getline(lines, line); ++count)
    require(std::regex_match(line, whole), "a stderr line was broken");
  require(count == 8 * 300, "stderr lines were lost or merged");
}

// A notice carries the time first, as the server's console lines do, and
// stays on its line: an exception message's newline becomes a space.
void testNoticesCarryTheTime() {
  const std::string notice = capturedStderr([] {
    logLine("Weights restored in ", std::fixed, std::setprecision(2), 1.5,
            " s (first line\nsecond line)");
  });
  require(std::regex_match(notice,
                           std::regex("[0-9]{2}:[0-9]{2}:[0-9]{2} Weights restored in "
                                      "1\\.50 s \\(first line second line\\)\n")),
          "a notice did not carry the time on one line: " + notice);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " status_golden.json\n";
    return 2;
  }
  try {
    testCleanRuntimeStatus(argv[1]);
    testCurrentReadinessAndSimultaneousPeak();
    testWarmupStepsReportMeasurementTruth();
    testMemoryPressureTelemetry();
    testResourceWaitDiagnostics();
    testWeightsStatus();
    testStderrLinesStayWhole();
    testNoticesCarryTheTime();
    std::cout << "runtime status tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "runtime status tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
