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
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::capturedStderr;
using splash::test::require;

// The 27B's plan, or with another KV layout.
EngineMemoryPlan plan(const kv::Layout &kvLayout = kv::Layout{16, 4, 256}) {
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
  ModelMemoryProfile profile = test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB);
  profile.targetKvLayout = kvLayout;
  return test::requireMemoryPlan(device, profile);
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
// (dev/tests/server/test_status_contract.py).
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
  engine.deduplicatedKvTokens = 96;
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
  engine.stateCaptures = 6;
  engine.captureCuts = 1;
  engine.scheduler.prefillBatches = 4;
  engine.scheduler.prefillRows = 4096;
  engine.scheduler.decodeBatches = 4;
  engine.scheduler.decodeBatchesByWidth = {1, 1, 1, 1};
  engine.activeRequests = {{.id = 7,
                            .phase = engine::Phase::Prefill,
                            .priority = engine::RequestPriority::Foreground,
                            .promptTokens = 4096,
                            .promptProcessed = 2048,
                            .maxNewTokens = 256,
                            .ageMilliseconds = 1500.5},
                           {.id = 9,
                            .phase = engine::Phase::Decode,
                            .promptTokens = 512,
                            .promptProcessed = 512,
                            .generatedTokens = 40,
                            .maxNewTokens = 128,
                            .ageMilliseconds = 2250.25}};
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
  engine.resources.kvTier.capacityBytes = 34359738368;
  engine.resources.kvTier.limitBytes = 8589934592;
  engine.resources.kvTier.usedBytes = 4096;
  engine.resources.kvTier.volumeAvailableBytes = 46170898432;
  engine.resources.kvTier.writesGated = engine::DiskWritesGate::Budget;
  engine.resources.kvTier.writesDeferredMicroseconds = 2'500;
  engine.resources.persistent = true;
  engine.resources.kvTier.copies = 64;
  engine.resources.adoption = {2, 96, 4096, 1};
  engine.writeBehind = {1, 5, 3, 0};

  WarmupReport warmup;
  warmup.maximumPrefill = WarmupStepStatus::Complete;
  warmup.decodeBatches.fill(WarmupStepStatus::Complete);
  warmup.compositeStateRestore = WarmupStepStatus::Complete;
  warmup.maximumPrefillDetail = "rows=2048";

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
  governor.hostSpareBytes = 8 * kGiB;
  governor.hostSpareReserveBytes = 2 * kGiB;
  governor.hostHeadroomBytes = 6 * kGiB;
  governor.hostGrowthAllowed = true;

  model::ModelTelemetry executorTelemetry;
  executorTelemetry.stateAllocatedBytes = 350'224'384;
  executorTelemetry.idleGdnCells = 1;
  executorTelemetry.idleDraftRings = 2;
  executorTelemetry.idleContextWindows = 3;
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
  executorTelemetry.imageEmbeddingEvictions = 9;
  executorTelemetry.visionArenaBytes = 5;
  executorTelemetry.embeddingCacheBytes = 6;
  executorTelemetry.stateHeldImageBytes = 7;
  executorTelemetry.imageRowsBytes = 8;
  executorTelemetry.aneFfnReruns = 1;
  executorTelemetry.verifyBudget = true;
  const AneFfnSnapshot aneFfn{AneFfnSnapshot::State::Stopped,
                              "an evaluation did not complete within 2 s",
                              0.235,
                              1037,
                              12,
                              768,
                              10752.5};
  const ResourceWaitSnapshot wait{.memory = 2, .concurrency = 1, .heldBehindRefusal = 4,
                                  .restoring = 1, .suspended = 1,
                                  .oldestWaitMilliseconds = 1250.0, .draining = true};
  const std::string json = runtimeStatusJson(
      memoryPlan, engine, metal, warmup, audit(memoryPlan), metrics, executorTelemetry,
      identity, governor, true, {}, wait, NativeLoopTiming{1843.25}, {600.0, false, false, 2, 1}, aneFfn,
      ThermalState::Fair);
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
  require(json.find("\"disk\":{\"capacity_bytes\":34359738368,\"limit_bytes\":8589934592,"
                    "\"used_bytes\":4096,\"file_bytes\":24680,"
                    "\"volume_available_bytes\":46170898432,\"writes_gated\":\"budget\","
                    "\"write_budget_bytes\":0,") != std::string::npos,
          "the disk tier's limit, volume and write budget were not exposed");
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
                    "head_k_token_major_v_dimension_major\",\"block_tokens\":32,\"rope_scaling\":null}") !=
                  std::string::npos &&
              json.find("runtime_cache_namespace") == std::string::npos,
          "status lost the cache identity or reported a cache namespace");
  auto bf16Identity = identity;
  bf16Identity.kvLayout = kv::Layout{16, 4, 256, kv::Format::BFloat16};
  const auto bf16Status = runtimeStatusJson(memoryPlan, engine, metal, warmup, audit(memoryPlan),
                        metrics, executorTelemetry, bf16Identity, governor, true,
                        {}, {}, {}, {}, {}, {});
  require(bf16Status.find("\"format\":\"bf16\"") != std::string::npos &&
              bf16Status.find("\"scale_type\":\"none\"") != std::string::npos,
          "BF16 cache identity advertised INT8 storage");
  // YaRN's factor, to the digits that read back as it, as a context of a
  // million tokens takes it, and its original window, which every key went
  // through; the rest of the document keeps its precision.
  auto yarnIdentity = identity;
  yarnIdentity.ropeScaling = {3.814697265625, 262144};
  std::string yarnExpected = json;
  const std::string unscaled = "\"rope_scaling\":null";
  yarnExpected.replace(yarnExpected.find(unscaled), unscaled.size(),
                       "\"rope_scaling\":{\"type\":\"yarn\",\"factor\":3.814697265625,"
                       "\"original_context_tokens\":262144}");
  require(runtimeStatusJson(memoryPlan, engine, metal, warmup, audit(memoryPlan), metrics, executorTelemetry,
                            yarnIdentity, governor, true, {}, wait, NativeLoopTiming{1843.25},
                            {600.0, false, false, 2, 1}, aneFfn, ThermalState::Fair) == yarnExpected,
          "the cache identity did not report YaRN, or the status after it lost its precision");
  auto tokenMajorIdentity = identity;
  tokenMajorIdentity.kvLayout.valuesTokenMajor = true;
  const auto tokenMajorStatus = runtimeStatusJson(memoryPlan, engine, metal, warmup, audit(memoryPlan),
                        metrics, executorTelemetry, tokenMajorIdentity, governor, true,
                        {}, {}, {}, {}, {}, {});
  require(tokenMajorStatus.find("\"dtype\":\"q8s8_f32_scale_per_token_head_k_token_major_v_token_major\"") !=
                  std::string::npos &&
              tokenMajorStatus.find("\"value_layout\":\"token_major\"") != std::string::npos,
          "token-major values were reported dimension-major");
  require(json.find("\"schema_version\":6") != std::string::npos &&
              json.find("\"ready\":true") != std::string::npos,
          "status readiness/schema is wrong");
  require(json.find("\"ready\":true,\"maximum_context_tokens\":102400,") !=
              std::string::npos,
          "status advertised model capacity instead of the active engine limit");
  require(json.find("\"images\":{\"encodes\":3,\"embedding_reuses\":4,"
                    "\"embedding_evictions\":9,\"arena_bytes\":5,\"cached_bytes\":6,"
                    "\"state_held_bytes\":7,\"rows_bytes\":8}") !=
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
                        metrics, {}, identity, governor, true, {}, {}, {}, {}, {}, {});
  require(unmeasured.find("\"model_timing\":{\"scope\":\"model_lifetime\","
                          "\"prefill\":{\"last_gpu_ms\":0,\"last_wall_ms\":0,"
                          "\"total_gpu_ms\":0,\"total_wall_ms\":0},"
                          "\"decode\":{\"last_gpu_ms\":0,\"last_wall_ms\":0,"
                          "\"total_gpu_ms\":0,\"total_wall_ms\":0}}") !=
              std::string::npos,
          "status invented model timings from request metrics");
  require(json.find("\"verify_rows\":\"auto\"") != std::string::npos &&
              unmeasured.find("\"verify_rows\":\"all\"") != std::string::npos,
          "status lost the rows decode cycles verify");
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
              json.find("\"idle_gdn_cells\":1,\"idle_draft_rings\":2,\"idle_context_windows\":3,") !=
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
                    "\"disk_state_publications\":4,"
                    "\"state_captures\":6,\"capture_cuts\":1,") !=
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
                             {}, {}, {}, governor, true, {}, {}, {}, {}, {}, {});
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
                             {}, {}, {}, governor, true, {}, {}, {}, {}, {}, {});
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
                             {}, {}, {}, {}, {});
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
      memoryPlan, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, wait, {}, {}, {}, {});
  require(json.find("\"admission\":{\"waiting\":3,\"waiting_memory\":2,"
                    "\"waiting_concurrency\":1,\"held_behind_refusal\":4,\"restoring\":1,"
                    "\"suspended\":1,\"draining\":true,"
                    "\"oldest_wait_ms\":1250},\"loop\":{\"max_tick_ms\":0}") !=
              std::string::npos,
          "resource wait summary is missing or inaccurate");
  const std::string ticked = runtimeStatusJson(
      memoryPlan, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, wait,
      NativeLoopTiming{1843.25}, {}, {}, {});
  require(ticked.find("\"loop\":{\"max_tick_ms\":1843.25}") != std::string::npos &&
              ticked.find("\"schema_version\":6") != std::string::npos,
          "the loop's longest tick is missing, or changed the status schema");
}

// The weights' idle release, null with --idle-release off, whether they are
// released and whether a restore is under way, how often they were written
// back and how often a restore gave up.
void testWeightsStatus() {
  const auto memoryPlan = plan();
  const auto status = [&](WeightsSnapshot weights) {
    return runtimeStatusJson(memoryPlan, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, {}, {},
                             weights, {}, {});
  };
  require(status({600.0, true, false, 2, 1})
                  .find("\"weights\":{\"idle_release_seconds\":600,\"released\":true,"
                        "\"restoring\":false,\"restores\":2,\"restore_failures\":1}") !=
              std::string::npos,
          "the weights' status is missing or inaccurate");
  require(status({600.0, true, true, 2, 1})
                  .find("\"released\":true,\"restoring\":true,") != std::string::npos,
          "a restore under way is missing from the weights' status");
  require(status({1234567.5, false, false, 0}).find("\"idle_release_seconds\":1234567.5,") !=
              std::string::npos,
          "the idle release lost precision");
  require(status({std::numeric_limits<double>::infinity(), false, false, 0})
                  .find("\"weights\":{\"idle_release_seconds\":null,\"released\":false,"
                        "\"restoring\":false,\"restores\":0,\"restore_failures\":0}") !=
              std::string::npos,
          "an idle release that is off is not null");
}

// Streamed routed experts' store (--expert-memory): null without one; with
// one its arena, staging, slots, residency, loads, hit rate, the
// latencies of its host steps' round trips and its failure. An added field:
// the schema version stays.
void testStreamedExpertsStatus() {
  const auto memoryPlan = plan();
  const auto status = [&](const std::optional<model::ExpertStoreStats> &experts) {
    return runtimeStatusJson(memoryPlan, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, {}, {}, {}, {}, {}, std::nullopt,
                             experts);
  };
  require(status(std::nullopt).find(",\"streamed_experts\":null,") != std::string::npos,
          "a model without streamed experts does not report null");
  model::ExpertStoreStats experts{
      .arenaBytes = 17179869184, .stagingBytes = 452984832, .slots = 8192, .experts = 25088, .resident = 8000};
  experts.uses[1] = {.loads = 480,
                     .routed = 4000,
                     .read = 1000,
                     .prefetched = 600,
                     .readAhead = 900,
                     .readBytes = 1973000000,
                     .pickup = {.count = 500, .totalMilliseconds = 11.5, .maximumMilliseconds = 0.09,
                                .p50Milliseconds = 0.032, .p95Milliseconds = 0.032, .p99Milliseconds = 0.064},
                     .work = {.count = 500, .totalMilliseconds = 240, .maximumMilliseconds = 9.5,
                              .p50Milliseconds = 0.512, .p95Milliseconds = 2.048, .p99Milliseconds = 8.192},
                     .resume = {.count = 500, .totalMilliseconds = 48.5, .maximumMilliseconds = 12.5,
                                .p50Milliseconds = 0.128, .p95Milliseconds = 0.128, .p99Milliseconds = 16.384}};
  experts.failure = "injected";
  const std::string none = "{\"p50\":0,\"p95\":0,\"p99\":0,\"max\":0,\"total\":0,\"samples\":0}";
  require(status(experts).find(
              "\"streamed_experts\":{\"arena_bytes\":17179869184,\"staging_bytes\":452984832,\"slots\":8192,"
              "\"experts\":25088,\"resident\":8000,\"prefill\":{\"loads\":0,\"routed\":0,\"read\":0,\"prefetched\":0,"
              "\"read_ahead\":0,\"hit_rate\":0,\"read_bytes\":0,\"host_steps\":{\"pickup_ms\":" +
              none + ",\"work_ms\":" + none + ",\"resume_ms\":" + none +
              "}},\"decode\":{\"loads\":480,\"routed\":4000,\"read\":1000,\"prefetched\":600,"
              "\"read_ahead\":900,\"hit_rate\":0.75,\"read_bytes\":1973000000,\"host_steps\":"
              "{\"pickup_ms\":{\"p50\":0.032,\"p95\":0.032,\"p99\":0.064,\"max\":0.09,\"total\":11.5,"
              "\"samples\":500},\"work_ms\":{\"p50\":0.512,\"p95\":2.048,\"p99\":8.192,\"max\":9.5,"
              "\"total\":240,\"samples\":500},\"resume_ms\":{\"p50\":0.128,\"p95\":0.128,\"p99\":16.384,"
              "\"max\":12.5,\"total\":48.5,\"samples\":500}}},\"failure\":\"injected\"}") != std::string::npos,
          "streamed experts' status differs");
}

// A target whose KV can spill reports what spilling does under kv_spill,
// and each active request's kv_mode; one whose KV stays in memory reports
// neither, as before (its plan's kv_spill is null). The pages that hold only their
// index are the pool's pages of that form in use, each a spilled block's,
// which a request holds as the cache does: counted once. An added section:
// the schema version stays.
void testKvSpillStatus() {
  engine::EngineSnapshot engine;
  engine.activeRequests = {{.id = 7, .phase = engine::Phase::Decode, .promptTokens = 4096, .maxNewTokens = 256},
                           {.id = 9,
                            .phase = engine::Phase::Decode,
                            .promptTokens = 1'000'000,
                            .promptProcessed = 1'000'000,
                            .maxNewTokens = 512,
                            .kvMode = engine::KvMode::Spill}};
  // That one spilled, one other did after it converted, and one was refused
  // for want of quota; another start waits for it. Its prefill streamed once
  // beside a decoder, which waited 40 ms.
  engine.spill = {.spill = 1, .resident = 3, .converted = 1, .refusedQuota = 1};
  engine.scheduler.contendedStreamingPrefillBatches = 1;
  engine.scheduler.decodeGap = {.count = 1, .totalMilliseconds = 40, .maximumMilliseconds = 40,
                                .p50Milliseconds = 40, .p95Milliseconds = 40, .p99Milliseconds = 40};
  // 31,250 of its pages hold only their index, of 24 KiB, and a free extent of
  // them is the runway.
  engine.resources.pool.index = {.pagesAllocated = 31'506,
                                 .pagesFree = 256,
                                 .pagesActive = 31'250,
                                 .pagesPrefix = 31'250,
                                 .allocatedBytes = 31'506 * 24'576ULL};
  ResourceWaitSnapshot wait;
  wait.diskQuota = 1;
  KvSpillStatus spill;
  spill.cache = {.spilledPages = 31'250,
                 .spilledBytes = 13'312'000'000,
                 .writeThroughPages = 31'250,
                 .writeThroughBytes = 13'312'000'000,
                 .writeThroughInFlight = 3,
                 .spillMoves = 31'250,
                 .indexRestores = 12,
                 .wholeRestores = 2,
                 .restore = {.count = 2, .totalMilliseconds = 1.5, .maximumMilliseconds = 1, .p50Milliseconds = 0.5,
                             .p95Milliseconds = 1, .p99Milliseconds = 1},
                 .activeLimitBytes = 32 * kGiB,
                 .heldBytes = 13'312'000'000,
                 .committedBytes = 55'000'000};
  spill.blockCache = model::KvBlockCacheStats{.bytes = 408'944'640,
                                              .targetBytes = 408'944'640,
                                              .floorBytes = 408'944'640,
                                              .slotsPerLayer = 8'192,
                                              .lanes = 1,
                                              .reads = {.resolves = 24,
                                                        .entries = 98'304,
                                                        .paged = 1'024,
                                                        .hits = 60'000,
                                                        .misses = 37'280,
                                                        .ghostHits = 512,
                                                        .readBytes = 310'000'000}};
  // Two commands read 24 layers' history in 5 s.
  spill.historyStream = model::KvHistoryStreamStats{
      .commands = 2,
      .layers = 24,
      .pages = 750'000,
      .bytes = 24'960'000'000,
      .stagingBytes = 1'056'440'320,
      .buffers = 2,
      .read = {.count = 24, .totalMilliseconds = 5'000, .maximumMilliseconds = 250, .p50Milliseconds = 262.144,
               .p95Milliseconds = 262.144, .p99Milliseconds = 262.144}};
  const auto status = [&](const EngineMemoryPlan &memoryPlan, const std::optional<KvSpillStatus> &kvSpill) {
    return runtimeStatusJson(memoryPlan, engine, {}, {}, {}, {}, {}, {}, {}, true, {}, wait, {}, {}, {}, {},
                             std::nullopt, std::nullopt, kvSpill);
  };
  const std::string resident = status(plan(), std::nullopt);
  const size_t planned = resident.find("\"kv_spill\":null");
  require(planned < resident.find("\"memory_actual\"") &&
              resident.find("\"kv_spill\"", planned + 1) == std::string::npos &&
              resident.find("\"kv_mode\"") == std::string::npos,
          "a target whose KV stays in memory reported spilling");
  const std::string json = status(plan(test::sparseKvLayout()), spill);
  require(json.find("\"id\":7,") < json.find("\"kv_mode\":\"resident\"") &&
              json.find("\"id\":9,") < json.find("\"kv_mode\":\"spill\""),
          "the active requests' KV modes are missing");
  const std::string none = "{\"p50\":0,\"p95\":0,\"p99\":0,\"max\":0,\"total\":0,\"samples\":0}";
  const std::string expected =
      "\"kv_spill\":{\"requests\":{\"spill\":1,\"resident\":3,\"converted\":1,\"bounded\":0,\"length_ends\":0,"
      "\"refused\":{\"quota\":1,\"write_budget\":0,\"closed\":0},\"waiting_quota\":1},\"pages\":{\"index\":31250,"
      "\"index_bytes\":768000000,\"spilled\":31250,\"spilled_bytes\":13312000000},\"write_through\":{\"pages\":31250,"
      "\"bytes\":13312000000,\"in_flight\":3,\"failures\":0,\"paused\":false},\"spill_moves\":31250,\"unspills\":0,"
      "\"rewrites\":0,\"restores\":{\"index_pages\":12,\"whole_pages\":2,\"ms\":{\"p50\":0.5,\"p95\":1,\"p99\":1,"
      "\"max\":1,\"total\":1.5,\"samples\":2}},\"quota\":{\"active_limit_bytes\":34359738368,\"held_bytes\":"
      "13312000000,\"committed_bytes\":55000000,\"revoked\":0},\"index_runway_bytes\":6291456,\"block_cache\":"
      "{\"bytes\":408944640,\"target_bytes\":408944640,\"floor_bytes\":408944640,\"slots_per_layer\":8192,\"lanes\":1,"
      "\"resolves\":24,\"entries\":98304,\"paged\":1024,\"hits\":60000,\"read\":37280,\"hit_rate\":0.6167763158,"
      "\"ghost_hits\":512,\"read_bytes\":310000000,\"read_failures\":0,\"resolve_ms\":" +
      none + ",\"read_ms\":" + none + ",\"host_steps\":{\"pickup_ms\":" + none + ",\"work_ms\":" + none +
      ",\"resume_ms\":" + none +
      "}},\"history_stream\":{\"commands\":2,\"contended_commands\":1,\"layers\":24,\"pages\":750000,"
      "\"bytes\":24960000000,\"staging_bytes\":1056440320,\"buffers\":2,\"read_ms\":{\"p50\":262.144,"
      "\"p95\":262.144,\"p99\":262.144,\"max\":250,\"total\":5000,\"samples\":24},\"wait_ms\":" +
      none + ",\"decode_gap_ms\":{\"p50\":40,\"p95\":40,\"p99\":40,\"max\":40,\"total\":40,\"samples\":1}}}";
  require(json.find(expected) != std::string::npos, "the spilling target's kv_spill differs:\n" + json);
}

// A streamed table's store (--ple-memory): null without one; with one its
// memory, per use its stages, rows, hit rate and stage latencies, its reads,
// prefetches and failure. An added field: the schema version stays.
void testStreamedTableStatus() {
  const auto memoryPlan = plan();
  const auto status = [&](const std::optional<model::SparseRowStats> &table) {
    return runtimeStatusJson(memoryPlan, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, {}, {}, {}, {}, {}, table);
  };
  require(status(std::nullopt).find(",\"streamed_table\":null,") != std::string::npos,
          "a model without a streamed table does not report null");
  model::SparseRowStats table;
  table.table = "per_layer_token_embd.weight";
  table.tableBytes = 28800138240;
  table.cacheBytes = 2147483648;
  table.cacheCapacityRows = 18000000;
  table.cachedRows = 1200;
  table.uses[1] = {10, 1280, 1000, 750, 250, {10, 12.5, 3.25, 1.024, 2.048, 4.096}};
  table.readBytes = 22500;
  table.prefetchRequestedRows = 32768;
  table.prefetchReadRows = 30000;
  table.poisoned = true;
  table.failure = "injected";
  const std::string json = status(table);
  require(json.find("\"streamed_table\":{\"table\":\"per_layer_token_embd.weight\",\"table_bytes\":28800138240,"
                    "\"cache_bytes\":2147483648,\"cache_capacity_rows\":18000000,\"cached_rows\":1200,"
                    "\"evictions\":0,\"prefill\":{\"stages\":0,") != std::string::npos &&
              json.find("\"decode\":{\"stages\":10,\"requested_rows\":1280,\"unique_rows\":1000,"
                        "\"hit_rows\":750,\"read_rows\":250,\"hit_rate\":0.75,\"stage_ms\":{\"p50\":1.024,"
                        "\"p95\":2.048,\"p99\":4.096,\"max\":3.25,\"total\":12.5,\"samples\":10}}") !=
                  std::string::npos &&
              json.find("\"read_errors\":0,\"read_ms\":{\"p50\":0,\"p95\":0,\"p99\":0,\"max\":0,\"total\":0,"
                        "\"samples\":0},\"prefetch_requested_rows\":32768,") != std::string::npos &&
              json.find("\"prefetch_read_rows\":30000,\"prefetch_dropped_rows\":0,\"poisoned\":true,"
                        "\"failure\":\"injected\"}") != std::string::npos &&
              json.find("\"schema_version\":6,") != std::string::npos,
          "the streamed table's status is missing or inaccurate");
}

// The prefill FFN's Neural Engine split: off with the start's reason and no
// share, serving with the start's, stopped with why, and what it ran beside
// the chunks the GPU ran again.
void testAneFfnStatus() {
  const auto memoryPlan = plan();
  const auto status = [&](const AneFfnSnapshot &split, uint64_t reruns) {
    model::ModelTelemetry telemetry;
    telemetry.aneFfnReruns = reruns;
    return runtimeStatusJson(memoryPlan, {}, {}, {}, {}, {}, telemetry, {}, {}, true, {}, {}, {},
                             {}, split, {});
  };
  require(status({.reason = "as given"}, 0)
                  .find("\"ane_ffn\":{\"state\":\"off\",\"share\":0,\"minimum_rows\":0,"
                        "\"reason\":\"as given\",\"split_commands\":0,\"reruns\":0,\"ane_ms\":0,"
                        "\"evaluations\":0},\"kv\":") != std::string::npos,
          "a split the start left off is missing or inaccurate");
  require(status({AneFfnSnapshot::State::Split, "at share 0.41 for chunks of 640 rows or more", 0.41, 640, 3,
                  192, 1234.5},
                 0)
                  .find("\"ane_ffn\":{\"state\":\"split\",\"share\":0.41,\"minimum_rows\":640,"
                        "\"reason\":\"at share 0.41 for chunks of 640 rows or more\",\"split_commands\":3,"
                        "\"reruns\":0,\"ane_ms\":1234.5,\"evaluations\":192}") != std::string::npos,
          "a split that serves is missing or inaccurate");
  require(status({AneFfnSnapshot::State::Stopped,
                  "losing to the GPU alone (19.4 ms on the Neural Engine per 2048-row layer against 19.0 ms "
                  "on the GPU alone)",
                  0.26, 1037, 8, 512, 9932.8},
                 0)
                  .find("\"state\":\"stopped\",\"share\":0.26,\"minimum_rows\":1037,\"reason\":\"losing "
                        "to the GPU alone (19.4 ms on the Neural Engine per 2048-row layer against 19.0 ms on the "
                        "GPU alone)\",\"split_commands\":8,\"reruns\":0,\"ane_ms\":9932.8,"
                        "\"evaluations\":512}") != std::string::npos,
          "a split the breaker stopped is missing or inaccurate");
  require(status({AneFfnSnapshot::State::Stopped, "an evaluation failed", 0.26, 1037, 2, 128, 100.0}, 1)
                  .find("\"reason\":\"an evaluation failed\",\"split_commands\":2,\"reruns\":1,") !=
              std::string::npos,
          "a split that failed lost its rerun");
}

// The thermal state beside the memory pressure, in each of its spellings; a
// hot Mac serves slower but stays ready.
void testThermalStateStatus() {
  const auto memoryPlan = plan();
  const auto status = [&](ThermalState state) {
    return runtimeStatusJson(memoryPlan, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, {}, {},
                             {}, {}, state);
  };
  for (const auto &[state, name] :
       {std::pair{ThermalState::Nominal, "nominal"}, std::pair{ThermalState::Fair, "fair"},
        std::pair{ThermalState::Serious, "serious"},
        std::pair{ThermalState::Critical, "critical"}}) {
    const std::string json = status(state);
    require(json.find("\"memory_pressure\":\"normal\",\"thermal_state\":\"" +
                      std::string(name) + "\",\"admission\":") != std::string::npos,
            std::string("the thermal state ") + name + " is missing or misspelled");
    require(json.find("\"ready\":true") != std::string::npos,
            std::string("the thermal state ") + name + " made the engine unready");
  }
}

// The log names the thermal state once per change: a start at nominal says
// nothing, a warm start names its state, and a repeat says nothing.
void testThermalStateTransitions() {
  ThermalStateReporter cool;
  require(cool.update(ThermalState::Nominal).empty() &&
              cool.state() == ThermalState::Nominal,
          "a start at nominal was logged");
  require(cool.update(ThermalState::Fair) == "Thermal state: nominal → fair" &&
              cool.update(ThermalState::Fair).empty() &&
              cool.update(ThermalState::Serious) == "Thermal state: fair → serious" &&
              cool.update(ThermalState::Nominal) == "Thermal state: serious → nominal" &&
              cool.state() == ThermalState::Nominal,
          "a thermal transition was not logged once");
  ThermalStateReporter warm;
  require(warm.state() == ThermalState::Nominal &&
              warm.update(ThermalState::Serious) == "Thermal state: serious" &&
              warm.state() == ThermalState::Serious &&
              warm.update(ThermalState::Serious).empty(),
          "a warm start was not logged once");
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
    testAneFfnStatus();
    testStreamedTableStatus();
    testStreamedExpertsStatus();
    testKvSpillStatus();
    testThermalStateStatus();
    testThermalStateTransitions();
    testStderrLinesStayWhole();
    testNoticesCarryTheTime();
    std::cout << "runtime status tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "runtime status tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
