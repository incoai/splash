#include "engine/Status.hpp"

#include "engine/Json.hpp"
#include "engine/Protocol.hpp"
#include "engine/RuntimeResources.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace splash::engine {
namespace {

const char *boolean(bool value) noexcept { return value ? "true" : "false"; }

uint64_t saturatingAdd(uint64_t left, uint64_t right) noexcept {
  return right > std::numeric_limits<uint64_t>::max() - left
             ? std::numeric_limits<uint64_t>::max()
             : left + right;
}

void appendBatch(std::ostringstream &out,
                 const RuntimeBatchMetricsSnapshot &batch) {
  out << '{' << "\"valid\":" << boolean(batch.valid)
      << ",\"width\":" << batch.width
      << ",\"input_tokens\":" << batch.inputTokens
      << ",\"output_tokens\":" << batch.outputTokens
      << ",\"drafted_tokens\":" << batch.draftedTokens
      << ",\"accepted_draft_tokens\":" << batch.acceptedDraftTokens
      << ",\"wall_ms\":" << batch.wallMilliseconds
      << ",\"tokens_per_second\":" << batch.tokensPerSecond << '}';
}

} // namespace

std::string MemoryStatusReporter::update(const ResourceWaitSnapshot &wait,
                                         bool growthAllowed) {
  const unsigned state = (!growthAllowed ? 1u : 0u) |
                         (wait.memory ? 2u : 0u) |
                         (wait.suspended ? 4u : 0u) |
                         (wait.draining ? 8u : 0u);
  if (state == state_)
    return {};
  state_ = state;
  if (!state)
    return "Memory: growth available; resource wait cleared";
  std::ostringstream out;
  out << "Memory: growth " << (growthAllowed ? "available" : "paused")
      << "; waiting=" << wait.memory << "; suspended=" << wait.suspended;
  if (wait.draining)
    out << "; waiting for resident requests to finish";
  return out.str();
}

std::string runtimeStatusJson(
    const EngineMemoryPlan &plan, const engine::EngineSnapshot &core,
    const metal::MetalMemoryStats &metalMemory, const WarmupReport &warmup,
    const MemoryAuditResult &memoryAudit, const RuntimeMetricsSnapshot &metrics,
    const model::ModelTelemetry &executorTelemetry,
    const engine::RuntimeCacheIdentity &cacheIdentity,
    const MemoryGovernorSnapshot &memoryGovernor, bool metalHealthy,
    std::string metalFailureReason, const ResourceWaitSnapshot &resourceWait) {
  const auto &resources = core.resources;
  const auto &scheduler = core.scheduler;
  const auto &pool = resources.pool;
  const auto &state = resources.stateCache;
  const auto &lookup = resources.lookup;
  const uint64_t trackedPhysical = saturatingAdd(
      metalMemory.allocatedBytes, metalMemory.sparseResidentBytes);
  const uint64_t currentBytes =
      std::max(trackedPhysical, metalMemory.deviceCurrentAllocatedBytes);
  const uint64_t peakBytes = std::max(
      {currentBytes, metalMemory.peakResidentBytes,
       metalMemory.devicePeakAllocatedBytes});
  // Warning pressure pauses growth but permits serving; only the governor's
  // critical verdict makes host pressure a readiness failure.
  const bool hostSafe = memoryGovernor.pressure != MemoryPressure::Critical;
  const bool ready = warmup.ready() && memoryAudit.valid && metalHealthy &&
                     hostSafe && currentBytes <= plan.breakdown().hardBudgetBytes;
  const double hitRate =
      core.cacheHits + core.coldMisses
          ? double(core.cacheHits) / double(core.cacheHits + core.coldMisses)
          : 0.0;

  std::ostringstream out;
  out << std::setprecision(10) << '{' << "\"schema_version\":" << protocol::kStatusSchemaVersion << ','
      << "\"ready\":" << boolean(ready)
      << ",\"maximum_context_tokens\":" << core.maximumContextTokens
      << ",\"memory_pressure\":"
      << json::quote(memoryPressureName(memoryGovernor.pressure))
      << ",\"admission\":{\"waiting\":"
      << resourceWait.memory + resourceWait.concurrency
      << ",\"waiting_memory\":" << resourceWait.memory
      << ",\"waiting_concurrency\":" << resourceWait.concurrency
      << ",\"suspended\":" << resourceWait.suspended
      << ",\"draining\":" << boolean(resourceWait.draining)
      << ",\"oldest_wait_ms\":" << resourceWait.oldestWaitMilliseconds << "}"
      << ",\"identity\":{\"cache\":{"
      << "\"loaded_model_layout_sha256\":"
      << json::quote(cacheIdentity.modelLayoutSha256)
      << ",\"runtime_cache_namespace\":"
      << json::quote(cacheIdentity.namespaceSha256)
      << ",\"build_id\":" << json::quote(cacheIdentity.buildId)
      << ",\"dtype\":" << json::quote(kQ8FormatName)
      << ",\"block_tokens\":" << kv::kPageTokens
      << "},\"q8\":{\"target_model_sha256\":"
      << json::quote(digestHex(cacheIdentity.q8Layout.modelArtifactSha256))
      << ",\"quantization\":\"symmetric_int8\""
      << ",\"scale_type\":\"float32\""
      << ",\"key_layout\":\"token_major\""
      << ",\"value_layout\":\"dimension_major\"}},"
      << "\"memory_plan\":" << plan.toStatusJson()
      << ",\"memory_actual\":{\"dense_bytes\":" << metalMemory.allocatedBytes
      << ",\"sparse_virtual_bytes\":" << metalMemory.sparseVirtualBytes
      << ",\"sparse_resident_bytes\":" << metalMemory.sparseResidentBytes
      << ",\"current_bytes\":" << currentBytes
      << ",\"peak_bytes\":" << peakBytes << "}"
      << ",\"memory_governor\":{\"limit_bytes\":" << memoryGovernor.limitBytes
      << ",\"observed_resident_bytes\":" << memoryGovernor.observedResidentBytes
      << ",\"reserved_bytes\":" << memoryGovernor.reservedBytes
      << ",\"headroom_bytes\":" << memoryGovernor.headroomBytes
      << ",\"growth_allowed\":" << boolean(memoryGovernor.growthAllowed)
      << ",\"denied_reservations\":" << memoryGovernor.deniedReservations
      << ",\"system_pressure\":"
      << json::quote(memoryPressureName(memoryGovernor.systemPressure))
      << ",\"host_measurement_valid\":"
      << boolean(memoryGovernor.hostMeasurementValid)
      << ",\"host_available_bytes\":" << memoryGovernor.hostAvailableBytes
      << ",\"host_reserve_bytes\":" << memoryGovernor.hostReserveBytes
      << ",\"host_headroom_bytes\":" << memoryGovernor.hostHeadroomBytes << "}"
      << ",\"memory_audit\":" << memoryAudit.toStatusJson()
      << ",\"kv\":{\"block_tokens\":" << kv::kPageTokens
      << ",\"blocks\":" << resources.kvCache.blocks
      << ",\"cache_bytes\":" << resources.kvCache.bytes
      << ",\"pages_total\":" << pool.pagesTotal
      << ",\"pages_active\":" << pool.pagesActive
      << ",\"pages_cache\":" << pool.pagesPrefix
      << ",\"pages_free\":" << pool.pagesFree
      << ",\"pages_free_resident\":" << pool.pagesFreeResident
      << ",\"pages_resident\":" << pool.pagesResident
      << ",\"resident_backing_bytes\":" << pool.residentBackingBytes
      << ",\"reclaimable_backing_bytes\":" << pool.reclaimableBackingBytes
      << ",\"sparse_tile_bytes\":" << metalMemory.sparseTileBytes
      << ",\"pending_unmaps\":" << metalMemory.pendingSparseUnmaps
      << ",\"pending_unmap_ms\":"
      << metalMemory.pendingSparseUnmapSeconds * 1000.0
      << ",\"unmaps_completed\":" << metalMemory.completedSparseUnmaps
      << ",\"unmap_last_ms\":" << metalMemory.lastSparseUnmapSeconds * 1000.0
      << ",\"unmap_max_ms\":" << metalMemory.maxSparseUnmapSeconds * 1000.0
      << ",\"map_wait_event\":" << metalMemory.sparseMapWaitEvent
      << ",\"pending_map_wait_ms\":" << metalMemory.pendingSparseMapWaitSeconds * 1000.0
      << ",\"map_wait_last_ms\":" << metalMemory.lastSparseMapWaitSeconds * 1000.0
      << ",\"map_wait_max_ms\":" << metalMemory.maxSparseMapWaitSeconds * 1000.0
      << "}"
      << ",\"state\":{\"entries\":" << state.entries
      << ",\"pinned\":" << state.pinned << ",\"bytes\":" << state.bytes
      << ",\"resident_bytes\":" << executorTelemetry.stateResidentBytes
      << ",\"active_cells\":" << resources.activeRequests
      << ",\"warm_idle_cells\":" << executorTelemetry.warmIdleStateCells
      << ",\"cell_ceiling\":" << model::ExecutionLimits::maximumBatchWidth
      << ",\"hits\":" << state.hits << ",\"misses\":" << state.misses
      << ",\"publications\":" << state.publications
      << ",\"deduplicated_publications\":" << state.deduplicatedPublications
      << ",\"evictions\":" << state.evictions
      << ",\"checkpoint_entries\":" << state.checkpointEntries
      << ",\"checkpoint_bytes\":" << state.checkpointBytes
      << ",\"checkpoint_evictions\":" << state.checkpointEvictions
      << ",\"checkpoint_retirements\":" << state.checkpointRetirements << "}"
      << ",\"cache\":{\"lookups\":" << lookup.lookups
      << ",\"hits\":" << core.cacheHits
      << ",\"cold_misses\":" << core.coldMisses << ",\"hit_rate\":" << hitRate
      << ",\"kv_hit_tokens\":" << lookup.kvHitTokens
      << ",\"state_hit_tokens\":" << lookup.stateHitTokens
      << ",\"reused_tokens\":" << core.reusedTokens
      << ",\"replay_state_publications\":" << core.replayStatePublications
      << ",\"deduplicated_state_publications\":"
      << core.deduplicatedStatePublications
      << ",\"recycled_state_publications\":"
      << core.recycledStatePublications
      << ",\"replay_state_publication_failures\":"
      << core.replayStatePublicationFailures
      << ",\"lazy_junctions\":" << lookup.lazyJunctions
      << ",\"junction_materializations\":" << core.junctionMaterializations
      << ",\"junction_materialization_failures\":"
      << core.junctionMaterializationFailures
      << ",\"checkpoint_publications\":" << core.checkpointPublications
      << ",\"checkpoint_publication_failures\":"
      << core.checkpointPublicationFailures
      << ",\"resource_suspensions\":" << core.resourceSuspensions
      << ",\"resource_resumptions\":" << core.resourceResumptions
      << ",\"resource_replay_tokens\":" << core.resourceReplayTokens << "}"
      << ",\"draft_context\":{\"target_prefill_rows\":"
      << executorTelemetry.targetPrefillRows
      << ",\"active_rows\":" << executorTelemetry.draftContextRowsActive
      << ",\"materialization_rows\":"
      << executorTelemetry.draftContextRowsMaterialization
      << ",\"avoided_rows\":" << executorTelemetry.draftContextRowsAvoided
      << ",\"restore_skipped\":" << executorTelemetry.draftStateRestoreSkipped
      << ",\"resets\":" << executorTelemetry.draftStateResets << "}"
      // Model-lifetime timings include warmup; request metrics do not.
      << ",\"model_timing\":{\"scope\":\"model_lifetime\""
      << ",\"prefill\":{\"last_gpu_ms\":"
      << executorTelemetry.lastPrefillGpuSeconds * 1000.0
      << ",\"last_wall_ms\":" << executorTelemetry.lastPrefillWallSeconds * 1000.0
      << ",\"total_gpu_ms\":" << executorTelemetry.totalPrefillGpuSeconds * 1000.0
      << ",\"total_wall_ms\":" << executorTelemetry.totalPrefillWallSeconds * 1000.0
      << "},\"decode\":{\"last_gpu_ms\":"
      << executorTelemetry.lastDecodeGpuSeconds * 1000.0
      << ",\"last_wall_ms\":" << executorTelemetry.lastDecodeWallSeconds * 1000.0
      << ",\"total_gpu_ms\":" << executorTelemetry.totalDecodeGpuSeconds * 1000.0
      << ",\"total_wall_ms\":" << executorTelemetry.totalDecodeWallSeconds * 1000.0
      << "}}"
      << ",\"constraint_masks\":{\"overlap_batches\":"
      << executorTelemetry.constrainedMaskOverlapBatches
      << ",\"overlap_requests\":"
      << executorTelemetry.constrainedMaskOverlapRequests
      << ",\"last_target_forward_gpu_ms\":"
      << executorTelemetry.lastConstrainedTargetForwardGpuSeconds * 1000.0
      << ",\"total_target_forward_gpu_ms\":"
      << executorTelemetry.totalConstrainedTargetForwardGpuSeconds * 1000.0
      << ",\"last_residual_wait_ms\":"
      << executorTelemetry.lastConstrainedMaskWaitSeconds * 1000.0
      << ",\"total_residual_wait_ms\":"
      << executorTelemetry.totalConstrainedMaskWaitSeconds * 1000.0 << "}"
      << ",\"images\":{\"encodes\":" << executorTelemetry.imageEncodes
      << ",\"embedding_reuses\":" << executorTelemetry.imageEmbeddingReuses
      << "}"
      << ",\"scheduler\":{\"queued\":" << scheduler.queued
      << ",\"waiting_resources\":" << scheduler.waitingResources
      << ",\"waiting_prefix\":" << scheduler.waitingPrefix
      << ",\"prefilling\":" << scheduler.prefilling
      << ",\"decoding\":" << scheduler.decoding
      << ",\"waiting_mask\":" << scheduler.waitingMask
      << ",\"terminal\":" << scheduler.terminal
      << ",\"prefill_batches\":" << scheduler.prefillBatches
      << ",\"prefill_rows\":" << scheduler.prefillRows
      << ",\"decode_batches\":" << scheduler.decodeBatches
      << ",\"decode_batches_by_width\":{\"b1\":"
      << scheduler.decodeBatchesByWidth[0]
      << ",\"b2\":" << scheduler.decodeBatchesByWidth[1]
      << ",\"b3\":" << scheduler.decodeBatchesByWidth[2]
      << ",\"b4\":" << scheduler.decodeBatchesByWidth[3] << "}}"
      << ",\"requests\":{\"submitted\":" << core.submitted
      << ",\"completed\":" << core.completed
      << ",\"cancelled\":" << core.cancelled << ",\"failed\":" << core.failed
      << "}"
      << ",\"metrics\":{\"ttft_ms\":{\"p50\":" << metrics.ttftP50Milliseconds
      << ",\"p95\":" << metrics.ttftP95Milliseconds
      << ",\"samples\":" << metrics.ttftSamples
      << "},\"itl_ms\":{\"p50\":" << metrics.itlP50Milliseconds
      << ",\"p95\":" << metrics.itlP95Milliseconds
      << ",\"samples\":" << metrics.itlSamples
      << "},\"prefill_input_tokens\":" << metrics.prefillInputTokens
      << ",\"prefill_wall_ms\":" << metrics.prefillWallMilliseconds
      << ",\"prefill_tokens_per_second\":" << metrics.prefillTokensPerSecond
      << ",\"decode_output_tokens\":" << metrics.decodeOutputTokens
      << ",\"decode_wall_ms\":" << metrics.decodeWallMilliseconds
      << ",\"decode_tokens_per_second\":" << metrics.decodeTokensPerSecond
      << ",\"drafted_tokens\":" << metrics.draftedTokens
      << ",\"accepted_draft_tokens\":" << metrics.acceptedDraftTokens
      << ",\"draft_acceptance_rate\":" << metrics.draftAcceptanceRate
      << ",\"capacity_failures\":" << metrics.capacityFailures
      << ",\"metal_failures\":" << metrics.metalFailures
      << ",\"current_prefill_batch\":";
  appendBatch(out, metrics.currentPrefillBatch);
  out << ",\"current_decode_batch\":";
  appendBatch(out, metrics.currentDecodeBatch);
  const std::string prefillName =
      "prefill_" + std::to_string(SPLASH_PREFILL_TOKEN_BUDGET);
  const std::pair<std::string_view, WarmupStepStatus> warmupSteps[] = {
      {prefillName, warmup.maximumPrefill},
      {"decode_b1", warmup.decodeBatches[0]},
      {"decode_b2", warmup.decodeBatches[1]},
      {"decode_b3", warmup.decodeBatches[2]},
      {"decode_b4", warmup.decodeBatches[3]},
      {"draft_verify_commit", warmup.draftVerifyCommit},
      {"composite_state_restore", warmup.compositeStateRestore},
  };
  out << "},\"warmup\":{";
  for (const auto &[name, status] : warmupSteps) {
    out << json::quote(name) << ':'
        << boolean(status == WarmupStepStatus::Complete) << ',';
  }
  out << "\"memory_limited_steps\":[";
  bool separator = false;
  for (const auto &[name, status] : warmupSteps) {
    if (status != WarmupStepStatus::MemoryLimited)
      continue;
    if (separator)
      out << ',';
    out << json::quote(name);
    separator = true;
  }
  out << "],\"memory_budget_validated\":"
      << boolean(warmup.memoryBudgetValidated)
      << ",\"actual_peak_bytes\":" << warmup.actualPeakBytes
      << ",\"detail\":" << json::quote(warmup.maximumPrefillDetail)
      << ",\"error\":" << json::quote(warmup.error) << "}"
      << ",\"metal\":{\"healthy\":" << boolean(metalHealthy)
      << ",\"failure_reason\":" << json::quote(metalFailureReason) << "}}";
  return out.str();
}

} // namespace splash::engine
