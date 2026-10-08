#include "engine/Status.hpp"

#include "engine/Json.hpp"
#include "engine/Protocol.hpp"
#include "engine/RuntimeResources.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <utility>

namespace splash::engine {
namespace {

const char *boolean(bool value) noexcept { return value ? "true" : "false"; }

// idle_release_seconds is null while the engine keeps them (--idle-release off).
void appendWeights(std::ostringstream &out, const WeightsSnapshot &weights) {
  out << "{\"idle_release_seconds\":";
  if (std::isinf(weights.idleReleaseSeconds))
    out << "null";
  else
    out << weights.idleReleaseSeconds;
  out << ",\"released\":" << boolean(weights.released)
      << ",\"restores\":" << weights.restores << '}';
}

// reruns: the chunks the GPU ran again alone once the split's work for them
// failed (model::ModelTelemetry::aneFfnReruns).
void appendAneFfn(std::ostringstream &out, const AneFfnSnapshot &split, uint64_t reruns) {
  constexpr std::string_view kStates[] = {"off", "split", "stopped"};
  out << "{\"state\":" << json::quote(kStates[static_cast<size_t>(split.state)])
      << ",\"share\":" << split.share << ",\"minimum_rows\":" << split.minimumRows
      << ",\"reason\":" << json::quote(split.reason) << ",\"split_commands\":" << split.commands
      << ",\"reruns\":" << reruns << ",\"ane_ms\":" << split.aneMilliseconds
      << ",\"evaluations\":" << split.evaluations << '}';
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

std::string runtimeStatusJson(
    const EngineMemoryPlan &plan, const engine::EngineSnapshot &core,
    const metal::MetalMemoryStats &metalMemory, const WarmupReport &warmup,
    const MemoryAuditResult &memoryAudit, const RuntimeMetricsSnapshot &metrics,
    const model::ModelTelemetry &executorTelemetry,
    const engine::RuntimeCacheIdentity &cacheIdentity,
    const MemoryGovernorSnapshot &memoryGovernor, bool metalHealthy,
    std::string metalFailureReason, const ResourceWaitSnapshot &resourceWait,
    const NativeLoopTiming &loop, const WeightsSnapshot &weights,
    const AneFfnSnapshot &aneFfn) {
  const auto &resources = core.resources;
  const auto &scheduler = core.scheduler;
  const auto &pool = resources.pool;
  const auto &state = resources.stateCache;
  const auto &lookup = resources.lookup;
  const uint64_t currentBytes = std::max(
      metalMemory.allocatedBytes, metalMemory.deviceCurrentAllocatedBytes);
  const uint64_t peakBytes = std::max(
      {currentBytes, metalMemory.peakAllocatedBytes,
       metalMemory.devicePeakAllocatedBytes});
  // Warning pressure pauses growth but permits serving; only the governor's
  // critical verdict makes host pressure a readiness failure. A status exists
  // only after warmup and the memory audit passed.
  const bool hostSafe = memoryGovernor.pressure != MemoryPressure::Critical;
  const bool ready = metalHealthy && hostSafe &&
                     currentBytes <= plan.breakdown().hardBudgetBytes;
  const double hitRate =
      core.cacheHits + core.coldMisses
          ? double(core.cacheHits) / double(core.cacheHits + core.coldMisses)
          : 0.0;

  const kv::Format kvFormat = cacheIdentity.kvLayout.format;

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
      << ",\"held_behind_refusal\":" << resourceWait.heldBehindRefusal
      << ",\"restoring\":" << resourceWait.restoring
      << ",\"suspended\":" << resourceWait.suspended
      << ",\"draining\":" << boolean(resourceWait.draining)
      << ",\"oldest_wait_ms\":" << resourceWait.oldestWaitMilliseconds << "}"
      << ",\"loop\":{\"max_tick_ms\":" << loop.maxTickMilliseconds << "}"
      << ",\"identity\":{\"cache\":{"
      << "\"loaded_model_layout_sha256\":"
      << json::quote(cacheIdentity.modelLayoutSha256)
      << ",\"build_id\":" << json::quote(cacheIdentity.buildId)
      << ",\"dtype\":" << json::quote(kv::storageFormatName(kvFormat))
      << ",\"block_tokens\":" << kv::kPageTokens
      << "},\"kv\":{\"target_model_sha256\":"
      << json::quote(cacheIdentity.targetModelSha256)
      << ",\"format\":" << json::quote(kv::formatName(kvFormat))
      << ",\"quantization\":"
      << json::quote(kvFormat == kv::Format::Int8 ? "symmetric_int8" : "none")
      << ",\"scale_type\":" << json::quote(kvFormat == kv::Format::Int8 ? "float32" : "none")
      << ",\"key_layout\":\"token_major\""
      << ",\"value_layout\":\"dimension_major\"}},"
      << "\"memory_plan\":" << plan.toStatusJson()
      << ",\"memory_actual\":{\"allocated_bytes\":" << metalMemory.allocatedBytes
      << ",\"current_bytes\":" << currentBytes
      << ",\"peak_bytes\":" << peakBytes << "}"
      << ",\"memory_governor\":{\"limit_bytes\":" << memoryGovernor.limitBytes
      << ",\"charged_bytes\":" << memoryGovernor.chargedBytes
      << ",\"headroom_bytes\":" << memoryGovernor.headroomBytes
      << ",\"growth_allowed\":" << boolean(memoryGovernor.hostGrowthAllowed)
      << ",\"denied_reservations\":" << memoryGovernor.deniedReservations
      << ",\"system_pressure\":"
      << json::quote(memoryPressureName(memoryGovernor.systemPressure))
      << ",\"host_measurement_valid\":"
      << boolean(memoryGovernor.hostMeasurementValid)
      << ",\"host_available_bytes\":" << memoryGovernor.hostAvailableBytes
      << ",\"host_reserve_bytes\":" << memoryGovernor.hostReserveBytes
      << ",\"host_headroom_bytes\":" << memoryGovernor.hostHeadroomBytes
      << ",\"host_debt_bytes\":" << memoryGovernor.hostDebtBytes << "}"
      << ",\"memory_audit\":" << memoryAudit.toStatusJson()
      << ",\"weights\":";
  appendWeights(out, weights);
  out << ",\"ane_ffn\":";
  appendAneFfn(out, aneFfn, executorTelemetry.aneFfnReruns);
  out << ",\"kv\":{\"block_tokens\":" << kv::kPageTokens
      << ",\"pages_allocated\":" << pool.pagesAllocated
      << ",\"pages_active\":" << pool.pagesActive
      << ",\"pages_cache\":" << pool.pagesPrefix
      << ",\"pages_free\":" << pool.pagesFree
      << ",\"allocated_bytes\":" << pool.allocatedBytes
      << ",\"reclaimable_bytes\":" << pool.reclaimableBytes
      << ",\"extent_allocations\":" << pool.extentAllocations
      << ",\"extent_releases\":" << pool.extentReleases
      << ",\"extent_allocate_max_ms\":" << pool.extentAllocateMaxMilliseconds
      << ",\"extent_release_max_ms\":" << pool.extentReleaseMaxMilliseconds
      << ",\"extent_compactions\":" << pool.extentCompactions
      << ",\"pages_moved\":" << pool.pagesMoved
      << ",\"extent_compact_max_ms\":" << resources.extentCompactMaxMilliseconds
      << "}"
      << ",\"state\":{\"entries\":" << state.entries
      << ",\"pinned\":" << state.pinned << ",\"in_use\":" << state.inUse
      << ",\"in_use_evictions\":" << state.inUseEvictions << ",\"bytes\":" << state.bytes
      << ",\"allocated_bytes\":" << executorTelemetry.stateAllocatedBytes
      << ",\"active_lanes\":" << resources.activeRequests
      << ",\"idle_gdn_cells\":" << executorTelemetry.idleGdnCells
      << ",\"idle_draft_rings\":" << executorTelemetry.idleDraftRings
      << ",\"publications\":" << state.publications
      << ",\"evictions\":" << state.evictions
      << ",\"checkpoint_entries\":" << state.checkpointEntries
      << ",\"checkpoint_bytes\":" << state.checkpointBytes
      << ",\"checkpoint_evictions\":" << state.checkpointEvictions
      << ",\"checkpoint_retirements\":" << state.checkpointRetirements
      << ",\"disk_hits\":" << lookup.stateDiskHits
      << ",\"disk_promotions\":" << state.promotions
      << ",\"disk_promotions_skipped\":" << state.promotionsSkipped
      << ",\"disk_bytes\":" << state.diskBytes
      << ",\"offloads\":" << state.offloads
      << ",\"offload_failures\":" << state.offloadFailures
      << ",\"invalidations\":" << state.invalidations
      << "}"
      << ",\"disk\":{\"capacity_bytes\":" << resources.kvTier.capacityBytes
      << ",\"used_bytes\":" << resources.kvTier.usedBytes
      << ",\"file_bytes\":" << resources.kvTier.fileBytes
      << ",\"read_bytes\":" << resources.kvTier.readBytes
      << ",\"written_bytes\":" << resources.kvTier.writtenBytes
      << ",\"kv_blocks\":" << resources.kvTier.diskBlocks
      << ",\"kv_bytes\":" << resources.kvTier.diskBytes
      << ",\"kv_demotions\":" << resources.kvTier.demotions
      << ",\"kv_demotion_failures\":" << resources.kvTier.demotionFailures
      << ",\"kv_demotions_refused\":" << resources.kvTier.demotionsRefused
      << ",\"kv_restores\":" << resources.kvTier.restores
      << ",\"kv_restore_failures\":" << resources.kvTier.restoreFailures
      << ",\"kv_pending_pages\":" << resources.kvTier.pendingPages
      << ",\"persistent\":" << boolean(resources.persistent)
      << ",\"kv_copies\":" << resources.kvTier.copies
      << ",\"kv_copy_failures\":" << resources.kvTier.copyFailures
      << ",\"taken_back\":{\"states\":" << resources.adoption.states
      << ",\"kv_blocks\":" << resources.adoption.blocks
      << ",\"bytes\":" << resources.adoption.bytes
      << ",\"left_behind\":" << resources.adoption.dropped << "}"
      << ",\"write_behind\":{\"waiting\":" << core.writeBehind.waiting
      << ",\"durable\":" << core.writeBehind.durable
      << ",\"unneeded\":" << core.writeBehind.unneeded
      << ",\"refused\":" << core.writeBehind.refused << "}"
      << "}"
      << ",\"cache\":{\"probe_hashed_blocks\":" << lookup.probeHashedBlocks
      << ",\"hits\":" << core.cacheHits
      << ",\"cold_misses\":" << core.coldMisses << ",\"hit_rate\":" << hitRate
      << ",\"kv_hit_tokens\":" << lookup.kvHitTokens
      << ",\"kv_disk_hit_tokens\":" << resources.kvTier.restores * kv::kPageTokens
      << ",\"lost_state_misses\":" << lookup.lostStateMisses
      << ",\"reused_tokens\":" << core.reusedTokens
      << ",\"replay_state_publications\":" << core.replayStatePublications
      << ",\"deduplicated_state_publications\":"
      << core.deduplicatedStatePublications
      << ",\"recycled_state_publications\":"
      << core.recycledStatePublications
      << ",\"disk_state_publications\":" << core.diskStatePublications
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
      << ",\"priority_suspensions\":" << core.prioritySuspensions
      << ",\"resource_resumptions\":" << core.resourceResumptions
      << ",\"resource_replay_tokens\":" << core.resourceReplayTokens << "}"
      << ",\"draft_context\":{\"target_prefill_rows\":"
      << executorTelemetry.targetPrefillRows
      << ",\"prompt_end_rows\":" << executorTelemetry.draftContextRowsActive
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
      << ",\"arena_bytes\":" << executorTelemetry.visionArenaBytes
      << ",\"cached_bytes\":" << executorTelemetry.embeddingCacheBytes
      << ",\"state_held_bytes\":" << executorTelemetry.stateHeldImageBytes
      << ",\"rows_bytes\":" << executorTelemetry.imageRowsBytes
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
      << ",\"decode_cycle_ms\":" << metrics.decodeCycleMilliseconds
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
  out << "],\"detail\":" << json::quote(warmup.maximumPrefillDetail) << "}"
      << ",\"metal\":{\"healthy\":" << boolean(metalHealthy)
      << ",\"failure_reason\":" << json::quote(metalFailureReason) << "}}";
  return out.str();
}

} // namespace splash::engine
