#include "engine/Status.hpp"

#include "Supervision.hpp"
#include "engine/Json.hpp"
#include "engine/Protocol.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/Model.hpp"

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace splash::engine {
namespace {

// The RoPE scaling every key of the cache went through: YaRN's factor, to the
// digits that read back as it, over its original window; null without YaRN.
// The rest of the document keeps its own precision.
void writeRopeScaling(json::Writer &w, const model::RopeScaling &rope) {
  if (!rope.yarn()) {
    w.null();
    return;
  }
  w.object([&] {
    w.field("type", "yarn").key("factor").exact(rope.factor);
    w.field("original_context_tokens", rope.originalContextTokens);
  });
}

// idle_release_seconds is infinite with --idle-release off, and so null.
void writeWeights(json::Writer &w, const WeightsSnapshot &weights) {
  w.object([&] {
    w.field("idle_release_seconds", weights.idleReleaseSeconds)
        .field("released", weights.released)
        .field("restoring", weights.restoring)
        .field("restores", weights.restores)
        .field("restore_failures", weights.restoreFailures);
  });
}

// reruns: the chunks the GPU ran again alone once the split's work for them
// failed (model::ModelTelemetry::aneFfnReruns).
void writeAneFfn(json::Writer &w, const AneFfnSnapshot &split, uint64_t reruns) {
  constexpr std::string_view kStates[] = {"off", "split", "stopped"};
  w.object([&] {
    w.field("state", kStates[static_cast<size_t>(split.state)])
        .field("share", split.share)
        .field("minimum_rows", split.minimumRows)
        .field("reason", split.reason)
        .field("split_commands", split.commands)
        .field("reruns", reruns)
        .field("ane_ms", split.aneMilliseconds)
        .field("evaluations", split.evaluations);
  });
}

void writeBatch(json::Writer &w, const RuntimeBatchMetricsSnapshot &batch) {
  w.object([&] {
    w.field("valid", batch.valid)
        .field("width", batch.width)
        .field("input_tokens", batch.inputTokens)
        .field("output_tokens", batch.outputTokens)
        .field("drafted_tokens", batch.draftedTokens)
        .field("accepted_draft_tokens", batch.acceptedDraftTokens)
        .field("wall_ms", batch.wallMilliseconds)
        .field("tokens_per_second", batch.tokensPerSecond);
  });
}

const char *phaseName(engine::Phase phase) noexcept {
  switch (phase) {
  case engine::Phase::Queued: return "queued";
  case engine::Phase::WaitingResources: return "waiting_resources";
  case engine::Phase::WaitingPrefix: return "waiting_prefix";
  case engine::Phase::Prefill: return "prefill";
  case engine::Phase::Decode: return "decode";
  case engine::Phase::WaitingMask: return "waiting_mask";
  case engine::Phase::Completed: return "completed";
  case engine::Phase::Cancelled: return "cancelled";
  case engine::Phase::Failed: return "failed";
  }
  return "unknown";
}

// The API's names (server REQUEST_PRIORITIES).
const char *priorityName(engine::RequestPriority priority) noexcept {
  switch (priority) {
  case engine::RequestPriority::Foreground: return "foreground";
  case engine::RequestPriority::Normal: return "normal";
  case engine::RequestPriority::Background: return "background";
  }
  return "unknown";
}

const char *writesGatedName(DiskWritesGate gate) noexcept {
  switch (gate) {
  case DiskWritesGate::No: return "no";
  case DiskWritesGate::Budget: return "budget";
  case DiskWritesGate::Closed: return "closed";
  }
  return "unknown";
}

// What the streamed stores count by, in the order of their uses
// (model::SparseRowUse, and ops::MoePhase for model::ExpertStoreStats::uses).
constexpr std::string_view kStreamedUses[] = {"prefill", "decode"};
static_assert(std::size(kStreamedUses) == model::kSparseRowUses);
// The causes of decode gaps past the bound, by GapCause.
constexpr std::string_view kGapCauseNames[] = {"prefill_unit", "claim_wait",   "ssd_read",
                                               "pool_release", "replan", "other"};
static_assert(std::size(kGapCauseNames) == kGapCauses);

// A latency histogram (model::LatencyStats) in milliseconds, as ttft_ms is:
// its quantiles, maximum and total, and its samples.
void writeLatency(json::Writer &w, const model::LatencyStats &latency) {
  w.object([&] {
    w.field("p50", latency.p50Milliseconds)
        .field("p95", latency.p95Milliseconds)
        .field("p99", latency.p99Milliseconds)
        .field("max", latency.maximumMilliseconds)
        .field("total", latency.totalMilliseconds)
        .field("samples", latency.count);
  });
}

// A streamed table's store (--ple-memory), or null: its memory, its cache,
// per use (prefill, decode) its stages, rows and stage latencies, its reads
// and prefetches, and whether a failure poisoned it.
void writeStreamedTable(json::Writer &w, const std::optional<model::SparseRowStats> &table) {
  if (!table) {
    w.null();
    return;
  }
  w.object([&] {
    w.field("table", table->table)
        .field("table_bytes", table->tableBytes)
        .field("cache_bytes", table->cacheBytes)
        .field("cache_capacity_rows", table->cacheCapacityRows)
        .field("cached_rows", table->cachedRows)
        .field("evictions", table->evictions);
    for (uint32_t use = 0; use < model::kSparseRowUses; ++use) {
      const model::SparseRowStats::Use &counters = table->uses[use];
      w.object(kStreamedUses[use], [&] {
        w.field("stages", counters.stages)
            .field("requested_rows", counters.requestedRows)
            .field("unique_rows", counters.uniqueRows)
            .field("hit_rows", counters.hitRows)
            .field("read_rows", counters.readRows)
            .field("hit_rate",
                   counters.uniqueRows ? double(counters.hitRows) / double(counters.uniqueRows) : 0.0);
        w.key("stage_ms");
        writeLatency(w, counters.stage);
      });
    }
    w.field("read_bytes", table->readBytes)
        .field("read_retries", table->readRetries)
        .field("read_errors", table->readErrors)
        .key("read_ms");
    writeLatency(w, table->read);
    w.field("prefetch_requested_rows", table->prefetchRequestedRows)
        .field("prefetch_read_rows", table->prefetchReadRows)
        .field("prefetch_dropped_rows", table->prefetchDroppedRows)
        .field("poisoned", table->poisoned)
        .field("failure", table->failure);
  });
}

// The round trips of a store's host steps, one per round: pickup, work and
// resume.
void writeHostSteps(json::Writer &w, const model::LatencyStats &pickup, const model::LatencyStats &work,
                    const model::LatencyStats &resume) {
  w.object([&] {
    w.key("pickup_ms");
    writeLatency(w, pickup);
    w.key("work_ms");
    writeLatency(w, work);
    w.key("resume_ms");
    writeLatency(w, resume);
  });
}

// Streamed routed experts' store (--expert-memory), or null: its bytes at
// its cap and at its floor, its live chunks and those at the cap, the
// chunks it gave back while memory was short and took back after, its live
// chunks' bytes, its staging, its live slots and the experts resident in
// them, per use (prefill, decode) its loads: the distinct experts they
// routed, those not resident (read) and of them those read ahead
// (prefetched), the experts read ahead whole, the bytes read from the
// sources and their host steps' round trips; and why a load failed, which
// fails every later load.
void writeStreamedExperts(json::Writer &w, const std::optional<model::ExpertStoreStats> &experts) {
  if (!experts) {
    w.null();
    return;
  }
  w.object([&] {
    w.field("cap_bytes", experts->capBytes)
        .field("floor_bytes", experts->floorBytes)
        .field("chunks", experts->chunks)
        .field("cap_chunks", experts->capChunks)
        .field("shrinks", experts->shrinks)
        .field("regrows", experts->regrows)
        .field("arena_bytes", experts->arenaBytes)
        .field("staging_bytes", experts->stagingBytes)
        .field("slots", experts->slots)
        .field("experts", experts->experts)
        .field("resident", experts->resident);
    for (size_t use = 0; use < std::size(kStreamedUses); ++use) {
      const model::ExpertStoreStats::Use &counters = experts->uses[use];
      w.object(kStreamedUses[use], [&] {
        w.field("loads", counters.loads)
            .field("routed", counters.routed)
            .field("read", counters.read)
            .field("prefetched", counters.prefetched)
            .field("read_ahead", counters.readAhead)
            .field("hit_rate",
                   counters.routed ? double(counters.routed - counters.read) / double(counters.routed) : 0.0)
            .field("read_bytes", counters.readBytes)
            .key("host_steps");
        writeHostSteps(w, counters.pickup, counters.work, counters.resume);
      });
    }
    w.field("failure", experts->failure);
  });
}

// A windowed layout's KV (kv.window): the pages a window and a state's tail
// keep Full, whether pages past every window give up their windowed layers'
// K/V, the pool's pages of each form in use, the blocks states' tails pin,
// and the pages trimmed, the blocks that took the page of a writer that
// computed them again and the states refused a tail that was not Full
// (Cache::snapshot).
void writeKvWindow(json::Writer &w, const KvWindowSnapshot &window, const KvPoolSnapshot &pool) {
  const uint32_t slim = pool.slim.pagesAllocated - pool.slim.pagesFree;
  const uint32_t used = pool.pagesAllocated - pool.pagesFree;
  w.object([&] {
    w.field("window_pages", window.windowPages)
        .field("trim", window.trim)
        .field("full_pages", used - slim)
        .field("slim_pages", slim)
        .field("slim_bytes", pool.slim.allocatedBytes)
        .field("pinned_blocks", window.pinnedBlocks)
        .field("trimmed_pages", window.trimmedPages)
        .field("upgrades", window.upgrades)
        .field("refused_states", window.refusedStates);
  });
}

// A spill-capable target's KV on the SSD cache (kv_spill): the engine's
// requests by KV mode, conversions, bounds, ends (and of those, the converted
// lanes' bounds and ends) and refusals, and the starts
// waiting for the quota; the pages that hold only their index, the pool's
// pages of that form in use (each a spilled block's, which requests may hold
// as the cache does), of `indexPageBytes` each, and those spilled; the
// write-through of Spill requests' pages; the moves between page forms,
// rewrites and restores; the quota of active KV; the Index runway, the free
// pages of that form; and the readers' memory and reads. A target that keeps
// its KV in memory has no kv_spill section.
void writeKvSpill(json::Writer &w, const StatusSample::KvSpill &spill, uint64_t indexPageBytes,
                  const engine::EngineSnapshot &core, const ResourceWaitSnapshot &resourceWait) {
  const KvSpillSnapshot &cache = spill.cache;
  const engine::SpillRequestsSnapshot &requests = core.spill;
  const KvPoolFormSnapshot &index = core.resources.pool.index;
  const uint32_t indexPages = index.pagesAllocated - index.pagesFree;
  w.object([&] {
    w.object("requests", [&] {
      w.field("spill", requests.spill)
          .field("resident", requests.resident)
          .field("converted", requests.converted)
          .field("bounded", requests.bounded)
          .field("length_ends", requests.lengthEnds)
          .field("converted_bounded", requests.convertedBounded)
          .field("converted_length_ends", requests.convertedLengthEnds);
      w.object("refused", [&] {
        w.field("quota", requests.refusedQuota)
            .field("write_budget", requests.refusedWriteBudget)
            .field("closed", requests.refusedClosed);
      });
      w.field("waiting_quota", resourceWait.diskQuota);
    });
    w.object("pages", [&] {
      w.field("index", indexPages)
          .field("index_bytes", indexPages * indexPageBytes)
          .field("spilled", cache.spilledPages)
          .field("spilled_bytes", cache.spilledBytes);
    });
    w.object("write_through", [&] {
      w.field("pages", cache.writeThroughPages)
          .field("bytes", cache.writeThroughBytes)
          .field("in_flight", cache.writeThroughInFlight)
          .field("failures", cache.writeThroughFailures)
          .field("paused", cache.writeThroughPaused);
    });
    w.field("spill_moves", cache.spillMoves).field("unspills", cache.unspills).field("rewrites", cache.rewrites);
    w.object("restores", [&] {
      w.field("index_pages", cache.indexRestores).field("whole_pages", cache.wholeRestores).key("ms");
      writeLatency(w, cache.restore);
    });
    w.object("quota", [&] {
      w.field("active_limit_bytes", cache.activeLimitBytes)
          .field("held_bytes", cache.heldBytes)
          .field("committed_bytes", cache.committedBytes)
          .field("revoked", cache.revokedCommitments);
    });
    w.field("index_runway_bytes", index.pagesFree * indexPageBytes).key("block_cache");
    if (const std::optional<model::KvBlockCacheStats> &hot = spill.blockCache) {
      w.object([&] {
        w.field("bytes", hot->bytes)
            .field("target_bytes", hot->targetBytes)
            .field("floor_bytes", hot->floorBytes)
            .field("slots_per_layer", hot->slotsPerLayer)
            .field("lanes", hot->lanes);
        // What the resolves of spilled K/V did (model::KvReadStats): the
        // entries of their block lists, those of Full pages, the blocks found
        // cached and those read, as the expert store counts its own
        // (writeStreamedExperts).
        const model::KvReadStats &reads = hot->reads;
        w.field("resolves", reads.resolves)
            .field("entries", reads.entries)
            .field("paged", reads.paged)
            .field("hits", reads.hits)
            .field("read", reads.misses)
            .field("hit_rate",
                   reads.hits + reads.misses ? double(reads.hits) / double(reads.hits + reads.misses) : 0.0)
            .field("ghost_hits", reads.ghostHits)
            .field("read_bytes", reads.readBytes)
            .field("read_failures", reads.readFailures)
            .key("resolve_ms");
        writeLatency(w, hot->resolve);
        w.key("read_ms");
        writeLatency(w, hot->read);
        w.key("host_steps");
        writeHostSteps(w, hot->pickup, hot->work, hot->resume);
      });
    } else {
      w.null();
    }
    w.key("history_stream");
    if (const std::optional<model::KvHistoryStreamStats> &stream = spill.historyStream) {
      w.object([&] {
        w.field("commands", stream->commands)
            .field("layers", stream->layers)
            .field("pages", stream->pages)
            .field("bytes", stream->bytes)
            .field("staging_bytes", stream->stagingBytes)
            .field("buffers", stream->buffers)
            .key("read_ms");
        writeLatency(w, stream->read);
        w.key("wait_ms");
        writeLatency(w, stream->wait);
      });
    } else {
      w.null();
    }
  });
}

void writeMemoryGovernor(json::Writer &w, const MemoryGovernorSnapshot &governor) {
  w.object([&] {
    w.field("limit_bytes", governor.limitBytes)
        .field("charged_bytes", governor.chargedBytes)
        .field("headroom_bytes", governor.headroomBytes)
        .field("growth_allowed", governor.hostGrowthAllowed)
        .field("denied_reservations", governor.deniedReservations)
        .field("system_pressure", memoryPressureName(governor.systemPressure))
        .field("host_measurement_valid", governor.hostMeasurementValid)
        .field("host_available_bytes", governor.hostAvailableBytes)
        .field("host_reserve_bytes", governor.hostReserveBytes)
        .field("host_spare_bytes", governor.hostSpareBytes)
        .field("host_spare_reserve_bytes", governor.hostSpareReserveBytes)
        .field("host_headroom_bytes", governor.hostHeadroomBytes)
        .key("host_telemetry");
    if (const std::optional<HostTelemetry> &host = governor.hostTelemetry) {
      w.object([&] {
        w.field("phys_footprint_bytes", host->physFootprintBytes)
            .field("compressed_bytes", host->compressedBytes)
            .field("file_backed_bytes", host->fileBackedBytes)
            .field("inactive_bytes", host->inactiveBytes)
            .field("filecache_min_bytes", host->fileCacheMinimumBytes);
      });
    } else {
      w.null();
    }
    // The memory ledger: each account's floor, current bytes and cap, its
    // unit, and the units it took and gave back; the bytes transfers will
    // return; and the claims that wait for memory.
    w.array("accounts", [&] {
      for (const MemoryAccountSnapshot &account : governor.accounts) {
        w.object([&] {
          w.field("name", account.name)
              .field("floor_bytes", account.floorBytes)
              .field("current_bytes", account.currentBytes)
              .field("cap_bytes", account.capBytes)
              .field("unit_bytes", account.unitBytes)
              .field("grows", account.grows)
              .field("shrinks", account.shrinks);
        });
      }
    });
    w.object("owed", [&] {
      w.field("transfers", governor.owed.transfers)
          .field("bytes", governor.owed.bytes)
          .field("to_claims_bytes", governor.owed.toClaimsBytes)
          .field("oldest_ms", governor.owed.oldestMilliseconds);
    });
    w.object("claims", [&] {
      w.field("waiting", governor.claims.waiting)
          .field("bytes", governor.claims.bytes)
          .field("oldest_ms", governor.claims.oldestMilliseconds);
    });
  });
}

void writeIdentity(json::Writer &w, const engine::RuntimeCacheIdentity &cacheIdentity) {
  const kv::Format kvFormat = cacheIdentity.kvLayout.format;
  w.object([&] {
    w.object("cache", [&] {
      w.field("loaded_model_layout_sha256", cacheIdentity.modelLayoutSha256)
          .field("build_id", cacheIdentity.buildId)
          .field("dtype", kv::storageFormatName(kvFormat, cacheIdentity.kvLayout.valuesTokenMajor))
          .field("block_tokens", kv::kPageTokens)
          .key("rope_scaling");
      writeRopeScaling(w, cacheIdentity.ropeScaling);
    });
    w.object("kv", [&] {
      w.field("target_model_sha256", cacheIdentity.targetModelSha256)
          .field("format", kv::formatName(kvFormat))
          .field("quantization", kvFormat == kv::Format::Int8 ? "symmetric_int8" : "none")
          .field("scale_type", kvFormat == kv::Format::Int8 ? "float32" : "none")
          .field("key_layout", "token_major")
          .field("value_layout", cacheIdentity.kvLayout.valuesTokenMajor ? "token_major" : "dimension_major");
    });
  });
}

void writeKv(json::Writer &w, const CacheSnapshot &resources) {
  const KvPoolSnapshot &pool = resources.pool;
  w.object([&] {
    w.field("block_tokens", kv::kPageTokens)
        .field("pages_allocated", pool.pagesAllocated)
        .field("pages_active", pool.pagesActive)
        .field("extents_active", pool.extentsActive)
        .field("active_bytes", pool.activeBytes)
        .field("pages_cache", pool.pagesPrefix)
        .field("pages_free", pool.pagesFree)
        .field("allocated_bytes", pool.allocatedBytes)
        .field("reclaimable_bytes", pool.reclaimableBytes)
        .field("extent_allocations", pool.extentAllocations)
        .field("extent_releases", pool.extentReleases)
        .field("extent_allocate_max_ms", pool.extentAllocateMaxMilliseconds)
        .field("extent_release_max_ms", pool.extentReleaseMaxMilliseconds)
        .field("extent_compactions", pool.extentCompactions)
        .field("pages_moved", pool.pagesMoved)
        .field("extent_compact_max_ms", resources.extentCompactMaxMilliseconds);
    if (resources.window) {
      w.key("window");
      writeKvWindow(w, *resources.window, pool);
    }
  });
}

void writeState(json::Writer &w, const CacheSnapshot &resources, const model::ModelTelemetry &telemetry) {
  const auto &state = resources.stateCache;
  w.object([&] {
    w.field("entries", state.entries)
        .field("pinned", state.pinned)
        .field("in_use", state.inUse)
        .field("in_use_evictions", state.inUseEvictions)
        .field("bytes", state.bytes)
        .field("allocated_bytes", telemetry.stateAllocatedBytes)
        .field("active_lanes", resources.activeRequests)
        .field("idle_gdn_cells", telemetry.idleGdnCells)
        .field("idle_draft_rings", telemetry.idleDraftRings)
        .field("idle_context_windows", telemetry.idleContextWindows)
        .field("publications", state.publications)
        .field("evictions", state.evictions)
        .field("checkpoint_entries", state.checkpointEntries)
        .field("checkpoint_bytes", state.checkpointBytes)
        .field("checkpoint_evictions", state.checkpointEvictions)
        .field("checkpoint_retirements", state.checkpointRetirements)
        .field("disk_hits", resources.lookup.stateDiskHits)
        .field("disk_promotions", state.promotions)
        .field("disk_promotions_skipped", state.promotionsSkipped)
        .field("disk_bytes", state.diskBytes)
        .field("offloads", state.offloads)
        .field("offload_failures", state.offloadFailures)
        .field("invalidations", state.invalidations);
  });
}

void writeDisk(json::Writer &w, const engine::EngineSnapshot &core) {
  const CacheSnapshot &resources = core.resources;
  const auto &tier = resources.kvTier;
  w.object([&] {
    w.field("capacity_bytes", tier.capacityBytes)
        .field("limit_bytes", tier.limitBytes)
        .field("used_bytes", tier.usedBytes)
        .field("file_bytes", tier.fileBytes)
        .field("volume_available_bytes", tier.volumeAvailableBytes)
        .field("writes_gated", writesGatedName(tier.writesGated))
        .field("write_budget_bytes", tier.writeBudgetBytes)
        .field("writes_deferred_ms", tier.writesDeferredMicroseconds / 1000.0)
        .field("read_bytes", tier.readBytes)
        .field("written_bytes", tier.writtenBytes)
        .field("kv_blocks", tier.diskBlocks)
        .field("kv_bytes", tier.diskBytes)
        .field("kv_demotions", tier.demotions)
        .field("kv_demotion_failures", tier.demotionFailures)
        .field("kv_demotions_refused", tier.demotionsRefused)
        .field("kv_restores", tier.restores)
        .field("kv_restore_failures", tier.restoreFailures)
        .field("kv_pending_pages", tier.pendingPages)
        .field("persistent", resources.persistent)
        .field("kv_copies", tier.copies)
        .field("kv_copy_failures", tier.copyFailures);
    w.object("taken_back", [&] {
      w.field("states", resources.adoption.states)
          .field("kv_blocks", resources.adoption.blocks)
          .field("bytes", resources.adoption.bytes)
          .field("left_behind", resources.adoption.dropped);
    });
    w.object("write_behind", [&] {
      w.field("waiting", core.writeBehind.waiting)
          .field("durable", core.writeBehind.durable)
          .field("unneeded", core.writeBehind.unneeded)
          .field("refused", core.writeBehind.refused);
    });
  });
}

void writeCache(json::Writer &w, const engine::EngineSnapshot &core) {
  const auto &lookup = core.resources.lookup;
  const double hitRate = core.cacheHits + core.coldMisses
                             ? double(core.cacheHits) / double(core.cacheHits + core.coldMisses)
                             : 0.0;
  w.object([&] {
    w.field("probe_hashed_blocks", lookup.probeHashedBlocks)
        .field("hits", core.cacheHits)
        .field("cold_misses", core.coldMisses)
        .field("hit_rate", hitRate)
        .field("kv_hit_tokens", lookup.kvHitTokens)
        .field("kv_disk_hit_tokens", core.resources.kvTier.restores * kv::kPageTokens)
        .field("lost_state_misses", lookup.lostStateMisses)
        .field("reused_tokens", core.reusedTokens)
        .field("deduplicated_kv_tokens", core.deduplicatedKvTokens)
        .field("replay_state_publications", core.replayStatePublications)
        .field("deduplicated_state_publications", core.deduplicatedStatePublications)
        .field("recycled_state_publications", core.recycledStatePublications)
        .field("disk_state_publications", core.diskStatePublications)
        .field("state_captures", core.stateCaptures)
        .field("capture_cuts", core.captureCuts)
        .field("replay_state_publication_failures", core.replayStatePublicationFailures)
        .field("lazy_junctions", lookup.lazyJunctions)
        .field("junction_materializations", core.junctionMaterializations)
        .field("junction_materialization_failures", core.junctionMaterializationFailures)
        .field("checkpoint_publications", core.checkpointPublications)
        .field("checkpoint_publication_failures", core.checkpointPublicationFailures)
        .field("resource_suspensions", core.resourceSuspensions)
        .field("priority_suspensions", core.prioritySuspensions)
        .field("resource_resumptions", core.resourceResumptions)
        .field("resource_replay_tokens", core.resourceReplayTokens);
  });
}

// Model-lifetime timings include warmup; request metrics do not.
void writeModelTiming(json::Writer &w, const model::ModelTelemetry &telemetry) {
  w.object([&] {
    w.field("scope", "model_lifetime");
    w.object("prefill", [&] {
      w.field("last_gpu_ms", telemetry.lastPrefillGpuSeconds * 1000.0)
          .field("last_wall_ms", telemetry.lastPrefillWallSeconds * 1000.0)
          .field("total_gpu_ms", telemetry.totalPrefillGpuSeconds * 1000.0)
          .field("total_wall_ms", telemetry.totalPrefillWallSeconds * 1000.0);
    });
    w.object("decode", [&] {
      w.field("last_gpu_ms", telemetry.lastDecodeGpuSeconds * 1000.0)
          .field("last_wall_ms", telemetry.lastDecodeWallSeconds * 1000.0)
          .field("total_gpu_ms", telemetry.totalDecodeGpuSeconds * 1000.0)
          .field("total_wall_ms", telemetry.totalDecodeWallSeconds * 1000.0);
    });
  });
}

void writeScheduler(json::Writer &w, const engine::EngineSnapshot &core) {
  const auto &scheduler = core.scheduler;
  w.object([&] {
    w.field("queued", scheduler.queued)
        .field("waiting_resources", scheduler.waitingResources)
        .field("waiting_prefix", scheduler.waitingPrefix)
        .field("prefilling", scheduler.prefilling)
        .field("decoding", scheduler.decoding)
        .field("waiting_mask", scheduler.waitingMask)
        .field("terminal", scheduler.terminal)
        .field("prefill_batches", scheduler.prefillBatches)
        .field("prefill_rows", scheduler.prefillRows)
        .field("decode_batches", scheduler.decodeBatches);
    w.object("decode_batches_by_width", [&] {
      for (uint32_t width = 1; width <= model::ExecutionLimits::maximumBatchWidth; ++width)
        w.key("b" + std::to_string(width)).value(scheduler.decodeBatchesByWidth[width - 1]);
    });
    w.field("decode_gap_target_ms", scheduler.decodeGapTargetMilliseconds).key("contended_prefill_ms");
    writeLatency(w, scheduler.contendedPrefill);
    w.field("prefill_units", scheduler.prefillUnits)
        .field("pass_commands", scheduler.passCommands)
        .field("abandoned_pass_commands", scheduler.abandonedPassCommands)
        .field("bound_cuts", scheduler.boundCuts)
        .field("index_page_conflicts", scheduler.indexPageConflicts)
        .field("replans", core.replans);
  });
}

// Where a target's KV may spill (kvModes), where each request's is kept.
void writeActiveRequests(json::Writer &w, const engine::EngineSnapshot &core, bool kvModes) {
  w.array([&] {
    for (const engine::ActiveRequestSnapshot &request : core.activeRequests) {
      w.object([&] {
        w.field("id", request.id)
            .field("phase", phaseName(request.phase))
            .field("priority", priorityName(request.priority))
            .field("prompt_tokens", request.promptTokens)
            .field("prompt_processed", request.promptProcessed)
            .field("generated_tokens", request.generatedTokens)
            .field("max_new_tokens", request.maxNewTokens);
        if (kvModes)
          w.field("kv_mode", request.kvMode == engine::KvMode::Spill ? "spill" : "resident");
        w.field("age_ms", request.ageMilliseconds);
      });
    }
  });
}

void writeMetrics(json::Writer &w, const RuntimeMetricsSnapshot &metrics, const engine::EngineSnapshot &core) {
  w.object([&] {
    w.object("ttft_ms", [&] {
      w.field("p50", metrics.ttftP50Milliseconds)
          .field("p95", metrics.ttftP95Milliseconds)
          .field("samples", metrics.ttftSamples);
    });
    w.object("ttft_parts_ms", [&] {
      for (size_t part = 0; part < kTtftParts; ++part) {
        w.key(kTtftPartNames[part]);
        writeLatency(w, metrics.ttftParts[part]);
      }
    });
    w.object("itl_ms", [&] {
      w.field("p50", metrics.itlP50Milliseconds)
          .field("p95", metrics.itlP95Milliseconds)
          .field("samples", metrics.itlSamples);
    });
    w.key("decode_gap_ms");
    writeLatency(w, metrics.decodeGap);
    w.object("decode_gap_causes", [&] {
      for (size_t cause = 0; cause < kGapCauses; ++cause) {
        w.object(kGapCauseNames[cause], [&] {
          w.field("gaps", core.decodeGapCauses.gaps[cause]).field("ms", core.decodeGapCauses.milliseconds[cause]);
        });
      }
      w.field("prefill_ms", core.decodeGapCauses.prefillMilliseconds);
    });
    w.field("prefill_input_tokens", metrics.prefillInputTokens)
        .field("prefill_wall_ms", metrics.prefillWallMilliseconds)
        .field("prefill_tokens_per_second", metrics.prefillTokensPerSecond)
        .field("decode_output_tokens", metrics.decodeOutputTokens)
        .field("decode_wall_ms", metrics.decodeWallMilliseconds)
        .field("decode_cycle_ms", metrics.decodeCycleMilliseconds)
        .field("decode_tokens_per_second", metrics.decodeTokensPerSecond)
        .field("drafted_tokens", metrics.draftedTokens)
        .field("accepted_draft_tokens", metrics.acceptedDraftTokens)
        .field("draft_acceptance_rate", metrics.draftAcceptanceRate)
        .field("capacity_failures", metrics.capacityFailures)
        .field("metal_failures", metrics.metalFailures)
        .key("current_prefill_batch");
    writeBatch(w, metrics.currentPrefillBatch);
    w.key("current_decode_batch");
    writeBatch(w, metrics.currentDecodeBatch);
  });
}

// Each warmup step, whether it ran, and the steps memory kept from running.
void writeWarmup(json::Writer &w, const WarmupReport &warmup) {
  std::vector<std::pair<std::string, WarmupStepStatus>> steps{
      {"prefill_" + std::to_string(model::ExecutionLimits::prefillTokenBudget), warmup.maximumPrefill}};
  for (uint32_t width = 1; width <= model::ExecutionLimits::maximumBatchWidth; ++width)
    steps.emplace_back("decode_b" + std::to_string(width), warmup.decodeBatches[width - 1]);
  steps.emplace_back("composite_state_restore", warmup.compositeStateRestore);
  w.object([&] {
    for (const auto &[name, status] : steps)
      w.key(name).value(status == WarmupStepStatus::Complete);
    w.array("memory_limited_steps", [&] {
      for (const auto &[name, status] : steps)
        if (status == WarmupStepStatus::MemoryLimited)
          w.value(name);
    });
    w.field("detail", warmup.maximumPrefillDetail);
  });
}

} // namespace

std::string runtimeStatusJson(const StatusStart &start, const StatusSample &sample) {
  const EngineMemoryPlan &plan = start.plan;
  const EngineSnapshot &core = sample.engine;
  const metal::MetalMemoryStats &metalMemory = sample.metalMemory;
  const MemoryGovernorSnapshot &memoryGovernor = sample.memoryGovernor;
  const ResourceWaitSnapshot &resourceWait = sample.resourceWait;
  const model::ModelTelemetry &executorTelemetry = sample.telemetry;
  const uint64_t currentBytes = std::max(
      metalMemory.allocatedBytes, metalMemory.deviceCurrentAllocatedBytes);
  const uint64_t peakBytes = std::max(
      {currentBytes, metalMemory.peakAllocatedBytes,
       metalMemory.devicePeakAllocatedBytes});
  // Warning pressure sheds cache but permits serving; only the governor's
  // critical verdict makes host pressure a readiness failure. A status exists
  // only after warmup and the memory audit passed. The thermal state slows
  // serving but never stops it, so it is reported and readiness ignores it.
  const bool hostSafe = memoryGovernor.pressure != MemoryPressure::Critical;
  const bool ready = sample.metalHealthy && hostSafe &&
                     currentBytes <= plan.breakdown().hardBudgetBytes;

  json::Writer w;
  w.object([&] {
    w.field("schema_version", protocol::kStatusSchemaVersion)
        .field("ready", ready)
        .field("maximum_context_tokens", core.maximumContextTokens)
        .field("memory_pressure", memoryPressureName(memoryGovernor.pressure))
        .field("thermal_state", thermalStateName(sample.thermalState));
    w.object("admission", [&] {
      w.field("waiting", resourceWait.memory + resourceWait.concurrency)
          .field("waiting_memory", resourceWait.memory)
          .field("waiting_concurrency", resourceWait.concurrency)
          .field("held_behind_refusal", resourceWait.heldBehindRefusal)
          .field("restoring", resourceWait.restoring)
          .field("suspended", resourceWait.suspended)
          .field("draining", resourceWait.draining)
          .field("oldest_wait_ms", resourceWait.oldestWaitMilliseconds);
    });
    w.object("loop", [&] {
      w.field("max_tick_ms", sample.loopMaxTickMilliseconds)
          .field("stall_budget_ms", supervision::kLoopStall.count())
          .key("tick_host_ms");
      writeLatency(w, core.tickHostMilliseconds);
    });
    w.key("identity");
    writeIdentity(w, start.cacheIdentity);
    w.key("memory_plan");
    writeJson(w, plan);
    w.object("memory_actual", [&] {
      w.field("allocated_bytes", metalMemory.allocatedBytes)
          .field("current_bytes", currentBytes)
          .field("peak_bytes", peakBytes);
    });
    w.key("memory_governor");
    writeMemoryGovernor(w, memoryGovernor);
    w.key("memory_audit");
    writeJson(w, start.memoryAudit);
    w.key("weights");
    writeWeights(w, sample.weights);
    w.key("streamed_table");
    writeStreamedTable(w, sample.streamedTable);
    w.key("streamed_experts");
    writeStreamedExperts(w, sample.streamedExperts);
    if (sample.kvSpill) {
      w.key("kv_spill");
      writeKvSpill(w, *sample.kvSpill, plan.breakdown().kvIndexPageBytes, core, resourceWait);
    }
    w.key("ane_ffn");
    writeAneFfn(w, sample.aneFfn, executorTelemetry.aneFfnReruns);
    w.key("kv");
    writeKv(w, core.resources);
    w.key("state");
    writeState(w, core.resources, executorTelemetry);
    w.key("disk");
    writeDisk(w, core);
    w.key("cache");
    writeCache(w, core);
    w.object("draft_context", [&] {
      w.field("target_prefill_rows", executorTelemetry.targetPrefillRows)
          .field("prompt_end_rows", executorTelemetry.draftContextRowsActive)
          .field("materialization_rows", executorTelemetry.draftContextRowsMaterialization)
          .field("avoided_rows", executorTelemetry.draftContextRowsAvoided)
          .field("restore_skipped", executorTelemetry.draftStateRestoreSkipped)
          .field("resets", executorTelemetry.draftStateResets);
    });
    // The rows decode cycles verify (--verify-rows): auto where the verify
    // budget chooses them, all where they are every row up to each lane's
    // retention limit.
    w.field("verify_rows", executorTelemetry.verifyBudget ? "auto" : "all");
    w.key("model_timing");
    writeModelTiming(w, executorTelemetry);
    w.object("constraint_masks", [&] {
      w.field("overlap_batches", executorTelemetry.constrainedMaskOverlapBatches)
          .field("overlap_requests", executorTelemetry.constrainedMaskOverlapRequests)
          .field("last_target_forward_gpu_ms", executorTelemetry.lastConstrainedTargetForwardGpuSeconds * 1000.0)
          .field("total_target_forward_gpu_ms", executorTelemetry.totalConstrainedTargetForwardGpuSeconds * 1000.0)
          .field("last_residual_wait_ms", executorTelemetry.lastConstrainedMaskWaitSeconds * 1000.0)
          .field("total_residual_wait_ms", executorTelemetry.totalConstrainedMaskWaitSeconds * 1000.0);
    });
    w.object("images", [&] {
      w.field("encodes", executorTelemetry.imageEncodes)
          .field("embedding_reuses", executorTelemetry.imageEmbeddingReuses)
          .field("embedding_evictions", executorTelemetry.imageEmbeddingEvictions)
          .field("arena_bytes", executorTelemetry.visionArenaBytes)
          .field("cached_bytes", executorTelemetry.embeddingCacheBytes)
          .field("state_held_bytes", executorTelemetry.stateHeldImageBytes)
          .field("rows_bytes", executorTelemetry.imageRowsBytes);
    });
    w.key("scheduler");
    writeScheduler(w, core);
    w.key("active_requests");
    writeActiveRequests(w, core, sample.kvSpill.has_value());
    w.object("requests", [&] {
      w.field("submitted", core.submitted)
          .field("completed", core.completed)
          .field("cancelled", core.cancelled)
          .field("failed", core.failed);
    });
    w.key("metrics");
    writeMetrics(w, sample.metrics, core);
    w.key("warmup");
    writeWarmup(w, start.warmup);
    w.object("metal", [&] {
      w.field("healthy", sample.metalHealthy).field("failure_reason", sample.metalFailureReason);
    });
  });
  return w.str();
}

} // namespace splash::engine
