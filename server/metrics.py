"""Usage, metrics and Prometheus text rendering for request results."""

from __future__ import annotations

import math

if __package__:
    from .latency import prometheus_latency
else:
    from latency import prometheus_latency


def is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def prometheus_metrics(status):
    """Render low-cardinality metrics directly from the native status."""

    def value(path):
        current = status
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        if isinstance(current, bool):
            return 1 if current else 0
        if is_finite_number(current):
            return current
        return None

    metrics = {
        "splash_ready": ("ready",),
        "splash_metal_healthy": ("metal", "healthy"),
        "splash_transport_pending": ("transport", "pending"),
        "splash_frontend_active": ("frontend", "active"),
        "splash_frontend_waiting": ("frontend", "waiting"),
        "splash_requests_submitted_total": ("requests", "submitted"),
        "splash_requests_completed_total": ("requests", "completed"),
        "splash_requests_cancelled_total": ("requests", "cancelled"),
        "splash_requests_failed_total": ("requests", "failed"),
        "splash_scheduler_queued": ("scheduler", "queued"),
        "splash_scheduler_waiting_resources": ("scheduler", "waiting_resources"),
        "splash_scheduler_waiting_prefix": ("scheduler", "waiting_prefix"),
        "splash_cache_resource_suspensions_total": ("cache", "resource_suspensions"),
        "splash_cache_resource_resumptions_total": ("cache", "resource_resumptions"),
        "splash_cache_resource_replay_tokens_total": (
            "cache",
            "resource_replay_tokens",
        ),
        "splash_scheduler_prefilling": ("scheduler", "prefilling"),
        "splash_scheduler_decoding": ("scheduler", "decoding"),
        "splash_scheduler_waiting_mask": ("scheduler", "waiting_mask"),
        "splash_scheduler_prefill_batches_total": ("scheduler", "prefill_batches"),
        "splash_scheduler_prefill_rows_total": ("scheduler", "prefill_rows"),
        "splash_scheduler_decode_batches_total": ("scheduler", "decode_batches"),
        "splash_scheduler_decode_b1_total": (
            "scheduler",
            "decode_batches_by_width",
            "b1",
        ),
        "splash_scheduler_decode_b2_total": (
            "scheduler",
            "decode_batches_by_width",
            "b2",
        ),
        "splash_scheduler_decode_b3_total": (
            "scheduler",
            "decode_batches_by_width",
            "b3",
        ),
        "splash_scheduler_decode_b4_total": (
            "scheduler",
            "decode_batches_by_width",
            "b4",
        ),
        "splash_kv_blocks": ("kv", "blocks"),
        "splash_kv_pages_total": ("kv", "pages_total"),
        "splash_kv_pages_free": ("kv", "pages_free"),
        "splash_kv_pages_active": ("kv", "pages_active"),
        "splash_kv_pages_cache": ("kv", "pages_cache"),
        "splash_kv_pages_resident": ("kv", "pages_resident"),
        "splash_kv_pages_free_resident": ("kv", "pages_free_resident"),
        "splash_kv_resident_backing_bytes": ("kv", "resident_backing_bytes"),
        "splash_kv_reclaimable_backing_bytes": ("kv", "reclaimable_backing_bytes"),
        "splash_kv_sparse_tile_bytes": ("kv", "sparse_tile_bytes"),
        "splash_kv_pending_unmaps": ("kv", "pending_unmaps"),
        "splash_kv_pending_unmap_ms": ("kv", "pending_unmap_ms"),
        "splash_kv_unmaps_completed": ("kv", "unmaps_completed"),
        "splash_kv_unmap_last_ms": ("kv", "unmap_last_ms"),
        "splash_kv_unmap_max_ms": ("kv", "unmap_max_ms"),
        "splash_kv_map_wait_event": ("kv", "map_wait_event"),
        "splash_kv_pending_map_wait_ms": ("kv", "pending_map_wait_ms"),
        "splash_kv_map_wait_last_ms": ("kv", "map_wait_last_ms"),
        "splash_kv_map_wait_max_ms": ("kv", "map_wait_max_ms"),
        "splash_state_entries": ("state", "entries"),
        "splash_state_pinned": ("state", "pinned"),
        "splash_state_bytes": ("state", "bytes"),
        "splash_state_active_cells": ("state", "active_cells"),
        "splash_state_hits_total": ("state", "hits"),
        "splash_state_misses_total": ("state", "misses"),
        "splash_state_publications_total": ("state", "publications"),
        "splash_state_evictions_total": ("state", "evictions"),
        "splash_cache_hits_total": ("cache", "hits"),
        "splash_cache_cold_misses_total": ("cache", "cold_misses"),
        "splash_cache_reused_tokens_total": ("cache", "reused_tokens"),
        "splash_cache_lazy_junctions_total": ("cache", "lazy_junctions"),
        "splash_target_prefill_rows_total": (
            "draft_context",
            "target_prefill_rows",
        ),
        "splash_draft_context_active_rows_total": (
            "draft_context",
            "active_rows",
        ),
        "splash_draft_context_materialization_rows_total": (
            "draft_context",
            "materialization_rows",
        ),
        "splash_draft_context_avoided_rows_total": (
            "draft_context",
            "avoided_rows",
        ),
        "splash_draft_state_restore_skipped_total": (
            "draft_context",
            "restore_skipped",
        ),
        "splash_draft_state_resets_total": ("draft_context", "resets"),
        "splash_constraint_mask_overlap_batches_total": (
            "constraint_masks",
            "overlap_batches",
        ),
        "splash_constraint_mask_overlap_requests_total": (
            "constraint_masks",
            "overlap_requests",
        ),
        "splash_constraint_mask_target_forward_gpu_milliseconds": (
            "constraint_masks",
            "last_target_forward_gpu_ms",
        ),
        "splash_constraint_mask_residual_wait_milliseconds": (
            "constraint_masks",
            "last_residual_wait_ms",
        ),
        "splash_image_encodes_total": ("images", "encodes"),
        "splash_image_embedding_reuses_total": ("images", "embedding_reuses"),
        "splash_memory_current_bytes": ("memory_actual", "current_bytes"),
        "splash_memory_peak_bytes": ("memory_actual", "peak_bytes"),
        "splash_memory_denied_reservations_total": (
            "memory_governor",
            "denied_reservations",
        ),
        "splash_memory_limit_bytes": ("memory_governor", "limit_bytes"),
        "splash_memory_headroom_bytes": ("memory_governor", "headroom_bytes"),
        "splash_admission_waiting_memory": ("admission", "waiting_memory"),
        "splash_admission_waiting_concurrency": ("admission", "waiting_concurrency"),
        "splash_admission_suspended": ("admission", "suspended"),
        "splash_admission_oldest_wait_milliseconds": ("admission", "oldest_wait_ms"),
        "splash_ttft_p50_milliseconds": ("metrics", "ttft_ms", "p50"),
        "splash_ttft_p95_milliseconds": ("metrics", "ttft_ms", "p95"),
        "splash_itl_p50_milliseconds": ("metrics", "itl_ms", "p50"),
        "splash_itl_p95_milliseconds": ("metrics", "itl_ms", "p95"),
        "splash_prefill_input_tokens_total": ("metrics", "prefill_input_tokens"),
        "splash_prefill_wall_milliseconds_total": ("metrics", "prefill_wall_ms"),
        "splash_prefill_tokens_per_second": (
            "metrics",
            "prefill_tokens_per_second",
        ),
        "splash_decode_output_tokens_total": ("metrics", "decode_output_tokens"),
        "splash_decode_wall_milliseconds_total": ("metrics", "decode_wall_ms"),
        "splash_decode_tokens_per_second": (
            "metrics",
            "decode_tokens_per_second",
        ),
        "splash_drafted_tokens_total": ("metrics", "drafted_tokens"),
        "splash_accepted_draft_tokens_total": ("metrics", "accepted_draft_tokens"),
        "splash_draft_acceptance_ratio": ("metrics", "draft_acceptance_rate"),
        "splash_capacity_failures_total": ("metrics", "capacity_failures"),
        "splash_metal_failures_total": ("metrics", "metal_failures"),
        "splash_response_store_entries": ("response_store", "entries"),
        "splash_response_store_bytes": ("response_store", "bytes"),
    }
    lines = [
        "# HELP splash_info Splash runtime metrics.",
        "# TYPE splash_info gauge",
        'splash_info{runtime="native"} 1',
    ]
    pressure = status.get("memory_pressure")
    for state in ("normal", "warning", "critical"):
        lines.append(
            f'splash_memory_pressure{{state="{state}"}} {1 if pressure == state else 0}'
        )
    for name, path in metrics.items():
        metric_value = value(path)
        if metric_value is not None:
            lines.append(f"{name} {metric_value}")
    lines.extend(prometheus_latency(status.get("latency", {})))
    return "\n".join(lines) + "\n"


def usage_dict(result, job):
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": result.cache.matched_tokens},
        "completion_tokens_details": {"reasoning_tokens": job.reasoning_tokens},
    }


def metrics_dict(result):
    # The first native emission can contain a whole speculative block. Its
    # generation precedes TTFT, so none of those tokens belong in the interval
    # from first emission to Done. Native command throughput lives in /status.
    decode_tokens = max(0, result.completion_tokens - result.first_token_batch_tokens)
    latency = {}
    intervals = (
        ("start_to_first_token_ms", result.start_to_first_token_ms),
        ("first_token_to_done_ms", result.first_token_to_done_ms),
        ("wall_ms", result.request_wall_ms),
    )
    for name, milliseconds in intervals:
        if math.isfinite(milliseconds) and milliseconds >= 0:
            latency[name] = milliseconds
    if (
        result.completion_tokens > 0
        and "wall_ms" in latency
        and "first_token_to_done_ms" in latency
        and latency["wall_ms"] >= latency["first_token_to_done_ms"]
    ):
        latency["ttft_ms"] = latency["wall_ms"] - latency["first_token_to_done_ms"]
        if (
            "start_to_first_token_ms" in latency
            and latency["ttft_ms"] >= latency["start_to_first_token_ms"]
        ):
            latency["queue_to_start_ms"] = (
                latency["ttft_ms"] - latency["start_to_first_token_ms"]
            )
    if (
        result.first_token_batch_tokens > 0
        and decode_tokens > 0
        and latency.get("first_token_to_done_ms", 0) > 0
    ):
        rate = decode_tokens * 1000.0 / latency["first_token_to_done_ms"]
        if math.isfinite(rate):
            latency["stream_tokens_per_second"] = rate
    cache_info = result.cache
    cache = {
        "status": cache_info.status,
        "matched_tokens": cache_info.matched_tokens,
        "capacity": cache_info.capacity,
        "slot": cache_info.slot if cache_info.slot >= 0 else None,
    }
    metrics = {
        "prefill": {"tokens": result.prefill_tokens},
        "decode": {"tokens": decode_tokens},
        # Native DoneEvent intervals are request lifecycle timing, not summed
        # executor command/GPU time. Native batch throughput lives in /status.
        "request_latency": latency,
        "cache": cache,
    }
    return metrics
