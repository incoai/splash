#!/usr/bin/env python3
"""Read completed reports only; independent optional SSD PLE service audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_flash_v8_qsa_service import compare, idle, inputs, summary

SSD_ROUTE = "ple-ssd-original-q4g32-bounded-staging-gpu-id-checked-v1"
RAW_ROUTE = "ple-exact-i64-hash-direct128-q4g32-tier2"
IO_FIELDS = (
    "prepared_batches",
    "requested_rows",
    "unique_miss_rows",
    "cache_hit_rows",
    "duplicate_miss_rows",
    "read_requests",
    "requested_read_bytes",
    "completed_read_bytes",
    "cache_evictions",
    "failed_batches",
    "source_validation_failures",
    "host_read_ms",
)
STAGE_FIELDS = (
    "singleton_staging_bytes",
    "batch_decode_staging_bytes",
    "joint_verify_staging_bytes",
    "batch_prefill_staging_bytes",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def semantic_record(record: dict) -> dict:
    """Retain output/protocol facts; generated IDs and timing are not semantic."""
    fields = (
        "text",
        "reasoning_text",
        "finish_reason",
        "usage",
        "http_status",
        "errors",
        "error_frames",
        "done",
    )
    return {
        **{field: record.get(field) for field in fields},
        "tool_calls": [
            {"type": call.get("type"), "function": call.get("function")}
            for call in record.get("tool_calls", [])
        ],
    }


def quality_records(report: dict) -> dict:
    return {
        case["id"]: [semantic_record(record) for record in case["records"]]
        for case in report["cases"]
    }


def generation_records(report: dict) -> list:
    return [
        semantic_record(record)
        for section in ("warmups", "waves")
        for wave in report[section]
        for record in wave["records"]
    ]


def io_delta(before: dict, after: dict) -> dict:
    result = {key: after[key] - before[key] for key in IO_FIELDS}
    rows = result["requested_rows"]
    result["cache_hit_percent_of_requested_rows"] = (
        100 * result["cache_hit_rows"] / rows if rows else None
    )
    result["row_accounting_exact"] = rows == sum(
        result[key]
        for key in ("unique_miss_rows", "cache_hit_rows", "duplicate_miss_rows")
    )
    result["logical_unique_miss_bytes"] = 100 * result["unique_miss_rows"]
    result["completed_bytes_equal_exact_requested_rows"] = (
        result["completed_read_bytes"]
        == result["logical_unique_miss_bytes"]
        == result["requested_read_bytes"]
    )
    result["mean_host_read_ms_per_pread"] = (
        result["host_read_ms"] / result["read_requests"]
        if result["read_requests"]
        else None
    )
    return result


def storage(status: dict) -> dict:
    result = status["ple_storage"].copy()
    result["total_executor_gpu_staging_bytes"] = sum(
        result[key] for key in STAGE_FIELDS
    )
    result["cpu_cache_plus_read_scratch_bytes"] = (
        result["cache_accounted_bytes"] + result["read_scratch_limit_bytes"]
    )
    result["native_original_partition_exact"] = (
        result["gpu_mapped_original_bytes"] + result["disk_only_payload_bytes"]
        == result["original_payload_bytes"]
    )
    result["native_weight_ledger_matches_retained_windows"] = (
        status["memory_audit"]["weight_bytes"] == result["gpu_mapped_original_bytes"]
    )
    result["host_cache_within_configured_budget"] = (
        result["cache_accounted_bytes"] <= result["cache_budget_bytes"]
    )
    result["memory_scope"] = (
        "Native allocation ledger excludes CPU row cache, bounded CPU I/O scratch, "
        "temporary batch metadata and unmeasured OS/file/device cache. Native original "
        "weight omission does not imply identical process-RSS or physical-wired savings."
    )
    return result


def cache_disabled(report: dict) -> bool:
    statuses = [report[key] for key in ("runtime_before", "runtime_after")]
    return all(
        status["cache"]["enabled"] is False
        and status["cache"]["hits"] == 0
        and status["cache"]["reused_tokens"] == 0
        for status in statuses
    )


def per_wave_io(report: dict) -> list:
    return [
        {
            "sample": wave["sample"],
            "context": wave["context"],
            "width": wave["width"],
            **io_delta(
                wave["runtime_before"]["ple_storage"],
                wave["runtime_after"]["ple_storage"],
            ),
            "before_idle": idle(wave["runtime_before"]),
            "after_idle": idle(wave["runtime_after"]),
            "snapshot_scope": wave["runtime_after"].get("status_snapshot"),
        }
        for wave in report["waves"]
        if "runtime_before" in wave and "runtime_after" in wave
    ]


def idle_proof(report: dict) -> dict:
    before = report["runtime_before"]["idle_residency_maintenance"]
    after = report["runtime_after"]["idle_residency_maintenance"]
    return {
        "valid": report["valid"],
        "vm_scope": report["vm_scope"],
        "maintenance_commands_completed_during_probe": after["completed_commands"]
        - before["completed_commands"],
        "unchanged_owner_count_and_backing": all(
            before[key] == after[key]
            for key in ("immutable_owner_count", "immutable_owner_bytes")
        ),
        "failures_added": after["maintenance_failures"]
        - before["maintenance_failures"],
        "requests": [
            {
                "label": row["label"],
                "idle_seconds": row["idle_seconds"],
                "valid": row["valid"],
                "client_first_content_ms": row["record"]["first_content_ms"],
                "native_request_latency": row["record"]["metrics"]["request_latency"],
                "usage": row["record"]["usage"],
                "idle_snapshots": [
                    {
                        "elapsed": snapshot["elapsed"],
                        "systemwide_wired_bytes": snapshot["wired_bytes"],
                        "native_idle": idle(snapshot["runtime"]),
                        "immutable_owner_count": snapshot["runtime"][
                            "idle_residency_maintenance"
                        ]["immutable_owner_count"],
                        "immutable_owner_bytes": snapshot["runtime"][
                            "idle_residency_maintenance"
                        ]["immutable_owner_bytes"],
                        "maintenance_commands": snapshot["runtime"][
                            "idle_residency_maintenance"
                        ]["completed_commands"],
                    }
                    for snapshot in row["snapshots"]
                ],
            }
            for row in report["records"]
        ],
        "scope": "One SSD-mode 9-second idle screen after model and row-cache warmup; no claim that systemwide wired memory is all this model or that physical pinning is guaranteed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("build/release/flash"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/release/flash/ple-ssd-independent-service-audit-v12.json"),
    )
    args = parser.parse_args()
    folder = args.report_dir
    names = {
        "accepted_quality": "v11-production-maintenance-on-quality.json",
        "ssd_initial": "ple-ssd-initial-status-v12.json",
        "ssd_quality": "ple-ssd-quality-v12.json",
        "ssd_quality_comparison": "ple-ssd-quality-comparison-v12.json",
        "ssd_http": "ple-ssd-ssd-http-performance-v12.json",
        "ssd_idle": "ple-ssd-ssd-final-idle-status-v12.json",
        "ssd_idle_proof": "ple-ssd-idle-maintenance-proof-v12.json",
        "raw_http": "ple-ssd-raw-http-performance-v12.json",
        "raw_idle": "ple-ssd-raw-final-idle-status-v12.json",
        "raw_quality": "ple-ssd-raw-quality-v12.json",
        "raw_quality_comparison": "ple-ssd-raw-quality-comparison-v12.json",
    }
    missing = [name for name in names.values() if not (folder / name).exists()]
    if missing:
        raise SystemExit(
            "Completed evidence required before auditing: " + ", ".join(missing)
        )
    reports = {key: read(folder / name) for key, name in names.items()}
    ssd, raw = reports["ssd_http"], reports["raw_http"]
    quality, baseline = reports["ssd_quality"], reports["accepted_quality"]
    left, right = quality_records(baseline), quality_records(quality)
    pair = compare(raw, ssd)
    # Storage changes the PLE route marker; the rest of the kernels must remain identical.
    route_fields = ("kernel_routes", "batch_prefill_kernel_routes")
    unchanged_routes = {
        key: raw["runtime_after"]["identity"][key]
        == ssd["runtime_after"]["identity"][key].replace(SSD_ROUTE, RAW_ROUTE)
        for key in route_fields
    }
    source_fields = (
        "source",
        "loaded_model_layout_sha256",
        "forward_semantics",
        "worker_semantics",
        "mtp_semantics",
        "mtp_attention_route",
        "joint_head_semantics",
        "joint_head_attention_route",
        "joint_head_vocabulary_route",
        "batch_mtp_prefill_attention_route",
        "weight_format",
    )
    source_equal = {
        key: raw["runtime_after"]["identity"][key]
        == ssd["runtime_after"]["identity"][key]
        for key in source_fields
    }
    per_case_io = [
        {
            "id": case["id"],
            **io_delta(
                case["status_before"]["ple_storage"],
                case["status_after"]["ple_storage"],
            ),
        }
        for case in quality["cases"]
    ]
    before_io, after_io = (
        ssd[key]["ple_storage"] for key in ("runtime_before", "runtime_after")
    )
    raw_memory, ssd_memory = (
        report["runtime_after"]["memory_audit"] for report in (raw, ssd)
    )
    unchanged_allocations = (
        "state_bytes_per_request",
        "mtp_workspace_bytes",
        "mtp_extra_state_bytes_per_eligible_request",
        "joint_head_workspace_bytes",
        "joint_owned_hidden_input_bytes",
        "batch_prefill_owned_hidden_input_bytes",
        "batch_mtp_prefill_workspace_bytes",
        "batch_mtp_prefill_owned_hidden_input_bytes",
    )
    stage_allocation_names = {
        "workspace_bytes": "singleton_staging_bytes",
        "batch_workspace_bytes": "batch_decode_staging_bytes",
        "joint_verifier_workspace_bytes": "joint_verify_staging_bytes",
        "batch_prefill_workspace_bytes": "batch_prefill_staging_bytes",
    }
    # Singleton SSD lookup replaces the original reflected readonly pointer table.
    # Its byte count is 384 pointers; only singleton owns that persistent table.
    raw_singleton_argument_bytes = 384 * 8
    allocation_delta = {
        key: {
            "candidate_minus_raw_bytes": ssd_memory[key] - raw_memory[key],
            "added_gpu_row_staging_bytes": after_io[stage],
            "removed_original_fused_argument_bytes": raw_singleton_argument_bytes
            if key == "workspace_bytes"
            else 0,
            "matches_staging_minus_removed_argument_bytes": ssd_memory[key]
            - raw_memory[key]
            == after_io[stage]
            - (raw_singleton_argument_bytes if key == "workspace_bytes" else 0),
        }
        for key, stage in stage_allocation_names.items()
    }
    storage_omission = raw_memory["weight_bytes"] - ssd_memory["weight_bytes"]
    real_native_dense_saving = (
        raw["runtime_after"]["memory_actual"]["dense_bytes"]
        - ssd["runtime_after"]["memory_actual"]["dense_bytes"]
    )
    diagnostic_delta = (
        ssd["runtime_after"]["idle_residency_maintenance"][
            "added_diagnostic_allocation_bytes"
        ]
        - raw["runtime_after"]["idle_residency_maintenance"][
            "added_diagnostic_allocation_bytes"
        ]
    )
    peak_saving = (
        raw["runtime_after"]["memory_actual"]["peak_bytes"]
        - ssd["runtime_after"]["memory_actual"]["peak_bytes"]
    )
    benchmark_sources = generation_records(raw), generation_records(ssd)
    all_records = [
        record
        for section in ("warmups", "waves")
        for wave in ssd[section]
        for record in wave["records"]
    ]
    manual_record = right["long_factual_prose"]
    result = {
        "schema": "splash-flash-optional-ple-ssd-independent-service-audit-v1",
        "gpu_commands": 0,
        "inputs": [
            {
                "name": name,
                "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest(),
            }
            for name in names.values()
        ],
        "quality": {
            "valid": quality["valid"],
            "passed": quality["passed_cases"],
            "failed": quality["failed_cases"],
            "skipped": quality["skipped_cases"],
            "full_plan_coverage": quality["full_plan_coverage"],
            "same_frozen_quality_plan": baseline["plan_content_sha256"]
            == quality["plan_content_sha256"],
            "new_task_regressions": reports["ssd_quality_comparison"][
                "new_task_regressions"
            ],
            "all_28_semantic_records_equal": left == right
            and sum(map(len, right.values())) == 28,
            "differing_case_ids": [
                key
                for key in left.keys() | right.keys()
                if left.get(key) != right.get(key)
            ],
            "manual_factual_prose": manual_record,
            "manual_factual_prose_unchanged": left["long_factual_prose"]
            == manual_record,
            "per_case_ssd_io": per_case_io,
            "final_idle": idle(quality["final_status"]),
        },
        "raw_default_regression_guard": {
            "valid": reports["raw_quality"]["valid"],
            "passed": reports["raw_quality"]["passed_cases"],
            "failed": reports["raw_quality"]["failed_cases"],
            "full_plan_coverage": reports["raw_quality"]["full_plan_coverage"],
            "all_28_semantic_records_equal": quality_records(reports["raw_quality"])
            == left,
            "new_task_regressions": reports["raw_quality_comparison"][
                "new_task_regressions"
            ],
            "final_idle": idle(reports["raw_quality"]["final_status"]),
        },
        "benchmark": {
            "same_frozen_plan": raw["plan_sha256"] == ssd["plan_sha256"],
            "raw_valid": raw["valid"],
            "ssd_valid": ssd["valid"],
            "all_21_inputs_equal": len(inputs(raw)) == len(inputs(ssd)) == 21
            and inputs(raw) == inputs(ssd),
            "all_21_semantic_outputs_equal": len(benchmark_sources[0])
            == len(benchmark_sources[1])
            == 21
            and benchmark_sources[0] == benchmark_sources[1],
            "acceptance_and_verifier_work_equal": pair[
                "acceptance_and_verifier_work_equal"
            ],
            "source_and_math_identity_equal": source_equal,
            "non_storage_kernel_routes_equal": unchanged_routes,
            "persisted_operands_equal": pair["persisted_operand_identity_equal"],
            "persisted_experts_equal": pair["persisted_expert_identity_equal"],
            "raw_cross_request_kv_cache_disabled": cache_disabled(raw),
            "ssd_cross_request_kv_cache_disabled": cache_disabled(ssd),
            "all_21_zero_cached_prompt_tokens": len(all_records) == 21
            and all(
                record["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
                for record in all_records
            ),
            "raw_trace_disabled": raw["runtime_after"]["request_command_trace"][
                "enabled"
            ]
            is False,
            "ssd_trace_disabled": ssd["runtime_after"]["request_command_trace"][
                "enabled"
            ]
            is False,
            "cells": pair["cells"],
            "phase_change": pair["phase_change"],
            "raw": summary(raw),
            "ssd": summary(ssd),
            "total_ssd_io": io_delta(before_io, after_io),
            "per_wave_ssd_io_available": all(
                "runtime_before" in wave and "runtime_after" in wave
                for wave in ssd["waves"]
            ),
            "per_wave_ssd_io": per_wave_io(ssd),
            "per_wave_ssd_io_scope": "Native safe-point snapshots surround the measured HTTP wave, outside its clock. A post-wave snapshot may precede final bookkeeping; retained final idle snapshot is the lifecycle evidence. Warmup has no per-wave snapshots and is included only in aggregate I/O.",
        },
        "storage": {
            "initial_ssd": storage(reports["ssd_initial"]),
            "final_ssd": storage(reports["ssd_idle"]),
            "final_raw": storage(reports["raw_idle"]),
            "original_weight_backing_omitted_bytes": storage_omission,
            "original_weight_backing_omitted_gib": storage_omission / 2**30,
            "original_weight_omission_matches_declared_disk_only": storage_omission
            == after_io["disk_only_payload_bytes"],
            "executor_staging_allocation_delta": allocation_delta,
            "unchanged_other_workspace_and_per_request_state": {
                key: raw_memory[key] == ssd_memory[key] for key in unchanged_allocations
            },
            "net_native_dense_saving_bytes": real_native_dense_saving,
            "net_native_dense_saving_gib": real_native_dense_saving / 2**30,
            "raw_singleton_fused_argument_owner_replaced_bytes": raw_singleton_argument_bytes,
            "maintenance_diagnostic_delta_bytes": diagnostic_delta,
            "net_native_dense_saving_matches_table_minus_staging_plus_removed_argument_minus_diagnostic_delta": real_native_dense_saving
            == storage_omission
            - sum(after_io[key] for key in STAGE_FIELDS)
            + raw_singleton_argument_bytes
            - diagnostic_delta,
            "rounded_native_peak_saving_bytes": peak_saving,
            "rounded_native_peak_saving_gib": peak_saving / 2**30,
            "dense_vs_rounded_peak_saving_difference_bytes": real_native_dense_saving
            - peak_saving,
            "rounding_scope": "Dense bytes are logical allocation extents; current/peak charge rounded native backing. The replaced 3,072-byte PLE argument owner and +56-byte reflected maintenance layout produce a 3,016-byte difference in logical versus rounded savings.",
        },
        "idle": {key: idle(reports[key]) for key in ("ssd_idle", "raw_idle")},
        "maintenance": {
            key: reports[key]["idle_residency_maintenance"]
            for key in ("ssd_idle", "raw_idle")
        },
        "ssd_idle_proof": idle_proof(reports["ssd_idle_proof"]),
        "scope": {
            "optional": "SSD storage mode remains optional; accepted in-memory default profile is unchanged.",
            "ssd_cache": "Shared bounded original-row CPU cache is independent of KV/prefix cache and can benefit later requests.",
            "io": "pread payload bytes and synchronous host read scope; not physical SSD traffic, bandwidth or every staging CPU cost.",
            "gpu_math": "Main table coefficient reconstruction, canonical GPU IDs and model math are unchanged; head features omit PLE.",
            "prefill": "Trunk prefill subtracts trained-head priming once; batched head priming is already a subset.",
            "statistics": "Two throughput samples per cell and a preceding quality-warmed row cache; these are matched local screens, not statistical significance or uncached-device claims.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "quality": result["quality"]["valid"],
                "all_28_semantic_records_equal": result["quality"][
                    "all_28_semantic_records_equal"
                ],
                "all_21_semantic_outputs_equal": result["benchmark"][
                    "all_21_semantic_outputs_equal"
                ],
                "gpu_commands": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
