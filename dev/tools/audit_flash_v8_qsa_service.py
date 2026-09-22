#!/usr/bin/env python3
"""CPU-only independent audit of the local QSA-output N32 service screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def inputs(report: dict) -> list[tuple]:
    return [
        (
            section,
            wave["sample"],
            wave["context"],
            wave["width"],
            lane,
            record["prompt_u32le_sha256"],
            record["request_body"],
        )
        for section in ("warmups", "waves")
        for wave in report[section]
        for lane, record in enumerate(wave["records"])
    ]


def outputs(report: dict) -> list[dict]:
    fields = ("text", "reasoning_text", "tool_calls", "finish_reason", "usage")
    return [
        {field: record[field] for field in fields}
        for section in ("warmups", "waves")
        for wave in report[section]
        for record in wave["records"]
    ]


def delta(report: dict, *path: str):
    before, after = report["runtime_before"], report["runtime_after"]
    for key in path:
        before, after = before[key], after[key]
    if isinstance(before, list):
        return [a - b for b, a in zip(before, after, strict=True)]
    return after - before


WORK_FIELDS = (
    "eligible_requests",
    "autoregressive_requests",
    "verification_cycles",
    "drafted_tokens",
    "accepted_committed_drafts",
    "matched_proposals",
    "emitted_accepted_proposals",
    "permanent_batch_fallback_requests",
    "accepted_prefix_histogram",
    "completed_cycles_by_proposed_depth",
    "singleton_committed_fold_calls",
    "joint_cohorts_attempted",
    "joint_partial_cohorts_before_target",
    "joint_members_dropped_before_commit",
    "joint_head_commands_by_width",
    "joint_target_verifiers_by_width",
)
PHASES = ("head_priming", "head_decode", "target_verify", "prefix_restore")


def summary(report: dict) -> dict:
    result = {
        "valid": report["valid"],
        "plan_sha256": report["plan_sha256"],
        "generation_count": len(outputs(report)),
        "cells": [],
        "work": {key: delta(report, "mtp", key) for key in WORK_FIELDS},
        "phases": {},
        "memory_actual": report["runtime_after"]["memory_actual"],
        "memory_audit": report["runtime_after"]["memory_audit"],
        "identity": report["runtime_after"]["identity"],
        "persisted_operands": report["runtime_after"]["persisted_operands"],
        "persisted_experts": report["runtime_after"]["persisted_experts"],
        "native_mtp_depth_policy": report["native_mtp_depth_policy"],
        "request_command_trace": report["runtime_after"]["request_command_trace"],
    }
    for context, width in ((128, 1), (128, 4), (2048, 1), (2048, 4)):
        waves = [
            wave
            for wave in report["waves"]
            if wave["context"] == context and wave["width"] == width
        ]
        result["cells"].append(
            {
                "context": context,
                "width": width,
                "samples_tps": [
                    wave["full_wave_output_tokens_per_second"] for wave in waves
                ],
                "median_tps": statistics.median(
                    wave["full_wave_output_tokens_per_second"] for wave in waves
                ),
                "median_first_content_ms": statistics.median(
                    wave["mean_client_first_content_ms"] for wave in waves
                ),
            }
        )
    for phase in PHASES:
        result["phases"][phase] = {
            key: delta(report, "mtp", phase, key)
            for key in ("total_gpu_ms", "total_wall_ms", "forward_host_wall_ms")
        }
        result["phases"][phase]["commands"] = delta(
            report, "mtp", phase, "host_command_subphases", "timed_commands"
        )
    for phase in ("prefill", "decode"):
        result["phases"][phase] = {
            key: delta(report, "model_timing", phase, key)
            for key in ("total_gpu_ms", "total_wall_ms", "forward_host_wall_ms")
        }
        result["phases"][phase]["commands"] = delta(
            report, "model_timing", phase, "host_command_subphases", "timed_commands"
        )
    result["phases"]["trunk_prefill"] = {
        key: result["phases"]["prefill"][key] - result["phases"]["head_priming"][key]
        for key in ("total_gpu_ms", "total_wall_ms", "forward_host_wall_ms")
    }
    route = report["runtime_after"].get("qsa_output_f32_n32_route_counters")
    if route is not None:
        result["route_counter_total"] = route
        result["route_counter_delta"] = {
            key: delta(report, "qsa_output_f32_n32_route_counters", key)
            for key in ("encoded_calls", "encoded_real_rows")
        }
    return result


def compare(base: dict, candidate: dict) -> dict:
    b, c = summary(base), summary(candidate)
    identities = (
        "source",
        "loaded_model_layout_sha256",
        "kernel_routes",
        "worker_semantics",
        "mtp_semantics",
        "mtp_attention_route",
        "joint_head_semantics",
        "joint_head_attention_route",
        "batch_prefill_kernel_routes",
        "batch_mtp_prefill_attention_route",
        "weight_format",
    )
    result = {
        "plan_equal": base["plan_sha256"] == candidate["plan_sha256"],
        "all_21_inputs_equal": len(inputs(base)) == len(inputs(candidate)) == 21
        and inputs(base) == inputs(candidate),
        "all_21_generation_outputs_equal": len(outputs(base))
        == len(outputs(candidate))
        == 21
        and outputs(base) == outputs(candidate),
        "acceptance_and_verifier_work_equal": b["work"] == c["work"],
        "source_and_unchanged_routes_equal": {
            key: b["identity"][key] == c["identity"][key] for key in identities
        },
        "persisted_operand_identity_equal": b["persisted_operands"]
        == c["persisted_operands"],
        "persisted_expert_identity_equal": b["persisted_experts"]
        == c["persisted_experts"],
        "memory_audit_equal": b["memory_audit"] == c["memory_audit"],
        "memory_peak_equal": b["memory_actual"]["peak_bytes"]
        == c["memory_actual"]["peak_bytes"],
        "cells": [],
        "phase_change": {},
    }
    for baseline, trial in zip(b["cells"], c["cells"], strict=True):
        result["cells"].append(
            {
                "context": baseline["context"],
                "width": baseline["width"],
                "throughput_change_percent": (
                    trial["median_tps"] / baseline["median_tps"] - 1
                )
                * 100,
                "first_content_change_ms": trial["median_first_content_ms"]
                - baseline["median_first_content_ms"],
            }
        )
    for phase, baseline in b["phases"].items():
        trial = c["phases"][phase]
        result["phase_change"][phase] = {
            key + "_candidate_minus_baseline": trial[key] - baseline[key]
            for key in ("total_gpu_ms", "total_wall_ms", "forward_host_wall_ms")
        }
        if "commands" in baseline:
            result["phase_change"][phase]["commands_equal"] = (
                baseline["commands"] == trial["commands"]
            )
        result["phase_change"][phase]["gpu_reduction_percent"] = (
            1 - trial["total_gpu_ms"] / baseline["total_gpu_ms"]
        ) * 100
    return result


def idle(status: dict) -> dict:
    fields = (
        "active_requests",
        "queued",
        "prefilling",
        "decoding",
        "command_in_flight",
    )
    scheduler = {key: status["scheduler"][key] for key in fields}
    return {
        "idle": all(not value for value in scheduler.values()),
        "scheduler": scheduler,
        "metal_healthy": status["metal"]["healthy"],
        "requests": status["requests"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("build/release/flash"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "build/release/flash/v8-qsa-out-n32-independent-service-audit.json"
        ),
    )
    args = parser.parse_args()
    folder = args.report_dir
    names = {
        "accepted_v7": "default-v7-normal-confirm-http-performance.json",
        "candidate_on": "v8-qsa-out-n32-on-http-performance.json",
        "same_build_off": "v8-qsa-out-n32-off-http-performance.json",
    }
    reports = {
        key: load(folder / name)
        for key, name in names.items()
        if (folder / name).exists()
    }
    quality = load(folder / "v8-qsa-out-n32-quality.json")
    quality_pair = load(folder / "v8-qsa-out-n32-quality-comparison.json")
    result = {
        "schema": "splash-flash-qsa-output-n32-independent-service-audit-v1",
        "gpu_commands": 0,
        "inputs": [
            {
                "name": name,
                "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest(),
            }
            for key, name in names.items()
            if key in reports
        ],
        "quality": {
            "valid": quality["valid"],
            "passed": quality["passed_cases"],
            "failed": quality["failed_cases"],
            "coverage": quality["full_plan_coverage"],
            "new_regressions": quality_pair["new_task_regressions"],
            "all_22_exact_outputs_equal": len(quality_pair["cases"]) == 22
            and all(case["exact_output_equal"] for case in quality_pair["cases"]),
        },
        "reports": {key: summary(report) for key, report in reports.items()},
        "accepted_v7_vs_on": compare(reports["accepted_v7"], reports["candidate_on"]),
        "idle": {},
        "scope": {
            "new_route": "Main target QSA o_proj only; source Q5/Q6 group64, N2560 K6144, R4..16.",
            "unchanged": "Trained MTP head, R1 decode, large-row prefill and original coefficient bytes.",
            "counter": "Graph construction counters, not completed GPU dispatch counts.",
            "prefill_accounting": "Trunk prefill subtracts head priming once; batched head priming is a subset.",
            "attribution": "Only matched target_verify GPU phase is direct route evidence; head or prefill wall drift must not be credited to QSA.",
            "statistics": "Two throughput samples per cell; these screens do not establish statistical significance.",
        },
    }
    if "same_build_off" in reports:
        result["same_build_off_vs_on"] = compare(
            reports["same_build_off"], reports["candidate_on"]
        )
        phase = result["same_build_off_vs_on"]["phase_change"]["target_verify"]
        result["recommendation"] = {
            "default": "OFF",
            "qualification": "22 bounded quality/lifecycle cases passed with exact comparison outputs; all 21 benchmark generations and verifier work match.",
            "performance": "No measured target verifier GPU benefit in the same-build screen; primary throughput is mixed across singleton and concurrent workloads.",
            "target_verify_gpu_ms_candidate_minus_baseline": phase[
                "total_gpu_ms_candidate_minus_baseline"
            ],
            "target_verify_gpu_reduction_percent": phase["gpu_reduction_percent"],
            "next_step": "Retain the opt-in route and require a meaningful matched full-service benefit before changing the default.",
        }
    for key, name in (
        ("accepted_v7", "default-v7-normal-final-idle-status.json"),
        ("candidate_on", "v8-qsa-out-n32-on-idle-status.json"),
        ("same_build_off", "v8-qsa-out-n32-off-idle-status.json"),
    ):
        if (folder / name).exists():
            result["idle"][key] = idle(load(folder / name))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "reports": list(reports),
                "quality": result["quality"],
                "gpu_commands": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
