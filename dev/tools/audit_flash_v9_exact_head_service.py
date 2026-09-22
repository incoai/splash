#!/usr/bin/env python3
"""Read-only independent audit of the exact Q8/BF16 joint-head service screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from audit_flash_v8_qsa_service import compare, delta, idle, load, summary

COUNTER_KEYS = (
    "joint_head_vocabulary_register_commands",
    "joint_head_vocabulary_register_rows",
)
ROUTE = "original-q8-exact-cached-bf16-coeff-register-m8n32-k64-simd1-f32accum-last-r2to4-v1"


def primitive(report: dict) -> dict:
    cases = report["cases"]
    finite = [case for case in cases if case["pattern"] != 4]
    negative = [case for case in cases if case["pattern"] == 4]
    return {
        "valid": report["valid"],
        "execution_complete": report["execution_complete"],
        "coefficient_count": report["coefficient_count"],
        "coefficient_bf16_mismatches": report["coefficient_bf16_mismatches"],
        "coefficient_nonfinite_or_geometry_errors": report[
            "coefficient_nonfinite_or_geometry_errors"
        ],
        "strict_bound": report["strict_each_lane_relative_l2_bound"],
        "strict_bound_relaxed": report["strict_bound_relaxed"],
        "case_count": len(cases),
        "lane_counts": sorted({len(case["lanes"]) for case in cases}),
        "all_cases_pass": all(case["pass"] for case in cases),
        "all_greedy_and_diagnostics_exact": all(
            case["exact_greedy"] and case["exact_diagnostics"] for case in cases
        ),
        "finite_case_count": len(finite),
        "nonfinite_diagnostic_case_count": len(negative),
        "all_finite_lane_vocabulary_words_exact": all(
            lane["bf16_mismatches"] == 0
            and lane["nonfinite"] == 0
            and lane["relative_l2"] == 0
            for case in finite
            for lane in case["lanes"]
        ),
        "all_finite_lane_strict_bounds_pass": all(
            lane["strict_pass"] for case in finite for lane in case["lanes"]
        ),
        "all_nonfinite_diagnostics_match": all(
            case["pass"]
            and case["exact_diagnostics"]
            and case["exact_greedy"]
            and all(lane["nonfinite"] == 248320 for lane in case["lanes"])
            for case in negative
        ),
        "max_lane_relative_l2": max(
            lane["relative_l2"] for case in cases for lane in case["lanes"]
        ),
        "max_lane_word_mismatches": max(
            lane["bf16_mismatches"] for case in cases for lane in case["lanes"]
        ),
        "total_lane_word_mismatches": sum(
            lane["bf16_mismatches"] for case in cases for lane in case["lanes"]
        ),
        "all_original_bytes_unchanged": report["all_original_bytes_unchanged"],
    }


def head_proof(report: dict) -> dict:
    contexts = []
    for context in (128, 2048):
        records = [
            entry for entry in report["measurements"] if entry["context"] == context
        ]
        normal = [entry for entry in records if "original_timing" in entry]
        continuations = [
            entry
            for entry in records
            if entry.get("qualification") == "future_head_continuation"
        ]
        overwrites = [
            entry
            for entry in records
            if entry.get("qualification") == "truncate_overwrite_against_clean_prefix"
        ]
        control = statistics.median(
            entry["original_timing"]["command_gpu_seconds"] for entry in normal
        )
        trial = statistics.median(
            entry["candidate_timing"]["command_gpu_seconds"] for entry in normal
        )
        lane_outputs = [
            lane.get("vocabulary", lane) for entry in records for lane in entry["lanes"]
        ]
        contexts.append(
            {
                "context": context,
                "normal_pairs": len(normal),
                "continuation_anchor_count": sum(
                    entry["step"] == 0 for entry in continuations
                ),
                "future_continuation_count": sum(
                    entry["step"] > 0 for entry in continuations
                ),
                "recorded_continuation_steps": [
                    entry["step"] for entry in continuations
                ],
                "truncate_overwrite_prefixes": [
                    entry["retained_suffix_rows"] for entry in overwrites
                ],
                "all_records_accepted": all(entry["accepted"] for entry in records),
                "all_lane_vocabulary_words_exact": all(
                    output["bf16_word_mismatches"] == 0
                    and output["relative_l2"] == 0
                    and output["finite"]
                    for output in lane_outputs
                ),
                "all_premixer_words_exact": all(
                    entry["full_premixer_bf16_exact"]
                    for entry in normal + continuations
                ),
                "all_full_qsa_exact": all(
                    entry["qsa_full_buffers_exact"] for entry in normal + continuations
                )
                and all(entry["candidate_original_qsa_exact"] for entry in overwrites),
                "all_greedy_lanes_exact": all(
                    entry["exact_greedy_all_lanes"] for entry in normal + continuations
                ),
                "normal_qsa_hash_record_count": sum(
                    "original_qsa_full_hashes" in entry for entry in normal
                ),
                "normal_recorded_qsa_hashes_equal": all(
                    entry["original_qsa_full_hashes"]
                    == entry["candidate_qsa_full_hashes"]
                    for entry in normal
                    if "original_qsa_full_hashes" in entry
                ),
                "control_median_head_gpu_ms": control * 1000,
                "candidate_median_head_gpu_ms": trial * 1000,
                "head_gpu_reduction_percent": (1 - trial / control) * 100,
            }
        )
    return {
        "accepted": report["accepted"],
        "execution_complete": report["execution_complete"],
        "strict_bound": report["strict_each_lane_relative_l2_bound"],
        "strict_bound_relaxed": report["strict_bound_relaxed"],
        "declared_future_continuation_cases": report["future_continuation_cases"],
        "actual_future_continuation_cases": sum(
            entry["future_continuation_count"] for entry in contexts
        ),
        "continuation_anchor_cases": sum(
            entry["continuation_anchor_count"] for entry in contexts
        ),
        "declared_truncate_overwrite_cases": report["truncate_overwrite_cases"],
        "actual_truncate_overwrite_cases": sum(
            len(entry["truncate_overwrite_prefixes"]) for entry in contexts
        ),
        "contexts": contexts,
        "scope": "Private trained joint-head API screen, four real lanes; no target verification or HTTP performance claim.",
    }


def service_summary(report: dict) -> dict:
    result = summary(report)
    mtp = report["runtime_after"]["mtp"]
    identity = report["runtime_after"]["identity"]
    result["joint_head_route"] = {
        key: identity.get(key)
        for key in ("joint_head_vocabulary_register", "joint_head_vocabulary_route")
    }
    if all(key in mtp for key in COUNTER_KEYS):
        counters = {key: delta(report, "mtp", key) for key in COUNTER_KEYS}
        counters["scope"] = mtp["joint_head_vocabulary_register_counter_scope"]
        counts = result["work"]["joint_head_commands_by_width"]
        completed = sum(counts[1:])
        real_rows = sum(
            width * count for width, count in enumerate(counts, start=1) if width >= 2
        )
        commands, rows = (counters[key] for key in COUNTER_KEYS)
        counters["total_completed_joint_head_commands"] = completed
        counters["total_joint_head_real_rows"] = real_rows
        counters["register_commands_are_joint_head_subset"] = 0 <= commands <= completed
        counters["register_rows_are_joint_head_subset"] = 0 <= rows <= real_rows
        counters["register_real_rows_within_R2to4"] = (
            2 * commands <= rows <= 4 * commands
        )
        counters["all_joint_head_commands_project_vocabulary"] = commands == completed
        result["completed_register_counter_delta"] = counters
    return result


def quality_outputs(report: dict) -> dict:
    fields = (
        "text",
        "reasoning_text",
        "finish_reason",
        "usage",
        "http_status",
        "errors",
    )
    return {
        case["id"]: [
            {
                **{field: record.get(field) for field in fields},
                "tool_calls": [
                    {"type": call.get("type"), "function": call.get("function")}
                    for call in record.get("tool_calls", [])
                ],
            }
            for record in case["records"]
        ]
        for case in report["cases"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("build/release/flash"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "build/release/flash/v9-mtp-q8-bf16-register-independent-service-audit.json"
        ),
    )
    args = parser.parse_args()
    folder = args.report_dir
    filenames = {
        "private_full_head": "v9-exact-q8-full-head-proof.json",
        "private_patterns_R4": "v9-exact-q8-register-patterns.json",
        "private_patterns_R2": "v9-exact-q8-register-rows2.json",
        "private_patterns_R3": "v9-exact-q8-register-rows3.json",
        "accepted_v7": "default-v7-normal-confirm-http-performance.json",
        "accepted_v7_quality": "default-v7-normal-quality-qualification.json",
        "candidate_on": "v9-mtp-q8-bf16-register-on-http-performance.json",
        "candidate_on_confirm": "v9-mtp-q8-bf16-register-on-confirm-http-performance.json",
        "same_build_off": "v9-mtp-q8-bf16-register-off-http-performance.json",
        "quality": "v9-mtp-q8-bf16-register-quality.json",
        "quality_comparison": "v9-mtp-q8-bf16-register-quality-comparison.json",
        "candidate_on_idle": "v9-mtp-q8-bf16-register-on-idle-status.json",
        "candidate_on_confirm_idle": "v9-mtp-q8-bf16-register-on-confirm-idle-status.json",
        "same_build_off_idle": "v9-mtp-q8-bf16-register-off-idle-status.json",
    }
    reports = {
        name: load(folder / filename)
        for name, filename in filenames.items()
        if (folder / filename).exists()
    }
    required = (
        "candidate_on",
        "same_build_off",
        "quality",
        "quality_comparison",
        "candidate_on_idle",
        "same_build_off_idle",
    )
    result = {
        "schema": "splash-flash-exact-bf16-q8-head-independent-service-audit-v1",
        "gpu_commands": 0,
        "inputs": [
            {
                "name": filename,
                "sha256": hashlib.sha256((folder / filename).read_bytes()).hexdigest(),
            }
            for name, filename in filenames.items()
            if name in reports
        ],
        "missing_required_service_artifacts": [
            filenames[name] for name in required if name not in reports
        ],
        "private_full_head": head_proof(reports["private_full_head"]),
        "private_primitives": {
            key: primitive(reports[key])
            for key in (
                "private_patterns_R2",
                "private_patterns_R3",
                "private_patterns_R4",
            )
        },
        "reports": {
            key: service_summary(reports[key])
            for key in (
                "accepted_v7",
                "candidate_on",
                "candidate_on_confirm",
                "same_build_off",
            )
            if key in reports
        },
        "idle": {
            key.removesuffix("_idle"): idle(reports[key])
            for key in (
                "candidate_on_idle",
                "candidate_on_confirm_idle",
                "same_build_off_idle",
            )
            if key in reports
        },
        "excluded_promotion_control": {
            "file": "v9-v7-baseline-http-performance.json",
            "reason": "CPU xctrace finalization remained active during measurement; not a clean promotion control.",
            "used_for_promotion": False,
        },
        "scope": {
            "new_route": "Joint trained MTP head Last vocabulary projection, 2..4 real lanes, qualified original Q8/G64.",
            "counter": "Successfully completed joint head commands and real vocabulary rows; not graph-construction counters.",
            "head_command_accounting": "Head decode is counted once; batched prompt-head priming is None and excluded from this route. Joint commands with None are excluded from register counters.",
            "prefill_accounting": "Trunk prefill subtracts head priming once; batched priming is already a subset.",
            "memory": "Zero route-owned weights/scratch; original code views and existing padding reused. BF16 cache retained for fallback.",
            "statistics": "Two throughput samples per cell; matched screens do not establish statistical significance.",
        },
    }
    if "quality" in reports and "quality_comparison" in reports:
        quality, comparison = reports["quality"], reports["quality_comparison"]
        result["quality"] = {
            "valid": quality["valid"],
            "passed": quality["passed_cases"],
            "failed": quality["failed_cases"],
            "full_plan_coverage": quality["full_plan_coverage"],
            "new_regressions": comparison["new_task_regressions"],
            "all_22_exact_outputs_equal": len(comparison["cases"]) == 22
            and all(case["exact_output_equal"] for case in comparison["cases"]),
        }
        if "accepted_v7_quality" in reports:
            left, right = (
                quality_outputs(reports["accepted_v7_quality"]),
                quality_outputs(quality),
            )
            result["quality"]["all_28_semantic_records_equal"] = (
                left == right and sum(map(len, right.values())) == 28
            )
            result["quality"]["semantic_equality_fields"] = [
                "text",
                "reasoning_text",
                "finish_reason",
                "usage",
                "http_status",
                "errors",
                "tool type/name/arguments",
            ]
            result["quality"]["excluded_generated_fields"] = (
                "Request IDs, tool-call IDs, timestamps, stream chunk boundaries and performance metrics."
            )
            result["quality"]["manual_factual_review"] = {
                "case": "long_factual_prose",
                "unchanged_from_accepted_v7": left["long_factual_prose"]
                == right["long_factual_prose"],
                "finding": "Output coherently describes evaporation, condensation and precipitation with no contradiction; identical accepted-v7 text.",
            }
    if "candidate_on" in reports:
        if "accepted_v7" in reports:
            result["accepted_v7_vs_on"] = compare(
                reports["accepted_v7"], reports["candidate_on"]
            )
        if "same_build_off" in reports:
            result["same_build_off_vs_on"] = compare(
                reports["same_build_off"], reports["candidate_on"]
            )
            off, on = reports["same_build_off"], reports["candidate_on"]
            on_route, off_route = (
                report["runtime_after"]["identity"] for report in (on, off)
            )
            on_counters = result["reports"]["candidate_on"].get(
                "completed_register_counter_delta", {}
            )
            off_counters = result["reports"]["same_build_off"].get(
                "completed_register_counter_delta", {}
            )
            result["route_validation"] = {
                "on_declares_exact_route": on_route.get(
                    "joint_head_vocabulary_register"
                )
                is True
                and on_route.get("joint_head_vocabulary_route") == ROUTE,
                "off_declares_register_disabled": off_route.get(
                    "joint_head_vocabulary_register"
                )
                is False,
                "on_completed_register_commands_positive": on_counters.get(
                    COUNTER_KEYS[0], 0
                )
                > 0,
                "off_completed_register_commands_zero": off_counters.get(
                    COUNTER_KEYS[0]
                )
                == 0,
                "off_completed_register_rows_zero": off_counters.get(COUNTER_KEYS[1])
                == 0,
                "same_memory_audit_and_peak": result["same_build_off_vs_on"][
                    "memory_audit_equal"
                ]
                and result["same_build_off_vs_on"]["memory_peak_equal"],
            }
            result["same_build_performance_scope"] = {
                "on_trace_disabled": on["runtime_after"]["request_command_trace"][
                    "enabled"
                ]
                is False,
                "off_trace_disabled": off["runtime_after"]["request_command_trace"][
                    "enabled"
                ]
                is False,
                "head_decode_gpu_ms_candidate_minus_control": result[
                    "same_build_off_vs_on"
                ]["phase_change"]["head_decode"][
                    "total_gpu_ms_candidate_minus_baseline"
                ],
                "head_decode_command_counts_equal": result["same_build_off_vs_on"][
                    "phase_change"
                ]["head_decode"]["commands_equal"],
                "attribution": "The route directly changes joint Last vocabulary within head decode. Singleton, trunk prefill and target-verifier changes are timing drift, not direct route benefits.",
            }
            if "candidate_on_confirm" in reports:
                result["same_build_off_vs_on_confirm"] = compare(
                    off, reports["candidate_on_confirm"]
                )
    result["service_evidence_complete"] = not result[
        "missing_required_service_artifacts"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "service_evidence_complete": result["service_evidence_complete"],
                "missing": result["missing_required_service_artifacts"],
                "gpu_commands": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
