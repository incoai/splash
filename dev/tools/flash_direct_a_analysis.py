"""CPU-only attribution of three frozen native HTTP benchmark reports."""

import hashlib
import json
import statistics
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[2] / "build/release/flash"
PATHS = {
    "qualified_v4": DIRECTORY / "splash-default-v4-zero-copy-http-performance.json",
    "saved_operands_v4": DIRECTORY / "splash-saved-operands-v4-http-performance.json",
    "direct_a_saved_operands": DIRECTORY
    / "splash-direct-a-saved-operands-http-performance.json",
    "omlx_refresh": DIRECTORY / "omlx-mtp3-refresh-http-performance.json",
}
PHASE_FIELDS = ("total_gpu_ms", "total_wall_ms", "forward_host_wall_ms")


def delta(report, path):
    def get(moment):
        value = report[moment]
        for key in path.split("."):
            value = value[key]
        return value

    after, before = get("runtime_after"), get("runtime_before")
    return (
        [a - b for a, b in zip(after, before, strict=True)]
        if isinstance(after, list)
        else after - before
    )


def native_statistics(report):
    inclusive = {
        field: delta(report, "model_timing.prefill." + field) for field in PHASE_FIELDS
    }
    head = {field: delta(report, "mtp.head_priming." + field) for field in PHASE_FIELDS}
    mtp = {
        key: delta(report, "mtp." + key)
        for key in (
            "drafted_tokens",
            "accepted_committed_drafts",
            "matched_proposals",
            "verification_cycles",
            "singleton_committed_fold_calls",
            "joint_cohorts_attempted",
            "joint_head_commands_by_width",
            "joint_target_verifiers_by_width",
            "accepted_prefix_histogram",
            "completed_cycles_by_proposed_depth",
        )
    }
    mtp["committed_acceptance_fraction"] = (
        mtp["accepted_committed_drafts"] / mtp["drafted_tokens"]
    )
    return {
        "timing_scope": "worker-lifetime counter deltas covering warmup plus 20 measured requests",
        "prefill_including_head_ms": inclusive,
        "head_priming_ms": head,
        "target_trunk_prefill_ms": {
            field: inclusive[field] - head[field] for field in PHASE_FIELDS
        },
        "decode_including_mtp_ms": {
            field: delta(report, "model_timing.decode." + field)
            for field in PHASE_FIELDS
        },
        "mtp": mtp,
        "after_scheduler_active": report["runtime_after"]["scheduler"][
            "active_requests"
        ],
        "after_requests": report["runtime_after"]["requests"],
        "after_memory_actual": report["runtime_after"]["memory_actual"],
    }


def records(report):
    return {
        (
            kind,
            wave["sample"],
            wave["context"],
            wave["width"],
            record["plan_lane"],
        ): record
        for kind in ("warmups", "waves")
        for wave in report[kind]
        for record in wave["records"]
    }


def paired_comparison(baseline, candidate):
    left = {
        k: v
        for k, v in baseline["runtime_before"]["identity"].items()
        if k != "engine_instance_id"
    }
    right = {
        k: v
        for k, v in candidate["runtime_before"]["identity"].items()
        if k != "engine_instance_id"
    }
    before, after = records(baseline), records(candidate)
    fields = (
        "request_body",
        "prompt_u32le_sha256",
        "text",
        "reasoning_text",
        "tool_calls",
        "finish_reason",
        "usage",
        "response_sha256",
        "http_status",
        "errors",
        "error_frames",
        "performance_validation_errors",
    )
    changed = []
    for key in before.keys() | after.keys():
        different = [
            field
            for field in fields
            if before.get(key, {}).get(field) != after.get(key, {}).get(field)
        ]
        if different:
            changed.append({"request": list(key), "changed_fields": different})
    return {
        "identity_fields_changed": {
            key: {"baseline": left.get(key), "candidate": right.get(key)}
            for key in left.keys() | right.keys()
            if left.get(key) != right.get(key)
        },
        "request_count": len(before),
        "paired_records_all_exact": not changed and before.keys() == after.keys(),
        "record_differences": changed,
        "native_mtp_depth_policy_equal": baseline["native_mtp_depth_policy"][
            "runtime_before"
        ]
        == candidate["native_mtp_depth_policy"]["runtime_before"],
    }


def main():
    reports = {label: json.loads(path.read_text()) for label, path in PATHS.items()}
    if (
        not all(report["valid"] is True for report in reports.values())
        or len({report["plan_sha256"] for report in reports.values()}) != 1
    ):
        raise ValueError("all benchmark reports must be valid and share a frozen plan")
    cohorts = []
    for context, width in ((128, 1), (128, 4), (2048, 1), (2048, 4)):
        row = {"prompt_tokens": context, "concurrency": width, "backends": {}}
        for label, report in reports.items():
            waves = [
                wave
                for wave in report["waves"]
                if wave["context"] == context and wave["width"] == width
            ]
            requests = [record for wave in waves for record in wave["records"]]
            row["backends"][label] = {
                "median_http_output_tokens_per_second": statistics.median(
                    wave["full_wave_output_tokens_per_second"] for wave in waves
                ),
                "sample_http_output_tokens_per_second": [
                    wave["full_wave_output_tokens_per_second"] for wave in waves
                ],
                "mean_wave_wall_seconds": statistics.mean(
                    wave["wall_seconds"] for wave in waves
                ),
                "mean_http_visible_first_content_ms": statistics.mean(
                    record["first_content_ms"] for record in requests
                ),
                "mean_first_content_to_http_end_ms": statistics.mean(
                    record["http_total_ms"] - record["first_content_ms"]
                    for record in requests
                ),
            }
        for baseline in ("qualified_v4", "saved_operands_v4", "omlx_refresh"):
            row["direct_relative_throughput_gain_vs_" + baseline] = (
                row["backends"]["direct_a_saved_operands"][
                    "median_http_output_tokens_per_second"
                ]
                / row["backends"][baseline]["median_http_output_tokens_per_second"]
                - 1
            )
        cohorts.append(row)
    native = {
        label: native_statistics(report)
        for label, report in reports.items()
        if label != "omlx_refresh"
    }
    reductions = {
        baseline: {
            field: 1
            - native["direct_a_saved_operands"]["target_trunk_prefill_ms"][field]
            / native[baseline]["target_trunk_prefill_ms"][field]
            for field in PHASE_FIELDS
        }
        for baseline in ("qualified_v4", "saved_operands_v4")
    }
    quality_path = DIRECTORY / "persisted-int8-quality-direct-a-comparison-v1.json"
    quality = json.loads(quality_path.read_text())
    result = {
        "schema": "splash-flash-direct-a-saved-operands-cpu-analysis-v1",
        "gpu_executed_by_analysis": False,
        "reports": {
            label: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "valid": reports[label]["valid"],
            }
            for label, path in PATHS.items()
        },
        "plan_sha256": reports["qualified_v4"]["plan_sha256"],
        "cohorts": cohorts,
        "native_counter_phase_deltas": native,
        "saved_vs_qualified_v4": paired_comparison(
            reports["qualified_v4"], reports["saved_operands_v4"]
        ),
        "direct_vs_qualified_v4": paired_comparison(
            reports["qualified_v4"], reports["direct_a_saved_operands"]
        ),
        "direct_vs_saved_v4": paired_comparison(
            reports["saved_operands_v4"], reports["direct_a_saved_operands"]
        ),
        "target_trunk_prefill_relative_reductions": reductions,
        "quality": {
            "pair_report": str(quality_path),
            "new_task_regressions": quality["new_task_regressions"],
            "full_plan_coverage": quality["full_plan_coverage"],
            "exact_case_outputs_equal": sum(
                case["exact_output_equal"] for case in quality["cases"]
            ),
            "cases": len(quality["cases"]),
            "manual_prose_review": "Baseline prose is coherent but says Condensation cools this vapor, reversing ordinary causal wording. Preserve this baseline-model finding when evaluating conversion regressions.",
        },
        "limits": [
            "Saved-v4 benchmark overlapped CPU conversion; no isolated saved-store performance conclusion.",
            "Direct-A full-model gain is not its 28-37 percent primitive gain.",
            "HTTP first content includes queue/prefill/head prime/transport; oMLX buffers initial bursts.",
            "Inclusive native prefill already includes each head-priming command; subtract head priming once for target-trunk attribution.",
            "Older v4 runtime_after still has one active request; that snapshot may omit final bookkeeping despite complete 128-token HTTP records.",
            "Two measured samples and finite quality fixtures do not establish general quality.",
        ],
    }
    output = DIRECTORY / "direct-a-saved-operands-cpu-analysis-v1.json"
    with output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print("artifact", output)
    for label, value in native.items():
        print(
            label,
            "target",
            value["target_trunk_prefill_ms"],
            "head",
            value["head_priming_ms"],
            "acceptance",
            value["mtp"]["committed_acceptance_fraction"],
        )
    print("target reductions", reductions)
    for row in cohorts:
        print(
            row["prompt_tokens"],
            row["concurrency"],
            "rates",
            {
                label: value["median_http_output_tokens_per_second"]
                for label, value in row["backends"].items()
            },
        )
    print(
        "exact",
        {
            key: result[key]["paired_records_all_exact"]
            for key in (
                "saved_vs_qualified_v4",
                "direct_vs_qualified_v4",
                "direct_vs_saved_v4",
            )
        },
    )
    print("quality", result["quality"])


if __name__ == "__main__":
    main()
