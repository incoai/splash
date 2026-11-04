#!/usr/bin/env python3
"""Fixed16 lookup: all predeclared edits, ordinary controls, and natural EOS.

One production binary, eight OFF warmup passes and three adjacent per-case ABBA
groups. Never builds or selects cases from their results. --audit is CPU-only.
"""

import argparse
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

if __package__:
    from . import experiment_common as common
    from . import lookup_task_score as task_score
else:
    import experiment_common as common
    import lookup_task_score as task_score

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures/lookup"
ORDER = common.ORDER
require = common.require
TOKENIZER_SHA256 = task_score.TOKENIZER_SHA256
TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
GEOMETRY = {
    "prefill_rows": 2048,
    "verify_rows": 8,
    "batch_width": 4,
    "kv_block_tokens": 32,
}
COHORTS = {
    "complete": {
        "names": (
            "lookup_edit_python",
            "lookup_edit_markdown",
            "lookup_edit_config",
            "lookup_control_intervals",
            "lookup_control_lru",
            "lookup_control_http",
        ),
        "edits": 3,
        "sha256": "26f7064d04065fe523914d65d516c475b8877c5603fa1f1795b5bcca53c49861",
        "version": 3,
        "native": "predeclared_complete_edits_cap1024_controls256",
    },
    "fresh": {
        "names": (
            "fresh_edit_sql",
            "fresh_edit_toml",
            "fresh_edit_js",
            "fresh_edit_markdown",
            "fresh_edit_csv",
            "fresh_edit_clause",
            "fresh_control_backpressure",
            "fresh_control_rehearsal",
            "fresh_control_experiment",
        ),
        "edits": 6,
        "sha256": "bcc7efbe695147f3c10df0394ce82190291c429267e2da9d1ebc7c919e300f02",
        "version": 5,
        "native": "predeclared_fresh_six_edits_cap1024_three_controls256",
    },
}
LOOKUP = (
    "prompt_lookup_searches",
    "prompt_lookup_hits",
    "prompt_lookup_bypassed_cycles",
    "prompt_lookup_proposed_tokens",
    "prompt_lookup_accepted_tokens",
)
COUNTERS = LOOKUP + (
    "decode_batches",
    "decode_output_tokens",
    "drafted_tokens",
    "accepted_draft_tokens",
    "matched_tokens",
    "target_prefill_rows",
    "draft_context_rows",
    "draft_context_rows_avoided",
    "draft_state_restore_skipped",
    "junction_materializations",
    "junction_materialization_failures",
    "state_allocated_before_bytes",
    "state_allocated_after_bytes",
    "state_allocated_peak_observed_bytes",
    "cached_state_bytes",
    "state_entries",
    "state_evictions",
)
SEMANTIC_COUNTERS = tuple(
    key
    for key in COUNTERS
    if not key.startswith("state_allocated_")
    and key not in ("cached_state_bytes", "state_entries")
)
TIMINGS = (
    "wall_ms",
    "ttft_ms",
    "prefill_gpu_ms",
    "decode_gpu_ms",
    "decode_wall_ms",
    "prompt_lookup_search_wall_ms",
)
STOP_TOKENS = (248044, 248046)


def fixture_document(cohort):
    path = FIXTURES / (cohort + ".json")
    require(
        common.digest(path) == COHORTS[cohort]["sha256"],
        "fixture bytes differ from predeclared cohort",
    )
    doc = json.loads(path.read_text())
    require(
        doc["tokenizer_sha256"] == TOKENIZER_SHA256
        and doc["template_sha256"] == TEMPLATE_SHA256,
        "fixture tokenizer/template differs",
    )
    require(
        tuple(row["name"] for row in doc["fixtures"]) == COHORTS[cohort]["names"],
        "fixture names/order differ",
    )
    return doc


def valid_tokens(tokens):
    return (
        isinstance(tokens, list)
        and bool(tokens)
        and all(type(t) is int and 0 <= t < 248320 for t in tokens)
    )


def check_row(row, name, enabled, inputs, cap):
    require(row.get("scenario") == name, "unexpected case/order")
    actual_input, output = row.get("input_tokens"), row.get("output_tokens")
    require(
        valid_tokens(actual_input) and actual_input == inputs,
        "full input array differs: " + name,
    )
    require(
        row.get("prompt_tokens") == len(inputs)
        and row.get("input_token_hash") == common.token_hash(inputs),
        "input count/hash differs: " + name,
    )
    require(
        valid_tokens(output) and 0 < len(output) <= cap,
        "full output array/count invalid: " + name,
    )
    require(
        row.get("requested_output_tokens") == cap
        and row.get("actual_output_tokens") == len(output),
        "requested/actual count differs: " + name,
    )
    require(
        not any(t in STOP_TOKENS for t in output[:-1]),
        "output continues after stop: " + name,
    )
    require(
        len(output) == cap or output[-1] in STOP_TOKENS,
        "short output lacks natural EOS: " + name,
    )
    for key in TIMINGS:
        value = row.get(key)
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "invalid timing: " + key,
        )
    require(
        all(row[key] > 0 for key in ("wall_ms", "ttft_ms", "prefill_gpu_ms")),
        "missing request timing",
    )
    require(
        row["ttft_ms"] <= row["wall_ms"] + 0.1
        and row["decode_wall_ms"] <= row["wall_ms"] + 0.1,
        "component exceeds request wall",
    )
    require(
        all(type(row.get(key)) is int and row[key] >= 0 for key in COUNTERS),
        "invalid/missing counter",
    )
    require(
        row.get("cache_status") == "miss"
        and row["matched_tokens"] == 0
        and row["target_prefill_rows"] == len(inputs),
        "fixture was not cold",
    )
    require(
        not any(
            row[key]
            for key in (
                "state_evictions",
                "junction_materializations",
                "junction_materialization_failures",
            )
        ),
        "cache work confounds fixture",
    )
    require(
        row["state_allocated_peak_observed_bytes"]
        >= max(row["state_allocated_before_bytes"], row["state_allocated_after_bytes"]),
        "invalid allocation peak",
    )
    cycles = row["decode_batches"]
    if cycles:
        require(
            0 < row["decode_output_tokens"] <= len(output)
            and row["decode_gpu_ms"] > 0
            and row["decode_wall_ms"] > 0,
            "decode counters inconsistent",
        )
    else:
        require(
            row["decode_output_tokens"]
            == row["decode_gpu_ms"]
            == row["decode_wall_ms"]
            == 0
            and len(output) == 1
            and output[-1] in STOP_TOKENS,
            "zero-decode output is not immediate EOS",
        )
    require(
        0 <= row["accepted_draft_tokens"] <= row["drafted_tokens"] <= 7 * cycles,
        "proposal/acceptance counters inconsistent",
    )
    searches, hits, bypasses, proposed, accepted = (row[key] for key in LOOKUP)
    if enabled:
        require(
            searches == cycles
            and hits == bypasses <= searches
            and bypasses <= proposed <= 7 * bypasses
            and accepted <= proposed <= row["drafted_tokens"]
            and accepted <= row["accepted_draft_tokens"],
            "lookup counters inconsistent",
        )
    else:
        require(
            not any(row[key] for key in LOOKUP)
            and row["prompt_lookup_search_wall_ms"] == 0,
            "OFF request executed lookup",
        )


def analyze(document, cohort, model_pins=None, model_root=None):
    config = COHORTS[cohort]
    names, edit_count = config["names"], config["edits"]
    edits, controls = names[:edit_count], names[edit_count:]
    inputs = {
        row["name"]: row["input_tokens"] for row in fixture_document(cohort)["fixtures"]
    }
    pins = model_pins or json.loads((FIXTURES / "model-pins.json").read_text())
    expected = {
        "schema_version": 2,
        "experimental_prompt_lookup": False,
        "lookup_output_tokens": 256,
        "lookup_edit_output_tokens": 1024,
        "lookup_control_output_tokens": 256,
        "lookup_fixture_version": config["version"],
        "lookup_fixture_cohort": config["native"],
        "lookup_fixture_sha256": config["sha256"],
        "prompt_lookup_match_tokens": 16,
        "prompt_lookup_search_tokens": 65536,
        "prompt_lookup_proposal_tokens": 7,
        "lookup_tokenizer_sha256": TOKENIZER_SHA256,
        "lookup_template_sha256": TEMPLATE_SHA256,
        "lookup_runtime_instrumented": False,
        "lookup_logits_capture_enabled": False,
        "lookup_compilation_profile": "production_O3_flags",
        "lookup_abba_enabled": True,
        "lookup_abba_groups": 3,
        "lookup_abba_order": list(ORDER),
        "lookup_thermal_warmup_requests": 8 * len(names),
        "lookup_thermal_warmup_in_primary": False,
        "ane_ffn_share": 0,
        "ane_ffn_bytes": 0,
        "max_context_tokens": 16384,
        "geometry": GEOMETRY,
        "skipped_context_lengths": [],
        "performance_pass": True,
        "performance_failures": [],
    }
    for key, value in expected.items():
        require(
            document.get(key) == value
            and (type(value) is not bool or type(document.get(key)) is bool),
            "root invariant differs: " + key,
        )
    require(
        document.get("experimental_adaptive_prompt_lookup", False) is False,
        "adaptive policy is out of scope",
    )
    identity = document.get("identity", {})
    if model_root is not None:
        require(
            identity.get("model_root") == str(model_root),
            "native model root differs from pinned installation",
        )
    require(
        document.get("prompt_lookup_candidate_policy", "fixed16") == "fixed16",
        "candidate policy differs",
    )
    require(
        identity.get("loaded_model_layout_sha256") == pins["loaded_model_layout_sha256"]
        and bool(identity.get("device"))
        and bool(document.get("build_id")),
        "model/device/build identity differs",
    )
    require(
        document.get("no_ane_context_tokens", 0) >= 16384, "non-ANE memory plan differs"
    )
    warmups, raw = (
        document.get("lookup_thermal_warmup_measurements", []),
        document.get("measurements", []),
    )
    require(
        len(warmups) == 8 * len(names) and len(raw) == 12 * len(names),
        "incomplete retained request coverage",
    )
    references = {}

    def checked(row, name, enabled):
        check_row(row, name, enabled, inputs[name], 1024 if name in edits else 256)
        if name not in references:
            references[name] = row
        reference = references[name]
        require(
            row["output_tokens"] == reference["output_tokens"],
            "full output differs from first OFF warmup: " + name,
        )
        if not enabled:
            require(
                all(row[key] == reference[key] for key in SEMANTIC_COUNTERS),
                "OFF semantic counters differ: " + name,
            )

    for repetition in range(8):
        for index, name in enumerate(names):
            row = warmups[repetition * len(names) + index]
            require(
                type(row.get("sample")) is int
                and row["sample"] == repetition
                and not any(key.startswith("lookup_abba_") for key in row),
                "warmup order/labels differ",
            )
            checked(row, name, False)
    groups, cursor = [], 0
    for group in range(1, 4):
        slots = [dict() for _ in ORDER]
        for name in names:
            for slot, variant in enumerate(ORDER, 1):
                row = raw[cursor]
                cursor += 1
                require(
                    (
                        row.get("scenario"),
                        row.get("sample"),
                        row.get("lookup_abba_group"),
                        row.get("lookup_abba_slot"),
                        row.get("lookup_abba_variant"),
                    )
                    == (name, group - 1, group, slot, variant),
                    "measured order is not adjacent per-case ABBA",
                )
                require(
                    all(
                        type(row.get(k)) is int
                        for k in ("sample", "lookup_abba_group", "lookup_abba_slot")
                    ),
                    "invalid ABBA label types",
                )
                checked(row, name, variant == "on")
                slots[slot - 1][name] = row
        for slot in (1, 2):
            require(
                sum(
                    slots[slot][name]["prompt_lookup_accepted_tokens"] for name in edits
                )
                > 0,
                "ON slot did not exercise lookup acceptance",
            )

        def summed(case_names, key):
            return [sum(rows[name][key] for name in case_names) for rows in slots]

        def rates(case_names, time_key, count_key=None):
            counts = (
                [sum(rows[name][count_key] for name in case_names) for rows in slots]
                if count_key
                else [
                    sum(len(rows[name]["output_tokens"]) for name in case_names)
                    for rows in slots
                ]
            )
            walls = summed(case_names, time_key)
            return [
                1000 * count / wall if wall else None
                for count, wall in zip(counts, walls)
            ]

        groups.append(
            {
                "group": group,
                "primary_edit_request_wall_ms": common.metric(summed(edits, "wall_ms")),
                "all_cases_request_wall_ms": common.metric(summed(names, "wall_ms")),
                "edit_decode_wall_ms_abba": summed(edits, "decode_wall_ms"),
                "edit_decode_output_tokens_abba": summed(edits, "decode_output_tokens"),
                "edit_decode_tokens_per_second_abba": rates(
                    edits, "decode_wall_ms", "decode_output_tokens"
                ),
                "edit_request_tokens_per_second_abba": rates(edits, "wall_ms"),
                "per_case": {
                    name: {
                        "wall_ms": common.metric(
                            [rows[name]["wall_ms"] for rows in slots]
                        ),
                        "actual_output_tokens": len(references[name]["output_tokens"]),
                        "decode_wall_ms_abba": [
                            rows[name]["decode_wall_ms"] for rows in slots
                        ],
                        "counter_abba": [
                            {key: rows[name][key] for key in COUNTERS} for rows in slots
                        ],
                        "lookup_search_wall_ms_abba": [
                            rows[name]["prompt_lookup_search_wall_ms"] for rows in slots
                        ],
                    }
                    for name in names
                },
            }
        )
    primary = common.decision([g["primary_edit_request_wall_ms"] for g in groups])
    per_case = {
        name: common.decision([g["per_case"][name]["wall_ms"] for g in groups])
        for name in names
    }
    unstable = [name for name in names if not per_case[name]["stable"]]
    regressions = [
        name
        for name in controls
        if per_case[name]["stable"]
        and per_case[name]["reduction_pct"] < -per_case[name]["predeclared_floor_pct"]
    ]
    nonpositive = [name for name in edits if not per_case[name]["all_groups_improve"]]
    if unstable or regressions or nonpositive:
        primary.update(
            qualified=False,
            status="inconclusive_case_drift"
            if unstable
            else "control_regression"
            if regressions
            else "edit_case_not_positive",
        )
    decode_walls = [
        statistics.mean(
            (g["edit_decode_wall_ms_abba"][0] + g["edit_decode_wall_ms_abba"][3]) / 2
            if arm == "off"
            else statistics.mean(g["edit_decode_wall_ms_abba"][1:3])
            for g in groups
        )
        for arm in ("off", "on")
    ]
    decode_counts = [
        statistics.mean(
            (
                g["edit_decode_output_tokens_abba"][0]
                + g["edit_decode_output_tokens_abba"][3]
            )
            / 2
            if arm == "off"
            else statistics.mean(g["edit_decode_output_tokens_abba"][1:3])
            for g in groups
        )
        for arm in ("off", "on")
    ]
    return {
        "schema": 1,
        "cohort": cohort,
        "complete": True,
        "raw_validation_pass": True,
        "groups": groups,
        "primary_decision": primary,
        "per_case_decisions": per_case,
        "unstable_cases": unstable,
        "material_control_regressions": regressions,
        "nonpositive_edits": nonpositive,
        "all_cases": list(names),
        "ordinary_controls": list(controls),
        "retained_warmup_requests": len(warmups),
        "retained_measured_requests": len(raw),
        "warmup_wall_ms": sum(row["wall_ms"] for row in warmups),
        "edit_decode_tokens_per_second_off_on": [
            1000 * count / wall if wall else None
            for count, wall in zip(decode_counts, decode_walls)
        ],
        "all_full_arrays_and_off_semantic_counters_match": True,
        "scope": "These fixed greedy B1 edit/control fixtures, pinned model, and measured machine only.",
        "rules": "All edits retained, including unsuccessful tasks. Exact three groups; every edit positive each group; "
        "aggregate gain > max(2%,2*maxA drift), A/B drift <=5% for every case and aggregate; no material "
        "control regression. Actual natural EOS counts retained. Task success never filters timing.",
        "setup_scope": "Model load/startup and eight complete OFF warmup passes excluded from primary but retained. "
        "Equal idle toggles and cache evictions precede each request timer.",
    }


def native_command(worktree, model, out, cohort):
    return [
        str(worktree / "build/engine-tests/backend-benchmark"),
        str(worktree / "build/splash.metallib"),
        str(model),
        "--scenario",
        "lookup-abba",
        "--lookup-cohort",
        cohort,
        "--samples",
        "1",
        "--ane-ffn-share",
        "0",
        "--max-context",
        "16384",
        "--progress",
        str(out / "progress.jsonl"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-revision")
    parser.add_argument("--model-pins", type=Path, default=FIXTURES / "model-pins.json")
    parser.add_argument(
        "--machine-lock",
        type=Path,
        default=Path(tempfile.gettempdir()) / "splash-inference-benchmark.lock",
    )
    parser.add_argument("--timeout", type=float, default=21600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--audit",
        type=Path,
        help="audit saved portable runner output; does not launch inference",
    )
    mode.add_argument(
        "--raw",
        type=Path,
        help="audit native JSON arrays/arithmetic only; no provenance claim",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="optional pinned tokenizer for offline exact task diagnostics",
    )
    args = parser.parse_args()
    try:
        pins = json.loads(args.model_pins.read_text())
        require(
            pins == json.loads((FIXTURES / "model-pins.json").read_text()),
            "model pins differ from fixed fixture model",
        )
        fixture_document(args.cohort)
        if args.raw:
            document = json.loads(args.raw.read_text())
            result = analyze(document, args.cohort, pins)
            result["audit_scope"] = (
                "Raw arrays/work counters/arithmetic only; no source-to-binary or model-content claim."
            )
            if args.tokenizer:
                result["task_diagnostics"] = task_score.score_document(
                    document, FIXTURES / (args.cohort + ".json"), args.tokenizer
                )
            print(json.dumps(result, indent=2))
        elif args.audit:
            manifest = json.loads((args.audit / "manifest.json").read_text())
            saved = manifest["pin"]
            command = native_command(
                Path(saved["build"]["worktree"]),
                Path(saved["model"]["root"]),
                Path(manifest["command"][-1]).parent,
                args.cohort,
            )
            common.audit_integrity(args.audit, command)
            require(
                manifest["model_pins"] == pins, "saved model pins differ from fixture"
            )
            document = json.loads((args.audit / "cohort.json").read_text())
            result = analyze(document, args.cohort, pins, saved["model"]["root"])
            require(
                result == json.loads((args.audit / "summary.json").read_text()),
                "saved summary differs from raw recomputation",
            )
            if args.tokenizer:
                diagnostics = task_score.score_document(
                    document, FIXTURES / (args.cohort + ".json"), args.tokenizer
                )
                require(
                    diagnostics
                    == json.loads((args.audit / "task-success.json").read_text()),
                    "task diagnostics differ",
                )
            print(
                json.dumps(
                    {
                        "audit_pass": True,
                        "task_diagnostics_recomputed": bool(args.tokenizer),
                        "primary_decision": result["primary_decision"],
                    },
                    indent=2,
                )
            )
        else:
            require(
                args.worktree and args.model_root and args.out,
                "--worktree, --model-root and --out required",
            )
            require(
                math.isfinite(args.timeout) and args.timeout > 0,
                "timeout must be positive and finite",
            )
            worktree, model, out = (
                args.worktree.resolve(),
                args.model_root.resolve(),
                args.out.resolve(),
            )
            tokenizer = model / "tokenizer/tokenizer.json"
            require(
                common.digest(tokenizer) == TOKENIZER_SHA256
                and common.digest(model / "tokenizer/chat_template.jinja")
                == TEMPLATE_SHA256,
                "installed tokenizer/template differs from fixed fixtures",
            )
            result = common.run_cohort(
                args,
                native_command(worktree, model, out, args.cohort),
                lambda doc: analyze(doc, args.cohort, pins, model),
                [
                    Path(__file__),
                    Path(common.__file__),
                    Path(task_score.__file__),
                    args.model_pins,
                    FIXTURES / (args.cohort + ".json"),
                ],
            )
            document = json.loads((out / "cohort.json").read_text())
            common.write_json(
                out / "task-success.json",
                task_score.score_document(
                    document, FIXTURES / (args.cohort + ".json"), tokenizer
                ),
            )
            print(json.dumps(result["primary_decision"], indent=2))
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print("lookup regression failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
