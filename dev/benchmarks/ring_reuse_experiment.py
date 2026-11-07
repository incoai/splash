#!/usr/bin/env python3
"""Run or audit the fixed draft-ring ABBA experiments, retaining every request.

The one-token cohort measures restore/TTFT latency. The cap-128 cohort additionally
requires at least 64 actual outputs in every case before qualification. A raw JSON
audit checks arithmetic and full token arrays, not source/model provenance.
"""

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path

import experiment_common as common

ORDER = ("off", "on", "on", "off")
ROWS = (1, 8, 16, 32)
NAMES = tuple("ring_short_r" + str(n) for n in ROWS)
RING_BYTES = 40 << 20
COMPOSITE_BYTES = 160038912
MODEL_PINS = Path(__file__).parent / "fixtures/ring/model-pins.json"
GEOMETRY = {
    "prefill_rows": 2048,
    "verify_rows": 8,
    "batch_width": 4,
    "kv_block_tokens": 32,
}
DECODE = (
    "decode_batches",
    "decode_output_tokens",
    "drafted_tokens",
    "accepted_draft_tokens",
)
COUNTERS = (
    "matched_tokens",
    "target_prefill_rows",
    "draft_context_rows",
    "draft_context_rows_avoided",
    "draft_state_restore_skipped",
    "draft_ring_rebuild_rows",
    "draft_ring_cache_hits",
    "draft_ring_cache_builds",
    "draft_ring_cache_refusals",
    "draft_ring_cache_evictions",
    "draft_ring_cache_bytes",
    "draft_ring_cache_before_bytes",
    "draft_ring_cache_peak_observed_bytes",
    "junction_materializations",
    "junction_materialization_failures",
    "state_allocated_before_bytes",
    "state_allocated_after_bytes",
    "state_allocated_peak_observed_bytes",
    "cached_state_bytes",
    "state_entries",
    "state_evictions",
)


require = common.require
token_hash = common.token_hash
metric = common.metric


def number(value, label, positive=False):
    require(
        type(value) in (float, int)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        "invalid number: " + label,
    )
    return value


def exact(document, expected, label):
    for key, value in expected.items():
        require(
            document.get(key) == value and type(document.get(key)) is type(value),
            label + ": " + key,
        )


def fixture_inputs():
    def prompt(length, salt):
        state, result = salt ^ 0x9E3779B97F4A7C15, []
        for _ in range(length):
            state = (state * 6364136223846793005 + 1442695040888963407) & (
                (1 << 64) - 1
            )
            result.append(100 + ((state >> 17) % 200000))
        return result

    prefix = prompt(4096, 0x53484F5254524553)
    result = {"ring_short_seed": prefix + [200100]}
    for i, (name, rows) in enumerate(zip(NAMES, ROWS)):
        tail = prompt(rows, 0x5441494C ^ rows)
        tail[0] = 200101 + i
        result[name] = prefix + tail
        result[name + "_cold"] = result[name]
    return result


def verdict(values):
    a, b = (
        statistics.mean(v["baseline"] for v in values),
        statistics.mean(v["candidate"] for v in values),
    )
    drift = max(v["a_endpoint_symmetric_drift_pct"] for v in values)
    spread = max(v["b_symmetric_spread_pct"] for v in values)
    reduction, floor = 100 * (1 - b / a), max(2.0, 2 * drift)
    stable = drift <= 5 and spread <= 5
    positive = all(v["reduction_pct"] > 0 for v in values)
    qualified = len(values) == 3 and stable and positive and reduction > floor
    return dict(
        baseline_ms=a,
        candidate_ms=b,
        reduction_pct=reduction,
        max_a_endpoint_symmetric_drift_pct=drift,
        max_b_symmetric_spread_pct=spread,
        predeclared_floor_pct=floor,
        all_groups_improve=positive,
        drift_within5pct=stable,
        qualified=qualified,
    )


def analyze(document, cap, model_root=None, layout=None):
    """Validate fixed fixtures, every retained row, charged work and ABBA arithmetic."""
    accounting = document.get("ring_state_accounting", "inclusive_state_v1")
    require(
        accounting in ("inclusive_state_v1", "exclusive_pool_v2"),
        "unknown state accounting contract",
    )
    warmups = 96 if cap == 128 else 1280
    exact(
        document,
        {
            "schema_version": 2,
            "experimental_draft_ring_reuse": False,
            "ring_short_prefix_tokens": 4096,
            "ring_short_output_tokens": cap,
            "ring_short_replay_rows": list(ROWS),
            "max_context_tokens": 16384,
            "geometry": GEOMETRY,
            "skipped_context_lengths": [],
            "performance_pass": True,
            "performance_failures": [],
            "ane_ffn_bytes": 0,
            "ring_runtime_instrumented": False,
            "ring_abba_enabled": True,
            "ring_abba_groups": 3,
            "ring_abba_order": list(ORDER),
            "ring_compilation_profile": "production_O3_flags",
            "ring_thermal_warmup_requests": warmups,
            "ring_abba_replicates": 8,
            "ring_conditioned_cohort": "off96_generation128_workload8_v1"
            if cap == 128
            else "off1280_workload8_v1",
            "ring_thermal_warmup_in_primary": False,
            "ring_toggle_once_per_four_request_workload": True,
            "ring_toggle_in_primary": True,
        },
        "root invariant",
    )
    require(
        document.get("ane_ffn_share") == 0
        and type(document.get("ane_ffn_share")) in (int, float),
        "ANE differs",
    )
    require(
        document.get("no_ane_context_tokens", 0) >= 16384, "memory plan lacks context"
    )
    identity = document.get("identity", {})
    require(
        identity.get("device")
        and identity.get("loaded_model_layout_sha256")
        and document.get("build_id"),
        "missing identity",
    )
    if model_root is not None:
        require(
            Path(identity.get("model_root", "")).resolve()
            == Path(model_root).resolve(),
            "model root differs",
        )
    if layout is not None:
        require(
            identity["loaded_model_layout_sha256"] == layout, "model layout differs"
        )
    if cap == 128:
        exact(
            document,
            {"ring_seed_output_tokens": 1, "ring_generation_minimum_actual_tokens": 64},
            "generation metadata",
        )
    raw, warm = (
        document.get("measurements", []),
        document.get("ring_thermal_warmup_measurements", []),
    )
    require(
        len(raw) == 389 and len(warm) == warmups, "incomplete or extra measurements"
    )
    seed, cold = raw[0], raw[-4:]
    require(
        seed.get("scenario") == "ring_short_seed"
        and [r.get("scenario") for r in cold] == [n + "_cold" for n in NAMES],
        "seed/cold order differs",
    )
    inputs = fixture_inputs()

    def row_check(row, name, enabled=False, index=None):
        require(row.get("scenario") == name, "row scenario differs")
        for key in COUNTERS + ("input_token_hash", "sample"):
            require(
                type(row.get(key)) is int and row[key] >= 0, "invalid counter: " + key
            )
        expected_input = inputs[name]
        require(
            row.get("input_tokens") == expected_input
            and all(type(t) is int for t in row["input_tokens"])
            and row.get("prompt_tokens") == len(expected_input)
            and row["input_token_hash"] == token_hash(expected_input),
            "full input fixture differs: " + name,
        )
        for key in ("wall_ms", "ttft_ms", "prefill_gpu_ms", "decode_gpu_ms"):
            number(row.get(key), key, positive=key != "decode_gpu_ms")
        require(row["ttft_ms"] <= row["wall_ms"] + 0.1, "TTFT exceeds wall")
        tokens, budget = (
            row.get("output_tokens"),
            1 if name == "ring_short_seed" else cap,
        )
        require(
            isinstance(tokens, list)
            and 0 < len(tokens) <= budget
            and all(type(t) is int and 0 <= t < 248320 for t in tokens),
            "invalid full outputs",
        )
        require(
            not any(t in (248044, 248046) for t in tokens[:-1])
            and (len(tokens) == budget or tokens[-1] in (248044, 248046)),
            "invalid EOS truncation",
        )
        if cap == 128 or "requested_output_tokens" in row:
            exact(
                row,
                {
                    "requested_output_tokens": budget,
                    "actual_output_tokens": len(tokens),
                },
                "output counts",
            )
            for key in DECODE:
                require(
                    type(row.get(key)) is int and row[key] >= 0,
                    "invalid decode counter",
                )
            number(row.get("decode_wall_ms"), "decode_wall_ms")
            require(
                row["decode_wall_ms"] <= row["wall_ms"] + 0.1,
                "decode wall exceeds wall",
            )
            cycles = row["decode_batches"]
            if cycles:
                require(
                    0 < row["decode_output_tokens"] <= len(tokens)
                    and row["decode_gpu_ms"] > 0
                    and row["decode_wall_ms"] > 0,
                    "decode work missing",
                )
            else:
                require(
                    row["decode_output_tokens"]
                    == row["decode_gpu_ms"]
                    == row["decode_wall_ms"]
                    == 0
                    and len(tokens) == 1,
                    "zero-decode outputs differ",
                )
            require(
                0
                <= row["accepted_draft_tokens"]
                <= row["drafted_tokens"]
                <= 7 * cycles,
                "acceptance count invalid",
            )
        if budget == 1:
            require(
                len(tokens) == 1 and row["decode_gpu_ms"] == 0,
                "one-token request did decode work",
            )
        restored, hit = index is not None, enabled and index is not None and index > 0
        exact(
            row,
            {
                "state_evictions": 0,
                "junction_materializations": 0,
                "junction_materialization_failures": 0,
                "draft_state_restore_skipped": 0,
                "draft_ring_rebuild_rows": 2048 if restored and not hit else 0,
                "draft_ring_cache_hits": int(hit),
                "draft_ring_cache_builds": int(enabled and restored and index == 0),
                "draft_ring_cache_refusals": 0,
                "draft_ring_cache_evictions": 0,
                "draft_ring_cache_before_bytes": RING_BYTES if hit else 0,
                "draft_ring_cache_bytes": RING_BYTES if enabled and restored else 0,
                "draft_ring_cache_peak_observed_bytes": RING_BYTES
                if enabled and restored
                else 0,
            },
            "ring work/memory",
        )
        if restored:
            exact(
                row,
                {
                    "cache_status": "prefix_hit",
                    "matched_tokens": 4096,
                    "target_prefill_rows": ROWS[index],
                    "draft_context_rows": ROWS[index],
                    "state_entries": 1,
                    "cached_state_bytes": COMPOSITE_BYTES,
                    "ring_short_probe_kv_tokens": 4096,
                    "cold_output_match": True,
                },
                "restored row",
            )
            for which in ("before", "after", "peak_observed"):
                ring_key = {
                    "before": "before_",
                    "after": "",
                    "peak_observed": "peak_observed_",
                }[which]
                require(
                    row["state_allocated_" + which + "_bytes"]
                    == seed["state_allocated_after_bytes"]
                    + (
                        row["draft_ring_cache_" + ring_key + "bytes"]
                        if accounting == "inclusive_state_v1"
                        else 0
                    ),
                    "allocation changed outside ring",
                )
            require(
                tokens == cold[index]["output_tokens"], "full warm/cold outputs differ"
            )
            if cap == 128:
                require(
                    all(row[k] == cold[index][k] for k in DECODE),
                    "warm/cold decode work differs",
                )
        else:
            exact(
                row,
                {
                    "sample": 0,
                    "cache_status": "miss",
                    "matched_tokens": 0,
                    "target_prefill_rows": len(expected_input),
                },
                "cold row",
            )
            require(
                not any(k.startswith("ring_abba_") for k in row)
                and not row.get("ring_short_probe_kv_tokens"),
                "cold row carries cohort labels",
            )

    row_check(seed, "ring_short_seed")
    exact(
        seed, {"state_entries": 1, "cached_state_bytes": COMPOSITE_BYTES}, "seed state"
    )
    for i, row in enumerate(cold):
        row_check(row, NAMES[i] + "_cold")
    for repetition in range(warmups // 4):
        for i, name in enumerate(NAMES):
            row = warm[4 * repetition + i]
            row_check(row, name, index=i)
            exact(
                row,
                {"sample": repetition, "ring_bytes_before_toggle": 0},
                "warmup labels",
            )
            require(
                not any(k.startswith("ring_abba_") for k in row)
                and "ring_workload_wall_ms" not in row,
                "warmup carries measurement labels",
            )
            toggle = number(row.get("ring_toggle_wall_ms"), "warmup toggle")
            require(i == 0 or toggle == 0, "toggle inside warmup workload")

    groups, cursor, prior_bytes = [], 1, 0
    for group in range(1, 4):
        arms = []
        for slot, mode in enumerate(ORDER, 1):
            workloads = []
            for replicate in range(1, 9):
                workload, charged = [], 0.0
                for i, name in enumerate(NAMES):
                    row = raw[cursor]
                    cursor += 1
                    row_check(row, name, mode == "on", i)
                    exact(
                        row,
                        {
                            "sample": group - 1,
                            "ring_abba_group": group,
                            "ring_abba_slot": slot,
                            "ring_abba_variant": mode,
                            "ring_abba_replicate": replicate,
                            "ring_bytes_before_toggle": prior_bytes if i == 0 else 0,
                        },
                        "ABBA order/reset",
                    )
                    toggle = number(row.get("ring_toggle_wall_ms"), "toggle")
                    require(i == 0 or toggle == 0, "toggle inside measured workload")
                    charged += row["wall_ms"] + toggle
                    recorded = number(
                        row.get("ring_workload_wall_ms"),
                        "charged cumulative wall",
                        True,
                    )
                    require(
                        abs(recorded - charged) <= 0.05,
                        "toggle, fill or request omitted from charged wall",
                    )
                    workload.append(row)
                prior_bytes = RING_BYTES if mode == "on" else 0
                workloads.append(workload)
            arms.append(workloads)
        primary = metric(
            [
                statistics.mean(w[-1]["ring_workload_wall_ms"] for w in arm)
                for arm in arms
            ]
        )
        per_case = {
            name: metric(
                [statistics.mean(w[i]["wall_ms"] for w in arm) for arm in arms]
            )
            for i, name in enumerate(NAMES)
        }
        totals = {
            key: [sum(row[key] for w in arm for row in w) for arm in arms]
            for key in (
                "draft_ring_cache_builds",
                "draft_ring_cache_hits",
                "draft_ring_rebuild_rows",
            )
        }
        exact(
            totals,
            {
                "draft_ring_cache_builds": [0, 8, 8, 0],
                "draft_ring_cache_hits": [0, 24, 24, 0],
                "draft_ring_rebuild_rows": [65536, 16384, 16384, 65536],
            },
            "complete arm work",
        )
        groups.append(
            dict(
                group=group,
                primary=primary,
                per_case_wall=per_case,
                ring_work_per_arm_totals=totals,
            )
        )
    decision = verdict([g["primary"] for g in groups])
    cases = {
        name: verdict([g["per_case_wall"][name] for g in groups]) for name in NAMES
    }
    unstable = [n for n in NAMES if not cases[n]["drift_within5pct"]]
    insufficient = [
        n
        for n, row in zip(NAMES, cold)
        if cap == 128 and len(row["output_tokens"]) < 64
    ]
    decision["qualified"] &= not unstable and not insufficient
    decision["unstable_individual_wall_cases"] = unstable
    decision["insufficient_generation_cases"] = insufficient
    return dict(
        complete=True,
        cap=cap,
        full_arrays_cold_references_and_work_counters_match=True,
        groups=groups,
        primary_decision=decision,
        per_case_wall_decisions=cases,
        actual_outputs={n: len(r["output_tokens"]) for n, r in zip(NAMES, cold)},
        retained_ring_peak_observed_bytes=RING_BYTES,
        shared_seed_wall_ms=seed["wall_ms"],
        conditioning_requests=warmups,
        conditioning_wall_ms=sum(r["wall_ms"] + r["ring_toggle_wall_ms"] for r in warm),
        measured_requests=384,
        no_rows_excluded=True,
        scope="Mean of eight complete four-request workloads per ABBA arm, charging reset and first fill. One-token is latency, not decode throughput. Model/source provenance must be verified separately.",
    )


def native_command(worktree, model_root, out, cap):
    return [
        str(worktree / "build/engine-tests/backend-benchmark"),
        str(worktree / "build/splash.metallib"),
        str(model_root),
        "--scenario",
        "ring-generation-abba" if cap == 128 else "ring-short-abba",
        "--samples",
        "1",
        "--ane-ffn-share",
        "0",
        "--max-context",
        "16384",
        "--progress",
        str(out / "progress.jsonl"),
    ]


def audit_results(directory, cap):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(
        manifest["model_pins"] == json.loads(MODEL_PINS.read_text()),
        "saved pinned model/tokenizer fixture differs",
    )
    build = manifest["pin"]["build"]
    # Reconstruct every native option; paths are the original saved provenance,
    # not paths that must still exist on the reviewing machine.
    original_out = Path(
        json.loads((directory / "receipt.json").read_text())["stdout"]["path"]
    ).parent
    command = native_command(
        Path(build["worktree"]),
        Path(manifest["pin"]["model"]["root"]),
        original_out,
        cap,
    )
    common.audit_integrity(directory, command)
    result = analyze(
        json.loads((directory / "cohort.json").read_text()),
        cap,
        manifest["pin"]["model"]["root"],
        manifest["pin"]["model"]["loaded_model_layout_sha256"],
    )
    require(
        result == json.loads((directory / "summary.json").read_text()),
        "summary differs from recomputation",
    )
    return dict(
        saved_provenance_consistent=True,
        summary=result,
        scope="Saved model content, before/after artifact identities, commands and raw results agree; hashes do not prove source-to-binary compilation.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--raw", type=Path, help="CPU-only native JSON arithmetic/token audit"
    )
    mode.add_argument(
        "--audit-results",
        type=Path,
        help="CPU-only saved run provenance and result audit",
    )
    parser.add_argument("--cap", type=int, choices=(1, 128), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--model-pins", type=Path, default=MODEL_PINS)
    parser.add_argument(
        "--machine-lock",
        type=Path,
        default=Path(tempfile.gettempdir()) / "splash-inference-benchmark.lock",
    )
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--expected-revision")
    args = parser.parse_args()
    if args.raw:
        report = analyze(json.loads(args.raw.read_text()), args.cap)
        report["audit_scope"] = (
            "raw arithmetic/token/work validation only; no compilation or model-content claim"
        )
    elif args.audit_results:
        require(args.out is None, "--out is not used with --audit-results")
        report = audit_results(args.audit_results, args.cap)
    else:
        if (
            not all((args.worktree, args.model_root, args.out))
            or not math.isfinite(args.timeout)
            or args.timeout <= 0
        ):
            parser.error(
                "live mode requires --worktree --model-root --out and positive timeout"
            )
        args.worktree, args.model_root, args.out = (
            args.worktree.resolve(),
            args.model_root.resolve(),
            args.out.resolve(),
        )
        pins = json.loads(args.model_pins.read_text())
        require(
            pins == json.loads(MODEL_PINS.read_text()),
            "pinned model/tokenizer fixture differs",
        )
        layout = pins["loaded_model_layout_sha256"]
        report = common.run_cohort(
            args,
            native_command(args.worktree, args.model_root, args.out, args.cap),
            lambda document: analyze(document, args.cap, args.model_root, layout),
            extra_files=(Path(__file__), Path(common.__file__), args.model_pins),
        )
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.raw and args.out:
        with args.out.open("x") as stream:
            stream.write(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
