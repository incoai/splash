"""Evaluate mixed HTTP throughput with independent prefix and phase-output gates."""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def body_hash(case, ident):
    body = {
        "model": "local/dspine-final-review",
        "messages": case["messages"],
        "max_completion_tokens": case["max_completion_tokens"],
        "temperature": 0,
        "reasoning_effort": "none",
        "seed": 42,
        "cache_salt": "dspine-mixed-" + ident + "-" + case["name"],
    }
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def identity(status, record, arm):
    assert status["ready"] is True and status["metal"]["healthy"] is True
    assert status["transport"]["restarts"] == 0
    assert status["instance"]["pid"] == record["server_pid"]
    assert status["instance"]["id"] == record["initial_status"]["instance"]["id"]
    assert status["identity"]["cache"]["build_id"] == arm["build_id"]
    assert (
        status["identity"]["cache"]["loaded_model_layout_sha256"]
        == arm["loaded_model_layout_sha256"]
    )
    assert status.get("draft_block_size", 8) == arm["rows"]
    assert status["maximum_context_tokens"] == 32768
    assert status["verify_rows"] == "all"
    assert status["identity"]["kv"]["format"] == "int8"
    assert status["memory_plan"]["model"]["state_format"] == "fp32"
    assert status["ane_ffn"]["state"] == "off" and status["ane_ffn"]["share"] == 0


def load_round(root, protocol, ident, job_key, arm):
    job = protocol[job_key]
    receipt_path = root / "results" / (ident + ".receipt.json")
    operating_path = root / "results" / (ident + ".operating.json")
    result_path = root / "http" / ident / "result.json"
    receipt, operating, record = (
        read(receipt_path),
        read(operating_path),
        read(result_path),
    )
    assert (
        receipt["id"] == ident
        and receipt["state"] == "complete"
        and receipt["exit_code"] == 0
    )
    assert receipt["protocol_sha256"] == protocol["_file_sha256"]
    assert receipt["job"] == job
    assert (
        receipt["artifact_sha256"]
        == receipt["artifact_sha256_after"]
        == job["expected_sha256"]
    )
    assert operating["id"] == ident and operating["exit_code"] == 0
    assert operating["command"] == job["command"][3:]
    for key in [
        "operating_gate_passed",
        "no_serious_or_critical_state",
        "no_swap",
        "ac_throughout_measurement",
    ]:
        assert operating.get(key) is True, key + " failed"
    observations = [
        event for event in operating["events"] if event["phase"] == "measurement"
    ]
    assert observations
    assert all(
        event["ac_power"]
        and event["thermal_state"] < 2
        and event["memory"]["swap_usage"] == 0
        for event in observations
    )
    assert (
        record["id"] == ident
        and record["schema_version"] == 1
        and record["operational_pass"] is True
    )
    assert record["rows"] == arm["rows"] and record["model_root"] == arm["model_root"]
    assert (
        record["binary"] == arm["binary"]
        and record["observed_build_id"] == arm["build_id"]
    )
    assert record["cohort_sha256"] == protocol["cohort_sha256"]
    assert record["script_sha256"] == protocol["driver_sha256"]
    assert record["protocol_draft_sha256"] == protocol["workload_draft_sha256"]
    assert (
        record["artifact_sha256"]
        == record["artifact_sha256_after"]
        == arm["bundle_sha256"]
    )
    assert (
        record["server_python_sources_sha256"]
        == protocol["server_python_sources_sha256"]
    )
    assert record["target_sources"] == protocol["target_sources"]
    assert record["shutdown"]["exit"] == 0 and not record["shutdown"]["left_running"]
    identity(record["initial_status"], record, arm)
    assert len(record["scenarios"]) == 2
    assert [phase["scenario"] for phase in record["scenarios"]] == ["cold", "warm"]
    cases = {case["name"]: case for case in protocol["cases"]}
    measured = {}
    for phase in record["scenarios"]:
        label = phase["scenario"]
        assert phase["all_requests_complete"] is True
        assert not phase["request_errors"] and not phase["status_poll_errors"]
        assert phase["status_samples"] > 0
        identity(phase["status_before"], record, arm)
        identity(phase["status_after"], record, arm)
        assert len(phase["requests"]) == 4
        requests = {request["case"]: request for request in phase["requests"]}
        assert set(requests) == set(cases)
        total = 0
        normalized = {}
        cache_ok = True
        for name, request in requests.items():
            case = cases[name]
            assert request["scenario"] == label
            assert request["scheduled_arrival_seconds"] == case["arrival_seconds"]
            assert request["max_completion_tokens"] == case["max_completion_tokens"]
            assert request["messages_sha256"] == case["messages_sha256"]
            assert request["request_body_sha256"] == body_hash(case, ident)
            assert request["source_prompt_tokens"] == case["source_prompt_tokens"]
            assert isinstance(request["text"], str)
            assert (
                request["text_sha256"]
                == hashlib.sha256(request["text"].encode()).hexdigest()
            )
            assert request["text"] == (
                request["assistant_message"].get("content") or ""
            )
            count = request["usage"]["completion_tokens"]
            assert type(count) is int and 0 <= count <= case["max_completion_tokens"]
            assert positive(request["client_wall_seconds"])
            assert request["done_seconds"] >= request["sent_seconds"] >= 0
            assert math.isclose(
                request["done_seconds"] - request["sent_seconds"],
                request["client_wall_seconds"],
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
            latency, cache = (
                request["metrics"]["request_latency"],
                request["metrics"]["cache"],
            )
            matched = cache["matched_tokens"]
            cache_ok &= matched == 0 if label == "cold" else matched > 0
            stream = latency.get("stream_tokens_per_second")
            assert stream is None or positive(stream)
            total += count
            normalized[name] = {
                "_text": request["text"],
                "text_sha256": request["text_sha256"],
                "completion_tokens": count,
                "finish_reason": request["finish_reason"],
                "client_wall_ms": 1000 * request["client_wall_seconds"],
                "stream_ms_per_token": 1000 / stream if stream is not None else None,
                "request_latency": latency,
                "usage": request["usage"],
                "cache": cache,
                "prefill": request["metrics"].get("prefill"),
                "decode": request["metrics"].get("decode"),
                "sent_seconds": request["sent_seconds"],
                "done_seconds": request["done_seconds"],
            }
        assert cache_ok == phase["cache_contract_passed"]
        assert total == phase["completion_tokens"] and total > 0
        assert positive(phase["client_wall_seconds"])
        assert math.isclose(
            phase["aggregate_completion_tokens_per_second"],
            total / phase["client_wall_seconds"],
            rel_tol=1e-12,
        )
        assert phase["client_wall_seconds"] >= max(
            row["done_seconds"] for row in requests.values()
        )
        before_widths = phase["status_before"]["scheduler"]["decode_batches_by_width"]
        after_widths = phase["status_after"]["scheduler"]["decode_batches_by_width"]
        widths = {
            name: after_widths[name] - before_widths[name] for name in before_widths
        }
        assert widths == phase["decode_batches_by_width_delta"]
        assert all(type(value) is int and value >= 0 for value in widths.values())
        measured[label] = {
            "milliseconds_per_returned_token": 1000
            * phase["client_wall_seconds"]
            / total,
            "client_wall_ms": 1000 * phase["client_wall_seconds"],
            "completion_tokens": total,
            "cache_contract_passed": cache_ok,
            "requests": normalized,
            "decode_batches_by_width_delta": widths,
            "scheduler_before": phase["status_before"]["scheduler"],
            "scheduler_after": phase["status_after"]["scheduler"],
            "model_timing_before": phase["status_before"]["model_timing"],
            "model_timing_after": phase["status_after"]["model_timing"],
        }
    parity = cold_warm_comparison(measured, record["cold_warm_comparison"], cases)
    measured["warm"]["cold_warm_comparison"] = parity
    return measured, {
        "job": ident,
        "receipt_sha256": sha(receipt_path),
        "operating_sha256": sha(operating_path),
        "result_sha256": sha(result_path),
        "observed_build_id": record["observed_build_id"],
        "rows": record["rows"],
        "model_root": record["model_root"],
        "cold_warm_comparison": parity,
        "driver_cold_warm_summary_validated": True,
    }


def signature(request):
    return (
        request["_text"],
        request["completion_tokens"],
        request["finish_reason"],
    )


def cold_warm_comparison(measured, reported, cases):
    """Derive prefix parity from full text and counts, independently of flags."""
    assert set(reported) == set(cases), "Cold/warm summary case set differs"
    comparisons = {}
    for name in cases:
        cold, warm = (measured[phase]["requests"][name] for phase in ["cold", "warm"])
        comparisons[name] = {
            "text_matches": cold["_text"] == warm["_text"],
            "completion_count_matches": cold["completion_tokens"]
            == warm["completion_tokens"],
            "finish_reason_matches": cold["finish_reason"] == warm["finish_reason"],
            "all_match": signature(cold) == signature(warm),
        }
        for key in ["text_matches", "completion_count_matches"]:
            assert reported[name][key] is comparisons[name][key], (
                "Driver cold/warm summary differs from raw outputs"
            )
    return comparisons


def public_request(request):
    return {key: value for key, value in request.items() if not key.startswith("_")}


def endpoint(
    compare,
    values,
    output_match,
    operating_ok,
    cache_ok=True,
    cold_warm_ok=None,
    cross_group_output_ok=True,
):
    gates = {
        "output_text_and_count_match": output_match,
        "cache_contract_passed": cache_ok,
        "cold_warm_output_match": cold_warm_ok,
        "cross_group_output_match": cross_group_output_ok,
    }
    if any(value is None for value in values):
        return {
            "available": False,
            "reason": "At least one request has no stream-decode interval.",
            **gates,
        }
    assert all(positive(value) for value in values)
    paired = compare(*values)
    quiet = paired["noise"] <= 0.05
    effect = abs(paired["regression"]) > paired["bound"]
    measurement = (
        operating_ok
        and output_match
        and cache_ok
        and quiet
        and cold_warm_ok is not False
        and cross_group_output_ok
    )
    return {
        "available": True,
        "abba": paired,
        **gates,
        "quiet": quiet,
        "effect_gate_passed": effect,
        "measurement_qualified": measurement,
        "qualified": measurement and effect,
        "candidate_rate_change_percent": 100
        * (paired["baseline"] / paired["candidate"] - 1),
        "candidate_time_change_percent": 100 * paired["regression"],
        "direction": "candidate faster"
        if paired["regression"] < 0
        else "control faster",
    }


def overall(groups, phase, case=None, metric=None):
    values = []
    for group in groups:
        if phase not in group.get("phases", {}):
            continue
        value = (
            group["phases"][phase]["primary"]
            if case is None
            else group["phases"][phase]["cases"][case][metric]
        )
        if value.get("available"):
            values.append(value)
    complete = len(values) == 3
    directions = sorted({value["direction"] for value in values})
    qualified = complete and all(value["qualified"] for value in values)
    return {
        "groups_with_endpoint": len(values),
        "all_three_qualified": qualified,
        "directions": directions,
        "consistent_direction": len(directions) == 1,
        "reportable_effect": qualified and len(directions) == 1,
        "candidate_rate_change_percent_by_group": [
            value["candidate_rate_change_percent"] for value in values
        ],
        "candidate_time_change_percent_by_group": [
            value["candidate_time_change_percent"] for value in values
        ],
    }


def phase_output_identity(loaded, case_names):
    result = {}
    for phase in ["cold", "warm"]:
        result[phase] = {}
        for name in case_names:
            requests = {
                key: value[0][phase]["requests"][name] for key, value in loaded.items()
            }
            result[phase][name] = {
                "registered_rounds": 12,
                "observed_rounds": len(requests),
                "all_12_round_outputs_match": len(requests) == 12
                and len({signature(row) for row in requests.values()}) == 1,
                "text_sha256_by_round": {
                    key: row["text_sha256"] for key, row in requests.items()
                },
                "completion_tokens_by_round": {
                    key: row["completion_tokens"] for key, row in requests.items()
                },
                "finish_reason_by_round": {
                    key: row["finish_reason"] for key, row in requests.items()
                },
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = read(args.protocol)
    assert sha(Path(__file__)) == protocol["analyzer_sha256"]
    helper = Path(__file__).with_name("abba.py")
    assert sha(helper) == protocol["abba_sha256"]
    module = importlib.util.spec_from_file_location("frozen_mixed_abba", helper)
    abba = importlib.util.module_from_spec(module)
    module.loader.exec_module(abba)
    protocol["_file_sha256"] = sha(args.protocol)
    assert protocol["analysis_contract_version"] == 2
    assert protocol["groups"] == 3 and protocol["case_names"] == [
        case["name"] for case in protocol["cases"]
    ]
    expected_keys = [
        f"g{group}_{suffix}"
        for group in range(1, 4)
        for suffix in ["a1", "b1", "b2", "a2"]
    ]
    assert protocol["job_keys"] == expected_keys
    loaded, load_failures = {}, {}
    for key in expected_keys:
        ident = protocol[key]["id"]
        arm = protocol["arms"]["control" if key.endswith(("a1", "a2")) else "candidate"]
        try:
            loaded[key] = load_round(args.root, protocol, ident, key, arm)
        except (OSError, AssertionError, KeyError, ValueError) as error:
            load_failures[key] = ident + ": " + (str(error) or type(error).__name__)
    global_outputs = phase_output_identity(loaded, protocol["case_names"])
    groups = []
    for group_number in range(1, 4):
        problems, rounds, sources = [], [], []
        ids = []
        for suffix in ["a1", "b1", "b2", "a2"]:
            key = f"g{group_number}_{suffix}"
            ident = protocol[key]["id"]
            ids.append(ident)
            if key in loaded:
                measured, source = loaded[key]
                rounds.append(measured)
                sources.append(source)
            else:
                problems.append(load_failures[key])
        group = {
            "group": group_number,
            "rounds": ids,
            "sources": sources,
            "problems": problems,
            "phases": {},
        }
        if len(rounds) == 4:
            group["cold_warm_comparison_by_round"] = [
                value["warm"]["cold_warm_comparison"] for value in rounds
            ]
            for phase in ["cold", "warm"]:
                cases = {}
                for name in protocol["case_names"]:
                    rows = [round[phase]["requests"][name] for round in rounds]
                    same = len({signature(row) for row in rows}) == 1
                    prefix_ok = (
                        all(
                            value["warm"]["cold_warm_comparison"][name]["all_match"]
                            for value in rounds
                        )
                        if phase == "warm"
                        else None
                    )
                    all_groups_same = global_outputs[phase][name][
                        "all_12_round_outputs_match"
                    ]
                    cache_ok = all(
                        row["cache"]["matched_tokens"] == 0
                        if phase == "cold"
                        else row["cache"]["matched_tokens"] > 0
                        for row in rows
                    )
                    cases[name] = {
                        "text_sha256_by_round": [row["text_sha256"] for row in rows],
                        "completion_tokens_by_round": [
                            row["completion_tokens"] for row in rows
                        ],
                        "finish_reason_by_round": [
                            row["finish_reason"] for row in rows
                        ],
                        "client_wall": endpoint(
                            abba.compare,
                            [row["client_wall_ms"] for row in rows],
                            same,
                            not problems,
                            cache_ok,
                            prefix_ok,
                            all_groups_same,
                        ),
                        "stream_decode": endpoint(
                            abba.compare,
                            [row["stream_ms_per_token"] for row in rows],
                            same,
                            not problems,
                            cache_ok,
                            prefix_ok,
                            all_groups_same,
                        ),
                        "descriptive_requests_by_round": [
                            public_request(row) for row in rows
                        ],
                    }
                same = all(
                    case["client_wall"]["output_text_and_count_match"]
                    for case in cases.values()
                )
                cache_ok = all(
                    round[phase]["cache_contract_passed"] for round in rounds
                )
                prefix_ok = (
                    all(
                        value["warm"]["cold_warm_comparison"][name]["all_match"]
                        for value in rounds
                        for name in protocol["case_names"]
                    )
                    if phase == "warm"
                    else None
                )
                all_groups_same = all(
                    global_outputs[phase][name]["all_12_round_outputs_match"]
                    for name in protocol["case_names"]
                )
                group["phases"][phase] = {
                    "primary": endpoint(
                        abba.compare,
                        [
                            round[phase]["milliseconds_per_returned_token"]
                            for round in rounds
                        ],
                        same,
                        not problems,
                        cache_ok,
                        prefix_ok,
                        all_groups_same,
                    ),
                    "cases": cases,
                    "descriptive_widths_by_round": [
                        round[phase]["decode_batches_by_width_delta"]
                        for round in rounds
                    ],
                    "descriptive_scheduler_by_round": [
                        {
                            "before": round[phase]["scheduler_before"],
                            "after": round[phase]["scheduler_after"],
                        }
                        for round in rounds
                    ],
                    "descriptive_model_timing_by_round": [
                        {
                            "before": round[phase]["model_timing_before"],
                            "after": round[phase]["model_timing_after"],
                        }
                        for round in rounds
                    ],
                    "scenario_client_wall_ms_by_round": [
                        round[phase]["client_wall_ms"] for round in rounds
                    ],
                    "scenario_completion_tokens_by_round": [
                        round[phase]["completion_tokens"] for round in rounds
                    ],
                }
        groups.append(group)
    endpoints = {phase: overall(groups, phase) for phase in ["cold", "warm"]}
    cases = {
        phase: {
            name: {
                metric: overall(groups, phase, name, metric)
                for metric in ["client_wall", "stream_decode"]
            }
            for name in protocol["case_names"]
        }
        for phase in ["cold", "warm"]
    }
    protocol.pop("_file_sha256")
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": sha(args.protocol),
        "protocol": protocol,
        "groups": groups,
        "global_phase_output_identity": global_outputs,
        "all_warm_prefix_outputs_match_cold": len(loaded) == 12
        and all(
            value[0]["warm"]["cold_warm_comparison"][name]["all_match"]
            for value in loaded.values()
            for name in protocol["case_names"]
        ),
        "warm_prefix_output_failures": {
            key: {
                name: comparison
                for name, comparison in value[0]["warm"]["cold_warm_comparison"].items()
                if not comparison["all_match"]
            }
            for key, value in loaded.items()
            if any(
                not comparison["all_match"]
                for comparison in value[0]["warm"]["cold_warm_comparison"].values()
            )
        },
        "primary_endpoints": endpoints,
        "case_endpoints": cases,
        "all_scheduled_rounds_valid": all(not group["problems"] for group in groups),
        "scope": (
            "All three ABBA groups and every fixed request retained. Cold and warm client makespan "
            "per actually returned completion token have separate output, cache, operating, noise and effect gates. "
            "Every phase requires exact output identity across all 12 rounds. Every warm case also requires "
            "within-round cold/warm text, completion-count and finish-reason parity; warm aggregate requires every case. "
            "Cold qualification is independent of this prefix-output gate. Parity is recomputed from raw HTTP payloads. "
            "Per-case client wall and stream decode use their own gates. Actual width, queue, prefill and cache "
            "statistics are descriptive. This report contains generated-text hashes and counts, with raw text "
            "retained only in the original per-round HTTP results."
        ),
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "primary_endpoints": endpoints,
                "all_warm_prefix_outputs_match_cold": report[
                    "all_warm_prefix_outputs_match_cold"
                ],
                "warm_prefix_output_failure_rounds": len(
                    report["warm_prefix_output_failures"]
                ),
                "valid_rounds": len(
                    [source for group in groups for source in group["sources"]]
                ),
                "problem_count": sum(len(group["problems"]) for group in groups),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
