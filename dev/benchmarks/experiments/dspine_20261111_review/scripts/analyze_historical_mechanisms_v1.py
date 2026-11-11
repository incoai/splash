#!/usr/bin/env python3
"""Describe completed historical31 work with its actual caps and returned lengths."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "historical31-dflash8-vs-review16-v1"
WORK_HELPER_SHA256 = "0bef7457b4b49a45442d272e8f30ad7256a3ccd3c601854ad0a8286b0fc96e17"
SLOWER = {"sql_latest", "technical_memory", "linear_equations", "french_cache_explanation",
          "arabic_atomicity", "japanese_lru_notes"}
INPUTS = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw(root, relative):
    data = (root / relative).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(relative not in INPUTS or INPUTS[relative] == digest, "input changed: " + relative)
    INPUTS[relative] = digest
    return data


def read(root, relative):
    return json.loads(raw(root, relative))


def summarize(work, records):
    result = work.summarize(records)
    result["returned_lengths"] = sorted({record["returned"] for record in records})
    result["registered_output_caps"] = sorted({record["output_cap"] for record in records})
    result["censoring"]["requests_below_cap"] = sum(record["returned"] < record["output_cap"] for record in records)
    return result


def evaluate(root):
    INPUTS.clear()
    helper = "scripts/analyze_agent_mechanisms_v1.py"
    require(hashlib.sha256(raw(root, helper)).hexdigest() == WORK_HELPER_SHA256, "generic work helper changed")
    spec = importlib.util.spec_from_file_location("frozen_request_work", root / helper)
    work = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(work)
    # Only its cap-parameterized request_work/summarize/ratios functions are
    # reused. Its agent12 evaluator and fixed-cohort assumptions are not run.
    protocol = read(root, "notes/" + PREFIX + ".protocol.json")
    comparison = read(root, "results/" + PREFIX + ".comparison.json")
    review = read(root, "results/" + PREFIX + ".review.json")
    require(comparison["protocol"] == protocol and protocol["groups"] == 3 and protocol["cases"] == 31 and
            protocol["samples_per_round"] == 1 and protocol["clone_width"] == 1 and
            not protocol.get("exclude_primary_cases"), "unexpected complete primary cohort")
    require(review["status"] == "complete" and review["rounds_validated"] == 12 and
            review["protocol_sha256"] == INPUTS["notes/" + PREFIX + ".protocol.json"], "review provenance differs")
    cases = {case["name"]: case for case in protocol["registered_cases"]}
    require(len(cases) == 31 and "count_secondary" in cases, "registered cases differ")
    expected = {source["job"]: source["sha256"] for group in comparison["groups"] for source in group["sources"]}
    expected_ids = [f"{PREFIX}-g{group}-{suffix}" for group in range(1, 4) for suffix in ["a1", "b1", "b2", "a2"]]
    require(set(expected) == set(expected_ids), "complete source rounds differ")
    by_case = {name: {"control": [], "candidate": []} for name in cases}
    rounds = []
    for ident in expected_ids:
        arm = "control" if ident.endswith(("a1", "a2")) else "candidate"
        receipt = read(root, "results/" + ident + ".receipt.json")
        require(receipt["state"] == "complete" and receipt["exit_code"] == 0 and
                receipt["artifact_sha256"] == receipt["artifact_sha256_after"], "round is incomplete or changed")
        document = read(root, "results/" + ident + ".stdout")
        require(INPUTS["results/" + ident + ".stdout"] == expected[ident], "raw source differs from frozen comparison")
        rows = document["geometry"]["verify_rows"]
        require(rows == protocol["arms"][arm]["rows"] and document["study_temperature"] == 0, "round policy differs")
        requests = document["measurements"]
        require(len(requests) == 31 and {row["scenario"] for row in requests} == set(cases), "case set differs")
        for row in requests:
            case = cases[row["scenario"]]
            require(row["sample"] == 0 and row["prompt_tokens"] == case["prompt_tokens"], "sample or prompt differs")
            by_case[row["scenario"]][arm].append(work.request_work(row, rows, case["output_cap"], ident))
        rounds.append({"job": ident, "arm": arm, "build_id": document["build_id"], "raw_sha256": expected[ident]})
    summaries = {}
    for name, arms in by_case.items():
        require(all(len(records) == 6 for records in arms.values()), "not six runs per arm")
        require(len({record["output_sha256"] for records in arms.values() for record in records}) == 1,
                "complete returned arrays differ")
        summary = {arm: summarize(work, records) for arm, records in arms.items()}
        changes = [next(row["throughput_change"] for row in group["cases"] if row["case"] == name)
                   for group in comparison["groups"]]
        summaries[name] = {
            "arms": summary, "candidate_divided_by_control": work.ratios(summary["control"], summary["candidate"]),
            "all_output_arrays_match": True, "registered_group_throughput_changes": changes,
            "cycle_trace_identical_within_arm": {arm: len({row["cycle_trace_sha256"] for row in records}) == 1
                                                 for arm, records in arms.items()},
        }
    faster = sorted(name for name, value in summaries.items() if min(value["registered_group_throughput_changes"]) > 0)
    slower = sorted(name for name, value in summaries.items() if max(value["registered_group_throughput_changes"]) < 0)
    require(len(faster) == 25 and set(slower) == SLOWER, "observed sign count differs from the completed report")
    largest = max(summaries, key=lambda name: statistics.median(summaries[name]["registered_group_throughput_changes"]))
    focus_names = sorted(SLOWER) + ["python_lru", largest, "count_secondary"]
    focus = {}
    for name in focus_names:
        case = summaries[name]
        ratio = case["candidate_divided_by_control"]
        focus[name] = {
            "registered_output_cap": cases[name]["output_cap"],
            "observed_returned_length": sorted({record["returned"] for records in by_case[name].values() for record in records}),
            "measured": f"Emitted tokens per recorded cycle change by {(ratio['emitted_tokens_per_cycle'] - 1) * 100:.2f}%; explicit decode GPU milliseconds per cycle change by {(ratio['reported_decode_gpu_ms_per_cycle'] - 1) * 100:.2f}%; pooled wall decode throughput changes by {(ratio['pooled_registered_wall_decode_tokens_per_second'] - 1) * 100:.2f}%.",
            "inference": ("Higher emitted work per cycle outweighs the higher bundled GPU cost, consistent with the observed gain."
                          if ratio["reported_gpu_emitted_tokens_per_second"] > 1 else
                          "Emitted work per cycle does not offset the higher bundled GPU cost, consistent with the observed regression."),
            "requests": by_case[name],
        }
    aggregate = {arm: summarize(work, [record for arms in by_case.values() for record in arms[arm]])
                 for arm in ["control", "candidate"]}
    raw(root, "scripts/study_driver_v2.mm")
    require(INPUTS["scripts/study_driver_v2.mm"] == "2b0b3898db9bee8941e169d8852c8f6659f256f7d00e9de9f01a2a054783879d",
            "study driver source differs")
    raw(root, "scripts/analyze_historical_mechanisms_v1.py")
    return {
        "scope": "Descriptive work accounting of all 12 completed historical31 rounds: six requests per case/arm, 186 requests per arm. All 31 registered cases are primary, including count_secondary despite its name. No agent12, partial historical, or mixed HTTP observations are read; the agent helper supplies only generic counter/profile functions.",
        "reference_method": work.REFERENCE, "rounds": rounds, "cases": summaries, "focus": focus,
        "observed_registered_effect_signs": {"positive_in_all_groups": faster, "negative_in_all_groups": slower,
                                             "scope": "Signs of already registered effects, not a new per-case qualification."},
        "largest_median_registered_gain_case": largest,
        "aggregate31": {"arms": aggregate, "candidate_divided_by_control": work.ratios(aggregate["control"], aggregate["candidate"])},
        "registered_verdict_unchanged": {"comparison_qualified": comparison["qualified"],
                                         "review_primary_reportable": review["primary_reportable"],
                                         "registered_criterion": review["criterion"],
                                         "group_throughput_changes": [group["primary_throughput_change"] for group in comparison["groups"]]},
        "censoring": "Use every case's registered cap and actual returned arrays. A below-cap completion is recorded as such; this driver omits explicit finish reasons, so EOS versus another early-finish reason is not asserted. All-cycle emission sums must equal each returned array. Eligible-prefix profiles exclude every final cycle conservatively and earlier partial-width cycles. Fixed-cap or fixed-length assumptions from agent12 are not applied. Eligible prefixes remain each drafter's encountered prefixes, not matched prefixes.",
        "timing_semantics": "GPU cost uses explicit decode_gpu_ms, separately from prefill_gpu_ms; it is not request GPU minus wall TTFT. Wall decode is wall_ms-ttft_ms with returned tokens minus one. Per-recorded-cycle averages are descriptive and the traces do not isolate draft, target, attention, projection or cache phase costs. No new p-values, qualification, or source tuning.",
        "inputs_sha256": dict(INPUTS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.evidence.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"focus": {name: row["measured"] for name, row in result["focus"].items()},
                      "registered_verdict_unchanged": result["registered_verdict_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
