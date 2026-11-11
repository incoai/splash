#!/usr/bin/env python3
"""Describe work in all completed agent12 rounds; do not create a new verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "agent12-dflash8-vs-review16-v1"
REFERENCE = {
    "repository_path": "dev/benchmarks/draft_acceptance.py",
    "sha256": "be1c8e1957b1299a92839d6f6ce9ad5c77691931ec301816115ad586c38844dc",
    "rule": "Each request excludes its final cycle conservatively, then excludes earlier cycles with fewer than verify_rows-1 proposals. Profiles describe each drafter's encountered prefixes, not matched prefixes.",
}
INPUTS = {}


def read(root, relative):
    data = (root / relative).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if relative in INPUTS and INPUTS[relative] != digest:
        raise ValueError("input changed: " + relative)
    INPUTS[relative] = digest
    return json.loads(data)


def require(condition, message):
    if not condition:
        raise ValueError(message)


# Unchanged prefix_profile algorithm from the reference file identified above.
def prefix_profile(cycles: list[dict], positions: int) -> dict:
    if positions < 1:
        raise ValueError("a draft must propose at least one token")
    for cycle in cycles:
        drafted, accepted = cycle["drafted_tokens"], cycle["accepted_draft_tokens"]
        if not 0 <= accepted <= drafted <= positions:
            raise ValueError("invalid accepted prefix or draft width")
    eligible = [c for c in cycles[:-1] if c["drafted_tokens"] == positions]
    histogram = [0] * (positions + 1)
    for cycle in eligible:
        histogram[cycle["accepted_draft_tokens"]] += 1
    count = len(eligible)
    rows = []
    for position in range(1, positions + 1):
        reached = sum(histogram[position - 1 :])
        accepted = sum(histogram[position:])
        rows.append(
            {
                "position": position,
                "reached": reached,
                "accepted": accepted,
                "conditional_acceptance": accepted / reached if reached else None,
                "survival": accepted / count if count else None,
            }
        )
    return {
        "cycles": len(cycles),
        "eligible_cycles": count,
        "excluded_final_cycles": int(bool(cycles)),
        "excluded_partial_cycles": max(0, len(cycles) - 1 - count),
        "accepted_prefix_histogram": histogram,
        "mean_accepted_prefix": sum(i * n for i, n in enumerate(histogram)) / count
        if count
        else None,
        "first_proposal_rejection_fraction": histogram[0] / count if count else None,
        "full_acceptance_fraction": histogram[-1] / count if count else None,
        "positions": rows,
    }


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def request_work(measurement, rows, cap, job):
    tokens, cycles = measurement["output_tokens"], measurement["draft_cycles"]
    require(tokens and cycles and len(tokens) <= cap, "empty request or output beyond cap")
    require(all(type(token) is int and token >= 0 for token in tokens), "invalid output token")
    profile = prefix_profile(cycles, rows - 1)
    for index, cycle in enumerate(cycles):
        require(all(type(cycle[key]) is int and cycle[key] >= 0 for key in
                    ["drafted_tokens", "accepted_draft_tokens", "output_tokens"]), "invalid cycle counter")
        if index < len(cycles) - 1:
            require(cycle["output_tokens"] == cycle["accepted_draft_tokens"] + 1,
                    "nonterminal emission differs from accepted prefix plus one")
    emitted = sum(cycle["output_tokens"] for cycle in cycles)
    require(emitted == len(tokens), "cycle emissions differ from returned token array")
    timing = {key: measurement[key] for key in ["wall_ms", "ttft_ms", "prefill_gpu_ms", "decode_gpu_ms"]}
    require(all(math.isfinite(value) and value >= 0 for value in timing.values()), "invalid timing")
    require(timing["wall_ms"] > timing["ttft_ms"] and timing["decode_gpu_ms"] > 0, "nonpositive decode time")
    eligible = [cycle for cycle in cycles[:-1] if cycle["drafted_tokens"] == rows - 1]
    require(sum(cycle["output_tokens"] for cycle in eligible) ==
            len(eligible) + sum(cycle["accepted_draft_tokens"] for cycle in eligible),
            "eligible emission accounting differs")
    return {
        "job": job, "scenario": measurement["scenario"], "sample": measurement["sample"], "verify_rows": rows,
        "prompt_tokens": measurement["prompt_tokens"], "output_cap": cap,
        "returned": len(tokens), "registered_decode_tokens": len(tokens) - 1,
        "output_sha256": token_hash(tokens), "cycle_trace_sha256": token_hash(cycles),
        "cycles": len(cycles), "emitted": emitted,
        "drafted": sum(cycle["drafted_tokens"] for cycle in cycles),
        "accepted": sum(cycle["accepted_draft_tokens"] for cycle in cycles),
        **timing, "decode_wall_ms": timing["wall_ms"] - timing["ttft_ms"],
        "at_output_cap": len(tokens) == cap,
        "final_cycle": cycles[-1],
        "final_emission_minus_accepted_minus_one": cycles[-1]["output_tokens"] - cycles[-1]["accepted_draft_tokens"] - 1,
        "profile": {key: value for key, value in profile.items() if key != "positions"},
    }


def summarize(records):
    require(records and len({record["verify_rows"] for record in records}) == 1, "mixed verification widths")
    rows = records[0]["verify_rows"]
    fields = ["returned", "registered_decode_tokens", "cycles", "emitted", "drafted", "accepted",
              "wall_ms", "ttft_ms", "prefill_gpu_ms", "decode_gpu_ms", "decode_wall_ms"]
    totals = {field: sum(record[field] for record in records) for field in fields}
    histogram = [sum(record["profile"]["accepted_prefix_histogram"][i] for record in records) for i in range(rows)]
    eligible = sum(histogram)
    accepted = sum(index * count for index, count in enumerate(histogram))
    positions = []
    for position in range(1, rows):
        reached, survived = sum(histogram[position - 1:]), sum(histogram[position:])
        positions.append({"position": position, "reached": reached, "accepted": survived,
                          "conditional_acceptance": survived / reached if reached else None,
                          "survival": survived / eligible if eligible else None})
    require(eligible == sum(record["profile"]["eligible_cycles"] for record in records), "profile histogram count differs")
    return {
        "requests": len(records), "verify_rows": rows, "totals": totals,
        "all_cycles": {
            "emitted_tokens_per_cycle": totals["emitted"] / totals["cycles"],
            "accepted_fraction_of_drafted": totals["accepted"] / totals["drafted"],
            "reported_decode_gpu_ms_per_cycle": totals["decode_gpu_ms"] / totals["cycles"],
            "amortized_decode_wall_ms_per_recorded_cycle": totals["decode_wall_ms"] / totals["cycles"],
            "reported_gpu_emitted_tokens_per_second": 1000 * totals["emitted"] / totals["decode_gpu_ms"],
            "pooled_registered_wall_decode_tokens_per_second": 1000 * totals["registered_decode_tokens"] / totals["decode_wall_ms"],
        },
        "full_width_nonfinal": {
            "eligible_cycles": eligible, "excluded_final_cycles": len(records),
            "excluded_earlier_partial_cycles": sum(record["profile"]["excluded_partial_cycles"] for record in records),
            "accepted_prefix_histogram": histogram, "accepted_tokens": accepted,
            "emitted_tokens": eligible + accepted,
            "mean_accepted_prefix": accepted / eligible if eligible else None,
            "emitted_tokens_per_eligible_cycle": 1 + accepted / eligible if eligible else None,
            "first_proposal_rejection_fraction": histogram[0] / eligible if eligible else None,
            "accepted_tokens_beyond_position7": sum(max(0, index - 7) * count for index, count in enumerate(histogram)),
            "cycles_accepting_more_than7": sum(histogram[8:]), "positions": positions,
        },
        "censoring": {
            "requests_at_output_cap": sum(record["at_output_cap"] for record in records),
            "final_partial_width_cycles": sum(record["final_cycle"]["drafted_tokens"] < rows - 1 for record in records),
            "final_emission_minus_accepted_minus_one_total": sum(record["final_emission_minus_accepted_minus_one"] for record in records),
            "terminal_cycles": [{"job": record["job"], "scenario": record["scenario"], "cycle": record["final_cycle"],
                                 "emission_minus_accepted_minus_one": record["final_emission_minus_accepted_minus_one"]} for record in records],
        },
    }


def ratios(control, candidate):
    result = {field: candidate["all_cycles"][field] / control["all_cycles"][field] for field in control["all_cycles"]}
    result["cycle_count"] = candidate["totals"]["cycles"] / control["totals"]["cycles"]
    # For fixed returned work this is the exact work/cost decomposition of GPU throughput.
    require(math.isclose(result["reported_gpu_emitted_tokens_per_second"],
                         result["emitted_tokens_per_cycle"] / result["reported_decode_gpu_ms_per_cycle"]),
            "GPU work/cost ratio identity failed")
    return result


def evaluate(root):
    INPUTS.clear()
    protocol = read(root, "notes/" + PREFIX + ".protocol.json")
    comparison = read(root, "results/" + PREFIX + ".comparison.json")
    review = read(root, "results/" + PREFIX + ".review.json")
    driver = (root / "scripts/study_driver_v2.mm").read_bytes()
    driver_sha = hashlib.sha256(driver).hexdigest()
    require(driver_sha == "2b0b3898db9bee8941e169d8852c8f6659f256f7d00e9de9f01a2a054783879d", "study driver source differs")
    INPUTS["scripts/study_driver_v2.mm"] = driver_sha
    INPUTS["scripts/analyze_agent_mechanisms_v1.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    require(comparison["protocol"] == protocol and protocol["groups"] == 3 and
            protocol["samples_per_round"] == 1 and protocol["clone_width"] == 1, "unexpected completed campaign")
    require(review["status"] == "complete" and review["rounds_validated"] == 12 and
            review["protocol_sha256"] == INPUTS["notes/" + PREFIX + ".protocol.json"], "review provenance differs")
    cases = {case["name"]: case for case in protocol["registered_cases"]}
    require(len(cases) == 12 and not protocol.get("exclude_primary_cases"), "unexpected agent cohort")
    expected = {source["job"]: source["sha256"] for group in comparison["groups"] for source in group["sources"]}
    expected_ids = [f"{PREFIX}-g{group}-{suffix}" for group in range(1, 4) for suffix in ["a1", "b1", "b2", "a2"]]
    require(set(expected) == set(expected_ids), "completed source set differs")
    by_case = {case: {"control": [], "candidate": []} for case in cases}
    rounds = []
    for ident in expected_ids:
        arm = "control" if ident.endswith(("a1", "a2")) else "candidate"
        receipt = read(root, "results/" + ident + ".receipt.json")
        require(receipt["state"] == "complete" and receipt["exit_code"] == 0 and
                receipt["artifact_sha256"] == receipt["artifact_sha256_after"], "round is not complete and stable")
        document = read(root, "results/" + ident + ".stdout")
        require(INPUTS["results/" + ident + ".stdout"] == expected[ident], "raw round differs from frozen comparison")
        require(document["geometry"]["verify_rows"] == protocol["arms"][arm]["rows"] and
                document["study_temperature"] == 0, "round policy differs")
        requests = document["measurements"]
        require(len(requests) == 12 and {row["scenario"] for row in requests} == set(cases), "case set differs")
        for row in requests:
            require(row["sample"] == 0 and row["prompt_tokens"] == cases[row["scenario"]]["prompt_tokens"], "sample or prompt differs")
            by_case[row["scenario"]][arm].append(request_work(row, document["geometry"]["verify_rows"],
                                                              cases[row["scenario"]]["output_cap"], ident))
        rounds.append({"job": ident, "arm": arm, "build_id": document["build_id"], "raw_sha256": expected[ident]})
    result_cases = {}
    for name, arms in by_case.items():
        require(all(len(records) == 6 for records in arms.values()), "not six runs per arm")
        require(len({record["output_sha256"] for records in arms.values() for record in records}) == 1,
                "returned arrays differ")
        summary = {arm: summarize(records) for arm, records in arms.items()}
        result_cases[name] = {
            "arms": summary, "candidate_divided_by_control": ratios(summary["control"], summary["candidate"]),
            "all_output_arrays_match": True,
            "cycle_trace_identical_within_arm": {arm: len({record["cycle_trace_sha256"] for record in records}) == 1
                                                 for arm, records in arms.items()},
            "registered_group_throughput_changes": [next(row["throughput_change"] for row in group["cases"]
                                                         if row["case"] == name) for group in comparison["groups"]],
            "requests": arms,
        }
    aggregate = {arm: summarize([record for arms in by_case.values() for record in arms[arm]])
                 for arm in ["control", "candidate"]}
    focus = {}
    for name in ["medium_cancel", "medium_router", "short_rust", "short_sql"]:
        case = result_cases[name]
        ratio = case["candidate_divided_by_control"]
        focus[name] = {
            "measured": f"Across six runs per arm, emitted tokens per recorded cycle change by {(ratio['emitted_tokens_per_cycle'] - 1) * 100:.2f}%; explicit decode GPU milliseconds per cycle change by {(ratio['reported_decode_gpu_ms_per_cycle'] - 1) * 100:.2f}%. Pooled wall decode throughput changes by {(ratio['pooled_registered_wall_decode_tokens_per_second'] - 1) * 100:.2f}%.",
            "inference": ("The increase in emitted work per cycle exceeds the higher GPU cost per cycle, consistent with the observed gain."
                          if ratio["reported_gpu_emitted_tokens_per_second"] > 1 else
                          "Emitted work per cycle does not offset the higher GPU cost per cycle, consistent with the observed regression."),
        }
    return {
        "scope": "Descriptive accounting of every completed agent12 round: six native requests per case and arm, 72 per arm. No partial historical study is read; no new qualification, hypothesis test or source choice is made.",
        "reference_method": REFERENCE, "rounds": rounds, "cases": result_cases,
        "aggregate12": {"arms": aggregate, "candidate_divided_by_control": ratios(aggregate["control"], aggregate["candidate"])},
        "focus_rationale": focus,
        "registered_verdict_unchanged": {"comparison_qualified": comparison["qualified"],
                                         "review_primary_reportable": review["primary_reportable"],
                                         "registered_criterion": review["criterion"],
                                         "review_cross_group_output_gate": review["cross_group_output_gate"],
                                         "group_throughput_changes": [group["primary_throughput_change"] for group in comparison["groups"]]},
        "timing_semantics": {
            "gpu": "Use the explicit decode_gpu_ms field only: study_driver_v2.mm records the delta of totalDecodeGpuSeconds, separately from prefill_gpu_ms. No wall-clock TTFT is subtracted from GPU time.",
            "wall": "Registered decode wall is wall_ms minus ttft_ms and counts returned tokens minus one. Dividing that interval by all recorded cycles is only an amortized descriptor: TTFT already excludes an initial interval, so this is not a measured single-cycle latency.",
            "attribution": "Reported decode GPU time bundles pipeline work; these traces do not separate draft generation, target verification, attention, projections or cache work. Wider verification and higher measured cycle cost are consistent, but their causal phase shares cannot be inferred.",
        },
        "censoring": "All-cycle sums retain terminal and partial cycles. Eligible profiles exclude each request's final cycle even when full-width, plus earlier partial-width cycles. Output-cap requests and terminal extra emissions are recorded separately. A fraction accepted out of 15 proposals is not directly the same quality measure as a fraction out of seven; eligible prefixes are not matched between drafters.",
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
    print(json.dumps(result["focus_rationale"], indent=2))


if __name__ == "__main__":
    main()
