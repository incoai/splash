"""Summarize per-position acceptance from backend-benchmark --scenario acceptance.

Run each installed drafter with --ane-ffn-share 0 --max-context 32768 and
save its JSON. Then run this module with --baseline FILE --candidate FILE
--output FILE. It compares the same greedy coding prompts as
draft_comparison.py. The last cycle is excluded conservatively because EOS
or the output cap may censor it. Earlier partial verification cycles are
excluded too. Conditional acceptance counts only cycles that reached a
position; survival counts all eligible cycles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def summarize(document: dict) -> dict:
    positions = document["geometry"]["verify_rows"] - 1
    rows = []
    for request in document["measurements"]:
        if request["scenario"].startswith("acceptance_"):
            rows.append(
                {
                    "scenario": request["scenario"],
                    "sample": request["sample"],
                    "output_tokens": request["output_tokens"],
                    "profile": prefix_profile(request["draft_cycles"], positions),
                }
            )
    if not rows:
        raise ValueError("no acceptance scenarios found")
    return {
        "build_id": document["build_id"],
        "proposal_positions": positions,
        "requests": rows,
    }


def compare(baseline: dict, candidate: dict) -> dict:
    baseline, candidate = summarize(baseline), summarize(candidate)
    a = {(r["scenario"], r["sample"]): r["output_tokens"] for r in baseline["requests"]}
    b = {
        (r["scenario"], r["sample"]): r["output_tokens"] for r in candidate["requests"]
    }
    return {
        "schema_version": 1,
        "method": "Full-width cycles only; final cycle excluded conservatively for EOS/output-cap censoring. This is an acceptance study, not a speed comparison.",
        "output_tokens_match": a == b,
        "baseline": baseline,
        "candidate": candidate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
