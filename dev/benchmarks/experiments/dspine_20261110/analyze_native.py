#!/usr/bin/env python3
"""Recompute the registered native ABBA gates from extracted evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def compare(rounds):
    if len(rounds) != 4 or not all(math.isfinite(v) and v > 0 for v in rounds):
        raise ValueError("Four positive finite ABBA observations are required")
    a1, b1, b2, a2 = rounds
    baseline, candidate = (a1 + a2) / 2, (b1 + b2) / 2
    noise = max(abs(a1 - a2) / baseline, abs(b1 - b2) / candidate)
    regression = candidate / baseline - 1
    bound = max(0.02, 2 * noise)
    verdict = (
        "inconclusive" if noise > 0.05 else "pass" if regression <= bound else "fail"
    )
    return {
        "rounds": rounds,
        "baseline": baseline,
        "candidate": candidate,
        "noise": noise,
        "regression": regression,
        "bound": bound,
        "verdict": verdict,
        "pass": verdict == "pass",
    }


def metrics(document):
    values = {}
    for row in document["measurements"]:
        key = f"{row['scenario']}:{row['sample']}"
        if key in values:
            raise ValueError("Duplicate case/sample: " + key)
        tokens = row["output_tokens"]
        if len(tokens) < 2:
            raise ValueError("Decode timing requires at least two emitted tokens")
        values[key] = {
            "scenario": row["scenario"],
            "token_sha256": hashlib.sha256(
                json.dumps(tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "decode_tokens": len(tokens) - 1,
            "decode_wall_ms": row["wall_ms"] - row["ttft_ms"],
            "cycles": len(row.get("draft_cycles", [])),
        }
    return values


def evaluate(root, prefix):
    groups = []
    for group in range(1, 4):
        ids = [f"{prefix}-g{group}-{suffix}" for suffix in ("a1", "b1", "b2", "a2")]
        rows, problems = [], []
        for ident in ids:
            receipt = root / "results" / f"{ident}.receipt.json"
            if not receipt.exists():
                problems.append(ident + ": pending")
                continue
            record = json.loads(receipt.read_text())
            if record["exit_code"] != 0:
                problems.append(f"{ident}: exit {record['exit_code']}")
                continue
            operating = json.loads(
                (root / "results" / f"{ident}.operating.json").read_text()
            )
            for gate in (
                "operating_gate_passed",
                "no_serious_or_critical_state",
                "no_swap",
            ):
                if not operating.get(gate):
                    problems.append(ident + ": " + gate + " failed")
            rows.append(
                metrics(json.loads((root / "results" / f"{ident}.stdout").read_text()))
            )
        result = {
            "group": group,
            "rounds": ids,
            "problems": problems,
            "qualified": False,
        }
        if len(rows) == 4:
            if any(row.keys() != rows[0].keys() for row in rows):
                raise ValueError("Round case sets differ")
            cases = []
            for key in rows[0]:
                observations = [row[key] for row in rows]
                cases.append(
                    {
                        "key": key,
                        "output_arrays_match": len(
                            {r["token_sha256"] for r in observations}
                        )
                        == 1,
                        "cycles": [r["cycles"] for r in observations],
                        "abba_milliseconds_per_token": compare(
                            [
                                r["decode_wall_ms"] / r["decode_tokens"]
                                for r in observations
                            ]
                        ),
                    }
                )
            totals = []
            for row in rows:
                primary = [
                    v for v in row.values() if v["scenario"] != "count_secondary"
                ]
                totals.append(
                    sum(v["decode_wall_ms"] for v in primary)
                    / sum(v["decode_tokens"] for v in primary)
                )
            overall = compare(totals)
            same = all(c["output_arrays_match"] for c in cases)
            quiet = all(
                c["abba_milliseconds_per_token"]["noise"] <= 0.05 for c in cases
            )
            improved = (
                overall["regression"] < -overall["bound"] and overall["noise"] <= 0.05
            )
            result.update(
                primary_abba_milliseconds_per_token=overall,
                primary_throughput_change=overall["baseline"] / overall["candidate"]
                - 1,
                all_output_arrays_match=same,
                all_case_control_spreads_at_most_five_percent=quiet,
                primary_improvement_exceeds_noise_allowance=improved,
                cases=cases,
                qualified=not problems and same and quiet and improved,
            )
        groups.append(result)
    return {
        "prefix": prefix,
        "groups_required": 3,
        "qualified": all(g["qualified"] for g in groups),
        "groups": groups,
        "scope": "Three fixed groups; all retained. Serial native requests, the selected frozen cohort; count_secondary excluded from primary totals when present. This is a positive improvement gate, stricter than a non-regression check.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="Extracted evidence root containing results/"
    )
    parser.add_argument("prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.root, args.prefix)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "qualified": result["qualified"],
                "groups": [
                    {k: v for k, v in g.items() if k != "cases"}
                    for g in result["groups"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
