"""Describe selector-prefix confidence against observed greedy prefix survival.

Full-width labels and truncated/censored labels stay separate. These descriptive
calibration statistics neither fit a policy nor prove sampling correctness.
Only complete B1 decode reports have an unambiguous request-to-workload mapping.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

from dev.benchmarks import trace_report


def observations(report, document, server):
    if not report.get("complete") or report["scenario"] != "decode":
        raise ValueError("calibration needs a complete decode report")
    if report["settings"]["batch"] != 1:
        raise ValueError("request mapping requires B1")
    trace_report.summarize(document)
    if report["residency"] == "sequential":
        warm = report["server_runs"][server]["warmup"]["rounds_per_arm"]
    else:
        warm = report["warmup"]["rounds_per_arm"]
    expected = [
        case
        for case in report["cases"]
        for row in case["rounds"]
        if (
            row["arm"] == server + 1
            if report["residency"] == "sequential"
            else row["server"] == server
        )
    ]
    active = sorted(
        (e for e in document["traceEvents"] if e.get("name") == "request.active"),
        key=lambda e: e["ts"],
    )[warm:]
    if len(active) != len(expected):
        raise ValueError("request count differs from the recorded workload sequence")
    mapping = {}
    for event, case in zip(active, expected, strict=True):
        args = event["args"]
        if args["rows"] != case["context"] or args["id"] in mapping:
            raise ValueError("request geometry or identity differs from report")
        mapping[args["id"]] = case
    records = []
    for event in document["traceEvents"]:
        if event.get("name") != "draft.confidence":
            continue
        args = event["args"]
        case = mapping.get(args["id"])
        if case is None:  # Startup and HTTP warmups are outside the dataset.
            continue
        confidence = args["value"]
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid selector confidence")
        if not 1 <= args["rows"] <= 7 or not 1 <= args["context"] <= 8:
            raise ValueError("invalid confidence geometry")
        records.append(
            {
                "context": case["context"],
                "workload": case["workload"],
                "position": args["rows"],
                "eligible_rows": args["context"],
                "observed": bool(args["flags"] & 1),
                "accepted_prefix": bool(args["flags"] & 2),
                "confidence": confidence,
            }
        )
    if not records:
        raise ValueError("trace has no workload confidence observations")
    return records


def calibrate(records):
    groups = collections.defaultdict(list)
    for row in records:
        groups[(row["context"], row["workload"], row["position"])].append(row)
    result = []
    for (context, workload, position), rows in sorted(groups.items()):
        bins = collections.defaultdict(list)
        for row in rows:
            if row["observed"]:
                bins[min(9, int(row["confidence"] * 10))].append(row)
        result.append(
            {
                "context": context,
                "workload": workload,
                "position": position,
                "total": len(rows),
                "censored": sum(not row["observed"] for row in rows),
                "bins": [
                    {
                        "lower_confidence": index / 10,
                        "upper_confidence": (index + 1) / 10,
                        "count": len(values),
                        "mean_confidence": statistics.mean(
                            r["confidence"] for r in values
                        ),
                        "prefix_survival": statistics.mean(
                            int(r["accepted_prefix"]) for r in values
                        ),
                        "brier_score": statistics.mean(
                            (r["confidence"] - int(r["accepted_prefix"])) ** 2
                            for r in values
                        ),
                    }
                    for index, values in sorted(bins.items())
                ],
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.directory / "report.json").read_text())
    arms = collections.defaultdict(list)
    inputs = []
    servers = len(report["server_runs"]) if report["residency"] == "sequential" else 2
    for server in range(servers):
        name = (
            f"round-{server + 1}-trace.json"
            if report["residency"] == "sequential"
            else f"server-{server}-trace.json"
        )
        data = (args.directory / name).read_bytes()
        rows = observations(report, json.loads(data), server)
        arm = (
            report["server_runs"][server]["server"] + 1
            if report["residency"] == "sequential"
            else server + 1
        )
        if arm == 1 and any(r["eligible_rows"] != 8 or not r["observed"] for r in rows):
            raise ValueError("reference arm is not an uncensored eight-row dataset")
        arms[arm].extend(rows)
        inputs.append(
            {
                "trace": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "arm": arm,
                "observations": len(rows),
            }
        )
    result = {
        "complete": True,
        "model": report["model"],
        "model_manifest_sha256": report["model_manifest_sha256"],
        "binaries": report["binaries"],
        "environments": report["environments"],
        "environment": report["environment"],
        "serving_tool_sha256": report["tool_sha256"],
        "analysis_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": report["settings"],
        "inputs": inputs,
        "arms": {str(arm): calibrate(rows) for arm, rows in arms.items()},
        "note": "Descriptive greedy-prefix statistics; code/prose and context remain separate. "
        "Censored suffixes are excluded, not labelled rejected. Repeated deterministic requests and "
        "positions within a prefix are correlated; counts are not independent trials. "
        "This selected-path cumulative feature differs from PR #487's max-candidate feature.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
