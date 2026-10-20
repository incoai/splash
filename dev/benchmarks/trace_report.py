"""Validate a serving trace and summarize phase times and GPU gaps.

Durations of nested host scopes overlap and must not be added. ANE intervals
are host event-notification/report times, not hardware execution timestamps.
GPU command spans may contain event waits; segment intervals retain the
actual command-buffer boundaries without claiming exact wait attribution.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {
        "count": len(values),
        "median_ms": statistics.median(values) / 1000,
        "p95_ms": percentile(values, 0.95) / 1000,
        "maximum_ms": max(values) / 1000,
    }


def summarize(document):
    if document.get("dropped_host_events") or document.get("dropped_commands"):
        raise ValueError("trace overflowed; shorten the traced workload")
    if document.get("exit_status", 0) != 0:
        raise ValueError("trace belongs to an unsuccessful serving process")
    if document.get("command_in_flight_at_stop"):
        raise ValueError("trace stopped before the workload completed")
    groups = collections.defaultdict(list)
    commands = []
    observations = []
    counters = {}
    prefill_lanes = collections.defaultdict(list)
    seeds = []
    confidence = []
    active = []
    for event in document["traceEvents"]:
        if event.get("ph") != "X":
            continue
        if (
            any(
                not isinstance(event.get(key), (int, float))
                or not math.isfinite(event[key])
                for key in ("ts", "dur")
            )
            or event["dur"] < 0
            or event["ts"] < 0
        ):
            raise ValueError("nonfinite or invalid trace interval")
        groups[event["name"]].append(event["dur"])
        if event["name"] == "request.active":
            active.append((event["ts"], event["ts"] + event["dur"]))
        if event["name"] == "metal.command_span":
            commands.append(event)
        if event["name"] == "prefill.lane":
            values = event["args"]
            prefill_lanes[values["id"]].append(
                {
                    "lane": values["flags"],
                    "rows": values["rows"],
                    "history": values["context"],
                }
            )
        if event["name"] == "prefill.seed":
            values = event["args"]
            seeds.append({"rows": values["rows"], "wall_ms": values["value"]})
        if event["name"] == "batch.counters":
            values = event["args"]
            counters[values["id"]] = {
                "output_tokens": values["rows"],
                "proposed_tokens": values["context"],
                "accepted_tokens": values["flags"],
            }
        if event["name"] == "draft.confidence":
            values = event["args"]
            confidence.append(
                {
                    "position": values["rows"],
                    "eligible_rows": values["context"],
                    "observed": bool(values["flags"] & 1),
                    "accepted_prefix": bool(values["flags"] & 2),
                    "confidence": values["value"],
                    "width": values["width"],
                }
            )
        if event["name"] == "batch.observation":
            values = event["args"]
            if values["flags"] & 1:
                observations.append(
                    {
                        "batch_id": values["id"],
                        "rows": values["rows"],
                        "width": values["width"],
                        "maximum_history": values["context"],
                        "representative": bool(values["flags"] & 2),
                        "wall_ms": values["value"],
                        "timestamp_us": event["ts"],
                    }
                )
    commands.sort(key=lambda event: event["ts"])
    merged = []
    for begin, end in sorted(active):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    active = merged
    gaps = []
    busy_gaps = []
    for previous, following in zip(commands, commands[1:]):
        end = previous["ts"] + previous["dur"]
        gap = following["ts"] - end
        if gap < -1:
            raise ValueError("GPU command spans overlap on the serial workload")
        gaps.append(max(0, gap))
        if any(begin <= end and following["ts"] <= stop for begin, stop in active):
            busy_gaps.append(max(0, gap))
    return {
        "phase_times": {key: distribution(values) for key, values in groups.items()},
        "gpu_inter_command_gaps": distribution(gaps) if gaps else None,
        "gpu_gaps_with_a_request_active": distribution(busy_gaps)
        if busy_gaps
        else None,
        "prefill_seeds": seeds,
        "prefill_observations": [
            {
                **row,
                **counters.get(row["batch_id"], {}),
                "lanes": sorted(
                    prefill_lanes.get(row["batch_id"], []),
                    key=lambda lane: lane["lane"],
                ),
            }
            for row in observations
        ],
        "confidence_observations": confidence,
        "notes": [
            "Nested scope durations overlap; do not sum the phase table.",
            "GPU gaps include real request idleness and constrained-stage waits.",
            "ANE ready/report intervals include callback latency and may overlap GPU work.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(json.loads(args.trace.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, values in report["phase_times"].items():
        print(f"{name}: {values['count']} samples, median {values['median_ms']:.4f} ms")


if __name__ == "__main__":
    main()
