"""Compare prefill timing predictions using only earlier trace observations.

This is an offline estimator experiment, not a latency or scheduling result.
Ragged batches require exact per-lane geometry and use row-weighted history.
Execution paths and batch widths have separate bounded sample histories.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from dev.benchmarks import trace_report


def bucket(rows, history):
    row_bucket = sum(rows > boundary for boundary in (128, 512))
    history_bucket = sum(history > boundary for boundary in (1023, 4095, 16383))
    return row_bucket, history_bucket


def evaluate(observations, minimum_samples=3, seed=None):
    global_cost = None
    if seed is not None:
        if (
            seed["rows"] < 64
            or not math.isfinite(seed["wall_ms"])
            or seed["wall_ms"] <= 0
        ):
            raise ValueError("invalid startup prefill seed")
        global_cost = seed["wall_ms"] / seed["rows"]
    buckets = {}
    predictions = []
    excluded = {"nonrepresentative": 0, "small": 0, "ragged": 0}
    for row in observations:
        if not row["representative"]:
            excluded["nonrepresentative"] += 1
            continue
        if row["rows"] < 64:
            excluded["small"] += 1
            continue
        rows, history, actual = row["rows"], row["maximum_history"], row["wall_ms"]
        width = row["width"]
        if not 1 <= width <= 4:
            raise ValueError("invalid prefill width")
        lanes = row.get("lanes", [])
        if lanes:
            if (
                len(lanes) != width
                or {lane["lane"] for lane in lanes} != set(range(width))
                or sum(lane["rows"] for lane in lanes) != rows
                or any(lane["rows"] <= 0 or lane["history"] < 0 for lane in lanes)
            ):
                raise ValueError("invalid ragged prefill geometry")
            history = sum(lane["rows"] * lane["history"] for lane in lanes) / rows
        elif width != 1:
            excluded["ragged"] += 1
            continue
        if (
            not math.isfinite(history)
            or history < 0
            or rows > 2048
            or not math.isfinite(actual)
            or actual <= 0
        ):
            raise ValueError("invalid prefill observation")
        path = row.get("predicted_ane")
        key = (*bucket(rows, history), width, path)
        count, cost = buckets.get(key, (0, 0.0))
        if global_cost is not None:
            local = cost if count >= minimum_samples else global_cost
            predictions.append(
                {
                    "rows": rows,
                    "history": history,
                    "width": width,
                    "predicted_ane": path,
                    "actual_ms": actual,
                    "ema_ms": rows * global_cost,
                    "bucketed_ms": rows * local,
                    "uses_bucket": count >= minimum_samples,
                    "recorded_prediction_ms": row.get("predicted_wall_ms"),
                }
            )
        observed = actual / rows
        global_cost = (
            observed if global_cost is None else 0.75 * global_cost + 0.25 * observed
        )
        buckets[key] = (
            count + 1,
            observed if count == 0 else 0.75 * cost + 0.25 * observed,
        )
    errors = {}
    for policy in ("ema", "bucketed"):
        values = [
            abs(row[f"{policy}_ms"] / row["actual_ms"] - 1) for row in predictions
        ]
        errors[policy] = {
            "mean_absolute_relative_error": statistics.mean(values) if values else None,
            "p95_absolute_relative_error": trace_report.percentile(values, 0.95)
            if values
            else None,
        }
    return {
        "predictions": predictions,
        "errors": errors,
        "excluded": excluded,
        "minimum_prior_bucket_samples": minimum_samples,
        "note": "A recorded startup seed is used when present; otherwise the first observation seeds both estimators. "
        "Ragged history is row-weighted across every lane; width and GPU/ANE path have separate cells. "
        "Missing execution-path metadata is kept separate from known GPU/ANE paths. "
        "Prediction error is not measured latency or throughput benefit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {"complete": False, "runs": []}
    for path in args.traces:
        summary = trace_report.summarize(json.loads(path.read_text()))
        seeds = summary.get("prefill_seeds", [])
        if len(seeds) > 1:
            raise ValueError("multiple startup seeds require a timestamped replay")
        result["runs"].append(
            {
                "trace": str(path),
                **evaluate(
                    summary["prefill_observations"], seed=seeds[0] if seeds else None
                ),
            }
        )
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
