"""Measure experimental live verification with ordinary native execution.

Run as ``python -m dev.benchmarks.verification_sweep --help``. Each threshold
gets an independent off/on/on/off comparison. Native cycles can advance by
different token counts, so use matched HTTP workloads before promoting a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from . import abba


def counts(value):
    values = [int(part) for part in value.split(",")]
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("counts must be positive")
    return values


def thresholds(value):
    values = [float(part) for part in value.split(",")]
    if not values or any(not math.isfinite(v) or not 0 < v <= 1 for v in values):
        raise argparse.ArgumentTypeError("thresholds must be in (0, 1]")
    return values


def environment():
    return {
        name: subprocess.check_output(command, text=True).strip()
        for name, command in (
            ("power", ["pmset", "-g", "batt"]),
            ("thermal", ["pmset", "-g", "therm"]),
        )
    }


def fingerprint(path):
    with path.open("rb") as source:
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.file_digest(source, "sha256").hexdigest(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--metallib", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=counts, default=[2048])
    parser.add_argument("--thresholds", type=thresholds, default=[0.01, 0.05, 0.1, 0.2])
    parser.add_argument("--cycles", type=int, default=16)
    parser.add_argument("--kv-format", choices=["int8", "bf16"], default="int8")
    parser.add_argument("--state-format", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("cycles must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "model_root": str(args.model_root.resolve()),
        "profile": fingerprint(args.profile),
        "metallib": fingerprint(args.metallib),
        "model_manifest": fingerprint(args.model_root / "model.json"),
        "comparisons": [],
    }
    for context in args.contexts:
        for threshold in args.thresholds:
            rounds = []
            for index, enabled in enumerate([False, True, True, False], 1):
                label = f"c{context}-q{threshold:g}-r{index}"
                samples_path = args.output / f"{label}.json"
                command = [
                    str(args.profile.resolve()),
                    str(args.metallib.resolve()),
                    str(args.model_root.resolve()),
                    "--prompt-tokens",
                    str(context),
                    "--cycles",
                    str(args.cycles),
                    "--live-verify-threshold",
                    str(threshold if enabled else 0),
                    "--attribution",
                    "off",
                    "--kv-format",
                    args.kv_format,
                    "--state-format",
                    args.state_format,
                    "--json-output",
                    str(samples_path),
                ]
                command += ["--trace-output", str(args.output / f"{label}-trace.json")]
                print(f"{label}: {'on' if enabled else 'off'}", flush=True)
                before = environment()
                with (args.output / f"{label}.log").open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=1800,
                    )
                samples = json.loads(samples_path.read_text())
                if len(samples) != args.cycles * 4:
                    raise RuntimeError(f"{label}: missing native samples")
                rounds.append(
                    {"enabled": enabled, "environment": before, "samples": samples}
                )
            comparisons = []
            for width in range(1, 5):
                times, eligible, tokens = [], [], []
                for record in rounds:
                    rows = [row for row in record["samples"] if row["width"] == width]
                    if any(
                        row["committed_tokens"] < 1 or row["wall_seconds"] <= 0
                        for row in rows
                    ):
                        raise RuntimeError(
                            "invalid native timing or committed token count"
                        )
                    produced = sum(row["committed_tokens"] for row in rows)
                    times.append(sum(row["wall_seconds"] for row in rows) / produced)
                    eligible.append(
                        sum(row["eligible_rows"] for row in rows) / len(rows)
                    )
                    tokens.append(produced / len(rows))
                result = abba.compare(*times)
                result.update(
                    width=width,
                    tokens_per_cycle=tokens,
                    eligible_rows_per_cycle=eligible,
                )
                comparisons.append(result)
                print(f"  B{width}: {abba.describe(result)}", flush=True)
            report["comparisons"].append(
                {
                    "context": context,
                    "threshold": threshold,
                    "rounds": rounds,
                    "widths": comparisons,
                }
            )
            (args.output / "summary.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
