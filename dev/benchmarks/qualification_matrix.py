"""Prepare or execute a pinned Qwen qualification matrix on one Mac.

The default writes a reviewable run manifest. --execute runs it sequentially;
an unexecuted job is never reported as passing. Install models and freeze both
build directories first. Hardware results belong to the machine that ran them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def fingerprint(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def jobs(args, models):
    result = []
    candidate = args.candidate.resolve()
    baseline = args.baseline.resolve()
    for model in models:
        name = model["label"]
        root = str(Path(model["root"]).expanduser().resolve())
        if args.suite in ("all", "correctness"):
            for kv in ("int8", "bf16"):
                for state in ("fp32", "bf16"):
                    command = [
                        str(candidate / "engine-tests/model-runtime-oracle"),
                        str(candidate / "splash.metallib"),
                        root,
                        "--kv-format",
                        kv,
                        "--state-format",
                        state,
                    ]
                    result.append(
                        {
                            "id": f"{name}-oracle-{kv}-{state}",
                            "command": command,
                            "environment": {"MTL_SHADER_VALIDATION": "1"},
                        }
                    )
        if args.suite in ("all", "serving"):
            for width in range(1, 5):
                contexts = "2048,8192,31744" if width == 1 else "2048"
                for scenario in ("decode", "mixed") if width == 1 else ("decode",):
                    identity = f"{name}-{scenario}-b{width}"
                    command = [
                        sys.executable,
                        "-m",
                        "dev.benchmarks.serving_regression",
                        "--model",
                        model["model"],
                        "--model-root",
                        root,
                        "--baseline-binary",
                        str(baseline / "splash"),
                        "--binary",
                        str(candidate / "splash"),
                        "--baseline-verify-rows",
                        args.baseline_verify_rows,
                        "--candidate-verify-rows",
                        args.candidate_verify_rows,
                        "--contexts",
                        contexts,
                        "--batch",
                        str(width),
                        "--workloads",
                        "code,math,prose",
                        "--scenario",
                        scenario,
                        "--residency",
                        args.residency,
                        "--warmup-context",
                        "2048",
                        "--warmup-tokens",
                        "512",
                        "--warmup-seconds",
                        "0",
                        "--warmup-rounds",
                        "4",
                        "--tokens",
                        str(args.tokens),
                        "--samples",
                        str(args.samples),
                        "--max-memory",
                        model.get("max_memory", args.max_memory),
                        "--probes",
                        "--output",
                        str(args.output.resolve() / identity),
                    ]
                    for value in args.baseline_env:
                        command += ["--baseline-env", value]
                    for value in args.candidate_env:
                        command += ["--candidate-env", value]
                    result.append(
                        {"id": identity, "command": command, "environment": {}}
                    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument(
        "--baseline", required=True, type=Path, help="frozen build directory"
    )
    parser.add_argument(
        "--candidate", required=True, type=Path, help="frozen build directory"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--suite", choices=("all", "correctness", "serving"), default="all"
    )
    parser.add_argument(
        "--baseline-verify-rows", choices=("all", "auto"), default="all"
    )
    parser.add_argument(
        "--candidate-verify-rows", choices=("all", "auto"), default="auto"
    )
    parser.add_argument("--baseline-env", action="append", default=[])
    parser.add_argument("--candidate-env", action="append", default=[])
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-memory", default="20G")
    parser.add_argument(
        "--residency", choices=("sequential", "paired"), default="sequential"
    )
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.tokens < 128:
        parser.error("need positive samples and at least 128 tokens")
    models = json.loads(args.models.read_text())
    labels = [model["label"] for model in models]
    if (
        not models
        or len(labels) != len(set(labels))
        or any(
            not label
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label)
            for label in labels
        )
    ):
        parser.error("model labels must be unique lowercase identifiers")
    args.output.mkdir(parents=True, exist_ok=True)
    work = jobs(args, models)
    report = {
        "status": "prepared",
        "machine": platform.platform(),
        "models": models,
        "jobs": [{**job, "status": "unexecuted"} for job in work],
        "artifacts": {},
        "complete": False,
        "notes": [
            "Preparation and absent hardware are never passing test results.",
            "Serving verdicts and noise bounds remain in each child report.",
            "B2-B4 long-context expansion needs additional memory qualification.",
            "These runs do not change a shared default.",
        ],
    }
    path = args.output / "qualification.json"

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(f"Prepared {len(work)} sequential jobs in {path}", flush=True)
    if not args.execute:
        return
    artifacts = {Path(job["command"][0]) for job in work}
    for build in (args.baseline, args.candidate):
        artifacts.update(
            (build.resolve() / "splash", build.resolve() / "splash.metallib")
        )
    artifacts.update(
        Path(model["root"]).expanduser().resolve() / "model.json" for model in models
    )
    for artifact in sorted(artifacts):
        report["artifacts"][str(artifact)] = fingerprint(artifact)
    report["status"] = "running"
    save()
    try:
        for record in report["jobs"]:
            print(record["id"], flush=True)
            record.update(status="running", started_unix=time.time())
            save()
            environment = {**os.environ, "HF_HUB_OFFLINE": "1"}
            for key in (
                "MTL_SHADER_VALIDATION",
                "SPLASH_SERVING_TRACE",
                "SPLASH_LIVE_VERIFY_THRESHOLD",
                "SPLASH_LIVE_GDN",
                "SPLASH_PREFILL_COST_MODEL",
                "SPLASH_BENCH_FIXED_VERIFY_ROWS",
            ):
                environment.pop(key, None)
            environment.update(record["environment"])
            with (args.output / f"{record['id']}.log").open("w") as log:
                value = subprocess.run(
                    record["command"],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
            record.update(
                exit_status=value.returncode,
                finished_unix=time.time(),
                status="executed" if value.returncode == 0 else "failed",
            )
            for artifact, expected in report["artifacts"].items():
                if fingerprint(Path(artifact)) != expected:
                    raise RuntimeError(
                        f"artifact changed during qualification: {artifact}"
                    )
            save()
            if value.returncode:
                raise RuntimeError(f"job failed: {record['id']}; inspect its log")
        report.update(status="executed", complete=True)
    except BaseException as error:
        report.update(status="incomplete", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
