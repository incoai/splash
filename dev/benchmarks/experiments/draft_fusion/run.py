"""Build and run this standalone experiment; outputs stay under ignored build/."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "dev/benchmarks"))


def command(args):
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import abba

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-validation", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/experiments/draft-fusion"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    command(
        ["make", "-j8", "build/engine/libsplash.a", "build/metal/shared/residency.air"]
    )
    flags = [
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Iruntime",
        "-mmacosx-version-min=26.4",
    ]
    command(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal4.0",
            *flags,
            "-c",
            HERE / "kernels.metal",
            "-o",
            out / "experiment.air",
        ]
    )
    command(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metallib",
            ROOT / "build/metal/shared/residency.air",
            out / "experiment.air",
            "-o",
            out / "experiment.metallib",
        ]
    )
    command(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "clang++",
            "-std=c++20",
            "-fobjc-arc",
            *flags,
            HERE / "benchmark.mm",
            ROOT / "build/engine/libsplash.a",
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-framework",
            "IOKit",
            "-framework",
            "IOSurface",
            "-o",
            out / "benchmark",
        ]
    )
    env = os.environ.copy()
    if args.shader_validation:
        env["MTL_SHADER_VALIDATION"] = "1"
    else:
        env.pop("MTL_SHADER_VALIDATION", None)
    with (
        (out / "raw.json").open("w") as output,
        (out / "stderr.log").open("w") as error,
    ):
        subprocess.run(
            [str(out / "benchmark"), str(out / "experiment.metallib")],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=error,
            check=True,
            timeout=900,
        )
    raw = json.loads((out / "raw.json").read_text())
    comparisons = []
    for case in raw["cases"]:
        for variant in case.get("variants", [case]):
            rounds = variant["rounds_abba_us"]
            result = abba.compare_samples(zip(*rounds))
            result["blocks"] = [abba.compare(*r) for r in rounds]
            result["case"] = {
                k: v for k, v in case.items() if k not in ("variants", "rounds_abba_us")
            }
            if "mode" in variant:
                result["mode"] = variant["mode"]
            comparisons.append(result)
    identity = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_diff": subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=ROOT, text=True
        ),
        "macos": platform.mac_ver()[0],
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "compiler": subprocess.check_output(
            ["xcrun", "clang++", "--version"], text=True
        ).splitlines()[0],
        "shader_validation": args.shader_validation,
        "sha256": {
            str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p)
            for p in (
                HERE / "benchmark.mm",
                HERE / "kernels.metal",
                HERE / "run.py",
                ROOT / "build/engine/libsplash.a",
                out / "experiment.metallib",
                out / "benchmark",
                out / "raw.json",
            )
        },
    }
    result = {
        "identity": identity,
        "comparisons": comparisons,
        "verdict_counts": dict(collections.Counter(r["verdict"] for r in comparisons)),
        "model_ttft_ms": None,
        "model_tokens_per_second": None,
        "scope": "Component GPU timing; no model inference or end-to-end efficiency claim.",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), "verdict_counts": result["verdict_counts"]}))


if __name__ == "__main__":
    main()
