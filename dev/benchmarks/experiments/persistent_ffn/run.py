"""Compare native, static-helper and persistent weighted AF4G64 feed-forward blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "dev/benchmarks"))
import abba  # noqa: E402


def command(args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shader-validation", action="store_true")
    parser.add_argument("--register-inputs", action="store_true")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=17408)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--workers", type=int, default=160)
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    command(["make", "-j8", "build/engine/libsplash.a", "build/splash.metallib"])
    command(
        [
            "xcrun",
            "metal",
            "-std=metal4.0",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Iruntime",
            "-mmacosx-version-min=26.4",
            "-c",
            HERE / "kernels.metal",
            "-o",
            out / "experiment.air",
        ]
    )
    airs = sorted((ROOT / "build/metal").glob("*/*.air"))
    command(
        [
            "xcrun",
            "metallib",
            *airs,
            out / "experiment.air",
            "-o",
            out / "experiment.metallib",
        ]
    )
    command(
        [
            "xcrun",
            "clang++",
            "-std=c++20",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fobjc-arc",
            "-mmacosx-version-min=26.4",
            "-Iruntime",
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
    env.pop("MTL_SHADER_VALIDATION", None)
    if args.shader_validation:
        env["MTL_SHADER_VALIDATION"] = "1"
    results = []
    for rows in args.rows:
        path = out / f"r{rows}.json"
        with path.open("w") as stdout, (out / f"r{rows}.stderr").open("w") as stderr:
            subprocess.run(
                [
                    str(out / "benchmark"),
                    str(out / "experiment.metallib"),
                    str(args.hidden),
                    str(args.intermediate),
                    str(rows),
                    str(args.workers),
                    str(0 if args.shader_validation else args.blocks),
                    str(args.splits),
                    str(int(args.shader_validation)),
                    str(args.seed),
                    str(int(args.register_inputs)),
                ],
                cwd=ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=True,
                timeout=600,
            )
        raw = json.loads(path.read_text())
        for comparison in raw["result"]["comparisons"]:
            if not comparison["exact"]:
                raise RuntimeError("failed bitwise validation")
            if not args.shader_validation:
                blocks = [abba.compare(*block) for block in comparison["abba_us"]]
                comparison["aggregate"] = abba.compare_samples(
                    zip(*comparison["abba_us"])
                )
                comparison["blocks"] = blocks
        results.append(raw)
        print(json.dumps({"rows": rows, "completed": True}), flush=True)
    sources = [HERE / "benchmark.mm", HERE / "kernels.metal", HERE / "run.py"]
    files = sources + [
        ROOT / "build/engine/libsplash.a",
        out / "benchmark",
        out / "experiment.metallib",
    ]
    files += list(out.glob("r*.json"))
    summary = {
        "scope": "Complete weighted RMSNorm/SwiGLU/down/residual block, not attention or a whole model VM.",
        "configuration": vars(args) | {"output": str(out)},
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_diff": subprocess.check_output(
            ["git", "diff", "HEAD", "--", str(HERE)], cwd=ROOT, text=True
        ),
        "results": results,
        "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "limits": [
            "AF4G64 only; generated weights and activations, no model TTFT/tokens/sec measurement.",
            "Bounded polling detects lack of progress; it does not guarantee residency on other GPUs.",
            "native_selection_matched must be true before calling a case a production-selected comparison.",
            "Full readbacks outside normal timing; four extra warm-up ABBA blocks; all verdicts retained.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
