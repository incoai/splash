#!/usr/bin/env python3
"""Build isolated research variants and run the frozen native checks on macOS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = "33a488d4bbb3978b32b34e8bfc6cefa92e350d7f"
HERE = Path(__file__).resolve().parent
WIDE = {
    "block16",
    "wide-policy",
    "gateup-tiles",
    "gateup-shared16",
    "gdn-phased16",
    "linear-shared16",
}
GDN_FIXED_WIDE = {"wide-policy", "gateup-shared16", "gdn-phased16", "linear-shared16"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(command, **kw):
    print(" ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kw)


def compile_oracle(tree, source, destination, extra=()):
    call(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "clang++",
            "-std=c++20",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(tree / "runtime"),
            "-I" + str(tree / "dev/tests/engine"),
            "-mmacosx-version-min=26.4",
            "-fobjc-arc",
            *extra,
            source,
            tree / "build/engine/libsplash.a",
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-framework",
            "IOKit",
            "-framework",
            "IOSurface",
            "-o",
            destination,
        ]
    )


def build(args):
    if sys.platform != "darwin":
        raise SystemExit(
            "Native reproduction needs macOS and the recorded Metal 4 toolchain."
        )
    repo = args.repo.resolve()
    output = args.output.resolve()
    tree = output / "worktrees" / args.variant
    bundle = output / "artifacts" / args.variant
    if tree.exists() or bundle.exists():
        raise SystemExit(
            "Use a fresh variant output; existing worktrees/artifacts are preserved."
        )
    call(["git", "-C", repo, "cat-file", "-e", BASE + "^{commit}"])
    patch = HERE / "patches" / (args.variant + ".patch")
    if not patch.is_file():
        raise SystemExit("No exported source patch for " + args.variant)
    tree.parent.mkdir(parents=True, exist_ok=True)
    call(["git", "-C", repo, "worktree", "add", "--detach", tree, BASE])
    call(["git", "-C", tree, "apply", "--check", patch])
    call(["git", "-C", tree, "apply", patch])
    call(["git", "-C", tree, "diff", "--check"])
    tests = [
        "backend-benchmark",
        "weight-digests",
        "dspine",
        "draft-attention",
        "gdn-decode",
        "decode-profile",
        "linear-plan",
    ]
    call(
        [
            "make",
            "-C",
            tree,
            "-j" + str(args.jobs),
            "all",
            *["build/engine-tests/" + t for t in tests],
        ]
    )
    bundle.mkdir(parents=True)
    for name in ["splash", "splash.metallib", "host-step.metallib"]:
        shutil.copy2(tree / "build" / name, bundle / name)
    for name in tests:
        shutil.copy2(tree / "build/engine-tests" / name, bundle / name)
    driver = HERE / "sources/study_driver.mm"
    call(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "clang++",
            "-std=c++20",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(tree / "runtime"),
            "-I" + str(tree / "dev/benchmarks"),
            "-mmacosx-version-min=26.4",
            "-fobjc-arc",
            "-include",
            tree / "build/engine/BuildIdentity.hpp",
            driver,
            tree / "build/engine/libsplash.a",
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-framework",
            "IOKit",
            "-framework",
            "IOSurface",
            "-o",
            bundle / "study-driver",
        ]
    )
    if args.variant in GDN_FIXED_WIDE:
        compile_oracle(
            tree,
            HERE / "sources/gdn_plain16_oracle.mm",
            bundle / "gdn-plain16",
            [
                '-DDSPINE_GDN_ORACLE_SOURCE="'
                + str(tree / "dev/tests/engine/gdn_decode_metal_test.mm")
                + '"'
            ],
        )
    if args.variant in {"gateup-tiles", "gateup-shared8", "gateup-shared16"}:
        oracle = (
            "gateup_shared_oracle.mm"
            if args.variant.startswith("gateup-shared")
            else "gateup_tile_oracle.mm"
        )
        compile_oracle(tree, HERE / ("sources/" + oracle), bundle / "gateup-oracle")
    if args.variant == "linear-shared16":
        compile_oracle(
            tree,
            HERE / "sources/linear_shared_oracle.mm",
            bundle / "linear-shared-oracle",
        )
    record = {
        "base": BASE,
        "variant": args.variant,
        "patch_sha256": sha(patch),
        "study_source_sha256": sha(driver),
        "artifact_sha256": {p.name: sha(p) for p in bundle.iterdir() if p.is_file()},
        "scope": "Fresh build of the exported research source. Toolchain/path changes may change binary hashes; no production rollout implied.",
    }
    (bundle / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(bundle)


def run(args):
    bundle = args.output.resolve() / "artifacts" / args.variant
    if not (bundle / "manifest.json").exists():
        raise SystemExit("Build this variant first.")
    record = json.loads((bundle / "manifest.json").read_text())
    for name, digest in record["artifact_sha256"].items():
        if sha(bundle / name) != digest:
            raise SystemExit("Artifact changed: " + name)
    env = os.environ.copy()
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator or key not in {
            "SPLASH_DSPINE_DRAFT_BITS",
            "SPLASH_DSPINE_DRAFT_SCOPE",
            "SPLASH_DSPINE_HEAD_BITS",
            "SPLASH_DSPINE_HEAD_CHECKPOINT",
            "SPLASH_GATE_UP_SUBROWS",
            "SPLASH_GATE_UP_SHARED",
            "SPLASH_GATE_UP_TWO_PASS64",
            "SPLASH_GDN_PHASED",
            "SPLASH_LINEAR_SHARED_ROWS",
        }:
            raise SystemExit(
                "Use an explicitly documented research environment setting."
            )
        env[key] = value
    if args.shader:
        env.update(
            MTL_SHADER_VALIDATION="1", MTL_SHADER_VALIDATION_REPORT_TO_STDERR="1"
        )
    if args.mode == "oracles":
        names = ["dspine", "draft-attention"]
        if args.variant in GDN_FIXED_WIDE:
            names.append("gdn-plain16")
        elif args.variant not in WIDE:
            names.append("gdn-decode")
        else:
            print(
                "GDN is not checked by this original fixture; use wide-policy for the corrected plain16 oracle.",
                flush=True,
            )
        for name in names:
            call(
                [bundle / name, bundle / "splash.metallib"],
                env=env,
                timeout=args.timeout,
            )
        if args.variant in {"gateup-tiles", "gateup-shared8", "gateup-shared16"}:
            call(
                [bundle / "gateup-oracle", bundle / "splash.metallib"],
                env=env,
                timeout=args.timeout,
            )
        if args.variant == "linear-shared16":
            call(
                [bundle / "linear-shared-oracle", bundle / "splash.metallib"],
                env=env,
                timeout=args.timeout,
            )
        return
    if args.model_root is None:
        raise SystemExit("--model-root is required for native model measurements.")
    command = [
        bundle / "study-driver",
        bundle / "splash.metallib",
        args.model_root.resolve(),
        "--scenario",
        "study" if args.mode == "cohort" else args.mode,
        "--samples",
        str(args.samples),
        "--temperature",
        str(args.temperature),
        "--seed-base",
        str(args.seed_base),
        "--ane-ffn-share",
        "0",
        "--max-context",
        "32768",
    ]
    if args.mode == "cohort":
        command.extend(["--cohort", HERE / "cohorts" / args.cohort])
    logs = args.output.resolve() / "runs"
    logs.mkdir(exist_ok=True)
    stdout = logs / (args.name + ".json")
    stderr = logs / (args.name + ".stderr")
    if stdout.exists() or stderr.exists():
        raise SystemExit("Use a new --name; prior results are retained.")
    with stdout.open("w") as out, stderr.open("w") as err:
        timed_out = False
        try:
            completed = subprocess.run(
                list(map(str, command)),
                check=False,
                env=env,
                stdout=out,
                stderr=err,
                timeout=args.timeout,
            )
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = -9
            timed_out = True
    receipt = {
        "command": list(map(str, command)),
        "declared_env": dict(item.split("=", 1) for item in args.env),
        "shader_validation": args.shader,
        "exit_code": code,
        "timed_out": timed_out,
        "stdout_sha256": sha(stdout),
        "stderr_sha256": sha(stderr),
    }
    (logs / (args.name + ".receipt.json")).write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))
    raise SystemExit(code)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build")
    b.add_argument("--repo", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--variant", required=True)
    b.add_argument("--jobs", type=int, default=6)
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--variant", required=True)
    r.add_argument(
        "--mode",
        choices=["oracles", "cohort", "decode", "acceptance"],
        default="cohort",
    )
    r.add_argument("--cohort", default="cohort31.txt")
    r.add_argument("--model-root", type=Path)
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--temperature", type=float, default=0)
    r.add_argument("--seed-base", type=int, default=20261110)
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--shader", action="store_true")
    r.add_argument("--name", default="study")
    r.add_argument("--timeout", type=int, default=7200)
    args = p.parse_args()
    if "/" in args.variant or args.variant.startswith("."):
        p.error("variant is a published simple name")
    if args.action == "run" and (
        Path(args.cohort).name != args.cohort or Path(args.name).name != args.name
    ):
        p.error("cohort/name must be simple filenames")
    if args.action == "build" and not 1 <= args.jobs <= 18:
        p.error("--jobs must be 1..18")
    if args.action == "run":
        if (
            not 1 <= args.samples <= 2000
            or args.seed_base < 0
            or args.timeout < 1
            or not math.isfinite(args.temperature)
            or args.temperature < 0
        ):
            p.error("invalid samples/seed/timeout/temperature")
        if args.mode != "cohort" and args.temperature != 0:
            p.error(
                "positive-temperature measurement uses --mode cohort; decode/acceptance are greedy"
            )
    (build if args.action == "build" else run)(args)


if __name__ == "__main__":
    main()
