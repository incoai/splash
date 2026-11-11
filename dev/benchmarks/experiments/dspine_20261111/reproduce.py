#!/usr/bin/env python3
"""Rebuild the two sources, run functional fixtures, and recompute saved gates."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = "425ad7400f3a00d2a1e6745bdacfa2881922d544"
HERE = Path(__file__).resolve().parent


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tree = output / "source"
    subprocess.run(
        ["git", "-C", str(args.repo), "worktree", "add", "--detach", str(tree), BASE],
        check=True,
    )
    patch = (
        HERE
        / "patches"
        / (
            "m48-research-v1.patch"
            if args.variant == "research-m48"
            else "production-final-v1.patch"
        )
    )
    if args.variant != "control":
        subprocess.run(["git", "-C", str(tree), "apply", str(patch)], check=True)
    command = [
        "make",
        "-j" + str(args.jobs),
        "all",
        "test-engine-cpu",
        "BUILD_ID_PYTHON=" + str(Path(args.python).resolve()),
    ]
    subprocess.run(command, cwd=tree, check=True)
    bundle = output / "bundle"
    bundle.mkdir()
    driver = HERE / "sources/study_driver_v2.mm"
    command_compile = [
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
        str(tree / "build/engine/BuildIdentity.hpp"),
        str(driver),
        str(tree / "build/engine/libsplash.a"),
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-framework",
        "IOKit",
        "-framework",
        "IOSurface",
        "-o",
        str(bundle / "study-driver"),
    ]
    subprocess.run(command_compile, check=True)
    for name in ["splash", "splash.metallib", "host-step.metallib"]:
        shutil.copy2(tree / "build" / name, bundle / name)
    if args.metal:
        subprocess.run(
            [
                "make",
                "-j" + str(args.jobs),
                "test-engine-metal",
                "BUILD_ID_PYTHON=" + str(Path(args.python).resolve()),
            ],
            cwd=tree,
            check=True,
        )
    record = {
        "base": BASE,
        "variant": args.variant,
        "source": str(tree),
        "patch_sha256": sha(patch) if args.variant != "control" else None,
        "build_command": command,
        "driver_command": command_compile,
        "artifacts": {p.name: sha(p) for p in bundle.iterdir()},
        "scope": "Fresh rebuild identity. Compiler/library bytes need not match another SDK; pin toolchain for binary reproduction.",
    }
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")


def run(args):
    identity = json.loads((args.build / "build.json").read_text())
    if args.rows == 16:
        assert identity["variant"] in {"candidate", "research-m48"}
    bundle = args.build / "bundle"
    assert all(
        sha(bundle / name) == digest for name, digest in identity["artifacts"].items()
    ), "bundle changed"
    command = [
        str(bundle / "study-driver"),
        str(bundle / "splash.metallib"),
        str(args.model_root),
        "--draft-block-size",
        str(args.rows),
        "--cohort",
        str(args.cohort),
        "--samples",
        "1",
        "--ane-ffn-share",
        "0",
        "--max-context",
        str(args.context),
    ]
    if args.width > 1:
        command += ["--study-width", str(args.width)]
    environment = dict(os.environ)
    if args.shader:
        environment["MTL_SHADER_VALIDATION"] = "1"
    with (
        args.output.open("x") as stream,
        args.output.with_suffix(".stderr").open("x") as errors,
    ):
        subprocess.run(
            command, env=environment, stdout=stream, stderr=errors, check=True
        )
    print(
        "Saved functional output arrays. This single run is not a controlled throughput comparison."
    )


def compare(args):
    evidence = args.evidence.resolve()
    sys.path.insert(0, str(evidence / "scripts"))
    protocol = json.loads(
        (evidence / "notes" / (args.comparison + ".protocol.json")).read_text()
    )
    analyzer = protocol.get("analyzer_file", "analyze_abba_v1.py")
    assert analyzer in {"analyze_abba_v1.py", "analyze_abba_v2.py"}
    module_path = evidence / "scripts" / analyzer
    assert sha(evidence / "scripts" / "abba.py") == protocol["abba_sha256"]
    spec = importlib.util.spec_from_file_location("frozen_abba_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert sha(module_path) == protocol["analyzer_sha256"]
    result = module.evaluate(evidence, protocol)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "qualified": result["qualified"],
                "groups": [v["qualified"] for v in result["groups"]],
            }
        )
    )


def frozen_script(args):
    # Exact original analyzer bytes; redirect only its filesystem root by
    # copying it into a temporary evidence layout. Never overwrite raw evidence.
    evidence = args.evidence.resolve()
    script, result = {
        "sampling": ("analyze_sampling4_v1.py", "sampling4-comparison-v1.json"),
        "prefix": ("analyze_prefix_numerics_v2.py", "prefix-numerics-v2.json"),
        "refinement": (
            "analyze_refinement_saved_v1.py",
            "refinement-saved-check-v1.json",
        ),
    }[args.action]
    if args.action == "sampling":
        nomination = json.loads(
            (evidence / "notes/sampling4-nominations-v1.json").read_text()
        )
        assert sha(evidence / "scripts" / script) == nomination["analyzer_sha256"]
    with tempfile.TemporaryDirectory(prefix="dspine-reanalysis-") as temporary:
        work = Path(temporary)
        (work / "scripts").mkdir()
        shutil.copy2(evidence / "scripts" / script, work / "scripts" / script)
        (work / "results").mkdir()
        for path in (evidence / "results").iterdir():
            if path.is_file() and path.name != result:
                (work / "results" / path.name).symlink_to(path)
        for name in ["notes", "cohorts", "logit-probes", "refinement-probes"]:
            if (evidence / name).is_dir():
                (work / name).symlink_to(evidence / name, target_is_directory=True)
        subprocess.run([sys.executable, str(work / "scripts" / script)], check=True)
        shutil.copy2(work / "results" / result, args.output)


def verify(args):
    manifest = json.loads(args.manifest.read_text())
    count = 0
    for relative, expected in manifest["files"].items():
        path = args.evidence / relative
        assert path.is_file() and path.stat().st_size == expected["bytes"], relative
        assert sha(path) == expected["sha256"], relative
        count += 1
    print("Verified", count, "evidence files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser("build")
    command.add_argument("--repo", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument(
        "--variant", choices=["control", "candidate", "research-m48"], required=True
    )
    command.add_argument("--python", default=sys.executable)
    command.add_argument("--jobs", type=int, default=2)
    command.add_argument("--metal", action="store_true")
    command.set_defaults(function=build)
    command = commands.add_parser("run")
    command.add_argument("--build", type=Path, required=True)
    command.add_argument("--model-root", type=Path, required=True)
    command.add_argument("--cohort", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--rows", type=int, choices=[8, 16], default=8)
    command.add_argument("--width", type=int, choices=[1, 2, 3, 4], default=1)
    command.add_argument("--context", type=int, default=32768)
    command.add_argument("--shader", action="store_true")
    command.set_defaults(function=run)
    command = commands.add_parser("analyze")
    command.add_argument("--comparison", required=True)
    command.set_defaults(function=compare)
    for name in ["sampling", "prefix", "refinement"]:
        command = commands.add_parser(name)
        command.set_defaults(function=frozen_script)
    for name in ["analyze", "sampling", "prefix", "refinement"]:
        command = commands.choices[name]
        command.add_argument("--evidence", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("verify")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--evidence", type=Path, required=True)
    command.set_defaults(function=verify)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
