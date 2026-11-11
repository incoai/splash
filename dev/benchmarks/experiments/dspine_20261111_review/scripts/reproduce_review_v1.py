#!/usr/bin/env python3
"""Rebuild the final-review sources, run native fixtures, or reanalyze saved ABBA rounds.

Keep this helper under scripts/ beside study_driver_v2.mm, with patches/ and
notes/ in the parent report directory. Builds and functional runs never replace
an existing output. A functional run is not a controlled throughput comparison.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

BASE = "37bdd4f8710319bfad43709de2725df89ef776d3"
REPORT = Path(__file__).resolve().parents[1]
PATCHES = {
    "baseline": None,
    "qkv": "qkv-narrow-v1.patch",
    "combined-v2": "combined-v2.patch",
}
ASSET_SHA256 = {
    "patches/qkv-narrow-v1.patch": "6a830fd355f3373d006ef4debf7081ff7305654fce9538ee8ae91b4438959325",
    "patches/combined-v2.patch": "f27dc90a8d0104b7350f03c58a3ad72ff35c3aa7f5c09f29657cf14680e7e60d",
    "scripts/study_driver_v2.mm": "2b0b3898db9bee8941e169d8852c8f6659f256f7d00e9de9f01a2a054783879d",
}
ANALYZER_SHA256 = {
    "analyze_abba_v2.py": "e2c3f42e4e5fce41d89251399beb7339bf9bdc4c4af51e15435cf984390da634",
    "abba.py": "7afc7b4fa24916acfc41745fc29cc659b5edada0bfa1ce55e19a685808475c0d",
}
BUNDLE_FILES = ("study-driver", "splash", "splash.metallib", "host-step.metallib")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def assets(report):
    found = {}
    for relative, expected in ASSET_SHA256.items():
        found[relative] = sha(report / relative)
        require(found[relative] == expected, f"frozen asset changed: {relative}")
    return found


def source_check(repo, report, variants):
    """Check patches against BASE in a temporary index; leave every tree alone."""
    found = assets(report)
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", BASE + "^{commit}"], text=True
    ).strip()
    require(head == BASE, "base commit differs")
    with tempfile.TemporaryDirectory(prefix="dspine-review-index-") as temporary:
        for variant in variants:
            patch = PATCHES[variant]
            if patch is None:
                continue
            environment = dict(
                os.environ, GIT_INDEX_FILE=str(Path(temporary) / (variant + ".index"))
            )
            subprocess.run(
                ["git", "-C", str(repo), "read-tree", BASE], env=environment, check=True
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "apply",
                    "--cached",
                    "--check",
                    "--whitespace=error",
                    str(report / "patches" / patch),
                ],
                env=environment,
                check=True,
            )
    return {"base": BASE, "variants": list(variants), "assets_sha256": found}


def check(args):
    variants = [args.variant] if args.variant else list(PATCHES)
    print(
        json.dumps(
            source_check(args.repo.resolve(), args.evidence.resolve(), variants),
            indent=2,
        )
    )


def build(args):
    repo, report, output = (
        args.repo.resolve(),
        args.evidence.resolve(),
        args.output.absolute(),
    )
    identity = source_check(repo, report, [args.variant])
    output.mkdir(parents=True, exist_ok=False)
    tree, bundle = output / "source", output / "bundle"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(tree), BASE],
        check=True,
    )
    patch = PATCHES[args.variant]
    if patch:
        subprocess.run(
            ["git", "-C", str(tree), "apply", str(report / "patches" / patch)],
            check=True,
        )
    python = str(Path(args.python).resolve())
    command = ["make", "-j" + str(args.jobs), "all"]
    if args.cpu:
        command.append("test-engine-cpu")
    if args.metal:
        command += [
            "build/engine-tests/linear-plan",
            "build/engine-tests/gguf-projection",
            "build/engine-tests/production-and-test.metallib",
        ]
    command.append("BUILD_ID_PYTHON=" + python)
    subprocess.run(command, cwd=tree, check=True)
    bundle.mkdir()
    driver = report / "scripts/study_driver_v2.mm"
    inputs = [
        driver,
        tree / "build/engine/libsplash.a",
        tree / "build/engine/BuildIdentity.hpp",
        tree / "build/splash.metallib",
        tree / "dev/benchmarks/PrefillWork.hpp",
        tree / "dev/benchmarks/TranscriptPrompts.hpp",
    ]
    before = {str(path): sha(path) for path in inputs}
    compile_command = [
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
    subprocess.run(compile_command, check=True)
    require(
        before == {str(path): sha(path) for path in inputs},
        "study build inputs changed",
    )
    for name in BUNDLE_FILES[1:]:
        shutil.copy2(tree / "build" / name, bundle / name)
    tests = []
    if args.metal:
        environment = dict(os.environ, MTL_SHADER_VALIDATION="1")
        tests = [
            [
                str(tree / "build/engine-tests/linear-plan"),
                str(bundle / "splash.metallib"),
            ],
            [
                str(tree / "build/engine-tests/gguf-projection"),
                str(tree / "build/engine-tests/production-and-test.metallib"),
                "--register16",
            ],
        ]
        if args.variant == "combined-v2":
            tests.append(
                [
                    str(tree / "build/engine-tests/gguf-projection"),
                    str(tree / "build/engine-tests/production-and-test.metallib"),
                    "--m48",
                ]
            )
        for test in tests:
            subprocess.run(test, cwd=tree, env=environment, check=True)
    require(
        identity["assets_sha256"] == assets(report),
        "frozen source assets changed during build",
    )
    record = {
        **identity,
        "variant": args.variant,
        "source": str(tree),
        "build_command": command,
        "driver_command": compile_command,
        "driver_inputs_sha256": before,
        "cpu_tests_requested": args.cpu,
        "shader_validation_commands": tests,
        "artifacts": {name: sha(bundle / name) for name in BUNDLE_FILES},
        "scope": "Fresh rebuild. Pin SDK/compiler for binary reproduction; hashes need not match another toolchain.",
    }
    with (output / "build.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(output)


def run(args):
    build_root = args.build.resolve()
    record = json.loads((build_root / "build.json").read_text())
    require(
        record["base"] == BASE and record["variant"] in PATCHES,
        "unexpected build source identity",
    )
    bundle = build_root / "bundle"
    require(set(record["artifacts"]) == set(BUNDLE_FILES), "unexpected bundle file set")
    before = {name: sha(bundle / name) for name in BUNDLE_FILES}
    require(before == record["artifacts"], "bundle changed since build")
    model, cohort, output = (
        args.model_root.resolve(),
        args.cohort.resolve(),
        args.output.absolute(),
    )
    require(model.is_dir() and cohort.is_file(), "model root or cohort does not exist")
    cohort_before = sha(cohort)
    command = [
        str(bundle / "study-driver"),
        str(bundle / "splash.metallib"),
        str(model),
        "--draft-block-size",
        str(args.rows),
        "--cohort",
        str(cohort),
        "--samples",
        "1",
        "--ane-ffn-share",
        "0",
        "--max-context",
        str(args.context),
        "--temperature",
        "0",
    ]
    if args.width > 1:
        command += ["--study-width", str(args.width)]
    environment = dict(os.environ)
    if args.shader:
        environment["MTL_SHADER_VALIDATION"] = "1"
    stderr, receipt = Path(str(output) + ".stderr"), Path(str(output) + ".run.json")
    with ExitStack() as stack:
        streams = [
            stack.enter_context(path.open("x")) for path in (output, stderr, receipt)
        ]
        code = subprocess.run(
            command, stdout=streams[0], stderr=streams[1], env=environment
        ).returncode
        after = {name: sha(bundle / name) for name in BUNDLE_FILES}
        cohort_after = sha(cohort)
        json.dump(
            {
                "command": command,
                "exit_code": code,
                "build_manifest_sha256": sha(build_root / "build.json"),
                "bundle_sha256": before,
                "bundle_sha256_after": after,
                "cohort_sha256": cohort_before,
                "cohort_sha256_after": cohort_after,
                "shader_validation_requested": args.shader,
                "scope": "One functional fixture; no controlled throughput claim.",
            },
            streams[2],
            indent=2,
        )
        streams[2].write("\n")
    require(before == after, "bundle changed during functional fixture")
    require(cohort_before == cohort_after, "cohort changed during functional fixture")
    require(
        code == 0,
        f"functional fixture failed with exit code {code}; output and receipt retained",
    )
    print(
        "Saved functional output arrays; this single run is not a controlled throughput comparison."
    )


def analyze(args):
    evidence = args.evidence.resolve()
    protocol_path = args.protocol.resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    require(
        protocol.get("analyzer_file") == "analyze_abba_v2.py",
        "unexpected native ABBA analyzer",
    )
    scripts = evidence / "scripts"
    for name, expected in ANALYZER_SHA256.items():
        field = "analyzer_sha256" if name != "abba.py" else "abba_sha256"
        require(
            protocol[field] == expected and sha(scripts / name) == expected,
            f"frozen analyzer differs: {name}",
        )
    original_abba = sys.modules.get("abba")
    try:
        for name, path in [
            ("abba", scripts / "abba.py"),
            ("frozen_native_review", scripts / "analyze_abba_v2.py"),
        ]:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            if name == "abba":
                sys.modules[name] = module
            spec.loader.exec_module(module)
        result = module.evaluate(evidence, protocol)
    finally:
        if original_abba is None:
            sys.modules.pop("abba", None)
        else:
            sys.modules["abba"] = original_abba
    require(
        protocol_path.read_bytes() == protocol_bytes, "protocol changed during analysis"
    )
    for name, expected in ANALYZER_SHA256.items():
        require(
            sha(scripts / name) == expected, f"analyzer changed during analysis: {name}"
        )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "qualified": result["qualified"],
                "groups": [group["qualified"] for group in result["groups"]],
            }
        )
    )


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)
    command = commands.add_parser(
        "check", help="validate frozen assets and patches without a checkout or build"
    )
    command.add_argument("--repo", type=Path, required=True)
    command.add_argument("--evidence", type=Path, default=REPORT)
    command.add_argument("--variant", choices=PATCHES)
    command.set_defaults(function=check)
    command = commands.add_parser(
        "build",
        help="create an exclusive detached worktree and rebuild its native bundle",
    )
    command.add_argument("--repo", type=Path, required=True)
    command.add_argument("--evidence", type=Path, default=REPORT)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--variant", choices=PATCHES, required=True)
    command.add_argument("--python", default=sys.executable)
    command.add_argument("--jobs", type=positive, default=2)
    command.add_argument("--cpu", action="store_true", help="also run test-engine-cpu")
    command.add_argument(
        "--metal",
        action="store_true",
        help="also run targeted Metal checks with shader validation",
    )
    command.set_defaults(function=build)
    command = commands.add_parser(
        "run", help="save one functional native fixture using a verified rebuilt bundle"
    )
    command.add_argument("--build", type=Path, required=True)
    command.add_argument("--model-root", type=Path, required=True)
    command.add_argument("--cohort", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--rows", type=int, choices=[8, 16], default=8)
    command.add_argument("--width", type=int, choices=[1, 2, 3, 4], default=1)
    command.add_argument("--context", type=positive, default=32768)
    command.add_argument("--shader", action="store_true")
    command.set_defaults(function=run)
    command = commands.add_parser(
        "analyze",
        help="call the unchanged original native ABBA evaluator on saved evidence",
    )
    command.add_argument("--evidence", type=Path, default=REPORT)
    command.add_argument("--protocol", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=analyze)
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        args.function(args)
    except (
        OSError,
        ValueError,
        AssertionError,
        KeyError,
        subprocess.CalledProcessError,
    ) as error:
        root.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
