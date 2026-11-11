"""Build the focused width diagnostic without replacing production artifacts."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--expected-commit", default="37bdd4f8710319bfad43709de2725df89ef776d3"
    )
    args = parser.parse_args()
    tree = args.tree.resolve()
    out = args.output.resolve()
    source = Path(__file__).resolve().with_name("width_probe_driver_v1.mm")
    commit = subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == args.expected_commit
    out.mkdir(parents=True, exist_ok=False)
    inputs = [
        source,
        tree / "runtime/model/Runtime.cpp",
        tree / "runtime/model/RuntimeInstrumentation.hpp",
        tree / "build/engine/libsplash.a",
        tree / "build/engine/BuildIdentity.hpp",
        tree / "build/splash.metallib",
        tree / "build/host-step.metallib",
    ]
    before = {str(path): sha(path) for path in inputs}
    source_diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(tree),
            "diff",
            "--binary",
            "HEAD",
            "--",
            "runtime",
            "Makefile",
            "dev/native.mk",
            "dev/tools",
        ]
    )
    (out / "source-diff.patch").write_bytes(source_diff)
    instrumented = tree / "build/engine/model/RuntimeInstrumented.o"
    build = [
        "make",
        "-C",
        str(tree),
        "-j2",
        "build/engine/model/RuntimeInstrumented.o",
        "BUILD_ID_PYTHON=" + str(args.python.resolve()),
    ]
    with (
        (out / "build.stdout").open("x") as stdout,
        (out / "build.stderr").open("x") as stderr,
    ):
        subprocess.run(build, stdout=stdout, stderr=stderr, check=True)
    command = [
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
        str(source),
        str(instrumented),
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
        str(out / "width-probe"),
    ]
    with (
        (out / "compile.stdout").open("x") as stdout,
        (out / "compile.stderr").open("x") as stderr,
    ):
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
    assert before == {str(path): sha(path) for path in inputs}, (
        "a production input changed during the build"
    )
    for name in ["splash.metallib", "host-step.metallib"]:
        shutil.copy2(tree / "build" / name, out / name)
    record = {
        "commit": commit,
        "source_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
        "production_inputs": before,
        "instrumented_runtime_sha256": sha(instrumented),
        "build_command": build,
        "link_command": command,
        "artifacts": {
            name: sha(out / name)
            for name in ["width-probe", "splash.metallib", "host-step.metallib"]
        },
        "scope": "Same source runtime linked with its existing instrumentation path. Host copies perturb timing. "
        "Each run uses one cold serial greedy case and the original cohort cap.",
    }
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
