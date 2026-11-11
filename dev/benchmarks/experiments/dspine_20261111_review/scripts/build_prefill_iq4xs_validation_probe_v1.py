#!/usr/bin/env python3
"""CPU-link a minimal prefill probe against an existing tree and supplied library; never run it."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    tree, library, output = args.tree.resolve(), args.library.resolve(), args.output.absolute()
    source = Path(__file__).with_name("prefill_iq4xs_validation_probe_v1.mm").resolve()
    tests = tree / "dev/tests/engine"
    inputs = [source, tree / "build/engine/libsplash.a", tree / "build/engine/BuildIdentity.hpp", library,
              tests / "gguf_projection_test.mm", tests / "GgufFormatReference.hpp", tests / "LinearNumerics.hpp",
              tests / "TestBuffers.hpp", tests / "TestChecks.hpp", tree / "runtime/ops/Linear.hpp"]
    before = {str(path): sha(path) for path in inputs}
    output.mkdir(parents=True, exist_ok=False)
    command = ["xcrun", "-sdk", "macosx", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
               "-mmacosx-version-min=26.4", "-fobjc-arc", "-I" + str(tree / "runtime"), "-I" + str(tests),
               "-include", str(tree / "build/engine/BuildIdentity.hpp"), str(source),
               str(tree / "build/engine/libsplash.a"), "-framework", "Foundation", "-framework", "Metal",
               "-framework", "IOKit", "-framework", "IOSurface", "-o", str(output / "prefill-iq4xs-probe")]
    with (output / "build.stdout").open("x") as stdout, (output / "build.stderr").open("x") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
    if before != {str(path): sha(path) for path in inputs}:
        raise RuntimeError("probe build inputs changed")
    shutil.copy2(library, output / "probe.metallib")
    head = subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(tree), "diff", "--binary", "HEAD"])
    record = {"label": args.label, "tree_head": head, "tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
              "command": command, "inputs_sha256": before,
              "artifacts_sha256": {name: sha(output / name) for name in ["prefill-iq4xs-probe", "probe.metallib"]},
              "scope": "CPU compilation/link only. Parent owns all Metal inspection/dispatch runs."}
    with (output / "build.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
