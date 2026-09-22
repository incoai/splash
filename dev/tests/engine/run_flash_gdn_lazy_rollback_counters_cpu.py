#!/usr/bin/env python3
"""Fresh, counters-only CPU build. No backend or model code executes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build/flash-gdn-lazy-counters-cpu-v1")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    build = root / args.build_dir
    build.mkdir(parents=True, exist_ok=True)
    binary = build / "flash-gdn-lazy-counters-cpu"
    sources = [
        root / "dev/tests/engine/flash_gdn_lazy_rollback_counters_cpu_test.cpp",
        root / "runtime/flash/FlashGDNLazyRollback.cpp",
    ]
    fingerprints = sources + [
        root / "runtime/flash/FlashGDNLazyRollback.hpp",
        root / "runtime/metal/MetalBackend.hpp",
        Path(__file__).resolve(),
    ]
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in fingerprints
    }
    command = [
        "xcrun",
        "-sdk",
        "macosx",
        "clang++",
        "-std=c++20",
        "-O1" if args.sanitize else "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Iruntime",
        "-mmacosx-version-min=27.0",
        "-DSPLASH_INT8_EXPERIMENT=1",
    ]
    if args.sanitize:
        command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]
    command += [str(path) for path in sources] + ["-Wl,-dead_strip", "-o", str(binary)]
    subprocess.run(command, cwd=root, check=True)
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == value
        for path, value in before.items()
    ), "source changed during build"
    result = subprocess.run(
        [str(binary)], cwd=root, check=True, text=True, capture_output=True
    )
    payload = json.loads(result.stdout)
    assert payload["pass"] and payload["gpu_commands"] == 0
    assert payload["metal_backend_constructions"] == 0
    report = {
        **payload,
        "fresh_counter_and_class_source_compiled": True,
        "backend_objects_linked": 0,
        "metal_framework_linked": False,
        "sanitizers": "address,undefined" if args.sanitize else None,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "files": [
            {"path": str(path), "sha256": value} for path, value in before.items()
        ],
    }
    destination = build / "cpu-qualification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "files"}
            | {"report": str(destination)}
        )
    )


if __name__ == "__main__":
    main()
