#!/usr/bin/env python3
"""Compile and exercise production GDN ILP metadata; never load Metal/models."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build/flash-gdn-batch-ilp-cpu-v1")
    parser.add_argument("--backend-build", default="build/flash-gdn-batch-ilp-v2")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    build = root / args.build_dir
    build.mkdir(parents=True, exist_ok=True)
    binary = build / "flash-gdn-batch-ilp-cpu"
    sources = [
        root / "dev/tests/engine/flash_gdn_batch_ilp_cpu_test.cpp",
        root / "runtime/flash/FlashGDNBatchILP.cpp",
    ]
    backend = root / args.backend_build
    objects = [
        backend / f"fresh-{name}.o"
        for name in ("FlashGDN", "FlashGDNStaged", "MetalBackend", "DeviceCapabilities")
    ]
    for path in sources + objects:
        if not path.is_file():
            raise RuntimeError(f"missing required source/frozen fresh object: {path}")
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
    command += [str(path) for path in sources + objects]
    command += [
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-framework",
        "IOKit",
        "-o",
        str(binary),
    ]
    subprocess.run(command, cwd=root, check=True)

    # Each value is exercised in a fresh process because policy is frozen.
    cases = [
        ("@unset", "@unset", "@unset", "0"),
        ("@unset", "malformed", "malformed", "0"),
        ("0", "1", "1", "0"),
        ("0", "@unset", "@unset", "0"),
        ("0", "malformed", "malformed", "0"),
        ("1", "1", "1", "1"),
    ]
    for value in ("", "true", "false", "01", "00", " 1", "1 ", "-1", "2"):
        cases.append((value, "1", "1", "reject"))
    for staged, prefill in (
        ("@unset", "1"),
        ("0", "1"),
        ("1", "@unset"),
        ("1", "0"),
        ("@unset", "@unset"),
        ("0", "0"),
        ("malformed", "1"),
        ("1", "malformed"),
        ("01", "1"),
        ("1", " 1"),
    ):
        cases.append(("1", staged, prefill, "reject"))
    results = []
    for parameters in [None] + cases:
        invocation = [str(binary)]
        if parameters is not None:
            invocation += ["--policy-value", *parameters]
        result = subprocess.run(
            invocation, cwd=root, check=True, text=True, capture_output=True
        )
        payload = json.loads(result.stdout)
        assert payload["pass"] and payload["gpu_commands"] == 0
        assert payload["metal_backend_constructions"] == 0
        results.append({"arguments": invocation[1:], **payload})
    fingerprint_paths = (
        sources
        + objects
        + [
            root / "runtime/flash/FlashGDNBatchILP.hpp",
            root / "runtime/metal/abi/FlashGDNBatchILP.h",
            root / "runtime/metal/MetalBackend.hpp",
            Path(__file__).resolve(),
        ]
    )
    report = {
        "pass": True,
        "process_cases": len(results),
        "cpu_checks_total": sum(row["cpu_checks"] for row in results),
        "geometry_ownership_checks": results[0]["cpu_checks"],
        "policy_process_cases": len(cases),
        "gpu_commands": 0,
        "metal_backend_constructions": 0,
        "sanitizers": "address,undefined" if args.sanitize else None,
        "command_timing_size_bytes": 200,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "files": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in fingerprint_paths
        ],
        "cases": results,
    }
    destination = build / "cpu-qualification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("files", "cases")
            }
            | {"report": str(destination)}
        )
    )


if __name__ == "__main__":
    main()
