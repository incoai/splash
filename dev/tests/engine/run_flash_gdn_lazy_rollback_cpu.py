#!/usr/bin/env python3
"""Fresh CPU compilation and metadata checks; never construct Metal/models."""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build/flash-gdn-lazy-rollback-cpu-v1")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    build = root / args.build_dir
    build.mkdir(parents=True, exist_ok=True)
    binary = build / "flash-gdn-lazy-rollback-cpu"
    common = [
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
        common += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]
    dependencies = [
        root / name
        for name in (
            "runtime/flash/FlashGDN.cpp",
            "runtime/flash/FlashGDNFused.cpp",
            "runtime/metal/MetalBackend.mm",
            "runtime/metal/DeviceCapabilities.cpp",
        )
    ]
    sources = [
        root / name
        for name in (
            "dev/tests/engine/flash_gdn_lazy_rollback_cpu_test.cpp",
            "runtime/flash/FlashGDNLazyRollback.cpp",
            "runtime/flash/FlashForward.cpp",
        )
    ]
    fingerprints = (
        dependencies
        + sources
        + [
            root / name
            for name in (
                "runtime/flash/FlashGDNLazyRollback.hpp",
                "runtime/flash/FlashForward.hpp",
                "runtime/metal/abi/FlashGDNLazyRollback.h",
                "runtime/metal/MetalBackend.hpp",
                "runtime/metal/kernels/shared/flash_gdn_lazy_rollback.metal",
            )
        ]
    )
    fingerprints += [Path(__file__).resolve()]
    before = {path: digest(path) for path in fingerprints}
    objects = [build / f"fresh-{path.stem}.o" for path in dependencies]

    def compile_dependency(pair):
        source, target = pair
        invocation = common + (["-fobjc-arc"] if source.suffix == ".mm" else [])
        subprocess.run(
            invocation + ["-c", str(source), "-o", str(target)], cwd=root, check=True
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(compile_dependency, zip(dependencies, objects)))
    # Mach-O dead stripping lets us call the real fresh-compiled Forward
    # metadata method without retaining or constructing unrelated model code.
    subprocess.run(
        common
        + [str(path) for path in sources + objects]
        + [
            "-Wl,-dead_strip",
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-framework",
            "IOKit",
            "-o",
            str(binary),
        ],
        cwd=root,
        check=True,
    )
    assert all(digest(path) == value for path, value in before.items()), (
        "source changed during compile"
    )
    cases = [
        ("@unset", "@unset", "0"),
        ("@unset", "malformed", "0"),
        ("0", "1", "0"),
        ("0", "@unset", "0"),
        ("0", "malformed", "0"),
        ("1", "1", "1"),
    ]
    for value in ("", "true", "false", "01", "00", " 1", "1 ", "-1", "2"):
        cases.append((value, "1", "reject"))
    for prerequisite in ("@unset", "0", "malformed", "01", " 1"):
        cases.append(("1", prerequisite, "reject"))
    results = []
    for parameters in [None] + cases:
        invocation = [str(binary)]
        if parameters is not None:
            invocation += ["--policy-value", *parameters]
        result = subprocess.run(
            invocation, cwd=root, check=True, capture_output=True, text=True
        )
        payload = json.loads(result.stdout)
        assert payload["pass"] and payload["gpu_commands"] == 0
        assert payload["metal_backend_constructions"] == 0
        results.append({"arguments": invocation[1:], **payload})
    report = {
        "pass": True,
        "process_cases": len(results),
        "cpu_checks_total": sum(row["cpu_checks"] for row in results),
        "footprint_checks": results[0]["cpu_checks"],
        "source_forward_planner_process_cases": sum(
            row["source_forward_planner_checked"] for row in results
        ),
        "policy_process_cases": len(cases),
        "gpu_commands": 0,
        "metal_backend_constructions": 0,
        "all_dependency_objects_recompiled_here": True,
        "sanitizers": "address,undefined" if args.sanitize else None,
        "command_timing_size_bytes": 200,
        "binary_sha256": digest(binary),
        "files": [
            {"path": str(path), "sha256": value} for path, value in before.items()
        ]
        + [{"path": str(path), "sha256": digest(path)} for path in objects],
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
