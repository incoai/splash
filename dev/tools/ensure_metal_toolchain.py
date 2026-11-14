"""Verify CI's Metal compiler/linker before depending on Apple's asset catalog.

No SDK selection, production configuration or GPU execution is changed. A shim
path or successful component download alone is insufficient: compile and link
Metal 4.0/BF16 source with the production deployment target before proceeding.
"""

import shlex
import subprocess
import tempfile
from pathlib import Path

ATTEMPTS = 3
SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void ci_probe(device bfloat *output [[buffer(0)]],
                     uint index [[thread_position_in_grid]]) {
  output[index] = bfloat(1.0f);
}
"""


def run(arguments, timeout=60):
    print("+ " + shlex.join(arguments), flush=True)
    try:
        return subprocess.run(arguments, timeout=timeout, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Metal setup command failed: {error}", flush=True)
        return False


def probe(root):
    source, air, library = (
        root / name for name in ("probe.metal", "probe.air", "probe.metallib")
    )
    # Never let outputs from an earlier attempt qualify a later failed attempt.
    air.unlink(missing_ok=True)
    library.unlink(missing_ok=True)
    source.write_text(SOURCE)
    if not run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal4.0",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-mmacosx-version-min=26.4",
            "-c",
            str(source),
            "-o",
            str(air),
        ]
    ):
        return False
    if not air.is_file() or not air.stat().st_size:
        return False
    return (
        run(["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(library)])
        and library.is_file()
        and library.stat().st_size > 0
    )


def main():
    # Log which runner SDK is being used without mutating xcode-select.
    run(["xcode-select", "-p"])
    run(["xcodebuild", "-version"])
    run(["xcrun", "-sdk", "macosx", "--show-sdk-path"])
    with tempfile.TemporaryDirectory(prefix="splash-metal-check-") as temporary:
        root = Path(temporary)
        if probe(root):
            print(
                "Existing Metal compiler and linker verified; no download needed.",
                flush=True,
            )
            return 0
        for attempt in range(1, ATTEMPTS + 1):
            print(
                f"Metal toolchain installation attempt {attempt}/{ATTEMPTS}", flush=True
            )
            installed = run(
                ["xcodebuild", "-downloadComponent", "MetalToolchain"], timeout=180
            )
            # An installer can return an error after making the tools usable.
            # Conversely, success without a working compiler must remain failure.
            if probe(root):
                print(
                    f"Metal compiler and linker verified (installer success={installed}).",
                    flush=True,
                )
                return 0
            if attempt < ATTEMPTS:
                run(["sleep", "10"], timeout=15)
    print(
        "Metal compiler/linker remain unusable after three attempts; inspect the Xcode/SDK and catalog errors above.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
