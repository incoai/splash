#!/usr/bin/env python3
"""Compare the AIR of kernels in two Metal libraries.

Run as ``python dev/tools/kernel_identity.py BASELINE CANDIDATE [--match
PATTERN ...]``, each a metallib or a directory whose splash.metallib or
build/splash.metallib it compares. metal-objdump disassembles each function of
a library as an LLVM module of its own, which holds no source path or build
id. Two functions are identical when their modules' text is, but for the
module's offset in its library, which the disassembly prints before it.

By default it compares the kernels the decode path may run: every function
but those named for prefill (prefill_*, gguf_prefill_*), the vision encoder
(vision_*) or the prefill FFN's Neural Engine split (ane_ffn_*), which leaves
the decode, verify and draft kernels and the shared ones. --match takes shell
patterns of the names to compare instead. It prints per function whether it
is identical, different or in one library only, and exits 1 unless every one
is identical.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
import subprocess
import sys
from pathlib import Path

# The kernels only prefill or the vision encoder runs.
OTHER_PATHS = ("*prefill*", "vision_*", "ane_ffn_*")
# A function's header in the disassembly: its module's offset, then its name.
HEADER = re.compile(r"^0x[0-9a-f]+ -- (.+):$", re.MULTILINE)


def library(path: Path) -> Path:
    """The metallib a path names: itself, or the splash.metallib of a build
    directory or of a checkout's build/."""
    if path.is_file():
        return path
    for metallib in (path / "splash.metallib", path / "build/splash.metallib"):
        if metallib.is_file():
            return metallib
    raise FileNotFoundError(f"no splash.metallib in {path}")


def functions(metallib: Path) -> dict[str, str]:
    """The SHA-256 of each function's disassembled AIR in a metallib."""
    listing = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-objdump", "--metallib", "--disassemble"]
        + [str(metallib)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # The file name and section title, then each function's name and module.
    parts = HEADER.split(listing)[1:]
    names, modules = parts[0::2], parts[1::2]
    if not names:
        raise RuntimeError(f"metal-objdump disassembled no function of {metallib}")
    if len(set(names)) != len(names):
        raise RuntimeError(f"{metallib} names a function twice")
    # Blank lines separate the modules.
    return {
        name: hashlib.sha256(module.strip().encode()).hexdigest()
        for name, module in zip(names, modules)
    }


def matches(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def compare(baseline: dict[str, str], candidate: dict[str, str], chosen) -> dict:
    """The verdict on each function of either library that `chosen` takes."""
    verdicts = {}
    for name in sorted(baseline.keys() | candidate.keys()):
        if not chosen(name):
            continue
        if name not in baseline:
            verdicts[name] = "only in candidate"
        elif name not in candidate:
            verdicts[name] = "only in baseline"
        else:
            verdicts[name] = (
                "identical" if baseline[name] == candidate[name] else "different"
            )
    return verdicts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("baseline", "candidate"):
        parser.add_argument(
            name, type=Path, help="a metallib, or a build directory or checkout"
        )
    parser.add_argument(
        "--match",
        action="append",
        metavar="PATTERN",
        help="compare the functions whose name matches; repeatable "
        "(default: the kernels the decode path may run)",
    )
    args = parser.parse_args(argv)
    try:
        libraries = [library(args.baseline), library(args.candidate)]
    except FileNotFoundError as error:
        parser.error(str(error))
    try:
        baseline, candidate = (functions(metallib) for metallib in libraries)
    except subprocess.CalledProcessError as error:
        print(f"kernel identity: {error.stderr.strip()}", file=sys.stderr)
        return 2
    verdicts = compare(
        baseline,
        candidate,
        (lambda name: matches(name, args.match))
        if args.match
        else (lambda name: not matches(name, OTHER_PATHS)),
    )
    if not verdicts:
        parser.error("no function matches")
    for name, verdict in verdicts.items():
        print(f"{verdict}: {name}")
    differing = sum(verdict != "identical" for verdict in verdicts.values())
    print(
        f"kernel identity: {len(verdicts) - differing} of {len(verdicts)} "
        f"identical ({libraries[0]} against {libraries[1]})"
    )
    return 1 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
