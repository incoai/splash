#!/usr/bin/env python3
"""Fingerprint the code that turns source tensors into prepared weight bytes.

A prepared file's key hashes its adapter's identity, its plan and the source
tensors it reads. The identity covers only the sources of the code that
writes the bytes, by content: an inference change or a rename keeps every
prepared file, an edit to one of these files prepares its files again."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Run as a script (make): import siblings as the dev.tools package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "dev.tools"

from .build_identity import update_if_changed

DOMAIN = b"splash-weight-preparation-identity-v1\0"
# Each adapter's byte-defining code. Every project header these files
# include is either listed here or reviewed in the identity test as defining
# no prepared byte.
INPUTS = {
    "AFFINE": (
        "runtime/model/AffinePreparation.cpp",
        "runtime/model/WeightLayout.hpp",
    ),
    "GGUF": (
        "runtime/model/GgufPreparation.cpp",
        "runtime/model/GgufImageLayout.hpp",
        "runtime/model/Bfloat16.hpp",
        "runtime/metal/abi/GgufRepack.h",
        "runtime/metal/abi/QuantFormat.h",
        "runtime/metal/kernels/shared/gguf_repack.metal",
    ),
    "VISION": (
        "runtime/model/VisionPreparation.cpp",
        "runtime/model/WeightLayout.hpp",
        "runtime/model/Bfloat16.hpp",
    ),
}


class PreparationIdentityError(RuntimeError):
    pass


def identity(root: Path, inputs) -> str:
    """SHA-256 of the inputs' contents, whatever their names or order."""
    digests = []
    for name in inputs:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise PreparationIdentityError(
                f"preparation input is not a regular file: {name}"
            )
        digests.append(hashlib.sha256(path.read_bytes()).digest())
    hasher = hashlib.sha256(DOMAIN)
    for digest in sorted(digests):
        hasher.update(digest)
    return hasher.hexdigest()


def header(root: Path) -> bytes:
    lines = ["// Generated preparation identities; do not edit.", "#pragma once"]
    for name, inputs in INPUTS.items():
        lines.append(f'#define SPLASH_{name}_PREPARATION_ID "{identity(root, inputs)}"')
    return ("\n".join(lines) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument(
        "--stale",
        action="store_true",
        help="print the header's path when it differs from the identities, write nothing",
    )
    args = parser.parse_args()
    if not args.stale:
        update_if_changed(args.header, header(args.root))
    elif not args.header.is_file() or args.header.read_bytes() != header(args.root):
        print(args.header)


if __name__ == "__main__":
    try:
        main()
    except PreparationIdentityError as error:
        raise SystemExit(f"weight_preparation_identity: {error}") from error
