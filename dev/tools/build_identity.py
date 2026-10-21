#!/usr/bin/env python3
"""Create and verify the deterministic Splash production build identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
DOMAIN = b"splash-production-build-identity-v1\0"
BUILD_ID_PATTERN = re.compile(rb"src-[0-9a-f]{64}")


class BuildIdentityError(RuntimeError):
    pass


def production_input_paths(root: Path = ROOT) -> tuple[str, ...]:
    sources = []
    source_root = root / "runtime"
    for suffix in ("*.c", "*.cc", "*.cpp", "*.h", "*.hpp", "*.m", "*.mm"):
        sources.extend(source_root.rglob(suffix))
    sources.extend((source_root / "metal" / "kernels").rglob("*.metal"))
    relative_sources = {path.relative_to(root).as_posix() for path in sources}
    return tuple(sorted(relative_sources | {"dev/tools/build_identity.py"}))


def _record(hasher, kind: bytes, name: bytes, value: bytes) -> None:
    hasher.update(kind)
    hasher.update(len(name).to_bytes(8, "big"))
    hasher.update(name)
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def source_digest(
    root: Path,
    inputs: tuple[str, ...] | list[str],
    constants: dict[str, str] | None = None,
) -> str:
    root = root.resolve()
    normalized = tuple(sorted(set(inputs)))
    if not normalized:
        raise BuildIdentityError("production build identity has no inputs")
    if len(normalized) != len(inputs):
        raise BuildIdentityError("production build identity inputs are duplicated")

    hasher = hashlib.sha256(DOMAIN)
    for name, value in sorted((constants or {}).items()):
        if not name or "\x00" in name:
            raise BuildIdentityError("build identity constant name is invalid")
        _record(hasher, b"constant\0", name.encode(), value.encode())

    for relative in normalized:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise BuildIdentityError(f"unsafe production input path: {relative}")
        path = root / relative_path
        try:
            if path.is_symlink() or not path.is_file():
                raise OSError("input is not a regular non-symlink file")
            content = path.read_bytes()
        except OSError as error:
            raise BuildIdentityError(
                f"cannot read production input {relative}: {error}"
            ) from error
        _record(hasher, b"file\0", relative_path.as_posix().encode(), content)
    return hasher.hexdigest()


def build_id(
    root: Path = ROOT,
    inputs: tuple[str, ...] | list[str] | None = None,
    constants: dict[str, str] | None = None,
) -> str:
    selected = production_input_paths(root) if inputs is None else inputs
    return "src-" + source_digest(root, selected, constants)


def header_bytes(identity: str) -> bytes:
    if re.fullmatch(r"src-[0-9a-f]{64}", identity) is None:
        raise BuildIdentityError("cannot render an invalid production build id")
    digest = identity.removeprefix("src-")
    return (
        "#pragma once\n"
        f'#define SPLASH_BUILD_ID "{identity}"\n'
        f'#define SPLASH_BUILD_SOURCE_SHA256 "{digest}"\n'
    ).encode()


def stamp_bytes(identity: str) -> bytes:
    return (
        json.dumps(
            {
                "build_id": identity,
                "schema_version": SCHEMA_VERSION,
                "source_sha256": identity.removeprefix("src-"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def update_if_changed(path: Path, content: bytes) -> bool:
    try:
        if path.read_bytes() == content:
            return False
    except FileNotFoundError:
        pass
    except OSError as error:
        raise BuildIdentityError(
            f"cannot read generated identity {path}: {error}"
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def parse_constants(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, content = value.partition("=")
        if not separator or not name or name in result:
            raise BuildIdentityError(f"invalid or duplicate --constant: {value}")
        result[name] = content
    return result


def expected_identity(args) -> str:
    return build_id(args.root, constants=parse_constants(args.constant))


def verify_generated(args, identity: str) -> None:
    expected_header = header_bytes(identity)
    expected_stamp = stamp_bytes(identity)
    try:
        actual_header = args.header.read_bytes()
        actual_stamp = args.stamp.read_bytes()
    except OSError as error:
        raise BuildIdentityError(
            f"cannot read generated build identity: {error}"
        ) from error
    if actual_header != expected_header or actual_stamp != expected_stamp:
        raise BuildIdentityError("generated build identity is stale")
    for binary in args.binary:
        try:
            embedded = {
                match.decode()
                for match in BUILD_ID_PATTERN.findall(binary.read_bytes())
            }
        except OSError as error:
            raise BuildIdentityError(
                f"cannot read identity binary {binary}: {error}"
            ) from error
        if embedded != {identity}:
            raise BuildIdentityError(
                f"binary {binary} embeds build identities {sorted(embedded)}, "
                f"expected only {identity}"
            )


def add_identity_arguments(parser) -> None:
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--constant", action="append", default=[])
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--stamp", type=Path, required=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write")
    add_identity_arguments(write)
    check = commands.add_parser("check")
    add_identity_arguments(check)
    check.add_argument("--binary", action="append", type=Path, default=[])
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    identity = expected_identity(args)
    if args.command == "write":
        header_changed = update_if_changed(args.header, header_bytes(identity))
        stamp_changed = update_if_changed(args.stamp, stamp_bytes(identity))
        print(
            json.dumps(
                {
                    "build_id": identity,
                    "header_changed": header_changed,
                    "stamp_changed": stamp_changed,
                },
                sort_keys=True,
            )
        )
        return 0
    verify_generated(args, identity)
    print(json.dumps({"build_id": identity, "verified": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildIdentityError as error:
        raise SystemExit(f"build_identity: {error}") from error
