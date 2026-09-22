"""Tiny, independently checked persisted affine operands; no MLX/Metal imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import struct
from pathlib import Path

SCHEMA = "splash-local-affine-operands-v1"
ALIGNMENT = 16 * 1024
SOURCE_IDENTITY = "1" * 64
WEIGHTS_FINGERPRINT = "2" * 64
MATH_FORMATS = {
    "BF16": "mlx-affine-f32-reconstruct-rounded-bf16-row-major-v1",
    "F32": "original-affine-contractoff-f32-coefficients-row-major-v1",
}
MATH_VERSION = "contract-off,f32-separated-mul-add,bf16-round-to-nearest-even,v1"
MATH_DIGEST = hashlib.sha256(
    (
        MATH_FORMATS["BF16"] + "\n" + MATH_FORMATS["F32"] + "\n" + MATH_VERSION + "\n"
    ).encode()
).hexdigest()
ROOT_KEYS = {
    "schema",
    "alignment_bytes",
    "source_identity_sha256",
    "weights_manifest_fingerprint",
    "math_version_sha256",
    "entries",
}
ENTRY_KEYS = {
    "projection",
    "format",
    "operand_math",
    "shape",
    "logical_bytes",
    "allocated_bytes",
    "file",
    "offset_bytes",
    "payload_sha256",
    "source",
}
SOURCE_KEYS = {
    "experts",
    "output_size",
    "input_size",
    "bits",
    "group_size",
    "weight_row_stride_bytes",
    "weight_expert_stride_bytes",
    "parameter_row_stride_bytes",
    "parameter_expert_stride_bytes",
}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_U64 = (1 << 64) - 1


class StoreError(ValueError):
    pass


def canonical(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def strict_json(raw: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StoreError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise StoreError("nonfinite JSON constant")

    def integer(token):
        if token.startswith("-"):
            raise StoreError("negative JSON integer syntax")
        return int(token)

    try:
        return json.loads(
            raw, object_pairs_hook=unique, parse_constant=constant, parse_int=integer
        )
    except (ValueError, UnicodeError, RecursionError) as error:
        raise StoreError("invalid JSON metadata") from error


def _object(value, keys: set[str], label: str):
    if type(value) is not dict or value.keys() != keys:
        raise StoreError(f"{label} keys do not match schema")
    return value


def _integer(value, label: str, *, positive=False, maximum=MAX_U64) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value > maximum:
        raise StoreError(f"{label} must be an in-range integer")
    return value


def _digest(value, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise StoreError(f"{label} must be lowercase SHA256")
    return value


def _file(root: Path, name: str) -> Path:
    if (
        type(name) is not str
        or len(name) > 128
        or ".." in name
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
    ):
        raise StoreError("file must be a flat ASCII basename")
    path = root / name
    try:
        info = path.lstat()
    except OSError as error:
        raise StoreError("store file is missing") from error
    if not stat.S_ISREG(info.st_mode) or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise StoreError("store file is not a contained regular file")
    return path


def _bounded(root: Path, name: str, maximum=MAX_JSON_BYTES) -> bytes:
    path = _file(root, name)
    if path.stat().st_size > maximum:
        raise StoreError("metadata exceeds bounded read")
    return path.read_bytes()


def _align(value: int) -> int:
    if value > MAX_U64 - (ALIGNMENT - 1):
        raise StoreError("allocation extent overflows uint64")
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def verify_store(
    directory: Path,
    *,
    source_identity=SOURCE_IDENTITY,
    weights_fingerprint=WEIGHTS_FINGERPRINT,
    verify_payloads=True,
):
    """Independent CPU verifier for the native operand-store contract.

    This checks stored bytes and schema, not whether arbitrary coefficients
    match an original checkpoint. The native cache validates every source view
    before mapping an operand, and export constructs it with the qualified GPU
    converter. Fixture source geometry is intentionally tiny.
    """
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise StoreError("store root must be a regular directory")
    _digest(source_identity, "expected source identity")
    _digest(weights_fingerprint, "expected weights fingerprint")
    raw = _bounded(root, "manifest.json")
    checksum = _bounded(root, "manifest.sha256", 65)
    if checksum != (hashlib.sha256(raw).hexdigest() + "\n").encode():
        raise StoreError("manifest byte checksum mismatch")
    manifest = _object(strict_json(raw), ROOT_KEYS, "manifest")
    if manifest["schema"] != SCHEMA:
        raise StoreError("unsupported store schema")
    if _integer(manifest["alignment_bytes"], "alignment") != ALIGNMENT:
        raise StoreError("unsupported store alignment")
    for field, expected in (
        ("source_identity_sha256", source_identity),
        ("weights_manifest_fingerprint", weights_fingerprint),
        ("math_version_sha256", MATH_DIGEST),
    ):
        if _digest(manifest[field], field) != expected:
            raise StoreError(f"{field} mismatch")
    entries = manifest["entries"]
    if type(entries) is not list or not entries or len(entries) > 8192:
        raise StoreError("store entries must be a bounded nonempty array")
    seen_files, seen_operands = set(), set()
    for item in entries:
        item = _object(item, ENTRY_KEYS, "entry")
        projection = item["projection"]
        if (
            type(projection) is not str
            or not projection
            or len(projection.encode("utf-8")) > 1024
            or "\0" in projection
            or "embed_tokens" in projection
            or "ngram_embedding" in projection
        ):
            raise StoreError("projection must be a bounded dense projection name")
        fmt = item["format"]
        if (
            type(fmt) is not str
            or fmt not in MATH_FORMATS
            or item["operand_math"] != MATH_FORMATS[fmt]
        ):
            raise StoreError("unsupported operand format/math")
        if (projection, fmt) in seen_operands:
            raise StoreError("duplicate operand")
        seen_operands.add((projection, fmt))
        shape = item["shape"]
        if type(shape) is not list or len(shape) != 2:
            raise StoreError("operand shape must be [N,K]")
        n, k = (
            _integer(v, "shape", positive=True, maximum=(1 << 32) - 1) for v in shape
        )
        source = _object(item["source"], SOURCE_KEYS, "source")
        values = {
            name: _integer(value, name, positive=True) for name, value in source.items()
        }
        if (
            values["experts"] != 1
            or values["output_size"] != n
            or values["input_size"] != k
        ):
            raise StoreError("source geometry differs from operand")
        bits, group = values["bits"], values["group_size"]
        if (
            n % 64
            or k > 32768
            or bits not in (4, 5, 6, 8)
            or group not in (32, 64, 128)
            or k % group
        ):
            raise StoreError("unsupported dense source geometry/quantization")
        packed_bytes, parameter_bytes = k * bits // 8, k // group * 2
        if (
            k * bits % 32
            or values["weight_row_stride_bytes"] < packed_bytes
            or values["weight_row_stride_bytes"] % 4
            or values["parameter_row_stride_bytes"] < parameter_bytes
            or values["parameter_row_stride_bytes"] % 2
            or values["weight_expert_stride_bytes"]
            < n * values["weight_row_stride_bytes"]
            or values["parameter_expert_stride_bytes"]
            < n * values["parameter_row_stride_bytes"]
        ):
            raise StoreError("source strides are short or misaligned")
        logical = n * k * (2 if fmt == "BF16" else 4)
        if (
            _integer(item["logical_bytes"], "logical bytes", positive=True) != logical
            or _integer(item["allocated_bytes"], "allocated bytes", positive=True)
            != _align(logical)
            or _integer(item["offset_bytes"], "offset bytes") != 0
        ):
            raise StoreError("operand byte extent/alignment mismatch")
        filename = item["file"]
        if type(filename) is not str or not filename.endswith(".bin"):
            raise StoreError("payload filename must end in .bin")
        path = _file(root, filename)
        if filename in seen_files:
            raise StoreError("duplicate payload file")
        seen_files.add(filename)
        expected_hash = _digest(item["payload_sha256"], "payload digest")
        if path.stat().st_size != item["allocated_bytes"]:
            raise StoreError("payload size mismatch")
        if verify_payloads:
            before = path.stat()
            digest, cursor = hashlib.sha256(), 0
            with path.open("rb") as stream:
                while block := stream.read(8 * 1024 * 1024):
                    digest.update(block)
                    if any(block[max(0, logical - cursor) :]):
                        raise StoreError("payload padding must be zero")
                    cursor += len(block)
            after = path.stat()
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or cursor != item["allocated_bytes"]:
                raise StoreError("payload changed while being verified")
            if digest.hexdigest() != expected_hash:
                raise StoreError("payload checksum mismatch")
    return manifest


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return struct.pack("<H", rounded & 0xFFFF)


def coefficient_payload(fmt: str, n=64, k=64) -> bytes:
    """Signed BF16 SF/bias, separate F32 multiply/add, then one final cast."""
    raw = bytearray()
    for row in range(n):
        for column in range(k):
            code = (row * 5 + column * 3) % 16
            if row < 3:
                # Both BF16 midpoint parities and their negative counterpart;
                # all source SF/bias values are exactly stored BF16 values.
                scale = -1.0 if row == 2 else 1.0
                bias = (-1 if row == 2 else (1 if row == 0 else 3)) / 256
            else:
                scale = (
                    (1 if (row + column // 32) % 2 else -1)
                    * (1 + (row % 4) / 128)
                    / 256
                )
                bias = (row % 5 - 2) / 8192
            value = _f32(_f32(code * scale) + bias)
            raw.extend(_bf16(value) if fmt == "BF16" else struct.pack("<f", value))
    return bytes(raw)


def write_manifest(root: Path, manifest: dict, *, raw: bytes | None = None):
    raw = canonical(manifest) if raw is None else raw
    (root / "manifest.json").write_bytes(raw)
    (root / "manifest.sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="ascii"
    )


def create_fixture(root: Path) -> dict:
    """Create 32 KiB of payloads suitable for the native CPU-only checker."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": SCHEMA,
        "alignment_bytes": ALIGNMENT,
        "source_identity_sha256": SOURCE_IDENTITY,
        "weights_manifest_fingerprint": WEIGHTS_FINGERPRINT,
        "math_version_sha256": MATH_DIGEST,
        "entries": [],
    }
    for fmt in MATH_FORMATS:
        raw = coefficient_payload(fmt)
        payload = raw + bytes(_align(len(raw)) - len(raw))
        filename = fmt.lower() + ".bin"
        (root / filename).write_bytes(payload)
        manifest["entries"].append(
            {
                "projection": "fixture.signed_affine",
                "format": fmt,
                "operand_math": MATH_FORMATS[fmt],
                "shape": [64, 64],
                "logical_bytes": len(raw),
                "allocated_bytes": len(payload),
                "file": filename,
                "offset_bytes": 0,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "source": {
                    "experts": 1,
                    "output_size": 64,
                    "input_size": 64,
                    "bits": 4,
                    "group_size": 32,
                    "weight_row_stride_bytes": 32,
                    "weight_expert_stride_bytes": 2048,
                    "parameter_row_stride_bytes": 4,
                    "parameter_expert_stride_bytes": 256,
                },
            }
        )
    write_manifest(root, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    if args.create:
        create_fixture(args.directory)
    manifest = verify_store(args.directory)
    print(
        json.dumps(
            {
                "valid": True,
                "entries": len(manifest["entries"]),
                "math_version_sha256": MATH_DIGEST,
            }
        )
    )


if __name__ == "__main__":
    main()
