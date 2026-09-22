#!/usr/bin/env python3
"""Stream a byte-preserving, aligned local Qwen4 MLX bundle. No MLX/GPU imports."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
from pathlib import Path
from typing import Callable

SCHEMA = "splash-local-qwen4-affine-v1"
ALIGNMENT = 16 * 1024
CHUNK_BYTES = 8 * 1024 * 1024
MAX_HEADER_BYTES = 8 * 1024 * 1024
MAX_SMALL_BYTES = 64 * 1024 * 1024
DTYPE_BYTES = {"U32": 4, "BF16": 2, "I64": 8}
SMALL_NAMES = (
    "config.json",
    "model.safetensors.index.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "README.md",
    "oq_imatrix_report.json",
)


class BundleError(RuntimeError):
    pass


def _json(raw: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BundleError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as error:
        raise BundleError("invalid JSON metadata") from error


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _chunk(chunk_bytes: int) -> int:
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 64 * 1024 * 1024:
        raise BundleError("chunk size must be between one byte and 64 MiB")
    return chunk_bytes


def _snapshot(path: Path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise BundleError("source entry is not a regular file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_exact(stream, count: int, chunk_bytes: int) -> bytes:
    result = bytearray()
    while count:
        block = stream.read(min(count, chunk_bytes))
        if not block:
            raise BundleError("source file truncated during import")
        result.extend(block)
        count -= len(block)
    return bytes(result)


def _file_digest(path: Path, chunk_bytes: int) -> tuple[int, str]:
    before = _snapshot(path)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        while block := source.read(chunk_bytes):
            count += len(block)
            digest.update(block)
    if _snapshot(path) != before or count != before[2]:
        raise BundleError("source file changed while being verified")
    return count, digest.hexdigest()


def _source_plan(source: Path, chunk_bytes: int):
    if not source.is_dir():
        raise BundleError("source model directory does not exist")
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        if not (source / name).is_file():
            raise BundleError("source model is missing required small metadata")
    small = []
    for name in SMALL_NAMES:
        path = source / name
        if path.exists():
            snapshot = _snapshot(path)
            if snapshot[2] > MAX_SMALL_BYTES:
                raise BundleError("small metadata file exceeds bounded copy size")
            small.append({"path": name, "snapshot": snapshot})
    config = _json((source / "config.json").read_bytes())
    index = _json((source / "model.safetensors.index.json").read_bytes())
    if not isinstance(config, dict) or not isinstance(index, dict):
        raise BundleError("source configuration or index is not an object")
    if config.get("model_type") != "qwen4_exp":
        raise BundleError("importer requires qwen4_exp model configuration")
    quantization = config.get("quantization", config.get("quantization_config"))
    if not isinstance(quantization, dict):
        raise BundleError("missing mixed quantization metadata")
    if (
        "quantization" in config
        and "quantization_config" in config
        and config["quantization"] != config["quantization_config"]
    ):
        raise BundleError("quantization metadata copies disagree")
    defaults = {name: quantization.get(name) for name in ("bits", "group_size", "mode")}
    if any(
        not isinstance(value, dict)
        for name, value in quantization.items()
        if name not in defaults
    ):
        raise BundleError("per-module quantization override is not an object")
    for item in (
        defaults,
        *(value for value in quantization.values() if isinstance(value, dict)),
    ):
        if (
            item.get("bits") not in (4, 5, 6, 8)
            or item.get("group_size") not in (32, 64, 128)
            or item.get("mode") != "affine"
        ):
            raise BundleError("unsupported affine quantization metadata")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise BundleError("source tensor index is missing")
    if any(
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".safetensors")
        for name in weight_map.values()
    ):
        raise BundleError("source shard path is not a local basename")
    shard_names = sorted(set(weight_map.values()))
    if shard_names != sorted(path.name for path in source.glob("*.safetensors")):
        raise BundleError("source shard index does not cover the checkpoint")
    tensors = {}
    plans = []
    for name in shard_names:
        path = source / name
        snapshot = _snapshot(path)
        with path.open("rb") as stream:
            raw_length = _read_exact(stream, 8, chunk_bytes)
            length = struct.unpack("<Q", raw_length)[0]
            if not 2 <= length <= MAX_HEADER_BYTES or length + 8 > snapshot[2]:
                raise BundleError("safetensors header length is invalid")
            header = _json(_read_exact(stream, length, chunk_bytes))
        if (
            not isinstance(header, dict)
            or not isinstance(header.get("__metadata__", {}), dict)
            or header.get("__metadata__", {}).get("format", "mlx") != "mlx"
        ):
            raise BundleError("safetensors shard is not MLX metadata")
        entries = []
        for tensor, item in header.items():
            if tensor == "__metadata__":
                continue
            if (
                not isinstance(tensor, str)
                or not tensor
                or "\0" in tensor
                or tensor in tensors
                or weight_map.get(tensor) != name
                or not isinstance(item, dict)
            ):
                raise BundleError("duplicate or unindexed tensor")
            dtype, shape, offsets = (
                item.get("dtype"),
                item.get("shape"),
                item.get("data_offsets"),
            )
            if (
                dtype not in DTYPE_BYTES
                or not isinstance(shape, list)
                or len(shape) > 8
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int or offset < 0 for offset in offsets)
            ):
                raise BundleError("tensor dtype, shape or offsets are invalid")
            size = math.prod(shape) * DTYPE_BYTES[dtype]
            begin, end = offsets
            if end - begin != size or end + length + 8 > snapshot[2]:
                raise BundleError("tensor byte count disagrees with dtype and shape")
            item = {
                "name": tensor,
                "dtype": dtype,
                "shape": shape,
                "length": size,
                "source_offset": 8 + length + begin,
            }
            entries.append(item)
            tensors[tensor] = item
        entries.sort(key=lambda item: (item["source_offset"], item["name"]))
        source_offset = length + 8
        target_offset = 0
        for item in entries:
            if item["source_offset"] != source_offset:
                raise BundleError("source tensor ranges overlap or leave gaps")
            item["offset"] = _align(target_offset)
            target_offset = item["offset"] + item["length"]
            source_offset += item["length"]
        if source_offset != snapshot[2] or not entries or _snapshot(path) != snapshot:
            raise BundleError("source shard has unconsumed or changing bytes")
        plans.append(
            {
                "source_path": name,
                "source_bytes": snapshot[2],
                "snapshot": snapshot,
                "header_bytes": length + 8,
                "path": "weights/" + Path(name).with_suffix(".bin").name,
                "bytes": _align(target_offset),
                "entries": entries,
            }
        )
    if set(tensors) != set(weight_map):
        raise BundleError("source tensor index contains missing tensors")
    for name, item in tensors.items():
        prefix = name.removesuffix(".weight")
        if name.endswith(".weight") and prefix + ".scales" in tensors:
            scales, biases = (
                tensors[prefix + ".scales"],
                tensors.get(prefix + ".biases"),
            )
            quant = quantization.get(prefix, defaults)
            if (
                not biases
                or item["dtype"] != "U32"
                or scales["dtype"] != "BF16"
                or biases["dtype"] != "BF16"
                or scales["shape"] != biases["shape"]
                or len(item["shape"]) not in (2, 3)
                or len(scales["shape"]) != len(item["shape"])
                or item["shape"][:-1] != scales["shape"][:-1]
                or item["shape"][-1] * 32
                != scales["shape"][-1] * quant["group_size"] * quant["bits"]
            ):
                raise BundleError(
                    "affine tensor triplet has inconsistent logical shape"
                )
    if any(_snapshot(source / item["path"]) != item["snapshot"] for item in small):
        raise BundleError("small source metadata changed while planning")
    return config, quantization, plans, small


def _pad(target, digest, count: int, chunk_bytes: int):
    zeros = bytes(min(ALIGNMENT, chunk_bytes))
    while count:
        block = zeros[: min(count, len(zeros))]
        target.write(block)
        digest.update(block)
        count -= len(block)


def _copy_shard(source_path: Path, target_path: Path, plan: dict, chunk_bytes: int):
    if _snapshot(source_path) != plan["snapshot"]:
        raise BundleError("source shard changed after planning")
    source_hash, target_hash = hashlib.sha256(), hashlib.sha256()
    with source_path.open("rb") as source, target_path.open("xb") as target:
        source_hash.update(_read_exact(source, plan["header_bytes"], chunk_bytes))
        for item in plan["entries"]:
            _pad(target, target_hash, item["offset"] - target.tell(), chunk_bytes)
            if source.tell() != item["source_offset"]:
                raise BundleError("source stream departed from tensor plan")
            remaining = item["length"]
            while remaining:
                block = source.read(min(remaining, chunk_bytes))
                if not block:
                    raise BundleError("source shard truncated during tensor copy")
                source_hash.update(block)
                target_hash.update(block)
                target.write(block)
                remaining -= len(block)
        _pad(target, target_hash, plan["bytes"] - target.tell(), chunk_bytes)
        if source.read(1) or source.tell() != plan["source_bytes"]:
            raise BundleError("source shard grew during copy")
        target.flush()
        os.fsync(target.fileno())
    if _snapshot(source_path) != plan["snapshot"]:
        raise BundleError("source shard changed while copying")
    return source_hash.hexdigest(), target_hash.hexdigest()


def _source_identity(shards, small_files):
    records = [
        {
            "path": item["source_path"],
            "bytes": item["source_bytes"],
            "sha256": item["source_sha256"],
        }
        for item in shards
    ]
    records += [
        {key: item[key] for key in ("path", "bytes", "sha256")} for item in small_files
    ]
    return hashlib.sha256(
        _canonical(
            {
                "schema": SCHEMA,
                "source_files": sorted(records, key=lambda item: item["path"]),
            }
        )
    ).hexdigest()


def _bundle_path(root: Path, relative: str):
    path = Path(relative)
    if (
        not isinstance(relative, str)
        or path.is_absolute()
        or not path.parts
        or any(part in (".", "..") for part in path.parts)
    ):
        raise BundleError("bundle contains unsafe relative path")
    candidate = root / path
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or not candidate.resolve().is_relative_to(root.resolve())
    ):
        raise BundleError("bundle entry is not a contained regular file")
    return candidate


def verify_bundle(
    bundle: Path, *, source: Path | None = None, chunk_bytes: int = CHUNK_BYTES
):
    chunk_bytes = _chunk(chunk_bytes)
    bundle = Path(bundle)
    if bundle.is_symlink() or not bundle.is_dir():
        raise BundleError("existing bundle is not a regular directory")
    raw = _bundle_path(bundle, "manifest.json").read_bytes()
    checksum = _bundle_path(bundle, "manifest.sha256").read_text("ascii").strip()
    if checksum != hashlib.sha256(raw).hexdigest():
        raise BundleError("bundle manifest checksum mismatch")
    manifest = _json(raw)
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("alignment") != ALIGNMENT
        or manifest.get("config") != "config.json"
        or manifest.get("index") != "model.safetensors.index.json"
    ):
        raise BundleError("bundle schema is unsupported")
    shards, small_files, tensors = (
        manifest.get("shards"),
        manifest.get("small_files"),
        manifest.get("tensors"),
    )
    if (
        not isinstance(shards, list)
        or not shards
        or not isinstance(small_files, list)
        or not isinstance(tensors, dict)
        or not tensors
    ):
        raise BundleError("bundle manifest is incomplete")
    if manifest.get("source_identity_sha256") != _source_identity(shards, small_files):
        raise BundleError("bundle source identity mismatch")
    shard_paths = {}
    for item in shards:
        path = item["path"]
        if path in shard_paths or item["bytes"] % ALIGNMENT:
            raise BundleError("bundle payload shard metadata is invalid")
        shard_paths[path] = item
        if _file_digest(_bundle_path(bundle, path), chunk_bytes) != (
            item["bytes"],
            item["sha256"],
        ):
            raise BundleError("bundle payload checksum mismatch")
    for item in small_files:
        if _file_digest(_bundle_path(bundle, item["path"]), chunk_bytes) != (
            item["bytes"],
            item["sha256"],
        ):
            raise BundleError("bundle small metadata checksum mismatch")
    by_shard = {path: [] for path in shard_paths}
    for name, item in tensors.items():
        if (
            item.get("shard") not in shard_paths
            or item.get("dtype") not in DTYPE_BYTES
            or not isinstance(item.get("shape"), list)
            or any(type(dim) is not int or dim <= 0 for dim in item["shape"])
            or type(item.get("offset")) is not int
            or item["offset"] < 0
            or item["offset"] % ALIGNMENT
            or item.get("length")
            != math.prod(item["shape"]) * DTYPE_BYTES[item["dtype"]]
        ):
            raise BundleError("bundle tensor descriptor is invalid")
        by_shard[item["shard"]].append(item)
    for path, items in by_shard.items():
        cursor = 0
        with _bundle_path(bundle, path).open("rb") as stream:
            for item in sorted(items, key=lambda value: value["offset"]):
                if (
                    item["offset"] != _align(cursor)
                    or item["offset"] + item["length"] > shard_paths[path]["bytes"]
                ):
                    raise BundleError("bundle tensors overlap or exceed their shard")
                remaining = item["offset"] - cursor
                while remaining:
                    block = stream.read(min(remaining, chunk_bytes))
                    if not block or any(block):
                        raise BundleError(
                            "bundle padding is not canonical zero storage"
                        )
                    remaining -= len(block)
                stream.seek(item["length"], os.SEEK_CUR)
                cursor = item["offset"] + item["length"]
            if _align(cursor) != shard_paths[path]["bytes"]:
                raise BundleError("bundle payload has unconsumed bytes")
            remaining = shard_paths[path]["bytes"] - cursor
            while remaining:
                block = stream.read(min(remaining, chunk_bytes))
                if not block or any(block):
                    raise BundleError(
                        "bundle tail padding is not canonical zero storage"
                    )
                remaining -= len(block)
    copied_config = _json(_bundle_path(bundle, "config.json").read_bytes())
    if manifest.get("quantization") != copied_config.get(
        "quantization", copied_config.get("quantization_config")
    ):
        raise BundleError("bundle mixed quantization metadata differs from config")
    if source is not None:
        _, quantization, plans, current_small = _source_plan(Path(source), chunk_bytes)
        if (
            quantization != manifest["quantization"]
            or len(plans) != len(shards)
            or {item["path"] for item in current_small}
            != {item["path"] for item in small_files}
        ):
            raise BundleError("existing bundle differs from current source metadata")
        expected_tensors = {}
        for plan in plans:
            actual = shard_paths.get(plan["path"])
            if (
                actual is None
                or actual["source_path"] != plan["source_path"]
                or _file_digest(Path(source) / plan["source_path"], chunk_bytes)
                != (actual["source_bytes"], actual["source_sha256"])
            ):
                raise BundleError("existing bundle source shard identity mismatch")
            for item in plan["entries"]:
                expected_tensors[item["name"]] = {
                    key: item[key]
                    for key in ("offset", "length", "dtype", "shape", "source_offset")
                }
                expected_tensors[item["name"]].update(
                    shard=plan["path"], source_shard=plan["source_path"]
                )
        if tensors != expected_tensors:
            raise BundleError("existing bundle tensor descriptors differ from source")
        for item in small_files:
            if _file_digest(Path(source) / item["path"], chunk_bytes) != (
                item["bytes"],
                item["sha256"],
            ):
                raise BundleError("existing bundle small source identity mismatch")
    return manifest


def _publish_exclusive(source: Path, destination: Path):
    # Darwin's exclusive rename never replaces even an empty race-created root.
    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameatx_np
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-2, os.fsencode(source), -2, os.fsencode(destination), 4):
        raise OSError(ctypes.get_errno(), "exclusive bundle publication failed")


def import_bundle(
    source: Path,
    destination: Path,
    *,
    alias: str | None = None,
    chunk_bytes: int = CHUNK_BYTES,
    progress: Callable[[str], None] | None = None,
):
    chunk_bytes = _chunk(chunk_bytes)
    source, destination = Path(source).resolve(), Path(destination).absolute()
    alias = alias or destination.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", alias) or alias in (
        ".",
        "..",
    ):
        raise BundleError("model alias is invalid")
    if destination.resolve() == source or destination.resolve().is_relative_to(source):
        raise BundleError(
            "derived bundle must be separate from the original checkpoint"
        )
    if destination.exists() or destination.is_symlink():
        return verify_bundle(destination, source=source, chunk_bytes=chunk_bytes)
    _, quantization, plans, small = _source_plan(source, chunk_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = sum(plan["bytes"] for plan in plans) + sum(
        item["snapshot"][2] for item in small
    )
    if shutil.disk_usage(destination.parent).free < required + MAX_SMALL_BYTES:
        raise BundleError("not enough free disk space for the complete derived bundle")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{alias}-import-", dir=destination.parent)
    )
    try:
        (temporary / "weights").mkdir()
        manifest = {
            "schema": SCHEMA,
            "alias": alias,
            "alignment": ALIGNMENT,
            "config": "config.json",
            "index": "model.safetensors.index.json",
            "quantization": quantization,
            "shards": [],
            "small_files": [],
            "tensors": {},
        }
        for number, plan in enumerate(plans, 1):
            if progress:
                progress(
                    f"Copying shard {number}/{len(plans)} ({plan['bytes']} aligned bytes)"
                )
            source_hash, target_hash = _copy_shard(
                source / plan["source_path"],
                temporary / plan["path"],
                plan,
                chunk_bytes,
            )
            manifest["shards"].append(
                {
                    key: plan[key]
                    for key in ("path", "bytes", "source_path", "source_bytes")
                }
                | {"source_sha256": source_hash, "sha256": target_hash}
            )
            for item in plan["entries"]:
                manifest["tensors"][item["name"]] = {
                    key: item[key]
                    for key in ("offset", "length", "dtype", "shape", "source_offset")
                }
                manifest["tensors"][item["name"]].update(
                    shard=plan["path"], source_shard=plan["source_path"]
                )
        for item in small:
            name = item["path"]
            if _snapshot(source / name) != item["snapshot"]:
                raise BundleError("small source metadata changed after planning")
            with (
                (source / name).open("rb") as original,
                (temporary / name).open("xb") as target,
            ):
                digest = hashlib.sha256()
                while block := original.read(chunk_bytes):
                    target.write(block)
                    digest.update(block)
                target.flush()
                os.fsync(target.fileno())
            if _snapshot(source / name) != item["snapshot"]:
                raise BundleError("small source metadata changed during copy")
            manifest["small_files"].append(
                {
                    "path": name,
                    "bytes": item["snapshot"][2],
                    "sha256": digest.hexdigest(),
                }
            )
        # A previously copied shard must not have changed while later shards
        # were streamed; the imported root is one coherent source snapshot.
        if any(
            _snapshot(source / plan["source_path"]) != plan["snapshot"]
            for plan in plans
        ) or any(
            _snapshot(source / item["path"]) != item["snapshot"] for item in small
        ):
            raise BundleError("source checkpoint changed before publication")
        manifest["source_identity_sha256"] = _source_identity(
            manifest["shards"], manifest["small_files"]
        )
        raw = _canonical(manifest) + b"\n"
        for name, contents in (
            ("manifest.json", raw),
            (
                "manifest.sha256",
                (hashlib.sha256(raw).hexdigest() + "\n").encode("ascii"),
            ),
        ):
            with (temporary / name).open("xb") as target:
                target.write(contents)
                target.flush()
                os.fsync(target.fileno())
        for path in (temporary / "weights", temporary):
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        # Read the complete derived payloads back before making the root
        # visible. Their digests cover actual stored bytes, including padding.
        verify_bundle(temporary, chunk_bytes=chunk_bytes)
        _publish_exclusive(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--alias")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "install/local-models",
    )
    parser.add_argument("--chunk-bytes", type=int, default=CHUNK_BYTES)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    alias = arguments.alias or arguments.source.name
    destination = arguments.output_root / alias
    try:
        manifest = (
            verify_bundle(
                destination, source=arguments.source, chunk_bytes=arguments.chunk_bytes
            )
            if arguments.verify_only
            else import_bundle(
                arguments.source,
                destination,
                alias=alias,
                chunk_bytes=arguments.chunk_bytes,
                progress=lambda text: print(text, flush=True),
            )
        )
    except (BundleError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Import failed: {error}\n")
    print(
        json.dumps(
            {
                "bundle": str(destination),
                "schema": SCHEMA,
                "shards": len(manifest["shards"]),
                "tensors": len(manifest["tensors"]),
                "aligned_payload_bytes": sum(
                    item["bytes"] for item in manifest["shards"]
                ),
                "source_identity_sha256": manifest["source_identity_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
