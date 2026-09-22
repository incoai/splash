"""Copy a smaller exact INT8 expert inventory from a verified readonly store.

This never reads the original checkpoint, reconstructs coefficients, or changes
integer codes/scales. The native consumer remains responsible for full payload
checksum verification before GPU use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path

import flash_int8_expert_store_convert as store

INVENTORIES = (32, 64, 128)


def selected_inventory(record):
    values = store.validate_plan(record, store.EXPECTED_SOURCE_IDENTITY)
    count = record["requested_limit"]
    if count not in INVENTORIES or any(len(ids) != count for ids in values):
        raise store.StoreError("selected inventory must be uniformly32/64/128")
    return values


def subset_ranks(source_ids, requested_ids):
    if source_ids != sorted(set(source_ids)) or requested_ids != sorted(
        set(requested_ids)
    ):
        raise store.StoreError("source and subset IDs must be sorted and unique")
    if any(
        type(value) is not int or not 0 <= value < 512
        for value in source_ids + requested_ids
    ):
        raise store.StoreError("expert IDs must be integers below512")
    ranks = {expert: rank for rank, expert in enumerate(source_ids)}
    if any(expert not in ranks for expert in requested_ids):
        raise store.StoreError("requested inventory is not a subset of stored IDs")
    return [ranks[expert] for expert in requested_ids]


def copy_plane(source_fd, output, *, source_offset, expert_bytes, ranks, file_digest):
    digest = hashlib.sha256()
    for rank in ranks:
        value = os.pread(source_fd, expert_bytes, source_offset + rank * expert_bytes)
        if len(value) != expert_bytes:
            raise store.StoreError(
                "stored source plane was truncated during subset copy"
            )
        if output.write(value) != len(value):
            raise store.StoreError("short subset plane write")
        digest.update(value)
        file_digest.update(value)
    return digest.hexdigest()


def subset(source, plan_path, output):
    source, plan_path, output = (
        Path(source).resolve(),
        Path(plan_path).resolve(),
        Path(output).resolve(),
    )
    if (
        output.exists()
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise store.StoreError(
            "subset destination must be fresh and separate from source"
        )
    manifest_path = source / "manifest.json"
    raw_source, raw_plan = manifest_path.read_bytes(), plan_path.read_bytes()
    source_record, plan = store._json(raw_source), store._json(raw_plan)
    if (
        set(source_record) != store.MANIFEST_KEYS
        or source_record["schema"] != store.SCHEMA
        or source_record["source_identity_sha256"] != store.EXPECTED_SOURCE_IDENTITY
        or source_record["coefficient_policy"] != store.COEFFICIENT_POLICY
        or source_record["quantization_format"] != store.QUANTIZATION_FORMAT
        or source_record["integer_rounding"] != store.INTEGER_ROUNDING
        or source_record["alignment"] != store.ALIGNMENT
        or source_record["target_layers"] != 48
    ):
        raise store.StoreError("source sidecar contract is incompatible")
    selected = selected_inventory(plan)
    old_ids = source_record["selected_experts"]
    if (
        len(old_ids) != 48
        or len(source_record["layers"]) != 48
        or len(old_ids[0]) not in INVENTORIES
        or any(len(ids) != len(old_ids[0]) for ids in old_ids)
    ):
        raise store.StoreError(
            "source inventory must be uniformly32/64/128 across48 layers"
        )
    ranks = [
        subset_ranks(previous, chosen) for previous, chosen in zip(old_ids, selected)
    ]
    layers = [store.layout_layer(index, len(ids)) for index, ids in enumerate(selected)]
    snapshots = {
        manifest_path: store._snapshot(manifest_path),
        plan_path: store._snapshot(plan_path),
    }
    for index, entry in enumerate(source_record["layers"]):
        expected = store.layout_layer(index, len(old_ids[index]))
        if (
            entry["layer_index"] != index
            or entry["path"] != expected["path"]
            or entry["bytes"] != expected["bytes"]
        ):
            raise store.StoreError("source layer layout mismatch")
        for projection in store.PROJECTIONS:
            a, b = entry["projections"][projection], expected["projections"][projection]
            if (
                a["source_prefix"] != b["source_prefix"]
                or a["dimensions"] != b["dimensions"]
            ):
                raise store.StoreError("source projection binding/geometry mismatch")
            for name in ("codes", "scales"):
                if any(
                    a[name][key] != b[name][key]
                    for key in ("dtype", "shape", "offset", "length")
                ) or not store._valid_sha(a[name]["sha256"]):
                    raise store.StoreError("source plane extent/format/hash mismatch")
        path = source / entry["path"]
        details = path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_mode & 0o222
            or details.st_size != entry["bytes"]
        ):
            raise store.StoreError(
                "source payload must be readonly regular and exact-size"
            )
        snapshots[path] = store._snapshot(path)
    total = sum(layer["bytes"] for layer in layers)
    if (
        not output.parent.is_dir()
        or shutil.disk_usage(output.parent).free < total + 64 * 1024 * 1024
    ):
        raise store.StoreError(
            "subset destination parent/free disk space is insufficient"
        )
    record = {key: source_record[key] for key in store.MANIFEST_KEYS}
    record.update(
        plan_sha256=store._sha(raw_plan),
        selected_experts=selected,
        layers=layers,
        total_bytes=total,
        planned_allocation_bytes=total + 48 * store.ALIGNMENT,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    started = time.monotonic()
    try:
        for index, layer in enumerate(layers):
            previous = source_record["layers"][index]
            source_fd = os.open(
                source / previous["path"], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            file_digest = hashlib.sha256()
            try:
                path = temporary / layer["path"]
                with path.open("wb", buffering=8 * 1024 * 1024) as stream:
                    for name, (n, k) in store.PROJECTIONS.items():
                        for plane_name, expert_bytes in (
                            ("codes", n * k),
                            ("scales", n * 4),
                        ):
                            plane = layer["projections"][name][plane_name]
                            padding = bytes(plane["offset"] - stream.tell())
                            stream.write(padding)
                            file_digest.update(padding)
                            plane["sha256"] = copy_plane(
                                source_fd,
                                stream,
                                source_offset=previous["projections"][name][plane_name][
                                    "offset"
                                ],
                                expert_bytes=expert_bytes,
                                ranks=ranks[index],
                                file_digest=file_digest,
                            )
                    padding = bytes(layer["bytes"] - stream.tell())
                    stream.write(padding)
                    file_digest.update(padding)
                    stream.flush()
                    os.fsync(stream.fileno())
                path.chmod(0o444)
                layer["sha256"] = file_digest.hexdigest()
            finally:
                os.close(source_fd)
        if any(store._snapshot(path) != before for path, before in snapshots.items()):
            raise store.StoreError("source sidecar/plan changed during subset copy")
        raw_manifest = (
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        with (temporary / "manifest.json").open("wb") as stream:
            stream.write(raw_manifest)
            stream.flush()
            os.fsync(stream.fileno())
        (temporary / "manifest.json").chmod(0o444)
        report = {
            "source_store": str(source),
            "source_store_manifest_sha256": store._sha(raw_source),
            "source_payload_full_rehash": False,
            "source_and_plan_snapshots_unchanged": True,
            "copy_policy": "exact selected signed-I8 codes and F32 scales; no requantization or original checkpoint read",
            "selected_per_layer": len(selected[0]),
            "total_bytes": total,
            "elapsed_seconds": time.monotonic() - started,
        }
        (temporary / "subset-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            raise store.StoreError("subset destination appeared during copy")
        temporary.rename(output)
        return {
            **report,
            "manifest_sha256": store._sha(raw_manifest),
            "output": str(output),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(subset(args.source, args.plan, args.output), sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
