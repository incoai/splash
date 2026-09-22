"""Stream a selected Flash INT8 expert sidecar without modifying source weights.

The store is separate from the aligned model package and preserves its source
identity. Eight experts are converted at a time; there is no GPU, inference,
activation calibration, BF16 coefficient duplication, or production selection.
The native reader must verify the strict metadata and all file/plane hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "_flash_expert_int8_pilot", Path(__file__).with_name("flash_expert_int8_convert.py")
)
_PILOT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PILOT)

SCHEMA = "splash-flash-int8-expert-store-v1"
SOURCE_SCHEMA = "splash-local-qwen4-affine-v1"
PLAN_SCHEMA = "splash-flash-hot-expert-plan-v1"
EXPECTED_SOURCE_IDENTITY = (
    "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"
)
COEFFICIENT_POLICY = "source_q4_g64_f32_separate_multiply_add_then_bf16_rne_v1"
QUANTIZATION_FORMAT = "signed-symmetric-int8-rowwise-f32-scale"
INTEGER_ROUNDING = "F32 absmax/127; F32 division; nearest-even integer; clamp [-127,127]; zero row scale=1"
ALIGNMENT = 16384
TARGET_LAYERS = 48
BATCH_EXPERTS = 8
MIN_FREE_BYTES = 31_000_000_000
PROJECTIONS = {
    "gate_proj": (640, 2560),
    "up_proj": (640, 2560),
    "down_proj": (2560, 640),
}
MANIFEST_KEYS = frozenset(
    {
        "schema",
        "source_identity_sha256",
        "source_manifest_sha256",
        "plan_sha256",
        "alignment",
        "target_layers",
        "selected_experts",
        "coefficient_policy",
        "quantization_format",
        "integer_rounding",
        "layers",
        "total_bytes",
        "planned_allocation_bytes",
    }
)


class StoreError(ValueError):
    pass


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StoreError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite(value):
        raise StoreError(f"non-finite JSON token: {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=finite)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _valid_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _align(size):
    if type(size) is not int or size < 0:
        raise StoreError("alignment requires a nonnegative integer byte count")
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def validate_plan(plan, identity):
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("source_identity") != identity
    ):
        raise StoreError("hot-expert plan schema/source identity does not match")
    if (
        type(plan.get("requested_limit")) is not int
        or not 1 <= plan["requested_limit"] <= 128
    ):
        raise StoreError("hot-expert plan limit must be an integer in [1,128]")
    selected = plan.get("selected_experts")
    if not isinstance(selected, list) or len(selected) != TARGET_LAYERS:
        raise StoreError("hot-expert plan must select exactly 48 target layers")
    for layer, ids in enumerate(selected):
        if not isinstance(ids, list) or not 1 <= len(ids) <= plan["requested_limit"]:
            raise StoreError(
                f"layer {layer}: selected expert count must be in [1,limit]"
            )
        if any(
            type(expert) is not int or not 0 <= expert < 512 for expert in ids
        ) or ids != sorted(set(ids)):
            raise StoreError(
                f"layer {layer}: expert IDs must be sorted unique integers in [0,511]"
            )
    return [list(ids) for ids in selected]


def layout_layer(layer, selected_count, *, projections=PROJECTIONS):
    """Plan six independently aligned planes and a final aligned file size."""
    if (
        type(layer) is not int
        or not 0 <= layer < TARGET_LAYERS
        or type(selected_count) is not int
        or not 1 <= selected_count <= 128
    ):
        raise StoreError(
            "layer/count must be integer target layer and selected count in [1,128]"
        )
    offset, entries = 0, {}
    for name, dimensions in projections.items():
        if (
            not isinstance(name, str)
            or len(dimensions) != 2
            or any(type(value) is not int or value <= 0 for value in dimensions)
        ):
            raise StoreError(
                "projection geometry must contain positive integer N/K dimensions"
            )
        n, k = dimensions
        planes = {}
        for plane, dtype, shape, size in (
            ("codes", "I8", [selected_count, n, k], selected_count * n * k),
            ("scales", "F32", [selected_count, n], selected_count * n * 4),
        ):
            offset = _align(offset)
            planes[plane] = {
                "dtype": dtype,
                "shape": shape,
                "offset": offset,
                "length": size,
                "sha256": None,
            }
            offset += size
        entries[name] = {
            "source_prefix": f"language_model.model.layers.{layer}.mlp.switch_mlp.{name}",
            "dimensions": [selected_count, n, k],
            **planes,
        }
    return {
        "layer_index": layer,
        "path": f"layer-{layer:02d}.bin",
        "bytes": _align(offset),
        "sha256": None,
        "projections": entries,
    }


def _source_file(package, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise StoreError("source shard path must be a nonempty relative path")
    path = (package / relative).resolve()
    if not path.is_relative_to(package) or not path.is_file():
        raise StoreError("source shard path escapes the package or is missing")
    return path


def validate_tensor(package, descriptor, shape, dtype, shard_records, snapshots):
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("dtype") != dtype
        or descriptor.get("shape") != list(shape)
    ):
        raise StoreError("source expert tensor geometry/dtype mismatch")
    item_bytes = 4 if dtype == "U32" else 2
    length, offset = math.prod(shape) * item_bytes, descriptor.get("offset")
    if (
        type(offset) is not int
        or offset < 0
        or offset % ALIGNMENT
        or type(descriptor.get("length")) is not int
        or descriptor["length"] != length
    ):
        raise StoreError("source expert tensor length/alignment mismatch")
    relative = descriptor.get("shard")
    path = _source_file(package, relative)
    shard = shard_records.get(relative)
    if (
        not isinstance(shard, dict)
        or type(shard.get("bytes")) is not int
        or shard["bytes"] <= 0
        or not _valid_sha(shard.get("sha256"))
    ):
        raise StoreError("source tensor must reference a hashed manifest shard")
    snapshot = snapshots.setdefault(path, _snapshot(path))
    if (
        _snapshot(path) != snapshot
        or shard["bytes"] != snapshot[2]
        or offset + length > snapshot[2]
    ):
        raise StoreError("source shard changed, mismatches manifest, or is truncated")
    return {
        "path": path,
        "shape": tuple(shape),
        "dtype": "<u4" if dtype == "U32" else "<u2",
        "offset": offset,
    }


def prepare_source(package, source, selected, snapshots):
    if (
        not isinstance(source, dict)
        or source.get("schema") != SOURCE_SCHEMA
        or source.get("source_identity_sha256") != EXPECTED_SOURCE_IDENTITY
    ):
        raise StoreError(
            "aligned source schema/identity is incompatible with this Flash sidecar"
        )
    quantization, tensors = source.get("quantization"), source.get("tensors")
    shards = source.get("shards")
    if (
        not isinstance(quantization, dict)
        or not isinstance(tensors, dict)
        or not isinstance(shards, list)
    ):
        raise StoreError("source manifest must contain quantization/tensors/shards")
    records = {}
    for shard in shards:
        if (
            not isinstance(shard, dict)
            or not isinstance(shard.get("path"), str)
            or shard["path"] in records
        ):
            raise StoreError("source manifest shard paths must be unique")
        records[shard["path"]] = shard
    prepared = []
    for layer, ids in enumerate(selected):
        projections = {}
        for projection, (n, k) in PROJECTIONS.items():
            prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp.{projection}"
            q = quantization.get(
                prefix,
                {
                    "bits": quantization.get("bits"),
                    "group_size": quantization.get("group_size"),
                    "mode": "affine",
                },
            )
            if (
                not isinstance(q, dict)
                or type(q.get("bits")) is not int
                or q["bits"] != 4
                or type(q.get("group_size")) is not int
                or q["group_size"] != 64
                or q.get("mode") != "affine"
            ):
                raise StoreError(
                    f"{prefix}: selected source experts must be affine Q4 G64"
                )
            projections[projection] = {
                suffix: validate_tensor(
                    package,
                    tensors.get(f"{prefix}.{suffix}"),
                    shape,
                    dtype,
                    records,
                    snapshots,
                )
                for suffix, shape, dtype in (
                    ("weight", (512, n, k // 8), "U32"),
                    ("scales", (512, n, k // 64), "BF16"),
                    ("biases", (512, n, k // 64), "BF16"),
                )
            }
        prepared.append(projections)
    return prepared, records


def check_destination(
    package, output, plan_path, required_bytes, *, disk_usage=shutil.disk_usage
):
    if (
        output.exists()
        or output.is_relative_to(package)
        or package.is_relative_to(output)
        or plan_path.is_relative_to(output)
    ):
        raise StoreError("output must be a new directory separate from source and plan")
    if not output.parent.is_dir():
        raise StoreError("output parent directory must already exist")
    minimum = max(MIN_FREE_BYTES, required_bytes + 64 * 1024 * 1024)
    if disk_usage(output.parent).free < minimum:
        raise StoreError(f"insufficient disk space; need at least {minimum} free bytes")


def write_projection(
    stream, layout, source_planes, experts, *, batch_size=BATCH_EXPERTS
):
    """Write compact selected expert codes/scales; no entire plane expansion."""
    if type(batch_size) is not int or not 1 <= batch_size <= BATCH_EXPERTS:
        raise StoreError("expert batch size must be an integer in [1,8]")
    h, n, k = layout["dimensions"]
    if len(experts) != h:
        raise StoreError("selected expert count differs from plane geometry")
    mapped = {
        name: np.memmap(
            item["path"],
            dtype=item["dtype"],
            mode="r",
            offset=item["offset"],
            shape=item["shape"],
        )
        for name, item in source_planes.items()
    }
    code_digest, scale_digest = hashlib.sha256(), hashlib.sha256()
    max_batch = 0
    try:
        for begin in range(0, h, batch_size):
            chosen = experts[begin : begin + batch_size]
            selected = {
                name: np.array(value[chosen], copy=True)
                for name, value in mapped.items()
            }
            reference = _PILOT.reconstruct_q4_bf16(
                selected["weight"],
                selected["scales"],
                selected["biases"],
                group_size=64,
            )
            codes, scales = _PILOT.symmetric_int8(reference)
            raw_codes = np.ascontiguousarray(codes).tobytes()
            raw_scales = np.ascontiguousarray(
                scales.reshape(len(chosen), n).astype("<f4")
            ).tobytes()
            stream.seek(layout["codes"]["offset"] + begin * n * k)
            if stream.write(raw_codes) != len(raw_codes):
                raise StoreError("short INT8 code-plane write")
            stream.seek(layout["scales"]["offset"] + begin * n * 4)
            if stream.write(raw_scales) != len(raw_scales):
                raise StoreError("short F32 scale-plane write")
            code_digest.update(raw_codes)
            scale_digest.update(raw_scales)
            max_batch = max(max_batch, len(chosen))
            del selected, reference, codes, scales, raw_codes, raw_scales
    finally:
        del mapped
    layout["codes"]["sha256"] = code_digest.hexdigest()
    layout["scales"]["sha256"] = scale_digest.hexdigest()
    return max_batch


def convert(package, plan_path, output, *, emit=print):
    package, plan_path, output = (
        Path(package).resolve(),
        Path(plan_path).resolve(),
        Path(output).resolve(),
    )
    # Refuse an existing destination before reading or allocating source views.
    if output.exists():
        raise StoreError("output already exists; a fresh sidecar path is required")
    manifest_path = package / "manifest.json"
    snapshots = {
        manifest_path: _snapshot(manifest_path),
        plan_path: _snapshot(plan_path),
    }
    raw_source, raw_plan = manifest_path.read_bytes(), plan_path.read_bytes()
    source, plan = _json(raw_source), _json(raw_plan)
    if (
        not isinstance(source, dict)
        or source.get("source_identity_sha256") != EXPECTED_SOURCE_IDENTITY
    ):
        raise StoreError("source identity must match the installed Flash checkpoint")
    selected = validate_plan(plan, EXPECTED_SOURCE_IDENTITY)
    layers = [layout_layer(layer, len(ids)) for layer, ids in enumerate(selected)]
    total_bytes = sum(layer["bytes"] for layer in layers)
    check_destination(package, output, plan_path, total_bytes)
    prepared, source_records = prepare_source(package, source, selected, snapshots)
    record = {
        "schema": SCHEMA,
        "source_identity_sha256": EXPECTED_SOURCE_IDENTITY,
        "source_manifest_sha256": _sha(raw_source),
        "plan_sha256": _sha(raw_plan),
        "alignment": ALIGNMENT,
        "target_layers": TARGET_LAYERS,
        "selected_experts": selected,
        "coefficient_policy": COEFFICIENT_POLICY,
        "quantization_format": QUANTIZATION_FORMAT,
        "integer_rounding": INTEGER_ROUNDING,
        "layers": layers,
        "total_bytes": total_bytes,
        "planned_allocation_bytes": total_bytes + TARGET_LAYERS * ALIGNMENT,
    }
    assert set(record) == MANIFEST_KEYS
    source_paths = sorted(
        {
            item["path"]
            for layer in prepared
            for planes in layer.values()
            for item in planes.values()
        }
    )
    started = time.monotonic()
    emit(
        json.dumps(
            {
                "phase": "preflight",
                "layers": TARGET_LAYERS,
                "selected_experts": sum(map(len, selected)),
                "total_bytes": total_bytes,
                "planned_allocation_bytes": record["planned_allocation_bytes"],
                "source_shards_to_verify": len(source_paths),
            }
        )
    )
    for index, path in enumerate(source_paths):
        relative = path.relative_to(package).as_posix()
        if (
            _sha_file(path) != source_records[relative]["sha256"]
            or _snapshot(path) != snapshots[path]
        ):
            raise StoreError(f"source shard SHA/snapshot mismatch: {relative}")
        emit(
            json.dumps(
                {
                    "phase": "source_sha_verified",
                    "shard": index + 1,
                    "of": len(source_paths),
                    "path": relative,
                }
            )
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for layer in layers:
            index = layer["layer_index"]
            path = temporary / layer["path"]
            with path.open("w+b") as stream:
                stream.truncate(layer["bytes"])
                for projection in PROJECTIONS:
                    write_projection(
                        stream,
                        layer["projections"][projection],
                        prepared[index][projection],
                        selected[index],
                    )
                stream.flush()
                os.fsync(stream.fileno())
            path.chmod(0o444)
            if path.stat().st_size != layer["bytes"]:
                raise StoreError(
                    "derived layer file size differs from planned aligned size"
                )
            layer["sha256"] = _sha_file(path)
            emit(
                json.dumps(
                    {
                        "phase": "layer_stored",
                        "layer": index,
                        "selected": len(selected[index]),
                        "bytes": layer["bytes"],
                        "sha256": layer["sha256"],
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                )
            )
        for path, snapshot in snapshots.items():
            if _snapshot(path) != snapshot:
                raise StoreError(
                    f"source/plan snapshot changed during conversion: {path}"
                )
        if (
            _sha(manifest_path.read_bytes()) != record["source_manifest_sha256"]
            or _sha(plan_path.read_bytes()) != record["plan_sha256"]
        ):
            raise StoreError("source manifest or plan SHA changed during conversion")
        with (temporary / "manifest.json").open("w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        (temporary / "manifest.json").chmod(0o444)
        # The strict native manifest remains free of converter-only fields.
        report = {
            "scope": "CPU-only selected expert sidecar; GPU/model correctness and performance unqualified",
            "source_shards_verified_against_original_manifest_sha": len(source_paths),
            "source_plan_and_shard_snapshots_unchanged": True,
            "source_manifest_and_plan_sha_reverified_at_end": True,
            "expert_batch_size": BATCH_EXPERTS,
            "max_expanded_bf16_coefficients_bytes_per_batch": BATCH_EXPERTS
            * max(n * k for n, k in PROJECTIONS.values())
            * 2,
            "source": str(package),
            "plan": str(plan_path),
            "elapsed_seconds": time.monotonic() - started,
            "stored_coefficient_and_scale_bytes": total_bytes,
            "rank_map_planned_bytes": TARGET_LAYERS * ALIGNMENT,
            "source_readonly_snapshots": {
                str(path): list(snapshot) for path, snapshot in snapshots.items()
            },
        }
        (temporary / "conversion-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            raise StoreError(
                "destination appeared during conversion; refusing to replace it"
            )
        temporary.rename(output)
        emit(
            json.dumps(
                {
                    "phase": "complete",
                    "output": str(output),
                    "total_bytes": total_bytes,
                    "manifest_sha256": _sha_file(output / "manifest.json"),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
        )
        return record
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def inspect_plan(package, plan_path):
    source_path = Path(package) / "manifest.json"
    raw_source, raw_plan = source_path.read_bytes(), Path(plan_path).read_bytes()
    source, plan = _json(raw_source), _json(raw_plan)
    if (
        source.get("schema") != SOURCE_SCHEMA
        or source.get("source_identity_sha256") != EXPECTED_SOURCE_IDENTITY
    ):
        raise StoreError("source package schema/identity mismatch")
    selected = validate_plan(plan, EXPECTED_SOURCE_IDENTITY)
    total = sum(
        layout_layer(layer, len(ids))["bytes"] for layer, ids in enumerate(selected)
    )
    return {
        "source_identity_sha256": EXPECTED_SOURCE_IDENTITY,
        "source_manifest_sha256": _sha(raw_source),
        "plan_sha256": _sha(raw_plan),
        "target_layers": TARGET_LAYERS,
        "counts": list(map(len, selected)),
        "total_bytes": total,
        "planned_allocation_bytes": total + TARGET_LAYERS * ALIGNMENT,
        "minimum_free_bytes": max(MIN_FREE_BYTES, total + 64 * 1024 * 1024),
        "batch_experts": BATCH_EXPERTS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("install/local-models/Flash-Next-oQ4e-mtp-v1"),
    )
    parser.add_argument(
        "--plan", type=Path, default=Path("build/release/flash/hot-expert-plan128.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("install/local-models/Flash-Next-int8-experts-top128-v1"),
    )
    parser.add_argument("--inspect-plan", action="store_true")
    args = parser.parse_args()
    try:
        if args.inspect_plan:
            print(
                json.dumps(
                    inspect_plan(args.package, args.plan), indent=2, sort_keys=True
                )
            )
        else:
            convert(
                args.package,
                args.plan,
                args.output,
                emit=lambda line: print(line, flush=True),
            )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"expert store conversion failed: {error}\n")


if __name__ == "__main__":
    main()
