"""Save exact BF16 operands for original Flash Q4/G64 expert projections.

This CPU-only candidate stores each group's 16 possible coefficients, retaining
the original packed Q4 codes. Two explicit float32 operations reproduce the
contract-off Metal reconstruction; the final integer operation rounds once to
BF16 with ties to even. No checkpoint is modified and no GPU library is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = "splash-flash-exact-expert-bf16-lut-v1"
LAYOUT = "E,Nblock64,Kgroup64,Nlane64,Q16"
POLICY = "contract-off-F32-multiply-then-F32-add-once-rounded-BF16-RNE-v1"
ALIGNMENT = 16384
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def bf16_lut(scales_u16, biases_u16):
    """Return uint16 [...,16] exact operands, rejecting nonfinite coefficients."""
    scales_u16 = np.asarray(scales_u16)
    biases_u16 = np.asarray(biases_u16)
    if scales_u16.dtype != np.dtype("uint16") or biases_u16.dtype != np.dtype("uint16"):
        raise ValueError("source parameters must be uint16 BF16 storage")
    if scales_u16.shape != biases_u16.shape or not scales_u16.size:
        raise ValueError("source scale/bias extents must agree and be nonempty")
    if np.any((scales_u16 & 0x7F80) == 0x7F80) or np.any(
        (biases_u16 & 0x7F80) == 0x7F80
    ):
        raise ValueError("source parameters contain NaN or infinity")
    scales = (scales_u16.astype(np.uint32) << 16).view(np.float32)
    biases = (biases_u16.astype(np.uint32) << 16).view(np.float32)
    # Separate ufunc calls prevent FMA contraction or a float64 intermediate.
    with np.errstate(over="ignore", invalid="ignore"):
        multiplied = np.multiply(
            scales[..., None], np.arange(16, dtype=np.float32), dtype=np.float32
        )
        reconstructed = np.add(multiplied, biases[..., None], dtype=np.float32)
    if not np.all(np.isfinite(reconstructed)):
        raise ValueError("FP32 reconstruction contains NaN or infinity")
    words = reconstructed.view(np.uint32)
    operands = ((words + np.uint32(0x7FFF) + ((words >> 16) & 1)) >> 16).astype(
        np.uint16
    )
    if np.any((operands & 0x7F80) == 0x7F80):
        raise ValueError("BF16 coefficient rounding overflows to infinity")
    return operands


def tiled_lut(array):
    array = np.asarray(array)
    if array.dtype != np.dtype("uint16") or array.ndim != 4 or array.shape[-1] != 16:
        raise ValueError("LUT must be uint16[E,N,Kgroup,16]")
    experts, n, groups, _ = array.shape
    if not experts or not n or n % 64 or not groups:
        raise ValueError("LUT dimensions must be positive and N divisible by 64")
    return np.ascontiguousarray(
        array.reshape(experts, n // 64, 64, groups, 16).transpose(0, 1, 3, 2, 4)
    )


def lut_offset_bytes(expert, n, k, code, output_size, input_size):
    for value, name in ((output_size, "output size"), (input_size, "input size")):
        _positive_integer(value, name)
        if value % 64:
            raise ValueError(f"{name} must be divisible by 64")
    for value, limit, name in (
        (expert, 512, "expert"),
        (n, output_size, "n"),
        (k, input_size, "k"),
        (code, 16, "code"),
    ):
        if type(value) is not int or not 0 <= value < limit:
            raise ValueError(f"{name} is outside its logical extent")
    return (
        (
            (
                (
                    (expert * (output_size // 64) + n // 64) * (input_size // 64)
                    + k // 64
                )
                * 64
                + n % 64
            )
            * 16
        )
        + code
    ) * 2


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _safe_source(package, tensor):
    shard = tensor.get("shard")
    if not isinstance(shard, str):
        raise ValueError("source shard path is missing")
    path = (package / shard).resolve()
    if not path.is_relative_to(package):
        raise ValueError("source shard escapes package")
    offset = tensor.get("offset")
    length = tensor.get("length")
    if (
        type(offset) is not int
        or type(length) is not int
        or offset < 0
        or length <= 0
        or offset % ALIGNMENT
    ):
        raise ValueError("source tensor has an invalid aligned byte extent")
    if offset + length > path.stat().st_size:
        raise ValueError("source tensor exceeds its shard")
    return path


def convert_layer(package: Path, output: Path, layer: int, *, chunk_experts=8):
    package = package.resolve()
    output = output.resolve()
    if type(layer) is not int or not 0 <= layer < 48:
        raise ValueError("layer must be an integer in [0,47]")
    if type(chunk_experts) is not int or not 1 <= chunk_experts <= 64:
        raise ValueError("chunk experts must be an integer in [1,64]")
    if output.exists() or output.is_relative_to(package):
        raise ValueError("output must be new and outside the immutable source package")
    manifest_path = package / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    recorded_manifest_sha = (package / "manifest.sha256").read_text().split()[0]
    if manifest_sha != recorded_manifest_sha:
        raise ValueError("source manifest hash differs")
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("schema") != "splash-local-qwen4-affine-v1"
        or manifest.get("alignment") != ALIGNMENT
    ):
        raise ValueError("unsupported immutable source package")
    source_identity = manifest.get("source_identity_sha256")
    if (
        not isinstance(source_identity, str)
        or len(source_identity) != 64
        or any(c not in "0123456789abcdef" for c in source_identity)
    ):
        raise ValueError("source identity is not a SHA256")
    prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp"
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    tracked = {
        manifest_path: _snapshot(manifest_path),
        package / "manifest.sha256": _snapshot(package / "manifest.sha256"),
    }
    records = []
    verified_shards = {}
    try:
        for projection in PROJECTIONS:
            name = f"{prefix}.{projection}"
            n, k = (2560, 640) if projection == "down_proj" else (640, 2560)
            expected_shape = [512, n, k // 64]
            source_tensors = {}
            for suffix in ("scales", "biases", "weight"):
                tensor = manifest["tensors"].get(f"{name}.{suffix}")
                if not isinstance(tensor, dict):
                    raise ValueError(f"missing source tensor: {name}.{suffix}")
                expected = [512, n, k // 8] if suffix == "weight" else expected_shape
                dtype = "U32" if suffix == "weight" else "BF16"
                length = (
                    512 * n * k // 2 if suffix == "weight" else 512 * n * k // 64 * 2
                )
                if (
                    tensor.get("shape") != expected
                    or tensor.get("dtype") != dtype
                    or tensor.get("length") != length
                ):
                    raise ValueError(
                        f"unsupported source tensor geometry: {name}.{suffix}"
                    )
                path = _safe_source(package, tensor)
                tracked.setdefault(path, _snapshot(path))
                if path not in verified_shards:
                    source_record = next(
                        (
                            s
                            for s in manifest.get("shards", [])
                            if s.get("path") == tensor["shard"]
                        ),
                        None,
                    )
                    if (
                        not isinstance(source_record, dict)
                        or source_record.get("bytes") != path.stat().st_size
                        or _sha(path) != source_record.get("sha256")
                    ):
                        raise ValueError(f"source shard hash/extent differs: {path}")
                    verified_shards[path] = source_record["sha256"]
                source_tensors[suffix] = {"name": f"{name}.{suffix}", **tensor}
            source_arrays = {}
            for suffix in ("scales", "biases"):
                t = source_tensors[suffix]
                source_arrays[suffix] = np.memmap(
                    package / t["shard"],
                    dtype="<u2",
                    mode="r",
                    offset=t["offset"],
                    shape=expected_shape,
                )
            path = staging / f"{projection}.bf16"
            digest = hashlib.sha256()
            group_count = 0
            with path.open("xb") as stream:
                for begin in range(0, 512, chunk_experts):
                    end = min(512, begin + chunk_experts)
                    tables = bf16_lut(
                        source_arrays["scales"][begin:end],
                        source_arrays["biases"][begin:end],
                    )
                    packed = tiled_lut(tables)
                    raw = memoryview(packed).cast("B")
                    stream.write(raw)
                    digest.update(raw)
                    group_count += (end - begin) * n * (k // 64)
                stream.flush()
                os.fsync(stream.fileno())
            del source_arrays
            expected_bytes = 512 * n * k // 64 * 16 * 2
            if path.stat().st_size != expected_bytes:
                raise ValueError("converted LUT extent differs")
            records.append(
                {
                    "projection": projection,
                    "prefix": name,
                    "path": path.name,
                    "dtype": "BF16",
                    "shape": [512, n // 64, k // 64, 64, 16],
                    "bytes": expected_bytes,
                    "sha256": digest.hexdigest(),
                    "source_tensors": source_tensors,
                    "groups_checked": group_count,
                    "coefficients_checked": group_count * 16,
                }
            )
            print(
                json.dumps(
                    {
                        "projection": name,
                        "saved_bytes": expected_bytes,
                        "groups_checked": group_count,
                        "gpu_work": False,
                    }
                ),
                flush=True,
            )
        for path, before in tracked.items():
            if _snapshot(path) != before:
                raise ValueError(f"source changed during conversion: {path}")
        metadata = {
            "schema": SCHEMA,
            "layout": LAYOUT,
            "coefficient_policy": POLICY,
            "source_identity_sha256": source_identity,
            "source_manifest_sha256": manifest_sha,
            "source_package": str(package),
            "verified_source_shards": [
                {"path": str(path.relative_to(package)), "sha256": sha}
                for path, sha in sorted(verified_shards.items())
            ],
            "prefix": prefix,
            "layer": layer,
            "source_codes_reused": True,
            "original_checkpoint_modified": False,
            "nonfinite_policy": "reject any nonfinite source parameter, F32 reconstructed coefficient or BF16 rounded coefficient before publication",
            "alignment": ALIGNMENT,
            "payloads": records,
            "saved_lut_bytes": sum(t["bytes"] for t in records),
            "all_48_layers_lut_bytes": 48 * sum(t["bytes"] for t in records),
        }
        metadata_raw = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
        (staging / "manifest.json").write_bytes(metadata_raw)
        (staging / "manifest.sha256").write_text(
            hashlib.sha256(metadata_raw).hexdigest() + "\n"
        )
        os.rename(staging, output)
        return metadata
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--chunk-experts", type=int, default=8)
    args = parser.parse_args()
    metadata = convert_layer(
        args.package, args.output, args.layer, chunk_experts=args.chunk_experts
    )
    print(
        json.dumps(
            {
                "pass": True,
                "gpu_work": False,
                "output": str(args.output.resolve()),
                "saved_lut_bytes": metadata["saved_lut_bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
