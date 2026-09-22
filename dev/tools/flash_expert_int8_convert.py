"""CPU-only signed INT8 expert coefficient pilot for the aligned Flash package.

This writes a separate derived package. Source tensors are mapped read-only;
the tool neither changes the model checkpoint nor selects a production route.
The BF16 reference is the blocked-MoE operand boundary: separate F32 multiply
and add for q*scale+bias, followed by round-to-nearest-even BF16. CPU output
proxies are error diagnostics, not model correctness or GPU speed evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = "splash-flash-expert-int8-pilot-v1"
SOURCE_SCHEMA = "splash-local-qwen4-affine-v1"
ALIGNMENT = 16384
PROJECTIONS = {
    "gate_proj": (640, 2560),
    "up_proj": (640, 2560),
    "down_proj": (2560, 640),
}
SOURCE_COEFFICIENT_POLICY = "F32(q)*F32(scale) then separate F32 add bias; BF16 RNE"


class ConversionError(ValueError):
    pass


def bf16_to_f32(bits):
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ConversionError("non-finite values cannot become pilot BF16 operands")
    bits = values.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def reconstruct_q4_bf16(packed, scales, biases, *, group_size=64):
    """Return exact finite BF16 coefficient bits from contiguous packed Q4."""
    packed = np.asarray(packed)
    scales, biases = np.asarray(scales), np.asarray(biases)
    if (
        packed.dtype != np.dtype("uint32")
        or scales.dtype != np.dtype("uint16")
        or biases.dtype != np.dtype("uint16")
    ):
        raise ConversionError("Q4 requires U32 packed codes and U16 BF16 parameters")
    if packed.ndim < 2 or scales.shape != biases.shape or scales.ndim != packed.ndim:
        raise ConversionError("Q4 tensor rank or parameter geometry is invalid")
    if type(group_size) is not int or group_size <= 0 or group_size % 8:
        raise ConversionError("Q4 group size must be a positive multiple of eight")
    k = packed.shape[-1] * 8
    if k % group_size or scales.shape != (*packed.shape[:-1], k // group_size):
        raise ConversionError("Q4 packed/input/parameter geometry does not agree")
    shifts = np.arange(8, dtype=np.uint32) * np.uint32(4)
    codes = ((packed[..., None] >> shifts) & np.uint32(15)).reshape(
        *packed.shape[:-1], k
    )
    sf = np.repeat(bf16_to_f32(scales), group_size, axis=-1)
    bias = np.repeat(bf16_to_f32(biases), group_size, axis=-1)
    product = np.multiply(codes.astype(np.float32), sf, dtype=np.float32)
    original = np.add(product, bias, dtype=np.float32)
    return f32_to_bf16(original)


def symmetric_int8(reference_bf16, *, group_size=0, fit_rounds=0, allow_clipping=False):
    """Quantize BF16 coefficients with separate F32 symmetric scale per group.

    group_size=0 means one scale for the whole input row. The all-zero group
    uses scale=1 and codes=0; -128 is never produced. Division and scale
    construction use F32, and integer ties use round-to-nearest-even.
    Optional least-squares scale refitting evaluates finite codes and F64
    sufficient statistics, storing each scale as F32. Without allow_clipping,
    every refit is bounded below by the original absmax/127 scale. The default
    fit_rounds=0 exactly preserves the initial pilot's payload bytes.
    """
    bits = np.asarray(reference_bf16)
    if bits.dtype != np.dtype("uint16") or bits.ndim < 2 or not bits.shape[-1]:
        raise ConversionError("reference must contain nonempty BF16 rows")
    k = bits.shape[-1]
    if type(group_size) is not int or group_size < 0 or (group_size and k % group_size):
        raise ConversionError("INT8 group size must be zero or divide the input row")
    if (
        type(fit_rounds) is not int
        or not 0 <= fit_rounds <= 4
        or type(allow_clipping) is not bool
    ):
        raise ConversionError(
            "scale fitting requires zero to four rounds and a boolean clipping policy"
        )
    effective_group = group_size or k
    coefficients = bf16_to_f32(bits)
    if not np.isfinite(coefficients).all():
        raise ConversionError("source BF16 coefficients are non-finite")
    grouped = coefficients.reshape(
        *bits.shape[:-1], k // effective_group, effective_group
    )
    maximum = np.max(np.abs(grouped), axis=-1)
    scales = np.divide(maximum, np.float32(127), dtype=np.float32)
    scales = np.where(maximum == np.float32(0), np.float32(1), scales).astype(
        np.float32
    )
    if (scales <= 0).any() or not np.isfinite(scales).all():
        raise ConversionError("INT8 scales are non-finite or underflowed to zero")
    normalized = np.divide(grouped, scales[..., None], dtype=np.float32)
    codes = np.clip(np.rint(normalized), -127, 127).astype(np.int8)
    lower_bound = scales.copy()
    for _ in range(fit_rounds):
        operands = grouped.astype(np.float64)
        integer = codes.astype(np.float64)
        numerator = np.sum(operands * integer, axis=-1, dtype=np.float64)
        denominator = np.sum(integer * integer, axis=-1, dtype=np.float64)
        fitted = np.divide(
            numerator,
            denominator,
            out=scales.astype(np.float64),
            where=denominator != 0,
        ).astype(np.float32)
        if not allow_clipping:
            fitted = np.maximum(fitted, lower_bound)
        if not np.isfinite(fitted).all() or (fitted <= 0).any():
            raise ConversionError(
                "least-squares scale fitting produced invalid F32 scales"
            )
        scales = fitted
        normalized = np.divide(grouped, scales[..., None], dtype=np.float32)
        codes = np.clip(np.rint(normalized), -127, 127).astype(np.int8)
    return codes.reshape(bits.shape), scales


def reconstruct_int8(codes, scales, *, group_size=0):
    codes, scales = np.asarray(codes), np.asarray(scales)
    if (
        codes.dtype != np.dtype("int8")
        or codes.ndim < 2
        or (codes == np.int8(-128)).any()
    ):
        raise ConversionError("symmetric INT8 codes must be finite signed [-127,127]")
    k = codes.shape[-1]
    if type(group_size) is not int or group_size < 0 or (group_size and k % group_size):
        raise ConversionError("INT8 group size must be zero or divide the input row")
    effective_group = group_size or k
    if scales.dtype != np.dtype("float32") or scales.shape != (
        *codes.shape[:-1],
        k // effective_group,
    ):
        raise ConversionError("INT8 scale geometry or precision is invalid")
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ConversionError("INT8 scales must be finite and positive")
    operands = codes.reshape(
        *codes.shape[:-1], k // effective_group, effective_group
    ).astype(np.float32)
    return np.multiply(operands, scales[..., None], dtype=np.float32).reshape(
        codes.shape
    )


def error_metrics(reference, candidate):
    reference, candidate = (
        np.asarray(reference, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
    )
    if (
        reference.shape != candidate.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
    ):
        raise ConversionError("error comparison requires matching finite arrays")
    delta = candidate - reference
    norm = float(np.linalg.norm(reference.reshape(-1)))
    difference = float(np.linalg.norm(delta.reshape(-1)))
    return {
        "rl2": difference / norm if norm else (0.0 if difference == 0 else None),
        "max_absolute_error": float(np.max(np.abs(delta), initial=0)),
        "rmse": float(math.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "elements": int(delta.size),
    }


def _snapshot(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_path(package, relative):
    if not isinstance(relative, str):
        raise ConversionError("source shard path must be text")
    path = (package / relative).resolve()
    if not path.is_relative_to(package) or not path.is_file():
        raise ConversionError("source shard path is missing or escapes the package")
    return path


def _read_tensor(package, descriptor, expected_shape, expected_dtype, snapshots):
    if (
        descriptor.get("shape") != list(expected_shape)
        or descriptor.get("dtype") != expected_dtype
    ):
        raise ConversionError(
            "source tensor geometry/dtype does not match Q4 G64 expert coefficients"
        )
    item_bytes = 4 if expected_dtype == "U32" else 2
    length = math.prod(expected_shape) * item_bytes
    offset = descriptor.get("offset")
    if (
        type(offset) is not int
        or offset < 0
        or offset % ALIGNMENT
        or descriptor.get("length") != length
    ):
        raise ConversionError("source tensor length/alignment is invalid")
    path = _source_path(package, descriptor.get("shard"))
    before = snapshots.setdefault(path, _snapshot(path))
    if _snapshot(path) != before or offset + length > before[2]:
        raise ConversionError("source shard changed or tensor exceeds its shard")
    dtype = "<u4" if expected_dtype == "U32" else "<u2"
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=expected_shape)


def _write_aligned(stream, array):
    offset = (stream.tell() + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    stream.write(bytes(offset - stream.tell()))
    raw = np.ascontiguousarray(array).tobytes()
    stream.write(raw)
    return {
        "offset": offset,
        "length": len(raw),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _output_proxy(reference, candidates, *, seed, bf16_operand_rounding=True):
    """Exact F64 sum of products of finite BF16 operands; then BF16 RNE."""
    coefficients = bf16_to_f32(reference)
    k = reference.shape[-1]
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((4, k)).astype(np.float32)
    structured = np.zeros((4, k), dtype=np.float32)
    structured[0] = np.float32(1)
    structured[1] = np.where(np.arange(k) % 2, np.float32(-1), np.float32(1))
    structured[2, 0] = np.float32(1)
    structured[3, k - 1] = np.float32(-1)
    inputs = bf16_to_f32(f32_to_bf16(np.concatenate((inputs, structured))))
    original = coefficients.astype(np.float64) @ inputs.astype(np.float64).T
    original_bits = f32_to_bf16(original.astype(np.float32))
    source_rounded = bf16_to_f32(original_bits)
    report = {
        "scope": "8 BF16 input vectors; F64 sum of products; no GPU reduction or nonlinear MoE claims",
        "candidate_coefficient_boundary": "BF16 RNE"
        if bf16_operand_rounding
        else "F32 reconstructed code*scale, without BF16 coefficient rounding",
        "vectors": 8,
        "seed": seed,
        "input_sha256": _digest_array(f32_to_bf16(inputs)),
    }
    for label, candidate in candidates.items():
        reconstructed = (
            bf16_to_f32(f32_to_bf16(candidate)) if bf16_operand_rounding else candidate
        )
        result = reconstructed.astype(np.float64) @ inputs.astype(np.float64).T
        rounded = f32_to_bf16(result.astype(np.float32))
        report[label] = {
            "before_output_bf16": error_metrics(original, result),
            "after_output_bf16": error_metrics(source_rounded, bf16_to_f32(rounded)),
            "bf16_output_bits_changed": int(np.count_nonzero(original_bits != rounded)),
            "output_elements": int(rounded.size),
            "nonzero_sign_changes": int(np.count_nonzero((original * result) < 0)),
        }
    return report


def model_byte_estimate(*, target_layers=48, mtp_layers=1, experts=512):
    layers = target_layers + mtp_layers
    coefficients = layers * experts * sum(n * k for n, k in PROJECTIONS.values())
    rows = layers * experts * sum(n for n, _ in PROJECTIONS.values())
    return {
        "scope": "unrounded expert coefficients and scales only; original dense/PLE weights, workspaces and alignment excluded",
        "target_layers": target_layers,
        "mtp_layers": mtp_layers,
        "experts_per_layer": experts,
        "source_q4_g64_bf16_scale_bias_bytes": coefficients // 2
        + coefficients // 64 * 4,
        "int8_rowwise_f32_scale_bytes": coefficients + rows * 4,
        "int8_g64_f32_scale_bytes": coefficients + coefficients // 64 * 4,
        "bf16_coefficients_bytes": coefficients * 2,
    }


def convert(
    package,
    output,
    *,
    layer=0,
    experts=(0, 127),
    group_size=0,
    include_reference=True,
    raw_planes=False,
    fit_rounds=0,
    allow_clipping=False,
    seed=801,
):
    package, output = Path(package).resolve(), Path(output).resolve()
    if type(layer) is not int or not 0 <= layer < 48:
        raise ConversionError("pilot target layer must be in [0,47]")
    if (
        not experts
        or len(experts) > 32
        or len(set(experts)) != len(experts)
        or any(type(e) is not int or not 0 <= e < 512 for e in experts)
    ):
        raise ConversionError(
            "pilot requires one to 32 unique integer expert IDs in [0,511]"
        )
    if group_size not in (0, 64):
        raise ConversionError("pilot supports per-row or G64 F32 scales")
    if (
        output.exists()
        or output.is_relative_to(package)
        or package.is_relative_to(output)
    ):
        raise ConversionError(
            "derived output must be a new directory separate from the source package"
        )
    manifest_path = package / "manifest.json"
    source_before = _snapshot(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    identity = manifest.get("source_identity_sha256")
    if (
        manifest.get("schema") != SOURCE_SCHEMA
        or not isinstance(identity, str)
        or len(identity) != 64
        or any(c not in "0123456789abcdef" for c in identity)
    ):
        raise ConversionError("source aligned package schema or identity is invalid")
    tensors, quantization = (
        manifest.get("tensors", {}),
        manifest.get("quantization", {}),
    )
    source_snapshots = {manifest_path: source_before}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    result = {
        "schema": SCHEMA,
        "source_identity_sha256": identity,
        "source_package_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_package": str(package),
        "alignment": ALIGNMENT,
        "quantization_format": "signed-symmetric-int8-rowwise-f32-scale"
        if not group_size
        else "signed-symmetric-int8-g64-f32-scale",
        "scale_precision": "F32",
        "coefficient_operand_policy": (
            "BF16 activations times signed INT8 codes; F32 per-K64 partial dots; F32 per-output-row/group scale for each partial dot; F32 sum of scaled partials; final BF16 output RNE"
            if group_size == 64
            else "BF16 activations times signed INT8 codes; F32 dot accumulation; F32 per-row scale after accumulation; BF16 output RNE"
        ),
        "source_coefficient_policy": SOURCE_COEFFICIENT_POLICY,
        "integer_rounding": "F32 division; nearest-even integer; clamp [-127,127]; all-zero group scale=1",
        "scale_fit_rounds": fit_rounds,
        "scale_fit_allow_clipping": allow_clipping,
        "expert_ids": list(experts),
        "source_expert_to_compact_rank": [
            experts.index(e) if e in experts else -1 for e in range(512)
        ],
        "layer": layer,
        "payload": "weights.bin",
        "reference_payload": "reference-bf16.bin" if include_reference else None,
        "matrices": {},
        "model_byte_estimate": model_byte_estimate(),
        "qualification": "CPU reconstruction and output proxies only; no model or GPU qualification",
    }
    try:
        with (temporary / "weights.bin").open("wb") as payload:
            reference_stream = (
                (temporary / "reference-bf16.bin").open("wb")
                if include_reference
                else None
            )
            try:
                for index, (projection, (n, k)) in enumerate(PROJECTIONS.items()):
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
                        q.get("bits") != 4
                        or q.get("group_size") != 64
                        or q.get("mode") != "affine"
                    ):
                        raise ConversionError(f"{prefix}: source must be affine Q4 G64")
                    descriptors = {}
                    for suffix, shape, dtype in (
                        ("weight", (512, n, k // 8), "U32"),
                        ("scales", (512, n, k // 64), "BF16"),
                        ("biases", (512, n, k // 64), "BF16"),
                    ):
                        descriptor = tensors.get(f"{prefix}.{suffix}")
                        if not isinstance(descriptor, dict):
                            raise ConversionError(
                                f"missing source tensor {prefix}.{suffix}"
                            )
                        descriptors[suffix] = descriptor
                    mapped = {
                        suffix: _read_tensor(
                            package, descriptors[suffix], shape, dtype, source_snapshots
                        )
                        for suffix, shape, dtype in (
                            ("weight", (512, n, k // 8), "U32"),
                            ("scales", (512, n, k // 64), "BF16"),
                            ("biases", (512, n, k // 64), "BF16"),
                        )
                    }
                    selected = {
                        suffix: np.asarray(value[list(experts)]).copy()
                        for suffix, value in mapped.items()
                    }
                    del mapped
                    reference = reconstruct_q4_bf16(
                        selected["weight"], selected["scales"], selected["biases"]
                    )
                    codes, scales = symmetric_int8(
                        reference,
                        group_size=group_size,
                        fit_rounds=fit_rounds,
                        allow_clipping=allow_clipping,
                    )
                    candidate = reconstruct_int8(codes, scales, group_size=group_size)
                    row_codes, row_scales = symmetric_int8(reference)
                    row_candidate = reconstruct_int8(row_codes, row_scales)
                    g64_codes, g64_scales = symmetric_int8(reference, group_size=64)
                    g64_candidate = reconstruct_int8(
                        g64_codes, g64_scales, group_size=64
                    )
                    coefficient_reference = bf16_to_f32(reference)
                    matrices = {
                        "source_prefix": prefix,
                        "dimensions": [len(experts), n, k],
                        "layout": "compact expert, output row, input coefficient; contiguous K then N then E",
                        "effective_group_size": group_size or k,
                        "coefficient_dtype": "I8",
                        "scale_dtype": "F32",
                        "coefficients": _write_aligned(payload, codes),
                        "scales": _write_aligned(payload, scales.astype("<f4")),
                        "source_tensors": descriptors,
                        "source_selected_array_sha256": {
                            suffix: _digest_array(value)
                            for suffix, value in selected.items()
                        },
                        "source_bf16_coefficients_sha256": _digest_array(
                            reference.astype("<u2")
                        ),
                        "quality": {},
                    }
                    normalized = np.divide(
                        coefficient_reference.reshape(
                            *reference.shape[:-1],
                            k // (group_size or k),
                            group_size or k,
                        ),
                        scales[..., None],
                        dtype=np.float32,
                    )
                    matrices["chosen_candidate_quality"] = {
                        "f32_coefficient_error": error_metrics(
                            coefficient_reference, candidate
                        ),
                        "bf16_operand_error": error_metrics(
                            coefficient_reference, bf16_to_f32(f32_to_bf16(candidate))
                        ),
                        "normalized_values_exceeding_127": int(
                            np.count_nonzero(np.abs(normalized) > np.float32(127))
                        ),
                        "rounded_codes_clipped": int(
                            np.count_nonzero(
                                np.abs(np.rint(normalized)) > np.float32(127)
                            )
                        ),
                        "coefficient_elements": int(normalized.size),
                    }
                    for label, reconstruction in (
                        ("rowwise_f32_scale", row_candidate),
                        ("g64_f32_scale", g64_candidate),
                    ):
                        rounded = f32_to_bf16(reconstruction)
                        matrices["quality"][label] = {
                            "f32_coefficient_error": error_metrics(
                                coefficient_reference, reconstruction
                            ),
                            "bf16_operand_error": error_metrics(
                                coefficient_reference, bf16_to_f32(rounded)
                            ),
                            "bf16_coefficients_changed": int(
                                np.count_nonzero(reference != rounded)
                            ),
                        }
                    # Every coefficient is compared above. A bounded subset of
                    # two experts is enough for representative output proxies;
                    # the GPU oracle will exercise all selected route IDs.
                    output_candidates = {
                        "rowwise_f32_scale": row_candidate[:2],
                        "g64_f32_scale": g64_candidate[:2],
                    }
                    if fit_rounds:
                        output_candidates["chosen_fitted_candidate"] = candidate[:2]
                    matrices["cpu_output_proxy"] = _output_proxy(
                        reference[:2],
                        output_candidates,
                        seed=seed + index,
                        bf16_operand_rounding=False,
                    )
                    matrices["cpu_output_proxy"]["expert_ids"] = list(experts[:2])
                    matrices["cpu_bf16_coefficient_output_proxy"] = _output_proxy(
                        reference[:2], output_candidates, seed=seed + index
                    )
                    matrices["cpu_bf16_coefficient_output_proxy"]["expert_ids"] = list(
                        experts[:2]
                    )
                    if raw_planes:
                        directory = temporary / "raw-planes"
                        directory.mkdir(exist_ok=True)
                        for label, array, suffix in (
                            ("codes", codes, "i8"),
                            ("scales", scales.astype("<f4"), "f32"),
                        ):
                            filename = f"raw-planes/{projection}.{label}.{suffix}.bin"
                            (temporary / filename).write_bytes(
                                np.ascontiguousarray(array).tobytes()
                            )
                            matrices[f"raw_{label}_plane"] = {
                                "path": filename,
                                "length": array.nbytes,
                                "sha256": _digest_array(array),
                            }
                    if reference_stream:
                        matrices["reference_bf16"] = _write_aligned(
                            reference_stream, reference.astype("<u2")
                        )
                    result["matrices"][projection] = matrices
                    print(
                        json.dumps(
                            {
                                "projection": projection,
                                "experts": list(experts),
                                "rowwise_coefficient_rl2": matrices["quality"][
                                    "rowwise_f32_scale"
                                ]["bf16_operand_error"]["rl2"],
                                "g64_coefficient_rl2": matrices["quality"][
                                    "g64_f32_scale"
                                ]["bf16_operand_error"]["rl2"],
                            }
                        ),
                        flush=True,
                    )
            finally:
                if reference_stream:
                    reference_stream.close()
            payload.flush()
            os.fsync(payload.fileno())
        for path, snapshot in source_snapshots.items():
            if _snapshot(path) != snapshot:
                raise ConversionError(f"source changed during conversion: {path}")
        result["source_readonly_snapshots_unchanged"] = True
        result["payload_bytes"] = (temporary / "weights.bin").stat().st_size
        result["payload_sha256"] = _digest_file(temporary / "weights.bin")
        if include_reference:
            result["reference_payload_bytes"] = (
                (temporary / "reference-bf16.bin").stat().st_size
            )
            result["reference_payload_sha256"] = _digest_file(
                temporary / "reference-bf16.bin"
            )
        (temporary / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(output)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("install/local-models/Flash-Next-oQ4e-mtp-v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", default="0,127")
    parser.add_argument("--group-size", type=int, choices=(0, 64), default=0)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument(
        "--raw-planes",
        action="store_true",
        help="write independent raw coefficient and scale plane files for isolated oracles",
    )
    parser.add_argument("--fit-scale-rounds", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument(
        "--fit-allow-clipping",
        action="store_true",
        help="allow least-squares fitted scales below absmax/127; report any clipped codes",
    )
    args = parser.parse_args()
    try:
        experts = tuple(int(part) for part in args.experts.split(","))
        result = convert(
            args.package,
            args.output,
            layer=args.layer,
            experts=experts,
            group_size=args.group_size,
            include_reference=not args.no_reference,
            raw_planes=args.raw_planes,
            fit_rounds=args.fit_scale_rounds,
            allow_clipping=args.fit_allow_clipping,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "payload_bytes": result["payload_bytes"],
                    "payload_sha256": result["payload_sha256"],
                },
                sort_keys=True,
            )
        )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"conversion failed: {error}\n")


if __name__ == "__main__":
    main()
