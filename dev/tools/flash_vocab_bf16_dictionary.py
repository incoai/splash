"""CPU-only exact BF16 vocabulary dictionary; never initializes a GPU.

The original Q8 codes remain in the immutable source model. Each original
BF16 (scale,bias) pair is assigned a uint16 index into an exact table of all
256 once-rounded affine coefficients. This format changes storage only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

PREFIX = "language_model.lm_head"
SCHEMA = "splash-private-exact-bf16-q8-vocabulary-dictionary-v1"
POLICY = "F32-promoted-original-BF16-scale-times-uint8-plus-original-BF16-bias-once-BF16-RNE-v1"
N, K, G = 248320, 2560, 64


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def bf16_words(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("F32 coefficient is nonfinite")
    words = values.view(np.uint32)
    rounded = ((words + np.uint32(0x7FFF) + ((words >> 16) & 1)) >> 16).astype(
        np.uint16
    )
    if np.any((rounded & 0x7F80) == 0x7F80):
        raise ValueError("BF16 coefficient is nonfinite")
    return rounded


def bf16_float(words):
    return (np.asarray(words, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def make_dictionary(pairs):
    pairs = np.asarray(pairs, dtype=np.uint32)
    scales = bf16_float((pairs & 0xFFFF).astype(np.uint16))
    biases = bf16_float((pairs >> 16).astype(np.uint16))
    if not np.all(np.isfinite(scales)) or not np.all(np.isfinite(biases)):
        raise ValueError("source affine parameters are nonfinite")
    codes = np.arange(256, dtype=np.float32)
    # Separate ufuncs guarantee no FMA and no implicit float64 intermediate.
    with np.errstate(over="ignore", invalid="ignore"):
        product = np.multiply(scales[:, None], codes, dtype=np.float32)
        reconstructed = np.add(product, biases[:, None], dtype=np.float32)
    return bf16_words(reconstructed)


def round_f64_to_bf16(values):
    """Independent RNE using adjacent BF16 cells and exact float64 distances.

    Each input is an exact affine sum for the actual normal BF16 source
    parameters: eight-bit significand times an eight-bit integer has <=16
    significant bits; actual exponent differences are checked in the report.
    """
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("independent reconstruction is nonfinite")
    magnitude = np.abs(values)
    # F32 rounding can cross a BF16 cell only at its own much finer boundary;
    # comparing the candidate and both adjacent cells covers that crossing.
    approximate = (magnitude.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    candidates = np.stack(
        (
            np.maximum(approximate.astype(np.int32) - 1, 0),
            approximate.astype(np.int32),
            approximate.astype(np.int32) + 1,
        ),
        axis=-1,
    ).astype(np.uint16)
    numeric = bf16_float(candidates).astype(np.float64)
    distances = np.abs(numeric - magnitude[..., None])
    minimum = distances.min(axis=-1, keepdims=True)
    # Tie-even takes precedence; otherwise the unique nearest cell wins.
    # Rank parity first, then bitword for deterministic zero.
    ranked = np.where(
        distances == minimum,
        (candidates & 1).astype(np.uint32) * 65536 + candidates,
        np.uint32(0xFFFFFFFF),
    )
    best = np.take_along_axis(candidates, ranked.argmin(axis=-1)[..., None], axis=-1)[
        ..., 0
    ]
    return best | (np.signbit(values).astype(np.uint16) << 15)


def snapshot(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def source_array(package, manifest, suffix):
    record = manifest["tensors"][PREFIX + "." + suffix]
    shape = [N, K // 4] if suffix == "weight" else [N, K // G]
    dtype, length = ("U32", N * K) if suffix == "weight" else ("BF16", N * K // G * 2)
    if (
        record["shape"] != shape
        or record["dtype"] != dtype
        or record["length"] != length
    ):
        raise ValueError("unexpected source geometry: " + suffix)
    path = (package / record["shard"]).resolve()
    if (
        not path.is_relative_to(package)
        or record["offset"] < 0
        or record["offset"] % 16384
    ):
        raise ValueError("source extent or path is invalid")
    if record["offset"] + length > path.stat().st_size:
        raise ValueError("source extent exceeds file")
    mapped_shape = [N, K] if suffix == "weight" else shape
    return (
        np.memmap(
            path,
            dtype="u1" if suffix == "weight" else "<u2",
            mode="r",
            offset=record["offset"],
            shape=tuple(mapped_shape),
        ),
        record,
        path,
    )


def cancellation_proofs():
    # Include actual source pairs where BF16 product-first rounding changes the
    # once-rounded coefficient, plus exact cancellation and subnormal cases.
    traps = [
        (0x3C01, 0xBD00, 127),
        (0x3B81, 0xBF01, 128),
        (0xBC01, 0x3D00, 127),
        (0x0001, 0x8001, 1),
        (0xB8A1, 0x3C22, 25),
        (0x38A1, 0xBC22, 25),
    ]
    rows = []
    for scale, bias, code in traps:
        pair = np.array([scale | (bias << 16)], dtype=np.uint32)
        actual = int(make_dictionary(pair)[0, code])
        sf, bs = map(
            lambda word: float(bf16_float(np.array([word], dtype=np.uint16))[0]),
            (scale, bias),
        )
        expected = int(round_f64_to_bf16(np.array([sf * code + bs]))[0])
        product_first = int(
            bf16_words(
                np.add(
                    bf16_float(bf16_words(np.array([sf * code], dtype=np.float32))),
                    np.float32(bs),
                    dtype=np.float32,
                )
            )[0]
        )
        if actual != expected:
            raise ValueError("independent cancellation coefficient differs")
        rows.append(
            {
                "scale_bf16": scale,
                "bias_bf16": bias,
                "code": code,
                "exact_bf16": actual,
                "incorrect_product_first_bf16": product_first,
                "distinguishes_product_first": actual != product_first,
            }
        )
    if not any(row["distinguishes_product_first"] for row in rows):
        raise ValueError("no cancellation convention trap")
    zeros = np.array([0.0, -0.0], dtype=np.float32)
    if not np.array_equal(bf16_words(zeros), np.array([0, 0x8000], dtype=np.uint16)):
        raise ValueError("signed zero rounding changed")
    return rows


def convert(package, output, operand_store, report_path):
    start = time.monotonic()
    package, output = package.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(package):
        raise ValueError("output must be new and outside original package")
    if report_path is not None and (
        report_path.exists() or report_path.resolve().is_relative_to(package)
    ):
        raise ValueError("report must be new and outside original package")
    manifest_path = package / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest_sha = sha_bytes(raw)
    if manifest_sha != (package / "manifest.sha256").read_text().split()[0]:
        raise ValueError("source manifest hash differs")
    manifest = json.loads(raw)
    if manifest.get("schema") != "splash-local-qwen4-affine-v1":
        raise ValueError("unexpected source schema")
    tracked = {
        manifest_path: snapshot(manifest_path),
        package / "manifest.sha256": snapshot(package / "manifest.sha256"),
    }
    scale, scale_record, path = source_array(package, manifest, "scales")
    tracked[path] = snapshot(path)
    bias, bias_record, path = source_array(package, manifest, "biases")
    tracked[path] = snapshot(path)
    pairs = scale.astype(np.uint32) | (bias.astype(np.uint32) << 16)
    unique, inverse, counts = np.unique(pairs, return_inverse=True, return_counts=True)
    if unique.size > 65536:
        raise ValueError("pair dictionary does not fit uint16")
    indices = inverse.reshape(N, K // G).astype("<u2")
    if not np.array_equal(unique[indices], pairs):
        raise ValueError("dictionary-index pair reconstruction differs")
    dictionary = make_dictionary(unique)
    scales = bf16_float((unique & 0xFFFF).astype(np.uint16)).astype(np.float64)
    biases = bf16_float((unique >> 16).astype(np.uint16)).astype(np.float64)
    mismatches = 0
    exact_f32_mismatches = 0
    product_first_changes = 0
    for begin in range(0, len(unique), 256):
        end = min(len(unique), begin + 256)
        q = np.arange(256, dtype=np.float64)
        exact = scales[begin:end, None] * q + biases[begin:end, None]
        independent = round_f64_to_bf16(exact)
        mismatches += int(np.count_nonzero(independent != dictionary[begin:end]))
        exact_f32_mismatches += int(
            np.count_nonzero(exact != exact.astype(np.float32).astype(np.float64))
        )
        product_first = bf16_float(
            bf16_words((scales[begin:end, None] * q).astype(np.float32))
        )
        wrong = bf16_words(
            np.add(
                product_first,
                biases[begin:end, None].astype(np.float32),
                dtype=np.float32,
            )
        )
        product_first_changes += int(np.count_nonzero(wrong != dictionary[begin:end]))
    if mismatches:
        raise ValueError(f"independent BF16 coefficient mismatches: {mismatches}")
    traps = cancellation_proofs()
    sample = None
    code_record = manifest["tensors"][PREFIX + ".weight"]
    if operand_store is not None:
        operand_store = operand_store.resolve()
        operand_manifest_path = operand_store / "manifest.json"
        tracked[operand_manifest_path] = snapshot(operand_manifest_path)
        operand_manifest_raw = operand_manifest_path.read_bytes()
        if (
            sha_bytes(operand_manifest_raw)
            != (operand_store / "manifest.sha256").read_text().split()[0]
        ):
            raise ValueError("operand manifest hash differs")
        operands = json.loads(operand_manifest_raw)
        if operands["source_identity_sha256"] != manifest["source_identity_sha256"]:
            raise ValueError("persisted operand source identity differs")
        entry = next(
            v
            for v in operands["entries"]
            if v["projection"] == PREFIX and v["format"] == "BF16"
        )
        if entry["shape"] != [N, K] or entry["logical_bytes"] != N * K * 2:
            raise ValueError("persisted cache geometry differs")
        saved_path = (operand_store / entry["file"]).resolve()
        if not saved_path.is_relative_to(operand_store):
            raise ValueError("persisted cache path escapes store")
        tracked[saved_path] = snapshot(saved_path)
        cached = np.memmap(
            saved_path,
            dtype="<u2",
            mode="r",
            offset=entry["offset_bytes"],
            shape=(N, K),
        )
        codes, code_record, path = source_array(package, manifest, "weight")
        tracked[path] = snapshot(path)
        # Every BF16 scale/bias pair appears at least once; sample all 256 codes
        # for each pair in synthetic proof, then sample real saved coefficients
        # in 257 vocabulary rows including both EOS IDs and aligned boundaries.
        sample_rows = np.unique(
            np.concatenate(
                (
                    np.linspace(0, N - 1, 257, dtype=np.int64),
                    np.array([248044, 248046, 63, 64, 127, 128]),
                )
            )
        )
        real_codes = np.asarray(codes[sample_rows])
        decoded = dictionary[
            indices[sample_rows[:, None], np.arange(K) // G], real_codes
        ]
        saved = np.asarray(cached[sample_rows])
        changed = int(np.count_nonzero(decoded != saved))
        if changed:
            raise ValueError(f"persisted cache coefficient mismatches: {changed}")
        sample = {
            "rows": int(sample_rows.size),
            "coefficients": int(decoded.size),
            "mismatches": changed,
            "source_code_bytes_read": int(real_codes.nbytes),
            "cached_bf16_bytes_read": int(saved.nbytes),
            "cached_payload_recorded_sha256": entry["payload_sha256"],
            "sample_decoded_sha256": sha_bytes(decoded.tobytes()),
            "sample_saved_sha256": sha_bytes(saved.tobytes()),
            "full_cached_payload_sha_verified": False,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        payloads = []
        for filename, array, layout in (
            ("dictionary.bf16", dictionary.astype("<u2"), "pair,code"),
            ("group_indices.u16", indices, "output_channel,Kgroup64"),
        ):
            payload = array.tobytes()
            with (staging / filename).open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            payloads.append(
                {
                    "file": filename,
                    "shape": list(array.shape),
                    "bytes": len(payload),
                    "sha256": sha_bytes(payload),
                    "dtype": "BF16" if filename.endswith("bf16") else "U16",
                    "layout": layout,
                    "endianness": "little",
                }
            )
        for path, before in tracked.items():
            if snapshot(path) != before:
                raise ValueError("source changed during conversion: " + str(path))
        metadata = {
            "schema": SCHEMA,
            "coefficient_policy": POLICY,
            "source_identity_sha256": manifest["source_identity_sha256"],
            "source_manifest_sha256": manifest_sha,
            "source_package": str(package),
            "projection": PREFIX,
            "source": {
                "codes": code_record,
                "scales": scale_record,
                "biases": bias_record,
            },
            "output_size": N,
            "input_size": K,
            "group_size": G,
            "bits": 8,
            "dictionary_pair_count": int(unique.size),
            "pair_sort": "ascending unsigned(scale_bf16 | bias_bf16 <<16)",
            "decode": "dictionary[group_indices[n*40+k/64]*256+original_code_bytes[n*2560+k]]",
            "source_codes_reused": True,
            "original_model_modified": False,
            "gpu_work": False,
            "source_parameter_bytes_read": scale.nbytes + bias.nbytes,
            "all_dictionary_coefficients_checked": int(dictionary.size),
            "independent_rounding_mismatches": mismatches,
            "exact_f32_affine_vs_f64_mismatches": exact_f32_mismatches,
            "incorrect_product_first_bf16_changes": product_first_changes,
            "cancellation_traps": traps,
            "persisted_cache_sample": sample,
            "payloads": payloads,
            "original_BF16_cache_bytes": N * K * 2,
            "original_Q8_plus_coefficients_bytes": N * K + scale.nbytes + bias.nbytes,
            "dictionary_plus_indices_bytes": dictionary.nbytes + indices.nbytes,
            "Q8_plus_dictionary_plus_indices_bytes": N * K
            + dictionary.nbytes
            + indices.nbytes,
            "unique_scale_count": int(np.unique(scale).size),
            "unique_bias_count": int(np.unique(bias).size),
            "most_common_pair_group_counts": sorted(map(int, counts))[-10:],
            "performance_claim": "none; storage/coefficient proof only",
            "seconds": time.monotonic() - start,
        }
        metadata_raw = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
        manifest_digest = sha_bytes(metadata_raw)
        (staging / "manifest.json").write_bytes(metadata_raw)
        (staging / "manifest.sha256").write_text(manifest_digest + "\n")
        os.rename(staging, output)
        report = {
            "schema": SCHEMA + "-cpu-qualification",
            "pass": True,
            "gpu_work": False,
            "sidecar": str(output),
            "sidecar_manifest_sha256": manifest_digest,
            **metadata,
        }
        if report_path is not None:
            with report_path.open("x") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operand-store", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = convert(args.package, args.output, args.operand_store, args.report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "pass",
                    "gpu_work",
                    "sidecar",
                    "sidecar_manifest_sha256",
                    "dictionary_pair_count",
                    "all_dictionary_coefficients_checked",
                    "independent_rounding_mismatches",
                    "Q8_plus_dictionary_plus_indices_bytes",
                    "seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
