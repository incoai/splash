"""Compare small original PLE row reads against the SSD store and BF16 stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding."


def bf16(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(
        0xFFFF0000
    )
    return rounded.view(np.float32)


def decode_planar(words, scales, biases, shared):
    columns = np.arange(160, dtype=np.uint32)
    q = (words[columns // 8] >> ((columns % 8) * 4)) & np.uint32(15)
    scale = (scales.astype(np.uint32) << 16).view(np.float32)
    bias = (biases.astype(np.uint32) << 16).view(np.float32)
    return bf16(
        bf16(q.astype(np.float32) * scale[columns // 32] + bias[columns // 32]) * shared
    )


def decode_packed(payload, shared):
    # Independently consume byte nibbles and the 100-byte packed staging order.
    output = []
    for column in range(160):
        code = (payload[column // 2] >> ((column % 2) * 4)) & 15
        (sb,) = struct.unpack_from("<H", payload, 80 + (column // 32) * 2)
        (bb,) = struct.unpack_from("<H", payload, 90 + (column // 32) * 2)
        (scale,) = struct.unpack("<f", struct.pack("<I", sb << 16))
        (bias,) = struct.unpack("<f", struct.pack("<I", bb << 16))
        product = np.float32(code) * np.float32(scale)
        affine = bf16(np.float32(product + np.float32(bias)))
        output.append(bf16(np.float32(affine) * shared).item())
    return np.asarray(output, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=Path,
        default=ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1",
    )
    p.add_argument(
        "--build",
        type=Path,
        default=ROOT / "build/flash-ple-ssd-independent-checkpoint-v12",
    )
    a = p.parse_args()
    a.build.mkdir(parents=True, exist_ok=True)
    m = json.loads((a.model / "manifest.json").read_bytes())
    shards = {s["path"]: i for i, s in enumerate(m["shards"])}
    part = []
    for i in range(128):
        family = [
            m["tensors"][PREFIX + f"shards.{i}.{name}"]
            for name in ("weight", "scales", "biases")
        ]
        assert family[0]["shape"] == [2500012, 20] and family[1]["shape"] == family[2][
            "shape"
        ] == [2500012, 5]
        part.append(family)
    ids = [i * 2500012 + local for i in range(128) for local in (0, 2500011)]
    fixture = a.build / "checkpoint-fixture.bin"
    with fixture.open("wb") as f:
        f.write(struct.pack("<I", len(m["shards"])))
        for s in m["shards"]:
            path = str((a.model / s["path"]).resolve()).encode()
            f.write(struct.pack("<QQ", s["bytes"], len(path)))
            f.write(path)
        f.write(struct.pack("<I", 128))
        for family in part:
            f.write(struct.pack("<Q", 2500012))
            for r, stride in zip(family, (80, 10, 10)):
                f.write(struct.pack("<IQQ", shards[r["shard"]], r["offset"], stride))
        f.write(struct.pack("<I", len(ids)))
        f.write(struct.pack("<" + str(len(ids)) + "q", *ids))
    flags = [
        "-std=c++20",
        "-O1",
        "-g",
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        "-I" + str(ROOT / "runtime"),
    ]
    exe = a.build / "checkpoint-store-oracle"
    subprocess.run(
        [
            "xcrun",
            "clang++",
            *flags,
            str(ROOT / "dev/tests/flash/flash_ple_ssd_checkpoint_store_cpu.cpp"),
            str(ROOT / "runtime/flash/FlashPLESSDStore.cpp"),
            "-o",
            str(exe),
        ],
        check=True,
    )
    output = a.build / "actual-packed-rows.bin"
    result = json.loads(
        subprocess.check_output([str(exe), str(fixture), str(output)], text=True)
    )
    actual = output.read_bytes()
    expected = bytearray()
    decoded_checks = 0
    reference_bytes = 0
    shared_record = m["tensors"][PREFIX + "weight_scale"]
    with (a.model / shared_record["shard"]).open("rb") as f:
        f.seek(shared_record["offset"])
        (shared_bits,) = struct.unpack("<H", f.read(2))
    shared = np.float32(struct.unpack("<f", struct.pack("<I", shared_bits << 16))[0])
    for selection, id0 in enumerate(ids):
        shard, row = divmod(id0, 2500012)
        chunks = []
        for record, stride in zip(part[shard], (80, 10, 10)):
            with (a.model / record["shard"]).open("rb") as f:
                f.seek(record["offset"] + row * stride)
                data = f.read(stride)
            assert len(data) == stride
            chunks.append(data)
            reference_bytes += stride
        expected.extend(b"".join(chunks))
        planar = decode_planar(
            np.frombuffer(chunks[0], dtype="<u4"),
            np.frombuffer(chunks[1], dtype="<u2"),
            np.frombuffer(chunks[2], dtype="<u2"),
            shared,
        )
        packed = decode_packed(actual[selection * 100 : (selection + 1) * 100], shared)
        np.testing.assert_array_equal(planar.view(np.uint32), packed.view(np.uint32))
        assert np.isfinite(packed).all()
        decoded_checks += 160
    assert actual == expected
    result.update(
        {
            "schema": "flash-ple-ssd-checkpoint-store-cpu-v1",
            "sanitizers": ["address", "undefined"],
            "reference_row_bytes_read": reference_bytes,
            "small_scalar_bytes_read": 2,
            "decoded_bf16_values_compared": decoded_checks,
            "packed_bytes_compared": len(actual),
            "packed_rows_sha256": hashlib.sha256(actual).hexdigest(),
            "all_128_shard_first_last_rows": True,
            "limitations": [
                "CPU byte and decode proof only",
                "No Metal kernel or model/service execution",
                "F_NOCACHE does not guarantee physical SSD misses",
            ],
        }
    )
    report = a.build / "qualification.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "valid": True,
                "packed_bytes_compared": len(actual),
                "decoded_bf16_values_compared": decoded_checks,
                "report": str(report),
            }
        )
    )


if __name__ == "__main__":
    main()
