"""Independent CPU qualification for optional SSD PLE hashing and map windows.

This script reads 282 bytes of actual PLE scalar/I64 checkpoint data, creates
small deterministic fixtures, and links only production CPU hash/layout code.
No MLX imports, Metal device creation, table mappings, or GPU work occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PLE = "language_model.model.layers.1.ple.ple_embedding."
MASK64 = (1 << 64) - 1


def signed(value):
    value &= MASK64
    return value - (1 << 64) if value >= 1 << 63 else value


def reference(tokens, history, multipliers, sizes, offsets, eos):
    """Segment-based source algorithm; no two-token recurrence reuse."""
    whole = list(history) + list(tokens)
    shifted = []
    for shift in range(3):
        previous_eos = -1
        values = []
        for position, token in enumerate(whole):
            source = position - shift
            values.append(whole[source] if source >= max(0, previous_eos + 1) else eos)
            if token == eos:
                previous_eos = position
        shifted.append(values)
    rows = []
    for position in range(2, len(whole)):
        for head in range(16):
            value = 0
            for shift in range(2 if head < 8 else 3):
                value = signed(
                    value ^ signed(shifted[shift][position] * multipliers[shift])
                )
            rows.append(value % sizes[head] + offsets[head])
    return rows, whole[-2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=Path,
        default=ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1",
    )
    ap.add_argument(
        "--build", type=Path, default=ROOT / "build/flash-ple-ssd-independent-cpu-v12"
    )
    ap.add_argument("--sanitize", action="store_true")
    args = ap.parse_args()
    args.build.mkdir(parents=True, exist_ok=True)
    manifest_data = (args.model / "manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    config = json.loads((args.model / "config.json").read_bytes())["text_config"]
    bytes_read = 0
    reads = []

    def scalar(name, fmt):
        nonlocal bytes_read
        record = manifest["tensors"][PLE + name]
        with (args.model / record["shard"]).open("rb") as f:
            f.seek(record["offset"])
            payload = f.read(record["length"])
        if len(payload) != struct.calcsize(fmt):
            raise RuntimeError("unexpected small hash/scalar tensor layout")
        bytes_read += len(payload)
        reads.append(
            {
                "name": PLE + name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        return struct.unpack(fmt, payload)

    multipliers = scalar("layer_multipliers", "<3q")
    sizes = scalar("ngram_heads_vocab_sizes", "<16q")
    offsets = scalar("ngram_heads_offsets", "<16q")
    (scale_bits,) = scalar("ngram_embedding.weight_scale", "<H")
    eos, vocabulary = config["eos_token_id"], config["vocab_size"]
    if isinstance(eos, list):
        eos = eos[0]
    assert eos == 248044 and vocabulary == 248320
    table_rows = sum(
        record["shape"][0]
        for name, record in manifest["tensors"].items()
        if name.startswith(PLE + "ngram_embedding.shards.") and name.endswith(".weight")
    )
    assert table_rows == 320001536 and bytes_read == 282
    rng = random.Random(381616)
    cases = []
    chunk_comparisons = 0
    for lanes in range(1, 5):
        for rows in (1, 2, 4, 16, 512, 2048):
            histories = [
                [eos, eos]
                if lane == 0
                else [lane * 17, eos]
                if lane == 1
                else [eos, lane * 19]
                if lane == 2
                else [29, 31]
                for lane in range(lanes)
            ]
            token_lanes = [
                [rng.randrange(vocabulary) for _ in range(rows)] for _ in range(lanes)
            ]
            for lane, tokens in enumerate(token_lanes):
                for position in (0, 1, rows // 2, rows - 1):
                    if position < rows:
                        tokens[position] = eos if (position + lane) % 2 == 0 else 248046
            expected, after = [], []
            for tokens, history in zip(token_lanes, histories):
                ids, final = reference(
                    tokens, history, multipliers, sizes, offsets, eos
                )
                expected.extend(ids)
                after.extend(final)
                chunked, state = [], history
                cuts = sorted(
                    {
                        0,
                        rows,
                        1,
                        min(rows, 2),
                        min(rows, 4),
                        min(rows, 16),
                        min(rows, 512),
                    }
                )
                for begin, end in zip(cuts, cuts[1:]):
                    before_chunk = list(state)
                    ids0, state = reference(
                        tokens[begin:end], state, multipliers, sizes, offsets, eos
                    )
                    chunked.extend(ids0)
                    cases.append(
                        (
                            1,
                            end - begin,
                            table_rows,
                            multipliers,
                            sizes,
                            offsets,
                            tokens[begin:end],
                            before_chunk,
                            ids0,
                            state,
                        )
                    )
                assert chunked == ids and state == final
                chunk_comparisons += 1
                # Explicit ragged histories: each real lane length is separate,
                # so unused physical padding never becomes hash context.
                for length in sorted({1, max(1, rows - lane), rows}):
                    ragged_ids, ragged_after = reference(
                        tokens[:length], history, multipliers, sizes, offsets, eos
                    )
                    cases.append(
                        (
                            1,
                            length,
                            table_rows,
                            multipliers,
                            sizes,
                            offsets,
                            tokens[:length],
                            history,
                            ragged_ids,
                            ragged_after,
                        )
                    )
            cases.append(
                (
                    lanes,
                    rows,
                    table_rows,
                    multipliers,
                    sizes,
                    offsets,
                    sum(token_lanes, []),
                    sum(histories, []),
                    expected,
                    after,
                )
            )
    # Prefix restoration is request-owned elsewhere. Check the hash contract
    # after every retained-prefix length, including zero/full and EOS-adjacent.
    for rows in (1, 2, 4, 16):
        tokens = [11 + row for row in range(rows)]
        if rows > 1:
            tokens[1] = eos
        if rows > 3:
            tokens[3] = 248046
        before = [37, 41]
        next_tokens = [51, eos, 53, 54]
        for keep in range(rows + 1):
            retained = (before + tokens[:keep])[-2:]
            future, final = reference(
                next_tokens, retained, multipliers, sizes, offsets, eos
            )
            complete, _ = reference(
                tokens[:keep] + next_tokens, before, multipliers, sizes, offsets, eos
            )
            assert future == complete[-len(next_tokens) * 16 :]
            cases.append(
                (
                    1,
                    len(next_tokens),
                    table_rows,
                    multipliers,
                    sizes,
                    offsets,
                    next_tokens,
                    retained,
                    future,
                    final,
                )
            )
    # Synthetic signed products force INT64_MIN and negative remainder cases.
    for mm in (
        (-(1 << 63), 0, 0),
        ((1 << 63) - 1, -1, -(1 << 63)),
        (-9223372036854775791, 391, -37),
    ):
        for tokens in ([1], [1, eos, 248046, 2], [eos, eos, 0, 1, 2, vocabulary - 1]):
            history = [eos, eos]
            ss = (1, 2, 3, 17, 97, 20000003, 20000023, 20000033) * 2
            oo = tuple(range(16))
            ids, after = reference(tokens, history, mm, ss, oo, eos)
            cases.append(
                (1, len(tokens), table_rows, mm, ss, oo, tokens, history, ids, after)
            )
    # PLE EOS differs from the second generation stop: it must retain history.
    a, _ = reference([7, 248046, 11], [3, 4], multipliers, sizes, offsets, eos)
    b, _ = reference([7, eos, 11], [3, 4], multipliers, sizes, offsets, eos)
    assert a[-16:] != b[-16:]
    fixture = args.build / "independent-fixtures.bin"
    with fixture.open("wb") as f:
        f.write(struct.pack("<QI", 0x31564453534C4550, len(cases)))
        for lanes, rows, table, mm, ss, oo, tokens, history, ids, after in cases:
            f.write(struct.pack("<IIQ", lanes, rows, table))
            for values in (mm, ss, oo, tokens, history, ids, after):
                f.write(struct.pack("<" + str(len(values)) + "q", *values))
        f.write(struct.pack("<I", len(manifest["shards"])))
        for shard in manifest["shards"]:
            records = [
                (name, record)
                for name, record in manifest["tensors"].items()
                if record["shard"] == shard["path"]
            ]
            f.write(struct.pack("<QI", shard["bytes"], len(records)))
            for name, record in reversed(records):  # sorting is planner's job.
                disk = name.startswith(PLE + "ngram_embedding.shards.") and name.rsplit(
                    ".", 1
                )[-1] in {"weight", "scales", "biases"}
                f.write(
                    struct.pack(
                        "<QQI",
                        record["offset"],
                        record["offset"] + record["length"],
                        disk,
                    )
                )
    flags = [
        "-std=c++20",
        "-O1" if args.sanitize else "-O2",
        "-ffunction-sections",
        "-fdata-sections",
        "-I" + str(ROOT / "runtime"),
    ]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    obj, executable = args.build / "FlashPLE.o", args.build / "independent-cpu-oracle"
    subprocess.run(
        [
            "xcrun",
            "clang++",
            *flags,
            "-c",
            str(ROOT / "runtime/flash/FlashPLE.cpp"),
            "-o",
            str(obj),
        ],
        check=True,
    )
    subprocess.run(
        [
            "xcrun",
            "clang++",
            *flags,
            str(ROOT / "dev/tests/flash/flash_ple_ssd_independent_cpu.cpp"),
            str(obj),
            "-Wl,-dead_strip",
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = json.loads(
        subprocess.check_output([str(executable), str(fixture)], text=True)
    )
    window_inventory = []
    disk_logical = 0
    non_disk_logical = 0
    for shard in manifest["shards"]:
        records = sorted(
            [
                (name, record)
                for name, record in manifest["tensors"].items()
                if record["shard"] == shard["path"]
            ],
            key=lambda pair: pair[1]["offset"],
        )
        windows = []
        for name, record in records:
            disk = name.startswith(PLE + "ngram_embedding.shards.") and name.rsplit(
                ".", 1
            )[-1] in {"weight", "scales", "biases"}
            begin, end = (
                record["offset"],
                (record["offset"] + record["length"] + 16383) // 16384 * 16384,
            )
            if disk:
                disk_logical += record["length"]
            else:
                non_disk_logical += record["length"]
                if windows and windows[-1]["end"] == begin:
                    windows[-1]["end"] = end
                    windows[-1]["tensor_names"].append(name)
                else:
                    windows.append({"begin": begin, "end": end, "tensor_names": [name]})
        window_inventory.append(
            {
                "source": shard["path"],
                "file_bytes": shard["bytes"],
                "native_windows": windows,
            }
        )
    assert (
        disk_logical == 32000153600
        and len([w for shard in window_inventory for w in shard["native_windows"]])
        == 28
    )
    result.update(
        {
            "schema": "flash-ple-ssd-independent-cpu-v1",
            "sanitizers": args.sanitize,
            "checkpoint_data_bytes_read": bytes_read,
            "checkpoint_reads": reads,
            "checkpoint_multipliers": multipliers,
            "checkpoint_head_sizes": sizes,
            "checkpoint_head_offsets": offsets,
            "checkpoint_shared_scale_bits": scale_bits,
            "checkpoint_ple_eos": eos,
            "generation_second_stop_is_not_ple_eos": True,
            "segment_chunk_comparisons": chunk_comparisons,
            "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
            "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "disk_only_logical_bytes": disk_logical,
            "non_ple_logical_bytes": non_disk_logical,
            "native_window_ownership": window_inventory,
            "mapping_lifetime_note": "Static tensor membership checked; native Metal view retaining Mapping ownership requires the Root backend oracle",
            "limitations": [
                "No Metal kernel execution",
                "No SSD store I/O exercised by this executable",
                "No model output, state restoration, cancellation or HTTP service qualification",
            ],
        }
    )
    report = args.build / "qualification.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "checks": result["checks"],
                "compared_source_ids": result["compared_source_ids"],
                "report": str(report),
            }
        )
    )


if __name__ == "__main__":
    main()
