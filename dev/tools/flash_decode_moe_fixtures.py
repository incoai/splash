"""Extract original-source decode expert IDs for private CPU/GPU qualification.

This tool uses only Python's standard library and never loads a model, creates a
Metal backend, or submits GPU work. Hidden inputs are not captured by this file:
the consuming oracle supplies synthetic hidden inputs. A 16-row fixture uses
the 15 captured decode rows plus an explicitly recorded repeat of the last row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

SOURCE_IDENTITY = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"
SCHEMA = "splash-flash-decode-moe-fixtures-v1"
LAYERS = 48
EXPERTS = 512
TOP_K = 10
CAPTURED_DECODE_ROWS = 15
MAX_PREFILL_ROWS = 2048
MAX_LINE_BYTES = 16 * 1024 * 1024
FIXTURE_LAYERS = (0, 24, 47)
FIXTURE_ROWS = (2, 4, 8, 16)
DEFAULT_CAPTURE = "build/release/flash/expert-calibration-sample0.json.expert-ids.jsonl"
DEFAULT_OUTPUT = "build/flash-moe-decode-f32"


class FixtureError(ValueError):
    pass


def parse_record(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise FixtureError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise FixtureError(f"non-finite JSON value: {value}")

    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        record = json.loads(text, object_pairs_hook=unique, parse_constant=constant)
    except (ValueError, UnicodeError, TypeError, RecursionError) as error:
        raise FixtureError(f"invalid capture JSON: {error}") from error
    validate_record(record)
    return record


def validate_record(record):
    if not isinstance(record, dict) or set(record) != {
        "source_identity",
        "phase",
        "rows",
        "layer_ids",
    }:
        raise FixtureError(
            "capture must contain exactly source_identity, phase, rows and layer_ids"
        )
    if record["source_identity"] != SOURCE_IDENTITY:
        raise FixtureError(
            "capture source identity differs from the original Flash checkpoint"
        )
    phase, rows = record["phase"], record["rows"]
    if phase not in ("prefill", "decode"):
        raise FixtureError("capture phase must be prefill or decode")
    if type(rows) is not int or not 1 <= rows <= MAX_PREFILL_ROWS:
        raise FixtureError("capture rows must be a non-boolean integer in [1,2048]")
    if phase == "decode" and rows != 1:
        raise FixtureError("decode capture must contain exactly one real row")
    layers = record["layer_ids"]
    if not isinstance(layers, list) or len(layers) != LAYERS:
        raise FixtureError("capture must contain exactly 48 layers")
    for layer, ids in enumerate(layers):
        if not isinstance(ids, list) or len(ids) != rows * TOP_K:
            raise FixtureError(f"layer {layer} must contain exactly rows*10 IDs")
        for row in range(rows):
            selected = ids[row * TOP_K : (row + 1) * TOP_K]
            if any(
                type(expert) is not int or not 0 <= expert < EXPERTS
                for expert in selected
            ):
                raise FixtureError(
                    f"layer {layer} row {row} IDs must be non-boolean integers in [0,511]"
                )
            if len(set(selected)) != TOP_K:
                raise FixtureError(
                    f"layer {layer} row {row} must select ten distinct experts"
                )


def snapshot(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def read_capture(filename):
    path = Path(filename).resolve()
    try:
        before = snapshot(path)
        digest = hashlib.sha256()
        decode_records, lines = [], []
        record_counts = {"prefill": 0, "decode": 0}
        row_counts = {"prefill": 0, "decode": 0}
        byte_count = 0
        with path.open("rb") as stream:
            line_number = 0
            while raw := stream.readline(MAX_LINE_BYTES + 1):
                line_number += 1
                if len(raw) > MAX_LINE_BYTES:
                    raise FixtureError(f"{path}:{line_number}: line exceeds 16MiB")
                digest.update(raw)
                byte_count += len(raw)
                try:
                    record = parse_record(raw)
                except FixtureError as error:
                    raise FixtureError(f"{path}:{line_number}: {error}") from error
                phase = record["phase"]
                record_counts[phase] += 1
                row_counts[phase] += record["rows"]
                if phase == "decode":
                    decode_records.append(record["layer_ids"])
                    lines.append(line_number)
        if before != snapshot(path) or byte_count != before[2]:
            raise FixtureError("capture changed while reading")
    except OSError as error:
        raise FixtureError(f"could not read capture {path}: {error}") from error
    if len(decode_records) != CAPTURED_DECODE_ROWS:
        raise FixtureError(
            f"expected 15 original captured decode rows, got {len(decode_records)}"
        )
    return (
        decode_records,
        lines,
        {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "bytes": byte_count,
            "validated_records": sum(record_counts.values()),
            "records_by_phase": record_counts,
            "rows_by_phase": row_counts,
            "decode_record_line_numbers": lines,
            "validation": "Every record is strictly validated before excluding prefill; source file unchanged during reading",
        },
    )


def bucket_stats(ids):
    if not ids or len(ids) % TOP_K:
        raise FixtureError("bucket statistics require complete top10 rows")
    counts = [0] * EXPERTS
    for start in range(0, len(ids), TOP_K):
        selected = ids[start : start + TOP_K]
        if (
            any(
                type(expert) is not int or not 0 <= expert < EXPERTS
                for expert in selected
            )
            or len(set(selected)) != TOP_K
        ):
            raise FixtureError(
                "bucket statistics contain invalid or duplicate expert IDs"
            )
        for expert in selected:
            counts[expert] += 1
    active = sum(count != 0 for count in counts)
    return {
        "rows": len(ids) // TOP_K,
        "routed_assignments": len(ids),
        "active_experts": active,
        "assignments_reusing_an_already_selected_expert": len(ids) - active,
        "experts_selected_in_multiple_rows": sum(count > 1 for count in counts),
        "maximum_rows_reusing_one_expert": max(counts),
        "expert_bucket_counts": counts,
        "count_unit": "routed expert assignment; distinct top10 IDs make each count equal the number of rows selecting that expert",
    }


def fixture_bytes(records, layer, rows):
    if (
        type(layer) is not int
        or layer not in FIXTURE_LAYERS
        or type(rows) is not int
        or rows not in FIXTURE_ROWS
    ):
        raise FixtureError("fixture geometry must be layer0/24/47 and rows2/4/8/16")
    if len(records) != CAPTURED_DECODE_ROWS:
        raise FixtureError("fixture requires exactly 15 real captured decode rows")
    real = min(rows, len(records))
    indices = list(range(real))
    if rows == 16:
        indices.append(len(records) - 1)
    ids = [expert for index in indices for expert in records[index][layer]]
    stats = bucket_stats(ids)
    payload = struct.pack("<" + "q" * len(ids), *ids)
    return payload, indices, stats


def atomic_write(path, payload):
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_fixtures(capture, output_dir):
    records, lines, provenance = read_capture(capture)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for layer in FIXTURE_LAYERS:
        for rows in FIXTURE_ROWS:
            payload, indices, stats = fixture_bytes(records, layer, rows)
            path = output / f"captured-layer{layer}-rows{rows}-ids.i64"
            atomic_write(path, payload)
            fixtures.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "dtype": "little-endian signed I64",
                    "shape": [rows, TOP_K],
                    "layer": layer,
                    "real_captured_rows": min(rows, CAPTURED_DECODE_ROWS),
                    "synthetic_repeated_rows": rows - min(rows, CAPTURED_DECODE_ROWS),
                    "source_decode_row_indices_zero_based": indices,
                    "source_record_line_numbers_one_based": [
                        lines[index] for index in indices
                    ],
                    "synthetic_repeat_policy": "Append a copy of captured decode row14 as fixture row15"
                    if rows == 16
                    else "none",
                    "hidden_inputs": "synthetic, supplied by the consuming oracle; not captured or written by this tool",
                    "bucket_statistics": stats,
                }
            )
    report = {
        "schema": SCHEMA,
        "pass": True,
        "gpu_work": False,
        "source_identity": SOURCE_IDENTITY,
        "layers": LAYERS,
        "experts_per_layer": EXPERTS,
        "top_k": TOP_K,
        "real_captured_decode_rows": len(records),
        "capture": provenance,
        "hidden_input_policy": "synthetic hidden inputs will be supplied by the consuming oracle; capture contains expert IDs only",
        "fixture_selection_policy": "First N decode rows in capture order; N=16 uses all 15 real captured rows and repeats the final row once",
        "capture_decode_bucket_statistics": [
            {
                "layer": layer,
                **bucket_stats(
                    [expert for record in records for expert in record[layer]]
                ),
            }
            for layer in range(LAYERS)
        ],
        "fixtures": fixtures,
    }
    report_path = output / "decode-fixtures.json"
    atomic_write(
        report_path, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    )
    return report_path, report


def self_test():
    good = {
        "source_identity": SOURCE_IDENTITY,
        "phase": "decode",
        "rows": 1,
        "layer_ids": [list(range(TOP_K)) for _ in range(LAYERS)],
    }
    parse_record(json.dumps(good))
    checks = 1

    def rejects(value):
        nonlocal checks
        try:
            parse_record(
                value if isinstance(value, (str, bytes)) else json.dumps(value)
            )
        except FixtureError:
            checks += 1
            return
        raise AssertionError("invalid fixture record was accepted")

    for field, value in (
        ("source_identity", "0" * 64),
        ("phase", "verify"),
        ("phase", True),
        ("rows", True),
        ("rows", 1.0),
        ("rows", 0),
        ("rows", 2),
        ("rows", 2049),
        ("layer_ids", good["layer_ids"][:-1]),
        ("layer_ids", None),
    ):
        bad = copy.deepcopy(good)
        bad[field] = value
        rejects(bad)
    for bad_id in (-1, 512, 2**63, True, 1.0, "1", None):
        bad = copy.deepcopy(good)
        bad["layer_ids"][24][0] = bad_id
        rejects(bad)
    for ids in ([0] * TOP_K, list(range(TOP_K - 1)), list(range(TOP_K + 1))):
        bad = copy.deepcopy(good)
        bad["layer_ids"][47] = ids
        rejects(bad)
    for invalid in (
        '{"phase":"decode","phase":"decode"}',
        '{"rows":NaN}',
        b"\xff",
        "\n",
        "[]",
    ):
        rejects(invalid)
    bad = copy.deepcopy(good)
    bad["extra"] = 1
    rejects(bad)
    records = [
        [[(row + expert) % EXPERTS for expert in range(TOP_K)] for _ in range(LAYERS)]
        for row in range(CAPTURED_DECODE_ROWS)
    ]
    for layer in FIXTURE_LAYERS:
        for rows in FIXTURE_ROWS:
            payload, indices, stats = fixture_bytes(records, layer, rows)
            ids = struct.unpack("<" + "q" * (rows * TOP_K), payload)
            assert len(payload) == rows * TOP_K * 8
            assert list(ids[:TOP_K]) == records[0][layer]
            assert sum(stats["expert_bucket_counts"]) == rows * TOP_K
            assert len(indices) == rows
            if rows == 16:
                assert indices == list(range(15)) + [14]
                assert ids[-TOP_K:] == ids[-2 * TOP_K : -TOP_K]
            checks += 1
    for layer, rows in ((True, 2), (1, 2), (0, True), (0, 15), (0, 17)):
        try:
            fixture_bytes(records, layer, rows)
        except FixtureError:
            checks += 1
        else:
            raise AssertionError("invalid fixture geometry was accepted")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "capture.jsonl"
        path.write_text((json.dumps(good) + "\n") * CAPTURED_DECODE_ROWS)
        captured, lines, provenance = read_capture(path)
        assert len(captured) == 15 and lines == list(range(1, 16))
        assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        checks += 1
        for count in (0, 14, 16):
            path.write_text((json.dumps(good) + "\n") * count)
            try:
                read_capture(path)
            except FixtureError:
                checks += 1
            else:
                raise AssertionError("incorrect number of captured rows was accepted")
    return {"pass": True, "gpu_work": False, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", default=DEFAULT_CAPTURE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            result = self_test()
        else:
            path, report = create_fixtures(args.capture, args.output_dir)
            result = {
                "pass": True,
                "gpu_work": False,
                "report": str(path),
                "capture_sha256": report["capture"]["sha256"],
                "real_captured_decode_rows": report["real_captured_decode_rows"],
                "fixtures": len(report["fixtures"]),
            }
        print(json.dumps(result, allow_nan=False))
    except (FixtureError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
