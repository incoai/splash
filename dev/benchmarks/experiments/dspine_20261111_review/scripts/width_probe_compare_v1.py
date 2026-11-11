"""Compare captured target rows only while both runs have the same prefix."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(directory):
    result = json.loads((directory / "result.json").read_text())
    assert result["all_captured_argmax_mappings_match"] is True
    assert result["cache_status"] == "miss" and result["matched_tokens"] == 0
    assert result["temperature"] == 0 and result["ane_ffn_share"] == 0
    captures = {}
    binary = directory / "logits.f32"
    for line in (directory / "rows.jsonl").read_text().splitlines():
        row = json.loads(line)
        index = row["prediction_index"]
        if index < len(result["output_tokens"]):
            assert row["top8"][0][0] == result["output_tokens"][index], (
                directory,
                index,
            )
        if row["binary_offset"] is None:
            continue
        assert index not in captures
        values = np.fromfile(
            binary, dtype="<f4", count=row["vocabulary"], offset=row["binary_offset"]
        )
        assert values.size == row["vocabulary"] and np.isfinite(values).all()
        assert int(np.argmax(values)) == row["top8"][0][0], (directory, index)
        captures[index] = values
    return (
        result,
        captures,
        {
            name: sha(directory / name)
            for name in ["result.json", "rows.jsonl", "logits.f32"]
        },
    )


def metrics(a, b):
    bitwise_equal = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    a, b = a.astype(np.float64), b.astype(np.float64)
    delta = b - a
    a_shift, b_shift = a - a.max(), b - b.max()
    loga = a_shift - np.log(np.exp(a_shift).sum())
    logb = b_shift - np.log(np.exp(b_shift).sum())
    pa, pb = np.exp(loga), np.exp(logb)
    topa, topb = np.argsort(-a, kind="stable")[:2], np.argsort(-b, kind="stable")[:2]
    centered = np.sqrt(np.mean((delta - delta.mean()) ** 2))
    return {
        "bitwise_equal": bitwise_equal,
        "raw_rms_delta": float(np.sqrt(np.mean(delta**2))),
        "mean_delta": float(delta.mean()),
        "maximum_absolute_delta": float(np.abs(delta).max()),
        "centered_relative_rms_delta": float(centered / np.std(a))
        if np.std(a)
        else None,
        "temperature1_full_vocabulary_total_variation": float(
            np.abs(pa - pb).sum() / 2
        ),
        "temperature1_full_vocabulary_kl_a_to_b": float(np.sum(pa * (loga - logb))),
        "a_top2": [[int(i), float(a[i])] for i in topa],
        "b_top2": [[int(i), float(b[i])] for i in topb],
        "a_top2_margin": float(a[topa[0]] - a[topa[1]]),
        "b_top2_margin": float(b[topb[0]] - b[topb[1]]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, ac, ah = read(args.control)
    b, bc, bh = read(args.candidate)
    shared = [
        "case",
        "model_root",
        "loaded_model_layout_sha256",
        "device",
        "prompt_tokens",
        "output_cap",
        "context_limit",
        "temperature",
        "ane_ffn_share",
        "cache_status",
        "matched_tokens",
    ]
    assert all(a[key] == b[key] for key in shared)
    outputs_a, outputs_b = a["output_tokens"], b["output_tokens"]
    first = next(
        (i for i, (x, y) in enumerate(zip(outputs_a, outputs_b)) if x != y), None
    )
    if first is None and len(outputs_a) != len(outputs_b):
        first = min(len(outputs_a), len(outputs_b))
    rows = {}
    excluded = []
    for index in sorted(ac.keys() & bc.keys()):
        common = outputs_a[:index] == outputs_b[:index]
        if not common or index >= min(len(outputs_a), len(outputs_b)):
            excluded.append(index)
            continue
        rows[str(index)] = metrics(ac[index], bc[index])
    record = {
        "case": a["case"],
        "control_rows": a["verify_rows"],
        "candidate_rows": b["verify_rows"],
        "control_build_id": a["build_id"],
        "candidate_build_id": b["build_id"],
        "same_build_id": a["build_id"] == b["build_id"],
        "sources": {"control": ah, "candidate": bh},
        "prompt_tokens": a["prompt_tokens"],
        "historical_expected_first_difference": 42,
        "actual_first_difference": first,
        "complete_output_arrays_equal": outputs_a == outputs_b,
        "output_lengths": [len(outputs_a), len(outputs_b)],
        "first_difference_capture_available": first is not None and str(first) in rows,
        "rows_with_identical_conditional_prefix": rows,
        "excluded_captured_indices": excluded,
        "scope": "One selected historical prompt with its original output cap and cold serial decode. "
        "Each captured argmax is validated against returned tokens. Metrics compare only "
        "rows with identical prior output tokens. No tolerance or distribution-equivalence "
        "verdict is applied, and host-copy timing is unused.",
    }
    with args.output.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in [
                    "case",
                    "control_rows",
                    "candidate_rows",
                    "same_build_id",
                    "actual_first_difference",
                    "complete_output_arrays_equal",
                    "first_difference_capture_available",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
