"""Explain greedy DFlash rejections from decode-profile --draft-dump captures.

Counterfactuals reuse the recorded continuation at the original cycle starts.
They estimate local acceptance, not a new trajectory or serving throughput.
No model dependencies are needed: python -m dev.benchmarks.draft_replay FILE.
"""

import argparse
import gzip
import json
import math
import struct
from pathlib import Path


def probabilities(scores):
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    mass = sum(weights)
    return [weight / mass for weight in weights]


def scores_at(unary, edge, edge_scale=1.0):
    # The captured operands are fp32, and Metal rounds their sum before
    # comparing it. Python's double addition can otherwise break a GPU tie.
    return [
        struct.unpack("f", struct.pack("f", u + edge_scale * e))[0]
        for u, e in zip(unary, edge, strict=True)
    ]


def select(unary, edges, objective="greedy", edge_scale=1.0):
    """Select candidate indices with stable lowest-index ties.

    expected_length maximizes sum of prefix probabilities under the draft's
    conditional distributions; viterbi maximizes their joint probability.
    Neither objective assumes those probabilities equal target acceptance.
    """
    n, k = len(unary), len(unary[0])
    if objective not in ("greedy", "expected_length", "viterbi"):
        raise ValueError("unknown selector objective")
    if not math.isfinite(edge_scale) or edge_scale < 0:
        raise ValueError("edge scales must be finite and nonnegative")
    choices = []
    values = [0.0] * k
    for position in reversed(range(n)):
        row_choices, row_values = [], []
        for edge in edges[position]:
            scores = scores_at(unary[position], edge, edge_scale)
            if objective == "greedy":
                objectives = scores
            elif objective == "expected_length":
                q = probabilities(scores)
                objectives = [p * (1 + v) for p, v in zip(q, values, strict=True)]
            else:
                maximum = max(scores)
                normalizer = maximum + math.log(
                    sum(math.exp(score - maximum) for score in scores)
                )
                objectives = [
                    score - normalizer + value
                    for score, value in zip(scores, values, strict=True)
                ]
            best = max(range(k), key=objectives.__getitem__)
            row_choices.append(best)
            row_values.append(objectives[best])
        choices.append(row_choices)
        values = row_values
    previous, path = 0, []
    for row in reversed(choices):
        previous = row[previous]
        path.append(previous)
    return path


def _integer(value, lo=0, hi=2**32 - 1):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError("invalid integer in draft capture")
    return value


def _vector(values, count, tokens=False):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("invalid draft capture shape")
    for value in values:
        if tokens:
            _integer(value)
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("nonfinite or invalid draft score")
    return values


def validate(record, n, k, width):
    _integer(record["request_id"], 1, 2**64 - 1)
    _integer(record["lane"], 0, width - 1)
    _integer(record["cycle"])
    _integer(record["position"], 0, 2**64 - 1)
    verified = _integer(record["verify_rows"], 1, n + 1)
    accepted = _integer(record["accepted"], 0, verified - 1)
    retained = _integer(record["retained"], 1, accepted + 1)
    proposed = _vector(record["proposed"], n, True)
    candidates = _vector(record["candidates"], n * k, True)
    unary = _vector(record["unary"], n * k)
    confidence = _vector(record["confidence"], n)
    target = _vector(record["target_tokens"], retained, True)
    output = record["output_tokens"]
    _vector(output, len(output), True)
    if len(output) not in (retained, retained + 1):
        raise ValueError("invalid emitted token count")
    if output[1:retained] != target[: retained - 1]:
        raise ValueError("emitted tokens disagree with retained target tokens")
    if proposed[: min(accepted, retained)] != target[: min(accepted, retained)]:
        raise ValueError("accepted proposals disagree with target")
    if (
        accepted < verified - 1
        and retained == accepted + 1
        and target[accepted] == proposed[accepted]
    ):
        raise ValueError("first rejected proposal agrees with target")
    if len(record["edges"]) != n:
        raise ValueError("invalid edge position count")
    edges = []
    for position in range(n):
        row = _vector(record["edges"][position], (1 if position == 0 else k) * k)
        edges.append([row[i : i + k] for i in range(0, len(row), k)])
    candidates = [candidates[i : i + k] for i in range(0, n * k, k)]
    unary = [unary[i : i + k] for i in range(0, n * k, k)]
    if any(len(set(row)) != k for row in candidates):
        raise ValueError("duplicate draft candidates")
    path = select(unary, edges)
    if [row[index] for row, index in zip(candidates, path, strict=True)] != proposed:
        raise ValueError("greedy replay disagrees with captured proposals")
    previous = 0
    for position, chosen in enumerate(path):
        scores = scores_at(unary[position], edges[position][previous])
        if not math.isclose(
            max(probabilities(scores)), confidence[position], abs_tol=2e-5
        ):
            raise ValueError("confidence replay disagrees with capture")
        previous = chosen
    return candidates, unary, edges


def summarize(metadata, records, edge_scales=()):
    if (
        type(metadata.get("schema")) is not int
        or metadata["schema"] != 1
        or metadata.get("sampling") is not False
    ):
        raise ValueError("only schema 1 greedy captures are supported")
    n = _integer(metadata["positions"], 1, 64)
    k = _integer(metadata["candidates"], 1, 256)
    width = _integer(metadata["width"], 1, 4)
    if not records:
        raise ValueError("empty draft capture")
    if any(not math.isfinite(scale) or scale < 0 for scale in edge_scales):
        raise ValueError("edge scales must be finite and nonnegative")
    prepared, continuations, previous, batches = [], {}, {}, {}
    for record in records:
        data = validate(record, n, k, width)
        request = record["request_id"]
        position = record["position"]
        batch = batches.setdefault(record["cycle"], set())
        if record["lane"] in batch:
            raise ValueError("duplicate batch lane")
        batch.add(record["lane"])
        if request in previous:
            before = previous[request]
            if (
                record["cycle"] != before["cycle"] + 1
                or record["lane"] != before["lane"]
                or position != before["position"] + before["retained"]
            ):
                raise ValueError("noncontiguous request capture")
        elif record["cycle"] != 0:
            raise ValueError("request capture must start at cycle zero")
        previous[request] = record
        continuation = continuations.setdefault(request, {})
        # target_tokens[-1] is the pending correction or bonus, not yet
        # necessarily emitted; it is still known at the following position.
        known = [(position + i, t) for i, t in enumerate(record["output_tokens"])]
        known += [(position + i + 1, t) for i, t in enumerate(record["target_tokens"])]
        for offset, token in known:
            if offset in continuation and continuation[offset] != token:
                raise ValueError("conflicting target continuations")
            continuation[offset] = token
        prepared.append((record, data))
    if any(lanes != set(range(width)) for lanes in batches.values()):
        raise ValueError("incomplete batch capture")

    reached, successes = [0] * n, [0] * n
    bins = [{"count": 0, "sum_confidence": 0.0, "accepted": 0} for _ in range(10)]
    misses = {"selection": 0, "candidate_recall": 0, "censored": 0}
    ranks = [0] * k
    comparisons = []
    scale_comparisons = {scale: [] for scale in edge_scales}
    for record, (candidates, unary, edges) in prepared:
        accepted, retained = record["accepted"], record["retained"]
        passed = min(accepted, retained - 1)
        rejected = (
            accepted < record["verify_rows"] - 1
            and retained == accepted + 1
            and len(record["output_tokens"]) == retained
            and record["target_tokens"][accepted] != record["proposed"][accepted]
        )
        for position in range(passed + int(rejected)):
            reached[position] += 1
            success = int(position < passed)
            successes[position] += success
            confidence = record["confidence"][position]
            bucket = bins[min(9, int(confidence * 10))]
            bucket["count"] += 1
            bucket["sum_confidence"] += confidence
            bucket["accepted"] += success
        if rejected:
            token = record["target_tokens"][accepted]
            row = candidates[accepted]
            if token in row:
                misses["selection"] += 1
                ranks[row.index(token)] += 1
            else:
                misses["candidate_recall"] += 1
        elif passed < n:
            misses["censored"] += 1

        continuation = continuations[record["request_id"]]
        offsets = [record["position"] + 1 + i for i in range(n)]
        if not all(offset in continuation for offset in offsets):
            continue
        reference = [continuation[offset] for offset in offsets]

        def prefix(path):
            for i, token in enumerate(path):
                if token != reference[i]:
                    return i
            return n

        row = {}
        for objective in ("greedy", "expected_length", "viterbi"):
            indices = select(unary, edges, objective)
            path = [candidates[i][index] for i, index in enumerate(indices)]
            row[objective] = prefix(path) + 1
        oracle = next(
            (i for i, token in enumerate(reference) if token not in candidates[i]), n
        )
        row["candidate_oracle"] = oracle + 1
        comparisons.append(row)

        for scale, observations in scale_comparisons.items():
            indices = select(unary, edges, edge_scale=scale)
            path = [candidates[i][index] for i, index in enumerate(indices)]
            observations.append((prefix(path) + 1, path != record["proposed"]))

    report = {
        "metadata": metadata,
        "cycles": len(records),
        "mean_retained_tokens": sum(r["retained"] for r in records) / len(records),
        "verified_proposals": sum(r["verify_rows"] - 1 for r in records),
        "accepted_proposals": sum(
            min(r["accepted"], r["retained"] - 1) for r in records
        ),
        "first_rejections": misses,
        "selection_miss_unary_rank_counts": ranks,
        "conditional_acceptance": [
            {
                "position": i + 1,
                "reached": reached[i],
                "accepted": successes[i],
                "rate": successes[i] / reached[i] if reached[i] else None,
            }
            for i in range(n)
        ],
        "confidence_bins": [
            {
                "count": b["count"],
                "mean_confidence": b["sum_confidence"] / b["count"]
                if b["count"]
                else None,
                "acceptance_rate": b["accepted"] / b["count"] if b["count"] else None,
            }
            for b in bins
        ],
        "counterfactual": {
            "complete_reference_cycles": len(comparisons),
            "censored_reference_cycles": len(records) - len(comparisons),
            "mean_tokens_at_original_starts": {
                objective: sum(row[objective] for row in comparisons) / len(comparisons)
                if comparisons
                else None
                for objective in (
                    "greedy",
                    "expected_length",
                    "viterbi",
                    "candidate_oracle",
                )
            },
            "limitation": "Fixed greedy continuation at original cycle starts; not a rollout or throughput estimate.",
        },
    }
    if edge_scales:
        report["edge_scale_replay"] = {
            "complete_reference_cycles": len(comparisons),
            "scales": [
                {
                    "edge_scale": scale,
                    "mean_tokens_at_original_starts": sum(value for value, _ in rows)
                    / len(rows)
                    if rows
                    else None,
                    "changed_path_cycles": sum(changed for _, changed in rows),
                }
                for scale, rows in scale_comparisons.items()
            ],
            "limitation": "Exploratory score scaling on fixed candidates and continuations; not held-out calibration, a quantized checkpoint export, or a rollout.",
        }
    return report


def read_capture(path):
    path = Path(path)
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as file:
        lines = [json.loads(line) for line in file if line.strip()]
    if any(not isinstance(line, dict) for line in lines):
        raise ValueError("draft capture records must be objects")
    if (
        len(lines) < 3
        or lines[0].get("kind") != "metadata"
        or lines[-1].get("kind") != "end"
    ):
        raise ValueError("incomplete draft capture")
    records = lines[1:-1]
    count = _integer(lines[-1].get("records"), 0, 2**64 - 1)
    if any(record.get("kind") != "cycle" for record in records) or count != len(
        records
    ):
        raise ValueError("invalid draft capture record count")
    return lines[0], records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--edge-scales",
        help="comma-separated nonnegative scales, e.g. 0,0.5,1,1.5,2; exploratory replay only",
    )
    args = parser.parse_args()
    try:
        scales = (
            tuple(float(value) for value in args.edge_scales.split(","))
            if args.edge_scales
            else ()
        )
        if len(scales) > 32 or len(set(scales)) != len(scales):
            raise ValueError("provide at most 32 distinct edge scales")
        report = summarize(*read_capture(args.capture), edge_scales=scales)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
        parser.exit(1, f"draft-replay: {error}\n")
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
