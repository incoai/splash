"""Compare two logits dumps by their next-token distributions.

Run as ``python -m dev.benchmarks.logit_compare REFERENCE CANDIDATE``. Each is
a directory that backend-benchmark's logits scenario writes
(``--scenario logits --logits DIR``, the candidate with
``--logits-from REFERENCE`` so that it scores the reference's prefixes), or
that another engine writes in its layout:

- sequences.txt: one token sequence per line, its ids separated by spaces;
- records.txt: one scored prefix per line, the index of its sequence and the
  prefix's length in tokens;
- logits.f32: per record, in record order, the logits after the prefix over
  the whole vocabulary as little-endian fp32 values (numpy's tofile), so
  that another engine's dump of its logits after the same prefixes compares
  the same way.

Records are matched by the tokens of their prefixes, so the two dumps may hold
them in any order; the candidate must hold every prefix of the reference. Per
record it computes the KL divergence of the candidate's softmax from the
reference's in fp64, whether their most likely tokens agree, and the change in
the negative log likelihood of the token the sequence goes on with. It prints
the top-1 agreement, the mean, median, p99 and largest KL and the mean NLL
change, and fails (exit status 1) past the bounds that --min-top1,
--max-mean-kl and --max-p99-kl give.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_dump(directory: Path) -> tuple[dict, np.ndarray]:
    """A dump's records by the tokens of their prefix, each the row of its
    logits and the token its sequence goes on with, and the logits rows."""
    sequences = [
        tuple(int(token) for token in line.split())
        for line in (directory / "sequences.txt").read_text().splitlines()
    ]
    records = [
        tuple(int(value) for value in line.split())
        for line in (directory / "records.txt").read_text().splitlines()
        if line.strip()
    ]
    logits = np.fromfile(directory / "logits.f32", dtype="<f4")
    if not records or logits.size % len(records):
        raise ValueError(f"{directory}: logits.f32 does not hold one row per record")
    rows = logits.reshape(len(records), -1)
    keyed = {}
    for row, (sequence, prefix) in enumerate(records):
        tokens = sequences[sequence]
        if not 0 < prefix < len(tokens):
            raise ValueError(f"{directory}: record {row} is no proper prefix")
        keyed[tokens[:prefix]] = (row, tokens[prefix])
    return keyed, rows


def log_softmax(row: np.ndarray) -> np.ndarray:
    values = row.astype(np.float64)
    shifted = values - values.max()
    return shifted - np.log(np.exp(shifted).sum())


def compare(reference: Path, candidate: Path) -> dict:
    """The per-record comparison of candidate against reference, summarized."""
    expected, expected_rows = read_dump(reference)
    got, got_rows = read_dump(candidate)
    if expected_rows.shape[1] != got_rows.shape[1]:
        raise ValueError("the dumps have different vocabularies")
    if missing := len(expected.keys() - got.keys()):
        raise ValueError(f"the candidate lacks {missing} of the reference's prefixes")
    kl, agree, nll = [], 0, []
    for prefix, (row, following) in expected.items():
        p, q = log_softmax(expected_rows[row]), log_softmax(got_rows[got[prefix][0]])
        kl.append(float(np.exp(p) @ (p - q)))
        agree += int(expected_rows[row].argmax() == got_rows[got[prefix][0]].argmax())
        nll.append(float(p[following] - q[following]))
    kl = np.array(kl)
    return {
        "records": len(kl),
        "top1_agreement": agree / len(kl),
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_max": float(kl.max()),
        "nll_change_mean": float(np.mean(nll)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--min-top1", type=float, help="least top-1 agreement")
    parser.add_argument("--max-mean-kl", type=float, help="largest mean KL")
    parser.add_argument("--max-p99-kl", type=float, help="largest p99 KL")
    parser.add_argument("--json", type=Path, help="also write the summary here")
    args = parser.parse_args(argv)
    summary = compare(args.reference, args.candidate)
    failures = [
        f"{name} {summary[key]:.4g} {relation} {bound:.4g}"
        for name, key, bound, relation, passes in (
            (
                "top-1 agreement",
                "top1_agreement",
                args.min_top1,
                "<",
                lambda v, b: v >= b,
            ),
            ("mean KL", "kl_mean", args.max_mean_kl, ">", lambda v, b: v <= b),
            ("p99 KL", "kl_p99", args.max_p99_kl, ">", lambda v, b: v <= b),
        )
        if bound is not None and not passes(summary[key], bound)
    ]
    summary["failures"] = failures
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{summary['records']} prefixes: top-1 agreement {summary['top1_agreement']:.4f}, "
        f"KL mean {summary['kl_mean']:.3e} median {summary['kl_median']:.3e} "
        f"p99 {summary['kl_p99']:.3e} max {summary['kl_max']:.3e}, "
        f"next-token NLL change {summary['nll_change_mean']:+.2e}"
    )
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
