"""Frozen independent-seed distribution regression screen; not an equivalence proof."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path, help="Extracted evidence root")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
r = args.root
protocol = json.loads((r / "notes/sampling-protocol.json").read_text())
nomination = json.loads((r / "notes/sampling-nominations.json").read_text())


def load(arm):
    cases = defaultdict(dict)
    sources = []
    for ident in arm["jobs"]:
        path = r / f"results/{ident}.stdout"
        receipt = json.loads((r / f"results/{ident}.receipt.json").read_text())
        assert receipt["exit_code"] == 0 and receipt["state"] == "complete"
        d = json.loads(path.read_text())
        assert d["study_temperature"] == 1
        for m in d["measurements"]:
            seed = d["study_seed_base"] + m["sample"]
            name = m["scenario"]
            assert seed not in cases[name], "duplicate sample seed"
            cases[name][seed] = m["output_tokens"]
        sources.append(
            {
                "job": ident,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "build_id": d["build_id"],
                "geometry": d["geometry"],
            }
        )
    expected = set(range(arm["seed_start"], arm["seed_start"] + 1500))
    assert len(cases) == 3 and all(set(v) == expected for v in cases.values())
    return cases, sources


def feature(sequence, indices):
    return tuple(sequence[i] if i < len(sequence) else -1 for i in indices)


def test(a, b):
    na, nb = sum(a.values()), sum(b.values())
    total = a + b
    common = sorted(
        (key for key, count in total.items() if count * min(na, nb) / (na + nb) >= 5),
        key=repr,
    )
    bins = [[a[key], b[key]] for key in common]
    rare = [na - sum(v[0] for v in bins), nb - sum(v[1] for v in bins)]
    if sum(rare):
        if sum(rare) * min(na, nb) / (na + nb) >= 5 or not bins:
            bins.append(rare)
        else:
            largest = max(range(len(bins)), key=lambda i: sum(bins[i]))
            bins[largest] = [bins[largest][i] + rare[i] for i in range(2)]
    if len(bins) < 2:
        return {
            "pvalue": 1.0,
            "informative": False,
            "sample_counts": [na, nb],
            "bins": len(bins),
        }
    table = np.array(bins, dtype=np.int64).T
    result = chi2_contingency(table, correction=False)
    assert result.expected_freq.min() >= 5 - 1e-9
    return {
        "pvalue": float(result.pvalue),
        "statistic": float(result.statistic),
        "degrees_of_freedom": result.dof,
        "minimum_expected_count": float(result.expected_freq.min()),
        "bins": len(bins),
        "sample_counts": [na, nb],
        "informative": True,
        "cramers_v": float(np.sqrt(result.statistic / (na + nb))),
        "empirical_pooled_total_variation": float(
            np.abs(table[0] / na - table[1] / nb).sum() / 2
        ),
        "note": "Empirical distance includes finite-sample noise and pooling can hide rare-category changes.",
    }


baseline, baseline_sources = load(nomination["control"])
features = [(i,) for i in protocol["positions"]] + [
    (i, i + 1) for i in protocol["adjacent_pair_starts"]
]
family_size = protocol["family_size_reserved"]
results = []
candidate_sources = {}
for arm in nomination["candidates"]:
    values, sources = load(arm)
    candidate_sources[arm["name"]] = sources
    assert values.keys() == baseline.keys()
    for name in sorted(baseline):
        for indices in features:
            a = Counter(feature(v, indices) for v in baseline[name].values())
            b = Counter(feature(v, indices) for v in values[name].values())
            results.append(
                dict(
                    candidate=arm["name"],
                    context=name,
                    positions_zero_based=list(indices),
                    **test(a, b),
                )
            )
assert len(results) <= family_size
ordered = sorted(range(len(results)), key=lambda i: results[i]["pvalue"])
previous = 0.0
for rank, index in enumerate(ordered):
    adjusted = min(1.0, max(previous, (family_size - rank) * results[index]["pvalue"]))
    results[index]["holm_adjusted_pvalue"] = adjusted
    results[index]["reject_at_family_alpha_0_05"] = adjusted < 0.05
    previous = adjusted
record = {
    "protocol": protocol,
    "nomination": nomination,
    "baseline_sources": baseline_sources,
    "candidate_sources": candidate_sources,
    "family_size_reserved": family_size,
    "tests_observed": len(results),
    "rejections": sum(v["reject_at_family_alpha_0_05"] for v in results),
    "results": results,
    "interpretation": "A finite regression screen. No rejection is not proof of identical distributions, especially for rare outputs. Retain all comparisons and report uninformative tests.",
}
args.output.write_text(json.dumps(record, indent=2) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in record.items()
            if k in ["family_size_reserved", "tests_observed", "rejections"]
        }
    )
)
