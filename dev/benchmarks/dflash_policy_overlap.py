"""Compare proposal policies on identical, explicitly bound target contexts.

Input p is the full deployed target distribution AFTER its request policy.
No model loading, runtime sampling changes, or top-16 target renormalization.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def context_digest(prompt, output, prefix):
    return hashlib.sha256(
        json.dumps([prompt, output, prefix], separators=(",", ":")).encode()
    ).hexdigest()


def penalize(
    score, token, prompt, output_counts, repetition=1.0, presence=0.0, frequency=0.0
):
    # Matches Sampling::penalize_logit: prompt only activates repetition;
    # presence/frequency count generated output (including selected prefix).
    count = output_counts[token]
    if token in prompt or count:
        score = score / repetition if score > 0 else score * repetition
        if count:
            score -= presence + frequency * count
    return score


def proposal(ids, scores, temperature, vocabulary):
    q = [0.0] * vocabulary
    if temperature == 0:
        best = min(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
        q[ids[best]] = 1.0
    else:
        maximum = max(scores)
        weights = [math.exp((score - maximum) / temperature) for score in scores]
        total = sum(weights)
        for token, value in zip(ids, weights, strict=True):
            q[token] = value / total
    return q


def distribution(values):
    if (
        not isinstance(values, list)
        or not values
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values
        )
    ):
        raise ValueError("invalid full probability distribution")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("target distribution must include its full tail mass")
    return values


def rejection_reference(p, q):
    """Exact one-step accept/correct law, with q being the actual draw law."""
    distribution(p)
    distribution(q)
    if len(p) != len(q):
        raise ValueError("p and q vocabulary differ")
    accepted = [min(a, b) for a, b in zip(p, q, strict=True)]
    residual = [max(a - b, 0.0) for a, b in zip(p, q, strict=True)]
    rejected = sum(residual)
    correction = [x / rejected for x in residual] if rejected else [0.0] * len(p)
    resulting = [a + rejected * c for a, c in zip(accepted, correction, strict=True)]
    return {"acceptance": sum(accepted), "correction": correction, "result": resulting}


def token_list(values, vocabulary):
    if not isinstance(values, list) or any(
        type(x) is not int or not 0 <= x < vocabulary for x in values
    ):
        raise ValueError("token list is outside the target vocabulary")
    return values


def analyze_row(row):
    p = distribution(row["target_probabilities"])
    vocabulary = len(p)
    prompt = token_list(row["prompt_tokens"], vocabulary)
    output = token_list(row["output_tokens"], vocabulary)
    prefix = token_list(row["within_block_prefix"], vocabulary)
    digest = context_digest(prompt, output, prefix)
    if (
        row["target_context_sha256"] != digest
        or row["proposal_context_sha256"] != digest
    ):
        raise ValueError("target and proposal must score the identical correct prefix")
    ids = token_list(row["candidate_ids"], vocabulary)
    scores = row["candidate_scores"]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or len(scores) != len(ids)
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in scores)
    ):
        raise ValueError("candidate ids/scores must be unique finite aligned rows")
    temperature = row["proposal_temperature"]
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("proposal temperature must be finite and nonnegative")
    policy = row["penalties"]
    if (
        set(policy) != {"repetition", "presence", "frequency"}
        or any(
            type(x) not in (int, float) or not math.isfinite(x) for x in policy.values()
        )
        or policy["repetition"] <= 0
    ):
        raise ValueError("invalid proposal penalties")
    support = sum(p[token] for token in ids)
    variants = {}
    for name, generated in (
        ("existing", None),
        ("history", output),
        ("history_and_prefix", output + prefix),
    ):
        adjusted = (
            scores
            if generated is None
            else [
                penalize(s, token, set(prompt), Counter(generated), **policy)
                for token, s in zip(ids, scores, strict=True)
            ]
        )
        q = proposal(ids, adjusted, temperature, vocabulary)
        variants[name] = {
            "overlap": sum(min(a, b) for a, b in zip(p, q, strict=True)),
            "proposal_entropy": -sum(x * math.log(x) for x in q if x),
        }
    return {
        "context_sha256": digest,
        "position_in_block": len(prefix),
        "target_mass_in_support": support,
        "target_mass_outside_support": 1 - support,
        "variants": variants,
    }


def analyze(document):
    if document.get("schema") != "dflash-policy-overlap-v1":
        raise ValueError("unsupported policy-overlap schema")
    if document.get("distribution_source") not in ("native_capture", "test_fixture"):
        raise ValueError("declare native_capture or test_fixture provenance")
    for field in ("target_identity", "draft_identity", "tokenizer_identity"):
        if not isinstance(document.get(field), str) or not document[field]:
            raise ValueError(f"missing {field}")
    records = document["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("provide at least one matched-context row")
    rows = [analyze_row(row) for row in records]
    return {
        "schema": "dflash-policy-overlap-report-v1",
        "distribution_source": document["distribution_source"],
        "identities": {
            k: document[k]
            for k in ("target_identity", "draft_identity", "tokenizer_identity")
        },
        "scope": "fixed support, correct-context one-step counterfactual; not rollout speed or quality",
        "rows": rows,
        "mean_overlap": {
            name: sum(row["variants"][name]["overlap"] for row in rows) / len(rows)
            for name in ("existing", "history", "history_and_prefix")
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(json.loads(args.capture.read_text()))
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
