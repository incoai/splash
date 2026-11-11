"""The one speed rule of the release check's baseline comparisons.

A comparison runs four rounds on one idle machine: R1 baseline, R2 candidate,
R3 candidate, R4 baseline. Each round contributes the median of its samples of
a metric where lower is better, such as GPU milliseconds. With b = (R1+R4)/2
and c = (R2+R3)/2, the run's own spread is noise = max(|R1-R4|/b, |R2-R3|/c)
and the candidate's regression is c/b - 1 (positive: slower).

A regression passes up to max(2%, 2 x noise): the M3's measured ABBA spread was
at most 1.4%, so the floor passes real noise and still catches a 3% regression,
and scaling with the measured spread avoids false alarms on a warm machine
without widening the bound silently. A spread above 5% measures nothing: the
comparison is inconclusive, which fails ("rerun idle"). A noisy run is never a
pass.
"""

from __future__ import annotations

import math
import statistics

REGRESSION_FLOOR = 0.02
NOISE_LIMIT = 0.05

PASS = "pass"
FAIL = "fail"
INCONCLUSIVE = "inconclusive"


def positive(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def compare(r1: float, r2: float, r3: float, r4: float) -> dict:
    """The verdict on the round medians R1 baseline, R2 candidate, R3
    candidate and R4 baseline of a lower-is-better metric."""
    rounds = (r1, r2, r3, r4)
    if not all(positive(value) for value in rounds):
        raise ValueError(f"ABBA round medians must be positive numbers: {rounds!r}")
    baseline = (r1 + r4) / 2
    candidate = (r2 + r3) / 2
    noise = max(abs(r1 - r4) / baseline, abs(r2 - r3) / candidate)
    regression = candidate / baseline - 1
    bound = max(REGRESSION_FLOOR, 2 * noise)
    if noise > NOISE_LIMIT:
        verdict = INCONCLUSIVE
    elif regression <= bound:
        verdict = PASS
    else:
        verdict = FAIL
    return {
        "rounds": list(rounds),
        "baseline": baseline,
        "candidate": candidate,
        "noise": noise,
        "regression": regression,
        "bound": bound,
        "verdict": verdict,
        "pass": verdict == PASS,
    }


def compare_samples(rounds) -> dict:
    """compare() of the medians of four rounds' samples, in ABBA order. Each
    sample must be a positive number: with a NaN among them, the median is
    NaN or an arbitrary sample, by the position of the NaN."""
    rounds = [list(samples) for samples in rounds]
    if len(rounds) != 4 or not all(rounds):
        raise ValueError("an ABBA comparison needs samples from four rounds")
    if not all(positive(sample) for samples in rounds for sample in samples):
        raise ValueError(f"ABBA samples must be positive numbers: {rounds!r}")
    return compare(*(statistics.median(samples) for samples in rounds))


def describe(result: dict) -> str:
    """One line for a log: the verdict, regression, bound and noise."""
    note = " (rerun idle)" if result["verdict"] == INCONCLUSIVE else ""
    return (
        f"{result['verdict'].upper()}{note}: baseline {result['baseline']:.3f}, "
        f"candidate {result['candidate']:.3f}, regression "
        f"{result['regression']:+.2%} (bound {result['bound']:.2%}), "
        f"noise {result['noise']:.2%}"
    )
