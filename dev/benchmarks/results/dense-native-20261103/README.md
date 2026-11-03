# Native dense experiment — 2026-11-03

Tracking issue: https://github.com/incoai/splash/issues/658.
Base: `031a25a6125af2d7acf7096580afe56eb33d65bc`. Both experimental branches start
independently from this main commit, without depending on the safe-M16 PR.
See the [experiment document](../../dense-verify-experiment.md) for implementation
scope, build commands and remaining work.

**No tested comparison establishes a native speedup above its practical floor.**

## Matched short requests

Six existing TranscriptPrompts, 256 output tokens each, B1 greedy, GPU-only,
INT8 KV and FP32 GDN state. Calls go directly through native Runtime. The
primary measure includes prefill and first-token work, excludes process
preparation/warmup, and is **not HTTP serving throughput**. Each family uses
one unchanged executable/library for its control and candidate. Absolute rates
across families were observed at different times and must not be ranked as a
controlled cross-family comparison.

| Comparison | Control tok/s | Candidate tok/s | Rate change | Practical floor | Stable/all groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-prefix5 | 31.342 | 31.840 | +1.590% | 8.138% | 3/4 |
| dense-adaptive | 37.564 | 37.544 | -0.052% | 2.000% | 3/3 |

`dense-prefix5` holds the selected prefix at five and changes only GDN suffix
skipping. `dense-adaptive` compares the existing full verification control
with the externally calibrated dense budget and prefix GDN together.
`dictionary` compares dictionary off/on, without the separate GPU audit in
timed trials. Only the applicable families are included in each PR's archive.

Order was A1/B1/B2/A2, uninterrupted within each group. An endpoint spread
above 5% excludes that group; it is retained. Up to five groups were allowed,
stopping at three stable groups. The floor is max(2%, twice the largest stable
endpoint spread). A positive qualification additionally requires latency
reduction above that floor, a positive lower descriptive bootstrap endpoint,
and matching complete output arrays. Raw groups and 5,000-resample group
bootstrap intervals are in `summary.json`. The interval describes these fixed
prompts; it does not estimate general workload or task quality.

Dense calibration used one fixed-row pass per cost on one separate short
input, before confirmation. It is a prototype cost table, not a validated
smooth or context-dependent model. Frequency transition is visible in raw
telemetry; broader interleaved cost calibration remains follow-up work.

All complete input/output arrays in retained short groups match: **yes**.
Acceptance, selected-row histograms where instrumented, cycle wall/GPU cost,
TTFT, preparation, warmup, complete process time and process peak RSS are in
the summary and raw observations. Process RSS and payload bytes do not measure
the complete Metal allocator peak. Raw macmon telemetry is retained; there is
no controlled energy-efficiency claim. Runs used battery power, recorded power
mode 0 as observed, and the recorded frequency/memory conditions. Early clock
drift is visible in exclusions rather than hidden by averaging it into a gain.

## Costs and acceptance

Each cell lists control / candidate means over stable-group observations.
These are the six fixed short prompts, not independent quality samples.

| Comparison | Emitted tokens/cycle | Cycle wall ms | Preparation s | Complete process s | Process peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-prefix5 | 3.491 / 3.491 | 108.843 / 107.124 | 3.178 / 3.161 | 52.618 / 51.836 | 2.294 / 2.299 |
| dense-adaptive | 4.267 / 4.267 | 110.477 / 110.481 | 3.184 / 3.177 | 44.440 / 44.454 | 2.297 / 2.298 |

## Long-context fidelity and pilots

Frozen source-reading inputs of 8,192/16,384 tokens, 64 output tokens each.
These are unreplicated probes, not qualified long-context speed results.
The adaptive arm deliberately reuses short costs to expose that limitation;
it is not a context-aware policy.

| Input | Candidate | Complete output | Unreplicated request rate change |
| --- | --- | --- | ---: |
| runtime-8192 | dense-five | match | -2.103% |
| runtime-8192 | dense-adaptive | match | -0.438% |
| runtime-16384 | dense-five | match | -0.433% |
| runtime-16384 | dense-adaptive | match | -0.130% |

Repeated known-output fidelity does not establish broad quality noninferiority.
No model arithmetic, serving defaults, model weights or on-disk formats are
changed. The experiment remains a draft regardless of timing verdict.

## Audit and reproduce the numbers

`observations.tar.gz` contains full token/cycle observations, per-process
commands/timing/RSS/power records, all exclusions, protocol, frozen input
tokens, source/artifact hashes and validation logs. Dictionary construction
and audit remain outside the timed request. Full-model coefficient audit data
is included in the dictionary PR. No model tensors are bundled.

```sh
mkdir extracted
tar -xzf observations.tar.gz -C extracted
python3 analyze.py extracted
```

The resulting `extracted/analysis.json` reproduces `summary.json`.
`SHA256SUMS.json` binds the archive, summary and analyzer. The full native CPU
suite, focused shader-validated suite, source architecture check and original
AIR isolation passed. This is focused validation, not a full `make check` or
qualification of scheduling, preemption, constraints, other devices or all
cache/state restoration paths. Historical MLX gains are not native results.

Publication removed one trailing blank line from the new GDN header to satisfy
the whitespace gate. The library/tool were rebuilt and all three recorded
artifacts remain byte-identical to the timed artifacts. The exact frozen
header, final source delta, binary hashes and rebuild log are in the archive.
