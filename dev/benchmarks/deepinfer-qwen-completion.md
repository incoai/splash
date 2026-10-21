# DeepInfer/Qwen investigation: decisions and review map

Tracking issue: [#508](https://github.com/incoai/splash/issues/508).
Current-main contribution: [#523](https://github.com/incoai/splash/pull/523).
This file records what the six-step [plan](deepinfer-qwen-plan.md) produced,
which implementation was measured, and why each conditional change was kept
or rejected. All six frozen-source local experiments now have complete measured decisions.
Fresh current-main model/HTTP validation remains explicit in its separate
manifests. It uncovered a cross-graph penalty-oracle assertion; the original
failure and a separately verified test-only correction will be reported
independently.

## Scope and source identities

Only the available Apple M5 Pro, 20 GPU cores, 48 GiB, macOS 26.6.0 was used.
Other hardware receives a [repeatable run package](qwen-maintainer-runs.md);
those jobs remain unexecuted. No shared performance default is qualified.
The user's original `pr-283-live-verify` branch at `1f825453` and stash
`440f39ba` are preserved.

| Source | Role |
| --- | --- |
| DeepInfer `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` | Original scheduling/pipeline/confidence-head comparison. |
| Splash `be3b083c5f413673d44e44881042e875e674e333` | Original source audit and native profile baseline. |
| Measurement base `bc09da81` | Frozen legacy full-engine trace and threshold experiment builds. |
| GDN implementation `46a074ee` | Pending-tape selective arithmetic experiment, subsequently corrected and validated on its archived branch. |
| Host implementation `7a843cb852d6d7f63fa3ec395dc7316c0baba9d8` | Owned initializer-list buffer binding experiment. |
| Prefill implementation `02a1162e` | Bounded row/history/width/path estimator experiment. |
| Upstream #487 `67bb94069179470125d393da5e5725e3f1501f9d` | Fixed-depth sweeps and held-out adaptive-policy evaluation. |
| Merged #487 head `507f6f3fe1e28d6090bc1a8623b955ec5dc7bb0d` | Later invalid-confidence-label filtering; different from the evaluated head. |
| Current main `1482f6bbc2abf12375332725302019594150b6cb` | Consolidated PR base and fresh functional/model validation target. |

The [current-main audit](current-main-audit.md) explains the reconciliation.
Main now owns adaptive verification, product `--verify-rows all|auto`, fixed
and per-row prefill cost, attention-pair limits, streaming, passes and newer
diagnostics. Historical runtime patches would overlap or omit that behavior.
PR #523 therefore changes benchmark tooling and documentation/evidence only;
it contains no production kernel, scheduler, protocol, model or default change.

## Decisions by workstream

| Plan step | Deliverable and measured decision | Evidence |
| --- | --- | --- |
| 1. Measure runtime | Fused native profiles and per-width operator attribution, full-engine/mask/ANE diagnostics, native-overlap HTTP workloads. Dense profiles point predominantly to linear work; MoE also has substantial expert work. Attribution is diagnostic, not a fused speedup estimate. | [Operator tables](results/qwen-runtime-measurements-m5-pro/README.md), [trace diagnostics](results/qwen-serving-trace-m5-pro/diagnostics.json), [full display timelines](results/qwen-serving-timeline-m5-pro/README.md), [host bounds](results/qwen-host-overhead-m5-pro/host-bound.json). |
| 2. Greedy MoE prefixes | Ported the old opt-in prototype, retained the eight-row storage ABI and sampled/constrained fallback, and tested quantized formats, contexts and widths. Selective expert work improves some recorded legacy workloads. Merged #487 supersedes this runtime integration; the old threshold is not added to current main. | [Initial implementation/results](deepinfer-qwen-results.md), [expanded serving matrix](results/qwen-serving-qualification-m5-pro/README.md). |
| 3. Pending-tape GDN | Reworked the stash idea for pending tapes, both state types and poisoned suffixes. Correctness passes, but native/HTTP gains do not meet promotion. Keep it out of production. | [Design/invariants](gdn-live-prefixes.md), [4,096 cycle records and oracles](results/qwen-gdn-live-prefixes-m5-pro/README.md). |
| 4. Cost-aware policy | Calibrated the old cumulative selector heuristic; evaluated existing #487 rather than building a competing adaptive scheduler. Fourteen training reports select GGUF 8/MLX 6 row controls. Eighteen held-out comparisons include two stable regressions, so that recorded revision fails its local gate. This is not a regression claim about later main. | [Calibration](confidence-calibration.md), [fixed/adaptive held-out evidence](results/qwen-policy-evaluation-m5-pro/README.md). |
| 5. Host work | A safe binding change removes one allocation per dispatch, with GPU ownership/growth tests. Awake production repeats show no qualified throughput gain. Retain the baseline; the measured opportunity does not justify a larger graph-reuse or future-step pipeline. | [Allocation probe, patch, tests and HTTP controls](results/qwen-host-overhead-m5-pro/README.md). |
| 6. Mixed prefill/decode | Implemented a bounded optional estimator and tested ragged geometry/fairness, cold/cache/continuation workloads and actual native overlap. Prior-only prediction improves, but serving does not establish a repeatable promotion benefit; MoE 2k long TTFT improves 12.7–14.5% while maximum gaps regress 41.4–45.0%, and a dense cached trial also regresses in maximum gap. Dedicated scheduler ASAN/UBSAN and TSAN pass. Keep the old estimator prototype experimental and do not replace current main's newer scheduler. | [Design](prefill-cost-model.md), [prediction and serving evidence](results/qwen-prefill-cost-m5-pro/README.md). |

## Measurement validity

Four-arm ABBA comparisons retain every arm, actual counters, output hashes,
cache reuse, memory and environment snapshots. Useful throughput gains must
exceed `max(2%,2*measured noise)`; spread above 5% is inconclusive. A regression
`pass` is not itself a speed gain. An aggregate cannot erase a per-workload
regression. Cold requests must have zero cached prompt tokens and every
measured response must produce its complete requested output budget.

The 48 GiB Mac cannot hold every pair of models safely: a paired MLX attempt
failed during its second startup, and dense paired admission serialized the
arriving prefill. The final method uses one resident server per complete ABBA
round with matched warmups/history. Mixed qualification proves native long
prefill or fully cached continuation progresses while the original native
request decodes. HTTP lifetime overlap alone is rejected.

The Mac slept during several overnight captures. A [power-log audit](results/qwen-serving-qualification-m5-pro/power-audit.json)
invalidates those entire series, even their apparently stable trials. Originals
remain marked excluded. Replacements use scoped `caffeinate -is` and reject an
elapsed UTC/awake-clock difference of one second or more. No global power
setting is changed. Power/thermal snapshots and small ABBA spread alone do not
prove absence of suspend or stable clocks.

GPU command spans, host scopes and ANE callbacks measure different intervals.
Nested phase medians are not added. Serial dispatch attribution perturbs the
graph; ANE ready/report spans include callback delay. Trace and shader
validation are disabled for performance and used separately for diagnostics.
Initial ANE availability does not prove every contended command actually uses
the split path. RAM-prefix continuation is not a memory-pressure suspension test.

Greedy hashes can differ, including between repetitions of the same fixed
policy. The dedicated exact-prompt replay of all three affected GGUF cases completes
with matching hashes across clean main and the disabled candidate. It does not
reproduce the earlier differences or establish their cause.
Timing, numerical correctness, output equality and quality are separate claims;
none of these finite greedy/probe runs establishes a sampling distribution.
Confidence labels are descriptive and correlated. Censored suffixes are
excluded rather than labelled failures. The cumulative selected-path heuristic
is different from #487's maximum-candidate feature and DeepInfer's trained head.

## Review and reproduction

Review the tool commits before the large evidence commits. The behavioral
changes are `serving_regression.py`, `qualification_matrix.py`, trace,
prediction and calibration analyzers, and their focused tests. The confidence
analysis regression test covers sequential round numbering separately from
binary/policy numbering, pooling baseline rounds 1/4 and candidate rounds 2/3.

Current runs use matching server/native protocols and product all/auto controls.
The harness confirms the effective policy in native status. Dense `auto` may
report `all`. The matrix prepares 36 explicitly unexecuted jobs across four
model/storage cells; `--execute` runs them serially. Example model revisions
are checked against actual manifests. A current ordinary binary is rejected
for historical `--trace`, because it has no trace-output hook.

To reproduce historical runtime experiments, use the archived PR branches and
the source/build/tool/model identities in their reports. Their obsolete flags,
protocols and benchmark-only patches are not current-main instructions. The
current [maintainer recipe](qwen-maintainer-runs.md) requires no private runtime
patch. Other hardware results can be returned against #508/#523 without
changing the interpretation of this Mac's measured decisions.
