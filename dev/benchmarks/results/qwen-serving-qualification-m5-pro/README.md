# Expanded greedy MoE serving qualification

**Historical data:** these measurements apply to the source/build identities in this bundle. Current main has since merged a different runtime stack; see [the source reconciliation](../../current-main-audit.md). They are not current-main qualification.


This extends PR #507's initial native and HTTP evidence on the same M5 Pro,
20 GPU cores, 48 GiB, macOS 26.6.0 machine. **The local eight-report matrix is complete: 72 four-arm trials.**
`summary.json` records all contexts, widths, gates and exclusions. The feature stays experimental
and threshold 0 (off) remains the default.

The frozen production measurement-base executable (`bc09da81`) runs in both
arms with threshold 0 versus 0.1. Tracing and shader validation are disabled.
The primary metric is complete native `decode_cycle_ms / decode_output_tokens`,
including inter-command gaps. Reports retain binary/metallib/tool/model hashes,
actual counters, prompt hashes, output hashes, cache conditions, memory,
thermal/power state and all four ABBA arms. Generated answer text is omitted.
Every measured request must produce the requested output budget with no cached
prompt tokens; cache salts distinguish cold requests.

## Completed GGUF B1 matrix

The initial expanded matrix covers 2,048 / 8,192 / 31,744 prompt tokens,
code/math/prose, three independent trials each and 512 output tokens. It uses
two resident MoE GGUF servers, with complete matched warmup rounds until both
accumulate at least ten seconds of 2k/512-token warmup work. All 27 trials
finish with valid native counters and released request resources.

| Context | Trials | Gains above gate | Inconclusive | Detected regressions | Changed output hashes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2k | 9 | 3 | 2 | 0 | 0 |
| 8k | 9 | 4 | 1 | 0 | 3 |
| 31,744 | 9 | 6 | 0 | 0 | 0 |

Descriptive median gains are 2.75%, 3.43% and 3.22% respectively; these are not
an aggregate qualified speedup. Promotion requires each applicable comparison
to exceed `max(2%,2*noise)`; spread >5% is inconclusive. A regression `pass`
does not mean a speed gain. All failed/noisy comparisons remain in the reports.

Three 8k trials change output hashes: math sample 2 and prose samples 0 and 2.
Some differences occur between repeats of the same policy, including the
full-width baseline. They do not isolate a pruning effect. An exact-prompt
single-server replay against clean main and the disabled candidate is queued.
No output-equality claim is made for those cases, and no quality assessment is
inferred from timing. The independent operator/model state oracles exercise
their existing numerical contract; they do not prove every autoregressive
boundary produces identical text. All requested output counts are retained,
so useful-token throughput does not silently compare different budgets.

## Completed awake MLX B1 matrix

The replacement completes the same 27 B1 trials (three contexts,
code/math/prose, three trials, 512 outputs). It uses one resident server per
complete ABBA round, four matched 2k/512-token warmups and a 26 GiB cap.
Every output hash and both fallback-probe hashes match. The complete-series
UTC/awake discrepancy is **0.011906 seconds**, below the one-second gate.

| Context | Trials | Gains above gate | Inconclusive | Detected regressions | Changed output hashes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2k | 9 | 6 | 0 | 0 | 0 |
| 8k | 9 | 0 | 0 | 0 | 0 |
| 31,744 | 9 | 2 | 3 | 0 | 0 |

Descriptive median gains are 3.18%, 2.51% and 2.36%. In particular, the 8k
median does not establish a qualified gain: every trial remains below its own
noise-adjusted gain bound. The three noisy 31,744 trials remain inconclusive.

## Completed B2–B4 checks

Both storage paths now have three code trials at 2,048 prompt tokens and
**256 output tokens per lane**, widths 2/3/4. They use one resident server per
complete ABBA round, four matched 2k/512-token warmup requests and a 26 GiB cap
for MLX (20 GiB for GGUF). These width checks do not establish long-context
B2–B4 capacity or substitute for the 512-output B1 matrix.

| Storage / width | Trials | Gains above gate | Inconclusive | Detected regressions | Changed hashes |
| --- | ---: | ---: | ---: | ---: | ---: |
| GGUF / B2 | 3 | 1 | 0 | 0 | 0 |
| GGUF / B3 | 3 | 1 | 0 | 0 | 0 |
| GGUF / B4 | 3 | 0 | 0 | 0 | 0 |
| MLX / B2 | 3 | 3 | 0 | 0 | 0 |
| MLX / B3 | 3 | 2 | 0 | 0 | 0 |
| MLX / B4 | 3 | 2 | 0 | 0 | 0 |

All sampled/strict-JSON probes match in these width checks. Per-trial values,
actual width histograms and every arm remain in `prefix-*-b[234].json`.

## Memory and excluded captures

The first paired MLX B1 attempt failed while loading its second resident model
with a Metal out-of-memory error, before any measured case. It is not a failed
kernel-correctness check or a measured latency comparison. The repeat uses
`--residency sequential --max-memory 26G --warmup-seconds 0 --warmup-rounds 4`,
one server per complete ABBA round, matching workload and warmup histories.
This trades a longer separation between arms for natural single-server memory
conditions. B2–B4 use the completed single-resident method documented above.

A power-log audit subsequently invalidated the overnight single-resident MLX
B1 matrix because it crossed repeated sleep/dark-wake intervals. It is excluded in full. Its completed awake replacement is reported above;
low ABBA spread alone never makes a sleep-affected trial usable. The completed expanded GGUF B1 matrix is unaffected.

The current local scope is the available machine. Other GPU-family and small
memory qualification remain unexecuted in the current repeatable maintainer
package in PR #523; merged #487 supersedes this historical runtime prototype. No shared default is qualified by these local/historical results. Across the
72 trials, 30 gains clear their per-trial bounds, six are inconclusive, no
throughput regression exceeds its tolerance, and three GGUF B1 trials contain
differing hashes. These counts are not a current-main or aggregate speed claim.
