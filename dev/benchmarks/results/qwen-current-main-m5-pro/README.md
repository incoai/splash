# Current-main validation and prepared maintainer matrix

Runtime source: `1482f6bbc2abf12375332725302019594150b6cb`.
Machine: Apple M5 Pro, 20 GPU cores, 48 GiB, macOS 26.6.0.
This bundle is separate from the historical performance experiments. PR #523
changes tools/docs/evidence, with no production runtime edits. Native model
checks and current-server functional runs are queued after all historical
timing, diagnostics and exact-prompt replay; their status is not implied by
the already-passing CI or legacy oracle logs.

## Execution scopes

| Scope | Status / interpretation |
| --- | --- |
| Current native build, CPU and ASAN/UBSAN/TSAN suites | Queued; see `jobs.json`/`summary.json` once execution starts. |
| Current full Python/lint/format check | Queued locally; current PR CI is recorded separately. |
| Four models × INT8/BF16 KV × FP32/BF16 recurrent state | 16 fresh shader-validated oracle cells queued; no legacy log is substituted. |
| MoE GGUF and dense MLX all/auto decode/mixed HTTP | Four functional reports queued; one trial each, not broad performance qualification. |
| Other hardware | **Unexecuted.** `maintainer-prepared.json` is preparation, not a passing test. |

The frozen build contains the unchanged native source from the revision above.
`jobs.json` records commands, model manifests/target and draft revisions, binary
hashes, times, exit codes and environment. `summary.json` records actual exit
results and oracle pass markers; a missing file or unfinished job is not a pass.
Compiler/test logs and each complete serving report accompany those summaries.
Local roots are replaced with `$SPLASH` and `$CURRENT_WORKTREE`; generated
answer text is omitted while hashes/usage/native counters are retained.

## Numerical and functional contracts

Oracle cells use `MTL_SHADER_VALIDATION=1`, `--ane-ffn-share 0` and a 32,768
maximum context. They exercise the current oracle's same-graph equivalence,
ragged/permuted lanes, state/KV storage, capture/restore/replay, preemption,
sampled/constrained behavior, mask commits, retained/output budgets and its
controlled adaptive/full-row cases. Cross-shape numerical equality remains
subject to the upstream oracle's stated contract. This is real-model execution
of that contract, not a quality evaluation or exhaustive every-context test.

Functional HTTP runs use the same frozen production executable in both arms,
with product `--verify-rows all` versus `--verify-rows auto`. The harness checks
effective native status: sparse MoE should expose adaptive execution; dense
auto may correctly report all. One server is resident per full ABBA round,
with four matched 2k/512-token warmups and a 34 GiB cap. Decode uses a 2k code
prompt and 512 outputs; mixed uses a 2k active stream plus 8k long/512-token
short arrivals, with 512/32/32 requested outputs. Every request must finish its
budget and release resources; mixed requires native work overlap.

The serving runs use scoped wake prevention and the UTC/awake-clock check.
Neither tracing nor shader validation is enabled during their timings. Their
single trial and finite fallback probes establish functional execution only;
they do not qualify a speed, tail latency, sampling distribution or quality
claim. Greedy hashes are reported, including expected adaptive differences.

## Runs to return from maintainer hardware

`maintainer-prepared.json` contains all 36 jobs with `status=unexecuted` and
`complete=false`: four target/storage cells, each with four KV/state oracles,
B1 decode/mixed across 2k/8k/31,744 and B2/B3/B4 decode at 2k. It uses
placeholder paths and was prepared on this Mac; it records no execution on any
other machine. JSON command-array placeholders are illustrative and are not
expanded by the runner. Regenerate the manifest after configuring actual paths
and model bytes using [the maintainer recipe](../../qwen-maintainer-runs.md).

The prepared serving workload uses code/math/prose, three trials, 512 output
tokens and sampled/strict-JSON probes. Hardware capacity and native protocols
must match; unavailable cells remain unexecuted. Return manifests, child
reports, logs and device/power identities against #508/#523. Do not include
model weights or infer that preparation supplies missing hardware coverage.
