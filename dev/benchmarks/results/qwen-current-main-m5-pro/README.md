# Current-main validation and prepared maintainer matrix

Runtime source: `1482f6bbc2abf12375332725302019594150b6cb`.
Machine: Apple M5 Pro, 20 GPU cores, 48 GiB, macOS 26.6.0.
This bundle is separate from the historical performance experiments. PR #523
changes tools/docs/evidence, with no production runtime edits. Historical timing, diagnostics and exact-prompt replay are complete. Current
native/model checks run afterward with frozen matching builds. Their status
is recorded independently of CI and legacy oracle logs.

## Execution scopes

| Scope | Status / interpretation |
| --- | --- |
| Current native build, CPU and ASAN/UBSAN/TSAN suites | PASS locally; complete build and CPU/sanitizer logs are attached. |
| Current full Python/lint/format check | PASS locally: 1,348 tests, 27 skips; compileall, Ruff lint and format pass (188 files). |
| Four models × INT8/BF16 KV × FP32/BF16 recurrent state | Fresh shader-validated matrix is in progress. One MoE GGUF BF16-KV/FP32-state cell fails the upstream cross-graph penalty assertion; its original failing log is retained. An isolated test-oracle correction is being verified. No legacy log is substituted. |
| MoE GGUF and dense MLX all/auto decode/mixed HTTP | Four functional reports queued; one trial each, not broad performance qualification. |
| Other hardware | **Unexecuted.** `maintainer-prepared.json` is preparation, not a passing test. |

The frozen build contains the unchanged native source from the revision above.
`models.json` records all four installed manifests, source revisions and recorded
file digests; `toolchain.json` records actual compiler and Python dependencies.
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

## Penalty oracle finding during fresh validation

The frozen upstream oracle fails `moe-gguf-oracle-bf16-fp32` with
`reference_loss=0.205502`, above its cold-score allowance `0.1`. That check
compares incremental selections with logits recomputed through prefill.
It therefore also depends on cross-graph numerical drift, whose bound cannot
be inferred from a different unpenalized trajectory. A separate test-only
patch is under verification: snapshot the same forward before penalties,
independently calculate the policy from prompt/output/prefix counts in fp64,
and check every actual policy logit and the greedy winner with an fp32
arithmetic allowance. The original failure remains a failure in `summary.json`;
it is not converted into a passing upstream run.
