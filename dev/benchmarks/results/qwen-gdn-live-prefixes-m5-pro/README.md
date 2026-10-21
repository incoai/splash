# GDN prefix experiment: correctness passes, performance gate not met

**Historical data:** these measurements apply to the source/build identities in this bundle. Current main has since merged a different runtime stack; see [the source reconciliation](../../current-main-audit.md). They are not current-main qualification.


Measured on October 21, 2026, on an Apple M5 Pro (20 GPU cores, 48 GiB),
macOS 26.6.0, Apple GPU family Apple10. AC power was connected and the model
download was stopped throughout timing. Power/thermal snapshots accompany each
job. No reported thermal warning is a frequency-stability guarantee; the ABBA
noise check still applies. Builds and metallibs were frozen during all runs.

**Decision: keep `SPLASH_LIVE_GDN=0` and PR #511 in draft.** The pending-tape
implementation passes the correctness gates exercised here, but no stable
native or production HTTP comparison clears the promotion threshold. The
experiment is complete on this machine; it does not establish a useful serving
optimization or qualify a shared default.

## Native results

Each comparison is off/on/on/off with the same binary, model, threshold and
2,048-token context. The primary metric is total fused command wall seconds
divided by committed tokens within each arm and width. This is the native
command metric, not the complete HTTP cycle metric. INT8 KV and FP32 recurrent
state are used for timing. Tracing and shader validation are off.

| Model / threshold | Cycles per arm per width | B1 gain | B2 gain | B3 gain | B4 gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| GGUF IQ3_XXS / 0.1 | 32 | 0.11% | 0.55% | 0.45% | 0.66% |
| GGUF IQ3_XXS / 0.2 | 32 | 0.55% | 0.95% | 0.83% | 0.90% |
| MLX 4-bit / 0.1, warmed repeat | 64 | 0.54% | 0.75% | 0.66% | 0.61% |
| MLX 4-bit / 0.2, warmed repeat | 64 | 1.20% | 0.98% | 0.81% | 1.04% |

GGUF noise is at most 1.75%; warmed MLX noise is at most 0.90%. Gains in these
series are below the 2% minimum. The initial 32-cycle MLX series includes
inconclusive cells with apparent 8–10% gains and 13–17% noise. It is retained
in full, rather than counted as evidence of a win. Its B2/0.1 cell gains 2.50%
at 3.44% noise, below its 6.88% promotion threshold. A separate unmeasured
16-cycle-per-width warmup preceded the 64-cycle MLX series.

`native-cycles.csv` contains all **4,096 individual cycle samples**, including
the initial MLX series and its repeat. `native-summary.json` contains all 24
width comparisons, round costs, eligible rows, committed tokens, noise and
gain decisions. The `quiet` and `final` stage names distinguish initial runs
from the warmed repeat; neither means the noise test was bypassed.

For round costs R1–R4, baseline=(R1+R4)/2, candidate=(R2+R3)/2,
noise=max(abs(R1-R4)/baseline, abs(R2-R3)/candidate), and
gain=baseline/candidate-1. Promotion requires noise <=5% and
gain >max(2%,2*noise). A `pass` verdict in the shared ABBA utility means
no detected regression; it does not mean a qualified gain.

## Production serving and disabled control

`gdn-http.json` compares the same production executable with live GDN off/on
at threshold 0.1, B1, code prompts, three independent ABBA trials at each of
2,048 and 8,192 context tokens, and 512 output tokens. Its primary metric is
native engine `decode_cycle_ms / decode_output_tokens`, including the gaps
between decode commands. All six trials finish, all output hashes match, and
sampled/constrained fallback probes match.

The three 2k gains are 0.16%, 0.44%, 0.40%; the three 8k gains are 1.32%,
-0.47%, 0.55%. None exceeds its noise-adjusted promotion threshold. This
series uses the earlier startup warmup rather than the later sustained warmup
protocol; it supplies no positive speed claim that could depend on that choice.

`gdn-disabled-control.json` compares the measurement-base binary against the
GDN binary with GDN disabled in both arms at threshold 0.1. All output hashes
and fallback probes match. The first trial is inconclusive (11.49% noise);
the two stable trials regress 0.26% and 0.25%, inside the 2% floor. This checks
for a material disabled-path regression in this workload, not universal parity.

HTTP reports retain settings, tool/binary/metallib hashes, status counters,
memory, power, thermal snapshots, output hashes and usage. Generated answer
text is removed, prompts are represented by SHA-256, and local roots are
replaced with placeholders. The 2k/8k requests are paired resident MoE GGUF
servers; these are decode tests, not the contended-prefill tests affected by
dual-server admission pressure.

## Correctness and reproduction

`validation/` contains the final shader-validated GDN suite and four complete
real-model oracle logs: GGUF/MLX MoE crossed with FP32/BF16 recurrent state,
all with INT8 KV, threshold 0.1 and live GDN enabled. The shader suite passes
128 geometry/width/norm/state/head-layout/preparation cases with 130 steps
each, including poisoned suffixes, pending/full-prefix transitions and both
tape parities. The oracle covers restore, preemption, masks, sampling, lane
movement, thinking transitions and output limits under its numerical contract.

The first implementation failed the oracle's pipeline-cache invariant when a
sampled request followed live greedy decoding. Preparing the ordinary fallback
variants before serving fixes this; the attached logs describe the corrected
implementation. A failed launch using an obsolete model-link path was corrected
to the manifest-selected resolved root before the MLX oracle runs.

`identities.json` pins sources, binaries, model manifests, target revisions
and the draft revision. `jobs.json` records exact commands and timing settings;
replace `$SPLASH` and `$GDN_WORKTREE` with local worktrees. Build/test commands
and state invariants are in [the design note](../../gdn-live-prefixes.md).
For each native job run its recorded command in off/on/on/off order, keeping
the binary and metallib immutable. HTTP is reproduced with
`python -m dev.benchmarks.serving_regression` and the corresponding recorded
arguments. The shared harness may have later measurement-only fixes; use the
report's `tool_sha256` to distinguish versions.

The local model oracle runs do **not** cover BF16 KV or another Apple GPU
family. PR #510 prepares those maintainer jobs and explicitly records them as
unexecuted. No claim here depends on unavailable hardware.
