# Qwen live verification on this Mac

**Versioned historical experiment:** runtime instructions and results below apply to their pinned legacy revisions. The runtime flags/build targets are not current-main settings. See [the October 21 current-main audit](current-main-audit.md) and [current qualification recipe](qwen-maintainer-runs.md). Replacement runs and final validation are still being completed.


The first implementation improves native MoE decode throughput by 6–7% and
one long-prompt HTTP workload by 6.5%. Short-prompt HTTP comparisons remain
inconclusive because of clock and thermal drift, including a warmed rerun.
Keep the feature experimental and disabled by default.

## Machine and source

Evaluated October 20, 2026 on an Apple M5 Pro with 20 GPU cores, 48 GiB unified
memory and macOS 26.6.0. The measurements reported below ran on AC power.
Earlier battery runs are retained as exploratory artifacts. The missing Metal
Toolchain component was installed with `xcodebuild -downloadComponent MetalToolchain`
before building; `xcrun metal` then reported version 32023.921.

The clean baseline is Splash `be3b083c5f413673d44e44881042e875e674e333`.
Implementation is on `deepinfer-qwen-eval`, reusing the existing
`pr-283-live-verify` prototype while preserving current main's execution plans,
in-place MoE path, pending GDN tapes, thinking budgets and BF16 state support.
The original branch and stash were preserved. See the
[source audit and conditional plan](deepinfer-qwen-plan.md) for pinned DeepInfer
references and the rationale for choosing this work.

Two already installed models were used:

| Role | Model | Storage |
| --- | --- | --- |
| MoE optimization and timing | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_XXS` | GGUF |
| Dense fallback and state verification | `mlx-community/Qwen3.8-27B-4bit` | Safetensors, MLX affine quantization |

Timing used INT8 KV and FP32 GDN state. Native profiles use GPU execution;
model oracles explicitly disable ANE FFN. No dense-model speedup is claimed.
The checked-in [measurement bundle](results/qwen-live-verification-m5-pro/README.md)
contains binary/model fingerprints, raw native cycles, HTTP metric deltas,
output hashes, environments and summaries, including the inconclusive runs.

## Implemented behavior

`SPLASH_LIVE_VERIFY_THRESHOLD` accepts a finite number in `[0, 1]`; zero is the
default. An enabled greedy MoE lane keeps its anchor and a contiguous prefix
selected by cumulative confidence from the existing draft selector. This is a
heuristic, not DeepInfer's trained acceptance head. No new model weights are
required.

For position `p` and candidate `c`, the score is its unary score plus the edge
from the previously selected candidate. Confidence is the softmax probability
of the selected candidate among that position's selector candidates. The budget
uses the product of those confidences through each position. This is not a
full-vocabulary proposal probability or a calibrated target-acceptance estimate.

The GPU excludes inactive suffix rows from routed and shared expert routes,
then limits acceptance to the eligible prefix. The one-lane staged path keeps
its in-place row layout. Tiles may still compute padded rows. Eight-row
storage, draft work, dense projections, attention and GDN arithmetic retain
their existing shapes. Eligible rows therefore do not measure all target work.

Sampled lanes keep full verification, including in mixed batches. Constrained
batches, dense targets and cycles forcing a thinking-end token also retain
the existing path. Invalid selector confidence falls back to full width.
Startup prepares ordinary acceptance even when a greedy warmup uses live
acceptance, preventing a later sampled request from compiling that pipeline.

The profiler now saves per-cycle timing, committed output counts, proposal and
acceptance counts, prefix survival and eligible expert rows. Optional ordinary
command traces separate graph construction, backend preparation, Metal encoding,
GPU execution, completion notification and finalization. Production builds do
not include backend tracing. The new [sweep](verification_sweep.py) reuses the
existing ABBA comparison rule.

## Performance

All comparisons use off/on/on/off rounds. The regression bound is
`max(2%, 2 × measured spread)`; spread above 5% is inconclusive. A useful gain
must exceed that bound. Native throughput divides total committed tokens by
total measured cycle wall time, including host work.

At 2,048 prompt tokens, 32 cycles per width and threshold **0.1**, all native
comparisons were stable. Baseline and candidate committed the same number of
tokens per cycle in each width.

| Lanes | Off, tok/s | On, tok/s | Throughput gain | Spread | Mean eligible rows, on/full |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 114.09 | 121.12 | 6.16% | 0.20% | 6.09 / 8 |
| 2 | 174.37 | 186.45 | 6.93% | 0.24% | 12.28 / 16 |
| 3 | 204.75 | 219.21 | 7.06% | 0.04% | 18.63 / 24 |
| 4 | 248.35 | 265.22 | 6.79% | 0.12% | 25.28 / 32 |

Raw cycles are in [native-cycles.csv](results/qwen-live-verification-m5-pro/native-cycles.csv);
summaries, environment samples and fingerprints are in
[measurements.json](results/qwen-live-verification-m5-pro/measurements.json).
Complete local Chrome traces remain in the ignored build directory.
Threshold 0.2 was noisier and is not established as a better choice. Earlier
prototype sweeps at 0.01 and 0.05 also do not justify promoting them.

HTTP comparisons used the clean baseline and candidate binaries with 34 GiB
memory, 32K maximum context, disk prefix cache disabled, greedy decoding and
512 requested output tokens. The metric is the `/status` delta of
`decode_cycle_ms / decode_output_tokens`, so time between decode commands is
included. It measures serving decode throughput rather than total request
latency or prefill performance.

The **9,641-token coding prompt** improved from **203.27 to 216.53 tok/s**
(**6.52%**), with **0.66% spread** and identical generated messages in all four
rounds. See `http.ac-http-0.1` in the checked-in measurement JSON.
Code, math, prose and four concurrent requests had excessive spread, so their
apparent gains are not evidence of improvement. The warmed rerun
(`http.ac-enabled-warmed`)
remained inconclusive at 14–16% spread as the server reported thermal state
changing from nominal to fair. No broad HTTP speedup is claimed.

With pruning disabled, the candidate passed the regression bound for code,
math, long code and four concurrent requests. Prose passed on its
warmed rerun (`http.ac-disabled-prose-warmed`),
with 0.22% extra seconds per token and 0.54% spread. The other
disabled controls (`http.ac-http-0`) ranged
from a 0.25% reduction to a 1.06% increase in seconds per token. All measured
HTTP message hashes matched their baseline counterparts, including concurrent
outputs. Seeded sampled probes also matched; strict JSON probes remained valid.
These finite checks do not establish universal output equivalence.

## Correctness checks

The full `make -j8 check` gate passes, including the CPU and shader-validated
Metal suites, all native test/tool builds, full Python discovery, compileall
and Ruff checks. Architecture and whitespace checks also pass. Focused Metal
tests ran with `MTL_SHADER_VALIDATION=1`:

- Live-row selection and acceptance: cutoffs, invalid confidence, sampled
  fallback and mixed lanes.
- GGUF MoE: existing FP64 numerical checks, plus bitwise live-prefix comparisons
  for every supported expert format, staged/register paths, B1–B4 and every
  cutoff. The final in-place refinement passed both the all-format live-only
  suite and the full numerical suite repeated during `make check`.
- Target sampling, draft selector, GDN decode and Metal backend, including
  trace execution and clock-order checks.

Real-model runtime oracles pass for both installed models with FP32 and BF16
GDN state, INT8 KV and threshold 0.1. The final in-place MoE refinement also
passes both state formats at threshold 0.2. These exercise lane permutations,
restore/preemption, sampling, constraints and thinking budgets under the
existing oracle's numerical contracts. Another 143 Python tests pass across
benchmark/smoke harnesses and sampling, thinking budgets, structured tools and
partial tool output. Full Python discovery subsequently completed 1,270 tests
with 27 skipped; compileall and Ruff checks also pass. The full local validation
command and outcome are recorded
in the measurement bundle's [validation record](results/qwen-live-verification-m5-pro/validation.md).
Complete logs remain in the local ignored build directory.

## Decisions and remaining work

Operator attribution places expert kernels at roughly 40–50% of decode time;
GDN is roughly 4–7%. Attributed dispatches execute separately, so these shares
identify candidates rather than predict fused speedups.

The ordinary native trace shows a median GPU-to-next-GPU gap of about
0.50–0.53 ms. Median graph construction is 0.145 ms, backend preparation
0.027 ms, encoding 0.142 ms, completion notification 0.078 ms and finalization
0.019 ms. These are native harness observations, not a complete engine-loop
timeline. They suggest a small remaining host budget on this machine; complex
device-side pipelining is deferred.

GDN pruning is also deferred: it must be redesigned for pending tapes, and
its measured share offers limited headroom. The old stash was not applied.
Adaptive cost selection, dense row compaction and context-aware mixed-prefill
scheduling remain separate conditional steps in the plan. Their gates require
additional measurements; no scheduler or GDN rewrite is included here.

Before changing defaults, qualify short-prompt HTTP workloads under stable
clocks, additional contexts and quantization formats, and other Apple GPU
families. Full engine-loop tracing and mixed-load p50/p95 latency measurements
are also outstanding. Keep threshold 0.1 opt-in for this machine/model.

## Reproduce locally

Build with `make -j8 all`. Enable the experiment with:

```sh
SPLASH_LIVE_VERIFY_THRESHOLD=0.1 ./splash serve \
  --model unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_XXS
```

Unset the environment variable or set it to zero to use full verification.
For a native sweep, resolve the installed model root and run:

```sh
make -j8 BUILD=build/deepinfer-candidate all \
  build/deepinfer-candidate/engine-tests/decode-profile
.venv/bin/python -m dev.benchmarks.verification_sweep \
  --profile build/deepinfer-candidate/engine-tests/decode-profile \
  --metallib build/deepinfer-candidate/splash.metallib \
  --model-root /path/to/installed-model \
  --output build/verification-repeat --thresholds 0.1 --cycles 32
```

The exact HTTP workload and prompts are retained as an
[experiment snapshot](results/qwen-live-verification-m5-pro/http_workload.py).
Build clean main separately, then compare both servers using the same installed
model root:

```sh
git worktree add --detach ../splash-live-baseline be3b083c5f413673d44e44881042e875e674e333
make -C ../splash-live-baseline -j8 all
.venv/bin/python -m dev.benchmarks.results.qwen-live-verification-m5-pro.http_workload \
  --baseline-binary ../splash-live-baseline/build/splash \
  --candidate-binary build/deepinfer-candidate/splash \
  --model-root /absolute/path/to/installed-model \
  --threshold 0.1 --cases long-code --tokens 512 --samples 1 \
  --output build/verification-http-repeat
```

The snapshot starts and stops its own servers; it never measures an existing
server. Model bytes and complete local traces/logs are excluded from the PR.
The checked-in bundle preserves the numerical evidence needed to audit the
reported claims.
