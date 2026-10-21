# Bounded context-aware prefill cost experiment

**Versioned historical experiment:** runtime instructions and results below apply to their pinned legacy revisions. The runtime flags/build targets are not current-main settings. See [the October 21 current-main audit](current-main-audit.md) and [current qualification recipe](qwen-maintainer-runs.md). Replacement runs and final validation are still being completed.


This implements the conditional scheduler experiment in step 6 of
[#508](https://github.com/incoai/splash/issues/508). It is based on measurement
base `bc09da81`, with subsequent HTTP-harness fixes. Set
`SPLASH_PREFILL_COST_MODEL=bucketed` to enable it; `ema` or an absent variable
keeps the existing policy. Invalid values fail startup with a usage error.
**The default stays EMA.** Local serving qualification is in progress.

## Problem and evidence

The baseline estimates prefill time as new rows multiplied by one global
milliseconds-per-token EWMA. Equal row counts can cost differently at long
history or on the ANE split path. A ragged command can include a short leading
request and a much longer-history second lane; the leading request alone does
not describe its cost.

Ordinary single-resident mixed-load traces now retain the startup seed, exact
row count/history for every submitted lane, expected execution path, and the
prediction made before each observation. They confirm actual prefill progress
while another native request decodes. Earlier dual-server dense traces were
serialized by admission pressure and are excluded from contended-latency claims.

Prequential offline evaluation of four newer MoE control traces (74–75 eligible
observations each, including actual two-lane ragged commands) reduces mean
absolute relative prediction error from 9.67–10.40% to 5.89–7.34%, and p95 error
from 24.44–27.31% to 13.95–17.92%. Every prediction uses only earlier samples.
This motivates the experiment; prediction accuracy is not latency benefit.
The control serving runs themselves are noisy and do not qualify a gain.
The [evidence bundle](results/qwen-prefill-cost-m5-pro/) retains those controls,
all prediction records and prefill-only trace excerpts with source-trace hashes.
The excerpts deliberately omit GPU events and cannot reconstruct a full timeline;
they can reproduce the estimator errors with `prefill_prediction.py`.

## Runtime contract

`Scheduler` stores 96 fixed cells: three row bands (<=128, <=512, >512), four
row-weighted history bands (<=1023, <=4095, <=16383, >16383), four widths and
two GPU/ANE path classes. The table is approximately 1.5 KiB per engine and
never grows or persists. Model/draft/KV/state/build identity is implicit in the
owning runtime; no curve is shared across runtimes.

Only finite positive representative text-prefill times with at least 64 rows
train a cell, using EWMA weight 0.25. Three matching prior observations are
required. Until then, the existing capacity-based global EMA fallback is
preserved exactly, including a short leader using less than the trial budget.
Restore/image/auxiliary timing does not train ordinary text-prefill cells.

For each existing halved trial budget, provisional stack geometry contains
every selected ragged lane. History is weighted by that lane's actual new
rows. The path callback checks the model's currently available ANE split.
At submission, the scheduler captures that path class; if a split retires
while completing, the completed observation still trains the submitted class.
The callback is read on the engine thread and its resources outlive the engine.

The existing 500 ms contention target, 64-row minimum, 2,048-row ceiling,
priority, shortest-first ordering, overtaking bounds, admission and decode debt
remain in force. The experiment only changes the cost estimate used to halve
a contended prefill budget. The trace's after-cut prediction describes the
actual submitted geometry; it is distinct from the provisional capacity estimate.

## Validation and remaining gate

The full CPU suite and existing ASAN/UBSAN/TSAN suite pass, followed by focused
engine, native-loop and ragged-scheduler tests. Added scheduler cases cover
three-prior-sample fallback, unseen context/width, GPU/ANE separation, submitted
path retention across ANE retirement, auxiliary timing exclusion, invalid
geometry, and a non-leading long-history lane that lowers the budget from
2,048 to 1,024 while preserving shortest-first order. Existing fairness,
priority, debt and overtaking cases run under both policies.

Dedicated ragged-scheduler ASAN/UBSAN and TSAN targets are now included in
`make test-sanitizers`; their first local execution is queued after the frozen
performance binaries finish. Building them during performance measurement
would contaminate those results. Production/profile binaries remain frozen.

```sh
make -j8 test-engine-cpu test-sanitizers build/splash build/serving-profile
python -m unittest dev.tests.tools.test_prefill_prediction
python -m dev.benchmarks.prefill_prediction /path/to/round-1-trace.json \
  --output build/prefill-prediction.json
python -m dev.benchmarks.serving_regression \
  --model MODEL --model-root /path/to/model \
  --baseline-binary build/splash --binary build/splash \
  --baseline-env SPLASH_PREFILL_COST_MODEL=ema \
  --candidate-env SPLASH_PREFILL_COST_MODEL=bucketed \
  --scenario mixed --residency sequential --max-memory 34G \
  --contexts 2048,8192,31744 --tokens 512 --samples 3 \
  --warmup-seconds 0 --warmup-rounds 4 \
  --output build/prefill-cost
```

Repeat for cold, `--cache-mode cached` and `--cache-mode prefix-resume`, for
MoE and dense targets. Prefix-resume is RAM-prefix continuation, not suspension
under resource pressure; actual cached counts are checked. Restore and suffix
prefill are inside measurement while priming is outside. The harness rejects
serialized admission, stale status and missing output budgets, and checks that
resources are released after each workload.

Promotion requires lower contended latency without starvation or a throughput
regression outside `max(2%,2*noise)`; spread >5% is inconclusive. Report short
and long TTFT, p95/max active-stream gaps, actual output counts, scheduled widths,
memory and cache reuse for every trial. Other-device jobs in #510 remain
unexecuted. A negative or inconclusive local gate leaves the default unchanged.
