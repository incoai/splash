# Serving qualification tools

`serving_regression.py` measures compatible frozen builds on the same Mac.
The current-main recipe uses production `--verify-rows all|auto`, separately
for each arm, and confirms effective behavior from native status. Historical
DeepInfer-inspired runtime prototypes and trace captures have their own pinned
sources; they are not current-main feature flags.

```sh
python -m dev.benchmarks.serving_regression \
  --model MODEL --model-root /path/to/model \
  --baseline-binary /path/to/build/splash --binary /path/to/build/splash \
  --baseline-verify-rows all --candidate-verify-rows auto \
  --contexts 2048,8192,31744 --tokens 512 --samples 3 \
  --residency sequential --warmup-context 2048 --warmup-tokens 512 \
  --warmup-seconds 0 --warmup-rounds 4 --probes --output build/serving
```

Each complete ABBA round runs one resident server, preserving workload and
warmup history, and closes it before the next. Paired residency remains an
option when two models fit; it serializes measured requests between resident
servers to shorten separation between arms. Do not mix the methods silently.
The report retains exact tool/build/model hashes, settings, status and raw arms.
The decode metric is native decode-cycle seconds per output token, including
inter-command gaps, not just GPU execution.

`--scenario mixed --batch 1` introduces a long prompt and a 512-token
interactive prompt beside an active 2k streaming request. The active request
generates the selected budget; both peers generate 32 tokens. It records total
workload seconds/output token, short/long TTFT and p95/max content-chunk gaps.
Chunk arrival is not one timestamp per token. Native progress every 100 ms must
prove actual contention. The initial snapshot must be fresh. Later stale polls
are recorded and excluded from overlap evidence; without fresh progress the
run still fails. Serialized admission is rejected. Per-round stale-poll counts
remain visible so a transient transport response is not mistaken for progress.

`--cache-mode cached` primes the full long prompt, while `prefix-resume` primes
its leading half by character count. Fresh cache salts isolate arms. Priming
is outside measurement; restore and suffix work are inside. Actual reuse and
all prompt/output budgets are checked. This is RAM-prefix continuation, not
resource-pressure suspension. Per-round actual memory, pressure and scheduled
decode widths are retained; scheduled width is not a physical verify-row count.

All measured workloads require full output budgets and released resources.
Sampled/strict-JSON probes separately check response behavior. Hash equality is
diagnostic under adaptive verification, not a proof of universal numerical or
sampling equivalence. A `pass` under ABBA means no detected regression, not a
qualified gain. Promotion requires gain above `max(2%,2*noise)` and the agreed
latency/correctness gates; spread >5% is inconclusive.

The benchmark temporarily prevents sleep on macOS, releases its assertion on
failure too, and compares elapsed UTC against the awake monotonic clock. A
discrepancy >=1 second rejects the run. Earlier sleep-affected captures are
retained as excluded data and replaced. Stable ABBA costs alone do not prove
the machine remained awake or held a stable frequency.

`trace_report.py`, `prefill_prediction.py` and `confidence_calibration.py`
analyze the versioned historical trace format. Traces with dropped records,
unfinished commands, failures or invalid clocks are rejected. Prefill estimates
use only earlier observations, exact row-weighted ragged geometry and distinct
width/path cells. Confidence analysis keeps censored suffixes out of rejection
labels and separates workload/context/position. These prediction diagnostics
do not establish latency benefit or current-main acceptance calibration.

Use [the full maintainer recipe](qwen-maintainer-runs.md) for the 36-job matrix
and [the current source audit](current-main-audit.md) for what upstream already
implements. `--trace` requires a matching instrumented checkout; current main's
production binary has no historical trace-output hook or `serving-profile`
target, and the harness refuses to silently ignore that option.
