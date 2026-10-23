# Serving qualification tools

`serving_regression.py` measures compatible frozen builds on the same Mac.
The recipe uses production `--verify-rows all|auto`, separately for each arm,
and confirms effective behavior from native status. It compares a setting in
the same frozen build or two compatible builds; it does not change inference
code. See [the experiment summary](qwen-serving-experiments.md) for the
separate historical research archive.

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
discrepancy >=1 second rejects the run. Stable ABBA costs alone do not prove
the machine remained awake or held a stable frequency.

`trace_report.py` can inspect the versioned experimental trace format. It
rejects dropped records, unfinished commands, failures and invalid clocks.
Current production binaries do not emit that format. For an existing capture:

```sh
python -m dev.benchmarks.trace_report /path/to/capture.trace.json \
  --output build/trace-summary.json
```

The confidence/prefill analysis programs and their results are preserved in
the [research archive](https://github.com/linson007/splash/tree/3e8911fc89a6ed009e7b0561bfa7c23e58ea594f/dev/benchmarks).
Their results do not establish latency benefit or current acceptance calibration.

Use [the full maintainer recipe](qwen-maintainer-runs.md) for the 36-job matrix.
`--trace` requires a matching instrumented checkout; current main's
production binary has no historical trace-output hook or `serving-profile`
target, and the harness refuses to silently ignore that option.
