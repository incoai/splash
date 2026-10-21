# Context-aware prefill experiment on M5 Pro

**Historical experiment; no current-main scheduler change.** Runtime
implementation `02a1162e` is based on measurement source `bc09da81` and archived
in [PR #522](https://github.com/incoai/splash/pull/522). Main
`1482f6bbc2abf12375332725302019594150b6cb` has a newer fixed/per-row estimator,
attention-pair limits and streaming/multi-pass scheduling. The prototype and
its environment flag are not transplanted into that runtime. See the
[current-main audit](../../current-main-audit.md).

## Question and bounded implementation

The old scheduler estimated prefill as rows times one global EMA. The optional
bucket policy classifies actual ragged rows, row-weighted history, width and
GPU/ANE path. Its 96 fixed cells require three earlier matching observations;
otherwise it uses the exact baseline fallback. Cells train only on finite,
positive, representative text-prefill observations of at least 64 rows.
The 500 ms contention target, 64-row minimum, 2,048-row ceiling, admission,
priorities, overtaking limits and decode debt are preserved. No curve is shared
between runtimes. The default stays `ema` on the archived branch.

The useful question is whether more accurate estimates improve contended
latency without harming throughput or fairness. Prediction error alone cannot
answer that question.

## Portable evidence map

| File | What it establishes |
| --- | --- |
| `prediction-errors.json` | Prior-only predictions from four control traces; no future observation trains its own prediction. |
| `round-*-prefill-excerpt.json` | Exact per-lane geometry, seeds, observations and source-trace hashes needed to reproduce those errors. These are excerpts, not full GPU timelines. |
| `geometry-control.json` | Complete mixed-workload control, including native overlap, cache, memory and output accounting. |
| `serving-summary.json` | Completion/pending state, per-context throughput and latency gates, ANE setup, budgets and descriptive distributions. |
| `moe-*.json`, `dense-*.json` | Readable four-arm comparisons with times, budgets, counters, hashes and decisions. |
| `*.full.json.gz` | Complete cleaned reports, including every native poll and chunk timestamp. Each readable report records its archive hash and the original source-report hash. |
| `excluded-*-sleep.json` | Invalid overnight MoE controls retained with explicit exclusion; their apparent gains are not evidence. |
| `qualification.json`, `jobs.json`, `prefill-*.log` | Source identity and correctness execution status. |

The four geometry controls have 74–75 eligible observations each, including
actual two-lane ragged commands. Mean absolute relative prediction error falls
from **9.67–10.40%** to **5.89–7.34%**; p95 falls from **24.44–27.31%** to
**13.95–17.92%**. The evaluation is prequential: both estimators predict each
observation before ingesting it. Warmups remain part of the trace history;
these error counts are not independent latency trials.

Readable reports omit the repeated `native_samples` and `responses[].chunks`
arrays. Their deterministic gzip archives retain both arrays in full. They
remove generated answers and replace local roots in the same way as the other
portable reports. To inspect one, use `gzip -dc dense-cold.full.json.gz` or
Python `gzip.open`. Archive round-trip checks preserve every comparison and
per-round native counter exactly. Summary latency distributions are computed
from the complete source reports before reduction.

## Serving protocol

One resident server runs each complete baseline/candidate/candidate/baseline
round, then closes. Both policies use the same frozen production executable,
metallib and model bytes, four matched 2k/512-token warmups and identical case
order. The server cap is 34 GiB on the 48 GiB M5 Pro. Shader validation and
tracing are disabled for timing; diagnostic traces run separately.

Each workload starts a 2k active stream requesting 512 output tokens, then
introduces a long request and a 512-token interactive request, both requesting
32 tokens. The harness verifies all three full budgets, exact prompt counts,
released resources and native long-prefill/continuation progress while the
original request decodes. HTTP lifetime overlap alone is insufficient.
The primary cost is complete workload wall time divided by 576 output tokens;
short/long TTFT and p95/maximum active-stream chunk gaps are separate gates.

Cold cases cover 2k/8k/31,744 contexts, three trials each. Cached and RAM-prefix
continuation cases cover 8k, three trials each. Priming is outside measurement;
restore/suffix work is inside. Continuation does not simulate suspension under
memory pressure. Reports retain actual cached counts and native polling.

Gains must exceed `max(2%,2*noise)`; noise above 5% is inconclusive. A throughput
regression `pass` does not establish a speed gain. The six arm observations per
context in the summary's p50/p95 tables are descriptive; they are not a
high-confidence estimate of a production tail distribution.

## Measured decision

The awake dense cold/cache/continuation series has no qualified throughput
gain. One cached trial increases the maximum active-stream gap by **5.74%**
at **2.04% noise**, beyond its **4.08%** cost bound. One continuation trial
reduces short TTFT and the maximum gap beyond its bound, but other trials do
not repeat the latency win; that same trial also has differing output hashes.
These isolated improvements do not clear a workload-wide promotion gate.
The MoE continuation series is inconclusive at 10–13% throughput noise.
Awake MoE cold/cache replacements and final diagnostic/sanitizer outputs are
tracked explicitly in the manifests until they finish.

Initial ANE configuration is held constant per comparison: dense runs report
share 0.4117647059 and minimum split rows 1801. This does not imply that every
contended command uses ANE: halved budgets can fall below the minimum. The
actual submitted path must be checked in diagnostic traces before attributing
latency to an ANE handoff. Host ready/report events include callback delay and
are not hardware ANE timing.

The old prototype remains an experiment. Better offline prediction does not
justify changing current main's estimator. Only this Mac is measured; the
[maintainer package](../../qwen-maintainer-runs.md) leaves other hardware jobs
explicitly unexecuted.
