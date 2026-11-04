# Prompt-lookup experiment

`dev/benchmarks/lookup_regression.py` compares the opt-in fixed16 prompt lookup
with ordinary DFlash in the same production binary. This is a narrow experiment
for greedy B1 requests that repeat part of their prompt. It does not establish a
general generation or batched throughput improvement. The runtime option remains
off by default.

The two frozen synthetic cohorts are alternative workload sets for the same
optimization. `complete` contains three whole-document edits and three ordinary
controls. `fresh` contains six additional whole-document edits and three ordinary
controls. The committed fixture files retain source text, requested edits, expected
complete responses, and full token arrays. Input selection precedes measurement;
unsuccessful edits and zero-hit cases remain in every timing aggregate.

Use the exact target and draft revisions in
`dev/benchmarks/fixtures/lookup/model-pins.json`. The fixed token arrays require
that model's tokenizer and template. The runner verifies source revisions, file
sizes and content hashes, including all model weights, before and after the run.
It rejects a different model rather than retokenizing or changing quantization.
Model binaries and tokenizer assets are not part of this change.

Build the production benchmark on an otherwise idle Apple Silicon Mac, commit
the source being tested, and put results outside the checkout:

```sh
make build/engine-tests/backend-benchmark build/splash.metallib
python3 dev/benchmarks/lookup_regression.py \
  --cohort complete --worktree "$PWD" --model-root /path/to/installed/model \
  --expected-revision "$(git rev-parse HEAD)" --out /path/to/results/lookup-complete
python3 dev/benchmarks/lookup_regression.py \
  --cohort fresh --worktree "$PWD" --model-root /path/to/installed/model \
  --expected-revision "$(git rev-parse HEAD)" --out /path/to/results/lookup-fresh
```

Run the cohorts sequentially. The shared machine lock prevents overlapping runs
of these experiment drivers; it cannot detect unrelated GPU applications. Retain
the build log and record Mac model, GPU cores, RAM, OS, power state and other load
alongside the results. Hashes identify the measured artifacts but cannot prove
that an existing binary was compiled from the recorded source revision.

Each process retains eight complete OFF warmup passes (48 or 72 requests), followed
by exactly three groups of adjacent per-case OFF/ON/ON/OFF requests (72 or 108).
Every request starts with an empty cache. Edit caps are 1024 tokens and control
caps are 256; natural EOS remains in all full arrays and actual token counts.
Model startup and warmups are excluded from the steady-state primary metric but
are retained in the receipt and raw JSON. Idle toggles and cache eviction precede
request timing equally in both arms.

The primary metric sums every edit request's wall time. All edits must improve in
every group, and the aggregate reduction must exceed `max(2%, 2 × maximum A drift)`.
Both A endpoint drift and B spread must be at most 5%, for the aggregate and every
case. Material ordinary-control regressions block qualification. Complete full
outputs, cold-cache counters, lookup work, and all OFF semantic counters are
checked across both arms and all warmups. Lookup acceptance counters may differ
from DFlash; they are retained and checked for consistency rather than treated as
output equality.

Decode tokens/s uses actual decode-output counters divided by decode wall time.
It is reported separately from whole-request latency/rate. Exact complete-edit
success is a separate diagnostic and never an inclusion filter. The pinned
ByteLevel decoder removes special tokens for that diagnostic only; raw EOS token
arrays remain unchanged. The fresh fictional-clause fixture can therefore report
an incomplete edit without disappearing from the performance result.

Audit a saved runner output without running inference:

```sh
python3 dev/benchmarks/lookup_regression.py --cohort fresh \
  --audit /path/to/results/lookup-fresh \
  --tokenizer /path/to/installed/model/tokenizer/tokenizer.json
```

The tokenizer is optional for an offline audit; without it, the audit recomputes
all timing/array/work gates but does not independently recompute decoded task
success. `--raw /path/to/native.json` checks a standalone native JSON file, useful
for historical evidence. That mode explicitly makes no build/model provenance
claim. A saved-run audit additionally verifies command options, receipts, raw
artifact hashes, before/after pins, model-content check coverage and the saved
summary. Preserve failed or interrupted outputs; do not retry and silently replace
an unfavorable cohort.

CPU mutation tests do not require model assets:

```sh
python3 -m unittest dev.tests.tools.test_lookup_regression
```

Generated reports and historical raw results belong outside source commits, as
described in [validate.md](validate.md). Publication evidence should distinguish
the historical research revision from a newly rebased PR head. Historical gains
must not be presented as measurements of an unmeasured publication head.
