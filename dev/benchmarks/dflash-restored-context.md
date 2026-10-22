# Native restored-prefix experiment (#548)

Depends on #550. `decode-profile --draft-dump FILE --prompt-file PROMPT.u32
--capture-hidden yes --cycles 4 --materialize-prefix N` records a fresh continuation
with the same context-capture and prefill chunk boundaries. Repeat with
`--restore-prefix N` to snapshot at a page-aligned prefix before the prompt
ends, release the request, move to another native state lane, restore the real
composite snapshot, rebuild draft rings from its stored context window, and
prefill the remaining identical suffix. N must be positive, page aligned, less
than prompt length, and width must be one. KV pages remain resident in this
initial experiment; it measures composite state/ring restore, not SSD restore.

Compare with `python -m dev.benchmarks.dflash_context_compare fresh.jsonl
restored.jsonl --output report.json`. It checks model/prompt/precision/policy
identity and pairs only identical logical position, lane and complete context
hash (including the pending anchor). Divergent later trajectories are counted
as unmatched, not treated as causal restore differences. Complete-file,
proposal replay and censored acceptance validation reuse #550's analyzer.
Optional BF16 final draft hidden rows measure max/RMS/relative-L2 drift; this
is an observable downstream of context reconstruction, not a direct dump of
context-window quantization error. Target-prefix changes are reported too.

Default profiling and serving paths are unchanged. The hidden accessor exists
only in instrumented builds. Capture host copies/file I/O invalidate serving
latency measurement. `--verify-budget all` is the default controlled scoring
probe. `--verify-budget auto --budget-learning frozen` resets the acceptance
prior/cost seed before each cycle, isolating confidence-driven budget changes
from online wall-time learning. `--budget-learning live` keeps normal online
learning, whose different observed timings confound a causal restore comparison.
The comparator requires equal budget modes and materialization boundaries.
Native accepted scratch may match beyond the verified/retained prefix; those
suffixes are censored rather than counted as accepted proposals. The fresh control must use `--materialize-prefix N` equal to the restored
boundary; the comparator rejects mismatched materialization plans. This
controls prefill chunk/context capture boundaries when interpreting drift.

Use independent retrieval/edit/tool prompts at 2K/8K/32K/128K where native
memory planning fits. Do not use repeated filler to claim retrieval quality.
Compare INT8/BF16 KV as separate matched pairs. A short mechanics smoke run
cannot select global memory or a larger trained window. SSD/suspend/mixed-lane
matrix, direct ring reconstruction oracle and trained architecture/serving
comparisons remain gates in #548; no degradation or benefit is presumed.

## Local evidence

`results/dflash-restored-context-m5-pro/manifest.json` pins 12 compressed native
captures (48 cycles, 24 matched-context pairs), prompts, reports, file hashes
and the native capture source patch. The prompt text uses actual Splash source
excerpts and asks for a source review; these are mechanics probes, not a broad
retrieval, editing or task-quality benchmark. Both targets use the pinned MLX
4-bit checkpoints and released DFlash2 weights described in #550.

| Target / prompt / KV | Verify budget | Max hidden relative L2 | Max confidence difference | Proposal/retained changes |
| --- | --- | --- | --- | --- |
| dense / 2112 / INT8 | all | 0.04373 | 0.04075 | 0 / 0 |
| dense / 8256 / BF16 | all | 0.04763 | 0.09203 | 0 / 0 |
| MoE / 2112 / BF16 | all | 0.05658 | 0.08729 | 0 / 0 |
| MoE / 8256 / INT8 | all | 0.04590 | 0.05923 | 0 / 0 |
| MoE / 2112 / BF16 | frozen prior/seed | 0.05588 | 0.17532 | 0 / 0 |
| MoE / 8256 / INT8 | frozen prior/seed | 0.04608 | 0.07059 | 0 / 0 |

Every row contains four cycles per treatment. Max confidence difference is in
absolute probability units, across all seven proposal positions; many suffix
positions were never retained. In the MoE 2112 frozen-budget pair, one matched
cycle chose six verify rows fresh versus five after restore. Both retained
three tokens. This motivates measuring calibration and cycle cost after
restore, rather than interpreting confidence drift as lost acceptance.
The treatment includes snapshot restore AND state-lane movement; these data
do not isolate context-window quantization from every other numerical effect.

No proposal or retained-length degradation was seen in this small matrix. It
cannot establish that global memory, a larger window, or a precision change
would help. Timing, SSD/suspend restoration, larger contexts, independent task
quality, and real trained-checkpoint comparisons remain unqualified.

Recompute one report with:

```sh
python -m dev.benchmarks.dflash_context_compare \
  dev/benchmarks/results/dflash-restored-context-m5-pro/dense-2112-int8-fresh.jsonl.gz \
  dev/benchmarks/results/dflash-restored-context-m5-pro/dense-2112-int8-restored.jsonl.gz \
  --output /tmp/context-report.json
```

To recapture, use the archived `.u32` prompt, the manifest's model root/fingerprint,
four cycles, its KV format and verify-budget/learning settings; use
`--materialize-prefix 2048` (or 8192) for fresh and `--restore-prefix` at the
same boundary for restored. `--capture-hidden yes` is required for hidden drift.
The archive integrity/reproduction test checks every file digest and all six
reports. Production `Runtime.cpp` preprocesses identically to upstream main
(SHA256 recorded in the manifest); all added instrumentation is compiled out.

Extra validation: a B3/two-cycle capture passed with
`MTL_SHADER_VALIDATION=1`. Enabling Metal API validation as well on this host
hit a `prefill_gdn_scan_vh32` threadgroup-memory assertion (35072 > 32768)
in the unchanged target kernel. This combined-validation run did not pass;
it is not included in the successful matrix. See `validation.json`.
