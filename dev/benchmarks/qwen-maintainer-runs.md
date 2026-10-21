# Repeatable Qwen qualification on maintainer Macs

This recipe targets main `1482f6bbc2abf12375332725302019594150b6cb` and its
supported `--verify-rows all|auto` control. See [the source audit](current-main-audit.md).
The historical threshold/GDN flags are not current-main settings. Only the
contributor's M5 Pro (20 GPU cores, 48 GiB, Apple10, macOS 26.6.0) is available.
Other GPU families/core/memory configurations remain **unexecuted** until a
maintainer runs them; preparation never means a test passed.

## Freeze and identify the inputs

Use isolated checkouts and the repository build targets for production `splash`,
`engine-tests/model-runtime-oracle` and `engine-tests/decode-profile`. Match
toolchain, configuration and per-target flags between comparison arms. Keep
`splash.metallib` beside each frozen production executable; oracle/profile
commands pass the library path explicitly. Never
rebuild files used by an active server. Record commits, dirty diffs, toolchain,
OS, GPU/core count, memory and power conditions. The runner hashes executable,
library and model-manifest bytes and refuses changes during execution.

Both native builds must be compatible with the server checkout's protocol and
support `--verify-rows`. A protocol-10 historical binary must use its own
matching checkout/server; it cannot be loaded by the newer protocol's harness.
Keep source-version comparisons separate from all/auto comparisons within one
frozen build. To compare just policies, use the same build in both arguments.

Copy [the model list](qwen-qualification-models.example.json), replace all four
paths, and check target/draft revisions against each installed `model.json`.
The runner records actual manifest hashes; example revision metadata is not
proof of arbitrary installed bytes. Keep model bytes, memory, KV/state and ANE
settings constant within a comparison. The MoE GGUF cell is IQ3_XXS and the
dense GGUF cell UD-Q4_K_M. Install through the normal pinned installer.

## Prepare and execute

From the candidate checkout:

```sh
.venv/bin/python -m dev.benchmarks.qualification_matrix \
  --models /path/to/models.json \
  --baseline /path/to/frozen/build --candidate /path/to/frozen/build \
  --baseline-verify-rows all --candidate-verify-rows auto \
  --output build/qualification-apple9
```

This prepares **36 jobs**, each explicitly `unexecuted`: four model/storage
cells, each with INT8/BF16 KV crossed with FP32/BF16 state oracles, B1 decode
and mixed workloads at 2k/8k/31,744 contexts, and B2/B3/B4 decode at 2k.
Serving uses code/math/prose, three trials and 512 output tokens, plus sampled
and strict-JSON probes. The model oracle runs its own documented numerical
contract, including its controlled full-row and budgeted cases. Oracle commands
explicitly select `--max-context 32768 --ane-ffn-share 0`, matching the local
matrix rather than inheriting the target's automatic context/ANE configuration.
Serving retains its recorded native ANE configuration; GPU-only oracles do not
qualify ANE execution.

Review the manifest and repeat with `--execute`; `--suite correctness` and
`--suite serving` permit separate output directories. `--baseline-env KEY=VALUE`
and `--candidate-env KEY=VALUE` record explicit experimental overrides when
the corresponding builds actually support them. No legacy threshold is
silently injected. Defaults compare product `all` against product `auto`;
unlike the old qualification patch, `all` also disables the adaptive fitter.

Single residency is the default: one server per complete ABBA round, closed
before the next. Each has four matched 2k/512-token warmup requests. This avoids
the observed second-server MLX OOM and dense admission serialization on 48 GiB.
Paired residency is optional when capacity permits. Example memory caps are
upper bounds, not guaranteed capacity; record unsupported cells as unexecuted.
B2–B4 at 8k/32k require further memory qualification.

## Validity and interpretation

Serving holds a temporary process-scoped `caffeinate -is` assertion on macOS.
It rejects discrepancies of at least one second between elapsed UTC and the
Mac's awake clock. This caught overnight sleep-affected captures that looked
plausible under native timers. Global power settings are unchanged. Keep power
and thermal conditions stable; wake prevention is not a frequency guarantee.

Measured requests must be cold, produce the full requested output budget and
release resources. Mixed workloads require native long-prefill/continuation
progress while another native request decodes; overlapping HTTP lifetimes do
not suffice. The initial snapshot must be fresh. Later stale polls are retained
with an explicit flag and counted, but cannot prove overlap; no fresh progress
still fails the run. Cached and RAM-prefix continuation are
available directly in `serving_regression.py`; they do not simulate suspension
under resource pressure.

The native all/auto choice is checked through `/status.verify_rows`. Dense
`auto` can correctly report `all`. Greedy hashes can change under adaptive
verification; report them rather than treating fixed seeds as distributional
validation. Timing, complete execution, numerical correctness and quality are
different claims. The runner's `complete` means all child jobs executed, not
that a policy cleared promotion.

No tracing or shader validation is enabled for speed. `--trace` is reserved
for matching instrumented builds and rejects ordinary binaries that would
ignore its environment flag. Current main has no `serving-profile` target;
historical trace analyzers consume the archived experimental captures.

A useful gain must exceed `max(2%,2*measured noise)`; noise >5% is inconclusive.
Retain per-device/per-workload throughput, TTFT and streaming-gap results,
actual counters, cache reuse, ANE configuration, memory, failures and output
hash differences. Never promote from an aggregate hiding a regression. Return
manifests, child reports and logs; omit model weights and generated answer text.
