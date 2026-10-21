# Optional GDN verification prefixes with pending tapes

**Versioned historical experiment:** runtime instructions and results below apply to their pinned legacy revisions. The runtime flags/build targets are not current-main settings. See [the October 21 current-main audit](current-main-audit.md) and [current qualification recipe](qwen-maintainer-runs.md). The local experiment and awake replacements are complete; see the [completion ledger](deepinfer-qwen-completion.md) for measured decisions. Current-main validation is reported separately.


This is step 3 of the [Qwen roadmap](deepinfer-qwen-plan.md), tracked in
[#508](https://github.com/incoai/splash/issues/508). It adds a separately
switchable arithmetic experiment on top of the greedy MoE prefix in #507.
`SPLASH_LIVE_GDN=1` enables it only where that existing policy runs.
The default is off. Sampled lanes retain eight eligible rows; constrained,
dense and forced thinking-end batches use the existing full path.

## State and tape invariants

A decode step writes the state **before its own rows**, after folding all
retained pending rows from the previous step. It leaves its current rows on
the step tape. Acceptance chooses how many of those rows the next step must
fold into the state. Current prefix length and previous pending length are
independent: pending 8/current 1 must still consume all eight pending rows.

The live shader keeps the eight-row tape capacity, slot offsets, state-cell
layout and projection geometry. It changes only the current GDN arithmetic:

- The prologue always loads every retained pending row. Inactive current-row
  SIMD groups copy any convolution carry they own, then skip convolution,
  normalization, gates and current tape writes.
- All eight SIMD groups enter both threadgroup barriers and the scan. They
  partition the recurrent state's dimensions; an inactive row group cannot
  leave the kernel early. The pending scan runs in full; the current scan
  stops at that lane's live count.
- Inactive hidden rows and the shared rows feeding Table16 are explicitly
  zero. Their packed projections can be poisoned without being read by the
  GDN gate. Full-width downstream consumers receive defined hidden/table data.
- Current tape suffixes may remain unwritten. They are never retained under
  the existing acceptance prefix, and snapshot/materialization consumes only
  retained pending rows. Invalid counts conservatively execute all eight rows.

`GdnDecodeBuffers::liveRows` is optional and appended to the existing aggregate.
The host validates one 32-byte `VerifyLiveRows` per active lane. It selects a
separate `_live_rows` entry for each norm/state/preparation/geometry variant;
the ordinary entries pass a compile-time full count. Plain live entries append
metadata at buffer 15 and parameters at 16. Table entries append metadata at
17 and parameters at 18. Existing ordinary bindings stay unchanged.

`QwenTargetVerifyBuffers::gdnLiveRows` is distinct from the MoE routing switch.
This allows an incremental comparison with identical MoE prefixes. When a
width first constructs a live graph, QwenTarget prepares its ordinary GDN
variants in an unsubmitted graph too. This prevents later sampled/constrained
fallback requests from compiling an ordinary pipeline during serving. It
does not execute a second recurrence or change parity. A real-model oracle
caught the missing preparation in the first implementation; the corrected
path passes that startup invariant.

## Verification

The shader-validated GDN suite exercises 128 combinations: two compiled Qwen
geometries, B1–B4, FP32/BF16 norm storage, FP32/BF16 recurrent state, grouped/tiled
heads and plain/Table16 output preparation. Each case exercises 130 steps,
including every live count, every retained prefix, full/short transitions,
both tape parities, zero/invalid metadata fallback and poisoned rejected
suffixes. Active hidden rows, each live tape field and the entire folded
state match the ordinary shader bit for bit. The ordinary shader is separately
checked against the existing independent CPU recurrence and prefill gate.
Inactive hidden rows are zero and full prepared tables match independent
preparation from those hidden rows. A short metadata extent is rejected before
encoding a graph.

Changing/permuted batches and independent lanes remain covered by the existing
GDN replay suite. Real model oracles exercise changing widths, retained tape
materialization, RAM/disk restore, lane movement, preemption, seeded sampling,
mask overlap, EOS/token limits and thinking transitions. Results are recorded
per target/storage/state/KV cell; one format does not qualify another.

```sh
make -j8 build/engine-tests/gdn-decode build/engine-tests/model-runtime-oracle \
  build/engine-tests/decode-profile build/splash
MTL_SHADER_VALIDATION=1 build/engine-tests/gdn-decode build/splash.metallib
MTL_SHADER_VALIDATION=1 build/engine-tests/model-runtime-oracle \
  build/splash.metallib /path/to/installed-model --kv-format int8 \
  --state-format fp32 --live-verify-threshold 0.1 --live-gdn on
```

## Performance gate

Compare `--live-gdn off/on/on/off` using the same frozen `decode-profile`,
threshold, model, context and state/KV formats. Warm the pipelines first. Retain
committed tokens, actual acceptances and eligible rows at every width. Then
compare production serving binaries with the same threshold in both arms and
`SPLASH_LIVE_GDN=0` versus `1`. Tracing and shader validation are disabled for
speed runs. Freeze executables and metallibs throughout each experiment.

GDN was only about 4–7% of the initial attributed cycle, and previous-pending
work and state traffic remain. A kernel speedup alone cannot establish a useful
serving gain. The experimental flag stays off unless complete serving runs
clear `max(2%, 2 * measured noise)` without correctness/latency regressions;
spread above 5% is inconclusive. Other Apple GPU families remain unexecuted
until the [maintainer package](qwen-maintainer-runs.md) runs there.

Local performance qualification is complete. The [M5 Pro evidence bundle](results/qwen-gdn-live-prefixes-m5-pro/README.md)
contains 4,096 native cycle samples, production HTTP comparisons, disabled-path
controls, source/model/build identities and five passing shader/oracle logs.
GGUF and warmed MLX native gains are 0.11–1.20%, below the 2% minimum; the
initial MLX series also has no gain above its noise-adjusted gate. No HTTP trial
clears its noise-adjusted gate. Keep the historical flag off. PR #511 is closed as an archived experiment;
#523 retains its evidence. This is a measured decision against promotion.
