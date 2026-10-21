# Runtime bottleneck measurements

**Historical data:** these measurements apply to the source/build identities in this bundle. Current main has since merged a different runtime stack; see [the source reconciliation](../../current-main-audit.md). They are not current-main qualification.


This supplies the per-width operator table for step 1 of the
[roadmap](../../deepinfer-qwen-plan.md). The initial attribution logs are
**exploratory diagnostics**, not speed qualification. Their session began on
battery and ended on AC; per-profile power was not pinned. They use the initial
main baseline `be3b083c5f413673d44e44881042e875e674e333`, 2,048 prompt tokens,
INT8 KV, FP32 state, and the GPU execution path. The log hashes and complete
pipeline tables are retained in `operator-tables.json`.

Ordinary fused GPU medians use four cycles per width for MoE GGUF IQ3_XXS and
three for dense MLX 4-bit. Attribution then submits dispatches separately.
Those serial replay fractions identify expensive operator families; they are
not percentages of ordinary fused serving time or predicted savings. The
small samples and exploratory power conditions do not qualify regressions or
cross-format/device comparisons.

| Target / width | Ordinary fused GPU ms | Serial expert share | Serial other-linear share | Serial GDN share |
| --- | ---: | ---: | ---: | ---: |
| MoE GGUF B1 | 21.78 | 39.2% | 43.9% | 4.3% |
| MoE GGUF B2 | 30.73 | 46.4% | 34.1% | 5.2% |
| MoE GGUF B3 | 38.01 | 48.3% | 31.5% | 5.6% |
| MoE GGUF B4 | 42.49 | 49.7% | 28.0% | 7.0% |
| Dense MLX B1 | 69.33 | — | 92.8% | 2.8% |
| Dense MLX B2 | 74.07 | — | 89.3% | 4.6% |
| Dense MLX B3 | 85.10 | — | 87.4% | 5.9% |
| Dense MLX B4 | 87.12 | — | 84.9% | 7.1% |

Expert is `moe_expert*`; other linear is the listed `gguf_decode*` kernels,
including packed affine kernels used by MLX weights and shared draft/head
work. It is not a target-only projection measurement. GDN is `verify_gdn*`.
The JSON separately retains attention, routing and other operations, with
every original pipeline/count/GPU-time row, avoiding ambiguity from grouping.

These measurements motivated selective MoE work first and a separately gated
GDN experiment. The latter's later AC/native/HTTP comparisons fail promotion;
the exploratory operator share alone does not justify it. Dense targets spend
most replay time in linear operations, so a MoE-only eligibility policy does
not establish a dense speedup or justify changing the shared eight-row ABI.

The later ordinary serving traces and per-parent host analysis are documented
in [serving measurements](../../serving-measurements.md) and
[host-overhead evidence](../qwen-host-overhead-m5-pro/README.md). Their host
scopes overlap, ANE ready/report spans are host notifications, and GPU command
spans can include event waits. Do not sum those phase medians or mix their
percentages with this serial replay table. The dense dual-server trace's
serialized admission is excluded from contended-latency claims; an awake,
single-resident ANE contention trace is queued separately.
