# Qwen3.8-27B on the 48 GB M5 Pro

Investigation base: `3135ab45` (2026-10-24). Results below are local
measurements, not promises for another GPU, checkpoint or power source.
The working branch is `perf/qwen38-m5-investigation`.

Two local runtime changes are retained: measured M16 decode tiles and a
short-prefill FFN instance. Complete-cycle counting throughput improved
**2.32–2.45% in two HTTP ABBA runs**. Cold 64/160-token prompts saw
**5.17%/9.95% lower TTFT** in a separate paired test. Both whole HTTP
performance checks remain **inconclusive at cold 2K TTFT**; representative
coding timing did not qualify a general speedup. These limits accompany
the positive cells throughout the evidence.

The ten prioritized workstreams are consolidated upstream in
[#580](https://github.com/incoai/splash/issues/580), with implementation and
release qualification in [#581](https://github.com/incoai/splash/issues/581)
and [#582](https://github.com/incoai/splash/issues/582), and existing
acceptance/adaptation research in [#542–548](https://github.com/incoai/splash/issues/542).
The original ten fork tickets are historical records. Eight bounded
investigations completed; release qualification and held-out adaptation
remain open. Portable summaries, raw measurements and experiment sources
are under [`results/qwen38-m5-pro`](results/qwen38-m5-pro/summary.json).

Publication splits the tooling and two runtime changes into separate PRs.
The measurements below belong to the pinned combined experimental patch;
branch-specific validation is listed in each PR. The source archive retains
the exact measured patch rather than relabeling it as a publication commit.

## What the implementation actually does

The reference is the installed `mlx-community/Qwen3.8-27B-4bit`, target
revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, with
`incoai/Qwen3.8-27B-DFlash2` revision
`015e795645c74b1a0eeef3b570031fb62e769bc5`. MLX storage does not mean MLX
execution: SafetensorsImage prepares affine block planes and both GGUF and
MLX targets execute through LinearGguf and the same Metal backend.

The target has hidden width 5120, intermediate width 17408, vocabulary
248320 and 64 layers. Every fourth layer has full attention (24 query
heads, four KV heads, dimension 256); the other 48 layers use GDN (48
value heads, dimension 128). The five-layer draft consumes target layers
5, 19, 33, 47 and 61. It has a 2048-token sliding window. Its eight input
rows are one known anchor and seven masks. Dynamic local convolutions,
attention, FFNs, the shared vocabulary projection and the rank-256
top-16 candidate selector produce seven proposals in parallel.

`Runtime::decodeAsync` builds one command for an unconstrained cycle:
RoPE tables → draft embedding/backbone/head/selector → verify input and
embedding → all target layers/head → target policy and rejection
correction → target auxiliary commit and draft context commit. Structured
constraints add a host mask round trip. A shortened output budget or
verify-row limit does not shrink the dense projection graph.

GDN already avoids a second full state update: the next cycle first folds
the preceding cycle's retained tape into the state, then computes this
cycle's trial outputs. Gate/up projections already fuse on this GPU.
Staged projection kernels already prefetch weight payloads and metadata,
spread their K walks and select split-K counts. Activation producers already
prepare the register tile's input representation when needed.

KV extents and weights are held resident through Metal residency sets;
the command queue holds the indirect KV extents, rather than every weight.
The memory governor separately accounts for engine allocations and host
headroom. A cached recurrent state holds one cell; an active lane needs
two, plus draft rings. Replacing residency with blanket mlock, emptying OS
caches, or increasing the wired-memory limit is not an optimization plan.

Relevant source map:

| Path | Responsibility |
| --- | --- |
| `runtime/model/{ModelDescriptor.mm,ModelFactory.cpp,SafetensorsImage.cpp,GgufImage.cpp}` | Shapes, preparation and model loading |
| `runtime/model/{Qwen3_8.hpp,QwenTarget.cpp}` | Target geometry and layer execution |
| `runtime/model/{DFlashDraft.cpp,Runtime.cpp,VerifyBudget.cpp}` | Draft, exact verification, state commit and MoE depth policy |
| `runtime/ops/{Linear.cpp,LinearGguf.cpp,ExecutionPlans.cpp}` | Projection planning, storage and scratch bounds |
| `runtime/metal/kernels/common/{gguf_staged_tile.h,gguf_staged.h}` | Weight staging, prefetch and matmul loops |
| `runtime/metal/kernels/{shared/gguf_linear.metal,decode/gdn.metal}` | Dense kernels and recurrent verification |
| `runtime/ops/{PagedAttention.cpp,DraftSelector.cpp,Sampling.cpp}` | Attention, candidate selection and sampling contracts |
| `runtime/metal/{CommandGraph.hpp,MetalBackend.mm,Residency.hpp}` | Host graph, resource lifetime and command encoding |
| `runtime/engine/{Engine.cpp,Scheduler.cpp,MemoryPlan.cpp,MemoryGovernor.cpp}` | Serving, batching and memory admission |
| `dev/benchmarks/{decode_profile.mm,backend_benchmark.mm,http_regression.py}` | Attribution and complete-cycle measurements |

## Cost model and initial evidence

For conditional proposal acceptance probabilities p1…p7, expected emitted
tokens per ordinary cycle are

```
E[T] = 1 + p1 + p1*p2 + ... + p1*p2*...*p7
throughput = E[T] / (draft + verify + commit + host)
```

Eight rows reuse the same dense weights. Affine 4-bit/group-64 storage is
4.5 bits per weight, so eight useful matrix rows provide about 28.4 FLOP
per weight byte (before padding and other traffic). A few-row projection
can still be limited by staging latency and occupancy rather than peak
matmul throughput. Increasing tile height or K staging is useful only if
its additional storage/register pressure does not remove that overlap.

The two 248320x5120 heads read about 1.43 GB of affine weights per cycle.
At the M5 Pro's nominal 307 GB/s, weight reads alone require at least
4.66 ms, compared with the observed attributed 5.10 ms. Fusing a logits
store/selection can remove some activation traffic but cannot remove those
weight reads. An approximate draft-head shortlist would be a different
experiment: saving one 2.5 ms head from a 68 ms cycle only wins if expected
retained length loses less than roughly 3.7%. Exact target verification and
the proposal probabilities used for rejection correction must remain.

For a fixed sequence of conditional acceptances, early improvements have
larger leverage: `dE[T]/dp_i = sum(j=i..7, product(k<=j,k!=i,p_k))`. This motivates
measuring reached prefixes and testing existing prefix-survival training
work before making the draft backbone or candidate support larger.

Use observed retained tokens and full wall time to evaluate a change.
Proposal acceptance alone omits draft cost and token clipping. A change in
floating-point accumulation can change future proposals, so unchanged
kernel error bounds alone do not establish unchanged serving output.

The first run used macOS 26.6, M5 Pro (20 GPU cores, 18 CPU cores),
48 GiB unified memory, normal power mode **on battery**. Other desktop
applications were open; swap was already in use. These are exploratory
measurements. Preserve power/thermal/memory provenance in paired runs.

At 512 prompt tokens, five cycles per width:

| Lanes | Fused GPU ms/cycle | Wall ms/cycle |
| --- | ---: | ---: |
| 1 | 67.89 | 68.43 |
| 2 | 71.66 | 72.26 |
| 3 | 83.30 | 83.82 |
| 4 | 86.90 | 87.51 |

One unconstrained cycle has 626 dispatches in one command. At B1 the
separately replayed attribution is 39.0% gate/up, 24.8% residual
projections, 19.0% fused input projections, 7.4% the two vocabulary heads,
and 3.1% other staged projections. GDN is 2.8%, norms about 1.7%, and
short-context attention under 1%. Attribution replay changes execution;
these shares identify targets, not additive fused timing guarantees.
The fused wall/GPU difference is only 0.54 ms: even eliminating it entirely
would improve this workload by less than 0.8%.

One FP32 recurrent cell has 48*48*128*128*4 = 144 MiB of recurrent
elements, plus convolution history. BF16 saves 72 MiB per cell, but changes
recurrent rounding. Target INT8 KV needs 32.5 KiB per token before
extent/alignment overhead, versus 64 KiB in BF16. At 32K, context KV is
about 1.0 GiB versus 2.0 GiB. Format changes trade accuracy and capacity
against bandwidth; benchmark them explicitly rather than silently changing
precision.

Initial native decode/short characterization: three samples, ANE disabled,
max context 32768; all benchmark invariants passed. Raw output, progress
and preserved baseline executables are under
`build/qwen38-investigation/` (ignored local artifacts).

## Prioritized issues and decisions

Each item is a bounded experiment. A measured rejection is a useful result;
it is not permission to claim a speedup. Existing upstream research and
earlier local experiments are controls, not evidence for the current head.

| Order | Experiment | Required decision | Outcome |
| --- | --- | --- | --- |
| [1](https://github.com/incoai/splash/issues/580) | Preserve a current-head baseline and reproducible complete-cycle measurements | Separate throughput, acceptance, GPU/host cost, power and memory effects | baseline/tooling complete |
| [2](https://github.com/incoai/splash/issues/581) | Sweep affine-4 decode tile height and split-K at real 27B shapes | Qualify a measured tile policy with bitwise outputs, storage bounds and B1–B4 serving | M16 candidate retained; repeatable B1 gain; cold2K release qualification OPEN |
| [3](https://github.com/incoai/splash/issues/580) | Ablate one-pass gate/up and weight-stage scheduling | Test memory-latency overlap and occupancy without reassociating sums | one-pass retained; K64 staging rejected |
| [4](https://github.com/incoai/splash/issues/580) | Measure the shared vocabulary head with its actual FP32 output | Add missing benchmark fidelity before considering head specialization | FP32 benchmark added; existing head policy retained |
| [5](https://github.com/incoai/splash/issues/582) | Calibrate ANE prefill and chunk-size crossover on this 48 GB machine | Compare short TTFT and sustained prefill including ANE memory cost | short SG16 retained; automatic ANE retained |
| [6](https://github.com/incoai/splash/issues/580) | Measure INT8/BF16 KV and FP32/BF16 recurrent-state tradeoffs | Report speed, state/output changes and real working-set cost | precision characterized; existing defaults retained |
| [7](https://github.com/incoai/splash/issues/580) | Profile long-context attention and split choices | Optimize only where context traffic is a measured material cost | partition study complete; existing attention retained |
| [8](https://github.com/incoai/splash/issues/580) | Bound graph construction, binding and allocation overhead | Pursue reuse only if a repeatable serving gain exceeds its lifetime/complexity cost | host bound measured; graph reuse deferred |
| [9](https://github.com/incoai/splash/issues/580) | Evaluate dense verification truncation against its physical cost | Quantify why/when fewer retained rows save work; preserve exact acceptance and pending tapes | dense truncation rejected; full verification retained |
| [10](https://github.com/incoai/splash/issues/542) | Diagnose acceptance headroom against existing draft-training work | Use reached-prefix evidence; avoid duplicate training/architecture proposals | held-out/training and narrow precision adaptation OPEN; full BF16 draft rejected |

Every promotion requires repeated paired measurements, actual token counts,
no unexplained transcript/state differences for a scheduling-only change,
and relevant CPU/Metal/model-oracle checks. Numerical changes additionally
need same-prefix logit comparisons. No extrapolation to Apple9 or other M5
core counts from this one machine.

## Experiment decisions

### 1–2. Baseline fidelity and decode tiling

The new decode rule is restricted to raw Apple GPU family 10, 20 GPU
cores, one lane, unsplit, unrotated affine-4/group-64 projections, K=5120
and N in {14336,16640,17408}. It selects existing M16 kernels, including
matching gate/up plans. `ProjectionShape` carries the exact format
classification so allocation and dispatch reserve the same padded rows.
Other devices, formats, widths and multi-lane plans keep their policies.

Cold-weight alternating sweeps measured:

| Projection | Existing M8 ms | M16 ms | Decision |
| --- | ---: | ---: | --- |
| FFN gate/up,17408x5120 | 0.3723 | 0.3543 | Candidate |
| Fused GDN input,16512 active/16640 stored | 0.1998 | 0.1946 | Candidate |
| Fused attention input,14336x5120 | 0.1904 | 0.1832 | Candidate |
| Down residual,5120x17408,split2 | 0.2025 | 0.2061 | Keep M8 |

The FFN-only native counting ABBA gained 1.56%, below the declared2%
promotion floor. Including both measured fused-input shapes gained 2.30%
with 0.48% round spread and matching tokens/draft counts. B2 was neutral;
B3/B4 timing drift was over 5% and is inconclusive. Those widths do not
select this new one-lane rule.

A stronger HTTP/native ABBA, four fresh servers and four counting samples
per arm, measured complete decode-cycle milliseconds per emitted token:
8.55892 baseline versus 8.35452 candidate, or **2.45% higher throughput**,
with 0.60% round spread. Outputs and loaded weight bytes matched. Cached
TTFT was near neutral. Cold 2048-token TTFT had 7.43% spread, so the **whole
HTTP performance run was inconclusive/failed its aggregate gate**. The
positive decode cell is not a claim that every workload improved.

The final combined-candidate rerun reproduced this result:8.60803→8.41276
complete-cycle ms/token, **2.32% higher throughput**, with 0.16% round spread.
Cold 256 and cached 256/2048 cells passed regression checks. Cold 2048 again
exceeded the 5% noise ceiling (6.31%), so aggregate `performance_pass`
remains false. This is the remaining release-qualification item in [#581](https://github.com/incoai/splash/issues/581).

The 64-request representative transcript ABBA also matched every output,
including code, Chinese prose, math,8K and 16K contexts. Most cells had
7–13% round spread. Even16K's apparent3.21% gain failed to exceed twice
its 1.81% spread. These timings do not establish a general coding speedup.
A shared-weight, separate-state in-process control measured B1+1.90%
runtime wall and B2–B4 near neutral; it is below the 2% promotion floor.

The backend benchmark now records the engine's batch-completion
`decode_cycle_ms`, output counts and accepted/drafted counts. Counting
adds `aggregate_decode_cycle_tokens_per_second`; transcript records retain
GPU time as a separate field. The observed totals agree. Earlier frozen
transcript tools measured GPU time only, which is labeled in the evidence.

### 3–4. Gate/up staging and the FP32 vocabulary head

`gguf-projection-benchmark` now accepts `output=f32` for a single plain
quantized decode projection, allocates the correct output bytes and reports
the type. `gateup=both` compares the existing one-/two-pass schedules at
equal splits within the same alternating, rotating-weight case set.
Invalid FP32/prefill/epilogue combinations fail before opening Metal.

The 25-round FP32 head sweep measured B1 M8/split1 at 2.4920 ms, M16 at
2.5271 ms and the best register tile at 2.8390 ms. Extra splits lost. Keep
the existing head policy. B2/B3/B4 M8 head times were 2.5363/2.7672/2.7626 ms.

Gate/up two-pass/one-pass measured 0.4025/0.3721 ms on M8 and
0.3939/0.3557 ms on M16 at B1. The current one-pass schedule remains.
A generated, isolated AF4 K-stage64 prototype doubled shared staging from
K32 and lost: down split2 B1 rose0.2031→0.3305 ms, B4
0.2318→0.4357 ms in ABBA microbenchmarks. Reject it. It also changes
accumulation order and was never numerically qualified. Larger staging
can reduce latency hiding when shared-memory/register occupancy falls;
more bytes per fetch does not imply more useful bandwidth.

### 5. Short prefill and ANE

The production short-prefill candidate uses eight 16-row SIMD groups
instead of four 32-row groups, retaining 128 rows per threadgroup. It is
restricted to the same 20-core/family 10 device, AF4G64 unrotated 27B FFN
shapes, and 33–192 active rows. Partial groups round to 16 rather than32;
shorter matrix chains also change occupancy. These are explanations
consistent with the sweep, not direct hardware-counter measurements.

Three extra kernel instances share the original K loop and are prepared
before readiness. Active outputs match the original kernels bitwise,
including fused segments and partial tiles, under Metal shader validation.
The CPU policy test covers32/33/192/193 row boundaries and other hardware,
shapes and formats. Larger chunks keep the original instance. A64-row
SIMD-group prototype was about four times slower at large chunks and was
rejected.

A fresh-server HTTP ABBA compared against the frozen M16-only build,
eight unique cold prompts per context per arm. All 192 responses matched,
and every measured prompt reported zero cached tokens:

| Prompt tokens | Original TTFT ms | SG16 TTFT ms | Latency reduction | Round spread |
| --- | ---: | ---: | ---: | ---: |
| 64 | 194.52 | 184.45 | 5.17% | 1.28% |
| 128 | 272.04 | 269.88 | 0.79% | 1.76% |
| 160 | 411.46 | 370.52 | 9.95% | 2.14% |
| 192 | 431.38 | 413.15 | 4.23% | 2.09% |
| 193,original policy | 465.48 | 470.43 | -1.06% | 2.03% |
| 256,original policy | 498.80 | 505.63 | -1.37% | 2.79% |

64 and 160 exceed the 2%/twice-spread gain gate;192 narrowly exceeds it.
128 is neutral. The unchanged-policy controls stay within measured timing
variation. The result supports short prompts, not a gain at every length.
The isolated projection sweep found5–26% gains across33–160 rows;
those percentages must not be presented as end-to-end speedups.

Three-sample ANE characterization at 480–2016 rows compared GPU-only,
fixed shares rounded to 12/34 and 16/34 intermediate blocks, and automatic
calibration. Auto selected 13/34 (38.24%), minimum 512; an earlier run chose
14/34 (41.18%), also512. At 512 rows GPU-only took 995 ms versus 817 ms auto;
at 2016,3892 ms versus 3092 ms. Neighboring fixed shares did not win
consistently. These are characterization runs, not paired evidence between
nearby shares. Keep automatic calibration and its crossover.

The reported ANE allocation is the engine's **Metal-budget** charge:
261,685,248 or277,676,032 bytes for those two automatic outcomes, versus
zero GPU-only. It does not measure the entire ANE service/OS footprint.
Prefill throughput, initialization and memory admission must be considered
together; allocating more ANE work is not free capacity.

### 6. Precision versus capacity

The existing INT8/BF16 KV and FP32/BF16 state options were crossed at 512
and 8192 prompt tokens, three cycles per width B1–B4, with allocation/power/
thermal/VM provenance. These sequential profiles characterize cost; they
are not paired evidence of a precision speedup. At 8K, B1 INT8/FP32 took
69.22 ms GPU versus 74.61 ms BF16/FP32. Some short-profile cells drifted,
so use the raw data rather than attributing every difference to precision.

KV storage before extent/alignment overhead is
`16 * 2 * 4 * (256+4) = 33280 bytes/token` for INT8 (FP32 scales), versus
65536 for BF16. At 32K this is1.015625 GiB versus 2 GiB. BF16 recurrent state
saves 72 MiB per cached recurrent cell and 144 MiB per active lane's two cells;
convolution history, tapes and draft rings are additional. Cache-heavy
sessions may value that capacity even if short decode gains are small.

Same-prefix next-token logits use129 rows after 2048/8192-token cold
prefills, the same affine target weights, and BF16-KV/FP32-state as the
numerical reference. INT8/FP32 top-1 agreement was 127/129 and 129/129;
mean KL was 0.000173 and 0.000052. With INT8 KV held fixed, changing state
to BF16 gave128/129 and 129/129 top-1 agreement, mean KL 0.000049/0.000062.
Individual low-probability logits can move much more than mean probability
metrics suggest (default versus reference maximum centered logit difference
10.72 at 8K). Full RMSE, maximum KL/TV and NLL differences are preserved.
This source-text probe is a sensitivity check, not a quality/perplexity
benchmark. At 32K, default INT8/FP32 matched 129/129 reference top tokens,
mean KL 0.000400, maximum per-row TV 0.0409. With INT8 KV held fixed,
BF16 state matched 128/129, mean KL 0.000443. The scored positions are
correlated and from one source-text prefix per context. Keep the existing
precision defaults.

Fresh greedy serving probes crossed all four formats with code, math, Chinese
prose, plain JSON and thinking-mode code, each cold and restored from a
32-token cached prefix (256 output tokens/request). All 40 candidate
responses matched pristine 3135ab45 in message hashes, usage, speculative
counters and cache reuse. Ten of 20 fresh/restored pairs differ **within both
baseline and candidate**. This reproduces existing chunk-boundary numerical
variation and must not be presented as a regression from these kernels.
Each domain has one authored prompt; these are controls and diagnostics,
not held-out quality scores.

### 7. Long-context attention

Whole-model INT8/FP32 profiles measured B1 fused GPU cycles of 69.22 ms at
8K and 79.83 ms at 32K. The separately replayed attention split reached 3.5%
and 9.9% respectively; at 32K B4 it reached 24.1% (125.27 ms fused cycle).
Attention is therefore increasingly relevant, while short-context projection
work remains the first target. Attribution replay is not an additive fused
cost decomposition.

The one-layer production store+attention sweep covered0/512/2K/8K/32K/131K
histories, B1/B3/B4 and both KV formats. At 131K, B1 INT8/BF16 took
1.820/2.559 ms, B4 6.772/9.817 ms. This is synthetic layer work, not full-model
throughput. Scratch is reserved for 128 splits per lane; the current rule
uses at least32, approximately one split per 512 visible tokens, capped128,
and depends on each lane's history so batching preserves its arithmetic.

An isolated CPU-plan override swept16/32/64/128 splits and repeated the
production default before/after. Default round spread was 0.04–1.54%.
At 8K B1,16 splits took 0.1578 ms versus 0.1754 ms default, but its whole-cycle
headroom is under1%. At 32K B4,32 splits took 1.8063 ms versus 1.9144 ms
(about 1.4% projected whole-cycle headroom using the attributed share).
No fixed count won across contexts and widths. Nondefault partitions changed
some BF16 output bits (maximum active difference0.0009765625 in these
cases); none received model-level qualification. Retain the production
rule. A131K deployment still needs complete-cycle/acceptance measurements
before any numerical attention change is promoted.

### 8. Host graph cost

One ordinary B1 counting sample measured 595.856 ms GPU,599.018 ms
submit/completion wall, and 601.145 ms complete-cycle time over 9 batches.
Even removing **all** time outside the GPU saves only 0.88% latency there;
B1–B4 samples ranged0.71–0.88%. Graph creation/binding is only part of
that gap. Persistent graph reuse would add buffer/parameter/resource
lifetime complexity without demonstrated headroom. Retain the current
path. This bound does not cover structured constraints' host mask work.
A prior archived graph-cache experiment also lacked a qualified serving gain.

### 9. Dense verification depth

`budgetsVerifyRows` enables adaptive row budgets for sparse MoE only.
The dense graph always computes eight physical rows per lane through its
projections, full attention and GDN trial. The pending tape's retained rows
belong to the preceding cycle. Changing the sampler limit alone removes
none of the dominant projection traffic; MoE's seeded cost curve does not
apply.

All ten October22 archived acceptance captures, decompressed payloads and
reports were checksum-verified. They use the same pinned weights/layout,
but are **historical diagnostic probes**, not fresh-head measurements.
One authored prompt and 64 correlated cycles per domain gives:

| Probe | Full retained tokens/cycle | Yield lost at 4 rows | Yield lost at 7 rows |
| --- | ---: | ---: | ---: |
| Code | 4.34375 | 29.14% | 5.04% |
| Math | 4.78125 | 32.03% | 4.90% |
| Prose | 3.125 | 15.50% | 2.50% |
| JSON | 5.375 | 38.37% | 7.85% |

The calculation is `E[min(T,r)]`. A truncated cycle wins only when its
fractional time saving exceeds its fractional yield loss. The entire GDN
attribution was 2.8% at short-context B1; skipping one trial row cannot
remove the whole GDN cost. Keep full dense verification. A future design
must physically avoid useful amounts of work, preserve rejection correction
and pending tapes, and measure actual conditional costs.

### 10. Acceptance research

The historical first-rejection probes attribute84–96% of dense misses
to selection: the target token is already among the 16 candidates.
On complete-reference code cycles, expected-length replay changed tau
4.344→4.406 and Viterbi4.344→4.531; neither improved every domain.
Scalar edge calibration also had no nondefault value that improved all
probes. These fixed-continuation replays are not alternative rollouts or
throughput results. More selector/backbone capacity is not yet justified.

The next higher-upside study is selector adaptation to the deployed
quantized teacher and prefix-survival objective, with conversation/repository
held-out splits, natural candidate coverage, correct-prefix labels, a
checkpoint rounding control, and native rollout/cycle cost evaluation.
Existing upstream issues 542–548 and draft training work already cover it;
this investigation does not duplicate them or claim training was performed.
The archived score captures lack the hidden features/full target probabilities
needed to train. A dense 27B BF16 teacher alone needs about 54 GB before draft,
activations and optimizer state, so it does not fit this48 GiB machine.
A frozen feature cache or quantized teacher is the practical local route;
larger training needs different resources and a separate quality study.

The new default-format Chinese prose probe used about 148 speculative cycles
for 256 emitted tokens, versus 45 for JSON and 70 for code. These are
different trajectories, not a controlled language comparison. Its support
versus selection misses still need fresh capture; the historical English
miss percentages cannot be applied to it. Thinking/restored context results
are recorded separately rather than folded into a single acceptance rate.

Two generated private loaders also test original BF16 draft tensors already
installed on this Mac, with the shared target head still AF4:

| BF16 draft subset | Sequential screening | Decision |
| --- | --- | --- |
| Context fusion `fc` and selector hidden projection | -3.37% to+5.24% apparent throughput change; positive thinking/cold cell does not recur restored | No consistent or held-out win; retain as research |
| Every draft matrix | About 5–23% slower across all 10 requests, sometimes with fewer cycles;3,878,043,648 draft-weight bytes in native plan | Reject full BF16 inference |

All 20 greedy messages match the ordinary draft's outputs in these controls.
Production loading remains AF4; no training or download was performed.
Higher proposal acceptance is useful only after accounting for added
weight traffic and full-cycle cost. Separate context/selector ablations and
held-out deployment-aware adaptation remain TODO [#542–548](https://github.com/incoai/splash/issues/542).

A longer block also requires a compatible trained checkpoint, selector and
state/tape geometry. Illustratively, uniform conditional acceptance 0.8 gives
expected yield4.161 for 8 rows and 4.859 for 16: only 16.8% more. It wins only
if cycle cost grows by less than16.8%; unused M16 padding alone does not
establish that. This is a future training/architecture hypothesis, not an
inference flag to enable on the seven-proposal checkpoint.

## Validation and reproduction

The latest shader-validated complete projection suite passes, including
44 formats, split-scratch poison, FP32 heads, M8/M16 padding, one-/two-pass
gate/up equality and SG16/SG32 active-output equality. CPU plan tests cover
hardware/format/row boundaries and fused affine/float allocation predicates.
The first older invocation used a production-only library and stopped at a
missing test kernel; the corrected production-plus-test library passes.

Full `make check` passes: native CPU/Metal, architecture, Python and
Ruff/format checks. The Python suite ran 1321 tests with 27 skips. The final
installed 27B runtime oracle passes at both INT8-KV/FP32-state and
BF16-KV/BF16-state, max context 32768, ANE disabled. It covers continuation,
batching, sampling/penalties, preemption, and state/cache restoration. These
oracles ran without shader validation; the separate projection run above
used shader validation. Commands, exit codes, durations and PASS markers
are preserved in the packaged `final-validation.json` record.

Portable evidence consists of `summary.json`, `manifest.json`,
`measurements.jsonl.gz` and `exploration-sources.tar.gz`. The manifest records
original/normalized measurement hashes, source patches, retained executable
hashes, compiler, hardware and model revisions. Checkpoint weights, binaries
and full vocabulary logits are retained locally, outside the source tree's
tracked results. Original raw data remains in the ignored
`build/qwen38-investigation/`. Portable measurements replace machine paths
with `$REPO` and named historical-checkout placeholders; both hashes remain.

Freeze each comparison's binary beside its own `splash.metallib` and
`host-step.metallib`. Build `weight-digests` in the same checkout against
matching private headers/library; `ProjectionShape` changed layout. Rebuild
pristine 3135ab45 in an isolated worktree for the baseline. The archived
`m16-wide.patch` reconstructs the M16-only short-prefill control, and
`current-runtime.patch` reconstructs the combined candidate. Regenerate
isolated prototype sources using the archived generator scripts.

Core commands (one GPU experiment at a time, connected to AC):

```sh
QWEN_ROOT=install/models/mlx-community/Qwen3.8-27B-4bit
make -j6 check
MTL_SHADER_VALIDATION=1 build/engine-tests/gguf-projection \
  build/engine-tests/production-and-test.metallib
caffeinate -is build/engine-tests/decode-profile build/splash.metallib \
  "$QWEN_ROOT" --prompt-tokens 8192 --cycles 3 --kv-format int8 --state-format fp32
build/engine-tests/gguf-projection-benchmark build/splash.metallib \
  af4g64 248320 5120 none 25 output=f32
build/engine-tests/gguf-projection-benchmark build/splash.metallib \
  af4g64 17408 5120 gateup 25 gateup=both
build/engine-tests/gguf-projection-benchmark build/splash.metallib \
  af4g64 17408 5120 gateup 25 prefill=33,64,128,160,192,193,256
caffeinate -is .venv/bin/python -m dev.benchmarks.http_regression \
  --model mlx-community/Qwen3.8-27B-4bit --model-root "$QWEN_ROOT" \
  --baseline-binary build/qwen38-investigation/baseline/splash \
  --binary build/splash --contexts 256,2048 --samples 8 --max-context 32768 \
  --max-memory 34G --max-cache-disk 0 --disable-ane --state-format fp32 \
  --output build/qwen38-investigation/final-http.json
```

The complete one-off commands, warmups, ABBA samples and negative prototypes
are in the source/measurement archives. `abba.py` uses round medians,
noise=max(relative same-build round differences), bound=max(2%,2*noise),
and declares spread over 5% inconclusive. A neutral regression pass is not
a demonstrated gain. No global OS power, wired-memory or cache settings
were changed.

References: [DFlash2](https://inco.ai/blog/dflash2/),
[M5 Pro specifications](https://support.apple.com/en-us/126319),
and upstream research [542](https://github.com/incoai/splash/issues/542),
[543](https://github.com/incoai/splash/issues/543),
[544](https://github.com/incoai/splash/issues/544),
[545](https://github.com/incoai/splash/issues/545),
[546](https://github.com/incoai/splash/issues/546),
[547](https://github.com/incoai/splash/issues/547),
[548](https://github.com/incoai/splash/issues/548).
