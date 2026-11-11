# Why DSpine is slow, and which work is worth doing next

This analysis concerns the installed Qwen3.8-27B target and DSpine-b16 draft on
one 20-GPU-core M5 Pro. Kernel profiles are diagnostics. Only experiments that
pass their complete registered request-level gates support throughput claims.

## Useful tokens per target sweep

A speculative cycle pays for the target projections even when later proposals
are rejected. For a block of width `w`, away from termination and output caps,

```
expected emitted tokens = 1 + sum(P(accepted prefix >= j), j=1..w-1)
cost per emitted token = cycle cost / expected emitted tokens
```

A wider block can improve useful tokens per sweep while increasing projection,
attention, recurrent-state and sampling costs. Compare the ratio at the
physical batch width, context length and pending-state history. Overlapping
prefix probes describe the sampled continuations.

The preceding eight-hour study found 1,303 versus 1,062 cycles on the historical
primary corpus for eight/sixteen DSpine rows, with identical complete outputs.
Its controlled serial prototype result was +15.3–16.2% decode throughput.
That prototype motivated implementing fixed geometry in the runtime. The main
report gives the integrated source's independent comparisons.

The same earlier matched-anchor probe found mean first-cycle emitted counts of
5.955 for DFlash2, 5.940 for eight-row DSpine and 9.507 for sixteen-row DSpine
over 67 overlapping eligible primary prefixes. The first two means are close
on this cohort. The issue's four coding prompts cover a different workload.

## Serial projection cost is already substantial weight traffic

The current diagnostic DS16 B1 profile reports a 73.49 ms fused cycle. Separately
replayed dispatches attribute 25.939 ms to 69 AF4G64 gate/up pairs. The logical
payload of those pairs is

```
69 × 2 × 17,408 × 5,120 × (0.5 + 4/64) = 6,918,635,520 bytes
```

Each affine group has 64 four-bit codes plus two BF16 metadata values. Dividing
this logical payload by the attributed time gives approximately 267 GB/s.
Apple specifies 307 GB/s for this M5 Pro. The estimate divides logical weight
bytes by dispatch replay time. Actual DRAM traffic depends on cache reuse,
other traffic, overlap and scheduling in the fused graph.
[M5 Pro specifications](https://support.apple.com/en-us/126319)

Both installed drafts have five layers, hidden width 5,120, intermediate
width 17,408, and 32 query and eight KV heads of width 128. Both invoke the
shared full-vocabulary head before their selectors. DSpine adds bridge writes
and exact-sender refinement. DFlash2 uses grouped convolution and a learned
selector. In this profile,
DSpine's fifteen refinement-score dispatches total 1.177 ms, about 1.6% of
attributed B1 GPU time. That dispatch share gives an upper bound on the work
removed from this replay profile by eliminating that family.

In the 8,193-token profile, five draft attention split dispatches total
1.569 ms, about 2.0% of the 79.40 ms separately
replayed work. Target attention split plus reduction totals 4.646 ms, about
5.9%. Those dispatch shares describe this device, context and geometry.

Prioritize more useful accepted work per target sweep and measured dominant
projection shapes before another bridge or selector rewrite. More precise draft
weights were already screened in the preceding study and gave little cycle
reduction relative to their traffic and storage cost. Testing features from the
BF16 training target requires its original weights.

## Wide batches expose a separate kernel cliff

Three/four sixteen-row lanes select a physical M64 projection tile. The original
one-pass pair accumulates gate and up together. The two-pass path runs each
separately. At B3, the original M64 gate/up family accounts for 177.326 ms
of 273.85 ms separately replayed GPU work. The narrow host policy change retains
the same quantization, K partitions and traversal, and chooses existing separate
passes only for the measured device and AF4G64 MLP shape.

All three registered fixed-width B3 request groups qualify at +73.9%, +75.5%
and +75.1% decode throughput relative to the uncorrected sixteen-row path.
B4 has two clean groups. Its third group lost AC power, leaving the full B4
component comparison unqualified.

The 512-token diagnostic profiles give these fused GPU milliseconds per cycle.
The three probes used different source snapshots and ran outside the controlled
request comparisons.

| Cloned lanes | DFlash2, 8 rows | DSpine, 8 rows | Corrected DSpine, 16 rows |
| --- | ---: | ---: | ---: |
| 1 | 67.87 | 68.99 | 73.31 |
| 2 | 71.83 | 73.48 | 86.84 |
| 3 | 83.55 | 85.32 | 159.30 |
| 4 | 87.27 | 88.58 | 167.26 |

The eight-row cycle times are close in these probes. At four lanes, named
projection families grow from 73.67 ms for DFlash2 to 137.49 ms for sixteen-row
DSpine in the separate dispatch replay. That accounts for about 87% of the
73.21 ms increase in replay time. Dispatches named for the DSpine bridge and
refinement total 2.75 ms. The B3 attribution shows the same concentration in
projections. The archived profile outputs are
`decode-profile-dflash8-runtime16-v12-v1.stdout`,
`decode-profile-dspine8-v9.stdout` and `decode-profile-wide-gateup-v1.stdout`.

The fused profiles imply heuristic break-even emitted-token ratios of 1.87 at
B3 and 1.89 at B4 against eight-row DSpine. Against DFlash2 they are about 1.91
and 1.92. The break-even estimate motivates a batch-dependent policy test that
charges each width for its own cycle cost and useful tokens.

Large accumulator working sets make register pressure plausible. Measuring
allocated registers, spills and occupancy would test that explanation. The
one-pass shader uses 4,112 bytes of static threadgroup memory, and the separate
projection uses 8,208. Another prototype used 2,064 bytes and lost roughly
13% operator throughput at both primary shapes.

Three sixteen-row lanes contain 48 useful rows. A separate M48 tile reduces the
primary operator time from 0.8120 to 0.6025 ms (+34.78% operator throughput),
passing all registered split/spread and numerical checks. It keeps the 64-row
plan and allocation capacity. The guarded graph selects M48 for the measured
M5 Pro AF4G64 gate/up policy at B3. Nine complete lane arrays match, and the
profile confirms 69 gate and 69 up calls there. Static threadgroup storage is
8,208 bytes in both paths. Row extent falls 25%, and primary operator latency
falls 25.8%. Avoided padding, accumulator capacity and compiler scheduling are
plausible contributors. The registered whole-model comparison remained unrun
at the research cutoff.

## Work in the retained comparisons

| Control → candidate | Complete groups | GPU time per batch | Emitted tokens per lane per batch | GPU token rate |
| --- | ---: | ---: | ---: | ---: |
| Committed DSpine (8 rows) → final DSpine (16 rows), 31 historical cases | 3/3 | -1.30% to +0.28% | +24.17% | +23.83% to +25.80% |
| DFlash2 → DSpine (16 rows), 31 historical cases | 3/3 | +1.57% to +1.81% | +27.60% | +25.33% to +25.63% |
| DFlash2 → DSpine (16 rows), four cloned lanes | 3/3 | +85.70% to +86.84% | +16.00% | -37.91% to -37.53% |
| DFlash2 → DSpine (16 rows), 12 synthetic agent prompts | 0/3 | — | — | — |
| M64 → M48, three cloned sixteen-row lanes | 0/3 | — | — | — |

The GPU time per batch and emitted tokens per lane per batch divide the same
pooled totals. Their ratios give the candidate's GPU token rate relative to
control at fixed lane count. Pooled comparisons describe the actual output
sequences each arm generated. The registered wall-time comparisons use their
own repetition aggregation and require complete output equality.
[Saved work accounting](results/decode-work-v1.json)

In the four-lane fixture, the identical-output comparison qualifies in favor
of DFlash2. Sixteen-row DSpine emits 1.16 times as many tokens per batch at
1.86–1.87 times the GPU time. At that token yield it would need about 38% less
GPU time per batch to reach DFlash2's GPU token rate. The three cloned prompts
share warmed prefixes within each request batch.

## Memory and long-prompt cost have different endpoints

Bounding verify scratch to an explicit 32K served context saves 47.619 MiB at
eight rows and 95.238 MiB at sixteen rows before allocator rounding. It preserves
the actual per-lane dispatch split count, staging stride and shader. Automatic
context sizing retains the previous global bound.

When the entire served context is at most 2,048 tokens, none of the later
positions in an 8,192-token draft window can be consumed. Selecting the already
compiled 2K layout freed 564.375 MiB of actual pooled/cached state allocations
in the four-lane, sixteen-row fixture. The elastic memory budget stayed fixed.
Larger and automatic contexts retain the checkpoint's 8K window. The 2K state
layout has a separate persistent-cache namespace.

Each lane loses 6,144 slots from five BF16 key/value rings with eight KV heads
of dimension 128. That removes 120 MiB per lane. Each encoded context window
loses another 16.875 MiB. The four lanes and one cached window in the fixture
account for the observed 564.375 MiB.

The 8,193-token diagnostic prefill takes about 18 seconds, while B1 decode
cycles take about 82 ms. Long cold request wall time can therefore improve far
less than decode throughput. Prefix reuse should be evaluated on client wall
time including checkpoint materialization, replay, eviction and later reuse.

The same long prompt demonstrates a numerical qualification issue. A cached
prefix of 8,128 tokens followed by 65 replayed tokens changes the greedy output
at index 34 in both the committed eight-row baseline and the new paths. Seeding
the full prompt gives an actual 8,192-token hit plus one replay token. Cold and
warm outputs and all fourteen sampled full-vocabulary logit rows then match bitwise
within each tested width. Eight versus sixteen cold execution still diverges
at index 34. The first divergent baseline cold/warm choice has only about a
0.019-logit top-two margin. The changed replay geometry gives a concrete way
to localize the difference with matched-prefix logit and layer probes.

Static feature-capture review finds the same call-point convention in the
inspected native and pinned MLX paths. Configured layer indices select post-layer
outputs, and concatenation follows configuration order. Native verification
captures after the FFN and before the final head norm. Its prefill path captures
the corresponding layer output. Matched layer outputs, token and anchor
alignment and cached replay remain open numerical comparisons.

## Follow-up order

1. Investigate target-prefill chunk numerical drift and guarded prefix junctions
   using exact replay metadata and saved pre-policy logits. Measure cache
   creation, reuse and retained memory with DSpine's 8K restore plan. The current
   junction admission uses a 2K minimum saved-prefix gain. Boundary feasibility
   separately uses the selected draft window. Measure a junction's savings and
   retained-window cost before tuning either policy.
   The same-build eight-row functional pass reproduces both eight-row controls
   on all 31 historical cases. Compare final eight and sixteen rows at the
   first divergent token, using the common prefix through that position. Keep
   the committed eight-row build as a source control.
   A separate eight-row M16 policy control can test whether target tile choice
   explains part of the difference. Record pre-policy logits and margins before
   changing the drafter or interpreting the rate difference as fixed-work speed.
2. Evaluate M48 in heterogeneous batches once the fixed-lane evidence supports
   that step. Carry compute height explicitly alongside storage capacity.
   Add permanent selection and oracle coverage and retain M64 as the fallback.
   [M48 plan](M48_NEXT.md)
3. Extend the identical-input tail check to backbone and candidate
   generation and target-feature swaps. The 22 tested conditional rows agree on
   argmax and same-uniform choices, with maximum total variation 0.00036. The
   pinned MLX loader rejects the checkpoint's sliding-window configuration.
   Add explicit reference support for the trained window and validate its
   attention before running the full-model comparison.
4. Compare fixed eight-row and sixteen-row DSpine at B3 and B4 with exact-output
   prompts and heterogeneous arrivals after resolving the fixed-width output
   differences. For adaptive width, a sixteen-to-eight transition may inherit
   9–16 pending recurrent-state rows. Keep replay capacity independent of the
   next execution width or explicitly flush state and charge its cost. Test
   every pending count, lane permutation, suspension and disk restore. Freeze
   the policy before measuring a new mixed workload.
