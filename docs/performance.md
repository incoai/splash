# Performance

[Back to Splash](../README.md#performance)

## Splash 1.0 launch benchmarks

Measured for the Splash 1.0 release (September 2026) on an M5 Pro (16-core
GPU, 48 GB), serving the Qwen3.8-27B and Qwen3.6-35B-A3B Splash packages:
selected SPEED-Bench coding prompts over HTTP, a 1,024-token output limit,
reasoning on (medium for the 27B). The ratio in each cell is against the
next-fastest engine we measured. At the time, the MLX 4-bit models prepared to
the packages' target weights, byte for byte but for the 27B's 48 per-layer GDN
decay vectors, each within a float ULP, and decoded within 0.5% of them on this
M5 Pro ([upstream loading](../dev/benchmarks/upstream-loading.md)). MLX targets
now load into the weight layout of GGUF targets and run on the same kernels,
which [GGUF against llama.cpp](#gguf-against-llamacpp) measures.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt | 210 tok/s (1.7×) | 74 tok/s (2.0×) |
| Prefill · 32K prompt | 2,011 tok/s (1.3×) | 363 tok/s (1.2×) |
| Cached time to first token · 32K replay | 123 ms (6.6×) | 282 ms (7.3×) |
| Aggregate decode · 4 concurrent short prompts | 357 tok/s (2.0×) | 170 tok/s (3.9×) |

Splash 1.0 led on every measure at every prompt length we tested, and the lead
grew with load: 3.8× at four concurrent 32K requests on the 35B. The
[launch post](https://inco.ai/blog/splash/) has the method and the full
comparison against oMLX, Lily, uzu, and Ollama.

For repeatable measurements on your Mac, see [local benchmarks](validate.md#local-benchmarks).

## GGUF against llama.cpp

We compared Splash with llama.cpp (e6ab7c1, Metal) on the same Unsloth
UD-Q4_K_M files. For accuracy, both read the same text, 16,384 positions of
prose, code and chat, and at each position we compared the tokens they rank
first:

| Same token ranked first | Qwen3.8-27B | Qwen3.6-35B-A3B |
| --- | ---: | ---: |
| Splash and llama.cpp | 99.30–99.45% | 97.83–98.14% |
| llama.cpp on the CPU and on Metal | 97.8% | 96.5–96.9% |
| llama.cpp one token at a time and batched | 99.65–99.75% | 97.95% |

The positions where they differ are near-ties: there, llama.cpp's two best
tokens are a median 0.03–0.10 nats apart, against 2.6–2.7 nats over all
positions. Splash's perplexity is 0.1–0.4% (27B) and 0.1–0.9% (35B) above
llama.cpp's; llama.cpp's CPU backend is 1.7–1.8% above its Metal on the 27B.
Splash's figures cover an M5 Pro and an M3 Max with `--kv-format bf16` and
`--disable-ane`; the default INT8 cache gives 99.23–99.25% and 97.92–97.94% on
the M5 Pro.

Speed uses the selected SPEED-Bench coding prompts described above, with
greedy sampling. llama-server runs with its default settings, which do not
speculate, and for the 27B also
with the MTP draft Unsloth ships:

| Decode tok/s | M5 Pro, 20-core GPU | M3 Max, 40-core GPU |
| --- | ---: | ---: |
| Qwen3.6-35B-A3B · Splash | 175 | 209 |
| Qwen3.6-35B-A3B · llama.cpp | 69 | 66 |
| Qwen3.8-27B · Splash | 74 | 92 |
| Qwen3.8-27B · llama.cpp | 16 | 17 |
| Qwen3.8-27B · llama.cpp with MTP | 27 | 20 |

Splash decodes 2.5–3.2× as fast as llama.cpp on the 35B and 4.5–5.3× on
the 27B (2.7–4.6× against its MTP); the ratios are of the unrounded rates.
Prefilling a 2,048-token chunk, it runs at 559 tok/s against 374 on the 27B
and 3,662 against 1,968 on the 35B on the M5 Pro, and at 245 against 193 and
1,814 against 1,575 on the M3 Max.

## Experimental DSpine against DFlash2

An exploratory comparison of the initial serial refinement implementation
on November 9, 2026 used an Apple M5 Pro
(18-core CPU, 20-core GPU, 48 GB), macOS 26.6, and the same
`mlx-community/Qwen3.8-27B-4bit` target for both drafters. The native DSpine
backend currently serves **eight rows of its trained sixteen-row block**;
DFlash2 also serves eight rows. Each retains its own draft context window:
8,192 tokens for DSpine and 2,048 for DFlash2. This measures the current
implementation, with both backbones prepared as affine 4-bit groups of 64
and DSpine's bridge kept in BF16.

Four rounds ran DFlash2, DSpine, DSpine, then DFlash2, restarting the server
between rounds. The server used a 32K target context, INT8 KV, FP32 recurrent
state, `--disable-ane`, `--verify-rows all`, and a persistent 4 GiB SSD cache.
The short requests used distinct cache salts, temperature 0, seed 42,
`reasoning_effort: "none"`, and a 256-token output limit. Fibonacci stopped
after 95 tokens; each other short prompt reached the output limit. Each
prompt ran once per round. Startup and warmup are excluded.

| Workload | DFlash2 tok/s | DSpine tok/s | DSpine throughput change |
| --- | ---: | ---: | ---: |
| Python iterative Fibonacci | 106.1 | 97.4 | −8.2% |
| Python stable merge sort | 91.3 | 78.0 | −14.6% |
| Python LRU cache | 81.0 | 64.8 | −20.1% |
| TypeScript retry with backoff | 78.3 | 85.6 | +9.2% † |
| All four solo requests, combined decode rate | 85.2 | 77.0 | −9.6% |
| Four simultaneous requests, end-to-end aggregate | 192.7 | 172.6 | −10.4% † |

The per-prompt rates are means of each backend's two runs. The combined solo
rate is total decode tokens divided by total decode time, excluding the
first token selected during prefill. The four-request aggregate divides all
863 completion tokens by the client wall time from submitting the four
requests until all finish, so it includes prefill, queueing and decode; the
table averages the two round rates.

† These observations are **inconclusive** under the repository's
[ABBA rule](../dev/benchmarks/abba.py): the TypeScript baseline spread is
7.7%, and the four-request baseline spread is 13.4%, above the 5% noise
limit. The combined solo spread is 2.9%; its slowdown exceeds the rule's
noise allowance. No continuous machine-quietness or thermal trace was
captured. Four short prompts with capped outputs are a small sample, so the
figures establish neither a general speedup nor a general ranking of the
drafter architectures. Both drafters ran on the same candidate build; this
comparison does not measure default DFlash2 against an unmodified main
build. DFlash2 remains the default.

All four solo greedy outputs matched byte for byte across all four rounds.
Additional serving checks passed for temperature sampling, four concurrent
requests, a 14,316-token prompt with prefix reuse, disk restore of 14,304
cached tokens, greedy and sampled JSON-schema output, and stream cancellation
followed by another request. The disk, schema and cancellation checks also
ran with Metal GPU validation enabled.

The [measurement record](../dev/benchmarks/results/dspine-qwen38-27b-2026-11-09.json)
contains the exact prompts, model revisions, measured build identity,
per-request timings and speculative counts, output hashes, round totals and
ABBA calculations. Its local paths and request identifiers are omitted.
For use and current compatibility limits, see
[experimental DSpine](models.md#experimental-dspine).

### Parallel DSpine refinement

On November 10, profiling identified final-write refinement as the largest
DSpine-specific cost: about 5.8 ms per decode cycle. Its single threadgroup
per request performed the message and hidden-state matrix-vector products
with long scalar loops. The parallel implementation distributes their output
coordinates across threadgroups with SIMD reductions. Each of the seven
positions runs message, state and scoring dispatches in order, preserving
the selected-predecessor dependency and the conditional sampling
probabilities within the existing GPU command. FP32 scratch is planned in
the decode arena, 22 KiB per lane, and reused between positions.

A fresh HTTP ABBA comparison used the same hardware, models, quantization,
prompts and serving options above. The baseline was `8de9b727`; each server
restarted, waited for nominal thermal state, and completed an untimed
128-token Fibonacci warmup before measurement. Thermal state was sampled at
round boundaries, with some rounds ending fair; there was no continuous
machine-quietness trace.

| Workload | Serial DSpine tok/s | Parallel DSpine tok/s | Throughput change |
| --- | ---: | ---: | ---: |
| Python iterative Fibonacci | 97.3 | 104.6 | +7.5% |
| Python stable merge sort | 77.9 | 83.8 | +7.5% |
| Python LRU cache | 64.7 | 69.3 | +7.0% |
| TypeScript retry with backoff | 85.4 | 91.2 | +6.7% |
| Combined solo decode rate | 77.0 | 82.4 | +7.1% |

These rates are `1000 / mean milliseconds per token` over each backend's
two rounds. The combined metric sums decode tokens and decode times within
each round before applying ABBA. Its spread was 1.1%; all solo output hashes
and speculative cycle counts matched across the four rounds. The
four-concurrent result was inconclusive at 9.9% spread.

A separate ABBA comparison on the parallel candidate build measured the two
drafters again. All 66 loaded target weight images were byte-identical
between their installations, checked with `weight-digests`; the draft
images differ. All four solo output hashes also matched.

| Workload | DFlash2 tok/s | Parallel DSpine tok/s | DSpine throughput change |
| --- | ---: | ---: | ---: |
| Python iterative Fibonacci | 105.4 | 103.9 | −1.5% |
| Python stable merge sort | 91.4 | 82.7 | −9.4% |
| Python LRU cache | 81.1 | 68.1 | −16.0% |
| TypeScript retry with backoff | 81.2 | 89.4 | +10.1% |
| Combined solo decode rate | 86.2 | 81.2 | −5.8% |
| Four simultaneous requests, end-to-end aggregate | 201.6 | 176.6 | −12.4% |

Combined solo spread was 0.2%. Fibonacci's small difference is within the
ABBA noise allowance. The four-request spread was 4.8%, close to the 5%
inconclusive threshold, though its slowdown exceeded the rule's allowance.
This remains an exploratory sample, not a general ranking of the drafters.
The kernel improvement leaves acceptance unchanged on these prompts:
DSpine still uses 44 cycles for merge sort against DFlash2's 41, and 53 for
LRU against 46; it uses 40 for retry against 46. Improving refinement alone
does not remove that difference in cycles.

Kernel checks against independent CPU equations passed under Metal shader
validation for batch widths one through four, mixed sampling, SIMD boundary
cases and production hidden/code/rank dimensions. Live shader-validated
serving also passed temperature sampling, four concurrent requests, a
14,316-token cold prompt, prefix and suffix reuse, a 14,304-token disk
restore, greedy and sampled JSON-schema output, and stream cancellation
followed by another request.

The [measurement record](../dev/benchmarks/results/dspine-refinement-qwen38-27b-2026-11-10.json)
contains both comparisons, exact prompts, revisions, binary/build identities,
request metrics, output hashes and ABBA calculations. Repeat the before/after
comparison with saved builds (each binary needs its matching Metal libraries
beside it) and the same installed DSpine root:

```sh
python -m dev.benchmarks.draft_comparison \
  --baseline-binary BASELINE/splash --candidate-binary CANDIDATE/splash \
  --baseline-model-root DSPINE_ROOT --candidate-model-root DSPINE_ROOT \
  --output build/dspine-refinement-abba
```

### Acceptance by proposal position

A follow-up native trace on November 10 used the same four prompts, target,
draft precision, eight-row blocks and 256-token output caps, with one greedy
continuation per prompt. The native outputs matched each other and the HTTP
output hashes; their speculative cycle counts also reproduced the HTTP
comparison. Each cycle records its verified draft count and accepted prefix.
The analysis excludes the final cycle conservatively for EOS/output-cap
censoring, then excludes any other partial-width cycle.

| Workload | DFlash2 cycles accepting all seven proposals | DSpine cycles accepting all seven proposals |
| --- | ---: | ---: |
| Fibonacci | 10/12 (83.3%) | 10/12 (83.3%) |
| Merge sort | 25/40 (62.5%) | 20/43 (46.5%) |
| LRU cache | 19/45 (42.2%) | 15/51 (29.4%) |
| TypeScript retry | 21/45 (46.7%) | 20/39 (51.3%) |

The denominators are eligible full-width, non-final cycles. In LRU, DSpine
rejects fewer first proposals (3/51) than DFlash2 (4/45); the larger losses
in this trace occur at positions two and six. Merge sort loses more at
positions five and seven. Those small conditional samples identify places
to inspect, without establishing a model defect or a quantization cause.

Fibonacci frequently reaches the block cap, making it a useful case for
measuring native sixteen-row execution. A longer block can extend a full
accepted prefix, but these fractions do not predict the acceptance of later
positions or their verification cost. These questions motivated the
precision and sixteen-row investigation below.

The [trace record](../dev/benchmarks/results/dspine-acceptance-qwen38-27b-2026-11-10.json)
contains every cycle, output tokens, conditional acceptance and survival by
position. These are four short deterministic continuations, not a general
quality evaluation. To collect and summarize traces:

```sh
build/engine-tests/backend-benchmark METALLIB MODEL_ROOT \
  --scenario acceptance --samples 1 --ane-ffn-share 0 --max-context 32768 \
  > TRACE.json
python -m dev.benchmarks.draft_acceptance \
  --baseline DFLASH_TRACE.json --candidate DSPINE_TRACE.json \
  --output SUMMARY.json
```

### Eight-hour DSpine investigation

A November 10 follow-up on the same M5 Pro tested precision, matched prefixes,
sixteen-row execution and several kernel alternatives. The native benchmark
now also honors the selected draft's context window; serving already did.
The **sixteen-row research prototype improves serial native decode throughput
by 15.3–16.2%** against eight-row DSpine with the M16 policy, across three
fixed ABBA groups. All complete output arrays, operating checks and per-case
noise gates passed. The corpus has 30 primary tasks plus one secondary
control, short prompts of 33–78 tokens, and output caps of 384–1,024.
It reuses historical tasks and does not establish long-context speed or a
general drafter ranking. Power receipts show AC/battery transitions during
the first group; groups two and three stayed on AC at recorded boundaries.
The registered native gates did not require AC. Results retain that limitation,
and the report includes the per-round power audit.

| Experiment | Result | Qualification |
| --- | --- | --- |
| M16 policy alone, eight rows | +1.61%, +1.74%, +2.07% serial decode | First two groups below the 2% effect floor |
| Sixteen rows against eight-row M16 | +16.19%, +15.32%, +16.04% serial decode | All three registered groups pass; see power caveat |
| Wide gate/up two-pass against one-pass | +56.8–64.4% concurrent HTTP, descriptive | Two groups fail output equality |

The primary cycle count falls from 1,303 to 1,062 with sixteen rows. Increasing
draft precision changes it much less: 1,291 with eight-bit weights and 1,289
with native BF16 planes, with all 31 greedy outputs matching. These precision
timings are unpaired screens; native BF16 storage is not a full upstream BF16
arithmetic-equivalence test. Several prose cases have lower mean throughput
with sixteen rows. At 67 matched short primary prefixes, DF8 and DS8 have
very similar first-cycle output means (5.955 versus 5.940); DS16 reaches
9.507. These overlapping historical prefixes do not predict production TPS
or establish a general drafter acceptance ranking. Even perfect whole-task
selection in hindsight adds only
1.51% over fixed sixteen rows on these short-case means, before switching or
prediction costs; this does not bound per-cycle or concurrent policies.

At batches three and four, sixteen-row verification uses 48 and 64 actual
rows, covered by the M64 tile. Its original fused gate/up dominates the cycle; the existing two-pass path avoids
most of that cost. Smaller-accumulator and shared-storage prototypes passed
independent numerical checks but established no further whole-model gain.
Actual selected K partitions matter: an operator win at two partitions did
not beat the deployed one-partition plan. The broader short-corpus native
DFlash2 comparison shows +18.3–18.9% aggregate TPS descriptively, but fails
qualification because two groups exceed the 5% per-case noise limit. The
fresh DFlash2-versus-sixteen-row HTTP comparison remains inconclusive at
11–46% endpoint spread. A fixed-width
native probe reproduced the batch-dependent merge-sort output variation under
both gate/up policies; this does not make the failed HTTP groups pass.

A separate two-case native long comparison qualifies at +6.0–7.0% decode
TPS against DFlash2+M16 across all three groups: the 8K prose case slows
about 3.5%, while 16K code improves about 17–19%. All outputs and registered
gates pass. The two cases were selected from the original four before any
of those prompts ran, to fit the eight-hour window; both protocols remain.
Summed cold request-wall means improve only 0.91% descriptively because
prefill dominates. These measurements do not establish a general ranking.

Functional coverage includes 9,000 cold sampling requests, long prefix/suffix
reuse, a shader-validated restart restoring 15,904 cached tokens, and actual
priority suspension/replay/resumption with matching complete clone outputs.
None of 78 early-token sampling comparisons rejected after Holm correction;
this finite screen does not prove identical distributions or cover long-range
changes and EOS termination. The separate synthetic memory-pressure probe
failed to trigger suspension and does not validate that mechanism.

The serving backend on this PR remains **eight rows**. Production widening
needs per-model geometry, preserving DFlash2's eight-row capacity, plus explicit
support or rejection of Table16/register layouts. The full sixteen-row Table16
test finds an out-of-bounds store; the prototype disables register paths and
passes a separately scoped plain16 oracle. A late packing-only follow-up
passes the complete sixteen-row GDN suite under Metal validation, while
register consumers and per-model dispatch remain future work. The measured
prototype's additional decode scratch
reduces the available dynamic budget by about 260.7 MiB on this setup.

The [investigation report and implementation plan](../dev/benchmarks/experiments/dspine_20261110/README.md)
describe every result's scope, including the separate broader native DFlash2
comparisons and target-feature diagnostics. The
[result index](../dev/benchmarks/results/dspine-investigation-qwen38-27b-2026-11-10.json)
links the compressed original receipts, failures and hardware trace.
[Source patches, frozen cohorts and the rebuild harness](../dev/benchmarks/experiments/dspine_20261110/reproduce.py)
reproduce the experiments without changing serving defaults.

## Smaller GGUFs on 24 GB Macs

Measured on a 24 GB M6 (12-core GPU), with each model's DFlash2 draft:

| Unsloth GGUF | Code decode | Advertised context limit |
| --- | ---: | ---: |
| `Qwen3.8-27B-GGUF:UD-IQ3_XXS` | 43.5 tok/s | 53,241 tokens (81,913 with `--language-only`) |
| `Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL` | 145 tok/s | 256K tokens |

The context column reports capacity, not the prompt length of the decode
measurement. The SSD cache, on by default, sets its state staging aside within
this memory: on the M6 it costs the 27B 4,096 tokens of advertised context,
which `--max-cache-disk 0` gives back (57,337 tokens, 86,009 with
`--language-only`). Coding agents need about 100K tokens (Claude Code's own
prompt is about 33K, and at 64K it compacted repeatedly and stopped), so serve
the 27B with `--language-only` for them; with `--disable-ane` and
`--max-cache-disk 0` as well it advertises 102,393 tokens.

Splash grows into free memory only while macOS reports no memory pressure and
its spare memory, the free, file-backed and purgeable pages and half of what
compressing other applications would free, keeps a reserve of 7% of the Mac's
memory (2 to 7 GiB) and a 1 GiB margin, 3 GiB on these Macs. It gives cached
memory back when that spare memory runs short of the margin, not for macOS's
warning alone, but a request in service still takes the memory it needs within
`--max-memory`, so these models leave other applications little room. A request that cannot get memory waits for it,
and so do the requests of its priority or below that the scheduler would start
after it. While an earlier request is at work, as one prefilling a prompt near
the context limit can be for many minutes, they wait with no time limit; without
one, the request refused memory fails after 30 s with `resource_timeout`: close
memory-heavy applications, or serve with `--language-only`. To bound the wait,
give requests a `timeout`, or serve with `--request-timeout`, which bounds the
long request too. A streaming request receives a keepalive every 2 s while it
waits; a non-streaming one receives nothing until its answer, so a client's
read timeout may end it first.
Critical memory pressure can suspend a long request; the
[SSD cache](server.md#ssd-cache), on by default, keeps its progress and
cached prefixes, and writes whenever memory runs short: serving
Ternary-Bonsai-2-27B PQ2_0 to six clients' mixed traffic for 30 minutes, a
16 GiB SSD cache on the M6 wrote 26 GB, about 50 GB an hour, and read 36 GB.
`--max-cache-disk-writes` sets a write budget to spare the drive, such as
`25G` a day on a 256 GB Mac: once it is spent, states other than the ones
conversations resume from are dropped instead of written.

See [the low-bit GGUF measurements](https://github.com/incoai/splash/pull/160)
for the workloads, memory pressure, SSD settings and limitations, and
[the Bonsai measurements](https://github.com/incoai/splash/pull/166) for PQ2_0.
