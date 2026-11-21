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
`--disable-ane`; the INT8 cache with bf16 queries (`--activations bf16`)
gives 99.23–99.25% and 97.92–97.94% on the M5 Pro.

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

## Apple8 (M2)

Apple8 takes the Apple9 kernel plans, whose matrix operations run on the FP32
pipe it shares, with one attention row of its own: it scores INT8 queries from
eight pages a split, where Apple9 never does (below).

Measured on an M2 Max (38-core GPU, 64 GB, macOS 27.0.1) serving Qwen3.8-27B
Heretic IQ4_XS with its DFlash2 draft at the model's 256K window, greedy, one
warm session climbing 65K, 131K and 250K tokens so each rung reuses the
previous prefix:

| Context | Decode | Prefill |
| --- | ---: | ---: |
| Empty | 53.1 tok/s | — |
| 65,571 | 39.1 tok/s | 112 tok/s |
| 131,047 | 29.0 tok/s | 61 tok/s |
| 249,935 | 20.0 tok/s | 38 tok/s |

The empty-context figure is the median of three programming questions (53.1,
40.8 and 67.7 tok/s; the spread is draft acceptance). Prefill is the rate of
the tokens each rung computes: a first cold send at 65K, the newly sent tail
above it.

Attention scores INT8 queries where a split reads enough pages for the
smaller operand traffic to repay the query codes: from eight pages on Apple8,
Apple10 and Apple11, and from 32 on larger Apple9 GPUs, whose 10-core M4
measurement at short histories kept BF16. For Apple8 that is 1.41x at 131,072
tokens and 1.54x at 249,935 in `attention-sweep`'s one-layer verify case (9.30
-> 6.58 and 18.41 -> 11.98 ms), and 1.57x on a 2,048-row prefill chunk at
16,384 tokens (310.7 -> 198.2 ms). The gain rises with a split's pages: on the
same Mac a verify reads 16 pages a split at 16,384 tokens for 1.19x and 62 at
249,935 for 1.49x. Shorter histories keep BF16 queries.

Qwen3.8-27B UD-Q4_K_M on the 1.6.1 tree, `--max-memory 46G --language-only`,
greedy, 400 output tokens, medians of three runs:

| Metric | M2 Max |
| --- | ---: |
| Decode · code prompt | 30.6 tok/s |
| Decode · prose prompt | 26.0 tok/s |
| Decode · 4 concurrent short prompts | 46.9 tok/s |
| Prefill · 6,430-token prompt | 146 tok/s |
| Prefill · 25,540-token prompt | 123 tok/s |
| Cached time to first token · 25.5K replay | 1.0 s |
| Context at `--max-memory 46G` | 256K tokens |

The prompts are the author's; each prefill is a first, cold send of its
prompt, and the decode figures are the engine's own
`predicted_per_second`. A 2,048-row prefill command holds the GPU for about 16
seconds at the 25.5K rate above, past the few seconds macOS gives a command
while the display needs the GPU, so a long cold prompt can be aborted with
`Impacting Interactivity` under desktop load, which fails that request and
restarts the engine. A lone lane's prefill commands now keep to eight seconds
(`EngineConfig::isolatedPrefillMilliseconds`, set by the native loop): that
holds a 512-row chunk, the Neural Engine FFN split's least, in one command,
where a three-second bound cuts to about 200 rows and never runs the split. A
cold 131K prefill runs 45 -> 61 tok/s and no 250K run aborted.

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
