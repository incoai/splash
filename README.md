# Splash

[![CI](https://github.com/incoai/splash/actions/workflows/ci.yml/badge.svg)](https://github.com/incoai/splash/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-black.svg)](#quick-start)

**A local inference engine for Apple silicon, built around the model.**

Splash serves a small set of models to coding agents and to any OpenAI or
Anthropic compatible client, on one Mac. On a 48 GB M5 Pro, Splash 1.0 decoded
Qwen3.8-27B at 2× the speed of the next-fastest engine we measured and, with a
32K context cached, returned the first token in 282 ms
([Performance](#performance)). Its kernels, draft model, and memory
plan are specialized for each model it serves. That is why it is fast, and why
there is nothing to configure.

## Quick start

Apple M3 or newer, macOS 26.4 or later, [Homebrew](https://brew.sh), 36 GB
of unified memory (48 GB or more recommended), and free disk for the model,
its draft and a prepared copy of their weights (up to about 40 GB in total for
Qwen3.8-27B and 48 GB for Qwen3.6-35B-A3B). Macs with 24 GB run the smaller
GGUF files: on a 24 GB M6 (12-core GPU), `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS`
with its DFlash2 draft advertises a 102,393-token context and decodes code at
43.5 tok/s, and `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL` advertises the full
256K context and decodes at about 100 tok/s. Where memory cannot hold a long
context, startup suggests `--max-cache-disk`.

```bash
brew install incoai/tap/splash
splash serve --model mlx-community/Qwen3.8-27B-4bit
```

The first run checks that the Mac's GPU and macOS are supported, downloads the
model and its matching DFlash2 draft, prepares weights for the Metal kernels,
checks available memory, and starts serving on `127.0.0.1:8000`. Later starts
reuse the prepared weights.

Once it prints `Ready`, leave this terminal open. Open <http://127.0.0.1:8000>
in your browser, or run an installed coding agent from another terminal:

```bash
splash opencode    # or: splash claude / splash codex / splash hermes / splash pi
```

`splash pi` adds a `splash` provider to Pi's `models.json` (`splash-<port>` for
a server on another port) and leaves Pi's other providers, settings and
sessions alone.

Press Ctrl+C in the server terminal to stop Splash.

## Use the API

Splash speaks OpenAI Chat Completions (`/v1/chat/completions`), OpenAI Responses
(`/v1/responses`), and Anthropic Messages (`/v1/messages`), all with streaming,
tool calls, JSON Schema output, images, and inline PDFs. `/tokenize` and
`/apply-template` return token IDs and the rendered prompt without running the
model.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3.8-27B-4bit",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}]
  }'
```

`model` is optional. If set, use the served model ID or a configured
[model alias](DEVELOPMENT.md#api-model-aliases).
Reasoning follows the model default unless a [server default](DEVELOPMENT.md#default-reasoning-effort)
is configured. `"reasoning_effort": "none"` turns it off, and
Qwen3.8-27B also takes `low`, `medium`, and `xhigh`.

`/v1/judgments` and `/v1/systemone` provide scoring without generation.
See [judgment contracts](DEVELOPMENT.md#judgment-contracts) for details.

## Models

| Base model | MLX affine example | GGUF example |
| --- | --- | --- |
| Qwen3.8-27B | `mlx-community/Qwen3.8-27B-4bit` | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` |
| Qwen3.6-35B-A3B | `mlx-community/Qwen3.6-35B-A3B-4bit` | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` |

`--model` accepts the upstream repository directly: an MLX affine 4-bit,
group-64 checkpoint (such as the mlx-community `-4bit` conversions) or a GGUF,
selected with `OWNER/REPO:VARIANT` (for example `:UD-Q4_K_M`). Splash
identifies the model from its own metadata, its architecture and dimensions,
before downloading any weights, and pairs the DFlash2 draft trained for it;
`--draft-model` replaces that draft with another DFlash2 checkpoint, a
repository or a local directory. The tokenizer, configuration and chat template come
from the target repository for MLX and from the selected GGUF file itself for
GGUF, never from another repository: unsupported or incomplete tokenizer
metadata is an error. GGUF variants whose tensor types Splash cannot load are
rejected before download. Of Unsloth's files, every one loads for both models,
from UD-IQ1_S up, except UD-Q8_K_XL and BF16
([GGUF targets](DEVELOPMENT.md#gguf-targets)). Prism ML's
`prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0`, a Qwen3.8-27B target, loads as well,
with its input rotation and vision projector. Legacy Splash packages such as
`incoai/Qwen3.8-27B-Splash` remain loadable.

Vision comes from the same source: embedded vision tensors for MLX, or the
repository's companion BF16 or F32 `mmproj` GGUF. Both are prepared as
BF16; an F32 or F16 tensor loads only when every value is exactly a BF16, as in
Unsloth's mmproj files.
Use `--language-only` to skip vision loading and preparation. It also skips the
GGUF mmproj download; MLX vision tensors share the language model's shards, so
those shards still download in full. The server then rejects image and PDF input
and reports `vision: false` in `/status` and `/v1/models`.

The prepared weights live in `~/Library/Caches/Splash/weights`
(`SPLASH_WEIGHT_CACHE` relocates them); preparation uses bounded temporary
memory, and later starts reuse the result. Each start checks the upstream
revisions of the model and of its draft, with one Hub request each of at most
5 seconds, and installs a new commit before serving it; without the Hub, or
when the new commit cannot be installed, the installed model starts.
`--revision` selects an upstream branch, tag or commit of the model (a commit
is never checked again, nor is its draft); otherwise the default branch is
followed. Private repositories need `HF_TOKEN`. Downloads use the Hugging Face
cache, and `brew upgrade splash` preserves models and agent sessions.

For LM Studio Bionic, follow its [Splash setup guide](https://lmstudio.ai/blog/splash-engine):
install the Splash runtime, then paste the full Hugging Face model link into its
model search. These integrations manage their own runtime and settings.

If a client’s model catalog does not list a model, the full model ID in
this table still works with `splash serve --model OWNER/REPO[:VARIANT]`. The browser chat
and the agent launchers connect to that server without a catalog search.

For a custom model download location, see [model cache](DEVELOPMENT.md#model-cache).

## Settings

There is no config file. The server binds `127.0.0.1:8000` by default.
Context supports up to the model’s native 256K window; usable capacity
depends on available memory. `splash serve --help` lists server options and examples.
The startup summary and `maximum_context_tokens` in `/status` show the effective
server limit. `/v1/models` and `/v1/models/{id}` report the same limit as
`max_model_len` and its compatibility alias `context_length`, including model aliases.
Clients can impose a smaller limit. With enough memory,
request the full window using `--max-context 256K`. This is a capacity limit, not a guarantee
that a long uncached prompt will reach its first token quickly.

`splash serve` accepts these optional flags:

- `--revision`: upstream branch, tag or commit. Default: the default branch.
- `--draft-model`: another DFlash2 draft, a repository or local directory.
  Default: the draft trained for the model.
- `--language-only`: skip vision; image and PDF input is then rejected.
- `--host`: HTTP bind address. Default: `127.0.0.1`. Clients connect by IP
  address or `localhost`; other names need `--allowed-host`.
- `--port`: HTTP port. Defaults to `SPLASH_PORT` or `8000`.
- `--max-memory`: ceiling on Metal allocations, e.g. `28G`. Default: auto.
- `--max-context`: context limit, up to `256K`, e.g. `100K`. Default: auto.
- `--max-cache-disk`: SSD tier for the cache, e.g. `5G`. Default: 0 (off).
  Startup suggests it when memory cannot hold the context; with it, a long
  request that runs out of memory keeps its progress on SSD and replays far
  less of its prompt.
- `--kv-format`: target KV cache storage, `int8` (default) or `bf16`.
- `--max-image-pixels`: maximum resized pixels per image. Default: 4,194,304.
- `--allowed-host`: extra HTTP `Host` name to accept, such as `mymac.local`;
  not a bind address. Repeatable.
- `--api-key`: require this key on API requests, as a bearer token or
  `x-api-key`. Defaults to `SPLASH_API_KEY`.
- `--no-webui`: turn off the chat page.

To use BF16 target KV, select it when starting the server:

```bash
splash serve --model mlx-community/Qwen3.8-27B-4bit --kv-format bf16
```

BF16 avoids target KV quantization, uses approximately twice the target KV
memory, and can be slower at long contexts. Model weights are unchanged.
Restart the server to switch formats. Omit `--kv-format` or use
`--kv-format int8` for the default INT8 cache.

If the model does not fit in the memory available, startup prints a memory
budget breakdown and stops.

`--max-cache-disk` works with either KV format and preserves its stored bytes
without further quantization. Disk cache is temporary and does not survive a
server restart. For disk cache behavior and memory overhead, see
[disk cache](DEVELOPMENT.md#disk-cache).

Authentication is off by default. Set `SPLASH_API_KEY` in the shell that runs
`splash serve` and in the shell that runs an agent, and both sides use it.
Health and readiness probes stay public.

For LAN access and multiple servers, see
[server configuration](DEVELOPMENT.md#server-configuration).

## Performance

Measured for the Splash 1.0 release (September 2026) on an M5 Pro (16-core
GPU, 48 GB), serving the Qwen3.8-27B and Qwen3.6-35B-A3B Splash packages:
selected SPEED-Bench coding prompts over HTTP, a 1,024-token output limit,
reasoning on (medium for the 27B). The ratio in each cell is against the
next-fastest engine we measured. The MLX 4-bit models prepare to the packages'
target weights, byte for byte but for the 27B's 48 per-layer GDN decay vectors,
each within a float ULP, and decode within 0.5% of them on this M5 Pro
([upstream loading](dev/benchmarks/upstream-loading.md)). GGUF targets run
other kernels; [GGUF against llama.cpp](#gguf-against-llamacpp) compares them.

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

For repeatable measurements on your Mac, see [local benchmarks](DEVELOPMENT.md#local-benchmarks).

### GGUF against llama.cpp

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
Splash's figures cover an M5 Pro and an M3 Max with `--kv-format bf16`; the
default INT8 cache gives 99.23–99.25% and 97.92–97.94% on the M5 Pro.

Speed uses the prompts and greedy settings of the table above. llama-server
runs with its default settings, which do not speculate, and for the 27B also
with the MTP draft Unsloth ships:

| Decode tok/s | M5 Pro, 20-core GPU | M3 Max, 40-core GPU |
| --- | ---: | ---: |
| Qwen3.6-35B-A3B · Splash | 175 | 209 |
| Qwen3.6-35B-A3B · llama.cpp | 69 | 66 |
| Qwen3.8-27B · Splash | 74 | 92 |
| Qwen3.8-27B · llama.cpp | 16 | 17 |
| Qwen3.8-27B · llama.cpp with MTP | 27 | 20 |

Splash decodes 2.5–3.2× as fast as llama.cpp on the 35B and 4.5–5.3× on
the 27B (2.7–4.6× against its MTP). Prefilling a 2,048-token chunk, it runs
at 559 tok/s against 374 on the 27B and 3,662 against 1,968 on the 35B on
the M5 Pro, and at 245 against 193 and 1,814 against 1,575 on the M3 Max.

## Design

The runtime, scheduler, cache, and API are shared. Everything else is rebuilt
per model:

- **A draft trained for the model.** Speculative decoding is the decode path in
  Splash, not an option. Each supported base model has a matching [DFlash
  2](https://inco.ai/blog/dflash2/) draft, and one pass of the target verifies a
  block of tokens in parallel.
- **Kernels for the model's shapes.** Fused Metal kernels for the models'
  attention, GDN and MoE dimensions, with dispatch policies measured offline
  per GPU family and core count. MLX weights are prepared once into layouts
  packed for these kernels. GGUF weights keep their llama.cpp quantization,
  repacked once into planes that kernels chosen by GPU family, core count and
  format decode directly, without per-shape tuning. Both are mapped zero-copy
  from disk. Everything ships precompiled: no Xcode, no compiler toolchain,
  nothing tuned on your machine.
- **A memory plan computed for this machine.** Context, KV capacity, and batch
  limits are worked out at startup from the memory Metal recommends, less the
  weights, the draft, and each request's state.

The [launch post](https://inco.ai/blog/splash/) covers the design in depth.

## More

- [DEVELOPMENT.md](DEVELOPMENT.md): building from source, model loading, tests
  and release packaging.
- Apache-2.0, see [LICENSE](LICENSE); the GGUF kernels include MIT-licensed
  material from llama.cpp, see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
  Model weights keep their own licenses.
