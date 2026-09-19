# Splash

[![CI](https://github.com/incoai/splash/actions/workflows/ci.yml/badge.svg)](https://github.com/incoai/splash/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-black.svg)](#quick-start)

**A local inference engine for Apple silicon, built around the model.**

Splash serves a small set of models to coding agents and to any OpenAI or
Anthropic compatible client, on one Mac. On a 48 GB M5 Pro it decodes
Qwen3.8-27B at 2× the speed of the next-fastest engine and, with a 32K context
cached, returns the first token in 282 ms. Its kernels, draft model, and memory
plan are specialized for each model it serves. That is why it is fast, and why
there is nothing to configure.

## Quick start

Apple M3 or newer, macOS 26.4 or later, [Homebrew](https://brew.sh), and 36 GB
of unified memory (48 GB or more recommended).

```bash
brew install incoai/tap/splash
splash serve --model incoai/Qwen3.8-27B-Splash
```

The first run downloads and verifies the model package, checks available
memory, and starts serving on `127.0.0.1:8000`.

Once it prints `Ready`, leave this terminal open. Open <http://127.0.0.1:8000>
in your browser, or run an installed coding agent from another terminal:

```bash
splash opencode    # or: splash claude / splash codex / splash hermes
```

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
    "model": "incoai/Qwen3.8-27B-Splash",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}]
  }'
```

`model` is optional. If you set it, it must match the package you served.
Reasoning is on by default. `"reasoning_effort": "none"` turns it off, and
Qwen3.8-27B also takes `low`, `medium`, and `xhigh`.

### Typed judgments without generation

`POST /v1/systemone` accepts the [TypeSafe System One](https://docs.typesafe.ai/)
request and response shapes: `noul`, `choice`, and `score` questions over a
shared state. It works with the official `typesafe-sdk` (verified with 0.7.0):

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(
    base_url="http://127.0.0.1:8000",
    api_key="local",  # Use SPLASH_API_KEY's value if server authentication is on.
    model="incoai/Qwen3.8-27B-Splash",
) as client:
    result = client.system_one(
        state={"message": "I was charged twice. Please fix this today."},
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "department": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": None, "technical": None, "sales": None},
            ),
            "urgency": Score(
                instructions="How urgent is the request?",
                criteria=["No urgency", "This week", "Today"],
            ),
        },
    )
    print(result.choices["department"].choice)
```

Use the actual served model ID, not a hosted Jev model name. `/v1/models`
supports both OpenAI model discovery and the TypeSafe SDK's `models.list()`.

For [SemIf](https://github.com/TheoLeeCJ/SemIf)'s `direct-options-v1` prompt and
raw option logits, use `POST /v1/judgments`:

```bash
curl http://127.0.0.1:8000/v1/judgments \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "approval",
    "state": "The proposal is awaiting approval.",
    "question": "What is the current approval status?",
    "options": [
      {"id": "approved", "description": "Approval was explicitly given."},
      {"id": "pending", "description": "Approval has not been given."}
    ]
  }'
```

SemIf accepts 2–16 options; System One accepts up to 255 choice labels or score
levels, subject to single-token slot availability. Both endpoints are text-only,
disable thinking, and read final-position logits without sampling or decoding.
Output-token usage is zero. Prompts that exceed the context limit are rejected,
not truncated.

**These are local model scores, not Jev predictions or calibrated confidence.**
Probabilities are a softmax over the declared answer slots. Choice/score
`confidence` is normalized entropy concentration, `1 - H(p) / log(K)`, not an
estimate of correctness. Score answers are probability-weighted level indices.
Measure accuracy and calibrate on representative held-out data before using
thresholds to make consequential decisions.

## Models

| Package (`--model`) | Contents | Download |
| --- | --- | ---: |
| [`incoai/Qwen3.8-27B-Splash`](https://huggingface.co/incoai/Qwen3.8-27B-Splash) | Qwen3.8-27B, 4-bit, with its DFlash 2 draft | 17.4 GB |
| [`incoai/Qwen3.6-35B-A3B-Splash`](https://huggingface.co/incoai/Qwen3.6-35B-A3B-Splash) | Qwen3.6-35B-A3B, 4-bit, with its DFlash 2 draft | 20.9 GB |

`--model` takes any `owner/repo` that holds a Splash package, a format
[DEVELOPMENT.md](DEVELOPMENT.md#model-packages) describes. Plain MLX or
Transformers checkpoints do not work. Private repositories need `HF_TOKEN`.
Packages download into the Hugging Face cache, and `brew upgrade splash` keeps
them, along with model links and agent sessions.

## Settings

There is no config file. The server binds `127.0.0.1:8000`, one server at a
time. Context supports up to the model’s native 256K window; usable capacity
depends on available memory.

`splash serve` accepts these optional flags:

- `--max-memory`: ceiling on Metal allocations, e.g. `28G`. Default: auto.
- `--max-context`: context limit, up to `256K`, e.g. `100K`. Default: auto.
- `--max-image-pixels`: maximum resized pixels per image. Default: 4,194,304.
- `--allowed-host`: extra HTTP `Host` name to accept, for a proxy. Repeatable.
- `--api-key`: require this key on API requests, as a bearer token or
  `x-api-key`. Defaults to `SPLASH_API_KEY`.
- `--no-webui`: turn off the chat page.

If the model does not fit in the memory available, startup prints a memory
budget breakdown and stops.

Authentication is off by default. Set `SPLASH_API_KEY` in the shell that runs
`splash serve` and in the shell that runs an agent, and both sides use it.
Health and readiness probes stay public.

- Experimental cache offloading: [PR #3](https://github.com/incoai/splash/pull/3)
  adds SSD offloading for KV cache and GDN states. Build from that branch and
  set `--max-cache-disk 8G` to enable it. This helps preserve reusable prefixes
  when RAM is limited, reducing repeated prefill.

## Performance

Measured on an M5 Pro (16-core GPU, 48 GB): selected SPEED-Bench coding prompts
over HTTP, a 1,024-token output limit, reasoning on (medium for the 27B). The
ratio in each cell is against the next-fastest engine we measured.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt | 210 tok/s (1.7×) | 74 tok/s (2.0×) |
| Prefill · 32K prompt | 2,011 tok/s (1.3×) | 363 tok/s (1.2×) |
| Cached time to first token · 32K replay | 123 ms (6.6×) | 282 ms (7.3×) |
| Aggregate decode · 4 concurrent short prompts | 357 tok/s (2.0×) | 170 tok/s (3.9×) |

Splash led on every measure at every prompt length we tested, and the lead
grows with load: 3.8× at four concurrent 32K requests on the 35B. The
[launch post](https://inco.ai/blog/splash/) has the method and the full
comparison against oMLX, Lily, uzu, and Ollama.

## Design

The runtime, scheduler, cache, and API are shared. Everything else is rebuilt
per model:

- **A draft trained for the model.** Speculative decoding is the decode path in
  Splash, not an option. Every model ships with its own [DFlash
  2](https://inco.ai/blog/dflash2/) draft, and one pass of the target verifies a
  block of tokens in parallel.
- **Kernels compiled for exact shapes.** Fused Metal kernels, written and tuned
  by our in-house kernel agents for the model's dimensions, read weights packed
  for them and mapped zero-copy from disk. They ship precompiled: no Xcode, no
  compiler toolchain, nothing tuned on your machine.
- **A memory plan computed for this machine.** Context, KV capacity, and batch
  limits are worked out at startup from the memory Metal recommends, less the
  weights, the draft, and each request's state.

The [launch post](https://inco.ai/blog/splash/) covers the design in depth.

## More

- [DEVELOPMENT.md](DEVELOPMENT.md): building from source, tests, model packages,
  and release packaging.
- Apache-2.0, see [LICENSE](LICENSE). Model weights keep their own licenses.
