# Development

Use Apple Silicon with macOS 26.4+, Xcode 26 or newer with Metal tools, and
Python 3.12–3.14. Packaged users need none of these development tools.

## Build and run

```sh
git clone https://github.com/incoai/splash.git
cd splash
make -j4
./splash serve --model incoai/Qwen3.8-27B-Splash
```

`--model` requires a full Hugging Face `owner/repo` containing a Splash package.
The first serve sets up Python dependencies, resolves a repository commit and
verifies its manifest and artifacts. Later launches reuse the installed snapshot
offline. Public packages need no login; private/gated packages need `HF_TOKEN`
or `hf auth login`. Ctrl+C stops serving; stop before upgrading.

Use `--max-context 100K` or `--max-memory 28G` to set optional limits. Memory
limits cap Metal allocations, not combined process RSS. Agents must already be
installed; `./splash claude|opencode|codex|hermes` connects to the running server.
Arguments pass through, for example `./splash codex resume --last`.

Set `SPLASH_API_KEY` in the server and agent shells to require authentication;
`serve --api-key KEY` overrides the server's environment value. API requests
then require `Authorization: Bearer KEY` (or Anthropic's `x-api-key`). Health
and readiness probes and the chat page remain public; enter the key in the
chat page to send requests. The page does not persist the key. Use
`serve --no-webui` to disable the page. Authentication is off by default.

Source `install/completions/splash.bash` for Bash or
`install/completions/_splash` for Zsh after `compinit`. Completion suggests
commands, bundled official model IDs and installed models without network access.

## Model packages

Packages contain `manifest.json`, packed `target/`, `draft/`, `vision/` weights
and `tokenizer/`. The manifest lists artifact paths, sizes and SHA-256 hashes.
Dense packages use schema 3 / `splash-packed-q4`; MoE uses schema 4 /
`splash-packed-q4-moe`.

These formats encode Qwen3.8-27B and Qwen3.6-35B-A3B layouts. Compatible community
fine-tunes may use any nonempty manifest model name. Native loading validates
geometry, tensor sizes, binary headers, tokenizer and target/draft compatibility.
New architectures require engine support; ordinary HF weights need conversion.

## Code and API boundaries

- `server/`: OpenAI Chat/Responses, Anthropic Messages/count_tokens, templates,
  streaming and input processing. No client-version branches.
- `runtime/engine/`: scheduling, memory admission and reusable request state.
- `runtime/model/`: target/draft execution and vision.
- `runtime/ops/` and `runtime/metal/`: operators and Metal kernels.
- `install/`: launcher, client configuration and model installation.
- `dev/`: maintained tests, benchmarks and build/release tools.

Within `server/`, `server.py` owns HTTP and startup; `frontend.py` prepares
requests and history; `backend.py` owns native request lifecycles. `output.py`
parses generated text for both streaming and complete responses, and
`constraints.py` compiles token constraints. `make architecture-check` prevents
lower layers from importing the HTTP entry module.

Tools can be combined with structured answers. Original schemas validate output
even when generation cannot enforce every assertion. Tool arguments must declare
object properties directly; root references, composition, conditionals,
dependencies, object-wide `enum`/`const`, property-count limits and
`patternProperties` return 400. Hosted search is unsupported;
configure client-owned tools such as MCP. Omitted effort uses the model default.
Hidden thinking signatures use a persistent user key; imported encrypted thinking
preserves visible history without recovering the private reasoning.

`/status.admission` distinguishes memory and concurrency waits, reports suspended
requests, recovery draining and the oldest current wait age. Memory transitions
also appear in the console. Warning pressure can pause growth while `/ready`
remains healthy for work that fits existing allocations.

PDF input supports base64 documents up to 10 MiB / 20 pages, subject to cumulative
rendering budgets. URL inputs, opening passwords and citations are unsupported.
Responses automatic truncation and unsupported history edits return errors.

`POST /tokenize` accepts `{"content":"hello","add_special":false}` and returns
`{"tokens":[...]}` using the loaded tokenizer. Special-token strings are recognized;
`parse_special:false` and `with_pieces:true` are unsupported.
`POST /apply-template` accepts Chat-style `messages`, `tools` and reasoning options,
and returns `{"prompt":"..."}` using the same template as generation.
`add_generation_prompt` defaults to true. Image prompts retain textual placeholders;
raw tokenization does not account for image embeddings (use `count_tokens` for that).
Both endpoints run without inference and share bounded preparation capacity with
`count_tokens`; they can inspect prompts larger than the serving context limit.

Streaming requests accept `"return_progress":true` (default false). Before output,
`prompt_progress` reports `{total, cache, processed, time_ms}`: prompt tokens,
initial cached tokens, completed tokens including cache, and elapsed milliseconds
since prefill admission. Updates follow completed chunks and never regress during
recovery; they are not a time estimate. Chat uses empty-delta chunks, Responses
uses `response.in_progress`, and Messages uses `ping`. Queueing and prompt
preparation do not advance this counter. Non-streaming requests cannot enable it.

`GET /status` returns instance identity and the effective context limit as JSON.
Proxy consumers can use these fields; additional fields may be added:

| Field | Meaning |
| --- | --- |
| `requests.submitted`, `completed`, `cancelled`, `failed` | Native request counters since engine start |
| `memory_actual.current_bytes`, `peak_bytes` | Metal allocations, not process RSS |
| `metrics.decode_tokens_per_second` | Aggregate native decode throughput, not a request's end-to-end rate |
| `maximum_context_tokens` | Declared context limit; available memory may limit admission |

`GET /metrics` exposes the same counters in Prometheus text format. Both endpoints
require the API key when authentication is enabled. Consumers should tolerate
missing native fields while the engine is unavailable, and counter resets after
an engine restart. Chat streams include token usage when the request sets
`"stream_options":{"include_usage":true}`; non-streaming Chat responses always
include usage. A proxy must consume these fields to display statistics.

HTTP bodies require Content-Length, and browser
Origin must match Host. `--allowed-host` permits additional hostnames. Request
logs omit bodies; full crash traces require explicit `SPLASH_CRASH_TRACE=1` and
can contain private conversation data.

Requests sharing a cold prefix can wait for a resident request's planned recovery
point, then enter through the ordinary cache restore path. Waiting requests hold
no active state cell or KV pages and return to ordinary admission when no useful
producer remains. Late arrivals can extend the plan at complete state boundaries.
Higher-priority work does not wait for a lower-priority producer. `/status` exposes
`scheduler.waiting_prefix` separately from resource waits.

Long prefill uses disposable rolling checkpoints every 4096 tokens. Contended
prefill adapts toward a 500 ms slice, keeping 2048-token chunks for long unopposed
work. These policies do not extend client deadlines. Memory recovery waits are
bounded, but readiness does not guarantee that a request-sized allocation fits.

## Validate

```sh
make check
make install test-real test-http-real MODEL=incoai/Qwen3.8-27B-Splash
```

`make check-native-cpu` builds production and runs native CPU tests without a
GPU. `make check-native-metal` requires a supported Metal device; `make check`
includes both. Hosted CI runs CPU checks and sanitizers; the hardware release
gate runs the full suite.

Repeat model tests with `MODEL=incoai/Qwen3.6-35B-A3B-Splash`.
Before release, install all four agents and run `make release-check MODEL=...`
for both models from a clean checkout. It includes correctness, sanitizers,
real HTTP/client behavior and performance checks.

Compare performance on the same idle Mac with the same model and workload.
`make tune-kernels MODEL=...` measures kernel policies. Keep generated reports,
profiles, local paths and experiment notes out of the source tree and commits.

## Package

Release archives contain no Hugging Face credentials and use the official model
list committed with the source. The model-catalog workflow updates that list
from the official collection independently of packaging.
Users accessing private models supply their own `HF_TOKEN` or Hugging Face login.

Release versions are three-part, `x.y.z`, with no `v` prefix: `1.0.0`, then
`1.0.1` for a fix and `1.1.0` for a feature. Use the same version in all three
commands:

```sh
make package RELEASE_VERSION=1.0.0
make package-bottle RELEASE_VERSION=1.0.0
make package-check RELEASE_VERSION=1.0.0
```

The archive, checksum, formula and bottle go to `dist/`; these commands do not
publish. Build bottles on the oldest supported macOS. Bottle/check commands use
a temporary tap and remove their installation; they refuse to replace an existing
Splash installation. The install check requires a poured bottle and runs the
bundled launcher without a compiler or separate Python installation.

To publish: create a GitHub Release on the public mirror `incoai/splash` with
the archive, bottle, checksum files and `SHA256SUMS`, then copy `dist/splash.rb`
over `Formula/splash.rb` in `incoai/homebrew-tap`. Both public repositories
hold exactly one squashed commit of this repository's `main`, authored by Jian
Chen with Zhijian Liu as co-author, and are updated by force-push, never by
pull request. Before re-squashing, `main` must contain no references to the internal
or academic mirrors of the model repositories. Then, on a clean machine:
`brew install incoai/tap/splash && splash --help`, and
`brew audit --strict --online incoai/tap/splash`.

The runtime package allowlists engine, Python, server and launcher files; tests,
benchmarks and developer documents are excluded. User model links and Hermes
sessions survive upgrades; downloads remain in the Hugging Face cache.
